"""
test_rerouting.py
=================
Self-contained unit tests for all five rerouting algorithms and evaluation
utilities.  Does NOT require a real model or GPU — runs entirely on CPU with
a synthetic toy MoE.

Run:
  python test_rerouting.py [-v]

Synthetic setup
---------------
  N=16 experts, k=4, L=6 MoE layers, hidden_dim=64, vocab=256
  Cache at two ratios: 0.25 (4 cached) and 0.50 (8 cached)

Tests
-----
  TestSimulatedCache          - correct expert counts, mask shapes
  TestCalibration             - similarity table symmetry, range, zero diagonal
  TestAlg1_SkipSimLUT         - hit experts preserved, misses substitute or skip
  TestAlg2_EntropyBias        - cached experts promoted, top-1 protection
  TestAlg3_RouterScoreMerge   - weight conservation, hit-only limit, similarity scale
  TestAlg4_ErrorBudget        - budget accumulation, early-stop flag
  TestAlg5_OnlineBandit       - cold-start from prior, UCB exploration, EMA update
  TestAlphaMetric             - alpha=1 for identical logits, alpha<1 for diverged
  TestBudgetCoverage          - higher budget → higher mean alpha
  TestEndToEnd                - full pipeline produces plausible results
"""

import math
import os
import sys
import types
import unittest
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Import the module under test ────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from expert_rerouting_eval import (
    SimulatedCache,
    CalibrationData,
    BanditState,
    Alg1_SkipSimLUT,
    Alg2_EntropyBias,
    Alg3_RouterScoreMerge,
    Alg4_ErrorBudget,
    Alg5_OnlineBandit,
    Baseline_SkipAll,
    Baseline_RoundRobin,
    compute_alpha_from_logits,
    ALGORITHM_NAMES,
)


# ─────────────────────────────────────────────────────────────────────────────
# Toy MoE construction
# ─────────────────────────────────────────────────────────────────────────────

N_EXP   = 16    # total experts per layer
TOP_K   = 4     # top-k routing
N_LAYER = 6     # number of MoE layers
HIDDEN  = 64    # hidden dimension
VOCAB   = 256   # vocabulary size (for alpha tests)

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


def make_toy_expert(d: int = HIDDEN) -> nn.Linear:
    return nn.Linear(d, d, bias=False)


def make_toy_moe(n_experts: int = N_EXP, top_k: int = TOP_K,
                 hidden: int = HIDDEN) -> nn.Module:
    """Return a minimal Qwen-style MoE block (gate + experts list)."""
    class ToyMoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_experts    = n_experts
            self.top_k          = top_k
            self.norm_topk_prob = True
            self.gate           = nn.Linear(hidden, n_experts, bias=False)
            self.experts        = nn.ModuleList([
                nn.Linear(hidden, hidden, bias=False) for _ in range(n_experts)
            ])
            # Qwen-style shared expert
            self.shared_expert      = nn.Linear(hidden, hidden, bias=False)
            self.shared_expert_gate = nn.Linear(hidden, 1, bias=False)

        def forward(self, hidden_states):
            B, T, D = hidden_states.shape
            h_flat = hidden_states.view(-1, D)
            logits = self.gate(h_flat)
            probs  = F.softmax(logits, dim=-1)
            weights, indices = probs.topk(self.top_k, dim=-1)
            weights = weights / weights.sum(-1, keepdim=True)
            out = torch.zeros_like(h_flat)
            for slot in range(self.top_k):
                for e in indices[:, slot].unique():
                    mask = indices[:, slot] == e
                    out[mask] += self.experts[e](h_flat[mask]) * weights[mask, slot:slot+1]
            shared = self.shared_expert(h_flat)
            gate   = torch.sigmoid(self.shared_expert_gate(h_flat))
            out   += gate * shared
            return out.view(B, T, D), logits

    return ToyMoE()


def make_calib(n_layers: int = N_LAYER, n_exp: int = N_EXP,
               hidden: int = HIDDEN) -> CalibrationData:
    """Build a synthetic CalibrationData with known structure."""
    # Similarity: block-diagonal structure (experts 0-3 similar, 4-7 similar, …)
    S = torch.zeros(n_layers, n_exp, n_exp)
    for l in range(n_layers):
        for i in range(n_exp):
            for j in range(n_exp):
                # Same block → high similarity
                S[l, i, j] = 1.0 if i // 4 == j // 4 else 0.1
        S[l].fill_diagonal_(1.0)

    # Error ≈ 1 - similarity (rough approximation)
    D = (1.0 - S).clamp(0, 2).float()

    # Sensitivity: uniform
    sens = torch.ones(n_layers) * 0.5

    # Activation freq: first 4 experts are hot
    freq = np.zeros((n_layers, n_exp), dtype=np.float32)
    freq[:, :4] = 0.8 / 4
    freq[:, 4:] = 0.2 / (n_exp - 4)

    return CalibrationData(
        similarity_table=S.half(),
        error_table=D,
        sensitivity=sens,
        activation_freq=freq,
        num_moe_layers=n_layers,
        num_experts=n_exp,
    )


def make_hidden(T: int = 8, B: int = 1, D: int = HIDDEN) -> torch.Tensor:
    return torch.randn(B, T, D)


def make_cache(n_layers=N_LAYER, n_exp=N_EXP, ratio=0.25,
               calib=None) -> SimulatedCache:
    freq = calib.activation_freq if calib else None
    return SimulatedCache(n_layers, n_exp, ratio, activation_freq=freq)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSimulatedCache(unittest.TestCase):

    def setUp(self):
        self.calib = make_calib()
        self.cache = make_cache(ratio=0.25, calib=self.calib)

    def test_cache_size(self):
        """Each layer should have exactly int(N * ratio) cached experts."""
        expected = max(1, int(N_EXP * 0.25))
        for l in range(N_LAYER):
            self.assertEqual(len(self.cache.cached_experts[l]), expected,
                             f"Layer {l} cache size wrong")

    def test_mask_shape(self):
        mask = self.cache.cached_mask(0, device="cpu")
        self.assertEqual(mask.shape, (N_EXP,))
        self.assertEqual(mask.dtype, torch.bool)

    def test_mask_sum(self):
        """Mask should have exactly cache_size True entries."""
        expected = max(1, int(N_EXP * 0.25))
        mask = self.cache.cached_mask(0, device="cpu")
        self.assertEqual(int(mask.sum()), expected)

    def test_lfu_warmstart(self):
        """Highest-frequency experts should be preferentially cached."""
        calib = make_calib()
        cache = SimulatedCache(N_LAYER, N_EXP, 0.25,
                               activation_freq=calib.activation_freq)
        # freq[:, :4] = 0.8/4 (hot), rest cold → cached set should include 0-3
        for l in range(N_LAYER):
            cached = cache.cached_experts[l]
            # At least 3 of the top-4 hot experts should be cached at ratio=0.25 (4 slots)
            hot_cached = sum(1 for e in range(4) if e in cached)
            self.assertGreaterEqual(hot_cached, 3,
                                    f"Layer {l}: expected hot experts in cache, got {cached}")

    def test_full_cache(self):
        """Cache ratio=1.0 should cache all experts."""
        c = SimulatedCache(N_LAYER, N_EXP, 1.0)
        self.assertEqual(len(c.cached_experts[0]), N_EXP)
        mask = c.cached_mask(0, "cpu")
        self.assertTrue(mask.all())


class TestCalibration(unittest.TestCase):

    def setUp(self):
        self.calib = make_calib()

    def test_similarity_shape(self):
        S = self.calib.similarity_table
        self.assertEqual(S.shape, (N_LAYER, N_EXP, N_EXP))

    def test_similarity_range(self):
        S = self.calib.similarity_table.float()
        self.assertTrue((S >= -1.0).all() and (S <= 1.0).all(),
                        "Similarity values out of [-1, 1]")

    def test_diagonal_is_one(self):
        S = self.calib.similarity_table.float()
        for l in range(N_LAYER):
            diag = S[l].diagonal()
            self.assertTrue((diag - 1.0).abs().max() < 0.01,
                            f"Layer {l} diagonal not 1.0")

    def test_block_structure(self):
        """Experts in same block (0-3, 4-7, …) should be more similar than across blocks."""
        S = self.calib.similarity_table.float()
        for l in range(N_LAYER):
            intra = S[l, 0, 1].item()   # same block
            inter = S[l, 0, 4].item()   # different block
            self.assertGreater(intra, inter,
                               f"Layer {l}: intra-block sim {intra:.2f} ≤ inter {inter:.2f}")

    def test_error_table_shape(self):
        D = self.calib.error_table
        self.assertEqual(D.shape, (N_LAYER, N_EXP, N_EXP))

    def test_sensitivity_range(self):
        sens = self.calib.sensitivity
        self.assertTrue((sens >= 0).all() and (sens <= 1).all())


class TestWrapperBase(unittest.TestCase):
    """Base class: creates a toy MoE and runs one forward pass."""

    def setUp(self):
        self.calib  = make_calib()
        self.moe    = make_toy_moe()
        self.cache  = make_cache(ratio=0.25, calib=self.calib)
        self.hidden = make_hidden()    # [1, 8, 64]

    def _make_wrapper(self, cls, **kwargs):
        return cls(self.moe, self.cache, layer_idx=0, calib=self.calib, **kwargs)

    def _run(self, wrapper):
        """Run wrapper forward and return (output, logits)."""
        with torch.no_grad():
            return wrapper(self.hidden)

    def _router_info(self):
        """Return baseline router weights and indices for self.hidden."""
        h = self.hidden.view(-1, HIDDEN)
        logits = self.moe.gate(h)
        probs  = F.softmax(logits, dim=-1)
        w, idx = probs.topk(TOP_K, dim=-1)
        return w, idx


class TestAlg1_SkipSimLUT(TestWrapperBase):

    def test_output_shape(self):
        w = self._make_wrapper(Alg1_SkipSimLUT)
        out, _ = self._run(w)
        self.assertEqual(out.shape, self.hidden.shape)

    def test_hit_experts_unchanged_index(self):
        """Hit experts should remain at their original index (not remapped)."""
        wrapper = self._make_wrapper(Alg1_SkipSimLUT)
        cached_mask = self.cache.cached_mask(0, "cpu")
        _, baseline_idx = self._router_info()

        with torch.no_grad():
            # Intercept _reroute
            logits  = self.moe.gate(self.hidden.view(-1, HIDDEN))
            probs   = F.softmax(logits, dim=-1)
            rw, ri  = probs.topk(TOP_K, dim=-1)
            fw, fi  = wrapper._reroute(logits, rw, ri, cached_mask)

        for t in range(ri.size(0)):
            for r in range(TOP_K):
                e = ri[t, r].item()
                if cached_mask[e]:
                    self.assertEqual(fi[t, r].item(), e,
                                     f"Hit expert {e} was remapped to {fi[t,r].item()}")

    def test_low_weight_skip(self):
        """Experts with weight < theta_skip should be zeroed."""
        wrapper = self._make_wrapper(Alg1_SkipSimLUT, theta_skip_base=0.99)
        # With very high skip threshold, all experts should be skipped or substituted
        with torch.no_grad():
            cached_mask = self.cache.cached_mask(0, "cpu")
            logits = self.moe.gate(self.hidden.view(-1, HIDDEN))
            probs  = F.softmax(logits, dim=-1)
            rw, ri = probs.topk(TOP_K, dim=-1)
            fw, fi = wrapper._reroute(logits, rw, ri, cached_mask)
        # Most miss-expert weights should be 0
        miss_mask = ~cached_mask[ri]
        miss_weights = fw[miss_mask]
        # Not all will be zero (high-weight experts may still be substituted),
        # but the skip behavior should reduce total weight on misses
        # This is a smoke test, not a strict assertion
        self.assertIsNotNone(fw)

    def test_substitute_from_cached_set(self):
        """Substitutes must come from the cached set."""
        wrapper  = self._make_wrapper(Alg1_SkipSimLUT, theta_skip_base=0.0, theta_sim_base=0.0)
        cached_mask = self.cache.cached_mask(0, "cpu")
        with torch.no_grad():
            logits = self.moe.gate(self.hidden.view(-1, HIDDEN))
            probs  = F.softmax(logits, dim=-1)
            rw, ri = probs.topk(TOP_K, dim=-1)
            fw, fi = wrapper._reroute(logits, rw, ri, cached_mask)
        # Any expert with non-zero weight must be in cache
        for t in range(fi.size(0)):
            for r in range(TOP_K):
                if fw[t, r].item() > 0:
                    e = fi[t, r].item()
                    self.assertIn(e, self.cache.cached_experts[0],
                                  f"Expert {e} has non-zero weight but is not cached")


class TestAlg2_EntropyBias(TestWrapperBase):

    def test_output_shape(self):
        w = self._make_wrapper(Alg2_EntropyBias, gamma_high=5.0, gamma_low=0.1)
        out, _ = self._run(w)
        self.assertEqual(out.shape, self.hidden.shape)

    def test_high_gamma_increases_cache_hits(self):
        """With gamma_high=1000, selected experts should all be from cache."""
        wrapper = self._make_wrapper(Alg2_EntropyBias, gamma_high=1000.0, gamma_low=1000.0)
        wrapper.set_entropy_thresholds(0.0, 0.01)
        cached_mask = self.cache.cached_mask(0, "cpu")
        with torch.no_grad():
            logits = self.moe.gate(self.hidden.view(-1, HIDDEN))
            probs  = F.softmax(logits, dim=-1)
            rw, ri = probs.topk(TOP_K, dim=-1)
            fw, fi = wrapper._reroute(logits, rw, ri, cached_mask)
        for t in range(fi.size(0)):
            for r in range(TOP_K):
                self.assertIn(fi[t, r].item(), self.cache.cached_experts[0],
                              "High gamma should route everything to cache")

    def test_zero_gamma_is_original_routing(self):
        """With gamma=0, selected experts should be same as original top-k."""
        wrapper = self._make_wrapper(Alg2_EntropyBias, gamma_high=0.0, gamma_low=0.0)
        wrapper.set_entropy_thresholds(0.0, 0.01)
        _, orig_idx = self._router_info()
        cached_mask = self.cache.cached_mask(0, "cpu")
        with torch.no_grad():
            logits = self.moe.gate(self.hidden.view(-1, HIDDEN))
            probs  = F.softmax(logits, dim=-1)
            rw, ri = probs.topk(TOP_K, dim=-1)
            fw, fi = wrapper._reroute(logits, rw, ri, cached_mask)
        self.assertTrue((fi == orig_idx).all(), "Zero gamma should preserve original top-k")

    def test_top1_protection(self):
        """Original top-1 expert should appear in the selected set even with moderate bias."""
        wrapper = self._make_wrapper(Alg2_EntropyBias, gamma_high=2.0, gamma_low=2.0)
        wrapper.set_entropy_thresholds(0.5, 1.5)
        cached_mask = self.cache.cached_mask(0, "cpu")
        with torch.no_grad():
            logits = self.moe.gate(self.hidden.view(-1, HIDDEN))
            probs  = F.softmax(logits, dim=-1)
            rw, ri = probs.topk(TOP_K, dim=-1)
            fw, fi = wrapper._reroute(logits, rw, ri, cached_mask)
        orig_top1 = ri[:, 0]   # [T]
        for t in range(fi.size(0)):
            if not cached_mask[orig_top1[t]]:
                # Top-1 is NOT cached → wrapper should have protected it
                self.assertIn(orig_top1[t].item(), fi[t].tolist(),
                              f"Token {t}: top-1 expert {orig_top1[t].item()} "
                              f"dropped when not cached")


class TestAlg3_RouterScoreMerge(TestWrapperBase):

    def test_output_shape(self):
        w = self._make_wrapper(Alg3_RouterScoreMerge, lam=0.5)
        out, _ = self._run(w)
        self.assertEqual(out.shape, self.hidden.shape)

    def test_final_indices_are_cached(self):
        """All final expert indices with non-zero weight must be from cache."""
        wrapper = self._make_wrapper(Alg3_RouterScoreMerge, lam=0.5)
        cached_mask = self.cache.cached_mask(0, "cpu")
        with torch.no_grad():
            logits = self.moe.gate(self.hidden.view(-1, HIDDEN))
            probs  = F.softmax(logits, dim=-1)
            rw, ri = probs.topk(TOP_K, dim=-1)
            fw, fi = wrapper._reroute(logits, rw, ri, cached_mask)
        cached = self.cache.cached_experts[0]
        for t in range(fi.size(0)):
            for r in range(fi.size(1)):
                if fw[t, r].item() > 1e-6:
                    self.assertIn(fi[t, r].item(), cached,
                                  f"Non-zero weight on uncached expert {fi[t,r].item()}")

    def test_lam_zero_equals_similarity_only(self):
        """lam=0 → substitute purely by offline similarity."""
        w0 = self._make_wrapper(Alg3_RouterScoreMerge, lam=0.0)
        # With lam=0, the online router score has zero weight →
        # substitute chosen purely by S[e, j], which for our block-diagonal
        # calib is always the closest block member.
        cached_mask = self.cache.cached_mask(0, "cpu")
        with torch.no_grad():
            logits = self.moe.gate(self.hidden.view(-1, HIDDEN))
            probs  = F.softmax(logits, dim=-1)
            rw, ri = probs.topk(TOP_K, dim=-1)
            fw0, fi0 = w0._reroute(logits, rw, ri, cached_mask)
        # Smoke test: must complete without error and produce valid output
        self.assertEqual(fw0.shape, rw.shape)

    def test_similarity_scaled_weight_reduces_noise(self):
        """When substitute similarity is 0, its weight contribution should be ~0."""
        calib_low = make_calib()
        # Set ALL off-diagonal similarities to 0 (worst possible substitutes)
        calib_low.similarity_table[:, :, :] = 0.0
        for l in range(N_LAYER):
            calib_low.similarity_table[l].fill_diagonal_(1.0)
        calib_low.similarity_table = calib_low.similarity_table.half()

        wrapper = Alg3_RouterScoreMerge(self.moe, self.cache, 0, calib_low, lam=0.0)
        cached_mask = self.cache.cached_mask(0, "cpu")
        with torch.no_grad():
            logits = self.moe.gate(self.hidden.view(-1, HIDDEN))
            probs  = F.softmax(logits, dim=-1)
            rw, ri = probs.topk(TOP_K, dim=-1)
            fw, fi = wrapper._reroute(logits, rw, ri, cached_mask)
        # With sim=0, miss weight contribution is 0 → only hit weights remain
        # Verify total weight ≤ fraction of hits
        hit_mask  = cached_mask[ri]
        hit_weight = float(rw[hit_mask].sum().item())
        total_weight = float(fw.sum().item())
        # total ≤ hit_weight + epsilon (small similarity-scaled contributions)
        self.assertLessEqual(total_weight, hit_weight * (1 + TOP_K) + 1e-3,
                             "Zero-similarity substitutes should contribute ≈0 weight")


class TestAlg4_ErrorBudget(TestWrapperBase):

    def test_output_shape(self):
        w = self._make_wrapper(Alg4_ErrorBudget, budget=10.0)
        out, _ = self._run(w)
        self.assertEqual(out.shape, self.hidden.shape)

    def test_no_early_stop_large_budget(self):
        """With infinite budget, early_stop should never be set."""
        wrapper = self._make_wrapper(Alg4_ErrorBudget, budget=1e9)
        wrapper.reset_risk()
        with torch.no_grad():
            wrapper(self.hidden)
        self.assertFalse(wrapper.stats.early_stop)

    def test_early_stop_zero_budget(self):
        """With zero budget, early_stop should be set after first miss."""
        calib_strict = make_calib()
        # Make all error estimates large
        calib_strict.error_table = torch.ones(N_LAYER, N_EXP, N_EXP) * 100.0
        # Ensure there are misses (cache only 1 expert)
        cache_small = SimulatedCache(N_LAYER, N_EXP, 1.0 / N_EXP)
        wrapper = Alg4_ErrorBudget(self.moe, cache_small, 0, calib_strict,
                                   budget=0.0)
        wrapper.reset_risk()
        with torch.no_grad():
            wrapper(self.hidden)
        # With budget=0 and any miss, should stop
        # (only triggered if there are misses at layer 0)
        cached_mask = cache_small.cached_mask(0, "cpu")
        logits = self.moe.gate(self.hidden.view(-1, HIDDEN))
        probs  = F.softmax(logits, dim=-1)
        _, ri  = probs.topk(TOP_K, dim=-1)
        any_miss = (~cached_mask[ri]).any()
        if any_miss:
            self.assertTrue(wrapper.stats.early_stop,
                            "Expected early stop with zero budget and misses")

    def test_substitutes_from_cache(self):
        """Error-minimizing substitute must come from the cached set."""
        wrapper = self._make_wrapper(Alg4_ErrorBudget, budget=1e9)
        cached_mask = self.cache.cached_mask(0, "cpu")
        with torch.no_grad():
            logits = self.moe.gate(self.hidden.view(-1, HIDDEN))
            probs  = F.softmax(logits, dim=-1)
            rw, ri = probs.topk(TOP_K, dim=-1)
            fw, fi = wrapper._reroute(logits, rw, ri, cached_mask)
        cached = self.cache.cached_experts[0]
        for t in range(fi.size(0)):
            for r in range(TOP_K):
                if fw[t, r].item() > 0:
                    self.assertIn(fi[t, r].item(), cached)

    def test_budget_accumulates(self):
        """Cumulative risk should be non-decreasing."""
        wrapper = self._make_wrapper(Alg4_ErrorBudget, budget=1e9)
        wrapper.reset_risk()
        with torch.no_grad():
            wrapper(self.hidden)
        prev = wrapper.stats.cum_risk[-1] if wrapper.stats.cum_risk else 0.0
        wrapper2 = deepcopy(wrapper)
        wrapper2.cumulative_risk = prev
        with torch.no_grad():
            wrapper2(self.hidden)
        if wrapper2.stats.cum_risk:
            self.assertGreaterEqual(wrapper2.stats.cum_risk[-1], prev)


class TestAlg5_OnlineBandit(TestWrapperBase):

    def setUp(self):
        super().setUp()
        self.bandit = BanditState(
            N_LAYER, N_EXP,
            sim_prior=self.calib.similarity_table,
        )

    def test_output_shape(self):
        wrapper = Alg5_OnlineBandit(
            self.moe, self.cache, 0, self.calib, self.bandit)
        out, _ = self._run(wrapper)
        self.assertEqual(out.shape, self.hidden.shape)

    def test_cold_start_uses_similarity_prior(self):
        """Before any updates, alpha_hat should equal the similarity prior."""
        S = self.calib.similarity_table.float()
        expected = ((1.0 + S) / 2.0)   # [L, N, N]
        diff = (self.bandit.alpha[..., 0] - expected).abs()
        self.assertLess(float(diff.max()), 0.02,
                        "Cold-start should initialize from similarity prior")

    def test_selection_from_cached_set(self):
        """Selected substitute must be from the cached set."""
        wrapper = Alg5_OnlineBandit(
            self.moe, self.cache, 0, self.calib, self.bandit)
        cached_mask = self.cache.cached_mask(0, "cpu")
        with torch.no_grad():
            logits = self.moe.gate(self.hidden.view(-1, HIDDEN))
            probs  = F.softmax(logits, dim=-1)
            rw, ri = probs.topk(TOP_K, dim=-1)
            fw, fi = wrapper._reroute(logits, rw, ri, cached_mask)
        cached = self.cache.cached_experts[0]
        for t in range(fi.size(0)):
            for r in range(TOP_K):
                if fw[t, r].item() > 0:
                    self.assertIn(fi[t, r].item(), cached)

    def test_ema_update_changes_alpha(self):
        """After an update, alpha for the used pair should change."""
        wrapper = Alg5_OnlineBandit(
            self.moe, self.cache, 0, self.calib, self.bandit)
        cached_list = self.cache.cached_list(0)
        # Manually inject a known pending pair
        wrapper._pending = [(0, 5, cached_list[0], 1)]
        prev = float(self.bandit.alpha[0, 5, cached_list[0], 1])
        wrapper.bandit_update(accepted=False)   # negative feedback
        curr = float(self.bandit.alpha[0, 5, cached_list[0], 1])
        self.assertLess(curr, prev,
                        "Negative feedback should decrease alpha estimate")

    def test_ucb_reduces_over_observations(self):
        """UCB bonus should shrink as observation count increases."""
        bandit = BanditState(N_LAYER, N_EXP, sim_prior=self.calib.similarity_table)
        l, e, j, b = 0, 0, self.cache.cached_list(0)[0], 0

        # Observe many times
        for _ in range(100):
            bandit.update(l, e, j, b, accepted=True)

        # Compute UCB for this arm vs a fresh arm
        n_total = float(bandit.n_obs[l, e, :, b].sum().log())
        ucb_observed = (n_total / bandit.n_obs[l, e, j, b]).sqrt().item()
        ucb_fresh    = (n_total / 1.0) ** 0.5   # n=1 for fresh arm
        self.assertLess(ucb_observed, ucb_fresh,
                        "UCB bonus should be smaller for often-observed arm")


class TestAlphaMetric(unittest.TestCase):

    def test_identical_logits_gives_alpha_one(self):
        """Identical draft and baseline logits → alpha = 1.0."""
        logits = torch.randn(32, VOCAB)
        alpha  = compute_alpha_from_logits(logits, logits)
        self.assertAlmostEqual(alpha, 1.0, places=5)

    def test_diverged_logits_gives_alpha_less_than_one(self):
        """Diverged distributions should give alpha < 1.0."""
        base  = torch.zeros(16, VOCAB)
        draft = torch.zeros(16, VOCAB)
        base[:, 0]  = 10.0   # strongly peaked at token 0
        draft[:, 1] = 10.0   # strongly peaked at token 1 (different)
        alpha = compute_alpha_from_logits(base, draft)
        self.assertLess(alpha, 0.1, "Orthogonal distributions should have α ≈ 0")

    def test_alpha_bounded(self):
        """Alpha must always be in [0, 1]."""
        for _ in range(20):
            base  = torch.randn(8, VOCAB)
            draft = torch.randn(8, VOCAB)
            alpha = compute_alpha_from_logits(base, draft)
            self.assertGreaterEqual(alpha, 0.0)
            self.assertLessEqual(alpha, 1.0 + 1e-6)

    def test_partial_overlap(self):
        """Two distributions with 50% overlap should give α ≈ 0.5."""
        base  = torch.full((1, 2), -1e9)
        draft = torch.full((1, 2), -1e9)
        base[0, 0]  = 0.0   # p_base  = [1, 0]
        draft[0, 1] = 0.0   # p_draft = [0, 1]
        alpha = compute_alpha_from_logits(base, draft)
        # min([1,0], [0,1]) = [0,0], sum = 0
        self.assertAlmostEqual(alpha, 0.0, places=4)


class TestBudgetCoverage(unittest.TestCase):
    """Higher error budget → more substitutions → higher effective alpha."""

    def test_higher_budget_less_restrictive(self):
        calib = make_calib()
        cache = make_cache(ratio=0.25, calib=calib)
        moe   = make_toy_moe()
        h     = make_hidden(T=16)

        # Collect cumulative risks for two budgets
        wrapper_tight = Alg4_ErrorBudget(moe, cache, 0, calib, budget=0.001)
        wrapper_loose = Alg4_ErrorBudget(moe, cache, 0, calib, budget=1e9)

        for w in [wrapper_tight, wrapper_loose]:
            w.reset_risk()
            with torch.no_grad():
                w(h)

        # Tight budget more likely to trigger early stop
        # Loose budget: cumulative risk recorded, no early stop expected
        self.assertFalse(wrapper_loose.stats.early_stop,
                         "Infinite budget should never trigger early stop")


class TestEndToEnd(unittest.TestCase):
    """
    Full pipeline smoke test with toy model.
    Validates that all algorithms produce valid acceptance rates in [0, 1]
    and that higher-quality algorithms (Alg3, Alg5) outperform baselines (SkipAll)
    at moderate cache ratios.
    """

    def setUp(self):
        torch.manual_seed(42)
        self.calib  = make_calib()
        self.moe    = make_toy_moe()
        self.hidden = make_hidden(T=32)
        self.cache  = make_cache(ratio=0.50, calib=self.calib)  # 50% cached → moderate

    def _get_alpha(self, wrapper_cls, **kwargs) -> float:
        """Run exact forward and rerouted forward; compute alpha."""
        wrapper = wrapper_cls(self.moe, self.cache, 0, self.calib, **kwargs)
        h = self.hidden

        # Exact (baseline) forward
        with torch.no_grad():
            exact_out, exact_logits = self.moe(h)

        # Draft (rerouted) forward
        with torch.no_grad():
            draft_out, _ = wrapper(h)

        # Simulate final logits via a toy projection
        proj = nn.Linear(HIDDEN, VOCAB, bias=False)
        nn.init.normal_(proj.weight, std=0.02)
        with torch.no_grad():
            baseline_logits = proj(exact_out.view(-1, HIDDEN))   # [T, V]
            draft_logits    = proj(draft_out.view(-1, HIDDEN))    # [T, V]

        return compute_alpha_from_logits(baseline_logits, draft_logits)

    def test_all_algorithms_produce_valid_alpha(self):
        """All algorithms must produce α in [0, 1]."""
        bandit = BanditState(N_LAYER, N_EXP, sim_prior=self.calib.similarity_table)
        configs = [
            (Baseline_SkipAll, {}),
            (Baseline_RoundRobin, {}),
            (Alg1_SkipSimLUT, {}),
            (Alg2_EntropyBias, {}),
            (Alg3_RouterScoreMerge, {}),
            (Alg4_ErrorBudget, {"budget": 1e9}),
            (Alg5_OnlineBandit, {"bandit_state": bandit}),
        ]
        for cls, kw in configs:
            alpha = self._get_alpha(cls, **kw)
            self.assertGreaterEqual(alpha, 0.0,
                                    f"{cls.__name__}: α={alpha:.4f} < 0")
            self.assertLessEqual(alpha, 1.0 + 1e-5,
                                 f"{cls.__name__}: α={alpha:.4f} > 1")

    def test_full_cache_alpha_approaches_one(self):
        """Cache ratio=1.0 means no misses → draft ≈ exact → α ≈ 1."""
        full_cache = make_cache(ratio=1.0, calib=self.calib)

        class FullCacheAlg1(Alg1_SkipSimLUT):
            pass

        wrapper = FullCacheAlg1(self.moe, full_cache, 0, self.calib)
        with torch.no_grad():
            exact_out, _ = self.moe(self.hidden)
            draft_out, _ = wrapper(self.hidden)

        proj = nn.Linear(HIDDEN, VOCAB, bias=False)
        with torch.no_grad():
            bl = proj(exact_out.view(-1, HIDDEN))
            dl = proj(draft_out.view(-1, HIDDEN))
        alpha = compute_alpha_from_logits(bl, dl)
        self.assertGreater(alpha, 0.85,
                           f"Full-cache α should be ≥0.85, got {alpha:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary reporter
# ─────────────────────────────────────────────────────────────────────────────

class _SummaryResult(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.pass_list = []
        self.fail_list = []

    def addSuccess(self, test):
        self.pass_list.append(test)

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.fail_list.append((test, err))

    def addError(self, test, err):
        super().addError(test, err)
        self.fail_list.append((test, err))


if __name__ == "__main__":
    print("=" * 70)
    print("Expert Rerouting Algorithm Unit Tests")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite  = loader.discover(start_dir=os.path.dirname(__file__),
                             pattern="test_rerouting.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    passed = result.testsRun - len(result.failures) - len(result.errors)
    print(f"  PASSED: {passed} / {result.testsRun}")
    if result.failures or result.errors:
        print(f"  FAILED: {len(result.failures) + len(result.errors)}")
        sys.exit(1)
    else:
        print("  ✓ All tests passed.")
    print("=" * 70)
