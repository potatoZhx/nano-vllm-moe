"""
draft_decode_eval.py
====================
Proper simulation of speculative-decoding draft-phase expert rerouting.

Why this script exists (and why the previous one is wrong)
----------------------------------------------------------
expert_rerouting_eval.py feeds 256-token chunks through the model with
rerouting active for ALL tokens — this is a prefill-style batch forward.
In the real system, expert substitution ONLY happens during the draft phase:

    Prompt tokens  →  Full model prefill  →  KV cache
                                               │
    Draft token 1: decode (reuse KV cache), substitution active
    Draft token 2: decode (reuse extended KV cache), substitution active
    ...
    Draft token K: decode, substitution active
                                               │
    Verify:        Full model verifies all K draft tokens

This script correctly models that scenario.

Two complementary measurements
-------------------------------
  isolated   Per draft step t, BOTH models receive the same gold KV context
             (built step-by-step with the full model). Measures single-step
             substitution quality with no error accumulation. Upper bound on α.

  sequential Draft model generates its own tokens; each step's output becomes
             the next step's input. Errors accumulate. Measures real-world α.

The gap between isolated and sequential quantifies error accumulation.

Run command
-----------
    python draft_decode_eval.py \
        --model /zx_data1/models/Qwen--Qwen3-30B-A3B-Base/ \
        --data_file /zx_data1/models/datasets/wikitext2_test.txt \
        --cache_ratios 0.5 0.25 \
        --draft_len 8 \
        --prompt_len 128 \
        --n_prompts 32 \
        --outdir ./results_draft

Requires expert_rerouting_eval.py in the same directory.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Import infrastructure from existing eval script
# ─────────────────────────────────────────────────────────────────────────────

_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from expert_rerouting_eval import (
    # Model / data utilities
    load_model_and_tokenizer,
    detect_moe_config,
    get_moe_layers,
    prepare_text_chunks,
    # Calibration
    CalibrationData,
    build_calibration_data,
    # Cache simulation
    SimulatedCache,
    # Algorithm registry
    ALGORITHM_NAMES,
    ALG_GROUPS,
    COLORS,
    MARKERS,
    build_wrappers_for_algorithm,
    # Shared states for stateful algorithms
    BanditState,
    OnlineABState,
    EBBSharedState,
    # Wrapper classes (for isinstance checks during state reset)
    Alg4_ErrorBudget,
    Alg5_OnlineBandit,
    StaticEBB,
    OnlineAB,
    HybridCP,
    # Model patching
    patch_model,
    restore_model,
    # Alpha metric
    compute_alpha_from_logits,
)


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 – KV cache utilities
# ─────────────────────────────────────────────────────────────────────────────

def clone_kv_cache(past_key_values):
    """
    Return a detached clone of a KV cache that is safe to branch from.

    Supports both old-style tuple-of-tuples and new-style DynamicCache
    objects.  The clone is on the same device as the originals.
    """
    if past_key_values is None:
        return None
    # Old format: tuple( tuple(k_tensor, v_tensor), ... ) per layer
    if isinstance(past_key_values, tuple):
        return tuple(
            tuple(t.detach().clone() for t in layer_kv)
            for layer_kv in past_key_values
        )
    # New format: Cache object (transformers ≥ 4.46) — deepcopy is safe here
    # because the tensors inside are still just GPU tensors.
    return copy.deepcopy(past_key_values)


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 – Wrapper state management
# ─────────────────────────────────────────────────────────────────────────────

def reset_between_prompts(wrappers, ebb_shared: Optional[EBBSharedState]):
    """
    Reset all per-prompt accumulators (risk budget, EBB state).
    Bandit Q-tables are NOT reset — they persist across prompts to allow
    the online learner to accumulate experience.
    """
    if ebb_shared is not None:
        ebb_shared.reset()
    for w in wrappers:
        if isinstance(w, Alg4_ErrorBudget):
            w.reset_risk()
        if isinstance(w, (StaticEBB, OnlineAB)):
            if hasattr(w, "shared"):
                w.shared.reset()
            if isinstance(w, OnlineAB) and hasattr(w.ab_state, "_pending"):
                w.ab_state._pending.clear()
        if isinstance(w, HybridCP):
            if hasattr(w, "shared"):
                w.shared.reset()


def reset_between_isolated_steps(wrappers, ebb_shared: Optional[EBBSharedState]):
    """
    For the isolated measurement, each step is independent, so reset
    per-step risk accumulators (but NOT bandit Q-tables).
    """
    if ebb_shared is not None:
        ebb_shared.reset()
    for w in wrappers:
        if isinstance(w, Alg4_ErrorBudget):
            w.reset_risk()
        if isinstance(w, (StaticEBB, OnlineAB)):
            if hasattr(w, "shared"):
                w.shared.reset()


def call_post_forward_checks(wrappers):
    """Call post_forward_check() on all wrappers that have it (EBB, HybridCP)."""
    for w in wrappers:
        if isinstance(w, (StaticEBB, OnlineAB)):
            w.post_forward_check()
        elif isinstance(w, HybridCP):
            pass  # HybridCP calls post_forward_check inside its own forward()


def update_online_learners(wrappers, accepted: bool):
    """Propagate accept/reject signal to bandit-based wrappers."""
    for w in wrappers:
        if isinstance(w, Alg5_OnlineBandit):
            w.bandit_update(accepted)
        if isinstance(w, OnlineAB):
            w.bandit_update(accepted)


def would_terminate(wrappers) -> bool:
    """Return True if any error-budget wrapper signals early termination."""
    for w in wrappers:
        if isinstance(w, (StaticEBB, OnlineAB, Alg4_ErrorBudget)):
            if getattr(w, "should_terminate", False):
                return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 – Core simulation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def build_gold_trajectory(
    model,
    moe_attr: str,
    moe_layer_indices: List[int],
    originals: List,
    prompt_ids: torch.Tensor,   # [1, prompt_len]
    draft_len: int,
    device: str,
) -> Tuple[object, List[torch.Tensor], List[torch.Tensor]]:
    """
    Run the FULL model for one prefill + K decode steps.

    Returns
    -------
    kv_prefill : KV cache after the prompt (starting point for all wrappers)
    base_logits: list of K tensors [1, vocab] — full-model logit at each step
    gold_tokens : list of K tensors [1, 1]   — greedy next token at each step
    """
    prompt_ids = prompt_ids.to(device)
    restore_model(model, moe_attr, originals, moe_layer_indices)

    # Prefill
    pf_out = model(prompt_ids, use_cache=True, output_hidden_states=False)
    kv_prefill = clone_kv_cache(pf_out.past_key_values)

    # K decode steps with the full model
    base_logits: List[torch.Tensor] = []
    gold_tokens: List[torch.Tensor] = []
    kv_cur   = clone_kv_cache(kv_prefill)
    tok_cur  = prompt_ids[:, -1:]        # [1, 1]

    for _ in range(draft_len):
        out = model(tok_cur, past_key_values=kv_cur, use_cache=True)
        logit = out.logits[:, -1, :].detach().float()   # [1, vocab], always float32
        base_logits.append(logit)
        kv_cur = out.past_key_values
        next_tok = logit.argmax(-1, keepdim=True)
        gold_tokens.append(next_tok)
        tok_cur = next_tok

    return kv_prefill, base_logits, gold_tokens


@torch.no_grad()
def simulate_isolated(
    model,
    moe_attr: str,
    moe_layer_indices: List[int],
    originals: List,
    wrappers: List,
    ebb_shared: Optional[EBBSharedState],
    kv_prefill,
    base_logits: List[torch.Tensor],
    gold_tokens: List[torch.Tensor],
    draft_len: int,
    device: str,
) -> Tuple[List[float], List[bool]]:
    """
    Isolated measurement: at each step, BOTH models get the same gold context.
    No substitution errors accumulate.

    Returns
    -------
    iso_alphas     : [draft_len] per-step α
    early_stop_flags: [draft_len] whether the budget wrapper would have stopped
    """
    iso_alphas: List[float] = []
    early_stop_flags: List[bool] = []

    kv_iso  = clone_kv_cache(kv_prefill)
    tok_iso = gold_tokens[0].new_empty(1, 1)   # placeholder; set below
    # The first input is the last prompt token.  We don't have it directly,
    # so we back it out: it's what we fed to the full model to get gold_tokens[0].
    # Use gold_tokens[-1] shifted: actually just use the gold_tokens list offset.
    # Note: at step t, the INPUT is gold_tokens[t-1] (or last prompt tok for t=0).
    # We capture this via the fact that build_gold_trajectory uses prompt[:, -1:]
    # as tok_cur for t=0, then gold_tokens[t-1] for t>0.
    # We replicate that here by maintaining tok_iso = last gold token fed.
    # But we don't have the last prompt token separately, so re-derive from prefill:
    # The last prompt token was already consumed into kv_prefill.
    # At t=0, we feed that token again to get the logit.
    # We need to store it.  Use a sentinel: tok_iso starts as the token we WANT
    # to feed at step 0, which is the same token that started the gold trajectory.
    # Since we don't have it after build_gold_trajectory, we re-derive it:
    # tok_iso at step 0 = the token whose logit == base_logits[0]
    # We don't need the token value for isolation — we just need to extend kv_iso
    # with the gold token AFTER each step.  At step 0, we extend with gold_tokens[0].
    # So tok_iso[0] is whatever was fed to produce base_logits[0].
    # We don't store that explicitly, but we CAN infer it is not needed here:
    # kv_iso starts at prefill state.  The DRAFT model is fed the same tok as
    # the BASE model.  We need to store that token for step 0.
    # Simplest fix: re-derive by noting that the base model was called with
    # the last prompt token at step 0.  We pass that as input.
    # The caller (run_simulation) has access to the original prompt, so let's
    # change the interface to accept `first_input_token`.
    # For now, we'll work around by feeding an arbitrary valid token at step 0
    # and noting that both models get the same input, so the comparison is still valid.
    # ACTUALLY: this won't work because the base_logits were computed with a specific
    # input token.  If we feed a different token, the logit comparison is meaningless.
    #
    # Solution: we need `first_token` passed in.  See simulate_prompt() below which
    # handles this correctly.
    raise NotImplementedError("Call simulate_prompt() instead")


@torch.no_grad()
def simulate_prompt(
    model,
    moe_attr: str,
    moe_layer_indices: List[int],
    originals: List,
    wrappers: List,
    ebb_shared: Optional[EBBSharedState],
    prompt_ids: torch.Tensor,   # [1, prompt_len]
    draft_len: int,
    device: str,
) -> Dict:
    """
    Full simulation for one prompt: gold trajectory + isolated + sequential.

    Returns a dict with keys:
        base_logits       : list[Tensor[1, vocab]] — full model logits, K steps
        iso_alphas        : list[float] — per-step α, gold context (no accumulation)
        seq_alphas        : list[float] — per-step α, draft context (accumulated)
        iso_early_stop    : list[bool]  — would budget wrapper have stopped? (isolated)
        seq_early_stop    : list[bool]  — same for sequential
        seq_tokens        : list[Tensor[1,1]] — tokens generated by draft model
    """
    prompt_ids = prompt_ids.to(device)
    first_tok  = prompt_ids[:, -1:]   # [1, 1] — input to first decode step

    # ── Gold trajectory ───────────────────────────────────────────────────────
    restore_model(model, moe_attr, originals, moe_layer_indices)
    pf_out    = model(prompt_ids, use_cache=True, output_hidden_states=False)
    kv_prefill = clone_kv_cache(pf_out.past_key_values)

    base_logits: List[torch.Tensor] = []
    gold_tokens: List[torch.Tensor] = []
    kv_cur  = clone_kv_cache(kv_prefill)
    tok_cur = first_tok

    for _ in range(draft_len):
        out = model(tok_cur, past_key_values=kv_cur, use_cache=True)
        logit = out.logits[:, -1, :].detach().float()
        base_logits.append(logit)
        kv_cur  = out.past_key_values
        next_tok = logit.argmax(-1, keepdim=True)
        gold_tokens.append(next_tok)
        tok_cur = next_tok

    # ── Isolated measurement ──────────────────────────────────────────────────
    # At each step t:
    #   input  = same token that was fed to base model at step t
    #   kv     = gold KV cache extended up to step t with full model outputs
    # → no error accumulation, measures single-step substitution quality

    iso_alphas: List[float]    = []
    iso_early_stop: List[bool] = []

    kv_iso   = clone_kv_cache(kv_prefill)
    tok_iso  = first_tok    # same starting token as base model

    for t in range(draft_len):
        reset_between_isolated_steps(wrappers, ebb_shared)

        # Draft forward with substitution (same input as base model at step t)
        patch_model(model, moe_attr, wrappers, moe_layer_indices)
        out_d = model(tok_iso, past_key_values=clone_kv_cache(kv_iso), use_cache=True)
        logit_d = out_d.logits[:, -1, :].detach().float()
        call_post_forward_checks(wrappers)
        stopped = would_terminate(wrappers)
        restore_model(model, moe_attr, originals, moe_layer_indices)

        alpha_t = compute_alpha_from_logits(base_logits[t], logit_d)
        iso_alphas.append(alpha_t if alpha_t is not None else float("nan"))
        iso_early_stop.append(stopped)

        # Propagate accept signal to online learners
        if alpha_t is not None:
            update_online_learners(wrappers, alpha_t > 0.5)

        # Advance gold KV cache with gold token using FULL model
        # (so next step's kv_iso has the same gold context as next base_logit)
        out_ext = model(gold_tokens[t], past_key_values=kv_iso, use_cache=True)
        kv_iso  = out_ext.past_key_values
        tok_iso = gold_tokens[t]   # next input is the gold token

    # ── Sequential measurement ────────────────────────────────────────────────
    # At each step t:
    #   input  = token generated by draft model at step t-1
    #   kv     = KV cache extended with draft model's previous outputs
    # → errors accumulate, measures real-world performance

    seq_alphas: List[float]    = []
    seq_early_stop: List[bool] = []
    seq_tokens: List[torch.Tensor] = []

    kv_seq   = clone_kv_cache(kv_prefill)
    tok_seq  = first_tok

    reset_between_prompts(wrappers, ebb_shared)

    for t in range(draft_len):
        # Draft forward
        patch_model(model, moe_attr, wrappers, moe_layer_indices)
        out_d = model(tok_seq, past_key_values=kv_seq, use_cache=True)
        logit_d = out_d.logits[:, -1, :].detach().float()
        call_post_forward_checks(wrappers)
        stopped = would_terminate(wrappers)
        # Keep draft kv_cache for the next step (accumulate error)
        kv_seq_next = out_d.past_key_values
        restore_model(model, moe_attr, originals, moe_layer_indices)

        alpha_t = compute_alpha_from_logits(base_logits[t], logit_d)
        seq_alphas.append(alpha_t if alpha_t is not None else float("nan"))
        seq_early_stop.append(stopped)
        seq_tokens.append(logit_d.argmax(-1, keepdim=True))

        if alpha_t is not None:
            update_online_learners(wrappers, alpha_t > 0.5)

        # Advance with DRAFT token (error accumulates in kv)
        kv_seq  = kv_seq_next
        tok_seq = seq_tokens[-1]

    return {
        "base_logits":     base_logits,
        "iso_alphas":      iso_alphas,
        "seq_alphas":      seq_alphas,
        "iso_early_stop":  iso_early_stop,
        "seq_early_stop":  seq_early_stop,
        "seq_tokens":      seq_tokens,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 – Experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def make_shared_states(
    L: int, N: int, calib: CalibrationData, ebb_budget: float = 0.5
) -> Tuple[BanditState, OnlineABState, EBBSharedState]:
    """Create one set of shared states per (algorithm, cache_ratio) pair."""
    bandit_state = BanditState(
        num_layers=L, num_experts=N,
        sim_prior=calib.similarity_table,
    ) if calib.similarity_table is not None else BanditState(L, N)

    ab_state = OnlineABState(num_layers=L, num_experts=N, calib=calib)

    ebb_shared = EBBSharedState(budget=ebb_budget)

    return bandit_state, ab_state, ebb_shared


def run_all_simulations(
    model,
    moe_cfg: dict,
    calib: CalibrationData,
    prompts: List[torch.Tensor],
    cache_ratios: List[float],
    draft_len: int,
    device: str,
    outdir: str,
    algorithms: Optional[List[str]] = None,
    ebb_budget: float = 0.5,
) -> Dict:
    """
    Main loop: for each (algorithm, cache_ratio), run simulate_prompt() on
    every prompt and aggregate per-step α statistics.

    Result structure:
        results[alg_name][cache_ratio] = {
            "iso_alpha_mean":  [draft_len]   mean α per step (isolated)
            "iso_alpha_std":   [draft_len]
            "seq_alpha_mean":  [draft_len]   mean α per step (sequential)
            "seq_alpha_std":   [draft_len]
            "iso_alpha_total": float          mean across all steps
            "seq_alpha_total": float
            "alpha_decay":     float          seq_alpha[-1] / seq_alpha[0]
            "iso_stop_rate":   float          fraction of steps where budget fires
            "seq_stop_rate":   float
        }
    """
    moe_attr          = moe_cfg["moe_attr"]
    moe_layers        = get_moe_layers(model, moe_attr)
    L                 = len(moe_layers)
    N                 = moe_cfg["num_experts"]
    moe_layer_indices = [gi for gi, _ in moe_layers]

    originals = [
        getattr(model.model.layers[gi], moe_attr) for gi in moe_layer_indices
    ]

    run_names = algorithms if algorithms is not None else ALGORITHM_NAMES
    results: Dict[str, Dict] = {a: {} for a in run_names}

    for cache_ratio in cache_ratios:
        print(f"\n{'='*62}")
        print(f"  Cache ratio: {cache_ratio:.3f}  "
              f"({int(N * cache_ratio)}/{N} experts cached per layer)")
        print(f"{'='*62}")

        cache = SimulatedCache(L, N, cache_ratio,
                               activation_freq=calib.activation_freq)

        for alg_name in run_names:
            print(f"\n  ▶ {alg_name}", flush=True)
            t0 = time.time()

            bandit_state, ab_state, ebb_shared = make_shared_states(
                L, N, calib, ebb_budget)

            wrappers = build_wrappers_for_algorithm(
                alg_name, moe_layers, cache, calib,
                bandit_state = bandit_state if alg_name == "Alg5_Bandit" else None,
                ab_state     = ab_state     if alg_name == "OnlineAB"    else None,
                ebb_shared   = ebb_shared   if alg_name in (
                    "StaticEBB", "OnlineAB", "HybridCP") else None,
            )

            # Per-step accumulators across prompts
            iso_per_step = [[] for _ in range(draft_len)]
            seq_per_step = [[] for _ in range(draft_len)]
            iso_stop_per_step = [[] for _ in range(draft_len)]
            seq_stop_per_step = [[] for _ in range(draft_len)]

            for pi, prompt in enumerate(tqdm(
                    prompts, desc=f"    prompts", leave=False)):
                try:
                    sim = simulate_prompt(
                        model, moe_attr, moe_layer_indices, originals,
                        wrappers, ebb_shared,
                        prompt, draft_len, device,
                    )
                except Exception as ex:
                    print(f"    ⚠ Prompt {pi} failed: {ex}")
                    continue

                for t in range(draft_len):
                    v_iso = sim["iso_alphas"][t]
                    v_seq = sim["seq_alphas"][t]
                    if math.isfinite(v_iso):
                        iso_per_step[t].append(v_iso)
                    if math.isfinite(v_seq):
                        seq_per_step[t].append(v_seq)
                    iso_stop_per_step[t].append(float(sim["iso_early_stop"][t]))
                    seq_stop_per_step[t].append(float(sim["seq_early_stop"][t]))

            # Aggregate
            def safe_mean(lst): return float(np.mean(lst)) if lst else float("nan")
            def safe_std(lst):  return float(np.std(lst))  if lst else float("nan")

            iso_mean = [safe_mean(iso_per_step[t]) for t in range(draft_len)]
            iso_std  = [safe_std(iso_per_step[t])  for t in range(draft_len)]
            seq_mean = [safe_mean(seq_per_step[t]) for t in range(draft_len)]
            seq_std  = [safe_std(seq_per_step[t])  for t in range(draft_len)]
            iso_stop = [safe_mean(iso_stop_per_step[t]) for t in range(draft_len)]
            seq_stop = [safe_mean(seq_stop_per_step[t]) for t in range(draft_len)]

            finite_iso = [v for v in iso_mean if math.isfinite(v)]
            finite_seq = [v for v in seq_mean if math.isfinite(v)]
            iso_total  = safe_mean(finite_iso)
            seq_total  = safe_mean(finite_seq)

            # α decay: how much does sequential α fall from step 0 to step K-1?
            if (math.isfinite(seq_mean[0]) and math.isfinite(seq_mean[-1])
                    and seq_mean[0] > 1e-6):
                alpha_decay = seq_mean[-1] / seq_mean[0]
            else:
                alpha_decay = float("nan")

            results[alg_name][cache_ratio] = {
                "iso_alpha_mean":  iso_mean,
                "iso_alpha_std":   iso_std,
                "seq_alpha_mean":  seq_mean,
                "seq_alpha_std":   seq_std,
                "iso_alpha_total": iso_total,
                "seq_alpha_total": seq_total,
                "alpha_decay":     alpha_decay,
                "iso_stop_rate":   safe_mean([v for s in iso_stop for v in [s]]),
                "seq_stop_rate":   safe_mean([v for s in seq_stop for v in [s]]),
            }

            elapsed = time.time() - t0
            print(f"    iso_α={iso_total:.4f}  seq_α={seq_total:.4f}  "
                  f"decay={alpha_decay:.3f}  ({elapsed:.1f}s)")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 – Reporting and plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_alpha_vs_step(results: Dict, cache_ratios: List[float],
                       draft_len: int, run_names: List[str], outdir: str):
    """
    Per-step α curves for each cache ratio.
    Each plot shows all algorithms, with solid=isolated and dashed=sequential.
    One figure per cache ratio.
    """
    steps = list(range(1, draft_len + 1))

    for ratio in cache_ratios:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        ax_iso, ax_seq = axes

        for alg in run_names:
            if alg not in results or ratio not in results[alg]:
                continue
            d     = results[alg][ratio]
            color = COLORS.get(alg, "gray")
            marker = MARKERS.get(alg, "o")

            # Isolated
            y_iso = d["iso_alpha_mean"]
            e_iso = d["iso_alpha_std"]
            ax_iso.plot(steps, y_iso, label=alg, color=color,
                        marker=marker, linewidth=1.8, markersize=5)
            ax_iso.fill_between(
                steps,
                [max(0, y - e) for y, e in zip(y_iso, e_iso)],
                [min(1, y + e) for y, e in zip(y_iso, e_iso)],
                color=color, alpha=0.12)

            # Sequential
            y_seq = d["seq_alpha_mean"]
            e_seq = d["seq_alpha_std"]
            ax_seq.plot(steps, y_seq, label=alg, color=color,
                        marker=marker, linewidth=1.8, markersize=5,
                        linestyle="--")
            ax_seq.fill_between(
                steps,
                [max(0, y - e) for y, e in zip(y_seq, e_seq)],
                [min(1, y + e) for y, e in zip(y_seq, e_seq)],
                color=color, alpha=0.12)

        for ax, title in [(ax_iso, "Isolated (gold context at each step)"),
                          (ax_seq, "Sequential (error accumulates)")]:
            ax.set_xlabel("Draft Step", fontsize=11)
            ax.set_ylabel("Theoretical Acceptance Rate α", fontsize=11)
            ax.set_title(title, fontsize=11)
            ax.set_xlim(0.8, draft_len + 0.2)
            ax.set_ylim(0, 1)
            ax.axhline(0.7, color="gray", linestyle=":", alpha=0.5)
            ax.set_xticks(steps)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, ncol=2, loc="upper right")

        fig.suptitle(f"Per-step Draft α  [cache_ratio={ratio:.3f}]", fontsize=13)
        plt.tight_layout()
        path = os.path.join(outdir, f"alpha_vs_step_cache{ratio:.3f}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")


def plot_summary_bars(results: Dict, cache_ratios: List[float],
                      run_names: List[str], outdir: str):
    """
    Bar chart comparing iso_alpha_total and seq_alpha_total per algorithm,
    one sub-figure per cache ratio.
    """
    n_alg   = len(run_names)
    n_ratio = len(cache_ratios)
    fig, axes = plt.subplots(1, n_ratio, figsize=(5 * n_ratio + 1, 5), sharey=True)
    if n_ratio == 1:
        axes = [axes]

    x = np.arange(n_alg)
    w = 0.35

    for ax, ratio in zip(axes, cache_ratios):
        iso_vals = []
        seq_vals = []
        colors   = []
        for alg in run_names:
            d = results.get(alg, {}).get(ratio, {})
            iso_vals.append(d.get("iso_alpha_total", float("nan")))
            seq_vals.append(d.get("seq_alpha_total", float("nan")))
            colors.append(COLORS.get(alg, "gray"))

        bars_iso = ax.bar(x - w/2, iso_vals, w, label="Isolated",
                          color=colors, alpha=0.9, edgecolor="k", linewidth=0.5)
        bars_seq = ax.bar(x + w/2, seq_vals, w, label="Sequential",
                          color=colors, alpha=0.45, edgecolor="k", linewidth=0.5,
                          hatch="//")

        ax.set_xticks(x)
        ax.set_xticklabels(run_names, rotation=40, ha="right", fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_title(f"cache_ratio={ratio:.3f}", fontsize=10)
        ax.axhline(0.7, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, axis="y", alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("Mean α (all steps)", fontsize=11)
            ax.legend(fontsize=9)

    fig.suptitle("Mean Acceptance Rate: Isolated vs Sequential", fontsize=13)
    plt.tight_layout()
    path = os.path.join(outdir, "summary_bars.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_alpha_decay(results: Dict, cache_ratios: List[float],
                     run_names: List[str], outdir: str):
    """
    α decay = seq_alpha[step K-1] / seq_alpha[step 0].
    Values < 1 mean α degrades with draft depth (error accumulation).
    """
    n_alg   = len(run_names)
    n_ratio = len(cache_ratios)
    fig, axes = plt.subplots(1, n_ratio, figsize=(5 * n_ratio + 1, 4), sharey=True)
    if n_ratio == 1:
        axes = [axes]

    for ax, ratio in zip(axes, cache_ratios):
        vals   = []
        colors = []
        for alg in run_names:
            d = results.get(alg, {}).get(ratio, {})
            vals.append(d.get("alpha_decay", float("nan")))
            colors.append(COLORS.get(alg, "gray"))

        x = np.arange(n_alg)
        ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="k", linewidth=0.5)
        ax.axhline(1.0, color="green",  linestyle="--", alpha=0.6, label="no decay")
        ax.axhline(0.5, color="orange", linestyle=":",  alpha=0.6, label="50% decay")
        ax.set_xticks(x)
        ax.set_xticklabels(run_names, rotation=40, ha="right", fontsize=7)
        ax.set_title(f"cache_ratio={ratio:.3f}", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("α decay  (seq α[K-1] / α[0])", fontsize=10)
            ax.legend(fontsize=8)

    fig.suptitle("α Decay Across Draft Steps (sequential mode)", fontsize=12)
    plt.tight_layout()
    path = os.path.join(outdir, "alpha_decay.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_iso_vs_seq_scatter(results: Dict, cache_ratios: List[float],
                            run_names: List[str], outdir: str):
    """
    Scatter: x = iso_alpha_total, y = seq_alpha_total.
    Algorithms on the diagonal suffer no accumulation; below diagonal means error accumulates.
    """
    fig, axes = plt.subplots(1, len(cache_ratios),
                             figsize=(5 * len(cache_ratios), 4.5))
    if len(cache_ratios) == 1:
        axes = [axes]

    for ax, ratio in zip(axes, cache_ratios):
        for alg in run_names:
            d = results.get(alg, {}).get(ratio, {})
            x = d.get("iso_alpha_total", float("nan"))
            y = d.get("seq_alpha_total", float("nan"))
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            ax.scatter(x, y, color=COLORS.get(alg, "gray"),
                       marker=MARKERS.get(alg, "o"), s=80, zorder=3)
            ax.annotate(alg, (x, y), fontsize=6,
                        xytext=(3, 3), textcoords="offset points")

        lo, hi = 0.0, 1.0
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.35, linewidth=1)  # diagonal
        ax.set_xlabel("Isolated α (upper bound)", fontsize=10)
        ax.set_ylabel("Sequential α (actual)", fontsize=10)
        ax.set_title(f"cache_ratio={ratio:.3f}", fontsize=10)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Isolated vs Sequential α  (diagonal = no accumulation)",
                 fontsize=12)
    plt.tight_layout()
    path = os.path.join(outdir, "iso_vs_seq_scatter.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def save_csv(results: Dict, cache_ratios: List[float],
             run_names: List[str], draft_len: int, outdir: str):
    path = os.path.join(outdir, "results_draft.csv")
    fieldnames = (
        ["algorithm", "cache_ratio", "step"]
        + ["iso_alpha_mean", "iso_alpha_std", "seq_alpha_mean", "seq_alpha_std"]
        + ["iso_total", "seq_total", "alpha_decay", "iso_stop_rate", "seq_stop_rate"]
    )
    rows = []
    for alg in run_names:
        for ratio in cache_ratios:
            d = results.get(alg, {}).get(ratio, {})
            for t in range(draft_len):
                rows.append({
                    "algorithm":       alg,
                    "cache_ratio":     f"{ratio:.3f}",
                    "step":            t + 1,
                    "iso_alpha_mean":  f"{d.get('iso_alpha_mean', [float('nan')] * draft_len)[t]:.4f}",
                    "iso_alpha_std":   f"{d.get('iso_alpha_std',  [float('nan')] * draft_len)[t]:.4f}",
                    "seq_alpha_mean":  f"{d.get('seq_alpha_mean', [float('nan')] * draft_len)[t]:.4f}",
                    "seq_alpha_std":   f"{d.get('seq_alpha_std',  [float('nan')] * draft_len)[t]:.4f}",
                    "iso_total":       f"{d.get('iso_alpha_total', float('nan')):.4f}",
                    "seq_total":       f"{d.get('seq_alpha_total', float('nan')):.4f}",
                    "alpha_decay":     f"{d.get('alpha_decay', float('nan')):.4f}",
                    "iso_stop_rate":   f"{d.get('iso_stop_rate', float('nan')):.4f}",
                    "seq_stop_rate":   f"{d.get('seq_stop_rate', float('nan')):.4f}",
                })
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {path}")

    # Summary table to stdout
    print("\n  ── Summary (seq_alpha_total) ────────────────────────────────")
    header = f"{'Algorithm':<26}" + "".join(f"  r={r:.3f}" for r in cache_ratios)
    print(header)
    print("-" * len(header))
    for alg in run_names:
        row = f"{alg:<26}"
        for ratio in cache_ratios:
            v = results.get(alg, {}).get(ratio, {}).get("seq_alpha_total", float("nan"))
            row += f"  {v:.4f}"
        print(row)


def save_json(results: Dict, cache_ratios: List[float],
              run_names: List[str], outdir: str):
    serializable = {}
    for alg in run_names:
        serializable[alg] = {}
        for ratio in cache_ratios:
            d = results.get(alg, {}).get(ratio, {})
            serializable[alg][str(ratio)] = {
                k: (v if not isinstance(v, list) else
                    [x if math.isfinite(x) else None for x in v])
                for k, v in d.items()
                if k != "base_logits"  # don't serialize tensors
            }
    path = os.path.join(outdir, "results_draft.json")
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 – Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Draft-decode simulation for expert rerouting evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", required=True,
                        help="Path to Qwen-style MoE model.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--cache_ratios", nargs="+", type=float,
                        default=[0.125, 0.25, 0.5])
    parser.add_argument("--draft_len", type=int, default=8,
                        help="Number of draft tokens to simulate per prompt.")
    parser.add_argument("--prompt_len", type=int, default=128,
                        help="Number of tokens in each prompt (prefill length).")
    parser.add_argument("--n_prompts", type=int, default=32,
                        help="Number of prompts to simulate.")
    parser.add_argument("--n_calib", type=int, default=64,
                        help="Number of chunks for offline calibration.")
    parser.add_argument("--seq_len", type=int, default=256,
                        help="Sequence length for calibration chunks.")
    parser.add_argument("--ebb_budget", type=float, default=0.5,
                        help="Risk budget for StaticEBB / OnlineAB.")
    parser.add_argument("--outdir", default="./results_draft")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data_file", default=None, metavar="PATH",
        help="Local plain-text file for calibration and prompts. "
             "Required when the `datasets` module is not installed.",
    )
    parser.add_argument(
        "--algorithms", nargs="+", default=None, metavar="ALG",
        help=(
            "Algorithms to run. Accepts individual names or group names: "
            "baseline, sim_table, weight_corr, existing, document. "
            "Default: all algorithms."
        ),
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    dtype_map = {"float16": torch.float16,
                 "bfloat16": torch.bfloat16,
                 "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # ── Resolve algorithm list ────────────────────────────────────────────────
    if args.algorithms is None:
        selected_algs = ALGORITHM_NAMES[:]
    else:
        selected_algs = []
        for token in args.algorithms:
            if token in ALG_GROUPS:
                selected_algs.extend(ALG_GROUPS[token])
            elif token in ALGORITHM_NAMES:
                selected_algs.append(token)
            else:
                raise SystemExit(
                    f"Unknown algorithm or group: '{token}'\n"
                    f"Groups: {list(ALG_GROUPS)}\n"
                    f"Algorithms: {ALGORITHM_NAMES}"
                )
        # Deduplicate, preserve order
        seen: set = set()
        selected_algs = [a for a in selected_algs
                         if not (a in seen or seen.add(a))]
    print(f"\nAlgorithms to run ({len(selected_algs)}): {selected_algs}")

    # ── Load model ────────────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(args.model, args.device, dtype)
    moe_cfg = detect_moe_config(model)
    print(f"\nModel config: {moe_cfg}")

    # ── Prepare prompts ───────────────────────────────────────────────────────
    # Use shorter chunks for prompts (prompt_len) and longer for calibration (seq_len)
    print(f"\nPreparing {args.n_prompts} prompts (length {args.prompt_len}) ...")
    prompt_chunks = prepare_text_chunks(
        tokenizer, args.n_prompts, args.prompt_len,
        data_file=args.data_file,
    )
    # Move prompts to device only during simulation to save memory
    prompts = prompt_chunks[:args.n_prompts]

    print(f"\nPreparing {args.n_calib} calibration chunks (length {args.seq_len}) ...")
    calib_chunks = prepare_text_chunks(
        tokenizer, args.n_calib, args.seq_len,
        data_file=args.data_file,
    )

    # ── Calibration ───────────────────────────────────────────────────────────
    print("\n── Phase 1: Offline Calibration ─────────────────────────────────")
    calib = build_calibration_data(
        model, tokenizer,
        moe_cfg["moe_attr"],
        calib_chunks,
        args.device,
        moe_cfg["num_experts"],
    )

    # ── Simulation ────────────────────────────────────────────────────────────
    print("\n── Phase 2: Draft-Decode Simulation ─────────────────────────────")
    print(f"  prompt_len={args.prompt_len}  draft_len={args.draft_len}  "
          f"n_prompts={args.n_prompts}")

    results = run_all_simulations(
        model       = model,
        moe_cfg     = moe_cfg,
        calib       = calib,
        prompts     = prompts,
        cache_ratios = args.cache_ratios,
        draft_len   = args.draft_len,
        device      = args.device,
        outdir      = args.outdir,
        algorithms  = selected_algs,
        ebb_budget  = args.ebb_budget,
    )

    # ── Reporting ─────────────────────────────────────────────────────────────
    print("\n── Phase 3: Reporting ───────────────────────────────────────────")
    plot_alpha_vs_step(results, args.cache_ratios,
                       args.draft_len, selected_algs, args.outdir)
    plot_summary_bars(results, args.cache_ratios,
                      selected_algs, args.outdir)
    plot_alpha_decay(results, args.cache_ratios,
                     selected_algs, args.outdir)
    plot_iso_vs_seq_scatter(results, args.cache_ratios,
                            selected_algs, args.outdir)
    save_csv(results, args.cache_ratios, selected_algs, args.draft_len, args.outdir)
    save_json(results, args.cache_ratios, selected_algs, args.outdir)

    print(f"\n✓ Results saved to {args.outdir}/")


if __name__ == "__main__":
    main()
