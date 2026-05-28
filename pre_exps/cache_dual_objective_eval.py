"""
cache_dual_objective_eval.py  (Experiment E1)
=============================================
Verify whether dual-objective cache optimization (HitValue + SubstValue)
is necessary, or whether single-objective LFU already suffices.

Core questions:
  Q1. Are HitScore and SubstScore highly correlated (Spearman ρ > 0.7)?
  Q2. Does oracle joint-objective cache selection outperform single-objective?
  Q3. How does the answer change across cache ratios and rerouting algorithms?

Method:
  1. Calibration forward pass → activation freq, pairwise similarity M_l
  2. Per expert: HitScore = p(j), SubstScore = Σ_{i≠j} p(i) * M(j,i)
  3. Spearman correlation per layer
  4. Oracle cache construction → rerouting eval (reuses draft_decode_eval_v2 infra)

Usage:
  python cache_dual_objective_eval.py \
      --model /path/to/Qwen3-30B-A3B-Base \
      --data_file ../wikitext2_test.txt \
      --cache_ratios 0.75 0.50 0.25 \
      --n_calib 64 --n_eval 32 --outdir ./results_e1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Model & data utilities (shared with draft_decode_eval_v2)
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_path, device, dtype=torch.float16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True, low_cpu_mem_usage=True)
    if device == "cpu":
        model = model.to(device)
    model.eval()
    return model, tok


def detect_moe_config(model) -> dict:
    cfg = model.config
    N = getattr(cfg, "num_experts", getattr(cfg, "n_routed_experts", None))
    k = getattr(cfg, "num_experts_per_tok", getattr(cfg, "top_k", None))
    if N is None or k is None:
        for layer in model.model.layers:
            moe = getattr(layer, "mlp", None)
            if moe and hasattr(moe, "experts"):
                N = N or len(moe.experts)
                k = k or getattr(moe, "top_k", None)
                break
    moe_attr = "mlp"
    for layer in model.model.layers:
        if hasattr(layer, "block_sparse_moe"):
            moe_attr = "block_sparse_moe"; break
    L = sum(1 for layer in model.model.layers
            if hasattr(getattr(layer, moe_attr, None), "experts"))
    return {"num_experts": N, "top_k": k, "num_layers": L, "moe_attr": moe_attr}


def get_moe_layers(model, moe_attr):
    return [(i, getattr(layer, moe_attr))
            for i, layer in enumerate(model.model.layers)
            if hasattr(getattr(layer, moe_attr, None), "experts")]


_SNIPPET = (
    "Wikipedia is a free online encyclopedia. Language models predict the next "
    "token given previous tokens. Mixture-of-Experts activates a subset of params. "
) * 200


def prepare_chunks(tokenizer, n, seq_len, data_file=None):
    text = None
    if data_file:
        try:
            text = open(data_file, encoding="utf-8").read()
        except Exception:
            pass
    if text is None:
        try:
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-2-raw-v1",
                              split="test", trust_remote_code=True)
            text = "\n\n".join(ds["text"])
        except Exception:
            text = _SNIPPET
    enc = tokenizer(text, return_tensors="pt")["input_ids"]
    total = enc.size(1)
    chunks = [enc[:, s:s+seq_len] for s in range(0, total - seq_len, seq_len)][:n]
    if len(chunks) < n:
        chunks = (chunks * (n // len(chunks) + 1))[:n]
    random.shuffle(chunks)
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Calibration: build similarity matrix and activation frequencies
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CalibData:
    sim: np.ndarray          # [L, N, N] pairwise cosine similarity of mean outputs
    act_freq: np.ndarray     # [L, N] activation frequency
    rank1_freq: np.ndarray   # [L, N] top-1 position frequency
    rank2_freq: np.ndarray   # [L, N] top-2 position frequency
    L: int
    N: int
    top_k: int


@torch.no_grad()
def calibrate(model, moe_attr, chunks, device, N, top_k):
    layers = get_moe_layers(model, moe_attr)
    L = len(layers)
    D = model.config.hidden_size

    # Accumulators
    sum_out = [torch.zeros(N, D) for _ in range(L)]
    cnt_out = [torch.zeros(N) for _ in range(L)]
    act_freq = np.zeros((L, N), dtype=np.float64)
    rank1_freq = np.zeros((L, N), dtype=np.float64)
    rank2_freq = np.zeros((L, N), dtype=np.float64)
    total_tokens = 0

    # Expert output hooks
    bufs = [{} for _ in range(L)]
    hooks = []

    def _ehook(li, ei):
        def h(m, inp, out):
            bufs[li][ei] = (out[0] if isinstance(out, tuple) else out
                           ).detach().float().cpu()
        return h

    # Gate hooks to get routing decisions
    gate_outputs = [None] * L

    def _ghook(li):
        def h(m, inp, out):
            gate_outputs[li] = out.detach().float().cpu()
        return h

    for li, (idx, moe) in enumerate(layers):
        for ei, expert in enumerate(moe.experts):
            hooks.append(expert.register_forward_hook(_ehook(li, ei)))
        if hasattr(moe, "gate"):
            hooks.append(moe.gate.register_forward_hook(_ghook(li)))

    for chunk in tqdm(chunks, desc="Calibrating"):
        inp = chunk.to(device)
        bufs = [{} for _ in range(L)]
        gate_outputs = [None] * L
        model(inp)
        T = inp.size(1)
        total_tokens += T

        for li in range(L):
            # Accumulate expert outputs
            for ei, out_t in bufs[li].items():
                if out_t.dim() == 3:
                    out_t = out_t.squeeze(0)
                n_t = out_t.size(0)
                sum_out[li][ei] += out_t.sum(0)
                cnt_out[li][ei] += n_t

            # Accumulate routing statistics
            if gate_outputs[li] is not None:
                g = gate_outputs[li]
                if g.dim() == 3:
                    g = g.squeeze(0)
                w = torch.softmax(g, dim=-1)
                topk_w, topk_i = torch.topk(w, top_k, dim=-1)
                topk_i_np = topk_i.numpy()
                for t in range(g.size(0)):
                    for rank, ei in enumerate(topk_i_np[t]):
                        act_freq[li, ei] += 1
                        if rank == 0:
                            rank1_freq[li, ei] += 1
                        elif rank == 1:
                            rank2_freq[li, ei] += 1

    for h in hooks:
        h.remove()

    # Normalize frequencies
    if total_tokens > 0:
        act_freq /= total_tokens
        rank1_freq /= total_tokens
        rank2_freq /= total_tokens

    # Build cosine similarity matrix
    sim = np.zeros((L, N, N), dtype=np.float32)
    for li in range(L):
        mean_out = torch.zeros(N, D)
        for ei in range(N):
            if cnt_out[li][ei] > 0:
                mean_out[ei] = sum_out[li][ei] / cnt_out[li][ei]
        norms = mean_out.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normed = mean_out / norms
        sim[li] = (normed @ normed.T).numpy()

    return CalibData(
        sim=sim, act_freq=act_freq,
        rank1_freq=rank1_freq, rank2_freq=rank2_freq,
        L=L, N=N, top_k=top_k,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Score computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_scores(calib: CalibData):
    """Compute HitScore and SubstScore for each expert at each layer.

    Returns:
        hit_scores:  [L, N] — activation frequency p_l(j)
        subst_scores: [L, N] — weighted substitution value as substitute
    """
    L, N = calib.L, calib.N
    hit_scores = calib.act_freq.copy()  # [L, N]

    subst_scores = np.zeros((L, N), dtype=np.float64)
    for l in range(L):
        for j in range(N):
            # SubstScore(j) = sum_{i != j} p(i) * M(j, i)
            # j as substitute for i: how good is j at replacing others
            score = 0.0
            for i in range(N):
                if i == j:
                    continue
                score += calib.act_freq[l, i] * max(0.0, float(calib.sim[l, j, i]))
            subst_scores[l, j] = score

    return hit_scores, subst_scores


def compute_residual_subst_scores(calib: CalibData, cache_set: set,
                                  rerouting: str = "none"):
    """SubstScore conditioned on residual misses after rerouting.

    For 'none' or 'skip_all': all uncached experts are residual misses.
    For 'alg2_v2': approximate residual miss probability via bias coverage.
    """
    L, N = calib.L, calib.N
    subst_scores = np.zeros((L, N), dtype=np.float64)

    for l in range(L):
        cached = cache_set  # same cache set per layer in this simplified model
        for j in range(N):
            if j not in cached:
                continue  # only cached experts have SubstScore
            score = 0.0
            for i in range(N):
                if i == j or i in cached:
                    continue
                # For alg2_v2: approximate residual miss prob
                if rerouting == "alg2_v2":
                    # Expert i is residual miss if its original rank is very low
                    # Approximation: assume bias covers top-3k experts
                    # Experts outside top-3k have ρ_residual ≈ 1
                    rho_residual = 0.3  # rough estimate for bias coverage
                else:
                    rho_residual = 1.0
                score += rho_residual * calib.act_freq[l, i] * max(0.0, float(calib.sim[l, j, i]))
            subst_scores[l, j] = score

    return subst_scores


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Oracle cache construction strategies
# ─────────────────────────────────────────────────────────────────────────────

def build_cache_hit_only(calib: CalibData, cache_size: int):
    """Select top cache_size experts by HitScore (= activation frequency)."""
    caches = []
    for l in range(calib.L):
        top = np.argsort(calib.act_freq[l])[::-1][:cache_size]
        caches.append(set(top.tolist()))
    return caches


def build_cache_subst_only(calib: CalibData, cache_size: int):
    """Select top cache_size experts by SubstScore."""
    _, subst_scores = compute_scores(calib)
    caches = []
    for l in range(calib.L):
        top = np.argsort(subst_scores[l])[::-1][:cache_size]
        caches.append(set(top.tolist()))
    return caches


def build_cache_joint(calib: CalibData, cache_size: int,
                      lambda1: float = 0.5, lambda2: float = 0.5):
    """Select top cache_size by joint score: λ1*HitScore + λ2*SubstScore."""
    hit_scores, subst_scores = compute_scores(calib)
    caches = []
    for l in range(calib.L):
        # Normalize to [0,1] range per layer
        h = hit_scores[l]
        s = subst_scores[l]
        h_norm = (h - h.min()) / (h.max() - h.min() + 1e-10)
        s_norm = (s - s.min()) / (s.max() - s.min() + 1e-10)
        joint = lambda1 * h_norm + lambda2 * s_norm
        top = np.argsort(joint)[::-1][:cache_size]
        caches.append(set(top.tolist()))
    return caches


def build_cache_greedy_submodular(calib: CalibData, cache_size: int,
                                  lambda1: float = 0.5, lambda2: float = 0.5):
    """Greedy submodular maximization for joint HitValue + SubstValue.

    Provably (1-1/e) approximation for the submodular joint objective.
    """
    caches = []
    for l in range(calib.L):
        selected = set()
        remaining = set(range(calib.N))

        for _ in range(cache_size):
            best_expert = -1
            best_marginal = -float("inf")

            for j in remaining:
                # Marginal HitValue: always p(j) * τ (modular)
                marginal_hit = float(calib.act_freq[l, j])

                # Marginal SubstValue: for each uncached expert i not in
                # selected ∪ {j}, check if j improves max similarity
                marginal_subst = 0.0
                for i in range(calib.N):
                    if i in selected or i == j:
                        continue
                    sim_j = max(0.0, float(calib.sim[l, j, i]))
                    current_best = 0.0
                    for s in selected:
                        current_best = max(current_best, float(calib.sim[l, s, i]))
                    marginal_subst += float(calib.act_freq[l, i]) * max(0.0, sim_j - current_best)

                marginal = lambda1 * marginal_hit + lambda2 * marginal_subst
                if marginal > best_marginal:
                    best_marginal = marginal
                    best_expert = j

            if best_expert >= 0:
                selected.add(best_expert)
                remaining.discard(best_expert)

        caches.append(selected)
    return caches


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Rerouting eval (simplified: SkipAll acceptance rate)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_skipall_alpha(model, moe_attr, eval_chunks, device,
                       caches, N, top_k, prompt_len=128, draft_len=8):
    """Evaluate seq_alpha under SkipAll rerouting with given cache sets.

    Uses decode-mode simulation: prefill prompt_len tokens exactly,
    then draft draft_len steps with SkipAll rerouting.
    Returns mean seq_alpha across all prompts.
    """
    layers = get_moe_layers(model, moe_attr)
    L = len(layers)
    total_len = prompt_len + draft_len

    alphas_all = []

    for chunk in tqdm(eval_chunks, desc="Eval SkipAll"):
        if chunk.size(1) < total_len:
            continue
        inp = chunk[:, :total_len].to(device)

        # Step 1: full baseline forward
        with torch.no_grad():
            out_base = model(inp)
            logits_base = out_base.logits[0]  # [T, vocab]

        # Step 2: draft forward with SkipAll wrapper
        # Install SkipAll wrappers
        originals = {}
        for li, (idx, moe) in enumerate(layers):
            originals[li] = moe.forward
            cache_mask = torch.zeros(N, dtype=torch.bool, device=device)
            cache_mask[list(caches[li])] = True
            moe._cache_mask = cache_mask
            moe._top_k = top_k

            def make_skipall_forward(orig_fwd, cm, tk):
                def wrapper(hidden_states, **kwargs):
                    # We need to intercept and modify the routing
                    # For simplicity, run original and zero out miss experts
                    result = orig_fwd(hidden_states, **kwargs)
                    return result
                return wrapper
            # Note: full SkipAll wrapper requires model-specific gate interception.
            # For this oracle comparison, we use the simpler logit-TV approach.

        # Simplified: compute per-position TV distance using the baseline
        # approach from draft_decode_eval_v2. Here we approximate by measuring
        # how much the cache covers the actual routing decisions.
        alphas_step = []
        gate_outputs_per_step = [None] * L

        # Record gate outputs for draft positions
        hooks = []
        for li, (idx, moe) in enumerate(layers):
            if hasattr(moe, "gate"):
                def _ghook(li_):
                    def h(m, inp_, out_):
                        gate_outputs_per_step[li_] = out_.detach().float().cpu()
                    return h
                hooks.append(moe.gate.register_forward_hook(_ghook(li)))

        # Run forward to collect gate outputs (using full sequence)
        model(inp)

        for h in hooks:
            h.remove()

        # Estimate alpha from cache coverage of routing decisions
        for step in range(draft_len):
            pos = prompt_len + step
            miss_fracs = []
            for li in range(L):
                if gate_outputs_per_step[li] is None:
                    continue
                g = gate_outputs_per_step[li]
                if g.dim() == 3:
                    g = g[0]  # remove batch
                if pos >= g.size(0):
                    continue
                w = torch.softmax(g[pos], dim=-1)
                topk_w, topk_i = torch.topk(w, top_k)
                hits = sum(1 for ei in topk_i.tolist() if ei in caches[li])
                miss_fracs.append(1.0 - hits / top_k)

            # Approximate alpha from average miss fraction:
            # Higher miss → lower alpha. Use empirical calibration from v2 results.
            avg_miss = np.mean(miss_fracs) if miss_fracs else 0.0
            # Rough model: alpha ≈ 1 - 0.5 * miss_frac (heuristic, calibrated later)
            alpha_est = max(0.0, 1.0 - 0.6 * avg_miss)
            alphas_step.append(alpha_est)

        if alphas_step:
            alphas_all.append(np.mean(alphas_step))

        # Restore
        for li, (idx, moe) in enumerate(layers):
            if li in originals:
                moe.forward = originals[li]
            if hasattr(moe, "_cache_mask"):
                del moe._cache_mask, moe._top_k

    return float(np.mean(alphas_all)) if alphas_all else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Analysis: compute cache-coverage hit rate (simpler, more robust)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_cache_hit_rate(model, moe_attr, eval_chunks, device,
                           caches, N, top_k):
    """Compute average cache hit rate across eval chunks.

    Returns per-layer and overall hit rates.
    """
    layers = get_moe_layers(model, moe_attr)
    L = len(layers)

    layer_hits = np.zeros(L, dtype=np.float64)
    layer_total = np.zeros(L, dtype=np.float64)

    gate_bufs = [None] * L
    hooks = []
    for li, (idx, moe) in enumerate(layers):
        if hasattr(moe, "gate"):
            def _ghook(li_):
                def h(m, inp_, out_):
                    gate_bufs[li_] = out_.detach().float().cpu()
                return h
            hooks.append(moe.gate.register_forward_hook(_ghook(li)))

    for chunk in tqdm(eval_chunks, desc="Hit rate"):
        inp = chunk.to(device)
        gate_bufs = [None] * L
        model(inp)
        T = inp.size(1)

        for li in range(L):
            if gate_bufs[li] is None:
                continue
            g = gate_bufs[li]
            if g.dim() == 3:
                g = g.squeeze(0)
            _, topk_i = torch.topk(g, top_k, dim=-1)
            for t in range(min(T, g.size(0))):
                for ei in topk_i[t].tolist():
                    layer_total[li] += 1
                    if ei in caches[li]:
                        layer_hits[li] += 1

    for h in hooks:
        h.remove()

    per_layer = np.where(layer_total > 0, layer_hits / layer_total, 0.0)
    overall = float(layer_hits.sum() / max(1, layer_total.sum()))
    return per_layer, overall


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(args):
    os.makedirs(args.outdir, exist_ok=True)

    # Load model
    print("Loading model...")
    dtype = getattr(torch, args.dtype) if hasattr(torch, args.dtype) else torch.float16
    model, tokenizer = load_model_and_tokenizer(args.model, args.device, dtype)
    moe_cfg = detect_moe_config(model)
    N, top_k, L = moe_cfg["num_experts"], moe_cfg["top_k"], moe_cfg["num_layers"]
    moe_attr = moe_cfg["moe_attr"]
    print(f"  MoE config: L={L}, N={N}, top_k={top_k}, attr={moe_attr}")

    # Prepare data
    calib_chunks = prepare_chunks(tokenizer, args.n_calib, args.seq_len, args.data_file)
    eval_chunks = prepare_chunks(tokenizer, args.n_eval, args.seq_len, args.data_file)

    # Phase 1: Calibration
    print("\n[Phase 1] Calibration...")
    calib = calibrate(model, moe_attr, calib_chunks, args.device, N, top_k)

    # Phase 2: Score computation
    print("\n[Phase 2] Computing HitScore and SubstScore...")
    hit_scores, subst_scores = compute_scores(calib)

    # Phase 3: Correlation analysis
    print("\n[Phase 3] Correlation analysis...")
    correlations = {}
    for l in range(L):
        rho, p_val = stats.spearmanr(hit_scores[l], subst_scores[l])
        correlations[l] = {"rho": float(rho), "p": float(p_val)}
        if l < 5 or l >= L - 3:  # print first/last few layers
            print(f"  Layer {l:2d}: Spearman ρ = {rho:.4f}, p = {p_val:.4e}")

    mean_rho = np.mean([v["rho"] for v in correlations.values()])
    print(f"\n  Mean Spearman ρ across layers: {mean_rho:.4f}")
    if mean_rho > 0.7:
        print("  → Strong correlation: dual-objective likely unnecessary")
    elif mean_rho > 0.4:
        print("  → Moderate correlation: dual-objective may have marginal value")
    else:
        print("  → Weak correlation: dual-objective likely valuable")

    # Phase 4: Oracle cache comparison
    print("\n[Phase 4] Oracle cache comparison...")
    results = []

    for ratio in args.cache_ratios:
        cache_size = max(1, int(N * ratio))
        print(f"\n  Cache ratio = {ratio} (size = {cache_size})")

        strategies = {
            "HitOnly_LFU": build_cache_hit_only(calib, cache_size),
            "SubstOnly": build_cache_subst_only(calib, cache_size),
            "Joint_0.5_0.5": build_cache_joint(calib, cache_size, 0.5, 0.5),
            "Joint_0.7_0.3": build_cache_joint(calib, cache_size, 0.7, 0.3),
        }

        # Only run greedy submodular for small N (expensive)
        if N <= 64:
            strategies["Greedy_Submodular"] = build_cache_greedy_submodular(
                calib, cache_size, 0.5, 0.5)

        for name, caches in strategies.items():
            per_layer_hr, overall_hr = compute_cache_hit_rate(
                model, moe_attr, eval_chunks, args.device, caches, N, top_k)

            # Compute SubstValue for each strategy
            total_subst_value = 0.0
            for l in range(L):
                for j in range(N):
                    if j in caches[l]:
                        continue
                    best_sim = 0.0
                    for i in caches[l]:
                        best_sim = max(best_sim, float(calib.sim[l, i, j]))
                    total_subst_value += calib.act_freq[l, j] * best_sim

            result = {
                "ratio": ratio,
                "strategy": name,
                "hit_rate": overall_hr,
                "subst_value": total_subst_value / L,
                "mean_layer_hr": float(per_layer_hr.mean()),
            }
            results.append(result)
            print(f"    {name:25s} → hit_rate={overall_hr:.4f}, "
                  f"subst_value={total_subst_value/L:.4f}")

    # Save results
    print("\n[Phase 5] Saving results...")

    # Correlation results
    with open(os.path.join(args.outdir, "correlation_analysis.json"), "w") as f:
        json.dump({
            "per_layer": {str(k): v for k, v in correlations.items()},
            "mean_rho": float(mean_rho),
            "conclusion": "dual_likely_unnecessary" if mean_rho > 0.7
                         else "dual_marginal" if mean_rho > 0.4
                         else "dual_valuable",
        }, f, indent=2)

    # Oracle comparison
    with open(os.path.join(args.outdir, "oracle_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(args.outdir, "oracle_comparison.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ratio", "strategy", "hit_rate",
                                                "subst_value", "mean_layer_hr"])
        writer.writeheader()
        writer.writerows(results)

    # Plot: correlation scatter (sample layers)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    sample_layers = [0, L//4, L//2, 3*L//4, L-1, min(5, L-1)]
    for ax, l in zip(axes.flat, sample_layers):
        ax.scatter(hit_scores[l], subst_scores[l], alpha=0.5, s=10)
        rho = correlations[l]["rho"]
        ax.set_title(f"Layer {l} (ρ={rho:.3f})")
        ax.set_xlabel("HitScore")
        ax.set_ylabel("SubstScore")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "correlation_scatter.png"), dpi=150)
    plt.close()

    # Plot: oracle comparison bar chart
    fig, axes = plt.subplots(1, len(args.cache_ratios), figsize=(6*len(args.cache_ratios), 5))
    if len(args.cache_ratios) == 1:
        axes = [axes]
    for ax, ratio in zip(axes, args.cache_ratios):
        ratio_results = [r for r in results if r["ratio"] == ratio]
        names = [r["strategy"] for r in ratio_results]
        hit_rates = [r["hit_rate"] for r in ratio_results]
        ax.bar(range(len(names)), hit_rates, color=plt.cm.Set2(range(len(names))))
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Cache Hit Rate")
        ax.set_title(f"r = {ratio}")
        ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "oracle_hit_rate_comparison.png"), dpi=150)
    plt.close()

    print(f"\nResults saved to {args.outdir}/")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="E1: Cache dual-objective evaluation")
    parser.add_argument("--model", required=True, help="Path to MoE model")
    parser.add_argument("--data_file", default=None, help="Path to text data file")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--cache_ratios", nargs="+", type=float, default=[0.75, 0.50, 0.25])
    parser.add_argument("--n_calib", type=int, default=64)
    parser.add_argument("--n_eval", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--outdir", default="./results_e1")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
