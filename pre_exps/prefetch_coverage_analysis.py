"""
prefetch_coverage_analysis.py  (Experiment E4)
===============================================
Quantify the relationship between prefetch coverage and verify stall,
and validate PrefetchCoverage as a dynamic K signal.

Core questions:
  Q1. How does prefetch coverage scale with K and prefetch rate m?
  Q2. At what coverage threshold does verify stall become negligible?
  Q3. Is PrefetchCoverage a useful auxiliary signal for dynamic K?
  Q4. What is the interaction between coverage, K, and throughput?

Method:
  1. Decode-mode simulation: collect original routing per step (= verify need)
  2. Simulate different prefetch rates m = 0, 1, 2, 3, 4 experts/step
  3. At each step k: PrefetchCoverage(k) = |P_ready| / |P_need(K)|
  4. Estimate verify stall from uncovered experts
  5. Compute theoretical throughput for (K, m) combinations
  6. Validate PrefetchCoverage as a signal vs actual verify stall time

Standalone — no dependency on nano-vllm-moe runtime.

Usage:
  python prefetch_coverage_analysis.py \
      --model /path/to/Qwen3-30B-A3B-Base \
      --data_file ../wikitext2_test.txt \
      --cache_ratios 0.75 0.50 0.25 \
      --k_max 12 --prompt_len 128 \
      --n_calib 8 --n_eval 16 --outdir ./results_e4
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
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Model & data utilities (shared)
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
# Section 2 — SimulatedCache
# ─────────────────────────────────────────────────────────────────────────────

class SimulatedCache:
    def __init__(self, L: int, N: int, ratio: float,
                 act_freq: Optional[np.ndarray] = None):
        self.N = N
        self.cache_size = max(1, int(N * ratio))
        self.sets: List[set] = []
        for l in range(L):
            if act_freq is not None:
                top = np.argsort(act_freq[l])[::-1][:self.cache_size]
                self.sets.append(set(top.tolist()))
            else:
                self.sets.append(set(random.sample(range(N), self.cache_size)))

    def is_cached(self, layer: int, expert_id: int) -> bool:
        return expert_id in self.sets[layer]


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Calibration (lightweight)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def calibrate_freq(model, moe_attr, chunks, device, N, top_k):
    layers = get_moe_layers(model, moe_attr)
    L = len(layers)
    act_freq = np.zeros((L, N), dtype=np.float64)
    total_tokens = 0

    gate_outputs = [None] * L
    hooks = []
    for li, (idx, moe) in enumerate(layers):
        if hasattr(moe, "gate"):
            def _ghook(li_):
                def h(m, inp, out):
                    gate_outputs[li_] = out.detach().float().cpu()
                return h
            hooks.append(moe.gate.register_forward_hook(_ghook(li)))

    for chunk in tqdm(chunks, desc="Calibrating"):
        inp = chunk.to(device)
        gate_outputs = [None] * L
        model(inp)
        T = inp.size(1)
        total_tokens += T
        for li in range(L):
            if gate_outputs[li] is None:
                continue
            g = gate_outputs[li]
            if g.dim() == 3:
                g = g.squeeze(0)
            _, topk_i = torch.topk(g, top_k, dim=-1)
            for t in range(g.size(0)):
                for ei in topk_i[t].tolist():
                    act_freq[li, ei] += 1

    for h in hooks:
        h.remove()
    if total_tokens > 0:
        act_freq /= total_tokens
    return act_freq


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Collect original routing decisions (= verify needs)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RoutingTrace:
    """Per-sequence routing decisions for draft positions."""
    # routing[step] = {layer_idx: set_of_expert_ids}
    routing: List[Dict[int, Set[int]]]


@torch.no_grad()
def collect_routing_traces(
    model, moe_attr, prompts, device,
    N, top_k, L,
    prompt_len, k_max,
) -> List[RoutingTrace]:
    """Run full model (no rerouting) and record routing at draft positions."""
    layers = get_moe_layers(model, moe_attr)
    total_len = prompt_len + k_max

    gate_bufs = [None] * L
    hooks = []
    for li, (idx, moe) in enumerate(layers):
        if hasattr(moe, "gate"):
            def _ghook(li_):
                def h(m, inp, out):
                    gate_bufs[li_] = out.detach().float().cpu()
                return h
            hooks.append(moe.gate.register_forward_hook(_ghook(li)))

    traces = []

    for prompt in tqdm(prompts, desc="Collecting routing traces"):
        prompt = prompt.to(device)
        T = prompt.size(1)
        if T < total_len:
            continue

        inp = prompt[:, :total_len]

        # Full forward pass: collect gate outputs for ALL positions
        gate_bufs = [None] * L
        model(inp)

        routing_steps = []
        for step in range(k_max):
            pos = prompt_len + step
            step_routing = {}
            for li in range(L):
                if gate_bufs[li] is None:
                    continue
                g = gate_bufs[li]
                if g.dim() == 3:
                    g = g[0]
                if pos >= g.size(0):
                    continue
                w = torch.softmax(g[pos], dim=-1)
                _, topk_i = torch.topk(w, top_k)
                step_routing[li] = set(topk_i.tolist())
            routing_steps.append(step_routing)

        traces.append(RoutingTrace(routing=routing_steps))

    for h in hooks:
        h.remove()

    return traces


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Prefetch coverage simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate_prefetch_coverage(
    traces: List[RoutingTrace],
    cache: SimulatedCache,
    L: int, top_k: int,
    k_max: int,
    prefetch_rates: List[int],
    tau_expert: float = 1.5,
) -> dict:
    """Simulate prefetch coverage for different (K, m) combinations.

    Prefetch model:
    - At each draft step, the prefetcher observes the original routing
    - It can start loading m experts per step (PCIe bandwidth constraint)
    - An expert loaded at step k becomes available at step k+1
      (simplification: 1-step latency)
    - At verify time (after K draft steps), count how many needed experts
      are already in cache or have been prefetched

    Returns: dict of (K, m) → {coverage, stall_ms, n_miss_uncovered, ...}
    """
    results = {}

    for m in prefetch_rates:
        for K in range(1, k_max + 1):
            coverages = []
            stall_times = []
            total_needed_list = []
            total_miss_list = []

            for trace in traces:
                if len(trace.routing) < K:
                    continue

                # Collect ALL experts needed across K verify steps
                # (verify runs K+1 tokens but we measure K draft steps' needs)
                all_needed = defaultdict(set)  # layer → set of expert ids
                for step in range(K):
                    for layer, experts in trace.routing[step].items():
                        all_needed[layer].update(experts)

                # Count total needed (not in cache)
                total_needed = 0
                miss_experts = defaultdict(set)  # layer → uncached experts
                for layer, experts in all_needed.items():
                    for eid in experts:
                        if not cache.is_cached(layer, eid):
                            total_needed += 1
                            miss_experts[layer].add(eid)

                # Simulate prefetch: FIFO queue, m experts per step
                # Priority: earliest-seen miss experts first
                prefetched = defaultdict(set)  # layer → set of prefetched expert ids
                prefetch_queue = []  # list of (layer, expert_id)

                for step in range(K):
                    # After observing routing at step, add new misses to queue
                    for layer, experts in trace.routing[step].items():
                        for eid in experts:
                            if (not cache.is_cached(layer, eid)
                                    and eid not in prefetched[layer]
                                    and (layer, eid) not in prefetch_queue):
                                prefetch_queue.append((layer, eid))

                    # Prefetch m experts this step (available next step)
                    n_to_fetch = min(m, len(prefetch_queue))
                    for _ in range(n_to_fetch):
                        if prefetch_queue:
                            pl, pe = prefetch_queue.pop(0)
                            prefetched[pl].add(pe)

                # Coverage = prefetched / total needed
                n_prefetched = sum(len(v) for v in prefetched.values())
                coverage = n_prefetched / max(1, total_needed)
                coverages.append(coverage)

                # Stall = uncovered misses × tau_expert
                n_uncovered = total_needed - n_prefetched
                stall = max(0, n_uncovered) * tau_expert
                stall_times.append(stall)
                total_needed_list.append(total_needed)
                total_miss_list.append(n_uncovered)

            results[(K, m)] = {
                "K": K,
                "m": m,
                "mean_coverage": float(np.mean(coverages)) if coverages else 0,
                "std_coverage": float(np.std(coverages)) if coverages else 0,
                "mean_stall_ms": float(np.mean(stall_times)) if stall_times else 0,
                "std_stall_ms": float(np.std(stall_times)) if stall_times else 0,
                "mean_total_needed": float(np.mean(total_needed_list)) if total_needed_list else 0,
                "mean_uncovered": float(np.mean(total_miss_list)) if total_miss_list else 0,
                "n_traces": len(coverages),
            }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Throughput model with prefetch
# ─────────────────────────────────────────────────────────────────────────────

def compute_throughput_with_prefetch(
    coverage_results: dict,
    alpha_per_step: List[float],
    t_draft_step: float = 2.0,
    t_verify_layer: float = 0.5,
    n_moe_layers: int = 48,
) -> dict:
    """Compute throughput for (K, m) combinations using coverage data."""
    throughput = {}

    for (K, m), cov in coverage_results.items():
        # E[A(K)]
        cumulative_alpha = 1.0
        expected_accepted = 0.0
        for k in range(min(K, len(alpha_per_step))):
            a = alpha_per_step[k]
            if not math.isfinite(a):
                break
            cumulative_alpha *= a
            expected_accepted += cumulative_alpha
        expected_accepted += 1.0

        t_draft = K * t_draft_step
        t_verify = n_moe_layers * t_verify_layer
        t_stall = cov["mean_stall_ms"]
        t_cycle = t_draft + t_verify + t_stall
        tp = expected_accepted / t_cycle if t_cycle > 0 else 0

        throughput[(K, m)] = {
            "K": K, "m": m,
            "expected_accepted": expected_accepted,
            "t_draft_ms": t_draft,
            "t_verify_ms": t_verify,
            "t_stall_ms": t_stall,
            "t_cycle_ms": t_cycle,
            "throughput_tok_per_ms": tp,
            "coverage": cov["mean_coverage"],
        }

    return throughput


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — Coverage as dynamic K signal analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_coverage_signal(
    coverage_results: dict,
    throughput_results: dict,
    prefetch_rates: List[int],
    k_max: int,
    coverage_thresholds: List[float] = [0.4, 0.6, 0.8, 0.9],
) -> dict:
    """Analyze whether coverage threshold predicts optimal stopping.

    For each m, find the K where coverage first exceeds each threshold,
    and compare to K* (argmax throughput for that m).
    """
    signal_analysis = {}

    for m in prefetch_rates:
        # Find K* for this m
        k_star = 1
        best_tp = 0
        for K in range(1, k_max + 1):
            if (K, m) in throughput_results:
                tp = throughput_results[(K, m)]["throughput_tok_per_ms"]
                if tp > best_tp:
                    best_tp = tp
                    k_star = K

        # Find coverage crossing thresholds
        thresh_hits = {}
        for thr in coverage_thresholds:
            cross_k = None
            for K in range(1, k_max + 1):
                if (K, m) in coverage_results:
                    if coverage_results[(K, m)]["mean_coverage"] >= thr:
                        cross_k = K
                        break
            thresh_hits[str(thr)] = {
                "threshold": thr,
                "first_k_above": cross_k,
                "k_star": k_star,
                "delta": (cross_k - k_star) if cross_k else None,
            }

        signal_analysis[str(m)] = {
            "k_star": k_star,
            "best_throughput": best_tp,
            "thresholds": thresh_hits,
        }

    return signal_analysis


# ─────────────────────────────────────────────────────────────────────────────
# Section 8 — Estimate per-step α from miss rate (for traces without full eval)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_alpha_from_traces(
    traces: List[RoutingTrace],
    cache: SimulatedCache,
    L: int, top_k: int,
    k_max: int,
) -> List[float]:
    """Estimate per-step α from cache miss rate.

    Uses the heuristic: α ≈ 1 - β * miss_rate
    where β is calibrated from draft_decode_eval_v2 results (β ≈ 0.6).
    """
    beta = 0.6
    step_alphas = [[] for _ in range(k_max)]

    for trace in traces:
        for step in range(min(k_max, len(trace.routing))):
            n_miss = 0
            n_total = 0
            for layer, experts in trace.routing[step].items():
                for eid in experts:
                    n_total += 1
                    if not cache.is_cached(layer, eid):
                        n_miss += 1
            miss_rate = n_miss / max(1, n_total)
            alpha_est = max(0.0, 1.0 - beta * miss_rate)
            step_alphas[step].append(alpha_est)

    return [float(np.mean(b)) if b else float("nan") for b in step_alphas]


# ─────────────────────────────────────────────────────────────────────────────
# Section 9 — Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(args):
    os.makedirs(args.outdir, exist_ok=True)

    print("Loading model...")
    dtype = getattr(torch, args.dtype) if hasattr(torch, args.dtype) else torch.float16
    model, tokenizer = load_model_and_tokenizer(args.model, args.device, dtype)
    moe_cfg = detect_moe_config(model)
    N, top_k, L = moe_cfg["num_experts"], moe_cfg["top_k"], moe_cfg["num_layers"]
    moe_attr = moe_cfg["moe_attr"]
    print(f"  MoE config: L={L}, N={N}, top_k={top_k}")

    calib_chunks = prepare_chunks(tokenizer, args.n_calib, args.seq_len, args.data_file)
    eval_chunks = prepare_chunks(tokenizer, args.n_eval,
                                 args.prompt_len + args.k_max + 2, args.data_file)

    # Phase 1: Calibration
    print("\n[Phase 1] Calibration...")
    act_freq = calibrate_freq(model, moe_attr, calib_chunks, args.device, N, top_k)

    prefetch_rates = args.prefetch_rates
    all_results = {}

    for ratio in args.cache_ratios:
        print(f"\n{'='*60}")
        print(f"  Cache ratio = {ratio}")
        print(f"{'='*60}")

        cache = SimulatedCache(L, N, ratio, act_freq)

        # Phase 2: Collect routing traces
        print("\n[Phase 2] Collecting routing traces...")
        traces = collect_routing_traces(
            model, moe_attr, eval_chunks, args.device,
            N, top_k, L, args.prompt_len, args.k_max)
        print(f"  Collected {len(traces)} traces")

        # Phase 3: Simulate prefetch coverage
        print("\n[Phase 3] Simulating prefetch coverage...")
        coverage_results = simulate_prefetch_coverage(
            traces, cache, L, top_k, args.k_max,
            prefetch_rates, args.tau_expert)

        # Estimate α from traces
        alpha_est = estimate_alpha_from_traces(
            traces, cache, L, top_k, args.k_max)
        print(f"  Estimated α per step: {[f'{a:.3f}' for a in alpha_est]}")

        # Phase 4: Throughput computation
        print("\n[Phase 4] Computing throughput...")
        throughput = compute_throughput_with_prefetch(
            coverage_results, alpha_est,
            args.t_draft, args.t_verify_layer, L)

        # Phase 5: Coverage signal analysis
        print("\n[Phase 5] Coverage signal analysis...")
        signal = analyze_coverage_signal(
            coverage_results, throughput, prefetch_rates, args.k_max)

        # Summary
        print("\n  Coverage × K × m matrix:")
        print(f"  {'K':>4}", end="")
        for m in prefetch_rates:
            print(f"  m={m:>2} (cov)", end="")
        print()
        for K in range(1, args.k_max + 1):
            print(f"  {K:4d}", end="")
            for m in prefetch_rates:
                if (K, m) in coverage_results:
                    cov = coverage_results[(K, m)]["mean_coverage"]
                    print(f"    {cov:6.3f}", end="")
                else:
                    print(f"    {'N/A':>6}", end="")
            print()

        print(f"\n  K* per prefetch rate:")
        for m in prefetch_rates:
            best_K = 1
            best_tp = 0
            for K in range(1, args.k_max + 1):
                if (K, m) in throughput:
                    tp = throughput[(K, m)]["throughput_tok_per_ms"]
                    if tp > best_tp:
                        best_tp = tp
                        best_K = K
            print(f"    m={m}: K*={best_K}, throughput={best_tp:.4f} tok/ms")

        all_results[str(ratio)] = {
            "coverage": {f"K{k}_m{m}": v for (k, m), v in coverage_results.items()},
            "throughput": {f"K{k}_m{m}": v for (k, m), v in throughput.items()},
            "signal_analysis": signal,
            "alpha_estimated": alpha_est,
        }

    # ── Save results ───────────────────────────────────────────────────────────
    print("\n[Phase 6] Saving results...")
    with open(os.path.join(args.outdir, "e4_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # ── Plots ──────────────────────────────────────────────────────────────────

    # Plot 1: Coverage vs K for different m
    fig, axes = plt.subplots(1, len(args.cache_ratios),
                             figsize=(6 * len(args.cache_ratios), 5))
    if len(args.cache_ratios) == 1:
        axes = [axes]
    m_colors = {0: "#d32f2f", 1: "#ff9800", 2: "#4CAF50", 3: "#2196F3", 4: "#9c27b0"}
    for ax, ratio in zip(axes, args.cache_ratios):
        key = str(ratio)
        if key not in all_results:
            continue
        cov = all_results[key]["coverage"]
        for m in prefetch_rates:
            ks = list(range(1, args.k_max + 1))
            covs = [cov.get(f"K{k}_m{m}", {}).get("mean_coverage", 0) for k in ks]
            ax.plot(ks, covs, marker="o", label=f"m={m}",
                    color=m_colors.get(m, "gray"))
        ax.axhline(0.8, color="gray", linestyle="--", alpha=0.5, label="80% threshold")
        ax.set_xlabel("Draft length K")
        ax.set_ylabel("Prefetch Coverage")
        ax.set_title(f"r = {ratio}")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "coverage_vs_k.png"), dpi=150)
    plt.close()

    # Plot 2: Stall vs K for different m
    fig, axes = plt.subplots(1, len(args.cache_ratios),
                             figsize=(6 * len(args.cache_ratios), 5))
    if len(args.cache_ratios) == 1:
        axes = [axes]
    for ax, ratio in zip(axes, args.cache_ratios):
        key = str(ratio)
        if key not in all_results:
            continue
        cov = all_results[key]["coverage"]
        for m in prefetch_rates:
            ks = list(range(1, args.k_max + 1))
            stalls = [cov.get(f"K{k}_m{m}", {}).get("mean_stall_ms", 0) for k in ks]
            ax.plot(ks, stalls, marker="s", label=f"m={m}",
                    color=m_colors.get(m, "gray"))
        ax.set_xlabel("Draft length K")
        ax.set_ylabel("Verify Stall (ms)")
        ax.set_title(f"r = {ratio}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "stall_vs_k.png"), dpi=150)
    plt.close()

    # Plot 3: Throughput heatmap (K × m)
    for ratio in args.cache_ratios:
        key = str(ratio)
        if key not in all_results:
            continue
        tp_data = all_results[key]["throughput"]
        k_vals = list(range(1, args.k_max + 1))
        m_vals = prefetch_rates

        heatmap = np.zeros((len(m_vals), len(k_vals)))
        for mi, m in enumerate(m_vals):
            for ki, k in enumerate(k_vals):
                entry = tp_data.get(f"K{k}_m{m}", {})
                heatmap[mi, ki] = entry.get("throughput_tok_per_ms", 0)

        fig, ax = plt.subplots(figsize=(max(8, args.k_max * 0.7), max(4, len(m_vals) * 0.8)))
        im = ax.imshow(heatmap, aspect="auto", cmap="YlGn")
        ax.set_xticks(range(len(k_vals)))
        ax.set_xticklabels(k_vals)
        ax.set_yticks(range(len(m_vals)))
        ax.set_yticklabels([f"m={m}" for m in m_vals])
        ax.set_xlabel("Draft length K")
        ax.set_ylabel("Prefetch rate m")
        ax.set_title(f"Throughput (tok/ms), r={ratio}")

        # Annotate cells
        for mi in range(len(m_vals)):
            for ki in range(len(k_vals)):
                val = heatmap[mi, ki]
                ax.text(ki, mi, f"{val:.3f}", ha="center", va="center",
                        fontsize=6, color="black" if val > heatmap.max() * 0.5 else "black")

        # Mark K* per m
        for mi, m in enumerate(m_vals):
            best_ki = np.argmax(heatmap[mi])
            ax.add_patch(plt.Rectangle((best_ki - 0.5, mi - 0.5), 1, 1,
                                       fill=False, edgecolor="red", linewidth=2))

        plt.colorbar(im, ax=ax, label="tok/ms")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f"throughput_heatmap_r{ratio}.png"), dpi=150)
        plt.close()

    print(f"\nResults saved to {args.outdir}/")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="E4: Prefetch coverage analysis")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data_file", default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--cache_ratios", nargs="+", type=float, default=[0.75, 0.50, 0.25])
    parser.add_argument("--prefetch_rates", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--k_max", type=int, default=12)
    parser.add_argument("--n_calib", type=int, default=8)
    parser.add_argument("--n_eval", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--prompt_len", type=int, default=128)
    parser.add_argument("--t_draft", type=float, default=2.0)
    parser.add_argument("--t_verify_layer", type=float, default=0.5)
    parser.add_argument("--tau_expert", type=float, default=1.5)
    parser.add_argument("--outdir", default="./results_e4")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
