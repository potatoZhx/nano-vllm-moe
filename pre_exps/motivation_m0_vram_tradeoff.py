"""
motivation_m0_vram_tradeoff.py
==============================
M0 Experiment: VRAM-Cache Ratio Trade-off Analysis

Quantifies how INT4 draft model VRAM overhead (as in MoE-SpeQ) reduces
available expert cache space compared to our zero-overhead approach.

Can run in two modes:
  1. Analytical-only (no model): uses CLI-provided or default Qwen3-30B-A3B params
  2. With model: auto-detects config, optionally runs rerouting eval at each ratio

Usage:
  # Analytical only:
  python motivation_m0_vram_tradeoff.py --outdir ./results_m0

  # Auto-detect from model:
  python motivation_m0_vram_tradeoff.py \
      --model /path/to/Qwen3-30B-A3B-Base \
      --outdir ./results_m0

  # With rerouting eval:
  python motivation_m0_vram_tradeoff.py \
      --model /path/to/Qwen3-30B-A3B-Base \
      --data_file ../wikitext2_test.txt \
      --run_eval --outdir ./results_m0
"""

from __future__ import annotations

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Model parameter defaults (Qwen3-30B-A3B)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS = {
    "num_layers":               48,
    "num_experts":             128,
    "top_k":                     8,
    "hidden_size":            2048,
    "moe_intermediate_size":  1408,
    "num_attention_heads":      32,
    "num_key_value_heads":       4,
    "head_dim":                128,
    "vocab_size":           151936,
    "shared_expert_intermediate_size": 5632,
}


def detect_from_model(model_path: str) -> dict:
    """Auto-detect model parameters from config."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    return {
        "num_layers": cfg.num_hidden_layers,
        "num_experts": getattr(cfg, "num_experts",
                               getattr(cfg, "n_routed_experts", DEFAULTS["num_experts"])),
        "top_k": getattr(cfg, "num_experts_per_tok",
                         getattr(cfg, "top_k", DEFAULTS["top_k"])),
        "hidden_size": cfg.hidden_size,
        "moe_intermediate_size": getattr(cfg, "moe_intermediate_size",
                                         DEFAULTS["moe_intermediate_size"]),
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": getattr(cfg, "num_key_value_heads",
                                       cfg.num_attention_heads),
        "head_dim": getattr(cfg, "head_dim",
                            cfg.hidden_size // cfg.num_attention_heads),
        "vocab_size": cfg.vocab_size,
        "shared_expert_intermediate_size": getattr(
            cfg, "shared_expert_intermediate_size",
            DEFAULTS["shared_expert_intermediate_size"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# VRAM budget computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_vram_breakdown(params: dict, max_seq_len: int = 2048) -> dict:
    """Compute VRAM usage breakdown in bytes."""
    L   = params["num_layers"]
    N   = params["num_experts"]
    D   = params["hidden_size"]
    D_e = params["moe_intermediate_size"]
    D_s = params["shared_expert_intermediate_size"]
    n_h = params["num_attention_heads"]
    n_kv= params["num_key_value_heads"]
    d_h = params["head_dim"]
    V   = params["vocab_size"]

    # Expert: gate_proj + up_proj + down_proj = 3 * D * D_e
    expert_params = 3 * D * D_e
    expert_bytes_fp16 = expert_params * 2
    expert_bytes_int4 = expert_params // 2  # 4 bit per param

    total_expert_bytes_fp16 = N * L * expert_bytes_fp16
    total_expert_bytes_int4 = N * L * expert_bytes_int4

    # Shared expert per layer: 3 * D * D_s
    shared_expert_bytes = L * 3 * D * D_s * 2

    # Gate (router) per layer: D * N
    gate_bytes = L * D * N * 2

    # Attention per layer: Q + K + V + O projections
    attn_bytes = L * (D * n_h * d_h + 2 * D * n_kv * d_h + n_h * d_h * D) * 2

    # Layer norms (2 per layer: attention + MoE)
    norm_bytes = L * 2 * D * 2

    # Embedding + LM head
    embed_bytes = V * D * 2 * 2

    non_expert_bytes = (shared_expert_bytes + gate_bytes + attn_bytes
                        + norm_bytes + embed_bytes)

    # KV cache: 2 * L * n_kv * d_h * max_seq_len * 2 bytes
    kv_cache_bytes = 2 * L * n_kv * d_h * max_seq_len * 2

    overhead_bytes = 512 * 1024 * 1024  # ~512 MB CUDA overhead

    return {
        "expert_bytes_fp16_per": expert_bytes_fp16,
        "expert_bytes_int4_per": expert_bytes_int4,
        "total_expert_bytes_fp16": total_expert_bytes_fp16,
        "total_expert_bytes_int4": total_expert_bytes_int4,
        "non_expert_bytes": non_expert_bytes,
        "kv_cache_bytes": kv_cache_bytes,
        "overhead_bytes": overhead_bytes,
        "L": L, "N": N,
    }


def compute_cache_ratios(breakdown: dict,
                         vram_budgets_gb: list[float]) -> dict:
    """For each VRAM budget, compute cache ratio for ours vs MoE-SpeQ."""
    L = breakdown["L"]
    N = breakdown["N"]
    expert_fp16 = breakdown["expert_bytes_fp16_per"]
    non_expert  = breakdown["non_expert_bytes"]
    kv_cache    = breakdown["kv_cache_bytes"]
    overhead    = breakdown["overhead_bytes"]
    int4_total  = breakdown["total_expert_bytes_int4"]

    results = {"vram_gb": [], "ours_ratio": [], "speq_ratio": [],
               "ours_per_layer": [], "speq_per_layer": []}

    for budget_gb in vram_budgets_gb:
        budget = budget_gb * (1024 ** 3)

        # Ours: all VRAM available for caching
        avail_ours = max(0, budget - non_expert - kv_cache - overhead)
        cached_total_ours = avail_ours / expert_fp16
        per_layer_ours = cached_total_ours / L
        ratio_ours = min(per_layer_ours / N, 1.0)

        # MoE-SpeQ: must also store INT4 expert weights
        avail_speq = max(0, budget - non_expert - kv_cache - overhead - int4_total)
        cached_total_speq = avail_speq / expert_fp16
        per_layer_speq = max(0, cached_total_speq / L)
        ratio_speq = min(per_layer_speq / N, 1.0)

        results["vram_gb"].append(budget_gb)
        results["ours_ratio"].append(ratio_ours)
        results["speq_ratio"].append(ratio_speq)
        results["ours_per_layer"].append(per_layer_ours)
        results["speq_per_layer"].append(per_layer_speq)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_breakdown(params: dict, breakdown: dict):
    GB = 1024 ** 3
    print("\n" + "=" * 60)
    print("  VRAM Breakdown")
    print("=" * 60)
    print(f"  Model: L={breakdown['L']} layers, N={breakdown['N']} experts/layer")
    print(f"  Expert FP16 size:  {breakdown['expert_bytes_fp16_per']/1e6:.2f} MB each")
    print(f"  Expert INT4 size:  {breakdown['expert_bytes_int4_per']/1e6:.2f} MB each")
    print(f"  Total FP16 experts: {breakdown['total_expert_bytes_fp16']/GB:.2f} GB")
    print(f"  Total INT4 experts: {breakdown['total_expert_bytes_int4']/GB:.2f} GB")
    print(f"  Non-expert params:  {breakdown['non_expert_bytes']/GB:.2f} GB")
    print(f"  KV cache (2048):    {breakdown['kv_cache_bytes']/GB:.2f} GB")
    print(f"  CUDA overhead:      {breakdown['overhead_bytes']/GB:.2f} GB")
    print()


def print_comparison(results: dict, breakdown: dict):
    N = breakdown["N"]
    print(f"\n{'VRAM (GB)':>10} | {'Ours ratio':>12} {'(per layer)':>12} | "
          f"{'SpeQ ratio':>12} {'(per layer)':>12} | {'Delta':>8}")
    print("-" * 80)
    for i, gb in enumerate(results["vram_gb"]):
        ro = results["ours_ratio"][i]
        rs = results["speq_ratio"][i]
        po = results["ours_per_layer"][i]
        ps = results["speq_per_layer"][i]
        delta = ro - rs
        print(f"  {gb:>7.0f}   |  {ro:>8.4f}    {po:>8.1f}/{N}  | "
              f" {rs:>8.4f}    {ps:>8.1f}/{N}  |  {delta:>+.4f}")


def plot_comparison(results: dict, outdir: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    xs = results["vram_gb"]

    # Cache ratio
    ax1.plot(xs, results["ours_ratio"], "o-", color="#1D9E75",
             linewidth=2.5, markersize=8, label="Ours (zero overhead)")
    ax1.plot(xs, results["speq_ratio"], "s--", color="#d4640c",
             linewidth=2.5, markersize=8, label="MoE-SpeQ (INT4 draft)")
    ax1.set_xlabel("Total VRAM Budget (GB)", fontsize=12)
    ax1.set_ylabel("Expert Cache Ratio", fontsize=12)
    ax1.set_title("Cache Ratio vs VRAM Budget", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Cached experts per layer
    ax2.plot(xs, results["ours_per_layer"], "o-", color="#1D9E75",
             linewidth=2.5, markersize=8, label="Ours")
    ax2.plot(xs, results["speq_per_layer"], "s--", color="#d4640c",
             linewidth=2.5, markersize=8, label="MoE-SpeQ")
    ax2.set_xlabel("Total VRAM Budget (GB)", fontsize=12)
    ax2.set_ylabel("Cached Experts per Layer", fontsize=12)
    ax2.set_title("Cache Capacity vs VRAM Budget", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    # Annotate key VRAM budgets
    for ax in [ax1, ax2]:
        for gb_mark in [24, 40, 80]:
            if gb_mark in xs or any(abs(x - gb_mark) < 0.5 for x in xs):
                ax.axvline(gb_mark, color="gray", linestyle=":", alpha=0.5)

    plt.suptitle("M0: VRAM-Cache Trade-off (Expert Rerouting vs MoE-SpeQ)",
                 fontsize=14)
    plt.tight_layout()
    path = os.path.join(outdir, "m0_vram_tradeoff.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def save_results(results: dict, breakdown: dict, outdir: str):
    data = {"breakdown": {k: (v if isinstance(v, (int, float)) else v)
                          for k, v in breakdown.items()},
            "comparison": results}
    path = os.path.join(outdir, "m0_results.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="M0: VRAM-Cache Ratio Trade-off (analytical)")
    p.add_argument("--model", default=None,
                   help="Model path for auto-detecting config (optional)")
    p.add_argument("--vram_budgets", nargs="+", type=float,
                   default=[16, 20, 24, 32, 40, 48, 80],
                   help="VRAM budgets to evaluate (GB)")
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--outdir", default="./results_m0")
    # Manual overrides (used when --model is not provided)
    p.add_argument("--num_layers", type=int, default=None)
    p.add_argument("--num_experts", type=int, default=None)
    p.add_argument("--hidden_size", type=int, default=None)
    p.add_argument("--moe_intermediate_size", type=int, default=None)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Get model parameters
    if args.model:
        print(f"Auto-detecting config from {args.model} ...")
        params = detect_from_model(args.model)
    else:
        params = dict(DEFAULTS)

    # Apply manual overrides
    for key in ["num_layers", "num_experts", "hidden_size",
                "moe_intermediate_size"]:
        v = getattr(args, key, None)
        if v is not None:
            params[key] = v

    print(f"Model params: L={params['num_layers']}, N={params['num_experts']}, "
          f"D={params['hidden_size']}, D_e={params['moe_intermediate_size']}")

    # Compute VRAM breakdown
    breakdown = compute_vram_breakdown(params, args.max_seq_len)
    print_breakdown(params, breakdown)

    # Compute comparison
    results = compute_cache_ratios(breakdown, args.vram_budgets)
    print_comparison(results, breakdown)

    # Save & plot
    plot_comparison(results, args.outdir)
    save_results(results, breakdown, args.outdir)

    print(f"\nDone. Results in {args.outdir}/")


if __name__ == "__main__":
    main()
