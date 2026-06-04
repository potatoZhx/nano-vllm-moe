#!/usr/bin/env python3
"""
Visualize MoE expert overlap results from test_moe_expert_overlap.py.

Usage:
    python visualize_overlap.py expert_overlap_results.json
    python visualize_overlap.py expert_overlap_results.json --save overlap_plots.png
"""

import argparse
import json
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_results(results: dict, save_path: str | None = None):
    config = results["config"]
    summary = results["global_summary"]
    moe_layers = config["moe_layer_indices"]
    segment_lengths = config["segment_lengths"]

    # Layout: for each segment length, one column with CacheHit + Jaccard overlay
    # Plus one final column for token persistence
    n_cols = len(segment_lengths) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(6.5 * n_cols, 5.5), squeeze=False)
    axes = axes[0]

    c_hit_color = "#D85A30"   # coral
    jaccard_color = "#378ADD"  # blue

    # ---- Main plots: CacheHit & Jaccard per layer ----
    for si, seg_len in enumerate(segment_lengths):
        ax = axes[si]
        key = f"seg_len_{seg_len}"
        if key not in summary:
            ax.set_visible(False)
            continue

        layer_data = summary[key].get("per_layer", {})
        layers = sorted(int(k) for k in layer_data.keys())

        # CacheHit
        ch_means = [layer_data[str(l)].get("cache_hit_mean", 0) for l in layers]
        ch_stds = [layer_data[str(l)].get("cache_hit_std", 0) for l in layers]

        ax.fill_between(
            layers,
            [m - s for m, s in zip(ch_means, ch_stds)],
            [m + s for m, s in zip(ch_means, ch_stds)],
            alpha=0.15, color=c_hit_color,
        )
        ax.plot(layers, ch_means, "o-", color=c_hit_color, markersize=3,
                linewidth=1.8, label="CacheHit |A∩B|/|B|")

        overall_ch = summary[key].get("overall_cache_hit_mean")
        if overall_ch is not None:
            ax.axhline(y=overall_ch, color=c_hit_color, linestyle="--",
                       alpha=0.4, linewidth=1,
                       label=f"CacheHit μ={overall_ch:.3f}")

        # Jaccard (lighter, secondary)
        j_means = [layer_data[str(l)].get("jaccard_mean", 0) for l in layers]
        j_stds = [layer_data[str(l)].get("jaccard_std", 0) for l in layers]

        ax.fill_between(
            layers,
            [m - s for m, s in zip(j_means, j_stds)],
            [m + s for m, s in zip(j_means, j_stds)],
            alpha=0.1, color=jaccard_color,
        )
        ax.plot(layers, j_means, "s-", color=jaccard_color, markersize=2,
                linewidth=1.2, alpha=0.7, label="Jaccard |A∩B|/|A∪B|")

        overall_j = summary[key].get("overall_jaccard_mean")
        if overall_j is not None:
            ax.axhline(y=overall_j, color=jaccard_color, linestyle=":",
                       alpha=0.35, linewidth=1,
                       label=f"Jaccard μ={overall_j:.3f}")

        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Overlap Rate")
        ax.set_title(f"Segment length = {seg_len} tokens")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.25)

    # ---- Last plot: token-level persistence ----
    ax = axes[-1]
    tp = summary.get("token_persistence", {})
    if tp:
        layers = sorted(int(k) for k in tp.keys())
        vals = [tp[str(l)] for l in layers]
        ax.bar(layers, vals, color="steelblue", alpha=0.7, width=0.8)
        ax.axhline(y=np.mean(vals), color="red", linestyle="--", alpha=0.6,
                    label=f"mean={np.mean(vals):.3f}")
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Persistence Rate")
        ax.set_title("Token-level expert persistence\n(consecutive token overlap)")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
    else:
        ax.set_visible(False)

    fig.suptitle(
        f"MoE Expert Activation Overlap – {config['model']}\n"
        f"({config['num_moe_layers']} MoE layers, {config['num_experts']} experts, "
        f"top-{config['top_k']}, {config['num_prompts']} prompts)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[*] Plot saved to {save_path}")
    else:
        plt.show()


def plot_per_prompt_heatmap(results: dict, seg_len: int, save_path: str | None = None):
    """Heatmap: rows = prompts, columns = layers, color = CacheHit rate."""
    config = results["config"]
    moe_layers = config["moe_layer_indices"]
    per_prompt = results["per_prompt"]

    matrix = []
    labels = []

    for pi, pr in enumerate(per_prompt):
        seg_data = pr.get("segment_analyses", {}).get(str(seg_len), {})
        row = []
        for li in moe_layers:
            layer_m = seg_data.get(str(li), {})
            # Prefer cache_hit_mean; fall back to jaccard_mean for old results
            val = layer_m.get("cache_hit_mean", layer_m.get("jaccard_mean", np.nan))
            row.append(val)
        matrix.append(row)
        labels.append(f"P{pi+1}: {pr['prompt'][:30]}...")

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(max(14, len(moe_layers) * 0.3), max(4, len(labels) * 0.5)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(moe_layers)))
    ax.set_xticklabels(moe_layers, fontsize=6, rotation=90)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("MoE Layer Index")
    ax.set_ylabel("Prompt")
    ax.set_title(f"CacheHit Rate |A∩B|/|B| per Prompt × Layer  (segment = {seg_len} tokens)")
    plt.colorbar(im, ax=ax, label="CacheHit Rate")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[*] Heatmap saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", help="JSON results from test_moe_expert_overlap.py")
    parser.add_argument("--save", type=str, default=None, help="Save plot to file")
    parser.add_argument("--heatmap", action="store_true", help="Also plot per-prompt heatmap")
    parser.add_argument("--heatmap_save", type=str, default=None)
    args = parser.parse_args()

    results = load_results(args.results_file)
    plot_results(results, save_path=args.save)

    if args.heatmap:
        seg_len = results["config"]["segment_lengths"][0]
        plot_per_prompt_heatmap(results, seg_len, save_path=args.heatmap_save)
