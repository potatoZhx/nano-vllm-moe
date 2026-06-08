#!/usr/bin/env python3
"""
Visualize MoE segment-history expert overlap results.

Input JSON format expected from:
  test_moe_segment_history_overlap.py

Main fields used:
  data["config"]["target_segments"]
  data["config"]["history_k"]
  data["global_summary"][target_n]["hit_ratio_by_layer_distance"][layer][distance]["mean"]
  data["global_summary"][target_n]["hit_ratio_by_layer_distance"][layer][distance]["std"]
  data["global_summary"][target_n]["overall_by_distance"][distance]["hit_mean"]

The output is one figure containing:
  1. Heatmap panels: layer x history distance hit_ratio for each target segment.
  2. A bottom line plot: overall hit_ratio vs history distance for all target segments.

Example:
  python plot_moe_segment_history_overlap.py \
    --input segment_history_overlap_results.json \
    --output moe_segment_history_overlap.png

Optional:
  python plot_moe_segment_history_overlap.py \
    --input segment_history_overlap_results.json \
    --metric hit_ratio \
    --max_targets 6 \
    --dpi 200
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot layer-wise MoE expert overlap between target segment n and previous k segments."
    )
    p.add_argument("--input", "-i", required=True, help="Result JSON from test_moe_segment_history_overlap.py")
    p.add_argument("--output", "-o", default="moe_segment_history_overlap.png", help="Output image path")
    p.add_argument(
        "--metric",
        choices=["hit_ratio", "jaccard"],
        default="hit_ratio",
        help="Metric to visualize. hit_ratio = |S_n ∩ S_{n-d}| / |S_n|."
    )
    p.add_argument(
        "--targets",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset of target segments to plot, e.g. --targets 64 128 256."
    )
    p.add_argument(
        "--max_targets",
        type=int,
        default=8,
        help="Maximum number of target-segment heatmaps to show in one figure."
    )
    p.add_argument(
        "--layer_stride",
        type=int,
        default=4,
        help="Y-axis tick stride for layer labels."
    )
    p.add_argument(
        "--distance_stride",
        type=int,
        default=1,
        help="X-axis tick stride for history distance labels."
    )
    p.add_argument(
        "--vmin",
        type=float,
        default=0.0,
        help="Heatmap lower bound."
    )
    p.add_argument(
        "--vmax",
        type=float,
        default=1.0,
        help="Heatmap upper bound."
    )
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional figure title."
    )
    return p.parse_args()


def _metric_keys(metric: str) -> tuple[str, str, str]:
    if metric == "hit_ratio":
        return "hit_ratio_by_layer_distance", "hit_mean", "Hit ratio"
    if metric == "jaccard":
        return "jaccard_by_layer_distance", "jaccard_mean", "Jaccard"
    raise ValueError(f"unsupported metric: {metric}")


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_targets(data: dict[str, Any], requested: list[int] | None, max_targets: int) -> list[int]:
    global_summary = data.get("global_summary", {})
    available = sorted(int(x) for x in global_summary.keys() if str(x).isdigit())

    if requested is None:
        targets = available
    else:
        available_set = set(available)
        missing = [x for x in requested if x not in available_set]
        if missing:
            raise ValueError(f"requested targets not found in global_summary: {missing}; available={available}")
        targets = requested

    if not targets:
        raise ValueError("no target segments found in global_summary")

    if len(targets) > max_targets:
        print(f"[warn] {len(targets)} target segments found; plotting first {max_targets}: {targets[:max_targets]}")
        targets = targets[:max_targets]

    return targets


def infer_layers_and_distances(data: dict[str, Any], targets: list[int], metric: str) -> tuple[list[int], list[int]]:
    layer_metric_key, _, _ = _metric_keys(metric)
    layers = set()
    distances = set()

    for target in targets:
        target_summary = data["global_summary"].get(str(target), {})
        by_layer = target_summary.get(layer_metric_key, {})
        for layer_str, dist_map in by_layer.items():
            layers.add(int(layer_str))
            for d_str, stat in dist_map.items():
                if isinstance(stat, dict) and "mean" in stat:
                    distances.add(int(d_str))

    if not layers:
        raise ValueError(
            f"No layer-distance values found for metric={metric}. "
            f"Expected global_summary[*][{layer_metric_key}][layer][distance]['mean']."
        )
    if not distances:
        raise ValueError("No history distances found in global_summary")

    return sorted(layers), sorted(distances)


def matrix_for_target(
    data: dict[str, Any],
    target: int,
    layers: list[int],
    distances: list[int],
    metric: str,
) -> np.ndarray:
    layer_metric_key, _, _ = _metric_keys(metric)
    by_layer = data["global_summary"][str(target)].get(layer_metric_key, {})

    mat = np.full((len(layers), len(distances)), np.nan, dtype=float)
    layer_to_i = {layer: i for i, layer in enumerate(layers)}
    dist_to_j = {d: j for j, d in enumerate(distances)}

    for layer_str, dist_map in by_layer.items():
        layer = int(layer_str)
        if layer not in layer_to_i:
            continue
        i = layer_to_i[layer]
        for d_str, stat in dist_map.items():
            d = int(d_str)
            if d not in dist_to_j:
                continue
            if isinstance(stat, dict) and "mean" in stat:
                mat[i, dist_to_j[d]] = float(stat["mean"])

    return mat


def overall_curve(
    data: dict[str, Any],
    target: int,
    distances: list[int],
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    _, overall_key, _ = _metric_keys(metric)
    overall = data["global_summary"][str(target)].get("overall_by_distance", {})

    y = np.full(len(distances), np.nan, dtype=float)
    yerr = np.full(len(distances), np.nan, dtype=float)

    std_key = "hit_std" if metric == "hit_ratio" else "jaccard_std"

    for j, d in enumerate(distances):
        stat = overall.get(str(d), {})
        if overall_key in stat:
            y[j] = float(stat[overall_key])
        if std_key in stat:
            yerr[j] = float(stat[std_key])

    return y, yerr


def set_heatmap_ticks(ax, layers: list[int], distances: list[int], layer_stride: int, distance_stride: int):
    x_idx = list(range(0, len(distances), max(1, distance_stride)))
    y_idx = list(range(0, len(layers), max(1, layer_stride)))

    ax.set_xticks(x_idx)
    ax.set_xticklabels([str(distances[i]) for i in x_idx])
    ax.set_yticks(y_idx)
    ax.set_yticklabels([str(layers[i]) for i in y_idx])

    ax.set_xlabel("History distance d: compare S_n with S_{n-d}")
    ax.set_ylabel("MoE layer")


def plot(data: dict[str, Any], args) -> None:
    targets = infer_targets(data, args.targets, args.max_targets)
    layers, distances = infer_layers_and_distances(data, targets, args.metric)
    _, _, metric_label = _metric_keys(args.metric)

    n_targets = len(targets)
    heatmap_cols = min(n_targets, 4)
    heatmap_rows = math.ceil(n_targets / heatmap_cols)

    # One bottom row is reserved for the overall distance curves.
    fig_width = max(7.5, 4.5 * heatmap_cols)
    fig_height = max(6.0, 3.5 * heatmap_rows + 3.2)

    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    gs = fig.add_gridspec(
        heatmap_rows + 1,
        heatmap_cols,
        height_ratios=[1.0] * heatmap_rows + [0.85],
    )

    last_im = None

    for idx, target in enumerate(targets):
        r = idx // heatmap_cols
        c = idx % heatmap_cols
        ax = fig.add_subplot(gs[r, c])

        mat = matrix_for_target(data, target, layers, distances, args.metric)
        last_im = ax.imshow(
            mat,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            vmin=args.vmin,
            vmax=args.vmax,
        )
        ax.set_title(f"Target segment n={target}")
        set_heatmap_ticks(ax, layers, distances, args.layer_stride, args.distance_stride)

    # Hide any unused heatmap slots.
    for idx in range(n_targets, heatmap_rows * heatmap_cols):
        r = idx // heatmap_cols
        c = idx % heatmap_cols
        ax = fig.add_subplot(gs[r, c])
        ax.axis("off")

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=fig.axes[:n_targets], shrink=0.86)
        cbar.set_label(metric_label)

    # Overall line plot across distances.
    ax_line = fig.add_subplot(gs[heatmap_rows, :])
    x = np.array(distances, dtype=int)

    for target in targets:
        y, _ = overall_curve(data, target, distances, args.metric)
        valid = np.isfinite(y)
        if valid.any():
            ax_line.plot(x[valid], y[valid], marker="o", linewidth=1.8, label=f"n={target}")

    ax_line.set_xlabel("History distance d")
    ax_line.set_ylabel(f"Overall {metric_label}")
    ax_line.set_title(f"Layer-averaged {metric_label} vs. history distance")
    ax_line.set_ylim(args.vmin, args.vmax)
    ax_line.grid(True, alpha=0.35)
    ax_line.legend(title="Target segment", ncols=min(len(targets), 4))

    config = data.get("config", {})
    model = config.get("model", "unknown model")
    c = config.get("segment_size", "?")
    k = config.get("history_k", "?")
    num_prompts = config.get("num_prompts", "?")

    default_title = (
        f"MoE segment-history expert overlap: {metric_label}\n"
        f"model={model}, segment_size c={c}, history_k={k}, prompts={num_prompts}"
    )
    fig.suptitle(args.title or default_title, fontsize=13)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    print(f"[ok] saved figure to {output}")


def main():
    args = parse_args()
    data = load_json(args.input)
    plot(data, args)


if __name__ == "__main__":
    main()
