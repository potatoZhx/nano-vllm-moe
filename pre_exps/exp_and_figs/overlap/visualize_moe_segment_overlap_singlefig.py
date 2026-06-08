#!/usr/bin/env python3
"""
Visualize MoE segment-history expert-overlap results.

This script reads the JSON produced by test_moe_segment_history_overlap.py and
generates:
  1. layer x history-lag heatmaps for each target segment
  2. mean metric vs history lag curves
  3. mean metric vs layer curves
  4. selected-layer history-lag curves
  5. optional single combined figure containing all panels
  6. a flattened CSV table for custom analysis

Typical usage:
  python visualize_moe_segment_overlap.py \
    --input segment_history_overlap_results.json \
    --output_dir figs \
    --metric hit_ratio_mean

Put all plots into one large figure:
  python visualize_moe_segment_overlap.py \
    --input segment_history_overlap_results.json \
    --output_dir figs \
    --metric hit_ratio_mean \
    --single_figure

Only draw the combined figure, skip separate PNGs:
  python visualize_moe_segment_overlap.py \
    --input segment_history_overlap_results.json \
    --output_dir figs \
    --metric hit_ratio_mean \
    --single_figure \
    --only_single_figure

Supported metric names usually include:
  hit_ratio, hit_ratio_mean
  reverse_hit_ratio, reverse_hit_ratio_mean
  jaccard, jaccard_mean
  overlap_coefficient, overlap_coefficient_mean
  intersection_size, intersection_size_mean
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------
# JSON flattening utilities
# -----------------------------

LAYER_KEYS = ("layer", "layer_idx", "layer_id", "layer_index")
TARGET_KEYS = ("target_segment", "target_n", "target", "n")
LAG_KEYS = ("history_delta", "delta", "lag", "distance", "history_lag", "d")
PROMPT_KEYS = ("prompt_idx", "prompt_id", "prompt_index")


def _as_int(x: Any) -> Optional[int]:
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float) and x.is_integer():
        return int(x)
    if isinstance(x, str):
        m = re.search(r"-?\d+", x)
        if m:
            return int(m.group(0))
    return None


def _first_int(d: Dict[str, Any], keys: Iterable[str]) -> Optional[int]:
    for k in keys:
        if k in d:
            v = _as_int(d[k])
            if v is not None:
                return v
    return None


def _extract_metric_value(d: Dict[str, Any], metric: str) -> Optional[float]:
    candidates = [
        metric,
        metric.replace("_mean", ""),
        metric + "_mean" if not metric.endswith("_mean") else metric,
    ]

    aliases = {
        "hit_ratio": ["hit", "cache_hit", "cache_hit_rate"],
        "hit_ratio_mean": ["hit_mean", "cache_hit_mean", "cache_hit_rate_mean"],
        "jaccard": ["jaccard_similarity"],
        "jaccard_mean": ["jaccard_similarity_mean"],
    }
    for c in list(candidates):
        candidates.extend(aliases.get(c, []))

    seen = set()
    for k in candidates:
        if k in seen:
            continue
        seen.add(k)
        if k in d and isinstance(d[k], (int, float)):
            return float(d[k])
    return None


def _candidate_record(d: Dict[str, Any], metric: str) -> Optional[Dict[str, Any]]:
    layer = _first_int(d, LAYER_KEYS)
    target = _first_int(d, TARGET_KEYS)
    lag = _first_int(d, LAG_KEYS)
    value = _extract_metric_value(d, metric)

    if layer is None or target is None or lag is None or value is None:
        return None

    rec = {
        "prompt_idx": _first_int(d, PROMPT_KEYS),
        "target_segment": target,
        "history_delta": lag,
        "layer": layer,
        "value": value,
    }

    for k in [
        "intersection_size",
        "intersection_size_mean",
        "target_size",
        "history_size",
        "target_expert_count",
        "history_expert_count",
        "jaccard",
        "jaccard_mean",
        "hit_ratio",
        "hit_ratio_mean",
        "reverse_hit_ratio",
        "reverse_hit_ratio_mean",
        "overlap_coefficient",
        "overlap_coefficient_mean",
    ]:
        if k in d and isinstance(d[k], (int, float)):
            rec[k] = float(d[k])

    return rec


def _walk_generic(obj: Any, metric: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    if isinstance(obj, dict):
        rec = _candidate_record(obj, metric)
        if rec is not None:
            rows.append(rec)

        for v in obj.values():
            rows.extend(_walk_generic(v, metric))

    elif isinstance(obj, list):
        for v in obj:
            rows.extend(_walk_generic(v, metric))

    return rows


def _parse_global_summary_style(data: Dict[str, Any], metric: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    summary = data.get("global_summary")
    if not isinstance(summary, dict):
        return rows

    for target_key, target_obj in summary.items():
        target = _as_int(target_key)
        if target is None or not isinstance(target_obj, dict):
            continue

        per_layer = target_obj.get("per_layer", target_obj.get("layers"))
        if not isinstance(per_layer, dict):
            continue

        for layer_key, layer_obj in per_layer.items():
            layer = _as_int(layer_key)
            if layer is None or not isinstance(layer_obj, dict):
                continue

            history_obj = (
                layer_obj.get("history")
                or layer_obj.get("history_deltas")
                or layer_obj.get("lags")
                or layer_obj.get("per_history")
                or layer_obj.get("per_lag")
            )

            if isinstance(history_obj, dict):
                for lag_key, metric_obj in history_obj.items():
                    lag = _as_int(lag_key)
                    if lag is None or not isinstance(metric_obj, dict):
                        continue
                    value = _extract_metric_value(metric_obj, metric)
                    if value is not None:
                        rows.append({
                            "prompt_idx": None,
                            "target_segment": target,
                            "history_delta": lag,
                            "layer": layer,
                            "value": value,
                        })

            for maybe_lag_key, metric_obj in layer_obj.items():
                if not isinstance(metric_obj, dict):
                    continue
                lag = _as_int(maybe_lag_key)
                if lag is None:
                    continue
                value = _extract_metric_value(metric_obj, metric)
                if value is not None:
                    rows.append({
                        "prompt_idx": None,
                        "target_segment": target,
                        "history_delta": lag,
                        "layer": layer,
                        "value": value,
                    })

    return rows


def load_records(path: Path, metric: str) -> pd.DataFrame:
    data = json.loads(path.read_text())

    rows = []
    if isinstance(data, dict):
        rows.extend(_parse_global_summary_style(data, metric))
    rows.extend(_walk_generic(data, metric))

    if not rows:
        raise RuntimeError(
            f"No plottable records found for metric={metric!r}. "
            "Try --metric hit_ratio, --metric hit_ratio_mean, --metric jaccard, "
            "or inspect your JSON schema."
        )

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(
        subset=["prompt_idx", "target_segment", "history_delta", "layer", "value"]
    )

    df = (
        df.groupby(["target_segment", "history_delta", "layer"], as_index=False)
        .agg(value=("value", "mean"), count=("value", "size"))
        .sort_values(["target_segment", "history_delta", "layer"])
    )
    return df


# -----------------------------
# Plotting utilities
# -----------------------------

def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def choose_representative_layers(df: pd.DataFrame) -> List[int]:
    layers = sorted(df["layer"].unique().tolist())
    if not layers:
        return []
    idxs = [0, len(layers) // 4, len(layers) // 2, 3 * len(layers) // 4, len(layers) - 1]
    return sorted({layers[i] for i in idxs})


def _plot_heatmap_on_ax(ax, df: pd.DataFrame, metric: str, target: int):
    sub = df[df["target_segment"] == target]
    pivot = sub.pivot(index="layer", columns="history_delta", values="value").sort_index()

    im = ax.imshow(pivot.values, aspect="auto", origin="lower")
    ax.set_title(f"Heatmap, target={target}")
    ax.set_xlabel("history delta d")
    ax.set_ylabel("layer")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45, ha="right", fontsize=8)

    layers = list(pivot.index)
    if len(layers) <= 24:
        tick_positions = np.arange(len(layers))
    else:
        step = max(1, math.ceil(len(layers) / 12))
        tick_positions = np.arange(0, len(layers), step)

    ax.set_yticks(tick_positions)
    ax.set_yticklabels([str(layers[i]) for i in tick_positions], fontsize=8)
    return im


def _plot_vs_history_on_ax(ax, df: pd.DataFrame, metric: str, target: int):
    sub = df[df["target_segment"] == target]
    curve = (
        sub.groupby("history_delta", as_index=False)
        .agg(mean=("value", "mean"), std=("value", "std"), n=("value", "size"))
        .sort_values("history_delta")
    )
    curve["std"] = curve["std"].fillna(0.0)

    ax.plot(curve["history_delta"], curve["mean"], marker="o")
    ax.fill_between(
        curve["history_delta"],
        curve["mean"] - curve["std"],
        curve["mean"] + curve["std"],
        alpha=0.2,
    )

    ax.set_title(f"Mean vs history lag, target={target}")
    ax.set_xlabel("history delta d")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)


def _plot_vs_layer_on_ax(ax, df: pd.DataFrame, metric: str, target: int):
    sub = df[df["target_segment"] == target]
    curve = (
        sub.groupby("layer", as_index=False)
        .agg(mean=("value", "mean"), std=("value", "std"), n=("value", "size"))
        .sort_values("layer")
    )
    curve["std"] = curve["std"].fillna(0.0)

    ax.plot(curve["layer"], curve["mean"], marker="o", markersize=3)
    ax.fill_between(
        curve["layer"],
        curve["mean"] - curve["std"],
        curve["mean"] + curve["std"],
        alpha=0.2,
    )

    ax.set_title(f"Mean vs layer, target={target}")
    ax.set_xlabel("layer")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)


def _plot_selected_layers_on_ax(ax, df: pd.DataFrame, metric: str, target: int, layers: List[int]):
    sub = df[(df["target_segment"] == target) & (df["layer"].isin(layers))]

    for layer, s in sub.groupby("layer"):
        s = s.sort_values("history_delta")
        ax.plot(s["history_delta"], s["value"], marker="o", markersize=3, label=f"L{layer}")

    ax.set_title(f"Selected layers vs lag, target={target}")
    ax.set_xlabel("history delta d")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)


def plot_heatmap(df: pd.DataFrame, out_dir: Path, metric: str, target: int) -> None:
    sub = df[df["target_segment"] == target]
    pivot = sub.pivot(index="layer", columns="history_delta", values="value").sort_index()

    fig_w = max(8, 0.55 * len(pivot.columns) + 4)
    fig_h = max(6, 0.16 * len(pivot.index) + 3)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = _plot_heatmap_on_ax(ax, df, metric, target)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric)

    fig.tight_layout()
    fig.savefig(out_dir / f"heatmap_target_{target}_{metric}.png", dpi=200)
    plt.close(fig)


def plot_metric_vs_history(df: pd.DataFrame, out_dir: Path, metric: str, target: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_vs_history_on_ax(ax, df, metric, target)
    fig.tight_layout()
    fig.savefig(out_dir / f"vs_history_target_{target}_{metric}.png", dpi=200)
    plt.close(fig)


def plot_metric_vs_layer(df: pd.DataFrame, out_dir: Path, metric: str, target: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    _plot_vs_layer_on_ax(ax, df, metric, target)
    fig.tight_layout()
    fig.savefig(out_dir / f"vs_layer_target_{target}_{metric}.png", dpi=200)
    plt.close(fig)


def plot_layer_history_lines(df: pd.DataFrame, out_dir: Path, metric: str, target: int, layers: List[int]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    _plot_selected_layers_on_ax(ax, df, metric, target, layers)
    fig.tight_layout()
    fig.savefig(out_dir / f"selected_layers_vs_history_target_{target}_{metric}.png", dpi=200)
    plt.close(fig)


def plot_all_targets_vs_history(df: pd.DataFrame, out_dir: Path, metric: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    for target, sub in df.groupby("target_segment"):
        curve = (
            sub.groupby("history_delta", as_index=False)
            .agg(mean=("value", "mean"))
            .sort_values("history_delta")
        )
        ax.plot(curve["history_delta"], curve["mean"], marker="o", label=f"target={target}")

    ax.set_title(f"{metric} vs history lag, all target segments")
    ax.set_xlabel("history delta d")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / f"all_targets_vs_history_{metric}.png", dpi=200)
    plt.close(fig)


def plot_single_combined_figure(
    df: pd.DataFrame,
    out_dir: Path,
    metric: str,
    targets: List[int],
    selected_layers: Optional[List[int]] = None,
    file_prefix: str = "combined",
) -> None:
    """
    One row per target segment, four columns:
      1. layer x history lag heatmap
      2. mean metric vs history lag
      3. mean metric vs layer
      4. selected layer curves vs history lag
    """
    nrows = len(targets)
    ncols = 4

    fig_w = 24
    fig_h = max(5.2 * nrows, 6)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), squeeze=False)
    fig.suptitle(f"MoE Segment-History Expert Overlap Visualization: {metric}", fontsize=16, y=0.995)

    for r, target in enumerate(targets):
        target_df = df[df["target_segment"] == target]
        layers = selected_layers or choose_representative_layers(target_df)

        im = _plot_heatmap_on_ax(axes[r][0], df, metric, target)
        _plot_vs_history_on_ax(axes[r][1], df, metric, target)
        _plot_vs_layer_on_ax(axes[r][2], df, metric, target)
        _plot_selected_layers_on_ax(axes[r][3], df, metric, target, layers)

        cbar = fig.colorbar(im, ax=axes[r][0], fraction=0.046, pad=0.04)
        cbar.set_label(metric)

    fig.tight_layout(rect=[0, 0, 1, 0.985])

    png_path = out_dir / f"{file_prefix}_{metric}.png"
    pdf_path = out_dir / f"{file_prefix}_{metric}.pdf"

    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"[+] Single combined PNG saved to: {png_path}")
    print(f"[+] Single combined PDF saved to: {pdf_path}")


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize MoE expert segment-history overlap JSON results.")
    p.add_argument("--input", required=True, type=Path, help="Path to result JSON.")
    p.add_argument("--output_dir", type=Path, default=Path("moe_overlap_figs"), help="Directory for figures.")
    p.add_argument("--metric", default="hit_ratio_mean", help="Metric to plot, e.g. hit_ratio_mean, hit_ratio, jaccard_mean.")
    p.add_argument(
        "--targets",
        type=int,
        nargs="*",
        default=None,
        help="Optional target segments to plot. Default: all targets found in JSON.",
    )
    p.add_argument(
        "--selected_layers",
        type=int,
        nargs="*",
        default=None,
        help="Optional layers for selected-layer history-lag plots. Default: representative layers.",
    )
    p.add_argument(
        "--single_figure",
        action="store_true",
        help="Put all requested targets and plot types into one large combined figure.",
    )
    p.add_argument(
        "--only_single_figure",
        action="store_true",
        help="When used with --single_figure, skip separate per-plot PNG files.",
    )
    p.add_argument(
        "--combined_prefix",
        type=str,
        default="combined",
        help="Filename prefix for the combined figure.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir)

    df = load_records(args.input, args.metric)

    csv_path = args.output_dir / f"flattened_{args.metric}.csv"
    df.to_csv(csv_path, index=False)

    found_targets = sorted(df["target_segment"].unique().tolist())
    targets = args.targets if args.targets else found_targets
    targets = [t for t in targets if t in found_targets]

    if not targets:
        raise RuntimeError(f"No requested targets found. Available targets: {found_targets}")

    print(f"[+] Loaded {len(df)} aggregated records")
    print(f"[+] Available targets: {found_targets}")
    print(f"[+] Plotting targets: {targets}")
    print(f"[+] Available layers: {df['layer'].min()}..{df['layer'].max()} ({df['layer'].nunique()} layers)")
    print(f"[+] Available history deltas: {sorted(df['history_delta'].unique().tolist())}")
    print(f"[+] Flattened CSV saved to: {csv_path}")

    if args.single_figure:
        plot_single_combined_figure(
            df=df[df["target_segment"].isin(targets)],
            out_dir=args.output_dir,
            metric=args.metric,
            targets=targets,
            selected_layers=args.selected_layers,
            file_prefix=args.combined_prefix,
        )

    if not args.only_single_figure:
        for target in targets:
            plot_heatmap(df, args.output_dir, args.metric, target)
            plot_metric_vs_history(df, args.output_dir, args.metric, target)
            plot_metric_vs_layer(df, args.output_dir, args.metric, target)

            layers = args.selected_layers or choose_representative_layers(df[df["target_segment"] == target])
            if layers:
                plot_layer_history_lines(df, args.output_dir, args.metric, target, layers)

        if len(targets) > 1:
            plot_all_targets_vs_history(df[df["target_segment"].isin(targets)], args.output_dir, args.metric)

    print(f"[+] Figures saved under: {args.output_dir}")


if __name__ == "__main__":
    main()
