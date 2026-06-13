#!/usr/bin/env python3
"""
Visualize MoE segment unique expert counts from
`test_moe_segment_unique_expert_counts.py` outputs.

Default visualization: one compact heatmap figure.
  - x-axis: MoE layer index
  - y-axis: (start token n, segment size c)
  - cell value/color: mean unique_expert_count averaged across prompts and segments

This makes different n and c directly comparable in one image.

Examples:
  python plot_moe_segment_unique_expert_counts.py \
      --csv segment_unique_expert_counts.csv \
      --out segment_unique_expert_counts_heatmap.png

  python plot_moe_segment_unique_expert_counts.py \
      --json segment_unique_expert_counts.json \
      --out segment_unique_expert_counts_heatmap.png

  python plot_moe_segment_unique_expert_counts.py \
      --csv segment_unique_expert_counts.csv \
      --plot_type lines \
      --start_tokens 64 1024 \
      --segment_sizes 3 5 8 \
      --out segment_unique_expert_counts_lines.png

  python plot_moe_segment_unique_expert_counts.py \
      --csv segment_unique_expert_counts.csv \
      --plot_type c_sweep \
      --layers 0 12 24 36 47 \
      --start_tokens 64 1024 \
      --segment_sizes 3 5 8 \
      --out segment_unique_expert_counts_c_sweep.png

    python plot_moe_segment_unique_expert_counts_v2.py \
        --csv segment_unique_expert_counts.csv \
        --plot_type c_sweep \
        --start_tokens 64 1024 4096 \
        --segment_sizes 3 5 8 12 16 \
        --layers 0 12 24 36 47 \
        --out unique_count_vs_c_selected_layers.png

    python plot_moe_segment_unique_expert_counts_v2.py \
        --csv segment_unique_expert_counts.csv \
        --plot_type lines \
        --start_tokens 64 1024 \
        --segment_sizes 3 5 8 12 16 \
        --out unique_count_by_layer_lines_v2.png

Requirements:
  pip install pandas numpy matplotlib
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REQUIRED_CSV_COLUMNS = {
    "prompt_index",
    "start_token_n_1based",
    "segment_size_c",
    "layer_idx",
    "segment_index",
    "unique_expert_count",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot unique routed expert counts by layer for different start tokens n and segment sizes c."
    )
    p.add_argument("--csv", type=str, default=None, help="CSV output from the experiment script.")
    p.add_argument("--json", type=str, default=None, help="JSON output from the experiment script. Used if --csv is absent.")
    p.add_argument("--out", type=str, default="segment_unique_expert_counts_heatmap.png", help="Output figure path: .png/.pdf/.svg.")
    p.add_argument("--summary_out", type=str, default=None, help="Optional path to save the aggregated plotting table as CSV.")
    p.add_argument(
        "--plot_type",
        choices=["heatmap", "lines", "c_sweep"],
        default="heatmap",
        help=(
            "Visualization style. "
            "heatmap: one matrix over layers and n/c conditions; "
            "lines: y=count, x=layer, color=c, linestyle=n; "
            "c_sweep: y=count, x=c, one subplot per selected layer, color=n."
        ),
    )
    p.add_argument(
        "--start_tokens",
        type=int,
        nargs="+",
        default=None,
        help="Optional filter: only plot these 1-based decode start tokens n, e.g. --start_tokens 64 1024.",
    )
    p.add_argument(
        "--segment_sizes",
        type=int,
        nargs="+",
        default=None,
        help="Optional filter: only plot these segment sizes c, e.g. --segment_sizes 3 5 8.",
    )
    p.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional layer filter. Especially useful for --plot_type c_sweep, "
            "e.g. --layers 0 12 24 36 47."
        ),
    )
    p.add_argument("--aggregate", choices=["mean", "median"], default="mean", help="Aggregate raw counts across prompts and segments.")
    p.add_argument("--layer_stride", type=int, default=4, help="Show every k-th layer tick on the x-axis.")
    p.add_argument("--dpi", type=int, default=220, help="Figure DPI.")
    p.add_argument("--fig_width", type=float, default=14.0, help="Figure width in inches.")
    p.add_argument("--fig_height", type=float, default=None, help="Figure height in inches. Default depends on number of rows.")
    p.add_argument("--annotate", action="store_true", help="Annotate heatmap cells with numeric values. Useful for small grids.")
    p.add_argument("--title", type=str, default=None, help="Optional custom title.")
    return p.parse_args()


def load_from_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_CSV_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    return normalize_raw_df(df)


def load_from_json(path: str | Path) -> pd.DataFrame:
    """
    Prefer raw per_prompt measurements so aggregation semantics match CSV.
    Falls back to JSON summary means if raw measurements are unavailable.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows: list[dict[str, Any]] = []
    for prompt_obj in data.get("per_prompt", []):
        prompt_index = int(prompt_obj.get("prompt_index", len(rows)))
        measurements = prompt_obj.get("measurements", {})
        for n_str, c_map in measurements.items():
            n = int(n_str)
            for c_str, c_obj in c_map.items():
                c = int(c_str)
                layers = c_obj.get("layers", {})
                for layer_str, layer_obj in layers.items():
                    layer_idx = int(layer_str)
                    for seg in layer_obj.get("segments", []):
                        rows.append(
                            {
                                "prompt_index": prompt_index,
                                "start_token_n_1based": n,
                                "segment_size_c": c,
                                "layer_idx": layer_idx,
                                "segment_index": int(seg["segment_index"]),
                                "unique_expert_count": int(seg["unique_expert_count"]),
                            }
                        )

    if rows:
        return normalize_raw_df(pd.DataFrame(rows))

    # Fallback: summary-only JSON. This cannot recover per-prompt/per-segment distribution,
    # but it is enough to plot mean values by n/c/layer.
    summary_rows: list[dict[str, Any]] = []
    summary = data.get("summary", {})
    for n_str, c_map in summary.items():
        n = int(n_str)
        for c_str, c_obj in c_map.items():
            c = int(c_str)
            per_layer = c_obj.get("per_layer", {})
            for layer_str, layer_obj in per_layer.items():
                stats = layer_obj.get("all_segments_all_prompts", {})
                mean_val = stats.get("mean")
                if mean_val is None:
                    continue
                summary_rows.append(
                    {
                        "prompt_index": -1,
                        "start_token_n_1based": n,
                        "segment_size_c": c,
                        "layer_idx": int(layer_str),
                        "segment_index": -1,
                        "unique_expert_count": float(mean_val),
                    }
                )
    if not summary_rows:
        raise ValueError("JSON does not contain usable per_prompt measurements or summary data.")
    return normalize_raw_df(pd.DataFrame(summary_rows))


def normalize_raw_df(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "prompt_index",
        "start_token_n_1based",
        "segment_size_c",
        "layer_idx",
        "segment_index",
        "unique_expert_count",
    ]
    df = df[keep].copy()
    for col in keep:
        df[col] = pd.to_numeric(df[col], errors="raise")
    df["start_token_n_1based"] = df["start_token_n_1based"].astype(int)
    df["segment_size_c"] = df["segment_size_c"].astype(int)
    df["layer_idx"] = df["layer_idx"].astype(int)
    df["segment_index"] = df["segment_index"].astype(int)
    return df


def load_data(args: argparse.Namespace) -> pd.DataFrame:
    if args.csv:
        return load_from_csv(args.csv)
    if args.json:
        return load_from_json(args.json)
    raise ValueError("Provide at least one of --csv or --json.")


def filter_raw_df(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Apply user-requested n/c/layer filters before aggregation."""
    out = df.copy()
    if args.start_tokens is not None:
        start_tokens = set(int(x) for x in args.start_tokens)
        out = out[out["start_token_n_1based"].isin(start_tokens)]
    if args.segment_sizes is not None:
        segment_sizes = set(int(x) for x in args.segment_sizes)
        out = out[out["segment_size_c"].isin(segment_sizes)]
    if args.layers is not None:
        layers = set(int(x) for x in args.layers)
        out = out[out["layer_idx"].isin(layers)]
    if out.empty:
        raise ValueError(
            "No rows left after filtering. Check --start_tokens, --segment_sizes, and --layers "
            "against the values in the CSV/JSON output."
        )
    return out


def aggregate_for_plot(df: pd.DataFrame, aggregate: str) -> pd.DataFrame:
    group_cols = ["start_token_n_1based", "segment_size_c", "layer_idx"]
    if aggregate == "mean":
        agg = df.groupby(group_cols, as_index=False)["unique_expert_count"].mean()
    elif aggregate == "median":
        agg = df.groupby(group_cols, as_index=False)["unique_expert_count"].median()
    else:
        raise ValueError(f"Unknown aggregate: {aggregate}")
    agg = agg.rename(columns={"unique_expert_count": f"unique_expert_count_{aggregate}"})

    spread = (
        df.groupby(group_cols)["unique_expert_count"]
        .agg(count="count", std="std", min="min", max="max")
        .reset_index()
    )
    out = agg.merge(spread, on=group_cols, how="left")
    out["condition"] = out.apply(
        lambda r: f"n={int(r['start_token_n_1based'])}, c={int(r['segment_size_c'])}",
        axis=1,
    )
    return out.sort_values(["start_token_n_1based", "segment_size_c", "layer_idx"])


def plot_heatmap(plot_df: pd.DataFrame, args: argparse.Namespace) -> None:
    value_col = f"unique_expert_count_{args.aggregate}"
    layers = sorted(plot_df["layer_idx"].unique())
    conditions = (
        plot_df[["start_token_n_1based", "segment_size_c", "condition"]]
        .drop_duplicates()
        .sort_values(["start_token_n_1based", "segment_size_c"])
    )
    condition_labels = conditions["condition"].tolist()

    matrix = np.full((len(condition_labels), len(layers)), np.nan, dtype=float)
    layer_to_x = {layer: i for i, layer in enumerate(layers)}
    cond_to_y = {cond: i for i, cond in enumerate(condition_labels)}

    for _, row in plot_df.iterrows():
        y = cond_to_y[row["condition"]]
        x = layer_to_x[int(row["layer_idx"])]
        matrix[y, x] = float(row[value_col])

    fig_height = args.fig_height if args.fig_height is not None else max(3.8, 0.55 * len(condition_labels) + 1.8)
    fig, ax = plt.subplots(figsize=(args.fig_width, fig_height))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest")

    title = args.title or f"Unique routed experts per segment by layer ({args.aggregate} across prompts and segments)"
    ax.set_title(title)
    ax.set_xlabel("MoE layer index")
    ax.set_ylabel("Start token n and segment size c")

    tick_positions = list(range(0, len(layers), max(1, args.layer_stride)))
    if (len(layers) - 1) not in tick_positions:
        tick_positions.append(len(layers) - 1)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(layers[i]) for i in tick_positions], rotation=0)
    ax.set_yticks(range(len(condition_labels)))
    ax.set_yticklabels(condition_labels)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"{args.aggregate} unique expert count")

    if args.annotate:
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                if np.isfinite(matrix[y, x]):
                    ax.text(x, y, f"{matrix[y, x]:.1f}", ha="center", va="center", fontsize=7)

    ax.set_xticks(np.arange(-0.5, len(layers), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(condition_labels), 1), minor=True)
    ax.grid(which="minor", linewidth=0.25, alpha=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def _style_maps(values: list[int], linestyles: bool = False) -> dict[int, Any]:
    """Return deterministic matplotlib style maps for integer values."""
    values = sorted(set(int(v) for v in values))
    if linestyles:
        styles = ["-", "--", "-.", ":"]
        return {v: styles[i % len(styles)] for i, v in enumerate(values)}

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not prop_cycle:
        prop_cycle = [f"C{i}" for i in range(10)]
    return {v: prop_cycle[i % len(prop_cycle)] for i, v in enumerate(values)}


def plot_lines(plot_df: pd.DataFrame, args: argparse.Namespace) -> None:
    """
    Existing layer-wise line plot, now with requested visual encoding:
      - x-axis: layer index
      - y-axis: unique expert count
      - color: segment size c
      - linestyle: start token n
    """
    value_col = f"unique_expert_count_{args.aggregate}"
    fig_height = args.fig_height if args.fig_height is not None else 6.2
    fig, ax = plt.subplots(figsize=(args.fig_width, fig_height))

    ns = sorted(plot_df["start_token_n_1based"].unique())
    cs = sorted(plot_df["segment_size_c"].unique())
    n_to_ls = _style_maps(ns, linestyles=True)
    c_to_color = _style_maps(cs, linestyles=False)

    for (n, c), sub in plot_df.groupby(["start_token_n_1based", "segment_size_c"], sort=True):
        sub = sub.sort_values("layer_idx")
        ax.plot(
            sub["layer_idx"],
            sub[value_col],
            marker="o",
            markersize=3,
            linewidth=1.5,
            linestyle=n_to_ls[int(n)],
            color=c_to_color[int(c)],
            label=f"n={int(n)}, c={int(c)}",
        )

    title = args.title or f"Unique routed experts per segment across layers ({args.aggregate} across prompts and segments)"
    ax.set_title(title)
    ax.set_xlabel("MoE layer index")
    ax.set_ylabel(f"{args.aggregate} unique expert count")
    ax.grid(True, linewidth=0.4, alpha=0.35)

    layers = sorted(plot_df["layer_idx"].unique())
    tick_positions = layers[:: max(1, args.layer_stride)]
    if layers[-1] not in tick_positions:
        tick_positions.append(layers[-1])
    ax.set_xticks(tick_positions)
    ax.legend(title="Condition (color=c, linestyle=n)", ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def plot_c_sweep(plot_df: pd.DataFrame, args: argparse.Namespace) -> None:
    """
    One figure for selected layers:
      - x-axis: segment size c
      - y-axis: unique expert count
      - color: start token n
      - one subplot per selected layer

    This is the clearest way to compare how increasing c changes the unique
    expert-set size, while keeping n color-coded consistently across layers.
    """
    value_col = f"unique_expert_count_{args.aggregate}"
    layers = sorted(plot_df["layer_idx"].unique())
    ns = sorted(plot_df["start_token_n_1based"].unique())
    cs = sorted(plot_df["segment_size_c"].unique())

    n_to_color = _style_maps(ns, linestyles=False)

    if len(layers) == 1:
        ncols, nrows = 1, 1
    else:
        ncols = min(3, len(layers))
        nrows = int(np.ceil(len(layers) / ncols))

    default_height = max(3.8, 3.0 * nrows)
    fig_height = args.fig_height if args.fig_height is not None else default_height
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(args.fig_width, fig_height), squeeze=False)
    axes_flat = axes.ravel()

    for ax_idx, layer in enumerate(layers):
        ax = axes_flat[ax_idx]
        layer_df = plot_df[plot_df["layer_idx"] == layer]

        for n in ns:
            sub = layer_df[layer_df["start_token_n_1based"] == n].sort_values("segment_size_c")
            if sub.empty:
                continue
            ax.plot(
                sub["segment_size_c"],
                sub[value_col],
                marker="o",
                markersize=4,
                linewidth=1.7,
                color=n_to_color[int(n)],
                label=f"n={int(n)}",
            )

        ax.set_title(f"Layer {int(layer)}")
        ax.set_xlabel("Segment size c")
        ax.set_ylabel(f"{args.aggregate} unique expert count")
        ax.set_xticks(cs)
        ax.grid(True, linewidth=0.4, alpha=0.35)

    for j in range(len(layers), len(axes_flat)):
        axes_flat[j].axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="Start token n", loc="upper center", ncols=min(len(ns), 6), frameon=False)

    title = args.title or f"Unique routed experts vs segment size c by selected layer ({args.aggregate} across prompts and segments)"
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94 if handles else 0.97))
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    raw_df = load_data(args)
    raw_df = filter_raw_df(raw_df, args)
    plot_df = aggregate_for_plot(raw_df, args.aggregate)

    if args.summary_out:
        plot_df.to_csv(args.summary_out, index=False)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.plot_type == "heatmap":
        plot_heatmap(plot_df, args)
    elif args.plot_type == "lines":
        plot_lines(plot_df, args)
    elif args.plot_type == "c_sweep":
        plot_c_sweep(plot_df, args)
    else:
        raise ValueError(f"Unsupported plot_type: {args.plot_type}")

    print(f"Loaded {len(raw_df)} raw rows")
    print(f"Plotted {plot_df['condition'].nunique()} n/c conditions across {plot_df['layer_idx'].nunique()} layers")
    print(f"Wrote figure: {args.out}")
    if args.summary_out:
        print(f"Wrote aggregated table: {args.summary_out}")


if __name__ == "__main__":
    main()
