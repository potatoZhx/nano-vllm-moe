#!/usr/bin/env python3
"""
Plot MoE segment unique expert counts for one start token n and multiple
segment sizes c.

Input is the output of `test_moe_segment_unique_expert_counts.py`:
  - Prefer --csv segment_unique_expert_counts.csv
  - Or use --json segment_unique_expert_counts.json

Visualization:
  - x-axis: MoE layer index
  - y-axis: aggregated unique expert count
  - one solid line per segment size c
  - for each c, draw a same-color dashed horizontal reference line y = c * top_k
    where top_k defaults to 8 for Qwen3 MoE top-8 routing

Examples:
    python plot_moe_segment_unique_expert_counts_v3.py \
        --csv /data2/group_谈海生/mumura/nano_moe/motivaion/segment_unique_expert_counts_optimized_avg1024.csv \
        --avg \
        --segment_sizes 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 \
        --out res/unique_count_by_layer_avg1024.png

  python plot_moe_segment_unique_expert_counts_v3.py \
      --json segment_unique_expert_counts.json \
      --start_tokens 1024 \
      --segment_sizes 3 5 8 \
      --top_k 8 \
      --aggregate mean \
      --summary_out unique_count_plot_summary_n1024.csv \
      --out unique_count_by_layer_starts.pdf

  python plot_moe_segment_unique_expert_counts_v3.py \
    --json segment_unique_expert_counts.json \
    --start_tokens 64 1024 4096 \
    --segment_sizes 3 8 16 \
    --out unique_count_by_starts.png

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
        description=(
            "Plot layer-wise unique routed expert counts for one or multiple "
            "start tokens n and multiple segment sizes c. Same-c curves share "
            "a color; multi-start curves use different line styles."
        )
    )
    p.add_argument("--csv", type=str, default=None, help="CSV output from the experiment script.")
    p.add_argument("--json", type=str, default=None, help="JSON output from the experiment script. Used if --csv is absent.")
    p.add_argument("--out", type=str, default="unique_count_by_layer.png", help="Output figure path: .png/.pdf/.svg.")
    p.add_argument("--summary_out", type=str, default=None, help="Optional path to save the aggregated plotting table as CSV.")

    p.add_argument(
        "--start_token",
        type=int,
        default=None,
        help=(
            "Exactly one 1-based decode start token n, e.g. --start_token 64. "
            "Required unless --start_tokens or --avg is enabled."
        ),
    )
    p.add_argument(
        "--start_tokens",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Plot multiple 1-based start tokens together. Curves with the "
            "same segment size use the same color, while start tokens use "
            "different line styles."
        ),
    )
    p.add_argument(
        "--avg",
        action="store_true",
        help=(
            "Average across all available start tokens. When enabled, "
            "--start_token and --start_tokens are ignored."
        ),
    )
    p.add_argument(
        "--segment_sizes",
        type=int,
        nargs="+",
        required=True,
        help="Segment size list c, e.g. --segment_sizes 3 5 8.",
    )
    p.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Optional layer filter, e.g. --layers 0 12 24 36 47. Default: all layers.",
    )

    p.add_argument("--aggregate", choices=["mean", "median"], default="mean", help="Aggregate raw counts across prompts and segments.")
    p.add_argument("--top_k", type=int, default=8, help="Experts selected per token. Dashed baseline is y = c * top_k. Default: 8.")
    p.add_argument("--layer_stride", type=int, default=4, help="Show every k-th layer tick on the x-axis.")
    p.add_argument("--dpi", type=int, default=220, help="Figure DPI.")
    p.add_argument("--fig_width", type=float, default=14.0, help="Figure width in inches.")
    p.add_argument("--fig_height", type=float, default=6.2, help="Figure height in inches.")
    p.add_argument("--title", type=str, default=None, help="Optional custom title.")
    p.add_argument(
        "--show_baseline_labels",
        action="store_true",
        help="Add text labels near the dashed y=c*top_k reference lines.",
    )
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
    df["prompt_index"] = df["prompt_index"].astype(int)
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
    requested_c = [int(x) for x in args.segment_sizes]
    if args.avg:
        out = df[df["segment_size_c"].isin(requested_c)].copy()
    elif args.start_tokens is not None:
        if args.start_token is not None:
            raise ValueError(
                "Use either --start_token or --start_tokens, not both."
            )
        requested_n = [int(x) for x in args.start_tokens]
        out = df[
            (df["start_token_n_1based"].isin(requested_n))
            & (df["segment_size_c"].isin(requested_c))
        ].copy()
    else:
        if args.start_token is None:
            raise ValueError(
                "Provide --start_token, --start_tokens, or --avg."
            )
        out = df[
            (df["start_token_n_1based"] == int(args.start_token))
            & (df["segment_size_c"].isin(requested_c))
        ].copy()

    if args.layers is not None:
        requested_layers = set(int(x) for x in args.layers)
        out = out[out["layer_idx"].isin(requested_layers)]

    if out.empty:
        available_n = sorted(df["start_token_n_1based"].unique().tolist())
        available_c = sorted(df["segment_size_c"].unique().tolist())
        available_layers = sorted(df["layer_idx"].unique().tolist())
        if args.avg:
            requested_start = "all (--avg)"
        elif args.start_tokens is not None:
            requested_start = args.start_tokens
        else:
            requested_start = args.start_token
        raise ValueError(
            "No rows left after filtering.\n"
            f"Requested start_token={requested_start}, segment_sizes={requested_c}, layers={args.layers}.\n"
            f"Available start tokens: {available_n}\n"
            f"Available segment sizes: {available_c}\n"
            f"Available layer range: {available_layers[:5]} ... {available_layers[-5:]}"
        )

    missing_c = sorted(set(requested_c) - set(out["segment_size_c"].unique()))
    if missing_c:
        print(f"[warning] Requested segment sizes not found after filtering: {missing_c}")

    if not args.avg and args.start_tokens is not None:
        available_requested_n = set(
            out["start_token_n_1based"].unique().tolist()
        )
        missing_n = sorted(set(args.start_tokens) - available_requested_n)
        if missing_n:
            print(
                "[warning] Requested start tokens not found after filtering: "
                f"{missing_n}"
            )

    return out


def aggregate_for_plot(
    df: pd.DataFrame,
    aggregate: str,
    average_start_tokens: bool = False,
) -> pd.DataFrame:
    if average_start_tokens:
        group_cols = ["segment_size_c", "layer_idx"]
    else:
        group_cols = [
            "start_token_n_1based",
            "segment_size_c",
            "layer_idx",
        ]
    if aggregate == "mean":
        center = df.groupby(group_cols, as_index=False)["unique_expert_count"].mean()
    elif aggregate == "median":
        center = df.groupby(group_cols, as_index=False)["unique_expert_count"].median()
    else:
        raise ValueError(f"Unknown aggregate: {aggregate}")

    center = center.rename(columns={"unique_expert_count": f"unique_expert_count_{aggregate}"})
    spread = (
        df.groupby(group_cols)["unique_expert_count"]
        .agg(count="count", std="std", min="min", max="max")
        .reset_index()
    )
    out = center.merge(spread, on=group_cols, how="left")
    if average_start_tokens:
        out.insert(0, "start_token_n_1based", -1)
    return out.sort_values(["segment_size_c", "layer_idx"])


def _color_map(values: list[int]) -> dict[int, str]:
    values = sorted(set(int(v) for v in values))
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not prop_cycle:
        prop_cycle = [f"C{i}" for i in range(10)]
    return {v: prop_cycle[i % len(prop_cycle)] for i, v in enumerate(values)}


def _linestyle_map(values: list[int]) -> dict[int, Any]:
    values = sorted(set(int(v) for v in values))
    standard_styles: list[Any] = ["-", "--", "-.", ":"]
    result: dict[int, Any] = {}
    for index, value in enumerate(values):
        if index < len(standard_styles):
            result[value] = standard_styles[index]
        else:
            result[value] = (0, (index + 2, 1, 1, 1))
    return result


def plot_layer_lines_one_n(plot_df: pd.DataFrame, args: argparse.Namespace) -> None:
    value_col = f"unique_expert_count_{args.aggregate}"
    n_values = sorted(plot_df["start_token_n_1based"].unique())
    multi_start_mode = not args.avg and args.start_tokens is not None
    if not args.avg and not multi_start_mode and len(n_values) != 1:
        raise ValueError(f"Expected exactly one start token after filtering, got {n_values}")
    n = int(n_values[0]) if n_values else None

    cs = sorted(plot_df["segment_size_c"].unique())
    c_to_color = _color_map(cs)
    n_to_linestyle = _linestyle_map(n_values)

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))

    max_measured = float(plot_df[value_col].max()) if not plot_df.empty else 0.0
    max_baseline = max(min(int(c) * int(args.top_k), 128) for c in cs)

    for c in cs:
        color = c_to_color[int(c)]
        c_sub = plot_df[plot_df["segment_size_c"] == c]
        for start_token in n_values:
            sub = c_sub[
                c_sub["start_token_n_1based"] == start_token
            ].sort_values("layer_idx")
            if sub.empty:
                continue
            if multi_start_mode:
                label = f"c={int(c)}, n={int(start_token)}"
                linestyle = n_to_linestyle[int(start_token)]
            else:
                label = f"c={int(c)} measured"
                linestyle = "-"
            ax.plot(
                sub["layer_idx"],
                sub[value_col],
                marker="o",
                markersize=3,
                linewidth=1.7,
                linestyle=linestyle,
                color=color,
                label=label,
            )

        baseline = min(int(c) * int(args.top_k), 128)
        ax.axhline(
            baseline,
            linestyle=(0, (2, 2)),
            linewidth=1.2,
            color=color,
            alpha=0.8,
            label=f"c={int(c)} × top-{int(args.top_k)} = {baseline}",
        )

        if args.show_baseline_labels:
            x_right = (
                float(c_sub["layer_idx"].max())
                if not c_sub.empty
                else 0.0
            )
            ax.text(
                x_right,
                baseline,
                f"  c={int(c)} baseline {baseline}",
                color=color,
                va="center",
                ha="left",
                fontsize=8,
            )

    if args.title:
        title = args.title
    elif args.avg:
        title = (
            "Average unique routed expert count by layer "
            f"({args.aggregate} across start tokens, prompts, and segments)"
        )
    elif multi_start_mode:
        title = (
            "Unique routed expert count by layer for "
            f"n={n_values} ({args.aggregate} across prompts and segments)"
        )
    else:
        title = (
            f"Unique routed expert count by layer at n={n} "
            f"({args.aggregate} across prompts and segments)"
        )
    ax.set_title(title)
    ax.set_xlabel("MoE layer index")
    ax.set_ylabel(f"{args.aggregate} unique expert count")
    ax.grid(True, linewidth=0.4, alpha=0.35)

    layers = sorted(plot_df["layer_idx"].unique())
    tick_positions = layers[:: max(1, args.layer_stride)]
    if layers and layers[-1] not in tick_positions:
        tick_positions.append(layers[-1])
    ax.set_xticks(tick_positions)

    # Leave enough vertical room for dashed reference lines if measured counts are below c*top_k.
    y_top = max(max_measured, float(max_baseline)) * 1.08
    ax.set_ylim(bottom=0, top=y_top)

    legend_title = (
        "Segment size c and start token n"
        if multi_start_mode
        else "Segment size c"
    )
    ax.legend(title=legend_title, ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    raw_df = load_data(args)
    raw_df = filter_raw_df(raw_df, args)
    plot_df = aggregate_for_plot(
        raw_df,
        args.aggregate,
        average_start_tokens=args.avg,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        plot_df.to_csv(summary_path, index=False)

    plot_layer_lines_one_n(plot_df, args)

    print(f"Loaded {len(raw_df)} raw rows")
    if args.avg:
        available_starts = sorted(
            raw_df["start_token_n_1based"].unique().tolist()
        )
        print(
            "Start token n: ignored (--avg); averaged values: "
            f"{available_starts}"
        )
    elif args.start_tokens is not None:
        plotted_starts = sorted(
            plot_df["start_token_n_1based"].unique().tolist()
        )
        print(f"Start tokens n: {plotted_starts}")
    else:
        print(f"Start token n: {args.start_token}")
    print(f"Segment sizes c: {sorted(plot_df['segment_size_c'].unique().tolist())}")
    print(f"Layers plotted: {plot_df['layer_idx'].nunique()}")
    print(f"Dashed reference lines: y = c * top_k, top_k={args.top_k}")
    print(f"Wrote figure: {args.out}")
    if args.summary_out:
        print(f"Wrote aggregated table: {args.summary_out}")


if __name__ == "__main__":
    main()
