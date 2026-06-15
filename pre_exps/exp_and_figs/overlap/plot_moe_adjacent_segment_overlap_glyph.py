#!/usr/bin/env python3
"""
Plot adjacent-segment overlap results as a refined blue glyph heatmap.

Input must be JSON produced by test_moe_adjacent_segment_overlap.py.
Following refined_glyph_heatmap_blue:

  - cell color: mean hit ratio
  - vertical whisker: min to max
  - hollow box: Q25 to Q75
  - orange dot: mean
  - cell text: mean

By default, the axes use the same selections as the reference script:
  layers = 0 7 14 21 28 35 42 47
  segment sizes = 1 3 5 7 9

Example:
  python plot_moe_adjacent_segment_overlap_glyph.py \
      --input adjacent_segment_overlap_results.json \
      --output res/adjacent_segment_overlap_glyph.png

Custom axis selections:
  python plot_moe_adjacent_segment_overlap_glyph.py \
      --input adjacent_segment_overlap_results.json \
      --layers 0 8 16 24 32 40 47 \
      --segment_sizes 1 2 4 8 16 \
      --output res/adjacent_segment_overlap_glyph.svg
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


DEFAULT_LAYERS = [0, 7, 14, 21, 28, 35, 42, 47]
DEFAULT_SEGMENT_SIZES = [1, 3, 5, 7, 9]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot adjacent-segment hit ratios as a blue heatmap with "
            "min/max, quartile, and mean glyphs."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="JSON output from test_moe_adjacent_segment_overlap.py.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="moe_adjacent_segment_overlap_glyph.png",
        help="Output image path (.png, .pdf, or .svg).",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=DEFAULT_LAYERS,
        help=(
            "Layer ids and display order. Default: "
            + " ".join(map(str, DEFAULT_LAYERS))
        ),
    )
    parser.add_argument(
        "--segment_sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SEGMENT_SIZES,
        help=(
            "Segment sizes and display order. Default: "
            + " ".join(map(str, DEFAULT_SEGMENT_SIZES))
        ),
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Adjacent-segment expert overlap",
        help="Figure title.",
    )
    parser.add_argument(
        "--fig_width",
        type=float,
        default=10.2,
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--fig_height",
        type=float,
        default=6.2,
        help="Figure height in inches.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--cmap",
        type=str,
        default="Blues",
        help="Matplotlib colormap name.",
    )
    parser.add_argument("--vmin", type=float, default=0.0)
    parser.add_argument("--vmax", type=float, default=1.0)
    parser.add_argument(
        "--no_legend",
        action="store_true",
        help="Hide the cell-glyph legend panel.",
    )
    return parser.parse_args()


def load_result(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)
    if not isinstance(data.get("per_prompt"), list):
        raise ValueError("Input JSON is missing the 'per_prompt' list")
    if not isinstance(data.get("summary"), dict):
        raise ValueError("Input JSON is missing the 'summary' object")
    return data


def validate_selection(
    data: dict[str, Any],
    layers: list[int],
    segment_sizes: list[int],
) -> tuple[list[int], list[int]]:
    selected_layers = list(dict.fromkeys(layers))
    selected_sizes = list(dict.fromkeys(segment_sizes))
    if not selected_layers:
        raise ValueError("--layers must not be empty")
    if not selected_sizes:
        raise ValueError("--segment_sizes must not be empty")

    available_sizes = sorted(
        int(key) for key in data["summary"] if str(key).isdigit()
    )
    missing_sizes = [
        size for size in selected_sizes if size not in set(available_sizes)
    ]
    if missing_sizes:
        raise ValueError(
            f"Requested segment sizes are unavailable: {missing_sizes}; "
            f"available={available_sizes}"
        )

    available_layers: set[int] = set()
    for size in selected_sizes:
        per_layer = data["summary"][str(size)].get("per_layer", {})
        available_layers.update(
            int(key) for key in per_layer if str(key).isdigit()
        )
    missing_layers = [
        layer for layer in selected_layers if layer not in available_layers
    ]
    if missing_layers:
        raise ValueError(
            f"Requested layers are unavailable: {missing_layers}; "
            f"available={sorted(available_layers)}"
        )
    return selected_layers, selected_sizes


def collect_hit_ratios(
    data: dict[str, Any],
    layers: list[int],
    segment_sizes: list[int],
) -> dict[tuple[int, int], list[float]]:
    selected_layers = set(layers)
    selected_sizes = set(segment_sizes)
    values: dict[tuple[int, int], list[float]] = defaultdict(list)

    for prompt in data["per_prompt"]:
        measurements = prompt.get("measurements", {})
        for size_key, measurement in measurements.items():
            try:
                segment_size = int(size_key)
            except (TypeError, ValueError):
                continue
            if segment_size not in selected_sizes:
                continue
            for layer_key, layer_data in measurement.get(
                "per_layer", {}
            ).items():
                try:
                    layer = int(layer_key)
                except (TypeError, ValueError):
                    continue
                if layer not in selected_layers:
                    continue
                for pair in layer_data.get("adjacent_pairs", []):
                    hit_ratio = pair.get("hit_ratio")
                    if hit_ratio is not None:
                        values[(layer, segment_size)].append(
                            float(hit_ratio)
                        )
    return values


def build_statistics(
    values: dict[tuple[int, int], list[float]],
    layers: list[int],
    segment_sizes: list[int],
) -> dict[str, np.ndarray]:
    shape = (len(layers), len(segment_sizes))
    statistics = {
        name: np.full(shape, np.nan, dtype=np.float64)
        for name in ("mean", "q25", "q75", "min", "max")
    }

    for row, layer in enumerate(layers):
        for column, segment_size in enumerate(segment_sizes):
            samples = values.get((layer, segment_size), [])
            if not samples:
                continue
            array = np.asarray(samples, dtype=np.float64)
            statistics["mean"][row, column] = np.mean(array)
            statistics["q25"][row, column] = np.quantile(array, 0.25)
            statistics["q75"][row, column] = np.quantile(array, 0.75)
            statistics["min"][row, column] = np.min(array)
            statistics["max"][row, column] = np.max(array)

    if np.isnan(statistics["mean"]).all():
        raise ValueError(
            "No adjacent-pair hit ratios found for the selected axes"
        )
    return statistics


def y_in_cell(row: int, value: float) -> float:
    return row + 0.42 - 0.84 * value


def add_cell_grid(
    axis: plt.Axes,
    num_layers: int,
    num_segment_sizes: int,
) -> None:
    axis.set_xticks(
        np.arange(-0.5, num_segment_sizes, 1),
        minor=True,
    )
    axis.set_yticks(np.arange(-0.5, num_layers, 1), minor=True)
    axis.grid(which="minor", linewidth=0.7, color="#bfc7d5")
    axis.tick_params(which="minor", bottom=False, left=False)


def add_cell_glyphs(
    axis: plt.Axes,
    statistics: dict[str, np.ndarray],
) -> None:
    mean = statistics["mean"]
    for row in range(mean.shape[0]):
        for column in range(mean.shape[1]):
            if np.isnan(mean[row, column]):
                continue
            y_min = y_in_cell(row, statistics["min"][row, column])
            y_max = y_in_cell(row, statistics["max"][row, column])
            y_q25 = y_in_cell(row, statistics["q25"][row, column])
            y_q75 = y_in_cell(row, statistics["q75"][row, column])
            y_mean = y_in_cell(row, mean[row, column])

            axis.plot(
                [column, column],
                [y_min, y_max],
                color="black",
                linewidth=1.4,
                zorder=3,
            )
            axis.add_patch(
                Rectangle(
                    (column - 0.17, min(y_q25, y_q75)),
                    0.34,
                    abs(y_q75 - y_q25),
                    fill=False,
                    edgecolor="black",
                    linewidth=1.25,
                    zorder=4,
                )
            )
            axis.plot(
                column,
                y_mean,
                marker="o",
                markersize=4.2,
                color="#d95f02",
                zorder=5,
            )
            axis.text(
                column + 0.26,
                row + 0.33,
                f"{mean[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=7.2,
                color="#0b2545",
            )


def add_glyph_legend(axis: plt.Axes) -> None:
    axis.set_title("Cell glyph legend", pad=10)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    cell_x, cell_y, cell_width, cell_height = 0.17, 0.14, 0.42, 0.74
    axis.add_patch(
        Rectangle(
            (cell_x, cell_y),
            cell_width,
            cell_height,
            facecolor="#b9d8f0",
            edgecolor="#7aa6c2",
            linewidth=1.0,
        )
    )

    example_values = {
        "min": 0.18,
        "Q25": 0.36,
        "mean": 0.55,
        "Q75": 0.71,
        "max": 0.89,
    }

    def legend_y(value: float) -> float:
        return cell_y + cell_height * value

    x_mid = cell_x + cell_width * 0.50
    box_width = cell_width * 0.48
    coordinates = {
        name: legend_y(value)
        for name, value in example_values.items()
    }
    axis.plot(
        [x_mid, x_mid],
        [coordinates["min"], coordinates["max"]],
        color="black",
        linewidth=1.5,
    )
    axis.add_patch(
        Rectangle(
            (x_mid - box_width / 2, coordinates["Q25"]),
            box_width,
            coordinates["Q75"] - coordinates["Q25"],
            fill=False,
            edgecolor="black",
            linewidth=1.3,
        )
    )
    axis.plot(
        x_mid,
        coordinates["mean"],
        marker="o",
        markersize=5,
        color="#d95f02",
    )

    label_x = 0.72
    for label in ("max", "Q75", "mean", "Q25", "min"):
        y = coordinates[label]
        axis.plot(
            [x_mid + box_width / 2 + 0.03, label_x - 0.03],
            [y, y],
            color="black",
            linewidth=0.8,
        )
        axis.text(label_x, y, label, va="center", fontsize=9)

    axis.text(
        cell_x + cell_width / 2,
        cell_y - 0.05,
        "0",
        ha="center",
        fontsize=9,
    )
    axis.text(
        cell_x + cell_width / 2,
        cell_y + cell_height + 0.03,
        "1",
        ha="center",
        fontsize=9,
    )
    axis.text(
        cell_x + cell_width / 2,
        0.05,
        "vertical scale within each cell",
        ha="center",
        fontsize=8.8,
    )
    axis.text(
        0.72,
        0.20,
        "cell color = mean",
        fontsize=9,
        color="#0b2545",
    )


def plot(
    data: dict[str, Any],
    layers: list[int],
    segment_sizes: list[int],
    statistics: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> None:
    if args.vmax <= args.vmin:
        raise ValueError("--vmax must be greater than --vmin")

    figure = plt.figure(
        figsize=(args.fig_width, args.fig_height),
        dpi=args.dpi,
    )
    if args.no_legend:
        grid = figure.add_gridspec(1, 1)
        axis = figure.add_subplot(grid[0, 0])
        legend_axis = None
    else:
        grid = figure.add_gridspec(
            1,
            2,
            width_ratios=[6.4, 2.4],
            wspace=0.18,
        )
        axis = figure.add_subplot(grid[0, 0])
        legend_axis = figure.add_subplot(grid[0, 1])

    image = axis.imshow(
        statistics["mean"],
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
    )
    axis.set_title(args.title, pad=12)
    axis.set_xlabel("Segment size c")
    axis.set_ylabel("MoE layer")
    axis.set_xticks(np.arange(len(segment_sizes)))
    axis.set_xticklabels(segment_sizes)
    axis.set_yticks(np.arange(len(layers)))
    axis.set_yticklabels(layers)
    axis.set_xlim(-0.5, len(segment_sizes) - 0.5)
    axis.set_ylim(-0.5, len(layers) - 0.5)
    axis.invert_yaxis()
    axis.set_aspect("equal")
    add_cell_grid(axis, len(layers), len(segment_sizes))
    add_cell_glyphs(axis, statistics)

    colorbar = figure.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
    )
    colorbar.set_label("Mean hit ratio")

    if legend_axis is not None:
        add_glyph_legend(legend_axis)

    figure.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.10,
        top=0.91,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    print(f"[*] Glyph heatmap saved to: {output_path}")
    print(
        f"[*] Axes: {len(layers)} layers x "
        f"{len(segment_sizes)} segment sizes"
    )


def main() -> None:
    args = parse_args()
    data = load_result(args.input)
    layers, segment_sizes = validate_selection(
        data=data,
        layers=args.layers,
        segment_sizes=args.segment_sizes,
    )
    values = collect_hit_ratios(
        data=data,
        layers=layers,
        segment_sizes=segment_sizes,
    )
    statistics = build_statistics(
        values=values,
        layers=layers,
        segment_sizes=segment_sizes,
    )
    plot(
        data=data,
        layers=layers,
        segment_sizes=segment_sizes,
        statistics=statistics,
        args=args,
    )


if __name__ == "__main__":
    main()
