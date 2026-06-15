"""Analyze draft original-route accuracy against target model routes.

Each observation is one decode step at one MoE layer:

    accuracy = |draft original expert ids intersect target expert ids|
               / |target expert ids|

By default, all target original-route experts are evaluated. With ``--top-p``,
only the ``p`` target experts with the highest original routing weights are
used as the target set, while the complete draft original route is retained.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-layer draft original-route accuracy from an "
            "acceptance_summary JSONL file."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to acceptance_summary_*.jsonl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSON path. Defaults to <input_stem>_route_accuracy"
            "[_topP].json next to the input file."
        ),
    )
    parser.add_argument(
        "--top-p",
        type=int,
        default=None,
        metavar="P",
        help=(
            "Evaluate only the P highest-weight target route experts. "
            "Omit this option to evaluate the full target route."
        ),
    )
    return parser.parse_args()


def _flatten_single_token(values: Any, field_name: str) -> list[Any]:
    if not isinstance(values, list):
        raise ValueError(f"{field_name} must be a list")
    if len(values) == 1 and isinstance(values[0], list):
        values = values[0]
    if not isinstance(values, list) or any(isinstance(value, list) for value in values):
        raise ValueError(f"{field_name} must contain one token's 1-D values")
    return values


def _expert_ids(layer: dict[str, Any], field_name: str = "original_ids") -> list[int]:
    values = _flatten_single_token(layer.get(field_name), field_name)
    try:
        return [int(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} contains a non-integer expert id") from exc


def select_target_ids(layer: dict[str, Any], top_p: int | None) -> set[int]:
    ids = _expert_ids(layer)
    if top_p is None:
        selected = ids
    else:
        weights = _flatten_single_token(
            layer.get("original_weights"), "original_weights"
        )
        if len(weights) != len(ids):
            raise ValueError(
                "target original_ids and original_weights have different lengths"
            )
        try:
            ranked = sorted(
                zip(ids, (float(weight) for weight in weights)),
                key=lambda item: item[1],
                reverse=True,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("original_weights contains a non-numeric value") from exc
        if any(not math.isfinite(weight) for _, weight in ranked):
            raise ValueError("original_weights contains a non-finite value")
        selected = [expert_id for expert_id, _ in ranked[:top_p]]

    target_ids = set(selected)
    if not target_ids:
        raise ValueError("selected target route is empty")
    return target_ids


def route_accuracy(
    draft_layer: dict[str, Any],
    target_layer: dict[str, Any],
    top_p: int | None,
) -> float:
    draft_ids = set(_expert_ids(draft_layer))
    target_ids = select_target_ids(target_layer, top_p)
    return len(draft_ids & target_ids) / len(target_ids)


def _layers_by_index(
    layers: Any,
    field_name: str,
) -> dict[int, dict[str, Any]]:
    if not isinstance(layers, list):
        raise ValueError(f"{field_name} must be a list")

    result: dict[int, dict[str, Any]] = {}
    for layer in layers:
        if not isinstance(layer, dict) or "layer_idx" not in layer:
            raise ValueError(f"{field_name} contains an invalid layer record")
        layer_idx = int(layer["layer_idx"])
        if layer_idx in result:
            raise ValueError(f"{field_name} contains duplicate layer_idx={layer_idx}")
        result[layer_idx] = layer
    return result


def iter_step_accuracies(
    record: dict[str, Any],
    top_p: int | None,
) -> Iterable[tuple[int, float]]:
    steps = record.get("steps")
    if not isinstance(steps, list):
        raise ValueError("record.steps must be a list")

    for step_position, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"steps[{step_position}] must be an object")
        # The collector names the draft route "router".
        draft_layers = _layers_by_index(step.get("router"), "step.router")
        target_layers = _layers_by_index(
            step.get("target_router"), "step.target_router"
        )
        if draft_layers.keys() != target_layers.keys():
            draft_only = sorted(draft_layers.keys() - target_layers.keys())
            target_only = sorted(target_layers.keys() - draft_layers.keys())
            raise ValueError(
                "draft/target layer mismatch: "
                f"draft_only={draft_only}, target_only={target_only}"
            )
        for layer_idx in sorted(draft_layers):
            yield layer_idx, route_accuracy(
                draft_layers[layer_idx],
                target_layers[layer_idx],
                top_p,
            )


def summarize(values: list[float]) -> dict[str, int | float]:
    if not values:
        raise ValueError("cannot summarize an empty value list")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "count": len(values),
        "mean": mean,
        "variance": variance,
        "max": max(values),
        "min": min(values),
    }


def analyze_file(input_path: Path, top_p: int | None) -> dict[str, Any]:
    if top_p is not None and top_p <= 0:
        raise ValueError(f"--top-p must be a positive integer, got {top_p}")

    per_layer: dict[int, list[float]] = defaultdict(list)
    all_values: list[float] = []
    record_count = 0
    step_count = 0

    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("JSONL record must be an object")
                step_count += len(record.get("steps", []))
                for layer_idx, accuracy in iter_step_accuracies(record, top_p):
                    per_layer[layer_idx].append(accuracy)
                    all_values.append(accuracy)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"{input_path}:{line_number}: {exc}") from exc
            record_count += 1

    if not all_values:
        raise ValueError(f"No route observations found in {input_path}")

    return {
        "input_file": str(input_path.resolve()),
        "top_p": top_p,
        "target_selection": (
            "all target original_ids"
            if top_p is None
            else f"top {top_p} target original_ids by original_weights"
        ),
        "draft_selection": "all draft original_ids",
        "accuracy_formula": (
            "|draft_original_ids intersection selected_target_ids| "
            "/ |selected_target_ids|"
        ),
        "variance": "population variance (ddof=0)",
        "record_count": record_count,
        "step_count": step_count,
        "observation_count": len(all_values),
        "overall": summarize(all_values),
        "layers": [
            {"layer_idx": layer_idx, **summarize(per_layer[layer_idx])}
            for layer_idx in sorted(per_layer)
        ],
    }


def default_output_path(input_path: Path, top_p: int | None) -> Path:
    suffix = "_route_accuracy" if top_p is None else f"_route_accuracy_top{top_p}"
    return input_path.with_name(f"{input_path.stem}{suffix}.json")


def main() -> None:
    args = parse_args()
    result = analyze_file(args.input, args.top_p)
    output_path = args.output or default_output_path(args.input, args.top_p)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved route accuracy statistics to {output_path}")


if __name__ == "__main__":
    main()
