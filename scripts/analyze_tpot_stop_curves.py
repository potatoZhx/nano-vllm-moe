#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from statistics import mean
from pathlib import Path

from nanovllm.engine.speculative.acceptance_calibration import (
    AcceptanceAlphaCalibration,
)


def _paths(values: list[str]) -> list[Path]:
    paths: set[Path] = set()
    for value in values:
        path = Path(value)
        if path.is_dir():
            paths.update(path.rglob("sample*.json"))
        else:
            paths.update(Path(item) for item in glob.glob(value, recursive=True))
    return sorted(path.resolve() for path in paths if path.is_file())


def _step_curves(
    trace: dict,
    alpha_calibration: AcceptanceAlphaCalibration | None = None,
) -> list[dict[str, object]]:
    predictions = trace.get("verify_cost_predictions", [])
    draft_ms = trace.get("draft_call_ms", [])
    sequences = trace.get("sequences", [])
    if not isinstance(predictions, list) or not isinstance(draft_ms, list):
        return []
    if not isinstance(sequences, list) or not sequences:
        return []
    if alpha_calibration is None:
        alpha_rows = [
            row.get("calibrated_alpha", row.get("predicted_alpha", []))
            for row in sequences
        ]
    else:
        alpha_rows = [
            [
                alpha_calibration.calibrate(value)
                for value in row.get("predicted_alpha", [])
            ]
            for row in sequences
        ]
    draft_length = min(len(draft_ms), *(len(row) for row in alpha_rows))
    if draft_length <= 0:
        return []
    predictions_by_len: dict[int, dict] = {}
    has_explicit_lengths = all(
        isinstance(row, dict) and "verify_cost_candidate_len" in row
        for row in predictions
    )
    if has_explicit_lengths:
        predictions_by_len = {
            int(row["verify_cost_candidate_len"]): row
            for row in predictions
        }
    else:
        predictions_by_len = {
            index + 1: row
            for index, row in enumerate(predictions[:draft_length])
            if isinstance(row, dict)
        }
    candidate_lengths = sorted(
        candidate_len
        for candidate_len in predictions_by_len
        if 0 <= candidate_len <= draft_length
    )
    if not candidate_lengths:
        return []
    cumulative = [1.0] * len(alpha_rows)
    accepted_sum = 0.0
    cumulative_draft_ms = 0.0
    curve = []
    next_candidate_index = 0
    for candidate_len in range(draft_length + 1):
        if candidate_len > 0:
            draft_index = candidate_len - 1
            cumulative_draft_ms += float(draft_ms[draft_index])
            for seq_index, row in enumerate(alpha_rows):
                cumulative[seq_index] *= float(row[draft_index])
                accepted_sum += cumulative[seq_index]
        if (
            next_candidate_index >= len(candidate_lengths)
            or candidate_lengths[next_candidate_index] != candidate_len
        ):
            continue
        prediction = predictions_by_len[candidate_len]
        next_candidate_index += 1
        verify_ms = float(prediction["verify_cost_prediction_ms"])
        tpot_ms = (cumulative_draft_ms + verify_ms) / (
            float(len(alpha_rows)) + accepted_sum
        )
        curve.append(
            {
                "draft_len": candidate_len,
                "expected_tpot_ms": tpot_ms,
                "cumulative_draft_ms": cumulative_draft_ms,
                "verify_ms": verify_ms,
                "sequence_count": len(alpha_rows),
                "accepted_sum": accepted_sum,
                "cumulative_acceptance": list(cumulative),
                "current_alpha": (
                    [float(row[candidate_len - 1]) for row in alpha_rows]
                    if candidate_len > 0
                    else []
                ),
                "last_draft_ms": (
                    float(draft_ms[candidate_len - 1])
                    if candidate_len > 0
                    else None
                ),
                "lookahead_verify_ms": (
                    float(prediction["verify_cost_lookahead_prediction_ms"])
                    if "verify_cost_lookahead_prediction_ms" in prediction
                    else None
                ),
            }
        )
    return curve


def _first_increase(curve: list[dict[str, object]], margin: float) -> int:
    previous = float(curve[0]["expected_tpot_ms"])
    for index in range(1, len(curve)):
        current = float(curve[index]["expected_tpot_ms"])
        if current > previous * (1.0 + margin):
            return int(curve[index]["draft_len"])
        previous = current
    return int(curve[-1]["draft_len"])


def _is_unimodal(curve: list[dict[str, object]], tolerance_ms: float) -> bool:
    values = [float(row["expected_tpot_ms"]) for row in curve]
    best = min(range(len(values)), key=values.__getitem__)
    decreases_to_best = all(
        values[index + 1] <= values[index] + tolerance_ms
        for index in range(0, best)
    )
    increases_after_best = all(
        values[index + 1] + tolerance_ms >= values[index]
        for index in range(best, len(values) - 1)
    )
    return decreases_to_best and increases_after_best


def _project_with_persistent_alpha(
    current: dict[str, object],
    *,
    next_draft_ms: float,
    next_verify_ms: float,
) -> float:
    cumulative = [float(value) for value in current["cumulative_acceptance"]]
    current_alpha = [float(value) for value in current["current_alpha"]]
    projected_acceptance = sum(
        value * alpha
        for value, alpha in zip(cumulative, current_alpha, strict=True)
    )
    denominator = (
        float(current["sequence_count"])
        + float(current["accepted_sum"])
        + projected_acceptance
    )
    return (
        float(current["cumulative_draft_ms"])
        + float(next_draft_ms)
        + float(next_verify_ms)
    ) / denominator


def _lookahead_selection(
    curve: list[dict[str, object]],
    margin: float,
    mode: str,
) -> int | None:
    """Select before the predicted next increase.

    ``oracle_next`` reads the complete next point. ``alpha_oracle_workload``
    repeats the current alpha but still reads the next observed draft time and
    post-draft verify prediction. ``runtime_proxy`` is causal: it repeats both
    the current alpha and latest draft-call time, and consumes the lookahead
    verify prediction emitted before the next draft call.
    """
    if len(curve) < 2:
        return None
    best = float(curve[0]["expected_tpot_ms"])
    for index in range(1, len(curve) - 1):
        current = curve[index]
        following = curve[index + 1]
        current_value = float(current["expected_tpot_ms"])
        best = min(best, current_value)
        if mode == "oracle_next":
            projected = float(following["expected_tpot_ms"])
        elif mode == "alpha_oracle_workload":
            projected = _project_with_persistent_alpha(
                current,
                next_draft_ms=(
                    float(following["cumulative_draft_ms"])
                    - float(current["cumulative_draft_ms"])
                ),
                next_verify_ms=float(following["verify_ms"]),
            )
        elif mode == "runtime_proxy":
            lookahead_verify_ms = current.get("lookahead_verify_ms")
            last_draft_ms = current.get("last_draft_ms")
            if lookahead_verify_ms is None or last_draft_ms is None:
                return None
            projected = _project_with_persistent_alpha(
                current,
                next_draft_ms=float(last_draft_ms),
                next_verify_ms=float(lookahead_verify_ms),
            )
        else:
            raise ValueError(f"unknown lookahead mode: {mode}")
        if projected > best * (1.0 + float(margin)):
            return int(current["draft_len"])
    return int(curve[-1]["draft_len"])


def _project_to_boundary(
    current: dict[str, object],
    target: dict[str, object],
) -> float:
    """Project persistent alpha to a later efficient verify boundary.

    The target verify cost comes from the later, post-route curve point, so this
    is an offline upper bound. The online policy instead asks the causal proxy
    for that endpoint before computing the intervening draft tokens.
    """
    horizon = int(target["draft_len"]) - int(current["draft_len"])
    if horizon <= 0:
        raise ValueError("boundary target must follow the current point")
    cumulative = [float(value) for value in current["cumulative_acceptance"]]
    current_alpha = [float(value) for value in current["current_alpha"]]
    accepted_sum = float(current["accepted_sum"])
    for _ in range(horizon):
        cumulative = [
            value * alpha
            for value, alpha in zip(cumulative, current_alpha, strict=True)
        ]
        accepted_sum += sum(cumulative)
    return (
        float(current["cumulative_draft_ms"])
        + float(current["last_draft_ms"]) * float(horizon)
        + float(target["verify_ms"])
    ) / (float(current["sequence_count"]) + accepted_sum)


def _boundary_selection(
    curve: list[dict[str, object]],
    *,
    boundaries: list[int],
    min_steps: int,
    margin: float,
) -> int | None:
    rows_by_len = {int(row["draft_len"]): row for row in curve}
    eligible = [
        rows_by_len[value]
        for value in boundaries
        if value >= int(min_steps) and value in rows_by_len
    ]
    if len(eligible) < 2:
        return None
    best = float("inf")
    for current, target in zip(eligible, eligible[1:], strict=False):
        best = min(best, float(current["expected_tpot_ms"]))
        projected = _project_to_boundary(current, target)
        if projected > best * (1.0 + float(margin)):
            return int(current["draft_len"])
    return int(eligible[-1]["draft_len"])


def _histogram(values: list[int]) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(Counter(values).items())
    }


def _boundary_metrics(
    records: list[dict[str, object]],
    *,
    boundaries: list[int],
    min_steps: int,
    margin: float,
) -> dict[str, object]:
    selected = [
        value
        for record in records
        if (
            value := _boundary_selection(
                record["curve"],
                boundaries=boundaries,
                min_steps=min_steps,
                margin=margin,
            )
        )
        is not None
    ]
    if not selected:
        return {"curve_count": 0, "coverage": 0.0}
    return {
        "curve_count": len(selected),
        "coverage": len(selected) / len(records),
        "selected_draft_len_mean": mean(selected),
        "selected_draft_len_histogram": _histogram(selected),
    }


def _selection_metrics(
    records: list[dict[str, object]],
    *,
    selection_key: str,
) -> dict[str, object]:
    selected = [row for row in records if row.get(selection_key) is not None]
    if not selected:
        return {"curve_count": 0, "coverage": 0.0}
    regrets = [float(row[f"{selection_key}_regret_ratio"]) for row in selected]
    return {
        "curve_count": len(selected),
        "coverage": len(selected) / len(records),
        "global_optimum_rate": sum(
            int(row[selection_key]) == int(row["best_draft_len"])
            for row in selected
        )
        / len(selected),
        "regret_ms_mean": mean(
            float(row[f"{selection_key}_regret_ms"]) for row in selected
        ),
        "regret_ratio_mean": mean(regrets),
        "regret_ratio_p90": sorted(regrets)[
            min(len(regrets) - 1, int(0.9 * len(regrets)))
        ],
    }


def _attach_selection(
    record: dict[str, object],
    curve: list[dict[str, object]],
    *,
    key: str,
    draft_len: int | None,
    best_value: float,
) -> None:
    record[key] = draft_len
    if draft_len is None:
        return
    selected = next(row for row in curve if int(row["draft_len"]) == draft_len)
    selected_value = float(selected["expected_tpot_ms"])
    record[f"{key}_regret_ms"] = selected_value - best_value
    record[f"{key}_regret_ratio"] = (
        selected_value / best_value - 1.0 if best_value > 0.0 else 0.0
    )


def _curve_summary(records: list[dict[str, object]]) -> dict[str, object]:
    if not records:
        return {"curve_count": 0}
    return {
        "curve_count": len(records),
        "unimodal_rate": sum(bool(row["unimodal"]) for row in records)
        / len(records),
        "first_increase_global_optimum_rate": sum(
            row["best_draft_len"] == row["first_increase_draft_len"]
            for row in records
        )
        / len(records),
        "first_increase_regret_ms_mean": mean(
            float(row["regret_ms"]) for row in records
        ),
        "first_increase_regret_ratio_mean": mean(
            float(row["regret_ratio"]) for row in records
        ),
        "oracle_next": _selection_metrics(
            records,
            selection_key="oracle_next_draft_len",
        ),
        "alpha_oracle_workload": _selection_metrics(
            records,
            selection_key="alpha_oracle_workload_draft_len",
        ),
        "runtime_proxy": _selection_metrics(
            records,
            selection_key="runtime_proxy_draft_len",
        ),
    }


def _margin_metrics(
    records: list[dict[str, object]],
    *,
    mode: str,
    margin: float,
) -> dict[str, object]:
    selected = []
    for record in records:
        curve = record["curve"]
        draft_len = _lookahead_selection(curve, margin, mode)
        if draft_len is None:
            continue
        best_value = min(float(row["expected_tpot_ms"]) for row in curve)
        selected_value = next(
            float(row["expected_tpot_ms"])
            for row in curve
            if int(row["draft_len"]) == draft_len
        )
        selected.append(
            {
                "best_draft_len": int(record["best_draft_len"]),
                "selection": draft_len,
                "selection_regret_ms": selected_value - best_value,
                "selection_regret_ratio": (
                    selected_value / best_value - 1.0
                    if best_value > 0.0
                    else 0.0
                ),
            }
        )
    if not selected:
        return {"curve_count": 0, "coverage": 0.0}
    regrets = [float(row["selection_regret_ratio"]) for row in selected]
    return {
        "curve_count": len(selected),
        "coverage": len(selected) / len(records),
        "global_optimum_rate": sum(
            int(row["selection"]) == int(row["best_draft_len"])
            for row in selected
        )
        / len(selected),
        "regret_ms_mean": mean(
            float(row["selection_regret_ms"]) for row in selected
        ),
        "regret_ratio_mean": mean(regrets),
        "regret_ratio_p90": sorted(regrets)[
            min(len(regrets) - 1, int(0.9 * len(regrets)))
        ],
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    alpha_calibration = None
    if str(getattr(args, "alpha_calibration", "") or ""):
        artifact = json.loads(
            Path(args.alpha_calibration).read_text(encoding="utf-8")
        )
        alpha_calibration = AcceptanceAlphaCalibration(artifact)
    records = []
    for path in _paths(args.profiles):
        data = json.loads(path.read_text(encoding="utf-8"))
        traces = data.get("spec_step_traces", data.get("model_spec_step_traces", []))
        for trace in traces:
            if not isinstance(trace, dict):
                continue
            curve = _step_curves(trace, alpha_calibration)
            if len(curve) < 2:
                continue
            best_index = min(
                range(len(curve)),
                key=lambda index: float(curve[index]["expected_tpot_ms"]),
            )
            selected_draft_len = _first_increase(curve, float(args.margin))
            selected_index = next(
                index
                for index, row in enumerate(curve)
                if int(row["draft_len"]) == selected_draft_len
            )
            best_value = float(curve[best_index]["expected_tpot_ms"])
            selected_value = float(curve[selected_index]["expected_tpot_ms"])
            record = {
                "source": str(path),
                "step_index": int(trace.get("step_index", -1)),
                "candidate_count": len(curve),
                "unimodal": _is_unimodal(curve, float(args.tolerance_ms)),
                "best_draft_len": int(curve[best_index]["draft_len"]),
                "first_increase_draft_len": selected_draft_len,
                "regret_ms": selected_value - best_value,
                "regret_ratio": (
                    selected_value / best_value - 1.0 if best_value > 0.0 else 0.0
                ),
                "curve": curve,
            }
            for mode in (
                "oracle_next",
                "alpha_oracle_workload",
                "runtime_proxy",
            ):
                _attach_selection(
                    record,
                    curve,
                    key=f"{mode}_draft_len",
                    draft_len=_lookahead_selection(
                        curve,
                        float(args.margin),
                        mode,
                    ),
                    best_value=best_value,
                )
            records.append(record)
    if not records:
        raise SystemExit("no shadow verify-cost curves found")
    output = {
        **_curve_summary(records),
        "margin": float(args.margin),
        "alpha_calibration_id": (
            alpha_calibration.calibration_id
            if alpha_calibration is not None
            else None
        ),
        "lookahead_contract": {
            "oracle_next": "uses the complete future T(k+1) point; optimistic upper bound only",
            "alpha_oracle_workload": "repeats alpha(k), but uses future draft latency and post-draft verify prediction",
            "runtime_proxy": "repeats alpha(k) and latest observed draft latency; uses the pre-draft proxy prediction for verify(k+1)",
            "curve_target": "model-predicted verify cost plus observed draft calls; not measured counterfactual TPOT",
            "online_policy_claim": False,
        },
        "by_candidate_count": {
            str(candidate_count): _curve_summary(
                [
                    row
                    for row in records
                    if int(row["candidate_count"]) == candidate_count
                ]
            )
            for candidate_count in sorted(
                {int(row["candidate_count"]) for row in records}
            )
        },
        "full_horizon": _curve_summary(
            [
                row
                for row in records
                if int(row["candidate_count"])
                == max(int(item["candidate_count"]) for item in records)
            ]
        ),
        "lookahead_margin_sweep": {
            str(margin): {
                mode: _margin_metrics(
                    records,
                    mode=mode,
                    margin=margin,
                )
                for mode in ("oracle_next", "runtime_proxy")
            }
            for margin in [
                float(value)
                for value in str(args.margin_sweep).split(",")
                if value.strip()
            ]
        },
        "records": records,
    }
    raw_boundaries = str(getattr(args, "draft_boundaries", "") or "")
    if raw_boundaries:
        boundaries = sorted(
            {
                int(value)
                for value in raw_boundaries.split(",")
                if value.strip()
            }
        )
        output["bucket_boundary_analysis"] = {
            "boundaries": boundaries,
            "min_steps": int(args.min_steps),
            "target_contract": (
                "persistent current alpha and draft latency with the future "
                "post-route endpoint verify prediction; offline upper bound, "
                "not an online TPOT claim"
            ),
            "margin_sweep": {
                str(margin): _boundary_metrics(
                    records,
                    boundaries=boundaries,
                    min_steps=int(args.min_steps),
                    margin=margin,
                )
                for margin in [
                    float(value)
                    for value in str(args.margin_sweep).split(",")
                    if value.strip()
                ]
            },
        }
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "records"}, indent=2))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", nargs="+", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--margin-sweep", default="0,0.02,0.05,0.08,0.1")
    parser.add_argument("--tolerance-ms", type=float, default=0.1)
    parser.add_argument("--alpha-calibration", default="")
    parser.add_argument("--draft-boundaries", default="")
    parser.add_argument("--min-steps", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
