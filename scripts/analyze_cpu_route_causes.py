#!/usr/bin/env python3
"""Attribute verify CPU routes and summarize admission-shadow telemetry."""

from __future__ import annotations

import argparse
import bisect
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CPU_ROUTE_STATUS = 2


def _next_window_count(
    steps_and_counts: tuple[list[int], list[int]],
    *,
    start_step: int,
    horizon_steps: int,
) -> int:
    steps, counts = steps_and_counts
    lo = bisect.bisect_left(steps, int(start_step))
    hi = bisect.bisect_right(steps, int(start_step) + int(horizon_steps))
    return sum(counts[lo:hi])


def analyze_profile(payload: dict[str, Any], *, horizon_steps: int = 8) -> dict[str, Any]:
    records = list(payload.get("model_verify_call_records") or [])
    events = list(payload.get("model_transfer_lifecycle_events") or [])
    if not records:
        raise ValueError("profile has no model_verify_call_records")

    first_layer_rows = records[0].get("metadata_layer_execution_route_rows") or {}
    num_layers = len(first_layer_rows)
    if num_layers <= 0:
        raise ValueError("verify records have no execution route rows")
    first_rows = next(iter(first_layer_rows.values()))
    routes_per_token = len(first_rows[0]) if first_rows else 0
    if routes_per_token <= 0:
        raise ValueError("verify execution route rows are empty")

    events_by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if "step_id" in event:
            events_by_step[int(event["step_id"])].append(event)

    layer_active = Counter()
    layer_cpu_routes = Counter()
    layer_cpu_experts = Counter()
    route_causes = Counter()
    drafted_prediction = Counter()
    usage_counts: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)

    previous_verify_step = min(events_by_step, default=int(records[0]["step_id"])) - 1
    for record in records:
        step_id = int(record["step_id"])
        round_events = [
            event
            for event_step in range(previous_verify_step + 1, step_id + 1)
            for event in events_by_step.get(event_step, ())
        ]
        candidate_keys = {
            (int(event["layer_idx"]), int(event["expert_idx"]))
            for event in round_events
            if event.get("event") == "admission_candidate"
        }
        submitted_keys = {
            (int(event["layer_idx"]), int(event["expert_idx"]))
            for event in round_events
            if event.get("event") == "submit"
        }
        evicted_keys = {
            (int(event["layer_idx"]), int(event["expert_idx"]))
            for event in round_events
            if event.get("event") == "evict"
        }

        route_rows = record["metadata_layer_execution_route_rows"]
        route_status = record["metadata_layer_execution_route_status"]
        draft_rows = record.get("draft_original_route_rows") or []
        for layer_key, token_rows in route_rows.items():
            layer_idx = int(layer_key)
            status_rows = route_status[layer_key]
            cpu_experts_this_call: set[int] = set()
            for token_idx, (expert_row, status_row) in enumerate(
                zip(token_rows, status_rows, strict=True)
            ):
                predicted = (
                    set(int(expert) for expert in draft_rows[token_idx - 1][layer_idx][0])
                    if token_idx > 0 and token_idx - 1 < len(draft_rows)
                    else set()
                )
                for expert_idx, status in zip(expert_row, status_row, strict=True):
                    expert_idx = int(expert_idx)
                    status = int(status)
                    key = (layer_idx, expert_idx)
                    layer_active[layer_idx] += 1
                    usage_counts[key][step_id] += 1
                    if token_idx > 0:
                        drafted_prediction["routes"] += 1
                        if expert_idx in predicted:
                            drafted_prediction["predicted_routes"] += 1
                    if status != CPU_ROUTE_STATUS:
                        continue

                    layer_cpu_routes[layer_idx] += 1
                    cpu_experts_this_call.add(expert_idx)
                    if token_idx > 0:
                        drafted_prediction["cpu_routes"] += 1
                        if expert_idx in predicted:
                            drafted_prediction["predicted_cpu_routes"] += 1

                    if key in evicted_keys:
                        cause = "evicted_before_use"
                    elif key in submitted_keys:
                        cause = "submitted_but_still_cpu"
                    elif key in candidate_keys:
                        cause = "candidate_not_admitted"
                    else:
                        cause = "not_in_candidate_set"
                    route_causes[cause] += 1
            layer_cpu_experts[layer_idx] += len(cpu_experts_this_call)
        previous_verify_step = step_id

    usage_index = {
        key: (sorted(counts), [counts[step] for step in sorted(counts)])
        for key, counts in usage_counts.items()
    }
    admission_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    for event in events:
        if event.get("event") != "submit":
            continue
        victim_expert = int(event.get("active_slot_prev_expert", -1))
        if victim_expert < 0:
            continue
        step_id = int(event["step_id"])
        layer_idx = int(event["layer_idx"])
        incoming_expert = int(event["expert_idx"])
        incoming_routes = _next_window_count(
            usage_index.get((layer_idx, incoming_expert), ([], [])),
            start_step=step_id,
            horizon_steps=horizon_steps,
        )
        victim_routes = _next_window_count(
            usage_index.get((layer_idx, victim_expert), ([], [])),
            start_step=step_id,
            horizon_steps=horizon_steps,
        )
        source_stats = admission_by_source[str(event.get("source", "unknown"))]
        source_stats["submissions_with_victim"] += 1
        source_stats["incoming_routes"] += incoming_routes
        source_stats["victim_routes"] += victim_routes
        if incoming_routes > victim_routes:
            source_stats["positive"] += 1
        elif incoming_routes < victim_routes:
            source_stats["negative"] += 1
            source_stats["perfect_future_guard_saved_routes"] += (
                victim_routes - incoming_routes
            )
        else:
            source_stats["tie"] += 1

    active_routes = sum(layer_active.values())
    cpu_routes = sum(layer_cpu_routes.values())
    cpu_expert_calls = sum(layer_cpu_experts.values())
    verify_calls = len(records)
    verified_tokens = active_routes / (num_layers * routes_per_token)

    cause_total = sum(route_causes.values())
    prediction_routes = drafted_prediction["routes"]
    prediction_cpu_routes = drafted_prediction["cpu_routes"]
    layer_rows = []
    for layer_idx in range(num_layers):
        active = layer_active[layer_idx]
        cpu = layer_cpu_routes[layer_idx]
        layer_rows.append(
            {
                "layer_idx": layer_idx,
                "active_routes": active,
                "cpu_routes": cpu,
                "cpu_route_ratio": cpu / active if active else 0.0,
                "cpu_routes_per_verify": cpu / verify_calls,
                "cpu_experts_per_verify": layer_cpu_experts[layer_idx]
                / verify_calls,
            }
        )

    return {
        "verify_calls": verify_calls,
        "num_layers": num_layers,
        "routes_per_token": routes_per_token,
        "verified_tokens": verified_tokens,
        "active_routes": active_routes,
        "cpu_routes": cpu_routes,
        "cpu_route_ratio": cpu_routes / active_routes,
        "cpu_routes_per_verify_per_layer": cpu_routes / (verify_calls * num_layers),
        "cpu_expert_calls": cpu_expert_calls,
        "cpu_experts_per_verify_per_layer": cpu_expert_calls
        / (verify_calls * num_layers),
        "cpu_routes_per_verified_token_per_layer": cpu_routes
        / (verified_tokens * num_layers),
        "route_cause_counts": dict(route_causes),
        "route_cause_ratios": {
            cause: count / cause_total for cause, count in route_causes.items()
        },
        "draft_prediction": {
            "routes": prediction_routes,
            "predicted_routes": drafted_prediction["predicted_routes"],
            "route_recall": drafted_prediction["predicted_routes"]
            / prediction_routes,
            "cpu_routes": prediction_cpu_routes,
            "predicted_cpu_routes": drafted_prediction["predicted_cpu_routes"],
            "unpredicted_cpu_routes": prediction_cpu_routes
            - drafted_prediction["predicted_cpu_routes"],
            "unpredicted_share_of_cpu_routes": (
                prediction_cpu_routes - drafted_prediction["predicted_cpu_routes"]
            )
            / prediction_cpu_routes,
        },
        "admission_horizon_steps": int(horizon_steps),
        "admission_by_source": {
            source: dict(stats) for source, stats in admission_by_source.items()
        },
        "layers": layer_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--horizon-steps", type=int, default=8)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.profile.read_text(encoding="utf-8"))
    result = analyze_profile(payload, horizon_steps=args.horizon_steps)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
