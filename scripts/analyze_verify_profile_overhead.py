#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _summary(path: str) -> dict:
    target = Path(path)
    if target.is_dir():
        target = target / "summary.json"
    return json.loads(target.read_text(encoding="utf-8"))


def _profiles(path: str) -> dict[tuple[str, str], dict]:
    root = Path(path)
    if root.is_file():
        root = root.parent
    profiles = {}
    for profile_path in root.rglob("sample*.json"):
        data = json.loads(profile_path.read_text(encoding="utf-8"))
        measurement = data.get("verify_cost_measurement", {})
        sample = measurement.get("sample", {}) if isinstance(measurement, dict) else {}
        key = (str(sample.get("dataset", "")), str(sample.get("sample_id", "")))
        if all(key):
            profiles[key] = data
    return profiles


def _traces(data: dict) -> dict[int, dict]:
    rows = data.get("spec_step_traces", data.get("model_spec_step_traces", []))
    return {
        int(row.get("verify_call_index", index)): row
        for index, row in enumerate(rows)
        if isinstance(row, dict)
    }


def _records(data: dict) -> dict[int, dict]:
    rows = data.get("model_verify_call_records", data.get("verify_call_records", []))
    return {
        int(row.get("call_index", index)): row
        for index, row in enumerate(rows)
        if isinstance(row, dict)
    }


def _cluster_bootstrap_ci(
    values_by_cluster: dict[tuple[str, str], list[float]],
    *,
    seed: int,
    iterations: int,
) -> list[float] | None:
    clusters = sorted(values_by_cluster)
    if len(clusters) < 2:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        sampled = [rng.choice(clusters) for _ in clusters]
        values = [value for cluster in sampled for value in values_by_cluster[cluster]]
        means.append(sum(values) / len(values))
    means.sort()
    return [
        means[int(0.025 * (len(means) - 1))],
        means[int(0.975 * (len(means) - 1))],
    ]


def run(args: argparse.Namespace) -> dict[str, object]:
    reference_summary = _summary(args.reference)
    instrumented_summary = _summary(args.instrumented)
    reference_rows = {
        (str(row.get("dataset", "")), str(row.get("sample_id", ""))): row
        for row in reference_summary.get("rows", [])
        if row.get("status") == "ok"
    }
    instrumented_rows = {
        (str(row.get("dataset", "")), str(row.get("sample_id", ""))): row
        for row in instrumented_summary.get("rows", [])
        if row.get("status") == "ok"
    }
    keys = sorted(set(reference_rows) & set(instrumented_rows))
    if not keys:
        raise SystemExit("no paired request rows")
    request_pairs = []
    for key in keys:
        reference = reference_rows[key]
        instrumented = instrumented_rows[key]
        request_pairs.append(
            {
                "dataset": key[0],
                "sample_id": key[1],
                "digest_equal": reference.get("outputs_digest")
                == instrumented.get("outputs_digest"),
                "reference_tpot_ms": float(reference["tpot_ms"]),
                "instrumented_tpot_ms": float(instrumented["tpot_ms"]),
                "delta_tpot_ms": float(instrumented["tpot_ms"])
                - float(reference["tpot_ms"]),
            }
        )

    reference_profiles = _profiles(args.reference)
    instrumented_profiles = _profiles(args.instrumented)
    call_pairs = []
    call_deltas_by_request: dict[tuple[str, str], list[float]] = {}
    stream_deltas_by_request: dict[tuple[str, str], list[float]] = {}
    for key in sorted(set(reference_profiles) & set(instrumented_profiles)):
        reference_records = _records(reference_profiles[key])
        instrumented_records = _records(instrumented_profiles[key])
        reference_traces = _traces(reference_profiles[key])
        instrumented_traces = _traces(instrumented_profiles[key])
        for call_index in sorted(
            set(reference_records)
            & set(instrumented_records)
            & set(reference_traces)
            & set(instrumented_traces)
        ):
            reference_record = reference_records[call_index]
            instrumented_record = instrumented_records[call_index]
            comparable = (
                int(reference_record.get("bucket", -1))
                == int(instrumented_record.get("bucket", -2))
                and int(reference_record.get("token_count", -1))
                == int(instrumented_record.get("token_count", -2))
            )
            reference_target = float(
                reference_traces[call_index].get("verify_accept_ready_ms", 0.0)
            )
            instrumented_target = float(
                instrumented_traces[call_index].get("verify_accept_ready_ms", 0.0)
            )
            target_delta = instrumented_target - reference_target
            reference_stream = float(reference_record.get("stream_ms", 0.0) or 0.0)
            instrumented_stream = float(
                instrumented_record.get("stream_ms", 0.0) or 0.0
            )
            stream_delta = instrumented_stream - reference_stream
            call_pairs.append(
                {
                    "dataset": key[0],
                    "sample_id": key[1],
                    "call_index": call_index,
                    "comparable_shape": comparable,
                    "bucket": int(instrumented_record.get("bucket", 0)),
                    "token_count": int(instrumented_record.get("token_count", 0)),
                    "reference_accept_ready_ms": reference_target,
                    "instrumented_accept_ready_ms": instrumented_target,
                    "delta_accept_ready_ms": target_delta,
                    "reference_stream_ms": reference_stream,
                    "instrumented_stream_ms": instrumented_stream,
                    "delta_stream_ms": stream_delta,
                }
            )
            if comparable:
                call_deltas_by_request.setdefault(key, []).append(target_delta)
                stream_deltas_by_request.setdefault(key, []).append(stream_delta)

    call_deltas = [value for values in call_deltas_by_request.values() for value in values]
    stream_deltas = [
        value for values in stream_deltas_by_request.values() for value in values
    ]
    output: dict[str, object] = {
        "request_pair_count": len(request_pairs),
        "all_output_digests_equal": all(row["digest_equal"] for row in request_pairs),
        "request_pairs": request_pairs,
        "call_pair_count": len(call_pairs),
        "all_call_shapes_equal": all(row["comparable_shape"] for row in call_pairs),
        "call_shape_match_rate": (
            sum(bool(row["comparable_shape"]) for row in call_pairs) / len(call_pairs)
            if call_pairs
            else None
        ),
        "accept_ready_overhead_ms_mean": (
            sum(call_deltas) / len(call_deltas) if call_deltas else None
        ),
        "accept_ready_overhead_ms_cluster_bootstrap_95ci": _cluster_bootstrap_ci(
            call_deltas_by_request,
            seed=int(args.seed),
            iterations=int(args.bootstrap_iterations),
        ),
        "stream_overhead_ms_mean": (
            sum(stream_deltas) / len(stream_deltas) if stream_deltas else None
        ),
        "stream_overhead_ms_cluster_bootstrap_95ci": _cluster_bootstrap_ci(
            stream_deltas_by_request,
            seed=int(args.seed) + 1,
            iterations=int(args.bootstrap_iterations),
        ),
        "call_pairs": call_pairs,
    }
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    if not output["all_output_digests_equal"]:
        raise SystemExit(1)
    if args.require_shape_match and not output["all_call_shapes_equal"]:
        raise SystemExit(1)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--instrumented", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument(
        "--require-shape-match",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
