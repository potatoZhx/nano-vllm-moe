#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
from pathlib import Path


def _paths(values: list[str]) -> list[Path]:
    paths: set[Path] = set()
    for value in values:
        path = Path(value)
        if path.is_dir():
            paths.update(path.rglob("sample*.json"))
        else:
            paths.update(Path(item) for item in glob.glob(value, recursive=True))
    return sorted(path.resolve() for path in paths if path.is_file())


def _rows(data: dict, key: str) -> list:
    value = data.get(f"model_{key}", data.get(key, []))
    return value if isinstance(value, list) else []


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    )
    left_sq = sum((value - left_mean) ** 2 for value in left)
    right_sq = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_sq * right_sq)
    return numerator / denominator if denominator > 1e-12 else None


def _stats(values: list[float]) -> dict[str, float | None]:
    return {
        "count": len(values),
        "mean_ms": sum(values) / len(values) if values else None,
        "p50_ms": _percentile(values, 50),
        "p90_ms": _percentile(values, 90),
    }


def _route_band(routes: float) -> str:
    if routes <= 0:
        return "0"
    if routes <= 3:
        return "1-3"
    if routes <= 7:
        return "4-7"
    if routes <= 15:
        return "8-15"
    if routes <= 31:
        return "16-31"
    return ">=32"


def run(args: argparse.Namespace) -> dict[str, object]:
    paths = _paths(args.profiles)
    if not paths:
        raise SystemExit("no op-event profile JSON files matched")
    calls = []
    layer_sync_rows = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        measurement = data.get("verify_cost_measurement")
        if not isinstance(measurement, dict) or not bool(measurement.get("enabled")):
            raise ValueError(f"{path}: full execution workload profile is required")
        if bool(measurement.get("profile_cuda_sync")):
            raise ValueError(f"{path}: engine profile CUDA sync must remain disabled")
        traces = {
            int(row.get("verify_call_index", index)): row
            for index, row in enumerate(_rows(data, "spec_step_traces"))
            if isinstance(row, dict)
        }
        events_by_step: dict[int, list[dict]] = {}
        for event in _rows(data, "verify_op_event_records"):
            if not isinstance(event, dict):
                continue
            if event.get("error"):
                raise ValueError(f"{path}: op event error: {event['error']}")
            elapsed_ms = float(event.get("elapsed_ms", 0.0) or 0.0)
            if not math.isfinite(elapsed_ms) or elapsed_ms < 0.0:
                raise ValueError(f"{path}: invalid op event duration")
            events_by_step.setdefault(int(event.get("step_id", -1)), []).append(event)

        for ordinal, record in enumerate(_rows(data, "verify_call_records")):
            if ordinal < int(args.drop_first_calls) or not isinstance(record, dict):
                continue
            call_index = int(record.get("call_index", ordinal))
            step_id = int(record.get("step_id", -1))
            trace = traces.get(call_index)
            events = events_by_step.get(step_id, [])
            if trace is None or not events:
                raise ValueError(
                    f"{path}: call {call_index} lacks joined trace/op events"
                )
            route_counts = record.get("metadata_layer_execution_cpu_route_counts")
            if not isinstance(route_counts, dict):
                raise ValueError(f"{path}: call {call_index} lacks execution workload")
            label_ms: dict[str, float] = {}
            for event in events:
                label = str(event.get("label", ""))
                elapsed_ms = float(event.get("elapsed_ms", 0.0) or 0.0)
                label_ms[label] = label_ms.get(label, 0.0) + elapsed_ms
                if label == "kt.cpuinfer_sync" and int(event.get("layer_idx", -1)) >= 0:
                    layer_idx = int(event["layer_idx"])
                    counts = [float(value) for value in route_counts[str(layer_idx)]]
                    routes = sum(counts)
                    layer_sync_rows.append(
                        {
                            "routes": routes,
                            "experts": float(sum(value > 0.0 for value in counts)),
                            "sync_ms": elapsed_ms,
                            "bucket": int(record.get("bucket", 0)),
                            "layer_idx": layer_idx,
                        }
                    )
            target_ms = float(trace.get("verify_accept_ready_ms", 0.0) or 0.0)
            stream_ms = float(record.get("stream_ms", 0.0) or 0.0)
            calls.append(
                {
                    "source": str(path),
                    "call_index": call_index,
                    "step_id": step_id,
                    "bucket": int(record.get("bucket", 0)),
                    "logical_tokens": int(record.get("token_count", 0)),
                    "target_ms": target_ms,
                    "stream_ms": stream_ms,
                    "stream_external_residual_ms": target_ms - stream_ms,
                    "cpu_routes": float(
                        record.get("metadata_execution_cpu_routes_sum", 0.0) or 0.0
                    ),
                    "cpu_experts": float(
                        record.get("metadata_execution_cpu_experts_sum", 0.0) or 0.0
                    ),
                    "labels": label_ms,
                }
            )
    if not calls:
        raise SystemExit("no joined op-event verify calls found")

    target = [float(row["target_ms"]) for row in calls]
    routes = [float(row["cpu_routes"]) for row in calls]
    experts = [float(row["cpu_experts"]) for row in calls]
    labels = sorted({label for row in calls for label in row["labels"]})
    label_stats = {}
    for label in labels:
        values = [float(row["labels"].get(label, 0.0)) for row in calls]
        label_stats[label] = {
            **_stats(values),
            "vs_accept_ready_correlation": _correlation(values, target),
            "vs_cpu_routes_correlation": _correlation(values, routes),
            "vs_cpu_experts_correlation": _correlation(values, experts),
        }
    label_stats = dict(
        sorted(
            label_stats.items(),
            key=lambda item: float(item[1]["mean_ms"] or 0.0),
            reverse=True,
        )
    )

    bands = {}
    for band in ("0", "1-3", "4-7", "8-15", "16-31", ">=32"):
        rows = [row for row in layer_sync_rows if _route_band(float(row["routes"])) == band]
        bands[band] = {
            "row_count": len(rows),
            "cpu_routes_mean": (
                sum(float(row["routes"]) for row in rows) / len(rows) if rows else None
            ),
            "cpu_experts_mean": (
                sum(float(row["experts"]) for row in rows) / len(rows) if rows else None
            ),
            "cpuinfer_sync_ms_mean": (
                sum(float(row["sync_ms"]) for row in rows) / len(rows) if rows else None
            ),
            "cpuinfer_sync_ms_p90": _percentile(
                [float(row["sync_ms"]) for row in rows],
                90,
            ),
        }

    by_bucket = {}
    for bucket in sorted({int(row["bucket"]) for row in calls}):
        rows = [row for row in calls if int(row["bucket"]) == bucket]
        by_bucket[str(bucket)] = {
            "call_count": len(rows),
            "accept_ready": _stats([float(row["target_ms"]) for row in rows]),
            "stream": _stats([float(row["stream_ms"]) for row in rows]),
            "stream_external_residual": _stats(
                [float(row["stream_external_residual_ms"]) for row in rows]
            ),
            "cpu_routes_mean": sum(float(row["cpu_routes"]) for row in rows)
            / len(rows),
            "cpu_experts_mean": sum(float(row["cpu_experts"]) for row in rows)
            / len(rows),
        }

    output = {
        "diagnostic_only": True,
        "event_accounting": (
            "CUDA event labels are nested; parent and child rows are not additive. "
            "Op-event synchronization perturbs scheduling, so these profiles are "
            "excluded from model fitting and TPOT performance claims."
        ),
        "profile_manifest": [
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in paths
        ],
        "call_count": len(calls),
        "accept_ready": _stats(target),
        "stream": _stats([float(row["stream_ms"]) for row in calls]),
        "stream_external_residual": _stats(
            [float(row["stream_external_residual_ms"]) for row in calls]
        ),
        "cpu_routes_vs_accept_ready_correlation": _correlation(routes, target),
        "cpu_experts_vs_accept_ready_correlation": _correlation(experts, target),
        "label_stats": label_stats,
        "layer_cpuinfer_sync": {
            "row_count": len(layer_sync_rows),
            "routes_correlation": _correlation(
                [float(row["routes"]) for row in layer_sync_rows],
                [float(row["sync_ms"]) for row in layer_sync_rows],
            ),
            "experts_correlation": _correlation(
                [float(row["experts"]) for row in layer_sync_rows],
                [float(row["sync_ms"]) for row in layer_sync_rows],
            ),
            "route_bands": bands,
        },
        "by_bucket": by_bucket,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    report_path = output_path.with_suffix(".md")
    lines = [
        "# Verify Op Breakdown",
        "",
        f"- calls: `{len(calls)}`",
        "- diagnostic only: `true`",
        "- event rows are nested and are not summed into a wall-time total",
        "",
        "| label | mean ms | p90 ms | corr target | corr CPU routes |",
        "|:---|---:|---:|---:|---:|",
    ]
    for label, stats in list(label_stats.items())[:30]:
        lines.append(
            f"| `{label}` | {float(stats['mean_ms'] or 0.0):.3f} | "
            f"{float(stats['p90_ms'] or 0.0):.3f} | "
            f"{float(stats['vs_accept_ready_correlation'] or 0.0):.3f} | "
            f"{float(stats['vs_cpu_routes_correlation'] or 0.0):.3f} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "label_stats"}, indent=2))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--drop-first-calls", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
