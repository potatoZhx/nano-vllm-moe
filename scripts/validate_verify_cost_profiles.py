#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Iterable


def _expand_inputs(values: Iterable[str]) -> list[Path]:
    paths: set[Path] = set()
    for value in values:
        path = Path(value)
        if path.is_dir():
            paths.update(path.rglob("sample*.json"))
        else:
            paths.update(Path(item) for item in glob.glob(value, recursive=True))
    return sorted(path.resolve() for path in paths if path.is_file())


def _profile_list(data: dict, key: str) -> list:
    value = data.get(f"model_{key}", data.get(key, []))
    return value if isinstance(value, list) else []


def validate_profile(
    path: Path,
    *,
    num_layers: int,
    num_experts: int,
    top_k: int,
) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    measurement = data.get("verify_cost_measurement")
    if not isinstance(measurement, dict) or not measurement.get("enabled"):
        errors.append("missing enabled verify_cost_measurement")
    else:
        if measurement.get("target") != "spec.verify_accept_ready_ms":
            errors.append("unexpected measurement target")
        if bool(measurement.get("profile_cuda_sync")):
            errors.append("profile_cuda_sync must be false")

    records = _profile_list(data, "verify_call_records")
    traces = _profile_list(data, "spec_step_traces")
    trace_by_call: dict[int, dict] = {}
    for index, row in enumerate(traces):
        if not isinstance(row, dict):
            errors.append(f"trace {index}: record is not an object")
            continue
        if "verify_call_index" not in row:
            errors.append(f"trace {index}: missing verify_call_index")
            continue
        trace_call_index = int(row["verify_call_index"])
        if trace_call_index in trace_by_call:
            errors.append(
                f"trace {index}: duplicate verify_call_index={trace_call_index}"
            )
            continue
        trace_by_call[trace_call_index] = row
    seen_call_indices: set[int] = set()
    seen_step_ids: set[int] = set()
    targets: list[float] = []
    streams: list[float] = []
    cpu_routes: list[float] = []
    buckets: set[int] = set()

    if not records:
        errors.append("no verify_call_records")
    for ordinal, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"call {ordinal}: record is not an object")
            continue
        call_index = int(record.get("call_index", ordinal))
        step_id = int(record.get("step_id", -1))
        if call_index in seen_call_indices:
            errors.append(f"call {ordinal}: duplicate call_index={call_index}")
        if step_id < 0 or step_id in seen_step_ids:
            errors.append(f"call {ordinal}: invalid/duplicate step_id={step_id}")
        seen_call_indices.add(call_index)
        seen_step_ids.add(step_id)

        bucket = int(record.get("bucket", 0) or 0)
        logical_tokens = int(record.get("token_count", 0) or 0)
        buckets.add(bucket)
        if bucket < logical_tokens or logical_tokens <= 0:
            errors.append(
                f"call {ordinal}: invalid logical/bucket tokens {logical_tokens}/{bucket}"
            )
        if not bool(record.get("used_cuda_graph")):
            errors.append(f"call {ordinal}: verify did not use a CUDA graph")
        if not bool(record.get("used_kt_hybrid")):
            errors.append(f"call {ordinal}: verify did not use KT hybrid execution")
        if not bool(record.get("outputs_ready")):
            errors.append(f"call {ordinal}: verify outputs were not host-ready")
        if bool(record.get("return_logits")):
            errors.append(f"call {ordinal}: calibration must use greedy token outputs")
        if not bool(record.get("metadata_execution_available")):
            errors.append(f"call {ordinal}: execution metadata is unavailable")
        padding_tokens = int(record.get("padding_token_count", -1))
        if padding_tokens != bucket - logical_tokens:
            errors.append(
                f"call {ordinal}: padding tokens {padding_tokens} != "
                f"bucket-logical {bucket - logical_tokens}"
            )
        layer_count = int(float(record.get("metadata_execution_layer_count", 0) or 0))
        if layer_count != num_layers:
            errors.append(f"call {ordinal}: execution layers {layer_count} != {num_layers}")
        active_routes = float(
            record.get("metadata_execution_active_routes_sum", 0.0) or 0.0
        )
        expected_active_routes = float(bucket * top_k * num_layers)
        if active_routes != expected_active_routes:
            errors.append(
                f"call {ordinal}: execution routes {active_routes} != "
                f"bucket*top_k*layers {expected_active_routes}"
            )

        counts = record.get("metadata_layer_execution_cpu_route_counts")
        if not isinstance(counts, dict) or set(counts) != {
            str(index) for index in range(num_layers)
        }:
            errors.append(f"call {ordinal}: incomplete layer execution route counts")
            continue
        counted_cpu_routes = 0.0
        counted_cpu_experts = 0
        for layer_idx in range(num_layers):
            row = counts[str(layer_idx)]
            if not isinstance(row, list) or len(row) != num_experts:
                errors.append(
                    f"call {ordinal}: layer {layer_idx} count width is not {num_experts}"
                )
                continue
            if any(float(value) < 0.0 for value in row):
                errors.append(f"call {ordinal}: layer {layer_idx} has negative counts")
            if any(not float(value).is_integer() for value in row):
                errors.append(
                    f"call {ordinal}: layer {layer_idx} has fractional CPU counts"
                )
            counted_cpu_routes += sum(float(value) for value in row)
            counted_cpu_experts += sum(float(value) > 0.0 for value in row)
        recorded_cpu_routes = float(
            record.get("metadata_execution_cpu_routes_sum", 0.0) or 0.0
        )
        if counted_cpu_routes != recorded_cpu_routes:
            errors.append(
                f"call {ordinal}: CPU route vector sum {counted_cpu_routes} != "
                f"aggregate {recorded_cpu_routes}"
            )
        recorded_cpu_experts = float(
            record.get("metadata_execution_cpu_experts_sum", 0.0) or 0.0
        )
        if counted_cpu_experts != recorded_cpu_experts:
            errors.append(
                f"call {ordinal}: CPU expert vector count {counted_cpu_experts} != "
                f"aggregate {recorded_cpu_experts}"
            )
        cpu_routes.append(recorded_cpu_routes)

        all_counts = record.get("metadata_layer_execution_route_counts")
        if not isinstance(all_counts, dict) or set(all_counts) != {
            str(index) for index in range(num_layers)
        }:
            errors.append(f"call {ordinal}: incomplete full execution route counts")
        else:
            for layer_idx in range(num_layers):
                row = all_counts[str(layer_idx)]
                if not isinstance(row, list) or len(row) != num_experts:
                    errors.append(
                        f"call {ordinal}: layer {layer_idx} full count width is not {num_experts}"
                    )
                    continue
                if any(float(value) < 0.0 for value in row):
                    errors.append(
                        f"call {ordinal}: layer {layer_idx} full counts are negative"
                    )
                if any(not float(value).is_integer() for value in row):
                    errors.append(
                        f"call {ordinal}: layer {layer_idx} has fractional full counts"
                    )
                layer_routes = sum(float(value) for value in row)
                expected_layer_routes = float(bucket * top_k)
                if layer_routes != expected_layer_routes:
                    errors.append(
                        f"call {ordinal}: layer {layer_idx} full routes {layer_routes} != "
                        f"bucket*top_k {expected_layer_routes}"
                    )
                cpu_row = counts.get(str(layer_idx), [])
                if len(cpu_row) == len(row) and any(
                    float(cpu_value) > float(all_value)
                    for cpu_value, all_value in zip(cpu_row, row, strict=True)
                ):
                    errors.append(
                        f"call {ordinal}: layer {layer_idx} CPU counts exceed full counts"
                    )

        if not bool(record.get("stream_ms_available")):
            errors.append(f"call {ordinal}: stream timing unavailable")
        stream_ms = float(record.get("stream_ms", 0.0) or 0.0)
        if not math.isfinite(stream_ms) or stream_ms <= 0.0:
            errors.append(f"call {ordinal}: invalid stream_ms={stream_ms}")
        streams.append(stream_ms)

        trace = trace_by_call.get(call_index)
        if trace is None:
            errors.append(f"call {ordinal}: no matching spec trace")
            continue
        target_ms = float(trace.get("verify_accept_ready_ms", 0.0) or 0.0)
        model_call_ms = float(trace.get("verify_model_call_ms", 0.0) or 0.0)
        accept_ms = float(trace.get("verify_accept_ms", 0.0) or 0.0)
        if not all(math.isfinite(value) and value >= 0.0 for value in (target_ms, model_call_ms, accept_ms)):
            errors.append(f"call {ordinal}: non-finite acceptance-ready timing")
        if target_ms <= 0.0 or target_ms + 1e-6 < model_call_ms:
            errors.append(
                f"call {ordinal}: acceptance-ready {target_ms} precedes model call {model_call_ms}"
            )
        if target_ms + 1e-6 < stream_ms:
            errors.append(
                f"call {ordinal}: acceptance-ready {target_ms} precedes stream {stream_ms}"
            )
        trace_tokens = int(trace.get("verify_token_count", -1))
        if trace_tokens != logical_tokens:
            errors.append(
                f"call {ordinal}: trace verify tokens {trace_tokens} != "
                f"record logical tokens {logical_tokens}"
            )
        targets.append(target_ms)

    orphan_trace_calls = sorted(set(trace_by_call) - seen_call_indices)
    if orphan_trace_calls:
        errors.append(
            f"spec traces have unknown verify call indices: {orphan_trace_calls[:5]}"
        )

    op_records = _profile_list(data, "verify_op_event_records")
    orphan_op_steps = sorted(
        {
            int(row.get("step_id", -1))
            for row in op_records
            if isinstance(row, dict) and int(row.get("step_id", -1)) not in seen_step_ids
        }
    )
    if orphan_op_steps:
        errors.append(f"op-event rows have unknown step ids: {orphan_op_steps[:5]}")

    return {
        "path": str(path),
        "valid": not errors,
        "errors": errors,
        "verify_calls": len(records),
        "buckets": sorted(buckets),
        "accept_ready_ms_mean": sum(targets) / len(targets) if targets else None,
        "stream_ms_mean": sum(streams) / len(streams) if streams else None,
        "execution_cpu_routes_mean": sum(cpu_routes) / len(cpu_routes) if cpu_routes else None,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    paths = _expand_inputs(args.profiles)
    if not paths:
        raise SystemExit("no profile JSON files matched --profiles")
    results = [
        validate_profile(
            path,
            num_layers=int(args.num_layers),
            num_experts=int(args.num_experts),
            top_k=int(args.top_k),
        )
        for path in paths
    ]
    output = {
        "valid": all(bool(result["valid"]) for result in results),
        "profile_count": len(results),
        "verify_call_count": sum(int(result["verify_calls"]) for result in results),
        "results": results,
    }
    print(json.dumps(output, indent=2))
    if not output["valid"]:
        raise SystemExit(1)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", nargs="+", required=True)
    parser.add_argument("--num-layers", type=int, default=48)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
