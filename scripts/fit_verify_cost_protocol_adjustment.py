#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import statistics
from pathlib import Path

from nanovllm.engine.speculative.verify_cost_model import (
    VerifyTimeCostModel,
    compute_model_id,
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


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _metrics(residuals: list[float]) -> dict[str, float | int]:
    return {
        "sample_count": len(residuals),
        "mean_signed_error_ms": statistics.mean(residuals),
        "mae_ms": statistics.mean(abs(value) for value in residuals),
        "p90_abs_error_ms": _percentile(
            [abs(value) for value in residuals],
            90,
        ),
    }


def _training_rows(
    paths: list[Path],
    *,
    model_id: str,
    drop_first_calls: int,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        measurement = data.get("verify_cost_measurement")
        if not isinstance(measurement, dict):
            raise ValueError(f"{path}: missing verify cost measurement contract")
        if bool(measurement.get("enabled")):
            raise ValueError(f"{path}: execution workload instrumentation is enabled")
        if measurement.get("target") != "spec.verify_accept_ready_ms":
            raise ValueError(f"{path}: unexpected latency target")
        if bool(measurement.get("profile_cuda_sync")):
            raise ValueError(f"{path}: synchronous CUDA profiling is enabled")
        records = data.get(
            "model_verify_call_records",
            data.get("verify_call_records", []),
        )
        records_by_call = {
            int(record.get("call_index", index)): record
            for index, record in enumerate(records)
            if isinstance(record, dict)
        }
        traces = data.get(
            "spec_step_traces",
            data.get("model_spec_step_traces", []),
        )
        for ordinal, trace in enumerate(traces):
            if ordinal < drop_first_calls or not isinstance(trace, dict):
                continue
            predictions = trace.get("verify_cost_predictions", [])
            if not isinstance(predictions, list) or not predictions:
                continue
            prediction = predictions[-1]
            if not isinstance(prediction, dict):
                continue
            call_index = int(trace.get("verify_call_index", ordinal))
            record = records_by_call.get(call_index)
            if not isinstance(record, dict):
                raise ValueError(f"{path}: missing verify record {call_index}")
            if str(record.get("verify_cost_model_mode", "")) != "shadow":
                raise ValueError(f"{path}: call {call_index} is not shadow mode")
            if bool(record.get("metadata_execution_available")):
                raise ValueError(f"{path}: call {call_index} exposes execution workload")
            if not bool(record.get("stream_ms_available")):
                raise ValueError(f"{path}: call {call_index} lacks stream timing")
            sequences = trace.get("sequences", [])
            sampling_ready = (
                bool(record.get("return_logits"))
                and float(trace.get("verify_accept_ms", 0.0) or 0.0) > 0.0
                and isinstance(sequences, list)
                and bool(sequences)
                and all(
                    isinstance(sequence, dict)
                    and sequence.get("acceptance_mode") == "standard_sampling"
                    and isinstance(sequence.get("next_token"), int)
                    for sequence in sequences
                )
            )
            if not sampling_ready:
                raise ValueError(
                    f"{path}: call {call_index} is not sampling accept-ready"
                )
            actual_model_id = str(
                prediction.get("verify_cost_model_id", "") or ""
            )
            if actual_model_id != model_id:
                raise ValueError(
                    f"{path}: call {call_index} model id {actual_model_id!r} "
                    f"!= {model_id!r}"
                )
            bucket = int(record.get("bucket", -1))
            if int(prediction.get("verify_cost_bucket", -1)) != bucket:
                raise ValueError(f"{path}: call {call_index} bucket mismatch")
            target_ms = float(trace.get("verify_accept_ready_ms", 0.0) or 0.0)
            prediction_ms = float(
                trace.get("verify_cost_prediction_ms", 0.0) or 0.0
            )
            if not all(
                math.isfinite(value) and value > 0.0
                for value in (target_ms, prediction_ms)
            ):
                continue
            rows.append(
                {
                    "source": str(path),
                    "call_index": call_index,
                    "bucket": bucket,
                    "target_ms": target_ms,
                    "prediction_ms": prediction_ms,
                    "residual_ms": target_ms - prediction_ms,
                }
            )
    return rows


def run(args: argparse.Namespace) -> dict[str, object]:
    paths = _paths(args.profiles)
    if not paths:
        raise SystemExit("no sampling shadow profiles matched")
    artifact_path = Path(args.artifact).resolve()
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise ValueError("verify cost artifact must be a JSON object")
    base_model = VerifyTimeCostModel(artifact)
    if base_model.protocol_adjustment is not None:
        raise ValueError("input artifact already has a protocol adjustment")
    rows = _training_rows(
        paths,
        model_id=base_model.model_id,
        drop_first_calls=int(args.drop_first_calls),
    )
    if len(rows) < int(args.minimum_samples):
        raise SystemExit(
            f"only {len(rows)} training samples; need {args.minimum_samples}"
        )
    offsets: dict[str, float] = {}
    affine: dict[str, dict[str, float]] = {}
    by_bucket: dict[str, object] = {}
    adjusted_residuals: list[float] = []
    for bucket in base_model.buckets:
        bucket_rows = [row for row in rows if int(row["bucket"]) == bucket]
        if len(bucket_rows) < int(args.minimum_bucket_samples):
            raise SystemExit(
                f"bucket {bucket} has {len(bucket_rows)} training samples; "
                f"need {args.minimum_bucket_samples}"
            )
        residuals = [float(row["residual_ms"]) for row in bucket_rows]
        if args.estimator == "median_offset":
            offset = float(statistics.median(residuals))
            offsets[str(bucket)] = offset
            corrected = [value - offset for value in residuals]
            parameters: dict[str, float] = {"offset_ms": offset}
        else:
            predictions = [float(row["prediction_ms"]) for row in bucket_rows]
            targets = [float(row["target_ms"]) for row in bucket_rows]
            prediction_mean = statistics.mean(predictions)
            target_mean = statistics.mean(targets)
            denominator = sum(
                (value - prediction_mean) ** 2 for value in predictions
            )
            if denominator <= 0.0:
                raise ValueError(f"bucket {bucket} prediction variance is zero")
            slope = sum(
                (prediction - prediction_mean) * (target - target_mean)
                for prediction, target in zip(predictions, targets, strict=True)
            ) / denominator
            intercept = target_mean - slope * prediction_mean
            if not 0.25 <= slope <= 2.0:
                raise ValueError(
                    f"bucket {bucket} affine slope {slope} is outside [0.25, 2.0]"
                )
            affine[str(bucket)] = {
                "slope": float(slope),
                "intercept_ms": float(intercept),
            }
            corrected = [
                target - (slope * prediction + intercept)
                for prediction, target in zip(predictions, targets, strict=True)
            ]
            parameters = {
                "slope": float(slope),
                "intercept_ms": float(intercept),
            }
        adjusted_residuals.extend(corrected)
        by_bucket[str(bucket)] = {
            **parameters,
            "base_metrics": _metrics(residuals),
            "adjusted_metrics": _metrics(corrected),
        }

    adjustment_kind = (
        "bucket_offset_ms"
        if args.estimator == "median_offset"
        else "bucket_affine"
    )
    manifest = [
        {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in paths
    ]
    adjustment = {
        "kind": adjustment_kind,
        "acceptance_strategy": str(args.acceptance_strategy),
        "temperature": float(args.temperature),
        "estimator": str(args.estimator),
        "training": {
            "drop_first_calls": int(args.drop_first_calls),
            "minimum_bucket_samples": int(args.minimum_bucket_samples),
            "base_metrics": _metrics(
                [float(row["residual_ms"]) for row in rows]
            ),
            "adjusted_metrics": _metrics(adjusted_residuals),
            "by_bucket": by_bucket,
            "profiles": manifest,
        },
    }
    if adjustment_kind == "bucket_offset_ms":
        adjustment["bucket_offsets_ms"] = offsets
    else:
        adjustment["bucket_affine"] = affine
    adjusted_artifact = dict(artifact)
    adjusted_artifact.pop("deployment_validation", None)
    adjusted_artifact.pop("sampling_deployment_validation", None)
    adjusted_artifact["base_model_id"] = base_model.model_id
    adjusted_artifact["protocol_adjustment"] = adjustment
    adjusted_artifact["model_id"] = compute_model_id(adjusted_artifact)
    VerifyTimeCostModel(adjusted_artifact)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(adjusted_artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "base_model_id": base_model.model_id,
        "adjusted_model_id": adjusted_artifact["model_id"],
        "sample_count": len(rows),
        "protocol_adjustment": {
            "kind": adjustment_kind,
            "bucket_offsets_ms": offsets,
            "bucket_affine": affine,
        },
        "base_metrics": adjustment["training"]["base_metrics"],
        "adjusted_metrics": adjustment["training"]["adjusted_metrics"],
        "output": str(output_path.resolve()),
    }
    print(json.dumps(summary, indent=2))
    return adjusted_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", nargs="+", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--drop-first-calls", type=int, default=2)
    parser.add_argument("--minimum-samples", type=int, default=100)
    parser.add_argument("--minimum-bucket-samples", type=int, default=20)
    parser.add_argument(
        "--estimator",
        choices=["median_offset", "bucket_affine_ols"],
        default="median_offset",
    )
    parser.add_argument(
        "--acceptance-strategy",
        choices=["standard_sampling"],
        default="standard_sampling",
    )
    parser.add_argument("--temperature", type=float, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
