#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import random
import re
from pathlib import Path

from nanovllm.engine.speculative.verify_cost_model import VerifyTimeCostModel


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


def _ranking(
    targets: list[float],
    predictions: list[float],
    sources: list[str] | None = None,
    maximum_pairs: int = 100_000,
) -> tuple[float, int, int]:
    rng = random.Random(0)
    sampled: list[bool] = []
    eligible = 0
    for left in range(len(targets)):
        for right in range(left + 1, len(targets)):
            if sources is not None and sources[left] != sources[right]:
                continue
            target_delta = targets[left] - targets[right]
            if abs(target_delta) < 1.0:
                continue
            prediction_delta = predictions[left] - predictions[right]
            correct = target_delta * prediction_delta > 0.0
            eligible += 1
            if len(sampled) < maximum_pairs:
                sampled.append(correct)
                continue
            replacement = rng.randrange(eligible)
            if replacement < maximum_pairs:
                sampled[replacement] = correct
    return (
        sum(sampled) / len(sampled) if sampled else 0.0,
        len(sampled),
        eligible,
    )


def _metrics(
    targets: list[float],
    predictions: list[float],
    sources: list[str],
) -> dict[str, object]:
    errors = [abs(target - prediction) for target, prediction in zip(targets, predictions, strict=True)]
    ranking, ranking_pairs, ranking_eligible = _ranking(targets, predictions)
    within_source, within_source_pairs, within_source_eligible = _ranking(
        targets,
        predictions,
        sources,
    )
    return {
        "mae_ms": sum(errors) / len(errors),
        "p90_abs_error_ms": _percentile(errors, 90),
        "ranking_accuracy": ranking,
        "ranking_pairs": ranking_pairs,
        "ranking_eligible_pairs": ranking_eligible,
        "ranking_sampling": "uniform_reservoir_seed_0",
        "within_source_ranking_accuracy": within_source,
        "within_source_ranking_pairs": within_source_pairs,
        "within_source_ranking_eligible_pairs": within_source_eligible,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    deployment_field = str(
        getattr(args, "deployment_field", "deployment_validation")
    )
    if not re.fullmatch(r"[a-z][a-z0-9_]*", deployment_field):
        raise ValueError(f"invalid deployment validation field: {deployment_field}")
    allow_return_logits = bool(getattr(args, "allow_return_logits", False))
    paths = _paths(args.profiles)
    if not paths:
        raise SystemExit("no shadow profile JSON files matched")
    artifact = None
    expected_model_id = ""
    source_replay_model = None
    if args.artifact:
        artifact_path = Path(args.artifact)
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        if not isinstance(artifact, dict):
            raise ValueError("verify cost artifact must be a JSON object")
        expected_model_id = str(artifact.get("model_id", "") or "")
        if not expected_model_id:
            raise ValueError("verify cost artifact lacks model_id")
    replay_base_path = str(getattr(args, "replay_base_artifact", "") or "")
    replay_base_model = None
    if replay_base_path:
        if artifact is None:
            raise ValueError("--replay-base-artifact requires --artifact")
        source_replay_model = VerifyTimeCostModel(artifact)
        replay_base_model = VerifyTimeCostModel.load(replay_base_path)
        source_shape = (
            source_replay_model.num_layers,
            source_replay_model.num_experts,
            source_replay_model.top_k,
            source_replay_model.buckets,
        )
        replay_shape = (
            replay_base_model.num_layers,
            replay_base_model.num_experts,
            replay_base_model.top_k,
            replay_base_model.buckets,
        )
        if replay_shape != source_shape:
            raise ValueError(
                f"replay base shape/buckets {replay_shape} != source {source_shape}"
            )
    samples = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        measurement = data.get("verify_cost_measurement")
        if not isinstance(measurement, dict):
            raise ValueError(f"{path}: missing verify_cost_measurement metadata")
        if bool(measurement.get("enabled")):
            raise ValueError(f"{path}: verify workload instrumentation must be disabled")
        if measurement.get("target") != "spec.verify_accept_ready_ms":
            raise ValueError(f"{path}: unexpected verify target")
        if bool(measurement.get("profile_cuda_sync")):
            raise ValueError(f"{path}: synchronous CUDA profiling must be disabled")
        records = data.get("model_verify_call_records", data.get("verify_call_records", []))
        records_by_call = {
            int(record.get("call_index", index)): record
            for index, record in enumerate(records)
            if isinstance(record, dict)
        }
        traces = data.get("spec_step_traces", data.get("model_spec_step_traces", []))
        for ordinal, trace in enumerate(traces):
            if ordinal < int(args.drop_first_calls) or not isinstance(trace, dict):
                continue
            if "verify_cost_prediction_ms" not in trace:
                continue
            call_index = int(trace.get("verify_call_index", ordinal))
            record = records_by_call.get(call_index, {})
            if not record:
                raise ValueError(f"{path}: trace call {call_index} has no verify record")
            execution_maps = (
                record.get("metadata_layer_execution_cpu_route_counts"),
                record.get("metadata_layer_execution_route_counts"),
                record.get("metadata_layer_execution_cpu_routes"),
                record.get("metadata_layer_execution_cpu_experts"),
            )
            if (
                bool(record.get("metadata_execution_available"))
                or float(record.get("metadata_execution_cpu_routes_sum", 0.0) or 0.0)
                != 0.0
                or float(record.get("metadata_execution_active_routes_sum", 0.0) or 0.0)
                != 0.0
                or any(bool(value) for value in execution_maps)
            ):
                raise ValueError(
                    f"{path}: call {call_index} contains execution workload in shadow"
                )
            if str(record.get("verify_cost_model_mode", "")) != "shadow":
                raise ValueError(f"{path}: call {call_index} did not run in shadow mode")
            if not bool(record.get("outputs_ready")):
                sampling_sequences = trace.get("sequences", [])
                sampling_ready = (
                    allow_return_logits
                    and bool(record.get("return_logits"))
                    and float(trace.get("verify_accept_ms", 0.0) or 0.0) > 0.0
                    and isinstance(sampling_sequences, list)
                    and bool(sampling_sequences)
                    and all(
                        isinstance(sequence, dict)
                        and sequence.get("acceptance_mode") == "standard_sampling"
                        and isinstance(sequence.get("next_token"), int)
                        for sequence in sampling_sequences
                    )
                )
                if not sampling_ready:
                    raise ValueError(
                        f"{path}: call {call_index} outputs were not host-ready"
                    )
            if not bool(record.get("stream_ms_available")):
                raise ValueError(f"{path}: call {call_index} lacks stream timing")
            target = float(trace.get("verify_accept_ready_ms", 0.0) or 0.0)
            prediction = float(trace.get("verify_cost_prediction_ms", 0.0) or 0.0)
            if not all(math.isfinite(value) and value > 0.0 for value in (target, prediction)):
                continue
            predicted_bucket = None
            prediction_rows = trace.get("verify_cost_predictions", [])
            if isinstance(prediction_rows, list) and prediction_rows:
                final_prediction = prediction_rows[-1]
                predicted_bucket = int(final_prediction.get("verify_cost_bucket", -1))
                actual_model_id = str(
                    final_prediction.get("verify_cost_model_id", "") or ""
                )
                if expected_model_id and actual_model_id != expected_model_id:
                    raise ValueError(
                        f"{path}: call {call_index} model_id {actual_model_id!r} "
                        f"!= artifact {expected_model_id!r}"
                    )
                logical_tokens = int(record.get("token_count", -1))
                predicted_logical_tokens = int(
                    final_prediction.get("verify_cost_logical_tokens", -1)
                )
                if predicted_logical_tokens != logical_tokens:
                    raise ValueError(
                        f"{path}: call {call_index} proxy logical tokens "
                        f"{predicted_logical_tokens} != actual {logical_tokens}"
                    )
            else:
                raise ValueError(f"{path}: call {call_index} lacks prediction details")
            actual_bucket = int(record.get("bucket", predicted_bucket or -1))
            if predicted_bucket is not None and predicted_bucket != actual_bucket:
                raise ValueError(
                    f"{path}: call {call_index} proxy bucket {predicted_bucket} "
                    f"!= actual {actual_bucket}"
                )
            sample = {
                    "source": str(path),
                    "call_index": call_index,
                    "bucket": actual_bucket,
                    "target_ms": target,
                    "prediction_ms": prediction,
                    "abs_error_ms": abs(target - prediction),
                    "stream_ms": float(record.get("stream_ms", 0.0) or 0.0),
                    "model_call_ms": float(
                        trace.get("verify_model_call_ms", 0.0) or 0.0
                    ),
                    "accept_ms": float(trace.get("verify_accept_ms", 0.0) or 0.0),
                }
            if replay_base_model is not None and source_replay_model is not None:
                cpu_routes = float(final_prediction["verify_cost_cpu_routes"])
                cpu_experts = float(final_prediction["verify_cost_cpu_experts"])
                source_replay = source_replay_model.predict_cpu_counts(
                    bucket=actual_bucket,
                    logical_tokens=logical_tokens,
                    cpu_routes=cpu_routes,
                    cpu_experts=cpu_experts,
                )
                replay = replay_base_model.predict_cpu_counts(
                    bucket=actual_bucket,
                    logical_tokens=logical_tokens,
                    cpu_routes=cpu_routes,
                    cpu_experts=cpu_experts,
                )
                sample["source_replay_prediction_ms"] = float(
                    source_replay.total_ms
                )
                sample["replay_prediction_ms"] = float(replay.total_ms)
            samples.append(sample)
    if len(samples) < int(args.minimum_samples):
        raise SystemExit(
            f"only {len(samples)} shadow samples; need {args.minimum_samples}"
        )
    targets = [float(sample["target_ms"]) for sample in samples]
    predictions = [float(sample["prediction_ms"]) for sample in samples]
    metrics = _metrics(
        targets,
        predictions,
        [str(sample["source"]) for sample in samples],
    )
    by_bucket = {}
    bucket_gate_passed = True
    for bucket in sorted({int(sample["bucket"]) for sample in samples}):
        bucket_samples = [sample for sample in samples if int(sample["bucket"]) == bucket]
        bucket_metrics = _metrics(
            [float(sample["target_ms"]) for sample in bucket_samples],
            [float(sample["prediction_ms"]) for sample in bucket_samples],
            [str(sample["source"]) for sample in bucket_samples],
        )
        enough = len(bucket_samples) >= int(args.minimum_bucket_samples)
        bucket_passed = (
            enough
            and float(bucket_metrics["mae_ms"]) <= float(args.max_bucket_mae_ms)
            and float(bucket_metrics["p90_abs_error_ms"])
            <= float(args.max_bucket_p90_ms)
        )
        bucket_gate_passed = bucket_gate_passed and bucket_passed
        by_bucket[str(bucket)] = {
            "sample_count": len(bucket_samples),
            "metrics": bucket_metrics,
            "passed": bucket_passed,
        }
    passed = bucket_gate_passed and (
        metrics["mae_ms"] <= float(args.max_mae_ms)
        and metrics["p90_abs_error_ms"] <= float(args.max_p90_ms)
        and metrics["ranking_accuracy"] >= float(args.min_ranking_accuracy)
    )
    validation = {
        "passed": passed,
        "gate_version": str(getattr(args, "gate_version", "v1")),
        "mode": "uninstrumented_shadow",
        "model_id": expected_model_id,
        "protocol": str(getattr(args, "protocol", "unspecified")),
        "sample_count": len(samples),
        "metrics": metrics,
        "by_bucket": by_bucket,
        "latency_boundary": {
            "return_logits_accepted": allow_return_logits,
            "accept_ready_ms_mean": sum(targets) / len(targets),
            "stream_ms_mean": sum(float(sample["stream_ms"]) for sample in samples)
            / len(samples),
            "accept_ready_minus_stream_ms_mean": sum(
                float(sample["target_ms"]) - float(sample["stream_ms"])
                for sample in samples
            )
            / len(samples),
            "model_call_ms_mean": sum(
                float(sample["model_call_ms"]) for sample in samples
            )
            / len(samples),
            "accept_ms_mean": sum(float(sample["accept_ms"]) for sample in samples)
            / len(samples),
        },
        "gate": {
            "max_mae_ms": float(args.max_mae_ms),
            "max_p90_abs_error_ms": float(args.max_p90_ms),
            "min_ranking_accuracy": float(args.min_ranking_accuracy),
            "minimum_bucket_samples": int(args.minimum_bucket_samples),
            "max_bucket_mae_ms": float(args.max_bucket_mae_ms),
            "max_bucket_p90_abs_error_ms": float(args.max_bucket_p90_ms),
        },
        "profiles": [
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in paths
        ],
    }
    replay_validation = None
    if replay_base_model is not None and source_replay_model is not None:
        replay_predictions = [
            float(sample["replay_prediction_ms"]) for sample in samples
        ]
        replay_metrics = _metrics(targets, replay_predictions, [
            str(sample["source"]) for sample in samples
        ])
        replay_by_bucket = {}
        replay_bucket_gate_passed = True
        for bucket in sorted({int(sample["bucket"]) for sample in samples}):
            bucket_samples = [
                sample for sample in samples if int(sample["bucket"]) == bucket
            ]
            bucket_metrics = _metrics(
                [float(sample["target_ms"]) for sample in bucket_samples],
                [float(sample["replay_prediction_ms"]) for sample in bucket_samples],
                [str(sample["source"]) for sample in bucket_samples],
            )
            enough = len(bucket_samples) >= int(args.minimum_bucket_samples)
            bucket_passed = (
                enough
                and float(bucket_metrics["mae_ms"])
                <= float(args.max_bucket_mae_ms)
                and float(bucket_metrics["p90_abs_error_ms"])
                <= float(args.max_bucket_p90_ms)
            )
            replay_bucket_gate_passed = replay_bucket_gate_passed and bucket_passed
            replay_by_bucket[str(bucket)] = {
                "sample_count": len(bucket_samples),
                "metrics": bucket_metrics,
                "passed": bucket_passed,
            }
        source_replay_max_delta = max(
            abs(
                float(sample["source_replay_prediction_ms"])
                - float(sample["prediction_ms"])
            )
            for sample in samples
        )
        replay_passed = replay_bucket_gate_passed and (
            float(replay_metrics["mae_ms"]) <= float(args.max_mae_ms)
            and float(replay_metrics["p90_abs_error_ms"])
            <= float(args.max_p90_ms)
            and float(replay_metrics["ranking_accuracy"])
            >= float(args.min_ranking_accuracy)
        )
        replay_validation = {
            "mode": "frozen_proxy_counts_alternate_base_replay",
            "source_pipeline_model_id": source_replay_model.model_id,
            "replay_base_model_id": replay_base_model.model_id,
            "sample_count": len(samples),
            "source_replay_max_abs_delta_ms": source_replay_max_delta,
            "metrics": replay_metrics,
            "by_bucket": replay_by_bucket,
            "passed_fixed_gates": replay_passed,
            "deployment_validation_transferred": False,
            "note": (
                "Diagnostic replay of stored causal CPU-count predictions; "
                "it is not an instrumentation-off run of the replay model id."
            ),
        }
    output = {
        "validation_field": deployment_field,
        "deployment_validation": validation,
        "samples": samples,
    }
    if replay_validation is not None:
        output["alternate_base_replay"] = replay_validation
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    if args.artifact:
        artifact_path = Path(args.artifact)
        assert artifact is not None
        artifact[deployment_field] = validation
        destination = Path(args.output_artifact or artifact_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(validation, indent=2))
    if not passed:
        raise SystemExit(1)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", nargs="+", required=True)
    parser.add_argument("--artifact", default="")
    parser.add_argument("--replay-base-artifact", default="")
    parser.add_argument("--output-artifact", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--drop-first-calls", type=int, default=2)
    parser.add_argument("--minimum-samples", type=int, default=20)
    parser.add_argument("--max-mae-ms", type=float, default=5.0)
    parser.add_argument("--max-p90-ms", type=float, default=10.0)
    parser.add_argument("--min-ranking-accuracy", type=float, default=0.9)
    parser.add_argument("--minimum-bucket-samples", type=int, default=5)
    parser.add_argument("--max-bucket-mae-ms", type=float, default=7.5)
    parser.add_argument("--max-bucket-p90-ms", type=float, default=12.5)
    parser.add_argument("--gate-version", default="v1")
    parser.add_argument("--protocol", default="unspecified")
    parser.add_argument(
        "--allow-return-logits",
        action="store_true",
        help=(
            "Accept sampling calls whose GPU logits become host-observable only "
            "at the recorded acceptance-ready boundary."
        ),
    )
    parser.add_argument(
        "--deployment-field",
        default="deployment_validation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
