#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanovllm.engine.speculative.verify_cost_model import (  # noqa: E402
    SCHEMA_VERSION,
    VerifyCpuWorkload,
    compute_model_id,
    feature_values,
)


@dataclass
class Sample:
    source: str
    group: str
    call_index: int
    workload: VerifyCpuWorkload
    target_ms: float
    stream_ms: float | None
    cpuinfer_sync_ms: float | None
    layer_route_counts: tuple[tuple[float, ...], ...] | None


def _expand_inputs(values: Iterable[str]) -> list[Path]:
    paths: set[Path] = set()
    for value in values:
        path = Path(value)
        if path.is_dir():
            paths.update(path.rglob("sample*.json"))
            continue
        matches = [Path(item) for item in glob.glob(value, recursive=True)]
        if matches:
            paths.update(item for item in matches if item.is_file())
        elif path.is_file():
            paths.add(path)
    return sorted(path.resolve() for path in paths)


def _profile_list(data: dict, key: str) -> list:
    value = data.get(f"model_{key}", data.get(key, []))
    return value if isinstance(value, list) else []


def _load_samples(
    path: Path,
    *,
    num_layers: int,
    num_experts: int,
    drop_first_calls: int,
    require_measurement_metadata: bool,
) -> list[Sample]:
    data = json.loads(path.read_text(encoding="utf-8"))
    measurement = data.get("verify_cost_measurement")
    if require_measurement_metadata:
        if not isinstance(measurement, dict) or not bool(measurement.get("enabled")):
            raise ValueError(f"{path}: missing enabled verify_cost_measurement metadata")
        if measurement.get("target") != "spec.verify_accept_ready_ms":
            raise ValueError(f"{path}: unsupported verify cost target metadata")
        if bool(measurement.get("profile_cuda_sync")):
            raise ValueError(f"{path}: profile_cuda_sync perturbs the measured path")
    records = _profile_list(data, "verify_call_records")
    traces = _profile_list(data, "spec_step_traces")
    trace_by_call = {
        int(trace.get("verify_call_index", index)): trace
        for index, trace in enumerate(traces)
        if isinstance(trace, dict)
    }
    sync_by_step: dict[int, float] = {}
    for row in _profile_list(data, "verify_op_event_records"):
        if not isinstance(row, dict) or row.get("label") != "kt.cpuinfer_sync":
            continue
        step_id = int(row.get("step_id", -1))
        sync_by_step[step_id] = sync_by_step.get(step_id, 0.0) + float(
            row.get("elapsed_ms", 0.0) or 0.0
        )

    samples: list[Sample] = []
    for ordinal, record in enumerate(records):
        if ordinal < int(drop_first_calls) or not isinstance(record, dict):
            continue
        route_counts = record.get("metadata_layer_execution_cpu_route_counts")
        if not isinstance(route_counts, dict):
            continue
        if require_measurement_metadata:
            expected_layers = {str(index) for index in range(int(num_layers))}
            if set(route_counts) != expected_layers:
                raise ValueError(
                    f"{path}: call {ordinal} has incomplete execution layers "
                    f"({len(route_counts)} != {num_layers})"
                )
            if not bool(record.get("metadata_execution_available")):
                raise ValueError(f"{path}: call {ordinal} lacks execution metadata")
            if not bool(record.get("stream_ms_available")):
                raise ValueError(f"{path}: call {ordinal} lacks verify stream timing")
        workload = VerifyCpuWorkload.from_mapping(
            bucket=int(record.get("bucket", record.get("token_count", 0)) or 0),
            logical_tokens=int(record.get("token_count", 0) or 0),
            layer_route_counts=route_counts,
            num_layers=num_layers,
            num_experts=num_experts,
        )
        all_route_counts = record.get("metadata_layer_execution_route_counts")
        all_workload = None
        if isinstance(all_route_counts, dict):
            all_workload = VerifyCpuWorkload.from_mapping(
                bucket=int(record.get("bucket", record.get("token_count", 0)) or 0),
                logical_tokens=int(record.get("token_count", 0) or 0),
                layer_route_counts=all_route_counts,
                num_layers=num_layers,
                num_experts=num_experts,
            )
        elif require_measurement_metadata:
            raise ValueError(f"{path}: call {ordinal} lacks full execution route counts")
        call_index = int(record.get("call_index", ordinal) or ordinal)
        trace = trace_by_call.get(call_index)
        if trace is None or "verify_accept_ready_ms" not in trace:
            if require_measurement_metadata:
                raise ValueError(
                    f"{path}: call {call_index} has no joined acceptance-ready trace"
                )
            target_ms = float(record.get("total_ms", 0.0) or 0.0)
        else:
            target_ms = float(trace["verify_accept_ready_ms"])
        if not math.isfinite(target_ms) or target_ms <= 0.0:
            continue
        stream_ms = (
            float(record["stream_ms"])
            if bool(record.get("stream_ms_available"))
            else None
        )
        step_id = int(record.get("step_id", -1))
        samples.append(
            Sample(
                source=str(path),
                group=str(path),
                call_index=call_index,
                workload=workload,
                target_ms=target_ms,
                stream_ms=stream_ms,
                cpuinfer_sync_ms=sync_by_step.get(step_id),
                layer_route_counts=(
                    all_workload.layer_route_counts if all_workload is not None else None
                ),
            )
        )
    return samples


def _make_groups(samples: list[Sample], minimum_groups: int = 5) -> None:
    unique = sorted({sample.group for sample in samples})
    if len(unique) >= minimum_groups:
        return
    by_source: dict[str, list[Sample]] = {}
    for sample in samples:
        by_source.setdefault(sample.source, []).append(sample)
    for source, rows in by_source.items():
        rows.sort(key=lambda sample: sample.call_index)
        chunk_count = min(minimum_groups, len(rows))
        for chunk_index, indices in enumerate(np.array_split(np.arange(len(rows)), chunk_count)):
            for index in indices.tolist():
                rows[index].group = f"{source}#chunk{chunk_index}"


def _feature_sets(buckets: list[int], num_layers: int) -> dict[str, list[str]]:
    # Use the first bucket as the reference category. A full one-hot vector plus
    # an intercept is singular, and including both logical and padding tokens is
    # also redundant because bucket == logical + padding.
    bucket_features = [f"bucket_{bucket}" for bucket in buckets[1:]]
    base = bucket_features + ["logical_tokens"]
    global_counts = base + ["cpu_routes", "cpu_experts"]
    layer_counts = global_counts + [
        feature
        for layer_idx in range(int(num_layers))
        for feature in (
            f"layer_{layer_idx}_cpu_routes",
            f"layer_{layer_idx}_cpu_experts",
        )
    ]
    shape = layer_counts + [
        "route_sq_sum",
        "max_route_per_expert_sum",
        "singleton_experts",
        "multi_route_experts",
        "nonempty_layers",
        "max_layer_routes",
        "std_layer_routes",
        "cpu_routes_x_bucket",
        "cpu_experts_x_bucket",
    ]
    return {
        "bucket_only": base,
        "global_counts": global_counts,
        "layer_counts": layer_counts,
        "layer_route_shape": shape,
    }


def _matrix(samples: list[Sample], names: list[str]) -> np.ndarray:
    return np.asarray(
        [feature_values(sample.workload, names) for sample in samples],
        dtype=np.float64,
    )


def _fit(
    x: np.ndarray,
    y: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    normalized = (x - mean) / scale
    design = np.column_stack([np.ones(len(x)), normalized])
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    system = design.T @ design + penalty
    target = design.T @ y
    try:
        coefficients = np.linalg.solve(system, target)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(system, target, rcond=None)[0]
    return mean, scale, coefficients[1:], float(coefficients[0])


def _design_diagnostics(x: np.ndarray) -> dict[str, object]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    return {
        "row_count": int(design.shape[0]),
        "column_count": int(design.shape[1]),
        "rank": int(np.linalg.matrix_rank(design)),
        "condition_number": float(np.linalg.cond(design)),
        "full_column_rank": bool(
            np.linalg.matrix_rank(design) == design.shape[1]
        ),
    }


def _predict(
    x: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    coefficients: np.ndarray,
    intercept: float,
) -> np.ndarray:
    return float(intercept) + ((x - mean) / scale) @ coefficients


def _ranking_accuracy(
    y: np.ndarray,
    prediction: np.ndarray,
    groups: list[str] | None = None,
    *,
    maximum_pairs: int = 100_000,
) -> tuple[float, int, int]:
    rng = random.Random(0)
    sampled: list[bool] = []
    eligible_pairs = 0
    for left in range(len(y)):
        for right in range(left + 1, len(y)):
            if groups is not None and groups[left] != groups[right]:
                continue
            true_delta = float(y[left] - y[right])
            if abs(true_delta) < 1.0:
                continue
            predicted_delta = float(prediction[left] - prediction[right])
            correct = true_delta * predicted_delta > 0.0
            eligible_pairs += 1
            if len(sampled) < maximum_pairs:
                sampled.append(correct)
                continue
            replacement = rng.randrange(eligible_pairs)
            if replacement < maximum_pairs:
                sampled[replacement] = correct
    accuracy = float(sum(sampled) / len(sampled)) if sampled else 0.0
    return accuracy, len(sampled), eligible_pairs


def _metrics(y: np.ndarray, prediction: np.ndarray, groups: list[str]) -> dict[str, object]:
    error = np.abs(y - prediction)
    denominator = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - prediction) ** 2)) / denominator if denominator > 0 else 0.0
    ranking, ranking_pairs, ranking_eligible = _ranking_accuracy(y, prediction)
    within_group, within_group_pairs, within_group_eligible = _ranking_accuracy(
        y,
        prediction,
        groups,
    )
    return {
        "mae_ms": float(error.mean()),
        "p90_abs_error_ms": float(np.percentile(error, 90)),
        "r2": r2,
        "ranking_accuracy": ranking,
        "ranking_pairs": int(ranking_pairs),
        "ranking_eligible_pairs": int(ranking_eligible),
        "ranking_sampling": "uniform_reservoir_seed_0",
        "within_group_ranking_accuracy": within_group,
        "within_group_ranking_pairs": int(within_group_pairs),
        "within_group_ranking_eligible_pairs": int(within_group_eligible),
    }


def _cross_validated_mae(
    samples: list[Sample],
    names: list[str],
    ridge: float,
) -> float:
    groups = sorted({sample.group for sample in samples})
    if len(groups) < 2:
        return float("inf")
    predictions = np.empty(len(samples), dtype=np.float64)
    y = np.asarray([sample.target_ms for sample in samples], dtype=np.float64)
    x = _matrix(samples, names)
    for group in groups:
        test = np.asarray([sample.group == group for sample in samples], dtype=bool)
        train = ~test
        mean, scale, coefficients, intercept = _fit(x[train], y[train], ridge)
        predictions[test] = _predict(x[test], mean, scale, coefficients, intercept)
    return float(np.mean(np.abs(y - predictions)))


def _fingerprint(args: argparse.Namespace) -> dict[str, object]:
    cpu_model = platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    gpu_model = "unknown"
    try:
        gpu_model = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
                "--id=0",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    try:
        kt_version = importlib.metadata.version("kt-kernel")
    except importlib.metadata.PackageNotFoundError:
        kt_version = "unknown"
    return {
        "cpu_model": cpu_model,
        "gpu_model": gpu_model,
        "kt_kernel_version": kt_version,
        "kt_num_threads": int(args.kt_num_threads),
        "kt_backend": str(args.kt_backend),
    }


def _manifest(paths: list[Path]) -> list[dict[str, object]]:
    rows = []
    for path in paths:
        payload = path.read_bytes()
        rows.append(
            {
                "path": str(path),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return rows


def _bucket_priors(
    samples: list[Sample],
    buckets: list[int],
    num_layers: int,
    num_experts: int,
) -> dict[str, list[list[float]]]:
    priors: dict[str, list[list[float]]] = {}
    for bucket in buckets:
        rows = [sample for sample in samples if sample.workload.bucket == bucket]
        if not rows:
            continue
        accumulator = np.zeros((num_layers, num_experts), dtype=np.float64)
        usable = [sample for sample in rows if sample.layer_route_counts is not None]
        for sample in usable:
            accumulator += np.asarray(sample.layer_route_counts, dtype=np.float64)
        if not usable:
            continue
        accumulator /= float(len(usable) * max(1, bucket))
        priors[str(bucket)] = np.round(accumulator, 6).tolist()
    return priors


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.std() < 1e-9 or right_array.std() < 1e-9:
        return None
    return float(np.corrcoef(left_array, right_array)[0, 1])


def _breakdown(samples: list[Sample]) -> dict[str, object]:
    target = [sample.target_ms for sample in samples]
    stream_pairs = [
        (sample.target_ms, sample.stream_ms)
        for sample in samples
        if sample.stream_ms is not None
    ]
    sync_pairs = [
        (sample.target_ms, sample.cpuinfer_sync_ms)
        for sample in samples
        if sample.cpuinfer_sync_ms is not None
    ]
    stream = [float(pair[1]) for pair in stream_pairs]
    stream_target = [float(pair[0]) for pair in stream_pairs]
    sync = [float(pair[1]) for pair in sync_pairs]
    sync_target = [float(pair[0]) for pair in sync_pairs]
    by_bucket: dict[str, dict[str, object]] = {}
    for bucket in sorted({sample.workload.bucket for sample in samples}):
        rows = [sample for sample in samples if sample.workload.bucket == bucket]
        bucket_target = [sample.target_ms for sample in rows]
        bucket_stream = [
            float(sample.stream_ms)
            for sample in rows
            if sample.stream_ms is not None
        ]
        routes = [
            float(sum(sum(layer) for layer in sample.workload.layer_route_counts))
            for sample in rows
        ]
        experts = [
            float(
                sum(
                    sum(float(value) > 0.0 for value in layer)
                    for layer in sample.workload.layer_route_counts
                )
            )
            for sample in rows
        ]
        by_bucket[str(bucket)] = {
            "sample_count": len(rows),
            "accept_ready_ms_mean": float(np.mean(bucket_target)),
            "accept_ready_ms_std": float(np.std(bucket_target)),
            "stream_ms_mean": float(np.mean(bucket_stream)) if bucket_stream else None,
            "execution_cpu_routes_mean": float(np.mean(routes)),
            "execution_cpu_experts_mean": float(np.mean(experts)),
            "cpu_routes_vs_accept_ready_correlation": _correlation(
                routes, bucket_target
            ),
            "cpu_experts_vs_accept_ready_correlation": _correlation(
                experts, bucket_target
            ),
        }
    return {
        "sample_count": len(samples),
        "accept_ready_ms_mean": float(np.mean(target)),
        "accept_ready_ms_p90": float(np.percentile(target, 90)),
        "stream_timing_sample_count": len(stream_pairs),
        "stream_ms_mean": float(np.mean(stream)) if stream else None,
        "accept_ready_minus_stream_ms_mean": (
            float(np.mean(np.asarray(stream_target) - np.asarray(stream)))
            if stream
            else None
        ),
        "stream_vs_accept_ready_correlation": _correlation(stream_target, stream),
        "cpuinfer_sync_sample_count": len(sync_pairs),
        "cpuinfer_sync_ms_mean": float(np.mean(sync)) if sync else None,
        "cpuinfer_sync_vs_accept_ready_correlation": _correlation(sync_target, sync),
        "by_bucket": by_bucket,
    }


def _model_decomposition(
    holdout: list[Sample],
    selected: dict[str, object],
    candidate_results: list[dict[str, object]],
    num_layers: int,
    num_experts: int,
    minimum_ms: float,
) -> dict[str, object]:
    feature_names = list(selected["feature_names"])
    mean = np.asarray(selected["feature_mean"])
    scale = np.asarray(selected["feature_scale"])
    coefficients = np.asarray(selected["coefficients"])
    intercept = float(selected["intercept"])
    fixed_x = np.asarray(
        [
            feature_values(
                VerifyCpuWorkload(
                    bucket=sample.workload.bucket,
                    logical_tokens=sample.workload.logical_tokens,
                    layer_route_counts=tuple(
                        (0.0,) * num_experts for _ in range(num_layers)
                    ),
                ),
                feature_names,
            )
            for sample in holdout
        ],
        dtype=np.float64,
    )
    total_prediction = np.maximum(
        float(minimum_ms),
        _predict(_matrix(holdout, feature_names), mean, scale, coefficients, intercept),
    )
    fixed_prediction = np.maximum(
        float(minimum_ms),
        _predict(fixed_x, mean, scale, coefficients, intercept),
    )
    exposed_cpu = np.maximum(0.0, total_prediction - fixed_prediction)
    bucket_baseline = next(
        result for result in candidate_results if result["name"] == "bucket_only"
    )
    by_bucket = {}
    for bucket in sorted({sample.workload.bucket for sample in holdout}):
        indices = [
            index
            for index, sample in enumerate(holdout)
            if sample.workload.bucket == bucket
        ]
        by_bucket[str(bucket)] = {
            "sample_count": len(indices),
            "fixed_ms_mean": float(np.mean(fixed_prediction[indices])),
            "exposed_cpu_ms_mean": float(np.mean(exposed_cpu[indices])),
            "predicted_total_ms_mean": float(np.mean(total_prediction[indices])),
            "actual_total_ms_mean": float(
                np.mean([holdout[index].target_ms for index in indices])
            ),
        }
    return {
        "bucket_only_mae_ms": float(bucket_baseline["metrics"]["mae_ms"]),
        "selected_mae_ms": float(selected["metrics"]["mae_ms"]),
        "mae_reduction_vs_bucket_only_ms": float(
            bucket_baseline["metrics"]["mae_ms"]
            - selected["metrics"]["mae_ms"]
        ),
        "mae_reduction_vs_bucket_only_ratio": float(
            1.0
            - selected["metrics"]["mae_ms"]
            / max(1e-9, bucket_baseline["metrics"]["mae_ms"])
        ),
        "fixed_ms_mean": float(np.mean(fixed_prediction)),
        "exposed_cpu_ms_mean": float(np.mean(exposed_cpu)),
        "exposed_cpu_vs_actual_correlation": _correlation(
            exposed_cpu.tolist(),
            [sample.target_ms for sample in holdout],
        ),
        "by_bucket": by_bucket,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    paths = _expand_inputs(args.profiles)
    if not paths:
        raise SystemExit("no profile JSON files matched --profiles")
    samples: list[Sample] = []
    for path in paths:
        samples.extend(
            _load_samples(
                path,
                num_layers=int(args.num_layers),
                num_experts=int(args.num_experts),
                drop_first_calls=int(args.drop_first_calls),
                require_measurement_metadata=bool(args.require_measurement_metadata),
            )
        )
    if len(samples) < 20:
        raise SystemExit(
            f"only {len(samples)} valid execution-workload samples found; need at least 20"
        )
    _make_groups(samples)
    groups = sorted({sample.group for sample in samples})
    rng = random.Random(int(args.seed))
    rng.shuffle(groups)
    holdout_count = max(1, int(round(len(groups) * float(args.holdout_fraction))))
    holdout_groups = set(groups[:holdout_count])
    train = [sample for sample in samples if sample.group not in holdout_groups]
    holdout = [sample for sample in samples if sample.group in holdout_groups]
    if len(train) < 10 or len(holdout) < 5:
        raise SystemExit(
            f"insufficient grouped split: train={len(train)} holdout={len(holdout)}"
        )

    buckets = sorted({sample.workload.bucket for sample in samples})
    minimum_ms = float(
        max(0.0, np.percentile([sample.target_ms for sample in train], 1) * 0.8)
    )
    candidates = _feature_sets(buckets, int(args.num_layers))
    ridge_values = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]
    results: list[dict[str, object]] = []
    for name, feature_names in candidates.items():
        ridge = min(
            ridge_values,
            key=lambda value: _cross_validated_mae(train, feature_names, value),
        )
        train_x = _matrix(train, feature_names)
        train_y = np.asarray([sample.target_ms for sample in train], dtype=np.float64)
        mean, scale, coefficients, intercept = _fit(train_x, train_y, ridge)
        holdout_x = _matrix(holdout, feature_names)
        holdout_y = np.asarray([sample.target_ms for sample in holdout], dtype=np.float64)
        prediction = np.maximum(
            minimum_ms,
            _predict(holdout_x, mean, scale, coefficients, intercept),
        )
        metrics = _metrics(
            holdout_y,
            prediction,
            [sample.group for sample in holdout],
        )
        passed = (
            metrics["mae_ms"] <= float(args.max_mae_ms)
            and metrics["p90_abs_error_ms"] <= float(args.max_p90_ms)
            and metrics["ranking_accuracy"] >= float(args.min_ranking_accuracy)
        )
        results.append(
            {
                "name": name,
                "feature_names": feature_names,
                "ridge": ridge,
                "feature_mean": mean,
                "feature_scale": scale,
                "coefficients": coefficients,
                "intercept": intercept,
                "metrics": metrics,
                "passed": passed,
            }
        )

    passing = [
        result
        for result in results
        if result["passed"] and result["name"] != "bucket_only"
    ]
    cpu_candidates = [
        result for result in results if result["name"] != "bucket_only"
    ]
    if passing:
        selected = min(
            passing,
            key=lambda result: (
                len(result["feature_names"]),
                result["metrics"]["mae_ms"],
            ),
        )
    else:
        selected = min(
            cpu_candidates,
            key=lambda result: result["metrics"]["mae_ms"],
        )

    deployment_feature_names = list(selected["feature_names"])
    deployment_x = _matrix(samples, deployment_feature_names)
    deployment_y = np.asarray(
        [sample.target_ms for sample in samples],
        dtype=np.float64,
    )
    (
        deployment_mean,
        deployment_scale,
        deployment_coefficients,
        deployment_intercept,
    ) = _fit(deployment_x, deployment_y, float(selected["ridge"]))
    deployment_minimum_ms = float(
        max(0.0, np.percentile(deployment_y, 1) * 0.8)
    )

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "target": "verify_accept_ready_ms",
        "model_kind": str(selected["name"]),
        "requires_cpu_workload": True,
        "num_layers": int(args.num_layers),
        "num_experts": int(args.num_experts),
        "top_k": int(args.top_k),
        "buckets": buckets,
        "feature_names": deployment_feature_names,
        "feature_mean": deployment_mean.tolist(),
        "feature_scale": deployment_scale.tolist(),
        "coefficients": deployment_coefficients.tolist(),
        "intercept": float(deployment_intercept),
        "minimum_ms": deployment_minimum_ms,
        "validation_metrics": dict(selected["metrics"]),
        "accuracy_gate_passed": bool(selected["passed"]),
        "accuracy_gate": {
            "max_mae_ms": float(args.max_mae_ms),
            "max_p90_abs_error_ms": float(args.max_p90_ms),
            "min_ranking_accuracy": float(args.min_ranking_accuracy),
        },
        "fingerprint": _fingerprint(args),
        "unknown_row_expert_route_priors": _bucket_priors(
            samples,
            buckets,
            int(args.num_layers),
            int(args.num_experts),
        ),
        "training_manifest": _manifest(paths),
        "analyzer_identity": {
            "path": str(Path(__file__).resolve()),
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "design_diagnostics": _design_diagnostics(deployment_x),
        "measurement_validation": {
            "required_metadata": bool(args.require_measurement_metadata),
            "breakdown": _breakdown(samples),
        },
        "latency_decomposition": _model_decomposition(
            holdout,
            selected,
            results,
            int(args.num_layers),
            int(args.num_experts),
            minimum_ms,
        ),
        "split": {
            "seed": int(args.seed),
            "train_samples": len(train),
            "holdout_samples": len(holdout),
            "holdout_groups": sorted(holdout_groups),
            "deployment_refit_samples": len(samples),
            "validation_minimum_ms": minimum_ms,
        },
    }
    artifact["model_id"] = compute_model_id(artifact)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = output.with_suffix(".md")
    lines = [
        "# Verify Time Cost Model",
        "",
        f"- samples: `{len(samples)}` (train `{len(train)}`, holdout `{len(holdout)}`)",
        f"- buckets: `{buckets}`",
        f"- selected: `{selected['name']}`",
        f"- accuracy gate: `{'PASS' if selected['passed'] else 'FAIL'}`",
        f"- strict measurement metadata: `{bool(args.require_measurement_metadata)}`",
        "",
        "| model | features | ridge | MAE ms | P90 ms | R2 | ranking | gate |",
        "|:---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for result in results:
        metrics = result["metrics"]
        lines.append(
            f"| {result['name']} | {len(result['feature_names'])} | {result['ridge']} | "
            f"{metrics['mae_ms']:.3f} | {metrics['p90_abs_error_ms']:.3f} | "
            f"{metrics['r2']:.3f} | {metrics['ranking_accuracy']:.3f} | "
            f"{'PASS' if result['passed'] else 'FAIL'} |"
        )
    breakdown = artifact["measurement_validation"]["breakdown"]
    decomposition = artifact["latency_decomposition"]
    lines.extend(
        [
            "",
            "## Timing Breakdown",
            "",
            f"- acceptance-ready mean: `{breakdown['accept_ready_ms_mean']:.3f} ms`",
            f"- stream timing samples: `{breakdown['stream_timing_sample_count']}`",
            f"- CPUInfer sync timing samples: `{breakdown['cpuinfer_sync_sample_count']}`",
            f"- selected MAE improvement vs bucket-only: "
            f"`{decomposition['mae_reduction_vs_bucket_only_ms']:.3f} ms` "
            f"(`{decomposition['mae_reduction_vs_bucket_only_ratio']:.1%}`)",
            f"- modeled fixed component mean: `{decomposition['fixed_ms_mean']:.3f} ms`",
            f"- modeled exposed CPU component mean: "
            f"`{decomposition['exposed_cpu_ms_mean']:.3f} ms`",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    predictions_path = output.with_suffix(".holdout.csv")
    selected_x = _matrix(holdout, list(selected["feature_names"]))
    selected_prediction = np.maximum(
        minimum_ms,
        _predict(
            selected_x,
            np.asarray(selected["feature_mean"]),
            np.asarray(selected["feature_scale"]),
            np.asarray(selected["coefficients"]),
            float(selected["intercept"]),
        ),
    )
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["source", "group", "call_index", "bucket", "logical_tokens", "target_ms", "prediction_ms", "abs_error_ms"]
        )
        for sample, prediction in zip(holdout, selected_prediction, strict=True):
            writer.writerow(
                [
                    sample.source,
                    sample.group,
                    sample.call_index,
                    sample.workload.bucket,
                    sample.workload.logical_tokens,
                    f"{sample.target_ms:.6f}",
                    f"{prediction:.6f}",
                    f"{abs(sample.target_ms - prediction):.6f}",
                ]
            )
    print(json.dumps({"artifact": str(output), "report": str(report), "selected": selected["name"], "metrics": selected["metrics"], "passed": selected["passed"]}, indent=2))
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-layers", type=int, default=48)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--drop-first-calls", type=int, default=2)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--max-mae-ms", type=float, default=5.0)
    parser.add_argument("--max-p90-ms", type=float, default=10.0)
    parser.add_argument("--min-ranking-accuracy", type=float, default=0.9)
    parser.add_argument("--kt-num-threads", type=int, default=16)
    parser.add_argument("--kt-backend", default="avx2_bf16")
    parser.add_argument(
        "--require-measurement-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
