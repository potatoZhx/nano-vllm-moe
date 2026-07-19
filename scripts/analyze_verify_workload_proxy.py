#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanovllm.engine.speculative.verify_cost_model import (  # noqa: E402
    VerifyCpuWorkload,
    VerifyTimeCostModel,
    compute_model_id,
)


@dataclass
class Sample:
    source: str
    bucket: int
    logical_tokens: int
    target_ms: float
    actual_cpu_routes: float
    actual_cpu_experts: float
    values: dict[str, float]


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


def _load(path: Path, *, base_model_id: str, drop_first_calls: int) -> list[Sample]:
    data = json.loads(path.read_text(encoding="utf-8"))
    measurement = data.get("verify_cost_measurement")
    if not isinstance(measurement, dict) or not bool(measurement.get("enabled")):
        raise ValueError(f"{path}: proxy calibration requires execution instrumentation")
    records = {
        int(row.get("call_index", index)): row
        for index, row in enumerate(_rows(data, "verify_call_records"))
        if isinstance(row, dict)
    }
    samples = []
    for ordinal, trace in enumerate(_rows(data, "spec_step_traces")):
        if ordinal < drop_first_calls or not isinstance(trace, dict):
            continue
        predictions = trace.get("verify_cost_predictions", [])
        if not isinstance(predictions, list) or not predictions:
            continue
        prediction = predictions[-1]
        call_index = int(trace.get("verify_call_index", ordinal))
        record = records.get(call_index)
        if record is None:
            raise ValueError(f"{path}: trace call {call_index} has no verify record")
        if str(prediction.get("verify_cost_model_id", "")) != base_model_id:
            raise ValueError(f"{path}: proxy calibration used a different base model")
        layer_routes = prediction.get("verify_cost_layer_proxy_cpu_routes")
        layer_experts = prediction.get("verify_cost_layer_proxy_cpu_experts")
        if not isinstance(layer_routes, list) or not isinstance(layer_experts, list):
            raise ValueError(f"{path}: proxy layer features are missing")
        if len(layer_routes) != len(layer_experts):
            raise ValueError(f"{path}: proxy layer feature lengths differ")
        bucket = int(record.get("bucket", 0))
        logical_tokens = int(record.get("token_count", 0))
        predicted_logical = int(prediction.get("verify_cost_logical_tokens", -1))
        if logical_tokens != predicted_logical:
            raise ValueError(f"{path}: proxy/actual logical tokens differ")
        values = {
            "logical_tokens": float(logical_tokens),
            "padding_tokens": float(max(0, bucket - logical_tokens)),
            "known_rows": float(prediction.get("verify_cost_known_rows", 0.0)),
            "unknown_rows": float(prediction.get("verify_cost_unknown_rows", 0.0)),
            "known_row_fraction": float(prediction.get("verify_cost_known_rows", 0.0))
            / float(max(1, bucket)),
            "known_cpu_routes": float(
                prediction.get("verify_cost_known_cpu_routes", 0.0)
            ),
            "prior_cpu_routes": float(
                prediction.get("verify_cost_prior_cpu_routes", 0.0)
            ),
            "proxy_cpu_routes": float(
                prediction.get(
                    "verify_cost_proxy_cpu_routes",
                    prediction.get("verify_cost_cpu_routes", 0.0),
                )
            ),
            "proxy_cpu_experts": float(
                prediction.get(
                    "verify_cost_proxy_cpu_experts",
                    prediction.get("verify_cost_cpu_experts", 0.0),
                )
            ),
            "cached_expert_count": float(
                prediction.get("verify_cost_cached_expert_count", 0.0)
            ),
            "ready_direct_active_experts": float(
                prediction.get("verify_cost_ready_direct_active_experts", 0.0)
            ),
        }
        values["proxy_cpu_routes_per_row"] = values["proxy_cpu_routes"] / float(
            max(1, bucket)
        )
        values["proxy_cpu_experts_per_row"] = values["proxy_cpu_experts"] / float(
            max(1, bucket)
        )
        values["proxy_cpu_routes_sq"] = values["proxy_cpu_routes"] ** 2
        values["proxy_cpu_experts_sq"] = values["proxy_cpu_experts"] ** 2
        values["proxy_routes_x_experts"] = (
            values["proxy_cpu_routes"] * values["proxy_cpu_experts"]
        )
        values["known_routes_sq"] = values["known_cpu_routes"] ** 2
        for candidate_bucket in (3, 5, 7, 10, 13):
            flag = float(bucket == candidate_bucket)
            for base_name in (
                "proxy_cpu_routes",
                "proxy_cpu_experts",
                "known_cpu_routes",
                "prior_cpu_routes",
                "ready_direct_active_experts",
            ):
                values[f"b{candidate_bucket}_x_{base_name}"] = (
                    flag * values[base_name]
                )
        for layer_idx, (routes, experts) in enumerate(
            zip(layer_routes, layer_experts, strict=True)
        ):
            values[f"layer_{layer_idx}_proxy_cpu_routes"] = float(routes)
            values[f"layer_{layer_idx}_proxy_cpu_experts"] = float(experts)
        samples.append(
            Sample(
                source=str(path),
                bucket=bucket,
                logical_tokens=logical_tokens,
                target_ms=float(trace.get("verify_accept_ready_ms", 0.0) or 0.0),
                actual_cpu_routes=float(
                    record.get("metadata_execution_cpu_routes_sum", 0.0) or 0.0
                ),
                actual_cpu_experts=float(
                    record.get("metadata_execution_cpu_experts_sum", 0.0) or 0.0
                ),
                values=values,
            )
        )
    return samples


def _feature_names(buckets: list[int], num_layers: int) -> dict[str, list[str]]:
    base = [f"bucket_{bucket}" for bucket in buckets] + [
        "logical_tokens",
        "padding_tokens",
        "known_rows",
        "unknown_rows",
        "known_row_fraction",
        "known_cpu_routes",
        "prior_cpu_routes",
        "proxy_cpu_routes",
        "proxy_cpu_experts",
        "proxy_cpu_routes_per_row",
        "proxy_cpu_experts_per_row",
        "cached_expert_count",
        "ready_direct_active_experts",
    ]
    layer = base + [
        name
        for layer_idx in range(num_layers)
        for name in (
            f"layer_{layer_idx}_proxy_cpu_routes",
            f"layer_{layer_idx}_proxy_cpu_experts",
        )
    ]
    polynomial = base + [
        "proxy_cpu_routes_sq",
        "proxy_cpu_experts_sq",
        "proxy_routes_x_experts",
        "known_routes_sq",
    ] + [
        f"b{bucket}_x_{base_name}"
        for bucket in buckets
        for base_name in (
            "proxy_cpu_routes",
            "proxy_cpu_experts",
            "known_cpu_routes",
            "prior_cpu_routes",
            "ready_direct_active_experts",
        )
    ]
    return {
        "proxy_global": base,
        "proxy_global_poly": polynomial,
        "proxy_layer": layer,
    }


def _matrix(samples: list[Sample], names: list[str]) -> np.ndarray:
    return np.asarray(
        [
            [
                float(sample.bucket == int(name.split("_", 1)[1]))
                if name.startswith("bucket_")
                and name.removeprefix("bucket_").isdigit()
                else float(sample.values[name])
                for name in names
            ]
            for sample in samples
        ],
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
    design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    penalty = np.eye(design.shape[1]) * float(ridge)
    penalty[0, 0] = 0.0
    system = design.T @ design + penalty
    target = design.T @ y
    try:
        coefficients = np.linalg.solve(system, target)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(system, target, rcond=None)[0]
    return mean, scale, coefficients[1:], float(coefficients[0])


def _predict(
    x: np.ndarray,
    fit: tuple[np.ndarray, np.ndarray, np.ndarray, float],
) -> np.ndarray:
    mean, scale, coefficients, intercept = fit
    return intercept + ((x - mean) / scale) @ coefficients


def _latency_prediction(
    base_model: VerifyTimeCostModel,
    samples: list[Sample],
    routes: np.ndarray,
    experts: np.ndarray,
) -> np.ndarray:
    result = []
    for sample, cpu_routes, cpu_experts in zip(
        samples,
        routes,
        experts,
        strict=True,
    ):
        workload = VerifyCpuWorkload(
            bucket=sample.bucket,
            logical_tokens=sample.logical_tokens,
            layer_route_counts=tuple(
                (0.0,) * base_model.num_experts
                for _ in range(base_model.num_layers)
            ),
        )
        max_routes = float(
            sample.bucket * base_model.top_k * base_model.num_layers
        )
        cpu_routes = min(max_routes, max(0.0, float(cpu_routes)))
        cpu_experts = min(
            float(base_model.num_layers * base_model.num_experts),
            cpu_routes,
            max(0.0, float(cpu_experts)),
        )
        result.append(
            base_model._prediction_with_global_cpu_counts(
                workload,
                cpu_routes=cpu_routes,
                cpu_experts=cpu_experts,
            )
        )
    return np.asarray(result, dtype=np.float64)


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = np.abs(target - prediction)
    denominator = float(np.sum((target - target.mean()) ** 2))
    correct = compared = 0
    for left in range(len(target)):
        for right in range(left + 1, len(target)):
            delta = float(target[left] - target[right])
            if abs(delta) < 1.0:
                continue
            correct += int(delta * float(prediction[left] - prediction[right]) > 0.0)
            compared += 1
    return {
        "mae_ms": float(error.mean()),
        "p90_abs_error_ms": float(np.percentile(error, 90)),
        "r2": (
            1.0 - float(np.sum((target - prediction) ** 2)) / denominator
            if denominator > 0.0
            else 0.0
        ),
        "ranking_accuracy": float(correct / compared) if compared else 0.0,
        "ranking_pairs": float(compared),
    }


def _workload_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = np.abs(target - prediction)
    denominator = float(np.sum((target - target.mean()) ** 2))
    return {
        "mae": float(error.mean()),
        "p90_abs_error": float(np.percentile(error, 90)),
        "r2": (
            1.0 - float(np.sum((target - prediction) ** 2)) / denominator
            if denominator > 0.0
            else 0.0
        ),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    base_path = Path(args.base_artifact).resolve()
    base_artifact = json.loads(base_path.read_text(encoding="utf-8"))
    base_model = VerifyTimeCostModel(base_artifact)
    unsupported = [
        name
        for name in base_model.feature_names
        if name.startswith("layer_")
        or name
        in {
            "route_sq_sum",
            "max_route_per_expert_sum",
            "singleton_experts",
            "multi_route_experts",
            "nonempty_layers",
            "max_layer_routes",
            "std_layer_routes",
        }
    ]
    if unsupported:
        raise SystemExit(
            "proxy workload calibration requires a global-count verify model; "
            f"unsupported features={unsupported}"
        )
    paths = _paths(args.profiles)
    samples = [
        sample
        for path in paths
        for sample in _load(
            path,
            base_model_id=base_model.model_id,
            drop_first_calls=int(args.drop_first_calls),
        )
    ]
    if len(samples) < 50:
        raise SystemExit(f"only {len(samples)} proxy calibration samples; need 50")
    groups = sorted({sample.source for sample in samples})
    rng = random.Random(int(args.seed))
    rng.shuffle(groups)
    holdout_count = max(1, round(len(groups) * float(args.holdout_fraction)))
    holdout_groups = set(groups[:holdout_count])
    remaining_groups = groups[holdout_count:]
    tuning_count = max(1, round(len(remaining_groups) * float(args.tuning_fraction)))
    tuning_groups = set(remaining_groups[:tuning_count])
    fit_samples = [
        sample
        for sample in samples
        if sample.source not in holdout_groups and sample.source not in tuning_groups
    ]
    tuning = [sample for sample in samples if sample.source in tuning_groups]
    outer_train = [sample for sample in samples if sample.source not in holdout_groups]
    holdout = [sample for sample in samples if sample.source in holdout_groups]
    if min(len(fit_samples), len(tuning), len(holdout)) < 10:
        raise SystemExit("proxy grouped split is too small")

    route_fit_y = np.asarray(
        [sample.actual_cpu_routes for sample in fit_samples], dtype=np.float64
    )
    expert_fit_y = np.asarray(
        [sample.actual_cpu_experts for sample in fit_samples], dtype=np.float64
    )
    tuning_y = np.asarray([sample.target_ms for sample in tuning], dtype=np.float64)
    candidates = _feature_names(base_model.buckets, base_model.num_layers)
    leaderboard = []
    for name, names in candidates.items():
        fit_x = _matrix(fit_samples, names)
        tuning_x = _matrix(tuning, names)
        for ridge in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0):
            route_fit = _fit(fit_x, route_fit_y, ridge)
            expert_fit = _fit(fit_x, expert_fit_y, ridge)
            predicted_routes = _predict(tuning_x, route_fit)
            predicted_experts = _predict(tuning_x, expert_fit)
            latency = _latency_prediction(
                base_model,
                tuning,
                predicted_routes,
                predicted_experts,
            )
            leaderboard.append(
                {
                    "name": name,
                    "feature_names": names,
                    "ridge": ridge,
                    "tuning_metrics": _metrics(tuning_y, latency),
                }
            )
    selected = min(
        leaderboard,
        key=lambda row: (
            float(row["tuning_metrics"]["mae_ms"]),
            len(row["feature_names"]),
        ),
    )
    names = list(selected["feature_names"])
    ridge = float(selected["ridge"])
    outer_x = _matrix(outer_train, names)
    route_outer_y = np.asarray(
        [sample.actual_cpu_routes for sample in outer_train], dtype=np.float64
    )
    expert_outer_y = np.asarray(
        [sample.actual_cpu_experts for sample in outer_train], dtype=np.float64
    )
    route_outer_fit = _fit(outer_x, route_outer_y, ridge)
    expert_outer_fit = _fit(outer_x, expert_outer_y, ridge)
    holdout_x = _matrix(holdout, names)
    predicted_routes = _predict(holdout_x, route_outer_fit)
    predicted_experts = _predict(holdout_x, expert_outer_fit)
    holdout_target = np.asarray(
        [sample.target_ms for sample in holdout], dtype=np.float64
    )
    holdout_latency = _latency_prediction(
        base_model,
        holdout,
        predicted_routes,
        predicted_experts,
    )
    validation_metrics = _metrics(holdout_target, holdout_latency)
    by_bucket = {}
    bucket_gate_passed = True
    for bucket in sorted({sample.bucket for sample in holdout}):
        indices = [
            index for index, sample in enumerate(holdout) if sample.bucket == bucket
        ]
        bucket_metrics = _metrics(
            holdout_target[indices],
            holdout_latency[indices],
        )
        bucket_passed = (
            len(indices) >= int(args.minimum_bucket_samples)
            and bucket_metrics["mae_ms"] <= float(args.max_bucket_mae_ms)
            and bucket_metrics["p90_abs_error_ms"]
            <= float(args.max_bucket_p90_ms)
        )
        bucket_gate_passed = bucket_gate_passed and bucket_passed
        by_bucket[str(bucket)] = {
            "sample_count": len(indices),
            "metrics": bucket_metrics,
            "passed": bucket_passed,
        }
    oracle_latency = _latency_prediction(
        base_model,
        holdout,
        np.asarray([sample.actual_cpu_routes for sample in holdout]),
        np.asarray([sample.actual_cpu_experts for sample in holdout]),
    )
    oracle_metrics = _metrics(holdout_target, oracle_latency)
    gate_passed = bucket_gate_passed and (
        validation_metrics["mae_ms"] <= float(args.max_mae_ms)
        and validation_metrics["p90_abs_error_ms"] <= float(args.max_p90_ms)
        and validation_metrics["ranking_accuracy"]
        >= float(args.min_ranking_accuracy)
    )

    all_x = _matrix(samples, names)
    route_all_y = np.asarray(
        [sample.actual_cpu_routes for sample in samples], dtype=np.float64
    )
    expert_all_y = np.asarray(
        [sample.actual_cpu_experts for sample in samples], dtype=np.float64
    )
    route_all_fit = _fit(all_x, route_all_y, ridge)
    expert_all_fit = _fit(all_x, expert_all_y, ridge)
    mean, scale, route_coefficients, route_intercept = route_all_fit
    _, _, expert_coefficients, expert_intercept = expert_all_fit
    proxy_model = {
        "model_kind": str(selected["name"]),
        "feature_names": names,
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "route_coefficients": route_coefficients.tolist(),
        "route_intercept": route_intercept,
        "expert_coefficients": expert_coefficients.tolist(),
        "expert_intercept": expert_intercept,
        "ridge": ridge,
        "validation_metrics": validation_metrics,
        "oracle_actual_workload_metrics": oracle_metrics,
        "actual_route_metrics": _workload_metrics(
            np.asarray([sample.actual_cpu_routes for sample in holdout]),
            predicted_routes,
        ),
        "actual_expert_metrics": _workload_metrics(
            np.asarray([sample.actual_cpu_experts for sample in holdout]),
            predicted_experts,
        ),
        "split": {
            "seed": int(args.seed),
            "fit_samples": len(fit_samples),
            "tuning_samples": len(tuning),
            "outer_train_samples": len(outer_train),
            "holdout_samples": len(holdout),
            "deployment_refit_samples": len(samples),
            "holdout_groups": sorted(holdout_groups),
            "tuning_groups": sorted(tuning_groups),
        },
        "training_manifest": [
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in paths
        ],
        "tuning_leaderboard": sorted(
            [
                {
                    "name": row["name"],
                    "ridge": row["ridge"],
                    "feature_count": len(row["feature_names"]),
                    "tuning_metrics": row["tuning_metrics"],
                }
                for row in leaderboard
            ],
            key=lambda row: float(row["tuning_metrics"]["mae_ms"]),
        ),
    }
    output_artifact = dict(base_artifact)
    output_artifact.pop("deployment_validation", None)
    output_artifact["proxy_workload_model"] = proxy_model
    output_artifact["proxy_workload_gate_passed"] = gate_passed
    gate_version = str(getattr(args, "gate_version", "v1"))
    output_artifact["proxy_workload_gate_version"] = gate_version
    output_artifact["proxy_workload_validation"] = {
        "gate_version": gate_version,
        "passed": gate_passed,
        "metrics": validation_metrics,
        "by_bucket": by_bucket,
        "gate": {
            "max_mae_ms": float(args.max_mae_ms),
            "max_p90_abs_error_ms": float(args.max_p90_ms),
            "min_ranking_accuracy": float(args.min_ranking_accuracy),
            "minimum_bucket_samples": int(args.minimum_bucket_samples),
            "max_bucket_mae_ms": float(args.max_bucket_mae_ms),
            "max_bucket_p90_abs_error_ms": float(args.max_bucket_p90_ms),
        },
    }
    output_artifact["model_id"] = compute_model_id(output_artifact)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(output_artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = output.with_suffix(".md")
    report.write_text(
        "\n".join(
            [
                "# Verify Workload Proxy",
                "",
                f"- samples: `{len(samples)}`",
                f"- selected: `{selected['name']}` ridge `{ridge}`",
                f"- holdout MAE: `{validation_metrics['mae_ms']:.3f} ms`",
                f"- holdout P90: `{validation_metrics['p90_abs_error_ms']:.3f} ms`",
                f"- holdout ranking: `{validation_metrics['ranking_accuracy']:.3f}`",
                f"- gate: `{'PASS' if gate_passed else 'FAIL'}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "artifact": str(output),
                "model_id": output_artifact["model_id"],
                "selected": selected["name"],
                "ridge": ridge,
                "metrics": validation_metrics,
                "passed": gate_passed,
            },
            indent=2,
        )
    )
    if not gate_passed:
        raise SystemExit(1)
    return output_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-artifact", required=True)
    parser.add_argument("--profiles", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--drop-first-calls", type=int, default=2)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--tuning-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--max-mae-ms", type=float, default=5.0)
    parser.add_argument("--max-p90-ms", type=float, default=10.0)
    parser.add_argument("--min-ranking-accuracy", type=float, default=0.9)
    parser.add_argument("--minimum-bucket-samples", type=int, default=5)
    parser.add_argument("--max-bucket-mae-ms", type=float, default=7.5)
    parser.add_argument("--max-bucket-p90-ms", type=float, default=12.5)
    parser.add_argument("--gate-version", default="v1")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
