#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanovllm.engine.speculative.verify_cost_model import (  # noqa: E402
    VerifyTimeCostModel,
    compute_model_id,
)
from scripts.analyze_verify_workload_proxy import (  # noqa: E402
    _latency_prediction,
    _load,
    _matrix,
    _metrics,
    _paths,
    _predict,
)


def _base_artifact(proxy_artifact: dict[str, object]) -> dict[str, object]:
    base = dict(proxy_artifact)
    for key in (
        "proxy_workload_model",
        "proxy_workload_gate_passed",
        "proxy_workload_gate_version",
        "proxy_workload_validation",
        "deployment_validation",
        "model_id",
    ):
        base.pop(key, None)
    base["model_id"] = compute_model_id(base)
    return base


def run(args: argparse.Namespace) -> dict[str, object]:
    source_path = Path(args.artifact).resolve()
    artifact = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise ValueError("verify proxy artifact must be an object")
    proxy_model = artifact.get("proxy_workload_model")
    if not isinstance(proxy_model, dict):
        raise ValueError("artifact lacks a frozen proxy workload model")
    frozen_model = VerifyTimeCostModel(artifact)
    base_model = VerifyTimeCostModel(_base_artifact(artifact))
    split = proxy_model.get("split", {})
    if not isinstance(split, dict):
        raise ValueError("proxy workload model lacks a frozen split")
    holdout_groups = {
        str(Path(value).resolve()) for value in split.get("holdout_groups", [])
    }
    if not holdout_groups:
        raise ValueError("proxy workload model has no frozen holdout groups")

    paths = _paths(args.profiles)
    paths_by_name = {str(path): path for path in paths}
    training_manifest = proxy_model.get("training_manifest", [])
    if not isinstance(training_manifest, list) or not training_manifest:
        raise ValueError("proxy workload model lacks a training manifest")
    for row in training_manifest:
        expected_path = str(Path(row["path"]).resolve())
        path = paths_by_name.get(expected_path)
        if path is None:
            raise ValueError(f"proxy training profile is missing: {expected_path}")
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha != str(row["sha256"]):
            raise ValueError(f"proxy training profile hash changed: {expected_path}")
    samples = [
        sample
        for path in paths
        for sample in _load(
            path,
            base_model_id=base_model.model_id,
            drop_first_calls=int(args.drop_first_calls),
        )
        if str(Path(sample.source).resolve()) in holdout_groups
    ]
    observed_groups = {str(Path(sample.source).resolve()) for sample in samples}
    missing_groups = sorted(holdout_groups - observed_groups)
    if missing_groups:
        raise ValueError(f"frozen proxy holdout profile is missing: {missing_groups[0]}")
    if len(samples) < int(args.minimum_samples):
        raise SystemExit(f"only {len(samples)} frozen proxy holdout samples")

    names = [str(value) for value in proxy_model["feature_names"]]
    values = _matrix(samples, names)
    fit_common = (
        np.asarray(proxy_model["feature_mean"], dtype=np.float64),
        np.asarray(proxy_model["feature_scale"], dtype=np.float64),
    )
    predicted_routes = _predict(
        values,
        (
            *fit_common,
            np.asarray(proxy_model["route_coefficients"], dtype=np.float64),
            float(proxy_model["route_intercept"]),
        ),
    )
    predicted_experts = _predict(
        values,
        (
            *fit_common,
            np.asarray(proxy_model["expert_coefficients"], dtype=np.float64),
            float(proxy_model["expert_intercept"]),
        ),
    )
    target = np.asarray([sample.target_ms for sample in samples], dtype=np.float64)
    prediction = _latency_prediction(
        base_model,
        samples,
        predicted_routes,
        predicted_experts,
    )
    replay_metrics = _metrics(target, prediction)
    recorded_metrics = proxy_model.get("validation_metrics", {})
    if not isinstance(recorded_metrics, dict):
        raise ValueError("proxy workload model lacks grouped holdout metrics")
    metrics = {
        key: float(value) for key, value in recorded_metrics.items()
    }

    by_bucket = {}
    bucket_gate_passed = True
    for bucket in sorted({sample.bucket for sample in samples}):
        indices = [
            index for index, sample in enumerate(samples) if sample.bucket == bucket
        ]
        bucket_metrics = _metrics(target[indices], prediction[indices])
        bucket_passed = (
            len(indices) >= int(args.minimum_bucket_samples)
            and float(bucket_metrics["mae_ms"]) <= float(args.max_bucket_mae_ms)
            and float(bucket_metrics["p90_abs_error_ms"])
            <= float(args.max_bucket_p90_ms)
        )
        bucket_gate_passed = bucket_gate_passed and bucket_passed
        by_bucket[str(bucket)] = {
            "sample_count": len(indices),
            "metrics": bucket_metrics,
            "passed": bucket_passed,
        }
    passed = (
        float(metrics["mae_ms"]) <= float(args.max_mae_ms)
        and float(metrics["p90_abs_error_ms"]) <= float(args.max_p90_ms)
        and float(metrics["ranking_accuracy"]) >= float(args.min_ranking_accuracy)
    )
    validation = {
        "gate_version": str(args.gate_version),
        "passed": passed,
        "mode": "manifest_verified_grouped_holdout",
        "model_id": frozen_model.model_id,
        "base_model_id": base_model.model_id,
        "sample_count": len(samples),
        "metrics": metrics,
        "deployment_refit_holdout_replay": {
            "metrics": replay_metrics,
            "by_bucket": by_bucket,
            "bucket_thresholds_passed": bucket_gate_passed,
            "diagnostic_only": True,
            "note": (
                "The persisted deployment coefficients were refit on all proxy "
                "profiles. Their holdout replay is not an out-of-sample gate."
            ),
        },
        "gate": {
            "max_mae_ms": float(args.max_mae_ms),
            "max_p90_abs_error_ms": float(args.max_p90_ms),
            "min_ranking_accuracy": float(args.min_ranking_accuracy),
            "minimum_bucket_samples": int(args.minimum_bucket_samples),
            "max_bucket_mae_ms": float(args.max_bucket_mae_ms),
            "max_bucket_p90_abs_error_ms": float(args.max_bucket_p90_ms),
        },
        "source_artifact": str(source_path),
        "training_manifest_verified": True,
        "holdout_groups": sorted(holdout_groups),
    }
    artifact["proxy_workload_gate_passed"] = passed
    artifact["proxy_workload_gate_version"] = str(args.gate_version)
    artifact["proxy_workload_validation"] = validation
    if compute_model_id(artifact) != frozen_model.model_id:
        raise AssertionError("gate metadata changed the frozen verify model id")
    artifact["model_id"] = frozen_model.model_id

    output_artifact = Path(args.output_artifact)
    output_artifact.parent.mkdir(parents=True, exist_ok=True)
    output_artifact.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2))
    if not passed:
        raise SystemExit(1)
    return validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--profiles", nargs="+", required=True)
    parser.add_argument("--output-artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--drop-first-calls", type=int, default=2)
    parser.add_argument("--minimum-samples", type=int, default=20)
    parser.add_argument("--max-mae-ms", type=float, default=7.0)
    parser.add_argument("--max-p90-ms", type=float, default=14.0)
    parser.add_argument("--min-ranking-accuracy", type=float, default=0.82)
    parser.add_argument("--minimum-bucket-samples", type=int, default=5)
    parser.add_argument("--max-bucket-mae-ms", type=float, default=8.0)
    parser.add_argument("--max-bucket-p90-ms", type=float, default=17.0)
    parser.add_argument("--gate-version", default="v2")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
