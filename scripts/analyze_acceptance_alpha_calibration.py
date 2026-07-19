#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from nanovllm.engine.speculative.acceptance_calibration import (
    ARTIFACT_KIND,
    SCHEMA_VERSION,
    acceptance_predictor_identity,
    compute_calibration_id,
)


def _paths(values: list[str]) -> list[Path]:
    paths: set[Path] = set()
    for value in values:
        target = Path(value)
        if target.is_dir():
            paths.update(target.rglob("sample*.json"))
        else:
            paths.update(Path(item) for item in glob.glob(value, recursive=True))
    return sorted(path.resolve() for path in paths if path.is_file())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _profile_manifest(paths: list[Path]) -> dict[str, object]:
    files = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in paths
    ]
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "profile_count": len(files),
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        "files": files,
    }


def _points(
    paths: list[Path],
    *,
    label_source: str = "realized_prefix",
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    predicted: list[float] = []
    realized: list[float] = []
    trace_count = sequence_count = rejected_sequence_count = 0
    acceptance_modes: dict[str, int] = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        traces = data.get("spec_step_traces", [])
        if not isinstance(traces, list):
            continue
        for trace in traces:
            if not isinstance(trace, dict):
                continue
            trace_count += 1
            sequences = trace.get("sequences", [])
            if not isinstance(sequences, list):
                continue
            for sequence in sequences:
                if not isinstance(sequence, dict):
                    continue
                raw = sequence.get("predicted_alpha", [])
                if not isinstance(raw, list) or not raw:
                    continue
                mode = str(sequence.get("acceptance_mode", "unknown"))
                acceptance_modes[mode] = acceptance_modes.get(mode, 0) + 1
                drafted = min(
                    len(raw),
                    max(0, int(sequence.get("drafted_tokens", len(raw)) or 0)),
                )
                if drafted <= 0:
                    continue
                accepted = min(
                    drafted,
                    max(0, int(sequence.get("accepted_draft_tokens", 0) or 0)),
                )
                sequence_count += 1
                if accepted < drafted:
                    rejected_sequence_count += 1
                if label_source == "accept_probs":
                    probabilities = sequence.get("accept_probs", [])
                    if not isinstance(probabilities, list) or not probabilities:
                        continue
                    observed = min(drafted, len(probabilities))
                else:
                    probabilities = None
                    observed = accepted + 1 if accepted < drafted else drafted
                for position in range(observed):
                    alpha = float(raw[position])
                    if not math.isfinite(alpha):
                        continue
                    label = (
                        float(probabilities[position])
                        if probabilities is not None
                        else float(position < accepted)
                    )
                    if not math.isfinite(label):
                        continue
                    predicted.append(min(1.0, max(0.0, alpha)))
                    realized.append(min(1.0, max(0.0, label)))
    return (
        np.asarray(predicted, dtype=np.float64),
        np.asarray(realized, dtype=np.float64),
        {
            "trace_count": trace_count,
            "sequence_count": sequence_count,
            "rejected_sequence_count": rejected_sequence_count,
            "point_count": len(predicted),
            "acceptance_modes": dict(sorted(acceptance_modes.items())),
        },
    )


def _fit_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if x.size < 2 or float(np.var(x)) <= 1e-12:
        return 1.0, 0.0
    design = np.column_stack([x, np.ones_like(x)])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coefficients[0]), float(coefficients[1])


def _metrics(predicted: np.ndarray, realized: np.ndarray) -> dict[str, float | int]:
    clipped = np.clip(predicted, 0.0, 1.0)
    epsilon = 1e-7
    log_values = np.clip(clipped, epsilon, 1.0 - epsilon)
    return {
        "point_count": int(clipped.size),
        "predicted_mean": float(np.mean(clipped)),
        "realized_mean": float(np.mean(realized)),
        "mean_bias": float(np.mean(clipped - realized)),
        "brier": float(np.mean((clipped - realized) ** 2)),
        "log_loss": float(
            -np.mean(
                realized * np.log(log_values)
                + (1.0 - realized) * np.log(1.0 - log_values)
            )
        ),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    train_paths = _paths(args.train_profiles)
    validation_paths = _paths(args.validation_profiles)
    if not train_paths or not validation_paths:
        raise SystemExit("train and validation profiles must both be non-empty")
    overlap = sorted(set(train_paths) & set(validation_paths))
    if overlap:
        raise SystemExit(f"train/validation profiles overlap: {overlap[0]}")
    label_source = str(getattr(args, "label_source", "realized_prefix"))
    acceptance_strategy = str(getattr(args, "acceptance_strategy", "greedy"))
    if label_source == "accept_probs" and acceptance_strategy != "standard_sampling":
        raise SystemExit("accept_probs labels require standard_sampling")
    train_x, train_y, train_counts = _points(
        train_paths, label_source=label_source
    )
    validation_x, validation_y, validation_counts = _points(
        validation_paths, label_source=label_source
    )
    if train_x.size < int(args.minimum_points):
        raise SystemExit(f"only {train_x.size} training points")
    if validation_x.size < int(args.minimum_validation_points):
        raise SystemExit(f"only {validation_x.size} validation points")

    slope, intercept = _fit_affine(train_x, train_y)
    train_identity = _metrics(train_x, train_y)
    train_affine = _metrics(slope * train_x + intercept, train_y)
    validation_identity = _metrics(validation_x, validation_y)
    validation_affine = _metrics(slope * validation_x + intercept, validation_y)
    identity_brier = float(validation_identity["brier"])
    affine_brier = float(validation_affine["brier"])
    brier_improvement = (
        (identity_brier - affine_brier) / identity_brier
        if identity_brier > 0.0
        else 0.0
    )
    passed = (
        brier_improvement >= float(args.minimum_brier_improvement)
        and abs(float(validation_affine["mean_bias"]))
        <= float(args.maximum_abs_mean_bias)
    )
    selected = "affine" if passed else "identity"
    artifact: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "acceptance_predictor": acceptance_predictor_identity(args.predictor_path),
        "calibration_context": {
            "acceptance_strategy": acceptance_strategy,
            "temperature": float(getattr(args, "temperature", 0.0)),
        },
        "label_contract": {
            "kind": (
                "draft_token_true_accept_probability"
                if label_source == "accept_probs"
                else "conditional_until_first_rejection"
            ),
            "source": label_source,
            "accepted_prefix_positions": 1,
            "first_rejected_position": 0,
            "positions_after_first_rejection": (
                "included_from_accept_probs"
                if label_source == "accept_probs"
                else "excluded"
            ),
        },
        "transform": {
            "selected": selected,
            "slope": slope,
            "intercept": intercept,
            "clip_min": 0.0,
            "clip_max": 1.0,
        },
        "gate": {
            "passed": passed,
            "minimum_validation_brier_improvement": float(
                args.minimum_brier_improvement
            ),
            "maximum_validation_abs_mean_bias": float(args.maximum_abs_mean_bias),
            "validation_brier_improvement": brier_improvement,
        },
        "training": {
            "profiles": _profile_manifest(train_paths),
            "counts": train_counts,
            "identity_metrics": train_identity,
            "affine_metrics": train_affine,
        },
        "validation": {
            "profiles": _profile_manifest(validation_paths),
            "counts": validation_counts,
            "identity_metrics": validation_identity,
            "affine_metrics": validation_affine,
        },
    }
    artifact["calibration_id"] = compute_calibration_id(artifact)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact, indent=2))
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-profiles", nargs="+", required=True)
    parser.add_argument("--validation-profiles", nargs="+", required=True)
    parser.add_argument("--predictor-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--label-source",
        choices=["realized_prefix", "accept_probs"],
        default="realized_prefix",
    )
    parser.add_argument(
        "--acceptance-strategy",
        choices=["greedy", "standard_sampling"],
        default="greedy",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--minimum-points", type=int, default=1000)
    parser.add_argument("--minimum-validation-points", type=int, default=1000)
    parser.add_argument("--minimum-brier-improvement", type=float, default=0.10)
    parser.add_argument("--maximum-abs-mean-bias", type=float, default=0.02)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
