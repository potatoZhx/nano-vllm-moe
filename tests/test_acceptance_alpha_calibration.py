import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nanovllm.engine.speculative.acceptance_calibration import (
    ARTIFACT_KIND,
    SCHEMA_VERSION,
    AcceptanceAlphaCalibration,
    acceptance_predictor_identity,
    compute_calibration_id,
)
from scripts.analyze_acceptance_alpha_calibration import run


def _predictor(tmp_path):
    path = tmp_path / "predictor"
    path.mkdir()
    (path / "config.json").write_text('{"meta": {}}', encoding="utf-8")
    (path / "best_model.pth").write_bytes(b"weights")
    return path


def _artifact(predictor, *, selected="affine"):
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "acceptance_predictor": acceptance_predictor_identity(predictor),
        "calibration_context": {
            "acceptance_strategy": "greedy",
            "temperature": 0.0,
        },
        "transform": {
            "selected": selected,
            "slope": 0.5,
            "intercept": 0.25,
        },
    }
    artifact["calibration_id"] = compute_calibration_id(artifact)
    return artifact


def test_runtime_calibration_is_identity_bound_and_clipped(tmp_path):
    predictor = _predictor(tmp_path)
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(_artifact(predictor)), encoding="utf-8")

    calibration = AcceptanceAlphaCalibration.load(
        path,
        acceptance_predictor_path=predictor,
    )

    assert calibration.calibrate(0.5) == pytest.approx(0.5)
    assert calibration.calibrate(10.0) == 1.0
    (predictor / "best_model.pth").write_bytes(b"different")
    with pytest.raises(ValueError, match="predictor mismatch"):
        AcceptanceAlphaCalibration.load(path, acceptance_predictor_path=predictor)


def _write_profile(path, count):
    traces = []
    for index in range(count):
        accepted = index % 2
        traces.append(
            {
                "sequences": [
                    {
                        "drafted_tokens": 1,
                        "accepted_draft_tokens": accepted,
                        "predicted_alpha": [0.9 if accepted else 0.8],
                    }
                ]
            }
        )
    path.write_text(json.dumps({"spec_step_traces": traces}), encoding="utf-8")


def test_analyzer_selects_affine_on_external_validation(tmp_path):
    predictor = _predictor(tmp_path)
    train = tmp_path / "train_sample.json"
    validation = tmp_path / "validation_sample.json"
    _write_profile(train, 100)
    _write_profile(validation, 100)
    output = tmp_path / "calibration.json"

    artifact = run(
        Namespace(
            train_profiles=[str(train)],
            validation_profiles=[str(validation)],
            predictor_path=str(predictor),
            output=str(output),
            minimum_points=10,
            minimum_validation_points=10,
            minimum_brier_improvement=0.1,
            maximum_abs_mean_bias=0.02,
        )
    )

    assert artifact["gate"]["passed"] is True
    assert artifact["transform"]["selected"] == "affine"
    assert artifact["validation"]["affine_metrics"]["brier"] == pytest.approx(0.0)
    calibration = AcceptanceAlphaCalibration.load(
        output,
        acceptance_predictor_path=predictor,
    )
    assert calibration.calibrate(0.85) == pytest.approx(0.5)


def test_calibration_rejects_acceptance_strategy_mismatch(tmp_path):
    predictor = _predictor(tmp_path)
    calibration = AcceptanceAlphaCalibration(_artifact(predictor))

    with pytest.raises(ValueError, match="strategy mismatch"):
        calibration.validate_acceptance_strategy("standard_sampling")


def test_analyzer_uses_sampling_accept_probabilities(tmp_path):
    predictor = _predictor(tmp_path)
    traces = [
        {
            "sequences": [
                {
                    "drafted_tokens": 2,
                    "accepted_draft_tokens": 0,
                    "predicted_alpha": [0.2, 0.8],
                    "accept_probs": [0.4, 0.6],
                    "acceptance_mode": "standard_sampling",
                }
            ]
        }
        for _ in range(20)
    ]
    train = tmp_path / "train_sampling.json"
    validation = tmp_path / "validation_sampling.json"
    payload = json.dumps({"spec_step_traces": traces})
    train.write_text(payload, encoding="utf-8")
    validation.write_text(payload, encoding="utf-8")

    artifact = run(
        Namespace(
            train_profiles=[str(train)],
            validation_profiles=[str(validation)],
            predictor_path=str(predictor),
            output=str(tmp_path / "sampling_calibration.json"),
            label_source="accept_probs",
            acceptance_strategy="standard_sampling",
            temperature=0.8,
            minimum_points=10,
            minimum_validation_points=10,
            minimum_brier_improvement=0.1,
            maximum_abs_mean_bias=0.02,
        )
    )

    assert artifact["gate"]["passed"] is True
    assert artifact["label_contract"]["source"] == "accept_probs"
    assert artifact["calibration_context"]["acceptance_strategy"] == "standard_sampling"
    assert artifact["training"]["counts"]["point_count"] == 40
