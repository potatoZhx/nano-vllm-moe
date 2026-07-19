import json
from argparse import Namespace

import pytest

from nanovllm.engine.speculative.verify_cost_model import (
    VerifyTimeCostModel,
    compute_model_id,
)
from scripts.validate_verify_cost_shadow import run


def test_shadow_validation_updates_artifact_after_gate_passes(tmp_path):
    profiles = []
    for profile_index in range(2):
        path = tmp_path / f"sample{profile_index:04d}.json"
        records = []
        traces = []
        for call_index in range(6):
            target = 20.0 + 2.0 * call_index + profile_index
            records.append(
                {
                    "call_index": call_index,
                    "bucket": 3 if call_index < 3 else 5,
                    "token_count": 2 if call_index < 3 else 4,
                    "outputs_ready": True,
                    "stream_ms_available": True,
                    "verify_cost_model_mode": "shadow",
                }
            )
            traces.append(
                {
                    "verify_call_index": call_index,
                    "verify_accept_ready_ms": target,
                    "verify_cost_prediction_ms": target + 0.25,
                    "verify_cost_predictions": [
                        {
                            "verify_cost_prediction_ms": target + 0.25,
                            "verify_cost_bucket": 3 if call_index < 3 else 5,
                            "verify_cost_logical_tokens": 2 if call_index < 3 else 4,
                            "verify_cost_model_id": "synthetic-model",
                        }
                    ],
                }
            )
        path.write_text(
            json.dumps(
                {
                    "verify_cost_measurement": {
                        "enabled": False,
                        "target": "spec.verify_accept_ready_ms",
                        "profile_cuda_sync": False,
                    },
                    "model_verify_call_records": records,
                    "spec_step_traces": traces,
                }
            ),
            encoding="utf-8",
        )
        profiles.append(str(path))

    artifact_path = tmp_path / "model.json"
    artifact_path.write_text(
        json.dumps({"schema_version": 1, "model_id": "synthetic-model"}),
        encoding="utf-8",
    )
    output_artifact = tmp_path / "validated.json"
    result = run(
        Namespace(
            profiles=profiles,
            artifact=str(artifact_path),
            output_artifact=str(output_artifact),
            output="",
            replay_base_artifact="",
            drop_first_calls=0,
            minimum_samples=10,
            max_mae_ms=1.0,
            max_p90_ms=1.0,
            min_ranking_accuracy=0.9,
            minimum_bucket_samples=5,
            max_bucket_mae_ms=1.0,
            max_bucket_p90_ms=1.0,
        )
    )

    assert result["deployment_validation"]["passed"] is True
    assert result["deployment_validation"]["model_id"] == "synthetic-model"
    updated = json.loads(output_artifact.read_text(encoding="utf-8"))
    assert updated["deployment_validation"]["sample_count"] == 12


def test_shadow_validation_can_write_sampling_gate_without_replacing_greedy(
    tmp_path,
):
    profile = tmp_path / "sample0000.json"
    records = []
    traces = []
    for call_index in range(4):
        target = 20.0 + call_index
        records.append(
            {
                "call_index": call_index,
                "bucket": 3,
                "token_count": 2,
                "outputs_ready": True,
                "stream_ms_available": True,
                "verify_cost_model_mode": "shadow",
            }
        )
        traces.append(
            {
                "verify_call_index": call_index,
                "verify_accept_ready_ms": target,
                "verify_cost_prediction_ms": target,
                "verify_cost_predictions": [
                    {
                        "verify_cost_prediction_ms": target,
                        "verify_cost_bucket": 3,
                        "verify_cost_logical_tokens": 2,
                        "verify_cost_model_id": "synthetic-model",
                    }
                ],
            }
        )
    profile.write_text(
        json.dumps(
            {
                "verify_cost_measurement": {
                    "enabled": False,
                    "target": "spec.verify_accept_ready_ms",
                    "profile_cuda_sync": False,
                },
                "model_verify_call_records": records,
                "spec_step_traces": traces,
            }
        ),
        encoding="utf-8",
    )
    artifact = tmp_path / "model.json"
    artifact.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": "synthetic-model",
                "deployment_validation": {"passed": True, "gate_version": "v2"},
            }
        ),
        encoding="utf-8",
    )
    output_artifact = tmp_path / "validated.json"

    run(
        Namespace(
            profiles=[str(profile)],
            artifact=str(artifact),
            output_artifact=str(output_artifact),
            output="",
            replay_base_artifact="",
            drop_first_calls=0,
            minimum_samples=4,
            max_mae_ms=1.0,
            max_p90_ms=1.0,
            min_ranking_accuracy=0.9,
            minimum_bucket_samples=4,
            max_bucket_mae_ms=1.0,
            max_bucket_p90_ms=1.0,
            gate_version="v2",
            protocol="standard_sampling_temperature_0.8",
            deployment_field="sampling_deployment_validation",
        )
    )

    updated = json.loads(output_artifact.read_text(encoding="utf-8"))
    assert updated["deployment_validation"]["passed"] is True
    sampling = updated["sampling_deployment_validation"]
    assert sampling["passed"] is True
    assert sampling["protocol"] == "standard_sampling_temperature_0.8"


def test_shadow_validation_accepts_sampling_logits_only_at_accept_ready_boundary(
    tmp_path,
):
    profile = tmp_path / "sample0000.json"
    profile.write_text(
        json.dumps(
            {
                "verify_cost_measurement": {
                    "enabled": False,
                    "target": "spec.verify_accept_ready_ms",
                    "profile_cuda_sync": False,
                },
                "model_verify_call_records": [
                    {
                        "call_index": 0,
                        "bucket": 3,
                        "token_count": 2,
                        "outputs_ready": False,
                        "return_logits": True,
                        "stream_ms_available": True,
                        "verify_cost_model_mode": "shadow",
                    }
                ],
                "spec_step_traces": [
                    {
                        "verify_call_index": 0,
                        "verify_accept_ready_ms": 20.0,
                        "verify_accept_ms": 1.5,
                        "verify_cost_prediction_ms": 20.0,
                        "verify_cost_predictions": [
                            {
                                "verify_cost_prediction_ms": 20.0,
                                "verify_cost_bucket": 3,
                                "verify_cost_logical_tokens": 2,
                                "verify_cost_model_id": "synthetic-model",
                            }
                        ],
                        "sequences": [
                            {
                                "acceptance_mode": "standard_sampling",
                                "next_token": 42,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    artifact = tmp_path / "model.json"
    artifact.write_text(
        json.dumps({"schema_version": 1, "model_id": "synthetic-model"}),
        encoding="utf-8",
    )
    common = dict(
        profiles=[str(profile)],
        artifact=str(artifact),
        output_artifact="",
        output="",
        replay_base_artifact="",
        drop_first_calls=0,
        minimum_samples=1,
        max_mae_ms=1.0,
        max_p90_ms=1.0,
        min_ranking_accuracy=0.0,
        minimum_bucket_samples=1,
        max_bucket_mae_ms=1.0,
        max_bucket_p90_ms=1.0,
    )

    with pytest.raises(ValueError, match="outputs were not host-ready"):
        run(Namespace(**common, allow_return_logits=False))

    result = run(Namespace(**common, allow_return_logits=True))
    assert result["deployment_validation"]["passed"] is True
    assert result["deployment_validation"]["latency_boundary"][
        "return_logits_accepted"
    ] is True


def test_shadow_validation_rejects_execution_workload_instrumentation(tmp_path):
    profile = tmp_path / "sample0000.json"
    profile.write_text(
        json.dumps(
            {
                "verify_cost_measurement": {
                    "enabled": False,
                    "target": "spec.verify_accept_ready_ms",
                    "profile_cuda_sync": False,
                },
                "model_verify_call_records": [
                    {
                        "call_index": 0,
                        "bucket": 3,
                        "token_count": 2,
                        "outputs_ready": True,
                        "stream_ms_available": True,
                        "verify_cost_model_mode": "shadow",
                        "metadata_execution_available": True,
                        "metadata_execution_cpu_routes_sum": 8.0,
                    }
                ],
                "spec_step_traces": [
                    {
                        "verify_call_index": 0,
                        "verify_accept_ready_ms": 20.0,
                        "verify_cost_prediction_ms": 20.0,
                        "verify_cost_predictions": [
                            {
                                "verify_cost_prediction_ms": 20.0,
                                "verify_cost_bucket": 3,
                                "verify_cost_logical_tokens": 2,
                                "verify_cost_model_id": "synthetic-model",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    artifact = tmp_path / "model.json"
    artifact.write_text(
        json.dumps({"schema_version": 1, "model_id": "synthetic-model"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="contains execution workload"):
        run(
            Namespace(
                profiles=[str(profile)],
                artifact=str(artifact),
                output_artifact="",
                output="",
                replay_base_artifact="",
                drop_first_calls=0,
                minimum_samples=1,
                max_mae_ms=1.0,
                max_p90_ms=1.0,
                min_ranking_accuracy=0.9,
                minimum_bucket_samples=1,
                max_bucket_mae_ms=1.0,
                max_bucket_p90_ms=1.0,
            )
        )


def test_shadow_validation_replays_frozen_cpu_counts_through_alternate_base(
    tmp_path,
):
    def make_artifact(route_coefficient):
        artifact = {
            "schema_version": 1,
            "target": "verify_accept_ready_ms",
            "model_kind": "global_counts",
            "num_layers": 1,
            "num_experts": 2,
            "top_k": 1,
            "buckets": [3, 5],
            "feature_names": ["bucket_5", "logical_tokens", "cpu_routes", "cpu_experts"],
            "feature_mean": [0.0, 0.0, 0.0, 0.0],
            "feature_scale": [1.0, 1.0, 1.0, 1.0],
            "coefficients": [1.0, 0.5, route_coefficient, 0.25],
            "intercept": 10.0,
            "minimum_ms": 1.0,
            "validation_metrics": {"p90_abs_error_ms": 1.0},
            "fingerprint": {},
        }
        artifact["model_id"] = compute_model_id(artifact)
        return artifact

    source_artifact = make_artifact(2.0)
    replay_artifact = make_artifact(1.5)
    source_model = VerifyTimeCostModel(source_artifact)
    profiles = []
    for profile_index in range(2):
        records = []
        traces = []
        for call_index in range(6):
            bucket = 3 if call_index < 3 else 5
            logical_tokens = 2 if bucket == 3 else 4
            cpu_routes = float(1 + (call_index % 2))
            cpu_experts = 1.0
            prediction = source_model.predict_cpu_counts(
                bucket=bucket,
                logical_tokens=logical_tokens,
                cpu_routes=cpu_routes,
                cpu_experts=cpu_experts,
            ).total_ms
            records.append(
                {
                    "call_index": call_index,
                    "bucket": bucket,
                    "token_count": logical_tokens,
                    "outputs_ready": True,
                    "stream_ms_available": True,
                    "verify_cost_model_mode": "shadow",
                }
            )
            traces.append(
                {
                    "verify_call_index": call_index,
                    "verify_accept_ready_ms": prediction,
                    "verify_cost_prediction_ms": prediction,
                    "verify_cost_predictions": [
                        {
                            "verify_cost_prediction_ms": prediction,
                            "verify_cost_bucket": bucket,
                            "verify_cost_logical_tokens": logical_tokens,
                            "verify_cost_cpu_routes": cpu_routes,
                            "verify_cost_cpu_experts": cpu_experts,
                            "verify_cost_model_id": source_model.model_id,
                        }
                    ],
                }
            )
        profile = tmp_path / f"sample{profile_index:04d}.json"
        profile.write_text(
            json.dumps(
                {
                    "verify_cost_measurement": {
                        "enabled": False,
                        "target": "spec.verify_accept_ready_ms",
                        "profile_cuda_sync": False,
                    },
                    "model_verify_call_records": records,
                    "spec_step_traces": traces,
                }
            ),
            encoding="utf-8",
        )
        profiles.append(str(profile))

    source_path = tmp_path / "source.json"
    replay_path = tmp_path / "replay.json"
    source_path.write_text(json.dumps(source_artifact), encoding="utf-8")
    replay_path.write_text(json.dumps(replay_artifact), encoding="utf-8")
    result = run(
        Namespace(
            profiles=profiles,
            artifact=str(source_path),
            replay_base_artifact=str(replay_path),
            output_artifact="",
            output="",
            drop_first_calls=0,
            minimum_samples=10,
            max_mae_ms=1.0,
            max_p90_ms=1.0,
            min_ranking_accuracy=0.5,
            minimum_bucket_samples=5,
            max_bucket_mae_ms=1.0,
            max_bucket_p90_ms=1.0,
        )
    )

    replay = result["alternate_base_replay"]
    assert replay["source_replay_max_abs_delta_ms"] == pytest.approx(0.0)
    assert replay["replay_base_model_id"] == replay_artifact["model_id"]
    assert replay["deployment_validation_transferred"] is False
