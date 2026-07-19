import json

import numpy as np
import pytest

from nanovllm.engine.speculative.verify_cost_model import (
    DraftRouteCostProxy,
    SCHEMA_VERSION,
    VerifyCpuWorkload,
    VerifyTimeCostModel,
    compute_model_id,
    feature_values,
)


def _artifact():
    return {
        "schema_version": SCHEMA_VERSION,
        "num_layers": 2,
        "num_experts": 4,
        "top_k": 2,
        "buckets": [3, 5],
        "feature_names": ["bucket_3", "bucket_5", "cpu_routes", "cpu_experts"],
        "feature_mean": [0.0, 0.0, 0.0, 0.0],
        "feature_scale": [1.0, 1.0, 1.0, 1.0],
        "coefficients": [10.0, 20.0, 2.0, 3.0],
        "intercept": 5.0,
        "minimum_ms": 1.0,
        "validation_metrics": {"p90_abs_error_ms": 4.5},
        "fingerprint": {"kt_num_threads": 16},
    }


def test_workload_mapping_and_shape_features():
    workload = VerifyCpuWorkload.from_mapping(
        bucket=5,
        logical_tokens=4,
        layer_route_counts={"0": [2, 0, 1, 0], "1": [0, 3, 0, 0]},
        num_layers=2,
        num_experts=4,
    )

    assert workload.padding_tokens == 1
    assert feature_values(
        workload,
        ["cpu_routes", "cpu_experts", "route_sq_sum", "nonempty_layers"],
    ) == [6.0, 3.0, 14.0, 2.0]


def test_model_prediction_and_breakdown(tmp_path):
    path = tmp_path / "model.json"
    path.write_text(json.dumps(_artifact()), encoding="utf-8")
    model = VerifyTimeCostModel.load(path)
    workload = VerifyCpuWorkload.from_mapping(
        bucket=3,
        logical_tokens=2,
        layer_route_counts={0: [1, 0, 0, 0], 1: [0, 2, 0, 0]},
        num_layers=2,
        num_experts=4,
    )

    prediction = model.predict(workload)
    assert prediction.total_ms == pytest.approx(27.0)
    assert prediction.fixed_ms == pytest.approx(15.0)
    assert prediction.exposed_cpu_ms == pytest.approx(12.0)
    assert prediction.error_p90_ms == pytest.approx(4.5)
    count_prediction = model.predict_cpu_counts(
        bucket=3,
        logical_tokens=2,
        cpu_routes=3.0,
        cpu_experts=2.0,
    )
    assert count_prediction.total_ms == pytest.approx(27.0)
    with pytest.raises(ValueError, match="outside physical route bounds"):
        model.predict_cpu_counts(
            bucket=3,
            logical_tokens=2,
            cpu_routes=1.0,
            cpu_experts=2.0,
        )
    model.validate_fingerprint({"kt_num_threads": 16})
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        model.validate_fingerprint({"kt_num_threads": 8})


def test_model_rejects_unknown_bucket():
    model = VerifyTimeCostModel(_artifact())
    workload = VerifyCpuWorkload.from_mapping(
        bucket=7,
        logical_tokens=6,
        layer_route_counts={},
        num_layers=2,
        num_experts=4,
    )
    with pytest.raises(ValueError, match="not calibrated"):
        model.predict(workload)


def test_aggregate_count_prediction_rejects_shape_model():
    artifact = _artifact()
    artifact["feature_names"].append("route_sq_sum")
    artifact["feature_mean"].append(0.0)
    artifact["feature_scale"].append(1.0)
    artifact["coefficients"].append(1.0)
    model = VerifyTimeCostModel(artifact)

    with pytest.raises(ValueError, match="incompatible with model features"):
        model.predict_cpu_counts(
            bucket=3,
            logical_tokens=2,
            cpu_routes=2.0,
            cpu_experts=2.0,
        )


def test_draft_route_proxy_uses_cache_for_known_and_prior_rows():
    artifact = _artifact()
    artifact["unknown_row_expert_route_priors"] = {
        "3": [
            [1.0, 0.5, 0.5, 0.0],
            [0.0, 1.0, 1.0, 0.0],
        ]
    }
    model = VerifyTimeCostModel(artifact)
    proxy = DraftRouteCostProxy(model)
    proxy.observe(
        [
            [[0, 1]],
            [[2, 3]],
        ]
    )

    estimate = proxy.estimate(
        bucket=3,
        logical_tokens=2,
        cached_experts={0: {0}, 1: {2}},
    )

    assert estimate.known_rows == 1
    assert estimate.unknown_rows == 2
    assert estimate.known_cpu_routes == 2.0
    assert estimate.prior_cpu_routes == 4.0
    assert estimate.workload.layer_route_counts == (
        (0.0, 2.0, 1.0, 0.0),
        (0.0, 2.0, 0.0, 1.0),
    )


def test_draft_route_proxy_accepts_cache_owned_host_masks():
    artifact = _artifact()
    artifact["unknown_row_expert_route_priors"] = {
        "3": [[1.0, 0.5, 0.5, 0.0], [0.0, 1.0, 1.0, 0.0]]
    }
    proxy = DraftRouteCostProxy(VerifyTimeCostModel(artifact))
    proxy.observe(np.asarray([[[0, 1]], [[2, 3]]], dtype=np.float32))
    mask = proxy.build_uncached_mask_from_host_masks(
        [(0, np.asarray([True, False, False, False])),
         (1, np.asarray([False, False, True, False]))]
    )
    estimate = proxy.estimate_summary(
        bucket=3,
        logical_tokens=2,
        uncached_mask=mask,
    )

    assert estimate.known_cpu_routes == 2.0
    assert estimate.prior_cpu_routes == 4.0


def test_proxy_workload_stage_predicts_execution_counts_before_latency():
    artifact = _artifact()
    artifact["proxy_workload_model"] = {
        "feature_names": ["proxy_cpu_routes"],
        "feature_mean": [0.0],
        "feature_scale": [1.0],
        "route_coefficients": [0.5],
        "route_intercept": 0.0,
        "expert_coefficients": [0.25],
        "expert_intercept": 0.0,
        "validation_metrics": {"p90_abs_error_ms": 2.0},
    }
    artifact["model_id"] = compute_model_id(artifact)
    model = VerifyTimeCostModel(artifact)
    estimate = type("Estimate", (), {})()
    estimate.workload = VerifyCpuWorkload.from_mapping(
        bucket=3,
        logical_tokens=2,
        layer_route_counts={0: [2, 0, 1, 0], 1: [0, 3, 0, 0]},
        num_layers=2,
        num_experts=4,
    )
    estimate.known_rows = 1
    estimate.unknown_rows = 2
    estimate.known_cpu_routes = 2.0
    estimate.prior_cpu_routes = 4.0

    prediction = model.predict_proxy(
        estimate,
        cached_expert_count=2,
        ready_direct_active_experts=0,
    )

    assert prediction.estimated_cpu_routes == pytest.approx(3.0)
    assert prediction.estimated_cpu_experts == pytest.approx(1.5)
    assert prediction.total_ms == pytest.approx(25.5)
    assert prediction.error_p90_ms == pytest.approx(2.0)


def test_model_identity_rejects_tampered_coefficients():
    artifact = _artifact()
    artifact["model_id"] = compute_model_id(artifact)
    artifact["coefficients"][0] += 1.0

    with pytest.raises(ValueError, match="identity mismatch"):
        VerifyTimeCostModel(artifact)


def test_protocol_adjustment_is_model_bound_and_strategy_checked():
    artifact = _artifact()
    artifact["protocol_adjustment"] = {
        "kind": "bucket_offset_ms",
        "acceptance_strategy": "standard_sampling",
        "temperature": 0.8,
        "bucket_offsets_ms": {"3": 4.0, "5": -2.0},
    }
    base_id = compute_model_id(_artifact())
    artifact["model_id"] = compute_model_id(artifact)
    model = VerifyTimeCostModel(artifact)

    assert model.model_id != base_id
    prediction = model.predict_cpu_counts(
        bucket=3,
        logical_tokens=2,
        cpu_routes=3.0,
        cpu_experts=2.0,
    )
    assert prediction.total_ms == pytest.approx(31.0)
    assert prediction.fixed_ms == pytest.approx(19.0)
    assert prediction.exposed_cpu_ms == pytest.approx(12.0)
    model.validate_protocol(
        acceptance_strategy="sampling",
        temperature=0.8,
    )
    with pytest.raises(ValueError, match="requires acceptance strategy"):
        model.validate_protocol(acceptance_strategy="greedy")
    with pytest.raises(ValueError, match="requires temperature"):
        model.validate_protocol(
            acceptance_strategy="standard_sampling",
            temperature=1.0,
        )


def test_protocol_bucket_affine_preserves_positive_cpu_cost_signal():
    artifact = _artifact()
    artifact["protocol_adjustment"] = {
        "kind": "bucket_affine",
        "acceptance_strategy": "standard_sampling",
        "temperature": 0.8,
        "bucket_affine": {
            "3": {"slope": 1.25, "intercept_ms": 2.0},
            "5": {"slope": 0.75, "intercept_ms": 4.0},
        },
    }
    artifact["model_id"] = compute_model_id(artifact)
    model = VerifyTimeCostModel(artifact)

    prediction = model.predict_cpu_counts(
        bucket=3,
        logical_tokens=2,
        cpu_routes=3.0,
        cpu_experts=2.0,
    )
    assert prediction.total_ms == pytest.approx(35.75)
    assert prediction.fixed_ms == pytest.approx(20.75)
    assert prediction.exposed_cpu_ms == pytest.approx(15.0)
