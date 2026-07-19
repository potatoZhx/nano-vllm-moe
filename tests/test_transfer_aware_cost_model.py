import json
from types import SimpleNamespace

import numpy as np
import pytest

from nanovllm.engine.speculative.transfer_aware_cost_model import (
    CalibratedVerifyDemandPredictor,
    SIMULATOR_SEMANTICS_VERSION,
    ShadowCacheState,
    ShadowLayerState,
    ShadowTransfer,
    TransferAwareVerifyCostModel,
    bind_runtime_cache_host_matrices,
    compute_model_id,
    expected_distinct_experts,
    select_verify_bucket,
    select_shadow_victim,
    snapshot_runtime_state,
)
from nanovllm.expert.prefetcher import select_predictive_victim_slot


def _artifact(num_layers=2):
    artifact = {
        "schema_version": 3,
        "model_kind": "transfer_aware_verify",
        "simulator_semantics_version": SIMULATOR_SEMANTICS_VERSION,
        "num_layers": num_layers,
        "num_experts": 4,
        "top_k": 2,
        "max_draft_tokens": 12,
        "buckets": [5, 7, 8, 9, 10, 11, 12, 13],
        "protocol": {
            "batch_size": 1,
            "acceptance_strategy": "standard_sampling",
            "temperature": 0.8,
            "cache_ratio": 0.5,
            "max_draft_tokens": 12,
            "prefetch_runtime_kind": "predictive",
            "buckets": [5, 7, 8, 9, 10, 11, 12, 13],
        },
        "fingerprint": {
            "cpu_model": "test-cpu",
            "gpu_model": "test-gpu",
            "kt_kernel_version": "test-kt",
            "kt_num_threads": 16,
            "kt_backend": "avx2_bf16",
        },
        "demand_model": {
            "retention": 1.0,
            "layer_prior": [[0.5, 0.5, 0.5, 0.5]]
            * num_layers,
            "position_prior": [
                [[0.5, 0.5, 0.5, 0.5]] * num_layers
                for _ in range(13)
            ],
            "padding_prior": [[0.0] * 4] * num_layers,
            "next_recent_weight": 1.0,
        },
        "transfer_model": {
            "segment_size": 1,
            "expert_transfer_ms": 1.0,
            "max_inflight": 4,
            "draft_budget": 2,
            "verify_attention_ratio": 1.0,
            "segment_compute_ms": [4.0] * num_layers,
        },
        "latency_model": {
            "bucket_base_ms": {
                str(bucket): 10.0 for bucket in [5, 7, 8, 9, 10, 11, 12, 13]
            },
            "segment_coefficients": {
                str(segment): {
                    "cpu_experts": 1.0,
                    "cpu_routes": 2.0,
                    "max_layer_experts": 3.0,
                }
                for segment in range(num_layers)
            },
            "error_p90_ms": 2.0,
        },
    }
    artifact["model_id"] = compute_model_id(artifact)
    return artifact


def _state():
    return ShadowCacheState(
        layers=(
            ShadowLayerState(
                slots=(0, 1),
                pending=(-1, -1),
                access_values=(1.0, 2.0, 3.0, 4.0),
            ),
            ShadowLayerState(
                slots=(0, 1),
                pending=(-1, -1),
                access_values=(1.0, 2.0, 3.0, 4.0),
            ),
        )
    )


def test_dense_bucket_mapping_k6_through_k12():
    expected = {6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13}
    buckets = [5, 7, 8, 9, 10, 11, 12, 13]
    assert {
        k: select_verify_bucket(k + 1, buckets) for k in range(6, 13)
    } == expected


def test_route_row_alignment_and_next_row_prediction():
    predictor = CalibratedVerifyDemandPredictor(
        num_layers=2,
        num_experts=4,
        top_k=2,
        max_draft_tokens=12,
        artifact=_artifact()["demand_model"],
    )
    routes = np.array([[[1, 3]], [[0, 2]]], dtype=np.int64)
    predictor.observe(routes)

    rows = predictor.predict_rows(logical_tokens=2, bucket=5)

    # The first draft forward maps to verify logical row 0.
    assert rows[0, 0].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert rows[0, 1].tolist() == [1.0, 0.0, 1.0, 0.0]
    # next_recent_weight=1 makes verify-next repeat the calibrated latest row.
    np.testing.assert_allclose(rows[1], rows[0])
    # Tail padding has its independent zero prior.
    np.testing.assert_array_equal(rows[2:], 0.0)


def test_fractional_prior_mass_is_not_counted_as_every_expert():
    dense_expected_routes = np.full(128, 8.0 / 128.0)

    assert expected_distinct_experts(dense_expected_routes) == pytest.approx(
        8.0
    )
    assert expected_distinct_experts(
        np.asarray([3.0, 2.0, 0.0, 0.0])
    ) == pytest.approx(2.0)


def test_one_step_forecast_supplies_missing_aligned_and_next_rows():
    predictor = CalibratedVerifyDemandPredictor(
        num_layers=2,
        num_experts=4,
        top_k=2,
        max_draft_tokens=12,
        artifact=_artifact()["demand_model"],
    )
    predictor.observe(np.array([[[1, 3]], [[0, 2]]], dtype=np.int64))

    rows = predictor.predict_rows(logical_tokens=3, bucket=5)

    np.testing.assert_allclose(rows[1], rows[0])
    np.testing.assert_allclose(rows[2], rows[1])
    np.testing.assert_allclose(
        predictor.predict_aggregate(logical_tokens=3, bucket=5),
        rows.sum(axis=0),
        rtol=1e-6,
        atol=1e-6,
    )
    current, lookahead = predictor.predict_aggregate_pair(
        logical_tokens=2,
        bucket=5,
        next_bucket=5,
        current_out=np.empty((2, 4), dtype=np.float32),
        next_out=np.empty((2, 4), dtype=np.float32),
    )
    np.testing.assert_allclose(
        current,
        predictor.predict_aggregate(
            logical_tokens=2, bucket=5
        ),
    )
    np.testing.assert_allclose(
        lookahead,
        predictor.predict_aggregate(
            logical_tokens=3, bucket=5
        ),
    )


def test_shadow_simulator_respects_pending_inflight_and_vpb_without_mutation():
    model = TransferAwareVerifyCostModel(_artifact())
    model.observe(np.array([[[2, 3]], [[2, 3]]], dtype=np.int64))
    original = _state()
    state = ShadowCacheState(
        layers=original.layers,
        inflight=(
            ShadowTransfer(
                layer_idx=0,
                expert_idx=2,
                slot_idx=0,
                previous_expert=0,
                remaining_ms=0.5,
                source="draft_segment_indexed",
            ),
        ),
    )

    current_v0, _ = model.predict_pair(
        state=state, logical_tokens=2, vpb=0, next_draft_ms=2.0
    )
    current_v2, _ = model.predict_pair(
        state=state, logical_tokens=2, vpb=2, next_draft_ms=2.0
    )

    assert current_v2.cpu_routes <= current_v0.cpu_routes
    assert current_v2.transfer_submits <= 4
    assert state.layers == original.layers
    assert state.inflight[0].remaining_ms == 0.5


def test_runtime_fast_pair_matches_reference_shadow_transitions():
    model = TransferAwareVerifyCostModel(_artifact())
    model.observe(np.array([[[2, 3]], [[2, 3]]], dtype=np.int64))
    state = _state()
    logical_tokens = 2
    current_rows = model.demand.predict_rows(
        logical_tokens=logical_tokens, bucket=5
    )
    next_rows = model.demand.predict_rows(
        logical_tokens=logical_tokens + 1, bucket=5
    )
    current_reference = model.predict_simulation(
        model.simulator.simulate_verify(
            state, demand_rows=current_rows, vpb=2
        ),
        bucket=5,
        logical_tokens=logical_tokens,
    )
    next_state = model.simulator.simulate_next_draft(
        state, demand=next_rows.sum(axis=0), draft_ms=2.0
    )
    next_reference = model.predict_simulation(
        model.simulator.simulate_verify(
            next_state, demand_rows=next_rows, vpb=2
        ),
        bucket=5,
        logical_tokens=logical_tokens + 1,
    )

    current_fast, next_fast = model.predict_pair(
        state=state,
        logical_tokens=logical_tokens,
        vpb=2,
        next_draft_ms=2.0,
    )

    assert current_fast.cpu_routes == pytest.approx(
        current_reference.cpu_routes
    )
    assert current_fast.cpu_experts == pytest.approx(
        current_reference.cpu_experts
    )
    assert current_fast.transfer_submits == (
        current_reference.transfer_submits
    )
    assert next_fast.cpu_routes == pytest.approx(
        next_reference.cpu_routes
    )
    assert next_fast.cpu_experts == pytest.approx(
        next_reference.cpu_experts
    )
    assert next_fast.transfer_submits == next_reference.transfer_submits


def test_expert_only_batch_fast_path_matches_materialized_reference():
    artifact = _artifact(num_layers=4)
    fast_model = TransferAwareVerifyCostModel(artifact)
    reference_model = TransferAwareVerifyCostModel(artifact)
    routes = np.asarray(
        [[[2, 3]], [[3, 2]], [[2, 3]], [[3, 2]]],
        dtype=np.int64,
    )
    fast_model.observe(routes)
    reference_model.observe(routes)
    layer_caches = {}
    for layer_idx in range(4):
        resident = np.asarray([True, True, False, False])
        accesses = np.asarray(
            [layer_idx + 1, layer_idx + 5, 0, 0],
            dtype=np.int32,
        )
        layer_caches[layer_idx] = SimpleNamespace(
            num_slots=2,
            cached_expert_mask_host=resident,
            last_access_step_array=accesses,
            last_access_step=accesses.tolist(),
            access_count=[1, 1, 0, 0],
            slot_to_expert=[0, 1],
            active_slot_pending_expert=[-1, -1],
        )
    runtime = SimpleNamespace(inflight={}, _round_loaded={})
    cache_host = bind_runtime_cache_host_matrices(
        layer_caches=layer_caches,
        num_layers=4,
        num_experts=4,
    )
    assert np.shares_memory(
        layer_caches[0].cached_expert_mask_host,
        cache_host.resident,
    )
    assert np.shares_memory(
        layer_caches[0].last_access_step_array,
        cache_host.last_access,
    )
    fast_state = snapshot_runtime_state(
        layer_caches=layer_caches,
        prefetch_runtime=runtime,
        num_layers=4,
        num_experts=4,
        transfer_ms=1.0,
        materialize_layers=False,
        resident_host_matrix=cache_host.resident,
        access_host_matrix=cache_host.last_access,
        slot_counts_host=cache_host.slot_counts,
    )
    reference_state = snapshot_runtime_state(
        layer_caches=layer_caches,
        prefetch_runtime=runtime,
        num_layers=4,
        num_experts=4,
        transfer_ms=1.0,
        materialize_layers=True,
    )

    fast_pair = fast_model.predict_pair(
        state=fast_state,
        logical_tokens=3,
        vpb=2,
        next_draft_ms=2.0,
    )
    reference_pair = reference_model.predict_pair(
        state=reference_state,
        logical_tokens=3,
        vpb=2,
        next_draft_ms=2.0,
    )

    assert not fast_state.layers
    for fast, reference in zip(
        fast_pair, reference_pair, strict=True
    ):
        assert fast.cpu_routes == pytest.approx(
            reference.cpu_routes
        )
        assert fast.cpu_experts == pytest.approx(
            reference.cpu_experts
        )
        assert fast.transfer_submits == reference.transfer_submits
        assert fast.transfer_pending == reference.transfer_pending
        assert [
            (
                segment.cpu_routes,
                segment.cpu_experts,
                segment.transfer_submits,
            )
            for segment in fast.segments
        ] == pytest.approx(
            [
                (
                    segment.cpu_routes,
                    segment.cpu_experts,
                    segment.transfer_submits,
                )
                for segment in reference.segments
            ]
        )


def test_vpb_is_runtime_tunable_not_bound_to_artifact():
    model = TransferAwareVerifyCostModel(_artifact())
    model.observe(np.array([[[2, 3]], [[2, 3]]], dtype=np.int64))

    pred0, _ = model.predict_pair(
        state=_state(), logical_tokens=2, vpb=0, next_draft_ms=2.0
    )
    pred1, _ = model.predict_pair(
        state=_state(), logical_tokens=2, vpb=1, next_draft_ms=2.0
    )

    assert pred0.transfer_submits == 0
    assert 0 < pred1.transfer_submits <= 2


def test_verify_submit_helps_next_segment_not_current_segment():
    model = TransferAwareVerifyCostModel(_artifact())
    model.observe(np.array([[[2, 2]], [[2, 2]]], dtype=np.int64))

    without_prefetch, _ = model.predict_pair(
        state=_state(), logical_tokens=2, vpb=0, next_draft_ms=2.0
    )
    with_prefetch, _ = model.predict_pair(
        state=_state(), logical_tokens=2, vpb=1, next_draft_ms=2.0
    )

    # Segment zero selects a transfer for segment one only after segment zero's
    # replay starts, so its own CPU work cannot improve.
    assert (
        with_prefetch.segments[0].cpu_routes
        == without_prefetch.segments[0].cpu_routes
    )
    assert (
        with_prefetch.segments[1].cpu_routes
        < without_prefetch.segments[1].cpu_routes
    )


def test_next_draft_drain_publishes_existing_ticket_without_mutation():
    model = TransferAwareVerifyCostModel(_artifact())
    layers = list(_state().layers)
    layers[0] = ShadowLayerState(
        slots=layers[0].slots,
        pending=(3, -1),
        access_values=layers[0].access_values,
    )
    state = ShadowCacheState(
        layers=tuple(layers),
        inflight=(
            ShadowTransfer(
                layer_idx=0,
                expert_idx=3,
                slot_idx=0,
                previous_expert=0,
                remaining_ms=0.75,
                source="draft_segment_indexed",
            ),
        ),
    )
    demand = np.zeros((2, 4), dtype=np.float32)

    advanced = model.simulator.simulate_next_draft(
        state, demand=demand, draft_ms=2.0
    )

    assert advanced.layers[0].slots[0] == 3
    assert not advanced.inflight
    assert state.layers[0].slots[0] == 0
    assert state.inflight[0].remaining_ms == 0.75


def test_shadow_and_live_prefetch_use_identical_victim_rule():
    layer = ShadowLayerState(
        slots=(0, 1, 2),
        pending=(-1, 3, -1),
        access_values=(5.0, 0.0, 2.0, 9.0),
        protected_experts=frozenset({2}),
    )

    assert select_shadow_victim(layer) == select_predictive_victim_slot(
        slots=layer.slots,
        pending=layer.pending,
        access_values=layer.access_values,
        protected_experts=layer.protected_experts,
    )
    # Slot 1 is pending and expert 2 is protected, so expert 0 is the only
    # non-protected usable victim.
    assert select_shadow_victim(layer) == 0


def test_v3_model_id_protocol_and_legacy_rejection(tmp_path):
    artifact = _artifact()
    path = tmp_path / "model.v3.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    model = TransferAwareVerifyCostModel.load(path)
    model.validate_runtime(
        {
            "batch_size": 1,
            "acceptance_strategy": "standard_sampling",
            "temperature": 0.8,
            "cache_ratio": 0.5,
            "max_draft_tokens": 12,
            "prefetch_runtime_kind": "predictive",
            "buckets": [5, 7, 8, 9, 10, 11, 12, 13],
        }
    )
    with pytest.raises(ValueError, match="temperature"):
        model.validate_runtime(
            {
                "batch_size": 1,
                "acceptance_strategy": "standard_sampling",
                "temperature": 1.0,
                "cache_ratio": 0.5,
                "max_draft_tokens": 12,
                "prefetch_runtime_kind": "predictive",
                "buckets": [5, 7, 8, 9, 10, 11, 12, 13],
            }
        )
    legacy = dict(artifact)
    legacy["schema_version"] = 1
    with pytest.raises(ValueError, match="schema_version=3"):
        TransferAwareVerifyCostModel(legacy)


def test_model_id_covers_transfer_and_latency_components():
    left = _artifact()
    right = _artifact()
    right.pop("model_id")
    right["transfer_model"]["expert_transfer_ms"] = 2.0
    assert compute_model_id(left) != compute_model_id(right)
