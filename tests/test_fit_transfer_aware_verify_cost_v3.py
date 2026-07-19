import json

import numpy as np
import pytest

from scripts.fit_transfer_aware_verify_cost_v3 import (
    _fit_demand,
    _fit_draft_vpb_timing,
    _fit_segment_compute_timing,
    _load_clean_latency_samples,
    _nnls_ridge,
    _partition_checkpoint_vpb,
    _split_groups,
    _trajectory_match_diagnostics,
)


def _sample(group_id, route):
    logical = np.asarray(
        [
            [[route, 1], [route, 2]],
            [[route, 1], [route, 2]],
        ],
        dtype=np.int64,
    )
    return {
        "group_id": group_id,
        "draft_routes": [logical[0]],
        "logical_routes": logical,
        "execution_routes": logical,
    }


def test_grouped_split_never_leaks_a_generation_trajectory():
    rows = [
        {"group_id": "a", "row": 0},
        {"group_id": "a", "row": 1},
        {"group_id": "b", "row": 0},
        {"group_id": "c", "row": 0},
    ]

    train, holdout = _split_groups(
        rows, holdout_fraction=0.34, seed=17
    )

    assert {row["group_id"] for row in train}.isdisjoint(
        {row["group_id"] for row in holdout}
    )
    assert len(train) + len(holdout) == len(rows)


def test_checkpoint_vpb_is_removed_by_complete_group():
    rows = [
        {"group_id": "train4", "vpb": 4},
        {"group_id": "train10", "vpb": 10},
        {"group_id": "check7", "vpb": 7},
        {"group_id": "check7", "vpb": 7},
    ]

    fit, checkpoint = _partition_checkpoint_vpb(rows, 7)

    assert {row["group_id"] for row in fit} == {"train4", "train10"}
    assert {row["group_id"] for row in checkpoint} == {"check7"}
    assert set(map(id, fit)).isdisjoint(set(map(id, checkpoint)))


def test_checkpoint_vpb_rejects_a_mixed_group():
    with pytest.raises(ValueError, match="must not mix"):
        _partition_checkpoint_vpb(
            [
                {"group_id": "bad", "vpb": 7},
                {"group_id": "bad", "vpb": 10},
            ],
            7,
        )


def test_demand_fit_uses_aligned_draft_row_and_verify_next_prior():
    artifact, diagnostics = _fit_demand(
        [_sample("a", 0), _sample("b", 3)],
        num_layers=2,
        num_experts=4,
        top_k=2,
        max_draft_tokens=12,
    )

    retention = np.asarray(artifact["retention"])
    assert retention[0].tolist() == [1.0, 1.0]
    assert diagnostics["aligned_layer_expert_overlap_mean"] == 1.0
    assert np.asarray(artifact["position_prior"]).shape == (13, 2, 4)


def test_latency_fit_is_nonnegative():
    x = np.asarray(
        [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
    )
    y = np.asarray([2.0, 3.0, 4.0, 5.0])

    beta = _nnls_ridge(x, y, ridge=1e-8)

    assert np.all(beta >= 0.0)
    np.testing.assert_allclose(x @ beta, y, atol=1e-4)


def test_segment_compute_timing_uses_existing_whole_stream_event():
    values, diagnostics = _fit_segment_compute_timing(
        [
            {
                "profile": {
                    "model_verify_call_records": [
                        {"stream_ms_available": True, "stream_ms": 40.0},
                        {"stream_ms_available": True, "stream_ms": 48.0},
                    ]
                }
            }
        ],
        segment_count=4,
        fallback_ms=12.0,
    )

    assert values == [11.0] * 4
    assert diagnostics["sample_count"] == 2
    assert diagnostics["source"].startswith("whole_verify_cuda_stream")


def test_offline_vpb_fit_charges_clean_draft_transfer_pressure():
    model, diagnostics = _fit_draft_vpb_timing(
        [
            {
                "vpb": 4,
                "draft_ms_sum": 180.0,
                "draft_call_count": 10,
            },
            {
                "vpb": 10,
                "draft_ms_sum": 186.0,
                "draft_call_count": 10,
            },
        ],
        checkpoint=[
            {
                "vpb": 7,
                "draft_ms_sum": 183.0,
                "draft_call_count": 10,
            }
        ],
    )

    assert model["slope_ms_per_vpb"] == pytest.approx(0.1)
    assert diagnostics["external_checkpoint_by_vpb"]["7"][
        "error_ms"
    ] == pytest.approx(0.0)


def test_clean_latency_uses_its_own_execution_workload(tmp_path):
    route_rows = [[0, 1]] * 5
    profile = {
        "verify_cost_measurement": {
            "enabled": False,
            "transfer_aware_enabled": False,
            "runtime_seed": 17,
            "sample": {"dataset": "test", "sample_id": "sample-0"},
            "case": {"max_draft_tokens": 6},
            "output_validation": {
                "output_sequence_count": 1,
                "fixed_length_ok": True,
                "error": "",
                "outputs_digest": "clean-digest",
            },
            "steady_draft_gate": {
                "steady_draft_call_count": 2,
                "steady_draft_call_mean_ms": 18.0,
            },
        },
        "model_verify_call_records": [
            {
                "call_index": 0,
                "token_count": 5,
                "bucket": 5,
                "used_cuda_graph": True,
                "dynamic_budget_value": 4,
                "metadata_layer_execution_route_rows": {
                    "0": route_rows,
                    "1": route_rows,
                },
                "metadata_layer_execution_cpu_route_counts": {
                    "0": [3, 2, 0, 0],
                    "1": [1, 4, 0, 0],
                },
            }
        ],
        "spec_step_traces": [
            {
                "verify_call_index": 0,
                "verify_accept_ready_ms": 12.5,
                "draft_call_ms": [18.0] * 6,
                "sequences": [{"calibrated_alpha": [0.8] * 6}],
            }
        ],
    }
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(profile), encoding="utf-8")

    rows, summaries = _load_clean_latency_samples(
        [path],
        num_layers=2,
        num_experts=4,
        top_k=2,
        segment_size=1,
    )

    assert len(rows) == 1
    assert rows[0]["latency_source"] == "instrumentation_off_own_trajectory"
    assert rows[0]["target_ms"] == 12.5
    np.testing.assert_array_equal(
        rows[0]["cpu_counts"],
        np.asarray([[3, 2, 0, 0], [1, 4, 0, 0]]),
    )
    assert summaries[0]["outputs_digest"] == "clean-digest"


def test_diverged_clean_digest_is_reported_without_cross_attachment():
    workload = [
        {
            "request_key": ("test", "sample", 17, 6, 4),
            "outputs_digest": "workload",
        }
    ]
    clean = [
        {
            "request_key": ("test", "sample", 17, 6, 4),
            "outputs_digest": "clean",
        }
    ]

    diagnostics = _trajectory_match_diagnostics(workload, clean)

    assert diagnostics["diverged_output_digest_count"] == 1
    assert diagnostics["matching_output_digest_count"] == 0
    assert diagnostics["latency_labels_attached_across_trajectories"] is False
