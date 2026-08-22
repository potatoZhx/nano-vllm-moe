from scripts.analyze_cpu_route_causes import analyze_profile


def _record(step_id, rows, statuses, draft_row):
    return {
        "step_id": step_id,
        "metadata_layer_execution_route_rows": {"0": rows},
        "metadata_layer_execution_route_status": {"0": statuses},
        "draft_original_route_rows": [[[[*draft_row]]]],
    }


def test_analyze_profile_reports_per_layer_compute_and_route_causes():
    payload = {
        "model_verify_call_records": [
            _record(2, [[0, 1], [2, 3]], [[1, 2], [2, 1]], [2, 4]),
            _record(4, [[4, 5], [6, 7]], [[2, 2], [2, 1]], [6, 9]),
        ],
        "model_transfer_lifecycle_events": [
            {
                "event": "admission_candidate",
                "step_id": 1,
                "layer_idx": 0,
                "expert_idx": 2,
            },
            {
                "event": "evict",
                "step_id": 1,
                "layer_idx": 0,
                "expert_idx": 1,
            },
            {
                "event": "admission_candidate",
                "step_id": 3,
                "layer_idx": 0,
                "expert_idx": 4,
            },
            {
                "event": "admission_candidate",
                "step_id": 3,
                "layer_idx": 0,
                "expert_idx": 6,
            },
            {
                "event": "submit",
                "step_id": 3,
                "layer_idx": 0,
                "expert_idx": 4,
                "active_slot_prev_expert": 0,
                "source": "verify_segment",
            },
        ],
    }

    result = analyze_profile(payload, horizon_steps=8)

    assert result["verify_calls"] == 2
    assert result["active_routes"] == 8
    assert result["cpu_routes"] == 5
    assert result["cpu_route_ratio"] == 0.625
    assert result["cpu_routes_per_verify_per_layer"] == 2.5
    assert result["cpu_experts_per_verify_per_layer"] == 2.5
    assert result["cpu_routes_per_verified_token_per_layer"] == 1.25
    assert result["route_cause_counts"] == {
        "evicted_before_use": 1,
        "candidate_not_admitted": 2,
        "submitted_but_still_cpu": 1,
        "not_in_candidate_set": 1,
    }
    assert result["draft_prediction"]["route_recall"] == 0.5
    assert result["draft_prediction"]["unpredicted_share_of_cpu_routes"] == 0.0
