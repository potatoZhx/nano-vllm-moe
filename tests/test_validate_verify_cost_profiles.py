import json

from scripts.validate_verify_cost_profiles import validate_profile


def test_validate_profile_checks_bucket_execution_and_step_join(tmp_path):
    path = tmp_path / "sample0000.json"
    path.write_text(
        json.dumps(
            {
                "verify_cost_measurement": {
                    "enabled": True,
                    "target": "spec.verify_accept_ready_ms",
                    "profile_cuda_sync": False,
                },
                "model_verify_call_records": [
                    {
                        "call_index": 0,
                        "step_id": 7,
                        "token_count": 2,
                        "bucket": 3,
                        "metadata_execution_layer_count": 2,
                        "metadata_execution_active_routes_sum": 12,
                        "metadata_execution_cpu_routes_sum": 3,
                        "metadata_execution_cpu_experts_sum": 3,
                        "metadata_execution_available": True,
                        "padding_token_count": 1,
                        "used_cuda_graph": True,
                        "used_kt_hybrid": True,
                        "outputs_ready": True,
                        "return_logits": False,
                        "metadata_layer_execution_cpu_route_counts": {
                            "0": [1, 0, 1],
                            "1": [0, 1, 0],
                        },
                        "metadata_layer_execution_route_counts": {
                            "0": [2, 2, 2],
                            "1": [1, 3, 2],
                        },
                        "stream_ms_available": True,
                        "stream_ms": 11.0,
                    }
                ],
                "spec_step_traces": [
                    {
                        "verify_call_index": 0,
                        "verify_accept_ready_ms": 12.0,
                        "verify_model_call_ms": 11.5,
                        "verify_accept_ms": 0.25,
                        "verify_token_count": 2,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result = validate_profile(path, num_layers=2, num_experts=3, top_k=2)
    assert result["valid"] is True
    assert result["verify_calls"] == 1


def test_validate_profile_rejects_logical_only_route_count(tmp_path):
    path = tmp_path / "sample0000.json"
    path.write_text(
        json.dumps(
            {
                "verify_cost_measurement": {
                    "enabled": True,
                    "target": "spec.verify_accept_ready_ms",
                    "profile_cuda_sync": False,
                },
                "model_verify_call_records": [
                    {
                        "call_index": 0,
                        "step_id": 7,
                        "token_count": 2,
                        "bucket": 3,
                        "metadata_execution_layer_count": 2,
                        "metadata_execution_active_routes_sum": 8,
                        "metadata_execution_cpu_routes_sum": 0,
                        "metadata_execution_cpu_experts_sum": 0,
                        "metadata_execution_available": True,
                        "padding_token_count": 1,
                        "used_cuda_graph": True,
                        "used_kt_hybrid": True,
                        "outputs_ready": True,
                        "return_logits": False,
                        "metadata_layer_execution_cpu_route_counts": {
                            "0": [0, 0, 0],
                            "1": [0, 0, 0],
                        },
                        "metadata_layer_execution_route_counts": {
                            "0": [2, 2, 2],
                            "1": [1, 3, 2],
                        },
                        "stream_ms_available": True,
                        "stream_ms": 11.0,
                    }
                ],
                "spec_step_traces": [
                    {
                        "verify_call_index": 0,
                        "verify_accept_ready_ms": 12.0,
                        "verify_model_call_ms": 11.5,
                        "verify_accept_ms": 0.25,
                        "verify_token_count": 2,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result = validate_profile(path, num_layers=2, num_experts=3, top_k=2)
    assert result["valid"] is False
    assert "bucket*top_k*layers" in "\n".join(result["errors"])


def test_validate_profile_rejects_duplicate_trace_and_non_subset_cpu_counts(
    tmp_path,
):
    path = tmp_path / "sample0000.json"
    path.write_text(
        json.dumps(
            {
                "verify_cost_measurement": {
                    "enabled": True,
                    "target": "spec.verify_accept_ready_ms",
                    "profile_cuda_sync": False,
                },
                "model_verify_call_records": [
                    {
                        "call_index": 0,
                        "step_id": 7,
                        "token_count": 2,
                        "bucket": 3,
                        "padding_token_count": 1,
                        "used_cuda_graph": True,
                        "used_kt_hybrid": True,
                        "outputs_ready": True,
                        "return_logits": False,
                        "metadata_execution_available": True,
                        "metadata_execution_layer_count": 1,
                        "metadata_execution_active_routes_sum": 6,
                        "metadata_execution_cpu_routes_sum": 3,
                        "metadata_execution_cpu_experts_sum": 1,
                        "metadata_layer_execution_cpu_route_counts": {
                            "0": [3, 0, 0],
                        },
                        "metadata_layer_execution_route_counts": {
                            "0": [2, 2, 2],
                        },
                        "stream_ms_available": True,
                        "stream_ms": 11.0,
                    }
                ],
                "spec_step_traces": [
                    {
                        "verify_call_index": 0,
                        "verify_accept_ready_ms": 12.0,
                        "verify_model_call_ms": 11.5,
                        "verify_accept_ms": 0.25,
                        "verify_token_count": 2,
                    },
                    {
                        "verify_call_index": 0,
                        "verify_accept_ready_ms": 12.0,
                        "verify_model_call_ms": 11.5,
                        "verify_accept_ms": 0.25,
                        "verify_token_count": 2,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    result = validate_profile(path, num_layers=1, num_experts=3, top_k=2)
    errors = "\n".join(result["errors"])
    assert result["valid"] is False
    assert "duplicate verify_call_index=0" in errors
    assert "CPU counts exceed full counts" in errors
