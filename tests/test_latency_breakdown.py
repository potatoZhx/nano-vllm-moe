from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "scripts")
)

from latency_breakdown import (  # noqa: E402
    aggregate_breakdowns,
    build_request_breakdown,
)


def synthetic_row() -> dict:
    return {
        "dataset": "mt_bench",
        "sample_id": "81",
        "sample_index": 0,
        "runtime_seed": 20260719,
        "outputs_digest": "digest",
        "generated_output_tokens": 101,
        "decode_token_intervals": 100,
        "decode_sec": 1.0,
    }


def synthetic_profile() -> dict:
    return {
        "spec_draft_total_ms": 600.0,
        "spec_verify_accept_ready_ms": 300.0,
        "model_draft_segment_cuda_event_ms": 350.0,
        "model_latency_draft_tail_cuda_event_ms": 30.0,
        "model_latency_draft_sample_cuda_event_ms": 20.0,
        "model_direct_active_prefetch_sync_wait_ms": 40.0,
        "model_verify_prefetch_transfer_wait_ms": 10.0,
        "model_verify_segment_cuda_event_ms": 200.0,
        "model_latency_verify_lm_head_cuda_event_ms": 20.0,
        "model_verify_op_kt_cpuinfer_sync_ms": 50.0,
        "model_verify_op_kt_output_cpu_to_gpu_copy_ms": 20.0,
        "spec_run_draft_infer_ms_total": 500.0,
        "spec_draft_loop_ms": 520.0,
        "spec_start_draft_ms": 5.0,
        "spec_rollback_ms": 5.0,
        "spec_prepare_verify_ms": 10.0,
        "spec_draft_entry_ms": 20.0,
        "spec_draft_initial_policy_ms": 30.0,
        "spec_verify_prefetch_call_ms": 10.0,
        "spec_run_verify_infer_ms_total": 260.0,
        "model_verify_prepare_prefill_ms": 10.0,
        "spec_accept_ms": 30.0,
        "decode_step_wall_ms": 950.0,
        "decode_schedule_ms": 20.0,
        "decode_spec_engine_ms": 905.0,
        "decode_postprocess_ms": 10.0,
        "spec_run_draft_calls": 10.0,
        "spec_run_verify_calls": 5.0,
        "model_draft_segment_cuda_event_count": 40.0,
        "model_latency_draft_tail_cuda_event_count": 10.0,
        "model_latency_draft_sample_cuda_event_count": 10.0,
        "model_verify_segment_cuda_event_count": 20.0,
        "model_latency_verify_lm_head_cuda_event_count": 5.0,
        "model_verify_op_kt_cpuinfer_sync_count": 240.0,
        "model_verify_op_kt_output_cpu_to_gpu_copy_count": 240.0,
        # Hidden transfer diagnostics must never enter the additive hierarchy.
        "model_prefetch_completion_latency_ms": 9999.0,
        "model_draft_segment_indexed_prefetch_est_transfer_ms": 8888.0,
    }


def test_request_breakdown_closes_and_attributes_residuals() -> None:
    result = build_request_breakdown(
        synthetic_row(), synthetic_profile()
    )

    assert result["passed"]
    assert result["totals_ms"] == {
        "draft_gpu_compute": 400.0,
        "draft_transfer_exposed": 50.0,
        "draft_other": 150.0,
        "verify_gpu_compute": 150.0,
        "verify_cpu_compute_exposed": 50.0,
        "verify_transfer_exposed": 20.0,
        "verify_other": 80.0,
        "decode_residual": 100.0,
    }
    assert abs(result["closure"]["total_error_ms"]) < 1e-9
    assert result["other_sources_ms"]["draft_other"] == {
        "draft_call_host_and_tail": 60.0,
        "draft_policy_and_sequence": 20.0,
        "draft_setup_and_handoff": 20.0,
        "draft_entry_and_initial_policy": 50.0,
        "draft_prefetch_orchestration": 0.0,
        "draft_other_unattributed": 0.0,
    }
    assert result["other_sources_ms"]["verify_other"] == {
        "verify_prepare_and_context": 10.0,
        "verify_modelrunner_external_after_prepare": 30.0,
        "verify_acceptance": 30.0,
        "verify_call_boundary": 10.0,
        "verify_other_unattributed": 0.0,
    }
    assert result["other_sources_ms"]["decode_residual"] == {
        "decode_scheduler": 20.0,
        "decode_spec_post_verify": 5.0,
        "decode_postprocess": 10.0,
        "decode_engine_wrapper": 15.0,
        "decode_driver": 50.0,
        "decode_residual_unattributed": 0.0,
    }


def test_hidden_transfer_and_overlapping_diagnostics_are_not_additive() -> None:
    profile = synthetic_profile()
    base = build_request_breakdown(synthetic_row(), profile)
    profile["model_prefetch_completion_latency_ms"] *= 100.0
    profile["model_verify_segment_prefetch_hook_ms"] = 700.0
    changed = build_request_breakdown(synthetic_row(), profile)

    assert changed["totals_ms"] == base["totals_ms"]
    diagnostic = next(
        item
        for item in changed["diagnostics"]
        if item["source"] == "verify_prefetch_hook"
    )
    assert diagnostic["total_ms"] == 700.0
    assert diagnostic["overlaps_gpu"]
    assert not diagnostic["additive"]


def test_missing_event_coverage_fails_request_gate() -> None:
    profile = synthetic_profile()
    profile["model_verify_op_kt_cpuinfer_sync_count"] = 239.0
    result = build_request_breakdown(synthetic_row(), profile)

    assert not result["passed"]
    assert any(
        "event coverage verify_cpu_event" in message
        for message in result["errors"]
    )


def test_pooled_values_use_total_intervals() -> None:
    first = build_request_breakdown(
        synthetic_row(), synthetic_profile()
    )
    second_row = synthetic_row()
    second_row["decode_token_intervals"] = 200
    second_row["decode_sec"] = 2.0
    second_profile = {
        key: value * 2.0 if isinstance(value, float) else value
        for key, value in synthetic_profile().items()
    }
    # Counts scale with calls in a second, twice-as-long request.
    second_profile["spec_run_draft_calls"] = 20.0
    second_profile["spec_run_verify_calls"] = 10.0
    second = build_request_breakdown(second_row, second_profile)

    aggregate = aggregate_breakdowns([first, second])
    assert aggregate["total_decode_token_intervals"] == 300
    assert aggregate["pooled_per_token_ms"]["tpot"] == 10.0
    assert (
        aggregate["pooled_per_token_ms"]["draft_gpu_compute"]
        == 4.0
    )
