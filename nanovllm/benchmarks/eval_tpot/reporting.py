"""CSV, JSON-adjacent, and Markdown reporting for TPOT benchmarks."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "status",
        "case_name",
        "dataset",
        "sample_index",
        "source_index",
        "sample_id",
        "optimized_config",
        "allocation_mode",
        "segment_size",
        "cache_ratio",
        "max_output_tokens",
        "ignore_eos",
        "max_draft_tokens",
        "draft_stop_policy",
        "verify_prefetch_max_per_boundary",
        "verify_prefetch_rank_multiplier",
        "repeat",
        "runtime_seed",
        "decode_driver",
        "batch_size",
        "prompt_tokens_original",
        "prompt_tokens",
        "prompt_truncated",
        "generated_output_tokens",
        "decode_token_intervals",
        "output_sequence_count",
        "output_fixed_length_ok",
        "output_validation_error",
        "max_repeated_token_run",
        "prefill_sec",
        "decode_sec",
        "prefill_step_wall_ms_sum",
        "decode_step_wall_ms_sum",
        "decode_step_wall_ms_mean",
        "decode_step_wall_ms_p50",
        "decode_step_wall_ms_p90",
        "decode_step_wall_ms_p95",
        "decode_step_wall_ms_max",
        "elapsed_sec",
        "tpot_ms",
        "decode_tok_s",
        "aggregate_tpot_ms_per_output_token",
        "aggregate_decode_tok_s",
        "throughput_output_tok_s",
        "prefill_steps",
        "decode_steps",
        "max_tokens_limit",
        "ignore_eos",
        "stopped_by",
        "outputs_digest",
        "profile_decode_phase_output_tok_s",
        "profile_wall_minus_spec_step_ms",
        "profile_wall_minus_engine_step_ms",
        "profile_wall_ms_per_verify",
        "profile_spec_step_ms_per_verify",
        "profile_acceptance_rate",
        "profile_spec_spec_step_ms",
        "profile_step_ms",
        "profile_spec_engine_ms",
        "profile_spec_draft_ms",
        "profile_spec_verify_ms",
        "profile_spec_run_draft_calls",
        "profile_spec_run_verify_calls",
        "profile_spec_draft_tpot_early_stop_count",
        "profile_spec_draft_alpha_early_stop_count",
        "profile_spec_draft_tpot_draft_ms_ema",
        "profile_spec_draft_tpot_verify_ms_ema",
        "profile_draft_forward_ms",
        "profile_verify_forward_ms",
        "profile_model_graph_hit_rate",
        "profile_model_graph_replay_count",
        "profile_model_realized_cpu_expert_count",
        "profile_model_verify_kt_hybrid_segment_graph_replay_count",
        "profile_model_verify_cpu_routes_sum",
        "profile_model_verify_realized_cpu_expert_count_sum",
        "profile_model_verify_pre_transfer_cache_miss_sum",
        "profile_model_verify_pre_transfer_active_count_sum",
        "profile_model_run_verify_kt_hybrid_metadata_wait_ms",
        "profile_model_run_verify_kt_hybrid_metadata_collect_ms",
        "profile_model_run_verify_kt_hybrid_metadata_observe_ms",
        "profile_model_verify_segment_graph_replay_enqueue_ms",
        "profile_model_verify_tpot_dynamic_budget_applied_count",
        "profile_model_verify_tpot_dynamic_budget_token_sum",
        "profile_model_verify_tpot_dynamic_budget_value_sum",
        "profile_model_draft_perfect_reject_events",
        "profile_model_draft_perfect_followup_events",
        "profile_model_draft_perfect_checked_tokens",
        "profile_model_draft_perfect_perfect_tokens",
        "profile_model_draft_perfect_token_rate",
        "profile_model_draft_perfect_prefix_ge1_events",
        "profile_model_draft_perfect_prefix_ge1_rate",
        "profile_model_draft_perfect_perfect_prefix_token_sum",
        "profile_model_draft_perfect_route_total",
        "profile_model_draft_perfect_route_miss",
        "profile_model_draft_perfect_route_miss_ratio",
        "profile_model_draft_perfect_coverage_total",
        "profile_model_draft_perfect_coverage_hit",
        "profile_model_draft_perfect_coverage_ratio",
        "profile_model_draft_perfect_pred_row_match_ratio",
        "profile_model_draft_perfect_input_row_match_ratio",
        "profile_model_draft_perfect_oracle_covered_tokens",
        "profile_model_draft_perfect_oracle_covered_token_rate",
        "profile_model_draft_perfect_oracle_prefix_token_sum",
        "profile_model_draft_perfect_oracle_prefix_ge1_events",
        "profile_model_draft_perfect_oracle_prefix_ge1_rate",
        "profile_model_draft_perfect_refill_events",
        "profile_model_draft_perfect_refill_promoted",
        "profile_model_draft_perfect_refill_cpu_experts",
        "profile_model_draft_perfect_refill_skipped_inflight_events",
        "profile_model_draft_perfect_refill_skipped_inflight_count",
        "profile_model_metadata_offload_ms",
        "profile_model_prefetch_wait_ms",
        "profile_json",
        "skip_reason",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def write_summary_csv(summaries: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "dataset",
        "optimized_config",
        "allocation_mode",
        "segment_size",
        "cache_ratio",
        "max_output_tokens",
        "ignore_eos",
        "max_draft_tokens",
        "draft_stop_policy",
        "verify_prefetch_max_per_boundary",
        "repeat",
        "batch_size",
        "sample_count",
        "ok_count",
        "tpot_ms_mean",
        "tpot_ms_p50",
        "tpot_ms_p90",
        "tpot_ms_p99",
        "decode_tok_s_mean",
        "throughput_output_tok_s_mean",
        "aggregate_tpot_ms_per_output_token_mean",
        "aggregate_decode_tok_s_mean",
        "prompt_tokens_mean",
        "generated_output_tokens_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)

def write_markdown_report(summary: dict[str, Any], path: Path) -> None:
    metadata = summary["metadata"]
    lines = [
        "# Evaluation Workload TPOT Benchmark",
        "",
        f"- timestamp: `{metadata['timestamp']}`",
        f"- model: `{metadata['model_path']}`",
        f"- request mode: `{metadata['request_mode']}`",
        f"- datasets: `{', '.join(metadata['datasets'])}`",
        f"- optimized config: `{metadata.get('optimized_config', 'none')}`",
        f"- optimized env: `{metadata.get('optimized_env_overrides', {})}`",
        f"- batch size: `{metadata.get('batch_size', 1)}`",
        f"- output directory: `{metadata['output_dir']}`",
        f"- warmup prompt: `{metadata.get('warmup_prompt', '')}`",
        f"- decode driver: `{metadata.get('decode_driver', 'step')}`",
        f"- reset profile after warmup: `{metadata.get('reset_profile_after_warmup', False)}`",
        f"- reset profile before request: `{metadata.get('reset_profile_before_request', False)}`",
        f"- reset seed after warmup: `{metadata.get('reset_seed_after_warmup', False)}`",
        f"- fail on output validation error: `{metadata.get('fail_on_output_validation_error', True)}`",
        f"- profile collected: `{metadata.get('collect_profile', False)}`",
        f"- engine profile: `{metadata.get('engine_profile', False)}`",
        "",
        "## Summary",
        "",
        "| dataset | opt | B | alloc | seg | ratio | max out | ignore EOS | K | stop | vpb | rep | ok/sample | request TPOT ms | aggregate ms/tok | aggregate tok/s | P50 | P90 | P99 | prompt tok mean |",
        "|:---|:---|---:|:---|---:|---:|---:|:---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["summaries"]:
        lines.append(
            "| "
            f"{row['dataset']} | {row.get('optimized_config', 'none')} | "
            f"{row.get('batch_size', 1)} | "
            f"{row['allocation_mode']} | "
            f"{row['segment_size']} | {row['cache_ratio']:.4f} | "
            f"{'EOS' if int(row['max_output_tokens']) <= 0 else row['max_output_tokens']} | "
            f"{'true' if row.get('ignore_eos', False) else 'false'} | "
            f"{row['max_draft_tokens']} | "
            f"{row.get('draft_stop_policy', '')} | "
            f"{row.get('verify_prefetch_max_per_boundary', 0)} | "
            f"{row['repeat']} | "
            f"{row['ok_count']}/{row['sample_count']} | "
            f"{row['tpot_ms_mean']:.3f} | "
            f"{row.get('aggregate_tpot_ms_per_output_token_mean', row['tpot_ms_mean']):.3f} | "
            f"{row.get('aggregate_decode_tok_s_mean', row['decode_tok_s_mean']):.3f} | "
            f"{row['tpot_ms_p50']:.3f} | "
            f"{row['tpot_ms_p90']:.3f} | {row['tpot_ms_p99']:.3f} | "
            f"{row['prompt_tokens_mean']:.1f} |"
        )
    if summary["failures"]:
        lines.extend(["", "## Failures", ""])
        for failure in summary["failures"]:
            lines.append(
                f"- `{failure.get('case_name', '')}` sample=`{failure.get('sample_id', '')}`: "
                f"{failure.get('error', failure.get('skip_reason', 'unknown'))}"
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

def flatten_rows(case_summaries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_summary in case_summaries:
        rows.extend(case_summary.get("rows", []))
    return rows
