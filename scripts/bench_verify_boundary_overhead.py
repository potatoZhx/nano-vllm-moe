#!/usr/bin/env python3
"""Compare verify latency with and without verify-boundary prefetch.

This script is intentionally narrower than ``bench_segment_graph_no_prefetch.py``:
it keeps the predictive prefetch runtime enabled so draft-side prefetch, history,
metadata collection, and cache policy stay close to the normal per-layer-slot
benchmark.  Only verify segment-boundary prefetch is disabled in the
``verify_prefetch_off`` cases by setting the verify boundary budget to zero.

Use the ``verify_prefetch_off`` K sweep to pick a no-prefetch case with similar
CPU route work to the prefetch-enabled reference.  That makes the verify latency
delta a cleaner estimate of boundary transfer/scheduling overhead.

Example:

    conda activate nano_moe
    cd /home/linke/nano-vllm-moe
    rm -rf results/verify_boundary_overhead
    CUDA_VISIBLE_DEVICES=2 python scripts/bench_verify_boundary_overhead.py \
        --output-dir results/verify_boundary_overhead \
        --gpu-memory-utilization 0.99 \
        --cache-ratios 0.3125 \
        --output-lens 512 \
        --prefetch-on-max-draft-tokens-values 12 \
        --verify-off-max-draft-tokens-values 8,10,12 \
        --segment-sizes 12 \
        --allocation-mode profile_weighted \
        --slot-buckets 4 \
        --slot-max-bucket-ratio 2.0 \
        --slot-profile-csv pre_exps/exp_and_figs/unique/unique_count_plot_summary_n1024.csv \
        --kt-num-threads 16
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROMPT_TEXT = (
    "Sparse mixture-of-experts inference keeps only part of each layer's expert weights "
    "in GPU memory. Explain how speculative decoding can overlap expert prefetch with "
    "draft and verify segment computation while preserving exact verification semantics. "
    "Discuss routing-score metadata, bounded transfer budgets, cache eviction protection, "
    "and why late best-effort transfers should be discarded instead of blocking compute."
)

MODEL_PATH = "/data1/models/Qwen3-30B-A3B"
DEFAULT_PROFILE = "results/reroute_impl_20260531/offline_profile_20260531_203257.safetensors"
DEFAULT_PREDICTOR_PATH = "random_cache_srdp_scripts-1/res/run_20260614_133025"
DEFAULT_SLOT_PROFILE_CSV = "pre_exps/exp_and_figs/unique/unique_count_plot_summary_n1024.csv"
NUM_MOE_LAYERS = 48


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {value}")


def _parse_csv(values: str, cast) -> list:
    return [cast(item.strip()) for item in values.split(",") if item.strip()]


def _mode_values(args: argparse.Namespace) -> list[str]:
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    valid = {"verify_prefetch_on", "verify_prefetch_off"}
    bad = [mode for mode in modes if mode not in valid]
    if bad:
        raise argparse.ArgumentTypeError(f"invalid modes: {','.join(bad)}")
    if not modes:
        raise argparse.ArgumentTypeError("--modes must not be empty")
    return modes


def build_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    modes = _mode_values(args)
    on_ks = _parse_csv(args.prefetch_on_max_draft_tokens_values, int)
    off_ks = _parse_csv(args.verify_off_max_draft_tokens_values, int)
    cases: list[dict[str, Any]] = []
    for output_len in _parse_csv(args.output_lens, int):
        for cache_ratio in _parse_csv(args.cache_ratios, float):
            for segment_size in _parse_csv(args.segment_sizes, int):
                for repeat in range(int(args.repeats)):
                    if "verify_prefetch_on" in modes:
                        for max_draft_tokens in on_ks:
                            cases.append(
                                {
                                    "mode": "verify_prefetch_on",
                                    "verify_prefetch_enabled": True,
                                    "output_len": int(output_len),
                                    "cache_ratio": float(cache_ratio),
                                    "max_draft_tokens": int(max_draft_tokens),
                                    "segment_size": int(segment_size),
                                    "repeat": int(repeat),
                                }
                            )
                    if "verify_prefetch_off" in modes:
                        for max_draft_tokens in off_ks:
                            cases.append(
                                {
                                    "mode": "verify_prefetch_off",
                                    "verify_prefetch_enabled": False,
                                    "output_len": int(output_len),
                                    "cache_ratio": float(cache_ratio),
                                    "max_draft_tokens": int(max_draft_tokens),
                                    "segment_size": int(segment_size),
                                    "repeat": int(repeat),
                                }
                            )
    return cases


def _verify_bucket_steps(args: argparse.Namespace, cases: list[dict[str, Any]]) -> str:
    buckets = set(_parse_csv(args.verify_cuda_graph_bucket_steps, int))
    if args.auto_add_verify_k_plus_one:
        for case in cases:
            buckets.add(int(case["max_draft_tokens"]) + 1)
    return ",".join(str(value) for value in sorted(buckets))


def _case_name(case: dict[str, Any]) -> str:
    ratio_pct = int(round(float(case["cache_ratio"]) * 10000))
    return (
        f"{case['mode']}_seg{int(case['segment_size'])}_"
        f"ratio{ratio_pct:04d}_l{int(case['output_len'])}_"
        f"k{int(case['max_draft_tokens'])}_r{int(case['repeat'])}"
    )


def _ep_float(engine_profile: dict[str, Any], key: str) -> float:
    value = engine_profile.get(key)
    if value is None:
        value = engine_profile.get(f"model_{key}", 0.0)
    return float(value or 0.0)


def _ep_int(engine_profile: dict[str, Any], key: str) -> int:
    return int(round(_ep_float(engine_profile, key)))


def _has_ep(engine_profile: dict[str, Any], key: str) -> bool:
    return key in engine_profile or f"model_{key}" in engine_profile


def _ep_float_prefer(
    engine_profile: dict[str, Any],
    preferred_key: str,
    fallback_key: str,
) -> float:
    if _has_ep(engine_profile, preferred_key):
        return _ep_float(engine_profile, preferred_key)
    return _ep_float(engine_profile, fallback_key)


def _per_layer_sum(
    engine_profile: dict[str, Any],
    suffix: str,
) -> list[float]:
    values: list[float] = []
    for layer_idx in range(NUM_MOE_LAYERS):
        values.append(
            _ep_float_prefer(
                engine_profile,
                f"verify_layer_{layer_idx}_{suffix}",
                f"layer_{layer_idx}_{suffix}",
            )
        )
    return values


def _per_call(values: list[float], calls: int) -> list[float]:
    if calls <= 0:
        return [0.0 for _ in values]
    return [float(value / calls) for value in values]


def _top_layer_cpu_experts(
    cpu_experts_per_call: list[float],
    cpu_routes_per_call: list[float],
    active_experts_per_call: list[float],
    active_routes_per_call: list[float],
    *,
    limit: int = 12,
) -> list[dict[str, float | int]]:
    rows = [
        {
            "layer": layer_idx,
            "cpu_experts_per_call": float(cpu_experts_per_call[layer_idx]),
            "cpu_routes_per_call": float(cpu_routes_per_call[layer_idx]),
            "active_experts_per_call": float(active_experts_per_call[layer_idx]),
            "active_routes_per_call": float(active_routes_per_call[layer_idx]),
        }
        for layer_idx in range(NUM_MOE_LAYERS)
    ]
    rows.sort(
        key=lambda item: (
            -float(item["cpu_experts_per_call"]),
            -float(item["cpu_routes_per_call"]),
            int(item["layer"]),
        )
    )
    return rows[:limit]


def _row_from_raw(
    case: dict[str, Any],
    raw: dict[str, Any],
    wall_elapsed_sec: float,
) -> dict[str, Any]:
    summary = raw.get("summary", {})
    acceptance = summary.get("acceptance", {})
    cache = summary.get("cache", {})
    cuda_graph = summary.get("cuda_graph", {})
    prefetch = summary.get("prefetch", {})
    dual_queue = summary.get("dual_queue", {})
    engine_profile = raw.get("engine_profile", {})

    verify_calls = int(cuda_graph.get("verify_call_count", 0) or 0)
    verify_segment_submit = int(
        prefetch.get("verify_segment_prefetch_submit_count", 0) or 0
    )
    verify_segment_calls = int(
        prefetch.get("verify_segment_prefetch_call_count", 0) or 0
    )
    visible_ms = _ep_float(
        engine_profile, "verify_segment_prefetch_visible_overhead_ms"
    )
    rank_ms = _ep_float(engine_profile, "verify_segment_prefetch_rank_ms")
    rank_limited_count = _ep_float(engine_profile, "verify_segment_prefetch_rank_limited_count")
    rank_limit_sum = _ep_float(engine_profile, "verify_segment_prefetch_rank_limit_sum")
    candidate_ranked_count = _ep_float(engine_profile, "verify_segment_prefetch_candidate_ranked_count")
    candidate_merge_count = _ep_float(engine_profile, "verify_segment_prefetch_candidate_merge_count")
    rank_scan_ms = _ep_float(engine_profile, "verify_segment_prefetch_rank_scan_ms")
    rank_sort_ms = _ep_float(engine_profile, "verify_segment_prefetch_rank_sort_ms")
    prefetch_filter_ms = _ep_float(engine_profile, "verify_segment_prefetch_filter_ms")
    prefetch_victim_ms = _ep_float(engine_profile, "verify_segment_prefetch_victim_select_ms")
    prefetch_reserve_ms = _ep_float(engine_profile, "verify_segment_prefetch_reservation_ms")
    prefetch_transfer_call_ms = _ep_float(engine_profile, "verify_segment_prefetch_begin_transfer_call_ms")
    prefetch_transfer_enqueue_ms = _ep_float(engine_profile, "verify_segment_prefetch_transfer_enqueue_ms")
    prefetch_bookkeeping_ms = _ep_float(engine_profile, "verify_segment_prefetch_bookkeeping_ms")
    boundary_async_enqueue_ms = _ep_float(engine_profile, "verify_boundary_async_prefetch_enqueue_ms")
    boundary_async_worker_queue_wait_ms = _ep_float(engine_profile, "verify_boundary_async_prefetch_worker_queue_wait_ms")
    boundary_async_worker_submit_ms = _ep_float(engine_profile, "verify_boundary_async_prefetch_worker_submit_ms")
    boundary_async_drain_wait_ms = _ep_float(engine_profile, "verify_boundary_async_prefetch_drain_wait_ms")
    metadata_enqueue_ms = _ep_float(engine_profile, "run_verify_kt_hybrid_metadata_enqueue_ms")
    metadata_wait_ms = _ep_float(engine_profile, "run_verify_kt_hybrid_metadata_wait_ms")
    metadata_collect_ms = _ep_float(engine_profile, "run_verify_kt_hybrid_metadata_collect_ms")
    metadata_observe_ms = _ep_float(engine_profile, "run_verify_kt_hybrid_metadata_observe_ms")
    metadata_observe_call_ms = _ep_float(engine_profile, "run_verify_kt_hybrid_metadata_observe_call_ms")
    metadata_record_consumed_ms = _ep_float(engine_profile, "run_verify_kt_hybrid_metadata_record_consumed_ms")
    metadata_mark_access_ms = _ep_float(engine_profile, "run_verify_kt_hybrid_metadata_mark_access_ms")
    metadata_queue_update_ms = _ep_float(engine_profile, "run_verify_kt_hybrid_metadata_queue_update_ms")
    metadata_observe_verify_rank_guard_ms = _ep_float(
        engine_profile, "run_verify_kt_hybrid_metadata_observe_verify_rank_guard_ms"
    )
    metadata_observe_verify_segment_index_ms = _ep_float(
        engine_profile, "run_verify_kt_hybrid_metadata_observe_verify_segment_index_ms"
    )
    metadata_segment_index_rank_cache_rebuild_ms = _ep_float(
        engine_profile, "run_verify_kt_hybrid_metadata_segment_index_rank_cache_rebuild_ms"
    )
    metadata_segment_index_rank_cache_rebuild_count = _ep_float(
        engine_profile, "run_verify_kt_hybrid_metadata_segment_index_rank_cache_rebuild_count"
    )
    metadata_observe_verify_runtime_meta_call_ms = _ep_float(
        engine_profile, "run_verify_kt_hybrid_metadata_observe_verify_runtime_meta_call_ms"
    )
    metadata_profile_async_loop_ms = _ep_float(engine_profile, "verify_metadata_profile_async_loop_ms")
    metadata_sync_status_cpu_ms = _ep_float(engine_profile, "verify_metadata_status_cpu_ms")
    metadata_sync_activation_cpu_ms = _ep_float(engine_profile, "verify_metadata_activation_cpu_ms")
    metadata_sync_profile_loop_ms = _ep_float(engine_profile, "verify_metadata_profile_loop_ms")
    verify_tokens = _ep_float(engine_profile, "verify_tokens_in_total")
    cpu_routes = _ep_float(engine_profile, "verify_cpu_routes_sum")
    cpu_routes_source = "verify_profile" if _has_ep(engine_profile, "verify_cpu_routes_sum") else "aggregate_fallback"
    cpu_compute_source = "verify_profile" if _has_ep(engine_profile, "verify_cpu_compute_ms") else "aggregate_fallback"
    realized_cpu_experts_source = (
        "verify_profile"
        if _has_ep(engine_profile, "verify_realized_cpu_expert_count_sum")
        else "aggregate_fallback"
    )
    cpu_compute_ms = _ep_float(engine_profile, "verify_cpu_compute_ms")
    cpu_prepare_ms = _ep_float(engine_profile, "verify_cpu_prepare_ms")
    cpu_merge_ms = _ep_float(engine_profile, "verify_cpu_to_gpu_merge_ms")
    gpu_compute_ms = _ep_float(engine_profile, "verify_gpu_compute_ms")
    parallel_wall_ms = _ep_float(engine_profile, "verify_parallel_wall_ms")
    parallel_critical_path_ms = _ep_float(engine_profile, "verify_parallel_critical_path_est_ms")
    parallel_overlap_ms = _ep_float(engine_profile, "verify_parallel_overlap_est_ms")
    cpu_wait_ms = _ep_float(engine_profile, "verify_cpu_wait_ms")
    gpu_wait_ms = _ep_float(engine_profile, "verify_gpu_wait_ms")
    realized_cpu_experts = _ep_float(engine_profile, "verify_realized_cpu_expert_count_sum")
    if cpu_compute_ms <= 0.0:
        cpu_compute_ms = _ep_float(engine_profile, "cpu_compute_ms")
    if cpu_prepare_ms <= 0.0:
        cpu_prepare_ms = _ep_float(engine_profile, "cpu_prepare_ms")
    if cpu_merge_ms <= 0.0:
        cpu_merge_ms = _ep_float(engine_profile, "cpu_to_gpu_merge_ms")
    if gpu_compute_ms <= 0.0:
        gpu_compute_ms = _ep_float(engine_profile, "gpu_compute_ms")
    if realized_cpu_experts <= 0.0:
        realized_cpu_experts = _ep_float(engine_profile, "realized_cpu_expert_count_sum")
    layer_cpu_expert_sums = _per_layer_sum(
        engine_profile, "realized_cpu_expert_count_sum"
    )
    layer_cpu_route_sums = _per_layer_sum(engine_profile, "cpu_routes_sum")
    layer_active_expert_sums = _per_layer_sum(
        engine_profile, "active_expert_count_sum"
    )
    layer_active_route_sums = _per_layer_sum(engine_profile, "active_routes_sum")
    layer_profile_counts = _per_layer_sum(engine_profile, "moe_profile_count")
    layer_cpu_experts_per_call = _per_call(layer_cpu_expert_sums, verify_calls)
    layer_cpu_routes_per_call = _per_call(layer_cpu_route_sums, verify_calls)
    layer_active_experts_per_call = _per_call(layer_active_expert_sums, verify_calls)
    layer_active_routes_per_call = _per_call(layer_active_route_sums, verify_calls)
    layer_top_cpu_experts = _top_layer_cpu_experts(
        layer_cpu_experts_per_call,
        layer_cpu_routes_per_call,
        layer_active_experts_per_call,
        layer_active_routes_per_call,
    )
    segment_event_ms = _ep_float(engine_profile, "verify_segment_cuda_event_ms")
    segment_event_count = _ep_float(engine_profile, "verify_segment_cuda_event_count")
    pre_transfer_miss = _ep_float(
        engine_profile, "verify_pre_transfer_cache_miss_sum"
    )
    pre_transfer_active = _ep_float(
        engine_profile, "verify_pre_transfer_active_count_sum"
    )
    moe_profile_count = _ep_float(engine_profile, "verify_moe_profile_count")
    verify_segment_graph_replays = int(
        cuda_graph.get("verify_kt_hybrid_segment_graph_replay_count", 0) or 0
    )
    submitted_bytes_by_source = dual_queue.get("submitted_bytes_by_source", {})
    verify_segment_bytes = float(submitted_bytes_by_source.get("verify_segment", 0.0) or 0.0)

    return {
        "name": _case_name(case),
        "mode": str(case["mode"]),
        "verify_prefetch_enabled": bool(case["verify_prefetch_enabled"]),
        "output_len": int(case["output_len"]),
        "cache_ratio": float(case["cache_ratio"]),
        "max_draft_tokens": int(case["max_draft_tokens"]),
        "segment_size": int(case["segment_size"]),
        "repeat": int(case["repeat"]),
        "wall_elapsed_sec": float(wall_elapsed_sec),
        "generated_output_tokens": int(raw.get("generated_output_tokens", 0) or 0),
        "throughput_output_tok_s": float(summary.get("throughput_output_tok_s", 0.0) or 0.0),
        "decode_phase_output_tok_s": float(summary.get("decode_phase_output_tok_s", 0.0) or 0.0),
        "draft_forward_ms_avg": float(summary.get("draft_forward_ms_avg", 0.0) or 0.0),
        "verify_forward_ms_avg": float(summary.get("verify_forward_ms_avg", 0.0) or 0.0),
        "acceptance_rate": float(acceptance.get("acceptance_rate", 0.0) or 0.0),
        "drafted_tokens_total": int(acceptance.get("drafted_tokens_total", 0) or 0),
        "accepted_draft_tokens_total": int(acceptance.get("accepted_draft_tokens_total", 0) or 0),
        "route_hit_rate": float(cache.get("true_route_hit_rate", cache.get("route_hit_rate", 0.0)) or 0.0),
        "avg_miss_routes_per_layer": float(cache.get("avg_miss_per_layer", 0.0) or 0.0),
        "avg_active_per_layer": float(cache.get("avg_active_per_layer", 0.0) or 0.0),
        "verify_calls": verify_calls,
        "verify_segment_graph_replays": verify_segment_graph_replays,
        "verify_graph_coverage": (
            float(verify_segment_graph_replays / verify_calls) if verify_calls else 0.0
        ),
        "verify_tokens_total": verify_tokens,
        "verify_tokens_per_call": float(verify_tokens / verify_calls) if verify_calls else 0.0,
        "verify_cpu_routes_total": cpu_routes,
        "verify_cpu_routes_source": cpu_routes_source,
        "verify_cpu_routes_per_call": float(cpu_routes / verify_calls) if verify_calls else 0.0,
        "verify_cpu_routes_per_layer_call": (
            float(cpu_routes / (verify_calls * NUM_MOE_LAYERS)) if verify_calls else 0.0
        ),
        "verify_realized_cpu_expert_count_total": realized_cpu_experts,
        "verify_realized_cpu_expert_count_source": realized_cpu_experts_source,
        "verify_realized_cpu_expert_count_per_call": (
            float(realized_cpu_experts / verify_calls) if verify_calls else 0.0
        ),
        "verify_realized_cpu_expert_count_per_layer_call": (
            float(realized_cpu_experts / (verify_calls * NUM_MOE_LAYERS)) if verify_calls else 0.0
        ),
        "verify_layer_realized_cpu_expert_count_total": layer_cpu_expert_sums,
        "verify_layer_realized_cpu_expert_count_per_call": layer_cpu_experts_per_call,
        "verify_layer_cpu_routes_total": layer_cpu_route_sums,
        "verify_layer_cpu_routes_per_call": layer_cpu_routes_per_call,
        "verify_layer_active_expert_count_total": layer_active_expert_sums,
        "verify_layer_active_expert_count_per_call": layer_active_experts_per_call,
        "verify_layer_active_routes_total": layer_active_route_sums,
        "verify_layer_active_routes_per_call": layer_active_routes_per_call,
        "verify_layer_moe_profile_count": layer_profile_counts,
        "verify_layer_cpu_expert_top": layer_top_cpu_experts,
        "verify_cpu_compute_ms_total": cpu_compute_ms,
        "verify_cpu_compute_source": cpu_compute_source,
        "verify_cpu_compute_ms_per_call": (
            float(cpu_compute_ms / verify_calls) if verify_calls else 0.0
        ),
        "verify_cpu_compute_ms_per_route": (
            float(cpu_compute_ms / cpu_routes) if cpu_routes > 0.0 else 0.0
        ),
        "verify_cpu_compute_ms_per_expert": (
            float(cpu_compute_ms / realized_cpu_experts) if realized_cpu_experts > 0.0 else 0.0
        ),
        "verify_cpu_prepare_ms_total": cpu_prepare_ms,
        "verify_cpu_prepare_ms_per_call": (
            float(cpu_prepare_ms / verify_calls) if verify_calls else 0.0
        ),
        "verify_cpu_to_gpu_merge_ms_total": cpu_merge_ms,
        "verify_cpu_to_gpu_merge_ms_per_call": (
            float(cpu_merge_ms / verify_calls) if verify_calls else 0.0
        ),
        "verify_gpu_compute_ms_total": gpu_compute_ms,
        "verify_gpu_compute_ms_per_call": (
            float(gpu_compute_ms / verify_calls) if verify_calls else 0.0
        ),
        "verify_parallel_wall_ms_total": parallel_wall_ms,
        "verify_parallel_wall_ms_per_call": (
            float(parallel_wall_ms / verify_calls) if verify_calls else 0.0
        ),
        "verify_parallel_critical_path_ms_total": parallel_critical_path_ms,
        "verify_parallel_overlap_ms_total": parallel_overlap_ms,
        "verify_cpu_wait_ms_total": cpu_wait_ms,
        "verify_gpu_wait_ms_total": gpu_wait_ms,
        "verify_segment_cuda_event_ms_total": segment_event_ms,
        "verify_segment_cuda_event_count": int(round(segment_event_count)),
        "verify_segment_cuda_event_ms_per_call": (
            float(segment_event_ms / verify_calls) if verify_calls else 0.0
        ),
        "verify_segment_cuda_event_ms_per_segment": (
            float(segment_event_ms / segment_event_count) if segment_event_count > 0.0 else 0.0
        ),
        "verify_pre_transfer_miss_total": pre_transfer_miss,
        "verify_pre_transfer_miss_per_call": (
            float(pre_transfer_miss / verify_calls) if verify_calls else 0.0
        ),
        "verify_pre_transfer_active_per_call": (
            float(pre_transfer_active / verify_calls) if verify_calls else 0.0
        ),
        "verify_moe_profile_count": int(round(moe_profile_count)),
        "verify_segment_prefetch_call_count": verify_segment_calls,
        "verify_segment_prefetch_submit_count": verify_segment_submit,
        "verify_segment_prefetch_submit_per_call": (
            float(verify_segment_submit / verify_calls) if verify_calls else 0.0
        ),
        "verify_segment_prefetch_visible_overhead_ms": visible_ms,
        "verify_segment_prefetch_visible_overhead_per_call_ms": (
            float(visible_ms / verify_calls) if verify_calls else 0.0
        ),
        "verify_segment_prefetch_rank_ms": rank_ms,
        "verify_segment_prefetch_rank_per_call_ms": (
            float(rank_ms / verify_calls) if verify_calls else 0.0
        ),
        "verify_segment_prefetch_rank_limited_count": int(round(rank_limited_count)),
        "verify_segment_prefetch_rank_limit_per_submit": (
            float(rank_limit_sum / rank_limited_count) if rank_limited_count > 0.0 else 0.0
        ),
        "verify_segment_prefetch_candidate_ranked_per_submit": (
            float(candidate_ranked_count / verify_segment_calls) if verify_segment_calls else 0.0
        ),
        "verify_segment_prefetch_candidate_merge_per_submit": (
            float(candidate_merge_count / verify_segment_calls) if verify_segment_calls else 0.0
        ),
        "verify_segment_prefetch_rank_scan_per_call_ms": (
            float(rank_scan_ms / verify_calls) if verify_calls else 0.0
        ),
        "verify_segment_prefetch_rank_sort_per_call_ms": (
            float(rank_sort_ms / verify_calls) if verify_calls else 0.0
        ),
        "verify_segment_prefetch_filter_per_call_ms": (
            float(prefetch_filter_ms / verify_calls) if verify_calls else 0.0
        ),
        "verify_segment_prefetch_victim_select_per_call_ms": (
            float(prefetch_victim_ms / verify_calls) if verify_calls else 0.0
        ),
        "verify_segment_prefetch_reservation_per_call_ms": (
            float(prefetch_reserve_ms / verify_calls) if verify_calls else 0.0
        ),
        "verify_segment_prefetch_begin_transfer_call_per_call_ms": (
            float(prefetch_transfer_call_ms / verify_calls) if verify_calls else 0.0
        ),
        "verify_segment_prefetch_transfer_enqueue_per_call_ms": (
            float(prefetch_transfer_enqueue_ms / verify_calls) if verify_calls else 0.0
        ),
        "verify_segment_prefetch_bookkeeping_per_call_ms": (
            float(prefetch_bookkeeping_ms / verify_calls) if verify_calls else 0.0
        ),
        "verify_boundary_async_enqueue_per_call_ms": (
            float(boundary_async_enqueue_ms / verify_calls) if verify_calls else 0.0
        ),
        "verify_boundary_async_worker_queue_wait_per_call_ms": (
            float(boundary_async_worker_queue_wait_ms / verify_calls) if verify_calls else 0.0
        ),
        "verify_boundary_async_worker_submit_per_call_ms": (
            float(boundary_async_worker_submit_ms / verify_calls) if verify_calls else 0.0
        ),
        "verify_boundary_async_drain_wait_per_call_ms": (
            float(boundary_async_drain_wait_ms / verify_calls) if verify_calls else 0.0
        ),
        "verify_metadata_enqueue_per_call_ms": (
            float(metadata_enqueue_ms / verify_calls) if verify_calls else 0.0
        ),
        "verify_metadata_wait_per_call_ms": (
            float(metadata_wait_ms / verify_calls) if verify_calls else 0.0
        ),
        "verify_metadata_collect_per_call_ms": (
            float(metadata_collect_ms / verify_calls) if verify_calls else 0.0
        ),
        "verify_metadata_observe_per_call_ms": (
            float(metadata_observe_ms / verify_calls) if verify_calls else 0.0
        ),
        "verify_metadata_observe_call_per_call_ms": (
            float(metadata_observe_call_ms / verify_calls) if verify_calls else 0.0
        ),
        "verify_metadata_record_consumed_per_call_ms": (
            float(metadata_record_consumed_ms / verify_calls) if verify_calls else 0.0
        ),
        "verify_metadata_mark_access_per_call_ms": (
            float(metadata_mark_access_ms / verify_calls) if verify_calls else 0.0
        ),
        "verify_metadata_queue_update_per_call_ms": (
            float(metadata_queue_update_ms / verify_calls) if verify_calls else 0.0
        ),
        "verify_metadata_observe_verify_rank_guard_per_call_ms": (
            float(metadata_observe_verify_rank_guard_ms / verify_calls) if verify_calls else 0.0
        ),
        "verify_metadata_observe_verify_segment_index_per_call_ms": (
            float(metadata_observe_verify_segment_index_ms / verify_calls) if verify_calls else 0.0
        ),
        "verify_metadata_segment_index_rank_cache_rebuild_per_call_ms": (
            float(metadata_segment_index_rank_cache_rebuild_ms / verify_calls) if verify_calls else 0.0
        ),
        "verify_metadata_segment_index_rank_cache_rebuild_per_call": (
            float(metadata_segment_index_rank_cache_rebuild_count / verify_calls) if verify_calls else 0.0
        ),
        "verify_metadata_observe_verify_runtime_meta_call_per_call_ms": (
            float(metadata_observe_verify_runtime_meta_call_ms / verify_calls) if verify_calls else 0.0
        ),
        "verify_metadata_profile_async_loop_per_call_ms": (
            float(metadata_profile_async_loop_ms / verify_calls) if verify_calls else 0.0
        ),
        "verify_metadata_sync_status_cpu_per_call_ms": (
            float(metadata_sync_status_cpu_ms / verify_calls) if verify_calls else 0.0
        ),
        "verify_metadata_sync_activation_cpu_per_call_ms": (
            float(metadata_sync_activation_cpu_ms / verify_calls) if verify_calls else 0.0
        ),
        "verify_metadata_sync_profile_loop_per_call_ms": (
            float(metadata_sync_profile_loop_ms / verify_calls) if verify_calls else 0.0
        ),
        "verify_segment_submitted_bytes": verify_segment_bytes,
        "verify_segment_submitted_mb_per_call": (
            float(verify_segment_bytes / verify_calls / 1_000_000.0) if verify_calls else 0.0
        ),
        "prefetch_submit_count": int(prefetch.get("submit_count", 0) or 0),
        "prefetch_completed_count": int(prefetch.get("completed_count", 0) or 0),
        "prefetch_late_count": _ep_int(engine_profile, "prefetch_late_count"),
        "outputs_digest": str(raw.get("outputs_digest", "")),
    }


def _command(
    args: argparse.Namespace,
    repo_root: Path,
    prompt_file: Path,
    case: dict[str, Any],
    output_path: Path,
    case_index: int,
    verify_bucket_steps: str,
) -> list[str]:
    single_case_script = (
        repo_root / "benchmarks" / "scripts" / "spec_verify_expert_count_stats.py"
    )
    segment_size = int(case["segment_size"])
    verify_prefetch_enabled = bool(case["verify_prefetch_enabled"])
    verify_boundary_visible_ms = (
        float(args.verify_prefetch_visible_budget_ms) if verify_prefetch_enabled else 0.0
    )
    verify_boundary_max = (
        int(args.verify_prefetch_max_per_boundary) if verify_prefetch_enabled else 0
    )
    verify_layer_budget = (
        int(args.prefetch_verify_layer_max_budget) if verify_prefetch_enabled else 0
    )

    return [
        sys.executable,
        str(single_case_script),
        "--single-case",
        "--model-path",
        args.model_path,
        "--prompt-text-file",
        str(prompt_file),
        "--output",
        str(output_path),
        "--dist-port",
        str(args.dist_port_base + case_index),
        "--cache-ratio",
        str(case["cache_ratio"]),
        "--slots-per-layer",
        "0",
        "--slot-allocation",
        args.allocation_mode,
        "--slot-buckets",
        str(args.slot_buckets),
        "--slot-max-bucket-ratio",
        str(args.slot_max_bucket_ratio),
        "--slot-profile-csv",
        args.slot_profile_csv,
        "--num-seqs",
        "1",
        "--input-len",
        "1",
        "--output-len",
        str(case["output_len"]),
        "--max-draft-tokens",
        str(case["max_draft_tokens"]),
        "--draft-top-c",
        "0",
        "--draft-reroute-policy",
        args.draft_reroute_policy,
        "--draft-reroute-artifact",
        args.profile_artifact,
        "--temperature",
        str(args.temperature),
        "--acceptance-strategy",
        args.acceptance_strategy,
        "--acceptance-threshold",
        str(args.acceptance_threshold),
        "--acceptance-predictor-enabled",
        str(args.acceptance_predictor_enabled).lower(),
        "--acceptance-predictor-path",
        args.acceptance_predictor_path,
        "--acceptance-predictor-step-horizon",
        str(args.acceptance_predictor_step_horizon),
        "--draft-alpha-stop-threshold",
        str(args.draft_alpha_stop_threshold),
        "--draft-stop-policy",
        args.draft_stop_policy,
        "--draft-tpot-td-ms",
        str(args.draft_tpot_td_ms),
        "--draft-tpot-tv-ms",
        str(args.draft_tpot_tv_ms),
        "--prefetch-enabled",
        "true",
        "--prefetch-runtime-mode",
        "draft_segment_indexed",
        "--prefetch-runtime-kind",
        "predictive",
        "--dual-queue-segment-size",
        str(segment_size),
        "--prefetch-strategy",
        "history_window",
        "--prefetch-staging-slots-per-layer",
        str(args.prefetch_staging_slots_per_layer),
        "--prefetch-max-inflight",
        str(args.prefetch_max_inflight),
        "--prefetch-transfer-stream-count",
        str(args.prefetch_transfer_stream_count),
        "--prefetch-metadata-host-buffer-pool-size",
        str(args.prefetch_metadata_host_buffer_pool_size),
        "--prefetch-step-budget",
        str(args.prefetch_step_budget),
        "--prefetch-verify-layer-max-budget",
        str(verify_layer_budget),
        "--prefetch-verify-wait-ms",
        "0",
        "--prefetch-verify-attention-ratio",
        str(args.prefetch_verify_attention_ratio),
        "--cache-eviction-budget-per-step",
        str(args.cache_eviction_budget_per_step),
        "--prefetch-global-queue-capacity",
        str(args.prefetch_global_queue_capacity),
        "--draft-cuda-graph-enabled",
        "true",
        "--draft-cuda-graph-cpu-backend",
        "none",
        "--draft-prefetch-segment-size",
        str(segment_size),
        "--draft-prefetch-segment-host-buffer-pool-size",
        str(args.draft_segment_host_buffer_pool_size),
        "--draft-prefetch-visible-budget-ms",
        str(args.draft_prefetch_visible_budget_ms),
        "--draft-prefetch-min-per-boundary",
        "0",
        "--draft-prefetch-max-per-boundary",
        str(args.draft_prefetch_max_per_boundary),
        "--verify-cuda-graph",
        "true",
        "--verify-cuda-graph-bucket-steps",
        verify_bucket_steps,
        "--verify-prefetch-segment-size",
        str(segment_size),
        "--verify-prefetch-visible-budget-ms",
        str(verify_boundary_visible_ms),
        "--verify-prefetch-min-per-boundary",
        "0",
        "--verify-prefetch-max-per-boundary",
        str(verify_boundary_max),
        "--spec-verify-miss-policy",
        "cpu",
        "--cache-strategy",
        args.cache_strategy,
        "--cpu-expert-execution-enabled",
        "true",
        "--cpu-expert-backend",
        "kt_direct",
        "--cpu-expert-pin-memory",
        "true",
        "--cpu-expert-workspace-max-routes",
        str(args.cpu_expert_workspace_max_routes),
        "--cpu-expert-packed-min-routes",
        "1",
        "--cpu-expert-parallel-mode",
        "serial",
        "--cpu-expert-num-threads",
        str(args.cpu_expert_num_threads),
        "--kt-num-threads",
        str(args.kt_num_threads),
        "--kt-threadpool-count",
        str(args.kt_threadpool_count),
        "--kt-chunked-prefill-size",
        str(args.kt_chunked_prefill_size),
        "--kt-direct-backend",
        args.kt_direct_backend,
        "--kt-numa-nodes",
        args.kt_numa_nodes,
        "--kt-capture-bs",
        args.kt_capture_bs,
        "--cpu-gpu-parallel-execution-enabled",
        "auto",
        "--cpu-gpu-parallel-min-cpu-route-ratio",
        "0.0",
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        "1",
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--enforce-eager",
        "false",
        "--seed",
        str(args.seed),
        "--sync-layer-timing",
        str(args.sync_layer_timing).lower(),
        "--verify-max-cpu-routes-per-layer",
        str(args.verify_max_cpu_routes_per_layer),
    ]


def run_case(
    args: argparse.Namespace,
    repo_root: Path,
    prompt_file: Path,
    case: dict[str, Any],
    case_index: int,
    verify_bucket_steps: str,
) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    name = _case_name(case)
    case_json = output_dir / f"{name}.json"
    case_log = output_dir / f"{name}.log"

    if args.skip_existing and case_json.exists():
        raw = json.loads(case_json.read_text(encoding="utf-8"))
        return _row_from_raw(case, raw, float(raw.get("elapsed_sec", 0.0) or 0.0))

    cmd = _command(args, repo_root, prompt_file, case, case_json, case_index, verify_bucket_steps)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    print(f"[{case_index + 1}] running {name}", flush=True)
    started = time.time()
    with case_log.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            cmd,
            cwd=repo_root,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.case_timeout_sec,
        )
    wall_elapsed = time.time() - started
    print(
        f"[{case_index + 1}] {name} exit={process.returncode} elapsed={wall_elapsed:.1f}s",
        flush=True,
    )
    if process.returncode != 0:
        tail = case_log.read_text(encoding="utf-8", errors="replace")[-5000:]
        raise RuntimeError(f"case failed: {name}\n{tail}")

    raw = json.loads(case_json.read_text(encoding="utf-8"))
    row = _row_from_raw(case, raw, wall_elapsed)
    print(
        f"  mode={row['mode']} K={row['max_draft_tokens']} "
        f"tok/s={row['throughput_output_tok_s']:.3f} "
        f"draft_ms={row['draft_forward_ms_avg']:.3f} "
        f"verify_ms={row['verify_forward_ms_avg']:.3f} "
        f"graph={row['verify_graph_coverage']:.3f} "
        f"hit={row['route_hit_rate']:.4f} accept={row['acceptance_rate']:.4f}",
        flush=True,
    )
    print(
        f"  verify work: tokens/call={row['verify_tokens_per_call']:.2f} "
        f"cpu_routes/call={row['verify_cpu_routes_per_call']:.1f} "
        f"cpu_routes/L/call={row['verify_cpu_routes_per_layer_call']:.2f} "
        f"cpu_exp/call={row['verify_realized_cpu_expert_count_per_call']:.1f} "
        f"cpu_compute/call={row['verify_cpu_compute_ms_per_call']:.3f}ms "
        f"cpu_src={row['verify_cpu_compute_source']} "
        f"cpu_compute/route={row['verify_cpu_compute_ms_per_route']:.5f}ms "
        f"seg_event/call={row['verify_segment_cuda_event_ms_per_call']:.3f}ms "
        f"miss/call={row['verify_pre_transfer_miss_per_call']:.1f}",
        flush=True,
    )
    top_layers = ", ".join(
        f"L{int(item['layer'])}:{float(item['cpu_experts_per_call']):.1f}exp/"
        f"{float(item['cpu_routes_per_call']):.1f}routes"
        for item in row["verify_layer_cpu_expert_top"][:6]
        if float(item["cpu_experts_per_call"]) > 0.0
    )
    print(f"  verify per-layer cpu_exp top: {top_layers or 'none'}", flush=True)
    print(
        f"  verify boundary: submit/call={row['verify_segment_prefetch_submit_per_call']:.1f} "
        f"visible/call={row['verify_segment_prefetch_visible_overhead_per_call_ms']:.3f}ms "
        f"rank/call={row['verify_segment_prefetch_rank_per_call_ms']:.3f}ms "
        f"rank_limit/submit={row['verify_segment_prefetch_rank_limit_per_submit']:.1f} "
        f"cand/submit={row['verify_segment_prefetch_candidate_ranked_per_submit']:.1f} "
        f"async_submit/call={row['verify_boundary_async_worker_submit_per_call_ms']:.3f}ms "
        f"async_drain/call={row['verify_boundary_async_drain_wait_per_call_ms']:.3f}ms "
        f"MB/call={row['verify_segment_submitted_mb_per_call']:.1f}",
        flush=True,
    )
    print(
        f"  verify metadata: enqueue/call={row['verify_metadata_enqueue_per_call_ms']:.3f}ms "
        f"wait/call={row['verify_metadata_wait_per_call_ms']:.3f}ms "
        f"collect/call={row['verify_metadata_collect_per_call_ms']:.3f}ms "
        f"observe/call={row['verify_metadata_observe_per_call_ms']:.3f}ms "
        f"record_consumed/call={row['verify_metadata_record_consumed_per_call_ms']:.3f}ms "
        f"rank_cache_rebuild/call={row['verify_metadata_segment_index_rank_cache_rebuild_per_call_ms']:.3f}ms "
        f"profile_async_loop/call={row['verify_metadata_profile_async_loop_per_call_ms']:.3f}ms "
        f"sync_loop/call={row['verify_metadata_sync_profile_loop_per_call_ms']:.3f}ms",
        flush=True,
    )
    return row


def _comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refs: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if row["mode"] != "verify_prefetch_on":
            continue
        key = (
            row["output_len"],
            row["cache_ratio"],
            row["segment_size"],
            row["repeat"],
        )
        current = refs.get(key)
        if current is None or row["max_draft_tokens"] > current["max_draft_tokens"]:
            refs[key] = row

    comparisons: list[dict[str, Any]] = []
    for row in rows:
        if row["mode"] != "verify_prefetch_off":
            continue
        key = (
            row["output_len"],
            row["cache_ratio"],
            row["segment_size"],
            row["repeat"],
        )
        ref = refs.get(key)
        if ref is None:
            continue
        ref_work = float(ref["verify_cpu_routes_per_call"])
        row_work = float(row["verify_cpu_routes_per_call"])
        work_delta_pct = (
            100.0 * (row_work - ref_work) / ref_work if ref_work > 0.0 else 0.0
        )
        verify_delta_ms = float(row["verify_forward_ms_avg"] - ref["verify_forward_ms_avg"])
        comparisons.append(
            {
                "output_len": row["output_len"],
                "cache_ratio": row["cache_ratio"],
                "segment_size": row["segment_size"],
                "repeat": row["repeat"],
                "prefetch_on_k": ref["max_draft_tokens"],
                "verify_off_k": row["max_draft_tokens"],
                "prefetch_on_verify_ms": ref["verify_forward_ms_avg"],
                "verify_off_verify_ms": row["verify_forward_ms_avg"],
                "verify_delta_ms": verify_delta_ms,
                "verify_delta_pct": (
                    100.0 * verify_delta_ms / ref["verify_forward_ms_avg"]
                    if ref["verify_forward_ms_avg"] > 0.0
                    else 0.0
                ),
                "prefetch_on_visible_boundary_ms_per_call": ref[
                    "verify_segment_prefetch_visible_overhead_per_call_ms"
                ],
                "prefetch_on_rank_ms_per_call": ref[
                    "verify_segment_prefetch_rank_per_call_ms"
                ],
                "prefetch_on_boundary_submit_per_call": ref[
                    "verify_segment_prefetch_submit_per_call"
                ],
                "prefetch_on_boundary_mb_per_call": ref[
                    "verify_segment_submitted_mb_per_call"
                ],
                "prefetch_on_cpu_routes_per_call": ref["verify_cpu_routes_per_call"],
                "verify_off_cpu_routes_per_call": row["verify_cpu_routes_per_call"],
                "cpu_work_delta_pct": work_delta_pct,
                "prefetch_on_tokens_per_call": ref["verify_tokens_per_call"],
                "verify_off_tokens_per_call": row["verify_tokens_per_call"],
                "prefetch_on_hit_rate": ref["route_hit_rate"],
                "verify_off_hit_rate": row["route_hit_rate"],
                "prefetch_on_acceptance": ref["acceptance_rate"],
                "verify_off_acceptance": row["acceptance_rate"],
                "prefetch_on_graph_coverage": ref["verify_graph_coverage"],
                "verify_off_graph_coverage": row["verify_graph_coverage"],
            }
        )
    comparisons.sort(
        key=lambda item: (
            item["output_len"],
            item["cache_ratio"],
            item["segment_size"],
            item["repeat"],
            abs(item["cpu_work_delta_pct"]),
            item["verify_off_k"],
        )
    )
    return comparisons


def write_markdown_report(summary: dict[str, Any], path: Path) -> None:
    metadata = summary["metadata"]
    rows = summary["rows"]
    comparisons = summary["comparisons"]
    lines = [
        "# Verify Boundary Prefetch Overhead",
        "",
        f"- timestamp: `{metadata['timestamp']}`",
        f"- model: `{metadata['model_path']}`",
        f"- allocation: `{metadata['allocation_mode']}`",
        f"- verify graph buckets: `{metadata['verify_cuda_graph_bucket_steps']}`",
        f"- output directory: `{metadata['output_dir']}`",
        "",
        "## Cases",
        "",
        "| mode | seg | out | ratio | K | rep | tok/s | draft ms | verify ms | graph | hit | accept | vtok/call | cpu routes/call | cpu exp/call | cpu ms/call | cpu ms/route | seg event/call | verify submit/call | boundary ms/call | rank ms/call | MB/call |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['mode']} | {row['segment_size']} | {row['output_len']} | "
            f"{row['cache_ratio']:.4f} | {row['max_draft_tokens']} | {row['repeat']} | "
            f"{row['throughput_output_tok_s']:.3f} | "
            f"{row['draft_forward_ms_avg']:.3f} | "
            f"{row['verify_forward_ms_avg']:.3f} | "
            f"{row['verify_graph_coverage']:.3f} | "
            f"{row['route_hit_rate']:.4f} | {row['acceptance_rate']:.4f} | "
            f"{row['verify_tokens_per_call']:.2f} | "
            f"{row['verify_cpu_routes_per_call']:.1f} | "
            f"{row['verify_realized_cpu_expert_count_per_call']:.1f} | "
            f"{row['verify_cpu_compute_ms_per_call']:.3f} | "
            f"{row['verify_cpu_compute_ms_per_route']:.5f} | "
            f"{row['verify_segment_cuda_event_ms_per_call']:.3f} | "
            f"{row['verify_segment_prefetch_submit_per_call']:.1f} | "
            f"{row['verify_segment_prefetch_visible_overhead_per_call_ms']:.3f} | "
            f"{row['verify_segment_prefetch_rank_per_call_ms']:.3f} | "
            f"{row['verify_segment_submitted_mb_per_call']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Per-Layer CPU Experts",
            "",
            "Full 48-layer arrays are in `summary.json`; the CSV export is `per_layer_cpu_experts.csv`.",
            "",
            "| case | K | layer | CPU experts/call | CPU routes/call | active experts/call | active routes/call |",
            "|:---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        for item in row["verify_layer_cpu_expert_top"]:
            if float(item["cpu_experts_per_call"]) <= 0.0:
                continue
            lines.append(
                "| "
                f"{row['name']} | {row['max_draft_tokens']} | "
                f"{int(item['layer'])} | "
                f"{float(item['cpu_experts_per_call']):.3f} | "
                f"{float(item['cpu_routes_per_call']):.3f} | "
                f"{float(item['active_experts_per_call']):.3f} | "
                f"{float(item['active_routes_per_call']):.3f} |"
            )

    lines.extend(
        [
            "",
            "## Verify Boundary Breakdown",
            "",
            "| mode | K | visible/call | rank scan | rank sort | rank limit/submit | candidates/submit | merged/submit | filter | victim | reserve | transfer enqueue | bookkeeping | async enqueue | async worker submit | async queue wait | async drain |",
            "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"{row['mode']} | {row['max_draft_tokens']} | "
            f"{row['verify_segment_prefetch_visible_overhead_per_call_ms']:.3f} | "
            f"{row['verify_segment_prefetch_rank_scan_per_call_ms']:.3f} | "
            f"{row['verify_segment_prefetch_rank_sort_per_call_ms']:.3f} | "
            f"{row['verify_segment_prefetch_rank_limit_per_submit']:.1f} | "
            f"{row['verify_segment_prefetch_candidate_ranked_per_submit']:.1f} | "
            f"{row['verify_segment_prefetch_candidate_merge_per_submit']:.1f} | "
            f"{row['verify_segment_prefetch_filter_per_call_ms']:.3f} | "
            f"{row['verify_segment_prefetch_victim_select_per_call_ms']:.3f} | "
            f"{row['verify_segment_prefetch_reservation_per_call_ms']:.3f} | "
            f"{row['verify_segment_prefetch_transfer_enqueue_per_call_ms']:.3f} | "
            f"{row['verify_segment_prefetch_bookkeeping_per_call_ms']:.3f} | "
            f"{row['verify_boundary_async_enqueue_per_call_ms']:.3f} | "
            f"{row['verify_boundary_async_worker_submit_per_call_ms']:.3f} | "
            f"{row['verify_boundary_async_worker_queue_wait_per_call_ms']:.3f} | "
            f"{row['verify_boundary_async_drain_wait_per_call_ms']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Verify Metadata Breakdown",
            "",
            "| mode | K | enqueue | readback wait | collect | observe | observe call | record consumed | mark access | queue update | verify segment index | rank cache rebuild | verify runtime meta | async profile loop | sync profile loop |",
            "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"{row['mode']} | {row['max_draft_tokens']} | "
            f"{row['verify_metadata_enqueue_per_call_ms']:.3f} | "
            f"{row['verify_metadata_wait_per_call_ms']:.3f} | "
            f"{row['verify_metadata_collect_per_call_ms']:.3f} | "
            f"{row['verify_metadata_observe_per_call_ms']:.3f} | "
            f"{row['verify_metadata_observe_call_per_call_ms']:.3f} | "
            f"{row['verify_metadata_record_consumed_per_call_ms']:.3f} | "
            f"{row['verify_metadata_mark_access_per_call_ms']:.3f} | "
            f"{row['verify_metadata_queue_update_per_call_ms']:.3f} | "
            f"{row['verify_metadata_observe_verify_segment_index_per_call_ms']:.3f} | "
            f"{row['verify_metadata_segment_index_rank_cache_rebuild_per_call_ms']:.3f} | "
            f"{row['verify_metadata_observe_verify_runtime_meta_call_per_call_ms']:.3f} | "
            f"{row['verify_metadata_profile_async_loop_per_call_ms']:.3f} | "
            f"{row['verify_metadata_sync_profile_loop_per_call_ms']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Prefetch-On vs Verify-Prefetch-Off",
            "",
            "| out | ratio | seg | rep | on K | off K | on verify ms | off verify ms | delta ms | delta % | on boundary ms/call | on rank ms/call | on submit/call | on MB/call | CPU work delta % | on routes/call | off routes/call | on graph | off graph |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            "| "
            f"{row['output_len']} | {row['cache_ratio']:.4f} | "
            f"{row['segment_size']} | {row['repeat']} | "
            f"{row['prefetch_on_k']} | {row['verify_off_k']} | "
            f"{row['prefetch_on_verify_ms']:.3f} | "
            f"{row['verify_off_verify_ms']:.3f} | "
            f"{row['verify_delta_ms']:+.3f} | "
            f"{row['verify_delta_pct']:+.2f} | "
            f"{row['prefetch_on_visible_boundary_ms_per_call']:.3f} | "
            f"{row['prefetch_on_rank_ms_per_call']:.3f} | "
            f"{row['prefetch_on_boundary_submit_per_call']:.1f} | "
            f"{row['prefetch_on_boundary_mb_per_call']:.1f} | "
            f"{row['cpu_work_delta_pct']:+.2f} | "
            f"{row['prefetch_on_cpu_routes_per_call']:.1f} | "
            f"{row['verify_off_cpu_routes_per_call']:.1f} | "
            f"{row['prefetch_on_graph_coverage']:.3f} | "
            f"{row['verify_off_graph_coverage']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## How To Read This",
            "",
            "- `verify_prefetch_off` does not disable the whole prefetch runtime. It only sets verify segment boundary submission to zero.",
            "- Use the off-K row with the smallest `CPU work delta %` as the cleanest comparison against the prefetch-on reference.",
            "- If `verify ms` drops while `CPU work delta %` is small and graph coverage stays near 1.0, the removed latency is attributable to verify boundary prefetch scheduling/copy/sync rather than MoE CPU compute.",
            "- `boundary ms/call` is the measured Python-visible submit/rank overhead, not the full CUDA stream synchronization cost. The latter is visible in torch profiler traces.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_per_layer_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "case",
        "mode",
        "output_len",
        "cache_ratio",
        "segment_size",
        "max_draft_tokens",
        "repeat",
        "layer",
        "cpu_experts_total",
        "cpu_experts_per_call",
        "cpu_routes_total",
        "cpu_routes_per_call",
        "active_experts_total",
        "active_experts_per_call",
        "active_routes_total",
        "active_routes_per_call",
        "layer_profile_count",
        "verify_calls",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for layer_idx in range(NUM_MOE_LAYERS):
                writer.writerow(
                    {
                        "case": row["name"],
                        "mode": row["mode"],
                        "output_len": row["output_len"],
                        "cache_ratio": row["cache_ratio"],
                        "segment_size": row["segment_size"],
                        "max_draft_tokens": row["max_draft_tokens"],
                        "repeat": row["repeat"],
                        "layer": layer_idx,
                        "cpu_experts_total": row[
                            "verify_layer_realized_cpu_expert_count_total"
                        ][layer_idx],
                        "cpu_experts_per_call": row[
                            "verify_layer_realized_cpu_expert_count_per_call"
                        ][layer_idx],
                        "cpu_routes_total": row["verify_layer_cpu_routes_total"][layer_idx],
                        "cpu_routes_per_call": row["verify_layer_cpu_routes_per_call"][layer_idx],
                        "active_experts_total": row[
                            "verify_layer_active_expert_count_total"
                        ][layer_idx],
                        "active_experts_per_call": row[
                            "verify_layer_active_expert_count_per_call"
                        ][layer_idx],
                        "active_routes_total": row["verify_layer_active_routes_total"][layer_idx],
                        "active_routes_per_call": row[
                            "verify_layer_active_routes_per_call"
                        ][layer_idx],
                        "layer_profile_count": row["verify_layer_moe_profile_count"][layer_idx],
                        "verify_calls": row["verify_calls"],
                    }
                )


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = output_dir / "verify_boundary_prompt.txt"
    prompt_file.write_text(PROMPT_TEXT + "\n", encoding="utf-8")

    cases = build_cases(args)
    verify_bucket_steps = _verify_bucket_steps(args, cases)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        try:
            print("=" * 80)
            rows.append(
                run_case(
                    args,
                    repo_root,
                    prompt_file,
                    case,
                    case_index,
                    verify_bucket_steps,
                )
            )
        except Exception as error:
            failures.append({"case": case, "error": str(error)})
            if args.fail_fast:
                raise

    summary = {
        "metadata": {
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "model_path": args.model_path,
            "profile_artifact": args.profile_artifact,
            "slot_profile_csv": args.slot_profile_csv,
            "output_dir": str(output_dir),
            "allocation_mode": args.allocation_mode,
            "slot_buckets": int(args.slot_buckets),
            "slot_max_bucket_ratio": float(args.slot_max_bucket_ratio),
            "segment_sizes": _parse_csv(args.segment_sizes, int),
            "cache_ratios": _parse_csv(args.cache_ratios, float),
            "output_lens": _parse_csv(args.output_lens, int),
            "prefetch_on_max_draft_tokens_values": _parse_csv(
                args.prefetch_on_max_draft_tokens_values, int
            ),
            "verify_off_max_draft_tokens_values": _parse_csv(
                args.verify_off_max_draft_tokens_values, int
            ),
            "verify_cuda_graph_bucket_steps": verify_bucket_steps,
            "verify_max_cpu_routes_per_layer": int(args.verify_max_cpu_routes_per_layer),
            "repeats": int(args.repeats),
            "per_layer_cpu_experts_csv": str(output_dir / "per_layer_cpu_experts.csv"),
            "argv": sys.argv,
        },
        "rows": rows,
        "comparisons": _comparison_rows(rows),
        "failures": failures,
    }
    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"
    write_per_layer_csv(rows, output_dir / "per_layer_cpu_experts.csv")
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown_report(summary, summary_md)
    if args.report_doc:
        write_markdown_report(summary, Path(args.report_doc))

    print(f"summary_json={summary_json}")
    print(f"summary_md={summary_md}")
    for comp in summary["comparisons"]:
        print(
            f"  offK={comp['verify_off_k']} vs onK={comp['prefetch_on_k']} "
            f"verify_delta={comp['verify_delta_ms']:+.3f}ms "
            f"cpu_work_delta={comp['cpu_work_delta_pct']:+.2f}% "
            f"on_boundary={comp['prefetch_on_visible_boundary_ms_per_call']:.3f}ms/call",
            flush=True,
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare verify latency with verify segment prefetch on vs off."
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--profile-artifact", default=DEFAULT_PROFILE)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-doc", default="")
    parser.add_argument(
        "--modes",
        default="verify_prefetch_on,verify_prefetch_off",
        help="CSV subset of verify_prefetch_on,verify_prefetch_off.",
    )
    parser.add_argument("--allocation-mode", choices=["uniform", "profile_weighted"], default="profile_weighted")
    parser.add_argument("--slot-buckets", type=int, default=4)
    parser.add_argument("--slot-max-bucket-ratio", type=float, default=2.0)
    parser.add_argument("--slot-profile-csv", default=DEFAULT_SLOT_PROFILE_CSV)
    parser.add_argument("--output-lens", default="512")
    parser.add_argument("--cache-ratios", default="0.3125")
    parser.add_argument("--prefetch-on-max-draft-tokens-values", default="12")
    parser.add_argument(
        "--verify-off-max-draft-tokens-values",
        default="8,10,12",
        help="K sweep for verify_prefetch_off. Lower K helps match CPU routes/call.",
    )
    parser.add_argument("--segment-sizes", default="12")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--cache-strategy", default="lru")
    parser.add_argument("--draft-reroute-policy", default="entropy_cache_bias")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--acceptance-strategy", default="standard_sampling")
    parser.add_argument("--acceptance-threshold", type=float, default=0.7)
    parser.add_argument("--acceptance-predictor-enabled", type=str2bool, default=True)
    parser.add_argument("--acceptance-predictor-path", default=DEFAULT_PREDICTOR_PATH)
    parser.add_argument("--acceptance-predictor-step-horizon", type=int, default=32)
    parser.add_argument("--draft-alpha-stop-threshold", type=float, default=0.89)
    parser.add_argument(
        "--draft-stop-policy",
        choices=["none", "alpha_threshold", "tpot"],
        default="tpot",
    )
    parser.add_argument("--draft-tpot-td-ms", type=float, default=19.0)
    parser.add_argument("--draft-tpot-tv-ms", type=float, default=80.0)

    parser.add_argument("--prefetch-step-budget", type=int, default=16)
    parser.add_argument("--prefetch-max-inflight", type=int, default=16)
    parser.add_argument("--prefetch-transfer-stream-count", type=int, default=1)
    parser.add_argument("--prefetch-staging-slots-per-layer", type=int, default=2)
    parser.add_argument("--prefetch-metadata-host-buffer-pool-size", type=int, default=3)
    parser.add_argument("--prefetch-global-queue-capacity", type=int, default=4096)
    parser.add_argument("--prefetch-verify-layer-max-budget", type=int, default=8)
    parser.add_argument("--prefetch-verify-attention-ratio", type=float, default=1.0)
    parser.add_argument("--cache-eviction-budget-per-step", type=int, default=2)
    parser.add_argument("--draft-segment-host-buffer-pool-size", type=int, default=0)
    parser.add_argument("--draft-prefetch-visible-budget-ms", type=float, default=3.0)
    parser.add_argument("--draft-prefetch-max-per-boundary", type=int, default=16)
    parser.add_argument("--verify-prefetch-visible-budget-ms", type=float, default=12.0)
    parser.add_argument("--verify-prefetch-max-per-boundary", type=int, default=4)

    parser.add_argument("--cpu-expert-workspace-max-routes", type=int, default=327680)
    parser.add_argument("--cpu-expert-num-threads", type=int, default=4)
    parser.add_argument("--kt-num-threads", type=int, default=16)
    parser.add_argument("--kt-threadpool-count", type=int, default=1)
    parser.add_argument("--kt-chunked-prefill-size", type=int, default=4096)
    parser.add_argument(
        "--kt-direct-backend",
        choices=["auto", "amx_bf16", "avx2_bf16"],
        default="auto",
    )
    parser.add_argument("--kt-numa-nodes", default="")
    parser.add_argument("--kt-capture-bs", default="1,2,4,8,16,32")
    parser.add_argument(
        "--verify-max-cpu-routes-per-layer",
        type=int,
        default=0,
        help="Optional hard cap for verify CPU routes per layer. 0 disables capping.",
    )

    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.99)
    parser.add_argument("--verify-cuda-graph-bucket-steps", default="3,5,8,12")
    parser.add_argument("--auto-add-verify-k-plus-one", type=str2bool, default=True)
    parser.add_argument("--dist-port-base", type=int, default=30900)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sync-layer-timing", type=str2bool, default=False)
    parser.add_argument("--case-timeout-sec", type=int, default=2400)
    parser.add_argument("--skip-existing", type=str2bool, default=True)
    parser.add_argument("--fail-fast", type=str2bool, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    print("running with arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    run(args)


if __name__ == "__main__":
    main()
