"""Runtime configuration boundary for the evaluation TPOT benchmark.

This module is intentionally independent from argparse.  The legacy CLI may
still parse into a Namespace, but all values crossing into ``LLM`` are
normalized here.  Keeping one kwargs builder prevents benchmark entry points
from silently drifting apart as :class:`nanovllm.config.Config` evolves.
"""
from __future__ import annotations

import argparse
import random
from dataclasses import fields
from typing import Any, Callable, TypeVar

from nanovllm.config import Config


T = TypeVar("T")


def parse_csv(values: str, cast: Callable[[str], T]) -> list[T]:
    parsed = [cast(item.strip()) for item in values.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one value")
    return parsed


def resolved_kt_capture_bs(args: argparse.Namespace) -> list[int]:
    """Return persistent KT buffer sizes, including every verify graph bucket."""
    return sorted(
        set(parse_csv(args.kt_capture_bs, int))
        | set(parse_csv(args.verify_cuda_graph_bucket_steps, int))
    )


def runtime_seed(args: argparse.Namespace, case: dict[str, Any], sample_index: int = 0) -> int:
    return int(args.seed) + int(case.get("repeat", 0)) + int(sample_index)


def reset_runtime_seed(seed: int) -> None:
    random.seed(int(seed))
    try:
        import torch
    except Exception:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolved_runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    """Return the effective engine topology persisted in benchmark metadata."""
    mode = str(args.inference_mode)
    prefetch_enabled = bool(args.spec_enable_prefetch) and mode == "spec"
    runtime_class = None
    if prefetch_enabled:
        runtime_class = {
            "legacy": "PrefetchRuntime",
            "predictive": "PredictivePrefetchRuntime",
            "dual_queue": "DualQueuePrefetchRuntime",
        }[str(args.prefetch_runtime_kind)]
    return {
        "inference_mode": mode,
        "enable_heterogeneous": mode in {"heter", "spec"},
        "enable_speculative": mode == "spec",
        "batch_size": int(args.batch_size),
        "enforce_eager": bool(args.enforce_eager),
        "spec_enable_prefetch": bool(args.spec_enable_prefetch),
        "prefetch_runtime_mode": str(args.prefetch_runtime_mode),
        "prefetch_runtime_kind": str(args.prefetch_runtime_kind),
        "prefetch_runtime_class": runtime_class,
        "draft_cuda_graph_enabled": bool(args.draft_cuda_graph_enabled),
        "verify_cuda_graph": bool(args.verify_cuda_graph),
        "draft_reroute_policy": str(args.draft_reroute_policy),
        "cache_strategy": str(args.cache_strategy),
        "cpu_expert_pin_memory": bool(args.cpu_expert_pin_memory),
        "kt_num_threads": int(args.kt_num_threads),
        "kt_threadpool_count": int(args.kt_threadpool_count),
        "kt_direct_backend": str(args.kt_direct_backend),
        "kt_llamafile_extension_path": str(args.kt_llamafile_extension_path),
        "kt_single_weight": bool(args.kt_single_weight),
        "kt_numa_nodes": (
            parse_csv(args.kt_numa_nodes, int) if args.kt_numa_nodes else []
        ),
        "kt_capture_bs": resolved_kt_capture_bs(args),
    }


def build_llm_kwargs(
    args: argparse.Namespace,
    case: dict[str, Any],
    case_index: int,
    *,
    num_experts: int,
) -> dict[str, Any]:
    """Normalize CLI/case values into the single supported LLM config map."""
    slots = int(args.slots_per_layer)
    if slots <= 0:
        slots = int(round(int(num_experts) * float(case["cache_ratio"])))

    segment_size = int(case["segment_size"])
    mode = str(args.inference_mode)
    kwargs: dict[str, Any] = {
        "dist_port": int(args.dist_port_base) + int(case_index),
        "enforce_eager": bool(args.enforce_eager),
        "max_num_batched_tokens": int(args.max_num_batched_tokens),
        "max_num_seqs": int(args.batch_size),
        "max_model_len": int(args.max_model_len),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "inference_mode": mode,
        "enable_heterogeneous": mode in {"heter", "spec"},
        "enable_speculative": mode == "spec",
        "heterogeneous_slots_per_layer": slots,
        "heterogeneous_slot_allocation": str(case["allocation_mode"]),
        "heterogeneous_slot_buckets": int(args.slot_buckets),
        "heterogeneous_slot_max_bucket_ratio": float(args.slot_max_bucket_ratio),
        "heterogeneous_slot_profile_csv": str(args.slot_profile_csv),
        "max_draft_tokens": int(case["max_draft_tokens"]),
        "draft_top_c": 0,
        "draft_reroute_policy": str(args.draft_reroute_policy),
        "draft_reroute_artifact": str(args.profile_artifact),
        "acceptance_strategy": str(args.acceptance_strategy),
        "acceptance_threshold": float(args.acceptance_threshold),
        "acceptance_predictor_enabled": bool(args.acceptance_predictor_enabled),
        "acceptance_predictor_path": str(args.acceptance_predictor_path),
        "acceptance_predictor_step_horizon": int(args.acceptance_predictor_step_horizon),
        "draft_alpha_stop_threshold": float(args.draft_alpha_stop_threshold),
        "draft_stop_policy": str(args.draft_stop_policy),
        "draft_tpot_td_ms": float(args.draft_tpot_td_ms),
        "draft_tpot_tv_ms": float(args.draft_tpot_tv_ms),
        "draft_tpot_cost_model": str(args.draft_tpot_cost_model),
        "draft_tpot_history_alpha": float(args.draft_tpot_history_alpha),
        "draft_tpot_min_steps": int(args.draft_tpot_min_steps),
        "draft_tpot_stop_margin": float(args.draft_tpot_stop_margin),
        "draft_tpot_stop_patience": int(args.draft_tpot_stop_patience),
        "draft_tpot_lookahead_cache_credit_ms_per_step": float(
            args.draft_tpot_lookahead_cache_credit_ms_per_step
        ),
        "draft_tpot_short_verify_penalty_ms": float(args.draft_tpot_short_verify_penalty_ms),
        "draft_tpot_verify_cost_floor_ms": float(args.draft_tpot_verify_cost_floor_ms),
        "draft_tpot_alpha_error_p90": float(args.draft_tpot_alpha_error_p90),
        "draft_tpot_draft_error_p90_ms": float(args.draft_tpot_draft_error_p90_ms),
        "draft_tpot_uncertainty_scale": float(args.draft_tpot_uncertainty_scale),
        "draft_tpot_stop_rule": str(args.draft_tpot_stop_rule),
        "draft_tpot_verify_model_mode": str(args.draft_tpot_verify_model_mode),
        "draft_tpot_verify_model_path": str(args.draft_tpot_verify_model_path),
        "draft_tpot_alpha_calibration_path": str(args.draft_tpot_alpha_calibration_path),
        "transfer_aware_profile": bool(args.transfer_aware_profile),
        "cpu_expert_execution_enabled": True,
        "cpu_expert_pin_memory": bool(args.cpu_expert_pin_memory),
        "cpu_expert_backend": "kt_direct",
        "cpu_expert_workspace_max_routes": int(args.cpu_expert_workspace_max_routes),
        "cpu_expert_packed_min_routes": 1,
        "cpu_expert_parallel_mode": "serial",
        "cpu_expert_num_threads": int(args.cpu_expert_num_threads),
        "kt_num_threads": int(args.kt_num_threads),
        "kt_threadpool_count": int(args.kt_threadpool_count),
        "kt_chunked_prefill_size": int(args.kt_chunked_prefill_size),
        "kt_direct_backend": str(args.kt_direct_backend),
        "kt_llamafile_extension_path": str(args.kt_llamafile_extension_path),
        "kt_single_weight": bool(args.kt_single_weight),
        "kt_numa_nodes": parse_csv(args.kt_numa_nodes, int) if args.kt_numa_nodes else [],
        "kt_capture_bs": resolved_kt_capture_bs(args),
        "cpu_gpu_parallel_execution_enabled": "auto",
        "cpu_gpu_parallel_min_cpu_route_ratio": 0.0,
        "spec_verify_miss_policy": "cpu",
        "spec_profile": False,
        "engine_profile": bool(args.engine_profile),
        "engine_profile_cuda_sync": bool(args.engine_profile_cuda_sync),
        "spec_enable_prefetch": bool(args.spec_enable_prefetch),
        "cache_strategy": str(args.cache_strategy),
        "rank_guard_threshold": float(args.rank_guard_threshold),
        "rank_guard_ema_alpha": float(args.rank_guard_ema_alpha),
        "prefetch_strategy": "history_window",
        "prefetch_runtime_mode": str(args.prefetch_runtime_mode),
        "prefetch_runtime_kind": str(args.prefetch_runtime_kind),
        "dual_queue_segment_size": segment_size,
        "dual_queue_ground_truth_decay": float(args.dual_queue_ground_truth_decay),
        "dual_queue_ground_truth_ttl_rounds": int(args.dual_queue_ground_truth_ttl_rounds),
        "dual_queue_ground_truth_count_weight": float(args.dual_queue_ground_truth_count_weight),
        "dual_queue_budget_safety_ratio": float(args.dual_queue_budget_safety_ratio),
        "dual_queue_segment_time_ema_alpha": float(args.dual_queue_segment_time_ema_alpha),
        "dual_queue_secondary_index_weight": float(args.dual_queue_secondary_index_weight),
        "prefetch_verify_attention_ratio": float(args.prefetch_verify_attention_ratio),
        "predictive_phase1_budget": int(args.predictive_phase1_budget),
        "prefetch_staging_slots_per_layer": int(args.prefetch_staging_slots_per_layer),
        "prefetch_max_inflight": int(args.prefetch_max_inflight),
        "prefetch_transfer_stream_count": int(args.prefetch_transfer_stream_count),
        "prefetch_metadata_host_buffer_pool_size": int(args.prefetch_metadata_host_buffer_pool_size),
        "prefetch_verify_layer_max_budget": int(args.prefetch_verify_layer_max_budget),
        "prefetch_step_budget": int(args.prefetch_step_budget),
        "cache_eviction_budget_per_step": int(args.cache_eviction_budget_per_step),
        "prefetch_verify_wait_ms": 0.0,
        "prefetch_global_queue_capacity": int(args.prefetch_global_queue_capacity),
        "prefetch_history_decay": float(args.prefetch_history_decay),
        "prefetch_history_ttl_steps": int(args.prefetch_history_ttl_steps),
        "prefetch_source_weight_prefill": float(args.prefetch_source_weight_prefill),
        "prefetch_source_weight_verify": float(args.prefetch_source_weight_verify),
        "prefetch_source_weight_draft": float(args.prefetch_source_weight_draft),
        "prefetch_activation_count_weight": float(args.prefetch_activation_count_weight),
        "prefetch_age_penalty": float(args.prefetch_age_penalty),
        "prefetch_use_prefill_history": bool(args.prefetch_use_prefill_history),
        "prefetch_use_verify_history": bool(args.prefetch_use_verify_history),
        "prefetch_use_draft_live": bool(args.prefetch_use_draft_live),
        "draft_cuda_graph_enabled": bool(args.draft_cuda_graph_enabled),
        "draft_cuda_graph_cpu_backend": "none",
        "draft_prefetch_segment_size": segment_size,
        "draft_prefetch_segment_host_buffer_pool_size": int(args.draft_segment_host_buffer_pool_size),
        "draft_prefetch_visible_budget_ms": float(args.draft_prefetch_visible_budget_ms),
        "draft_prefetch_min_per_boundary": 0,
        "draft_prefetch_max_per_boundary": int(args.draft_prefetch_max_per_boundary),
        "verify_cuda_graph": bool(args.verify_cuda_graph),
        "verify_cuda_graph_bucket_steps": parse_csv(args.verify_cuda_graph_bucket_steps, int),
        "verify_prefetch_segment_size": segment_size,
        "verify_prefetch_visible_budget_ms": float(args.verify_prefetch_visible_budget_ms),
        "verify_prefetch_min_per_boundary": 0,
        "verify_prefetch_max_per_boundary": int(args.verify_prefetch_max_per_boundary),
        "verify_prefetch_tpot_dynamic_budget_enabled": bool(
            args.verify_prefetch_tpot_dynamic_budget_enabled
        ),
        "verify_prefetch_tpot_dynamic_budget_token_threshold": int(
            args.verify_prefetch_tpot_dynamic_budget_token_threshold
        ),
        "verify_prefetch_tpot_dynamic_budget_small": int(
            args.verify_prefetch_tpot_dynamic_budget_small
        ),
    }
    config_fields = {field.name for field in fields(Config)}
    unknown = sorted(set(kwargs) - config_fields)
    if unknown:
        raise RuntimeError(
            "benchmark produced unknown LLM Config fields: " + ", ".join(unknown)
        )
    return kwargs


def create_llm(args: argparse.Namespace, case: dict[str, Any], case_index: int) -> Any:
    from nanovllm import LLM
    from transformers import AutoConfig

    reset_runtime_seed(runtime_seed(args, case, 0))
    hf_config = AutoConfig.from_pretrained(args.model_path)
    kwargs = build_llm_kwargs(
        args,
        case,
        case_index,
        num_experts=int(getattr(hf_config, "num_experts")),
    )
    return LLM(args.model_path, **kwargs)


def warmup_llm(llm: Any, *, temperature: float, prompt: str) -> None:
    from nanovllm import SamplingParams

    sampling = SamplingParams(temperature=temperature, ignore_eos=True, max_tokens=4)
    llm.generate([prompt], sampling, use_tqdm=False)


def validate_kv_cache_capacity(
    llm: Any,
    *,
    prompt_tokens: int,
    max_tokens: int,
    batch_size: int = 1,
) -> dict[str, int | float]:
    """Validate persistent KV capacity for a synchronous request batch."""
    config = llm.config
    block_size = int(config.kvcache_block_size)
    available_blocks = int(config.num_kvcache_blocks)
    required_tokens = int(prompt_tokens) + int(max_tokens)
    blocks_per_request = (required_tokens + block_size - 1) // block_size
    required_blocks = int(batch_size) * blocks_per_request
    capacity_tokens = available_blocks * block_size
    result: dict[str, int | float] = {
        "available_blocks": available_blocks,
        "required_blocks": required_blocks,
        "capacity_tokens": capacity_tokens,
        "required_tokens": required_tokens,
        "batch_size": int(batch_size),
        "blocks_per_request": blocks_per_request,
        "block_size": block_size,
    }
    if required_blocks <= available_blocks:
        return result

    current_utilization = float(config.gpu_memory_utilization)
    block_bytes = int(getattr(config, "kvcache_block_bytes", 0) or 0)
    total_bytes = int(getattr(config, "gpu_total_memory_bytes", 0) or 0)
    suggested = None
    if block_bytes > 0 and total_bytes > 0:
        missing_bytes = (required_blocks - available_blocks) * block_bytes
        suggested = current_utilization + missing_bytes / total_bytes + 0.001
        result["suggested_gpu_memory_utilization"] = min(0.999, suggested)

    suggestion = ""
    if suggested is not None and suggested <= 0.999:
        suggestion = f" Try --gpu-memory-utilization {min(0.999, suggested):.3f}."
    raise RuntimeError(
        "Insufficient KV cache capacity for this request batch: "
        f"batch_size={batch_size}, prompt_tokens={prompt_tokens}, max_tokens={max_tokens}, "
        f"required_blocks={required_blocks}, available_blocks={available_blocks}, "
        f"block_size={block_size} (required_tokens={required_tokens}, "
        f"capacity_tokens={capacity_tokens}).{suggestion} "
        "Alternatively reduce --cache-ratios or --output-lens."
    )
