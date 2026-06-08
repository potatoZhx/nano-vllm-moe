#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from random import Random
from statistics import mean
from time import perf_counter
from typing import Any


BASE_PROMPTS = [
    "Summarize why sparse MoE inference can reduce memory traffic compared with dense inference.",
    "Explain top-k expert routing with a short example and one deployment caveat.",
    "Describe a practical GPU expert cache policy for a memory-limited inference server.",
    "Give a compact checklist for validating speculative decoding after a kernel change.",
]


DRAFT_LAYER_EVENTS: list[dict[str, Any]] = []
VERIFY_LAYER_EVENTS: list[dict[str, Any]] = []
SYNC_LAYER_TIMING = True


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def _parse_int_csv(values: str) -> list[int]:
    out = [int(x.strip()) for x in values.split(",") if x.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return out


def _make_prompts(num_seqs: int, input_len: int, seed: int) -> list[str]:
    rng = Random(seed)
    prompts: list[str] = []
    for i in range(num_seqs):
        base = BASE_PROMPTS[i % len(BASE_PROMPTS)]
        words: list[str] = []
        while len(words) < input_len:
            score = rng.randint(0, 100)
            words.extend(
                (
                    f"case {i}",
                    f"score {score}",
                    "cache pressure verify routing latency expert balance",
                )
            )
        context = " ".join(words[:input_len])
        prompts.append(f"Task: {base}\nContext: {context}\nAnswer briefly.")
    return prompts


def _tokenize_prompts_to_length(tokenizer: Any, prompts: list[str], input_len: int) -> list[list[int]]:
    token_prompts: list[list[int]] = []
    for prompt in prompts:
        token_ids = tokenizer.encode(prompt)
        if len(token_ids) < input_len:
            raise ValueError(f"generated prompt has {len(token_ids)} tokens, shorter than requested {input_len}")
        token_prompts.append(token_ids[:input_len])
    return token_prompts


def _install_verify_layer_probe(sync_layer_timing: bool) -> None:
    global SYNC_LAYER_TIMING
    SYNC_LAYER_TIMING = bool(sync_layer_timing)

    import torch
    from nanovllm.models import qwen3_moe

    cls = qwen3_moe.Qwen3MoeHeterogeneousSparseMoeBlock
    if getattr(cls, "_verify_count_stats_patched", False):
        return

    original_forward = cls.forward

    def patched_forward(self, hidden_states):  # type: ignore[no-untyped-def]
        mode = getattr(self, "execution_mode", "normal")
        is_verify = mode == "verify"
        is_draft = mode == "draft"
        token_count = int(hidden_states.shape[0])
        if (is_verify or is_draft) and token_count > 16:
            print(f"[PATCHED_DEBUG] mode={mode} layer={getattr(self, 'layer_idx', -1)} "
                  f"tokens={token_count} top_k={getattr(self, 'num_selected', 0)}",
                  flush=True)
        if is_verify and SYNC_LAYER_TIMING and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = perf_counter()
        out = original_forward(self, hidden_states)
        if is_verify or is_draft:
            if is_verify and SYNC_LAYER_TIMING and torch.cuda.is_available():
                torch.cuda.synchronize()
            prof = dict(getattr(self, "_last_profile", {}) or {})
            cpu_routes = int(round(float(prof.get("cpu_routes_sum", 0.0))))
            cpu_compute_ms = float(prof.get("cpu_compute_ms", 0.0))
            per_route_cpu_compute_ms = cpu_compute_ms / float(cpu_routes) if cpu_routes > 0 else 0.0
            event = {
                "layer_idx": int(getattr(self, "layer_idx", -1)),
                "token_count": int(hidden_states.shape[0]),
                "total_expert_count": int(round(float(prof.get("activated_expert_set_size_sum", 0.0)))),
                "cpu_expert_count": int(round(float(prof.get("realized_cpu_expert_count_sum", 0.0)))),
                "cpu_route_count": cpu_routes,
                "layer_moe_wall_ms": (perf_counter() - t0) * 1000.0,
                "route_ms": float(prof.get("route_ms", 0.0)),
                "plan_ms": float(prof.get("plan_ms", 0.0)),
                "gpu_compute_ms": float(prof.get("gpu_compute_ms", 0.0)),
                "cpu_prepare_ms": float(prof.get("cpu_prepare_ms", 0.0)),
                "cpu_compute_ms": cpu_compute_ms,
                "cpu_to_gpu_merge_ms": float(prof.get("cpu_to_gpu_merge_ms", 0.0)),
                "cpu_route_ratio": float(prof.get("cpu_route_ratio_sum", 0.0)),
                "per_route_cpu_compute_ms": per_route_cpu_compute_ms,
            }
            if is_verify:
                VERIFY_LAYER_EVENTS.append(event)
            else:
                DRAFT_LAYER_EVENTS.append(event)
        return out

    cls.forward = patched_forward
    cls._verify_count_stats_patched = True


def _stat(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
    return {
        "avg_ms": float(mean(values)),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
    }


def _hist_by(events: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[tuple(int(event[k]) for k in keys)].append(event)

    total = len(events)
    rows: list[dict[str, Any]] = []
    for key_values, group in sorted(grouped.items()):
        row = {k: v for k, v in zip(keys, key_values)}
        row["frequency"] = len(group)
        row["percent"] = float(len(group) / total * 100.0) if total else 0.0
        row["layer_moe_wall"] = _stat([float(x["layer_moe_wall_ms"]) for x in group])
        row["cpu_compute"] = _stat([float(x["cpu_compute_ms"]) for x in group])
        row["per_route_cpu_compute"] = _stat([float(x.get("per_route_cpu_compute_ms", 0.0)) for x in group])
        row["gpu_compute"] = _stat([float(x["gpu_compute_ms"]) for x in group])
        row["plan"] = _stat([float(x["plan_ms"]) for x in group])
        row["route"] = _stat([float(x["route_ms"]) for x in group])
        rows.append(row)
    return rows


def _layer_groups(events: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    if not events:
        return []
    valid_layers = [int(event.get("layer_idx", -1)) for event in events if int(event.get("layer_idx", -1)) >= 0]
    if not valid_layers:
        return []
    num_layers = max(valid_layers) + 1
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for event in events:
        layer_idx = int(event.get("layer_idx", -1))
        if current and layer_idx == 0:
            groups.append(current)
            current = []
        current.append(event)
        if len(current) == num_layers:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def _m3_perfect_fraction(
    draft_layer_events: list[dict[str, Any]],
    draft_steps_per_step: list[int] | None = None,
) -> dict[str, Any]:
    groups = _layer_groups(draft_layer_events)
    perfect_flags = [
        bool(group) and all(int(event.get("cpu_expert_count", 0) or 0) == 0 for event in group)
        for group in groups
    ]
    route_hits = [
        1.0 - (sum(float(event.get("cpu_route_ratio", 0.0) or 0.0) for event in group) / float(len(group)))
        for group in groups
        if group
    ]
    step0_indices: list[int] = []
    group_index = 0
    for draft_steps in draft_steps_per_step or []:
        steps = int(draft_steps)
        if steps <= 0:
            continue
        if group_index < len(groups):
            step0_indices.append(group_index)
        group_index += steps
    if not step0_indices and groups:
        step0_indices = [0]
    step0_flags = [perfect_flags[i] for i in step0_indices if i < len(perfect_flags)]
    return {
        "group_count": int(len(groups)),
        "perfect_count": int(sum(1 for flag in perfect_flags if flag)),
        "perfect_fraction": float(sum(1 for flag in perfect_flags if flag) / len(perfect_flags)) if perfect_flags else 0.0,
        "step0_group_count": int(len(step0_flags)),
        "step0_perfect_count": int(sum(1 for flag in step0_flags if flag)),
        "step0_perfect_fraction": float(sum(1 for flag in step0_flags if flag) / len(step0_flags)) if step0_flags else 0.0,
        "draft_layer_route_hit_rate_mean": float(mean(route_hits)) if route_hits else 0.0,
    }


def _acceptance_stats(engine_profile: dict[str, Any]) -> dict[str, Any]:
    traces = engine_profile.get("spec_step_traces", [])
    drafted = 0
    accepted = 0
    accepted_dist: Counter[int] = Counter()
    drafted_dist: Counter[int] = Counter()
    position_drafted: Counter[int] = Counter()
    position_accepted: Counter[int] = Counter()
    rejection_count = 0
    step_count = 0
    for step in traces:
        for seq in step.get("sequences", []):
            d = int(seq.get("drafted_tokens", 0) or 0)
            a = int(seq.get("accepted_draft_tokens", 0) or 0)
            a = max(0, min(a, d))
            drafted += d
            accepted += a
            accepted_dist[a] += 1
            drafted_dist[d] += 1
            for position in range(1, d + 1):
                position_drafted[position] += 1
            for position in range(1, a + 1):
                position_accepted[position] += 1
            rejection_count += int(bool(seq.get("rejected", False)))
            step_count += 1
    draft_position_acceptance = []
    for position in sorted(position_drafted):
        drafted_count = int(position_drafted[position])
        accepted_count = int(position_accepted[position])
        draft_position_acceptance.append(
            {
                "position": int(position),
                "drafted_count": drafted_count,
                "accepted_count": accepted_count,
                "acceptance_rate": float(accepted_count / drafted_count) if drafted_count else 0.0,
            }
        )
    return {
        "step_sequence_count": step_count,
        "drafted_tokens_total": drafted,
        "accepted_draft_tokens_total": accepted,
        "acceptance_rate": float(accepted / drafted) if drafted else 0.0,
        "rejection_count": int(rejection_count),
        "rejection_rate_per_step": float(rejection_count / step_count) if step_count else 0.0,
        "accepted_tokens_per_step_frequency": {str(k): int(v) for k, v in sorted(accepted_dist.items())},
        "drafted_tokens_per_step_frequency": {str(k): int(v) for k, v in sorted(drafted_dist.items())},
        "draft_position_acceptance": draft_position_acceptance,
    }


def _summarize_case(raw: dict[str, Any], layer_events: list[dict[str, Any]]) -> dict[str, Any]:
    ep = raw.get("engine_profile", {})
    pair_hist = _hist_by(layer_events, ("total_expert_count", "cpu_expert_count"))
    triple_hist = _hist_by(layer_events, ("total_expert_count", "cpu_expert_count", "cpu_route_count"))
    total_hist = _hist_by(layer_events, ("total_expert_count",))
    cpu_hist = _hist_by(layer_events, ("cpu_expert_count",))
    verify_calls = int(ep.get("spec_run_verify_calls", 0) or 0)
    generated_tokens = int(raw.get("generated_output_tokens", 0) or 0)
    spec_step_ms_total = float(ep.get("spec_spec_step_ms", 0.0) or 0.0)
    cpu_route_ratio = float(ep.get("model_cpu_route_ratio", ep.get("cpu_route_ratio", 0.0)) or 0.0)
    cpu_weight_mass_ratio = float(
        ep.get("model_cpu_weight_mass_ratio", ep.get("cpu_weight_mass_ratio", 0.0)) or 0.0
    )
    pre_transfer_miss = float(ep.get("model_pre_transfer_cache_miss", 0.0) or 0.0)
    pre_transfer_active = float(ep.get("model_pre_transfer_active_count", 0.0) or 0.0)
    if pre_transfer_active > 0:
        true_route_hit_rate = float(max(0.0, min(1.0, 1.0 - (pre_transfer_miss / pre_transfer_active))))
    else:
        # Fallback for old benchmarks without pre-transfer data:
        # use post-transfer cpu_route_ratio (accurate for "cpu" mode, inflated for cache_fill modes)
        true_route_hit_rate = float(max(0.0, min(1.0, 1.0 - cpu_route_ratio)))
    m3 = _m3_perfect_fraction(
        raw.get("draft_layer_events", []),
        draft_steps_per_step=list(ep.get("spec_draft_steps_per_step", []) or []),
    )
    runtime_m3_groups = int(ep.get("model_draft_m3_group_count", 0) or 0)
    if runtime_m3_groups > 0:
        m3 = {
            "group_count": runtime_m3_groups,
            "perfect_count": int(ep.get("model_draft_m3_perfect_count", 0) or 0),
            "perfect_fraction": float(ep.get("model_draft_m3_perfect_fraction", 0.0) or 0.0),
            "step0_group_count": int(ep.get("model_draft_m3_step0_group_count", 0) or 0),
            "step0_perfect_count": int(ep.get("model_draft_m3_step0_perfect_count", 0) or 0),
            "step0_perfect_fraction": float(ep.get("model_draft_m3_step0_perfect_fraction", 0.0) or 0.0),
            "draft_layer_route_hit_rate_mean": float(m3.get("draft_layer_route_hit_rate_mean", 0.0) or 0.0),
            "source": "runtime_metadata",
        }
    else:
        m3["source"] = "draft_layer_probe"
    return {
        "case": raw.get("case", {}),
        "elapsed_sec": float(raw.get("elapsed_sec", 0.0)),
        "generated_output_tokens": generated_tokens,
        "throughput_output_tok_s": float(raw.get("throughput_output_tok_s", 0.0)),
        "decode_phase_output_tok_s": (
            float(generated_tokens / (spec_step_ms_total / 1000.0)) if spec_step_ms_total > 0 else 0.0
        ),
        "outputs_digest": raw.get("outputs_digest", ""),
        "verify_calls": verify_calls,
        "verify_forward_ms_avg": float(ep.get("verify_forward_ms", 0.0)),
        "verify_forward_ms_total": float(ep.get("spec_run_verify_infer_ms_total", 0.0)),
        "draft_forward_ms_avg": float(ep.get("draft_forward_ms", 0.0)),
        "draft_forward_ms_total": float(ep.get("spec_run_draft_infer_ms_total", 0.0)),
        "prefill_forward_ms_total": float(ep.get("prefill_runner_ms", 0.0)),
        "spec_step_ms_total": spec_step_ms_total,
        "cuda_graph": {
            "draft_replay_count": int(ep.get("model_draft_graph_replay_count", 0) or 0),
            "standard_replay_count": int(ep.get("model_standard_graph_replay_count", 0) or 0),
            "total_replay_count": int(ep.get("model_graph_replay_count", 0) or 0),
            "hit_rate": float(ep.get("model_graph_hit_rate", 0.0) or 0.0),
            "verify_enabled": bool(raw.get("case", {}).get("verify_cuda_graph", False)),
            "verify_call_count": int(ep.get("model_verify_graph_call_count", 0) or 0),
            "verify_prefix_replay_count": int(ep.get("model_verify_prefix_graph_replay_count", 0) or 0),
            "verify_prefix_fallback_count": int(ep.get("model_verify_prefix_graph_fallback_count", 0) or 0),
            "verify_dense_replay_count": int(ep.get("model_verify_dense_graph_replay_count", 0) or 0),
            "verify_dense_fallback_count": int(ep.get("model_verify_dense_graph_fallback_count", 0) or 0),
            "verify_kt_hybrid_replay_count": int(ep.get("model_verify_kt_hybrid_graph_replay_count", 0) or 0),
        },
        "cache": {
            "route_hit_rate": float(max(0.0, min(1.0, 1.0 - cpu_route_ratio))),
            "route_miss_rate": cpu_route_ratio,
            "weight_hit_rate": float(max(0.0, min(1.0, 1.0 - cpu_weight_mass_ratio))),
            "weight_miss_rate": cpu_weight_mass_ratio,
            "activated_expert_set_size": float(ep.get("model_activated_expert_set_size", 0.0) or 0.0),
            "realized_cpu_expert_count": float(ep.get("model_realized_cpu_expert_count", 0.0) or 0.0),
            "true_route_hit_rate": true_route_hit_rate,
            "true_route_miss_rate": float(1.0 - true_route_hit_rate),
            "avg_miss_per_layer": pre_transfer_miss,
            "avg_active_per_layer": pre_transfer_active,
        },
        "prefetch": {
            "enabled": bool(raw.get("case", {}).get("prefetch_enabled", False)),
            "submit_count": int(ep.get("model_prefetch_submit_count", 0) or 0),
            "completed_count": int(ep.get("model_prefetch_completed_count", 0) or 0),
            "consumed_count": int(ep.get("model_prefetch_consumed_count", 0) or 0),
            "wait_ms_total": float(ep.get("model_prefetch_wait_ms", 0.0)),
            "verify_wait_ms_total": float(ep.get("spec_verify_prefetch_wait_ms", 0.0)),
            "draft_segment_indexed_submit_count": int(ep.get("model_draft_segment_indexed_prefetch_submit_count", 0) or 0),
            "draft_segment_indexed_ready_count": int(ep.get("model_draft_segment_indexed_prefetch_ready_count", 0) or 0),
            "draft_segment_indexed_publish_count": int(ep.get("model_draft_segment_indexed_prefetch_publish_count", 0) or 0),
            "draft_segment_indexed_consumed_count": int(ep.get("model_draft_segment_indexed_prefetch_consumed_count", 0) or 0),
            "draft_segment_indexed_submit_by_segment": ep.get("model_draft_segment_indexed_prefetch_submit_count_by_segment", {}),
            "draft_segment_indexed_consumed_by_segment": ep.get("model_draft_segment_indexed_prefetch_consumed_count_by_segment", {}),
            "predictive_phase1_submit_count": int(ep.get("model_predictive_phase1_prefetch_submit_count", 0) or 0),
            "predictive_draft_stale_observe_count": int(ep.get("model_predictive_draft_stale_observe_count", 0) or 0),
            "verify_layer_submit_count": int(ep.get("model_verify_layer_prefetch_submit_count", 0) or 0),
            "verify_layer_ready_count": int(ep.get("model_verify_layer_prefetch_ready_count", 0) or 0),
            "verify_layer_publish_count": int(ep.get("model_verify_layer_prefetch_publish_count", 0) or 0),
            "verify_layer_consumed_count": int(ep.get("model_verify_layer_prefetch_consumed_count", 0) or 0),
        },
        "verify_cache_fill": {
            "policy": raw.get("case", {}).get("spec_verify_miss_policy", "cpu"),
            "promoted_expert_count": int(ep.get("model_verify_cache_fill_promoted_expert_count", 0) or 0),
            "cpu_expert_count": int(ep.get("model_verify_cache_fill_cpu_expert_count", 0) or 0),
            "evicted_expert_count": int(ep.get("model_verify_cache_fill_evicted_expert_count", 0) or 0),
            "skipped_pending_count": int(ep.get("model_verify_cache_fill_skipped_pending_count", 0) or 0),
            "transfer_ms_total": float(ep.get("model_verify_cache_fill_transfer_ms", 0.0)),
            "no_cpu_remaining_miss_count": int(
                ep.get("model_verify_cache_fill_no_cpu_remaining_miss_count", 0) or 0
            ),
            "no_cpu_remaining_miss_expert_count": int(
                ep.get("model_verify_cache_fill_no_cpu_remaining_miss_expert_count", 0) or 0
            ),
            "no_cpu_remaining_miss_route_count": int(
                ep.get("model_verify_cache_fill_no_cpu_remaining_miss_route_count", 0) or 0
            ),
            "no_cpu_fallback_count": int(ep.get("model_verify_cache_fill_no_cpu_fallback_count", 0) or 0),
        },
        "acceptance": _acceptance_stats(ep),
        "m3": m3,
        "verify_layer_event_count": len(layer_events),
        "draft_layer_event_count": len(raw.get("draft_layer_events", [])),
        "hist_by_total_and_cpu_experts": pair_hist,
        "hist_by_total_experts": total_hist,
        "hist_by_cpu_experts": cpu_hist,
        "hist_by_total_cpu_and_routes": triple_hist,
    }


def run_single_case(args: argparse.Namespace) -> None:
    _install_verify_layer_probe(sync_layer_timing=args.sync_layer_timing)

    import torch
    from nanovllm import LLM, SamplingParams
    from transformers import AutoConfig

    DRAFT_LAYER_EVENTS.clear()
    VERIFY_LAYER_EVENTS.clear()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    hf_config = AutoConfig.from_pretrained(args.model_path)
    num_experts = int(getattr(hf_config, "num_experts"))
    slots = int(args.slots_per_layer)
    if slots <= 0:
        slots = int(round(num_experts * args.cache_ratio))
    effective_cache_ratio = float(slots / num_experts)

    case_info = {
        "cache_ratio": effective_cache_ratio,
        "slots_per_layer": slots,
        "prefetch_enabled": bool(args.prefetch_enabled),
        "acceptance_strategy": args.acceptance_strategy,
        "temperature": float(args.temperature),
        "num_seqs": int(args.num_seqs),
        "input_len": int(args.input_len),
        "output_len": int(args.output_len),
        "max_draft_tokens": int(args.max_draft_tokens),
        "draft_top_c": int(args.draft_top_c),
        "draft_reroute_policy": args.draft_reroute_policy,
        "draft_reroute_artifact": args.draft_reroute_artifact,
        "cpu_expert_backend": args.cpu_expert_backend,
        "kt_direct_backend": args.kt_direct_backend,
        "kt_num_threads": int(args.kt_num_threads),
        "kt_threadpool_count": int(args.kt_threadpool_count),
        "kt_chunked_prefill_size": int(args.kt_chunked_prefill_size),
        "kt_numa_nodes": args.kt_numa_nodes,
        "kt_capture_bs": args.kt_capture_bs,
        "cpu_expert_pin_memory": bool(args.cpu_expert_pin_memory),
        "spec_verify_miss_policy": args.spec_verify_miss_policy,
        "cache_strategy": args.cache_strategy,
        "rank_guard_threshold": float(args.rank_guard_threshold),
        "rank_guard_ema_alpha": float(args.rank_guard_ema_alpha),
        "prefetch_runtime_mode": args.prefetch_runtime_mode,
        "prefetch_runtime_kind": args.prefetch_runtime_kind,
        "prefetch_verify_attention_ratio": float(args.prefetch_verify_attention_ratio),
        "predictive_phase1_budget": int(args.predictive_phase1_budget),
        "draft_cuda_graph_enabled": bool(args.draft_cuda_graph_enabled),
        "draft_cuda_graph_cpu_backend": args.draft_cuda_graph_cpu_backend,
        "verify_cuda_graph": bool(args.verify_cuda_graph),
        "verify_cuda_graph_bucket_steps": _parse_int_csv(args.verify_cuda_graph_bucket_steps),
    }

    llm = LLM(
        args.model_path,
        dist_port=args.dist_port,
        enforce_eager=args.enforce_eager,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        inference_mode="spec",
        enable_heterogeneous=True,
        enable_speculative=True,
        heterogeneous_slots_per_layer=slots,
        max_draft_tokens=args.max_draft_tokens,
        draft_top_c=args.draft_top_c,
        draft_reroute_policy=args.draft_reroute_policy,
        draft_reroute_artifact=args.draft_reroute_artifact,
        acceptance_strategy=args.acceptance_strategy,
        acceptance_threshold=args.acceptance_threshold,
        cpu_expert_execution_enabled=True,
        cpu_expert_pin_memory=args.cpu_expert_pin_memory,
        cpu_expert_backend=args.cpu_expert_backend,
        cpu_expert_workspace_max_routes=args.cpu_expert_workspace_max_routes,
        cpu_expert_packed_min_routes=args.cpu_expert_packed_min_routes,
        cpu_expert_parallel_mode=args.cpu_expert_parallel_mode,
        cpu_expert_num_threads=args.cpu_expert_num_threads,
        kt_num_threads=args.kt_num_threads,
        kt_threadpool_count=args.kt_threadpool_count,
        kt_chunked_prefill_size=args.kt_chunked_prefill_size,
        kt_direct_backend=args.kt_direct_backend,
        kt_numa_nodes=_parse_int_csv(args.kt_numa_nodes) if args.kt_numa_nodes else [],
        kt_capture_bs=_parse_int_csv(args.kt_capture_bs),
        cpu_gpu_parallel_execution_enabled=args.cpu_gpu_parallel_execution_enabled,
        cpu_gpu_parallel_min_cpu_route_ratio=args.cpu_gpu_parallel_min_cpu_route_ratio,
        spec_verify_miss_policy=args.spec_verify_miss_policy,
        spec_profile=True,
        engine_profile=True,
        engine_profile_cuda_sync=True,
        spec_enable_prefetch=args.prefetch_enabled,
        cache_strategy=args.cache_strategy,
        rank_guard_threshold=args.rank_guard_threshold,
        rank_guard_ema_alpha=args.rank_guard_ema_alpha,
        prefetch_strategy=args.prefetch_strategy,
        prefetch_runtime_mode=args.prefetch_runtime_mode,
        prefetch_runtime_kind=args.prefetch_runtime_kind,
        prefetch_verify_attention_ratio=args.prefetch_verify_attention_ratio,
        predictive_phase1_budget=args.predictive_phase1_budget,
        prefetch_staging_slots_per_layer=args.prefetch_staging_slots_per_layer,
        prefetch_max_inflight=args.prefetch_max_inflight,
        prefetch_verify_layer_max_budget=args.prefetch_verify_layer_max_budget,
        prefetch_step_budget=args.prefetch_step_budget,
        cache_eviction_budget_per_step=args.cache_eviction_budget_per_step,
        prefetch_verify_wait_ms=args.prefetch_verify_wait_ms,
        prefetch_global_queue_capacity=args.prefetch_global_queue_capacity,
        prefetch_history_decay=args.prefetch_history_decay,
        prefetch_history_ttl_steps=args.prefetch_history_ttl_steps,
        prefetch_source_weight_prefill=args.prefetch_source_weight_prefill,
        prefetch_source_weight_verify=args.prefetch_source_weight_verify,
        prefetch_source_weight_draft=args.prefetch_source_weight_draft,
        prefetch_activation_count_weight=args.prefetch_activation_count_weight,
        prefetch_age_penalty=args.prefetch_age_penalty,
        prefetch_use_prefill_history=args.prefetch_use_prefill_history,
        prefetch_use_verify_history=args.prefetch_use_verify_history,
        prefetch_use_draft_live=args.prefetch_use_draft_live,
        draft_cuda_graph_enabled=args.draft_cuda_graph_enabled,
        draft_cuda_graph_cpu_backend=args.draft_cuda_graph_cpu_backend,
        verify_cuda_graph=args.verify_cuda_graph,
        verify_cuda_graph_bucket_steps=_parse_int_csv(args.verify_cuda_graph_bucket_steps),
    )

    custom_prompt = args.prompt_text
    if not custom_prompt and args.prompt_text_file:
        custom_prompt = Path(args.prompt_text_file).read_text(encoding="utf-8")
    if custom_prompt:
        prompt_texts = [custom_prompt]
        prompts = [llm.tokenizer.encode(custom_prompt)]
        case_info["actual_input_tokens"] = [len(prompts[0])]
    else:
        prompt_texts = _make_prompts(args.num_seqs, args.input_len, args.seed)
        prompts = _tokenize_prompts_to_length(llm.tokenizer, prompt_texts, args.input_len)
        case_info["actual_input_tokens"] = [len(prompt) for prompt in prompts]
    case_info["actual_input_tokens"] = [len(prompt) for prompt in prompts]
    sampling = [
        SamplingParams(temperature=args.temperature, ignore_eos=True, max_tokens=args.output_len)
        for _ in range(args.num_seqs)
    ]

    warmup_params = SamplingParams(temperature=args.temperature, ignore_eos=True, max_tokens=4)
    llm.generate(["Warmup request for verify layer profile."], warmup_params, use_tqdm=False)
    llm.get_profile(reset=True)
    DRAFT_LAYER_EVENTS.clear()
    VERIFY_LAYER_EVENTS.clear()

    t0 = time.time()
    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    elapsed = time.time() - t0
    profile = llm.get_profile(reset=True)

    llm.exit()

    token_ids = [x["token_ids"] for x in outputs]
    generated_text = [x.get("text", "") for x in outputs]
    generated_output_tokens = sum(len(x) for x in token_ids)
    digest_payload = "|".join(",".join(str(t) for t in seq) for seq in token_ids).encode("utf-8")
    import hashlib

    raw = {
        "case": case_info,
        "elapsed_sec": elapsed,
        "generated_output_tokens": generated_output_tokens,
        "throughput_output_tok_s": generated_output_tokens / elapsed if elapsed > 0 else 0.0,
        "outputs_digest": hashlib.sha256(digest_payload).hexdigest(),
        "generated_token_ids": token_ids,
        "generated_text": generated_text,
        "engine_profile": profile,
        "draft_layer_events": list(DRAFT_LAYER_EVENTS),
        "verify_layer_events": list(VERIFY_LAYER_EVENTS),
    }
    summary = _summarize_case(raw, list(VERIFY_LAYER_EVENTS))
    raw["summary"] = summary

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _case_name(ratio: float, prefetch: bool, backend: str, policy: str) -> str:
    ratio_pct = int(round(ratio * 100))
    safe_backend = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in backend)
    safe_policy = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in policy)
    return f"{safe_policy}_{safe_backend}_ratio{ratio_pct}_prefetch_{'on' if prefetch else 'off'}"


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Spec verify expert-count statistics",
        "",
        f"- model: `{summary['metadata']['model_path']}`",
        f"- timestamp: `{summary['metadata']['timestamp']}`",
        f"- output_dir: `{summary['metadata']['output_dir']}`",
        f"- layer timing: synchronized per verify MoE layer = `{summary['metadata']['sync_layer_timing']}`",
        f"- draft_top_c: `{summary['metadata']['draft_top_c']}`",
        f"- draft_reroute_policy: `{summary['metadata']['draft_reroute_policy']}`",
        f"- acceptance_strategy: `{summary['metadata']['acceptance_strategy']}`",
        f"- temperature: `{summary['metadata']['temperature']}`",
        f"- cpu expert backends: `{', '.join(summary['metadata']['cpu_expert_backends'])}`",
        "",
        "## Case summary",
        "",
        "| backend | cache ratio | prefetch | slots | verify calls | layer events | accept rate | rejects | drafted | accepted | draft graph replays | verify avg ms | draft avg ms | layer MoE avg ms | prefetch consumed | decode tok/s | e2e tok/s |",
        "|:---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in summary["cases"]:
        c = case["case"]
        acc = case["acceptance"]
        layer_avg = 0.0
        all_times = []
        for row in case["hist_by_total_and_cpu_experts"]:
            all_times.extend([row["layer_moe_wall"]["avg_ms"]] * row["frequency"])
        if all_times:
            layer_avg = float(mean(all_times))
        lines.append(
            "| "
            f"{c['cpu_expert_backend']} | "
            f"{c['cache_ratio']:.2f} | "
            f"{'on' if c['prefetch_enabled'] else 'off'} | "
            f"{c['slots_per_layer']} | "
            f"{case['verify_calls']} | "
            f"{case['verify_layer_event_count']} | "
            f"{acc['acceptance_rate']:.4f} | "
            f"{acc['rejection_count']} | "
            f"{acc['drafted_tokens_total']} | "
            f"{acc['accepted_draft_tokens_total']} | "
            f"{case['cuda_graph']['draft_replay_count']} | "
            f"{case['verify_forward_ms_avg']:.3f} | "
            f"{case['draft_forward_ms_avg']:.3f} | "
            f"{layer_avg:.3f} | "
            f"{case['prefetch']['consumed_count']} | "
            f"{case['decode_phase_output_tok_s']:.3f} | "
            f"{case['throughput_output_tok_s']:.3f} |"
        )

    for case in summary["cases"]:
        c = case["case"]
        lines.extend(
            [
                "",
                f"## backend={c['cpu_expert_backend']}, ratio={c['cache_ratio']:.2f}, prefetch={'on' if c['prefetch_enabled'] else 'off'}",
                "",
                "### By total expert count and CPU expert count",
                "",
                "| total experts | CPU experts | freq | percent | layer wall avg/min/max ms | CPU compute avg/min/max ms | GPU compute avg/min/max ms |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in case["hist_by_total_and_cpu_experts"]:
            wall = row["layer_moe_wall"]
            cpu = row["cpu_compute"]
            gpu = row["gpu_compute"]
            lines.append(
                "| "
                f"{row['total_expert_count']} | {row['cpu_expert_count']} | "
                f"{row['frequency']} | {row['percent']:.2f} | "
                f"{wall['avg_ms']:.3f}/{wall['min_ms']:.3f}/{wall['max_ms']:.3f} | "
                f"{cpu['avg_ms']:.3f}/{cpu['min_ms']:.3f}/{cpu['max_ms']:.3f} | "
                f"{gpu['avg_ms']:.3f}/{gpu['min_ms']:.3f}/{gpu['max_ms']:.3f} |"
            )
        lines.extend(
            [
                "",
                "### By total experts, CPU experts, and CPU routes (key for CPU compute stability)",
                "",
                "| total experts | CPU experts | CPU routes | freq | percent | layer wall avg/min/max ms | CPU compute avg/min/max ms | per-route CPU compute avg/min/max ms | gpu compute avg/min/max ms |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in case["hist_by_total_cpu_and_routes"]:
            wall = row["layer_moe_wall"]
            cpu = row["cpu_compute"]
            per_route_cpu = row["per_route_cpu_compute"]
            gpu = row["gpu_compute"]
            lines.append(
                "| "
                f"{row['total_expert_count']} | {row['cpu_expert_count']} | "
                f"{row['cpu_route_count']} | {row['frequency']} | {row['percent']:.2f} | "
                f"{wall['avg_ms']:.3f}/{wall['min_ms']:.3f}/{wall['max_ms']:.3f} | "
                f"{cpu['avg_ms']:.3f}/{cpu['min_ms']:.3f}/{cpu['max_ms']:.3f} | "
                f"{per_route_cpu['avg_ms']:.3f}/{per_route_cpu['min_ms']:.3f}/{per_route_cpu['max_ms']:.3f} | "
                f"{gpu['avg_ms']:.3f}/{gpu['min_ms']:.3f}/{gpu['max_ms']:.3f} |"
            )
        lines.extend(
            [
                "",
                "### Marginal by total experts",
                "",
                "| total experts | freq | percent | layer wall avg/min/max ms |",
                "|---:|---:|---:|---:|",
            ]
        )
        for row in case["hist_by_total_experts"]:
            wall = row["layer_moe_wall"]
            lines.append(
                f"| {row['total_expert_count']} | {row['frequency']} | {row['percent']:.2f} | "
                f"{wall['avg_ms']:.3f}/{wall['min_ms']:.3f}/{wall['max_ms']:.3f} |"
            )
        lines.extend(
            [
                "",
                "### Marginal by CPU experts",
                "",
                "| CPU experts | freq | percent | layer wall avg/min/max ms |",
                "|---:|---:|---:|---:|",
            ]
        )
        for row in case["hist_by_cpu_experts"]:
            wall = row["layer_moe_wall"]
            lines.append(
                f"| {row['cpu_expert_count']} | {row['frequency']} | {row['percent']:.2f} | "
                f"{wall['avg_ms']:.3f}/{wall['min_ms']:.3f}/{wall['max_ms']:.3f} |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_suite(args: argparse.Namespace) -> None:
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    ratios = [float(x) for x in args.cache_ratios.split(",")]
    prefetch_values = [False, True] if args.prefetch_order == "off,on" else [True, False]
    backend_values = [x.strip() for x in args.cpu_expert_backends.split(",") if x.strip()]
    if not backend_values:
        backend_values = [args.cpu_expert_backend]
    invalid_backends = sorted(set(backend_values) - {"torch", "torch_packed", "fused", "kt_kernel", "kt_direct"})
    if invalid_backends:
        raise ValueError(f"Invalid CPU expert backend(s): {invalid_backends}")
    script_path = Path(__file__).resolve()
    cases: list[dict[str, Any]] = []

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")

    case_index = 0
    for backend in backend_values:
        for ratio in ratios:
            for prefetch in prefetch_values:
                name = _case_name(ratio, prefetch, backend, args.draft_reroute_policy)
                case_json = outdir / f"{name}.json"
                case_log = outdir / f"{name}.log"
                cmd = [
                    sys.executable,
                    str(script_path),
                    "--single-case",
                    "--model-path",
                    args.model_path,
                    "--cache-ratio",
                    str(ratio),
                    "--slots-per-layer",
                    "0",
                    "--prefetch-enabled",
                    str(prefetch).lower(),
                    "--output",
                    str(case_json),
                    "--dist-port",
                    str(args.dist_port_base + case_index),
                    "--num-seqs",
                    str(args.num_seqs),
                    "--input-len",
                    str(args.input_len),
                    "--output-len",
                    str(args.output_len),
                    "--max-draft-tokens",
                    str(args.max_draft_tokens),
                    "--draft-top-c",
                    str(args.draft_top_c),
                    "--draft-reroute-policy",
                    args.draft_reroute_policy,
                    "--draft-reroute-artifact",
                    args.draft_reroute_artifact,
                    "--temperature",
                    str(args.temperature),
                    "--acceptance-strategy",
                    args.acceptance_strategy,
                    "--acceptance-threshold",
                    str(args.acceptance_threshold),
                    "--cpu-expert-backend",
                    backend,
                    "--cpu-expert-pin-memory",
                    str(args.cpu_expert_pin_memory).lower(),
                    "--cpu-expert-workspace-max-routes",
                    str(args.cpu_expert_workspace_max_routes),
                    "--cpu-expert-packed-min-routes",
                    str(args.cpu_expert_packed_min_routes),
                    "--cpu-expert-parallel-mode",
                    args.cpu_expert_parallel_mode,
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
                    args.cpu_gpu_parallel_execution_enabled,
                    "--cpu-gpu-parallel-min-cpu-route-ratio",
                    str(args.cpu_gpu_parallel_min_cpu_route_ratio),
                    "--spec-verify-miss-policy",
                    args.spec_verify_miss_policy,
                    "--max-num-batched-tokens",
                    str(args.max_num_batched_tokens),
                    "--max-num-seqs",
                    str(args.max_num_seqs),
                    "--max-model-len",
                    str(args.max_model_len),
                    "--gpu-memory-utilization",
                    str(args.gpu_memory_utilization),
                    "--enforce-eager",
                    str(args.enforce_eager).lower(),
                    "--prefetch-verify-wait-ms",
                    str(args.prefetch_verify_wait_ms),
                    "--prefetch-step-budget",
                    str(args.prefetch_step_budget),
                    "--prefetch-max-inflight",
                    str(args.prefetch_max_inflight),
                    "--prefetch-staging-slots-per-layer",
                    str(args.prefetch_staging_slots_per_layer),
                    "--cache-eviction-budget-per-step",
                    str(args.cache_eviction_budget_per_step),
                    "--prefetch-runtime-mode",
                    args.prefetch_runtime_mode,
                    "--prefetch-runtime-kind",
                    args.prefetch_runtime_kind,
                    "--prefetch-verify-attention-ratio",
                    str(args.prefetch_verify_attention_ratio),
                    "--predictive-phase1-budget",
                    str(args.predictive_phase1_budget),
                    "--prefetch-global-queue-capacity",
                    str(args.prefetch_global_queue_capacity),
                    "--prefetch-history-decay",
                    str(args.prefetch_history_decay),
                    "--prefetch-history-ttl-steps",
                    str(args.prefetch_history_ttl_steps),
                    "--prefetch-source-weight-prefill",
                    str(args.prefetch_source_weight_prefill),
                    "--prefetch-source-weight-verify",
                    str(args.prefetch_source_weight_verify),
                    "--prefetch-source-weight-draft",
                    str(args.prefetch_source_weight_draft),
                    "--prefetch-activation-count-weight",
                    str(args.prefetch_activation_count_weight),
                    "--prefetch-age-penalty",
                    str(args.prefetch_age_penalty),
                    "--prefetch-use-prefill-history",
                    str(args.prefetch_use_prefill_history).lower(),
                    "--prefetch-use-verify-history",
                    str(args.prefetch_use_verify_history).lower(),
                    "--prefetch-use-draft-live",
                    str(args.prefetch_use_draft_live).lower(),
                    "--draft-cuda-graph-enabled",
                    str(args.draft_cuda_graph_enabled).lower(),
                    "--draft-cuda-graph-cpu-backend",
                    args.draft_cuda_graph_cpu_backend,
                    "--verify-cuda-graph",
                    str(args.verify_cuda_graph).lower(),
                    "--verify-cuda-graph-bucket-steps",
                    args.verify_cuda_graph_bucket_steps,
                    "--rank-guard-threshold",
                    str(args.rank_guard_threshold),
                    "--rank-guard-ema-alpha",
                    str(args.rank_guard_ema_alpha),
                    "--seed",
                    str(args.seed),
                    "--sync-layer-timing",
                    str(args.sync_layer_timing).lower(),
                ]
                print(f"[{time.strftime('%H:%M:%S')}] running {name}", flush=True)
                t0 = time.time()
                with case_log.open("w", encoding="utf-8") as log_f:
                    proc = subprocess.run(
                        cmd,
                        cwd=Path(__file__).resolve().parents[2],
                        env=env,
                        stdout=log_f,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=args.case_timeout_sec,
                    )
                dt = time.time() - t0
                print(f"[{time.strftime('%H:%M:%S')}] {name} exit={proc.returncode} elapsed={dt:.1f}s", flush=True)
                if proc.returncode != 0:
                    tail = case_log.read_text(encoding="utf-8", errors="replace")[-4000:]
                    raise RuntimeError(f"case failed: {name}\n{tail}")
                raw = json.loads(case_json.read_text(encoding="utf-8"))
                cases.append(raw["summary"])
                case_index += 1

    summary = {
        "metadata": {
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "model_path": args.model_path,
            "output_dir": str(outdir),
            "cache_ratios": ratios,
            "cpu_expert_backends": backend_values,
            "draft_top_c": int(args.draft_top_c),
            "draft_reroute_policy": args.draft_reroute_policy,
            "draft_reroute_artifact": args.draft_reroute_artifact,
            "cache_strategy": args.cache_strategy,
            "spec_verify_miss_policy": args.spec_verify_miss_policy,
            "prefetch_runtime_mode": args.prefetch_runtime_mode,
            "prefetch_runtime_kind": args.prefetch_runtime_kind,
            "prefetch_verify_attention_ratio": float(args.prefetch_verify_attention_ratio),
            "predictive_phase1_budget": int(args.predictive_phase1_budget),
            "draft_cuda_graph_enabled": bool(args.draft_cuda_graph_enabled),
            "draft_cuda_graph_cpu_backend": args.draft_cuda_graph_cpu_backend,
            "verify_cuda_graph": bool(args.verify_cuda_graph),
            "verify_cuda_graph_bucket_steps": _parse_int_csv(args.verify_cuda_graph_bucket_steps),
            "acceptance_strategy": args.acceptance_strategy,
            "temperature": float(args.temperature),
            "cpu_expert_pin_memory": bool(args.cpu_expert_pin_memory),
            "prefetch_order": args.prefetch_order,
            "sync_layer_timing": bool(args.sync_layer_timing),
            "argv": sys.argv,
        },
        "cases": cases,
    }
    summary_json = outdir / "summary.json"
    summary_md = outdir / "summary.md"
    summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    _write_markdown(summary, summary_md)
    print(f"summary_json={summary_json}")
    print(f"summary_md={summary_md}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Collect verify-phase per-layer expert-count statistics in spec mode.")
    p.add_argument("--single-case", action="store_true")
    p.add_argument("--model-path", default="/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B")
    p.add_argument("--output-dir", default="")
    p.add_argument("--output", default="")
    p.add_argument("--cache-ratios", default="0.75,0.50,0.25")
    p.add_argument("--cache-ratio", type=float, default=0.75)
    p.add_argument("--slots-per-layer", type=int, default=0)
    p.add_argument("--prefetch-order", choices=["off,on", "on,off"], default="off,on")
    p.add_argument("--prefetch-enabled", type=str2bool, default=False)
    p.add_argument("--num-seqs", type=int, default=1)
    p.add_argument("--input-len", type=int, default=12)
    p.add_argument("--output-len", type=int, default=24)
    p.add_argument("--max-draft-tokens", type=int, default=4)
    p.add_argument("--draft-top-c", type=int, default=0)
    p.add_argument("--draft-reroute-policy", default="entropy_cache_bias",
                   choices=["round_robin", "drop_miss", "entropy_cache_bias",
                            "bounded_cache_bias", "similarity_replace"])
    p.add_argument("--draft-reroute-artifact", default="")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--acceptance-strategy", default="standard_sampling")
    p.add_argument("--acceptance-threshold", type=float, default=0.7)
    p.add_argument("--cpu-expert-backend", default="fused")
    p.add_argument(
        "--cpu-expert-backends",
        default="fused,torch",
        help="Comma-separated backend list for suite mode. Single-case mode uses --cpu-expert-backend.",
    )
    p.add_argument("--cpu-expert-workspace-max-routes", type=int, default=8192)
    p.add_argument("--cpu-expert-pin-memory", type=str2bool, default=True)
    p.add_argument("--cpu-expert-packed-min-routes", type=int, default=1)
    p.add_argument("--cpu-expert-parallel-mode", default="serial")
    p.add_argument("--cpu-expert-num-threads", type=int, default=4)
    p.add_argument("--kt-num-threads", type=int, default=0)
    p.add_argument("--kt-threadpool-count", type=int, default=1)
    p.add_argument("--kt-chunked-prefill-size", type=int, default=4096)
    p.add_argument(
        "--kt-direct-backend",
        choices=["auto", "amx_bf16", "avx2_bf16"],
        default="auto",
    )
    p.add_argument("--kt-numa-nodes", default="")
    p.add_argument("--kt-capture-bs", default="1,2,4,8,16,32")
    p.add_argument("--cpu-gpu-parallel-execution-enabled", default="auto")
    p.add_argument("--cpu-gpu-parallel-min-cpu-route-ratio", type=float, default=0.0)
    p.add_argument("--spec-verify-miss-policy", choices=["cpu", "cache_fill", "cache_fill_no_cpu"], default="cpu")
    p.add_argument("--max-num-batched-tokens", type=int, default=512)
    p.add_argument("--max-num-seqs", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=512)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--enforce-eager", type=str2bool, default=False)
    p.add_argument("--cache-strategy", default="lru")
    p.add_argument("--rank-guard-threshold", type=float, default=0.15)
    p.add_argument("--rank-guard-ema-alpha", type=float, default=0.95)
    p.add_argument("--prefetch-strategy", default="history_window")
    p.add_argument(
        "--prefetch-runtime-mode",
        choices=["baseline_staging", "draft_direct_active", "draft_segment_indexed"],
        default="baseline_staging",
    )
    p.add_argument("--prefetch-runtime-kind", choices=["legacy", "predictive"], default="legacy")
    p.add_argument("--prefetch-verify-attention-ratio", type=float, default=0.3)
    p.add_argument("--predictive-phase1-budget", type=int, default=4)
    p.add_argument("--prefetch-staging-slots-per-layer", type=int, default=2)
    p.add_argument("--prefetch-max-inflight", type=int, default=8)
    p.add_argument("--prefetch-step-budget", type=int, default=4)
    p.add_argument("--prefetch-verify-layer-max-budget", type=int, default=2)
    p.add_argument("--cache-eviction-budget-per-step", type=int, default=2)
    p.add_argument("--prefetch-verify-wait-ms", type=float, default=0.0)
    p.add_argument("--prefetch-global-queue-capacity", type=int, default=4096)
    p.add_argument("--prefetch-history-decay", type=float, default=0.9)
    p.add_argument("--prefetch-history-ttl-steps", type=int, default=64)
    p.add_argument("--prefetch-source-weight-prefill", type=float, default=1.0)
    p.add_argument("--prefetch-source-weight-verify", type=float, default=1.2)
    p.add_argument("--prefetch-source-weight-draft", type=float, default=1.5)
    p.add_argument("--prefetch-activation-count-weight", type=float, default=0.1)
    p.add_argument("--prefetch-age-penalty", type=float, default=0.02)
    p.add_argument("--prefetch-use-prefill-history", type=str2bool, default=True)
    p.add_argument("--prefetch-use-verify-history", type=str2bool, default=True)
    p.add_argument("--prefetch-use-draft-live", type=str2bool, default=True)
    p.add_argument("--draft-cuda-graph-enabled", type=str2bool, default=True)
    p.add_argument("--draft-cuda-graph-cpu-backend", choices=["none", "fused", "fused_sync"], default="none")
    p.add_argument("--verify-cuda-graph", type=str2bool, default=False)
    p.add_argument("--verify-cuda-graph-bucket-steps", default="4,8,12,16")
    p.add_argument("--dist-port", type=int, default=12345)
    p.add_argument("--dist-port-base", type=int, default=26500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sync-layer-timing", type=str2bool, default=True)
    p.add_argument("--prompt-text", default="",
                   help="Optional custom prompt text.")
    p.add_argument("--prompt-text-file", default="",
                   help="Optional path to file containing custom prompt text.")
    p.add_argument("--case-timeout-sec", type=int, default=1800)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.single_case:
        if not args.output:
            raise SystemExit("--output is required with --single-case")
        run_single_case(args)
    else:
        if not args.output_dir:
            ts = time.strftime("%Y%m%d_%H%M%S")
            args.output_dir = f"/home/mumura/moe_spec/logs/spec_verify_expert_count_stats_{ts}"
        run_suite(args)


if __name__ == "__main__":
    main()
