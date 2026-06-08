#!/usr/bin/env python3
"""Profile spec-verify with miss_policy=cpu + kt_direct, tracking prefetch effectiveness.

Key questions:
  - How many experts does prefetch transfer per step?
  - How many CPU routes remain per layer during verify?
  - Is prefetch not submitting enough, submitting the wrong experts, or arriving too late?
  - What is the kt_direct per-route CPU compute cost?

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/verify_kt_direct_profile.py \
      --output-len 256 --cache-ratio 0.25 \
      --cpu-expert-backend kt_direct --spec-verify-miss-policy cpu \
      --prefetch-enabled true \
      --max-model-len 8192 --max-draft-tokens 8 \
      --output results/verify_kt_direct_profile.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any, Union

VERIFY_LAYER_EVENTS: list[dict[str, Any]] = []
DRAFT_LAYER_EVENTS: list[dict[str, Any]] = []
MODEL_FORWARD_EVENTS: list[dict[str, Any]] = []
DECODER_LAYER_EVENTS: list[dict[str, Any]] = []
KT_DIRECT_FORWARD_EVENTS: list[dict[str, Any]] = []
PREFETCH_STEP_SNAPSHOTS: list[dict[str, Any]] = []
SYNC_LAYER_TIMING = True


def install_kt_direct_probe() -> None:
    """Wrap kt_direct forward() with perf_counter to verify cpu_compute_ms."""
    import torch
    try:
        from nanovllm.layers.fuse_moe.kt_direct_backend import KtDirectCpuMoeBackend
    except ImportError:
        return
    if getattr(KtDirectCpuMoeBackend, "_kt_direct_probe_patched", False):
        return

    orig_forward = KtDirectCpuMoeBackend.forward

    def patched_kt_forward(self, **kwargs):
        t0 = perf_counter()
        result = orig_forward(self, **kwargs)
        wall_ms = (perf_counter() - t0) * 1000.0
        KT_DIRECT_FORWARD_EVENTS.append({
            "wall_ms": wall_ms,
            "prep_ms": result.prep_ms,
            "compute_ms": result.compute_ms,
            "num_routes": int(kwargs.get("cpu_indices", torch.empty(0)).numel()),
        })
        return result

    KtDirectCpuMoeBackend.forward = patched_kt_forward
    KtDirectCpuMoeBackend._kt_direct_probe_patched = True


def install_verify_layer_probe(sync_layer_timing: bool = True) -> None:
    global SYNC_LAYER_TIMING
    SYNC_LAYER_TIMING = bool(sync_layer_timing)

    import torch
    from nanovllm.models import qwen3_moe

    cls = qwen3_moe.Qwen3MoeHeterogeneousSparseMoeBlock
    if getattr(cls, "_verify_per_layer_probe_patched", False):
        return

    original_forward = cls.forward

    def patched_forward(self, hidden_states):
        mode = getattr(self, "execution_mode", "normal")
        is_verify = mode == "verify"
        is_draft = mode == "draft"

        gpu_start: Any = None
        gpu_end: Any = None
        # NOTE: do NOT sync before t0 — that would wait for this layer's
        # attention to finish on GPU and inflate the MoE wall time.
        if is_verify and torch.cuda.is_available():
            gpu_start = torch.cuda.Event(enable_timing=True)
            gpu_end = torch.cuda.Event(enable_timing=True)
            gpu_start.record()
        t0 = perf_counter()
        out = original_forward(self, hidden_states)
        gpu_wall_ms = 0.0
        if is_verify and torch.cuda.is_available():
            gpu_end.record()
            torch.cuda.synchronize()
            gpu_wall_ms = gpu_start.elapsed_time(gpu_end)

        if is_verify or is_draft:
            prof = dict(getattr(self, "_last_profile", {}) or {})
            cpu_routes = int(round(float(prof.get("cpu_routes_sum", 0.0))))

            # Pre-transfer cache miss (measured before any cache_fill)
            pre_miss = int(round(float(prof.get("pre_transfer_cache_miss_sum", 0.0))))
            pre_active = int(round(float(prof.get("pre_transfer_active_count_sum", 0.0))))
            gpu_routes = max(0, pre_active - pre_miss)
            total_expert_count = int(round(float(prof.get("activated_expert_set_size_sum", 0.0))))
            cpu_expert_count = int(round(float(prof.get("realized_cpu_expert_count_sum", 0.0))))
            # Integrity check: with miss_policy=cpu, pre_miss should equal cpu_routes
            if is_verify and pre_active > 0:
                delta = abs(pre_miss - cpu_routes)
                if delta > max(1, pre_active * 0.05):
                    import sys as _sys
                    print(f"[WARN] layer {self.layer_idx}: pre_miss={pre_miss} "
                          f"cpu_routes={cpu_routes} delta={delta} — "
                          f"expected equal with miss_policy=cpu", file=_sys.stderr)

            cpu_wall_ms = (perf_counter() - t0) * 1000.0
            # parallel_wall_ms is the CPU wall time inside heterogeneous_moe_forward
            parallel_wall = float(prof.get("parallel_wall_ms", 0.0))
            # CPU overhead: CPU wall minus GPU time minus route/plan (all CPU-side)
            cpu_overhead_ms = max(0.0, cpu_wall_ms - gpu_wall_ms
                                  - float(prof.get("route_ms", 0.0))
                                  - float(prof.get("plan_ms", 0.0)))

            event = {
                "layer_idx": int(getattr(self, "layer_idx", -1)),
                "token_count": int(hidden_states.shape[0]),
                "total_expert_count": total_expert_count,
                "cpu_expert_count": cpu_expert_count,
                "gpu_expert_count": max(0, total_expert_count - cpu_expert_count),
                "cpu_route_count": cpu_routes,
                "gpu_route_count": gpu_routes,
                "pre_miss_route_count": pre_miss,
                "total_active_routes": pre_active,
                "pre_transfer_hit_rate": (gpu_routes / pre_active) if pre_active > 0 else 0.0,
                # Timing (all ms): CPU wall = total, GPU wall = GPU execution via CUDA events
                "layer_moe_wall_ms": cpu_wall_ms,
                "layer_moe_gpu_wall_ms": gpu_wall_ms,
                "cpu_overhead_ms": cpu_overhead_ms,
                "parallel_wall_ms": parallel_wall,
                # Sub-timers from profile (perf_counter based = kernel launch overhead for GPU ops)
                "plan_ms": float(prof.get("plan_ms", 0.0)),
                "route_ms": float(prof.get("route_ms", 0.0)),
                "gpu_gather_ms": float(prof.get("gpu_gather_ms", 0.0)),
                "gpu_compute_ms": float(prof.get("gpu_compute_ms", 0.0)),
                "cpu_prepare_ms": float(prof.get("cpu_prepare_ms", 0.0)),
                "cpu_compute_ms": float(prof.get("cpu_compute_ms", 0.0)),
                "scatter_ms": float(prof.get("scatter_ms", 0.0)),
                "cpu_to_gpu_merge_ms": float(prof.get("cpu_to_gpu_merge_ms", 0.0)),
                "cpu_route_ratio": float(prof.get("cpu_route_ratio_sum", 0.0)),
            }
            if is_verify:
                VERIFY_LAYER_EVENTS.append(event)
            else:
                DRAFT_LAYER_EVENTS.append(event)
        return out

    cls.forward = patched_forward
    cls._verify_per_layer_probe_patched = True


def install_model_forward_probe() -> None:
    """Patch Qwen3MoeModel.forward_layers and decoder_layer to measure
    model-forward and per-decoder-layer wall time (attention + MoE)."""
    import torch
    from nanovllm.models import qwen3_moe

    model_cls = qwen3_moe.Qwen3MoeModel
    if getattr(model_cls, "_model_forward_probe_patched", False):
        return

    # Patch decoder_layer (one transformer block = attention + MoE)
    layer_cls = qwen3_moe.Qwen3MoeDecoderLayer
    orig_decoder_forward = layer_cls.forward

    def patched_decoder_forward(self, hidden_states, position_ids, *args, **kwargs):
        t0 = perf_counter()
        out = orig_decoder_forward(self, hidden_states, position_ids, *args, **kwargs)
        if getattr(getattr(self, "mlp", None), "execution_mode", "normal") == "verify":
            DECODER_LAYER_EVENTS.append({
                "layer_idx": int(getattr(self, "layer_idx", -1)),
                "wall_ms": (perf_counter() - t0) * 1000.0,
            })
        return out

    layer_cls.forward = patched_decoder_forward

    # Patch forward_layers to get total model forward time
    orig_forward_layers = model_cls.forward_layers

    def patched_forward_layers(self, hidden_states, position_ids, *,
                               start_layer=0, end_layer=None, apply_norm=True):
        if end_layer is None:
            end_layer = len(self.layers)
        t0 = perf_counter()
        out = orig_forward_layers(self, hidden_states, position_ids,
                                  start_layer=start_layer, end_layer=end_layer,
                                  apply_norm=apply_norm)
        MODEL_FORWARD_EVENTS.append({
            "wall_ms": (perf_counter() - t0) * 1000.0,
            "num_layers": end_layer - start_layer,
        })
        return out

    model_cls.forward_layers = patched_forward_layers
    model_cls._model_forward_probe_patched = True


def install_prefetch_snapshot_probe(model_runner) -> None:
    """Patch model_runner to snapshot prefetch counters before each verify call."""
    if getattr(model_runner, "_prefetch_snapshot_patched", False):
        return

    orig_run_verify = model_runner.run_verify

    def patched_run_verify(*args, **kwargs):
        try:
            return orig_run_verify(*args, **kwargs)
        finally:
            ep = getattr(model_runner, "_profile", {}) or {}
            # Also read prefetcher runtime counters (stored in a separate dict)
            pf_ep = {}
            prefetch_runtime = getattr(model_runner, "prefetch_runtime", None)
            if prefetch_runtime is not None:
                try:
                    pf_ep = prefetch_runtime.get_profile(reset=False)
                except Exception:
                    pass
            snapshot = {
                "prefetch_submit": int(float(ep.get("prefetch_submit_count", 0.0)
                    or pf_ep.get("prefetch_submit_count", 0))),
                "prefetch_completed": int(float(ep.get("prefetch_completed_count", 0.0)
                    or pf_ep.get("prefetch_completed_count", 0))),
                "prefetch_consumed": int(float(ep.get("prefetch_consumed_count", 0.0)
                    or pf_ep.get("prefetch_consumed_count", 0))),
                "verify_layer_prefetch_submit": int(
                    float(ep.get("verify_layer_prefetch_submit_count", 0.0)
                        or pf_ep.get("verify_layer_prefetch_submit_count", 0))
                ),
                "verify_layer_prefetch_consumed": int(
                    float(ep.get("verify_layer_prefetch_consumed_count", 0.0)
                        or pf_ep.get("verify_layer_prefetch_consumed_count", 0))
                ),
                "verify_layer_prefetch_ready": int(
                    float(pf_ep.get("verify_layer_prefetch_ready_count", 0))
                ),
                "verify_layer_prefetch_publish": int(
                    float(pf_ep.get("verify_layer_prefetch_publish_count", 0))
                ),
                "model_cpu_route_ratio": float(ep.get("cpu_route_ratio", 0.0)),
                "model_realized_cpu_expert_count": float(ep.get("realized_cpu_expert_count", 0.0)),
            }
            PREFETCH_STEP_SNAPSHOTS.append(snapshot)

    model_runner.run_verify = patched_run_verify
    model_runner._prefetch_snapshot_patched = True


def _stat(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    return {
        "avg": float(mean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "median": float(median(values)),
    }


def analyze_verify_layer_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    if not events:
        return {"error": "no verify layer events"}

    by_layer: dict[int, list[dict]] = defaultdict(list)
    for e in events:
        by_layer[e["layer_idx"]].append(e)

    num_layers = max(by_layer) + 1
    per_layer: dict[str, Any] = {}
    all_cpu_compute: list[float] = []
    all_cpu_prepare: list[float] = []
    all_gpu_gather: list[float] = []
    all_gpu_compute: list[float] = []
    all_plan: list[float] = []
    all_route: list[float] = []
    all_scatter: list[float] = []
    all_merge: list[float] = []
    all_wall: list[float] = []
    all_gpu_wall: list[float] = []
    all_cpu_overhead: list[float] = []
    all_parallel_wall: list[float] = []
    all_cpu_experts: list[float] = []
    all_gpu_experts: list[float] = []
    all_total_experts: list[float] = []
    all_cpu_routes: list[float] = []
    all_gpu_routes: list[float] = []
    all_pre_hit_rate: list[float] = []

    for li in sorted(by_layer):
        group = by_layer[li]
        per_layer[str(li)] = {
            "call_count": len(group),
            "total_expert_count": _stat([e["total_expert_count"] for e in group]),
            "cpu_expert_count": _stat([e["cpu_expert_count"] for e in group]),
            "gpu_expert_count": _stat([e["gpu_expert_count"] for e in group]),
            "cpu_route_count": _stat([e["cpu_route_count"] for e in group]),
            "gpu_route_count": _stat([e["gpu_route_count"] for e in group]),
            "pre_miss_route_count": _stat([e["pre_miss_route_count"] for e in group]),
            "pre_transfer_hit_rate": _stat([e["pre_transfer_hit_rate"] for e in group]),
            "layer_moe_wall_ms": _stat([e["layer_moe_wall_ms"] for e in group]),
            "layer_moe_gpu_wall_ms": _stat([e["layer_moe_gpu_wall_ms"] for e in group]),
            "cpu_overhead_ms": _stat([e["cpu_overhead_ms"] for e in group]),
            "parallel_wall_ms": _stat([e["parallel_wall_ms"] for e in group]),
            "plan_ms": _stat([e["plan_ms"] for e in group]),
            "route_ms": _stat([e["route_ms"] for e in group]),
            "gpu_gather_ms": _stat([e["gpu_gather_ms"] for e in group]),
            "gpu_compute_ms": _stat([e["gpu_compute_ms"] for e in group]),
            "cpu_prepare_ms": _stat([e["cpu_prepare_ms"] for e in group]),
            "cpu_compute_ms": _stat([e["cpu_compute_ms"] for e in group]),
            "scatter_ms": _stat([e["scatter_ms"] for e in group]),
            "merge_ms": _stat([e["cpu_to_gpu_merge_ms"] for e in group]),
        }
        all_cpu_compute.extend(e["cpu_compute_ms"] for e in group)
        all_cpu_prepare.extend(e["cpu_prepare_ms"] for e in group)
        all_gpu_gather.extend(e["gpu_gather_ms"] for e in group)
        all_gpu_compute.extend(e["gpu_compute_ms"] for e in group)
        all_plan.extend(e["plan_ms"] for e in group)
        all_route.extend(e["route_ms"] for e in group)
        all_scatter.extend(e["scatter_ms"] for e in group)
        all_merge.extend(e["cpu_to_gpu_merge_ms"] for e in group)
        all_wall.extend(e["layer_moe_wall_ms"] for e in group)
        all_gpu_wall.extend(e["layer_moe_gpu_wall_ms"] for e in group)
        all_cpu_overhead.extend(e["cpu_overhead_ms"] for e in group)
        all_parallel_wall.extend(e["parallel_wall_ms"] for e in group)
        all_cpu_experts.extend(e["cpu_expert_count"] for e in group)
        all_gpu_experts.extend(e["gpu_expert_count"] for e in group)
        all_total_experts.extend(e["total_expert_count"] for e in group)
        all_cpu_routes.extend(e["cpu_route_count"] for e in group)
        all_gpu_routes.extend(e["gpu_route_count"] for e in group)
        all_pre_hit_rate.extend(e["pre_transfer_hit_rate"] for e in group)

    per_layer_cpu = sum(
        float(per_layer[str(li)]["cpu_compute_ms"]["avg"]) for li in range(num_layers) if str(li) in per_layer
    )
    per_layer_gpu = sum(
        float(per_layer[str(li)]["gpu_compute_ms"]["avg"]) for li in range(num_layers) if str(li) in per_layer
    )
    per_layer_route = sum(
        float(per_layer[str(li)]["route_ms"]["avg"]) for li in range(num_layers) if str(li) in per_layer
    )
    per_layer_gpu_gather = sum(
        float(per_layer[str(li)]["gpu_gather_ms"]["avg"]) for li in range(num_layers) if str(li) in per_layer
    )
    per_layer_scatter = sum(
        float(per_layer[str(li)]["scatter_ms"]["avg"]) for li in range(num_layers) if str(li) in per_layer
    )
    per_layer_plan = sum(
        float(per_layer[str(li)]["plan_ms"]["avg"]) for li in range(num_layers) if str(li) in per_layer
    )
    per_layer_merge = sum(
        float(per_layer[str(li)]["merge_ms"]["avg"]) for li in range(num_layers) if str(li) in per_layer
    )
    per_layer_cpu_prep = sum(
        float(per_layer[str(li)]["cpu_prepare_ms"]["avg"]) for li in range(num_layers) if str(li) in per_layer
    )
    per_layer_gpu_wall = sum(
        float(per_layer[str(li)]["layer_moe_gpu_wall_ms"]["avg"]) for li in range(num_layers) if str(li) in per_layer
    )
    per_layer_cpu_overhead = sum(
        float(per_layer[str(li)]["cpu_overhead_ms"]["avg"]) for li in range(num_layers) if str(li) in per_layer
    )
    per_layer_wall = sum(
        float(per_layer[str(li)]["layer_moe_wall_ms"]["avg"]) for li in range(num_layers) if str(li) in per_layer
    )

    per_route_cpu_ms: list[float] = []
    for e in events:
        routes = e["cpu_route_count"]
        if routes > 0:
            per_route_cpu_ms.append(e["cpu_compute_ms"] / float(routes))

    top_cpu = sorted(
        [(li, float(per_layer[str(li)]["cpu_compute_ms"]["avg"])) for li in range(num_layers) if str(li) in per_layer],
        key=lambda x: x[1], reverse=True,
    )[:8]

    return {
        "num_layers": num_layers,
        "total_verify_layer_events": len(events),
        "per_layer": per_layer,
        "aggregate": {
            "cpu_compute_ms": _stat(all_cpu_compute),
            "gpu_gather_ms": _stat(all_gpu_gather),
            "gpu_compute_ms": _stat(all_gpu_compute),
            "cpu_prepare_ms": _stat(all_cpu_prepare),
            "plan_ms": _stat(all_plan),
            "route_ms": _stat(all_route),
            "scatter_ms": _stat(all_scatter),
            "merge_ms": _stat(all_merge),
            "layer_moe_wall_ms": _stat(all_wall),
            "layer_moe_gpu_wall_ms": _stat(all_gpu_wall),
            "cpu_overhead_ms": _stat(all_cpu_overhead),
            "parallel_wall_ms": _stat(all_parallel_wall),
            "per_route_cpu_ms": _stat(per_route_cpu_ms),
            "cpu_expert_count": _stat(all_cpu_experts),
            "gpu_expert_count": _stat(all_gpu_experts),
            "total_expert_count": _stat(all_total_experts),
            "cpu_route_count": _stat(all_cpu_routes),
            "gpu_route_count": _stat(all_gpu_routes),
            "pre_transfer_hit_rate": _stat(all_pre_hit_rate),
        },
        "budget_per_verify_forward": {
            "plan_ms_sum": per_layer_plan,
            "route_ms_sum": per_layer_route,
            "gpu_gather_ms_sum": per_layer_gpu_gather,
            "gpu_compute_ms_sum": per_layer_gpu,
            "cpu_prepare_ms_sum": per_layer_cpu_prep,
            "cpu_compute_ms_sum": per_layer_cpu,
            "scatter_ms_sum": per_layer_scatter,
            "merge_ms_sum": per_layer_merge,
            "gpu_wall_ms_sum": per_layer_gpu_wall,
            "cpu_overhead_ms_sum": per_layer_cpu_overhead,
            "moe_wall_ms_sum": per_layer_wall,
        },
        "top_cpu_heavy_layers": [{"layer_idx": li, "cpu_compute_ms_avg": v} for li, v in top_cpu],
    }


def analyze_prefetch_snapshots(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute deltas between consecutive prefetch snapshots to see per-step activity."""
    if len(snapshots) < 2:
        return {"error": "not enough snapshots", "count": len(snapshots)}

    deltas: list[dict[str, Any]] = []
    for i in range(1, len(snapshots)):
        prev = snapshots[i - 1]
        curr = snapshots[i]
        deltas.append({k: curr[k] - prev[k] for k in curr})

    keys = [k for k in deltas[0].keys() if "prefetch" in k or "cache_fill" in k]
    delta_stats: dict[str, dict] = {}
    for k in keys:
        vals = [d[k] for d in deltas]
        delta_stats[k] = {
            "per_step_avg": float(mean(vals)),
            "per_step_min": float(min(vals)),
            "per_step_max": float(max(vals)),
            "per_step_sum": float(sum(vals)),
        }

    # analyze correlation: does prefetch_submit correlate with reduced cpu_routes?
    return {
        "snapshot_count": len(snapshots),
        "delta_count": len(deltas),
        "per_step_deltas": delta_stats,
        "first_snapshot": snapshots[0],
        "last_snapshot": snapshots[-1],
        "total_delta": {k: snapshots[-1][k] - snapshots[0][k] for k in snapshots[0]},
    }


# ── main ─────────────────────────────────────────────────────────────────────
def run(args: argparse.Namespace) -> dict[str, Any]:
    install_verify_layer_probe(sync_layer_timing=args.sync_layer_timing)
    install_model_forward_probe()
    install_kt_direct_probe()

    import torch
    from nanovllm import LLM, SamplingParams
    from transformers import AutoConfig

    VERIFY_LAYER_EVENTS.clear()
    DRAFT_LAYER_EVENTS.clear()
    MODEL_FORWARD_EVENTS.clear()
    DECODER_LAYER_EVENTS.clear()
    KT_DIRECT_FORWARD_EVENTS.clear()
    PREFETCH_STEP_SNAPSHOTS.clear()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    hf_config = AutoConfig.from_pretrained(args.model_path)
    num_experts = int(getattr(hf_config, "num_experts"))
    slots = int(args.slots_per_layer)
    if slots <= 0:
        slots = int(round(num_experts * args.cache_ratio))
    effective_cache_ratio = float(slots / num_experts)

    kt_numa = [int(x) for x in args.kt_numa_nodes.split(",") if x.strip()] if args.kt_numa_nodes else []
    kt_bs = [int(x) for x in args.kt_capture_bs.split(",") if x.strip()] if args.kt_capture_bs else [1, 2, 4, 8, 16, 32]
    verify_buckets = [int(x) for x in args.verify_cuda_graph_bucket_steps.split(",") if x.strip()] if args.verify_cuda_graph_bucket_steps else [4, 8, 12, 16]

    prefetch_enabled = args.prefetch_enabled

    print(f"Model:      {args.model_path}")
    print(f"num_experts={num_experts}  slots={slots}  cache_ratio={effective_cache_ratio:.3f}")
    print(f"cpu_expert_backend={args.cpu_expert_backend}")
    print(f"spec_verify_miss_policy={args.spec_verify_miss_policy}")
    print(f"prefetch_enabled={prefetch_enabled}  "
          f"prefetch_step_budget={args.prefetch_step_budget}  "
          f"prefetch_max_inflight={args.prefetch_max_inflight}")
    print(f"max_draft_tokens={args.max_draft_tokens}  output_len={args.output_len}")
    print()

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
        temperature=args.temperature,
        cpu_expert_execution_enabled=True,
        cpu_expert_backend=args.cpu_expert_backend,
        cpu_expert_pin_memory=args.cpu_expert_pin_memory,
        cpu_expert_workspace_max_routes=args.cpu_expert_workspace_max_routes,
        cpu_expert_packed_min_routes=1,
        cpu_expert_parallel_mode="serial",
        cpu_expert_num_threads=args.cpu_expert_num_threads,
        cpu_expert_strict_dtype=args.cpu_expert_strict_dtype,
        kt_num_threads=args.kt_num_threads,
        kt_threadpool_count=args.kt_threadpool_count,
        kt_chunked_prefill_size=args.kt_chunked_prefill_size,
        kt_direct_backend=args.kt_direct_backend,
        kt_numa_nodes=kt_numa,
        kt_capture_bs=kt_bs,
        spec_verify_miss_policy=args.spec_verify_miss_policy,
        cache_strategy=args.cache_strategy,
        spec_enable_prefetch=prefetch_enabled,
        prefetch_step_budget=args.prefetch_step_budget,
        prefetch_max_inflight=args.prefetch_max_inflight,
        prefetch_verify_layer_max_budget=args.prefetch_verify_layer_max_budget,
        prefetch_staging_slots_per_layer=args.prefetch_staging_slots_per_layer,
        prefetch_verify_attention_ratio=args.prefetch_verify_attention_ratio,
        cache_eviction_budget_per_step=args.cache_eviction_budget_per_step,
        prefetch_global_queue_capacity=args.prefetch_global_queue_capacity,
        prefetch_use_prefill_history=args.prefetch_use_prefill_history,
        prefetch_use_verify_history=args.prefetch_use_verify_history,
        prefetch_use_draft_live=args.prefetch_use_draft_live,
        cpu_gpu_parallel_execution_enabled="auto",
        cpu_gpu_parallel_min_cpu_route_ratio=0.0,
        draft_cuda_graph_enabled=args.draft_cuda_graph_enabled,
        draft_cuda_graph_cpu_backend=args.draft_cuda_graph_cpu_backend,
        verify_cuda_graph=args.verify_cuda_graph,
        verify_cuda_graph_bucket_steps=verify_buckets,
        seed=args.seed,
    )

    # Install prefetch counter probe
    install_prefetch_snapshot_probe(llm.model_runner)

    prompt = (
        "Expert caching for sparse mixture-of-experts inference is a practical systems problem. "
        "A serving engine usually keeps only part of the expert weights in GPU memory and leaves "
        "the rest in CPU memory. When routing selects an uncached expert, the engine must either "
        "compute on CPU or transfer weights before the layer needs them."
    )
    sp = SamplingParams(temperature=args.temperature, max_tokens=args.output_len, ignore_eos=True)

    print("Warmup ...", flush=True)
    _ = llm.generate([prompt], sp)

    VERIFY_LAYER_EVENTS.clear()
    DRAFT_LAYER_EVENTS.clear()
    MODEL_FORWARD_EVENTS.clear()
    DECODER_LAYER_EVENTS.clear()
    KT_DIRECT_FORWARD_EVENTS.clear()
    PREFETCH_STEP_SNAPSHOTS.clear()

    print(f"Benchmark run (output_len={args.output_len}) ...", flush=True)
    t0 = perf_counter()
    result = llm.generate([prompt], sp)
    elapsed = perf_counter() - t0

    profile = llm.get_profile(reset=False)
    ep = profile.get("engine_profile", {})
    generated_tokens = int(profile.get("generated_output_tokens", 0) or profile.get("decode_count", 0) or 0)
    throughput = float(profile.get("throughput_output_tok_s", 0.0) or 0.0)

    print(f"Generated {generated_tokens} tokens in {elapsed:.1f}s ({throughput:.2f} tok/s)")

    # ── acceptance rate ──────────────────────────────────────────────────
    drafted = float(profile.get("spec_draft_tokens_total", 0) or 0)
    accepted = float(profile.get("spec_accepted_tokens_total", 0) or 0)
    acceptance_rate = float(accepted / drafted) if drafted else 0.0

    # ── cache hit rate ───────────────────────────────────────────────────
    pre_miss_sum = float(profile.get("pre_transfer_cache_miss", 0.0) or 0.0)
    pre_active_sum = float(profile.get("pre_transfer_active_count", 0.0) or 0.0)
    true_route_hit_rate = (
        float(max(0.0, min(1.0, 1.0 - pre_miss_sum / pre_active_sum)))
        if pre_active_sum > 0 else 0.0
    )
    avg_miss = pre_miss_sum
    avg_active = pre_active_sum

    # prefetch counter sums (from merged profile, model_runner keys)
    pf_submit = int(float(profile.get("prefetch_submit_count", 0)))
    pf_completed = int(float(profile.get("prefetch_completed_count", 0)))
    pf_consumed = int(float(profile.get("prefetch_consumed_count", 0)))
    draft_graph = 0  # not in this profile
    draft_ms = float(profile.get("draft_forward_ms", 0))
    verify_ms = float(profile.get("verify_forward_ms", 0))

    # ── analysis ─────────────────────────────────────────────────────────
    verify_analysis = analyze_verify_layer_events(VERIFY_LAYER_EVENTS)
    draft_analysis = analyze_verify_layer_events(DRAFT_LAYER_EVENTS) if DRAFT_LAYER_EVENTS else {}
    prefetch_analysis = analyze_prefetch_snapshots(PREFETCH_STEP_SNAPSHOTS)

    agg = verify_analysis["aggregate"]
    budget = verify_analysis["budget_per_verify_forward"]
    verify_calls = int(ep.get("spec_run_verify_calls", 0) or 0)
    if verify_calls <= 0:
        verify_calls = len(PREFETCH_STEP_SNAPSHOTS)
    pre_hit = agg["pre_transfer_hit_rate"]
    # Compute unaccounted MoE time
    # GPU wall time (CUDA events) vs CPU overhead
    gpu_wall_sum = budget.get("gpu_wall_ms_sum", 0)
    cpu_overhead_sum = budget.get("cpu_overhead_ms_sum", 0)
    cpu_overhead_sum = max(0, cpu_overhead_sum)  # may be slightly negative due to measurement error
    # accounted = all sub-timers + GPU wall + CPU overhead
    accounted_ms_sum = (budget.get("plan_ms_sum", 0) + budget.get("route_ms_sum", 0)
                        + budget.get("cpu_prepare_ms_sum", 0)
                        + budget.get("cpu_compute_ms_sum", 0)
                        + gpu_wall_sum + cpu_overhead_sum)
    unaccounted_ms_sum = max(0, budget["moe_wall_ms_sum"] - accounted_ms_sum)

    # ── report ───────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("VERIFY PROFILE  (miss_policy=cpu)")
    print("=" * 72)

    print(f"\n  Verify calls: {verify_calls}")
    print(f"  Verify forward ms avg:  {verify_ms:.3f}")
    print(f"  Draft forward ms avg:   {draft_ms:.3f}")
    print(f"  Total layer events:     {verify_analysis['total_verify_layer_events']}")

    # ── model forward / decoder layer analysis ─────────────────────────────
    if MODEL_FORWARD_EVENTS:
        mf_avg = mean([e["wall_ms"] for e in MODEL_FORWARD_EVENTS])
        print(f"\n  Model forward (layers+norm): {mf_avg:.1f} ms  ({len(MODEL_FORWARD_EVENTS)} calls)")
    if DECODER_LAYER_EVENTS:
        dl_avg = mean([e["wall_ms"] for e in DECODER_LAYER_EVENTS])
        dl_sum = sum([e["wall_ms"] for e in DECODER_LAYER_EVENTS])
        num_dl = len(DECODER_LAYER_EVENTS)
        num_layers_per_call = num_dl // max(len(MODEL_FORWARD_EVENTS), 1)
        dl_per_call = dl_sum / max(len(MODEL_FORWARD_EVENTS), 1)
        print(f"  Decoder layer avg:           {dl_avg:.1f} ms  ({num_dl} events, {num_layers_per_call} layers/call)")
        print(f"  Decoder layers per fwd:      {dl_per_call:.1f} ms  (attention + MoE)")
        # non-MoE per forward = decoder_layers_sum - MoE_wall_sum
        moe_wall_total = budget.get("moe_wall_ms_sum", 0)
        non_moe_from_decoder = max(0, dl_per_call - moe_wall_total)
        print(f"  Non-MoE (decoder - MoE):     {non_moe_from_decoder:.1f} ms  (attention + norm + gaps)")
        # verify_forward breakdown
        verify_fwd = verify_ms
        non_model = max(0, verify_fwd - dl_per_call)
        print(f"  Verify forward:              {verify_fwd:.1f} ms")
        print(f"  ├─ Decoder layers:           {dl_per_call:.1f} ms  ({dl_per_call/max(verify_fwd,0.001)*100:.0f}%)")
        print(f"  │   ├─ MoE wall (sum):       {moe_wall_total:.1f} ms")
        print(f"  │   └─ Non-MoE (attn+norm):  {non_moe_from_decoder:.1f} ms")
        print(f"  └─ Non-model (emb+head+spec):{non_model:.1f} ms  ({non_model/max(verify_fwd,0.001)*100:.0f}%)")

    print(f"\n{'─' * 60}")
    print("  Per-layer aggregate:")
    print(f"    Pre-transfer cache hit rate: {pre_hit['avg']:.3f}  "
          f"(avg {agg['gpu_route_count']['avg']:.1f} GPU / {agg['gpu_route_count']['avg'] + agg['cpu_route_count']['avg']:.1f} total routes)")
    print(f"    Total unique experts: avg={agg['total_expert_count']['avg']:.1f}  "
          f"min={agg['total_expert_count']['min']:.0f}  max={agg['total_expert_count']['max']:.0f}")
    print(f"    GPU expert count:    avg={agg['gpu_expert_count']['avg']:.1f}  "
          f"min={agg['gpu_expert_count']['min']:.0f}  max={agg['gpu_expert_count']['max']:.0f}")
    print(f"    CPU expert count:    avg={agg['cpu_expert_count']['avg']:.1f}  "
          f"min={agg['cpu_expert_count']['min']:.0f}  max={agg['cpu_expert_count']['max']:.0f}")
    print(f"    CPU route count:     avg={agg['cpu_route_count']['avg']:.1f}  "
          f"min={agg['cpu_route_count']['min']:.0f}  max={agg['cpu_route_count']['max']:.0f}")
    print(f"    GPU route count:     avg={agg['gpu_route_count']['avg']:.1f}  "
          f"min={agg['gpu_route_count']['min']:.0f}  max={agg['gpu_route_count']['max']:.0f}")
    print(f"    --- Timing breakdown ---")
    print(f"    MoE wall (CPU):      avg={agg['layer_moe_wall_ms']['avg']:.3f} ms")
    print(f"    MoE GPU exec:        avg={agg['layer_moe_gpu_wall_ms']['avg']:.3f} ms  (CUDA events)")
    print(f"    Route+Plan (CPU):    avg={agg['plan_ms']['avg'] + agg.get('route_ms', {}).get('avg', 0):.3f} ms")
    print(f"    CPU compute:         avg={agg['cpu_compute_ms']['avg']:.3f} ms  (kt_direct, from profile)")
    print(f"    CPU prepare:         avg={agg['cpu_prepare_ms']['avg']:.3f} ms  (data xfer)")
    # kt_direct direct measurement
    if KT_DIRECT_FORWARD_EVENTS:
        kt_walls = [e["wall_ms"] for e in KT_DIRECT_FORWARD_EVENTS]
        kt_preps = [e["prep_ms"] for e in KT_DIRECT_FORWARD_EVENTS]
        kt_comps = [e["compute_ms"] for e in KT_DIRECT_FORWARD_EVENTS]
        kt_routes = [e["num_routes"] for e in KT_DIRECT_FORWARD_EVENTS if e["num_routes"] > 0]
        print(f"    --- kt_direct forward() DIRECT measurement ---")
        print(f"    kt_direct total wall: avg={mean(kt_walls):.4f} ms  ({len(kt_walls)} calls)")
        print(f"    kt_direct prep_ms:    avg={mean(kt_preps):.4f} ms")
        print(f"    kt_direct compute_ms: avg={mean(kt_comps):.4f} ms")
        if kt_routes:
            print(f"    kt_direct per-route:  avg={mean(kt_walls)/mean(kt_routes):.4f} ms  (wall/routes)")
            print(f"    kt_direct routes/call:avg={mean(kt_routes):.1f}")
    print(f"    GPU kernel launch:   avg={agg['gpu_gather_ms']['avg'] + agg['gpu_compute_ms']['avg'] + agg['scatter_ms']['avg']:.3f} ms")
    print(f"    CPU overhead:        avg={agg['cpu_overhead_ms']['avg']:.3f} ms  (meta, sync, gap)")
    print(f"    Per-route CPU ms:    avg={agg['per_route_cpu_ms']['avg']:.4f}  "
          f"median={agg['per_route_cpu_ms']['median']:.4f}")

    mw = max(float(budget.get("moe_wall_ms_sum", 0.0)), 0.001)
    print(f"\n{'─' * 60}")
    print("  Per-verify-forward budget (sum of per-layer avg):")
    print(f"    Route (CPU):        {budget.get('route_ms_sum', 0):8.3f} ms  ({budget.get('route_ms_sum', 0)/mw*100:5.1f}%)")
    print(f"    Plan (CPU):         {budget.get('plan_ms_sum', 0):8.3f} ms  ({budget.get('plan_ms_sum', 0)/mw*100:5.1f}%)")
    print(f"    CPU prepare:        {budget.get('cpu_prepare_ms_sum', 0):8.3f} ms  ({budget.get('cpu_prepare_ms_sum', 0)/mw*100:5.1f}%)")
    print(f"    CPU compute:        {budget.get('cpu_compute_ms_sum', 0):8.3f} ms  ({budget.get('cpu_compute_ms_sum', 0)/mw*100:5.1f}%)")
    print(f"    GPU exec (CUDA evt):{gpu_wall_sum:8.3f} ms  ({gpu_wall_sum/mw*100:5.1f}%)")
    print(f"    CPU overhead:       {cpu_overhead_sum:8.3f} ms  ({cpu_overhead_sum/mw*100:5.1f}%)")
    print(f"    {'─' * 40}")
    print(f"    Accounted:          {accounted_ms_sum:8.3f} ms  ({accounted_ms_sum/mw*100:5.1f}%)")
    print(f"    Remaining gap:      {unaccounted_ms_sum:8.3f} ms  ({unaccounted_ms_sum/mw*100:5.1f}%)")
    print(f"    MoE total (CPU wall){float(budget.get('moe_wall_ms_sum', 0)):8.3f} ms")

    # ── prefetch analysis ────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("PREFETCH ANALYSIS")
    print("=" * 72)

    pd = prefetch_analysis.get("per_step_deltas", {})
    total = prefetch_analysis.get("total_delta", {})

    if pd:
        print(f"\n  Snapshots: {prefetch_analysis['snapshot_count']}  (verify calls)")
        print(f"\n  Per-step prefetch deltas (averaged across {prefetch_analysis.get('delta_count', 0)} steps):")

        for k in sorted(pd):
            stats = pd[k]
            if stats["per_step_sum"] == 0:
                continue
            print(f"    {k:45s}  avg={stats['per_step_avg']:7.1f}  "
                  f"min={stats['per_step_min']:5.0f}  max={stats['per_step_max']:5.0f}  total={stats['per_step_sum']:7.0f}")

        print(f"\n  Cumulative over entire benchmark:")
        for k in sorted(total):
            if total[k] == 0:
                continue
            print(f"    {k:45s}  Δ={total[k]:7.0f}")

        # Prefetch effectiveness: cumulative delta from snapshots
        prefetch_delta = total.get("prefetch_consumed", 0)

        print(f"\n  Prefetch effectiveness assessment:")

        if prefetch_delta > 0:
            # How many CPU routes per verify? From layer aggregates
            cpu_routes_per_verify = budget.get("cpu_compute_ms_sum", 0)
            # Each prefetched expert eliminates routes to that expert
            # If prefetch consumed N experts but we still have high CPU routes,
            # it means either prefetch is too few or arrives too late
            total_cpu_layers = verify_analysis["total_verify_layer_events"]
            avg_valid_layers = max(1, total_cpu_layers - verify_calls)  # skip non-MoE
            print(f"    Prefetch consumed:    {prefetch_delta:.0f} experts total")
            print(f"    Prefetch per step:    {pd.get('prefetch_consumed', {}).get('per_step_avg', 0):.1f} experts/step")
            print(f"    CPU routes/layer:     {agg['cpu_route_count']['avg']:.1f} (avg)")
            print(f"    Cache hit rate:       {pre_hit['avg']:.3f} ({pre_hit['avg']*100:.1f}%)")
            print(f"    With {slots} pre-loaded slots + prefetch, only {pre_hit['avg']*100:.1f}% routes hit GPU cache.")

            if pre_hit["avg"] < effective_cache_ratio:
                print(f"    WARNING: Pre-transfer hit rate ({pre_hit['avg']*100:.1f}%) is below")
                print(f"    static cache ratio ({effective_cache_ratio*100:.1f}%). This means prefetch is")
                print(f"    NOT keeping up with routing demand — experts are evicted faster than prefetched.")
            if pre_hit["avg"] < 0.5:
                print(f"    ROOT CAUSE: With only {slots}/{num_experts} slots, static capacity is the")
                print(f"    bottleneck. Prefetch helps at the margin but cannot overcome the slot limit.")
        else:
            print(f"    ** Prefetch submitted/completed/consumed ZERO experts. **")
            print(f"    The cache hit rate ({pre_hit['avg']*100:.1f}%) comes entirely from the")
            print(f"    {slots} pre-loaded experts ({slots}/{num_experts}).")
            print(f"    With miss_policy=cpu, there is NO on-demand cache_fill during verify.")
            if not args.prefetch_enabled:
                print(f"    Check: prefetch_enabled={args.prefetch_enabled} -> prefetch is DISABLED.")
            print(f"    VERDICT: Prefetch is not filling the gap. Static cache capacity")
            print(f"    ({slots}/{num_experts}={effective_cache_ratio*100:.0f}%) is the binding constraint.")

    # ── per-layer detail ─────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  Per-layer (avg over calls for this layer):")
    print(f"  {'layer':>6s}  {'hit%':>6s}  {'tot_exp':>7s}  {'cpu_exp':>7s}  {'cpu_rt':>7s}  "
          f"{'wall':>7s}  {'gpu':>7s}  {'cpu_ms':>7s}  {'plan':>7s}  {'ovhd':>7s}")
    for li in sorted(verify_analysis["per_layer"]):
        pl = verify_analysis["per_layer"][li]
        hit_pct = pl["pre_transfer_hit_rate"]["avg"] * 100
        print(f"  {li:>6s}  {hit_pct:5.1f}%  {pl['total_expert_count']['avg']:>7.1f}  "
              f"{pl['cpu_expert_count']['avg']:>7.1f}  {pl['cpu_route_count']['avg']:>7.1f}  "
              f"{pl['layer_moe_wall_ms']['avg']:>7.3f}  {pl['layer_moe_gpu_wall_ms']['avg']:>7.3f}  "
              f"{pl['cpu_compute_ms']['avg']:>7.3f}  {pl['plan_ms']['avg']:>7.3f}  "
              f"{pl['cpu_overhead_ms']['avg']:>7.3f}")

    # ── bottleneck analysis ──────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("BOTTLENECK ANALYSIS")
    print("=" * 72)

    cpu_comp_pct = budget.get("cpu_compute_ms_sum", 0) / mw * 100
    gpu_exec_pct = gpu_wall_sum / mw * 100
    plan_route_pct = (budget.get("plan_ms_sum", 0) + budget.get("route_ms_sum", 0)) / mw * 100
    cpu_ovhd_pct = cpu_overhead_sum / mw * 100

    print(f"\n  Route+Plan (CPU):   {plan_route_pct:.1f}% of MoE wall")
    print(f"  GPU execution:      {gpu_exec_pct:.1f}% of MoE wall  (actual matmul/gather/scatter)")
    print(f"  CPU compute:        {cpu_comp_pct:.1f}% of MoE wall  (kt_direct)")
    print(f"  CPU overhead:       {cpu_ovhd_pct:.1f}% of MoE wall  (kernel launch, sync, gaps)")
    print(f"  Remaining gap:      {unaccounted_ms_sum/mw*100:.1f}% of MoE wall")

    if cpu_comp_pct > 50:
        print(f"\n  CPU compute is the PRIMARY bottleneck ({cpu_comp_pct:.1f}% of MoE wall).")
    elif cpu_comp_pct > 15:
        print(f"\n  CPU compute is significant ({cpu_comp_pct:.1f}% of MoE wall).")
    else:
        print(f"\n  CPU compute is NOT the bottleneck ({cpu_comp_pct:.1f}% of MoE wall).")

    if gpu_exec_pct > 50:
        print(f"  GPU execution is the PRIMARY bottleneck ({gpu_exec_pct:.1f}% of MoE wall).")
    elif plan_route_pct > 20:
        print(f"  Route+Plan overhead is significant ({plan_route_pct:.1f}% of MoE wall).")

    if pre_hit["avg"] < 0.3:
        print(f"\n  Cache is severely undersized ({pre_hit['avg']*100:.0f}% hit rate).")
        print(f"  Recommended: increase cache_ratio or decrease spec_verify_miss_policy threshold.")

    # ── compact summary line (like small_bench.py) ───────────────────────
    print(f"\n{'=' * 72}")
    print(f"  accept={acceptance_rate:.4f} true_hit={true_route_hit_rate:.4f} "
          f"miss/layer={avg_miss:.2f} active/layer={avg_active:.2f} "
          f"tok/s={throughput:.3f} draft_ms={draft_ms:.3f} "
          f"verify_ms={verify_ms:.3f} graph={draft_graph} "
          f"prefetch={pf_submit}/{pf_completed}/{pf_consumed} "
          f"kt_threads={args.kt_num_threads} backend={args.cpu_expert_backend}")

    # ── output ───────────────────────────────────────────────────────────
    out = {
        "config": {
            "model_path": args.model_path,
            "cache_ratio": effective_cache_ratio,
            "slots": slots,
            "num_experts": num_experts,
            "cpu_expert_backend": args.cpu_expert_backend,
            "spec_verify_miss_policy": args.spec_verify_miss_policy,
            "prefetch_enabled": prefetch_enabled,
            "prefetch_step_budget": args.prefetch_step_budget,
            "prefetch_max_inflight": args.prefetch_max_inflight,
            "output_len": args.output_len,
            "max_draft_tokens": args.max_draft_tokens,
            "verify_cuda_graph": args.verify_cuda_graph,
        },
        "summary": {
            "generated_tokens": generated_tokens,
            "elapsed_sec": elapsed,
            "throughput_tok_s": throughput,
            "verify_calls": verify_calls,
            "verify_forward_ms_avg": float(ep.get("verify_forward_ms", 0)),
            "draft_forward_ms_avg": float(ep.get("draft_forward_ms", 0)),
        },
        "verify_per_layer": verify_analysis,
        "draft_per_layer": draft_analysis,
        "prefetch": prefetch_analysis,
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {args.output}")

    del llm
    torch.cuda.empty_cache()
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Profile verify + kt_direct with prefetch analysis")
    p.add_argument("--model-path", default="/data1/models/Qwen3-30B-A3B")
    p.add_argument("--output", default=None)
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--cache-ratio", type=float, default=0.25)
    p.add_argument("--slots-per-layer", type=int, default=0)
    p.add_argument("--max-draft-tokens", type=int, default=8)
    p.add_argument("--draft-top-c", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-num-batched-tokens", type=int, default=16384)
    p.add_argument("--max-num-seqs", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--enforce-eager", type=lambda x: x.lower() == "true", default="false")

    p.add_argument("--cpu-expert-backend", default="kt_direct",
                   choices=["torch", "torch_packed", "fused", "kt_direct"])
    p.add_argument("--cpu-expert-pin-memory", type=lambda x: x.lower() == "true", default="true")
    p.add_argument("--cpu-expert-workspace-max-routes", type=int, default=327680)
    p.add_argument("--cpu-expert-num-threads", type=int, default=4)
    p.add_argument("--cpu-expert-strict-dtype", type=lambda x: x.lower() == "true", default="true")

    p.add_argument("--kt-num-threads", type=int, default=0)
    p.add_argument("--kt-threadpool-count", type=int, default=1)
    p.add_argument("--kt-chunked-prefill-size", type=int, default=4096)
    p.add_argument("--kt-direct-backend", default="auto",
                   choices=["auto", "amx_bf16", "avx2_bf16"])
    p.add_argument("--kt-numa-nodes", default="")
    p.add_argument("--kt-capture-bs", default="1,2,4,8,16,32")

    p.add_argument("--acceptance-strategy", default="standard_sampling")
    p.add_argument("--acceptance-threshold", type=float, default=0.7)
    p.add_argument("--spec-verify-miss-policy", default="cpu",
                   choices=["cpu", "cache_fill", "cache_fill_no_cpu"])
    p.add_argument("--cache-strategy", default="lru")
    p.add_argument("--draft-reroute-policy", default="entropy_cache_bias")
    p.add_argument("--draft-reroute-artifact",
                   default="results/reroute_impl_20260531/offline_profile_20260531_203257.safetensors")

    # prefetch
    p.add_argument("--prefetch-enabled", type=lambda x: x.lower() == "true", default="true")
    p.add_argument("--prefetch-step-budget", type=int, default=8)
    p.add_argument("--prefetch-max-inflight", type=int, default=16)
    p.add_argument("--prefetch-verify-layer-max-budget", type=int, default=8)
    p.add_argument("--prefetch-staging-slots-per-layer", type=int, default=2)
    p.add_argument("--prefetch-verify-attention-ratio", type=float, default=1.0)
    p.add_argument("--cache-eviction-budget-per-step", type=int, default=2)
    p.add_argument("--prefetch-global-queue-capacity", type=int, default=4096)
    p.add_argument("--prefetch-use-prefill-history", type=lambda x: x.lower() == "true", default="true")
    p.add_argument("--prefetch-use-verify-history", type=lambda x: x.lower() == "true", default="true")
    p.add_argument("--prefetch-use-draft-live", type=lambda x: x.lower() == "true", default="true")

    p.add_argument("--draft-cuda-graph-enabled", type=lambda x: x.lower() == "true", default="true")
    p.add_argument("--draft-cuda-graph-cpu-backend", default="none")
    p.add_argument("--verify-cuda-graph", type=lambda x: x.lower() == "true", default="false")
    p.add_argument("--verify-cuda-graph-bucket-steps", default="4,8,12,16")

    p.add_argument("--sync-layer-timing", type=lambda x: x.lower() == "true", default="true")
    p.add_argument("--dist-port", type=int, default=29500)

    return p


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
