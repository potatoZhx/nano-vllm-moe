from __future__ import annotations

from time import perf_counter

import torch
import torch.nn.functional as F

from nanovllm.expert.cache import LayerExpertCache
from nanovllm.expert.placement import MoEExecutionPlan, build_moe_execution_plan
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.fuse_moe.functional import fused_moe_linear


def heterogeneous_moe_forward(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_cache: LayerExpertCache,
    cpu_expert_pool: dict[int, dict[str, torch.Tensor]] | None,
    act_fn: SiluAndMul,
    plan: MoEExecutionPlan | None = None,
    cpu_expert_execution_enabled: bool = False,
    cpu_expert_parallel_mode: str = "serial",
    cpu_expert_num_threads: int = 4,
    profile: dict | None = None,
) -> torch.Tensor:
    """Run MoE with GPU cached experts + fallback path for uncached experts."""
    _ = cpu_expert_num_threads
    top_k = routing_weights.size(1)
    flat_selected = selected_experts.reshape(-1)
    flat_weights = routing_weights.reshape(-1)

    output = torch.zeros_like(hidden_states)
    if plan is None:
        plan = build_moe_execution_plan(selected_experts, expert_cache)

    # GPU path: cached experts are remapped to contiguous slot buffers.
    if plan.gpu_route_indices.numel() > 0 and plan.gpu_m_sizes is not None:
        t_gather0 = perf_counter()
        gpu_token_indices = torch.div(plan.gpu_route_indices, top_k, rounding_mode="floor")
        gpu_hidden = hidden_states[gpu_token_indices]
        gpu_weights = flat_weights.index_select(0, plan.gpu_route_indices)
        _prof_add(profile, "gpu_gather_ms", perf_counter() - t_gather0)

        gate_up_buffer, down_buffer = expert_cache.get_layer_buffers()

        t_comp0 = perf_counter()
        gate_up = fused_moe_linear(gpu_hidden, gate_up_buffer, plan.gpu_m_sizes)
        gpu_expert_out = fused_moe_linear(act_fn(gate_up), down_buffer, plan.gpu_m_sizes)
        gpu_expert_out.mul_(gpu_weights.unsqueeze(-1))
        _prof_add(profile, "gpu_compute_ms", perf_counter() - t_comp0)

        t_scatter0 = perf_counter()
        output.index_add_(0, gpu_token_indices, gpu_expert_out)
        _prof_add(profile, "scatter_ms", perf_counter() - t_scatter0)

    # CPU path for uncached experts.
    # TODO: 清理/统一实现
    cpu_indices = plan.cpu_route_indices
    if cpu_indices is not None and cpu_indices.numel() > 0:
        if cpu_expert_pool is None:
            raise RuntimeError("Missing cpu_expert_pool for uncached expert fallback.")

        if cpu_expert_execution_enabled:
            if cpu_expert_parallel_mode == "serial":
                prep_ms, compute_ms, merge_ms = _run_real_cpu_expert_execution(
                    hidden_states=hidden_states,
                    output=output,
                    flat_weights=flat_weights,
                    top_k=top_k,
                    cpu_indices=cpu_indices,
                    cpu_task_expert_ids=plan.cpu_task_expert_ids,
                    cpu_task_offsets=plan.cpu_task_offsets,
                    flat_selected_original=plan.flat_selected_original,
                    cpu_expert_pool=cpu_expert_pool,
                    act_fn=act_fn,
                )
            else:
                # First implementation keeps expert-level serial execution for stability.
                prep_ms, compute_ms, merge_ms = _run_real_cpu_expert_execution(
                    hidden_states=hidden_states,
                    output=output,
                    flat_weights=flat_weights,
                    top_k=top_k,
                    cpu_indices=cpu_indices,
                    cpu_task_expert_ids=plan.cpu_task_expert_ids,
                    cpu_task_offsets=plan.cpu_task_offsets,
                    flat_selected_original=plan.flat_selected_original,
                    cpu_expert_pool=cpu_expert_pool,
                    act_fn=act_fn,
                )
            _prof_add(profile, "cpu_prepare_ms", prep_ms / 1000.0)
            _prof_add(profile, "cpu_compute_ms", compute_ms / 1000.0)
            _prof_add(profile, "cpu_to_gpu_merge_ms", merge_ms / 1000.0)
        else:
            _run_legacy_gpu_fallback(
                hidden_states=hidden_states,
                output=output,
                flat_weights=flat_weights,
                top_k=top_k,
                cpu_indices=cpu_indices,
                flat_selected_original=plan.flat_selected_original,
                cpu_expert_pool=cpu_expert_pool,
                act_fn=act_fn,
            )

    return output


def _run_real_cpu_expert_execution(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    flat_weights: torch.Tensor,
    top_k: int,
    cpu_indices: torch.Tensor,
    cpu_task_expert_ids: torch.Tensor | None,
    cpu_task_offsets: torch.Tensor | None,
    flat_selected_original: torch.Tensor,
    cpu_expert_pool: dict[int, dict[str, torch.Tensor]],
    act_fn: SiluAndMul,
) -> tuple[float, float, float]:
    prep_ms = 0.0
    compute_ms = 0.0
    merge_ms = 0.0
    if cpu_task_expert_ids is None or cpu_task_offsets is None:
        cpu_experts = flat_selected_original.index_select(0, cpu_indices)
        sorted_experts, sort_idx = torch.sort(cpu_experts)
        cpu_indices = cpu_indices.index_select(0, sort_idx)
        cpu_task_expert_ids, counts = torch.unique_consecutive(sorted_experts, return_counts=True)
        cpu_task_offsets = torch.zeros(
            cpu_task_expert_ids.numel() + 1,
            dtype=torch.int64,
            device=cpu_indices.device,
        )
        cpu_task_offsets[1:] = torch.cumsum(counts.to(torch.int64), dim=0)

    for i in range(int(cpu_task_expert_ids.numel())):
        start = int(cpu_task_offsets[i].item())
        end = int(cpu_task_offsets[i + 1].item())
        if start >= end:
            continue

        expert_idx = int(cpu_task_expert_ids[i].item())
        params = cpu_expert_pool.get(expert_idx)
        if params is None:
            raise RuntimeError(f"Missing CPU expert weights for expert {expert_idx}")

        route_slice = cpu_indices[start:end]
        token_indices = torch.div(route_slice, top_k, rounding_mode="floor")

        # Keep compute in float32 on CPU for portability and stable numerics.
        t0 = perf_counter()
        hidden_cpu = hidden_states.index_select(0, token_indices).to("cpu", dtype=torch.float32, non_blocking=True)
        weights_cpu = flat_weights.index_select(0, route_slice).to("cpu", dtype=torch.float32, non_blocking=True)
        gate_up_weight = params["gate_up"].to(dtype=torch.float32)
        down_weight = params["down"].to(dtype=torch.float32)
        prep_ms += (perf_counter() - t0) * 1000.0

        t0 = perf_counter()
        gate_up = F.linear(hidden_cpu, gate_up_weight)
        cpu_out = F.linear(act_fn(gate_up), down_weight)
        cpu_out.mul_(weights_cpu.unsqueeze(-1))
        compute_ms += (perf_counter() - t0) * 1000.0

        t0 = perf_counter()
        out = cpu_out.to(device=hidden_states.device, dtype=hidden_states.dtype, non_blocking=True)
        output.index_add_(0, token_indices, out)
        merge_ms += (perf_counter() - t0) * 1000.0

    return prep_ms, compute_ms, merge_ms


def _run_legacy_gpu_fallback(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    flat_weights: torch.Tensor,
    top_k: int,
    cpu_indices: torch.Tensor,
    flat_selected_original: torch.Tensor,
    cpu_expert_pool: dict[int, dict[str, torch.Tensor]],
    act_fn: SiluAndMul,
) -> None:
    cpu_token_indices = torch.div(cpu_indices, top_k, rounding_mode="floor")
    cpu_hidden = hidden_states[cpu_token_indices]
    cpu_experts = flat_selected_original.index_select(0, cpu_indices)
    cpu_weights = flat_weights.index_select(0, cpu_indices)

    for expert_idx in cpu_experts.unique().tolist():
        expert_mask = cpu_experts == expert_idx
        h = cpu_hidden[expert_mask]
        params = cpu_expert_pool[expert_idx]
        gate_up_weight = params["gate_up"].to(device=hidden_states.device, dtype=hidden_states.dtype, non_blocking=True)
        down_weight = params["down"].to(device=hidden_states.device, dtype=hidden_states.dtype, non_blocking=True)
        gate_up = F.linear(h, gate_up_weight)
        out = F.linear(act_fn(gate_up), down_weight)
        out = out * cpu_weights[expert_mask].unsqueeze(-1)
        output.index_add_(0, cpu_token_indices[expert_mask], out)


def _prof_add(profile: dict | None, key: str, dt_sec: float) -> None:
    if profile is None:
        return
    profile[key] = float(profile.get(key, 0.0) + dt_sec * 1000.0)
