from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import torch
import torch.nn.functional as F

from nanovllm.expert.cache import LayerExpertCache
from nanovllm.expert.placement import MoEExecutionPlan, build_moe_execution_plan
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.fuse_moe.cpu_backend import TorchPackedCpuMoeBackend, get_cpu_expert_weights
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
    cpu_gpu_parallel_execution_enabled: str = "auto",
    cpu_gpu_parallel_min_cpu_route_ratio: float = 0.0,
    cpu_gpu_parallel_stream: torch.cuda.Stream | None = None,
    cpu_backend: TorchPackedCpuMoeBackend | None = None,
    cpu_backend_min_routes: int = 32,
    profile: dict | None = None,
) -> torch.Tensor:
    """Run MoE with GPU cached experts + fallback path for uncached experts."""
    top_k = routing_weights.size(1)
    flat_selected = selected_experts.reshape(-1)
    flat_weights = routing_weights.reshape(-1)

    output = torch.zeros_like(hidden_states)
    if plan is None:
        plan = build_moe_execution_plan(selected_experts, expert_cache)

    has_gpu_work = plan.gpu_route_indices.numel() > 0 and plan.gpu_m_sizes is not None
    cpu_indices = plan.cpu_route_indices
    has_cpu_work = cpu_indices is not None and cpu_indices.numel() > 0
    cpu_route_ratio = (float(cpu_indices.numel()) / float(flat_selected.numel())) if has_cpu_work else 0.0
    active_cpu_backend = cpu_backend if has_cpu_work and int(cpu_indices.numel()) >= int(cpu_backend_min_routes) else None

    if profile is not None:
        profile["cpu_route_ratio_sum"] = float(cpu_route_ratio)
        if has_cpu_work:
            cpu_weight_mass = float(flat_weights.index_select(0, cpu_indices).sum().item())
            total_weight_mass = float(flat_weights.sum().item())
            profile["cpu_weight_mass_ratio_sum"] = (
                cpu_weight_mass / total_weight_mass if total_weight_mass > 0 else 0.0
            )
        else:
            profile["cpu_weight_mass_ratio_sum"] = 0.0
        if plan.cpu_task_expert_ids is not None:
            profile["realized_cpu_expert_count_sum"] = float(plan.cpu_task_expert_ids.numel())
        else:
            profile["realized_cpu_expert_count_sum"] = 0.0
        if has_cpu_work and cpu_backend is not None and active_cpu_backend is None:
            profile["cpu_backend_fallback_count"] = float(profile.get("cpu_backend_fallback_count", 0.0) + 1.0)

    if has_cpu_work and cpu_expert_pool is None and active_cpu_backend is None:
        raise RuntimeError("Missing cpu_expert_pool for uncached expert fallback.")

    parallel_mode = cpu_gpu_parallel_execution_enabled
    if parallel_mode == "auto":
        can_overlap_cpu_gpu = (
            has_gpu_work
            and has_cpu_work
            and hidden_states.is_cuda
            and cpu_expert_execution_enabled
        )
    elif parallel_mode == "on":
        can_overlap_cpu_gpu = (
            has_gpu_work
            and has_cpu_work
            and hidden_states.is_cuda
            and cpu_expert_execution_enabled
            and cpu_route_ratio >= float(cpu_gpu_parallel_min_cpu_route_ratio)
        )
    else:
        can_overlap_cpu_gpu = False

    if can_overlap_cpu_gpu:
        t_parallel0 = perf_counter()
        if cpu_gpu_parallel_stream is not None:
            gpu_stream = cpu_gpu_parallel_stream
        else:
            gpu_stream = torch.cuda.Stream(device=hidden_states.device)
        current_stream = torch.cuda.current_stream(device=hidden_states.device)
        gpu_stream.wait_stream(current_stream)
        with torch.cuda.stream(gpu_stream):
            gpu_token_indices, gpu_expert_out, gpu_gather_ms, gpu_compute_ms = _run_gpu_cached_expert_path(
                hidden_states=hidden_states,
                flat_weights=flat_weights,
                top_k=top_k,
                gpu_route_indices=plan.gpu_route_indices,
                gpu_m_sizes=plan.gpu_m_sizes,
                expert_cache=expert_cache,
                act_fn=act_fn,
            )

        if active_cpu_backend is not None:
            cpu_result = active_cpu_backend.forward(
                hidden_states=hidden_states,
                flat_weights=flat_weights,
                top_k=top_k,
                cpu_indices=cpu_indices,
                cpu_task_expert_ids=plan.cpu_task_expert_ids,
                cpu_task_offsets=plan.cpu_task_offsets,
                act_fn=act_fn,
                parallel_mode=cpu_expert_parallel_mode,
                num_threads=cpu_expert_num_threads,
                cpu_task_expert_ids_host=plan.cpu_task_expert_ids_host,
                cpu_task_offsets_host=plan.cpu_task_offsets_host,
            )
            cpu_token_indices = cpu_result.token_indices
            cpu_outputs = cpu_result.outputs_cpu
            prep_ms = cpu_result.prep_ms
            compute_ms = cpu_result.compute_ms
        else:
            cpu_token_indices, cpu_outputs, prep_ms, compute_ms = _compute_real_cpu_expert_outputs(
                hidden_states=hidden_states,
                flat_weights=flat_weights,
                top_k=top_k,
                cpu_indices=cpu_indices,
                cpu_task_expert_ids=plan.cpu_task_expert_ids,
                cpu_task_offsets=plan.cpu_task_offsets,
                flat_selected_original=plan.flat_selected_original,
                cpu_expert_pool=cpu_expert_pool,
                act_fn=act_fn,
                cpu_expert_parallel_mode=cpu_expert_parallel_mode,
                cpu_expert_num_threads=cpu_expert_num_threads,
            )

        torch.cuda.current_stream(device=hidden_states.device).wait_stream(gpu_stream)

        t_scatter0 = perf_counter()
        output.index_add_(0, gpu_token_indices, gpu_expert_out)
        scatter_ms = (perf_counter() - t_scatter0) * 1000.0

        if active_cpu_backend is not None:
            merge_ms = _merge_packed_cpu_outputs(
                output=output,
                token_indices=cpu_token_indices,
                outputs_cpu=cpu_outputs,
                output_dtype=hidden_states.dtype,
                output_device=hidden_states.device,
            )
        else:
            merge_ms = _merge_real_cpu_outputs(
                output=output,
                token_indices=cpu_token_indices,
                cpu_outputs=cpu_outputs,
                output_dtype=hidden_states.dtype,
                output_device=hidden_states.device,
            )

        gpu_core_ms = gpu_gather_ms + gpu_compute_ms
        cpu_core_ms = prep_ms + compute_ms
        overlap_est_ms = min(gpu_core_ms, cpu_core_ms)
        critical_path_est_ms = max(gpu_core_ms, cpu_core_ms) + scatter_ms + merge_ms
        parallel_wall_ms = (perf_counter() - t_parallel0) * 1000.0

        _prof_add(profile, "gpu_gather_ms", gpu_gather_ms / 1000.0)
        _prof_add(profile, "gpu_compute_ms", gpu_compute_ms / 1000.0)
        _prof_add(profile, "scatter_ms", scatter_ms / 1000.0)
        _prof_add(profile, "cpu_prepare_ms", prep_ms / 1000.0)
        _prof_add(profile, "cpu_compute_ms", compute_ms / 1000.0)
        _prof_add(profile, "cpu_to_gpu_merge_ms", merge_ms / 1000.0)
        _prof_add(profile, "parallel_overlap_est_ms", overlap_est_ms / 1000.0)
        _prof_add(profile, "parallel_critical_path_est_ms", critical_path_est_ms / 1000.0)
        _prof_add(profile, "parallel_wall_ms", parallel_wall_ms / 1000.0)
        _prof_add(profile, "cpu_wait_ms", max(0.0, gpu_core_ms - cpu_core_ms) / 1000.0)
        _prof_add(profile, "gpu_wait_ms", max(0.0, cpu_core_ms - gpu_core_ms) / 1000.0)
        if profile is not None:
            profile["parallel_enabled_count"] = float(profile.get("parallel_enabled_count", 0.0) + 1.0)
        return output

    # GPU path: cached experts are remapped to contiguous slot buffers.
    if has_gpu_work:
        gpu_token_indices, gpu_expert_out, gpu_gather_ms, gpu_compute_ms = _run_gpu_cached_expert_path(
            hidden_states=hidden_states,
            flat_weights=flat_weights,
            top_k=top_k,
            gpu_route_indices=plan.gpu_route_indices,
            gpu_m_sizes=plan.gpu_m_sizes,
            expert_cache=expert_cache,
            act_fn=act_fn,
        )
        _prof_add(profile, "gpu_gather_ms", gpu_gather_ms / 1000.0)
        _prof_add(profile, "gpu_compute_ms", gpu_compute_ms / 1000.0)

        t_scatter0 = perf_counter()
        if not has_cpu_work:
            _accumulate_gpu_routes_deterministic(
                output=output,
                gpu_route_indices=plan.gpu_route_indices,
                gpu_expert_out=gpu_expert_out,
                top_k=top_k,
            )
        else:
            output.index_add_(0, gpu_token_indices, gpu_expert_out)
        _prof_add(profile, "scatter_ms", perf_counter() - t_scatter0)

    # CPU path for uncached experts.
    if has_cpu_work:
        if cpu_expert_execution_enabled and active_cpu_backend is not None:
            prep_ms, compute_ms, merge_ms = _run_packed_cpu_expert_execution(
                hidden_states=hidden_states,
                output=output,
                flat_weights=flat_weights,
                top_k=top_k,
                cpu_indices=cpu_indices,
                cpu_task_expert_ids=plan.cpu_task_expert_ids,
                cpu_task_offsets=plan.cpu_task_offsets,
                cpu_backend=active_cpu_backend,
                act_fn=act_fn,
                cpu_expert_parallel_mode=cpu_expert_parallel_mode,
                cpu_expert_num_threads=cpu_expert_num_threads,
                cpu_task_expert_ids_host=plan.cpu_task_expert_ids_host,
                cpu_task_offsets_host=plan.cpu_task_offsets_host,
            )
            _prof_add(profile, "cpu_prepare_ms", prep_ms / 1000.0)
            _prof_add(profile, "cpu_compute_ms", compute_ms / 1000.0)
            _prof_add(profile, "cpu_to_gpu_merge_ms", merge_ms / 1000.0)
        elif cpu_expert_execution_enabled:
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
                cpu_expert_parallel_mode=cpu_expert_parallel_mode,
                cpu_expert_num_threads=cpu_expert_num_threads,
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


def _accumulate_gpu_routes_deterministic(
    output: torch.Tensor,
    gpu_route_indices: torch.Tensor,
    gpu_expert_out: torch.Tensor,
    top_k: int,
) -> None:
    """Accumulate route outputs with fixed token/top-k reduction order."""
    if gpu_route_indices.numel() == 0:
        return

    num_tokens, hidden_dim = output.shape
    num_routes = num_tokens * top_k
    route_buffer = torch.zeros((num_routes, hidden_dim), dtype=gpu_expert_out.dtype, device=output.device)
    route_buffer.index_copy_(0, gpu_route_indices.to(torch.int64), gpu_expert_out)
    token_output = route_buffer.view(num_tokens, top_k, hidden_dim).sum(dim=1)
    output.add_(token_output.to(dtype=output.dtype))


def _run_gpu_cached_expert_path(
    hidden_states: torch.Tensor,
    flat_weights: torch.Tensor,
    top_k: int,
    gpu_route_indices: torch.Tensor,
    gpu_m_sizes: torch.Tensor,
    expert_cache: LayerExpertCache,
    act_fn: SiluAndMul,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    t_gather0 = perf_counter()
    gpu_token_indices = torch.div(gpu_route_indices, top_k, rounding_mode="floor")
    gpu_hidden = hidden_states[gpu_token_indices]
    gpu_weights = flat_weights.index_select(0, gpu_route_indices)
    gather_ms = (perf_counter() - t_gather0) * 1000.0

    gate_up_buffer, down_buffer = expert_cache.get_layer_buffers()
    t_comp0 = perf_counter()
    gate_up = fused_moe_linear(gpu_hidden, gate_up_buffer, gpu_m_sizes)
    gpu_expert_out = fused_moe_linear(act_fn(gate_up), down_buffer, gpu_m_sizes)
    gpu_expert_out.mul_(gpu_weights.unsqueeze(-1))
    compute_ms = (perf_counter() - t_comp0) * 1000.0
    return gpu_token_indices, gpu_expert_out, gather_ms, compute_ms


def _compute_real_cpu_expert_outputs(
    hidden_states: torch.Tensor,
    flat_weights: torch.Tensor,
    top_k: int,
    cpu_indices: torch.Tensor,
    cpu_task_expert_ids: torch.Tensor | None,
    cpu_task_offsets: torch.Tensor | None,
    flat_selected_original: torch.Tensor,
    cpu_expert_pool: dict[int, dict[str, torch.Tensor]],
    act_fn: SiluAndMul,
    cpu_expert_parallel_mode: str,
    cpu_expert_num_threads: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], float, float]:
    prep_t0 = perf_counter()
    if cpu_task_expert_ids is None or cpu_task_offsets is None:
        cpu_experts = flat_selected_original.index_select(0, cpu_indices)
        route_order = torch.argsort(cpu_indices, stable=True)
        experts_by_route = cpu_experts.index_select(0, route_order)
        expert_order = torch.argsort(experts_by_route, stable=True)
        final_order = route_order.index_select(0, expert_order)
        sorted_experts = cpu_experts.index_select(0, final_order)
        cpu_indices = cpu_indices.index_select(0, final_order)
        cpu_task_expert_ids, counts = torch.unique_consecutive(sorted_experts, return_counts=True)
        cpu_task_offsets = torch.zeros(
            cpu_task_expert_ids.numel() + 1,
            dtype=torch.int64,
            device=cpu_indices.device,
        )
        cpu_task_offsets[1:] = torch.cumsum(counts.to(torch.int64), dim=0)

    # Match model activation dtype to reduce CPU/GPU numerical drift.
    compute_dtype = hidden_states.dtype
    token_indices = torch.div(cpu_indices, top_k, rounding_mode="floor")
    hidden_cpu = hidden_states.index_select(0, token_indices).to("cpu", dtype=compute_dtype, non_blocking=False)
    weights_cpu = flat_weights.index_select(0, cpu_indices).to("cpu", dtype=compute_dtype, non_blocking=False)
    prep_ms = (perf_counter() - prep_t0) * 1000.0

    num_tasks = int(cpu_task_expert_ids.numel())
    task_offsets_host = [int(x) for x in cpu_task_offsets.detach().to("cpu", non_blocking=False).tolist()]
    task_expert_ids_host = [int(x) for x in cpu_task_expert_ids.detach().to("cpu", non_blocking=False).tolist()]
    token_indices_chunks: list[torch.Tensor] = []
    cpu_outputs: list[torch.Tensor] = [torch.empty(0)] * num_tasks

    def run_task(task_idx: int) -> torch.Tensor:
        start = task_offsets_host[task_idx]
        end = task_offsets_host[task_idx + 1]
        if start >= end:
            return torch.empty((0, hidden_states.shape[-1]), dtype=compute_dtype, device="cpu")

        expert_idx = task_expert_ids_host[task_idx]
        params = cpu_expert_pool.get(expert_idx)
        if params is None:
            raise RuntimeError(f"Missing CPU expert weights for expert {expert_idx}")

        hidden_chunk = hidden_cpu[start:end]
        weight_chunk = weights_cpu[start:end]
        gate_up_weight, down_weight = get_cpu_expert_weights(
            cpu_expert_pool,
            expert_idx,
            compute_dtype,
            strict_packed_dtype=False,
        )
        gate_up = F.linear(hidden_chunk, gate_up_weight)
        cpu_out = F.linear(act_fn(gate_up), down_weight)
        cpu_out.mul_(weight_chunk.unsqueeze(-1))
        return cpu_out

    for i in range(num_tasks):
        start = task_offsets_host[i]
        end = task_offsets_host[i + 1]
        token_indices_chunks.append(token_indices[start:end])

    compute_t0 = perf_counter()
    can_parallel = cpu_expert_parallel_mode == "expert_parallel" and cpu_expert_num_threads > 1 and num_tasks > 1
    if can_parallel:
        max_workers = min(cpu_expert_num_threads, num_tasks)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(run_task, i) for i in range(num_tasks)]
            for i, fut in enumerate(futures):
                cpu_outputs[i] = fut.result()
    else:
        for i in range(num_tasks):
            cpu_outputs[i] = run_task(i)
    compute_ms = (perf_counter() - compute_t0) * 1000.0

    return token_indices_chunks, cpu_outputs, prep_ms, compute_ms


def _merge_real_cpu_outputs(
    output: torch.Tensor,
    token_indices: list[torch.Tensor],
    cpu_outputs: list[torch.Tensor],
    output_dtype: torch.dtype,
    output_device: torch.device,
) -> float:
    merge_t0 = perf_counter()
    non_empty_tokens: list[torch.Tensor] = []
    non_empty_outputs: list[torch.Tensor] = []
    for tokens, cpu_out in zip(token_indices, cpu_outputs):
        if tokens.numel() > 0:
            non_empty_tokens.append(tokens)
            non_empty_outputs.append(cpu_out)

    if non_empty_outputs:
        packed_tokens = torch.cat(non_empty_tokens, dim=0).to(device=output_device, dtype=torch.int64)
        if len(non_empty_outputs) == 1:
            packed_cpu = non_empty_outputs[0]
        else:
            packed_cpu = torch.cat(non_empty_outputs, dim=0)
        out = packed_cpu.to(device=output_device, dtype=output_dtype, non_blocking=False)
        output.index_add_(0, packed_tokens, out)
    return (perf_counter() - merge_t0) * 1000.0


def _merge_packed_cpu_outputs(
    output: torch.Tensor,
    token_indices: torch.Tensor,
    outputs_cpu: torch.Tensor,
    output_dtype: torch.dtype,
    output_device: torch.device,
) -> float:
    merge_t0 = perf_counter()
    if token_indices.numel() > 0:
        non_blocking = bool(output_device.type == "cuda" and outputs_cpu.is_pinned())
        out = outputs_cpu.to(device=output_device, dtype=output_dtype, non_blocking=non_blocking)
        output.index_add_(0, token_indices, out)
    return (perf_counter() - merge_t0) * 1000.0


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
    cpu_expert_parallel_mode: str = "serial",
    cpu_expert_num_threads: int = 4,
) -> tuple[float, float, float]:
    token_indices, cpu_outputs, prep_ms, compute_ms = _compute_real_cpu_expert_outputs(
        hidden_states=hidden_states,
        flat_weights=flat_weights,
        top_k=top_k,
        cpu_indices=cpu_indices,
        cpu_task_expert_ids=cpu_task_expert_ids,
        cpu_task_offsets=cpu_task_offsets,
        flat_selected_original=flat_selected_original,
        cpu_expert_pool=cpu_expert_pool,
        act_fn=act_fn,
        cpu_expert_parallel_mode=cpu_expert_parallel_mode,
        cpu_expert_num_threads=cpu_expert_num_threads,
    )
    merge_ms = _merge_real_cpu_outputs(
        output=output,
        token_indices=token_indices,
        cpu_outputs=cpu_outputs,
        output_dtype=hidden_states.dtype,
        output_device=hidden_states.device,
    )
    return prep_ms, compute_ms, merge_ms


def _run_packed_cpu_expert_execution(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    flat_weights: torch.Tensor,
    top_k: int,
    cpu_indices: torch.Tensor,
    cpu_task_expert_ids: torch.Tensor,
    cpu_task_offsets: torch.Tensor,
    cpu_backend: TorchPackedCpuMoeBackend,
    act_fn: SiluAndMul,
    cpu_expert_parallel_mode: str = "serial",
    cpu_expert_num_threads: int = 4,
    cpu_task_expert_ids_host: list[int] | None = None,
    cpu_task_offsets_host: list[int] | None = None,
) -> tuple[float, float, float]:
    result = cpu_backend.forward(
        hidden_states=hidden_states,
        flat_weights=flat_weights,
        top_k=top_k,
        cpu_indices=cpu_indices,
        cpu_task_expert_ids=cpu_task_expert_ids,
        cpu_task_offsets=cpu_task_offsets,
        act_fn=act_fn,
        parallel_mode=cpu_expert_parallel_mode,
        num_threads=cpu_expert_num_threads,
        cpu_task_expert_ids_host=cpu_task_expert_ids_host,
        cpu_task_offsets_host=cpu_task_offsets_host,
    )
    merge_ms = _merge_packed_cpu_outputs(
        output=output,
        token_indices=result.token_indices,
        outputs_cpu=result.outputs_cpu,
        output_dtype=hidden_states.dtype,
        output_device=hidden_states.device,
    )
    return result.prep_ms, result.compute_ms, merge_ms


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
