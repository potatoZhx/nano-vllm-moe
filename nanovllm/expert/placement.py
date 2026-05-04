from __future__ import annotations

from dataclasses import dataclass

import torch

from nanovllm.expert.cache import LayerExpertCache
from nanovllm.scheduling.draft_scheduler import DraftScheduler


@dataclass
class MoEExecutionPlan:
    layer_idx: int
    gpu_route_indices: torch.Tensor
    gpu_m_sizes: torch.Tensor | None
    cpu_route_indices: torch.Tensor | None
    cpu_task_expert_ids: torch.Tensor | None
    cpu_task_offsets: torch.Tensor | None
    cpu_task_expert_ids_host: list[int] | None = None
    cpu_task_offsets_host: list[int] | None = None
    flat_selected_original: torch.Tensor | None = None
    flat_selected_effective: torch.Tensor | None = None
    substitution_lut: torch.Tensor | None = None
    gpu_route_mask: torch.Tensor | None = None
    cpu_route_mask: torch.Tensor | None = None

    @property
    def m_sizes(self) -> torch.Tensor | None:
        # Backward-compatible alias used by existing call sites/tests.
        return self.gpu_m_sizes

    @property
    def substitution_map(self) -> dict[int, int]:
        # Kept only for compatibility/debug access. Hot path should use substitution_lut.
        if self.substitution_lut is None:
            return {}
        lut_cpu = self.substitution_lut.detach().to("cpu")
        idx = torch.nonzero(lut_cpu != torch.arange(lut_cpu.numel(), dtype=lut_cpu.dtype), as_tuple=False).flatten()
        out: dict[int, int] = {}
        for i in idx.tolist():
            out[int(i)] = int(lut_cpu[i].item())
        return out


def _build_grouped_layout(
    gpu_slots: torch.Tensor,
    gpu_route_indices: torch.Tensor,
    num_slots: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Group routes by slot id with deterministic tie-break on original route index.
    # This keeps intra-slot order stable across runs and matches route-major semantics.
    route_order = torch.argsort(gpu_route_indices, stable=True)
    slots_by_route = gpu_slots.index_select(0, route_order)
    slot_order = torch.argsort(slots_by_route, stable=True)
    final_order = route_order.index_select(0, slot_order)
    sorted_slots = gpu_slots.index_select(0, final_order)
    sorted_gpu_route_indices = gpu_route_indices.index_select(0, final_order)
    # Use fixed-shape scatter_add to keep graph capture friendly on some runtimes.
    m_sizes = torch.zeros(num_slots, dtype=torch.int32, device=sorted_slots.device)
    ones = torch.ones_like(sorted_slots, dtype=torch.int32)
    m_sizes.scatter_add_(0, sorted_slots.to(torch.int64), ones)
    return m_sizes, sorted_gpu_route_indices


def _build_topc0_substitution_lut(
    num_experts: int,
    cached_expert_mask: torch.Tensor,
    slot_to_expert_lut: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Build a deterministic substitution table for draft top_c=0.

    Strategy:
    - Cached experts keep identity mapping.
    - Uncached experts map to cached experts via round-robin over cache slots.
    - This produces fixed-shape tensors and avoids host-side branching in hot path.
    """
    expert_ids = torch.arange(num_experts, dtype=torch.int64, device=device)
    if slot_to_expert_lut.numel() == 0:
        return expert_ids

    rr_slot = torch.remainder(expert_ids, int(slot_to_expert_lut.numel()))
    fallback_experts = slot_to_expert_lut.index_select(0, rr_slot)
    # Defensive fallback: if a slot is not populated, keep identity for that id.
    fallback_experts = torch.where(fallback_experts >= 0, fallback_experts, expert_ids)
    return torch.where(cached_expert_mask, expert_ids, fallback_experts.to(torch.int64))


def _build_cpu_task_layout(
    flat_selected_original: torch.Tensor,
    cpu_route_indices: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, list[int] | None, list[int] | None]:
    if cpu_route_indices.numel() == 0:
        return None, None, None, None, None

    cpu_experts = flat_selected_original.index_select(0, cpu_route_indices)
    route_order = torch.argsort(cpu_route_indices, stable=True)
    experts_by_route = cpu_experts.index_select(0, route_order)
    expert_order = torch.argsort(experts_by_route, stable=True)
    final_order = route_order.index_select(0, expert_order)
    sorted_experts = cpu_experts.index_select(0, final_order)
    sorted_route_indices = cpu_route_indices.index_select(0, final_order)
    task_expert_ids, counts = torch.unique_consecutive(sorted_experts, return_counts=True)
    task_offsets = torch.zeros(
        task_expert_ids.numel() + 1,
        dtype=torch.int64,
        device=cpu_route_indices.device,
    )
    task_offsets[1:] = torch.cumsum(counts.to(torch.int64), dim=0)
    ids_cpu = task_expert_ids.to(torch.int64)
    # Pre-extract host metadata to avoid GPU->CPU transfer in hot path.
    ids_host = [int(x) for x in ids_cpu.detach().to("cpu", non_blocking=False).tolist()]
    offsets_host = [int(x) for x in task_offsets.detach().to("cpu", non_blocking=False).tolist()]
    return sorted_route_indices, ids_cpu, task_offsets, ids_host, offsets_host


def _flatten_experts(selected_experts: torch.Tensor) -> torch.Tensor:
    return selected_experts.reshape(-1).to(torch.int64)


def _flatten_weights(routing_weights: torch.Tensor) -> torch.Tensor:
    return routing_weights.reshape(-1).float()


def flatten_selected_and_weights(
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _flatten_experts(selected_experts), _flatten_weights(routing_weights)


def build_runtime_meta_view(
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return selected_experts, routing_weights


def build_prefill_plan(
    layer_idx: int,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_cache: LayerExpertCache,
    num_experts: int,
) -> MoEExecutionPlan:
    return build_prefill_plan_gpu(
        layer_idx=layer_idx,
        selected_experts=selected_experts,
        routing_weights=routing_weights,
        expert_cache=expert_cache,
        num_experts=num_experts,
    )


def build_prefill_plan_gpu(
    layer_idx: int,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_cache: LayerExpertCache,
    num_experts: int,
) -> MoEExecutionPlan:
    flat_selected = _flatten_experts(selected_experts)
    _ = num_experts
    slot_indices, gpu_route_mask = expert_cache.remap_experts_to_slots(flat_selected)
    gpu_route_indices = torch.nonzero(gpu_route_mask, as_tuple=False).flatten()

    if gpu_route_indices.numel() > 0:
        gpu_slots = slot_indices.index_select(0, gpu_route_indices)
        m_sizes, gpu_route_indices = _build_grouped_layout(
            gpu_slots,
            gpu_route_indices,
            expert_cache.num_slots,
        )
    else:
        m_sizes = None

    cpu_route_mask = ~gpu_route_mask
    cpu_route_indices_raw = torch.nonzero(cpu_route_mask, as_tuple=False).flatten()
    cpu_route_indices, cpu_task_expert_ids, cpu_task_offsets, ids_host, offsets_host = _build_cpu_task_layout(
        flat_selected,
        cpu_route_indices_raw,
    )

    return MoEExecutionPlan(
        layer_idx=layer_idx,
        gpu_route_indices=gpu_route_indices,
        gpu_m_sizes=m_sizes,
        cpu_route_indices=cpu_route_indices,
        cpu_task_expert_ids=cpu_task_expert_ids,
        cpu_task_offsets=cpu_task_offsets,
        cpu_task_expert_ids_host=ids_host,
        cpu_task_offsets_host=offsets_host,
        flat_selected_original=flat_selected,
        flat_selected_effective=flat_selected,
        substitution_lut=None,
        gpu_route_mask=gpu_route_mask,
        cpu_route_mask=cpu_route_mask,
    )


def build_draft_plan(
    layer_idx: int,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_cache: LayerExpertCache,
    draft_scheduler: DraftScheduler,
    num_experts: int,
    top_c: int,
) -> MoEExecutionPlan:
    return build_draft_plan_gpu(
        layer_idx=layer_idx,
        selected_experts=selected_experts,
        routing_weights=routing_weights,
        expert_cache=expert_cache,
        draft_scheduler=draft_scheduler,
        num_experts=num_experts,
        top_c=top_c,
    )


def build_draft_plan_gpu(
    layer_idx: int,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_cache: LayerExpertCache,
    draft_scheduler: DraftScheduler,
    num_experts: int,
    top_c: int,
) -> MoEExecutionPlan:
    flat_selected = _flatten_experts(selected_experts)
    flat_weights = _flatten_weights(routing_weights)
    device = flat_selected.device

    cached_expert_mask = expert_cache.get_cached_expert_mask()

    # Unified top_c=0 path (no CPU execution): map uncached experts to cached experts
    # through substitution LUT, then run fully on GPU.
    if top_c <= 0:
        substitution_lut = _build_topc0_substitution_lut(
            num_experts=num_experts,
            cached_expert_mask=cached_expert_mask,
            slot_to_expert_lut=expert_cache.get_slot_to_expert_lut(),
            device=device,
        )
        flat_effective = substitution_lut.index_select(0, flat_selected)
        gpu_slots = expert_cache.expert_to_slot_lut.index_select(0, flat_effective)
        gpu_route_indices = torch.arange(flat_selected.numel(), dtype=torch.int64, device=device)
        m_sizes, gpu_route_indices = _build_grouped_layout(
            gpu_slots,
            gpu_route_indices,
            expert_cache.num_slots,
        )
        gpu_route_mask = torch.ones_like(flat_selected, dtype=torch.bool)
        cpu_route_mask = torch.zeros_like(flat_selected, dtype=torch.bool)
        return MoEExecutionPlan(
            layer_idx=layer_idx,
            gpu_route_indices=gpu_route_indices,
            gpu_m_sizes=m_sizes,
            cpu_route_indices=None,
            cpu_task_expert_ids=None,
            cpu_task_offsets=None,
            cpu_task_expert_ids_host=None,
            cpu_task_offsets_host=None,
            flat_selected_original=flat_selected,
            flat_selected_effective=flat_effective,
            substitution_lut=substitution_lut,
            gpu_route_mask=gpu_route_mask,
            cpu_route_mask=cpu_route_mask,
        )

    slot_indices, gpu_mask = expert_cache.remap_experts_to_slots(flat_selected)
    uncached_expert_mask = torch.zeros((num_experts,), dtype=torch.bool, device=device)
    uncached_expert_ids = torch.unique(flat_selected[~gpu_mask])
    uncached_expert_mask[uncached_expert_ids] = True

    cpu_expert_mask = draft_scheduler.select_cpu_experts_gpu(
        uncached_expert_mask=uncached_expert_mask,
        routing_weights_flat=flat_weights,
        selected_experts_flat=flat_selected,
        top_c=top_c,
    )
    need_substitution_mask = uncached_expert_mask & (~cpu_expert_mask)
    substitution_lut = draft_scheduler.build_substitution_lut_gpu(
        cpu_expert_mask=need_substitution_mask,
        cached_expert_mask=cached_expert_mask,
        num_experts=num_experts,
        device=device,
    )

    flat_effective = substitution_lut.index_select(0, flat_selected)
    selected_cpu_mask = cpu_expert_mask.index_select(0, flat_selected)
    slot_eff, gpu_mask_eff = expert_cache.remap_experts_to_slots(flat_effective)
    gpu_route_mask = gpu_mask_eff & (~selected_cpu_mask)
    cpu_route_mask = ~gpu_route_mask

    gpu_route_indices = torch.nonzero(gpu_route_mask, as_tuple=False).flatten()
    if gpu_route_indices.numel() > 0:
        gpu_slots = slot_eff.index_select(0, gpu_route_indices)
        m_sizes, gpu_route_indices = _build_grouped_layout(
            gpu_slots,
            gpu_route_indices,
            expert_cache.num_slots,
        )
    else:
        m_sizes = None

    cpu_route_indices_raw = torch.nonzero(cpu_route_mask, as_tuple=False).flatten()
    cpu_route_indices, cpu_task_expert_ids, cpu_task_offsets, ids_host, offsets_host = _build_cpu_task_layout(
        flat_selected,
        cpu_route_indices_raw,
    )

    return MoEExecutionPlan(
        layer_idx=layer_idx,
        gpu_route_indices=gpu_route_indices,
        gpu_m_sizes=m_sizes,
        cpu_route_indices=cpu_route_indices,
        cpu_task_expert_ids=cpu_task_expert_ids,
        cpu_task_offsets=cpu_task_offsets,
        cpu_task_expert_ids_host=ids_host,
        cpu_task_offsets_host=offsets_host,
        flat_selected_original=flat_selected,
        flat_selected_effective=flat_effective,
        substitution_lut=substitution_lut,
        gpu_route_mask=gpu_route_mask,
        cpu_route_mask=cpu_route_mask,
    )


def build_verify_plan_gpu(
    layer_idx: int,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_cache: LayerExpertCache,
    num_experts: int,
) -> MoEExecutionPlan:
    return build_prefill_plan_gpu(
        layer_idx=layer_idx,
        selected_experts=selected_experts,
        routing_weights=routing_weights,
        expert_cache=expert_cache,
        num_experts=num_experts,
    )


def build_moe_execution_plan(
    selected_experts: torch.Tensor,
    expert_cache: LayerExpertCache,
) -> MoEExecutionPlan:
    """Backward-compatible wrapper for prefill/standard heterogeneous planning."""
    fake_weights = torch.ones_like(selected_experts, dtype=torch.float32)
    return build_prefill_plan_gpu(
        layer_idx=-1,
        selected_experts=selected_experts,
        routing_weights=fake_weights,
        expert_cache=expert_cache,
        num_experts=expert_cache.num_experts,
    )
