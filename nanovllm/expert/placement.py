from __future__ import annotations

from dataclasses import dataclass

import torch

from nanovllm.expert.cache import LayerExpertCache
from nanovllm.scheduling.draft_scheduler import DraftScheduler


@dataclass
class MoEExecutionPlan:
    layer_idx: int
    gpu_route_indices: torch.Tensor
    cpu_route_indices: torch.Tensor | None
    m_sizes: torch.Tensor | None
    substitution_map: dict[int, int] | None = None
    flat_selected_original: torch.Tensor | None = None
    flat_selected_effective: torch.Tensor | None = None


def _build_grouped_layout(
    gpu_slots: torch.Tensor,
    gpu_route_indices: torch.Tensor,
    num_slots: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Group tokens by slot id for grouped GEMM.
    sorted_slots, sort_idx = torch.sort(gpu_slots)
    sorted_gpu_route_indices = gpu_route_indices.index_select(0, sort_idx)
    m_sizes = torch.bincount(sorted_slots, minlength=num_slots).to(torch.int32)
    return m_sizes, sorted_gpu_route_indices


def _flatten_experts(selected_experts: torch.Tensor) -> torch.Tensor:
    return selected_experts.reshape(-1).to(torch.int64)


def _flatten_weights(routing_weights: torch.Tensor) -> torch.Tensor:
    return routing_weights.reshape(-1).float()


def build_prefill_plan(
    layer_idx: int,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_cache: LayerExpertCache,
    num_experts: int,
) -> MoEExecutionPlan:
    flat_selected = _flatten_experts(selected_experts)
    slot_indices, gpu_mask = expert_cache.remap_experts_to_slots(flat_selected)
    gpu_route_indices = torch.nonzero(gpu_mask, as_tuple=False).flatten()

    if gpu_route_indices.numel() > 0:
        gpu_slots = slot_indices.index_select(0, gpu_route_indices)
        m_sizes, gpu_route_indices = _build_grouped_layout(
            gpu_slots,
            gpu_route_indices,
            expert_cache.num_slots,
        )
    else:
        m_sizes = None

    cpu_route_indices = None
    if gpu_route_indices.numel() < slot_indices.numel():
        cpu_route_indices = torch.nonzero(~gpu_mask, as_tuple=False).flatten()

    return MoEExecutionPlan(
        layer_idx=layer_idx,
        gpu_route_indices=gpu_route_indices,
        cpu_route_indices=cpu_route_indices,
        m_sizes=m_sizes,
        substitution_map={},
        flat_selected_original=flat_selected,
        flat_selected_effective=flat_selected,
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
    flat_selected = _flatten_experts(selected_experts)
    flat_weights = _flatten_weights(routing_weights)

    slot_indices, gpu_mask = expert_cache.remap_experts_to_slots(flat_selected)
    uncached = torch.unique(flat_selected[~gpu_mask]).tolist() if (~gpu_mask).any() else []
    cached_experts = set(expert_cache.expert_to_slot.keys())

    cpu_experts = set(
        draft_scheduler.select_cpu_experts(uncached, flat_weights, flat_selected, top_c)
    )
    need_substitution = [e for e in uncached if e not in cpu_experts]
    substitution_map = draft_scheduler.select_gpu_substitutes(
        need_substitution=need_substitution,
        cached_experts=cached_experts,
        all_experts=list(range(num_experts)),
    )

    flat_effective = flat_selected.clone()
    for src, dst in substitution_map.items():
        flat_effective[flat_selected == src] = dst

    slot_eff, gpu_mask_eff = expert_cache.remap_experts_to_slots(flat_effective)

    if cpu_experts:
        cpu_mask = torch.isin(flat_selected, torch.tensor(sorted(cpu_experts), device=flat_selected.device))
        gpu_mask_eff = gpu_mask_eff & (~cpu_mask)

    gpu_route_indices = torch.nonzero(gpu_mask_eff, as_tuple=False).flatten()
    if gpu_route_indices.numel() > 0:
        gpu_slots = slot_eff.index_select(0, gpu_route_indices)
        m_sizes, gpu_route_indices = _build_grouped_layout(
            gpu_slots,
            gpu_route_indices,
            expert_cache.num_slots,
        )
    else:
        m_sizes = None

    cpu_route_indices = None
    cpu_mask_all = ~gpu_mask_eff
    if cpu_mask_all.any():
        cpu_route_indices = torch.nonzero(cpu_mask_all, as_tuple=False).flatten()

    return MoEExecutionPlan(
        layer_idx=layer_idx,
        gpu_route_indices=gpu_route_indices,
        cpu_route_indices=cpu_route_indices,
        m_sizes=m_sizes,
        substitution_map=substitution_map,
        flat_selected_original=flat_selected,
        flat_selected_effective=flat_effective,
    )


def build_moe_execution_plan(
    selected_experts: torch.Tensor,
    expert_cache: LayerExpertCache,
) -> MoEExecutionPlan:
    """Backward-compatible wrapper for prefill/standard heterogeneous planning."""
    fake_weights = torch.ones_like(selected_experts, dtype=torch.float32)
    return build_prefill_plan(
        layer_idx=-1,
        selected_experts=selected_experts,
        routing_weights=fake_weights,
        expert_cache=expert_cache,
        num_experts=expert_cache.num_experts,
    )
