from __future__ import annotations

from dataclasses import dataclass

import torch

from nanovllm.expert.cache import LayerExpertCache


@dataclass
class MoEExecutionPlan:
    gpu_route_indices: torch.Tensor
    cpu_route_indices: torch.Tensor | None
    m_sizes: torch.Tensor | None


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


def build_moe_execution_plan(
    selected_experts: torch.Tensor,
    expert_cache: LayerExpertCache,
) -> MoEExecutionPlan:
    """Build per-layer execution plan for heterogeneous MoE execution."""
    slot_indices, gpu_mask = expert_cache.remap_experts_to_slots(selected_experts)
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

    if gpu_route_indices.numel() < slot_indices.numel():
        cpu_route_indices = torch.nonzero(~gpu_mask, as_tuple=False).flatten()
    else:
        cpu_route_indices = None

    return MoEExecutionPlan(
        gpu_route_indices=gpu_route_indices,
        cpu_route_indices=cpu_route_indices,
        m_sizes=m_sizes,
    )
