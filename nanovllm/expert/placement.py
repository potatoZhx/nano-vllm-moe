from __future__ import annotations

from dataclasses import dataclass

import torch

from nanovllm.expert.cache import LayerExpertCache


@dataclass
class MoEExecutionPlan:
    gpu_mask: torch.Tensor
    gpu_indices: torch.Tensor
    gpu_slots: torch.Tensor
    m_sizes: torch.Tensor | None
    sort_idx: torch.Tensor | None


def _build_grouped_layout(
    gpu_slots: torch.Tensor,
    num_slots: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Group tokens by slot id for grouped GEMM.
    sorted_slots, sort_idx = torch.sort(gpu_slots)
    m_sizes = torch.bincount(sorted_slots, minlength=num_slots).to(torch.int32)
    return m_sizes, sort_idx


def build_moe_execution_plan(
    selected_experts: torch.Tensor,
    expert_cache: LayerExpertCache,
) -> MoEExecutionPlan:
    """Build per-layer execution plan for heterogeneous MoE execution."""
    slot_indices, gpu_mask = expert_cache.remap_experts_to_slots(selected_experts)
    gpu_indices = torch.nonzero(gpu_mask, as_tuple=False).flatten()
    gpu_slots = slot_indices[gpu_indices]

    if gpu_slots.numel() > 0:
        m_sizes, sort_idx = _build_grouped_layout(gpu_slots, expert_cache.num_slots)
    else:
        m_sizes = sort_idx = None

    return MoEExecutionPlan(
        gpu_mask=gpu_mask,
        gpu_indices=gpu_indices,
        gpu_slots=gpu_slots,
        m_sizes=m_sizes,
        sort_idx=sort_idx,
    )
