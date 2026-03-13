from __future__ import annotations

from dataclasses import dataclass

import torch

from nanovllm.layers.fuse_moe import get_expert_counts_and_idx
from nanovllm.expert.cache import LayerExpertCache


@dataclass
class MoEExecutionPlan:
    slot_indices: torch.Tensor
    gpu_mask: torch.Tensor
    gpu_token_count: int
    cpu_token_count: int
    m_sizes: torch.Tensor | None
    sort_idx: torch.Tensor | None
    inv_sort_idx: torch.Tensor | None


def build_moe_execution_plan(
    selected_experts: torch.Tensor,
    expert_cache: LayerExpertCache,
) -> MoEExecutionPlan:
    """Build per-layer execution plan for heterogeneous MoE execution."""
    slot_indices, gpu_mask = expert_cache.remap_experts_to_slots(selected_experts)
    gpu_token_count = int(gpu_mask.sum().item())
    cpu_token_count = selected_experts.numel() - gpu_token_count

    if gpu_token_count > 0:
        gpu_slots = slot_indices[gpu_mask]
        m_sizes, sort_idx, inv_sort_idx = get_expert_counts_and_idx(gpu_slots, expert_cache.num_slots)
    else:
        m_sizes = sort_idx = inv_sort_idx = None

    return MoEExecutionPlan(
        slot_indices=slot_indices,
        gpu_mask=gpu_mask,
        gpu_token_count=gpu_token_count,
        cpu_token_count=cpu_token_count,
        m_sizes=m_sizes,
        sort_idx=sort_idx,
        inv_sort_idx=inv_sort_idx,
    )
