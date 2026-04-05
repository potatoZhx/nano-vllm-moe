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
    flat_selected_original: torch.Tensor
    flat_selected_effective: torch.Tensor
    substitution_lut: torch.Tensor | None
    gpu_route_mask: torch.Tensor | None
    cpu_route_mask: torch.Tensor | None

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
    # Group tokens by slot id for grouped GEMM.
    sorted_slots, sort_idx = torch.sort(gpu_slots)
    sorted_gpu_route_indices = gpu_route_indices.index_select(0, sort_idx)
    m_sizes = torch.bincount(sorted_slots, minlength=num_slots).to(torch.int32)
    return m_sizes, sorted_gpu_route_indices


def _build_cpu_task_layout(
    flat_selected_original: torch.Tensor,
    cpu_route_indices: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if cpu_route_indices.numel() == 0:
        return None, None, None

    cpu_experts = flat_selected_original.index_select(0, cpu_route_indices)
    sorted_experts, sort_idx = torch.sort(cpu_experts)
    sorted_route_indices = cpu_route_indices.index_select(0, sort_idx)
    task_expert_ids, counts = torch.unique_consecutive(sorted_experts, return_counts=True)
    task_offsets = torch.zeros(
        task_expert_ids.numel() + 1,
        dtype=torch.int64,
        device=cpu_route_indices.device,
    )
    task_offsets[1:] = torch.cumsum(counts.to(torch.int64), dim=0)
    return sorted_route_indices, task_expert_ids.to(torch.int64), task_offsets


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
    cpu_route_indices, cpu_task_expert_ids, cpu_task_offsets = _build_cpu_task_layout(
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

    slot_indices, gpu_mask = expert_cache.remap_experts_to_slots(flat_selected)
    if gpu_mask.all():
        gpu_route_indices = torch.arange(flat_selected.numel(), dtype=torch.int64, device=device)
        gpu_slots = slot_indices
        m_sizes, gpu_route_indices = _build_grouped_layout(
            gpu_slots,
            gpu_route_indices,
            expert_cache.num_slots,
        )
        return MoEExecutionPlan(
            layer_idx=layer_idx,
            gpu_route_indices=gpu_route_indices,
            gpu_m_sizes=m_sizes,
            cpu_route_indices=None,
            cpu_task_expert_ids=None,
            cpu_task_offsets=None,
            flat_selected_original=flat_selected,
            flat_selected_effective=flat_selected,
            substitution_lut=None,
            gpu_route_mask=gpu_mask,
            cpu_route_mask=~gpu_mask,
        )

    cached_expert_mask = expert_cache.get_cached_expert_mask()
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
    cpu_route_indices, cpu_task_expert_ids, cpu_task_offsets = _build_cpu_task_layout(
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
