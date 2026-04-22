from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class DraftSchedulerContext:
    uncached_expert_mask: torch.Tensor
    routing_weights_flat: torch.Tensor
    selected_experts_flat: torch.Tensor
    top_c: int


def build_draft_scheduler_context(
    uncached_expert_mask: torch.Tensor,
    routing_weights_flat: torch.Tensor,
    selected_experts_flat: torch.Tensor,
    top_c: int,
) -> DraftSchedulerContext:
    return DraftSchedulerContext(
        uncached_expert_mask=uncached_expert_mask,
        routing_weights_flat=routing_weights_flat,
        selected_experts_flat=selected_experts_flat,
        top_c=int(top_c),
    )


class DraftScheduler(ABC):
    @abstractmethod
    def select_cpu_experts(
        self,
        uncached_experts: list[int],
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        top_c: int,
    ) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def select_gpu_substitutes(
        self,
        need_substitution: list[int],
        cached_experts: set[int],
        all_experts: list[int],
    ) -> dict[int, int]:
        raise NotImplementedError

    @abstractmethod
    def select_experts_to_transfer(
        self,
        recent_activations: list,
        cached_experts: set[int],
        cache_capacity: int,
    ) -> list[tuple[int, int]]:
        raise NotImplementedError

    @abstractmethod
    def select_cpu_experts_gpu(
        self,
        uncached_expert_mask: torch.Tensor,
        routing_weights_flat: torch.Tensor,
        selected_experts_flat: torch.Tensor,
        top_c: int,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def build_substitution_lut_gpu(
        self,
        cpu_expert_mask: torch.Tensor,
        cached_expert_mask: torch.Tensor,
        num_experts: int,
        device: torch.device,
    ) -> torch.Tensor:
        raise NotImplementedError


class SimpleDraftScheduler(DraftScheduler):
    def select_cpu_experts_gpu(
        self,
        uncached_expert_mask: torch.Tensor,
        routing_weights_flat: torch.Tensor,
        selected_experts_flat: torch.Tensor,
        top_c: int,
    ) -> torch.Tensor:
        num_experts = int(uncached_expert_mask.numel())
        out = torch.zeros((num_experts,), dtype=torch.bool, device=uncached_expert_mask.device)
        if top_c <= 0:
            return out

        candidate_mask = uncached_expert_mask.index_select(0, selected_experts_flat.to(torch.int64))
        if not candidate_mask.any():
            return out

        candidate_experts = selected_experts_flat[candidate_mask].to(torch.int64)
        candidate_weights = routing_weights_flat[candidate_mask].float()
        score = torch.zeros((num_experts,), dtype=torch.float32, device=uncached_expert_mask.device)
        score.scatter_add_(0, candidate_experts, candidate_weights)

        candidate_ids = torch.nonzero(uncached_expert_mask, as_tuple=False).flatten().to(torch.int64)
        if candidate_ids.numel() == 0:
            return out

        # Deterministic tie-break: expert id ascending for equal score.
        order_by_id = torch.argsort(candidate_ids, stable=True)
        candidate_ids = candidate_ids.index_select(0, order_by_id)
        candidate_scores = score.index_select(0, candidate_ids)
        ranked = torch.argsort(candidate_scores, descending=True, stable=True)
        pick_n = min(int(top_c), int(candidate_ids.numel()))
        picked = candidate_ids.index_select(0, ranked[:pick_n])
        out[picked] = True
        return out

    def build_substitution_lut_gpu(
        self,
        cpu_expert_mask: torch.Tensor,
        cached_expert_mask: torch.Tensor,
        num_experts: int,
        device: torch.device,
    ) -> torch.Tensor:
        lut = torch.arange(num_experts, dtype=torch.int64, device=device)
        cpu_ids = torch.nonzero(cpu_expert_mask, as_tuple=False).flatten().to(torch.int64)
        cached_ids = torch.nonzero(cached_expert_mask, as_tuple=False).flatten().to(torch.int64)
        if cpu_ids.numel() == 0 or cached_ids.numel() == 0:
            return lut

        rr = torch.arange(cpu_ids.numel(), dtype=torch.int64, device=device) % cached_ids.numel()
        subs = cached_ids.index_select(0, rr)
        lut.scatter_(0, cpu_ids, subs)
        return lut

    def select_cpu_experts(
        self,
        uncached_experts: list[int],
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        top_c: int,
    ) -> list[int]:
        if not uncached_experts or top_c <= 0:
            return []
        flat_selected = selected_experts.reshape(-1).to(torch.int64)
        flat_weights = routing_weights.reshape(-1).float()
        num_experts = int(max(int(flat_selected.max().item()) + 1, max(uncached_experts) + 1))
        uncached_mask = torch.zeros((num_experts,), dtype=torch.bool, device=flat_selected.device)
        uncached_idx = torch.tensor(uncached_experts, dtype=torch.int64, device=flat_selected.device)
        uncached_mask[uncached_idx] = True
        cpu_mask = self.select_cpu_experts_gpu(uncached_mask, flat_weights, flat_selected, top_c)
        picked = torch.nonzero(cpu_mask, as_tuple=False).flatten().tolist()
        return picked

    def select_gpu_substitutes(
        self,
        need_substitution: list[int],
        cached_experts: set[int],
        all_experts: list[int],
    ) -> dict[int, int]:
        if not need_substitution or not cached_experts or not all_experts:
            return {}

        num_experts = max(max(all_experts) + 1, max(cached_experts) + 1, max(need_substitution) + 1)
        device = torch.device("cpu")
        cpu_mask = torch.zeros((num_experts,), dtype=torch.bool, device=device)
        cached_mask = torch.zeros((num_experts,), dtype=torch.bool, device=device)
        cpu_mask[torch.tensor(need_substitution, dtype=torch.int64)] = True
        cached_mask[torch.tensor(sorted(cached_experts), dtype=torch.int64)] = True
        lut = self.build_substitution_lut_gpu(cpu_mask, cached_mask, num_experts, device=device)

        mapping: dict[int, int] = {}
        for src in need_substitution:
            mapping[src] = int(lut[src].item())
        return mapping

    def select_experts_to_transfer(
        self,
        recent_activations: list,
        cached_experts: set[int],
        cache_capacity: int,
    ) -> list[tuple[int, int]]:
        return []


def create_draft_scheduler(name: str) -> DraftScheduler:
    normalized = name.strip().lower()
    if normalized == "simple":
        return SimpleDraftScheduler()
    raise ValueError(f"Unsupported draft scheduler: {name}")
