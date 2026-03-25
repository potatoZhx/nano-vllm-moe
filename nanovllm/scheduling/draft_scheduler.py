from __future__ import annotations

from abc import ABC, abstractmethod

import torch


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


class SimpleDraftScheduler(DraftScheduler):
    def select_cpu_experts(
        self,
        uncached_experts: list[int],
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        top_c: int,
    ) -> list[int]:
        if top_c <= 0 or not uncached_experts:
            return []

        flat_selected = selected_experts.reshape(-1).to(torch.int64)
        flat_weights = routing_weights.reshape(-1).float()
        score_map: dict[int, float] = {e: 0.0 for e in uncached_experts}

        for idx, expert_idx in enumerate(flat_selected.tolist()):
            if expert_idx in score_map:
                score_map[expert_idx] += float(flat_weights[idx].item())

        ranked = sorted(score_map.items(), key=lambda kv: (-kv[1], kv[0]))
        return [x[0] for x in ranked[:top_c]]

    def select_gpu_substitutes(
        self,
        need_substitution: list[int],
        cached_experts: set[int],
        all_experts: list[int],
    ) -> dict[int, int]:
        if not need_substitution or not cached_experts:
            return {}

        cached_sorted = sorted(cached_experts)
        mapping: dict[int, int] = {}
        for i, expert_idx in enumerate(need_substitution):
            mapping[expert_idx] = cached_sorted[i % len(cached_sorted)]
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
