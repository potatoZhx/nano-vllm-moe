from __future__ import annotations

from abc import ABC, abstractmethod

from nanovllm.expert.cache import LayerCacheSnapshot


class CacheStrategy(ABC):
    @abstractmethod
    def select_victim_slot(
        self,
        snapshot: LayerCacheSnapshot,
        incoming_expert_idx: int,
        step_id: int,
    ) -> int | None:
        raise NotImplementedError


class LRUCacheStrategy(CacheStrategy):
    def select_victim_slot(
        self,
        snapshot: LayerCacheSnapshot,
        incoming_expert_idx: int,
        step_id: int,
    ) -> int | None:
        _ = incoming_expert_idx, step_id
        best_slot = None
        best_access = None
        for slot_idx, expert_idx in enumerate(snapshot.slot_to_expert_lut.tolist()):
            if expert_idx < 0:
                return slot_idx
            last = snapshot.last_access_step[expert_idx]
            if best_access is None or last < best_access:
                best_access = last
                best_slot = slot_idx
        return best_slot


class LFUCacheStrategy(CacheStrategy):
    def select_victim_slot(
        self,
        snapshot: LayerCacheSnapshot,
        incoming_expert_idx: int,
        step_id: int,
    ) -> int | None:
        _ = incoming_expert_idx, step_id
        best_slot = None
        best_count = None
        for slot_idx, expert_idx in enumerate(snapshot.slot_to_expert_lut.tolist()):
            if expert_idx < 0:
                return slot_idx
            cnt = snapshot.access_count[expert_idx]
            if best_count is None or cnt < best_count:
                best_count = cnt
                best_slot = slot_idx
        return best_slot


class AdaptiveCacheStrategy(CacheStrategy):
    """Placeholder adaptive strategy that currently falls back to LRU."""

    def __init__(self) -> None:
        self._fallback = LRUCacheStrategy()

    def select_victim_slot(
        self,
        snapshot: LayerCacheSnapshot,
        incoming_expert_idx: int,
        step_id: int,
    ) -> int | None:
        return self._fallback.select_victim_slot(snapshot, incoming_expert_idx, step_id)


def create_cache_strategy(name: str) -> CacheStrategy:
    normalized = name.strip().lower()
    if normalized == "lru":
        return LRUCacheStrategy()
    if normalized == "lfu":
        return LFUCacheStrategy()
    if normalized == "adaptive":
        return AdaptiveCacheStrategy()
    raise ValueError(f"Unsupported cache strategy: {name}")
