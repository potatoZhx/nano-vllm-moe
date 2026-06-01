from __future__ import annotations

from abc import ABC, abstractmethod

import torch

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


class LFURankGuardStrategy(CacheStrategy):
    """LFU with top-1/2 eviction protection.

    Never evicts an expert whose rank_score exceeds protect_threshold,
    unless all cached experts are protected (safety valve falls back to
    pure LFU).

    rank_scores track how often each expert appears as rank-1 or rank-2
    in verify routing.  They are initialised to zero and updated online
    via EMA from verify routing metadata.
    """

    def __init__(
        self,
        num_experts: int = 128,
        protect_threshold: float = 0.15,
        ema_alpha: float = 0.95,
    ) -> None:
        self._num_experts = int(num_experts)
        self._protect_threshold = float(protect_threshold)
        self._ema_alpha = float(ema_alpha)
        self._rank_scores: dict[int, list[float]] = {}
        self.profile_seed_enabled = True

    @property
    def num_experts(self) -> int:
        return self._num_experts

    @property
    def protect_threshold(self) -> float:
        return self._protect_threshold

    def get_rank_scores(self, layer_idx: int) -> list[float] | None:
        return self._rank_scores.get(layer_idx)

    def set_rank_scores(self, layer_idx: int, scores: list[float]) -> None:
        self._rank_scores[layer_idx] = list(scores)

    def load_rank_scores_dict(
        self, data: dict[int, list[float]]
    ) -> None:
        for layer_idx, scores in data.items():
            self._rank_scores[int(layer_idx)] = list(scores)

    def is_protected(self, layer_idx: int, expert_idx: int) -> bool:
        scores = self._rank_scores.get(layer_idx)
        if scores is None:
            return False
        if not (0 <= expert_idx < len(scores)):
            return False
        return scores[expert_idx] >= self._protect_threshold

    def update_rank_scores_from_routing(
        self,
        layer_idx: int,
        selected_experts: torch.Tensor,
    ) -> None:
        """EMA-update rank scores from a verify routing tensor.

        ``selected_experts`` has shape ``[num_tokens, topk]``.
        Column 0 is rank-1; column 1 is rank-2.
        """
        if selected_experts.numel() == 0:
            return
        if selected_experts.dim() == 1:
            selected_experts = selected_experts.unsqueeze(0)
        num_tokens = selected_experts.size(0)
        if num_tokens <= 0:
            return

        ne = self._num_experts
        if layer_idx not in self._rank_scores:
            self._rank_scores[layer_idx] = [0.0] * ne

        current = [0.0] * ne

        col0 = selected_experts[:, 0].reshape(-1)
        if col0.device.type != "cpu" or col0.dtype != torch.int64:
            col0 = col0.to(device="cpu", dtype=torch.int64)
        for eid in col0.tolist():
            if 0 <= eid < ne:
                current[eid] += 2.0 / num_tokens

        if selected_experts.size(1) > 1:
            col1 = selected_experts[:, 1].reshape(-1)
            if col1.device.type != "cpu" or col1.dtype != torch.int64:
                col1 = col1.to(device="cpu", dtype=torch.int64)
            for eid in col1.tolist():
                if 0 <= eid < ne:
                    current[eid] += 1.0 / num_tokens

        alpha = self._ema_alpha
        scores = self._rank_scores[layer_idx]
        for j in range(ne):
            scores[j] = alpha * scores[j] + (1.0 - alpha) * current[j]

    def select_victim_slot(
        self,
        snapshot: LayerCacheSnapshot,
        incoming_expert_idx: int,
        step_id: int,
    ) -> int | None:
        _ = incoming_expert_idx, step_id
        best_slot: int | None = None
        best_count: int | None = None
        all_protected = True

        for slot_idx, expert_idx in enumerate(
            snapshot.slot_to_expert_lut.tolist()
        ):
            if expert_idx < 0:
                return slot_idx

            if not self.is_protected(snapshot.layer_idx, expert_idx):
                all_protected = False
                cnt = snapshot.access_count[expert_idx]
                if best_count is None or cnt < best_count:
                    best_count = cnt
                    best_slot = slot_idx

        if best_slot is not None:
            return best_slot

        if all_protected:
            return self._fallback_lfu(snapshot)

        return best_slot

    @staticmethod
    def _fallback_lfu(snapshot: LayerCacheSnapshot) -> int | None:
        best_slot: int | None = None
        best_count: int | None = None
        for slot_idx, expert_idx in enumerate(
            snapshot.slot_to_expert_lut.tolist()
        ):
            if expert_idx < 0:
                return slot_idx
            cnt = snapshot.access_count[expert_idx]
            if best_count is None or cnt < best_count:
                best_count = cnt
                best_slot = slot_idx
        return best_slot

    def select_victim_slot_cpu(
        self,
        cache_slot_to_expert: list[int],
        cache_access_count: list[int],
        layer_idx: int,
        is_pending_fn: object,
    ) -> int | None:
        """Fast victim selection without snapshot (used by segment-indexed prefetch)."""
        best_slot: int | None = None
        best_count: int | None = None
        all_protected = True

        for slot_idx, slot_expert in enumerate(cache_slot_to_expert):
            if is_pending_fn(slot_idx):
                continue
            if int(slot_expert) < 0:
                return slot_idx
            expert = int(slot_expert)
            if not self.is_protected(layer_idx, expert):
                all_protected = False
                cnt = int(cache_access_count[expert])
                if best_count is None or cnt < best_count:
                    best_count = cnt
                    best_slot = slot_idx

        if best_slot is not None:
            return best_slot

        if all_protected:
            best_fb_slot: int | None = None
            best_fb_count: int | None = None
            for slot_idx, slot_expert in enumerate(cache_slot_to_expert):
                if is_pending_fn(slot_idx):
                    continue
                if int(slot_expert) < 0:
                    return slot_idx
                cnt = int(cache_access_count[int(slot_expert)])
                if best_fb_count is None or cnt < best_fb_count:
                    best_fb_count = cnt
                    best_fb_slot = slot_idx
            return best_fb_slot

        return best_slot


class LFUOnlineRankGuardStrategy(LFURankGuardStrategy):
    """LFU-RankGuard variant that learns protection only from online routing."""

    def __init__(
        self,
        num_experts: int = 128,
        protect_threshold: float = 0.15,
        ema_alpha: float = 0.95,
    ) -> None:
        super().__init__(
            num_experts=num_experts,
            protect_threshold=protect_threshold,
            ema_alpha=ema_alpha,
        )
        self.profile_seed_enabled = False


def create_cache_strategy(name: str, **kwargs: object) -> CacheStrategy:
    normalized = name.strip().lower()
    if normalized == "lru":
        return LRUCacheStrategy()
    if normalized == "lfu":
        return LFUCacheStrategy()
    if normalized == "adaptive":
        return AdaptiveCacheStrategy()
    if normalized == "lfu_rankguard":
        return LFURankGuardStrategy(**kwargs)  # type: ignore[arg-type]
    if normalized == "lfu_rankguard_online":
        return LFUOnlineRankGuardStrategy(**kwargs)  # type: ignore[arg-type]
    raise ValueError(f"Unsupported cache strategy: {name}")
