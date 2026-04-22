from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from nanovllm.config import Config

if TYPE_CHECKING:
    from nanovllm.expert.prefetcher import PrefetchCandidate


class PrefetchStrategy(ABC):
    @abstractmethod
    def rank(
        self,
        candidates: list["PrefetchCandidate"],
        step_id: int,
    ) -> list["PrefetchCandidate"]:
        raise NotImplementedError


class NoopPrefetchStrategy(PrefetchStrategy):
    def rank(
        self,
        candidates: list["PrefetchCandidate"],
        step_id: int,
    ) -> list["PrefetchCandidate"]:
        _ = step_id
        return candidates


class HistoryWindowPrefetchStrategy(PrefetchStrategy):
    def __init__(self, config: Config):
        self.config = config

    def rank(
        self,
        candidates: list["PrefetchCandidate"],
        step_id: int,
    ) -> list["PrefetchCandidate"]:
        ttl = int(self.config.prefetch_history_ttl_steps)
        filtered = [c for c in candidates if step_id - c.last_seen_step <= ttl]
        filtered.sort(key=lambda c: (-c.priority, c.layer_idx, c.expert_idx))
        return filtered


def create_prefetch_strategy(name: str, config: Config) -> PrefetchStrategy:
    normalized = name.strip().lower()
    if normalized == "noop":
        return NoopPrefetchStrategy()
    if normalized == "history_window":
        return HistoryWindowPrefetchStrategy(config)
    raise ValueError(f"Unsupported prefetch strategy: {name}")
