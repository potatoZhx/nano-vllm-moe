from __future__ import annotations

import os
import time
from collections import defaultdict
from dataclasses import dataclass
from math import isfinite

import torch

from nanovllm.config import Config
from nanovllm.expert.cache import ActiveReservation, LayerExpertCache, PublishedExpert, StagingReservation
from nanovllm.expert.cpu_weights import copy_expert_tensor
from nanovllm.expert.runtime_meta import LayerRuntimeMetaCPU, ModelRuntimeMetaRecorder
from nanovllm.scheduling.cache_strategy import CacheStrategy, LFURankGuardStrategy
from nanovllm.scheduling.prefetch_strategy import PrefetchStrategy


@dataclass
class PrefetchCandidate:
    layer_idx: int
    expert_idx: int
    source: str
    score_sum: float
    activation_count: int
    first_seen_step: int
    last_seen_step: int
    priority: float


@dataclass
class PrefetchTicket:
    step_id: int
    layer_idx: int
    expert_idx: int
    source: str
    staging_slot_idx: int
    staging_generation: int
    submit_ts_ms: float
    ready_event: object
    ready: bool = False
    direct_active: bool = False
    active_slot_idx: int = -1
    active_generation: int = -1
    active_slot_prev_expert: int = -1  # expert evicted by this prefetch (-1 if slot was empty)
    segment_id: int = -1
    num_bytes: int = 0
    transfer_enqueue_ms: float = 0.0
    transfer_stream_idx: int = 0


@dataclass
class PrefetchResidency:
    sequence: int
    source: str
    publish_step: int
    num_bytes: int
    first_consume_step: int = -1
    last_consume_step: int = -1
    consume_count: int = 0
    evict_step: int = -1
    replacement_source: str = ""


def compute_priority(
    source: str,
    score_sum: float,
    activation_count: int,
    age: int,
    config: Config,
) -> float:
    source_weight = {
        "prefill_history": config.prefetch_source_weight_prefill,
        "verify_history": config.prefetch_source_weight_verify,
        "draft_live": config.prefetch_source_weight_draft,
    }.get(source, 1.0)
    return (
        source_weight * float(score_sum)
        + float(config.prefetch_activation_count_weight) * float(activation_count)
        - float(config.prefetch_age_penalty) * float(age)
    )


def select_predictive_victim_slot(
    *,
    slots: list[int] | tuple[int, ...],
    pending: list[int] | tuple[int, ...],
    access_values,
    protected_experts: set[int] | frozenset[int] = frozenset(),
) -> int | None:
    """Pure predictive victim rule shared by live and shadow runtimes.

    Empty slots win, pending slots are unavailable, then the least-accessed
    unprotected expert is selected.  If every usable expert is protected, the
    same least-accessed fallback keeps prefetch progressing.
    """
    selected: tuple[float, int] | None = None
    fallback: tuple[float, int] | None = None
    for slot_idx, raw_expert in enumerate(slots):
        if slot_idx < len(pending) and int(pending[slot_idx]) >= 0:
            continue
        expert_idx = int(raw_expert)
        if expert_idx < 0:
            return slot_idx
        value = (
            float(access_values[expert_idx])
            if 0 <= expert_idx < len(access_values)
            else 0.0
        )
        candidate = (value, slot_idx)
        if fallback is None or candidate < fallback:
            fallback = candidate
        if expert_idx in protected_experts:
            continue
        if selected is None or candidate < selected:
            selected = candidate
    chosen = selected if selected is not None else fallback
    return chosen[1] if chosen is not None else None


class GlobalWarmStartQueue:
    def __init__(self, config: Config):
        self.config = config
        self.entries: dict[tuple[int, int], PrefetchCandidate] = {}

    def update_from_runtime_meta(
        self,
        runtime_meta: dict[int, LayerRuntimeMetaCPU] | None,
        source: str,
        step_id: int,
        layer_caches: dict[int, LayerExpertCache],
    ) -> dict[str, float]:
        stats = defaultdict(float)
        if not runtime_meta:
            return stats

        for layer_idx, meta in runtime_meta.items():
            cache = layer_caches.get(layer_idx)
            if cache is None:
                continue
            stats["queue_layer_count"] += 1.0
            aggregate_t0 = time.perf_counter()
            if meta.aggregated_expert_ids is None or meta.aggregated_score_sum is None or meta.aggregated_activation_count is None:
                if meta.selected_experts is None or meta.routing_weights is None:
                    continue
                flat_experts = meta.selected_experts.reshape(-1)
                if flat_experts.device.type != "cpu" or flat_experts.dtype != torch.int64:
                    flat_experts = flat_experts.to(device="cpu", dtype=torch.int64)
                if flat_experts.numel() == 0:
                    continue
                flat_weights = meta.routing_weights.reshape(-1)
                if flat_weights.device.type != "cpu" or flat_weights.dtype != torch.float32:
                    flat_weights = flat_weights.to(device="cpu", dtype=torch.float32)
                unique_ids, inverse = torch.unique(flat_experts, return_inverse=True)
                score_sum = torch.zeros((unique_ids.numel(),), dtype=torch.float32, device=torch.device("cpu"))
                score_sum.scatter_add_(0, inverse, flat_weights)
                counts = torch.zeros((unique_ids.numel(),), dtype=torch.int64, device=torch.device("cpu"))
                counts.scatter_add_(0, inverse, torch.ones_like(inverse, dtype=torch.int64))
            else:
                unique_ids = meta.aggregated_expert_ids
                if unique_ids.device.type != "cpu" or unique_ids.dtype != torch.int64:
                    unique_ids = unique_ids.to(device="cpu", dtype=torch.int64)
                score_sum = meta.aggregated_score_sum
                if score_sum.device.type != "cpu" or score_sum.dtype != torch.float32:
                    score_sum = score_sum.to(device="cpu", dtype=torch.float32)
                counts = meta.aggregated_activation_count
                if counts.device.type != "cpu" or counts.dtype != torch.int64:
                    counts = counts.to(device="cpu", dtype=torch.int64)
            stats["queue_aggregate_ms"] += (time.perf_counter() - aggregate_t0) * 1000.0
            if unique_ids.numel() == 0:
                continue
            filter_t0 = time.perf_counter()
            uncached_entries: list[tuple[int, float, int]] = []
            for expert_idx, new_score, new_count in zip(unique_ids.tolist(), score_sum.tolist(), counts.tolist()):
                expert_idx = int(expert_idx)
                if cache.is_cached_cpu(expert_idx):
                    continue
                uncached_entries.append((expert_idx, float(new_score), int(new_count)))
            stats["queue_filter_ms"] += (time.perf_counter() - filter_t0) * 1000.0
            stats["queue_uncached_candidate_count"] += float(len(uncached_entries))
            if not uncached_entries:
                continue

            update_t0 = time.perf_counter()
            for expert_idx, new_score, new_count in uncached_entries:
                key = (int(layer_idx), expert_idx)
                if key in self.entries:
                    entry = self.entries[key]
                    decay = float(self.config.prefetch_history_decay)
                    entry.score_sum = decay * entry.score_sum + new_score
                    entry.activation_count = int(round(decay * entry.activation_count)) + new_count
                    entry.last_seen_step = int(step_id)
                    entry.source = source
                else:
                    entry = PrefetchCandidate(
                        layer_idx=int(layer_idx),
                        expert_idx=int(expert_idx),
                        source=source,
                        score_sum=new_score,
                        activation_count=new_count,
                        first_seen_step=int(step_id),
                        last_seen_step=int(step_id),
                        priority=0.0,
                    )
                    self.entries[key] = entry

                age = max(0, int(step_id) - int(entry.last_seen_step))
                entry.priority = compute_priority(
                    source=entry.source,
                    score_sum=entry.score_sum,
                    activation_count=entry.activation_count,
                    age=age,
                    config=self.config,
                )
            stats["queue_entry_update_ms"] += (time.perf_counter() - update_t0) * 1000.0
        return stats

    def prune(self, step_id: int, layer_caches: dict[int, LayerExpertCache]) -> None:
        stale_keys = []
        ttl = int(self.config.prefetch_history_ttl_steps)
        for key, entry in self.entries.items():
            layer_idx, expert_idx = key
            cache = layer_caches.get(layer_idx)
            if cache is None:
                stale_keys.append(key)
                continue
            if int(step_id) - int(entry.last_seen_step) > ttl:
                stale_keys.append(key)
                continue
            if cache.is_cached_cpu(expert_idx):
                stale_keys.append(key)
                continue

        for key in stale_keys:
            self.entries.pop(key, None)

        cap = int(self.config.prefetch_global_queue_capacity)
        if cap > 0 and len(self.entries) > cap:
            ranked = sorted(self.entries.items(), key=lambda kv: kv[1].priority, reverse=True)
            self.entries = dict(ranked[:cap])

    def ranked_candidates(
        self,
        step_id: int,
        layer_caches: dict[int, LayerExpertCache],
        inflight_keys: set[tuple[int, int]],
        max_layer_idx: int | None = None,
    ) -> list[PrefetchCandidate]:
        self.prune(step_id, layer_caches)
        max_layer = None if max_layer_idx is None else int(max_layer_idx)
        ranked: list[PrefetchCandidate] = []
        for key, entry in self.entries.items():
            layer_idx, expert_idx = key
            if max_layer is not None and int(layer_idx) > max_layer:
                continue
            if key in inflight_keys:
                continue
            cache = layer_caches.get(layer_idx)
            if cache is None:
                continue
            if cache.is_cached_cpu(expert_idx):
                continue
            age = max(0, int(step_id) - int(entry.last_seen_step))
            entry.priority = compute_priority(
                source=entry.source,
                score_sum=entry.score_sum,
                activation_count=entry.activation_count,
                age=age,
                config=self.config,
            )
            ranked.append(entry)
        ranked.sort(key=lambda x: (-x.priority, x.layer_idx, x.expert_idx))
        return ranked


class SegmentCandidateIndex:
    def __init__(self, config: Config):
        self.config = config
        granularity = str(getattr(config, "draft_prefetch_frontier_granularity", "segment"))
        self.segment_size = 1 if granularity == "layer" else max(1, int(getattr(config, "draft_prefetch_segment_size", 12)))
        self.entries_by_segment: dict[int, dict[tuple[int, int], PrefetchCandidate]] = defaultdict(dict)
        self._ranked_cache_by_segment: dict[int, list[PrefetchCandidate]] = {}
        self._dirty_segments: set[int] = set()

    def clear(self) -> None:
        self.entries_by_segment.clear()
        self._ranked_cache_by_segment.clear()
        self._dirty_segments.clear()

    def _segment_id(self, layer_idx: int) -> int:
        return int(layer_idx) // int(self.segment_size)

    def _mark_dirty(self, segment_id: int) -> None:
        self._dirty_segments.add(int(segment_id))

    def _ranked_entries_for_segment(self, segment_id: int) -> list[PrefetchCandidate]:
        segment_id = int(segment_id)
        if segment_id not in self._dirty_segments and segment_id in self._ranked_cache_by_segment:
            return self._ranked_cache_by_segment[segment_id]
        entries = self.entries_by_segment.get(segment_id)
        if not entries:
            self._ranked_cache_by_segment.pop(segment_id, None)
            self._dirty_segments.discard(segment_id)
            return []
        age_penalty = float(getattr(self.config, "prefetch_age_penalty", 0.0))

        def static_rank(entry: PrefetchCandidate) -> tuple[float, int, int]:
            base_priority = compute_priority(
                source=entry.source,
                score_sum=entry.score_sum,
                activation_count=entry.activation_count,
                age=0,
                config=self.config,
            )
            age_adjusted = float(base_priority) + age_penalty * float(entry.last_seen_step)
            return (-age_adjusted, int(entry.layer_idx), int(entry.expert_idx))

        ranked = sorted(entries.values(), key=static_rank)
        self._ranked_cache_by_segment[segment_id] = ranked
        self._dirty_segments.discard(segment_id)
        return ranked

    def update_from_runtime_meta(
        self,
        runtime_meta: dict[int, LayerRuntimeMetaCPU] | None,
        source: str,
        step_id: int,
        layer_caches: dict[int, LayerExpertCache],
    ) -> dict[str, float]:
        stats = defaultdict(float)
        if not runtime_meta:
            return stats
        updated_segments: set[int] = set()

        for layer_idx, meta in runtime_meta.items():
            cache = layer_caches.get(layer_idx)
            if cache is None:
                continue
            aggregate_t0 = time.perf_counter()
            if meta.aggregated_expert_ids is None or meta.aggregated_score_sum is None or meta.aggregated_activation_count is None:
                if meta.selected_experts is None or meta.routing_weights is None:
                    continue
                flat_experts = meta.selected_experts.reshape(-1)
                if flat_experts.device.type != "cpu" or flat_experts.dtype != torch.int64:
                    flat_experts = flat_experts.to(device="cpu", dtype=torch.int64)
                if flat_experts.numel() == 0:
                    continue
                flat_weights = meta.routing_weights.reshape(-1)
                if flat_weights.device.type != "cpu" or flat_weights.dtype != torch.float32:
                    flat_weights = flat_weights.to(device="cpu", dtype=torch.float32)
                unique_ids, inverse = torch.unique(flat_experts, return_inverse=True)
                score_sum = torch.zeros((unique_ids.numel(),), dtype=torch.float32, device=torch.device("cpu"))
                score_sum.scatter_add_(0, inverse, flat_weights)
                counts = torch.zeros((unique_ids.numel(),), dtype=torch.int64, device=torch.device("cpu"))
                counts.scatter_add_(0, inverse, torch.ones_like(inverse, dtype=torch.int64))
            else:
                unique_ids = meta.aggregated_expert_ids
                if unique_ids.device.type != "cpu" or unique_ids.dtype != torch.int64:
                    unique_ids = unique_ids.to(device="cpu", dtype=torch.int64)
                score_sum = meta.aggregated_score_sum
                if score_sum.device.type != "cpu" or score_sum.dtype != torch.float32:
                    score_sum = score_sum.to(device="cpu", dtype=torch.float32)
                counts = meta.aggregated_activation_count
                if counts.device.type != "cpu" or counts.dtype != torch.int64:
                    counts = counts.to(device="cpu", dtype=torch.int64)
            stats["segment_index_aggregate_ms"] += (time.perf_counter() - aggregate_t0) * 1000.0
            if unique_ids.numel() == 0:
                continue

            filter_t0 = time.perf_counter()
            uncached_entries: list[tuple[int, float, int]] = []
            for expert_idx, new_score, new_count in zip(unique_ids.tolist(), score_sum.tolist(), counts.tolist()):
                expert_idx = int(expert_idx)
                if cache.is_cached_cpu(expert_idx) or cache.is_pending_cpu(expert_idx):
                    continue
                uncached_entries.append((expert_idx, float(new_score), int(new_count)))
            stats["segment_index_filter_ms"] += (time.perf_counter() - filter_t0) * 1000.0
            if not uncached_entries:
                continue

            update_t0 = time.perf_counter()
            segment_id = self._segment_id(int(layer_idx))
            segment_entries = self.entries_by_segment[segment_id]
            for expert_idx, new_score, new_count in uncached_entries:
                key = (int(layer_idx), int(expert_idx))
                if key in segment_entries:
                    entry = segment_entries[key]
                    decay = float(self.config.prefetch_history_decay)
                    entry.score_sum = decay * entry.score_sum + new_score
                    entry.activation_count = int(round(decay * entry.activation_count)) + new_count
                    entry.last_seen_step = int(step_id)
                    entry.source = source
                else:
                    entry = PrefetchCandidate(
                        layer_idx=int(layer_idx),
                        expert_idx=int(expert_idx),
                        source=source,
                        score_sum=new_score,
                        activation_count=new_count,
                        first_seen_step=int(step_id),
                        last_seen_step=int(step_id),
                        priority=0.0,
                    )
                    segment_entries[key] = entry
                age = max(0, int(step_id) - int(entry.last_seen_step))
                entry.priority = compute_priority(
                    source=entry.source,
                    score_sum=entry.score_sum,
                    activation_count=entry.activation_count,
                    age=age,
                    config=self.config,
                )
                stats["segment_index_candidate_count"] += 1.0
            self._mark_dirty(segment_id)
            updated_segments.add(segment_id)
            stats["segment_index_entry_update_ms"] += (time.perf_counter() - update_t0) * 1000.0
        if updated_segments:
            rebuild_t0 = time.perf_counter()
            for segment_id in updated_segments:
                self._ranked_entries_for_segment(segment_id)
            stats["segment_index_rank_cache_rebuild_count"] += float(len(updated_segments))
            stats["segment_index_rank_cache_rebuild_ms"] += (time.perf_counter() - rebuild_t0) * 1000.0
        return stats

    def ranked_candidates(
        self,
        *,
        segment_id: int,
        step_id: int,
        layer_caches: dict[int, LayerExpertCache],
        inflight_keys: set[tuple[int, int]],
    ) -> list[PrefetchCandidate]:
        segment_entries = self.entries_by_segment.get(int(segment_id))
        if not segment_entries:
            return []

        ttl = int(self.config.prefetch_history_ttl_steps)
        stale_keys = []
        ranked: list[PrefetchCandidate] = []
        for key, entry in segment_entries.items():
            layer_idx, expert_idx = key
            cache = layer_caches.get(layer_idx)
            if cache is None:
                stale_keys.append(key)
                continue
            if int(step_id) - int(entry.last_seen_step) > ttl:
                stale_keys.append(key)
                continue
            if key in inflight_keys or cache.is_cached_cpu(expert_idx) or cache.is_pending_cpu(expert_idx):
                continue
            age = max(0, int(step_id) - int(entry.last_seen_step))
            entry.priority = compute_priority(
                source=entry.source,
                score_sum=entry.score_sum,
                activation_count=entry.activation_count,
                age=age,
                config=self.config,
            )
            ranked.append(entry)
        for key in stale_keys:
            segment_entries.pop(key, None)
        if stale_keys:
            self._mark_dirty(segment_id)
        ranked.sort(key=lambda x: (-x.priority, x.layer_idx, x.expert_idx))
        return ranked

    def candidates(
        self,
        *,
        segment_id: int,
        step_id: int,
        layer_caches: dict[int, LayerExpertCache],
        inflight_keys: set[tuple[int, int]],
    ) -> list[PrefetchCandidate]:
        segment_entries = self.entries_by_segment.get(int(segment_id))
        if not segment_entries:
            return []

        ttl = int(self.config.prefetch_history_ttl_steps)
        stale_keys = []
        out: list[PrefetchCandidate] = []
        for key, entry in segment_entries.items():
            layer_idx, expert_idx = key
            cache = layer_caches.get(layer_idx)
            if cache is None:
                stale_keys.append(key)
                continue
            if int(step_id) - int(entry.last_seen_step) > ttl:
                stale_keys.append(key)
                continue
            if key in inflight_keys or cache.is_cached_cpu(expert_idx) or cache.is_pending_cpu(expert_idx):
                continue
            age = max(0, int(step_id) - int(entry.last_seen_step))
            entry.priority = compute_priority(
                source=entry.source,
                score_sum=entry.score_sum,
                activation_count=entry.activation_count,
                age=age,
                config=self.config,
            )
            out.append(entry)
        for key in stale_keys:
            segment_entries.pop(key, None)
        if stale_keys:
            self._mark_dirty(segment_id)
        return out

    def candidates_for_layer_range(
        self,
        *,
        layer_start: int,
        layer_end: int,
        step_id: int,
        layer_caches: dict[int, LayerExpertCache],
        inflight_keys: set[tuple[int, int]],
        max_candidates: int | None = None,
    ) -> list[PrefetchCandidate]:
        start = max(0, int(layer_start))
        end = max(start, int(layer_end))
        if start >= end:
            return []

        first_segment = self._segment_id(start)
        last_segment = self._segment_id(end - 1)
        limit = None if max_candidates is None else max(1, int(max_candidates))
        if limit is not None:
            selected: list[PrefetchCandidate] = []
            ttl = int(self.config.prefetch_history_ttl_steps)
            for segment_id in range(first_segment, last_segment + 1):
                segment_entries = self.entries_by_segment.get(int(segment_id))
                if not segment_entries:
                    continue
                stale_keys = []
                segment_selected = 0
                for entry in self._ranked_entries_for_segment(segment_id):
                    layer_idx = int(entry.layer_idx)
                    expert_idx = int(entry.expert_idx)
                    key = (layer_idx, expert_idx)
                    if segment_entries.get(key) is not entry:
                        self._mark_dirty(segment_id)
                        continue
                    if not (start <= int(layer_idx) < end):
                        continue
                    cache = layer_caches.get(layer_idx)
                    if cache is None:
                        stale_keys.append(key)
                        continue
                    age = max(0, int(step_id) - int(entry.last_seen_step))
                    if age > ttl:
                        stale_keys.append(key)
                        continue
                    if (
                        key in inflight_keys
                        or cache.is_cached_cpu(expert_idx)
                        or cache.is_pending_cpu(expert_idx)
                    ):
                        continue
                    priority = compute_priority(
                        source=entry.source,
                        score_sum=entry.score_sum,
                        activation_count=entry.activation_count,
                        age=age,
                        config=self.config,
                    )
                    selected.append(
                        PrefetchCandidate(
                            layer_idx=layer_idx,
                            expert_idx=expert_idx,
                            source=entry.source,
                            score_sum=entry.score_sum,
                            activation_count=entry.activation_count,
                            first_seen_step=entry.first_seen_step,
                            last_seen_step=entry.last_seen_step,
                            priority=priority,
                        )
                    )
                    segment_selected += 1
                    if segment_selected >= limit:
                        break
                for key in stale_keys:
                    segment_entries.pop(key, None)
                if stale_keys:
                    self._mark_dirty(segment_id)
            selected.sort(key=lambda candidate: (-candidate.priority, candidate.layer_idx, candidate.expert_idx))
            return selected[:limit]

        out: list[PrefetchCandidate] = []
        for segment_id in range(first_segment, last_segment + 1):
            candidates = self.candidates(
                segment_id=segment_id,
                step_id=step_id,
                layer_caches=layer_caches,
                inflight_keys=inflight_keys,
            )
            out.extend(
                candidate
                for candidate in candidates
                if start <= int(candidate.layer_idx) < end
            )
        return out


@dataclass
class DualQueueEntry:
    layer_idx: int
    expert_idx: int
    score_sum: float
    activation_count: float
    last_seen_round: int


class DualQueueSegmentIndex:
    """Segment index used only by the dual-queue runtime.

    Draft entries accumulate within one speculative round. Ground-truth
    entries retain normalized verify/prefill history with round-based decay.
    Cache residency is deliberately filtered at selection time, not update
    time, so a currently cached expert remains useful history after eviction.
    """

    def __init__(self, *, segment_size: int, history: bool, config: Config):
        self.segment_size = max(1, int(segment_size))
        self.history = bool(history)
        self.config = config
        self._decay = float(getattr(config, "dual_queue_ground_truth_decay", 0.9))
        self._count_weight = float(getattr(config, "dual_queue_ground_truth_count_weight", 0.1))
        self._ttl = int(getattr(config, "dual_queue_ground_truth_ttl_rounds", 64))
        self._decay_table = [self._decay ** i for i in range(self._ttl + 2)]
        self.entries_by_segment: dict[int, dict[tuple[int, int], DualQueueEntry]] = defaultdict(dict)

    def clear(self) -> None:
        self.entries_by_segment.clear()

    def _segment_id(self, layer_idx: int) -> int:
        return int(layer_idx) // self.segment_size

    @staticmethod
    def _aggregated(meta: LayerRuntimeMetaCPU) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if meta.aggregated_expert_ids is not None and meta.aggregated_score_sum is not None:
            expert_ids = meta.aggregated_expert_ids.to(device="cpu", dtype=torch.int64)
            counts = (
                meta.aggregated_activation_count.to(device="cpu", dtype=torch.float32)
                if meta.aggregated_activation_count is not None
                else torch.zeros(expert_ids.numel(), dtype=torch.float32)
            )
            return (
                expert_ids,
                meta.aggregated_score_sum.to(device="cpu", dtype=torch.float32),
                counts,
            )
        if meta.selected_experts is None or meta.routing_weights is None:
            return None
        flat_ids = meta.selected_experts.reshape(-1).to(device="cpu", dtype=torch.int64)
        if flat_ids.numel() == 0:
            return None
        flat_weights = meta.routing_weights.reshape(-1).to(device="cpu", dtype=torch.float32)
        unique_ids, inverse = torch.unique(flat_ids, return_inverse=True)
        score_sum = torch.zeros(unique_ids.numel(), dtype=torch.float32)
        score_sum.scatter_add_(0, inverse, flat_weights)
        counts = torch.zeros(unique_ids.numel(), dtype=torch.float32)
        counts.scatter_add_(0, inverse, torch.ones_like(inverse, dtype=torch.float32))
        return unique_ids, score_sum, counts

    def update(
        self,
        runtime_meta: dict[int, LayerRuntimeMetaCPU] | None,
        *,
        round_id: int,
    ) -> set[tuple[int, int]]:
        activated: set[tuple[int, int]] = set()
        if not runtime_meta:
            return activated
        decay = self._decay
        for layer_idx, meta in runtime_meta.items():
            aggregated = self._aggregated(meta)
            if aggregated is None:
                continue
            expert_ids, score_sums, counts = aggregated
            token_count = max(1, int(meta.token_count))
            segment_entries = self.entries_by_segment[self._segment_id(int(layer_idx))]
            for expert_idx, score_sum, count in zip(
                expert_ids.tolist(), score_sums.tolist(), counts.tolist(), strict=True
            ):
                key = (int(layer_idx), int(expert_idx))
                activated.add(key)
                new_score = float(score_sum)
                new_count = float(count)
                if self.history:
                    new_score /= token_count
                    new_count /= token_count
                entry = segment_entries.get(key)
                if entry is None:
                    segment_entries[key] = DualQueueEntry(
                        layer_idx=key[0],
                        expert_idx=key[1],
                        score_sum=new_score,
                        activation_count=new_count,
                        last_seen_round=int(round_id),
                    )
                    continue
                if self.history and entry.last_seen_round != int(round_id):
                    gap = max(1, int(round_id) - int(entry.last_seen_round))
                    factor = decay ** gap
                    entry.score_sum *= factor
                    entry.activation_count *= factor
                entry.score_sum += new_score
                entry.activation_count += new_count
                entry.last_seen_round = int(round_id)
        return activated

    def _priority(self, entry: DualQueueEntry, round_id: int) -> float:
        if not self.history:
            return float(entry.score_sum)
        age = max(0, int(round_id) - int(entry.last_seen_round))
        decay = self._decay_table[min(age, self._ttl + 1)]
        return decay * (float(entry.score_sum) + self._count_weight * float(entry.activation_count))

    def ranked_for_range(
        self,
        *,
        layer_start: int,
        layer_end: int,
        round_id: int,
        layer_caches: dict[int, LayerExpertCache],
        inflight_keys: set[tuple[int, int]] | dict,
        source: str,
    ) -> list[PrefetchCandidate]:
        start = int(layer_start)
        end = int(layer_end)
        if start >= end:
            return []
        ttl = self._ttl
        first_segment = self._segment_id(start)
        last_segment = self._segment_id(end - 1)
        ranked: list[PrefetchCandidate] = []
        for segment_id in range(first_segment, last_segment + 1):
            entries = self.entries_by_segment.get(segment_id)
            if not entries:
                continue
            stale: list[tuple[int, int]] = []
            for key, entry in entries.items():
                layer_idx, expert_idx = key
                if not (start <= layer_idx < end):
                    continue
                age = max(0, int(round_id) - int(entry.last_seen_round))
                if self.history and age > ttl:
                    stale.append(key)
                    continue
                cache = layer_caches.get(layer_idx)
                if (
                    cache is None
                    or key in inflight_keys
                    or cache.is_cached_cpu(expert_idx)
                    or cache.is_pending_cpu(expert_idx)
                ):
                    continue
                ranked.append(
                    PrefetchCandidate(
                        layer_idx=layer_idx,
                        expert_idx=expert_idx,
                        source=source,
                        score_sum=float(entry.score_sum),
                        activation_count=int(round(entry.activation_count)),
                        first_seen_step=int(entry.last_seen_round),
                        last_seen_step=int(entry.last_seen_round),
                        priority=self._priority(entry, int(round_id)),
                    )
                )
            for key in stale:
                entries.pop(key, None)
        ranked.sort(key=lambda item: (-item.priority, item.layer_idx, item.expert_idx))
        return ranked

    def priority_for(self, layer_idx: int, expert_idx: int, round_id: int) -> float:
        entries = self.entries_by_segment.get(self._segment_id(int(layer_idx)))
        if not entries:
            return 0.0
        entry = entries.get((int(layer_idx), int(expert_idx)))
        return 0.0 if entry is None else self._priority(entry, int(round_id))


_NON_DEFERRED_SOURCES = frozenset({
    "draft_direct_active",
    "verify_layer_predict",
    "predictive_phase1",
})


class PrefetchRuntime:
    def __init__(
        self,
        config: Config,
        layer_caches: dict[int, LayerExpertCache],
        cpu_expert_pool: dict[int, dict[int, dict[str, torch.Tensor]]],
        cache_strategy: CacheStrategy,
        prefetch_strategy: PrefetchStrategy,
        runtime_meta_recorder: ModelRuntimeMetaRecorder,
    ):
        self.config = config
        self.layer_caches = layer_caches
        self.cpu_expert_pool = cpu_expert_pool
        self.cache_strategy = cache_strategy
        self.prefetch_strategy = prefetch_strategy
        self.runtime_meta_recorder = runtime_meta_recorder
        # Draft-M3 cache-hit accounting is diagnostic-only: it never feeds a
        # candidate index, victim choice, or prefetch budget.  Keep its fairly
        # expensive per-layer torch.unique/list walk out of production runs.
        self._diagnostic_profile_enabled = bool(
            getattr(config, "engine_profile", False)
            or getattr(config, "spec_profile", False)
            or getattr(config, "transfer_aware_profile", False)
        )
        # Source lifecycle attribution is analysis-only.  It deliberately
        # shares the profile gate so production does not retain per-publish
        # residency objects or scan them on verify metadata consumption.
        self._source_lifecycle_profile_enabled = self._diagnostic_profile_enabled

        self.global_queue = GlobalWarmStartQueue(config)
        self.long_term_segment_index = SegmentCandidateIndex(config)
        self.draft_segment_index = SegmentCandidateIndex(config)
        self.verify_segment_index = SegmentCandidateIndex(config)
        self.verify_segment_index.segment_size = max(1, int(getattr(config, "verify_prefetch_segment_size", 12)))
        self.transfer_streams: list[torch.cuda.Stream | None] = []
        self.transfer_stream: torch.cuda.Stream | None = None
        self._next_transfer_stream_idx = 0
        self._initialize_transfer_stream_pool()
        self.metadata_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.publish_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        self.inflight: dict[tuple[int, int], PrefetchTicket] = {}
        self._profile = defaultdict(float)
        self._recent_published: dict[tuple[int, int], int] = {}
        self._recent_published_source: dict[tuple[int, int], str] = {}
        self._prefetch_active_residencies: dict[
            tuple[int, int], PrefetchResidency
        ] = {}
        self._prefetch_closed_residencies_by_key: dict[
            tuple[int, int], list[PrefetchResidency]
        ] = defaultdict(list)
        self._prefetch_residency_sequence = 0
        self._draft_direct_active_budget = self._initial_draft_direct_active_budget()
        self._draft_segment_indexed_budget = self._initial_draft_direct_active_budget()
        self._draft_iteration_open = False
        self._active_draft_iteration_steps: set[int] = set()
        self._draft_segment_indexed_submit_by_segment = defaultdict(int)
        self._draft_segment_indexed_ready_by_segment = defaultdict(int)
        self._draft_segment_indexed_success_by_segment = defaultdict(int)
        self._draft_segment_indexed_consumed_by_segment = defaultdict(int)
        self._prefetch_submit_count_by_source = defaultdict(int)
        self._prefetch_completed_count_by_source = defaultdict(int)
        self._prefetch_published_count_by_source = defaultdict(int)
        self._prefetch_late_count_by_source = defaultdict(int)
        self._prefetch_submitted_bytes_by_source = defaultdict(int)
        self._prefetch_completed_bytes_by_source = defaultdict(int)
        self._prefetch_published_bytes_by_source = defaultdict(int)
        self._prefetch_late_bytes_by_source = defaultdict(int)
        self._prefetch_transfer_enqueue_ms_by_source = defaultdict(float)
        self._prefetch_completion_latency_ms_by_source = defaultdict(float)
        self._prefetch_submit_count_by_stream = defaultdict(int)
        self._prefetch_submitted_bytes_by_stream = defaultdict(int)
        self._prefetch_transfer_enqueue_ms_by_stream = defaultdict(float)
        self._prefetch_completion_latency_ms_by_stream = defaultdict(float)
        self._transfer_lifecycle_events: list[dict[str, object]] = []
        self._draft_m3_layers_by_step: dict[int, dict[int, bool]] = {}
        self._draft_m3_step0_steps: set[int] = set()
        self._draft_m3_next_is_step0 = True

    def _initialize_transfer_stream_pool(self) -> None:
        stream_count = max(
            1,
            int(getattr(self.config, "prefetch_transfer_stream_count", 1)),
        )
        if torch.cuda.is_available():
            self.transfer_streams = [torch.cuda.Stream() for _ in range(stream_count)]
        else:
            self.transfer_streams = [None] * stream_count
        self.transfer_stream = self.transfer_streams[0]
        self._next_transfer_stream_idx = 0

    def _acquire_transfer_stream(self) -> tuple[int, torch.cuda.Stream | None]:
        stream_idx = self._next_transfer_stream_idx
        self._next_transfer_stream_idx = (stream_idx + 1) % len(self.transfer_streams)
        return stream_idx, self.transfer_streams[stream_idx]

    @staticmethod
    def _expert_weight_bytes(weights: dict[str, torch.Tensor]) -> int:
        return sum(
            int(weights[name].numel()) * int(weights[name].element_size())
            for name in ("gate_up", "down")
        )

    def _begin_prefetch_transfer(
        self,
        *,
        cache: LayerExpertCache,
        reservation: ActiveReservation | StagingReservation,
        weights: dict[str, torch.Tensor],
        source: str,
        direct_active: bool,
    ) -> tuple[object, float, float, int, int]:
        stream_idx, transfer_stream = self._acquire_transfer_stream()
        submit_ts_ms = time.perf_counter() * 1000.0
        enqueue_t0 = time.perf_counter()
        if direct_active:
            ready_event = cache.begin_async_put_to_active(
                reservation=reservation,
                gate_up_cpu=weights["gate_up"],
                down_cpu=weights["down"],
                stream=transfer_stream,
            )
        else:
            ready_event = cache.begin_async_put_to_staging(
                reservation=reservation,
                gate_up_cpu=weights["gate_up"],
                down_cpu=weights["down"],
                stream=transfer_stream,
            )
        enqueue_ms = (time.perf_counter() - enqueue_t0) * 1000.0
        num_bytes = self._expert_weight_bytes(weights)
        self._profile["prefetch_transfer_enqueue_ms"] += enqueue_ms
        self._profile[f"{source}_prefetch_transfer_enqueue_ms"] += enqueue_ms
        self._prefetch_transfer_enqueue_ms_by_source[source] += enqueue_ms
        self._prefetch_transfer_enqueue_ms_by_stream[stream_idx] += enqueue_ms
        return ready_event, submit_ts_ms, enqueue_ms, num_bytes, stream_idx

    def _record_prefetch_submitted(self, ticket: PrefetchTicket) -> None:
        source = str(ticket.source)
        num_bytes = int(ticket.num_bytes)
        self._profile["prefetch_submitted_bytes"] += num_bytes
        self._profile[f"{source}_prefetch_submitted_bytes"] += num_bytes
        self._prefetch_submit_count_by_source[source] += 1
        self._prefetch_submitted_bytes_by_source[source] += num_bytes
        self._prefetch_submit_count_by_stream[int(ticket.transfer_stream_idx)] += 1
        self._prefetch_submitted_bytes_by_stream[int(ticket.transfer_stream_idx)] += num_bytes
        self._profile["prefetch_max_inflight_observed"] = max(
            float(self._profile.get("prefetch_max_inflight_observed", 0.0)),
            float(len(self.inflight)),
        )
        if (
            ticket.direct_active
            and ticket.source in _NON_DEFERRED_SOURCES
            and int(ticket.active_slot_prev_expert) >= 0
        ):
            # Non-deferred reservations unmap the victim before the H2D starts.
            self._record_prefetch_residency_evicted(
                layer_idx=int(ticket.layer_idx),
                expert_idx=int(ticket.active_slot_prev_expert),
                step_id=int(ticket.step_id),
                replacement_source=str(ticket.source),
            )
        self._record_transfer_lifecycle("submit", ticket)

    def _record_prefetch_completed(self, ticket: PrefetchTicket) -> None:
        source = str(ticket.source)
        num_bytes = int(ticket.num_bytes)
        latency_ms = max(0.0, time.perf_counter() * 1000.0 - float(ticket.submit_ts_ms))
        self._profile["prefetch_completed_bytes"] += num_bytes
        self._profile["prefetch_completion_latency_ms"] += latency_ms
        self._profile[f"{source}_prefetch_completed_bytes"] += num_bytes
        self._profile[f"{source}_prefetch_completion_latency_ms"] += latency_ms
        self._prefetch_completed_count_by_source[source] += 1
        self._prefetch_completed_bytes_by_source[source] += num_bytes
        self._prefetch_completion_latency_ms_by_source[source] += latency_ms
        self._prefetch_completion_latency_ms_by_stream[int(ticket.transfer_stream_idx)] += latency_ms
        self._record_transfer_lifecycle("ready", ticket)

    def _record_prefetch_published(self, ticket: PrefetchTicket) -> None:
        source = str(ticket.source)
        num_bytes = int(ticket.num_bytes)
        self._profile["prefetch_published_bytes"] += num_bytes
        self._profile[f"{source}_prefetch_published_bytes"] += num_bytes
        self._prefetch_published_count_by_source[source] += 1
        self._prefetch_published_bytes_by_source[source] += num_bytes
        self._record_transfer_lifecycle("publish", ticket)

    def _record_prefetch_late(self, ticket: PrefetchTicket) -> None:
        source = str(ticket.source)
        num_bytes = int(ticket.num_bytes)
        self._profile["prefetch_late_bytes"] += num_bytes
        self._profile[f"{source}_prefetch_late_bytes"] += num_bytes
        self._prefetch_late_count_by_source[source] += 1
        self._prefetch_late_bytes_by_source[source] += num_bytes
        self._record_transfer_lifecycle("late", ticket)

    def _record_transfer_lifecycle(
        self,
        event: str,
        ticket: PrefetchTicket,
    ) -> None:
        if not bool(getattr(self.config, "transfer_aware_profile", False)):
            return
        now_ms = time.perf_counter() * 1000.0
        self._transfer_lifecycle_events.append(
            {
                "event": str(event),
                "timestamp_ms": float(now_ms),
                "latency_from_submit_ms": max(
                    0.0, now_ms - float(ticket.submit_ts_ms)
                ),
                "step_id": int(ticket.step_id),
                "layer_idx": int(ticket.layer_idx),
                "expert_idx": int(ticket.expert_idx),
                "source": str(ticket.source),
                "direct_active": bool(ticket.direct_active),
                "active_slot_idx": int(ticket.active_slot_idx),
                "active_slot_prev_expert": int(
                    ticket.active_slot_prev_expert
                ),
                "segment_id": int(ticket.segment_id),
                "num_bytes": int(ticket.num_bytes),
                "transfer_stream_idx": int(ticket.transfer_stream_idx),
            }
        )

    def _record_prefetch_residency_published(
        self,
        ticket: PrefetchTicket,
        *,
        step_id: int,
    ) -> None:
        key = (int(ticket.layer_idx), int(ticket.expert_idx))
        self._recent_published[key] = int(step_id)
        self._recent_published_source[key] = str(ticket.source)
        if not bool(getattr(self, "_source_lifecycle_profile_enabled", False)):
            return

        # A duplicate publication should normally be filtered before submit.
        # Closing it here keeps telemetry bounded and makes the anomaly visible
        # without changing cache or admission behavior.
        if key in self._prefetch_active_residencies:
            self._record_prefetch_residency_evicted(
                layer_idx=key[0],
                expert_idx=key[1],
                step_id=int(step_id),
                replacement_source="republish",
            )
            self._profile["prefetch_source_republish_count"] += 1

        self._prefetch_residency_sequence += 1
        self._prefetch_active_residencies[key] = PrefetchResidency(
            sequence=int(self._prefetch_residency_sequence),
            source=str(ticket.source),
            publish_step=int(step_id),
            num_bytes=int(ticket.num_bytes),
        )

    def _record_prefetch_residency_evicted(
        self,
        *,
        layer_idx: int,
        expert_idx: int,
        step_id: int,
        replacement_source: str,
    ) -> None:
        if not bool(getattr(self, "_source_lifecycle_profile_enabled", False)):
            return
        key = (int(layer_idx), int(expert_idx))
        residency = self._prefetch_active_residencies.pop(key, None)
        if residency is None:
            return
        residency.evict_step = int(step_id)
        residency.replacement_source = str(replacement_source)
        self._prefetch_closed_residencies_by_key[key].append(residency)
        if bool(getattr(self.config, "transfer_aware_profile", False)):
            self._transfer_lifecycle_events.append(
                {
                    "event": "evict",
                    "step_id": int(step_id),
                    "layer_idx": key[0],
                    "expert_idx": key[1],
                    "source": residency.source,
                    "replacement_source": str(replacement_source),
                    "publish_step": int(residency.publish_step),
                    "residence_steps": max(
                        0, int(step_id) - int(residency.publish_step)
                    ),
                    "consumed": bool(residency.consume_count > 0),
                    "num_bytes": int(residency.num_bytes),
                }
            )

    def _find_prefetch_residency(
        self,
        *,
        layer_idx: int,
        expert_idx: int,
        step_id: int,
    ) -> PrefetchResidency | None:
        key = (int(layer_idx), int(expert_idx))
        candidates: list[PrefetchResidency] = []
        # Prefer a residency closed in the same step: segment metadata can be
        # consumed asynchronously after a later segment has replaced its hit.
        for residency in reversed(
            self._prefetch_closed_residencies_by_key.get(key, ())
        ):
            if (
                int(residency.publish_step) <= int(step_id)
                and int(residency.evict_step) >= int(step_id)
            ):
                candidates.append(residency)
                break
            if int(residency.evict_step) < int(step_id):
                break
        active = self._prefetch_active_residencies.get(key)
        if active is not None and int(active.publish_step) <= int(step_id):
            candidates.append(active)
        if len(candidates) > 1:
            self._profile["prefetch_source_ambiguous_consume_count"] += 1
        return candidates[0] if candidates else None

    def _record_source_prefetch_consumed(
        self,
        *,
        layer_idx: int,
        expert_idx: int,
        step_id: int,
        was_cached_at_execution: bool,
    ) -> None:
        if (
            not bool(getattr(self, "_source_lifecycle_profile_enabled", False))
            or not was_cached_at_execution
        ):
            return
        residency = self._find_prefetch_residency(
            layer_idx=int(layer_idx),
            expert_idx=int(expert_idx),
            step_id=int(step_id),
        )
        if residency is None:
            key = (int(layer_idx), int(expert_idx))
            if key in self._recent_published:
                self._profile["prefetch_source_unattributed_consumed_count"] += 1
            return
        if int(residency.last_consume_step) == int(step_id):
            return
        residency.last_consume_step = int(step_id)
        residency.consume_count += 1
        if residency.first_consume_step < 0:
            residency.first_consume_step = int(step_id)
            if bool(getattr(self.config, "transfer_aware_profile", False)):
                self._transfer_lifecycle_events.append(
                    {
                        "event": "first_consume",
                        "step_id": int(step_id),
                        "layer_idx": int(layer_idx),
                        "expert_idx": int(expert_idx),
                        "source": residency.source,
                        "publish_step": int(residency.publish_step),
                        "latency_steps": max(
                            0, int(step_id) - int(residency.publish_step)
                        ),
                        "num_bytes": int(residency.num_bytes),
                    }
                )

    def _prune_unmapped_prefetch_residencies(self, *, step_id: int) -> None:
        if not bool(getattr(self, "_source_lifecycle_profile_enabled", False)):
            return
        for layer_idx, expert_idx in list(self._prefetch_active_residencies):
            cache = self.layer_caches.get(int(layer_idx))
            if cache is not None and cache.is_cached_cpu(int(expert_idx)):
                continue
            self._record_prefetch_residency_evicted(
                layer_idx=int(layer_idx),
                expert_idx=int(expert_idx),
                step_id=int(step_id),
                replacement_source="external",
            )

    def _source_lifecycle_profile(self) -> dict[str, object]:
        metric_names = (
            "consumed_count",
            "first_consumed_count",
            "first_consumed_bytes",
            "first_consume_latency_steps",
            "first_consume_latency_steps_max",
            "evicted_count",
            "evicted_bytes",
            "evicted_after_consume_count",
            "evicted_without_consume_count",
            "evicted_without_consume_bytes",
            "residence_steps",
            "resident_count",
            "resident_consumed_count",
        )
        metrics: dict[str, defaultdict[str, int]] = {
            name: defaultdict(int) for name in metric_names
        }
        replacement_pairs: defaultdict[str, int] = defaultdict(int)

        closed = [
            residency
            for rows in self._prefetch_closed_residencies_by_key.values()
            for residency in rows
        ]
        active = list(self._prefetch_active_residencies.values())
        for residency in closed + active:
            source = str(residency.source)
            metrics["consumed_count"][source] += int(residency.consume_count)
            if residency.first_consume_step >= 0:
                metrics["first_consumed_count"][source] += 1
                metrics["first_consumed_bytes"][source] += int(residency.num_bytes)
                latency = max(
                    0,
                    int(residency.first_consume_step)
                    - int(residency.publish_step),
                )
                metrics["first_consume_latency_steps"][source] += latency
                metrics["first_consume_latency_steps_max"][source] = max(
                    metrics["first_consume_latency_steps_max"][source],
                    latency,
                )
        for residency in closed:
            source = str(residency.source)
            metrics["evicted_count"][source] += 1
            metrics["evicted_bytes"][source] += int(residency.num_bytes)
            metrics["residence_steps"][source] += max(
                0, int(residency.evict_step) - int(residency.publish_step)
            )
            pair = f"{source}->{residency.replacement_source or 'unknown'}"
            replacement_pairs[pair] += 1
            if residency.consume_count > 0:
                metrics["evicted_after_consume_count"][source] += 1
            else:
                metrics["evicted_without_consume_count"][source] += 1
                metrics["evicted_without_consume_bytes"][source] += int(
                    residency.num_bytes
                )
        for residency in active:
            source = str(residency.source)
            metrics["resident_count"][source] += 1
            if residency.consume_count > 0:
                metrics["resident_consumed_count"][source] += 1

        out: dict[str, object] = {
            f"prefetch_{name}_by_source": dict(values)
            for name, values in metrics.items()
        }
        out["prefetch_eviction_count_by_source_pair"] = dict(replacement_pairs)
        out["prefetch_source_tracked_residency_count"] = len(closed) + len(active)
        out["prefetch_source_closed_residency_count"] = len(closed)
        out["prefetch_source_active_residency_count"] = len(active)
        out["prefetch_source_ambiguous_consume_count"] = int(
            self._profile.get("prefetch_source_ambiguous_consume_count", 0.0)
        )
        out["prefetch_source_unattributed_consumed_count"] = int(
            self._profile.get("prefetch_source_unattributed_consumed_count", 0.0)
        )
        out["prefetch_source_republish_count"] = int(
            self._profile.get("prefetch_source_republish_count", 0.0)
        )
        return out

    def _initial_draft_direct_active_budget(self) -> int:
        return max(
            0,
            int(getattr(self.config, "draft_prefetch_max_per_boundary", getattr(self.config, "prefetch_step_budget", 0))),
        )

    def _segment_indexed_enabled(self) -> bool:
        return str(getattr(self.config, "prefetch_runtime_mode", "baseline_staging")) == "draft_segment_indexed"

    def _segment_id_for_layer(self, layer_idx: int) -> int:
        return int(layer_idx) // int(self.draft_segment_index.segment_size)

    def _draft_segment_indexed_target_segment_id(self, frontier_layer_idx: int | None) -> int | None:
        frontier = None if frontier_layer_idx is None else int(frontier_layer_idx)
        if frontier is None:
            if not self.layer_caches:
                return None
            frontier = max(int(layer_idx) for layer_idx in self.layer_caches.keys())
        return self._segment_id_for_layer(int(frontier))

    def _rollback_round_loaded_prefetch(self, layer_idx: int, expert_idx: int) -> None:
        return None

    def _mark_draft_m3_step_start(self, step_id: int) -> None:
        if not self._diagnostic_profile_enabled:
            return
        sid = int(step_id)
        if self._draft_m3_next_is_step0:
            self._draft_m3_step0_steps.add(sid)
            self._draft_m3_next_is_step0 = False

    def _record_draft_m3_cache_hits(
        self,
        runtime_meta: dict[int, LayerRuntimeMetaCPU] | None,
        step_id: int,
    ) -> None:
        if (
            not self._diagnostic_profile_enabled
            or not runtime_meta
            or not self.layer_caches
        ):
            return
        sid = int(step_id)
        layer_hits = self._draft_m3_layers_by_step.setdefault(sid, {})
        for layer_idx, meta in runtime_meta.items():
            li = int(layer_idx)
            cache = self.layer_caches.get(li)
            if cache is None:
                continue
            if meta.aggregated_expert_ids is not None:
                expert_ids = meta.aggregated_expert_ids.reshape(-1)
            elif meta.selected_experts is not None:
                expert_ids = meta.selected_experts.reshape(-1)
            else:
                continue
            if expert_ids.numel() <= 0:
                continue
            unique_ids = torch.unique(expert_ids.to(device=torch.device("cpu"), dtype=torch.int64))
            layer_hits[li] = all(cache.is_cached_cpu(int(eid)) for eid in unique_ids.tolist())

        expected_layers = len(self.layer_caches)
        if len(layer_hits) < expected_layers:
            return
        perfect = all(layer_hits.get(int(layer_idx), False) for layer_idx in self.layer_caches)
        self._profile["draft_m3_group_count"] += 1.0
        if perfect:
            self._profile["draft_m3_perfect_count"] += 1.0
        if sid in self._draft_m3_step0_steps:
            self._profile["draft_m3_step0_group_count"] += 1.0
            if perfect:
                self._profile["draft_m3_step0_perfect_count"] += 1.0
            self._draft_m3_step0_steps.discard(sid)
        self._draft_m3_layers_by_step.pop(sid, None)

    @staticmethod
    def _format_segment_counts(counts: defaultdict[int, int] | dict[int, int]) -> dict[str, int]:
        return {str(int(segment_id)): int(value) for segment_id, value in sorted(counts.items())}

    def begin_draft_iteration(self, step_id: int) -> None:
        if not self._segment_indexed_enabled():
            return
        if not self._draft_iteration_open:
            self.draft_segment_index.clear()
            self._active_draft_iteration_steps.clear()
            self._mark_draft_m3_step_start(step_id)
            self._draft_iteration_open = True
        self._active_draft_iteration_steps.add(int(step_id))

    def end_draft_iteration(self) -> None:
        if not self._segment_indexed_enabled():
            return
        self.draft_segment_index.clear()
        self._active_draft_iteration_steps.clear()
        self._draft_iteration_open = False

    def observe_runtime_meta(
        self,
        runtime_meta: dict[int, LayerRuntimeMetaCPU] | None,
        source: str,
        step_id: int,
        *,
        update_global_queue: bool = True,
        segment_index: SegmentCandidateIndex | None = None,
    ) -> dict[str, float]:
        if runtime_meta is None:
            return {}
        observe_t0 = time.perf_counter()
        mark_access_t0 = time.perf_counter()
        for layer_idx, meta in runtime_meta.items():
            cache = self.layer_caches.get(layer_idx)
            if cache is None:
                continue
            if meta.aggregated_expert_ids is not None:
                cache.mark_access_aggregated(
                    meta.aggregated_expert_ids,
                    meta.aggregated_activation_count,
                    meta.aggregated_score_sum,
                    step_id=step_id,
                )
            elif meta.selected_experts is not None:
                cache.mark_access(meta.selected_experts, meta.routing_weights, step_id=step_id)
        mark_access_ms = (time.perf_counter() - mark_access_t0) * 1000.0

        queue_update_t0 = time.perf_counter()
        queue_stats = defaultdict(float)
        if update_global_queue:
            queue_stats.update(
                self.global_queue.update_from_runtime_meta(
                    runtime_meta=runtime_meta,
                    source=source,
                    step_id=step_id,
                    layer_caches=self.layer_caches,
                )
            )
        if segment_index is not None:
            segment_stats = segment_index.update_from_runtime_meta(
                runtime_meta=runtime_meta,
                source=source,
                step_id=step_id,
                layer_caches=self.layer_caches,
            )
            for key, value in segment_stats.items():
                queue_stats[key] += float(value)
        queue_update_ms = (time.perf_counter() - queue_update_t0) * 1000.0
        self._profile["observe_runtime_meta_count"] += 1.0
        self._profile["observe_runtime_meta_ms"] += (time.perf_counter() - observe_t0) * 1000.0
        self._profile["observe_mark_access_ms"] += mark_access_ms
        self._profile["observe_queue_update_ms"] += queue_update_ms
        for key, value in queue_stats.items():
            self._profile[key] += float(value)
        out = {
            "mark_access_ms": mark_access_ms,
            "queue_update_ms": queue_update_ms,
            "queue_aggregate_ms": float(queue_stats.get("queue_aggregate_ms", 0.0)),
            "queue_filter_ms": float(queue_stats.get("queue_filter_ms", 0.0)),
            "queue_entry_update_ms": float(queue_stats.get("queue_entry_update_ms", 0.0)),
            "segment_index_aggregate_ms": float(queue_stats.get("segment_index_aggregate_ms", 0.0)),
            "segment_index_filter_ms": float(queue_stats.get("segment_index_filter_ms", 0.0)),
            "segment_index_entry_update_ms": float(queue_stats.get("segment_index_entry_update_ms", 0.0)),
            "segment_index_rank_cache_rebuild_ms": float(
                queue_stats.get("segment_index_rank_cache_rebuild_ms", 0.0)
            ),
            "segment_index_rank_cache_rebuild_count": float(
                queue_stats.get("segment_index_rank_cache_rebuild_count", 0.0)
            ),
        }
        return out

    def observe_prefill(self, runtime_meta: dict[int, LayerRuntimeMetaCPU] | None, step_id: int) -> dict[str, float]:
        if bool(self.config.prefetch_use_prefill_history):
            return self.observe_runtime_meta(
                runtime_meta,
                source="prefill_history",
                step_id=step_id,
                segment_index=self.long_term_segment_index if self._segment_indexed_enabled() else None,
            )
        return {}

    def observe_draft(self, runtime_meta: dict[int, LayerRuntimeMetaCPU] | None, step_id: int) -> dict[str, float]:
        if bool(self.config.prefetch_use_draft_live):
            if self._segment_indexed_enabled():
                active_steps = self._active_draft_iteration_steps
                is_active = not active_steps or int(step_id) in active_steps
                if not is_active:
                    self._profile["draft_segment_indexed_stale_metadata_observe_count"] += 1
                else:
                    self._record_draft_m3_cache_hits(runtime_meta, step_id)
                return self.observe_runtime_meta(
                    runtime_meta,
                    source="draft_live",
                    step_id=step_id,
                    update_global_queue=False,
                    segment_index=self.draft_segment_index if is_active else None,
                )
            return self.observe_runtime_meta(runtime_meta, source="draft_live", step_id=step_id)
        return {}

    def observe_verify(self, runtime_meta: dict[int, LayerRuntimeMetaCPU] | None, step_id: int) -> dict[str, float]:
        rank_guard_t0 = time.perf_counter()
        self._update_rank_guard_scores(runtime_meta)
        rank_guard_ms = (time.perf_counter() - rank_guard_t0) * 1000.0
        segment_index_ms = 0.0
        if runtime_meta is not None:
            segment_index_t0 = time.perf_counter()
            self.verify_segment_index.update_from_runtime_meta(
                runtime_meta, "verify_history", step_id, self.layer_caches,
            )
            segment_index_ms = (time.perf_counter() - segment_index_t0) * 1000.0
        self._profile["observe_verify_rank_guard_ms"] += rank_guard_ms
        self._profile["observe_verify_segment_index_ms"] += segment_index_ms
        if bool(self.config.prefetch_use_verify_history):
            runtime_t0 = time.perf_counter()
            out = self.observe_runtime_meta(
                runtime_meta,
                source="verify_history",
                step_id=step_id,
                segment_index=self.long_term_segment_index if self._segment_indexed_enabled() else None,
            )
            runtime_ms = (time.perf_counter() - runtime_t0) * 1000.0
            self._profile["observe_verify_runtime_meta_call_ms"] += runtime_ms
            out["observe_verify_rank_guard_ms"] = rank_guard_ms
            out["observe_verify_segment_index_ms"] = segment_index_ms
            out["observe_verify_runtime_meta_call_ms"] = runtime_ms
            return out
        return {}

    def _update_rank_guard_scores(
        self,
        runtime_meta: dict[int, LayerRuntimeMetaCPU] | None,
    ) -> None:
        if not isinstance(self.cache_strategy, LFURankGuardStrategy):
            return
        if not runtime_meta:
            return
        for layer_idx, meta in runtime_meta.items():
            if meta.selected_experts is not None:
                self.cache_strategy.update_rank_scores_from_routing(
                    int(layer_idx), meta.selected_experts,
                )

    def submit_from_global_queue(self, step_id: int, phase: str) -> int:
        _ = phase
        if int(self.config.prefetch_step_budget) <= 0:
            return 0

        inflight_keys = set(self.inflight.keys())
        ranked = self.global_queue.ranked_candidates(
            step_id=step_id,
            layer_caches=self.layer_caches,
            inflight_keys=inflight_keys,
        )
        ranked = self.prefetch_strategy.rank(ranked, step_id=step_id)

        submitted = 0
        max_submit = max(0, int(self.config.prefetch_step_budget))
        inflight_budget = max(0, int(self.config.prefetch_max_inflight) - len(self.inflight))
        dispatch_budget = min(max_submit, inflight_budget)

        for candidate in ranked:
            if submitted >= dispatch_budget:
                break

            layer_idx = int(candidate.layer_idx)
            expert_idx = int(candidate.expert_idx)
            key = (layer_idx, expert_idx)
            cache = self.layer_caches.get(layer_idx)
            if cache is None:
                continue
            if cache.is_cached_cpu(expert_idx):
                continue
            if key in self.inflight:
                continue

            reservation = cache.reserve_staging_slot(expert_idx)
            if reservation is None:
                continue
            reservation.layer_idx = layer_idx

            weights = self.cpu_expert_pool.get(layer_idx, {}).get(expert_idx)
            if not weights or "gate_up" not in weights or "down" not in weights:
                cache.cancel_staging_reservation(reservation)
                continue

            ready_event, submit_ts_ms, enqueue_ms, num_bytes, stream_idx = self._begin_prefetch_transfer(
                cache=cache,
                reservation=reservation,
                weights=weights,
                source=str(candidate.source),
                direct_active=False,
            )

            ticket = PrefetchTicket(
                step_id=step_id,
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                source=candidate.source,
                staging_slot_idx=reservation.staging_slot_idx,
                staging_generation=reservation.generation,
                submit_ts_ms=submit_ts_ms,
                ready_event=ready_event,
                ready=False,
                num_bytes=num_bytes,
                transfer_enqueue_ms=enqueue_ms,
                transfer_stream_idx=stream_idx,
            )
            self.inflight[key] = ticket
            self._record_prefetch_submitted(ticket)
            submitted += 1
            self._profile["prefetch_submit_count"] += 1
            self._profile["staging_prefetch_submit_count"] += 1
            if candidate.source == "prefill_history":
                self._profile["history_prefetch_submit_count"] += 1
            elif candidate.source == "verify_history":
                self._profile["verify_history_prefetch_submit_count"] += 1
            elif candidate.source == "draft_live":
                self._profile["draft_live_prefetch_submit_count"] += 1

        return submitted

    def _record_source_submit(self, source: str) -> None:
        if source == "prefill_history":
            self._profile["history_prefetch_submit_count"] += 1
        elif source == "verify_history":
            self._profile["verify_history_prefetch_submit_count"] += 1
        elif source == "draft_live":
            self._profile["draft_live_prefetch_submit_count"] += 1

    def _adjust_draft_direct_active_budget(self, visible_ms: float) -> None:
        min_budget = max(0, int(getattr(self.config, "draft_prefetch_min_per_boundary", 0)))
        max_budget = max(
            min_budget,
            int(getattr(self.config, "draft_prefetch_max_per_boundary", self.config.prefetch_step_budget)),
        )
        current = max(min_budget, min(max_budget, int(self._draft_direct_active_budget)))
        budget_ms = float(getattr(self.config, "draft_prefetch_visible_budget_ms", 3.0))
        if budget_ms <= 0.0:
            self._draft_direct_active_budget = current
            self._profile["draft_direct_active_prefetch_adaptive_budget"] = float(current)
            return

        if visible_ms > budget_ms and current > min_budget:
            current -= 1
            self._profile["draft_direct_active_prefetch_budget_decrease_count"] += 1
        elif visible_ms < budget_ms * 0.5 and current < max_budget:
            current += 1
            self._profile["draft_direct_active_prefetch_budget_increase_count"] += 1
        self._draft_direct_active_budget = current
        self._profile["draft_direct_active_prefetch_adaptive_budget"] = float(current)

    def submit_draft_direct_active_prefetch(
        self,
        *,
        step_id: int,
        phase: str,
        frontier_layer_idx: int | None,
        visible_budget_ms: float | None = None,
    ) -> int:
        _ = phase
        if int(self.config.prefetch_step_budget) <= 0:
            return 0

        submit_t0 = time.perf_counter()
        min_submit = max(0, int(getattr(self.config, "draft_prefetch_min_per_boundary", 0)))
        configured_max = max(
            min_submit,
            int(getattr(self.config, "draft_prefetch_max_per_boundary", self.config.prefetch_step_budget)),
        )
        adaptive_budget = max(min_submit, min(configured_max, int(self._draft_direct_active_budget)))
        max_submit = min(max(0, int(self.config.prefetch_step_budget)), adaptive_budget)
        inflight_budget = max(0, int(self.config.prefetch_max_inflight) - len(self.inflight))
        if max_submit <= 0:
            self._profile["draft_direct_active_prefetch_skipped_by_budget_count"] += 1
            self._adjust_draft_direct_active_budget(visible_ms=0.0)
            return 0
        if inflight_budget <= 0:
            self._profile["draft_direct_active_prefetch_skipped_by_pending_count"] += 1
            return 0
        dispatch_budget = min(max_submit, inflight_budget)

        frontier = None if frontier_layer_idx is None else int(frontier_layer_idx)
        transfer_budget_ms = (
            float(visible_budget_ms)
            if visible_budget_ms is not None
            else float(getattr(self.config, "draft_prefetch_visible_budget_ms", 3.0))
        )
        inflight_keys = set(self.inflight.keys())
        ranked = self.global_queue.ranked_candidates(
            step_id=step_id,
            layer_caches=self.layer_caches,
            inflight_keys=inflight_keys,
            max_layer_idx=frontier,
        )
        ranked = self.prefetch_strategy.rank(ranked, step_id=step_id)

        submitted = 0
        used_transfer_ms = 0.0

        for candidate in ranked:
            if submitted >= dispatch_budget:
                break

            layer_idx = int(candidate.layer_idx)
            expert_idx = int(candidate.expert_idx)
            if frontier is not None and layer_idx > frontier:
                self._profile["draft_direct_active_prefetch_skipped_by_frontier_count"] += 1
                continue

            key = (layer_idx, expert_idx)
            cache = self.layer_caches.get(layer_idx)
            if cache is None:
                continue
            if cache.is_cached_cpu(expert_idx):
                continue
            if key in self.inflight:
                self._profile["draft_direct_active_prefetch_skipped_by_pending_count"] += 1
                continue

            weights = self.cpu_expert_pool.get(layer_idx, {}).get(expert_idx)
            if not weights or "gate_up" not in weights or "down" not in weights:
                continue

            transfer_ms = self._estimated_expert_transfer_ms(weights)
            if not isfinite(transfer_ms):
                continue
            if (
                transfer_budget_ms > 0.0
                and submitted >= min_submit
                and used_transfer_ms + transfer_ms > transfer_budget_ms
            ):
                self._profile["draft_direct_active_prefetch_skipped_by_budget_count"] += 1
                break

            victim_slot = self._select_publish_slot(
                cache,
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                step_id=step_id,
            )
            if victim_slot is None:
                self._profile["draft_direct_active_prefetch_skipped_by_pending_count"] += 1
                continue

            reservation = cache.reserve_active_slot_for_prefetch(
                layer_idx=layer_idx,
                active_slot_idx=victim_slot,
                expert_idx=expert_idx,
            )
            if reservation is None:
                self._profile["draft_direct_active_prefetch_skipped_by_pending_count"] += 1
                continue

            ready_event, submit_ts_ms, enqueue_ms, num_bytes, stream_idx = self._begin_prefetch_transfer(
                cache=cache,
                reservation=reservation,
                weights=weights,
                source="draft_direct_active",
                direct_active=True,
            )

            ticket = PrefetchTicket(
                step_id=step_id,
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                source="draft_direct_active",
                staging_slot_idx=-1,
                staging_generation=-1,
                submit_ts_ms=submit_ts_ms,
                ready_event=ready_event,
                ready=False,
                direct_active=True,
                active_slot_idx=reservation.active_slot_idx,
                active_generation=reservation.generation,
                active_slot_prev_expert=int(getattr(reservation, "prev_expert", -1)),
                num_bytes=num_bytes,
                transfer_enqueue_ms=enqueue_ms,
                transfer_stream_idx=stream_idx,
            )
            self.inflight[key] = ticket
            self._record_prefetch_submitted(ticket)
            submitted += 1
            used_transfer_ms += transfer_ms
            self._profile["prefetch_submit_count"] += 1
            self._profile["direct_active_prefetch_submit_count"] += 1
            self._profile["draft_direct_active_prefetch_submit_count"] += 1
            self._profile["draft_direct_active_prefetch_est_transfer_ms"] += transfer_ms
            self._record_source_submit(candidate.source)

        visible_ms = (time.perf_counter() - submit_t0) * 1000.0
        self._profile["draft_direct_active_prefetch_visible_overhead_ms"] += visible_ms
        self._profile["draft_direct_active_prefetch_used_transfer_budget_ms"] += used_transfer_ms
        self._adjust_draft_direct_active_budget(visible_ms)
        return submitted

    def submit_draft_segment_indexed_prefetch(
        self,
        *,
        step_id: int,
        phase: str,
        frontier_layer_idx: int | None,
        visible_budget_ms: float | None = None,
    ) -> int:
        _ = phase
        if int(self.config.prefetch_step_budget) <= 0:
            return 0

        submit_t0 = time.perf_counter()
        min_submit = max(0, int(getattr(self.config, "draft_prefetch_min_per_boundary", 0)))
        configured_max = max(
            min_submit,
            int(getattr(self.config, "draft_prefetch_max_per_boundary", self.config.prefetch_step_budget)),
        )
        adaptive_budget = max(min_submit, min(configured_max, int(self._draft_segment_indexed_budget)))
        max_submit = min(max(0, int(self.config.prefetch_step_budget)), adaptive_budget)
        inflight_budget = max(0, int(self.config.prefetch_max_inflight) - len(self.inflight))
        if max_submit <= 0:
            self._profile["draft_segment_indexed_prefetch_skipped_by_budget_count"] += 1
            self._adjust_draft_segment_indexed_budget(visible_ms=0.0)
            return 0
        if inflight_budget <= 0:
            self._profile["draft_segment_indexed_prefetch_skipped_by_pending_count"] += 1
            return 0
        dispatch_budget = min(max_submit, inflight_budget)

        segment_id = self._draft_segment_indexed_target_segment_id(frontier_layer_idx)
        if segment_id is None:
            return 0
        transfer_budget_ms = (
            float(visible_budget_ms)
            if visible_budget_ms is not None
            else float(getattr(self.config, "draft_prefetch_visible_budget_ms", 3.0))
        )
        inflight_keys = set(self.inflight.keys())
        rank_t0 = time.perf_counter()
        ranked_by_key: dict[tuple[int, int], PrefetchCandidate] = {}
        for index in (self.long_term_segment_index, self.draft_segment_index):
            candidates = index.candidates(
                segment_id=segment_id,
                step_id=step_id,
                layer_caches=self.layer_caches,
                inflight_keys=inflight_keys,
            )
            self._profile["draft_segment_indexed_candidate_scan_count"] += 1
            self._profile["draft_segment_indexed_candidate_ranked_count"] += len(candidates)
            for candidate in candidates:
                key = (int(candidate.layer_idx), int(candidate.expert_idx))
                prev = ranked_by_key.get(key)
                if prev is None or candidate.priority > prev.priority:
                    ranked_by_key[key] = candidate
        ranked = sorted(ranked_by_key.values(), key=lambda x: (-x.priority, x.layer_idx, x.expert_idx))
        self._profile["draft_segment_indexed_candidate_merge_count"] += len(ranked)
        self._profile["draft_segment_indexed_rank_ms"] += (time.perf_counter() - rank_t0) * 1000.0

        submitted = 0
        used_transfer_ms = 0.0
        for candidate in ranked:
            if submitted >= dispatch_budget:
                break

            layer_idx = int(candidate.layer_idx)
            expert_idx = int(candidate.expert_idx)
            key = (layer_idx, expert_idx)
            cache = self.layer_caches.get(layer_idx)
            if cache is None:
                continue
            if cache.is_cached_cpu(expert_idx) or cache.is_pending_cpu(expert_idx):
                continue
            if key in self.inflight:
                self._profile["draft_segment_indexed_prefetch_skipped_by_pending_count"] += 1
                continue

            weights = self.cpu_expert_pool.get(layer_idx, {}).get(expert_idx)
            if not weights or "gate_up" not in weights or "down" not in weights:
                continue

            transfer_ms = self._estimated_expert_transfer_ms(weights)
            if not isfinite(transfer_ms):
                continue
            if (
                transfer_budget_ms > 0.0
                and submitted >= min_submit
                and used_transfer_ms + transfer_ms > transfer_budget_ms
            ):
                self._profile["draft_segment_indexed_prefetch_skipped_by_budget_count"] += 1
                break

            victim_t0 = time.perf_counter()
            victim_slot = self._select_publish_slot_cpu(
                cache,
                expert_idx=expert_idx,
                step_id=step_id,
                layer_idx=layer_idx,
            )
            self._profile["draft_segment_indexed_victim_select_ms"] += (time.perf_counter() - victim_t0) * 1000.0
            if victim_slot is None:
                self._rollback_round_loaded_prefetch(layer_idx, expert_idx)
                self._profile["draft_segment_indexed_prefetch_skipped_by_pending_count"] += 1
                continue

            reservation_t0 = time.perf_counter()
            reservation = cache.reserve_active_slot_for_prefetch_deferred(
                layer_idx=layer_idx,
                active_slot_idx=victim_slot,
                expert_idx=expert_idx,
            )
            self._profile["draft_segment_indexed_prefetch_reservation_ms"] += (
                time.perf_counter() - reservation_t0
            ) * 1000.0
            if reservation is None:
                self._rollback_round_loaded_prefetch(layer_idx, expert_idx)
                self._profile["draft_segment_indexed_prefetch_skipped_by_pending_count"] += 1
                continue

            ready_event, submit_ts_ms, enqueue_ms, num_bytes, stream_idx = self._begin_prefetch_transfer(
                cache=cache,
                reservation=reservation,
                weights=weights,
                source="draft_segment_indexed",
                direct_active=True,
            )

            ticket = PrefetchTicket(
                step_id=step_id,
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                source="draft_segment_indexed",
                staging_slot_idx=-1,
                staging_generation=-1,
                submit_ts_ms=submit_ts_ms,
                ready_event=ready_event,
                ready=False,
                direct_active=True,
                active_slot_idx=reservation.active_slot_idx,
                active_generation=reservation.generation,
                active_slot_prev_expert=int(getattr(reservation, "prev_expert", -1)),
                segment_id=segment_id,
                num_bytes=num_bytes,
                transfer_enqueue_ms=enqueue_ms,
                transfer_stream_idx=stream_idx,
            )
            self.inflight[key] = ticket
            self._record_prefetch_submitted(ticket)
            submitted += 1
            used_transfer_ms += transfer_ms
            inflight_keys.add(key)
            self._profile["prefetch_submit_count"] += 1
            self._profile["direct_active_prefetch_submit_count"] += 1
            self._profile["draft_segment_indexed_prefetch_submit_count"] += 1
            self._profile["draft_segment_indexed_prefetch_est_transfer_ms"] += transfer_ms
            self._draft_segment_indexed_submit_by_segment[int(segment_id)] += 1
            self._record_source_submit(candidate.source)

        visible_ms = (time.perf_counter() - submit_t0) * 1000.0
        self._profile["draft_segment_indexed_prefetch_visible_overhead_ms"] += visible_ms
        self._profile["draft_segment_indexed_prefetch_used_transfer_budget_ms"] += used_transfer_ms
        self._adjust_draft_segment_indexed_budget(visible_ms)
        return submitted

    def _adjust_draft_segment_indexed_budget(self, visible_ms: float) -> None:
        min_budget = max(0, int(getattr(self.config, "draft_prefetch_min_per_boundary", 0)))
        max_budget = max(
            min_budget,
            int(getattr(self.config, "draft_prefetch_max_per_boundary", self.config.prefetch_step_budget)),
        )
        current = max(min_budget, min(max_budget, int(self._draft_segment_indexed_budget)))
        budget_ms = float(getattr(self.config, "draft_prefetch_visible_budget_ms", 3.0))
        if budget_ms <= 0.0:
            self._draft_segment_indexed_budget = current
            self._profile["draft_segment_indexed_prefetch_adaptive_budget"] = float(current)
            return

        if visible_ms > budget_ms and current > min_budget:
            current -= 1
            self._profile["draft_segment_indexed_prefetch_budget_decrease_count"] += 1
        elif visible_ms < budget_ms * 0.5 and current < max_budget:
            current += 1
            self._profile["draft_segment_indexed_prefetch_budget_increase_count"] += 1
        self._draft_segment_indexed_budget = current
        self._profile["draft_segment_indexed_prefetch_adaptive_budget"] = float(current)

    def submit_verify_segment_prefetch(
        self,
        *,
        step_id: int,
        target_layer_start: int,
        target_layer_end: int,
        visible_budget_ms: float | None = None,
    ) -> int:
        """Submit prefetch for a verify segment's layers. Reads from draft-prepared candidate index."""
        submit_t0 = time.perf_counter()
        self._profile["verify_segment_prefetch_call_count"] += 1
        if int(self.config.prefetch_step_budget) <= 0:
            self._profile["verify_segment_prefetch_skipped_by_budget_count"] += 1
            return 0

        target_start = int(target_layer_start)
        target_end = int(target_layer_end)
        target_seg = self.verify_segment_index._segment_id(target_start)

        min_submit = max(0, int(getattr(self.config, "verify_prefetch_min_per_boundary", 0)))
        configured_max = max(
            min_submit,
            int(getattr(self.config, "verify_prefetch_max_per_boundary", 16)),
        )
        max_submit = min(max(0, int(self.config.prefetch_step_budget)), configured_max)
        inflight_budget = max(0, int(self.config.prefetch_max_inflight) - len(self.inflight))
        if max_submit <= 0:
            self._profile["verify_segment_prefetch_skipped_by_budget_count"] += 1
            return 0
        if inflight_budget <= 0:
            self._profile["verify_segment_prefetch_skipped_by_pending_count"] += 1
            return 0
        dispatch_budget = min(max_submit, inflight_budget)

        transfer_budget_ms = float(
            visible_budget_ms
            if visible_budget_ms is not None
            else getattr(self.config, "verify_prefetch_visible_budget_ms", 12.0)
        )
        inflight_keys = set(self.inflight.keys())
        self._profile["verify_segment_prefetch_dispatch_budget_sum"] += float(dispatch_budget)
        self._profile["verify_segment_prefetch_inflight_budget_sum"] += float(inflight_budget)
        rank_multiplier_raw = os.getenv("NANOVLLM_VERIFY_PREFETCH_RANK_MULTIPLIER", "1").strip()
        try:
            rank_multiplier = int(rank_multiplier_raw)
        except ValueError:
            rank_multiplier = 1
        rank_limit = None
        if rank_multiplier > 0:
            rank_limit = max(dispatch_budget, dispatch_budget * rank_multiplier)
            self._profile["verify_segment_prefetch_rank_limited_count"] += 1
            self._profile["verify_segment_prefetch_rank_limit_sum"] += float(rank_limit)

        rank_t0 = time.perf_counter()
        ranked_by_key: dict[tuple[int, int], PrefetchCandidate] = {}
        scan_t0 = time.perf_counter()
        for index in (self.draft_segment_index, self.verify_segment_index, self.long_term_segment_index):
            candidates = index.candidates_for_layer_range(
                layer_start=target_start,
                layer_end=target_end,
                step_id=step_id,
                layer_caches=self.layer_caches,
                inflight_keys=inflight_keys,
                max_candidates=rank_limit,
            )
            self._profile["verify_segment_prefetch_candidate_scan_count"] += 1
            self._profile["verify_segment_prefetch_candidate_ranked_count"] += len(candidates)
            for candidate in candidates:
                key = (int(candidate.layer_idx), int(candidate.expert_idx))
                prev = ranked_by_key.get(key)
                if prev is None or candidate.priority > prev.priority:
                    ranked_by_key[key] = candidate
        self._profile["verify_segment_prefetch_rank_scan_ms"] += (time.perf_counter() - scan_t0) * 1000.0
        sort_t0 = time.perf_counter()
        ranked = sorted(ranked_by_key.values(), key=lambda x: (-x.priority, x.layer_idx, x.expert_idx))
        self._profile["verify_segment_prefetch_rank_sort_ms"] += (time.perf_counter() - sort_t0) * 1000.0
        self._profile["verify_segment_prefetch_candidate_merge_count"] += len(ranked)
        self._profile["verify_segment_prefetch_rank_ms"] += (time.perf_counter() - rank_t0) * 1000.0
        if not ranked:
            self._profile["verify_segment_prefetch_no_candidate_count"] += 1

        submitted = 0
        used_transfer_ms = 0.0
        loop_t0 = time.perf_counter()
        filter_ms = 0.0
        victim_select_ms = 0.0
        reservation_ms = 0.0
        transfer_call_ms = 0.0
        bookkeeping_ms = 0.0
        for candidate in ranked:
            if submitted >= dispatch_budget:
                break

            filter_t0 = time.perf_counter()
            layer_idx = int(candidate.layer_idx)
            expert_idx = int(candidate.expert_idx)
            key = (layer_idx, expert_idx)
            cache = self.layer_caches.get(layer_idx)
            if cache is None:
                filter_ms += (time.perf_counter() - filter_t0) * 1000.0
                continue
            if cache.is_cached_cpu(expert_idx) or cache.is_pending_cpu(expert_idx):
                filter_ms += (time.perf_counter() - filter_t0) * 1000.0
                continue
            if key in self.inflight:
                filter_ms += (time.perf_counter() - filter_t0) * 1000.0
                continue

            weights = self.cpu_expert_pool.get(layer_idx, {}).get(expert_idx)
            if not weights or "gate_up" not in weights or "down" not in weights:
                self._profile["verify_segment_prefetch_missing_weights_count"] += 1
                filter_ms += (time.perf_counter() - filter_t0) * 1000.0
                continue

            transfer_ms = self._estimated_expert_transfer_ms(weights)
            if not isfinite(transfer_ms):
                self._profile["verify_segment_prefetch_skipped_by_budget_count"] += 1
                filter_ms += (time.perf_counter() - filter_t0) * 1000.0
                continue
            if (
                transfer_budget_ms > 0.0
                and submitted >= min_submit
                and used_transfer_ms + transfer_ms > transfer_budget_ms
            ):
                self._profile["verify_segment_prefetch_skipped_by_budget_count"] += 1
                filter_ms += (time.perf_counter() - filter_t0) * 1000.0
                break
            filter_ms += (time.perf_counter() - filter_t0) * 1000.0

            victim_t0 = time.perf_counter()
            victim_slot = self._select_publish_slot_cpu(
                cache, expert_idx=expert_idx, step_id=step_id, layer_idx=layer_idx,
            )
            victim_select_ms += (time.perf_counter() - victim_t0) * 1000.0
            if victim_slot is None:
                self._rollback_round_loaded_prefetch(layer_idx, expert_idx)
                self._profile["verify_segment_prefetch_skipped_by_pending_count"] += 1
                continue

            reservation_t0 = time.perf_counter()
            reservation = cache.reserve_active_slot_for_prefetch_deferred(
                layer_idx=layer_idx, active_slot_idx=victim_slot, expert_idx=expert_idx,
            )
            reservation_ms += (time.perf_counter() - reservation_t0) * 1000.0
            if reservation is None:
                self._rollback_round_loaded_prefetch(layer_idx, expert_idx)
                self._profile["verify_segment_prefetch_skipped_by_pending_count"] += 1
                continue

            transfer_call_t0 = time.perf_counter()
            ready_event, submit_ts_ms, enqueue_ms, num_bytes, stream_idx = self._begin_prefetch_transfer(
                cache=cache, reservation=reservation, weights=weights,
                source="verify_segment", direct_active=True,
            )
            transfer_call_ms += (time.perf_counter() - transfer_call_t0) * 1000.0

            bookkeeping_t0 = time.perf_counter()
            ticket = PrefetchTicket(
                step_id=step_id,
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                source="verify_segment",
                staging_slot_idx=-1,
                staging_generation=-1,
                submit_ts_ms=submit_ts_ms,
                ready_event=ready_event,
                ready=False,
                direct_active=True,
                active_slot_idx=reservation.active_slot_idx,
                active_generation=reservation.generation,
                active_slot_prev_expert=int(getattr(reservation, "prev_expert", -1)),
                segment_id=target_seg,
                num_bytes=num_bytes,
                transfer_enqueue_ms=enqueue_ms,
                transfer_stream_idx=stream_idx,
            )
            self.inflight[key] = ticket
            self._record_prefetch_submitted(ticket)
            submitted += 1
            used_transfer_ms += transfer_ms
            inflight_keys.add(key)
            self._profile["prefetch_submit_count"] += 1
            self._profile["direct_active_prefetch_submit_count"] += 1
            self._profile["verify_segment_prefetch_submit_count"] += 1
            self._profile["verify_segment_prefetch_est_transfer_ms"] += transfer_ms
            self._record_source_submit(candidate.source)
            bookkeeping_ms += (time.perf_counter() - bookkeeping_t0) * 1000.0

        self._profile["verify_segment_prefetch_loop_ms"] += (time.perf_counter() - loop_t0) * 1000.0
        self._profile["verify_segment_prefetch_filter_ms"] += filter_ms
        self._profile["verify_segment_prefetch_victim_select_ms"] += victim_select_ms
        self._profile["verify_segment_prefetch_reservation_ms"] += reservation_ms
        self._profile["verify_segment_prefetch_begin_transfer_call_ms"] += transfer_call_ms
        self._profile["verify_segment_prefetch_bookkeeping_ms"] += bookkeeping_ms
        self._profile["verify_segment_prefetch_used_transfer_budget_ms"] += used_transfer_ms
        self._profile["verify_segment_prefetch_visible_overhead_ms"] += (
            time.perf_counter() - submit_t0
        ) * 1000.0
        return submitted

    def _estimated_expert_transfer_ms(self, weights: dict[str, torch.Tensor]) -> float:
        bandwidth_gbps = float(getattr(self.config, "prefetch_verify_layer_transfer_bandwidth_gbps", 12.0))
        if bandwidth_gbps <= 0.0:
            return float("inf")
        gate_up = weights.get("gate_up")
        down = weights.get("down")
        if gate_up is None or down is None:
            return float("inf")
        num_bytes = int(gate_up.numel() * gate_up.element_size() + down.numel() * down.element_size())
        return (float(num_bytes) / (bandwidth_gbps * 1_000_000_000.0)) * 1000.0

    def _select_publish_slot_cpu(
        self,
        cache: LayerExpertCache,
        *,
        expert_idx: int,
        step_id: int,
        layer_idx: int | None = None,
    ) -> int | None:
        _ = expert_idx, step_id
        for slot_idx, slot_expert in enumerate(cache.slot_to_expert):
            if int(slot_expert) < 0 and not cache.is_active_slot_pending(slot_idx):
                return slot_idx

        strategy_name = str(getattr(self.config, "cache_strategy", "lru")).strip().lower()

        if (
            strategy_name == "lfu_rankguard"
            and isinstance(self.cache_strategy, LFURankGuardStrategy)
            and layer_idx is not None
        ):
            return self.cache_strategy.select_victim_slot_cpu(
                cache.slot_to_expert,
                cache.access_count,
                int(layer_idx),
                cache.is_active_slot_pending,
            )

        best_slot = None
        best_value = None
        for slot_idx, slot_expert in enumerate(cache.slot_to_expert):
            if cache.is_active_slot_pending(slot_idx):
                continue
            if int(slot_expert) < 0:
                return slot_idx
            expert = int(slot_expert)
            if strategy_name in ("lfu", "lfu_rankguard"):
                value = int(cache.access_count[expert])
            else:
                value = int(cache.last_access_step[expert])
            if best_value is None or value < best_value:
                best_value = value
                best_slot = slot_idx
        return best_slot

    def _select_publish_slot(
        self,
        cache: LayerExpertCache,
        *,
        layer_idx: int,
        expert_idx: int,
        step_id: int,
    ) -> int | None:
        snapshot = cache.snapshot(layer_idx=layer_idx)
        victim_slot = self.cache_strategy.select_victim_slot(
            snapshot=snapshot,
            incoming_expert_idx=expert_idx,
            step_id=step_id,
        )
        if victim_slot is not None and not cache.is_active_slot_pending(int(victim_slot)):
            return int(victim_slot)

        slot_to_expert = snapshot.slot_to_expert_lut.tolist()
        for slot_idx, slot_expert in enumerate(slot_to_expert):
            if slot_expert < 0 and not cache.is_active_slot_pending(slot_idx):
                return slot_idx
        for slot_idx, _ in enumerate(slot_to_expert):
            if not cache.is_active_slot_pending(slot_idx):
                return slot_idx
        return None

    def _select_empty_publish_slot(
        self,
        cache: LayerExpertCache,
    ) -> int | None:
        """Select an empty (expert=-1), non-pending slot without evicting.

        Used by verify-layer prefetch to avoid changing cache residency,
        which would break deterministic output between prefetch-ON and
        prefetch-OFF runs.
        """
        # Use live lookup, not snapshot, to match current GPU LUT state.
        slot_to_expert = cache.slot_to_expert_lut.tolist()
        for slot_idx, slot_expert in enumerate(slot_to_expert):
            if int(slot_expert) < 0 and not cache.is_active_slot_pending(slot_idx):
                return slot_idx
        return None

    def submit_verify_layer_prefetch(
        self,
        *,
        step_id: int,
        target_layer_idx: int,
        available_ms: float,
    ) -> int:
        # Debug instrumentation (activated by NANOVLLM_DEBUG_PREFETCH_SUBMIT=1)
        _dbg_enabled = int(os.environ.get("NANOVLLM_DEBUG_PREFETCH_SUBMIT", "0")) > 0
        _dbg = None
        if _dbg_enabled:
            _dbg = getattr(self, "_vls_dbg", None)
            if _dbg is None:
                self._vls_dbg = {
                    "call": 0, "disabled": 0, "budget_zero": 0, "avail_zero": 0,
                    "no_cache": 0, "inflight_full": 0, "no_candidates": 0,
                    "max_submit": 0, "slot_none": 0, "reserve_fail": 0,
                    "submitted": 0, "budget_stop": 0, "cached_cpu": 0,
                    "in_flight_skip": 0, "no_weights": 0,
                    "ranked_total": 0, "ranked_after_filter": 0,
                    "global_q_size": 0, "avail_ms_sum": 0.0,
                }
                _dbg = self._vls_dbg
            _dbg["call"] += 1
            _dbg["global_q_size"] = len(self.global_queue.entries)
            _dbg["avail_ms_sum"] += float(available_ms)

        if not bool(getattr(self.config, "prefetch_verify_layer_enabled", True)):
            if _dbg_enabled: _dbg["disabled"] += 1
            return 0
        if int(self.config.prefetch_step_budget) <= 0:
            if _dbg_enabled: _dbg["budget_zero"] += 1
            return 0
        if available_ms <= 0.0:
            if _dbg_enabled: _dbg["avail_zero"] += 1
            return 0

        layer_idx = int(target_layer_idx)
        cache = self.layer_caches.get(layer_idx)
        if cache is None:
            if _dbg_enabled: _dbg["no_cache"] += 1
            return 0

        self.publish_direct_active_ready(step_id=step_id)
        inflight_budget = max(0, int(self.config.prefetch_max_inflight) - len(self.inflight))
        if inflight_budget <= 0:
            if _dbg_enabled: _dbg["inflight_full"] += 1
            return 0

        inflight_keys = set(self.inflight.keys())
        ranked = self.global_queue.ranked_candidates(
            step_id=step_id,
            layer_caches=self.layer_caches,
            inflight_keys=inflight_keys,
        )
        if _dbg_enabled: _dbg["ranked_total"] += len(ranked)
        ranked = [c for c in self.prefetch_strategy.rank(ranked, step_id=step_id) if int(c.layer_idx) == layer_idx]
        if _dbg_enabled: _dbg["ranked_after_filter"] += len(ranked)
        if not ranked:
            if _dbg_enabled: _dbg["no_candidates"] += 1
            return 0

        max_submit = min(
            max(0, int(self.config.prefetch_step_budget)),
            max(0, int(getattr(self.config, "prefetch_verify_layer_max_budget", self.config.prefetch_step_budget))),
            inflight_budget,
        )
        if _dbg_enabled: _dbg["max_submit"] += max_submit
        submitted = 0
        used_budget_ms = 0.0

        for candidate in ranked:
            if submitted >= max_submit:
                break
            expert_idx = int(candidate.expert_idx)
            key = (layer_idx, expert_idx)
            if cache.is_cached_cpu(expert_idx):
                if _dbg_enabled: _dbg["cached_cpu"] += 1
                continue
            if key in self.inflight:
                if _dbg_enabled: _dbg["in_flight_skip"] += 1
                continue

            weights = self.cpu_expert_pool.get(layer_idx, {}).get(expert_idx)
            if not weights or "gate_up" not in weights or "down" not in weights:
                if _dbg_enabled: _dbg["no_weights"] += 1
                continue

            transfer_ms = self._estimated_expert_transfer_ms(weights)
            if not isfinite(transfer_ms):
                continue
            if used_budget_ms + transfer_ms > available_ms:
                self._profile["verify_layer_prefetch_budget_stop_count"] += 1
                if _dbg_enabled: _dbg["budget_stop"] += 1
                break

            victim_slot = self._select_publish_slot(
                cache,
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                step_id=step_id,
            )
            if victim_slot is None:
                if _dbg_enabled: _dbg["slot_none"] += 1
                continue

            reservation = cache.reserve_active_slot_for_prefetch(
                layer_idx=layer_idx,
                active_slot_idx=victim_slot,
                expert_idx=expert_idx,
            )
            if reservation is None:
                if _dbg_enabled: _dbg["reserve_fail"] += 1
                continue

            ready_event, submit_ts_ms, enqueue_ms, num_bytes, stream_idx = self._begin_prefetch_transfer(
                cache=cache,
                reservation=reservation,
                weights=weights,
                source="verify_layer_predict",
                direct_active=True,
            )

            ticket = PrefetchTicket(
                step_id=step_id,
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                source="verify_layer_predict",
                staging_slot_idx=-1,
                staging_generation=-1,
                submit_ts_ms=submit_ts_ms,
                ready_event=ready_event,
                ready=False,
                direct_active=True,
                active_slot_idx=reservation.active_slot_idx,
                active_generation=reservation.generation,
                active_slot_prev_expert=int(getattr(reservation, "prev_expert", -1)),
                num_bytes=num_bytes,
                transfer_enqueue_ms=enqueue_ms,
                transfer_stream_idx=stream_idx,
            )
            self.inflight[key] = ticket
            self._record_prefetch_submitted(ticket)
            submitted += 1
            used_budget_ms += transfer_ms
            if _dbg_enabled: _dbg["submitted"] += 1
            self._profile["prefetch_submit_count"] += 1
            self._profile["direct_active_prefetch_submit_count"] += 1
            self._profile["verify_layer_prefetch_submit_count"] += 1
            self._profile["verify_layer_prefetch_est_transfer_ms"] += transfer_ms
            self._profile["verify_layer_prefetch_available_ms"] += float(available_ms)
            self._profile["verify_layer_prefetch_used_budget_ms"] += used_budget_ms

        return submitted

    def poll_ready_tickets(self) -> list[PrefetchTicket]:
        ready: list[PrefetchTicket] = []
        for key, ticket in list(self.inflight.items()):
            if ticket.direct_active:
                continue
            if ticket.ready:
                ready.append(ticket)
                continue
            if not bool(ticket.ready_event.query()):
                continue

            cache = self.layer_caches[ticket.layer_idx]
            reservation = StagingReservation(
                layer_idx=ticket.layer_idx,
                staging_slot_idx=ticket.staging_slot_idx,
                expert_idx=ticket.expert_idx,
                generation=ticket.staging_generation,
            )
            ok = cache.mark_staging_ready(reservation)
            if not ok:
                self.inflight.pop(key, None)
                self._profile["prefetch_late_count"] += 1
                self._record_prefetch_late(ticket)
                continue

            ticket.ready = True
            ready.append(ticket)
            self._profile["prefetch_completed_count"] += 1
            self._record_prefetch_completed(ticket)
        return ready

    def publish_ready(self, step_id: int, max_publish: int | None = None) -> int:
        t0 = time.perf_counter()
        ready = self.poll_ready_tickets()
        if not ready:
            return 0

        publish_budget = int(self.config.cache_eviction_budget_per_step)
        if max_publish is not None:
            publish_budget = min(publish_budget, int(max_publish))

        published = 0
        for ticket in ready:
            if publish_budget >= 0 and published >= publish_budget:
                break

            cache = self.layer_caches[ticket.layer_idx]
            victim_slot = self._select_publish_slot(
                cache,
                layer_idx=ticket.layer_idx,
                expert_idx=ticket.expert_idx,
                step_id=step_id,
            )
            if victim_slot is None:
                continue

            reservation = StagingReservation(
                layer_idx=ticket.layer_idx,
                staging_slot_idx=ticket.staging_slot_idx,
                expert_idx=ticket.expert_idx,
                generation=ticket.staging_generation,
            )
            published_item = cache.publish_ready_staging_to_active(
                reservation=reservation,
                active_slot_idx=victim_slot,
                stream=self.publish_stream,
            )
            if published_item is None:
                self.inflight.pop((ticket.layer_idx, ticket.expert_idx), None)
                self._profile["prefetch_late_count"] += 1
                self._record_prefetch_late(ticket)
                continue

            prev_expert = int(cache.slot_to_expert[published_item.active_slot_idx])
            self._finalize_publish(cache, published_item)
            self._record_prefetch_residency_evicted(
                layer_idx=int(ticket.layer_idx),
                expert_idx=prev_expert,
                step_id=int(step_id),
                replacement_source=str(ticket.source),
            )
            self.inflight.pop((ticket.layer_idx, ticket.expert_idx), None)
            self._record_prefetch_residency_published(ticket, step_id=int(step_id))
            published += 1
            self._profile["publish_count"] += 1
            self._profile["staging_prefetch_publish_count"] += 1
            self._record_prefetch_published(ticket)

        publish_ms = (time.perf_counter() - t0) * 1000.0
        self._profile["publish_ms"] += publish_ms
        self._profile["staging_prefetch_publish_ms"] += publish_ms
        return published

    def publish_direct_active_ready(
        self,
        step_id: int,
        *,
        layer_idx: int | None = None,
        source: str | None = None,
    ) -> int:
        t0 = time.perf_counter()
        published = 0
        for key, ticket in list(self.inflight.items()):
            if not ticket.direct_active:
                continue
            if layer_idx is not None and int(ticket.layer_idx) != int(layer_idx):
                continue
            if source is not None and str(ticket.source) != str(source):
                continue
            self._profile["direct_active_prefetch_ready_scan_count"] += 1
            if not bool(ticket.ready_event.query()):
                continue
            self._profile["direct_active_prefetch_ready_count"] += 1
            self._profile["prefetch_completed_count"] += 1
            self._record_prefetch_completed(ticket)
            if ticket.source == "draft_direct_active":
                self._profile["draft_direct_active_prefetch_ready_count"] += 1
            elif ticket.source == "draft_segment_indexed":
                self._profile["draft_segment_indexed_prefetch_ready_count"] += 1
                if int(ticket.segment_id) >= 0:
                    self._draft_segment_indexed_ready_by_segment[int(ticket.segment_id)] += 1
            elif ticket.source == "verify_layer_predict":
                self._profile["verify_layer_prefetch_ready_count"] += 1
            cache = self.layer_caches.get(ticket.layer_idx)
            if cache is None:
                self.inflight.pop(key, None)
                self._profile["prefetch_late_count"] += 1
                self._record_prefetch_late(ticket)
                continue
            reservation = ActiveReservation(
                layer_idx=ticket.layer_idx,
                active_slot_idx=ticket.active_slot_idx,
                expert_idx=ticket.expert_idx,
                generation=ticket.active_generation,
                prev_expert=ticket.active_slot_prev_expert,
            )
            _non_deferred = ticket.source in _NON_DEFERRED_SOURCES
            if _non_deferred:
                published_item = cache.commit_active_prefetch(reservation)
            else:
                published_item = cache.commit_deferred_active_prefetch(reservation)
            self.inflight.pop(key, None)
            if published_item is None:
                if not _non_deferred:
                    cache.cancel_deferred_active_prefetch(reservation)
                self._profile["prefetch_late_count"] += 1
                self._record_prefetch_late(ticket)
                continue
            if not _non_deferred:
                self._record_prefetch_residency_evicted(
                    layer_idx=int(ticket.layer_idx),
                    expert_idx=int(ticket.active_slot_prev_expert),
                    step_id=int(step_id),
                    replacement_source=str(ticket.source),
                )
            self._record_prefetch_residency_published(ticket, step_id=int(step_id))
            published += 1
            self._profile["publish_count"] += 1
            self._profile["direct_active_prefetch_publish_count"] += 1
            self._record_prefetch_published(ticket)
            if ticket.source == "draft_direct_active":
                self._profile["draft_direct_active_prefetch_publish_count"] += 1
            elif ticket.source == "draft_segment_indexed":
                self._profile["draft_segment_indexed_prefetch_publish_count"] += 1
                if int(ticket.segment_id) >= 0:
                    self._draft_segment_indexed_success_by_segment[int(ticket.segment_id)] += 1
            elif ticket.source == "verify_layer_predict":
                self._profile["verify_layer_prefetch_publish_count"] += 1
        publish_ms = (time.perf_counter() - t0) * 1000.0
        self._profile["publish_ms"] += publish_ms
        self._profile["direct_active_prefetch_publish_ms"] += publish_ms
        return published

    def drain_direct_active_ready(self, step_id: int, *, source: str | None = None) -> int:
        t0 = time.perf_counter()
        waited = 0
        draft_waited = 0
        indexed_waited = 0
        sync_wait_ms = 0.0
        for ticket in list(self.inflight.values()):
            if not ticket.direct_active:
                continue
            if source is not None and str(ticket.source) != str(source):
                continue
            sync_t0 = time.perf_counter()
            ticket.ready_event.synchronize()
            ticket_sync_ms = (time.perf_counter() - sync_t0) * 1000.0
            sync_wait_ms += ticket_sync_ms
            self._profile[
                f"{ticket.source}_prefetch_sync_wait_ms"
            ] += ticket_sync_ms
            waited += 1
            if ticket.source == "draft_direct_active":
                draft_waited += 1
            elif ticket.source == "draft_segment_indexed":
                indexed_waited += 1
        published = self.publish_direct_active_ready(step_id=step_id, source=source)
        drain_ms = (time.perf_counter() - t0) * 1000.0
        self._profile["direct_active_prefetch_drain_count"] += waited
        self._profile["direct_active_prefetch_drain_ms"] += drain_ms
        self._profile["direct_active_prefetch_sync_wait_ms"] += sync_wait_ms
        if source == "draft_direct_active" or source is None:
            self._profile["draft_direct_active_prefetch_drain_count"] += draft_waited
            self._profile["draft_direct_active_prefetch_drain_ms"] += drain_ms
        if source == "draft_segment_indexed" or source is None:
            self._profile["draft_segment_indexed_prefetch_drain_count"] += indexed_waited
            self._profile["draft_segment_indexed_prefetch_drain_ms"] += drain_ms
        return published

    def _finalize_publish(self, cache: LayerExpertCache, published_item: PublishedExpert) -> None:
        if self.publish_stream is not None:
            torch.cuda.current_stream().wait_stream(self.publish_stream)
        cache.commit_published_expert(published_item)

    def wait_for_verify(self, step_id: int, timeout_ms: float) -> None:
        t0 = time.perf_counter()
        self.publish_direct_active_ready(step_id=step_id)
        self.publish_ready(step_id=step_id)
        self._profile["verify_ready_before_wait_count"] += self._count_ready_relevant_experts()

        transfer_wait_ms = 0.0
        if timeout_ms > 0.0:
            deadline = t0 + timeout_ms / 1000.0
            transfer_wait_t0 = (
                time.perf_counter() if self.inflight else None
            )
            while time.perf_counter() < deadline:
                published = self.publish_direct_active_ready(step_id=step_id)
                if published > 0 or not self.inflight:
                    break
                published = self.publish_ready(step_id=step_id)
                if published > 0 or not self.inflight:
                    break
                time.sleep(0.0002)
            if self.inflight:
                self._profile["prefetch_timeout_count"] += 1
            if transfer_wait_t0 is not None:
                transfer_wait_ms = (
                    time.perf_counter() - transfer_wait_t0
                ) * 1000.0

        self._profile["prefetch_wait_ms"] += (time.perf_counter() - t0) * 1000.0
        self._profile["verify_prefetch_transfer_wait_ms"] += transfer_wait_ms
        self._profile["verify_ready_after_wait_count"] += self._count_ready_relevant_experts()

    def _count_ready_relevant_experts(self) -> int:
        count = 0
        for ticket in self.inflight.values():
            if ticket.ready or bool(ticket.ready_event.query()):
                count += 1
        return count

    def record_verify_consumed(
        self,
        runtime_meta: dict[int, LayerRuntimeMetaCPU] | None,
        step_id: int,
    ) -> None:
        if not runtime_meta:
            return
        consumed = 0
        for layer_idx, meta in runtime_meta.items():
            cache = self.layer_caches.get(layer_idx)
            if cache is None:
                continue
            if meta.aggregated_expert_ids is not None:
                unique_ids_tensor = meta.aggregated_expert_ids
                if unique_ids_tensor.device.type != "cpu" or unique_ids_tensor.dtype != torch.int64:
                    unique_ids_tensor = unique_ids_tensor.to(device="cpu", dtype=torch.int64)
                unique_ids = unique_ids_tensor.tolist()
            elif meta.selected_experts is not None:
                flat_experts = meta.selected_experts.reshape(-1)
                if flat_experts.device.type != "cpu" or flat_experts.dtype != torch.int64:
                    flat_experts = flat_experts.to(device="cpu", dtype=torch.int64)
                unique_ids = torch.unique(flat_experts).tolist()
            else:
                continue
            for expert_idx in unique_ids:
                key = (int(layer_idx), int(expert_idx))
                was_cached_at_execution = cache.is_cached_cpu(expert_idx)
                if meta.expert_status is not None:
                    status_idx = int(expert_idx)
                    if 0 <= status_idx < int(meta.expert_status.numel()):
                        was_cached_at_execution = bool(
                            int(meta.expert_status[status_idx].item()) == 1
                        )
                self._record_source_prefetch_consumed(
                    layer_idx=int(layer_idx),
                    expert_idx=int(expert_idx),
                    step_id=int(step_id),
                    was_cached_at_execution=was_cached_at_execution,
                )
                if key not in self._recent_published:
                    continue
                if cache.is_cached_cpu(expert_idx):
                    consumed += 1
                    source = self._recent_published_source.get(key, "")
                    if source == "draft_direct_active":
                        self._profile["draft_direct_active_prefetch_consumed_count"] += 1
                    elif source == "draft_segment_indexed":
                        self._profile["draft_segment_indexed_prefetch_consumed_count"] += 1
                        self._draft_segment_indexed_consumed_by_segment[
                            self._segment_id_for_layer(int(layer_idx))
                        ] += 1
                    elif source == "verify_layer_predict":
                        self._profile["verify_layer_prefetch_consumed_count"] += 1
                    elif source == "dual_queue_draft_predict":
                        self._profile["dual_queue_draft_predict_consumed_count"] += 1
                    elif source == "dual_queue_ground_truth":
                        self._profile["dual_queue_ground_truth_consumed_count"] += 1
        self._profile["prefetch_consumed_count"] += consumed
        self._prune_unmapped_prefetch_residencies(step_id=int(step_id))

        stale = []
        ttl = int(self.config.prefetch_history_ttl_steps)
        for key, published_step in self._recent_published.items():
            if int(step_id) - int(published_step) > ttl:
                stale.append(key)
        for key in stale:
            self._recent_published.pop(key, None)
            self._recent_published_source.pop(key, None)

    def record_metadata_offload(
        self,
        *,
        mode: str,
        num_bytes: int,
        enqueue_ms: float = 0.0,
        transfer_wait_ms: float = 0.0,
        collect_ms: float = 0.0,
        observe_ms: float = 0.0,
    ) -> None:
        mode_key = mode.strip().lower() if mode else "unknown"
        total_ms = float(enqueue_ms) + float(transfer_wait_ms) + float(collect_ms) + float(observe_ms)

        self._profile["metadata_offload_count"] += 1.0
        self._profile["metadata_offload_ms"] += total_ms
        self._profile["metadata_offload_bytes"] += float(num_bytes)
        self._profile["metadata_offload_enqueue_ms"] += float(enqueue_ms)
        self._profile["metadata_offload_transfer_wait_ms"] += float(transfer_wait_ms)
        self._profile["metadata_offload_collect_ms"] += float(collect_ms)
        self._profile["metadata_offload_observe_ms"] += float(observe_ms)

        self._profile[f"metadata_offload_{mode_key}_count"] += 1.0
        self._profile[f"metadata_offload_{mode_key}_ms"] += total_ms
        self._profile[f"metadata_offload_{mode_key}_bytes"] += float(num_bytes)

    def get_profile(self, reset: bool = False) -> dict:
        out = {
            "prefetch_submit_count": int(self._profile.get("prefetch_submit_count", 0.0)),
            "prefetch_completed_count": int(self._profile.get("prefetch_completed_count", 0.0)),
            "prefetch_late_count": int(self._profile.get("prefetch_late_count", 0.0)),
            "prefetch_submitted_bytes": int(self._profile.get("prefetch_submitted_bytes", 0.0)),
            "prefetch_completed_bytes": int(self._profile.get("prefetch_completed_bytes", 0.0)),
            "prefetch_published_bytes": int(self._profile.get("prefetch_published_bytes", 0.0)),
            "prefetch_late_bytes": int(self._profile.get("prefetch_late_bytes", 0.0)),
            "prefetch_transfer_enqueue_ms": float(self._profile.get("prefetch_transfer_enqueue_ms", 0.0)),
            "prefetch_completion_latency_ms": float(self._profile.get("prefetch_completion_latency_ms", 0.0)),
            "prefetch_max_inflight_observed": int(self._profile.get("prefetch_max_inflight_observed", 0.0)),
            "prefetch_transfer_stream_count": len(self.transfer_streams),
            "prefetch_submit_count_by_stream": {
                str(k): int(v) for k, v in self._prefetch_submit_count_by_stream.items()
            },
            "prefetch_submitted_bytes_by_stream": {
                str(k): int(v) for k, v in self._prefetch_submitted_bytes_by_stream.items()
            },
            "prefetch_transfer_enqueue_ms_by_stream": {
                str(k): float(v) for k, v in self._prefetch_transfer_enqueue_ms_by_stream.items()
            },
            "prefetch_completion_latency_ms_by_stream": {
                str(k): float(v) for k, v in self._prefetch_completion_latency_ms_by_stream.items()
            },
            "prefetch_submit_count_by_source": dict(self._prefetch_submit_count_by_source),
            "prefetch_completed_count_by_source": dict(self._prefetch_completed_count_by_source),
            "prefetch_published_count_by_source": dict(self._prefetch_published_count_by_source),
            "prefetch_late_count_by_source": dict(self._prefetch_late_count_by_source),
            "prefetch_submitted_bytes_by_source": dict(self._prefetch_submitted_bytes_by_source),
            "prefetch_completed_bytes_by_source": dict(self._prefetch_completed_bytes_by_source),
            "prefetch_published_bytes_by_source": dict(self._prefetch_published_bytes_by_source),
            "prefetch_late_bytes_by_source": dict(self._prefetch_late_bytes_by_source),
            "prefetch_transfer_enqueue_ms_by_source": dict(
                self._prefetch_transfer_enqueue_ms_by_source
            ),
            "prefetch_completion_latency_ms_by_source": dict(
                self._prefetch_completion_latency_ms_by_source
            ),
            "transfer_lifecycle_events": list(
                self._transfer_lifecycle_events
            ),
            "prefetch_wait_ms": float(self._profile.get("prefetch_wait_ms", 0.0)),
            "verify_prefetch_transfer_wait_ms": float(
                self._profile.get("verify_prefetch_transfer_wait_ms", 0.0)
            ),
            "prefetch_consumed_count": int(self._profile.get("prefetch_consumed_count", 0.0)),
            "prefetch_timeout_count": int(self._profile.get("prefetch_timeout_count", 0.0)),
            "publish_count": int(self._profile.get("publish_count", 0.0)),
            "publish_ms": float(self._profile.get("publish_ms", 0.0)),
            "observe_runtime_meta_count": int(self._profile.get("observe_runtime_meta_count", 0.0)),
            "observe_runtime_meta_ms": float(self._profile.get("observe_runtime_meta_ms", 0.0)),
            "observe_mark_access_ms": float(self._profile.get("observe_mark_access_ms", 0.0)),
            "observe_queue_update_ms": float(self._profile.get("observe_queue_update_ms", 0.0)),
            "observe_verify_rank_guard_ms": float(self._profile.get("observe_verify_rank_guard_ms", 0.0)),
            "observe_verify_segment_index_ms": float(self._profile.get("observe_verify_segment_index_ms", 0.0)),
            "observe_verify_runtime_meta_call_ms": float(
                self._profile.get("observe_verify_runtime_meta_call_ms", 0.0)
            ),
            "queue_layer_count": int(self._profile.get("queue_layer_count", 0.0)),
            "queue_aggregate_ms": float(self._profile.get("queue_aggregate_ms", 0.0)),
            "queue_filter_ms": float(self._profile.get("queue_filter_ms", 0.0)),
            "queue_entry_update_ms": float(self._profile.get("queue_entry_update_ms", 0.0)),
            "queue_uncached_candidate_count": float(self._profile.get("queue_uncached_candidate_count", 0.0)),
            "segment_index_aggregate_ms": float(self._profile.get("segment_index_aggregate_ms", 0.0)),
            "segment_index_filter_ms": float(self._profile.get("segment_index_filter_ms", 0.0)),
            "segment_index_entry_update_ms": float(self._profile.get("segment_index_entry_update_ms", 0.0)),
            "segment_index_candidate_count": float(self._profile.get("segment_index_candidate_count", 0.0)),
            "metadata_offload_count": int(self._profile.get("metadata_offload_count", 0.0)),
            "metadata_offload_ms": float(self._profile.get("metadata_offload_ms", 0.0)),
            "metadata_offload_bytes": float(self._profile.get("metadata_offload_bytes", 0.0)),
            "metadata_offload_enqueue_ms": float(self._profile.get("metadata_offload_enqueue_ms", 0.0)),
            "metadata_offload_transfer_wait_ms": float(self._profile.get("metadata_offload_transfer_wait_ms", 0.0)),
            "metadata_offload_collect_ms": float(self._profile.get("metadata_offload_collect_ms", 0.0)),
            "metadata_offload_observe_ms": float(self._profile.get("metadata_offload_observe_ms", 0.0)),
            "metadata_offload_prefill_count": int(self._profile.get("metadata_offload_prefill_count", 0.0)),
            "metadata_offload_prefill_ms": float(self._profile.get("metadata_offload_prefill_ms", 0.0)),
            "metadata_offload_prefill_bytes": float(self._profile.get("metadata_offload_prefill_bytes", 0.0)),
            "metadata_offload_draft_count": int(self._profile.get("metadata_offload_draft_count", 0.0)),
            "metadata_offload_draft_ms": float(self._profile.get("metadata_offload_draft_ms", 0.0)),
            "metadata_offload_draft_bytes": float(self._profile.get("metadata_offload_draft_bytes", 0.0)),
            "metadata_offload_verify_count": int(self._profile.get("metadata_offload_verify_count", 0.0)),
            "metadata_offload_verify_ms": float(self._profile.get("metadata_offload_verify_ms", 0.0)),
            "metadata_offload_verify_bytes": float(self._profile.get("metadata_offload_verify_bytes", 0.0)),
            "staging_prefetch_submit_count": int(self._profile.get("staging_prefetch_submit_count", 0.0)),
            "staging_prefetch_publish_count": int(self._profile.get("staging_prefetch_publish_count", 0.0)),
            "staging_prefetch_publish_ms": float(self._profile.get("staging_prefetch_publish_ms", 0.0)),
            "direct_active_prefetch_submit_count": int(self._profile.get("direct_active_prefetch_submit_count", 0.0)),
            "direct_active_prefetch_ready_scan_count": int(self._profile.get("direct_active_prefetch_ready_scan_count", 0.0)),
            "direct_active_prefetch_ready_count": int(self._profile.get("direct_active_prefetch_ready_count", 0.0)),
            "direct_active_prefetch_publish_count": int(self._profile.get("direct_active_prefetch_publish_count", 0.0)),
            "direct_active_prefetch_publish_ms": float(self._profile.get("direct_active_prefetch_publish_ms", 0.0)),
            "direct_active_prefetch_drain_count": int(self._profile.get("direct_active_prefetch_drain_count", 0.0)),
            "direct_active_prefetch_drain_ms": float(self._profile.get("direct_active_prefetch_drain_ms", 0.0)),
            "direct_active_prefetch_sync_wait_ms": float(
                self._profile.get("direct_active_prefetch_sync_wait_ms", 0.0)
            ),
            "history_prefetch_submit_count": int(self._profile.get("history_prefetch_submit_count", 0.0)),
            "verify_history_prefetch_submit_count": int(self._profile.get("verify_history_prefetch_submit_count", 0.0)),
            "draft_live_prefetch_submit_count": int(self._profile.get("draft_live_prefetch_submit_count", 0.0)),
            "verify_layer_prefetch_submit_count": int(self._profile.get("verify_layer_prefetch_submit_count", 0.0)),
            "verify_layer_prefetch_ready_count": int(self._profile.get("verify_layer_prefetch_ready_count", 0.0)),
            "verify_layer_prefetch_publish_count": int(self._profile.get("verify_layer_prefetch_publish_count", 0.0)),
            "verify_layer_prefetch_consumed_count": int(self._profile.get("verify_layer_prefetch_consumed_count", 0.0)),
            "verify_layer_prefetch_budget_stop_count": int(self._profile.get("verify_layer_prefetch_budget_stop_count", 0.0)),
            "verify_layer_prefetch_est_transfer_ms": float(self._profile.get("verify_layer_prefetch_est_transfer_ms", 0.0)),
            "verify_layer_prefetch_available_ms": float(self._profile.get("verify_layer_prefetch_available_ms", 0.0)),
            "verify_layer_prefetch_used_budget_ms": float(self._profile.get("verify_layer_prefetch_used_budget_ms", 0.0)),
            "verify_segment_prefetch_call_count": int(self._profile.get("verify_segment_prefetch_call_count", 0.0)),
            "verify_segment_prefetch_submit_count": int(self._profile.get("verify_segment_prefetch_submit_count", 0.0)),
            "verify_segment_prefetch_candidate_scan_count": int(
                self._profile.get("verify_segment_prefetch_candidate_scan_count", 0.0)
            ),
            "verify_segment_prefetch_candidate_ranked_count": int(
                self._profile.get("verify_segment_prefetch_candidate_ranked_count", 0.0)
            ),
            "verify_segment_prefetch_candidate_merge_count": int(
                self._profile.get("verify_segment_prefetch_candidate_merge_count", 0.0)
            ),
            "verify_segment_prefetch_rank_limited_count": int(
                self._profile.get("verify_segment_prefetch_rank_limited_count", 0.0)
            ),
            "verify_segment_prefetch_rank_limit_sum": float(
                self._profile.get("verify_segment_prefetch_rank_limit_sum", 0.0)
            ),
            "verify_segment_prefetch_no_candidate_count": int(
                self._profile.get("verify_segment_prefetch_no_candidate_count", 0.0)
            ),
            "verify_segment_prefetch_skipped_by_budget_count": int(
                self._profile.get("verify_segment_prefetch_skipped_by_budget_count", 0.0)
            ),
            "verify_segment_prefetch_skipped_by_pending_count": int(
                self._profile.get("verify_segment_prefetch_skipped_by_pending_count", 0.0)
            ),
            "verify_segment_prefetch_missing_weights_count": int(
                self._profile.get("verify_segment_prefetch_missing_weights_count", 0.0)
            ),
            "verify_segment_prefetch_rank_ms": float(
                self._profile.get("verify_segment_prefetch_rank_ms", 0.0)
            ),
            "verify_segment_prefetch_rank_scan_ms": float(
                self._profile.get("verify_segment_prefetch_rank_scan_ms", 0.0)
            ),
            "verify_segment_prefetch_rank_sort_ms": float(
                self._profile.get("verify_segment_prefetch_rank_sort_ms", 0.0)
            ),
            "verify_segment_prefetch_loop_ms": float(
                self._profile.get("verify_segment_prefetch_loop_ms", 0.0)
            ),
            "verify_segment_prefetch_filter_ms": float(
                self._profile.get("verify_segment_prefetch_filter_ms", 0.0)
            ),
            "verify_segment_prefetch_victim_select_ms": float(
                self._profile.get("verify_segment_prefetch_victim_select_ms", 0.0)
            ),
            "verify_segment_prefetch_reservation_ms": float(
                self._profile.get("verify_segment_prefetch_reservation_ms", 0.0)
            ),
            "verify_segment_prefetch_begin_transfer_call_ms": float(
                self._profile.get("verify_segment_prefetch_begin_transfer_call_ms", 0.0)
            ),
            "verify_segment_prefetch_bookkeeping_ms": float(
                self._profile.get("verify_segment_prefetch_bookkeeping_ms", 0.0)
            ),
            "verify_segment_prefetch_dispatch_budget_sum": float(
                self._profile.get("verify_segment_prefetch_dispatch_budget_sum", 0.0)
            ),
            "verify_segment_prefetch_inflight_budget_sum": float(
                self._profile.get("verify_segment_prefetch_inflight_budget_sum", 0.0)
            ),
            "verify_segment_prefetch_transfer_enqueue_ms": float(
                self._profile.get("verify_segment_prefetch_transfer_enqueue_ms", 0.0)
            ),
            "verify_segment_prefetch_visible_overhead_ms": float(
                self._profile.get("verify_segment_prefetch_visible_overhead_ms", 0.0)
            ),
            "verify_segment_prefetch_est_transfer_ms": float(
                self._profile.get("verify_segment_prefetch_est_transfer_ms", 0.0)
            ),
            "verify_segment_prefetch_used_transfer_budget_ms": float(
                self._profile.get("verify_segment_prefetch_used_transfer_budget_ms", 0.0)
            ),
            # Debug counters for submit_verify_layer_prefetch early-return reasons
            "_vls_dbg_call": int(self._vls_dbg.get("call", 0)) if hasattr(self, "_vls_dbg") else 0,
            "_vls_dbg_disabled": int(self._vls_dbg.get("disabled", 0)) if hasattr(self, "_vls_dbg") else 0,
            "_vls_dbg_budget_zero": int(self._vls_dbg.get("budget_zero", 0)) if hasattr(self, "_vls_dbg") else 0,
            "_vls_dbg_avail_zero": int(self._vls_dbg.get("avail_zero", 0)) if hasattr(self, "_vls_dbg") else 0,
            "_vls_dbg_no_cache": int(self._vls_dbg.get("no_cache", 0)) if hasattr(self, "_vls_dbg") else 0,
            "_vls_dbg_inflight_full": int(self._vls_dbg.get("inflight_full", 0)) if hasattr(self, "_vls_dbg") else 0,
            "_vls_dbg_no_candidates": int(self._vls_dbg.get("no_candidates", 0)) if hasattr(self, "_vls_dbg") else 0,
            "_vls_dbg_ranked_total_sum": int(self._vls_dbg.get("ranked_total", 0)) if hasattr(self, "_vls_dbg") else 0,
            "_vls_dbg_ranked_after_filter_sum": int(self._vls_dbg.get("ranked_after_filter", 0)) if hasattr(self, "_vls_dbg") else 0,
            "_vls_dbg_max_submit_sum": int(self._vls_dbg.get("max_submit", 0)) if hasattr(self, "_vls_dbg") else 0,
            "_vls_dbg_submitted": int(self._vls_dbg.get("submitted", 0)) if hasattr(self, "_vls_dbg") else 0,
            "_vls_dbg_budget_stop": int(self._vls_dbg.get("budget_stop", 0)) if hasattr(self, "_vls_dbg") else 0,
            "_vls_dbg_cached_cpu": int(self._vls_dbg.get("cached_cpu", 0)) if hasattr(self, "_vls_dbg") else 0,
            "_vls_dbg_in_flight_skip": int(self._vls_dbg.get("in_flight_skip", 0)) if hasattr(self, "_vls_dbg") else 0,
            "_vls_dbg_no_weights": int(self._vls_dbg.get("no_weights", 0)) if hasattr(self, "_vls_dbg") else 0,
            "_vls_dbg_slot_none": int(self._vls_dbg.get("slot_none", 0)) if hasattr(self, "_vls_dbg") else 0,
            "_vls_dbg_reserve_fail": int(self._vls_dbg.get("reserve_fail", 0)) if hasattr(self, "_vls_dbg") else 0,
            "_vls_dbg_global_q_size": int(self._vls_dbg.get("global_q_size", 0)) if hasattr(self, "_vls_dbg") else 0,
            "_vls_dbg_avail_ms_sum": float(self._vls_dbg.get("avail_ms_sum", 0.0)) if hasattr(self, "_vls_dbg") else 0.0,
            "draft_direct_active_prefetch_submit_count": int(self._profile.get("draft_direct_active_prefetch_submit_count", 0.0)),
            "draft_direct_active_prefetch_ready_count": int(self._profile.get("draft_direct_active_prefetch_ready_count", 0.0)),
            "draft_direct_active_prefetch_publish_count": int(self._profile.get("draft_direct_active_prefetch_publish_count", 0.0)),
            "draft_direct_active_prefetch_consumed_count": int(self._profile.get("draft_direct_active_prefetch_consumed_count", 0.0)),
            "draft_direct_active_prefetch_drain_count": int(self._profile.get("draft_direct_active_prefetch_drain_count", 0.0)),
            "draft_direct_active_prefetch_drain_ms": float(self._profile.get("draft_direct_active_prefetch_drain_ms", 0.0)),
            "draft_direct_active_prefetch_sync_wait_ms": float(
                self._profile.get("draft_direct_active_prefetch_sync_wait_ms", 0.0)
            ),
            "draft_direct_active_prefetch_skipped_by_frontier_count": int(self._profile.get("draft_direct_active_prefetch_skipped_by_frontier_count", 0.0)),
            "draft_direct_active_prefetch_skipped_by_budget_count": int(self._profile.get("draft_direct_active_prefetch_skipped_by_budget_count", 0.0)),
            "draft_direct_active_prefetch_skipped_by_pending_count": int(self._profile.get("draft_direct_active_prefetch_skipped_by_pending_count", 0.0)),
            "draft_direct_active_prefetch_adaptive_budget": int(
                self._profile.get("draft_direct_active_prefetch_adaptive_budget", float(self._draft_direct_active_budget))
            ),
            "draft_direct_active_prefetch_budget_increase_count": int(self._profile.get("draft_direct_active_prefetch_budget_increase_count", 0.0)),
            "draft_direct_active_prefetch_budget_decrease_count": int(self._profile.get("draft_direct_active_prefetch_budget_decrease_count", 0.0)),
            "draft_direct_active_prefetch_visible_overhead_ms": float(self._profile.get("draft_direct_active_prefetch_visible_overhead_ms", 0.0)),
            "draft_direct_active_prefetch_est_transfer_ms": float(self._profile.get("draft_direct_active_prefetch_est_transfer_ms", 0.0)),
            "draft_direct_active_prefetch_used_transfer_budget_ms": float(self._profile.get("draft_direct_active_prefetch_used_transfer_budget_ms", 0.0)),
            "draft_segment_indexed_prefetch_submit_count": int(self._profile.get("draft_segment_indexed_prefetch_submit_count", 0.0)),
            "draft_segment_indexed_prefetch_ready_count": int(self._profile.get("draft_segment_indexed_prefetch_ready_count", 0.0)),
            "draft_segment_indexed_prefetch_publish_count": int(self._profile.get("draft_segment_indexed_prefetch_publish_count", 0.0)),
            "draft_segment_indexed_prefetch_consumed_count": int(self._profile.get("draft_segment_indexed_prefetch_consumed_count", 0.0)),
            "draft_segment_indexed_prefetch_drain_count": int(self._profile.get("draft_segment_indexed_prefetch_drain_count", 0.0)),
            "draft_segment_indexed_prefetch_drain_ms": float(self._profile.get("draft_segment_indexed_prefetch_drain_ms", 0.0)),
            "draft_segment_indexed_prefetch_sync_wait_ms": float(
                self._profile.get("draft_segment_indexed_prefetch_sync_wait_ms", 0.0)
            ),
            "draft_segment_indexed_prefetch_skipped_by_budget_count": int(self._profile.get("draft_segment_indexed_prefetch_skipped_by_budget_count", 0.0)),
            "draft_segment_indexed_prefetch_skipped_by_pending_count": int(self._profile.get("draft_segment_indexed_prefetch_skipped_by_pending_count", 0.0)),
            "draft_segment_indexed_prefetch_adaptive_budget": int(
                self._profile.get("draft_segment_indexed_prefetch_adaptive_budget", float(self._draft_segment_indexed_budget))
            ),
            "draft_segment_indexed_prefetch_budget_increase_count": int(self._profile.get("draft_segment_indexed_prefetch_budget_increase_count", 0.0)),
            "draft_segment_indexed_prefetch_budget_decrease_count": int(self._profile.get("draft_segment_indexed_prefetch_budget_decrease_count", 0.0)),
            "draft_segment_indexed_prefetch_visible_overhead_ms": float(self._profile.get("draft_segment_indexed_prefetch_visible_overhead_ms", 0.0)),
            "draft_segment_indexed_prefetch_est_transfer_ms": float(self._profile.get("draft_segment_indexed_prefetch_est_transfer_ms", 0.0)),
            "draft_segment_indexed_prefetch_used_transfer_budget_ms": float(self._profile.get("draft_segment_indexed_prefetch_used_transfer_budget_ms", 0.0)),
            "draft_segment_indexed_rank_ms": float(self._profile.get("draft_segment_indexed_rank_ms", 0.0)),
            "draft_segment_indexed_victim_select_ms": float(self._profile.get("draft_segment_indexed_victim_select_ms", 0.0)),
            "draft_segment_indexed_prefetch_reservation_ms": float(
                self._profile.get("draft_segment_indexed_prefetch_reservation_ms", 0.0)
            ),
            "draft_segment_indexed_prefetch_transfer_enqueue_ms": float(
                self._profile.get("draft_segment_indexed_prefetch_transfer_enqueue_ms", 0.0)
            ),
            "draft_segment_indexed_prefetch_completion_latency_ms": float(
                self._profile.get("draft_segment_indexed_prefetch_completion_latency_ms", 0.0)
            ),
            "draft_segment_indexed_prefetch_submitted_bytes": int(
                self._profile.get("draft_segment_indexed_prefetch_submitted_bytes", 0.0)
            ),
            "draft_segment_indexed_prefetch_completed_bytes": int(
                self._profile.get("draft_segment_indexed_prefetch_completed_bytes", 0.0)
            ),
            "draft_segment_indexed_prefetch_published_bytes": int(
                self._profile.get("draft_segment_indexed_prefetch_published_bytes", 0.0)
            ),
            "draft_segment_indexed_prefetch_late_bytes": int(
                self._profile.get("draft_segment_indexed_prefetch_late_bytes", 0.0)
            ),
            "draft_segment_indexed_candidate_scan_count": int(self._profile.get("draft_segment_indexed_candidate_scan_count", 0.0)),
            "draft_segment_indexed_candidate_ranked_count": int(self._profile.get("draft_segment_indexed_candidate_ranked_count", 0.0)),
            "draft_segment_indexed_candidate_merge_count": int(self._profile.get("draft_segment_indexed_candidate_merge_count", 0.0)),
            "draft_segment_indexed_prefetch_submit_count_by_segment": self._format_segment_counts(
                self._draft_segment_indexed_submit_by_segment
            ),
            "draft_segment_indexed_prefetch_ready_count_by_segment": self._format_segment_counts(
                self._draft_segment_indexed_ready_by_segment
            ),
            "draft_segment_indexed_prefetch_success_count_by_segment": self._format_segment_counts(
                self._draft_segment_indexed_success_by_segment
            ),
            "draft_segment_indexed_prefetch_consumed_count_by_segment": self._format_segment_counts(
                self._draft_segment_indexed_consumed_by_segment
            ),
            "draft_segment_indexed_stale_metadata_observe_count": int(self._profile.get("draft_segment_indexed_stale_metadata_observe_count", 0.0)),
            "draft_segment_indexed_missed_prefetch_window_count": int(self._profile.get("draft_segment_indexed_missed_prefetch_window_count", 0.0)),
            "verify_ready_before_wait_count": int(self._profile.get("verify_ready_before_wait_count", 0.0)),
            "verify_ready_after_wait_count": int(self._profile.get("verify_ready_after_wait_count", 0.0)),
            "draft_m3_group_count": int(self._profile.get("draft_m3_group_count", 0.0)),
            "draft_m3_perfect_count": int(self._profile.get("draft_m3_perfect_count", 0.0)),
            "draft_m3_perfect_fraction": (
                float(self._profile.get("draft_m3_perfect_count", 0.0) / self._profile.get("draft_m3_group_count", 0.0))
                if self._profile.get("draft_m3_group_count", 0.0) > 0
                else 0.0
            ),
            "draft_m3_step0_group_count": int(self._profile.get("draft_m3_step0_group_count", 0.0)),
            "draft_m3_step0_perfect_count": int(self._profile.get("draft_m3_step0_perfect_count", 0.0)),
            "draft_m3_step0_perfect_fraction": (
                float(
                    self._profile.get("draft_m3_step0_perfect_count", 0.0)
                    / self._profile.get("draft_m3_step0_group_count", 0.0)
                )
                if self._profile.get("draft_m3_step0_group_count", 0.0) > 0
                else 0.0
            ),
        }
        if bool(getattr(self, "_source_lifecycle_profile_enabled", False)):
            out.update(self._source_lifecycle_profile())
        if reset:
            self._profile.clear()
            self._draft_direct_active_budget = self._initial_draft_direct_active_budget()
            self._draft_segment_indexed_budget = self._initial_draft_direct_active_budget()
            self._draft_segment_indexed_submit_by_segment.clear()
            self._draft_segment_indexed_ready_by_segment.clear()
            self._draft_segment_indexed_success_by_segment.clear()
            self._draft_segment_indexed_consumed_by_segment.clear()
            self._prefetch_submit_count_by_source.clear()
            self._prefetch_completed_count_by_source.clear()
            self._prefetch_published_count_by_source.clear()
            self._prefetch_late_count_by_source.clear()
            self._prefetch_submitted_bytes_by_source.clear()
            self._prefetch_completed_bytes_by_source.clear()
            self._prefetch_published_bytes_by_source.clear()
            self._prefetch_late_bytes_by_source.clear()
            self._prefetch_transfer_enqueue_ms_by_source.clear()
            self._prefetch_completion_latency_ms_by_source.clear()
            self._prefetch_submit_count_by_stream.clear()
            self._prefetch_submitted_bytes_by_stream.clear()
            self._prefetch_transfer_enqueue_ms_by_stream.clear()
            self._prefetch_completion_latency_ms_by_stream.clear()
            self._transfer_lifecycle_events.clear()
            if hasattr(self, "_prefetch_active_residencies"):
                self._prefetch_active_residencies.clear()
            if hasattr(self, "_prefetch_closed_residencies_by_key"):
                self._prefetch_closed_residencies_by_key.clear()
            if hasattr(self, "_prefetch_residency_sequence"):
                self._prefetch_residency_sequence = 0
            self._draft_m3_layers_by_step.clear()
            self._draft_m3_step0_steps.clear()
            self._draft_m3_next_is_step0 = True
        return out


class PredictivePrefetchRuntime(PrefetchRuntime):
    """Conservative-mode predictive prefetcher (see docs/prefetch_redesign_plan.md Part E).

    Data sources are split cleanly (B.2): the expert cache ``access_count`` records
    *only* prefill/verify ground-truth frequency, while the draft prediction queue
    (``draft_segment_index``) is fed *only* by draft routing.  Draft prefetch reuses
    the inherited segment-indexed submit (each completed segment is prefetched =
    conservative safety rule); the draft-only queue arises naturally by never
    feeding ``long_term_segment_index``.

    Single-round eviction protection (B.7) lives entirely in the victim selectors:
    the incoming expert is recorded into ``_round_loaded`` and round-loaded experts
    are skipped as victims (composed with rankguard protection, E.7-A), with a
    safety valve that ignores protection when every slot is protected (E.7-B).
    Protection spans draft+verify prefetch evictions and is released per layer at
    verify compute start (``on_verify_layer_start``).

    Selected as ``prefetch_runtime_kind == "predictive"`` (forces
    ``prefetch_runtime_mode == "draft_segment_indexed"`` so model_runner gates fire).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # layer_idx -> set of expert_idx prefetched this round (B.7).
        self._round_loaded: dict[int, set[int]] = defaultdict(set)
        self._valid_draft_iteration_steps: set[int] = set()
        # Phase-1 fires once per round; begin_draft_iteration arms this and
        # maybe_submit_phase1 (called after the run_draft drain) consumes it.
        self._phase1_pending: bool = False

    # Always use the segment-indexed plumbing regardless of the config string.
    def _segment_indexed_enabled(self) -> bool:
        return True

    # ----- data-source separation (B.2) -----------------------------------
    def observe_prefill(self, runtime_meta, step_id: int) -> dict[str, float]:
        # Ground-truth frequency only (mark_access); never feed the queue.
        return self.observe_runtime_meta(
            runtime_meta,
            source="prefill_history",
            step_id=step_id,
            update_global_queue=False,
            segment_index=None,
        )

    def observe_verify(self, runtime_meta, step_id: int) -> dict[str, float]:
        if runtime_meta is not None:
            self.verify_segment_index.update_from_runtime_meta(
                runtime_meta, "verify_history", step_id, self.layer_caches,
            )
        out = self.observe_runtime_meta(
            runtime_meta,
            source="verify_history",
            step_id=step_id,
            update_global_queue=False,
            segment_index=None,
        )
        self._update_rank_guard_scores(runtime_meta)
        return out

    def observe_draft(self, runtime_meta, step_id: int) -> dict[str, float]:
        # Draft prediction queue only; no mark_access (no ground-truth pollution).
        if not runtime_meta:
            return {}
        if int(step_id) not in self._valid_draft_iteration_steps:
            self._profile["predictive_draft_stale_observe_count"] += 1.0
            return {}
        self._record_draft_m3_cache_hits(runtime_meta, step_id)
        self.draft_segment_index.update_from_runtime_meta(
            runtime_meta=runtime_meta,
            source="draft_live",
            step_id=step_id,
            layer_caches=self.layer_caches,
        )
        return {}

    # ----- round lifecycle (queue lives draft->verify; E.3) ---------------
    def begin_draft_iteration(self, step_id: int) -> None:
        if not self._draft_iteration_open:
            # New round: clear draft queue + round protection.
            self.draft_segment_index.clear()
            self._active_draft_iteration_steps.clear()
            self._valid_draft_iteration_steps.clear()
            self._round_loaded.clear()
            self._draft_iteration_open = True
            self._mark_draft_m3_step_start(step_id)
            self._active_draft_iteration_steps.add(int(step_id))
            self._valid_draft_iteration_steps.add(int(step_id))
            # Arm phase-1; it must be submitted AFTER the run_draft drain (E.9.1),
            # not here (this runs before the drain, which would synchronize on it).
            self._phase1_pending = True
        else:
            self._active_draft_iteration_steps.add(int(step_id))
            self._valid_draft_iteration_steps.add(int(step_id))

    def maybe_submit_phase1(self, step_id: int) -> int:
        """Submit phase-1 once per round, after the run_draft drain (E.9.1)."""
        if not self._phase1_pending:
            return 0
        self._phase1_pending = False
        return self.submit_phase1_prefetch(step_id=int(step_id))

    def end_draft_iteration(self) -> None:
        # Keep draft queue + _round_loaded alive through verify; they are reset
        # at the next round's begin_draft_iteration.
        self._active_draft_iteration_steps.clear()
        self._draft_iteration_open = False

    def on_verify_layer_start(self, layer_idx: int) -> None:
        # Release this layer's round protection as verify reaches it (hygiene;
        # cache_fill does not consult it anyway, E.7-C).
        self._round_loaded.pop(int(layer_idx), None)

    # ----- round-protected victim selection (B.7 + E.7-A/B) ---------------
    def _round_protected(self, layer_idx: int, expert_idx: int) -> bool:
        s = self._round_loaded.get(int(layer_idx))
        return s is not None and int(expert_idx) in s

    def _draft_segment_indexed_target_segment_id(self, frontier_layer_idx: int | None) -> int | None:
        segment_id = super()._draft_segment_indexed_target_segment_id(frontier_layer_idx)
        if segment_id is None:
            return None
        if segment_id == 0 and self.layer_caches:
            return max(self._segment_id_for_layer(int(layer_idx)) for layer_idx in self.layer_caches)
        return segment_id

    def _rollback_round_loaded_prefetch(self, layer_idx: int, expert_idx: int) -> None:
        loaded = self._round_loaded.get(int(layer_idx))
        if loaded is None:
            return
        loaded.discard(int(expert_idx))
        if not loaded:
            self._round_loaded.pop(int(layer_idx), None)

    def _select_protected_victim(self, cache, layer_idx, incoming_expert_idx) -> int | None:
        if layer_idx is not None:
            self._round_loaded[int(layer_idx)].add(int(incoming_expert_idx))
        strategy_name = str(getattr(self.config, "cache_strategy", "lru")).strip().lower()
        use_lfu = strategy_name in ("lfu", "lfu_rankguard", "lfu_rankguard_online")
        rankguard = self.cache_strategy if isinstance(self.cache_strategy, LFURankGuardStrategy) else None
        li = None if layer_idx is None else int(layer_idx)

        protected = set(self._round_loaded.get(li, ())) if li is not None else set()
        if rankguard is not None and li is not None:
            protected.update(
                int(expert_idx)
                for expert_idx in cache.slot_to_expert
                if int(expert_idx) >= 0
                and rankguard.is_protected(li, int(expert_idx))
            )
        access_values = (
            cache.access_count if use_lfu else cache.last_access_step
        )
        return select_predictive_victim_slot(
            slots=cache.slot_to_expert,
            pending=cache.active_slot_pending_expert,
            access_values=access_values,
            protected_experts=protected,
        )

    def _select_publish_slot_cpu(self, cache, *, expert_idx, step_id, layer_idx=None) -> int | None:
        return self._select_protected_victim(cache, layer_idx, expert_idx)

    def _select_publish_slot(self, cache, *, layer_idx, expert_idx, step_id) -> int | None:
        return self._select_protected_victim(cache, layer_idx, expert_idx)

    # ----- phase-1 cold start (E.9) ---------------------------------------
    def submit_phase1_prefetch(self, *, step_id: int) -> int:
        budget = int(getattr(self.config, "predictive_phase1_budget", 4))
        if budget <= 0 or not self.layer_caches:
            return 0
        if int(self.config.prefetch_step_budget) <= 0:
            return 0
        last_segment_id = max(self._segment_id_for_layer(int(l)) for l in self.layer_caches)

        target_layers = [
            int(layer_idx)
            for layer_idx in self.layer_caches
            if self._segment_id_for_layer(int(layer_idx)) == last_segment_id
        ]
        if not target_layers:
            return 0

        # The retained policy scans lifetime ground-truth frequency.  The
        # opt-in policy first reuses the previous verify route index, whose
        # score/count/age priority is both more recent and already cached by
        # SegmentCandidateIndex.  Fall back on the first round before verify
        # metadata exists.
        candidates: list[tuple[float, int, int]] = []
        use_recent_verify = bool(
            getattr(self.config, "predictive_phase1_recent_verify", False)
        )
        if use_recent_verify:
            recent = self.verify_segment_index.candidates_for_layer_range(
                layer_start=min(target_layers),
                layer_end=max(target_layers) + 1,
                step_id=int(step_id),
                layer_caches=self.layer_caches,
                inflight_keys=set(self.inflight),
                max_candidates=budget,
            )
            candidates = [
                (
                    float(candidate.priority),
                    int(candidate.layer_idx),
                    int(candidate.expert_idx),
                )
                for candidate in recent
            ]
            self._profile["predictive_phase1_recent_verify_candidate_count"] += len(
                candidates
            )
            if candidates:
                self._profile["predictive_phase1_recent_verify_round_count"] += 1

        if not candidates:
            if use_recent_verify:
                self._profile["predictive_phase1_frequency_fallback_round_count"] += 1
            for layer_idx in target_layers:
                cache = self.layer_caches[layer_idx]
                pool = self.cpu_expert_pool.get(layer_idx)
                if not pool:
                    continue
                ac = cache.access_count
                for expert_idx in pool.keys():
                    eid = int(expert_idx)
                    if cache.is_cached_cpu(eid) or cache.is_pending_cpu(eid):
                        continue
                    candidates.append((float(ac[eid]), int(layer_idx), eid))
        if not candidates:
            return 0
        candidates.sort(
            key=lambda candidate: (-candidate[0], candidate[1], candidate[2])
        )

        inflight_budget = max(0, int(self.config.prefetch_max_inflight) - len(self.inflight))
        max_submit = min(budget, inflight_budget)
        submitted = 0
        for _priority, layer_idx, expert_idx in candidates:
            if submitted >= max_submit:
                break
            key = (layer_idx, expert_idx)
            if key in self.inflight:
                continue
            cache = self.layer_caches.get(layer_idx)
            if cache is None:
                continue
            weights = self.cpu_expert_pool.get(layer_idx, {}).get(expert_idx)
            if not weights or "gate_up" not in weights or "down" not in weights:
                continue
            victim_slot = self._select_publish_slot_cpu(
                cache, expert_idx=expert_idx, step_id=step_id, layer_idx=layer_idx,
            )
            if victim_slot is None:
                self._rollback_round_loaded_prefetch(layer_idx, expert_idx)
                continue
            # Non-deferred reserve (E.9.2): segment n-1 is still computed this
            # forward, so the slot must be unmapped immediately (LUT cleared) to
            # avoid a buffer race with the segment n-1 read. Pairs with the
            # commit_active path (source != "draft_segment_indexed").
            reservation = cache.reserve_active_slot_for_prefetch(
                layer_idx=layer_idx, active_slot_idx=victim_slot, expert_idx=expert_idx,
            )
            if reservation is None:
                self._rollback_round_loaded_prefetch(layer_idx, expert_idx)
                continue
            ready_event, submit_ts_ms, enqueue_ms, num_bytes, stream_idx = self._begin_prefetch_transfer(
                cache=cache,
                reservation=reservation,
                weights=weights,
                source="predictive_phase1",
                direct_active=True,
            )
            ticket = PrefetchTicket(
                step_id=step_id,
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                source="predictive_phase1",
                staging_slot_idx=-1,
                staging_generation=-1,
                submit_ts_ms=submit_ts_ms,
                ready_event=ready_event,
                ready=False,
                direct_active=True,
                active_slot_idx=reservation.active_slot_idx,
                active_generation=reservation.generation,
                active_slot_prev_expert=int(getattr(reservation, "prev_expert", -1)),
                segment_id=int(last_segment_id),
                num_bytes=num_bytes,
                transfer_enqueue_ms=enqueue_ms,
                transfer_stream_idx=stream_idx,
            )
            self.inflight[key] = ticket
            self._record_prefetch_submitted(ticket)
            submitted += 1
            self._profile["prefetch_submit_count"] += 1
            self._profile["direct_active_prefetch_submit_count"] += 1
            self._profile["predictive_phase1_prefetch_submit_count"] += 1
        return submitted

    # ----- verify prefetch from draft queue, attention-window budget (B.4/E.13)
    def submit_verify_layer_prefetch(self, *, step_id: int, target_layer_idx: int, available_ms: float) -> int:
        if not bool(getattr(self.config, "prefetch_verify_layer_enabled", True)):
            return 0
        if int(self.config.prefetch_step_budget) <= 0:
            return 0
        # E.13 (approach A): confine prefetch to ~attention window.
        available_ms = float(available_ms) * float(getattr(self.config, "prefetch_verify_attention_ratio", 0.3))
        if available_ms <= 0.0:
            return 0
        layer_idx = int(target_layer_idx)
        cache = self.layer_caches.get(layer_idx)
        if cache is None:
            return 0
        self.publish_direct_active_ready(step_id=step_id)
        inflight_budget = max(0, int(self.config.prefetch_max_inflight) - len(self.inflight))
        if inflight_budget <= 0:
            return 0

        segment_id = self._segment_id_for_layer(layer_idx)
        inflight_keys = set(self.inflight.keys())
        ranked = self.draft_segment_index.candidates(
            segment_id=segment_id,
            step_id=step_id,
            layer_caches=self.layer_caches,
            inflight_keys=inflight_keys,
        )
        ranked = [c for c in ranked if int(c.layer_idx) == layer_idx]
        if not ranked:
            return 0
        ranked.sort(key=lambda x: (-x.priority, x.layer_idx, x.expert_idx))

        max_submit = min(
            max(0, int(self.config.prefetch_step_budget)),
            max(0, int(getattr(self.config, "prefetch_verify_layer_max_budget", self.config.prefetch_step_budget))),
            inflight_budget,
        )
        submitted = 0
        used_budget_ms = 0.0
        for candidate in ranked:
            if submitted >= max_submit:
                break
            expert_idx = int(candidate.expert_idx)
            key = (layer_idx, expert_idx)
            if cache.is_cached_cpu(expert_idx) or key in self.inflight:
                continue
            weights = self.cpu_expert_pool.get(layer_idx, {}).get(expert_idx)
            if not weights or "gate_up" not in weights or "down" not in weights:
                continue
            transfer_ms = self._estimated_expert_transfer_ms(weights)
            if not isfinite(transfer_ms):
                continue
            if used_budget_ms + transfer_ms > available_ms:
                self._profile["verify_layer_prefetch_budget_stop_count"] += 1
                break
            victim_slot = self._select_publish_slot(
                cache, layer_idx=layer_idx, expert_idx=expert_idx, step_id=step_id,
            )
            if victim_slot is None:
                self._rollback_round_loaded_prefetch(layer_idx, expert_idx)
                continue
            reservation = cache.reserve_active_slot_for_prefetch(
                layer_idx=layer_idx, active_slot_idx=victim_slot, expert_idx=expert_idx,
            )
            if reservation is None:
                self._rollback_round_loaded_prefetch(layer_idx, expert_idx)
                continue
            ready_event, submit_ts_ms, enqueue_ms, num_bytes, stream_idx = self._begin_prefetch_transfer(
                cache=cache,
                reservation=reservation,
                weights=weights,
                source="verify_layer_predict",
                direct_active=True,
            )
            ticket = PrefetchTicket(
                step_id=step_id,
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                source="verify_layer_predict",
                staging_slot_idx=-1,
                staging_generation=-1,
                submit_ts_ms=submit_ts_ms,
                ready_event=ready_event,
                ready=False,
                direct_active=True,
                active_slot_idx=reservation.active_slot_idx,
                active_generation=reservation.generation,
                active_slot_prev_expert=int(getattr(reservation, "prev_expert", -1)),
                num_bytes=num_bytes,
                transfer_enqueue_ms=enqueue_ms,
                transfer_stream_idx=stream_idx,
            )
            self.inflight[key] = ticket
            self._record_prefetch_submitted(ticket)
            submitted += 1
            used_budget_ms += transfer_ms
            self._profile["prefetch_submit_count"] += 1
            self._profile["direct_active_prefetch_submit_count"] += 1
            self._profile["verify_layer_prefetch_submit_count"] += 1
            self._profile["verify_layer_prefetch_est_transfer_ms"] += transfer_ms
        return submitted

    def get_profile(self, reset: bool = False) -> dict:
        extra = {
            "predictive_phase1_prefetch_submit_count": int(
                self._profile.get("predictive_phase1_prefetch_submit_count", 0.0)
            ),
            "predictive_draft_stale_observe_count": int(
                self._profile.get("predictive_draft_stale_observe_count", 0.0)
            ),
            "predictive_phase1_recent_verify_candidate_count": int(
                self._profile.get(
                    "predictive_phase1_recent_verify_candidate_count", 0.0
                )
            ),
            "predictive_phase1_recent_verify_round_count": int(
                self._profile.get(
                    "predictive_phase1_recent_verify_round_count", 0.0
                )
            ),
            "predictive_phase1_frequency_fallback_round_count": int(
                self._profile.get(
                    "predictive_phase1_frequency_fallback_round_count", 0.0
                )
            ),
        }
        out = super().get_profile(reset=reset)
        out.update(extra)
        return out


class DualQueuePrefetchRuntime(PrefetchRuntime):
    """Segment prefetcher with isolated draft-predict and ground-truth queues."""

    DRAFT_SOURCE = "dual_queue_draft_predict"
    GROUND_TRUTH_SOURCE = "dual_queue_ground_truth"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        segment_size = max(1, int(getattr(self.config, "dual_queue_segment_size", 12)))
        self.draft_predict_index = DualQueueSegmentIndex(
            segment_size=segment_size,
            history=False,
            config=self.config,
        )
        self.ground_truth_index = DualQueueSegmentIndex(
            segment_size=segment_size,
            history=True,
            config=self.config,
        )
        self._round_id = 0
        self._round_active = False
        self._draft_forward_index = -1
        self._round_protected: dict[int, set[int]] = defaultdict(set)
        self._round_loaded: dict[int, set[int]] = defaultdict(set)
        self._calibrating = False
        self._calibrated = False
        self._segment_compute_ms: dict[str, dict[int, float]] = {
            "draft": {},
            "verify": {},
        }
        self._expired_tickets: set[tuple[int, int]] = set()
        self._expert_transfer_ms = self._estimated_representative_transfer_ms()
        self._ema_alpha = float(getattr(self.config, "dual_queue_segment_time_ema_alpha", 0.2))
        self._budget_recompute_interval = 8
        self._budget_recompute_counter = 0
        self._verify_segment_budgets: dict[int, int] = {}

    def _segment_indexed_enabled(self) -> bool:
        return True

    def begin_draft_iteration(self, step_id: int) -> None:
        if not self._round_active:
            self._round_id += 1
            self._round_active = True
            self._draft_forward_index = 0
            self._round_protected.clear()
            self.draft_predict_index.clear()
        else:
            self._draft_forward_index += 1
        self._draft_iteration_open = True
        self._active_draft_iteration_steps.add(int(step_id))

    def end_draft_iteration(self) -> None:
        self._active_draft_iteration_steps.clear()
        self._draft_iteration_open = False

    def is_first_draft_forward(self) -> bool:
        return self._draft_forward_index == 0

    def finish_verify_round(self) -> None:
        self.draft_predict_index.clear()
        self._round_protected.clear()
        self._round_loaded.clear()
        self._active_draft_iteration_steps.clear()
        self._draft_iteration_open = False
        self._round_active = False
        self._draft_forward_index = -1
        self._profile["dual_queue_round_clear_count"] += 1

    def _record_prefetch_published(self, ticket: PrefetchTicket) -> None:
        super()._record_prefetch_published(ticket)
        if ticket.source in (self.DRAFT_SOURCE, self.GROUND_TRUTH_SOURCE):
            self._round_loaded[int(ticket.layer_idx)].add(int(ticket.expert_idx))

    def set_calibrating(self, enabled: bool) -> None:
        self._calibrating = bool(enabled)
        if enabled:
            self._segment_compute_ms = {"draft": {}, "verify": {}}
            self._verify_segment_budgets = {}

    @staticmethod
    def _mark_cache_access(
        layer_caches: dict[int, LayerExpertCache],
        runtime_meta: dict[int, LayerRuntimeMetaCPU] | None,
        step_id: int,
    ) -> None:
        if not runtime_meta:
            return
        for layer_idx, meta in runtime_meta.items():
            cache = layer_caches.get(int(layer_idx))
            if cache is None:
                continue
            if meta.aggregated_expert_ids is not None:
                cache.mark_access_aggregated(
                    meta.aggregated_expert_ids,
                    meta.aggregated_activation_count,
                    meta.aggregated_score_sum,
                    step_id=int(step_id),
                )
            elif meta.selected_experts is not None:
                cache.mark_access(meta.selected_experts, meta.routing_weights, step_id=int(step_id))

    def observe_prefill(self, runtime_meta, step_id: int) -> dict[str, float]:
        t0 = time.perf_counter()
        self._mark_cache_access(self.layer_caches, runtime_meta, step_id)
        self.ground_truth_index.update(runtime_meta, round_id=self._round_id)
        return {"queue_update_ms": (time.perf_counter() - t0) * 1000.0}

    def observe_verify(self, runtime_meta, step_id: int) -> dict[str, float]:
        t0 = time.perf_counter()
        self._mark_cache_access(self.layer_caches, runtime_meta, step_id)
        self.ground_truth_index.update(runtime_meta, round_id=self._round_id)
        self._update_rank_guard_scores(runtime_meta)
        return {"queue_update_ms": (time.perf_counter() - t0) * 1000.0}

    def observe_draft(self, runtime_meta, step_id: int) -> dict[str, float]:
        if int(step_id) not in self._active_draft_iteration_steps:
            self._profile["dual_queue_stale_draft_metadata_count"] += 1
            return {}
        t0 = time.perf_counter()
        activated = self.draft_predict_index.update(runtime_meta, round_id=self._round_id)
        for layer_idx, expert_idx in activated:
            self._round_protected[int(layer_idx)].add(int(expert_idx))
        return {"queue_update_ms": (time.perf_counter() - t0) * 1000.0}

    def _estimated_representative_transfer_ms(self) -> float:
        for layer_pool in self.cpu_expert_pool.values():
            for weights in layer_pool.values():
                return max(1e-6, self._estimated_expert_transfer_ms(weights))
        return 1.0

    def calibrate_expert_transfer_ms(self, repeats: int = 3) -> float:
        representative = None
        for layer_pool in self.cpu_expert_pool.values():
            if layer_pool:
                representative = next(iter(layer_pool.values()))
                break
        if representative is None or not torch.cuda.is_available():
            return float(self._expert_transfer_ms)

        gate_up = representative["gate_up"]
        down = representative["down"]
        target_gate_up = torch.empty(tuple(gate_up.shape), dtype=gate_up.dtype, device="cuda")
        target_down = torch.empty(tuple(down.shape), dtype=down.dtype, device="cuda")
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            copy_expert_tensor(target_gate_up, gate_up, non_blocking=True)
            copy_expert_tensor(target_down, down, non_blocking=True)
        stream.synchronize()

        samples: list[float] = []
        for _ in range(max(1, int(repeats))):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream):
                start.record(stream)
                copy_expert_tensor(target_gate_up, gate_up, non_blocking=True)
                copy_expert_tensor(target_down, down, non_blocking=True)
                end.record(stream)
            end.synchronize()
            samples.append(float(start.elapsed_time(end)))
        self._expert_transfer_ms = max(1e-6, max(samples))
        return float(self._expert_transfer_ms)

    def record_segment_compute_ms(self, phase: str, segment_id: int, compute_ms: float) -> None:
        phase_key = str(phase)
        if phase_key not in self._segment_compute_ms or compute_ms < 0.0:
            return
        values = self._segment_compute_ms[phase_key]
        previous = values.get(int(segment_id))
        alpha = self._ema_alpha
        values[int(segment_id)] = (
            float(compute_ms)
            if previous is None
            else (1.0 - alpha) * float(previous) + alpha * float(compute_ms)
        )
        if self._calibrated and not self._calibrating:
            self._budget_recompute_counter += 1
            if self._budget_recompute_counter >= self._budget_recompute_interval:
                self._budget_recompute_counter = 0
                self._recompute_prefetch_budgets()

    def _visible_budget_ms_for_segment(self, segment_id: int) -> float | None:
        seg_times = self._segment_compute_ms.get("verify", {})
        seg_time = seg_times.get(int(segment_id))
        if seg_time is not None and seg_time > 0.0:
            safety = float(getattr(self.config, "dual_queue_budget_safety_ratio", 0.8))
            return safety * float(seg_time)
        return None

    def _verify_budget_for_segment(self, segment_id: int) -> int:
        if self._verify_segment_budgets:
            return self._verify_segment_budgets.get(
                int(segment_id),
                int(getattr(self.config, "verify_prefetch_max_per_boundary", 0)),
            )
        return int(getattr(self.config, "verify_prefetch_max_per_boundary", 0))

    def _recompute_prefetch_budgets(self) -> tuple[int, int]:
        safety = float(getattr(self.config, "dual_queue_budget_safety_ratio", 0.8))
        max_inflight = max(0, int(getattr(self.config, "prefetch_max_inflight", 0)))

        def phase_budget(phase: str, fallback: int) -> int:
            samples = self._segment_compute_ms.get(phase, {})
            if not samples or self._expert_transfer_ms <= 0.0:
                return max(0, min(max_inflight, int(fallback)))
            capacity = int(safety * min(samples.values()) / self._expert_transfer_ms)
            return max(0, min(max_inflight, capacity))

        draft_budget = phase_budget(
            "draft", int(getattr(self.config, "draft_prefetch_max_per_boundary", 0))
        )
        verify_budget = phase_budget(
            "verify", int(getattr(self.config, "verify_prefetch_max_per_boundary", 0))
        )

        verify_samples = self._segment_compute_ms.get("verify", {})
        if verify_samples and self._expert_transfer_ms > 0.0:
            per_seg: dict[int, int] = {}
            for seg_id, compute_ms in verify_samples.items():
                capacity = int(safety * float(compute_ms) / self._expert_transfer_ms)
                per_seg[int(seg_id)] = max(0, min(max_inflight, capacity))
            self._verify_segment_budgets = per_seg
        else:
            self._verify_segment_budgets = {}

        self.config.draft_prefetch_max_per_boundary = draft_budget
        self.config.verify_prefetch_max_per_boundary = verify_budget
        self._profile["dual_queue_draft_budget"] = draft_budget
        self._profile["dual_queue_verify_budget"] = verify_budget
        return draft_budget, verify_budget

    def finalize_calibration(self) -> dict:
        self.calibrate_expert_transfer_ms()
        self._calibrated = True
        self._calibrating = False
        draft_budget, verify_budget = self._recompute_prefetch_budgets()
        result: dict = {
            "draft_prefetch_max_per_boundary": int(draft_budget),
            "verify_prefetch_max_per_boundary": int(verify_budget),
            "expert_transfer_ms": float(self._expert_transfer_ms),
            "draft_min_segment_ms": float(min(self._segment_compute_ms["draft"].values(), default=0.0)),
            "verify_min_segment_ms": float(min(self._segment_compute_ms["verify"].values(), default=0.0)),
        }
        if self._verify_segment_budgets:
            result["verify_segment_budgets"] = dict(self._verify_segment_budgets)
        return result

    def _select_dual_queue_victim(
        self,
        cache: LayerExpertCache,
        *,
        layer_idx: int,
        source: str,
        boundary_protected: set[int] | None = None,
    ) -> int | None:
        for slot_idx, slot_expert in enumerate(cache.slot_to_expert):
            if int(slot_expert) < 0 and not cache.is_active_slot_pending(slot_idx):
                return slot_idx

        protected = self._round_protected.get(int(layer_idx), frozenset())
        round_loaded = self._round_loaded.get(int(layer_idx), frozenset())
        protected_this_boundary = boundary_protected or frozenset()
        best: tuple[float, int, int] | None = None
        fallback: tuple[float, int, int] | None = None
        for slot_idx, slot_expert in enumerate(cache.slot_to_expert):
            if cache.is_active_slot_pending(slot_idx):
                continue
            expert_idx = int(slot_expert)
            score = self.ground_truth_index.priority_for(layer_idx, expert_idx, self._round_id)
            last_access = int(cache.last_access_step[expert_idx]) if expert_idx >= 0 else -1
            candidate = (float(score), last_access, int(slot_idx))
            if fallback is None or candidate < fallback:
                fallback = candidate
            if expert_idx in protected_this_boundary:
                continue
            if expert_idx in round_loaded:
                continue
            if expert_idx in protected:
                continue
            if best is None or candidate < best:
                best = candidate
        if best is not None:
            return int(best[2])
        if fallback is not None:
            self._profile["dual_queue_protection_safety_valve_count"] += 1
            return int(fallback[2])
        return None

    def submit_segment_prefetch(
        self,
        *,
        step_id: int,
        target_layer_start: int,
        target_layer_end: int,
        target_segment_id: int,
        source: str,
        phase: str,
    ) -> int:
        if self._calibrating:
            return 0
        if source == self.GROUND_TRUTH_SOURCE:
            primary_index = self.ground_truth_index
            secondary_index = self.draft_predict_index
            secondary_source = self.DRAFT_SOURCE
        elif source == self.DRAFT_SOURCE:
            primary_index = self.draft_predict_index
            secondary_index = self.ground_truth_index
            secondary_source = self.GROUND_TRUTH_SOURCE
        else:
            raise ValueError(f"unsupported dual-queue source: {source}")

        if phase == "draft":
            budget = int(getattr(self.config, "draft_prefetch_max_per_boundary", 0))
        else:
            budget = self._verify_budget_for_segment(target_segment_id)
        inflight_budget = max(0, int(self.config.prefetch_max_inflight) - len(self.inflight))
        dispatch_budget = min(max(0, budget), inflight_budget)
        if dispatch_budget <= 0:
            return 0

        range_kwargs = dict(
            layer_start=int(target_layer_start),
            layer_end=int(target_layer_end),
            round_id=self._round_id,
            layer_caches=self.layer_caches,
            inflight_keys=self.inflight,
        )
        ranked_by_key: dict[tuple[int, int], PrefetchCandidate] = {}
        for c in primary_index.ranked_for_range(**range_kwargs, source=source):
            ranked_by_key[(c.layer_idx, c.expert_idx)] = c

        secondary_weight = float(self.config.dual_queue_secondary_index_weight)
        if secondary_weight > 0.0:
            for c in secondary_index.ranked_for_range(**range_kwargs, source=secondary_source):
                key = (c.layer_idx, c.expert_idx)
                penalized_priority = c.priority * secondary_weight
                prev = ranked_by_key.get(key)
                if prev is None or penalized_priority > prev.priority:
                    ranked_by_key[key] = PrefetchCandidate(
                        layer_idx=c.layer_idx,
                        expert_idx=c.expert_idx,
                        source=secondary_source,
                        score_sum=c.score_sum,
                        activation_count=c.activation_count,
                        first_seen_step=c.first_seen_step,
                        last_seen_step=c.last_seen_step,
                        priority=penalized_priority,
                    )

        ranked = sorted(ranked_by_key.values(), key=lambda c: (-c.priority, c.layer_idx, c.expert_idx))
        submitted = 0
        for candidate in ranked:
            if submitted >= dispatch_budget:
                break
            layer_idx = int(candidate.layer_idx)
            expert_idx = int(candidate.expert_idx)
            key = (layer_idx, expert_idx)
            cache = self.layer_caches.get(layer_idx)
            if cache is None or key in self.inflight:
                continue
            if cache.is_cached_cpu(expert_idx) or cache.is_pending_cpu(expert_idx):
                continue
            weights = self.cpu_expert_pool.get(layer_idx, {}).get(expert_idx)
            if not weights or "gate_up" not in weights or "down" not in weights:
                continue
            victim_slot = self._select_dual_queue_victim(
                cache, layer_idx=layer_idx, source=source,
            )
            if victim_slot is None:
                self._profile["dual_queue_all_slots_protected_count"] += 1
                continue
            reservation = cache.reserve_active_slot_for_prefetch_deferred(
                layer_idx=layer_idx, active_slot_idx=victim_slot, expert_idx=expert_idx,
            )
            if reservation is None:
                continue
            candidate_source = candidate.source
            ready_event, submit_ts_ms, enqueue_ms, num_bytes, stream_idx = self._begin_prefetch_transfer(
                cache=cache,
                reservation=reservation,
                weights=weights,
                source=candidate_source,
                direct_active=True,
            )
            ticket = PrefetchTicket(
                step_id=int(step_id),
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                source=candidate_source,
                staging_slot_idx=-1,
                staging_generation=-1,
                submit_ts_ms=submit_ts_ms,
                ready_event=ready_event,
                direct_active=True,
                active_slot_idx=reservation.active_slot_idx,
                active_generation=reservation.generation,
                active_slot_prev_expert=int(getattr(reservation, "prev_expert", -1)),
                segment_id=int(target_segment_id),
                num_bytes=num_bytes,
                transfer_enqueue_ms=enqueue_ms,
                transfer_stream_idx=stream_idx,
            )
            self.inflight[key] = ticket
            self._record_prefetch_submitted(ticket)
            submitted += 1
            self._profile["prefetch_submit_count"] += 1
            self._profile["direct_active_prefetch_submit_count"] += 1
            self._profile[f"{candidate_source}_submit_count"] += 1
        return submitted

    def _reap_expired_tickets(self) -> None:
        for key in list(self._expired_tickets):
            ticket = self.inflight.get(key)
            if ticket is None:
                self._expired_tickets.discard(key)
                continue
            if not bool(ticket.ready_event.query()):
                continue
            cache = self.layer_caches.get(ticket.layer_idx)
            if cache is not None:
                if ticket.direct_active:
                    cache.cancel_deferred_active_prefetch(
                        ActiveReservation(
                            layer_idx=ticket.layer_idx,
                            active_slot_idx=ticket.active_slot_idx,
                            expert_idx=ticket.expert_idx,
                            generation=ticket.active_generation,
                            prev_expert=ticket.active_slot_prev_expert,
                        )
                    )
                else:
                    cache.cancel_staging_reservation(
                        StagingReservation(
                            layer_idx=ticket.layer_idx,
                            staging_slot_idx=ticket.staging_slot_idx,
                            expert_idx=ticket.expert_idx,
                            generation=ticket.staging_generation,
                        )
                    )
            self.inflight.pop(key, None)
            self._expired_tickets.discard(key)
            self._profile["dual_queue_expired_transfer_count"] += 1
            self._profile["prefetch_late_count"] += 1
            self._record_prefetch_late(ticket)

    def _expire_unfinished_round_tickets(self) -> None:
        for key, ticket in self.inflight.items():
            if ticket.source not in {self.DRAFT_SOURCE, self.GROUND_TRUTH_SOURCE}:
                continue
            if key in self._expired_tickets:
                continue
            self._expired_tickets.add(key)
            self._profile["dual_queue_round_end_discard_count"] += 1

    def publish_segment_ready(self, *, step_id: int, segment_id: int) -> int:
        return self.publish_direct_active_ready(
            step_id=step_id,
            source=None,
        )

    def on_draft_segment_start(
        self,
        *,
        step_id: int,
        segment_id: int,
        boundaries: list[tuple[int, int]],
    ) -> int:
        return self.publish_direct_active_ready(step_id=step_id)

    def on_draft_segment_end(
        self,
        *,
        step_id: int,
        segment_id: int,
        boundaries: list[tuple[int, int]],
    ) -> int:
        return 0

    def on_verify_segment_start(
        self,
        *,
        step_id: int,
        segment_id: int,
        boundaries: list[tuple[int, int]],
    ) -> int:
        return self.publish_direct_active_ready(step_id=step_id)

    def submit_deferred_draft_segment(
        self,
        *,
        step_id: int,
        frontier_layer_idx: int,
        boundaries: list[tuple[int, int]],
    ) -> int:
        if self._calibrating:
            return 0
        num_segments = len(boundaries)
        if num_segments <= 1:
            return 0
        segment_id = self._segment_id_for_frontier(frontier_layer_idx, boundaries)
        submitted = 0
        if self.is_first_draft_forward():
            if segment_id == 0:
                target = 1
                start, end = boundaries[target]
                submitted += self.submit_segment_prefetch(
                    step_id=step_id, target_layer_start=start, target_layer_end=end,
                    target_segment_id=target, source=self.GROUND_TRUTH_SOURCE, phase="draft",
                )
            if segment_id <= num_segments - 3:
                target = segment_id + 2
                start, end = boundaries[target]
                submitted += self.submit_segment_prefetch(
                    step_id=step_id, target_layer_start=start, target_layer_end=end,
                    target_segment_id=target, source=self.GROUND_TRUTH_SOURCE, phase="draft",
                )
            elif segment_id == num_segments - 2:
                start, end = boundaries[0]
                submitted += self.submit_segment_prefetch(
                    step_id=step_id, target_layer_start=start, target_layer_end=end,
                    target_segment_id=0, source=self.DRAFT_SOURCE, phase="draft",
                )
            elif segment_id == num_segments - 1:
                start, end = boundaries[1]
                submitted += self.submit_segment_prefetch(
                    step_id=step_id, target_layer_start=start, target_layer_end=end,
                    target_segment_id=1, source=self.DRAFT_SOURCE, phase="draft",
                )
        else:
            target = (segment_id + 1) % num_segments
            start, end = boundaries[target]
            submitted += self.submit_segment_prefetch(
                step_id=step_id, target_layer_start=start, target_layer_end=end,
                target_segment_id=target, source=self.DRAFT_SOURCE, phase="draft",
            )
        self._profile["dual_queue_draft_phase_submit_count"] += submitted
        return submitted

    def submit_deferred_verify_segment(
        self,
        *,
        step_id: int,
        frontier_layer_idx: int,
        boundaries: list[tuple[int, int]],
    ) -> int:
        if self._calibrating or len(boundaries) <= 1:
            return 0
        num_segments = len(boundaries)
        segment_id = self._segment_id_for_frontier(frontier_layer_idx, boundaries)
        target = (segment_id + 1) % num_segments
        start, end = boundaries[target]
        source = (
            self.GROUND_TRUTH_SOURCE
            if segment_id == num_segments - 1
            else self.DRAFT_SOURCE
        )
        submitted = self.submit_segment_prefetch(
            step_id=step_id, target_layer_start=start, target_layer_end=end,
            target_segment_id=target, source=source, phase="verify",
        )
        self._profile["dual_queue_verify_phase_submit_count"] += submitted
        return submitted

    def submit_verify_segment_prefetch(
        self,
        *,
        step_id: int,
        target_layer_start: int,
        target_layer_end: int,
        target_segment_id: int,
        visible_budget_ms: float | None = None,
    ) -> int:
        """Synchronous verify prefetch merging both draft-predict and ground-truth indices."""
        if self._calibrating:
            return 0
        budget = max(0, self._verify_budget_for_segment(target_segment_id))
        inflight_budget = max(0, int(self.config.prefetch_max_inflight) - len(self.inflight))
        dispatch_budget = min(budget, inflight_budget)
        if dispatch_budget <= 0:
            return 0

        if visible_budget_ms is None:
            visible_budget_ms = self._visible_budget_ms_for_segment(target_segment_id)
        transfer_budget_ms = float(
            visible_budget_ms
            if visible_budget_ms is not None
            else getattr(self.config, "verify_prefetch_visible_budget_ms", 12.0)
        )

        range_kwargs = dict(
            layer_start=int(target_layer_start),
            layer_end=int(target_layer_end),
            round_id=self._round_id,
            layer_caches=self.layer_caches,
            inflight_keys=self.inflight,
        )

        ranked_by_key: dict[tuple[int, int], PrefetchCandidate] = {}
        for c in self.draft_predict_index.ranked_for_range(**range_kwargs, source=self.DRAFT_SOURCE):
            ranked_by_key[(c.layer_idx, c.expert_idx)] = c
        for c in self.ground_truth_index.ranked_for_range(**range_kwargs, source=self.GROUND_TRUTH_SOURCE):
            key = (c.layer_idx, c.expert_idx)
            prev = ranked_by_key.get(key)
            if prev is None or c.priority > prev.priority:
                ranked_by_key[key] = c

        ranked = sorted(ranked_by_key.values(), key=lambda c: (-c.priority, c.layer_idx, c.expert_idx))

        submitted = 0
        used_transfer_ms = 0.0
        for candidate in ranked:
            if submitted >= dispatch_budget:
                break
            layer_idx = int(candidate.layer_idx)
            expert_idx = int(candidate.expert_idx)
            key = (layer_idx, expert_idx)
            cache = self.layer_caches.get(layer_idx)
            if cache is None or key in self.inflight:
                continue
            if cache.is_cached_cpu(expert_idx) or cache.is_pending_cpu(expert_idx):
                continue
            weights = self.cpu_expert_pool.get(layer_idx, {}).get(expert_idx)
            if not weights or "gate_up" not in weights or "down" not in weights:
                continue
            transfer_ms = self._estimated_expert_transfer_ms(weights)
            if isfinite(transfer_ms) and transfer_budget_ms > 0.0 and used_transfer_ms + transfer_ms > transfer_budget_ms:
                break
            victim_slot = self._select_dual_queue_victim(
                cache, layer_idx=layer_idx, source=candidate.source,
            )
            if victim_slot is None:
                self._profile["dual_queue_all_slots_protected_count"] += 1
                continue
            reservation = cache.reserve_active_slot_for_prefetch_deferred(
                layer_idx=layer_idx, active_slot_idx=victim_slot, expert_idx=expert_idx,
            )
            if reservation is None:
                continue
            ready_event, submit_ts_ms, enqueue_ms, num_bytes, stream_idx = self._begin_prefetch_transfer(
                cache=cache, reservation=reservation, weights=weights,
                source=candidate.source, direct_active=True,
            )
            ticket = PrefetchTicket(
                step_id=int(step_id),
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                source=candidate.source,
                staging_slot_idx=-1,
                staging_generation=-1,
                submit_ts_ms=submit_ts_ms,
                ready_event=ready_event,
                direct_active=True,
                active_slot_idx=reservation.active_slot_idx,
                active_generation=reservation.generation,
                active_slot_prev_expert=int(getattr(reservation, "prev_expert", -1)),
                segment_id=int(target_segment_id),
                num_bytes=num_bytes,
                transfer_enqueue_ms=enqueue_ms,
                transfer_stream_idx=stream_idx,
            )
            self.inflight[key] = ticket
            self._record_prefetch_submitted(ticket)
            submitted += 1
            if isfinite(transfer_ms):
                used_transfer_ms += transfer_ms
            self._profile["prefetch_submit_count"] += 1
            self._profile["direct_active_prefetch_submit_count"] += 1
            self._profile[f"{candidate.source}_submit_count"] += 1
        self._profile["dual_queue_verify_phase_submit_count"] += submitted
        return submitted

    def _segment_id_for_frontier(self, frontier_layer_idx: int, boundaries: list[tuple[int, int]]) -> int:
        for seg_id, (start, end) in enumerate(boundaries):
            if start <= frontier_layer_idx < end:
                return seg_id
        return len(boundaries) - 1

    def complete_verify_round(self, *, step_id: int) -> None:
        # The final verify segment is the last overlap window for segment 0.
        self.publish_segment_ready(step_id=step_id, segment_id=0)
        self._expire_unfinished_round_tickets()
        self.finish_verify_round()
        self._reap_expired_tickets()

    def discard_verify_round(self) -> None:
        self._expire_unfinished_round_tickets()
        self.finish_verify_round()
        self._reap_expired_tickets()

    def drain_direct_active_ready(self, step_id: int, *, source: str | None = None) -> int:
        self._reap_expired_tickets()
        return self.publish_direct_active_ready(step_id=int(step_id), source=source)

    def wait_for_verify(self, step_id: int, timeout_ms: float) -> None:
        _ = timeout_ms
        self.publish_direct_active_ready(step_id=int(step_id))

    def get_profile(self, reset: bool = False) -> dict:
        extra = {
            "dual_queue_round_id": int(self._round_id),
            "dual_queue_draft_forward_index": int(self._draft_forward_index),
            "dual_queue_draft_budget": int(getattr(self.config, "draft_prefetch_max_per_boundary", 0)),
            "dual_queue_verify_budget": int(getattr(self.config, "verify_prefetch_max_per_boundary", 0)),
            "dual_queue_expert_transfer_ms": float(self._expert_transfer_ms),
            "dual_queue_target_miss_count": int(self._profile.get("dual_queue_target_miss_count", 0.0)),
            "dual_queue_round_end_discard_count": int(
                self._profile.get("dual_queue_round_end_discard_count", 0.0)
            ),
            "dual_queue_expired_transfer_count": int(
                self._profile.get("dual_queue_expired_transfer_count", 0.0)
            ),
            "dual_queue_stale_draft_metadata_count": int(
                self._profile.get("dual_queue_stale_draft_metadata_count", 0.0)
            ),
            "dual_queue_round_clear_count": int(
                self._profile.get("dual_queue_round_clear_count", 0.0)
            ),
            "dual_queue_all_slots_protected_count": int(
                self._profile.get("dual_queue_all_slots_protected_count", 0.0)
            ),
            "dual_queue_draft_phase_submit_count": int(
                self._profile.get("dual_queue_draft_phase_submit_count", 0.0)
            ),
            "dual_queue_verify_phase_submit_count": int(
                self._profile.get("dual_queue_verify_phase_submit_count", 0.0)
            ),
            "dual_queue_draft_predict_size": sum(
                len(entries) for entries in self.draft_predict_index.entries_by_segment.values()
            ),
            "dual_queue_ground_truth_size": sum(
                len(entries) for entries in self.ground_truth_index.entries_by_segment.values()
            ),
            "dual_queue_draft_predict_consumed_count": int(
                self._profile.get("dual_queue_draft_predict_consumed_count", 0.0)
            ),
            "dual_queue_ground_truth_consumed_count": int(
                self._profile.get("dual_queue_ground_truth_consumed_count", 0.0)
            ),
            "dual_queue_protection_safety_valve_count": int(
                self._profile.get("dual_queue_protection_safety_valve_count", 0.0)
            ),
        }
        if self._verify_segment_budgets:
            extra["dual_queue_verify_segment_budgets"] = dict(self._verify_segment_budgets)
            seg_vals = list(self._verify_segment_budgets.values())
            extra["dual_queue_verify_budget_min"] = min(seg_vals)
            extra["dual_queue_verify_budget_max"] = max(seg_vals)
        else:
            fallback = int(getattr(self.config, "verify_prefetch_max_per_boundary", 0))
            extra["dual_queue_verify_budget_min"] = fallback
            extra["dual_queue_verify_budget_max"] = fallback
        out = super().get_profile(reset=reset)
        out.update(extra)
        return out
