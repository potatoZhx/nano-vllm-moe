from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from math import isfinite

import torch

from nanovllm.config import Config
from nanovllm.expert.cache import ActiveReservation, LayerExpertCache, PublishedExpert, StagingReservation
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

    def clear(self) -> None:
        self.entries_by_segment.clear()

    def _segment_id(self, layer_idx: int) -> int:
        return int(layer_idx) // int(self.segment_size)

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
            segment_entries = self.entries_by_segment[self._segment_id(int(layer_idx))]
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
            stats["segment_index_entry_update_ms"] += (time.perf_counter() - update_t0) * 1000.0
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
        return out


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

        self.global_queue = GlobalWarmStartQueue(config)
        self.long_term_segment_index = SegmentCandidateIndex(config)
        self.draft_segment_index = SegmentCandidateIndex(config)
        self.transfer_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.metadata_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.publish_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        self.inflight: dict[tuple[int, int], PrefetchTicket] = {}
        self._profile = defaultdict(float)
        self._recent_published: dict[tuple[int, int], int] = {}
        self._recent_published_source: dict[tuple[int, int], str] = {}
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
        self._draft_m3_layers_by_step: dict[int, dict[int, bool]] = {}
        self._draft_m3_step0_steps: set[int] = set()
        self._draft_m3_next_is_step0 = True

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
    ) -> tuple[object, float, float, int]:
        submit_ts_ms = time.perf_counter() * 1000.0
        enqueue_t0 = time.perf_counter()
        if direct_active:
            ready_event = cache.begin_async_put_to_active(
                reservation=reservation,
                gate_up_cpu=weights["gate_up"],
                down_cpu=weights["down"],
                stream=self.transfer_stream,
            )
        else:
            ready_event = cache.begin_async_put_to_staging(
                reservation=reservation,
                gate_up_cpu=weights["gate_up"],
                down_cpu=weights["down"],
                stream=self.transfer_stream,
            )
        enqueue_ms = (time.perf_counter() - enqueue_t0) * 1000.0
        num_bytes = self._expert_weight_bytes(weights)
        self._profile["prefetch_transfer_enqueue_ms"] += enqueue_ms
        self._profile[f"{source}_prefetch_transfer_enqueue_ms"] += enqueue_ms
        self._prefetch_transfer_enqueue_ms_by_source[source] += enqueue_ms
        return ready_event, submit_ts_ms, enqueue_ms, num_bytes

    def _record_prefetch_submitted(self, ticket: PrefetchTicket) -> None:
        source = str(ticket.source)
        num_bytes = int(ticket.num_bytes)
        self._profile["prefetch_submitted_bytes"] += num_bytes
        self._profile[f"{source}_prefetch_submitted_bytes"] += num_bytes
        self._prefetch_submit_count_by_source[source] += 1
        self._prefetch_submitted_bytes_by_source[source] += num_bytes
        self._profile["prefetch_max_inflight_observed"] = max(
            float(self._profile.get("prefetch_max_inflight_observed", 0.0)),
            float(len(self.inflight)),
        )

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

    def _record_prefetch_published(self, ticket: PrefetchTicket) -> None:
        source = str(ticket.source)
        num_bytes = int(ticket.num_bytes)
        self._profile["prefetch_published_bytes"] += num_bytes
        self._profile[f"{source}_prefetch_published_bytes"] += num_bytes
        self._prefetch_published_count_by_source[source] += 1
        self._prefetch_published_bytes_by_source[source] += num_bytes

    def _record_prefetch_late(self, ticket: PrefetchTicket) -> None:
        source = str(ticket.source)
        num_bytes = int(ticket.num_bytes)
        self._profile["prefetch_late_bytes"] += num_bytes
        self._profile[f"{source}_prefetch_late_bytes"] += num_bytes
        self._prefetch_late_count_by_source[source] += 1
        self._prefetch_late_bytes_by_source[source] += num_bytes

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
        sid = int(step_id)
        if self._draft_m3_next_is_step0:
            self._draft_m3_step0_steps.add(sid)
            self._draft_m3_next_is_step0 = False

    def _record_draft_m3_cache_hits(
        self,
        runtime_meta: dict[int, LayerRuntimeMetaCPU] | None,
        step_id: int,
    ) -> None:
        if not runtime_meta or not self.layer_caches:
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
        self._update_rank_guard_scores(runtime_meta)
        if bool(self.config.prefetch_use_verify_history):
            return self.observe_runtime_meta(
                runtime_meta,
                source="verify_history",
                step_id=step_id,
                segment_index=self.long_term_segment_index if self._segment_indexed_enabled() else None,
            )
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

            ready_event, submit_ts_ms, enqueue_ms, num_bytes = self._begin_prefetch_transfer(
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

            ready_event, submit_ts_ms, enqueue_ms, num_bytes = self._begin_prefetch_transfer(
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

            ready_event, submit_ts_ms, enqueue_ms, num_bytes = self._begin_prefetch_transfer(
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
        if not bool(getattr(self.config, "prefetch_verify_layer_enabled", True)):
            return 0
        if int(self.config.prefetch_step_budget) <= 0:
            return 0
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

        inflight_keys = set(self.inflight.keys())
        ranked = self.global_queue.ranked_candidates(
            step_id=step_id,
            layer_caches=self.layer_caches,
            inflight_keys=inflight_keys,
        )
        ranked = [c for c in self.prefetch_strategy.rank(ranked, step_id=step_id) if int(c.layer_idx) == layer_idx]
        if not ranked:
            return 0

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
            if cache.is_cached_cpu(expert_idx):
                continue
            if key in self.inflight:
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
                cache,
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                step_id=step_id,
            )
            if victim_slot is None:
                continue

            reservation = cache.reserve_active_slot_for_prefetch(
                layer_idx=layer_idx,
                active_slot_idx=victim_slot,
                expert_idx=expert_idx,
            )
            if reservation is None:
                continue

            ready_event, submit_ts_ms, enqueue_ms, num_bytes = self._begin_prefetch_transfer(
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
            )
            self.inflight[key] = ticket
            self._record_prefetch_submitted(ticket)
            submitted += 1
            used_budget_ms += transfer_ms
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

            self._finalize_publish(cache, published_item)
            self.inflight.pop((ticket.layer_idx, ticket.expert_idx), None)
            self._recent_published[(ticket.layer_idx, ticket.expert_idx)] = int(step_id)
            self._recent_published_source[(ticket.layer_idx, ticket.expert_idx)] = str(ticket.source)
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
            if ticket.source == "draft_segment_indexed":
                published_item = cache.commit_deferred_active_prefetch(reservation)
            else:
                published_item = cache.commit_active_prefetch(reservation)
            self.inflight.pop(key, None)
            if published_item is None:
                if ticket.source == "draft_segment_indexed":
                    cache.cancel_deferred_active_prefetch(reservation)
                self._profile["prefetch_late_count"] += 1
                self._record_prefetch_late(ticket)
                continue
            self._recent_published[(ticket.layer_idx, ticket.expert_idx)] = int(step_id)
            self._recent_published_source[(ticket.layer_idx, ticket.expert_idx)] = str(ticket.source)
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
        for ticket in list(self.inflight.values()):
            if not ticket.direct_active:
                continue
            if source is not None and str(ticket.source) != str(source):
                continue
            ticket.ready_event.synchronize()
            waited += 1
            if ticket.source == "draft_direct_active":
                draft_waited += 1
            elif ticket.source == "draft_segment_indexed":
                indexed_waited += 1
        published = self.publish_direct_active_ready(step_id=step_id, source=source)
        drain_ms = (time.perf_counter() - t0) * 1000.0
        self._profile["direct_active_prefetch_drain_count"] += waited
        self._profile["direct_active_prefetch_drain_ms"] += drain_ms
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

        if timeout_ms > 0.0:
            deadline = t0 + timeout_ms / 1000.0
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

        self._profile["prefetch_wait_ms"] += (time.perf_counter() - t0) * 1000.0
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
        self._profile["prefetch_consumed_count"] += consumed

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
            "prefetch_wait_ms": float(self._profile.get("prefetch_wait_ms", 0.0)),
            "prefetch_consumed_count": int(self._profile.get("prefetch_consumed_count", 0.0)),
            "prefetch_timeout_count": int(self._profile.get("prefetch_timeout_count", 0.0)),
            "publish_count": int(self._profile.get("publish_count", 0.0)),
            "publish_ms": float(self._profile.get("publish_ms", 0.0)),
            "observe_runtime_meta_count": int(self._profile.get("observe_runtime_meta_count", 0.0)),
            "observe_runtime_meta_ms": float(self._profile.get("observe_runtime_meta_ms", 0.0)),
            "observe_mark_access_ms": float(self._profile.get("observe_mark_access_ms", 0.0)),
            "observe_queue_update_ms": float(self._profile.get("observe_queue_update_ms", 0.0)),
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
            "draft_direct_active_prefetch_submit_count": int(self._profile.get("draft_direct_active_prefetch_submit_count", 0.0)),
            "draft_direct_active_prefetch_ready_count": int(self._profile.get("draft_direct_active_prefetch_ready_count", 0.0)),
            "draft_direct_active_prefetch_publish_count": int(self._profile.get("draft_direct_active_prefetch_publish_count", 0.0)),
            "draft_direct_active_prefetch_consumed_count": int(self._profile.get("draft_direct_active_prefetch_consumed_count", 0.0)),
            "draft_direct_active_prefetch_drain_count": int(self._profile.get("draft_direct_active_prefetch_drain_count", 0.0)),
            "draft_direct_active_prefetch_drain_ms": float(self._profile.get("draft_direct_active_prefetch_drain_ms", 0.0)),
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

        best_slot: int | None = None
        best_value: int | None = None
        fb_slot: int | None = None      # fallback ignoring protection (safety valve)
        fb_value: int | None = None
        for slot_idx, slot_expert in enumerate(cache.slot_to_expert):
            if cache.is_active_slot_pending(slot_idx):
                continue
            e = int(slot_expert)
            if e < 0:
                return slot_idx  # empty slot: no eviction needed
            value = int(cache.access_count[e]) if use_lfu else int(cache.last_access_step[e])
            if fb_value is None or value < fb_value:
                fb_value = value
                fb_slot = slot_idx
            if li is not None and self._round_protected(li, e):
                continue
            if rankguard is not None and li is not None and rankguard.is_protected(li, e):
                continue
            if best_value is None or value < best_value:
                best_value = value
                best_slot = slot_idx
        if best_slot is not None:
            return best_slot
        return fb_slot  # all protected -> safety valve keeps prefetch progressing

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

        # Highest ground-truth-frequency uncached experts in segment n-1.
        candidates: list[tuple[int, int, int]] = []
        for layer_idx, cache in self.layer_caches.items():
            if self._segment_id_for_layer(int(layer_idx)) != last_segment_id:
                continue
            pool = self.cpu_expert_pool.get(layer_idx)
            if not pool:
                continue
            ac = cache.access_count
            for expert_idx in pool.keys():
                eid = int(expert_idx)
                if cache.is_cached_cpu(eid) or cache.is_pending_cpu(eid):
                    continue
                candidates.append((int(ac[eid]), int(layer_idx), eid))
        if not candidates:
            return 0
        candidates.sort(key=lambda x: (-x[0], x[1], x[2]))

        inflight_budget = max(0, int(self.config.prefetch_max_inflight) - len(self.inflight))
        max_submit = min(budget, inflight_budget)
        submitted = 0
        for _freq, layer_idx, expert_idx in candidates:
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
            ready_event, submit_ts_ms, enqueue_ms, num_bytes = self._begin_prefetch_transfer(
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
            ready_event, submit_ts_ms, enqueue_ms, num_bytes = self._begin_prefetch_transfer(
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
        }
        out = super().get_profile(reset=reset)
        out.update(extra)
        return out
