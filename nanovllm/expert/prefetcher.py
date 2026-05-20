from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from math import isfinite

import torch

from nanovllm.config import Config
from nanovllm.expert.cache import ActiveReservation, LayerExpertCache, PublishedExpert, StagingReservation
from nanovllm.expert.runtime_meta import LayerRuntimeMetaCPU, ModelRuntimeMetaRecorder
from nanovllm.scheduling.cache_strategy import CacheStrategy
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
    ) -> list[PrefetchCandidate]:
        self.prune(step_id, layer_caches)
        ranked: list[PrefetchCandidate] = []
        for key, entry in self.entries.items():
            layer_idx, expert_idx = key
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
        self.transfer_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.metadata_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.publish_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        self.inflight: dict[tuple[int, int], PrefetchTicket] = {}
        self._profile = defaultdict(float)
        self._recent_published: dict[tuple[int, int], int] = {}

    def observe_runtime_meta(
        self,
        runtime_meta: dict[int, LayerRuntimeMetaCPU] | None,
        source: str,
        step_id: int,
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
        queue_stats = self.global_queue.update_from_runtime_meta(
            runtime_meta=runtime_meta,
            source=source,
            step_id=step_id,
            layer_caches=self.layer_caches,
        )
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
        }
        return out

    def observe_prefill(self, runtime_meta: dict[int, LayerRuntimeMetaCPU] | None, step_id: int) -> dict[str, float]:
        if bool(self.config.prefetch_use_prefill_history):
            return self.observe_runtime_meta(runtime_meta, source="prefill_history", step_id=step_id)
        return {}

    def observe_draft(self, runtime_meta: dict[int, LayerRuntimeMetaCPU] | None, step_id: int) -> dict[str, float]:
        if bool(self.config.prefetch_use_draft_live):
            return self.observe_runtime_meta(runtime_meta, source="draft_live", step_id=step_id)
        return {}

    def observe_verify(self, runtime_meta: dict[int, LayerRuntimeMetaCPU] | None, step_id: int) -> dict[str, float]:
        if bool(self.config.prefetch_use_verify_history):
            return self.observe_runtime_meta(runtime_meta, source="verify_history", step_id=step_id)
        return {}

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

            ready_event = cache.begin_async_put_to_staging(
                reservation=reservation,
                gate_up_cpu=weights["gate_up"],
                down_cpu=weights["down"],
                stream=self.transfer_stream,
            )

            self.inflight[key] = PrefetchTicket(
                step_id=step_id,
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                source=candidate.source,
                staging_slot_idx=reservation.staging_slot_idx,
                staging_generation=reservation.generation,
                submit_ts_ms=time.perf_counter() * 1000.0,
                ready_event=ready_event,
                ready=False,
            )
            submitted += 1
            self._profile["prefetch_submit_count"] += 1
            if candidate.source == "prefill_history":
                self._profile["history_prefetch_submit_count"] += 1
            elif candidate.source == "verify_history":
                self._profile["verify_history_prefetch_submit_count"] += 1
            elif candidate.source == "draft_live":
                self._profile["draft_live_prefetch_submit_count"] += 1

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

            ready_event = cache.begin_async_put_to_active(
                reservation=reservation,
                gate_up_cpu=weights["gate_up"],
                down_cpu=weights["down"],
                stream=self.transfer_stream,
            )

            self.inflight[key] = PrefetchTicket(
                step_id=step_id,
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                source="verify_layer_predict",
                staging_slot_idx=-1,
                staging_generation=-1,
                submit_ts_ms=time.perf_counter() * 1000.0,
                ready_event=ready_event,
                ready=False,
                direct_active=True,
                active_slot_idx=reservation.active_slot_idx,
                active_generation=reservation.generation,
                active_slot_prev_expert=int(getattr(reservation, "prev_expert", -1)),
            )
            submitted += 1
            used_budget_ms += transfer_ms
            self._profile["prefetch_submit_count"] += 1
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
                continue

            ticket.ready = True
            ready.append(ticket)
            self._profile["prefetch_completed_count"] += 1
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
                continue

            self._finalize_publish(cache, published_item)
            self.inflight.pop((ticket.layer_idx, ticket.expert_idx), None)
            self._recent_published[(ticket.layer_idx, ticket.expert_idx)] = int(step_id)
            published += 1
            self._profile["publish_count"] += 1

        self._profile["publish_ms"] += (time.perf_counter() - t0) * 1000.0
        return published

    def publish_direct_active_ready(self, step_id: int) -> int:
        t0 = time.perf_counter()
        published = 0
        for key, ticket in list(self.inflight.items()):
            if not ticket.direct_active:
                continue
            if not bool(ticket.ready_event.query()):
                continue
            cache = self.layer_caches.get(ticket.layer_idx)
            if cache is None:
                self.inflight.pop(key, None)
                continue
            reservation = ActiveReservation(
                layer_idx=ticket.layer_idx,
                active_slot_idx=ticket.active_slot_idx,
                expert_idx=ticket.expert_idx,
                generation=ticket.active_generation,
                prev_expert=ticket.active_slot_prev_expert,
            )
            published_item = cache.commit_active_prefetch(reservation)
            self.inflight.pop(key, None)
            if published_item is None:
                self._profile["prefetch_late_count"] += 1
                continue
            self._recent_published[(ticket.layer_idx, ticket.expert_idx)] = int(step_id)
            published += 1
            self._profile["prefetch_completed_count"] += 1
            self._profile["publish_count"] += 1
            self._profile["verify_layer_prefetch_publish_count"] += 1
        self._profile["publish_ms"] += (time.perf_counter() - t0) * 1000.0
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
        self._profile["prefetch_consumed_count"] += consumed

        stale = []
        ttl = int(self.config.prefetch_history_ttl_steps)
        for key, published_step in self._recent_published.items():
            if int(step_id) - int(published_step) > ttl:
                stale.append(key)
        for key in stale:
            self._recent_published.pop(key, None)

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
            "history_prefetch_submit_count": int(self._profile.get("history_prefetch_submit_count", 0.0)),
            "verify_history_prefetch_submit_count": int(self._profile.get("verify_history_prefetch_submit_count", 0.0)),
            "draft_live_prefetch_submit_count": int(self._profile.get("draft_live_prefetch_submit_count", 0.0)),
            "verify_layer_prefetch_submit_count": int(self._profile.get("verify_layer_prefetch_submit_count", 0.0)),
            "verify_layer_prefetch_publish_count": int(self._profile.get("verify_layer_prefetch_publish_count", 0.0)),
            "verify_layer_prefetch_budget_stop_count": int(self._profile.get("verify_layer_prefetch_budget_stop_count", 0.0)),
            "verify_layer_prefetch_est_transfer_ms": float(self._profile.get("verify_layer_prefetch_est_transfer_ms", 0.0)),
            "verify_layer_prefetch_available_ms": float(self._profile.get("verify_layer_prefetch_available_ms", 0.0)),
            "verify_layer_prefetch_used_budget_ms": float(self._profile.get("verify_layer_prefetch_used_budget_ms", 0.0)),
            "verify_ready_before_wait_count": int(self._profile.get("verify_ready_before_wait_count", 0.0)),
            "verify_ready_after_wait_count": int(self._profile.get("verify_ready_after_wait_count", 0.0)),
        }
        if reset:
            self._profile.clear()
        return out
