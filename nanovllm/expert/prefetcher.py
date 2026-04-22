from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass

import torch

from nanovllm.config import Config
from nanovllm.expert.cache import LayerExpertCache, PublishedExpert, StagingReservation
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
    ) -> None:
        if not runtime_meta:
            return

        for layer_idx, meta in runtime_meta.items():
            cache = layer_caches.get(layer_idx)
            if cache is None:
                continue

            flat_experts = meta.selected_experts.reshape(-1).to(device="cpu", dtype=torch.int64)
            if flat_experts.numel() == 0:
                continue
            flat_weights = meta.routing_weights.reshape(-1).to(device="cpu", dtype=torch.float32)
            unique_ids, inverse = torch.unique(flat_experts, return_inverse=True)
            # Keep aggregation tensors on CPU to avoid default-device leakage (e.g. torch.device("cuda") context).
            score_sum = torch.zeros((unique_ids.numel(),), dtype=torch.float32, device=torch.device("cpu"))
            score_sum.scatter_add_(0, inverse, flat_weights)
            counts = torch.zeros((unique_ids.numel(),), dtype=torch.int64, device=torch.device("cpu"))
            counts.scatter_add_(0, inverse, torch.ones_like(inverse, dtype=torch.int64))

            cached_mask_all = cache.get_cached_expert_mask().detach().to("cpu")
            cached_mask = cached_mask_all.index_select(0, unique_ids)

            for i, expert_idx in enumerate(unique_ids.tolist()):
                if bool(cached_mask[i].item()):
                    continue
                key = (int(layer_idx), int(expert_idx))
                new_score = float(score_sum[i].item())
                new_count = int(counts[i].item())
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
            if bool(cache.get_cached_expert_mask()[expert_idx].item()):
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
            if bool(cache.get_cached_expert_mask()[expert_idx].item()):
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
    ) -> None:
        if runtime_meta is None:
            return
        for layer_idx, meta in runtime_meta.items():
            cache = self.layer_caches.get(layer_idx)
            if cache is None:
                continue
            cache.mark_access(meta.selected_experts, meta.routing_weights, step_id=step_id)

        self.global_queue.update_from_runtime_meta(
            runtime_meta=runtime_meta,
            source=source,
            step_id=step_id,
            layer_caches=self.layer_caches,
        )

    def observe_prefill(self, runtime_meta: dict[int, LayerRuntimeMetaCPU] | None, step_id: int) -> None:
        if bool(self.config.prefetch_use_prefill_history):
            self.observe_runtime_meta(runtime_meta, source="prefill_history", step_id=step_id)

    def observe_draft(self, runtime_meta: dict[int, LayerRuntimeMetaCPU] | None, step_id: int) -> None:
        if bool(self.config.prefetch_use_draft_live):
            self.observe_runtime_meta(runtime_meta, source="draft_live", step_id=step_id)

    def observe_verify(self, runtime_meta: dict[int, LayerRuntimeMetaCPU] | None, step_id: int) -> None:
        if bool(self.config.prefetch_use_verify_history):
            self.observe_runtime_meta(runtime_meta, source="verify_history", step_id=step_id)

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
            if bool(cache.get_cached_expert_mask()[expert_idx].item()):
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

    def poll_ready_tickets(self) -> list[PrefetchTicket]:
        ready: list[PrefetchTicket] = []
        for key, ticket in list(self.inflight.items()):
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
            snapshot = cache.snapshot(layer_idx=ticket.layer_idx)
            victim_slot = self.cache_strategy.select_victim_slot(
                snapshot=snapshot,
                incoming_expert_idx=ticket.expert_idx,
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

    def _finalize_publish(self, cache: LayerExpertCache, published_item: PublishedExpert) -> None:
        if self.publish_stream is not None:
            torch.cuda.current_stream().wait_stream(self.publish_stream)
        cache.commit_published_expert(published_item)

    def wait_for_verify(self, step_id: int, timeout_ms: float) -> None:
        t0 = time.perf_counter()
        self.publish_ready(step_id=step_id)
        self._profile["verify_ready_before_wait_count"] += self._count_ready_relevant_experts()

        if timeout_ms > 0.0:
            deadline = t0 + timeout_ms / 1000.0
            while time.perf_counter() < deadline:
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
            unique_ids = torch.unique(meta.selected_experts.reshape(-1).to(torch.int64)).tolist()
            for expert_idx in unique_ids:
                key = (int(layer_idx), int(expert_idx))
                if key not in self._recent_published:
                    continue
                if bool(cache.get_cached_expert_mask()[expert_idx].item()):
                    consumed += 1
        self._profile["prefetch_consumed_count"] += consumed

        stale = []
        ttl = int(self.config.prefetch_history_ttl_steps)
        for key, published_step in self._recent_published.items():
            if int(step_id) - int(published_step) > ttl:
                stale.append(key)
        for key in stale:
            self._recent_published.pop(key, None)

    def record_metadata_offload(self, dt_ms: float, num_bytes: int) -> None:
        self._profile["metadata_offload_ms"] += float(dt_ms)
        self._profile["metadata_offload_bytes"] += float(num_bytes)

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
            "metadata_offload_ms": float(self._profile.get("metadata_offload_ms", 0.0)),
            "metadata_offload_bytes": float(self._profile.get("metadata_offload_bytes", 0.0)),
            "history_prefetch_submit_count": int(self._profile.get("history_prefetch_submit_count", 0.0)),
            "verify_history_prefetch_submit_count": int(self._profile.get("verify_history_prefetch_submit_count", 0.0)),
            "draft_live_prefetch_submit_count": int(self._profile.get("draft_live_prefetch_submit_count", 0.0)),
            "verify_ready_before_wait_count": int(self._profile.get("verify_ready_before_wait_count", 0.0)),
            "verify_ready_after_wait_count": int(self._profile.get("verify_ready_after_wait_count", 0.0)),
        }
        if reset:
            self._profile.clear()
        return out
