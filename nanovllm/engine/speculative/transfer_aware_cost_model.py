"""Transfer-aware verify demand and latency prediction (artifact schema v3).

The legacy verify-cost proxy estimates CPU work from a frozen cache snapshot.
This module deliberately keeps a separate artifact and runtime contract:

``draft routes -> verify demand -> shadow cache/transfers -> CPU work -> latency``.

The shadow simulator contains no tensors and never mutates the live cache or
prefetch runtime.  Its immutable snapshots also make the alignment and transfer
state transitions straightforward to test with synthetic traces.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 3
SIMULATOR_SEMANTICS_VERSION = "transfer-shadow-v3-soft-distinct-v1"


def expected_distinct_experts(counts: np.ndarray, axis=None):
    """Compute a soft distinct-expert count from expected route masses.

    Integer truth counts retain the exact nonzero-expert count. Predicted
    demand is fractional and dense because it mixes calibrated priors, so
    treating every epsilon as one full CPU expert would grossly inflate cost.
    """
    values = np.asarray(counts)
    return np.minimum(np.maximum(values, 0.0), 1.0).sum(axis=axis)


def compute_model_id(artifact: Mapping[str, object]) -> str:
    """Return an identity bound to every component used by active policy."""
    payload = {
        key: value
        for key, value in artifact.items()
        if key not in {"model_id", "created_at", "notes", "observed_results"}
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _as_array(
    value: object,
    *,
    shape: tuple[int, ...],
    default: float,
) -> np.ndarray:
    if value is None:
        return np.full(shape, default, dtype=np.float32)
    out = np.asarray(value, dtype=np.float32)
    if out.shape == shape:
        return out.copy()
    try:
        return np.broadcast_to(out, shape).astype(np.float32, copy=True)
    except ValueError as error:
        raise ValueError(f"cannot broadcast calibration shape {out.shape} to {shape}") from error


def select_verify_bucket(logical_tokens: int, buckets: Sequence[int]) -> int:
    logical_tokens = max(1, int(logical_tokens))
    for bucket in sorted({int(value) for value in buckets}):
        if bucket >= logical_tokens:
            return bucket
    raise ValueError(
        f"no verify graph bucket for {logical_tokens} logical tokens; "
        f"available={list(buckets)}"
    )


@dataclass(frozen=True)
class ShadowTransfer:
    layer_idx: int
    expert_idx: int
    slot_idx: int
    previous_expert: int
    remaining_ms: float
    source: str
    deferred_publish: bool = True


@dataclass(frozen=True)
class ShadowLayerState:
    slots: tuple[int, ...]
    pending: tuple[int, ...]
    access_values: tuple[float, ...]
    protected_experts: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        if len(self.slots) != len(self.pending):
            raise ValueError("shadow slots and pending arrays must have equal length")


@dataclass(frozen=True)
class ShadowCacheState:
    layers: tuple[ShadowLayerState, ...]
    inflight: tuple[ShadowTransfer, ...] = ()
    elapsed_ms: float = 0.0
    array_state: object | None = field(
        default=None, compare=False, repr=False
    )


@dataclass(frozen=True)
class RuntimeCacheHostMatrices:
    """Contiguous live host mirrors used by boundary snapshots."""

    resident: np.ndarray
    last_access: np.ndarray
    access_count: np.ndarray
    slot_counts: np.ndarray


@dataclass(frozen=True)
class SegmentWorkload:
    segment_id: int
    first_layer: int
    last_layer: int
    cpu_experts: float
    cpu_routes: float
    max_layer_experts: float
    transfer_submits: int


@dataclass(frozen=True)
class ShadowSimulation:
    state: ShadowCacheState
    cpu_route_counts: np.ndarray
    cpu_experts: float
    cpu_routes: float
    segments: tuple[SegmentWorkload, ...]
    transfer_submits: int
    transfer_pending: int


@dataclass(frozen=True)
class ShadowWorkload:
    """Latency-facing result from the allocation-free runtime fast path."""

    cpu_route_counts: np.ndarray
    cpu_experts: float
    cpu_routes: float
    segments: tuple[SegmentWorkload, ...]
    transfer_submits: int
    transfer_pending: int


@dataclass
class _ArrayShadowState:
    slots: np.ndarray
    pending_slots: np.ndarray
    valid_slots: np.ndarray
    access_values: np.ndarray
    protected: np.ndarray
    resident: np.ndarray
    pending_experts: np.ndarray
    inflight: list[ShadowTransfer]
    empty_slots: np.ndarray | None = None
    reserved_victims: np.ndarray | None = None
    victim_hints: np.ndarray | None = None
    expert_only: bool = False

    def clone(self) -> "_ArrayShadowState":
        return _ArrayShadowState(
            slots=self.slots.copy(),
            pending_slots=self.pending_slots.copy(),
            valid_slots=self.valid_slots,
            access_values=self.access_values,
            protected=self.protected.copy(),
            resident=self.resident.copy(),
            pending_experts=self.pending_experts.copy(),
            inflight=list(self.inflight),
            empty_slots=(
                None
                if self.empty_slots is None
                else self.empty_slots.copy()
            ),
            reserved_victims=(
                None
                if self.reserved_victims is None
                else self.reserved_victims.copy()
            ),
            victim_hints=(
                None
                if self.victim_hints is None
                else self.victim_hints.copy()
            ),
            expert_only=bool(self.expert_only),
        )


@dataclass(frozen=True)
class TransferAwarePrediction:
    total_ms: float
    error_p90_ms: float
    bucket: int
    logical_tokens: int
    cpu_experts: float
    cpu_routes: float
    transfer_submits: int
    transfer_pending: int
    segments: tuple[SegmentWorkload, ...]


def rank_demand_candidates(
    demand: np.ndarray,
    *,
    layer_indices: Sequence[int],
    state: ShadowCacheState,
) -> list[tuple[float, int, int]]:
    """Rank uncached predicted experts using the runtime's stable tie-breaks."""
    scores, layers, experts = _rank_demand_candidate_arrays(
        demand,
        layer_indices=layer_indices,
        state=state,
    )
    return [
        (float(score), int(layer_idx), int(expert_idx))
        for score, layer_idx, expert_idx in zip(
            scores, layers, experts, strict=True
        )
    ]


def _rank_demand_candidate_arrays(
    demand: np.ndarray,
    *,
    layer_indices: Sequence[int],
    state: ShadowCacheState,
    candidate_limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized candidate filtering with score/layer/expert stable order."""
    layer_array = np.fromiter(
        (int(value) for value in layer_indices), dtype=np.int64
    )
    if layer_array.size == 0:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty.astype(np.int64), empty.astype(np.int64)
    selected = np.asarray(demand, dtype=np.float32)[layer_array]
    eligible = selected > 0.0
    inflight = {
        (int(ticket.layer_idx), int(ticket.expert_idx))
        for ticket in state.inflight
    }
    for local_idx, layer_idx in enumerate(layer_array):
        layer_idx = int(layer_idx)
        layer = state.layers[layer_idx]
        blocked = [
            int(value)
            for value in (*layer.slots, *layer.pending)
            if 0 <= int(value) < selected.shape[1]
        ]
        blocked.extend(
            expert_idx
            for inflight_layer, expert_idx in inflight
            if inflight_layer == layer_idx
        )
        if blocked:
            eligible[local_idx, np.asarray(blocked, dtype=np.int64)] = False
    flat_indices = np.flatnonzero(eligible)
    if flat_indices.size == 0:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty.astype(np.int64), empty.astype(np.int64)
    flat_scores = selected.reshape(-1)[flat_indices]
    if (
        candidate_limit is not None
        and 0 < int(candidate_limit) < flat_indices.size
    ):
        limit = int(candidate_limit)
        threshold = np.partition(
            flat_scores, flat_scores.size - limit
        )[flat_scores.size - limit]
        above = flat_indices[flat_scores > threshold]
        tied = flat_indices[flat_scores == threshold]
        flat_indices = np.concatenate(
            (above, tied[: max(0, limit - above.size)])
        )
    local_layers = flat_indices // selected.shape[1]
    experts = flat_indices % selected.shape[1]
    layers = layer_array[local_layers]
    scores = selected[local_layers, experts]
    order = np.lexsort((experts, layers, -scores))
    return scores[order], layers[order], experts[order]


def select_shadow_victim(layer: ShadowLayerState) -> int | None:
    """Mirror predictive LRU/LFU selection over a copied layer state."""
    from nanovllm.expert.prefetcher import select_predictive_victim_slot

    return select_predictive_victim_slot(
        slots=layer.slots,
        pending=layer.pending,
        access_values=layer.access_values,
        protected_experts=layer.protected_experts,
    )


class CalibratedVerifyDemandPredictor:
    """Compact layer/position calibration from original draft routes.

    Draft forward ``i`` predicts verify logical row ``i - 1``.  After K draft
    forwards, one additional verify-next row remains unknown and is predicted
    from the most recent calibrated row plus a position prior.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_experts: int,
        top_k: int,
        max_draft_tokens: int,
        artifact: Mapping[str, object],
    ) -> None:
        self.num_layers = int(num_layers)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.max_draft_tokens = int(max_draft_tokens)
        positions = self.max_draft_tokens + 1
        self.retention = np.clip(
            _as_array(
                artifact.get("retention"),
                shape=(positions, self.num_layers),
                default=0.75,
            ),
            0.0,
            1.0,
        )
        uniform = float(self.top_k) / float(max(1, self.num_experts))
        self.layer_prior = _as_array(
            artifact.get("layer_prior"),
            shape=(self.num_layers, self.num_experts),
            default=uniform,
        )
        self.position_prior = _as_array(
            artifact.get("position_prior"),
            shape=(positions, self.num_layers, self.num_experts),
            default=uniform,
        )
        self.padding_prior = _as_array(
            artifact.get("padding_prior"),
            shape=(self.num_layers, self.num_experts),
            default=0.0,
        )
        self.next_recent_weight = np.clip(
            _as_array(
                artifact.get("next_recent_weight"),
                shape=(positions, self.num_layers),
                default=0.6,
            ),
            0.0,
            1.0,
        )
        self._flat_route_layers = np.repeat(
            np.arange(self.num_layers, dtype=np.int64),
            self.top_k,
        )
        self._one_hot_scratch = np.zeros(
            (self.num_layers, self.num_experts), dtype=np.float32
        )
        self._forecast_scratch = np.zeros_like(
            self._one_hot_scratch
        )
        self._forecast_delta_scratch = np.zeros_like(
            self._one_hot_scratch
        )
        self._normalize_priors()
        self.reset()

    def _normalize_rows(self, rows: np.ndarray, target: float) -> np.ndarray:
        rows = np.maximum(rows, 0.0)
        totals = rows.sum(axis=-1, keepdims=True)
        fallback = np.full_like(rows, target / float(max(1, self.num_experts)))
        return np.where(
            totals > 1e-8,
            rows * (target / np.maximum(totals, 1e-8)),
            fallback,
        ).astype(np.float32, copy=False)

    def _normalize_priors(self) -> None:
        self.layer_prior = self._normalize_rows(self.layer_prior, float(self.top_k))
        self.position_prior = self._normalize_rows(
            self.position_prior, float(self.top_k)
        )
        # A padded execution row may still execute captured routing.  Preserve a
        # fitted zero row if calibration observed no padding work.
        totals = self.padding_prior.sum(axis=-1, keepdims=True)
        self.padding_prior = np.where(
            totals > 1e-8,
            self.padding_prior
            * (float(self.top_k) / np.maximum(totals, 1e-8)),
            0.0,
        ).astype(np.float32, copy=False)

    def reset(self) -> None:
        self._observed: list[np.ndarray] = []
        self._observed_sum = np.zeros(
            (self.num_layers, self.num_experts), dtype=np.float32
        )

    @property
    def known_rows(self) -> int:
        return len(self._observed)

    def observe(self, original_routes: np.ndarray) -> None:
        routes = np.asarray(original_routes)
        if routes.ndim == 2:
            routes = routes[:, None, :]
        expected = (self.num_layers, routes.shape[1], self.top_k)
        if routes.ndim != 3 or routes.shape != expected:
            raise ValueError(
                "draft routes must be [layers,batch,top_k], got "
                f"{routes.shape}; expected layers={self.num_layers}, top_k={self.top_k}"
            )
        for batch_idx in range(routes.shape[1]):
            position = min(len(self._observed), self.max_draft_tokens)
            one_hot = self._one_hot_scratch
            one_hot.fill(0.0)
            experts = np.asarray(
                routes[:, batch_idx, :], dtype=np.int64
            ).reshape(-1)
            valid = (experts >= 0) & (experts < self.num_experts)
            np.add.at(
                one_hot,
                (
                    self._flat_route_layers[valid],
                    experts[valid],
                ),
                1.0,
            )
            keep = self.retention[position, :, None]
            calibrated = (
                keep * one_hot + (1.0 - keep) * self.layer_prior
            )
            calibrated = self._normalize_rows(
                calibrated, float(self.top_k)
            )
            self._observed.append(calibrated)
            self._observed_sum += calibrated

    def predict_aggregate(
        self,
        *,
        logical_tokens: int,
        bucket: int,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict aggregate demand without materializing token rows."""
        logical_tokens = int(logical_tokens)
        bucket = int(bucket)
        if logical_tokens < 1 or bucket < logical_tokens:
            raise ValueError("invalid logical/bucket token counts")
        known_count = min(
            len(self._observed), max(0, logical_tokens - 1)
        )
        if out is None:
            aggregate = np.empty_like(self._observed_sum)
        else:
            aggregate = np.asarray(out, dtype=np.float32)
            if aggregate.shape != self._observed_sum.shape:
                raise ValueError(
                    "aggregate output shape mismatch: "
                    f"{aggregate.shape} != {self._observed_sum.shape}"
                )
        if known_count == 0:
            aggregate.fill(0.0)
        elif known_count == len(self._observed):
            np.copyto(aggregate, self._observed_sum)
        else:
            aggregate.fill(0.0)
            for row in self._observed[:known_count]:
                aggregate += row
        recent = (
            self._observed[known_count - 1]
            if known_count
            else self.layer_prior
        )
        position = known_count
        while position < logical_tokens - 1:
            calibrated_position = min(
                position, self.max_draft_tokens
            )
            mix = self.next_recent_weight[
                calibrated_position, :, None
            ]
            prior = self.position_prior[calibrated_position]
            np.subtract(
                recent,
                prior,
                out=self._forecast_delta_scratch,
            )
            np.multiply(
                self._forecast_delta_scratch,
                mix,
                out=self._forecast_delta_scratch,
            )
            np.copyto(self._forecast_scratch, prior)
            self._forecast_scratch += self._forecast_delta_scratch
            recent = self._forecast_scratch
            aggregate += recent
            position += 1
        next_position = min(
            logical_tokens - 1, self.max_draft_tokens
        )
        mix = self.next_recent_weight[next_position, :, None]
        prior = self.position_prior[next_position]
        np.subtract(
            recent,
            prior,
            out=self._forecast_delta_scratch,
        )
        np.multiply(
            self._forecast_delta_scratch,
            mix,
            out=self._forecast_delta_scratch,
        )
        aggregate += prior
        aggregate += self._forecast_delta_scratch
        padding_rows = bucket - logical_tokens
        if padding_rows > 0:
            np.multiply(
                self.padding_prior,
                float(padding_rows),
                out=self._forecast_delta_scratch,
            )
            aggregate += self._forecast_delta_scratch
        np.maximum(aggregate, 0.0, out=aggregate)
        return aggregate

    def predict_aggregate_pair(
        self,
        *,
        logical_tokens: int,
        bucket: int,
        next_bucket: int,
        current_out: np.ndarray,
        next_out: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict adjacent endpoints while reusing their shared forecast."""
        logical_tokens = int(logical_tokens)
        bucket = int(bucket)
        next_bucket = int(next_bucket)
        if (
            logical_tokens < 1
            or bucket < logical_tokens
            or next_bucket < logical_tokens + 1
        ):
            raise ValueError("invalid adjacent logical/bucket counts")
        # At each active boundary, every aligned current row has just been
        # observed. Keep a general fallback for diagnostic callers.
        if len(self._observed) != logical_tokens - 1:
            return (
                self.predict_aggregate(
                    logical_tokens=logical_tokens,
                    bucket=bucket,
                    out=current_out,
                ),
                self.predict_aggregate(
                    logical_tokens=logical_tokens + 1,
                    bucket=next_bucket,
                    out=next_out,
                ),
            )
        current = np.asarray(current_out, dtype=np.float32)
        next_aggregate = np.asarray(next_out, dtype=np.float32)
        if (
            current.shape != self._observed_sum.shape
            or next_aggregate.shape != self._observed_sum.shape
        ):
            raise ValueError("adjacent aggregate output shape mismatch")
        np.copyto(current, self._observed_sum)
        recent = (
            self._observed[-1]
            if self._observed
            else self.layer_prior
        )
        position = min(
            logical_tokens - 1, self.max_draft_tokens
        )
        prior = self.position_prior[position]
        mix = self.next_recent_weight[position, :, None]
        np.subtract(
            recent, prior, out=self._forecast_delta_scratch
        )
        np.multiply(
            self._forecast_delta_scratch,
            mix,
            out=self._forecast_delta_scratch,
        )
        np.copyto(self._forecast_scratch, prior)
        self._forecast_scratch += self._forecast_delta_scratch
        current += self._forecast_scratch
        np.copyto(next_aggregate, current)

        next_position = min(
            logical_tokens, self.max_draft_tokens
        )
        next_prior = self.position_prior[next_position]
        next_mix = self.next_recent_weight[
            next_position, :, None
        ]
        np.subtract(
            self._forecast_scratch,
            next_prior,
            out=self._forecast_delta_scratch,
        )
        np.multiply(
            self._forecast_delta_scratch,
            next_mix,
            out=self._forecast_delta_scratch,
        )
        next_aggregate += next_prior
        next_aggregate += self._forecast_delta_scratch

        padding_rows = bucket - logical_tokens
        if padding_rows > 0:
            np.multiply(
                self.padding_prior,
                float(padding_rows),
                out=self._forecast_delta_scratch,
            )
            current += self._forecast_delta_scratch
        next_padding_rows = next_bucket - logical_tokens - 1
        if next_padding_rows > 0:
            np.multiply(
                self.padding_prior,
                float(next_padding_rows),
                out=self._forecast_delta_scratch,
            )
            next_aggregate += self._forecast_delta_scratch
        np.maximum(current, 0.0, out=current)
        np.maximum(
            next_aggregate, 0.0, out=next_aggregate
        )
        return current, next_aggregate

    def predict_rows(
        self,
        *,
        logical_tokens: int,
        bucket: int,
    ) -> np.ndarray:
        logical_tokens = int(logical_tokens)
        bucket = int(bucket)
        if logical_tokens < 1 or bucket < logical_tokens:
            raise ValueError("invalid logical/bucket token counts")
        known_count = min(len(self._observed), max(0, logical_tokens - 1))
        rows: list[np.ndarray] = [
            self._observed[idx].copy() for idx in range(known_count)
        ]
        # Any missing aligned rows use their own position prior.  This path is
        # the one-step forecast for endpoint K+1, before that route is observed.
        while len(rows) < logical_tokens - 1:
            position = min(len(rows), self.max_draft_tokens)
            recent = rows[-1] if rows else self.layer_prior
            mix = self.next_recent_weight[position, :, None]
            forecast = (
                mix * recent
                + (1.0 - mix) * self.position_prior[position]
            )
            rows.append(self._normalize_rows(forecast, float(self.top_k)))
        next_position = min(logical_tokens - 1, self.max_draft_tokens)
        recent = rows[-1] if rows else self.layer_prior
        mix = self.next_recent_weight[next_position, :, None]
        next_row = mix * recent + (1.0 - mix) * self.position_prior[next_position]
        rows.append(self._normalize_rows(next_row, float(self.top_k)))
        while len(rows) < bucket:
            rows.append(self.padding_prior.copy())
        return np.stack(rows, axis=0).astype(np.float32, copy=False)


class TransferCacheShadowSimulator:
    def __init__(
        self,
        *,
        num_layers: int,
        num_experts: int,
        segment_size: int,
        transfer_ms: float,
        budget_transfer_ms: float | None = None,
        max_inflight: int,
        draft_budget: int,
        verify_attention_ratio: float,
        segment_compute_ms: Sequence[float],
        draft_visible_budget_ms: float | None = None,
        verify_visible_budget_ms: float | None = None,
    ) -> None:
        self.num_layers = int(num_layers)
        self.num_experts = int(num_experts)
        self.segment_size = max(1, int(segment_size))
        self.transfer_ms = max(1e-6, float(transfer_ms))
        self.budget_transfer_ms = max(
            1e-6,
            float(
                self.transfer_ms
                if budget_transfer_ms is None
                else budget_transfer_ms
            ),
        )
        self.max_inflight = max(0, int(max_inflight))
        self.draft_budget = max(0, int(draft_budget))
        self.verify_attention_ratio = max(0.0, float(verify_attention_ratio))
        self.draft_visible_budget_ms = (
            None
            if draft_visible_budget_ms is None
            else max(0.0, float(draft_visible_budget_ms))
        )
        self.verify_visible_budget_ms = (
            None
            if verify_visible_budget_ms is None
            else max(0.0, float(verify_visible_budget_ms))
        )
        num_segments = math.ceil(self.num_layers / self.segment_size)
        raw_segment_ms = [max(0.0, float(value)) for value in segment_compute_ms]
        if not raw_segment_ms:
            raw_segment_ms = [0.0] * num_segments
        if len(raw_segment_ms) < num_segments:
            raw_segment_ms.extend([raw_segment_ms[-1]] * (num_segments - len(raw_segment_ms)))
        self.segment_compute_ms = tuple(raw_segment_ms[:num_segments])
        self._boundaries = tuple(
            (
                first,
                min(self.num_layers, first + self.segment_size),
            )
            for first in range(
                0, self.num_layers, self.segment_size
            )
        )
        self._batch_target_ranges = self._boundaries[2:]
        self._verify_submit_budget_caps = tuple(
            int(
                (
                    (
                        segment_ms * self.verify_attention_ratio
                        if self.verify_visible_budget_ms is None
                        else self.verify_visible_budget_ms
                    )
                    // self.budget_transfer_ms
                )
            )
            for segment_ms in self.segment_compute_ms
        )
        self._can_batch_completed_segments = bool(
            len(self._boundaries) > 1
            and all(
                float(value) + 1e-9 >= self.transfer_ms
                for value in self.segment_compute_ms
            )
            and len(
                {
                    last - first
                    for first, last in self._boundaries
                }
            )
            == 1
        )
        self._pair_current_scratch: _ArrayShadowState | None = None
        self._pair_next_scratch: _ArrayShadowState | None = None
        workload_shape = (self.num_layers, self.num_experts)
        self._current_cpu_scratch = np.zeros(
            workload_shape, dtype=np.float32
        )
        self._next_cpu_scratch = np.zeros_like(
            self._current_cpu_scratch
        )
        self._current_soft_cpu_scratch = np.zeros_like(
            self._current_cpu_scratch
        )
        self._next_soft_cpu_scratch = np.zeros_like(
            self._current_cpu_scratch
        )
        self._victim_score_scratch = np.empty(
            self.num_experts, dtype=np.float32
        )
        self._victim_prepare_scores = np.empty(
            workload_shape, dtype=np.float32
        )

    @staticmethod
    def _copy_array_state(
        source: _ArrayShadowState,
        target: _ArrayShadowState | None,
    ) -> _ArrayShadowState:
        compatible = (
            isinstance(target, _ArrayShadowState)
            and target.slots.shape == source.slots.shape
            and target.resident.shape == source.resident.shape
            and bool(target.expert_only) == bool(source.expert_only)
        )
        if not compatible:
            return source.clone()
        np.copyto(target.slots, source.slots)
        np.copyto(target.pending_slots, source.pending_slots)
        target.valid_slots = source.valid_slots
        np.copyto(target.access_values, source.access_values)
        np.copyto(target.protected, source.protected)
        np.copyto(target.resident, source.resident)
        np.copyto(target.pending_experts, source.pending_experts)
        if target.empty_slots is not None and source.empty_slots is not None:
            np.copyto(target.empty_slots, source.empty_slots)
        if (
            target.reserved_victims is not None
            and source.reserved_victims is not None
        ):
            np.copyto(
                target.reserved_victims, source.reserved_victims
            )
        if (
            target.victim_hints is not None
            and source.victim_hints is not None
        ):
            np.copyto(target.victim_hints, source.victim_hints)
        elif source.victim_hints is None:
            target.victim_hints = None
        else:
            target.victim_hints = source.victim_hints.copy()
        target.inflight = list(source.inflight)
        return target

    def _to_array_state(
        self, state: ShadowCacheState
    ) -> _ArrayShadowState:
        cached = state.array_state
        if isinstance(cached, _ArrayShadowState):
            return cached
        max_slots = max(
            (len(layer.slots) for layer in state.layers), default=0
        )
        slots = np.full(
            (self.num_layers, max_slots), -2, dtype=np.int16
        )
        pending_slots = np.full_like(slots, -2)
        valid_slots = np.zeros_like(slots, dtype=bool)
        access = np.zeros(
            (self.num_layers, self.num_experts), dtype=np.float32
        )
        protected = np.zeros_like(access, dtype=bool)
        resident = np.zeros_like(access, dtype=bool)
        pending_experts = np.zeros_like(access, dtype=bool)
        for layer_idx, layer in enumerate(state.layers):
            count = len(layer.slots)
            if count:
                slots[layer_idx, :count] = layer.slots
                pending_slots[layer_idx, :count] = layer.pending
                valid_slots[layer_idx, :count] = True
                slot_values = np.asarray(layer.slots, dtype=np.int64)
                valid = (slot_values >= 0) & (
                    slot_values < self.num_experts
                )
                resident[layer_idx, slot_values[valid]] = True
                pending_values = np.asarray(
                    layer.pending, dtype=np.int64
                )
                valid_pending = (pending_values >= 0) & (
                    pending_values < self.num_experts
                )
                pending_experts[
                    layer_idx, pending_values[valid_pending]
                ] = True
            row_access = np.asarray(
                layer.access_values, dtype=np.float32
            )
            width = min(self.num_experts, row_access.size)
            access[layer_idx, :width] = row_access[:width]
            if layer.protected_experts:
                protected[
                    layer_idx,
                    np.fromiter(
                        (
                            int(value)
                            for value in layer.protected_experts
                            if 0 <= int(value) < self.num_experts
                        ),
                        dtype=np.int64,
                    ),
                ] = True
        return _ArrayShadowState(
            slots=slots,
            pending_slots=pending_slots,
            valid_slots=valid_slots,
            access_values=access,
            protected=protected,
            resident=resident,
            pending_experts=pending_experts,
            inflight=list(state.inflight),
            expert_only=False,
        )

    def _prepare_victim_hints(
        self, state: _ArrayShadowState
    ) -> None:
        if not state.expert_only:
            state.victim_hints = None
            return
        scores = self._victim_prepare_scores
        scores.fill(np.inf)
        eligible = state.resident & ~state.protected
        if state.reserved_victims is not None:
            eligible &= ~state.reserved_victims
        np.copyto(scores, state.access_values, where=eligible)
        hints = np.argmin(scores, axis=1).astype(
            np.int16, copy=False
        )
        rows = np.arange(self.num_layers)
        hints[~np.isfinite(scores[rows, hints])] = -1
        if (
            state.victim_hints is None
            or state.victim_hints.shape != hints.shape
        ):
            state.victim_hints = hints.copy()
        else:
            np.copyto(state.victim_hints, hints)

    @staticmethod
    def _array_advance(
        state: _ArrayShadowState, delta_ms: float
    ) -> None:
        delta_ms = max(0.0, float(delta_ms))
        remaining: list[ShadowTransfer] = []
        for ticket in state.inflight:
            left = float(ticket.remaining_ms) - delta_ms
            if left > 1e-9:
                remaining.append(replace(ticket, remaining_ms=left))
                continue
            layer_idx = int(ticket.layer_idx)
            slot_idx = int(ticket.slot_idx)
            expert_idx = int(ticket.expert_idx)
            if state.expert_only:
                previous = int(ticket.previous_expert)
                if 0 <= previous < state.resident.shape[1]:
                    state.resident[layer_idx, previous] = False
                    if state.reserved_victims is not None:
                        state.reserved_victims[
                            layer_idx, previous
                        ] = False
                if 0 <= expert_idx < state.resident.shape[1]:
                    state.resident[layer_idx, expert_idx] = True
                    state.pending_experts[layer_idx, expert_idx] = False
                continue
            if (
                0 <= layer_idx < state.slots.shape[0]
                and 0 <= slot_idx < state.slots.shape[1]
                and state.valid_slots[layer_idx, slot_idx]
                and int(state.pending_slots[layer_idx, slot_idx])
                == expert_idx
            ):
                previous = int(state.slots[layer_idx, slot_idx])
                if 0 <= previous < state.resident.shape[1]:
                    state.resident[layer_idx, previous] = False
                state.slots[layer_idx, slot_idx] = expert_idx
                state.pending_slots[layer_idx, slot_idx] = -1
                if 0 <= expert_idx < state.resident.shape[1]:
                    state.resident[layer_idx, expert_idx] = True
                    state.pending_experts[layer_idx, expert_idx] = False
        state.inflight = remaining

    def _array_victim(
        self, state: _ArrayShadowState, layer_idx: int
    ) -> tuple[int, int] | None:
        if state.expert_only:
            if (
                state.empty_slots is not None
                and state.empty_slots[layer_idx] > 0
            ):
                state.empty_slots[layer_idx] -= 1
                return -1, -1
            if state.victim_hints is not None:
                previous = int(state.victim_hints[layer_idx])
                if (
                    0 <= previous < state.resident.shape[1]
                    and state.resident[layer_idx, previous]
                    and not state.protected[layer_idx, previous]
                    and (
                        state.reserved_victims is None
                        or not state.reserved_victims[
                            layer_idx, previous
                        ]
                    )
                ):
                    state.victim_hints[layer_idx] = -1
                    if state.reserved_victims is not None:
                        state.reserved_victims[
                            layer_idx, previous
                        ] = True
                    return -1, previous
            eligible = (
                state.resident[layer_idx]
                & ~state.protected[layer_idx]
            )
            if state.reserved_victims is not None:
                eligible &= ~state.reserved_victims[layer_idx]
            scores = self._victim_score_scratch
            scores.fill(np.inf)
            np.copyto(
                scores,
                state.access_values[layer_idx],
                where=eligible,
            )
            previous = int(np.argmin(scores))
            if not math.isfinite(float(scores[previous])):
                return None
            if state.reserved_victims is not None:
                state.reserved_victims[layer_idx, previous] = True
            return -1, previous
        usable = (
            state.valid_slots[layer_idx]
            & (state.pending_slots[layer_idx] < 0)
        )
        empty = np.flatnonzero(
            usable & (state.slots[layer_idx] < 0)
        )
        if empty.size:
            return int(empty[0]), -1
        candidates = np.flatnonzero(usable)
        if not candidates.size:
            return None
        experts = state.slots[layer_idx, candidates].astype(
            np.int64, copy=False
        )
        not_protected = ~state.protected[layer_idx, experts]
        candidates = candidates[not_protected]
        experts = experts[not_protected]
        if not candidates.size:
            return None
        values = state.access_values[layer_idx, experts]
        slot_idx = int(candidates[int(np.argmin(values))])
        return slot_idx, int(state.slots[layer_idx, slot_idx])

    def _array_submit(
        self,
        state: _ArrayShadowState,
        *,
        demand: np.ndarray,
        layer_start: int,
        layer_end: int,
        budget: int,
        source: str,
        completion_window_ms: float | None = None,
    ) -> int:
        available = max(0, self.max_inflight - len(state.inflight))
        budget = min(max(0, int(budget)), available)
        if budget <= 0 or layer_start >= layer_end:
            return 0
        completes_in_window = (
            completion_window_ms is not None
            and float(completion_window_ms) + 1e-9
            >= self.transfer_ms
        )
        completed: list[tuple[int, int, int, int]] = []

        def record_transfer(
            layer_idx: int,
            expert_idx: int,
            victim_slot: int,
            previous: int,
        ) -> None:
            if completes_in_window:
                completed.append(
                    (
                        int(layer_idx),
                        int(expert_idx),
                        int(victim_slot),
                        int(previous),
                    )
                )
                return
            state.inflight.append(
                ShadowTransfer(
                    layer_idx=layer_idx,
                    expert_idx=expert_idx,
                    slot_idx=victim_slot,
                    previous_expert=previous,
                    remaining_ms=self.transfer_ms,
                    source=str(source),
                    deferred_publish=True,
                )
            )

        def finish(submitted: int) -> int:
            for layer_idx, expert_idx, slot_idx, previous in completed:
                if state.expert_only:
                    if 0 <= previous < self.num_experts:
                        state.resident[layer_idx, previous] = False
                        if state.reserved_victims is not None:
                            state.reserved_victims[
                                layer_idx, previous
                            ] = False
                    state.resident[layer_idx, expert_idx] = True
                    state.pending_experts[layer_idx, expert_idx] = False
                elif (
                    0 <= slot_idx < state.slots.shape[1]
                    and state.valid_slots[layer_idx, slot_idx]
                    and int(
                        state.pending_slots[layer_idx, slot_idx]
                    )
                    == expert_idx
                ):
                    if 0 <= previous < self.num_experts:
                        state.resident[layer_idx, previous] = False
                    state.slots[layer_idx, slot_idx] = expert_idx
                    state.pending_slots[layer_idx, slot_idx] = -1
                    state.resident[layer_idx, expert_idx] = True
                    state.pending_experts[layer_idx, expert_idx] = False
            return submitted
        selected = np.asarray(demand, dtype=np.float32)[
            int(layer_start) : int(layer_end)
        ]
        eligible = (
            (selected > 0.0)
            & ~state.resident[int(layer_start) : int(layer_end)]
            & ~state.pending_experts[int(layer_start) : int(layer_end)]
        )

        if budget == 1:
            ranked_scores = np.where(
                eligible, selected, -np.inf
            )
            submitted = 0
            while submitted < budget:
                flat_idx = int(np.argmax(ranked_scores))
                score = float(ranked_scores.reshape(-1)[flat_idx])
                if not math.isfinite(score) or score <= 0.0:
                    break
                local_layer, expert_idx = divmod(
                    flat_idx, self.num_experts
                )
                ranked_scores[local_layer, expert_idx] = -np.inf
                layer_idx = int(layer_start) + int(local_layer)
                expert_idx = int(expert_idx)
                if (
                    state.resident[layer_idx, expert_idx]
                    or state.pending_experts[layer_idx, expert_idx]
                ):
                    continue
                victim = self._array_victim(state, layer_idx)
                if victim is None:
                    continue
                victim_slot, previous = victim
                if not state.expert_only:
                    state.pending_slots[
                        layer_idx, victim_slot
                    ] = expert_idx
                state.pending_experts[layer_idx, expert_idx] = True
                state.protected[layer_idx, expert_idx] = True
                record_transfer(
                    layer_idx,
                    expert_idx,
                    victim_slot,
                    previous,
                )
                submitted += 1
            return finish(submitted)

        ranked_scores = np.where(eligible, selected, -np.inf)
        candidate_limit = min(
            ranked_scores.size, max(8, budget * 2)
        )
        submitted = 0
        examined = 0
        while submitted < budget and examined < candidate_limit:
            flat_idx = int(np.argmax(ranked_scores))
            score = float(ranked_scores.reshape(-1)[flat_idx])
            if not math.isfinite(score) or score <= 0.0:
                break
            local_layer, expert_idx = divmod(
                flat_idx, self.num_experts
            )
            ranked_scores[local_layer, expert_idx] = -np.inf
            examined += 1
            layer_idx = int(layer_start) + int(local_layer)
            expert_idx = int(expert_idx)
            if (
                state.resident[layer_idx, expert_idx]
                or state.pending_experts[layer_idx, expert_idx]
            ):
                continue
            victim = self._array_victim(state, layer_idx)
            if victim is None:
                continue
            victim_slot, previous = victim
            if not state.expert_only:
                state.pending_slots[
                    layer_idx, victim_slot
                ] = expert_idx
            state.pending_experts[layer_idx, expert_idx] = True
            state.protected[layer_idx, expert_idx] = True
            record_transfer(
                layer_idx,
                expert_idx,
                victim_slot,
                previous,
            )
            submitted += 1
        return finish(submitted)

    def _array_submit_completed_batch(
        self,
        state: _ArrayShadowState,
        *,
        demand: np.ndarray,
        target_ranges: Sequence[tuple[int, int]],
        budget: int,
    ) -> list[int]:
        """Batch independent segment selections whose transfers all complete."""
        if not state.expert_only or budget <= 0 or not target_ranges:
            return [0] * len(target_ranges)
        widths = {
            (int(last) - int(first)) * self.num_experts
            for first, last in target_ranges
        }
        if len(widths) != 1:
            return [
                self._array_submit(
                    state,
                    demand=demand,
                    layer_start=first,
                    layer_end=last,
                    budget=budget,
                    source="verify_batch_fallback",
                    completion_window_ms=self.transfer_ms,
                )
                for first, last in target_ranges
            ]
        selected = np.stack(
            [
                np.asarray(demand[first:last], dtype=np.float32)
                .reshape(-1)
                for first, last in target_ranges
            ]
        )
        eligible = np.stack(
            [
                (
                    ~state.resident[first:last]
                    & ~state.pending_experts[first:last]
                ).reshape(-1)
                for first, last in target_ranges
            ]
        )
        scores = np.where(eligible, selected, -np.inf)
        candidate_limit = min(
            scores.shape[1], max(8, int(budget) * 2)
        )
        submitted_by_range: list[int] = []
        for range_idx, (first, last) in enumerate(target_ranges):
            completed: list[tuple[int, int, int]] = []
            submitted = 0
            examined = 0
            ranked_scores = scores[range_idx]
            while (
                submitted < int(budget)
                and examined < candidate_limit
            ):
                flat_idx = int(np.argmax(ranked_scores))
                score = float(ranked_scores[flat_idx])
                if not math.isfinite(score) or score <= 0.0:
                    break
                ranked_scores[flat_idx] = -np.inf
                examined += 1
                local_layer, expert_idx = divmod(
                    flat_idx, self.num_experts
                )
                layer_idx = int(first) + int(local_layer)
                expert_idx = int(expert_idx)
                if (
                    state.resident[layer_idx, expert_idx]
                    or state.pending_experts[layer_idx, expert_idx]
                ):
                    continue
                victim = self._array_victim(state, layer_idx)
                if victim is None:
                    continue
                _, previous = victim
                state.pending_experts[layer_idx, expert_idx] = True
                state.protected[layer_idx, expert_idx] = True
                completed.append((layer_idx, expert_idx, previous))
                submitted += 1
            for layer_idx, expert_idx, previous in completed:
                if 0 <= previous < self.num_experts:
                    state.resident[layer_idx, previous] = False
                    if state.reserved_victims is not None:
                        state.reserved_victims[
                            layer_idx, previous
                        ] = False
                state.resident[layer_idx, expert_idx] = True
                state.pending_experts[layer_idx, expert_idx] = False
            submitted_by_range.append(submitted)
        return submitted_by_range

    def _array_next_draft(
        self,
        state: _ArrayShadowState,
        *,
        demand: np.ndarray,
        draft_ms: float,
    ) -> None:
        draft_ms = max(0.0, float(draft_ms))
        drain_ms = min(
            draft_ms,
            max(
                (
                    float(ticket.remaining_ms)
                    for ticket in state.inflight
                ),
                default=0.0,
            ),
        )
        self._array_advance(state, drain_ms)
        visible_ms = max(0.0, draft_ms - drain_ms)
        scheduling_window_ms = (
            visible_ms
            if self.draft_visible_budget_ms is None
            else min(visible_ms, self.draft_visible_budget_ms)
        )
        budget = min(
            self.draft_budget,
            int(scheduling_window_ms // self.budget_transfer_ms),
        )
        self._array_submit(
            state,
            demand=demand,
            layer_start=0,
            layer_end=self.num_layers,
            budget=budget,
            source="draft_shadow",
            completion_window_ms=visible_ms,
        )
        self._array_advance(state, visible_ms)

    def _array_verify(
        self,
        state: _ArrayShadowState,
        *,
        demand: np.ndarray,
        vpb: int,
        cpu_out: np.ndarray | None = None,
        soft_cpu_out: np.ndarray | None = None,
    ) -> ShadowWorkload:
        self._array_advance(state, 0.0)
        cpu = (
            np.zeros_like(demand, dtype=np.float32)
            if cpu_out is None
            else np.asarray(cpu_out, dtype=np.float32)
        )
        soft_cpu = (
            np.zeros_like(demand, dtype=np.float32)
            if soft_cpu_out is None
            else np.asarray(soft_cpu_out, dtype=np.float32)
        )
        boundaries = self._boundaries
        segments: list[SegmentWorkload] = []
        total_submitted = 0
        total_cpu_experts = 0.0
        batch_submit_counts: list[int] | None = None
        can_batch_remaining_submits = bool(
            state.expert_only
            and int(vpb) > 0
            and self._can_batch_completed_segments
        )
        for segment_id, (first, last) in enumerate(boundaries):
            # Residency can change between segment boundaries.
            cpu_rows = cpu[first:last]
            np.copyto(cpu_rows, demand[first:last])
            cpu_rows[state.resident[first:last]] = 0.0
            soft_rows = soft_cpu[first:last]
            np.minimum(
                demand[first:last],
                1.0,
                out=soft_rows,
            )
            soft_rows[state.resident[first:last]] = 0.0
            rows = cpu_rows
            layer_experts = soft_rows.sum(axis=1)
            total_cpu_experts += float(layer_experts.sum())
            target_segment = (segment_id + 1) % len(boundaries)
            target_first, target_last = boundaries[target_segment]
            submit_budget = min(
                max(0, int(vpb)),
                self._verify_submit_budget_caps[segment_id],
            )
            if (
                can_batch_remaining_submits
                and segment_id == 1
                and batch_submit_counts is None
            ):
                target_ranges = self._batch_target_ranges
                batch_submit_counts = (
                    self._array_submit_completed_batch(
                        state,
                        demand=demand,
                        target_ranges=target_ranges,
                        budget=submit_budget,
                    )
                )
                # The last boundary targets segment zero after segment zero has
                # already computed. Its cache mutation cannot affect this
                # endpoint's latency, so count feasible submits without ranking
                # or materializing a discarded terminal state.
                terminal_first, terminal_last = boundaries[0]
                terminal_eligible = (
                    (demand[terminal_first:terminal_last] > 0.0)
                    & ~state.resident[terminal_first:terminal_last]
                    & ~state.pending_experts[
                        terminal_first:terminal_last
                    ]
                )
                terminal_resident = state.resident[
                    terminal_first:terminal_last
                ]
                terminal_victims = (
                    terminal_resident
                    & ~state.protected[terminal_first:terminal_last]
                )
                if state.reserved_victims is not None:
                    terminal_victims &= ~state.reserved_victims[
                        terminal_first:terminal_last
                    ]
                terminal_capacity = terminal_victims.sum(
                    axis=1, dtype=np.int16
                )
                if state.empty_slots is not None:
                    terminal_capacity += (
                        state.empty_slots[
                            terminal_first:terminal_last
                        ]
                    )
                terminal_candidates = terminal_eligible.sum(
                    axis=1, dtype=np.int16
                )
                terminal_feasible = int(
                    np.minimum(
                        terminal_candidates, terminal_capacity
                    ).sum()
                )
                batch_submit_counts.append(
                    min(submit_budget, terminal_feasible)
                )
            if (
                batch_submit_counts is not None
                and segment_id >= 1
            ):
                submitted = batch_submit_counts[segment_id - 1]
            else:
                submitted = self._array_submit(
                    state,
                    demand=demand,
                    layer_start=target_first,
                    layer_end=target_last,
                    budget=submit_budget,
                    source=f"verify_segment_{segment_id}",
                    completion_window_ms=self.segment_compute_ms[
                        segment_id
                    ],
                )
            total_submitted += submitted
            segments.append(
                SegmentWorkload(
                    segment_id=segment_id,
                    first_layer=first,
                    last_layer=last - 1,
                    cpu_experts=float(layer_experts.sum()),
                    cpu_routes=float(rows.sum()),
                    max_layer_experts=float(
                        layer_experts.max(initial=0)
                    ),
                    transfer_submits=submitted,
                )
            )
            self._array_advance(
                state, self.segment_compute_ms[segment_id]
            )
            state.protected[first:last] = False
        return ShadowWorkload(
            cpu_route_counts=cpu,
            cpu_experts=total_cpu_experts,
            cpu_routes=float(cpu.sum()),
            segments=tuple(segments),
            transfer_submits=int(total_submitted),
            transfer_pending=len(state.inflight),
        )

    def simulate_pair_fast(
        self,
        state: ShadowCacheState,
        *,
        current_demand: np.ndarray,
        next_demand: np.ndarray,
        vpb: int,
        next_draft_ms: float,
    ) -> tuple[ShadowWorkload, ShadowWorkload]:
        """Simulate both endpoints after a single host-state conversion."""
        base = self._to_array_state(state)
        current_state = self._copy_array_state(
            base, self._pair_current_scratch
        )
        self._pair_current_scratch = current_state
        self._prepare_victim_hints(current_state)
        current = self._array_verify(
            current_state,
            demand=np.asarray(current_demand, dtype=np.float32),
            vpb=int(vpb),
            cpu_out=self._current_cpu_scratch,
            soft_cpu_out=self._current_soft_cpu_scratch,
        )
        next_state = self._copy_array_state(
            base, self._pair_next_scratch
        )
        self._pair_next_scratch = next_state
        self._prepare_victim_hints(next_state)
        self._array_next_draft(
            next_state,
            demand=np.asarray(next_demand, dtype=np.float32),
            draft_ms=float(next_draft_ms),
        )
        next_workload = self._array_verify(
            next_state,
            demand=np.asarray(next_demand, dtype=np.float32),
            vpb=int(vpb),
            cpu_out=self._next_cpu_scratch,
            soft_cpu_out=self._next_soft_cpu_scratch,
        )
        return current, next_workload

    def _advance(self, state: ShadowCacheState, delta_ms: float) -> ShadowCacheState:
        delta_ms = max(0.0, float(delta_ms))
        layers = list(state.layers)
        remaining: list[ShadowTransfer] = []
        completed = sorted(
            state.inflight,
            key=lambda ticket: (
                float(ticket.remaining_ms),
                int(ticket.layer_idx),
                int(ticket.expert_idx),
            ),
        )
        for ticket in completed:
            left = float(ticket.remaining_ms) - delta_ms
            if left > 1e-9:
                remaining.append(replace(ticket, remaining_ms=left))
                continue
            layer = layers[ticket.layer_idx]
            slots = list(layer.slots)
            pending = list(layer.pending)
            slot_idx = int(ticket.slot_idx)
            if 0 <= slot_idx < len(slots) and pending[slot_idx] == ticket.expert_idx:
                slots[slot_idx] = int(ticket.expert_idx)
                pending[slot_idx] = -1
                layers[ticket.layer_idx] = replace(
                    layer,
                    slots=tuple(slots),
                    pending=tuple(pending),
                )
        return ShadowCacheState(
            layers=tuple(layers),
            inflight=tuple(remaining),
            elapsed_ms=float(state.elapsed_ms) + delta_ms,
        )

    def _submit(
        self,
        state: ShadowCacheState,
        *,
        demand: np.ndarray,
        layer_indices: Sequence[int],
        budget: int,
        source: str,
    ) -> tuple[ShadowCacheState, int]:
        available = max(0, self.max_inflight - len(state.inflight))
        budget = min(max(0, int(budget)), available)
        if budget <= 0:
            return state, 0
        layers = list(state.layers)
        inflight = list(state.inflight)
        submitted = 0
        candidate_limit = max(64, budget * 8)
        ranked_scores, ranked_layers, ranked_experts = _rank_demand_candidate_arrays(
            demand,
            layer_indices=layer_indices,
            state=state,
            candidate_limit=candidate_limit,
        )
        seen: set[tuple[int, int]] = set()

        def submit_ranked(
            scores: np.ndarray,
            ranked_layer_indices: np.ndarray,
            ranked_expert_indices: np.ndarray,
        ) -> None:
            nonlocal submitted
            for _score, layer_idx, expert_idx in zip(
                scores,
                ranked_layer_indices,
                ranked_expert_indices,
                strict=True,
            ):
                if submitted >= budget:
                    break
                layer_idx = int(layer_idx)
                expert_idx = int(expert_idx)
                key = (layer_idx, expert_idx)
                if key in seen:
                    continue
                seen.add(key)
                layer = layers[layer_idx]
                # Earlier selections may have changed this copied layer.
                resident = {value for value in layer.slots if value >= 0}
                pending_experts = {
                    value for value in layer.pending if value >= 0
                }
                if expert_idx in resident or expert_idx in pending_experts:
                    continue
                victim = select_shadow_victim(layer)
                if victim is None:
                    continue
                slots = list(layer.slots)
                pending = list(layer.pending)
                previous = int(slots[victim])
                pending[victim] = int(expert_idx)
                protected = set(layer.protected_experts)
                protected.add(int(expert_idx))
                layers[layer_idx] = replace(
                    layer,
                    slots=tuple(slots),
                    pending=tuple(pending),
                    protected_experts=frozenset(protected),
                )
                inflight.append(
                    ShadowTransfer(
                        layer_idx=int(layer_idx),
                        expert_idx=int(expert_idx),
                        slot_idx=int(victim),
                        previous_expert=previous,
                        remaining_ms=self.transfer_ms,
                        source=str(source),
                        deferred_publish=True,
                    )
                )
                submitted += 1

        submit_ranked(ranked_scores, ranked_layers, ranked_experts)
        if (
            submitted < budget
            and ranked_scores.size >= candidate_limit
        ):
            # Rare safety fallback for a top slice dominated by layers whose
            # slots are all pending/protected.
            full_scores, full_layers, full_experts = (
                _rank_demand_candidate_arrays(
                    demand,
                    layer_indices=layer_indices,
                    state=state,
                )
            )
            submit_ranked(full_scores, full_layers, full_experts)
        return (
            ShadowCacheState(
                layers=tuple(layers),
                inflight=tuple(inflight),
                elapsed_ms=state.elapsed_ms,
            ),
            submitted,
        )

    def simulate_next_draft(
        self,
        state: ShadowCacheState,
        *,
        demand: np.ndarray,
        draft_ms: float,
    ) -> ShadowCacheState:
        # The live draft path drains all direct-active tickets at its opening
        # boundary. Account for that wait inside (not in addition to) the
        # observed/EMA draft-call budget, then simulate this draft's submits.
        draft_ms = max(0.0, float(draft_ms))
        drain_ms = max(
            (float(ticket.remaining_ms) for ticket in state.inflight),
            default=0.0,
        )
        drain_ms = min(draft_ms, max(0.0, drain_ms))
        state = self._advance(state, drain_ms)
        visible_ms = max(0.0, draft_ms - drain_ms)
        scheduling_window_ms = (
            visible_ms
            if self.draft_visible_budget_ms is None
            else min(visible_ms, self.draft_visible_budget_ms)
        )
        budget_by_time = int(
            scheduling_window_ms // self.budget_transfer_ms
        )
        budget = min(self.draft_budget, budget_by_time)
        state, _ = self._submit(
            state,
            demand=demand,
            layer_indices=range(self.num_layers),
            budget=budget,
            source="draft_shadow",
        )
        return self._advance(state, visible_ms)

    def simulate_verify(
        self,
        state: ShadowCacheState,
        *,
        demand_rows: np.ndarray,
        vpb: int,
    ) -> ShadowSimulation:
        state = self._advance(state, 0.0)
        demand = np.asarray(demand_rows, dtype=np.float32).sum(axis=0)
        cpu = np.zeros_like(demand, dtype=np.float32)
        segments: list[SegmentWorkload] = []
        total_submitted = 0
        boundaries = [
            (first, min(self.num_layers, first + self.segment_size))
            for first in range(0, self.num_layers, self.segment_size)
        ]
        for segment_id, (first, last) in enumerate(boundaries):
            layer_indices = range(first, last)
            # The live segment graph publishes ready tickets at segment start.
            # Consequently new transfers submitted after this segment's replay
            # can help the *next* segment, never the segment that selected them.
            state = self._advance(state, 0.0)
            segment_layer_experts: list[float] = []
            segment_routes = 0.0
            segment_experts = 0.0
            for layer_idx in layer_indices:
                resident = {
                    int(value)
                    for value in state.layers[layer_idx].slots
                    if int(value) >= 0
                }
                resident_mask = np.zeros(self.num_experts, dtype=bool)
                if resident:
                    resident_mask[list(resident)] = True
                cpu[layer_idx] = np.where(
                    resident_mask, 0.0, demand[layer_idx]
                )
                layer_experts = float(
                    expected_distinct_experts(cpu[layer_idx])
                )
                layer_routes = float(cpu[layer_idx].sum())
                segment_layer_experts.append(layer_experts)
                segment_experts += layer_experts
                segment_routes += layer_routes

            # After replay is enqueued, the live helper targets the following
            # segment (wrapping the final boundary to segment zero).
            target_segment = (segment_id + 1) % len(boundaries)
            target_first, target_last = boundaries[target_segment]
            visible_ms = (
                self.segment_compute_ms[segment_id] * self.verify_attention_ratio
            )
            scheduling_window_ms = (
                visible_ms
                if self.verify_visible_budget_ms is None
                else self.verify_visible_budget_ms
            )
            time_budget = int(
                scheduling_window_ms // self.budget_transfer_ms
            )
            state, submitted = self._submit(
                state,
                demand=demand,
                layer_indices=range(target_first, target_last),
                budget=min(max(0, int(vpb)), time_budget),
                source=f"verify_segment_{segment_id}",
            )
            total_submitted += submitted
            segments.append(
                SegmentWorkload(
                    segment_id=segment_id,
                    first_layer=first,
                    last_layer=last - 1,
                    cpu_experts=segment_experts,
                    cpu_routes=segment_routes,
                    max_layer_experts=max(segment_layer_experts, default=0.0),
                    transfer_submits=submitted,
                )
            )
            remaining_compute = max(
                0.0, self.segment_compute_ms[segment_id] - visible_ms
            )
            state = self._advance(state, visible_ms + remaining_compute)
            # Round protection is released once this segment has computed.
            layers = list(state.layers)
            for layer_idx in layer_indices:
                layers[layer_idx] = replace(
                    layers[layer_idx], protected_experts=frozenset()
                )
            state = replace(state, layers=tuple(layers))
        return ShadowSimulation(
            state=state,
            cpu_route_counts=cpu,
            cpu_experts=float(expected_distinct_experts(cpu)),
            cpu_routes=float(cpu.sum()),
            segments=tuple(segments),
            transfer_submits=int(total_submitted),
            transfer_pending=len(state.inflight),
        )


class TransferAwareVerifyCostModel:
    def __init__(self, artifact: Mapping[str, object]) -> None:
        self.artifact = dict(artifact)
        if int(self.artifact.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(
                "transfer-aware active policy requires artifact schema_version=3"
            )
        if (
            str(self.artifact.get("simulator_semantics_version", ""))
            != SIMULATOR_SEMANTICS_VERSION
        ):
            raise ValueError(
                "transfer-aware artifact simulator semantics do not match "
                f"runtime={SIMULATOR_SEMANTICS_VERSION}"
            )
        self.num_layers = int(self.artifact["num_layers"])
        self.num_experts = int(self.artifact["num_experts"])
        self.top_k = int(self.artifact["top_k"])
        self.max_draft_tokens = int(self.artifact.get("max_draft_tokens", 12))
        self.buckets = tuple(int(value) for value in self.artifact["buckets"])
        if tuple(sorted(set(self.buckets))) != self.buckets:
            raise ValueError("v3 verify buckets must be sorted and unique")
        demand_artifact = self.artifact.get("demand_model", {})
        if not isinstance(demand_artifact, Mapping):
            raise ValueError("v3 demand_model must be an object")
        self.demand = CalibratedVerifyDemandPredictor(
            num_layers=self.num_layers,
            num_experts=self.num_experts,
            top_k=self.top_k,
            max_draft_tokens=self.max_draft_tokens,
            artifact=demand_artifact,
        )
        self._current_demand_scratch = np.zeros(
            (self.num_layers, self.num_experts), dtype=np.float32
        )
        self._next_demand_scratch = np.zeros_like(
            self._current_demand_scratch
        )
        transfer = self.artifact.get("transfer_model", {})
        if not isinstance(transfer, Mapping):
            raise ValueError("v3 transfer_model must be an object")
        self.simulator = TransferCacheShadowSimulator(
            num_layers=self.num_layers,
            num_experts=self.num_experts,
            segment_size=int(transfer.get("segment_size", 12)),
            transfer_ms=float(transfer.get("expert_transfer_ms", 1.0)),
            budget_transfer_ms=float(
                transfer.get(
                    "budget_expert_transfer_ms",
                    transfer.get("expert_transfer_ms", 1.0),
                )
            ),
            max_inflight=int(transfer.get("max_inflight", 16)),
            draft_budget=int(transfer.get("draft_budget", 4)),
            verify_attention_ratio=float(
                transfer.get("verify_attention_ratio", 0.3)
            ),
            segment_compute_ms=transfer.get("segment_compute_ms", ()),
            draft_visible_budget_ms=(
                float(transfer["draft_visible_budget_ms"])
                if "draft_visible_budget_ms" in transfer
                else None
            ),
            verify_visible_budget_ms=(
                float(transfer["verify_visible_budget_ms"])
                if "verify_visible_budget_ms" in transfer
                else None
            ),
        )
        latency = self.artifact.get("latency_model", {})
        if not isinstance(latency, Mapping):
            raise ValueError("v3 latency_model must be an object")
        self.latency_model = dict(latency)
        bucket_bases = self.latency_model.get(
            "bucket_base_ms", {}
        )
        default_base = float(
            self.latency_model.get("default_bucket_base_ms", 0.0)
        )
        self._bucket_bases = {
            bucket: float(
                bucket_bases.get(
                    str(bucket),
                    bucket_bases.get(bucket, default_base),
                )
            )
            if isinstance(bucket_bases, Mapping)
            else default_base
            for bucket in self.buckets
        }
        coefficient_rows = self.latency_model.get(
            "segment_coefficients", {}
        )
        self._segment_coefficients_array = np.zeros(
            (len(self.simulator.segment_compute_ms), 4),
            dtype=np.float64,
        )
        for segment_id in range(
            self._segment_coefficients_array.shape[0]
        ):
            row = (
                coefficient_rows.get(
                    str(segment_id),
                    coefficient_rows.get(segment_id, {}),
                )
                if isinstance(coefficient_rows, Mapping)
                else {}
            )
            if not isinstance(row, Mapping):
                continue
            self._segment_coefficients_array[segment_id] = (
                float(row.get("cpu_experts", 0.0)),
                float(row.get("cpu_routes", 0.0)),
                float(row.get("max_layer_experts", 0.0)),
                float(row.get("transfer_submits", 0.0)),
            )
        default_error = float(
            self.latency_model.get("error_p90_ms", 0.0)
        )
        errors = self.latency_model.get(
            "error_p90_ms_by_bucket", {}
        )
        self._error_p90_by_bucket = {
            bucket: float(
                errors.get(
                    str(bucket),
                    errors.get(bucket, default_error),
                )
            )
            if isinstance(errors, Mapping)
            else default_error
            for bucket in self.buckets
        }
        self._include_transfer_submits = bool(
            self.latency_model.get(
                "include_transfer_submits", False
            )
        )
        expected_id = str(self.artifact.get("model_id", "") or "")
        self.model_id = compute_model_id(self.artifact)
        if expected_id and expected_id != self.model_id:
            raise ValueError(
                f"v3 artifact model_id mismatch: stored={expected_id} computed={self.model_id}"
            )

    @classmethod
    def load(cls, path: str | Path) -> "TransferAwareVerifyCostModel":
        with Path(path).open("r", encoding="utf-8") as stream:
            artifact = json.load(stream)
        if not isinstance(artifact, dict):
            raise ValueError("v3 verify cost artifact must contain a JSON object")
        return cls(artifact)

    def validate_runtime(self, runtime: Mapping[str, object]) -> None:
        protocol = self.artifact.get("protocol")
        if not isinstance(protocol, Mapping):
            raise ValueError("v3 artifact is missing protocol identity")
        for key in (
            "batch_size",
            "acceptance_strategy",
            "temperature",
            "cache_ratio",
            "max_draft_tokens",
            "prefetch_runtime_kind",
        ):
            expected = protocol.get(key)
            actual = runtime.get(key)
            if expected is None or actual is None:
                raise ValueError(f"v3 protocol identity is incomplete for {key}")
            if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
                if abs(float(expected) - float(actual)) > 1e-6:
                    raise ValueError(
                        f"v3 protocol mismatch for {key}: expected={expected} actual={actual}"
                    )
            elif str(expected) != str(actual):
                raise ValueError(
                    f"v3 protocol mismatch for {key}: expected={expected!r} actual={actual!r}"
                )
        runtime_buckets = tuple(int(value) for value in runtime.get("buckets", ()))
        required = tuple(int(value) for value in protocol.get("buckets", self.buckets))
        if not set(required).issubset(runtime_buckets):
            raise ValueError(
                f"v3 dense graph buckets missing: required={required} actual={runtime_buckets}"
            )

    def validate_fingerprint(self, runtime: Mapping[str, object]) -> None:
        expected = self.artifact.get("fingerprint")
        if not isinstance(expected, Mapping):
            raise ValueError("v3 artifact is missing hardware/kernel fingerprint")
        for key, expected_value in expected.items():
            if expected_value in (None, "", "unknown"):
                continue
            actual_value = runtime.get(str(key))
            if actual_value in (None, "") or str(actual_value) != str(expected_value):
                raise ValueError(
                    "v3 fingerprint mismatch for "
                    f"{key}: expected={expected_value!r} actual={actual_value!r}"
                )

    def _bucket_base(self, bucket: int) -> float:
        bases = self.latency_model.get("bucket_base_ms", {})
        if isinstance(bases, Mapping):
            raw = bases.get(str(bucket), bases.get(bucket))
            if raw is not None:
                return float(raw)
        return float(self.latency_model.get("default_bucket_base_ms", 0.0))

    def _segment_coefficients(self, segment_id: int) -> Mapping[str, object]:
        rows = self.latency_model.get("segment_coefficients", {})
        if isinstance(rows, Mapping):
            row = rows.get(str(segment_id), rows.get(segment_id))
            if isinstance(row, Mapping):
                return row
        return {}

    def predict_simulation(
        self,
        simulation: ShadowSimulation | ShadowWorkload,
        *,
        bucket: int,
        logical_tokens: int,
    ) -> TransferAwarePrediction:
        total = self._bucket_bases.get(int(bucket))
        if total is None:
            total = self._bucket_base(bucket)
        for segment in simulation.segments:
            segment_id = int(segment.segment_id)
            if (
                0
                <= segment_id
                < self._segment_coefficients_array.shape[0]
            ):
                coeff = self._segment_coefficients_array[
                    segment_id
                ]
                total += coeff[0] * segment.cpu_experts
                total += coeff[1] * segment.cpu_routes
                total += coeff[2] * segment.max_layer_experts
                if self._include_transfer_submits:
                    total += (
                        coeff[3] * segment.transfer_submits
                    )
            else:
                coeff = self._segment_coefficients(segment_id)
                total += (
                    float(coeff.get("cpu_experts", 0.0))
                    * segment.cpu_experts
                )
                total += (
                    float(coeff.get("cpu_routes", 0.0))
                    * segment.cpu_routes
                )
                total += (
                    float(
                        coeff.get(
                            "max_layer_experts", 0.0
                        )
                    )
                    * segment.max_layer_experts
                )
                if self._include_transfer_submits:
                    total += (
                        float(
                            coeff.get(
                                "transfer_submits", 0.0
                            )
                        )
                        * segment.transfer_submits
                    )
        error = self._error_p90_by_bucket.get(int(bucket), 0.0)
        return TransferAwarePrediction(
            total_ms=max(1e-6, float(total)),
            error_p90_ms=max(0.0, float(error)),
            bucket=int(bucket),
            logical_tokens=int(logical_tokens),
            cpu_experts=simulation.cpu_experts,
            cpu_routes=simulation.cpu_routes,
            transfer_submits=simulation.transfer_submits,
            transfer_pending=simulation.transfer_pending,
            segments=simulation.segments,
        )

    def reset(self) -> None:
        self.demand.reset()

    def observe(self, original_routes: np.ndarray) -> None:
        self.demand.observe(original_routes)

    def predict_pair(
        self,
        *,
        state: ShadowCacheState,
        logical_tokens: int,
        vpb: int,
        next_draft_ms: float,
    ) -> tuple[TransferAwarePrediction, TransferAwarePrediction]:
        current_bucket = select_verify_bucket(logical_tokens, self.buckets)
        next_logical = int(logical_tokens) + 1
        next_bucket = select_verify_bucket(next_logical, self.buckets)
        current_demand, next_demand = (
            self.demand.predict_aggregate_pair(
                logical_tokens=logical_tokens,
                bucket=current_bucket,
                next_bucket=next_bucket,
                current_out=self._current_demand_scratch,
                next_out=self._next_demand_scratch,
            )
        )
        current_sim, next_sim = self.simulator.simulate_pair_fast(
            state,
            current_demand=current_demand,
            next_demand=next_demand,
            vpb=int(vpb),
            next_draft_ms=float(next_draft_ms),
        )
        current = self.predict_simulation(
            current_sim,
            bucket=current_bucket,
            logical_tokens=logical_tokens,
        )
        next_prediction = self.predict_simulation(
            next_sim,
            bucket=next_bucket,
            logical_tokens=next_logical,
        )
        return current, next_prediction


def bind_runtime_cache_host_matrices(
    *,
    layer_caches: Mapping[int, object],
    num_layers: int,
    num_experts: int,
) -> RuntimeCacheHostMatrices:
    """Bind per-layer cache mirrors to rows of contiguous matrices."""
    resident = np.zeros(
        (int(num_layers), int(num_experts)), dtype=bool
    )
    last_access = np.full(
        (int(num_layers), int(num_experts)),
        -1,
        dtype=np.int32,
    )
    access_count = np.zeros(
        (int(num_layers), int(num_experts)),
        dtype=np.int32,
    )
    slot_counts = np.zeros(int(num_layers), dtype=np.int16)
    for layer_idx in range(int(num_layers)):
        cache = layer_caches.get(layer_idx)
        if cache is None:
            continue
        mask = np.asarray(
            getattr(cache, "cached_expert_mask_host", ()),
            dtype=bool,
        )
        width = min(int(num_experts), int(mask.size))
        if width:
            resident[layer_idx, :width] = mask[:width]
        last = np.asarray(
            getattr(
                cache,
                "last_access_step_array",
                getattr(cache, "last_access_step", ()),
            ),
            dtype=np.int32,
        )
        width = min(int(num_experts), int(last.size))
        if width:
            last_access[layer_idx, :width] = last[:width]
        counts = np.asarray(
            getattr(
                cache,
                "access_count_array",
                getattr(cache, "access_count", ()),
            ),
            dtype=np.int32,
        )
        width = min(int(num_experts), int(counts.size))
        if width:
            access_count[layer_idx, :width] = counts[:width]
        slot_counts[layer_idx] = int(
            getattr(cache, "num_slots", 0)
        )
        # Cache mutations already update these mirrors element-wise. Rebinding
        # them to matrix rows turns 48 small snapshot copies into two bulk
        # copies without changing cache ownership or synchronization.
        cache.cached_expert_mask_host = resident[layer_idx]
        cache.last_access_step_array = last_access[layer_idx]
        cache.access_count_array = access_count[layer_idx]
    return RuntimeCacheHostMatrices(
        resident=resident,
        last_access=last_access,
        access_count=access_count,
        slot_counts=slot_counts,
    )


def snapshot_runtime_state(
    *,
    layer_caches: Mapping[int, object],
    prefetch_runtime: object | None,
    num_layers: int,
    num_experts: int,
    transfer_ms: float,
    materialize_layers: bool = True,
    array_scratch: object | None = None,
    resident_host_matrix: np.ndarray | None = None,
    access_host_matrix: np.ndarray | None = None,
    slot_counts_host: np.ndarray | None = None,
) -> ShadowCacheState:
    """Copy live host metadata without querying or synchronizing CUDA work."""
    now_ms = perf_counter() * 1000.0
    inflight: list[ShadowTransfer] = []
    for ticket in getattr(prefetch_runtime, "inflight", {}).values():
        if not bool(getattr(ticket, "direct_active", False)):
            continue
        ready = bool(getattr(ticket, "ready", False))
        ready_event = getattr(ticket, "ready_event", None)
        if not ready and ready_event is not None:
            # event.query is non-blocking and is also used by the live runtime.
            ready = bool(ready_event.query())
        elapsed = max(
            0.0,
            now_ms - float(getattr(ticket, "submit_ts_ms", now_ms)),
        )
        remaining = (
            0.0
            if ready
            else max(1e-6, float(transfer_ms) - elapsed)
        )
        source = str(getattr(ticket, "source", "runtime"))
        inflight.append(
            ShadowTransfer(
                layer_idx=int(ticket.layer_idx),
                expert_idx=int(ticket.expert_idx),
                slot_idx=int(
                    getattr(ticket, "active_slot_idx", -1)
                ),
                previous_expert=int(
                    getattr(ticket, "active_slot_prev_expert", -1)
                ),
                remaining_ms=remaining,
                source=source,
                deferred_publish=source
                in {
                    "draft_segment_indexed",
                    "verify_segment_indexed",
                    "dual_queue_draft_predict",
                    "dual_queue_verify_predict",
                },
            )
        )

    if not materialize_layers:
        caches = [layer_caches.get(idx) for idx in range(int(num_layers))]
        scratch = (
            array_scratch
            if isinstance(array_scratch, _ArrayShadowState)
            and bool(array_scratch.expert_only)
            and array_scratch.resident.shape
            == (int(num_layers), int(num_experts))
            else None
        )
        if scratch is None:
            resident = np.zeros(
                (int(num_layers), int(num_experts)), dtype=bool
            )
            access = np.zeros(
                (int(num_layers), int(num_experts)),
                dtype=np.float32,
            )
            protected = np.zeros_like(resident, dtype=bool)
            pending_experts = np.zeros_like(resident, dtype=bool)
            reserved_victims = np.zeros_like(resident, dtype=bool)
            empty_slots = np.zeros(int(num_layers), dtype=np.int16)
            scratch = _ArrayShadowState(
                slots=np.empty(
                    (int(num_layers), 0), dtype=np.int16
                ),
                pending_slots=np.empty(
                    (int(num_layers), 0), dtype=np.int16
                ),
                valid_slots=np.empty(
                    (int(num_layers), 0), dtype=bool
                ),
                access_values=access,
                protected=protected,
                resident=resident,
                pending_experts=pending_experts,
                inflight=[],
                empty_slots=empty_slots,
                reserved_victims=reserved_victims,
                expert_only=True,
            )
        resident = scratch.resident
        access = scratch.access_values
        protected = scratch.protected
        pending_experts = scratch.pending_experts
        reserved_victims = scratch.reserved_victims
        empty_slots = scratch.empty_slots
        resident.fill(False)
        access.fill(0.0)
        protected.fill(False)
        pending_experts.fill(False)
        if reserved_victims is not None:
            reserved_victims.fill(False)
        if empty_slots is not None:
            empty_slots.fill(0)
        matrix_shape = (int(num_layers), int(num_experts))
        has_resident_matrix = (
            resident_host_matrix is not None
            and np.shape(resident_host_matrix) == matrix_shape
        )
        has_access_matrix = (
            access_host_matrix is not None
            and np.shape(access_host_matrix) == matrix_shape
        )
        if has_resident_matrix:
            np.copyto(
                resident,
                np.asarray(resident_host_matrix, dtype=bool),
            )
        if has_access_matrix:
            np.copyto(
                access,
                np.asarray(access_host_matrix),
                casting="unsafe",
            )
        if not has_resident_matrix or not has_access_matrix:
            for layer_idx, cache in enumerate(caches):
                if cache is None:
                    continue
                if not has_resident_matrix:
                    mask = np.asarray(
                        getattr(
                            cache,
                            "cached_expert_mask_host",
                            (),
                        ),
                        dtype=bool,
                    )
                    width = min(
                        int(num_experts), int(mask.size)
                    )
                    if width:
                        resident[layer_idx, :width] = mask[:width]
                if has_access_matrix:
                    continue
                strategy_values = getattr(
                    cache, "last_access_step_array", None
                )
                if strategy_values is None:
                    strategy_values = getattr(
                        cache, "last_access_step", None
                    )
                if strategy_values is None:
                    strategy_values = getattr(
                        cache, "access_count_array", None
                    )
                if strategy_values is None:
                    strategy_values = getattr(
                        cache, "access_count", ()
                    )
                values = np.asarray(
                    strategy_values, dtype=np.float32
                )
                width = min(
                    int(num_experts), int(values.size)
                )
                if width:
                    access[layer_idx, :width] = values[:width]
        protected_by_layer = (
            getattr(prefetch_runtime, "_round_loaded", {}) or {}
        )
        for layer_idx, values in protected_by_layer.items():
            layer_idx = int(layer_idx)
            if not (0 <= layer_idx < int(num_layers)):
                continue
            expert_values = np.fromiter(
                (
                    int(value)
                    for value in values
                    if 0 <= int(value) < int(num_experts)
                ),
                dtype=np.int64,
            )
            protected[layer_idx, expert_values] = True
        pending_empty = np.zeros(int(num_layers), dtype=np.int16)
        for ticket in inflight:
            layer_idx = int(ticket.layer_idx)
            expert_idx = int(ticket.expert_idx)
            previous = int(ticket.previous_expert)
            if (
                0 <= layer_idx < int(num_layers)
                and 0 <= expert_idx < int(num_experts)
            ):
                pending_experts[layer_idx, expert_idx] = True
            if (
                0 <= layer_idx < int(num_layers)
                and 0 <= previous < int(num_experts)
            ):
                reserved_victims[layer_idx, previous] = True
            elif 0 <= layer_idx < int(num_layers):
                pending_empty[layer_idx] += 1
        if (
            slot_counts_host is not None
            and np.shape(slot_counts_host) == (int(num_layers),)
        ):
            slot_counts = np.asarray(
                slot_counts_host, dtype=np.int16
            )
        else:
            slot_counts = np.fromiter(
                (
                    int(getattr(cache, "num_slots", 0))
                    if cache is not None
                    else 0
                    for cache in caches
                ),
                dtype=np.int16,
                count=int(num_layers),
            )
        np.subtract(
            slot_counts,
            resident.sum(axis=1, dtype=np.int16),
            out=empty_slots,
        )
        empty_slots -= pending_empty
        np.maximum(empty_slots, 0, out=empty_slots)
        scratch.inflight = list(inflight)
        return ShadowCacheState(
            layers=(),
            inflight=tuple(inflight),
            array_state=scratch,
        )

    layers: list[ShadowLayerState] = []
    protected_by_layer = getattr(prefetch_runtime, "_round_loaded", {}) or {}
    max_slots = max(
        (
            len(getattr(cache, "slot_to_expert", ()))
            for cache in layer_caches.values()
        ),
        default=0,
    )
    slot_array = np.full(
        (int(num_layers), max_slots), -2, dtype=np.int16
    )
    pending_array = np.full_like(slot_array, -2)
    valid_slots = np.zeros_like(slot_array, dtype=bool)
    access_array = np.zeros(
        (int(num_layers), int(num_experts)), dtype=np.float32
    )
    protected_array = np.zeros_like(access_array, dtype=bool)
    resident_array = np.zeros_like(access_array, dtype=bool)
    pending_expert_array = np.zeros_like(access_array, dtype=bool)
    for layer_idx in range(int(num_layers)):
        cache = layer_caches.get(layer_idx)
        if cache is None:
            layers.append(
                ShadowLayerState(
                    slots=(),
                    pending=(),
                    access_values=(0.0,) * int(num_experts),
                )
            )
            continue
        slots = tuple(int(value) for value in getattr(cache, "slot_to_expert", ()))
        pending = tuple(
            int(value)
            for value in getattr(
                cache, "active_slot_pending_expert", (-1,) * len(slots)
            )
        )
        strategy_values = getattr(cache, "last_access_step", None)
        if strategy_values is None:
            strategy_values = getattr(cache, "access_count", ())
        access_values = tuple(float(value) for value in strategy_values)
        if len(access_values) < int(num_experts):
            access_values += (0.0,) * (int(num_experts) - len(access_values))
        protected_experts = frozenset(
            int(value)
            for value in protected_by_layer.get(layer_idx, ())
        )
        layers.append(
            ShadowLayerState(
                slots=slots,
                pending=pending,
                access_values=access_values[: int(num_experts)],
                protected_experts=protected_experts,
            )
        )
        count = len(slots)
        if count:
            slot_array[layer_idx, :count] = slots
            pending_array[layer_idx, :count] = pending
            valid_slots[layer_idx, :count] = True
            slot_values = np.asarray(slots, dtype=np.int64)
            resident_values = slot_values[
                (slot_values >= 0)
                & (slot_values < int(num_experts))
            ]
            resident_array[layer_idx, resident_values] = True
            pending_values = np.asarray(pending, dtype=np.int64)
            pending_values = pending_values[
                (pending_values >= 0)
                & (pending_values < int(num_experts))
            ]
            pending_expert_array[layer_idx, pending_values] = True
        access_array[layer_idx] = np.asarray(
            access_values[: int(num_experts)], dtype=np.float32
        )
        if protected_experts:
            protected_values = np.fromiter(
                (
                    value
                    for value in protected_experts
                    if 0 <= value < int(num_experts)
                ),
                dtype=np.int64,
            )
            protected_array[layer_idx, protected_values] = True

    array_state = _ArrayShadowState(
        slots=slot_array,
        pending_slots=pending_array,
        valid_slots=valid_slots,
        access_values=access_array,
        protected=protected_array,
        resident=resident_array,
        pending_experts=pending_expert_array,
        inflight=list(inflight),
    )
    return ShadowCacheState(
        layers=tuple(layers),
        inflight=tuple(inflight),
        array_state=array_state,
    )


def prediction_to_runtime_dict(
    model: TransferAwareVerifyCostModel,
    current: TransferAwarePrediction,
    next_prediction: TransferAwarePrediction,
) -> dict[str, object]:
    """Flatten a one-step prediction pair for the existing RPC/trace boundary."""
    return {
        "verify_cost_schema_version": SCHEMA_VERSION,
        "verify_cost_model_id": model.model_id,
        "verify_cost_prediction_ms": current.total_ms,
        "verify_cost_error_p90_ms": current.error_p90_ms,
        "verify_cost_bucket": current.bucket,
        "verify_cost_logical_tokens": current.logical_tokens,
        "verify_cost_cpu_experts": current.cpu_experts,
        "verify_cost_cpu_routes": current.cpu_routes,
        "verify_cost_transfer_submits": current.transfer_submits,
        "verify_cost_transfer_pending": current.transfer_pending,
        "verify_cost_lookahead_prediction_ms": next_prediction.total_ms,
        "verify_cost_lookahead_error_p90_ms": next_prediction.error_p90_ms,
        "verify_cost_lookahead_bucket": next_prediction.bucket,
        "verify_cost_lookahead_logical_tokens": next_prediction.logical_tokens,
        "verify_cost_lookahead_cpu_experts": next_prediction.cpu_experts,
        "verify_cost_lookahead_cpu_routes": next_prediction.cpu_routes,
        "verify_cost_lookahead_transfer_submits": next_prediction.transfer_submits,
        "verify_cost_lookahead_transfer_pending": next_prediction.transfer_pending,
        "verify_cost_state_complete": True,
    }
