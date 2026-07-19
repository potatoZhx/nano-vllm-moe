#!/usr/bin/env python3
"""Fit the schema-v3 transfer-aware verify cost artifact.

Inputs are profile JSON files produced by ``bench_eval_workload_tpot.py
--transfer-aware-profile true``.  A whole request/profile/seed is one split
group, so rows from a generation trajectory can never cross train/holdout.

The report intentionally treats accuracy thresholds as diagnostics.  Protocol
identity, row alignment, usable truth metadata, and finite predictions remain
hard validity checks; only the steady draft-call mean is a performance gate.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import hashlib
import json
import math
from pathlib import Path
import platform
from statistics import median
from typing import Any, Iterable, Mapping

import numpy as np

from nanovllm.engine.speculative.transfer_aware_cost_model import (
    CalibratedVerifyDemandPredictor,
    SIMULATOR_SEMANTICS_VERSION,
    ShadowCacheState,
    ShadowLayerState,
    ShadowTransfer,
    TransferAwareVerifyCostModel,
    compute_model_id,
    expected_distinct_experts,
    select_verify_bucket,
)


DENSE_BUCKETS = (5, 7, 8, 9, 10, 11, 12, 13)


def percentile(values: Iterable[float], pct: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * float(pct) / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    weight = rank - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _expand_inputs(patterns: list[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            paths.update(Path(value).resolve() for value in matches)
        else:
            path = Path(pattern)
            if path.exists():
                paths.add(path.resolve())
    return sorted(paths)


def _profile_group_id(path: Path, profile: Mapping[str, Any]) -> str:
    measurement = profile.get("verify_cost_measurement", {})
    sample = measurement.get("sample", {}) if isinstance(measurement, dict) else {}
    parts = (
        str(path.resolve()),
        str(sample.get("dataset", "")),
        str(sample.get("sample_id", "")),
        str(measurement.get("runtime_seed", "")) if isinstance(measurement, dict) else "",
    )
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _profile_request_key(
    profile: Mapping[str, Any],
    *,
    vpb: int,
) -> tuple[object, ...]:
    measurement = profile.get("verify_cost_measurement", {})
    measurement = measurement if isinstance(measurement, Mapping) else {}
    sample = measurement.get("sample", {})
    sample = sample if isinstance(sample, Mapping) else {}
    case = measurement.get("case", {})
    case = case if isinstance(case, Mapping) else {}
    return (
        str(sample.get("dataset", "")),
        str(sample.get("sample_id", "")),
        int(measurement.get("runtime_seed", -1)),
        int(case.get("max_draft_tokens", -1)),
        int(vpb),
    )


def _output_validation(
    path: Path,
    profile: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any]]:
    measurement = profile.get("verify_cost_measurement", {})
    measurement = measurement if isinstance(measurement, Mapping) else {}
    output = measurement.get("output_validation", {})
    output = output if isinstance(output, Mapping) else {}
    if (
        int(output.get("output_sequence_count", 0)) != 1
        or not bool(output.get("fixed_length_ok", False))
        or str(output.get("error", ""))
    ):
        raise ValueError(f"{path}: output validation failed")
    return str(output.get("outputs_digest", "")), measurement


def _join_key(
    profile: Mapping[str, Any],
    trace: Mapping[str, Any],
    call: Mapping[str, Any],
) -> tuple[object, ...]:
    measurement = profile.get("verify_cost_measurement", {})
    measurement = measurement if isinstance(measurement, Mapping) else {}
    sample = measurement.get("sample", {})
    sample = sample if isinstance(sample, Mapping) else {}
    return (
        str(sample.get("dataset", "")),
        str(sample.get("sample_id", "")),
        int(measurement.get("runtime_seed", -1)),
        int(trace.get("step_index", -1)),
        int(trace.get("draft_steps_actual", -1)),
        int(call.get("bucket", -1)),
        int(call.get("token_count", -1)),
        int(call.get("dynamic_budget_value", -1)),
    )


def _load_clean_latency_samples(
    paths: list[Path],
    *,
    num_layers: int,
    num_experts: int,
    top_k: int,
    segment_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load clean latency rows using their own execution workload.

    Profiling can perturb asynchronous transfer completion and therefore the
    sampled trajectory.  A clean latency label must never be attached to an
    instrumented route trace merely because request/seed/step happen to match.
    Clean profiles already contain graph execution routes and per-layer CPU
    route counts, so latency fitting can stay both instrumentation-free and
    trajectory-correct.
    """
    samples: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for path in paths:
        profile = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(profile, dict):
            continue
        measurement = profile.get("verify_cost_measurement", {})
        if (
            isinstance(measurement, Mapping)
            and bool(measurement.get("transfer_aware_enabled", False))
        ):
            raise ValueError(
                f"{path}: latency profile still has transfer-aware instrumentation"
            )
        if isinstance(measurement, Mapping) and bool(
            measurement.get("enabled", False)
        ):
            raise ValueError(
                f"{path}: latency profile still has verify-cost instrumentation"
            )
        digest, measurement = _output_validation(path, profile)
        calls = profile.get(
            "model_verify_call_records", profile.get("verify_call_records", [])
        )
        traces = profile.get("spec_step_traces", [])
        if not isinstance(calls, list) or not isinstance(traces, list):
            continue
        call_by_index = {
            int(row.get("call_index", index)): row
            for index, row in enumerate(calls)
            if isinstance(row, dict)
        }
        group_id = _profile_group_id(path, profile)
        call_vpbs = {
            int(row.get("dynamic_budget_value", -1))
            for row in calls
            if isinstance(row, Mapping)
        }
        if len(call_vpbs) != 1:
            raise ValueError(
                f"{path}: clean profile must contain exactly one vpb, got "
                f"{sorted(call_vpbs)}"
            )
        profile_vpb = next(iter(call_vpbs))
        summaries.append(
            {
                "path": str(path),
                "group_id": group_id,
                "request_key": _profile_request_key(
                    profile, vpb=profile_vpb
                ),
                "outputs_digest": digest,
                "steady_draft_gate": measurement.get(
                    "steady_draft_gate", {}
                ),
                "profile": profile,
            }
        )
        for trace in traces:
            if not isinstance(trace, dict):
                continue
            call = call_by_index.get(int(trace.get("verify_call_index", -1)))
            if not isinstance(call, dict):
                continue
            target = float(trace.get("verify_accept_ready_ms", 0.0))
            execution = _as_layer_rows(
                call.get("metadata_layer_execution_route_rows"),
                num_layers=num_layers,
                top_k=top_k,
            )
            cpu_counts = _cpu_counts_from_record(
                call,
                num_layers=num_layers,
                num_experts=num_experts,
            )
            workload_source = "execution_per_expert_counts"
            if cpu_counts is None:
                cpu_counts = _cpu_counts_from_layer_aggregates(
                    call,
                    num_layers=num_layers,
                    num_experts=num_experts,
                )
                workload_source = "logical_layer_aggregate_counts"
            token_count = int(call.get("token_count", -1))
            bucket = int(call.get("bucket", -1))
            errors = []
            if not (target > 0.0 and math.isfinite(target)):
                errors.append("invalid verify_accept_ready_ms")
            if cpu_counts is None:
                errors.append("missing clean compute-time CPU workload")
            if bucket != select_verify_bucket(token_count, DENSE_BUCKETS):
                errors.append(
                    f"bucket={bucket} does not match logical tokens={token_count}"
                )
            if execution is not None and execution.shape[0] != bucket:
                errors.append("execution row count does not match graph bucket")
            if not bool(call.get("used_cuda_graph", False)):
                errors.append("unexpected eager verify fallback")
            if errors:
                raise ValueError(
                    f"{path}: invalid clean latency call "
                    f"{call.get('call_index')}: " + "; ".join(errors)
                )
            seq_rows = trace.get("sequences", [])
            alphas: list[list[float]] = []
            if isinstance(seq_rows, list) and seq_rows:
                row = seq_rows[0] if isinstance(seq_rows[0], dict) else {}
                raw_alphas = row.get(
                    "calibrated_alpha", row.get("predicted_alpha", [])
                )
                if isinstance(raw_alphas, list):
                    alphas = [[float(value)] for value in raw_alphas]
            samples.append(
                {
                    "path": str(path),
                    "group_id": group_id,
                    "request_key": _profile_request_key(
                        profile,
                        vpb=int(call.get("dynamic_budget_value", -1)),
                    ),
                    "bucket": bucket,
                    "logical_tokens": token_count,
                    "vpb": int(call.get("dynamic_budget_value", 0)),
                    "execution_routes": execution,
                    "cpu_counts": cpu_counts,
                    "cpu_workload_source": workload_source,
                    "padding_workload_accounting": (
                        "explicit_execution_rows"
                        if workload_source == "execution_per_expert_counts"
                        else "bucket_base_with_logical_layer_aggregates"
                    ),
                    # Clean mode deliberately has no ticket lifecycle. Keep an
                    # explicit zero vector so submit-feature selection can only
                    # win when a future clean collector supplies real deltas.
                    "transfer_submits_by_segment": [
                        0
                        for _ in range(
                            math.ceil(num_layers / segment_size)
                        )
                    ],
                    "target_ms": target,
                    "latency_source": "instrumentation_off_own_trajectory",
                    "draft_ms_sum": float(
                        sum(
                            float(value)
                            for value in trace.get("draft_call_ms", [])
                        )
                    ),
                    "draft_call_count": len(
                        trace.get("draft_call_ms", [])
                    ),
                    "expected_output": _expected_output(alphas),
                    "outputs_digest": digest,
                }
            )
    return samples, summaries


def _as_layer_rows(
    mapping: object,
    *,
    num_layers: int,
    top_k: int,
) -> np.ndarray | None:
    if not isinstance(mapping, Mapping):
        return None
    rows = []
    row_count = None
    for layer_idx in range(num_layers):
        layer = mapping.get(str(layer_idx), mapping.get(layer_idx))
        if layer is None:
            return None
        array = np.asarray(layer, dtype=np.int64)
        if array.ndim != 2 or array.shape[1] != top_k:
            return None
        if row_count is None:
            row_count = int(array.shape[0])
        if int(array.shape[0]) != row_count:
            return None
        rows.append(array)
    return np.stack(rows, axis=1)  # [row, layer, top_k]


def _draft_route_steps(
    value: object,
    *,
    num_layers: int,
    top_k: int,
) -> list[np.ndarray]:
    if not isinstance(value, list):
        return []
    out: list[np.ndarray] = []
    for step in value:
        array = np.asarray(step, dtype=np.int64)
        if array.ndim == 3 and array.shape[1] == 1:
            array = array[:, 0, :]
        if array.shape != (num_layers, top_k):
            continue
        out.append(array)
    return out


def _expected_output(alpha_rows: list[list[float]]) -> float:
    if not alpha_rows:
        return 1.0
    product = 1.0
    total = 1.0
    for row in alpha_rows:
        alpha = float(row[0]) if row else 0.0
        product *= min(1.0, max(0.0, alpha))
        total += product
    return total


def _cpu_counts_from_record(
    record: Mapping[str, Any],
    *,
    num_layers: int,
    num_experts: int,
) -> np.ndarray | None:
    mapping = record.get("metadata_layer_execution_cpu_route_counts")
    if not isinstance(mapping, Mapping):
        return None
    rows = np.zeros((num_layers, num_experts), dtype=np.float32)
    for layer_idx in range(num_layers):
        values = mapping.get(str(layer_idx), mapping.get(layer_idx))
        if values is None:
            return None
        array = np.asarray(values, dtype=np.float32)
        if array.shape != (num_experts,):
            return None
        rows[layer_idx] = array
    return rows


def _cpu_counts_from_layer_aggregates(
    record: Mapping[str, Any],
    *,
    num_layers: int,
    num_experts: int,
) -> np.ndarray | None:
    """Preserve the sufficient clean-latency statistics per layer.

    Instrumentation-off profiles deliberately avoid offloading per-expert
    graph execution rows. They still retain compute-time logical CPU route and
    expert counts. The latency design consumes only route sum, nonzero expert
    count, and maximum layer expert count, so compact synthetic rows can
    preserve those statistics without inventing expert identities.
    """
    route_mapping = record.get("metadata_layer_cpu_routes")
    expert_mapping = record.get("metadata_layer_cpu_experts")
    if not isinstance(route_mapping, Mapping) or not isinstance(
        expert_mapping, Mapping
    ):
        return None
    rows = np.zeros((num_layers, num_experts), dtype=np.float32)
    for layer_idx in range(num_layers):
        routes = route_mapping.get(str(layer_idx), route_mapping.get(layer_idx))
        experts = expert_mapping.get(
            str(layer_idx), expert_mapping.get(layer_idx)
        )
        if routes is None or experts is None:
            return None
        route_count = max(0.0, float(routes))
        expert_count = min(
            num_experts, max(0, int(round(float(experts))))
        )
        if expert_count <= 0:
            if route_count > 0.0:
                return None
            continue
        if route_count + 1e-6 < expert_count:
            return None
        rows[layer_idx, :expert_count] = 1.0
        rows[layer_idx, 0] += route_count - float(expert_count)
    return rows


def _demand_counts(route_rows: np.ndarray, num_experts: int) -> np.ndarray:
    counts = np.zeros(
        (route_rows.shape[1], int(num_experts)), dtype=np.float32
    )
    for layer_idx in range(route_rows.shape[1]):
        np.add.at(
            counts[layer_idx],
            route_rows[:, layer_idx, :].reshape(-1),
            1.0,
        )
    return counts


def _actual_cpu_mask(
    record: Mapping[str, Any],
    *,
    num_layers: int,
    num_experts: int,
) -> np.ndarray:
    mapping = record.get("metadata_layer_execution_route_status")
    mask = np.zeros((num_layers, num_experts), dtype=bool)
    route_mapping = record.get("metadata_layer_execution_route_rows")
    if not isinstance(mapping, Mapping) or not isinstance(route_mapping, Mapping):
        cpu = _cpu_counts_from_record(
            record, num_layers=num_layers, num_experts=num_experts
        )
        return (cpu > 0.0) if cpu is not None else mask
    for layer_idx in range(num_layers):
        statuses = np.asarray(mapping.get(str(layer_idx), []), dtype=np.int64)
        routes = np.asarray(route_mapping.get(str(layer_idx), []), dtype=np.int64)
        if statuses.shape != routes.shape:
            continue
        for expert_idx in routes[statuses == 2]:
            if 0 <= int(expert_idx) < num_experts:
                mask[layer_idx, int(expert_idx)] = True
    return mask


def _segment_features(
    cpu_counts: np.ndarray,
    *,
    segment_size: int,
    transfer_submits: list[int] | None = None,
) -> list[dict[str, float]]:
    out = []
    for segment_id, first in enumerate(
        range(0, cpu_counts.shape[0], int(segment_size))
    ):
        rows = cpu_counts[first : first + int(segment_size)]
        layer_experts = expected_distinct_experts(rows, axis=1)
        out.append(
            {
                "cpu_experts": float(layer_experts.sum()),
                "cpu_routes": float(rows.sum()),
                "max_layer_experts": float(
                    layer_experts.max(initial=0)
                ),
                "transfer_submits": float(
                    transfer_submits[segment_id]
                    if transfer_submits
                    and segment_id < len(transfer_submits)
                    else 0
                ),
            }
        )
    return out


def _pair_profiles(
    paths: list[Path],
    *,
    num_layers: int,
    num_experts: int,
    top_k: int,
    segment_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples: list[dict[str, Any]] = []
    profile_summaries: list[dict[str, Any]] = []
    for path in paths:
        profile = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(profile, dict):
            continue
        digest, measurement = _output_validation(path, profile)
        if not bool(measurement.get("transfer_aware_enabled", False)):
            raise ValueError(
                f"{path}: workload profile is missing transfer-aware instrumentation"
            )
        calls = profile.get(
            "model_verify_call_records", profile.get("verify_call_records", [])
        )
        traces = profile.get("spec_step_traces", [])
        if not isinstance(calls, list) or not isinstance(traces, list):
            continue
        call_by_index = {
            int(row.get("call_index", index)): row
            for index, row in enumerate(calls)
            if isinstance(row, dict)
        }
        lifecycle_events = profile.get(
            "model_transfer_lifecycle_events",
            profile.get("transfer_lifecycle_events", []),
        )
        submit_by_step_segment: dict[int, dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        if isinstance(lifecycle_events, list):
            for event in lifecycle_events:
                if (
                    isinstance(event, dict)
                    and event.get("event") == "submit"
                    and int(event.get("segment_id", -1)) >= 0
                ):
                    submit_by_step_segment[int(event.get("step_id", -1))][
                        int(event["segment_id"])
                    ] += 1
        group_id = _profile_group_id(path, profile)
        call_vpbs = {
            int(row.get("dynamic_budget_value", -1))
            for row in calls
            if isinstance(row, Mapping)
        }
        if len(call_vpbs) != 1:
            raise ValueError(
                f"{path}: workload profile must contain exactly one vpb, got "
                f"{sorted(call_vpbs)}"
            )
        profile_vpb = next(iter(call_vpbs))
        steady = (
            profile.get("verify_cost_measurement", {}).get(
                "steady_draft_gate", {}
            )
            if isinstance(profile.get("verify_cost_measurement"), dict)
            else {}
        )
        profile_summaries.append(
            {
                "path": str(path),
                "group_id": group_id,
                "request_key": _profile_request_key(
                    profile, vpb=profile_vpb
                ),
                "outputs_digest": digest,
                "steady_draft_gate": steady,
                "transfer_lifecycle_events": profile.get(
                    "model_transfer_lifecycle_events",
                    profile.get("transfer_lifecycle_events", []),
                ),
                "profile": profile,
            }
        )
        for trace in traces:
            if not isinstance(trace, dict):
                continue
            call = call_by_index.get(int(trace.get("verify_call_index", -1)))
            if not isinstance(call, dict):
                continue
            logical = _as_layer_rows(
                call.get("metadata_layer_logical_route_rows"),
                num_layers=num_layers,
                top_k=top_k,
            )
            execution = _as_layer_rows(
                call.get("metadata_layer_execution_route_rows"),
                num_layers=num_layers,
                top_k=top_k,
            )
            drafts = _draft_route_steps(
                call.get("draft_original_route_rows"),
                num_layers=num_layers,
                top_k=top_k,
            )
            cpu_counts = _cpu_counts_from_record(
                call, num_layers=num_layers, num_experts=num_experts
            )
            if logical is None or execution is None or not drafts or cpu_counts is None:
                continue
            k = min(
                int(trace.get("draft_steps_actual", len(drafts))),
                len(drafts),
            )
            if logical.shape[0] < k + 1:
                continue
            token_count = int(call.get("token_count", logical.shape[0]))
            bucket = int(call.get("bucket", execution.shape[0]))
            expected_bucket = select_verify_bucket(
                token_count, DENSE_BUCKETS
            )
            validity_errors = []
            if len(drafts) != int(trace.get("draft_steps_actual", len(drafts))):
                validity_errors.append("draft route row count mismatch")
            if logical.shape[0] != token_count:
                validity_errors.append(
                    "logical verify route row count mismatch"
                )
            if execution.shape[0] != bucket:
                validity_errors.append(
                    "execution verify route row count mismatch"
                )
            if bucket != expected_bucket:
                validity_errors.append(
                    f"bucket={bucket} expected={expected_bucket}"
                )
            if not bool(call.get("used_cuda_graph", False)):
                validity_errors.append("unexpected eager verify fallback")
            pre_verify_state = call.get("transfer_aware_pre_verify_state")
            if (
                not isinstance(pre_verify_state, Mapping)
                or not isinstance(
                    pre_verify_state.get("cache_layers"), Mapping
                )
                or len(pre_verify_state["cache_layers"]) != num_layers
            ):
                validity_errors.append(
                    "incomplete resident/pending/inflight snapshot"
                )
            target_ms = float(
                trace.get(
                    "verify_accept_ready_ms",
                    call.get("total_ms", 0.0),
                )
            )
            if not (target_ms > 0.0 and math.isfinite(target_ms)):
                validity_errors.append("invalid verify_accept_ready_ms")
            measurement = profile.get("verify_cost_measurement", {})
            output_validation = (
                measurement.get("output_validation", {})
                if isinstance(measurement, Mapping)
                else {}
            )
            if output_validation and (
                int(output_validation.get("output_sequence_count", 0)) != 1
                or not bool(output_validation.get("fixed_length_ok", False))
                or str(output_validation.get("error", ""))
            ):
                validity_errors.append("output validation failed")
            if validity_errors:
                raise ValueError(
                    f"{path}: invalid profile call {call.get('call_index')}: "
                    + "; ".join(validity_errors)
                )
            seq_rows = trace.get("sequences", [])
            alphas: list[list[float]] = []
            if isinstance(seq_rows, list) and seq_rows:
                row = seq_rows[0] if isinstance(seq_rows[0], dict) else {}
                raw_alphas = row.get(
                    "calibrated_alpha", row.get("predicted_alpha", [])
                )
                if isinstance(raw_alphas, list):
                    alphas = [[float(value)] for value in raw_alphas[:k]]
            samples.append(
                {
                    "path": str(path),
                    "group_id": group_id,
                    "bucket": bucket,
                    "logical_tokens": token_count,
                    "vpb": int(call.get("dynamic_budget_value", 0)),
                    "draft_routes": drafts[:k],
                    "logical_routes": logical,
                    "execution_routes": execution,
                    "cpu_counts": cpu_counts,
                    "actual_cpu_mask": _actual_cpu_mask(
                        call,
                        num_layers=num_layers,
                        num_experts=num_experts,
                    ),
                    "pre_verify_state": call.get(
                        "transfer_aware_pre_verify_state", {}
                    ),
                    "transfer_submits_by_segment": [
                        int(
                            submit_by_step_segment[
                                int(call.get("step_id", -1))
                            ].get(segment_id, 0)
                        )
                        for segment_id in range(
                            math.ceil(num_layers / segment_size)
                        )
                    ],
                    "target_ms": target_ms,
                    "latency_source": "instrumented_workload_profile",
                    "draft_ms_sum": float(
                        sum(
                            float(value)
                            for value in trace.get("draft_call_ms", [])
                        )
                    ),
                    "draft_call_count": len(
                        trace.get("draft_call_ms", [])
                    ),
                    "expected_output": _expected_output(alphas),
                }
            )
    return samples, profile_summaries


def _split_groups(
    samples: list[dict[str, Any]],
    *,
    holdout_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups = sorted({str(row["group_id"]) for row in samples})
    if len(groups) < 2:
        raise ValueError(
            "at least two complete request/profile groups are required"
        )
    scored = sorted(
        groups,
        key=lambda value: hashlib.sha256(
            f"{seed}:{value}".encode("utf-8")
        ).digest(),
    )
    holdout_count = max(
        1, min(len(scored) - 1, round(len(scored) * holdout_fraction))
    )
    holdout_groups = set(scored[:holdout_count])
    train = [row for row in samples if row["group_id"] not in holdout_groups]
    holdout = [row for row in samples if row["group_id"] in holdout_groups]
    return train, holdout


def _partition_checkpoint_vpb(
    samples: list[dict[str, Any]],
    checkpoint_vpb: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove a complete-vpb checkpoint before any fit split is formed."""
    if checkpoint_vpb is None:
        return list(samples), []
    group_vpbs: dict[str, set[int]] = defaultdict(set)
    for row in samples:
        group_vpbs[str(row["group_id"])].add(int(row["vpb"]))
    mixed = {
        group_id: sorted(values)
        for group_id, values in group_vpbs.items()
        if checkpoint_vpb in values and values != {checkpoint_vpb}
    }
    if mixed:
        raise ValueError(
            "checkpoint-vpb groups must not mix runtime budgets: "
            + json.dumps(mixed, sort_keys=True)
        )
    checkpoint_groups = {
        group_id
        for group_id, values in group_vpbs.items()
        if values == {checkpoint_vpb}
    }
    fit = [
        row for row in samples if str(row["group_id"]) not in checkpoint_groups
    ]
    checkpoint = [
        row for row in samples if str(row["group_id"]) in checkpoint_groups
    ]
    return fit, checkpoint


def _fit_demand(
    rows: list[dict[str, Any]],
    *,
    num_layers: int,
    num_experts: int,
    top_k: int,
    max_draft_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    positions = max_draft_tokens + 1
    layer_counts = np.ones((num_layers, num_experts), dtype=np.float64) * 1e-3
    position_counts = (
        np.ones((positions, num_layers, num_experts), dtype=np.float64) * 1e-3
    )
    padding_counts = np.zeros((num_layers, num_experts), dtype=np.float64)
    retention_sum = np.zeros((positions, num_layers), dtype=np.float64)
    retention_n = np.zeros((positions, num_layers), dtype=np.float64)
    overlaps = []
    route_errors = []
    for sample in rows:
        logical = sample["logical_routes"]
        execution = sample["execution_routes"]
        for row_idx, verify_row in enumerate(logical):
            position = min(row_idx, positions - 1)
            for layer_idx in range(num_layers):
                np.add.at(
                    layer_counts[layer_idx],
                    verify_row[layer_idx],
                    1.0,
                )
                np.add.at(
                    position_counts[position, layer_idx],
                    verify_row[layer_idx],
                    1.0,
                )
        for draft_idx, draft in enumerate(sample["draft_routes"]):
            if draft_idx >= logical.shape[0] - 1:
                break
            position = min(draft_idx, positions - 1)
            verify = logical[draft_idx]
            for layer_idx in range(num_layers):
                overlap = len(
                    set(int(value) for value in draft[layer_idx])
                    & set(int(value) for value in verify[layer_idx])
                ) / float(top_k)
                retention_sum[position, layer_idx] += overlap
                retention_n[position, layer_idx] += 1.0
                overlaps.append(overlap)
                route_errors.append(
                    abs(
                        len(set(int(value) for value in draft[layer_idx]))
                        - len(set(int(value) for value in verify[layer_idx]))
                    )
                )
        if execution.shape[0] > logical.shape[0]:
            for row in execution[logical.shape[0] :]:
                for layer_idx in range(num_layers):
                    np.add.at(padding_counts[layer_idx], row[layer_idx], 1.0)

    def normalize(array):
        total = array.sum(axis=-1, keepdims=True)
        return array * (float(top_k) / np.maximum(total, 1e-12))

    layer_prior = normalize(layer_counts)
    position_prior = normalize(position_counts)
    padding_prior = normalize(padding_counts)
    retention = np.divide(
        retention_sum,
        np.maximum(retention_n, 1.0),
    )
    missing = retention_n <= 0
    retention[missing] = np.broadcast_to(
        np.mean(retention, axis=0, keepdims=True),
        retention.shape,
    )[missing]

    # Fit the next-row blend on a compact grid, independently per
    # layer/position. This preserves interpretability and avoids overfitting.
    next_weight = np.full((positions, num_layers), 0.5, dtype=np.float64)
    for position in range(1, positions):
        for layer_idx in range(num_layers):
            observations = []
            for sample in rows:
                logical = sample["logical_routes"]
                if logical.shape[0] <= position:
                    continue
                prev = np.bincount(
                    logical[position - 1, layer_idx],
                    minlength=num_experts,
                ).astype(np.float64)
                target = np.bincount(
                    logical[position, layer_idx],
                    minlength=num_experts,
                ).astype(np.float64)
                observations.append((prev, target))
            if not observations:
                continue
            best = (float("inf"), 0.5)
            for weight in np.linspace(0.0, 1.0, 21):
                error = 0.0
                for prev, target in observations:
                    pred = (
                        weight * prev
                        + (1.0 - weight)
                        * position_prior[position, layer_idx]
                    )
                    error += float(np.abs(pred - target).sum())
                best = min(best, (error, float(weight)))
            next_weight[position, layer_idx] = best[1]

    artifact = {
        "retention": retention.tolist(),
        "layer_prior": layer_prior.tolist(),
        "position_prior": position_prior.tolist(),
        "padding_prior": padding_prior.tolist(),
        "next_recent_weight": next_weight.tolist(),
    }
    diagnostics = {
        "aligned_layer_expert_overlap_mean": float(
            np.mean(overlaps) if overlaps else 0.0
        ),
        "aligned_layer_expert_overlap_p10": percentile(overlaps, 10),
        "aligned_route_count_abs_error_mean": float(
            np.mean(route_errors) if route_errors else 0.0
        ),
    }
    return artifact, diagnostics


def _prediction_demand(
    demand_artifact: Mapping[str, Any],
    sample: Mapping[str, Any],
    *,
    num_layers: int,
    num_experts: int,
    top_k: int,
    max_draft_tokens: int,
) -> np.ndarray:
    predictor = CalibratedVerifyDemandPredictor(
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k,
        max_draft_tokens=max_draft_tokens,
        artifact=demand_artifact,
    )
    for routes in sample["draft_routes"]:
        predictor.observe(np.asarray(routes)[:, None, :])
    rows = predictor.predict_rows(
        logical_tokens=int(sample["logical_tokens"]),
        bucket=int(sample["bucket"]),
    )
    return rows


def _demand_diagnostics(
    demand_artifact: Mapping[str, Any],
    rows: list[dict[str, Any]],
    *,
    num_layers: int,
    num_experts: int,
    top_k: int,
    max_draft_tokens: int,
) -> dict[str, Any]:
    overlap_by_position: dict[int, list[float]] = defaultdict(list)
    route_l1_by_position: dict[int, list[float]] = defaultdict(list)
    next_errors = []
    for sample in rows:
        predicted = _prediction_demand(
            demand_artifact,
            sample,
            num_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k,
            max_draft_tokens=max_draft_tokens,
        )
        actual = sample["logical_routes"]
        for position in range(min(actual.shape[0], predicted.shape[0])):
            for layer_idx in range(num_layers):
                actual_set = set(int(v) for v in actual[position, layer_idx])
                top = np.argsort(predicted[position, layer_idx])[-top_k:]
                overlap_by_position[position].append(
                    len(actual_set & set(int(v) for v in top)) / float(top_k)
                )
                actual_counts = np.bincount(
                    actual[position, layer_idx], minlength=num_experts
                )
                route_l1_by_position[position].append(
                    float(
                        np.abs(
                            predicted[position, layer_idx] - actual_counts
                        ).sum()
                    )
                )
        next_position = len(sample["draft_routes"])
        if next_position < actual.shape[0]:
            for layer_idx in range(num_layers):
                actual_counts = np.bincount(
                    actual[next_position, layer_idx], minlength=num_experts
                )
                next_errors.append(
                    float(
                        np.abs(
                            predicted[next_position, layer_idx]
                            - actual_counts
                        ).sum()
                    )
                )
    return {
        "expert_overlap_by_position": {
            str(key): float(np.mean(value))
            for key, value in sorted(overlap_by_position.items())
        },
        "route_count_l1_by_position": {
            str(key): float(np.mean(value))
            for key, value in sorted(route_l1_by_position.items())
        },
        "verify_next_route_count_l1_mean": float(
            np.mean(next_errors) if next_errors else 0.0
        ),
    }


def _fit_transfer_timing(
    profiles: list[dict[str, Any]],
    *,
    default_ms: float,
    bandwidth_gbps: float,
) -> tuple[float, dict[str, Any]]:
    latencies = []
    byte_counts = []
    by_source: dict[str, list[float]] = defaultdict(list)
    for profile in profiles:
        events = profile.get("transfer_lifecycle_events", [])
        if not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, dict) or event.get("event") != "ready":
                continue
            value = float(event.get("latency_from_submit_ms", 0.0))
            if value > 0.0 and math.isfinite(value):
                latencies.append(value)
                by_source[str(event.get("source", "unknown"))].append(
                    value
                )
                num_bytes = int(event.get("num_bytes", 0))
                if num_bytes > 0:
                    byte_counts.append(num_bytes)
    estimate = median(latencies) if latencies else float(default_ms)
    budget_estimate_ms = (
        median(byte_counts)
        / (max(1e-6, float(bandwidth_gbps)) * 1_000_000.0)
        if byte_counts
        else float(default_ms)
    )
    return max(1e-6, estimate), {
        "sample_count": len(latencies),
        "median_ms": estimate,
        "p90_ms": percentile(latencies, 90),
        "budget_estimate_ms": max(1e-6, budget_estimate_ms),
        "budget_bandwidth_gbps": float(bandwidth_gbps),
        "median_expert_bytes": (
            int(median(byte_counts)) if byte_counts else None
        ),
        "by_source": {
            source: {
                "sample_count": len(values),
                "median_ms": median(values),
                "p90_ms": percentile(values, 90),
            }
            for source, values in sorted(by_source.items())
        },
    }


def _fit_draft_vpb_timing(
    rows: list[dict[str, Any]],
    *,
    checkpoint: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Fit the clean per-draft-call cost of changing the runtime vpb.

    Offline vpb simulation otherwise sees only verify-cache benefits and will
    monotonically choose the largest budget. The draft EMA used online is
    observed, but offline selection needs this small transfer-pressure term.
    """
    usable = [
        row
        for row in rows
        if int(row.get("draft_call_count", 0)) > 0
        and math.isfinite(float(row.get("draft_ms_sum", 0.0)))
    ]
    if not usable:
        return (
            {"intercept_ms": 0.0, "slope_ms_per_vpb": 0.0},
            {"sample_count": 0, "by_vpb": {}},
        )
    x = np.asarray([float(row["vpb"]) for row in usable])
    y = np.asarray(
        [
            float(row["draft_ms_sum"])
            / float(row["draft_call_count"])
            for row in usable
        ]
    )
    weights = np.asarray(
        [float(row["draft_call_count"]) for row in usable]
    )
    x_mean = float(np.average(x, weights=weights))
    y_mean = float(np.average(y, weights=weights))
    denominator = float(np.sum(weights * (x - x_mean) ** 2))
    slope = (
        float(np.sum(weights * (x - x_mean) * (y - y_mean)))
        / denominator
        if denominator > 1e-12
        else 0.0
    )
    # More transfer pressure cannot receive an offline draft-time credit.
    slope = max(0.0, slope)
    intercept = max(0.0, y_mean - slope * x_mean)
    fitted_vpbs = sorted({int(row["vpb"]) for row in usable})

    def summarize(values: list[dict[str, Any]]) -> dict[str, Any]:
        by_vpb = {}
        for vpb in sorted({int(row["vpb"]) for row in values}):
            selected = [
                row
                for row in values
                if int(row["vpb"]) == vpb
                and int(row.get("draft_call_count", 0)) > 0
            ]
            call_count = sum(
                int(row["draft_call_count"]) for row in selected
            )
            actual = (
                sum(float(row["draft_ms_sum"]) for row in selected)
                / float(call_count)
                if call_count
                else 0.0
            )
            predicted = intercept + slope * float(vpb)
            by_vpb[str(vpb)] = {
                "row_count": len(selected),
                "draft_call_count": call_count,
                "actual_mean_ms": actual,
                "predicted_mean_ms": predicted,
                "error_ms": predicted - actual,
            }
        return by_vpb

    checkpoint_rows = checkpoint or []
    checkpoint_by_vpb = summarize(checkpoint_rows)
    checkpoint_rates = []
    for value, row in checkpoint_by_vpb.items():
        vpb = int(value)
        distance = min(
            (abs(vpb - fitted) for fitted in fitted_vpbs),
            default=1,
        )
        if distance > 0:
            checkpoint_rates.append(
                abs(float(row["error_ms"])) / float(distance)
            )
    extrapolation_penalty = (
        max(checkpoint_rates) if checkpoint_rates else 0.0
    )
    model = {
        "intercept_ms": float(intercept),
        "slope_ms_per_vpb": float(slope),
        "fit_vpb_min": float(min(fitted_vpbs)),
        "fit_vpb_max": float(max(fitted_vpbs)),
        "extrapolation_penalty_ms_per_call_per_vpb": float(
            extrapolation_penalty
        ),
    }
    return (
        model,
        {
            "sample_count": len(usable),
            "fit_by_vpb": summarize(usable),
            "external_checkpoint_by_vpb": checkpoint_by_vpb,
            "constraint": "nonnegative_vpb_slope",
            "offline_selection_uncertainty": (
                "checkpoint absolute draft-call error per distance to the "
                "nearest fitted vpb is charged only when extrapolating "
                "outside the fitted range"
            ),
        },
    )


def _fit_segment_compute_timing(
    profiles: list[dict[str, Any]],
    *,
    segment_count: int,
    fallback_ms: float,
) -> tuple[list[float], dict[str, Any]]:
    """Estimate the overlap window without adding segment instrumentation.

    Transfer-aware workload profiles already carry a whole-verify CUDA stream
    event. Splitting its duration evenly is deliberately coarse, but preserves
    the clean latency run and is preferable to baking in an unrelated constant.
    Explicit ``--segment-compute-ms`` remains available when per-segment event
    calibration has been collected separately.
    """
    per_segment = []
    for summary in profiles:
        profile = summary.get("profile", {})
        calls = (
            profile.get("model_verify_call_records", [])
            if isinstance(profile, Mapping)
            else []
        )
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            stream_ms = float(call.get("stream_ms", 0.0))
            if (
                bool(call.get("stream_ms_available", False))
                and stream_ms > 0.0
                and math.isfinite(stream_ms)
            ):
                per_segment.append(stream_ms / float(max(1, segment_count)))
    estimate = (
        median(per_segment) if per_segment else max(0.0, float(fallback_ms))
    )
    return [float(estimate)] * int(segment_count), {
        "source": (
            "whole_verify_cuda_stream_divided_by_segment_count"
            if per_segment
            else "fallback"
        ),
        "sample_count": len(per_segment),
        "median_segment_ms": float(estimate),
        "p90_segment_ms": percentile(per_segment, 90),
    }


def _nnls_ridge(
    x: np.ndarray,
    y: np.ndarray,
    *,
    ridge: float,
    iterations: int = 8000,
) -> np.ndarray:
    if x.size == 0:
        raise ValueError("empty latency design matrix")
    scale = np.maximum(np.linalg.norm(x, axis=0), 1e-8)
    xs = x / scale
    gram = xs.T @ xs + float(ridge) * np.eye(xs.shape[1])
    rhs = xs.T @ y
    eigen_max = float(np.linalg.eigvalsh(gram).max(initial=1.0))
    step = 1.0 / max(eigen_max, 1e-8)
    beta = np.maximum(0.0, np.linalg.lstsq(gram, rhs, rcond=None)[0])
    for _ in range(iterations):
        updated = np.maximum(0.0, beta - step * (gram @ beta - rhs))
        if float(np.max(np.abs(updated - beta))) < 1e-10:
            beta = updated
            break
        beta = updated
    return beta / scale


def _design(
    samples: list[dict[str, Any]],
    *,
    segment_size: int,
    include_submits: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    segment_count = math.ceil(samples[0]["cpu_counts"].shape[0] / segment_size)
    names = [f"bucket_{bucket}" for bucket in DENSE_BUCKETS]
    for segment_id in range(segment_count):
        names.extend(
            [
                f"segment_{segment_id}_cpu_experts",
                f"segment_{segment_id}_cpu_routes",
                f"segment_{segment_id}_max_layer_experts",
            ]
        )
        if include_submits:
            names.append(f"segment_{segment_id}_transfer_submits")
    matrix = []
    targets = []
    for sample in samples:
        features = [
            float(int(sample["bucket"]) == bucket) for bucket in DENSE_BUCKETS
        ]
        segments = _segment_features(
            sample["cpu_counts"],
            segment_size=segment_size,
            transfer_submits=sample.get("transfer_submits_by_segment"),
        )
        for segment in segments:
            features.extend(
                [
                    segment["cpu_experts"],
                    segment["cpu_routes"],
                    segment["max_layer_experts"],
                ]
            )
            if include_submits:
                features.append(segment["transfer_submits"])
        matrix.append(features)
        targets.append(float(sample["target_ms"]))
    return (
        np.asarray(matrix, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
        names,
    )


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    return {
        "count": int(actual.size),
        "bias_ms": float(error.mean()) if error.size else 0.0,
        "mae_ms": float(np.abs(error).mean()) if error.size else 0.0,
        "rmse_ms": float(np.sqrt(np.mean(error * error))) if error.size else 0.0,
        "p90_abs_error_ms": percentile(np.abs(error), 90),
    }


def _metrics_by(
    rows: list[dict[str, Any]],
    predicted: np.ndarray,
    *,
    key: str,
) -> dict[str, dict[str, float]]:
    out = {}
    for value in sorted({row[key] for row in rows}):
        indices = [
            index for index, row in enumerate(rows) if row[key] == value
        ]
        actual = np.asarray(
            [rows[index]["target_ms"] for index in indices],
            dtype=np.float64,
        )
        out[str(value)] = _metrics(actual, predicted[indices])
    return out


def _latency_artifact(
    beta: np.ndarray,
    names: list[str],
    holdout: list[dict[str, Any]],
    predicted: np.ndarray,
    *,
    segment_size: int,
    include_submits: bool,
) -> dict[str, Any]:
    coefficients = dict(zip(names, beta.tolist(), strict=True))
    by_bucket = {}
    actual = np.asarray([row["target_ms"] for row in holdout])
    for bucket in DENSE_BUCKETS:
        errors = [
            abs(float(pred) - float(target))
            for pred, target, row in zip(
                predicted, actual, holdout, strict=True
            )
            if int(row["bucket"]) == bucket
        ]
        if errors:
            by_bucket[str(bucket)] = percentile(errors, 90)
    segment_count = math.ceil(
        holdout[0]["cpu_counts"].shape[0] / segment_size
    )
    return {
        "bucket_base_ms": {
            str(bucket): coefficients.get(f"bucket_{bucket}", 0.0)
            for bucket in DENSE_BUCKETS
        },
        "segment_coefficients": {
            str(segment_id): {
                "cpu_experts": coefficients.get(
                    f"segment_{segment_id}_cpu_experts", 0.0
                ),
                "cpu_routes": coefficients.get(
                    f"segment_{segment_id}_cpu_routes", 0.0
                ),
                "max_layer_experts": coefficients.get(
                    f"segment_{segment_id}_max_layer_experts", 0.0
                ),
                "transfer_submits": coefficients.get(
                    f"segment_{segment_id}_transfer_submits", 0.0
                ),
            }
            for segment_id in range(segment_count)
        },
        "include_transfer_submits": bool(include_submits),
        "error_p90_ms": percentile(np.abs(predicted - actual), 90),
        "error_p90_ms_by_bucket": by_bucket,
        "fit_constraint": "nonnegative_ridge",
    }


def _snapshot_from_profile(
    value: object,
    *,
    num_layers: int,
    num_experts: int,
    transfer_ms: float,
) -> ShadowCacheState:
    value = value if isinstance(value, Mapping) else {}
    cache_layers = value.get("cache_layers", {})
    layers = []
    for layer_idx in range(num_layers):
        row = (
            cache_layers.get(str(layer_idx), {})
            if isinstance(cache_layers, Mapping)
            else {}
        )
        slots = tuple(int(v) for v in row.get("slots", ()))
        pending = tuple(int(v) for v in row.get("pending", (-1,) * len(slots)))
        access = tuple(float(v) for v in row.get("last_access_step", ()))
        if len(access) < num_experts:
            access += (0.0,) * (num_experts - len(access))
        round_loaded = value.get("round_loaded", {})
        protected = (
            round_loaded.get(str(layer_idx), ())
            if isinstance(round_loaded, Mapping)
            else ()
        )
        layers.append(
            ShadowLayerState(
                slots=slots,
                pending=pending,
                access_values=access[:num_experts],
                protected_experts=frozenset(int(v) for v in protected),
            )
        )
    inflight = []
    for row in value.get("inflight", ()):
        if not isinstance(row, Mapping) or not bool(row.get("direct_active", False)):
            continue
        remaining = (
            0.0
            if bool(row.get("ready", False))
            else max(1e-6, transfer_ms - float(row.get("age_ms", 0.0)))
        )
        inflight.append(
            ShadowTransfer(
                layer_idx=int(row["layer_idx"]),
                expert_idx=int(row["expert_idx"]),
                slot_idx=int(row.get("active_slot_idx", -1)),
                previous_expert=int(row.get("active_slot_prev_expert", -1)),
                remaining_ms=remaining,
                source=str(row.get("source", "profile")),
            )
        )
    return ShadowCacheState(layers=tuple(layers), inflight=tuple(inflight))


def _ablation_predictions(
    artifact: Mapping[str, Any],
    rows: list[dict[str, Any]],
    *,
    num_layers: int,
    num_experts: int,
    top_k: int,
    segment_size: int,
    draft_vpb_slope_ms: float = 0.0,
    draft_vpb_model: Mapping[str, float] | None = None,
) -> tuple[dict[str, Any], dict[int, float], dict[int, float]]:
    model = TransferAwareVerifyCostModel(artifact)
    labels = (
        "actual_demand+actual_cache",
        "predicted_demand+actual_cache",
        "actual_demand+simulated_cache",
        "predicted_demand+simulated_cache",
    )
    predictions = {label: [] for label in labels}
    actual_latency = []
    predicted_tpot_by_vpb: dict[int, list[float]] = defaultdict(list)
    raw_predicted_tpot_by_vpb: dict[int, list[float]] = defaultdict(list)
    draft_vpb_model = draft_vpb_model or {}
    fit_vpb_min = float(draft_vpb_model.get("fit_vpb_min", 0.0))
    fit_vpb_max = float(draft_vpb_model.get("fit_vpb_max", 16.0))
    extrapolation_penalty = float(
        draft_vpb_model.get(
            "extrapolation_penalty_ms_per_call_per_vpb", 0.0
        )
    )
    for sample in rows:
        predicted_rows = _prediction_demand(
            artifact["demand_model"],
            sample,
            num_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k,
            max_draft_tokens=int(artifact["max_draft_tokens"]),
        )
        actual_counts = sample["cpu_counts"]
        predicted_actual_cache_counts = (
            predicted_rows.sum(axis=0) * sample["actual_cpu_mask"]
        )
        state = _snapshot_from_profile(
            sample["pre_verify_state"],
            num_layers=num_layers,
            num_experts=num_experts,
            transfer_ms=model.simulator.transfer_ms,
        )
        actual_sim = model.simulator.simulate_verify(
            state,
            demand_rows=sample["execution_routes"].shape[0]
            and np.stack(
                [
                    np.eye(num_experts, dtype=np.float32)[
                        sample["execution_routes"][row_idx]
                    ].sum(axis=1)
                    for row_idx in range(sample["execution_routes"].shape[0])
                ]
            ),
            vpb=int(sample["vpb"]),
        )
        predicted_sim = model.simulator.simulate_verify(
            state,
            demand_rows=predicted_rows,
            vpb=int(sample["vpb"]),
        )
        workload_variants = {
            labels[0]: (
                actual_counts,
                sample["transfer_submits_by_segment"],
            ),
            labels[1]: (
                predicted_actual_cache_counts,
                sample["transfer_submits_by_segment"],
            ),
            labels[2]: (
                actual_sim.cpu_route_counts,
                [
                    segment.transfer_submits
                    for segment in actual_sim.segments
                ],
            ),
            labels[3]: (
                predicted_sim.cpu_route_counts,
                [
                    segment.transfer_submits
                    for segment in predicted_sim.segments
                ],
            ),
        }
        include_submits = bool(
            artifact["latency_model"].get(
                "include_transfer_submits", False
            )
        )
        for label, (counts, transfer_submits) in workload_variants.items():
            segments = _segment_features(
                counts,
                segment_size=segment_size,
                transfer_submits=transfer_submits,
            )
            # Use the fitted latency model through a lightweight simulation shell.
            total = float(
                artifact["latency_model"]["bucket_base_ms"].get(
                    str(sample["bucket"]), 0.0
                )
            )
            for segment_id, features in enumerate(segments):
                coeff = artifact["latency_model"]["segment_coefficients"][
                    str(segment_id)
                ]
                total += sum(
                    float(coeff.get(name, 0.0)) * float(features[name])
                    for name in (
                        "cpu_experts",
                        "cpu_routes",
                        "max_layer_experts",
                        *(
                            ("transfer_submits",)
                            if include_submits
                            else ()
                        ),
                    )
                )
            predictions[label].append(total)
        actual_latency.append(float(sample["target_ms"]))

        for vpb in range(17):
            sim = model.simulator.simulate_verify(
                state, demand_rows=predicted_rows, vpb=vpb
            )
            pred = model.predict_simulation(
                sim,
                bucket=int(sample["bucket"]),
                logical_tokens=int(sample["logical_tokens"]),
            )
            draft_call_count = float(
                sample.get("draft_call_count", 0)
            )
            adjusted_draft_ms = max(
                0.0,
                float(sample["draft_ms_sum"])
                + draft_call_count
                * float(draft_vpb_slope_ms)
                * (vpb - int(sample["vpb"])),
            )
            denominator = max(
                1e-8, float(sample["expected_output"])
            )
            raw_predicted_tpot_by_vpb[vpb].append(
                (adjusted_draft_ms + pred.total_ms) / denominator
            )
            outside_distance = max(
                0.0,
                fit_vpb_min - float(vpb),
                float(vpb) - fit_vpb_max,
            )
            predicted_tpot_by_vpb[vpb].append(
                (
                    adjusted_draft_ms
                    + draft_call_count
                    * extrapolation_penalty
                    * outside_distance
                    + pred.total_ms
                )
                / denominator
            )
    actual_array = np.asarray(actual_latency)
    report = {
        label: _metrics(actual_array, np.asarray(values))
        for label, values in predictions.items()
    }
    mean_tpot = {
        vpb: float(np.mean(values))
        for vpb, values in predicted_tpot_by_vpb.items()
    }
    raw_mean_tpot = {
        vpb: float(np.mean(values))
        for vpb, values in raw_predicted_tpot_by_vpb.items()
    }
    return report, mean_tpot, raw_mean_tpot


def _partial_correlations(rows: list[dict[str, Any]]) -> dict[str, float]:
    if len(rows) < 4:
        return {}
    target = np.asarray([row["target_ms"] for row in rows], dtype=np.float64)
    controls = np.asarray(
        [
            [
                1.0,
                *[
                    float(int(row["bucket"]) == bucket)
                    for bucket in DENSE_BUCKETS
                ],
                float(row["logical_tokens"]),
                float(row["expected_output"]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    target_residual = target - controls @ np.linalg.lstsq(
        controls, target, rcond=None
    )[0]
    features = {
        "cpu_routes": np.asarray(
            [row["cpu_counts"].sum() for row in rows], dtype=np.float64
        ),
        "cpu_experts": np.asarray(
            [
                expected_distinct_experts(row["cpu_counts"])
                for row in rows
            ],
            dtype=np.float64,
        ),
        "transfer_submits": np.asarray(
            [
                sum(row["transfer_submits_by_segment"])
                for row in rows
            ],
            dtype=np.float64,
        ),
        "inflight_at_verify": np.asarray(
            [
                len(
                    row.get("pre_verify_state", {}).get("inflight", [])
                    if isinstance(row.get("pre_verify_state"), Mapping)
                    else []
                )
                for row in rows
            ],
            dtype=np.float64,
        ),
    }
    out = {}
    for name, values in features.items():
        residual = values - controls @ np.linalg.lstsq(
            controls, values, rcond=None
        )[0]
        if np.std(residual) <= 1e-12 or np.std(target_residual) <= 1e-12:
            out[name] = 0.0
        else:
            out[name] = float(np.corrcoef(residual, target_residual)[0, 1])
    return out


def _trajectory_match_diagnostics(
    workload_profiles: list[dict[str, Any]],
    latency_profiles: list[dict[str, Any]],
) -> dict[str, Any]:
    def index(rows: list[dict[str, Any]], label: str):
        out = {}
        for row in rows:
            key = tuple(row["request_key"])
            if key in out:
                raise ValueError(f"duplicate {label} request identity: {key}")
            out[key] = str(row.get("outputs_digest", ""))
        return out

    workload = index(workload_profiles, "workload")
    latency = index(latency_profiles, "latency")
    shared = sorted(set(workload) & set(latency))
    matched = sum(
        bool(workload[key])
        and bool(latency[key])
        and workload[key] == latency[key]
        for key in shared
    )
    return {
        "workload_request_count": len(workload),
        "clean_latency_request_count": len(latency),
        "shared_request_count": len(shared),
        "matching_output_digest_count": matched,
        "diverged_output_digest_count": len(shared) - matched,
        "missing_clean_request_count": len(set(workload) - set(latency)),
        "missing_workload_request_count": len(set(latency) - set(workload)),
        "latency_labels_attached_across_trajectories": False,
    }


def _validate_protocol_profiles(
    summaries: list[dict[str, Any]],
    *,
    label: str,
) -> dict[str, Any]:
    signatures = set()
    for summary in summaries:
        profile = summary.get("profile", {})
        measurement = (
            profile.get("verify_cost_measurement", {})
            if isinstance(profile, Mapping)
            else {}
        )
        protocol = (
            measurement.get("protocol", {})
            if isinstance(measurement, Mapping)
            else {}
        )
        if not isinstance(protocol, Mapping):
            protocol = {}
        signature = (
            int(protocol.get("batch_size", -1)),
            str(protocol.get("acceptance_strategy", "")),
            float(protocol.get("temperature", float("nan"))),
            float(protocol.get("cache_ratio", float("nan"))),
            str(protocol.get("prefetch_runtime_kind", "")),
            tuple(int(value) for value in protocol.get("verify_buckets", ())),
        )
        signatures.add(signature)
    if len(signatures) != 1:
        raise ValueError(
            f"{label} profiles contain mixed protocol identities: "
            f"{sorted(map(str, signatures))}"
        )
    signature = next(iter(signatures))
    expected = (
        1,
        "standard_sampling",
        0.8,
        0.3125,
        "predictive",
        DENSE_BUCKETS,
    )
    if (
        signature[0] != expected[0]
        or signature[1] != expected[1]
        or not math.isclose(signature[2], expected[2])
        or not math.isclose(signature[3], expected[3])
        or signature[4] != expected[4]
        or signature[5] != expected[5]
    ):
        raise ValueError(
            f"{label} protocol does not match the validated active scope: "
            f"{signature}"
        )
    return {
        "batch_size": signature[0],
        "acceptance_strategy": signature[1],
        "temperature": signature[2],
        "cache_ratio": signature[3],
        "prefetch_runtime_kind": signature[4],
        "verify_buckets": list(signature[5]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("profiles", nargs="+")
    parser.add_argument(
        "--latency-profiles",
        nargs="*",
        default=[],
        help=(
            "Instrumentation-off profile JSONs. Their own execution workload "
            "is used for latency fitting; labels are never copied onto a "
            "different instrumented trajectory."
        ),
    )
    parser.add_argument(
        "--allow-instrumented-latency",
        action="store_true",
        help="Pre-analysis only: fit latency from instrumented workload profiles.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-layers", type=int, default=48)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-draft-tokens", type=int, default=12)
    parser.add_argument("--segment-size", type=int, default=12)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--split-seed", type=int, default=20260719)
    parser.add_argument(
        "--checkpoint-vpb",
        type=int,
        default=-1,
        help=(
            "Exclude this vpb and all of its complete request groups from every "
            "fit, then report it as an external checkpoint. Negative disables."
        ),
    )
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--default-transfer-ms", type=float, default=1.0)
    parser.add_argument("--transfer-bandwidth-gbps", type=float, default=12.0)
    parser.add_argument("--max-inflight", type=int, default=16)
    parser.add_argument("--draft-budget", type=int, default=16)
    parser.add_argument("--draft-visible-budget-ms", type=float, default=3.0)
    parser.add_argument("--verify-visible-budget-ms", type=float, default=12.0)
    parser.add_argument("--verify-attention-ratio", type=float, default=1.0)
    parser.add_argument("--segment-compute-ms", default="")
    parser.add_argument("--cpu-model", default=platform.processor() or "unknown")
    parser.add_argument("--gpu-model", required=True)
    parser.add_argument("--kt-kernel-version", required=True)
    parser.add_argument("--kt-num-threads", type=int, default=16)
    parser.add_argument("--kt-backend", default="avx2_bf16")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = _expand_inputs(args.profiles)
    if not paths:
        raise SystemExit("no profile JSON files matched")
    latency_paths = _expand_inputs(args.latency_profiles)
    if not latency_paths and not args.allow_instrumented_latency:
        raise SystemExit(
            "active v3 fitting requires --latency-profiles from an "
            "instrumentation-off run"
        )
    samples, profiles = _pair_profiles(
        paths,
        num_layers=args.num_layers,
        num_experts=args.num_experts,
        top_k=args.top_k,
        segment_size=args.segment_size,
    )
    if latency_paths:
        latency_samples, latency_profiles = _load_clean_latency_samples(
            latency_paths,
            num_layers=args.num_layers,
            num_experts=args.num_experts,
            top_k=args.top_k,
            segment_size=args.segment_size,
        )
    else:
        latency_samples, latency_profiles = samples, profiles
    if not samples:
        raise SystemExit(
            "no valid v3 rows; profiles must include draft, logical/execution "
            "routes, CPU execution counts, and verify_accept_ready_ms"
        )
    if not latency_samples:
        raise SystemExit(
            "no valid clean latency rows with execution CPU counts and "
            "verify_accept_ready_ms"
        )
    workload_protocol = _validate_protocol_profiles(
        profiles, label="workload"
    )
    latency_protocol = _validate_protocol_profiles(
        latency_profiles, label="clean latency"
    )
    if workload_protocol != latency_protocol:
        raise ValueError(
            "workload and clean latency protocol identities do not match"
        )
    trajectory_diagnostics = _trajectory_match_diagnostics(
        profiles, latency_profiles
    )
    checkpoint_vpb = (
        int(args.checkpoint_vpb) if int(args.checkpoint_vpb) >= 0 else None
    )
    fit_samples, checkpoint = _partition_checkpoint_vpb(
        samples, checkpoint_vpb
    )
    latency_fit_samples, latency_checkpoint = _partition_checkpoint_vpb(
        latency_samples, checkpoint_vpb
    )
    if checkpoint_vpb is not None and not checkpoint:
        raise SystemExit(
            f"no workload rows found for external checkpoint vpb={checkpoint_vpb}"
        )
    if checkpoint_vpb is not None and not latency_checkpoint:
        raise SystemExit(
            f"no clean latency rows found for external checkpoint "
            f"vpb={checkpoint_vpb}"
        )
    train, holdout = _split_groups(
        fit_samples,
        holdout_fraction=args.holdout_fraction,
        seed=args.split_seed,
    )
    latency_train, latency_holdout = _split_groups(
        latency_fit_samples,
        holdout_fraction=args.holdout_fraction,
        seed=args.split_seed,
    )
    demand_artifact, train_demand = _fit_demand(
        train,
        num_layers=args.num_layers,
        num_experts=args.num_experts,
        top_k=args.top_k,
        max_draft_tokens=args.max_draft_tokens,
    )
    holdout_demand = _demand_diagnostics(
        demand_artifact,
        holdout,
        num_layers=args.num_layers,
        num_experts=args.num_experts,
        top_k=args.top_k,
        max_draft_tokens=args.max_draft_tokens,
    )
    fit_group_ids = {
        str(row["group_id"]) for row in fit_samples
    }
    transfer_ms, transfer_diagnostics = _fit_transfer_timing(
        [
            summary
            for summary in profiles
            if str(summary["group_id"]) in fit_group_ids
        ],
        default_ms=args.default_transfer_ms,
        bandwidth_gbps=args.transfer_bandwidth_gbps,
    )
    if int(transfer_diagnostics["sample_count"]) <= 0:
        raise SystemExit(
            "no submit->ready transfer lifecycle samples; v3 transfer timing "
            "cannot be calibrated"
        )
    fit_profiles = [
        summary
        for summary in profiles
        if str(summary["group_id"]) in fit_group_ids
    ]
    segment_count = math.ceil(args.num_layers / args.segment_size)
    if args.segment_compute_ms:
        segment_compute = [
            float(value)
            for value in args.segment_compute_ms.split(",")
            if value
        ]
        segment_timing_diagnostics = {
            "source": "explicit_cli",
            "sample_count": 0,
            "values_ms": list(segment_compute),
        }
    else:
        segment_compute, segment_timing_diagnostics = (
            _fit_segment_compute_timing(
                fit_profiles,
                segment_count=segment_count,
                fallback_ms=12.0,
            )
        )
    draft_vpb_model, draft_vpb_diagnostics = _fit_draft_vpb_timing(
        latency_train,
        checkpoint=latency_checkpoint,
    )

    fit_candidates = {}
    for include_submits in (False, True):
        train_x, train_y, names = _design(
            latency_train,
            segment_size=args.segment_size,
            include_submits=include_submits,
        )
        holdout_x, holdout_y, _ = _design(
            latency_holdout,
            segment_size=args.segment_size,
            include_submits=include_submits,
        )
        beta = _nnls_ridge(train_x, train_y, ridge=args.ridge)
        predicted = holdout_x @ beta
        fit_candidates[include_submits] = {
            "beta": beta,
            "names": names,
            "predicted": predicted,
            "actual": holdout_y,
            "metrics": _metrics(holdout_y, predicted),
        }
    base_mae = fit_candidates[False]["metrics"]["mae_ms"]
    submit_mae = fit_candidates[True]["metrics"]["mae_ms"]
    # Keep transfer-submit only when its holdout residual improvement is real.
    include_submits = bool(submit_mae + 1e-9 < base_mae)
    selected = fit_candidates[include_submits]
    latency_artifact = _latency_artifact(
        selected["beta"],
        selected["names"],
        latency_holdout,
        selected["predicted"],
        segment_size=args.segment_size,
        include_submits=include_submits,
    )
    latency_artifact["clean_cpu_workload_sources"] = {
        source: sum(
            str(row.get("cpu_workload_source", "")) == source
            for row in latency_samples
        )
        for source in sorted(
            {
                str(row.get("cpu_workload_source", "unknown"))
                for row in latency_samples
            }
        )
    }
    latency_artifact["padding_accounting"] = (
        "exact bucket execution rows when available; otherwise logical "
        "compute-time layer aggregates plus bucket base"
    )
    protocol_case = (
        profiles[0]["profile"].get("verify_cost_measurement", {}).get("case", {})
        if profiles
        else {}
    )
    cache_ratio = float(protocol_case.get("cache_ratio", 0.3125))
    artifact: dict[str, Any] = {
        "schema_version": 3,
        "model_kind": "transfer_aware_verify",
        "simulator_semantics_version": SIMULATOR_SEMANTICS_VERSION,
        "target": "verify_accept_ready_ms",
        "num_layers": args.num_layers,
        "num_experts": args.num_experts,
        "top_k": args.top_k,
        "max_draft_tokens": args.max_draft_tokens,
        "buckets": list(DENSE_BUCKETS),
        "protocol": {
            "batch_size": 1,
            "acceptance_strategy": "standard_sampling",
            "temperature": 0.8,
            "cache_ratio": cache_ratio,
            "max_draft_tokens": args.max_draft_tokens,
            "prefetch_runtime_kind": "predictive",
            "buckets": list(DENSE_BUCKETS),
        },
        "fingerprint": {
            "cpu_model": args.cpu_model,
            "gpu_model": args.gpu_model,
            "kt_kernel_version": args.kt_kernel_version,
            "kt_num_threads": args.kt_num_threads,
            "kt_backend": args.kt_backend,
        },
        "demand_model": demand_artifact,
        "transfer_model": {
            "expert_transfer_ms": transfer_ms,
            "budget_expert_transfer_ms": transfer_diagnostics[
                "budget_estimate_ms"
            ],
            "max_inflight": args.max_inflight,
            "draft_budget": args.draft_budget,
            "draft_visible_budget_ms": args.draft_visible_budget_ms,
            "verify_visible_budget_ms": args.verify_visible_budget_ms,
            "segment_size": args.segment_size,
            "verify_attention_ratio": args.verify_attention_ratio,
            "segment_compute_ms": segment_compute,
            "draft_call_vpb_slope_ms": draft_vpb_model[
                "slope_ms_per_vpb"
            ],
            "vpb_fit_range": [
                int(draft_vpb_model["fit_vpb_min"]),
                int(draft_vpb_model["fit_vpb_max"]),
            ],
            "vpb_extrapolation_penalty_ms_per_call_per_vpb": (
                draft_vpb_model[
                    "extrapolation_penalty_ms_per_call_per_vpb"
                ]
            ),
            "vpb_runtime_range": [0, 16],
        },
        "latency_model": latency_artifact,
        "fit_split": {
            "kind": "complete_request_profile_seed_group",
            "seed": args.split_seed,
            "workload_train_groups": sorted(
                {row["group_id"] for row in train}
            ),
            "workload_holdout_groups": sorted(
                {row["group_id"] for row in holdout}
            ),
            "latency_train_groups": sorted(
                {row["group_id"] for row in latency_train}
            ),
            "latency_holdout_groups": sorted(
                {row["group_id"] for row in latency_holdout}
            ),
            "external_checkpoint_vpb": checkpoint_vpb,
            "workload_external_checkpoint_groups": sorted(
                {row["group_id"] for row in checkpoint}
            ),
            "latency_external_checkpoint_groups": sorted(
                {row["group_id"] for row in latency_checkpoint}
            ),
        },
    }
    artifact["model_id"] = compute_model_id(artifact)
    ablations, vpb_tpot, raw_vpb_tpot = _ablation_predictions(
        artifact,
        holdout,
        num_layers=args.num_layers,
        num_experts=args.num_experts,
        top_k=args.top_k,
        segment_size=args.segment_size,
        draft_vpb_slope_ms=draft_vpb_model["slope_ms_per_vpb"],
        draft_vpb_model=draft_vpb_model,
    )
    ablation_baseline_mae = ablations[
        "actual_demand+actual_cache"
    ]["mae_ms"]
    for values in ablations.values():
        values["mae_delta_vs_actual_demand_actual_cache_ms"] = (
            float(values["mae_ms"]) - float(ablation_baseline_mae)
        )
    checkpoint_demand: dict[str, Any] | None = None
    checkpoint_latency: dict[str, Any] | None = None
    checkpoint_latency_by_bucket: dict[str, Any] = {}
    checkpoint_ablations: dict[str, Any] = {}
    if checkpoint:
        checkpoint_demand = _demand_diagnostics(
            demand_artifact,
            checkpoint,
            num_layers=args.num_layers,
            num_experts=args.num_experts,
            top_k=args.top_k,
            max_draft_tokens=args.max_draft_tokens,
        )
        checkpoint_x, checkpoint_y, _ = _design(
            latency_checkpoint,
            segment_size=args.segment_size,
            include_submits=include_submits,
        )
        checkpoint_predicted = checkpoint_x @ selected["beta"]
        checkpoint_latency = _metrics(
            checkpoint_y, checkpoint_predicted
        )
        checkpoint_latency_by_bucket = _metrics_by(
            latency_checkpoint,
            checkpoint_predicted,
            key="bucket",
        )
        checkpoint_ablations, _, _ = _ablation_predictions(
            artifact,
            checkpoint,
            num_layers=args.num_layers,
            num_experts=args.num_experts,
            top_k=args.top_k,
            segment_size=args.segment_size,
            draft_vpb_slope_ms=draft_vpb_model[
                "slope_ms_per_vpb"
            ],
            draft_vpb_model=draft_vpb_model,
        )
        checkpoint_baseline_mae = checkpoint_ablations[
            "actual_demand+actual_cache"
        ]["mae_ms"]
        for values in checkpoint_ablations.values():
            values["mae_delta_vs_actual_demand_actual_cache_ms"] = (
                float(values["mae_ms"])
                - float(checkpoint_baseline_mae)
            )
    best_vpb = min(vpb_tpot, key=lambda value: (vpb_tpot[value], value))
    steady_values = [
        float(summary["steady_draft_gate"].get("steady_draft_call_mean_ms"))
        for summary in latency_profiles
        if isinstance(summary.get("steady_draft_gate"), Mapping)
        and summary["steady_draft_gate"].get("steady_draft_call_count", 0)
    ]
    report = {
        "artifact_model_id": artifact["model_id"],
        "data_validity": {
            "workload_profile_count": len(paths),
            "workload_sample_count": len(samples),
            "workload_fit_pool_sample_count": len(fit_samples),
            "workload_train_sample_count": len(train),
            "workload_holdout_sample_count": len(holdout),
            "clean_latency_profile_count": len(latency_paths),
            "clean_latency_sample_count": len(latency_samples),
            "clean_latency_fit_pool_sample_count": len(
                latency_fit_samples
            ),
            "clean_latency_train_sample_count": len(latency_train),
            "clean_latency_holdout_sample_count": len(latency_holdout),
            "external_checkpoint_vpb": checkpoint_vpb,
            "workload_external_checkpoint_sample_count": len(checkpoint),
            "clean_latency_external_checkpoint_sample_count": len(
                latency_checkpoint
            ),
            "workload_group_leakage": bool(
                {row["group_id"] for row in train}
                & {row["group_id"] for row in holdout}
            ),
            "clean_latency_group_leakage": bool(
                {row["group_id"] for row in latency_train}
                & {row["group_id"] for row in latency_holdout}
            ),
            "workload_checkpoint_group_leakage": bool(
                ({row["group_id"] for row in train}
                | {row["group_id"] for row in holdout})
                & {row["group_id"] for row in checkpoint}
            ),
            "clean_latency_checkpoint_group_leakage": bool(
                ({row["group_id"] for row in latency_train}
                | {row["group_id"] for row in latency_holdout})
                & {row["group_id"] for row in latency_checkpoint}
            ),
            "workload_buckets_seen": sorted(
                {row["bucket"] for row in samples}
            ),
            "clean_latency_buckets_seen": sorted(
                {row["bucket"] for row in latency_samples}
            ),
            "latency_source": sorted(
                {str(row["latency_source"]) for row in latency_samples}
            ),
            "clean_latency_cpu_workload_sources": latency_artifact[
                "clean_cpu_workload_sources"
            ],
            "protocol": workload_protocol,
            "trajectory_pairing": trajectory_diagnostics,
        },
        "demand_train": train_demand,
        "demand_holdout": holdout_demand,
        "demand_external_checkpoint": checkpoint_demand,
        "transfer_timing": transfer_diagnostics,
        "segment_compute_timing": segment_timing_diagnostics,
        "draft_vpb_timing": {
            "model": draft_vpb_model,
            **draft_vpb_diagnostics,
        },
        "latency_holdout_without_transfer_submit": fit_candidates[False][
            "metrics"
        ],
        "latency_holdout_with_transfer_submit": fit_candidates[True]["metrics"],
        "latency_holdout_selected_by_bucket": _metrics_by(
            latency_holdout,
            selected["predicted"],
            key="bucket",
        ),
        "latency_holdout_selected_by_vpb": _metrics_by(
            latency_holdout,
            selected["predicted"],
            key="vpb",
        ),
        "latency_external_checkpoint": checkpoint_latency,
        "latency_external_checkpoint_by_bucket": (
            checkpoint_latency_by_bucket
        ),
        "transfer_submit_feature_selected": include_submits,
        "partial_correlations_clean_latency_controlling_bucket_logical_and_acceptance": (
            _partial_correlations(latency_holdout)
        ),
        "partial_correlations_instrumented_workload_controlling_bucket_logical_and_acceptance": (
            _partial_correlations(holdout)
        ),
        "transfer_ablations": ablations,
        "transfer_ablations_external_checkpoint": checkpoint_ablations,
        "transfer_ablation_target_source": (
            "instrumented_workload_profile_only; clean labels are not "
            "cross-attached after sampled trajectories diverge"
        ),
        "offline_vpb_predicted_tpot_ms": {
            str(key): value for key, value in sorted(vpb_tpot.items())
        },
        "offline_vpb_raw_predicted_tpot_ms": {
            str(key): value
            for key, value in sorted(raw_vpb_tpot.items())
        },
        "offline_vpb_selection_objective": (
            "predicted TPOT plus checkpoint-derived extrapolation "
            "uncertainty outside the fitted vpb range"
        ),
        "offline_vpb_selected": best_vpb,
        "steady_draft_call_mean_ms_across_profiles": (
            float(np.mean(steady_values)) if steady_values else None
        ),
        "steady_draft_under_19ms_gate_passed": bool(
            steady_values and float(np.mean(steady_values)) < 19.0
        ),
        "provisional_steady_draft_gate_ms": 21.0,
        "provisional_steady_draft_gate_passed": bool(
            steady_values and float(np.mean(steady_values)) < 21.0
        ),
        "target_policy": (
            "all prediction/TPOT accuracy values are diagnostic soft targets; "
            "the first active screen provisionally uses steady draft mean "
            "<21 ms; restoring <19 ms remains a hot-path TODO"
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    report_path = output.with_suffix(output.suffix + ".report.json")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "artifact": str(output),
                "report": str(report_path),
                "model_id": artifact["model_id"],
                "selected_vpb": best_vpb,
                "steady_draft_gate_passed": report[
                    "provisional_steady_draft_gate_passed"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
