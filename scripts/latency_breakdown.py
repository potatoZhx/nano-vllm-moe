#!/usr/bin/env python3
"""TPOT latency-breakdown accounting and report generation.

The additive hierarchy in this module deliberately uses request-wall and
coarse CUDA-event boundaries.  Host counters that may overlap GPU execution
are retained as diagnostics and never silently added to the critical path.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


SCHEMA_VERSION = 1
ABS_CLOSURE_TOLERANCE_MS_PER_TOKEN = 0.05
REL_CLOSURE_TOLERANCE = 0.005
UNATTRIBUTED_ABS_WARNING_MS_PER_TOKEN = 0.5
UNATTRIBUTED_PARENT_WARNING_RATIO = 0.05
EXPECTED_SEGMENTS = 4
EXPECTED_VERIFY_LAYERS = 48

TOP_LEVEL_FIELDS = (
    "draft_gpu_compute",
    "draft_transfer_exposed",
    "draft_other",
    "verify_gpu_compute",
    "verify_cpu_compute_exposed",
    "verify_transfer_exposed",
    "verify_other",
    "decode_residual",
)


def numeric(values: dict[str, Any], key: str) -> float:
    value = values.get(key, 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(pct) / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return (
        ordered[lower] * (1.0 - fraction)
        + ordered[upper] * fraction
    )


def _diagnostic(
    section: str,
    source: str,
    total_ms: float,
    *,
    overlaps_gpu: bool,
    description: str,
) -> dict[str, Any]:
    return {
        "section": section,
        "source": source,
        "total_ms": float(total_ms),
        "overlaps_gpu": bool(overlaps_gpu),
        "additive": False,
        "description": description,
    }


def build_request_breakdown(
    row: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    intervals = int(row.get("decode_token_intervals", 0) or 0)
    wall_ms = float(row.get("decode_sec", 0.0) or 0.0) * 1000.0
    if intervals <= 0:
        raise ValueError("decode_token_intervals must be positive")
    if wall_ms <= 0.0:
        raise ValueError("decode_sec must be positive")

    draft_total = numeric(profile, "spec_draft_total_ms")
    verify_total = numeric(profile, "spec_verify_accept_ready_ms")

    draft_segment_gpu = numeric(
        profile, "model_draft_segment_cuda_event_ms"
    )
    draft_tail_gpu = numeric(
        profile, "model_latency_draft_tail_cuda_event_ms"
    )
    draft_sample_gpu = numeric(
        profile, "model_latency_draft_sample_cuda_event_ms"
    )
    draft_gpu = draft_segment_gpu + draft_tail_gpu + draft_sample_gpu

    draft_incall_transfer = numeric(
        profile, "model_direct_active_prefetch_sync_wait_ms"
    )
    draft_preverify_transfer = numeric(
        profile, "model_verify_prefetch_transfer_wait_ms"
    )
    draft_transfer = draft_incall_transfer + draft_preverify_transfer
    draft_other = draft_total - draft_gpu - draft_transfer

    verify_segment_stream = numeric(
        profile, "model_verify_segment_cuda_event_ms"
    )
    verify_lm_head = numeric(
        profile, "model_latency_verify_lm_head_cuda_event_ms"
    )
    verify_stream = verify_segment_stream + verify_lm_head
    verify_cpu = numeric(
        profile, "model_verify_op_kt_cpuinfer_sync_ms"
    )
    verify_transfer = numeric(
        profile, "model_verify_op_kt_output_cpu_to_gpu_copy_ms"
    )
    verify_gpu = verify_stream - verify_cpu - verify_transfer
    verify_other = (
        verify_total - verify_gpu - verify_cpu - verify_transfer
    )

    decode_residual = wall_ms - draft_total - verify_total
    top_level = {
        "draft_gpu_compute": draft_gpu,
        "draft_transfer_exposed": draft_transfer,
        "draft_other": draft_other,
        "verify_gpu_compute": verify_gpu,
        "verify_cpu_compute_exposed": verify_cpu,
        "verify_transfer_exposed": verify_transfer,
        "verify_other": verify_other,
        "decode_residual": decode_residual,
    }

    draft_call_wall = numeric(
        profile, "spec_run_draft_infer_ms_total"
    )
    draft_loop = numeric(profile, "spec_draft_loop_ms")
    draft_setup = (
        numeric(profile, "spec_start_draft_ms")
        + numeric(profile, "spec_rollback_ms")
        + numeric(profile, "spec_prepare_verify_ms")
    )
    draft_entry_initial = (
        numeric(profile, "spec_draft_entry_ms")
        + numeric(profile, "spec_draft_initial_policy_ms")
    )
    verify_prefetch_call = numeric(
        profile, "spec_verify_prefetch_call_ms"
    )
    draft_sources = {
        "draft_call_host_and_tail": (
            draft_call_wall - draft_gpu - draft_incall_transfer
        ),
        "draft_policy_and_sequence": draft_loop - draft_call_wall,
        "draft_setup_and_handoff": draft_setup,
        "draft_entry_and_initial_policy": draft_entry_initial,
        "draft_prefetch_orchestration": (
            verify_prefetch_call - draft_preverify_transfer
        ),
    }
    draft_sources["draft_other_unattributed"] = (
        draft_other - sum(draft_sources.values())
    )

    verify_call_wall = numeric(
        profile, "spec_run_verify_infer_ms_total"
    )
    verify_prepare = numeric(
        profile, "model_verify_prepare_prefill_ms"
    )
    verify_modelrunner_external = verify_call_wall - verify_stream
    verify_accept = numeric(profile, "spec_accept_ms")
    verify_sources = {
        "verify_prepare_and_context": verify_prepare,
        "verify_modelrunner_external_after_prepare": (
            verify_modelrunner_external - verify_prepare
        ),
        "verify_acceptance": verify_accept,
        "verify_call_boundary": (
            verify_total - verify_call_wall - verify_accept
        ),
    }
    verify_sources["verify_other_unattributed"] = (
        verify_other - sum(verify_sources.values())
    )

    engine_step = numeric(profile, "decode_step_wall_ms")
    engine_schedule = numeric(profile, "decode_schedule_ms")
    engine_spec = numeric(profile, "decode_spec_engine_ms")
    engine_post = numeric(profile, "decode_postprocess_ms")
    spec_post_verify = engine_spec - draft_total - verify_total
    engine_other = (
        engine_step - engine_schedule - engine_spec - engine_post
    )
    decode_sources = {
        "decode_scheduler": engine_schedule,
        "decode_spec_post_verify": spec_post_verify,
        "decode_postprocess": engine_post,
        "decode_engine_wrapper": engine_other,
        "decode_driver": wall_ms - engine_step,
    }
    decode_sources["decode_residual_unattributed"] = (
        decode_residual - sum(decode_sources.values())
    )

    diagnostics = [
        _diagnostic(
            "draft_other",
            "run_draft_mode_set",
            numeric(profile, "model_run_draft_mode_set_ms"),
            overlaps_gpu=False,
            description="draft graph-mode and execution-mode setup",
        ),
        _diagnostic(
            "draft_other",
            "run_draft_prefetch_before",
            numeric(profile, "model_run_draft_prefetch_before_ms"),
            overlaps_gpu=False,
            description=(
                "drain, phase-1 submission, buffer reuse and metadata arm; "
                "its sync-wait child is already classified as transfer"
            ),
        ),
        _diagnostic(
            "draft_other",
            "acceptance_readback",
            numeric(profile, "model_acceptance_readback_ms"),
            overlaps_gpu=False,
            description="acceptance predictor output and route readback",
        ),
        _diagnostic(
            "draft_other",
            "verify_cost_proxy",
            numeric(profile, "model_verify_cost_proxy_total_ms"),
            overlaps_gpu=False,
            description="transfer-aware verify-cost prediction",
        ),
        _diagnostic(
            "draft_other",
            "draft_segment_metadata_enqueue",
            numeric(profile, "model_draft_segment_metadata_enqueue_ms"),
            overlaps_gpu=True,
            description="asynchronous metadata enqueue between graph segments",
        ),
        _diagnostic(
            "draft_other",
            "draft_prefetch_visible_overhead",
            numeric(
                profile,
                "model_draft_segment_indexed_prefetch_visible_overhead_ms",
            ),
            overlaps_gpu=True,
            description="rank/reserve/enqueue wall observed by the prefetcher",
        ),
        _diagnostic(
            "verify_other",
            "verify_graph_setup_enqueue",
            numeric(
                profile, "model_verify_segment_graph_setup_enqueue_ms"
            ),
            overlaps_gpu=False,
            description="static input/context setup before segment replays",
        ),
        _diagnostic(
            "verify_other",
            "verify_prefetch_hook",
            numeric(profile, "model_verify_segment_prefetch_hook_ms"),
            overlaps_gpu=True,
            description="publish/rank/prefetch hooks between GPU segments",
        ),
        _diagnostic(
            "verify_other",
            "verify_boundary_submit",
            numeric(profile, "model_verify_segment_boundary_submit_ms"),
            overlaps_gpu=True,
            description="boundary prefetch submission",
        ),
        _diagnostic(
            "verify_other",
            "verify_graph_replay_enqueue",
            numeric(
                profile,
                "model_verify_segment_graph_replay_enqueue_ms",
            ),
            overlaps_gpu=True,
            description="host CUDA-graph replay enqueue time",
        ),
        _diagnostic(
            "verify_other",
            "verify_metadata_enqueue",
            numeric(profile, "model_verify_segment_metadata_enqueue_ms"),
            overlaps_gpu=True,
            description="deferred verify metadata enqueue",
        ),
        _diagnostic(
            "verify_other",
            "verify_metadata_worker",
            (
                numeric(
                    profile,
                    "model_run_verify_kt_hybrid_metadata_collect_ms",
                )
                + numeric(
                    profile,
                    "model_run_verify_kt_hybrid_metadata_observe_ms",
                )
            ),
            overlaps_gpu=True,
            description="async metadata collect and observe worker work",
        ),
        _diagnostic(
            "decode_residual",
            "spec_trace_finalize",
            max(0.0, spec_post_verify),
            overlaps_gpu=False,
            description=(
                "finished-sequence cleanup, profile trace construction and "
                "acceptance-state cleanup after verify acceptance"
            ),
        ),
    ]
    diagnostics.sort(
        key=lambda item: (
            str(item["section"]),
            -float(item["total_ms"]),
        )
    )

    total_closure_error = wall_ms - sum(top_level.values())
    per_token = {
        key: value / intervals for key, value in top_level.items()
    }
    per_token["tpot"] = wall_ms / intervals
    shares = {
        key: value / wall_ms for key, value in top_level.items()
    }
    closure_per_token = total_closure_error / intervals
    closure_rel = (
        abs(total_closure_error) / wall_ms if wall_ms > 0.0 else 0.0
    )

    draft_calls = numeric(profile, "spec_run_draft_calls")
    verify_calls = numeric(profile, "spec_run_verify_calls")
    coverage = {
        "draft_segment": {
            "actual": numeric(
                profile, "model_draft_segment_cuda_event_count"
            ),
            "expected": draft_calls * EXPECTED_SEGMENTS,
        },
        "draft_tail": {
            "actual": numeric(
                profile,
                "model_latency_draft_tail_cuda_event_count",
            ),
            "expected": draft_calls,
        },
        "draft_sample": {
            "actual": numeric(
                profile,
                "model_latency_draft_sample_cuda_event_count",
            ),
            "expected": draft_calls,
        },
        "verify_segment": {
            "actual": numeric(
                profile, "model_verify_segment_cuda_event_count"
            ),
            "expected": verify_calls * EXPECTED_SEGMENTS,
        },
        "verify_lm_head": {
            "actual": numeric(
                profile,
                "model_latency_verify_lm_head_cuda_event_count",
            ),
            "expected": verify_calls,
        },
        "verify_cpu_event": {
            "actual": numeric(
                profile, "model_verify_op_kt_cpuinfer_sync_count"
            ),
            "expected": verify_calls * EXPECTED_VERIFY_LAYERS,
        },
        "verify_transfer_event": {
            "actual": numeric(
                profile,
                "model_verify_op_kt_output_cpu_to_gpu_copy_count",
            ),
            "expected": verify_calls * EXPECTED_VERIFY_LAYERS,
        },
    }

    errors: list[str] = []
    warnings: list[str] = []
    if (
        abs(closure_per_token) > ABS_CLOSURE_TOLERANCE_MS_PER_TOKEN
        and closure_rel > REL_CLOSURE_TOLERANCE
    ):
        errors.append(
            "top-level TPOT closure exceeds tolerance: "
            f"{closure_per_token:.6f} ms/token ({closure_rel:.3%})"
        )
    for key in (
        "draft_gpu_compute",
        "draft_transfer_exposed",
        "verify_gpu_compute",
        "verify_cpu_compute_exposed",
        "verify_transfer_exposed",
    ):
        if top_level[key] < -1e-3:
            errors.append(f"negative critical-path category: {key}")
    for name, item in coverage.items():
        if (
            item["expected"] > 0
            and abs(item["actual"] - item["expected"]) > 0.5
        ):
            errors.append(
                f"event coverage {name}={item['actual']:.0f}, "
                f"expected={item['expected']:.0f}"
            )
    if numeric(profile, "model_verify_op_event_sync_count") > 0.0:
        errors.append("legacy verify op-event forced synchronization was used")
    if numeric(profile, "model_draft_op_event_sync_count") > 0.0:
        errors.append("legacy draft op-event forced synchronization was used")

    source_groups = {
        "draft_other": draft_sources,
        "verify_other": verify_sources,
        "decode_residual": decode_sources,
    }
    parent_values = {
        "draft_other": draft_other,
        "verify_other": verify_other,
        "decode_residual": decode_residual,
    }
    for parent, sources in source_groups.items():
        unattributed_name = f"{parent}_unattributed"
        unattributed = float(sources.get(unattributed_name, 0.0))
        ratio = (
            abs(unattributed) / abs(parent_values[parent])
            if abs(parent_values[parent]) > 1e-9
            else 0.0
        )
        if (
            abs(unattributed) / intervals
            > UNATTRIBUTED_ABS_WARNING_MS_PER_TOKEN
            or ratio > UNATTRIBUTED_PARENT_WARNING_RATIO
        ):
            warnings.append(
                f"{unattributed_name}={unattributed / intervals:.6f} "
                f"ms/token ({ratio:.2%} of parent)"
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": str(row.get("dataset", "")),
        "sample_id": str(row.get("sample_id", "")),
        "sample_index": int(row.get("sample_index", 0) or 0),
        "runtime_seed": int(row.get("runtime_seed", 0) or 0),
        "outputs_digest": str(row.get("outputs_digest", "")),
        "generated_output_tokens": int(
            row.get("generated_output_tokens", 0) or 0
        ),
        "decode_token_intervals": intervals,
        "decode_wall_ms": wall_ms,
        "totals_ms": top_level,
        "per_token_ms": per_token,
        "shares_of_tpot": shares,
        "parents_ms": {
            "draft_total": draft_total,
            "verify_total": verify_total,
            "verify_stream": verify_stream,
            "decode_wall": wall_ms,
        },
        "other_sources_ms": source_groups,
        "diagnostics": diagnostics,
        "coverage": coverage,
        "closure": {
            "total_error_ms": total_closure_error,
            "error_ms_per_token": closure_per_token,
            "relative_error": closure_rel,
        },
        "warnings": warnings,
        "errors": errors,
        "passed": not errors,
    }


def aggregate_breakdowns(
    requests: list[dict[str, Any]],
) -> dict[str, Any]:
    if not requests:
        raise ValueError("cannot aggregate an empty request list")
    total_intervals = sum(
        int(item["decode_token_intervals"]) for item in requests
    )
    total_wall_ms = sum(float(item["decode_wall_ms"]) for item in requests)
    pooled_total_ms = {
        key: sum(float(item["totals_ms"][key]) for item in requests)
        for key in TOP_LEVEL_FIELDS
    }
    pooled_per_token = {
        key: value / total_intervals
        for key, value in pooled_total_ms.items()
    }
    pooled_per_token["tpot"] = total_wall_ms / total_intervals

    distributions = {}
    for key in ("tpot", *TOP_LEVEL_FIELDS):
        values = [
            float(item["per_token_ms"][key]) for item in requests
        ]
        distributions[key] = {
            "mean": mean(values),
            "p50": percentile(values, 50),
            "p90": percentile(values, 90),
        }

    pooled_sources: dict[str, dict[str, float]] = {}
    for parent in ("draft_other", "verify_other", "decode_residual"):
        names = {
            name
            for item in requests
            for name in item["other_sources_ms"][parent]
        }
        pooled_sources[parent] = {
            name: sum(
                float(
                    item["other_sources_ms"][parent].get(name, 0.0)
                )
                for item in requests
            )
            / total_intervals
            for name in sorted(names)
        }

    diagnostic_totals: dict[tuple[str, str], dict[str, Any]] = {}
    for item in requests:
        for diagnostic in item["diagnostics"]:
            key = (
                str(diagnostic["section"]),
                str(diagnostic["source"]),
            )
            target = diagnostic_totals.setdefault(
                key,
                {
                    "section": key[0],
                    "source": key[1],
                    "total_ms": 0.0,
                    "overlaps_gpu": bool(
                        diagnostic["overlaps_gpu"]
                    ),
                    "additive": False,
                    "description": str(
                        diagnostic["description"]
                    ),
                },
            )
            target["total_ms"] += float(diagnostic["total_ms"])
    diagnostics = []
    for value in diagnostic_totals.values():
        row = dict(value)
        row["ms_per_token"] = row["total_ms"] / total_intervals
        diagnostics.append(row)
    diagnostics.sort(
        key=lambda item: (
            str(item["section"]),
            -abs(float(item["ms_per_token"])),
        )
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "request_count": len(requests),
        "passed_request_count": sum(
            1 for item in requests if bool(item["passed"])
        ),
        "total_decode_token_intervals": total_intervals,
        "total_decode_wall_ms": total_wall_ms,
        "pooled_totals_ms": pooled_total_ms,
        "pooled_per_token_ms": pooled_per_token,
        "request_distributions_ms_per_token": distributions,
        "other_sources_pooled_ms_per_token": pooled_sources,
        "diagnostics_pooled": diagnostics,
        "warnings": [
            {
                "sample_id": item["sample_id"],
                "messages": list(item["warnings"]),
            }
            for item in requests
            if item["warnings"]
        ],
        "errors": [
            {
                "sample_id": item["sample_id"],
                "messages": list(item["errors"]),
            }
            for item in requests
            if item["errors"]
        ],
        "passed": all(bool(item["passed"]) for item in requests),
    }


def flatten_request(item: dict[str, Any]) -> dict[str, Any]:
    row = {
        "dataset": item["dataset"],
        "sample_id": item["sample_id"],
        "sample_index": item["sample_index"],
        "runtime_seed": item["runtime_seed"],
        "outputs_digest": item["outputs_digest"],
        "generated_output_tokens": item["generated_output_tokens"],
        "decode_token_intervals": item["decode_token_intervals"],
        "decode_wall_ms": item["decode_wall_ms"],
        "passed": item["passed"],
        "warnings": "; ".join(item["warnings"]),
        "errors": "; ".join(item["errors"]),
        "closure_error_ms_per_token": item["closure"][
            "error_ms_per_token"
        ],
    }
    for key, value in item["per_token_ms"].items():
        row[f"{key}_ms_per_token"] = value
    for parent, sources in item["other_sources_ms"].items():
        for key, value in sources.items():
            row[f"{parent}.{key}_ms_per_token"] = (
                float(value) / item["decode_token_intervals"]
            )
    return row


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(
        {key for row in materialized for key in row}
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(materialized)
    temporary.replace(path)


def render_markdown(summary: dict[str, Any]) -> str:
    pooled = summary["pooled_per_token_ms"]
    lines = [
        "# TPOT Latency Breakdown",
        "",
        (
            f"- Requests: {summary['request_count']} "
            f"({summary['passed_request_count']} passed)"
        ),
        (
            "- Pooled TPOT: "
            f"{float(pooled['tpot']):.4f} ms/output-token"
        ),
        f"- Gate: {'PASS' if summary['passed'] else 'FAIL'}",
        "",
        "## Additive TPOT composition",
        "",
        "| Component | ms/token | TPOT share |",
        "|---|---:|---:|",
    ]
    for key in TOP_LEVEL_FIELDS:
        value = float(pooled[key])
        share = value / float(pooled["tpot"]) if pooled["tpot"] else 0.0
        lines.append(f"| `{key}` | {value:.4f} | {share:.2%} |")
    lines.extend(
        [
            "",
            "## Residual source analysis",
            "",
        ]
    )
    for parent, sources in summary[
        "other_sources_pooled_ms_per_token"
    ].items():
        lines.extend(
            [
                f"### `{parent}`",
                "",
                "| Additive child | ms/token |",
                "|---|---:|",
            ]
        )
        for name, value in sorted(
            sources.items(),
            key=lambda pair: -abs(float(pair[1])),
        ):
            lines.append(f"| `{name}` | {float(value):.4f} |")
        lines.append("")

    lines.extend(
        [
            "## Overlapping diagnostic sources",
            "",
            (
                "These counters help explain the residuals but are not added "
                "to the critical path when they may overlap GPU execution."
            ),
            "",
            "| Parent | Source | ms/token | overlaps GPU | Interpretation |",
            "|---|---|---:|:---:|---|",
        ]
    )
    for item in summary["diagnostics_pooled"]:
        lines.append(
            f"| `{item['section']}` | `{item['source']}` | "
            f"{float(item['ms_per_token']):.4f} | "
            f"{'yes' if item['overlaps_gpu'] else 'no'} | "
            f"{item['description']} |"
        )

    if summary["warnings"]:
        lines.extend(["", "## Warnings", ""])
        grouped_warnings: dict[str, list[str]] = {}
        for item in summary["warnings"]:
            for message in item["messages"]:
                grouped_warnings.setdefault(message, []).append(
                    str(item["sample_id"])
                )
        for message, sample_ids in grouped_warnings.items():
            preview = ", ".join(sample_ids[:8])
            if len(sample_ids) > 8:
                preview += f", … (+{len(sample_ids) - 8})"
            lines.append(
                f"- {message} Affected samples ({len(sample_ids)}): "
                f"`{preview}`"
            )
    if summary["errors"]:
        lines.extend(["", "## Errors", ""])
        for item in summary["errors"]:
            for message in item["messages"]:
                lines.append(
                    f"- sample `{item['sample_id']}`: {message}"
                )
    return "\n".join(lines) + "\n"
