from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass
class VerifyOpEvent:
    phase: str
    bucket: int
    segment: int
    label: str
    layer_idx: int
    start: torch.cuda.Event
    end: torch.cuda.Event


_capture_stack: list[tuple[str, int, int]] = []
_events_by_graph: dict[tuple[str, int, int], list[VerifyOpEvent]] = {}
_LATENCY_BREAKDOWN_LABELS = {
    "kt.cpuinfer_sync",
    "kt.output_cpu_to_gpu_copy",
}


def latency_breakdown_event_enabled() -> bool:
    return (
        os.getenv("NANOVLLM_LATENCY_BREAKDOWN", "").strip().lower()
        in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
    )


def verify_op_event_enabled() -> bool:
    return (
        latency_breakdown_event_enabled()
        or
        os.getenv("NANOVLLM_VERIFY_OP_EVENT_TIMING", "").strip().lower()
        in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        or os.getenv("NANOVLLM_DRAFT_OP_EVENT_TIMING", "").strip().lower()
        in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
    )


def draft_op_event_enabled() -> bool:
    return os.getenv("NANOVLLM_DRAFT_OP_EVENT_TIMING", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


@contextmanager
def verify_op_capture_context(bucket: int, segment: int, phase: str = "verify") -> Iterator[None]:
    if str(phase) == "draft":
        enabled = draft_op_event_enabled()
    else:
        enabled = (
            latency_breakdown_event_enabled()
            or os.getenv(
                "NANOVLLM_VERIFY_OP_EVENT_TIMING", ""
            ).strip().lower()
            in {
                "1",
                "true",
                "yes",
                "y",
                "on",
            }
        )
    if not enabled:
        yield
        return
    _capture_stack.append((str(phase), int(bucket), int(segment)))
    try:
        yield
    finally:
        _capture_stack.pop()


@contextmanager
def verify_op_event(label: str, layer_idx: int = -1):
    if (
        not verify_op_event_enabled()
        or not _capture_stack
        or not torch.cuda.is_available()
    ):
        with nullcontext():
            yield
        return

    phase, bucket, segment = _capture_stack[-1]
    if (
        latency_breakdown_event_enabled()
        and os.getenv(
            "NANOVLLM_VERIFY_OP_EVENT_TIMING", ""
        ).strip().lower()
        not in {"1", "true", "yes", "y", "on"}
        and str(label) not in _LATENCY_BREAKDOWN_LABELS
    ):
        with nullcontext():
            yield
        return
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    start.record(torch.cuda.current_stream())
    try:
        yield
    finally:
        end.record(torch.cuda.current_stream())
        _events_by_graph.setdefault((phase, bucket, segment), []).append(
            VerifyOpEvent(
                phase=str(phase),
                bucket=int(bucket),
                segment=int(segment),
                label=str(label),
                layer_idx=int(layer_idx),
                start=start,
                end=end,
            )
        )


def collect_verify_op_events(
    bucket: int,
    segment: int,
    phase: str = "verify",
) -> list[dict[str, float | int | str]]:
    events = _events_by_graph.get((str(phase), int(bucket), int(segment)), [])
    rows: list[dict[str, float | int | str]] = []
    for event in events:
        try:
            elapsed_ms = float(event.start.elapsed_time(event.end))
        except Exception as exc:
            rows.append(
                {
                    "phase": str(event.phase),
                    "bucket": int(event.bucket),
                    "segment": int(event.segment),
                    "layer_idx": int(event.layer_idx),
                    "label": str(event.label),
                    "elapsed_ms": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        rows.append(
            {
                "phase": str(event.phase),
                "bucket": int(event.bucket),
                "segment": int(event.segment),
                "layer_idx": int(event.layer_idx),
                "label": str(event.label),
                "elapsed_ms": elapsed_ms,
            }
        )
    return rows
