from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass
class VerifyOpEvent:
    bucket: int
    segment: int
    label: str
    layer_idx: int
    start: torch.cuda.Event
    end: torch.cuda.Event


_capture_stack: list[tuple[int, int]] = []
_events_by_graph: dict[tuple[int, int], list[VerifyOpEvent]] = {}


def verify_op_event_enabled() -> bool:
    return os.getenv("NANOVLLM_VERIFY_OP_EVENT_TIMING", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


@contextmanager
def verify_op_capture_context(bucket: int, segment: int) -> Iterator[None]:
    if not verify_op_event_enabled():
        yield
        return
    _capture_stack.append((int(bucket), int(segment)))
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
        or not torch.cuda.is_current_stream_capturing()
    ):
        with nullcontext():
            yield
        return

    bucket, segment = _capture_stack[-1]
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    start.record(torch.cuda.current_stream())
    try:
        yield
    finally:
        end.record(torch.cuda.current_stream())
        _events_by_graph.setdefault((bucket, segment), []).append(
            VerifyOpEvent(
                bucket=int(bucket),
                segment=int(segment),
                label=str(label),
                layer_idx=int(layer_idx),
                start=start,
                end=end,
            )
        )


def collect_verify_op_events(bucket: int, segment: int) -> list[dict[str, float | int | str]]:
    events = _events_by_graph.get((int(bucket), int(segment)), [])
    rows: list[dict[str, float | int | str]] = []
    for event in events:
        try:
            elapsed_ms = float(event.start.elapsed_time(event.end))
        except Exception as exc:
            rows.append(
                {
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
                "bucket": int(event.bucket),
                "segment": int(event.segment),
                "layer_idx": int(event.layer_idx),
                "label": str(event.label),
                "elapsed_ms": elapsed_ms,
            }
        )
    return rows
