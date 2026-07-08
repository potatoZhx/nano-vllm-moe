#!/usr/bin/env python3
"""Analyze draft segment-graph op-event profiles."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _ep(ep: dict[str, Any], key: str) -> Any:
    if key in ep:
        return ep[key]
    return ep.get(f"model_{key}")


def _ep_float(ep: dict[str, Any], key: str) -> float:
    return _float(_ep(ep, key))


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _records(ep: dict[str, Any]) -> list[dict[str, Any]]:
    recs = _ep(ep, "draft_op_event_records")
    if isinstance(recs, list):
        return recs
    return []


def _draft_calls(raw: dict[str, Any]) -> int:
    ep = raw.get("engine_profile", {})
    value = _ep(ep, "run_draft_count")
    if value:
        return max(1, _int(value, 1))
    return max(1, _int(raw.get("summary", {}).get("draft_forward_count"), 1))


def summarize_labels(records: list[dict[str, Any]], calls: int, scale: float) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in records:
        grouped[str(row.get("label", ""))].append(_float(row.get("elapsed_ms")))

    out: list[dict[str, Any]] = []
    for label, values in grouped.items():
        if not values:
            continue
        ordered = sorted(values)
        p95 = ordered[int(0.95 * (len(ordered) - 1))]
        total = float(sum(values))
        out.append(
            {
                "label": label,
                "count": len(values),
                "total_ms": total,
                "ms_per_call": total / calls,
                "scaled_ms_per_call": total / calls * scale,
                "avg_ms_op": float(statistics.mean(values)),
                "p95_ms_op": p95,
                "max_ms_op": max(values),
            }
        )
    out.sort(key=lambda item: (-float(item["ms_per_call"]), str(item["label"])))
    return out


def per_segment(records: list[dict[str, Any]], calls: int, labels: list[str]) -> list[dict[str, Any]]:
    segments = sorted({_int(row.get("segment")) for row in records})
    totals: dict[tuple[int, str], float] = defaultdict(float)
    for row in records:
        totals[(_int(row.get("segment")), str(row.get("label", "")))] += _float(
            row.get("elapsed_ms")
        )
    rows = []
    for segment in segments:
        item: dict[str, Any] = {"segment": segment}
        for label in labels:
            item[label] = totals[(segment, label)] / calls
        rows.append(item)
    return rows


def build_summary(op_raw: dict[str, Any], segment_raw: dict[str, Any] | None) -> dict[str, Any]:
    op_ep = op_raw.get("engine_profile", {})
    seg_raw = segment_raw or op_raw
    seg_ep = seg_raw.get("engine_profile", {})
    op_calls = _draft_calls(op_raw)
    seg_calls = _draft_calls(seg_raw)
    records = _records(op_ep)

    op_segment_ms_per_call = _ep_float(op_ep, "draft_segment_cuda_event_ms") / op_calls
    normal_segment_ms_per_call = _ep_float(seg_ep, "draft_segment_cuda_event_ms") / seg_calls
    normal_draft_ms = _float(seg_raw.get("summary", {}).get("draft_forward_ms_avg"))
    scale = (
        normal_segment_ms_per_call / op_segment_ms_per_call
        if op_segment_ms_per_call > 0.0 and normal_segment_ms_per_call > 0.0
        else 1.0
    )
    labels = summarize_labels(records, op_calls, scale)
    label_by_name = {row["label"]: row for row in labels}
    layer_total = _float(label_by_name.get("layer.total", {}).get("ms_per_call"))
    final_norm = _float(label_by_name.get("model.final_norm", {}).get("ms_per_call"))
    op_uninstrumented = max(0.0, op_segment_ms_per_call - layer_total - final_norm)

    external = [
        ("draft wall", normal_draft_ms),
        ("draft segment CUDA event", normal_segment_ms_per_call),
        ("graph-external gap", max(0.0, normal_draft_ms - normal_segment_ms_per_call)),
        ("run_draft mode set", _ep_float(seg_ep, "run_draft_mode_set_ms") / seg_calls),
        ("run_draft prefetch before", _ep_float(seg_ep, "run_draft_prefetch_before_ms") / seg_calls),
        ("run_draft core_run", _ep_float(seg_ep, "run_draft_core_run_ms") / seg_calls),
        ("core_run minus segment event", max(0.0, _ep_float(seg_ep, "run_draft_core_run_ms") / seg_calls - normal_segment_ms_per_call)),
        ("sample decode", _ep_float(seg_ep, "sample_decode_ms") / seg_calls),
        ("prepare sample decode", _ep_float(seg_ep, "prepare_sample_decode_ms") / seg_calls),
        ("draft metadata enqueue", _ep_float(seg_ep, "draft_segment_metadata_enqueue_ms") / seg_calls),
        ("draft prefetch visible overhead", _ep_float(seg_ep, "draft_segment_indexed_prefetch_visible_overhead_ms") / seg_calls),
        ("draft prefetch rank", _ep_float(seg_ep, "draft_segment_indexed_rank_ms") / seg_calls),
        ("draft prefetch transfer enqueue", _ep_float(seg_ep, "draft_segment_indexed_prefetch_transfer_enqueue_ms") / seg_calls),
        ("metadata collect worker", _ep_float(seg_ep, "run_draft_metadata_collect_ms") / seg_calls),
        ("metadata observe worker", _ep_float(seg_ep, "run_draft_metadata_observe_ms") / seg_calls),
        ("run_draft submit_after worker", _ep_float(seg_ep, "run_draft_submit_after_ms") / seg_calls),
    ]

    return {
        "op_profile": {
            "draft_calls": op_calls,
            "record_count": len(records),
            "op_segment_ms_per_call": op_segment_ms_per_call,
            "op_event_sync_ms_per_call": _ep_float(op_ep, "draft_op_event_sync_ms") / op_calls,
            "op_uninstrumented_segment_gap_ms_per_call": op_uninstrumented,
        },
        "normal_profile": {
            "draft_calls": seg_calls,
            "draft_forward_ms_avg": normal_draft_ms,
            "segment_ms_per_call": normal_segment_ms_per_call,
            "graph_external_gap_ms_per_call": max(0.0, normal_draft_ms - normal_segment_ms_per_call),
            "scale_op_to_normal_segment": scale,
        },
        "external_breakdown": [
            {"item": name, "ms_per_call": value} for name, value in external
        ],
        "labels": labels,
        "segments": per_segment(
            records,
            op_calls,
            [
                "layer.total",
                "layer.moe",
                "layer.attention",
                "moe.heterogeneous_forward",
                "moe.gpu_gate_up",
                "moe.gpu_down",
                "moe.plan",
                "moe.draft_reroute",
                "moe.runtime_metadata_record",
            ],
        ),
    }


def write_records_csv(records: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "step_id",
        "bucket",
        "segment",
        "token_count",
        "layer_idx",
        "label",
        "elapsed_ms",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    normal = summary["normal_profile"]
    op = summary["op_profile"]
    lines = [
        "# Draft Op Breakdown",
        "",
        "## Profile Scope",
        "",
        f"- normal draft calls: `{normal['draft_calls']}`",
        f"- normal draft forward: `{normal['draft_forward_ms_avg']:.3f} ms/call`",
        f"- normal draft segment CUDA event: `{normal['segment_ms_per_call']:.3f} ms/call`",
        f"- graph-external gap: `{normal['graph_external_gap_ms_per_call']:.3f} ms/call`",
        f"- op-event records: `{op['record_count']}`",
        f"- op-event segment CUDA event: `{op['op_segment_ms_per_call']:.3f} ms/call`",
        f"- op-event sync overhead: `{op['op_event_sync_ms_per_call']:.3f} ms/call`",
        f"- scale op-event labels to normal segment: `{normal['scale_op_to_normal_segment']:.4f}`",
        "",
        "## Graph-External Breakdown",
        "",
        "| item | ms/call |",
        "|:---|---:|",
    ]
    for item in summary["external_breakdown"]:
        lines.append(f"| {item['item']} | {item['ms_per_call']:.3f} |")
    lines.extend(
        [
            "",
            "## Top Op Labels",
            "",
            "| label | count | ms/call raw | ms/call scaled | avg ms/op | p95 | max |",
            "|:---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["labels"][:32]:
        lines.append(
            "| "
            f"`{row['label']}` | {row['count']} | "
            f"{row['ms_per_call']:.3f} | {row['scaled_ms_per_call']:.3f} | "
            f"{row['avg_ms_op']:.4f} | {row['p95_ms_op']:.4f} | {row['max_ms_op']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Per Segment",
            "",
            "| segment | layer.total | layer.moe | attention | hetero | gate_up | down | plan | reroute | metadata_record |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["segments"]:
        lines.append(
            "| "
            f"{row['segment']} | {row['layer.total']:.3f} | "
            f"{row['layer.moe']:.3f} | {row['layer.attention']:.3f} | "
            f"{row['moe.heterogeneous_forward']:.3f} | "
            f"{row['moe.gpu_gate_up']:.3f} | {row['moe.gpu_down']:.3f} | "
            f"{row['moe.plan']:.3f} | {row['moe.draft_reroute']:.3f} | "
            f"{row['moe.runtime_metadata_record']:.3f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze draft op-event breakdown.")
    parser.add_argument("--op-json", required=True)
    parser.add_argument("--segment-json", default="")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    op_raw = _load(Path(args.op_json))
    segment_raw = _load(Path(args.segment_json)) if args.segment_json else None
    summary = build_summary(op_raw, segment_raw)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    op_records = _records(op_raw.get("engine_profile", {}))
    write_records_csv(op_records, out_dir / "draft_op_event_records.csv")
    (out_dir / "draft_op_breakdown_summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(summary, out_dir / "draft_op_breakdown_summary.md")
    print(f"records_csv={out_dir / 'draft_op_event_records.csv'}")
    print(f"summary_json={out_dir / 'draft_op_breakdown_summary.json'}")
    print(f"summary_md={out_dir / 'draft_op_breakdown_summary.md'}")


if __name__ == "__main__":
    main()
