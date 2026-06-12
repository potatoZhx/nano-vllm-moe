#!/usr/bin/env python3
"""Aggregate JSON results from verify_segment_bench_more into a CSV."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def main() -> None:
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/home/linke/nano-vllm-moe/results/verify_segment_bench_more"
    )
    output_csv = results_dir / "aggregated.csv"

    rows = []
    for json_path in sorted(results_dir.glob("*.json")):
        name = json_path.stem
        data = json.loads(json_path.read_text(encoding="utf-8"))
        s = data.get("summary", data)

        accept = s.get("acceptance", {})
        cache = s.get("cache", {})
        cuda_graph = s.get("cuda_graph", {})
        prefetch = s.get("prefetch", {})

        row = {
            "name": name,
            "decode_phase_output_tok_s": s.get("decode_phase_output_tok_s", ""),
            "acceptance_rate": accept.get("acceptance_rate", ""),
            "true_route_hit_rate": cache.get("true_route_hit_rate", ""),
            "draft_forward_ms_avg": s.get("draft_forward_ms_avg", ""),
            "verify_forward_ms_avg": s.get("verify_forward_ms_avg", ""),
            "verify_call_count": cuda_graph.get("verify_call_count", ""),
            "kt_hybrid_replay_count": cuda_graph.get("verify_kt_hybrid_replay_count", ""),
            "segment_replay_count": cuda_graph.get("verify_kt_hybrid_segment_graph_replay_count", ""),
            "graph_hit_rate": cuda_graph.get("hit_rate", ""),
            "seg_prefetch_submit": prefetch.get("verify_segment_prefetch_submit_count", ""),
            "seg_prefetch_candidate_ranked": prefetch.get("verify_segment_prefetch_candidate_ranked_count", ""),
            "seg_prefetch_no_candidate": prefetch.get("verify_segment_prefetch_no_candidate_count", ""),
            "seg_prefetch_skipped_budget": prefetch.get("verify_segment_prefetch_skipped_by_budget_count", ""),
        }
        rows.append(row)

    if not rows:
        print("No JSON files found.")
        return

    fieldnames = list(rows[0].keys())
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()
