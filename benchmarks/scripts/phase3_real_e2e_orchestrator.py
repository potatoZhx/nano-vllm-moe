#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REQUIRED_METRICS = [
    "route_ms",
    "plan_ms",
    "gpu_gather_ms",
    "gpu_compute_ms",
    "cpu_prepare_ms",
    "cpu_compute_ms",
    "cpu_to_gpu_merge_ms",
    "scatter_ms",
    "draft_ms",
    "verify_ms",
    "spec_step_ms",
    "graph_hit_rate",
    "graph_replay_count",
    "cpu_route_ratio",
    "cpu_weight_mass_ratio",
    "activated_expert_set_size",
    "realized_cpu_expert_count",
    "prefetch_submit_count",
    "prefetch_completed_count",
    "prefetch_late_count",
    "prefetch_wait_ms",
    "prefetch_consumed_count",
    "prefetch_timeout_count",
    "publish_count",
    "publish_ms",
    "metadata_offload_ms",
    "metadata_offload_bytes",
    "history_prefetch_submit_count",
    "verify_history_prefetch_submit_count",
    "draft_live_prefetch_submit_count",
    "verify_ready_before_wait_count",
    "verify_ready_after_wait_count",
]


@dataclass
class Case:
    name: str
    args: list[str]


def _build_cases(prefetch_wait_ms: float) -> list[Case]:
    return [
        Case("standard_eager", ["--mode", "standard", "--enforce-eager", "true"]),
        Case("standard_graph", ["--mode", "standard", "--enforce-eager", "false"]),
        Case("heter_baseline", ["--mode", "heter", "--enforce-eager", "true"]),
        Case(
            "spec_prefetch_off",
            [
                "--mode",
                "spec",
                "--spec-enable-prefetch",
                "false",
            ],
        ),
        Case(
            "spec_prefetch_on_full",
            [
                "--mode",
                "spec",
                "--spec-enable-prefetch",
                "true",
                "--prefetch-verify-wait-ms",
                str(prefetch_wait_ms),
                "--prefetch-use-prefill-history",
                "true",
                "--prefetch-use-verify-history",
                "true",
                "--prefetch-use-draft-live",
                "true",
                "--cache-strategy",
                "lru",
                "--prefetch-strategy",
                "history_window",
            ],
        ),
        Case(
            "spec_ablate_draft_live",
            [
                "--mode",
                "spec",
                "--spec-enable-prefetch",
                "true",
                "--prefetch-verify-wait-ms",
                str(prefetch_wait_ms),
                "--prefetch-use-prefill-history",
                "true",
                "--prefetch-use-verify-history",
                "true",
                "--prefetch-use-draft-live",
                "false",
            ],
        ),
        Case(
            "spec_ablate_verify_history",
            [
                "--mode",
                "spec",
                "--spec-enable-prefetch",
                "true",
                "--prefetch-verify-wait-ms",
                str(prefetch_wait_ms),
                "--prefetch-use-prefill-history",
                "true",
                "--prefetch-use-verify-history",
                "false",
                "--prefetch-use-draft-live",
                "true",
            ],
        ),
        Case(
            "spec_ablate_prefill_history",
            [
                "--mode",
                "spec",
                "--spec-enable-prefetch",
                "true",
                "--prefetch-verify-wait-ms",
                str(prefetch_wait_ms),
                "--prefetch-use-prefill-history",
                "false",
                "--prefetch-use-verify-history",
                "true",
                "--prefetch-use-draft-live",
                "true",
            ],
        ),
        Case(
            "spec_ablate_wait_zero",
            [
                "--mode",
                "spec",
                "--spec-enable-prefetch",
                "true",
                "--prefetch-verify-wait-ms",
                "0.0",
                "--prefetch-use-prefill-history",
                "true",
                "--prefetch-use-verify-history",
                "true",
                "--prefetch-use-draft-live",
                "true",
            ],
        ),
        Case(
            "spec_ablate_cache_lfu",
            [
                "--mode",
                "spec",
                "--spec-enable-prefetch",
                "true",
                "--prefetch-verify-wait-ms",
                str(prefetch_wait_ms),
                "--prefetch-use-prefill-history",
                "true",
                "--prefetch-use-verify-history",
                "true",
                "--prefetch-use-draft-live",
                "true",
                "--cache-strategy",
                "lfu",
            ],
        ),
    ]


def _extract_json(stdout: str) -> dict:
    lines = [x.strip() for x in stdout.splitlines() if x.strip()]
    for line in reversed(lines):
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise ValueError("no JSON payload found")


def _run_case(case: Case, common: list[str]) -> dict:
    cmd = [sys.executable, "examples/heterogeneous_benchmark_case.py", *common, *case.args]
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=False,
    )

    if proc.returncode != 0:
        return {
            "case": case.name,
            "status": "failed",
            "return_code": proc.returncode,
            "stderr_tail": "\n".join(proc.stderr.splitlines()[-40:]),
            "stdout_tail": "\n".join(proc.stdout.splitlines()[-40:]),
        }

    payload = _extract_json(proc.stdout)
    profile = payload.get("engine_profile", {})
    return {
        "case": case.name,
        "status": "ok",
        "payload": payload,
        "metrics": {name: profile.get(name, None) for name in REQUIRED_METRICS},
        "missing_metrics": [name for name in REQUIRED_METRICS if name not in profile],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run phase3 real-model cases in isolated subprocesses")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--num-seqs", type=int, default=2)
    parser.add_argument("--input-len", type=int, default=64)
    parser.add_argument("--output-len", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.999)
    parser.add_argument("--slots-per-layer", type=int, default=4)
    parser.add_argument("--prefetch-wait-ms", type=float, default=1.0)
    parser.add_argument("--base-dist-port", type=int, default=29700)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    common = [
        "--model-path",
        args.model_path,
        "--num-seqs",
        str(args.num_seqs),
        "--input-len",
        str(args.input_len),
        "--output-len",
        str(args.output_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--slots-per-layer",
        str(args.slots_per_layer),
        "--max-draft-tokens",
        "4",
        "--draft-top-c",
        "0",
        "--temperature",
        "0.0",
        "--engine-profile",
        "true",
        "--engine-profile-cuda-sync",
        "true",
        "--spec-profile",
        "true",
        "--enforce-eager",
        "false",
        "--return-token-ids",
        "true",
        "--return-text",
        "false",
        "--return-prompts",
        "false",
    ]

    rows = []
    cases = _build_cases(prefetch_wait_ms=args.prefetch_wait_ms)
    for idx, case in enumerate(cases):
        row = _run_case(
            case=Case(case.name, ["--dist-port", str(args.base_dist_port + idx), *case.args]),
            common=common,
        )
        rows.append(row)

    digest_ref_eager = None
    digest_ref_graph = None
    for row in rows:
        if row.get("status") == "ok" and row.get("case") == "standard_eager":
            digest_ref_eager = row["payload"].get("outputs_digest")
        if row.get("status") == "ok" and row.get("case") == "standard_graph":
            digest_ref_graph = row["payload"].get("outputs_digest")

    digest_match_vs_standard_eager = {}
    digest_match_vs_standard_graph = {}
    if digest_ref_eager is not None:
        for row in rows:
            if row.get("status") != "ok":
                continue
            digest_match_vs_standard_eager[row["case"]] = row["payload"].get("outputs_digest") == digest_ref_eager
    if digest_ref_graph is not None:
        for row in rows:
            if row.get("status") != "ok":
                continue
            digest_match_vs_standard_graph[row["case"]] = row["payload"].get("outputs_digest") == digest_ref_graph

    result = {
        "benchmark": "phase3_real_e2e_orchestrator",
        "required_metrics": REQUIRED_METRICS,
        "rows": rows,
        "digest_match_vs_standard_eager": digest_match_vs_standard_eager,
        "digest_match_vs_standard_graph": digest_match_vs_standard_graph,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "total_cases": len(rows),
        "ok_cases": sum(1 for x in rows if x.get("status") == "ok"),
        "failed_cases": [x.get("case") for x in rows if x.get("status") != "ok"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
