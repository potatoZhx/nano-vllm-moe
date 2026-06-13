#!/usr/bin/env python3
"""Benchmark draft/verify segment-graph latency without prefetch transfers.

Measures raw forward latency when both draft and verify use segment-graph
CUDA graphs but no expert prefetch is submitted, so there is zero transfer
overlap or blocking.  Useful as a latency baseline for the prefetch-enabled
runs produced by ``bench_dual_queue_prefetch.py``.

Optionally caps per-layer CPU expert routes during verify via
``--verify-max-cpu-routes-per-layer``, which monkey-patches the plan builder
to discard excess CPU routes.  This creates a controlled baseline for
measuring how verify latency scales with bounded CPU work per MoE layer.

Example:

    conda activate nano_moe
    cd /home/linke/nano-vllm-moe
    python scripts/bench_segment_graph_no_prefetch.py \
        --output-dir results/segment_graph_no_prefetch \
        --gpu-memory-utilization 0.99 \
        --cache-ratios 0.25,0.3125 \
        --output-lens 128,512 \
        --max-draft-tokens-values 4,8 \
        --segment-sizes 12 \
        --verify-max-cpu-routes-per-layer 4,8,16
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROMPT_TEXT = (
    "Sparse mixture-of-experts inference keeps only part of each layer's expert weights "
    "in GPU memory. Explain how speculative decoding can overlap expert prefetch with "
    "draft and verify segment computation while preserving exact verification semantics. "
    "Discuss routing-score metadata, bounded transfer budgets, cache eviction protection, "
    "and why late best-effort transfers should be discarded instead of blocking compute."
)

MODEL_PATH = "/data1/models/Qwen3-30B-A3B"


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {value}")


def _parse_csv(values: str, cast) -> list:
    return [cast(item.strip()) for item in values.split(",") if item.strip()]


def build_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    cpu_caps = _parse_csv(args.verify_max_cpu_routes_per_layer, int)
    if not cpu_caps:
        cpu_caps = [0]  # 0 means "no cap"
    cases: list[dict[str, Any]] = []
    for output_len in _parse_csv(args.output_lens, int):
        for cache_ratio in _parse_csv(args.cache_ratios, float):
            for max_draft_tokens in _parse_csv(args.max_draft_tokens_values, int):
                for segment_size in _parse_csv(args.segment_sizes, int):
                    for cpu_cap in cpu_caps:
                        for repeat in range(int(args.repeats)):
                            cases.append(
                                {
                                    "output_len": int(output_len),
                                    "cache_ratio": float(cache_ratio),
                                    "max_draft_tokens": int(max_draft_tokens),
                                    "segment_size": int(segment_size),
                                    "verify_max_cpu_routes_per_layer": int(cpu_cap),
                                    "repeat": int(repeat),
                                }
                            )
    return cases


def _case_name(case: dict[str, Any]) -> str:
    ratio_pct = int(round(float(case["cache_ratio"]) * 10000))
    cpu_cap = int(case.get("verify_max_cpu_routes_per_layer", 0) or 0)
    return (
        f"noseg_seg{int(case['segment_size'])}_"
        f"ratio{ratio_pct:04d}_l{int(case['output_len'])}_"
        f"k{int(case['max_draft_tokens'])}_"
        f"cpucap{int(cpu_cap)}_r{int(case['repeat'])}"
    )


def _row_from_raw(
    case: dict[str, Any],
    raw: dict[str, Any],
    wall_elapsed_sec: float,
) -> dict[str, Any]:
    summary = raw.get("summary", {})
    acceptance = summary.get("acceptance", {})
    cache = summary.get("cache", {})
    cuda_graph = summary.get("cuda_graph", {})

    return {
        "name": _case_name(case),
        "output_len": int(case["output_len"]),
        "cache_ratio": float(case["cache_ratio"]),
        "max_draft_tokens": int(case["max_draft_tokens"]),
        "segment_size": int(case["segment_size"]),
        "verify_max_cpu_routes_per_layer": int(case.get("verify_max_cpu_routes_per_layer", 0) or 0),
        "repeat": int(case["repeat"]),
        "wall_elapsed_sec": float(wall_elapsed_sec),
        "generated_output_tokens": int(raw.get("generated_output_tokens", 0) or 0),
        "throughput_output_tok_s": float(summary.get("throughput_output_tok_s", 0.0) or 0.0),
        "decode_phase_output_tok_s": float(summary.get("decode_phase_output_tok_s", 0.0) or 0.0),
        "draft_forward_ms_avg": float(summary.get("draft_forward_ms_avg", 0.0) or 0.0),
        "verify_forward_ms_avg": float(summary.get("verify_forward_ms_avg", 0.0) or 0.0),
        "acceptance_rate": float(acceptance.get("acceptance_rate", 0.0) or 0.0),
        "route_hit_rate": float(
            cache.get("true_route_hit_rate", cache.get("route_hit_rate", 0.0)) or 0.0
        ),
        "avg_miss_routes_per_layer": float(cache.get("avg_miss_per_layer", 0.0) or 0.0),
        "avg_active_per_layer": float(cache.get("avg_active_per_layer", 0.0) or 0.0),
        "verify_calls": int(cuda_graph.get("verify_call_count", 0) or 0),
        "outputs_digest": str(raw.get("outputs_digest", "")),
    }


def _command(
    args: argparse.Namespace,
    repo_root: Path,
    prompt_file: Path,
    case: dict[str, Any],
    output_path: Path,
    case_index: int,
) -> list[str]:
    single_case_script = (
        repo_root / "benchmarks" / "scripts" / "spec_verify_expert_count_stats.py"
    )
    segment_size = int(case["segment_size"])
    return [
        sys.executable,
        str(single_case_script),
        "--single-case",
        "--model-path",
        args.model_path,
        "--prompt-text-file",
        str(prompt_file),
        "--output",
        str(output_path),
        "--dist-port",
        str(args.dist_port_base + case_index),
        "--cache-ratio",
        str(case["cache_ratio"]),
        "--slots-per-layer",
        "0",
        "--num-seqs",
        "1",
        "--input-len",
        "1",
        "--output-len",
        str(case["output_len"]),
        "--max-draft-tokens",
        str(case["max_draft_tokens"]),
        "--draft-top-c",
        "0",
        "--draft-reroute-policy",
        args.draft_reroute_policy,
        "--draft-reroute-artifact",
        args.profile_artifact,
        "--temperature",
        str(args.temperature),
        "--acceptance-strategy",
        args.acceptance_strategy,
        "--acceptance-threshold",
        str(args.acceptance_threshold),
        # -- disable prefetch entirely --
        "--prefetch-enabled",
        "false",
        # -- both draft and verify use segment graphs --
        "--draft-cuda-graph-enabled",
        "true",
        "--draft-cuda-graph-cpu-backend",
        "none",
        "--draft-prefetch-segment-size",
        str(segment_size),
        "--draft-prefetch-segment-host-buffer-pool-size",
        "0",
        "--draft-prefetch-visible-budget-ms",
        "0",
        "--draft-prefetch-min-per-boundary",
        "0",
        "--draft-prefetch-max-per-boundary",
        "0",
        "--verify-cuda-graph",
        "true",
        "--verify-cuda-graph-bucket-steps",
        args.verify_cuda_graph_bucket_steps,
        "--verify-prefetch-segment-size",
        str(segment_size),
        "--verify-prefetch-visible-budget-ms",
        "0",
        "--verify-prefetch-min-per-boundary",
        "0",
        "--verify-prefetch-max-per-boundary",
        "0",
        "--spec-verify-miss-policy",
        "cpu",
        "--cache-strategy",
        args.cache_strategy,
        "--cpu-expert-execution-enabled",
        "true",
        "--cpu-expert-backend",
        "kt_direct",
        "--cpu-expert-pin-memory",
        "true",
        "--cpu-expert-workspace-max-routes",
        str(args.cpu_expert_workspace_max_routes),
        "--cpu-expert-packed-min-routes",
        "1",
        "--cpu-expert-parallel-mode",
        "serial",
        "--cpu-expert-num-threads",
        str(args.cpu_expert_num_threads),
        "--kt-num-threads",
        str(args.kt_num_threads),
        "--kt-threadpool-count",
        str(args.kt_threadpool_count),
        "--kt-chunked-prefill-size",
        str(args.kt_chunked_prefill_size),
        "--kt-direct-backend",
        args.kt_direct_backend,
        "--kt-numa-nodes",
        args.kt_numa_nodes,
        "--kt-capture-bs",
        args.kt_capture_bs,
        "--cpu-gpu-parallel-execution-enabled",
        "auto",
        "--cpu-gpu-parallel-min-cpu-route-ratio",
        "0.0",
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        "1",
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--enforce-eager",
        "false",
        "--seed",
        str(args.seed),
        "--sync-layer-timing",
        str(args.sync_layer_timing).lower(),
        "--verify-max-cpu-routes-per-layer",
        str(case["verify_max_cpu_routes_per_layer"]),
    ]


def run_case(
    args: argparse.Namespace,
    repo_root: Path,
    prompt_file: Path,
    case: dict[str, Any],
    case_index: int,
) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    name = _case_name(case)
    case_json = output_dir / f"{name}.json"
    case_log = output_dir / f"{name}.log"

    if args.skip_existing and case_json.exists():
        raw = json.loads(case_json.read_text(encoding="utf-8"))
        return _row_from_raw(case, raw, float(raw.get("elapsed_sec", 0.0) or 0.0))

    cmd = _command(args, repo_root, prompt_file, case, case_json, case_index)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    print(f"[{case_index + 1}] running {name}", flush=True)
    started = time.time()
    with case_log.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            cmd,
            cwd=repo_root,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.case_timeout_sec,
        )
    wall_elapsed = time.time() - started
    print(
        f"[{case_index + 1}] {name} exit={process.returncode} "
        f"elapsed={wall_elapsed:.1f}s",
        flush=True,
    )
    if process.returncode != 0:
        tail = case_log.read_text(encoding="utf-8", errors="replace")[-5000:]
        raise RuntimeError(f"case failed: {name}\n{tail}")

    raw = json.loads(case_json.read_text(encoding="utf-8"))
    row = _row_from_raw(case, raw, wall_elapsed)
    cpu_cap = int(row.get("verify_max_cpu_routes_per_layer", 0) or 0)
    cap_str = f" cpu_cap={cpu_cap}" if cpu_cap > 0 else ""
    print(
        f"  tok/s={row['throughput_output_tok_s']:.3f} "
        f"decode_tok/s={row['decode_phase_output_tok_s']:.3f} "
        f"draft_ms={row['draft_forward_ms_avg']:.3f} "
        f"verify_ms={row['verify_forward_ms_avg']:.3f} "
        f"hit={row['route_hit_rate']:.4f} accept={row['acceptance_rate']:.4f} "
        f"miss/L={row['avg_miss_routes_per_layer']:.2f} "
        f"active/L={row['avg_active_per_layer']:.2f}{cap_str}",
        flush=True,
    )
    return row


def write_markdown_report(summary: dict[str, Any], path: Path) -> None:
    metadata = summary["metadata"]
    rows = summary["rows"]
    lines = [
        "# Segment-Graph Latency Baseline (No Prefetch)",
        "",
        f"- timestamp: `{metadata['timestamp']}`",
        f"- model: `{metadata['model_path']}`",
        f"- segment sizes: `{', '.join(str(x) for x in metadata['segment_sizes'])}`",
        f"- output directory: `{metadata['output_dir']}`",
        "",
        "## Cases",
        "",
        "| seg | out | ratio | K | cpu_cap | rep | tok/s | decode tok/s | draft ms | verify ms | hit | accept | miss/L | active/L |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        cpu_cap = int(row.get("verify_max_cpu_routes_per_layer", 0) or 0)
        lines.append(
            "| "
            f"{row['segment_size']} | {row['output_len']} | "
            f"{row['cache_ratio']:.4f} | {row['max_draft_tokens']} | "
            f"{cpu_cap} | {row['repeat']} | "
            f"{row['throughput_output_tok_s']:.3f} | "
            f"{row['decode_phase_output_tok_s']:.3f} | "
            f"{row['draft_forward_ms_avg']:.3f} | {row['verify_forward_ms_avg']:.3f} | "
            f"{row['route_hit_rate']:.4f} | {row['acceptance_rate']:.4f} | "
            f"{row['avg_miss_routes_per_layer']:.2f} | "
            f"{row['avg_active_per_layer']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Prefetch is **disabled** — no H2D transfers are submitted, so forward "
            "latency is measured without any transfer overlap or blocking.",
            "- Both draft and verify use per-segment CUDA graphs (segment size = `-s`).",
            "- `draft_ms` and `verify_ms` are the mean forward latencies per call.",
            "- `cpu_cap` > 0 means verify CPU routes per layer were capped at that count "
            "(excess routes discarded); 0 means no cap.",
            "- Compare these numbers against prefetch-enabled runs to quantify how much "
            "transfer scheduling adds to (or hides behind) forward time.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = output_dir / "noseg_prompt.txt"
    prompt_file.write_text(PROMPT_TEXT + "\n", encoding="utf-8")

    cases = build_cases(args)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        try:
            print("=" * 80)
            rows.append(run_case(args, repo_root, prompt_file, case, case_index))
        except Exception as error:
            failures.append({"case": case, "error": str(error)})
            if args.fail_fast:
                raise

    summary = {
        "metadata": {
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "model_path": args.model_path,
            "profile_artifact": args.profile_artifact,
            "output_dir": str(output_dir),
            "segment_sizes": _parse_csv(args.segment_sizes, int),
            "cache_ratios": _parse_csv(args.cache_ratios, float),
            "output_lens": _parse_csv(args.output_lens, int),
            "max_draft_tokens_values": _parse_csv(args.max_draft_tokens_values, int),
            "verify_max_cpu_routes_per_layer": _parse_csv(args.verify_max_cpu_routes_per_layer, int),
            "repeats": int(args.repeats),
            "argv": sys.argv,
        },
        "rows": rows,
        "failures": failures,
    }
    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown_report(summary, summary_md)
    if args.report_doc:
        write_markdown_report(summary, Path(args.report_doc))
    print(f"summary_json={summary_json}")
    print(f"summary_md={summary_md}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark draft/verify segment-graph latency without prefetch."
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--profile-artifact", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-doc", default="")
    parser.add_argument("--output-lens", default="128,512")
    parser.add_argument("--cache-ratios", default="0.25,0.3125,0.50")
    parser.add_argument("--max-draft-tokens-values", default="4,8")
    parser.add_argument("--segment-sizes", default="12")
    parser.add_argument(
        "--verify-max-cpu-routes-per-layer", default="0",
        help="CSV of per-layer CPU route caps for verify (0 = no cap). "
             "Each value spawns its own sweep dimension."
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--cache-strategy", default="lru")
    parser.add_argument("--draft-reroute-policy", default="entropy_cache_bias")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--acceptance-strategy", default="standard_sampling")
    parser.add_argument("--acceptance-threshold", type=float, default=0.7)

    parser.add_argument("--cpu-expert-workspace-max-routes", type=int, default=327680)
    parser.add_argument("--cpu-expert-num-threads", type=int, default=4)
    parser.add_argument("--kt-num-threads", type=int, default=0)
    parser.add_argument("--kt-threadpool-count", type=int, default=1)
    parser.add_argument("--kt-chunked-prefill-size", type=int, default=4096)
    parser.add_argument(
        "--kt-direct-backend",
        choices=["auto", "amx_bf16", "avx2_bf16"],
        default="auto",
    )
    parser.add_argument("--kt-numa-nodes", default="")
    parser.add_argument("--kt-capture-bs", default="1,2,4,8,16,32")

    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--verify-cuda-graph-bucket-steps", default="3,5,8,12")
    parser.add_argument("--dist-port-base", type=int, default=30500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sync-layer-timing", type=str2bool, default=False)
    parser.add_argument("--case-timeout-sec", type=int, default=2400)
    parser.add_argument("--skip-existing", type=str2bool, default=True)
    parser.add_argument("--fail-fast", type=str2bool, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    print("running with arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    run(args)


if __name__ == "__main__":
    main()
