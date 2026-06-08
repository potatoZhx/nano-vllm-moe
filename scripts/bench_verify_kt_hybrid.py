#!/usr/bin/env python3
"""Benchmark verify CUDA graph kt_hybrid path.

Compares verify performance with kt_hybrid graph on vs off under
--spec-verify-miss-policy cpu --cpu-expert-backend kt_direct.

Usage example (slurm):

    TS=$(date +%Y%m%d_%H%M%S)
    LOG=/home/mumura/moe_spec/logs/bench_kt_hybrid_${TS}.log
    srun --jobid=30169 --ntasks=1 bash -s <<EOS 2>&1 | tee "$LOG"
    source /opt/Software/Anaconda3/etc/profile.d/conda.sh
    conda activate nano_moe
    cd /home/mumura/moe_spec/nano-vllm-moe/
    python scripts/bench_verify_kt_hybrid.py \
        --gpu-memory-utilization 0.99 \
        --output-dir results/kt_hybrid_bench \
        --cache-ratios 0.25,0.3125 \
        --output-lens 128 \
        --max-draft-tokens-values 3
    EOS
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
    "Expert caching for sparse mixture-of-experts inference is a practical systems problem. "
    "A serving engine usually keeps only part of the expert weights in GPU memory and leaves "
    "the rest in CPU memory. When routing selects an uncached expert, the engine must either "
    "compute on CPU or transfer weights before the layer needs them. Speculative decoding adds "
    "another constraint: draft routing can hint at future expert demand, but inaccurate draft "
    "signals must not change the verified model output. Explain how a conservative predictive "
    "prefetcher can use draft routing, prefill/verify access counts, and a limited transfer "
    "budget while preserving exact verification semantics."
)

DEFAULT_PROFILE = "results/reroute_impl_20260531/offline_profile_20260531_203257.safetensors"
MODEL_PATH = "/data1/models/Qwen3-30B-A3B"


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def _parse_csv(values: str, cast) -> list:
    return [cast(x.strip()) for x in values.split(",") if x.strip()]


def build_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    mdt_values = _parse_csv(args.max_draft_tokens_values, int) if args.max_draft_tokens_values else [8]
    for output_len in _parse_csv(args.output_lens, int):
        for ratio in _parse_csv(args.cache_ratios, float):
            for mdt in mdt_values:
                for vg in [True, False]:
                    cases.append(
                        {
                            "output_len": int(output_len),
                            "cache_ratio": float(ratio),
                            "max_draft_tokens": int(mdt),
                            "verify_cuda_graph": bool(vg),
                            "cache_strategy": args.cache_strategy,
                        }
                    )
    return cases


def _case_name(case: dict[str, Any]) -> str:
    ratio_pct = int(round(float(case["cache_ratio"]) * 100))
    vg = "kt_hybrid" if case.get("verify_cuda_graph") else "eager"
    return (
        f"kt_{vg}_ratio{ratio_pct}_"
        f"l{int(case['output_len'])}_k{int(case['max_draft_tokens'])}"
    )


def _row_from_raw(case: dict[str, Any], raw: dict[str, Any], elapsed_sec: float) -> dict[str, Any]:
    summary = raw.get("summary", {})
    acceptance = summary.get("acceptance", {})
    cache = summary.get("cache", {})
    cuda_graph = summary.get("cuda_graph", {})
    return {
        "name": _case_name(case),
        "output_len": int(case["output_len"]),
        "cache_ratio": float(case["cache_ratio"]),
        "max_draft_tokens": int(case["max_draft_tokens"]),
        "verify_cuda_graph": bool(case["verify_cuda_graph"]),
        "elapsed_sec": float(elapsed_sec),
        "generated_output_tokens": int(raw.get("generated_output_tokens", 0) or 0),
        "acceptance_rate": float(acceptance.get("acceptance_rate", 0.0) or 0.0),
        "route_hit_rate": float(cache.get("true_route_hit_rate", cache.get("route_hit_rate", 0.0)) or 0.0),
        "avg_miss_per_layer": float(cache.get("avg_miss_per_layer", 0.0) or 0.0),
        "avg_active_per_layer": float(cache.get("avg_active_per_layer", 0.0) or 0.0),
        "throughput_output_tok_s": float(summary.get("throughput_output_tok_s", 0.0) or 0.0),
        "decode_phase_output_tok_s": float(summary.get("decode_phase_output_tok_s", 0.0) or 0.0),
        "draft_forward_ms_avg": float(summary.get("draft_forward_ms_avg", 0.0) or 0.0),
        "verify_forward_ms_avg": float(summary.get("verify_forward_ms_avg", 0.0) or 0.0),
        "verify_call_count": int(cuda_graph.get("verify_call_count", 0) or 0),
        "verify_kt_hybrid_replay_count": int(cuda_graph.get("verify_kt_hybrid_replay_count", 0) or 0),
        "verify_prefix_replay_count": int(cuda_graph.get("verify_prefix_replay_count", 0) or 0),
        "draft_graph_replay_count": int(cuda_graph.get("draft_replay_count", 0) or 0),
        "graph_hit_rate": float(cuda_graph.get("hit_rate", 0.0) or 0.0),
        "outputs_digest": str(raw.get("outputs_digest", "")),
    }


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
    script_path = repo_root / "benchmarks" / "scripts" / "spec_verify_expert_count_stats.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        str(script_path),
        "--single-case",
        "--model-path", args.model_path,
        "--prompt-text-file", str(prompt_file),
        "--cache-ratio", str(case["cache_ratio"]),
        "--slots-per-layer", "0",
        "--prefetch-enabled", "true",
        "--prefetch-runtime-mode", "draft_segment_indexed",
        "--prefetch-runtime-kind", args.prefetch_runtime_kind,
        "--prefetch-verify-attention-ratio", str(args.prefetch_verify_attention_ratio),
        "--output", str(case_json),
        "--dist-port", str(args.dist_port_base + case_index),
        "--num-seqs", "1",
        "--input-len", "1",
        "--output-len", str(case["output_len"]),
        "--max-draft-tokens", str(case["max_draft_tokens"]),
        "--draft-top-c", "0",
        "--draft-reroute-policy", "entropy_cache_bias",
        "--draft-reroute-artifact", args.profile_artifact,
        "--temperature", str(args.temperature),
        "--acceptance-strategy", args.acceptance_strategy,
        "--acceptance-threshold", str(args.acceptance_threshold),
        "--spec-verify-miss-policy", "cpu",
        "--cache-strategy", str(case.get("cache_strategy", "lru")),
        "--cpu-expert-backend", "kt_direct",
        "--cpu-expert-pin-memory", "true",
        "--cpu-expert-workspace-max-routes", str(args.cpu_expert_workspace_max_routes),
        "--cpu-expert-packed-min-routes", "1",
        "--cpu-expert-parallel-mode", "serial",
        "--cpu-expert-num-threads", str(args.cpu_expert_num_threads),
        "--kt-num-threads", str(args.kt_num_threads),
        "--kt-threadpool-count", str(args.kt_threadpool_count),
        "--kt-chunked-prefill-size", str(args.kt_chunked_prefill_size),
        "--kt-direct-backend", args.kt_direct_backend,
        "--kt-numa-nodes", args.kt_numa_nodes,
        "--kt-capture-bs", args.kt_capture_bs,
        "--cpu-gpu-parallel-execution-enabled", "auto",
        "--cpu-gpu-parallel-min-cpu-route-ratio", "0.0",
        "--max-num-batched-tokens", str(args.max_num_batched_tokens),
        "--max-num-seqs", "1",
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--enforce-eager", "false",
        "--draft-cuda-graph-enabled", "true",
        "--draft-cuda-graph-cpu-backend", "none",
        "--verify-cuda-graph", str(case["verify_cuda_graph"]).lower(),
        "--verify-cuda-graph-bucket-steps", args.verify_cuda_graph_bucket_steps,
        "--prefetch-step-budget", str(args.prefetch_step_budget),
        "--prefetch-verify-layer-max-budget", str(args.prefetch_verify_layer_max_budget),
        "--prefetch-max-inflight", str(args.prefetch_max_inflight),
        "--prefetch-staging-slots-per-layer", str(args.prefetch_staging_slots_per_layer),
        "--cache-eviction-budget-per-step", str(args.cache_eviction_budget_per_step),
        "--prefetch-global-queue-capacity", str(args.prefetch_global_queue_capacity),
        "--prefetch-use-prefill-history", "true",
        "--prefetch-use-verify-history", "true",
        "--prefetch-use-draft-live", "true",
        "--seed", str(args.seed),
        "--sync-layer-timing", str(args.sync_layer_timing).lower(),
    ]

    print(f"[{case_index + 1}] running {name}", flush=True)
    t0 = time.time()
    with case_log.open("w", encoding="utf-8") as log_f:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.case_timeout_sec,
        )
    elapsed = time.time() - t0
    print(f"[{case_index + 1}] {name} exit={proc.returncode} elapsed={elapsed:.1f}s", flush=True)
    if proc.returncode != 0:
        tail = case_log.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"case failed: {name}\n{tail}")

    raw = json.loads(case_json.read_text(encoding="utf-8"))
    row = _row_from_raw(case, raw, elapsed)
    print(
        f"  accept={row['acceptance_rate']:.4f} hit={row['route_hit_rate']:.4f} "
        f"miss/layer={row['avg_miss_per_layer']:.2f} active/layer={row['avg_active_per_layer']:.2f} "
        f"tok/s={row['throughput_output_tok_s']:.3f} verify_ms={row['verify_forward_ms_avg']:.3f} "
        f"kt_hybrid_replay={row['verify_kt_hybrid_replay_count']} "
        f"graph={'on' if row['verify_cuda_graph'] else 'off'}",
        flush=True,
    )
    return row


def _delta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple, dict[bool, dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["output_len"]), round(float(row["cache_ratio"]), 4), int(row["max_draft_tokens"]))
        vg = bool(row.get("verify_cuda_graph", False))
        by_key.setdefault(key, {})[vg] = row

    out: list[dict[str, Any]] = []
    for key, pair in by_key.items():
        row_on = pair.get(True)
        row_off = pair.get(False)
        if row_on is None or row_off is None:
            continue
        out.append(
            {
                "output_len": int(row_on["output_len"]),
                "cache_ratio": float(row_on["cache_ratio"]),
                "max_draft_tokens": int(row_on["max_draft_tokens"]),
                "acceptance_delta": float(row_on["acceptance_rate"]) - float(row_off["acceptance_rate"]),
                "route_hit_delta": float(row_on["route_hit_rate"]) - float(row_off["route_hit_rate"]),
                "miss_per_layer_delta": float(row_on["avg_miss_per_layer"]) - float(row_off["avg_miss_per_layer"]),
                "throughput_delta": float(row_on["throughput_output_tok_s"]) - float(row_off["throughput_output_tok_s"]),
                "verify_ms_delta": float(row_on["verify_forward_ms_avg"]) - float(row_off["verify_forward_ms_avg"]),
                "draft_ms_delta": float(row_on["draft_forward_ms_avg"]) - float(row_off["draft_forward_ms_avg"]),
            }
        )
    return out


def write_markdown_report(
    *,
    summary: dict[str, Any],
    path: Path,
    status: str,
) -> None:
    rows = summary.get("rows", [])
    deltas = summary.get("deltas", [])
    metadata = summary.get("metadata", {})
    lines = [
        "# Verify CUDA Graph kt_hybrid Benchmark Report",
        "",
        f"Date: {time.strftime('%Y-%m-%d')}",
        "",
        "## 1. Experiment Design",
        "",
        "Compare speculative decoding verify performance with kt_hybrid CUDA graph **on** vs **off**.",
        "",
        "### Fixed settings",
        "",
        "```",
        f"Model:                    {metadata.get('model_path', '')}",
        f"Profile:                  {metadata.get('profile_artifact', '')}",
        "spec_verify_miss_policy:  cpu",
        "cpu_expert_backend:       kt_direct",
        f"kt_direct_backend:        {metadata.get('kt_direct_backend', '')}",
        f"kt threads/pools:         {metadata.get('kt_num_threads', '')}/{metadata.get('kt_threadpool_count', '')}",
        f"kt capture_bs:            {metadata.get('kt_capture_bs', '')}",
        f"verify graph buckets:     {', '.join(str(x) for x in metadata.get('verify_cuda_graph_bucket_steps', []))}",
        "```",
        "",
        "### Sweep dimensions",
        "",
        "| Dimension | Values |",
        "|---|---|",
        f"| Cache ratios | `{', '.join(str(x) for x in metadata.get('cache_ratios', []))}` |",
        f"| Output lengths | `{', '.join(str(x) for x in metadata.get('output_lens', []))}` |",
        f"| max_draft_tokens | `{', '.join(str(x) for x in metadata.get('max_draft_tokens_values', []))}` |",
        "| verify_cuda_graph | `true, false` |",
        "",
        "---",
        "",
        "## 2. Results",
        "",
        "| mode | out | ratio | K | accept | hit | miss/layer | active/layer | output tok/s | draft ms | verify ms | kt_hybrid replay | verify calls | graph hit rate |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        mode = "kt_hybrid" if row["verify_cuda_graph"] else "eager"
        lines.append(
            "| "
            f"{mode} | "
            f"{row['output_len']} | "
            f"{row['cache_ratio']:.2f} | "
            f"{row['max_draft_tokens']} | "
            f"{row['acceptance_rate']:.4f} | "
            f"{row['route_hit_rate']:.4f} | "
            f"{row['avg_miss_per_layer']:.2f} | "
            f"{row['avg_active_per_layer']:.2f} | "
            f"{row['throughput_output_tok_s']:.3f} | "
            f"{row['draft_forward_ms_avg']:.3f} | "
            f"{row['verify_forward_ms_avg']:.3f} | "
            f"{row['verify_kt_hybrid_replay_count']} | "
            f"{row['verify_call_count']} | "
            f"{row['graph_hit_rate']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 3. kt_hybrid Graph Delta (On vs Off)",
            "",
            "Positive throughput delta = kt_hybrid improves throughput. "
            "Negative verify ms delta = kt_hybrid reduces verify latency.",
            "",
            "| out | ratio | K | accept delta | hit delta | miss/layer delta | tok/s delta | verify ms delta | draft ms delta |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in deltas:
        lines.append(
            "| "
            f"{row['output_len']} | "
            f"{row['cache_ratio']:.2f} | "
            f"{row['max_draft_tokens']} | "
            f"{row['acceptance_delta']:+.4f} | "
            f"{row['route_hit_delta']:+.4f} | "
            f"{row['miss_per_layer_delta']:+.2f} | "
            f"{row['throughput_delta']:+.3f} | "
            f"{row['verify_ms_delta']:+.3f} | "
            f"{row['draft_ms_delta']:+.3f} |"
        )

    lines.extend(["", f"Status: `{status}`", ""])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = output_dir / "meaningful_prompt.txt"
    prompt_file.write_text(PROMPT_TEXT + "\n", encoding="utf-8")

    cases = build_cases(args)
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        rows.append(run_case(args, repo_root, prompt_file, case, index))

    metadata = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "model_path": args.model_path,
        "profile_artifact": args.profile_artifact,
        "output_dir": str(output_dir),
        "cache_ratios": sorted({float(c["cache_ratio"]) for c in cases}),
        "output_lens": sorted({int(c["output_len"]) for c in cases}),
        "max_draft_tokens_values": sorted({int(c["max_draft_tokens"]) for c in cases}),
        "kt_num_threads": int(args.kt_num_threads),
        "kt_threadpool_count": int(args.kt_threadpool_count),
        "kt_direct_backend": args.kt_direct_backend,
        "kt_capture_bs": args.kt_capture_bs,
        "verify_cuda_graph_bucket_steps": _parse_csv(args.verify_cuda_graph_bucket_steps, int),
        "max_model_len": int(args.max_model_len),
        "argv": sys.argv,
    }
    summary = {
        "metadata": metadata,
        "rows": rows,
        "deltas": _delta_rows(rows),
    }
    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"
    summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    write_markdown_report(summary=summary, path=summary_md, status="completed")
    if args.report_doc:
        write_markdown_report(summary=summary, path=Path(args.report_doc), status="completed")
    print(f"summary_json={summary_json}")
    print(f"summary_md={summary_md}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark verify CUDA graph kt_hybrid (GPU cached + kt_direct miss) path."
    )
    p.add_argument("--model-path", default=MODEL_PATH)
    p.add_argument("--profile-artifact", default=DEFAULT_PROFILE)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--report-doc", default="")
    p.add_argument("--output-lens", default="128,256")
    p.add_argument("--cache-ratios", default="0.25,0.50,0.75")
    p.add_argument("--cache-strategy", default="lru")
    p.add_argument("--max-draft-tokens-values", default="4,8")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--acceptance-strategy", default="standard_sampling")
    p.add_argument("--acceptance-threshold", type=float, default=0.7)
    p.add_argument("--cpu-expert-workspace-max-routes", type=int, default=327680)
    p.add_argument("--cpu-expert-num-threads", type=int, default=4)
    p.add_argument("--kt-num-threads", type=int, default=0)
    p.add_argument("--kt-threadpool-count", type=int, default=1)
    p.add_argument("--kt-chunked-prefill-size", type=int, default=4096)
    p.add_argument("--kt-direct-backend", choices=["auto", "amx_bf16", "avx2_bf16"], default="auto")
    p.add_argument("--kt-numa-nodes", default="")
    p.add_argument("--kt-capture-bs", default="1,2,4,8,16,32")
    p.add_argument("--max-num-batched-tokens", type=int, default=16384)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--verify-cuda-graph-bucket-steps", default="4,8,12,16")
    p.add_argument("--prefetch-runtime-kind", default="legacy")
    p.add_argument("--prefetch-verify-attention-ratio", type=float, default=1.0)
    p.add_argument("--prefetch-step-budget", type=int, default=8)
    p.add_argument("--prefetch-verify-layer-max-budget", type=int, default=8)
    p.add_argument("--prefetch-max-inflight", type=int, default=16)
    p.add_argument("--prefetch-staging-slots-per-layer", type=int, default=2)
    p.add_argument("--cache-eviction-budget-per-step", type=int, default=2)
    p.add_argument("--prefetch-global-queue-capacity", type=int, default=4096)
    p.add_argument("--dist-port-base", type=int, default=29500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sync-layer-timing", type=str2bool, default=True)
    p.add_argument("--case-timeout-sec", type=int, default=2400)
    return p


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
