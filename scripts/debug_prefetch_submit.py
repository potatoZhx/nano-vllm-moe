#!/usr/bin/env python3
"""Minimal benchmark with debug instrumentation to trace prefetch submission gaps."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_SCRIPT = REPO_ROOT / "benchmarks" / "scripts" / "spec_verify_expert_count_stats.py"
MODEL_PATH = "/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B"
PROFILE = "results/reroute_impl_20260531/offline_profile_20260531_203257.safetensors"

PROMPT_TEXT = (
    "Expert caching for sparse mixture-of-experts inference is a practical systems problem. "
    "A serving engine usually keeps only part of the expert weights in GPU memory and leaves "
    "the rest in CPU memory. When routing selects an uncached expert, the engine must either "
    "compute on CPU or transfer weights before the layer needs them."
)


def main():
    output_dir = Path("results/debug_prefetch")
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = output_dir / "prompt.txt"
    prompt_file.write_text(PROMPT_TEXT + "\n")

    name = "debug_legacy_ratio25_l128_k2_vg1"
    case_json = output_dir / f"{name}.json"
    case_log = output_dir / f"{name}.log"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["NANOVLLM_PREFETCH_DEBUG"] = "1"

    cmd = [
        sys.executable, str(BENCH_SCRIPT),
        "--single-case",
        "--model-path", MODEL_PATH,
        "--prompt-text-file", str(prompt_file),
        "--cache-ratio", "0.25",
        "--slots-per-layer", "0",
        "--prefetch-enabled", "true",
        "--prefetch-runtime-mode", "draft_segment_indexed",
        "--prefetch-runtime-kind", "legacy",
        "--prefetch-verify-attention-ratio", "1.0",
        "--predictive-phase1-budget", "4",
        "--output", str(case_json),
        "--dist-port", "29500",
        "--num-seqs", "1",
        "--input-len", "1",
        "--output-len", "128",
        "--max-draft-tokens", "2",
        "--draft-top-c", "0",
        "--draft-reroute-policy", "entropy_cache_bias",
        "--draft-reroute-artifact", PROFILE,
        "--temperature", "0.8",
        "--acceptance-strategy", "standard_sampling",
        "--acceptance-threshold", "0.7",
        "--spec-verify-miss-policy", "cache_fill_no_cpu",
        "--cache-strategy", "lru",
        "--cpu-expert-backend", "fused",
        "--cpu-expert-pin-memory", "true",
        "--cpu-expert-workspace-max-routes", "327680",
        "--cpu-expert-packed-min-routes", "1",
        "--cpu-expert-parallel-mode", "serial",
        "--cpu-expert-num-threads", "4",
        "--kt-num-threads", "0",
        "--kt-threadpool-count", "1",
        "--kt-chunked-prefill-size", "4096",
        "--kt-direct-backend", "auto",
        "--kt-numa-nodes", "",
        "--kt-capture-bs", "1,2,4,8,16,32",
        "--cpu-gpu-parallel-execution-enabled", "auto",
        "--cpu-gpu-parallel-min-cpu-route-ratio", "0.0",
        "--max-num-batched-tokens", "16384",
        "--max-num-seqs", "1",
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.90",
        "--enforce-eager", "false",
        "--draft-cuda-graph-enabled", "true",
        "--draft-cuda-graph-cpu-backend", "none",
        "--verify-cuda-graph", "true",
        "--verify-cuda-graph-bucket-steps", "4,8,12,16",
        "--prefetch-verify-wait-ms", "0.0",
        "--prefetch-step-budget", "8",
        "--prefetch-max-inflight", "16",
        "--prefetch-staging-slots-per-layer", "2",
        "--cache-eviction-budget-per-step", "2",
        "--prefetch-global-queue-capacity", "4096",
        "--prefetch-use-prefill-history", "true",
        "--prefetch-use-verify-history", "true",
        "--prefetch-use-draft-live", "true",
        "--seed", "42",
        "--sync-layer-timing", "true",
        "--case-timeout-sec", "600",
    ]

    print(f"Running {name}...", flush=True)
    t0 = time.time()
    with case_log.open("w") as log_f:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, stdout=log_f, stderr=subprocess.STDOUT, text=True, timeout=600)
    elapsed = time.time() - t0
    print(f"Exit={proc.returncode} elapsed={elapsed:.1f}s", flush=True)

    if proc.returncode != 0:
        tail = case_log.read_text()[-3000:]
        print(f"FAILED:\n{tail}")
        return

    raw = json.loads(case_json.read_text())
    summary = raw.get("summary", {})
    cache = summary.get("cache", {})
    prefetch = summary.get("prefetch", {})
    verify_fill = summary.get("verify_cache_fill", {})
    cuda_graph = summary.get("cuda_graph", {})

    print(f"\n=== Results ===")
    print(f"verify_forward_ms_avg:    {summary.get('verify_forward_ms_avg', 0):.1f}")
    print(f"throughput_output_tok_s:  {summary.get('throughput_output_tok_s', 0):.3f}")
    print(f"true_route_hit_rate:      {cache.get('true_route_hit_rate', 0):.4f}")
    print(f"avg_miss_per_layer:       {cache.get('avg_miss_per_layer', 0):.2f}")
    print(f"avg_active_per_layer:     {cache.get('avg_active_per_layer', 0):.2f}")
    print(f"verify_graph_call_count:  {cuda_graph.get('verify_call_count', 0)}")
    print()
    print(f"--- Cache Fill ---")
    print(f"transfer_ms_total:        {verify_fill.get('transfer_ms_total', 0):.0f}")
    print(f"promoted_expert_count:    {verify_fill.get('promoted_expert_count', 0)}")
    print(f"evicted_expert_count:     {verify_fill.get('evicted_expert_count', 0)}")
    print()
    print(f"--- Prefetch ---")
    for k in sorted(prefetch.keys()):
        v = prefetch[k]
        if isinstance(v, float):
            print(f"  {k}: {v:.1f}" if abs(v) > 0.01 else f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")

    # Check the log for debug output
    log_text = case_log.read_text()
    debug_lines = [l for l in log_text.splitlines() if "PF_DEBUG" in l or "prefetch_debug" in l.lower()]
    if debug_lines:
        print(f"\n--- Debug lines ({len(debug_lines)}) ---")
        for dl in debug_lines[:50]:
            print(dl)


if __name__ == "__main__":
    main()
