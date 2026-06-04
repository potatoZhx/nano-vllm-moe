#!/usr/bin/env python3
"""Test script for verify CUDA graph prefix optimization.

Runs baseline (eager) and graph-enabled verify paths, compares latency and
numerical correctness.  Outputs per-case JSON and an optional comparison table.

Usage:
    # Graph case only
    python scripts/verify_profile/test_verify_cuda_graph.py \
        --model-path /path/to/Qwen3-30B-A3B --verify-cuda-graph-on

    # Auto baseline + graph comparison
    python scripts/verify_profile/test_verify_cuda_graph.py \
        --model-path /path/to/Qwen3-30B-A3B --compare-baseline
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig

from nanovllm import LLM, SamplingParams


DEFAULT_MODEL_PATH = "/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B"


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _summary_from_profile(profile: dict[str, Any]) -> dict[str, Any]:
    verify_calls = float(profile.get("spec_run_verify_calls", 0.0) or 0.0)
    return {
        "verify_calls": verify_calls,
        "verify_forward_ms_avg": float(profile.get("verify_forward_ms", 0.0) or 0.0),
        "verify_forward_ms_total": float(profile.get("spec_run_verify_infer_ms_total", 0.0) or 0.0),
        "run_verify_total_ms_avg": _safe_div(
            float(profile.get("model_run_verify_total_ms", 0.0) or 0.0), verify_calls,
        ),
        "verify_tokens_per_call": _safe_div(
            float(profile.get("model_verify_tokens_in_total", 0.0) or 0.0), verify_calls,
        ),
        "verify_plan_ms_total": float(profile.get("model_verify_plan_ms", 0.0) or 0.0),
        "verify_gpu_compute_ms_total": float(profile.get("model_verify_gpu_compute_ms", 0.0) or 0.0),
        "verify_route_ms_total": float(profile.get("model_verify_route_ms", 0.0) or 0.0),
        "verify_gpu_gather_ms_total": float(profile.get("model_verify_gpu_gather_ms", 0.0) or 0.0),
        "verify_scatter_ms_total": float(profile.get("model_verify_scatter_ms", 0.0) or 0.0),
    }


def run_single_case(
    args: argparse.Namespace,
    case_name: str,
    verify_cuda_graph: bool,
) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    hf_config = AutoConfig.from_pretrained(args.model_path)
    num_experts = int(getattr(hf_config, "num_experts"))
    slots = int(round(num_experts * float(args.cache_ratio)))

    prompt_text = "Please explain the concept of mixture-of-experts models in deep learning."

    llm: LLM | None = None
    try:
        llm = LLM(
            args.model_path,
            dist_port=args.dist_port,
            enforce_eager=False,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=1,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            inference_mode="spec",
            enable_heterogeneous=True,
            enable_speculative=True,
            heterogeneous_slots_per_layer=slots,
            max_draft_tokens=args.max_draft_tokens,
            draft_top_c=0,
            draft_reroute_policy="entropy_cache_bias",
            acceptance_strategy="standard_sampling",
            acceptance_threshold=0.7,
            cpu_expert_execution_enabled=True,
            cpu_expert_pin_memory=True,
            cpu_expert_backend="fused",
            cpu_expert_workspace_max_routes=8192,
            cpu_expert_packed_min_routes=1,
            cpu_expert_parallel_mode="serial",
            cpu_expert_num_threads=4,
            cpu_gpu_parallel_execution_enabled="auto",
            cpu_gpu_parallel_min_cpu_route_ratio=0.0,
            spec_verify_miss_policy="cache_fill_no_cpu",
            spec_profile=True,
            engine_profile=True,
            engine_profile_cuda_sync=True,
            spec_enable_prefetch=True,
            cache_strategy="lru",
            prefetch_strategy="history_window",
            prefetch_runtime_mode="draft_segment_indexed",
            prefetch_runtime_kind="predictive",
            prefetch_verify_layer_enabled=True,
            draft_cuda_graph_enabled=True,
            verify_cuda_graph=verify_cuda_graph,
        )

        prompts = [llm.tokenizer.encode(prompt_text)]
        warmup_params = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=4)
        llm.generate(["Warmup request."], warmup_params, use_tqdm=False)
        llm.get_profile(reset=True)

        sampling = [SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=args.output_len)]
        t0 = time.time()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        elapsed = time.time() - t0
        profile = llm.get_profile(reset=True)

        token_ids = [x["token_ids"] for x in outputs]
        generated = sum(len(x) for x in token_ids)
        digest_payload = "|".join(",".join(str(t) for t in seq) for seq in token_ids).encode("utf-8")

        result = {
            "case": {
                "name": case_name,
                "verify_cuda_graph": verify_cuda_graph,
                "cache_ratio": float(args.cache_ratio),
                "slots_per_layer": slots,
                "output_len": int(args.output_len),
                "max_draft_tokens": int(args.max_draft_tokens),
            },
            "elapsed_sec": elapsed,
            "generated_output_tokens": generated,
            "throughput_output_tok_s": generated / elapsed if elapsed > 0 else 0.0,
            "outputs_digest": hashlib.sha256(digest_payload).hexdigest(),
            "token_ids": token_ids,
            "engine_profile": profile,
            "summary": _summary_from_profile(profile),
        }
        return result
    finally:
        if llm is not None:
            llm.exit()


def compare_results(baseline: dict, graph: dict, output_dir: str) -> None:
    lines = ["# Verify CUDA Graph Comparison\n"]
    lines.append(f"| Metric | Baseline | Graph | Delta |")
    lines.append(f"|---|---:|---:|---:|")

    for key in ["verify_forward_ms_avg", "run_verify_total_ms_avg", "verify_tokens_per_call",
                 "verify_plan_ms_total", "verify_gpu_compute_ms_total", "verify_route_ms_total",
                 "verify_gpu_gather_ms_total", "verify_scatter_ms_total"]:
        bv = baseline["summary"].get(key, 0.0)
        gv = graph["summary"].get(key, 0.0)
        delta = gv - bv
        lines.append(f"| {key} | {bv:.2f} | {gv:.2f} | {delta:+.2f} |")

    lines.append("")
    bt = baseline.get("throughput_output_tok_s", 0.0)
    gt = graph.get("throughput_output_tok_s", 0.0)
    lines.append(f"| throughput_tok_s | {bt:.2f} | {gt:.2f} | {gt - bt:+.2f} |")
    lines.append("")

    bd = baseline.get("outputs_digest", "")
    gd = graph.get("outputs_digest", "")
    match = "MATCH" if bd == gd else "MISMATCH"
    lines.append(f"**Output digest**: {match}")
    lines.append(f"- baseline: `{bd[:16]}...`")
    lines.append(f"- graph:    `{gd[:16]}...`")

    if bd != gd:
        lines.append("")
        lines.append("**Token divergence**:")
        b_ids = baseline.get("token_ids", [[]])[0]
        g_ids = graph.get("token_ids", [[]])[0]
        for i, (bt, gt) in enumerate(zip(b_ids, g_ids)):
            if bt != gt:
                lines.append(f"  First divergence at position {i}: baseline={bt}, graph={gt}")
                break
        else:
            if len(b_ids) != len(g_ids):
                lines.append(f"  Length mismatch: baseline={len(b_ids)}, graph={len(g_ids)}")

    text = "\n".join(lines) + "\n"
    path = os.path.join(output_dir, "comparison.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"Comparison written to {path}")


def main():
    parser = argparse.ArgumentParser(description="Test verify CUDA graph optimization")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--verify-cuda-graph-on", action="store_true",
                        help="Enable verify CUDA graph for a single run")
    parser.add_argument("--compare-baseline", action="store_true",
                        help="Run both baseline and graph cases, then compare")
    parser.add_argument("--output-dir", default="results/verify_cuda_graph_test")
    parser.add_argument("--output-len", type=int, default=512)
    parser.add_argument("--cache-ratio", type=float, default=0.75)
    parser.add_argument("--max-draft-tokens", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--dist-port", type=int, default=2333)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.compare_baseline:
        print("=" * 60)
        print("Running BASELINE (verify_cuda_graph=False)")
        print("=" * 60)
        baseline = run_single_case(args, "baseline", verify_cuda_graph=False)
        baseline_path = os.path.join(args.output_dir, "baseline.json")
        baseline_save = {k: v for k, v in baseline.items() if k != "token_ids"}
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump(baseline_save, f, indent=2, ensure_ascii=True)
        print(f"Baseline result: {baseline_path}")
        print(f"  verify_forward_ms_avg: {baseline['summary']['verify_forward_ms_avg']:.2f}")
        print(f"  throughput: {baseline['throughput_output_tok_s']:.2f} tok/s")

        print()
        print("=" * 60)
        print("Running GRAPH (verify_cuda_graph=True)")
        print("=" * 60)
        graph = run_single_case(args, "graph", verify_cuda_graph=True)
        graph_path = os.path.join(args.output_dir, "graph.json")
        graph_save = {k: v for k, v in graph.items() if k != "token_ids"}
        with open(graph_path, "w", encoding="utf-8") as f:
            json.dump(graph_save, f, indent=2, ensure_ascii=True)
        print(f"Graph result: {graph_path}")
        print(f"  verify_forward_ms_avg: {graph['summary']['verify_forward_ms_avg']:.2f}")
        print(f"  throughput: {graph['throughput_output_tok_s']:.2f} tok/s")

        print()
        compare_results(baseline, graph, args.output_dir)
    else:
        use_graph = args.verify_cuda_graph_on
        name = "graph" if use_graph else "baseline"
        print(f"Running single case: {name} (verify_cuda_graph={use_graph})")
        result = run_single_case(args, name, verify_cuda_graph=use_graph)
        result_path = os.path.join(args.output_dir, f"{name}.json")
        result_save = {k: v for k, v in result.items() if k != "token_ids"}
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result_save, f, indent=2, ensure_ascii=True)
        print(f"Result: {result_path}")
        print(f"  verify_forward_ms_avg: {result['summary']['verify_forward_ms_avg']:.2f}")
        print(f"  throughput: {result['throughput_output_tok_s']:.2f} tok/s")
        print(f"  digest: {result['outputs_digest'][:16]}...")


if __name__ == "__main__":
    main()
