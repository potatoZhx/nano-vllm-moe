#!/usr/bin/env python3
"""Benchmark per-layer expert cache slot allocation (uniform vs profile_weighted).

Compares uniform allocation (same slots for every MoE layer) against
profile-weighted allocation (slots proportional to per-layer expert demand,
bucketed to a small number of discrete sizes) at the same total GPU memory
budget.

Uses ``benchmarks/scripts/spec_verify_expert_count_stats.py --single-case``
as the single-case runner, same as ``bench_acceptance_predictor.py`` and
``bench_dual_queue_prefetch.py``.

Example:

    conda activate nano_moe
    cd /home/linke/nano-vllm-moe
    rm -rf results/per_layer_slots_bench_76
    CUDA_VISIBLE_DEVICES=2 python scripts/bench_per_layer_slots.py \
        --output-dir results/per_layer_slots_bench \
        --gpu-memory-utilization 0.99 \
        --cache-ratios 0.3125 \
        --output-lens 512 \
        --max-draft-tokens-values 12 \
        --segment-sizes 12 \
        --allocation-modes profile_weighted \
        --slot-buckets 4 \
        --slot-max-bucket-ratio 2.0 \
        --slot-profile-csv pre_exps/exp_and_figs/unique/unique_count_plot_summary_n1024.csv \
        --kt-num-threads 16 \
        --verify-cuda-graph-bucket-steps 3,5,7,10,13
"""
from __future__ import annotations

import argparse
import csv
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

DEFAULT_PROFILE = "results/reroute_impl_20260531/offline_profile_20260531_203257.safetensors"
DEFAULT_PREDICTOR_PATH = "random_cache_srdp_scripts-1/res/run_20260614_133025"
MODEL_PATH = "/data1/models/Qwen3-30B-A3B"
NUM_MOE_LAYERS = 48


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


def _parse_allocation_modes(values: str) -> list[str]:
    modes = []
    for item in values.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item in {"uniform", "profile_weighted"}:
            modes.append(item)
        else:
            raise argparse.ArgumentTypeError(f"invalid allocation mode: {item}")
    if not modes:
        raise argparse.ArgumentTypeError(
            "--allocation-modes must list at least one of uniform/profile_weighted"
        )
    return modes


def build_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    allocation_modes = _parse_allocation_modes(args.allocation_modes)
    cases: list[dict[str, Any]] = []
    for output_len in _parse_csv(args.output_lens, int):
        for cache_ratio in _parse_csv(args.cache_ratios, float):
            for max_draft_tokens in _parse_csv(args.max_draft_tokens_values, int):
                for segment_size in _parse_csv(args.segment_sizes, int):
                    for repeat in range(int(args.repeats)):
                        for mode in allocation_modes:
                            cases.append(
                                {
                                    "allocation_mode": mode,
                                    "output_len": int(output_len),
                                    "cache_ratio": float(cache_ratio),
                                    "max_draft_tokens": int(max_draft_tokens),
                                    "segment_size": int(segment_size),
                                    "repeat": int(repeat),
                                }
                            )
    return cases


def _case_name(case: dict[str, Any]) -> str:
    ratio_pct = int(round(float(case["cache_ratio"]) * 10000))
    return (
        f"{case['allocation_mode']}_seg{int(case['segment_size'])}_"
        f"ratio{ratio_pct:04d}_l{int(case['output_len'])}_"
        f"k{int(case['max_draft_tokens'])}_r{int(case['repeat'])}"
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
    prefetch = summary.get("prefetch", {})
    engine_profile = raw.get("engine_profile", {})

    segment_size = int(case.get("segment_size", 12) or 12)
    draft_seg_replays = int(
        engine_profile.get("model_draft_segment_graph_replay_count", 0) or 0
    )
    verify_calls = int(cuda_graph.get("verify_call_count", 0) or 0)
    num_segments = max(1, 48 // segment_size) if segment_size > 0 else 1
    draft_forward_count = (
        max(1, draft_seg_replays // num_segments) if draft_seg_replays else 1
    )

    pred_draft_seg_submit = int(
        prefetch.get("draft_segment_indexed_submit_count", 0) or 0
    )
    pred_draft_live_submit = int(
        engine_profile.get("model_draft_live_prefetch_submit_count", 0) or 0
    )
    pred_phase1_submit = int(
        prefetch.get("predictive_phase1_submit_count", 0) or 0
    )
    pred_verify_seg_submit = int(
        prefetch.get("verify_segment_prefetch_submit_count", 0) or 0
    )
    draft_phase_submit = (
        pred_draft_seg_submit + pred_draft_live_submit + pred_phase1_submit
    )
    verify_phase_submit = pred_verify_seg_submit
    def ep_float(key: str) -> float:
        value = engine_profile.get(key)
        if value is None:
            value = engine_profile.get(f"model_{key}", 0.0)
        return float(value or 0.0)

    def has_ep(key: str) -> bool:
        return key in engine_profile or f"model_{key}" in engine_profile

    def ep_float_prefer(preferred_key: str, fallback_key: str) -> float:
        if has_ep(preferred_key):
            return ep_float(preferred_key)
        return ep_float(fallback_key)

    def per_layer_sum(suffix: str) -> list[float]:
        return [
            ep_float_prefer(
                f"verify_layer_{layer_idx}_{suffix}",
                f"layer_{layer_idx}_{suffix}",
            )
            for layer_idx in range(NUM_MOE_LAYERS)
        ]

    def per_call(values: list[float]) -> list[float]:
        if verify_calls <= 0:
            return [0.0 for _ in values]
        return [float(value / verify_calls) for value in values]

    layer_cpu_expert_sums = per_layer_sum("realized_cpu_expert_count_sum")
    layer_cpu_route_sums = per_layer_sum("cpu_routes_sum")
    layer_active_expert_sums = per_layer_sum("active_expert_count_sum")
    layer_active_route_sums = per_layer_sum("active_routes_sum")
    layer_profile_counts = per_layer_sum("moe_profile_count")
    layer_cpu_experts_per_call = per_call(layer_cpu_expert_sums)
    layer_cpu_routes_per_call = per_call(layer_cpu_route_sums)
    layer_active_experts_per_call = per_call(layer_active_expert_sums)
    layer_active_routes_per_call = per_call(layer_active_route_sums)
    layer_top_cpu_experts = [
        {
            "layer": layer_idx,
            "cpu_experts_per_call": float(layer_cpu_experts_per_call[layer_idx]),
            "cpu_routes_per_call": float(layer_cpu_routes_per_call[layer_idx]),
            "active_experts_per_call": float(layer_active_experts_per_call[layer_idx]),
            "active_routes_per_call": float(layer_active_routes_per_call[layer_idx]),
        }
        for layer_idx in range(NUM_MOE_LAYERS)
    ]
    layer_top_cpu_experts.sort(
        key=lambda item: (
            -float(item["cpu_experts_per_call"]),
            -float(item["cpu_routes_per_call"]),
            int(item["layer"]),
        )
    )

    return {
        "name": _case_name(case),
        "allocation_mode": str(case["allocation_mode"]),
        "output_len": int(case["output_len"]),
        "cache_ratio": float(case["cache_ratio"]),
        "max_draft_tokens": int(case["max_draft_tokens"]),
        "segment_size": int(case["segment_size"]),
        "repeat": int(case["repeat"]),
        "wall_elapsed_sec": float(wall_elapsed_sec),
        "generated_output_tokens": int(
            raw.get("generated_output_tokens", 0) or 0
        ),
        "throughput_output_tok_s": float(
            summary.get("throughput_output_tok_s", 0.0) or 0.0
        ),
        "decode_phase_output_tok_s": float(
            summary.get("decode_phase_output_tok_s", 0.0) or 0.0
        ),
        "draft_forward_ms_avg": float(
            summary.get("draft_forward_ms_avg", 0.0) or 0.0
        ),
        "verify_forward_ms_avg": float(
            summary.get("verify_forward_ms_avg", 0.0) or 0.0
        ),
        "acceptance_rate": float(
            acceptance.get("acceptance_rate", 0.0) or 0.0
        ),
        "route_hit_rate": float(
            cache.get(
                "true_route_hit_rate", cache.get("route_hit_rate", 0.0)
            )
            or 0.0
        ),
        "avg_miss_routes_per_layer": float(
            cache.get("avg_miss_per_layer", 0.0) or 0.0
        ),
        "avg_active_per_layer": float(
            cache.get("avg_active_per_layer", 0.0) or 0.0
        ),
        "draft_forward_count": draft_forward_count,
        "verify_calls": verify_calls,
        "verify_segment_graph_replays": int(
            cuda_graph.get(
                "verify_kt_hybrid_segment_graph_replay_count", 0
            )
            or 0
        ),
        "draft_phase_submit": draft_phase_submit,
        "verify_phase_submit": verify_phase_submit,
        "draft_prefetch_per_forward": float(
            draft_phase_submit / draft_forward_count
        ),
        "verify_prefetch_per_forward": (
            float(verify_phase_submit / verify_calls) if verify_calls else 0.0
        ),
        "verify_layer_realized_cpu_expert_count_total": layer_cpu_expert_sums,
        "verify_layer_realized_cpu_expert_count_per_call": layer_cpu_experts_per_call,
        "verify_layer_cpu_routes_total": layer_cpu_route_sums,
        "verify_layer_cpu_routes_per_call": layer_cpu_routes_per_call,
        "verify_layer_active_expert_count_total": layer_active_expert_sums,
        "verify_layer_active_expert_count_per_call": layer_active_experts_per_call,
        "verify_layer_active_routes_total": layer_active_route_sums,
        "verify_layer_active_routes_per_call": layer_active_routes_per_call,
        "verify_layer_moe_profile_count": layer_profile_counts,
        "verify_layer_cpu_expert_top": layer_top_cpu_experts[:12],
        "prefetch_submit_count": int(
            prefetch.get("submit_count", 0) or 0
        ),
        "prefetch_completed_count": int(
            prefetch.get("completed_count", 0) or 0
        ),
        "prefetch_late_count": int(
            engine_profile.get("model_prefetch_late_count", 0) or 0
        ),
        "prefetch_publish_count": int(
            engine_profile.get("model_publish_count", 0) or 0
        ),
        "prefetch_consumed_count": int(
            prefetch.get("consumed_count", 0) or 0
        ),
        "predicted_alpha_avg": float(
            acceptance.get("predicted_alpha_avg", 0.0) or 0.0
        ),
        "predicted_alpha_count": int(
            acceptance.get("predicted_alpha_count", 0) or 0
        ),
        "predicted_alpha_min": float(
            acceptance.get("predicted_alpha_min", 0.0) or 0.0
        ),
        "predicted_alpha_max": float(
            acceptance.get("predicted_alpha_max", 0.0) or 0.0
        ),
        "outputs_digest": str(raw.get("outputs_digest", "")),
    }


def _comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[int, float, int, int, int], dict[str, dict[str, Any]]
    ] = {}
    for row in rows:
        key = (
            int(row["output_len"]),
            round(float(row["cache_ratio"]), 6),
            int(row["max_draft_tokens"]),
            int(row["segment_size"]),
            int(row["repeat"]),
        )
        grouped.setdefault(key, {})[str(row["allocation_mode"])] = row

    comparisons: list[dict[str, Any]] = []
    for pair in grouped.values():
        uniform = pair.get("uniform")
        weighted = pair.get("profile_weighted")
        if uniform is None or weighted is None:
            continue
        u_tps = float(uniform["throughput_output_tok_s"])
        w_tps = float(weighted["throughput_output_tok_s"])
        comparisons.append(
            {
                "output_len": int(uniform["output_len"]),
                "cache_ratio": float(uniform["cache_ratio"]),
                "max_draft_tokens": int(uniform["max_draft_tokens"]),
                "segment_size": int(uniform["segment_size"]),
                "repeat": int(uniform["repeat"]),
                "digest_match": uniform["outputs_digest"]
                == weighted["outputs_digest"],
                "throughput_uniform": u_tps,
                "throughput_weighted": w_tps,
                "throughput_delta": w_tps - u_tps,
                "throughput_change_pct": (
                    100.0 * (w_tps - u_tps) / u_tps if u_tps > 0.0 else 0.0
                ),
                "hit_rate_uniform": float(uniform["route_hit_rate"]),
                "hit_rate_weighted": float(weighted["route_hit_rate"]),
                "hit_rate_delta": float(weighted["route_hit_rate"])
                - float(uniform["route_hit_rate"]),
                "acceptance_uniform": float(uniform["acceptance_rate"]),
                "acceptance_weighted": float(weighted["acceptance_rate"]),
                "acceptance_delta": float(weighted["acceptance_rate"])
                - float(uniform["acceptance_rate"]),
                "draft_ms_uniform": float(uniform["draft_forward_ms_avg"]),
                "draft_ms_weighted": float(weighted["draft_forward_ms_avg"]),
                "draft_ms_delta": float(weighted["draft_forward_ms_avg"])
                - float(uniform["draft_forward_ms_avg"]),
                "verify_ms_uniform": float(uniform["verify_forward_ms_avg"]),
                "verify_ms_weighted": float(
                    weighted["verify_forward_ms_avg"]
                ),
                "verify_ms_delta": float(weighted["verify_forward_ms_avg"])
                - float(uniform["verify_forward_ms_avg"]),
                "miss_per_layer_uniform": float(
                    uniform["avg_miss_routes_per_layer"]
                ),
                "miss_per_layer_weighted": float(
                    weighted["avg_miss_routes_per_layer"]
                ),
                "predicted_alpha_uniform": float(
                    uniform["predicted_alpha_avg"]
                ),
                "predicted_alpha_weighted": float(
                    weighted["predicted_alpha_avg"]
                ),
            }
        )
    return comparisons


def _command(
    args: argparse.Namespace,
    repo_root: Path,
    prompt_file: Path,
    case: dict[str, Any],
    output_path: Path,
    case_index: int,
) -> list[str]:
    single_case_script = (
        repo_root
        / "benchmarks"
        / "scripts"
        / "spec_verify_expert_count_stats.py"
    )
    segment_size = int(case["segment_size"])
    allocation_mode = str(case["allocation_mode"])
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
        "--slot-allocation",
        allocation_mode,
        "--slot-buckets",
        str(args.slot_buckets),
        "--slot-max-bucket-ratio",
        str(args.slot_max_bucket_ratio),
        "--slot-profile-csv",
        args.slot_profile_csv,
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
        "--acceptance-predictor-enabled",
        "true",
        "--acceptance-predictor-path",
        args.acceptance_predictor_path,
        "--acceptance-predictor-step-horizon",
        str(args.acceptance_predictor_step_horizon),
        "--draft-alpha-stop-threshold",
        str(args.draft_alpha_stop_threshold),
        "--draft-stop-policy",
        args.draft_stop_policy,
        "--draft-tpot-td-ms",
        str(args.draft_tpot_td_ms),
        "--draft-tpot-tv-ms",
        str(args.draft_tpot_tv_ms),
        "--prefetch-enabled",
        "true",
        "--prefetch-runtime-mode",
        "draft_segment_indexed",
        "--prefetch-runtime-kind",
        "predictive",
        "--dual-queue-segment-size",
        str(segment_size),
        "--prefetch-strategy",
        "history_window",
        "--prefetch-staging-slots-per-layer",
        str(args.prefetch_staging_slots_per_layer),
        "--prefetch-max-inflight",
        str(args.prefetch_max_inflight),
        "--prefetch-transfer-stream-count",
        str(args.prefetch_transfer_stream_count),
        "--prefetch-metadata-host-buffer-pool-size",
        str(args.prefetch_metadata_host_buffer_pool_size),
        "--prefetch-step-budget",
        str(args.prefetch_step_budget),
        "--prefetch-verify-layer-max-budget",
        str(args.prefetch_verify_layer_max_budget),
        "--prefetch-verify-wait-ms",
        "0",
        "--prefetch-verify-attention-ratio",
        str(args.prefetch_verify_attention_ratio),
        "--cache-eviction-budget-per-step",
        str(args.cache_eviction_budget_per_step),
        "--prefetch-global-queue-capacity",
        str(args.prefetch_global_queue_capacity),
        "--draft-cuda-graph-enabled",
        "true",
        "--draft-cuda-graph-cpu-backend",
        "none",
        "--draft-prefetch-segment-size",
        str(segment_size),
        "--draft-prefetch-segment-host-buffer-pool-size",
        str(args.draft_segment_host_buffer_pool_size),
        "--draft-prefetch-visible-budget-ms",
        str(args.draft_prefetch_visible_budget_ms),
        "--draft-prefetch-min-per-boundary",
        "0",
        "--draft-prefetch-max-per-boundary",
        str(args.draft_prefetch_max_per_boundary),
        "--verify-cuda-graph",
        "true",
        "--verify-cuda-graph-bucket-steps",
        args.verify_cuda_graph_bucket_steps,
        "--verify-prefetch-segment-size",
        str(segment_size),
        "--verify-prefetch-visible-budget-ms",
        str(args.verify_prefetch_visible_budget_ms),
        "--verify-prefetch-min-per-boundary",
        "0",
        "--verify-prefetch-max-per-boundary",
        str(args.verify_prefetch_max_per_boundary),
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
        return _row_from_raw(
            case, raw, float(raw.get("elapsed_sec", 0.0) or 0.0)
        )

    cmd = _command(
        args, repo_root, prompt_file, case, case_json, case_index
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    )
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
    print(
        f"  tok/s={row['throughput_output_tok_s']:.3f} "
        f"decode_tok/s={row['decode_phase_output_tok_s']:.3f} "
        f"draft_ms={row['draft_forward_ms_avg']:.3f} "
        f"verify_ms={row['verify_forward_ms_avg']:.3f} "
        f"hit={row['route_hit_rate']:.4f} accept={row['acceptance_rate']:.4f} "
        f"miss/L={row['avg_miss_routes_per_layer']:.2f} "
        f"active/L={row['avg_active_per_layer']:.2f}",
        flush=True,
    )
    top_layers = ", ".join(
        f"L{int(item['layer'])}:{float(item['cpu_experts_per_call']):.1f}exp/"
        f"{float(item['cpu_routes_per_call']):.1f}routes"
        for item in row["verify_layer_cpu_expert_top"][:6]
        if float(item["cpu_experts_per_call"]) > 0.0
    )
    print(f"  verify per-layer cpu_exp top: {top_layers or 'none'}", flush=True)
    print(
        f"  prefetch: submit={row['prefetch_submit_count']} "
        f"publish={row['prefetch_publish_count']} "
        f"late={row['prefetch_late_count']} "
        f"consumed={row['prefetch_consumed_count']} "
        f"draft_phase={row['draft_phase_submit']} "
        f"verify_phase={row['verify_phase_submit']} "
        f"draft/fwd={row['draft_prefetch_per_forward']:.1f} "
        f"verify/fwd={row['verify_prefetch_per_forward']:.1f}",
        flush=True,
    )
    print(
        f"  predictor: predict_alpha_avg={row['predicted_alpha_avg']:.4f} "
        f"count={row['predicted_alpha_count']} "
        f"min={row['predicted_alpha_min']:.4f} max={row['predicted_alpha_max']:.4f} ",
        flush=True,
    )
    return row


def write_markdown_report(summary: dict[str, Any], path: Path) -> None:
    metadata = summary["metadata"]
    rows = summary["rows"]
    comparisons = summary["comparisons"]
    lines = [
        "# Per-Layer Slot Allocation Benchmark",
        "",
        f"- timestamp: `{metadata['timestamp']}`",
        f"- model: `{metadata['model_path']}`",
        f"- profile artifact: `{metadata['profile_artifact']}`",
        f"- slot profile CSV: `{metadata.get('slot_profile_csv', '')}`",
        f"- allocation modes: `{', '.join(metadata['allocation_modes'])}`",
        f"- slot buckets: `{metadata['slot_buckets']}`",
        f"- max bucket ratio: `{metadata['slot_max_bucket_ratio']}`",
        f"- segment sizes: `{', '.join(str(x) for x in metadata['segment_sizes'])}`",
        f"- output directory: `{metadata['output_dir']}`",
        "",
        "## Cases",
        "",
        "| alloc | seg | out | ratio | K | rep | tok/s | draft ms | verify ms | hit | accept | pred_alpha | miss/L | submit | publish | late |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['allocation_mode']} | {row['segment_size']} | "
            f"{row['output_len']} | {row['cache_ratio']:.4f} | {row['max_draft_tokens']} | "
            f"{row['repeat']} | {row['throughput_output_tok_s']:.3f} | "
            f"{row['draft_forward_ms_avg']:.3f} | {row['verify_forward_ms_avg']:.3f} | "
            f"{row['route_hit_rate']:.4f} | {row['acceptance_rate']:.4f} | "
            f"{row['predicted_alpha_avg']:.4f} | "
            f"{row['avg_miss_routes_per_layer']:.2f} | "
            f"{row['prefetch_submit_count']} | {row['prefetch_publish_count']} | "
            f"{row['prefetch_late_count']} |"
        )

    lines.extend(
        [
            "",
            "## Per-Layer CPU Experts",
            "",
            "Full 48-layer arrays are in `summary.json`; the CSV export is `per_layer_cpu_experts.csv`.",
            "",
            "| case | alloc | K | layer | CPU experts/call | CPU routes/call | active experts/call | active routes/call |",
            "|:---|:---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        for item in row["verify_layer_cpu_expert_top"]:
            if float(item["cpu_experts_per_call"]) <= 0.0:
                continue
            lines.append(
                "| "
                f"{row['name']} | {row['allocation_mode']} | "
                f"{row['max_draft_tokens']} | {int(item['layer'])} | "
                f"{float(item['cpu_experts_per_call']):.3f} | "
                f"{float(item['cpu_routes_per_call']):.3f} | "
                f"{float(item['active_experts_per_call']):.3f} | "
                f"{float(item['active_routes_per_call']):.3f} |"
            )

    lines.extend(
        [
            "",
            "## Uniform vs Profile-Weighted Comparison",
            "",
            "| seg | out | ratio | K | rep | digest | tok/s uniform | tok/s weighted | change % | hit uniform | hit weighted | hit delta | accept delta | alpha uniform | alpha weighted | draft ms delta | verify ms delta | miss/L uniform | miss/L weighted |",
            "|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            "| "
            f"{row['segment_size']} | {row['output_len']} | {row['cache_ratio']:.4f} | "
            f"{row['max_draft_tokens']} | {row['repeat']} | "
            f"{'match' if row['digest_match'] else 'DIFF'} | "
            f"{row['throughput_uniform']:.3f} | {row['throughput_weighted']:.3f} | "
            f"{row['throughput_change_pct']:+.2f} | "
            f"{row['hit_rate_uniform']:.4f} | {row['hit_rate_weighted']:.4f} | "
            f"{row['hit_rate_delta']:+.4f} | "
            f"{row['acceptance_delta']:+.4f} | "
            f"{row['predicted_alpha_uniform']:.4f} | {row['predicted_alpha_weighted']:.4f} | "
            f"{row['draft_ms_delta']:+.3f} | {row['verify_ms_delta']:+.3f} | "
            f"{row['miss_per_layer_uniform']:.2f} | {row['miss_per_layer_weighted']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `change %` = throughput improvement of profile_weighted over uniform (positive = better).",
            "- `hit delta` = route hit rate improvement (positive = better cache utilisation).",
            "- `digest` should typically differ: different cache contents may cause different rerouting",
            "  decisions and therefore different generated tokens. `match` is a bonus, not a requirement.",
            "- `miss/L` = average miss routes per layer; lower is better.",
            "- All other columns mirror `bench_dual_queue_prefetch.py` for direct comparison.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_per_layer_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "case",
        "allocation_mode",
        "output_len",
        "cache_ratio",
        "segment_size",
        "max_draft_tokens",
        "repeat",
        "layer",
        "cpu_experts_total",
        "cpu_experts_per_call",
        "cpu_routes_total",
        "cpu_routes_per_call",
        "active_experts_total",
        "active_experts_per_call",
        "active_routes_total",
        "active_routes_per_call",
        "layer_profile_count",
        "verify_calls",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for layer_idx in range(NUM_MOE_LAYERS):
                writer.writerow(
                    {
                        "case": row["name"],
                        "allocation_mode": row["allocation_mode"],
                        "output_len": row["output_len"],
                        "cache_ratio": row["cache_ratio"],
                        "segment_size": row["segment_size"],
                        "max_draft_tokens": row["max_draft_tokens"],
                        "repeat": row["repeat"],
                        "layer": layer_idx,
                        "cpu_experts_total": row[
                            "verify_layer_realized_cpu_expert_count_total"
                        ][layer_idx],
                        "cpu_experts_per_call": row[
                            "verify_layer_realized_cpu_expert_count_per_call"
                        ][layer_idx],
                        "cpu_routes_total": row["verify_layer_cpu_routes_total"][layer_idx],
                        "cpu_routes_per_call": row["verify_layer_cpu_routes_per_call"][layer_idx],
                        "active_experts_total": row[
                            "verify_layer_active_expert_count_total"
                        ][layer_idx],
                        "active_experts_per_call": row[
                            "verify_layer_active_expert_count_per_call"
                        ][layer_idx],
                        "active_routes_total": row["verify_layer_active_routes_total"][layer_idx],
                        "active_routes_per_call": row[
                            "verify_layer_active_routes_per_call"
                        ][layer_idx],
                        "layer_profile_count": row["verify_layer_moe_profile_count"][layer_idx],
                        "verify_calls": row["verify_calls"],
                    }
                )


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = output_dir / "per_layer_slots_prompt.txt"
    prompt_file.write_text(PROMPT_TEXT + "\n", encoding="utf-8")

    cases = build_cases(args)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        try:
            print("=" * 80)
            rows.append(
                run_case(args, repo_root, prompt_file, case, case_index)
            )
        except Exception as error:
            failures.append({"case": case, "error": str(error)})
            if args.fail_fast:
                raise

    allocation_modes = _parse_allocation_modes(args.allocation_modes)
    summary = {
        "metadata": {
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "model_path": args.model_path,
            "profile_artifact": args.profile_artifact,
            "slot_profile_csv": args.slot_profile_csv,
            "output_dir": str(output_dir),
            "allocation_modes": allocation_modes,
            "slot_buckets": int(args.slot_buckets),
            "slot_max_bucket_ratio": float(args.slot_max_bucket_ratio),
            "segment_sizes": _parse_csv(args.segment_sizes, int),
            "cache_ratios": _parse_csv(args.cache_ratios, float),
            "output_lens": _parse_csv(args.output_lens, int),
            "max_draft_tokens_values": _parse_csv(
                args.max_draft_tokens_values, int
            ),
            "repeats": int(args.repeats),
            "per_layer_cpu_experts_csv": str(output_dir / "per_layer_cpu_experts.csv"),
            "argv": sys.argv,
        },
        "rows": rows,
        "comparisons": _comparison_rows(rows),
        "failures": failures,
    }
    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"
    write_per_layer_csv(rows, output_dir / "per_layer_cpu_experts.csv")
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown_report(summary, summary_md)
    if args.report_doc:
        write_markdown_report(summary, Path(args.report_doc))

    print(f"summary_json={summary_json}")
    print(f"summary_md={summary_md}")
    if summary["comparisons"]:
        for comp in summary["comparisons"]:
            print(
                f"  out={comp['output_len']} ratio={comp['cache_ratio']:.4f} "
                f"throughput: {comp['throughput_change_pct']:+.2f}% "
                f"hit_delta={comp['hit_rate_delta']:+.4f} "
                f"accept_delta={comp['acceptance_delta']:+.4f}"
            )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark per-layer expert cache slot allocation "
            "(uniform vs profile_weighted) in the draft segment-graph path."
        )
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--profile-artifact", default=DEFAULT_PROFILE)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-doc", default="")
    parser.add_argument(
        "--allocation-modes",
        default="uniform,profile_weighted",
    )
    parser.add_argument("--slot-buckets", type=int, default=4)
    parser.add_argument("--slot-max-bucket-ratio", type=float, default=2.0)
    parser.add_argument("--slot-profile-csv", default="")
    parser.add_argument("--output-lens", default="128,512")
    parser.add_argument("--cache-ratios", default="0.25,0.3125,0.50")
    parser.add_argument("--max-draft-tokens-values", default="4,8")
    parser.add_argument("--segment-sizes", default="12")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--cache-strategy", default="lru")
    parser.add_argument("--draft-reroute-policy", default="entropy_cache_bias")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--acceptance-strategy", default="standard_sampling")
    parser.add_argument("--acceptance-threshold", type=float, default=0.7)
    parser.add_argument("--acceptance-predictor-path", default=DEFAULT_PREDICTOR_PATH)
    parser.add_argument("--acceptance-predictor-step-horizon", type=int, default=32)
    parser.add_argument("--draft-alpha-stop-threshold", type=float, default=0.89)
    parser.add_argument("--draft-stop-policy", choices=["none", "alpha_threshold", "tpot"], default="tpot")
    parser.add_argument("--draft-tpot-td-ms", type=float, default=19.0)
    parser.add_argument("--draft-tpot-tv-ms", type=float, default=80.0)

    parser.add_argument("--prefetch-step-budget", type=int, default=16)
    parser.add_argument("--prefetch-max-inflight", type=int, default=16)
    parser.add_argument(
        "--prefetch-transfer-stream-count", type=int, default=1
    )
    parser.add_argument(
        "--prefetch-staging-slots-per-layer", type=int, default=2
    )
    parser.add_argument(
        "--prefetch-metadata-host-buffer-pool-size", type=int, default=3
    )
    parser.add_argument(
        "--prefetch-global-queue-capacity", type=int, default=4096
    )
    parser.add_argument(
        "--prefetch-verify-layer-max-budget", type=int, default=8
    )
    parser.add_argument(
        "--prefetch-verify-attention-ratio", type=float, default=1.0
    )
    parser.add_argument(
        "--cache-eviction-budget-per-step", type=int, default=2
    )
    parser.add_argument(
        "--draft-segment-host-buffer-pool-size", type=int, default=0
    )
    parser.add_argument(
        "--draft-prefetch-visible-budget-ms", type=float, default=3.0
    )
    parser.add_argument(
        "--draft-prefetch-max-per-boundary", type=int, default=16
    )
    parser.add_argument(
        "--verify-prefetch-visible-budget-ms", type=float, default=12.0
    )
    parser.add_argument(
        "--verify-prefetch-max-per-boundary", type=int, default=4
    )

    parser.add_argument(
        "--cpu-expert-workspace-max-routes", type=int, default=327680
    )
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
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.99)
    parser.add_argument(
        "--verify-cuda-graph-bucket-steps", default="3,5,8,12"
    )
    parser.add_argument("--dist-port-base", type=int, default=30800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sync-layer-timing", type=str2bool, default=False
    )
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
