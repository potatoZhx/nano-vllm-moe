#!/usr/bin/env python3
"""Benchmark the dual-queue segment prefetch runtime.

The benchmark reuses the single-case runner from
``benchmarks/scripts/spec_verify_expert_count_stats.py`` and compares
``dual_queue`` against ``predictive`` under the same cache ratio, output
length, draft length, and segment size.

Example:

    conda activate nano_moe
    cd /home/linke/nano-vllm-moe
    python scripts/bench_dual_queue_prefetch.py \
        --output-dir results/dual_queue_bench \
        --gpu-memory-utilization 0.99 \
        --cache-ratios 0.25,0.3125 \
        --output-lens 128,512 \
        --budget-safety-ratio 0.6 \
        --max-draft-tokens-values 4,8 \
        --segment-sizes 12 \
        --runtime-kinds dual_queue,predictive \
        --kt-num-threads 32
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

DEFAULT_PROFILE = "results/reroute_impl_20260531/offline_profile_20260531_203257.safetensors"
MODEL_PATH = "/data1/models/Qwen3-30B-A3B"
DUAL_DRAFT_SOURCE = "dual_queue_draft_predict"
DUAL_GROUND_TRUTH_SOURCE = "dual_queue_ground_truth"


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
    runtime_kinds = _parse_csv(args.runtime_kinds, str)
    invalid = sorted(set(runtime_kinds) - {"dual_queue", "predictive"})
    if invalid:
        raise ValueError(f"unsupported runtime kinds: {', '.join(invalid)}")

    cases: list[dict[str, Any]] = []
    for output_len in _parse_csv(args.output_lens, int):
        for cache_ratio in _parse_csv(args.cache_ratios, float):
            for max_draft_tokens in _parse_csv(args.max_draft_tokens_values, int):
                for segment_size in _parse_csv(args.segment_sizes, int):
                    for repeat in range(int(args.repeats)):
                        for runtime_kind in runtime_kinds:
                            cases.append(
                                {
                                    "runtime_kind": runtime_kind,
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
        f"{case['runtime_kind']}_seg{int(case['segment_size'])}_"
        f"ratio{ratio_pct:04d}_l{int(case['output_len'])}_"
        f"k{int(case['max_draft_tokens'])}_r{int(case['repeat'])}"
    )


def _mapping_int(mapping: Any, key: str) -> int:
    if not isinstance(mapping, dict):
        return 0
    return int(mapping.get(key, 0) or 0)


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
    dual_queue = summary.get("dual_queue", {})
    engine_profile = raw.get("engine_profile", {})

    submit_by_source = dual_queue.get("submit_count_by_source", {})
    completed_by_source = dual_queue.get("completed_count_by_source", {})
    published_by_source = dual_queue.get("published_count_by_source", {})
    late_by_source = dual_queue.get("late_count_by_source", {})
    draft_submit = _mapping_int(submit_by_source, DUAL_DRAFT_SOURCE)
    gt_submit = _mapping_int(submit_by_source, DUAL_GROUND_TRUTH_SOURCE)
    draft_published = _mapping_int(published_by_source, DUAL_DRAFT_SOURCE)
    gt_published = _mapping_int(published_by_source, DUAL_GROUND_TRUTH_SOURCE)
    dual_submit = draft_submit + gt_submit
    dual_published = draft_published + gt_published

    return {
        "name": _case_name(case),
        "runtime_kind": str(case["runtime_kind"]),
        "output_len": int(case["output_len"]),
        "cache_ratio": float(case["cache_ratio"]),
        "max_draft_tokens": int(case["max_draft_tokens"]),
        "segment_size": int(case["segment_size"]),
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
        "verify_calls": int(cuda_graph.get("verify_call_count", 0) or 0),
        "verify_segment_graph_replays": int(
            cuda_graph.get("verify_kt_hybrid_segment_graph_replay_count", 0) or 0
        ),
        "verify_segment_metadata_enqueue_count": int(
            cuda_graph.get("verify_segment_metadata_enqueue_count", 0) or 0
        ),
        "prefetch_submit_count": int(prefetch.get("submit_count", 0) or 0),
        "prefetch_completed_count": int(prefetch.get("completed_count", 0) or 0),
        "prefetch_late_count": int(engine_profile.get("model_prefetch_late_count", 0) or 0),
        "prefetch_publish_count": int(engine_profile.get("model_publish_count", 0) or 0),
        "prefetch_consumed_count": int(prefetch.get("consumed_count", 0) or 0),
        "draft_budget": int(dual_queue.get("draft_budget", 0) or 0),
        "verify_budget": int(dual_queue.get("verify_budget", 0) or 0),
        "expert_transfer_ms": float(dual_queue.get("expert_transfer_ms", 0.0) or 0.0),
        "draft_predict_size": int(dual_queue.get("draft_predict_size", 0) or 0),
        "ground_truth_size": int(dual_queue.get("ground_truth_size", 0) or 0),
        "dual_submit_count": dual_submit,
        "dual_completed_count": (
            _mapping_int(completed_by_source, DUAL_DRAFT_SOURCE)
            + _mapping_int(completed_by_source, DUAL_GROUND_TRUTH_SOURCE)
        ),
        "dual_published_count": dual_published,
        "dual_late_count": (
            _mapping_int(late_by_source, DUAL_DRAFT_SOURCE)
            + _mapping_int(late_by_source, DUAL_GROUND_TRUTH_SOURCE)
        ),
        "draft_submit_count": draft_submit,
        "ground_truth_submit_count": gt_submit,
        "draft_published_count": draft_published,
        "ground_truth_published_count": gt_published,
        "dual_publish_ratio": float(dual_published / dual_submit) if dual_submit else 0.0,
        "target_miss_count": int(dual_queue.get("target_miss_count", 0) or 0),
        "round_end_discard_count": int(dual_queue.get("round_end_discard_count", 0) or 0),
        "expired_transfer_count": int(dual_queue.get("expired_transfer_count", 0) or 0),
        "stale_draft_metadata_count": int(
            dual_queue.get("stale_draft_metadata_count", 0) or 0
        ),
        "round_clear_count": int(dual_queue.get("round_clear_count", 0) or 0),
        "all_slots_protected_count": int(
            dual_queue.get("all_slots_protected_count", 0) or 0
        ),
        "metadata_host_buffer_drop_count": int(
            dual_queue.get("metadata_host_buffer_drop_count", 0) or 0
        ),
        "metadata_drain_wait_ms": float(
            engine_profile.get("model_prefetch_async_drain_wait_ms", 0.0) or 0.0
        ),
        "metadata_buffer_reuse_wait_ms": float(
            engine_profile.get("model_prefetch_async_buffer_reuse_wait_ms", 0.0) or 0.0
        ),
        "outputs_digest": str(raw.get("outputs_digest", "")),
    }


def _comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, float, int, int, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (
            int(row["output_len"]),
            round(float(row["cache_ratio"]), 6),
            int(row["max_draft_tokens"]),
            int(row["segment_size"]),
            int(row["repeat"]),
        )
        grouped.setdefault(key, {})[str(row["runtime_kind"])] = row

    comparisons: list[dict[str, Any]] = []
    for pair in grouped.values():
        dual = pair.get("dual_queue")
        predictive = pair.get("predictive")
        if dual is None or predictive is None:
            continue
        comparisons.append(
            {
                "output_len": int(dual["output_len"]),
                "cache_ratio": float(dual["cache_ratio"]),
                "max_draft_tokens": int(dual["max_draft_tokens"]),
                "segment_size": int(dual["segment_size"]),
                "repeat": int(dual["repeat"]),
                "digest_match": dual["outputs_digest"] == predictive["outputs_digest"],
                "throughput_delta": (
                    float(dual["throughput_output_tok_s"])
                    - float(predictive["throughput_output_tok_s"])
                ),
                "throughput_ratio": (
                    float(dual["throughput_output_tok_s"])
                    / float(predictive["throughput_output_tok_s"])
                    if float(predictive["throughput_output_tok_s"]) > 0.0
                    else 0.0
                ),
                "draft_ms_delta": (
                    float(dual["draft_forward_ms_avg"])
                    - float(predictive["draft_forward_ms_avg"])
                ),
                "verify_ms_delta": (
                    float(dual["verify_forward_ms_avg"])
                    - float(predictive["verify_forward_ms_avg"])
                ),
                "route_hit_delta": (
                    float(dual["route_hit_rate"]) - float(predictive["route_hit_rate"])
                ),
                "acceptance_delta": (
                    float(dual["acceptance_rate"]) - float(predictive["acceptance_rate"])
                ),
                "dual_publish_ratio": float(dual["dual_publish_ratio"]),
                "dual_target_miss_count": int(dual["target_miss_count"]),
                "dual_round_end_discard_count": int(dual["round_end_discard_count"]),
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
        "--prefetch-enabled",
        "true",
        "--prefetch-runtime-mode",
        "draft_segment_indexed",
        "--prefetch-runtime-kind",
        str(case["runtime_kind"]),
        "--dual-queue-segment-size",
        str(segment_size),
        "--dual-queue-ground-truth-decay",
        str(args.ground_truth_decay),
        "--dual-queue-ground-truth-ttl-rounds",
        str(args.ground_truth_ttl_rounds),
        "--dual-queue-ground-truth-count-weight",
        str(args.ground_truth_count_weight),
        "--dual-queue-budget-safety-ratio",
        str(args.budget_safety_ratio),
        "--dual-queue-segment-time-ema-alpha",
        str(args.segment_time_ema_alpha),
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
    print(
        f"  tok/s={row['throughput_output_tok_s']:.3f} "
        f"draft_ms={row['draft_forward_ms_avg']:.3f} "
        f"verify_ms={row['verify_forward_ms_avg']:.3f} "
        f"hit={row['route_hit_rate']:.4f} accept={row['acceptance_rate']:.4f} "
        f"budget={row['draft_budget']}/{row['verify_budget']} "
        f"dual_submit/publish/late={row['dual_submit_count']}/"
        f"{row['dual_published_count']}/{row['dual_late_count']} "
        f"target_miss={row['target_miss_count']}",
        flush=True,
    )
    return row


def write_markdown_report(summary: dict[str, Any], path: Path) -> None:
    metadata = summary["metadata"]
    rows = summary["rows"]
    comparisons = summary["comparisons"]
    lines = [
        "# Dual-Queue Prefetch Benchmark",
        "",
        f"- timestamp: `{metadata['timestamp']}`",
        f"- model: `{metadata['model_path']}`",
        f"- runtime kinds: `{', '.join(metadata['runtime_kinds'])}`",
        f"- segment sizes: `{', '.join(str(x) for x in metadata['segment_sizes'])}`",
        f"- output directory: `{metadata['output_dir']}`",
        "",
        "## Cases",
        "",
        "| runtime | seg | out | ratio | K | rep | tok/s | draft ms | verify ms | hit | accept | budget D/V | submit | publish | late | target miss | round discard | metadata drop |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['runtime_kind']} | {row['segment_size']} | {row['output_len']} | "
            f"{row['cache_ratio']:.4f} | {row['max_draft_tokens']} | {row['repeat']} | "
            f"{row['throughput_output_tok_s']:.3f} | "
            f"{row['draft_forward_ms_avg']:.3f} | {row['verify_forward_ms_avg']:.3f} | "
            f"{row['route_hit_rate']:.4f} | {row['acceptance_rate']:.4f} | "
            f"{row['draft_budget']}/{row['verify_budget']} | "
            f"{row['dual_submit_count']} | {row['dual_published_count']} | "
            f"{row['dual_late_count']} | {row['target_miss_count']} | "
            f"{row['round_end_discard_count']} | {row['metadata_host_buffer_drop_count']} |"
        )

    lines.extend(
        [
            "",
            "## Dual Queue vs Predictive",
            "",
            "| seg | out | ratio | K | rep | digest | tok/s delta | speed ratio | draft ms delta | verify ms delta | hit delta | accept delta | publish ratio | target miss | round discard |",
            "|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            "| "
            f"{row['segment_size']} | {row['output_len']} | {row['cache_ratio']:.4f} | "
            f"{row['max_draft_tokens']} | {row['repeat']} | "
            f"{'match' if row['digest_match'] else 'DIFF'} | "
            f"{row['throughput_delta']:+.3f} | {row['throughput_ratio']:.3f} | "
            f"{row['draft_ms_delta']:+.3f} | {row['verify_ms_delta']:+.3f} | "
            f"{row['route_hit_delta']:+.4f} | {row['acceptance_delta']:+.4f} | "
            f"{row['dual_publish_ratio']:.3f} | {row['dual_target_miss_count']} | "
            f"{row['dual_round_end_discard_count']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `submit/publish/late` count only dual-queue sources. Predictive rows therefore show zero.",
            "- `target miss` means H2D was not ready when the target segment started; the transfer was expired.",
            "- `round discard` counts transfers still unfinished when verify ended.",
            "- `metadata drop` counts samples skipped because every host metadata buffer was busy.",
            "- A non-zero metadata drain wait for dual queue indicates a regression in best-effort behavior.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = output_dir / "dual_queue_prompt.txt"
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
            "runtime_kinds": _parse_csv(args.runtime_kinds, str),
            "segment_sizes": _parse_csv(args.segment_sizes, int),
            "cache_ratios": _parse_csv(args.cache_ratios, float),
            "output_lens": _parse_csv(args.output_lens, int),
            "max_draft_tokens_values": _parse_csv(args.max_draft_tokens_values, int),
            "repeats": int(args.repeats),
            "argv": sys.argv,
        },
        "rows": rows,
        "comparisons": _comparison_rows(rows),
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
        description="Benchmark dual_queue segment prefetch against predictive prefetch."
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--profile-artifact", default=DEFAULT_PROFILE)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-doc", default="")
    parser.add_argument("--runtime-kinds", default="dual_queue,predictive")
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

    parser.add_argument("--ground-truth-decay", type=float, default=0.9)
    parser.add_argument("--ground-truth-ttl-rounds", type=int, default=64)
    parser.add_argument("--ground-truth-count-weight", type=float, default=0.1)
    parser.add_argument("--budget-safety-ratio", type=float, default=0.8)
    parser.add_argument("--segment-time-ema-alpha", type=float, default=0.2)

    parser.add_argument("--prefetch-step-budget", type=int, default=16)
    parser.add_argument("--prefetch-max-inflight", type=int, default=16)
    parser.add_argument("--prefetch-transfer-stream-count", type=int, default=1)
    parser.add_argument("--prefetch-staging-slots-per-layer", type=int, default=2)
    parser.add_argument("--prefetch-metadata-host-buffer-pool-size", type=int, default=3)
    parser.add_argument("--prefetch-global-queue-capacity", type=int, default=4096)
    parser.add_argument("--prefetch-verify-layer-max-budget", type=int, default=8)
    parser.add_argument("--cache-eviction-budget-per-step", type=int, default=2)
    parser.add_argument("--draft-segment-host-buffer-pool-size", type=int, default=0)
    parser.add_argument("--draft-prefetch-visible-budget-ms", type=float, default=3.0)
    parser.add_argument("--draft-prefetch-max-per-boundary", type=int, default=16)
    parser.add_argument("--verify-prefetch-visible-budget-ms", type=float, default=12.0)
    parser.add_argument("--verify-prefetch-max-per-boundary", type=int, default=16)

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
