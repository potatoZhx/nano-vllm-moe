#!/usr/bin/env python3
"""Benchmark the on-GPU acceptance predictor in the draft segment-graph path.

This mirrors ``scripts/bench_dual_queue_prefetch.py`` (it reuses the same
single-case runner ``benchmarks/scripts/spec_verify_expert_count_stats.py`` and
emits every field that benchmark emits) so results are directly comparable. The
comparison dimension here is the acceptance predictor: ``on`` vs ``off`` under the
otherwise identical ``predictive`` (draft_segment_indexed) configuration, which
isolates the predictor's overhead and surfaces the predicted acceptance alpha.

Per the predictor design, alpha is collected per draft sub-step and written into
the profile file (``summary.acceptance.predicted_alpha_*`` and the per-sequence
``spec_step_traces[*].sequences[*].predicted_alpha``). The console output adds
``predict_alpha_avg``.

Example:

    conda activate nano_moe
    cd /home/linke/nano-vllm-moe
    rm -rf results/acc_predictor_bench
    CUDA_VISIBLE_DEVICES=2 python scripts/bench_acceptance_predictor.py \
        --output-dir results/acc_predictor_bench15 \
        --acceptance-predictor-path random_cache_srdp_scripts-1/res/run_20260614_133025 \
        --gpu-memory-utilization 0.99 \
        --cache-ratios 0.3125 \
        --output-lens 512 \
        --draft-alpha-stop-threshold 0.89 \
        --max-draft-tokens-values 15 \
        --segment-sizes 12 \
        --predictor-modes on \
        --kt-num-threads 16
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
DEFAULT_PREDICTOR_PATH = "random_cache_srdp_scripts-1/res/run_20260614_133025"


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


def _parse_predictor_modes(values: str) -> list[bool]:
    modes = []
    for item in values.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item in {"on", "true", "1", "yes"}:
            modes.append(True)
        elif item in {"off", "false", "0", "no"}:
            modes.append(False)
        else:
            raise argparse.ArgumentTypeError(f"invalid predictor mode: {item}")
    if not modes:
        raise argparse.ArgumentTypeError("--predictor-modes must list at least one of on/off")
    return modes


def build_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    predictor_modes = _parse_predictor_modes(args.predictor_modes)
    cases: list[dict[str, Any]] = []
    for output_len in _parse_csv(args.output_lens, int):
        for cache_ratio in _parse_csv(args.cache_ratios, float):
            for max_draft_tokens in _parse_csv(args.max_draft_tokens_values, int):
                for segment_size in _parse_csv(args.segment_sizes, int):
                    for repeat in range(int(args.repeats)):
                        for predictor_enabled in predictor_modes:
                            cases.append(
                                {
                                    "predictor_enabled": bool(predictor_enabled),
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
    pred = "predON" if case["predictor_enabled"] else "predOFF"
    return (
        f"{pred}_seg{int(case['segment_size'])}_"
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
    draft_forward_count = max(1, draft_seg_replays // num_segments) if draft_seg_replays else 1

    pred_draft_seg_submit = int(prefetch.get("draft_segment_indexed_submit_count", 0) or 0)
    pred_draft_live_submit = int(
        engine_profile.get("model_draft_live_prefetch_submit_count", 0) or 0
    )
    pred_phase1_submit = int(prefetch.get("predictive_phase1_submit_count", 0) or 0)
    pred_verify_seg_submit = int(prefetch.get("verify_segment_prefetch_submit_count", 0) or 0)
    draft_phase_submit = pred_draft_seg_submit + pred_draft_live_submit + pred_phase1_submit
    verify_phase_submit = pred_verify_seg_submit

    return {
        "name": _case_name(case),
        "predictor_enabled": bool(case["predictor_enabled"]),
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
        "avg_active_per_layer": float(cache.get("avg_active_per_layer", 0.0) or 0.0),
        "draft_forward_count": draft_forward_count,
        "verify_calls": verify_calls,
        "verify_segment_graph_replays": int(
            cuda_graph.get("verify_kt_hybrid_segment_graph_replay_count", 0) or 0
        ),
        # -- per-phase prefetch --
        "draft_phase_submit": draft_phase_submit,
        "verify_phase_submit": verify_phase_submit,
        "draft_prefetch_per_forward": float(draft_phase_submit / draft_forward_count),
        "verify_prefetch_per_forward": (
            float(verify_phase_submit / verify_calls) if verify_calls else 0.0
        ),
        # -- total prefetch --
        "prefetch_submit_count": int(prefetch.get("submit_count", 0) or 0),
        "prefetch_completed_count": int(prefetch.get("completed_count", 0) or 0),
        "prefetch_late_count": int(engine_profile.get("model_prefetch_late_count", 0) or 0),
        "prefetch_publish_count": int(engine_profile.get("model_publish_count", 0) or 0),
        "prefetch_consumed_count": int(prefetch.get("consumed_count", 0) or 0),
        # -- predictive-specific breakdown --
        "pred_draft_seg_submit": pred_draft_seg_submit,
        "pred_draft_live_submit": pred_draft_live_submit,
        "pred_phase1_submit": pred_phase1_submit,
        "pred_verify_seg_submit": pred_verify_seg_submit,
        # -- acceptance predictor outputs --
        "predicted_alpha_avg": float(acceptance.get("predicted_alpha_avg", 0.0) or 0.0),
        "predicted_alpha_count": int(acceptance.get("predicted_alpha_count", 0) or 0),
        "predicted_alpha_min": float(acceptance.get("predicted_alpha_min", 0.0) or 0.0),
        "predicted_alpha_max": float(acceptance.get("predicted_alpha_max", 0.0) or 0.0),
        "predicted_alpha_position": acceptance.get("predicted_alpha_position", []),
        "outputs_digest": str(raw.get("outputs_digest", "")),
    }


def _comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, float, int, int, int], dict[bool, dict[str, Any]]] = {}
    for row in rows:
        key = (
            int(row["output_len"]),
            round(float(row["cache_ratio"]), 6),
            int(row["max_draft_tokens"]),
            int(row["segment_size"]),
            int(row["repeat"]),
        )
        grouped.setdefault(key, {})[bool(row["predictor_enabled"])] = row

    comparisons: list[dict[str, Any]] = []
    for pair in grouped.values():
        on = pair.get(True)
        off = pair.get(False)
        if on is None or off is None:
            continue
        on_tps = float(on["throughput_output_tok_s"])
        off_tps = float(off["throughput_output_tok_s"])
        comparisons.append(
            {
                "output_len": int(on["output_len"]),
                "cache_ratio": float(on["cache_ratio"]),
                "max_draft_tokens": int(on["max_draft_tokens"]),
                "segment_size": int(on["segment_size"]),
                "repeat": int(on["repeat"]),
                "digest_match": on["outputs_digest"] == off["outputs_digest"],
                "predicted_alpha_avg": float(on["predicted_alpha_avg"]),
                "measured_acceptance_rate": float(on["acceptance_rate"]),
                "alpha_vs_acceptance_delta": float(on["predicted_alpha_avg"]) - float(on["acceptance_rate"]),
                "throughput_on": on_tps,
                "throughput_off": off_tps,
                "throughput_delta": on_tps - off_tps,
                "throughput_overhead_pct": (
                    100.0 * (off_tps - on_tps) / off_tps if off_tps > 0.0 else 0.0
                ),
                "draft_ms_on": float(on["draft_forward_ms_avg"]),
                "draft_ms_off": float(off["draft_forward_ms_avg"]),
                "draft_ms_delta": float(on["draft_forward_ms_avg"]) - float(off["draft_forward_ms_avg"]),
                "acceptance_delta": float(on["acceptance_rate"]) - float(off["acceptance_rate"]),
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
    predictor_enabled = bool(case["predictor_enabled"])
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
        # -- acceptance predictor --
        "--acceptance-predictor-enabled",
        "true" if predictor_enabled else "false",
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
        # -- predictive segment-indexed prefetch (required for the draft tail graph) --
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
        f"decode_tok/s={row['decode_phase_output_tok_s']:.3f} "
        f"draft_ms={row['draft_forward_ms_avg']:.3f} "
        f"verify_ms={row['verify_forward_ms_avg']:.3f} "
        f"hit={row['route_hit_rate']:.4f} accept={row['acceptance_rate']:.4f} "
        f"miss/L={row['avg_miss_routes_per_layer']:.2f} "
        f"active/L={row['avg_active_per_layer']:.2f}",
        flush=True,
    )
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
    if row["predictor_enabled"]:
        print(
            f"  predictor: predict_alpha_avg={row['predicted_alpha_avg']:.4f} "
            f"count={row['predicted_alpha_count']} "
            f"min={row['predicted_alpha_min']:.4f} max={row['predicted_alpha_max']:.4f} "
            f"measured_accept={row['acceptance_rate']:.4f}",
            flush=True,
        )
    else:
        print("  predictor: disabled (baseline for overhead comparison)", flush=True)
    return row


def write_markdown_report(summary: dict[str, Any], path: Path) -> None:
    metadata = summary["metadata"]
    rows = summary["rows"]
    comparisons = summary["comparisons"]
    lines = [
        "# Acceptance Predictor Benchmark",
        "",
        f"- timestamp: `{metadata['timestamp']}`",
        f"- model: `{metadata['model_path']}`",
        f"- predictor path: `{metadata['acceptance_predictor_path']}`",
        f"- predictor modes: `{', '.join('on' if m else 'off' for m in metadata['predictor_modes'])}`",
        f"- segment sizes: `{', '.join(str(x) for x in metadata['segment_sizes'])}`",
        f"- output directory: `{metadata['output_dir']}`",
        "",
        "## Cases",
        "",
        "| predictor | seg | out | ratio | K | rep | tok/s | draft ms | verify ms | hit | accept | predict_alpha_avg | submit | publish | late | draft/fwd | verify/fwd |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        alpha_str = f"{row['predicted_alpha_avg']:.4f}" if row["predictor_enabled"] else "-"
        lines.append(
            "| "
            f"{'on' if row['predictor_enabled'] else 'off'} | {row['segment_size']} | "
            f"{row['output_len']} | {row['cache_ratio']:.4f} | {row['max_draft_tokens']} | "
            f"{row['repeat']} | {row['throughput_output_tok_s']:.3f} | "
            f"{row['draft_forward_ms_avg']:.3f} | {row['verify_forward_ms_avg']:.3f} | "
            f"{row['route_hit_rate']:.4f} | {row['acceptance_rate']:.4f} | {alpha_str} | "
            f"{row['prefetch_submit_count']} | {row['prefetch_publish_count']} | "
            f"{row['prefetch_late_count']} | {row['draft_prefetch_per_forward']:.1f} | "
            f"{row['verify_prefetch_per_forward']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Predictor on vs off (overhead + prediction quality)",
            "",
            "| seg | out | ratio | K | rep | digest | tok/s on | tok/s off | overhead % | draft ms delta | predict_alpha_avg | measured accept | alpha-accept delta |",
            "|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            "| "
            f"{row['segment_size']} | {row['output_len']} | {row['cache_ratio']:.4f} | "
            f"{row['max_draft_tokens']} | {row['repeat']} | "
            f"{'match' if row['digest_match'] else 'DIFF'} | "
            f"{row['throughput_on']:.3f} | {row['throughput_off']:.3f} | "
            f"{row['throughput_overhead_pct']:+.2f} | {row['draft_ms_delta']:+.3f} | "
            f"{row['predicted_alpha_avg']:.4f} | {row['measured_acceptance_rate']:.4f} | "
            f"{row['alpha_vs_acceptance_delta']:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `predict_alpha_avg` is the mean predicted theoretical acceptance alpha over all draft sub-steps.",
            "- `overhead %` = throughput reduction from enabling the predictor (predictor-off is the baseline).",
            "- `digest` should be `match`: the predictor must not change generated tokens (it only observes).",
            "- `alpha-accept delta` compares predicted alpha against the measured acceptance rate; a large gap",
            "  indicates distribution shift between the predictor's training mechanism and the live draft.",
            "- All other columns mirror `bench_dual_queue_prefetch.py` for direct comparison.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = output_dir / "acc_predictor_prompt.txt"
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

    predictor_modes = _parse_predictor_modes(args.predictor_modes)
    summary = {
        "metadata": {
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "model_path": args.model_path,
            "profile_artifact": args.profile_artifact,
            "acceptance_predictor_path": args.acceptance_predictor_path,
            "acceptance_predictor_step_horizon": int(args.acceptance_predictor_step_horizon),
            "output_dir": str(output_dir),
            "predictor_modes": predictor_modes,
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

    # Aggregate predicted-alpha summary across all predictor-on rows.
    on_rows = [r for r in rows if r["predictor_enabled"] and r["predicted_alpha_count"] > 0]
    if on_rows:
        total_count = sum(r["predicted_alpha_count"] for r in on_rows)
        weighted = sum(r["predicted_alpha_avg"] * r["predicted_alpha_count"] for r in on_rows)
        predict_alpha_avg = weighted / total_count if total_count else 0.0
        print(f"predict_alpha_avg={predict_alpha_avg:.4f} (over {total_count} draft sub-steps)")
    print(f"summary_json={summary_json}")
    print(f"summary_md={summary_md}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the on-GPU acceptance predictor (on vs off) in the draft segment-graph path."
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--profile-artifact", default=DEFAULT_PROFILE)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-doc", default="")
    parser.add_argument("--predictor-modes", default="on,off")
    parser.add_argument("--acceptance-predictor-path", default=DEFAULT_PREDICTOR_PATH)
    parser.add_argument("--acceptance-predictor-step-horizon", type=int, default=32)
    parser.add_argument("--draft-alpha-stop-threshold", type=float, default=-1.0)
    parser.add_argument("--draft-stop-policy", choices=["none", "alpha_threshold", "tpot"], default="tpot")
    parser.add_argument("--draft-tpot-td-ms", type=float, default=19.0)
    parser.add_argument("--draft-tpot-tv-ms", type=float, default=80.0)
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

    parser.add_argument("--prefetch-step-budget", type=int, default=16)
    parser.add_argument("--prefetch-max-inflight", type=int, default=16)
    parser.add_argument("--prefetch-transfer-stream-count", type=int, default=1)
    parser.add_argument("--prefetch-staging-slots-per-layer", type=int, default=2)
    parser.add_argument("--prefetch-metadata-host-buffer-pool-size", type=int, default=3)
    parser.add_argument("--prefetch-global-queue-capacity", type=int, default=4096)
    parser.add_argument("--prefetch-verify-layer-max-budget", type=int, default=8)
    parser.add_argument("--prefetch-verify-attention-ratio", type=float, default=1.0)
    parser.add_argument("--cache-eviction-budget-per-step", type=int, default=2)
    parser.add_argument("--draft-segment-host-buffer-pool-size", type=int, default=0)
    parser.add_argument("--draft-prefetch-visible-budget-ms", type=float, default=3.0)
    parser.add_argument("--draft-prefetch-max-per-boundary", type=int, default=16)
    parser.add_argument("--verify-prefetch-visible-budget-ms", type=float, default=12.0)
    parser.add_argument("--verify-prefetch-max-per-boundary", type=int, default=4)

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
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.99)
    parser.add_argument("--verify-cuda-graph-bucket-steps", default="3,5,8,12")
    parser.add_argument("--dist-port-base", type=int, default=30700)
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
