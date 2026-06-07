#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

"""
TS=$(date +%Y%m%d_%H%M%S)
LOG=/home/mumura/moe_spec/logs/tmp_vpre_${TS}.log
srun --jobid=30169 --ntasks=1 bash -s <<EOS 2>&1 | tee "$LOG"
source /opt/Software/Anaconda3/etc/profile.d/conda.sh
conda activate nano_moe
cd /home/mumura/moe_spec/nano-vllm-moe/
python - <<'PY'
import torch, nanovllm
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
PY
python scripts/small_bench.py \
    --output-dir results/tmp_vpre \
    --runtime-kinds legacy,predictive \
    --output-lens 256 \
    --cache-ratios 0.25,0.75 \
    --verify-cuda-graph-values true \
    --cache-strategies lru,lfu \
    --max-draft-tokens-values 8 \
    --max-model-len 8192 \
    --prefetch-verify-attention-ratio 1.0 \
    --prefetch-step-budget 8 \
    --prefetch-verify-layer-max-budget 8 \
    --prefetch-max-inflight 16
EOS
"""

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
MODEL_PATH = "/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B"


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


def max_draft_tokens_for_ratio(ratio: float) -> int:
    rounded = round(float(ratio), 2)
    # 配置表：[(比率阈值, 返回值), ...]
    ratio_config = [
        (0.2375, 1),
        (0.25, 2),
        (0.45, 4),
        (0.50, 6),
        (0.75, 8),
        (float('inf'), 10)  # 默认值
    ]
    
    for threshold, value in ratio_config:
        if rounded <= threshold:
            return value
    
    return 10


def assess_text_quality(text: str) -> dict[str, Any]:
    reasons: list[str] = []
    if not text.strip():
        reasons.append("empty")
    if "\ufffd" in text:
        reasons.append("replacement_char")
    if any((ord(ch) < 32 and ch not in "\n\r\t") for ch in text):
        reasons.append("control_char")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 4:
        most_common = max(lines.count(line) for line in set(lines))
        if most_common / len(lines) >= 0.5:
            reasons.append("repeated_lines")

    words = text.lower().split()
    if len(words) >= 80:
        window = 12
        grams = [" ".join(words[i : i + window]) for i in range(0, len(words) - window + 1)]
        if grams:
            max_repeat = max(grams.count(gram) for gram in set(grams))
            if max_repeat >= 3:
                reasons.append("repeated_12gram")

    return {"ok": not reasons, "reasons": reasons}


def _vg_values(args: argparse.Namespace) -> list[bool]:
    if args.verify_cuda_graph_values:
        return _parse_csv(args.verify_cuda_graph_values, str2bool)
    return [bool(args.verify_cuda_graph)]


def build_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    vg_values = _vg_values(args)
    cache_strategies = _parse_csv(args.cache_strategies, str)
    if args.lightweight:
        runtime_kinds = _parse_csv(args.runtime_kinds, str)
        return [
            {
                "runtime_kind": runtime_kind,
                "output_len": int(args.lightweight_output_len),
                "cache_ratio": float(args.lightweight_cache_ratio),
                "max_draft_tokens": int(args.lightweight_max_draft_tokens),
                "verify_cuda_graph": bool(vg),
                "cache_strategy": strategy,
            }
            for runtime_kind in runtime_kinds
            for vg in vg_values
            for strategy in cache_strategies
        ]

    cases: list[dict[str, Any]] = []
    mdt_values = _parse_csv(args.max_draft_tokens_values, int) if args.max_draft_tokens_values else []
    for runtime_kind in _parse_csv(args.runtime_kinds, str):
        for output_len in _parse_csv(args.output_lens, int):
            for ratio in _parse_csv(args.cache_ratios, float):
                mdts = mdt_values if mdt_values else [max_draft_tokens_for_ratio(float(ratio))]
                for mdt in mdts:
                    for vg in vg_values:
                        for strategy in cache_strategies:
                            cases.append(
                                {
                                    "runtime_kind": runtime_kind,
                                    "output_len": int(output_len),
                                    "cache_ratio": float(ratio),
                                    "max_draft_tokens": int(mdt),
                                    "verify_cuda_graph": bool(vg),
                                    "cache_strategy": str(strategy),
                                }
                            )
    return cases


def _case_name(case: dict[str, Any]) -> str:
    ratio_pct = int(round(float(case["cache_ratio"]) * 100))
    vg = "1" if case.get("verify_cuda_graph", True) else "0"
    strategy = case.get("cache_strategy", "lru")
    return (
        f"{case['runtime_kind']}_ratio{ratio_pct}_"
        f"l{int(case['output_len'])}_k{int(case['max_draft_tokens'])}_vg{vg}_{strategy}"
    )


def _row_from_raw(case: dict[str, Any], raw: dict[str, Any], quality: dict[str, Any], elapsed_sec: float) -> dict[str, Any]:
    summary = raw.get("summary", {})
    acceptance = summary.get("acceptance", {})
    cache = summary.get("cache", {})
    prefetch = summary.get("prefetch", {})
    cuda_graph = summary.get("cuda_graph", {})
    m3 = summary.get("m3", {})
    verify_cache_fill = summary.get("verify_cache_fill", {})
    return {
        "name": _case_name(case),
        "runtime_kind": case["runtime_kind"],
        "output_len": int(case["output_len"]),
        "cache_ratio": float(case["cache_ratio"]),
        "cache_strategy": str(case.get("cache_strategy", "lru")),
        "max_draft_tokens": int(case["max_draft_tokens"]),
        "elapsed_sec": float(elapsed_sec),
        "generated_output_tokens": int(raw.get("generated_output_tokens", 0) or 0),
        "acceptance_rate": float(acceptance.get("acceptance_rate", 0.0) or 0.0),
        "route_hit_rate": float(cache.get("true_route_hit_rate", cache.get("route_hit_rate", 0.0)) or 0.0),
        "route_hit_rate_post_transfer": float(cache.get("route_hit_rate", 0.0) or 0.0),
        "avg_miss_per_layer": float(cache.get("avg_miss_per_layer", 0.0) or 0.0),
        "avg_active_per_layer": float(cache.get("avg_active_per_layer", 0.0) or 0.0),
        "weight_hit_rate": float(cache.get("weight_hit_rate", 0.0) or 0.0),
        "throughput_output_tok_s": float(summary.get("throughput_output_tok_s", 0.0) or 0.0),
        "decode_phase_output_tok_s": float(summary.get("decode_phase_output_tok_s", 0.0) or 0.0),
        "draft_forward_ms_avg": float(summary.get("draft_forward_ms_avg", 0.0) or 0.0),
        "verify_forward_ms_avg": float(summary.get("verify_forward_ms_avg", 0.0) or 0.0),
        "verify_cuda_graph": bool(cuda_graph.get("verify_enabled", False)),
        "verify_graph_call_count": int(cuda_graph.get("verify_call_count", 0) or 0),
        "verify_prefix_graph_replay_count": int(cuda_graph.get("verify_prefix_replay_count", 0) or 0),
        "verify_prefix_graph_fallback_count": int(cuda_graph.get("verify_prefix_fallback_count", 0) or 0),
        "draft_graph_replay_count": int(cuda_graph.get("draft_replay_count", 0) or 0),
        "graph_hit_rate": float(cuda_graph.get("hit_rate", 0.0) or 0.0),
        "prefetch_submit_count": int(prefetch.get("submit_count", 0) or 0),
        "prefetch_completed_count": int(prefetch.get("completed_count", 0) or 0),
        "prefetch_consumed_count": int(prefetch.get("consumed_count", 0) or 0),
        "draft_segment_indexed_submit_count": int(prefetch.get("draft_segment_indexed_submit_count", 0) or 0),
        "draft_segment_indexed_consumed_count": int(prefetch.get("draft_segment_indexed_consumed_count", 0) or 0),
        "predictive_phase1_submit_count": int(prefetch.get("predictive_phase1_submit_count", 0) or 0),
        "verify_layer_submit_count": int(prefetch.get("verify_layer_submit_count", 0) or 0),
        "verify_layer_consumed_count": int(prefetch.get("verify_layer_consumed_count", 0) or 0),
        "verify_cache_fill_promoted_expert_count": int(verify_cache_fill.get("promoted_expert_count", 0) or 0),
        "perfect_fraction": float(m3.get("perfect_fraction", 0.0) or 0.0),
        "step0_perfect_fraction": float(m3.get("step0_perfect_fraction", 0.0) or 0.0),
        "m3_group_count": int(m3.get("group_count", 0) or 0),
        "m3_step0_group_count": int(m3.get("step0_group_count", 0) or 0),
        "outputs_digest": str(raw.get("outputs_digest", "")),
        "text_quality_ok": bool(quality.get("ok", False)),
        "text_quality_reasons": list(quality.get("reasons", [])),
    }


def run_case(args: argparse.Namespace, repo_root: Path, prompt_file: Path, case: dict[str, Any], case_index: int) -> dict[str, Any]:
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
        "--model-path",
        args.model_path,
        "--prompt-text-file",
        str(prompt_file),
        "--cache-ratio",
        str(case["cache_ratio"]),
        "--slots-per-layer",
        "0",
        "--prefetch-enabled",
        "true",
        "--prefetch-runtime-mode",
        "draft_segment_indexed",
        "--prefetch-runtime-kind",
        str(case["runtime_kind"]),
        "--prefetch-verify-attention-ratio",
        str(args.prefetch_verify_attention_ratio),
        "--predictive-phase1-budget",
        str(args.predictive_phase1_budget),
        "--output",
        str(case_json),
        "--dist-port",
        str(args.dist_port_base + case_index),
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
        "entropy_cache_bias",
        "--draft-reroute-artifact",
        args.profile_artifact,
        "--temperature",
        str(args.temperature),
        "--acceptance-strategy",
        args.acceptance_strategy,
        "--acceptance-threshold",
        str(args.acceptance_threshold),
        "--spec-verify-miss-policy",
        args.spec_verify_miss_policy,
        "--cache-strategy",
        str(case.get("cache_strategy", "lru")),
        "--cpu-expert-backend",
        args.cpu_expert_backend,
        "--cpu-expert-pin-memory",
        str(args.cpu_expert_pin_memory).lower(),
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
        "--draft-cuda-graph-enabled",
        "true",
        "--draft-cuda-graph-cpu-backend",
        "none",
        "--verify-cuda-graph",
        str(case.get("verify_cuda_graph", args.verify_cuda_graph)).lower(),
        "--verify-cuda-graph-bucket-steps",
        args.verify_cuda_graph_bucket_steps,
        "--prefetch-verify-wait-ms",
        str(args.prefetch_verify_wait_ms),
        "--prefetch-step-budget",
        str(args.prefetch_step_budget),
        "--prefetch-verify-layer-max-budget",
        str(args.prefetch_verify_layer_max_budget),
        "--prefetch-max-inflight",
        str(args.prefetch_max_inflight),
        "--prefetch-staging-slots-per-layer",
        str(args.prefetch_staging_slots_per_layer),
        "--cache-eviction-budget-per-step",
        str(args.cache_eviction_budget_per_step),
        "--prefetch-global-queue-capacity",
        str(args.prefetch_global_queue_capacity),
        "--prefetch-use-prefill-history",
        "true",
        "--prefetch-use-verify-history",
        "true",
        "--prefetch-use-draft-live",
        "true",
        "--seed",
        str(args.seed),
        "--sync-layer-timing",
        str(args.sync_layer_timing).lower(),
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
    text = "\n".join(str(x) for x in raw.get("generated_text", []))
    quality = assess_text_quality(text)
    row = _row_from_raw(case, raw, quality, elapsed)
    row["case_verify_cuda_graph"] = bool(case.get("verify_cuda_graph", args.verify_cuda_graph))
    print(
        f"  accept={row['acceptance_rate']:.4f} hit={row['route_hit_rate']:.4f} "
        f"post_xfer_hit={row['route_hit_rate_post_transfer']:.4f} "
        f"miss/layer={row['avg_miss_per_layer']:.2f} active/layer={row['avg_active_per_layer']:.2f} "
        f"tok/s={row['throughput_output_tok_s']:.3f} draft_ms={row['draft_forward_ms_avg']:.3f} "
        f"verify_ms={row['verify_forward_ms_avg']:.3f} graph={row['draft_graph_replay_count']} "
        f"perfect={row['perfect_fraction']:.4f} step0={row['step0_perfect_fraction']:.4f} "
        f"text_ok={row['text_quality_ok']}",
        flush=True,
    )
    return row


def _delta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (int(row["output_len"]), round(float(row["cache_ratio"]), 4), str(row["runtime_kind"]), str(row.get("cache_strategy", "lru"))): row
        for row in rows
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["runtime_kind"] != "predictive":
            continue
        strategy = str(row.get("cache_strategy", "lru"))
        key = (int(row["output_len"]), round(float(row["cache_ratio"]), 4), "legacy", strategy)
        legacy = by_key.get(key)
        if legacy is None:
            continue
        out.append(
            {
                "output_len": int(row["output_len"]),
                "cache_ratio": float(row["cache_ratio"]),
                "cache_strategy": str(row.get("cache_strategy", "lru")),
                "acceptance_delta": float(row["acceptance_rate"]) - float(legacy["acceptance_rate"]),
                "route_hit_delta": float(row["route_hit_rate"]) - float(legacy["route_hit_rate"]),
                "avg_miss_per_layer_delta": float(row["avg_miss_per_layer"]) - float(legacy["avg_miss_per_layer"]),
                "throughput_delta": float(row["throughput_output_tok_s"]) - float(legacy["throughput_output_tok_s"]),
                "draft_forward_ms_delta": float(row["draft_forward_ms_avg"]) - float(legacy["draft_forward_ms_avg"]),
                "verify_forward_ms_delta": float(row["verify_forward_ms_avg"]) - float(legacy["verify_forward_ms_avg"]),
                "perfect_fraction_delta": float(row["perfect_fraction"]) - float(legacy["perfect_fraction"]),
                "step0_perfect_fraction_delta": (
                    float(row["step0_perfect_fraction"]) - float(legacy["step0_perfect_fraction"])
                ),
            }
        )
    return out


def _delta_rows_verify_graph(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare verify_cuda_graph=True vs False for each (output_len, cache_ratio, runtime_kind, cache_strategy)."""
    by_key: dict[tuple, dict[bool, dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["output_len"]), round(float(row["cache_ratio"]), 4), str(row["runtime_kind"]), str(row.get("cache_strategy", "lru")))
        vg = bool(row.get("case_verify_cuda_graph", True))
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
                "runtime_kind": str(row_on["runtime_kind"]),
                "cache_strategy": str(row_on.get("cache_strategy", "lru")),
                "acceptance_delta": float(row_on["acceptance_rate"]) - float(row_off["acceptance_rate"]),
                "route_hit_delta": float(row_on["route_hit_rate"]) - float(row_off["route_hit_rate"]),
                "avg_miss_per_layer_delta": float(row_on["avg_miss_per_layer"]) - float(row_off["avg_miss_per_layer"]),
                "throughput_delta": float(row_on["throughput_output_tok_s"]) - float(row_off["throughput_output_tok_s"]),
                "draft_forward_ms_delta": float(row_on["draft_forward_ms_avg"]) - float(row_off["draft_forward_ms_avg"]),
                "verify_forward_ms_delta": float(row_on["verify_forward_ms_avg"]) - float(row_off["verify_forward_ms_avg"]),
                "graph_hit_rate_delta": float(row_on["graph_hit_rate"]) - float(row_off["graph_hit_rate"]),
                "perfect_fraction_delta": float(row_on["perfect_fraction"]) - float(row_off["perfect_fraction"]),
                "step0_perfect_fraction_delta": (
                    float(row_on["step0_perfect_fraction"]) - float(row_off["step0_perfect_fraction"])
                ),
            }
        )
    return out


def write_markdown_report(
    *,
    summary: dict[str, Any],
    path: Path,
    status: str,
    slurm_job_id: str = "",
    eta_text: str = "",
) -> None:
    rows = summary.get("rows", [])
    deltas = summary.get("deltas", [])
    vg_deltas = summary.get("vg_deltas", [])
    metadata = summary.get("metadata", {})
    lines = [
        "# Predictive Prefetch Validation Report",
        "",
        f"Date: {time.strftime('%Y-%m-%d')}",
        "",
        "## 1. Experiment Design",
        "",
        "### 1.1 Objectives",
        "",
        "1. Validate `prefetch_runtime_kind=predictive` against legacy segment-indexed prefetch.",
        "2. Confirm generated text quality under the meaningful reroute prompt.",
        "3. Compare acceptance rate, cache hit rate, throughput, draft/verify forward time, graph replays, and prefetch counters.",
        "4. Report M3-style `perfect_fraction` and `step0_perfect_fraction`.",
        "",
        "### 1.2 Test Matrix",
        "",
        "| Dimension | Values |",
        "|---|---|",
        f"| Runtime kinds | `{', '.join(metadata.get('runtime_kinds', []))}` |",
        f"| Cache strategies | `{', '.join(metadata.get('cache_strategies', []))}` |",
        f"| Cache ratios | `{', '.join(str(x) for x in metadata.get('cache_ratios', []))}` |",
        f"| Output lengths | `{', '.join(str(x) for x in metadata.get('output_lens', []))}` |",
        f"| max_draft_tokens | {metadata.get('max_draft_tokens_desc', 'derived from cache_ratio via max_draft_tokens_for_ratio()')} |",
        "| Reroute policy | `entropy_cache_bias` |",
        "| Prefetch mode | `draft_segment_indexed` |",
        f"| Expert cache strategy | `{', '.join(metadata.get('cache_strategies', ['lru']))}` |",
        f"| Verify miss policy | `{metadata.get('spec_verify_miss_policy', '')}` |",
        f"| Verify CUDA graph | `{metadata.get('verify_cuda_graph', False)}` |",
        f"| Verify graph buckets | `{', '.join(str(x) for x in metadata.get('verify_cuda_graph_bucket_steps', []))}` |",
        "",
        "### 1.3 Common Settings",
        "",
        "```",
        f"Model:        {metadata.get('model_path', '')}",
        f"Profile:      {metadata.get('profile_artifact', '')}",
        "num_seqs:     1",
        f"max_model_len:{metadata.get('max_model_len', '')}",
        "draft_top_c:  0",
        "draft graph:  enabled",
        f"verify graph: {metadata.get('verify_cuda_graph', False)}",
        f"CPU backend:  {metadata.get('cpu_expert_backend', '')}",
        f"workspace:    cpu_expert_workspace_max_routes={metadata.get('cpu_expert_workspace_max_routes', '')}",
        f"kt direct:    backend={metadata.get('kt_direct_backend', '')}, "
        f"threads={metadata.get('kt_num_threads', '')}, "
        f"pools={metadata.get('kt_threadpool_count', '')}, "
        f"numa={metadata.get('kt_numa_nodes', '')}, "
        f"capture_bs={metadata.get('kt_capture_bs', '')}",
        "```",
        "",
        "Metric definitions:",
        "",
        "- `true hit`: cache hit rate measured BEFORE `cache_fill` transfers miss experts into cache slots.",
        "- `post-xfer hit`: cache hit rate measured AFTER transfers (legacy metric, artificially high).",
        "- `miss/layer`: average number of miss routes per MoE layer forward (pre-transfer).",
        "- `active/layer`: average number of active routes per MoE layer forward (= tokens * top_k).",
        "- `perfect_fraction`: fraction of draft forwards where every recorded MoE layer had zero CPU experts.",
        "- `step0_perfect_fraction`: same metric restricted to the first draft forward of each speculative step.",
        "- `prefetch submit/done/used`: benchmark `submit_count/completed_count/consumed_count`.",
        "",
        "### 1.4 Run Status",
        "",
        f"- status: `{status}`",
        f"- slurm_job_id: `{slurm_job_id}`",
        f"- eta: `{eta_text}`",
        f"- output_dir: `{metadata.get('output_dir', '')}`",
        "",
        "---",
        "",
        "## 2. Code Review And Fixes",
        "",
        "Implemented fixes before benchmarking:",
        "",
        "1. Same-round async draft metadata remains valid after `end_draft_iteration` until the next draft round begins.",
        "2. Predictive Phase 2 maps segment-0 frontier prefetches to the last segment, matching Part B.",
        "3. Failed predictive reservations roll back temporary round-protection entries.",
        "4. Benchmark CLIs now expose predictive runtime knobs and preserve them in result metadata.",
        "5. Benchmark summaries include predictive prefetch counters and M3 perfect-fraction metrics.",
        "",
        "## 3. Results",
        "",
        "| kind | out | ratio | strategy | K | accept | true hit | post-xfer hit | miss/layer | active/layer | weight hit | output tok/s | draft ms | verify ms | verify graph | verify replay/fallback | draft graph | prefetch submit/done/used | phase1 | verify-layer | perfect | step0 perfect | text |",
        "|:---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---|---:|:---|---:|---:|---:|---:|:---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['runtime_kind']} | "
            f"{row['output_len']} | "
            f"{row['cache_ratio']:.2f} | "
            f"{row.get('cache_strategy', 'lru')} | "
            f"{row['max_draft_tokens']} | "
            f"{row['acceptance_rate']:.4f} | "
            f"{row['route_hit_rate']:.4f} | "
            f"{row['route_hit_rate_post_transfer']:.4f} | "
            f"{row['avg_miss_per_layer']:.2f} | "
            f"{row['avg_active_per_layer']:.2f} | "
            f"{row['weight_hit_rate']:.4f} | "
            f"{row['throughput_output_tok_s']:.3f} | "
            f"{row['draft_forward_ms_avg']:.3f} | "
            f"{row['verify_forward_ms_avg']:.3f} | "
            f"{'on' if row['verify_cuda_graph'] else 'off'} | "
            f"{row['verify_graph_call_count']}/{row['verify_prefix_graph_replay_count']}/{row['verify_prefix_graph_fallback_count']} | "
            f"{row['draft_graph_replay_count']} | "
            f"{row['prefetch_submit_count']}/{row['prefetch_completed_count']}/{row['prefetch_consumed_count']} | "
            f"{row['predictive_phase1_submit_count']} | "
            f"{row['verify_layer_submit_count']}/{row['verify_layer_consumed_count']} | "
            f"{row['perfect_fraction']:.4f} | "
            f"{row['step0_perfect_fraction']:.4f} | "
            f"{'ok' if row['text_quality_ok'] else ','.join(row['text_quality_reasons'])} |"
        )

    lines.extend(
        [
            "",
            "## 4. Predictive Delta Versus Legacy",
            "",
            "Positive throughput, acceptance, cache hit, and perfect-fraction deltas are improvements. Negative draft/verify ms deltas are improvements.",
            "",
            "| out | ratio | strategy | accept delta | true-hit delta | miss/layer delta | tok/s delta | draft ms delta | verify ms delta | perfect delta | step0 perfect delta |",
            "|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in deltas:
        lines.append(
            "| "
            f"{row['output_len']} | "
            f"{row['cache_ratio']:.2f} | "
            f"{row.get('cache_strategy', 'lru')} | "
            f"{row['acceptance_delta']:+.4f} | "
            f"{row['route_hit_delta']:+.4f} | "
            f"{row['avg_miss_per_layer_delta']:+.2f} | "
            f"{row['throughput_delta']:+.3f} | "
            f"{row['draft_forward_ms_delta']:+.3f} | "
            f"{row['verify_forward_ms_delta']:+.3f} | "
            f"{row['perfect_fraction_delta']:+.4f} | "
            f"{row['step0_perfect_fraction_delta']:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## 5. Verify CUDA Graph Delta (On vs Off)",
            "",
            "Positive throughput and acceptance deltas mean `verify-cuda-graph=true` improves the metric. "
            "Negative draft/verify ms deltas mean the graph reduces forward latency.",
            "",
            "| kind | out | ratio | strategy | accept delta | true-hit delta | miss/layer delta | tok/s delta | draft ms delta | verify ms delta | graph-hit delta | perfect delta | step0 perfect delta |",
            "|:---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in vg_deltas:
        lines.append(
            "| "
            f"{row['runtime_kind']} | "
            f"{row['output_len']} | "
            f"{row['cache_ratio']:.2f} | "
            f"{row.get('cache_strategy', 'lru')} | "
            f"{row['acceptance_delta']:+.4f} | "
            f"{row['route_hit_delta']:+.4f} | "
            f"{row['avg_miss_per_layer_delta']:+.2f} | "
            f"{row['throughput_delta']:+.3f} | "
            f"{row['draft_forward_ms_delta']:+.3f} | "
            f"{row['verify_forward_ms_delta']:+.3f} | "
            f"{row['graph_hit_rate_delta']:+.4f} | "
            f"{row['perfect_fraction_delta']:+.4f} | "
            f"{row['step0_perfect_fraction_delta']:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## 6. Text Quality",
            "",
        ]
    )
    if rows:
        bad = [row for row in rows if not row["text_quality_ok"]]
        if bad:
            lines.append(f"{len(bad)} cases failed the text-quality guard.")
        else:
            lines.append("All completed cases passed the automated text-quality guard.")
    else:
        lines.append("No completed cases yet.")

    lines.extend(
        [
            "",
            "## 7. Conclusion",
            "",
        ]
    )
    if status == "completed" and rows:
        lines.append("The benchmark completed; use the tables above for the legacy-vs-predictive and verify-graph comparisons.")
    else:
        lines.append("The full matrix is still running or pending; this document will be updated by the batch job when results are complete.")

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
        "runtime_kinds": _parse_csv(args.runtime_kinds, str),
        "cache_strategies": _parse_csv(args.cache_strategies, str),
        "cache_ratios": sorted({float(c["cache_ratio"]) for c in cases}),
        "output_lens": sorted({int(case["output_len"]) for case in cases}),
        "cpu_expert_backend": args.cpu_expert_backend,
        "cpu_expert_workspace_max_routes": int(args.cpu_expert_workspace_max_routes),
        "kt_num_threads": int(args.kt_num_threads),
        "kt_threadpool_count": int(args.kt_threadpool_count),
        "kt_chunked_prefill_size": int(args.kt_chunked_prefill_size),
        "kt_direct_backend": args.kt_direct_backend,
        "kt_numa_nodes": args.kt_numa_nodes,
        "kt_capture_bs": args.kt_capture_bs,
        "spec_verify_miss_policy": args.spec_verify_miss_policy,
        "verify_cuda_graph": bool(args.verify_cuda_graph),
        "verify_cuda_graph_values": args.verify_cuda_graph_values,
        "verify_cuda_graph_bucket_steps": _parse_csv(args.verify_cuda_graph_bucket_steps, int),
        "max_draft_tokens_values": _parse_csv(args.max_draft_tokens_values, int) if args.max_draft_tokens_values else [],
        "max_draft_tokens_desc": (
            f"explicit sweep: {args.max_draft_tokens_values}"
            if args.max_draft_tokens_values
            else "derived from cache_ratio via max_draft_tokens_for_ratio()"
        ),
        "max_model_len": int(args.max_model_len),
        "argv": sys.argv,
    }
    summary = {
        "metadata": metadata,
        "rows": rows,
        "deltas": _delta_rows(rows),
        "vg_deltas": _delta_rows_verify_graph(rows),
    }
    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"
    summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    write_markdown_report(summary=summary, path=summary_md, status="completed", slurm_job_id=slurm_job_id)
    if args.report_doc:
        write_markdown_report(summary=summary, path=Path(args.report_doc), status="completed", slurm_job_id=slurm_job_id)
    print(f"summary_json={summary_json}")
    print(f"summary_md={summary_md}")
    if args.report_doc:
        print(f"report_doc={args.report_doc}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate legacy vs predictive prefetch on a meaningful reroute prompt.")
    p.add_argument("--model-path", default=MODEL_PATH)
    p.add_argument("--profile-artifact", default=DEFAULT_PROFILE)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--report-doc", default="")
    p.add_argument("--runtime-kinds", default="legacy,predictive")
    p.add_argument("--output-lens", default="128,512")
    p.add_argument("--cache-ratios", default="0.25,0.50,0.75")
    p.add_argument("--cache-strategies", default="lru",
                   help="CSV of expert cache strategies to sweep (e.g. lru,lfu,fifo).")
    p.add_argument("--cache-ratio", dest="cache_ratios", default=argparse.SUPPRESS)
    p.add_argument("--lightweight", type=str2bool, default=False)
    p.add_argument("--lightweight-output-len", type=int, default=64)
    p.add_argument("--lightweight-cache-ratio", type=float, default=0.50)
    p.add_argument("--lightweight-max-draft-tokens", type=int, default=6)
    p.add_argument("--max-draft-tokens-values", default="",
                   help="CSV of max_draft_tokens values to sweep. "
                        "When non-empty, each value is tested independently instead of "
                        "deriving max_draft_tokens from cache_ratio via max_draft_tokens_for_ratio().")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--acceptance-strategy", default="standard_sampling")
    p.add_argument("--acceptance-threshold", type=float, default=0.7)
    p.add_argument("--spec-verify-miss-policy", choices=["cpu", "cache_fill", "cache_fill_no_cpu"], default="cache_fill_no_cpu")
    p.add_argument("--cpu-expert-backend", default="fused")
    p.add_argument("--cpu-expert-pin-memory", type=str2bool, default=True)
    p.add_argument("--cpu-expert-workspace-max-routes", type=int, default=327680)
    p.add_argument("--cpu-expert-num-threads", type=int, default=4)
    p.add_argument("--kt-num-threads", type=int, default=0)
    p.add_argument("--kt-threadpool-count", type=int, default=1)
    p.add_argument("--kt-chunked-prefill-size", type=int, default=4096)
    p.add_argument(
        "--kt-direct-backend",
        choices=["auto", "amx_bf16", "avx2_bf16"],
        default="auto",
    )
    p.add_argument("--kt-numa-nodes", default="")
    p.add_argument("--kt-capture-bs", default="1,2,4,8,16,32")
    p.add_argument("--max-num-batched-tokens", type=int, default=16384)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--prefetch-verify-wait-ms", type=float, default=0.0)
    p.add_argument("--prefetch-step-budget", type=int, default=4)
    p.add_argument("--prefetch-verify-layer-max-budget", type=int, default=2)
    p.add_argument("--prefetch-max-inflight", type=int, default=8)
    p.add_argument("--prefetch-staging-slots-per-layer", type=int, default=2)
    p.add_argument("--cache-eviction-budget-per-step", type=int, default=2)
    p.add_argument("--prefetch-global-queue-capacity", type=int, default=4096)
    p.add_argument("--prefetch-verify-attention-ratio", type=float, default=0.3)
    p.add_argument("--predictive-phase1-budget", type=int, default=4)
    p.add_argument("--verify-cuda-graph", type=str2bool, default=True)
    p.add_argument("--verify-cuda-graph-values", default="",
                   help="CSV of true/false values to sweep for verify-cuda-graph. "
                        "When empty, the global --verify-cuda-graph is used for all cases.")
    p.add_argument("--verify-cuda-graph-bucket-steps", default="4,8,12,16")
    p.add_argument("--dist-port-base", type=int, default=29400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sync-layer-timing", type=str2bool, default=True)
    p.add_argument("--case-timeout-sec", type=int, default=2400)
    return p


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
