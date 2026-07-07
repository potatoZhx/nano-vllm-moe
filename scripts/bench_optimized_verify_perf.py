#!/usr/bin/env python3
"""Run the optimized verify performance benchmark.

This is a thin wrapper around ``scripts/bench_per_layer_slots.py`` so the
model path, profile-weighted slot allocation, runtime prefetch, kt_direct CPU
experts, and verify CUDA graph path stay identical to the per-layer slot
benchmark.  The defaults use the optimized verify knobs validated in
``docs/optimize_ops/verify_per_op_call_chain_breakdown.md``.

CUDA_VISIBLE_DEVICES=2 python scripts/bench_optimized_verify_perf.py \
    --output-dir results/optimized_verify_perf_k15_budget10_stopnone_l512 \
    --max-draft-tokens-values 15 \
    --verify-prefetch-max-per-boundary 10 \
    --fail-on-target-miss false \
    --verify-cuda-graph-bucket-steps 3,5,7,10,13,16 


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


DEFAULT_SLOT_PROFILE_CSV = (
    "pre_exps/exp_and_figs/unique/unique_count_plot_summary_n1024.csv"
)
DEFAULT_VERIFY_BUCKETS = "3,5,7,10,13"


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {value}")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _ep_float(engine_profile: dict[str, Any], key: str) -> float:
    value = engine_profile.get(key)
    if value is None:
        value = engine_profile.get(f"model_{key}", 0.0)
    return _as_float(value)


def _top_layers(row: dict[str, Any], limit: int = 8) -> str:
    parts: list[str] = []
    for item in row.get("verify_layer_cpu_expert_top", [])[:limit]:
        cpu_experts = _as_float(item.get("cpu_experts_per_call"))
        cpu_routes = _as_float(item.get("cpu_routes_per_call"))
        if cpu_experts <= 0.0 and cpu_routes <= 0.0:
            continue
        parts.append(
            f"L{_as_int(item.get('layer'))}:"
            f"{cpu_experts:.1f}exp/{cpu_routes:.1f}routes"
        )
    return ", ".join(parts)


def _verify_submitted_bytes(engine_profile: dict[str, Any]) -> float:
    by_source = engine_profile.get("prefetch_submitted_bytes_by_source")
    if by_source is None:
        by_source = engine_profile.get("model_prefetch_submitted_bytes_by_source")
    if isinstance(by_source, dict):
        return _as_float(by_source.get("verify_segment"))
    return _ep_float(engine_profile, "verify_segment_prefetch_submitted_bytes")


def build_bench_command(args: argparse.Namespace, extra_args: list[str]) -> list[str]:
    repo_root = Path(__file__).resolve().parents[1]
    bench_script = repo_root / "scripts" / "bench_per_layer_slots.py"
    cmd = [
        sys.executable,
        str(bench_script),
        "--output-dir",
        str(args.output_dir),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--cache-ratios",
        args.cache_ratios,
        "--output-lens",
        args.output_lens,
        "--max-draft-tokens-values",
        args.max_draft_tokens_values,
        "--segment-sizes",
        args.segment_sizes,
        "--allocation-modes",
        args.allocation_modes,
        "--slot-buckets",
        str(args.slot_buckets),
        "--slot-max-bucket-ratio",
        str(args.slot_max_bucket_ratio),
        "--slot-profile-csv",
        args.slot_profile_csv,
        "--kt-num-threads",
        str(args.kt_num_threads),
        "--verify-cuda-graph-bucket-steps",
        args.verify_cuda_graph_bucket_steps,
        "--verify-prefetch-max-per-boundary",
        str(args.verify_prefetch_max_per_boundary),
        "--verify-prefetch-visible-budget-ms",
        str(args.verify_prefetch_visible_budget_ms),
        "--repeats",
        str(args.repeats),
        "--dist-port-base",
        str(args.dist_port_base),
        "--case-timeout-sec",
        str(args.case_timeout_sec),
        "--skip-existing",
        str(args.skip_existing).lower(),
        "--fail-fast",
        str(args.fail_fast).lower(),
    ]
    if args.report_doc:
        cmd.extend(["--report-doc", args.report_doc])
    cmd.extend(extra_args)
    return cmd


def optimized_env(args: argparse.Namespace) -> tuple[dict[str, str], dict[str, str]]:
    env = os.environ.copy()
    overrides = {
        "NANOVLLM_VERIFY_PREFETCH_RANK_MULTIPLIER": str(
            args.verify_prefetch_rank_multiplier
        ),
        "NANOVLLM_VERIFY_DEFER_SEGMENT_METADATA": "1",
        "NANOVLLM_VERIFY_BOUNDARY_PREFETCH_ASYNC": "0",
        "NANOVLLM_VERIFY_SEGMENT_CUDA_EVENT_TIMING": (
            "1" if args.segment_event_timing else "0"
        ),
    }
    for key, value in overrides.items():
        env[key] = value

    if not args.allow_op_event_timing:
        env.pop("NANOVLLM_VERIFY_OP_EVENT_TIMING", None)
        env.pop("NANOVLLM_VERIFY_DEEP_PROFILE_SYNC", None)
        env.pop("NANOVLLM_VERIFY_BREAKDOWN_SYNC", None)

    if not args.preserve_conflicting_env:
        env.pop("NANOVLLM_VERIFY_DISABLE_RUNTIME_METADATA", None)
        env.pop("NANOVLLM_VERIFY_SKIP_METADATA_OFFLOAD", None)
        env.pop("NANOVLLM_VERIFY_SYNC_METADATA_PROFILE_READBACK", None)

    return env, overrides


def _load_raw_case(output_dir: Path, row: dict[str, Any]) -> dict[str, Any]:
    case_json = output_dir / f"{row['name']}.json"
    if not case_json.exists():
        return {}
    return json.loads(case_json.read_text(encoding="utf-8"))


def summarize_rows(
    summary: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in summary.get("rows", []):
        raw = _load_raw_case(output_dir, row)
        engine_profile = raw.get("engine_profile", {}) if isinstance(raw, dict) else {}
        verify_calls = max(1, _as_int(row.get("verify_calls"), 1))
        segment_event_ms = _ep_float(engine_profile, "verify_segment_cuda_event_ms")
        segment_event_per_call = (
            segment_event_ms / verify_calls if segment_event_ms > 0.0 else 0.0
        )
        verify_ms = _as_float(row.get("verify_forward_ms_avg"))
        graph_external_gap_ms = (
            max(0.0, verify_ms - segment_event_per_call)
            if segment_event_per_call > 0.0
            else 0.0
        )
        cpu_experts_per_call = sum(
            _as_float(v)
            for v in row.get("verify_layer_realized_cpu_expert_count_per_call", [])
        )
        cpu_routes_per_call = sum(
            _as_float(v) for v in row.get("verify_layer_cpu_routes_per_call", [])
        )
        boundary_ms = _ep_float(engine_profile, "verify_segment_boundary_submit_ms")
        rank_ms = _ep_float(engine_profile, "verify_segment_prefetch_rank_ms")
        transfer_enqueue_ms = _ep_float(
            engine_profile, "verify_segment_prefetch_transfer_enqueue_ms"
        )
        submitted = _ep_float(
            engine_profile, "verify_segment_prefetch_submit_count"
        )
        submitted_bytes = _verify_submitted_bytes(engine_profile)
        submitted_mb_per_call = submitted_bytes / 1_000_000.0 / verify_calls
        item = {
            "name": row.get("name", ""),
            "allocation_mode": row.get("allocation_mode", ""),
            "output_len": _as_int(row.get("output_len")),
            "cache_ratio": _as_float(row.get("cache_ratio")),
            "max_draft_tokens": _as_int(row.get("max_draft_tokens")),
            "segment_size": _as_int(row.get("segment_size")),
            "repeat": _as_int(row.get("repeat")),
            "verify_calls": verify_calls,
            "draft_forward_ms_avg": _as_float(row.get("draft_forward_ms_avg")),
            "verify_forward_ms_avg": verify_ms,
            "decode_phase_output_tok_s": _as_float(
                row.get("decode_phase_output_tok_s")
            ),
            "throughput_output_tok_s": _as_float(row.get("throughput_output_tok_s")),
            "acceptance_rate": _as_float(row.get("acceptance_rate")),
            "route_hit_rate": _as_float(row.get("route_hit_rate")),
            "verify_segment_cuda_event_ms_per_call": segment_event_per_call,
            "graph_external_gap_ms_per_call": graph_external_gap_ms,
            "verify_boundary_submit_ms_per_call": boundary_ms / verify_calls,
            "verify_prefetch_rank_ms_per_call": rank_ms / verify_calls,
            "verify_prefetch_transfer_enqueue_ms_per_call": (
                transfer_enqueue_ms / verify_calls
            ),
            "verify_prefetch_submit_per_call": submitted / verify_calls,
            "verify_prefetch_submitted_mb_per_call": submitted_mb_per_call,
            "verify_cpu_experts_per_call": cpu_experts_per_call,
            "verify_cpu_routes_per_call": cpu_routes_per_call,
            "top_cpu_expert_layers": _top_layers(row),
        }
        item["target_verify_ok"] = (
            args.target_verify_min_ms
            <= item["verify_forward_ms_avg"]
            <= args.target_verify_max_ms
        )
        item["target_decode_ok"] = (
            args.target_decode_min_tok_s
            <= item["decode_phase_output_tok_s"]
            <= args.target_decode_max_tok_s
        )
        item["target_ok"] = bool(
            item["target_verify_ok"] and item["target_decode_ok"]
        )
        rows.append(item)
    return rows


def write_optimized_summary(
    output_dir: Path,
    command: list[str],
    env_overrides: dict[str, str],
    source_summary: dict[str, Any],
    optimized_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    result = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "source_summary_json": str(output_dir / "summary.json"),
        "source_per_layer_cpu_experts_csv": str(
            output_dir / "per_layer_cpu_experts.csv"
        ),
        "command": command,
        "env_overrides": env_overrides,
        "targets": {
            "verify_ms": [
                float(args.target_verify_min_ms),
                float(args.target_verify_max_ms),
            ],
            "decode_phase_output_tok_s": [
                float(args.target_decode_min_tok_s),
                float(args.target_decode_max_tok_s),
            ],
        },
        "rows": optimized_rows,
        "target_ok": bool(optimized_rows and all(r["target_ok"] for r in optimized_rows)),
        "source_metadata": source_summary.get("metadata", {}),
    }
    (output_dir / "optimized_verify_command.txt").write_text(
        " ".join(command) + "\n", encoding="utf-8"
    )
    (output_dir / "optimized_verify_summary.json").write_text(
        json.dumps(result, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(output_dir / "optimized_verify_summary.md", result)
    return result


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    targets = result["targets"]
    lines = [
        "# Optimized Verify Performance",
        "",
        f"- source summary: `{result['source_summary_json']}`",
        f"- per-layer CPU experts CSV: `{result['source_per_layer_cpu_experts_csv']}`",
        f"- target verify ms: `{targets['verify_ms'][0]:.1f}-{targets['verify_ms'][1]:.1f}`",
        f"- target decode tok/s: `{targets['decode_phase_output_tok_s'][0]:.1f}-{targets['decode_phase_output_tok_s'][1]:.1f}`",
        f"- target result: `{'PASS' if result['target_ok'] else 'FAIL'}`",
        "",
        "## Environment Overrides",
        "",
        "| key | value |",
        "|:---|:---|",
    ]
    for key, value in result["env_overrides"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| case | out | K | verify ms | decode tok/s | total tok/s | draft ms | hit | accept | segment event | gap | boundary | rank | H2D enqueue | CPU routes | CPU experts | target |",
            "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in result["rows"]:
        lines.append(
            "| "
            f"{row['name']} | {row['output_len']} | {row['max_draft_tokens']} | "
            f"{row['verify_forward_ms_avg']:.3f} | "
            f"{row['decode_phase_output_tok_s']:.3f} | "
            f"{row['throughput_output_tok_s']:.3f} | "
            f"{row['draft_forward_ms_avg']:.3f} | "
            f"{row['route_hit_rate']:.4f} | "
            f"{row['acceptance_rate']:.4f} | "
            f"{row['verify_segment_cuda_event_ms_per_call']:.3f} | "
            f"{row['graph_external_gap_ms_per_call']:.3f} | "
            f"{row['verify_boundary_submit_ms_per_call']:.3f} | "
            f"{row['verify_prefetch_rank_ms_per_call']:.3f} | "
            f"{row['verify_prefetch_transfer_enqueue_ms_per_call']:.3f} | "
            f"{row['verify_cpu_routes_per_call']:.1f} | "
            f"{row['verify_cpu_experts_per_call']:.1f} | "
            f"{'PASS' if row['target_ok'] else 'FAIL'} |"
        )
    lines.extend(["", "## Top CPU Expert Layers", ""])
    for row in result["rows"]:
        lines.append(f"- `{row['name']}`: {row['top_cpu_expert_layers'] or 'none'}")
    lines.extend(
        [
            "",
            "## Command",
            "",
            "```bash",
            " ".join(result["command"]),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace, extra_args: list[str]) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = build_bench_command(args, extra_args)
    env, env_overrides = optimized_env(args)
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")

    if args.dry_run:
        print(" ".join(command))
        print(json.dumps(env_overrides, ensure_ascii=True, indent=2))
        return {
            "target_ok": False,
            "rows": [],
            "command": command,
            "env_overrides": env_overrides,
        }

    started = time.time()
    process = subprocess.run(
        command,
        cwd=repo_root,
        env=env,
        text=True,
        timeout=args.bench_timeout_sec,
    )
    elapsed = time.time() - started
    if process.returncode != 0:
        raise SystemExit(process.returncode)

    source_summary_path = output_dir / "summary.json"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    optimized_rows = summarize_rows(source_summary, output_dir, args)
    result = write_optimized_summary(
        output_dir, command, env_overrides, source_summary, optimized_rows, args
    )
    print(f"optimized_summary_json={output_dir / 'optimized_verify_summary.json'}")
    print(f"optimized_summary_md={output_dir / 'optimized_verify_summary.md'}")
    for row in optimized_rows:
        print(
            f"  {row['name']}: verify={row['verify_forward_ms_avg']:.3f} ms "
            f"decode={row['decode_phase_output_tok_s']:.3f} tok/s "
            f"gap={row['graph_external_gap_ms_per_call']:.3f} ms "
            f"cpu_routes={row['verify_cpu_routes_per_call']:.1f} "
            f"cpu_experts={row['verify_cpu_experts_per_call']:.1f} "
            f"target={'PASS' if row['target_ok'] else 'FAIL'}",
            flush=True,
        )
    print(f"bench_elapsed_sec={elapsed:.1f}")
    if args.fail_on_target_miss and not result["target_ok"]:
        raise SystemExit(2)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark optimized verify latency using the same feature path as "
            "bench_per_layer_slots.py, then check verify/decode targets."
        )
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.99)
    parser.add_argument("--cache-ratios", default="0.3125")
    parser.add_argument("--output-lens", default="512")
    parser.add_argument(
        "--max-draft-tokens-values",
        default="4",
        help=(
            "Optimized default is K=4.  Pass 12 to reproduce the original "
            "per-layer-slots command's draft length."
        ),
    )
    parser.add_argument("--segment-sizes", default="12")
    parser.add_argument("--allocation-modes", default="profile_weighted")
    parser.add_argument("--slot-buckets", type=int, default=4)
    parser.add_argument("--slot-max-bucket-ratio", type=float, default=2.0)
    parser.add_argument("--slot-profile-csv", default=DEFAULT_SLOT_PROFILE_CSV)
    parser.add_argument("--kt-num-threads", type=int, default=16)
    parser.add_argument(
        "--verify-cuda-graph-bucket-steps", default=DEFAULT_VERIFY_BUCKETS
    )
    parser.add_argument("--verify-prefetch-max-per-boundary", type=int, default=4)
    parser.add_argument("--verify-prefetch-visible-budget-ms", type=float, default=12.0)
    parser.add_argument("--verify-prefetch-rank-multiplier", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--dist-port-base", type=int, default=30800)
    parser.add_argument("--case-timeout-sec", type=int, default=2400)
    parser.add_argument("--bench-timeout-sec", type=int, default=3600)
    parser.add_argument("--skip-existing", type=str2bool, default=False)
    parser.add_argument("--fail-fast", type=str2bool, default=True)
    parser.add_argument("--fail-on-target-miss", type=str2bool, default=True)
    parser.add_argument("--target-verify-min-ms", type=float, default=50.0)
    parser.add_argument("--target-verify-max-ms", type=float, default=80.0)
    parser.add_argument("--target-decode-min-tok-s", type=float, default=25.0)
    parser.add_argument("--target-decode-max-tok-s", type=float, default=40.0)
    parser.add_argument("--segment-event-timing", type=str2bool, default=True)
    parser.add_argument("--allow-op-event-timing", type=str2bool, default=False)
    parser.add_argument("--preserve-conflicting-env", type=str2bool, default=False)
    parser.add_argument("--report-doc", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args, extra_args = parser.parse_known_args()
    run(args, extra_args)


if __name__ == "__main__":
    main()
