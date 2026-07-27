#!/usr/bin/env python3
"""Run and plot the fixed Qwen3-30B-A3B cumulative TPOT breakdown.

The experiment manifest is intentionally closed: callers may select stages,
but may not alter mechanism combinations.  Each stage runs in its own process,
port, log, and result directory.  ``--phase all`` gates the 80-request
validation run on all selected smoke stages passing their mechanism checks.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = REPO_ROOT / "scripts" / "bench_eval_workload_tpot.py"
PYTHON = Path("/home/linke/miniconda3/envs/nano_moe/bin/python")
MODEL_PATH = Path("/data1/models/Qwen3-30B-A3B")
MT_BENCH_PATH = Path("/data1/datasets/mt_bench/question.jsonl")
VERIFY_MODEL_PATH = Path(
    "results/transfer_v3_artifact_20260719/verify_cost_v3.json"
)
CUDA_VISIBLE_DEVICES = "2"
CPU_LIST = "64-96"
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20260719
EXPECTED_VALIDATION_REQUESTS = 80
EXPECTED_OUTPUT_TOKENS = 512
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Stage:
    stage_id: str
    directory: str
    label: str
    port: int
    cumulative: bool
    args: tuple[str, ...]


FIXED_K_ARGS = (
    "--draft-stop-policy",
    "none",
    "--draft-tpot-stop-rule",
    "first_increase",
    "--draft-tpot-verify-model-mode",
    "off",
    "--acceptance-predictor-enabled",
    "false",
)

STAGES = (
    Stage(
        "p0_eager",
        "p0_eager_exact_hetero_ar",
        "P0 Eager Ref.",
        37969,
        False,
        (
            "--inference-mode",
            "heter",
            "--spec-enable-prefetch",
            "false",
            "--prefetch-runtime-kind",
            "legacy",
            "--prefetch-runtime-mode",
            "baseline_staging",
            "--enforce-eager",
            "true",
            "--draft-cuda-graph-enabled",
            "false",
            "--verify-cuda-graph",
            "false",
            "--segment-sizes",
            "48",
            "--draft-reroute-policy",
            "round_robin",
            *FIXED_K_ARGS,
        ),
    ),
    Stage(
        "p0",
        "p0_exact_hetero_ar",
        "Exact Hetero AR + Graph",
        37970,
        True,
        (
            "--inference-mode",
            "heter",
            "--spec-enable-prefetch",
            "false",
            "--prefetch-runtime-kind",
            "legacy",
            "--prefetch-runtime-mode",
            "baseline_staging",
            "--enforce-eager",
            "false",
            "--draft-cuda-graph-enabled",
            "false",
            "--verify-cuda-graph",
            "false",
            "--segment-sizes",
            "48",
            "--draft-reroute-policy",
            "round_robin",
            *FIXED_K_ARGS,
        ),
    ),
    Stage(
        "p1",
        "p1_drafter",
        "+ Drafter",
        37971,
        True,
        (
            "--inference-mode",
            "spec",
            "--spec-enable-prefetch",
            "false",
            "--prefetch-runtime-kind",
            "legacy",
            "--prefetch-runtime-mode",
            "baseline_staging",
            "--enforce-eager",
            "false",
            "--draft-cuda-graph-enabled",
            "true",
            "--verify-cuda-graph",
            "true",
            "--segment-sizes",
            "48",
            "--draft-reroute-policy",
            "round_robin",
            *FIXED_K_ARGS,
        ),
    ),
    Stage(
        "p2",
        "p2_rerouter",
        "+ Rerouter",
        37972,
        True,
        (
            "--inference-mode",
            "spec",
            "--spec-enable-prefetch",
            "false",
            "--prefetch-runtime-kind",
            "legacy",
            "--prefetch-runtime-mode",
            "baseline_staging",
            "--enforce-eager",
            "false",
            "--draft-cuda-graph-enabled",
            "true",
            "--verify-cuda-graph",
            "true",
            "--segment-sizes",
            "48",
            "--draft-reroute-policy",
            "entropy_cache_bias",
            *FIXED_K_ARGS,
        ),
    ),
    Stage(
        "r_eager",
        "r_eager",
        "Eager Ref.",
        37973,
        False,
        (
            "--inference-mode",
            "spec",
            "--spec-enable-prefetch",
            "false",
            "--prefetch-runtime-kind",
            "legacy",
            "--prefetch-runtime-mode",
            "draft_segment_indexed",
            "--enforce-eager",
            "true",
            "--draft-cuda-graph-enabled",
            "false",
            "--verify-cuda-graph",
            "false",
            "--segment-sizes",
            "12",
            "--draft-reroute-policy",
            "entropy_cache_bias",
            *FIXED_K_ARGS,
        ),
    ),
    Stage(
        "p3",
        "p3_segment_graph",
        "+ Segment Graph",
        37974,
        True,
        (
            "--inference-mode",
            "spec",
            "--spec-enable-prefetch",
            "false",
            "--prefetch-runtime-kind",
            "legacy",
            "--prefetch-runtime-mode",
            "draft_segment_indexed",
            "--enforce-eager",
            "false",
            "--draft-cuda-graph-enabled",
            "true",
            "--verify-cuda-graph",
            "true",
            "--segment-sizes",
            "12",
            "--draft-reroute-policy",
            "entropy_cache_bias",
            *FIXED_K_ARGS,
        ),
    ),
    Stage(
        "p4",
        "p4_predictive_prefetch",
        "+ Predictive Prefetch",
        37975,
        True,
        (
            "--inference-mode",
            "spec",
            "--spec-enable-prefetch",
            "true",
            "--prefetch-runtime-kind",
            "predictive",
            "--prefetch-runtime-mode",
            "draft_segment_indexed",
            "--enforce-eager",
            "false",
            "--draft-cuda-graph-enabled",
            "true",
            "--verify-cuda-graph",
            "true",
            "--segment-sizes",
            "12",
            "--draft-reroute-policy",
            "entropy_cache_bias",
            *FIXED_K_ARGS,
        ),
    ),
    Stage(
        "p5",
        "p5_early_stop_full",
        "+ Early Stop (Ours)",
        37976,
        True,
        (
            "--inference-mode",
            "spec",
            "--spec-enable-prefetch",
            "true",
            "--prefetch-runtime-kind",
            "predictive",
            "--prefetch-runtime-mode",
            "draft_segment_indexed",
            "--enforce-eager",
            "false",
            "--draft-cuda-graph-enabled",
            "true",
            "--verify-cuda-graph",
            "true",
            "--segment-sizes",
            "12",
            "--draft-reroute-policy",
            "entropy_cache_bias",
            "--draft-stop-policy",
            "tpot",
            "--draft-tpot-cost-model",
            "history",
            "--draft-tpot-stop-rule",
            "transfer_aware_step",
            "--draft-tpot-min-steps",
            "6",
            "--draft-tpot-stop-margin",
            "0.0",
            "--draft-tpot-lookahead-cache-credit-ms-per-step",
            "0.0",
            "--draft-tpot-verify-model-mode",
            "active",
            "--draft-tpot-verify-model-path",
            str(VERIFY_MODEL_PATH),
            "--draft-tpot-uncertainty-scale",
            "0.0",
            "--acceptance-predictor-enabled",
            "true",
        ),
    ),
)

STAGE_BY_ID = {stage.stage_id: stage for stage in STAGES}
PLOT_STAGE_IDS = (
    "p0_eager",
    "p0",
    "p1",
    "p2",
    "p3",
    "r_eager",
    "p4",
    "p5",
)
MAIN_STAGE_IDS = ("p0", "p1", "p2", "p3", "p4", "p5")


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv_atomic(
    path: Path,
    fieldnames: Iterable[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def common_benchmark_args(
    *,
    result_dir: Path,
    num_samples: str,
    engine_profile: bool,
) -> list[str]:
    return [
        "taskset",
        "--cpu-list",
        CPU_LIST,
        str(PYTHON),
        str(BENCHMARK_SCRIPT),
        "--model-path",
        str(MODEL_PATH),
        "--dataset",
        "mt_bench",
        "--mt-bench-path",
        str(MT_BENCH_PATH),
        "--request-mode",
        "dataset",
        "--num-samples",
        num_samples,
        "--optimized-config",
        "k12_transfer_step",
        "--cache-ratios",
        "0.3125",
        "--output-lens",
        "512",
        "--max-draft-tokens-values",
        "12",
        "--allocation-modes",
        "profile_weighted",
        "--slot-buckets",
        "4",
        "--slot-max-bucket-ratio",
        "2.0",
        "--slot-profile-csv",
        "pre_exps/exp_and_figs/unique/unique_count_plot_summary_n1024.csv",
        "--kt-num-threads",
        "16",
        "--kt-direct-backend",
        "avx2_bf16",
        "--verify-cuda-graph-bucket-steps",
        "5,7,8,9,10,11,12,13",
        "--verify-prefetch-max-per-boundary",
        "4",
        "--verify-prefetch-rank-multiplier",
        "1",
        "--gpu-memory-utilization",
        "0.99",
        "--temperature",
        "0.8",
        "--acceptance-strategy",
        "standard_sampling",
        "--decode-driver",
        "generate",
        "--collect-profile",
        "true",
        "--engine-profile",
        "true" if engine_profile else "false",
        "--engine-profile-cuda-sync",
        "false",
        "--save-profile-json",
        "true",
        "--save-token-ids",
        "true",
        "--save-text",
        "true",
        "--reset-profile-after-warmup",
        "true",
        "--reset-seed-after-warmup",
        "true",
        "--reset-profile-before-request",
        "true",
        "--repeats",
        "1",
        "--skip-existing",
        "false",
        "--fail-fast",
        "true",
        "--fail-on-output-validation-error",
        "true",
        "--seed",
        str(BOOTSTRAP_SEED),
        "--cache-strategy",
        "lru",
        "--output-dir",
        str(result_dir),
    ]


def benchmark_command(stage: Stage, phase: str, result_dir: Path) -> list[str]:
    command = common_benchmark_args(
        result_dir=result_dir,
        num_samples="1" if phase == "smoke" else "all",
        engine_profile=phase == "smoke",
    )
    command.extend(["--dist-port-base", str(stage.port)])
    command.extend(stage.args)
    return command


def command_text(command: list[str]) -> str:
    return (
        f"CUDA_VISIBLE_DEVICES={shlex.quote(CUDA_VISIBLE_DEVICES)} "
        + shlex.join(command)
    )


def run_command(command: list[str], log_path: Path) -> int:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        rendered = command_text(command)
        log.write(f"$ {rendered}\n")
        log.flush()
        print(f"$ {rendered}", flush=True)
        try:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as error:
            message = f"failed to start benchmark: {error}\n"
            log.write(message)
            print(message, end="", flush=True)
            return 127
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        return process.wait()


def load_summary(result_dir: Path) -> dict[str, Any]:
    path = result_dir / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing benchmark summary: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"benchmark summary is not an object: {path}")
    return value


def load_first_profile(
    result_dir: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    rows = summary.get("rows", [])
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict) or row.get("status") != "ok":
                continue
            configured = str(row.get("profile_json", "") or "")
            if configured:
                path = Path(configured)
                if not path.is_absolute():
                    path = REPO_ROOT / path
                if path.is_file():
                    value = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(value, dict):
                        return value
    profiles = sorted(result_dir.glob("*_profiles/sample0000.json"))
    if not profiles:
        raise FileNotFoundError(
            f"missing sample0000 mechanism profile under {result_dir}"
        )
    value = json.loads(profiles[0].read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"profile is not an object: {profiles[0]}")
    return value


def numeric(profile: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = profile.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def graph_counts(profile: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in profile.items()
        if key.endswith("graph_replay_count")
        and isinstance(value, (int, float))
    }


def transfer_observed(profile: dict[str, Any]) -> dict[str, float]:
    return {
        "submit": numeric(
            profile,
            "model_prefetch_submit_count",
            "prefetch_submit_count",
        ),
        "ready": numeric(
            profile,
            "model_direct_active_prefetch_ready_count",
            "direct_active_prefetch_ready_count",
        ),
        "publish": numeric(
            profile,
            "model_publish_count",
            "publish_count",
        ),
        "consume": numeric(
            profile,
            "model_prefetch_consumed_count",
            "prefetch_consumed_count",
        ),
    }


def cache_mutation_count(profile: dict[str, Any]) -> float:
    total = 0.0
    for key, value in profile.items():
        if not isinstance(value, (int, float)):
            continue
        lowered = key.lower()
        if (
            ("promoted" in lowered or "promotion" in lowered or "evicted" in lowered)
            and lowered.endswith("count")
        ):
            total += float(value)
    return total


def output_validation(
    summary: dict[str, Any],
    *,
    expected_requests: int,
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    rows_value = summary.get("rows", [])
    rows = rows_value if isinstance(rows_value, list) else []
    failures = summary.get("failures", [])
    ok_rows = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("status") == "ok"
    ]
    if len(rows) != expected_requests:
        errors.append(
            f"row_count={len(rows)} expected={expected_requests}"
        )
    if len(ok_rows) != expected_requests:
        errors.append(
            f"ok_count={len(ok_rows)} expected={expected_requests}"
        )
    if isinstance(failures, list) and failures:
        errors.append(f"summary_failures={len(failures)}")
    wrong_lengths = [
        index
        for index, row in enumerate(ok_rows)
        if int(row.get("generated_output_tokens", -1) or -1)
        != EXPECTED_OUTPUT_TOKENS
    ]
    if wrong_lengths:
        errors.append(
            f"non_{EXPECTED_OUTPUT_TOKENS}_token_rows="
            f"{len(wrong_lengths)} first={wrong_lengths[0]}"
        )
    output_errors = [
        index
        for index, row in enumerate(ok_rows)
        if str(row.get("output_validation_error", "") or "").strip()
    ]
    if output_errors:
        errors.append(
            f"output_validation_error_rows={len(output_errors)} "
            f"first={output_errors[0]}"
        )
    return errors, {
        "row_count": len(rows),
        "ok_count": len(ok_rows),
        "expected_requests": expected_requests,
        "expected_output_tokens": EXPECTED_OUTPUT_TOKENS,
        "wrong_output_length_count": len(wrong_lengths),
        "output_validation_error_count": len(output_errors),
    }


def stage_arg(stage: Stage, option: str) -> str:
    indexes = [
        index
        for index, value in enumerate(stage.args)
        if value == option
    ]
    if not indexes:
        raise KeyError(f"{stage.stage_id} has no fixed manifest option {option}")
    index = indexes[-1]
    if index + 1 >= len(stage.args):
        raise ValueError(
            f"{stage.stage_id} fixed manifest option has no value: {option}"
        )
    return str(stage.args[index + 1])


def bool_arg(stage: Stage, option: str) -> bool:
    return stage_arg(stage, option).strip().lower() == "true"


def resolved_metadata_validation(
    stage: Stage,
    phase: str,
    summary: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    metadata_value = summary.get("metadata", {})
    metadata = metadata_value if isinstance(metadata_value, dict) else {}
    expected = {
        "model_path": str(MODEL_PATH),
        "request_mode": "dataset",
        "datasets": ["mt_bench"],
        "optimized_config": "k12_transfer_step",
        "num_samples": "1" if phase == "smoke" else "all",
        "allocation_modes": ["profile_weighted"],
        "cache_ratios": [0.3125],
        "max_output_tokens_values": [512],
        "output_lens_compat_mode": True,
        "max_draft_tokens_values": [12],
        "segment_sizes": [int(stage_arg(stage, "--segment-sizes"))],
        "inference_mode": stage_arg(stage, "--inference-mode"),
        "enforce_eager": bool_arg(stage, "--enforce-eager"),
        "spec_enable_prefetch": bool_arg(
            stage, "--spec-enable-prefetch"
        ),
        "prefetch_runtime_mode": stage_arg(
            stage, "--prefetch-runtime-mode"
        ),
        "prefetch_runtime_kind": stage_arg(
            stage, "--prefetch-runtime-kind"
        ),
        "draft_cuda_graph_enabled": bool_arg(
            stage, "--draft-cuda-graph-enabled"
        ),
        "verify_cuda_graph": bool_arg(stage, "--verify-cuda-graph"),
        "draft_reroute_policy": stage_arg(
            stage, "--draft-reroute-policy"
        ),
        "draft_stop_policy": stage_arg(stage, "--draft-stop-policy"),
        "acceptance_predictor_enabled": bool_arg(
            stage, "--acceptance-predictor-enabled"
        ),
        "cache_strategy": "lru",
        "verify_cuda_graph_bucket_steps": [5, 7, 8, 9, 10, 11, 12, 13],
        "verify_prefetch_max_per_boundary": 4,
        "temperature": 0.8,
        "decode_driver": "generate",
        "collect_profile": True,
        "engine_profile": phase == "smoke",
        "engine_profile_cuda_sync": False,
    }
    mismatches = {
        key: {
            "actual": metadata.get(key),
            "expected": value,
        }
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    errors = []
    if mismatches:
        errors.append(
            "resolved manifest metadata mismatches: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    return errors, {
        "mismatches": mismatches,
        "checked_fields": sorted(expected),
    }


def require_equal_count(
    errors: list[str],
    *,
    name: str,
    actual: float,
    expected: float,
) -> None:
    if int(actual) != int(expected):
        errors.append(f"{name}={actual:g} expected={expected:g}")


def mechanism_validation(
    stage: Stage,
    summary: dict[str, Any],
    profile: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    metadata_value = summary.get("metadata", {})
    metadata = metadata_value if isinstance(metadata_value, dict) else {}
    graphs = graph_counts(profile)
    transfers = transfer_observed(profile)
    observed: dict[str, Any] = {
        "graph_replay_counts": graphs,
        "transfers": transfers,
        "cache_mutation_count": cache_mutation_count(profile),
        "prefetch_runtime_class": metadata.get("prefetch_runtime_class"),
    }

    if stage.stage_id in {"p0_eager", "p0"}:
        standard = numeric(
            profile,
            "model_standard_graph_replay_count",
            "standard_graph_replay_count",
        )
        cpu_routes = numeric(
            profile,
            "model_cpu_routes_sum",
            "cpu_routes_sum",
            "model_verify_cpu_routes_sum",
        )
        observed.update(
            standard_graph_replay_count=standard,
            cpu_routes_sum=cpu_routes,
        )
        if stage.stage_id == "p0":
            hybrid = numeric(
                profile,
                "model_standard_kt_hybrid_graph_replay_count",
                "standard_kt_hybrid_graph_replay_count",
            )
            observed["standard_kt_hybrid_graph_replay_count"] = hybrid
            if standard <= 0:
                errors.append("standard CUDA graph replay was not observed")
            if hybrid <= 0:
                errors.append(
                    "exact KT-direct hybrid standard graph replay was not observed"
                )
        else:
            nonzero = {
                key: value for key, value in graphs.items() if value != 0
            }
            if nonzero:
                errors.append(
                    f"P0 eager reference replayed CUDA graphs: {nonzero}"
                )
        if cpu_routes <= 0:
            errors.append("kt_direct CPU routes were not observed")
        if any(value != 0 for value in transfers.values()):
            errors.append(f"unexpected runtime transfer activity: {transfers}")
        if observed["cache_mutation_count"] != 0:
            errors.append(
                "unexpected cache promotion/eviction activity: "
                f"{observed['cache_mutation_count']:g}"
            )

    elif stage.stage_id in {"p1", "p2"}:
        draft_calls = numeric(profile, "spec_run_draft_calls")
        verify_calls = numeric(profile, "spec_run_verify_calls")
        draft_graphs = numeric(profile, "model_draft_graph_replay_count")
        verify_graphs = numeric(
            profile,
            "model_verify_kt_hybrid_graph_replay_count",
        )
        draft_segments = numeric(
            profile,
            "model_draft_segment_graph_replay_count",
        )
        verify_segments = numeric(
            profile,
            "model_verify_segment_graph_replay_enqueue_count",
        )
        observed.update(
            draft_calls=draft_calls,
            verify_calls=verify_calls,
            draft_graph_replays=draft_graphs,
            verify_graph_replays=verify_graphs,
            draft_segment_replays=draft_segments,
            verify_segment_replays=verify_segments,
        )
        if draft_calls <= 0 or verify_calls <= 0:
            errors.append("draft/verify forwards were not observed")
        require_equal_count(
            errors,
            name="draft_graph_replays",
            actual=draft_graphs,
            expected=draft_calls,
        )
        require_equal_count(
            errors,
            name="verify_graph_replays",
            actual=verify_graphs,
            expected=verify_calls,
        )
        if draft_segments != 0 or verify_segments != 0:
            errors.append(
                "monolithic stage unexpectedly used segment graph replay"
            )
        if transfers["submit"] != 0 or transfers["publish"] != 0:
            errors.append(f"unexpected runtime transfer activity: {transfers}")

    elif stage.stage_id == "r_eager":
        nonzero = {
            key: value for key, value in graphs.items() if value != 0
        }
        if nonzero:
            errors.append(f"eager reference replayed CUDA graphs: {nonzero}")

    elif stage.stage_id == "p3":
        draft_calls = numeric(profile, "spec_run_draft_calls")
        verify_calls = numeric(profile, "spec_run_verify_calls")
        draft_segments = numeric(
            profile,
            "model_draft_segment_graph_replay_count",
        )
        verify_segments = numeric(
            profile,
            "model_verify_segment_graph_replay_enqueue_count",
        )
        observed.update(
            draft_calls=draft_calls,
            verify_calls=verify_calls,
            draft_segment_replays=draft_segments,
            verify_segment_replays=verify_segments,
        )
        if draft_calls <= 0 or verify_calls <= 0:
            errors.append("draft/verify forwards were not observed")
        require_equal_count(
            errors,
            name="draft_segment_replays",
            actual=draft_segments,
            expected=4 * draft_calls,
        )
        require_equal_count(
            errors,
            name="verify_segment_replays",
            actual=verify_segments,
            expected=4 * verify_calls,
        )
        if transfers["submit"] != 0 or transfers["publish"] != 0:
            errors.append(f"unexpected runtime transfer activity: {transfers}")

    elif stage.stage_id in {"p4", "p5"}:
        for name in ("submit", "ready", "publish", "consume"):
            if transfers[name] <= 0:
                errors.append(f"predictive prefetch {name} was not observed")
        if metadata.get("prefetch_runtime_class") != "PredictivePrefetchRuntime":
            errors.append(
                "resolved runtime class is not PredictivePrefetchRuntime"
            )

    if stage.stage_id == "p5":
        early_stops = numeric(profile, "spec_draft_tpot_early_stop_count")
        raw_lengths = profile.get("spec_draft_steps_per_step", [])
        all_positive_lengths = [
            int(value)
            for value in raw_lengths
            if isinstance(value, (int, float)) and int(value) > 0
        ] if isinstance(raw_lengths, list) else []
        traces = profile.get("spec_step_traces", [])
        policy_lengths = [
            int(
                trace.get(
                    "draft_steps_actual",
                    trace.get("draft_steps", 0),
                )
                or 0
            )
            for trace in traces
            if isinstance(trace, dict)
            and isinstance(trace.get("draft_tpot_costs"), list)
            and bool(trace["draft_tpot_costs"])
        ] if isinstance(traces, list) else []
        # The last fixed-length request step may be clipped below min-K by the
        # remaining output-token budget.  It has no TPOT decision trace and is
        # not an early-stop policy choice.
        lengths = policy_lengths or all_positive_lengths
        observed.update(
            draft_tpot_early_stop_count=early_stops,
            draft_length_min=min(lengths) if lengths else None,
            draft_length_max=max(lengths) if lengths else None,
            terminal_clipped_draft_lengths=[
                value
                for value in all_positive_lengths
                if value < 6
            ],
        )
        if early_stops <= 0:
            errors.append("draft_tpot_early_stop was not observed")
        invalid_lengths = [
            value for value in lengths if value < 6 or value > 12
        ]
        if not lengths:
            errors.append("no policy-controlled draft lengths were recorded")
        elif invalid_lengths:
            errors.append(
                "policy-controlled draft lengths outside [6, 12]: "
                + ",".join(str(value) for value in invalid_lengths[:8])
            )
        expected_metadata = {
            "optimized_config": "k12_transfer_step",
            "inference_mode": "spec",
            "enable_heterogeneous": True,
            "enable_speculative": True,
            "enforce_eager": False,
            "spec_enable_prefetch": True,
            "prefetch_runtime_mode": "draft_segment_indexed",
            "prefetch_runtime_kind": "predictive",
            "prefetch_runtime_class": "PredictivePrefetchRuntime",
            "draft_cuda_graph_enabled": True,
            "verify_cuda_graph": True,
            "draft_reroute_policy": "entropy_cache_bias",
            "cache_strategy": "lru",
            "max_draft_tokens_values": [12],
            "segment_sizes": [12],
            "draft_stop_policy": "tpot",
            "draft_tpot_cost_model": "history",
            "draft_tpot_stop_rule": "transfer_aware_step",
            "draft_tpot_min_steps": 6,
            "draft_tpot_stop_margin": 0.0,
            "draft_tpot_lookahead_cache_credit_ms_per_step": 0.0,
            "draft_tpot_verify_model_mode": "active",
            "draft_tpot_verify_model_path": str(VERIFY_MODEL_PATH),
            "draft_tpot_uncertainty_scale": 0.0,
            "acceptance_predictor_enabled": True,
            "verify_prefetch_max_per_boundary": 4,
            "verify_cuda_graph_bucket_steps": [5, 7, 8, 9, 10, 11, 12, 13],
        }
        mismatches = {
            key: {
                "actual": metadata.get(key),
                "expected": expected,
            }
            for key, expected in expected_metadata.items()
            if metadata.get(key) != expected
        }
        if mismatches:
            errors.append(
                "P5 resolved config mismatches: "
                + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
            )
        observed["resolved_config_mismatches"] = mismatches

    return errors, observed


def validate_result_dir(
    stage: Stage,
    phase: str,
    result_dir: Path,
) -> dict[str, Any]:
    expected_requests = (
        1 if phase == "smoke" else EXPECTED_VALIDATION_REQUESTS
    )
    errors: list[str] = []
    observed: dict[str, Any] = {}
    try:
        summary = load_summary(result_dir)
        output_errors, output_observed = output_validation(
            summary,
            expected_requests=expected_requests,
        )
        errors.extend(output_errors)
        observed["output"] = output_observed
        metadata_errors, metadata_observed = resolved_metadata_validation(
            stage,
            phase,
            summary,
        )
        errors.extend(metadata_errors)
        observed["resolved_metadata"] = metadata_observed
        if phase == "smoke" and not output_errors:
            profile = load_first_profile(result_dir, summary)
            mechanism_errors, mechanism_observed = mechanism_validation(
                stage,
                summary,
                profile,
            )
            errors.extend(mechanism_errors)
            observed["mechanism"] = mechanism_observed
    except (OSError, ValueError, json.JSONDecodeError) as error:
        errors.append(str(error))
    return {
        "status": "passed" if not errors else "failed",
        "stage": stage.stage_id,
        "phase": phase,
        "result_dir": str(result_dir),
        "checked_utc": utc_timestamp(),
        "errors": errors,
        "observed": observed,
    }


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_pooled_tpot_ci(
    requests: list[tuple[float, int]],
    *,
    seed: int,
) -> tuple[float, float]:
    if not requests:
        return 0.0, 0.0
    rng = random.Random(seed)
    count = len(requests)
    estimates: list[float] = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        decode_sec = 0.0
        intervals = 0
        for _ in range(count):
            sample_decode_sec, sample_intervals = requests[
                rng.randrange(count)
            ]
            decode_sec += sample_decode_sec
            intervals += sample_intervals
        estimates.append(
            1000.0 * decode_sec / intervals if intervals > 0 else 0.0
        )
    return percentile(estimates, 2.5), percentile(estimates, 97.5)


def aggregate_stage(
    stage: Stage,
    result_dir: Path,
    validation: dict[str, Any],
    *,
    bootstrap_seed: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "stage": stage.stage_id,
        "label": stage.label,
        "cumulative": stage.cumulative,
        "status": validation.get("status", "failed"),
        "result_dir": str(result_dir),
        "request_count": 0,
        "decode_sec_sum": None,
        "decode_token_intervals_sum": None,
        "pooled_tpot_ms": None,
        "request_tpot_ms_p50": None,
        "request_tpot_ms_p90": None,
        "bootstrap_95ci_low_ms": None,
        "bootstrap_95ci_high_ms": None,
    }
    if validation.get("status") != "passed":
        row["errors"] = validation.get("errors", [])
        return row
    summary = load_summary(result_dir)
    rows = [
        value
        for value in summary.get("rows", [])
        if isinstance(value, dict) and value.get("status") == "ok"
    ]
    requests = [
        (
            float(value.get("decode_sec", 0.0) or 0.0),
            int(value.get("generated_output_tokens", 0) or 0) - 1,
        )
        for value in rows
    ]
    decode_sec_sum = sum(value[0] for value in requests)
    interval_sum = sum(value[1] for value in requests)
    request_tpots = [
        1000.0 * decode_sec / intervals
        for decode_sec, intervals in requests
        if intervals > 0
    ]
    ci_low, ci_high = bootstrap_pooled_tpot_ci(
        requests,
        seed=bootstrap_seed,
    )
    row.update(
        request_count=len(requests),
        decode_sec_sum=decode_sec_sum,
        decode_token_intervals_sum=interval_sum,
        pooled_tpot_ms=(
            1000.0 * decode_sec_sum / interval_sum
            if interval_sum > 0
            else 0.0
        ),
        request_tpot_ms_p50=percentile(request_tpots, 50),
        request_tpot_ms_p90=percentile(request_tpots, 90),
        bootstrap_95ci_low_ms=ci_low,
        bootstrap_95ci_high_ms=ci_high,
    )
    return row


def add_cumulative_deltas(rows: list[dict[str, Any]]) -> None:
    by_id = {str(row["stage"]): row for row in rows}
    baseline = by_id.get("p0", {}).get("pooled_tpot_ms")
    previous: float | None = None
    for stage_id in MAIN_STAGE_IDS:
        row = by_id.get(stage_id)
        if row is None:
            continue
        current = row.get("pooled_tpot_ms")
        if not isinstance(current, (int, float)):
            continue
        row["reduction_vs_p0_percent"] = (
            100.0 * (float(baseline) - float(current)) / float(baseline)
            if isinstance(baseline, (int, float)) and baseline != 0
            else None
        )
        row["incremental_reduction_percent"] = (
            100.0 * (previous - float(current)) / previous
            if previous not in (None, 0.0)
            else None
        )
        previous = float(current)


AGGREGATE_FIELDS = (
    "stage",
    "label",
    "cumulative",
    "status",
    "request_count",
    "decode_sec_sum",
    "decode_token_intervals_sum",
    "pooled_tpot_ms",
    "request_tpot_ms_p50",
    "request_tpot_ms_p90",
    "bootstrap_95ci_low_ms",
    "bootstrap_95ci_high_ms",
    "reduction_vs_p0_percent",
    "incremental_reduction_percent",
    "result_dir",
)


def svg_text(
    x: float,
    y: float,
    text: str,
    *,
    anchor: str = "middle",
    size: int = 14,
    weight: str = "normal",
    fill: str = "#172033",
    extra: str = "",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" {extra}>'
        f"{html.escape(text)}</text>"
    )


def write_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    by_id = {str(row["stage"]): row for row in rows}
    plotted = [
        (STAGE_BY_ID[stage_id], by_id.get(stage_id, {}))
        for stage_id in PLOT_STAGE_IDS
        if stage_id in by_id
    ]
    width = 1180
    height = 680
    margin_left = 100
    margin_right = 45
    margin_top = 110
    margin_bottom = 150
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    numeric_highs = [
        float(row.get("bootstrap_95ci_high_ms") or row.get("pooled_tpot_ms"))
        for _, row in plotted
        if isinstance(
            row.get("bootstrap_95ci_high_ms") or row.get("pooled_tpot_ms"),
            (int, float),
        )
    ]
    y_max = max(numeric_highs, default=1.0) * 1.22
    if y_max <= 0:
        y_max = 1.0
    slot_width = chart_width / max(1, len(plotted))
    bar_width = min(88.0, slot_width * 0.62)

    def y_of(value: float) -> float:
        return margin_top + chart_height * (1.0 - value / y_max)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" '
            'role="img" aria-labelledby="title desc">'
        ),
        '<title id="title">Qwen3-30B-A3B TPOT Performance Breakdown</title>',
        (
            '<desc id="desc">Pooled time per output token with request-level '
            'bootstrap 95 percent confidence intervals.</desc>'
        ),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(
            width / 2,
            38,
            "Qwen3-30B-A3B Cumulative TPOT Performance Breakdown",
            size=22,
            weight="bold",
        ),
        svg_text(
            width / 2,
            65,
            (
                "MT-Bench, 80 requests × 512 tokens · pooled TPOT · "
                "request bootstrap 95% CI"
            ),
            size=13,
            fill="#556070",
        ),
    ]
    for tick in range(6):
        value = y_max * tick / 5
        y = y_of(value)
        lines.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" '
            f'x2="{width - margin_right}" y2="{y:.1f}" '
            'stroke="#e3e7ed" stroke-width="1"/>'
        )
        lines.append(
            svg_text(
                margin_left - 12,
                y + 5,
                f"{value:.1f}",
                anchor="end",
                size=12,
                fill="#556070",
            )
        )
    baseline_y = margin_top + chart_height
    lines.append(
        f'<line x1="{margin_left}" y1="{baseline_y:.1f}" '
        f'x2="{width - margin_right}" y2="{baseline_y:.1f}" '
        'stroke="#475569" stroke-width="1.5"/>'
    )
    lines.append(
        svg_text(
            27,
            margin_top + chart_height / 2,
            "Pooled TPOT (ms/token, lower is better)",
            size=13,
            weight="bold",
            extra=(
                f'transform="rotate(-90 27 '
                f'{margin_top + chart_height / 2:.1f})"'
            ),
        )
    )

    centers: dict[str, float] = {}
    palette = {
        "p0_eager": "#9aa2ad",
        "p0": "#315a9b",
        "p1": "#3e6dac",
        "p2": "#4b80bd",
        "p3": "#5794cd",
        "p4": "#2f9c95",
        "p5": "#e07a2f",
        "r_eager": "#9aa2ad",
    }
    for index, (stage, row) in enumerate(plotted):
        center = margin_left + slot_width * (index + 0.5)
        centers[stage.stage_id] = center
        value = row.get("pooled_tpot_ms")
        if not isinstance(value, (int, float)):
            lines.append(
                svg_text(
                    center,
                    baseline_y - 12,
                    "N/A",
                    size=13,
                    fill="#7a8492",
                )
            )
        else:
            top = y_of(float(value))
            bar_height = baseline_y - top
            lines.append(
                f'<rect x="{center - bar_width / 2:.1f}" y="{top:.1f}" '
                f'width="{bar_width:.1f}" height="{bar_height:.1f}" '
                f'rx="4" fill="{palette[stage.stage_id]}"/>'
            )
            low = row.get("bootstrap_95ci_low_ms")
            high = row.get("bootstrap_95ci_high_ms")
            if isinstance(low, (int, float)) and isinstance(high, (int, float)):
                y_low = y_of(float(low))
                y_high = y_of(float(high))
                lines.extend(
                    [
                        (
                            f'<line x1="{center:.1f}" y1="{y_high:.1f}" '
                            f'x2="{center:.1f}" y2="{y_low:.1f}" '
                            'stroke="#172033" stroke-width="2"/>'
                        ),
                        (
                            f'<line x1="{center - 9:.1f}" y1="{y_high:.1f}" '
                            f'x2="{center + 9:.1f}" y2="{y_high:.1f}" '
                            'stroke="#172033" stroke-width="2"/>'
                        ),
                        (
                            f'<line x1="{center - 9:.1f}" y1="{y_low:.1f}" '
                            f'x2="{center + 9:.1f}" y2="{y_low:.1f}" '
                            'stroke="#172033" stroke-width="2"/>'
                        ),
                    ]
                )
            lines.append(
                svg_text(
                    center,
                    max(margin_top + 16, top - 12),
                    f"{float(value):.2f}",
                    size=13,
                    weight="bold",
                )
            )

        words = stage.label.split()
        if len(words) <= 2:
            label_lines = [" ".join(words)]
        else:
            split = (len(words) + 1) // 2
            label_lines = [
                " ".join(words[:split]),
                " ".join(words[split:]),
            ]
        lines.append(
            svg_text(
                center,
                baseline_y + 26,
                stage.stage_id.upper().replace("_", "-"),
                size=12,
                weight="bold",
                fill="#475569",
            )
        )
        for line_index, label in enumerate(label_lines):
            lines.append(
                svg_text(
                    center,
                    baseline_y + 48 + 18 * line_index,
                    label,
                    size=12,
                    fill="#263244",
                )
            )

    if "p3" in centers and "r_eager" in centers:
        x1 = centers["p3"]
        x2 = centers["r_eager"]
        bracket_y = baseline_y + 105
        lines.extend(
            [
                (
                    f'<path d="M {x1:.1f} {bracket_y + 9:.1f} '
                    f'V {bracket_y:.1f} H {x2:.1f} '
                    f'V {bracket_y + 9:.1f}" fill="none" '
                    'stroke="#697386" stroke-width="1.5"/>'
                ),
                svg_text(
                    (x1 + x2) / 2,
                    bracket_y + 28,
                    "Segment Graph vs Eager",
                    size=12,
                    weight="bold",
                    fill="#697386",
                ),
            ]
        )
    if "p0_eager" in centers and "p0" in centers:
        x1 = centers["p0_eager"]
        x2 = centers["p0"]
        bracket_y = baseline_y + 105
        lines.extend(
            [
                (
                    f'<path d="M {x1:.1f} {bracket_y + 9:.1f} '
                    f'V {bracket_y:.1f} H {x2:.1f} '
                    f'V {bracket_y + 9:.1f}" fill="none" '
                    'stroke="#697386" stroke-width="1.5"/>'
                ),
                svg_text(
                    (x1 + x2) / 2,
                    bracket_y + 28,
                    "P0 Graph vs Eager",
                    size=12,
                    weight="bold",
                    fill="#697386",
                ),
            ]
        )
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_phase(
    output_dir: Path,
    phase: str,
    selected: tuple[Stage, ...],
    validations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    phase_dir = output_dir / phase
    rows = [
        aggregate_stage(
            stage,
            phase_dir / stage.directory,
            validations.get(
                stage.stage_id,
                {
                    "status": "failed",
                    "errors": ["stage was not validated"],
                },
            ),
            bootstrap_seed=BOOTSTRAP_SEED + index,
        )
        for index, stage in enumerate(selected)
    ]
    add_cumulative_deltas(rows)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_timestamp(),
        "phase": phase,
        "metric": (
            "1000 * sum(decode_sec) / "
            "sum(generated_output_tokens - 1)"
        ),
        "request_percentiles": ["p50", "p90"],
        "bootstrap": {
            "unit": "request",
            "iterations": BOOTSTRAP_ITERATIONS,
            "confidence_level": 0.95,
            "seed": BOOTSTRAP_SEED,
            "statistic": "pooled_tpot_ms",
        },
        "rows": rows,
    }
    json_path = phase_dir / "breakdown.json"
    csv_path = phase_dir / "breakdown.csv"
    svg_path = phase_dir / "breakdown.svg"
    write_json_atomic(json_path, payload)
    write_csv_atomic(csv_path, AGGREGATE_FIELDS, rows)
    write_svg(svg_path, rows)
    if phase == "validation":
        shutil.copyfile(json_path, output_dir / "performance_breakdown.json")
        shutil.copyfile(csv_path, output_dir / "performance_breakdown.csv")
        shutil.copyfile(svg_path, output_dir / "performance_breakdown.svg")
    return payload


def parse_stage_selection(values: list[str]) -> tuple[Stage, ...]:
    if not values:
        return STAGES
    requested: list[str] = []
    aliases = {
        "p0-eager": "p0_eager",
        "p0eager": "p0_eager",
        "r-eager": "r_eager",
        "reager": "r_eager",
    }
    for raw_value in values:
        for raw_item in raw_value.split(","):
            item = raw_item.strip().lower()
            if not item:
                continue
            item = aliases.get(item, item)
            requested.append(item)
    if not requested:
        raise ValueError("--stage must name at least one stage")
    if "all" in requested:
        if len(requested) != 1:
            raise ValueError("--stage all cannot be combined with other stages")
        return STAGES
    unknown = sorted(set(requested) - set(STAGE_BY_ID))
    if unknown:
        raise ValueError(
            "unknown stage(s): "
            + ", ".join(unknown)
            + "; expected p0_eager,p0,p1,p2,r_eager,p3,p4,p5,all"
        )
    if len(requested) != len(set(requested)):
        raise ValueError("--stage contains duplicates")
    requested_set = set(requested)
    return tuple(
        stage for stage in STAGES if stage.stage_id in requested_set
    )


def preflight_errors() -> list[str]:
    errors: list[str] = []
    for path, description in (
        (PYTHON, "benchmark Python"),
        (BENCHMARK_SCRIPT, "benchmark script"),
        (MODEL_PATH, "model directory"),
        (MT_BENCH_PATH, "MT-Bench dataset"),
        (REPO_ROOT / VERIFY_MODEL_PATH, "transfer-aware v3 artifact"),
    ):
        if not path.exists():
            errors.append(f"{description} does not exist: {path}")
    return errors


def run_phase(
    args: argparse.Namespace,
    phase: str,
    selected: tuple[Stage, ...],
    suite: dict[str, Any],
) -> tuple[bool, dict[str, dict[str, Any]]]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    validations: dict[str, dict[str, Any]] = {}
    phase_record = suite["phases"].setdefault(phase, {})
    for stage in selected:
        result_dir = output_dir / phase / stage.directory
        log_path = output_dir / "logs" / phase / f"{stage.stage_id}.log"
        command = benchmark_command(stage, phase, result_dir)
        started = time.time()
        resumed = False
        return_code: int | None = None
        if args.resume:
            prior = validate_result_dir(stage, phase, result_dir)
            if prior["status"] == "passed":
                validation = prior
                resumed = True
                print(
                    f"[{phase}/{stage.stage_id}] resume: valid result reused",
                    flush=True,
                )
            else:
                return_code = run_command(command, log_path)
                validation = validate_result_dir(stage, phase, result_dir)
        else:
            return_code = run_command(command, log_path)
            validation = validate_result_dir(stage, phase, result_dir)
        if return_code not in (None, 0):
            validation["status"] = "failed"
            validation["errors"].insert(
                0, f"benchmark return_code={return_code}"
            )
        validation_path = result_dir / "mechanism_validation.json"
        write_json_atomic(validation_path, validation)
        validations[stage.stage_id] = validation
        phase_record[stage.stage_id] = {
            "status": validation["status"],
            "resumed": resumed,
            "return_code": return_code,
            "elapsed_sec": time.time() - started,
            "command": command,
            "command_text": command_text(command),
            "port": stage.port,
            "result_dir": str(result_dir),
            "log_path": str(log_path),
            "validation_path": str(validation_path),
            "errors": validation["errors"],
        }
        write_json_atomic(output_dir / "run_status.json", suite)
        print(
            f"[{phase}/{stage.stage_id}] {validation['status']}",
            flush=True,
        )
        for error in validation["errors"]:
            print(f"  - {error}", flush=True)
    aggregate_phase(output_dir, phase, selected, validations)
    return (
        all(
            validation["status"] == "passed"
            for validation in validations.values()
        ),
        validations,
    )


def run(args: argparse.Namespace) -> int:
    try:
        selected = parse_stage_selection(args.stage)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    output_dir = Path(args.output_dir).expanduser().resolve()
    phases = (
        ("smoke", "validation")
        if args.phase == "all"
        else (str(args.phase),)
    )
    if args.dry_run or args.print_commands:
        for phase in phases:
            for stage in selected:
                result_dir = output_dir / phase / stage.directory
                command = benchmark_command(stage, phase, result_dir)
                print(f"[{phase}/{stage.stage_id}] {command_text(command)}")
        if args.dry_run:
            return 0

    errors = preflight_errors()
    if errors:
        for error in errors:
            print(f"preflight: {error}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    suite: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "started_utc": utc_timestamp(),
        "finished_utc": None,
        "phase": args.phase,
        "selected_stages": [stage.stage_id for stage in selected],
        "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
        "cpu_list": CPU_LIST,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "resume": bool(args.resume),
        "phases": {},
    }
    status_path = output_dir / "run_status.json"
    write_json_atomic(status_path, suite)

    overall_passed = True
    if "smoke" in phases:
        smoke_passed, _ = run_phase(
            args,
            "smoke",
            selected,
            suite,
        )
        overall_passed = overall_passed and smoke_passed
        if args.phase == "all" and not smoke_passed:
            suite["validation_gate"] = {
                "status": "blocked",
                "reason": "one or more smoke stages failed",
            }
            suite["finished_utc"] = utc_timestamp()
            suite["status"] = "failed"
            write_json_atomic(status_path, suite)
            print(
                "validation blocked because not all smoke stages passed",
                file=sys.stderr,
            )
            return 1

    if "validation" in phases:
        validation_passed, _ = run_phase(
            args,
            "validation",
            selected,
            suite,
        )
        overall_passed = overall_passed and validation_passed

    suite["finished_utc"] = utc_timestamp()
    suite["status"] = "passed" if overall_passed else "failed"
    write_json_atomic(status_path, suite)
    return 0 if overall_passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed Qwen3-30B-A3B cumulative TPOT performance "
            "breakdown and generate CSV/JSON/SVG outputs."
        )
    )
    parser.add_argument(
        "--phase",
        choices=["smoke", "validation", "all"],
        default="all",
    )
    parser.add_argument(
        "--stage",
        action="append",
        default=[],
        help=(
            "Run one or more fixed manifest stages. Repeat the option or use "
            "commas: p0_eager,p0,p1,p2,r_eager,p3,p4,p5. Default: all."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results/tpot_performance_breakdown",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse a stage only when its existing output passes validation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print exact commands without preflight checks or writes.",
    )
    parser.add_argument(
        "--print-commands",
        action="store_true",
        help="Print every selected command before starting the run.",
    )
    return parser


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
