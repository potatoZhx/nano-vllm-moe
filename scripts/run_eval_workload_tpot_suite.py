#!/usr/bin/env python3
"""Run the requested TPOT benchmark datasets in failure-isolated processes.

Each dataset is written to its own directory so that a failed run cannot
overwrite another dataset's outputs.  The suite status and merged CSV files are
refreshed after every dataset:

* ``tpot_summary.csv``: one aggregate TPOT row per dataset (or a failure row)
* ``tpot_rows.csv``: all available per-request TPOT rows
* ``suite_status.json``: commands, return codes, timings, and output locations

The fixed K6/vpb4 configuration forces CUDA device 2, logical CPUs 64-96, and
the 70-row MMLU-Pro validation split.  The existing K12 transfer-aware
configuration remains available.  If the benchmark environment cannot read
MMLU-Pro parquet, the suite converts it once to JSONL with an available local
pyarrow installation.  A subprocess, preparation, or result-validation failure
is recorded but does not stop the remaining datasets.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = REPO_ROOT / "scripts" / "bench_eval_workload_tpot.py"
K12_OUTPUT_DIR = REPO_ROOT / "results" / "transfer_v3_active_screen_u000_20260719"
K6_VPB4_OUTPUT_DIR = (
    REPO_ROOT / "results" / "k6_vpb4_eval_workloads_seed20260719_20260720"
)
DEFAULT_SUITE_CONFIG = "k12_transfer_step"
SUITE_CONFIG_CHOICES = (DEFAULT_SUITE_CONFIG, "k6_vpb4")
OUTPUT_DIR_BY_CONFIG = {
    DEFAULT_SUITE_CONFIG: K12_OUTPUT_DIR,
    "k6_vpb4": K6_VPB4_OUTPUT_DIR,
}
DEFAULT_DATASETS = ("mmlu_pro", "mt_bench", "humaneval")
SUPPORTED_DATASETS = frozenset(DEFAULT_DATASETS)
NANO_MOE_PYTHON = Path("/home/linke/miniconda3/envs/nano_moe/bin/python")
DEFAULT_PYTHON = NANO_MOE_PYTHON if NANO_MOE_PYTHON.is_file() else Path(sys.executable)
DEFAULT_MMLU_PRO_SPLIT = "validation"
MMLU_PRO_PARQUET_BY_SPLIT = {
    split: Path(f"/data1/datasets/mmlu_pro/{split}-00000-of-00001.parquet")
    for split in ("validation", "test")
}
K6_VPB4_EXPECTED_SAMPLES = {
    "mmlu_pro": 70,
    "mt_bench": 80,
    "humaneval": 164,
}
K6_VPB4_CUDA_VISIBLE_DEVICES = "2"
K6_VPB4_CPU_LIST = "64-96"
FIXED_SEED = 20260719
FIXED_DIST_PORT_BASE = 37970
SUMMARY_FIELDS = (
    "dataset",
    "optimized_config",
    "allocation_mode",
    "segment_size",
    "cache_ratio",
    "max_output_tokens",
    "ignore_eos",
    "max_draft_tokens",
    "draft_stop_policy",
    "verify_prefetch_max_per_boundary",
    "repeat",
    "sample_count",
    "ok_count",
    "tpot_ms_mean",
    "tpot_ms_p50",
    "tpot_ms_p90",
    "tpot_ms_p99",
    "decode_tok_s_mean",
    "throughput_output_tok_s_mean",
    "prompt_tokens_mean",
    "generated_output_tokens_mean",
)
SUITE_SUMMARY_FIELDS = (
    "suite_status",
    "return_code",
    "elapsed_sec",
    "log_path",
    "result_dir",
    *SUMMARY_FIELDS,
)


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parse_datasets(value: str) -> list[str]:
    datasets = [item.strip() for item in value.split(",") if item.strip()]
    if not datasets:
        raise argparse.ArgumentTypeError("--datasets must not be empty")
    unknown = sorted(set(datasets) - SUPPORTED_DATASETS)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unsupported datasets: {', '.join(unknown)}"
        )
    if len(datasets) != len(set(datasets)):
        raise argparse.ArgumentTypeError("--datasets must not contain duplicates")
    return datasets


def suite_config(args: argparse.Namespace) -> str:
    return str(getattr(args, "suite_config", DEFAULT_SUITE_CONFIG))


def output_dir_for_args(args: argparse.Namespace) -> Path:
    configured = str(getattr(args, "output_dir", "") or "")
    if configured:
        return Path(configured).expanduser().resolve()
    return OUTPUT_DIR_BY_CONFIG[suite_config(args)].resolve()


def cpu_list_for_args(args: argparse.Namespace) -> str:
    if suite_config(args) == "k6_vpb4":
        return K6_VPB4_CPU_LIST
    return str(args.cpu_list)


def cuda_visible_devices_for_args(args: argparse.Namespace) -> str:
    if suite_config(args) == "k6_vpb4":
        return K6_VPB4_CUDA_VISIBLE_DEVICES
    return str(args.cuda_visible_devices)


def mmlu_split_for_args(args: argparse.Namespace) -> str:
    if suite_config(args) == "k6_vpb4":
        return DEFAULT_MMLU_PRO_SPLIT
    return str(args.mmlu_split)


def benchmark_command(
    args: argparse.Namespace,
    dataset: str,
    result_dir: Path,
    *,
    mmlu_pro_path: Path | None = None,
) -> list[str]:
    config = suite_config(args)
    command = [
        "taskset",
        "--cpu-list",
        cpu_list_for_args(args),
        str(Path(args.python).expanduser()),
        str(BENCHMARK_SCRIPT),
        "--model-path",
        str(args.model_path),
        "--dataset",
        dataset,
        "--request-mode",
        "dataset",
        "--num-samples",
        "all" if config == "k6_vpb4" else str(args.num_samples),
        "--output-dir",
        str(result_dir),
    ]
    if config == "k6_vpb4":
        command.extend(
            [
                "--optimized-config",
                "k6_decode",
                "--cache-ratios",
                "0.3125",
                "--output-lens",
                "512",
                "--max-draft-tokens-values",
                "6",
                "--segment-sizes",
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
                "3,5,7,10,13",
                "--verify-prefetch-max-per-boundary",
                "4",
                "--verify-prefetch-rank-multiplier",
                "1",
                "--draft-stop-policy",
                "none",
                "--acceptance-predictor-enabled",
                "false",
                "--gpu-memory-utilization",
                "0.99",
                "--temperature",
                "0.8",
                "--acceptance-strategy",
                "standard_sampling",
                "--decode-driver",
                "generate",
                "--reuse-engine-across-draft-lengths",
                "true",
                "--collect-profile",
                "false",
                "--engine-profile",
                "false",
                "--engine-profile-cuda-sync",
                "false",
                "--verify-cost-model-profile",
                "false",
                "--transfer-aware-profile",
                "false",
                "--save-profile-json",
                "false",
                "--save-token-ids",
                "true",
                "--save-text",
                "true",
                "--reset-profile-after-warmup",
                "false",
                "--reset-profile-before-request",
                "false",
                "--reset-seed-after-warmup",
                "true",
                "--repeats",
                "1",
                "--repeat-index-offset",
                "0",
                "--skip-existing",
                "false",
                "--fail-fast",
                "true",
                "--fail-on-output-validation-error",
                "true",
                "--seed",
                str(FIXED_SEED),
                "--dist-port-base",
                str(FIXED_DIST_PORT_BASE),
            ]
        )
    else:
        command.extend(
            [
                "--optimized-config",
                "k12_transfer_step",
                "--cache-ratios",
                "0.3125",
                "--output-lens",
                "512",
                "--max-draft-tokens-values",
                "12",
                "--segment-sizes",
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
                "--draft-stop-policy",
                "tpot",
                "--draft-tpot-stop-rule",
                "transfer_aware_step",
                "--draft-tpot-min-steps",
                "6",
                "--draft-tpot-verify-model-mode",
                "active",
                "--draft-tpot-verify-model-path",
                "results/transfer_v3_artifact_20260719/verify_cost_v3.json",
                "--draft-tpot-uncertainty-scale",
                "0.0",
                "--acceptance-predictor-enabled",
                "true",
                "--decode-driver",
                "generate",
                "--reuse-engine-across-draft-lengths",
                "true",
                "--collect-profile",
                "true",
                "--engine-profile",
                "false",
                "--engine-profile-cuda-sync",
                "false",
                "--verify-cost-model-profile",
                "false",
                "--transfer-aware-profile",
                "false",
                "--save-profile-json",
                "true",
                "--save-token-ids",
                "true",
                "--save-text",
                "true",
                "--reset-seed-after-warmup",
                "true",
                "--reset-profile-before-request",
                "true",
                "--skip-existing",
                "false",
                "--fail-fast",
                "true",
                "--seed",
                str(FIXED_SEED),
                "--dist-port-base",
                str(FIXED_DIST_PORT_BASE),
            ]
        )
    if dataset == "mmlu_pro" and mmlu_pro_path is not None:
        command.extend(["--mmlu-pro-path", str(mmlu_pro_path)])
    return command


def converter_candidates(args: argparse.Namespace) -> list[Path]:
    candidates: list[Path] = []
    if args.mmlu_converter_python:
        candidates.append(Path(args.mmlu_converter_python).expanduser())
    candidates.extend(
        [
            Path(args.python).expanduser(),
            Path(sys.executable),
            Path("/home/linke/miniconda3/envs/ktransformers/bin/python"),
        ]
    )
    candidates.extend(
        sorted(Path.home().glob("miniconda3/envs/*/bin/python"))
    )
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen and resolved.is_file():
            seen.add(resolved)
            unique.append(resolved)
    return unique


def python_has_pyarrow(python: Path, cpu_list: str) -> bool:
    probe = subprocess.run(
        [
            "taskset",
            "--cpu-list",
            cpu_list,
            str(python),
            "-c",
            "import pyarrow.parquet",
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return probe.returncode == 0


def ensure_mmlu_jsonl(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    if args.mmlu_pro_path:
        supplied = Path(args.mmlu_pro_path).expanduser().resolve()
        if not supplied.is_file():
            raise FileNotFoundError(f"MMLU-Pro input does not exist: {supplied}")
        supplied_record: dict[str, Any] = {
            "status": "supplied",
            "path": str(supplied),
        }
        if supplied.suffix in {".jsonl", ".json"}:
            supplied_record["row_count"] = count_jsonl_rows(supplied)
        return supplied, supplied_record

    requested_split = mmlu_split_for_args(args)
    parquet_override = (
        str(args.mmlu_pro_parquet)
        if suite_config(args) != "k6_vpb4"
        else ""
    )
    source = Path(
        parquet_override or MMLU_PRO_PARQUET_BY_SPLIT[requested_split]
    ).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"MMLU-Pro parquet does not exist: {source}")
    source_split = source.name.split("-", 1)[0]
    if source_split not in MMLU_PRO_PARQUET_BY_SPLIT:
        source_split = "custom"
    destination = output_dir / "inputs" / f"mmlu_pro_{source_split}.jsonl"
    if (
        destination.is_file()
        and destination.stat().st_mtime_ns >= source.stat().st_mtime_ns
    ):
        return destination, {
            "status": "reused",
            "split": source_split,
            "source": str(source),
            "path": str(destination),
            "row_count": count_jsonl_rows(destination),
        }

    converter = next(
        (
            python
            for python in converter_candidates(args)
            if python_has_pyarrow(python, cpu_list_for_args(args))
        ),
        None,
    )
    if converter is None:
        raise RuntimeError(
            "no local Python with pyarrow is available to convert MMLU-Pro"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".jsonl.tmp")
    conversion_code = (
        "import json, pathlib, pyarrow.parquet as pq, sys; "
        "source, output = map(pathlib.Path, sys.argv[1:3]); "
        "rows = pq.read_table(source).to_pylist(); "
        "stream = output.open('w', encoding='utf-8'); "
        "[stream.write(json.dumps(row, ensure_ascii=False) + '\\n') for row in rows]; "
        "stream.close()"
    )
    command = [
        "taskset",
        "--cpu-list",
        cpu_list_for_args(args),
        str(converter),
        "-c",
        conversion_code,
        str(source),
        str(temporary),
    ]
    print(f"[suite] converting MMLU-Pro: {shlex.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0 or not temporary.is_file():
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"MMLU-Pro conversion failed with return code {completed.returncode}"
        )
    temporary.replace(destination)
    row_count = count_jsonl_rows(destination)
    return destination, {
        "status": "converted",
        "split": source_split,
        "source": str(source),
        "path": str(destination),
        "converter_python": str(converter),
        "row_count": row_count,
    }


def count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_k6_vpb4_results(
    result_dir: Path,
    dataset: str,
) -> dict[str, Any]:
    expected_count = K6_VPB4_EXPECTED_SAMPLES[dataset]
    rows_path = result_dir / "rows.csv"
    rows = read_csv_rows(rows_path)
    errors: list[str] = []
    if not rows_path.is_file():
        errors.append("rows.csv is missing")
    if len(rows) != expected_count:
        errors.append(f"row_count={len(rows)} expected={expected_count}")

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if len(ok_rows) != expected_count:
        errors.append(f"ok_count={len(ok_rows)} expected={expected_count}")

    wrong_lengths: list[int] = []
    output_errors: list[int] = []
    for index, row in enumerate(ok_rows):
        try:
            generated = int(row.get("generated_output_tokens", ""))
        except (TypeError, ValueError):
            generated = -1
        if generated != 512:
            wrong_lengths.append(index)
        if str(row.get("output_validation_error", "")).strip():
            output_errors.append(index)
    if wrong_lengths:
        errors.append(
            f"non_512_output_rows={len(wrong_lengths)} "
            f"first_index={wrong_lengths[0]}"
        )
    if output_errors:
        errors.append(
            f"output_validation_error_rows={len(output_errors)} "
            f"first_index={output_errors[0]}"
        )

    first_output = None
    if rows:
        first_output = {
            "status": rows[0].get("status", ""),
            "generated_output_tokens": rows[0].get(
                "generated_output_tokens", ""
            ),
            "output_validation_error": rows[0].get(
                "output_validation_error", ""
            ),
        }
    return {
        "status": "passed" if not errors else "failed",
        "expected_count": expected_count,
        "row_count": len(rows),
        "ok_count": len(ok_rows),
        "all_ok_outputs_are_512_tokens": not wrong_lengths,
        "output_validation_error_count": len(output_errors),
        "first_output": first_output,
        "errors": errors,
    }


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


def aggregate_results(
    output_dir: Path,
    datasets: list[str],
    run_by_dataset: dict[str, dict[str, Any]],
) -> None:
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    detail_fields: list[str] = ["suite_status", "return_code", "result_dir"]

    for dataset in datasets:
        run = run_by_dataset.get(dataset, {})
        result_dir = Path(
            run.get("result_dir", output_dir / "datasets" / dataset)
        )
        summaries = read_csv_rows(result_dir / "summary.csv")
        details = read_csv_rows(result_dir / "rows.csv")
        suite_fields = {
            "suite_status": run.get("status", "pending"),
            "return_code": run.get("return_code", ""),
            "elapsed_sec": run.get("elapsed_sec", ""),
            "log_path": run.get(
                "log_path", str(output_dir / "logs" / f"{dataset}.log")
            ),
            "result_dir": str(result_dir),
        }

        if summaries:
            for row in summaries:
                summary_rows.append({**suite_fields, **row})
        else:
            summary_rows.append({**suite_fields, "dataset": dataset})

        detail_suite_fields = {
            "suite_status": suite_fields["suite_status"],
            "return_code": suite_fields["return_code"],
            "result_dir": suite_fields["result_dir"],
        }
        for row in details:
            detail_rows.append({**detail_suite_fields, **row})
            for field in row:
                if field not in detail_fields:
                    detail_fields.append(field)

    write_csv_atomic(
        output_dir / "tpot_summary.csv",
        SUITE_SUMMARY_FIELDS,
        summary_rows,
    )
    write_csv_atomic(
        output_dir / "tpot_rows.csv",
        detail_fields,
        detail_rows,
    )


def run_command(command: list[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {shlex.join(command)}\n")
        log.flush()
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


def run(args: argparse.Namespace) -> int:
    output_dir = output_dir_for_args(args)
    datasets = list(args.datasets)
    config = suite_config(args)
    mmlu_split = mmlu_split_for_args(args)
    cuda_visible_devices = cuda_visible_devices_for_args(args)
    cpu_list = cpu_list_for_args(args)
    if args.dry_run:
        for dataset in datasets:
            mmlu_pro_path = (
                Path(args.mmlu_pro_path).expanduser()
                if args.mmlu_pro_path
                else output_dir / "inputs" / f"mmlu_pro_{mmlu_split}.jsonl"
            )
            command = benchmark_command(
                args,
                dataset,
                output_dir / "datasets" / dataset,
                mmlu_pro_path=mmlu_pro_path,
            )
            print(f"[{dataset}] {shlex.join(command)}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "suite_status.json"
    suite: dict[str, Any] = {
        "schema_version": 1,
        "started_utc": utc_timestamp(),
        "finished_utc": None,
        "driver_pid": os.getpid(),
        "suite_config": config,
        "cuda_visible_devices": cuda_visible_devices,
        "cpu_list": cpu_list,
        "kt_num_threads": 16,
        "seed": FIXED_SEED,
        "repeat_count": 1,
        "repeat_index_offset": 0,
        "dist_port_base": FIXED_DIST_PORT_BASE,
        "datasets": datasets,
        "expected_sample_counts": (
            K6_VPB4_EXPECTED_SAMPLES if config == "k6_vpb4" else None
        ),
        "mmlu_pro_split": mmlu_split,
        "mmlu_pro_input": None,
        "runs": [],
    }
    run_by_dataset: dict[str, dict[str, Any]] = {}
    write_json_atomic(status_path, suite)
    aggregate_results(output_dir, datasets, run_by_dataset)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    for index, dataset in enumerate(datasets, start=1):
        result_dir = output_dir / "datasets" / dataset
        log_path = output_dir / "logs" / f"{dataset}.log"
        record: dict[str, Any] = {
            "dataset": dataset,
            "status": "preparing",
            "return_code": None,
            "started_utc": utc_timestamp(),
            "finished_utc": None,
            "elapsed_sec": None,
            "command": [],
            "command_shell": "",
            "log_path": str(log_path),
            "result_dir": str(result_dir),
        }
        suite["runs"].append(record)
        run_by_dataset[dataset] = record
        write_json_atomic(status_path, suite)
        aggregate_results(output_dir, datasets, run_by_dataset)

        mmlu_pro_path = None
        if dataset == "mmlu_pro":
            preparation_started = time.monotonic()
            try:
                mmlu_pro_path, input_record = ensure_mmlu_jsonl(args, output_dir)
                suite["mmlu_pro_input"] = input_record
                if config == "k6_vpb4":
                    actual_rows = input_record.get("row_count")
                    expected_rows = K6_VPB4_EXPECTED_SAMPLES["mmlu_pro"]
                    if actual_rows != expected_rows:
                        raise RuntimeError(
                            "MMLU-Pro validation input must contain exactly "
                            f"{expected_rows} rows; got {actual_rows!r}"
                        )
            except Exception as error:
                record["status"] = "failed"
                record["return_code"] = 125
                record["finished_utc"] = utc_timestamp()
                record["elapsed_sec"] = round(
                    time.monotonic() - preparation_started, 3
                )
                record["error"] = f"MMLU-Pro preparation failed: {error}"
                suite["mmlu_pro_input"] = {
                    **(suite.get("mmlu_pro_input") or {}),
                    "status": "failed",
                    "error": str(error),
                }
                write_json_atomic(status_path, suite)
                aggregate_results(output_dir, datasets, run_by_dataset)
                print(
                    f"[suite {index}/{len(datasets)}] dataset={dataset} "
                    f"status=failed preparation_error={error}",
                    flush=True,
                )
                continue

        command = benchmark_command(
            args,
            dataset,
            result_dir,
            mmlu_pro_path=mmlu_pro_path,
        )
        record["status"] = "running"
        record["command"] = command
        record["command_shell"] = shlex.join(command)
        write_json_atomic(status_path, suite)
        aggregate_results(output_dir, datasets, run_by_dataset)
        print(
            f"[suite {index}/{len(datasets)}] dataset={dataset} "
            f"CUDA_VISIBLE_DEVICES={cuda_visible_devices} "
            f"cpus={cpu_list}",
            flush=True,
        )
        started = time.monotonic()
        return_code = run_command(command, log_path, env)
        record["elapsed_sec"] = round(time.monotonic() - started, 3)
        record["return_code"] = return_code
        record["finished_utc"] = utc_timestamp()
        summary_exists = (result_dir / "summary.csv").is_file()
        validation = None
        if return_code == 0 and summary_exists and config == "k6_vpb4":
            validation = validate_k6_vpb4_results(result_dir, dataset)
            record["validation"] = validation
        record["status"] = (
            "completed"
            if (
                return_code == 0
                and summary_exists
                and (validation is None or validation["status"] == "passed")
            )
            else "failed"
        )
        if return_code == 0 and not summary_exists:
            record["error"] = "benchmark returned 0 but summary.csv is missing"
        elif validation is not None and validation["status"] != "passed":
            record["error"] = "result validation failed: " + "; ".join(
                validation["errors"]
            )
        write_json_atomic(status_path, suite)
        aggregate_results(output_dir, datasets, run_by_dataset)
        print(
            f"[suite {index}/{len(datasets)}] dataset={dataset} "
            f"status={record['status']} return_code={return_code} "
            f"elapsed={record['elapsed_sec']:.1f}s",
            flush=True,
        )

    suite["finished_utc"] = utc_timestamp()
    failed = [
        run["dataset"] for run in suite["runs"] if run["status"] != "completed"
    ]
    suite["status"] = "failed" if failed else "completed"
    suite["failed_datasets"] = failed
    write_json_atomic(status_path, suite)
    aggregate_results(output_dir, datasets, run_by_dataset)
    print(f"suite_status={suite['status']}", flush=True)
    print(f"suite_status_json={status_path}", flush=True)
    print(f"tpot_summary_csv={output_dir / 'tpot_summary.csv'}", flush=True)
    print(f"tpot_rows_csv={output_dir / 'tpot_rows.csv'}", flush=True)
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run mmlu_pro, mt_bench, and humaneval TPOT benchmarks in "
            "failure-isolated subprocesses."
        )
    )
    parser.add_argument(
        "--suite-config",
        "--config",
        dest="suite_config",
        choices=SUITE_CONFIG_CHOICES,
        default=DEFAULT_SUITE_CONFIG,
        help=(
            "Benchmark command preset. k12_transfer_step preserves the existing "
            "active-stop suite; k6_vpb4 selects the fixed K=6/vpb=4 suite."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "Suite root. Each dataset is saved under datasets/<name>. When "
            "omitted, a distinct path is selected for the chosen suite config."
        ),
    )
    parser.add_argument(
        "--datasets",
        type=parse_datasets,
        default=list(DEFAULT_DATASETS),
        help="Comma-separated dataset order.",
    )
    parser.add_argument(
        "--num-samples",
        default="all",
        help="Requests per dataset; defaults to the full dataset.",
    )
    parser.add_argument("--model-path", default="/data1/models/Qwen3-30B-A3B")
    parser.add_argument(
        "--mmlu-pro-path",
        default="",
        help="Optional ready-to-use MMLU-Pro JSONL or parquet input.",
    )
    parser.add_argument(
        "--mmlu-split",
        choices=("validation", "test"),
        default=DEFAULT_MMLU_PRO_SPLIT,
        help=(
            "MMLU-Pro split used when --mmlu-pro-path is empty. k6_vpb4 "
            "always uses validation."
        ),
    )
    parser.add_argument(
        "--mmlu-pro-parquet",
        default="",
        help=(
            "Optional parquet override converted to JSONL when "
            "--mmlu-pro-path is empty. By default it follows --mmlu-split."
        ),
    )
    parser.add_argument(
        "--mmlu-converter-python",
        default="",
        help="Optional Python with pyarrow for the one-time parquet conversion.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default="2",
        help="CUDA devices for K12; k6_vpb4 always forces device 2.",
    )
    parser.add_argument(
        "--cpu-list",
        default="64-96",
        help="CPU affinity for K12; k6_vpb4 always forces 64-96.",
    )
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
