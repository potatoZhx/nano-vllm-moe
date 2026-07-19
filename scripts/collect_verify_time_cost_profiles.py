#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "/data1/models/Qwen3-30B-A3B"
DEFAULT_PROFILE = "results/reroute_impl_20260531/offline_profile_20260531_203257.safetensors"
DEFAULT_PREDICTOR = "random_cache_srdp_scripts-1/res/run_20260614_133025"
CLEARED_MEASUREMENT_ENV = (
    "NANOVLLM_VERIFY_COST_MODEL_PROFILE",
    "NANOVLLM_VERIFY_STREAM_EVENT_TIMING",
    "NANOVLLM_VERIFY_OP_EVENT_TIMING",
    "NANOVLLM_VERIFY_BREAKDOWN_SYNC",
    "NANOVLLM_VERIFY_DEEP_PROFILE",
    "NANOVLLM_VERIFY_DEEP_PROFILE_SYNC",
    "NANOVLLM_VERIFY_SYNC_METADATA_PROFILE_READBACK",
    "NANOVLLM_VERIFY_SKIP_SYNC_METADATA_READBACK",
    "NANOVLLM_VERIFY_DISABLE_RUNTIME_METADATA",
    "NANOVLLM_VERIFY_SKIP_METADATA_OFFLOAD",
)


def _run_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(
            command,
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: str) -> dict[str, object]:
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        return {"path": str(target), "exists": False}
    stat = target.stat()
    return {
        "path": str(target),
        "exists": True,
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256(target),
    }


def _dataset_path(args: argparse.Namespace, dataset: str) -> str | None:
    defaults = {
        "sharegpt": "/data1/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json",
        "mt_bench": "/data1/datasets/mt_bench/question.jsonl",
        "humaneval": "/data1/datasets/humaneval/HumanEval.jsonl.gz",
    }
    override = getattr(args, f"{dataset}_path", "")
    return str(override or defaults.get(dataset, "")) or None


def _manifest(args: argparse.Namespace, commands: list[list[str]]) -> dict[str, object]:
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    dataset_files = {
        dataset: _file_identity(path)
        for dataset in datasets
        if (path := _dataset_path(args, dataset)) is not None
    }
    model_index = Path(args.model_path) / "model.safetensors.index.json"
    predictor_config = Path(args.acceptance_predictor_path) / "config.json"
    git_diff = _run_output(["git", "diff", "--binary"])
    return {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": _run_output(["lscpu"]),
        "gpu": _run_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        "git_commit": _run_output(["git", "rev-parse", "HEAD"]),
        "git_status": _run_output(["git", "status", "--short"]),
        "git_diff_sha256": hashlib.sha256(git_diff.encode()).hexdigest(),
        "model_index": _file_identity(str(model_index)),
        "reroute_profile": _file_identity(args.profile_artifact),
        "predictor_config": _file_identity(str(predictor_config)),
        "dataset_files": dataset_files,
        "commands": [shlex.join(command) for command in commands],
        "measurement_contract": {
            "target": "spec.verify_accept_ready_ms",
            "execution_rows": "CUDA graph bucket, including padding",
            "profile_cuda_sync": False,
            "temperature": 0.0,
            "fixed_output_tokens": int(args.output_tokens),
            "sample_offset": int(args.sample_offset),
            "draft_stop_policy": "none",
            "cleared_parent_environment": list(CLEARED_MEASUREMENT_ENV),
        },
    }


def _bench_command(
    args: argparse.Namespace,
    dataset_spec: str,
    cache_ratios: str,
    output_dir: Path,
    index: int,
) -> list[str]:
    datasets = [item for item in dataset_spec.split(",") if item]
    single_dataset = datasets[0] if len(datasets) == 1 else None
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "bench_eval_workload_tpot.py"),
        "--output-dir",
        str(
            output_dir
            / (
                (single_dataset or "workloads")
                if index == 0
                else f"{single_dataset or 'workloads'}_group{index}"
            )
        ),
        "--model-path",
        str(args.model_path),
        "--profile-artifact",
        str(args.profile_artifact),
        "--num-samples",
        "1" if single_dataset == "per_layer_slots" else str(args.num_samples),
        "--sample-offset",
        str(args.sample_offset),
        "--allocation-modes",
        "profile_weighted",
        "--cache-ratios",
        str(cache_ratios),
        "--output-lens",
        str(args.output_tokens),
        "--max-draft-tokens-values",
        str(args.draft_lengths),
        "--segment-sizes",
        str(args.segment_size),
        "--repeats",
        str(args.repeats),
        "--draft-stop-policy",
        "none",
        "--draft-tpot-verify-model-mode",
        "off",
        "--temperature",
        "0",
        "--acceptance-strategy",
        "greedy",
        "--acceptance-predictor-enabled",
        "true",
        "--acceptance-predictor-path",
        str(args.acceptance_predictor_path),
        "--kt-num-threads",
        str(args.kt_num_threads),
        "--kt-direct-backend",
        str(args.kt_backend),
        "--verify-cuda-graph-bucket-steps",
        str(args.verify_buckets),
        "--verify-prefetch-max-per-boundary",
        str(args.verify_prefetch_budget),
        "--verify-prefetch-rank-multiplier",
        "1",
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--dist-port-base",
        str(args.dist_port_base + index * 100),
        "--seed",
        str(args.seed),
        "--reset-seed-after-warmup",
        "true",
        "--verify-cost-model-profile",
        "true",
        "--skip-existing",
        "true" if args.resume else "false",
        "--fail-fast",
        "true",
        "--fail-on-output-validation-error",
        "true",
        "--reuse-engine-across-draft-lengths",
        "true" if args.reuse_engine_across_draft_lengths else "false",
        "--reuse-engine-case-order",
        "shuffle",
    ]
    if single_dataset == "per_layer_slots":
        command.extend(["--request-mode", "per_layer_slots"])
    elif single_dataset is None:
        command.extend(["--dataset-list", ",".join(datasets)])
    else:
        command.extend(["--dataset", single_dataset])
    for dataset in datasets:
        path = _dataset_path(args, dataset)
        override = getattr(args, f"{dataset}_path", "")
        if override and path:
            command.extend([f"--{dataset.replace('_', '-')}-path", path])
    return command


def run(args: argparse.Namespace) -> None:
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    allowed = {"per_layer_slots", "mt_bench", "humaneval", "sharegpt"}
    unknown = sorted(set(datasets) - allowed)
    if unknown:
        raise SystemExit(f"unsupported datasets: {unknown}")
    output_dir = Path(args.output_dir).resolve()
    if args.reuse_engine_across_draft_lengths:
        command_specs = [
            (",".join(datasets), ratio.strip())
            for ratio in str(args.cache_ratios).split(",")
            if ratio.strip()
        ]
    else:
        command_specs = [(dataset, str(args.cache_ratios)) for dataset in datasets]
    commands = [
        _bench_command(args, dataset_spec, cache_ratios, output_dir, index)
        for index, (dataset_spec, cache_ratios) in enumerate(command_specs)
    ]
    manifest = _manifest(args, commands)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "collection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    bench_env = os.environ.copy()
    for key in CLEARED_MEASUREMENT_ENV:
        bench_env.pop(key, None)
    completed = []
    for (dataset_spec, cache_ratios), command in zip(
        command_specs, commands, strict=True
    ):
        print(
            f"[collect] {dataset_spec} ratio={cache_ratios}: {shlex.join(command)}",
            flush=True,
        )
        started = time.time()
        subprocess.run(command, cwd=REPO_ROOT, env=bench_env, check=True)
        completed.append(
            {
                "datasets": dataset_spec,
                "cache_ratios": cache_ratios,
                "elapsed_sec": time.time() - started,
            }
        )
        manifest["completed"] = completed
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

    profiles_glob = str(output_dir / "**" / "sample*.json")
    validator = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "validate_verify_cost_profiles.py"),
        "--profiles",
        profiles_glob,
    ]
    subprocess.run(validator, cwd=REPO_ROOT, check=True)
    artifact = output_dir / "verify_time_cost_model.json"
    analyzer = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "analyze_verify_time_cost_model.py"),
        "--profiles",
        profiles_glob,
        "--output",
        str(artifact),
        "--kt-num-threads",
        str(args.kt_num_threads),
        "--kt-backend",
        str(args.kt_backend),
    ]
    subprocess.run(analyzer, cwd=REPO_ROOT, check=True)
    fitted = json.loads(artifact.read_text(encoding="utf-8"))
    manifest["artifact"] = str(artifact)
    manifest["accuracy_gate_passed"] = bool(fitted.get("accuracy_gate_passed"))
    manifest["completed_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not fitted.get("accuracy_gate_passed"):
        raise SystemExit("verify time model failed the configured accuracy gate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--profile-artifact", default=DEFAULT_PROFILE)
    parser.add_argument("--acceptance-predictor-path", default=DEFAULT_PREDICTOR)
    parser.add_argument(
        "--datasets",
        default="per_layer_slots,mt_bench,humaneval,sharegpt",
    )
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--cache-ratios", default="0.25,0.28125,0.3125")
    parser.add_argument("--draft-lengths", default="1,2,3,4,5,6,7,8,9,10,11,12")
    parser.add_argument("--verify-buckets", default="3,5,7,10,13")
    parser.add_argument("--segment-size", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--verify-prefetch-budget", type=int, default=10)
    parser.add_argument("--kt-num-threads", type=int, default=16)
    parser.add_argument(
        "--kt-backend",
        choices=["amx_bf16", "avx2_bf16"],
        default="avx2_bf16",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.99)
    parser.add_argument("--dist-port-base", type=int, default=33800)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--sharegpt-path", default="")
    parser.add_argument("--mt-bench-path", default="")
    parser.add_argument("--humaneval-path", default="")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--reuse-engine-across-draft-lengths",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
