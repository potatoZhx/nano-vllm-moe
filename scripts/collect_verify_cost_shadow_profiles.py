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
DEFAULT_PROFILE = (
    "results/reroute_impl_20260531/"
    "offline_profile_20260531_203257.safetensors"
)
DEFAULT_PREDICTOR = "random_cache_srdp_scripts-1/res/run_20260614_133025"
CLEARED_MEASUREMENT_ENV = (
    "NANOVLLM_VERIFY_COST_MODEL_PROFILE",
    "NANOVLLM_VERIFY_OP_EVENT_TIMING",
    "NANOVLLM_VERIFY_BREAKDOWN_SYNC",
    "NANOVLLM_VERIFY_DEEP_PROFILE",
    "NANOVLLM_VERIFY_DEEP_PROFILE_SYNC",
    "NANOVLLM_VERIFY_SYNC_METADATA_PROFILE_READBACK",
    "NANOVLLM_VERIFY_SKIP_SYNC_METADATA_READBACK",
    "NANOVLLM_VERIFY_DISABLE_RUNTIME_METADATA",
    "NANOVLLM_VERIFY_SKIP_METADATA_OFFLOAD",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: str | Path) -> dict[str, object]:
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


def _output(command: list[str]) -> str:
    try:
        return subprocess.check_output(
            command,
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _dataset_path(args: argparse.Namespace, dataset: str) -> str | None:
    defaults = {
        "sharegpt": (
            "/data1/datasets/sharegpt/"
            "ShareGPT_V3_unfiltered_cleaned_split.json"
        ),
        "mt_bench": "/data1/datasets/mt_bench/question.jsonl",
        "humaneval": "/data1/datasets/humaneval/HumanEval.jsonl.gz",
    }
    override = str(getattr(args, f"{dataset}_path", "") or "")
    return override or defaults.get(dataset)


def _command(
    args: argparse.Namespace,
    *,
    ratio: str,
    output_dir: Path,
    index: int,
) -> list[str]:
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "bench_eval_workload_tpot.py"),
        "--output-dir",
        str(output_dir / f"workloads_ratio_{ratio.replace('.', '')}"),
        "--model-path",
        str(args.model_path),
        "--profile-artifact",
        str(args.profile_artifact),
        "--dataset-list",
        ",".join(datasets),
        "--num-samples",
        str(args.num_samples),
        "--sample-offset",
        str(args.sample_offset),
        "--allocation-modes",
        "profile_weighted",
        "--cache-ratios",
        ratio,
        "--output-lens",
        str(args.output_tokens),
        "--max-draft-tokens-values",
        str(args.draft_lengths),
        "--segment-sizes",
        str(args.segment_size),
        "--repeats",
        "1",
        "--draft-stop-policy",
        "none",
        "--draft-tpot-verify-model-mode",
        "shadow",
        "--draft-tpot-verify-model-path",
        str(Path(args.artifact).resolve()),
        "--temperature",
        str(args.temperature),
        "--acceptance-strategy",
        str(args.acceptance_strategy),
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
        "--repeat-index-offset",
        str(args.repeat_index_offset),
        "--reset-seed-after-warmup",
        "true",
        "--verify-cost-model-profile",
        "true" if args.verify_workload_proxy_calibration else "false",
        "--engine-profile",
        "true",
        "--engine-profile-cuda-sync",
        "false",
        "--collect-profile",
        "true",
        "--save-profile-json",
        "true",
        "--skip-existing",
        "true" if args.resume else "false",
        "--fail-fast",
        "true",
        "--fail-on-output-validation-error",
        "true",
        "--reuse-engine-across-draft-lengths",
        "true",
        "--reuse-engine-case-order",
        "shuffle",
    ]
    for dataset in datasets:
        override = str(getattr(args, f"{dataset}_path", "") or "")
        if override:
            command.extend(
                [f"--{dataset.replace('_', '-')}-path", str(Path(override).resolve())]
            )
    return command


def run(args: argparse.Namespace) -> None:
    artifact_path = Path(args.artifact).resolve()
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not bool(artifact.get("accuracy_gate_passed")):
        raise SystemExit("training artifact did not pass its accuracy gate")
    if not artifact.get("model_id"):
        raise SystemExit("training artifact lacks model_id")

    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    allowed = {"per_layer_slots", "mt_bench", "humaneval", "sharegpt"}
    unknown = sorted(set(datasets) - allowed)
    if unknown:
        raise SystemExit(f"unsupported datasets: {unknown}")
    ratios = [item.strip() for item in args.cache_ratios.split(",") if item.strip()]
    output_dir = Path(args.output_dir).resolve()
    commands = [
        _command(args, ratio=ratio, output_dir=output_dir, index=index)
        for index, ratio in enumerate(ratios)
    ]
    diff = _output(["git", "diff", "--binary"])
    manifest: dict[str, object] = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": _output(["lscpu"]),
        "gpu": _output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        "git_commit": _output(["git", "rev-parse", "HEAD"]),
        "git_status": _output(["git", "status", "--short"]),
        "git_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "artifact": _identity(artifact_path),
        "model_index": _identity(
            Path(args.model_path) / "model.safetensors.index.json"
        ),
        "reroute_profile": _identity(args.profile_artifact),
        "predictor_config": _identity(
            Path(args.acceptance_predictor_path) / "config.json"
        ),
        "datasets": {
            dataset: _identity(path)
            for dataset in datasets
            if (path := _dataset_path(args, dataset)) is not None
        },
        "commands": [shlex.join(command) for command in commands],
        "measurement_contract": {
            "target": "spec.verify_accept_ready_ms",
            "verify_workload_instrumentation": bool(
                args.verify_workload_proxy_calibration
            ),
            "stream_event_timing": True,
            "profile_cuda_sync": False,
            "temperature": float(args.temperature),
            "acceptance_strategy": str(args.acceptance_strategy),
            "draft_stop_policy": "none",
            "draft_lengths": str(args.draft_lengths),
            "sample_offset": int(args.sample_offset),
            "cleared_parent_environment": list(CLEARED_MEASUREMENT_ENV),
        },
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "shadow_collection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env["NANOVLLM_VERIFY_STREAM_EVENT_TIMING"] = "1"
    for key in CLEARED_MEASUREMENT_ENV:
        env.pop(key, None)

    completed = []
    for ratio, command in zip(ratios, commands, strict=True):
        print(f"[shadow] ratio={ratio}: {shlex.join(command)}", flush=True)
        started = time.time()
        subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
        completed.append({"cache_ratio": ratio, "elapsed_sec": time.time() - started})
        manifest["completed"] = completed
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

    profiles_glob = str(output_dir / "**" / "sample*.json")
    gate_returncode = 0
    if args.verify_workload_proxy_calibration:
        validator = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "validate_verify_cost_profiles.py"),
            "--profiles",
            profiles_glob,
        ]
        subprocess.run(validator, cwd=REPO_ROOT, check=True)
        proxy_artifact = output_dir / "verify_time_cost_model.proxy.json"
        analyzer = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "analyze_verify_workload_proxy.py"),
            "--base-artifact",
            str(artifact_path),
            "--profiles",
            profiles_glob,
            "--output",
            str(proxy_artifact),
        ]
        analyzer_result = subprocess.run(analyzer, cwd=REPO_ROOT, check=False)
        gate_returncode = int(analyzer_result.returncode)
        if not proxy_artifact.is_file():
            raise SystemExit(
                f"proxy analyzer failed without artifact (exit {gate_returncode})"
            )
        proxy_model = json.loads(proxy_artifact.read_text(encoding="utf-8"))
        manifest["proxy_workload_gate_passed"] = bool(
            proxy_model.get("proxy_workload_gate_passed")
        )
        manifest["proxy_artifact"] = _identity(proxy_artifact)
    elif args.defer_shadow_validation:
        manifest["validation_deferred"] = True
    else:
        validation_path = output_dir / "shadow_validation.json"
        validated_artifact = output_dir / "verify_time_cost_model.validated.json"
        validator = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "validate_verify_cost_shadow.py"),
            "--profiles",
            profiles_glob,
            "--artifact",
            str(artifact_path),
            "--output-artifact",
            str(validated_artifact),
            "--output",
            str(validation_path),
            "--protocol",
            str(args.protocol),
            "--deployment-field",
            str(args.deployment_field),
        ]
        if args.allow_return_logits:
            validator.append("--allow-return-logits")
        validator_result = subprocess.run(validator, cwd=REPO_ROOT, check=False)
        gate_returncode = int(validator_result.returncode)
        if not validation_path.is_file() or not validated_artifact.is_file():
            raise SystemExit(
                f"shadow validator failed without reports (exit {gate_returncode})"
            )
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        manifest["validation"] = validation["deployment_validation"]
        manifest["validated_artifact"] = _identity(validated_artifact)
    manifest["completed_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ",
        time.gmtime(),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if gate_returncode != 0:
        raise SystemExit(gate_returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
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
    parser.add_argument("--output-tokens", type=int, default=96)
    parser.add_argument("--cache-ratios", default="0.25,0.28125,0.3125")
    parser.add_argument("--draft-lengths", default="1,3,5,8,12")
    parser.add_argument("--verify-buckets", default="3,5,7,10,13")
    parser.add_argument("--segment-size", type=int, default=12)
    parser.add_argument("--verify-prefetch-budget", type=int, default=10)
    parser.add_argument("--kt-num-threads", type=int, default=16)
    parser.add_argument(
        "--kt-backend",
        choices=["amx_bf16", "avx2_bf16"],
        default="avx2_bf16",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.99)
    parser.add_argument("--dist-port-base", type=int, default=34800)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--repeat-index-offset", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--acceptance-strategy",
        choices=["greedy", "standard_sampling"],
        default="greedy",
    )
    parser.add_argument("--protocol", default="unspecified")
    parser.add_argument(
        "--deployment-field",
        default="deployment_validation",
    )
    parser.add_argument("--allow-return-logits", action="store_true")
    parser.add_argument("--sharegpt-path", default="")
    parser.add_argument("--mt-bench-path", default="")
    parser.add_argument("--humaneval-path", default="")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--defer-shadow-validation",
        action="store_true",
        help="Collect profiles and defer the gate so independent runs can be pooled.",
    )
    parser.add_argument(
        "--verify-workload-proxy-calibration",
        action="store_true",
        help=(
            "Collect both draft-route proxy features and actual execution workload, "
            "then fit the proxy workload stage instead of issuing a shadow gate."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
