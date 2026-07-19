#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


def _command(
    args: argparse.Namespace,
    *,
    policy: dict[str, object],
    ratio: str,
    repeat: int,
    command_index: int,
    output_dir: Path,
) -> list[str]:
    name = str(policy["name"])
    destination = (
        output_dir
        / name
        / f"ratio_{ratio.replace('.', '')}"
        / f"repeat_{repeat}"
    )
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "bench_eval_workload_tpot.py"),
        "--output-dir",
        str(destination),
        "--model-path",
        str(args.model_path),
        "--profile-artifact",
        str(args.profile_artifact),
        "--dataset-list",
        str(args.datasets),
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
        str(policy["max_draft_tokens"]),
        "--segment-sizes",
        str(args.segment_size),
        "--repeats",
        "1",
        "--repeat-index-offset",
        str(repeat),
        "--draft-stop-policy",
        str(policy["stop_policy"]),
        "--draft-tpot-stop-rule",
        str(policy["stop_rule"]),
        "--draft-tpot-cost-model",
        str(policy["cost_model"]),
        "--draft-tpot-td-ms",
        str(policy["td_ms"]),
        "--draft-tpot-tv-ms",
        str(policy["tv_ms"]),
        "--draft-tpot-min-steps",
        str(policy["min_steps"]),
        "--draft-tpot-stop-margin",
        str(policy["stop_margin"]),
        "--draft-tpot-stop-patience",
        str(policy["stop_patience"]),
        "--draft-tpot-lookahead-cache-credit-ms-per-step",
        str(policy["lookahead_cache_credit_ms_per_step"]),
        "--draft-tpot-verify-model-mode",
        str(policy["verify_model_mode"]),
        "--temperature",
        str(args.temperature),
        "--acceptance-strategy",
        str(args.acceptance_strategy),
        "--acceptance-predictor-enabled",
        "true" if bool(policy["acceptance_predictor_enabled"]) else "false",
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
        str(args.dist_port_base + command_index * 20),
        "--seed",
        str(args.seed),
        "--reset-seed-after-warmup",
        "true",
        "--verify-cost-model-profile",
        "false",
        "--engine-profile",
        "false",
        "--collect-profile",
        "false",
        "--save-text",
        "true" if args.save_text else "false",
        "--save-token-ids",
        "true" if args.save_token_ids else "false",
        "--skip-existing",
        "true" if args.resume else "false",
        "--fail-fast",
        "true",
        "--fail-on-output-validation-error",
        "true",
        "--reuse-engine-across-draft-lengths",
        "true",
        "--reuse-engine-case-order",
        "declared",
    ]
    if str(policy["verify_model_mode"]) != "off":
        command.extend(
            [
                "--draft-tpot-verify-model-path",
                str(policy["verify_model_path"]),
            ]
        )
    if str(policy["alpha_calibration_path"]):
        command.extend(
            [
                "--draft-tpot-alpha-calibration-path",
                str(policy["alpha_calibration_path"]),
            ]
        )
    for dataset in ("sharegpt", "mt_bench", "humaneval"):
        override = str(getattr(args, f"{dataset}_path", "") or "")
        if override:
            command.extend(
                [f"--{dataset.replace('_', '-')}-path", str(Path(override).resolve())]
            )
    return command


def _policy_specs(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.policy_specs:
        raw = json.loads(Path(args.policy_specs).read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = raw.get("policies", [])
        if not isinstance(raw, list) or not raw:
            raise SystemExit("policy specs must be a non-empty JSON list")
    else:
        raw = [
            {
                "name": "active",
                "stop_policy": "tpot",
                "verify_model_mode": "active",
            },
            {
                "name": "static",
                "stop_policy": "tpot",
                "verify_model_mode": "off",
            },
            {
                "name": "none",
                "stop_policy": "none",
                "verify_model_mode": "off",
            },
        ]
    default_artifact = str(Path(args.artifact).resolve()) if args.artifact else ""
    default_calibration = (
        str(Path(args.alpha_calibration_path).resolve())
        if args.alpha_calibration_path
        else ""
    )
    specs = []
    names = set()
    for item in raw:
        if not isinstance(item, dict):
            raise SystemExit("each policy spec must be an object")
        name = str(item.get("name", "")).strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in names:
            raise SystemExit(f"invalid or duplicate policy name: {name!r}")
        names.add(name)
        spec = {
            "name": name,
            "max_draft_tokens": int(
                item.get("max_draft_tokens", args.max_draft_tokens)
            ),
            "stop_policy": str(item.get("stop_policy", "tpot")),
            "verify_model_mode": str(item.get("verify_model_mode", "off")),
            "stop_rule": str(item.get("stop_rule", args.stop_rule)),
            "stop_patience": int(
                item.get("stop_patience", args.stop_patience)
            ),
            "min_steps": int(item.get("min_steps", args.min_steps)),
            "stop_margin": float(item.get("stop_margin", args.stop_margin)),
            "lookahead_cache_credit_ms_per_step": float(
                item.get(
                    "lookahead_cache_credit_ms_per_step",
                    args.lookahead_cache_credit_ms_per_step,
                )
            ),
            "cost_model": str(item.get("cost_model", args.draft_cost_model)),
            "td_ms": float(item.get("td_ms", args.draft_td_ms)),
            "tv_ms": float(item.get("tv_ms", args.draft_tv_ms)),
            "verify_model_path": str(
                Path(item.get("verify_model_path", default_artifact)).resolve()
            )
            if item.get("verify_model_path", default_artifact)
            else "",
            "alpha_calibration_path": str(
                Path(
                    item.get("alpha_calibration_path", default_calibration)
                ).resolve()
            )
            if item.get("alpha_calibration_path", default_calibration)
            else "",
        }
        requested_predictor = item.get(
            "acceptance_predictor_enabled",
            getattr(args, "acceptance_predictor_enabled", None),
        )
        if requested_predictor is not None and not isinstance(
            requested_predictor, bool
        ):
            raise SystemExit(
                f"policy {name} acceptance_predictor_enabled must be boolean"
            )
        predictor_required = bool(
            spec["stop_policy"] != "none"
            or spec["verify_model_mode"] != "off"
            or spec["alpha_calibration_path"]
        )
        spec["acceptance_predictor_enabled"] = (
            predictor_required
            if requested_predictor is None
            else bool(requested_predictor)
        )
        if spec["max_draft_tokens"] < 1 or spec["stop_patience"] < 1:
            raise SystemExit(f"invalid numeric policy fields for {name}")
        if (
            spec["min_steps"] < 0
            or spec["stop_margin"] < 0.0
            or spec["lookahead_cache_credit_ms_per_step"] < 0.0
        ):
            raise SystemExit(f"invalid stop policy fields for {name}")
        if spec["stop_policy"] not in {"none", "alpha_threshold", "tpot"}:
            raise SystemExit(f"invalid stop_policy for {name}")
        if spec["verify_model_mode"] not in {"off", "shadow", "active"}:
            raise SystemExit(f"invalid verify_model_mode for {name}")
        if spec["stop_rule"] not in {
            "first_increase",
            "best_margin",
            "lookahead",
            "lookahead_hysteresis",
            "bucket_lookahead",
            "transfer_aware_step",
        }:
            raise SystemExit(f"invalid stop_rule for {name}")
        if spec["cost_model"] not in {"static", "history"}:
            raise SystemExit(f"invalid cost_model for {name}")
        if spec["verify_model_mode"] != "off" and not spec["verify_model_path"]:
            raise SystemExit(f"policy {name} requires verify_model_path")
        if predictor_required and not spec["acceptance_predictor_enabled"]:
            raise SystemExit(
                f"policy {name} requires acceptance_predictor_enabled=true"
            )
        if spec["alpha_calibration_path"] and not Path(
            spec["alpha_calibration_path"]
        ).is_file():
            raise SystemExit(f"policy {name} alpha calibration does not exist")
        specs.append(spec)
    return specs


def run(args: argparse.Namespace) -> None:
    policies = _policy_specs(args)
    normalized_acceptance = str(args.acceptance_strategy).strip().lower()
    is_sampling = normalized_acceptance in {
        "standard_sampling",
        "sampling",
        "spec_sampling",
    }
    deployment_field = (
        "sampling_deployment_validation"
        if is_sampling
        else "deployment_validation"
    )
    artifact_metadata = {}
    for policy in policies:
        if str(policy["verify_model_mode"]) == "off":
            continue
        artifact_path = Path(str(policy["verify_model_path"]))
        if not artifact_path.is_file():
            raise SystemExit(f"verify model artifact not found: {artifact_path}")
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        from nanovllm.engine.speculative.verify_cost_model import (
            VerifyTimeCostModel,
        )

        model = VerifyTimeCostModel(artifact)
        model.validate_protocol(
            acceptance_strategy=normalized_acceptance,
            temperature=float(args.temperature),
        )
        deployment = artifact.get(deployment_field, {})
        artifact_metadata[str(artifact_path)] = {
            "sha256": _sha256(artifact_path),
            "model_id": artifact.get("model_id"),
            "deployment_field": deployment_field,
            "shadow_gate_version": (
                deployment.get("gate_version")
                if isinstance(deployment, dict)
                else None
            ),
            "proxy_gate_version": artifact.get("proxy_workload_gate_version"),
        }
        if str(policy["verify_model_mode"]) != "active":
            continue
        model_id = str(artifact.get("model_id", "") or "")
        if not bool(artifact.get("accuracy_gate_passed")) or not bool(
            isinstance(deployment, dict) and deployment.get("passed")
        ):
            raise SystemExit("active artifact lacks passing training/shadow gates")
        if str(deployment.get("gate_version", "")) != "v2":
            raise SystemExit("active artifact requires a v2 shadow deployment gate")
        if str(deployment.get("model_id", "") or "") != model_id:
            raise SystemExit(
                "active artifact shadow validation has a different model id"
            )
        if not bool(artifact.get("proxy_workload_gate_passed")) or not isinstance(
            artifact.get("proxy_workload_model"), dict
        ):
            raise SystemExit(
                "active artifact lacks a passing causal workload proxy gate"
            )
        if str(artifact.get("proxy_workload_gate_version", "")) != "v2":
            raise SystemExit("active artifact requires a v2 causal workload proxy gate")

    output_dir = Path(args.output_dir).resolve()
    ratios = [item.strip() for item in args.cache_ratios.split(",") if item.strip()]
    execution_specs = [
        (policy, ratio, repeat)
        for repeat in range(int(args.repeats))
        for ratio in ratios
        for policy in policies
    ]
    random.Random(int(args.seed)).shuffle(execution_specs)
    commands = [
        _command(
            args,
            policy=policy,
            ratio=ratio,
            repeat=repeat,
            command_index=index,
            output_dir=output_dir,
        )
        for index, (policy, ratio, repeat) in enumerate(execution_specs)
    ]
    diff = _output(["git", "diff", "--binary"])
    manifest: dict[str, object] = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": sys.argv,
        "git_commit": _output(["git", "rev-parse", "HEAD"]),
        "git_status": _output(["git", "status", "--short"]),
        "git_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        "verify_model_artifacts": artifact_metadata,
        "policy_specs": policies,
        "commands": [shlex.join(command) for command in commands],
        "experiment_contract": {
            "paired_policies": [str(policy["name"]) for policy in policies],
            "verify_workload_instrumentation": False,
            "stream_event_timing": False,
            "engine_profile": False,
            "temperature": float(args.temperature),
            "acceptance_strategy": str(args.acceptance_strategy),
            "fixed_output_tokens": int(args.output_tokens),
            "independent_process_per_variant_ratio_repeat": True,
            "sample_offset": int(args.sample_offset),
            "save_text": bool(args.save_text),
            "save_token_ids": bool(args.save_token_ids),
            "execution_order": [
                {
                    "policy": str(policy["name"]),
                    "cache_ratio": ratio,
                    "repeat": repeat,
                }
                for policy, ratio, repeat in execution_specs
            ],
        },
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "policy_collection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    env = os.environ.copy()
    for key in (
        "NANOVLLM_VERIFY_COST_MODEL_PROFILE",
        "NANOVLLM_VERIFY_STREAM_EVENT_TIMING",
        "NANOVLLM_VERIFY_OP_EVENT_TIMING",
        "NANOVLLM_VERIFY_BREAKDOWN_SYNC",
        "NANOVLLM_VERIFY_DEEP_PROFILE_SYNC",
        "NANOVLLM_VERIFY_SYNC_METADATA_PROFILE_READBACK",
    ):
        env.pop(key, None)

    completed = []
    for spec, command in zip(execution_specs, commands, strict=True):
        policy, ratio, repeat = spec
        policy_name = str(policy["name"])
        print(
            f"[policy] {policy_name} ratio={ratio} repeat={repeat}: "
            f"{shlex.join(command)}",
            flush=True,
        )
        started = time.time()
        subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
        completed.append(
            {
                "policy": policy_name,
                "cache_ratio": ratio,
                "repeat": repeat,
                "elapsed_sec": time.time() - started,
            }
        )
        manifest["completed"] = completed
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

    analyzer = None
    if args.policy_specs and args.baseline_policy:
        analyzer = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "analyze_tpot_policy_results.py"),
            "--policies-root",
            str(output_dir),
            "--baseline-policy",
            str(args.baseline_policy),
        ]
        if args.candidate_policies:
            analyzer.extend(["--candidate-policies", str(args.candidate_policies)])
    elif not args.policy_specs:
        analyzer = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "analyze_tpot_policy_results.py"),
            "--active",
            str(output_dir / "active"),
            "--static",
            str(output_dir / "static"),
            "--none",
            str(output_dir / "none"),
        ]
    completed_analyzer = None
    if analyzer is not None:
        report_path = output_dir / "policy_validation.json"
        analyzer.extend(
            [
                "--output",
                str(report_path),
                "--minimum-improvement",
                str(args.minimum_improvement),
                "--minimum-pairs",
                str(args.minimum_pairs),
                "--minimum-clusters",
                str(args.minimum_clusters),
            ]
        )
        if args.require_policy_pass or not args.policy_specs:
            analyzer.append("--require-pass")
        completed_analyzer = subprocess.run(analyzer, cwd=REPO_ROOT, check=False)
        manifest["policy_validation"] = json.loads(
            report_path.read_text(encoding="utf-8")
        )
    manifest["completed_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ",
        time.gmtime(),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if completed_analyzer is not None and completed_analyzer.returncode != 0:
        raise SystemExit(completed_analyzer.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", default="")
    parser.add_argument("--policy-specs", default="")
    parser.add_argument("--alpha-calibration-path", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--profile-artifact", default=DEFAULT_PROFILE)
    parser.add_argument("--acceptance-predictor-path", default=DEFAULT_PREDICTOR)
    parser.add_argument(
        "--acceptance-predictor-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Default predictor mode for policy specs. Auto enables it only for "
            "policies that consume acceptance alpha or verify-cost routes."
        ),
    )
    parser.add_argument(
        "--datasets",
        default="per_layer_slots,mt_bench,humaneval,sharegpt",
    )
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--cache-ratios", default="0.25,0.28125,0.3125")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-draft-tokens", type=int, default=12)
    parser.add_argument("--segment-size", type=int, default=12)
    parser.add_argument("--verify-buckets", default="3,5,7,10,13")
    parser.add_argument("--verify-prefetch-budget", type=int, default=10)
    parser.add_argument("--kt-num-threads", type=int, default=16)
    parser.add_argument(
        "--kt-backend",
        choices=["amx_bf16", "avx2_bf16"],
        default="avx2_bf16",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.99)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--acceptance-strategy",
        choices=["greedy", "standard_sampling"],
        default="greedy",
    )
    parser.add_argument("--draft-cost-model", choices=["static", "history"], default="static")
    parser.add_argument("--draft-td-ms", type=float, default=19.0)
    parser.add_argument("--draft-tv-ms", type=float, default=80.0)
    parser.add_argument("--min-steps", type=int, default=0)
    parser.add_argument("--stop-margin", type=float, default=0.0)
    parser.add_argument("--stop-patience", type=int, default=1)
    parser.add_argument(
        "--lookahead-cache-credit-ms-per-step",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--stop-rule",
        choices=[
            "first_increase",
            "best_margin",
            "lookahead",
            "lookahead_hysteresis",
            "bucket_lookahead",
            "transfer_aware_step",
        ],
        default="lookahead",
    )
    parser.add_argument("--minimum-improvement", type=float, default=0.03)
    parser.add_argument("--minimum-pairs", type=int, default=20)
    parser.add_argument("--minimum-clusters", type=int, default=6)
    parser.add_argument("--baseline-policy", default="")
    parser.add_argument("--candidate-policies", default="")
    parser.add_argument("--require-policy-pass", action="store_true")
    parser.add_argument(
        "--save-text", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--save-token-ids", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--dist-port-base", type=int, default=35200)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--sharegpt-path", default="")
    parser.add_argument("--mt-bench-path", default="")
    parser.add_argument("--humaneval-path", default="")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
