"""Configuration and CLI schema for the evaluation TPOT benchmark."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from nanovllm.benchmarks.eval_tpot.runtime import parse_csv as _parse_csv


REPO_ROOT = Path(__file__).resolve().parents[3]


MODEL_PATH = "/data1/models/Qwen3-30B-A3B"

DEFAULT_PROFILE = "results/reroute_impl_20260531/offline_profile_20260531_203257.safetensors"

DEFAULT_PREDICTOR_PATH = "random_cache_srdp_scripts-1/res/run_20260614_133025"

DEFAULT_WARMUP_PROMPT = "Warmup request for verify layer profile."

TRANSFER_AWARE_V3_ARTIFACT = (
    "results/transfer_v3_artifact_20260719/verify_cost_v3.json"
)

DATASET_CHOICES = (
    "sharegpt",
    "mt_bench",
    "humaneval",
    "mmlu_pro",
    "all",
    "per_layer_slots",
)

REQUEST_MODE_CHOICES = ("dataset", "per_layer_slots")

OPTIMIZED_CONFIG_CHOICES = (
    "none",
    "k4_verify",
    "k1_f16_3080",
    "k3_3080",
    "k6_decode",
    "k12_decode",
    "k12_bucket_stop",
    "k12_transfer_step",
)

OPTIMIZED_CONFIG_PRESETS: dict[str, dict[str, Any]] = {
    "k4_verify": {
        "allocation_modes": "profile_weighted",
        "max_draft_tokens_values": "4",
        "verify_prefetch_max_per_boundary": 4,
        "draft_stop_policy": "tpot",
        "kt_num_threads": 16,
        "verify_cuda_graph_bucket_steps": "3,5,7,10,13",
        "verify_prefetch_rank_multiplier": 1,
    },
    # Measured on RTX 3080 10 GiB + 2 x Xeon Gold 5218R.  F16 expert
    # weights make the legacy llamafile CPU kernel substantially faster than
    # BF16 on Cascade Lake, while K=1 keeps verify at the qlen=2 grouped fast
    # path.  A budget of two prefetches per boundary was the best point in the
    # measured 0/1/2/4 sweep.  The segment-indexed direct-active runtime does
    # not use staging buffers, so their two slots are reclaimed as active cache.
    "k1_f16_3080": {
        "allocation_modes": "profile_weighted",
        "cache_ratios": "0.09375",
        "max_draft_tokens_values": "1",
        "segment_sizes": "16",
        "verify_prefetch_max_per_boundary": 2,
        "prefetch_staging_slots_per_layer": 0,
        "draft_stop_policy": "none",
        "acceptance_predictor_enabled": False,
        "cpu_expert_pin_memory": False,
        "kt_num_threads": 16,
        "kt_threadpool_count": 2,
        "kt_numa_nodes": "0,1",
        "kt_direct_backend": "llamafile_f16",
        "kt_capture_bs": "1,2,4,8,16,32",
        "verify_cuda_graph_bucket_steps": "2",
        "verify_prefetch_rank_multiplier": 1,
        "gpu_memory_utilization": 0.996,
        "decode_driver": "generate",
        "reset_seed_after_warmup": True,
    },
    "k3_3080": {
        "allocation_modes": "profile_weighted",
        "cache_ratios": "0.075",
        "max_draft_tokens_values": "3",
        "segment_sizes": "12",
        "verify_prefetch_max_per_boundary": 4,
        "draft_stop_policy": "none",
        "acceptance_predictor_enabled": False,
        "cpu_expert_pin_memory": False,
        "kt_num_threads": 16,
        "kt_threadpool_count": 2,
        "kt_numa_nodes": "0,1",
        "kt_direct_backend": "llamafile_bf16",
        "verify_cuda_graph_bucket_steps": "4",
        "verify_prefetch_rank_multiplier": 1,
        "gpu_memory_utilization": 0.996,
        "decode_driver": "generate",
        "reset_seed_after_warmup": True,
    },
    "k6_decode": {
        "allocation_modes": "profile_weighted",
        "cache_ratios": "0.3125",
        "max_draft_tokens_values": "6",
        "segment_sizes": "12",
        "verify_prefetch_max_per_boundary": 4,
        "draft_stop_policy": "none",
        "acceptance_predictor_enabled": False,
        "kt_num_threads": 16,
        "kt_direct_backend": "avx2_bf16",
        "verify_cuda_graph_bucket_steps": "3,5,7,10,13",
        "verify_prefetch_rank_multiplier": 1,
        "decode_driver": "generate",
        "reset_seed_after_warmup": True,
    },
    "k12_decode": {
        "allocation_modes": "profile_weighted",
        "max_draft_tokens_values": "12",
        "verify_prefetch_max_per_boundary": 10,
        "draft_stop_policy": "none",
        "kt_num_threads": 16,
        "verify_cuda_graph_bucket_steps": "3,5,7,10,13",
        "verify_prefetch_rank_multiplier": 1,
    },
    "k12_bucket_stop": {
        "allocation_modes": "profile_weighted",
        "cache_ratios": "0.3125",
        "max_draft_tokens_values": "12",
        "segment_sizes": "12",
        # The active verify model was deployment-shadowed with this budget.
        "verify_prefetch_max_per_boundary": 10,
        "draft_stop_policy": "tpot",
        "draft_tpot_stop_rule": "bucket_lookahead",
        "draft_tpot_min_steps": 6,
        "draft_tpot_stop_margin": 0.10,
        "draft_tpot_lookahead_cache_credit_ms_per_step": 8.5,
        "draft_tpot_verify_model_mode": "active",
        "acceptance_predictor_enabled": True,
        "kt_num_threads": 16,
        "kt_direct_backend": "avx2_bf16",
        "verify_cuda_graph_bucket_steps": "3,5,7,10,13",
        "verify_prefetch_rank_multiplier": 1,
        "decode_driver": "generate",
        "reset_seed_after_warmup": True,
    },
    "k12_transfer_step": {
        "allocation_modes": "profile_weighted",
        "cache_ratios": "0.3125",
        "max_draft_tokens_values": "12",
        "segment_sizes": "12",
        "verify_prefetch_max_per_boundary": 4,
        "draft_stop_policy": "tpot",
        "draft_tpot_cost_model": "history",
        "draft_tpot_stop_rule": "transfer_aware_step",
        "draft_tpot_min_steps": 6,
        "draft_tpot_stop_margin": 0.0,
        "draft_tpot_lookahead_cache_credit_ms_per_step": 0.0,
        "draft_tpot_verify_model_mode": "active",
        "acceptance_predictor_enabled": True,
        "kt_num_threads": 16,
        "kt_direct_backend": "avx2_bf16",
        "verify_cuda_graph_bucket_steps": "5,7,8,9,10,11,12,13",
        "verify_prefetch_rank_multiplier": 1,
        "decode_driver": "generate",
        "reset_seed_after_warmup": True,
    },
}

def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {value}")

def parse_num_samples(value: str | int) -> int:
    if isinstance(value, int):
        return value
    normalized = value.strip().lower()
    if normalized in {"all", "full", "default", "dataset"}:
        return 0
    parsed = int(normalized)
    if parsed < 0:
        raise argparse.ArgumentTypeError("--num-samples must be >= 0 or all")
    return parsed

def _num_samples_label(value: int) -> str:
    return "all" if int(value) == 0 else str(int(value))

def _parse_allocation_modes(values: str) -> list[str]:
    modes: list[str] = []
    for item in values.split(","):
        mode = item.strip().lower()
        if not mode:
            continue
        if mode not in {"uniform", "profile_weighted"}:
            raise argparse.ArgumentTypeError(f"invalid allocation mode: {mode}")
        modes.append(mode)
    if not modes:
        raise argparse.ArgumentTypeError(
            "--allocation-modes must include uniform and/or profile_weighted"
        )
    return modes

def _arg_was_provided(argv: list[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in argv)

def apply_optimized_config(args: argparse.Namespace, argv: list[str]) -> dict[str, Any]:
    preset_name = str(getattr(args, "optimized_config", "none") or "none")
    if preset_name == "none":
        return {"name": "none", "applied": {}, "manual_overrides": {}}
    preset = OPTIMIZED_CONFIG_PRESETS[preset_name]
    option_by_dest = {
        "allocation_modes": "--allocation-modes",
        "cache_ratios": "--cache-ratios",
        "max_draft_tokens_values": "--max-draft-tokens-values",
        "segment_sizes": "--segment-sizes",
        "verify_prefetch_max_per_boundary": "--verify-prefetch-max-per-boundary",
        "prefetch_staging_slots_per_layer": "--prefetch-staging-slots-per-layer",
        "draft_stop_policy": "--draft-stop-policy",
        "draft_tpot_stop_rule": "--draft-tpot-stop-rule",
        "draft_tpot_min_steps": "--draft-tpot-min-steps",
        "draft_tpot_stop_margin": "--draft-tpot-stop-margin",
        "draft_tpot_cost_model": "--draft-tpot-cost-model",
        "draft_tpot_lookahead_cache_credit_ms_per_step": (
            "--draft-tpot-lookahead-cache-credit-ms-per-step"
        ),
        "draft_tpot_verify_model_mode": "--draft-tpot-verify-model-mode",
        "acceptance_predictor_enabled": "--acceptance-predictor-enabled",
        "cpu_expert_pin_memory": "--cpu-expert-pin-memory",
        "kt_num_threads": "--kt-num-threads",
        "kt_threadpool_count": "--kt-threadpool-count",
        "kt_numa_nodes": "--kt-numa-nodes",
        "kt_direct_backend": "--kt-direct-backend",
        "kt_single_weight": "--kt-single-weight",
        "kt_capture_bs": "--kt-capture-bs",
        "verify_cuda_graph_bucket_steps": "--verify-cuda-graph-bucket-steps",
        "verify_prefetch_rank_multiplier": "--verify-prefetch-rank-multiplier",
        "gpu_memory_utilization": "--gpu-memory-utilization",
        "decode_driver": "--decode-driver",
        "reset_seed_after_warmup": "--reset-seed-after-warmup",
    }
    applied: dict[str, Any] = {}
    manual_overrides: dict[str, Any] = {}
    for dest, value in preset.items():
        option = option_by_dest[dest]
        if _arg_was_provided(argv, option):
            manual_overrides[dest] = getattr(args, dest)
            continue
        setattr(args, dest, value)
        applied[dest] = value
    return {
        "name": preset_name,
        "applied": applied,
        "manual_overrides": manual_overrides,
    }

def resolve_acceptance_predictor(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve the benchmark's auto predictor mode after policy presets."""
    requested = getattr(args, "acceptance_predictor_enabled", None)
    stop_policy = str(getattr(args, "draft_stop_policy", "none"))
    verify_model_mode = str(
        getattr(args, "draft_tpot_verify_model_mode", "off")
    )
    alpha_calibration_path = str(
        getattr(args, "draft_tpot_alpha_calibration_path", "") or ""
    )
    required_by = []
    if stop_policy != "none":
        required_by.append(f"draft_stop_policy={stop_policy}")
    if verify_model_mode != "off":
        required_by.append(f"verify_model_mode={verify_model_mode}")
    if alpha_calibration_path:
        required_by.append("alpha_calibration")
    if bool(getattr(args, "transfer_aware_profile", False)):
        required_by.append("transfer_aware_profile")

    effective = bool(required_by) if requested is None else bool(requested)
    if required_by and not effective:
        raise ValueError(
            "acceptance predictor cannot be disabled when required by "
            + ", ".join(required_by)
        )
    args.acceptance_predictor_enabled = effective
    return {
        "requested": "auto" if requested is None else bool(requested),
        "effective": effective,
        "required_by": required_by,
    }

def validate_runtime_config(args: argparse.Namespace) -> None:
    mode = str(args.inference_mode)
    if bool(args.spec_enable_prefetch) and mode != "spec":
        raise ValueError(
            "--spec-enable-prefetch=true requires --inference-mode=spec"
        )
    if mode == "heter":
        incompatible: list[str] = []
        if bool(args.spec_enable_prefetch):
            incompatible.append("speculative prefetch")
        if bool(args.acceptance_predictor_enabled):
            incompatible.append("acceptance predictor")
        if str(args.draft_stop_policy) != "none":
            incompatible.append("draft early stop")
        if incompatible:
            raise ValueError(
                "--inference-mode=heter cannot enable "
                + ", ".join(incompatible)
            )

    if bool(args.enforce_eager) and (
        bool(args.draft_cuda_graph_enabled) or bool(args.verify_cuda_graph)
    ):
        raise ValueError(
            "--enforce-eager=true requires both "
            "--draft-cuda-graph-enabled=false and --verify-cuda-graph=false"
        )

    if mode == "spec" and bool(args.verify_cuda_graph):
        max_draft_tokens = max(_parse_csv(args.max_draft_tokens_values, int))
        verify_buckets = _parse_csv(args.verify_cuda_graph_bucket_steps, int)
        required_verify_tokens = int(args.batch_size) * (max_draft_tokens + 1)
        if max(verify_buckets) < required_verify_tokens:
            raise ValueError(
                "verify CUDA graph buckets do not cover the configured draft "
                f"length: batch_size={args.batch_size}, "
                f"max_draft_tokens={max_draft_tokens} requires a bucket "
                f">= {required_verify_tokens} "
                "(batch_size * (draft tokens + verify-next token)), "
                f"but --verify-cuda-graph-bucket-steps={args.verify_cuda_graph_bucket_steps}. "
                "Without this bucket, verify silently falls back to the eager path."
            )

    if str(args.draft_tpot_stop_rule) == "transfer_aware_step":
        errors: list[str] = []
        if mode != "spec":
            errors.append("inference_mode must be spec")
        if not bool(args.spec_enable_prefetch):
            errors.append("spec_enable_prefetch must be true")
        if str(args.prefetch_runtime_kind) != "predictive":
            errors.append("prefetch_runtime_kind must be predictive")
        if str(args.prefetch_runtime_mode) != "draft_segment_indexed":
            errors.append(
                "prefetch_runtime_mode must be draft_segment_indexed"
            )
        segment_sizes = _parse_csv(args.segment_sizes, int)
        if segment_sizes != [12]:
            errors.append("segment_sizes must resolve to exactly 12")
        if not bool(args.verify_cuda_graph):
            errors.append("verify_cuda_graph must be true")
        requested_artifact = Path(
            str(args.draft_tpot_verify_model_path)
        ).expanduser()
        expected_artifact = (
            REPO_ROOT / TRANSFER_AWARE_V3_ARTIFACT
        )
        if requested_artifact.resolve() != expected_artifact.resolve():
            errors.append(
                "draft_tpot_verify_model_path must be "
                f"{TRANSFER_AWARE_V3_ARTIFACT}"
            )
        elif not requested_artifact.is_file():
            errors.append(
                f"transfer-aware artifact does not exist: {requested_artifact}"
            )
        if errors:
            raise ValueError(
                "invalid transfer_aware_step configuration: "
                + "; ".join(errors)
            )

def configure_optimized_env(args: argparse.Namespace) -> dict[str, str]:
    optimized_config = str(getattr(args, "optimized_config", "none") or "none")
    rank_multiplier = getattr(args, "verify_prefetch_rank_multiplier", None)
    verify_cost_profile = bool(getattr(args, "verify_cost_model_profile", False))
    transfer_aware_profile = bool(
        getattr(args, "transfer_aware_profile", False)
    )
    latency_breakdown_profile = bool(
        getattr(args, "latency_breakdown_profile", False)
    )
    should_configure = (
        optimized_config != "none"
        or rank_multiplier is not None
        or verify_cost_profile
        or transfer_aware_profile
        or latency_breakdown_profile
    )
    if not should_configure:
        os.environ.pop("NANOVLLM_VERIFY_COST_MODEL_PROFILE", None)
        os.environ.pop("NANOVLLM_TRANSFER_AWARE_PROFILE", None)
        os.environ.pop("NANOVLLM_LATENCY_BREAKDOWN", None)
        return {}

    env_overrides: dict[str, str] = {}
    if optimized_config != "none" or rank_multiplier is not None:
        env_overrides.update(
            {
                "NANOVLLM_VERIFY_PREFETCH_RANK_MULTIPLIER": str(
                    1 if rank_multiplier is None else int(rank_multiplier)
                ),
                "NANOVLLM_VERIFY_DEFER_SEGMENT_METADATA": "1",
                "NANOVLLM_VERIFY_BOUNDARY_PREFETCH_ASYNC": "0",
            }
        )
        if bool(getattr(args, "optimized_segment_event_timing", False)):
            env_overrides["NANOVLLM_VERIFY_SEGMENT_CUDA_EVENT_TIMING"] = "1"
        else:
            os.environ.pop("NANOVLLM_VERIFY_SEGMENT_CUDA_EVENT_TIMING", None)

    if verify_cost_profile:
        env_overrides["NANOVLLM_VERIFY_COST_MODEL_PROFILE"] = "1"
    else:
        os.environ.pop("NANOVLLM_VERIFY_COST_MODEL_PROFILE", None)
    if transfer_aware_profile:
        env_overrides["NANOVLLM_TRANSFER_AWARE_PROFILE"] = "1"
        # The v3 profile includes the legacy aggregate execution truth too.
        env_overrides["NANOVLLM_VERIFY_COST_MODEL_PROFILE"] = "1"
    else:
        os.environ.pop("NANOVLLM_TRANSFER_AWARE_PROFILE", None)

    if latency_breakdown_profile:
        env_overrides.update(
            {
                "NANOVLLM_LATENCY_BREAKDOWN": "1",
                "NANOVLLM_DRAFT_SEGMENT_CUDA_EVENT_TIMING": "1",
                "NANOVLLM_VERIFY_SEGMENT_CUDA_EVENT_TIMING": "1",
                "NANOVLLM_VERIFY_STREAM_EVENT_TIMING": "1",
            }
        )
    else:
        os.environ.pop("NANOVLLM_LATENCY_BREAKDOWN", None)
        os.environ.pop(
            "NANOVLLM_DRAFT_SEGMENT_CUDA_EVENT_TIMING", None
        )
        os.environ.pop("NANOVLLM_VERIFY_STREAM_EVENT_TIMING", None)

    for key, value in env_overrides.items():
        os.environ[key] = value

    if not bool(getattr(args, "preserve_optimized_env", False)):
        for key in (
            "NANOVLLM_VERIFY_DISABLE_RUNTIME_METADATA",
            "NANOVLLM_VERIFY_SKIP_METADATA_OFFLOAD",
            "NANOVLLM_VERIFY_SYNC_METADATA_PROFILE_READBACK",
            "NANOVLLM_VERIFY_OP_EVENT_TIMING",
            "NANOVLLM_VERIFY_DEEP_PROFILE_SYNC",
            "NANOVLLM_VERIFY_BREAKDOWN_SYNC",
        ):
            os.environ.pop(key, None)
    return env_overrides

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-size-1 TPOT benchmark for evaluation_plan.md workloads."
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--profile-artifact", default=DEFAULT_PROFILE)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--optimized-config",
        choices=OPTIMIZED_CONFIG_CHOICES,
        default="none",
        help=(
            "Apply an optimized inference preset. k4_verify uses the verified "
            "K=4 low-latency settings; k3_3080 is the validated dual-NUMA "
            "llamafile path for this RTX 3080 host; k6_decode uses the legacy "
            "fixed-K6 decode settings; k12_decode retains K=12. "
            "Explicit CLI options override preset values."
        ),
    )
    parser.add_argument(
        "--preserve-optimized-env",
        type=str2bool,
        default=False,
        help=(
            "When using an optimized config, keep externally set debug/profile "
            "environment variables instead of clearing known conflicting ones."
        ),
    )
    parser.add_argument(
        "--optimized-segment-event-timing",
        type=str2bool,
        default=False,
        help=(
            "Enable NANOVLLM_VERIFY_SEGMENT_CUDA_EVENT_TIMING during optimized "
            "runs. Off by default because workload TPOT should measure the fast path."
        ),
    )
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="sharegpt")
    parser.add_argument(
        "--dataset-list",
        default="",
        help="Comma-separated explicit dataset sequence; overrides --dataset.",
    )
    parser.add_argument(
        "--request-mode",
        choices=REQUEST_MODE_CHOICES,
        default="dataset",
        help=(
            "Use dataset prompts, or per_layer_slots for the exact single "
            "prompt used by scripts/bench_per_layer_slots.py."
        ),
    )
    parser.add_argument(
        "--num-samples",
        type=parse_num_samples,
        default=0,
        help="Number of requests per dataset. Use 0/all/full/default for the full dataset.",
    )
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--shuffle", type=str2bool, default=False)
    parser.add_argument("--sharegpt-path", default="")
    parser.add_argument("--mt-bench-path", default="")
    parser.add_argument("--humaneval-path", default="")
    parser.add_argument("--mmlu-pro-path", default="")
    parser.add_argument("--max-input-tokens", type=int, default=0)
    parser.add_argument("--truncate-prompts", type=str2bool, default=True)

    parser.add_argument("--allocation-modes", default="uniform,profile_weighted")
    parser.add_argument("--slot-buckets", type=int, default=4)
    parser.add_argument("--slot-max-bucket-ratio", type=float, default=2.0)
    parser.add_argument(
        "--slot-profile-csv",
        default="pre_exps/exp_and_figs/unique/unique_count_plot_summary_n1024.csv",
    )
    parser.add_argument("--slots-per-layer", type=int, default=0)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=0,
        help=(
            "Optional safety cap for generated tokens. 0 means use remaining "
            "model context and stop normally on EOS."
        ),
    )
    parser.add_argument(
        "--output-lens",
        default="",
        help=(
            "Compatibility alias for max-output token values. 0 means stop "
            "normally on EOS; N>0 matches the old fixed-output benchmark path "
            "by using N max output tokens and ignore_eos=True."
        ),
    )
    parser.add_argument("--cache-ratios", default="0.3125")
    parser.add_argument("--max-draft-tokens-values", default="12")
    parser.add_argument("--segment-sizes", default="12")
    parser.add_argument(
        "--inference-mode",
        choices=["heter", "spec"],
        default="spec",
    )
    parser.add_argument(
        "--spec-enable-prefetch",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--enforce-eager",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--draft-cuda-graph-enabled",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--verify-cuda-graph",
        type=str2bool,
        default=True,
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--repeat-index-offset",
        type=int,
        default=0,
        help="Offset persisted repeat ids when repetitions run in separate processes.",
    )
    parser.add_argument(
        "--reuse-engine-across-draft-lengths",
        type=str2bool,
        default=False,
        help=(
            "Reuse one loaded engine for cases that differ only by dataset/K at "
            "the same cache ratio, allocation, segment size, and repeat."
        ),
    )
    parser.add_argument(
        "--reuse-engine-case-order",
        choices=["declared", "shuffle"],
        default="declared",
    )

    parser.add_argument("--cache-strategy", default="lru")
    parser.add_argument("--draft-reroute-policy", default="entropy_cache_bias")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Sampling vocabulary cap; 0 disables top-k filtering.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling probability in (0, 1].",
    )
    parser.add_argument("--acceptance-strategy", default="standard_sampling")
    parser.add_argument("--acceptance-threshold", type=float, default=0.7)
    parser.add_argument(
        "--acceptance-predictor-enabled",
        type=str2bool,
        default=None,
        help=(
            "Enable the draft acceptance predictor. By default it is enabled "
            "only when the stop policy, alpha calibration, or verify-cost proxy "
            "needs it; fixed-K stop_policy=none runs keep it off."
        ),
    )
    parser.add_argument("--acceptance-predictor-path", default=DEFAULT_PREDICTOR_PATH)
    parser.add_argument("--acceptance-predictor-step-horizon", type=int, default=32)
    parser.add_argument("--draft-alpha-stop-threshold", type=float, default=0.89)
    parser.add_argument(
        "--draft-stop-policy",
        choices=["none", "alpha_threshold", "tpot"],
        default="tpot",
    )
    parser.add_argument("--draft-tpot-td-ms", type=float, default=19.0)
    parser.add_argument("--draft-tpot-tv-ms", type=float, default=80.0)
    parser.add_argument(
        "--draft-tpot-cost-model",
        choices=["static", "history"],
        default="static",
    )
    parser.add_argument("--draft-tpot-history-alpha", type=float, default=0.2)
    parser.add_argument("--draft-tpot-min-steps", type=int, default=0)
    parser.add_argument("--draft-tpot-stop-margin", type=float, default=0.0)
    parser.add_argument("--draft-tpot-stop-patience", type=int, default=1)
    parser.add_argument(
        "--draft-tpot-lookahead-cache-credit-ms-per-step",
        type=float,
        default=0.0,
    )
    parser.add_argument("--draft-tpot-short-verify-penalty-ms", type=float, default=0.0)
    parser.add_argument("--draft-tpot-verify-cost-floor-ms", type=float, default=0.0)
    parser.add_argument(
        "--draft-tpot-stop-rule",
        choices=[
            "first_increase",
            "best_margin",
            "lookahead",
            "lookahead_hysteresis",
            "bucket_lookahead",
            "transfer_aware_step",
        ],
        default="first_increase",
    )
    parser.add_argument(
        "--draft-tpot-verify-model-mode",
        choices=["off", "shadow", "active"],
        default="off",
    )
    parser.add_argument("--draft-tpot-verify-model-path", default="")
    parser.add_argument("--draft-tpot-alpha-calibration-path", default="")
    parser.add_argument("--draft-tpot-alpha-error-p90", type=float, default=0.05)
    parser.add_argument(
        "--draft-tpot-draft-error-p90-ms", type=float, default=1.0
    )
    parser.add_argument(
        "--draft-tpot-uncertainty-scale", type=float, default=1.0
    )

    parser.add_argument("--rank-guard-threshold", type=float, default=0.15)
    parser.add_argument("--rank-guard-ema-alpha", type=float, default=0.95)
    parser.add_argument("--prefetch-step-budget", type=int, default=16)
    parser.add_argument(
        "--prefetch-runtime-kind",
        choices=["legacy", "predictive", "dual_queue"],
        default="predictive",
    )
    parser.add_argument(
        "--prefetch-runtime-mode",
        choices=[
            "baseline_staging",
            "draft_direct_active",
            "draft_segment_indexed",
        ],
        default="draft_segment_indexed",
    )
    parser.add_argument("--prefetch-max-inflight", type=int, default=16)
    parser.add_argument("--prefetch-transfer-stream-count", type=int, default=1)
    parser.add_argument("--prefetch-staging-slots-per-layer", type=int, default=2)
    parser.add_argument("--prefetch-metadata-host-buffer-pool-size", type=int, default=3)
    parser.add_argument("--prefetch-global-queue-capacity", type=int, default=4096)
    parser.add_argument("--prefetch-verify-layer-max-budget", type=int, default=8)
    parser.add_argument("--prefetch-verify-attention-ratio", type=float, default=1.0)
    parser.add_argument("--predictive-phase1-budget", type=int, default=4)
    parser.add_argument("--dual-queue-ground-truth-decay", type=float, default=0.9)
    parser.add_argument("--dual-queue-ground-truth-ttl-rounds", type=int, default=64)
    parser.add_argument("--dual-queue-ground-truth-count-weight", type=float, default=0.1)
    parser.add_argument("--dual-queue-budget-safety-ratio", type=float, default=0.8)
    parser.add_argument("--dual-queue-segment-time-ema-alpha", type=float, default=0.2)
    parser.add_argument("--dual-queue-secondary-index-weight", type=float, default=0.5)
    parser.add_argument("--prefetch-history-decay", type=float, default=0.9)
    parser.add_argument("--prefetch-history-ttl-steps", type=int, default=64)
    parser.add_argument("--prefetch-source-weight-prefill", type=float, default=1.0)
    parser.add_argument("--prefetch-source-weight-verify", type=float, default=1.2)
    parser.add_argument("--prefetch-source-weight-draft", type=float, default=1.5)
    parser.add_argument("--prefetch-activation-count-weight", type=float, default=0.1)
    parser.add_argument("--prefetch-age-penalty", type=float, default=0.02)
    parser.add_argument("--prefetch-use-prefill-history", type=str2bool, default=True)
    parser.add_argument("--prefetch-use-verify-history", type=str2bool, default=True)
    parser.add_argument("--prefetch-use-draft-live", type=str2bool, default=True)
    parser.add_argument("--cache-eviction-budget-per-step", type=int, default=2)
    parser.add_argument("--draft-segment-host-buffer-pool-size", type=int, default=0)
    parser.add_argument("--draft-prefetch-visible-budget-ms", type=float, default=3.0)
    parser.add_argument("--draft-prefetch-max-per-boundary", type=int, default=16)
    parser.add_argument("--verify-prefetch-visible-budget-ms", type=float, default=12.0)
    parser.add_argument("--verify-prefetch-max-per-boundary", type=int, default=4)
    parser.add_argument("--verify-prefetch-tpot-dynamic-budget-enabled", type=str2bool, default=False)
    parser.add_argument("--verify-prefetch-tpot-dynamic-budget-token-threshold", type=int, default=10)
    parser.add_argument("--verify-prefetch-tpot-dynamic-budget-small", type=int, default=4)
    parser.add_argument(
        "--verify-prefetch-rank-multiplier",
        type=int,
        default=None,
        help=(
            "Set NANOVLLM_VERIFY_PREFETCH_RANK_MULTIPLIER. Optimized presets "
            "default to 1 unless this option is explicitly provided."
        ),
    )

    parser.add_argument("--cpu-expert-workspace-max-routes", type=int, default=327680)
    parser.add_argument("--cpu-expert-num-threads", type=int, default=4)
    parser.add_argument(
        "--cpu-expert-pin-memory",
        type=str2bool,
        default=True,
        help=(
            "Pin the full CPU expert pool in host memory. Disable this on "
            "memory-constrained hosts to reduce unreclaimable memory pressure."
        ),
    )
    parser.add_argument("--kt-num-threads", type=int, default=0)
    parser.add_argument("--kt-threadpool-count", type=int, default=1)
    parser.add_argument("--kt-chunked-prefill-size", type=int, default=4096)
    parser.add_argument(
        "--kt-direct-backend",
        choices=[
            "auto",
            "amx_bf16",
            "avx2_bf16",
            "llamafile_bf16",
            "llamafile_f16",
        ],
        default="auto",
    )
    parser.add_argument(
        "--kt-llamafile-extension-path",
        default="",
        help=(
            "Optional cpuinfer_ext shared library built from the fixed "
            "KTransformers llamafile backend. Required only when cpuinfer_ext "
            "is not importable in the active Python environment."
        ),
    )
    parser.add_argument(
        "--kt-single-weight",
        type=str2bool,
        default=True,
        help=(
            "Reuse the legacy CPUInfer NUMA-local buffers as the CPU expert "
            "pool, avoiding a second full raw-weight copy. Requires the "
            "single-weight cpuinfer_ext patch."
        ),
    )
    parser.add_argument("--kt-numa-nodes", default="")
    parser.add_argument("--kt-capture-bs", default="1,2,4,8,16,32")

    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Number of requests scheduled in one synchronous model batch. "
            "For batch_size>1 the selected prompt is replicated and TPOT is "
            "reported per request plus aggregate output-token throughput."
        ),
    )
    parser.add_argument(
        "--batch-sizes",
        default="",
        help=(
            "Comma-separated true-batch sweep used by "
            "scripts/bench_eval_true_batch_tpot.py. The regular workload "
            "runner uses --batch-size."
        ),
    )
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.99)
    parser.add_argument("--verify-cuda-graph-bucket-steps", default="3,5,8,13")
    parser.add_argument("--dist-port-base", type=int, default=31800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--warmup-prompt",
        default=DEFAULT_WARMUP_PROMPT,
        help="Warmup prompt text. Defaults to the per-layer benchmark warmup prompt.",
    )
    parser.add_argument(
        "--decode-driver",
        choices=["step", "generate"],
        default="step",
        help=(
            "step times the explicit LLM.step loop; generate uses LLM.generate "
            "with a step hook so the driver matches bench_per_layer_slots.py."
        ),
    )
    parser.add_argument(
        "--reset-profile-after-warmup",
        type=str2bool,
        default=True,
        help="Call llm.get_profile(reset=True) after warmup, outside timed requests.",
    )
    parser.add_argument(
        "--reset-profile-before-request",
        type=str2bool,
        default=False,
        help=(
            "When collecting profile, also reset immediately before each measured "
            "request. Off by default to avoid perturbing prefetch/cache state."
        ),
    )
    parser.add_argument(
        "--reset-seed-after-warmup",
        type=str2bool,
        default=False,
        help=(
            "Reset Python/Torch RNG before each measured request. This keeps "
            "warmup generation from changing the measured sampling trajectory."
        ),
    )
    parser.add_argument(
        "--collect-profile",
        type=str2bool,
        default=False,
        help=(
            "Collect llm.get_profile(reset=True) outside the timed request and "
            "write derived wall-vs-engine timing fields into result rows."
        ),
    )
    parser.add_argument(
        "--engine-profile",
        type=str2bool,
        default=False,
        help="Enable LLMEngine/ModelRunner perf_counter counters.",
    )
    parser.add_argument(
        "--engine-profile-cuda-sync",
        type=str2bool,
        default=False,
        help="Synchronize CUDA around engine profile counters when engine profile is enabled.",
    )
    parser.add_argument(
        "--verify-cost-model-profile",
        type=str2bool,
        default=False,
        help=(
            "Collect per-verify acceptance-ready timing and full CUDA-graph "
            "execution workload. This automatically enables engine/profile JSON "
            "collection and keeps synchronous profilers disabled."
        ),
    )
    parser.add_argument(
        "--latency-breakdown-profile",
        type=str2bool,
        default=False,
        help=(
            "Collect low-perturbation TPOT latency-breakdown counters. "
            "CUDA events are drained after natural synchronization or after "
            "the timed request; legacy per-op forced synchronization stays off."
        ),
    )
    parser.add_argument(
        "--transfer-aware-profile",
        type=str2bool,
        default=False,
        help=(
            "Collect v3 draft-route alignment, logical/execution verify rows, "
            "resident/pending/inflight snapshots, and transfer ticket lifecycle."
        ),
    )
    parser.add_argument("--skip-existing", type=str2bool, default=True)
    parser.add_argument("--fail-fast", type=str2bool, default=True)
    parser.add_argument("--fail-on-output-validation-error", type=str2bool, default=True)
    parser.add_argument("--save-profile-json", type=str2bool, default=False)
    parser.add_argument("--save-token-ids", type=str2bool, default=False)
    parser.add_argument("--save-text", type=str2bool, default=False)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse, resolve presets, and validate one benchmark configuration."""
    args = build_parser().parse_args(argv)
    args._optimized_config_applied = apply_optimized_config(args, argv)
    args._acceptance_predictor_resolution = resolve_acceptance_predictor(args)
    validate_runtime_config(args)
    if (
        bool(args.verify_cost_model_profile)
        or bool(args.transfer_aware_profile)
        or bool(args.latency_breakdown_profile)
    ):
        args.collect_profile = True
        args.engine_profile = True
        args.engine_profile_cuda_sync = False
        args.save_profile_json = True
    if args.num_samples < 0:
        raise ValueError("--num-samples must be >= 0 or all")
    if int(args.top_k) < 0:
        raise ValueError("--top-k must be non-negative")
    if not 0.0 < float(args.top_p) <= 1.0:
        raise ValueError("--top-p must be in (0, 1]")
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.repeat_index_offset < 0:
        raise ValueError("--repeat-index-offset must be >= 0")
    return args
