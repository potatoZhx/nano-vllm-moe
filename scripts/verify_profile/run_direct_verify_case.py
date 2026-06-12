#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig

from nanovllm import LLM, SamplingParams


DEFAULT_MODEL_PATH = "/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B"
DEFAULT_PROFILE_ARTIFACT = "results/reroute_impl_20260531/offline_profile_20260531_203257.safetensors"


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _summary_from_profile(profile: dict[str, Any]) -> dict[str, Any]:
    verify_calls = float(profile.get("spec_run_verify_calls", 0.0) or 0.0)
    draft_calls = float(profile.get("spec_run_draft_calls", 0.0) or 0.0)
    active = float(profile.get("model_pre_transfer_active_count", 0.0) or 0.0)
    miss = float(profile.get("model_pre_transfer_cache_miss", 0.0) or 0.0)
    return {
        "verify_calls": verify_calls,
        "verify_forward_ms_avg": float(profile.get("verify_forward_ms", 0.0) or 0.0),
        "verify_forward_ms_total": float(profile.get("spec_run_verify_infer_ms_total", 0.0) or 0.0),
        "run_verify_total_ms_avg": _safe_div(float(profile.get("model_run_verify_total_ms", 0.0) or 0.0), verify_calls),
        "verify_prepare_prefill_ms_avg": _safe_div(
            float(profile.get("model_verify_prepare_prefill_ms", 0.0) or 0.0),
            verify_calls,
        ),
        "verify_tokens_per_call": _safe_div(float(profile.get("model_verify_tokens_in_total", 0.0) or 0.0), verify_calls),
        "draft_forward_ms_avg": float(profile.get("draft_forward_ms", 0.0) or 0.0),
        "draft_calls": draft_calls,
        "spec_step_ms_total": float(profile.get("spec_spec_step_ms", 0.0) or 0.0),
        "prefetch_wait_ms_total": float(profile.get("model_prefetch_wait_ms", 0.0) or 0.0),
        "verify_layer_prefetch_hook_count": int(profile.get("model_verify_layer_prefetch_hook_count", 0) or 0),
        "verify_layer_prefetch_submit_count": int(profile.get("model_verify_layer_prefetch_submit_count", 0) or 0),
        "verify_layer_prefetch_consumed_count": int(profile.get("model_verify_layer_prefetch_consumed_count", 0) or 0),
        "verify_cache_fill_transfer_ms_total": float(profile.get("model_verify_cache_fill_transfer_ms", 0.0) or 0.0),
        "verify_cache_fill_promoted_expert_count": int(profile.get("model_verify_cache_fill_promoted_expert_count", 0) or 0),
        "verify_cache_fill_no_cpu_remaining_miss_count": int(
            profile.get("model_verify_cache_fill_no_cpu_remaining_miss_count", 0) or 0
        ),
        "verify_cache_fill_no_cpu_remaining_miss_route_count": int(
            profile.get("model_verify_cache_fill_no_cpu_remaining_miss_route_count", 0) or 0
        ),
        "verify_cache_fill_no_cpu_fallback_count": int(
            profile.get("model_verify_cache_fill_no_cpu_fallback_count", 0) or 0
        ),
        "verify_cpu_compute_ms_total": float(profile.get("model_verify_cpu_compute_ms", 0.0) or 0.0),
        "verify_gpu_compute_ms_total": float(profile.get("model_verify_gpu_compute_ms", 0.0) or 0.0),
        "verify_plan_ms_total": float(profile.get("model_verify_plan_ms", 0.0) or 0.0),
        "verify_route_ms_total": float(profile.get("model_verify_route_ms", 0.0) or 0.0),
        "verify_gpu_gather_ms_total": float(profile.get("model_verify_gpu_gather_ms", 0.0) or 0.0),
        "verify_scatter_ms_total": float(profile.get("model_verify_scatter_ms", 0.0) or 0.0),
        "metadata_offload_verify_ms_total": float(profile.get("model_metadata_offload_verify_ms", 0.0) or 0.0),
        "run_verify_submit_after_ms_total": float(profile.get("model_run_verify_submit_after_ms", 0.0) or 0.0),
        "route_hit_rate_post_transfer": float(1.0 - float(profile.get("model_cpu_route_ratio", 0.0) or 0.0)),
        "true_route_hit_rate": float(1.0 - _safe_div(miss, active)) if active > 0 else 0.0,
    }


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    hf_config = AutoConfig.from_pretrained(args.model_path)
    num_experts = int(getattr(hf_config, "num_experts"))
    slots = int(args.slots_per_layer)
    if slots <= 0:
        slots = int(round(num_experts * float(args.cache_ratio)))

    prompt_text = Path(args.prompt_text_file).read_text(encoding="utf-8")
    profile_dir = args.torch_profile_dir.strip()
    if args.torch_profile and not profile_dir:
        profile_dir = str(Path(args.output_dir) / f"{args.name}_torch_profile")
    if args.torch_profile:
        Path(profile_dir).mkdir(parents=True, exist_ok=True)
        os.environ["NANOVLLM_VERIFY_TORCH_PROFILE_DIR"] = profile_dir
    else:
        os.environ.pop("NANOVLLM_VERIFY_TORCH_PROFILE_DIR", None)

    llm: LLM | None = None
    try:
        llm = LLM(
            args.model_path,
            dist_port=args.dist_port,
            enforce_eager=args.enforce_eager,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=1,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            inference_mode="spec",
            enable_heterogeneous=True,
            enable_speculative=True,
            heterogeneous_slots_per_layer=slots,
            max_draft_tokens=args.max_draft_tokens,
            draft_top_c=0,
            draft_reroute_policy=args.draft_reroute_policy,
            draft_reroute_artifact=args.draft_reroute_artifact,
            acceptance_strategy=args.acceptance_strategy,
            acceptance_threshold=args.acceptance_threshold,
            cpu_expert_execution_enabled=True,
            cpu_expert_pin_memory=True,
            cpu_expert_backend=args.cpu_expert_backend,
            cpu_expert_workspace_max_routes=args.cpu_expert_workspace_max_routes,
            cpu_expert_packed_min_routes=1,
            cpu_expert_parallel_mode="serial",
            cpu_expert_num_threads=args.cpu_expert_num_threads,
            cpu_gpu_parallel_execution_enabled="auto",
            cpu_gpu_parallel_min_cpu_route_ratio=0.0,
            spec_verify_miss_policy=args.spec_verify_miss_policy,
            spec_profile=True,
            engine_profile=True,
            engine_profile_cuda_sync=True,
            spec_enable_prefetch=args.prefetch_enabled,
            cache_strategy="lru",
            rank_guard_threshold=0.15,
            rank_guard_ema_alpha=0.95,
            prefetch_strategy="history_window",
            prefetch_runtime_mode="draft_segment_indexed",
            prefetch_runtime_kind=args.prefetch_runtime_kind,
            dual_queue_segment_size=args.dual_queue_segment_size,
            prefetch_verify_attention_ratio=args.prefetch_verify_attention_ratio,
            predictive_phase1_budget=args.predictive_phase1_budget,
            prefetch_staging_slots_per_layer=2,
            prefetch_max_inflight=8,
            prefetch_step_budget=4,
            cache_eviction_budget_per_step=2,
            prefetch_verify_wait_ms=0.0,
            prefetch_global_queue_capacity=4096,
            prefetch_history_decay=0.9,
            prefetch_history_ttl_steps=64,
            prefetch_source_weight_prefill=1.0,
            prefetch_source_weight_verify=1.2,
            prefetch_source_weight_draft=1.5,
            prefetch_activation_count_weight=0.1,
            prefetch_age_penalty=0.02,
            prefetch_use_prefill_history=True,
            prefetch_use_verify_history=True,
            prefetch_use_draft_live=True,
            prefetch_verify_layer_enabled=args.prefetch_verify_layer_enabled,
            draft_cuda_graph_enabled=True,
            draft_cuda_graph_cpu_backend="none",
        )

        prompts = [llm.tokenizer.encode(prompt_text)]
        warmup_params = SamplingParams(temperature=args.temperature, ignore_eos=True, max_tokens=4)
        llm.generate(["Warmup request for verify layer profile."], warmup_params, use_tqdm=False)
        llm.get_profile(reset=True)

        sampling = [SamplingParams(temperature=args.temperature, ignore_eos=True, max_tokens=args.output_len)]
        t0 = time.time()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        elapsed = time.time() - t0
        profile = llm.get_profile(reset=True)

        token_ids = [x["token_ids"] for x in outputs]
        text = [x.get("text", "") for x in outputs]
        generated = sum(len(x) for x in token_ids)
        digest_payload = "|".join(",".join(str(t) for t in seq) for seq in token_ids).encode("utf-8")

        result = {
            "case": {
                "name": args.name,
                "cache_ratio": float(args.cache_ratio),
                "slots_per_layer": slots,
                "prefetch_enabled": bool(args.prefetch_enabled),
                "prefetch_verify_layer_enabled": bool(args.prefetch_verify_layer_enabled),
                "spec_verify_miss_policy": args.spec_verify_miss_policy,
                "prefetch_runtime_kind": args.prefetch_runtime_kind,
                "dual_queue_segment_size": int(args.dual_queue_segment_size),
                "output_len": int(args.output_len),
                "max_draft_tokens": int(args.max_draft_tokens),
                "actual_input_tokens": [len(prompts[0])],
                "torch_profile_dir": profile_dir if args.torch_profile else "",
            },
            "elapsed_sec": elapsed,
            "generated_output_tokens": generated,
            "throughput_output_tok_s": generated / elapsed if elapsed > 0 else 0.0,
            "outputs_digest": hashlib.sha256(digest_payload).hexdigest(),
            "generated_text": text,
            "engine_profile": profile,
            "summary": _summary_from_profile(profile),
        }
        return result
    finally:
        if llm is not None:
            llm.exit()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run one direct verify-profile case without benchmark monkeypatch probes.")
    p.add_argument("--name", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--prompt-text-file", required=True)
    p.add_argument("--draft-reroute-artifact", default=DEFAULT_PROFILE_ARTIFACT)
    p.add_argument("--dist-port", type=int, required=True)
    p.add_argument("--cache-ratio", type=float, default=0.75)
    p.add_argument("--slots-per-layer", type=int, default=0)
    p.add_argument("--prefetch-enabled", type=str2bool, default=True)
    p.add_argument("--prefetch-verify-layer-enabled", type=str2bool, default=True)
    p.add_argument(
        "--prefetch-runtime-kind",
        choices=["legacy", "predictive", "dual_queue"],
        default="legacy",
    )
    p.add_argument("--dual-queue-segment-size", type=int, default=12)
    p.add_argument("--prefetch-verify-attention-ratio", type=float, default=0.3)
    p.add_argument("--predictive-phase1-budget", type=int, default=4)
    p.add_argument("--spec-verify-miss-policy", choices=["cpu", "cache_fill", "cache_fill_no_cpu"], default="cache_fill")
    p.add_argument("--output-len", type=int, default=512)
    p.add_argument("--max-draft-tokens", type=int, default=8)
    p.add_argument("--draft-reroute-policy", default="entropy_cache_bias")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--acceptance-strategy", default="standard_sampling")
    p.add_argument("--acceptance-threshold", type=float, default=0.7)
    p.add_argument("--cpu-expert-backend", default="fused")
    p.add_argument("--cpu-expert-workspace-max-routes", type=int, default=16384)
    p.add_argument("--cpu-expert-num-threads", type=int, default=4)
    p.add_argument("--max-num-batched-tokens", type=int, default=16384)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--enforce-eager", type=str2bool, default=False)
    p.add_argument("--torch-profile", type=str2bool, default=False)
    p.add_argument("--torch-profile-dir", default="")
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_case(args)
    output_path = output_dir / f"{args.name}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
