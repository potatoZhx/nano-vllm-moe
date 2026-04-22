#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
import gc
from dataclasses import dataclass
from pathlib import Path

import torch

from nanovllm import LLM, SamplingParams


REQUIRED_METRICS = [
    # Existing canonical metrics from phase2_post
    "route_ms",
    "plan_ms",
    "gpu_gather_ms",
    "gpu_compute_ms",
    "cpu_prepare_ms",
    "cpu_compute_ms",
    "cpu_to_gpu_merge_ms",
    "scatter_ms",
    "draft_ms",
    "verify_ms",
    "spec_step_ms",
    "graph_hit_rate",
    "graph_replay_count",
    "cpu_route_ratio",
    "cpu_weight_mass_ratio",
    "activated_expert_set_size",
    "realized_cpu_expert_count",
    # Phase 3 prefetch metrics
    "prefetch_submit_count",
    "prefetch_completed_count",
    "prefetch_late_count",
    "prefetch_wait_ms",
    "prefetch_consumed_count",
    "prefetch_timeout_count",
    "publish_count",
    "publish_ms",
    "metadata_offload_ms",
    "metadata_offload_bytes",
    "history_prefetch_submit_count",
    "verify_history_prefetch_submit_count",
    "draft_live_prefetch_submit_count",
    "verify_ready_before_wait_count",
    "verify_ready_after_wait_count",
]


@dataclass
class CaseConfig:
    name: str
    mode: str
    kwargs: dict


PROMPTS = [
    "Explain how speculative decoding keeps correctness while improving latency.",
    "Summarize trade-offs of CPU-GPU heterogeneous MoE execution in production.",
    "Describe why replay-boundary publish can be safer than in-graph cache mutation.",
    "Give a concise checklist to validate prefetch hit quality and overhead.",
]


def _digest_outputs(token_id_lists: list[list[int]]) -> str:
    parts = []
    for token_ids in token_id_lists:
        parts.append(hashlib.sha256(",".join(str(x) for x in token_ids).encode("utf-8")).hexdigest())
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _count_prompt_tokens(llm: LLM, prompts: list[str]) -> int:
    return sum(len(llm.tokenizer.encode(prompt)) for prompt in prompts)


def _run_single_case(
    model_path: str,
    dist_port: int,
    num_seqs: int,
    output_len: int,
    temperature: float,
    slots_per_layer: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    case: CaseConfig,
) -> dict:
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(num_seqs)]
    sampling_params = [
        SamplingParams(
            temperature=temperature,
            ignore_eos=True,
            max_tokens=output_len,
        )
        for _ in range(num_seqs)
    ]

    llm_kwargs = {
        "dist_port": dist_port,
        "inference_mode": case.mode,
        "enable_heterogeneous": case.mode in {"heter", "spec"},
        "enable_speculative": case.mode == "spec",
        "engine_profile": True,
        "engine_profile_cuda_sync": True,
        "spec_profile": True,
        "enforce_eager": False,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_seqs": max_num_seqs,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "heterogeneous_slots_per_layer": slots_per_layer,
        "max_draft_tokens": 4,
        "draft_top_c": 0,
    }
    llm_kwargs.update(case.kwargs)

    llm = LLM(model_path, **llm_kwargs)

    # Warmup and clear profile counters.
    llm.generate(["Warmup run."], SamplingParams(temperature=temperature, max_tokens=4), use_tqdm=False)
    llm.get_profile(reset=True)

    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    elapsed = time.time() - t0

    profile = llm.get_profile(reset=True)
    token_ids = [item["token_ids"] for item in outputs]
    input_tokens = _count_prompt_tokens(llm, prompts)
    output_tokens = sum(len(x) for x in token_ids)

    llm.exit()

    row = {
        "case": case.name,
        "mode": case.mode,
        "elapsed_sec": elapsed,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "throughput_output_tok_s": output_tokens / elapsed if elapsed > 0 else 0.0,
        "throughput_total_tok_s": (input_tokens + output_tokens) / elapsed if elapsed > 0 else 0.0,
        "outputs_digest": _digest_outputs(token_ids),
        "profile": profile,
        "metrics": {name: profile.get(name, None) for name in REQUIRED_METRICS},
    }
    row["missing_metrics"] = [name for name in REQUIRED_METRICS if name not in profile]

    # Release references and CUDA cached memory before the next case.
    del outputs
    del token_ids
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return row


def _build_cases(prefetch_wait_ms: float) -> list[CaseConfig]:
    # Baseline mode comparison
    cases = [
        CaseConfig(name="standard_baseline", mode="standard", kwargs={}),
        CaseConfig(name="heter_baseline", mode="heter", kwargs={}),
        CaseConfig(
            name="spec_prefetch_off",
            mode="spec",
            kwargs={
                "spec_enable_prefetch": False,
            },
        ),
        CaseConfig(
            name="spec_prefetch_on_full",
            mode="spec",
            kwargs={
                "spec_enable_prefetch": True,
                "prefetch_verify_wait_ms": prefetch_wait_ms,
                "prefetch_use_prefill_history": True,
                "prefetch_use_verify_history": True,
                "prefetch_use_draft_live": True,
                "cache_strategy": "lru",
                "prefetch_strategy": "history_window",
            },
        ),
        # Ablations for major design elements.
        CaseConfig(
            name="spec_ablate_draft_live",
            mode="spec",
            kwargs={
                "spec_enable_prefetch": True,
                "prefetch_verify_wait_ms": prefetch_wait_ms,
                "prefetch_use_prefill_history": True,
                "prefetch_use_verify_history": True,
                "prefetch_use_draft_live": False,
            },
        ),
        CaseConfig(
            name="spec_ablate_verify_history",
            mode="spec",
            kwargs={
                "spec_enable_prefetch": True,
                "prefetch_verify_wait_ms": prefetch_wait_ms,
                "prefetch_use_prefill_history": True,
                "prefetch_use_verify_history": False,
                "prefetch_use_draft_live": True,
            },
        ),
        CaseConfig(
            name="spec_ablate_prefill_history",
            mode="spec",
            kwargs={
                "spec_enable_prefetch": True,
                "prefetch_verify_wait_ms": prefetch_wait_ms,
                "prefetch_use_prefill_history": False,
                "prefetch_use_verify_history": True,
                "prefetch_use_draft_live": True,
            },
        ),
        CaseConfig(
            name="spec_ablate_wait_zero",
            mode="spec",
            kwargs={
                "spec_enable_prefetch": True,
                "prefetch_verify_wait_ms": 0.0,
                "prefetch_use_prefill_history": True,
                "prefetch_use_verify_history": True,
                "prefetch_use_draft_live": True,
            },
        ),
        CaseConfig(
            name="spec_ablate_cache_lfu",
            mode="spec",
            kwargs={
                "spec_enable_prefetch": True,
                "prefetch_verify_wait_ms": prefetch_wait_ms,
                "prefetch_use_prefill_history": True,
                "prefetch_use_verify_history": True,
                "prefetch_use_draft_live": True,
                "cache_strategy": "lfu",
            },
        ),
    ]
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 real-model end-to-end evaluation suite")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--num-seqs", type=int, default=4)
    parser.add_argument("--output-len", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--base-dist-port", type=int, default=29500)
    parser.add_argument("--slots-per-layer", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.999)
    parser.add_argument("--prefetch-wait-ms", type=float, default=1.0)
    parser.add_argument("--cases", type=str, default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    cases = _build_cases(prefetch_wait_ms=args.prefetch_wait_ms)
    if args.cases.strip():
        selected = {x.strip() for x in args.cases.split(",") if x.strip()}
        cases = [c for c in cases if c.name in selected]
    for idx, case in enumerate(cases):
        row = _run_single_case(
            model_path=args.model_path,
            dist_port=args.base_dist_port + idx,
            num_seqs=args.num_seqs,
            output_len=args.output_len,
            temperature=args.temperature,
            slots_per_layer=args.slots_per_layer,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            case=case,
        )
        rows.append(row)

    digests = {row["case"]: row["outputs_digest"] for row in rows}
    digest_ref = digests.get("standard_baseline")
    digest_compare = {
        row["case"]: (row["outputs_digest"] == digest_ref)
        for row in rows
        if digest_ref is not None
    }

    result = {
        "benchmark": "phase3_real_e2e_eval",
        "required_metrics": REQUIRED_METRICS,
        "cases": rows,
        "digest_match_vs_standard": digest_compare,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "case_count": len(rows),
        "digest_match_vs_standard": digest_compare,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
