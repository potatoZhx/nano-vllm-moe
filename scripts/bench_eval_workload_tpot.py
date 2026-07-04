#!/usr/bin/env python3
"""Benchmark batch-size-1 TPOT on the evaluation-plan workloads.

This script uses the same nano-vllm-moe runtime configuration as
``scripts/bench_per_layer_slots.py`` but runs real workload prompts from:

* ShareGPT
* MT-Bench
* HumanEval
* MMLU-Pro
* the exact single prompt used by ``scripts/bench_per_layer_slots.py``

It intentionally does not install layer probes and does not enable
``engine_profile``/``spec_profile``. By default requests run with
``ignore_eos=False`` and stop on EOS. For legacy compatibility,
``--output-lens N`` with ``N > 0`` generates exactly like the older benchmark
path: it sets the maximum output tokens and ignores EOS. TPOT is measured with
wall-clock decode time by driving ``LLM.step()`` directly and excluding the
initial prefill step.

Example:

    CUDA_VISIBLE_DEVICES=2 python scripts/bench_eval_workload_tpot.py \
        --dataset sharegpt \
        --num-samples 16 \
        --output-dir results/eval_workload_tpot \
        --gpu-memory-utilization 0.99 \
        --cache-ratios 0.3125 \
        --max-output-tokens 0 \
        --max-draft-tokens-values 12 \
        --segment-sizes 12 \
        --allocation-modes profile_weighted \
        --slot-buckets 4 \
        --slot-max-bucket-ratio 2.0 \
        --slot-profile-csv pre_exps/exp_and_figs/unique/unique_count_plot_summary_n1024.csv \
        --kt-num-threads 16


        CUDA_VISIBLE_DEVICES=3 python scripts/bench_eval_workload_tpot.py \
    --request-mode per_layer_slots \
    --output-dir results/eval_workload_tpot_slots_prompt \
    --gpu-memory-utilization 0.99 \
    --cache-ratios 0.3125 \
    --output-lens 512 \
    --max-draft-tokens-values 12 \
    --segment-sizes 12 \
    --allocation-modes profile_weighted \
    --slot-buckets 4 \
    --slot-max-bucket-ratio 2.0 \
    --slot-profile-csv pre_exps/exp_and_figs/unique/unique_count_plot_summary_n1024.csv \
    --kt-num-threads 16
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Iterable


MODEL_PATH = "/data1/models/Qwen3-30B-A3B"
DEFAULT_PROFILE = "results/reroute_impl_20260531/offline_profile_20260531_203257.safetensors"
DEFAULT_PREDICTOR_PATH = "random_cache_srdp_scripts-1/res/run_20260614_133025"
PER_LAYER_SLOTS_PROMPT_TEXT = (
    "Sparse mixture-of-experts inference keeps only part of each layer's expert weights "
    "in GPU memory. Explain how speculative decoding can overlap expert prefetch with "
    "draft and verify segment computation while preserving exact verification semantics. "
    "Discuss routing-score metadata, bounded transfer budgets, cache eviction protection, "
    "and why late best-effort transfers should be discarded instead of blocking compute."
)

DATASET_PATHS = {
    "sharegpt": "/data1/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json",
    "mt_bench": "/data1/datasets/mt_bench/question.jsonl",
    "humaneval": "/data1/datasets/humaneval/HumanEval.jsonl.gz",
    "mmlu_pro": "/data1/datasets/mmlu_pro/test-00000-of-00001.parquet",
}

DATASET_CHOICES = (
    "sharegpt",
    "mt_bench",
    "humaneval",
    "mmlu_pro",
    "all",
    "per_layer_slots",
)
REQUEST_MODE_CHOICES = ("dataset", "per_layer_slots")


@dataclass
class PromptSample:
    dataset: str
    sample_id: str
    text: str
    source_index: int
    metadata: dict[str, Any]


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {value}")


def _parse_csv(values: str, cast) -> list:
    parsed = [cast(item.strip()) for item in values.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one value")
    return parsed


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


def _dataset_names(dataset_arg: str, request_mode: str = "dataset") -> list[str]:
    if request_mode == "per_layer_slots" or dataset_arg == "per_layer_slots":
        return ["per_layer_slots"]
    if dataset_arg == "all":
        return ["sharegpt", "mt_bench", "humaneval", "mmlu_pro"]
    return [dataset_arg]


def _effective_request_mode(args: argparse.Namespace) -> str:
    if args.request_mode == "per_layer_slots" or args.dataset == "per_layer_slots":
        return "per_layer_slots"
    return "dataset"


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)


def _format_chat_turns(turns: list[tuple[str, str]]) -> str:
    lines: list[str] = []
    for role, content in turns:
        role_norm = role.strip().lower()
        label = "Assistant" if role_norm in {"assistant", "gpt"} else "User"
        content = str(content).strip()
        if content:
            lines.append(f"{label}: {content}")
    lines.append("Assistant:")
    return "\n".join(lines)


def load_sharegpt(path: Path) -> list[PromptSample]:
    data = json.loads(path.read_text(encoding="utf-8"))
    samples: list[PromptSample] = []
    for index, item in enumerate(data):
        conv = item.get("conversations") or []
        if not isinstance(conv, list):
            continue
        last_human = -1
        for turn_index, turn in enumerate(conv):
            if str(turn.get("from", "")).lower() in {"human", "user"}:
                last_human = turn_index
        if last_human < 0:
            continue
        turns: list[tuple[str, str]] = []
        for turn in conv[: last_human + 1]:
            role = str(turn.get("from", "human"))
            content = str(turn.get("value", ""))
            if content.strip():
                turns.append((role, content))
        if not turns:
            continue
        samples.append(
            PromptSample(
                dataset="sharegpt",
                sample_id=str(item.get("id", f"sharegpt_{index}")),
                text=_format_chat_turns(turns),
                source_index=index,
                metadata={"turn_count": len(turns)},
            )
        )
    return samples


def load_mt_bench(path: Path) -> list[PromptSample]:
    samples: list[PromptSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            turns = item.get("turns") or []
            if not turns:
                continue
            prompt = _format_chat_turns([("human", str(turns[0]))])
            samples.append(
                PromptSample(
                    dataset="mt_bench",
                    sample_id=str(item.get("question_id", f"mt_bench_{index}")),
                    text=prompt,
                    source_index=index,
                    metadata={"category": item.get("category", "")},
                )
            )
    return samples


def load_humaneval(path: Path) -> list[PromptSample]:
    samples: list[PromptSample] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            prompt = str(item.get("prompt", "")).rstrip()
            if not prompt:
                continue
            samples.append(
                PromptSample(
                    dataset="humaneval",
                    sample_id=str(item.get("task_id", f"humaneval_{index}")),
                    text=prompt,
                    source_index=index,
                    metadata={"entry_point": item.get("entry_point", "")},
                )
            )
    return samples


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError:
        pq = None
    if pq is not None:
        return list(pq.read_table(path).to_pylist())

    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "MMLU-Pro is stored as parquet. Install pyarrow or pandas in the "
            "active environment, or pass --mmlu-pro-path to a converted JSONL file."
        ) from error
    return list(pd.read_parquet(path).to_dict(orient="records"))


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[arg-type]
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _option_label(index: int) -> str:
    return chr(ord("A") + index)


def _format_mmlu_pro_prompt(row: dict[str, Any]) -> str:
    question = str(row.get("question", "")).strip()
    options = row.get("options", [])
    if isinstance(options, str):
        try:
            parsed = json.loads(options)
            options = parsed if isinstance(parsed, list) else [options]
        except json.JSONDecodeError:
            options = [options]
    if options is None:
        options = []
    option_lines = []
    for idx, option in enumerate(list(options)):
        option_lines.append(f"{_option_label(idx)}. {str(option).strip()}")
    body = "\n".join(option_lines)
    if body:
        return f"Question: {question}\nOptions:\n{body}\nAnswer:"
    return f"Question: {question}\nAnswer:"


def load_mmlu_pro(path: Path) -> list[PromptSample]:
    if path.suffix in {".jsonl", ".gz"}:
        rows = _read_jsonl_rows(path)
    else:
        rows = _read_parquet_rows(path)
    samples: list[PromptSample] = []
    for index, row in enumerate(rows):
        prompt = _format_mmlu_pro_prompt(row)
        if not prompt.strip():
            continue
        sample_id = row.get("question_id", row.get("id", f"mmlu_pro_{index}"))
        samples.append(
            PromptSample(
                dataset="mmlu_pro",
                sample_id=str(sample_id),
                text=prompt,
                source_index=index,
                metadata={"category": row.get("category", row.get("subject", ""))},
            )
        )
    return samples


def load_dataset_samples(dataset: str, args: argparse.Namespace) -> list[PromptSample]:
    if dataset == "per_layer_slots":
        return [
            PromptSample(
                dataset="per_layer_slots",
                sample_id="bench_per_layer_slots_prompt",
                text=PER_LAYER_SLOTS_PROMPT_TEXT,
                source_index=0,
                metadata={
                    "source": "scripts/bench_per_layer_slots.py",
                    "description": "same measured prompt as bench_per_layer_slots.py",
                },
            )
        ]

    path_overrides = {
        "sharegpt": args.sharegpt_path,
        "mt_bench": args.mt_bench_path,
        "humaneval": args.humaneval_path,
        "mmlu_pro": args.mmlu_pro_path,
    }
    path = Path(path_overrides[dataset] or DATASET_PATHS[dataset])
    if not path.exists():
        raise FileNotFoundError(f"{dataset} path not found: {path}")
    if dataset == "sharegpt":
        samples = load_sharegpt(path)
    elif dataset == "mt_bench":
        samples = load_mt_bench(path)
    elif dataset == "humaneval":
        samples = load_humaneval(path)
    elif dataset == "mmlu_pro":
        samples = load_mmlu_pro(path)
    else:
        raise ValueError(f"unsupported dataset: {dataset}")
    if not samples:
        raise RuntimeError(f"no prompts loaded from {path}")
    return samples


def select_samples(
    samples: list[PromptSample],
    *,
    num_samples: int,
    sample_offset: int,
    shuffle: bool,
    seed: int,
) -> list[PromptSample]:
    selected = list(samples)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(selected)
        sample_offset = 0
    if sample_offset:
        selected = selected[sample_offset:]
    if num_samples > 0:
        selected = selected[:num_samples]
    return selected


def build_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    using_output_lens = bool(str(args.output_lens).strip())
    max_output_values = (
        _parse_csv(args.output_lens, int)
        if using_output_lens
        else [int(args.max_output_tokens)]
    )
    for dataset in _dataset_names(args.dataset, args.request_mode):
        for max_output_tokens in max_output_values:
            for cache_ratio in _parse_csv(args.cache_ratios, float):
                for max_draft_tokens in _parse_csv(args.max_draft_tokens_values, int):
                    for segment_size in _parse_csv(args.segment_sizes, int):
                        for repeat in range(int(args.repeats)):
                            for allocation_mode in _parse_allocation_modes(args.allocation_modes):
                                cases.append(
                                    {
                                        "dataset": dataset,
                                        "max_output_tokens": int(max_output_tokens),
                                        "ignore_eos": bool(
                                            using_output_lens and int(max_output_tokens) > 0
                                        ),
                                        "cache_ratio": float(cache_ratio),
                                        "max_draft_tokens": int(max_draft_tokens),
                                        "segment_size": int(segment_size),
                                        "allocation_mode": allocation_mode,
                                        "repeat": int(repeat),
                                    }
                                )
    return cases


def case_name(case: dict[str, Any]) -> str:
    ratio_pct = int(round(float(case["cache_ratio"]) * 10000))
    dataset = _safe_name(str(case["dataset"]))
    alloc = _safe_name(str(case["allocation_mode"]))
    max_out = int(case["max_output_tokens"])
    out_label = "eos" if max_out <= 0 else str(max_out)
    ignore_eos_label = "ieos1" if bool(case.get("ignore_eos", False)) else "ieos0"
    return (
        f"{dataset}_{alloc}_seg{int(case['segment_size'])}_"
        f"ratio{ratio_pct:04d}_maxout{out_label}_{ignore_eos_label}_"
        f"k{int(case['max_draft_tokens'])}_r{int(case['repeat'])}"
    )


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * pct / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    tpot = [float(row["tpot_ms"]) for row in ok_rows]
    decode_tps = [float(row["decode_tok_s"]) for row in ok_rows]
    e2e_tps = [float(row["throughput_output_tok_s"]) for row in ok_rows]
    generated_tokens = [int(row["generated_output_tokens"]) for row in ok_rows]
    prompt_tokens = [int(row["prompt_tokens"]) for row in ok_rows]
    return {
        "sample_count": len(rows),
        "ok_count": len(ok_rows),
        "tpot_ms_mean": float(mean(tpot)) if tpot else 0.0,
        "tpot_ms_p50": percentile(tpot, 50),
        "tpot_ms_p90": percentile(tpot, 90),
        "tpot_ms_p99": percentile(tpot, 99),
        "decode_tok_s_mean": float(mean(decode_tps)) if decode_tps else 0.0,
        "throughput_output_tok_s_mean": float(mean(e2e_tps)) if e2e_tps else 0.0,
        "generated_output_tokens_mean": float(mean(generated_tokens)) if generated_tokens else 0.0,
        "prompt_tokens_mean": float(mean(prompt_tokens)) if prompt_tokens else 0.0,
    }


def grouped_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["dataset"],
            row["allocation_mode"],
            int(row["segment_size"]),
            round(float(row["cache_ratio"]), 6),
            int(row["max_output_tokens"]),
            bool(row.get("ignore_eos", False)),
            int(row["max_draft_tokens"]),
            int(row["repeat"]),
        )
        groups.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for key, group_rows in sorted(groups.items()):
        (
            dataset,
            allocation_mode,
            segment_size,
            cache_ratio,
            max_output_tokens,
            ignore_eos,
            max_draft_tokens,
            repeat,
        ) = key
        summary = summarize_rows(group_rows)
        summary.update(
            {
                "dataset": dataset,
                "allocation_mode": allocation_mode,
                "segment_size": int(segment_size),
                "cache_ratio": float(cache_ratio),
                "max_output_tokens": int(max_output_tokens),
                "ignore_eos": bool(ignore_eos),
                "max_draft_tokens": int(max_draft_tokens),
                "repeat": int(repeat),
            }
        )
        summaries.append(summary)
    return summaries


def prepare_prompt_tokens(
    tokenizer: Any,
    sample: PromptSample,
    *,
    max_input_tokens: int,
    truncate_prompts: bool,
) -> tuple[list[int] | None, dict[str, Any]]:
    token_ids = tokenizer.encode(sample.text)
    original_len = len(token_ids)
    truncated = False
    if max_input_tokens > 0 and original_len > max_input_tokens:
        if not truncate_prompts:
            return None, {
                "skip_reason": "prompt_too_long",
                "prompt_tokens_original": original_len,
                "prompt_tokens": 0,
                "prompt_truncated": False,
            }
        token_ids = token_ids[-max_input_tokens:]
        truncated = True
    if not token_ids:
        return None, {
            "skip_reason": "empty_prompt",
            "prompt_tokens_original": original_len,
            "prompt_tokens": 0,
            "prompt_truncated": truncated,
        }
    return token_ids, {
        "skip_reason": "",
        "prompt_tokens_original": original_len,
        "prompt_tokens": len(token_ids),
        "prompt_truncated": truncated,
    }


def run_prompt(
    llm: Any,
    prompt_tokens: list[int],
    *,
    temperature: float,
    max_tokens: int,
    ignore_eos: bool,
    eos_token_id: int | None,
    max_model_len: int,
) -> dict[str, Any]:
    from nanovllm import SamplingParams

    sampling = SamplingParams(
        temperature=temperature,
        ignore_eos=ignore_eos,
        max_tokens=max_tokens,
    )
    llm.add_request(prompt_tokens, sampling)

    prefill_sec = 0.0
    decode_sec = 0.0
    prefill_steps = 0
    decode_steps = 0
    outputs: dict[int, list[int]] = {}
    elapsed_start = perf_counter()
    while not llm.is_finished():
        step_start = perf_counter()
        step_outputs, num_tokens = llm.step()
        step_elapsed = perf_counter() - step_start
        if num_tokens > 0 and decode_steps == 0:
            prefill_sec += step_elapsed
            prefill_steps += 1
        else:
            decode_sec += step_elapsed
            decode_steps += 1
        for seq_id, token_ids in step_outputs:
            outputs[int(seq_id)] = list(token_ids)
    elapsed_sec = perf_counter() - elapsed_start
    if not outputs:
        raise RuntimeError("request finished without returning output tokens")
    token_ids = next(iter(outputs.values()))
    generated = len(token_ids)
    digest_payload = ",".join(str(token_id) for token_id in token_ids).encode("utf-8")
    stopped_by = "finished"
    if (
        not ignore_eos
        and token_ids
        and eos_token_id is not None
        and token_ids[-1] == eos_token_id
    ):
        stopped_by = "eos"
    elif generated >= max_tokens:
        stopped_by = "max_output_tokens"
    elif len(prompt_tokens) + generated >= max_model_len:
        stopped_by = "max_model_len"
    return {
        "elapsed_sec": elapsed_sec,
        "prefill_sec": prefill_sec,
        "decode_sec": decode_sec,
        "prefill_steps": prefill_steps,
        "decode_steps": decode_steps,
        "generated_output_tokens": generated,
        "tpot_ms": (decode_sec * 1000.0 / generated) if generated else 0.0,
        "decode_tok_s": (generated / decode_sec) if decode_sec > 0 else 0.0,
        "throughput_output_tok_s": (generated / elapsed_sec) if elapsed_sec > 0 else 0.0,
        "max_tokens_limit": int(max_tokens),
        "ignore_eos": bool(ignore_eos),
        "stopped_by": stopped_by,
        "outputs_digest": hashlib.sha256(digest_payload).hexdigest(),
        "generated_token_ids": token_ids,
    }


def create_llm(args: argparse.Namespace, case: dict[str, Any], case_index: int) -> Any:
    import torch
    from nanovllm import LLM
    from transformers import AutoConfig

    torch.manual_seed(int(args.seed) + int(case.get("repeat", 0)))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed) + int(case.get("repeat", 0)))

    hf_config = AutoConfig.from_pretrained(args.model_path)
    num_experts = int(getattr(hf_config, "num_experts"))
    slots = int(args.slots_per_layer)
    if slots <= 0:
        slots = int(round(num_experts * float(case["cache_ratio"])))

    segment_size = int(case["segment_size"])
    return LLM(
        args.model_path,
        dist_port=int(args.dist_port_base) + int(case_index),
        enforce_eager=False,
        max_num_batched_tokens=int(args.max_num_batched_tokens),
        max_num_seqs=1,
        max_model_len=int(args.max_model_len),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        inference_mode="spec",
        enable_heterogeneous=True,
        enable_speculative=True,
        heterogeneous_slots_per_layer=slots,
        heterogeneous_slot_allocation=str(case["allocation_mode"]),
        heterogeneous_slot_buckets=int(args.slot_buckets),
        heterogeneous_slot_max_bucket_ratio=float(args.slot_max_bucket_ratio),
        heterogeneous_slot_profile_csv=str(args.slot_profile_csv),
        max_draft_tokens=int(case["max_draft_tokens"]),
        draft_top_c=0,
        draft_reroute_policy=str(args.draft_reroute_policy),
        draft_reroute_artifact=str(args.profile_artifact),
        acceptance_strategy=str(args.acceptance_strategy),
        acceptance_threshold=float(args.acceptance_threshold),
        acceptance_predictor_enabled=bool(args.acceptance_predictor_enabled),
        acceptance_predictor_path=str(args.acceptance_predictor_path),
        acceptance_predictor_step_horizon=int(args.acceptance_predictor_step_horizon),
        draft_alpha_stop_threshold=float(args.draft_alpha_stop_threshold),
        draft_stop_policy=str(args.draft_stop_policy),
        draft_tpot_td_ms=float(args.draft_tpot_td_ms),
        draft_tpot_tv_ms=float(args.draft_tpot_tv_ms),
        cpu_expert_execution_enabled=True,
        cpu_expert_pin_memory=True,
        cpu_expert_backend="kt_direct",
        cpu_expert_workspace_max_routes=int(args.cpu_expert_workspace_max_routes),
        cpu_expert_packed_min_routes=1,
        cpu_expert_parallel_mode="serial",
        cpu_expert_num_threads=int(args.cpu_expert_num_threads),
        kt_num_threads=int(args.kt_num_threads),
        kt_threadpool_count=int(args.kt_threadpool_count),
        kt_chunked_prefill_size=int(args.kt_chunked_prefill_size),
        kt_direct_backend=str(args.kt_direct_backend),
        kt_numa_nodes=_parse_csv(args.kt_numa_nodes, int) if args.kt_numa_nodes else [],
        kt_capture_bs=_parse_csv(args.kt_capture_bs, int),
        cpu_gpu_parallel_execution_enabled="auto",
        cpu_gpu_parallel_min_cpu_route_ratio=0.0,
        spec_verify_miss_policy="cpu",
        spec_profile=False,
        engine_profile=False,
        engine_profile_cuda_sync=False,
        spec_enable_prefetch=True,
        cache_strategy=str(args.cache_strategy),
        prefetch_strategy="history_window",
        prefetch_runtime_mode="draft_segment_indexed",
        prefetch_runtime_kind="predictive",
        dual_queue_segment_size=segment_size,
        prefetch_verify_attention_ratio=float(args.prefetch_verify_attention_ratio),
        prefetch_staging_slots_per_layer=int(args.prefetch_staging_slots_per_layer),
        prefetch_max_inflight=int(args.prefetch_max_inflight),
        prefetch_transfer_stream_count=int(args.prefetch_transfer_stream_count),
        prefetch_metadata_host_buffer_pool_size=int(args.prefetch_metadata_host_buffer_pool_size),
        prefetch_verify_layer_max_budget=int(args.prefetch_verify_layer_max_budget),
        prefetch_step_budget=int(args.prefetch_step_budget),
        cache_eviction_budget_per_step=int(args.cache_eviction_budget_per_step),
        prefetch_verify_wait_ms=0.0,
        prefetch_global_queue_capacity=int(args.prefetch_global_queue_capacity),
        draft_cuda_graph_enabled=True,
        draft_cuda_graph_cpu_backend="none",
        draft_prefetch_segment_size=segment_size,
        draft_prefetch_segment_host_buffer_pool_size=int(args.draft_segment_host_buffer_pool_size),
        draft_prefetch_visible_budget_ms=float(args.draft_prefetch_visible_budget_ms),
        draft_prefetch_min_per_boundary=0,
        draft_prefetch_max_per_boundary=int(args.draft_prefetch_max_per_boundary),
        verify_cuda_graph=True,
        verify_cuda_graph_bucket_steps=_parse_csv(args.verify_cuda_graph_bucket_steps, int),
        verify_prefetch_segment_size=segment_size,
        verify_prefetch_visible_budget_ms=float(args.verify_prefetch_visible_budget_ms),
        verify_prefetch_min_per_boundary=0,
        verify_prefetch_max_per_boundary=int(args.verify_prefetch_max_per_boundary),
    )


def warmup_llm(llm: Any, *, temperature: float) -> None:
    from nanovllm import SamplingParams

    sampling = SamplingParams(temperature=temperature, ignore_eos=True, max_tokens=4)
    llm.generate(["Warmup request for workload TPOT benchmark."], sampling, use_tqdm=False)


def run_case(
    args: argparse.Namespace,
    case: dict[str, Any],
    case_index: int,
    output_dir: Path,
) -> dict[str, Any]:
    name = case_name(case)
    case_json = output_dir / f"{name}.json"
    if bool(args.skip_existing) and case_json.exists():
        return json.loads(case_json.read_text(encoding="utf-8"))

    loaded = load_dataset_samples(str(case["dataset"]), args)
    selected = select_samples(
        loaded,
        num_samples=int(args.num_samples),
        sample_offset=int(args.sample_offset),
        shuffle=bool(args.shuffle),
        seed=int(args.seed) + int(case.get("repeat", 0)),
    )
    if not selected:
        raise RuntimeError(f"no samples selected for dataset={case['dataset']}")

    max_input_tokens = int(args.max_input_tokens)
    if max_input_tokens <= 0:
        max_input_tokens = int(args.max_model_len) - 1
    if max_input_tokens <= 0:
        raise ValueError(
            f"max input tokens must be positive; max_model_len={args.max_model_len}"
        )

    print(
        f"[{case_index + 1}] running {name}: "
        f"dataset={case['dataset']} samples={len(selected)} "
        f"max_input_tokens={max_input_tokens} "
        f"max_output_tokens={case['max_output_tokens']}",
        flush=True,
    )

    llm = None
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.time()
    try:
        llm = create_llm(args, case, case_index)
        warmup_llm(llm, temperature=float(args.temperature))

        for sample_index, sample in enumerate(selected):
            prompt_tokens, prompt_info = prepare_prompt_tokens(
                llm.tokenizer,
                sample,
                max_input_tokens=max_input_tokens,
                truncate_prompts=bool(args.truncate_prompts),
            )
            base_row = {
                "status": "ok",
                "case_name": name,
                "dataset": sample.dataset,
                "sample_index": sample_index,
                "source_index": sample.source_index,
                "sample_id": sample.sample_id,
                "allocation_mode": str(case["allocation_mode"]),
                "segment_size": int(case["segment_size"]),
                "cache_ratio": float(case["cache_ratio"]),
                "max_output_tokens": int(case["max_output_tokens"]),
                "ignore_eos": bool(case.get("ignore_eos", False)),
                "max_draft_tokens": int(case["max_draft_tokens"]),
                "repeat": int(case["repeat"]),
                **prompt_info,
                "metadata": sample.metadata,
            }
            if prompt_tokens is None:
                base_row["status"] = "skipped"
                rows.append(base_row)
                continue
            try:
                requested_max_output = int(case["max_output_tokens"])
                remaining_model_tokens = max(
                    1,
                    int(args.max_model_len) - int(base_row["prompt_tokens"]),
                )
                max_tokens = (
                    min(requested_max_output, remaining_model_tokens)
                    if requested_max_output > 0
                    else remaining_model_tokens
                )
                ignore_eos = bool(case.get("ignore_eos", False))
                result = run_prompt(
                    llm,
                    prompt_tokens,
                    temperature=float(args.temperature),
                    max_tokens=max_tokens,
                    ignore_eos=ignore_eos,
                    eos_token_id=getattr(llm.config, "eos", None),
                    max_model_len=int(args.max_model_len),
                )
                if bool(args.save_text):
                    token_ids = result.get("generated_token_ids")
                    result["generated_text"] = (
                        llm.tokenizer.decode(token_ids) if token_ids is not None else ""
                    )
                if not bool(args.save_token_ids):
                    result.pop("generated_token_ids", None)
                rows.append({**base_row, **result})
                print(
                    f"  [{sample_index + 1}/{len(selected)}] "
                    f"id={sample.sample_id} prompt={base_row['prompt_tokens']} "
                    f"out={result['generated_output_tokens']} "
                    f"stop={result['stopped_by']} "
                    f"ignore_eos={result['ignore_eos']} "
                    f"tpot={result['tpot_ms']:.3f}ms "
                    f"decode_tok/s={result['decode_tok_s']:.3f}",
                    flush=True,
                )
            except Exception as error:
                failure = {
                    **base_row,
                    "status": "failed",
                    "error": str(error),
                }
                rows.append(failure)
                failures.append(failure)
                if bool(args.fail_fast):
                    raise
                break
    finally:
        if llm is not None:
            llm.exit()

    elapsed = time.time() - started
    summary = {
        "case": dict(case),
        "case_name": name,
        "elapsed_wall_sec": elapsed,
        "dataset_path": str(
            Path(
                {
                    "sharegpt": args.sharegpt_path,
                    "mt_bench": args.mt_bench_path,
                    "humaneval": args.humaneval_path,
                    "mmlu_pro": args.mmlu_pro_path,
                }.get(str(case["dataset"]), "")
                or DATASET_PATHS.get(str(case["dataset"]), "")
            )
        ),
        "selected_sample_count": len(selected),
        "max_input_tokens": max_input_tokens,
        "max_output_tokens": int(case["max_output_tokens"]),
        "ignore_eos": bool(case.get("ignore_eos", False)),
        "rows": rows,
        "summary": summarize_rows(rows),
        "failures": failures,
    }
    case_json.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    s = summary["summary"]
    print(
        f"[{case_index + 1}] {name} done elapsed={elapsed:.1f}s "
        f"ok={s['ok_count']}/{s['sample_count']} "
        f"tpot_mean={s['tpot_ms_mean']:.3f}ms "
        f"p50={s['tpot_ms_p50']:.3f}ms "
        f"p90={s['tpot_ms_p90']:.3f}ms "
        f"decode_tok/s_mean={s['decode_tok_s_mean']:.3f}",
        flush=True,
    )
    return summary


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "status",
        "case_name",
        "dataset",
        "sample_index",
        "source_index",
        "sample_id",
        "allocation_mode",
        "segment_size",
        "cache_ratio",
        "max_output_tokens",
        "ignore_eos",
        "max_draft_tokens",
        "repeat",
        "prompt_tokens_original",
        "prompt_tokens",
        "prompt_truncated",
        "generated_output_tokens",
        "prefill_sec",
        "decode_sec",
        "elapsed_sec",
        "tpot_ms",
        "decode_tok_s",
        "throughput_output_tok_s",
        "prefill_steps",
        "decode_steps",
        "max_tokens_limit",
        "ignore_eos",
        "stopped_by",
        "outputs_digest",
        "skip_reason",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_csv(summaries: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "dataset",
        "allocation_mode",
        "segment_size",
        "cache_ratio",
        "max_output_tokens",
        "ignore_eos",
        "max_draft_tokens",
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
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)


def write_markdown_report(summary: dict[str, Any], path: Path) -> None:
    metadata = summary["metadata"]
    lines = [
        "# Evaluation Workload TPOT Benchmark",
        "",
        f"- timestamp: `{metadata['timestamp']}`",
        f"- model: `{metadata['model_path']}`",
        f"- request mode: `{metadata['request_mode']}`",
        f"- datasets: `{', '.join(metadata['datasets'])}`",
        f"- batch size: `1`",
        f"- output directory: `{metadata['output_dir']}`",
        f"- profile enabled: `false`",
        "",
        "## Summary",
        "",
        "| dataset | alloc | seg | ratio | max out | ignore EOS | K | rep | ok/sample | TPOT mean ms | P50 | P90 | P99 | decode tok/s mean | e2e tok/s mean | prompt tok mean |",
        "|:---|:---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["summaries"]:
        lines.append(
            "| "
            f"{row['dataset']} | {row['allocation_mode']} | "
            f"{row['segment_size']} | {row['cache_ratio']:.4f} | "
            f"{'EOS' if int(row['max_output_tokens']) <= 0 else row['max_output_tokens']} | "
            f"{'true' if row.get('ignore_eos', False) else 'false'} | "
            f"{row['max_draft_tokens']} | {row['repeat']} | "
            f"{row['ok_count']}/{row['sample_count']} | "
            f"{row['tpot_ms_mean']:.3f} | {row['tpot_ms_p50']:.3f} | "
            f"{row['tpot_ms_p90']:.3f} | {row['tpot_ms_p99']:.3f} | "
            f"{row['decode_tok_s_mean']:.3f} | "
            f"{row['throughput_output_tok_s_mean']:.3f} | "
            f"{row['prompt_tokens_mean']:.1f} |"
        )
    if summary["failures"]:
        lines.extend(["", "## Failures", ""])
        for failure in summary["failures"]:
            lines.append(
                f"- `{failure.get('case_name', '')}` sample=`{failure.get('sample_id', '')}`: "
                f"{failure.get('error', failure.get('skip_reason', 'unknown'))}"
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def flatten_rows(case_summaries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_summary in case_summaries:
        rows.extend(case_summary.get("rows", []))
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    os.environ["PYTHONPATH"] = str(repo_root) + os.pathsep + os.environ.get("PYTHONPATH", "")

    cases = build_cases(args)
    case_summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        try:
            case_summaries.append(run_case(args, case, case_index, output_dir))
        except Exception as error:
            failure = {
                "case": case,
                "case_name": case_name(case),
                "error": str(error),
            }
            failures.append(failure)
            print(f"[{case_index + 1}] failed {failure['case_name']}: {error}", flush=True)
            if bool(args.fail_fast):
                raise

    rows = flatten_rows(case_summaries)
    row_failures = [row for row in rows if row.get("status") == "failed"]
    all_failures = failures + row_failures
    summaries = grouped_summaries(rows)
    output = {
        "metadata": {
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "argv": sys.argv,
            "model_path": args.model_path,
            "profile_artifact": args.profile_artifact,
            "slot_profile_csv": args.slot_profile_csv,
            "output_dir": str(output_dir),
            "request_mode": _effective_request_mode(args),
            "datasets": _dataset_names(args.dataset, args.request_mode),
            "num_samples": _num_samples_label(int(args.num_samples)),
            "sample_offset": int(args.sample_offset),
            "shuffle": bool(args.shuffle),
            "allocation_modes": _parse_allocation_modes(args.allocation_modes),
            "cache_ratios": _parse_csv(args.cache_ratios, float),
            "max_output_tokens_values": (
                _parse_csv(args.output_lens, int)
                if str(args.output_lens).strip()
                else [int(args.max_output_tokens)]
            ),
            "output_lens_compat_mode": bool(str(args.output_lens).strip()),
            "max_draft_tokens_values": _parse_csv(args.max_draft_tokens_values, int),
            "segment_sizes": _parse_csv(args.segment_sizes, int),
            "batch_size": 1,
            "engine_profile": False,
            "spec_profile": False,
        },
        "summaries": summaries,
        "rows": rows,
        "failures": all_failures,
    }
    summary_json = output_dir / "summary.json"
    rows_csv = output_dir / "rows.csv"
    summary_csv = output_dir / "summary.csv"
    summary_md = output_dir / "summary.md"
    summary_json.write_text(
        json.dumps(output, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(rows, rows_csv)
    write_summary_csv(summaries, summary_csv)
    write_markdown_report(output, summary_md)
    print(f"summary_json={summary_json}")
    print(f"summary_csv={summary_csv}")
    print(f"rows_csv={rows_csv}")
    print(f"summary_md={summary_md}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-size-1 TPOT benchmark for evaluation_plan.md workloads."
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--profile-artifact", default=DEFAULT_PROFILE)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="sharegpt")
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
    parser.add_argument("--repeats", type=int, default=1)

    parser.add_argument("--cache-strategy", default="lru")
    parser.add_argument("--draft-reroute-policy", default="entropy_cache_bias")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--acceptance-strategy", default="standard_sampling")
    parser.add_argument("--acceptance-threshold", type=float, default=0.7)
    parser.add_argument("--acceptance-predictor-enabled", type=str2bool, default=True)
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

    parser.add_argument("--prefetch-step-budget", type=int, default=16)
    parser.add_argument("--prefetch-max-inflight", type=int, default=16)
    parser.add_argument("--prefetch-transfer-stream-count", type=int, default=1)
    parser.add_argument("--prefetch-staging-slots-per-layer", type=int, default=2)
    parser.add_argument("--prefetch-metadata-host-buffer-pool-size", type=int, default=3)
    parser.add_argument("--prefetch-global-queue-capacity", type=int, default=4096)
    parser.add_argument("--prefetch-verify-layer-max-budget", type=int, default=8)
    parser.add_argument("--prefetch-verify-attention-ratio", type=float, default=1.0)
    parser.add_argument("--cache-eviction-budget-per-step", type=int, default=2)
    parser.add_argument("--draft-segment-host-buffer-pool-size", type=int, default=0)
    parser.add_argument("--draft-prefetch-visible-budget-ms", type=float, default=3.0)
    parser.add_argument("--draft-prefetch-max-per-boundary", type=int, default=16)
    parser.add_argument("--verify-prefetch-visible-budget-ms", type=float, default=12.0)
    parser.add_argument("--verify-prefetch-max-per-boundary", type=int, default=16)

    parser.add_argument("--cpu-expert-workspace-max-routes", type=int, default=327680)
    parser.add_argument("--cpu-expert-num-threads", type=int, default=4)
    parser.add_argument("--kt-num-threads", type=int, default=0)
    parser.add_argument("--kt-threadpool-count", type=int, default=1)
    parser.add_argument("--kt-chunked-prefill-size", type=int, default=4096)
    parser.add_argument(
        "--kt-direct-backend",
        choices=["auto", "amx_bf16", "avx2_bf16"],
        default="auto",
    )
    parser.add_argument("--kt-numa-nodes", default="")
    parser.add_argument("--kt-capture-bs", default="1,2,4,8,16,32")

    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.99)
    parser.add_argument("--verify-cuda-graph-bucket-steps", default="3,5,8,12")
    parser.add_argument("--dist-port-base", type=int, default=31800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-existing", type=str2bool, default=True)
    parser.add_argument("--fail-fast", type=str2bool, default=True)
    parser.add_argument("--save-token-ids", type=str2bool, default=False)
    parser.add_argument("--save-text", type=str2bool, default=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.num_samples < 0:
        raise ValueError("--num-samples must be >= 0 or all")
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    run(args)


if __name__ == "__main__":
    main()
