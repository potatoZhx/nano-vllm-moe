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
wall-clock decode time by excluding the initial prefill step and dividing by
``generated_output_tokens - 1`` inter-token intervals. The default
``--decode-driver step`` drives ``LLM.step()`` directly; ``--decode-driver
generate`` uses the same ``LLM.generate`` driver as ``bench_per_layer_slots.py``
and times each internal step with a hook.

Example:

Validated fixed-K decode configuration for the 0.3125-cache, 512-token target:

    CUDA_VISIBLE_DEVICES=2 python scripts/bench_eval_workload_tpot.py \
        --request-mode per_layer_slots \
        --optimized-config k6_decode \
        --output-lens 512 \
        --output-dir results/eval_workload_tpot_k6_vpb4 \
        --save-token-ids true \
        --save-text true \
        --skip-existing false

    NANOVLLM_VERIFY_PREFETCH_RANK_MULTIPLIER=1 \
    NANOVLLM_VERIFY_DEFER_SEGMENT_METADATA=1 \
    NANOVLLM_VERIFY_BOUNDARY_PREFETCH_ASYNC=0 \
    CUDA_VISIBLE_DEVICES=2 python scripts/bench_eval_workload_tpot.py \
        --dataset mt_bench \
        --num-samples 0 \
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
        --kt-num-threads 16 \
        --verify-cuda-graph-bucket-steps 3,5,7,10,13 \
        --verify-prefetch-max-per-boundary 10 \
        --draft-stop-policy none \
        --verify-prefetch-rank-multiplier 1


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

Legacy K=12 per-layer-slots prompt benchmark:

    NANOVLLM_VERIFY_PREFETCH_RANK_MULTIPLIER=1 \
    NANOVLLM_VERIFY_DEFER_SEGMENT_METADATA=1 \
    NANOVLLM_VERIFY_BOUNDARY_PREFETCH_ASYNC=0 \
    CUDA_VISIBLE_DEVICES=2 python scripts/bench_eval_workload_tpot.py \
        --request-mode per_layer_slots \
        --output-dir results/eval_workload_tpot_k12_optimized \
        --gpu-memory-utilization 0.99 \
        --cache-ratios 0.3125 \
        --output-lens 512 \
        --max-draft-tokens-values 12 \
        --segment-sizes 12 \
        --allocation-modes profile_weighted \
        --slot-buckets 4 \
        --slot-max-bucket-ratio 2.0 \
        --slot-profile-csv pre_exps/exp_and_figs/unique/unique_count_plot_summary_n1024.csv \
        --kt-num-threads 16 \
        --verify-cuda-graph-bucket-steps 3,5,7,10,13 \
        --verify-prefetch-max-per-boundary 10 \
        --draft-stop-policy none \
        --verify-prefetch-rank-multiplier 1 \
        --decode-driver generate \
        --collect-profile true \
        --save-token-ids true
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
DEFAULT_WARMUP_PROMPT = "Warmup request for verify layer profile."
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
OPTIMIZED_CONFIG_CHOICES = (
    "none",
    "k4_verify",
    "k6_decode",
    "k12_decode",
    "k12_bucket_stop",
)
TPOT_DEFINITION = "decode_sec / (generated_output_tokens - 1)"

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
}


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


def runtime_seed(args: argparse.Namespace, case: dict[str, Any], sample_index: int = 0) -> int:
    return int(args.seed) + int(case.get("repeat", 0)) + int(sample_index)


def reset_runtime_seed(seed: int) -> None:
    random.seed(int(seed))
    try:
        import torch
    except Exception:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


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
        "draft_stop_policy": "--draft-stop-policy",
        "draft_tpot_stop_rule": "--draft-tpot-stop-rule",
        "draft_tpot_min_steps": "--draft-tpot-min-steps",
        "draft_tpot_stop_margin": "--draft-tpot-stop-margin",
        "draft_tpot_lookahead_cache_credit_ms_per_step": (
            "--draft-tpot-lookahead-cache-credit-ms-per-step"
        ),
        "draft_tpot_verify_model_mode": "--draft-tpot-verify-model-mode",
        "acceptance_predictor_enabled": "--acceptance-predictor-enabled",
        "kt_num_threads": "--kt-num-threads",
        "kt_direct_backend": "--kt-direct-backend",
        "verify_cuda_graph_bucket_steps": "--verify-cuda-graph-bucket-steps",
        "verify_prefetch_rank_multiplier": "--verify-prefetch-rank-multiplier",
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


def configure_optimized_env(args: argparse.Namespace) -> dict[str, str]:
    optimized_config = str(getattr(args, "optimized_config", "none") or "none")
    rank_multiplier = getattr(args, "verify_prefetch_rank_multiplier", None)
    verify_cost_profile = bool(getattr(args, "verify_cost_model_profile", False))
    should_configure = (
        optimized_config != "none"
        or rank_multiplier is not None
        or verify_cost_profile
    )
    if not should_configure:
        os.environ.pop("NANOVLLM_VERIFY_COST_MODEL_PROFILE", None)
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


def _requested_datasets(args: argparse.Namespace) -> list[str]:
    raw = str(getattr(args, "dataset_list", "") or "").strip()
    if not raw:
        return _dataset_names(args.dataset, args.request_mode)
    datasets = [item.strip() for item in raw.split(",") if item.strip()]
    if not datasets:
        raise argparse.ArgumentTypeError("--dataset-list must not be empty")
    invalid = sorted(set(datasets) - set(DATASET_CHOICES))
    if invalid or "all" in datasets:
        raise argparse.ArgumentTypeError(
            f"--dataset-list contains unsupported entries: {invalid or ['all']}"
        )
    return datasets


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
                text=PER_LAYER_SLOTS_PROMPT_TEXT + "\n",
                source_index=0,
                metadata={
                    "source": "scripts/bench_per_layer_slots.py",
                    "description": "same measured prompt bytes as bench_per_layer_slots.py",
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


def _effective_sample_offset(dataset: str, sample_offset: int) -> int:
    # The per-layer workload is a single fixed synthetic prompt, not an indexed
    # dataset split. Dataset holdout offsets therefore do not apply to it.
    return 0 if str(dataset) == "per_layer_slots" else int(sample_offset)


def build_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    using_output_lens = bool(str(args.output_lens).strip())
    max_output_values = (
        _parse_csv(args.output_lens, int)
        if using_output_lens
        else [int(args.max_output_tokens)]
    )
    for dataset in _requested_datasets(args):
        for max_output_tokens in max_output_values:
            for cache_ratio in _parse_csv(args.cache_ratios, float):
                for max_draft_tokens in _parse_csv(args.max_draft_tokens_values, int):
                    for segment_size in _parse_csv(args.segment_sizes, int):
                        for repeat_offset in range(int(args.repeats)):
                            repeat = int(args.repeat_index_offset) + repeat_offset
                            for allocation_mode in _parse_allocation_modes(args.allocation_modes):
                                cases.append(
                                    {
                                        "dataset": dataset,
                                        "optimized_config": str(args.optimized_config),
                                        "max_output_tokens": int(max_output_tokens),
                                        "ignore_eos": bool(
                                            using_output_lens and int(max_output_tokens) > 0
                                        ),
                                        "cache_ratio": float(cache_ratio),
                                        "max_draft_tokens": int(max_draft_tokens),
                                        "segment_size": int(segment_size),
                                        "allocation_mode": allocation_mode,
                                        "draft_stop_policy": str(args.draft_stop_policy),
                                        "acceptance_predictor_enabled": bool(
                                            args.acceptance_predictor_enabled
                                        ),
                                        "draft_tpot_verify_model_mode": str(
                                            args.draft_tpot_verify_model_mode
                                        ),
                                        "verify_prefetch_max_per_boundary": int(
                                            args.verify_prefetch_max_per_boundary
                                        ),
                                        "verify_prefetch_rank_multiplier": (
                                            int(args.verify_prefetch_rank_multiplier)
                                            if args.verify_prefetch_rank_multiplier is not None
                                            else None
                                        ),
                                        "repeat": int(repeat),
                                    }
                                )
    return cases


def case_name(case: dict[str, Any]) -> str:
    ratio_pct = int(round(float(case["cache_ratio"]) * 10000))
    dataset = _safe_name(str(case["dataset"]))
    alloc = _safe_name(str(case["allocation_mode"]))
    opt_config = _safe_name(str(case.get("optimized_config", "none")))
    opt_label = "" if opt_config == "none" else f"_{opt_config}"
    draft_stop_policy = str(case.get("draft_stop_policy", ""))
    verify_prefetch_budget = int(case.get("verify_prefetch_max_per_boundary", 0) or 0)
    rank_multiplier = case.get("verify_prefetch_rank_multiplier")
    stop_label = ""
    verify_model_mode = str(
        case.get("draft_tpot_verify_model_mode", "off")
    )
    include_tuning_label = (
        opt_config != "none"
        or (draft_stop_policy and draft_stop_policy != "tpot")
        or verify_prefetch_budget not in {0, 4}
        or rank_multiplier is not None
    )
    if include_tuning_label:
        stop_label = (
            f"_stop{_safe_name(draft_stop_policy)}"
            f"_vpb{verify_prefetch_budget}"
        )
        if rank_multiplier is not None:
            stop_label += f"_rank{int(rank_multiplier)}"
    if verify_model_mode != "off":
        stop_label += f"_vcost{_safe_name(verify_model_mode)}"
    max_out = int(case["max_output_tokens"])
    out_label = "eos" if max_out <= 0 else str(max_out)
    ignore_eos_label = "ieos1" if bool(case.get("ignore_eos", False)) else "ieos0"
    return (
        f"{dataset}_{alloc}{opt_label}_seg{int(case['segment_size'])}_"
        f"ratio{ratio_pct:04d}_maxout{out_label}_{ignore_eos_label}_"
        f"k{int(case['max_draft_tokens'])}{stop_label}_r{int(case['repeat'])}"
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
    decode_intervals = [int(row["decode_token_intervals"]) for row in ok_rows]
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
        "decode_token_intervals_mean": float(mean(decode_intervals)) if decode_intervals else 0.0,
        "prompt_tokens_mean": float(mean(prompt_tokens)) if prompt_tokens else 0.0,
    }


def grouped_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["dataset"],
            row.get("optimized_config", "none"),
            row["allocation_mode"],
            int(row["segment_size"]),
            round(float(row["cache_ratio"]), 6),
            int(row["max_output_tokens"]),
            bool(row.get("ignore_eos", False)),
            int(row["max_draft_tokens"]),
            row.get("draft_stop_policy", ""),
            int(row.get("verify_prefetch_max_per_boundary", 0) or 0),
            int(row["repeat"]),
        )
        groups.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for key, group_rows in sorted(groups.items()):
        (
            dataset,
            optimized_config,
            allocation_mode,
            segment_size,
            cache_ratio,
            max_output_tokens,
            ignore_eos,
            max_draft_tokens,
            draft_stop_policy,
            verify_prefetch_max_per_boundary,
            repeat,
        ) = key
        summary = summarize_rows(group_rows)
        summary.update(
            {
                "dataset": dataset,
                "optimized_config": optimized_config,
                "allocation_mode": allocation_mode,
                "segment_size": int(segment_size),
                "cache_ratio": float(cache_ratio),
                "max_output_tokens": int(max_output_tokens),
                "ignore_eos": bool(ignore_eos),
                "max_draft_tokens": int(max_draft_tokens),
                "draft_stop_policy": str(draft_stop_policy),
                "verify_prefetch_max_per_boundary": int(
                    verify_prefetch_max_per_boundary
                ),
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
    prefill_step_ms: list[float] = []
    decode_step_ms: list[float] = []
    outputs: dict[int, list[int]] = {}
    elapsed_start = perf_counter()
    while not llm.is_finished():
        step_start = perf_counter()
        step_outputs, num_tokens = llm.step()
        step_elapsed = perf_counter() - step_start
        step_ms = step_elapsed * 1000.0
        if num_tokens > 0 and decode_steps == 0:
            prefill_sec += step_elapsed
            prefill_steps += 1
            prefill_step_ms.append(step_ms)
        else:
            decode_sec += step_elapsed
            decode_steps += 1
            decode_step_ms.append(step_ms)
        for seq_id, token_ids in step_outputs:
            outputs[int(seq_id)] = list(token_ids)
    elapsed_sec = perf_counter() - elapsed_start
    if not outputs:
        raise RuntimeError("request finished without returning output tokens")
    token_ids = next(iter(outputs.values()))
    return finalize_prompt_result(
        token_ids,
        elapsed_sec=elapsed_sec,
        prefill_sec=prefill_sec,
        decode_sec=decode_sec,
        prefill_steps=prefill_steps,
        decode_steps=decode_steps,
        prefill_step_ms=prefill_step_ms,
        decode_step_ms=decode_step_ms,
        output_sequence_count=len(outputs),
        max_tokens=max_tokens,
        ignore_eos=ignore_eos,
        eos_token_id=eos_token_id,
        prompt_tokens=prompt_tokens,
        max_model_len=max_model_len,
    )


def run_prompt_generate(
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

    original_step = llm.step
    prefill_sec = 0.0
    decode_sec = 0.0
    prefill_steps = 0
    decode_steps = 0
    prefill_step_ms: list[float] = []
    decode_step_ms: list[float] = []

    def timed_step():
        nonlocal prefill_sec, decode_sec, prefill_steps, decode_steps
        step_start = perf_counter()
        step_outputs, num_tokens = original_step()
        step_elapsed = perf_counter() - step_start
        step_ms = step_elapsed * 1000.0
        if num_tokens > 0 and decode_steps == 0:
            prefill_sec += step_elapsed
            prefill_steps += 1
            prefill_step_ms.append(step_ms)
        else:
            decode_sec += step_elapsed
            decode_steps += 1
            decode_step_ms.append(step_ms)
        return step_outputs, num_tokens

    elapsed_start = perf_counter()
    llm.step = timed_step
    try:
        outputs = llm.generate([prompt_tokens], sampling, use_tqdm=False)
    finally:
        llm.step = original_step
    elapsed_sec = perf_counter() - elapsed_start

    if len(outputs) != 1:
        raise RuntimeError(f"expected exactly one output sequence, got {len(outputs)}")
    token_ids = list(outputs[0].get("token_ids", []))
    if not token_ids:
        raise RuntimeError("request finished without returning output tokens")
    return finalize_prompt_result(
        token_ids,
        elapsed_sec=elapsed_sec,
        prefill_sec=prefill_sec,
        decode_sec=decode_sec,
        prefill_steps=prefill_steps,
        decode_steps=decode_steps,
        prefill_step_ms=prefill_step_ms,
        decode_step_ms=decode_step_ms,
        output_sequence_count=len(outputs),
        max_tokens=max_tokens,
        ignore_eos=ignore_eos,
        eos_token_id=eos_token_id,
        prompt_tokens=prompt_tokens,
        max_model_len=max_model_len,
    )


def max_repeated_token_run(token_ids: list[int]) -> int:
    if not token_ids:
        return 0
    best = 1
    current = 1
    for prev, token_id in zip(token_ids, token_ids[1:]):
        if token_id == prev:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def finalize_prompt_result(
    token_ids: list[int],
    *,
    elapsed_sec: float,
    prefill_sec: float,
    decode_sec: float,
    prefill_steps: int,
    decode_steps: int,
    prefill_step_ms: list[float],
    decode_step_ms: list[float],
    output_sequence_count: int,
    max_tokens: int,
    ignore_eos: bool,
    eos_token_id: int | None,
    prompt_tokens: list[int],
    max_model_len: int,
) -> dict[str, Any]:
    generated = len(token_ids)
    # Prefill produces the first completion token and prefill_sec is excluded
    # from decode_sec. TPOT therefore spans the remaining inter-token intervals.
    decode_token_intervals = max(generated - 1, 0)
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
    validation_errors = []
    if output_sequence_count != 1:
        validation_errors.append(f"output_sequence_count={output_sequence_count}")
    if ignore_eos and max_tokens > 0 and generated != max_tokens:
        validation_errors.append(f"generated={generated} expected={max_tokens}")
    return {
        "elapsed_sec": elapsed_sec,
        "prefill_sec": prefill_sec,
        "decode_sec": decode_sec,
        "prefill_step_wall_ms_sum": sum(prefill_step_ms),
        "decode_step_wall_ms_sum": sum(decode_step_ms),
        "decode_step_wall_ms_mean": (
            sum(decode_step_ms) / len(decode_step_ms) if decode_step_ms else 0.0
        ),
        "decode_step_wall_ms_p50": percentile(decode_step_ms, 50),
        "decode_step_wall_ms_p90": percentile(decode_step_ms, 90),
        "decode_step_wall_ms_max": max(decode_step_ms) if decode_step_ms else 0.0,
        "prefill_steps": prefill_steps,
        "decode_steps": decode_steps,
        "generated_output_tokens": generated,
        "decode_token_intervals": decode_token_intervals,
        "output_sequence_count": int(output_sequence_count),
        "output_fixed_length_ok": bool(
            (not ignore_eos) or max_tokens <= 0 or generated == max_tokens
        ),
        "output_validation_error": ";".join(validation_errors),
        "max_repeated_token_run": max_repeated_token_run(token_ids),
        "tpot_ms": (
            decode_sec * 1000.0 / decode_token_intervals
            if decode_token_intervals
            else 0.0
        ),
        "decode_tok_s": (
            decode_token_intervals / decode_sec
            if decode_sec > 0.0 and decode_token_intervals
            else 0.0
        ),
        "throughput_output_tok_s": (generated / elapsed_sec) if elapsed_sec > 0 else 0.0,
        "max_tokens_limit": int(max_tokens),
        "ignore_eos": bool(ignore_eos),
        "stopped_by": stopped_by,
        "outputs_digest": hashlib.sha256(digest_payload).hexdigest(),
        "generated_token_ids": token_ids,
    }


def reset_llm_profile(llm: Any) -> None:
    if not hasattr(llm, "get_profile"):
        return
    llm.get_profile(reset=True)


def collect_profile_metrics(profile: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    scalar_keys = [
        "step_count",
        "step_ms",
        "spec_step_count",
        "spec_engine_ms",
        "spec_spec_step_count",
        "spec_spec_step_ms",
        "spec_draft_ms",
        "spec_verify_ms",
        "spec_draft_loop_ms",
        "spec_start_draft_ms",
        "spec_rollback_ms",
        "spec_prepare_verify_ms",
        "spec_accept_ms",
        "spec_run_draft_calls",
        "spec_run_verify_calls",
        "spec_run_draft_infer_ms_total",
        "spec_run_verify_infer_ms_total",
        "spec_draft_steps_total",
        "spec_draft_tpot_early_stop_count",
        "spec_draft_alpha_early_stop_count",
        "spec_draft_tpot_draft_ms_ema",
        "spec_draft_tpot_verify_ms_ema",
        "spec_accepted_tokens_total",
        "spec_draft_tokens_total",
        "spec_verify_trace_tokens_total",
        "model_graph_hit_rate",
        "model_graph_replay_count",
        "model_decode_count",
        "model_prefetch_submit_count",
        "model_prefetch_completed_count",
        "model_prefetch_late_count",
        "model_prefetch_wait_ms",
        "model_prefetch_consumed_count",
        "model_publish_count",
        "model_publish_ms",
        "model_metadata_offload_count",
        "model_metadata_offload_ms",
        "model_metadata_offload_bytes",
        "model_metadata_offload_enqueue_ms",
        "model_metadata_offload_transfer_wait_ms",
        "model_metadata_offload_collect_ms",
        "model_metadata_offload_observe_ms",
        "model_metadata_offload_draft_count",
        "model_metadata_offload_draft_ms",
        "model_metadata_offload_verify_count",
        "model_metadata_offload_verify_ms",
        "model_realized_cpu_expert_count",
        "model_verify_kt_hybrid_segment_graph_replay_count",
        "model_verify_cpu_routes_sum",
        "model_verify_realized_cpu_expert_count_sum",
        "model_verify_pre_transfer_cache_miss_sum",
        "model_verify_pre_transfer_active_count_sum",
        "model_run_verify_kt_hybrid_metadata_wait_ms",
        "model_run_verify_kt_hybrid_metadata_collect_ms",
        "model_run_verify_kt_hybrid_metadata_observe_ms",
        "model_verify_segment_graph_replay_enqueue_ms",
        "model_verify_tpot_dynamic_budget_applied_count",
        "model_verify_tpot_dynamic_budget_token_sum",
        "model_verify_tpot_dynamic_budget_value_sum",
        "model_draft_perfect_reject_events",
        "model_draft_perfect_followup_events",
        "model_draft_perfect_checked_tokens",
        "model_draft_perfect_perfect_tokens",
        "model_draft_perfect_token_rate",
        "model_draft_perfect_prefix_ge1_events",
        "model_draft_perfect_prefix_ge1_rate",
        "model_draft_perfect_perfect_prefix_token_sum",
        "model_draft_perfect_route_total",
        "model_draft_perfect_route_miss",
        "model_draft_perfect_route_miss_ratio",
        "model_draft_perfect_coverage_total",
        "model_draft_perfect_coverage_hit",
        "model_draft_perfect_coverage_ratio",
        "model_draft_perfect_pred_row_match_ratio",
        "model_draft_perfect_input_row_match_ratio",
        "model_draft_perfect_oracle_covered_tokens",
        "model_draft_perfect_oracle_covered_token_rate",
        "model_draft_perfect_oracle_prefix_token_sum",
        "model_draft_perfect_oracle_prefix_ge1_events",
        "model_draft_perfect_oracle_prefix_ge1_rate",
        "model_draft_perfect_refill_events",
        "model_draft_perfect_refill_promoted",
        "model_draft_perfect_refill_cpu_experts",
        "model_draft_perfect_refill_skipped_inflight_events",
        "model_draft_perfect_refill_skipped_inflight_count",
        "draft_forward_ms",
        "verify_forward_ms",
        "draft_ms",
        "verify_ms",
        "spec_step_ms",
    ]
    metrics: dict[str, Any] = {}
    for key in scalar_keys:
        value = profile.get(key)
        if isinstance(value, (bool, int, float)):
            metrics[f"profile_{key}"] = value
        elif key.endswith("_count"):
            metrics[f"profile_{key}"] = 0

    decode_intervals = int(result.get("decode_token_intervals", 0) or 0)
    wall_decode_ms = float(result.get("decode_sec", 0.0) or 0.0) * 1000.0
    spec_step_ms = float(profile.get("spec_spec_step_ms", 0.0) or 0.0)
    engine_step_ms = float(profile.get("step_ms", 0.0) or 0.0)
    if decode_intervals > 0 and spec_step_ms > 0.0:
        metrics["profile_decode_phase_output_tok_s"] = (
            decode_intervals / (spec_step_ms / 1000.0)
        )
    if spec_step_ms > 0.0:
        metrics["profile_wall_minus_spec_step_ms"] = wall_decode_ms - spec_step_ms
    if engine_step_ms > 0.0:
        metrics["profile_wall_minus_engine_step_ms"] = wall_decode_ms - engine_step_ms
    verify_calls = float(profile.get("spec_run_verify_calls", 0.0) or 0.0)
    if verify_calls > 0.0:
        metrics["profile_wall_ms_per_verify"] = wall_decode_ms / verify_calls
        metrics["profile_spec_step_ms_per_verify"] = spec_step_ms / verify_calls
    draft_tokens = float(profile.get("spec_draft_tokens_total", 0.0) or 0.0)
    accepted = float(profile.get("spec_accepted_tokens_total", 0.0) or 0.0)
    if draft_tokens > 0.0:
        metrics["profile_acceptance_rate"] = accepted / draft_tokens
    return metrics


def create_llm(args: argparse.Namespace, case: dict[str, Any], case_index: int) -> Any:
    from nanovllm import LLM
    from transformers import AutoConfig

    reset_runtime_seed(runtime_seed(args, case, 0))

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
        draft_tpot_cost_model=str(args.draft_tpot_cost_model),
        draft_tpot_history_alpha=float(args.draft_tpot_history_alpha),
        draft_tpot_min_steps=int(args.draft_tpot_min_steps),
        draft_tpot_stop_margin=float(args.draft_tpot_stop_margin),
        draft_tpot_stop_patience=int(args.draft_tpot_stop_patience),
        draft_tpot_lookahead_cache_credit_ms_per_step=float(
            args.draft_tpot_lookahead_cache_credit_ms_per_step
        ),
        draft_tpot_short_verify_penalty_ms=float(args.draft_tpot_short_verify_penalty_ms),
        draft_tpot_verify_cost_floor_ms=float(args.draft_tpot_verify_cost_floor_ms),
        draft_tpot_stop_rule=str(args.draft_tpot_stop_rule),
        draft_tpot_verify_model_mode=str(args.draft_tpot_verify_model_mode),
        draft_tpot_verify_model_path=str(args.draft_tpot_verify_model_path),
        draft_tpot_alpha_calibration_path=str(
            args.draft_tpot_alpha_calibration_path
        ),
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
        engine_profile=bool(args.engine_profile),
        engine_profile_cuda_sync=bool(args.engine_profile_cuda_sync),
        spec_enable_prefetch=True,
        cache_strategy=str(args.cache_strategy),
        rank_guard_threshold=float(args.rank_guard_threshold),
        rank_guard_ema_alpha=float(args.rank_guard_ema_alpha),
        prefetch_strategy="history_window",
        prefetch_runtime_mode="draft_segment_indexed",
        prefetch_runtime_kind=str(args.prefetch_runtime_kind),
        dual_queue_segment_size=segment_size,
        dual_queue_ground_truth_decay=float(args.dual_queue_ground_truth_decay),
        dual_queue_ground_truth_ttl_rounds=int(args.dual_queue_ground_truth_ttl_rounds),
        dual_queue_ground_truth_count_weight=float(args.dual_queue_ground_truth_count_weight),
        dual_queue_budget_safety_ratio=float(args.dual_queue_budget_safety_ratio),
        dual_queue_segment_time_ema_alpha=float(args.dual_queue_segment_time_ema_alpha),
        dual_queue_secondary_index_weight=float(args.dual_queue_secondary_index_weight),
        prefetch_verify_attention_ratio=float(args.prefetch_verify_attention_ratio),
        predictive_phase1_budget=int(args.predictive_phase1_budget),
        prefetch_staging_slots_per_layer=int(args.prefetch_staging_slots_per_layer),
        prefetch_max_inflight=int(args.prefetch_max_inflight),
        prefetch_transfer_stream_count=int(args.prefetch_transfer_stream_count),
        prefetch_metadata_host_buffer_pool_size=int(args.prefetch_metadata_host_buffer_pool_size),
        prefetch_verify_layer_max_budget=int(args.prefetch_verify_layer_max_budget),
        prefetch_step_budget=int(args.prefetch_step_budget),
        cache_eviction_budget_per_step=int(args.cache_eviction_budget_per_step),
        prefetch_verify_wait_ms=0.0,
        prefetch_global_queue_capacity=int(args.prefetch_global_queue_capacity),
        prefetch_history_decay=float(args.prefetch_history_decay),
        prefetch_history_ttl_steps=int(args.prefetch_history_ttl_steps),
        prefetch_source_weight_prefill=float(args.prefetch_source_weight_prefill),
        prefetch_source_weight_verify=float(args.prefetch_source_weight_verify),
        prefetch_source_weight_draft=float(args.prefetch_source_weight_draft),
        prefetch_activation_count_weight=float(args.prefetch_activation_count_weight),
        prefetch_age_penalty=float(args.prefetch_age_penalty),
        prefetch_use_prefill_history=bool(args.prefetch_use_prefill_history),
        prefetch_use_verify_history=bool(args.prefetch_use_verify_history),
        prefetch_use_draft_live=bool(args.prefetch_use_draft_live),
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
        verify_prefetch_tpot_dynamic_budget_enabled=bool(args.verify_prefetch_tpot_dynamic_budget_enabled),
        verify_prefetch_tpot_dynamic_budget_token_threshold=int(
            args.verify_prefetch_tpot_dynamic_budget_token_threshold
        ),
        verify_prefetch_tpot_dynamic_budget_small=int(args.verify_prefetch_tpot_dynamic_budget_small),
    )


def warmup_llm(llm: Any, *, temperature: float, prompt: str) -> None:
    from nanovllm import SamplingParams

    sampling = SamplingParams(temperature=temperature, ignore_eos=True, max_tokens=4)
    llm.generate([prompt], sampling, use_tqdm=False)


def run_case(
    args: argparse.Namespace,
    case: dict[str, Any],
    case_index: int,
    output_dir: Path,
    llm: Any | None = None,
) -> dict[str, Any]:
    name = case_name(case)
    case_json = output_dir / f"{name}.json"
    if bool(args.skip_existing) and case_json.exists():
        return json.loads(case_json.read_text(encoding="utf-8"))

    loaded = load_dataset_samples(str(case["dataset"]), args)
    sample_offset = _effective_sample_offset(
        str(case["dataset"]),
        int(args.sample_offset),
    )
    selected = select_samples(
        loaded,
        num_samples=int(args.num_samples),
        sample_offset=sample_offset,
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

    owns_llm = llm is None
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.time()
    try:
        if owns_llm:
            llm = create_llm(args, case, case_index)
            warmup_llm(
                llm,
                temperature=float(args.temperature),
                prompt=str(args.warmup_prompt),
            )
        else:
            llm.config.max_draft_tokens = int(case["max_draft_tokens"])
            llm.spec_engine.max_draft_tokens = int(case["max_draft_tokens"])
        if (
            owns_llm
            and bool(args.reset_profile_after_warmup)
        ) or bool(args.collect_profile):
            reset_llm_profile(llm)

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
                "optimized_config": str(case.get("optimized_config", "none")),
                "allocation_mode": str(case["allocation_mode"]),
                "segment_size": int(case["segment_size"]),
                "cache_ratio": float(case["cache_ratio"]),
                "max_output_tokens": int(case["max_output_tokens"]),
                "ignore_eos": bool(case.get("ignore_eos", False)),
                "max_draft_tokens": int(case["max_draft_tokens"]),
                "draft_stop_policy": str(
                    case.get("draft_stop_policy", args.draft_stop_policy)
                ),
                "draft_tpot_verify_model_mode": str(
                    case.get(
                        "draft_tpot_verify_model_mode",
                        args.draft_tpot_verify_model_mode,
                    )
                ),
                "verify_prefetch_max_per_boundary": int(
                    case.get(
                        "verify_prefetch_max_per_boundary",
                        args.verify_prefetch_max_per_boundary,
                    )
                ),
                "verify_prefetch_rank_multiplier": (
                    int(case["verify_prefetch_rank_multiplier"])
                    if case.get("verify_prefetch_rank_multiplier") is not None
                    else None
                ),
                "repeat": int(case["repeat"]),
                "runtime_seed": runtime_seed(args, case, sample_index),
                "decode_driver": str(args.decode_driver),
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
                if bool(args.reset_seed_after_warmup):
                    reset_runtime_seed(int(base_row["runtime_seed"]))
                if bool(args.collect_profile) and bool(args.reset_profile_before_request):
                    reset_llm_profile(llm)
                run_prompt_fn = (
                    run_prompt_generate
                    if str(args.decode_driver) == "generate"
                    else run_prompt
                )
                result = run_prompt_fn(
                    llm,
                    prompt_tokens,
                    temperature=float(args.temperature),
                    max_tokens=max_tokens,
                    ignore_eos=ignore_eos,
                    eos_token_id=getattr(llm.config, "eos", None),
                    max_model_len=int(args.max_model_len),
                )
                if bool(args.collect_profile):
                    profile = llm.get_profile(reset=True)
                    result.update(collect_profile_metrics(profile, result))
                    if bool(args.save_profile_json):
                        profile["verify_cost_measurement"] = {
                            "enabled": bool(args.verify_cost_model_profile),
                            "target": "spec.verify_accept_ready_ms",
                            "target_boundary": (
                                "before ModelRunner.run_verify through acceptance-result "
                                "consumption on the host"
                            ),
                            "execution_workload": (
                                "all CUDA-graph bucket rows, including padding"
                            ),
                            "profile_cuda_sync": bool(args.engine_profile_cuda_sync),
                            "case": dict(case),
                            "sample": {
                                "dataset": str(base_row["dataset"]),
                                "sample_id": str(base_row["sample_id"]),
                                "sample_index": int(base_row["sample_index"]),
                                "source_index": int(base_row["source_index"]),
                            },
                            "runtime_seed": int(base_row["runtime_seed"]),
                            "optimized_env_overrides": dict(
                                getattr(args, "_optimized_env_overrides", {})
                            ),
                        }
                        profile_dir = output_dir / f"{name}_profiles"
                        profile_dir.mkdir(parents=True, exist_ok=True)
                        profile_path = profile_dir / f"sample{sample_index:04d}.json"
                        profile_path.write_text(
                            json.dumps(profile, ensure_ascii=True, indent=2) + "\n",
                            encoding="utf-8",
                        )
                        result["profile_json"] = str(profile_path)
                if (
                    bool(args.fail_on_output_validation_error)
                    and result.get("output_validation_error")
                ):
                    raise RuntimeError(
                        f"output validation failed: {result['output_validation_error']}"
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
                    f"seqs={result.get('output_sequence_count', '')} "
                    f"stop={result['stopped_by']} "
                    f"ignore_eos={result['ignore_eos']} "
                    f"valid={result.get('output_fixed_length_ok', '')} "
                    f"tpot={result['tpot_ms']:.3f}ms "
                    f"decode_tok/s={result['decode_tok_s']:.3f}"
                    + (
                        f" profile_decode_tok/s={float(result['profile_decode_phase_output_tok_s']):.3f}"
                        f" profile_gap={float(result.get('profile_wall_minus_spec_step_ms', 0.0)):.3f}ms"
                        if "profile_decode_phase_output_tok_s" in result
                        else ""
                    ),
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
        if owns_llm and llm is not None:
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
        "optimized_config",
        "allocation_mode",
        "segment_size",
        "cache_ratio",
        "max_output_tokens",
        "ignore_eos",
        "max_draft_tokens",
        "draft_stop_policy",
        "verify_prefetch_max_per_boundary",
        "verify_prefetch_rank_multiplier",
        "repeat",
        "runtime_seed",
        "decode_driver",
        "prompt_tokens_original",
        "prompt_tokens",
        "prompt_truncated",
        "generated_output_tokens",
        "decode_token_intervals",
        "output_sequence_count",
        "output_fixed_length_ok",
        "output_validation_error",
        "max_repeated_token_run",
        "prefill_sec",
        "decode_sec",
        "prefill_step_wall_ms_sum",
        "decode_step_wall_ms_sum",
        "decode_step_wall_ms_mean",
        "decode_step_wall_ms_p50",
        "decode_step_wall_ms_p90",
        "decode_step_wall_ms_max",
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
        "profile_decode_phase_output_tok_s",
        "profile_wall_minus_spec_step_ms",
        "profile_wall_minus_engine_step_ms",
        "profile_wall_ms_per_verify",
        "profile_spec_step_ms_per_verify",
        "profile_acceptance_rate",
        "profile_spec_spec_step_ms",
        "profile_step_ms",
        "profile_spec_engine_ms",
        "profile_spec_draft_ms",
        "profile_spec_verify_ms",
        "profile_spec_run_draft_calls",
        "profile_spec_run_verify_calls",
        "profile_spec_draft_tpot_early_stop_count",
        "profile_spec_draft_alpha_early_stop_count",
        "profile_spec_draft_tpot_draft_ms_ema",
        "profile_spec_draft_tpot_verify_ms_ema",
        "profile_draft_forward_ms",
        "profile_verify_forward_ms",
        "profile_model_graph_hit_rate",
        "profile_model_graph_replay_count",
        "profile_model_realized_cpu_expert_count",
        "profile_model_verify_kt_hybrid_segment_graph_replay_count",
        "profile_model_verify_cpu_routes_sum",
        "profile_model_verify_realized_cpu_expert_count_sum",
        "profile_model_verify_pre_transfer_cache_miss_sum",
        "profile_model_verify_pre_transfer_active_count_sum",
        "profile_model_run_verify_kt_hybrid_metadata_wait_ms",
        "profile_model_run_verify_kt_hybrid_metadata_collect_ms",
        "profile_model_run_verify_kt_hybrid_metadata_observe_ms",
        "profile_model_verify_segment_graph_replay_enqueue_ms",
        "profile_model_verify_tpot_dynamic_budget_applied_count",
        "profile_model_verify_tpot_dynamic_budget_token_sum",
        "profile_model_verify_tpot_dynamic_budget_value_sum",
        "profile_model_draft_perfect_reject_events",
        "profile_model_draft_perfect_followup_events",
        "profile_model_draft_perfect_checked_tokens",
        "profile_model_draft_perfect_perfect_tokens",
        "profile_model_draft_perfect_token_rate",
        "profile_model_draft_perfect_prefix_ge1_events",
        "profile_model_draft_perfect_prefix_ge1_rate",
        "profile_model_draft_perfect_perfect_prefix_token_sum",
        "profile_model_draft_perfect_route_total",
        "profile_model_draft_perfect_route_miss",
        "profile_model_draft_perfect_route_miss_ratio",
        "profile_model_draft_perfect_coverage_total",
        "profile_model_draft_perfect_coverage_hit",
        "profile_model_draft_perfect_coverage_ratio",
        "profile_model_draft_perfect_pred_row_match_ratio",
        "profile_model_draft_perfect_input_row_match_ratio",
        "profile_model_draft_perfect_oracle_covered_tokens",
        "profile_model_draft_perfect_oracle_covered_token_rate",
        "profile_model_draft_perfect_oracle_prefix_token_sum",
        "profile_model_draft_perfect_oracle_prefix_ge1_events",
        "profile_model_draft_perfect_oracle_prefix_ge1_rate",
        "profile_model_draft_perfect_refill_events",
        "profile_model_draft_perfect_refill_promoted",
        "profile_model_draft_perfect_refill_cpu_experts",
        "profile_model_draft_perfect_refill_skipped_inflight_events",
        "profile_model_draft_perfect_refill_skipped_inflight_count",
        "profile_model_metadata_offload_ms",
        "profile_model_prefetch_wait_ms",
        "profile_json",
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
        f"- optimized config: `{metadata.get('optimized_config', 'none')}`",
        f"- optimized env: `{metadata.get('optimized_env_overrides', {})}`",
        f"- batch size: `1`",
        f"- output directory: `{metadata['output_dir']}`",
        f"- warmup prompt: `{metadata.get('warmup_prompt', '')}`",
        f"- decode driver: `{metadata.get('decode_driver', 'step')}`",
        f"- reset profile after warmup: `{metadata.get('reset_profile_after_warmup', False)}`",
        f"- reset profile before request: `{metadata.get('reset_profile_before_request', False)}`",
        f"- reset seed after warmup: `{metadata.get('reset_seed_after_warmup', False)}`",
        f"- fail on output validation error: `{metadata.get('fail_on_output_validation_error', True)}`",
        f"- profile collected: `{metadata.get('collect_profile', False)}`",
        f"- engine profile: `{metadata.get('engine_profile', False)}`",
        "",
        "## Summary",
        "",
        "| dataset | opt | alloc | seg | ratio | max out | ignore EOS | K | stop | vpb | rep | ok/sample | TPOT mean ms | P50 | P90 | P99 | decode tok/s mean | e2e tok/s mean | prompt tok mean |",
        "|:---|:---|:---|---:|---:|---:|:---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["summaries"]:
        lines.append(
            "| "
            f"{row['dataset']} | {row.get('optimized_config', 'none')} | "
            f"{row['allocation_mode']} | "
            f"{row['segment_size']} | {row['cache_ratio']:.4f} | "
            f"{'EOS' if int(row['max_output_tokens']) <= 0 else row['max_output_tokens']} | "
            f"{'true' if row.get('ignore_eos', False) else 'false'} | "
            f"{row['max_draft_tokens']} | "
            f"{row.get('draft_stop_policy', '')} | "
            f"{row.get('verify_prefetch_max_per_boundary', 0)} | "
            f"{row['repeat']} | "
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
    optimized_env_overrides = configure_optimized_env(args)
    args._optimized_env_overrides = dict(optimized_env_overrides)

    cases = build_cases(args)
    if bool(getattr(args, "dry_run", False)):
        output = {
            "metadata": {
                "timestamp": time.strftime("%Y%m%d_%H%M%S"),
                "argv": sys.argv,
                "model_path": args.model_path,
                "output_dir": str(output_dir),
                "request_mode": _effective_request_mode(args),
                "datasets": _requested_datasets(args),
                "optimized_config": str(args.optimized_config),
                "optimized_config_applied": getattr(
                    args, "_optimized_config_applied", {}
                ),
                "optimized_env_overrides": optimized_env_overrides,
                "acceptance_predictor_enabled": bool(
                    args.acceptance_predictor_enabled
                ),
                "acceptance_predictor_resolution": getattr(
                    args, "_acceptance_predictor_resolution", {}
                ),
                "tpot_definition": TPOT_DEFINITION,
                "warmup_prompt": str(args.warmup_prompt),
                "decode_driver": str(args.decode_driver),
                "reset_profile_after_warmup": bool(args.reset_profile_after_warmup),
                "reset_profile_before_request": bool(args.reset_profile_before_request),
                "reset_seed_after_warmup": bool(args.reset_seed_after_warmup),
                "fail_on_output_validation_error": bool(args.fail_on_output_validation_error),
                "collect_profile": bool(args.collect_profile),
                "engine_profile": bool(args.engine_profile),
                "engine_profile_cuda_sync": bool(args.engine_profile_cuda_sync),
                "verify_cost_model_profile": bool(args.verify_cost_model_profile),
                "case_count": len(cases),
            },
            "cases": cases,
        }
        dry_run_json = output_dir / "dry_run_summary.json"
        dry_run_json.write_text(
            json.dumps(output, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(output, ensure_ascii=True, indent=2))
        print(f"dry_run_summary_json={dry_run_json}")
        return output

    case_summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if bool(getattr(args, "reuse_engine_across_draft_lengths", False)):
        groups: dict[tuple[object, ...], list[tuple[int, dict[str, Any]]]] = {}
        for case_index, case in enumerate(cases):
            key = (
                str(case["allocation_mode"]),
                round(float(case["cache_ratio"]), 8),
                int(case["segment_size"]),
                int(case["repeat"]),
            )
            groups.setdefault(key, []).append((case_index, case))
        if len(groups) > 1:
            raise ValueError(
                "--reuse-engine-across-draft-lengths supports one cache ratio/"
                "allocation/segment/repeat per process; run separate processes "
                "for independent engine configurations"
            )
        for group_index, group in enumerate(groups.values()):
            if str(args.reuse_engine_case_order) == "shuffle":
                group = list(group)
                random.Random(int(args.seed) + group_index).shuffle(group)
            for sequence_index, (_, case) in enumerate(group):
                case["reuse_sequence_index"] = int(sequence_index)
            pending = [
                (case_index, case)
                for case_index, case in group
                if not (
                    bool(args.skip_existing)
                    and (output_dir / f"{case_name(case)}.json").exists()
                )
            ]
            if not pending:
                for case_index, case in group:
                    case_summaries.append(
                        run_case(args, case, case_index, output_dir)
                    )
                continue
            create_case = dict(pending[0][1])
            create_case["max_draft_tokens"] = max(
                int(case["max_draft_tokens"]) for _, case in group
            )
            llm = None
            active_case_index = int(pending[0][0])
            active_case = pending[0][1]
            try:
                llm = create_llm(args, create_case, group_index)
                warmup_llm(
                    llm,
                    temperature=float(args.temperature),
                    prompt=str(args.warmup_prompt),
                )
                reset_llm_profile(llm)
                for case_index, case in group:
                    active_case_index = int(case_index)
                    active_case = case
                    case_summaries.append(
                        run_case(
                            args,
                            case,
                            case_index,
                            output_dir,
                            llm=llm,
                        )
                    )
            except Exception as error:
                failure = {
                    "case": active_case,
                    "case_name": case_name(active_case),
                    "error": str(error),
                }
                failures.append(failure)
                print(
                    f"[{active_case_index + 1}] failed {failure['case_name']}: {error}",
                    flush=True,
                )
                if bool(args.fail_fast):
                    raise
            finally:
                if llm is not None:
                    llm.exit()
    else:
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
            "datasets": _requested_datasets(args),
            "reuse_engine_across_draft_lengths": bool(
                args.reuse_engine_across_draft_lengths
            ),
            "optimized_config": str(args.optimized_config),
            "optimized_config_applied": getattr(
                args, "_optimized_config_applied", {}
            ),
            "optimized_env_overrides": optimized_env_overrides,
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
            "temperature": float(args.temperature),
            "tpot_definition": TPOT_DEFINITION,
            "acceptance_strategy": str(args.acceptance_strategy),
            "acceptance_predictor_enabled": bool(
                args.acceptance_predictor_enabled
            ),
            "acceptance_predictor_resolution": getattr(
                args, "_acceptance_predictor_resolution", {}
            ),
            "acceptance_predictor_path": str(args.acceptance_predictor_path),
            "acceptance_trace_probs": bool(
                os.getenv("NANOVLLM_ACCEPTANCE_TRACE_PROBS", "").strip()
            ),
            "draft_stop_policy": str(args.draft_stop_policy),
            "draft_tpot_td_ms": float(args.draft_tpot_td_ms),
            "draft_tpot_tv_ms": float(args.draft_tpot_tv_ms),
            "draft_tpot_cost_model": str(args.draft_tpot_cost_model),
            "draft_tpot_history_alpha": float(args.draft_tpot_history_alpha),
            "draft_tpot_min_steps": int(args.draft_tpot_min_steps),
            "draft_tpot_stop_margin": float(args.draft_tpot_stop_margin),
            "draft_tpot_stop_patience": int(args.draft_tpot_stop_patience),
            "draft_tpot_lookahead_cache_credit_ms_per_step": float(
                args.draft_tpot_lookahead_cache_credit_ms_per_step
            ),
            "draft_tpot_short_verify_penalty_ms": float(args.draft_tpot_short_verify_penalty_ms),
            "draft_tpot_verify_cost_floor_ms": float(args.draft_tpot_verify_cost_floor_ms),
            "draft_tpot_stop_rule": str(args.draft_tpot_stop_rule),
            "draft_tpot_verify_model_mode": str(
                args.draft_tpot_verify_model_mode
            ),
            "draft_tpot_verify_model_path": str(
                args.draft_tpot_verify_model_path
            ),
            "draft_tpot_alpha_calibration_path": str(
                args.draft_tpot_alpha_calibration_path
            ),
            "verify_cost_model_profile": bool(args.verify_cost_model_profile),
            "verify_prefetch_max_per_boundary": int(
                args.verify_prefetch_max_per_boundary
            ),
            "verify_prefetch_tpot_dynamic_budget_enabled": bool(
                args.verify_prefetch_tpot_dynamic_budget_enabled
            ),
            "verify_prefetch_tpot_dynamic_budget_token_threshold": int(
                args.verify_prefetch_tpot_dynamic_budget_token_threshold
            ),
            "verify_prefetch_tpot_dynamic_budget_small": int(
                args.verify_prefetch_tpot_dynamic_budget_small
            ),
            "verify_prefetch_rank_multiplier": (
                int(args.verify_prefetch_rank_multiplier)
                if args.verify_prefetch_rank_multiplier is not None
                else None
            ),
            "rank_guard_threshold": float(args.rank_guard_threshold),
            "rank_guard_ema_alpha": float(args.rank_guard_ema_alpha),
            "prefetch_runtime_kind": str(args.prefetch_runtime_kind),
            "predictive_phase1_budget": int(args.predictive_phase1_budget),
            "prefetch_history_decay": float(args.prefetch_history_decay),
            "prefetch_history_ttl_steps": int(args.prefetch_history_ttl_steps),
            "prefetch_source_weight_prefill": float(args.prefetch_source_weight_prefill),
            "prefetch_source_weight_verify": float(args.prefetch_source_weight_verify),
            "prefetch_source_weight_draft": float(args.prefetch_source_weight_draft),
            "prefetch_activation_count_weight": float(args.prefetch_activation_count_weight),
            "prefetch_age_penalty": float(args.prefetch_age_penalty),
            "prefetch_use_prefill_history": bool(args.prefetch_use_prefill_history),
            "prefetch_use_verify_history": bool(args.prefetch_use_verify_history),
            "prefetch_use_draft_live": bool(args.prefetch_use_draft_live),
            "verify_cuda_graph_bucket_steps": _parse_csv(
                args.verify_cuda_graph_bucket_steps, int
            ),
            "kt_num_threads": int(args.kt_num_threads),
            "batch_size": 1,
            "warmup_prompt": str(args.warmup_prompt),
            "decode_driver": str(args.decode_driver),
            "reset_profile_after_warmup": bool(args.reset_profile_after_warmup),
            "reset_profile_before_request": bool(args.reset_profile_before_request),
            "reset_seed_after_warmup": bool(args.reset_seed_after_warmup),
            "fail_on_output_validation_error": bool(args.fail_on_output_validation_error),
            "collect_profile": bool(args.collect_profile),
            "engine_profile": bool(args.engine_profile),
            "engine_profile_cuda_sync": bool(args.engine_profile_cuda_sync),
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
    parser.add_argument(
        "--optimized-config",
        choices=OPTIMIZED_CONFIG_CHOICES,
        default="none",
        help=(
            "Apply an optimized inference preset. k4_verify uses the verified "
            "K=4 low-latency settings; k6_decode uses the validated fixed-K6 "
            "decode settings; k12_decode retains the legacy K=12 settings. "
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

    parser.add_argument("--rank-guard-threshold", type=float, default=0.15)
    parser.add_argument("--rank-guard-ema-alpha", type=float, default=0.95)
    parser.add_argument("--prefetch-step-budget", type=int, default=16)
    parser.add_argument(
        "--prefetch-runtime-kind",
        choices=["predictive", "dual_queue"],
        default="predictive",
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
    parser.add_argument("--skip-existing", type=str2bool, default=True)
    parser.add_argument("--fail-fast", type=str2bool, default=True)
    parser.add_argument("--fail-on-output-validation-error", type=str2bool, default=True)
    parser.add_argument("--save-profile-json", type=str2bool, default=False)
    parser.add_argument("--save-token-ids", type=str2bool, default=False)
    parser.add_argument("--save-text", type=str2bool, default=False)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    argv = sys.argv[1:]
    args = build_parser().parse_args(argv)
    args._optimized_config_applied = apply_optimized_config(args, argv)
    args._acceptance_predictor_resolution = resolve_acceptance_predictor(args)
    if bool(args.verify_cost_model_profile):
        args.collect_profile = True
        args.engine_profile = True
        args.engine_profile_cuda_sync = False
        args.save_profile_json = True
    if args.num_samples < 0:
        raise ValueError("--num-samples must be >= 0 or all")
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if args.repeat_index_offset < 0:
        raise ValueError("--repeat-index-offset must be >= 0")
    run(args)


if __name__ == "__main__":
    main()
