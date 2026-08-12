"""Dataset loading and prompt preparation for TPOT workloads."""
from __future__ import annotations

import argparse
import gzip
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nanovllm.benchmarks.eval_tpot.config import DATASET_CHOICES


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

@dataclass
class PromptSample:
    dataset: str
    sample_id: str
    text: str
    source_index: int
    metadata: dict[str, Any]


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
