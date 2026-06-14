#!/usr/bin/env python3
"""
Two-stage optimized MoE segment unique-expert analyzer.

The target JSON/CSV results and all original command-line arguments match
test_moe_segment_unique_expert_counts.py.

Default mode:
  1. Decode each prompt once using the maximum requested decode length.
  2. After each prompt, atomically checkpoint every decode token's routed
     expert ids to an intermediate .pt file.
  3. Read that intermediate data and write the target JSON/CSV results.

If collection is interrupted, rerun the same command. Completed prompts are
loaded from the checkpoint and skipped.

Post-processing-only mode:
  Pass --intermediate_file PATH to skip model loading and inference, then
  compute the requested start-token/segment-size results from PATH.

When --intermediate_file is omitted, the intermediate filename is derived
from --output_json as "<output-json-stem>_routing_trace.pt". Pass
--intermediate_output DIR to select its save directory; otherwise it is saved
beside --output_json.

python test_moe_segment_unique_expert_counts_optimized.py \
      --model_name /data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B \
      --prompts_file /home/mumura/moe_spec/nano-vllm-moe/pre_exps/exp_and_figs/unique/diverse_prompts_en_no_output_format_list.json \
      --start_tokens -1 \
      --start_from 1 \
      --end_pos 8192 \
      --segment_sizes 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 \
      --dtype bfloat16 \
      --intermediate_output /data2/group_谈海生/mumura/nano_moe/motivaion \
      --output_json /data2/group_谈海生/mumura/nano_moe/motivaion/segment_unique_expert_counts_optimized_avg8192.json \
      --output_csv /data2/group_谈海生/mumura/nano_moe/motivaion/segment_unique_expert_counts_optimized_avg8192.csv
      
/data2/group_谈海生/mumura/nano_moe/motivaion/
 64 128 256 512 1024 2048 4096 8192
      --num_prompts 3 \
      --start_tokens -1 \
      --num_segments 10 \
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import test_moe_segment_unique_expert_counts as original


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Count per-layer unique routed experts in decode segments, with "
            "inference data collection separated from result post-processing."
        )
    )
    p.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-30B-A3B",
        help="HuggingFace model name or local path",
    )
    p.add_argument(
        "--start_tokens",
        type=int,
        nargs="+",
        default=[64, 1024],
        help=(
            "1-based decoded-token start positions n, e.g. --start_tokens "
            "64 1024. Use only -1 to enable --start_from/--end_pos range mode."
        ),
    )
    p.add_argument(
        "--start_from",
        type=int,
        default=None,
        help=(
            "Range mode only: inclusive 1-based first decoded-token position. "
            "Requires --start_tokens -1."
        ),
    )
    p.add_argument(
        "--end_pos",
        type=int,
        default=None,
        help=(
            "Range mode only: inclusive 1-based last decoded-token position. "
            "Requires --start_tokens -1."
        ),
    )
    p.add_argument(
        "--segment_sizes",
        type=int,
        nargs="+",
        default=[3, 5, 8],
        help="Segment sizes c in decoded tokens, e.g. --segment_sizes 3 5 8",
    )
    p.add_argument(
        "--num_segments",
        type=int,
        default=16,
        help="Number of c-token segments to measure for each (n, c)",
    )
    p.add_argument(
        "--num_prompts",
        type=int,
        default=None,
        help="Number of built-in prompts to use; default = all",
    )
    p.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="JSON file containing a list of prompt strings; overrides built-in prompts",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype when not using quantization",
    )
    p.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Optional: load model in 4-bit quantization",
    )
    p.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Optional: load model in 8-bit quantization",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature; 0 means greedy decoding",
    )
    p.add_argument(
        "--disable_thinking",
        action="store_true",
        help="Append /no_think and disable Qwen3 thinking mode in chat template when supported",
    )
    p.add_argument(
        "--output_json",
        type=str,
        default="segment_unique_expert_counts.json",
        help="Path for detailed JSON output",
    )
    p.add_argument(
        "--output_csv",
        type=str,
        default="segment_unique_expert_counts.csv",
        help="Path for flat per-prompt/per-layer/per-segment CSV output",
    )
    p.add_argument(
        "--intermediate_file",
        type=str,
        default=None,
        help=(
            "Existing routing-trace .pt file. When provided, skip inference and "
            "only generate target JSON/CSV results from this file."
        ),
    )
    p.add_argument(
        "--intermediate_output",
        type=str,
        default=None,
        help=(
            "Directory for saving the generated routing-trace .pt file. "
            "The filename is derived automatically from --output_json. "
            "Defaults to the --output_json directory."
        ),
    )
    return p.parse_args()


def validate_measurement_args(
    args: argparse.Namespace,
) -> tuple[list[int], list[int], int, bool]:
    start_tokens = sorted(set(args.start_tokens))
    segment_sizes = sorted(set(args.segment_sizes))
    if any(value <= 0 for value in segment_sizes):
        raise ValueError("All --segment_sizes must be positive")
    if args.num_segments <= 0:
        raise ValueError("--num_segments must be positive")

    range_mode = start_tokens == [-1]
    if -1 in start_tokens and not range_mode:
        raise ValueError(
            "--start_tokens -1 must be used alone in range mode"
        )

    if range_mode:
        if args.start_from is None or args.end_pos is None:
            raise ValueError(
                "--start_tokens -1 requires both --start_from and --end_pos"
            )
        if args.start_from <= 0:
            raise ValueError("--start_from must be a positive 1-based index")
        if args.end_pos < args.start_from:
            raise ValueError("--end_pos must be >= --start_from")
        max_decode_tokens = args.end_pos
    else:
        if any(value <= 0 for value in start_tokens):
            raise ValueError(
                "All --start_tokens must be positive 1-based indices"
            )
        if args.start_from is not None or args.end_pos is not None:
            raise ValueError(
                "--start_from and --end_pos require --start_tokens -1"
            )
        max_decode_tokens = max(
            n + c * args.num_segments - 1
            for n in start_tokens
            for c in segment_sizes
        )
    return start_tokens, segment_sizes, max_decode_tokens, range_mode


def default_intermediate_path(
    output_json: str,
    intermediate_output: str | None = None,
) -> Path:
    output_path = Path(output_json)
    stem = output_path.stem if output_path.suffix else output_path.name
    output_directory = (
        Path(intermediate_output)
        if intermediate_output is not None
        else output_path.parent
    )
    return output_directory / f"{stem}_routing_trace.pt"


def save_routing_checkpoint(
    routing_data: dict[str, Any],
    intermediate_path: Path,
) -> None:
    """Atomically replace the trace after appending one completed prompt."""
    intermediate_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = intermediate_path.with_name(
        f".{intermediate_path.name}.tmp-{os.getpid()}"
    )
    try:
        torch.save(routing_data, temporary_path)
        os.replace(temporary_path, intermediate_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def validate_resume_data(
    routing_data: dict[str, Any],
    args: argparse.Namespace,
    prompts: list[str],
    max_decode_tokens: int,
) -> None:
    config = routing_data["collection_config"]
    expected_quantization = (
        "4bit" if args.load_in_4bit
        else "8bit" if args.load_in_8bit
        else None
    )
    expected_dtype = None if expected_quantization else args.dtype
    expected = {
        "model_name": args.model_name,
        "max_decode_tokens_requested": max_decode_tokens,
        "dtype": expected_dtype,
        "quantization": expected_quantization,
        "temperature": args.temperature,
        "disable_thinking": args.disable_thinking,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}: file={actual!r}, requested={requested!r}"
            for key, (actual, requested) in mismatches.items()
        )
        raise ValueError(
            "Existing routing trace is incompatible with this collection "
            f"request ({details}). Use a different output path or remove "
            "the old trace."
        )

    completed = routing_data["per_prompt"]
    if len(completed) > len(prompts):
        raise ValueError(
            "Existing routing trace contains more prompts than requested"
        )
    for expected_idx, prompt_data in enumerate(completed):
        prompt_idx = int(prompt_data.get("prompt_index", -1))
        if prompt_idx != expected_idx:
            raise ValueError(
                "Existing routing trace prompt indices must be contiguous "
                f"from zero; expected {expected_idx}, got {prompt_idx}"
            )
        if prompt_data.get("prompt") != prompts[expected_idx]:
            raise ValueError(
                f"Existing routing trace prompt {expected_idx} does not "
                "match the current prompt list"
            )


class LayerExpertBuffer:
    """Stores routed expert ids without synchronizing GPU and CPU per token."""

    CHUNK_SIZE = 1024

    def __init__(self) -> None:
        self._device_chunks: list[torch.Tensor] = []
        self._length = 0
        self._width: int | None = None
        self._cpu_tensor: torch.Tensor | None = None

    def append(self, expert_indices: torch.Tensor) -> None:
        if self._cpu_tensor is not None:
            raise RuntimeError("Cannot append expert indices after materialization")
        expert_indices = expert_indices.detach().reshape(-1)
        width = int(expert_indices.numel())
        if self._width is None:
            self._width = width
        elif width != self._width:
            raise RuntimeError(
                f"Expert-index width changed from {self._width} to {width}"
            )

        chunk_offset = self._length % self.CHUNK_SIZE
        if chunk_offset == 0:
            self._device_chunks.append(
                torch.empty(
                    (self.CHUNK_SIZE, width),
                    dtype=expert_indices.dtype,
                    device=expert_indices.device,
                )
            )
        self._device_chunks[-1][chunk_offset].copy_(expert_indices)
        self._length += 1

    def __len__(self) -> int:
        return self._length

    def as_cpu_tensor(self) -> torch.Tensor:
        if self._cpu_tensor is None:
            if not self._device_chunks:
                self._cpu_tensor = torch.empty((0, 0), dtype=torch.int32)
            else:
                final_chunk_length = (
                    (self._length - 1) % self.CHUNK_SIZE
                ) + 1
                valid_chunks = list(self._device_chunks[:-1])
                valid_chunks.append(
                    self._device_chunks[-1][:final_chunk_length]
                )
                rows = (
                    valid_chunks[0]
                    if len(valid_chunks) == 1
                    else torch.cat(valid_chunks, dim=0)
                )
                self._cpu_tensor = rows.to(
                    device="cpu",
                    dtype=torch.int32,
                    non_blocking=False,
                )
            self._device_chunks.clear()
        return self._cpu_tensor


class BatchedExpertTracker:
    """Records routed expert ids on device and transfers once per MoE layer."""

    def __init__(self, top_k: int):
        self.top_k = top_k
        self.records: defaultdict[int, LayerExpertBuffer] = defaultdict(
            LayerExpertBuffer
        )
        self._hooks: list[Any] = []
        self.tracking = False

    def _record_indices(
        self,
        layer_idx: int,
        router_indices: torch.Tensor,
    ) -> None:
        if router_indices.dim() == 3:
            selected = router_indices.reshape(
                -1, router_indices.size(-1)
            )[-1]
        elif router_indices.dim() == 2:
            selected = router_indices[-1]
        else:
            selected = router_indices.reshape(-1)[-self.top_k :]
        self.records[layer_idx].append(selected)

    def _make_router_hook(self, layer_idx: int):
        tracker = self

        def hook_fn(module, inp, output):
            if not tracker.tracking:
                return
            if isinstance(output, (tuple, list)) and len(output) >= 3:
                tracker._record_indices(layer_idx, output[2])
                return

            logits = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(logits, torch.Tensor):
                return
            if logits.dim() == 3:
                logits = logits.reshape(-1, logits.size(-1))
            selected = torch.topk(
                logits, k=tracker.top_k, dim=-1
            ).indices[-1]
            tracker.records[layer_idx].append(selected)

        return hook_fn

    def _make_linear_gate_hook(self, layer_idx: int):
        tracker = self

        def hook_fn(module, inp, output):
            if not tracker.tracking:
                return
            logits = output
            if logits.dim() == 3:
                logits = logits.reshape(-1, logits.size(-1))
            selected = torch.topk(
                logits, k=tracker.top_k, dim=-1
            ).indices[-1]
            tracker.records[layer_idx].append(selected)

        return hook_fn

    def register(self, model) -> list[int]:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "layers"):
            layers = model.layers
        else:
            raise RuntimeError(
                "Cannot locate decoder layers: "
                "expected model.model.layers or model.layers"
            )

        moe_layer_indices: list[int] = []
        for idx, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None)
            gate = getattr(mlp, "gate", None) if mlp is not None else None
            if gate is None:
                continue

            gate_cls_name = type(gate).__name__
            if "Router" in gate_cls_name or "TopK" in gate_cls_name:
                hook = self._make_router_hook(idx)
            elif isinstance(gate, torch.nn.Linear):
                hook = self._make_linear_gate_hook(idx)
            else:
                hook = self._make_router_hook(idx)

            self._hooks.append(gate.register_forward_hook(hook))
            moe_layer_indices.append(idx)

        if not moe_layer_indices:
            raise RuntimeError("No MoE gate/router modules were found.")
        return moe_layer_indices

    def start(self) -> None:
        self.tracking = True

    def stop(self) -> None:
        self.tracking = False

    def clear(self) -> None:
        self.records = defaultdict(LayerExpertBuffer)

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


def collect_routing_data(
    args: argparse.Namespace,
    intermediate_path: Path,
    max_decode_tokens: int,
) -> dict[str, Any]:
    prompts = original.load_prompts(args)
    routing_data: dict[str, Any] | None = None
    completed_prompts = 0
    if intermediate_path.is_file():
        routing_data = load_routing_data(intermediate_path)
        validate_resume_data(
            routing_data=routing_data,
            args=args,
            prompts=prompts,
            max_decode_tokens=max_decode_tokens,
        )
        completed_prompts = len(routing_data["per_prompt"])

    print("=" * 78)
    print("  MoE Routing Data Collection")
    print("=" * 78)
    print(f"  Model          : {args.model_name}")
    print(f"  Decode tokens  : {max_decode_tokens} per prompt")
    print(f"  Num prompts    : {len(prompts)}")
    print(f"  Intermediate   : {intermediate_path}")
    print(f"  Completed      : {completed_prompts}/{len(prompts)} prompts")
    print("=" * 78)

    if completed_prompts == len(prompts):
        print("[*] Routing trace is already complete; skipping inference.")
        return routing_data

    model, tokenizer = original.load_model_and_tokenizer(args)
    config = model.config
    top_k = getattr(config, "num_experts_per_tok", None)
    if top_k is None:
        top_k = getattr(config, "top_k", 8)
    num_experts = getattr(config, "num_experts", None)
    if num_experts is None:
        num_experts = getattr(config, "num_local_experts", None)

    tracker = BatchedExpertTracker(top_k=int(top_k))
    moe_layers = tracker.register(model)
    print(f"[*] MoE config: num_experts={num_experts}, top_k={top_k}")
    print(
        f"[*] Registered hooks on {len(moe_layers)} MoE layers: "
        f"[{moe_layers[0]}..{moe_layers[-1]}]"
    )

    if routing_data is None:
        routing_data = {
            "format": "moe_routing_trace",
            "format_version": 1,
            "collection_config": {
                "model_name": args.model_name,
                "max_decode_tokens_requested": max_decode_tokens,
                "dtype": (
                    None if (args.load_in_4bit or args.load_in_8bit)
                    else args.dtype
                ),
                "quantization": (
                    "4bit" if args.load_in_4bit
                    else "8bit" if args.load_in_8bit
                    else None
                ),
                "temperature": args.temperature,
                "disable_thinking": args.disable_thinking,
                "num_experts": num_experts,
                "top_k": int(top_k),
                "moe_layer_indices": moe_layers,
                "indexing": (
                    "routing rows are 0-based decoded-token positions"
                ),
            },
            "per_prompt": [],
        }
    else:
        saved_config = routing_data["collection_config"]
        if (
            int(saved_config["top_k"]) != int(top_k)
            or [int(idx) for idx in saved_config["moe_layer_indices"]]
            != moe_layers
        ):
            raise ValueError(
                "Existing routing trace MoE metadata does not match the "
                "loaded model"
            )

    try:
        for prompt_idx in range(completed_prompts, len(prompts)):
            prompt = prompts[prompt_idx]
            prompt_short = prompt[:80].replace("\n", "\\n")
            if len(prompt) > 80:
                prompt_short += "..."
            print(
                f"\nPrompt {prompt_idx + 1}/{len(prompts)}: {prompt_short}"
            )

            tracker.clear()
            started = time.time()
            decode_info = original.decode_and_track(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                tracker=tracker,
                total_decode_tokens=max_decode_tokens,
                temperature=args.temperature,
                disable_thinking=args.disable_thinking,
            )
            elapsed = time.time() - started
            decoded_per_layer = [
                len(tracker.records[layer_idx])
                for layer_idx in moe_layers
            ]
            actual_decoded = (
                min(decoded_per_layer) if decoded_per_layer else 0
            )

            routed_experts = {
                str(layer_idx): tracker.records[
                    layer_idx
                ].as_cpu_tensor()
                for layer_idx in moe_layers
            }
            routing_data["per_prompt"].append(
                {
                    "prompt_index": prompt_idx,
                    "prompt": prompt,
                    "tokens_decoded": actual_decoded,
                    "time_seconds": round(elapsed, 3),
                    "generated_text_preview": decode_info[
                        "generated_text"
                    ][:500],
                    "generated_token_ids": torch.tensor(
                        decode_info["generated_token_ids"],
                        dtype=torch.int64,
                    ),
                    "routed_experts": routed_experts,
                }
            )
            save_routing_checkpoint(routing_data, intermediate_path)
            print(
                f"Decoded/routed tokens: {actual_decoded} "
                f"in {elapsed:.2f}s; checkpointed "
                f"{prompt_idx + 1}/{len(prompts)} prompts"
            )
    finally:
        tracker.remove_hooks()

    print(f"\n[*] Routing data complete: {intermediate_path}")
    return routing_data


def load_routing_data(intermediate_path: Path) -> dict[str, Any]:
    if not intermediate_path.is_file():
        raise FileNotFoundError(
            f"Intermediate routing file does not exist: {intermediate_path}"
        )
    try:
        data = torch.load(
            intermediate_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        data = torch.load(intermediate_path, map_location="cpu")

    if not isinstance(data, dict):
        raise ValueError("Intermediate file must contain a dictionary")
    if data.get("format") != "moe_routing_trace":
        raise ValueError(
            "Unsupported intermediate file: missing moe_routing_trace format"
        )
    if data.get("format_version") != 1:
        raise ValueError(
            f"Unsupported intermediate format version: "
            f"{data.get('format_version')!r}"
        )
    if "collection_config" not in data or "per_prompt" not in data:
        raise ValueError(
            "Intermediate file is missing collection_config or per_prompt"
        )
    return data


def unique_counts_for_tensor(
    expert_rows: torch.Tensor,
    start_token_1based: int,
    segment_size: int,
    num_segments: int,
) -> list[dict[str, Any]]:
    start0 = start_token_1based - 1
    required_tokens = start0 + segment_size * num_segments
    if required_tokens > int(expert_rows.shape[0]):
        return []

    rows = expert_rows.numpy()
    out: list[dict[str, Any]] = []
    for segment_index in range(num_segments):
        start = start0 + segment_index * segment_size
        end = start + segment_size
        segment_rows = rows[start:end]
        experts = np.unique(segment_rows)
        per_token_counts = np.apply_along_axis(
            lambda values: np.unique(values).size,
            axis=1,
            arr=segment_rows,
        )
        out.append(
            {
                "segment_index": segment_index,
                "token_start_1based": start + 1,
                "token_end_1based_inclusive": end,
                "segment_size": segment_size,
                "unique_expert_count": int(experts.size),
                "experts": [int(expert) for expert in experts],
                "per_token_expert_count_mean": float(
                    np.mean(per_token_counts)
                ),
            }
        )
    return out


def measurement_spec(
    args: argparse.Namespace,
    start_token: int,
    segment_size: int,
    range_mode: bool,
) -> tuple[int, int, int | None]:
    """Return actual start, complete segment count, and required last token."""
    if range_mode:
        interval_length = args.end_pos - args.start_from + 1
        num_segments = interval_length // segment_size
        if num_segments == 0:
            return args.start_from, 0, None
        required_tokens = (
            args.start_from + segment_size * num_segments - 1
        )
        return args.start_from, num_segments, required_tokens

    required_tokens = (
        start_token + segment_size * args.num_segments - 1
    )
    return start_token, args.num_segments, required_tokens


def postprocess_routing_data(
    args: argparse.Namespace,
    routing_data: dict[str, Any],
    start_tokens: list[int],
    segment_sizes: list[int],
    max_decode_tokens: int,
    range_mode: bool,
) -> dict[str, Any]:
    collection_config = routing_data["collection_config"]
    moe_layers = [
        int(layer_idx)
        for layer_idx in collection_config["moe_layer_indices"]
    ]
    num_experts = collection_config.get("num_experts")
    top_k = int(collection_config["top_k"])

    results: dict[str, Any] = {
        "config": {
            "model_name": collection_config["model_name"],
            "start_tokens_1based": start_tokens,
            "segment_sizes": segment_sizes,
            "num_segments_requested": args.num_segments,
            "max_decode_tokens_requested": max_decode_tokens,
            "range_mode": range_mode,
            "start_from_1based_inclusive": (
                args.start_from if range_mode else None
            ),
            "end_pos_1based_inclusive": (
                args.end_pos if range_mode else None
            ),
            "range_segment_policy": (
                "consecutive non-overlapping complete segments; trailing "
                "incomplete tokens are ignored"
                if range_mode
                else None
            ),
            "dtype": collection_config.get("dtype"),
            "quantization": collection_config.get("quantization"),
            "temperature": collection_config.get("temperature"),
            "disable_thinking": collection_config.get(
                "disable_thinking", False
            ),
            "num_experts": num_experts,
            "top_k": top_k,
            "moe_layer_indices": moe_layers,
            "indexing": "start_tokens are 1-based decoded-token indices",
            "metric": (
                "unique_expert_count = size of union of routed expert ids "
                "within a segment for one layer"
            ),
        },
        "per_prompt": [],
        "summary": {},
    }

    aggregate_counts: dict[
        tuple[int, int, int], list[int]
    ] = defaultdict(list)
    aggregate_counts_by_segment: dict[
        tuple[int, int, int, int], list[int]
    ] = defaultdict(list)
    csv_rows: list[dict[str, Any]] = []

    for prompt_data in routing_data["per_prompt"]:
        prompt_idx = int(prompt_data["prompt_index"])
        prompt = prompt_data["prompt"]
        actual_decoded = int(prompt_data["tokens_decoded"])
        routed_experts = prompt_data["routed_experts"]
        prompt_result: dict[str, Any] = {
            "prompt_index": prompt_idx,
            "prompt": prompt,
            "tokens_decoded": actual_decoded,
            "time_seconds": prompt_data["time_seconds"],
            "generated_text_preview": prompt_data[
                "generated_text_preview"
            ],
            "measurements": {},
        }

        prompt_short = prompt[:80].replace("\n", "\\n")
        if len(prompt) > 80:
            prompt_short += "..."
        print(f"\nPrompt {prompt_idx + 1}: {prompt_short}")
        print(f"Available routed tokens: {actual_decoded}")

        for n in start_tokens:
            prompt_result["measurements"].setdefault(str(n), {})
            for c in segment_sizes:
                (
                    actual_start_token,
                    requested_segments,
                    required_tokens,
                ) = measurement_spec(
                    args=args,
                    start_token=n,
                    segment_size=c,
                    range_mode=range_mode,
                )
                key_nc = f"n={n},c={c}"
                layer_results: dict[str, Any] = {}

                if requested_segments == 0:
                    print(
                        f"  Skip {key_nc}: interval "
                        f"[{args.start_from}, {args.end_pos}] contains no "
                        f"complete {c}-token segment."
                    )
                elif required_tokens > actual_decoded:
                    print(
                        f"  Skip {key_nc}: need {required_tokens} routed "
                        f"tokens, got {actual_decoded}."
                    )
                else:
                    for layer_idx in moe_layers:
                        expert_rows = routed_experts.get(str(layer_idx))
                        if expert_rows is None:
                            continue
                        segments = unique_counts_for_tensor(
                            expert_rows=expert_rows,
                            start_token_1based=actual_start_token,
                            segment_size=c,
                            num_segments=requested_segments,
                        )
                        if not segments:
                            continue
                        counts = [
                            int(segment["unique_expert_count"])
                            for segment in segments
                        ]
                        layer_results[str(layer_idx)] = {
                            "segments": segments,
                            "summary": original.summarize_values(counts),
                        }
                        if range_mode:
                            layer_results[str(layer_idx)][
                                "range_average_unique_expert_count"
                            ] = layer_results[str(layer_idx)]["summary"][
                                "mean"
                            ]

                        for segment in segments:
                            count = int(
                                segment["unique_expert_count"]
                            )
                            segment_index = int(
                                segment["segment_index"]
                            )
                            aggregate_counts[
                                (n, c, layer_idx)
                            ].append(count)
                            aggregate_counts_by_segment[
                                (n, c, layer_idx, segment_index)
                            ].append(count)
                            csv_rows.append(
                                {
                                    "prompt_index": prompt_idx,
                                    "start_token_n_1based": n,
                                    "segment_size_c": c,
                                    "layer_idx": layer_idx,
                                    "segment_index": segment_index,
                                    "token_start_1based": segment[
                                        "token_start_1based"
                                    ],
                                    "token_end_1based_inclusive": segment[
                                        "token_end_1based_inclusive"
                                    ],
                                    "unique_expert_count": count,
                                    "experts": " ".join(
                                        map(str, segment["experts"])
                                    ),
                                }
                            )

                prompt_result["measurements"][str(n)][str(c)] = {
                    "layers": layer_results,
                }
                if range_mode:
                    prompt_result["measurements"][str(n)][str(c)].update(
                        {
                            "start_from_1based_inclusive": args.start_from,
                            "end_pos_1based_inclusive": args.end_pos,
                            "num_complete_segments": requested_segments,
                        }
                    )

        results["per_prompt"].append(prompt_result)

    summary: dict[str, Any] = {}
    for n in start_tokens:
        summary.setdefault(str(n), {})
        for c in segment_sizes:
            per_layer: dict[str, Any] = {}
            all_values: list[int] = []
            for layer_idx in moe_layers:
                values = aggregate_counts.get((n, c, layer_idx), [])
                if not values:
                    continue
                all_values.extend(values)
                by_segment: dict[str, Any] = {}
                _, requested_segments, _ = measurement_spec(
                    args=args,
                    start_token=n,
                    segment_size=c,
                    range_mode=range_mode,
                )
                for segment_index in range(requested_segments):
                    segment_values = aggregate_counts_by_segment.get(
                        (n, c, layer_idx, segment_index), []
                    )
                    if segment_values:
                        by_segment[str(segment_index)] = (
                            original.summarize_values(segment_values)
                        )
                per_layer[str(layer_idx)] = {
                    "all_segments_all_prompts": (
                        original.summarize_values(values)
                    ),
                    "by_segment_index": by_segment,
                }
                if range_mode:
                    per_layer[str(layer_idx)][
                        "range_average_unique_expert_count"
                    ] = per_layer[str(layer_idx)][
                        "all_segments_all_prompts"
                    ]["mean"]
            summary[str(n)][str(c)] = {
                "per_layer": per_layer,
                "overall_across_layers_segments_prompts": (
                    original.summarize_values(all_values)
                ),
            }
    results["summary"] = summary

    with open(args.output_json, "w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2, ensure_ascii=False)
    print(f"\n[*] Wrote JSON: {args.output_json}")

    if args.output_csv:
        with open(
            args.output_csv, "w", newline="", encoding="utf-8"
        ) as output_file:
            fieldnames = [
                "prompt_index",
                "start_token_n_1based",
                "segment_size_c",
                "layer_idx",
                "segment_index",
                "token_start_1based",
                "token_end_1based_inclusive",
                "unique_expert_count",
                "experts",
            ]
            writer = csv.DictWriter(
                output_file, fieldnames=fieldnames
            )
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"[*] Wrote CSV : {args.output_csv}")

    return results


def run(args: argparse.Namespace) -> dict[str, Any]:
    start_tokens, segment_sizes, max_decode_tokens, range_mode = (
        validate_measurement_args(args)
    )

    if args.intermediate_file:
        intermediate_path = Path(args.intermediate_file)
        print(
            f"[*] Post-processing only; reading intermediate file: "
            f"{intermediate_path}"
        )
        routing_data = load_routing_data(intermediate_path)
    else:
        intermediate_path = default_intermediate_path(
            args.output_json,
            args.intermediate_output,
        )
        routing_data = collect_routing_data(
            args=args,
            intermediate_path=intermediate_path,
            max_decode_tokens=max_decode_tokens,
        )
        routing_data = load_routing_data(intermediate_path)

    return postprocess_routing_data(
        args=args,
        routing_data=routing_data,
        start_tokens=start_tokens,
        segment_sizes=segment_sizes,
        max_decode_tokens=max_decode_tokens,
        range_mode=range_mode,
    )


if __name__ == "__main__":
    run(parse_args())
