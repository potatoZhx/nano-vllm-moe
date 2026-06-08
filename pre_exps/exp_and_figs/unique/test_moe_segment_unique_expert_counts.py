#!/usr/bin/env python3
"""
==========================================================================
MoE Segment Unique Expert Count Analyzer
==========================================================================

Experiment:
  For each prompt, run autoregressive decoding and record routed experts at
  every MoE layer for each decoded token. For each configured start token n
  and segment size c, treat token n as the beginning of segment 0, then split
  the decode stream into consecutive c-token segments:

      segment 0: tokens [n, n+c)
      segment 1: tokens [n+c, n+2c)
      ...

  For every MoE layer and every segment, compute the number of unique routed
  experts in that segment, i.e. the size of the union of routed experts across
  the c decoded tokens in that layer.

Indexing convention:
  --start_tokens are 1-based decode-token indices.
  Example: --start_tokens 64 means "start from the 64th decoded token".

Primary output:
  unique_expert_count = | union_{token in segment} experts(layer, token) |

Usage examples:
  # No quantization, BF16 weights
  python test_moe_segment_unique_expert_counts.py \
     --model_name /workspace/wyt/ktransformers1/Qwen--Qwen3-30B-A3B-Base \
      --start_tokens 64 1024 \
      --segment_sizes 3 5 8 \
      --num_segments 16 \
      --dtype bfloat16 \
      --output_json segment_unique_expert_counts.json

  # Quick smoke test
  python test_moe_segment_unique_expert_counts.py \
      --start_tokens 64 \
      --segment_sizes 3 5 8 \
      --num_segments 4 \
      --num_prompts 1 \
      --dtype bfloat16

Requirements:
  pip install torch transformers accelerate numpy
  Optional for quantization: pip install bitsandbytes
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_PROMPTS = [
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n",
    "Let's solve this step by step. If a train travels at 120 km/h and another train",
    "Please explain the difference between supervised learning and reinforcement learning in detail.",
    """\
Mixture-of-Experts (MoE) models have become a dominant architecture for scaling language models \
beyond the compute budget of dense transformers. Instead of activating every parameter for every \
input token, an MoE layer routes each token to a small subset of expert sub-networks via a learned \
gating function. This conditional computation allows MoE models to dramatically increase total \
parameter count while keeping per-token FLOPs roughly constant. During autoregressive decoding, \
one useful systems question is how many distinct experts are needed over short consecutive decode \
segments, and how this number changes as the starting position and segment size vary.\n""",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Count per-layer unique routed experts in decode segments starting "
            "from specified token positions."
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
        help="1-based decoded-token start positions n, e.g. --start_tokens 64 1024",
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
    return p.parse_args()


class ExpertTracker:
    """
    Records selected expert ids for each decoded token at each MoE layer.

    records[layer_idx][token_idx] is a set[int] of selected experts for one
    decoded token. token_idx is 0-based internally.
    """

    def __init__(self, top_k: int):
        self.top_k = top_k
        self.records: dict[int, list[set[int]]] = defaultdict(list)
        self._hooks: list[Any] = []
        self.tracking = False

    def _record_indices(self, layer_idx: int, router_indices: torch.Tensor) -> None:
        # Expected shapes: (seq_len, top_k), (batch, seq_len, top_k), or flattened.
        if router_indices.dim() == 3:
            selected = router_indices.reshape(-1, router_indices.size(-1))[-1]
        elif router_indices.dim() == 2:
            selected = router_indices[-1]
        else:
            selected = router_indices.reshape(-1)[-self.top_k :]
        self.records[layer_idx].append(set(int(x) for x in selected.detach().cpu().tolist()))

    def _make_router_hook(self, layer_idx: int):
        tracker = self

        def hook_fn(module, inp, output):
            if not tracker.tracking:
                return
            if isinstance(output, (tuple, list)) and len(output) >= 3:
                router_indices = output[2]
                tracker._record_indices(layer_idx, router_indices)
                return
            # Some implementations may return logits directly despite router-like naming.
            logits = output[0] if isinstance(output, (tuple, list)) else output
            if isinstance(logits, torch.Tensor):
                if logits.dim() == 3:
                    logits = logits.reshape(-1, logits.size(-1))
                _, selected = torch.topk(logits, k=tracker.top_k, dim=-1)
                tracker.records[layer_idx].append(
                    set(int(x) for x in selected[-1].detach().cpu().tolist())
                )

        return hook_fn

    def _make_linear_gate_hook(self, layer_idx: int):
        tracker = self

        def hook_fn(module, inp, output):
            if not tracker.tracking:
                return
            logits = output
            if logits.dim() == 3:
                logits = logits.reshape(-1, logits.size(-1))
            _, selected = torch.topk(logits, k=tracker.top_k, dim=-1)
            tracker.records[layer_idx].append(
                set(int(x) for x in selected[-1].detach().cpu().tolist())
            )

        return hook_fn

    def register(self, model) -> list[int]:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "layers"):
            layers = model.layers
        else:
            raise RuntimeError("Cannot locate decoder layers: expected model.model.layers or model.layers")

        moe_layer_indices: list[int] = []
        for idx, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue
            gate = getattr(mlp, "gate", None)
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
        self.records = defaultdict(list)

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            prompts = json.load(f)
        if not isinstance(prompts, list) or not all(isinstance(x, str) for x in prompts):
            raise ValueError("--prompts_file must contain a JSON list of strings")
    else:
        prompts = DEFAULT_PROMPTS
    if args.num_prompts is not None:
        prompts = prompts[: args.num_prompts]
    if not prompts:
        raise ValueError("No prompts to run")
    return prompts


def load_model_and_tokenizer(args: argparse.Namespace):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[*] Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": "auto",
    }

    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Choose at most one of --load_in_4bit and --load_in_8bit")

    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        print("[*] Loading model in 4-bit quantization...")
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif args.load_in_8bit:
        from transformers import BitsAndBytesConfig

        print("[*] Loading model in 8-bit quantization...")
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        print(f"[*] Loading model in {args.dtype} without quantization...")
        load_kwargs["torch_dtype"] = dtype_map[args.dtype]

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)
    model.eval()
    print(f"[*] Model loaded: {type(model).__name__}")
    return model, tokenizer


@torch.no_grad()
def decode_and_track(
    model,
    tokenizer,
    prompt: str,
    tracker: ExpertTracker,
    total_decode_tokens: int,
    temperature: float,
    disable_thinking: bool,
) -> dict[str, Any]:
    """Decode up to total_decode_tokens and track expert routing only during decode."""
    prompt_text = prompt + " /no_think" if disable_thinking else prompt

    try:
        messages = [{"role": "user", "content": prompt_text}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=not disable_thinking,
        )
    except Exception:
        text = prompt_text

    input_ids = tokenizer(text, return_tensors="pt").input_ids
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    # Prefill: do not track prompt-token routing.
    tracker.stop()
    outputs = model(input_ids, use_cache=True)
    past_key_values = outputs.past_key_values

    if temperature <= 0:
        next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    else:
        probs = F.softmax(outputs.logits[:, -1, :] / temperature, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)

    generated_ids: list[int] = []
    eos_token_id = tokenizer.eos_token_id

    tracker.start()
    for _ in range(total_decode_tokens):
        # This forward pass routes the previously selected decode token.
        routed_token_id = int(next_token_id.item())
        generated_ids.append(routed_token_id)

        outputs = model(next_token_id, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values

        if temperature <= 0:
            next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(outputs.logits[:, -1, :] / temperature, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

        if eos_token_id is not None and routed_token_id == eos_token_id:
            break

    tracker.stop()

    return {
        "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "generated_token_ids": generated_ids,
    }


def unique_counts_for_layer(
    token_expert_sets: list[set[int]],
    start_token_1based: int,
    segment_size: int,
    num_segments: int,
) -> list[dict[str, Any]]:
    """Return per-segment unique expert counts for one layer."""
    if start_token_1based <= 0:
        raise ValueError("start_token_1based must be >= 1")
    start0 = start_token_1based - 1
    out: list[dict[str, Any]] = []

    for seg_idx in range(num_segments):
        s = start0 + seg_idx * segment_size
        e = s + segment_size
        if e > len(token_expert_sets):
            break
        union_set: set[int] = set()
        token_counts: list[int] = []
        for token_set in token_expert_sets[s:e]:
            union_set |= token_set
            token_counts.append(len(token_set))
        out.append(
            {
                "segment_index": seg_idx,
                "token_start_1based": s + 1,
                "token_end_1based_inclusive": e,
                "segment_size": segment_size,
                "unique_expert_count": len(union_set),
                "experts": sorted(union_set),
                "per_token_expert_count_mean": float(np.mean(token_counts)) if token_counts else 0.0,
            }
        )
    return out


def summarize_values(vals: list[float | int]) -> dict[str, Any]:
    if not vals:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "median": None,
        }
    arr = np.asarray(vals, dtype=float)
    return {
        "count": int(arr.size),
        "mean": round(float(np.mean(arr)), 6),
        "std": round(float(np.std(arr)), 6),
        "min": round(float(np.min(arr)), 6),
        "max": round(float(np.max(arr)), 6),
        "median": round(float(np.median(arr)), 6),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    start_tokens = sorted(set(args.start_tokens))
    segment_sizes = sorted(set(args.segment_sizes))
    if any(x <= 0 for x in start_tokens):
        raise ValueError("All --start_tokens must be positive 1-based indices")
    if any(x <= 0 for x in segment_sizes):
        raise ValueError("All --segment_sizes must be positive")
    if args.num_segments <= 0:
        raise ValueError("--num_segments must be positive")

    prompts = load_prompts(args)
    max_decode_tokens = max(n + c * args.num_segments - 1 for n in start_tokens for c in segment_sizes)

    print("=" * 78)
    print("  MoE Segment Unique Expert Count Analyzer")
    print("=" * 78)
    print(f"  Model          : {args.model_name}")
    print(f"  Start tokens n : {start_tokens}  (1-based decode-token indices)")
    print(f"  Segment sizes c: {segment_sizes}")
    print(f"  Num segments   : {args.num_segments}")
    print(f"  Decode tokens  : {max_decode_tokens} per prompt")
    print(f"  Num prompts    : {len(prompts)}")
    print(f"  Quantization   : {'4-bit' if args.load_in_4bit else '8-bit' if args.load_in_8bit else 'none'}")
    print("=" * 78)

    model, tokenizer = load_model_and_tokenizer(args)

    config = model.config
    top_k = getattr(config, "num_experts_per_tok", None)
    if top_k is None:
        top_k = getattr(config, "top_k", 8)
    num_experts = getattr(config, "num_experts", None)
    if num_experts is None:
        num_experts = getattr(config, "num_local_experts", None)

    tracker = ExpertTracker(top_k=int(top_k))
    moe_layers = tracker.register(model)
    print(f"[*] MoE config: num_experts={num_experts}, top_k={top_k}")
    print(f"[*] Registered hooks on {len(moe_layers)} MoE layers: [{moe_layers[0]}..{moe_layers[-1]}]")

    results: dict[str, Any] = {
        "config": {
            "model_name": args.model_name,
            "start_tokens_1based": start_tokens,
            "segment_sizes": segment_sizes,
            "num_segments_requested": args.num_segments,
            "max_decode_tokens_requested": max_decode_tokens,
            "dtype": None if (args.load_in_4bit or args.load_in_8bit) else args.dtype,
            "quantization": "4bit" if args.load_in_4bit else "8bit" if args.load_in_8bit else None,
            "temperature": args.temperature,
            "disable_thinking": args.disable_thinking,
            "num_experts": num_experts,
            "top_k": int(top_k),
            "moe_layer_indices": moe_layers,
            "indexing": "start_tokens are 1-based decoded-token indices",
            "metric": "unique_expert_count = size of union of routed expert ids within a segment for one layer",
        },
        "per_prompt": [],
        "summary": {},
    }

    # Aggregation key: (n, c, layer) -> counts across prompts and segments.
    aggregate_counts: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    # Aggregation key: (n, c, layer, segment_index) -> counts across prompts.
    aggregate_counts_by_segment: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    csv_rows: list[dict[str, Any]] = []

    for prompt_idx, prompt in enumerate(prompts):
        prompt_short = prompt[:80].replace("\n", "\\n") + ("..." if len(prompt) > 80 else "")
        print(f"\n{'─' * 78}")
        print(f"Prompt {prompt_idx + 1}/{len(prompts)}: {prompt_short}")
        print(f"{'─' * 78}")

        tracker.clear()
        t0 = time.time()
        decode_info = decode_and_track(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            tracker=tracker,
            total_decode_tokens=max_decode_tokens,
            temperature=args.temperature,
            disable_thinking=args.disable_thinking,
        )
        elapsed = time.time() - t0
        decoded_per_layer = [len(tracker.records[li]) for li in moe_layers]
        actual_decoded = min(decoded_per_layer) if decoded_per_layer else 0
        print(f"Decoded/routed tokens: {actual_decoded} in {elapsed:.2f}s")

        prompt_result: dict[str, Any] = {
            "prompt_index": prompt_idx,
            "prompt": prompt,
            "tokens_decoded": actual_decoded,
            "time_seconds": round(elapsed, 3),
            "generated_text_preview": decode_info["generated_text"][:500],
            "measurements": {},
        }

        for n in start_tokens:
            prompt_result["measurements"].setdefault(str(n), {})
            for c in segment_sizes:
                key_nc = f"n={n},c={c}"
                if n + c * args.num_segments - 1 > actual_decoded:
                    print(
                        f"  ⚠ Skip {key_nc}: need {n + c * args.num_segments - 1} routed tokens, "
                        f"got {actual_decoded}."
                    )
                layer_results: dict[str, Any] = {}
                for layer_idx in moe_layers:
                    segs = unique_counts_for_layer(
                        tracker.records[layer_idx],
                        start_token_1based=n,
                        segment_size=c,
                        num_segments=args.num_segments,
                    )
                    if not segs:
                        continue
                    counts = [int(seg["unique_expert_count"]) for seg in segs]
                    layer_results[str(layer_idx)] = {
                        "segments": segs,
                        "summary": summarize_values(counts),
                    }

                    for seg in segs:
                        count = int(seg["unique_expert_count"])
                        seg_idx = int(seg["segment_index"])
                        aggregate_counts[(n, c, layer_idx)].append(count)
                        aggregate_counts_by_segment[(n, c, layer_idx, seg_idx)].append(count)
                        csv_rows.append(
                            {
                                "prompt_index": prompt_idx,
                                "start_token_n_1based": n,
                                "segment_size_c": c,
                                "layer_idx": layer_idx,
                                "segment_index": seg_idx,
                                "token_start_1based": seg["token_start_1based"],
                                "token_end_1based_inclusive": seg["token_end_1based_inclusive"],
                                "unique_expert_count": count,
                                "experts": " ".join(map(str, seg["experts"])),
                            }
                        )

                prompt_result["measurements"][str(n)][str(c)] = {
                    "layers": layer_results,
                }

                # Print compact summary using representative layers.
                reps = [
                    moe_layers[0],
                    moe_layers[len(moe_layers) // 4],
                    moe_layers[len(moe_layers) // 2],
                    moe_layers[(3 * len(moe_layers)) // 4],
                    moe_layers[-1],
                ]
                reps = sorted(set(reps))
                print(f"\n  {key_nc}: unique expert count per segment, mean across measured segments")
                print(f"  {'Layer':>8} │ {'mean':>8} │ {'std':>8} │ {'min':>6} │ {'max':>6} │ {'segments':>8}")
                print(f"  {'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*8}")
                for li in reps:
                    lr = layer_results.get(str(li))
                    if not lr:
                        continue
                    s = lr["summary"]
                    print(
                        f"  {li:>8} │ {s['mean']:>8.3f} │ {s['std']:>8.3f} │ "
                        f"{s['min']:>6.0f} │ {s['max']:>6.0f} │ {s['count']:>8}"
                    )

        results["per_prompt"].append(prompt_result)

    # Build global summary.
    summary: dict[str, Any] = {}
    for n in start_tokens:
        summary.setdefault(str(n), {})
        for c in segment_sizes:
            per_layer: dict[str, Any] = {}
            all_vals: list[int] = []
            for li in moe_layers:
                vals = aggregate_counts.get((n, c, li), [])
                if not vals:
                    continue
                all_vals.extend(vals)
                by_segment: dict[str, Any] = {}
                for seg_idx in range(args.num_segments):
                    seg_vals = aggregate_counts_by_segment.get((n, c, li, seg_idx), [])
                    if seg_vals:
                        by_segment[str(seg_idx)] = summarize_values(seg_vals)
                per_layer[str(li)] = {
                    "all_segments_all_prompts": summarize_values(vals),
                    "by_segment_index": by_segment,
                }
            summary[str(n)][str(c)] = {
                "per_layer": per_layer,
                "overall_across_layers_segments_prompts": summarize_values(all_vals),
            }
    results["summary"] = summary

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[*] Wrote JSON: {args.output_json}")

    if args.output_csv:
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
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
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"[*] Wrote CSV : {args.output_csv}")

    print("\n" + "=" * 78)
    print("Interpretation")
    print("=" * 78)
    print(
        "unique_expert_count is the number of distinct experts used by one layer "
        "inside one c-token segment. For Qwen-style top-k routing, the theoretical "
        "range is roughly [top_k, min(num_experts, c * top_k)] per layer, unless EOS "
        "or implementation details reduce the measured tokens."
    )

    tracker.remove_hooks()
    return results


if __name__ == "__main__":
    run(parse_args())
