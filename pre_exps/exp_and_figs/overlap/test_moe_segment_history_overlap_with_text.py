#!/usr/bin/env python3
"""
=========================================================================
MoE Segment-History Expert Overlap Analyzer
=========================================================================

Experiment:
  For each MoE layer, split generated decode tokens into fixed-size segments
  of c tokens. For each layer l and segment i, define:

      S[l, i] = union of routed experts activated by tokens in segment i

  For a target segment n, compute overlap/hit ratio against previous
  segments n-1, n-2, ..., n-k independently for every layer:

      hit_ratio(d) = |S[l, n] ∩ S[l, n-d]| / |S[l, n]|
      reverse_hit_ratio(d) = |S[l, n] ∩ S[l, n-d]| / |S[l, n-d]|
      jaccard(d) = |S[l, n] ∩ S[l, n-d]| / |S[l, n] ∪ S[l, n-d]|

  where d = 1..k.

Examples:
  python test_moe_segment_history_overlap_with_text.py \
  --prompts_file long_reasoning_prompts_en.json \
  --model_name /data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B \
  --segment_size 3 \
  --target_segments 8 16 32 64 128 256 512 1024 1536 2048 3072 4096 \
  --history_k 3 \
  --dtype bfloat16 \
  --save_generated_text \
  --generated_text_max_chars 50000 \
  --output segment_history_overlap_with_text_n.json

Requirements:
  pip install torch transformers accelerate bitsandbytes numpy
"""

import argparse
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
parameter count while keeping per-token FLOPs roughly constant.

A critical question for MoE inference systems is whether expert activation exhibits temporal \
locality during autoregressive decoding. If consecutive decode segments tend to activate similar \
sets of experts, then caching recently used experts in GPU memory can reduce expert loading and \
prefetch overhead. This experiment measures segment-level history locality by comparing a target \
segment's expert set with each of the previous k segments independently.\
""",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Measure per-layer MoE expert-set overlap between target segment n and previous k segments."
    )
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-30B-A3B")
    p.add_argument(
        "--segment_size", "-c", type=int, default=3,
        help="Number of decode tokens per segment, i.e. c."
    )
    p.add_argument(
        "--target_segment", type=int, default=64,
        help="1-based target segment index n. Ignored if --target_segments is provided."
    )
    p.add_argument(
        "--target_segments", type=int, nargs="+", default=None,
        help="Optional list of 1-based target segment indices, e.g. --target_segments 32 64 128."
    )
    p.add_argument(
        "--history_k", "-k", type=int, default=10,
        help="Compare target segment n with n-1, ..., n-k."
    )
    p.add_argument(
        "--extra_decode_segments", type=int, default=0,
        help="Decode extra segments after the largest target segment; not needed for the core metric."
    )
    p.add_argument("--num_prompts", type=int, default=None)
    p.add_argument("--prompts_file", type=str, default=None)
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--load_in_8bit", action="store_true")
    p.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype when not using quantization."
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--disable_thinking", action="store_true")
    p.add_argument(
        "--ignore_eos",
        action="store_true",
        help="Continue decoding even if EOS is produced. Useful for fixed-length routing measurements, but text may become degenerate."
    )
    p.add_argument(
        "--save_generated_text",
        action="store_true",
        help="Save generated text into the result JSON for repetition/debug inspection."
    )
    p.add_argument(
        "--generated_text_max_chars",
        type=int,
        default=20000,
        help="Maximum generated-text characters to save per prompt when --save_generated_text is enabled. Use -1 for full text."
    )
    p.add_argument("--output", type=str, default="segment_history_overlap_results.json")
    return p.parse_args()


class ExpertTracker:
    """
    Forward-hook tracker for MoE router/gate modules.

    records[layer_idx] is a list of sets, one set per tracked decode token.
    Each set contains the top-k expert ids selected for that token at that layer.
    """

    def __init__(self, top_k: int):
        self.top_k = top_k
        self.records: dict[int, list[set[int]]] = defaultdict(list)
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self.tracking = False

    def _make_hook_for_router(self, layer_idx: int):
        tracker = self

        def hook_fn(module, inp, output):
            if not tracker.tracking:
                return
            # Modern Qwen3 router usually returns:
            #   (router_logits, router_scores, router_indices)
            router_indices = output[2]
            experts = router_indices[-1].detach().cpu().tolist()
            tracker.records[layer_idx].append(set(int(x) for x in experts))

        return hook_fn

    def _make_hook_for_linear_gate(self, layer_idx: int):
        tracker = self

        def hook_fn(module, inp, output):
            if not tracker.tracking:
                return
            logits = output
            if logits.dim() == 3:
                logits = logits.view(-1, logits.size(-1))
            _, selected = torch.topk(logits, k=tracker.top_k, dim=-1)
            experts = selected[-1].detach().cpu().tolist()
            tracker.records[layer_idx].append(set(int(x) for x in experts))

        return hook_fn

    def register(self, model) -> list[int]:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "layers"):
            layers = model.layers
        else:
            raise RuntimeError("Cannot locate decoder layers: expected model.model.layers or model.layers")

        moe_layer_indices = []
        for idx, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue
            gate = getattr(mlp, "gate", None)
            if gate is None:
                continue

            gate_cls = type(gate).__name__
            if "Router" in gate_cls or "TopK" in gate_cls:
                hook_fn = self._make_hook_for_router(idx)
            elif isinstance(gate, torch.nn.Linear):
                hook_fn = self._make_hook_for_linear_gate(idx)
            else:
                # Best-effort fallback for custom router classes.
                hook_fn = self._make_hook_for_router(idx)

            self._hooks.append(gate.register_forward_hook(hook_fn))
            moe_layer_indices.append(idx)

        if not moe_layer_indices:
            raise RuntimeError("No MoE gate/router modules found. Check model architecture or hook path.")
        return moe_layer_indices

    def start(self):
        self.tracking = True

    def stop(self):
        self.tracking = False

    def clear(self):
        self.records = defaultdict(list)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def safe_div(num: int | float, den: int | float) -> float:
    return float(num / den) if den else 0.0


def build_segment_unions(expert_records: list[set[int]], segment_size: int) -> list[set[int]]:
    n_segments = len(expert_records) // segment_size
    segment_sets: list[set[int]] = []
    for s in range(n_segments):
        start = s * segment_size
        end = start + segment_size
        union_set: set[int] = set()
        for token_experts in expert_records[start:end]:
            union_set |= token_experts
        segment_sets.append(union_set)
    return segment_sets


def compute_generation_repetition_diagnostics(token_ids: list[int]) -> dict[str, Any]:
    """Lightweight diagnostics to help identify degenerate/repetitive long generations."""
    out: dict[str, Any] = {
        "num_generated_token_ids": len(token_ids),
    }
    if not token_ids:
        out.update({
            "unique_token_ratio": 0.0,
            "last_128_unique_token_ratio": 0.0,
            "last_512_unique_token_ratio": 0.0,
            "max_consecutive_same_token_run": 0,
        })
        return out

    out["unique_token_ratio"] = round(len(set(token_ids)) / len(token_ids), 6)

    for window in (128, 512, 2048):
        tail = token_ids[-window:]
        out[f"last_{window}_unique_token_ratio"] = round(len(set(tail)) / len(tail), 6)

    max_run = 1
    cur_run = 1
    for i in range(1, len(token_ids)):
        if token_ids[i] == token_ids[i - 1]:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 1
    out["max_consecutive_same_token_run"] = max_run

    # Repeated n-gram fraction over token ids. High values in the tail are a useful warning.
    for n in (2, 3, 4, 8, 16):
        total = max(0, len(token_ids) - n + 1)
        if total == 0:
            out[f"repeated_{n}gram_fraction"] = 0.0
            out[f"last_512_repeated_{n}gram_fraction"] = 0.0
            continue
        grams = [tuple(token_ids[i:i+n]) for i in range(total)]
        out[f"repeated_{n}gram_fraction"] = round(1.0 - len(set(grams)) / total, 6)

        tail = token_ids[-512:]
        tail_total = max(0, len(tail) - n + 1)
        if tail_total == 0:
            out[f"last_512_repeated_{n}gram_fraction"] = 0.0
        else:
            tail_grams = [tuple(tail[i:i+n]) for i in range(tail_total)]
            out[f"last_512_repeated_{n}gram_fraction"] = round(1.0 - len(set(tail_grams)) / tail_total, 6)

    return out


def compute_target_history_metrics(
    expert_records: list[set[int]],
    segment_size: int,
    target_segment_1based: int,
    history_k: int,
) -> dict[str, Any]:
    """Compute metrics for target segment n against n-1...n-k.

    target_segment_1based follows the user-facing convention: segment 1 is the
    first c decoded tokens, segment 2 is the next c decoded tokens, etc.
    """
    segment_sets = build_segment_unions(expert_records, segment_size)
    n_segments = len(segment_sets)

    if target_segment_1based < 1:
        return {"error": "target_segment must be >= 1"}
    if target_segment_1based > n_segments:
        return {
            "error": "target_segment exceeds available decoded segments",
            "available_segments": n_segments,
        }

    target_idx = target_segment_1based - 1
    target_set = segment_sets[target_idx]
    max_distance = min(history_k, target_idx)

    by_distance = {}
    for d in range(1, max_distance + 1):
        prev_idx = target_idx - d
        prev_set = segment_sets[prev_idx]
        inter = target_set & prev_set
        union = target_set | prev_set

        by_distance[str(d)] = {
            "previous_segment": prev_idx + 1,
            "target_segment": target_segment_1based,
            "intersection_size": len(inter),
            "target_expert_count": len(target_set),
            "previous_expert_count": len(prev_set),
            "union_size": len(union),
            # Primary metric: fraction of target segment demand covered by old segment.
            "hit_ratio": safe_div(len(inter), len(target_set)),
            # Useful when asking whether the old segment is mostly reused by the target.
            "reverse_hit_ratio": safe_div(len(inter), len(prev_set)),
            "jaccard": safe_div(len(inter), len(union)),
            "overlap_coefficient": safe_div(len(inter), min(len(target_set), len(prev_set))),
            "target_experts": sorted(target_set),
            "previous_experts": sorted(prev_set),
            "intersection_experts": sorted(inter),
        }

    return {
        "segment_size": segment_size,
        "target_segment": target_segment_1based,
        "history_k_requested": history_k,
        "history_k_available": max_distance,
        "num_segments_available": n_segments,
        "target_expert_count": len(target_set),
        "target_experts": sorted(target_set),
        "by_distance": by_distance,
    }


def summarize_across_prompts(values_by_layer_distance: dict[int, dict[int, list[float]]]) -> dict[str, Any]:
    summary = {}
    for layer_idx, dist_map in sorted(values_by_layer_distance.items()):
        layer_summary = {}
        for d, vals in sorted(dist_map.items()):
            if vals:
                layer_summary[str(d)] = {
                    "mean": round(float(np.mean(vals)), 6),
                    "std": round(float(np.std(vals)), 6),
                    "min": round(float(np.min(vals)), 6),
                    "max": round(float(np.max(vals)), 6),
                    "num_samples": len(vals),
                }
        summary[str(layer_idx)] = layer_summary
    return summary


def load_model_and_tokenizer(args):
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

    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        print("[*] Loading model in 4-bit quantization")
    elif args.load_in_8bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        print("[*] Loading model in 8-bit quantization")
    else:
        load_kwargs["torch_dtype"] = dtype_map[args.dtype]
        print(f"[*] Loading model in {args.dtype}")

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
    temperature: float = 0.0,
    disable_thinking: bool = False,
    ignore_eos: bool = False,
) -> dict[str, Any]:
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

    device = next(model.parameters()).device
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    # Prefill: do not track prompt-routing experts.
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
    stopped_by_eos = False

    # Decode: track exactly one router decision per generated decode token.
    tracker.start()
    for _ in range(total_decode_tokens):
        outputs = model(next_token_id, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values

        generated_ids.append(int(next_token_id.item()))

        if temperature <= 0:
            next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(outputs.logits[:, -1, :] / temperature, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

        if (
            eos_token_id is not None
            and int(next_token_id.item()) == eos_token_id
            and not ignore_eos
        ):
            # Continue is sometimes useful for fixed-length measurements, but by default
            # stop to avoid routing on padding/degenerate EOS continuation.
            stopped_by_eos = True
            break

    tracker.stop()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return {
        "generated_text": generated_text,
        "generated_token_ids": generated_ids,
        "num_generated_token_ids": len(generated_ids),
        "stopped_by_eos": stopped_by_eos,
        "eos_token_id": eos_token_id,
        "repetition_diagnostics": compute_generation_repetition_diagnostics(generated_ids),
    }


def run_analysis(args):
    assert args.segment_size > 0, "segment_size must be > 0"
    assert args.history_k > 0, "history_k must be > 0"

    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            prompts = json.load(f)
        assert isinstance(prompts, list) and all(isinstance(x, str) for x in prompts), \
            "prompts_file must contain a JSON list of strings"
    else:
        prompts = DEFAULT_PROMPTS

    if args.num_prompts is not None:
        prompts = prompts[: args.num_prompts]

    target_segments = args.target_segments or [args.target_segment]
    target_segments = sorted(set(target_segments))
    assert all(n > 0 for n in target_segments), "all target segments must be >= 1"

    max_target = max(target_segments)
    total_decode_tokens = args.segment_size * (max_target + args.extra_decode_segments)

    print("=" * 72)
    print("  MoE Segment-History Expert Overlap Analyzer")
    print("=" * 72)
    print(f"  Model             : {args.model_name}")
    print(f"  Segment size c    : {args.segment_size} decode tokens")
    print(f"  Target segments n : {target_segments} (1-based)")
    print(f"  History k         : {args.history_k}")
    print(f"  Decode tokens     : {total_decode_tokens} per prompt")
    print(f"  Num prompts       : {len(prompts)}")
    print(f"  Temperature       : {args.temperature}")
    print("=" * 72)

    model, tokenizer = load_model_and_tokenizer(args)

    config = model.config
    top_k = getattr(config, "num_experts_per_tok", None)
    if top_k is None:
        top_k = getattr(config, "top_k", 8)
    num_experts = getattr(config, "num_experts", None)
    if num_experts is None:
        num_experts = getattr(config, "num_local_experts", 128)

    tracker = ExpertTracker(top_k=int(top_k))
    moe_layers = tracker.register(model)
    print(f"[*] MoE config: {num_experts} experts, top-{top_k} routing")
    print(f"[*] Registered hooks on {len(moe_layers)} MoE layers: [{moe_layers[0]}..{moe_layers[-1]}]")

    results: dict[str, Any] = {
        "config": {
            "model": args.model_name,
            "num_experts": num_experts,
            "top_k": top_k,
            "moe_layer_indices": moe_layers,
            "num_moe_layers": len(moe_layers),
            "segment_size": args.segment_size,
            "target_segments": target_segments,
            "history_k": args.history_k,
            "total_decode_tokens": total_decode_tokens,
            "num_prompts": len(prompts),
            "temperature": args.temperature,
            "ignore_eos": args.ignore_eos,
            "save_generated_text": args.save_generated_text,
            "generated_text_max_chars": args.generated_text_max_chars,
            "metric_definitions": {
                "hit_ratio": "|S_target ∩ S_previous| / |S_target|",
                "reverse_hit_ratio": "|S_target ∩ S_previous| / |S_previous|",
                "jaccard": "|S_target ∩ S_previous| / |S_target ∪ S_previous|",
                "overlap_coefficient": "|S_target ∩ S_previous| / min(|S_target|, |S_previous|)",
            },
        },
        "per_prompt": [],
    }

    # target_n -> layer_idx -> distance -> list[metric]
    agg_hit: dict[int, dict[int, dict[int, list[float]]]] = {
        n: defaultdict(lambda: defaultdict(list)) for n in target_segments
    }
    agg_jaccard: dict[int, dict[int, dict[int, list[float]]]] = {
        n: defaultdict(lambda: defaultdict(list)) for n in target_segments
    }

    try:
        for pi, prompt in enumerate(prompts):
            prompt_short = prompt[:80].replace("\n", "\\n") + ("..." if len(prompt) > 80 else "")
            print(f"\n{'─' * 72}")
            print(f"  Prompt {pi + 1}/{len(prompts)}: \"{prompt_short}\"")
            print(f"{'─' * 72}")

            tracker.clear()
            t0 = time.time()
            generation_info = decode_and_track(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                tracker=tracker,
                total_decode_tokens=total_decode_tokens,
                temperature=args.temperature,
                disable_thinking=args.disable_thinking,
                ignore_eos=args.ignore_eos,
            )
            generated_text = generation_info["generated_text"]
            elapsed = time.time() - t0
            actual_tokens = len(next(iter(tracker.records.values()))) if tracker.records else 0
            actual_segments = actual_tokens // args.segment_size
            print(f"  Decoded/tracked {actual_tokens} tokens = {actual_segments} full segments in {elapsed:.1f}s")

            text_limit = args.generated_text_max_chars
            if args.save_generated_text:
                if text_limit is None or text_limit < 0:
                    saved_generated_text = generated_text
                    generated_text_truncated = False
                else:
                    saved_generated_text = generated_text[:text_limit]
                    generated_text_truncated = len(generated_text) > text_limit
            else:
                saved_generated_text = None
                generated_text_truncated = False

            prompt_result: dict[str, Any] = {
                "prompt_index": pi,
                "prompt": prompt,
                "generated_preview": generated_text[:500],
                "generated_text": saved_generated_text,
                "generated_text_saved": bool(args.save_generated_text),
                "generated_text_truncated": generated_text_truncated,
                "generated_text_num_chars": len(generated_text),
                "generation_stopped_by_eos": generation_info["stopped_by_eos"],
                "generation_eos_token_id": generation_info["eos_token_id"],
                "generation_repetition_diagnostics": generation_info["repetition_diagnostics"],
                "tokens_tracked": actual_tokens,
                "segments_available": actual_segments,
                "time_seconds": round(elapsed, 2),
                "targets": {},
            }

            for target_n in target_segments:
                target_result: dict[str, Any] = {"per_layer": {}}
                if actual_segments < target_n:
                    target_result["error"] = (
                        f"Only {actual_segments} full segments decoded; target segment {target_n} unavailable."
                    )
                    prompt_result["targets"][str(target_n)] = target_result
                    print(f"  ⚠ target n={target_n} unavailable: only {actual_segments} segments")
                    continue

                print(f"\n  ── Target segment n={target_n}; compare to n-1..n-{args.history_k} ──")
                representative = [
                    moe_layers[0],
                    moe_layers[len(moe_layers) // 4],
                    moe_layers[len(moe_layers) // 2],
                    moe_layers[(3 * len(moe_layers)) // 4],
                    moe_layers[-1],
                ]
                representative = sorted(set(representative))

                for layer_idx in moe_layers:
                    metrics = compute_target_history_metrics(
                        expert_records=tracker.records[layer_idx],
                        segment_size=args.segment_size,
                        target_segment_1based=target_n,
                        history_k=args.history_k,
                    )
                    target_result["per_layer"][str(layer_idx)] = metrics
                    if "error" in metrics:
                        continue
                    for d_str, m in metrics["by_distance"].items():
                        d = int(d_str)
                        agg_hit[target_n][layer_idx][d].append(m["hit_ratio"])
                        agg_jaccard[target_n][layer_idx][d].append(m["jaccard"])

                print(f"  {'Layer':>8} │ {'d=1 hit':>8} │ {'d=2 hit':>8} │ {'d=5 hit':>8} │ {'d=10 hit':>9} │ {'|S_n|':>6}")
                print(f"  {'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*9}─┼─{'─'*6}")
                for li in representative:
                    m_layer = target_result["per_layer"].get(str(li), {})
                    by_d = m_layer.get("by_distance", {})
                    def fmt(d: int) -> str:
                        return f"{by_d[str(d)]['hit_ratio']:.4f}" if str(d) in by_d else "   n/a  "
                    sn = m_layer.get("target_expert_count", 0)
                    print(f"  {li:>8} │ {fmt(1):>8} │ {fmt(2):>8} │ {fmt(5):>8} │ {fmt(10):>9} │ {sn:>6}")

                prompt_result["targets"][str(target_n)] = target_result

            results["per_prompt"].append(prompt_result)

        global_summary: dict[str, Any] = {}
        print(f"\n{'=' * 72}")
        print("  GLOBAL SUMMARY: hit_ratio = |S_n ∩ S_{n-d}| / |S_n|")
        print(f"{'=' * 72}")

        for target_n in target_segments:
            global_summary[str(target_n)] = {
                "hit_ratio_by_layer_distance": summarize_across_prompts(agg_hit[target_n]),
                "jaccard_by_layer_distance": summarize_across_prompts(agg_jaccard[target_n]),
                "overall_by_distance": {},
            }

            print(f"\n  ══ Target segment n={target_n} ══")
            print(f"  {'Distance d':>10} │ {'Hit μ':>8} │ {'Hit σ':>8} │ {'Jaccard μ':>9} │ {'Samples':>7}")
            print(f"  {'─'*10}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*9}─┼─{'─'*7}")

            for d in range(1, args.history_k + 1):
                hit_vals = []
                jac_vals = []
                for li in moe_layers:
                    hit_vals.extend(agg_hit[target_n][li].get(d, []))
                    jac_vals.extend(agg_jaccard[target_n][li].get(d, []))
                if not hit_vals:
                    continue
                d_summary = {
                    "hit_mean": round(float(np.mean(hit_vals)), 6),
                    "hit_std": round(float(np.std(hit_vals)), 6),
                    "jaccard_mean": round(float(np.mean(jac_vals)), 6),
                    "jaccard_std": round(float(np.std(jac_vals)), 6),
                    "num_samples": len(hit_vals),
                }
                global_summary[str(target_n)]["overall_by_distance"][str(d)] = d_summary
                print(
                    f"  {d:>10} │ {d_summary['hit_mean']:>8.4f} │ "
                    f"{d_summary['hit_std']:>8.4f} │ {d_summary['jaccard_mean']:>9.4f} │ "
                    f"{d_summary['num_samples']:>7}"
                )

        results["global_summary"] = global_summary

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[*] Detailed results saved to: {args.output}")

    finally:
        tracker.remove_hooks()

    return results


if __name__ == "__main__":
    run_analysis(parse_args())
