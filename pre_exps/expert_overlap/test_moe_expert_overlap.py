#!/usr/bin/env python3
"""
==========================================================================
MoE Expert Activation Overlap Analyzer for Qwen3-30B-A3B
==========================================================================

Tests whether adjacent decode token segments activate similar experts in
a Mixture-of-Experts model. For each request:
  1. Prefill the prompt (experts not tracked)
  2. Decode N tokens per segment, record activated experts per layer
  3. Repeat for multiple segments
  4. Compute overlap (Jaccard, intersection size, etc.) between adjacent
     segments at each layer
  5. Aggregate across multiple prompts

Architecture reference (Qwen3-30B-A3B):
  - 48 decoder layers (all MoE by default)
  - 128 routed experts per MoE layer
  - top-8 routing (8 experts activated per token)
  - ~30.5B total params, ~3.3B active per token

Usage:
  python test_moe_expert_overlap.py                                   # default settings
  python test_moe_expert_overlap.py --segment_lengths 16 32 64 128    # test multiple n
  python test_moe_expert_overlap.py --segment_lengths 256 --num_segments 4
  python test_moe_expert_overlap.py --load_in_4bit                    # 4-bit (~15GB)
  python test_moe_expert_overlap.py --load_in_8bit                    # 8-bit (~30GB)

Requirements:
  pip install torch transformers accelerate bitsandbytes numpy
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. Configuration & CLI
# ---------------------------------------------------------------------------

DEFAULT_PROMPTS = [
    # English – diverse topics
    # "The history of artificial intelligence began in the mid-20th century when",
    # "In quantum computing, qubits differ from classical bits because",
    # "To build a sustainable city, urban planners must consider",
    # "The relationship between music and mathematics has fascinated scholars for centuries.",
    # "Once upon a time in a distant galaxy, a lone explorer discovered",
    # Chinese
    # "大语言模型的训练过程通常包括预训练和微调两个阶段，其中预训练",
    # "中国古代四大发明对世界文明的发展产生了深远的影响，其中造纸术",
    # Code-style
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n",
    # Reasoning
    "Let's solve this step by step. If a train travels at 120 km/h and another train",
    "Please explain the difference between supervised learning and reinforcement learning in detail.",
    # ── Long context (~1200 tokens) ──
    # Tests whether a heavy prefill shifts expert routing patterns during decode.
    """\
Mixture-of-Experts (MoE) models have become a dominant architecture for scaling language models \
beyond the compute budget of dense transformers. Instead of activating every parameter for every \
input token, an MoE layer routes each token to a small subset of expert sub-networks (typically \
the top-k out of N total experts, where k << N) via a learned gating function. This conditional \
computation allows MoE models to dramatically increase total parameter count — and thus model \
capacity — while keeping per-token FLOPs roughly constant.

The Qwen3-30B-A3B model exemplifies this trade-off: with 30.5 billion total parameters distributed \
across 128 experts per MoE layer, only approximately 3.3 billion parameters (8 experts) are activated \
for any given token. The model uses 48 decoder layers with Grouped Query Attention (32 query heads, \
4 key-value heads) and supports a native context window of 32,768 tokens, extensible to 131,072 \
tokens via YaRN positional encoding.

A critical question for MoE inference systems is whether expert activation exhibits temporal \
locality during autoregressive decoding. If consecutive tokens tend to activate similar sets of \
experts, then caching recently-used experts in fast memory (e.g., GPU HBM) can significantly \
reduce the overhead of expert loading in distributed or offloading-based serving systems. \
Conversely, if expert routing is highly dynamic with little overlap between consecutive decode \
steps, then speculative expert prefetching becomes essential.

Several recent works have explored this question. Dejavu (Liu et al., 2023) observed that MoE \
expert activation patterns exhibit high temporal locality in decoder-only models, with adjacent \
tokens sharing 60-80% of their activated experts in middle layers. ExpertFlow (He et al., 2024) \
proposed an expert caching strategy based on this observation, achieving 1.8x speedup for \
offloading-based inference. MoE-Infinity (Xue et al., 2024) extended this to infinite-parameter \
regimes using LRU-based expert caches with activation-aware prefetching.

However, most existing studies measure token-level expert persistence (whether token t and token \
t+1 share experts), which conflates the question with the granularity of scheduling decisions. In \
practice, inference engines batch multiple decode steps before making scheduling decisions. A more \
operationally relevant metric is segment-level overlap: given two adjacent segments of n decode \
steps each, what fraction of the second segment's required experts were already present in the \
first segment's expert set?

Formally, let A_l = union of experts activated at layer l during decode steps [t, t+n), and \
B_l = union of experts activated at layer l during decode steps [t+n, t+2n). The cache hit rate \
is defined as |A_l ∩ B_l| / |B_l|, measuring what proportion of B's demand is already satisfied \
by caching A's experts. This is strictly more informative than Jaccard similarity (|A_l ∩ B_l| / \
|A_l ∪ B_l|) because the denominator reflects only the actual demand of the next segment.

Based on the above background, please provide a detailed analysis of the following questions:
1. How does segment length n affect the cache hit rate across layers?
2. Are there systematic differences between early, middle, and late layers?
3. What is the practical implication for expert caching strategy design?""",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Analyze MoE expert activation overlap between adjacent decode segments"
    )
    p.add_argument(
        "--model_name", type=str, default="Qwen/Qwen3-30B-A3B",
        help="HuggingFace model name or local path"
    )
    p.add_argument(
        "--segment_lengths", type=int, nargs="+", default=[16, 32, 64, 128],
        help="List of segment lengths (n) to test, e.g. --segment_lengths 16 32 64 128"
    )
    p.add_argument(
        "--num_segments", type=int, default=8,
        help="Number of segments to decode per request (for the largest segment length)"
    )
    p.add_argument(
        "--num_prompts", type=int, default=None,
        help="Number of prompts to test (default: all built-in prompts)"
    )
    p.add_argument(
        "--prompts_file", type=str, default=None,
        help="JSON file with a list of prompt strings (overrides built-in prompts)"
    )
    p.add_argument(
        "--load_in_4bit", action="store_true",
        help="Load model in 4-bit quantization (requires bitsandbytes)"
    )
    p.add_argument(
        "--load_in_8bit", action="store_true",
        help="Load model in 8-bit quantization (requires bitsandbytes)"
    )
    p.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype (ignored if using quantization)"
    )
    p.add_argument(
        "--output", type=str, default="expert_overlap_results.json",
        help="Path to save detailed JSON results"
    )
    p.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (0 = greedy)"
    )
    p.add_argument(
        "--disable_thinking", action="store_true",
        help="Add /no_think tag to disable Qwen3 thinking mode"
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# 2. Expert Tracker – hooks into the MoE gate/router
# ---------------------------------------------------------------------------

class ExpertTracker:
    """
    Registers forward hooks on every MoE gate module to capture which
    experts are selected for each decoded token.

    Compatible with two transformers versions:
      - Legacy (≤ v4.x): gate is nn.Linear, returns raw logits
      - Modern (≥ v5.x): gate is Qwen3MoeTopKRouter, returns
            (router_logits, router_scores, router_indices)
    """

    def __init__(self, top_k: int):
        self.top_k = top_k
        # layer_idx -> list of sets; each set = experts activated for one token
        self.records: dict[int, list[set[int]]] = defaultdict(list)
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self.tracking = False

    # ---- hook factories ----

    def _make_hook_for_router(self, layer_idx: int):
        """Hook for Qwen3MoeTopKRouter (modern transformers).
        Output is (router_logits, router_scores, router_indices)."""
        tracker = self

        def hook_fn(module, inp, output):
            if not tracker.tracking:
                return
            # output = (router_logits, router_scores, router_indices)
            # router_indices shape: (seq_len, top_k)
            router_indices = output[2]
            # During decode w/ KV cache, seq_len == 1; take last token
            experts = router_indices[-1].cpu().tolist()
            tracker.records[layer_idx].append(set(experts))

        return hook_fn

    def _make_hook_for_linear_gate(self, layer_idx: int):
        """Hook for nn.Linear gate (legacy transformers).
        Output is raw logits of shape (batch*seq_len, num_experts)."""
        tracker = self

        def hook_fn(module, inp, output):
            if not tracker.tracking:
                return
            logits = output  # (batch*seq_len, num_experts)
            if logits.dim() == 3:
                logits = logits.view(-1, logits.size(-1))
            # top-k on raw logits (monotonic ↔ softmax, same indices)
            _, selected = torch.topk(logits, k=tracker.top_k, dim=-1)
            experts = selected[-1].cpu().tolist()
            tracker.records[layer_idx].append(set(experts))

        return hook_fn

    # ---- registration ----

    def register(self, model) -> list[int]:
        """Auto-detect gate modules and register hooks. Returns MoE layer indices."""
        moe_layer_indices = []

        # Navigate model.model.layers[*].mlp
        layers = None
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "layers"):
            layers = model.layers
        else:
            raise RuntimeError(
                "Cannot locate decoder layers. Expected model.model.layers or model.layers"
            )

        for idx, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue

            gate = getattr(mlp, "gate", None)
            if gate is None:
                # Not an MoE layer (dense MLP)
                continue

            # Determine gate type
            gate_cls_name = type(gate).__name__
            if "Router" in gate_cls_name or "TopK" in gate_cls_name:
                # Modern: Qwen3MoeTopKRouter
                hook = self._make_hook_for_router(idx)
            elif isinstance(gate, torch.nn.Linear):
                # Legacy: nn.Linear
                hook = self._make_hook_for_linear_gate(idx)
            else:
                # Unknown – try the router-style hook (returns tuple)
                hook = self._make_hook_for_router(idx)

            h = gate.register_forward_hook(hook)
            self._hooks.append(h)
            moe_layer_indices.append(idx)

        return moe_layer_indices

    # ---- control ----

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


# ---------------------------------------------------------------------------
# 3. Overlap Metrics
# ---------------------------------------------------------------------------

def jaccard(a: set, b: set) -> float:
    """Jaccard similarity: |A∩B| / |A∪B|"""
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union


def overlap_coefficient(a: set, b: set) -> float:
    """|A∩B| / min(|A|, |B|)"""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def cache_hit_rate(a: set, b: set) -> float:
    """
    |A∩B| / |B|  —  the fraction of segment B's required experts
    that were already present (activated/cached) in segment A.

    Interpretation: if we cache all experts from the previous segment A,
    what proportion of the next segment B's experts would be cache hits?

    Returns 0.0 if B is empty (no experts needed → nothing to hit).
    """
    if not b:
        return 0.0
    return len(a & b) / len(b)


def compute_segment_metrics(
    expert_records: list[set[int]],
    segment_length: int,
) -> dict[str, Any]:
    """
    Given a list of expert sets (one per decoded token), split into
    segments and compute overlap metrics between adjacent segments.
    """
    n_tokens = len(expert_records)
    n_segments = n_tokens // segment_length
    if n_segments < 2:
        return {"error": "not enough tokens for 2 segments"}

    # Build segment-level union of experts
    segment_unions: list[set[int]] = []
    for s in range(n_segments):
        start = s * segment_length
        end = start + segment_length
        union_set: set[int] = set()
        for token_experts in expert_records[start:end]:
            union_set |= token_experts
        segment_unions.append(union_set)

    # Compute pairwise adjacent metrics
    jaccards = []
    overlaps = []
    cache_hits = []
    intersection_sizes = []
    union_sizes = []
    seg_expert_counts = [len(s) for s in segment_unions]

    for i in range(len(segment_unions) - 1):
        a, b = segment_unions[i], segment_unions[i + 1]
        jaccards.append(jaccard(a, b))
        overlaps.append(overlap_coefficient(a, b))
        cache_hits.append(cache_hit_rate(a, b))
        intersection_sizes.append(len(a & b))
        union_sizes.append(len(a | b))

    return {
        "jaccard_per_pair": jaccards,
        "jaccard_mean": float(np.mean(jaccards)),
        "jaccard_std": float(np.std(jaccards)),
        "cache_hit_per_pair": cache_hits,
        "cache_hit_mean": float(np.mean(cache_hits)),
        "cache_hit_std": float(np.std(cache_hits)),
        "overlap_coeff_per_pair": overlaps,
        "overlap_coeff_mean": float(np.mean(overlaps)),
        "intersection_size_per_pair": intersection_sizes,
        "intersection_size_mean": float(np.mean(intersection_sizes)),
        "union_size_per_pair": union_sizes,
        "segment_expert_counts": seg_expert_counts,
        "segment_expert_count_mean": float(np.mean(seg_expert_counts)),
        "num_segments": n_segments,
    }


def compute_token_level_persistence(
    expert_records: list[set[int]],
) -> dict[str, float]:
    """
    For consecutive tokens, what fraction of experts in token t are
    also present in token t+1?
    """
    if len(expert_records) < 2:
        return {"token_persistence_mean": 0.0}

    persistence_rates = []
    for i in range(len(expert_records) - 1):
        a, b = expert_records[i], expert_records[i + 1]
        if a:
            persistence_rates.append(len(a & b) / len(a))

    return {
        "token_persistence_mean": float(np.mean(persistence_rates)),
        "token_persistence_std": float(np.std(persistence_rates)),
        "token_persistence_median": float(np.median(persistence_rates)),
    }


# ---------------------------------------------------------------------------
# 4. Model Loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[*] Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )

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
        print("[*] Loading model in 4-bit quantization...")
    elif args.load_in_8bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        print("[*] Loading model in 8-bit quantization...")
    else:
        load_kwargs["torch_dtype"] = dtype_map[args.dtype]
        print(f"[*] Loading model in {args.dtype}...")

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)
    model.eval()

    print(f"[*] Model loaded: {type(model).__name__}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# 5. Decode Loop with Expert Tracking
# ---------------------------------------------------------------------------

@torch.no_grad()
def decode_and_track(
    model,
    tokenizer,
    prompt: str,
    tracker: ExpertTracker,
    total_decode_tokens: int,
    temperature: float = 0.0,
    disable_thinking: bool = False,
) -> str:
    """
    Run autoregressive decoding with KV cache, tracking expert activations
    for each generated token.

    Returns the generated text.
    """
    # Format prompt for chat-style models
    if disable_thinking:
        prompt_text = prompt + " /no_think"
    else:
        prompt_text = prompt

    # Try chat template first, fall back to raw prompt
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

    # === Prefill (no tracking) ===
    tracker.stop()
    outputs = model(input_ids, use_cache=True)
    past_key_values = outputs.past_key_values

    # Pick first token
    if temperature <= 0:
        next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    else:
        probs = F.softmax(outputs.logits[:, -1, :] / temperature, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)

    generated_ids = [next_token_id.item()]

    # === Decode with tracking ===
    tracker.start()
    eos_token_id = tokenizer.eos_token_id

    for step in range(total_decode_tokens):
        outputs = model(
            next_token_id,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

        if temperature <= 0:
            next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(outputs.logits[:, -1, :] / temperature, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

        token_id = next_token_id.item()
        generated_ids.append(token_id)

        if token_id == eos_token_id:
            break

    tracker.stop()

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text


# ---------------------------------------------------------------------------
# 6. Main Analysis Pipeline
# ---------------------------------------------------------------------------

def run_analysis(args):
    # --- Load prompts ---
    if args.prompts_file:
        with open(args.prompts_file, "r") as f:
            prompts = json.load(f)
        assert isinstance(prompts, list), "prompts_file must contain a JSON list of strings"
    else:
        prompts = DEFAULT_PROMPTS

    if args.num_prompts is not None:
        prompts = prompts[: args.num_prompts]

    segment_lengths = sorted(set(args.segment_lengths))
    assert all(n > 0 for n in segment_lengths), "All segment lengths must be > 0"

    max_seg_len = max(segment_lengths)
    total_decode = max_seg_len * args.num_segments

    print("=" * 70)
    print("  MoE Expert Activation Overlap Analyzer")
    print("=" * 70)
    print(f"  Model           : {args.model_name}")
    print(f"  Segment lengths : {segment_lengths}")
    print(f"  Num segments    : {args.num_segments} (for max seg_len={max_seg_len})")
    print(f"  Decode tokens   : {total_decode} per prompt")
    print(f"  Num prompts     : {len(prompts)}")
    print(f"  Temperature     : {args.temperature}")
    print("=" * 70)

    # --- Load model ---
    model, tokenizer = load_model_and_tokenizer(args)

    # --- Read config ---
    config = model.config
    top_k = getattr(config, "num_experts_per_tok", None)
    if top_k is None:
        top_k = getattr(config, "top_k", 8)
    num_experts = getattr(config, "num_experts", None)
    if num_experts is None:
        num_experts = getattr(config, "num_local_experts", 128)

    print(f"\n[*] MoE config: {num_experts} experts, top-{top_k} routing")

    # --- Set up tracker ---
    tracker = ExpertTracker(top_k=top_k)
    moe_layers = tracker.register(model)
    print(f"[*] Registered hooks on {len(moe_layers)} MoE layers: "
          f"[{moe_layers[0]}..{moe_layers[-1]}]")

    # --- Run experiments ---
    all_results: dict[str, Any] = {
        "config": {
            "model": args.model_name,
            "num_experts": num_experts,
            "top_k": top_k,
            "num_moe_layers": len(moe_layers),
            "moe_layer_indices": moe_layers,
            "segment_lengths": segment_lengths,
            "num_segments": args.num_segments,
            "total_decode_tokens": total_decode,
            "temperature": args.temperature,
            "num_prompts": len(prompts),
        },
        "per_prompt": [],
    }

    # Aggregators: seg_len -> layer_idx -> list of values
    global_jaccards: dict[int, dict[int, list[float]]] = {
        sl: defaultdict(list) for sl in segment_lengths
    }
    global_cache_hits: dict[int, dict[int, list[float]]] = {
        sl: defaultdict(list) for sl in segment_lengths
    }
    global_token_persist: dict[int, list[float]] = defaultdict(list)

    for pi, prompt in enumerate(prompts):
        prompt_short = prompt[:60].replace("\n", "\\n") + ("..." if len(prompt) > 60 else "")
        print(f"\n{'─'*70}")
        print(f"  Prompt {pi+1}/{len(prompts)}: \"{prompt_short}\"")
        print(f"{'─'*70}")

        tracker.clear()
        t0 = time.time()

        generated_text = decode_and_track(
            model, tokenizer, prompt, tracker,
            total_decode_tokens=total_decode,
            temperature=args.temperature,
            disable_thinking=args.disable_thinking,
        )

        elapsed = time.time() - t0
        actual_decoded = len(next(iter(tracker.records.values()))) if tracker.records else 0
        print(f"  Decoded {actual_decoded} tokens in {elapsed:.1f}s "
              f"({actual_decoded/elapsed:.1f} tok/s)")

        if actual_decoded < segment_lengths[0] * 2:
            print(f"  ⚠ Too few tokens decoded ({actual_decoded}), skipping.")
            continue

        prompt_result: dict[str, Any] = {
            "prompt": prompt,
            "tokens_decoded": actual_decoded,
            "time_seconds": round(elapsed, 2),
            "segment_analyses": {},
        }

        # --- Compute metrics for each segment length ---
        for seg_len in segment_lengths:
            if actual_decoded < seg_len * 2:
                continue

            seg_result: dict[str, Any] = {}
            print(f"\n  ── Segment length = {seg_len} ──")

            for layer_idx in moe_layers:
                records = tracker.records[layer_idx]
                metrics = compute_segment_metrics(records, seg_len)

                if "error" in metrics:
                    continue

                seg_result[layer_idx] = metrics
                global_jaccards[seg_len][layer_idx].extend(metrics["jaccard_per_pair"])
                global_cache_hits[seg_len][layer_idx].extend(metrics["cache_hit_per_pair"])

                # Token-level persistence (only compute once, for smallest seg_len)
                if seg_len == segment_lengths[0]:
                    tp = compute_token_level_persistence(records)
                    global_token_persist[layer_idx].append(tp["token_persistence_mean"])

            # Print summary for a few representative layers
            representative = [moe_layers[0], moe_layers[len(moe_layers)//4],
                              moe_layers[len(moe_layers)//2], moe_layers[3*len(moe_layers)//4],
                              moe_layers[-1]]
            representative = sorted(set(representative))

            print(f"  {'Layer':>8} │ {'CacheHit':>9} │ {'Jaccard':>9} │ "
                  f"{'∩ size':>8} │ {'|B|':>6} │ {'Experts/seg':>11}")
            print(f"  {'─'*8}─┼─{'─'*9}─┼─{'─'*9}─┼─"
                  f"{'─'*8}─┼─{'─'*6}─┼─{'─'*11}")
            for li in representative:
                if li in seg_result:
                    m = seg_result[li]
                    # |B| ≈ average expert count of all non-first segments
                    avg_b_size = np.mean(m["segment_expert_counts"][1:])
                    print(f"  {li:>8} │ {m['cache_hit_mean']:>9.4f} │ "
                          f"{m['jaccard_mean']:>9.4f} │ "
                          f"{m['intersection_size_mean']:>8.1f} │ "
                          f"{avg_b_size:>6.1f} │ "
                          f"{m['segment_expert_count_mean']:>11.1f}")

            prompt_result["segment_analyses"][seg_len] = {
                str(k): v for k, v in seg_result.items()
            }

        all_results["per_prompt"].append(prompt_result)

    # --- Aggregate global statistics ---
    print(f"\n{'='*70}")
    print("  GLOBAL STATISTICS (across all prompts)")
    print(f"{'='*70}")

    global_summary: dict[str, Any] = {}

    for seg_len in segment_lengths:
        print(f"\n  ══ Segment length = {seg_len} ══")
        layer_stats = {}

        print(f"  {'Layer':>8} │ {'CacheHit μ':>10} │ {'CacheHit σ':>10} │ "
              f"{'Jaccard μ':>9} │ {'Jaccard σ':>9} │ {'N pairs':>7}")
        print(f"  {'─'*8}─┼─{'─'*10}─┼─{'─'*10}─┼─"
              f"{'─'*9}─┼─{'─'*9}─┼─{'─'*7}")

        all_layer_jaccards = []
        all_layer_cache_hits = []

        for li in moe_layers:
            j_vals = global_jaccards[seg_len].get(li, [])
            c_vals = global_cache_hits[seg_len].get(li, [])
            if not j_vals:
                continue
            all_layer_jaccards.extend(j_vals)
            all_layer_cache_hits.extend(c_vals)

            mean_c = float(np.mean(c_vals))
            std_c = float(np.std(c_vals))
            mean_j = float(np.mean(j_vals))
            std_j = float(np.std(j_vals))

            layer_stats[li] = {
                "cache_hit_mean": round(mean_c, 5),
                "cache_hit_std": round(std_c, 5),
                "cache_hit_min": round(float(np.min(c_vals)), 5),
                "cache_hit_max": round(float(np.max(c_vals)), 5),
                "jaccard_mean": round(mean_j, 5),
                "jaccard_std": round(std_j, 5),
                "jaccard_min": round(float(np.min(j_vals)), 5),
                "jaccard_max": round(float(np.max(j_vals)), 5),
                "num_pairs": len(j_vals),
            }
            print(f"  {li:>8} │ {mean_c:>10.4f} │ {std_c:>10.4f} │ "
                  f"{mean_j:>9.4f} │ {std_j:>9.4f} │ {len(j_vals):>7}")

        if all_layer_cache_hits:
            overall_c_mean = float(np.mean(all_layer_cache_hits))
            overall_c_std = float(np.std(all_layer_cache_hits))
            overall_j_mean = float(np.mean(all_layer_jaccards))
            overall_j_std = float(np.std(all_layer_jaccards))
            print(f"  {'─'*8}─┼─{'─'*10}─┼─{'─'*10}─┼─"
                  f"{'─'*9}─┼─{'─'*9}─┼─{'─'*7}")
            print(f"  {'ALL':>8} │ {overall_c_mean:>10.4f} │ {overall_c_std:>10.4f} │ "
                  f"{overall_j_mean:>9.4f} │ {overall_j_std:>9.4f} │ "
                  f"{len(all_layer_cache_hits):>7}")

        global_summary[f"seg_len_{seg_len}"] = {
            "per_layer": {str(k): v for k, v in layer_stats.items()},
            "overall_cache_hit_mean": round(float(np.mean(all_layer_cache_hits)), 5) if all_layer_cache_hits else None,
            "overall_cache_hit_std": round(float(np.std(all_layer_cache_hits)), 5) if all_layer_cache_hits else None,
            "overall_jaccard_mean": round(float(np.mean(all_layer_jaccards)), 5) if all_layer_jaccards else None,
            "overall_jaccard_std": round(float(np.std(all_layer_jaccards)), 5) if all_layer_jaccards else None,
        }

    # Token persistence
    if global_token_persist:
        print(f"\n  ── Token-level Expert Persistence (consecutive tokens) ──")
        print(f"  {'Layer':>8} │ {'Persist μ':>9}")
        print(f"  {'─'*8}─┼─{'─'*9}")
        persist_vals = []
        for li in moe_layers:
            vals = global_token_persist.get(li, [])
            if vals:
                mean_p = float(np.mean(vals))
                persist_vals.append(mean_p)
                print(f"  {li:>8} │ {mean_p:>9.4f}")
        if persist_vals:
            print(f"  {'─'*8}─┼─{'─'*9}")
            print(f"  {'ALL':>8} │ {float(np.mean(persist_vals)):>9.4f}")
        global_summary["token_persistence"] = {
            str(li): round(float(np.mean(global_token_persist[li])), 5)
            for li in moe_layers if global_token_persist.get(li)
        }

    all_results["global_summary"] = global_summary

    # --- Save results ---
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[*] Detailed results saved to: {args.output}")

    # --- Cleanup ---
    tracker.remove_hooks()

    print(f"\n{'='*70}")
    print("  INTERPRETATION GUIDE")
    print(f"{'='*70}")
    print("""
    ★ CacheHit = |A∩B| / |B|  (PRIMARY METRIC)
      The fraction of segment B's required experts already in segment A.
      = "If I cache A's experts, what % of B's needs are satisfied?"

      CacheHit ≈ 1.0 : Almost all experts B needs were in A
                        → Caching previous segment's experts is highly effective
      CacheHit ≈ 0.5 : Half of B's experts are new
      CacheHit ≈ 0.0 : B needs completely different experts
                        → Caching A gives no benefit

    Jaccard = |A∩B| / |A∪B|
      Symmetric similarity. Always ≤ CacheHit (since |A∪B| ≥ |B|).
      Useful for understanding total expert "churn" between segments.

    Higher CacheHit → better potential for:
      • Expert prefetching / caching between decode batches
      • Reducing expert-loading overhead in distributed MoE inference
      • Predicting which experts will be needed next
    """)

    return all_results


# ---------------------------------------------------------------------------
# 7. Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    run_analysis(args)
