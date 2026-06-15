#!/usr/bin/env python3
"""
Measure adjacent-segment MoE expert hit ratios in a decoded-token interval.

For each segment size c, split the inclusive 1-based token interval
[start_n, end_n] into consecutive complete c-token segments. Trailing tokens
that do not fill a complete segment are ignored. For every MoE layer and every
adjacent pair (i - 1, i), compute:

    hit_ratio = |S_i intersect S_{i-1}| / |S_i|

where S_i is the union of experts routed by tokens in segment i.

Examples:
  python test_moe_adjacent_segment_overlap.py \
      --intermediate_file /data2/group_谈海生/mumura/nano_moe/motivaion/segment_unique_expert_counts_optimized_avg8192_routing_trace.pt \
      --segment_sizes 1 2 3 3 5 6 7 8 9 10 11 12 13 14 15 16\
      --start_n 1 --end_n 512 \
      --output adjacent_segment_overlap_results.json

  CUDA_VISIBLE_DEVICES=2 python test_moe_adjacent_segment_overlap.py \
      --model_name /path/to/Qwen3-30B-A3B \
      --prompts_file long_reasoning_prompts_en.json \
      --segment_sizes 2 4 8 16 \
      --start_n 1025 --end_n 8192 \
      --print_layers 0 12 24 36 47 \
      --dtype bfloat16 \
      --output adjacent_segment_overlap_results.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import test_moe_segment_history_overlap as history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure per-layer expert hit ratios for every pair of adjacent "
            "segments in an inclusive decoded-token interval."
        )
    )
    parser.add_argument(
        "--segment_sizes",
        type=int,
        nargs="+",
        required=True,
        help="Segment sizes in decoded tokens, e.g. --segment_sizes 1 2 4 8.",
    )
    parser.add_argument(
        "--start_n",
        type=int,
        required=True,
        help="Inclusive 1-based first decoded-token position.",
    )
    parser.add_argument(
        "--end_n",
        type=int,
        required=True,
        help="Inclusive 1-based last decoded-token position.",
    )
    parser.add_argument(
        "--print_layers",
        type=int,
        nargs="+",
        default=None,
        help=(
            "MoE layer indices to print. By default, print five "
            "representative layers."
        ),
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--num_prompts", type=int, default=None)
    parser.add_argument("--prompts_file", type=str, default=None)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--disable_thinking", action="store_true")
    parser.add_argument(
        "--output",
        type=str,
        default="adjacent_segment_overlap_results.json",
    )
    parser.add_argument(
        "--intermediate_file",
        type=str,
        default=None,
        help=(
            "Existing moe_routing_trace .pt file. When provided, skip "
            "model loading and inference."
        ),
    )
    parser.add_argument(
        "--intermediate_output",
        type=str,
        default=None,
        help=(
            "Directory for a newly collected routing trace. The filename is "
            "derived from --output; default is the output directory."
        ),
    )
    return parser.parse_args()


def summarize_values(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "mean": None,
            "variance": None,
            "min": None,
            "max": None,
            "num_samples": 0,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(np.mean(array)), 6),
        "variance": round(float(np.var(array)), 6),
        "min": round(float(np.min(array)), 6),
        "max": round(float(np.max(array)), 6),
        "num_samples": len(values),
    }


def compute_adjacent_hit_ratios(
    expert_records: torch.Tensor,
    segment_size: int,
    start_n: int,
    end_n: int,
) -> list[dict[str, Any]]:
    """Return metrics for complete adjacent segments in [start_n, end_n]."""
    interval_length = end_n - start_n + 1
    num_segments = interval_length // segment_size
    start_idx = start_n - 1
    segment_sets: list[set[int]] = []

    for segment_index in range(num_segments):
        token_start_idx = start_idx + segment_index * segment_size
        token_end_idx = token_start_idx + segment_size
        segment_sets.append(
            {
                int(expert)
                for expert in torch.unique(
                    expert_records[token_start_idx:token_end_idx]
                ).tolist()
            }
        )

    adjacent_pairs: list[dict[str, Any]] = []
    for current_index in range(1, num_segments):
        previous_set = segment_sets[current_index - 1]
        current_set = segment_sets[current_index]
        intersection_size = len(previous_set & current_set)
        previous_token_start = start_n + (current_index - 1) * segment_size
        current_token_start = start_n + current_index * segment_size
        adjacent_pairs.append(
            {
                "pair_index": current_index,
                "previous_segment_index": current_index,
                "current_segment_index": current_index + 1,
                "previous_token_start_1based": previous_token_start,
                "previous_token_end_1based_inclusive": (
                    previous_token_start + segment_size - 1
                ),
                "current_token_start_1based": current_token_start,
                "current_token_end_1based_inclusive": (
                    current_token_start + segment_size - 1
                ),
                "intersection_size": intersection_size,
                "previous_expert_count": len(previous_set),
                "current_expert_count": len(current_set),
                "hit_ratio": history.safe_div(
                    intersection_size, len(current_set)
                ),
            }
        )
    return adjacent_pairs


def representative_layers(moe_layers: list[int]) -> list[int]:
    if len(moe_layers) <= 5:
        return moe_layers
    last = len(moe_layers) - 1
    return sorted(
        {
            moe_layers[0],
            moe_layers[last // 4],
            moe_layers[last // 2],
            moe_layers[(3 * last) // 4],
            moe_layers[last],
        }
    )


def postprocess_routing_data(
    args: argparse.Namespace,
    routing_data: dict[str, Any],
    segment_sizes: list[int],
) -> dict[str, Any]:
    collection_config = routing_data["collection_config"]
    moe_layers = [
        int(layer_idx)
        for layer_idx in collection_config["moe_layer_indices"]
    ]
    interval_length = args.end_n - args.start_n + 1
    print_layers = (
        representative_layers(moe_layers)
        if args.print_layers is None
        else list(dict.fromkeys(args.print_layers))
    )
    unknown_layers = sorted(set(print_layers) - set(moe_layers))
    if unknown_layers:
        raise ValueError(
            f"--print_layers contains non-MoE layers: {unknown_layers}; "
            f"available layers are {moe_layers}"
        )

    results: dict[str, Any] = {
        "config": {
            "model": collection_config["model_name"],
            "num_experts": collection_config.get("num_experts"),
            "top_k": int(collection_config["top_k"]),
            "moe_layer_indices": moe_layers,
            "num_moe_layers": len(moe_layers),
            "segment_sizes": segment_sizes,
            "start_n_1based_inclusive": args.start_n,
            "end_n_1based_inclusive": args.end_n,
            "interval_length": interval_length,
            "segment_policy": (
                "consecutive non-overlapping complete segments starting at "
                "start_n; trailing incomplete tokens are ignored"
            ),
            "hit_ratio_definition": (
                "|S_current intersect S_previous| / |S_current|"
            ),
            "variance_definition": "population variance (ddof=0)",
            "num_prompts": len(routing_data["per_prompt"]),
            "routing_trace_max_decode_tokens": collection_config.get(
                "max_decode_tokens_requested"
            ),
            "temperature": collection_config.get("temperature"),
            "dtype": collection_config.get("dtype"),
            "quantization": collection_config.get("quantization"),
            "disable_thinking": collection_config.get(
                "disable_thinking", False
            ),
        },
        "per_prompt": [],
        "summary": {},
    }

    aggregate: dict[tuple[int, int], list[float]] = defaultdict(list)
    aggregate_by_pair: dict[tuple[int, int, int], list[float]] = defaultdict(
        list
    )

    print("=" * 78)
    print("  MoE Adjacent-Segment Expert Overlap Analyzer")
    print("=" * 78)
    print(f"  Routing model   : {collection_config['model_name']}")
    print(f"  Token interval  : [{args.start_n}, {args.end_n}] (1-based, inclusive)")
    print(f"  Segment sizes   : {segment_sizes}")
    print(f"  Print layers    : {print_layers}")
    print(f"  Prompts         : {len(routing_data['per_prompt'])}")
    print("=" * 78)

    for prompt_data in routing_data["per_prompt"]:
        prompt_index = int(prompt_data["prompt_index"])
        routed_experts = prompt_data["routed_experts"]
        layer_token_counts = [
            int(routed_experts[str(layer_idx)].shape[0])
            for layer_idx in moe_layers
            if isinstance(routed_experts.get(str(layer_idx)), torch.Tensor)
        ]
        actual_tokens = (
            min([int(prompt_data["tokens_decoded"]), *layer_token_counts])
            if layer_token_counts
            else 0
        )
        prompt_result: dict[str, Any] = {
            "prompt_index": prompt_index,
            "prompt": prompt_data["prompt"],
            "tokens_tracked": actual_tokens,
            "stop_reason": prompt_data.get("stop_reason"),
            "measurements": {},
        }

        if actual_tokens < args.end_n:
            prompt_result["error"] = (
                f"Need routed tokens through position {args.end_n}, "
                f"but only {actual_tokens} are available."
            )
            print(
                f"  Prompt {prompt_index + 1}: skipped, need "
                f"{args.end_n} tokens, got {actual_tokens}"
            )
            results["per_prompt"].append(prompt_result)
            continue

        for segment_size in segment_sizes:
            num_segments = interval_length // segment_size
            measurement: dict[str, Any] = {
                "segment_size": segment_size,
                "num_complete_segments": num_segments,
                "num_adjacent_pairs": max(0, num_segments - 1),
                "ignored_trailing_tokens": (
                    interval_length - num_segments * segment_size
                ),
                "per_layer": {},
            }
            for layer_idx in moe_layers:
                expert_records = routed_experts.get(str(layer_idx))
                if not isinstance(expert_records, torch.Tensor):
                    continue
                adjacent_pairs = compute_adjacent_hit_ratios(
                    expert_records=expert_records,
                    segment_size=segment_size,
                    start_n=args.start_n,
                    end_n=args.end_n,
                )
                hit_ratios = [
                    float(pair["hit_ratio"]) for pair in adjacent_pairs
                ]
                measurement["per_layer"][str(layer_idx)] = {
                    "adjacent_pairs": adjacent_pairs,
                    "summary": summarize_values(hit_ratios),
                }
                aggregate[(segment_size, layer_idx)].extend(hit_ratios)
                for pair in adjacent_pairs:
                    aggregate_by_pair[
                        (
                            segment_size,
                            layer_idx,
                            int(pair["pair_index"]),
                        )
                    ].append(float(pair["hit_ratio"]))
            prompt_result["measurements"][str(segment_size)] = measurement

        results["per_prompt"].append(prompt_result)

    for segment_size in segment_sizes:
        per_layer: dict[str, Any] = {}
        all_values: list[float] = []
        num_pairs = max(0, interval_length // segment_size - 1)
        for layer_idx in moe_layers:
            values = aggregate.get((segment_size, layer_idx), [])
            all_values.extend(values)
            by_pair_index = {
                str(pair_index): summarize_values(
                    aggregate_by_pair.get(
                        (segment_size, layer_idx, pair_index), []
                    )
                )
                for pair_index in range(1, num_pairs + 1)
            }
            per_layer[str(layer_idx)] = {
                "all_pairs_all_prompts": summarize_values(values),
                "by_pair_index": by_pair_index,
            }
        results["summary"][str(segment_size)] = {
            "num_complete_segments_per_prompt": (
                interval_length // segment_size
            ),
            "num_adjacent_pairs_per_prompt": num_pairs,
            "ignored_trailing_tokens": interval_length % segment_size,
            "per_layer": per_layer,
            "overall_across_layers_pairs_prompts": summarize_values(
                all_values
            ),
        }

    print(f"\n{'=' * 78}")
    print("  HIT RATIO SUMMARY (all adjacent pairs and prompts)")
    print(f"{'=' * 78}")
    for segment_size in segment_sizes:
        print(f"\n  segment_size={segment_size}")
        print(
            f"  {'Layer':>8} | {'Mean':>10} | {'Variance':>10} | "
            f"{'Min':>10} | {'Max':>10} | {'Samples':>9}"
        )
        print(f"  {'-' * 68}")
        for layer_idx in print_layers:
            stats = results["summary"][str(segment_size)]["per_layer"][
                str(layer_idx)
            ]["all_pairs_all_prompts"]

            def format_stat(name: str) -> str:
                value = stats[name]
                return "n/a" if value is None else f"{value:.6f}"

            print(
                f"  {layer_idx:>8} | {format_stat('mean'):>10} | "
                f"{format_stat('variance'):>10} | "
                f"{format_stat('min'):>10} | {format_stat('max'):>10} | "
                f"{stats['num_samples']:>9}"
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2, ensure_ascii=False)
    print(f"\n[*] Detailed results saved to: {output_path}")
    return results


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    segment_sizes = sorted(set(args.segment_sizes))
    if any(segment_size <= 0 for segment_size in segment_sizes):
        raise ValueError("All --segment_sizes must be positive")
    if args.start_n <= 0:
        raise ValueError("--start_n must be a positive 1-based index")
    if args.end_n < args.start_n:
        raise ValueError("--end_n must be >= --start_n")
    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError(
            "Choose at most one of --load_in_4bit and --load_in_8bit"
        )
    if all(
        (args.end_n - args.start_n + 1) // segment_size < 2
        for segment_size in segment_sizes
    ):
        raise ValueError(
            "The interval must contain at least two complete segments for "
            "at least one requested segment size"
        )

    if args.intermediate_file:
        intermediate_path = Path(args.intermediate_file)
        print(
            "[*] Post-processing only; reading intermediate file: "
            f"{intermediate_path}"
        )
        routing_data = history.load_routing_data(intermediate_path)
    else:
        prompts = history.load_prompts(args)
        intermediate_path = history.default_intermediate_path(
            args.output,
            args.intermediate_output,
        )
        history.collect_routing_data(
            args=args,
            prompts=prompts,
            intermediate_path=intermediate_path,
            total_decode_tokens=args.end_n,
        )
        routing_data = history.load_routing_data(intermediate_path)

    return postprocess_routing_data(
        args=args,
        routing_data=routing_data,
        segment_sizes=segment_sizes,
    )


if __name__ == "__main__":
    run_analysis(parse_args())
