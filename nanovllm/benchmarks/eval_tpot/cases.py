"""Deterministic workload-case construction and naming."""
from __future__ import annotations

import argparse
from typing import Any

from nanovllm.benchmarks.eval_tpot.config import _parse_allocation_modes
from nanovllm.benchmarks.eval_tpot.data import _requested_datasets
from nanovllm.benchmarks.eval_tpot.runtime import parse_csv as _parse_csv


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)

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

