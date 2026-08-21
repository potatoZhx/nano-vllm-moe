"""Execution orchestration for the evaluation TPOT benchmark."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

from nanovllm.benchmarks.eval_tpot.cases import build_cases, case_name
from nanovllm.benchmarks.eval_tpot.config import (
    _num_samples_label,
    _parse_allocation_modes,
    configure_optimized_env,
)
from nanovllm.benchmarks.eval_tpot.data import (
    DATASET_PATHS,
    _effective_request_mode,
    _effective_sample_offset,
    _requested_datasets,
    load_dataset_samples,
    prepare_prompt_tokens,
    select_samples,
)
from nanovllm.benchmarks.eval_tpot.metrics import (
    TPOT_DEFINITION,
    collect_profile_metrics,
    grouped_summaries,
    reset_llm_profile,
    run_prompt,
    run_prompt_batch_generate,
    run_prompt_generate,
    steady_draft_call_stats,
    summarize_rows,
)
from nanovllm.benchmarks.eval_tpot.reporting import (
    flatten_rows,
    write_csv,
    write_markdown_report,
    write_summary_csv,
)
from nanovllm.benchmarks.eval_tpot.runtime import (
    create_llm,
    parse_csv as _parse_csv,
    reset_runtime_seed,
    resolved_runtime_config,
    runtime_seed,
    validate_kv_cache_capacity,
    warmup_llm,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


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
                top_k=int(args.top_k),
                top_p=float(args.top_p),
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
                "batch_size": int(args.batch_size),
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
                validate_kv_cache_capacity(
                    llm,
                    prompt_tokens=int(base_row["prompt_tokens"]),
                    max_tokens=max_tokens,
                    batch_size=int(args.batch_size),
                )
                ignore_eos = bool(case.get("ignore_eos", False))
                if bool(args.reset_seed_after_warmup):
                    reset_runtime_seed(int(base_row["runtime_seed"]))
                if bool(args.collect_profile) and bool(args.reset_profile_before_request):
                    reset_llm_profile(llm)
                if int(args.batch_size) > 1:
                    result = run_prompt_batch_generate(
                        llm,
                        prompt_tokens,
                        batch_size=int(args.batch_size),
                        temperature=float(args.temperature),
                        top_k=int(args.top_k),
                        top_p=float(args.top_p),
                        max_tokens=max_tokens,
                        ignore_eos=ignore_eos,
                        eos_token_id=getattr(llm.config, "eos", None),
                        max_model_len=int(args.max_model_len),
                    )
                else:
                    run_prompt_fn = (
                        run_prompt_generate
                        if str(args.decode_driver) == "generate"
                        else run_prompt
                    )
                    result = run_prompt_fn(
                        llm,
                        prompt_tokens,
                        temperature=float(args.temperature),
                        top_k=int(args.top_k),
                        top_p=float(args.top_p),
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
                            "transfer_aware_enabled": bool(
                                args.transfer_aware_profile
                            ),
                            "target": "spec.verify_accept_ready_ms",
                            "target_boundary": (
                                "before ModelRunner.run_verify through acceptance-result "
                                "consumption on the host"
                            ),
                            "execution_workload": (
                                "all CUDA-graph bucket rows, including padding"
                            ),
                            "route_alignment": (
                                "draft forward i original routes -> verify "
                                "logical row i-1; one verify-next row remains "
                                "after the final draft"
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
                            "protocol": {
                                "batch_size": int(args.batch_size),
                                "acceptance_strategy": str(
                                    args.acceptance_strategy
                                ),
                                "temperature": float(args.temperature),
                                "cache_ratio": float(case["cache_ratio"]),
                                "max_draft_tokens": int(
                                    case["max_draft_tokens"]
                                ),
                                "prefetch_runtime_kind": str(
                                    args.prefetch_runtime_kind
                                ),
                                "verify_buckets": _parse_csv(
                                    args.verify_cuda_graph_bucket_steps, int
                                ),
                            },
                            "output_validation": {
                                "output_sequence_count": int(
                                    result.get("output_sequence_count", 0)
                                ),
                                "fixed_length_ok": bool(
                                    result.get("output_fixed_length_ok", False)
                                ),
                                "error": str(
                                    result.get(
                                        "output_validation_error", ""
                                    )
                                ),
                                "outputs_digest": str(
                                    result.get("outputs_digest", "")
                                ),
                            },
                            "steady_draft_gate": steady_draft_call_stats(
                                profile
                            ),
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
                    if token_ids and isinstance(token_ids[0], list):
                        result["generated_texts"] = [
                            llm.tokenizer.decode(row) for row in token_ids
                        ]
                    else:
                        result["generated_text"] = (
                            llm.tokenizer.decode(token_ids)
                            if token_ids is not None
                            else ""
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
                        f" aggregate_ms/tok={float(result['aggregate_tpot_ms_per_output_token']):.3f}"
                        f" aggregate_tok/s={float(result['aggregate_decode_tok_s']):.3f}"
                        if "aggregate_decode_tok_s" in result
                        else ""
                    )
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

def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = REPO_ROOT
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
                **resolved_runtime_config(args),
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
                    top_k=int(args.top_k),
                    top_p=float(args.top_p),
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
            **resolved_runtime_config(args),
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
            "top_k": int(args.top_k),
            "top_p": float(args.top_p),
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
            "draft_tpot_alpha_error_p90": float(
                args.draft_tpot_alpha_error_p90
            ),
            "draft_tpot_draft_error_p90_ms": float(
                args.draft_tpot_draft_error_p90_ms
            ),
            "draft_tpot_uncertainty_scale": float(
                args.draft_tpot_uncertainty_scale
            ),
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
            "transfer_aware_profile": bool(args.transfer_aware_profile),
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
            "predictive_phase1_budget": int(args.predictive_phase1_budget),
            "predictive_phase1_recent_verify": bool(
                getattr(args, "predictive_phase1_recent_verify", False)
            ),
            "predictive_ghost_window_steps": int(args.predictive_ghost_window_steps),
            "predictive_ghost_protect_steps": int(args.predictive_ghost_protect_steps),
            "fused_cache_lut_updates": bool(args.fused_cache_lut_updates),
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
            "batch_size": int(args.batch_size),
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
