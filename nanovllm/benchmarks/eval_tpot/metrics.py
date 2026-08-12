"""TPOT execution timing, output validation, and profile metrics."""
from __future__ import annotations

import hashlib
from statistics import mean
from time import perf_counter
from typing import Any


TPOT_DEFINITION = "decode_sec / (generated_output_tokens - 1)"

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
    aggregate_tpot = [
        float(row.get("aggregate_tpot_ms_per_output_token", row["tpot_ms"]))
        for row in ok_rows
    ]
    aggregate_decode_tps = [
        float(row.get("aggregate_decode_tok_s", row["decode_tok_s"]))
        for row in ok_rows
    ]
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
        "aggregate_tpot_ms_per_output_token_mean": (
            float(mean(aggregate_tpot)) if aggregate_tpot else 0.0
        ),
        "aggregate_decode_tok_s_mean": (
            float(mean(aggregate_decode_tps)) if aggregate_decode_tps else 0.0
        ),
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
            int(row.get("batch_size", 1)),
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
            batch_size,
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
                "batch_size": int(batch_size),
            }
        )
        summaries.append(summary)
    return summaries

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
        "decode_step_wall_ms_p95": percentile(decode_step_ms, 95),
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


def run_prompt_batch_generate(
    llm: Any,
    prompt_tokens: list[int],
    *,
    batch_size: int,
    temperature: float,
    max_tokens: int,
    ignore_eos: bool,
    eos_token_id: int | None,
    max_model_len: int,
) -> dict[str, Any]:
    """Run a true synchronous batch and report per-request and aggregate TPOT."""
    from nanovllm import SamplingParams

    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
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
    finish_decode_sec: dict[int, float] = {}

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
            for seq_id, _ in step_outputs:
                finish_decode_sec[int(seq_id)] = decode_sec
        return step_outputs, num_tokens

    elapsed_start = perf_counter()
    llm.step = timed_step
    try:
        outputs = llm.generate(
            [list(prompt_tokens) for _ in range(batch_size)],
            [sampling for _ in range(batch_size)],
            use_tqdm=False,
        )
    finally:
        llm.step = original_step
    elapsed_sec = perf_counter() - elapsed_start

    if len(outputs) != batch_size:
        raise RuntimeError(
            f"expected {batch_size} output sequences, got {len(outputs)}"
        )
    token_rows = [list(output.get("token_ids", [])) for output in outputs]
    lengths = [len(row) for row in token_rows]
    intervals = [max(length - 1, 0) for length in lengths]
    total_intervals = sum(intervals)
    common_intervals = min(intervals, default=0)
    validation_errors = []
    if ignore_eos and max_tokens > 0:
        invalid = [index for index, length in enumerate(lengths) if length != max_tokens]
        if invalid:
            validation_errors.append(
                f"fixed_length_mismatch_rows={invalid} lengths={lengths} expected={max_tokens}"
            )

    # Sequence ids are monotonic and generate() returns them sorted.  The
    # completion timestamps have the same ordering for this isolated batch.
    finish_times = [finish_decode_sec[key] for key in sorted(finish_decode_sec)]
    if len(finish_times) != batch_size:
        finish_times = [decode_sec] * batch_size
    per_sequence_tpot_ms = [
        finish * 1000.0 / interval if interval else 0.0
        for finish, interval in zip(finish_times, intervals, strict=True)
    ]
    digest_payload = ";".join(
        ",".join(str(token_id) for token_id in row) for row in token_rows
    ).encode("utf-8")
    fixed_ok = not validation_errors
    generated_per_request = min(lengths, default=0)
    stopped_by = (
        "max_output_tokens"
        if fixed_ok and ignore_eos and max_tokens > 0
        else "finished"
    )
    # KTransformers' batch benchmark captures its graph on decode step two and
    # reports only the following 61 replay samples for a 64-token completion.
    # nano's primary TPOT below deliberately remains the full request-visible
    # wall time.  Keep a second, explicitly named stable-replay view so a
    # comparison does not silently mix 63 nano steps with 61 KT replay steps.
    stable_replay_step_ms = decode_step_ms[2:]
    stable_replay_sum_ms = sum(stable_replay_step_ms)
    stable_replay_mean_ms = (
        stable_replay_sum_ms / len(stable_replay_step_ms)
        if stable_replay_step_ms
        else 0.0
    )
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
        "decode_step_wall_ms_p95": percentile(decode_step_ms, 95),
        "decode_step_wall_ms_max": max(decode_step_ms) if decode_step_ms else 0.0,
        "stable_replay_decode_steps": len(stable_replay_step_ms),
        "stable_replay_step_wall_ms_mean": stable_replay_mean_ms,
        "stable_replay_step_wall_ms_p50": percentile(stable_replay_step_ms, 50),
        "stable_replay_step_wall_ms_p95": percentile(stable_replay_step_ms, 95),
        "stable_replay_step_wall_ms_max": (
            max(stable_replay_step_ms) if stable_replay_step_ms else 0.0
        ),
        # Preserve the short (normally 63-value) true-batch trace.  Mean and
        # percentiles alone cannot show whether a slowdown is cold-start or a
        # route-dependent tail later in generation.
        "decode_step_wall_ms_samples": list(decode_step_ms),
        "stable_replay_aggregate_tpot_ms_per_output_token": (
            stable_replay_mean_ms / batch_size if stable_replay_step_ms else 0.0
        ),
        "stable_replay_aggregate_decode_tok_s": (
            batch_size * 1000.0 / stable_replay_mean_ms
            if stable_replay_mean_ms > 0.0
            else 0.0
        ),
        "prefill_steps": prefill_steps,
        "decode_steps": decode_steps,
        "batch_size": batch_size,
        "generated_output_tokens": generated_per_request,
        "generated_output_tokens_total": sum(lengths),
        "decode_token_intervals": common_intervals,
        "decode_token_intervals_total": total_intervals,
        "output_sequence_count": len(token_rows),
        "output_fixed_length_ok": fixed_ok,
        "output_validation_error": ";".join(validation_errors),
        "max_repeated_token_run": max(
            (max_repeated_token_run(row) for row in token_rows), default=0
        ),
        # Wall TPOT is the slowest-batch completion time divided by one
        # request's token intervals.  This matches request-visible latency.
        "tpot_ms": (
            decode_sec * 1000.0 / common_intervals if common_intervals else 0.0
        ),
        "decode_tok_s": (
            common_intervals / decode_sec
            if decode_sec > 0.0 and common_intervals
            else 0.0
        ),
        "aggregate_tpot_ms_per_output_token": (
            decode_sec * 1000.0 / total_intervals if total_intervals else 0.0
        ),
        "aggregate_decode_tok_s": (
            total_intervals / decode_sec
            if decode_sec > 0.0 and total_intervals
            else 0.0
        ),
        "per_sequence_tpot_ms": per_sequence_tpot_ms,
        "throughput_output_tok_s": (
            sum(lengths) / elapsed_sec if elapsed_sec > 0.0 else 0.0
        ),
        "max_tokens_limit": int(max_tokens),
        "ignore_eos": bool(ignore_eos),
        "stopped_by": stopped_by,
        "outputs_digest": hashlib.sha256(digest_payload).hexdigest(),
        "generated_token_ids": token_rows,
    }

def reset_llm_profile(llm: Any) -> None:
    if not hasattr(llm, "get_profile"):
        return
    llm.get_profile(reset=True)

# TODO(transfer-aware-hotpath): restore the promotion gate to 19 ms after the
# boundary simulator and first-draft phase-1 path are optimized. The user
# explicitly approved 21 ms as a provisional gate for the active screen.
PROVISIONAL_STEADY_DRAFT_GATE_MS = 21.0

def steady_draft_call_stats(
    profile: dict[str, Any],
    *,
    gate_ms: float = PROVISIONAL_STEADY_DRAFT_GATE_MS,
    drop_initial_rounds: int = 1,
) -> dict[str, Any]:
    """Flatten steady draft calls after warmup reset and first-round removal."""
    traces = profile.get("spec_step_traces", [])
    if not isinstance(traces, list):
        traces = []
    steady_traces = traces[max(0, int(drop_initial_rounds)) :]
    calls = [
        float(value)
        for trace in steady_traces
        if isinstance(trace, dict)
        for value in trace.get("draft_call_ms", [])
        if isinstance(value, (int, float))
    ]
    mean_ms = sum(calls) / len(calls) if calls else 0.0
    return {
        "steady_draft_call_count": len(calls),
        "steady_draft_call_mean_ms": mean_ms,
        "steady_draft_call_p50_ms": percentile(calls, 50),
        "steady_draft_call_p90_ms": percentile(calls, 90),
        "steady_draft_gate_ms": float(gate_ms),
        "steady_draft_gate_passed": bool(calls and mean_ms < float(gate_ms)),
        "steady_draft_dropped_initial_rounds": max(
            0, int(drop_initial_rounds)
        ),
    }

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
    for key, value in steady_draft_call_stats(profile).items():
        metrics[f"profile_{key}"] = value
    return metrics
