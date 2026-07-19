from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import os
from time import perf_counter

import torch

from nanovllm.engine.sequence import SequenceStatus
from nanovllm.engine.speculative.acceptance import create_acceptance_strategy


def expected_tpot_ms(step_alphas, num_seqs: int, td_ms: float, tv_ms: float) -> float:
    """Expected per-output-token latency (TPOT) for a batched draft of length k.

    ``step_alphas`` is a list over the k draft steps; each element is a list of
    per-sequence acceptance estimates (length ``num_seqs``). For one sequence the
    expected accepted length is the cumulative-product sum
    ``acc_len = sum_i prod_{j<=i} e_j`` (a draft token lands only if every earlier
    token in the run was accepted). The batch produces ``acc_len_s + 1`` tokens per
    sequence per spec iteration (the ``+1`` is the guaranteed verify token), at a
    cost of ``k * td + tv``::

        T(k) = (k * td_ms + tv_ms) / sum_s (acc_len_s + 1)

    With ``k == 0`` this is the no-draft baseline ``tv_ms / num_seqs``.
    """
    if num_seqs <= 0:
        return float("inf")
    k = len(step_alphas)
    cumprod = [1.0] * num_seqs
    acc_len_sum = 0.0
    for alphas in step_alphas:
        for s in range(num_seqs):
            cumprod[s] *= float(alphas[s])
            acc_len_sum += cumprod[s]
    denom = acc_len_sum + num_seqs
    return (k * td_ms + tv_ms) / denom


class SpeculativeEngine:
    """Minimal Phase 2 speculative entry point.

    This baseline keeps decode semantics aligned with the existing model runner
    while the full Draft-Verify-Accept loop is integrated in subsequent steps.
    """

    def __init__(self, model_runner, scheduler, config):
        self.model_runner = model_runner
        self.scheduler = scheduler
        self.config = config
        self.max_draft_tokens = getattr(config, "max_draft_tokens", 5)
        strategy_name = getattr(config, "acceptance_strategy", "greedy")
        self.acceptance_strategy_name = str(strategy_name).strip().lower()
        threshold = getattr(config, "acceptance_threshold", 0.7)
        self.acceptance_strategy = create_acceptance_strategy(strategy_name, threshold=threshold)
        self.draft_alpha_stop_threshold = float(getattr(config, "draft_alpha_stop_threshold", -1.0))
        self.draft_stop_policy = str(getattr(config, "draft_stop_policy", "none")).strip().lower()
        self.draft_tpot_td_ms = float(getattr(config, "draft_tpot_td_ms", 19.0))
        self.draft_tpot_tv_ms = float(getattr(config, "draft_tpot_tv_ms", 80.0))
        self.draft_tpot_cost_model = str(getattr(config, "draft_tpot_cost_model", "static")).strip().lower()
        self.draft_tpot_history_alpha = float(getattr(config, "draft_tpot_history_alpha", 0.2))
        self.draft_tpot_min_steps = int(getattr(config, "draft_tpot_min_steps", 0))
        self.draft_tpot_stop_margin = float(getattr(config, "draft_tpot_stop_margin", 0.0))
        self.draft_tpot_stop_patience = max(
            1, int(getattr(config, "draft_tpot_stop_patience", 1))
        )
        self.draft_tpot_lookahead_cache_credit_ms_per_step = float(
            getattr(
                config,
                "draft_tpot_lookahead_cache_credit_ms_per_step",
                0.0,
            )
        )
        self.draft_tpot_short_verify_penalty_ms = float(
            getattr(config, "draft_tpot_short_verify_penalty_ms", 0.0)
        )
        self.draft_tpot_verify_cost_floor_ms = float(
            getattr(config, "draft_tpot_verify_cost_floor_ms", 0.0)
        )
        self.draft_tpot_stop_rule = str(getattr(config, "draft_tpot_stop_rule", "first_increase")).strip().lower()
        self.draft_tpot_verify_model_mode = str(
            getattr(config, "draft_tpot_verify_model_mode", "off")
        ).strip().lower()
        self._verify_cost_sampling_validated = bool(
            getattr(
                config,
                "draft_tpot_verify_model_sampling_validated",
                False,
            )
        )
        self._alpha_calibration = None
        alpha_calibration_path = str(
            getattr(config, "draft_tpot_alpha_calibration_path", "") or ""
        )
        if alpha_calibration_path:
            from nanovllm.engine.speculative.acceptance_calibration import (
                AcceptanceAlphaCalibration,
            )

            self._alpha_calibration = AcceptanceAlphaCalibration.load(
                alpha_calibration_path,
                acceptance_predictor_path=str(config.acceptance_predictor_path),
            )
            self._alpha_calibration.validate_acceptance_strategy(
                self.acceptance_strategy_name
            )
        self.profile_enabled = getattr(config, "spec_profile", False)
        self._profile = defaultdict(float)
        self._draft_steps_per_step: list[int] = []
        self._step_traces: list[dict] = []
        self._tpot_draft_ms_ema: float | None = None
        self._tpot_verify_ms_ema: float | None = None
        self._tpot_verify_ms_by_len: dict[int, float] = {}
        self.perfect_match_trace_enabled = os.getenv(
            "NANOVLLM_DRAFT_PERFECT_MATCH_TRACE", "0"
        ).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _calibrate_tpot_alpha(self, alpha: float) -> float:
        if self._alpha_calibration is None:
            return min(1.0, max(0.0, float(alpha)))
        return self._alpha_calibration.calibrate(alpha)

    def get_profile(self, reset: bool = False) -> dict:
        out = {k: (int(v) if k.endswith("_count") else float(v)) for k, v in self._profile.items()}
        # Keep a canonical phase2_post field so benchmark/report layers don't have
        # to infer draft-stage latency from internal counters.
        if "draft_ms" not in out:
            out["draft_ms"] = float(out.get("draft_loop_ms", 0.0))

        draft_calls = float(out.get("run_draft_calls", 0.0))
        if draft_calls > 0:
            out["draft_forward_ms"] = float(out.get("run_draft_infer_ms_total", 0.0) / draft_calls)

        verify_calls = float(out.get("run_verify_calls", 0.0))
        if verify_calls > 0:
            out["verify_forward_ms"] = float(out.get("run_verify_infer_ms_total", 0.0) / verify_calls)

        if self._tpot_draft_ms_ema is not None:
            out["draft_tpot_draft_ms_ema"] = float(self._tpot_draft_ms_ema)
        if self._tpot_verify_ms_ema is not None:
            out["draft_tpot_verify_ms_ema"] = float(self._tpot_verify_ms_ema)
        out["draft_tpot_verify_ms_by_len"] = dict(self._tpot_verify_ms_by_len)
        out["draft_steps_per_step"] = list(self._draft_steps_per_step)
        out["step_traces"] = deepcopy(self._step_traces)
        if reset:
            self._profile.clear()
            self._draft_steps_per_step.clear()
            self._step_traces.clear()
            self._tpot_draft_ms_ema = None
            self._tpot_verify_ms_ema = None
            self._tpot_verify_ms_by_len.clear()
        return out

    def _update_ema(self, current: float | None, value: float) -> float:
        alpha = min(1.0, max(1e-6, self.draft_tpot_history_alpha))
        value = float(value)
        return value if current is None else (1.0 - alpha) * float(current) + alpha * value

    def _record_tpot_draft_cost(self, draft_ms: float) -> None:
        if self.draft_tpot_cost_model != "history":
            return
        self._tpot_draft_ms_ema = self._update_ema(self._tpot_draft_ms_ema, max(0.0, float(draft_ms)))

    def _record_tpot_verify_cost(self, verify_tokens: int, verify_ms: float) -> None:
        if self.draft_tpot_cost_model != "history":
            return
        verify_tokens = max(1, int(verify_tokens))
        verify_ms = max(0.0, float(verify_ms))
        self._tpot_verify_ms_ema = self._update_ema(self._tpot_verify_ms_ema, verify_ms)
        self._tpot_verify_ms_by_len[verify_tokens] = self._update_ema(
            self._tpot_verify_ms_by_len.get(verify_tokens),
            verify_ms,
        )

    def _tpot_draft_cost_ms(self) -> float:
        if self.draft_tpot_cost_model == "history" and self._tpot_draft_ms_ema is not None:
            return max(1e-6, float(self._tpot_draft_ms_ema))
        return max(1e-6, self.draft_tpot_td_ms)

    def _tpot_verify_cost_ms(
        self,
        draft_len: int,
        predicted_verify_ms: float | None = None,
    ) -> float:
        if (
            self.draft_tpot_verify_model_mode == "active"
            and predicted_verify_ms is not None
        ):
            return max(1e-6, float(predicted_verify_ms))
        if self.draft_tpot_cost_model != "history":
            return max(1e-6, self.draft_tpot_tv_ms)
        verify_len = max(1, int(draft_len) + 1)
        observed = self._tpot_verify_ms_by_len.get(verify_len)
        if observed is None:
            observed = self._tpot_verify_ms_ema
        base = float(observed) if observed is not None else self.draft_tpot_tv_ms
        if self.draft_tpot_verify_cost_floor_ms > 0.0:
            base = max(base, self.draft_tpot_verify_cost_floor_ms)
        # Short verify segments have worse CPU expert batching and expose more
        # graph-outside metadata work; model that as an additive opportunity cost.
        full_draft = max(0, int(self.max_draft_tokens))
        missing_draft = max(0, full_draft - int(draft_len))
        base += self.draft_tpot_short_verify_penalty_ms * float(missing_draft)
        return max(1e-6, base)

    def _expected_tpot_ms_for_len(
        self,
        step_alphas,
        num_seqs: int,
        draft_len: int,
        predicted_verify_ms: float | None = None,
    ) -> float:
        return expected_tpot_ms(
            step_alphas,
            num_seqs,
            self._tpot_draft_cost_ms(),
            self._tpot_verify_cost_ms(draft_len, predicted_verify_ms),
        )

    def _budget_draft_steps(self, seqs) -> int:
        limits = [self.max_draft_tokens]
        for seq in seqs:
            max_tokens = getattr(seq, "max_tokens", None)
            if max_tokens is None:
                continue
            completion = seq.num_tokens - seq.num_prompt_tokens
            remaining = max_tokens - completion
            # Need one slot for verify-next token, so accepted draft max is remaining-1.
            limits.append(max(0, remaining - 1))
        return min(limits) if limits else self.max_draft_tokens

    def _tpot_bucket_boundaries(
        self,
        *,
        num_seqs: int,
        draft_steps: int,
    ) -> list[int]:
        """Return draft lengths that fill the configured verify graph buckets."""
        num_seqs = max(1, int(num_seqs))
        draft_steps = max(0, int(draft_steps))
        raw_buckets = getattr(self.config, "verify_cuda_graph_bucket_steps", ())
        boundaries = {
            int(bucket) // num_seqs - 1
            for bucket in raw_buckets
            if int(bucket) >= num_seqs
        }
        selected = sorted(
            value for value in boundaries if 1 <= value <= draft_steps
        )
        if not selected:
            selected = list(range(1, draft_steps + 1))
        elif draft_steps not in selected:
            selected.append(draft_steps)
        return selected

    def speculative_step(self, seqs):
        if not seqs:
            return []

        has_sampling = any(getattr(seq, "temperature", 1.0) > 1e-10 for seq in seqs)
        expected_verify_temperature = getattr(
            self.config,
            "draft_tpot_verify_model_temperature",
            None,
        )
        if (
            has_sampling
            and self._verify_cost_sampling_validated
            and expected_verify_temperature is not None
            and any(
                abs(float(getattr(seq, "temperature", 1.0)) - float(expected_verify_temperature))
                > 1e-6
                for seq in seqs
            )
        ):
            raise ValueError(
                "active verify cost model sampling temperature does not match "
                f"calibration temperature {float(expected_verify_temperature)}"
            )
        use_predicted_verify_cost = bool(
            not has_sampling or self._verify_cost_sampling_validated
        )
        use_sampling_accept = self.acceptance_strategy_name in {
            "standard_sampling",
            "sampling",
            "spec_sampling",
        }
        if has_sampling and not use_sampling_accept:
            return self.model_runner.call("run", seqs, False)
        return_logits = bool(has_sampling and use_sampling_accept)

        step_t0 = perf_counter()
        self._profile["spec_step_count"] += 1
        step_index = int(self._profile["spec_step_count"])

        draft_steps = self._budget_draft_steps(seqs)
        self._profile["draft_steps_total"] += draft_steps
        self._draft_steps_per_step.append(int(draft_steps))

        step_trace = {
            "step_index": step_index,
            "draft_steps": int(draft_steps),
            "seq_count": len(seqs),
            "sequences": [],
        }
        for seq in seqs:
            max_tokens = getattr(seq, "max_tokens", None)
            completion_before = seq.num_tokens - seq.num_prompt_tokens
            remaining_before = (max_tokens - completion_before) if max_tokens is not None else None
            step_trace["sequences"].append({
                "seq_id": int(seq.seq_id),
                "completion_before": int(completion_before),
                "max_tokens": int(max_tokens) if max_tokens is not None else None,
                "remaining_before": int(remaining_before) if remaining_before is not None else None,
                "drafted_tokens": 0,
                "verify_trace_len": 0,
                "accepted_draft_tokens": 0,
                "next_token": None,
            })

        t0 = perf_counter()
        for seq in seqs:
            seq.start_draft()
            self.scheduler.start_draft_kv(seq)
        self._profile["start_draft_ms"] += (perf_counter() - t0) * 1000.0

        draft_tokens_map = {seq.seq_id: [] for seq in seqs}
        draft_logits_map = {seq.seq_id: [] for seq in seqs}
        draft_alpha_map = {seq.seq_id: [] for seq in seqs}
        draft_calibrated_alpha_map = {seq.seq_id: [] for seq in seqs}
        draft_prefetch_state = None
        draft_steps_actual = 0

        # TPOT-policy running state (cost-aware dynamic draft length).
        tpot_alpha_history: list[list[float]] = []   # per-step per-seq alpha
        verify_cost_prediction_series: list[dict[str, object]] = []
        initial_verify_cost = None
        tpot_bucket_boundaries = (
            self._tpot_bucket_boundaries(
                num_seqs=len(seqs),
                draft_steps=draft_steps,
            )
            if self.draft_tpot_stop_rule == "bucket_lookahead"
            else []
        )
        track_verify_cost = (
            self.draft_tpot_verify_model_mode == "shadow"
            or (
                self.draft_tpot_verify_model_mode == "active"
                and self.draft_stop_policy == "tpot"
            )
        )
        if track_verify_cost:
            initial_verify_cost = self.model_runner.call(
                "start_verify_cost_round",
                len(seqs),
                self.draft_tpot_stop_rule != "bucket_lookahead",
            )
        initial_verify_prediction_ms = None
        if isinstance(initial_verify_cost, dict):
            initial_row = {
                key: value
                for key, value in initial_verify_cost.items()
                if key.startswith("verify_cost_")
            }
            initial_row["verify_cost_candidate_len"] = 0
            verify_cost_prediction_series.append(initial_row)
            if use_predicted_verify_cost:
                initial_verify_prediction_ms = float(
                    initial_verify_cost["verify_cost_prediction_ms"]
                )
        # T(0): no draft, just the guaranteed verify token (one per seq).
        tpot_prev = self._expected_tpot_ms_for_len(
            [],
            len(seqs),
            0,
            initial_verify_prediction_ms,
        )
        # A minimum draft length makes shorter points infeasible. Do not retain
        # an unreachable T(0..min-1) as the comparison baseline.
        tpot_best = tpot_prev if self.draft_tpot_min_steps <= 0 else None
        tpot_series: list[float] = []
        tpot_cost_series: list[dict[str, float]] = []
        tpot_stop_streak = 0
        draft_call_ms_series: list[float] = []

        t0 = perf_counter()
        for step_idx in range(draft_steps):
            infer_t0 = perf_counter()
            candidate_len = step_idx + 1
            lookahead_candidate_len = None
            if (
                self.draft_tpot_stop_rule == "bucket_lookahead"
                and candidate_len >= max(0, self.draft_tpot_min_steps)
                and candidate_len in tpot_bucket_boundaries
            ):
                lookahead_candidate_len = next(
                    (
                        boundary
                        for boundary in tpot_bucket_boundaries
                        if boundary > candidate_len
                    ),
                    None,
                )
            boundary_prediction_needed = lookahead_candidate_len is not None
            predict_verify_cost = (
                self.draft_tpot_verify_model_mode == "shadow"
                or (
                    track_verify_cost
                    and (
                        boundary_prediction_needed
                        if self.draft_tpot_stop_rule == "bucket_lookahead"
                        else (
                            candidate_len >= max(1, self.draft_tpot_min_steps)
                            and candidate_len < draft_steps
                        )
                    )
                )
            )
            lookahead_verify_tokens = (
                (int(lookahead_candidate_len) + 1) * len(seqs)
                if boundary_prediction_needed
                else None
            )
            draft_result = self.model_runner.call(
                "run_draft",
                seqs,
                return_logits,
                track_verify_cost,
                predict_verify_cost,
                lookahead_verify_tokens,
            )
            draft_call_ms = (perf_counter() - infer_t0) * 1000.0
            draft_call_ms_series.append(float(draft_call_ms))
            self._profile["run_draft_infer_ms_total"] += draft_call_ms
            self._record_tpot_draft_cost(draft_call_ms)
            draft_logits = None
            step_alpha = None
            calibrated_step_alpha = None
            verify_cost_prediction_ms = None
            if isinstance(draft_result, tuple):
                token_ids = draft_result[0]
                if len(draft_result) > 1 and isinstance(draft_result[1], dict):
                    draft_prefetch_state = draft_result[1]
                    step_alpha = draft_prefetch_state.get("acceptance_alpha")
                    if step_alpha is not None:
                        calibrated_step_alpha = [
                            self._calibrate_tpot_alpha(value) for value in step_alpha
                        ]
                    raw_prediction = draft_prefetch_state.get(
                        "verify_cost_prediction_ms"
                    )
                    if raw_prediction is not None:
                        verify_cost_prediction_ms = float(raw_prediction)
                        prediction_row = {
                            key: value
                            for key, value in draft_prefetch_state.items()
                            if key.startswith("verify_cost_")
                        }
                        prediction_row["verify_cost_candidate_len"] = step_idx + 1
                        verify_cost_prediction_series.append(prediction_row)
                elif len(draft_result) > 1 and isinstance(draft_result[1], torch.Tensor):
                    draft_logits = draft_result[1]
                if len(draft_result) > 2 and isinstance(draft_result[2], torch.Tensor):
                    draft_logits = draft_result[2]
            else:
                token_ids = draft_result

            for row_idx, (seq, token_id) in enumerate(zip(seqs, token_ids)):
                seq.append_draft_token(token_id)
                draft_tokens_map[seq.seq_id].append(token_id)
                if draft_logits is not None:
                    draft_logits_map[seq.seq_id].append(draft_logits[row_idx])
                if step_alpha is not None and row_idx < len(step_alpha):
                    draft_alpha_map[seq.seq_id].append(float(step_alpha[row_idx]))
                if (
                    calibrated_step_alpha is not None
                    and row_idx < len(calibrated_step_alpha)
                ):
                    draft_calibrated_alpha_map[seq.seq_id].append(
                        float(calibrated_step_alpha[row_idx])
                    )
            draft_steps_actual = step_idx + 1

            # Dynamic draft-length stop policy. No-op when alpha is unavailable
            # (predictor off) or doesn't cover the whole batch.
            stop = False
            if step_alpha is not None and len(step_alpha) == len(seqs):
                if (self.draft_stop_policy == "alpha_threshold"
                        and self.draft_alpha_stop_threshold >= 0):
                    if all(a < self.draft_alpha_stop_threshold for a in step_alpha):
                        self._profile["draft_alpha_early_stop_count"] += 1
                        stop = True
                elif self.draft_stop_policy == "tpot":
                    tpot_alpha_history.append(
                        [float(a) for a in (calibrated_step_alpha or step_alpha)]
                    )
                    candidate_len = len(tpot_alpha_history)
                    tpot_now = self._expected_tpot_ms_for_len(
                        tpot_alpha_history,
                        len(seqs),
                        candidate_len,
                        verify_cost_prediction_ms if use_predicted_verify_cost else None,
                    )
                    tpot_series.append(tpot_now)
                    tpot_cost_series.append({
                        "draft_ms": float(self._tpot_draft_cost_ms()),
                        "verify_ms": float(
                            self._tpot_verify_cost_ms(
                                candidate_len,
                                verify_cost_prediction_ms
                                if use_predicted_verify_cost
                                else None,
                            )
                        ),
                    })
                    can_stop = candidate_len >= max(0, self.draft_tpot_min_steps)
                    if self.draft_tpot_stop_rule == "bucket_lookahead":
                        lookahead_verify_ms = (
                            draft_prefetch_state.get(
                                "verify_cost_lookahead_prediction_ms"
                            )
                            if isinstance(draft_prefetch_state, dict)
                            else None
                        )
                        boundary_can_stop = bool(
                            can_stop
                            and lookahead_candidate_len is not None
                            and lookahead_verify_ms is not None
                            and use_predicted_verify_cost
                        )
                        should_stop = False
                        if boundary_can_stop:
                            projection_row = [
                                float(value)
                                for value in (
                                    calibrated_step_alpha or step_alpha
                                )
                            ]
                            projection_horizon = (
                                int(lookahead_candidate_len) - candidate_len
                            )
                            projected_alpha = tpot_alpha_history + [
                                list(projection_row)
                                for _ in range(projection_horizon)
                            ]
                            raw_lookahead_verify_ms = float(lookahead_verify_ms)
                            cache_credit_ms = min(
                                max(0.0, raw_lookahead_verify_ms - 1e-6),
                                self.draft_tpot_lookahead_cache_credit_ms_per_step
                                * float(projection_horizon),
                            )
                            adjusted_lookahead_verify_ms = max(
                                1e-6,
                                raw_lookahead_verify_ms - cache_credit_ms,
                            )
                            tpot_next = self._expected_tpot_ms_for_len(
                                projected_alpha,
                                len(seqs),
                                int(lookahead_candidate_len),
                                adjusted_lookahead_verify_ms,
                            )
                            reachable_best = (
                                tpot_now
                                if tpot_best is None
                                else min(float(tpot_best), tpot_now)
                            )
                            threshold = reachable_best * (
                                1.0 + self.draft_tpot_stop_margin
                            )
                            should_stop = tpot_next > threshold
                            tpot_best = reachable_best
                            tpot_cost_series[-1].update(
                                {
                                    "lookahead_tpot_ms": float(tpot_next),
                                    "lookahead_verify_ms": float(
                                        adjusted_lookahead_verify_ms
                                    ),
                                    "lookahead_verify_raw_ms": float(
                                        raw_lookahead_verify_ms
                                    ),
                                    "lookahead_cache_credit_ms": float(
                                        cache_credit_ms
                                    ),
                                    "lookahead_draft_len": float(
                                        lookahead_candidate_len
                                    ),
                                    "lookahead_horizon": float(
                                        projection_horizon
                                    ),
                                }
                            )
                        can_stop = boundary_can_stop
                    elif self.draft_tpot_stop_rule in {
                        "lookahead",
                        "lookahead_hysteresis",
                    }:
                        lookahead_verify_ms = (
                            draft_prefetch_state.get(
                                "verify_cost_lookahead_prediction_ms"
                            )
                            if isinstance(draft_prefetch_state, dict)
                            else None
                        )
                        if (
                            lookahead_verify_ms is not None
                            and candidate_len < draft_steps
                            and use_predicted_verify_cost
                        ):
                            projected_alpha = tpot_alpha_history + [
                                [
                                    float(value)
                                    for value in (
                                        calibrated_step_alpha or step_alpha
                                    )
                                ]
                            ]
                            tpot_next = self._expected_tpot_ms_for_len(
                                projected_alpha,
                                len(seqs),
                                candidate_len + 1,
                                float(lookahead_verify_ms),
                            )
                            reachable_best = (
                                tpot_now
                                if tpot_best is None
                                else min(float(tpot_best), tpot_now)
                            )
                            threshold = reachable_best * (
                                1.0 + self.draft_tpot_stop_margin
                            )
                            should_stop = bool(
                                can_stop and tpot_next > threshold
                            )
                            if can_stop:
                                tpot_best = reachable_best
                            tpot_cost_series[-1]["lookahead_tpot_ms"] = float(
                                tpot_next
                            )
                            tpot_cost_series[-1]["lookahead_verify_ms"] = float(
                                lookahead_verify_ms
                            )
                        else:
                            # A static baseline has no route-based next-step
                            # prediction. Preserve the legacy policy in that case
                            # instead of silently drafting the full budget.
                            threshold = tpot_prev * (
                                1.0 + self.draft_tpot_stop_margin
                            )
                            should_stop = tpot_now > threshold
                            if not (can_stop and should_stop):
                                tpot_prev = tpot_now
                    elif self.draft_tpot_stop_rule == "best_margin":
                        reachable_best = (
                            tpot_now
                            if tpot_best is None
                            else min(float(tpot_best), tpot_now)
                        )
                        threshold = reachable_best * (
                            1.0 + self.draft_tpot_stop_margin
                        )
                        should_stop = bool(can_stop and tpot_now > threshold)
                        if can_stop:
                            tpot_best = reachable_best
                    else:
                        threshold = tpot_prev * (1.0 + self.draft_tpot_stop_margin)
                        should_stop = tpot_now > threshold
                        if not (can_stop and should_stop):
                            tpot_prev = tpot_now
                    raw_should_stop = bool(should_stop)
                    if self.draft_tpot_stop_rule == "lookahead_hysteresis":
                        if can_stop and raw_should_stop:
                            tpot_stop_streak += 1
                        else:
                            tpot_stop_streak = 0
                        should_stop = (
                            tpot_stop_streak >= self.draft_tpot_stop_patience
                        )
                    else:
                        tpot_stop_streak = int(can_stop and raw_should_stop)
                    tpot_cost_series[-1].update(
                        {
                            "stop_signal": float(raw_should_stop),
                            "stop_streak": float(tpot_stop_streak),
                            "stop_patience": float(self.draft_tpot_stop_patience),
                            "stop_decision": float(can_stop and should_stop),
                        }
                    )
                    if can_stop and should_stop:
                        self._profile["draft_tpot_early_stop_count"] += 1
                        stop = True
            if stop:
                break

            # schedule() already reserved the first decode append slot.
            # For multi-draft decoding, reserve the next slot between iterations.
            if step_idx + 1 < draft_steps:
                for seq in seqs:
                    self.scheduler.append_draft_kv(seq)
        self._profile["draft_loop_ms"] += (perf_counter() - t0) * 1000.0
        self._profile["run_draft_calls"] += draft_steps_actual

        t0 = perf_counter()
        for seq in seqs:
            self.scheduler.rollback_draft_kv(seq)
            seq.rollback_tokens_to_draft_start()
        self._profile["rollback_ms"] += (perf_counter() - t0) * 1000.0

        # Prepare one-shot verify inputs: existing last token + all draft tokens.
        original_cached_tokens = {seq.seq_id: getattr(seq, "num_cached_tokens", 0) for seq in seqs}
        verify_lengths = []
        base_tokens_map = {seq.seq_id: list(seq.token_ids) for seq in seqs}
        t0 = perf_counter()
        for seq in seqs:
            draft_tokens = draft_tokens_map[seq.seq_id]
            for i, token_id in enumerate(draft_tokens):
                # Reuse the slot reserved by schedule() for the first token,
                # then reserve one extra slot before each subsequent token.
                if i > 0:
                    self.scheduler.append_draft_kv(seq)
                seq.append_token(token_id)
            if draft_tokens:
                # Verify consumes the final proposed token as an input too. The
                # draft loop only reserved storage through the previous input.
                self.scheduler.append_draft_kv(seq)
            # Recompute from last accepted token (num_tokens before draft) and drafts.
            seq.num_cached_tokens = seq._draft_start_num_tokens - 1
            verify_lengths.append(len(draft_tokens) + 1)
        self._profile["prepare_verify_ms"] += (perf_counter() - t0) * 1000.0

        if draft_prefetch_state is not None and "prefetch_step_id" in draft_prefetch_state:
            wait_prof = self.model_runner.call(
                "wait_prefetch_for_verify",
                draft_prefetch_state["prefetch_step_id"],
            )
            if isinstance(wait_prof, dict):
                for key, value in wait_prof.items():
                    self._profile[key] += float(value)

        verify_call_index = int(self._profile["run_verify_calls"])
        infer_t0 = perf_counter()
        verify_results = self.model_runner.call("run_verify", seqs, verify_lengths, return_logits)
        infer_ms = (perf_counter() - infer_t0) * 1000.0
        self._profile["verify_ms"] += infer_ms
        self._profile["run_verify_infer_ms_total"] += infer_ms
        self._profile["run_verify_calls"] += 1

        verify_results_map = {}
        for seq, verify_result in zip(seqs, verify_results):
            verify_results_map[seq.seq_id] = verify_result
            trace_len = int(verify_result.size(0)) if isinstance(verify_result, torch.Tensor) else len(verify_result)
            self._profile["verify_trace_tokens_total"] += trace_len

        final_token_ids = []
        perfect_match_outcome = None
        accept_t0 = perf_counter()
        for seq in seqs:
            draft_tokens = draft_tokens_map[seq.seq_id]
            verify_result = verify_results_map[seq.seq_id]
            draft_logits = (
                torch.stack(draft_logits_map[seq.seq_id], dim=0)
                if draft_logits_map[seq.seq_id]
                else None
            )
            accept_result = self.acceptance_strategy.accept(
                draft_tokens,
                verify_result,
                seq.temperature,
                draft_logits,
            )
            num_accepted = int(accept_result["num_accepted"])

            # Keep accepted draft prefix in KV, then append one verify token in token list.
            keep_after_start = num_accepted
            max_tokens = getattr(seq, "max_tokens", None)
            if max_tokens is not None:
                start_completion = seq._draft_start_num_tokens - seq.num_prompt_tokens
                remaining_budget = max_tokens - start_completion
                keep_after_start = max(0, min(keep_after_start, remaining_budget - 1))

            verify_trace_len = int(verify_result.size(0)) if isinstance(verify_result, torch.Tensor) else len(verify_result)
            if verify_trace_len == 0:
                seq.finish_draft()
                self._maybe_mark_finished(seq)
                final_token_ids.append(seq.last_token)
                seq.num_cached_tokens = original_cached_tokens[seq.seq_id]
                continue

            # Deterministic acceptors return traces; sampling acceptors return a sampled token.
            if return_logits:
                next_token = int(accept_result["next_token"])
            else:
                next_pos = min(keep_after_start, verify_trace_len - 1)
                next_token = verify_result[next_pos]

            for seq_trace in step_trace["sequences"]:
                if seq_trace["seq_id"] == int(seq.seq_id):
                    seq_trace["drafted_tokens"] = int(len(draft_tokens))
                    seq_trace["verify_trace_len"] = int(verify_trace_len)
                    seq_trace["accepted_draft_tokens"] = int(keep_after_start)
                    seq_trace["next_token"] = int(next_token)
                    seq_trace["acceptance_mode"] = str(accept_result.get("mode", self.acceptance_strategy_name))
                    seq_trace["rejected"] = bool(accept_result.get("rejected", keep_after_start < len(draft_tokens)))
                    reject_position = accept_result.get("reject_position")
                    seq_trace["reject_position"] = int(reject_position) if reject_position is not None else None
                    predicted_alpha = draft_alpha_map.get(seq.seq_id)
                    if predicted_alpha:
                        seq_trace["predicted_alpha"] = list(predicted_alpha)
                    calibrated_alpha = draft_calibrated_alpha_map.get(seq.seq_id)
                    if calibrated_alpha:
                        seq_trace["calibrated_alpha"] = list(calibrated_alpha)
                    accept_probs = accept_result.get("accept_probs")
                    if accept_probs is not None:
                        seq_trace["accept_probs"] = [float(x) for x in accept_probs]
                    break

            self.scheduler.accept_draft_kv(seq, keep_after_start)
            base_tokens = base_tokens_map[seq.seq_id]
            accepted_draft = draft_tokens[:keep_after_start]
            seq.token_ids = base_tokens + accepted_draft + [next_token]
            seq.num_tokens = len(seq.token_ids)
            seq.last_token = next_token
            seq.finish_draft()
            seq.num_cached_tokens = original_cached_tokens[seq.seq_id]

            final_token_ids.append(seq.last_token)
            self._maybe_mark_finished(seq)
            self._profile["accepted_tokens_total"] += keep_after_start
            self._profile["draft_tokens_total"] += len(draft_tokens)
            if self.perfect_match_trace_enabled and perfect_match_outcome is None:
                perfect_match_outcome = {
                    "step_index": int(step_index),
                    "seq_id": int(seq.seq_id),
                    "drafted_tokens": int(len(draft_tokens)),
                    "accepted_draft_tokens": int(keep_after_start),
                    "rejected_tokens": int(max(0, len(draft_tokens) - keep_after_start)),
                    "verify_trace_len": int(verify_trace_len),
                }

        accept_ms = (perf_counter() - accept_t0) * 1000.0
        verify_accept_ready_ms = (perf_counter() - infer_t0) * 1000.0
        self._record_tpot_verify_cost(sum(verify_lengths), verify_accept_ready_ms)
        self._profile["accept_ms"] += accept_ms
        self._profile["verify_accept_ready_ms"] += verify_accept_ready_ms

        if self.perfect_match_trace_enabled and perfect_match_outcome is not None:
            self.model_runner.call(
                "record_spec_acceptance_for_perfect_match",
                perfect_match_outcome,
            )

        finished_ids = [
            int(seq.seq_id)
            for seq in seqs
            if getattr(seq, "status", None) == SequenceStatus.FINISHED
        ]
        if finished_ids:
            self.model_runner.call("forget_acceptance_state", finished_ids)

        step_dt_ms = (perf_counter() - step_t0) * 1000.0
        self._profile["spec_step_ms"] += step_dt_ms
        step_trace["step_ms"] = step_dt_ms
        step_trace["draft_steps_actual"] = int(draft_steps_actual)
        step_trace["draft_stop_policy"] = self.draft_stop_policy
        step_trace["verify_call_index"] = int(verify_call_index)
        step_trace["verify_token_count"] = int(sum(verify_lengths))
        step_trace["verify_model_call_ms"] = float(infer_ms)
        step_trace["verify_accept_ms"] = float(accept_ms)
        step_trace["verify_accept_ready_ms"] = float(verify_accept_ready_ms)
        step_trace["draft_call_ms"] = list(draft_call_ms_series)
        if verify_cost_prediction_series:
            step_trace["verify_cost_predictions"] = deepcopy(
                verify_cost_prediction_series
            )
            last_prediction_row = verify_cost_prediction_series[-1]
            step_trace["verify_cost_last_prediction_ms"] = float(
                last_prediction_row["verify_cost_prediction_ms"]
            )
            step_trace["verify_cost_last_prediction_candidate_len"] = int(
                last_prediction_row["verify_cost_candidate_len"]
            )
            realized_prediction = next(
                (
                    row
                    for row in reversed(verify_cost_prediction_series)
                    if int(row["verify_cost_candidate_len"])
                    == int(draft_steps_actual)
                ),
                None,
            )
            if realized_prediction is not None:
                prediction_ms = float(
                    realized_prediction["verify_cost_prediction_ms"]
                )
                step_trace["verify_cost_prediction_ms"] = prediction_ms
                step_trace["verify_cost_prediction_error_ms"] = (
                    prediction_ms - float(verify_accept_ready_ms)
                )
                step_trace["verify_cost_prediction_abs_error_ms"] = abs(
                    prediction_ms - float(verify_accept_ready_ms)
                )
        if tpot_series:
            step_trace["draft_tpot"] = list(tpot_series)
            step_trace["draft_tpot_costs"] = list(tpot_cost_series)
            step_trace["draft_tpot_cost_model"] = self.draft_tpot_cost_model
            step_trace["draft_tpot_stop_rule"] = self.draft_tpot_stop_rule
            step_trace["draft_tpot_verify_model_mode"] = (
                self.draft_tpot_verify_model_mode
            )
            if self._alpha_calibration is not None:
                step_trace["draft_tpot_alpha_calibration_id"] = (
                    self._alpha_calibration.calibration_id
                )
        self._step_traces.append(step_trace)

        return final_token_ids

    def _maybe_mark_finished(self, seq):
        eos = getattr(self.scheduler, "eos", -1)
        ignore_eos = getattr(seq, "ignore_eos", False)
        max_tokens = getattr(seq, "max_tokens", None)
        num_completion_tokens = getattr(seq, "num_completion_tokens", 0)

        reached_eos = (not ignore_eos) and seq.last_token == eos
        reached_max = (max_tokens is not None) and (num_completion_tokens >= max_tokens)
        if reached_eos or reached_max:
            seq.status = SequenceStatus.FINISHED
            if hasattr(self.scheduler, "block_manager"):
                self.scheduler.block_manager.deallocate(seq)
            if hasattr(self.scheduler, "running") and seq in self.scheduler.running:
                self.scheduler.running.remove(seq)
