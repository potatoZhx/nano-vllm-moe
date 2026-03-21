from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from time import perf_counter

from nanovllm.engine.sequence import SequenceStatus


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
        self.profile_enabled = getattr(config, "spec_profile", False)
        self._profile = defaultdict(float)
        self._draft_steps_per_step: list[int] = []
        self._step_traces: list[dict] = []

    def get_profile(self, reset: bool = False) -> dict:
        out = {k: (int(v) if k.endswith("_count") else float(v)) for k, v in self._profile.items()}
        out["draft_steps_per_step"] = list(self._draft_steps_per_step)
        out["step_traces"] = deepcopy(self._step_traces)
        if reset:
            self._profile.clear()
            self._draft_steps_per_step.clear()
            self._step_traces.clear()
        return out

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

    def speculative_step(self, seqs):
        if not seqs:
            return []

        # Sampling mode fallback: keep baseline decode semantics for now.
        if any(getattr(seq, "temperature", 1.0) > 1e-10 for seq in seqs):
            return self.model_runner.call("run", seqs, False)

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
        t0 = perf_counter()
        for _ in range(draft_steps):
            infer_t0 = perf_counter()
            draft_result = self.model_runner.call("run_draft", seqs)
            self._profile["run_draft_infer_ms_total"] += (perf_counter() - infer_t0) * 1000.0
            if isinstance(draft_result, tuple):
                token_ids = draft_result[0]
            else:
                token_ids = draft_result

            for seq, token_id in zip(seqs, token_ids):
                seq.append_draft_token(token_id)
                draft_tokens_map[seq.seq_id].append(token_id)
                self.scheduler.append_draft_kv(seq)
        self._profile["draft_loop_ms"] += (perf_counter() - t0) * 1000.0
        self._profile["run_draft_calls"] += draft_steps

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
            for token_id in draft_tokens:
                seq.append_token(token_id)
                self.scheduler.append_draft_kv(seq)
            # Recompute from last accepted token (num_tokens before draft) and drafts.
            seq.num_cached_tokens = seq._draft_start_num_tokens - 1
            verify_lengths.append(len(draft_tokens) + 1)
        self._profile["prepare_verify_ms"] += (perf_counter() - t0) * 1000.0

        infer_t0 = perf_counter()
        verify_traces = self.model_runner.call("run_verify", seqs, verify_lengths)
        infer_ms = (perf_counter() - infer_t0) * 1000.0
        self._profile["verify_ms"] += infer_ms
        self._profile["run_verify_infer_ms_total"] += infer_ms
        self._profile["run_verify_calls"] += 1

        verify_tokens_map = {}
        for seq, trace in zip(seqs, verify_traces):
            verify_tokens_map[seq.seq_id] = trace
            self._profile["verify_trace_tokens_total"] += len(trace)

        final_token_ids = []
        t0 = perf_counter()
        for seq in seqs:
            draft_tokens = draft_tokens_map[seq.seq_id]
            verify_tokens = verify_tokens_map[seq.seq_id]
            num_accepted = 0
            for draft_tok, verify_tok in zip(draft_tokens, verify_tokens):
                if draft_tok != verify_tok:
                    break
                num_accepted += 1

            # Keep accepted draft prefix in KV, then append one verify token in token list.
            keep_after_start = num_accepted
            max_tokens = getattr(seq, "max_tokens", None)
            if max_tokens is not None:
                start_completion = seq._draft_start_num_tokens - seq.num_prompt_tokens
                remaining_budget = max_tokens - start_completion
                keep_after_start = max(0, min(keep_after_start, remaining_budget - 1))

            if len(verify_tokens) == 0:
                seq.finish_draft()
                self._maybe_mark_finished(seq)
                final_token_ids.append(seq.last_token)
                seq.num_cached_tokens = original_cached_tokens[seq.seq_id]
                continue

            # Next token must follow the actually kept draft prefix.
            next_pos = min(keep_after_start, len(verify_tokens) - 1)
            next_token = verify_tokens[next_pos]

            for seq_trace in step_trace["sequences"]:
                if seq_trace["seq_id"] == int(seq.seq_id):
                    seq_trace["drafted_tokens"] = int(len(draft_tokens))
                    seq_trace["verify_trace_len"] = int(len(verify_tokens))
                    seq_trace["accepted_draft_tokens"] = int(keep_after_start)
                    seq_trace["next_token"] = int(next_token)
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

        self._profile["accept_ms"] += (perf_counter() - t0) * 1000.0
        step_dt_ms = (perf_counter() - step_t0) * 1000.0
        self._profile["spec_step_ms"] += step_dt_ms
        step_trace["step_ms"] = step_dt_ms
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
