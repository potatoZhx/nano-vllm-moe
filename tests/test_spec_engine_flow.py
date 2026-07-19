import unittest
from types import SimpleNamespace

import pytest
import torch

from nanovllm.engine.speculative.spec_engine import SpeculativeEngine


class _Seq:
    def __init__(self, seq_id: int, token_ids: list[int], temperature: float = 0.0, max_tokens: int = 16):
        self.seq_id = seq_id
        self.token_ids = list(token_ids)
        self.num_tokens = len(self.token_ids)
        self.num_cached_tokens = 0
        self.last_token = self.token_ids[-1]
        self._draft_start_num_tokens = self.num_tokens
        self.draft_token_ids = []
        self.is_drafting = False
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_prompt_tokens = len(token_ids)
        self.ignore_eos = False
        self.is_finished = False

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.num_tokens += 1
        self.last_token = token_id

    def start_draft(self):
        self._draft_start_num_tokens = self.num_tokens
        self.draft_token_ids = []
        self.is_drafting = True

    def append_draft_token(self, token_id: int):
        self.draft_token_ids.append(token_id)
        self.append_token(token_id)

    def rollback_tokens_to_draft_start(self):
        self.token_ids = self.token_ids[:self._draft_start_num_tokens]
        self.num_tokens = len(self.token_ids)
        self.last_token = self.token_ids[-1]

    def finish_draft(self):
        self.is_drafting = False
        self.draft_token_ids = []


class _DummyScheduler:
    def __init__(self):
        self.ops = []
        self.eos = -1
        self.running = []

    def start_draft_kv(self, seq):
        self.ops.append(("start", seq.seq_id))

    def append_draft_kv(self, seq):
        self.ops.append(("append", seq.seq_id))

    def rollback_draft_kv(self, seq):
        self.ops.append(("rollback", seq.seq_id))

    def accept_draft_kv(self, seq, num_accepted):
        self.ops.append(("accept", seq.seq_id, num_accepted))


class _DummyModelRunner:
    def __init__(self):
        self.draft_calls = 0
        self.verify_calls = 0
        self.last_verify_lengths = None
        self.wait_calls = 0

    def call(self, name, *args):
        if name == "run_draft":
            seqs = args[0]
            self.draft_calls += 1
            if self.draft_calls == 1:
                return [11 for _ in seqs], {"prefetch_step_id": 1}
            return [12 for _ in seqs], {"prefetch_step_id": 1}
        if name == "wait_prefetch_for_verify":
            self.wait_calls += 1
            return {"verify_prefetch_wait_ms": 0.1}
        if name == "run_verify":
            seqs = args[0]
            self.verify_calls += 1
            self.last_verify_lengths = list(args[1]) if len(args) > 1 else None
            return [[11, 12, 99] for _ in seqs]
        if name == "forget_acceptance_state":
            return None
        raise RuntimeError(name)


class _TpotModelRunner(_DummyModelRunner):
    def __init__(self, *, alpha: float, include_lookahead: bool):
        super().__init__()
        self.alpha = float(alpha)
        self.include_lookahead = bool(include_lookahead)
        self.verify_cost_controls = []
        self.verify_cost_lookahead_tokens = []
        self.start_verify_cost_controls = []

    def call(self, name, *args):
        if name == "start_verify_cost_round":
            self.start_verify_cost_controls.append(
                bool(args[1]) if len(args) > 1 else True
            )
            return {"verify_cost_prediction_ms": 80.0}
        if name == "run_draft":
            seqs = args[0]
            self.draft_calls += 1
            observe_verify_cost = bool(args[2]) if len(args) > 2 else True
            predict_verify_cost = bool(args[3]) if len(args) > 3 else True
            self.verify_cost_controls.append(
                (observe_verify_cost, predict_verify_cost)
            )
            self.verify_cost_lookahead_tokens.append(
                args[4] if len(args) > 4 else None
            )
            state = {
                "prefetch_step_id": self.draft_calls,
                "acceptance_alpha": [self.alpha for _ in seqs],
            }
            if predict_verify_cost:
                state["verify_cost_prediction_ms"] = 80.0
            if self.include_lookahead and predict_verify_cost:
                state["verify_cost_lookahead_prediction_ms"] = 200.0
            return [10 + self.draft_calls for _ in seqs], state
        return super().call(name, *args)


class _LowInitialTpotModelRunner(_TpotModelRunner):
    def call(self, name, *args):
        if name == "start_verify_cost_round":
            self.start_verify_cost_controls.append(
                bool(args[1]) if len(args) > 1 else True
            )
            return {"verify_cost_prediction_ms": 10.0}
        result = super().call(name, *args)
        if name == "run_draft" and isinstance(result, tuple):
            result[1]["verify_cost_lookahead_prediction_ms"] = 80.0
        return result


class _SamplingModelRunner:
    def __init__(self):
        self.calls = []

    def call(self, name, *args):
        self.calls.append((name, args))
        if name == "run":
            raise AssertionError("standard_sampling should not fall back to baseline sampling")
        if name == "run_draft":
            seqs = args[0]
            return_logits = bool(args[1]) if len(args) > 1 else False
            self.assert_return_logits = return_logits
            logits = torch.full((len(seqs), 16), -1000.0)
            logits[:, 11] = 0.0
            return [11 for _ in seqs], {"prefetch_step_id": 3}, logits
        if name == "wait_prefetch_for_verify":
            return {"verify_prefetch_wait_ms": 0.0}
        if name == "run_verify":
            seqs = args[0]
            return_logits = bool(args[2]) if len(args) > 2 else False
            self.assert_verify_return_logits = return_logits
            logits = torch.full((2, 16), -1000.0)
            logits[0, 11] = 0.0
            logits[1, 13] = 0.0
            return [logits for _ in seqs]
        raise RuntimeError(name)


class _SamplingLegacyDraftTupleModelRunner(_SamplingModelRunner):
    def call(self, name, *args):
        result = super().call(name, *args)
        if name == "run_draft":
            token_ids, _prefetch_state, logits = result
            return token_ids, logits
        return result


class _SamplingTpotModelRunner(_SamplingModelRunner):
    def __init__(self):
        super().__init__()
        self.draft_calls = 0

    def call(self, name, *args):
        if name == "start_verify_cost_round":
            return {"verify_cost_prediction_ms": 80.0}
        if name == "run_draft":
            seqs = args[0]
            self.draft_calls += 1
            logits = torch.full((len(seqs), 16), -1000.0)
            logits[:, 11] = 0.0
            return [11 for _ in seqs], {
                "prefetch_step_id": self.draft_calls,
                "acceptance_alpha": [0.9 for _ in seqs],
                "verify_cost_prediction_ms": 80.0,
                "verify_cost_lookahead_prediction_ms": 200.0,
            }, logits
        return super().call(name, *args)


class _TransferAwareTpotModelRunner(_DummyModelRunner):
    def __init__(self, *, state_complete=True, next_ms=500.0, error_ms=0.0):
        super().__init__()
        self.state_complete = bool(state_complete)
        self.next_ms = float(next_ms)
        self.error_ms = float(error_ms)
        self.controls = []
        self.next_draft_ms = []

    def call(self, name, *args):
        if name == "start_verify_cost_round":
            return None
        if name == "run_draft":
            seqs = args[0]
            self.draft_calls += 1
            predict = bool(args[3]) if len(args) > 3 else True
            self.controls.append(predict)
            self.next_draft_ms.append(args[5] if len(args) > 5 else None)
            state = {
                "prefetch_step_id": self.draft_calls,
                "acceptance_alpha": [0.9 for _ in seqs],
            }
            if predict:
                state.update(
                    {
                        "verify_cost_prediction_ms": 10.0,
                        "verify_cost_error_p90_ms": self.error_ms,
                        "verify_cost_lookahead_prediction_ms": self.next_ms,
                        "verify_cost_lookahead_error_p90_ms": self.error_ms,
                        "verify_cost_state_complete": self.state_complete,
                    }
                )
            return [10 + self.draft_calls for _ in seqs], state
        return super().call(name, *args)


class TestSpecEngineFlow(unittest.TestCase):
    def test_draft_verify_accept_flow(self):
        seq = _Seq(seq_id=1, token_ids=[1, 2, 3], temperature=0.0)
        scheduler = _DummyScheduler()
        scheduler.running = [seq]
        model_runner = _DummyModelRunner()
        config = SimpleNamespace(max_draft_tokens=2, acceptance_strategy="greedy", acceptance_threshold=0.7)

        engine = SpeculativeEngine(model_runner=model_runner, scheduler=scheduler, config=config)
        token_ids = engine.speculative_step([seq])

        self.assertEqual(token_ids, [99])
        self.assertEqual(model_runner.verify_calls, 1)
        self.assertEqual(model_runner.wait_calls, 1)
        self.assertEqual(seq.last_token, 99)
        self.assertEqual(seq.token_ids, [1, 2, 3, 11, 12, 99])
        self.assertFalse(seq.is_drafting)
        self.assertIn(("start", 1), scheduler.ops)
        self.assertIn(("rollback", 1), scheduler.ops)
        self.assertIn(("accept", 1, 2), scheduler.ops)
        append_ops = [x for x in scheduler.ops if x[0] == "append"]
        # One append is needed between draft iterations and two more reserve
        # KV slots for both draft tokens when verify replays the full trace.
        self.assertEqual(len(append_ops), 3)

    def test_accept_next_token_uses_clamped_keep_prefix(self):
        seq = _Seq(seq_id=1, token_ids=[1, 2, 3], temperature=0.0, max_tokens=2)
        scheduler = _DummyScheduler()
        scheduler.running = [seq]
        model_runner = _DummyModelRunner()
        config = SimpleNamespace(max_draft_tokens=4, acceptance_strategy="greedy", acceptance_threshold=0.7)

        engine = SpeculativeEngine(model_runner=model_runner, scheduler=scheduler, config=config)
        token_ids = engine.speculative_step([seq])

        # remaining budget = 2, so only 1 draft token can be kept before verify-next.
        self.assertEqual(token_ids, [12])
        self.assertEqual(seq.token_ids, [1, 2, 3, 11, 12])
        self.assertIn(("accept", 1, 1), scheduler.ops)

    def test_draft_steps_are_limited_by_remaining_budget(self):
        seq = _Seq(seq_id=1, token_ids=[1, 2, 3], temperature=0.0, max_tokens=2)
        scheduler = _DummyScheduler()
        scheduler.running = [seq]
        model_runner = _DummyModelRunner()
        config = SimpleNamespace(max_draft_tokens=8, acceptance_strategy="greedy", acceptance_threshold=0.7)

        engine = SpeculativeEngine(model_runner=model_runner, scheduler=scheduler, config=config)
        engine.speculative_step([seq])

        # Only one draft iteration is useful under this budget.
        self.assertEqual(model_runner.draft_calls, 1)
        self.assertEqual(model_runner.verify_calls, 1)
        self.assertEqual(model_runner.last_verify_lengths, [2])

    def test_tpot_lookahead_stops_before_predicted_expensive_step(self):
        seq = _Seq(seq_id=1, token_ids=[1, 2, 3], temperature=0.0)
        scheduler = _DummyScheduler()
        scheduler.running = [seq]
        model_runner = _TpotModelRunner(alpha=0.9, include_lookahead=True)
        config = SimpleNamespace(
            max_draft_tokens=4,
            acceptance_strategy="greedy",
            acceptance_threshold=0.7,
            acceptance_predictor_enabled=True,
            draft_stop_policy="tpot",
            draft_tpot_stop_rule="lookahead",
            draft_tpot_verify_model_mode="active",
        )

        engine = SpeculativeEngine(model_runner, scheduler, config)
        engine.speculative_step([seq])

        self.assertEqual(model_runner.draft_calls, 1)
        trace = engine.get_profile()["step_traces"][0]
        self.assertEqual(trace["draft_steps_actual"], 1)
        self.assertGreater(
            trace["draft_tpot_costs"][0]["lookahead_tpot_ms"],
            trace["draft_tpot"][0],
        )

    def test_tpot_lookahead_uses_calibrated_alpha_for_projection(self):
        seq = _Seq(seq_id=1, token_ids=[1, 2, 3], temperature=0.0)
        scheduler = _DummyScheduler()
        scheduler.running = [seq]
        model_runner = _TpotModelRunner(alpha=0.9, include_lookahead=True)
        config = SimpleNamespace(
            max_draft_tokens=4,
            acceptance_strategy="greedy",
            acceptance_threshold=0.7,
            acceptance_predictor_enabled=True,
            draft_stop_policy="tpot",
            draft_tpot_stop_rule="lookahead",
            draft_tpot_verify_model_mode="active",
        )

        engine = SpeculativeEngine(model_runner, scheduler, config)
        engine._alpha_calibration = SimpleNamespace(
            calibration_id="test-calibration",
            calibrate=lambda _value: 0.2,
        )
        engine.speculative_step([seq])

        cost = engine.get_profile()["step_traces"][0]["draft_tpot_costs"][0]
        self.assertAlmostEqual(
            cost["lookahead_tpot_ms"],
            (2.0 * 19.0 + 200.0) / (1.0 + 0.2 + 0.2 * 0.2),
            places=6,
        )

    def test_tpot_lookahead_without_prediction_falls_back_to_first_increase(self):
        seq = _Seq(seq_id=1, token_ids=[1, 2, 3], temperature=0.0)
        scheduler = _DummyScheduler()
        scheduler.running = [seq]
        model_runner = _TpotModelRunner(alpha=0.05, include_lookahead=False)
        config = SimpleNamespace(
            max_draft_tokens=4,
            acceptance_strategy="greedy",
            acceptance_threshold=0.7,
            acceptance_predictor_enabled=True,
            draft_stop_policy="tpot",
            draft_tpot_stop_rule="lookahead",
            draft_tpot_verify_model_mode="off",
        )

        engine = SpeculativeEngine(model_runner, scheduler, config)
        engine.speculative_step([seq])

        self.assertEqual(model_runner.draft_calls, 1)
        trace = engine.get_profile()["step_traces"][0]
        self.assertNotIn("lookahead_tpot_ms", trace["draft_tpot_costs"][0])

    def test_active_verify_prediction_is_deferred_until_min_steps(self):
        seq = _Seq(seq_id=1, token_ids=[1, 2, 3], temperature=0.0)
        scheduler = _DummyScheduler()
        scheduler.running = [seq]
        model_runner = _TpotModelRunner(alpha=0.9, include_lookahead=True)
        config = SimpleNamespace(
            max_draft_tokens=4,
            acceptance_strategy="greedy",
            acceptance_threshold=0.7,
            acceptance_predictor_enabled=True,
            draft_stop_policy="tpot",
            draft_tpot_stop_rule="lookahead",
            draft_tpot_verify_model_mode="active",
            draft_tpot_min_steps=3,
        )

        engine = SpeculativeEngine(model_runner, scheduler, config)
        engine.speculative_step([seq])

        self.assertEqual(
            model_runner.verify_cost_controls,
            [(True, False), (True, False), (True, True)],
        )
        self.assertEqual(model_runner.draft_calls, 3)

    def test_active_verify_prediction_is_skipped_at_final_draft_step(self):
        seq = _Seq(seq_id=1, token_ids=[1, 2, 3], temperature=0.0)
        scheduler = _DummyScheduler()
        scheduler.running = [seq]
        model_runner = _TpotModelRunner(alpha=0.99, include_lookahead=False)
        config = SimpleNamespace(
            max_draft_tokens=3,
            acceptance_strategy="greedy",
            acceptance_threshold=0.7,
            acceptance_predictor_enabled=True,
            draft_stop_policy="tpot",
            draft_tpot_stop_rule="lookahead",
            draft_tpot_verify_model_mode="active",
            draft_tpot_min_steps=2,
        )

        engine = SpeculativeEngine(model_runner, scheduler, config)
        engine.speculative_step([seq])

        self.assertEqual(
            model_runner.verify_cost_controls,
            [(True, False), (True, True), (True, False)],
        )

    def test_min_steps_excludes_unreachable_t0_from_lookahead_baseline(self):
        seq = _Seq(seq_id=1, token_ids=[1, 2, 3], temperature=0.0)
        scheduler = _DummyScheduler()
        scheduler.running = [seq]
        model_runner = _LowInitialTpotModelRunner(
            alpha=0.99,
            include_lookahead=True,
        )
        config = SimpleNamespace(
            max_draft_tokens=4,
            acceptance_strategy="greedy",
            acceptance_threshold=0.7,
            acceptance_predictor_enabled=True,
            draft_stop_policy="tpot",
            draft_tpot_stop_rule="lookahead",
            draft_tpot_verify_model_mode="active",
            draft_tpot_min_steps=3,
        )

        engine = SpeculativeEngine(model_runner, scheduler, config)
        engine.speculative_step([seq])

        # The artificial T(0)=10 ms point is below every feasible K>=3 point.
        # It must not force a stop at the minimum length.
        self.assertEqual(model_runner.draft_calls, 4)
        trace = engine.get_profile()["step_traces"][0]
        self.assertEqual(trace["verify_cost_last_prediction_candidate_len"], 3)
        self.assertNotIn("verify_cost_prediction_error_ms", trace)

    def test_bucket_lookahead_targets_next_efficient_verify_boundary(self):
        seq = _Seq(seq_id=1, token_ids=[1, 2, 3], temperature=0.0)
        scheduler = _DummyScheduler()
        scheduler.running = [seq]
        model_runner = _TpotModelRunner(alpha=0.9, include_lookahead=True)
        config = SimpleNamespace(
            max_draft_tokens=12,
            acceptance_strategy="greedy",
            acceptance_threshold=0.7,
            acceptance_predictor_enabled=True,
            draft_stop_policy="tpot",
            draft_tpot_stop_rule="bucket_lookahead",
            draft_tpot_verify_model_mode="active",
            draft_tpot_min_steps=6,
            draft_tpot_lookahead_cache_credit_ms_per_step=8.5,
            verify_cuda_graph_bucket_steps=[3, 5, 7, 10, 13],
        )

        engine = SpeculativeEngine(model_runner, scheduler, config)
        engine.speculative_step([seq])

        self.assertEqual(model_runner.start_verify_cost_controls, [False])
        self.assertEqual(model_runner.draft_calls, 6)
        self.assertEqual(
            model_runner.verify_cost_controls,
            [(True, False)] * 5 + [(True, True)],
        )
        self.assertEqual(
            model_runner.verify_cost_lookahead_tokens,
            [None] * 5 + [10],
        )
        trace = engine.get_profile()["step_traces"][0]
        boundary_cost = trace["draft_tpot_costs"][-1]
        self.assertEqual(boundary_cost["lookahead_draft_len"], 9.0)
        self.assertEqual(boundary_cost["lookahead_horizon"], 3.0)
        self.assertEqual(boundary_cost["lookahead_verify_raw_ms"], 200.0)
        self.assertEqual(boundary_cost["lookahead_cache_credit_ms"], 25.5)
        self.assertEqual(boundary_cost["lookahead_verify_ms"], 174.5)

    def test_lookahead_hysteresis_requires_consecutive_stop_signals(self):
        seq = _Seq(seq_id=1, token_ids=[1, 2, 3], temperature=0.0)
        scheduler = _DummyScheduler()
        scheduler.running = [seq]
        model_runner = _TpotModelRunner(alpha=0.9, include_lookahead=True)
        config = SimpleNamespace(
            max_draft_tokens=4,
            acceptance_strategy="greedy",
            acceptance_threshold=0.7,
            acceptance_predictor_enabled=True,
            draft_stop_policy="tpot",
            draft_tpot_stop_rule="lookahead_hysteresis",
            draft_tpot_verify_model_mode="active",
            draft_tpot_min_steps=1,
            draft_tpot_stop_patience=2,
        )

        engine = SpeculativeEngine(model_runner, scheduler, config)
        engine.speculative_step([seq])

        self.assertEqual(model_runner.draft_calls, 2)
        costs = engine.get_profile()["step_traces"][0]["draft_tpot_costs"]
        self.assertEqual(costs[0]["stop_streak"], 1.0)
        self.assertEqual(costs[0]["stop_decision"], 0.0)
        self.assertEqual(costs[1]["stop_streak"], 2.0)
        self.assertEqual(costs[1]["stop_decision"], 1.0)

    def test_standard_sampling_uses_logits_for_sampling_temperature(self):
        seq = _Seq(seq_id=1, token_ids=[1, 2, 3], temperature=1.0, max_tokens=4)
        scheduler = _DummyScheduler()
        scheduler.running = [seq]
        model_runner = _SamplingModelRunner()
        config = SimpleNamespace(
            max_draft_tokens=1,
            acceptance_strategy="standard_sampling",
            acceptance_threshold=0.7,
        )

        engine = SpeculativeEngine(model_runner=model_runner, scheduler=scheduler, config=config)
        token_ids = engine.speculative_step([seq])

        self.assertEqual(token_ids, [13])
        self.assertEqual(seq.token_ids, [1, 2, 3, 11, 13])
        self.assertTrue(model_runner.assert_return_logits)
        self.assertTrue(model_runner.assert_verify_return_logits)
        self.assertNotIn("run", [name for name, _ in model_runner.calls])

    def test_standard_sampling_accepts_legacy_draft_logits_tuple(self):
        seq = _Seq(seq_id=1, token_ids=[1, 2, 3], temperature=1.0, max_tokens=4)
        scheduler = _DummyScheduler()
        scheduler.running = [seq]
        model_runner = _SamplingLegacyDraftTupleModelRunner()
        config = SimpleNamespace(
            max_draft_tokens=1,
            acceptance_strategy="standard_sampling",
            acceptance_threshold=0.7,
        )

        engine = SpeculativeEngine(model_runner=model_runner, scheduler=scheduler, config=config)
        token_ids = engine.speculative_step([seq])

        self.assertEqual(token_ids, [13])
        self.assertEqual(seq.token_ids, [1, 2, 3, 11, 13])

    def test_validated_sampling_uses_verify_cost_lookahead(self):
        seq = _Seq(seq_id=1, token_ids=[1, 2, 3], temperature=0.8, max_tokens=8)
        scheduler = _DummyScheduler()
        scheduler.running = [seq]
        model_runner = _SamplingTpotModelRunner()
        config = SimpleNamespace(
            max_draft_tokens=4,
            acceptance_strategy="standard_sampling",
            acceptance_threshold=0.7,
            draft_stop_policy="tpot",
            draft_tpot_stop_rule="lookahead",
            draft_tpot_verify_model_mode="active",
            draft_tpot_verify_model_sampling_validated=True,
        )

        engine = SpeculativeEngine(model_runner, scheduler, config)
        engine.speculative_step([seq])

        trace = engine.get_profile()["step_traces"][0]
        self.assertEqual(model_runner.draft_calls, 1)
        self.assertIn("lookahead_tpot_ms", trace["draft_tpot_costs"][0])

    def test_validated_sampling_rejects_uncalibrated_temperature(self):
        seq = _Seq(seq_id=1, token_ids=[1, 2, 3], temperature=1.0, max_tokens=8)
        scheduler = _DummyScheduler()
        scheduler.running = [seq]
        model_runner = _SamplingTpotModelRunner()
        config = SimpleNamespace(
            max_draft_tokens=4,
            acceptance_strategy="standard_sampling",
            acceptance_threshold=0.7,
            draft_stop_policy="tpot",
            draft_tpot_stop_rule="lookahead",
            draft_tpot_verify_model_mode="active",
            draft_tpot_verify_model_sampling_validated=True,
            draft_tpot_verify_model_temperature=0.8,
        )

        engine = SpeculativeEngine(model_runner, scheduler, config)
        with self.assertRaisesRegex(ValueError, "sampling temperature"):
            engine.speculative_step([seq])

    def test_transfer_aware_step_compares_only_k_and_k_plus_one_at_k6(self):
        seq = _Seq(seq_id=1, token_ids=[1, 2, 3], temperature=0.0)
        scheduler = _DummyScheduler()
        scheduler.running = [seq]
        model_runner = _TransferAwareTpotModelRunner(next_ms=500.0)
        config = SimpleNamespace(
            max_draft_tokens=12,
            acceptance_strategy="greedy",
            acceptance_threshold=0.7,
            acceptance_predictor_enabled=True,
            draft_stop_policy="tpot",
            draft_tpot_stop_rule="transfer_aware_step",
            draft_tpot_verify_model_mode="active",
            draft_tpot_min_steps=6,
            draft_tpot_cost_model="history",
            draft_tpot_history_alpha=0.2,
            draft_tpot_alpha_error_p90=0.0,
            draft_tpot_draft_error_p90_ms=0.0,
            draft_tpot_uncertainty_scale=1.0,
            verify_cuda_graph_bucket_steps=[5, 7, 8, 9, 10, 11, 12, 13],
        )

        engine = SpeculativeEngine(model_runner, scheduler, config)
        engine.speculative_step([seq])

        assert model_runner.draft_calls == 6
        assert model_runner.controls == [False] * 5 + [True]
        assert all(value is not None for value in model_runner.next_draft_ms)
        trace = engine.get_profile()["step_traces"][0]
        cost = trace["draft_tpot_costs"][-1]
        assert cost["lookahead_horizon"] == 1.0
        assert cost["lookahead_draft_len"] == 7.0
        assert cost["stop_decision"] == 1.0
        assert cost["draft_ms"] == pytest.approx(sum(trace["draft_call_ms"]))

    def test_transfer_aware_uncertain_state_fails_open_until_k12(self):
        seq = _Seq(seq_id=1, token_ids=[1, 2, 3], temperature=0.0)
        scheduler = _DummyScheduler()
        scheduler.running = [seq]
        model_runner = _TransferAwareTpotModelRunner(
            state_complete=False,
            next_ms=500.0,
        )
        config = SimpleNamespace(
            max_draft_tokens=12,
            acceptance_strategy="greedy",
            acceptance_threshold=0.7,
            acceptance_predictor_enabled=True,
            draft_stop_policy="tpot",
            draft_tpot_stop_rule="transfer_aware_step",
            draft_tpot_verify_model_mode="active",
            draft_tpot_min_steps=6,
            verify_cuda_graph_bucket_steps=[5, 7, 8, 9, 10, 11, 12, 13],
        )

        engine = SpeculativeEngine(model_runner, scheduler, config)
        engine.speculative_step([seq])

        assert model_runner.draft_calls == 12
        assert model_runner.controls == [False] * 5 + [True] * 6 + [False]
        profile = engine.get_profile()
        assert profile["draft_transfer_aware_fail_open_count"] == 6
        assert profile["step_traces"][0]["draft_steps_actual"] == 12

    def test_transfer_aware_shadow_matches_hot_path_without_stopping(self):
        seq = _Seq(seq_id=1, token_ids=[1, 2, 3], temperature=0.0)
        scheduler = _DummyScheduler()
        scheduler.running = [seq]
        model_runner = _TransferAwareTpotModelRunner(next_ms=500.0)
        config = SimpleNamespace(
            max_draft_tokens=12,
            acceptance_strategy="greedy",
            acceptance_threshold=0.7,
            acceptance_predictor_enabled=True,
            draft_stop_policy="tpot",
            draft_tpot_stop_rule="transfer_aware_step",
            draft_tpot_verify_model_mode="shadow",
            draft_tpot_min_steps=6,
            draft_tpot_cost_model="history",
            draft_tpot_history_alpha=0.2,
            draft_tpot_alpha_error_p90=0.0,
            draft_tpot_draft_error_p90_ms=0.0,
            draft_tpot_uncertainty_scale=1.0,
            verify_cuda_graph_bucket_steps=[5, 7, 8, 9, 10, 11, 12, 13],
        )

        engine = SpeculativeEngine(model_runner, scheduler, config)
        engine.speculative_step([seq])

        assert model_runner.draft_calls == 12
        assert model_runner.controls == [False] * 5 + [True] * 6 + [False]
        profile = engine.get_profile()
        assert profile.get("draft_tpot_early_stop_count", 0) == 0
        assert profile.get("draft_transfer_aware_fail_open_count", 0) == 0
        evaluated = [
            row
            for row in profile["step_traces"][0]["draft_tpot_costs"]
            if row["prediction_complete"]
        ]
        assert len(evaluated) == 6
        assert all(row["stop_signal"] == 1.0 for row in evaluated)
        assert all(row["stop_decision"] == 0.0 for row in evaluated)

    def test_transfer_aware_step_rejects_batch_greater_than_one(self):
        seqs = [
            _Seq(seq_id=1, token_ids=[1, 2, 3]),
            _Seq(seq_id=2, token_ids=[1, 2, 3]),
        ]
        engine = SpeculativeEngine(
            _TransferAwareTpotModelRunner(),
            _DummyScheduler(),
            SimpleNamespace(
                max_draft_tokens=12,
                acceptance_strategy="greedy",
                draft_tpot_stop_rule="transfer_aware_step",
            ),
        )
        with pytest.raises(ValueError, match="batch size 1"):
            engine.speculative_step(seqs)


if __name__ == "__main__":
    unittest.main()
