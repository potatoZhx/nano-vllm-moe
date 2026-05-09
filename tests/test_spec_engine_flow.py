import unittest
from types import SimpleNamespace

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
        raise RuntimeError(name)


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
        self.assertEqual(len(append_ops), 2)

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


if __name__ == "__main__":
    unittest.main()
