import unittest
from types import SimpleNamespace

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

    def call(self, name, seqs, *args):
        if name == "run_draft":
            self.draft_calls += 1
            if self.draft_calls == 1:
                return [11 for _ in seqs], []
            return [12 for _ in seqs], []
        if name == "run_verify":
            self.verify_calls += 1
            self.last_verify_lengths = list(args[0]) if args else None
            return [[11, 12, 99] for _ in seqs]
        raise RuntimeError(name)


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
        self.assertEqual(seq.last_token, 99)
        self.assertEqual(seq.token_ids, [1, 2, 3, 11, 12, 99])
        self.assertFalse(seq.is_drafting)
        self.assertIn(("start", 1), scheduler.ops)
        self.assertIn(("rollback", 1), scheduler.ops)
        self.assertIn(("accept", 1, 2), scheduler.ops)

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


if __name__ == "__main__":
    unittest.main()
