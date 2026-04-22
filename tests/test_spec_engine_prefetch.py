import unittest
from types import SimpleNamespace

from nanovllm.engine.speculative.spec_engine import SpeculativeEngine


class _Seq:
    def __init__(self):
        self.seq_id = 1
        self.token_ids = [1, 2, 3]
        self.num_tokens = 3
        self.num_cached_tokens = 0
        self.last_token = 3
        self._draft_start_num_tokens = 3
        self.draft_token_ids = []
        self.is_drafting = False
        self.temperature = 0.0
        self.max_tokens = 8
        self.num_prompt_tokens = 3
        self.ignore_eos = False
        self.status = None

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.num_tokens += 1
        self.last_token = token_id

    def append_draft_token(self, token_id: int):
        self.draft_token_ids.append(token_id)
        self.append_token(token_id)

    def start_draft(self):
        self._draft_start_num_tokens = self.num_tokens
        self.draft_token_ids = []
        self.is_drafting = True

    def rollback_tokens_to_draft_start(self):
        self.token_ids = self.token_ids[: self._draft_start_num_tokens]
        self.num_tokens = len(self.token_ids)
        self.last_token = self.token_ids[-1]

    def finish_draft(self):
        self.is_drafting = False
        self.draft_token_ids = []


class _Scheduler:
    def __init__(self, seq):
        self.seq = seq
        self.running = [seq]
        self.ops = []
        self.eos = -1

    def start_draft_kv(self, seq):
        self.ops.append(("start", seq.seq_id))

    def append_draft_kv(self, seq):
        self.ops.append(("append", seq.seq_id))

    def rollback_draft_kv(self, seq):
        self.ops.append(("rollback", seq.seq_id))

    def accept_draft_kv(self, seq, keep):
        self.ops.append(("accept", seq.seq_id, keep))


class _ModelRunner:
    def __init__(self):
        self.calls = []

    def call(self, name, *args):
        self.calls.append((name, args))
        if name == "run_draft":
            return [11], {"prefetch_step_id": 99}
        if name == "wait_prefetch_for_verify":
            return {"verify_prefetch_wait_ms": 0.2}
        if name == "run_verify":
            return [[11, 44]]
        raise RuntimeError(name)


class TestSpecEnginePrefetch(unittest.TestCase):
    def test_wait_hook_runs_before_verify(self):
        seq = _Seq()
        scheduler = _Scheduler(seq)
        runner = _ModelRunner()
        cfg = SimpleNamespace(max_draft_tokens=1, acceptance_strategy="greedy", acceptance_threshold=0.7, spec_profile=True)

        engine = SpeculativeEngine(model_runner=runner, scheduler=scheduler, config=cfg)
        out = engine.speculative_step([seq])

        self.assertEqual(out, [44])
        names = [x[0] for x in runner.calls]
        self.assertIn("wait_prefetch_for_verify", names)
        self.assertLess(names.index("wait_prefetch_for_verify"), names.index("run_verify"))


if __name__ == "__main__":
    unittest.main()
