import unittest
from types import SimpleNamespace

from nanovllm.engine.speculative.spec_engine import SpeculativeEngine


class _DummyModelRunner:
    def __init__(self):
        self.calls = []

    def call(self, name, seqs, is_prefill):
        self.calls.append((name, is_prefill, len(seqs)))
        return [7 for _ in seqs]


class _DummyScheduler:
    pass


class TestSpecEngineBasic(unittest.TestCase):
    def test_spec_engine_returns_token_ids(self):
        mr = _DummyModelRunner()
        config = SimpleNamespace(max_draft_tokens=4)
        engine = SpeculativeEngine(model_runner=mr, scheduler=_DummyScheduler(), config=config)
        seqs = [SimpleNamespace(seq_id=1), SimpleNamespace(seq_id=2)]

        out = engine.speculative_step(seqs)
        self.assertEqual(out, [7, 7])
        self.assertEqual(mr.calls[0][0], "run")


if __name__ == "__main__":
    unittest.main()
