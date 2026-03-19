import unittest
from types import SimpleNamespace

from nanovllm.engine.llm_engine import LLMEngine


class _DummyScheduler:
    def __init__(self, seqs, is_prefill):
        self._seqs = seqs
        self._is_prefill = is_prefill
        self.post_calls = []

    def schedule(self):
        return self._seqs, self._is_prefill

    def postprocess(self, seqs, token_ids):
        self.post_calls.append((seqs, token_ids))


class _DummyRunner:
    def __init__(self):
        self.calls = []

    def call(self, name, seqs, is_prefill):
        self.calls.append((name, is_prefill, len(seqs)))
        return [42 for _ in seqs]


class _DummySpec:
    def __init__(self):
        self.calls = []

    def speculative_step(self, seqs):
        self.calls.append(len(seqs))
        return [99 for _ in seqs]


class TestLLMEngineModeDispatch(unittest.TestCase):
    def test_spec_mode_dispatches_to_spec_engine(self):
        seqs = [SimpleNamespace(seq_id=1, is_finished=False, completion_token_ids=[])]
        eng = object.__new__(LLMEngine)
        eng.config = SimpleNamespace(inference_mode="spec")
        eng.scheduler = _DummyScheduler(seqs, is_prefill=False)
        eng.model_runner = _DummyRunner()
        eng.spec_engine = _DummySpec()

        outputs, num_tokens = LLMEngine.step(eng)
        self.assertEqual(num_tokens, -1)
        self.assertEqual(eng.spec_engine.calls, [1])
        self.assertEqual(len(eng.scheduler.post_calls), 0)
        self.assertEqual(outputs, [])


if __name__ == "__main__":
    unittest.main()
