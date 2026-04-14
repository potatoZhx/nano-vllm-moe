import unittest
from collections import defaultdict
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

    def get_profile(self, reset=False):
        return {
            "route_ms": 3.0,
            "verify_ms": 0.0,
            "graph_hit_rate": 0.8,
            "graph_replay_count": 6,
            "cpu_route_ratio": 0.25,
            "cpu_weight_mass_ratio": 0.2,
            "activated_expert_set_size": 4.0,
            "realized_cpu_expert_count": 1.0,
        }


class _DummySpec:
    def __init__(self):
        self.calls = []

    def speculative_step(self, seqs):
        self.calls.append(len(seqs))
        return [99 for _ in seqs]

    def get_profile(self, reset=False):
        return {
            "draft_ms": 5.0,
            "verify_ms": 7.0,
            "spec_step_ms": 9.0,
        }


class TestLLMEngineModeDispatch(unittest.TestCase):
    def test_spec_mode_dispatches_to_spec_engine(self):
        seqs = [SimpleNamespace(seq_id=1, is_finished=False, completion_token_ids=[])]
        eng = object.__new__(LLMEngine)
        eng.config = SimpleNamespace(inference_mode="spec")
        eng.scheduler = _DummyScheduler(seqs, is_prefill=False)
        eng.model_runner = _DummyRunner()
        eng.spec_engine = _DummySpec()
        eng.profile_enabled = False
        eng._profile = defaultdict(float)

        outputs, num_tokens = LLMEngine.step(eng)
        self.assertEqual(num_tokens, -1)
        self.assertEqual(eng.spec_engine.calls, [1])
        self.assertEqual(len(eng.scheduler.post_calls), 0)
        self.assertEqual(outputs, [])

    def test_get_profile_adds_canonical_phase2_post_aliases(self):
        eng = object.__new__(LLMEngine)
        eng.profile_enabled = False
        eng._profile = defaultdict(float)
        eng.model_runner = _DummyRunner()
        eng.spec_engine = _DummySpec()

        profile = LLMEngine.get_profile(eng, reset=False)

        self.assertAlmostEqual(float(profile["route_ms"]), 3.0, places=6)
        self.assertAlmostEqual(float(profile["draft_ms"]), 5.0, places=6)
        self.assertAlmostEqual(float(profile["verify_ms"]), 7.0, places=6)
        self.assertAlmostEqual(float(profile["spec_step_ms"]), 9.0, places=6)
        self.assertAlmostEqual(float(profile["graph_hit_rate"]), 0.8, places=6)
        self.assertEqual(int(profile["graph_replay_count"]), 6)
        self.assertAlmostEqual(float(profile["cpu_route_ratio"]), 0.25, places=6)


if __name__ == "__main__":
    unittest.main()
