import unittest
from types import SimpleNamespace

import torch

from nanovllm.engine.model_runner import ModelRunner


class _DummyModel:
    def __init__(self):
        self.mode_calls = []
        self.lm_head = SimpleNamespace(weight=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32))

    def set_speculative_execution_mode(self, mode, draft_scheduler, draft_top_c):
        self.mode_calls.append((mode, draft_scheduler is not None, draft_top_c))

    def __call__(self, input_ids, positions):
        # Produce deterministic hidden states for argmax checks.
        return torch.tensor([[2.0, 0.0], [0.0, 3.0], [2.0, 0.0]], dtype=torch.float32)


class TestModelRunnerSpecModes(unittest.TestCase):
    def test_run_draft_switches_mode(self):
        mr = object.__new__(ModelRunner)
        mr.model = _DummyModel()
        mr.config = SimpleNamespace(draft_top_c=2)
        mr.draft_scheduler = object()
        mr.run = lambda seqs, is_prefill: [7 for _ in seqs]

        out, aux = ModelRunner.run_draft(mr, [SimpleNamespace(seq_id=1)])

        self.assertEqual(out, [7])
        self.assertEqual(aux, [])
        self.assertEqual(mr.model.mode_calls[0][0], "draft")
        self.assertEqual(mr.model.mode_calls[-1][0], "normal")

    def test_run_verify_switches_mode_and_returns_traces(self):
        mr = object.__new__(ModelRunner)
        mr.model = _DummyModel()
        mr.config = SimpleNamespace(draft_top_c=1)
        mr.draft_scheduler = object()
        mr.world_size = 1
        mr.rank = 0
        mr.prepare_prefill = lambda seqs: (torch.tensor([1, 2, 3]), torch.tensor([0, 1, 2]))

        traces = ModelRunner.run_verify(mr, [SimpleNamespace(seq_id=1)], [3])

        self.assertEqual(traces, [[0, 1, 0]])
        self.assertEqual(mr.model.mode_calls[0][0], "verify")
        self.assertEqual(mr.model.mode_calls[-1][0], "normal")


if __name__ == "__main__":
    unittest.main()
