import unittest
from collections import defaultdict
from statistics import median
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

import torch

from nanovllm.engine.model_runner import ModelRunner


class TestDraftCudaGraphPolicy(unittest.TestCase):
    def test_can_use_draft_cudagraph_requires_top_c_zero(self):
        mr = object.__new__(ModelRunner)
        mr.config = SimpleNamespace(
            draft_cuda_graph_enabled=True,
            draft_top_c=1,
            draft_cuda_graph_cpu_backend="none",
            draft_cuda_graph_max_bs=128,
        )
        mr.enforce_eager = False
        mr.draft_graphs = {1: object()}
        mr.draft_graph_bs = [1, 2, 4]

        self.assertFalse(ModelRunner._can_use_draft_cudagraph(mr, 1))

    def test_can_use_draft_cudagraph_allows_fused_cpu_bridge(self):
        mr = object.__new__(ModelRunner)
        mr.config = SimpleNamespace(
            draft_cuda_graph_enabled=True,
            draft_top_c=2,
            draft_cuda_graph_cpu_backend="fused",
            cpu_expert_backend="fused",
            cpu_expert_execution_enabled=True,
            draft_cuda_graph_max_bs=128,
        )
        mr.enforce_eager = False
        mr.draft_graphs = {1: object(), 4: object()}
        mr.draft_graph_bs = [1, 4]

        self.assertTrue(ModelRunner._can_use_draft_cudagraph(mr, 3))

    def test_can_use_draft_cudagraph_allows_fused_sync_cpu_bridge(self):
        mr = object.__new__(ModelRunner)
        mr.config = SimpleNamespace(
            draft_cuda_graph_enabled=True,
            draft_top_c=2,
            draft_cuda_graph_cpu_backend="fused_sync",
            cpu_expert_backend="fused",
            cpu_expert_execution_enabled=True,
            draft_cuda_graph_max_bs=128,
        )
        mr.enforce_eager = False
        mr.draft_graphs = {1: object(), 4: object()}
        mr.draft_graph_bs = [1, 4]

        self.assertTrue(ModelRunner._can_use_draft_cudagraph(mr, 3))

    def test_can_use_draft_cudagraph_requires_templates(self):
        mr = object.__new__(ModelRunner)
        mr.config = SimpleNamespace(
            draft_cuda_graph_enabled=True,
            draft_top_c=0,
            draft_cuda_graph_max_bs=128,
        )
        mr.enforce_eager = False
        mr.draft_graphs = {}
        mr.draft_graph_bs = []

        self.assertFalse(ModelRunner._can_use_draft_cudagraph(mr, 1))

    def test_can_use_draft_cudagraph_happy_path(self):
        mr = object.__new__(ModelRunner)
        mr.config = SimpleNamespace(
            draft_cuda_graph_enabled=True,
            draft_top_c=0,
            draft_cuda_graph_max_bs=128,
        )
        mr.enforce_eager = False
        mr.draft_graphs = {1: object(), 4: object()}
        mr.draft_graph_bs = [1, 4]

        self.assertTrue(ModelRunner._can_use_draft_cudagraph(mr, 1))
        self.assertTrue(ModelRunner._can_use_draft_cudagraph(mr, 3))
        self.assertFalse(ModelRunner._can_use_draft_cudagraph(mr, 8))


class _Context:
    def __init__(self, batch_size: int, max_blocks: int):
        self.slot_mapping = torch.arange(batch_size, dtype=torch.int32)
        self.context_lens = torch.full((batch_size,), 8, dtype=torch.int32)
        self.block_tables = torch.arange(batch_size * max_blocks, dtype=torch.int32).view(batch_size, max_blocks)


class _DummyModel:
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class _FakeGraph:
    def __init__(self, graph_vars: dict):
        self.graph_vars = graph_vars
        self.replay_calls = 0

    def replay(self):
        self.replay_calls += 1
        bs = self.graph_vars["input_ids"].size(0)
        # Deterministic synthetic replay math to mimic captured graph output refresh.
        base = self.graph_vars["input_ids"].to(dtype=torch.float32).unsqueeze(-1)
        pos = self.graph_vars["positions"].to(dtype=torch.float32).unsqueeze(-1)
        self.graph_vars["outputs"][:bs].copy_(base + pos)


def _build_runner_for_replay(max_bs: int = 8, hidden_size: int = 4, max_blocks: int = 4) -> ModelRunner:
    mr = object.__new__(ModelRunner)
    mr.model = _DummyModel()
    mr.profile_enabled = True
    mr.rank = 0
    mr._profile = defaultdict(float)

    input_ids = torch.zeros(max_bs, dtype=torch.int64)
    positions = torch.zeros(max_bs, dtype=torch.int64)
    slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
    context_lens = torch.zeros(max_bs, dtype=torch.int32)
    block_tables = torch.zeros(max_bs, max_blocks, dtype=torch.int32)
    outputs = torch.zeros(max_bs, hidden_size, dtype=torch.float32)

    mr.graph_bs = [max_bs]
    mr.graph_vars = {
        "input_ids": input_ids.clone(),
        "positions": positions.clone(),
        "slot_mapping": slot_mapping.clone(),
        "context_lens": context_lens.clone(),
        "block_tables": block_tables.clone(),
        "outputs": outputs.clone(),
    }
    mr.graphs = {max_bs: _FakeGraph(mr.graph_vars)}

    mr.draft_graph_bs = [max_bs]
    mr.draft_graph_vars = {
        "input_ids": input_ids.clone(),
        "positions": positions.clone(),
        "slot_mapping": slot_mapping.clone(),
        "context_lens": context_lens.clone(),
        "block_tables": block_tables.clone(),
        "outputs": outputs.clone(),
    }
    mr.draft_graphs = {max_bs: _FakeGraph(mr.draft_graph_vars)}
    return mr


class TestDraftCudaGraphRouting(unittest.TestCase):
    def test_run_model_draft_policy_uses_draft_replay_only(self):
        mr = object.__new__(ModelRunner)
        mr.enforce_eager = False
        mr._decode_graph_policy = "draft"
        mr.config = SimpleNamespace(
            draft_cuda_graph_enabled=True,
            draft_top_c=0,
            draft_cuda_graph_max_bs=128,
        )

        called = {"draft": 0, "standard": 0, "eager": 0}

        def _draft(*_args, **_kwargs):
            called["draft"] += 1
            return torch.tensor([[1.0]])

        def _standard(*_args, **_kwargs):
            called["standard"] += 1
            return torch.tensor([[2.0]])

        def _eager(*_args, **_kwargs):
            called["eager"] += 1
            return torch.tensor([[3.0]])

        mr._can_use_draft_cudagraph = lambda _bs: True
        mr._replay_draft_graph = _draft
        mr._replay_standard_graph = _standard
        mr._run_model_eager = _eager

        out = ModelRunner.run_model(mr, torch.tensor([1]), torch.tensor([2]), is_prefill=False)

        self.assertTrue(torch.equal(out, torch.tensor([[1.0]])))
        self.assertEqual(called["draft"], 1)
        self.assertEqual(called["standard"], 0)
        self.assertEqual(called["eager"], 0)

    def test_run_model_draft_policy_falls_back_to_eager_on_miss(self):
        mr = object.__new__(ModelRunner)
        mr.enforce_eager = False
        mr._decode_graph_policy = "draft"
        mr.config = SimpleNamespace(
            draft_cuda_graph_enabled=True,
            draft_top_c=0,
            draft_cuda_graph_max_bs=128,
        )

        called = {"draft": 0, "standard": 0, "eager": 0}
        mr._can_use_draft_cudagraph = lambda _bs: False
        mr._replay_draft_graph = lambda *_args, **_kwargs: called.__setitem__("draft", called["draft"] + 1)
        mr._replay_standard_graph = lambda *_args, **_kwargs: called.__setitem__("standard", called["standard"] + 1)

        def _eager(*_args, **_kwargs):
            called["eager"] += 1
            return torch.tensor([[5.0]])

        mr._run_model_eager = _eager
        out = ModelRunner.run_model(mr, torch.tensor([1]), torch.tensor([2]), is_prefill=False)

        self.assertTrue(torch.equal(out, torch.tensor([[5.0]])))
        self.assertEqual(called["draft"], 0)
        self.assertEqual(called["standard"], 0)
        self.assertEqual(called["eager"], 1)

    def test_run_model_standard_policy_uses_standard_graph(self):
        mr = object.__new__(ModelRunner)
        mr.enforce_eager = False
        mr._decode_graph_policy = "standard"
        mr.graphs = {1: object()}
        mr.graph_bs = [1]
        mr.config = SimpleNamespace(
            draft_cuda_graph_enabled=True,
            draft_top_c=0,
            draft_cuda_graph_max_bs=128,
        )

        called = {"draft": 0, "standard": 0, "eager": 0}
        mr._can_use_draft_cudagraph = lambda _bs: True
        mr._replay_draft_graph = lambda *_args, **_kwargs: called.__setitem__("draft", called["draft"] + 1)

        def _standard(*_args, **_kwargs):
            called["standard"] += 1
            return torch.tensor([[7.0]])

        mr._replay_standard_graph = _standard
        mr._run_model_eager = lambda *_args, **_kwargs: called.__setitem__("eager", called["eager"] + 1)
        out = ModelRunner.run_model(mr, torch.tensor([1]), torch.tensor([2]), is_prefill=False)

        self.assertTrue(torch.equal(out, torch.tensor([[7.0]])))
        self.assertEqual(called["draft"], 0)
        self.assertEqual(called["standard"], 1)
        self.assertEqual(called["eager"], 0)

    def test_run_model_standard_policy_falls_back_to_eager_without_graph(self):
        mr = object.__new__(ModelRunner)
        mr.enforce_eager = False
        mr._decode_graph_policy = "standard"
        mr.config = SimpleNamespace(
            draft_cuda_graph_enabled=True,
            draft_top_c=0,
            draft_cuda_graph_max_bs=128,
        )
        mr.graphs = {}
        mr.graph_bs = []

        called = {"standard": 0, "eager": 0}

        def _standard(*_args, **_kwargs):
            called["standard"] += 1
            return torch.tensor([[11.0]])

        def _eager(*_args, **_kwargs):
            called["eager"] += 1
            return torch.tensor([[13.0]])

        mr._replay_standard_graph = _standard
        mr._run_model_eager = _eager

        out = ModelRunner.run_model(mr, torch.tensor([1]), torch.tensor([2]), is_prefill=False)

        self.assertTrue(torch.equal(out, torch.tensor([[13.0]])))
        self.assertEqual(called["standard"], 0)
        self.assertEqual(called["eager"], 1)


class TestDraftCudaGraphReplayParity(unittest.TestCase):
    def test_draft_and_standard_replay_outputs_match(self):
        mr = _build_runner_for_replay(max_bs=8)
        input_ids = torch.tensor([2, 3, 5], dtype=torch.int64)
        positions = torch.tensor([1, 4, 7], dtype=torch.int64)
        context = _Context(batch_size=input_ids.size(0), max_blocks=4)

        with patch("nanovllm.engine.model_runner.get_context", return_value=context):
            std_out = ModelRunner._replay_standard_graph(mr, input_ids, positions)
            draft_out = ModelRunner._replay_draft_graph(mr, input_ids, positions)

        self.assertTrue(torch.equal(std_out, draft_out))
        self.assertEqual(mr._profile["graph_replay_count"], 2)
        self.assertEqual(mr._profile["graph_hit_count"], 2)
        self.assertEqual(mr._profile["standard_graph_replay_count"], 1)
        self.assertEqual(mr._profile["draft_graph_replay_count"], 1)

    def test_draft_replay_latency_is_close_to_standard(self):
        mr = _build_runner_for_replay(max_bs=16)
        input_ids = torch.arange(8, dtype=torch.int64)
        positions = torch.arange(8, dtype=torch.int64)
        context = _Context(batch_size=input_ids.size(0), max_blocks=4)

        # Disable profile counters to keep timing focused on replay path itself.
        mr.profile_enabled = False

        with patch("nanovllm.engine.model_runner.get_context", return_value=context):
            for _ in range(64):
                ModelRunner._replay_standard_graph(mr, input_ids, positions)
                ModelRunner._replay_draft_graph(mr, input_ids, positions)

            std_samples = []
            draft_samples = []
            rounds = 5
            iters = 300
            for _ in range(rounds):
                t0 = perf_counter()
                for _ in range(iters):
                    ModelRunner._replay_standard_graph(mr, input_ids, positions)
                std_samples.append((perf_counter() - t0) / iters)

                t0 = perf_counter()
                for _ in range(iters):
                    ModelRunner._replay_draft_graph(mr, input_ids, positions)
                draft_samples.append((perf_counter() - t0) / iters)

        std_median = median(std_samples)
        draft_median = median(draft_samples)
        ratio = draft_median / std_median

        # Draft replay should stay close to standard replay for same synthetic workload.
        self.assertLessEqual(ratio, 1.25)
        self.assertGreaterEqual(ratio, 0.80)


if __name__ == "__main__":
    unittest.main()
