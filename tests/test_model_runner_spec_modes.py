import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

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
        mr.profile_enabled = False
        mr.profile_cuda_sync = False
        mr.rank = 0
        mr._profile = {}
        mr._prefetch_step_id = 0
        mr._decode_graph_policy = "standard"
        mr.run = lambda seqs, is_prefill, return_logits=False: [7 for _ in seqs]

        out, aux = ModelRunner.run_draft(mr, [SimpleNamespace(seq_id=1)])

        self.assertEqual(out, [7])
        self.assertIn("prefetch_step_id", aux)
        self.assertEqual(mr.model.mode_calls[0][0], "draft")
        self.assertEqual(mr.model.mode_calls[-1][0], "normal")
        self.assertEqual(mr._decode_graph_policy, "standard")

    def test_run_verify_switches_mode_and_returns_traces(self):
        mr = object.__new__(ModelRunner)
        mr.model = _DummyModel()
        mr.config = SimpleNamespace(draft_top_c=1)
        mr.draft_scheduler = object()
        mr.profile_enabled = False
        mr.profile_cuda_sync = False
        mr.world_size = 1
        mr.rank = 0
        mr._profile = {}
        mr._prefetch_step_id = 0
        mr.prepare_prefill = lambda seqs: (torch.tensor([1, 2, 3]), torch.tensor([0, 1, 2]))

        traces = ModelRunner.run_verify(mr, [SimpleNamespace(seq_id=1)], [3])

        self.assertEqual(traces, [[0, 1, 0]])
        self.assertEqual(mr.model.mode_calls[0][0], "verify")
        self.assertEqual(mr.model.mode_calls[-1][0], "normal")

    def test_run_verify_cache_fill_no_cpu_skips_verify_metadata_offload(self):
        mr = object.__new__(ModelRunner)
        mr.model = _DummyModel()
        mr.config = SimpleNamespace(
            draft_top_c=1,
            spec_verify_miss_policy="cache_fill_no_cpu",
            prefetch_runtime_mode="draft_segment_indexed",
        )
        mr.draft_scheduler = object()
        mr.profile_enabled = True
        mr.profile_cuda_sync = False
        mr.world_size = 1
        mr.rank = 0
        mr._profile = defaultdict(float)
        mr._prefetch_step_id = 0
        mr.prepare_prefill = lambda seqs: (torch.tensor([1, 2, 3]), torch.tensor([0, 1, 2]))

        recorder = MagicMock()
        recorder.offload_async.return_value = None
        mr.runtime_meta_recorder = recorder
        runtime = MagicMock()
        runtime.metadata_stream = None
        mr.prefetch_runtime = runtime

        traces = ModelRunner.run_verify(mr, [SimpleNamespace(seq_id=1)], [3])

        self.assertEqual(traces, [[0, 1, 0]])
        recorder.arm.assert_not_called()
        recorder.offload_async.assert_not_called()
        recorder.reset.assert_not_called()
        runtime.publish_direct_active_ready.assert_called_once()
        runtime.end_draft_iteration.assert_called_once()
        self.assertEqual(mr._profile["verify_metadata_skipped_count"], 1.0)

    def test_get_profile_exposes_phase2_post_core_fields(self):
        mr = object.__new__(ModelRunner)
        mr.rank = 0
        mr._profile = defaultdict(
            float,
            {
                "decode_count": 4,
                "graph_hit_count": 3,
                "route_ms": 12.0,
                "plan_ms": 5.0,
                "gpu_gather_ms": 2.0,
                "gpu_compute_ms": 7.0,
                "cpu_prepare_ms": 1.5,
                "cpu_compute_ms": 4.5,
                "cpu_to_gpu_merge_ms": 0.8,
                "scatter_ms": 1.2,
                "graph_replay_count": 3,
                "moe_profile_count": 2,
                "cpu_route_ratio_sum": 0.9,
                "cpu_weight_mass_ratio_sum": 0.7,
                "activated_expert_set_size_sum": 10.0,
                "realized_cpu_expert_count_sum": 4.0,
            },
        )

        out = ModelRunner.get_profile(mr, reset=False)

        self.assertEqual(out["decode_count"], 4)
        self.assertAlmostEqual(out["graph_hit_rate"], 0.75, places=6)
        self.assertAlmostEqual(out["cpu_route_ratio"], 0.45, places=6)
        self.assertAlmostEqual(out["cpu_weight_mass_ratio"], 0.35, places=6)
        self.assertAlmostEqual(out["activated_expert_set_size"], 5.0, places=6)
        self.assertAlmostEqual(out["realized_cpu_expert_count"], 2.0, places=6)

        for key in [
            "route_ms",
            "plan_ms",
            "gpu_gather_ms",
            "gpu_compute_ms",
            "cpu_prepare_ms",
            "cpu_compute_ms",
            "cpu_to_gpu_merge_ms",
            "scatter_ms",
            "graph_replay_count",
        ]:
            self.assertIn(key, out)

        _ = ModelRunner.get_profile(mr, reset=True)
        self.assertEqual(len(mr._profile), 0)


if __name__ == "__main__":
    unittest.main()
