import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from nanovllm.engine.model_runner import ModelRunner


class TestModelRunnerPrefetch(unittest.TestCase):
    def test_run_draft_prefetch_hooks(self):
        mr = object.__new__(ModelRunner)
        mr.config = SimpleNamespace(draft_top_c=0, prefetch_verify_wait_ms=0.0)
        mr._prefetch_step_id = 0
        mr.profile_enabled = False
        mr.profile_cuda_sync = False
        mr.rank = 0
        mr._profile = {}
        mr._decode_graph_policy = "standard"
        mr.draft_graph_bs = [1, 2, 4]

        modes = []
        mr._set_speculative_execution_mode = lambda mode: modes.append(mode)
        mr._can_use_draft_cudagraph = lambda _bs: False
        mr.run = lambda seqs, is_prefill, return_logits=False: [101 for _ in seqs]

        handle = SimpleNamespace(buffer_bytes=256)
        recorder = MagicMock()
        recorder.offload_async.return_value = handle
        recorder.collect.return_value = {0: "meta"}
        mr.runtime_meta_recorder = recorder

        runtime = MagicMock()
        runtime.metadata_stream = None
        mr.prefetch_runtime = runtime

        out, aux = ModelRunner.run_draft(mr, [SimpleNamespace(seq_id=1)])

        self.assertEqual(out, [101])
        self.assertIn("prefetch_step_id", aux)
        runtime.publish_ready.assert_not_called()
        runtime.submit_from_global_queue.assert_called()
        runtime.observe_draft.assert_called_once()
        recorder.arm.assert_called_once()
        recorder.reset.assert_called_once()
        self.assertEqual(modes[0], "draft")
        self.assertEqual(modes[-1], "normal")

    def test_wait_prefetch_for_verify(self):
        mr = object.__new__(ModelRunner)
        mr.config = SimpleNamespace(prefetch_verify_wait_ms=1.0)
        runtime = MagicMock()
        mr.prefetch_runtime = runtime

        out = ModelRunner.wait_prefetch_for_verify(mr, 7)

        runtime.wait_for_verify.assert_called_once_with(step_id=7, timeout_ms=1.0)
        self.assertIn("verify_prefetch_wait_ms", out)

    def test_warmup_verify_layer_timings_primes_ema(self):
        class _FakeModel:
            def __init__(self):
                self.controller = None

            def set_verify_prefetch_controller(self, controller):
                self.controller = controller

            def __call__(self, input_ids, positions):
                self.controller.before_verify_layer(0)
                _ = torch.empty((128, 128)).sum()
                self.controller.after_verify_layer(0)
                return input_ids

        mr = object.__new__(ModelRunner)
        mr.config = SimpleNamespace(
            prefetch_verify_layer_enabled=True,
            max_draft_tokens=2,
            max_model_len=16,
            max_num_seqs=1,
            max_num_batched_tokens=16,
            draft_top_c=0,
        )
        mr.layer_caches = {1: object()}
        mr.model = _FakeModel()
        mr.profile_enabled = False
        mr.profile_cuda_sync = False
        mr.rank = 0
        mr._profile = {}
        mr._prefetch_step_id = 0
        mr.prefetch_runtime = None
        mr.prepare_prefill = lambda _seqs: (
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
        )
        modes = []
        mr._set_speculative_execution_mode = lambda mode: modes.append(mode)

        ModelRunner._warmup_verify_layer_timings(mr)

        self.assertIn(0, mr._verify_layer_compute_ms_ema)
        self.assertGreaterEqual(mr._verify_layer_compute_ms_ema[0], 0.0)
        self.assertEqual(modes[0], "verify")
        self.assertEqual(modes[-1], "normal")


if __name__ == "__main__":
    unittest.main()
