import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

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
        mr.run = lambda seqs, is_prefill: [101 for _ in seqs]

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
        runtime.publish_ready.assert_called()
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


if __name__ == "__main__":
    unittest.main()
