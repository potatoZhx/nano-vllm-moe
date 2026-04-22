import unittest
from types import SimpleNamespace

import torch

from nanovllm.expert.runtime_meta import ModelRuntimeMetaRecorder


class TestPrefetchRuntimeMeta(unittest.TestCase):
    def test_record_offload_collect(self):
        recorder = ModelRuntimeMetaRecorder(
            config=SimpleNamespace(),
            hf_config=SimpleNamespace(num_hidden_layers=2, num_experts_per_tok=2),
        )
        recorder.arm(mode="prefill", step_id=3, token_capacity=4, logical_token_count=3)

        selected = torch.tensor([[1, 2], [2, 3], [1, 3]], dtype=torch.int64)
        weights = torch.tensor([[0.5, 0.5], [0.7, 0.3], [0.2, 0.8]], dtype=torch.float32)
        recorder.record_layer(layer_idx=0, selected_experts=selected, routing_weights=weights)

        handle = recorder.offload_async(stream=None)
        out = recorder.collect(handle, wait=True)

        self.assertIsNotNone(handle)
        self.assertIn(0, out)
        self.assertEqual(out[0].token_count, 3)
        self.assertEqual(tuple(out[0].selected_experts.shape), (3, 2))

    def test_collect_respects_logical_token_count(self):
        recorder = ModelRuntimeMetaRecorder(
            config=SimpleNamespace(),
            hf_config=SimpleNamespace(num_hidden_layers=1, num_experts_per_tok=2),
        )
        recorder.arm(mode="draft", step_id=1, token_capacity=4, logical_token_count=2)

        selected = torch.tensor([[1, 2], [2, 3], [1, 0], [0, 3]], dtype=torch.int64)
        weights = torch.ones(4, 2, dtype=torch.float32)
        recorder.record_layer(layer_idx=0, selected_experts=selected, routing_weights=weights)

        out = recorder.collect(recorder.offload_async(stream=None), wait=True)
        self.assertEqual(out[0].token_count, 2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for pinned-host allocation regression test")
    def test_arm_with_cuda_default_device_allocates_host_mirror_on_cpu(self):
        recorder = ModelRuntimeMetaRecorder(
            config=SimpleNamespace(),
            hf_config=SimpleNamespace(num_hidden_layers=1, num_experts_per_tok=2),
        )

        with torch.device("cuda"):
            recorder.arm(mode="draft", step_id=1, token_capacity=2, logical_token_count=2)

        key = ("draft", 2)
        self.assertEqual(recorder.host_buffers[key]["selected_experts"].device.type, "cpu")
        self.assertEqual(recorder.host_buffers[key]["routing_weights"].device.type, "cpu")
        self.assertEqual(recorder.host_buffers[key]["token_count"].device.type, "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for graph-capture regression test")
    def test_record_layer_is_capture_safe_for_token_count_write(self):
        recorder = ModelRuntimeMetaRecorder(
            config=SimpleNamespace(),
            hf_config=SimpleNamespace(num_hidden_layers=1, num_experts_per_tok=2),
        )

        with torch.device("cuda"):
            recorder.arm(mode="draft", step_id=1, token_capacity=2, logical_token_count=2)
            selected = torch.tensor([[1, 2], [2, 3]], dtype=torch.int64, device="cuda")
            weights = torch.tensor([[0.7, 0.3], [0.4, 0.6]], dtype=torch.float16, device="cuda")

            graph = torch.cuda.CUDAGraph()
            recorder.record_layer(layer_idx=0, selected_experts=selected, routing_weights=weights)
            with torch.cuda.graph(graph):
                recorder.record_layer(layer_idx=0, selected_experts=selected, routing_weights=weights)

            graph.replay()
            torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
