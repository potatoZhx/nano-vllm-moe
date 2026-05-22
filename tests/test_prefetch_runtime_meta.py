import unittest
from types import SimpleNamespace

import torch

from nanovllm.expert.runtime_meta import ModelRuntimeMetaRecorder, _aggregate_layer_runtime_meta_cpu


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
        self.assertIsNone(out[0].selected_experts)
        self.assertIsNone(out[0].routing_weights)
        self.assertEqual(out[0].aggregated_expert_ids.tolist(), [1, 2, 3])
        self.assertEqual(out[0].aggregated_activation_count.tolist(), [2, 2, 2])
        self.assertEqual([round(x, 4) for x in out[0].aggregated_score_sum.tolist()], [0.7, 1.2, 1.1])

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
        self.assertIsNone(out[0].selected_experts)
        self.assertIsNone(out[0].routing_weights)
        self.assertEqual(out[0].aggregated_expert_ids.tolist(), [1, 2, 3])
        self.assertEqual(out[0].aggregated_activation_count.tolist(), [1, 2, 1])

    def test_histogram_metadata_respects_logical_token_count(self):
        recorder = ModelRuntimeMetaRecorder(
            config=SimpleNamespace(prefetch_runtime_mode="draft_segment_indexed"),
            hf_config=SimpleNamespace(num_hidden_layers=1, num_experts_per_tok=2, num_experts=4),
        )
        recorder.arm(mode="draft", step_id=1, token_capacity=4, logical_token_count=2)

        selected = torch.tensor([[1, 2], [2, 3], [1, 0], [0, 3]], dtype=torch.int64)
        weights = torch.ones(4, 2, dtype=torch.float32)
        recorder.record_layer(layer_idx=0, selected_experts=selected, routing_weights=weights)

        handle = recorder.offload_async(stream=None)
        out = recorder.collect(handle, wait=True)

        self.assertEqual(handle.metadata_format, "histogram")
        self.assertEqual(out[0].token_count, 2)
        self.assertIsNone(out[0].selected_experts)
        self.assertIsNone(out[0].routing_weights)
        self.assertEqual(out[0].aggregated_expert_ids.tolist(), [1, 2, 3])
        self.assertEqual(out[0].aggregated_activation_count.tolist(), [1, 2, 1])

    def test_collect_supports_host_buffer_pool_slot(self):
        recorder = ModelRuntimeMetaRecorder(
            config=SimpleNamespace(prefetch_metadata_host_buffer_pool_size=2),
            hf_config=SimpleNamespace(num_hidden_layers=1, num_experts_per_tok=2),
        )
        recorder.arm(mode="draft", step_id=5, token_capacity=2, logical_token_count=2)
        self.assertTrue(recorder.maybe_grow_host_buffer_pool("draft", 2))
        self.assertEqual(recorder.get_host_buffer_pool_size("draft", 2), 2)

        selected = torch.tensor([[2, 1], [2, 3]], dtype=torch.int64)
        weights = torch.tensor([[0.4, 0.6], [0.7, 0.3]], dtype=torch.float32)
        recorder.record_layer(layer_idx=0, selected_experts=selected, routing_weights=weights)

        handle = recorder.offload_async(stream=None, host_buffer_slot=1)
        out = recorder.collect(handle, wait=True)

        self.assertEqual(handle.host_buffer_slot, 1)
        self.assertEqual(out[0].aggregated_expert_ids.tolist(), [1, 2, 3])
        self.assertEqual(out[0].aggregated_activation_count.tolist(), [1, 2, 1])
        self.assertEqual([round(x, 4) for x in out[0].aggregated_score_sum.tolist()], [0.6, 1.1, 0.3])

    def test_draft_segment_mode_auto_expands_host_buffer_target(self):
        recorder = ModelRuntimeMetaRecorder(
            config=SimpleNamespace(
                prefetch_metadata_host_buffer_pool_size=3,
                prefetch_runtime_mode="draft_direct_active",
                draft_prefetch_frontier_granularity="segment",
                draft_prefetch_segment_size=2,
                draft_prefetch_segment_host_buffer_pool_size=0,
            ),
            hf_config=SimpleNamespace(num_hidden_layers=8, num_experts_per_tok=2),
        )
        recorder.arm(mode="draft", step_id=5, token_capacity=2, logical_token_count=2)

        self.assertEqual(recorder.target_host_buffer_pool_size("draft", 2), 6)
        while recorder.maybe_grow_host_buffer_pool("draft", 2):
            pass
        self.assertEqual(recorder.get_host_buffer_pool_size("draft", 2), 6)
        self.assertFalse(recorder.maybe_grow_host_buffer_pool("draft", 2))

    def test_draft_segment_host_buffer_target_can_be_configured(self):
        recorder = ModelRuntimeMetaRecorder(
            config=SimpleNamespace(
                prefetch_metadata_host_buffer_pool_size=3,
                prefetch_runtime_mode="draft_direct_active",
                draft_prefetch_frontier_granularity="segment",
                draft_prefetch_segment_size=2,
                draft_prefetch_segment_host_buffer_pool_size=4,
            ),
            hf_config=SimpleNamespace(num_hidden_layers=8, num_experts_per_tok=2),
        )

        self.assertEqual(recorder.target_host_buffer_pool_size("draft", 2), 4)

    def test_collect_can_limit_layer_range(self):
        recorder = ModelRuntimeMetaRecorder(
            config=SimpleNamespace(),
            hf_config=SimpleNamespace(num_hidden_layers=4, num_experts_per_tok=1),
        )
        recorder.arm(mode="draft", step_id=7, token_capacity=2, logical_token_count=2)

        for layer_idx in range(4):
            selected = torch.tensor([[layer_idx], [layer_idx + 1]], dtype=torch.int64)
            weights = torch.ones(2, 1, dtype=torch.float32)
            recorder.record_layer(layer_idx=layer_idx, selected_experts=selected, routing_weights=weights)

        handle = recorder.offload_async(stream=None, layer_start_idx=1, layer_end_idx=3)
        out = recorder.collect(handle, wait=True)

        self.assertEqual(sorted(out.keys()), [1, 2])
        self.assertEqual(handle.layer_start_idx, 1)
        self.assertEqual(handle.layer_end_idx, 3)
        self.assertLess(handle.buffer_bytes, recorder._buffer_bytes(("draft", 2)))

    def test_small_cpu_aggregate_helper_uses_small_fast_path(self):
        selected = torch.tensor([[3, 1], [3, 2]], dtype=torch.int64)
        weights = torch.tensor([[0.6, 0.4], [0.7, 0.3]], dtype=torch.float32)
        aggregated = _aggregate_layer_runtime_meta_cpu(selected, weights)
        self.assertIsNotNone(aggregated)
        expert_ids, score_sum, counts = aggregated
        self.assertEqual(expert_ids.tolist(), [1, 2, 3])
        self.assertEqual(counts.tolist(), [1, 1, 2])
        self.assertEqual([round(x, 4) for x in score_sum.tolist()], [0.4, 0.3, 1.3])

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

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for histogram graph-capture regression test")
    def test_histogram_record_layer_is_capture_safe(self):
        recorder = ModelRuntimeMetaRecorder(
            config=SimpleNamespace(prefetch_runtime_mode="draft_segment_indexed"),
            hf_config=SimpleNamespace(num_hidden_layers=1, num_experts_per_tok=2, num_experts=4),
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
            out = recorder.collect(recorder.offload_async(stream=None), wait=True)

        self.assertEqual(out[0].aggregated_expert_ids.tolist(), [1, 2, 3])
        self.assertEqual(out[0].aggregated_activation_count.tolist(), [1, 2, 1])


if __name__ == "__main__":
    unittest.main()
