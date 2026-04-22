import unittest
from types import SimpleNamespace

import torch

from nanovllm.expert.cache import LayerExpertCache
from nanovllm.expert.prefetcher import GlobalWarmStartQueue
from nanovllm.expert.runtime_meta import LayerRuntimeMetaCPU


class TestPrefetchGlobalQueue(unittest.TestCase):
    def _config(self):
        return SimpleNamespace(
            prefetch_source_weight_prefill=1.0,
            prefetch_source_weight_verify=1.2,
            prefetch_source_weight_draft=1.5,
            prefetch_activation_count_weight=0.1,
            prefetch_age_penalty=0.01,
            prefetch_history_decay=0.9,
            prefetch_history_ttl_steps=8,
            prefetch_global_queue_capacity=16,
        )

    def _cache(self):
        cache = LayerExpertCache(
            num_experts=4,
            slots_per_layer=2,
            gate_up_shape=(2, 2),
            down_shape=(2, 2),
            device=torch.device("cpu"),
            dtype=torch.float32,
            cpu_expert_pool={},
        )
        fake = torch.zeros(2, 2)
        cache.put_to_slot(0, 0, fake, fake)
        cache.put_to_slot(1, 1, fake, fake)
        return cache

    def test_update_dedup_and_rank(self):
        queue = GlobalWarmStartQueue(self._config())
        cache = self._cache()
        runtime_meta = {
            0: LayerRuntimeMetaCPU(
                step_id=1,
                mode="prefill",
                layer_idx=0,
                token_count=3,
                selected_experts=torch.tensor([[2, 3], [2, 2], [3, 2]], dtype=torch.int64),
                routing_weights=torch.tensor([[0.8, 0.2], [0.5, 0.5], [0.7, 0.3]], dtype=torch.float32),
            )
        }
        queue.update_from_runtime_meta(runtime_meta, source="prefill_history", step_id=1, layer_caches={0: cache})
        self.assertIn((0, 2), queue.entries)
        self.assertIn((0, 3), queue.entries)
        self.assertNotIn((0, 0), queue.entries)

        queue.update_from_runtime_meta(runtime_meta, source="draft_live", step_id=2, layer_caches={0: cache})
        self.assertEqual(queue.entries[(0, 2)].source, "draft_live")

        ranked = queue.ranked_candidates(step_id=3, layer_caches={0: cache}, inflight_keys={(0, 3)})
        keys = [(c.layer_idx, c.expert_idx) for c in ranked]
        self.assertIn((0, 2), keys)
        self.assertNotIn((0, 3), keys)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for default-device regression test")
    def test_update_under_cuda_default_device(self):
        queue = GlobalWarmStartQueue(self._config())
        cache = self._cache()
        runtime_meta = {
            0: LayerRuntimeMetaCPU(
                step_id=1,
                mode="prefill",
                layer_idx=0,
                token_count=2,
                selected_experts=torch.tensor([[2, 3], [3, 2]], dtype=torch.int64),
                routing_weights=torch.tensor([[0.7, 0.3], [0.2, 0.8]], dtype=torch.float32),
            )
        }

        with torch.device("cuda"):
            queue.update_from_runtime_meta(
                runtime_meta,
                source="prefill_history",
                step_id=1,
                layer_caches={0: cache},
            )

        self.assertIn((0, 2), queue.entries)
        self.assertIn((0, 3), queue.entries)


if __name__ == "__main__":
    unittest.main()
