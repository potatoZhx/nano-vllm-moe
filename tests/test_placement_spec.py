import unittest

import torch

from nanovllm.expert.cache import LayerExpertCache
from nanovllm.expert.placement import build_draft_plan, build_prefill_plan
from nanovllm.scheduling.draft_scheduler import SimpleDraftScheduler


class TestPlacementSpec(unittest.TestCase):
    def _build_cache(self):
        cache = LayerExpertCache(
            num_experts=8,
            slots_per_layer=3,
            gate_up_shape=(4, 4),
            down_shape=(4, 2),
            device=torch.device("cpu"),
            dtype=torch.float32,
            cpu_expert_pool={},
        )
        fake = torch.zeros(4, 4)
        fake_down = torch.zeros(4, 2)
        cache.put_to_slot(0, 0, fake, fake_down)
        cache.put_to_slot(1, 1, fake, fake_down)
        cache.put_to_slot(2, 2, fake, fake_down)
        return cache

    def test_prefill_plan_splits_gpu_cpu(self):
        cache = self._build_cache()
        selected = torch.tensor([0, 1, 4, 2], dtype=torch.int64)
        routing_w = torch.ones(2, 2)
        plan = build_prefill_plan(
            layer_idx=0,
            selected_experts=selected,
            routing_weights=routing_w,
            expert_cache=cache,
            num_experts=8,
        )
        self.assertEqual(plan.gpu_route_indices.tolist(), [0, 1, 3])
        self.assertEqual(plan.cpu_route_indices.tolist(), [2])

    def test_draft_plan_applies_substitution(self):
        cache = self._build_cache()
        selected = torch.tensor([0, 4, 5, 2], dtype=torch.int64)
        routing_w = torch.tensor([[0.9, 0.1], [0.6, 0.4]], dtype=torch.float32)
        scheduler = SimpleDraftScheduler()
        plan = build_draft_plan(
            layer_idx=0,
            selected_experts=selected,
            routing_weights=routing_w,
            expert_cache=cache,
            draft_scheduler=scheduler,
            num_experts=8,
            top_c=1,
        )
        self.assertGreaterEqual(len(plan.substitution_map), 1)
        self.assertIsNotNone(plan.m_sizes)


if __name__ == "__main__":
    unittest.main()
