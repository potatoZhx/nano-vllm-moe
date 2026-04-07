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
        self.assertIsNotNone(plan.gpu_m_sizes)
        self.assertIsNotNone(plan.cpu_task_expert_ids)
        self.assertIsNotNone(plan.cpu_task_offsets)

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
        self.assertIsNotNone(plan.substitution_lut)
        self.assertIsNotNone(plan.gpu_route_mask)
        self.assertIsNotNone(plan.cpu_route_mask)
        # top_c=1 should keep one uncached expert on CPU and substitute the other.
        self.assertIsNotNone(plan.cpu_route_indices)
        self.assertEqual(plan.cpu_route_indices.tolist(), [2])

    def test_draft_plan_topc_zero_prefers_gpu_substitution(self):
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
            top_c=0,
        )
        self.assertTrue(plan.cpu_route_indices is None or plan.cpu_route_indices.numel() == 0)
        self.assertTrue(plan.gpu_route_indices.numel() > 0)
        self.assertIsNotNone(plan.substitution_lut)

        # Experts 4/5 are uncached in this fixture and should be remapped into cached range [0,1,2].
        remapped = plan.substitution_lut.index_select(0, torch.tensor([4, 5], dtype=torch.int64))
        self.assertTrue(torch.all(remapped < 3).item())

    def test_draft_plan_all_cached_uses_gpu_only_fast_path(self):
        cache = LayerExpertCache(
            num_experts=8,
            slots_per_layer=8,
            gate_up_shape=(4, 4),
            down_shape=(4, 2),
            device=torch.device("cpu"),
            dtype=torch.float32,
            cpu_expert_pool={},
        )
        fake = torch.zeros(4, 4)
        fake_down = torch.zeros(4, 2)
        for expert_id in range(8):
            cache.put_to_slot(expert_id, expert_id, fake, fake_down)

        class _SchedulerShouldNotBeCalled:
            def select_cpu_experts_gpu(self, **_kwargs):
                raise AssertionError("select_cpu_experts_gpu should not be called in all-cached fast path")

            def build_substitution_lut_gpu(self, **_kwargs):
                raise AssertionError("build_substitution_lut_gpu should not be called in all-cached fast path")

        selected = torch.tensor([0, 4, 5, 2], dtype=torch.int64)
        routing_w = torch.ones(2, 2, dtype=torch.float32)
        plan = build_draft_plan(
            layer_idx=0,
            selected_experts=selected,
            routing_weights=routing_w,
            expert_cache=cache,
            draft_scheduler=_SchedulerShouldNotBeCalled(),
            num_experts=8,
            top_c=0,
        )

        self.assertIsNone(plan.cpu_route_indices)
        self.assertIsNotNone(plan.substitution_lut)
        self.assertEqual(plan.gpu_route_indices.numel(), selected.numel())
        # S=N acceptance criterion: top_c=0 still follows substitution path,
        # but actual replacement count is zero (identity LUT).
        expected = torch.arange(8, dtype=torch.int64)
        self.assertTrue(torch.equal(plan.substitution_lut.cpu(), expected))


if __name__ == "__main__":
    unittest.main()
