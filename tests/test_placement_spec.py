import unittest

import torch

from nanovllm.expert.cache import LayerExpertCache
from nanovllm.expert.placement import (
    apply_verify_cache_fill_policy,
    build_cache_fill_no_cpu_verify_plan_gpu,
    build_draft_plan,
    build_prefill_plan,
    build_verify_plan,
    build_runtime_meta_view,
    flatten_selected_and_weights,
)
from nanovllm.scheduling.draft_scheduler import SimpleDraftScheduler


class TestPlacementSpec(unittest.TestCase):
    def _build_cache_with_slots(self, slots: int, cached: list[int], num_experts: int = 8):
        fake_pool = {
            expert: {"gate_up": torch.full((4, 4), float(expert)), "down": torch.full((4, 2), float(expert))}
            for expert in range(num_experts)
        }
        cache = LayerExpertCache(
            num_experts=num_experts,
            slots_per_layer=slots,
            gate_up_shape=(4, 4),
            down_shape=(4, 2),
            device=torch.device("cpu"),
            dtype=torch.float32,
            cpu_expert_pool=fake_pool,
        )
        for slot_idx, expert_idx in enumerate(cached):
            cache.put_to_slot(slot_idx, expert_idx, fake_pool[expert_idx]["gate_up"], fake_pool[expert_idx]["down"])
            cache.last_access_step[expert_idx] = slot_idx + 10
        return cache

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

    def test_verify_cache_fill_promotes_all_misses_when_active_unique_fits(self):
        cache = self._build_cache_with_slots(slots=4, cached=[0, 1])
        selected = torch.tensor([[0, 2], [3, 2]], dtype=torch.int64)
        routing_w = torch.ones(2, 2, dtype=torch.float32)
        profile = {}

        result = apply_verify_cache_fill_policy(
            layer_idx=0,
            selected_experts=selected,
            routing_weights=routing_w,
            expert_cache=cache,
            step_id=7,
            profile=profile,
        )
        plan = build_verify_plan(
            layer_idx=0,
            selected_experts=selected,
            routing_weights=routing_w,
            expert_cache=cache,
            num_experts=8,
        )

        self.assertEqual(result.promoted_expert_count, 2)
        self.assertEqual(result.cpu_expert_count, 0)
        self.assertTrue(cache.is_cached_cpu(2))
        self.assertTrue(cache.is_cached_cpu(3))
        self.assertTrue(plan.cpu_route_indices is None or plan.cpu_route_indices.numel() == 0)
        self.assertEqual(profile["verify_cache_fill_promoted_expert_count"], 2.0)

    def test_verify_cache_fill_leaves_lowest_count_misses_on_cpu_when_over_capacity(self):
        cache = self._build_cache_with_slots(slots=3, cached=[0, 1])
        selected = torch.tensor([[0, 1], [2, 2], [3, 4]], dtype=torch.int64)
        routing_w = torch.ones(3, 2, dtype=torch.float32)

        result = apply_verify_cache_fill_policy(
            layer_idx=0,
            selected_experts=selected,
            routing_weights=routing_w,
            expert_cache=cache,
            step_id=8,
            profile={},
        )
        plan = build_verify_plan(
            layer_idx=0,
            selected_experts=selected,
            routing_weights=routing_w,
            expert_cache=cache,
            num_experts=8,
        )

        self.assertEqual(result.promoted_expert_ids, [2])
        self.assertEqual(result.cpu_expert_ids, [3, 4])
        self.assertTrue(cache.is_cached_cpu(2))
        self.assertFalse(cache.is_cached_cpu(3))
        self.assertFalse(cache.is_cached_cpu(4))
        self.assertEqual(plan.cpu_task_expert_ids.tolist(), [3, 4])

    def test_verify_cache_fill_never_evicts_active_cached_experts(self):
        cache = self._build_cache_with_slots(slots=3, cached=[0, 1, 5])
        cache.last_access_step[0] = -100
        cache.last_access_step[1] = -50
        cache.last_access_step[5] = 99
        selected = torch.tensor([[0, 1], [2, 2]], dtype=torch.int64)
        routing_w = torch.ones(2, 2, dtype=torch.float32)

        result = apply_verify_cache_fill_policy(
            layer_idx=0,
            selected_experts=selected,
            routing_weights=routing_w,
            expert_cache=cache,
            step_id=9,
            profile={},
        )

        self.assertEqual(result.evicted_expert_ids, [5])
        self.assertTrue(cache.is_cached_cpu(0))
        self.assertTrue(cache.is_cached_cpu(1))
        self.assertTrue(cache.is_cached_cpu(2))
        self.assertFalse(cache.is_cached_cpu(5))

    def test_verify_cache_fill_skips_pending_slots(self):
        cache = self._build_cache_with_slots(slots=3, cached=[0, 1, 5])
        reservation = cache.reserve_active_slot_for_prefetch_deferred(
            layer_idx=0,
            active_slot_idx=2,
            expert_idx=6,
        )
        self.assertIsNotNone(reservation)
        selected = torch.tensor([[0, 1], [2, 2]], dtype=torch.int64)
        routing_w = torch.ones(2, 2, dtype=torch.float32)

        result = apply_verify_cache_fill_policy(
            layer_idx=0,
            selected_experts=selected,
            routing_weights=routing_w,
            expert_cache=cache,
            step_id=10,
            profile={},
        )

        self.assertEqual(result.promoted_expert_count, 0)
        self.assertEqual(result.cpu_expert_ids, [2])
        self.assertTrue(cache.is_cached_cpu(5))
        self.assertFalse(cache.is_cached_cpu(2))
        self.assertEqual(result.skipped_pending_count, 1)

    def test_cache_fill_no_cpu_plan_uses_gpu_only_after_all_misses_loaded(self):
        cache = self._build_cache_with_slots(slots=4, cached=[0, 1])
        selected = torch.tensor([[0, 2], [3, 2]], dtype=torch.int64)
        routing_w = torch.ones(2, 2, dtype=torch.float32)
        profile = {}

        apply_verify_cache_fill_policy(
            layer_idx=0,
            selected_experts=selected,
            routing_weights=routing_w,
            expert_cache=cache,
            step_id=11,
            profile=profile,
        )
        plan = build_cache_fill_no_cpu_verify_plan_gpu(
            layer_idx=0,
            selected_experts=selected,
            routing_weights=routing_w,
            expert_cache=cache,
            num_experts=8,
            profile=profile,
        )

        self.assertIsNone(plan.cpu_route_indices)
        self.assertIsNone(plan.cpu_task_expert_ids)
        self.assertEqual(plan.gpu_route_indices.numel(), selected.numel())
        self.assertEqual(profile["verify_cache_fill_no_cpu_remaining_miss_count"], 0.0)
        self.assertEqual(profile["verify_cache_fill_no_cpu_fallback_count"], 0.0)

    def test_cache_fill_no_cpu_plan_records_remaining_misses_and_falls_back(self):
        cache = self._build_cache_with_slots(slots=3, cached=[0, 1])
        selected = torch.tensor([[0, 1], [2, 2], [3, 4]], dtype=torch.int64)
        routing_w = torch.ones(3, 2, dtype=torch.float32)
        profile = {}

        apply_verify_cache_fill_policy(
            layer_idx=0,
            selected_experts=selected,
            routing_weights=routing_w,
            expert_cache=cache,
            step_id=12,
            profile=profile,
        )
        plan = build_cache_fill_no_cpu_verify_plan_gpu(
            layer_idx=0,
            selected_experts=selected,
            routing_weights=routing_w,
            expert_cache=cache,
            num_experts=8,
            profile=profile,
        )

        self.assertEqual(profile["verify_cache_fill_no_cpu_remaining_miss_count"], 2.0)
        self.assertEqual(profile["verify_cache_fill_no_cpu_fallback_count"], 1.0)
        self.assertEqual(plan.cpu_task_expert_ids.tolist(), [3, 4])

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

    def test_draft_plan_graph_safe_topc_keeps_fixed_routes(self):
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
            graph_safe_cpu=True,
        )
        eager_plan = build_draft_plan(
            layer_idx=0,
            selected_experts=selected,
            routing_weights=routing_w,
            expert_cache=cache,
            draft_scheduler=scheduler,
            num_experts=8,
            top_c=1,
            graph_safe_cpu=False,
        )

        self.assertTrue(plan.cpu_graph_enabled)
        self.assertEqual(plan.gpu_route_indices.numel(), selected.numel())
        self.assertEqual(plan.cpu_route_indices.tolist(), [0, 1, 2, 3])
        self.assertEqual(plan.cpu_route_mask.tolist(), [False, False, True, False])
        self.assertTrue(torch.equal(plan.substitution_lut, eager_plan.substitution_lut))
        self.assertEqual(int(plan.substitution_lut[4].item()), 0)
        self.assertIsNotNone(plan.gpu_route_weights)
        self.assertEqual(float(plan.gpu_route_weights[2].item()), 0.0)
        self.assertIsNone(plan.cpu_task_expert_ids_host)
        self.assertIsNone(plan.cpu_task_offsets_host)

    def test_draft_plan_graph_async_uses_topc0_gpu_fallback(self):
        cache = self._build_cache()
        selected = torch.tensor([0, 4, 5, 2], dtype=torch.int64)
        routing_w = torch.tensor([[0.9, 0.1], [0.6, 0.4]], dtype=torch.float32)
        scheduler = SimpleDraftScheduler()
        topc0_plan = build_draft_plan(
            layer_idx=0,
            selected_experts=selected,
            routing_weights=routing_w,
            expert_cache=cache,
            draft_scheduler=scheduler,
            num_experts=8,
            top_c=0,
        )
        plan = build_draft_plan(
            layer_idx=0,
            selected_experts=selected,
            routing_weights=routing_w,
            expert_cache=cache,
            draft_scheduler=scheduler,
            num_experts=8,
            top_c=1,
            graph_safe_cpu=True,
            graph_async_cpu=True,
        )

        self.assertTrue(plan.cpu_graph_enabled)
        self.assertTrue(plan.cpu_graph_async)
        self.assertIsNone(plan.gpu_route_weights)
        self.assertEqual(plan.cpu_route_mask.tolist(), [False, False, True, False])
        self.assertTrue(torch.equal(plan.substitution_lut, topc0_plan.substitution_lut))
        self.assertTrue(torch.equal(plan.flat_selected_effective, topc0_plan.flat_selected_effective))

    def test_draft_plan_graph_safe_topc_ignores_inactive_bucket_rows(self):
        cache = self._build_cache()
        selected = torch.tensor([[0, 4], [5, 6]], dtype=torch.int64)
        routing_w = torch.tensor([[0.9, 0.1], [100.0, 0.2]], dtype=torch.float32)
        scheduler = SimpleDraftScheduler()
        plan = build_draft_plan(
            layer_idx=0,
            selected_experts=selected,
            routing_weights=routing_w,
            expert_cache=cache,
            draft_scheduler=scheduler,
            num_experts=8,
            top_c=1,
            graph_safe_cpu=True,
            active_token_mask=torch.tensor([True, False]),
        )

        self.assertEqual(plan.cpu_route_mask.tolist(), [False, True, False, False])

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

    def test_runtime_meta_helpers(self):
        selected = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        weights = torch.tensor([[0.1, 0.9], [0.3, 0.7]], dtype=torch.float32)
        flat_selected, flat_weights = flatten_selected_and_weights(selected, weights)
        view_selected, view_weights = build_runtime_meta_view(selected, weights)

        self.assertEqual(flat_selected.tolist(), [1, 2, 3, 4])
        self.assertEqual(len(flat_weights.tolist()), 4)
        self.assertTrue(torch.equal(view_selected, selected))
        self.assertTrue(torch.equal(view_weights, weights))


if __name__ == "__main__":
    unittest.main()
