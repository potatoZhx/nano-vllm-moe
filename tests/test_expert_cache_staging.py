import unittest

import torch

from nanovllm.expert.cache import LayerExpertCache, StagingReservation


class TestExpertCacheStaging(unittest.TestCase):
    def _build_cache(self) -> LayerExpertCache:
        return LayerExpertCache(
            num_experts=4,
            slots_per_layer=2,
            gate_up_shape=(4, 4),
            down_shape=(4, 2),
            device=torch.device("cpu"),
            dtype=torch.float32,
            cpu_expert_pool={},
            staging_slots_per_layer=2,
            enable_prefetch=True,
        )

    def test_staging_publish_lifecycle(self):
        cache = self._build_cache()
        gate_up_0 = torch.ones(4, 4)
        down_0 = torch.ones(4, 2)
        cache.put_to_slot(0, 0, gate_up_0, down_0)

        reservation = cache.reserve_staging_slot(expert_idx=3)
        self.assertIsNotNone(reservation)
        evt = cache.begin_async_put_to_staging(
            reservation=reservation,
            gate_up_cpu=torch.full((4, 4), 7.0),
            down_cpu=torch.full((4, 2), 9.0),
            stream=None,
        )
        self.assertTrue(evt.query())
        self.assertTrue(cache.mark_staging_ready(reservation))

        published = cache.publish_ready_staging_to_active(
            reservation=reservation,
            active_slot_idx=1,
            stream=None,
        )
        self.assertIsNotNone(published)
        cache.commit_published_expert(published)

        self.assertEqual(cache.get_slot_idx(3), 1)
        self.assertTrue(bool(cache.get_cached_expert_mask()[3].item()))

    def test_mark_access_updates_stats(self):
        cache = self._build_cache()
        experts = torch.tensor([[1, 2], [1, 1]], dtype=torch.int64)
        weights = torch.tensor([[0.3, 0.7], [0.2, 0.5]], dtype=torch.float32)
        cache.mark_access(experts, weights, step_id=11)

        self.assertEqual(cache.last_access_step[1], 11)
        self.assertEqual(cache.access_count[1], 1)
        self.assertGreater(cache.access_score_sum[1], 0.0)

    def test_deferred_active_prefetch_does_not_publish_mapping_until_commit(self):
        cache = self._build_cache()
        cache.put_to_slot(0, 0, torch.ones(4, 4), torch.ones(4, 2))

        reservation = cache.reserve_active_slot_for_prefetch_deferred(
            layer_idx=0,
            active_slot_idx=0,
            expert_idx=3,
        )
        self.assertIsNotNone(reservation)
        self.assertTrue(cache.is_active_slot_pending(0))
        self.assertTrue(cache.is_cached_cpu(0))
        self.assertFalse(cache.is_cached_cpu(3))

        evt = cache.begin_async_put_to_active(
            reservation=reservation,
            gate_up_cpu=torch.full((4, 4), 7.0),
            down_cpu=torch.full((4, 2), 9.0),
            stream=None,
        )
        self.assertTrue(evt.query())
        published = cache.commit_deferred_active_prefetch(reservation)

        self.assertIsNotNone(published)
        self.assertFalse(cache.is_active_slot_pending(0))
        self.assertFalse(cache.is_cached_cpu(0))
        self.assertTrue(cache.is_cached_cpu(3))


if __name__ == "__main__":
    unittest.main()
