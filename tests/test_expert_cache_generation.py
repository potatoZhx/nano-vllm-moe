import unittest

import torch

from nanovllm.expert.cache import LayerExpertCache, StagingReservation


class TestExpertCacheGeneration(unittest.TestCase):
    def test_stale_generation_is_rejected(self):
        cache = LayerExpertCache(
            num_experts=4,
            slots_per_layer=2,
            gate_up_shape=(4, 4),
            down_shape=(4, 2),
            device=torch.device("cpu"),
            dtype=torch.float32,
            cpu_expert_pool={},
            staging_slots_per_layer=1,
            enable_prefetch=True,
        )

        old_res = cache.reserve_staging_slot(2)
        self.assertIsNotNone(old_res)
        cache.cancel_staging_reservation(old_res)

        new_res = cache.reserve_staging_slot(3)
        self.assertIsNotNone(new_res)
        self.assertGreater(new_res.generation, old_res.generation)

        stale = StagingReservation(
            layer_idx=0,
            staging_slot_idx=old_res.staging_slot_idx,
            expert_idx=old_res.expert_idx,
            generation=old_res.generation,
        )
        self.assertFalse(cache.mark_staging_ready(stale))


if __name__ == "__main__":
    unittest.main()
