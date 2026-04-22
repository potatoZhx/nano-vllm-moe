import unittest

import torch

from nanovllm.expert.cache import LayerCacheSnapshot
from nanovllm.scheduling.cache_strategy import (
    LFUCacheStrategy,
    LRUCacheStrategy,
    create_cache_strategy,
)


class TestCacheStrategy(unittest.TestCase):
    def _snapshot(self) -> LayerCacheSnapshot:
        return LayerCacheSnapshot(
            layer_idx=0,
            cached_expert_mask=torch.tensor([True, True, True, False], dtype=torch.bool),
            expert_to_slot_lut=torch.tensor([0, 1, 2, -1], dtype=torch.int64),
            slot_to_expert_lut=torch.tensor([0, 1, 2], dtype=torch.int64),
            last_access_step=[5, 2, 9, -1],
            access_count=[10, 1, 4, 0],
            access_score_sum=[3.0, 0.5, 1.2, 0.0],
            slot_generation=[1, 1, 1],
        )

    def test_lru_selects_oldest(self):
        strategy = LRUCacheStrategy()
        victim = strategy.select_victim_slot(self._snapshot(), incoming_expert_idx=3, step_id=10)
        self.assertEqual(victim, 1)

    def test_lfu_selects_least_frequent(self):
        strategy = LFUCacheStrategy()
        victim = strategy.select_victim_slot(self._snapshot(), incoming_expert_idx=3, step_id=10)
        self.assertEqual(victim, 1)

    def test_factory(self):
        self.assertIsInstance(create_cache_strategy("lru"), LRUCacheStrategy)
        self.assertIsInstance(create_cache_strategy("lfu"), LFUCacheStrategy)
        with self.assertRaises(ValueError):
            create_cache_strategy("x")


if __name__ == "__main__":
    unittest.main()
