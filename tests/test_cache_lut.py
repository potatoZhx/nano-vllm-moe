import unittest

import torch

from nanovllm.expert.cache_lut import (
    commit_cache_slot,
    unmap_cache_slot,
    warmup_cache_lut_kernels,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestFusedCacheLut(unittest.TestCase):
    def test_dynamic_indices_preserve_mapping_semantics(self):
        device = torch.device("cuda")
        warmup_cache_lut_kernels(device)
        expert_to_slot = torch.full(
            (16,), -1, dtype=torch.int64, device=device
        )
        slot_to_expert = torch.full(
            (8,), -1, dtype=torch.int64, device=device
        )
        cached_mask = torch.zeros(
            (16,), dtype=torch.bool, device=device
        )

        commit_cache_slot(
            expert_to_slot,
            slot_to_expert,
            cached_mask,
            previous_expert=-1,
            expert_idx=7,
            slot_idx=4,
            evict_previous=False,
        )
        commit_cache_slot(
            expert_to_slot,
            slot_to_expert,
            cached_mask,
            previous_expert=7,
            expert_idx=11,
            slot_idx=4,
            evict_previous=True,
        )
        torch.cuda.synchronize()
        self.assertEqual(int(expert_to_slot[7]), -1)
        self.assertEqual(int(expert_to_slot[11]), 4)
        self.assertEqual(int(slot_to_expert[4]), 11)
        self.assertFalse(bool(cached_mask[7]))
        self.assertTrue(bool(cached_mask[11]))

        unmap_cache_slot(
            expert_to_slot,
            slot_to_expert,
            cached_mask,
            previous_expert=11,
            slot_idx=4,
        )
        torch.cuda.synchronize()
        self.assertEqual(int(expert_to_slot[11]), -1)
        self.assertEqual(int(slot_to_expert[4]), -1)
        self.assertFalse(bool(cached_mask[11]))
