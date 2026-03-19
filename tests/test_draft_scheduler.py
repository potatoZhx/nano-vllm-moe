import unittest

import torch

from nanovllm.scheduling.draft_scheduler import SimpleDraftScheduler


class TestSimpleDraftScheduler(unittest.TestCase):
    def test_select_cpu_experts(self):
        scheduler = SimpleDraftScheduler()
        uncached = [3, 5, 7]
        selected = torch.tensor([3, 3, 5, 7, 7, 7], dtype=torch.int64)
        weights = torch.tensor([0.9, 0.1, 0.2, 0.4, 0.3, 0.2], dtype=torch.float32)
        picked = scheduler.select_cpu_experts(uncached, weights, selected, top_c=2)
        self.assertEqual(picked, [3, 7])

    def test_select_gpu_substitutes(self):
        scheduler = SimpleDraftScheduler()
        mapping = scheduler.select_gpu_substitutes([10, 11, 12], {1, 2}, list(range(16)))
        self.assertEqual(mapping[10], 1)
        self.assertEqual(mapping[11], 2)
        self.assertEqual(mapping[12], 1)


if __name__ == "__main__":
    unittest.main()
