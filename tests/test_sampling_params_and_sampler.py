import unittest

import torch

from nanovllm.layers.sampler import Sampler, filtered_sampling_probs
from nanovllm.sampling_params import SamplingParams


class TestDeterministicSampling(unittest.TestCase):
    def test_sampling_params_allow_zero_temperature(self):
        sp = SamplingParams(temperature=0.0, max_tokens=8)
        self.assertEqual(sp.temperature, 0.0)

    def test_sampler_greedy_when_temperature_zero(self):
        sampler = Sampler()
        logits = torch.tensor([[0.1, 2.0, 1.0], [5.0, 1.0, 1.0]], device="cpu")
        temps = torch.tensor([0.0, 0.0], device="cpu")

        out = sampler(logits, temps)
        self.assertEqual(out.tolist(), [1, 0])

    def test_sampling_params_validate_top_k_and_top_p(self):
        sp = SamplingParams(temperature=0.6, top_k=20, top_p=0.95)
        self.assertEqual(sp.top_k, 20)
        self.assertEqual(sp.top_p, 0.95)
        with self.assertRaises(AssertionError):
            SamplingParams(top_k=-1)
        with self.assertRaises(AssertionError):
            SamplingParams(top_p=0.0)

    def test_filtered_probs_apply_top_k_then_top_p(self):
        logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
        probs = filtered_sampling_probs(
            logits, 1.0, top_k=2, top_p=0.7
        )
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=6)
        self.assertGreater(float(probs[0, 0]), 0.0)
        self.assertEqual(float(probs[0, 1]), 0.0)
        self.assertEqual(float(probs[0, 2]), 0.0)
        self.assertEqual(float(probs[0, 3]), 0.0)

    def test_sampler_top_k_one_always_selects_argmax(self):
        sampler = Sampler()
        logits = torch.tensor([[0.1, 2.0, 1.0], [5.0, 6.0, 1.0]])
        temps = torch.tensor([0.6, 0.6])
        out = sampler(logits, temps, [1, 1], [1.0, 1.0])
        self.assertEqual(out.tolist(), [1, 1])


if __name__ == "__main__":
    unittest.main()
