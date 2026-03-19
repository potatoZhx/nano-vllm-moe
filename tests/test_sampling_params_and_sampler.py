import unittest

import torch

from nanovllm.layers.sampler import Sampler
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


if __name__ == "__main__":
    unittest.main()
