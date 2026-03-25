import unittest

import torch

from nanovllm.engine.speculative.acceptance import (
    GreedyAcceptance,
    StandardAcceptance,
    create_acceptance_strategy,
)


class TestAcceptanceStrategies(unittest.TestCase):
    def test_factory_returns_expected_strategy(self):
        self.assertIsInstance(create_acceptance_strategy("greedy"), GreedyAcceptance)
        self.assertIsInstance(create_acceptance_strategy("standard", threshold=0.5), StandardAcceptance)

    def test_factory_rejects_unknown_strategy(self):
        with self.assertRaises(ValueError):
            create_acceptance_strategy("unknown")

    def test_greedy_acceptance(self):
        draft_tokens = [3, 1, 5]
        # step0->3 match, step1->1 match, step2-> argmax=2 mismatch
        verify_logits = torch.tensor([
            [0.1, 0.2, 0.3, 0.9],
            [0.1, 0.8, 0.2, 0.0],
            [0.1, 0.2, 1.5, 0.0],
            [0.6, 0.2, 0.1, 0.0],
        ])
        strategy = GreedyAcceptance()
        out = strategy.accept(draft_tokens, verify_logits, temperature=0.0)
        self.assertEqual(out["num_accepted"], 2)
        self.assertEqual(out["next_token"], 2)

    def test_greedy_acceptance_with_verify_trace(self):
        strategy = GreedyAcceptance()
        out = strategy.accept([11, 12, 13], [11, 12, 9, 7], temperature=0.0)
        self.assertEqual(out["num_accepted"], 2)
        self.assertEqual(out["next_token"], 9)

    def test_standard_acceptance_threshold(self):
        draft_tokens = [0, 1, 2]
        verify_logits = torch.tensor([
            [5.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
        ])
        strategy = StandardAcceptance(threshold=0.5)
        out = strategy.accept(draft_tokens, verify_logits, temperature=1.0)
        self.assertEqual(out["num_accepted"], 2)
        self.assertEqual(out["next_token"], 0)


if __name__ == "__main__":
    unittest.main()
