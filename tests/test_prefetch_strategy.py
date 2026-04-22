import unittest
from types import SimpleNamespace

from nanovllm.expert.prefetcher import PrefetchCandidate
from nanovllm.scheduling.prefetch_strategy import (
    HistoryWindowPrefetchStrategy,
    NoopPrefetchStrategy,
    create_prefetch_strategy,
)


class TestPrefetchStrategy(unittest.TestCase):
    def _config(self):
        return SimpleNamespace(prefetch_history_ttl_steps=3)

    def _candidates(self):
        return [
            PrefetchCandidate(0, 1, "prefill_history", 2.0, 4, 1, 1, 5.0),
            PrefetchCandidate(0, 2, "draft_live", 1.0, 1, 1, 4, 2.0),
            PrefetchCandidate(1, 0, "verify_history", 3.0, 3, 1, 5, 6.0),
        ]

    def test_noop_keeps_candidates(self):
        strategy = NoopPrefetchStrategy()
        ranked = strategy.rank(self._candidates(), step_id=6)
        self.assertEqual(len(ranked), 3)

    def test_history_window_filters_by_ttl_and_sorts(self):
        strategy = HistoryWindowPrefetchStrategy(self._config())
        ranked = strategy.rank(self._candidates(), step_id=6)
        self.assertEqual([(c.layer_idx, c.expert_idx) for c in ranked], [(1, 0), (0, 2)])

    def test_factory(self):
        cfg = self._config()
        self.assertIsInstance(create_prefetch_strategy("noop", cfg), NoopPrefetchStrategy)
        self.assertIsInstance(create_prefetch_strategy("history_window", cfg), HistoryWindowPrefetchStrategy)
        with self.assertRaises(ValueError):
            create_prefetch_strategy("x", cfg)


if __name__ == "__main__":
    unittest.main()
