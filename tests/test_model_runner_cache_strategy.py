import unittest
from types import SimpleNamespace

from nanovllm.engine.model_runner import _create_cache_strategy_from_config
from nanovllm.scheduling.cache_strategy import LFUOnlineRankGuardStrategy


class TestModelRunnerCacheStrategy(unittest.TestCase):
    def test_online_rankguard_receives_configured_parameters(self):
        strategy = _create_cache_strategy_from_config(
            SimpleNamespace(
                cache_strategy="lfu_rankguard_online",
                hf_config=SimpleNamespace(num_experts=4),
                rank_guard_threshold=0.2,
                rank_guard_ema_alpha=0.8,
            )
        )

        self.assertIsInstance(strategy, LFUOnlineRankGuardStrategy)
        self.assertEqual(strategy.num_experts, 4)
        self.assertEqual(strategy.protect_threshold, 0.2)


if __name__ == "__main__":
    unittest.main()
