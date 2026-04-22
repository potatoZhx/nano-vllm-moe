import unittest
from types import SimpleNamespace

import torch

from nanovllm.expert.cache import LayerExpertCache
from nanovllm.expert.prefetcher import PrefetchRuntime
from nanovllm.expert.runtime_meta import LayerRuntimeMetaCPU
from nanovllm.scheduling.cache_strategy import create_cache_strategy
from nanovllm.scheduling.prefetch_strategy import create_prefetch_strategy


class TestVerifyFeedback(unittest.TestCase):
    def test_verify_consumed_count(self):
        cfg = SimpleNamespace(
            prefetch_step_budget=1,
            prefetch_max_inflight=1,
            cache_eviction_budget_per_step=1,
            prefetch_verify_wait_ms=0.0,
            prefetch_source_weight_prefill=1.0,
            prefetch_source_weight_verify=1.2,
            prefetch_source_weight_draft=1.5,
            prefetch_activation_count_weight=0.1,
            prefetch_age_penalty=0.01,
            prefetch_history_decay=0.9,
            prefetch_history_ttl_steps=8,
            prefetch_global_queue_capacity=64,
            prefetch_use_prefill_history=True,
            prefetch_use_verify_history=True,
            prefetch_use_draft_live=True,
        )
        cpu_pool = {
            0: {
                1: {"gate_up": torch.zeros(2, 2), "down": torch.zeros(2, 2)},
            }
        }
        cache = LayerExpertCache(
            num_experts=3,
            slots_per_layer=1,
            gate_up_shape=(2, 2),
            down_shape=(2, 2),
            device=torch.device("cpu"),
            dtype=torch.float32,
            cpu_expert_pool=cpu_pool[0],
        )
        cache.put_to_slot(0, 1, cpu_pool[0][1]["gate_up"], cpu_pool[0][1]["down"])

        runtime = PrefetchRuntime(
            config=cfg,
            layer_caches={0: cache},
            cpu_expert_pool=cpu_pool,
            cache_strategy=create_cache_strategy("lru"),
            prefetch_strategy=create_prefetch_strategy("noop", cfg),
            runtime_meta_recorder=SimpleNamespace(),
        )
        runtime._recent_published[(0, 1)] = 2

        runtime.record_verify_consumed(
            {
                0: LayerRuntimeMetaCPU(
                    step_id=3,
                    mode="verify",
                    layer_idx=0,
                    token_count=1,
                    selected_experts=torch.tensor([[1, 1]], dtype=torch.int64),
                    routing_weights=torch.tensor([[0.5, 0.5]], dtype=torch.float32),
                )
            },
            step_id=3,
        )
        prof = runtime.get_profile(reset=False)
        self.assertEqual(prof["prefetch_consumed_count"], 1)


if __name__ == "__main__":
    unittest.main()
