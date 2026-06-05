import unittest
from types import SimpleNamespace

import torch

from nanovllm.expert.cache import LayerExpertCache
from nanovllm.expert.prefetcher import PrefetchRuntime
from nanovllm.expert.runtime_meta import LayerRuntimeMetaCPU
from nanovllm.scheduling.cache_strategy import LFURankGuardStrategy
from nanovllm.scheduling.cache_strategy import create_cache_strategy
from nanovllm.scheduling.prefetch_strategy import create_prefetch_strategy


class TestPrefetchRuntime(unittest.TestCase):
    def _config(self):
        return SimpleNamespace(
            prefetch_step_budget=4,
            prefetch_max_inflight=8,
            cache_eviction_budget_per_step=2,
            prefetch_verify_wait_ms=0.0,
            prefetch_source_weight_prefill=1.0,
            prefetch_source_weight_verify=1.2,
            prefetch_source_weight_draft=1.5,
            prefetch_activation_count_weight=0.1,
            prefetch_age_penalty=0.01,
            prefetch_history_decay=0.9,
            prefetch_history_ttl_steps=16,
            prefetch_global_queue_capacity=32,
            prefetch_use_prefill_history=True,
            prefetch_use_verify_history=True,
            prefetch_use_draft_live=True,
            prefetch_verify_layer_enabled=True,
            prefetch_verify_layer_transfer_bandwidth_gbps=12.0,
            prefetch_verify_layer_max_budget=2,
            prefetch_runtime_mode="baseline_staging",
            draft_prefetch_frontier_granularity="segment",
            draft_prefetch_segment_size=4,
            draft_prefetch_visible_budget_ms=3.0,
            draft_prefetch_min_per_boundary=0,
            draft_prefetch_max_per_boundary=4,
        )

    def _build_runtime(self):
        cfg = self._config()
        cpu_pool = {
            0: {
                0: {"gate_up": torch.zeros(2, 2), "down": torch.zeros(2, 2)},
                2: {"gate_up": torch.ones(2, 2), "down": torch.ones(2, 2)},
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
            staging_slots_per_layer=2,
            enable_prefetch=True,
        )
        cache.put_to_slot(0, 0, cpu_pool[0][0]["gate_up"], cpu_pool[0][0]["down"])

        runtime = PrefetchRuntime(
            config=cfg,
            layer_caches={0: cache},
            cpu_expert_pool=cpu_pool,
            cache_strategy=create_cache_strategy("lru"),
            prefetch_strategy=create_prefetch_strategy("noop", cfg),
            runtime_meta_recorder=SimpleNamespace(),
        )
        return runtime, cache

    def test_submit_and_publish(self):
        runtime, cache = self._build_runtime()
        runtime_meta = {
            0: LayerRuntimeMetaCPU(
                step_id=1,
                mode="prefill",
                layer_idx=0,
                token_count=2,
                selected_experts=torch.tensor([[2, 2]], dtype=torch.int64),
                routing_weights=torch.tensor([[0.6, 0.4]], dtype=torch.float32),
            )
        }
        runtime.observe_prefill(runtime_meta, step_id=1)
        submitted = runtime.submit_from_global_queue(step_id=1, phase="before_draft")
        self.assertGreaterEqual(submitted, 1)

        published = runtime.publish_ready(step_id=1)
        self.assertGreaterEqual(published, 1)
        self.assertTrue(bool(cache.get_cached_expert_mask()[2].item()))

        prof = runtime.get_profile(reset=False)
        self.assertGreaterEqual(prof["prefetch_submit_count"], 1)
        self.assertGreaterEqual(prof["publish_count"], 1)

    def test_verify_layer_prefetch_direct_active_publish(self):
        cfg = self._config()
        cpu_pool = {
            1: {
                0: {"gate_up": torch.zeros(2, 2), "down": torch.zeros(2, 2)},
                2: {"gate_up": torch.ones(2, 2), "down": torch.ones(2, 2)},
            }
        }
        cache = LayerExpertCache(
            num_experts=3,
            slots_per_layer=1,
            gate_up_shape=(2, 2),
            down_shape=(2, 2),
            device=torch.device("cpu"),
            dtype=torch.float32,
            cpu_expert_pool=cpu_pool[1],
            staging_slots_per_layer=0,
            enable_prefetch=False,
        )
        cache.put_to_slot(0, 0, cpu_pool[1][0]["gate_up"], cpu_pool[1][0]["down"])
        runtime = PrefetchRuntime(
            config=cfg,
            layer_caches={1: cache},
            cpu_expert_pool=cpu_pool,
            cache_strategy=create_cache_strategy("lru"),
            prefetch_strategy=create_prefetch_strategy("noop", cfg),
            runtime_meta_recorder=SimpleNamespace(),
        )
        runtime.observe_prefill(
            {
                1: LayerRuntimeMetaCPU(
                    step_id=1,
                    mode="prefill",
                    layer_idx=1,
                    token_count=1,
                    selected_experts=torch.tensor([[2, 2]], dtype=torch.int64),
                    routing_weights=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
                )
            },
            step_id=1,
        )

        submitted = runtime.submit_verify_layer_prefetch(
            step_id=2,
            target_layer_idx=1,
            available_ms=1.0,
        )
        self.assertEqual(submitted, 1)
        self.assertFalse(cache.is_cached_cpu(0))
        self.assertFalse(cache.is_cached_cpu(2))

        published = runtime.publish_direct_active_ready(step_id=2)
        self.assertEqual(published, 1)
        self.assertTrue(cache.is_cached_cpu(2))

        prof = runtime.get_profile(reset=False)
        self.assertEqual(prof["verify_layer_prefetch_submit_count"], 1)
        self.assertEqual(prof["verify_layer_prefetch_publish_count"], 1)

    def test_draft_direct_active_prefetch_respects_frontier(self):
        cfg = self._config()
        cfg.prefetch_runtime_mode = "draft_direct_active"
        cfg.prefetch_staging_slots_per_layer = 0
        cpu_pool = {
            0: {
                0: {"gate_up": torch.zeros(2, 2), "down": torch.zeros(2, 2)},
                2: {"gate_up": torch.ones(2, 2), "down": torch.ones(2, 2)},
            },
            1: {
                0: {"gate_up": torch.zeros(2, 2), "down": torch.zeros(2, 2)},
                2: {"gate_up": torch.ones(2, 2), "down": torch.ones(2, 2)},
            },
        }
        caches = {}
        for layer_idx in (0, 1):
            cache = LayerExpertCache(
                num_experts=3,
                slots_per_layer=1,
                gate_up_shape=(2, 2),
                down_shape=(2, 2),
                device=torch.device("cpu"),
                dtype=torch.float32,
                cpu_expert_pool=cpu_pool[layer_idx],
                staging_slots_per_layer=0,
                enable_prefetch=False,
            )
            cache.put_to_slot(0, 0, cpu_pool[layer_idx][0]["gate_up"], cpu_pool[layer_idx][0]["down"])
            caches[layer_idx] = cache

        runtime = PrefetchRuntime(
            config=cfg,
            layer_caches=caches,
            cpu_expert_pool=cpu_pool,
            cache_strategy=create_cache_strategy("lru"),
            prefetch_strategy=create_prefetch_strategy("noop", cfg),
            runtime_meta_recorder=SimpleNamespace(),
        )
        runtime.observe_draft(
            {
                0: LayerRuntimeMetaCPU(
                    step_id=3,
                    mode="draft",
                    layer_idx=0,
                    token_count=1,
                    selected_experts=torch.tensor([[2]], dtype=torch.int64),
                    routing_weights=torch.tensor([[1.0]], dtype=torch.float32),
                ),
                1: LayerRuntimeMetaCPU(
                    step_id=3,
                    mode="draft",
                    layer_idx=1,
                    token_count=1,
                    selected_experts=torch.tensor([[2]], dtype=torch.int64),
                    routing_weights=torch.tensor([[1.0]], dtype=torch.float32),
                ),
            },
            step_id=3,
        )

        submitted = runtime.submit_draft_direct_active_prefetch(
            step_id=3,
            phase="after_draft",
            frontier_layer_idx=0,
            visible_budget_ms=10.0,
        )
        self.assertEqual(submitted, 1)
        self.assertFalse(caches[0].is_cached_cpu(2))
        self.assertFalse(caches[1].is_cached_cpu(2))

        published = runtime.publish_direct_active_ready(step_id=3)
        self.assertEqual(published, 1)
        self.assertTrue(caches[0].is_cached_cpu(2))
        self.assertFalse(caches[1].is_cached_cpu(2))

        prof = runtime.get_profile(reset=False)
        self.assertEqual(prof["draft_direct_active_prefetch_submit_count"], 1)
        self.assertEqual(prof["draft_direct_active_prefetch_publish_count"], 1)
        self.assertEqual(prof["draft_direct_active_prefetch_skipped_by_frontier_count"], 0)
        self.assertEqual(prof["staging_prefetch_submit_count"], 0)

    def test_draft_segment_indexed_prefetch_uses_segment_index_and_deferred_mapping(self):
        cfg = self._config()
        cfg.prefetch_runtime_mode = "draft_segment_indexed"
        cfg.prefetch_staging_slots_per_layer = 0
        cfg.draft_prefetch_segment_size = 1
        cpu_pool = {
            0: {
                0: {"gate_up": torch.zeros(2, 2), "down": torch.zeros(2, 2)},
                2: {"gate_up": torch.ones(2, 2), "down": torch.ones(2, 2)},
            },
            1: {
                0: {"gate_up": torch.zeros(2, 2), "down": torch.zeros(2, 2)},
                2: {"gate_up": torch.ones(2, 2), "down": torch.ones(2, 2)},
            },
        }
        caches = {}
        for layer_idx in (0, 1):
            cache = LayerExpertCache(
                num_experts=3,
                slots_per_layer=1,
                gate_up_shape=(2, 2),
                down_shape=(2, 2),
                device=torch.device("cpu"),
                dtype=torch.float32,
                cpu_expert_pool=cpu_pool[layer_idx],
                staging_slots_per_layer=0,
                enable_prefetch=False,
            )
            cache.put_to_slot(0, 0, cpu_pool[layer_idx][0]["gate_up"], cpu_pool[layer_idx][0]["down"])
            caches[layer_idx] = cache

        runtime = PrefetchRuntime(
            config=cfg,
            layer_caches=caches,
            cpu_expert_pool=cpu_pool,
            cache_strategy=create_cache_strategy("lru"),
            prefetch_strategy=create_prefetch_strategy("noop", cfg),
            runtime_meta_recorder=SimpleNamespace(),
        )
        runtime.begin_draft_iteration(step_id=3)
        runtime.observe_draft(
            {
                0: LayerRuntimeMetaCPU(
                    step_id=3,
                    mode="draft",
                    layer_idx=0,
                    token_count=1,
                    selected_experts=torch.tensor([[2]], dtype=torch.int64),
                    routing_weights=torch.tensor([[1.0]], dtype=torch.float32),
                ),
                1: LayerRuntimeMetaCPU(
                    step_id=3,
                    mode="draft",
                    layer_idx=1,
                    token_count=1,
                    selected_experts=torch.tensor([[2]], dtype=torch.int64),
                    routing_weights=torch.tensor([[1.0]], dtype=torch.float32),
                ),
            },
            step_id=3,
        )

        submitted = runtime.submit_draft_segment_indexed_prefetch(
            step_id=3,
            phase="after_draft_segment",
            frontier_layer_idx=0,
            visible_budget_ms=10.0,
        )
        self.assertEqual(submitted, 1)
        self.assertTrue(caches[0].is_cached_cpu(0))
        self.assertFalse(caches[0].is_cached_cpu(2))
        self.assertFalse(caches[1].is_pending_cpu(2))

        published = runtime.publish_direct_active_ready(step_id=3)
        self.assertEqual(published, 1)
        self.assertFalse(caches[0].is_cached_cpu(0))
        self.assertTrue(caches[0].is_cached_cpu(2))
        self.assertFalse(caches[1].is_cached_cpu(2))
        runtime.record_verify_consumed(
            {
                0: LayerRuntimeMetaCPU(
                    step_id=4,
                    mode="verify",
                    layer_idx=0,
                    token_count=1,
                    selected_experts=torch.tensor([[2]], dtype=torch.int64),
                    routing_weights=torch.tensor([[1.0]], dtype=torch.float32),
                )
            },
            step_id=4,
        )

        prof = runtime.get_profile(reset=False)
        expected_bytes = (
            cpu_pool[0][2]["gate_up"].numel() * cpu_pool[0][2]["gate_up"].element_size()
            + cpu_pool[0][2]["down"].numel() * cpu_pool[0][2]["down"].element_size()
        )
        self.assertEqual(prof["draft_segment_indexed_prefetch_submit_count"], 1)
        self.assertEqual(prof["draft_segment_indexed_prefetch_publish_count"], 1)
        self.assertEqual(prof["draft_segment_indexed_prefetch_submit_count_by_segment"], {"0": 1})
        self.assertEqual(prof["draft_segment_indexed_prefetch_ready_count_by_segment"], {"0": 1})
        self.assertEqual(prof["draft_segment_indexed_prefetch_success_count_by_segment"], {"0": 1})
        self.assertEqual(prof["draft_segment_indexed_prefetch_consumed_count_by_segment"], {"0": 1})
        self.assertEqual(prof["draft_segment_indexed_prefetch_skipped_by_pending_count"], 0)
        self.assertEqual(prof["prefetch_submitted_bytes"], expected_bytes)
        self.assertEqual(prof["prefetch_completed_bytes"], expected_bytes)
        self.assertEqual(prof["prefetch_published_bytes"], expected_bytes)
        self.assertEqual(prof["prefetch_late_bytes"], 0)
        self.assertEqual(prof["draft_segment_indexed_prefetch_submitted_bytes"], expected_bytes)
        self.assertEqual(prof["draft_segment_indexed_prefetch_completed_bytes"], expected_bytes)
        self.assertEqual(prof["draft_segment_indexed_prefetch_published_bytes"], expected_bytes)
        self.assertEqual(prof["prefetch_submit_count_by_source"], {"draft_segment_indexed": 1})
        self.assertEqual(prof["prefetch_completed_count_by_source"], {"draft_segment_indexed": 1})
        self.assertEqual(prof["prefetch_published_count_by_source"], {"draft_segment_indexed": 1})
        self.assertEqual(prof["prefetch_submitted_bytes_by_source"], {"draft_segment_indexed": expected_bytes})
        self.assertEqual(prof["prefetch_completed_bytes_by_source"], {"draft_segment_indexed": expected_bytes})
        self.assertEqual(prof["prefetch_published_bytes_by_source"], {"draft_segment_indexed": expected_bytes})
        self.assertGreaterEqual(prof["prefetch_max_inflight_observed"], 1)
        self.assertGreaterEqual(prof["draft_segment_indexed_prefetch_reservation_ms"], 0.0)
        self.assertGreaterEqual(prof["draft_segment_indexed_prefetch_transfer_enqueue_ms"], 0.0)
        self.assertGreaterEqual(prof["draft_segment_indexed_prefetch_completion_latency_ms"], 0.0)

    def test_draft_direct_active_budget_can_adapt_down(self):
        runtime, _cache = self._build_runtime()
        runtime.config.draft_prefetch_min_per_boundary = 0
        runtime.config.draft_prefetch_max_per_boundary = 2
        runtime.config.draft_prefetch_visible_budget_ms = 0.1
        runtime._draft_direct_active_budget = 2

        runtime._adjust_draft_direct_active_budget(visible_ms=1.0)

        prof = runtime.get_profile(reset=False)
        self.assertEqual(prof["draft_direct_active_prefetch_adaptive_budget"], 1)
        self.assertEqual(prof["draft_direct_active_prefetch_budget_decrease_count"], 1)

    def test_draft_direct_active_budget_recovers_from_zero(self):
        runtime, _cache = self._build_runtime()
        runtime.config.draft_prefetch_min_per_boundary = 0
        runtime.config.draft_prefetch_max_per_boundary = 2
        runtime.config.draft_prefetch_visible_budget_ms = 3.0
        runtime._draft_direct_active_budget = 0

        submitted = runtime.submit_draft_direct_active_prefetch(
            step_id=4,
            phase="after_draft",
            frontier_layer_idx=0,
            visible_budget_ms=3.0,
        )

        self.assertEqual(submitted, 0)
        prof = runtime.get_profile(reset=False)
        self.assertEqual(prof["draft_direct_active_prefetch_adaptive_budget"], 1)
        self.assertEqual(prof["draft_direct_active_prefetch_budget_increase_count"], 1)
        self.assertEqual(prof["draft_direct_active_prefetch_skipped_by_budget_count"], 1)

    def test_lfu_rankguard_updates_scores_even_without_verify_history_queue(self):
        runtime, _cache = self._build_runtime()
        runtime.config.prefetch_use_verify_history = False
        strategy = LFURankGuardStrategy(num_experts=3, ema_alpha=0.0)
        runtime.cache_strategy = strategy

        runtime.observe_verify(
            {
                0: LayerRuntimeMetaCPU(
                    step_id=5,
                    mode="verify",
                    layer_idx=0,
                    token_count=2,
                    selected_experts=torch.tensor([[2, 1], [2, 0]], dtype=torch.int64),
                    routing_weights=torch.tensor([[0.6, 0.4], [0.7, 0.3]], dtype=torch.float32),
                )
            },
            step_id=5,
        )

        self.assertEqual(strategy.get_rank_scores(0), [0.5, 0.5, 2.0])


if __name__ == "__main__":
    unittest.main()
