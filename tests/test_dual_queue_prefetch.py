import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from nanovllm.expert.cache import LayerExpertCache
from nanovllm.expert.prefetcher import DualQueuePrefetchRuntime
from nanovllm.expert.runtime_meta import LayerRuntimeMetaCPU, ModelRuntimeMetaRecorder
from nanovllm.scheduling.cache_strategy import create_cache_strategy
from nanovllm.scheduling.prefetch_strategy import create_prefetch_strategy


def _config(**overrides):
    cfg = SimpleNamespace(
        prefetch_step_budget=4,
        prefetch_max_inflight=8,
        prefetch_transfer_stream_count=1,
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
        prefetch_verify_layer_enabled=False,
        prefetch_verify_layer_transfer_bandwidth_gbps=12.0,
        prefetch_verify_layer_max_budget=2,
        prefetch_runtime_mode="draft_segment_indexed",
        prefetch_runtime_kind="dual_queue",
        draft_prefetch_frontier_granularity="segment",
        draft_prefetch_segment_size=2,
        draft_prefetch_visible_budget_ms=3.0,
        draft_prefetch_min_per_boundary=0,
        draft_prefetch_max_per_boundary=2,
        verify_prefetch_max_per_boundary=2,
        verify_prefetch_segment_size=2,
        cache_strategy="lru",
        dual_queue_segment_size=2,
        dual_queue_ground_truth_decay=0.5,
        dual_queue_ground_truth_ttl_rounds=2,
        dual_queue_ground_truth_count_weight=0.25,
        dual_queue_budget_safety_ratio=0.8,
        dual_queue_segment_time_ema_alpha=0.2,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _weights(value=1.0):
    return {
        "gate_up": torch.full((2, 2), float(value)),
        "down": torch.full((2, 2), float(value)),
    }


def _cache(pool, slots=2, staging_slots=2):
    return LayerExpertCache(
        num_experts=4,
        slots_per_layer=slots,
        gate_up_shape=(2, 2),
        down_shape=(2, 2),
        device=torch.device("cpu"),
        dtype=torch.float32,
        cpu_expert_pool=pool,
        staging_slots_per_layer=staging_slots,
        enable_prefetch=True,
    )


def _runtime(num_layers=4, cache_slots=2, staging_slots=2, **config_overrides):
    cfg = _config(**config_overrides)
    cpu_pool = {
        layer_idx: {expert_idx: _weights(layer_idx + expert_idx + 1) for expert_idx in range(4)}
        for layer_idx in range(num_layers)
    }
    caches = {
        layer_idx: _cache(
            cpu_pool[layer_idx],
            slots=cache_slots,
            staging_slots=staging_slots,
        )
        for layer_idx in range(num_layers)
    }
    runtime = DualQueuePrefetchRuntime(
        config=cfg,
        layer_caches=caches,
        cpu_expert_pool=cpu_pool,
        cache_strategy=create_cache_strategy(cfg.cache_strategy),
        prefetch_strategy=create_prefetch_strategy("noop", cfg),
        runtime_meta_recorder=SimpleNamespace(),
    )
    return runtime, caches


def _aggregated_meta(layer_idx, experts, scores, counts=None, token_count=1, mode="draft"):
    return {
        layer_idx: LayerRuntimeMetaCPU(
            step_id=1,
            mode=mode,
            layer_idx=layer_idx,
            token_count=token_count,
            aggregated_expert_ids=torch.tensor(experts, dtype=torch.int64),
            aggregated_score_sum=torch.tensor(scores, dtype=torch.float32),
            aggregated_activation_count=(
                None if counts is None else torch.tensor(counts, dtype=torch.int64)
            ),
        )
    }


class _ManualEvent:
    def __init__(self, ready=False):
        self.ready = bool(ready)

    def query(self):
        return self.ready

    def synchronize(self):
        self.ready = True


class TestDualQueueData(unittest.TestCase):
    def test_draft_score_sum_accumulates_and_protects(self):
        runtime, _ = _runtime()
        runtime.begin_draft_iteration(step_id=7)
        runtime.observe_draft(
            _aggregated_meta(0, [2], [0.4], counts=None),
            step_id=7,
        )
        runtime.observe_draft(
            _aggregated_meta(0, [2], [0.6], counts=None),
            step_id=7,
        )

        entry = runtime.draft_predict_index.entries_by_segment[0][(0, 2)]
        self.assertAlmostEqual(entry.score_sum, 1.0)
        self.assertIn(2, runtime._round_protected[0])
        self.assertEqual(len(runtime.global_queue.entries), 0)
        self.assertEqual(len(runtime.draft_segment_index.entries_by_segment), 0)

    def test_stale_draft_metadata_is_dropped_after_transition(self):
        runtime, _ = _runtime()
        runtime.begin_draft_iteration(step_id=7)
        runtime.end_draft_iteration()
        runtime.observe_draft(_aggregated_meta(0, [2], [1.0]), step_id=7)

        self.assertEqual(len(runtime.draft_predict_index.entries_by_segment), 0)
        self.assertEqual(runtime.get_profile()["dual_queue_stale_draft_metadata_count"], 1)

    def test_ground_truth_normalizes_decays_and_expires(self):
        runtime, caches = _runtime()
        runtime._round_id = 1
        runtime.observe_verify(
            _aggregated_meta(
                2,
                [1],
                [2.0],
                counts=[4],
                token_count=4,
                mode="verify",
            ),
            step_id=3,
        )
        entry = runtime.ground_truth_index.entries_by_segment[1][(2, 1)]
        self.assertAlmostEqual(entry.score_sum, 0.5)
        self.assertAlmostEqual(entry.activation_count, 1.0)

        runtime._round_id = 2
        priority = runtime.ground_truth_index.priority_for(2, 1, runtime._round_id)
        self.assertAlmostEqual(priority, 0.5 * (0.5 + 0.25))

        runtime._round_id = 4
        ranked = runtime.ground_truth_index.ranked_for_range(
            layer_start=2,
            layer_end=4,
            round_id=runtime._round_id,
            layer_caches=caches,
            inflight_keys=set(),
            source=runtime.GROUND_TRUTH_SOURCE,
        )
        self.assertEqual(ranked, [])
        self.assertNotIn((2, 1), runtime.ground_truth_index.entries_by_segment[1])

    def test_range_selection_uses_only_requested_segment(self):
        runtime, caches = _runtime()
        runtime._round_id = 1
        runtime.ground_truth_index.update(
            {
                **_aggregated_meta(0, [2], [10.0], counts=[1], mode="verify"),
                **_aggregated_meta(2, [3], [1.0], counts=[1], mode="verify"),
            },
            round_id=1,
        )
        ranked = runtime.ground_truth_index.ranked_for_range(
            layer_start=2,
            layer_end=4,
            round_id=1,
            layer_caches=caches,
            inflight_keys=set(),
            source=runtime.GROUND_TRUTH_SOURCE,
        )
        self.assertEqual([(x.layer_idx, x.expert_idx) for x in ranked], [(2, 3)])


class TestDualQueueVictims(unittest.TestCase):
    def test_draft_source_preserves_round_activated_experts(self):
        runtime, caches = _runtime(num_layers=1)
        cache = caches[0]
        cache.put_to_slot(0, 0, *_weights().values())
        cache.put_to_slot(1, 1, *_weights().values())
        runtime._round_protected[0].update({0, 1})

        self.assertIsNone(
            runtime._select_dual_queue_victim(
                cache,
                layer_idx=0,
                source=runtime.DRAFT_SOURCE,
            )
        )

    def test_ground_truth_source_ignores_draft_protection(self):
        runtime, caches = _runtime(num_layers=1)
        cache = caches[0]
        cache.put_to_slot(0, 0, *_weights().values())
        cache.put_to_slot(1, 1, *_weights().values())
        runtime._round_protected[0].update({0, 1})

        victim = runtime._select_dual_queue_victim(
            cache,
            layer_idx=0,
            source=runtime.GROUND_TRUTH_SOURCE,
        )
        self.assertIn(victim, {0, 1})
        self.assertEqual(runtime._round_protected[0], {0, 1})


class TestDualQueueBestEffort(unittest.TestCase):
    def test_ready_target_is_published(self):
        runtime, caches = _runtime(num_layers=1)
        runtime._round_id = 1
        runtime.draft_predict_index.update(
            _aggregated_meta(0, [2], [1.0]),
            round_id=1,
        )
        submitted = runtime.submit_segment_prefetch(
            step_id=1,
            target_layer_start=0,
            target_layer_end=1,
            target_segment_id=0,
            source=runtime.DRAFT_SOURCE,
            phase="draft",
        )
        self.assertEqual(submitted, 1)
        self.assertEqual(runtime.publish_segment_ready(step_id=1, segment_id=0), 1)
        self.assertTrue(caches[0].is_cached_cpu(2))

    def test_unready_target_is_discarded_without_cache_mutation(self):
        runtime, caches = _runtime(num_layers=1)
        cache = caches[0]
        cache.put_to_slot(0, 0, *_weights().values())
        runtime._round_id = 1
        runtime.draft_predict_index.update(
            _aggregated_meta(0, [2], [1.0]),
            round_id=1,
        )
        runtime.submit_segment_prefetch(
            step_id=1,
            target_layer_start=0,
            target_layer_end=1,
            target_segment_id=0,
            source=runtime.DRAFT_SOURCE,
            phase="draft",
        )
        ticket = runtime.inflight[(0, 2)]
        event = _ManualEvent(ready=False)
        ticket.ready_event = event

        self.assertEqual(runtime.publish_segment_ready(step_id=1, segment_id=0), 0)
        self.assertEqual(cache.slot_to_expert[0], 0)
        self.assertIn((0, 2), runtime._expired_tickets)

        event.ready = True
        runtime._reap_expired_tickets()
        self.assertNotIn((0, 2), runtime.inflight)
        self.assertFalse(cache.is_cached_cpu(2))
        self.assertEqual(cache.staging_slot_state, [0, 0])

    def test_same_boundary_does_not_evict_just_published_expert(self):
        runtime, caches = _runtime(
            num_layers=1,
            cache_slots=1,
            draft_prefetch_max_per_boundary=2,
        )
        cache = caches[0]
        cache.put_to_slot(0, 0, *_weights().values())
        runtime._round_id = 1
        runtime.draft_predict_index.update(
            _aggregated_meta(0, [2, 3], [2.0, 1.0]),
            round_id=1,
        )
        submitted = runtime.submit_segment_prefetch(
            step_id=1,
            target_layer_start=0,
            target_layer_end=1,
            target_segment_id=0,
            source=runtime.DRAFT_SOURCE,
            phase="draft",
        )

        self.assertEqual(submitted, 2)
        self.assertEqual(runtime.publish_segment_ready(step_id=1, segment_id=0), 1)
        self.assertTrue(cache.is_cached_cpu(2))
        self.assertFalse(cache.is_cached_cpu(3))

    def test_round_end_discards_all_remaining_tickets(self):
        runtime, _ = _runtime(num_layers=1)
        runtime._round_id = 1
        runtime.draft_predict_index.update(
            _aggregated_meta(0, [2], [1.0]),
            round_id=1,
        )
        runtime.submit_segment_prefetch(
            step_id=1,
            target_layer_start=0,
            target_layer_end=1,
            target_segment_id=0,
            source=runtime.DRAFT_SOURCE,
            phase="verify",
        )
        ticket = runtime.inflight[(0, 2)]
        ticket.segment_id = 1
        ticket.ready_event = _ManualEvent(ready=False)

        runtime.complete_verify_round(step_id=1)
        self.assertIn((0, 2), runtime._expired_tickets)
        self.assertEqual(len(runtime.draft_predict_index.entries_by_segment), 0)

    def test_verify_fallback_discards_without_publishing(self):
        runtime, caches = _runtime(num_layers=1)
        cache = caches[0]
        cache.put_to_slot(0, 0, *_weights().values())
        runtime._round_id = 1
        runtime._round_active = True
        runtime.draft_predict_index.update(
            _aggregated_meta(0, [2], [1.0]),
            round_id=1,
        )
        runtime.submit_segment_prefetch(
            step_id=1,
            target_layer_start=0,
            target_layer_end=1,
            target_segment_id=0,
            source=runtime.DRAFT_SOURCE,
            phase="verify",
        )

        runtime.discard_verify_round()

        self.assertFalse(cache.is_cached_cpu(2))
        self.assertEqual(len(runtime.draft_predict_index.entries_by_segment), 0)
        self.assertFalse(runtime._round_active)

    def test_budget_uses_conservative_minimum_segment_time(self):
        runtime, _ = _runtime(
            draft_prefetch_max_per_boundary=99,
            verify_prefetch_max_per_boundary=99,
            prefetch_max_inflight=8,
        )
        runtime._expert_transfer_ms = 2.0
        runtime._segment_compute_ms = {
            "draft": {0: 10.0, 1: 5.0},
            "verify": {0: 20.0, 1: 8.0},
        }
        draft_budget, verify_budget = runtime._recompute_prefetch_budgets()
        self.assertEqual(draft_budget, 2)
        self.assertEqual(verify_budget, 3)


class TestDualQueueMetadata(unittest.TestCase):
    def test_draft_metadata_is_fixed_score_sum_vector(self):
        cfg = _config(prefetch_metadata_host_buffer_pool_size=2)
        hf = SimpleNamespace(
            num_hidden_layers=4,
            num_experts_per_tok=2,
            num_experts=4,
        )
        recorder = ModelRuntimeMetaRecorder(cfg, hf)
        recorder.arm("draft", step_id=9, token_capacity=2, logical_token_count=2)
        recorder.record_layer(
            1,
            torch.tensor([[2, 3], [2, 1]], dtype=torch.int64),
            torch.tensor([[0.6, 0.4], [0.7, 0.3]], dtype=torch.float32),
        )
        handle = recorder.offload_async(None, layer_start_idx=1, layer_end_idx=2)
        metadata = recorder.collect(handle, wait=False)

        self.assertEqual(handle.metadata_format, "score_sum")
        self.assertEqual(set(recorder.device_buffers[("draft", 2)]), {
            "score_sum",
            "token_count",
            "token_count_capture_value",
            "token_positions",
        })
        meta = metadata[1]
        values = dict(zip(
            meta.aggregated_expert_ids.tolist(),
            meta.aggregated_score_sum.tolist(),
            strict=True,
        ))
        self.assertAlmostEqual(values[1], 0.3)
        self.assertAlmostEqual(values[2], 1.3)
        self.assertAlmostEqual(values[3], 0.4)
        self.assertIsNone(meta.aggregated_activation_count)


class TestDualQueueSchedule(unittest.TestCase):
    def test_first_draft_uses_ground_truth_then_draft_predict(self):
        runtime, _ = _runtime()
        runtime.begin_draft_iteration(step_id=1)
        boundaries = [(0, 2), (2, 4), (4, 6)]
        calls = []

        def record(**kwargs):
            calls.append((kwargs["target_segment_id"], kwargs["source"]))
            return 0

        with patch.object(runtime, "publish_segment_ready", return_value=0), patch.object(
            runtime, "submit_segment_prefetch", side_effect=record
        ):
            runtime.on_draft_segment_start(step_id=1, segment_id=0, boundaries=boundaries)
            runtime.on_draft_segment_end(step_id=1, segment_id=0, boundaries=boundaries)
            runtime.on_draft_segment_end(step_id=1, segment_id=1, boundaries=boundaries)
            runtime.on_draft_segment_end(step_id=1, segment_id=2, boundaries=boundaries)

        self.assertEqual(calls, [
            (1, runtime.GROUND_TRUTH_SOURCE),
            (2, runtime.GROUND_TRUTH_SOURCE),
            (0, runtime.DRAFT_SOURCE),
            (1, runtime.DRAFT_SOURCE),
        ])

    def test_later_draft_and_verify_prefetch_next_segment(self):
        runtime, _ = _runtime()
        runtime.begin_draft_iteration(step_id=1)
        runtime.begin_draft_iteration(step_id=2)
        boundaries = [(0, 2), (2, 4), (4, 6)]
        calls = []

        def record(**kwargs):
            calls.append((kwargs["phase"], kwargs["target_segment_id"], kwargs["source"]))
            return 0

        with patch.object(runtime, "publish_segment_ready", return_value=0), patch.object(
            runtime, "submit_segment_prefetch", side_effect=record
        ):
            runtime.on_draft_segment_start(step_id=2, segment_id=2, boundaries=boundaries)
            runtime.on_verify_segment_start(step_id=2, segment_id=2, boundaries=boundaries)

        self.assertEqual(calls, [
            ("draft", 0, runtime.DRAFT_SOURCE),
            ("verify", 0, runtime.DRAFT_SOURCE),
        ])


if __name__ == "__main__":
    unittest.main()
