import unittest
from types import SimpleNamespace

import torch

from nanovllm.expert.cache import LayerExpertCache
from nanovllm.expert.prefetcher import PredictivePrefetchRuntime
from nanovllm.expert.runtime_meta import LayerRuntimeMetaCPU
from nanovllm.scheduling.cache_strategy import LFURankGuardStrategy, create_cache_strategy
from nanovllm.scheduling.prefetch_strategy import create_prefetch_strategy


def _config(**overrides):
    cfg = SimpleNamespace(
        prefetch_step_budget=4,
        prefetch_max_inflight=8,
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
        prefetch_runtime_mode="draft_segment_indexed",
        prefetch_runtime_kind="predictive",
        draft_prefetch_frontier_granularity="segment",
        draft_prefetch_segment_size=1,
        draft_prefetch_visible_budget_ms=3.0,
        draft_prefetch_min_per_boundary=0,
        draft_prefetch_max_per_boundary=4,
        cache_strategy="lfu",
        prefetch_verify_attention_ratio=1.0,
        predictive_phase1_budget=4,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _weights():
    return {"gate_up": torch.ones(2, 2), "down": torch.ones(2, 2)}


def _meta(layer_idx, experts, step_id, mode="draft", weights=None):
    sel = torch.tensor([experts], dtype=torch.int64)
    if weights is None:
        weights = [1.0 / len(experts)] * len(experts)
    w = torch.tensor([weights], dtype=torch.float32)
    return {
        layer_idx: LayerRuntimeMetaCPU(
            step_id=step_id,
            mode=mode,
            layer_idx=layer_idx,
            token_count=1,
            selected_experts=sel,
            routing_weights=w,
        )
    }


def _make_cache(num_experts=4, slots=2, cpu_pool=None):
    return LayerExpertCache(
        num_experts=num_experts,
        slots_per_layer=slots,
        gate_up_shape=(2, 2),
        down_shape=(2, 2),
        device=torch.device("cpu"),
        dtype=torch.float32,
        cpu_expert_pool=cpu_pool or {},
        staging_slots_per_layer=0,
        enable_prefetch=False,
    )


def _build_runtime(caches, cpu_pool, cfg=None, cache_strategy=None):
    cfg = cfg or _config()
    return PredictivePrefetchRuntime(
        config=cfg,
        layer_caches=caches,
        cpu_expert_pool=cpu_pool,
        cache_strategy=cache_strategy or create_cache_strategy(cfg.cache_strategy),
        prefetch_strategy=create_prefetch_strategy("noop", cfg),
        runtime_meta_recorder=SimpleNamespace(),
    )


class TestPredictiveDataSeparation(unittest.TestCase):
    def _single_layer(self):
        pool = {0: {2: _weights()}}
        cache = _make_cache(cpu_pool=pool[0])
        cache.put_to_slot(0, 0, _weights()["gate_up"], _weights()["down"])
        rt = _build_runtime({0: cache}, pool)
        return rt, cache

    def _queue_size(self, rt):
        return sum(len(v) for v in rt.draft_segment_index.entries_by_segment.values())

    def test_observe_prefill_marks_access_not_queue(self):
        rt, cache = self._single_layer()
        rt.observe_prefill(_meta(0, [2, 2], step_id=1, mode="prefill"), step_id=1)
        self.assertEqual(cache.access_count[2], 1)
        self.assertEqual(self._queue_size(rt), 0)
        self.assertEqual(len(rt.global_queue.entries), 0)

    def test_observe_verify_marks_access_not_queue(self):
        rt, cache = self._single_layer()
        rt.observe_verify(_meta(0, [2, 2], step_id=1, mode="verify"), step_id=1)
        self.assertEqual(cache.access_count[2], 1)
        self.assertEqual(self._queue_size(rt), 0)
        self.assertEqual(len(rt.global_queue.entries), 0)

    def test_observe_draft_updates_queue_not_access(self):
        rt, cache = self._single_layer()
        rt.begin_draft_iteration(step_id=3)
        rt.observe_draft(_meta(0, [2], step_id=3), step_id=3)
        self.assertEqual(cache.access_count[2], 0)  # no ground-truth pollution
        self.assertEqual(self._queue_size(rt), 1)

    def test_observe_draft_stale_step_dropped(self):
        rt, _cache = self._single_layer()
        # step 99 never armed via begin_draft_iteration -> stale, dropped.
        rt.observe_draft(_meta(0, [2], step_id=99), step_id=99)
        self.assertEqual(self._queue_size(rt), 0)
        prof = rt.get_profile(reset=False)
        self.assertEqual(prof["predictive_draft_stale_observe_count"], 1)


class TestPredictiveRoundProtection(unittest.TestCase):
    def _two_slot_cache(self, freq0, freq1):
        cache = _make_cache(slots=2)
        cache.put_to_slot(0, 0, _weights()["gate_up"], _weights()["down"])
        cache.put_to_slot(1, 1, _weights()["gate_up"], _weights()["down"])
        cache.access_count[0] = freq0
        cache.access_count[1] = freq1
        return cache

    def test_round_protection_skips_loaded_expert(self):
        cache = self._two_slot_cache(freq0=1, freq1=5)  # expert0 is LFU victim
        rt = _build_runtime({0: cache}, {0: {}})
        rt._round_loaded[0].add(0)  # expert0 protected this round
        # Incoming expert 3; expert0 (lowest freq) must be skipped -> slot1.
        self.assertEqual(rt._select_protected_victim(cache, 0, 3), 1)
        # Incoming recorded for protection.
        self.assertIn(3, rt._round_loaded[0])

    def test_rankguard_protection_composed(self):
        cache = self._two_slot_cache(freq0=1, freq1=5)
        strategy = LFURankGuardStrategy(num_experts=4, protect_threshold=0.15)
        strategy.set_rank_scores(0, [1.0, 0.0, 0.0, 0.0])  # expert0 protected
        rt = _build_runtime(
            {0: cache}, {0: {}}, cfg=_config(cache_strategy="lfu_rankguard"),
            cache_strategy=strategy,
        )
        # expert0 lowest freq but rankguard-protected -> victim is slot1.
        self.assertEqual(rt._select_protected_victim(cache, 0, 3), 1)

    def test_safety_valve_returns_lowest_when_all_protected(self):
        cache = self._two_slot_cache(freq0=1, freq1=5)
        rt = _build_runtime({0: cache}, {0: {}})
        rt._round_loaded[0].update({0, 1})  # both protected
        # Safety valve ignores protection and returns lowest-freq slot (slot0).
        self.assertEqual(rt._select_protected_victim(cache, 0, 3), 0)

    def test_empty_slot_preferred_no_eviction(self):
        cache = _make_cache(slots=2)
        cache.put_to_slot(0, 0, _weights()["gate_up"], _weights()["down"])  # slot1 empty
        rt = _build_runtime({0: cache}, {0: {}})
        self.assertEqual(rt._select_protected_victim(cache, 0, 3), 1)

    def test_on_verify_layer_start_releases_layer(self):
        rt = _build_runtime({0: _make_cache()}, {0: {}})
        rt._round_loaded[2].add(5)
        rt.on_verify_layer_start(2)
        self.assertNotIn(2, rt._round_loaded)


class TestPredictiveLifecycle(unittest.TestCase):
    def _rt(self):
        pool = {0: {2: _weights()}}
        cache = _make_cache(cpu_pool=pool[0])
        cache.put_to_slot(0, 0, _weights()["gate_up"], _weights()["down"])
        return _build_runtime({0: cache}, pool)

    def _queue_size(self, rt):
        return sum(len(v) for v in rt.draft_segment_index.entries_by_segment.values())

    def test_queue_persists_through_verify_cleared_next_round(self):
        rt = self._rt()
        rt.begin_draft_iteration(step_id=1)
        rt.observe_draft(_meta(0, [2], step_id=1), step_id=1)
        self.assertEqual(self._queue_size(rt), 1)
        rt.end_draft_iteration()  # before verify: queue must persist
        self.assertEqual(self._queue_size(rt), 1)
        self.assertFalse(rt._draft_iteration_open)
        rt.begin_draft_iteration(step_id=2)  # next round clears it
        self.assertEqual(self._queue_size(rt), 0)

    def test_round_loaded_reset_next_round(self):
        rt = self._rt()
        rt.begin_draft_iteration(step_id=1)
        rt._round_loaded[0].add(2)
        rt.end_draft_iteration()
        self.assertIn(2, rt._round_loaded[0])  # persists through verify
        rt.begin_draft_iteration(step_id=2)
        self.assertEqual(len(rt._round_loaded), 0)  # cleared next round


class TestPredictivePhase1(unittest.TestCase):
    def _two_layer_runtime(self, budget=4, freqs=None):
        # segment_size=1 => layer0 is segment 0, layer1 is segment n-1.
        freqs = freqs or {2: 5}
        pool = {0: {2: _weights()}, 1: {e: _weights() for e in freqs}}
        caches = {}
        for li in (0, 1):
            cache = _make_cache(slots=2, cpu_pool=pool[li])
            cache.put_to_slot(0, 0, _weights()["gate_up"], _weights()["down"])
            caches[li] = cache
        for eid, f in freqs.items():
            caches[1].access_count[eid] = f
        rt = _build_runtime(caches, pool, cfg=_config(predictive_phase1_budget=budget))
        return rt, caches

    def test_phase1_armed_in_begin_submitted_after_drain_once(self):
        rt, caches = self._two_layer_runtime()
        rt.begin_draft_iteration(step_id=1)
        self.assertTrue(rt._phase1_pending)
        self.assertEqual(len(rt.inflight), 0)  # begin must NOT submit (before drain)

        submitted = rt.maybe_submit_phase1(step_id=1)
        self.assertEqual(submitted, 1)
        self.assertFalse(rt._phase1_pending)
        self.assertIn((1, 2), rt.inflight)
        self.assertEqual(rt.inflight[(1, 2)].source, "predictive_phase1")
        # only the last segment (layer 1) is targeted
        self.assertNotIn((0, 2), rt.inflight)

        # idempotent within a round
        self.assertEqual(rt.maybe_submit_phase1(step_id=1), 0)

        published = rt.publish_direct_active_ready(step_id=1)
        self.assertEqual(published, 1)
        self.assertTrue(caches[1].is_cached_cpu(2))  # non-deferred -> commit_active

    def test_phase1_picks_highest_freq_uncached(self):
        rt, _caches = self._two_layer_runtime(budget=1, freqs={2: 5, 3: 1})
        rt.begin_draft_iteration(step_id=1)
        rt.maybe_submit_phase1(step_id=1)
        self.assertIn((1, 2), rt.inflight)   # highest freq
        self.assertNotIn((1, 3), rt.inflight)  # lower freq excluded by budget=1


class TestPredictiveVerifyPrefetch(unittest.TestCase):
    def _rt(self, attention_ratio=1.0):
        pool = {0: {2: _weights()}}
        cache = _make_cache(slots=2, cpu_pool=pool[0])
        cache.put_to_slot(0, 0, _weights()["gate_up"], _weights()["down"])
        cfg = _config(prefetch_verify_attention_ratio=attention_ratio)
        rt = _build_runtime({0: cache}, pool, cfg=cfg)
        rt.begin_draft_iteration(step_id=1)
        rt.observe_draft(_meta(0, [2], step_id=1), step_id=1)
        return rt, cache

    def test_verify_prefetch_from_draft_queue(self):
        rt, cache = self._rt()
        submitted = rt.submit_verify_layer_prefetch(
            step_id=2, target_layer_idx=0, available_ms=10.0,
        )
        self.assertEqual(submitted, 1)
        self.assertIn((0, 2), rt.inflight)
        self.assertEqual(rt.inflight[(0, 2)].source, "verify_layer_predict")
        published = rt.publish_direct_active_ready(step_id=2)
        self.assertEqual(published, 1)
        self.assertTrue(cache.is_cached_cpu(2))

    def test_verify_prefetch_attention_ratio_zero_skips(self):
        rt, _cache = self._rt(attention_ratio=0.0)
        submitted = rt.submit_verify_layer_prefetch(
            step_id=2, target_layer_idx=0, available_ms=10.0,
        )
        self.assertEqual(submitted, 0)
        self.assertEqual(len(rt.inflight), 0)


if __name__ == "__main__":
    unittest.main()
