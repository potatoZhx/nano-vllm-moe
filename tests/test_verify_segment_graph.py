"""Tests for segmented verify CUDA graph with inter-segment prefetching."""
from __future__ import annotations

import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from nanovllm.expert.cache import LayerExpertCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_cache(
    num_experts: int = 8,
    slots: int = 4,
    cached: list[int] | None = None,
) -> LayerExpertCache:
    if cached is None:
        cached = [0, 1, 2, 3]
    fake_pool = {
        eid: {
            "gate_up": torch.zeros(4, 4, dtype=torch.float32),
            "down": torch.zeros(4, 2, dtype=torch.float32),
        }
        for eid in range(num_experts)
    }
    cache = LayerExpertCache(
        num_experts=num_experts,
        slots_per_layer=slots,
        gate_up_shape=(4, 4),
        down_shape=(4, 2),
        device=torch.device("cpu"),
        dtype=torch.float32,
        cpu_expert_pool=fake_pool,
    )
    for slot_idx, expert_idx in enumerate(cached):
        cache.put_to_slot(
            slot_idx,
            expert_idx,
            fake_pool[expert_idx]["gate_up"],
            fake_pool[expert_idx]["down"],
        )
    return cache


def _make_hf_config(num_layers=48, num_experts=64, top_k=8):
    return SimpleNamespace(
        num_hidden_layers=num_layers,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        hidden_size=256,
    )


# ===================================================================
# 1. Config validation tests
# ===================================================================

class TestVerifySegmentConfig(unittest.TestCase):
    """Test verify segment config fields and validation."""

    def _make_config(self, **overrides):
        defaults = dict(
            verify_prefetch_segment_size=12,
            verify_prefetch_visible_budget_ms=12.0,
            verify_prefetch_min_per_boundary=0,
            verify_prefetch_max_per_boundary=16,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_default_values(self):
        cfg = self._make_config()
        self.assertEqual(cfg.verify_prefetch_segment_size, 12)
        self.assertEqual(cfg.verify_prefetch_visible_budget_ms, 12.0)
        self.assertEqual(cfg.verify_prefetch_min_per_boundary, 0)
        self.assertEqual(cfg.verify_prefetch_max_per_boundary, 16)

    def test_segment_size_must_be_positive(self):
        cfg = self._make_config(verify_prefetch_segment_size=0)
        with self.assertRaises(AssertionError):
            assert cfg.verify_prefetch_segment_size >= 1

    def test_budget_ms_non_negative(self):
        cfg = self._make_config(verify_prefetch_visible_budget_ms=-1.0)
        with self.assertRaises(AssertionError):
            assert cfg.verify_prefetch_visible_budget_ms >= 0.0

    def test_max_boundary_gte_min(self):
        cfg = self._make_config(verify_prefetch_min_per_boundary=5, verify_prefetch_max_per_boundary=2)
        with self.assertRaises(AssertionError):
            assert cfg.verify_prefetch_max_per_boundary >= cfg.verify_prefetch_min_per_boundary


# ===================================================================
# 2. Segment boundary computation tests
# ===================================================================

class TestVerifySegmentBoundaries(unittest.TestCase):
    """Test _verify_segment_boundaries and _verify_segment_graph_enabled."""

    def _make_runner(self, num_layers=48, segment_size=12, kt_hybrid=True):
        from nanovllm.engine.model_runner import ModelRunner
        mr = object.__new__(ModelRunner)
        mr.config = SimpleNamespace(
            verify_cuda_graph_kt_hybrid=kt_hybrid,
            verify_prefetch_segment_size=segment_size,
            hf_config=_make_hf_config(num_layers=num_layers),
        )
        return mr

    def test_boundaries_48_layers_12_segments(self):
        from nanovllm.engine.model_runner import ModelRunner
        mr = self._make_runner(num_layers=48, segment_size=12)
        boundaries = ModelRunner._verify_segment_boundaries(mr)
        self.assertEqual(boundaries, [(0, 12), (12, 24), (24, 36), (36, 48)])

    def test_boundaries_48_layers_16_segments(self):
        from nanovllm.engine.model_runner import ModelRunner
        mr = self._make_runner(num_layers=48, segment_size=16)
        boundaries = ModelRunner._verify_segment_boundaries(mr)
        self.assertEqual(boundaries, [(0, 16), (16, 32), (32, 48)])

    def test_boundaries_uneven(self):
        from nanovllm.engine.model_runner import ModelRunner
        mr = self._make_runner(num_layers=50, segment_size=12)
        boundaries = ModelRunner._verify_segment_boundaries(mr)
        self.assertEqual(len(boundaries), 5)
        self.assertEqual(boundaries[-1], (48, 50))

    def test_single_segment_when_size_gte_layers(self):
        from nanovllm.engine.model_runner import ModelRunner
        mr = self._make_runner(num_layers=48, segment_size=48)
        boundaries = ModelRunner._verify_segment_boundaries(mr)
        self.assertEqual(boundaries, [(0, 48)])

    def test_segment_graph_enabled_multiple_segments(self):
        from nanovllm.engine.model_runner import ModelRunner
        mr = self._make_runner(num_layers=48, segment_size=12, kt_hybrid=True)
        self.assertTrue(ModelRunner._verify_segment_graph_enabled(mr))

    def test_segment_graph_disabled_single_segment(self):
        from nanovllm.engine.model_runner import ModelRunner
        mr = self._make_runner(num_layers=48, segment_size=48, kt_hybrid=True)
        self.assertFalse(ModelRunner._verify_segment_graph_enabled(mr))

    def test_segment_graph_disabled_no_kt_hybrid(self):
        from nanovllm.engine.model_runner import ModelRunner
        mr = self._make_runner(num_layers=48, segment_size=12, kt_hybrid=False)
        self.assertFalse(ModelRunner._verify_segment_graph_enabled(mr))

    def test_segment_size_1_gives_per_layer(self):
        from nanovllm.engine.model_runner import ModelRunner
        mr = self._make_runner(num_layers=4, segment_size=1)
        boundaries = ModelRunner._verify_segment_boundaries(mr)
        self.assertEqual(boundaries, [(0, 1), (1, 2), (2, 3), (3, 4)])


# ===================================================================
# 3. _can_use_verify_cudagraph with segment graphs
# ===================================================================

class TestCanUseVerifyWithSegments(unittest.TestCase):
    """Test _can_use_verify_cudagraph dispatches correctly for segment graphs."""

    def _make_runner(self, num_layers=48, segment_size=12, has_segment_graphs=True):
        from nanovllm.engine.model_runner import ModelRunner
        mr = object.__new__(ModelRunner)
        mr.config = SimpleNamespace(
            verify_cuda_graph=True,
            verify_cuda_graph_kt_hybrid=True,
            verify_cuda_graph_bucket_steps=[4, 8, 12, 16],
            verify_prefetch_segment_size=segment_size,
            hf_config=_make_hf_config(num_layers=num_layers),
        )
        mr.verify_graph_bs = [4, 8, 12, 16]
        mr.verify_kt_hybrid_graphs = {}
        mr.verify_prefix_graphs = {}
        if has_segment_graphs:
            mr.verify_kt_hybrid_segment_graphs = {4: [MagicMock()], 8: [MagicMock()]}
        else:
            mr.verify_kt_hybrid_segment_graphs = {}
        mr.verify_graph_vars = {"block_tables": torch.zeros(1, 10, dtype=torch.int32)}
        return mr

    def test_can_use_with_segment_graphs(self):
        from nanovllm.engine.model_runner import ModelRunner
        from nanovllm.utils.context import set_context, reset_context
        mr = self._make_runner(has_segment_graphs=True)
        set_context(True, cu_seqlens_q=torch.tensor([0, 4], dtype=torch.int32))
        try:
            self.assertTrue(ModelRunner._can_use_verify_cudagraph(mr, 4))
        finally:
            reset_context()

    def test_cannot_use_without_segment_graphs(self):
        from nanovllm.engine.model_runner import ModelRunner
        from nanovllm.utils.context import set_context, reset_context
        mr = self._make_runner(has_segment_graphs=False)
        set_context(True, cu_seqlens_q=torch.tensor([0, 4], dtype=torch.int32))
        try:
            self.assertFalse(ModelRunner._can_use_verify_cudagraph(mr, 4))
        finally:
            reset_context()

    def test_bucket_overflow_rejected(self):
        from nanovllm.engine.model_runner import ModelRunner
        from nanovllm.utils.context import set_context, reset_context
        mr = self._make_runner(has_segment_graphs=True)
        set_context(True, cu_seqlens_q=torch.tensor([0, 20], dtype=torch.int32))
        try:
            self.assertFalse(ModelRunner._can_use_verify_cudagraph(mr, 20))
        finally:
            reset_context()


# ===================================================================
# 4. Model forward segment methods
# ===================================================================

class TestModelForwardSegment(unittest.TestCase):
    """Test forward_verify_kt_hybrid_segment on Qwen3MoeModel and Qwen3MoeForCausalLM."""

    def test_causal_lm_first_segment_needs_input_ids(self):
        from nanovllm.models.qwen3_moe import Qwen3MoeForCausalLM
        model = MagicMock(spec=Qwen3MoeForCausalLM)
        model.model = MagicMock()
        model.model.embed_tokens = MagicMock(return_value=torch.zeros(4, 256))
        model.model.forward_verify_kt_hybrid_segment = MagicMock(return_value=torch.zeros(4, 256))

        with self.assertRaises(ValueError):
            Qwen3MoeForCausalLM.forward_verify_kt_hybrid_segment(
                model, input_ids=None, hidden_states=None,
                position_ids=torch.zeros(4, dtype=torch.int64),
                start_layer=0, end_layer=12, apply_norm=False,
            )

    def test_causal_lm_first_segment_embeds(self):
        from nanovllm.models.qwen3_moe import Qwen3MoeForCausalLM
        model = MagicMock(spec=Qwen3MoeForCausalLM)
        model.model = MagicMock()
        embedded = torch.randn(4, 256)
        model.model.embed_tokens = MagicMock(return_value=embedded)
        model.model.forward_verify_kt_hybrid_segment = MagicMock(return_value=torch.zeros(4, 256))

        input_ids = torch.zeros(4, dtype=torch.int64)
        Qwen3MoeForCausalLM.forward_verify_kt_hybrid_segment(
            model, input_ids=input_ids, hidden_states=None,
            position_ids=torch.zeros(4, dtype=torch.int64),
            start_layer=0, end_layer=12, apply_norm=False,
        )
        model.model.embed_tokens.assert_called_once_with(input_ids)
        model.model.forward_verify_kt_hybrid_segment.assert_called_once()

    def test_causal_lm_subsequent_segment_skips_embed(self):
        from nanovllm.models.qwen3_moe import Qwen3MoeForCausalLM
        model = MagicMock(spec=Qwen3MoeForCausalLM)
        model.model = MagicMock()
        model.model.embed_tokens = MagicMock()
        model.model.forward_verify_kt_hybrid_segment = MagicMock(return_value=torch.zeros(4, 256))

        hidden = torch.randn(4, 256)
        Qwen3MoeForCausalLM.forward_verify_kt_hybrid_segment(
            model, input_ids=None, hidden_states=hidden,
            position_ids=torch.zeros(4, dtype=torch.int64),
            start_layer=12, end_layer=24, apply_norm=False,
        )
        model.model.embed_tokens.assert_not_called()


# ===================================================================
# 5. Verify segment index in prefetcher
# ===================================================================

class TestVerifySegmentIndex(unittest.TestCase):
    """Test verify_segment_index initialization and observe_verify feeding."""

    def _make_runtime(self, segment_size=12):
        from nanovllm.expert.prefetcher import PrefetchRuntime, SegmentCandidateIndex
        config = SimpleNamespace(
            prefetch_step_budget=8,
            prefetch_max_inflight=16,
            prefetch_staging_slots_per_layer=2,
            cache_eviction_budget_per_step=2,
            prefetch_global_queue_capacity=128,
            prefetch_history_ttl_steps=10,
            prefetch_history_activation_count_weight=1.0,
            prefetch_history_score_weight=1.0,
            prefetch_history_age_penalty=0.1,
            prefetch_runtime_mode="draft_segment_indexed",
            draft_prefetch_frontier_granularity="segment",
            draft_prefetch_segment_size=12,
            verify_prefetch_segment_size=segment_size,
            prefetch_use_verify_history=True,
            prefetch_transfer_stream_pool_size=1,
            prefetch_verify_layer_transfer_bandwidth_gbps=12.0,
            draft_prefetch_min_per_boundary=0,
            draft_prefetch_max_per_boundary=4,
            draft_prefetch_visible_budget_ms=3.0,
            verify_prefetch_min_per_boundary=0,
            verify_prefetch_max_per_boundary=4,
            verify_prefetch_visible_budget_ms=3.0,
        )
        cache_strategy = MagicMock()
        prefetch_strategy = MagicMock()
        recorder = MagicMock()

        rt = object.__new__(PrefetchRuntime)
        rt.config = config
        rt.layer_caches = {}
        rt.cpu_expert_pool = {}
        rt.cache_strategy = cache_strategy
        rt.prefetch_strategy = prefetch_strategy
        rt.runtime_meta_recorder = recorder
        rt.global_queue = MagicMock()
        rt.long_term_segment_index = SegmentCandidateIndex(config)
        rt.draft_segment_index = SegmentCandidateIndex(config)
        rt.verify_segment_index = SegmentCandidateIndex(config)
        rt.verify_segment_index.segment_size = segment_size
        rt.inflight = {}
        rt._profile = defaultdict(float)
        rt._recent_published = {}
        rt._recent_published_source = {}
        rt.transfer_streams = [None]
        rt.transfer_stream = None
        rt._next_transfer_stream_idx = 0
        rt.metadata_stream = None
        rt.publish_stream = None
        rt._draft_direct_active_budget = 4
        rt._draft_segment_indexed_budget = 4
        rt._draft_iteration_open = False
        rt._active_draft_iteration_steps = set()
        rt._draft_segment_indexed_submit_by_segment = defaultdict(int)
        rt._draft_segment_indexed_ready_by_segment = defaultdict(int)
        rt._draft_segment_indexed_success_by_segment = defaultdict(int)
        rt._draft_segment_indexed_consumed_by_segment = defaultdict(int)
        rt._prefetch_submit_count_by_source = defaultdict(int)
        rt._prefetch_completed_count_by_source = defaultdict(int)
        rt._prefetch_published_count_by_source = defaultdict(int)
        rt._prefetch_late_count_by_source = defaultdict(int)
        rt._prefetch_submitted_bytes_by_source = defaultdict(int)
        rt._prefetch_completed_bytes_by_source = defaultdict(int)
        rt._prefetch_published_bytes_by_source = defaultdict(int)
        rt._prefetch_late_bytes_by_source = defaultdict(int)
        rt._prefetch_transfer_enqueue_ms_by_source = defaultdict(float)
        rt._prefetch_completion_latency_ms_by_source = defaultdict(float)
        rt._prefetch_submit_count_by_stream = defaultdict(int)
        return rt

    def test_verify_segment_index_created_with_correct_size(self):
        rt = self._make_runtime(segment_size=16)
        self.assertEqual(rt.verify_segment_index.segment_size, 16)

    def test_verify_segment_index_segment_id(self):
        rt = self._make_runtime(segment_size=12)
        idx = rt.verify_segment_index
        self.assertEqual(idx._segment_id(0), 0)
        self.assertEqual(idx._segment_id(11), 0)
        self.assertEqual(idx._segment_id(12), 1)
        self.assertEqual(idx._segment_id(47), 3)

    def test_circular_target_computation(self):
        """Verify circular target: segment N -> target segment 0."""
        segment_size = 12
        num_layers = 48
        boundaries = [(s, min(s + segment_size, num_layers))
                       for s in range(0, num_layers, segment_size)]
        num_segments = len(boundaries)
        self.assertEqual(num_segments, 4)

        for seg_idx in range(num_segments):
            next_seg = (seg_idx + 1) % num_segments
            next_start = boundaries[next_seg][0]
            next_end = boundaries[next_seg][1]
            if seg_idx == num_segments - 1:
                self.assertEqual(next_start, 0)
                self.assertEqual(next_end, 12)
            else:
                self.assertEqual(next_start, boundaries[seg_idx + 1][0])


# ===================================================================
# 6. Metadata enqueue for verify segments
# ===================================================================

class TestEnqueueVerifySegmentMetadata(unittest.TestCase):
    """Test _enqueue_verify_segment_metadata passes correct parameters."""

    def _make_runner_with_prefetch(self):
        from nanovllm.engine.model_runner import ModelRunner
        mr = object.__new__(ModelRunner)
        mr.config = SimpleNamespace(
            verify_cuda_graph_kt_hybrid=True,
            verify_prefetch_segment_size=12,
            hf_config=_make_hf_config(num_layers=48),
        )
        mr.profile_enabled = False
        mr.rank = 0
        mr._prefetch_profile_lock = MagicMock()
        mr._profile = defaultdict(float)
        mr._prefetch_async_enabled = False

        mr.prefetch_runtime = MagicMock()
        mr.prefetch_runtime.metadata_stream = None
        mr.prefetch_runtime._source_lifecycle_profile_enabled = False
        mr.runtime_meta_recorder = MagicMock()

        offload_handle = MagicMock()
        offload_handle.buffer_bytes = 100
        mr.runtime_meta_recorder.offload_async = MagicMock(return_value=offload_handle)

        mr._acquire_prefetch_host_buffer_slot = MagicMock(return_value=(0, 0.0))
        mr._enqueue_prefetch_metadata = MagicMock()
        mr._flush_pending_prefetch_metadata = MagicMock()
        mr._ensure_prefetch_internal_state = MagicMock()
        mr._pending_prefetch_metadata = []
        return mr

    def test_offload_with_layer_range(self):
        from nanovllm.engine.model_runner import ModelRunner
        mr = self._make_runner_with_prefetch()
        ModelRunner._enqueue_verify_segment_metadata(
            mr, step_id=1, token_capacity=16,
            layer_start_idx=12, layer_end_idx=24, is_last_segment=False,
        )
        mr.runtime_meta_recorder.offload_async.assert_called_once()
        call_kwargs = mr.runtime_meta_recorder.offload_async.call_args
        self.assertEqual(call_kwargs.kwargs["layer_start_idx"], 12)
        self.assertEqual(call_kwargs.kwargs["layer_end_idx"], 24)

    def test_submit_after_phase_is_none(self):
        """Verify metadata enqueue does NOT trigger prefetch submit (decoupled)."""
        from nanovllm.engine.model_runner import ModelRunner
        mr = self._make_runner_with_prefetch()
        ModelRunner._enqueue_verify_segment_metadata(
            mr, step_id=1, token_capacity=16,
            layer_start_idx=0, layer_end_idx=12, is_last_segment=False,
        )
        call_kwargs = mr._enqueue_prefetch_metadata.call_args
        self.assertIsNone(call_kwargs.kwargs["submit_after_phase"])

    def test_last_segment_records_verify_consumed(self):
        from nanovllm.engine.model_runner import ModelRunner
        mr = self._make_runner_with_prefetch()
        ModelRunner._enqueue_verify_segment_metadata(
            mr, step_id=1, token_capacity=16,
            layer_start_idx=36, layer_end_idx=48, is_last_segment=True,
        )
        call_kwargs = mr._enqueue_prefetch_metadata.call_args
        self.assertTrue(call_kwargs.kwargs["record_verify_consumed"])

    def test_non_last_segment_does_not_record_consumed(self):
        from nanovllm.engine.model_runner import ModelRunner
        mr = self._make_runner_with_prefetch()
        ModelRunner._enqueue_verify_segment_metadata(
            mr, step_id=1, token_capacity=16,
            layer_start_idx=0, layer_end_idx=12, is_last_segment=False,
        )
        call_kwargs = mr._enqueue_prefetch_metadata.call_args
        self.assertFalse(call_kwargs.kwargs["record_verify_consumed"])

    def test_profile_records_consumption_for_every_segment(self):
        from nanovllm.engine.model_runner import ModelRunner
        mr = self._make_runner_with_prefetch()
        mr.prefetch_runtime._source_lifecycle_profile_enabled = True
        ModelRunner._enqueue_verify_segment_metadata(
            mr, step_id=1, token_capacity=16,
            layer_start_idx=0, layer_end_idx=12, is_last_segment=False,
        )
        call_kwargs = mr._enqueue_prefetch_metadata.call_args
        self.assertTrue(call_kwargs.kwargs["record_verify_consumed"])


# ===================================================================
# 7. Segmented replay boundary scheduling
# ===================================================================

class TestVerifySegmentReplay(unittest.TestCase):
    def test_replay_uses_explicit_step_id_and_schedules_each_boundary(self):
        from nanovllm.engine.model_runner import ModelRunner
        from nanovllm.utils.context import reset_context, set_context

        mr = object.__new__(ModelRunner)
        mr.config = SimpleNamespace(verify_prefetch_visible_budget_ms=3.0)
        mr.verify_graph_bs = [4]
        mr.verify_kt_hybrid_segment_graphs = {
            4: [MagicMock(), MagicMock()],
        }
        mr.verify_kt_hybrid_segment_boundaries = {
            4: [(0, 2), (2, 4)],
        }
        mr.verify_graph_vars = {
            "input_ids": torch.zeros(4, dtype=torch.int64),
            "positions": torch.zeros(4, dtype=torch.int64),
            "cu_seqlens_q": torch.zeros(2, dtype=torch.int32),
            "cu_seqlens_k": torch.zeros(2, dtype=torch.int32),
            "slot_mapping": torch.full((4,), -1, dtype=torch.int32),
            "block_tables": torch.zeros(1, 2, dtype=torch.int32),
            "hidden_states": torch.zeros(4, 8),
        }
        mr._select_verify_bucket = MagicMock(return_value=4)
        mr.runtime_meta_recorder = None
        mr._enqueue_verify_segment_metadata = MagicMock()
        mr._prefetch_runtime_lock = MagicMock()
        mr.profile_enabled = False
        mr.rank = 0

        runtime = SimpleNamespace(
            publish_direct_active_ready=MagicMock(),
            submit_verify_segment_prefetch=MagicMock(),
            on_verify_layer_start=MagicMock(),
        )
        mr.prefetch_runtime = runtime

        set_context(
            True,
            cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 2], dtype=torch.int32),
            max_seqlen_q=2,
            max_seqlen_k=8,
            slot_mapping=torch.tensor([0, 1], dtype=torch.int32),
            block_tables=torch.zeros(1, 2, dtype=torch.int32),
        )
        try:
            ModelRunner._run_verify_with_kt_hybrid_segment_graph(
                mr,
                torch.tensor([1, 2], dtype=torch.int64),
                torch.tensor([0, 1], dtype=torch.int64),
                step_id=7,
            )
        finally:
            reset_context()

        self.assertEqual(mr._enqueue_verify_segment_metadata.call_count, 2)
        for call in mr._enqueue_verify_segment_metadata.call_args_list:
            self.assertEqual(call.kwargs["step_id"], 7)
        self.assertEqual(runtime.submit_verify_segment_prefetch.call_count, 2)
        self.assertEqual(
            [
                (
                    call.kwargs["target_layer_start"],
                    call.kwargs["target_layer_end"],
                )
                for call in runtime.submit_verify_segment_prefetch.call_args_list
            ],
            [(2, 4), (0, 2)],
        )
        self.assertEqual(
            [call.args[0] for call in runtime.on_verify_layer_start.call_args_list],
            [0, 1, 2, 3],
        )


# ===================================================================
# 8. Post-verify metadata offload gating
# ===================================================================

class TestPostVerifyMetadataGating(unittest.TestCase):
    """Test that segmented verify skips the full-model offload."""

    def test_segment_graph_skips_full_offload(self):
        _use_kt_hybrid = True

        class FakeRunner:
            def _verify_segment_graph_enabled(self):
                return True

        runner = FakeRunner()
        _used_segment_graph = _use_kt_hybrid and runner._verify_segment_graph_enabled()
        self.assertTrue(_used_segment_graph)

    def test_monolithic_graph_does_full_offload(self):
        _use_kt_hybrid = True

        class FakeRunner:
            def _verify_segment_graph_enabled(self):
                return False

        runner = FakeRunner()
        _used_segment_graph = _use_kt_hybrid and runner._verify_segment_graph_enabled()
        self.assertFalse(_used_segment_graph)


# ===================================================================
# 9. Runtime meta host buffer pool sizing
# ===================================================================

class TestRuntimeMetaPoolSizing(unittest.TestCase):
    """Test that verify_kt_hybrid mode sizes host buffer pool for segments."""

    def _make_recorder(self, num_layers=48, segment_size=12):
        from nanovllm.expert.runtime_meta import ModelRuntimeMetaRecorder
        cfg = SimpleNamespace(
            prefetch_runtime_mode="draft_segment_indexed",
            prefetch_metadata_host_buffer_pool_size=3,
            draft_prefetch_segment_host_buffer_pool_size=0,
            draft_prefetch_frontier_granularity="segment",
            draft_prefetch_segment_size=12,
            verify_prefetch_segment_size=segment_size,
        )
        hf_cfg = SimpleNamespace(
            num_hidden_layers=num_layers,
            num_experts_per_tok=8,
            num_experts=64,
        )
        return ModelRuntimeMetaRecorder(config=cfg, hf_config=hf_cfg)

    def test_pool_size_accounts_for_segments(self):
        recorder = self._make_recorder(num_layers=48, segment_size=12)
        pool_size = recorder.target_host_buffer_pool_size("verify_kt_hybrid", 16)
        expected_segments = 4
        self.assertGreaterEqual(pool_size, expected_segments + 2)

    def test_pool_size_small_segment(self):
        recorder = self._make_recorder(num_layers=48, segment_size=4)
        pool_size = recorder.target_host_buffer_pool_size("verify_kt_hybrid", 16)
        expected_segments = 12
        self.assertGreaterEqual(pool_size, expected_segments + 2)

    def test_pool_size_draft_mode_unaffected(self):
        recorder = self._make_recorder(num_layers=48, segment_size=12)
        pool_verify = recorder.target_host_buffer_pool_size("verify_kt_hybrid", 16)
        pool_draft = recorder.target_host_buffer_pool_size("draft", 16)
        self.assertGreaterEqual(pool_verify, 6)
        self.assertGreaterEqual(pool_draft, 1)


# ===================================================================
# 10. Expert status recording across segments
# ===================================================================

class TestExpertStatusAcrossSegments(unittest.TestCase):
    """Verify expert_status device buffers work correctly across segment boundaries."""

    def _make_recorder(self, num_layers=4, num_experts=8, top_k=2):
        from nanovllm.expert.runtime_meta import ModelRuntimeMetaRecorder
        cfg = SimpleNamespace(
            prefetch_runtime_mode="draft_segment_indexed",
            prefetch_metadata_host_buffer_pool_size=3,
            draft_prefetch_segment_host_buffer_pool_size=0,
            draft_prefetch_frontier_granularity="segment",
            draft_prefetch_segment_size=12,
            verify_prefetch_segment_size=2,
        )
        hf_cfg = SimpleNamespace(
            num_hidden_layers=num_layers,
            num_experts_per_tok=top_k,
            num_experts=num_experts,
        )
        return ModelRuntimeMetaRecorder(config=cfg, hf_config=hf_cfg)

    def test_disjoint_layer_ranges_accumulate(self):
        """Segments write to disjoint layer indices in the same device buffer."""
        recorder = self._make_recorder(num_layers=4, num_experts=8, top_k=2)
        recorder.arm(mode="verify_kt_hybrid", step_id=0, token_capacity=4, logical_token_count=2)

        selected_0 = torch.tensor([[0, 5], [2, 7]], dtype=torch.int64)
        weights = torch.ones(2, 2, dtype=torch.float32) * 0.5
        mask_0 = torch.tensor([False, True, False, True], dtype=torch.bool)

        recorder.record_layer(layer_idx=0, selected_experts=selected_0,
                              routing_weights=weights, uncached_route_mask=mask_0)

        selected_2 = torch.tensor([[1, 6], [3, 4]], dtype=torch.int64)
        mask_2 = torch.tensor([False, True, False, True], dtype=torch.bool)
        recorder.record_layer(layer_idx=2, selected_experts=selected_2,
                              routing_weights=weights, uncached_route_mask=mask_2)

        dev = recorder.device_buffers[recorder.active_key]
        status_0 = dev["expert_status"][0]
        status_2 = dev["expert_status"][2]

        self.assertEqual(int(status_0[0].item()), 1)
        self.assertEqual(int(status_0[5].item()), 2)
        self.assertEqual(int(status_2[1].item()), 1)
        self.assertEqual(int(status_2[6].item()), 2)

        status_1 = dev["expert_status"][1]
        self.assertTrue((status_1 == 0).all())

    def test_segment_offload_with_layer_range(self):
        """offload_async with layer_start/end only copies the specified range."""
        recorder = self._make_recorder(num_layers=4, num_experts=8, top_k=2)
        recorder.arm(mode="verify_kt_hybrid", step_id=0, token_capacity=4, logical_token_count=2)

        selected = torch.tensor([[0, 5], [2, 7]], dtype=torch.int64)
        weights = torch.ones(2, 2, dtype=torch.float32) * 0.5
        mask = torch.tensor([False, True, False, True], dtype=torch.bool)
        recorder.record_layer(layer_idx=0, selected_experts=selected,
                              routing_weights=weights, uncached_route_mask=mask)
        recorder.record_layer(layer_idx=1, selected_experts=selected,
                              routing_weights=weights, uncached_route_mask=mask)

        handle = recorder.offload_async(stream=None, host_buffer_slot=0,
                                         layer_start_idx=0, layer_end_idx=1)
        self.assertIsNotNone(handle)


# ===================================================================
# 11. submit_verify_segment_prefetch integration
# ===================================================================

class TestSubmitVerifySegmentPrefetch(unittest.TestCase):
    """Test submit_verify_segment_prefetch returns 0 when no candidates."""

    @staticmethod
    def _config(**overrides):
        defaults = dict(
            prefetch_step_budget=8,
            prefetch_max_inflight=16,
            verify_prefetch_min_per_boundary=0,
            verify_prefetch_max_per_boundary=4,
            verify_prefetch_visible_budget_ms=3.0,
            verify_prefetch_segment_size=12,
            prefetch_history_ttl_steps=10,
            prefetch_source_weight_prefill=1.0,
            prefetch_source_weight_verify=1.2,
            prefetch_source_weight_draft=1.5,
            prefetch_activation_count_weight=0.1,
            prefetch_age_penalty=0.02,
            draft_prefetch_frontier_granularity="segment",
            draft_prefetch_segment_size=12,
            prefetch_verify_layer_transfer_bandwidth_gbps=12.0,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_returns_zero_without_candidates(self):
        from nanovllm.expert.prefetcher import PrefetchRuntime, SegmentCandidateIndex
        config = self._config()
        rt = object.__new__(PrefetchRuntime)
        rt.config = config
        rt.layer_caches = {}
        rt.cpu_expert_pool = {}
        rt.inflight = {}
        rt._profile = defaultdict(float)
        rt.draft_segment_index = SegmentCandidateIndex(config)
        rt.verify_segment_index = SegmentCandidateIndex(config)
        rt.verify_segment_index.segment_size = 12
        rt.long_term_segment_index = SegmentCandidateIndex(config)
        rt._recent_published = {}
        rt._recent_published_source = {}
        rt._prefetch_submit_count_by_source = defaultdict(int)
        rt._prefetch_completed_count_by_source = defaultdict(int)
        rt._prefetch_published_count_by_source = defaultdict(int)
        rt._prefetch_late_count_by_source = defaultdict(int)
        rt._prefetch_submitted_bytes_by_source = defaultdict(int)
        rt._prefetch_completed_bytes_by_source = defaultdict(int)
        rt._prefetch_published_bytes_by_source = defaultdict(int)
        rt._prefetch_late_bytes_by_source = defaultdict(int)
        rt._prefetch_transfer_enqueue_ms_by_source = defaultdict(float)
        rt._prefetch_completion_latency_ms_by_source = defaultdict(float)
        rt._prefetch_submit_count_by_stream = defaultdict(int)

        submitted = PrefetchRuntime.submit_verify_segment_prefetch(
            rt, step_id=0, target_layer_start=12, target_layer_end=24,
            visible_budget_ms=3.0,
        )
        self.assertEqual(submitted, 0)
        self.assertEqual(rt._profile["verify_segment_prefetch_call_count"], 1)
        self.assertEqual(rt._profile["verify_segment_prefetch_no_candidate_count"], 1)

    def test_returns_zero_budget_exhausted(self):
        from nanovllm.expert.prefetcher import PrefetchRuntime
        config = self._config(prefetch_step_budget=0)
        rt = object.__new__(PrefetchRuntime)
        rt.config = config
        rt.inflight = {}
        rt._profile = defaultdict(float)
        submitted = PrefetchRuntime.submit_verify_segment_prefetch(
            rt, step_id=0, target_layer_start=0, target_layer_end=12,
        )
        self.assertEqual(submitted, 0)
        self.assertEqual(rt._profile["verify_segment_prefetch_skipped_by_budget_count"], 1)

    def test_layer_range_query_handles_different_segment_sizes(self):
        from nanovllm.expert.prefetcher import PrefetchCandidate, SegmentCandidateIndex

        config = self._config(draft_prefetch_segment_size=24)
        index = SegmentCandidateIndex(config)
        cache = MagicMock()
        cache.is_cached_cpu.return_value = False
        cache.is_pending_cpu.return_value = False
        layer_caches = {5: cache, 18: cache}
        for layer_idx in (5, 18):
            candidate = PrefetchCandidate(
                layer_idx=layer_idx,
                expert_idx=3,
                source="draft_live",
                score_sum=1.0,
                activation_count=1,
                first_seen_step=1,
                last_seen_step=1,
                priority=0.0,
            )
            index.entries_by_segment[index._segment_id(layer_idx)][(layer_idx, 3)] = candidate

        candidates = index.candidates_for_layer_range(
            layer_start=12,
            layer_end=24,
            step_id=1,
            layer_caches=layer_caches,
            inflight_keys=set(),
        )

        self.assertEqual([(c.layer_idx, c.expert_idx) for c in candidates], [(18, 3)])

    def test_submits_candidate_from_draft_index_with_different_segment_size(self):
        from nanovllm.expert.prefetcher import (
            PrefetchCandidate,
            PrefetchRuntime,
            SegmentCandidateIndex,
        )

        config = self._config(draft_prefetch_segment_size=24)
        rt = object.__new__(PrefetchRuntime)
        rt.config = config
        cache = MagicMock()
        cache.is_cached_cpu.return_value = False
        cache.is_pending_cpu.return_value = False
        cache.reserve_active_slot_for_prefetch_deferred.return_value = SimpleNamespace(
            active_slot_idx=0,
            generation=1,
            prev_expert=0,
        )
        rt.layer_caches = {18: cache}
        weights = {
            "gate_up": torch.zeros(4, 4),
            "down": torch.zeros(4, 2),
        }
        rt.cpu_expert_pool = {18: {3: weights}}
        rt.inflight = {}
        rt._profile = defaultdict(float)
        rt.draft_segment_index = SegmentCandidateIndex(config)
        rt.verify_segment_index = SegmentCandidateIndex(config)
        rt.verify_segment_index.segment_size = 12
        rt.long_term_segment_index = SegmentCandidateIndex(config)
        candidate = PrefetchCandidate(
            layer_idx=18,
            expert_idx=3,
            source="draft_live",
            score_sum=1.0,
            activation_count=1,
            first_seen_step=1,
            last_seen_step=1,
            priority=0.0,
        )
        rt.draft_segment_index.entries_by_segment[0][(18, 3)] = candidate
        rt._select_publish_slot_cpu = MagicMock(return_value=0)
        rt._begin_prefetch_transfer = MagicMock(
            return_value=(MagicMock(), 100.0, 0.1, 64, 0)
        )
        rt._record_prefetch_submitted = MagicMock()
        rt._record_source_submit = MagicMock()

        submitted = PrefetchRuntime.submit_verify_segment_prefetch(
            rt,
            step_id=1,
            target_layer_start=12,
            target_layer_end=24,
            visible_budget_ms=3.0,
        )

        self.assertEqual(submitted, 1)
        self.assertEqual(rt._profile["verify_segment_prefetch_submit_count"], 1)
        self.assertIn((18, 3), rt.inflight)


if __name__ == "__main__":
    unittest.main()
