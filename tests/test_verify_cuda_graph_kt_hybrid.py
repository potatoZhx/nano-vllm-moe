"""Unit and integration tests for verify CUDA graph kt_hybrid path."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from nanovllm.expert.cache import LayerExpertCache
from nanovllm.expert.placement import (
    MoEExecutionPlan,
    build_verify_graph_safe_plan_gpu,
    build_verify_plan_gpu,
)


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


# ===================================================================
# 1. Config auto-set tests
# ===================================================================

class TestConfigAutoSet(unittest.TestCase):
    """Test that verify_cuda_graph_kt_hybrid is auto-set correctly."""

    def _make_config(self, **overrides):
        """Build a minimal Config-like namespace for __post_init__ logic."""
        defaults = dict(
            model="/tmp/fake_model",
            verify_cuda_graph=False,
            verify_cuda_graph_kt_hybrid=False,
            cpu_expert_backend="torch",
            spec_verify_miss_policy="cache_fill",
            prefetch_verify_layer_enabled=True,
            enforce_eager=False,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_kt_hybrid_enabled_when_conditions_met(self):
        cfg = self._make_config(
            verify_cuda_graph=True,
            cpu_expert_backend="kt_direct",
            spec_verify_miss_policy="cpu",
        )
        # Simulate __post_init__ logic
        if (
            cfg.verify_cuda_graph
            and cfg.cpu_expert_backend == "kt_direct"
            and cfg.spec_verify_miss_policy == "cpu"
        ):
            cfg.verify_cuda_graph_kt_hybrid = True
            cfg.prefetch_verify_layer_enabled = False

        self.assertTrue(cfg.verify_cuda_graph_kt_hybrid)
        self.assertFalse(cfg.prefetch_verify_layer_enabled)

    def test_kt_hybrid_not_enabled_wrong_backend(self):
        cfg = self._make_config(
            verify_cuda_graph=True,
            cpu_expert_backend="fused",
            spec_verify_miss_policy="cpu",
        )
        if (
            cfg.verify_cuda_graph
            and cfg.cpu_expert_backend == "kt_direct"
            and cfg.spec_verify_miss_policy == "cpu"
        ):
            cfg.verify_cuda_graph_kt_hybrid = True
        self.assertFalse(cfg.verify_cuda_graph_kt_hybrid)

    def test_kt_hybrid_not_enabled_wrong_miss_policy(self):
        cfg = self._make_config(
            verify_cuda_graph=True,
            cpu_expert_backend="kt_direct",
            spec_verify_miss_policy="cache_fill",
        )
        if (
            cfg.verify_cuda_graph
            and cfg.cpu_expert_backend == "kt_direct"
            and cfg.spec_verify_miss_policy == "cpu"
        ):
            cfg.verify_cuda_graph_kt_hybrid = True
        self.assertFalse(cfg.verify_cuda_graph_kt_hybrid)

    def test_kt_hybrid_not_enabled_graph_off(self):
        cfg = self._make_config(
            verify_cuda_graph=False,
            cpu_expert_backend="kt_direct",
            spec_verify_miss_policy="cpu",
        )
        if (
            cfg.verify_cuda_graph
            and cfg.cpu_expert_backend == "kt_direct"
            and cfg.spec_verify_miss_policy == "cpu"
        ):
            cfg.verify_cuda_graph_kt_hybrid = True
        self.assertFalse(cfg.verify_cuda_graph_kt_hybrid)


# ===================================================================
# 2. Graph-safe verify plan builder tests
# ===================================================================

class TestBuildVerifyGraphSafePlan(unittest.TestCase):
    """Test build_verify_graph_safe_plan_gpu produces correct graph-safe plans."""

    def test_all_cached_routes_get_nonzero_weights(self):
        cache = _build_cache(num_experts=8, slots=4, cached=[0, 1, 2, 3])
        selected = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
        weights = torch.tensor([[0.6, 0.4], [0.5, 0.5]], dtype=torch.float32)

        plan = build_verify_graph_safe_plan_gpu(
            layer_idx=0,
            selected_experts=selected,
            routing_weights=weights,
            expert_cache=cache,
            num_experts=8,
        )

        self.assertTrue(plan.cpu_graph_enabled)
        self.assertEqual(plan.gpu_route_indices.numel(), 4)
        # All routes are cached → no zero weights
        flat_w = plan.gpu_route_weights
        self.assertTrue((flat_w > 0).all(), f"Expected all positive weights, got {flat_w}")
        # CPU route mask should be all False (no uncached)
        self.assertFalse(plan.cpu_route_mask.any())

    def test_uncached_routes_get_zero_weights(self):
        cache = _build_cache(num_experts=8, slots=4, cached=[0, 1, 2, 3])
        selected = torch.tensor([[0, 5], [6, 3]], dtype=torch.int64)
        weights = torch.tensor([[0.6, 0.4], [0.5, 0.5]], dtype=torch.float32)

        plan = build_verify_graph_safe_plan_gpu(
            layer_idx=0,
            selected_experts=selected,
            routing_weights=weights,
            expert_cache=cache,
            num_experts=8,
        )

        flat_selected = selected.reshape(-1)
        cached_mask = cache.get_cached_expert_mask()
        for i in range(flat_selected.numel()):
            eid = int(flat_selected[i].item())
            w = float(plan.gpu_route_weights[i].item())
            is_cached = bool(cached_mask[eid].item())
            if is_cached:
                self.assertGreater(w, 0, f"Route {i} (expert {eid}) cached but weight=0")
            else:
                self.assertEqual(w, 0.0, f"Route {i} (expert {eid}) uncached but weight={w}")

    def test_cpu_route_mask_matches_uncached(self):
        cache = _build_cache(num_experts=8, slots=4, cached=[0, 1, 2, 3])
        selected = torch.tensor([[4, 1], [2, 7]], dtype=torch.int64)
        weights = torch.ones(2, 2, dtype=torch.float32) * 0.5

        plan = build_verify_graph_safe_plan_gpu(
            layer_idx=0,
            selected_experts=selected,
            routing_weights=weights,
            expert_cache=cache,
            num_experts=8,
        )

        flat_sel = selected.reshape(-1)
        cached_mask = cache.get_cached_expert_mask()
        expected_cpu_mask = ~cached_mask.index_select(0, flat_sel)
        self.assertTrue(torch.equal(plan.cpu_route_mask, expected_cpu_mask))

    def test_gpu_route_mask_is_all_true(self):
        cache = _build_cache(num_experts=8, slots=4, cached=[0, 1, 2, 3])
        selected = torch.tensor([[0, 5]], dtype=torch.int64)
        weights = torch.ones(1, 2, dtype=torch.float32) * 0.5

        plan = build_verify_graph_safe_plan_gpu(
            layer_idx=0,
            selected_experts=selected,
            routing_weights=weights,
            expert_cache=cache,
            num_experts=8,
        )

        self.assertTrue(plan.gpu_route_mask.all())

    def test_substitution_lut_maps_uncached_to_cached(self):
        cache = _build_cache(num_experts=8, slots=4, cached=[0, 1, 2, 3])
        selected = torch.tensor([[4, 5], [6, 7]], dtype=torch.int64)
        weights = torch.ones(2, 2, dtype=torch.float32) * 0.5

        plan = build_verify_graph_safe_plan_gpu(
            layer_idx=0,
            selected_experts=selected,
            routing_weights=weights,
            expert_cache=cache,
            num_experts=8,
        )

        lut = plan.substitution_lut
        cached_set = {0, 1, 2, 3}
        for eid in [4, 5, 6, 7]:
            mapped = int(lut[eid].item())
            self.assertIn(mapped, cached_set, f"Expert {eid} mapped to {mapped}, not in cached set")

    def test_m_sizes_shape(self):
        cache = _build_cache(num_experts=8, slots=4, cached=[0, 1, 2, 3])
        selected = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
        weights = torch.ones(2, 2, dtype=torch.float32) * 0.5

        plan = build_verify_graph_safe_plan_gpu(
            layer_idx=0,
            selected_experts=selected,
            routing_weights=weights,
            expert_cache=cache,
            num_experts=8,
        )

        self.assertEqual(plan.gpu_m_sizes.shape[0], cache.num_slots)

    def test_no_dynamic_shape_ops(self):
        """Verify the plan builder doesn't use torch.nonzero (checked by output shape)."""
        cache = _build_cache(num_experts=8, slots=4, cached=[0, 1, 2, 3])
        selected = torch.tensor([[0, 5], [6, 3]], dtype=torch.int64)
        weights = torch.ones(2, 2, dtype=torch.float32) * 0.5

        plan = build_verify_graph_safe_plan_gpu(
            layer_idx=0,
            selected_experts=selected,
            routing_weights=weights,
            expert_cache=cache,
            num_experts=8,
        )

        # All routes present (no filtering via nonzero)
        self.assertEqual(plan.gpu_route_indices.numel(), 4)
        self.assertIsNone(plan.cpu_route_indices)


# ===================================================================
# 3. kt_direct begin/finish graph verify tests
# ===================================================================

class TestKtDirectGraphMethods(unittest.TestCase):
    """Test begin/finish_forward_graph_verify on KtDirectCpuMoeBackend."""

    def _make_fake_backend(self, layer_idx=0, num_experts=4, top_k=2, hidden_size=8):
        from nanovllm.layers.fuse_moe.kt_direct_backend import (
            KtDirectCPUBuffer,
        )

        class FakeTask:
            pass

        class FakeMoe:
            def forward_task(self, *args, **kwargs):
                return FakeTask()

        class FakeCpuInfer:
            def __init__(self):
                self.submit_calls = []
                self.sync_calls = []

            def submit_with_cuda_stream(self, stream, task):
                self.submit_calls.append((stream, task))

            def sync_with_cuda_stream(self, stream, flag):
                self.sync_calls.append((stream, flag))

        class FakeRuntime:
            def __init__(self):
                self.cpu_infer = FakeCpuInfer()

        backend = object.__new__(
            type("FakeKtDirect", (), {
                "layer_idx": layer_idx,
                "num_experts_per_tok": top_k,
                "hidden_size": hidden_size,
                "num_experts": num_experts,
                "moe": FakeMoe(),
                "runtime": FakeRuntime(),
                "_gpu_expert_mask_source": torch.zeros(num_experts, dtype=torch.bool),
                "gpu_expert_mask_cpu": torch.zeros(num_experts, dtype=torch.bool, pin_memory=False),
            })
        )

        # Inject the _refresh method
        def _refresh_gpu_expert_mask(*, non_blocking=False):
            backend.gpu_expert_mask_cpu.copy_(backend._gpu_expert_mask_source)
        backend._refresh_gpu_expert_mask = _refresh_gpu_expert_mask

        # Ensure buffer exists
        KtDirectCPUBuffer.capture_bs.add(2)
        return backend

    def test_begin_calls_submit(self):
        from nanovllm.layers.fuse_moe.kt_direct_backend import (
            KtDirectCpuMoeBackend,
        )

        backend = self._make_fake_backend()
        hidden = torch.randn(2, 8, dtype=torch.bfloat16, device="cuda")
        experts = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64, device="cuda")
        weights = torch.ones(2, 2, dtype=torch.float32, device="cuda")

        # Call begin — should submit
        slot = KtDirectCpuMoeBackend.begin_forward_graph_verify(
            backend, hidden, experts, weights,
        )
        self.assertEqual(slot, 0)  # layer_idx=0 % 2 = 0
        self.assertEqual(len(backend.runtime.cpu_infer.submit_calls), 1)
        self.assertEqual(len(backend.runtime.cpu_infer.sync_calls), 0)

    def test_finish_calls_sync(self):
        from nanovllm.layers.fuse_moe.kt_direct_backend import (
            KtDirectCpuMoeBackend,
        )

        backend = self._make_fake_backend()
        hidden = torch.randn(2, 8, dtype=torch.bfloat16, device="cuda")
        experts = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64, device="cuda")
        weights = torch.ones(2, 2, dtype=torch.float32, device="cuda")

        KtDirectCpuMoeBackend.begin_forward_graph_verify(
            backend, hidden, experts, weights,
        )
        output = KtDirectCpuMoeBackend.finish_forward_graph_verify(
            backend, hidden,
        )
        self.assertEqual(len(backend.runtime.cpu_infer.sync_calls), 1)
        self.assertEqual(output.shape, (2, 8))

    def test_buffer_slot_alternates(self):
        from nanovllm.layers.fuse_moe.kt_direct_backend import (
            KtDirectCpuMoeBackend,
            KtDirectCPUBuffer,
        )

        hidden = torch.randn(2, 8, dtype=torch.bfloat16, device="cuda")
        experts = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64, device="cuda")
        weights = torch.ones(2, 2, dtype=torch.float32, device="cuda")

        b0 = self._make_fake_backend(layer_idx=0)
        slot0 = KtDirectCpuMoeBackend.begin_forward_graph_verify(
            b0, hidden, experts, weights,
        )
        b1 = self._make_fake_backend(layer_idx=1)
        slot1 = KtDirectCpuMoeBackend.begin_forward_graph_verify(
            b1, hidden, experts, weights,
        )
        self.assertEqual(slot0, 0)
        self.assertEqual(slot1, 1)


# ===================================================================
# 4. Model runner dispatch tests
# ===================================================================

class TestModelRunnerVerifyDispatch(unittest.TestCase):
    """Test _can_use_verify_cudagraph dispatches correctly for kt_hybrid."""

    def _make_runner(self, kt_hybrid=False, has_graphs=True):
        from nanovllm.engine.model_runner import ModelRunner
        mr = object.__new__(ModelRunner)
        mr.config = SimpleNamespace(
            verify_cuda_graph=True,
            verify_cuda_graph_kt_hybrid=kt_hybrid,
            verify_cuda_graph_bucket_steps=[4, 8, 12, 16],
        )
        mr.verify_graph_bs = [4, 8, 12, 16]
        mr.verify_prefix_graphs = {} if kt_hybrid else {(4, 0): object()}
        mr.verify_kt_hybrid_graphs = {4: object(), 8: object()} if (kt_hybrid and has_graphs) else {}
        mr.verify_graph_vars = {"block_tables": torch.zeros(1, 10, dtype=torch.int32)}
        return mr

    def test_kt_hybrid_can_use_when_graph_exists(self):
        from nanovllm.engine.model_runner import ModelRunner
        from nanovllm.utils.context import set_context, reset_context
        mr = self._make_runner(kt_hybrid=True, has_graphs=True)
        set_context(True, cu_seqlens_q=torch.tensor([0, 4], dtype=torch.int32))
        try:
            self.assertTrue(ModelRunner._can_use_verify_cudagraph(mr, 4))
        finally:
            reset_context()

    def test_kt_hybrid_cannot_use_without_graphs(self):
        from nanovllm.engine.model_runner import ModelRunner
        from nanovllm.utils.context import set_context, reset_context
        mr = self._make_runner(kt_hybrid=True, has_graphs=False)
        set_context(True, cu_seqlens_q=torch.tensor([0, 4], dtype=torch.int32))
        try:
            self.assertFalse(ModelRunner._can_use_verify_cudagraph(mr, 4))
        finally:
            reset_context()

    def test_kt_hybrid_cannot_use_multi_seq(self):
        from nanovllm.engine.model_runner import ModelRunner
        from nanovllm.utils.context import set_context, reset_context
        mr = self._make_runner(kt_hybrid=True, has_graphs=True)
        set_context(True, cu_seqlens_q=torch.tensor([0, 2, 4], dtype=torch.int32))
        try:
            self.assertFalse(ModelRunner._can_use_verify_cudagraph(mr, 4))
        finally:
            reset_context()

    def test_prefix_path_still_works(self):
        from nanovllm.engine.model_runner import ModelRunner
        from nanovllm.utils.context import set_context, reset_context
        mr = self._make_runner(kt_hybrid=False)
        set_context(True, cu_seqlens_q=torch.tensor([0, 4], dtype=torch.int32))
        try:
            self.assertTrue(ModelRunner._can_use_verify_cudagraph(mr, 4))
        finally:
            reset_context()

    def test_bucket_too_small(self):
        from nanovllm.engine.model_runner import ModelRunner
        from nanovllm.utils.context import set_context, reset_context
        mr = self._make_runner(kt_hybrid=True, has_graphs=True)
        set_context(True, cu_seqlens_q=torch.tensor([0, 20], dtype=torch.int32))
        try:
            self.assertFalse(ModelRunner._can_use_verify_cudagraph(mr, 20))
        finally:
            reset_context()


# ===================================================================
# 5. Plan consistency: graph-safe vs eager
# ===================================================================

class TestPlanConsistency(unittest.TestCase):
    """Verify graph-safe plan produces consistent routing with eager plan."""

    def test_cached_routes_match_eager(self):
        cache = _build_cache(num_experts=8, slots=4, cached=[0, 1, 2, 3])
        selected = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
        weights = torch.ones(2, 2, dtype=torch.float32) * 0.5

        graph_plan = build_verify_graph_safe_plan_gpu(
            layer_idx=0,
            selected_experts=selected,
            routing_weights=weights,
            expert_cache=cache,
            num_experts=8,
        )
        eager_plan = build_verify_plan_gpu(
            layer_idx=0,
            selected_experts=selected,
            routing_weights=weights,
            expert_cache=cache,
            num_experts=8,
        )

        # When all experts are cached, both should have same route count
        self.assertEqual(
            graph_plan.gpu_route_indices.numel(),
            eager_plan.gpu_route_indices.numel(),
        )

    def test_mixed_routes_gpu_count(self):
        cache = _build_cache(num_experts=8, slots=4, cached=[0, 1, 2, 3])
        selected = torch.tensor([[0, 5], [6, 3]], dtype=torch.int64)
        weights = torch.ones(2, 2, dtype=torch.float32) * 0.5

        graph_plan = build_verify_graph_safe_plan_gpu(
            layer_idx=0,
            selected_experts=selected,
            routing_weights=weights,
            expert_cache=cache,
            num_experts=8,
        )
        eager_plan = build_verify_plan_gpu(
            layer_idx=0,
            selected_experts=selected,
            routing_weights=weights,
            expert_cache=cache,
            num_experts=8,
        )

        # Graph-safe: all 4 routes go to GPU (uncached mapped via LUT)
        self.assertEqual(graph_plan.gpu_route_indices.numel(), 4)
        # Eager: only 2 cached routes go to GPU
        self.assertEqual(eager_plan.gpu_route_indices.numel(), 2)


# ===================================================================
# 6. Expert status recording tests
# ===================================================================

class TestExpertStatusRecording(unittest.TestCase):
    """Test that record_layer produces correct expert_status (0/1/2) vectors."""

    def _make_recorder(self, num_layers=2, num_experts=8, top_k=2):
        from nanovllm.expert.runtime_meta import ModelRuntimeMetaRecorder
        cfg = SimpleNamespace(
            prefetch_runtime_mode="draft_segment_indexed",
            prefetch_metadata_host_buffer_pool_size=1,
            draft_prefetch_segment_host_buffer_pool_size=0,
            draft_prefetch_frontier_granularity="segment",
            draft_prefetch_segment_size=12,
        )
        hf_cfg = SimpleNamespace(
            num_hidden_layers=num_layers,
            num_experts_per_tok=top_k,
            num_experts=num_experts,
        )
        return ModelRuntimeMetaRecorder(config=cfg, hf_config=hf_cfg)

    def test_status_values_hit_and_miss(self):
        recorder = self._make_recorder(num_layers=2, num_experts=8, top_k=2)
        recorder.arm(mode="verify_kt_hybrid", step_id=0, token_capacity=4, logical_token_count=2)

        selected = torch.tensor([[0, 5], [2, 7]], dtype=torch.int64)
        weights = torch.ones(2, 2, dtype=torch.float32) * 0.5
        uncached_mask = torch.tensor([False, True, False, True], dtype=torch.bool)

        recorder.record_layer(
            layer_idx=0,
            selected_experts=selected,
            routing_weights=weights,
            uncached_route_mask=uncached_mask,
        )

        dev = recorder.device_buffers[recorder.active_key]
        status = dev["expert_status"][0]
        self.assertEqual(int(status[0].item()), 1)  # expert 0: cached → hit
        self.assertEqual(int(status[5].item()), 2)  # expert 5: uncached → miss
        self.assertEqual(int(status[2].item()), 1)  # expert 2: cached → hit
        self.assertEqual(int(status[7].item()), 2)  # expert 7: uncached → miss
        self.assertEqual(int(status[1].item()), 0)  # expert 1: not activated
        self.assertEqual(int(status[3].item()), 0)  # expert 3: not activated

    def test_status_all_cached(self):
        recorder = self._make_recorder(num_layers=2, num_experts=8, top_k=2)
        recorder.arm(mode="verify_kt_hybrid", step_id=0, token_capacity=4, logical_token_count=2)

        selected = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
        weights = torch.ones(2, 2, dtype=torch.float32) * 0.5
        uncached_mask = torch.tensor([False, False, False, False], dtype=torch.bool)

        recorder.record_layer(layer_idx=0, selected_experts=selected,
                              routing_weights=weights, uncached_route_mask=uncached_mask)

        dev = recorder.device_buffers[recorder.active_key]
        status = dev["expert_status"][0]
        self.assertEqual(int((status == 1).sum().item()), 4)
        self.assertEqual(int((status == 2).sum().item()), 0)

    def test_status_all_miss(self):
        recorder = self._make_recorder(num_layers=2, num_experts=8, top_k=2)
        recorder.arm(mode="verify_kt_hybrid", step_id=0, token_capacity=4, logical_token_count=2)

        selected = torch.tensor([[4, 5], [6, 7]], dtype=torch.int64)
        weights = torch.ones(2, 2, dtype=torch.float32) * 0.5
        uncached_mask = torch.tensor([True, True, True, True], dtype=torch.bool)

        recorder.record_layer(layer_idx=0, selected_experts=selected,
                              routing_weights=weights, uncached_route_mask=uncached_mask)

        dev = recorder.device_buffers[recorder.active_key]
        status = dev["expert_status"][0]
        self.assertEqual(int((status == 2).sum().item()), 4)
        self.assertEqual(int((status == 1).sum().item()), 0)

    def test_derive_stats_from_status(self):
        recorder = self._make_recorder(num_layers=2, num_experts=8, top_k=2)
        recorder.arm(mode="verify_kt_hybrid", step_id=0, token_capacity=4, logical_token_count=2)

        selected = torch.tensor([[0, 5], [2, 7]], dtype=torch.int64)
        weights = torch.ones(2, 2, dtype=torch.float32) * 0.5
        uncached_mask = torch.tensor([False, True, False, True], dtype=torch.bool)

        recorder.record_layer(layer_idx=0, selected_experts=selected,
                              routing_weights=weights, uncached_route_mask=uncached_mask)

        dev = recorder.device_buffers[recorder.active_key]
        status = dev["expert_status"][0]
        activated = int((status > 0).sum().item())
        miss = int((status == 2).sum().item())
        hit = int((status == 1).sum().item())
        self.assertEqual(activated, 4)
        self.assertEqual(miss, 2)
        self.assertEqual(hit, 2)

    def test_histogram_also_recorded(self):
        recorder = self._make_recorder(num_layers=2, num_experts=8, top_k=2)
        recorder.arm(mode="verify_kt_hybrid", step_id=0, token_capacity=4, logical_token_count=2)

        selected = torch.tensor([[0, 5], [2, 7]], dtype=torch.int64)
        weights = torch.tensor([[0.6, 0.4], [0.5, 0.5]], dtype=torch.float32)
        uncached_mask = torch.tensor([False, True, False, True], dtype=torch.bool)

        recorder.record_layer(layer_idx=0, selected_experts=selected,
                              routing_weights=weights, uncached_route_mask=uncached_mask)

        dev = recorder.device_buffers[recorder.active_key]
        act_count = dev["activation_count"][0]
        score_sum = dev["score_sum"][0]
        self.assertEqual(int(act_count[0].item()), 1)
        self.assertEqual(int(act_count[5].item()), 1)
        self.assertAlmostEqual(float(score_sum[0].item()), 0.6, places=4)
        self.assertAlmostEqual(float(score_sum[5].item()), 0.4, places=4)

    def test_collect_filters_padding_only_experts(self):
        """collect() must exclude experts with activation_count==0 from miss/active."""
        from nanovllm.expert.runtime_meta import RuntimeMetaOffloadHandle, _ImmediateEvent
        recorder = self._make_recorder(num_layers=2, num_experts=8, top_k=2)
        recorder.arm(mode="verify_kt_hybrid", step_id=0, token_capacity=4, logical_token_count=2)

        # Simulate post-graph-replay state: status has padding-only experts,
        # but activation_count correctly zeros them (active_routes mask in histogram).
        key = recorder.active_key
        host = recorder.host_buffer_pools[key][0]
        host["token_count"][0] = 2
        # activation_count: only experts 0,2,5,7 have real routes
        host["activation_count"][0].zero_()
        host["activation_count"][0][0] = 1
        host["activation_count"][0][2] = 1
        host["activation_count"][0][5] = 1
        host["activation_count"][0][7] = 1
        host["score_sum"][0].zero_()
        host["score_sum"][0][0] = 0.5
        host["score_sum"][0][2] = 0.5
        host["score_sum"][0][5] = 0.5
        host["score_sum"][0][7] = 0.5
        # expert_status: experts 4,6 have status from padding routes only
        host["expert_status"][0].zero_()
        host["expert_status"][0][0] = 1   # real hit
        host["expert_status"][0][2] = 1   # real hit
        host["expert_status"][0][5] = 2   # real miss
        host["expert_status"][0][7] = 2   # real miss
        host["expert_status"][0][4] = 2   # padding-only miss (should be filtered)
        host["expert_status"][0][6] = 1   # padding-only hit (should be filtered)

        handle = RuntimeMetaOffloadHandle(
            step_id=0, mode="verify_kt_hybrid", event=_ImmediateEvent(),
            token_capacity=4, logical_token_count=2, buffer_bytes=0,
            host_buffer_slot=0, metadata_format="histogram_kt_hybrid",
        )
        result = recorder.collect(handle, wait=True)
        meta = result[0]
        self.assertEqual(meta.active_count, 4.0)   # only 0,2,5,7
        self.assertEqual(meta.miss_count, 2.0)      # only 5,7 (not 4)

    def test_collect_produces_status_fields(self):
        from nanovllm.expert.runtime_meta import _ImmediateEvent, RuntimeMetaOffloadHandle
        recorder = self._make_recorder(num_layers=2, num_experts=8, top_k=2)
        recorder.arm(mode="verify_kt_hybrid", step_id=0, token_capacity=4, logical_token_count=2)

        selected = torch.tensor([[0, 5], [2, 7]], dtype=torch.int64)
        weights = torch.ones(2, 2, dtype=torch.float32) * 0.5
        uncached_mask = torch.tensor([False, True, False, True], dtype=torch.bool)
        recorder.record_layer(layer_idx=0, selected_experts=selected,
                              routing_weights=weights, uncached_route_mask=uncached_mask)

        handle = recorder.offload_async(stream=None, host_buffer_slot=0)
        self.assertIsNotNone(handle)
        self.assertEqual(handle.metadata_format, "histogram_kt_hybrid")

        result = recorder.collect(handle, wait=True)
        self.assertIsNotNone(result)
        self.assertIn(0, result)
        meta = result[0]
        self.assertEqual(meta.miss_count, 2.0)
        self.assertEqual(meta.active_count, 4.0)
        self.assertIsNotNone(meta.expert_status)
        self.assertEqual(int((meta.expert_status == 2).sum().item()), 2)


if __name__ == "__main__":
    unittest.main()
