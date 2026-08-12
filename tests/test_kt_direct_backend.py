from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from nanovllm.expert.cache import LayerExpertCache
from nanovllm.expert.placement import build_prefill_plan_gpu
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.fuse_moe.cpu_backend import CpuMoeResult
from nanovllm.layers.fuse_moe.heterogeneous import heterogeneous_moe_forward
from nanovllm.layers.fuse_moe.kt_direct_backend import (
    KtDirectCpuMoeBackend,
    _build_bf16_weight_ptrs,
    _pack_llamafile_bf16_weights,
    _pack_llamafile_weights,
    _resolve_runtime_layout,
    _select_kt_bf16_moe_class,
    _split_threads,
)


class _FakeTask:
    pass


class _FakeMoe:
    def __init__(self, config) -> None:
        self.config = config
        self.load_calls = 0
        self.forward_calls = 0

    def load_weights_task(self, physical_to_logical_ptr: int) -> _FakeTask:
        self.load_calls += 1
        self.physical_to_logical_ptr = physical_to_logical_ptr
        return _FakeTask()

    def forward_task(
        self,
        batch_size_ptr: int,
        top_k: int,
        expert_ids_ptr: int,
        routing_weights_ptr: int,
        input_ptr: int,
        output_ptr: int,
        incremental: bool,
    ) -> _FakeTask:
        self.forward_calls += 1
        self.last_forward = {
            "batch_size_ptr": batch_size_ptr,
            "top_k": top_k,
            "expert_ids_ptr": expert_ids_ptr,
            "routing_weights_ptr": routing_weights_ptr,
            "input_ptr": input_ptr,
            "output_ptr": output_ptr,
            "incremental": incremental,
        }
        return _FakeTask()


class _FakeMoeConfig:
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        gpu_mask_ptr: int,
    ) -> None:
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gpu_mask_ptr = gpu_mask_ptr


class _FakeCpuInfer:
    def __init__(self) -> None:
        self.backend_ = object()
        self.submit_count = 0
        self.sync_count = 0

    def submit(self, task: _FakeTask) -> None:
        self.submit_count += 1

    def sync(self) -> None:
        self.sync_count += 1


class _FakeRuntime:
    def __init__(self) -> None:
        self.kt_threadpool_count = 1
        self.cpu_infer = _FakeCpuInfer()
        self.kt_moe = SimpleNamespace(
            MOEConfig=_FakeMoeConfig,
            AMXBF16_MOE=_FakeMoe,
            AVX2BF16_MOE=_FakeMoe,
        )


class _FakeLegacyMoeConfig:
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size


class _FakeLegacyMoe:
    def __init__(self, config) -> None:
        self.config = config
        self.load_calls = 0
        self.forward_calls = 0

    def load_weights(self) -> _FakeTask:
        self.load_calls += 1
        return _FakeTask()

    def forward(
        self,
        batch_size_ptr: int,
        top_k: int,
        expert_ids_ptr: int,
        routing_weights_ptr: int,
        input_ptr: int,
        output_ptr: int,
        incremental: bool,
    ) -> _FakeTask:
        self.forward_calls += 1
        self.last_forward = {
            "batch_size_ptr": batch_size_ptr,
            "top_k": top_k,
            "expert_ids_ptr": expert_ids_ptr,
            "routing_weights_ptr": routing_weights_ptr,
            "input_ptr": input_ptr,
            "output_ptr": output_ptr,
            "incremental": incremental,
        }
        return _FakeTask()


class _FakeLegacyRuntime:
    def __init__(self) -> None:
        self.kt_threadpool_count = 1
        self.cpu_infer = _FakeCpuInfer()
        self.kt_moe = SimpleNamespace(
            MOEConfig=_FakeLegacyMoeConfig,
            MOE=_FakeLegacyMoe,
        )


class _RecordingBackend:
    def __init__(self) -> None:
        self.selected_experts: torch.Tensor | None = None
        self.routing_weights: torch.Tensor | None = None

    def forward(self, **kwargs) -> CpuMoeResult:
        self.selected_experts = kwargs.get("selected_experts")
        self.routing_weights = kwargs.get("routing_weights")
        hidden_states = kwargs["hidden_states"]
        return CpuMoeResult(
            token_indices=torch.empty(0, dtype=torch.int64),
            outputs_cpu=torch.zeros_like(hidden_states),
            prep_ms=0.0,
            compute_ms=0.0,
        )


def _make_pool(
    *,
    num_experts: int = 4,
    hidden_size: int = 8,
    intermediate_size: int = 4,
) -> dict[int, dict[str, torch.Tensor]]:
    return {
        expert_idx: {
            "gate_up": torch.randn(
                intermediate_size * 2,
                hidden_size,
                dtype=torch.bfloat16,
            ).contiguous(),
            "down": torch.randn(
                hidden_size,
                intermediate_size,
                dtype=torch.bfloat16,
            ).contiguous(),
        }
        for expert_idx in range(num_experts)
    }


class TestKtDirectBackend(unittest.TestCase):
    def test_split_threads_distributes_remainder(self) -> None:
        self.assertEqual(_split_threads(10, 3), [4, 3, 3])

    def test_runtime_layout_auto_uses_physical_cores_in_selected_numa(self) -> None:
        self.assertEqual(
            _resolve_runtime_layout(
                kt_num_threads=0,
                kt_threadpool_count=1,
                kt_numa_nodes=None,
                core_capacities={0: 4, 1: 4},
            ),
            (4, [0], [4]),
        )
        self.assertEqual(
            _resolve_runtime_layout(
                kt_num_threads=0,
                kt_threadpool_count=2,
                kt_numa_nodes=None,
                core_capacities={0: 4, 1: 4},
            ),
            (8, [0, 1], [4, 4]),
        )

    def test_runtime_layout_rejects_explicit_core_oversubscription(self) -> None:
        with self.assertRaisesRegex(ValueError, "requests 8 cores on NUMA node 0"):
            _resolve_runtime_layout(
                kt_num_threads=8,
                kt_threadpool_count=1,
                kt_numa_nodes=[0],
                core_capacities={0: 4, 1: 4},
            )

    def test_auto_backend_does_not_select_unsupported_amx(self) -> None:
        kt_moe = SimpleNamespace(AMXBF16_MOE=object, AVX2BF16_MOE=None)
        with patch(
            "nanovllm.layers.fuse_moe.kt_direct_backend._cpu_has_flag",
            return_value=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "No supported KTransformers BF16"):
                _select_kt_bf16_moe_class(kt_moe, "auto")

    def test_build_weight_ptrs_uses_all_bf16_experts(self) -> None:
        pool = _make_pool()

        gate_ptrs, up_ptrs, down_ptrs, refs = _build_bf16_weight_ptrs(
            cpu_expert_pool=pool,
            num_experts=4,
            hidden_size=8,
            intermediate_size=4,
            threadpool_count=2,
            strict_dtype=True,
        )

        self.assertEqual(len(gate_ptrs), 2)
        self.assertEqual(len(gate_ptrs[0]), 4)
        self.assertEqual(len(up_ptrs[0]), 4)
        self.assertEqual(len(down_ptrs[0]), 4)
        self.assertEqual(len(refs), 12)
        self.assertEqual(gate_ptrs[0], gate_ptrs[1])
        self.assertEqual(up_ptrs[0], up_ptrs[1])
        self.assertEqual(down_ptrs[0], down_ptrs[1])

    def test_build_weight_ptrs_rejects_missing_expert(self) -> None:
        pool = _make_pool()
        del pool[2]

        with self.assertRaisesRegex(RuntimeError, "missing expert 2"):
            _build_bf16_weight_ptrs(
                cpu_expert_pool=pool,
                num_experts=4,
                hidden_size=8,
                intermediate_size=4,
                threadpool_count=1,
                strict_dtype=True,
            )

    def test_pack_llamafile_weights_preserves_expert_and_projection_order(self) -> None:
        pool = _make_pool()

        gate, up, down = _pack_llamafile_bf16_weights(
            cpu_expert_pool=pool,
            num_experts=4,
            hidden_size=8,
            intermediate_size=4,
            strict_dtype=True,
        )

        self.assertEqual(tuple(gate.shape), (4, 4, 8))
        self.assertEqual(tuple(up.shape), (4, 4, 8))
        self.assertEqual(tuple(down.shape), (4, 8, 4))
        for expert_idx in range(4):
            self.assertTrue(torch.equal(gate[expert_idx], pool[expert_idx]["gate_up"][:4]))
            self.assertTrue(torch.equal(up[expert_idx], pool[expert_idx]["gate_up"][4:]))
            self.assertTrue(torch.equal(down[expert_idx], pool[expert_idx]["down"]))

    def test_llamafile_backend_uses_legacy_api_and_masks_gpu_routes(self) -> None:
        runtime = _FakeLegacyRuntime()
        gpu_mask_source = torch.tensor([True, False, False, True])
        backend = KtDirectCpuMoeBackend(
            layer_idx=1,
            cpu_expert_pool=_make_pool(),
            max_routes=8,
            moe_intermediate_size=4,
            hidden_size=8,
            num_experts=4,
            num_experts_per_tok=2,
            gpu_expert_mask=gpu_mask_source,
            kt_direct_backend="llamafile_bf16",
            kt_capture_bs=[1, 2],
            runtime=runtime,
        )

        self.assertEqual(backend.kt_selected_backend, "llamafile_bf16")
        self.assertEqual(backend.moe.load_calls, 1)
        self.assertEqual(backend.moe.config.gate_type, 30)
        selected = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
        self.assertTrue(
            torch.equal(
                backend._cpu_topk_ids(selected),
                torch.tensor([[-1, 1], [2, -1]], dtype=torch.int64),
            )
        )

        routing_weights = torch.full((2, 2), 0.5, dtype=torch.float32)
        result = backend.forward(
            hidden_states=torch.randn(2, 8, dtype=torch.bfloat16),
            flat_weights=routing_weights.reshape(-1),
            top_k=2,
            cpu_indices=torch.tensor([1, 2], dtype=torch.int64),
            cpu_task_expert_ids=torch.tensor([1, 2], dtype=torch.int64),
            cpu_task_offsets=torch.tensor([0, 1, 2], dtype=torch.int64),
            act_fn=SiluAndMul(),
            selected_experts=selected,
            routing_weights=routing_weights,
        )

        self.assertEqual(backend.moe.forward_calls, 1)
        self.assertEqual(backend.moe.last_forward["top_k"], 2)
        self.assertEqual(tuple(result.outputs_cpu.shape), (2, 8))

    def test_llamafile_f16_converts_weights_but_keeps_bf16_hidden_type(self) -> None:
        pool = _make_pool()
        gate, up, down = _pack_llamafile_weights(
            cpu_expert_pool=pool,
            num_experts=4,
            hidden_size=8,
            intermediate_size=4,
            strict_dtype=True,
            weight_dtype=torch.float16,
        )
        self.assertEqual(gate.dtype, torch.float16)
        self.assertEqual(up.dtype, torch.float16)
        self.assertEqual(down.dtype, torch.float16)
        self.assertTrue(
            torch.equal(gate[0], pool[0]["gate_up"][:4].to(torch.float16))
        )

        backend = KtDirectCpuMoeBackend(
            layer_idx=1,
            cpu_expert_pool=pool,
            max_routes=8,
            moe_intermediate_size=4,
            hidden_size=8,
            num_experts=4,
            num_experts_per_tok=2,
            gpu_expert_mask=torch.tensor([True, False, False, True]),
            kt_direct_backend="llamafile_f16",
            kt_capture_bs=[1, 2],
            runtime=_FakeLegacyRuntime(),
        )
        self.assertEqual(backend.kt_selected_backend, "llamafile_f16")
        self.assertEqual(backend.moe.config.gate_type, 1)
        self.assertEqual(backend.moe.config.up_type, 1)
        self.assertEqual(backend.moe.config.down_type, 1)
        self.assertEqual(backend.moe.config.hidden_type, 30)

    def test_backend_packs_once_and_forwards_full_topk(self) -> None:
        runtime = _FakeRuntime()
        gpu_mask_source = torch.tensor([True, False, False, True])
        backend = KtDirectCpuMoeBackend(
            layer_idx=3,
            cpu_expert_pool=_make_pool(),
            max_routes=8,
            moe_intermediate_size=4,
            hidden_size=8,
            num_experts=4,
            num_experts_per_tok=2,
            gpu_expert_mask=gpu_mask_source,
            kt_direct_backend="avx2_bf16",
            kt_chunked_prefill_size=4096,
            kt_capture_bs=[1, 2, 4],
            runtime=runtime,
        )

        self.assertEqual(backend.load_count, 1)
        self.assertEqual(backend.moe.load_calls, 1)
        self.assertEqual(backend.moe.config.max_len, 4)
        self.assertTrue(backend.supports_batch_size(4))
        self.assertFalse(backend.supports_batch_size(5))
        self.assertEqual(runtime.cpu_infer.submit_count, 1)
        self.assertEqual(runtime.cpu_infer.sync_count, 1)

        hidden_states = torch.randn(2, 8, dtype=torch.bfloat16)
        selected_experts = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
        routing_weights = torch.tensor(
            [[0.75, 0.25], [0.6, 0.4]],
            dtype=torch.float32,
        )
        result = backend.forward(
            hidden_states=hidden_states,
            flat_weights=routing_weights.reshape(-1),
            top_k=2,
            cpu_indices=torch.tensor([1, 2], dtype=torch.int64),
            cpu_task_expert_ids=torch.tensor([1, 2], dtype=torch.int64),
            cpu_task_offsets=torch.tensor([0, 1, 2], dtype=torch.int64),
            act_fn=SiluAndMul(),
            selected_experts=selected_experts,
            routing_weights=routing_weights,
        )

        self.assertEqual(backend.forward_count, 1)
        self.assertEqual(backend.moe.forward_calls, 1)
        self.assertEqual(backend.moe.last_forward["top_k"], 2)
        self.assertFalse(backend.moe.last_forward["incremental"])
        self.assertEqual(tuple(result.outputs_cpu.shape), (2, 8))
        self.assertEqual(result.token_indices.numel(), 0)

        gpu_mask_source.copy_(torch.tensor([False, True, True, False]))
        backend.forward(
            hidden_states=hidden_states,
            flat_weights=routing_weights.reshape(-1),
            top_k=2,
            cpu_indices=torch.tensor([0, 3], dtype=torch.int64),
            cpu_task_expert_ids=torch.tensor([0, 3], dtype=torch.int64),
            cpu_task_offsets=torch.tensor([0, 1, 2], dtype=torch.int64),
            act_fn=SiluAndMul(),
            selected_experts=selected_experts,
            routing_weights=routing_weights,
        )

        self.assertTrue(torch.equal(backend.gpu_expert_mask_cpu, gpu_mask_source))
        self.assertEqual(backend.load_count, 1)
        self.assertEqual(backend.moe.load_calls, 1)

    def test_backend_rejects_batch_larger_than_configured_max_len(self) -> None:
        runtime = _FakeRuntime()
        backend = KtDirectCpuMoeBackend(
            layer_idx=0,
            cpu_expert_pool=_make_pool(),
            max_routes=16,
            moe_intermediate_size=4,
            hidden_size=8,
            num_experts=4,
            num_experts_per_tok=2,
            gpu_expert_mask=torch.zeros(4, dtype=torch.bool),
            kt_direct_backend="avx2_bf16",
            kt_chunked_prefill_size=4096,
            kt_capture_bs=[1, 2, 4],
            runtime=runtime,
        )
        hidden_states = torch.randn(5, 8, dtype=torch.bfloat16)
        selected_experts = torch.zeros((5, 2), dtype=torch.int64)
        routing_weights = torch.full((5, 2), 0.5, dtype=torch.float32)

        with self.assertRaisesRegex(RuntimeError, "batch size 5 exceeds max_len=4"):
            backend.forward(
                hidden_states=hidden_states,
                flat_weights=routing_weights.reshape(-1),
                top_k=2,
                cpu_indices=torch.arange(10, dtype=torch.int64),
                cpu_task_expert_ids=torch.tensor([0], dtype=torch.int64),
                cpu_task_offsets=torch.tensor([0, 10], dtype=torch.int64),
                act_fn=SiluAndMul(),
                selected_experts=selected_experts,
                routing_weights=routing_weights,
            )

    def test_non_parallel_heterogeneous_path_passes_full_routing(self) -> None:
        num_experts = 4
        hidden_size = 8
        intermediate_size = 4
        pool = _make_pool(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        cache = LayerExpertCache(
            num_experts=num_experts,
            slots_per_layer=1,
            gate_up_shape=(intermediate_size * 2, hidden_size),
            down_shape=(hidden_size, intermediate_size),
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            cpu_expert_pool=pool,
        )
        cache.put_to_slot(0, 0, pool[0]["gate_up"], pool[0]["down"])
        hidden_states = torch.randn(2, hidden_size, dtype=torch.bfloat16)
        selected_experts = torch.tensor([[1, 2], [2, 3]], dtype=torch.int64)
        routing_weights = torch.tensor(
            [[0.75, 0.25], [0.6, 0.4]],
            dtype=torch.bfloat16,
        )
        plan = build_prefill_plan_gpu(
            layer_idx=0,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            expert_cache=cache,
            num_experts=num_experts,
        )
        backend = _RecordingBackend()
        backend.min_routes = 1

        heterogeneous_moe_forward(
            hidden_states=hidden_states,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            expert_cache=cache,
            cpu_expert_pool=pool,
            act_fn=SiluAndMul(),
            plan=plan,
            cpu_expert_execution_enabled=True,
            cpu_gpu_parallel_execution_enabled="off",
            cpu_backend=backend,
            cpu_backend_min_routes=32,
        )

        self.assertIs(backend.selected_experts, selected_experts)
        self.assertIs(backend.routing_weights, routing_weights)

    def test_unsupported_large_batch_uses_existing_gpu_fallback(self) -> None:
        num_experts = 4
        hidden_size = 8
        intermediate_size = 4
        pool = _make_pool(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        cache = LayerExpertCache(
            num_experts=num_experts,
            slots_per_layer=1,
            gate_up_shape=(intermediate_size * 2, hidden_size),
            down_shape=(hidden_size, intermediate_size),
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            cpu_expert_pool=pool,
        )
        cache.put_to_slot(0, 0, pool[0]["gate_up"], pool[0]["down"])
        hidden_states = torch.randn(2, hidden_size, dtype=torch.bfloat16)
        selected_experts = torch.tensor([[1, 2], [2, 3]], dtype=torch.int64)
        routing_weights = torch.full((2, 2), 0.5, dtype=torch.bfloat16)
        plan = build_prefill_plan_gpu(
            layer_idx=0,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            expert_cache=cache,
            num_experts=num_experts,
        )
        backend = _RecordingBackend()
        backend.supports_batch_size = lambda batch_size: False

        with patch(
            "nanovllm.layers.fuse_moe.heterogeneous._compute_gpu_fallback_outputs",
            return_value=(
                torch.zeros((4, hidden_size), dtype=torch.bfloat16),
                0.0,
                0.0,
            ),
        ) as gpu_fallback:
            heterogeneous_moe_forward(
                hidden_states=hidden_states,
                selected_experts=selected_experts,
                routing_weights=routing_weights,
                expert_cache=cache,
                cpu_expert_pool=pool,
                act_fn=SiluAndMul(),
                plan=plan,
                cpu_expert_execution_enabled=True,
                cpu_gpu_parallel_execution_enabled="off",
                cpu_backend=backend,
                cpu_backend_min_routes=1,
            )

        gpu_fallback.assert_called_once()
        self.assertIsNone(backend.selected_experts)


if __name__ == "__main__":
    unittest.main()
