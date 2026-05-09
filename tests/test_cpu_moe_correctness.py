from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from nanovllm.expert.cache import LayerExpertCache
from nanovllm.expert.cpu_weights import CpuExpertWeights
from nanovllm.expert.placement import build_prefill_plan_gpu
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.fuse_moe.cpu_backend import FusedTorchCpuMoeBackend, TorchPackedCpuMoeBackend
from nanovllm.layers.fuse_moe.heterogeneous import heterogeneous_moe_forward
from nanovllm.utils.heterogeneous_loader import HeterogeneousModelLoader


def assert_close_moe_output(testcase: unittest.TestCase, ref: torch.Tensor, test: torch.Tensor, *, name: str) -> None:
    ref_f = ref.float()
    test_f = test.float()
    diff = (ref_f - test_f).abs()
    max_abs = float(diff.max().item())
    denom = ref_f.abs().clamp_min(1e-5)
    max_rel = float((diff / denom).max().item())
    testcase.assertLess(max_abs, 5e-2, f"{name}: max_abs too large: {max_abs}")
    testcase.assertLess(max_rel, 5e-2, f"{name}: max_rel too large: {max_rel}")


class TestCpuMoeCorrectness(unittest.TestCase):
    def test_loader_to_cpu_precasts_and_contiguates(self):
        loader = HeterogeneousModelLoader.__new__(HeterogeneousModelLoader)
        loader.hf_config = SimpleNamespace(torch_dtype=torch.bfloat16)
        loader.pin_memory = False

        source = torch.randn(8, 4, dtype=torch.float32).t()
        out = loader._to_cpu(source)

        self.assertEqual(out.device.type, "cpu")
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertTrue(out.is_contiguous())

    def test_cpu_expert_weights_validation(self):
        gate_up = torch.randn(16, 8, dtype=torch.bfloat16).contiguous()
        down = torch.randn(8, 8, dtype=torch.bfloat16).contiguous()
        packed = CpuExpertWeights(expert_idx=0, gate_up=gate_up, down=down, dtype=torch.bfloat16)
        packed.validate()

        bad = CpuExpertWeights(expert_idx=0, gate_up=gate_up.float(), down=down, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "gate_up dtype mismatch"):
            bad.validate()

    def test_torch_packed_backend_rejects_route_overflow(self):
        dtype = torch.float32
        gate_up = torch.randn(16, 8, dtype=dtype).contiguous()
        down = torch.randn(8, 8, dtype=dtype).contiguous()
        packed = CpuExpertWeights(expert_idx=0, gate_up=gate_up, down=down, dtype=dtype)
        cpu_pool: dict[int, dict[str, object]] = {0: {"gate_up": gate_up, "down": down, "packed": packed}}
        backend = TorchPackedCpuMoeBackend(
            layer_idx=0,
            cpu_expert_pool=cpu_pool,
            max_routes=1,
            strict_dtype=True,
        )

        with self.assertRaisesRegex(RuntimeError, "exceed max_routes"):
            backend.forward(
                hidden_states=torch.randn(2, 8, dtype=dtype),
                flat_weights=torch.ones(2, dtype=dtype),
                top_k=1,
                cpu_indices=torch.tensor([0, 1], dtype=torch.int64),
                cpu_task_expert_ids=torch.tensor([0], dtype=torch.int64),
                cpu_task_offsets=torch.tensor([0, 2], dtype=torch.int64),
                act_fn=SiluAndMul(),
            )

    def test_fused_backend_rejects_route_overflow(self):
        dtype = torch.float32
        gate_up = torch.randn(16, 8, dtype=dtype).contiguous()
        down = torch.randn(8, 8, dtype=dtype).contiguous()
        packed = CpuExpertWeights(expert_idx=0, gate_up=gate_up, down=down, dtype=dtype)
        cpu_pool: dict[int, dict[str, object]] = {0: {"gate_up": gate_up, "down": down, "packed": packed}}
        backend = FusedTorchCpuMoeBackend(
            layer_idx=0,
            cpu_expert_pool=cpu_pool,
            max_routes=1,
            moe_intermediate_size=8,
            strict_dtype=True,
        )

        with self.assertRaisesRegex(RuntimeError, "exceed max_routes"):
            backend.forward(
                hidden_states=torch.randn(2, 8, dtype=dtype),
                flat_weights=torch.ones(2, dtype=dtype),
                top_k=1,
                cpu_indices=torch.tensor([0, 1], dtype=torch.int64),
                cpu_task_expert_ids=torch.tensor([0], dtype=torch.int64),
                cpu_task_offsets=torch.tensor([0, 2], dtype=torch.int64),
                act_fn=SiluAndMul(),
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for packed CPU backend alignment")
    def test_torch_packed_backend_matches_torch_backend(self):
        torch.manual_seed(11)
        device = torch.device("cuda")
        dtype = torch.bfloat16

        num_experts = 8
        hidden_size = 128
        intermediate_size = 256
        top_k = 2
        num_tokens = 48
        act_fn = SiluAndMul()

        cpu_pool: dict[int, dict[str, object]] = {}
        for expert_idx in range(num_experts):
            gate_up = (torch.randn(intermediate_size * 2, hidden_size, dtype=dtype) * 0.02).contiguous()
            down = (torch.randn(hidden_size, intermediate_size, dtype=dtype) * 0.02).contiguous()
            packed = CpuExpertWeights(expert_idx=expert_idx, gate_up=gate_up, down=down, dtype=dtype)
            packed.validate()
            cpu_pool[expert_idx] = {"gate_up": gate_up, "down": down, "packed": packed}

        cache = LayerExpertCache(
            num_experts=num_experts,
            slots_per_layer=3,
            gate_up_shape=(intermediate_size * 2, hidden_size),
            down_shape=(hidden_size, intermediate_size),
            device=device,
            dtype=dtype,
            cpu_expert_pool=cpu_pool,
        )
        for slot in range(cache.num_slots):
            params = cpu_pool[slot]
            cache.put_to_slot(slot, slot, params["gate_up"], params["down"])

        hidden_states = (torch.randn(num_tokens, hidden_size, device=device, dtype=dtype) * 0.1).contiguous()
        flat_selected = torch.arange(num_tokens * top_k, device=device, dtype=torch.int64) % num_experts
        selected_experts = flat_selected.view(num_tokens, top_k)
        routing_weights = torch.rand(num_tokens, top_k, device=device, dtype=torch.float32)
        routing_weights = (routing_weights / routing_weights.sum(dim=-1, keepdim=True)).to(dtype)
        plan = build_prefill_plan_gpu(
            layer_idx=0,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            expert_cache=cache,
            num_experts=num_experts,
        )

        out_torch = heterogeneous_moe_forward(
            hidden_states=hidden_states,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            expert_cache=cache,
            cpu_expert_pool=cpu_pool,
            act_fn=act_fn,
            plan=plan,
            cpu_expert_execution_enabled=True,
        )
        packed_backend = TorchPackedCpuMoeBackend(
            layer_idx=0,
            cpu_expert_pool=cpu_pool,
            max_routes=num_tokens * top_k,
            strict_dtype=True,
        )
        profile: dict[str, float] = {}
        out_packed = heterogeneous_moe_forward(
            hidden_states=hidden_states,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            expert_cache=cache,
            cpu_expert_pool=cpu_pool,
            act_fn=act_fn,
            plan=plan,
            cpu_expert_execution_enabled=True,
            cpu_backend=packed_backend,
            profile=profile,
        )

        assert_close_moe_output(self, out_torch, out_packed, name="torch_packed")
        self.assertIn("cpu_to_gpu_merge_ms", profile)
        self.assertIsNotNone(packed_backend.workspace)
        self.assertFalse(packed_backend.workspace.hidden_pin_memory)
        self.assertFalse(packed_backend.workspace.output_pin_memory)

        hidden_data_ptr = packed_backend.workspace.hidden_cpu.data_ptr()
        weights_data_ptr = packed_backend.workspace.weights_cpu.data_ptr()
        outputs_data_ptr = packed_backend.workspace.outputs_cpu.data_ptr()
        out_packed_second = heterogeneous_moe_forward(
            hidden_states=hidden_states,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            expert_cache=cache,
            cpu_expert_pool=cpu_pool,
            act_fn=act_fn,
            plan=plan,
            cpu_expert_execution_enabled=True,
            cpu_backend=packed_backend,
        )

        assert_close_moe_output(self, out_torch, out_packed_second, name="torch_packed_reuse")
        self.assertEqual(hidden_data_ptr, packed_backend.workspace.hidden_cpu.data_ptr())
        self.assertEqual(weights_data_ptr, packed_backend.workspace.weights_cpu.data_ptr())
        self.assertEqual(outputs_data_ptr, packed_backend.workspace.outputs_cpu.data_ptr())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for fused CPU backend alignment")
    def test_fused_backend_matches_torch_backend(self):
        torch.manual_seed(17)
        device = torch.device("cuda")
        dtype = torch.bfloat16

        num_experts = 8
        hidden_size = 128
        intermediate_size = 256
        top_k = 2
        num_tokens = 48
        act_fn = SiluAndMul()

        cpu_pool: dict[int, dict[str, object]] = {}
        for expert_idx in range(num_experts):
            gate_up = (torch.randn(intermediate_size * 2, hidden_size, dtype=dtype) * 0.02).contiguous()
            down = (torch.randn(hidden_size, intermediate_size, dtype=dtype) * 0.02).contiguous()
            packed = CpuExpertWeights(expert_idx=expert_idx, gate_up=gate_up, down=down, dtype=dtype)
            packed.validate()
            cpu_pool[expert_idx] = {"gate_up": gate_up, "down": down, "packed": packed}

        cache = LayerExpertCache(
            num_experts=num_experts,
            slots_per_layer=3,
            gate_up_shape=(intermediate_size * 2, hidden_size),
            down_shape=(hidden_size, intermediate_size),
            device=device,
            dtype=dtype,
            cpu_expert_pool=cpu_pool,
        )
        for slot in range(cache.num_slots):
            params = cpu_pool[slot]
            cache.put_to_slot(slot, slot, params["gate_up"], params["down"])

        hidden_states = (torch.randn(num_tokens, hidden_size, device=device, dtype=dtype) * 0.1).contiguous()
        flat_selected = torch.arange(num_tokens * top_k, device=device, dtype=torch.int64) % num_experts
        selected_experts = flat_selected.view(num_tokens, top_k)
        routing_weights = torch.rand(num_tokens, top_k, device=device, dtype=torch.float32)
        routing_weights = (routing_weights / routing_weights.sum(dim=-1, keepdim=True)).to(dtype)
        plan = build_prefill_plan_gpu(
            layer_idx=0,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            expert_cache=cache,
            num_experts=num_experts,
        )

        out_torch = heterogeneous_moe_forward(
            hidden_states=hidden_states,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            expert_cache=cache,
            cpu_expert_pool=cpu_pool,
            act_fn=act_fn,
            plan=plan,
            cpu_expert_execution_enabled=True,
        )
        fused_backend = FusedTorchCpuMoeBackend(
            layer_idx=0,
            cpu_expert_pool=cpu_pool,
            max_routes=num_tokens * top_k,
            moe_intermediate_size=intermediate_size,
            strict_dtype=True,
        )
        profile: dict[str, float] = {}
        out_fused = heterogeneous_moe_forward(
            hidden_states=hidden_states,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            expert_cache=cache,
            cpu_expert_pool=cpu_pool,
            act_fn=act_fn,
            plan=plan,
            cpu_expert_execution_enabled=True,
            cpu_backend=fused_backend,
            profile=profile,
        )

        assert_close_moe_output(self, out_torch, out_fused, name="fused")
        self.assertIn("cpu_to_gpu_merge_ms", profile)
        self.assertIsNotNone(fused_backend.workspace)
        self.assertIsNotNone(fused_backend.workspace.gate_up_buf)
        self.assertIsNotNone(fused_backend.workspace.act_buf)
        self.assertIsNotNone(fused_backend.workspace.out_fp32_buf)

        gate_up_ptr = fused_backend.workspace.gate_up_buf.data_ptr()
        act_ptr = fused_backend.workspace.act_buf.data_ptr()
        out_ptr = fused_backend.workspace.out_fp32_buf.data_ptr()
        out_fused_second = heterogeneous_moe_forward(
            hidden_states=hidden_states,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            expert_cache=cache,
            cpu_expert_pool=cpu_pool,
            act_fn=act_fn,
            plan=plan,
            cpu_expert_execution_enabled=True,
            cpu_backend=fused_backend,
        )

        assert_close_moe_output(self, out_torch, out_fused_second, name="fused_reuse")
        self.assertEqual(gate_up_ptr, fused_backend.workspace.gate_up_buf.data_ptr())
        self.assertEqual(act_ptr, fused_backend.workspace.act_buf.data_ptr())
        self.assertEqual(out_ptr, fused_backend.workspace.out_fp32_buf.data_ptr())


if __name__ == "__main__":
    unittest.main()
