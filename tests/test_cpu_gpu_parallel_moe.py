import unittest

import torch

from nanovllm.expert.cache import LayerExpertCache
from nanovllm.expert.placement import build_prefill_plan_gpu
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.fuse_moe.heterogeneous import heterogeneous_moe_forward


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for CPU/GPU overlap test")
class TestCpuGpuParallelMoe(unittest.TestCase):
    def test_parallel_overlap_matches_serial_reference(self):
        torch.manual_seed(7)
        torch.cuda.manual_seed_all(7)

        num_experts = 8
        hidden_size = 256
        intermediate_size = 512
        top_k = 2
        num_tokens = 96

        device = torch.device("cuda")
        dtype = torch.bfloat16
        act_fn = SiluAndMul()

        cpu_pool: dict[int, dict[str, torch.Tensor]] = {}
        for eid in range(num_experts):
            cpu_pool[eid] = {
                "gate_up": torch.randn(intermediate_size * 2, hidden_size, dtype=dtype) * 0.02,
                "down": torch.randn(hidden_size, intermediate_size, dtype=dtype) * 0.02,
            }

        cache = LayerExpertCache(
            num_experts=num_experts,
            slots_per_layer=4,
            gate_up_shape=(intermediate_size * 2, hidden_size),
            down_shape=(hidden_size, intermediate_size),
            device=device,
            dtype=dtype,
            cpu_expert_pool=cpu_pool,
        )

        for slot in range(4):
            cache.put_to_slot(slot, slot, cpu_pool[slot]["gate_up"], cpu_pool[slot]["down"])

        hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
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

        out_serial = heterogeneous_moe_forward(
            hidden_states=hidden_states,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            expert_cache=cache,
            cpu_expert_pool=cpu_pool,
            act_fn=act_fn,
            plan=plan,
            cpu_expert_execution_enabled=True,
            cpu_expert_parallel_mode="serial",
            cpu_expert_num_threads=1,
            cpu_gpu_parallel_execution_enabled=False,
            profile=None,
        )

        profile_overlap: dict[str, float] = {}
        out_overlap = heterogeneous_moe_forward(
            hidden_states=hidden_states,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            expert_cache=cache,
            cpu_expert_pool=cpu_pool,
            act_fn=act_fn,
            plan=plan,
            cpu_expert_execution_enabled=True,
            cpu_expert_parallel_mode="serial",
            cpu_expert_num_threads=1,
            cpu_gpu_parallel_execution_enabled=True,
            cpu_gpu_parallel_min_cpu_route_ratio=0.0,
            profile=profile_overlap,
        )

        out_overlap_ep = heterogeneous_moe_forward(
            hidden_states=hidden_states,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            expert_cache=cache,
            cpu_expert_pool=cpu_pool,
            act_fn=act_fn,
            plan=plan,
            cpu_expert_execution_enabled=True,
            cpu_expert_parallel_mode="expert_parallel",
            cpu_expert_num_threads=2,
            cpu_gpu_parallel_execution_enabled=True,
            cpu_gpu_parallel_min_cpu_route_ratio=0.0,
            profile=None,
        )

        self.assertTrue(torch.allclose(out_serial.float(), out_overlap.float(), atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(out_serial.float(), out_overlap_ep.float(), atol=1e-5, rtol=1e-5))
        self.assertGreaterEqual(float(profile_overlap.get("parallel_enabled_count", 0.0)), 1.0)
        self.assertGreaterEqual(float(profile_overlap.get("parallel_overlap_est_ms", 0.0)), 0.0)


if __name__ == "__main__":
    unittest.main()
