import unittest

import torch

from nanovllm.layers.fuse_moe.heterogeneous import (
    _run_legacy_gpu_fallback,
    _run_real_cpu_expert_execution,
)


def _act_fn(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(a) * b


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for CPU/GPU expert operator alignment test")
class TestCpuGpuExpertOperatorAlignment(unittest.TestCase):
    def test_real_cpu_execution_matches_gpu_operator_with_tight_error(self):
        torch.manual_seed(123)
        device = torch.device("cuda")
        dtype = torch.bfloat16

        batch_size = 64
        hidden_size = 512
        intermediate_size = 1024
        expert_id = 3

        hidden_states = (torch.randn(batch_size, hidden_size, device=device, dtype=dtype) * 0.1).contiguous()
        selected_experts = torch.full((batch_size, 1), expert_id, device=device, dtype=torch.int64)
        routing_weights = torch.rand(batch_size, 1, device=device, dtype=dtype)

        flat_selected = selected_experts.reshape(-1)
        flat_weights = routing_weights.reshape(-1)
        route_indices = torch.arange(batch_size, device=device, dtype=torch.int64)
        cpu_task_expert_ids = torch.tensor([expert_id], device=device, dtype=torch.int64)
        cpu_task_offsets = torch.tensor([0, batch_size], device=device, dtype=torch.int64)

        cpu_pool = {
            expert_id: {
                "gate_up": torch.randn(2 * intermediate_size, hidden_size, dtype=dtype) * 0.02,
                "down": torch.randn(hidden_size, intermediate_size, dtype=dtype) * 0.02,
            }
        }

        out_cpu = torch.zeros_like(hidden_states)
        out_gpu = torch.zeros_like(hidden_states)

        _run_real_cpu_expert_execution(
            hidden_states=hidden_states,
            output=out_cpu,
            flat_weights=flat_weights,
            top_k=1,
            cpu_indices=route_indices,
            cpu_task_expert_ids=cpu_task_expert_ids,
            cpu_task_offsets=cpu_task_offsets,
            flat_selected_original=flat_selected,
            cpu_expert_pool=cpu_pool,
            act_fn=_act_fn,
        )
        _run_legacy_gpu_fallback(
            hidden_states=hidden_states,
            output=out_gpu,
            flat_weights=flat_weights,
            top_k=1,
            cpu_indices=route_indices,
            flat_selected_original=flat_selected,
            cpu_expert_pool=cpu_pool,
            act_fn=_act_fn,
        )

        diff = (out_cpu - out_gpu).float()
        max_abs = float(diff.abs().max().item())
        mean_abs = float(diff.abs().mean().item())
        rel_l2 = float(diff.norm().item() / (out_gpu.float().norm().item() + 1e-12))

        self.assertLessEqual(max_abs, 1e-4)
        self.assertLessEqual(mean_abs, 1e-6)
        self.assertLessEqual(rel_l2, 5e-4)


if __name__ == "__main__":
    unittest.main()
