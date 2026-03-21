from __future__ import annotations

import torch
import torch.nn.functional as F

from nanovllm.expert.cache import LayerExpertCache
from nanovllm.expert.placement import MoEExecutionPlan, build_moe_execution_plan
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.fuse_moe.functional import fused_moe_linear


def heterogeneous_moe_forward(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_cache: LayerExpertCache,
    cpu_expert_pool: dict[int, dict[str, torch.Tensor]] | None,
    act_fn: SiluAndMul,
    plan: MoEExecutionPlan | None = None,
) -> torch.Tensor:
    """Run MoE with GPU cached experts + fallback path for uncached experts."""
    M, _ = hidden_states.shape
    top_k = routing_weights.size(1)
    flat_selected = selected_experts.reshape(-1)
    flat_weights = routing_weights.reshape(-1)

    output = torch.zeros_like(hidden_states)
    if plan is None: # heter mode
        plan = build_moe_execution_plan(flat_selected, expert_cache)

    # GPU path: cached experts are remapped to contiguous slot buffers.
    if plan.gpu_route_indices.numel() > 0 and plan.m_sizes is not None:
        gpu_token_indices = torch.div(plan.gpu_route_indices, top_k, rounding_mode="floor")
        gpu_hidden = hidden_states[gpu_token_indices]
        gpu_weights = flat_weights.index_select(0, plan.gpu_route_indices)
        gate_up_buffer, down_buffer = expert_cache.get_layer_buffers()

        gate_up = fused_moe_linear(gpu_hidden, gate_up_buffer, plan.m_sizes)
        gpu_expert_out = fused_moe_linear(act_fn(gate_up), down_buffer, plan.m_sizes)
        gpu_expert_out.mul_(gpu_weights.unsqueeze(-1))
        output.index_add_(0, gpu_token_indices, gpu_expert_out)

    # Fallback path for uncached experts (kept for correctness in early integration).
    cpu_indices = plan.cpu_route_indices

    if cpu_indices is not None and cpu_indices.numel() > 0:
        if cpu_expert_pool is None:
            raise RuntimeError("Missing cpu_expert_pool for uncached expert fallback.")
        cpu_token_indices = torch.div(cpu_indices, top_k, rounding_mode="floor")
        cpu_hidden = hidden_states[cpu_token_indices]
        if plan.flat_selected_original is not None:
            cpu_experts = plan.flat_selected_original.index_select(0, cpu_indices)
        else:
            cpu_experts = flat_selected.index_select(0, cpu_indices)
        cpu_weights = flat_weights.index_select(0, cpu_indices)

        for expert_idx in cpu_experts.unique().tolist():
            expert_mask = cpu_experts == expert_idx
            h = cpu_hidden[expert_mask]
            params = cpu_expert_pool[expert_idx]
            gate_up_weight = params["gate_up"].to(device=hidden_states.device, dtype=hidden_states.dtype, non_blocking=True)
            down_weight = params["down"].to(device=hidden_states.device, dtype=hidden_states.dtype, non_blocking=True)
            gate_up = F.linear(h, gate_up_weight)
            out = F.linear(act_fn(gate_up), down_weight)
            out = out * cpu_weights[expert_mask].unsqueeze(-1)
            output.index_add_(0, cpu_token_indices[expert_mask], out)

    return output
