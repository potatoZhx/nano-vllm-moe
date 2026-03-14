from __future__ import annotations

import torch
import torch.nn.functional as F

from nanovllm.expert.cache import LayerExpertCache
from nanovllm.expert.placement import build_moe_execution_plan
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.fuse_moe.functional import fused_moe_linear


def heterogeneous_moe_forward(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_cache: LayerExpertCache,
    cpu_expert_pool: dict[int, dict[str, torch.Tensor]] | None,
    act_fn: SiluAndMul,
) -> torch.Tensor:
    """Run MoE with GPU cached experts + fallback path for uncached experts."""
    M, hidden_dim = hidden_states.shape
    top_k = routing_weights.size(1)
    flat_selected = selected_experts.reshape(-1)
    flat_weights = routing_weights.reshape(-1)

    expanded_hidden = hidden_states.repeat_interleave(top_k, dim=0)
    token_indices = torch.arange(M, device=hidden_states.device, dtype=torch.int64).repeat_interleave(top_k)

    output = torch.zeros_like(hidden_states)
    plan = build_moe_execution_plan(flat_selected, expert_cache)
    gpu_mask = plan.gpu_mask

    # GPU path: cached experts are remapped to contiguous slot buffers.
    if plan.gpu_slots.numel() > 0:
        gpu_hidden = expanded_hidden[gpu_mask][plan.sort_idx]
        gpu_token_indices = token_indices[gpu_mask][plan.sort_idx]
        gpu_weights = flat_weights[gpu_mask][plan.sort_idx]
        gate_up_buffer, down_buffer = expert_cache.get_layer_buffers()

        gate_up = fused_moe_linear(gpu_hidden, gate_up_buffer, plan.m_sizes)
        gpu_expert_out = fused_moe_linear(act_fn(gate_up), down_buffer, plan.m_sizes)
        output.index_add_(0, gpu_token_indices, gpu_expert_out * gpu_weights.unsqueeze(-1))

    # Fallback path for uncached experts (kept for correctness in early integration).
    if plan.gpu_slots.numel() < flat_selected.numel():
        cpu_indices = torch.nonzero(~gpu_mask, as_tuple=False).flatten()
    else:
        cpu_indices = None

    if cpu_indices is not None and cpu_indices.numel() > 0:
        if cpu_expert_pool is None:
            raise RuntimeError("Missing cpu_expert_pool for uncached expert fallback.")
        cpu_hidden = expanded_hidden[cpu_indices]
        cpu_experts = flat_selected[cpu_indices]
        cpu_token_indices = token_indices[cpu_indices]
        cpu_weights = flat_weights[cpu_indices]

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
