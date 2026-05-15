import pytest
import torch
import torch.nn.functional as F

from nanovllm.layers.fuse_moe.cpu_backend import FusedTorchCpuMoeBackend
from nanovllm.layers.fuse_moe.heterogeneous import _accumulate_graph_cpu_route_deltas


def _build_pool(num_experts: int, hidden_size: int, intermediate_size: int) -> dict:
    torch.manual_seed(0)
    return {
        expert_id: {
            "gate_up": torch.randn(intermediate_size * 2, hidden_size, dtype=torch.float32),
            "down": torch.randn(hidden_size, intermediate_size, dtype=torch.float32),
        }
        for expert_id in range(num_experts)
    }


def _reference(pool, hidden, selected, weights, mask, intermediate_size: int):
    hidden_cpu = hidden.detach().cpu()
    selected_cpu = selected.detach().cpu()
    weights_cpu = weights.detach().cpu()
    mask_cpu = mask.detach().cpu()
    batch, top_k = selected_cpu.shape
    out = torch.zeros(batch * top_k, hidden_cpu.shape[-1], dtype=torch.float32)
    for token_idx in range(batch):
        for k in range(top_k):
            route_idx = token_idx * top_k + k
            if not bool(mask_cpu[token_idx, k]):
                continue
            expert_id = int(selected_cpu[token_idx, k].item())
            gate_up_w = pool[expert_id]["gate_up"]
            down_w = pool[expert_id]["down"]
            gate_up = F.linear(hidden_cpu[token_idx:token_idx + 1], gate_up_w)
            gate = gate_up[:, :intermediate_size]
            up = gate_up[:, intermediate_size:]
            act = torch.sigmoid(gate) * gate * up
            route_out = F.linear(act, down_w)
            route_out.mul_(weights_cpu[token_idx, k])
            out[route_idx].copy_(route_out[0])
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph bridge requires CUDA")
def test_fused_graph_cpu_backend_replays_with_new_inputs():
    hidden_size = 4
    intermediate_size = 3
    top_k = 2
    batch = 2
    pool = _build_pool(num_experts=4, hidden_size=hidden_size, intermediate_size=intermediate_size)
    backend = FusedTorchCpuMoeBackend(
        layer_idx=0,
        cpu_expert_pool=pool,
        max_routes=batch * top_k,
        moe_intermediate_size=intermediate_size,
        strict_dtype=True,
    )

    hidden = torch.randn(batch, hidden_size, device="cuda", dtype=torch.float32)
    selected = torch.tensor([[0, 3], [1, 2]], device="cuda", dtype=torch.int64)
    weights = torch.tensor([[0.25, 0.75], [0.5, 0.5]], device="cuda", dtype=torch.float32)
    mask = torch.tensor([[True, False], [False, True]], device="cuda", dtype=torch.bool)

    warmup = backend.forward_graph(
        hidden_states=hidden,
        selected_experts=selected,
        routing_weights=weights,
        cpu_route_mask=mask,
        top_k=top_k,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        warmup.outputs_cpu.cpu(),
        _reference(pool, hidden, selected, weights, mask, intermediate_size),
        atol=1e-5,
        rtol=1e-5,
    )

    graph_out = torch.empty_like(warmup.outputs_cpu)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = backend.forward_graph(
            hidden_states=hidden,
            selected_experts=selected,
            routing_weights=weights,
            cpu_route_mask=mask,
            top_k=top_k,
        )
        graph_out.copy_(result.outputs_cpu)

    hidden.copy_(torch.randn_like(hidden))
    weights.copy_(torch.tensor([[0.1, 0.9], [0.8, 0.2]], device="cuda", dtype=torch.float32))
    mask.copy_(torch.tensor([[False, True], [True, False]], device="cuda", dtype=torch.bool))
    expected = _reference(pool, hidden, selected, weights, mask, intermediate_size)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_out.cpu(), expected, atol=1e-5, rtol=1e-5)


def test_graph_cpu_route_deltas_do_not_overwrite_gpu_routes():
    output = torch.zeros((2, 3), dtype=torch.float32)
    top_k = 2
    gpu_route_indices = torch.tensor([2, 0, 3, 1], dtype=torch.int64)
    gpu_expert_out = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # route 2 is CPU-selected on graph path
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [0.0, 0.0, 0.0],  # route 1 is CPU-selected on graph path
        ],
        dtype=torch.float32,
    )
    cpu_route_deltas = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    _accumulate_graph_cpu_route_deltas(
        output=output,
        top_k=top_k,
        gpu_route_indices=gpu_route_indices,
        gpu_expert_out=gpu_expert_out,
        cpu_route_deltas=cpu_route_deltas,
    )

    expected = torch.tensor(
        [
            [8.0, 10.0, 12.0],
            [14.0, 16.0, 18.0],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(output, expected)
