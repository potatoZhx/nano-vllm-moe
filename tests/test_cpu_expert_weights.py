from __future__ import annotations

import torch

from nanovllm.expert.cpu_weights import (
    NumaShardedExpertTensor,
    copy_expert_tensor,
)


def test_numa_sharded_gate_up_and_down_materialize_in_logical_order() -> None:
    gate = (
        torch.arange(0, 6, dtype=torch.float16).reshape(2, 3),
        torch.arange(6, 12, dtype=torch.float16).reshape(2, 3),
    )
    up = (
        torch.arange(12, 18, dtype=torch.float16).reshape(2, 3),
        torch.arange(18, 24, dtype=torch.float16).reshape(2, 3),
    )
    down = (
        torch.arange(0, 6, dtype=torch.float16).reshape(3, 2),
        torch.arange(6, 12, dtype=torch.float16).reshape(3, 2),
    )
    gate_up_source = NumaShardedExpertTensor(
        kind="gate_up",
        gate_shards=gate,
        up_shards=up,
    )
    down_source = NumaShardedExpertTensor(kind="down", down_shards=down)

    expected_gate_up = torch.cat((torch.cat(gate), torch.cat(up)))
    expected_down = torch.cat(down, dim=1)
    assert torch.equal(gate_up_source.materialize(), expected_gate_up)
    assert torch.equal(down_source.materialize(), expected_down)

    gate_up_target = torch.empty_like(expected_gate_up)
    down_target = torch.empty_like(expected_down)
    copy_expert_tensor(gate_up_target, gate_up_source)
    copy_expert_tensor(down_target, down_source)
    assert torch.equal(gate_up_target, expected_gate_up)
    assert torch.equal(down_target, expected_down)


def test_numa_sharded_tensor_reports_logical_storage_size() -> None:
    gate_up = NumaShardedExpertTensor(
        kind="gate_up",
        gate_shards=(torch.empty(2, 4, dtype=torch.bfloat16),) * 2,
        up_shards=(torch.empty(2, 4, dtype=torch.bfloat16),) * 2,
    )
    down = NumaShardedExpertTensor(
        kind="down",
        down_shards=(torch.empty(4, 2, dtype=torch.bfloat16),) * 2,
    )

    assert tuple(gate_up.shape) == (8, 4)
    assert tuple(down.shape) == (4, 4)
    assert gate_up.numel() * gate_up.element_size() == 64
    assert down.numel() * down.element_size() == 32
