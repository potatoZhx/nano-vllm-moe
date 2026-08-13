from __future__ import annotations

import pytest
import torch

from nanovllm.layers.fuse_moe.grouped_gemm import (
    fixed_qwen3_moe_config,
    grouped_gemm_forward,
)


def test_fixed_qwen3_gate_up_config_separates_decode_from_prefill() -> None:
    decode = fixed_qwen3_moe_config(16, 1536, 2048)
    prefill = fixed_qwen3_moe_config(1024, 1536, 2048)

    assert decode == {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "num_warps": 4,
        "num_stages": 4,
    }
    assert prefill == {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "num_warps": 4,
        "num_stages": 5,
    }


def test_fixed_qwen3_down_config_separates_decode_from_prefill() -> None:
    decode = fixed_qwen3_moe_config(8, 2048, 768)
    prefill = fixed_qwen3_moe_config(256, 2048, 768)

    assert decode["BLOCK_SIZE_M"] == 16
    assert decode["num_warps"] == 4
    assert decode["num_stages"] == 3
    assert prefill["BLOCK_SIZE_M"] == 32
    assert prefill["num_warps"] == 8
    assert prefill["num_stages"] == 4


def test_fixed_qwen3_config_rejects_unmeasured_shapes() -> None:
    with pytest.raises(ValueError, match="only supports"):
        fixed_qwen3_moe_config(16, 4096, 4096)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fixed_qwen3_dispatch_matches_bf16_reference(monkeypatch) -> None:
    monkeypatch.setenv("NANOVLLM_GROUPED_GEMM_FIXED_QWEN3", "1")
    torch.manual_seed(17)
    device = torch.device("cuda")
    m, k, n, experts = 4, 2048, 1536, 2
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    weights = torch.randn((experts, n, k), device=device, dtype=torch.bfloat16)
    sizes = torch.tensor([2, 2], device=device, dtype=torch.int32)

    actual = grouped_gemm_forward(x, weights, sizes)
    expected = torch.cat((x[:2] @ weights[0].T, x[2:] @ weights[1].T))

    assert torch.equal(actual, expected)
