from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=["previous_expert", "slot_idx"],
)
def _unmap_cache_slot_kernel(
    expert_to_slot_lut,
    slot_to_expert_lut,
    cached_expert_mask,
    previous_expert,
    slot_idx,
):
    valid_previous = previous_expert >= 0
    tl.store(
        expert_to_slot_lut + previous_expert,
        -1,
        mask=valid_previous,
    )
    tl.store(
        cached_expert_mask + previous_expert,
        0,
        mask=valid_previous,
    )
    tl.store(slot_to_expert_lut + slot_idx, -1)


@triton.jit(
    do_not_specialize=["previous_expert", "expert_idx", "slot_idx"],
)
def _commit_cache_slot_kernel(
    expert_to_slot_lut,
    slot_to_expert_lut,
    cached_expert_mask,
    previous_expert,
    expert_idx,
    slot_idx,
    EVICT_PREVIOUS: tl.constexpr,
):
    if EVICT_PREVIOUS:
        valid_previous = previous_expert >= 0
        tl.store(
            expert_to_slot_lut + previous_expert,
            -1,
            mask=valid_previous,
        )
        tl.store(
            cached_expert_mask + previous_expert,
            0,
            mask=valid_previous,
        )
    tl.store(slot_to_expert_lut + slot_idx, expert_idx)
    tl.store(expert_to_slot_lut + expert_idx, slot_idx)
    tl.store(cached_expert_mask + expert_idx, 1)


def unmap_cache_slot(
    expert_to_slot_lut: torch.Tensor,
    slot_to_expert_lut: torch.Tensor,
    cached_expert_mask: torch.Tensor,
    *,
    previous_expert: int,
    slot_idx: int,
) -> None:
    """Fuse the three CUDA mapping writes required by an eager eviction."""
    if not expert_to_slot_lut.is_cuda:
        raise ValueError("fused cache LUT updates require CUDA tensors")
    _unmap_cache_slot_kernel[(1,)](
        expert_to_slot_lut,
        slot_to_expert_lut,
        cached_expert_mask,
        int(previous_expert),
        int(slot_idx),
        num_warps=1,
    )


def commit_cache_slot(
    expert_to_slot_lut: torch.Tensor,
    slot_to_expert_lut: torch.Tensor,
    cached_expert_mask: torch.Tensor,
    *,
    previous_expert: int,
    expert_idx: int,
    slot_idx: int,
    evict_previous: bool,
) -> None:
    """Fuse one publication's CUDA LUT and cached-mask writes."""
    if not expert_to_slot_lut.is_cuda:
        raise ValueError("fused cache LUT updates require CUDA tensors")
    _commit_cache_slot_kernel[(1,)](
        expert_to_slot_lut,
        slot_to_expert_lut,
        cached_expert_mask,
        int(previous_expert),
        int(expert_idx),
        int(slot_idx),
        EVICT_PREVIOUS=bool(evict_previous),
        num_warps=1,
    )


def warmup_cache_lut_kernels(device: torch.device) -> None:
    """Compile and launch every runtime specialization before timed decode."""
    if device.type != "cuda":
        return
    expert_to_slot_lut = torch.full(
        (2,), -1, dtype=torch.int64, device=device
    )
    slot_to_expert_lut = torch.full(
        (1,), -1, dtype=torch.int64, device=device
    )
    cached_expert_mask = torch.zeros(
        (2,), dtype=torch.bool, device=device
    )
    unmap_cache_slot(
        expert_to_slot_lut,
        slot_to_expert_lut,
        cached_expert_mask,
        previous_expert=-1,
        slot_idx=0,
    )
    commit_cache_slot(
        expert_to_slot_lut,
        slot_to_expert_lut,
        cached_expert_mask,
        previous_expert=-1,
        expert_idx=0,
        slot_idx=0,
        evict_previous=False,
    )
    commit_cache_slot(
        expert_to_slot_lut,
        slot_to_expert_lut,
        cached_expert_mask,
        previous_expert=0,
        expert_idx=1,
        slot_idx=0,
        evict_previous=True,
    )
