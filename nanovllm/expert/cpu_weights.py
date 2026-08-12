from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class CpuExpertWeights:
    expert_idx: int
    gate_up: torch.Tensor
    down: torch.Tensor
    dtype: torch.dtype

    def validate(self) -> None:
        if self.gate_up.device.type != "cpu":
            raise ValueError("gate_up must be on CPU")
        if self.down.device.type != "cpu":
            raise ValueError("down must be on CPU")
        if not self.gate_up.is_contiguous():
            raise ValueError("gate_up must be contiguous")
        if not self.down.is_contiguous():
            raise ValueError("down must be contiguous")
        if self.gate_up.dtype != self.dtype:
            raise ValueError(f"gate_up dtype mismatch: {self.gate_up.dtype} != {self.dtype}")
        if self.down.dtype != self.dtype:
            raise ValueError(f"down dtype mismatch: {self.down.dtype} != {self.dtype}")


@dataclass(frozen=True)
class NumaShardedExpertTensor:
    """Logical expert tensor backed by CPUInfer's NUMA-local weight shards.

    The legacy llamafile kernel splits the intermediate dimension across its
    NUMA sub-pools.  Keeping these shards as the source for GPU cache fills
    avoids retaining a second, expert-major CPU tensor.
    """

    kind: Literal["gate_up", "down"]
    gate_shards: tuple[torch.Tensor, ...] = ()
    up_shards: tuple[torch.Tensor, ...] = ()
    down_shards: tuple[torch.Tensor, ...] = ()

    def __post_init__(self) -> None:
        if self.kind == "gate_up":
            if not self.gate_shards or len(self.gate_shards) != len(self.up_shards):
                raise ValueError("gate_up requires matching non-empty gate/up shards")
            shards = self.gate_shards + self.up_shards
        elif self.kind == "down":
            if not self.down_shards:
                raise ValueError("down requires non-empty shards")
            shards = self.down_shards
        else:
            raise ValueError(f"unsupported sharded tensor kind: {self.kind}")
        dtype = shards[0].dtype
        for shard in shards:
            if shard.device.type != "cpu":
                raise ValueError("CPUInfer weight shards must be CPU tensors")
            if shard.dtype != dtype:
                raise ValueError("CPUInfer weight shards must have one dtype")
            if not shard.is_contiguous():
                raise ValueError("CPUInfer weight shards must be contiguous")

    @property
    def dtype(self) -> torch.dtype:
        shards = self.gate_shards if self.kind == "gate_up" else self.down_shards
        return shards[0].dtype

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def shape(self) -> torch.Size:
        if self.kind == "gate_up":
            intermediate = sum(int(shard.shape[0]) for shard in self.gate_shards)
            return torch.Size((2 * intermediate, int(self.gate_shards[0].shape[1])))
        return torch.Size(
            (
                int(self.down_shards[0].shape[0]),
                sum(int(shard.shape[1]) for shard in self.down_shards),
            )
        )

    def numel(self) -> int:
        result = 1
        for size in self.shape:
            result *= int(size)
        return result

    def element_size(self) -> int:
        shards = self.gate_shards if self.kind == "gate_up" else self.down_shards
        return int(shards[0].element_size())

    def copy_to(self, target: torch.Tensor, *, non_blocking: bool = False) -> torch.Tensor:
        if tuple(target.shape) != tuple(self.shape):
            raise ValueError(
                f"sharded {self.kind} target shape mismatch: "
                f"got {tuple(target.shape)}, expected {tuple(self.shape)}"
            )
        if self.kind == "gate_up":
            intermediate = int(self.shape[0]) // 2
            offset = 0
            for gate, up in zip(self.gate_shards, self.up_shards, strict=True):
                width = int(gate.shape[0])
                target[offset : offset + width].copy_(gate, non_blocking=non_blocking)
                target[
                    intermediate + offset : intermediate + offset + width
                ].copy_(up, non_blocking=non_blocking)
                offset += width
        else:
            offset = 0
            for down in self.down_shards:
                width = int(down.shape[1])
                target[:, offset : offset + width].copy_(
                    down, non_blocking=non_blocking
                )
                offset += width
        return target

    def materialize(
        self,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> torch.Tensor:
        target = torch.empty(
            tuple(self.shape),
            device=device,
            dtype=self.dtype if dtype is None else dtype,
        )
        return self.copy_to(target, non_blocking=non_blocking)

    def to(
        self,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> torch.Tensor:
        return self.materialize(
            device=device,
            dtype=dtype,
            non_blocking=non_blocking,
        )


ExpertTensor = torch.Tensor | NumaShardedExpertTensor


def copy_expert_tensor(
    target: torch.Tensor,
    source: ExpertTensor,
    *,
    non_blocking: bool = False,
) -> torch.Tensor:
    if isinstance(source, NumaShardedExpertTensor):
        return source.copy_to(target, non_blocking=non_blocking)
    return target.copy_(source, non_blocking=non_blocking)
