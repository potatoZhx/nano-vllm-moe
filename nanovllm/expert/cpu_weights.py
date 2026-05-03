from __future__ import annotations

from dataclasses import dataclass

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
