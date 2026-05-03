from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CpuMoeWorkspace:
    max_routes: int
    hidden_size: int
    dtype: torch.dtype
    hidden_pin_memory: bool
    output_pin_memory: bool
    hidden_cpu: torch.Tensor
    weights_cpu: torch.Tensor
    outputs_cpu: torch.Tensor

    @property
    def pin_memory(self) -> bool:
        return self.output_pin_memory

    @classmethod
    def create(
        cls,
        *,
        max_routes: int,
        hidden_size: int,
        dtype: torch.dtype,
        hidden_pin_memory: bool,
        output_pin_memory: bool,
    ) -> "CpuMoeWorkspace":
        hidden_cpu = torch.empty(
            (max_routes, hidden_size),
            device="cpu",
            dtype=dtype,
            pin_memory=hidden_pin_memory,
        )
        weights_cpu = torch.empty(
            (max_routes,),
            device="cpu",
            dtype=dtype,
            pin_memory=hidden_pin_memory,
        )
        outputs_cpu = torch.empty(
            (max_routes, hidden_size),
            device="cpu",
            dtype=dtype,
            pin_memory=output_pin_memory,
        )
        return cls(
            max_routes=max_routes,
            hidden_size=hidden_size,
            dtype=dtype,
            hidden_pin_memory=hidden_pin_memory,
            output_pin_memory=output_pin_memory,
            hidden_cpu=hidden_cpu,
            weights_cpu=weights_cpu,
            outputs_cpu=outputs_cpu,
        )
