from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CpuMoeWorkspace:
    max_routes: int
    hidden_size: int
    intermediate_size: int
    dtype: torch.dtype
    hidden_pin_memory: bool
    output_pin_memory: bool
    hidden_cpu: torch.Tensor
    weights_cpu: torch.Tensor
    outputs_cpu: torch.Tensor
    gate_up_buf: torch.Tensor | None = None   # [max_routes, 2*I] fused backend
    act_buf: torch.Tensor | None = None        # [max_routes, I] fused backend
    out_fp32_buf: torch.Tensor | None = None   # [max_routes, H] fused backend

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
        intermediate_size: int = 0,
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
        gate_up_buf = None
        act_buf = None
        out_fp32_buf = None
        if intermediate_size > 0:
            gate_up_buf = torch.empty(
                (max_routes, intermediate_size * 2),
                device="cpu",
                dtype=dtype,
                pin_memory=False,
            )
            act_buf = torch.empty(
                (max_routes, intermediate_size),
                device="cpu",
                dtype=dtype,
                pin_memory=False,
            )
            out_fp32_buf = torch.empty(
                (max_routes, hidden_size),
                device="cpu",
                dtype=dtype,
                pin_memory=False,
            )
        return cls(
            max_routes=max_routes,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            hidden_pin_memory=hidden_pin_memory,
            output_pin_memory=output_pin_memory,
            hidden_cpu=hidden_cpu,
            weights_cpu=weights_cpu,
            outputs_cpu=outputs_cpu,
            gate_up_buf=gate_up_buf,
            act_buf=act_buf,
            out_fp32_buf=out_fp32_buf,
        )
