from __future__ import annotations

import torch


class LayerExpertCache:
    """Per-layer fixed-slot expert cache stored as contiguous [S, N, K] buffers."""

    def __init__(
        self,
        num_experts: int,
        slots_per_layer: int,
        gate_up_shape: tuple[int, int],
        down_shape: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
        cpu_expert_pool: dict[int, dict[str, torch.Tensor]] | None = None,
    ) -> None:
        self.num_experts = num_experts
        self.num_slots = max(1, min(slots_per_layer, self.num_experts))
        self.cpu_expert_pool = cpu_expert_pool or {}

        # Buffers are always contiguous and fixed-size for fused MoE kernels.
        self.gate_up_buffer = torch.empty((self.num_slots, gate_up_shape[0], gate_up_shape[1]), device=device, dtype=dtype).contiguous()
        self.down_buffer = torch.empty((self.num_slots, down_shape[0], down_shape[1]), device=device, dtype=dtype).contiguous()

        # Mapping starts empty and is populated by slot-level writes.
        self.slot_to_expert: list[int] = [-1] * self.num_slots
        self.expert_to_slot: dict[int, int] = {}
        self.expert_to_slot_lut = torch.full(
            (self.num_experts,),
            -1,
            dtype=torch.int64,
            device=device,
        )

    def put_to_slot(
        self,
        slot_idx: int,
        expert_idx: int,
        gate_up_cpu: torch.Tensor,
        down_cpu: torch.Tensor,
    ) -> None:
        assert 0 <= slot_idx < self.num_slots
        prev_expert = self.slot_to_expert[slot_idx]
        if prev_expert >= 0 and prev_expert in self.expert_to_slot:
            del self.expert_to_slot[prev_expert]
            self.expert_to_slot_lut[prev_expert] = -1

        self.gate_up_buffer[slot_idx].copy_(gate_up_cpu, non_blocking=True)
        self.down_buffer[slot_idx].copy_(down_cpu, non_blocking=True)
        self.slot_to_expert[slot_idx] = expert_idx
        self.expert_to_slot[expert_idx] = slot_idx
        self.expert_to_slot_lut[expert_idx] = slot_idx

    def get_slot_idx(self, expert_idx: int) -> int:
        return self.expert_to_slot.get(expert_idx, -1)

    def remap_experts_to_slots(self, selected_experts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map original expert ids to slot ids; uncached experts are marked as -1."""
        flat_selected = selected_experts.reshape(-1).to(torch.int64)
        slot_indices = self.expert_to_slot_lut.index_select(0, flat_selected).reshape_as(selected_experts)
        gpu_mask = slot_indices >= 0
        return slot_indices, gpu_mask

    def get_layer_buffers(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.gate_up_buffer, self.down_buffer

    def get_cpu_expert_weights(self, expert_idx: int) -> dict[str, torch.Tensor] | None:
        return self.cpu_expert_pool.get(expert_idx)
