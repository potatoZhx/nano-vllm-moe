from __future__ import annotations

from dataclasses import dataclass

import torch


class _ImmediateEvent:
    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        return


@dataclass
class StagingReservation:
    layer_idx: int
    staging_slot_idx: int
    expert_idx: int
    generation: int


@dataclass
class ActiveReservation:
    layer_idx: int
    active_slot_idx: int
    expert_idx: int
    generation: int
    prev_expert: int = -1  # expert evicted to make room; -1 if slot was empty


@dataclass
class PublishedExpert:
    layer_idx: int
    expert_idx: int
    active_slot_idx: int
    staging_slot_idx: int
    generation: int


@dataclass
class LayerCacheSnapshot:
    layer_idx: int
    cached_expert_mask: torch.Tensor
    expert_to_slot_lut: torch.Tensor
    slot_to_expert_lut: torch.Tensor
    last_access_step: list[int]
    access_count: list[int]
    access_score_sum: list[float]
    slot_generation: list[int]


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
        staging_slots_per_layer: int = 0,
        enable_prefetch: bool = False,
    ) -> None:
        self.num_experts = num_experts
        self.num_slots = max(1, min(slots_per_layer, self.num_experts))
        self.cpu_expert_pool = cpu_expert_pool or {}
        self.enable_prefetch = bool(enable_prefetch)
        self.num_staging_slots = int(staging_slots_per_layer) if self.enable_prefetch else 0

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
        self.slot_to_expert_lut = torch.full(
            (self.num_slots,),
            -1,
            dtype=torch.int64,
            device=device,
        )
        self.cached_expert_mask = torch.zeros(
            (self.num_experts,),
            dtype=torch.bool,
            device=device,
        )

        self.last_access_step = [-1] * self.num_experts
        self.access_count = [0] * self.num_experts
        self.access_score_sum = [0.0] * self.num_experts
        self.slot_generation = [0] * self.num_slots
        self._cache_fill_no_cpu_flag_device: torch.Tensor | None = None
        self._cache_fill_no_cpu_flag_host: torch.Tensor | None = None

        if self.num_staging_slots > 0:
            self.staging_gate_up_buffer = torch.empty(
                (self.num_staging_slots, gate_up_shape[0], gate_up_shape[1]),
                device=device,
                dtype=dtype,
            ).contiguous()
            self.staging_down_buffer = torch.empty(
                (self.num_staging_slots, down_shape[0], down_shape[1]),
                device=device,
                dtype=dtype,
            ).contiguous()
        else:
            self.staging_gate_up_buffer = None
            self.staging_down_buffer = None

        self.staging_slot_state = [0] * self.num_staging_slots
        self.staging_slot_to_expert = [-1] * self.num_staging_slots
        self.staging_slot_generation = [0] * self.num_staging_slots
        self.active_slot_pending_expert = [-1] * self.num_slots

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
            self.cached_expert_mask[prev_expert] = False

        self.gate_up_buffer[slot_idx].copy_(gate_up_cpu, non_blocking=True)
        self.down_buffer[slot_idx].copy_(down_cpu, non_blocking=True)
        self.slot_to_expert[slot_idx] = expert_idx
        self.slot_to_expert_lut[slot_idx] = expert_idx
        self.expert_to_slot[expert_idx] = slot_idx
        self.expert_to_slot_lut[expert_idx] = slot_idx
        self.cached_expert_mask[expert_idx] = True
        self.active_slot_pending_expert[slot_idx] = -1
        self.slot_generation[slot_idx] += 1

    def mark_access(
        self,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor | None,
        step_id: int,
    ) -> None:
        if expert_ids.numel() == 0:
            return
        flat_experts = expert_ids.reshape(-1).to(torch.int64)
        unique_ids, inverse = torch.unique(flat_experts, return_inverse=True)
        score_sum = None
        if routing_weights is not None and routing_weights.numel() == expert_ids.numel():
            flat_weights = routing_weights.reshape(-1).float()
            score_sum = torch.zeros((unique_ids.numel(),), dtype=torch.float32, device=flat_weights.device)
            score_sum.scatter_add_(0, inverse, flat_weights)

        for i, expert_idx in enumerate(unique_ids.tolist()):
            if not (0 <= expert_idx < self.num_experts):
                continue
            self.last_access_step[expert_idx] = int(step_id)
            self.access_count[expert_idx] += 1
            if score_sum is not None:
                self.access_score_sum[expert_idx] += float(score_sum[i].item())

    def mark_access_aggregated(
        self,
        expert_ids: torch.Tensor,
        activation_count: torch.Tensor | None,
        score_sum: torch.Tensor | None,
        step_id: int,
    ) -> None:
        if expert_ids.numel() == 0:
            return
        expert_ids_cpu = expert_ids if expert_ids.device.type == "cpu" and expert_ids.dtype == torch.int64 else expert_ids.to(device=torch.device("cpu"), dtype=torch.int64)
        expert_list = expert_ids_cpu.tolist()
        activation_list = (
            (activation_count if activation_count.device.type == "cpu" and activation_count.dtype == torch.int64 else activation_count.to(device=torch.device("cpu"), dtype=torch.int64)).tolist()
            if activation_count is not None
            else None
        )
        score_list = (
            (score_sum if score_sum.device.type == "cpu" and score_sum.dtype == torch.float32 else score_sum.to(device=torch.device("cpu"), dtype=torch.float32)).tolist()
            if score_sum is not None
            else None
        )

        for i, expert_idx in enumerate(expert_list):
            if not (0 <= expert_idx < self.num_experts):
                continue
            self.last_access_step[expert_idx] = int(step_id)
            self.access_count[expert_idx] += int(activation_list[i]) if activation_list is not None else 1
            if score_list is not None:
                self.access_score_sum[expert_idx] += float(score_list[i])

    def snapshot(self, layer_idx: int) -> LayerCacheSnapshot:
        return LayerCacheSnapshot(
            layer_idx=layer_idx,
            cached_expert_mask=self.cached_expert_mask.clone(),
            expert_to_slot_lut=self.expert_to_slot_lut.clone(),
            slot_to_expert_lut=self.slot_to_expert_lut.clone(),
            last_access_step=list(self.last_access_step),
            access_count=list(self.access_count),
            access_score_sum=list(self.access_score_sum),
            slot_generation=list(self.slot_generation),
        )

    def reserve_staging_slot(self, expert_idx: int) -> StagingReservation | None:
        if self.num_staging_slots <= 0:
            return None
        for slot_idx in range(self.num_staging_slots):
            if self.staging_slot_state[slot_idx] == 0:
                self.staging_slot_state[slot_idx] = 1
                self.staging_slot_to_expert[slot_idx] = int(expert_idx)
                self.staging_slot_generation[slot_idx] += 1
                return StagingReservation(
                    layer_idx=-1,
                    staging_slot_idx=slot_idx,
                    expert_idx=int(expert_idx),
                    generation=int(self.staging_slot_generation[slot_idx]),
                )
        return None

    def cancel_staging_reservation(self, reservation: StagingReservation) -> None:
        idx = reservation.staging_slot_idx
        if not (0 <= idx < self.num_staging_slots):
            return
        if self.staging_slot_generation[idx] != reservation.generation:
            return
        self.staging_slot_state[idx] = 0
        self.staging_slot_to_expert[idx] = -1

    def begin_async_put_to_staging(
        self,
        reservation: StagingReservation,
        gate_up_cpu: torch.Tensor,
        down_cpu: torch.Tensor,
        stream: torch.cuda.Stream | None,
    ) -> torch.cuda.Event | _ImmediateEvent:
        if self.staging_gate_up_buffer is None or self.staging_down_buffer is None:
            raise RuntimeError("staging buffers are not initialized")
        idx = reservation.staging_slot_idx
        if self.staging_slot_state[idx] != 1 or self.staging_slot_generation[idx] != reservation.generation:
            raise RuntimeError("invalid staging reservation state")

        target_gate_up = self.staging_gate_up_buffer[idx]
        target_down = self.staging_down_buffer[idx]
        if target_gate_up.is_cuda:
            active_stream = stream if stream is not None else torch.cuda.current_stream()
            with torch.cuda.stream(active_stream):
                target_gate_up.copy_(gate_up_cpu, non_blocking=True)
                target_down.copy_(down_cpu, non_blocking=True)
                event = torch.cuda.Event(blocking=False)
                event.record(active_stream)
            return event

        target_gate_up.copy_(gate_up_cpu)
        target_down.copy_(down_cpu)
        return _ImmediateEvent()

    def mark_staging_ready(self, reservation: StagingReservation) -> bool:
        idx = reservation.staging_slot_idx
        if not (0 <= idx < self.num_staging_slots):
            return False
        if self.staging_slot_state[idx] != 1:
            return False
        if self.staging_slot_generation[idx] != reservation.generation:
            return False
        self.staging_slot_state[idx] = 2
        return True

    def publish_ready_staging_to_active(
        self,
        reservation: StagingReservation,
        active_slot_idx: int,
        stream: torch.cuda.Stream | None,
    ) -> PublishedExpert | None:
        if self.staging_gate_up_buffer is None or self.staging_down_buffer is None:
            return None
        sidx = reservation.staging_slot_idx
        if not (0 <= sidx < self.num_staging_slots):
            return None
        if self.staging_slot_state[sidx] != 2:
            return None
        if self.staging_slot_generation[sidx] != reservation.generation:
            return None
        if not (0 <= active_slot_idx < self.num_slots):
            return None

        if self.gate_up_buffer.is_cuda:
            active_stream = stream if stream is not None else torch.cuda.current_stream()
            with torch.cuda.stream(active_stream):
                self.gate_up_buffer[active_slot_idx].copy_(self.staging_gate_up_buffer[sidx], non_blocking=True)
                self.down_buffer[active_slot_idx].copy_(self.staging_down_buffer[sidx], non_blocking=True)
        else:
            self.gate_up_buffer[active_slot_idx].copy_(self.staging_gate_up_buffer[sidx])
            self.down_buffer[active_slot_idx].copy_(self.staging_down_buffer[sidx])

        return PublishedExpert(
            layer_idx=reservation.layer_idx,
            expert_idx=reservation.expert_idx,
            active_slot_idx=active_slot_idx,
            staging_slot_idx=sidx,
            generation=reservation.generation,
        )

    def commit_published_expert(self, published: PublishedExpert) -> None:
        slot_idx = published.active_slot_idx
        expert_idx = published.expert_idx

        prev_expert = self.slot_to_expert[slot_idx]
        if prev_expert >= 0 and prev_expert in self.expert_to_slot:
            del self.expert_to_slot[prev_expert]
            self.expert_to_slot_lut[prev_expert] = -1
            self.cached_expert_mask[prev_expert] = False

        self.slot_to_expert[slot_idx] = expert_idx
        self.slot_to_expert_lut[slot_idx] = expert_idx
        self.expert_to_slot[expert_idx] = slot_idx
        self.expert_to_slot_lut[expert_idx] = slot_idx
        self.cached_expert_mask[expert_idx] = True
        self.active_slot_pending_expert[slot_idx] = -1
        self.slot_generation[slot_idx] += 1

        sidx = published.staging_slot_idx
        if 0 <= sidx < self.num_staging_slots and self.staging_slot_generation[sidx] == published.generation:
            self.staging_slot_state[sidx] = 0
            self.staging_slot_to_expert[sidx] = -1

    def is_active_slot_pending(self, slot_idx: int) -> bool:
        if not (0 <= slot_idx < self.num_slots):
            return False
        return self.active_slot_pending_expert[slot_idx] >= 0

    def reserve_active_slot_for_prefetch(
        self,
        layer_idx: int,
        active_slot_idx: int,
        expert_idx: int,
    ) -> ActiveReservation | None:
        if not (0 <= active_slot_idx < self.num_slots):
            return None
        if self.active_slot_pending_expert[active_slot_idx] >= 0:
            return None

        prev_expert = self.slot_to_expert[active_slot_idx]
        if prev_expert >= 0 and prev_expert in self.expert_to_slot:
            del self.expert_to_slot[prev_expert]
            self.expert_to_slot_lut[prev_expert] = -1
            self.cached_expert_mask[prev_expert] = False

        self.slot_to_expert[active_slot_idx] = -1
        self.slot_to_expert_lut[active_slot_idx] = -1
        self.active_slot_pending_expert[active_slot_idx] = int(expert_idx)
        self.slot_generation[active_slot_idx] += 1
        return ActiveReservation(
            layer_idx=int(layer_idx),
            active_slot_idx=int(active_slot_idx),
            expert_idx=int(expert_idx),
            generation=int(self.slot_generation[active_slot_idx]),
            prev_expert=int(prev_expert) if prev_expert is not None else -1,
        )

    def reserve_active_slot_for_prefetch_deferred(
        self,
        layer_idx: int,
        active_slot_idx: int,
        expert_idx: int,
    ) -> ActiveReservation | None:
        if not (0 <= active_slot_idx < self.num_slots):
            return None
        if self.active_slot_pending_expert[active_slot_idx] >= 0:
            return None

        prev_expert = self.slot_to_expert[active_slot_idx]
        self.active_slot_pending_expert[active_slot_idx] = int(expert_idx)
        self.slot_generation[active_slot_idx] += 1
        return ActiveReservation(
            layer_idx=int(layer_idx),
            active_slot_idx=int(active_slot_idx),
            expert_idx=int(expert_idx),
            generation=int(self.slot_generation[active_slot_idx]),
            prev_expert=int(prev_expert) if prev_expert is not None else -1,
        )

    def begin_async_put_to_active(
        self,
        reservation: ActiveReservation,
        gate_up_cpu: torch.Tensor,
        down_cpu: torch.Tensor,
        stream: torch.cuda.Stream | None,
    ) -> torch.cuda.Event | _ImmediateEvent:
        idx = reservation.active_slot_idx
        if not (0 <= idx < self.num_slots):
            raise RuntimeError("invalid active prefetch reservation slot")
        if self.slot_generation[idx] != reservation.generation:
            raise RuntimeError("stale active prefetch reservation")
        if self.active_slot_pending_expert[idx] != reservation.expert_idx:
            raise RuntimeError("invalid active prefetch reservation state")

        target_gate_up = self.gate_up_buffer[idx]
        target_down = self.down_buffer[idx]
        if target_gate_up.is_cuda:
            active_stream = stream if stream is not None else torch.cuda.current_stream()
            with torch.cuda.stream(active_stream):
                target_gate_up.copy_(gate_up_cpu, non_blocking=True)
                target_down.copy_(down_cpu, non_blocking=True)
                event = torch.cuda.Event(blocking=False)
                event.record(active_stream)
            return event

        target_gate_up.copy_(gate_up_cpu)
        target_down.copy_(down_cpu)
        return _ImmediateEvent()

    def commit_active_prefetch(self, reservation: ActiveReservation) -> PublishedExpert | None:
        slot_idx = reservation.active_slot_idx
        expert_idx = reservation.expert_idx
        if not (0 <= slot_idx < self.num_slots):
            return None
        if self.slot_generation[slot_idx] != reservation.generation:
            return None
        if self.active_slot_pending_expert[slot_idx] != expert_idx:
            return None

        self.slot_to_expert[slot_idx] = expert_idx
        self.slot_to_expert_lut[slot_idx] = expert_idx
        self.expert_to_slot[expert_idx] = slot_idx
        self.expert_to_slot_lut[expert_idx] = slot_idx
        self.cached_expert_mask[expert_idx] = True
        self.active_slot_pending_expert[slot_idx] = -1
        return PublishedExpert(
            layer_idx=reservation.layer_idx,
            expert_idx=expert_idx,
            active_slot_idx=slot_idx,
            staging_slot_idx=-1,
            generation=reservation.generation,
        )

    def commit_deferred_active_prefetch(self, reservation: ActiveReservation) -> PublishedExpert | None:
        slot_idx = reservation.active_slot_idx
        expert_idx = reservation.expert_idx
        if not (0 <= slot_idx < self.num_slots):
            return None
        if self.slot_generation[slot_idx] != reservation.generation:
            return None
        if self.active_slot_pending_expert[slot_idx] != expert_idx:
            return None
        prev_expert = int(getattr(reservation, "prev_expert", -1))
        if prev_expert >= 0 and self.slot_to_expert[slot_idx] != prev_expert:
            self.active_slot_pending_expert[slot_idx] = -1
            return None

        if prev_expert >= 0 and prev_expert in self.expert_to_slot:
            del self.expert_to_slot[prev_expert]
            self.expert_to_slot_lut[prev_expert] = -1
            self.cached_expert_mask[prev_expert] = False

        self.slot_to_expert[slot_idx] = expert_idx
        self.slot_to_expert_lut[slot_idx] = expert_idx
        self.expert_to_slot[expert_idx] = slot_idx
        self.expert_to_slot_lut[expert_idx] = slot_idx
        self.cached_expert_mask[expert_idx] = True
        self.active_slot_pending_expert[slot_idx] = -1
        return PublishedExpert(
            layer_idx=reservation.layer_idx,
            expert_idx=expert_idx,
            active_slot_idx=slot_idx,
            staging_slot_idx=-1,
            generation=reservation.generation,
        )

    def cancel_deferred_active_prefetch(self, reservation: ActiveReservation) -> None:
        slot_idx = reservation.active_slot_idx
        if not (0 <= slot_idx < self.num_slots):
            return
        if self.slot_generation[slot_idx] != reservation.generation:
            return
        if self.active_slot_pending_expert[slot_idx] != reservation.expert_idx:
            return
        self.active_slot_pending_expert[slot_idx] = -1

    def get_slot_idx(self, expert_idx: int) -> int:
        return self.expert_to_slot.get(expert_idx, -1)

    def is_cached_cpu(self, expert_idx: int) -> bool:
        return int(expert_idx) in self.expert_to_slot

    def is_pending_cpu(self, expert_idx: int) -> bool:
        return int(expert_idx) in self.active_slot_pending_expert

    def remap_experts_to_slots(self, selected_experts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map original expert ids to slot ids; uncached experts are marked as -1."""
        flat_selected = selected_experts.reshape(-1).to(torch.int64)
        slot_indices = self.expert_to_slot_lut.index_select(0, flat_selected).reshape_as(selected_experts)
        gpu_mask = slot_indices >= 0
        return slot_indices, gpu_mask

    def get_layer_buffers(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.gate_up_buffer, self.down_buffer

    def get_cached_expert_mask(self) -> torch.Tensor:
        return self.cached_expert_mask

    def get_slot_to_expert_lut(self) -> torch.Tensor:
        return self.slot_to_expert_lut

    def get_cache_fill_no_cpu_flag_buffers(self) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.expert_to_slot_lut.device
        flags = self._cache_fill_no_cpu_flag_device
        if flags is None or flags.device != device or flags.size(1) != self.num_experts:
            flags = torch.empty((2, self.num_experts), dtype=torch.bool, device=device)
            self._cache_fill_no_cpu_flag_device = flags
        host = self._cache_fill_no_cpu_flag_host
        if host is None or host.size(1) != self.num_experts:
            host_kwargs = {"pin_memory": True} if device.type == "cuda" else {}
            host = torch.empty((2, self.num_experts), dtype=torch.bool, device="cpu", **host_kwargs)
            self._cache_fill_no_cpu_flag_host = host
        return flags, host

    def get_cpu_expert_weights(self, expert_idx: int) -> dict[str, torch.Tensor] | None:
        return self.cpu_expert_pool.get(expert_idx)
