from __future__ import annotations

from dataclasses import dataclass

import torch

from nanovllm.config import Config


class _ImmediateEvent:
    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        return


@dataclass
class LayerRuntimeMetaCPU:
    step_id: int
    mode: str
    layer_idx: int
    token_count: int
    selected_experts: torch.Tensor
    routing_weights: torch.Tensor


@dataclass
class RuntimeMetaOffloadHandle:
    step_id: int
    mode: str
    event: torch.cuda.Event | _ImmediateEvent
    token_capacity: int
    logical_token_count: int
    buffer_bytes: int


class ModelRuntimeMetaRecorder:
    def __init__(self, config: Config, hf_config):
        self.config = config
        self.num_layers = int(hf_config.num_hidden_layers)
        self.top_k = int(hf_config.num_experts_per_tok)
        self.device_buffers: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
        self.host_buffers: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
        self.active_key: tuple[str, int] | None = None
        self.active_step_id: int = -1
        self.active_mode: str = "idle"
        self.active_logical_token_count: int = 0

    def _ensure_buffer(self, mode: str, token_capacity: int, device: torch.device) -> tuple[str, int]:
        key = (mode, int(token_capacity))
        if key in self.device_buffers:
            return key

        selected_device = torch.empty(
            (self.num_layers, token_capacity, self.top_k),
            dtype=torch.int64,
            device=device,
        )
        weights_device = torch.empty(
            (self.num_layers, token_capacity, self.top_k),
            dtype=torch.float32,
            device=device,
        )
        token_count_device = torch.zeros((self.num_layers,), dtype=torch.int32, device=device)
        token_count_capture_value = torch.zeros((1,), dtype=torch.int32, device=device)

        if device.type == "cuda":
            selected_host = torch.empty(
                (self.num_layers, token_capacity, self.top_k),
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            )
            weights_host = torch.empty(
                (self.num_layers, token_capacity, self.top_k),
                dtype=torch.float32,
                device="cpu",
                pin_memory=True,
            )
            token_count_host = torch.empty((self.num_layers,), dtype=torch.int32, device="cpu", pin_memory=True)
        else:
            selected_host = torch.empty((self.num_layers, token_capacity, self.top_k), dtype=torch.int64)
            weights_host = torch.empty((self.num_layers, token_capacity, self.top_k), dtype=torch.float32)
            token_count_host = torch.empty((self.num_layers,), dtype=torch.int32)

        self.device_buffers[key] = {
            "selected_experts": selected_device,
            "routing_weights": weights_device,
            "token_count": token_count_device,
            "token_count_capture_value": token_count_capture_value,
        }
        self.host_buffers[key] = {
            "selected_experts": selected_host,
            "routing_weights": weights_host,
            "token_count": token_count_host,
        }
        return key

    def arm(
        self,
        mode: str,
        step_id: int,
        token_capacity: int,
        logical_token_count: int | None = None,
    ) -> None:
        if token_capacity <= 0:
            self.reset()
            return
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        key = self._ensure_buffer(mode, int(token_capacity), device)
        self.active_key = key
        self.active_step_id = int(step_id)
        self.active_mode = mode
        self.active_logical_token_count = int(logical_token_count if logical_token_count is not None else token_capacity)
        dev = self.device_buffers[key]
        dev["token_count"].zero_()
        capture_count = min(int(token_capacity), self.active_logical_token_count)
        dev["token_count_capture_value"].fill_(capture_count)

    def record_layer(
        self,
        layer_idx: int,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> None:
        if self.active_key is None:
            return
        dev = self.device_buffers[self.active_key]
        capacity = int(dev["selected_experts"].size(1))
        token_count = min(int(selected_experts.size(0)), capacity)
        if token_count <= 0 or not (0 <= layer_idx < self.num_layers):
            return

        token_count_tensor = dev["token_count"]
        if token_count_tensor.is_cuda and torch.cuda.is_current_stream_capturing():
            # Capture-safe write: avoid host scalar assignment into CUDA tensor.
            token_count_tensor[layer_idx:layer_idx + 1].copy_(dev["token_count_capture_value"])
        else:
            token_count_tensor[layer_idx] = token_count
        dev["selected_experts"][layer_idx, :token_count].copy_(
            selected_experts[:token_count].to(torch.int64),
            non_blocking=True,
        )
        dev["routing_weights"][layer_idx, :token_count].copy_(
            routing_weights[:token_count].float(),
            non_blocking=True,
        )

    def offload_async(
        self,
        stream: torch.cuda.Stream | None,
    ) -> RuntimeMetaOffloadHandle | None:
        if self.active_key is None:
            return None

        key = self.active_key
        dev = self.device_buffers[key]
        host = self.host_buffers[key]
        buffer_bytes = self._buffer_bytes(key)

        if dev["selected_experts"].is_cuda:
            active_stream = stream if stream is not None else torch.cuda.current_stream()
            event = torch.cuda.Event(blocking=False)
            with torch.cuda.stream(active_stream):
                host["token_count"].copy_(dev["token_count"], non_blocking=True)
                host["selected_experts"].copy_(dev["selected_experts"], non_blocking=True)
                host["routing_weights"].copy_(dev["routing_weights"], non_blocking=True)
                event.record(active_stream)
        else:
            host["token_count"].copy_(dev["token_count"])
            host["selected_experts"].copy_(dev["selected_experts"])
            host["routing_weights"].copy_(dev["routing_weights"])
            event = _ImmediateEvent()

        return RuntimeMetaOffloadHandle(
            step_id=self.active_step_id,
            mode=self.active_mode,
            event=event,
            token_capacity=key[1],
            logical_token_count=self.active_logical_token_count,
            buffer_bytes=buffer_bytes,
        )

    def collect(
        self,
        handle: RuntimeMetaOffloadHandle | None,
        wait: bool = False,
    ) -> dict[int, LayerRuntimeMetaCPU] | None:
        if handle is None:
            return None
        if wait:
            handle.event.synchronize()
        elif not handle.event.query():
            return None

        key = (handle.mode, handle.token_capacity)
        host = self.host_buffers[key]
        token_counts = host["token_count"]
        out: dict[int, LayerRuntimeMetaCPU] = {}

        for layer_idx in range(self.num_layers):
            token_count = int(token_counts[layer_idx].item())
            token_count = min(token_count, int(handle.logical_token_count))
            if token_count <= 0:
                continue
            out[layer_idx] = LayerRuntimeMetaCPU(
                step_id=handle.step_id,
                mode=handle.mode,
                layer_idx=layer_idx,
                token_count=token_count,
                selected_experts=host["selected_experts"][layer_idx, :token_count].clone(),
                routing_weights=host["routing_weights"][layer_idx, :token_count].clone(),
            )
        return out

    def reset(self) -> None:
        self.active_key = None
        self.active_step_id = -1
        self.active_mode = "idle"
        self.active_logical_token_count = 0

    def _buffer_bytes(self, key: tuple[str, int]) -> int:
        host = self.host_buffers[key]
        total = 0
        for tensor in host.values():
            total += int(tensor.numel() * tensor.element_size())
        return total
