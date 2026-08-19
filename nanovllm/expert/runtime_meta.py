from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

from nanovllm.config import Config

_SMALL_META_AGGREGATE_NUMEL = 64
_DEFAULT_HOST_BUFFER_POOL_SIZE = 3


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


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
    aggregated_expert_ids: torch.Tensor | None = None
    aggregated_score_sum: torch.Tensor | None = None
    aggregated_activation_count: torch.Tensor | None = None
    selected_experts: torch.Tensor | None = None
    routing_weights: torch.Tensor | None = None
    miss_count: float = 0.0
    active_count: float = 0.0
    expert_status: torch.Tensor | None = None
    route_status: torch.Tensor | None = None
    execution_activation_count: torch.Tensor | None = None
    execution_selected_experts: torch.Tensor | None = None
    execution_routing_weights: torch.Tensor | None = None
    execution_route_status: torch.Tensor | None = None


@dataclass
class RuntimeMetaOffloadHandle:
    step_id: int
    mode: str
    event: torch.cuda.Event | _ImmediateEvent
    token_capacity: int
    logical_token_count: int
    buffer_bytes: int
    execution_token_count: int = 0
    host_buffer_slot: int = 0
    layer_start_idx: int = 0
    layer_end_idx: int | None = None
    metadata_format: str = "raw"


def _aggregate_layer_runtime_meta_cpu(
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    flat_experts = selected_experts.reshape(-1)
    numel = int(flat_experts.numel())
    if numel <= 0 or numel > _SMALL_META_AGGREGATE_NUMEL:
        return None

    flat_experts_cpu = flat_experts if flat_experts.device.type == "cpu" and flat_experts.dtype == torch.int64 else flat_experts.to(device="cpu", dtype=torch.int64)
    unique_ids, inverse = torch.unique(flat_experts_cpu, sorted=True, return_inverse=True)
    if unique_ids.numel() == 0:
        return (
            torch.empty((0,), dtype=torch.int64, device="cpu"),
            torch.empty((0,), dtype=torch.float32, device="cpu"),
            torch.empty((0,), dtype=torch.int64, device="cpu"),
        )

    aggregated_expert_ids = unique_ids.to(dtype=torch.int64, device="cpu")
    aggregated_activation_count = torch.zeros((unique_ids.numel(),), dtype=torch.int64, device="cpu")
    aggregated_activation_count.scatter_add_(0, inverse, torch.ones_like(inverse, dtype=torch.int64))
    aggregated_score_sum = torch.zeros((unique_ids.numel(),), dtype=torch.float32, device="cpu")
    if routing_weights is not None and routing_weights.numel() == selected_experts.numel():
        flat_weights = routing_weights.reshape(-1)
        flat_weights_cpu = flat_weights if flat_weights.device.type == "cpu" and flat_weights.dtype == torch.float32 else flat_weights.to(device="cpu", dtype=torch.float32)
        aggregated_score_sum.scatter_add_(0, inverse, flat_weights_cpu)
    return aggregated_expert_ids, aggregated_score_sum, aggregated_activation_count


class ModelRuntimeMetaRecorder:
    def __init__(self, config: Config, hf_config):
        self.config = config
        self.num_layers = int(hf_config.num_hidden_layers)
        self.top_k = int(hf_config.num_experts_per_tok)
        self.num_experts = int(getattr(hf_config, "num_experts", 0))
        self.device_buffers: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
        self.host_buffers: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
        self.host_buffer_pools: dict[tuple[str, int], list[dict[str, torch.Tensor]]] = {}
        self.host_buffer_pool_size = max(
            1,
            int(getattr(config, "prefetch_metadata_host_buffer_pool_size", _DEFAULT_HOST_BUFFER_POOL_SIZE)),
        )
        self.draft_segment_host_buffer_pool_size = max(
            0,
            int(getattr(config, "draft_prefetch_segment_host_buffer_pool_size", 0)),
        )
        self.active_key: tuple[str, int] | None = None
        self.active_step_id: int = -1
        self.active_mode: str = "idle"
        self.active_logical_token_count: int = 0
        self.active_execution_token_count: int = 0
        self.verify_metadata_lightweight = _env_truthy("NANOVLLM_VERIFY_METADATA_LIGHTWEIGHT")
        self.verify_metadata_omit_score_sum = (
            self.verify_metadata_lightweight
            or _env_truthy("NANOVLLM_VERIFY_METADATA_OMIT_SCORE_SUM")
        )
        self.verify_metadata_omit_status = (
            self.verify_metadata_lightweight
            or _env_truthy("NANOVLLM_VERIFY_METADATA_OMIT_STATUS")
        )
        self.perfect_match_trace_enabled = _env_truthy(
            "NANOVLLM_DRAFT_PERFECT_MATCH_TRACE"
        )
        self.wants_route_status = self.perfect_match_trace_enabled
        self.verify_cost_profile_enabled = _env_truthy(
            "NANOVLLM_VERIFY_COST_MODEL_PROFILE"
        )
        self.transfer_aware_profile_enabled = bool(
            getattr(config, "transfer_aware_profile", False)
        ) or _env_truthy("NANOVLLM_TRANSFER_AWARE_PROFILE")
        if self.transfer_aware_profile_enabled:
            self.wants_route_status = True

    def target_host_buffer_pool_size(self, mode: str, token_capacity: int) -> int:
        _ = token_capacity
        target = int(self.host_buffer_pool_size)
        if (
            str(mode) == "draft"
            and str(getattr(self.config, "prefetch_runtime_mode", "baseline_staging"))
            in {"draft_direct_active", "draft_segment_indexed"}
            and str(getattr(self.config, "draft_prefetch_frontier_granularity", "segment")) in {"segment", "layer"}
        ):
            configured = int(self.draft_segment_host_buffer_pool_size)
            if configured > 0:
                target = max(target, configured)
            else:
                granularity = str(getattr(self.config, "draft_prefetch_frontier_granularity", "segment"))
                segment_size = 1 if granularity == "layer" else max(1, int(getattr(self.config, "draft_prefetch_segment_size", 12)))
                segment_count = max(1, (self.num_layers + segment_size - 1) // segment_size)
                target = max(target, min(64, segment_count + 2))
        if str(mode) == "verify_kt_hybrid":
            segment_size = max(1, int(getattr(self.config, "verify_prefetch_segment_size", 12)))
            segment_count = max(1, (self.num_layers + segment_size - 1) // segment_size)
            target = max(target, min(64, segment_count + 2))
        return max(1, target)

    def _use_histogram_metadata(self, mode: str) -> bool:
        if self.transfer_aware_profile_enabled and str(mode) in {
            "verify",
            "verify_kt_hybrid",
        }:
            return False
        if self.perfect_match_trace_enabled and str(mode) in {
            "draft",
            "verify_kt_hybrid",
        }:
            return False
        if str(mode) == "verify_kt_hybrid" and self.num_experts > 0:
            return True
        return (
            str(mode) == "draft"
            and str(getattr(self.config, "prefetch_runtime_mode", "baseline_staging")) == "draft_segment_indexed"
            and str(getattr(self.config, "prefetch_runtime_kind", "legacy")) != "dual_queue"
            and self.num_experts > 0
        )

    def _use_score_sum_metadata(self, mode: str) -> bool:
        if self.perfect_match_trace_enabled and str(mode) in {
            "draft",
            "verify_kt_hybrid",
        }:
            return False
        return (
            str(mode) == "draft"
            and str(getattr(self.config, "prefetch_runtime_kind", "legacy")) == "dual_queue"
            and self.num_experts > 0
        )

    def _use_histogram_score_sum(self, mode: str) -> bool:
        return not (str(mode) == "verify_kt_hybrid" and self.verify_metadata_omit_score_sum)

    def _use_verify_expert_status(self, mode: str) -> bool:
        return str(mode) == "verify_kt_hybrid" and not self.verify_metadata_omit_status

    def _make_host_buffer(self, mode: str, token_capacity: int, device: torch.device) -> dict[str, torch.Tensor]:
        if self._use_score_sum_metadata(mode):
            pin = device.type == "cuda"
            return {
                "token_count": torch.empty((self.num_layers,), dtype=torch.int32, device="cpu", pin_memory=pin),
                "score_sum": torch.empty(
                    (self.num_layers, self.num_experts), dtype=torch.float32, device="cpu", pin_memory=pin,
                ),
            }
        if self._use_histogram_metadata(mode):
            is_kt_hybrid = str(mode) == "verify_kt_hybrid"
            pin = device.type == "cuda"
            buf: dict[str, torch.Tensor] = {
                "token_count": torch.empty((self.num_layers,), dtype=torch.int32, device="cpu", pin_memory=pin),
                "activation_count": torch.empty(
                    (self.num_layers, self.num_experts), dtype=torch.int32, device="cpu", pin_memory=pin,
                ),
            }
            if self._use_histogram_score_sum(mode):
                buf["score_sum"] = torch.empty(
                    (self.num_layers, self.num_experts), dtype=torch.float32, device="cpu", pin_memory=pin,
                )
            if is_kt_hybrid and self._use_verify_expert_status(mode):
                buf["expert_status"] = torch.empty(
                    (self.num_layers, self.num_experts), dtype=torch.int8, device="cpu", pin_memory=pin,
                )
            if is_kt_hybrid and self.verify_cost_profile_enabled:
                buf["execution_activation_count"] = torch.empty(
                    (self.num_layers, self.num_experts), dtype=torch.int32, device="cpu", pin_memory=pin,
                )
            return buf

        if device.type == "cuda":
            buf = {
                "selected_experts": torch.empty(
                    (self.num_layers, token_capacity, self.top_k),
                    dtype=torch.int64,
                    device="cpu",
                    pin_memory=True,
                ),
                "routing_weights": torch.empty(
                    (self.num_layers, token_capacity, self.top_k),
                    dtype=torch.float32,
                    device="cpu",
                    pin_memory=True,
                ),
                "token_count": torch.empty((self.num_layers,), dtype=torch.int32, device="cpu", pin_memory=True),
            }
            if self.wants_route_status:
                buf["route_status"] = torch.empty(
                    (self.num_layers, token_capacity, self.top_k),
                    dtype=torch.int8,
                    device="cpu",
                    pin_memory=True,
                )
            return buf
        buf = {
            "selected_experts": torch.empty((self.num_layers, token_capacity, self.top_k), dtype=torch.int64),
            "routing_weights": torch.empty((self.num_layers, token_capacity, self.top_k), dtype=torch.float32),
            "token_count": torch.empty((self.num_layers,), dtype=torch.int32),
        }
        if self.wants_route_status:
            buf["route_status"] = torch.empty(
                (self.num_layers, token_capacity, self.top_k),
                dtype=torch.int8,
            )
        return buf

    def _ensure_buffer(self, mode: str, token_capacity: int, device: torch.device) -> tuple[str, int]:
        key = (mode, int(token_capacity))
        if key in self.device_buffers:
            return key

        token_count_device = torch.zeros((self.num_layers,), dtype=torch.int32, device=device)
        token_count_capture_value = torch.zeros((1,), dtype=torch.int32, device=device)
        if self._use_score_sum_metadata(mode):
            self.device_buffers[key] = {
                "score_sum": torch.zeros((self.num_layers, self.num_experts), dtype=torch.float32, device=device),
                "token_count": token_count_device,
                "token_count_capture_value": token_count_capture_value,
                "token_positions": torch.arange(int(token_capacity), dtype=torch.int32, device=device),
            }
        elif self._use_histogram_metadata(mode):
            dev_buf: dict[str, torch.Tensor] = {
                "activation_count": torch.zeros((self.num_layers, self.num_experts), dtype=torch.int32, device=device),
                "token_count": token_count_device,
                "token_count_capture_value": token_count_capture_value,
                "token_positions": torch.arange(int(token_capacity), dtype=torch.int32, device=device),
                "one_count": torch.ones(int(token_capacity) * self.top_k, dtype=torch.int32, device=device),
            }
            if self._use_histogram_score_sum(mode):
                dev_buf["score_sum"] = torch.zeros(
                    (self.num_layers, self.num_experts), dtype=torch.float32, device=device,
                )
            if self._use_verify_expert_status(mode):
                dev_buf["expert_status"] = torch.zeros(
                    (self.num_layers, self.num_experts), dtype=torch.int8, device=device,
                )
                dev_buf["expert_status_hit_val"] = torch.ones(1, dtype=torch.int8, device=device)
                dev_buf["expert_status_miss_val"] = torch.full((1,), 2, dtype=torch.int8, device=device)
                dev_buf["expert_status_vals"] = torch.empty(
                    int(token_capacity) * self.top_k, dtype=torch.int8, device=device,
                )
            if str(mode) == "verify_kt_hybrid" and self.verify_cost_profile_enabled:
                dev_buf["execution_activation_count"] = torch.zeros(
                    (self.num_layers, self.num_experts), dtype=torch.int32, device=device,
                )
            self.device_buffers[key] = dev_buf
        else:
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
            self.device_buffers[key] = {
                "selected_experts": selected_device,
                "routing_weights": weights_device,
                "token_count": token_count_device,
                "token_count_capture_value": token_count_capture_value,
            }
            if self.wants_route_status:
                self.device_buffers[key]["route_status"] = torch.zeros(
                    (self.num_layers, token_capacity, self.top_k),
                    dtype=torch.int8,
                    device=device,
                )
                self.device_buffers[key]["route_status_hit_val"] = torch.ones(
                    1, dtype=torch.int8, device=device
                )
                self.device_buffers[key]["route_status_miss_val"] = torch.full(
                    (1,), 2, dtype=torch.int8, device=device
                )
        host_buffer = self._make_host_buffer(mode, token_capacity, device)
        self.host_buffers[key] = host_buffer
        self.host_buffer_pools[key] = [host_buffer]
        return key

    def get_host_buffer_pool_size(self, mode: str, token_capacity: int) -> int:
        key = (mode, int(token_capacity))
        pool = self.host_buffer_pools.get(key)
        return len(pool) if pool is not None else 0

    def maybe_grow_host_buffer_pool(self, mode: str, token_capacity: int) -> bool:
        key = (mode, int(token_capacity))
        if key not in self.device_buffers:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self._ensure_buffer(mode, int(token_capacity), device)
        pool = self.host_buffer_pools[key]
        if len(pool) >= self.target_host_buffer_pool_size(mode, int(token_capacity)):
            return False
        device = next(tensor.device for tensor in self.device_buffers[key].values() if isinstance(tensor, torch.Tensor))
        pool.append(self._make_host_buffer(mode, int(token_capacity), device))
        return True

    def arm(
        self,
        mode: str,
        step_id: int,
        token_capacity: int,
        logical_token_count: int | None = None,
        execution_token_count: int | None = None,
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
        self.active_execution_token_count = int(
            execution_token_count
            if execution_token_count is not None
            else token_capacity
        )
        dev = self.device_buffers[key]
        dev["token_count"].zero_()
        if "activation_count" in dev:
            dev["activation_count"].zero_()
            if "score_sum" in dev:
                dev["score_sum"].zero_()
        elif "score_sum" in dev:
            dev["score_sum"].zero_()
        if "expert_status" in dev:
            dev["expert_status"].zero_()
        if "execution_activation_count" in dev:
            dev["execution_activation_count"].zero_()
        if "route_status" in dev:
            dev["route_status"].zero_()
        capture_count = min(int(token_capacity), self.active_logical_token_count)
        dev["token_count_capture_value"].fill_(capture_count)

    def record_layer(
        self,
        layer_idx: int,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        uncached_route_mask: torch.Tensor | None = None,
    ) -> None:
        if self.active_key is None:
            return
        dev = self.device_buffers[self.active_key]
        capacity = int(self.active_key[1])
        token_count = min(int(selected_experts.size(0)), capacity)
        if token_count <= 0 or not (0 <= layer_idx < self.num_layers):
            return

        token_count_tensor = dev["token_count"]
        if token_count_tensor.is_cuda and torch.cuda.is_current_stream_capturing():
            token_count_tensor[layer_idx:layer_idx + 1].copy_(dev["token_count_capture_value"])
        else:
            token_count_tensor[layer_idx] = token_count
        if "activation_count" in dev:
            count_row = dev["activation_count"][layer_idx]
            execution_count_row = (
                dev["execution_activation_count"][layer_idx]
                if "execution_activation_count" in dev
                else None
            )
            score_row = dev["score_sum"][layer_idx] if "score_sum" in dev else None
            count_row.zero_()
            if execution_count_row is not None:
                execution_count_row.zero_()
            if score_row is not None:
                score_row.zero_()
            is_capturing = bool(count_row.is_cuda and torch.cuda.is_current_stream_capturing())
            histogram_token_count = token_count if is_capturing else min(token_count, int(self.active_logical_token_count))
            flat_ids = selected_experts[:histogram_token_count].reshape(-1).to(
                device=count_row.device,
                dtype=torch.int64,
            )
            execution_flat_ids = selected_experts[:token_count].reshape(-1).to(
                device=count_row.device,
                dtype=torch.int64,
            )
            if flat_ids.numel() <= 0:
                return
            flat_weights = None
            if score_row is not None:
                flat_weights = routing_weights[:histogram_token_count].reshape(-1).to(
                    device=score_row.device,
                    dtype=torch.float32,
                )
            if is_capturing:
                active_tokens = dev["token_positions"][:token_count].lt(dev["token_count_capture_value"][0])
                active_routes = active_tokens[:, None].expand(token_count, self.top_k).reshape(-1)
                count_values = active_routes.to(dtype=count_row.dtype)
                if flat_weights is not None:
                    score_values = flat_weights * active_routes.to(dtype=flat_weights.dtype)
            else:
                count_values = dev["one_count"][: flat_ids.numel()].to(dtype=count_row.dtype)
                if flat_weights is not None:
                    score_values = flat_weights
            count_row.scatter_add_(0, flat_ids, count_values)
            if execution_count_row is not None and execution_flat_ids.numel() > 0:
                execution_count_row.scatter_add_(
                    0,
                    execution_flat_ids,
                    dev["one_count"][: execution_flat_ids.numel()].to(
                        dtype=execution_count_row.dtype
                    ),
                )
            if score_row is not None and flat_weights is not None:
                score_row.scatter_add_(0, flat_ids, score_values)
            if "expert_status" in dev and uncached_route_mask is not None:
                status_row = dev["expert_status"][layer_idx]
                status_row.zero_()
                status_ids = execution_flat_ids
                is_cached = ~uncached_route_mask[:status_ids.numel()].to(device=status_row.device)
                hit_val = dev["expert_status_hit_val"]
                miss_val = dev["expert_status_miss_val"]
                status_vals = dev["expert_status_vals"][:status_ids.numel()]
                torch.where(is_cached, hit_val.expand(status_ids.numel()), miss_val.expand(status_ids.numel()), out=status_vals)
                # No active_routes mask: cached status is per-expert so all routes
                # (real + padding) to the same expert write the same value.
                # Padding-only experts are filtered out via activation_count > 0
                # at readback time.
                status_row.scatter_(0, status_ids, status_vals)
            return
        if "score_sum" in dev:
            score_row = dev["score_sum"][layer_idx]
            score_row.zero_()
            is_capturing = bool(score_row.is_cuda and torch.cuda.is_current_stream_capturing())
            histogram_token_count = token_count if is_capturing else min(
                token_count, int(self.active_logical_token_count)
            )
            flat_ids = selected_experts[:histogram_token_count].reshape(-1).to(
                device=score_row.device,
                dtype=torch.int64,
            )
            if flat_ids.numel() <= 0:
                return
            flat_weights = routing_weights[:histogram_token_count].reshape(-1).to(
                device=score_row.device,
                dtype=torch.float32,
            )
            if is_capturing:
                active_tokens = dev["token_positions"][:token_count].lt(dev["token_count_capture_value"][0])
                active_routes = active_tokens[:, None].expand(token_count, self.top_k).reshape(-1)
                flat_weights = flat_weights * active_routes.to(dtype=flat_weights.dtype)
            score_row.scatter_add_(0, flat_ids, flat_weights)
            return
        dev["selected_experts"][layer_idx, :token_count].copy_(
            selected_experts[:token_count].to(torch.int64),
            non_blocking=True,
        )
        dev["routing_weights"][layer_idx, :token_count].copy_(
            routing_weights[:token_count].float(),
            non_blocking=True,
        )
        if "route_status" in dev:
            status_row = dev["route_status"][layer_idx]
            status_row.zero_()
            if uncached_route_mask is not None:
                flat_status_count = token_count * self.top_k
                is_miss = uncached_route_mask[:flat_status_count].reshape(
                    token_count, self.top_k
                ).to(device=status_row.device)
                hit_val = dev["route_status_hit_val"]
                miss_val = dev["route_status_miss_val"]
                torch.where(
                    is_miss,
                    miss_val.expand_as(is_miss).to(dtype=torch.int8),
                    hit_val.expand_as(is_miss).to(dtype=torch.int8),
                    out=status_row[:token_count],
                )

    def offload_async(
        self,
        stream: torch.cuda.Stream | None,
        host_buffer_slot: int = 0,
        layer_start_idx: int | None = None,
        layer_end_idx: int | None = None,
    ) -> RuntimeMetaOffloadHandle | None:
        if self.active_key is None:
            return None

        key = self.active_key
        dev = self.device_buffers[key]
        host_pool = self.host_buffer_pools[key]
        host_slot = int(host_buffer_slot)
        if not (0 <= host_slot < len(host_pool)):
            raise IndexError(f"host_buffer_slot out of range: {host_slot}")
        host = host_pool[host_slot]
        layer_start = 0 if layer_start_idx is None else max(0, int(layer_start_idx))
        layer_end = self.num_layers if layer_end_idx is None else min(self.num_layers, int(layer_end_idx))
        layer_start = min(layer_start, self.num_layers)
        layer_end = max(layer_start, layer_end)
        buffer_bytes = self._buffer_bytes(key, layer_start, layer_end)
        has_status = "expert_status" in dev
        metadata_format = (
            "histogram_kt_hybrid"
            if has_status
            else (
                "histogram"
                if "activation_count" in dev
                else ("score_sum" if "score_sum" in dev else "raw")
            )
        )

        device_tensor = next(tensor for tensor in dev.values() if isinstance(tensor, torch.Tensor))
        if device_tensor.is_cuda:
            active_stream = stream if stream is not None else torch.cuda.current_stream()
            current_stream = torch.cuda.current_stream()
            event = torch.cuda.Event(blocking=False)
            with torch.cuda.stream(active_stream):
                if active_stream != current_stream:
                    active_stream.wait_stream(current_stream)
                host["token_count"][layer_start:layer_end].copy_(
                    dev["token_count"][layer_start:layer_end],
                    non_blocking=True,
                )
                if "activation_count" in dev:
                    host["activation_count"][layer_start:layer_end].copy_(
                        dev["activation_count"][layer_start:layer_end],
                        non_blocking=True,
                    )
                    if "score_sum" in dev:
                        host["score_sum"][layer_start:layer_end].copy_(
                            dev["score_sum"][layer_start:layer_end],
                            non_blocking=True,
                        )
                    if has_status:
                        host["expert_status"][layer_start:layer_end].copy_(
                            dev["expert_status"][layer_start:layer_end],
                            non_blocking=True,
                        )
                    if "execution_activation_count" in dev:
                        host["execution_activation_count"][layer_start:layer_end].copy_(
                            dev["execution_activation_count"][layer_start:layer_end],
                            non_blocking=True,
                        )
                elif "score_sum" in dev:
                    host["score_sum"][layer_start:layer_end].copy_(
                        dev["score_sum"][layer_start:layer_end],
                        non_blocking=True,
                    )
                else:
                    host["selected_experts"][layer_start:layer_end].copy_(
                        dev["selected_experts"][layer_start:layer_end],
                        non_blocking=True,
                    )
                    host["routing_weights"][layer_start:layer_end].copy_(
                        dev["routing_weights"][layer_start:layer_end],
                        non_blocking=True,
                    )
                    if "route_status" in dev:
                        host["route_status"][layer_start:layer_end].copy_(
                            dev["route_status"][layer_start:layer_end],
                            non_blocking=True,
                        )
                event.record(active_stream)
        else:
            host["token_count"][layer_start:layer_end].copy_(dev["token_count"][layer_start:layer_end])
            if "activation_count" in dev:
                host["activation_count"][layer_start:layer_end].copy_(dev["activation_count"][layer_start:layer_end])
                if "score_sum" in dev:
                    host["score_sum"][layer_start:layer_end].copy_(dev["score_sum"][layer_start:layer_end])
                if has_status:
                    host["expert_status"][layer_start:layer_end].copy_(dev["expert_status"][layer_start:layer_end])
                if "execution_activation_count" in dev:
                    host["execution_activation_count"][layer_start:layer_end].copy_(
                        dev["execution_activation_count"][layer_start:layer_end]
                    )
            elif "score_sum" in dev:
                host["score_sum"][layer_start:layer_end].copy_(dev["score_sum"][layer_start:layer_end])
            else:
                host["selected_experts"][layer_start:layer_end].copy_(dev["selected_experts"][layer_start:layer_end])
                host["routing_weights"][layer_start:layer_end].copy_(dev["routing_weights"][layer_start:layer_end])
                if "route_status" in dev:
                    host["route_status"][layer_start:layer_end].copy_(dev["route_status"][layer_start:layer_end])
            event = _ImmediateEvent()

        return RuntimeMetaOffloadHandle(
            step_id=self.active_step_id,
            mode=self.active_mode,
            event=event,
            token_capacity=key[1],
            logical_token_count=self.active_logical_token_count,
            execution_token_count=self.active_execution_token_count,
            buffer_bytes=buffer_bytes,
            host_buffer_slot=host_slot,
            layer_start_idx=layer_start,
            layer_end_idx=layer_end,
            metadata_format=metadata_format,
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
        host = self.host_buffer_pools[key][int(handle.host_buffer_slot)]
        token_counts = host["token_count"]
        out: dict[int, LayerRuntimeMetaCPU] = {}

        layer_start = max(0, int(getattr(handle, "layer_start_idx", 0)))
        layer_end_value = getattr(handle, "layer_end_idx", None)
        layer_end = self.num_layers if layer_end_value is None else min(self.num_layers, int(layer_end_value))
        layer_start = min(layer_start, self.num_layers)
        layer_end = max(layer_start, layer_end)
        fmt = getattr(handle, "metadata_format", "raw")
        histogram_token_counts = (
            token_counts.numpy() if fmt == "histogram" else None
        )
        histogram_counts = (
            host["activation_count"].numpy() if fmt == "histogram" else None
        )
        histogram_scores = (
            host["score_sum"].numpy()
            if fmt == "histogram" and "score_sum" in host
            else None
        )

        for layer_idx in range(layer_start, layer_end):
            token_count = int(
                histogram_token_counts[layer_idx]
                if histogram_token_counts is not None
                else token_counts[layer_idx].item()
            )
            token_count = min(token_count, int(handle.logical_token_count))
            if token_count <= 0:
                continue
            if fmt == "histogram":
                # This is the production draft-segment format.  The buffers
                # are already on CPU, and NumPy extracts these tiny sparse
                # rows with substantially less dispatcher overhead than a
                # per-layer chain of torch.nonzero/index_select/to calls.
                counts_array = histogram_counts[layer_idx]
                nonzero_array = np.flatnonzero(counts_array)
                if nonzero_array.size <= 0:
                    continue
                nonzero = torch.from_numpy(nonzero_array)
                counts = torch.from_numpy(
                    counts_array[nonzero_array].astype(np.int64, copy=False)
                )
                if histogram_scores is not None:
                    score_array = histogram_scores[layer_idx]
                    score_sum = torch.from_numpy(
                        score_array[nonzero_array].astype(np.float32, copy=False)
                    )
                else:
                    score_sum = counts.to(dtype=torch.float32)
                out[layer_idx] = LayerRuntimeMetaCPU(
                    step_id=handle.step_id,
                    mode=handle.mode,
                    layer_idx=layer_idx,
                    token_count=token_count,
                    aggregated_expert_ids=nonzero,
                    aggregated_score_sum=score_sum,
                    aggregated_activation_count=counts,
                )
                continue
            if fmt == "histogram_kt_hybrid":
                counts_row = host["activation_count"][layer_idx]
                nonzero = torch.nonzero(counts_row, as_tuple=False).reshape(-1)
                if nonzero.numel() <= 0:
                    continue
                counts = counts_row.index_select(0, nonzero).to(dtype=torch.int64, device=torch.device("cpu"))
                if "score_sum" in host:
                    score_sum = host["score_sum"][layer_idx].index_select(0, nonzero).to(
                        dtype=torch.float32,
                        device=torch.device("cpu"),
                    )
                else:
                    score_sum = counts.to(dtype=torch.float32, device=torch.device("cpu"))
                meta = LayerRuntimeMetaCPU(
                    step_id=handle.step_id,
                    mode=handle.mode,
                    layer_idx=layer_idx,
                    token_count=token_count,
                    aggregated_expert_ids=nonzero.to(dtype=torch.int64, device=torch.device("cpu")),
                    aggregated_score_sum=score_sum,
                    aggregated_activation_count=counts,
                )
                if "expert_status" in host:
                    status_row = host["expert_status"][layer_idx]
                    act_row = host["activation_count"][layer_idx]
                    is_real_active = act_row > 0
                    meta.expert_status = status_row.clone()
                    meta.active_count = float(is_real_active.sum().item())
                    meta.miss_count = float(((status_row == 2) & is_real_active).sum().item())
                    if "execution_activation_count" in host:
                        meta.execution_activation_count = host[
                            "execution_activation_count"
                        ][layer_idx].clone()
                out[layer_idx] = meta
                continue
            if fmt == "score_sum":
                score_row = host["score_sum"][layer_idx]
                nonzero = torch.nonzero(score_row, as_tuple=False).reshape(-1)
                if nonzero.numel() <= 0:
                    continue
                out[layer_idx] = LayerRuntimeMetaCPU(
                    step_id=handle.step_id,
                    mode=handle.mode,
                    layer_idx=layer_idx,
                    token_count=token_count,
                    aggregated_expert_ids=nonzero.to(dtype=torch.int64, device=torch.device("cpu")),
                    aggregated_score_sum=score_row.index_select(0, nonzero).to(
                        dtype=torch.float32,
                        device=torch.device("cpu"),
                    ),
                    aggregated_activation_count=None,
                )
                continue
            selected_experts = host["selected_experts"][layer_idx, :token_count]
            routing_weights = host["routing_weights"][layer_idx, :token_count]
            execution_token_count = int(
                getattr(handle, "execution_token_count", 0)
            )
            if execution_token_count <= 0:
                execution_token_count = token_count
            execution_token_count = min(
                execution_token_count, int(handle.token_capacity)
            )
            execution_selected_experts = host["selected_experts"][
                layer_idx, :execution_token_count
            ]
            execution_routing_weights = host["routing_weights"][
                layer_idx, :execution_token_count
            ]
            aggregated = _aggregate_layer_runtime_meta_cpu(selected_experts, routing_weights)
            route_status = (
                host["route_status"][layer_idx, :token_count].clone()
                if "route_status" in host
                else None
            )
            keep_route_rows = bool(
                self.perfect_match_trace_enabled
                or self.transfer_aware_profile_enabled
            )
            meta = LayerRuntimeMetaCPU(
                step_id=handle.step_id,
                mode=handle.mode,
                layer_idx=layer_idx,
                token_count=token_count,
                aggregated_expert_ids=aggregated[0] if aggregated is not None else None,
                aggregated_score_sum=aggregated[1] if aggregated is not None else None,
                aggregated_activation_count=aggregated[2] if aggregated is not None else None,
                selected_experts=(
                    selected_experts.clone()
                    if keep_route_rows or aggregated is None
                    else None
                ),
                routing_weights=(
                    routing_weights.clone()
                    if keep_route_rows or aggregated is None
                    else None
                ),
                route_status=route_status,
                execution_selected_experts=(
                    execution_selected_experts.clone()
                    if self.transfer_aware_profile_enabled
                    else None
                ),
                execution_routing_weights=(
                    execution_routing_weights.clone()
                    if self.transfer_aware_profile_enabled
                    else None
                ),
                execution_route_status=(
                    host["route_status"][
                        layer_idx, :execution_token_count
                    ].clone()
                    if self.transfer_aware_profile_enabled
                    and "route_status" in host
                    else None
                ),
            )
            if route_status is not None:
                active_routes = route_status > 0
                meta.active_count = float(active_routes.sum().item())
                meta.miss_count = float((route_status == 2).sum().item())
            out[layer_idx] = meta
        return out

    def reset(self) -> None:
        self.active_key = None
        self.active_step_id = -1
        self.active_mode = "idle"
        self.active_logical_token_count = 0
        self.active_execution_token_count = 0

    def _buffer_bytes(self, key: tuple[str, int], layer_start: int = 0, layer_end: int | None = None) -> int:
        host = self.host_buffers[key]
        end = self.num_layers if layer_end is None else int(layer_end)
        layer_count = max(0, min(self.num_layers, end) - max(0, int(layer_start)))
        total = 0
        total += int(layer_count * host["token_count"].element_size())
        if "activation_count" in host:
            total += int(layer_count * host["activation_count"].size(1) * host["activation_count"].element_size())
            if "score_sum" in host:
                total += int(layer_count * host["score_sum"].size(1) * host["score_sum"].element_size())
            if "expert_status" in host:
                total += int(layer_count * host["expert_status"].size(1) * host["expert_status"].element_size())
            if "execution_activation_count" in host:
                total += int(
                    layer_count
                    * host["execution_activation_count"].size(1)
                    * host["execution_activation_count"].element_size()
                )
        elif "score_sum" in host:
            total += int(layer_count * host["score_sum"].size(1) * host["score_sum"].element_size())
        else:
            total += int(layer_count * host["selected_experts"].size(1) * host["selected_experts"].size(2) * host["selected_experts"].element_size())
            total += int(layer_count * host["routing_weights"].size(1) * host["routing_weights"].size(2) * host["routing_weights"].element_size())
            if "route_status" in host:
                total += int(layer_count * host["route_status"].size(1) * host["route_status"].size(2) * host["route_status"].element_size())
        return total
