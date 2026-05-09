"""Optional kt-kernel CPU MoE backend — shared-wrapper architecture.

Uses a single NativeMoEWrapper for all layers to avoid creating 48
AMX contexts (which causes a segfault).  Before each layer's forward
call, weights are reloaded for that layer.
"""
from __future__ import annotations

import os
from time import perf_counter
from typing import Callable

import torch

from nanovllm.layers.fuse_moe.cpu_backend import CpuMoeResult

# ---------------------------------------------------------------------------
# AMX detection — does NOT import kt_kernel (avoids early C++ init conflicts)
# ---------------------------------------------------------------------------

def _detect_amx_hardware() -> bool:
    """Check /proc/cpuinfo for amx_bf16 flag."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags") and "amx_bf16" in line:
                    return True
    except Exception:
        pass
    return False

_HAS_AMX_HARDWARE = _detect_amx_hardware()

# ---------------------------------------------------------------------------
# Module-level shared singleton
# ---------------------------------------------------------------------------

_shared_kt_wrapper = None      # NativeMoEWrapper (one for all layers)
_shared_wrapper_layer = -1     # currently loaded layer_idx
_shared_wrapper_lock = None    # threading.Lock (lazy)

# Cached params that don't change per layer
_shared_num_experts = 0
_shared_top_k = 0
_shared_hidden_size = 0
_shared_intermediate_size = 0
_shared_gpu_mask: torch.Tensor | None = None
_shared_physical_to_logical: torch.Tensor | None = None
_zombie_moes: list = []   # keep old C++ MOE objects alive to avoid destructor crash


def _get_lock():
    global _shared_wrapper_lock
    if _shared_wrapper_lock is None:
        import threading
        _shared_wrapper_lock = threading.Lock()
    return _shared_wrapper_lock


def _detect_amx_hardware() -> bool:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags") and "amx_bf16" in line:
                    return True
    except Exception:
        pass
    return False


def _detect_available_cores() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        n = os.cpu_count()
        return n if n else 4


def _configure_kt_backend() -> str:
    try:
        import kt_kernel.utils.amx as amx_mod
        if amx_mod._HAS_BF16_SUPPORT and not _detect_amx_hardware():
            amx_mod._HAS_BF16_SUPPORT = False
            return "BF16_AVX2"
        return "BF16"
    except Exception:
        return "BF16"


def _init_shared_wrapper(
    *,
    num_experts: int,
    num_experts_per_tok: int,
    hidden_size: int,
    moe_intermediate_size: int,
    gpu_expert_mask: torch.Tensor,
    weight_path: str,
    kt_num_threads: int,
    kt_threadpool_count: int,
    kt_chunked_prefill_size: int,
) -> None:
    """Idempotent: create the shared NativeMoEWrapper once."""
    global _shared_kt_wrapper, _shared_num_experts, _shared_top_k
    global _shared_hidden_size, _shared_intermediate_size
    global _shared_gpu_mask, _shared_physical_to_logical

    if _shared_kt_wrapper is not None:
        return

    # Patch kt-kernel flags based on actual hardware (must happen before
    # the first NativeMoEWrapper is created but AFTER all other imports).
    import kt_kernel
    import kt_kernel.utils.amx as amx_mod
    if not _HAS_AMX_HARDWARE and getattr(amx_mod, "_HAS_BF16_SUPPORT", False):
        amx_mod._HAS_BF16_SUPPORT = False  # force AVX2 on Ice Lake
    _shared_num_experts = int(num_experts)
    _shared_top_k = int(num_experts_per_tok)
    _shared_hidden_size = int(hidden_size)
    _shared_intermediate_size = int(moe_intermediate_size)

    _shared_gpu_mask = gpu_expert_mask.detach().to("cpu", dtype=torch.bool)
    _shared_physical_to_logical = torch.arange(num_experts, dtype=torch.int64)

    wrapper = kt_kernel.KTMoEWrapper(
        layer_idx=0,  # will be overridden per layer
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        hidden_size=hidden_size,
        moe_intermediate_size=moe_intermediate_size,
        gpu_experts_mask=_shared_gpu_mask,
        cpuinfer_threads=kt_num_threads,
        threadpool_count=kt_threadpool_count,
        weight_path=weight_path,
        chunked_prefill_size=kt_chunked_prefill_size,
        cpu_save=False,
        max_deferred_experts_per_token=0,
        method="BF16",
    )
    # Global assignment must be in the same scope as 'global' declaration
    _shared_kt_wrapper = wrapper
    # Defer load_weights to first forward call (avoids init-time conflicts)


def _ensure_layer_weights(layer_idx: int) -> None:
    """Reload weights if the shared wrapper is on a different layer."""
    global _shared_wrapper_layer
    if _shared_wrapper_layer == layer_idx:
        return
    with _get_lock():
        if _shared_wrapper_layer == layer_idx:
            return
        wrapper = _shared_kt_wrapper
        # Zombie the old C++ moe: its destructor crashes on Ice Lake (no AMX).
        if hasattr(wrapper, "moe") and wrapper.moe is not None:
            _zombie_moes.append(wrapper.moe)
            wrapper.moe = None
        if hasattr(wrapper, "layer_idx"):
            wrapper.layer_idx = layer_idx
        wrapper.load_weights(_shared_physical_to_logical)
        _shared_wrapper_layer = layer_idx


# ---------------------------------------------------------------------------
# Per-layer lightweight backend
# ---------------------------------------------------------------------------

class KtKernelCpuMoeBackend:
    """Per-layer handle that delegates to the shared kt-kernel wrapper."""

    def __init__(
        self,
        *,
        layer_idx: int,
        cpu_expert_pool: dict[int, dict[str, object]] | None = None,
        max_routes: int = 8192,
        moe_intermediate_size: int,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        gpu_expert_mask: torch.Tensor,
        weight_path: str,
        kt_method: str = "BF16",
        kt_num_threads: int = 0,
        kt_threadpool_count: int = 1,
        kt_chunked_prefill_size: int = 4096,
    ) -> None:
        self.layer_idx = int(layer_idx)

        if kt_num_threads <= 0:
            kt_num_threads = _detect_available_cores()
        kt_num_threads = min(kt_num_threads, 16)

        _init_shared_wrapper(
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            gpu_expert_mask=gpu_expert_mask,
            weight_path=weight_path,
            kt_num_threads=kt_num_threads,
            kt_threadpool_count=kt_threadpool_count,
            kt_chunked_prefill_size=kt_chunked_prefill_size,
        )

    def submit(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        cuda_stream: int,
    ) -> None:
        _ensure_layer_weights(self.layer_idx)
        _shared_kt_wrapper.submit_forward(
            hidden_states.contiguous(),
            selected_experts.contiguous().to(torch.int64),
            routing_weights.contiguous().to(hidden_states.dtype),
            cuda_stream,
        )

    def sync(
        self,
        hidden_states: torch.Tensor,
        cuda_stream: int,
    ) -> torch.Tensor:
        return _shared_kt_wrapper.sync_forward(
            hidden_states.contiguous(),
            cuda_stream,
        )

    @torch.no_grad()
    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        flat_weights: torch.Tensor,
        top_k: int,
        cpu_indices: torch.Tensor,
        cpu_task_expert_ids: torch.Tensor,
        cpu_task_offsets: torch.Tensor,
        act_fn: Callable[[torch.Tensor], torch.Tensor],
        parallel_mode: str = "serial",
        num_threads: int = 4,
        cpu_task_expert_ids_host: list[int] | None = None,
        cpu_task_offsets_host: list[int] | None = None,
        selected_experts: torch.Tensor | None = None,
        routing_weights: torch.Tensor | None = None,
    ) -> CpuMoeResult:
        prep_t0 = perf_counter()

        if selected_experts is not None and routing_weights is not None:
            topk_ids = selected_experts.contiguous().to(torch.int64)
            topk_w = routing_weights.contiguous().to(hidden_states.dtype)
        else:
            num_tokens = int(hidden_states.size(0))
            topk_ids = cpu_indices.reshape(num_tokens, top_k).to(
                device=hidden_states.device
            )
            topk_w = flat_weights.reshape(num_tokens, top_k).to(
                device=hidden_states.device, dtype=hidden_states.dtype
            )

        prep_ms = (perf_counter() - prep_t0) * 1000.0

        _ensure_layer_weights(self.layer_idx)

        compute_t0 = perf_counter()
        stream = torch.cuda.current_stream(hidden_states.device).cuda_stream
        cpu_partial = _shared_kt_wrapper.forward(
            hidden_states.contiguous(),
            topk_ids,
            topk_w,
            stream,
        )
        compute_ms = (perf_counter() - compute_t0) * 1000.0

        # kt_kernel returns per-token accumulated output, not per-route.
        # Return empty token_indices to signal the caller to add directly.
        return CpuMoeResult(
            token_indices=torch.empty(0, dtype=torch.int64, device="cpu"),
            outputs_cpu=cpu_partial,
            prep_ms=prep_ms,
            compute_ms=compute_ms,
        )
