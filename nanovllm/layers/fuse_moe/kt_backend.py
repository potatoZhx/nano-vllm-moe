"""Optional kt-kernel CPU MoE backend.

This backend delegates CPU expert computation to kt-kernel's
AMX/AVX2/AVX512-accelerated kernels.  It is opt-in via
``cpu_expert_backend="kt_kernel"`` and requires ``kt_kernel`` to be
installed.

.. note::

   On Ice Lake Xeons (AVX512_BF16, no AMX), kt_kernel's AMX path
   will crash with SIGILL.  This backend automatically detects this
   and forces the AVX2 fallback, but AVX2 is typically slower than
   PyTorch's MKL AVX512 backend for small-M matmuls.

   On Sapphire Rapids (AMX-capable), the AMX_BF16 path works but
   has high per-call overhead (~80ms).  It may only be beneficial
   for large prefill (1000+ tokens).

"""
from __future__ import annotations

import os
from time import perf_counter
from typing import Callable

import torch

from nanovllm.layers.fuse_moe.cpu_backend import CpuMoeResult, get_cpu_expert_weights


def _detect_amx_hardware() -> bool:
    """Check whether the CPU actually supports AMX instructions."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags") and "amx_bf16" in line:
                    return True
    except Exception:
        pass
    return False


def _detect_available_cores() -> int:
    """Return number of CPU cores available to this process."""
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        n = os.cpu_count()
        return n if n else 4


def _detect_amx_supported() -> bool:
    """Check whether the CPU supports AMX instructions."""
    try:
        import kt_kernel.utils.amx as amx_mod
        return bool(getattr(amx_mod, "_HAS_BF16_SUPPORT", False))
    except Exception:
        return False


def _configure_kt_backend() -> str:
    """Return the effective kt-kernel method name.

    On Ice Lake (AVX512_BF16 but no AMX), the AMX_BF16 path crashes
    with SIGILL.  Force the AVX2 fallback in that case.

    Detection is done by reading /proc/cpuinfo for the ``amx_bf16``
    flag (not by trusting kt-kernel compile-time flags which may not
    reflect actual hardware).
    """
    try:
        import kt_kernel.utils.amx as amx_mod
        if amx_mod._HAS_BF16_SUPPORT and not _detect_amx_hardware():
            # AVX512_BF16 available but AMX hardware missing — force AVX2
            amx_mod._HAS_BF16_SUPPORT = False
            return "BF16_AVX2"
        return "BF16"
    except Exception:
        return "BF16"


class KtKernelCpuMoeBackend:
    """CPU MoE backend powered by kt-kernel.

    Parameters
    ----------
    layer_idx:
        MoE layer index (used for weight loading).
    cpu_expert_pool:
        Per-expert weight dict (unused when kt-kernel loads from disk).
    max_routes:
        Upper bound on CPU route count (unused, kt-kernel handles routing).
    moe_intermediate_size:
        Expert intermediate dimension (I).
    hidden_size:
        Model hidden dimension (H).
    num_experts:
        Total expert count per layer.
    num_experts_per_tok:
        Top-K value.
    gpu_expert_mask:
        Boolean mask: ``True`` = expert is GPU-cached (skip), ``False`` = CPU.
    weight_path:
        Path to safetensors directory.
    kt_method:
        kt-kernel method name (``"BF16"``, ``"AMXINT4"``, etc.).
    kt_num_threads:
        CPU infer thread count (0 = auto-detect).
    kt_threadpool_count:
        Number of NUMA sub-pools.
    kt_chunked_prefill_size:
        Prefill chunk size for kt-kernel.
    """

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
        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)

        effective_method = _configure_kt_backend()

        if kt_num_threads <= 0:
            kt_num_threads = _detect_available_cores()
        # kt-kernel's AMX has diminishing returns beyond 16 threads for
        # the M dimensions typical in MoE (AMX tile contention).
        kt_num_threads = min(kt_num_threads, 16)

        try:
            import kt_kernel
            self._kt = kt_kernel
        except ImportError:
            raise RuntimeError(
                "kt_kernel is not installed.  Install it with: pip install kt-kernel"
            )

        self._gpu_mask = gpu_expert_mask.detach().to("cpu", dtype=torch.bool).pin_memory()

        self._wrapper = self._kt.KTMoEWrapper(
            layer_idx=layer_idx,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            gpu_experts_mask=self._gpu_mask,
            cpuinfer_threads=kt_num_threads,
            threadpool_count=kt_threadpool_count,
            weight_path=weight_path,
            chunked_prefill_size=kt_chunked_prefill_size,
            cpu_save=False,
            max_deferred_experts_per_token=0,
            method=kt_method,
        )

        physical_to_logical = torch.arange(num_experts, dtype=torch.int64)
        self._wrapper.load_weights(physical_to_logical)

    def submit(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        cuda_stream: int,
    ) -> None:
        """Async submit CPU expert compute.

        Call ``sync()`` to retrieve the result.  Use this together with
        the GPU parallel path to overlap CPU and GPU work.
        """
        self._wrapper.submit_forward(
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
        """Wait for async submission and return CPU expert partial."""
        return self._wrapper.sync_forward(
            hidden_states.contiguous(),
            cuda_stream,
        )

    def update_gpu_expert_mask(self, mask: torch.Tensor) -> None:
        """Update which experts are GPU-cached.

        Called when the GPU cache state changes (e.g. after prefetch).
        """
        mask_cpu = mask.detach().to("cpu", dtype=torch.bool)
        if not torch.equal(mask_cpu, self._gpu_mask):
            self._gpu_mask.copy_(mask_cpu)
            # kt-kernel's wrapper reads gpu_experts_mask by reference
            # so updating the pinned tensor is sufficient.

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
        """Compute CPU expert outputs via kt-kernel.

        kt-kernel's forward() handles all selected experts
        internally, using ``gpu_experts_mask`` to skip cached ones.
        The result is the partial sum of CPU expert contributions.

        When ``selected_experts`` / ``routing_weights`` are provided,
        they are used as the full routing table (shape:
        ``[num_tokens, top_k]``).  Otherwise ``cpu_indices`` /
        ``flat_weights`` are used for backward compatibility.
        """
        prep_t0 = perf_counter()

        if selected_experts is not None and routing_weights is not None:
            topk_ids = selected_experts.contiguous().to(torch.int64)
            topk_w = routing_weights.contiguous().to(hidden_states.dtype)
            num_tokens = int(selected_experts.size(0))
            cpu_route_count = int(cpu_indices.numel())
            token_indices = torch.div(cpu_indices, top_k, rounding_mode="floor")
        else:
            num_tokens = int(hidden_states.size(0))
            cpu_route_count = int(cpu_indices.numel())
            token_indices = torch.div(cpu_indices, top_k, rounding_mode="floor")
            topk_ids = cpu_indices.reshape(num_tokens, top_k).to(
                device=hidden_states.device
            )
            topk_w = flat_weights.reshape(num_tokens, top_k).to(
                device=hidden_states.device, dtype=hidden_states.dtype
            )

        prep_ms = (perf_counter() - prep_t0) * 1000.0

        compute_t0 = perf_counter()
        stream = torch.cuda.current_stream(hidden_states.device).cuda_stream
        cpu_partial = self._wrapper.forward(
            hidden_states.contiguous(),
            topk_ids,
            topk_w,
            stream,
        )
        compute_ms = (perf_counter() - compute_t0) * 1000.0

        return CpuMoeResult(
            token_indices=token_indices,
            outputs_cpu=cpu_partial,
            prep_ms=prep_ms,
            compute_ms=compute_ms,
        )
