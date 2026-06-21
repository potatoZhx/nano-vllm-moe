from __future__ import annotations

import ctypes
import gc
import os
from time import perf_counter
from typing import Callable

import torch

from nanovllm.layers.fuse_moe.cpu_backend import CpuMoeResult


def _trim_cpu_allocator() -> None:
    gc.collect()
    try:
        malloc_trim = ctypes.CDLL(None).malloc_trim
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        malloc_trim(0)
    except (AttributeError, OSError):
        pass


def _numa_physical_core_capacities() -> dict[int, int]:
    try:
        allowed_cpus = os.sched_getaffinity(0)
    except Exception:
        allowed_cpus = set(range(os.cpu_count() or 4))

    cores_by_numa: dict[int, set[tuple[int, int]]] = {}
    try:
        for cpu in allowed_cpus:
            cpu_dir = f"/sys/devices/system/cpu/cpu{cpu}"
            with open(f"{cpu_dir}/topology/physical_package_id", encoding="utf-8") as stream:
                package_id = int(stream.read().strip())
            with open(f"{cpu_dir}/topology/core_id", encoding="utf-8") as stream:
                core_id = int(stream.read().strip())
            node_ids = [
                int(name[4:])
                for name in os.listdir(cpu_dir)
                if name.startswith("node") and name[4:].isdigit()
            ]
            numa_node = node_ids[0] if node_ids else package_id
            cores_by_numa.setdefault(numa_node, set()).add((package_id, core_id))
    except (OSError, ValueError):
        return {0: len(allowed_cpus)}

    return {
        numa_node: len(core_ids)
        for numa_node, core_ids in cores_by_numa.items()
        if core_ids
    }


def _split_threads(total: int, groups: int) -> list[int]:
    total = int(total)
    groups = int(groups)
    if total < 1:
        raise ValueError("kt_num_threads must be positive")
    if groups < 1:
        raise ValueError("kt_threadpool_count must be positive")
    if groups > total:
        raise ValueError("kt_threadpool_count cannot exceed kt_num_threads")
    base = total // groups
    remainder = total % groups
    return [base + (1 if index < remainder else 0) for index in range(groups)]


def _resolve_runtime_layout(
    *,
    kt_num_threads: int,
    kt_threadpool_count: int,
    kt_numa_nodes: list[int] | None,
    core_capacities: dict[int, int] | None = None,
) -> tuple[int, list[int], list[int]]:
    pools = int(kt_threadpool_count)
    if pools < 1:
        raise ValueError("kt_threadpool_count must be positive")
    capacities = dict(
        _numa_physical_core_capacities()
        if core_capacities is None
        else core_capacities
    )
    available_nodes = sorted(node for node, count in capacities.items() if int(count) > 0)
    if kt_numa_nodes is None:
        if len(available_nodes) < pools:
            raise ValueError(
                f"kt_threadpool_count={pools} exceeds available NUMA nodes {available_nodes}"
            )
        numa_nodes = available_nodes[:pools]
    else:
        numa_nodes = [int(node) for node in kt_numa_nodes]
        if len(numa_nodes) != pools:
            raise ValueError("kt_numa_nodes length must equal kt_threadpool_count")

    selected_capacities = []
    for node in numa_nodes:
        capacity = int(capacities.get(node, 0))
        if capacity < 1:
            raise ValueError(
                f"NUMA node {node} has no physical CPU cores in the current process affinity"
            )
        selected_capacities.append(capacity)

    total = int(kt_num_threads)
    if total <= 0:
        thread_counts = selected_capacities
        total = sum(thread_counts)
    else:
        thread_counts = _split_threads(total, pools)
        for node, requested, capacity in zip(
            numa_nodes,
            thread_counts,
            selected_capacities,
            strict=True,
        ):
            if requested > capacity:
                raise ValueError(
                    f"kt_direct requests {requested} cores on NUMA node {node}, "
                    f"but only {capacity} physical cores are available in the current affinity"
                )
    return total, numa_nodes, thread_counts


def _cpu_has_flag(flag: str) -> bool:
    normalized = flag.replace("-", "_")
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if not line.startswith("flags"):
                    continue
                flags = {item.replace("-", "_") for item in line.split(":", 1)[1].split()}
                return normalized in flags
    except OSError:
        return False
    return False


def _select_kt_bf16_moe_class(kt_moe, backend: str):
    normalized = str(backend).strip().lower()
    amx_cls = getattr(kt_moe, "AMXBF16_MOE", None)
    avx2_cls = getattr(kt_moe, "AVX2BF16_MOE", None)

    if normalized in {"", "auto", "bf16"}:
        if _cpu_has_flag("amx_bf16") and amx_cls is not None:
            return amx_cls, "amx_bf16"
        if _cpu_has_flag("avx2") and avx2_cls is not None:
            return avx2_cls, "avx2_bf16"
        raise RuntimeError(
            "No supported KTransformers BF16 MoE backend is available for this CPU"
        )

    if normalized in {"amx", "amx_bf16"}:
        if amx_cls is None:
            raise RuntimeError("AMXBF16_MOE is not available in kt_kernel_ext")
        if not _cpu_has_flag("amx_bf16"):
            raise RuntimeError("kt_direct_backend='amx_bf16' requires CPU AMX BF16 support")
        return amx_cls, "amx_bf16"

    if normalized in {"avx2", "avx2_bf16", "bf16_avx2"}:
        if avx2_cls is None:
            raise RuntimeError("AVX2BF16_MOE is not available in kt_kernel_ext")
        return avx2_cls, "avx2_bf16"

    raise ValueError(f"Unsupported kt_direct_backend={backend!r}")


class KtDirectGlobalRuntime:
    _instance: "KtDirectGlobalRuntime | None" = None

    @classmethod
    def get(
        cls,
        *,
        kt_num_threads: int,
        kt_threadpool_count: int,
        kt_numa_nodes: list[int] | None = None,
    ) -> "KtDirectGlobalRuntime":
        pools = int(kt_threadpool_count)
        threads, resolved_numa_nodes, _ = _resolve_runtime_layout(
            kt_num_threads=kt_num_threads,
            kt_threadpool_count=pools,
            kt_numa_nodes=kt_numa_nodes,
        )
        numa_nodes_tuple = tuple(resolved_numa_nodes)
        if cls._instance is None:
            cls._instance = cls(
                kt_num_threads=threads,
                kt_threadpool_count=pools,
                kt_numa_nodes=resolved_numa_nodes,
            )
        else:
            current = cls._instance
            requested = (threads, pools, numa_nodes_tuple)
            configured = (
                current.kt_num_threads,
                current.kt_threadpool_count,
                current.kt_numa_nodes,
            )
            if requested != configured:
                raise RuntimeError(
                    "kt_direct CPUInfer is already initialized with "
                    f"threads/pools/numa={configured}, requested={requested}"
                )
        return cls._instance

    def __init__(
        self,
        *,
        kt_num_threads: int,
        kt_threadpool_count: int,
        kt_numa_nodes: list[int] | None,
    ) -> None:
        _trim_cpu_allocator()
        try:
            from kt_kernel import kt_kernel_ext
        except Exception as exc:
            raise RuntimeError(
                "cpu_expert_backend='kt_direct' requires kt-kernel and kt_kernel_ext"
            ) from exc

        threads = int(kt_num_threads)
        pools = int(kt_threadpool_count)
        threads, resolved_numa_nodes, thread_counts = _resolve_runtime_layout(
            kt_num_threads=threads,
            kt_threadpool_count=pools,
            kt_numa_nodes=kt_numa_nodes,
        )
        numa_nodes_tuple = tuple(resolved_numa_nodes)

        worker_config = kt_kernel_ext.WorkerPoolConfig()
        worker_config.subpool_count = pools
        worker_config.subpool_numa_map = list(numa_nodes_tuple)
        if len(worker_config.subpool_numa_map) != pools:
            raise ValueError("kt_numa_nodes length must equal kt_threadpool_count")
        worker_config.subpool_thread_count = thread_counts

        self.kt_kernel_ext = kt_kernel_ext
        self.kt_moe = kt_kernel_ext.moe
        self.cpu_infer = kt_kernel_ext.CPUInfer(worker_config)
        self.kt_num_threads = threads
        self.kt_threadpool_count = pools
        self.kt_numa_nodes = numa_nodes_tuple
        self._moe_refs: list[object] = []

    def retain_moe(self, moe: object) -> None:
        self._moe_refs.append(moe)


def _get_raw_gate_up_down(
    params: dict[str, object],
    *,
    expert_idx: int,
    strict_dtype: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    packed = params.get("packed")
    if "gate_up" in params and "down" in params:
        gate_up = params["gate_up"]
        down = params["down"]
    elif packed is not None:
        gate_up = packed.gate_up
        down = packed.down
    else:
        raise RuntimeError(f"kt_direct expert {expert_idx} is missing gate_up/down weights")

    if not isinstance(gate_up, torch.Tensor) or not isinstance(down, torch.Tensor):
        raise TypeError(f"kt_direct expert {expert_idx} weights must be torch.Tensor instances")
    if strict_dtype and (gate_up.dtype != torch.bfloat16 or down.dtype != torch.bfloat16):
        raise RuntimeError(
            f"kt_direct expert {expert_idx} requires BF16 weights, "
            f"got gate_up={gate_up.dtype}, down={down.dtype}"
        )
    if gate_up.device.type != "cpu":
        gate_up = gate_up.to(device="cpu")
    if down.device.type != "cpu":
        down = down.to(device="cpu")
    if gate_up.dtype != torch.bfloat16:
        gate_up = gate_up.to(dtype=torch.bfloat16)
    if down.dtype != torch.bfloat16:
        down = down.to(dtype=torch.bfloat16)
    return gate_up, down


def _build_bf16_weight_ptrs(
    *,
    cpu_expert_pool: dict[int, dict[str, object]],
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    threadpool_count: int,
    strict_dtype: bool,
) -> tuple[list[list[int]], list[list[int]], list[list[int]], list[torch.Tensor]]:
    gate_ptrs_one_pool: list[int] = []
    up_ptrs_one_pool: list[int] = []
    down_ptrs_one_pool: list[int] = []
    refs: list[torch.Tensor] = []
    expected_gate_up_shape = (int(intermediate_size) * 2, int(hidden_size))
    expected_down_shape = (int(hidden_size), int(intermediate_size))

    for expert_idx in range(int(num_experts)):
        params = cpu_expert_pool.get(expert_idx)
        if params is None:
            raise RuntimeError(
                f"kt_direct requires all experts in cpu_expert_pool; missing expert {expert_idx}"
            )
        gate_up, down = _get_raw_gate_up_down(
            params,
            expert_idx=expert_idx,
            strict_dtype=strict_dtype,
        )
        if tuple(gate_up.shape) != expected_gate_up_shape:
            raise RuntimeError(
                f"kt_direct expert {expert_idx} gate_up shape mismatch: "
                f"got {tuple(gate_up.shape)}, expected {expected_gate_up_shape}"
            )
        if tuple(down.shape) != expected_down_shape:
            raise RuntimeError(
                f"kt_direct expert {expert_idx} down shape mismatch: "
                f"got {tuple(down.shape)}, expected {expected_down_shape}"
            )

        gate_up = gate_up if gate_up.is_contiguous() else gate_up.contiguous()
        down = down if down.is_contiguous() else down.contiguous()
        gate = gate_up[:intermediate_size]
        up = gate_up[intermediate_size:]
        if not gate.is_contiguous():
            gate = gate.contiguous()
        if not up.is_contiguous():
            up = up.contiguous()

        refs.extend((gate, up, down))
        gate_ptrs_one_pool.append(gate.data_ptr())
        up_ptrs_one_pool.append(up.data_ptr())
        down_ptrs_one_pool.append(down.data_ptr())

    gate_ptrs = [list(gate_ptrs_one_pool) for _ in range(int(threadpool_count))]
    up_ptrs = [list(up_ptrs_one_pool) for _ in range(int(threadpool_count))]
    down_ptrs = [list(down_ptrs_one_pool) for _ in range(int(threadpool_count))]
    return gate_ptrs, up_ptrs, down_ptrs, refs


class KtDirectCPUBuffer:
    buffer_depth = 2
    capture_bs: set[int] = {1, 2, 4, 8, 16, 32}
    capture_buffers: dict[tuple[int, int, int, torch.dtype, torch.device], tuple] = {}
    temp_key: tuple[int, int, int, torch.dtype, torch.device] | None = None
    temp_buffer: tuple | None = None

    @classmethod
    def set_capture_batch_sizes(cls, capture_bs: list[int]) -> None:
        cls.capture_bs = {int(batch_size) for batch_size in capture_bs if int(batch_size) > 0}

    @classmethod
    def get_buffer(
        cls,
        hidden_states: torch.Tensor,
        num_experts_per_tok: int,
    ) -> tuple:
        flat_hidden = hidden_states.view(-1, hidden_states.shape[-1])
        batch_size = int(flat_hidden.shape[0])
        hidden_size = int(flat_hidden.shape[1])
        top_k = int(num_experts_per_tok)
        key = (batch_size, hidden_size, top_k, flat_hidden.dtype, flat_hidden.device)
        if key in cls.capture_buffers:
            return cls.capture_buffers[key]
        if key == cls.temp_key and cls.temp_buffer is not None:
            return cls.temp_buffer

        pin_memory = bool(flat_hidden.is_cuda and torch.cuda.is_available())
        cpu_kwargs = {"device": "cpu", "pin_memory": pin_memory}
        input_cpu = [
            torch.empty((batch_size, hidden_size), dtype=torch.bfloat16, **cpu_kwargs)
            for _ in range(cls.buffer_depth)
        ]
        expert_ids_cpu = [
            torch.empty((batch_size, top_k), dtype=torch.int64, **cpu_kwargs)
            for _ in range(cls.buffer_depth)
        ]
        routing_weights_cpu = [
            torch.empty((batch_size, top_k), dtype=torch.float32, **cpu_kwargs)
            for _ in range(cls.buffer_depth)
        ]
        output_cpu = [
            torch.zeros((batch_size, hidden_size), dtype=torch.bfloat16, **cpu_kwargs)
            for _ in range(cls.buffer_depth)
        ]
        batch_size_cpu = [
            torch.full((1,), batch_size, dtype=torch.int32, **cpu_kwargs)
            for _ in range(cls.buffer_depth)
        ]
        output_device = [
            torch.empty(
                (batch_size, hidden_size),
                dtype=flat_hidden.dtype,
                device=flat_hidden.device,
            )
            for _ in range(cls.buffer_depth)
        ]
        buffers = (
            input_cpu,
            expert_ids_cpu,
            routing_weights_cpu,
            output_cpu,
            batch_size_cpu,
            output_device,
        )
        if batch_size in cls.capture_bs:
            cls.capture_buffers[key] = buffers
        else:
            cls.temp_key = key
            cls.temp_buffer = buffers
        return buffers


class KtDirectCpuMoeBackend:
    def __init__(
        self,
        *,
        layer_idx: int,
        cpu_expert_pool: dict[int, dict[str, object]],
        max_routes: int,
        moe_intermediate_size: int,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        gpu_expert_mask: torch.Tensor,
        kt_num_threads: int = 0,
        kt_threadpool_count: int = 1,
        kt_chunked_prefill_size: int = 4096,
        kt_direct_backend: str = "auto",
        kt_numa_nodes: list[int] | None = None,
        kt_capture_bs: list[int] | None = None,
        strict_dtype: bool = True,
        runtime: KtDirectGlobalRuntime | object | None = None,
    ) -> None:
        self.layer_idx = int(layer_idx)
        self.max_routes = int(max_routes)
        self.intermediate_size = int(moe_intermediate_size)
        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.strict_dtype = bool(strict_dtype)
        self.min_routes = 1
        self.load_count = 0
        self.forward_count = 0

        if runtime is None:
            runtime = KtDirectGlobalRuntime.get(
                kt_num_threads=kt_num_threads,
                kt_threadpool_count=kt_threadpool_count,
                kt_numa_nodes=kt_numa_nodes,
            )
        self.runtime = runtime
        capture_batch_sizes = (
            [int(batch_size) for batch_size in kt_capture_bs]
            if kt_capture_bs is not None
            else sorted(KtDirectCPUBuffer.capture_bs)
        )
        if not capture_batch_sizes or any(batch_size < 1 for batch_size in capture_batch_sizes):
            raise ValueError("kt_capture_bs must contain at least one positive batch size")
        KtDirectCPUBuffer.set_capture_batch_sizes(capture_batch_sizes)
        # kt-kernel allocates per-layer scratch space proportional to max_len.
        # Phase one targets decode/verify; larger prefill calls use the existing GPU fallback.
        self.max_tokens = min(
            int(kt_chunked_prefill_size),
            max(capture_batch_sizes),
        )

        if gpu_expert_mask.numel() != self.num_experts:
            raise ValueError(
                f"gpu_expert_mask has {gpu_expert_mask.numel()} values, expected {self.num_experts}"
            )
        self._gpu_expert_mask_source = gpu_expert_mask
        mask_pin_memory = bool(gpu_expert_mask.is_cuda and torch.cuda.is_available())
        self.gpu_expert_mask_cpu = torch.empty(
            self.num_experts,
            dtype=torch.bool,
            device="cpu",
            pin_memory=mask_pin_memory,
        )
        self._refresh_gpu_expert_mask(non_blocking=False)

        gate_ptrs, up_ptrs, down_ptrs, refs = _build_bf16_weight_ptrs(
            cpu_expert_pool=cpu_expert_pool,
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            threadpool_count=self.runtime.kt_threadpool_count,
            strict_dtype=self.strict_dtype,
        )
        self._weight_refs = refs

        moe_config = self.runtime.kt_moe.MOEConfig(
            self.num_experts,
            self.num_experts_per_tok,
            self.hidden_size,
            self.intermediate_size,
            self.gpu_expert_mask_cpu.data_ptr(),
        )
        moe_config.layer_idx = self.layer_idx
        moe_config.pool = self.runtime.cpu_infer.backend_
        moe_config.max_len = self.max_tokens
        moe_config.gate_proj = 0
        moe_config.up_proj = 0
        moe_config.down_proj = 0
        moe_config.gate_projs = gate_ptrs
        moe_config.up_projs = up_ptrs
        moe_config.down_projs = down_ptrs
        moe_config.gate_scales = [
            [0] * self.num_experts for _ in range(self.runtime.kt_threadpool_count)
        ]
        moe_config.up_scales = [
            [0] * self.num_experts for _ in range(self.runtime.kt_threadpool_count)
        ]
        moe_config.down_scales = [
            [0] * self.num_experts for _ in range(self.runtime.kt_threadpool_count)
        ]
        if hasattr(moe_config, "load"):
            moe_config.load = False
        if hasattr(moe_config, "save"):
            moe_config.save = False

        moe_cls, selected_backend = _select_kt_bf16_moe_class(
            self.runtime.kt_moe,
            kt_direct_backend,
        )
        self.kt_selected_backend = selected_backend
        self.moe = moe_cls(moe_config)
        if hasattr(self.runtime, "retain_moe"):
            self.runtime.retain_moe(self.moe)

        self.physical_to_logical = torch.arange(
            self.num_experts,
            dtype=torch.int64,
            device="cpu",
        ).contiguous()
        self.runtime.cpu_infer.submit(
            self.moe.load_weights_task(self.physical_to_logical.data_ptr())
        )
        self.runtime.cpu_infer.sync()
        self.load_count = 1

    def supports_batch_size(self, batch_size: int) -> bool:
        return int(batch_size) <= self.max_tokens

    def _refresh_gpu_expert_mask(
        self,
        *,
        non_blocking: bool,
        force_all_cpu: bool = False,
    ) -> None:
        if force_all_cpu:
            self.gpu_expert_mask_cpu.fill_(False)
            return
        source = self._gpu_expert_mask_source
        if source.device.type == "cpu":
            self.gpu_expert_mask_cpu.copy_(source.to(dtype=torch.bool))
        else:
            self.gpu_expert_mask_cpu.copy_(source, non_blocking=non_blocking)

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
        del flat_weights, cpu_task_expert_ids, cpu_task_offsets, act_fn
        del parallel_mode, num_threads, cpu_task_expert_ids_host, cpu_task_offsets_host
        if selected_experts is None or routing_weights is None:
            raise RuntimeError("kt_direct requires selected_experts and routing_weights")
        if int(top_k) != self.num_experts_per_tok:
            raise ValueError(
                f"kt_direct top_k={top_k} does not match configured top_k={self.num_experts_per_tok}"
            )
        if int(cpu_indices.numel()) > self.max_routes:
            raise RuntimeError(
                f"CPU MoE routes {cpu_indices.numel()} exceed max_routes={self.max_routes}"
            )

        prep_t0 = perf_counter()
        flat_hidden = hidden_states.view(-1, hidden_states.shape[-1]).contiguous()
        if int(flat_hidden.shape[0]) > self.max_tokens:
            raise RuntimeError(
                f"kt_direct batch size {flat_hidden.shape[0]} exceeds max_len={self.max_tokens}; "
                "increase kt_capture_bs or kt_chunked_prefill_size"
            )
        if int(flat_hidden.shape[-1]) != self.hidden_size:
            raise ValueError(
                f"kt_direct hidden size {flat_hidden.shape[-1]} does not match {self.hidden_size}"
            )
        if self.strict_dtype and flat_hidden.dtype != torch.bfloat16:
            raise RuntimeError(f"kt_direct requires BF16 hidden states, got {flat_hidden.dtype}")

        topk_ids = selected_experts.reshape(-1, self.num_experts_per_tok).contiguous()
        topk_weights = routing_weights.reshape(-1, self.num_experts_per_tok).contiguous()
        if int(topk_ids.shape[0]) != int(flat_hidden.shape[0]):
            raise ValueError("kt_direct selected_experts batch size does not match hidden_states")

        (
            input_cpu,
            expert_ids_cpu,
            routing_weights_cpu,
            output_cpu,
            batch_size_cpu,
            output_device,
        ) = KtDirectCPUBuffer.get_buffer(flat_hidden, self.num_experts_per_tok)
        slot = self.layer_idx % KtDirectCPUBuffer.buffer_depth
        self._refresh_gpu_expert_mask(non_blocking=flat_hidden.is_cuda)
        input_cpu[slot].copy_(flat_hidden, non_blocking=flat_hidden.is_cuda)
        expert_ids_cpu[slot].copy_(topk_ids, non_blocking=topk_ids.is_cuda)
        routing_weights_cpu[slot].copy_(topk_weights, non_blocking=topk_weights.is_cuda)
        prep_ms = (perf_counter() - prep_t0) * 1000.0

        compute_t0 = perf_counter()
        task = self.moe.forward_task(
            batch_size_cpu[slot].data_ptr(),
            self.num_experts_per_tok,
            expert_ids_cpu[slot].data_ptr(),
            routing_weights_cpu[slot].data_ptr(),
            input_cpu[slot].data_ptr(),
            output_cpu[slot].data_ptr(),
            False,
        )
        if flat_hidden.is_cuda:
            stream = torch.cuda.current_stream(flat_hidden.device).cuda_stream
            self.runtime.cpu_infer.submit_with_cuda_stream(stream, task)
            self.runtime.cpu_infer.sync_with_cuda_stream(stream, 0)
            output_device[slot].copy_(output_cpu[slot], non_blocking=True)
            outputs = output_device[slot]
        else:
            self.runtime.cpu_infer.submit(task)
            self.runtime.cpu_infer.sync()
            outputs = output_cpu[slot]
        compute_ms = (perf_counter() - compute_t0) * 1000.0
        self.forward_count += 1

        return CpuMoeResult(
            token_indices=torch.empty(0, dtype=torch.int64, device="cpu"),
            outputs_cpu=outputs,
            prep_ms=prep_ms,
            compute_ms=compute_ms,
        )

    @torch.no_grad()
    def begin_forward_graph_verify(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        force_all_cpu: bool = False,
    ) -> int:
        """Submit kt_direct CPU work for CUDA graph capture/replay.

        Call before GPU GEMM to allow GPU-CPU overlap.  Returns the buffer slot.
        """
        flat_hidden = hidden_states.view(-1, hidden_states.shape[-1])
        topk_ids = selected_experts.reshape(-1, self.num_experts_per_tok).contiguous()
        topk_weights = routing_weights.reshape(-1, self.num_experts_per_tok).contiguous()

        (
            input_cpu,
            expert_ids_cpu,
            routing_weights_cpu,
            output_cpu,
            batch_size_cpu,
            output_device,
        ) = KtDirectCPUBuffer.get_buffer(flat_hidden, self.num_experts_per_tok)
        slot = self.layer_idx % KtDirectCPUBuffer.buffer_depth

        if force_all_cpu:
            self._refresh_gpu_expert_mask(
                non_blocking=flat_hidden.is_cuda,
                force_all_cpu=True,
            )
        else:
            self._refresh_gpu_expert_mask(non_blocking=flat_hidden.is_cuda)
        input_cpu[slot].copy_(flat_hidden, non_blocking=True)
        expert_ids_cpu[slot].copy_(topk_ids, non_blocking=True)
        routing_weights_cpu[slot].copy_(topk_weights, non_blocking=True)

        task = self.moe.forward_task(
            batch_size_cpu[slot].data_ptr(),
            self.num_experts_per_tok,
            expert_ids_cpu[slot].data_ptr(),
            routing_weights_cpu[slot].data_ptr(),
            input_cpu[slot].data_ptr(),
            output_cpu[slot].data_ptr(),
            False,
        )
        stream = torch.cuda.current_stream(flat_hidden.device).cuda_stream
        self.runtime.cpu_infer.submit_with_cuda_stream(stream, task)
        return slot

    @torch.no_grad()
    def finish_forward_graph_verify(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Sync kt_direct CPU work + copy output back to GPU.

        Call after GPU GEMM.  Returns per-token output tensor on GPU.
        """
        flat_hidden = hidden_states.view(-1, hidden_states.shape[-1])
        slot = self.layer_idx % KtDirectCPUBuffer.buffer_depth
        (_, _, _, output_cpu, _, output_device) = KtDirectCPUBuffer.get_buffer(
            flat_hidden, self.num_experts_per_tok,
        )
        stream = torch.cuda.current_stream(flat_hidden.device).cuda_stream
        self.runtime.cpu_infer.sync_with_cuda_stream(stream, 0)
        output_device[slot].copy_(output_cpu[slot], non_blocking=True)
        return output_device[slot]
