from __future__ import annotations

import ctypes
import gc
import importlib
import importlib.util
import os
import sys
import warnings
from time import perf_counter
from typing import Callable

import torch

from nanovllm.expert.cpu_weights import NumaShardedExpertTensor
from nanovllm.layers.fuse_moe.cpu_backend import CpuMoeResult
from nanovllm.utils.verify_op_events import verify_op_event


def _trim_cpu_allocator() -> None:
    gc.collect()
    try:
        malloc_trim = ctypes.CDLL(None).malloc_trim
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        malloc_trim(0)
    except (AttributeError, OSError):
        pass


def _warn_if_process_affinity_is_restricted() -> None:
    """Warn when an outer taskset can fight CPUInfer's own hwloc binding."""
    try:
        allowed = os.sched_getaffinity(0)
    except Exception:
        return
    online_count = int(os.cpu_count() or len(allowed))
    if len(allowed) >= online_count:
        return
    warnings.warn(
        "kt_direct detected restricted process CPU affinity "
        f"({len(allowed)}/{online_count} logical CPUs). CPUInfer already binds "
        "its NUMA worker pools with hwloc; on this host an outer taskset made "
        "the MoE kernel 7-10x slower. Run without taskset unless a controlled "
        "microbenchmark proves the restricted mask is faster.",
        RuntimeWarning,
        stacklevel=2,
    )


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
        _warn_if_process_affinity_is_restricted()
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


def _load_cpuinfer_ext(extension_path: str):
    """Load KTransformers' legacy llamafile extension without polluting PYTHONPATH."""
    raw_path = str(extension_path).strip()
    if not raw_path:
        try:
            return importlib.import_module("cpuinfer_ext")
        except Exception as exc:
            raise RuntimeError(
                "a llamafile kt_direct_backend requires the fixed "
                "KTransformers cpuinfer_ext; install it in this environment or "
                "set kt_llamafile_extension_path"
            ) from exc
    requested_path = os.path.realpath(os.path.expanduser(raw_path))
    if not os.path.isfile(requested_path):
        raise FileNotFoundError(
            f"KTransformers cpuinfer_ext does not exist: {requested_path}"
        )

    loaded = sys.modules.get("cpuinfer_ext")
    if loaded is not None:
        loaded_path = os.path.realpath(str(getattr(loaded, "__file__", "")))
        if loaded_path != requested_path:
            raise RuntimeError(
                "cpuinfer_ext is already loaded from a different path: "
                f"loaded={loaded_path}, requested={requested_path}"
            )
        return loaded

    spec = importlib.util.spec_from_file_location("cpuinfer_ext", requested_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create an import spec for {requested_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["cpuinfer_ext"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop("cpuinfer_ext", None)
        raise
    return module


class KtLlamafileGlobalRuntime:
    """CPUInfer runtime backed by KTransformers' fixed legacy llamafile MOE."""

    _instance: "KtLlamafileGlobalRuntime | None" = None

    @classmethod
    def get(
        cls,
        *,
        kt_num_threads: int,
        kt_threadpool_count: int,
        kt_numa_nodes: list[int] | None = None,
        extension_path: str = "",
    ) -> "KtLlamafileGlobalRuntime":
        pools = int(kt_threadpool_count)
        threads, resolved_numa_nodes, _ = _resolve_runtime_layout(
            kt_num_threads=kt_num_threads,
            kt_threadpool_count=pools,
            kt_numa_nodes=kt_numa_nodes,
        )
        requested_path = (
            os.path.realpath(os.path.expanduser(str(extension_path)))
            if extension_path
            else ""
        )
        configured = (threads, pools, tuple(resolved_numa_nodes), requested_path)
        if cls._instance is None:
            cls._instance = cls(
                kt_num_threads=threads,
                kt_threadpool_count=pools,
                kt_numa_nodes=resolved_numa_nodes,
                extension_path=requested_path,
            )
        else:
            current = cls._instance
            current_config = (
                current.kt_num_threads,
                current.kt_threadpool_count,
                current.kt_numa_nodes,
                current.extension_path,
            )
            if configured != current_config:
                raise RuntimeError(
                    "llamafile CPUInfer is already initialized with "
                    f"threads/pools/numa/path={current_config}, requested={configured}"
                )
        return cls._instance

    def __init__(
        self,
        *,
        kt_num_threads: int,
        kt_threadpool_count: int,
        kt_numa_nodes: list[int],
        extension_path: str,
    ) -> None:
        _trim_cpu_allocator()
        _warn_if_process_affinity_is_restricted()
        extension = _load_cpuinfer_ext(extension_path)
        threads, resolved_numa_nodes, thread_counts = _resolve_runtime_layout(
            kt_num_threads=kt_num_threads,
            kt_threadpool_count=kt_threadpool_count,
            kt_numa_nodes=kt_numa_nodes,
        )
        worker_config = extension.WorkerPoolConfig()
        worker_config.subpool_count = int(kt_threadpool_count)
        worker_config.subpool_numa_map = list(resolved_numa_nodes)
        worker_config.subpool_thread_count = list(thread_counts)

        self.kt_kernel_ext = extension
        self.kt_moe = extension.moe
        self.cpu_infer = extension.CPUInfer(worker_config)
        self.kt_num_threads = int(threads)
        self.kt_threadpool_count = int(kt_threadpool_count)
        self.kt_numa_nodes = tuple(resolved_numa_nodes)
        self.extension_path = str(extension_path)
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


def _pack_llamafile_weights(
    *,
    cpu_expert_pool: dict[int, dict[str, object]],
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    strict_dtype: bool,
    weight_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build temporary expert-major arrays consumed by legacy load_weights()."""
    if weight_dtype not in {torch.bfloat16, torch.float16}:
        raise ValueError(f"unsupported llamafile weight dtype: {weight_dtype}")
    backend_label = (
        "llamafile_bf16" if weight_dtype == torch.bfloat16 else "llamafile_f16"
    )
    gates: list[torch.Tensor] = []
    ups: list[torch.Tensor] = []
    downs: list[torch.Tensor] = []
    for expert_idx in range(int(num_experts)):
        params = cpu_expert_pool.get(expert_idx)
        if params is None:
            raise RuntimeError(
                f"{backend_label} requires every CPU expert; missing {expert_idx}"
            )
        gate_up, down = _get_raw_gate_up_down(
            params,
            expert_idx=expert_idx,
            strict_dtype=strict_dtype,
        )
        expected_gate_up = (int(intermediate_size) * 2, int(hidden_size))
        expected_down = (int(hidden_size), int(intermediate_size))
        if tuple(gate_up.shape) != expected_gate_up:
            raise RuntimeError(
                f"{backend_label} expert {expert_idx} gate_up shape mismatch: "
                f"got {tuple(gate_up.shape)}, expected {expected_gate_up}"
            )
        if tuple(down.shape) != expected_down:
            raise RuntimeError(
                f"{backend_label} expert {expert_idx} down shape mismatch: "
                f"got {tuple(down.shape)}, expected {expected_down}"
            )
        gates.append(gate_up[:intermediate_size])
        ups.append(gate_up[intermediate_size:])
        downs.append(down)
    packed = (
        torch.stack(gates, dim=0).contiguous(),
        torch.stack(ups, dim=0).contiguous(),
        torch.stack(downs, dim=0).contiguous(),
    )
    if weight_dtype == torch.bfloat16:
        return packed
    return tuple(tensor.to(dtype=weight_dtype) for tensor in packed)


def _pack_llamafile_bf16_weights(
    *,
    cpu_expert_pool: dict[int, dict[str, object]],
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    strict_dtype: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compatibility wrapper for callers/tests of the original BF16 path."""
    return _pack_llamafile_weights(
        cpu_expert_pool=cpu_expert_pool,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        strict_dtype=strict_dtype,
        weight_dtype=torch.bfloat16,
    )


def _tensor_view_from_address(
    address: int,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, object]:
    """Create a non-owning CPU tensor view over a native uint16 weight buffer."""
    if dtype not in {torch.bfloat16, torch.float16}:
        raise ValueError(f"unsupported CPUInfer weight dtype: {dtype}")
    numel = 1
    for size in shape:
        numel *= int(size)
    if int(address) <= 0:
        raise RuntimeError("CPUInfer returned a null weight pointer")
    buffer_type = ctypes.c_uint16 * numel
    buffer = buffer_type.from_address(int(address))
    tensor = torch.frombuffer(buffer, dtype=dtype, count=numel).reshape(shape)
    return tensor, buffer


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
        kt_llamafile_extension_path: str = "",
        kt_numa_nodes: list[int] | None = None,
        kt_capture_bs: list[int] | None = None,
        kt_single_weight: bool = True,
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
        self.kt_single_weight = bool(kt_single_weight)
        self.min_routes = 1
        self.load_count = 0
        self.forward_count = 0

        normalized_backend = str(kt_direct_backend).strip().lower()
        self._legacy_llamafile = normalized_backend in {
            "llamafile_bf16",
            "llamafile_f16",
        }
        self._llamafile_weight_type = (
            1 if normalized_backend == "llamafile_f16" else 30
        )
        if runtime is None:
            if self._legacy_llamafile:
                runtime = KtLlamafileGlobalRuntime.get(
                    kt_num_threads=kt_num_threads,
                    kt_threadpool_count=kt_threadpool_count,
                    kt_numa_nodes=kt_numa_nodes,
                    extension_path=kt_llamafile_extension_path,
                )
            else:
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

        temporary_llamafile_weights: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor
        ] | None = None
        if self._legacy_llamafile:
            weight_dtype = (
                torch.float16
                if self._llamafile_weight_type == 1
                else torch.bfloat16
            )
            temporary_llamafile_weights = _pack_llamafile_weights(
                cpu_expert_pool=cpu_expert_pool,
                num_experts=self.num_experts,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                strict_dtype=self.strict_dtype,
                weight_dtype=weight_dtype,
            )
            gate, up, down = temporary_llamafile_weights
            moe_config = self.runtime.kt_moe.MOEConfig(
                self.num_experts,
                self.num_experts_per_tok,
                self.hidden_size,
                self.intermediate_size,
            )
            self._weight_refs: list[torch.Tensor] = []
        else:
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
        if self._legacy_llamafile:
            # The historical threshold (10) serializes qlen=2..9 through
            # forward_one.  On this dual-socket AVX2 host, the grouped path is
            # neutral at qlen=1 and 8-21% faster at qlen=2..7, which covers
            # both true-batch decode and speculative verify buckets.
            moe_config.m_block = 32
            moe_config.group_min_len = 1
            moe_config.group_max_len = self.max_tokens
            moe_config.gate_proj = gate.data_ptr()
            moe_config.up_proj = up.data_ptr()
            moe_config.down_proj = down.data_ptr()
            moe_config.gate_type = self._llamafile_weight_type
            moe_config.up_type = self._llamafile_weight_type
            moe_config.down_type = self._llamafile_weight_type
            moe_config.hidden_type = 30
            if hasattr(moe_config, "load"):
                moe_config.load = False
            if hasattr(moe_config, "save"):
                moe_config.save = False
            self.kt_selected_backend = normalized_backend
            self.moe = self.runtime.kt_moe.MOE(moe_config)
        else:
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
        if self._legacy_llamafile:
            self.runtime.cpu_infer.submit(self.moe.load_weights())
        else:
            self.runtime.cpu_infer.submit(
                self.moe.load_weights_task(self.physical_to_logical.data_ptr())
            )
        self.runtime.cpu_infer.sync()
        self.load_count = 1
        if self._legacy_llamafile and self.kt_single_weight:
            self._adopt_loaded_llamafile_weights(cpu_expert_pool)
        # Legacy load_weights() copies this layer into NUMA-local buffers.  Drop
        # the temporary expert-major arrays immediately.  In single-weight mode
        # the pool above now views those native buffers; compatibility mode keeps
        # the original per-expert tensors as the GPU cache source.
        if temporary_llamafile_weights is not None:
            del temporary_llamafile_weights, gate, up, down
            _trim_cpu_allocator()

    def _adopt_loaded_llamafile_weights(
        self,
        cpu_expert_pool: dict[int, dict[str, object]],
    ) -> None:
        """Make CPUInfer's loaded buffers the sole CPU source of expert weights."""
        getter = getattr(self.moe, "get_weight_ptrs", None)
        if getter is None:
            raise RuntimeError(
                "kt_single_weight requires a cpuinfer_ext build exposing "
                "MOE.get_weight_ptrs(); apply patches/ktransformers-single-weight.patch "
                "and rebuild the extension"
            )
        pointer_rows = getter()
        pool_count = int(self.runtime.kt_threadpool_count)
        if len(pointer_rows) != pool_count:
            raise RuntimeError(
                "CPUInfer returned an unexpected number of NUMA weight shards: "
                f"got {len(pointer_rows)}, expected {pool_count}"
            )
        if self.intermediate_size % pool_count != 0:
            raise RuntimeError(
                "single-weight llamafile requires intermediate_size divisible "
                "by kt_threadpool_count"
            )
        local_intermediate = self.intermediate_size // pool_count
        weight_dtype = (
            torch.float16 if self._llamafile_weight_type == 1 else torch.bfloat16
        )

        gate_layers: list[torch.Tensor] = []
        up_layers: list[torch.Tensor] = []
        down_layers: list[torch.Tensor] = []
        self._native_weight_buffers: list[object] = []
        for row in pointer_rows:
            if len(row) != 3:
                raise RuntimeError(
                    "CPUInfer get_weight_ptrs() must return gate/up/down pointers"
                )
            gate, gate_buffer = _tensor_view_from_address(
                int(row[0]),
                shape=(self.num_experts, local_intermediate, self.hidden_size),
                dtype=weight_dtype,
            )
            up, up_buffer = _tensor_view_from_address(
                int(row[1]),
                shape=(self.num_experts, local_intermediate, self.hidden_size),
                dtype=weight_dtype,
            )
            down, down_buffer = _tensor_view_from_address(
                int(row[2]),
                shape=(self.num_experts, self.hidden_size, local_intermediate),
                dtype=weight_dtype,
            )
            gate_layers.append(gate)
            up_layers.append(up)
            down_layers.append(down)
            self._native_weight_buffers.extend((gate_buffer, up_buffer, down_buffer))

        for expert_idx in range(self.num_experts):
            params = cpu_expert_pool.get(expert_idx)
            if params is None:
                raise RuntimeError(
                    f"single-weight conversion is missing expert {expert_idx}"
                )
            gate_up_source = NumaShardedExpertTensor(
                kind="gate_up",
                gate_shards=tuple(layer[expert_idx] for layer in gate_layers),
                up_shards=tuple(layer[expert_idx] for layer in up_layers),
            )
            down_source = NumaShardedExpertTensor(
                kind="down",
                down_shards=tuple(layer[expert_idx] for layer in down_layers),
            )
            # The dict is shared by LayerExpertCache and every prefetch runtime;
            # replacing it in place releases the old raw tensors everywhere.
            params.clear()
            params["gate_up"] = gate_up_source
            params["down"] = down_source

    def _cpu_topk_ids(
        self,
        topk_ids: torch.Tensor,
        cpu_route_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not bool(getattr(self, "_legacy_llamafile", False)):
            return topk_ids
        if cpu_route_mask is not None:
            if cpu_route_mask.numel() != topk_ids.numel():
                raise ValueError("cpu_route_mask shape does not match top-k routes")
            route_mask = cpu_route_mask.reshape_as(topk_ids)
            if route_mask.device != topk_ids.device:
                route_mask = route_mask.to(device=topk_ids.device)
            return torch.where(route_mask, topk_ids, -1)
        mask = self._gpu_expert_mask_source
        if mask.device != topk_ids.device:
            mask = mask.to(device=topk_ids.device)
        cached_routes = mask.index_select(0, topk_ids.reshape(-1)).view_as(topk_ids)
        return topk_ids.masked_fill(cached_routes, -1)

    def _make_forward_task(
        self,
        *,
        batch_size_ptr: int,
        top_k: int,
        expert_ids_ptr: int,
        routing_weights_ptr: int,
        input_ptr: int,
        output_ptr: int,
    ):
        args = (
            batch_size_ptr,
            top_k,
            expert_ids_ptr,
            routing_weights_ptr,
            input_ptr,
            output_ptr,
            False,
        )
        if bool(getattr(self, "_legacy_llamafile", False)):
            return self.moe.forward(*args)
        return self.moe.forward_task(*args)

    def supports_batch_size(self, batch_size: int) -> bool:
        return int(batch_size) <= self.max_tokens

    def _refresh_gpu_expert_mask(self, *, non_blocking: bool) -> None:
        source = self._gpu_expert_mask_source
        if source.device.type == "cpu":
            self.gpu_expert_mask_cpu.copy_(source.to(dtype=torch.bool))
        else:
            self.gpu_expert_mask_cpu.copy_(source, non_blocking=non_blocking)

    def _refresh_gpu_expert_mask_for_cpu_backend(
        self,
        *,
        non_blocking: bool,
    ) -> None:
        """Refresh the host mask only when the native CPU backend consumes it.

        The legacy llamafile binding does not receive ``gpu_expert_mask_cpu`` in
        its ``MOEConfig``.  Cached routes are already replaced with ``-1`` by
        :meth:`_cpu_topk_ids`, so copying the same mask to host memory on every
        layer/forward is both redundant and, under CUDA graph replay, an extra
        D2H memcpy node.
        """
        if bool(getattr(self, "_legacy_llamafile", False)):
            return
        self._refresh_gpu_expert_mask(non_blocking=non_blocking)

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
        topk_ids = KtDirectCpuMoeBackend._cpu_topk_ids(self, topk_ids)
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
        KtDirectCpuMoeBackend._refresh_gpu_expert_mask_for_cpu_backend(
            self,
            non_blocking=flat_hidden.is_cuda,
        )
        input_cpu[slot].copy_(flat_hidden, non_blocking=flat_hidden.is_cuda)
        expert_ids_cpu[slot].copy_(topk_ids, non_blocking=topk_ids.is_cuda)
        routing_weights_cpu[slot].copy_(topk_weights, non_blocking=topk_weights.is_cuda)
        prep_ms = (perf_counter() - prep_t0) * 1000.0

        compute_t0 = perf_counter()
        task = KtDirectCpuMoeBackend._make_forward_task(
            self,
            batch_size_ptr=batch_size_cpu[slot].data_ptr(),
            top_k=self.num_experts_per_tok,
            expert_ids_ptr=expert_ids_cpu[slot].data_ptr(),
            routing_weights_ptr=routing_weights_cpu[slot].data_ptr(),
            input_ptr=input_cpu[slot].data_ptr(),
            output_ptr=output_cpu[slot].data_ptr(),
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
        *,
        include_gpu_cached_routes: bool = False,
        cpu_route_mask: torch.Tensor | None = None,
    ) -> int:
        """Submit kt_direct CPU work for CUDA graph capture/replay.

        Call before GPU GEMM to allow GPU-CPU overlap.  Returns the buffer slot.
        """
        flat_hidden = hidden_states.view(-1, hidden_states.shape[-1])
        topk_ids = selected_experts.reshape(-1, self.num_experts_per_tok).contiguous()
        if not include_gpu_cached_routes:
            topk_ids = KtDirectCpuMoeBackend._cpu_topk_ids(
                self,
                topk_ids,
                cpu_route_mask=cpu_route_mask,
            )
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

        with verify_op_event("kt.cpu_prepare_copies", self.layer_idx):
            KtDirectCpuMoeBackend._refresh_gpu_expert_mask_for_cpu_backend(
                self,
                non_blocking=flat_hidden.is_cuda,
            )
            input_cpu[slot].copy_(flat_hidden, non_blocking=True)
            expert_ids_cpu[slot].copy_(topk_ids, non_blocking=True)
            routing_weights_cpu[slot].copy_(topk_weights, non_blocking=True)

        with verify_op_event("kt.forward_task_create", self.layer_idx):
            task = KtDirectCpuMoeBackend._make_forward_task(
                self,
                batch_size_ptr=batch_size_cpu[slot].data_ptr(),
                top_k=self.num_experts_per_tok,
                expert_ids_ptr=expert_ids_cpu[slot].data_ptr(),
                routing_weights_ptr=routing_weights_cpu[slot].data_ptr(),
                input_ptr=input_cpu[slot].data_ptr(),
                output_ptr=output_cpu[slot].data_ptr(),
            )
        stream = torch.cuda.current_stream(flat_hidden.device).cuda_stream
        with verify_op_event("kt.cpuinfer_submit", self.layer_idx):
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
        with verify_op_event("kt.cpuinfer_sync", self.layer_idx):
            self.runtime.cpu_infer.sync_with_cuda_stream(stream, 0)
        with verify_op_event("kt.output_cpu_to_gpu_copy", self.layer_idx):
            output_device[slot].copy_(output_cpu[slot], non_blocking=True)
        return output_device[slot]
