import pickle
import os
import json
import threading
from collections import defaultdict
from queue import Queue
from time import perf_counter
import torch
import torch.nn.functional as F
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models import Qwen3ForCausalLM, Qwen3MoeForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.scheduling.draft_scheduler import create_draft_scheduler
from nanovllm.scheduling.cache_strategy import create_cache_strategy
from nanovllm.scheduling.prefetch_strategy import create_prefetch_strategy
from nanovllm.expert.prefetcher import PrefetchRuntime
from nanovllm.expert.runtime_meta import ModelRuntimeMetaRecorder
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.utils.heterogeneous_loader import HeterogeneousModelLoader


class ModelRunner:
    MODEL_TYPE_DICT = {
        "qwen3": Qwen3ForCausalLM,
        "qwen3_moe": Qwen3MoeForCausalLM,
    }

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.profile_enabled = bool(getattr(config, "engine_profile", False))
        self.profile_cuda_sync = bool(getattr(config, "engine_profile_cuda_sync", True))
        self._profile = defaultdict(float)
        self.layer_caches = {}
        self.cpu_expert_pool = {}
        self.cache_strategy = create_cache_strategy(config.cache_strategy)
        self.prefetch_strategy = create_prefetch_strategy(config.prefetch_strategy, config)
        self.runtime_meta_recorder: ModelRuntimeMetaRecorder | None = None
        self.prefetch_runtime: PrefetchRuntime | None = None
        self.prefetch_effective_enabled = False
        self._pending_prefetch_metadata: list[dict[str, object]] = []
        self._prefetch_step_id = 0
        self._skip_metadata_observe = os.getenv("NANOVLLM_PREFETCH_SKIP_OBSERVE", "0").strip().lower() in {
            "1", "true", "yes", "on"
        }
        self._prefetch_async_enabled = False
        self._prefetch_runtime_lock = threading.RLock()
        self._prefetch_profile_lock = threading.RLock()
        self._prefetch_worker_queue: Queue | None = None
        self._prefetch_worker_cv = threading.Condition()
        self._prefetch_worker_outstanding = 0
        self._prefetch_worker_key_counts: dict[tuple[str, int, int], int] = defaultdict(int)
        self._prefetch_device_handles: dict[tuple[str, int], list[object]] = defaultdict(list)
        self._prefetch_worker_error: BaseException | None = None
        self._prefetch_worker_thread: threading.Thread | None = None
        self._prefetch_trace_events: list[dict[str, object]] = []

        dist_url = f"tcp://localhost:{config.dist_port}"
        dist.init_process_group("nccl", dist_url, world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        setattr(hf_config, "enable_heterogeneous", config.enable_heterogeneous)
        self.model = self.MODEL_TYPE_DICT[hf_config.model_type](hf_config)
        if config.enable_heterogeneous and hasattr(self.model, "enable_heterogeneous_mode"):
            loader = HeterogeneousModelLoader(config)
            layer_caches, cpu_expert_pool = loader.load(self.model, config.model)
            self.layer_caches = layer_caches
            self.cpu_expert_pool = cpu_expert_pool
            self.model.enable_heterogeneous_mode(
                layer_caches,
                cpu_expert_pool,
                cpu_expert_execution_enabled=getattr(config, "cpu_expert_execution_enabled", False),
                cpu_expert_parallel_mode=getattr(config, "cpu_expert_parallel_mode", "serial"),
                cpu_expert_num_threads=getattr(config, "cpu_expert_num_threads", 4),
                cpu_expert_backend=getattr(config, "cpu_expert_backend", "torch"),
                cpu_expert_workspace_max_routes=getattr(config, "cpu_expert_workspace_max_routes", 8192),
                cpu_expert_packed_min_routes=getattr(config, "cpu_expert_packed_min_routes", 32),
                cpu_expert_strict_dtype=getattr(config, "cpu_expert_strict_dtype", True),
                cpu_gpu_parallel_execution_enabled=getattr(config, "cpu_gpu_parallel_execution_enabled", "auto"),
                cpu_gpu_parallel_min_cpu_route_ratio=getattr(config, "cpu_gpu_parallel_min_cpu_route_ratio", 0.0),
            )
            self.draft_scheduler = create_draft_scheduler(getattr(config, "draft_scheduler", "simple"))
            self.prefetch_effective_enabled = bool(config.spec_enable_prefetch) and config.inference_mode == "spec"

            if self.prefetch_effective_enabled:
                self.runtime_meta_recorder = ModelRuntimeMetaRecorder(config=config, hf_config=config.hf_config)
                self.prefetch_runtime = PrefetchRuntime(
                    config=config,
                    layer_caches=self.layer_caches,
                    cpu_expert_pool=self.cpu_expert_pool,
                    cache_strategy=self.cache_strategy,
                    prefetch_strategy=self.prefetch_strategy,
                    runtime_meta_recorder=self.runtime_meta_recorder,
                )
                if hasattr(self.model, "set_runtime_meta_recorder"):
                    self.model.set_runtime_meta_recorder(self.runtime_meta_recorder)
                self._prefetch_async_enabled = bool(self.prefetch_effective_enabled)
                if self._prefetch_async_enabled:
                    self._prefetch_worker_queue = Queue()
                    self._prefetch_worker_thread = threading.Thread(
                        target=self._prefetch_worker_main,
                        name=f"prefetch-worker-rank{self.rank}",
                        daemon=True,
                    )
                    self._prefetch_worker_thread.start()
        else:
            load_model(self.model, config.model)
            self.draft_scheduler = None
        self._decode_graph_policy = "standard"
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            # Spec mode decode uses draft-specific graph path; capturing standard decode graph
            # can hit non-capture-safe MoE planning ops and is unnecessary for spec execution.
            if self.config.inference_mode != "spec":
                self.capture_cudagraph()
            else:
                self.graph_bs = []
                self.graphs = {}
                self.graph_vars = {}
                self.graph_pool = None
            self.capture_draft_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        self._flush_pending_prefetch_metadata(block=True)
        self._shutdown_prefetch_worker()
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
            if hasattr(self, "draft_graphs"):
                del self.draft_graphs
            if hasattr(self, "draft_graph_pool"):
                del self.draft_graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def _ensure_prefetch_internal_state(self) -> None:
        if not hasattr(self, "profile_enabled"):
            self.profile_enabled = False
        if not hasattr(self, "profile_cuda_sync"):
            self.profile_cuda_sync = False
        if not hasattr(self, "rank"):
            self.rank = 0
        if not hasattr(self, "_profile"):
            self._profile = defaultdict(float)
        if not hasattr(self, "_prefetch_async_enabled"):
            self._prefetch_async_enabled = False
        if not hasattr(self, "_prefetch_runtime_lock"):
            self._prefetch_runtime_lock = threading.RLock()
        if not hasattr(self, "_prefetch_profile_lock"):
            self._prefetch_profile_lock = threading.RLock()
        if not hasattr(self, "_prefetch_worker_cv"):
            self._prefetch_worker_cv = threading.Condition()
        if not hasattr(self, "_prefetch_worker_outstanding"):
            self._prefetch_worker_outstanding = 0
        if not hasattr(self, "_prefetch_worker_key_counts"):
            self._prefetch_worker_key_counts = defaultdict(int)
        if not hasattr(self, "_prefetch_device_handles"):
            self._prefetch_device_handles = defaultdict(list)
        if not hasattr(self, "_prefetch_worker_error"):
            self._prefetch_worker_error = None
        if not hasattr(self, "_prefetch_worker_queue"):
            self._prefetch_worker_queue = None
        if not hasattr(self, "_prefetch_worker_thread"):
            self._prefetch_worker_thread = None
        if not hasattr(self, "_prefetch_trace_events"):
            self._prefetch_trace_events = []

    def _record_profile(self, key: str, dt_sec: float) -> None:
        self._ensure_prefetch_internal_state()
        if self.profile_enabled and self.rank == 0:
            with self._prefetch_profile_lock:
                self._profile[key] += dt_sec * 1000.0

    def _next_prefetch_step_id(self) -> int:
        self._prefetch_step_id += 1
        return self._prefetch_step_id

    def _trace_prefetch_interval(
        self,
        *,
        name: str,
        start_ms: float,
        duration_ms: float,
        step_id: int,
        mode: str,
        tid: str,
    ) -> None:
        self._ensure_prefetch_internal_state()
        if not (self.profile_enabled and self.rank == 0):
            return
        if duration_ms <= 0.0:
            return
        with self._prefetch_profile_lock:
            self._prefetch_trace_events.append(
                {
                    "name": name,
                    "ph": "X",
                    "ts": float(start_ms) * 1000.0,
                    "dur": float(duration_ms) * 1000.0,
                    "pid": 1,
                    "tid": tid,
                    "args": {
                        "step_id": int(step_id),
                        "mode": str(mode),
                    },
                }
            )

    def _raise_prefetch_worker_error(self) -> None:
        self._ensure_prefetch_internal_state()
        error = getattr(self, "_prefetch_worker_error", None)
        if error is not None:
            raise RuntimeError("Background prefetch metadata worker failed") from error

    def _prefetch_worker_key(self, *, mode: str, handle) -> tuple[str, int]:
        return (str(mode), int(getattr(handle, "token_capacity", 0)))

    def _wait_for_prefetch_device_reuse(self, *, mode: str, token_capacity: int) -> float:
        self._ensure_prefetch_internal_state()
        if not getattr(self, "_prefetch_async_enabled", False):
            return 0.0
        wait_t0 = perf_counter()
        key = (str(mode), int(token_capacity))
        with self._prefetch_worker_cv:
            while True:
                pending_handles = []
                for handle in self._prefetch_device_handles.get(key, []):
                    event = getattr(handle, "event", None)
                    if event is not None and not event.query():
                        pending_handles.append(handle)
                if pending_handles:
                    self._prefetch_device_handles[key] = pending_handles
                else:
                    self._prefetch_device_handles.pop(key, None)
                    break
                self._raise_prefetch_worker_error()
                self._prefetch_worker_cv.wait(timeout=0.001)
            self._raise_prefetch_worker_error()
        wait_ms = (perf_counter() - wait_t0) * 1000.0
        if self.profile_enabled and self.rank == 0:
            prefix = f"run_{mode}_metadata"
            with self._prefetch_profile_lock:
                self._profile["prefetch_async_buffer_reuse_wait_count"] += 1
                self._profile["prefetch_async_buffer_reuse_wait_ms"] += wait_ms
                self._profile["prefetch_async_device_reuse_wait_ms"] += wait_ms
                self._profile[f"{prefix}_buffer_reuse_wait_ms"] += wait_ms
                self._profile[f"{prefix}_buffer_device_reuse_wait_ms"] += wait_ms
            self._trace_prefetch_interval(
                name="prefetch_device_reuse_wait",
                start_ms=wait_t0 * 1000.0,
                duration_ms=wait_ms,
                step_id=-1,
                mode=mode,
                tid="main",
            )
        return wait_ms

    def _acquire_prefetch_host_buffer_slot(self, *, mode: str, token_capacity: int) -> tuple[int, float]:
        self._ensure_prefetch_internal_state()
        runtime_meta_recorder = getattr(self, "runtime_meta_recorder", None)
        if runtime_meta_recorder is None:
            return 0, 0.0
        if not getattr(self, "_prefetch_async_enabled", False):
            return 0, 0.0
        wait_t0 = perf_counter()
        mode_key = str(mode)
        capacity = int(token_capacity)
        with self._prefetch_worker_cv:
            while True:
                pool_size = runtime_meta_recorder.get_host_buffer_pool_size(mode_key, capacity)
                for slot_idx in range(pool_size):
                    slot_key = (mode_key, capacity, slot_idx)
                    if self._prefetch_worker_key_counts.get(slot_key, 0) <= 0:
                        self._raise_prefetch_worker_error()
                        wait_ms = (perf_counter() - wait_t0) * 1000.0
                        if self.profile_enabled and self.rank == 0:
                            prefix = f"run_{mode}_metadata"
                            with self._prefetch_profile_lock:
                                self._profile["prefetch_async_buffer_reuse_wait_count"] += 1
                                self._profile["prefetch_async_buffer_reuse_wait_ms"] += wait_ms
                                self._profile["prefetch_async_host_reuse_wait_ms"] += wait_ms
                                self._profile[f"{prefix}_buffer_reuse_wait_ms"] += wait_ms
                                self._profile[f"{prefix}_buffer_host_reuse_wait_ms"] += wait_ms
                            self._trace_prefetch_interval(
                                name="prefetch_host_reuse_wait",
                                start_ms=wait_t0 * 1000.0,
                                duration_ms=wait_ms,
                                step_id=-1,
                                mode=mode,
                                tid="main",
                            )
                        return slot_idx, wait_ms
                if runtime_meta_recorder.maybe_grow_host_buffer_pool(mode_key, capacity):
                    continue
                self._raise_prefetch_worker_error()
                self._prefetch_worker_cv.wait(timeout=0.001)
        return 0, 0.0

    def _wait_for_prefetch_async_drain(self) -> float:
        self._ensure_prefetch_internal_state()
        if not getattr(self, "_prefetch_async_enabled", False):
            return 0.0
        wait_t0 = perf_counter()
        with self._prefetch_worker_cv:
            while self._prefetch_worker_outstanding > 0:
                self._raise_prefetch_worker_error()
                self._prefetch_worker_cv.wait(timeout=0.001)
            self._raise_prefetch_worker_error()
        wait_ms = (perf_counter() - wait_t0) * 1000.0
        if self.profile_enabled and self.rank == 0:
            with self._prefetch_profile_lock:
                self._profile["prefetch_async_drain_count"] += 1
                self._profile["prefetch_async_drain_wait_ms"] += wait_ms
        return wait_ms

    def _process_prefetch_metadata_item(
        self,
        item: dict[str, object],
        *,
        block: bool,
        processing_origin: str,
    ) -> bool:
        self._ensure_prefetch_internal_state()
        prefetch_runtime = getattr(self, "prefetch_runtime", None)
        runtime_meta_recorder = getattr(self, "runtime_meta_recorder", None)
        if prefetch_runtime is None or runtime_meta_recorder is None:
            return True

        handle = item["handle"]
        event = getattr(handle, "event", None)
        if event is not None and not block and not event.query():
            return False

        worker_wait_t0 = perf_counter()
        transfer_wait_ms = 0.0
        if event is not None and block:
            event.synchronize()
            transfer_wait_ms = (perf_counter() - worker_wait_t0) * 1000.0

        collect_t0 = perf_counter()
        runtime_meta = runtime_meta_recorder.collect(handle, wait=False)
        collect_ms = (perf_counter() - collect_t0) * 1000.0

        observe_stats: dict[str, float] = {}
        submit_after_ms = 0.0
        observe_t0 = perf_counter()
        with self._prefetch_runtime_lock:
            if not getattr(self, "_skip_metadata_observe", False):
                mode = str(item["mode"])
                step_id = int(item["step_id"])
                if mode == "prefill":
                    observe_stats = prefetch_runtime.observe_prefill(runtime_meta, step_id=step_id)
                elif mode == "draft":
                    observe_stats = prefetch_runtime.observe_draft(runtime_meta, step_id=step_id)
                elif mode == "verify":
                    observe_stats = prefetch_runtime.observe_verify(runtime_meta, step_id=step_id)
                    if bool(item["record_verify_consumed"]):
                        prefetch_runtime.record_verify_consumed(runtime_meta, step_id=step_id)
            elif self.profile_enabled and self.rank == 0:
                with self._prefetch_profile_lock:
                    self._profile["metadata_observe_skipped_count"] += 1
            observe_ms = (perf_counter() - observe_t0) * 1000.0

            submit_after_phase = item["submit_after_phase"]
            if submit_after_phase is not None:
                submit_after_t0 = perf_counter()
                prefetch_runtime.submit_from_global_queue(step_id=int(item["step_id"]), phase=str(submit_after_phase))
                submit_after_ms = (perf_counter() - submit_after_t0) * 1000.0

            prefetch_runtime.record_metadata_offload(
                mode=str(item["mode"]),
                num_bytes=int(handle.buffer_bytes),
                enqueue_ms=float(item["enqueue_ms"]),
                transfer_wait_ms=transfer_wait_ms,
                collect_ms=collect_ms,
                observe_ms=observe_ms,
            )

        if self.profile_enabled and self.rank == 0:
            prefix = f"run_{item['mode']}_metadata"
            turnaround_ms = (perf_counter() * 1000.0) - float(item["enqueue_ts_ms"])
            with self._prefetch_profile_lock:
                self._profile[f"{prefix}_enqueue_ms"] += float(item["enqueue_ms"])
                self._profile[f"{prefix}_wait_ms"] += transfer_wait_ms
                self._profile[f"{prefix}_collect_ms"] += collect_ms
                self._profile[f"{prefix}_observe_ms"] += observe_ms
                self._profile[f"{prefix}_mark_access_ms"] += float(observe_stats.get("mark_access_ms", 0.0))
                self._profile[f"{prefix}_queue_update_ms"] += float(observe_stats.get("queue_update_ms", 0.0))
                self._profile[f"{prefix}_queue_aggregate_ms"] += float(observe_stats.get("queue_aggregate_ms", 0.0))
                self._profile[f"{prefix}_queue_filter_ms"] += float(observe_stats.get("queue_filter_ms", 0.0))
                self._profile[f"{prefix}_queue_entry_update_ms"] += float(observe_stats.get("queue_entry_update_ms", 0.0))
                self._profile[f"{prefix}_async_turnaround_ms"] += turnaround_ms
                if item["mode"] in {"draft", "verify"}:
                    self._profile[f"run_{item['mode']}_submit_after_ms"] += submit_after_ms
                if processing_origin == "worker":
                    self._profile["prefetch_async_worker_item_count"] += 1
                    self._profile["prefetch_async_worker_wait_ms"] += transfer_wait_ms
                    self._profile["prefetch_async_worker_collect_ms"] += collect_ms
                    self._profile["prefetch_async_worker_observe_ms"] += observe_ms
                    self._profile["prefetch_async_worker_submit_ms"] += submit_after_ms
                    self._profile["prefetch_async_worker_turnaround_ms"] += turnaround_ms

            process_start_ms = float(item["enqueue_ts_ms"]) + float(item["enqueue_ms"])
            self._trace_prefetch_interval(
                name=f"{item['mode']}_metadata_wait",
                start_ms=process_start_ms,
                duration_ms=transfer_wait_ms,
                step_id=int(item["step_id"]),
                mode=str(item["mode"]),
                tid=processing_origin,
            )
            self._trace_prefetch_interval(
                name=f"{item['mode']}_metadata_process",
                start_ms=process_start_ms + transfer_wait_ms,
                duration_ms=collect_ms + observe_ms + submit_after_ms,
                step_id=int(item["step_id"]),
                mode=str(item["mode"]),
                tid=processing_origin,
            )

        return True

    def _prefetch_worker_main(self) -> None:
        while True:
            queue_obj = self._prefetch_worker_queue
            if queue_obj is None:
                return
            item = queue_obj.get()
            if item is None:
                return
            try:
                self._process_prefetch_metadata_item(item, block=True, processing_origin="worker")
            except BaseException as exc:
                with self._prefetch_worker_cv:
                    if self._prefetch_worker_error is None:
                        self._prefetch_worker_error = exc
            finally:
                key = item.get("buffer_key")
                handle = item.get("handle")
                host_buffer_slot = int(item.get("host_buffer_slot", 0))
                with self._prefetch_worker_cv:
                    self._prefetch_worker_outstanding = max(0, self._prefetch_worker_outstanding - 1)
                    if isinstance(key, tuple):
                        slot_key = (*key, host_buffer_slot)
                        remaining = self._prefetch_worker_key_counts.get(slot_key, 0) - 1
                        if remaining > 0:
                            self._prefetch_worker_key_counts[slot_key] = remaining
                        else:
                            self._prefetch_worker_key_counts.pop(slot_key, None)
                        device_handles = [x for x in self._prefetch_device_handles.get(key, []) if x is not handle]
                        if device_handles:
                            self._prefetch_device_handles[key] = device_handles
                        else:
                            self._prefetch_device_handles.pop(key, None)
                    self._prefetch_worker_cv.notify_all()

    def _shutdown_prefetch_worker(self) -> None:
        self._ensure_prefetch_internal_state()
        if not getattr(self, "_prefetch_async_enabled", False):
            return
        queue_obj = getattr(self, "_prefetch_worker_queue", None)
        worker = getattr(self, "_prefetch_worker_thread", None)
        if queue_obj is not None:
            queue_obj.put(None)
        if worker is not None:
            worker.join(timeout=5.0)
        self._prefetch_worker_queue = None
        self._prefetch_worker_thread = None

    def _enqueue_prefetch_metadata(
        self,
        *,
        mode: str,
        step_id: int,
        handle,
        enqueue_ms: float,
        host_buffer_slot: int = 0,
        submit_after_phase: str | None = None,
        record_verify_consumed: bool = False,
    ) -> None:
        self._ensure_prefetch_internal_state()
        if handle is None:
            return
        item = {
            "mode": mode,
            "step_id": int(step_id),
            "handle": handle,
            "enqueue_ms": float(enqueue_ms),
            "enqueue_ts_ms": perf_counter() * 1000.0,
            "submit_after_phase": submit_after_phase,
            "record_verify_consumed": bool(record_verify_consumed),
            "buffer_key": self._prefetch_worker_key(mode=mode, handle=handle),
            "host_buffer_slot": int(host_buffer_slot),
        }
        if getattr(self, "_prefetch_async_enabled", False):
            self._raise_prefetch_worker_error()
            queue_obj = getattr(self, "_prefetch_worker_queue", None)
            if queue_obj is None:
                raise RuntimeError("Prefetch async worker queue is not initialized")
            with self._prefetch_worker_cv:
                self._prefetch_worker_outstanding += 1
                self._prefetch_worker_key_counts[
                    (*item["buffer_key"], int(item["host_buffer_slot"]))
                ] += 1
                self._prefetch_device_handles[item["buffer_key"]].append(handle)
                queue_depth = self._prefetch_worker_outstanding
            if self.profile_enabled and self.rank == 0:
                with self._prefetch_profile_lock:
                    self._profile["prefetch_async_enqueue_count"] += 1
                    self._profile["prefetch_async_queue_depth_sum"] += queue_depth
                    self._profile["prefetch_async_queue_depth_max"] = max(
                        float(self._profile.get("prefetch_async_queue_depth_max", 0.0)),
                        float(queue_depth),
                    )
            queue_obj.put(item)
            return
        pending = getattr(self, "_pending_prefetch_metadata", None)
        if pending is None:
            pending = []
            self._pending_prefetch_metadata = pending
        pending.append(item)

    def _flush_pending_prefetch_metadata(self, block: bool) -> None:
        self._ensure_prefetch_internal_state()
        self._raise_prefetch_worker_error()
        if getattr(self, "_prefetch_async_enabled", False):
            if block:
                self._wait_for_prefetch_async_drain()
            return
        prefetch_runtime = getattr(self, "prefetch_runtime", None)
        runtime_meta_recorder = getattr(self, "runtime_meta_recorder", None)
        if prefetch_runtime is None or runtime_meta_recorder is None:
            pending = getattr(self, "_pending_prefetch_metadata", None)
            if pending is not None:
                pending.clear()
            return

        pending: list[dict[str, object]] = []
        for item in getattr(self, "_pending_prefetch_metadata", []):
            if not self._process_prefetch_metadata_item(item, block=block, processing_origin="foreground"):
                pending.append(item)

        self._pending_prefetch_metadata = pending

    def get_profile(self, reset: bool = False) -> dict:
        self._ensure_prefetch_internal_state()
        if self.rank != 0:
            return {}
        self._flush_pending_prefetch_metadata(block=True)
        with self._prefetch_profile_lock:
            out = {k: (int(v) if k.endswith("_count") else float(v)) for k, v in self._profile.items()}
        decode_count = int(self._profile.get("decode_count", 0))
        graph_hit_count = int(self._profile.get("graph_hit_count", 0))
        out["graph_hit_rate"] = float(graph_hit_count / decode_count) if decode_count > 0 else 0.0
        profile_count = float(self._profile.get("moe_profile_count", 0.0))
        if profile_count > 0:
            out["cpu_route_ratio"] = float(self._profile.get("cpu_route_ratio_sum", 0.0) / profile_count)
            out["cpu_weight_mass_ratio"] = float(self._profile.get("cpu_weight_mass_ratio_sum", 0.0) / profile_count)
            out["activated_expert_set_size"] = float(self._profile.get("activated_expert_set_size_sum", 0.0) / profile_count)
            out["realized_cpu_expert_count"] = float(self._profile.get("realized_cpu_expert_count_sum", 0.0) / profile_count)
        else:
            out["cpu_route_ratio"] = 0.0
            out["cpu_weight_mass_ratio"] = 0.0
            out["activated_expert_set_size"] = 0.0
            out["realized_cpu_expert_count"] = 0.0
        prefetch_runtime = getattr(self, "prefetch_runtime", None)
        if prefetch_runtime is not None:
            with self._prefetch_runtime_lock:
                out.update(prefetch_runtime.get_profile(reset=False))
        out["prefetch_async_enabled"] = bool(getattr(self, "_prefetch_async_enabled", False))
        out["prefetch_trace_events"] = list(getattr(self, "_prefetch_trace_events", []))
        worker_turnaround_ms = float(out.get("prefetch_async_worker_turnaround_ms", 0.0))
        exposed_ms = float(out.get("prefetch_async_buffer_reuse_wait_ms", 0.0)) + float(out.get("prefetch_async_drain_wait_ms", 0.0))
        out["prefetch_async_exposed_wait_ms"] = exposed_ms
        out["prefetch_async_hidden_ms"] = max(0.0, worker_turnaround_ms - exposed_ms)
        out["prefetch_async_hidden_ratio"] = float(out["prefetch_async_hidden_ms"] / worker_turnaround_ms) if worker_turnaround_ms > 0.0 else 0.0
        if reset:
            with self._prefetch_profile_lock:
                self._profile.clear()
                self._prefetch_trace_events.clear()
            pending = getattr(self, "_pending_prefetch_metadata", None)
            if pending is not None:
                pending.clear()
            if prefetch_runtime is not None:
                with self._prefetch_runtime_lock:
                    _ = prefetch_runtime.get_profile(reset=True)
        return out

    def _run_model_eager(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        logits = self.model.compute_logits(self.model(input_ids, positions))
        if hasattr(self.model, "get_and_reset_heterogeneous_profile") and self.rank == 0:
            prof = self.model.get_and_reset_heterogeneous_profile()
            for key, value in prof.items():
                self._profile[key] = float(self._profile.get(key, 0.0) + value)
        return logits

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * hf_config.head_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, hf_config.head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue

            # Build slot mapping token-by-token so non-block-aligned cached prefixes
            # (used by speculative verify) remain consistent with Q token count.
            for token_pos in range(seq.num_cached_tokens, seqlen):
                block_idx = token_pos // self.block_size
                offset_in_block = token_pos % self.block_size
                slot = seq.block_table[block_idx] * self.block_size + offset_in_block
                slot_mapping.append(slot)
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self._run_model_eager(input_ids, positions)

        if self._decode_graph_policy == "draft":
            if self._can_use_draft_cudagraph(input_ids.size(0)):
                return self._replay_draft_graph(input_ids, positions)
            # Correctness-first: draft mode must not replay standard decode graph.
            return self._run_model_eager(input_ids, positions)

        if self._can_use_standard_cudagraph(input_ids.size(0)):
            return self._replay_standard_graph(input_ids, positions)
        return self._run_model_eager(input_ids, positions)

    def _replay_standard_graph(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        bs = input_ids.size(0)
        context = get_context()
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        graph_vars = self.graph_vars
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        graph.replay()
        if self.profile_enabled and self.rank == 0:
            self._profile["graph_replay_count"] += 1
            self._profile["graph_hit_count"] += 1
            self._profile["standard_graph_replay_count"] += 1
        return self.model.compute_logits(graph_vars["outputs"][:bs])

    def _replay_draft_graph(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        bs = input_ids.size(0)
        context = get_context()
        graph = self.draft_graphs[next(x for x in self.draft_graph_bs if x >= bs)]
        graph_vars = self.draft_graph_vars
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        graph.replay()
        if self.profile_enabled and self.rank == 0:
            self._profile["graph_replay_count"] += 1
            self._profile["graph_hit_count"] += 1
            self._profile["draft_graph_replay_count"] += 1
        return self.model.compute_logits(graph_vars["outputs"][:bs])

    def _can_use_draft_cudagraph(self, bs: int) -> bool:
        if not getattr(self.config, "draft_cuda_graph_enabled", True):
            return False
        if self.enforce_eager:
            return False
        if getattr(self.config, "draft_top_c", 0) != 0:
            return False
        if bs > getattr(self.config, "draft_cuda_graph_max_bs", 512):
            return False
        if not hasattr(self, "draft_graphs") or not self.draft_graphs:
            return False
        return any(bucket >= bs for bucket in self.draft_graph_bs)

    def _can_use_standard_cudagraph(self, bs: int) -> bool:
        if self.enforce_eager:
            return False
        if not hasattr(self, "graphs") or not self.graphs:
            return False
        return any(bucket >= bs for bucket in self.graph_bs)

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        phase = "prefill" if is_prefill else "decode"
        t0 = perf_counter()
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        self._record_profile("prepare_prefill_ms" if is_prefill else "prepare_decode_ms", perf_counter() - t0)
        prefetch_runtime = getattr(self, "prefetch_runtime", None)
        runtime_meta_recorder = getattr(self, "runtime_meta_recorder", None)
        prefill_step_id = None
        if prefetch_runtime is not None and runtime_meta_recorder is not None:
            self._flush_pending_prefetch_metadata(block=False)
        if is_prefill and prefetch_runtime is not None and runtime_meta_recorder is not None:
            prefill_step_id = self._next_prefetch_step_id()
            token_count = int(input_ids.numel())
            self._wait_for_prefetch_device_reuse(mode="prefill", token_capacity=token_count)
            runtime_meta_recorder.arm(
                mode="prefill",
                step_id=prefill_step_id,
                token_capacity=token_count,
                logical_token_count=token_count,
            )
        if self.profile_enabled and self.rank == 0:
            self._profile["prepare_count"] += 1
            self._profile["prefill_count"] += int(is_prefill)
            self._profile["decode_count"] += int(not is_prefill)

        t0 = perf_counter()
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        dt = perf_counter() - t0
        self._record_profile("prepare_sample_ms", dt)
        self._record_profile(f"prepare_sample_{phase}_ms", dt)

        t0 = perf_counter()
        logits = self.run_model(input_ids, positions, is_prefill)
        if self.profile_enabled and self.profile_cuda_sync:
            torch.cuda.synchronize()
        dt = perf_counter() - t0
        self._record_profile("run_model_ms", dt)
        self._record_profile(f"run_model_{phase}_ms", dt)

        t0 = perf_counter()
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        if self.profile_enabled and self.profile_cuda_sync:
            torch.cuda.synchronize()
        dt = perf_counter() - t0
        self._record_profile("sample_ms", dt)
        self._record_profile(f"sample_{phase}_ms", dt)

        if self.profile_enabled and self.rank == 0:
            self._profile["tokens_in_total"] += int(input_ids.numel())
            self._profile["run_count"] += 1

        if is_prefill and prefetch_runtime is not None and runtime_meta_recorder is not None and prefill_step_id is not None:
            host_buffer_slot, _ = self._acquire_prefetch_host_buffer_slot(
                mode="prefill",
                token_capacity=int(input_ids.numel()),
            )
            enqueue_t0 = perf_counter()
            handle = runtime_meta_recorder.offload_async(
                prefetch_runtime.metadata_stream,
                host_buffer_slot=host_buffer_slot,
            )
            enqueue_ms = (perf_counter() - enqueue_t0) * 1000.0
            self._enqueue_prefetch_metadata(
                mode="prefill",
                step_id=prefill_step_id,
                handle=handle,
                enqueue_ms=enqueue_ms,
                host_buffer_slot=host_buffer_slot,
            )
            runtime_meta_recorder.reset()
            self._flush_pending_prefetch_metadata(block=False)

        reset_context()
        return token_ids

    def _set_speculative_execution_mode(self, mode: str):
        if hasattr(self.model, "set_speculative_execution_mode"):
            draft_top_c = getattr(self.config, "draft_top_c", 0)
            self.model.set_speculative_execution_mode(mode, self.draft_scheduler, draft_top_c)

    @torch.inference_mode()
    def run_draft(self, seqs: list[Sequence]) -> tuple[list[int], dict[str, object]]:
        """Draft decode path with explicit draft plan execution inside MoE blocks."""
        self._ensure_prefetch_internal_state()
        t0 = perf_counter()
        step_id = self._next_prefetch_step_id()
        prefetch_runtime = getattr(self, "prefetch_runtime", None)
        runtime_meta_recorder = getattr(self, "runtime_meta_recorder", None)
        if prefetch_runtime is not None and runtime_meta_recorder is not None:
            self._flush_pending_prefetch_metadata(block=False)

        mode_set_t0 = perf_counter()
        self._set_speculative_execution_mode("draft")
        self._decode_graph_policy = "draft"
        mode_set_ms = (perf_counter() - mode_set_t0) * 1000.0
        if self.profile_enabled and self.rank == 0:
            self._profile["run_draft_mode_set_ms"] += mode_set_ms

        try:
            prefetch_before_ms = 0.0
            if prefetch_runtime is not None and runtime_meta_recorder is not None:
                before_t0 = perf_counter()
                draft_capacity = len(seqs)
                if self._can_use_draft_cudagraph(len(seqs)):
                    draft_capacity = next(x for x in self.draft_graph_bs if x >= len(seqs))

                with self._prefetch_runtime_lock:
                    prefetch_runtime.publish_ready(step_id=step_id)
                    prefetch_runtime.submit_from_global_queue(step_id=step_id, phase="before_draft")
                self._wait_for_prefetch_device_reuse(mode="draft", token_capacity=draft_capacity)
                runtime_meta_recorder.arm(
                    mode="draft",
                    step_id=step_id,
                    token_capacity=draft_capacity,
                    logical_token_count=len(seqs),
                )
                prefetch_before_ms = (perf_counter() - before_t0) * 1000.0
                if self.profile_enabled and self.rank == 0:
                    with self._prefetch_profile_lock:
                        self._profile["run_draft_prefetch_before_ms"] += prefetch_before_ms
                    self._trace_prefetch_interval(
                        name="run_draft_prefetch_before",
                        start_ms=before_t0 * 1000.0,
                        duration_ms=prefetch_before_ms,
                        step_id=step_id,
                        mode="draft",
                        tid="main",
                    )

            core_run_t0 = perf_counter()
            token_ids = self.run(seqs, False)
            core_run_ms = (perf_counter() - core_run_t0) * 1000.0
            if self.profile_enabled and self.rank == 0:
                with self._prefetch_profile_lock:
                    self._profile["run_draft_core_run_ms"] += core_run_ms
                self._trace_prefetch_interval(
                    name="run_draft_core_run",
                    start_ms=core_run_t0 * 1000.0,
                    duration_ms=core_run_ms,
                    step_id=step_id,
                    mode="draft",
                    tid="main",
                )

            if prefetch_runtime is not None and runtime_meta_recorder is not None:
                host_buffer_slot, _ = self._acquire_prefetch_host_buffer_slot(
                    mode="draft",
                    token_capacity=draft_capacity,
                )
                enqueue_t0 = perf_counter()
                handle = runtime_meta_recorder.offload_async(
                    prefetch_runtime.metadata_stream,
                    host_buffer_slot=host_buffer_slot,
                )
                enqueue_ms = (perf_counter() - enqueue_t0) * 1000.0
                self._enqueue_prefetch_metadata(
                    mode="draft",
                    step_id=step_id,
                    handle=handle,
                    enqueue_ms=enqueue_ms,
                    host_buffer_slot=host_buffer_slot,
                    submit_after_phase="after_draft",
                )
                runtime_meta_recorder.reset()
                self._flush_pending_prefetch_metadata(block=False)
            return token_ids, {"prefetch_step_id": step_id}
        finally:
            self._decode_graph_policy = "standard"
            self._set_speculative_execution_mode("normal")
            if self.profile_enabled:
                if self.profile_cuda_sync:
                    torch.cuda.synchronize()
                self._record_profile("run_draft_total_ms", perf_counter() - t0)
                if self.rank == 0:
                    self._profile["run_draft_count"] += 1

    def wait_prefetch_for_verify(self, step_id: int) -> dict[str, float]:
        self._ensure_prefetch_internal_state()
        prefetch_runtime = getattr(self, "prefetch_runtime", None)
        if prefetch_runtime is None:
            return {}
        metadata_drain_ms = self._wait_for_prefetch_async_drain() if getattr(self, "_prefetch_async_enabled", False) else 0.0
        self._flush_pending_prefetch_metadata(block=True)
        t0 = perf_counter()
        with self._prefetch_runtime_lock:
            prefetch_runtime.wait_for_verify(
                step_id=step_id,
                timeout_ms=float(self.config.prefetch_verify_wait_ms),
            )
        verify_wait_ms = (perf_counter() - t0) * 1000.0
        self._trace_prefetch_interval(
            name="verify_prefetch_wait",
            start_ms=t0 * 1000.0,
            duration_ms=verify_wait_ms,
            step_id=step_id,
            mode="verify",
            tid="main",
        )
        return {
            "verify_prefetch_metadata_drain_ms": metadata_drain_ms,
            "verify_prefetch_wait_ms": verify_wait_ms,
        }

    @torch.inference_mode()
    def run_verify(self, seqs: list[Sequence], verify_lengths: list[int]) -> list[list[int]]:
        """Run one-shot verify in prefill-like mode and return per-sequence argmax traces."""
        self._ensure_prefetch_internal_state()
        total_t0 = perf_counter()
        step_id = self._next_prefetch_step_id()
        prefetch_runtime = getattr(self, "prefetch_runtime", None)
        runtime_meta_recorder = getattr(self, "runtime_meta_recorder", None)
        if prefetch_runtime is not None and runtime_meta_recorder is not None:
            self._flush_pending_prefetch_metadata(block=False)
        self._set_speculative_execution_mode("verify")
        t0 = perf_counter()
        input_ids, positions = self.prepare_prefill(seqs)
        self._record_profile("verify_prepare_prefill_ms", perf_counter() - t0)

        if prefetch_runtime is not None and runtime_meta_recorder is not None:
            token_count = int(input_ids.numel())
            self._wait_for_prefetch_device_reuse(mode="verify", token_capacity=token_count)
            runtime_meta_recorder.arm(
                mode="verify",
                step_id=step_id,
                token_capacity=token_count,
                logical_token_count=token_count,
            )

        # compute_logits() slices prefill outputs to last token per sequence.
        # Verify needs logits for every queried token position.
        try:
            t0 = perf_counter()
            profile_dir = os.getenv("NANOVLLM_VERIFY_TORCH_PROFILE_DIR", "").strip()
            capture_verify_profile = (
                bool(profile_dir)
                and self.rank == 0
                and not bool(getattr(self, "_verify_torch_profile_done", False))
            )
            if capture_verify_profile:
                os.makedirs(profile_dir, exist_ok=True)
                from torch.profiler import ProfilerActivity, profile

                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
                    hidden_states = self.model(input_ids, positions)
                    logits = F.linear(hidden_states, self.model.lm_head.weight)
                torch.cuda.synchronize()
                trace_path = os.path.join(profile_dir, f"verify_forward_rank{self.rank}.json")
                summary_path = os.path.join(profile_dir, f"verify_forward_rank{self.rank}_summary.json")
                prof.export_chrome_trace(trace_path)
                events = []
                for evt in prof.key_averages():
                    events.append(
                        {
                            "key": evt.key,
                            "count": int(evt.count),
                            "self_cpu_time_total_us": float(evt.self_cpu_time_total),
                            "cpu_time_total_us": float(evt.cpu_time_total),
                            "self_cuda_time_total_us": float(getattr(evt, "self_cuda_time_total", 0.0)),
                            "cuda_time_total_us": float(getattr(evt, "cuda_time_total", 0.0)),
                        }
                    )
                events_by_cuda = sorted(events, key=lambda x: x["self_cuda_time_total_us"], reverse=True)
                events_by_cpu = sorted(events, key=lambda x: x["self_cpu_time_total_us"], reverse=True)
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "trace_path": trace_path,
                            "verify_tokens": int(input_ids.numel()),
                            "top_self_cuda_events": events_by_cuda[:40],
                            "top_self_cpu_events": events_by_cpu[:40],
                        },
                        f,
                        ensure_ascii=True,
                        indent=2,
                    )
                self._verify_torch_profile_done = True
            else:
                hidden_states = self.model(input_ids, positions)
                logits = F.linear(hidden_states, self.model.lm_head.weight)
            if hasattr(self.model, "get_and_reset_heterogeneous_profile") and self.rank == 0:
                prof = self.model.get_and_reset_heterogeneous_profile()
                for key, value in prof.items():
                    value = float(value)
                    self._profile[key] = float(self._profile.get(key, 0.0) + value)
                    self._profile[f"verify_{key}"] = float(self._profile.get(f"verify_{key}", 0.0) + value)
            if self.world_size > 1:
                if self.rank == 0:
                    all_logits = [torch.empty_like(logits) for _ in range(self.world_size)]
                    dist.gather(logits, all_logits, 0)
                    logits = torch.cat(all_logits, dim=-1)
                else:
                    dist.gather(logits, None, 0)
            if self.profile_enabled and self.profile_cuda_sync:
                torch.cuda.synchronize()
            self._record_profile("verify_forward_ms", perf_counter() - t0)
        finally:
            self._set_speculative_execution_mode("normal")

        reset_context()

        if self.rank != 0:
            return None

        if self.profile_enabled:
            self._record_profile("run_verify_total_ms", perf_counter() - total_t0)
            self._profile["run_verify_count"] += 1
            self._profile["verify_tokens_in_total"] += int(input_ids.numel())

        verify_tokens_per_seq: list[list[int]] = []
        offset = 0
        for length in verify_lengths:
            seq_logits = logits[offset:offset + length]
            offset += length
            verify_tokens_per_seq.append(seq_logits.argmax(dim=-1).tolist())

        if prefetch_runtime is not None and runtime_meta_recorder is not None:
            host_buffer_slot, _ = self._acquire_prefetch_host_buffer_slot(
                mode="verify",
                token_capacity=int(input_ids.numel()),
            )
            enqueue_t0 = perf_counter()
            handle = runtime_meta_recorder.offload_async(
                prefetch_runtime.metadata_stream,
                host_buffer_slot=host_buffer_slot,
            )
            enqueue_ms = (perf_counter() - enqueue_t0) * 1000.0
            self._enqueue_prefetch_metadata(
                mode="verify",
                step_id=step_id,
                handle=handle,
                enqueue_ms=enqueue_ms,
                host_buffer_slot=host_buffer_slot,
                submit_after_phase="after_verify",
                record_verify_consumed=True,
            )
            runtime_meta_recorder.reset()
            self._flush_pending_prefetch_metadata(block=False)
        return verify_tokens_per_seq

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

    @torch.inference_mode()
    def capture_draft_cudagraph(self):
        self.draft_graphs = {}
        self.draft_graph_pool = None
        self.draft_graph_bs = []

        if not getattr(self.config, "draft_cuda_graph_enabled", True):
            return
        if getattr(self.config, "draft_top_c", 0) != 0:
            # Graph-safe subset: draft graph capture is only enabled for top_c == 0.
            return

        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, getattr(config, "draft_cuda_graph_max_bs", 512))
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)

        bucket_steps = sorted(set(int(x) for x in getattr(config, "draft_cuda_graph_bucket_steps", [1, 2, 4, 8]) if int(x) >= 1))
        self.draft_graph_bs = [x for x in bucket_steps if x <= max_bs]
        if max_bs >= 16:
            self.draft_graph_bs += list(range(16, max_bs + 1, 16))
        self.draft_graph_bs = sorted(set(self.draft_graph_bs))

        if not self.draft_graph_bs:
            return

        self._set_speculative_execution_mode("draft")
        runtime_meta_recorder = getattr(self, "runtime_meta_recorder", None)
        try:
            for bs in reversed(self.draft_graph_bs):
                graph = torch.cuda.CUDAGraph()
                set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
                if runtime_meta_recorder is not None:
                    runtime_meta_recorder.arm(
                        mode="draft",
                        step_id=-1,
                        token_capacity=bs,
                        logical_token_count=bs,
                    )
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
                with torch.cuda.graph(graph, self.draft_graph_pool):
                    outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
                if self.draft_graph_pool is None:
                    self.draft_graph_pool = graph.pool()
                self.draft_graphs[bs] = graph
                torch.cuda.synchronize()
                if runtime_meta_recorder is not None:
                    runtime_meta_recorder.reset()
                reset_context()
        finally:
            self._set_speculative_execution_mode("normal")

        self.draft_graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
