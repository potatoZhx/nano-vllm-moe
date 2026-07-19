import pickle
import os
import json
import importlib.metadata
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
from nanovllm.scheduling.cache_strategy import LFURankGuardStrategy, create_cache_strategy
from nanovllm.scheduling.prefetch_strategy import create_prefetch_strategy
from nanovllm.scheduling.draft_reroute import SIMILARITY_REPLACE
from nanovllm.scheduling.draft_reroute_profile import (
    load_draft_reroute_profile,
    seed_lfu_rank_guard_from_profile,
)
from nanovllm.expert.placement import (
    apply_verify_cache_fill_no_cpu_policy_ids,
    apply_verify_cache_fill_policy,
    collect_cache_fill_no_cpu_expert_ids,
)
from nanovllm.expert.prefetcher import (
    DualQueuePrefetchRuntime,
    PredictivePrefetchRuntime,
    PrefetchRuntime,
)
from nanovllm.expert.runtime_meta import ModelRuntimeMetaRecorder
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.utils.heterogeneous_loader import HeterogeneousModelLoader
from nanovllm.layers.fuse_moe.heterogeneous import GpuFallbackWorkspace


def _export_torch_profile_summary(prof, profile_dir: str, stem: str, rank: int, extra: dict) -> None:
    trace_path = os.path.join(profile_dir, f"{stem}_rank{rank}.json")
    summary_path = os.path.join(profile_dir, f"{stem}_rank{rank}_summary.json")
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
                **extra,
                "top_self_cuda_events": events_by_cuda[:80],
                "top_self_cpu_events": events_by_cpu[:80],
            },
            f,
            ensure_ascii=True,
            indent=2,
        )


def _cpu_model_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def _create_cache_strategy_from_config(config: Config):
    if str(config.cache_strategy).strip().lower() in {"lfu_rankguard", "lfu_rankguard_online"}:
        return create_cache_strategy(
            config.cache_strategy,
            num_experts=int(getattr(config.hf_config, "num_experts", 128)),
            protect_threshold=float(getattr(config, "rank_guard_threshold", 0.15)),
            ema_alpha=float(getattr(config, "rank_guard_ema_alpha", 0.95)),
        )
    return create_cache_strategy(config.cache_strategy)


def _create_gpu_fallback_workspace(
    cpu_expert_pool: dict[int, dict[int, dict[str, torch.Tensor]]],
    hf_config,
) -> GpuFallbackWorkspace | None:
    """Create a single shared GPU fallback workspace sized for all experts."""
    if not cpu_expert_pool:
        return None
    first_layer_pool = next(iter(cpu_expert_pool.values()))
    if not first_layer_pool:
        return None
    sample = next(iter(first_layer_pool.values()))
    gate_up_shape = tuple(sample["gate_up"].shape)
    down_shape = tuple(sample["down"].shape)
    num_experts = getattr(hf_config, "num_experts", 0)
    if num_experts <= 0 and hasattr(hf_config, "num_experts_per_tok"):
        num_experts = len(first_layer_pool)
    if num_experts <= 0:
        num_experts = len(first_layer_pool)
    return GpuFallbackWorkspace(
        max_experts=num_experts,
        gate_up_shape=gate_up_shape,
        down_shape=down_shape,
        device=torch.device("cuda"),
        dtype=hf_config.torch_dtype,
    )


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
        self.cache_strategy = _create_cache_strategy_from_config(config)
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
        self._verify_boundary_worker_queue: Queue | None = None
        self._verify_boundary_worker_cv = threading.Condition()
        self._verify_boundary_worker_outstanding = 0
        self._verify_boundary_worker_error: BaseException | None = None
        self._verify_boundary_worker_thread: threading.Thread | None = None
        self._prefetch_trace_events: list[dict[str, object]] = []
        self._verify_prefetch_active = False
        self._current_verify_prefetch_step_id = -1
        self._verify_layer_compute_ms_ema: dict[int, float] = {}
        self._verify_layer_timing_events: list[tuple[int, torch.cuda.Event, torch.cuda.Event]] = []
        self._verify_layer_active_timing: dict[int, object] = {}
        self._dual_queue_segment_timing_events: list[tuple[str, int, object, object]] = []
        self._verify_op_event_records: list[dict[str, object]] = []
        self._draft_op_event_records: list[dict[str, object]] = []
        self._verify_call_records: list[dict[str, object]] = []
        self._verify_metadata_records_by_step: dict[int, dict[str, object]] = {}
        self._verify_stream_timing_events: list[tuple[int, object, object]] = []
        self._verify_stream_ms_by_step: dict[int, float] = {}
        self._verify_cost_model = None
        self._verify_cost_proxy = None
        self._verify_cost_schema_version = 0
        self._verify_cost_round_active = False
        self._verify_cost_latest_prediction: dict[str, object] | None = None
        self._transfer_aware_profile_routes: list[object] = []
        self._active_draft_prefetch_step_id = -1
        self._draft_segment_metadata_enqueued_step_id = -1
        self._draft_perfect_trace_enabled = os.getenv(
            "NANOVLLM_DRAFT_PERFECT_MATCH_TRACE", "0"
        ).strip().lower() in {"1", "true", "yes", "y", "on"}
        self._draft_perfect_refill_rejected = os.getenv(
            "NANOVLLM_DRAFT_PERFECT_MATCH_REFILL_REJECTED", "0"
        ).strip().lower() in {"1", "true", "yes", "y", "on"}
        self._draft_perfect_detail_limit = int(
            os.getenv("NANOVLLM_DRAFT_PERFECT_MATCH_DETAIL_LIMIT", "256") or "256"
        )
        self._draft_perfect_profile = defaultdict(float)
        self._draft_perfect_records: list[dict[str, object]] = []
        self._draft_perfect_last_verify_meta: dict[int, dict[str, torch.Tensor]] | None = None
        self._draft_perfect_pending: dict[str, object] | None = None
        self._draft_perfect_draft_meta_by_step: dict[int, dict[int, dict[str, torch.Tensor]]] = {}

        dist_url = f"tcp://localhost:{config.dist_port}"
        dist.init_process_group("nccl", dist_url, world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        setattr(hf_config, "enable_heterogeneous", config.enable_heterogeneous)
        self.model = self.MODEL_TYPE_DICT[hf_config.model_type](hf_config)
        if config.enable_heterogeneous and hasattr(self.model, "enable_heterogeneous_mode"):
            draft_reroute_profile = None
            draft_reroute_artifact_path = getattr(config, "draft_reroute_artifact", "")
            if draft_reroute_artifact_path:
                draft_reroute_profile = load_draft_reroute_profile(
                    draft_reroute_artifact_path,
                    num_experts=int(getattr(hf_config, "num_experts")),
                    expected_top_k=int(
                        getattr(
                            hf_config,
                            "num_experts_per_tok",
                            getattr(hf_config, "top_k", 0),
                        )
                    )
                    or None,
                    hf_config=hf_config,
                )
            if getattr(config, "draft_reroute_policy", "round_robin") == SIMILARITY_REPLACE:
                if (
                    draft_reroute_profile is None
                    or draft_reroute_profile.cond_sim is None
                    or draft_reroute_profile.skip_err is None
                ):
                    raise ValueError("similarity_replace requires cond_sim and skip_err calibration tensors")

            loader = HeterogeneousModelLoader(config, draft_reroute_profile=draft_reroute_profile)
            layer_caches, cpu_expert_pool = loader.load(self.model, config.model)
            self.layer_caches = layer_caches
            self.cpu_expert_pool = cpu_expert_pool
            if isinstance(self.cache_strategy, LFURankGuardStrategy):
                seed_lfu_rank_guard_from_profile(
                    self.cache_strategy,
                    draft_reroute_profile,
                    layer_indices=sorted(self.layer_caches),
                    top_k=int(getattr(hf_config, "num_experts_per_tok", getattr(hf_config, "top_k", 1))),
                )
            # GPU fallback workspace: only needed when CPU expert execution is
            # disabled and uncached experts run on GPU for kernel alignment.
            if not getattr(config, "cpu_expert_execution_enabled", False):
                gpu_fallback_workspace = _create_gpu_fallback_workspace(
                    cpu_expert_pool, hf_config
                )
            else:
                gpu_fallback_workspace = None
            self.model.enable_heterogeneous_mode(
                layer_caches,
                cpu_expert_pool,
                cpu_expert_execution_enabled=getattr(config, "cpu_expert_execution_enabled", False),
                cpu_expert_parallel_mode=getattr(config, "cpu_expert_parallel_mode", "serial"),
                cpu_expert_num_threads=getattr(config, "cpu_expert_num_threads", 4),
                cpu_expert_backend=getattr(config, "cpu_expert_backend", "torch"),
                draft_cuda_graph_cpu_backend=getattr(config, "draft_cuda_graph_cpu_backend", "none"),
                cpu_expert_workspace_max_routes=getattr(config, "cpu_expert_workspace_max_routes", 8192),
                cpu_expert_packed_min_routes=getattr(config, "cpu_expert_packed_min_routes", 32),
                cpu_expert_strict_dtype=getattr(config, "cpu_expert_strict_dtype", True),
                cpu_gpu_parallel_execution_enabled=getattr(config, "cpu_gpu_parallel_execution_enabled", "auto"),
                cpu_gpu_parallel_min_cpu_route_ratio=getattr(config, "cpu_gpu_parallel_min_cpu_route_ratio", 0.0),
                spec_verify_miss_policy=getattr(config, "spec_verify_miss_policy", "cpu"),
                cache_strategy=getattr(config, "cache_strategy", "lru"),
                gpu_fallback_workspace=gpu_fallback_workspace,
                kt_weight_path=getattr(config, "model", ""),
                kt_method=getattr(config, "kt_method", "BF16"),
                kt_num_threads=getattr(config, "kt_num_threads", 0),
                kt_threadpool_count=getattr(config, "kt_threadpool_count", 1),
                kt_chunked_prefill_size=getattr(config, "kt_chunked_prefill_size", 4096),
                kt_direct_backend=getattr(config, "kt_direct_backend", "auto"),
                kt_numa_nodes=getattr(config, "kt_numa_nodes", None) or None,
                kt_capture_bs=getattr(config, "kt_capture_bs", None),
                draft_reroute_policy=getattr(config, "draft_reroute_policy", "round_robin"),
                draft_reroute_profile=draft_reroute_profile,
            )
            self.draft_scheduler = create_draft_scheduler(getattr(config, "draft_scheduler", "simple"))
            self.prefetch_effective_enabled = bool(config.spec_enable_prefetch) and config.inference_mode == "spec"

            if self.prefetch_effective_enabled:
                self.runtime_meta_recorder = ModelRuntimeMetaRecorder(config=config, hf_config=config.hf_config)
                runtime_kind = str(getattr(config, "prefetch_runtime_kind", "legacy"))
                prefetch_cls = {
                    "legacy": PrefetchRuntime,
                    "predictive": PredictivePrefetchRuntime,
                    "dual_queue": DualQueuePrefetchRuntime,
                }[runtime_kind]
                self.prefetch_runtime = prefetch_cls(
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
                    if self._verify_boundary_prefetch_async_enabled():
                        self._verify_boundary_worker_queue = Queue()
                        self._verify_boundary_worker_thread = threading.Thread(
                            target=self._verify_boundary_worker_main,
                            name=f"verify-boundary-prefetch-rank{self.rank}",
                            daemon=True,
                        )
                        self._verify_boundary_worker_thread.start()
        else:
            load_model(self.model, config.model)
            self.draft_scheduler = None

        # On-GPU acceptance predictor for the draft segment-graph path (rank 0 only;
        # sampling/acceptance happen on rank 0). Off unless explicitly enabled.
        self._acceptance_extractor = None
        self._pending_acceptance = False
        if getattr(config, "acceptance_predictor_enabled", False) and self.rank == 0:
            from nanovllm.engine.speculative.acceptance_predictor import (
                DraftAcceptanceFeatureExtractor,
                load_acceptance_predictor,
            )
            device = torch.device("cuda")
            predictor, pmeta = load_acceptance_predictor(config.acceptance_predictor_path, device)
            if pmeta.num_layers != int(hf_config.num_hidden_layers):
                raise ValueError(
                    f"acceptance predictor num_layers={pmeta.num_layers} != model "
                    f"num_hidden_layers={hf_config.num_hidden_layers}"
                )
            if pmeta.top_k != int(getattr(hf_config, "num_experts_per_tok", -1)):
                raise ValueError(
                    f"acceptance predictor top_k={pmeta.top_k} != model "
                    f"num_experts_per_tok={getattr(hf_config, 'num_experts_per_tok', None)}"
                )
            if pmeta.hidden_dim != int(hf_config.hidden_size):
                raise ValueError(
                    f"acceptance predictor hidden_dim={pmeta.hidden_dim} != model "
                    f"hidden_size={hf_config.hidden_size}"
                )
            pred_max_bs = min(config.max_num_seqs, getattr(config, "draft_cuda_graph_max_bs", 512))
            self._acceptance_extractor = DraftAcceptanceFeatureExtractor(
                predictor,
                pmeta,
                max_bs=pred_max_bs,
                hidden_size=int(hf_config.hidden_size),
                device=device,
                step_horizon=int(getattr(config, "acceptance_predictor_step_horizon", 32)),
            )
            self._acceptance_extractor.attach(self.model)

        self._configure_verify_cost_proxy()

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
            self.capture_verify_cudagraph()
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
        if not hasattr(self, "_verify_boundary_worker_queue"):
            self._verify_boundary_worker_queue = None
        if not hasattr(self, "_verify_boundary_worker_cv"):
            self._verify_boundary_worker_cv = threading.Condition()
        if not hasattr(self, "_verify_boundary_worker_outstanding"):
            self._verify_boundary_worker_outstanding = 0
        if not hasattr(self, "_verify_boundary_worker_error"):
            self._verify_boundary_worker_error = None
        if not hasattr(self, "_verify_boundary_worker_thread"):
            self._verify_boundary_worker_thread = None
        if not hasattr(self, "_prefetch_trace_events"):
            self._prefetch_trace_events = []
        if not hasattr(self, "_verify_prefetch_active"):
            self._verify_prefetch_active = False
        if not hasattr(self, "_current_verify_prefetch_step_id"):
            self._current_verify_prefetch_step_id = -1
        if not hasattr(self, "_verify_layer_compute_ms_ema"):
            self._verify_layer_compute_ms_ema = {}
        if not hasattr(self, "_verify_layer_timing_events"):
            self._verify_layer_timing_events = []
        if not hasattr(self, "_verify_layer_active_timing"):
            self._verify_layer_active_timing = {}
        if not hasattr(self, "_dual_queue_segment_timing_events"):
            self._dual_queue_segment_timing_events = []
        if not hasattr(self, "_verify_op_event_records"):
            self._verify_op_event_records = []
        if not hasattr(self, "_draft_op_event_records"):
            self._draft_op_event_records = []
        if not hasattr(self, "_verify_call_records"):
            self._verify_call_records = []
        if not hasattr(self, "_verify_metadata_records_by_step"):
            self._verify_metadata_records_by_step = {}
        if not hasattr(self, "_verify_stream_timing_events"):
            self._verify_stream_timing_events = []
        if not hasattr(self, "_verify_stream_ms_by_step"):
            self._verify_stream_ms_by_step = {}
        if not hasattr(self, "_verify_cost_model"):
            self._verify_cost_model = None
        if not hasattr(self, "_verify_cost_proxy"):
            self._verify_cost_proxy = None
        if not hasattr(self, "_verify_cost_schema_version"):
            self._verify_cost_schema_version = 0
        if not hasattr(self, "_verify_cost_round_active"):
            self._verify_cost_round_active = False
        if not hasattr(self, "_verify_cost_latest_prediction"):
            self._verify_cost_latest_prediction = None
        if not hasattr(self, "_transfer_aware_profile_routes"):
            self._transfer_aware_profile_routes = []
        if not hasattr(self, "_draft_perfect_trace_enabled"):
            self._draft_perfect_trace_enabled = False
        if not hasattr(self, "_draft_perfect_refill_rejected"):
            self._draft_perfect_refill_rejected = False
        if not hasattr(self, "_draft_perfect_detail_limit"):
            self._draft_perfect_detail_limit = 256
        if not hasattr(self, "_draft_perfect_profile"):
            self._draft_perfect_profile = defaultdict(float)
        if not hasattr(self, "_draft_perfect_records"):
            self._draft_perfect_records = []
        if not hasattr(self, "_draft_perfect_last_verify_meta"):
            self._draft_perfect_last_verify_meta = None
        if not hasattr(self, "_draft_perfect_pending"):
            self._draft_perfect_pending = None
        if not hasattr(self, "_draft_perfect_draft_meta_by_step"):
            self._draft_perfect_draft_meta_by_step = {}

    def _record_profile(self, key: str, dt_sec: float) -> None:
        self._ensure_prefetch_internal_state()
        if self.profile_enabled and self.rank == 0:
            with self._prefetch_profile_lock:
                self._profile[key] += dt_sec * 1000.0

    def _next_prefetch_step_id(self) -> int:
        self._prefetch_step_id += 1
        return self._prefetch_step_id

    def _prefetch_runtime_mode(self) -> str:
        return str(getattr(getattr(self, "config", None), "prefetch_runtime_mode", "baseline_staging"))

    def _dual_queue_prefetch_enabled(self) -> bool:
        return str(getattr(getattr(self, "config", None), "prefetch_runtime_kind", "legacy")) == "dual_queue"

    def _verify_boundary_prefetch_async_enabled(self) -> bool:
        raw = os.getenv("NANOVLLM_VERIFY_BOUNDARY_PREFETCH_ASYNC", "0").strip().lower()
        return raw not in {"0", "false", "no", "n", "off"}

    def _draft_prefetch_granularity(self) -> str:
        return str(getattr(getattr(self, "config", None), "draft_prefetch_frontier_granularity", "segment"))

    def _draft_segment_graph_enabled(self) -> bool:
        return (
            self._prefetch_runtime_mode() in {"draft_direct_active", "draft_segment_indexed"}
            and self._draft_prefetch_granularity() in {"segment", "layer"}
            and hasattr(self.model, "forward_draft_segment")
        )

    def _draft_segment_size(self) -> int:
        if self._draft_prefetch_granularity() == "layer":
            return 1
        return max(1, int(getattr(self.config, "draft_prefetch_segment_size", 12)))

    def _skip_verify_metadata_offload(self) -> bool:
        policy = str(getattr(getattr(self, "config", None), "spec_verify_miss_policy", "")).strip()
        env_skip = os.getenv("NANOVLLM_VERIFY_DISABLE_RUNTIME_METADATA", "").strip().lower() in {
            "1", "true", "yes", "y", "on"
        }
        env_skip = env_skip or os.getenv("NANOVLLM_VERIFY_SKIP_METADATA_OFFLOAD", "").strip().lower() in {
            "1", "true", "yes", "y", "on"
        }
        return env_skip or policy == "cache_fill_no_cpu"

    def _draft_segment_boundaries(self) -> list[tuple[int, int]]:
        num_layers = int(getattr(getattr(self.config, "hf_config", None), "num_hidden_layers", 0))
        if num_layers <= 0:
            layer_caches = getattr(self, "layer_caches", None)
            if layer_caches:
                num_layers = max(int(layer_idx) for layer_idx in layer_caches.keys()) + 1
        if num_layers <= 0:
            return []
        segment_size = self._draft_segment_size()
        return [(start, min(start + segment_size, num_layers)) for start in range(0, num_layers, segment_size)]

    def _verify_segment_size(self) -> int:
        return max(1, int(getattr(self.config, "verify_prefetch_segment_size", 12)))

    def _verify_segment_boundaries(self) -> list[tuple[int, int]]:
        num_layers = int(getattr(getattr(self.config, "hf_config", None), "num_hidden_layers", 0))
        if num_layers <= 0:
            return []
        segment_size = self._verify_segment_size()
        return [(s, min(s + segment_size, num_layers)) for s in range(0, num_layers, segment_size)]

    def _verify_segment_graph_enabled(self) -> bool:
        if not getattr(self.config, "verify_cuda_graph_kt_hybrid", False):
            return False
        return len(self._verify_segment_boundaries()) > 1

    def _draft_prefetch_frontier_layer_idx(self) -> int | None:
        layer_caches = getattr(self, "layer_caches", None)
        if not layer_caches:
            return None
        # Full-draft metadata is only submitted after replay, so every layer is
        # already behind the safety frontier. Segment metadata passes an
        # explicit frontier from _enqueue_draft_segment_metadata.
        return max(int(layer_idx) for layer_idx in layer_caches.keys())

    def _submit_prefetch_after_metadata(
        self,
        *,
        prefetch_runtime: PrefetchRuntime,
        mode: str,
        step_id: int,
        phase: str,
        frontier_layer_idx: int | None = None,
    ) -> int:
        if self._dual_queue_prefetch_enabled():
            frontier = (
                self._draft_prefetch_frontier_layer_idx()
                if frontier_layer_idx is None
                else int(frontier_layer_idx)
            )
            if frontier is None:
                return 0
            if mode == "draft":
                return prefetch_runtime.submit_deferred_draft_segment(
                    step_id=step_id,
                    frontier_layer_idx=frontier,
                    boundaries=self._draft_segment_boundaries(),
                )
            if mode in {"verify", "verify_kt_hybrid"}:
                return prefetch_runtime.submit_deferred_verify_segment(
                    step_id=step_id,
                    frontier_layer_idx=frontier,
                    boundaries=self._verify_segment_boundaries(),
                )
            return 0
        if mode == "draft" and self._prefetch_runtime_mode() == "draft_direct_active":
            return prefetch_runtime.submit_draft_direct_active_prefetch(
                step_id=step_id,
                phase=phase,
                frontier_layer_idx=(
                    self._draft_prefetch_frontier_layer_idx()
                    if frontier_layer_idx is None
                    else int(frontier_layer_idx)
                ),
                visible_budget_ms=float(getattr(self.config, "draft_prefetch_visible_budget_ms", 3.0)),
            )
        if mode == "draft" and self._prefetch_runtime_mode() == "draft_segment_indexed":
            return prefetch_runtime.submit_draft_segment_indexed_prefetch(
                step_id=step_id,
                phase=phase,
                frontier_layer_idx=(
                    self._draft_prefetch_frontier_layer_idx()
                    if frontier_layer_idx is None
                    else int(frontier_layer_idx)
                ),
                visible_budget_ms=float(getattr(self.config, "draft_prefetch_visible_budget_ms", 3.0)),
            )
        return prefetch_runtime.submit_from_global_queue(step_id=step_id, phase=phase)

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

    def _acquire_prefetch_host_buffer_slot(
        self,
        *,
        mode: str,
        token_capacity: int,
    ) -> tuple[int | None, float]:
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
                if self._dual_queue_prefetch_enabled():
                    self._profile["dual_queue_metadata_host_buffer_drop_count"] += 1
                    return None, (perf_counter() - wait_t0) * 1000.0
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

    def _raise_verify_boundary_worker_error(self) -> None:
        self._ensure_prefetch_internal_state()
        error = getattr(self, "_verify_boundary_worker_error", None)
        if error is not None:
            raise RuntimeError("Background verify boundary prefetch worker failed") from error

    def _enqueue_verify_boundary_prefetch(
        self,
        *,
        step_id: int,
        target_layer_start: int,
        target_layer_end: int,
        target_segment_id: int,
    ) -> bool:
        self._ensure_prefetch_internal_state()
        if not (
            getattr(self, "_prefetch_async_enabled", False)
            and self._verify_boundary_prefetch_async_enabled()
        ):
            return False
        queue_obj = getattr(self, "_verify_boundary_worker_queue", None)
        if queue_obj is None:
            return False
        self._raise_verify_boundary_worker_error()
        enqueue_t0 = perf_counter()
        item = {
            "step_id": int(step_id),
            "target_layer_start": int(target_layer_start),
            "target_layer_end": int(target_layer_end),
            "target_segment_id": int(target_segment_id),
            "enqueue_ts_ms": enqueue_t0 * 1000.0,
        }
        with self._verify_boundary_worker_cv:
            self._verify_boundary_worker_outstanding += 1
            queue_depth = self._verify_boundary_worker_outstanding
        queue_obj.put(item)
        enqueue_ms = (perf_counter() - enqueue_t0) * 1000.0
        if self.profile_enabled and self.rank == 0:
            with self._prefetch_profile_lock:
                self._profile["verify_boundary_async_prefetch_enqueue_count"] += 1.0
                self._profile["verify_boundary_async_prefetch_enqueue_ms"] += enqueue_ms
                self._profile["verify_boundary_async_prefetch_queue_depth_sum"] += float(queue_depth)
                self._profile["verify_boundary_async_prefetch_queue_depth_max"] = max(
                    float(self._profile.get("verify_boundary_async_prefetch_queue_depth_max", 0.0)),
                    float(queue_depth),
                )
        return True

    def _wait_for_verify_boundary_prefetch_drain(self) -> float:
        self._ensure_prefetch_internal_state()
        if getattr(self, "_verify_boundary_worker_queue", None) is None:
            return 0.0
        wait_t0 = perf_counter()
        with self._verify_boundary_worker_cv:
            while self._verify_boundary_worker_outstanding > 0:
                self._raise_verify_boundary_worker_error()
                self._verify_boundary_worker_cv.wait(timeout=0.001)
            self._raise_verify_boundary_worker_error()
        wait_ms = (perf_counter() - wait_t0) * 1000.0
        if self.profile_enabled and self.rank == 0:
            with self._prefetch_profile_lock:
                self._profile["verify_boundary_async_prefetch_drain_count"] += 1.0
                self._profile["verify_boundary_async_prefetch_drain_wait_ms"] += wait_ms
        return wait_ms

    def _process_verify_boundary_prefetch_item(self, item: dict[str, object]) -> None:
        prefetch_runtime = getattr(self, "prefetch_runtime", None)
        if prefetch_runtime is None:
            return
        queue_wait_ms = perf_counter() * 1000.0 - float(item.get("enqueue_ts_ms", 0.0))
        submit_t0 = perf_counter()
        with self._prefetch_runtime_lock:
            if self._dual_queue_prefetch_enabled():
                submitted = prefetch_runtime.submit_verify_segment_prefetch(
                    step_id=int(item["step_id"]),
                    target_layer_start=int(item["target_layer_start"]),
                    target_layer_end=int(item["target_layer_end"]),
                    target_segment_id=int(item["target_segment_id"]),
                )
            else:
                submitted = prefetch_runtime.submit_verify_segment_prefetch(
                    step_id=int(item["step_id"]),
                    target_layer_start=int(item["target_layer_start"]),
                    target_layer_end=int(item["target_layer_end"]),
                    visible_budget_ms=float(getattr(
                        self.config,
                        "verify_prefetch_visible_budget_ms",
                        12.0,
                    )),
                )
        submit_ms = (perf_counter() - submit_t0) * 1000.0
        if self.profile_enabled and self.rank == 0:
            with self._prefetch_profile_lock:
                self._profile["verify_boundary_async_prefetch_worker_count"] += 1.0
                self._profile["verify_boundary_async_prefetch_worker_queue_wait_ms"] += queue_wait_ms
                self._profile["verify_boundary_async_prefetch_worker_submit_ms"] += submit_ms
                self._profile["verify_boundary_async_prefetch_worker_submitted_count"] += float(submitted)
            self._trace_prefetch_interval(
                name="verify_boundary_async_prefetch_submit",
                start_ms=float(item.get("enqueue_ts_ms", 0.0)) + queue_wait_ms,
                duration_ms=submit_ms,
                step_id=int(item["step_id"]),
                mode="verify",
                tid="verify-boundary-worker",
            )

    def _verify_boundary_worker_main(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.set_device(self.rank)
        while True:
            queue_obj = self._verify_boundary_worker_queue
            if queue_obj is None:
                return
            item = queue_obj.get()
            if item is None:
                return
            try:
                self._process_verify_boundary_prefetch_item(item)
            except BaseException as exc:
                with self._verify_boundary_worker_cv:
                    if self._verify_boundary_worker_error is None:
                        self._verify_boundary_worker_error = exc
            finally:
                with self._verify_boundary_worker_cv:
                    self._verify_boundary_worker_outstanding = max(
                        0,
                        self._verify_boundary_worker_outstanding - 1,
                    )
                    self._verify_boundary_worker_cv.notify_all()

    def _record_verify_metadata_profile_from_runtime_meta(
        self,
        runtime_meta,
        *,
        mode: str,
        step_id: int | None = None,
    ) -> None:
        self._ensure_prefetch_internal_state()
        if runtime_meta is None or str(mode) not in {"verify", "verify_kt_hybrid"}:
            return
        loop_t0 = perf_counter()
        top_k = int(
            getattr(
                getattr(self.config, "hf_config", None),
                "num_experts_per_tok",
                getattr(getattr(self.config, "hf_config", None), "top_k", 1),
            )
        )
        layer_count = 0
        total_routes_sum = 0.0
        miss_routes_sum = 0.0
        miss_experts_sum = 0.0
        active_experts_sum = 0.0
        route_ratio_sum = 0.0
        execution_layer_count = 0
        execution_total_routes_sum = 0.0
        execution_cpu_routes_sum = 0.0
        execution_cpu_experts_sum = 0.0
        layer_cpu_experts: dict[int, float] = {}
        layer_cpu_routes: dict[int, float] = {}
        layer_active_experts: dict[int, float] = {}
        layer_active_routes: dict[int, float] = {}
        layer_execution_cpu_experts: dict[int, float] = {}
        layer_execution_cpu_routes: dict[int, float] = {}
        layer_execution_cpu_route_counts: dict[int, list[int]] = {}
        layer_execution_route_counts: dict[int, list[int]] = {}
        layer_logical_route_rows: dict[int, list[list[int]]] = {}
        layer_execution_route_rows: dict[int, list[list[int]]] = {}
        layer_execution_route_status: dict[int, list[list[int]]] = {}
        for _layer_idx, meta in runtime_meta.items():
            layer_idx = int(_layer_idx)
            layer_count += 1
            token_count = int(getattr(meta, "token_count", 0) or 0)
            total_routes = float(max(0, token_count * top_k))
            ids = getattr(meta, "aggregated_expert_ids", None)
            counts = getattr(meta, "aggregated_activation_count", None)
            status = getattr(meta, "expert_status", None)
            active_experts = 0.0
            miss_routes = 0.0
            miss_experts = 0.0
            if counts is not None:
                if counts.device.type != "cpu":
                    counts = counts.to(device="cpu")
                total_routes = float(counts.sum().item())
                active_experts = float((counts > 0).sum().item())
            if status is not None and ids is not None and counts is not None:
                if ids.device.type != "cpu" or ids.dtype != torch.int64:
                    ids = ids.to(device="cpu", dtype=torch.int64)
                if status.device.type != "cpu":
                    status = status.to(device="cpu")
                status_for_ids = status.index_select(0, ids)
                miss_mask = status_for_ids == 2
                if miss_mask.numel() > 0:
                    miss_routes = float(counts[miss_mask].sum().item())
                    miss_experts = float(miss_mask.sum().item())
            else:
                miss_experts = float(getattr(meta, "miss_count", 0.0) or 0.0)
            execution_counts = getattr(meta, "execution_activation_count", None)
            execution_cpu_counts_override = None
            if bool(getattr(self.config, "transfer_aware_profile", False)):
                logical_rows = getattr(meta, "selected_experts", None)
                execution_rows = getattr(
                    meta, "execution_selected_experts", None
                )
                execution_status_rows = getattr(
                    meta, "execution_route_status", None
                )
                if logical_rows is not None:
                    layer_logical_route_rows[layer_idx] = [
                        [int(value) for value in row]
                        for row in logical_rows.tolist()
                    ]
                if execution_rows is not None:
                    layer_execution_route_rows[layer_idx] = [
                        [int(value) for value in row]
                        for row in execution_rows.tolist()
                    ]
                if execution_status_rows is not None:
                    layer_execution_route_status[layer_idx] = [
                        [int(value) for value in row]
                        for row in execution_status_rows.tolist()
                    ]
                if (
                    execution_counts is None
                    and execution_rows is not None
                    and execution_status_rows is not None
                ):
                    flat_execution_ids = execution_rows.reshape(-1).to(
                        dtype=torch.int64, device="cpu"
                    )
                    flat_execution_status = execution_status_rows.reshape(
                        -1
                    ).to(dtype=torch.int64, device="cpu")
                    num_experts = int(
                        getattr(
                            getattr(self.config, "hf_config", None),
                            "num_experts",
                            0,
                        )
                    )
                    execution_counts = torch.bincount(
                        flat_execution_ids, minlength=num_experts
                    )
                    execution_cpu_counts_override = torch.bincount(
                        flat_execution_ids[flat_execution_status == 2],
                        minlength=num_experts,
                    )
            if execution_counts is not None:
                if execution_counts.device.type != "cpu":
                    execution_counts = execution_counts.to(device="cpu")
                execution_counts = execution_counts.to(dtype=torch.int64)
                execution_layer_count += 1
                execution_total_routes = float(execution_counts.sum().item())
                if execution_cpu_counts_override is not None:
                    execution_cpu_counts = (
                        execution_cpu_counts_override.to(dtype=torch.int64)
                    )
                elif status is not None:
                    status_cpu = status
                    if status_cpu.device.type != "cpu":
                        status_cpu = status_cpu.to(device="cpu")
                    execution_cpu_counts = torch.where(
                        status_cpu == 2,
                        execution_counts,
                        torch.zeros_like(execution_counts),
                    )
                else:
                    execution_cpu_counts = torch.zeros_like(execution_counts)
                execution_cpu_routes = float(execution_cpu_counts.sum().item())
                execution_cpu_experts = float(
                    (execution_cpu_counts > 0).sum().item()
                )
                execution_total_routes_sum += execution_total_routes
                execution_cpu_routes_sum += execution_cpu_routes
                execution_cpu_experts_sum += execution_cpu_experts
                layer_execution_cpu_routes[layer_idx] = execution_cpu_routes
                layer_execution_cpu_experts[layer_idx] = execution_cpu_experts
                layer_execution_cpu_route_counts[layer_idx] = [
                    int(value) for value in execution_cpu_counts.tolist()
                ]
                layer_execution_route_counts[layer_idx] = [
                    int(value) for value in execution_counts.tolist()
                ]
            total_routes = max(total_routes, 1.0)
            total_routes_sum += total_routes
            miss_routes_sum += miss_routes
            miss_experts_sum += miss_experts
            active_experts_sum += active_experts
            route_ratio_sum += miss_routes / total_routes
            layer_cpu_experts[layer_idx] = miss_experts
            layer_cpu_routes[layer_idx] = miss_routes
            layer_active_experts[layer_idx] = active_experts
            layer_active_routes[layer_idx] = total_routes

        loop_ms = (perf_counter() - loop_t0) * 1000.0
        if self.profile_enabled and self.rank == 0:
            with self._prefetch_profile_lock:
                self._profile["verify_metadata_profile_async_count"] += 1.0
                self._profile["verify_metadata_profile_async_loop_ms"] += loop_ms
                self._profile["verify_metadata_profile_async_layer_count"] += float(layer_count)
                self._profile["verify_execution_profile_layer_count"] += float(
                    execution_layer_count
                )
                self._profile["verify_execution_active_routes_sum"] += float(
                    execution_total_routes_sum
                )
                self._profile["verify_execution_cpu_routes_sum"] += float(
                    execution_cpu_routes_sum
                )
                self._profile["verify_execution_cpu_experts_sum"] += float(
                    execution_cpu_experts_sum
                )
                for prefix in ("", "verify_"):
                    self._profile[f"{prefix}moe_profile_count"] += float(layer_count)
                    self._profile[f"{prefix}pre_transfer_cache_miss_sum"] += miss_routes_sum
                    self._profile[f"{prefix}pre_transfer_active_count_sum"] += total_routes_sum
                    self._profile[f"{prefix}cpu_route_ratio_sum"] += route_ratio_sum
                    self._profile[f"{prefix}cpu_routes_sum"] += miss_routes_sum
                    self._profile[f"{prefix}cpu_weight_mass_ratio_sum"] += 0.0
                    self._profile[f"{prefix}realized_cpu_expert_count_sum"] += miss_experts_sum
                    self._profile[f"{prefix}activated_expert_set_size_sum"] += active_experts_sum
                    for layer_idx, cpu_experts in layer_cpu_experts.items():
                        base = f"{prefix}layer_{layer_idx}_"
                        self._profile[f"{base}realized_cpu_expert_count_sum"] += cpu_experts
                        self._profile[f"{base}cpu_routes_sum"] += layer_cpu_routes.get(layer_idx, 0.0)
                        self._profile[f"{base}active_expert_count_sum"] += layer_active_experts.get(layer_idx, 0.0)
                        self._profile[f"{base}active_routes_sum"] += layer_active_routes.get(layer_idx, 0.0)
                        self._profile[f"{base}moe_profile_count"] += 1.0
                if step_id is not None:
                    step_rec = self._verify_metadata_records_by_step.setdefault(int(step_id), {})
                    step_rec["metadata_item_count"] = float(step_rec.get("metadata_item_count", 0.0)) + 1.0
                    step_rec["metadata_layer_count"] = (
                        float(step_rec.get("metadata_layer_count", 0.0)) + float(layer_count)
                    )
                    step_rec["metadata_cpu_routes_sum"] = (
                        float(step_rec.get("metadata_cpu_routes_sum", 0.0)) + miss_routes_sum
                    )
                    step_rec["metadata_realized_cpu_expert_count_sum"] = (
                        float(step_rec.get("metadata_realized_cpu_expert_count_sum", 0.0)) + miss_experts_sum
                    )
                    step_rec["metadata_pre_transfer_cache_miss_sum"] = (
                        float(step_rec.get("metadata_pre_transfer_cache_miss_sum", 0.0)) + miss_routes_sum
                    )
                    step_rec["metadata_pre_transfer_active_count_sum"] = (
                        float(step_rec.get("metadata_pre_transfer_active_count_sum", 0.0)) + total_routes_sum
                    )
                    step_rec["metadata_activated_expert_set_size_sum"] = (
                        float(step_rec.get("metadata_activated_expert_set_size_sum", 0.0)) + active_experts_sum
                    )
                    step_rec["metadata_cpu_route_ratio_sum"] = (
                        float(step_rec.get("metadata_cpu_route_ratio_sum", 0.0)) + route_ratio_sum
                    )
                    step_rec["metadata_execution_layer_count"] = (
                        float(step_rec.get("metadata_execution_layer_count", 0.0))
                        + float(execution_layer_count)
                    )
                    step_rec["metadata_execution_active_routes_sum"] = (
                        float(step_rec.get("metadata_execution_active_routes_sum", 0.0))
                        + execution_total_routes_sum
                    )
                    step_rec["metadata_execution_cpu_routes_sum"] = (
                        float(step_rec.get("metadata_execution_cpu_routes_sum", 0.0))
                        + execution_cpu_routes_sum
                    )
                    step_rec["metadata_execution_cpu_experts_sum"] = (
                        float(step_rec.get("metadata_execution_cpu_experts_sum", 0.0))
                        + execution_cpu_experts_sum
                    )
                    layer_cpu_experts_rec = step_rec.setdefault("metadata_layer_cpu_experts", {})
                    layer_cpu_routes_rec = step_rec.setdefault("metadata_layer_cpu_routes", {})
                    layer_active_experts_rec = step_rec.setdefault("metadata_layer_active_experts", {})
                    layer_active_routes_rec = step_rec.setdefault("metadata_layer_active_routes", {})
                    layer_execution_cpu_experts_rec = step_rec.setdefault(
                        "metadata_layer_execution_cpu_experts", {}
                    )
                    layer_execution_cpu_routes_rec = step_rec.setdefault(
                        "metadata_layer_execution_cpu_routes", {}
                    )
                    layer_execution_route_counts_rec = step_rec.setdefault(
                        "metadata_layer_execution_cpu_route_counts", {}
                    )
                    layer_execution_all_route_counts_rec = step_rec.setdefault(
                        "metadata_layer_execution_route_counts", {}
                    )
                    logical_route_rows_rec = step_rec.setdefault(
                        "metadata_layer_logical_route_rows", {}
                    )
                    execution_route_rows_rec = step_rec.setdefault(
                        "metadata_layer_execution_route_rows", {}
                    )
                    execution_route_status_rec = step_rec.setdefault(
                        "metadata_layer_execution_route_status", {}
                    )
                    logical_route_rows_rec.update(
                        {
                            str(layer_idx): rows
                            for layer_idx, rows in layer_logical_route_rows.items()
                        }
                    )
                    execution_route_rows_rec.update(
                        {
                            str(layer_idx): rows
                            for layer_idx, rows in layer_execution_route_rows.items()
                        }
                    )
                    execution_route_status_rec.update(
                        {
                            str(layer_idx): rows
                            for layer_idx, rows in layer_execution_route_status.items()
                        }
                    )
                    for layer_idx, cpu_experts in layer_cpu_experts.items():
                        key = str(int(layer_idx))
                        layer_cpu_experts_rec[key] = (
                            float(layer_cpu_experts_rec.get(key, 0.0)) + float(cpu_experts)
                        )
                        layer_cpu_routes_rec[key] = (
                            float(layer_cpu_routes_rec.get(key, 0.0))
                            + float(layer_cpu_routes.get(layer_idx, 0.0))
                        )
                        layer_active_experts_rec[key] = (
                            float(layer_active_experts_rec.get(key, 0.0))
                            + float(layer_active_experts.get(layer_idx, 0.0))
                        )
                        layer_active_routes_rec[key] = (
                            float(layer_active_routes_rec.get(key, 0.0))
                            + float(layer_active_routes.get(layer_idx, 0.0))
                        )
                    for layer_idx, cpu_routes in layer_execution_cpu_routes.items():
                        key = str(int(layer_idx))
                        layer_execution_cpu_routes_rec[key] = (
                            float(layer_execution_cpu_routes_rec.get(key, 0.0))
                            + float(cpu_routes)
                        )
                        layer_execution_cpu_experts_rec[key] = (
                            float(layer_execution_cpu_experts_rec.get(key, 0.0))
                            + float(layer_execution_cpu_experts.get(layer_idx, 0.0))
                        )
                        current = layer_execution_route_counts_rec.get(key)
                        incoming = layer_execution_cpu_route_counts[layer_idx]
                        if isinstance(current, list) and len(current) == len(incoming):
                            layer_execution_route_counts_rec[key] = [
                                int(left) + int(right)
                                for left, right in zip(current, incoming, strict=True)
                            ]
                        else:
                            layer_execution_route_counts_rec[key] = list(incoming)
                        current_all = layer_execution_all_route_counts_rec.get(key)
                        incoming_all = layer_execution_route_counts[layer_idx]
                        if isinstance(current_all, list) and len(current_all) == len(incoming_all):
                            layer_execution_all_route_counts_rec[key] = [
                                int(left) + int(right)
                                for left, right in zip(
                                    current_all, incoming_all, strict=True
                                )
                            ]
                        else:
                            layer_execution_all_route_counts_rec[key] = list(incoming_all)

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
        self._observe_perfect_match_metadata(
            mode=str(item["mode"]),
            step_id=int(item["step_id"]),
            runtime_meta=runtime_meta,
        )
        self._record_verify_metadata_profile_from_runtime_meta(
            runtime_meta,
            mode=str(item["mode"]),
            step_id=int(item["step_id"]),
        )

        observe_stats: dict[str, float] = {}
        submit_after_ms = 0.0
        observe_call_ms = 0.0
        record_consumed_ms = 0.0
        observe_t0 = perf_counter()
        with self._prefetch_runtime_lock:
            if not getattr(self, "_skip_metadata_observe", False):
                mode = str(item["mode"])
                step_id = int(item["step_id"])
                if mode == "prefill":
                    observe_call_t0 = perf_counter()
                    observe_stats = prefetch_runtime.observe_prefill(runtime_meta, step_id=step_id)
                    observe_call_ms = (perf_counter() - observe_call_t0) * 1000.0
                elif mode == "draft":
                    observe_call_t0 = perf_counter()
                    observe_stats = prefetch_runtime.observe_draft(runtime_meta, step_id=step_id)
                    observe_call_ms = (perf_counter() - observe_call_t0) * 1000.0
                elif mode in ("verify", "verify_kt_hybrid"):
                    observe_call_t0 = perf_counter()
                    observe_stats = prefetch_runtime.observe_verify(runtime_meta, step_id=step_id)
                    observe_call_ms = (perf_counter() - observe_call_t0) * 1000.0
                    if bool(item["record_verify_consumed"]):
                        consumed_t0 = perf_counter()
                        prefetch_runtime.record_verify_consumed(runtime_meta, step_id=step_id)
                        record_consumed_ms = (perf_counter() - consumed_t0) * 1000.0
            elif self.profile_enabled and self.rank == 0:
                with self._prefetch_profile_lock:
                    self._profile["metadata_observe_skipped_count"] += 1
            observe_ms = (perf_counter() - observe_t0) * 1000.0

            submit_after_phase = item["submit_after_phase"]
            if submit_after_phase is not None:
                mode = str(item["mode"])
                stale_draft_submit = (
                    mode == "draft"
                    and (
                        (
                            self._prefetch_runtime_mode() == "draft_direct_active"
                            and int(item["step_id"]) != int(getattr(self, "_active_draft_prefetch_step_id", -1))
                        )
                        or (
                            self._prefetch_runtime_mode() == "draft_segment_indexed"
                            and bool(getattr(prefetch_runtime, "_active_draft_iteration_steps", set()))
                            and int(item["step_id"]) not in getattr(prefetch_runtime, "_active_draft_iteration_steps", set())
                        )
                    )
                )
                if stale_draft_submit:
                    self._profile["run_draft_submit_after_stale_skip_count"] += 1
                    if self._prefetch_runtime_mode() == "draft_segment_indexed":
                        self._profile["run_draft_missed_prefetch_window_count"] += 1
                        prefetch_runtime._profile["draft_segment_indexed_missed_prefetch_window_count"] += 1
                else:
                    submit_after_t0 = perf_counter()
                    self._submit_prefetch_after_metadata(
                        prefetch_runtime=prefetch_runtime,
                        mode=mode,
                        step_id=int(item["step_id"]),
                        phase=str(submit_after_phase),
                        frontier_layer_idx=item.get("frontier_layer_idx"),
                    )
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
                self._profile[f"{prefix}_observe_call_ms"] += observe_call_ms
                self._profile[f"{prefix}_record_consumed_ms"] += record_consumed_ms
                self._profile[f"{prefix}_mark_access_ms"] += float(observe_stats.get("mark_access_ms", 0.0))
                self._profile[f"{prefix}_queue_update_ms"] += float(observe_stats.get("queue_update_ms", 0.0))
                self._profile[f"{prefix}_queue_aggregate_ms"] += float(observe_stats.get("queue_aggregate_ms", 0.0))
                self._profile[f"{prefix}_queue_filter_ms"] += float(observe_stats.get("queue_filter_ms", 0.0))
                self._profile[f"{prefix}_queue_entry_update_ms"] += float(observe_stats.get("queue_entry_update_ms", 0.0))
                self._profile[f"{prefix}_segment_index_aggregate_ms"] += float(observe_stats.get("segment_index_aggregate_ms", 0.0))
                self._profile[f"{prefix}_segment_index_filter_ms"] += float(observe_stats.get("segment_index_filter_ms", 0.0))
                self._profile[f"{prefix}_segment_index_entry_update_ms"] += float(observe_stats.get("segment_index_entry_update_ms", 0.0))
                self._profile[f"{prefix}_segment_index_rank_cache_rebuild_ms"] += float(
                    observe_stats.get("segment_index_rank_cache_rebuild_ms", 0.0)
                )
                self._profile[f"{prefix}_segment_index_rank_cache_rebuild_count"] += float(
                    observe_stats.get("segment_index_rank_cache_rebuild_count", 0.0)
                )
                self._profile[f"{prefix}_observe_verify_rank_guard_ms"] += float(observe_stats.get("observe_verify_rank_guard_ms", 0.0))
                self._profile[f"{prefix}_observe_verify_segment_index_ms"] += float(observe_stats.get("observe_verify_segment_index_ms", 0.0))
                self._profile[f"{prefix}_observe_verify_runtime_meta_call_ms"] += float(observe_stats.get("observe_verify_runtime_meta_call_ms", 0.0))
                self._profile[f"{prefix}_async_turnaround_ms"] += turnaround_ms
                if item["mode"] in {"draft", "verify", "verify_kt_hybrid"}:
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
        boundary_queue = getattr(self, "_verify_boundary_worker_queue", None)
        boundary_worker = getattr(self, "_verify_boundary_worker_thread", None)
        if boundary_queue is not None:
            boundary_queue.put(None)
        if boundary_worker is not None:
            boundary_worker.join(timeout=5.0)
        self._verify_boundary_worker_queue = None
        self._verify_boundary_worker_thread = None

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
        frontier_layer_idx: int | None = None,
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
            "frontier_layer_idx": frontier_layer_idx,
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

    def _clone_perfect_match_meta(self, runtime_meta) -> dict[int, dict[str, torch.Tensor]]:
        out: dict[int, dict[str, torch.Tensor]] = {}
        if not runtime_meta:
            return out
        for layer_idx, meta in runtime_meta.items():
            selected = getattr(meta, "selected_experts", None)
            if selected is None:
                continue
            layer: dict[str, torch.Tensor] = {
                "selected_experts": selected.detach().to("cpu", dtype=torch.int64).clone(),
            }
            routing = getattr(meta, "routing_weights", None)
            if routing is not None:
                layer["routing_weights"] = routing.detach().to(
                    "cpu", dtype=torch.float32
                ).clone()
            route_status = getattr(meta, "route_status", None)
            if route_status is not None:
                layer["route_status"] = route_status.detach().to(
                    "cpu", dtype=torch.int8
                ).clone()
            out[int(layer_idx)] = layer
        return out

    def _observe_perfect_match_metadata(self, mode: str, step_id: int, runtime_meta) -> None:
        self._ensure_prefetch_internal_state()
        if not bool(getattr(self, "_draft_perfect_trace_enabled", False)):
            return
        if str(mode) in {"verify", "verify_kt_hybrid"}:
            cloned = self._clone_perfect_match_meta(runtime_meta)
            if cloned:
                self._draft_perfect_last_verify_meta = cloned
            return
        if str(mode) != "draft":
            return

        pending = getattr(self, "_draft_perfect_pending", None)
        if not pending:
            return
        min_step_id = int(pending.get("min_draft_step_id", -1))
        if int(step_id) < min_step_id:
            self._draft_perfect_profile["stale_draft_metadata_ignored_count"] += 1.0
            return
        cloned = self._clone_perfect_match_meta(runtime_meta)
        if not cloned:
            return
        step_key = int(step_id)
        pending_by_step = self._draft_perfect_draft_meta_by_step.setdefault(step_key, {})
        pending_by_step.update(cloned)
        expected_layers = len(getattr(self, "layer_caches", {}) or {})
        if expected_layers > 0 and len(pending_by_step) < expected_layers:
            return
        self._record_perfect_match_draft_token(pending, step_key, dict(pending_by_step))
        self._draft_perfect_draft_meta_by_step.pop(step_key, None)

    def _record_perfect_match_draft_token(
        self,
        pending: dict[str, object],
        step_id: int,
        draft_meta: dict[int, dict[str, torch.Tensor]],
    ) -> None:
        token_index = int(pending.get("checked_tokens", 0))
        max_check = int(pending.get("max_check_tokens", 0))
        if max_check > 0 and token_index >= max_check:
            return

        expected_layers = len(getattr(self, "layer_caches", {}) or {})
        present_layers = 0
        route_total = route_miss = route_hit = 0
        coverage_total = coverage_hit = 0
        pred_row_total = pred_row_hit = 0
        input_row_total = input_row_hit = 0
        pred_row_layer_exact_total = pred_row_layer_exact_hit = 0
        input_row_layer_exact_total = input_row_layer_exact_hit = 0

        verify_meta = pending.get("verify_meta")
        if not isinstance(verify_meta, dict):
            verify_meta = {}
        accepted = int(pending.get("accepted_draft_tokens", 0))
        rejected_start = int(pending.get("rejected_verify_row_start", accepted))

        for layer_idx, layer in draft_meta.items():
            selected = layer.get("selected_experts")
            if selected is None or selected.numel() == 0:
                continue
            draft_row = selected.reshape(selected.shape[0], -1)[0]
            present_layers += 1

            status = layer.get("route_status")
            if status is not None and status.numel() > 0:
                status_row = status.reshape(status.shape[0], -1)[0]
                route_total += int(status_row.numel())
                route_miss += int((status_row == 2).sum().item())
                route_hit += int((status_row == 1).sum().item())
            else:
                route_total += int(draft_row.numel())

            prev_layer = verify_meta.get(int(layer_idx))
            if not isinstance(prev_layer, dict):
                continue
            prev_selected = prev_layer.get("selected_experts")
            if prev_selected is None or prev_selected.numel() == 0:
                continue
            prev_selected = prev_selected.reshape(prev_selected.shape[0], -1)
            prev_rows = int(prev_selected.shape[0])

            if rejected_start < prev_rows:
                union_rows = prev_selected[rejected_start:prev_rows].reshape(-1)
                if union_rows.numel() > 0:
                    union = {int(x) for x in union_rows.tolist()}
                    coverage_total += int(draft_row.numel())
                    coverage_hit += sum(
                        1 for expert_id in draft_row.tolist() if int(expert_id) in union
                    )

            pred_row = accepted + token_index
            if 0 <= pred_row < prev_rows:
                prev_row = prev_selected[pred_row].reshape(-1)
                n = min(int(prev_row.numel()), int(draft_row.numel()))
                pred_row_total += n
                if n > 0:
                    pred_row_hit += int((draft_row[:n] == prev_row[:n]).sum().item())
                    pred_row_layer_exact_total += 1
                    pred_row_layer_exact_hit += int(
                        n == int(draft_row.numel())
                        and n == int(prev_row.numel())
                        and bool(torch.equal(draft_row, prev_row))
                    )

            input_row = accepted + 1 + token_index
            if 0 <= input_row < prev_rows:
                prev_row = prev_selected[input_row].reshape(-1)
                n = min(int(prev_row.numel()), int(draft_row.numel()))
                input_row_total += n
                if n > 0:
                    input_row_hit += int((draft_row[:n] == prev_row[:n]).sum().item())
                    input_row_layer_exact_total += 1
                    input_row_layer_exact_hit += int(
                        n == int(draft_row.numel())
                        and n == int(prev_row.numel())
                        and bool(torch.equal(draft_row, prev_row))
                    )

        token_perfect = (
            present_layers > 0
            and (expected_layers <= 0 or present_layers >= expected_layers)
            and route_total > 0
            and route_miss == 0
            and route_hit == route_total
        )
        pending["checked_tokens"] = token_index + 1
        for key, value in (
            ("route_total", route_total),
            ("route_miss", route_miss),
            ("coverage_total", coverage_total),
            ("coverage_hit", coverage_hit),
            ("pred_row_total", pred_row_total),
            ("pred_row_hit", pred_row_hit),
            ("input_row_total", input_row_total),
            ("input_row_hit", input_row_hit),
            ("pred_row_layer_exact_total", pred_row_layer_exact_total),
            ("pred_row_layer_exact_hit", pred_row_layer_exact_hit),
            ("input_row_layer_exact_total", input_row_layer_exact_total),
            ("input_row_layer_exact_hit", input_row_layer_exact_hit),
        ):
            pending[key] = int(pending.get(key, 0)) + int(value)

        if bool(pending.get("prefix_open", True)) and token_perfect:
            pending["perfect_prefix_len"] = int(pending.get("perfect_prefix_len", 0)) + 1
        else:
            pending["prefix_open"] = False
        if token_perfect:
            pending["perfect_tokens"] = int(pending.get("perfect_tokens", 0)) + 1

        oracle_covered = coverage_total > 0 and coverage_hit == coverage_total
        if bool(pending.get("oracle_prefix_open", True)) and oracle_covered:
            pending["oracle_prefix_len"] = int(pending.get("oracle_prefix_len", 0)) + 1
        else:
            pending["oracle_prefix_open"] = False
        if oracle_covered:
            pending["oracle_covered_tokens"] = int(pending.get("oracle_covered_tokens", 0)) + 1

        token_records = pending.setdefault("token_records", [])
        if isinstance(token_records, list) and len(token_records) < 32:
            token_records.append(
                {
                    "draft_step_id": int(step_id),
                    "token_index": int(token_index),
                    "perfect": bool(token_perfect),
                    "present_layers": int(present_layers),
                    "route_total": int(route_total),
                    "route_miss": int(route_miss),
                    "coverage_total": int(coverage_total),
                    "coverage_hit": int(coverage_hit),
                    "oracle_covered": bool(oracle_covered),
                    "pred_row_total": int(pred_row_total),
                    "pred_row_hit": int(pred_row_hit),
                    "input_row_total": int(input_row_total),
                    "input_row_hit": int(input_row_hit),
                }
            )

    def _finalize_perfect_match_pending(self) -> None:
        self._ensure_prefetch_internal_state()
        pending = getattr(self, "_draft_perfect_pending", None)
        if not pending:
            return
        checked = int(pending.get("checked_tokens", 0))
        prefix_len = int(pending.get("perfect_prefix_len", 0))
        perfect_tokens = int(pending.get("perfect_tokens", 0))
        oracle_prefix_len = int(pending.get("oracle_prefix_len", 0))
        oracle_covered_tokens = int(pending.get("oracle_covered_tokens", 0))
        route_total = int(pending.get("route_total", 0))
        route_miss = int(pending.get("route_miss", 0))
        coverage_total = int(pending.get("coverage_total", 0))
        coverage_hit = int(pending.get("coverage_hit", 0))
        profile = self._draft_perfect_profile
        if checked > 0:
            profile["followup_events"] += 1.0
            profile["checked_tokens"] += float(checked)
            profile["perfect_tokens"] += float(perfect_tokens)
            profile["perfect_prefix_token_sum"] += float(prefix_len)
            profile["oracle_covered_tokens"] += float(oracle_covered_tokens)
            profile["oracle_prefix_token_sum"] += float(oracle_prefix_len)
            profile["route_total"] += float(route_total)
            profile["route_miss"] += float(route_miss)
            profile["coverage_total"] += float(coverage_total)
            profile["coverage_hit"] += float(coverage_hit)
            for key in (
                "pred_row_total",
                "pred_row_hit",
                "input_row_total",
                "input_row_hit",
                "pred_row_layer_exact_total",
                "pred_row_layer_exact_hit",
                "input_row_layer_exact_total",
                "input_row_layer_exact_hit",
            ):
                profile[key] += float(int(pending.get(key, 0)))
            if prefix_len > 0:
                profile["prefix_ge1_events"] += 1.0
            if prefix_len >= checked:
                profile["prefix_full_checked_events"] += 1.0
            if oracle_prefix_len > 0:
                profile["oracle_prefix_ge1_events"] += 1.0
            if oracle_prefix_len >= checked:
                profile["oracle_prefix_full_checked_events"] += 1.0
        else:
            profile["no_followup_events"] += 1.0

        record = {
            "origin_step_index": int(pending.get("origin_step_index", -1)),
            "drafted_tokens": int(pending.get("drafted_tokens", 0)),
            "accepted_draft_tokens": int(pending.get("accepted_draft_tokens", 0)),
            "rejected_tokens": int(pending.get("rejected_tokens", 0)),
            "verify_trace_len": int(pending.get("verify_trace_len", 0)),
            "checked_tokens": checked,
            "perfect_tokens": perfect_tokens,
            "perfect_prefix_len": prefix_len,
            "oracle_covered_tokens": oracle_covered_tokens,
            "oracle_prefix_len": oracle_prefix_len,
            "route_total": route_total,
            "route_miss": route_miss,
            "coverage_total": coverage_total,
            "coverage_hit": coverage_hit,
            "pred_row_total": int(pending.get("pred_row_total", 0)),
            "pred_row_hit": int(pending.get("pred_row_hit", 0)),
            "input_row_total": int(pending.get("input_row_total", 0)),
            "input_row_hit": int(pending.get("input_row_hit", 0)),
            "refill_enabled": bool(pending.get("refill_enabled", False)),
            "refill_promoted": int(pending.get("refill_promoted", 0)),
            "refill_cpu_experts": int(pending.get("refill_cpu_experts", 0)),
            "refill_skipped_inflight": int(pending.get("refill_skipped_inflight", 0)),
            "token_records": list(pending.get("token_records", [])),
        }
        if len(self._draft_perfect_records) < int(getattr(self, "_draft_perfect_detail_limit", 256)):
            self._draft_perfect_records.append(record)
        self._draft_perfect_pending = None
        self._draft_perfect_draft_meta_by_step.clear()

    def _refill_rejected_verify_experts_for_perfect_match(
        self,
        pending: dict[str, object],
    ) -> None:
        if not bool(getattr(self, "_draft_perfect_refill_rejected", False)):
            return
        verify_meta = pending.get("verify_meta")
        if not isinstance(verify_meta, dict):
            return
        prefetch_runtime = getattr(self, "prefetch_runtime", None)
        if prefetch_runtime is not None:
            with self._prefetch_runtime_lock:
                publish_direct = getattr(prefetch_runtime, "publish_direct_active_ready", None)
                if publish_direct is not None:
                    publish_direct(step_id=int(getattr(self, "_prefetch_step_id", 0)))
                publish_ready = getattr(prefetch_runtime, "publish_ready", None)
                if publish_ready is not None:
                    publish_ready(step_id=int(getattr(self, "_prefetch_step_id", 0)))
                inflight_count = len(getattr(prefetch_runtime, "inflight", {}) or {})
            if inflight_count > 0:
                pending["refill_enabled"] = True
                pending["refill_skipped_inflight"] = int(inflight_count)
                self._draft_perfect_profile["refill_skipped_inflight_events"] += 1.0
                self._draft_perfect_profile["refill_skipped_inflight_count"] += float(inflight_count)
                return
        row_start = int(pending.get("rejected_verify_row_start", 0))
        profile = self._draft_perfect_profile
        promoted_total = cpu_total = evicted_total = skipped_total = 0
        transfer_ms_total = 0.0
        for layer_idx, layer in verify_meta.items():
            cache = getattr(self, "layer_caches", {}).get(int(layer_idx))
            if cache is None:
                continue
            selected = layer.get("selected_experts")
            if selected is None or selected.numel() == 0:
                continue
            selected = selected.reshape(selected.shape[0], -1)
            if row_start >= int(selected.shape[0]):
                continue
            rows = selected[row_start:]
            if rows.numel() == 0:
                continue
            routing = layer.get("routing_weights")
            if routing is None:
                routing_rows = torch.ones(rows.shape, dtype=torch.float32)
            else:
                routing = routing.reshape(routing.shape[0], -1)
                routing_rows = routing[row_start:].to(dtype=torch.float32)
            result = apply_verify_cache_fill_policy(
                layer_idx=int(layer_idx),
                selected_experts=rows,
                routing_weights=routing_rows,
                expert_cache=cache,
                step_id=int(getattr(self, "_prefetch_step_id", 0)),
                profile=None,
            )
            promoted_total += int(result.promoted_expert_count)
            cpu_total += int(result.cpu_expert_count)
            evicted_total += len(result.evicted_expert_ids)
            skipped_total += int(result.skipped_pending_count)
            transfer_ms_total += float(result.transfer_ms)
        if promoted_total > 0 and torch.cuda.is_available():
            torch.cuda.synchronize()
        pending["refill_enabled"] = True
        pending["refill_promoted"] = promoted_total
        pending["refill_cpu_experts"] = cpu_total
        pending["refill_evicted"] = evicted_total
        pending["refill_skipped_pending"] = skipped_total
        profile["refill_events"] += 1.0
        profile["refill_promoted"] += float(promoted_total)
        profile["refill_cpu_experts"] += float(cpu_total)
        profile["refill_evicted"] += float(evicted_total)
        profile["refill_skipped_pending"] += float(skipped_total)
        profile["refill_transfer_ms"] += transfer_ms_total

    def record_spec_acceptance_for_perfect_match(self, outcome: dict) -> None:
        self._ensure_prefetch_internal_state()
        if not bool(getattr(self, "_draft_perfect_trace_enabled", False)):
            return
        self._flush_pending_prefetch_metadata(block=True)
        self._finalize_perfect_match_pending()

        drafted = int(outcome.get("drafted_tokens", 0) or 0)
        accepted = int(outcome.get("accepted_draft_tokens", 0) or 0)
        rejected = int(outcome.get("rejected_tokens", max(0, drafted - accepted)) or 0)
        if drafted <= 0 or rejected <= 0:
            return
        verify_meta = self._draft_perfect_last_verify_meta
        if not verify_meta:
            self._draft_perfect_profile["reject_events_missing_verify_meta"] += 1.0
            return
        verify_trace_len = int(outcome.get("verify_trace_len", drafted + 1) or drafted + 1)
        max_check = int(
            os.getenv(
                "NANOVLLM_DRAFT_PERFECT_MATCH_MAX_CHECK_TOKENS",
                str(max(1, int(getattr(self.config, "max_draft_tokens", 1)))),
            )
            or "1"
        )
        pending: dict[str, object] = {
            "origin_step_index": int(outcome.get("step_index", -1) or -1),
            "drafted_tokens": drafted,
            "accepted_draft_tokens": accepted,
            "rejected_tokens": rejected,
            "verify_trace_len": verify_trace_len,
            "verify_meta": verify_meta,
            "rejected_verify_row_start": max(0, min(accepted, verify_trace_len)),
            "min_draft_step_id": int(getattr(self, "_prefetch_step_id", 0)) + 1,
            "max_check_tokens": max(1, max_check),
            "checked_tokens": 0,
            "perfect_tokens": 0,
            "perfect_prefix_len": 0,
            "prefix_open": True,
            "oracle_covered_tokens": 0,
            "oracle_prefix_len": 0,
            "oracle_prefix_open": True,
            "route_total": 0,
            "route_miss": 0,
            "coverage_total": 0,
            "coverage_hit": 0,
            "pred_row_total": 0,
            "pred_row_hit": 0,
            "input_row_total": 0,
            "input_row_hit": 0,
            "pred_row_layer_exact_total": 0,
            "pred_row_layer_exact_hit": 0,
            "input_row_layer_exact_total": 0,
            "input_row_layer_exact_hit": 0,
            "token_records": [],
        }
        self._draft_perfect_profile["reject_events"] += 1.0
        self._draft_perfect_profile["rejected_tokens"] += float(rejected)
        self._draft_perfect_pending = pending
        self._refill_rejected_verify_experts_for_perfect_match(pending)

    def get_profile(self, reset: bool = False) -> dict:
        self._ensure_prefetch_internal_state()
        if self.rank != 0:
            return {}
        self._poll_verify_layer_timing_events()
        self._poll_verify_stream_timings(block=bool(reset))
        dual_queue = self._dual_queue_prefetch_enabled()
        profile_flush_block = bool(reset) or not dual_queue
        self._poll_dual_queue_segment_timings(block=profile_flush_block)
        self._flush_pending_prefetch_metadata(block=profile_flush_block)
        self._wait_for_verify_boundary_prefetch_drain()
        with self._prefetch_profile_lock:
            out = {k: (int(v) if k.endswith("_count") else float(v)) for k, v in self._profile.items()}
            out["verify_op_event_records"] = list(getattr(self, "_verify_op_event_records", []))
            out["draft_op_event_records"] = list(getattr(self, "_draft_op_event_records", []))
            metadata_by_step = getattr(self, "_verify_metadata_records_by_step", {})
            stream_ms_by_step = getattr(self, "_verify_stream_ms_by_step", {})
            verify_call_records: list[dict[str, object]] = []
            for record in getattr(self, "_verify_call_records", []):
                merged = dict(record)
                stream_ms = stream_ms_by_step.get(int(merged.get("step_id", -1)))
                merged["stream_ms_available"] = stream_ms is not None
                merged["stream_ms"] = float(stream_ms or 0.0)
                meta = metadata_by_step.get(int(merged.get("step_id", -1)))
                if meta:
                    for key, value in meta.items():
                        if isinstance(value, dict):
                            merged[key] = dict(value)
                        else:
                            merged[key] = float(value)
                    token_count = float(merged.get("token_count", 0.0) or 0.0)
                    routes = float(merged.get("metadata_cpu_routes_sum", 0.0) or 0.0)
                    experts = float(merged.get("metadata_realized_cpu_expert_count_sum", 0.0) or 0.0)
                    active = float(merged.get("metadata_pre_transfer_active_count_sum", 0.0) or 0.0)
                    layer_count = float(merged.get("metadata_layer_count", 0.0) or 0.0)
                    merged["metadata_available"] = True
                    merged["metadata_cpu_routes_per_token"] = routes / token_count if token_count > 0.0 else 0.0
                    merged["metadata_cpu_experts_per_token"] = experts / token_count if token_count > 0.0 else 0.0
                    merged["metadata_cpu_route_miss_ratio"] = routes / active if active > 0.0 else 0.0
                    merged["metadata_cpu_experts_per_layer"] = experts / layer_count if layer_count > 0.0 else 0.0
                    execution_layer_count = float(
                        merged.get("metadata_execution_layer_count", 0.0) or 0.0
                    )
                    execution_routes = float(
                        merged.get("metadata_execution_cpu_routes_sum", 0.0) or 0.0
                    )
                    execution_experts = float(
                        merged.get("metadata_execution_cpu_experts_sum", 0.0) or 0.0
                    )
                    merged["metadata_execution_available"] = bool(
                        execution_layer_count > 0.0
                    )
                    merged.setdefault("metadata_execution_layer_count", 0.0)
                    merged.setdefault("metadata_execution_active_routes_sum", 0.0)
                    merged.setdefault("metadata_execution_cpu_routes_sum", 0.0)
                    merged.setdefault("metadata_execution_cpu_experts_sum", 0.0)
                    merged["metadata_execution_cpu_routes_per_layer"] = (
                        execution_routes / execution_layer_count
                        if execution_layer_count > 0.0
                        else 0.0
                    )
                    merged["metadata_execution_cpu_experts_per_layer"] = (
                        execution_experts / execution_layer_count
                        if execution_layer_count > 0.0
                        else 0.0
                    )
                else:
                    merged["metadata_available"] = False
                    merged.setdefault("metadata_cpu_routes_sum", 0.0)
                    merged.setdefault("metadata_realized_cpu_expert_count_sum", 0.0)
                    merged.setdefault("metadata_pre_transfer_active_count_sum", 0.0)
                    merged.setdefault("metadata_cpu_routes_per_token", 0.0)
                    merged.setdefault("metadata_cpu_experts_per_token", 0.0)
                    merged.setdefault("metadata_cpu_route_miss_ratio", 0.0)
                    merged.setdefault("metadata_cpu_experts_per_layer", 0.0)
                    merged.setdefault("metadata_execution_layer_count", 0.0)
                    merged.setdefault("metadata_execution_active_routes_sum", 0.0)
                    merged.setdefault("metadata_execution_cpu_routes_sum", 0.0)
                    merged.setdefault("metadata_execution_cpu_experts_sum", 0.0)
                    merged.setdefault("metadata_execution_available", False)
                    merged.setdefault("metadata_execution_cpu_routes_per_layer", 0.0)
                    merged.setdefault("metadata_execution_cpu_experts_per_layer", 0.0)
                merged["padding_token_count"] = max(
                    0,
                    int(merged.get("bucket", 0) or 0)
                    - int(merged.get("token_count", 0) or 0),
                )
                verify_call_records.append(merged)
            out["verify_call_records"] = verify_call_records
        decode_count = int(self._profile.get("decode_count", 0))
        graph_hit_count = int(self._profile.get("graph_hit_count", 0))
        out["graph_hit_rate"] = float(graph_hit_count / decode_count) if decode_count > 0 else 0.0
        profile_count = float(self._profile.get("moe_profile_count", 0.0))
        if profile_count > 0:
            out["cpu_route_ratio"] = float(self._profile.get("cpu_route_ratio_sum", 0.0) / profile_count)
            out["cpu_weight_mass_ratio"] = float(self._profile.get("cpu_weight_mass_ratio_sum", 0.0) / profile_count)
            out["activated_expert_set_size"] = float(self._profile.get("activated_expert_set_size_sum", 0.0) / profile_count)
            out["realized_cpu_expert_count"] = float(self._profile.get("realized_cpu_expert_count_sum", 0.0) / profile_count)
            out["pre_transfer_cache_miss"] = float(self._profile.get("pre_transfer_cache_miss_sum", 0.0) / profile_count)
            out["pre_transfer_active_count"] = float(self._profile.get("pre_transfer_active_count_sum", 0.0) / profile_count)
        else:
            out["cpu_route_ratio"] = 0.0
            out["cpu_weight_mass_ratio"] = 0.0
            out["activated_expert_set_size"] = 0.0
            out["realized_cpu_expert_count"] = 0.0
            out["pre_transfer_cache_miss"] = 0.0
            out["pre_transfer_active_count"] = 0.0
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
        if bool(getattr(self, "_draft_perfect_trace_enabled", False)):
            self._flush_pending_prefetch_metadata(block=True)
            self._finalize_perfect_match_pending()
            perfect_profile = getattr(self, "_draft_perfect_profile", {})
            for key, value in perfect_profile.items():
                out[f"draft_perfect_{key}"] = float(value)
            checked = float(out.get("draft_perfect_checked_tokens", 0.0))
            route_total = float(out.get("draft_perfect_route_total", 0.0))
            coverage_total = float(out.get("draft_perfect_coverage_total", 0.0))
            pred_row_total = float(out.get("draft_perfect_pred_row_total", 0.0))
            input_row_total = float(out.get("draft_perfect_input_row_total", 0.0))
            followup_events = float(out.get("draft_perfect_followup_events", 0.0))
            out["draft_perfect_trace_enabled"] = True
            out["draft_perfect_refill_rejected_enabled"] = bool(
                getattr(self, "_draft_perfect_refill_rejected", False)
            )
            out["draft_perfect_token_rate"] = (
                float(out.get("draft_perfect_perfect_tokens", 0.0)) / checked
                if checked > 0.0 else 0.0
            )
            out["draft_perfect_route_miss_ratio"] = (
                float(out.get("draft_perfect_route_miss", 0.0)) / route_total
                if route_total > 0.0 else 0.0
            )
            out["draft_perfect_coverage_ratio"] = (
                float(out.get("draft_perfect_coverage_hit", 0.0)) / coverage_total
                if coverage_total > 0.0 else 0.0
            )
            out["draft_perfect_pred_row_match_ratio"] = (
                float(out.get("draft_perfect_pred_row_hit", 0.0)) / pred_row_total
                if pred_row_total > 0.0 else 0.0
            )
            out["draft_perfect_input_row_match_ratio"] = (
                float(out.get("draft_perfect_input_row_hit", 0.0)) / input_row_total
                if input_row_total > 0.0 else 0.0
            )
            out["draft_perfect_prefix_ge1_rate"] = (
                float(out.get("draft_perfect_prefix_ge1_events", 0.0)) / followup_events
                if followup_events > 0.0 else 0.0
            )
            out["draft_perfect_oracle_covered_token_rate"] = (
                float(out.get("draft_perfect_oracle_covered_tokens", 0.0)) / checked
                if checked > 0.0 else 0.0
            )
            out["draft_perfect_oracle_prefix_ge1_rate"] = (
                float(out.get("draft_perfect_oracle_prefix_ge1_events", 0.0)) / followup_events
                if followup_events > 0.0 else 0.0
            )
            out["draft_perfect_records"] = list(getattr(self, "_draft_perfect_records", []))
        if reset:
            with self._prefetch_profile_lock:
                self._profile.clear()
                self._prefetch_trace_events.clear()
                self._verify_op_event_records.clear()
                self._draft_op_event_records.clear()
                self._verify_call_records.clear()
                self._verify_metadata_records_by_step.clear()
                self._verify_stream_timing_events.clear()
                self._verify_stream_ms_by_step.clear()
            if prefetch_runtime is not None:
                with self._prefetch_runtime_lock:
                    _ = prefetch_runtime.get_profile(reset=True)
        if bool(getattr(self, "_draft_perfect_trace_enabled", False)):
            self._draft_perfect_profile.clear()
            self._draft_perfect_records.clear()
            self._draft_perfect_pending = None
            self._draft_perfect_draft_meta_by_step.clear()
            pending = getattr(self, "_pending_prefetch_metadata", None)
            if pending is not None:
                pending.clear()
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
        self._warmup_verify_layer_timings()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def _warmup_verify_layer_timings(self) -> None:
        self._ensure_prefetch_internal_state()
        if not bool(getattr(self.config, "prefetch_verify_layer_enabled", True)):
            return
        if not getattr(self, "layer_caches", None):
            return
        if not hasattr(self.model, "set_verify_prefetch_controller"):
            return

        verify_tokens = max(1, int(getattr(self.config, "max_draft_tokens", 1)) + 1)
        verify_tokens = min(verify_tokens, int(self.config.max_model_len))
        num_seqs = max(1, min(self.config.max_num_seqs, self.config.max_num_batched_tokens // verify_tokens))
        seqs = [Sequence([0] * verify_tokens) for _ in range(num_seqs)]

        self._set_speculative_execution_mode("verify")
        self._verify_prefetch_active = True
        self._current_verify_prefetch_step_id = self._next_prefetch_step_id()
        self.model.set_verify_prefetch_controller(self)
        # Suppress prefetch during warmup: collect timing EMAs only, do not
        # modify GPU cache (publish/submit would permanently change active
        # slot contents and evict experts, breaking determinism between
        # prefetch-ON and prefetch-OFF runs).
        _saved_prefetch_runtime = getattr(self, "prefetch_runtime", None)
        if _saved_prefetch_runtime is not None:
            self.prefetch_runtime = None  # type: ignore[assignment]
        try:
            input_ids, positions = self.prepare_prefill(seqs)
            _ = self.model(input_ids, positions)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._poll_verify_layer_timing_events()
            if self.profile_enabled and self.rank == 0:
                with self._prefetch_profile_lock:
                    self._profile["verify_layer_timing_warmup_count"] += 1
        finally:
            if _saved_prefetch_runtime is not None:
                self.prefetch_runtime = _saved_prefetch_runtime  # type: ignore[assignment]
            self.model.set_verify_prefetch_controller(None)
            self._verify_prefetch_active = False
            self._current_verify_prefetch_step_id = -1
            self._set_speculative_execution_mode("normal")
            reset_context()

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
        if self.profile_enabled and self.rank == 0:
            replay_t0 = perf_counter()
            graph.replay()
            if getattr(self, "profile_cuda_sync", True) and torch.cuda.is_available():
                torch.cuda.synchronize()
            replay_ms = (perf_counter() - replay_t0) * 1000.0
            self._profile["graph_replay_count"] += 1
            self._profile["graph_hit_count"] += 1
            self._profile["standard_graph_replay_count"] += 1
            self._profile["standard_graph_replay_ms"] += replay_ms
        else:
            graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:bs])

    def _replay_draft_graph(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self._can_use_draft_segment_graph(input_ids.size(0)):
            return self._replay_draft_segment_graph(input_ids, positions)

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
        if self.profile_enabled and self.rank == 0:
            replay_t0 = perf_counter()
            graph.replay()
            if getattr(self, "profile_cuda_sync", True) and torch.cuda.is_available():
                torch.cuda.synchronize()
            replay_ms = (perf_counter() - replay_t0) * 1000.0
            self._profile["graph_replay_count"] += 1
            self._profile["graph_hit_count"] += 1
            self._profile["draft_graph_replay_count"] += 1
            self._profile["draft_graph_replay_ms"] += replay_ms
        else:
            graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:bs])

    def _can_use_draft_segment_graph(self, bs: int) -> bool:
        if not self._draft_segment_graph_enabled():
            return False
        if bs > getattr(getattr(self, "config", None), "draft_cuda_graph_max_bs", 512):
            return False
        graphs = getattr(self, "draft_segment_graphs", None)
        if not graphs:
            return False
        return any(bucket >= bs for bucket in self.draft_graph_bs)

    def _enqueue_draft_segment_metadata(
        self,
        *,
        step_id: int,
        token_capacity: int,
        layer_start_idx: int,
        layer_end_idx: int,
    ) -> None:
        defer_segment_metadata = os.getenv(
            "NANOVLLM_DRAFT_DEFER_SEGMENT_METADATA",
            "",
        ).strip().lower() in {"1", "true", "yes", "y", "on"}
        if defer_segment_metadata:
            if self.profile_enabled and self.rank == 0:
                with self._prefetch_profile_lock:
                    self._profile["draft_segment_metadata_deferred_count"] += 1
            return

        prefetch_runtime = getattr(self, "prefetch_runtime", None)
        runtime_meta_recorder = getattr(self, "runtime_meta_recorder", None)
        if prefetch_runtime is None or runtime_meta_recorder is None:
            return

        host_buffer_slot, _ = self._acquire_prefetch_host_buffer_slot(
            mode="draft",
            token_capacity=int(token_capacity),
        )
        if host_buffer_slot is None:
            return
        enqueue_t0 = perf_counter()
        handle = runtime_meta_recorder.offload_async(
            prefetch_runtime.metadata_stream,
            host_buffer_slot=host_buffer_slot,
            layer_start_idx=int(layer_start_idx),
            layer_end_idx=int(layer_end_idx),
        )
        enqueue_ms = (perf_counter() - enqueue_t0) * 1000.0
        if handle is None:
            return
        frontier = int(layer_end_idx) - 1
        self._enqueue_prefetch_metadata(
            mode="draft",
            step_id=int(step_id),
            handle=handle,
            enqueue_ms=enqueue_ms,
            host_buffer_slot=host_buffer_slot,
            submit_after_phase="after_draft_segment",
            frontier_layer_idx=frontier,
        )
        self._draft_segment_metadata_enqueued_step_id = int(step_id)
        if self.profile_enabled and self.rank == 0:
            with self._prefetch_profile_lock:
                self._profile["draft_segment_metadata_enqueue_count"] += 1
                self._profile["draft_segment_metadata_enqueue_ms"] += enqueue_ms
                self._profile["draft_segment_frontier_sum"] += frontier
        if not getattr(self, "_prefetch_async_enabled", False):
            self._flush_pending_prefetch_metadata(block=True)

    def _replay_draft_segment_graph(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        from nanovllm.utils.verify_op_events import draft_op_event_enabled

        bs = input_ids.size(0)
        context = get_context()
        bucket = next(x for x in self.draft_graph_bs if x >= bs)
        graphs = self.draft_segment_graphs[bucket]
        boundaries = self.draft_segment_boundaries[bucket]
        graph_vars = self.draft_segment_graph_vars
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables

        replay_t0 = perf_counter()
        step_id = int(getattr(self, "_active_draft_prefetch_step_id", -1))
        prefetch_runtime = getattr(self, "prefetch_runtime", None)
        dual_queue = self._dual_queue_prefetch_enabled()
        op_event_profile = draft_op_event_enabled()
        for segment_id, (graph, (layer_start, layer_end)) in enumerate(
            zip(graphs, boundaries, strict=True)
        ):
            if dual_queue and prefetch_runtime is not None:
                with self._prefetch_runtime_lock:
                    prefetch_runtime.on_draft_segment_start(
                        step_id=step_id,
                        segment_id=segment_id,
                        boundaries=boundaries,
                    )
            timing_start = self._start_dual_queue_segment_timing()
            segment_t0 = perf_counter()
            graph.replay()
            self._end_dual_queue_segment_timing("draft", segment_id, timing_start)
            segment_enqueue_ms = (perf_counter() - segment_t0) * 1000.0
            if self.profile_enabled and self.rank == 0:
                with self._prefetch_profile_lock:
                    self._profile["draft_segment_graph_replay_count"] += 1
                    self._profile["draft_segment_graph_replay_enqueue_ms"] += segment_enqueue_ms
            if op_event_profile:
                sync_t0 = perf_counter()
                torch.cuda.synchronize()
                sync_ms = (perf_counter() - sync_t0) * 1000.0
                if self.profile_enabled and self.rank == 0:
                    with self._prefetch_profile_lock:
                        self._profile["draft_op_event_sync_count"] += 1.0
                        self._profile["draft_op_event_sync_ms"] += sync_ms
                self._collect_draft_op_event_timings(
                    bucket=int(bucket),
                    segment_id=int(segment_id),
                    step_id=int(step_id),
                    token_count=int(bs),
                )
            if step_id >= 0:
                self._enqueue_draft_segment_metadata(
                    step_id=step_id,
                    token_capacity=int(bucket),
                    layer_start_idx=int(layer_start),
                    layer_end_idx=int(layer_end),
                )
            self._poll_dual_queue_segment_timings(block=False)

        if self.profile_enabled and self.rank == 0:
            if getattr(self, "profile_cuda_sync", True) and torch.cuda.is_available():
                torch.cuda.synchronize()
            replay_ms = (perf_counter() - replay_t0) * 1000.0
            self._profile["graph_replay_count"] += 1
            self._profile["graph_hit_count"] += 1
            self._profile["draft_graph_replay_count"] += 1
            self._profile["draft_graph_replay_ms"] += replay_ms
            self._profile["draft_segment_graph_replay_ms"] += replay_ms

        final_hidden = graph_vars["outputs"][:bs]
        extractor = getattr(self, "_acceptance_extractor", None)
        if extractor is not None:
            tail_graph = getattr(self, "draft_tail_graphs", {}).get(bucket)
            if tail_graph is not None:
                # Eager LM head -> eager token_features (stays on GPU) -> replay the
                # captured tail graph (route/history/predictor) -> alpha_buf/state_out.
                logits = self.model.compute_logits(final_hidden)
                extractor.set_token_features_from_logits(logits)
                tail_graph.replay()
                self._pending_acceptance = True
                return logits
        return self.model.compute_logits(final_hidden)

    def _can_use_draft_cudagraph(self, bs: int) -> bool:
        if not getattr(self.config, "draft_cuda_graph_enabled", True):
            return False
        if getattr(self, "enforce_eager", False):
            return False
        if getattr(self.config, "draft_top_c", 0) != 0 and not self._can_use_draft_cpu_cudagraph():
            return False
        if bs > getattr(self.config, "draft_cuda_graph_max_bs", 512):
            return False
        if self._can_use_draft_segment_graph(bs):
            return True
        if not hasattr(self, "draft_graphs") or not self.draft_graphs:
            return False
        return any(bucket >= bs for bucket in self.draft_graph_bs)

    def _can_use_draft_cpu_cudagraph(self) -> bool:
        return (
            getattr(self.config, "draft_top_c", 0) > 0
            and getattr(self.config, "draft_cuda_graph_cpu_backend", "none") in {"fused", "fused_sync"}
            and getattr(self.config, "cpu_expert_backend", "torch") == "fused"
            and bool(getattr(self.config, "cpu_expert_execution_enabled", False))
        )

    def _can_use_standard_cudagraph(self, bs: int) -> bool:
        if self.enforce_eager:
            return False
        if not hasattr(self, "graphs") or not self.graphs:
            return False
        return any(bucket >= bs for bucket in self.graph_bs)

    def run(self, seqs: list[Sequence], is_prefill: bool, return_logits: bool = False):
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
            if host_buffer_slot is not None:
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
        if return_logits and self.rank == 0:
            return token_ids, logits
        return token_ids

    def forget_acceptance_state(self, seq_ids) -> None:
        """Drop per-sequence acceptance-predictor history for finished sequences."""
        extractor = getattr(self, "_acceptance_extractor", None)
        if extractor is not None and seq_ids:
            extractor.forget(seq_ids)

    def _configure_verify_cost_proxy(self) -> None:
        mode = str(
            getattr(self.config, "draft_tpot_verify_model_mode", "off")
        ).strip().lower()
        if self.rank != 0:
            return
        extractor = getattr(self, "_acceptance_extractor", None)
        if mode == "off":
            if bool(getattr(self.config, "transfer_aware_profile", False)):
                if extractor is None:
                    raise ValueError(
                        "transfer-aware profile requires the acceptance predictor "
                        "route recorder"
                    )
                extractor.original_route_readback_enabled = True
            return

        artifact_path = str(
            getattr(self.config, "draft_tpot_verify_model_path", "")
        )
        stop_rule = str(
            getattr(self.config, "draft_tpot_stop_rule", "")
        ).strip().lower()
        if stop_rule == "transfer_aware_step":
            with open(artifact_path, encoding="utf-8") as stream:
                artifact_header = json.load(stream)
            schema_version = int(artifact_header.get("schema_version", 1))
            if schema_version != 3:
                raise ValueError(
                    "transfer_aware_step cannot use a legacy v2 verify-cost artifact"
                )
            self._configure_transfer_aware_verify_model(
                artifact_path=artifact_path,
                mode=mode,
            )
            return
        from nanovllm.engine.speculative.verify_cost_model import (
            DraftRouteCostProxy,
            VerifyTimeCostModel,
        )

        model = VerifyTimeCostModel.load(
            artifact_path
        )
        hf_config = self.config.hf_config
        expected = (
            int(hf_config.num_hidden_layers),
            int(hf_config.num_experts),
            int(hf_config.num_experts_per_tok),
        )
        calibrated = (model.num_layers, model.num_experts, model.top_k)
        if calibrated != expected:
            raise ValueError(
                f"verify cost model shape {calibrated} does not match runtime {expected}"
            )
        if mode == "active" and not bool(model.artifact.get("accuracy_gate_passed")):
            raise ValueError("active verify cost model did not pass its accuracy gate")
        if mode == "active":
            acceptance_strategy = str(
                getattr(self.config, "acceptance_strategy", "greedy")
            ).strip().lower()
            is_sampling = acceptance_strategy in {
                "standard_sampling",
                "sampling",
                "spec_sampling",
            }
            if getattr(model, "protocol_adjustment", None) is not None:
                model.validate_protocol(acceptance_strategy=acceptance_strategy)
            deployment_field = (
                "sampling_deployment_validation"
                if is_sampling
                else "deployment_validation"
            )
            deployment = model.artifact.get(deployment_field, {})
            if not isinstance(deployment, dict) or not bool(deployment.get("passed")):
                raise ValueError(
                    "active verify cost model lacks a passing "
                    + ("sampling " if is_sampling else "")
                    + "shadow deployment validation"
                )
            if str(deployment.get("gate_version", "")) != "v2":
                raise ValueError(
                    "active verify cost model requires a v2 "
                    + ("sampling " if is_sampling else "")
                    + "shadow deployment gate"
                )
            validated_model_id = str(deployment.get("model_id", "") or "")
            if validated_model_id != str(model.model_id):
                raise ValueError(
                    "shadow deployment validation is not bound to the active "
                    f"model id: validated={validated_model_id!r} "
                    f"active={model.model_id!r}"
                )
            if (
                model.proxy_workload_model is None
                or not bool(model.artifact.get("proxy_workload_gate_passed"))
            ):
                raise ValueError(
                    "active verify cost model did not pass its causal workload "
                    "proxy gate"
                )
            if str(model.artifact.get("proxy_workload_gate_version", "")) != "v2":
                raise ValueError(
                    "active verify cost model requires a v2 causal workload proxy gate"
                )
            setattr(
                self.config,
                "draft_tpot_verify_model_sampling_validated",
                bool(is_sampling),
            )
            protocol_adjustment = (
                getattr(model, "protocol_adjustment", None) or {}
            )
            setattr(
                self.config,
                "draft_tpot_verify_model_temperature",
                protocol_adjustment.get("temperature"),
            )

        resolved_backend = str(getattr(self.config, "kt_direct_backend", "auto"))
        resolved_threads = int(getattr(self.config, "kt_num_threads", 0))
        for layer in getattr(getattr(self.model, "model", None), "layers", []):
            backend = getattr(getattr(layer, "mlp", None), "cpu_backend", None)
            if backend is None:
                continue
            resolved_backend = str(
                getattr(backend, "kt_selected_backend", resolved_backend)
            )
            runtime = getattr(backend, "runtime", None)
            resolved_threads = int(
                getattr(runtime, "kt_num_threads", resolved_threads)
            )
            break
        try:
            kt_version = importlib.metadata.version("kt-kernel")
        except importlib.metadata.PackageNotFoundError:
            kt_version = "unknown"
        model.validate_fingerprint(
            {
                "cpu_model": _cpu_model_name(),
                "gpu_model": torch.cuda.get_device_name(torch.cuda.current_device()),
                "kt_kernel_version": kt_version,
                "kt_num_threads": resolved_threads,
                "kt_backend": resolved_backend,
            }
        )
        self._verify_cost_model = model
        self._verify_cost_proxy = DraftRouteCostProxy(model)
        self._verify_cost_schema_version = int(
            model.artifact.get("schema_version", 1)
        )
        self._verify_cost_cache_masks = tuple(
            (int(layer_idx), cache.cached_expert_mask_host)
            for layer_idx, cache in getattr(self, "layer_caches", {}).items()
            if hasattr(cache, "cached_expert_mask_host")
        )
        extractor = getattr(self, "_acceptance_extractor", None)
        if extractor is not None:
            extractor.original_route_readback_enabled = (
                mode == "shadow"
                or str(getattr(self.config, "draft_stop_policy", "none"))
                .strip()
                .lower()
                == "tpot"
            )

    def _configure_transfer_aware_verify_model(
        self,
        *,
        artifact_path: str,
        mode: str,
    ) -> None:
        from nanovllm.engine.speculative.transfer_aware_cost_model import (
            TransferAwareVerifyCostModel,
            bind_runtime_cache_host_matrices,
        )

        model = TransferAwareVerifyCostModel.load(artifact_path)
        hf_config = self.config.hf_config
        expected = (
            int(hf_config.num_hidden_layers),
            int(hf_config.num_experts),
            int(hf_config.num_experts_per_tok),
        )
        calibrated = (model.num_layers, model.num_experts, model.top_k)
        if calibrated != expected:
            raise ValueError(
                f"v3 verify model shape {calibrated} does not match runtime {expected}"
            )
        if mode == "active" and str(
            getattr(self.config, "draft_tpot_stop_rule", "")
        ).strip().lower() != "transfer_aware_step":
            raise ValueError(
                "a v3 active verify model requires transfer_aware_step"
            )

        slot_counts = [
            int(getattr(cache, "num_slots", len(getattr(cache, "slot_to_expert", ()))))
            for cache in self.layer_caches.values()
        ]
        cache_ratio = (
            float(sum(slot_counts))
            / float(max(1, len(slot_counts) * model.num_experts))
            if slot_counts
            else float(getattr(self.config, "heterogeneous_slots_per_layer", 0))
            / float(max(1, model.num_experts))
        )
        protocol = model.artifact.get("protocol", {})
        temperature = (
            protocol.get("temperature")
            if isinstance(protocol, dict)
            else None
        )
        model.validate_runtime(
            {
                "batch_size": int(getattr(self.config, "max_num_seqs", 1)),
                "acceptance_strategy": str(
                    getattr(self.config, "acceptance_strategy", "")
                ).strip().lower(),
                # Per-request temperature is checked in SpeculativeEngine before
                # the first draft. Bind the expected value here.
                "temperature": temperature,
                "cache_ratio": cache_ratio,
                "max_draft_tokens": int(self.config.max_draft_tokens),
                "prefetch_runtime_kind": str(
                    getattr(self.config, "prefetch_runtime_kind", "")
                ),
                "buckets": tuple(
                    int(value)
                    for value in self.config.verify_cuda_graph_bucket_steps
                ),
            }
        )

        resolved_backend = str(getattr(self.config, "kt_direct_backend", "auto"))
        resolved_threads = int(getattr(self.config, "kt_num_threads", 0))
        for layer in getattr(getattr(self.model, "model", None), "layers", []):
            backend = getattr(getattr(layer, "mlp", None), "cpu_backend", None)
            if backend is None:
                continue
            resolved_backend = str(
                getattr(backend, "kt_selected_backend", resolved_backend)
            )
            runtime = getattr(backend, "runtime", None)
            resolved_threads = int(
                getattr(runtime, "kt_num_threads", resolved_threads)
            )
            break
        try:
            kt_version = importlib.metadata.version("kt-kernel")
        except importlib.metadata.PackageNotFoundError:
            kt_version = "unknown"
        model.validate_fingerprint(
            {
                "cpu_model": _cpu_model_name(),
                "gpu_model": torch.cuda.get_device_name(
                    torch.cuda.current_device()
                ),
                "kt_kernel_version": kt_version,
                "kt_num_threads": resolved_threads,
                "kt_backend": resolved_backend,
            }
        )
        self._verify_cost_model = model
        self._verify_cost_cache_host_matrices = (
            bind_runtime_cache_host_matrices(
                layer_caches=self.layer_caches,
                num_layers=model.num_layers,
                num_experts=model.num_experts,
            )
        )
        # This marker also enables the already-compact original-route D2H slice.
        self._verify_cost_proxy = model.demand
        self._verify_cost_schema_version = 3
        extractor = getattr(self, "_acceptance_extractor", None)
        if extractor is None:
            raise ValueError(
                "v3 verify model requires the acceptance predictor route recorder"
            )
        extractor.original_route_readback_enabled = True
        acceptance_strategy = str(
            getattr(self.config, "acceptance_strategy", "")
        ).strip().lower()
        setattr(
            self.config,
            "draft_tpot_verify_model_sampling_validated",
            acceptance_strategy
            in {"standard_sampling", "sampling", "spec_sampling"},
        )
        setattr(
            self.config,
            "draft_tpot_verify_model_temperature",
            temperature,
        )

    def _predict_verify_cost_from_draft_routes(
        self,
        *,
        seq_count: int,
        original_routes=None,
        reset_round: bool = False,
        predict: bool = True,
        lookahead_logical_tokens: int | None = None,
        next_draft_ms: float | None = None,
    ) -> dict[str, object] | None:
        if int(getattr(self, "_verify_cost_schema_version", 0)) == 3:
            return self._predict_transfer_aware_verify_cost(
                seq_count=seq_count,
                original_routes=original_routes,
                reset_round=reset_round,
                predict=predict,
                next_draft_ms=next_draft_ms,
            )
        proxy = getattr(self, "_verify_cost_proxy", None)
        model = getattr(self, "_verify_cost_model", None)
        if proxy is None or model is None:
            return None
        profile_proxy = bool(self.profile_enabled and self.rank == 0)
        total_t0 = perf_counter() if profile_proxy else 0.0
        observe_t0 = perf_counter() if profile_proxy else 0.0
        if reset_round or not bool(getattr(self, "_verify_cost_round_active", False)):
            proxy.reset()
            self._verify_cost_round_active = True
            self._verify_cost_latest_prediction = None
        if original_routes is not None:
            proxy.observe(original_routes)
        observe_ms = (
            (perf_counter() - observe_t0) * 1000.0 if profile_proxy else 0.0
        )
        if not predict:
            if profile_proxy:
                with self._prefetch_profile_lock:
                    self._profile["verify_cost_route_observe_only_count"] += 1.0
                    self._profile["verify_cost_route_observe_only_ms"] += observe_ms
            return None
        logical_tokens = int(proxy.known_rows) + int(seq_count)
        if bool(getattr(self.config, "verify_cuda_graph", False)):
            bucket = int(self._select_verify_bucket(logical_tokens))
        else:
            bucket = logical_tokens
        cache_t0 = perf_counter() if profile_proxy else 0.0
        ready_direct_active = 0
        ready_experts: list[tuple[int, int]] = []
        prefetch_runtime = getattr(self, "prefetch_runtime", None)
        if prefetch_runtime is not None:
            with self._prefetch_runtime_lock:
                for ticket in getattr(prefetch_runtime, "inflight", {}).values():
                    if not bool(getattr(ticket, "direct_active", False)):
                        continue
                    ready = bool(getattr(ticket, "ready", False))
                    ready_event = getattr(ticket, "ready_event", None)
                    if not ready and ready_event is not None:
                        ready = bool(ready_event.query())
                    if not ready:
                        continue
                    layer_idx = int(ticket.layer_idx)
                    ready_experts.append((layer_idx, int(ticket.expert_idx)))
                    ready_direct_active += 1
                uncached_mask = proxy.build_uncached_mask_from_host_masks(
                    getattr(self, "_verify_cost_cache_masks", ()),
                    additional_cached_experts=ready_experts,
                )
        else:
            uncached_mask = proxy.build_uncached_mask_from_host_masks(
                getattr(self, "_verify_cost_cache_masks", ()),
            )
        cached_expert_count = int(uncached_mask.size - uncached_mask.sum())
        cache_ms = (
            (perf_counter() - cache_t0) * 1000.0 if profile_proxy else 0.0
        )
        estimate_t0 = perf_counter() if profile_proxy else 0.0
        estimate = proxy.estimate_summary(
            bucket=bucket,
            logical_tokens=logical_tokens,
            uncached_mask=uncached_mask,
        )
        estimate_ms = (
            (perf_counter() - estimate_t0) * 1000.0 if profile_proxy else 0.0
        )
        prediction_t0 = perf_counter() if profile_proxy else 0.0
        prediction = model.predict_proxy_summary(
            estimate,
            cached_expert_count=cached_expert_count,
            ready_direct_active_experts=ready_direct_active,
        )
        prediction_ms = (
            (perf_counter() - prediction_t0) * 1000.0 if profile_proxy else 0.0
        )
        result: dict[str, object] = {
            "verify_cost_model_id": str(model.model_id),
            "verify_cost_prediction_ms": float(prediction.total_ms),
            "verify_cost_fixed_ms": float(prediction.fixed_ms),
            "verify_cost_exposed_cpu_ms": float(prediction.exposed_cpu_ms),
            "verify_cost_error_p90_ms": float(prediction.error_p90_ms),
            "verify_cost_bucket": int(estimate.bucket),
            "verify_cost_logical_tokens": int(estimate.logical_tokens),
            "verify_cost_known_rows": int(estimate.known_rows),
            "verify_cost_unknown_rows": int(estimate.unknown_rows),
            "verify_cost_known_cpu_routes": float(estimate.known_cpu_routes),
            "verify_cost_prior_cpu_routes": float(estimate.prior_cpu_routes),
            "verify_cost_cpu_routes": float(prediction.estimated_cpu_routes),
            "verify_cost_cpu_experts": float(prediction.estimated_cpu_experts),
            "verify_cost_proxy_cpu_routes": float(estimate.proxy_cpu_routes),
            "verify_cost_proxy_cpu_experts": float(estimate.proxy_cpu_experts),
            "verify_cost_cached_expert_count": int(cached_expert_count),
            "verify_cost_ready_direct_active_experts": int(ready_direct_active),
        }
        lookahead_t0 = perf_counter() if profile_proxy else 0.0
        current_logical_tokens = int(proxy.known_rows) + int(seq_count)
        if lookahead_logical_tokens is None:
            lookahead_logical_tokens = current_logical_tokens + int(seq_count)
        else:
            lookahead_logical_tokens = int(lookahead_logical_tokens)
            if lookahead_logical_tokens <= current_logical_tokens:
                raise ValueError(
                    "verify-cost lookahead target must be beyond the current "
                    f"logical length: current={current_logical_tokens} "
                    f"target={lookahead_logical_tokens}"
                )
        if lookahead_logical_tokens <= max(model.buckets, default=0):
            lookahead_bucket = (
                int(self._select_verify_bucket(lookahead_logical_tokens))
                if bool(getattr(self.config, "verify_cuda_graph", False))
                else lookahead_logical_tokens
            )
            lookahead_estimate = proxy.estimate_summary(
                bucket=lookahead_bucket,
                logical_tokens=lookahead_logical_tokens,
                uncached_mask=uncached_mask,
            )
            lookahead_prediction = model.predict_proxy_summary(
                lookahead_estimate,
                cached_expert_count=cached_expert_count,
                ready_direct_active_experts=ready_direct_active,
            )
            result.update(
                {
                    "verify_cost_lookahead_prediction_ms": float(
                        lookahead_prediction.total_ms
                    ),
                    "verify_cost_lookahead_bucket": int(lookahead_bucket),
                    "verify_cost_lookahead_logical_tokens": int(
                        lookahead_logical_tokens
                    ),
                    "verify_cost_lookahead_cpu_routes": float(
                        lookahead_prediction.estimated_cpu_routes
                    ),
                    "verify_cost_lookahead_cpu_experts": float(
                        lookahead_prediction.estimated_cpu_experts
                    ),
                }
            )
        lookahead_ms = (
            (perf_counter() - lookahead_t0) * 1000.0 if profile_proxy else 0.0
        )
        if profile_proxy:
            result["verify_cost_layer_proxy_cpu_routes"] = [
                float(value)
                for value in estimate.layer_route_counts.sum(axis=1)
            ]
            result["verify_cost_layer_proxy_cpu_experts"] = [
                float(value)
                for value in estimate.layer_route_counts.clip(0.0, 1.0).sum(axis=1)
            ]
        if profile_proxy:
            total_ms = (perf_counter() - total_t0) * 1000.0
            result.update(
                {
                    "verify_cost_runtime_route_observe_ms": observe_ms,
                    "verify_cost_runtime_cache_snapshot_ms": cache_ms,
                    "verify_cost_runtime_current_estimate_ms": estimate_ms,
                    "verify_cost_runtime_model_predict_ms": prediction_ms,
                    "verify_cost_runtime_lookahead_ms": lookahead_ms,
                    "verify_cost_runtime_total_ms": total_ms,
                }
            )
        self._verify_cost_latest_prediction = dict(result)
        if profile_proxy:
            with self._prefetch_profile_lock:
                self._profile["verify_cost_proxy_prediction_count"] += 1.0
                self._profile["verify_cost_proxy_prediction_ms_sum"] += float(
                    prediction.total_ms
                )
                self._profile["verify_cost_route_observe_ms"] += observe_ms
                self._profile["verify_cost_cache_snapshot_ms"] += cache_ms
                self._profile["verify_cost_current_estimate_ms"] += estimate_ms
                self._profile["verify_cost_model_predict_ms"] += prediction_ms
                self._profile["verify_cost_lookahead_ms"] += lookahead_ms
                self._profile["verify_cost_proxy_total_ms"] += total_ms
        return result

    def _predict_transfer_aware_verify_cost(
        self,
        *,
        seq_count: int,
        original_routes=None,
        reset_round: bool = False,
        predict: bool = True,
        next_draft_ms: float | None = None,
    ) -> dict[str, object] | None:
        from nanovllm.engine.speculative.transfer_aware_cost_model import (
            prediction_to_runtime_dict,
            snapshot_runtime_state,
        )

        model = getattr(self, "_verify_cost_model", None)
        if model is None:
            return None
        profile_proxy = bool(self.profile_enabled and self.rank == 0)
        total_t0 = perf_counter() if profile_proxy else 0.0
        if reset_round or not bool(getattr(self, "_verify_cost_round_active", False)):
            model.reset()
            self._verify_cost_round_active = True
            self._verify_cost_latest_prediction = None
            self._transfer_aware_profile_routes.clear()
        if original_routes is not None:
            model.observe(original_routes)
            if bool(getattr(self.config, "transfer_aware_profile", False)):
                self._transfer_aware_profile_routes.append(
                    original_routes.astype("int16", copy=True).tolist()
                )
        if not predict:
            if profile_proxy:
                with self._prefetch_profile_lock:
                    self._profile[
                        "verify_cost_route_observe_only_count"
                    ] += 1.0
            return None

        snapshot_t0 = perf_counter() if profile_proxy else 0.0
        prefetch_runtime = getattr(self, "prefetch_runtime", None)
        snapshot_scratch = getattr(
            self, "_verify_cost_snapshot_scratch", None
        )
        cache_host = getattr(
            self, "_verify_cost_cache_host_matrices", None
        )
        cache_strategy = str(
            getattr(self.config, "cache_strategy", "lru")
        ).strip().lower()
        access_host = (
            cache_host.access_count
            if cache_host is not None
            and cache_strategy
            in {"lfu", "lfu_rankguard", "lfu_rankguard_online"}
            else (
                None
                if cache_host is None
                else cache_host.last_access
            )
        )
        resident_host = (
            None if cache_host is None else cache_host.resident
        )
        slot_counts_host = (
            None if cache_host is None else cache_host.slot_counts
        )
        if prefetch_runtime is not None:
            with self._prefetch_runtime_lock:
                state = snapshot_runtime_state(
                    layer_caches=self.layer_caches,
                    prefetch_runtime=prefetch_runtime,
                    num_layers=model.num_layers,
                    num_experts=model.num_experts,
                    transfer_ms=model.simulator.transfer_ms,
                    materialize_layers=False,
                    array_scratch=snapshot_scratch,
                    resident_host_matrix=resident_host,
                    access_host_matrix=access_host,
                    slot_counts_host=slot_counts_host,
                )
        else:
            state = snapshot_runtime_state(
                layer_caches=self.layer_caches,
                prefetch_runtime=None,
                num_layers=model.num_layers,
                num_experts=model.num_experts,
                transfer_ms=model.simulator.transfer_ms,
                materialize_layers=False,
                array_scratch=snapshot_scratch,
                resident_host_matrix=resident_host,
                access_host_matrix=access_host,
                slot_counts_host=slot_counts_host,
            )
        self._verify_cost_snapshot_scratch = state.array_state
        snapshot_ms = (
            (perf_counter() - snapshot_t0) * 1000.0 if profile_proxy else 0.0
        )
        logical_tokens = int(model.demand.known_rows) + int(seq_count)
        try:
            current, next_prediction = model.predict_pair(
                state=state,
                logical_tokens=logical_tokens,
                vpb=int(
                    getattr(
                        self.config,
                        "verify_prefetch_max_per_boundary",
                        0,
                    )
                ),
                next_draft_ms=float(
                    next_draft_ms
                    if next_draft_ms is not None
                    else getattr(self.config, "draft_tpot_td_ms", 19.0)
                ),
            )
        except (KeyError, ValueError, IndexError) as error:
            # A transient/incomplete state must never trigger an early stop.
            result = {
                "verify_cost_schema_version": 3,
                "verify_cost_model_id": str(model.model_id),
                "verify_cost_state_complete": False,
            }
            if profile_proxy:
                result["verify_cost_fail_open_reason"] = (
                    f"{type(error).__name__}: {error}"
                )
                with self._prefetch_profile_lock:
                    self._profile[
                        "verify_cost_transfer_aware_fail_open_count"
                    ] += 1.0
            self._verify_cost_latest_prediction = dict(result)
            return result
        result = prediction_to_runtime_dict(model, current, next_prediction)
        result["verify_cost_known_rows"] = int(model.demand.known_rows)
        if profile_proxy:
            result.update(
                {
                    "verify_cost_runtime_cache_snapshot_ms": snapshot_ms,
                    "verify_cost_runtime_total_ms": (
                        perf_counter() - total_t0
                    )
                    * 1000.0,
                }
            )
            if bool(
                getattr(self.config, "transfer_aware_profile", False)
            ) or str(
                getattr(self.config, "perf_profile_level", "basic")
            ).strip().lower() == "detailed":
                result.update(
                    {
                        "verify_cost_segment_workloads": [
                        {
                            "segment_id": segment.segment_id,
                            "first_layer": segment.first_layer,
                            "last_layer": segment.last_layer,
                            "cpu_experts": segment.cpu_experts,
                            "cpu_routes": segment.cpu_routes,
                            "max_layer_experts": segment.max_layer_experts,
                            "transfer_submits": segment.transfer_submits,
                        }
                        for segment in current.segments
                        ],
                        "verify_cost_lookahead_segment_workloads": [
                        {
                            "segment_id": segment.segment_id,
                            "first_layer": segment.first_layer,
                            "last_layer": segment.last_layer,
                            "cpu_experts": segment.cpu_experts,
                            "cpu_routes": segment.cpu_routes,
                            "max_layer_experts": segment.max_layer_experts,
                            "transfer_submits": segment.transfer_submits,
                        }
                        for segment in next_prediction.segments
                        ],
                    }
                )
            with self._prefetch_profile_lock:
                self._profile["verify_cost_proxy_prediction_count"] += 1.0
                self._profile["verify_cost_cache_snapshot_ms"] += snapshot_ms
                self._profile["verify_cost_proxy_total_ms"] += float(
                    result["verify_cost_runtime_total_ms"]
                )
        self._verify_cost_latest_prediction = dict(result)
        return result

    def start_verify_cost_round(
        self,
        seq_count: int,
        predict: bool = True,
    ) -> dict[str, object] | None:
        """Predict the no-draft verify baseline before the first draft call."""
        if (
            getattr(self, "_verify_cost_model", None) is None
            and bool(getattr(self.config, "transfer_aware_profile", False))
        ):
            self._transfer_aware_profile_routes.clear()
            self._verify_cost_round_active = True
            return None
        return self._predict_verify_cost_from_draft_routes(
            seq_count=int(seq_count),
            reset_round=True,
            predict=bool(predict),
        )

    def _set_speculative_execution_mode(self, mode: str):
        if hasattr(self.model, "set_speculative_execution_mode"):
            draft_top_c = getattr(self.config, "draft_top_c", 0)
            self.model.set_speculative_execution_mode(mode, self.draft_scheduler, draft_top_c)

    @torch.inference_mode()
    def run_draft(
        self,
        seqs: list[Sequence],
        return_logits: bool = False,
        observe_verify_cost: bool = True,
        predict_verify_cost: bool = True,
        verify_cost_lookahead_tokens: int | None = None,
        verify_cost_next_draft_ms: float | None = None,
    ) -> tuple:
        """Draft decode path with explicit draft plan execution inside MoE blocks."""
        self._ensure_prefetch_internal_state()
        t0 = perf_counter()
        step_id = self._next_prefetch_step_id()
        prefetch_runtime = getattr(self, "prefetch_runtime", None)
        runtime_meta_recorder = getattr(self, "runtime_meta_recorder", None)
        if prefetch_runtime is not None and runtime_meta_recorder is not None:
            self._flush_pending_prefetch_metadata(block=False)

        mode_set_t0 = perf_counter()
        draft_graph_replay_mode = self._can_use_draft_cudagraph(len(seqs))
        draft_cpu_graph_mode = False
        if not draft_graph_replay_mode:
            self._set_speculative_execution_mode("draft")
            draft_cpu_graph_mode = self._can_use_draft_cpu_cudagraph()
            if draft_cpu_graph_mode and hasattr(self.model, "set_draft_cpu_graph_mode"):
                self.model.set_draft_cpu_graph_mode(True)
        elif self.profile_enabled and self.rank == 0:
            self._profile["run_draft_graph_mode_set_skipped_count"] += 1
        self._decode_graph_policy = "draft"
        self._active_draft_prefetch_step_id = int(step_id)
        self._draft_segment_metadata_enqueued_step_id = -1
        if prefetch_runtime is not None and self._prefetch_runtime_mode() == "draft_segment_indexed":
            with self._prefetch_runtime_lock:
                prefetch_runtime.begin_draft_iteration(step_id=step_id)
        mode_set_ms = (perf_counter() - mode_set_t0) * 1000.0
        if self.profile_enabled and self.rank == 0:
            self._profile["run_draft_mode_set_ms"] += mode_set_ms

        try:
            prefetch_before_ms = 0.0
            draft_capacity = len(seqs)
            if prefetch_runtime is not None and runtime_meta_recorder is not None:
                before_t0 = perf_counter()
                if self._can_use_draft_cudagraph(len(seqs)):
                    draft_capacity = next(x for x in self.draft_graph_bs if x >= len(seqs))

                with self._prefetch_runtime_lock:
                    prefetch_runtime.drain_direct_active_ready(step_id=step_id)
                # Predictive prefetcher: submit phase-1 cold-start here, AFTER the
                # drain (so it is not synchronized by it) and BEFORE self.run, so
                # its async H2D overlaps with segment-0 compute (guarded; legacy
                # runtime has no such method -> no-op).
                maybe_submit_phase1 = getattr(prefetch_runtime, "maybe_submit_phase1", None)
                if maybe_submit_phase1 is not None:
                    with self._prefetch_runtime_lock:
                        maybe_submit_phase1(step_id)
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

            acceptance_extractor = getattr(self, "_acceptance_extractor", None)
            if acceptance_extractor is not None:
                # Stage per-sequence history state into the static input buffer
                # before the draft forward; the tail graph consumes it on-GPU.
                self._pending_acceptance = False
                acceptance_extractor.write_state_in(seqs)

            core_run_t0 = perf_counter()
            profile_dir = os.getenv("NANOVLLM_DRAFT_TORCH_PROFILE_DIR", "").strip()
            capture_draft_profile = (
                bool(profile_dir)
                and self.rank == 0
                and not bool(getattr(self, "_draft_torch_profile_done", False))
            )
            if capture_draft_profile:
                os.makedirs(profile_dir, exist_ok=True)
                from torch.profiler import ProfilerActivity, profile, record_function

                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
                    with record_function("nanovllm::draft.core_run_actual"):
                        run_result = self.run(seqs, False, return_logits=return_logits)
                torch.cuda.synchronize()
                _export_torch_profile_summary(
                    prof,
                    profile_dir,
                    "draft_forward",
                    self.rank,
                    {
                        "draft_tokens": int(len(seqs)),
                        "draft_capacity": int(draft_capacity),
                        "used_draft_segment_graph": bool(self._can_use_draft_segment_graph(len(seqs))),
                        "used_draft_cuda_graph": bool(self._can_use_draft_cudagraph(len(seqs))),
                    },
                )
                self._draft_torch_profile_done = True
            else:
                run_result = self.run(seqs, False, return_logits=return_logits)
            if return_logits and self.rank == 0:
                token_ids, draft_logits = run_result
            else:
                token_ids = run_result
                draft_logits = None

            acceptance_alpha = None
            verify_cost_prediction = None
            if acceptance_extractor is not None and getattr(self, "_pending_acceptance", False):
                # Reads alpha + updated history state (one D2H, after the sampler
                # sync already forced by self.run) and advances host history state.
                include_routes = (
                    (
                        getattr(self, "_verify_cost_proxy", None) is not None
                        or bool(
                            getattr(
                                self.config,
                                "transfer_aware_profile",
                                False,
                            )
                        )
                    )
                    and bool(observe_verify_cost)
                )
                readback_t0 = (
                    perf_counter()
                    if self.profile_enabled and self.rank == 0
                    else 0.0
                )
                acceptance_outputs = acceptance_extractor.read_outputs(
                    seqs,
                    include_original_routes=include_routes,
                )
                if self.profile_enabled and self.rank == 0:
                    readback_ms = (perf_counter() - readback_t0) * 1000.0
                    with self._prefetch_profile_lock:
                        self._profile["acceptance_readback_ms"] += readback_ms
                        self._profile["acceptance_readback_count"] += 1.0
                if include_routes:
                    acceptance_alpha, original_routes = acceptance_outputs
                    if (
                        getattr(self, "_verify_cost_model", None) is None
                        and bool(
                            getattr(
                                self.config,
                                "transfer_aware_profile",
                                False,
                            )
                        )
                    ):
                        self._transfer_aware_profile_routes.append(
                            original_routes.astype(
                                "int16", copy=True
                            ).tolist()
                        )
                    verify_cost_prediction = self._predict_verify_cost_from_draft_routes(
                        seq_count=len(seqs),
                        original_routes=original_routes,
                        predict=bool(predict_verify_cost),
                        lookahead_logical_tokens=verify_cost_lookahead_tokens,
                        next_draft_ms=verify_cost_next_draft_ms,
                    )
                    if (
                        verify_cost_prediction is not None
                        and self.profile_enabled
                        and self.rank == 0
                    ):
                        verify_cost_prediction["verify_cost_runtime_readback_ms"] = (
                            readback_ms
                        )
                        verify_cost_prediction["verify_cost_runtime_hotpath_ms"] = (
                            readback_ms
                            + float(
                                verify_cost_prediction.get(
                                    "verify_cost_runtime_total_ms", 0.0
                                )
                            )
                        )
                else:
                    acceptance_alpha = acceptance_outputs
                self._pending_acceptance = False
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
                segment_metadata_enqueued = (
                    int(getattr(self, "_draft_segment_metadata_enqueued_step_id", -1)) == int(step_id)
                )
                if not segment_metadata_enqueued:
                    host_buffer_slot, _ = self._acquire_prefetch_host_buffer_slot(
                        mode="draft",
                        token_capacity=draft_capacity,
                    )
                    if host_buffer_slot is not None:
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
                            submit_after_phase=None if self._dual_queue_prefetch_enabled() else "after_draft",
                        )
                runtime_meta_recorder.reset()
                self._flush_pending_prefetch_metadata(block=False)
            prefetch_state = {"prefetch_step_id": step_id}
            if acceptance_alpha is not None:
                prefetch_state["acceptance_alpha"] = acceptance_alpha
            if verify_cost_prediction is not None:
                prefetch_state.update(verify_cost_prediction)
            if return_logits and self.rank == 0:
                return token_ids, prefetch_state, draft_logits
            return token_ids, prefetch_state
        finally:
            self._decode_graph_policy = "standard"
            self._active_draft_prefetch_step_id = -1
            self._draft_segment_metadata_enqueued_step_id = -1
            if draft_cpu_graph_mode and hasattr(self.model, "set_draft_cpu_graph_mode"):
                self.model.set_draft_cpu_graph_mode(False)
            if not draft_graph_replay_mode:
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
        # Keep prefetch fully opportunistic: verify should consume only transfers
        # that are already complete instead of draining metadata work on the main path.
        metadata_drain_ms = 0.0
        self._flush_pending_prefetch_metadata(block=False)
        t0 = perf_counter()
        with self._prefetch_runtime_lock:
            prefetch_runtime.wait_for_verify(
                step_id=step_id,
                timeout_ms=float(self.config.prefetch_verify_wait_ms),
            )
            if self._prefetch_runtime_mode() == "draft_segment_indexed":
                prefetch_runtime.end_draft_iteration()
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

    def _poll_verify_layer_timing_events(self) -> None:
        self._ensure_prefetch_internal_state()
        pending = getattr(self, "_verify_layer_timing_events", [])
        if not pending:
            return
        remaining: list[tuple[int, torch.cuda.Event, torch.cuda.Event]] = []
        for layer_idx, start_event, end_event in pending:
            if not bool(end_event.query()):
                remaining.append((layer_idx, start_event, end_event))
                continue
            compute_ms = float(start_event.elapsed_time(end_event))
            prev = self._verify_layer_compute_ms_ema.get(layer_idx)
            self._verify_layer_compute_ms_ema[layer_idx] = compute_ms if prev is None else 0.8 * prev + 0.2 * compute_ms
            if self.profile_enabled and self.rank == 0:
                with self._prefetch_profile_lock:
                    self._profile["verify_layer_timing_sample_count"] += 1
                    self._profile["verify_layer_compute_ms_sample_sum"] += compute_ms
        self._verify_layer_timing_events = remaining

    def _start_dual_queue_segment_timing(self):
        force_timing = any(
            os.getenv(key, "").strip().lower() in {"1", "true", "yes", "y", "on"}
            for key in (
                "NANOVLLM_SEGMENT_CUDA_EVENT_TIMING",
                "NANOVLLM_VERIFY_SEGMENT_CUDA_EVENT_TIMING",
                "NANOVLLM_DRAFT_SEGMENT_CUDA_EVENT_TIMING",
            )
        )
        if not self._dual_queue_prefetch_enabled() and not force_timing:
            return None
        if torch.cuda.is_available() and not torch.cuda.is_current_stream_capturing():
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream())
            return event
        return None

    def _end_dual_queue_segment_timing(self, phase: str, segment_id: int, start) -> None:
        if start is None:
            return
        if isinstance(start, torch.cuda.Event):
            end = torch.cuda.Event(enable_timing=True)
            end.record(torch.cuda.current_stream())
            self._dual_queue_segment_timing_events.append((str(phase), int(segment_id), start, end))
            return
        prefetch_runtime = getattr(self, "prefetch_runtime", None)
        if prefetch_runtime is not None:
            prefetch_runtime.record_segment_compute_ms(
                str(phase),
                int(segment_id),
                (perf_counter() - float(start)) * 1000.0,
            )

    def _poll_dual_queue_segment_timings(self, *, block: bool) -> None:
        pending = getattr(self, "_dual_queue_segment_timing_events", [])
        if not pending:
            return
        prefetch_runtime = getattr(self, "prefetch_runtime", None)
        remaining = []
        for phase, segment_id, start, end in pending:
            if block:
                end.synchronize()
            elif not bool(end.query()):
                remaining.append((phase, segment_id, start, end))
                continue
            elapsed_ms = float(start.elapsed_time(end))
            record_segment_compute_ms = getattr(
                prefetch_runtime,
                "record_segment_compute_ms",
                None,
            )
            if record_segment_compute_ms is not None:
                record_segment_compute_ms(
                    phase,
                    segment_id,
                    elapsed_ms,
                )
            if self.profile_enabled and self.rank == 0:
                with self._prefetch_profile_lock:
                    self._profile[f"{phase}_segment_cuda_event_count"] = (
                        float(self._profile.get(f"{phase}_segment_cuda_event_count", 0.0)) + 1.0
                    )
                    self._profile[f"{phase}_segment_cuda_event_ms"] = (
                        float(self._profile.get(f"{phase}_segment_cuda_event_ms", 0.0)) + elapsed_ms
                    )
                    self._profile[f"{phase}_segment_{int(segment_id)}_cuda_event_ms"] = (
                        float(self._profile.get(f"{phase}_segment_{int(segment_id)}_cuda_event_ms", 0.0)) + elapsed_ms
                    )
        self._dual_queue_segment_timing_events = remaining

    def _verify_stream_timing_enabled(self) -> bool:
        return (
            torch.cuda.is_available()
            and (
                os.getenv("NANOVLLM_VERIFY_COST_MODEL_PROFILE", "").strip().lower()
                in {"1", "true", "yes", "y", "on"}
                or os.getenv("NANOVLLM_VERIFY_STREAM_EVENT_TIMING", "").strip().lower()
                in {"1", "true", "yes", "y", "on"}
            )
        )

    def _start_verify_stream_timing(self):
        if not self._verify_stream_timing_enabled():
            return None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream())
        return start, end

    def _finish_verify_stream_timing(self, step_id: int, timing) -> None:
        if timing is None:
            return
        start, end = timing
        end.record(torch.cuda.current_stream())
        self._verify_stream_timing_events.append((int(step_id), start, end))

    def _poll_verify_stream_timings(self, *, block: bool) -> None:
        self._ensure_prefetch_internal_state()
        pending = getattr(self, "_verify_stream_timing_events", [])
        if not pending:
            return
        remaining = []
        for step_id, start, end in pending:
            if block:
                end.synchronize()
            elif not bool(end.query()):
                remaining.append((step_id, start, end))
                continue
            elapsed_ms = float(start.elapsed_time(end))
            self._verify_stream_ms_by_step[int(step_id)] = elapsed_ms
            if self.profile_enabled and self.rank == 0:
                with self._prefetch_profile_lock:
                    self._profile["verify_stream_event_count"] += 1.0
                    self._profile["verify_stream_event_ms"] += elapsed_ms
        self._verify_stream_timing_events = remaining

    def _collect_verify_op_event_timings(
        self,
        *,
        bucket: int,
        segment_id: int,
        step_id: int,
        token_count: int,
    ) -> None:
        from nanovllm.utils.verify_op_events import collect_verify_op_events, verify_op_event_enabled

        if not verify_op_event_enabled() or self.rank != 0:
            return
        rows = collect_verify_op_events(int(bucket), int(segment_id))
        if not rows:
            return
        with self._prefetch_profile_lock:
            for row in rows:
                label = str(row["label"])
                layer_idx = int(row["layer_idx"])
                elapsed_ms = float(row["elapsed_ms"])
                safe_label = "".join(ch if ch.isalnum() else "_" for ch in label)
                error = row.get("error")
                if error:
                    self._profile["verify_op_event_error_count"] += 1.0
                self._profile["verify_op_event_count"] += 1.0
                self._profile["verify_op_event_ms"] += elapsed_ms
                self._profile[f"verify_op_{safe_label}_count"] += 1.0
                self._profile[f"verify_op_{safe_label}_ms"] += elapsed_ms
                if layer_idx >= 0:
                    self._profile[f"verify_op_layer_{layer_idx}_{safe_label}_count"] += 1.0
                    self._profile[f"verify_op_layer_{layer_idx}_{safe_label}_ms"] += elapsed_ms
                self._verify_op_event_records.append(
                    {
                        "step_id": int(step_id),
                        "bucket": int(bucket),
                        "segment": int(segment_id),
                        "token_count": int(token_count),
                        "layer_idx": int(layer_idx),
                        "label": label,
                        "elapsed_ms": elapsed_ms,
                        "error": str(error or ""),
                    }
                )

    def _collect_draft_op_event_timings(
        self,
        *,
        bucket: int,
        segment_id: int,
        step_id: int,
        token_count: int,
    ) -> None:
        from nanovllm.utils.verify_op_events import collect_verify_op_events, draft_op_event_enabled

        if not draft_op_event_enabled() or self.rank != 0:
            return
        rows = collect_verify_op_events(int(bucket), int(segment_id), phase="draft")
        if not rows:
            return
        with self._prefetch_profile_lock:
            for row in rows:
                label = str(row["label"])
                layer_idx = int(row["layer_idx"])
                elapsed_ms = float(row["elapsed_ms"])
                safe_label = "".join(ch if ch.isalnum() else "_" for ch in label)
                error = row.get("error")
                if error:
                    self._profile["draft_op_event_error_count"] += 1.0
                self._profile["draft_op_event_count"] += 1.0
                self._profile["draft_op_event_ms"] += elapsed_ms
                self._profile[f"draft_op_{safe_label}_count"] += 1.0
                self._profile[f"draft_op_{safe_label}_ms"] += elapsed_ms
                if layer_idx >= 0:
                    self._profile[f"draft_op_layer_{layer_idx}_{safe_label}_count"] += 1.0
                    self._profile[f"draft_op_layer_{layer_idx}_{safe_label}_ms"] += elapsed_ms
                self._draft_op_event_records.append(
                    {
                        "step_id": int(step_id),
                        "bucket": int(bucket),
                        "segment": int(segment_id),
                        "token_count": int(token_count),
                        "layer_idx": int(layer_idx),
                        "label": label,
                        "elapsed_ms": elapsed_ms,
                        "error": str(error or ""),
                    }
                )

    def begin_dual_queue_calibration(self) -> None:
        prefetch_runtime = getattr(self, "prefetch_runtime", None)
        if isinstance(prefetch_runtime, DualQueuePrefetchRuntime):
            with self._prefetch_runtime_lock:
                prefetch_runtime.set_calibrating(True)

    def finalize_dual_queue_calibration(self) -> dict[str, float | int]:
        prefetch_runtime = getattr(self, "prefetch_runtime", None)
        if not isinstance(prefetch_runtime, DualQueuePrefetchRuntime):
            return {}
        self._poll_dual_queue_segment_timings(block=True)
        with self._prefetch_runtime_lock:
            return prefetch_runtime.finalize_calibration()

    def _record_verify_layer_timing_start(self, layer_idx: int) -> None:
        self._ensure_prefetch_internal_state()
        if torch.cuda.is_available() and not torch.cuda.is_current_stream_capturing():
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record(torch.cuda.current_stream())
            self._verify_layer_active_timing[int(layer_idx)] = start_event
            return
        self._verify_layer_active_timing[int(layer_idx)] = perf_counter()

    def _record_verify_layer_timing_end(self, layer_idx: int) -> None:
        self._ensure_prefetch_internal_state()
        start = self._verify_layer_active_timing.pop(int(layer_idx), None)
        if start is None:
            return
        if isinstance(start, torch.cuda.Event):
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record(torch.cuda.current_stream())
            self._verify_layer_timing_events.append((int(layer_idx), start, end_event))
            return
        compute_ms = (perf_counter() - float(start)) * 1000.0
        prev = self._verify_layer_compute_ms_ema.get(int(layer_idx))
        self._verify_layer_compute_ms_ema[int(layer_idx)] = compute_ms if prev is None else 0.8 * prev + 0.2 * compute_ms

    def before_verify_layer(self, layer_idx: int) -> None:
        self._ensure_prefetch_internal_state()
        if not bool(getattr(self, "_verify_prefetch_active", False)):
            return

        prefetch_runtime = getattr(self, "prefetch_runtime", None)
        if prefetch_runtime is None:
            self._record_verify_layer_timing_start(layer_idx)
            return

        self._poll_verify_layer_timing_events()
        step_id = int(getattr(self, "_current_verify_prefetch_step_id", -1))
        # Predictive prefetcher: release this layer's single-round protection as
        # verify reaches it (guarded; legacy runtime has no such method -> no-op).
        on_verify_layer_start = getattr(prefetch_runtime, "on_verify_layer_start", None)
        if on_verify_layer_start is not None:
            with self._prefetch_runtime_lock:
                on_verify_layer_start(int(layer_idx))
        with self._prefetch_runtime_lock:
            prefetch_runtime.publish_direct_active_ready(step_id=step_id)
        target_layer_idx = int(layer_idx) + 1
        submitted = 0
        available_ms = 0.0
        if target_layer_idx in getattr(self, "layer_caches", {}):
            compute_ms = float(self._verify_layer_compute_ms_ema.get(int(layer_idx), 0.0))
            safety_ratio = float(getattr(self.config, "prefetch_verify_layer_safety_ratio", 0.8))
            min_compute_ms = float(getattr(self.config, "prefetch_verify_layer_min_compute_ms", 0.05))
            if compute_ms >= min_compute_ms:
                available_ms = compute_ms * safety_ratio
                with self._prefetch_runtime_lock:
                    submitted = prefetch_runtime.submit_verify_layer_prefetch(
                        step_id=step_id,
                        target_layer_idx=target_layer_idx,
                        available_ms=available_ms,
                    )

        if self.profile_enabled and self.rank == 0:
            with self._prefetch_profile_lock:
                self._profile["verify_layer_prefetch_hook_count"] += 1
                self._profile["verify_layer_prefetch_hook_available_ms"] += available_ms
                self._profile["verify_layer_prefetch_hook_submit_count"] += submitted

        self._record_verify_layer_timing_start(layer_idx)

    def after_verify_layer(self, layer_idx: int) -> None:
        self._record_verify_layer_timing_end(layer_idx)

    @torch.inference_mode()
    def run_verify(
        self,
        seqs: list[Sequence],
        verify_lengths: list[int],
        return_logits: bool = False,
    ):
        """Run one-shot verify in prefill-like mode and return traces or logits."""
        self._ensure_prefetch_internal_state()
        had_verify_cost_round = bool(getattr(self, "_verify_cost_round_active", False))
        self._verify_cost_round_active = False
        if not had_verify_cost_round:
            self._verify_cost_latest_prediction = None
        self._poll_verify_stream_timings(block=False)
        total_t0 = perf_counter()
        step_id = self._next_prefetch_step_id()
        prefetch_runtime = getattr(self, "prefetch_runtime", None)
        runtime_meta_recorder = getattr(self, "runtime_meta_recorder", None)
        if prefetch_runtime is not None and runtime_meta_recorder is not None:
            self._flush_pending_prefetch_metadata(block=False)
        verify_delta_keys = (
            "verify_cpu_routes_sum",
            "verify_realized_cpu_expert_count_sum",
            "verify_pre_transfer_cache_miss_sum",
            "verify_pre_transfer_active_count_sum",
            "verify_activated_expert_set_size_sum",
            "verify_moe_profile_count",
            "verify_kt_hybrid_segment_graph_replay_count",
            "verify_kt_hybrid_graph_replay_count",
            "verify_segment_graph_replay_enqueue_count",
            "verify_segment_graph_replay_enqueue_ms",
            "verify_segment_metadata_enqueue_count",
            "verify_segment_metadata_enqueue_ms",
            "verify_deferred_segment_metadata_enqueue_total_ms",
            "run_verify_kt_hybrid_metadata_wait_ms",
            "run_verify_kt_hybrid_metadata_collect_ms",
            "run_verify_kt_hybrid_metadata_observe_ms",
            "run_verify_kt_hybrid_metadata_enqueue_ms",
            "run_verify_kt_hybrid_metadata_record_consumed_ms",
            "run_verify_kt_hybrid_metadata_mark_access_ms",
            "verify_tpot_dynamic_budget_applied_count",
            "verify_tpot_dynamic_budget_token_sum",
            "verify_tpot_dynamic_budget_value_sum",
        )
        if self.profile_enabled and self.rank == 0:
            with self._prefetch_profile_lock:
                _verify_profile_start = {
                    key: float(self._profile.get(key, 0.0))
                    for key in verify_delta_keys
                }
        else:
            _verify_profile_start = {}
        self._set_speculative_execution_mode("verify")
        t0 = perf_counter()
        input_ids, positions = self.prepare_prefill(seqs)
        self._record_profile("verify_prepare_prefill_ms", perf_counter() - t0)
        token_count = int(input_ids.numel())
        verify_bucket = (
            int(self._select_verify_bucket(token_count))
            if bool(getattr(self.config, "verify_cuda_graph", False))
            and getattr(self, "verify_graph_bs", None)
            else int(token_count)
        )
        transfer_profile_snapshot = None
        if (
            bool(getattr(self.config, "transfer_aware_profile", False))
            and self.rank == 0
        ):
            snapshot_now_ms = perf_counter() * 1000.0
            with self._prefetch_runtime_lock:
                cache_rows = {}
                for layer_idx, cache in self.layer_caches.items():
                    cache_rows[str(int(layer_idx))] = {
                        "resident_experts": [
                            int(expert_idx)
                            for expert_idx, cached in enumerate(
                                getattr(cache, "cached_expert_mask_host", ())
                            )
                            if bool(cached)
                        ],
                        "slots": [
                            int(value)
                            for value in getattr(cache, "slot_to_expert", ())
                        ],
                        "pending": [
                            int(value)
                            for value in getattr(
                                cache, "active_slot_pending_expert", ()
                            )
                        ],
                        "last_access_step": [
                            int(value)
                            for value in getattr(cache, "last_access_step", ())
                        ],
                        "access_count": [
                            int(value)
                            for value in getattr(cache, "access_count", ())
                        ],
                    }
                inflight_rows = []
                for ticket in getattr(prefetch_runtime, "inflight", {}).values():
                    ready = bool(getattr(ticket, "ready", False))
                    ready_event = getattr(ticket, "ready_event", None)
                    if not ready and ready_event is not None:
                        ready = bool(ready_event.query())
                    inflight_rows.append(
                        {
                            "step_id": int(ticket.step_id),
                            "layer_idx": int(ticket.layer_idx),
                            "expert_idx": int(ticket.expert_idx),
                            "source": str(ticket.source),
                            "ready": bool(ready),
                            "direct_active": bool(ticket.direct_active),
                            "active_slot_idx": int(ticket.active_slot_idx),
                            "active_slot_prev_expert": int(
                                ticket.active_slot_prev_expert
                            ),
                            "segment_id": int(ticket.segment_id),
                            "submit_ts_ms": float(ticket.submit_ts_ms),
                            "age_ms": max(
                                0.0,
                                snapshot_now_ms - float(ticket.submit_ts_ms),
                            ),
                            "num_bytes": int(ticket.num_bytes),
                            "transfer_stream_idx": int(
                                ticket.transfer_stream_idx
                            ),
                        }
                    )
                transfer_profile_snapshot = {
                    "cache_layers": cache_rows,
                    "inflight": inflight_rows,
                    "round_loaded": {
                        str(int(layer_idx)): sorted(
                            int(value) for value in experts
                        )
                        for layer_idx, experts in getattr(
                            prefetch_runtime, "_round_loaded", {}
                        ).items()
                    },
                }
        verify_forward_ms = 0.0
        _original_verify_prefetch_max = None
        _dynamic_verify_prefetch_budget_applied = False
        _dynamic_verify_prefetch_budget_value = int(getattr(self.config, "verify_prefetch_max_per_boundary", 0))
        if (
            bool(getattr(self.config, "verify_prefetch_tpot_dynamic_budget_enabled", False))
            and str(getattr(self.config, "draft_stop_policy", "")).strip().lower() == "tpot"
            and token_count <= int(getattr(
                self.config,
                "verify_prefetch_tpot_dynamic_budget_token_threshold",
                10,
            ))
        ):
            _original_verify_prefetch_max = int(getattr(self.config, "verify_prefetch_max_per_boundary", 0))
            small_budget = max(0, int(getattr(
                self.config,
                "verify_prefetch_tpot_dynamic_budget_small",
                _original_verify_prefetch_max,
            )))
            self.config.verify_prefetch_max_per_boundary = min(_original_verify_prefetch_max, small_budget)
            _dynamic_verify_prefetch_budget_applied = True
            _dynamic_verify_prefetch_budget_value = int(self.config.verify_prefetch_max_per_boundary)
            if self.profile_enabled and self.rank == 0:
                with self._prefetch_profile_lock:
                    self._profile["verify_tpot_dynamic_budget_applied_count"] += 1.0
                    self._profile["verify_tpot_dynamic_budget_token_sum"] += float(token_count)
                    self._profile["verify_tpot_dynamic_budget_value_sum"] += float(
                        self.config.verify_prefetch_max_per_boundary
                    )

        _use_kt_hybrid = bool(getattr(self.config, "verify_cuda_graph_kt_hybrid", False))
        _verify_meta_mode = "verify_kt_hybrid" if _use_kt_hybrid else "verify"
        skip_verify_metadata = self._skip_verify_metadata_offload()
        if prefetch_runtime is not None and runtime_meta_recorder is not None and not skip_verify_metadata:
            _meta_capacity = max(self.verify_graph_bs) if (_use_kt_hybrid and self.verify_graph_bs) else token_count
            self._wait_for_prefetch_device_reuse(mode=_verify_meta_mode, token_capacity=_meta_capacity)
            if not _use_kt_hybrid:
                runtime_meta_recorder.arm(
                    mode=_verify_meta_mode,
                    step_id=step_id,
                    token_capacity=_meta_capacity,
                    logical_token_count=token_count,
                )

        verify_layer_prefetch_enabled = (
            prefetch_runtime is not None
            and runtime_meta_recorder is not None
            and bool(getattr(self.config, "prefetch_verify_layer_enabled", True))
            and hasattr(self.model, "set_verify_prefetch_controller")
        )
        if verify_layer_prefetch_enabled:
            self._verify_prefetch_active = True
            self._current_verify_prefetch_step_id = int(step_id)
            self.model.set_verify_prefetch_controller(self)

        used_verify_segment_graph = False
        # compute_logits() slices prefill outputs to last token per sequence.
        # Verify needs logits for every queried token position.
        try:
            t0 = perf_counter()
            verify_stream_timing = self._start_verify_stream_timing()

            def _execute_verify_forward():
                nonlocal used_verify_segment_graph
                breakdown_sync = bool(os.getenv("NANOVLLM_VERIFY_BREAKDOWN_SYNC", "").strip())
                if self._can_use_verify_cudagraph(int(input_ids.numel())):
                    if getattr(self.config, "verify_cuda_graph_kt_hybrid", False):
                        if self._verify_segment_graph_enabled():
                            used_verify_segment_graph = True
                            hidden = self._run_verify_with_kt_hybrid_segment_graph(
                                input_ids,
                                positions,
                                step_id=step_id,
                            )
                        else:
                            hidden = self._run_verify_with_kt_hybrid_graph(
                                input_ids,
                                positions,
                                step_id=step_id,
                            )
                    else:
                        hidden = self._run_verify_with_prefix_graph(input_ids, positions)
                    lm_head_t0 = perf_counter()
                    logits_out = F.linear(hidden, self.model.lm_head.weight)
                    self._record_profile("verify_lm_head_enqueue_ms", perf_counter() - lm_head_t0)
                    if self.profile_enabled and self.rank == 0:
                        with self._prefetch_profile_lock:
                            self._profile["verify_lm_head_count"] += 1.0
                    if breakdown_sync:
                        lm_head_sync_t0 = perf_counter()
                        torch.cuda.synchronize()
                        self._record_profile("verify_lm_head_sync_ms", perf_counter() - lm_head_sync_t0)
                    return hidden, logits_out
                hidden = self.model(input_ids, positions)
                lm_head_t0 = perf_counter()
                logits_out = F.linear(hidden, self.model.lm_head.weight)
                self._record_profile("verify_lm_head_enqueue_ms", perf_counter() - lm_head_t0)
                if self.profile_enabled and self.rank == 0:
                    with self._prefetch_profile_lock:
                        self._profile["verify_lm_head_count"] += 1.0
                if breakdown_sync:
                    lm_head_sync_t0 = perf_counter()
                    torch.cuda.synchronize()
                    self._record_profile("verify_lm_head_sync_ms", perf_counter() - lm_head_sync_t0)
                return hidden, logits_out

            profile_dir = os.getenv("NANOVLLM_VERIFY_TORCH_PROFILE_DIR", "").strip()
            capture_verify_profile = (
                bool(profile_dir)
                and self.rank == 0
                and not bool(getattr(self, "_verify_torch_profile_done", False))
            )
            if capture_verify_profile:
                os.makedirs(profile_dir, exist_ok=True)
                from torch.profiler import ProfilerActivity, profile, record_function

                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
                    with record_function("nanovllm::verify.forward_actual"):
                        hidden_states, logits = _execute_verify_forward()
                torch.cuda.synchronize()
                _export_torch_profile_summary(
                    prof,
                    profile_dir,
                    "verify_forward",
                    self.rank,
                    {
                        "verify_tokens": int(input_ids.numel()),
                        "used_verify_segment_graph": bool(used_verify_segment_graph),
                        "used_verify_cuda_graph": bool(self._can_use_verify_cudagraph(int(input_ids.numel()))),
                        "verify_cuda_graph_kt_hybrid": bool(
                            getattr(self.config, "verify_cuda_graph_kt_hybrid", False)
                        ),
                    },
                )
                self._verify_torch_profile_done = True
            else:
                hidden_states, logits = _execute_verify_forward()
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
            self._finish_verify_stream_timing(step_id, verify_stream_timing)
            if self.profile_enabled and self.profile_cuda_sync:
                torch.cuda.synchronize()
            verify_forward_ms = (perf_counter() - t0) * 1000.0
            self._record_profile("verify_forward_ms", verify_forward_ms / 1000.0)
        finally:
            if _dynamic_verify_prefetch_budget_applied and _original_verify_prefetch_max is not None:
                self.config.verify_prefetch_max_per_boundary = _original_verify_prefetch_max
            if verify_layer_prefetch_enabled:
                self.model.set_verify_prefetch_controller(None)
                self._verify_prefetch_active = False
                self._current_verify_prefetch_step_id = -1
            self._set_speculative_execution_mode("normal")

        reset_context()
        if (
            self._dual_queue_prefetch_enabled()
            and prefetch_runtime is not None
            and not used_verify_segment_graph
        ):
            with self._prefetch_runtime_lock:
                prefetch_runtime.discard_verify_round()

        if self.rank != 0:
            return None

        def _record_verify_call(outputs_ready: bool) -> None:
            if not (self.profile_enabled and self.rank == 0):
                return
            total_ms = (perf_counter() - total_t0) * 1000.0
            with self._prefetch_profile_lock:
                deltas = {
                    key: float(self._profile.get(key, 0.0)) - float(_verify_profile_start.get(key, 0.0))
                    for key in verify_delta_keys
                }
                record = {
                    "call_index": int(len(self._verify_call_records)),
                    "step_id": int(step_id),
                    "token_count": int(token_count),
                    "verify_lengths": [int(x) for x in verify_lengths],
                    "seq_count": int(len(seqs)),
                    "bucket": int(verify_bucket),
                    "padding_tokens": int(max(0, verify_bucket - token_count)),
                    "exact_graph_bucket": bool(verify_bucket == token_count),
                    "used_cuda_graph": bool(self._can_use_verify_cudagraph(int(token_count))),
                    "used_kt_hybrid": bool(getattr(self.config, "verify_cuda_graph_kt_hybrid", False)),
                    "used_segment_graph": bool(used_verify_segment_graph),
                    "dynamic_budget_applied": bool(_dynamic_verify_prefetch_budget_applied),
                    "dynamic_budget_value": int(_dynamic_verify_prefetch_budget_value),
                    "forward_ms": float(verify_forward_ms),
                    "total_ms": float(total_ms),
                    "outputs_ready": bool(outputs_ready),
                    "return_logits": bool(return_logits),
                }
                if transfer_profile_snapshot is not None:
                    record["transfer_aware_pre_verify_state"] = (
                        transfer_profile_snapshot
                    )
                    record["draft_original_route_rows"] = list(
                        self._transfer_aware_profile_routes
                    )
                latest_prediction = getattr(
                    self, "_verify_cost_latest_prediction", None
                )
                if isinstance(latest_prediction, dict):
                    record.update(latest_prediction)
                    record["verify_cost_model_mode"] = str(
                        getattr(
                            self.config,
                            "draft_tpot_verify_model_mode",
                            "off",
                        )
                    )
                record.update({f"delta_{key}": value for key, value in deltas.items()})
                routes = float(record.get("delta_verify_cpu_routes_sum", 0.0))
                experts = float(record.get("delta_verify_realized_cpu_expert_count_sum", 0.0))
                active = float(record.get("delta_verify_pre_transfer_active_count_sum", 0.0))
                record["cpu_routes_per_token"] = routes / float(token_count) if token_count > 0 else 0.0
                record["cpu_experts_per_token"] = experts / float(token_count) if token_count > 0 else 0.0
                record["cpu_route_miss_ratio"] = routes / active if active > 0.0 else 0.0
                self._verify_call_records.append(record)

        if self.profile_enabled:
            self._record_profile("run_verify_total_ms", perf_counter() - total_t0)
            self._profile["run_verify_count"] += 1
            self._profile["verify_tokens_in_total"] += int(input_ids.numel())

        verify_outputs = []
        offset = 0
        for length in verify_lengths:
            seq_logits = logits[offset:offset + length]
            offset += length
            if return_logits:
                verify_outputs.append(seq_logits)
            else:
                verify_outputs.append(seq_logits.argmax(dim=-1).tolist())

        if prefetch_runtime is not None and runtime_meta_recorder is not None:
            with self._prefetch_runtime_lock:
                prefetch_runtime.publish_direct_active_ready(step_id=step_id)
            if skip_verify_metadata:
                if self.profile_enabled and self.rank == 0:
                    with self._prefetch_profile_lock:
                        self._profile["verify_metadata_skipped_count"] += 1
                if self._prefetch_runtime_mode() == "draft_segment_indexed":
                    with self._prefetch_runtime_lock:
                        prefetch_runtime.end_draft_iteration()
                _record_verify_call(outputs_ready=not return_logits)
                return verify_outputs
            _used_segment_graph = used_verify_segment_graph
            if _used_segment_graph:
                runtime_meta_recorder.reset()
                self._flush_pending_prefetch_metadata(block=False)
            else:
                _offload_capacity = max(self.verify_graph_bs) if (_use_kt_hybrid and self.verify_graph_bs) else int(input_ids.numel())
                host_buffer_slot, _ = self._acquire_prefetch_host_buffer_slot(
                    mode=_verify_meta_mode,
                    token_capacity=_offload_capacity,
                )
                if host_buffer_slot is not None:
                    enqueue_t0 = perf_counter()
                    handle = runtime_meta_recorder.offload_async(
                        prefetch_runtime.metadata_stream,
                        host_buffer_slot=host_buffer_slot,
                    )
                    enqueue_ms = (perf_counter() - enqueue_t0) * 1000.0
                    self._enqueue_prefetch_metadata(
                        mode=_verify_meta_mode,
                        step_id=step_id,
                        handle=handle,
                        enqueue_ms=enqueue_ms,
                        host_buffer_slot=host_buffer_slot,
                        submit_after_phase=None if self._dual_queue_prefetch_enabled() else "after_verify",
                        record_verify_consumed=True,
                    )
                runtime_meta_recorder.reset()
                self._flush_pending_prefetch_metadata(block=False)
            if self._prefetch_runtime_mode() == "draft_segment_indexed":
                with self._prefetch_runtime_lock:
                    prefetch_runtime.end_draft_iteration()
        _record_verify_call(outputs_ready=not return_logits)
        return verify_outputs

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
        self.draft_segment_graphs = {}
        self.draft_segment_boundaries = {}
        self.draft_segment_graph_vars = {}
        self.draft_tail_graphs = {}
        self.draft_graph_pool = None
        self.draft_graph_bs = []

        if not getattr(self.config, "draft_cuda_graph_enabled", True):
            return
        draft_cpu_graph = self._can_use_draft_cpu_cudagraph()
        if getattr(self.config, "draft_top_c", 0) != 0 and not draft_cpu_graph:
            # Default graph-safe subset: top_c>0 requires an explicit CPU graph bridge.
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

        if self._draft_segment_graph_enabled():
            self._capture_draft_segment_cudagraph(
                input_ids=input_ids,
                positions=positions,
                slot_mapping=slot_mapping,
                context_lens=context_lens,
                block_tables=block_tables,
                outputs=outputs,
            )
            return

        self._set_speculative_execution_mode("draft")
        if hasattr(self.model, "set_draft_cpu_graph_mode"):
            self.model.set_draft_cpu_graph_mode(draft_cpu_graph)
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
                if draft_cpu_graph:
                    torch.cuda.synchronize()
                    if hasattr(self.model, "check_draft_cpu_graph_errors"):
                        self.model.check_draft_cpu_graph_errors()
                with torch.cuda.graph(graph, self.draft_graph_pool):
                    outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
                if self.draft_graph_pool is None:
                    self.draft_graph_pool = graph.pool()
                self.draft_graphs[bs] = graph
                torch.cuda.synchronize()
                if hasattr(self.model, "check_draft_cpu_graph_errors"):
                    self.model.check_draft_cpu_graph_errors()
                if runtime_meta_recorder is not None:
                    runtime_meta_recorder.reset()
                reset_context()
        finally:
            if hasattr(self.model, "set_draft_cpu_graph_mode"):
                self.model.set_draft_cpu_graph_mode(False)
            self._set_speculative_execution_mode("normal")

        self.draft_graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

    def _capture_draft_segment_cudagraph(
        self,
        *,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        context_lens: torch.Tensor,
        block_tables: torch.Tensor,
        outputs: torch.Tensor,
    ) -> None:
        boundaries = self._draft_segment_boundaries()
        if not boundaries:
            return

        hidden_size = int(getattr(self.config.hf_config, "hidden_size"))
        segment_outputs = [
            torch.zeros(input_ids.size(0), hidden_size, dtype=outputs.dtype, device=outputs.device)
            for _ in boundaries
        ]
        runtime_meta_recorder = getattr(self, "runtime_meta_recorder", None)
        draft_cpu_graph = self._can_use_draft_cpu_cudagraph()
        from nanovllm.utils.verify_op_events import verify_op_capture_context

        self._set_speculative_execution_mode("draft")
        if hasattr(self.model, "set_draft_cpu_graph_mode"):
            self.model.set_draft_cpu_graph_mode(draft_cpu_graph)
        try:
            for bs in reversed(self.draft_graph_bs):
                graphs = []
                set_context(
                    False,
                    slot_mapping=slot_mapping[:bs],
                    context_lens=context_lens[:bs],
                    block_tables=block_tables[:bs],
                )
                if runtime_meta_recorder is not None:
                    runtime_meta_recorder.arm(
                        mode="draft",
                        step_id=-1,
                        token_capacity=bs,
                        logical_token_count=bs,
                    )
                for segment_idx, (layer_start, layer_end) in enumerate(boundaries):
                    graph = torch.cuda.CUDAGraph()
                    apply_norm = int(layer_end) >= int(getattr(self.config.hf_config, "num_hidden_layers"))
                    if segment_idx == 0:
                        segment_outputs[segment_idx][:bs] = self.model.forward_draft_segment(
                            input_ids[:bs],
                            None,
                            positions[:bs],
                            start_layer=int(layer_start),
                            end_layer=int(layer_end),
                            apply_norm=apply_norm,
                        )
                    else:
                        segment_outputs[segment_idx][:bs] = self.model.forward_draft_segment(
                            None,
                            segment_outputs[segment_idx - 1][:bs],
                            positions[:bs],
                            start_layer=int(layer_start),
                            end_layer=int(layer_end),
                            apply_norm=apply_norm,
                        )
                    if draft_cpu_graph:
                        torch.cuda.synchronize()
                        if hasattr(self.model, "check_draft_cpu_graph_errors"):
                            self.model.check_draft_cpu_graph_errors()
                    with verify_op_capture_context(bs, segment_idx, phase="draft"):
                        with torch.cuda.graph(graph, self.draft_graph_pool):
                            if segment_idx == 0:
                                segment_outputs[segment_idx][:bs] = self.model.forward_draft_segment(
                                    input_ids[:bs],
                                    None,
                                    positions[:bs],
                                    start_layer=int(layer_start),
                                    end_layer=int(layer_end),
                                    apply_norm=apply_norm,
                                )
                            else:
                                segment_outputs[segment_idx][:bs] = self.model.forward_draft_segment(
                                    None,
                                    segment_outputs[segment_idx - 1][:bs],
                                    positions[:bs],
                                    start_layer=int(layer_start),
                                    end_layer=int(layer_end),
                                    apply_norm=apply_norm,
                                )
                    if self.draft_graph_pool is None:
                        self.draft_graph_pool = graph.pool()
                    graphs.append(graph)
                self.draft_segment_graphs[bs] = graphs
                self.draft_segment_boundaries[bs] = list(boundaries)
                torch.cuda.synchronize()
                if hasattr(self.model, "check_draft_cpu_graph_errors"):
                    self.model.check_draft_cpu_graph_errors()
                # Acceptance-predictor tail graph: LM head stays eager, but the
                # route/token/history feature math + predictor MLP are captured.
                # The per-layer routing buffers were just filled by the (captured)
                # segment forward above, so they hold valid contents to record over.
                extractor = getattr(self, "_acceptance_extractor", None)
                if extractor is not None:
                    final_hidden = segment_outputs[-1][:bs]
                    logits = self.model.compute_logits(final_hidden)
                    extractor.set_token_features_from_logits(logits)
                    extractor.run_predictor(bs, final_hidden)  # eager warmup
                    torch.cuda.synchronize()
                    tail_graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(tail_graph, self.draft_graph_pool):
                        extractor.run_predictor(bs, final_hidden)
                    self.draft_tail_graphs[bs] = tail_graph
                    torch.cuda.synchronize()
                if runtime_meta_recorder is not None:
                    runtime_meta_recorder.reset()
                reset_context()
        finally:
            if hasattr(self.model, "set_draft_cpu_graph_mode"):
                self.model.set_draft_cpu_graph_mode(False)
            self._set_speculative_execution_mode("normal")

        self.draft_segment_graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=segment_outputs[-1],
            segment_outputs=segment_outputs,
        )

    # ------------------------------------------------------------------ #
    #  Verify CUDA Graph: prefix graph capture and replay                 #
    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def capture_verify_cudagraph(self):
        """Capture per-layer prefix CUDA graphs for verify (prefill-like) mode."""
        from nanovllm.models.qwen3_moe import Qwen3MoeHeterogeneousSparseMoeBlock

        config = self.config
        if not getattr(config, "verify_cuda_graph", False):
            self.verify_prefix_graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
            self.verify_dense_graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
            self.verify_graph_bs: list[int] = []
            self.verify_graph_vars: dict = {}
            self.verify_kt_hybrid_graphs: dict[int, torch.cuda.CUDAGraph] = {}
            return

        if getattr(config, "verify_cuda_graph_kt_hybrid", False):
            self._capture_verify_cudagraph_kt_hybrid()
            return

        hf_config = config.hf_config
        hidden_size = int(hf_config.hidden_size)
        num_experts_per_tok = int(getattr(hf_config, "num_experts_per_tok", 8))
        max_seqs = min(config.max_num_seqs, 64)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        bucket_steps = sorted(set(getattr(config, "verify_cuda_graph_bucket_steps", [4, 8, 12, 16])))
        self.verify_graph_bs = bucket_steps
        max_bucket = max(bucket_steps)

        input_ids = torch.zeros(max_bucket, dtype=torch.int64)
        positions = torch.zeros(max_bucket, dtype=torch.int64)
        cu_seqlens_q = torch.zeros(max_seqs + 1, dtype=torch.int32)
        cu_seqlens_k = torch.zeros(max_seqs + 1, dtype=torch.int32)
        slot_mapping = torch.full((max_bucket,), -1, dtype=torch.int32)
        block_tables = torch.zeros(max_seqs, max_num_blocks, dtype=torch.int32)
        hidden_states = torch.zeros(max_bucket, hidden_size, dtype=self.model.lm_head.weight.dtype,
                                    device=self.model.lm_head.weight.device)
        residual_buf = torch.zeros_like(hidden_states)
        selected_experts_buf = torch.zeros(max_bucket, num_experts_per_tok, dtype=torch.int64,
                                           device=hidden_states.device)
        routing_weights_buf = torch.zeros(max_bucket, num_experts_per_tok, dtype=hidden_states.dtype,
                                          device=hidden_states.device)

        self.verify_prefix_graphs = {}
        self.verify_dense_graphs = {}
        self.verify_kt_hybrid_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        verify_graph_pool = getattr(self, "draft_graph_pool", None)

        self._set_speculative_execution_mode("verify")
        try:
            for bs in reversed(bucket_steps):
                cu_seqlens_q[0] = 0
                cu_seqlens_q[1] = bs
                cu_seqlens_k[0] = 0
                cu_seqlens_k[1] = bs
                set_context(
                    True,
                    cu_seqlens_q=cu_seqlens_q[:2],
                    cu_seqlens_k=cu_seqlens_k[:2],
                    max_seqlen_q=bs,
                    max_seqlen_k=config.max_model_len,
                    slot_mapping=slot_mapping[:bs],
                    block_tables=block_tables[:1],
                )

                hidden_states[:bs] = self.model.model.embed_tokens(input_ids[:bs])

                for layer in self.model.model.layers:
                    layer_idx = layer.layer_idx
                    is_moe = isinstance(layer.mlp, Qwen3MoeHeterogeneousSparseMoeBlock)

                    if is_moe:
                        h, se, rw = layer.forward_verify_prefix(
                            hidden_states[:bs], positions[:bs], residual_buf[:bs],
                        )
                        selected_experts_buf[:bs].copy_(se)
                        routing_weights_buf[:bs].copy_(rw)
                        hidden_states[:bs].copy_(h)
                        torch.cuda.synchronize()

                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, pool=verify_graph_pool):
                            h, se, rw = layer.forward_verify_prefix(
                                hidden_states[:bs], positions[:bs], residual_buf[:bs],
                            )
                            selected_experts_buf[:bs].copy_(se)
                            routing_weights_buf[:bs].copy_(rw)
                            hidden_states[:bs].copy_(h)
                        if verify_graph_pool is None:
                            verify_graph_pool = graph.pool()
                        self.verify_prefix_graphs[(bs, layer_idx)] = graph

                        # The captured prefix is value-independent; the suffix is
                        # intentionally left eager so real verify can first
                        # perform cache-fill for any selected experts that are
                        # not resident in GPU slots. Running the GPU-only suffix
                        # here would use dummy routing and can see -1 slots.
                    else:
                        hidden_states[:bs] = layer(hidden_states[:bs], positions[:bs])
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, pool=verify_graph_pool):
                            hidden_states[:bs] = layer(hidden_states[:bs], positions[:bs])
                        if verify_graph_pool is None:
                            verify_graph_pool = graph.pool()
                        self.verify_dense_graphs[(bs, layer_idx)] = graph

                torch.cuda.synchronize()
                reset_context()
        finally:
            self._set_speculative_execution_mode("normal")

        self.verify_graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            hidden_states=hidden_states,
            residual_buf=residual_buf,
            selected_experts_buf=selected_experts_buf,
            routing_weights_buf=routing_weights_buf,
        )
        self._verify_graph_pool = verify_graph_pool

    def _capture_verify_cudagraph_kt_hybrid(self):
        """Capture full-model CUDA graphs for verify with hybrid GPU + kt_direct."""
        if self._verify_segment_graph_enabled():
            self._capture_verify_cudagraph_kt_hybrid_segments()
            return
        from nanovllm.layers.fuse_moe.kt_direct_backend import KtDirectCPUBuffer
        from nanovllm.utils.verify_op_events import verify_op_capture_context

        config = self.config
        hf_config = config.hf_config
        hidden_size = int(hf_config.hidden_size)
        max_seqs = min(config.max_num_seqs, 64)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        bucket_steps = sorted(set(getattr(config, "verify_cuda_graph_bucket_steps", [4, 8, 12, 16])))
        self.verify_graph_bs = bucket_steps
        max_bucket = max(bucket_steps)

        for bs in bucket_steps:
            KtDirectCPUBuffer.capture_bs.add(bs)

        input_ids = torch.zeros(max_bucket, dtype=torch.int64)
        positions = torch.zeros(max_bucket, dtype=torch.int64)
        cu_seqlens_q = torch.zeros(max_seqs + 1, dtype=torch.int32)
        cu_seqlens_k = torch.zeros(max_seqs + 1, dtype=torch.int32)
        slot_mapping = torch.full((max_bucket,), -1, dtype=torch.int32)
        block_tables = torch.zeros(max_seqs, max_num_blocks, dtype=torch.int32)
        hidden_states = torch.zeros(
            max_bucket, hidden_size,
            dtype=self.model.lm_head.weight.dtype,
            device=self.model.lm_head.weight.device,
        )

        self.verify_kt_hybrid_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.verify_prefix_graphs = {}
        self.verify_dense_graphs = {}
        verify_graph_pool = getattr(self, "draft_graph_pool", None)

        runtime_meta_recorder = getattr(self, "runtime_meta_recorder", None)
        record_verify_metadata = (
            runtime_meta_recorder is not None
            and not self._skip_verify_metadata_offload()
        )

        self._set_speculative_execution_mode("verify")
        try:
            for bs in reversed(bucket_steps):
                cu_seqlens_q[0] = 0
                cu_seqlens_q[1] = bs
                cu_seqlens_k[0] = 0
                cu_seqlens_k[1] = bs
                set_context(
                    True,
                    cu_seqlens_q=cu_seqlens_q[:2],
                    cu_seqlens_k=cu_seqlens_k[:2],
                    max_seqlen_q=bs,
                    max_seqlen_k=config.max_model_len,
                    slot_mapping=slot_mapping[:bs],
                    block_tables=block_tables[:1],
                )

                if record_verify_metadata:
                    runtime_meta_recorder.arm(
                        mode="verify_kt_hybrid",
                        step_id=0,
                        token_capacity=max_bucket,
                        logical_token_count=bs,
                    )

                hidden_states[:bs] = self.model.model.embed_tokens(input_ids[:bs])
                self.model.model.forward_verify_kt_hybrid_layers(
                    hidden_states[:bs], positions[:bs], apply_norm=True,
                )
                torch.cuda.synchronize()

                if record_verify_metadata:
                    runtime_meta_recorder.arm(
                        mode="verify_kt_hybrid",
                        step_id=0,
                        token_capacity=max_bucket,
                        logical_token_count=bs,
                    )

                graph = torch.cuda.CUDAGraph()
                with verify_op_capture_context(bs, -1):
                    with torch.cuda.graph(graph, pool=verify_graph_pool):
                        hidden_states[:bs] = self.model.model.embed_tokens(input_ids[:bs])
                        hidden_states[:bs] = self.model.model.forward_verify_kt_hybrid_layers(
                            hidden_states[:bs], positions[:bs], apply_norm=True,
                        )
                if verify_graph_pool is None:
                    verify_graph_pool = graph.pool()
                self.verify_kt_hybrid_graphs[bs] = graph

                torch.cuda.synchronize()
                reset_context()

            if record_verify_metadata:
                runtime_meta_recorder.reset()
        finally:
            self._set_speculative_execution_mode("normal")

        self.verify_graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            hidden_states=hidden_states,
        )
        self._verify_graph_pool = verify_graph_pool

    def _capture_verify_cudagraph_kt_hybrid_segments(self):
        """Capture per-segment CUDA graphs for verify with hybrid GPU + kt_direct."""
        from nanovllm.layers.fuse_moe.kt_direct_backend import KtDirectCPUBuffer
        from nanovllm.utils.verify_op_events import verify_op_capture_context

        boundaries = self._verify_segment_boundaries()
        config = self.config
        hf_config = config.hf_config
        hidden_size = int(hf_config.hidden_size)
        num_hidden_layers = int(hf_config.num_hidden_layers)
        max_seqs = min(config.max_num_seqs, 64)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        bucket_steps = sorted(set(getattr(config, "verify_cuda_graph_bucket_steps", [4, 8, 12, 16])))
        self.verify_graph_bs = bucket_steps
        max_bucket = max(bucket_steps)

        for bs in bucket_steps:
            KtDirectCPUBuffer.capture_bs.add(bs)

        input_ids = torch.zeros(max_bucket, dtype=torch.int64)
        positions = torch.zeros(max_bucket, dtype=torch.int64)
        cu_seqlens_q = torch.zeros(max_seqs + 1, dtype=torch.int32)
        cu_seqlens_k = torch.zeros(max_seqs + 1, dtype=torch.int32)
        slot_mapping = torch.full((max_bucket,), -1, dtype=torch.int32)
        block_tables = torch.zeros(max_seqs, max_num_blocks, dtype=torch.int32)

        dtype = self.model.lm_head.weight.dtype
        device = self.model.lm_head.weight.device
        segment_outputs = [
            torch.zeros(max_bucket, hidden_size, dtype=dtype, device=device)
            for _ in boundaries
        ]

        self.verify_kt_hybrid_segment_graphs: dict[int, list[torch.cuda.CUDAGraph]] = {}
        self.verify_kt_hybrid_segment_boundaries: dict[int, list[tuple[int, int]]] = {}
        self.verify_kt_hybrid_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.verify_prefix_graphs = {}
        self.verify_dense_graphs = {}
        verify_graph_pool = getattr(self, "draft_graph_pool", None)

        runtime_meta_recorder = getattr(self, "runtime_meta_recorder", None)
        record_verify_metadata = (
            runtime_meta_recorder is not None
            and not self._skip_verify_metadata_offload()
        )

        self._set_speculative_execution_mode("verify")
        try:
            for bs in reversed(bucket_steps):
                graphs: list[torch.cuda.CUDAGraph] = []
                cu_seqlens_q[0] = 0
                cu_seqlens_q[1] = bs
                cu_seqlens_k[0] = 0
                cu_seqlens_k[1] = bs
                set_context(
                    True,
                    cu_seqlens_q=cu_seqlens_q[:2],
                    cu_seqlens_k=cu_seqlens_k[:2],
                    max_seqlen_q=bs,
                    max_seqlen_k=config.max_model_len,
                    slot_mapping=slot_mapping[:bs],
                    block_tables=block_tables[:1],
                )

                if record_verify_metadata:
                    runtime_meta_recorder.arm(
                        mode="verify_kt_hybrid",
                        step_id=0,
                        token_capacity=max_bucket,
                        logical_token_count=bs,
                    )

                for seg_idx, (layer_start, layer_end) in enumerate(boundaries):
                    apply_norm = int(layer_end) >= num_hidden_layers
                    if seg_idx == 0:
                        segment_outputs[0][:bs] = self.model.forward_verify_kt_hybrid_segment(
                            input_ids[:bs], None, positions[:bs],
                            start_layer=int(layer_start), end_layer=int(layer_end),
                            apply_norm=apply_norm,
                        )
                    else:
                        segment_outputs[seg_idx][:bs] = self.model.forward_verify_kt_hybrid_segment(
                            None, segment_outputs[seg_idx - 1][:bs], positions[:bs],
                            start_layer=int(layer_start), end_layer=int(layer_end),
                            apply_norm=apply_norm,
                        )
                torch.cuda.synchronize()

                if record_verify_metadata:
                    runtime_meta_recorder.arm(
                        mode="verify_kt_hybrid",
                        step_id=0,
                        token_capacity=max_bucket,
                        logical_token_count=bs,
                    )

                for seg_idx, (layer_start, layer_end) in enumerate(boundaries):
                    graph = torch.cuda.CUDAGraph()
                    apply_norm = int(layer_end) >= num_hidden_layers
                    with verify_op_capture_context(bs, seg_idx):
                        with torch.cuda.graph(graph, pool=verify_graph_pool):
                            if seg_idx == 0:
                                segment_outputs[0][:bs] = self.model.forward_verify_kt_hybrid_segment(
                                    input_ids[:bs], None, positions[:bs],
                                    start_layer=int(layer_start), end_layer=int(layer_end),
                                    apply_norm=apply_norm,
                                )
                            else:
                                segment_outputs[seg_idx][:bs] = self.model.forward_verify_kt_hybrid_segment(
                                    None, segment_outputs[seg_idx - 1][:bs], positions[:bs],
                                    start_layer=int(layer_start), end_layer=int(layer_end),
                                    apply_norm=apply_norm,
                                )
                    if verify_graph_pool is None:
                        verify_graph_pool = graph.pool()
                    graphs.append(graph)

                self.verify_kt_hybrid_segment_graphs[bs] = graphs
                self.verify_kt_hybrid_segment_boundaries[bs] = list(boundaries)
                torch.cuda.synchronize()
                if record_verify_metadata:
                    runtime_meta_recorder.reset()
                reset_context()
        finally:
            self._set_speculative_execution_mode("normal")

        self.verify_graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            hidden_states=segment_outputs[-1],
            segment_outputs=segment_outputs,
        )
        self._verify_graph_pool = verify_graph_pool

    def _can_use_verify_cudagraph(self, num_tokens: int) -> bool:
        if not getattr(self.config, "verify_cuda_graph", False):
            return False
        kt_hybrid = getattr(self.config, "verify_cuda_graph_kt_hybrid", False)
        if kt_hybrid:
            if self._verify_segment_graph_enabled():
                if not getattr(self, "verify_kt_hybrid_segment_graphs", None):
                    return False
            elif not getattr(self, "verify_kt_hybrid_graphs", None):
                return False
        else:
            if not self.verify_prefix_graphs:
                return False
        context = get_context()
        if context.cu_seqlens_q is not None and context.cu_seqlens_q.numel() > 2:
            return False
        if context.block_tables is not None and self.verify_graph_vars:
            gv_bt = self.verify_graph_vars.get("block_tables")
            if gv_bt is not None and context.block_tables.shape[1] > gv_bt.shape[1]:
                return False
        return any(b >= num_tokens for b in self.verify_graph_bs)

    def _select_verify_bucket(self, num_tokens: int) -> int:
        for b in self.verify_graph_bs:
            if b >= num_tokens:
                return b
        return self.verify_graph_bs[-1]

    def _verify_graph_eager_gap(
        self,
        layer_idx: int,
        layer,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        runtime_meta_recorder,
    ) -> bool:
        """Run the non-captured verify cache-fill gap.

        Returns True when the suffix must fall back to the normal MoE path because
        some active experts could not be made resident in GPU slots.
        """
        mlp = layer.mlp
        profile = getattr(mlp, "_last_profile", None)
        if profile is None:
            profile = {}
            mlp._last_profile = profile

        # ── pre-transfer cache stats (mirrors eager MoE forward path) ──
        # When verify_cuda_graph is enabled, the normal MoE forward() is
        # bypassed by prefix graph replay + eager gap + GPU-only suffix.
        # Record pre-transfer miss/active counts here so that cache hit
        # rate and miss-per-layer metrics are comparable to the eager path.
        flat_sel = selected_experts.reshape(-1)
        total_active = float(flat_sel.numel())
        if total_active > 0:
            _, gpu_mask = mlp.expert_cache.remap_experts_to_slots(selected_experts)
            miss_count = int((~gpu_mask).sum().item())
            profile["pre_transfer_cache_miss_sum"] = float(miss_count)
            profile["pre_transfer_active_count_sum"] = total_active
            # Debug: check if prefetcher published experts are hits (NANOVLLM_DEBUG_PREFETCH_SUBMIT=1)
            if int(os.environ.get("NANOVLLM_DEBUG_PREFETCH_SUBMIT", "0")) > 0:
                pf_runtime = getattr(self, "prefetch_runtime", None)
                if pf_runtime is not None and hasattr(pf_runtime, "_recent_published"):
                    published_hit = 0
                    published_miss = 0
                    flat_mask = gpu_mask.reshape(-1)
                    for idx in range(flat_sel.numel()):
                        eid = int(flat_sel[idx].item())
                        key = (int(mlp.layer_idx), eid)
                        if key in pf_runtime._recent_published:
                            if bool(flat_mask[idx].item()):
                                published_hit += 1
                            else:
                                published_miss += 1
                    if published_hit + published_miss > 0:
                        self._profile["_dbg_published_hit"] = int(self._profile.get("_dbg_published_hit", 0)) + published_hit
                        self._profile["_dbg_published_miss"] = int(self._profile.get("_dbg_published_miss", 0)) + published_miss
        else:
            profile["pre_transfer_cache_miss_sum"] = 0.0
            profile["pre_transfer_active_count_sum"] = 0.0
        profile["moe_profile_count"] = 1.0
        # GPU-only suffix path has zero CPU routes; defaults are overwritten
        # when the eager fallback path calls MoE forward().
        for key in ("cpu_route_ratio_sum", "cpu_routes_sum",
                     "cpu_weight_mass_ratio_sum", "realized_cpu_expert_count_sum"):
            if key not in profile:
                profile[key] = 0.0

        if runtime_meta_recorder is not None:
            runtime_meta_recorder.record_layer(
                layer_idx=mlp.layer_idx,
                selected_experts=selected_experts,
                routing_weights=routing_weights,
            )

        if mlp.spec_verify_miss_policy == "cache_fill_no_cpu":
            active_ids, miss_ids = collect_cache_fill_no_cpu_expert_ids(
                selected_experts=selected_experts,
                expert_cache=mlp.expert_cache,
                profile=profile,
            )
            profile["activated_expert_set_size_sum"] = float(len(active_ids))
            fill_result = apply_verify_cache_fill_no_cpu_policy_ids(
                layer_idx=mlp.layer_idx,
                active_expert_ids=active_ids,
                miss_expert_ids=miss_ids,
                expert_cache=mlp.expert_cache,
                step_id=0,
                cache_strategy=mlp.cache_strategy,
                profile=profile,
            )
            return fill_result.cpu_expert_count > 0

        if mlp.spec_verify_miss_policy == "cache_fill":
            fill_result = apply_verify_cache_fill_policy(
                layer_idx=mlp.layer_idx,
                selected_experts=selected_experts,
                routing_weights=routing_weights,
                expert_cache=mlp.expert_cache,
                step_id=0,
                profile=profile,
            )
            return fill_result.cpu_expert_count > 0

        return True

    def _run_verify_with_prefix_graph(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Execute verify forward using prefix CUDA graphs + eager gap + eager suffix."""
        from nanovllm.models.qwen3_moe import Qwen3MoeHeterogeneousSparseMoeBlock

        num_tokens = input_ids.numel()
        bucket = self._select_verify_bucket(num_tokens)
        _vg_debug = os.environ.get("NANOVLLM_VG_DEBUG", "")
        if _vg_debug:
            print(f"[VG_DEBUG] step_id={getattr(self, '_current_verify_prefetch_step_id', -1)} "
                  f"num_tokens={num_tokens} bucket={bucket} input_ids={input_ids.tolist()}",
                  flush=True)
        gv = self.verify_graph_vars
        context = get_context()

        gv["input_ids"][:num_tokens].copy_(input_ids)
        gv["positions"][:num_tokens].copy_(positions)
        if context.cu_seqlens_q is not None:
            n_seqs = context.cu_seqlens_q.numel()
            gv["cu_seqlens_q"][:n_seqs].copy_(context.cu_seqlens_q)
        if context.cu_seqlens_k is not None:
            n_seqs = context.cu_seqlens_k.numel()
            gv["cu_seqlens_k"][:n_seqs].copy_(context.cu_seqlens_k)
        gv["slot_mapping"].fill_(-1)
        if context.slot_mapping is not None:
            gv["slot_mapping"][:context.slot_mapping.numel()].copy_(context.slot_mapping)
        if context.block_tables is not None:
            bt = context.block_tables
            gv["block_tables"][:bt.shape[0], :bt.shape[1]].copy_(bt)

        n_seqs_actual = (context.cu_seqlens_q.numel() - 1) if context.cu_seqlens_q is not None else 1
        set_context(
            True,
            cu_seqlens_q=gv["cu_seqlens_q"][:n_seqs_actual + 1],
            cu_seqlens_k=gv["cu_seqlens_k"][:n_seqs_actual + 1],
            max_seqlen_q=context.max_seqlen_q,
            max_seqlen_k=context.max_seqlen_k,
            slot_mapping=gv["slot_mapping"][:bucket],
            block_tables=gv["block_tables"][:n_seqs_actual],
        )

        gv["hidden_states"][:num_tokens] = self.model.model.embed_tokens(gv["input_ids"][:num_tokens])

        runtime_meta_recorder = getattr(self, "runtime_meta_recorder", None)

        def prefix_replay_fn(layer_idx, layer, hidden_states, position_ids, residual_buf,
                             selected_experts_buf, routing_weights_buf):
            key = (bucket, layer_idx)
            if key in self.verify_prefix_graphs:
                self.verify_prefix_graphs[key].replay()
                if self.profile_enabled and self.rank == 0:
                    self._profile["verify_prefix_graph_replay_count"] += 1
            else:
                h, se, rw = layer.forward_verify_prefix(
                    hidden_states[:num_tokens], position_ids[:num_tokens], residual_buf[:num_tokens],
                )
                selected_experts_buf[:num_tokens].copy_(se)
                routing_weights_buf[:num_tokens].copy_(rw)
                hidden_states[:num_tokens].copy_(h)
                if self.profile_enabled and self.rank == 0:
                    self._profile["verify_prefix_graph_fallback_count"] += 1

        def eager_gap_fn(layer_idx, layer, selected_experts_buf, routing_weights_buf):
            sel = selected_experts_buf[:num_tokens]
            rw = routing_weights_buf[:num_tokens]
            return self._verify_graph_eager_gap(
                layer_idx,
                layer,
                sel,
                rw,
                runtime_meta_recorder,
            )

        def suffix_fn(layer_idx, layer, hidden_states, selected_experts_buf,
                      routing_weights_buf, residual_buf, fallback):
            sel = selected_experts_buf[:num_tokens]
            rw = routing_weights_buf[:num_tokens]
            h = hidden_states[:num_tokens]
            res = residual_buf[:num_tokens]

            if not fallback:
                out = layer.forward_verify_suffix_gpu_only(h, sel, rw, res)
                hidden_states[:num_tokens] = out
            else:
                out = layer.mlp(h)
                hidden_states[:num_tokens] = res + out
            return hidden_states

        def dense_replay_fn(layer_idx, layer, hidden_states, position_ids):
            key = (bucket, layer_idx)
            if key in self.verify_dense_graphs:
                self.verify_dense_graphs[key].replay()
                if self.profile_enabled and self.rank == 0:
                    self._profile["verify_dense_graph_replay_count"] += 1
            else:
                hidden_states[:num_tokens] = layer(
                    hidden_states[:num_tokens], position_ids[:num_tokens],
                )
                if self.profile_enabled and self.rank == 0:
                    self._profile["verify_dense_graph_fallback_count"] += 1
            return hidden_states

        hidden = self.model.model.forward_verify_prefix_suffix(
            gv["hidden_states"],
            gv["positions"],
            gv["residual_buf"],
            gv["selected_experts_buf"],
            gv["routing_weights_buf"],
            prefix_replay_fn=prefix_replay_fn,
            eager_gap_fn=eager_gap_fn,
            suffix_fn=suffix_fn,
            dense_replay_fn=dense_replay_fn,
            apply_norm=True,
        )
        if self.profile_enabled and self.rank == 0:
            self._profile["verify_graph_call_count"] += 1
        return hidden[:num_tokens]

    def _run_verify_with_kt_hybrid_graph(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        *,
        step_id: int,
    ) -> torch.Tensor:
        """Execute verify forward using full-model CUDA graph with hybrid GPU + kt_direct."""
        num_tokens = input_ids.numel()
        bucket = self._select_verify_bucket(num_tokens)
        gv = self.verify_graph_vars
        context = get_context()

        gv["input_ids"][:num_tokens].copy_(input_ids)
        gv["positions"][:num_tokens].copy_(positions)
        if context.cu_seqlens_q is not None:
            n_seqs = context.cu_seqlens_q.numel()
            gv["cu_seqlens_q"][:n_seqs].copy_(context.cu_seqlens_q)
        if context.cu_seqlens_k is not None:
            n_seqs = context.cu_seqlens_k.numel()
            gv["cu_seqlens_k"][:n_seqs].copy_(context.cu_seqlens_k)
        gv["slot_mapping"].fill_(-1)
        if context.slot_mapping is not None:
            gv["slot_mapping"][:context.slot_mapping.numel()].copy_(context.slot_mapping)
        if context.block_tables is not None:
            bt = context.block_tables
            gv["block_tables"][:bt.shape[0], :bt.shape[1]].copy_(bt)

        n_seqs_actual = (context.cu_seqlens_q.numel() - 1) if context.cu_seqlens_q is not None else 1
        set_context(
            True,
            cu_seqlens_q=gv["cu_seqlens_q"][:n_seqs_actual + 1],
            cu_seqlens_k=gv["cu_seqlens_k"][:n_seqs_actual + 1],
            max_seqlen_q=context.max_seqlen_q,
            max_seqlen_k=context.max_seqlen_k,
            slot_mapping=gv["slot_mapping"][:bucket],
            block_tables=gv["block_tables"][:n_seqs_actual],
        )

        runtime_meta_recorder = getattr(self, "runtime_meta_recorder", None)
        record_verify_metadata = (
            runtime_meta_recorder is not None
            and not self._skip_verify_metadata_offload()
        )
        max_bucket = max(self.verify_graph_bs)
        if record_verify_metadata:
            runtime_meta_recorder.arm(
                mode="verify_kt_hybrid",
                step_id=int(step_id),
                token_capacity=max_bucket,
                logical_token_count=num_tokens,
                execution_token_count=bucket,
            )

        graph = self.verify_kt_hybrid_graphs[bucket]
        graph.replay()
        if os.getenv("NANOVLLM_VERIFY_OP_EVENT_TIMING", "").strip().lower() in {
            "1", "true", "yes", "y", "on"
        }:
            sync_t0 = perf_counter()
            ready = torch.cuda.Event(blocking=False)
            ready.record(torch.cuda.current_stream())
            ready.synchronize()
            sync_ms = (perf_counter() - sync_t0) * 1000.0
            if self.profile_enabled and self.rank == 0:
                with self._prefetch_profile_lock:
                    self._profile["verify_op_event_sync_count"] += 1.0
                    self._profile["verify_op_event_sync_ms"] += sync_ms
            self._collect_verify_op_event_timings(
                bucket=int(bucket),
                segment_id=-1,
                step_id=int(step_id),
                token_count=int(num_tokens),
            )

        skip_sync_profile_readback = bool(
            os.getenv("NANOVLLM_VERIFY_SKIP_SYNC_METADATA_READBACK", "").strip()
        )
        sync_profile_readback = os.getenv(
            "NANOVLLM_VERIFY_SYNC_METADATA_PROFILE_READBACK",
            "",
        ).strip().lower() in {"1", "true", "yes", "y", "on"}
        if not record_verify_metadata:
            if self.profile_enabled and self.rank == 0:
                with self._prefetch_profile_lock:
                    self._profile["verify_runtime_metadata_disabled_count"] += 1
        elif skip_sync_profile_readback:
            if self.profile_enabled and self.rank == 0:
                with self._prefetch_profile_lock:
                    self._profile["verify_metadata_profile_readback_skipped_count"] = (
                        float(self._profile.get("verify_metadata_profile_readback_skipped_count", 0.0)) + 1.0
                    )
        elif not sync_profile_readback:
            if self.profile_enabled and self.rank == 0:
                with self._prefetch_profile_lock:
                    self._profile["verify_metadata_profile_readback_async_default_count"] = (
                        float(self._profile.get("verify_metadata_profile_readback_async_default_count", 0.0)) + 1.0
                    )
        elif runtime_meta_recorder is not None:
            key = runtime_meta_recorder.active_key
            if key is not None:
                dev = runtime_meta_recorder.device_buffers.get(key)
                if dev is not None and "expert_status" in dev:
                    from nanovllm.models.qwen3_moe import Qwen3MoeHeterogeneousSparseMoeBlock
                    status_cpu = dev["expert_status"].cpu()
                    act_count_cpu = dev["activation_count"].cpu()
                    for decoder_layer in self.model.model.layers:
                        mlp = decoder_layer.mlp
                        if not isinstance(mlp, Qwen3MoeHeterogeneousSparseMoeBlock):
                            continue
                        lidx = mlp.layer_idx
                        s = status_cpu[lidx]
                        ac = act_count_cpu[lidx]
                        is_real_active = ac > 0
                        miss_routes = int(ac[s == 2].sum().item())
                        total_routes = int(ac.sum().item()) or int(num_tokens * mlp.num_selected)
                        miss_experts = int(((s == 2) & is_real_active).sum().item())
                        mlp._last_profile = {
                            "pre_transfer_cache_miss_sum": float(miss_routes),
                            "pre_transfer_active_count_sum": float(total_routes),
                            "moe_profile_count": 1.0,
                            "cpu_route_ratio_sum": float(miss_routes) / max(float(total_routes), 1.0),
                            "cpu_routes_sum": float(miss_routes),
                            "cpu_weight_mass_ratio_sum": 0.0,
                            "realized_cpu_expert_count_sum": float(miss_experts),
                            f"layer_{lidx}_realized_cpu_expert_count_sum": float(miss_experts),
                            f"layer_{lidx}_cpu_routes_sum": float(miss_routes),
                            f"layer_{lidx}_active_expert_count_sum": float(is_real_active.sum().item()),
                            f"layer_{lidx}_active_routes_sum": float(total_routes),
                            f"layer_{lidx}_moe_profile_count": 1.0,
                        }

        if self.profile_enabled and self.rank == 0:
            self._profile["verify_kt_hybrid_graph_replay_count"] = (
                float(self._profile.get("verify_kt_hybrid_graph_replay_count", 0.0)) + 1.0
            )
        return gv["hidden_states"][:num_tokens]

    def _enqueue_verify_segment_metadata(
        self,
        *,
        step_id: int,
        token_capacity: int,
        layer_start_idx: int,
        layer_end_idx: int,
        is_last_segment: bool,
    ) -> None:
        prefetch_runtime = getattr(self, "prefetch_runtime", None)
        runtime_meta_recorder = getattr(self, "runtime_meta_recorder", None)
        if prefetch_runtime is None or runtime_meta_recorder is None:
            return

        host_buffer_slot, _ = self._acquire_prefetch_host_buffer_slot(
            mode="verify_kt_hybrid",
            token_capacity=int(token_capacity),
        )
        if host_buffer_slot is None:
            return
        enqueue_t0 = perf_counter()
        handle = runtime_meta_recorder.offload_async(
            prefetch_runtime.metadata_stream,
            host_buffer_slot=host_buffer_slot,
            layer_start_idx=int(layer_start_idx),
            layer_end_idx=int(layer_end_idx),
        )
        enqueue_ms = (perf_counter() - enqueue_t0) * 1000.0
        if handle is None:
            return
        self._enqueue_prefetch_metadata(
            mode="verify_kt_hybrid",
            step_id=int(step_id),
            handle=handle,
            enqueue_ms=enqueue_ms,
            host_buffer_slot=host_buffer_slot,
            submit_after_phase=None,
            frontier_layer_idx=int(layer_end_idx) - 1,
            record_verify_consumed=bool(is_last_segment),
        )
        if self.profile_enabled and self.rank == 0:
            with self._prefetch_profile_lock:
                self._profile["verify_segment_metadata_enqueue_count"] = (
                    float(self._profile.get("verify_segment_metadata_enqueue_count", 0.0)) + 1.0
                )
                self._profile["verify_segment_metadata_enqueue_ms"] = (
                    float(self._profile.get("verify_segment_metadata_enqueue_ms", 0.0)) + enqueue_ms
                )
        if not getattr(self, "_prefetch_async_enabled", False):
            self._flush_pending_prefetch_metadata(block=False)

    def _run_verify_with_kt_hybrid_segment_graph(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        *,
        step_id: int,
    ) -> torch.Tensor:
        """Execute verify forward using per-segment CUDA graphs with inter-segment prefetching."""
        num_tokens = input_ids.numel()
        bucket = self._select_verify_bucket(num_tokens)
        graphs = self.verify_kt_hybrid_segment_graphs[bucket]
        boundaries = self.verify_kt_hybrid_segment_boundaries[bucket]
        gv = self.verify_graph_vars
        context = get_context()
        breakdown_sync = bool(os.getenv("NANOVLLM_VERIFY_BREAKDOWN_SYNC", "").strip())

        setup_t0 = perf_counter()
        gv["input_ids"][:num_tokens].copy_(input_ids)
        gv["positions"][:num_tokens].copy_(positions)
        if context.cu_seqlens_q is not None:
            n_seqs = context.cu_seqlens_q.numel()
            gv["cu_seqlens_q"][:n_seqs].copy_(context.cu_seqlens_q)
        if context.cu_seqlens_k is not None:
            n_seqs = context.cu_seqlens_k.numel()
            gv["cu_seqlens_k"][:n_seqs].copy_(context.cu_seqlens_k)
        gv["slot_mapping"].fill_(-1)
        if context.slot_mapping is not None:
            gv["slot_mapping"][:context.slot_mapping.numel()].copy_(context.slot_mapping)
        if context.block_tables is not None:
            bt = context.block_tables
            gv["block_tables"][:bt.shape[0], :bt.shape[1]].copy_(bt)

        n_seqs_actual = (context.cu_seqlens_q.numel() - 1) if context.cu_seqlens_q is not None else 1
        set_context(
            True,
            cu_seqlens_q=gv["cu_seqlens_q"][:n_seqs_actual + 1],
            cu_seqlens_k=gv["cu_seqlens_k"][:n_seqs_actual + 1],
            max_seqlen_q=context.max_seqlen_q,
            max_seqlen_k=context.max_seqlen_k,
            slot_mapping=gv["slot_mapping"][:bucket],
            block_tables=gv["block_tables"][:n_seqs_actual],
        )
        self._record_profile("verify_segment_graph_setup_enqueue_ms", perf_counter() - setup_t0)
        if breakdown_sync:
            setup_sync_t0 = perf_counter()
            torch.cuda.synchronize()
            self._record_profile("verify_segment_graph_setup_sync_ms", perf_counter() - setup_sync_t0)

        runtime_meta_recorder = getattr(self, "runtime_meta_recorder", None)
        prefetch_runtime = getattr(self, "prefetch_runtime", None)
        max_bucket = max(self.verify_graph_bs)
        step_id = int(step_id)
        deep_profile = bool(os.getenv("NANOVLLM_VERIFY_DEEP_PROFILE", "").strip())
        deep_profile_sync = bool(os.getenv("NANOVLLM_VERIFY_DEEP_PROFILE_SYNC", "").strip())
        op_event_profile = os.getenv("NANOVLLM_VERIFY_OP_EVENT_TIMING", "").strip().lower() in {
            "1", "true", "yes", "y", "on"
        }
        defer_segment_metadata = os.getenv(
            "NANOVLLM_VERIFY_DEFER_SEGMENT_METADATA",
            "1",
        ).strip().lower() not in {"0", "false", "no", "n", "off"}
        skip_sync_profile_readback = bool(
            os.getenv("NANOVLLM_VERIFY_SKIP_SYNC_METADATA_READBACK", "").strip()
        )
        sync_profile_readback = os.getenv(
            "NANOVLLM_VERIFY_SYNC_METADATA_PROFILE_READBACK",
            "",
        ).strip().lower() in {"1", "true", "yes", "y", "on"}
        record_verify_metadata = (
            runtime_meta_recorder is not None
            and not self._skip_verify_metadata_offload()
        )
        if record_verify_metadata:
            runtime_meta_recorder.arm(
                mode="verify_kt_hybrid",
                step_id=step_id,
                token_capacity=max_bucket,
                logical_token_count=num_tokens,
                execution_token_count=bucket,
            )

        num_segments = len(graphs)
        async_boundary_prefetch = (
            prefetch_runtime is not None
            and not self._dual_queue_prefetch_enabled()
            and self._verify_boundary_prefetch_async_enabled()
            and getattr(self, "_verify_boundary_worker_queue", None) is not None
        )
        for seg_idx, (graph, (layer_start, layer_end)) in enumerate(
            zip(graphs, boundaries, strict=True)
        ):
            prefetch_hook_t0 = perf_counter()
            if prefetch_runtime is not None:
                with self._prefetch_runtime_lock:
                    if self._dual_queue_prefetch_enabled():
                        prefetch_runtime.on_verify_segment_start(
                            step_id=step_id,
                            segment_id=seg_idx,
                            boundaries=boundaries,
                        )
                        next_seg = (seg_idx + 1) % num_segments
                        next_start, next_end = boundaries[next_seg]
                        prefetch_runtime.submit_verify_segment_prefetch(
                            step_id=step_id,
                            target_layer_start=int(next_start),
                            target_layer_end=int(next_end),
                            target_segment_id=next_seg,
                        )
                    else:
                        on_verify_layer_start = getattr(prefetch_runtime, "on_verify_layer_start", None)
                        if on_verify_layer_start is not None:
                            for layer_idx in range(int(layer_start), int(layer_end)):
                                on_verify_layer_start(layer_idx)
                        prefetch_runtime.publish_direct_active_ready(step_id=step_id)
                        if async_boundary_prefetch:
                            next_seg = (seg_idx + 1) % num_segments
                            next_start, next_end = boundaries[next_seg]
                            self._enqueue_verify_boundary_prefetch(
                                step_id=step_id,
                                target_layer_start=int(next_start),
                                target_layer_end=int(next_end),
                                target_segment_id=int(next_seg),
                            )
            prefetch_hook_ms = (perf_counter() - prefetch_hook_t0) * 1000.0
            if self.profile_enabled and self.rank == 0:
                with self._prefetch_profile_lock:
                    self._profile["verify_segment_prefetch_hook_ms"] += prefetch_hook_ms
                    self._profile[f"verify_segment_{seg_idx}_prefetch_hook_ms"] += prefetch_hook_ms

            timing_start = self._start_dual_queue_segment_timing()
            replay_enqueue_t0 = perf_counter()
            if deep_profile:
                from torch.profiler import record_function
                with record_function(f"nanovllm::verify.segment_{seg_idx}.graph_replay_enqueue"):
                    graph.replay()
            else:
                graph.replay()
            replay_enqueue_ms = (perf_counter() - replay_enqueue_t0) * 1000.0
            if self.profile_enabled and self.rank == 0:
                with self._prefetch_profile_lock:
                    self._profile["verify_segment_graph_replay_enqueue_count"] = (
                        float(self._profile.get("verify_segment_graph_replay_enqueue_count", 0.0)) + 1.0
                    )
                    self._profile["verify_segment_graph_replay_enqueue_ms"] = (
                        float(self._profile.get("verify_segment_graph_replay_enqueue_ms", 0.0)) + replay_enqueue_ms
                    )
                    self._profile[f"verify_segment_{seg_idx}_graph_replay_enqueue_ms"] += replay_enqueue_ms
            if not op_event_profile and (deep_profile_sync or breakdown_sync):
                sync_t0 = perf_counter()
                torch.cuda.synchronize()
                sync_ms = (perf_counter() - sync_t0) * 1000.0
                if self.profile_enabled and self.rank == 0:
                    with self._prefetch_profile_lock:
                        self._profile["verify_segment_graph_replay_sync_count"] = (
                            float(self._profile.get("verify_segment_graph_replay_sync_count", 0.0)) + 1.0
                        )
                        self._profile["verify_segment_graph_replay_sync_ms"] = (
                            float(self._profile.get("verify_segment_graph_replay_sync_ms", 0.0)) + sync_ms
                        )
                        self._profile[f"verify_segment_{seg_idx}_graph_replay_sync_ms"] += sync_ms
            self._end_dual_queue_segment_timing("verify", seg_idx, timing_start)

            is_last = seg_idx == num_segments - 1
            if record_verify_metadata and not defer_segment_metadata:
                self._enqueue_verify_segment_metadata(
                    step_id=step_id,
                    token_capacity=max_bucket,
                    layer_start_idx=int(layer_start),
                    layer_end_idx=int(layer_end),
                    is_last_segment=True,
                )
            boundary_submit_t0 = perf_counter()
            if (
                prefetch_runtime is not None
                and not self._dual_queue_prefetch_enabled()
                and not async_boundary_prefetch
            ):
                next_seg = (seg_idx + 1) % num_segments
                next_start = boundaries[next_seg][0]
                next_end = boundaries[next_seg][1]
                with self._prefetch_runtime_lock:
                    prefetch_runtime.submit_verify_segment_prefetch(
                        step_id=step_id,
                        target_layer_start=int(next_start),
                        target_layer_end=int(next_end),
                        visible_budget_ms=float(getattr(
                            self.config, "verify_prefetch_visible_budget_ms", 12.0)),
                    )
            boundary_submit_ms = (perf_counter() - boundary_submit_t0) * 1000.0
            if self.profile_enabled and self.rank == 0:
                with self._prefetch_profile_lock:
                    self._profile["verify_segment_boundary_submit_ms"] += boundary_submit_ms
                    self._profile[f"verify_segment_{seg_idx}_boundary_submit_ms"] += boundary_submit_ms
            self._poll_dual_queue_segment_timings(block=False)

        if op_event_profile:
            sync_t0 = perf_counter()
            ready = torch.cuda.Event(blocking=False)
            ready.record(torch.cuda.current_stream())
            ready.synchronize()
            sync_ms = (perf_counter() - sync_t0) * 1000.0
            if self.profile_enabled and self.rank == 0:
                with self._prefetch_profile_lock:
                    self._profile["verify_op_event_sync_count"] += 1.0
                    self._profile["verify_op_event_sync_ms"] += sync_ms
            for seg_idx in range(num_segments):
                self._collect_verify_op_event_timings(
                    bucket=int(bucket),
                    segment_id=int(seg_idx),
                    step_id=int(step_id),
                    token_count=int(num_tokens),
                )

        if async_boundary_prefetch:
            self._wait_for_verify_boundary_prefetch_drain()

        if record_verify_metadata and defer_segment_metadata:
            metadata_t0 = perf_counter()
            self._enqueue_verify_segment_metadata(
                step_id=step_id,
                token_capacity=max_bucket,
                layer_start_idx=0,
                layer_end_idx=int(boundaries[-1][1]) if boundaries else 0,
                is_last_segment=True,
            )
            self._record_profile("verify_deferred_segment_metadata_enqueue_total_ms", perf_counter() - metadata_t0)

        if self._dual_queue_prefetch_enabled() and prefetch_runtime is not None:
            with self._prefetch_runtime_lock:
                prefetch_runtime.complete_verify_round(step_id=step_id)

        if not record_verify_metadata:
            if self.profile_enabled and self.rank == 0:
                with self._prefetch_profile_lock:
                    self._profile["verify_runtime_metadata_disabled_count"] += 1
        elif skip_sync_profile_readback:
            if self.profile_enabled and self.rank == 0:
                with self._prefetch_profile_lock:
                    self._profile["verify_metadata_profile_readback_skipped_count"] = (
                        float(self._profile.get("verify_metadata_profile_readback_skipped_count", 0.0)) + 1.0
                    )
        elif not sync_profile_readback:
            if self.profile_enabled and self.rank == 0:
                with self._prefetch_profile_lock:
                    self._profile["verify_metadata_profile_readback_async_default_count"] = (
                        float(self._profile.get("verify_metadata_profile_readback_async_default_count", 0.0)) + 1.0
                    )
        elif runtime_meta_recorder is not None:
            key = runtime_meta_recorder.active_key
            if key is not None:
                dev = runtime_meta_recorder.device_buffers.get(key)
                if dev is not None and "expert_status" in dev:
                    from nanovllm.models.qwen3_moe import Qwen3MoeHeterogeneousSparseMoeBlock
                    if deep_profile:
                        from torch.profiler import record_function
                    status_t0 = perf_counter()
                    if deep_profile:
                        with record_function("nanovllm::verify.metadata_status_cpu"):
                            status_cpu = dev["expert_status"].cpu()
                    else:
                        status_cpu = dev["expert_status"].cpu()
                    self._record_profile("verify_metadata_status_cpu_ms", perf_counter() - status_t0)

                    act_t0 = perf_counter()
                    if deep_profile:
                        with record_function("nanovllm::verify.metadata_activation_cpu"):
                            act_count_cpu = dev["activation_count"].cpu()
                    else:
                        act_count_cpu = dev["activation_count"].cpu()
                    self._record_profile("verify_metadata_activation_cpu_ms", perf_counter() - act_t0)

                    loop_t0 = perf_counter()
                    if deep_profile:
                        profile_loop_ctx = record_function("nanovllm::verify.metadata_profile_loop")
                    else:
                        profile_loop_ctx = None
                    if profile_loop_ctx is not None:
                        profile_loop_ctx.__enter__()
                    for decoder_layer in self.model.model.layers:
                        mlp = decoder_layer.mlp
                        if not isinstance(mlp, Qwen3MoeHeterogeneousSparseMoeBlock):
                            continue
                        lidx = mlp.layer_idx
                        s = status_cpu[lidx]
                        ac = act_count_cpu[lidx]
                        is_real_active = ac > 0
                        miss_routes = int(ac[s == 2].sum().item())
                        total_routes = int(ac.sum().item()) or int(num_tokens * mlp.num_selected)
                        miss_experts = int(((s == 2) & is_real_active).sum().item())
                        mlp._last_profile = {
                            "pre_transfer_cache_miss_sum": float(miss_routes),
                            "pre_transfer_active_count_sum": float(total_routes),
                            "moe_profile_count": 1.0,
                            "cpu_route_ratio_sum": float(miss_routes) / max(float(total_routes), 1.0),
                            "cpu_routes_sum": float(miss_routes),
                            "cpu_weight_mass_ratio_sum": 0.0,
                            "realized_cpu_expert_count_sum": float(miss_experts),
                            f"layer_{lidx}_realized_cpu_expert_count_sum": float(miss_experts),
                            f"layer_{lidx}_cpu_routes_sum": float(miss_routes),
                            f"layer_{lidx}_active_expert_count_sum": float(is_real_active.sum().item()),
                            f"layer_{lidx}_active_routes_sum": float(total_routes),
                            f"layer_{lidx}_moe_profile_count": 1.0,
                        }
                    if profile_loop_ctx is not None:
                        profile_loop_ctx.__exit__(None, None, None)
                    self._record_profile("verify_metadata_profile_loop_ms", perf_counter() - loop_t0)
                    if self.profile_enabled and self.rank == 0:
                        with self._prefetch_profile_lock:
                            self._profile["verify_metadata_profile_readback_count"] = (
                                float(self._profile.get("verify_metadata_profile_readback_count", 0.0)) + 1.0
                            )

        if self.profile_enabled and self.rank == 0:
            self._profile["verify_kt_hybrid_segment_graph_replay_count"] = (
                float(self._profile.get("verify_kt_hybrid_segment_graph_replay_count", 0.0)) + 1.0
            )
        return gv["hidden_states"][:num_tokens]
