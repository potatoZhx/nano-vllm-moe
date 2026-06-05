import os
from dataclasses import dataclass, field
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    inference_mode: str = "standard"
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    dist_port: int = 2333
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    enable_heterogeneous: bool = False
    enable_speculative: bool = False
    max_draft_tokens: int = 8
    draft_top_c: int = 0
    draft_reroute_policy: str = "entropy_cache_bias"
    draft_reroute_artifact: str = ""
    acceptance_strategy: str = "standard_sampling"
    acceptance_threshold: float = 0.7
    draft_scheduler: str = "simple"
    spec_verify_eager: bool = True
    spec_verify_miss_policy: str = "cache_fill"
    spec_enable_prefetch: bool = False
    cache_strategy: str = "lru"
    rank_guard_threshold: float = 0.15
    rank_guard_ema_alpha: float = 0.95
    prefetch_strategy: str = "history_window"
    prefetch_staging_slots_per_layer: int = 2
    prefetch_max_inflight: int = 8
    prefetch_transfer_stream_count: int = 1
    prefetch_step_budget: int = 4
    cache_eviction_budget_per_step: int = 2
    prefetch_verify_wait_ms: float = 0.0
    prefetch_global_queue_capacity: int = 4096
    prefetch_history_decay: float = 0.9
    prefetch_history_ttl_steps: int = 64
    prefetch_source_weight_prefill: float = 1.0
    prefetch_source_weight_verify: float = 1.2
    prefetch_source_weight_draft: float = 1.5
    prefetch_activation_count_weight: float = 0.1
    prefetch_age_penalty: float = 0.02
    prefetch_use_prefill_history: bool = True
    prefetch_use_verify_history: bool = True
    prefetch_use_draft_live: bool = True
    prefetch_verify_layer_enabled: bool = True
    prefetch_verify_layer_safety_ratio: float = 0.8
    prefetch_verify_layer_min_compute_ms: float = 0.05
    prefetch_verify_layer_transfer_bandwidth_gbps: float = 12.0
    prefetch_verify_layer_max_budget: int = 2
    prefetch_metadata_host_buffer_pool_size: int = 3
    prefetch_runtime_mode: str = "baseline_staging"
    prefetch_runtime_kind: str = "legacy"
    # Predictive prefetcher: fraction of layer compute used as the verify-prefetch
    # transfer budget, confining prefetch to roughly the attention window so the
    # MoE-stage demand-load keeps PCIe bandwidth (E.13, approach A).
    prefetch_verify_attention_ratio: float = 0.3
    # Phase-1 cold-start prefetch budget (number of experts) for segment n-1.
    predictive_phase1_budget: int = 4
    draft_prefetch_frontier_granularity: str = "segment"
    draft_prefetch_segment_size: int = 12
    draft_prefetch_segment_host_buffer_pool_size: int = 0
    draft_prefetch_visible_budget_ms: float = 3.0
    draft_prefetch_min_per_boundary: int = 0
    draft_prefetch_max_per_boundary: int = 4
    spec_profile: bool = False
    engine_profile: bool = False
    engine_profile_cuda_sync: bool = True
    heterogeneous_slots_per_layer: int = 0
    cpu_expert_pin_memory: bool = False  # pin_memory adds ~60s for 61GB model
    cpu_expert_execution_enabled: bool = False
    cpu_expert_backend: str = "torch"
    cpu_expert_workspace_max_routes: int = 8192
    cpu_expert_packed_min_routes: int = 32
    cpu_expert_strict_dtype: bool = True
    cpu_expert_num_threads: int = 4
    cpu_expert_parallel_mode: str = "serial"
    cpu_gpu_parallel_execution_enabled: str = "auto"  # "off" | "on" | "auto"
    cpu_gpu_parallel_min_cpu_route_ratio: float = 0.0  # only used when mode="on"
    kt_method: str = "BF16"                # kt-kernel backend method
    kt_num_threads: int = 0                # kt-kernel CPU threads (0=auto)
    kt_threadpool_count: int = 1           # kt-kernel NUMA sub-pools
    kt_chunked_prefill_size: int = 4096    # kt-kernel prefill chunk size
    gpu_plan_builder_enabled: bool = False
    gpu_plan_builder_fallback: bool = True
    draft_cuda_graph_enabled: bool = True
    draft_cuda_graph_max_bs: int = 512
    draft_cuda_graph_bucket_steps: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    draft_cuda_graph_cpu_backend: str = "none"  # "none" | "fused" | "fused_sync"; experimental top_c>0 graph bridge
    verify_cuda_graph: bool = False
    verify_cuda_graph_bucket_steps: list[int] = field(default_factory=lambda: [4, 8, 12, 16])
    perf_profile_level: str = "basic"

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert 1024 <= self.dist_port <= 65535

        valid_modes = {"standard", "heter", "spec", "auto"}
        assert self.inference_mode in valid_modes, f"invalid inference_mode: {self.inference_mode}"

        # Backward-compatible inference from legacy flags.
        if self.inference_mode == "auto" or (
            self.inference_mode == "standard" and (self.enable_heterogeneous or self.enable_speculative)
        ):
            if self.enable_speculative:
                self.inference_mode = "spec"
            elif self.enable_heterogeneous:
                self.inference_mode = "heter"
            else:
                self.inference_mode = "standard"

        if self.inference_mode == "standard":
            self.enable_speculative = False
        elif self.inference_mode == "heter":
            assert self.enable_heterogeneous, "heter mode requires enable_heterogeneous=True"
            self.enable_speculative = False
        else:  # spec
            assert self.enable_heterogeneous, "spec mode requires enable_heterogeneous=True"
            self.enable_speculative = True

        assert self.max_draft_tokens >= 1
        assert self.draft_top_c >= 0
        assert self.draft_reroute_policy in {
            "round_robin",
            "drop_miss",
            "entropy_cache_bias",
            "bounded_cache_bias",
            "similarity_replace",
        }
        if self.draft_reroute_policy != "round_robin":
            assert self.draft_top_c == 0, "draft reroute policies are implemented only for draft_top_c=0"
        if self.draft_reroute_policy == "similarity_replace":
            assert self.draft_reroute_artifact, "similarity_replace requires draft_reroute_artifact"
        assert self.cache_strategy in {"lru", "lfu", "adaptive", "lfu_rankguard", "lfu_rankguard_online"}
        assert 0.0 <= self.rank_guard_threshold <= 1.0
        assert 0.0 < self.rank_guard_ema_alpha <= 1.0
        assert self.prefetch_strategy in {"noop", "history_window"}
        assert self.prefetch_staging_slots_per_layer >= 0
        assert self.prefetch_max_inflight >= 0
        assert self.prefetch_transfer_stream_count >= 1
        assert self.prefetch_step_budget >= 0
        assert self.cache_eviction_budget_per_step >= 0
        assert self.prefetch_verify_wait_ms >= 0.0
        assert self.prefetch_global_queue_capacity >= 0
        assert 0.0 <= self.prefetch_history_decay <= 1.0
        assert self.prefetch_history_ttl_steps >= 1
        assert 0.0 < self.prefetch_verify_layer_safety_ratio <= 1.0
        assert self.prefetch_verify_layer_min_compute_ms >= 0.0
        assert self.prefetch_verify_layer_transfer_bandwidth_gbps > 0.0
        assert self.prefetch_verify_layer_max_budget >= 0
        assert self.prefetch_metadata_host_buffer_pool_size >= 1
        assert self.prefetch_runtime_kind in {"legacy", "predictive"}
        assert 0.0 < self.prefetch_verify_attention_ratio <= 1.0
        assert self.predictive_phase1_budget >= 0
        if self.prefetch_runtime_kind == "predictive":
            # The predictive prefetcher reuses the segment-indexed integration
            # gates in model_runner; force the mode so those gates fire.
            self.prefetch_runtime_mode = "draft_segment_indexed"
        assert self.prefetch_runtime_mode in {"baseline_staging", "draft_direct_active", "draft_segment_indexed"}
        assert self.draft_prefetch_frontier_granularity in {"iteration", "segment", "layer"}
        assert self.draft_prefetch_segment_size >= 1
        assert self.draft_prefetch_segment_host_buffer_pool_size >= 0
        assert self.draft_prefetch_visible_budget_ms >= 0.0
        assert self.draft_prefetch_min_per_boundary >= 0
        assert self.draft_prefetch_max_per_boundary >= self.draft_prefetch_min_per_boundary
        assert self.acceptance_strategy in {"greedy", "standard", "standard_sampling", "sampling", "spec_sampling"}
        assert self.spec_verify_miss_policy in {"cpu", "cache_fill", "cache_fill_no_cpu"}
        assert self.cpu_expert_backend in {"torch", "torch_packed", "fused", "kt_kernel"}
        assert self.cpu_expert_workspace_max_routes >= 1
        assert self.cpu_expert_packed_min_routes >= 1
        assert self.cpu_expert_num_threads >= 1
        assert self.cpu_expert_parallel_mode in {"serial", "expert_parallel", "auto"}
        assert self.cpu_gpu_parallel_execution_enabled in {"off", "on", "auto"}
        assert 0.0 <= self.cpu_gpu_parallel_min_cpu_route_ratio <= 1.0
        assert self.perf_profile_level in {"basic", "detailed"}
        assert self.draft_cuda_graph_max_bs >= 1
        assert len(self.draft_cuda_graph_bucket_steps) > 0
        assert all(x >= 1 for x in self.draft_cuda_graph_bucket_steps)
        assert self.draft_cuda_graph_cpu_backend in {"none", "fused", "fused_sync"}
        assert len(self.verify_cuda_graph_bucket_steps) > 0
        assert all(x >= 1 for x in self.verify_cuda_graph_bucket_steps)
        if self.verify_cuda_graph and self.enforce_eager:
            self.verify_cuda_graph = False

        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
