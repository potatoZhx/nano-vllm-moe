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
    prefetch_draft_layer_enabled: bool = True
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
    # Dual-queue segment prefetcher. A single segment size drives both draft and
    # verify graphs so queue segment ids have identical layer ranges.
    dual_queue_segment_size: int = 12
    dual_queue_ground_truth_decay: float = 0.9
    dual_queue_ground_truth_ttl_rounds: int = 64
    dual_queue_ground_truth_count_weight: float = 0.1
    dual_queue_budget_safety_ratio: float = 0.8
    dual_queue_segment_time_ema_alpha: float = 0.2
    dual_queue_secondary_index_weight: float = 0.5
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
    heterogeneous_slot_allocation: str = "uniform"  # "uniform" | "profile_weighted"
    heterogeneous_slot_buckets: int = 4
    heterogeneous_slot_max_bucket_ratio: float = 2.0
    heterogeneous_slot_profile_csv: str = ""
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
    kt_direct_backend: str = "auto"         # auto | amx_bf16 | avx2_bf16 | llamafile_bf16 | llamafile_f16
    # Optional path to KTransformers' fixed legacy cpuinfer_ext.  The
    # llamafile backends can also import cpuinfer_ext normally when it is
    # installed in the active environment.
    kt_llamafile_extension_path: str = ""
    kt_single_weight: bool = True
    kt_numa_nodes: list[int] = field(default_factory=list)
    kt_capture_bs: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    gpu_plan_builder_enabled: bool = False
    gpu_plan_builder_fallback: bool = True
    draft_cuda_graph_enabled: bool = True
    draft_cuda_graph_max_bs: int = 512
    draft_cuda_graph_bucket_steps: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    draft_cuda_graph_cpu_backend: str = "none"  # "none" | "fused" | "fused_sync"; experimental top_c>0 graph bridge
    verify_cuda_graph: bool = False
    verify_cuda_graph_bucket_steps: list[int] = field(default_factory=lambda: [4, 8, 12, 16])
    verify_cuda_graph_kt_hybrid: bool = False
    verify_prefetch_segment_size: int = 12
    verify_prefetch_visible_budget_ms: float = 12.0
    verify_prefetch_min_per_boundary: int = 0
    verify_prefetch_max_per_boundary: int = 4
    perf_profile_level: str = "basic"
    # On-GPU theoretical-acceptance predictor for the draft path. When enabled, a
    # lightweight MLP (trained by random_cache_srdp_scripts-1) predicts per-step
    # acceptance alpha inside the draft segment-graph tail. Off by default.
    acceptance_predictor_enabled: bool = False
    acceptance_predictor_path: str = ""
    acceptance_predictor_step_horizon: int = 32
    draft_alpha_stop_threshold: float = 0.85
    # Dynamic draft-length stop policy driven by the acceptance predictor.
    #   "none"            -> always draft the full budget
    #   "alpha_threshold" -> legacy: stop when every seq's alpha < threshold
    #   "tpot"            -> stop when expected per-output-token latency stops
    #                        decreasing (cost-aware, cumulative acceptance)
    draft_stop_policy: str = "tpot"
    draft_tpot_td_ms: float = 19.0         # fixed per-draft-step time (v1)
    draft_tpot_tv_ms: float = 80.0         # fixed per-verify time (v1)
    draft_tpot_cost_model: str = "static"  # "static" | "history"
    draft_tpot_history_alpha: float = 0.2
    draft_tpot_min_steps: int = 0
    draft_tpot_stop_margin: float = 0.0
    draft_tpot_stop_patience: int = 1
    draft_tpot_lookahead_cache_credit_ms_per_step: float = 0.0
    draft_tpot_short_verify_penalty_ms: float = 0.0
    draft_tpot_verify_cost_floor_ms: float = 0.0
    # Uncertainty used by the one-step transfer-aware rule.  The policy stops
    # only when optimistic T(K+1) is still worse than conservative T(K).
    draft_tpot_alpha_error_p90: float = 0.05
    draft_tpot_draft_error_p90_ms: float = 1.0
    draft_tpot_uncertainty_scale: float = 1.0
    draft_tpot_stop_rule: str = "first_increase"  # includes "bucket_lookahead"
    draft_tpot_verify_model_mode: str = "off"  # "off" | "shadow" | "active"
    draft_tpot_verify_model_path: str = ""
    draft_tpot_alpha_calibration_path: str = ""
    # Profile-only truth collection for the v3 route/cache/transfer pipeline.
    # This does not enable an active policy or require a fitted v3 artifact.
    transfer_aware_profile: bool = False
    verify_prefetch_tpot_dynamic_budget_enabled: bool = False
    verify_prefetch_tpot_dynamic_budget_token_threshold: int = 10
    verify_prefetch_tpot_dynamic_budget_small: int = 4

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert 1024 <= self.dist_port <= 65535

        valid_modes = {"standard", "heter", "spec", "auto"}
        assert self.inference_mode in valid_modes, f"invalid inference_mode: {self.inference_mode}"

        valid_stop_policies = {"none", "alpha_threshold", "tpot"}
        assert self.draft_stop_policy in valid_stop_policies, (
            f"invalid draft_stop_policy: {self.draft_stop_policy}"
        )

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
        assert self.heterogeneous_slot_allocation in {"uniform", "profile_weighted"}, (
            f"invalid heterogeneous_slot_allocation: {self.heterogeneous_slot_allocation}"
        )
        assert 2 <= self.heterogeneous_slot_buckets <= 4
        assert self.heterogeneous_slot_max_bucket_ratio > 1.0
        if self.heterogeneous_slot_allocation == "profile_weighted":
            assert self.draft_reroute_artifact or self.heterogeneous_slot_profile_csv, (
                "profile_weighted slot allocation requires draft_reroute_artifact or "
                "heterogeneous_slot_profile_csv"
            )
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
        assert self.prefetch_runtime_kind in {"legacy", "predictive", "dual_queue"}
        assert 0.0 < self.prefetch_verify_attention_ratio <= 1.0
        assert self.predictive_phase1_budget >= 0
        assert self.dual_queue_segment_size >= 1
        assert 0.0 <= self.dual_queue_ground_truth_decay <= 1.0
        assert self.dual_queue_ground_truth_ttl_rounds >= 1
        assert self.dual_queue_ground_truth_count_weight >= 0.0
        assert 0.0 < self.dual_queue_budget_safety_ratio <= 1.0
        assert 0.0 < self.dual_queue_segment_time_ema_alpha <= 1.0
        assert 0.0 <= self.dual_queue_secondary_index_weight <= 1.0
        if self.prefetch_runtime_kind in {"predictive", "dual_queue"}:
            # The predictive prefetcher reuses the segment-indexed integration
            # gates in model_runner; force the mode so those gates fire.
            self.prefetch_runtime_mode = "draft_segment_indexed"
        if self.prefetch_runtime_kind == "dual_queue":
            assert not self.enforce_eager, "dual_queue requires CUDA graph execution"
            assert self.cpu_expert_backend == "kt_direct", "dual_queue verify requires cpu_expert_backend=kt_direct"
            assert self.spec_verify_miss_policy == "cpu", "dual_queue verify requires spec_verify_miss_policy=cpu"
            self.draft_prefetch_frontier_granularity = "segment"
            self.draft_prefetch_segment_size = int(self.dual_queue_segment_size)
            self.verify_prefetch_segment_size = int(self.dual_queue_segment_size)
            self.draft_cuda_graph_enabled = True
            self.verify_cuda_graph = True
            self.verify_cuda_graph_kt_hybrid = True
        assert self.prefetch_runtime_mode in {"baseline_staging", "draft_direct_active", "draft_segment_indexed"}
        assert self.draft_prefetch_frontier_granularity in {"iteration", "segment", "layer"}
        assert self.draft_prefetch_segment_size >= 1
        assert self.draft_prefetch_segment_host_buffer_pool_size >= 0
        assert self.draft_prefetch_visible_budget_ms >= 0.0
        assert self.draft_prefetch_min_per_boundary >= 0
        assert self.draft_prefetch_max_per_boundary >= self.draft_prefetch_min_per_boundary
        assert self.acceptance_strategy in {"greedy", "standard", "standard_sampling", "sampling", "spec_sampling"}
        assert self.spec_verify_miss_policy in {"cpu", "cache_fill", "cache_fill_no_cpu"}
        assert self.cpu_expert_backend in {"torch", "torch_packed", "fused", "kt_kernel", "kt_direct"}
        assert self.cpu_expert_workspace_max_routes >= 1
        assert self.cpu_expert_packed_min_routes >= 1
        assert self.cpu_expert_num_threads >= 1
        assert self.cpu_expert_parallel_mode in {"serial", "expert_parallel", "auto"}
        assert self.cpu_gpu_parallel_execution_enabled in {"off", "on", "auto"}
        assert 0.0 <= self.cpu_gpu_parallel_min_cpu_route_ratio <= 1.0
        assert self.kt_num_threads >= 0
        assert self.kt_threadpool_count >= 1
        assert self.kt_chunked_prefill_size >= 1
        assert self.kt_direct_backend in {
            "auto",
            "amx_bf16",
            "avx2_bf16",
            "llamafile_bf16",
            "llamafile_f16",
        }
        if self.kt_direct_backend.startswith("llamafile_") and self.kt_llamafile_extension_path:
            assert os.path.isfile(self.kt_llamafile_extension_path), (
                "kt_llamafile_extension_path must point to cpuinfer_ext when "
                "using a llamafile kt_direct_backend"
            )
        assert not self.kt_numa_nodes or len(self.kt_numa_nodes) == self.kt_threadpool_count
        # kt_direct embeds pinned host pointers in CUDA-graph host callbacks.
        # Every verify bucket therefore needs a persistent buffer; a temporary
        # buffer can be replaced while later buckets are captured and leave an
        # earlier graph with a stale pointer.  Keep the public knob, but make
        # the safe relationship automatic for every entry point.
        self.kt_capture_bs = sorted(
            {
                *(int(batch_size) for batch_size in self.kt_capture_bs),
                *(
                    int(batch_size)
                    for batch_size in self.verify_cuda_graph_bucket_steps
                ),
            }
        )
        assert self.kt_capture_bs
        assert all(batch_size >= 1 for batch_size in self.kt_capture_bs)
        assert self.perf_profile_level in {"basic", "detailed"}
        assert self.draft_cuda_graph_max_bs >= 1
        assert len(self.draft_cuda_graph_bucket_steps) > 0
        assert all(x >= 1 for x in self.draft_cuda_graph_bucket_steps)
        assert self.draft_cuda_graph_cpu_backend in {"none", "fused", "fused_sync"}
        assert len(self.verify_cuda_graph_bucket_steps) > 0
        assert all(x >= 1 for x in self.verify_cuda_graph_bucket_steps)
        assert self.verify_prefetch_segment_size >= 1
        assert self.verify_prefetch_visible_budget_ms >= 0.0
        assert self.verify_prefetch_max_per_boundary >= self.verify_prefetch_min_per_boundary
        if self.verify_cuda_graph and self.enforce_eager:
            self.verify_cuda_graph = False
        if (
            self.verify_cuda_graph
            and self.cpu_expert_backend == "kt_direct"
            and self.spec_verify_miss_policy == "cpu"
        ):
            self.verify_cuda_graph_kt_hybrid = True
            self.prefetch_verify_layer_enabled = False

        assert self.acceptance_predictor_step_horizon >= 1
        if self.acceptance_predictor_enabled:
            assert self.inference_mode == "spec", "acceptance predictor requires spec mode"
            assert self.acceptance_predictor_path, "acceptance_predictor_enabled requires acceptance_predictor_path"
        if self.transfer_aware_profile:
            assert self.acceptance_predictor_enabled, (
                "transfer-aware profile requires acceptance_predictor_enabled=True"
            )
        assert self.draft_tpot_cost_model in {"static", "history"}
        assert 0.0 < self.draft_tpot_history_alpha <= 1.0
        assert self.draft_tpot_min_steps >= 0
        assert self.draft_tpot_stop_margin >= 0.0
        assert self.draft_tpot_stop_patience >= 1
        assert self.draft_tpot_lookahead_cache_credit_ms_per_step >= 0.0
        assert self.draft_tpot_short_verify_penalty_ms >= 0.0
        assert self.draft_tpot_verify_cost_floor_ms >= 0.0
        assert 0.0 <= self.draft_tpot_alpha_error_p90 <= 1.0
        assert self.draft_tpot_draft_error_p90_ms >= 0.0
        assert self.draft_tpot_uncertainty_scale >= 0.0
        assert self.draft_tpot_stop_rule in {
            "first_increase",
            "best_margin",
            "lookahead",
            "lookahead_hysteresis",
            "bucket_lookahead",
            "transfer_aware_step",
        }
        assert self.draft_tpot_verify_model_mode in {"off", "shadow", "active"}
        if self.draft_tpot_alpha_calibration_path:
            assert self.acceptance_predictor_enabled, (
                "alpha calibration requires acceptance_predictor_enabled=True"
            )
            assert os.path.isfile(self.draft_tpot_alpha_calibration_path), (
                "draft_tpot_alpha_calibration_path must be a calibration artifact"
            )
        if self.draft_tpot_verify_model_mode != "off":
            assert self.draft_tpot_verify_model_path, (
                "draft_tpot_verify_model_path is required for shadow/active mode"
            )
            assert self.acceptance_predictor_enabled, (
                "verify cost route proxy requires acceptance_predictor_enabled=True"
            )
        if self.draft_tpot_verify_model_mode == "active":
            assert self.draft_stop_policy == "tpot", (
                "active verify cost model requires draft_stop_policy='tpot'"
            )
        if self.draft_tpot_stop_rule == "bucket_lookahead":
            assert self.draft_tpot_verify_model_mode == "active", (
                "bucket_lookahead requires draft_tpot_verify_model_mode='active'"
            )
        if self.draft_tpot_stop_rule == "transfer_aware_step":
            dense_buckets = {5, 7, 8, 9, 10, 11, 12, 13}
            assert self.draft_tpot_verify_model_mode in {"shadow", "active"}, (
                "transfer_aware_step requires draft_tpot_verify_model_mode="
                "'shadow' or 'active'"
            )
            assert self.draft_stop_policy == "tpot", (
                "transfer_aware_step requires draft_stop_policy='tpot'"
            )
            assert self.max_draft_tokens == 12, (
                "transfer_aware_step is validated only with max_draft_tokens=12"
            )
            assert self.draft_tpot_min_steps == 6, (
                "transfer_aware_step is validated only with draft_tpot_min_steps=6"
            )
            assert self.verify_cuda_graph, (
                "transfer_aware_step requires verify CUDA graphs"
            )
            assert dense_buckets.issubset(
                {int(value) for value in self.verify_cuda_graph_bucket_steps}
            ), (
                "transfer_aware_step requires dense verify buckets "
                "5,7,8,9,10,11,12,13"
            )
            assert self.draft_tpot_lookahead_cache_credit_ms_per_step == 0.0, (
                "transfer_aware_step does not use legacy lookahead cache credit"
            )
        assert self.verify_prefetch_tpot_dynamic_budget_token_threshold >= 1
        assert self.verify_prefetch_tpot_dynamic_budget_small >= 0

        self.hf_config = AutoConfig.from_pretrained(self.model)
        if self.prefetch_runtime_kind == "dual_queue":
            assert self.prefetch_staging_slots_per_layer >= 1, (
                "dual_queue requires prefetch_staging_slots_per_layer >= 1"
            )
            assert int(self.hf_config.num_hidden_layers) > int(self.dual_queue_segment_size), (
                "dual_queue requires at least two segments"
            )
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
