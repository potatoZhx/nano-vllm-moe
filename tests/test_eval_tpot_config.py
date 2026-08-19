from dataclasses import fields

from nanovllm.benchmarks.eval_tpot.cases import build_cases
from nanovllm.benchmarks.eval_tpot.config import configure_optimized_env, parse_args
from nanovllm.benchmarks.eval_tpot.runtime import build_llm_kwargs
from nanovllm.config import Config


def test_parse_args_resolves_preset_and_manual_overrides(tmp_path):
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--request-mode",
            "per_layer_slots",
            "--optimized-config",
            "k6_decode",
            "--cache-ratios",
            "0.075",
            "--verify-cuda-graph-bucket-steps",
            "3,4,5,6,7",
            "--cpu-expert-pin-memory",
            "false",
            "--kt-threadpool-count",
            "2",
            "--kt-numa-nodes",
            "0,1",
        ]
    )

    case = build_cases(args)[0]
    assert case["cache_ratio"] == 0.075
    assert case["max_draft_tokens"] == 6
    assert args.decode_driver == "generate"
    assert args.cpu_expert_pin_memory is False
    assert args._optimized_config_applied["manual_overrides"]["cache_ratios"] == "0.075"


def test_parse_args_accepts_ktransformers_sampling_filters(tmp_path):
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--top-k",
            "20",
            "--top-p",
            "0.95",
        ]
    )

    assert args.top_k == 20
    assert args.top_p == 0.95


def test_k3_3080_preset_encodes_measured_low_latency_topology(tmp_path):
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--optimized-config",
            "k3_3080",
        ]
    )

    assert args.cache_ratios == "0.075"
    assert args.max_draft_tokens_values == "3"
    assert args.verify_cuda_graph_bucket_steps == "4"
    assert args.kt_direct_backend == "llamafile_bf16"
    assert args.kt_num_threads == 16
    assert args.kt_threadpool_count == 2
    assert args.kt_numa_nodes == "0,1"
    assert args.cpu_expert_pin_memory is False
    assert args.gpu_memory_utilization == 0.996


def test_k1_f16_3080_preset_encodes_measured_tpot_optimum(tmp_path):
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--optimized-config",
            "k1_f16_3080",
        ]
    )

    assert args.cache_ratios == "0.09375"
    assert args.max_draft_tokens_values == "1"
    assert args.segment_sizes == "16"
    assert args.verify_prefetch_max_per_boundary == 2
    assert args.prefetch_staging_slots_per_layer == 0
    assert args.verify_cuda_graph_bucket_steps == "2"
    assert args.kt_direct_backend == "llamafile_f16"
    assert args.kt_capture_bs == "1,2,4,8,16,32"
    assert args.kt_num_threads == 16
    assert args.kt_threadpool_count == 2
    assert args.kt_numa_nodes == "0,1"
    assert args.draft_stop_policy == "none"
    assert args.acceptance_predictor_enabled is False
    assert args.cpu_expert_pin_memory is False
    assert args.gpu_memory_utilization == 0.996
    assert args.warmup_model_tokens == 1024

    env = configure_optimized_env(args)
    assert env["NANOVLLM_GROUPED_GEMM_FIXED_QWEN3"] == "1"


def test_manual_warmup_model_tokens_override_preset(tmp_path):
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--optimized-config",
            "k1_f16_3080",
            "--warmup-model-tokens",
            "2048",
        ]
    )

    assert args.warmup_model_tokens == 2048
    assert args._optimized_config_applied["manual_overrides"]["warmup_model_tokens"] == 2048


def test_k2_dynamic_f16_3080_preset_retains_best_dynamic_policy(tmp_path):
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--optimized-config",
            "k2_dynamic_f16_3080",
        ]
    )

    assert args.max_draft_tokens_values == "2"
    assert args.draft_stop_policy == "tpot"
    assert args.draft_tpot_td_ms == 97.0
    assert args.draft_tpot_tv_ms == 100.0
    assert args.draft_tpot_min_steps == 1
    assert args.draft_tpot_stop_rule == "first_increase"
    assert args.acceptance_predictor_enabled is True
    assert args.verify_cuda_graph_bucket_steps == "2,3"
    assert args.gpu_memory_utilization == 0.97
    assert args.warmup_model_tokens == 1024

    env = configure_optimized_env(args)
    assert env["NANOVLLM_GROUPED_GEMM_FIXED_QWEN3"] == "1"


def test_dynamic_preset_costs_can_be_overridden_explicitly(tmp_path):
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--optimized-config",
            "k2_dynamic_f16_3080",
            "--draft-tpot-td-ms",
            "95",
        ]
    )

    assert args.draft_tpot_td_ms == 95.0
    assert args._optimized_config_applied["manual_overrides"]["draft_tpot_td_ms"] == 95.0


def test_active14_dynamic_preset_retains_workload_sized_tpot_optimum(tmp_path):
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--optimized-config",
            "k2_dynamic_f16_3080_active14",
        ]
    )

    assert args.cache_ratios == "0.109375"
    assert args.max_draft_tokens_values == "2"
    assert args.draft_stop_policy == "tpot"
    assert args.draft_tpot_td_ms == 97.0
    assert args.draft_tpot_tv_ms == 100.0
    assert args.acceptance_predictor_enabled is True
    assert args.verify_cuda_graph_bucket_steps == "2,3"
    assert args.gpu_memory_utilization == 0.98
    assert args.warmup_model_tokens == 1024
    assert args.predictive_phase1_recent_verify is False

    env = configure_optimized_env(args)
    assert env["NANOVLLM_GROUPED_GEMM_FIXED_QWEN3"] == "1"


def test_active14_recent_phase1_is_independent_opt_in_preset(tmp_path):
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--optimized-config",
            "k2_dynamic_f16_3080_active14_phase1_recent",
        ]
    )

    assert args.cache_ratios == "0.109375"
    assert args.max_draft_tokens_values == "2"
    assert args.draft_tpot_td_ms == 97.0
    assert args.gpu_memory_utilization == 0.98
    assert args.predictive_phase1_recent_verify is True
    env = configure_optimized_env(args)
    assert env["NANOVLLM_GROUPED_GEMM_FIXED_QWEN3"] == "1"


def test_parse_args_rejects_verify_buckets_that_force_eager_fallback(tmp_path):
    try:
        parse_args(
            [
                "--output-dir",
                str(tmp_path),
                "--optimized-config",
                "k6_decode",
                "--verify-cuda-graph-bucket-steps",
                "3,4,5,6",
            ]
        )
    except ValueError as error:
        assert "requires a bucket >= 7" in str(error)
        assert "silently falls back to the eager path" in str(error)
    else:
        raise AssertionError("an uncovered fixed-K verify length must be rejected")


def test_parse_args_requires_total_verify_rows_for_true_batch(tmp_path):
    try:
        parse_args(
            [
                "--output-dir",
                str(tmp_path),
                "--optimized-config",
                "k6_decode",
                "--batch-size",
                "3",
                "--verify-cuda-graph-bucket-steps",
                "7,14,20",
            ]
        )
    except ValueError as error:
        assert "requires a bucket >= 21" in str(error)
    else:
        raise AssertionError("batch verify buckets must cover all packed rows")


def test_profile_mode_implications_are_resolved_in_one_parse_step(tmp_path):
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--latency-breakdown-profile",
            "true",
        ]
    )

    assert args.collect_profile is True
    assert args.engine_profile is True
    assert args.engine_profile_cuda_sync is False
    assert args.save_profile_json is True


def test_llm_kwargs_match_config_schema_and_preserve_topology(tmp_path):
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--request-mode",
            "per_layer_slots",
            "--optimized-config",
            "k6_decode",
            "--cache-ratios",
            "0.075",
            "--cpu-expert-pin-memory",
            "false",
            "--kt-num-threads",
            "16",
            "--kt-threadpool-count",
            "2",
            "--kt-numa-nodes",
            "0,1",
            "--batch-size",
            "3",
            "--verify-cuda-graph-bucket-steps",
            "7,14,21",
        ]
    )
    case = build_cases(args)[0]

    kwargs = build_llm_kwargs(args, case, 3, num_experts=128)

    config_fields = {field.name for field in fields(Config)}
    assert not set(kwargs) - config_fields
    assert kwargs["dist_port"] == args.dist_port_base + 3
    assert kwargs["heterogeneous_slots_per_layer"] == 10
    assert kwargs["cpu_expert_pin_memory"] is False
    assert kwargs["kt_num_threads"] == 16
    assert kwargs["kt_threadpool_count"] == 2
    assert kwargs["kt_numa_nodes"] == [0, 1]
    assert kwargs["max_num_seqs"] == 3


def test_llm_kwargs_keep_every_verify_bucket_buffer_persistent(tmp_path):
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--request-mode",
            "per_layer_slots",
            "--optimized-config",
            "k6_decode",
            "--verify-cuda-graph-bucket-steps",
            "3,4,5,6,7",
            "--kt-capture-bs",
            "1,2,4,8",
        ]
    )
    case = build_cases(args)[0]

    kwargs = build_llm_kwargs(args, case, 0, num_experts=128)

    assert kwargs["kt_capture_bs"] == [1, 2, 3, 4, 5, 6, 7, 8]
