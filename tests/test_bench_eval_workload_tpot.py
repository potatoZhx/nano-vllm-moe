import sys
from types import SimpleNamespace

import pytest

from scripts.bench_eval_workload_tpot import (
    apply_optimized_config,
    build_cases,
    build_parser,
    collect_profile_metrics,
    create_llm,
    _effective_sample_offset,
    finalize_prompt_result,
    load_dataset_samples,
    resolve_acceptance_predictor,
    resolved_runtime_config,
    select_samples,
    steady_draft_call_stats,
    TPOT_DEFINITION,
    validate_kv_cache_capacity,
)
from nanovllm.benchmarks.eval_tpot.metrics import run_prompt_batch_generate


def test_tpot_excludes_prefill_first_token_from_decode_denominator():
    assert TPOT_DEFINITION == "decode_sec / (generated_output_tokens - 1)"
    result = finalize_prompt_result(
        list(range(8)),
        elapsed_sec=0.24,
        prefill_sec=0.10,
        decode_sec=0.14,
        prefill_steps=1,
        decode_steps=2,
        prefill_step_ms=[100.0],
        decode_step_ms=[70.0, 70.0],
        output_sequence_count=1,
        max_tokens=8,
        ignore_eos=True,
        eos_token_id=None,
        prompt_tokens=[42],
        max_model_len=64,
    )

    assert result["generated_output_tokens"] == 8
    assert result["decode_token_intervals"] == 7
    assert result["tpot_ms"] == 20.0
    assert result["decode_tok_s"] == pytest.approx(50.0)

    metrics = collect_profile_metrics({"spec_spec_step_ms": 140.0}, result)
    assert metrics["profile_decode_phase_output_tok_s"] == pytest.approx(50.0)


def test_per_layer_slots_ignores_dataset_holdout_offset():
    samples = load_dataset_samples("per_layer_slots", SimpleNamespace())

    selected = select_samples(
        samples,
        num_samples=2,
        sample_offset=_effective_sample_offset("per_layer_slots", 2),
        shuffle=False,
        seed=0,
    )

    assert len(selected) == 1
    assert selected[0].sample_id == "bench_per_layer_slots_prompt"
    assert _effective_sample_offset("sharegpt", 2) == 2


def test_fixed_policy_auto_disables_acceptance_predictor(tmp_path):
    args = build_parser().parse_args(
        ["--output-dir", str(tmp_path), "--draft-stop-policy", "none"]
    )

    resolution = resolve_acceptance_predictor(args)

    assert resolution == {
        "requested": "auto",
        "effective": False,
        "required_by": [],
    }
    assert args.acceptance_predictor_enabled is False


def test_tpot_policy_auto_enables_acceptance_predictor(tmp_path):
    args = build_parser().parse_args(["--output-dir", str(tmp_path)])

    resolution = resolve_acceptance_predictor(args)

    assert resolution["effective"] is True
    assert resolution["required_by"] == ["draft_stop_policy=tpot"]


def test_k6_decode_preset_uses_fast_path_controls(tmp_path):
    argv = ["--output-dir", str(tmp_path), "--optimized-config", "k6_decode"]
    args = build_parser().parse_args(argv)

    applied = apply_optimized_config(args, argv)
    resolution = resolve_acceptance_predictor(args)

    assert args.max_draft_tokens_values == "6"
    assert args.cache_ratios == "0.3125"
    assert args.segment_sizes == "12"
    assert args.verify_prefetch_max_per_boundary == 4
    assert args.draft_stop_policy == "none"
    assert args.kt_direct_backend == "avx2_bf16"
    assert args.decode_driver == "generate"
    assert args.reset_seed_after_warmup is True
    assert resolution["effective"] is False
    assert applied["applied"]["acceptance_predictor_enabled"] is False
    assert build_cases(args)[0]["acceptance_predictor_enabled"] is False


def test_cpu_expert_memory_and_dual_numa_options_reach_llm(tmp_path, monkeypatch):
    captured = {}

    class FakeLLM:
        def __init__(self, model_path, **kwargs):
            captured["model_path"] = model_path
            captured.update(kwargs)

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(_model_path):
            return SimpleNamespace(num_experts=128)

    monkeypatch.setitem(sys.modules, "nanovllm", SimpleNamespace(LLM=FakeLLM))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoConfig=FakeAutoConfig),
    )
    args = build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--cpu-expert-pin-memory",
            "false",
            "--kt-num-threads",
            "16",
            "--kt-threadpool-count",
            "2",
            "--kt-numa-nodes",
            "0,1",
        ]
    )

    create_llm(args, build_cases(args)[0], 0)

    assert captured["cpu_expert_pin_memory"] is False
    assert captured["kt_num_threads"] == 16
    assert captured["kt_threadpool_count"] == 2
    assert captured["kt_numa_nodes"] == [0, 1]
    assert resolved_runtime_config(args)["cpu_expert_pin_memory"] is False


def test_kv_cache_capacity_validation_reports_required_blocks():
    llm = SimpleNamespace(
        config=SimpleNamespace(
            kvcache_block_size=256,
            num_kvcache_blocks=3,
            gpu_memory_utilization=0.99,
            kvcache_block_bytes=24 * 1024**2,
            gpu_total_memory_bytes=10 * 1024**3,
        )
    )

    capacity = validate_kv_cache_capacity(
        llm,
        prompt_tokens=67,
        max_tokens=512,
    )

    assert capacity["required_blocks"] == 3
    assert capacity["available_blocks"] == 3


def test_kv_cache_capacity_multiplies_blocks_per_request_for_true_batch():
    llm = SimpleNamespace(
        config=SimpleNamespace(
            kvcache_block_size=256,
            num_kvcache_blocks=5,
            gpu_memory_utilization=0.999,
            kvcache_block_bytes=24 * 1024**2,
            gpu_total_memory_bytes=10 * 1024**3,
        )
    )

    capacity = validate_kv_cache_capacity(
        llm,
        prompt_tokens=67,
        max_tokens=64,
        batch_size=5,
    )

    assert capacity["blocks_per_request"] == 1
    assert capacity["required_blocks"] == 5


def test_true_batch_metrics_separate_request_tpot_and_aggregate_throughput():
    class FakeLLM:
        def __init__(self):
            self.step_index = 0

        def step(self):
            self.step_index += 1
            if self.step_index == 1:
                return [], 2
            return [(10, [1, 2, 3, 4]), (11, [5, 6, 7, 8])], -2

        def generate(self, prompts, sampling_params, use_tqdm=False):
            assert len(prompts) == 2
            assert len(sampling_params) == 2
            self.step()
            self.step()
            return [
                {"token_ids": [1, 2, 3, 4]},
                {"token_ids": [5, 6, 7, 8]},
            ]

    result = run_prompt_batch_generate(
        FakeLLM(),
        [42],
        batch_size=2,
        temperature=0.6,
        max_tokens=4,
        ignore_eos=True,
        eos_token_id=None,
        max_model_len=64,
    )

    assert result["output_sequence_count"] == 2
    assert result["output_fixed_length_ok"] is True
    assert result["decode_token_intervals"] == 3
    assert result["decode_token_intervals_total"] == 6
    assert result["aggregate_tpot_ms_per_output_token"] == pytest.approx(
        result["tpot_ms"] / 2
    )
    assert result["aggregate_decode_tok_s"] == pytest.approx(
        result["decode_tok_s"] * 2
    )
    assert result["stable_replay_decode_steps"] == 0
    assert result["stable_replay_step_wall_ms_mean"] == 0.0


def test_kv_cache_capacity_validation_recommends_memory_utilization():
    llm = SimpleNamespace(
        config=SimpleNamespace(
            kvcache_block_size=256,
            num_kvcache_blocks=1,
            gpu_memory_utilization=0.99,
            kvcache_block_bytes=24 * 1024**2,
            gpu_total_memory_bytes=10 * 1024**3,
        )
    )

    with pytest.raises(RuntimeError, match=r"required_blocks=3.*available_blocks=1") as exc:
        validate_kv_cache_capacity(
            llm,
            prompt_tokens=67,
            max_tokens=512,
        )

    assert "--gpu-memory-utilization 0.996" in str(exc.value)


def test_k12_bucket_stop_preset_uses_high_length_boundaries(tmp_path):
    argv = [
        "--output-dir",
        str(tmp_path),
        "--optimized-config",
        "k12_bucket_stop",
    ]
    args = build_parser().parse_args(argv)

    applied = apply_optimized_config(args, argv)
    predictor = resolve_acceptance_predictor(args)

    assert applied["name"] == "k12_bucket_stop"
    assert args.max_draft_tokens_values == "12"
    assert args.draft_stop_policy == "tpot"
    assert args.draft_tpot_stop_rule == "bucket_lookahead"
    assert args.draft_tpot_min_steps == 6
    assert args.draft_tpot_stop_margin == 0.10
    assert args.draft_tpot_lookahead_cache_credit_ms_per_step == 8.5
    assert args.draft_tpot_verify_model_mode == "active"
    assert args.verify_prefetch_max_per_boundary == 10
    assert predictor["effective"] is True


def test_explicitly_disabling_required_predictor_fails(tmp_path):
    args = build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--draft-stop-policy",
            "tpot",
            "--acceptance-predictor-enabled",
            "false",
        ]
    )

    try:
        resolve_acceptance_predictor(args)
    except ValueError as error:
        assert "draft_stop_policy=tpot" in str(error)
    else:
        raise AssertionError("required predictor disable should fail")


def test_k12_transfer_step_preset_uses_dense_one_step_policy(tmp_path):
    argv = [
        "--output-dir",
        str(tmp_path),
        "--optimized-config",
        "k12_transfer_step",
    ]
    args = build_parser().parse_args(argv)

    apply_optimized_config(args, argv)
    predictor = resolve_acceptance_predictor(args)

    assert args.max_draft_tokens_values == "12"
    assert args.draft_tpot_stop_rule == "transfer_aware_step"
    assert args.draft_tpot_min_steps == 6
    assert args.draft_tpot_cost_model == "history"
    assert args.draft_tpot_lookahead_cache_credit_ms_per_step == 0.0
    assert args.verify_cuda_graph_bucket_steps == "5,7,8,9,10,11,12,13"
    assert args.verify_prefetch_max_per_boundary == 4
    assert predictor["effective"] is True


def test_steady_draft_gate_drops_first_round_and_reports_distribution():
    stats = steady_draft_call_stats(
        {
            "spec_step_traces": [
                {"draft_call_ms": [50.0, 40.0]},
                {"draft_call_ms": [10.0, 12.0]},
                {"draft_call_ms": [14.0, 16.0]},
            ]
        }
    )

    assert stats["steady_draft_call_count"] == 4
    assert stats["steady_draft_call_mean_ms"] == 13.0
    assert stats["steady_draft_call_p50_ms"] == 13.0
    assert stats["steady_draft_call_p90_ms"] == pytest.approx(15.4)
    assert stats["steady_draft_gate_ms"] == 21.0
    assert stats["steady_draft_gate_passed"] is True
