from types import SimpleNamespace

import pytest

from scripts.bench_eval_workload_tpot import (
    apply_optimized_config,
    build_cases,
    build_parser,
    collect_profile_metrics,
    _effective_sample_offset,
    finalize_prompt_result,
    load_dataset_samples,
    resolve_acceptance_predictor,
    select_samples,
    TPOT_DEFINITION,
)


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
