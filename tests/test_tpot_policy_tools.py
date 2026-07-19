import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.analyze_tpot_policy_results import (  # noqa: E402
    _degeneration_reasons,
    run as analyze,
)
from scripts.analyze_fixed_k_tpot import analyze as analyze_fixed_k  # noqa: E402
from scripts.collect_tpot_policy_results import _policy_specs  # noqa: E402


def _collector_args(tmp_path, policy_path):
    return Namespace(
        policy_specs=str(policy_path),
        artifact="",
        alpha_calibration_path="",
        max_draft_tokens=12,
        stop_rule="lookahead",
        stop_patience=1,
        lookahead_cache_credit_ms_per_step=0.0,
        min_steps=0,
        stop_margin=0.0,
        draft_cost_model="static",
        draft_td_ms=19.0,
        draft_tv_ms=80.0,
        acceptance_predictor_enabled=None,
    )


def test_named_policy_specs_accept_fixed_and_hysteresis(tmp_path):
    path = tmp_path / "policies.json"
    path.write_text(
        json.dumps(
            [
                {"name": "fixed_k15", "max_draft_tokens": 15,
                 "stop_policy": "none"},
                {"name": "active_p2", "max_draft_tokens": 15,
                 "stop_policy": "tpot", "stop_rule": "lookahead_hysteresis",
                 "stop_patience": 2, "min_steps": 12, "stop_margin": 0.1},
            ]
        ),
        encoding="utf-8",
    )

    specs = _policy_specs(_collector_args(tmp_path, path))

    assert [row["name"] for row in specs] == ["fixed_k15", "active_p2"]
    assert specs[0]["max_draft_tokens"] == 15
    assert specs[0]["acceptance_predictor_enabled"] is False
    assert specs[1]["acceptance_predictor_enabled"] is True
    assert specs[1]["stop_patience"] == 2
    assert specs[1]["stop_rule"] == "lookahead_hysteresis"


def test_named_policy_specs_accept_bucket_lookahead(tmp_path):
    path = tmp_path / "policies.json"
    artifact = tmp_path / "verify-model.json"
    artifact.write_text("{}", encoding="utf-8")
    path.write_text(
        json.dumps(
            [
                {
                    "name": "high_k",
                    "max_draft_tokens": 12,
                    "stop_policy": "tpot",
                    "stop_rule": "bucket_lookahead",
                    "min_steps": 6,
                    "verify_model_mode": "active",
                    "verify_model_path": str(artifact),
                }
            ]
        ),
        encoding="utf-8",
    )

    specs = _policy_specs(_collector_args(tmp_path, path))

    assert specs[0]["stop_rule"] == "bucket_lookahead"
    assert specs[0]["min_steps"] == 6
    assert specs[0]["acceptance_predictor_enabled"] is True


def _write_policy(root, name, throughput, digest):
    rows = []
    for index in range(3):
        rows.append(
            {
                "status": "ok",
                "dataset": "dataset",
                "sample_id": f"sample-{index}",
                "cache_ratio": 0.25,
                "repeat": 0,
                "max_output_tokens": 64,
                "generated_output_tokens": 64,
                "decode_sec": 63.0 / throughput,
                "output_fixed_length_ok": True,
                "max_repeated_token_run": 1,
                "generated_text": f"unique output {index}",
                "generated_token_ids": list(range(64)),
                "outputs_digest": f"{digest}-{index}",
                "decode_tok_s": 1.0,
                "tpot_ms": 1000.0,
            }
        )
    path = root / name
    path.mkdir()
    (path / "summary.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")


def test_generic_analyzer_allows_digest_mismatch_but_checks_throughput(tmp_path):
    _write_policy(tmp_path, "baseline", 20.0, "base")
    _write_policy(tmp_path, "candidate", 22.0, "candidate")

    report = analyze(
        Namespace(
            policies_root=str(tmp_path),
            baseline_policy="baseline",
            candidate_policies="candidate",
            active="",
            static="",
            none="",
            output="",
            minimum_improvement=0.0,
            minimum_pairs=3,
            minimum_clusters=3,
            bootstrap_iterations=100,
            seed=7,
            require_pass=False,
        )
    )

    comparison = report["comparisons"]["candidate"]
    assert comparison["passed"] is True
    assert comparison["output_digest_mismatch_count"] == 3
    assert report["best_candidate_policy"] == "candidate"
    assert report["policies"]["candidate"]["decode_tps_geomean"] == pytest.approx(22.0)
    assert report["tpot_definition"] == "decode_sec / (generated_output_tokens - 1)"
    assert report["quality"]["candidate"]["degeneration_failure_count"] == 0


def test_degeneration_check_detects_repeated_12gram():
    repeated = list(range(12)) * 3
    reasons = _degeneration_reasons(
        {
            "generated_output_tokens": len(repeated),
            "output_fixed_length_ok": True,
            "max_repeated_token_run": 1,
            "generated_token_ids": repeated,
        }
    )
    assert "repeated_12gram_ge_3" in reasons


def _write_fixed_k_summary(root, throughputs):
    rows = []
    for draft_k, throughput in throughputs.items():
        for ratio in (0.25, 0.3125):
            rows.append(
                {
                    "status": "ok",
                    "dataset": "dataset",
                    "sample_id": "sample-0",
                    "cache_ratio": ratio,
                    "repeat": 0,
                    "max_output_tokens": 128,
                    "max_draft_tokens": draft_k,
                    "decode_tok_s": throughput,
                    "tpot_ms": 1000.0 / throughput,
                }
            )
    root.mkdir()
    (root / "summary.json").write_text(
        json.dumps({"metadata": {}, "rows": rows}), encoding="utf-8"
    )


def test_fixed_k_analyzer_uses_smallest_k_within_tie(tmp_path):
    source = tmp_path / "source"
    _write_fixed_k_summary(source, {4: 20.92, 6: 21.0, 8: 20.7})

    report = analyze_fixed_k([source])

    assert report["raw_best_k"] == 6
    assert report["selected_k"] == 4
    assert report["pairing"]["complete"] is True


def test_fixed_k_analyzer_requests_boundary_extension(tmp_path):
    source = tmp_path / "source"
    _write_fixed_k_summary(
        source,
        {4: 19.0, 10: 21.0, 12: 20.8, 15: 20.95},
    )

    report = analyze_fixed_k([source])

    boundary = report["boundary_extension"]
    assert boundary["condition_met"] is True
    assert boundary["extension_required"] is True
