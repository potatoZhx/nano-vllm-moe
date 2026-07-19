import json
from argparse import Namespace

import pytest

from nanovllm.engine.speculative.verify_cost_model import (
    VerifyCpuWorkload,
    VerifyTimeCostModel,
)
from scripts.analyze_verify_time_cost_model import run


def _write_profile(path, source_index: int) -> None:
    records = []
    traces = []
    buckets = (3, 5, 7)
    for call_index in range(12):
        bucket = buckets[call_index % len(buckets)]
        logical_tokens = bucket - (call_index % 2)
        first = (call_index + source_index) % 4
        second = (call_index * 2 + source_index) % 4
        layer_counts = {
            "0": [first, second, 0, 0],
            "1": [0, call_index % 3, source_index % 3, 1],
        }
        cpu_routes = sum(sum(row) for row in layer_counts.values())
        cpu_experts = sum(sum(value > 0 for value in row) for row in layer_counts.values())
        target_ms = 10.0 + bucket + 0.3 * cpu_routes + 0.1 * cpu_experts
        records.append(
            {
                "call_index": call_index,
                "step_id": 1000 * source_index + call_index,
                "bucket": bucket,
                "token_count": logical_tokens,
                "metadata_layer_execution_cpu_route_counts": layer_counts,
                "metadata_layer_execution_route_counts": layer_counts,
            }
        )
        traces.append(
            {
                "verify_call_index": call_index,
                "verify_accept_ready_ms": target_ms,
            }
        )
    path.write_text(
        json.dumps(
            {
                "model_verify_call_records": records,
                "spec_step_traces": traces,
            }
        ),
        encoding="utf-8",
    )


def test_analysis_builds_loadable_grouped_holdout_artifact(tmp_path):
    profiles = []
    for source_index in range(6):
        profile = tmp_path / f"sample{source_index:04d}.json"
        _write_profile(profile, source_index)
        profiles.append(str(profile))

    output = tmp_path / "verify_cost_model.json"
    artifact = run(
        Namespace(
            profiles=profiles,
            output=str(output),
            num_layers=2,
            num_experts=4,
            top_k=2,
            drop_first_calls=0,
            holdout_fraction=0.2,
            seed=7,
            max_mae_ms=0.05,
            max_p90_ms=0.1,
            min_ranking_accuracy=0.9,
            kt_num_threads=16,
            kt_backend="test",
            require_measurement_metadata=False,
        )
    )

    assert artifact["accuracy_gate_passed"] is True
    assert artifact["model_kind"] in {"global_counts", "layer_route_shape"}
    assert artifact["split"]["holdout_groups"]
    assert artifact["design_diagnostics"]["full_column_rank"] is True
    assert "padding_tokens" not in artifact["feature_names"]
    assert "bucket_3" not in artifact["feature_names"]
    assert output.with_suffix(".md").is_file()
    assert output.with_suffix(".holdout.csv").is_file()

    model = VerifyTimeCostModel.load(output)
    workload = VerifyCpuWorkload.from_mapping(
        bucket=5,
        logical_tokens=4,
        layer_route_counts={"0": [2, 1, 0, 0], "1": [0, 1, 1, 1]},
        num_layers=2,
        num_experts=4,
    )
    prediction = model.predict(workload)
    expected = 10.0 + 5.0 + 0.3 * 6.0 + 0.1 * 5.0
    assert prediction.total_ms == pytest.approx(expected, abs=0.05)
    assert prediction.exposed_cpu_ms >= 0.0
