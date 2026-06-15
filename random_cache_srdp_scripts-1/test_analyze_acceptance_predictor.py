import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPT_DIR = Path(__file__).resolve().parent


def _load_module():
    name = "analyze_acceptance_predictor_test"
    spec = importlib.util.spec_from_file_location(
        name,
        SCRIPT_DIR / "analyze_acceptance_predictor.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_distribution_summary_has_requested_percentiles():
    mod = _load_module()

    summary = mod.distribution_summary(np.arange(1, 101, dtype=np.float64))

    assert summary["count"] == 100
    assert summary["mean"] == 50.5
    assert summary["min"] == 1.0
    assert summary["max"] == 100.0
    assert summary["p1"] == 1.99
    assert summary["p50"] == 50.5
    assert summary["p99"] == 99.01


def test_regression_metrics_and_true_alpha_buckets():
    mod = _load_module()
    y_true = np.array([0.2, 0.6, 0.8, 0.9, 0.99])
    y_pred = np.array([0.3, 0.5, 0.7, 0.95, 0.97])

    metrics = mod.regression_metrics(y_true, y_pred)
    rows = mod.alpha_bucket_rows(y_true, y_pred, split="test")

    assert metrics["count"] == 5
    assert np.isclose(metrics["bias"], -0.014)
    assert [row["bucket"] for row in rows] == [
        "alpha < 0.5",
        "0.5 <= alpha < 0.7",
        "0.7 <= alpha < 0.85",
        "0.85 <= alpha < 0.95",
        "alpha >= 0.95",
    ]
    assert [row["count"] for row in rows] == [1, 1, 1, 1, 1]


def test_group_metrics_preserve_decode_step_values():
    mod = _load_module()
    y_true = np.array([0.2, 0.4, 0.6, 0.8])
    y_pred = np.array([0.1, 0.5, 0.5, 0.9])
    steps = np.array([1, 2, 1, 2])

    rows = mod.group_metric_rows(
        y_true,
        y_pred,
        steps,
        split="test",
        group_name="decode_step",
    )

    assert [row["decode_step"] for row in rows] == [1, 2]
    assert [row["count"] for row in rows] == [2, 2]
    assert all(np.isclose(row["mae"], 0.1) for row in rows)


def test_chain_metrics_compute_expected_accepted_length():
    mod = _load_module()
    y_true = np.array([0.5, 0.5, 0.8, 0.25])
    y_pred = np.array([0.4, 0.5, 0.8, 0.5])
    prompt_ids = np.array(["a", "a", "b", "b"])
    steps = np.array([1, 2, 1, 2])

    rows, summary = mod.prompt_chain_rows(
        y_true,
        y_pred,
        prompt_ids,
        steps,
        split="test",
    )

    assert len(rows) == 2
    first = rows[0]
    assert first["prompt_id"] == "a"
    assert np.isclose(first["true_expected_accepted"], 0.75)
    assert np.isclose(first["pred_expected_accepted"], 0.6)
    assert summary["count"] == 2


def test_calibration_and_threshold_rows_cover_decision_quality():
    mod = _load_module()
    y_true = np.array([0.2, 0.6, 0.8, 0.95])
    y_pred = np.array([0.1, 0.7, 0.75, 0.9])

    calibration = mod.calibration_rows(y_true, y_pred, split="test", bins=2)
    thresholds = mod.threshold_rows(
        y_true,
        y_pred,
        split="test",
        thresholds=(0.5, 0.85),
    )

    assert sum(row["count"] for row in calibration) == 4
    assert [row["threshold"] for row in thresholds] == [0.5, 0.85]
    assert thresholds[0]["accuracy"] == 1.0


def test_load_jsonl_metadata_and_validate_tensor_alignment(tmp_path):
    mod = _load_module()
    path = tmp_path / "samples.jsonl"
    records = [
        {
            "metadata": {
                "req_id": "r0",
                "article_id": 7,
                "window_start": 10,
                "prefill_len": 20,
                "cache_ratio": 0.25,
            },
            "steps": [
                {"step": 1, "alpha_theoretical": 0.2},
                {"step": 2, "alpha_theoretical": 0.4},
            ],
        },
        {
            "metadata": {
                "req_id": "r1",
                "article_id": 8,
                "window_start": 30,
                "prefill_len": 12,
                "cache_ratio": 0.5,
            },
            "steps": [{"step": 1, "alpha_theoretical": 0.9}],
        },
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    metadata = mod.load_jsonl_metadata(path)

    assert metadata["prompt_id"].tolist() == ["r0", "r0", "r1"]
    assert metadata["decode_step"].tolist() == [1, 2, 1]
    assert metadata["article_id"].tolist() == [7, 7, 8]
    assert mod.validate_alignment(
        np.array([0.2, 0.4, 0.9]),
        metadata["alpha"],
        split="train",
    ) == 0.0
    with pytest.raises(ValueError, match="label mismatch"):
        mod.validate_alignment(
            np.array([0.2, 0.5, 0.9]),
            metadata["alpha"],
            split="train",
        )


def test_split_leakage_reports_prompt_article_and_window_overlap():
    mod = _load_module()
    metadata = {
        "prompt_id": np.array(["r0", "r0", "r1", "r1", "r2", "r2"]),
        "article_id": np.array([7, 7, 7, 7, 8, 8], dtype=object),
        "window_start": np.array([0, 0, 10, 10, 100, 100]),
        "prefill_len": np.array([20, 20, 20, 20, 10, 10]),
    }
    train_indices = np.array([0, 2, 4])
    val_indices = np.array([1, 3, 5])

    report = mod.split_leakage_report(train_indices, val_indices, metadata)
    windows = mod.window_overlap_report(metadata)

    assert report["prompt"]["overlap_count"] == 3
    assert report["prompt"]["val_overlap_rate"] == 1.0
    assert report["article"]["overlap_count"] == 2
    assert windows["candidate_pairs"] == 1
    assert windows["overlap_pairs"] == 1
    assert windows["overlap_rate"] == 1.0


def test_distribution_drift_detects_shift_and_feature_rows():
    mod = _load_module()
    train = np.array([0.0, 0.0, 1.0, 1.0])
    test = np.array([1.0, 1.0, 2.0, 2.0])

    drift = mod.distribution_drift(train, test)
    branch_rows, dimension_rows = mod.feature_distribution_rows(
        {
            "route": torch.tensor([[0.0, 1.0], [1.0, 2.0]]),
            "hidden": torch.tensor([[2.0], [4.0]]),
        },
        {
            "route": torch.tensor([[1.0, 2.0], [2.0, 3.0]]),
            "hidden": torch.tensor([[3.0], [5.0]]),
        },
    )

    assert drift["mean_shift"] == 1.0
    assert drift["ks_statistic"] == 0.5
    assert drift["standardized_mean_difference"] > 1.0
    assert [row["branch"] for row in branch_rows] == ["hidden", "route"]
    assert len(dimension_rows) == 3
    assert dimension_rows[0]["dimension"] == 0


def test_metadata_dataset_summary_counts_prompts_articles_and_ratios():
    mod = _load_module()
    metadata = {
        "prompt_id": np.array(["r0", "r0", "r1"]),
        "decode_step": np.array([1, 2, 1]),
        "alpha": np.array([0.2, 0.4, 0.9]),
        "cache_ratio": np.array([0.25, 0.25, 0.5]),
        "prefill_len": np.array([20, 20, 12]),
        "article_id": np.array([7, 7, 8], dtype=object),
        "window_start": np.array([10, 10, 30]),
    }

    summary = mod.metadata_dataset_summary(metadata)

    assert summary["steps"] == 3
    assert summary["prompts"] == 2
    assert summary["articles"] == 2
    assert summary["decode_step_counts"] == {"1": 2, "2": 1}
    assert summary["cache_ratio_prompt_counts"] == {"0.25": 1, "0.5": 1}
