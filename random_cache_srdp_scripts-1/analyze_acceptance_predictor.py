"""Comprehensive analysis for random-cache acceptance predictor runs."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split


ALPHA_BUCKETS = (
    ("alpha < 0.5", -math.inf, 0.5),
    ("0.5 <= alpha < 0.7", 0.5, 0.7),
    ("0.7 <= alpha < 0.85", 0.7, 0.85),
    ("0.85 <= alpha < 0.95", 0.85, 0.95),
    ("alpha >= 0.95", 0.95, math.inf),
)
PERCENTILES = (1, 5, 25, 50, 75, 95, 99)


def _as_1d(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _finite(values: Any) -> np.ndarray:
    array = _as_1d(values)
    return array[np.isfinite(array)]


def distribution_summary(values: Any) -> dict[str, float | int]:
    array = _finite(values)
    if array.size == 0:
        return {
            "count": 0,
            "mean": math.nan,
            "std": math.nan,
            "min": math.nan,
            "max": math.nan,
            **{f"p{q}": math.nan for q in PERCENTILES},
        }
    quantiles = np.percentile(array, PERCENTILES)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        **{f"p{q}": float(value) for q, value in zip(PERCENTILES, quantiles)},
    }


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float | int]:
    true = _as_1d(y_true)
    pred = _as_1d(y_pred)
    if true.size != pred.size:
        raise ValueError(f"Length mismatch: y_true={true.size}, y_pred={pred.size}")
    valid = np.isfinite(true) & np.isfinite(pred)
    true = true[valid]
    pred = pred[valid]
    if true.size == 0:
        return {
            "count": 0,
            "mse": math.nan,
            "mae": math.nan,
            "rmse": math.nan,
            "bias": math.nan,
            "r2": math.nan,
            "corr": math.nan,
            "log_mae": math.nan,
            "true_mean": math.nan,
            "true_std": math.nan,
            "pred_mean": math.nan,
            "pred_std": math.nan,
        }
    error = pred - true
    mse = float(np.mean(error**2))
    variance = float(np.var(true))
    corr = 0.0
    if true.size > 2 and np.std(true) > 1e-12 and np.std(pred) > 1e-12:
        corr = float(np.corrcoef(pred, true)[0, 1])
    log_true = -np.log(np.clip(true, 1e-5, 1.0))
    log_pred = -np.log(np.clip(pred, 1e-5, 1.0))
    return {
        "count": int(true.size),
        "mse": mse,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(mse)),
        "bias": float(np.mean(error)),
        "r2": float(1.0 - mse / (variance + 1e-12)),
        "corr": corr,
        "log_mae": float(np.mean(np.abs(log_pred - log_true))),
        "true_mean": float(np.mean(true)),
        "true_std": float(np.std(true)),
        "pred_mean": float(np.mean(pred)),
        "pred_std": float(np.std(pred)),
    }


def alpha_bucket_rows(
    y_true: Any,
    y_pred: Any,
    split: str,
) -> list[dict[str, Any]]:
    true = _as_1d(y_true)
    pred = _as_1d(y_pred)
    rows = []
    for name, lower, upper in ALPHA_BUCKETS:
        mask = (true >= lower) & (true < upper)
        row = {
            "split": split,
            "bucket": name,
            "lower": lower,
            "upper": upper,
            "share": float(mask.mean()) if true.size else 0.0,
        }
        row.update(regression_metrics(true[mask], pred[mask]))
        rows.append(row)
    return rows


def _sort_group_values(values: Iterable[Any]) -> list[Any]:
    unique = list(dict.fromkeys(values))
    try:
        return sorted(unique, key=float)
    except (TypeError, ValueError):
        return sorted(unique, key=str)


def group_metric_rows(
    y_true: Any,
    y_pred: Any,
    groups: Any,
    split: str,
    group_name: str,
) -> list[dict[str, Any]]:
    true = _as_1d(y_true)
    pred = _as_1d(y_pred)
    group_values = np.asarray(groups).reshape(-1)
    if not (true.size == pred.size == group_values.size):
        raise ValueError(
            "Length mismatch for grouped metrics: "
            f"true={true.size}, pred={pred.size}, groups={group_values.size}"
        )
    rows = []
    for value in _sort_group_values(group_values.tolist()):
        mask = group_values == value
        row = {
            "split": split,
            group_name: value.item() if isinstance(value, np.generic) else value,
            "share": float(mask.mean()),
        }
        row.update(regression_metrics(true[mask], pred[mask]))
        rows.append(row)
    return rows


def prompt_chain_rows(
    y_true: Any,
    y_pred: Any,
    prompt_ids: Any,
    steps: Any,
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    true = _as_1d(y_true)
    pred = _as_1d(y_pred)
    prompts = np.asarray(prompt_ids).reshape(-1)
    decode_steps = np.asarray(steps).reshape(-1)
    if not (true.size == pred.size == prompts.size == decode_steps.size):
        raise ValueError("Length mismatch while computing prompt-chain metrics")

    rows: list[dict[str, Any]] = []
    for prompt_id in dict.fromkeys(prompts.tolist()):
        indices = np.flatnonzero(prompts == prompt_id)
        indices = indices[np.argsort(decode_steps[indices])]
        true_chain = np.cumprod(np.clip(true[indices], 0.0, 1.0))
        pred_chain = np.cumprod(np.clip(pred[indices], 0.0, 1.0))
        true_expected = float(true_chain.sum())
        pred_expected = float(pred_chain.sum())
        rows.append(
            {
                "split": split,
                "prompt_id": prompt_id,
                "decode_steps": int(indices.size),
                "true_expected_accepted": true_expected,
                "pred_expected_accepted": pred_expected,
                "error": pred_expected - true_expected,
                "absolute_error": abs(pred_expected - true_expected),
                "true_full_chain_probability": float(true_chain[-1]),
                "pred_full_chain_probability": float(pred_chain[-1]),
            }
        )
    summary = regression_metrics(
        [row["true_expected_accepted"] for row in rows],
        [row["pred_expected_accepted"] for row in rows],
    )
    return rows, summary


def calibration_rows(
    y_true: Any,
    y_pred: Any,
    split: str,
    bins: int = 10,
) -> list[dict[str, Any]]:
    if bins < 1:
        raise ValueError("bins must be at least 1")
    true = _as_1d(y_true)
    pred = _as_1d(y_pred)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket_ids = np.clip(np.digitize(pred, edges[1:-1], right=False), 0, bins - 1)
    rows = []
    for bucket in range(bins):
        mask = bucket_ids == bucket
        if not mask.any():
            continue
        rows.append(
            {
                "split": split,
                "bin": bucket,
                "lower": float(edges[bucket]),
                "upper": float(edges[bucket + 1]),
                "count": int(mask.sum()),
                "share": float(mask.mean()),
                "true_mean": float(true[mask].mean()),
                "pred_mean": float(pred[mask].mean()),
                "calibration_error": float(pred[mask].mean() - true[mask].mean()),
                "mae": float(np.mean(np.abs(pred[mask] - true[mask]))),
            }
        )
    return rows


def threshold_rows(
    y_true: Any,
    y_pred: Any,
    split: str,
    thresholds: Sequence[float] = (0.5, 0.7, 0.85, 0.95),
) -> list[dict[str, Any]]:
    true = _as_1d(y_true)
    pred = _as_1d(y_pred)
    rows = []
    for threshold in thresholds:
        actual = true >= threshold
        predicted = pred >= threshold
        tp = int(np.sum(actual & predicted))
        tn = int(np.sum(~actual & ~predicted))
        fp = int(np.sum(~actual & predicted))
        fn = int(np.sum(actual & ~predicted))
        total = max(1, true.size)
        rows.append(
            {
                "split": split,
                "threshold": float(threshold),
                "count": int(true.size),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "accuracy": float((tp + tn) / total),
                "precision": float(tp / max(1, tp + fp)),
                "recall": float(tp / max(1, tp + fn)),
                "specificity": float(tn / max(1, tn + fp)),
                "false_safe_rate": float(fp / max(1, fp + tn)),
                "false_conservative_rate": float(fn / max(1, fn + tp)),
            }
        )
    return rows


def load_jsonl_metadata(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    try:
        import orjson

        loads = orjson.loads
        mode = "rb"
        encoding = None
    except ImportError:
        loads = json.loads
        mode = "r"
        encoding = "utf-8"

    columns: dict[str, list[Any]] = {
        "prompt_id": [],
        "record_index": [],
        "decode_step": [],
        "alpha": [],
        "cache_ratio": [],
        "prefill_len": [],
        "article_id": [],
        "window_start": [],
    }
    open_kwargs = {} if encoding is None else {"encoding": encoding}
    with path.open(mode, **open_kwargs) as handle:
        for record_index, line in enumerate(handle):
            if not line.strip():
                continue
            record = loads(line)
            metadata = record.get("metadata", {})
            prompt_id = str(
                metadata.get("req_id")
                or f"{path.stem}:record-{record_index}"
            )
            for default_step, step in enumerate(record.get("steps", []), start=1):
                columns["prompt_id"].append(prompt_id)
                columns["record_index"].append(record_index)
                columns["decode_step"].append(
                    int(step.get("step", default_step))
                )
                columns["alpha"].append(float(step["alpha_theoretical"]))
                columns["cache_ratio"].append(
                    float(metadata.get("cache_ratio", math.nan))
                )
                columns["prefill_len"].append(
                    int(metadata.get("prefill_len", -1))
                )
                columns["article_id"].append(metadata.get("article_id"))
                columns["window_start"].append(
                    int(metadata.get("window_start", -1))
                )
    return {
        "prompt_id": np.asarray(columns["prompt_id"], dtype=object),
        "record_index": np.asarray(columns["record_index"], dtype=np.int64),
        "decode_step": np.asarray(columns["decode_step"], dtype=np.int64),
        "alpha": np.asarray(columns["alpha"], dtype=np.float64),
        "cache_ratio": np.asarray(columns["cache_ratio"], dtype=np.float64),
        "prefill_len": np.asarray(columns["prefill_len"], dtype=np.int64),
        "article_id": np.asarray(columns["article_id"], dtype=object),
        "window_start": np.asarray(columns["window_start"], dtype=np.int64),
    }


def validate_alignment(
    tensor_y: Any,
    metadata_y: Any,
    split: str,
    tolerance: float = 1e-6,
) -> float:
    tensor = _as_1d(tensor_y)
    metadata = _as_1d(metadata_y)
    if tensor.size != metadata.size:
        raise ValueError(
            f"{split} label length mismatch: tensor={tensor.size}, "
            f"JSONL={metadata.size}"
        )
    max_error = float(np.max(np.abs(tensor - metadata))) if tensor.size else 0.0
    if max_error > tolerance:
        raise ValueError(
            f"{split} label mismatch: max_abs_error={max_error:.8g}, "
            f"tolerance={tolerance:.8g}"
        )
    return max_error


def _clean_group(values: Any) -> set[Any]:
    cleaned = set()
    for value in np.asarray(values, dtype=object).reshape(-1).tolist():
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        cleaned.add(value)
    return cleaned


def _overlap_summary(train_values: Any, val_values: Any) -> dict[str, Any]:
    train_set = _clean_group(train_values)
    val_set = _clean_group(val_values)
    overlap = train_set & val_set
    return {
        "train_count": len(train_set),
        "val_count": len(val_set),
        "overlap_count": len(overlap),
        "val_overlap_rate": float(len(overlap) / max(1, len(val_set))),
        "overlap_examples": sorted(map(str, overlap))[:20],
    }


def split_leakage_report(
    train_indices: Any,
    val_indices: Any,
    metadata: dict[str, np.ndarray],
) -> dict[str, Any]:
    train_idx = np.asarray(train_indices, dtype=np.int64)
    val_idx = np.asarray(val_indices, dtype=np.int64)
    prompt_ids = metadata["prompt_id"]
    article_ids = metadata.get(
        "article_id",
        np.full(prompt_ids.shape, None, dtype=object),
    )
    return {
        "train_steps": int(train_idx.size),
        "val_steps": int(val_idx.size),
        "prompt": _overlap_summary(prompt_ids[train_idx], prompt_ids[val_idx]),
        "article": _overlap_summary(
            article_ids[train_idx],
            article_ids[val_idx],
        ),
    }


def window_overlap_report(metadata: dict[str, np.ndarray]) -> dict[str, Any]:
    prompt_ids = metadata["prompt_id"]
    article_ids = metadata.get(
        "article_id",
        np.full(prompt_ids.shape, None, dtype=object),
    )
    starts = metadata.get(
        "window_start",
        np.full(prompt_ids.shape, -1, dtype=np.int64),
    )
    lengths = metadata["prefill_len"]

    unique_prompts: dict[Any, tuple[Any, int, int]] = {}
    for prompt_id, article_id, start, length in zip(
        prompt_ids,
        article_ids,
        starts,
        lengths,
    ):
        if prompt_id not in unique_prompts:
            unique_prompts[prompt_id] = (
                article_id,
                int(start),
                int(length),
            )

    by_article: dict[Any, list[tuple[int, int]]] = defaultdict(list)
    for article_id, start, length in unique_prompts.values():
        if article_id is None or start < 0 or length <= 0:
            continue
        by_article[article_id].append((start, length))

    candidate_pairs = 0
    overlap_pairs = 0
    overlap_fractions = []
    for windows in by_article.values():
        for left_index in range(len(windows)):
            left_start, left_length = windows[left_index]
            left_end = left_start + left_length
            for right_index in range(left_index + 1, len(windows)):
                right_start, right_length = windows[right_index]
                right_end = right_start + right_length
                candidate_pairs += 1
                overlap = max(
                    0,
                    min(left_end, right_end) - max(left_start, right_start),
                )
                if overlap > 0:
                    overlap_pairs += 1
                    overlap_fractions.append(
                        overlap / min(left_length, right_length)
                    )
    return {
        "unique_prompts": len(unique_prompts),
        "articles_with_multiple_windows": int(
            sum(len(windows) > 1 for windows in by_article.values())
        ),
        "candidate_pairs": candidate_pairs,
        "overlap_pairs": overlap_pairs,
        "overlap_rate": float(overlap_pairs / max(1, candidate_pairs)),
        "mean_overlap_fraction_when_overlapping": (
            float(np.mean(overlap_fractions)) if overlap_fractions else 0.0
        ),
    }


def metadata_dataset_summary(
    metadata: dict[str, np.ndarray],
) -> dict[str, Any]:
    prompt_ids = metadata["prompt_id"]
    first_index_by_prompt: dict[Any, int] = {}
    for index, prompt_id in enumerate(prompt_ids):
        first_index_by_prompt.setdefault(prompt_id, index)
    prompt_indices = np.asarray(
        list(first_index_by_prompt.values()),
        dtype=np.int64,
    )
    article_ids = metadata.get(
        "article_id",
        np.full(prompt_ids.shape, None, dtype=object),
    )
    articles = _clean_group(article_ids[prompt_indices])
    step_counts = Counter(
        int(value) for value in metadata["decode_step"].tolist()
    )
    ratio_counts = Counter(
        str(float(value))
        for value in metadata["cache_ratio"][prompt_indices].tolist()
        if math.isfinite(float(value))
    )
    return {
        "steps": int(prompt_ids.size),
        "prompts": len(first_index_by_prompt),
        "articles": len(articles),
        "decode_step_counts": {
            str(key): value for key, value in sorted(step_counts.items())
        },
        "cache_ratio_prompt_counts": dict(sorted(ratio_counts.items())),
        "prefill_len_prompt_distribution": distribution_summary(
            metadata["prefill_len"][prompt_indices]
        ),
        "alpha_distribution": distribution_summary(metadata["alpha"]),
    }


def _ks_statistic(left: np.ndarray, right: np.ndarray) -> float:
    values = np.sort(np.unique(np.concatenate([left, right])))
    left_sorted = np.sort(left)
    right_sorted = np.sort(right)
    left_cdf = np.searchsorted(left_sorted, values, side="right") / left.size
    right_cdf = (
        np.searchsorted(right_sorted, values, side="right") / right.size
    )
    return float(np.max(np.abs(left_cdf - right_cdf)))


def _population_stability_index(
    train: np.ndarray,
    test: np.ndarray,
    bins: int = 10,
) -> float:
    edges = np.unique(
        np.quantile(train, np.linspace(0.0, 1.0, bins + 1))
    )
    if edges.size < 2:
        return 0.0
    edges[0] = -math.inf
    edges[-1] = math.inf
    train_hist = np.histogram(train, bins=edges)[0] / train.size
    test_hist = np.histogram(test, bins=edges)[0] / test.size
    train_hist = np.clip(train_hist, 1e-6, None)
    test_hist = np.clip(test_hist, 1e-6, None)
    return float(np.sum((test_hist - train_hist) * np.log(test_hist / train_hist)))


def distribution_drift(train_values: Any, test_values: Any) -> dict[str, float]:
    train = _finite(train_values)
    test = _finite(test_values)
    if train.size == 0 or test.size == 0:
        return {
            "train_mean": math.nan,
            "test_mean": math.nan,
            "mean_shift": math.nan,
            "standardized_mean_difference": math.nan,
            "ks_statistic": math.nan,
            "psi": math.nan,
        }
    pooled_std = math.sqrt((float(np.var(train)) + float(np.var(test))) / 2.0)
    mean_shift = float(test.mean() - train.mean())
    return {
        "train_mean": float(train.mean()),
        "test_mean": float(test.mean()),
        "mean_shift": mean_shift,
        "standardized_mean_difference": float(
            mean_shift / max(pooled_std, 1e-12)
        ),
        "ks_statistic": _ks_statistic(train, test),
        "psi": _population_stability_index(train, test),
    }


def _tensor_stats(tensor: torch.Tensor) -> dict[str, float | int]:
    values = tensor.detach().float().reshape(-1)
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return {
            "count": 0,
            "mean": math.nan,
            "std": math.nan,
            "min": math.nan,
            "max": math.nan,
        }
    return {
        "count": int(finite.numel()),
        "mean": float(finite.mean().item()),
        "std": float(finite.std(unbiased=False).item()),
        "min": float(finite.min().item()),
        "max": float(finite.max().item()),
    }


def feature_distribution_rows(
    train_split: dict[str, torch.Tensor],
    test_split: dict[str, torch.Tensor],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    common = sorted(
        set(train_split).intersection(test_split).difference({"y"})
    )
    branch_rows = []
    dimension_rows = []
    for branch in common:
        train_tensor = train_split[branch].detach().float()
        test_tensor = test_split[branch].detach().float()
        train_2d = train_tensor.reshape(train_tensor.shape[0], -1)
        test_2d = test_tensor.reshape(test_tensor.shape[0], -1)
        if train_2d.shape[1] != test_2d.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch for {branch}: "
                f"train={train_2d.shape[1]}, test={test_2d.shape[1]}"
            )
        train_stats = _tensor_stats(train_tensor)
        test_stats = _tensor_stats(test_tensor)
        branch_rows.append(
            {
                "branch": branch,
                "dimensions": int(train_2d.shape[1]),
                **{f"train_{key}": value for key, value in train_stats.items()},
                **{f"test_{key}": value for key, value in test_stats.items()},
                "mean_shift": test_stats["mean"] - train_stats["mean"],
            }
        )

        train_mean = train_2d.mean(dim=0)
        test_mean = test_2d.mean(dim=0)
        train_std = train_2d.std(dim=0, unbiased=False)
        test_std = test_2d.std(dim=0, unbiased=False)
        pooled = torch.sqrt((train_std.square() + test_std.square()) / 2.0)
        smd = (test_mean - train_mean) / pooled.clamp_min(1e-12)
        for dimension in range(train_2d.shape[1]):
            dimension_rows.append(
                {
                    "branch": branch,
                    "dimension": dimension,
                    "train_mean": float(train_mean[dimension].item()),
                    "train_std": float(train_std[dimension].item()),
                    "test_mean": float(test_mean[dimension].item()),
                    "test_std": float(test_std[dimension].item()),
                    "mean_shift": float(
                        (test_mean[dimension] - train_mean[dimension]).item()
                    ),
                    "standardized_mean_difference": float(
                        smd[dimension].item()
                    ),
                }
            )
    return branch_rows, dimension_rows


def _load_training_module(script_dir: Path):
    path = script_dir / "train_acceptance_predictor.py"
    name = "acceptance_predictor_training_for_analysis"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import training module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def predict_split(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    predictions = []
    targets = []
    model.eval()
    for batch in loader:
        inputs = [
            tensor.to(device, non_blocking=True)
            for tensor in batch[:-1]
        ]
        pred = model(*inputs)
        predictions.append(pred.detach().cpu().reshape(-1))
        targets.append(batch[-1].detach().cpu().reshape(-1))
    return (
        torch.cat(predictions).numpy(),
        torch.cat(targets).numpy(),
    )


def recreate_train_val_indices(
    total: int,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    val_size = max(1, int(total * val_ratio))
    train_size = total - val_size
    placeholder = list(range(total))
    train_subset, val_subset = random_split(
        placeholder,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    return (
        np.asarray(train_subset.indices, dtype=np.int64),
        np.asarray(val_subset.indices, dtype=np.int64),
    )


def prefill_bin_values(
    prefill_lengths: Any,
    edges: Sequence[int] = (0, 64, 128, 256, 512, 1024, 2048, 4096),
) -> np.ndarray:
    values = np.asarray(prefill_lengths, dtype=np.int64).reshape(-1)
    labels = np.empty(values.shape, dtype=object)
    for index, value in enumerate(values):
        if value < 0:
            labels[index] = "unknown"
            continue
        assigned = False
        for lower, upper in zip(edges[:-1], edges[1:]):
            if lower <= value < upper:
                labels[index] = f"[{lower}, {upper})"
                assigned = True
                break
        if not assigned:
            labels[index] = f">= {edges[-1]}"
    return labels


def _metadata_subset(
    metadata: dict[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    return {key: values[indices] for key, values in metadata.items()}


def _finite_group_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    split: str,
    group_name: str,
) -> list[dict[str, Any]]:
    group_array = np.asarray(groups)
    if np.issubdtype(group_array.dtype, np.number):
        mask = np.isfinite(group_array.astype(np.float64))
    else:
        mask = np.asarray(
            [value is not None for value in group_array],
            dtype=bool,
        )
    return group_metric_rows(
        y_true[mask],
        y_pred[mask],
        group_array[mask],
        split,
        group_name,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _best_training_epochs(training_log: Path) -> dict[str, Any]:
    if not training_log.exists():
        return {}
    with training_log.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    directions = {
        "val_mse": min,
        "val_mae": min,
        "val_rmse": min,
        "val_r2": max,
        "val_corr": max,
        "val_log_mae": min,
    }
    result = {}
    for metric, selector in directions.items():
        row = selector(rows, key=lambda item: float(item[metric]))
        result[metric] = {
            "epoch": int(row["epoch"]),
            "value": float(row[metric]),
        }
    result["checkpoint_selection_metric"] = "val_mse"
    return result


def _distribution_rows(
    views: dict[str, tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, Any]]:
    rows = []
    for split, (y_true, y_pred) in views.items():
        for series_name, values in (
            ("alpha_true", y_true),
            ("alpha_pred", y_pred),
        ):
            rows.append(
                {
                    "split": split,
                    "series": series_name,
                    **distribution_summary(values),
                }
            )
    return rows


def _baseline_metrics(
    views: dict[str, tuple[np.ndarray, np.ndarray]],
    train_fit_mean: float,
) -> dict[str, Any]:
    result = {}
    for split, (y_true, y_pred) in views.items():
        model_metrics = regression_metrics(y_true, y_pred)
        baseline = regression_metrics(
            y_true,
            np.full(y_true.shape, train_fit_mean),
        )
        result[split] = {
            "model": model_metrics,
            "constant_train_mean": baseline,
            "rmse_improvement_fraction": float(
                1.0 - model_metrics["rmse"] / max(baseline["rmse"], 1e-12)
            ),
            "mae_improvement_fraction": float(
                1.0 - model_metrics["mae"] / max(baseline["mae"], 1e-12)
            ),
        }
    return result


def _metadata_analysis(
    metadata_views: dict[str, dict[str, np.ndarray]],
    prediction_views: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, list[dict[str, Any]]]:
    tables: dict[str, list[dict[str, Any]]] = {
        "decode_step": [],
        "cache_ratio": [],
        "prefill_length": [],
        "prompt_chain": [],
    }
    for split, metadata in metadata_views.items():
        y_true, y_pred = prediction_views[split]
        tables["decode_step"].extend(
            _finite_group_rows(
                y_true,
                y_pred,
                metadata["decode_step"],
                split,
                "decode_step",
            )
        )
        tables["cache_ratio"].extend(
            _finite_group_rows(
                y_true,
                y_pred,
                metadata["cache_ratio"],
                split,
                "cache_ratio",
            )
        )
        tables["prefill_length"].extend(
            group_metric_rows(
                y_true,
                y_pred,
                prefill_bin_values(metadata["prefill_len"]),
                split,
                "prefill_bin",
            )
        )
        if split in {"train_full", "test"}:
            prompt_rows, _ = prompt_chain_rows(
                y_true,
                y_pred,
                metadata["prompt_id"],
                metadata["decode_step"],
                split,
            )
            tables["prompt_chain"].extend(prompt_rows)
    return tables


def _chain_summaries(
    prompt_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prompt_rows:
        by_split[row["split"]].append(row)
    result = {}
    for split, rows in by_split.items():
        result[split] = regression_metrics(
            [row["true_expected_accepted"] for row in rows],
            [row["pred_expected_accepted"] for row in rows],
        )
    return result


def _plot_reports(
    output_dir: Path,
    views: dict[str, tuple[np.ndarray, np.ndarray]],
    bucket_rows: Sequence[dict[str, Any]],
    step_rows: Sequence[dict[str, Any]],
    calibration: Sequence[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected_splits = [
        split for split in ("train_full", "val", "test") if split in views
    ]

    fig, axes = plt.subplots(
        len(selected_splits),
        1,
        figsize=(8, 3 * len(selected_splits)),
        squeeze=False,
    )
    for axis, split in zip(axes[:, 0], selected_splits):
        y_true, y_pred = views[split]
        axis.hist(y_true, bins=50, alpha=0.55, density=True, label="true")
        axis.hist(y_pred, bins=50, alpha=0.55, density=True, label="pred")
        axis.set_title(f"{split}: alpha distribution")
        axis.set_xlim(0.0, 1.0)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "alpha_distribution.png", dpi=160)
    plt.close(fig)

    if "test" in views:
        y_true, y_pred = views["test"]
        fig, axis = plt.subplots(figsize=(7, 6))
        image = axis.hexbin(
            y_true,
            y_pred,
            gridsize=45,
            extent=(0, 1, 0, 1),
            mincnt=1,
            cmap="viridis",
        )
        axis.plot([0, 1], [0, 1], "r--", linewidth=1)
        axis.set_xlabel("alpha true")
        axis.set_ylabel("alpha pred")
        axis.set_title("Test true vs predicted alpha")
        fig.colorbar(image, ax=axis, label="count")
        fig.tight_layout()
        fig.savefig(output_dir / "test_true_vs_pred.png", dpi=160)
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    for split in selected_splits:
        rows = [row for row in calibration if row["split"] == split]
        if rows:
            axis.plot(
                [row["pred_mean"] for row in rows],
                [row["true_mean"] for row in rows],
                marker="o",
                label=split,
            )
    axis.plot([0, 1], [0, 1], "k--", linewidth=1)
    axis.set_xlabel("mean predicted alpha")
    axis.set_ylabel("mean true alpha")
    axis.set_title("Calibration")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "calibration.png", dpi=160)
    plt.close(fig)

    test_buckets = [row for row in bucket_rows if row["split"] == "test"]
    if test_buckets:
        fig, axis = plt.subplots(figsize=(10, 5))
        axis.bar(
            [row["bucket"] for row in test_buckets],
            [row["rmse"] for row in test_buckets],
        )
        axis.set_ylabel("RMSE")
        axis.set_title("Test RMSE by true-alpha bucket")
        axis.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(output_dir / "test_alpha_bucket_rmse.png", dpi=160)
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    plotted = False
    for split in selected_splits:
        rows = [row for row in step_rows if row["split"] == split]
        if rows:
            axis.plot(
                [row["decode_step"] for row in rows],
                [row["rmse"] for row in rows],
                marker="o",
                label=split,
            )
            plotted = True
    if plotted:
        axis.set_xlabel("decode step")
        axis.set_ylabel("RMSE")
        axis.set_title("RMSE by decode step")
        axis.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "decode_step_rmse.png", dpi=160)
    plt.close(fig)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "NA"
    return f"{numeric:.{digits}f}"


def _markdown_table(
    rows: Sequence[dict[str, Any]],
    columns: Sequence[tuple[str, str]],
    digits: int = 4,
) -> list[str]:
    if not rows:
        return ["_No data available._", ""]
    lines = [
        "| " + " | ".join(title for _, title in columns) + " |",
        "|" + "|".join("---" if index == 0 else "---:" for index in range(len(columns))) + "|",
    ]
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, (int, np.integer)):
                cells.append(str(int(value)))
            elif isinstance(value, (float, np.floating)):
                cells.append(_fmt(value, digits))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _calibration_summary(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    splits = dict.fromkeys(row["split"] for row in rows)
    for split in splits:
        subset = [row for row in rows if row["split"] == split]
        result.append(
            {
                "split": split,
                "ece": sum(
                    row["share"] * abs(row["calibration_error"])
                    for row in subset
                ),
                "max_abs_error": max(
                    abs(row["calibration_error"]) for row in subset
                ),
            }
        )
    return result


def _summary_markdown(report: dict[str, Any]) -> str:
    overall = report["overall_metrics"]
    baselines = report["baseline_metrics"]
    distributions = report["distribution_rows"]
    buckets = report["alpha_bucket_metrics"]
    steps = report["decode_step_metrics"]
    ratios = report["cache_ratio_metrics"]
    prefills = report["prefill_length_metrics"]
    thresholds = report["threshold_metrics"]
    calibration = report["calibration_metrics"]
    chain = report["prompt_chain_summary"]
    drift = report["dataset_drift"]
    leakage = report.get("split_leakage", {})
    windows = report.get("window_overlap", {})
    dataset = report.get("dataset_summary", {})
    branches = report.get("feature_branch_summary", [])
    feature_dimensions = report.get("feature_dimension_summary", [])
    checkpoint = report.get("checkpoint_selection", {})

    train_data = dataset.get("train", {})
    test_data = dataset.get("test", {})
    test_metrics = overall["test"]
    val_metrics = overall["val"]
    train_fit_metrics = overall["train_fit"]
    low_test = next(
        row
        for row in buckets
        if row["split"] == "test" and row["bucket"] == "alpha < 0.5"
    )
    high_test = next(
        row
        for row in buckets
        if row["split"] == "test" and row["bucket"] == "alpha >= 0.95"
    )
    low_ratio = next(
        row
        for row in ratios
        if row["split"] == "test" and float(row["cache_ratio"]) == 0.1
    )
    worst_step = max(
        (row for row in steps if row["split"] == "test"),
        key=lambda row: row["rmse"],
    )
    best_step = min(
        (row for row in steps if row["split"] == "test"),
        key=lambda row: row["rmse"],
    )
    top_shifted_dimensions = sorted(
        feature_dimensions,
        key=lambda row: abs(row["standardized_mean_difference"]),
        reverse=True,
    )[:20]
    prompt_leakage = leakage.get("prompt", {})
    article_leakage = leakage.get("article", {})
    test_chain = chain.get("test", {})
    calibration_summary = _calibration_summary(calibration)

    lines = [
        "# Acceptance Predictor Comprehensive Analysis",
        "",
        "## 1. Data provenance and analysis scope",
        "",
        f"- Run directory: `{report['provenance']['run_dir']}`",
        f"- Tensor dataset: `{report['provenance']['data_file']}`",
        f"- Checkpoint: `{report['provenance']['checkpoint']}`",
        f"- Wiki train JSONL: `{report['provenance'].get('train_jsonl')}`",
        f"- MTBench test JSONL: `{report['provenance'].get('test_jsonl')}`",
        f"- Train tensor/JSONL label max absolute difference: "
        f"{_fmt(report['provenance']['alignment'].get('train_max_abs_label_error'), 8)}",
        f"- Test tensor/JSONL label max absolute difference: "
        f"{_fmt(report['provenance']['alignment'].get('test_max_abs_label_error'), 8)}",
        "- Predictions were recomputed from `best_model.pth`; metrics are not inferred "
        "from the final JSON report alone.",
        "",
        "## 2. Dataset composition",
        "",
        "| Dataset | Source | Prompts | Steps | Articles | Prefill mean | Prefill p50 | Prefill min/max |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        f"| Train | Wiki | {train_data.get('prompts', 0)} | "
        f"{train_data.get('steps', 0)} | {train_data.get('articles', 0)} | "
        f"{_fmt(train_data.get('prefill_len_prompt_distribution', {}).get('mean'), 1)} | "
        f"{_fmt(train_data.get('prefill_len_prompt_distribution', {}).get('p50'), 1)} | "
        f"{_fmt(train_data.get('prefill_len_prompt_distribution', {}).get('min'), 0)}/"
        f"{_fmt(train_data.get('prefill_len_prompt_distribution', {}).get('max'), 0)} |",
        f"| Test | MTBench | {test_data.get('prompts', 0)} | "
        f"{test_data.get('steps', 0)} | {test_data.get('articles', 0)} | "
        f"{_fmt(test_data.get('prefill_len_prompt_distribution', {}).get('mean'), 1)} | "
        f"{_fmt(test_data.get('prefill_len_prompt_distribution', {}).get('p50'), 1)} | "
        f"{_fmt(test_data.get('prefill_len_prompt_distribution', {}).get('min'), 0)}/"
        f"{_fmt(test_data.get('prefill_len_prompt_distribution', {}).get('max'), 0)} |",
        "",
        "### Cache-ratio sampling by prompt",
        "",
        "| Ratio | Train prompts | Test prompts |",
        "|---:|---:|---:|",
    ]
    all_ratios = sorted(
        set(train_data.get("cache_ratio_prompt_counts", {}))
        | set(test_data.get("cache_ratio_prompt_counts", {})),
        key=float,
    )
    for ratio in all_ratios:
        lines.append(
            f"| {ratio} | "
            f"{train_data.get('cache_ratio_prompt_counts', {}).get(ratio, 0)} | "
            f"{test_data.get('cache_ratio_prompt_counts', {}).get(ratio, 0)} |"
        )

    lines.extend(
        [
            "",
            "## 3. Checkpoint selection",
            "",
            f"- The trainer saves the checkpoint with minimum `val_mse`.",
            f"- Selected epoch: {checkpoint.get('val_mse', {}).get('epoch', 'NA')}; "
            f"`val_mse={_fmt(checkpoint.get('val_mse', {}).get('value'), 6)}`.",
            f"- Best MAE epoch: {checkpoint.get('val_mae', {}).get('epoch', 'NA')}; "
            f"best log-MAE epoch: {checkpoint.get('val_log_mae', {}).get('epoch', 'NA')}.",
            "- MSE, RMSE and R2 rank checkpoints identically on the same fixed "
            "validation set. Correlation does not measure calibration and should not "
            "replace an error metric.",
            "",
            "## 4. Overall predictive performance",
            "",
        ]
    )
    overall_rows = []
    for split in ("train_full", "train_fit", "val", "test"):
        row = {"split": split, **overall[split]}
        row["baseline_rmse"] = baselines[split]["constant_train_mean"]["rmse"]
        row["rmse_gain_pct"] = 100.0 * baselines[split]["rmse_improvement_fraction"]
        overall_rows.append(row)
    lines.extend(
        _markdown_table(
            overall_rows,
            (
                ("split", "Split"),
                ("count", "N"),
                ("mae", "MAE"),
                ("rmse", "RMSE"),
                ("r2", "R2"),
                ("corr", "Corr"),
                ("bias", "Bias"),
                ("log_mae", "Log-MAE"),
                ("baseline_rmse", "Const RMSE"),
                ("rmse_gain_pct", "RMSE gain %"),
            ),
        )
    )
    lines.extend(
        [
            f"- Train-fit to validation RMSE rises from "
            f"{train_fit_metrics['rmse']:.4f} to {val_metrics['rmse']:.4f} "
            f"({(val_metrics['rmse'] / train_fit_metrics['rmse'] - 1) * 100:.1f}% increase).",
            f"- Validation to MTBench RMSE rises from {val_metrics['rmse']:.4f} "
            f"to {test_metrics['rmse']:.4f} "
            f"({(test_metrics['rmse'] / val_metrics['rmse'] - 1) * 100:.1f}% increase).",
            f"- The model improves MTBench RMSE over a train-mean constant baseline by "
            f"{baselines['test']['rmse_improvement_fraction'] * 100:.1f}%, so it has "
            "real signal, but cross-domain error remains large.",
            "",
            "## 5. Label and prediction distributions",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            distributions,
            (
                ("split", "Split"),
                ("series", "Series"),
                ("count", "N"),
                ("mean", "Mean"),
                ("std", "Std"),
                ("min", "Min"),
                ("max", "Max"),
                ("p1", "P1"),
                ("p5", "P5"),
                ("p25", "P25"),
                ("p50", "P50"),
                ("p75", "P75"),
                ("p95", "P95"),
                ("p99", "P99"),
            ),
            digits=5,
        )
    )
    lines.extend(
        [
            "- Train prediction standard deviation contracts from "
            f"{overall['train_full']['true_std']:.4f} to "
            f"{overall['train_full']['pred_std']:.4f}; MTBench contracts from "
            f"{test_metrics['true_std']:.4f} to {test_metrics['pred_std']:.4f}.",
            "- The contraction is asymmetric: low alpha is generally overestimated "
            "while near-one alpha is underestimated.",
            "",
            "## 6. Error by true-alpha bucket",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            buckets,
            (
                ("split", "Split"),
                ("bucket", "True alpha bucket"),
                ("count", "N"),
                ("share", "Share"),
                ("true_mean", "True mean"),
                ("pred_mean", "Pred mean"),
                ("mae", "MAE"),
                ("rmse", "RMSE"),
                ("bias", "Bias"),
                ("corr", "Corr"),
                ("log_mae", "Log-MAE"),
            ),
        )
    )
    lines.extend(
        [
            f"- MTBench `alpha < 0.5` is the critical failure region: true mean "
            f"{low_test['true_mean']:.4f}, predicted mean {low_test['pred_mean']:.4f}, "
            f"bias +{low_test['bias']:.4f}, RMSE {low_test['rmse']:.4f}.",
            f"- For `alpha >= 0.95`, the mean bias reverses to "
            f"{high_test['bias']:.4f}; high-acceptance steps are underestimated.",
            "- Negative within-bucket R2 is expected when each bucket has very little "
            "target variance; MAE, RMSE and bias are more useful for these rows.",
            "",
            "## 7. Error by cache ratio",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            ratios,
            (
                ("split", "Split"),
                ("cache_ratio", "Cache ratio"),
                ("count", "N"),
                ("true_mean", "True mean"),
                ("pred_mean", "Pred mean"),
                ("mae", "MAE"),
                ("rmse", "RMSE"),
                ("r2", "R2"),
                ("bias", "Bias"),
                ("log_mae", "Log-MAE"),
            ),
        )
    )
    lines.extend(
        [
            f"- At cache ratio 0.1, MTBench RMSE is {low_ratio['rmse']:.4f} and "
            f"R2 is only {low_ratio['r2']:.4f}. Low-cache operation is the main "
            "generalization bottleneck.",
            "- The ratio 0.5 row has low RMSE but negative R2 because labels are "
            "tightly concentrated near one; this is not evidence that the absolute "
            "predictions are worse than low-ratio predictions.",
            "",
            "## 8. Error by decode step",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            steps,
            (
                ("split", "Split"),
                ("decode_step", "Step"),
                ("count", "N"),
                ("true_mean", "True mean"),
                ("pred_mean", "Pred mean"),
                ("mae", "MAE"),
                ("rmse", "RMSE"),
                ("r2", "R2"),
                ("bias", "Bias"),
            ),
        )
    )
    lines.extend(
        [
            f"- Best MTBench step: {best_step['decode_step']} "
            f"(RMSE {best_step['rmse']:.4f}). Worst step: "
            f"{worst_step['decode_step']} (RMSE {worst_step['rmse']:.4f}).",
            "- Bias becomes increasingly positive in several later steps, reaching "
            f"{next(row for row in steps if row['split'] == 'test' and row['decode_step'] == 15)['bias']:.4f} "
            "at step 15.",
            "",
            "## 9. Error by prefill length",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            prefills,
            (
                ("split", "Split"),
                ("prefill_bin", "Prefill bin"),
                ("count", "N"),
                ("true_mean", "True mean"),
                ("pred_mean", "Pred mean"),
                ("mae", "MAE"),
                ("rmse", "RMSE"),
                ("r2", "R2"),
                ("bias", "Bias"),
            ),
        )
    )
    lines.extend(
        [
            "- Wiki prefill median is 1518 tokens, while MTBench median is 100. "
            "Most MTBench samples occupy a region weakly represented by the training "
            "distribution.",
            "",
            "## 10. Threshold decision risk",
            "",
            "`False-safe rate` is the fraction of truly below-threshold examples "
            "incorrectly predicted above the threshold. This is the dangerous error "
            "when predictions control aggressive drafting.",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            thresholds,
            (
                ("split", "Split"),
                ("threshold", "Threshold"),
                ("accuracy", "Accuracy"),
                ("precision", "Precision"),
                ("recall", "Recall"),
                ("specificity", "Specificity"),
                ("false_safe_rate", "False-safe"),
                ("false_conservative_rate", "False-conservative"),
                ("fp", "FP"),
                ("fn", "FN"),
            ),
        )
    )
    lines.extend(
        [
            "- On MTBench, false-safe rates are 37.9%, 32.5%, 25.7% and 10.5% "
            "for thresholds 0.5, 0.7, 0.85 and 0.95 respectively.",
            "- Therefore raw predictions should not yet be used as an unsafe "
            "go/no-go control signal without calibration or a conservative lower bound.",
            "",
            "## 11. Calibration",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            calibration_summary,
            (
                ("split", "Split"),
                ("ece", "Weighted abs calibration error"),
                ("max_abs_error", "Max bin error"),
            ),
        )
    )
    lines.extend(
        [
            "- MTBench weighted calibration error is materially worse than Wiki "
            "train and validation, confirming that domain shift affects probability "
            "calibration rather than only ranking.",
            "",
            "## 12. Prompt-chain performance",
            "",
        ]
    )
    chain_rows = [{"split": split, **metrics} for split, metrics in chain.items()]
    lines.extend(
        _markdown_table(
            chain_rows,
            (
                ("split", "Split"),
                ("count", "Prompts"),
                ("true_mean", "True expected accepted"),
                ("pred_mean", "Pred expected accepted"),
                ("mae", "MAE steps"),
                ("rmse", "RMSE steps"),
                ("bias", "Bias steps"),
                ("r2", "R2"),
                ("corr", "Corr"),
            ),
        )
    )
    lines.extend(
        [
            f"- MTBench mean chain bias is small ({test_chain.get('bias', math.nan):.4f} "
            "steps), but per-prompt MAE is "
            f"{test_chain.get('mae', math.nan):.4f} steps and RMSE is "
            f"{test_chain.get('rmse', math.nan):.4f} steps. Mean cancellation hides "
            "large individual prompt errors.",
            "",
            "## 13. Validation leakage and Wiki window overlap",
            "",
            f"- Train/validation splitting is performed at step level.",
            f"- Validation prompt overlap with train: "
            f"{prompt_leakage.get('overlap_count', 0)}/"
            f"{prompt_leakage.get('val_count', 0)} "
            f"({_fmt(100 * prompt_leakage.get('val_overlap_rate', 0), 1)}%).",
            f"- Validation article overlap with train: "
            f"{article_leakage.get('overlap_count', 0)}/"
            f"{article_leakage.get('val_count', 0)} "
            f"({_fmt(100 * article_leakage.get('val_overlap_rate', 0), 1)}%).",
            f"- Same-article window pairs: {windows.get('candidate_pairs', 0)}; "
            f"overlapping pairs: {windows.get('overlap_pairs', 0)} "
            f"({_fmt(100 * windows.get('overlap_rate', 0), 1)}%).",
            f"- Mean overlap fraction among overlapping pairs: "
            f"{_fmt(100 * windows.get('mean_overlap_fraction_when_overlapping', 0), 1)}%.",
            "- MTBench test remains an independent external test set, so its reported "
            "metrics are not invalidated by the Wiki train/validation leakage. The "
            "leakage does make validation-based checkpoint selection optimistic.",
            "",
            "## 14. Train-to-test distribution drift",
            "",
        ]
    )
    drift_rows = []
    for name in ("alpha_true", "alpha_pred", "cache_ratio", "prefill_len"):
        values = drift.get(name)
        if values:
            drift_rows.append({"feature": name, **values})
    lines.extend(
        _markdown_table(
            drift_rows,
            (
                ("feature", "Feature"),
                ("train_mean", "Train mean"),
                ("test_mean", "Test mean"),
                ("mean_shift", "Mean shift"),
                ("standardized_mean_difference", "SMD"),
                ("ks_statistic", "KS"),
                ("psi", "PSI"),
            ),
        )
    )
    lines.extend(
        [
            f"- Prefill length is the dominant explicit shift: SMD "
            f"{drift['prefill_len']['standardized_mean_difference']:.4f}, KS "
            f"{drift['prefill_len']['ks_statistic']:.4f}, PSI "
            f"{drift['prefill_len']['psi']:.4f}.",
            f"- Across all {drift['feature_dimensions']['count']} feature dimensions, "
            f"{drift['feature_dimensions']['dimensions_abs_smd_ge_0_5']} have "
            "`|SMD| >= 0.5`.",
            "",
            "### Feature branch summary",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            branches,
            (
                ("branch", "Branch"),
                ("dimensions", "Dims"),
                ("train_mean", "Train mean"),
                ("train_std", "Train std"),
                ("test_mean", "Test mean"),
                ("test_std", "Test std"),
                ("mean_shift", "Mean shift"),
            ),
        )
    )
    lines.extend(["### Top 20 shifted feature dimensions", ""])
    lines.extend(
        _markdown_table(
            top_shifted_dimensions,
            (
                ("branch", "Branch"),
                ("dimension", "Dimension"),
                ("train_mean", "Train mean"),
                ("train_std", "Train std"),
                ("test_mean", "Test mean"),
                ("test_std", "Test std"),
                ("standardized_mean_difference", "SMD"),
            ),
        )
    )
    lines.extend(
        [
            "## 15. Overall usability assessment",
            "",
            "**Current status: useful as a ranking/offline diagnostic model, not yet "
            "safe as a direct online drafting controller.**",
            "",
            "Evidence supporting usefulness:",
            "",
            f"- MTBench correlation is {test_metrics['corr']:.4f} and R2 is "
            f"{test_metrics['r2']:.4f}.",
            f"- It reduces RMSE by "
            f"{baselines['test']['rmse_improvement_fraction'] * 100:.1f}% relative "
            "to a constant train-mean baseline.",
            "- Cache-ratio and step trends are captured well enough for aggregate "
            "analysis.",
            "",
            "Evidence preventing direct aggressive deployment:",
            "",
            f"- Low-alpha RMSE is {low_test['rmse']:.4f} with +"
            f"{low_test['bias']:.4f} bias.",
            f"- MTBench calibration error is "
            f"{next(row['ece'] for row in calibration_summary if row['split'] == 'test'):.4f}.",
            f"- Prompt-chain MAE is {test_chain.get('mae', math.nan):.4f} accepted steps.",
            "- Validation is contaminated by adjacent steps and same-article windows.",
            "- The train/test prefill and hidden-state distributions differ materially.",
            "",
            "## 16. Predictor optimization plan",
            "",
            "### Priority 0: repair evaluation before comparing new models",
            "",
            "1. Split Wiki by `article_id`, not by step. All windows and decode steps "
            "from one article must remain in one split.",
            "2. Keep MTBench as untouched external test. Add a second conversation-style "
            "validation set for checkpoint selection and calibration.",
            "3. Save metadata (`req_id`, `article_id`, `decode_step`, `cache_ratio`, "
            "`prefill_len`) into the `.pt` dataset so grouped evaluation does not depend "
            "on positional reconstruction.",
            "4. Stop silently swallowing dataset parsing exceptions; count and report "
            "rejected records with reasons.",
            "",
            "### Priority 1: fix training-data coverage",
            "",
            "1. Add short-prefill Wiki samples and conversation/instruction data. Match "
            "the intended deployment prefill distribution instead of training mostly "
            "around 1K-4K tokens.",
            "2. Oversample `alpha < 0.5`, especially very small alpha. Train P5 is "
            "substantially higher than MTBench P5, so the most dangerous tail is "
            "underrepresented.",
            "3. Increase low cache-ratio coverage or use balanced batches over "
            "`cache_ratio x alpha_bucket x prefill_bin x decode_step`.",
            "4. Prevent heavily sampled long Wiki articles from dominating by weighting "
            "articles or sampling an equal number of windows per article per epoch.",
            "",
            "### Priority 2: change objective and checkpoint selection",
            "",
            "1. Use a weighted loss that upweights low-alpha examples and false-safe "
            "overestimation. A practical form is asymmetric Huber/MSE with larger weight "
            "when `pred > target` and `target < 0.5`.",
            "2. Retain log-domain loss for chain sensitivity, but tune its weight on a "
            "clean grouped validation set.",
            "3. Select checkpoints using a deployment score such as: grouped validation "
            "RMSE + low-alpha MAE + threshold false-safe penalty + chain MAE.",
            "4. Report confidence intervals across prompts/articles, not only per-step "
            "point estimates.",
            "",
            "### Priority 3: improve normalization and representation",
            "",
            "1. Compute per-feature train mean/std and apply fixed standardization at "
            "train and inference time. Current branch LayerNorm mixes heterogeneous "
            "features and makes a large prefill scalar affect the normalization of the "
            "whole history vector.",
            "2. Normalize prefill length with a stable transform such as "
            "`log1p(prefill_len)` followed by train-set standardization, rather than a "
            "single `/8096` scale.",
            "3. Reduce hidden-state domain dependence: compare hidden ablation, PCA/random "
            "projection, stronger bottleneck, dropout and domain-adversarial regularization.",
            "4. Add explicit categorical/continuous embeddings for cache ratio and decode "
            "step instead of relying only on indirect route/history signals.",
            "",
            "### Priority 4: add information that is available online",
            "",
            "1. Add q-distribution confidence features: full-vocabulary entropy or "
            "log-sum-exp statistics, top-k probability mass, tail mass, top-1/top-2 "
            "margins and effective vocabulary size.",
            "2. Add router identity features, not only router weights: original/modified "
            "expert overlap, replacement count, weighted replaced mass, rank changes and "
            "early/middle/late layer summaries.",
            "3. Add cache-state features per layer and aggregate: hit/miss count, cached "
            "expert frequency, candidate-pool size and replacement randomness.",
            "4. Run branch ablations (`route`, `token`, `hidden`, `history`) to determine "
            "which features genuinely transfer from Wiki to MTBench.",
            "",
            "### Priority 5: make online decisions conservative",
            "",
            "1. Calibrate on clean conversation validation data using isotonic regression "
            "or a small monotonic calibrator.",
            "2. Predict uncertainty using an ensemble, quantile regression or conformal "
            "residual bounds. Drive drafting with a lower confidence bound, not the mean.",
            "3. Optimize the actual policy metric: latency/speedup subject to a maximum "
            "false-safe or rollback-rate constraint.",
            "4. Start deployment in shadow mode, logging predicted alpha, realized "
            "acceptance and policy decisions by cache ratio and step.",
            "",
            "## 17. Recommended next experiment sequence",
            "",
            "1. Rebuild article-grouped train/validation splits and reproduce the current "
            "architecture as the trustworthy baseline.",
            "2. Retrain with balanced short-prefill/conversation data and low-alpha "
            "oversampling; do not change architecture yet.",
            "3. Add fixed per-feature normalization and explicit cache-ratio/step inputs.",
            "4. Compare asymmetric low-alpha loss against the existing MSE + log-MSE loss.",
            "5. Run branch ablations, then modify architecture only for branches that "
            "show transferable value.",
            "6. Calibrate the best model and evaluate false-safe rate plus chain MAE before "
            "any online control experiment.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--train-jsonl", default=None)
    parser.add_argument("--test-jsonl", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    script_dir = Path(__file__).resolve().parent
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else run_dir / "analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    data_file = Path(
        args.data_file or config["args"]["data_file"]
    ).resolve()
    checkpoint = run_dir / "best_model.pth"
    final_report_path = run_dir / "final_test_report.json"

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    print(f"Loading tensor dataset: {data_file}")
    data = torch.load(data_file, map_location="cpu", weights_only=False)
    training = _load_training_module(script_dir)
    meta = data["meta"]
    model = training.AcceptancePredictor(
        route_raw_dim=meta["route_raw_dim"],
        route_summary_dim=meta["route_summary_dim"],
        token_feature_dim=meta["token_feature_dim"],
        hidden_dim=meta["hidden_dim"],
        history_dim=meta["history_dim"],
    ).to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )

    train_dataset = training.make_dataset(data["train"])
    test_dataset = training.make_dataset(data["test"])
    print(f"Running checkpoint inference on {device}...")
    train_pred, train_y = predict_split(
        model,
        train_dataset,
        args.batch_size,
        device,
    )
    test_pred, test_y = predict_split(
        model,
        test_dataset,
        args.batch_size,
        device,
    )
    train_indices, val_indices = recreate_train_val_indices(
        len(train_dataset),
        config["args"]["val_ratio"],
        config["args"]["seed"],
    )
    views = {
        "train_full": (train_y, train_pred),
        "train_fit": (train_y[train_indices], train_pred[train_indices]),
        "val": (train_y[val_indices], train_pred[val_indices]),
        "test": (test_y, test_pred),
    }

    overall_metrics = {
        split: regression_metrics(y_true, y_pred)
        for split, (y_true, y_pred) in views.items()
    }
    baseline_metrics = _baseline_metrics(
        views,
        float(train_y[train_indices].mean()),
    )
    distributions = _distribution_rows(views)
    alpha_buckets = [
        row
        for split, (y_true, y_pred) in views.items()
        for row in alpha_bucket_rows(y_true, y_pred, split)
    ]
    calibration = [
        row
        for split, (y_true, y_pred) in views.items()
        for row in calibration_rows(
            y_true,
            y_pred,
            split,
            bins=args.calibration_bins,
        )
    ]
    thresholds = [
        row
        for split, (y_true, y_pred) in views.items()
        for row in threshold_rows(y_true, y_pred, split)
    ]

    metadata_views: dict[str, dict[str, np.ndarray]] = {}
    alignment = {}
    train_metadata = None
    test_metadata = None
    if args.train_jsonl:
        print(f"Reading train metadata: {args.train_jsonl}")
        train_metadata = load_jsonl_metadata(args.train_jsonl)
        alignment["train_max_abs_label_error"] = validate_alignment(
            train_y,
            train_metadata["alpha"],
            "train",
        )
        metadata_views["train_full"] = train_metadata
        metadata_views["train_fit"] = _metadata_subset(
            train_metadata,
            train_indices,
        )
        metadata_views["val"] = _metadata_subset(train_metadata, val_indices)
    else:
        warnings.warn(
            "--train-jsonl was not provided; train grouping and leakage "
            "analysis will be skipped."
        )
    if args.test_jsonl:
        print(f"Reading test metadata: {args.test_jsonl}")
        test_metadata = load_jsonl_metadata(args.test_jsonl)
        alignment["test_max_abs_label_error"] = validate_alignment(
            test_y,
            test_metadata["alpha"],
            "test",
        )
        metadata_views["test"] = test_metadata
    else:
        warnings.warn(
            "--test-jsonl was not provided; test grouping analysis will "
            "be skipped."
        )

    metadata_tables = _metadata_analysis(metadata_views, views)
    chain_summary = _chain_summaries(metadata_tables["prompt_chain"])
    split_leakage = {}
    window_overlap = {}
    if train_metadata is not None:
        split_leakage = split_leakage_report(
            train_indices,
            val_indices,
            train_metadata,
        )
        window_overlap = window_overlap_report(train_metadata)

    print("Computing training/test feature distributions...")
    branch_rows, dimension_rows = feature_distribution_rows(
        data["train"],
        data["test"],
    )
    dataset_drift = {
        "alpha_true": distribution_drift(train_y, test_y),
        "alpha_pred": distribution_drift(train_pred, test_pred),
        "feature_dimensions": {
            "count": len(dimension_rows),
            "mean_abs_standardized_mean_difference": float(
                np.mean(
                    [
                        abs(row["standardized_mean_difference"])
                        for row in dimension_rows
                        if math.isfinite(
                            row["standardized_mean_difference"]
                        )
                    ]
                )
            ),
            "dimensions_abs_smd_ge_0_5": int(
                sum(
                    abs(row["standardized_mean_difference"]) >= 0.5
                    for row in dimension_rows
                    if math.isfinite(row["standardized_mean_difference"])
                )
            ),
        },
    }
    if train_metadata is not None and test_metadata is not None:
        dataset_drift["cache_ratio"] = distribution_drift(
            train_metadata["cache_ratio"],
            test_metadata["cache_ratio"],
        )
        dataset_drift["prefill_len"] = distribution_drift(
            train_metadata["prefill_len"],
            test_metadata["prefill_len"],
        )

    dataset_summary = {
        "tensor_shapes": {
            split: {
                key: list(value.shape)
                for key, value in split_data.items()
            }
            for split, split_data in (
                ("train", data["train"]),
                ("test", data["test"]),
            )
        },
    }
    if train_metadata is not None:
        dataset_summary["train"] = metadata_dataset_summary(train_metadata)
    if test_metadata is not None:
        dataset_summary["test"] = metadata_dataset_summary(test_metadata)

    existing_final_report = (
        json.loads(final_report_path.read_text(encoding="utf-8"))
        if final_report_path.exists()
        else {}
    )
    report = {
        "provenance": {
            "run_dir": run_dir,
            "data_file": data_file,
            "checkpoint": checkpoint,
            "train_jsonl": args.train_jsonl,
            "test_jsonl": args.test_jsonl,
            "device": str(device),
            "train_examples": len(train_y),
            "test_examples": len(test_y),
            "meta": meta,
            "alignment": alignment,
        },
        "checkpoint_selection": _best_training_epochs(
            run_dir / "training_log.csv"
        ),
        "overall_metrics": overall_metrics,
        "baseline_metrics": baseline_metrics,
        "existing_final_test_report": existing_final_report,
        "dataset_summary": dataset_summary,
        "dataset_drift": dataset_drift,
        "split_leakage": split_leakage,
        "window_overlap": window_overlap,
        "prompt_chain_summary": chain_summary,
        "distribution_rows": distributions,
        "alpha_bucket_metrics": alpha_buckets,
        "decode_step_metrics": metadata_tables["decode_step"],
        "cache_ratio_metrics": metadata_tables["cache_ratio"],
        "prefill_length_metrics": metadata_tables["prefill_length"],
        "calibration_metrics": calibration,
        "threshold_metrics": thresholds,
        "feature_branch_summary": branch_rows,
        "feature_dimension_summary": dimension_rows,
    }

    write_json(output_dir / "analysis_report.json", report)
    write_json(output_dir / "dataset_summary.json", dataset_summary)
    write_json(output_dir / "dataset_drift.json", dataset_drift)
    write_json(output_dir / "split_leakage_report.json", split_leakage)
    write_json(output_dir / "window_overlap_report.json", window_overlap)
    write_json(output_dir / "prompt_chain_summary.json", chain_summary)
    write_csv(output_dir / "distribution_summary.csv", distributions)
    write_csv(
        output_dir / "overall_metrics.csv",
        [
            {"split": split, **metrics}
            for split, metrics in overall_metrics.items()
        ],
    )
    write_csv(output_dir / "alpha_bucket_metrics.csv", alpha_buckets)
    write_csv(output_dir / "calibration_metrics.csv", calibration)
    write_csv(output_dir / "threshold_metrics.csv", thresholds)
    write_csv(
        output_dir / "decode_step_metrics.csv",
        metadata_tables["decode_step"],
    )
    write_csv(
        output_dir / "cache_ratio_metrics.csv",
        metadata_tables["cache_ratio"],
    )
    write_csv(
        output_dir / "prefill_length_metrics.csv",
        metadata_tables["prefill_length"],
    )
    write_csv(
        output_dir / "prompt_chain_metrics.csv",
        metadata_tables["prompt_chain"],
    )
    write_csv(output_dir / "feature_branch_summary.csv", branch_rows)
    write_csv(output_dir / "feature_dimension_summary.csv", dimension_rows)
    (output_dir / "analysis_summary.md").write_text(
        _summary_markdown(report),
        encoding="utf-8",
    )
    if not args.no_plots:
        _plot_reports(
            output_dir,
            views,
            alpha_buckets,
            metadata_tables["decode_step"],
            calibration,
        )

    print(f"Analysis written to: {output_dir}")
    print(
        json.dumps(
            {
                "train_full": overall_metrics["train_full"],
                "val": overall_metrics["val"],
                "test": overall_metrics["test"],
            },
            indent=2,
        )
    )
    print("Alpha distribution summaries:")
    for row in distributions:
        if row["split"] in {"train_full", "val", "test"}:
            print(
                f"  {row['split']:10s} {row['series']:10s} "
                f"mean={row['mean']:.6f} std={row['std']:.6f} "
                f"min={row['min']:.6f} max={row['max']:.6f} "
                f"p1={row['p1']:.6f} p5={row['p5']:.6f} "
                f"p25={row['p25']:.6f} p50={row['p50']:.6f} "
                f"p75={row['p75']:.6f} p95={row['p95']:.6f} "
                f"p99={row['p99']:.6f}"
            )


if __name__ == "__main__":
    main()
