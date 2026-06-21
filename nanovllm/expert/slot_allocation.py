from __future__ import annotations

import csv
from pathlib import Path

import torch


def compute_layer_demand_from_act_freq(
    act_freq: torch.Tensor,
    coverage: float = 0.95,
) -> torch.Tensor:
    """Compute per-layer effective expert count from activation frequencies.

    For each layer, sort expert frequencies descending and find the minimum
    number of experts whose cumulative frequency reaches ``coverage``.
    """
    num_layers = act_freq.shape[0]
    demand = torch.empty(num_layers, dtype=torch.float64)
    for layer in range(num_layers):
        freqs = act_freq[layer].to(torch.float64)
        total = freqs.sum()
        if total <= 0:
            demand[layer] = 1.0
            continue
        sorted_freqs, _ = freqs.sort(descending=True)
        cumsum = sorted_freqs.cumsum(dim=0)
        threshold = total * coverage
        indices = (cumsum >= threshold).nonzero(as_tuple=False)
        if indices.numel() == 0:
            demand[layer] = float(freqs.numel())
        else:
            demand[layer] = float(indices[0].item() + 1)
    return demand


def compute_layer_demand_from_csv(csv_path: str) -> torch.Tensor:
    """Compute per-layer demand by averaging unique_expert_count_mean across
    all segment sizes for each layer."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Slot profile CSV not found: {csv_path}")

    layer_sums: dict[int, float] = {}
    layer_counts: dict[int, int] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer_idx = int(row["layer_idx"])
            value = float(row["unique_expert_count_mean"])
            layer_sums[layer_idx] = layer_sums.get(layer_idx, 0.0) + value
            layer_counts[layer_idx] = layer_counts.get(layer_idx, 0) + 1

    if not layer_sums:
        raise ValueError(f"No data rows found in CSV: {csv_path}")

    num_layers = max(layer_sums.keys()) + 1
    demand = torch.zeros(num_layers, dtype=torch.float64)
    for layer_idx in range(num_layers):
        if layer_idx in layer_sums and layer_counts[layer_idx] > 0:
            demand[layer_idx] = layer_sums[layer_idx] / layer_counts[layer_idx]
        else:
            demand[layer_idx] = 1.0
    return demand


def allocate_slots_per_layer(
    demand: torch.Tensor,
    total_budget: int,
    num_experts: int,
    num_buckets: int,
    max_bucket_ratio: float = 2.0,
    min_slots: int = 1,
) -> list[int]:
    """Distribute *total_budget* slots across layers proportionally to *demand*,
    quantised to at most *num_buckets* distinct values with the largest bucket
    no more than *max_bucket_ratio* times the smallest."""
    num_layers = int(demand.shape[0])
    if num_layers == 0:
        return []

    demand_f = demand.to(torch.float64)
    total_demand = demand_f.sum()
    if total_demand <= 0:
        uniform = total_budget // num_layers
        return [max(min_slots, min(uniform, num_experts))] * num_layers

    weights = demand_f / total_demand
    raw = (weights * total_budget).tolist()
    raw = [max(min_slots, min(v, num_experts)) for v in raw]

    raw_min = min(raw)
    raw_max = max(raw)
    if raw_min > 0 and raw_max / raw_min > max_bucket_ratio:
        scale = (max_bucket_ratio - 1.0) / (raw_max / raw_min - 1.0)
        mean_val = sum(raw) / num_layers
        raw = [mean_val + (v - mean_val) * scale for v in raw]
        raw = [max(min_slots, min(v, num_experts)) for v in raw]

    current_sum = sum(raw)
    if current_sum > 0 and abs(current_sum - total_budget) > 0.5:
        factor = total_budget / current_sum
        raw = [max(min_slots, min(v * factor, num_experts)) for v in raw]

    raw_min = min(raw)
    raw_max = max(raw)
    if num_buckets >= num_layers or raw_max - raw_min < 1.0:
        bucketed = [round(v) for v in raw]
    else:
        step = (raw_max - raw_min) / num_buckets
        centers = [raw_min + step * (i + 0.5) for i in range(num_buckets)]
        bucketed = [round(_nearest(v, centers)) for v in raw]

    bucketed = [max(min_slots, min(v, num_experts)) for v in bucketed]

    delta = total_budget - sum(bucketed)
    iterations = 0
    max_iterations = abs(delta) + num_layers
    while delta != 0 and iterations < max_iterations:
        fractional = [(raw[i] - bucketed[i], i) for i in range(num_layers)]
        if delta > 0:
            fractional.sort(key=lambda x: -x[0])
        else:
            fractional.sort(key=lambda x: x[0])

        progress = False
        for _, idx in fractional:
            if delta == 0:
                break
            if delta > 0 and bucketed[idx] < num_experts:
                bucketed[idx] += 1
                delta -= 1
                progress = True
            elif delta < 0 and bucketed[idx] > min_slots:
                bucketed[idx] -= 1
                delta += 1
                progress = True
        if not progress:
            break
        iterations += 1

    return bucketed


def _nearest(value: float, centers: list[float]) -> float:
    best = centers[0]
    best_dist = abs(value - best)
    for c in centers[1:]:
        dist = abs(value - c)
        if dist < best_dist:
            best = c
            best_dist = dist
    return best
