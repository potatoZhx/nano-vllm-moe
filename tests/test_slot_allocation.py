"""Tests for nanovllm.expert.slot_allocation."""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest
import torch

from nanovllm.expert.slot_allocation import (
    allocate_slots_per_layer,
    compute_layer_demand_from_act_freq,
    compute_layer_demand_from_csv,
)


class TestComputeLayerDemandFromActFreq:
    def test_output_shape(self):
        act_freq = torch.rand(10, 64)
        demand = compute_layer_demand_from_act_freq(act_freq)
        assert demand.shape == (10,)

    def test_concentrated_layer(self):
        act_freq = torch.zeros(1, 64)
        act_freq[0, 0] = 1.0
        demand = compute_layer_demand_from_act_freq(act_freq)
        assert demand[0].item() == 1.0

    def test_uniform_layer(self):
        act_freq = torch.ones(1, 64) / 64.0
        demand = compute_layer_demand_from_act_freq(act_freq, coverage=0.95)
        expected = int(64 * 0.95)
        assert abs(demand[0].item() - expected) <= 1

    def test_zero_freq_layer(self):
        act_freq = torch.zeros(1, 64)
        demand = compute_layer_demand_from_act_freq(act_freq)
        assert demand[0].item() == 1.0

    def test_relative_ordering(self):
        act_freq = torch.zeros(2, 64)
        act_freq[0, :4] = 0.25
        act_freq[1] = 1.0 / 64.0
        demand = compute_layer_demand_from_act_freq(act_freq)
        assert demand[0].item() < demand[1].item()


class TestComputeLayerDemandFromCsv:
    def _write_csv(self, path: Path, rows: list[dict]) -> None:
        fieldnames = [
            "start_token_n_1based",
            "segment_size_c",
            "layer_idx",
            "unique_expert_count_mean",
            "count",
            "std",
            "min",
            "max",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_averages_across_segment_sizes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            self._write_csv(
                csv_path,
                [
                    {"start_token_n_1based": -1, "segment_size_c": 2, "layer_idx": 0,
                     "unique_expert_count_mean": 10.0, "count": 100, "std": 1.0, "min": 8, "max": 16},
                    {"start_token_n_1based": -1, "segment_size_c": 4, "layer_idx": 0,
                     "unique_expert_count_mean": 20.0, "count": 100, "std": 1.0, "min": 8, "max": 16},
                    {"start_token_n_1based": -1, "segment_size_c": 2, "layer_idx": 1,
                     "unique_expert_count_mean": 8.0, "count": 100, "std": 1.0, "min": 8, "max": 16},
                    {"start_token_n_1based": -1, "segment_size_c": 4, "layer_idx": 1,
                     "unique_expert_count_mean": 16.0, "count": 100, "std": 1.0, "min": 8, "max": 16},
                ],
            )
            demand = compute_layer_demand_from_csv(str(csv_path))
            assert demand.shape == (2,)
            assert abs(demand[0].item() - 15.0) < 1e-6
            assert abs(demand[1].item() - 12.0) < 1e-6

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            compute_layer_demand_from_csv("/nonexistent/path.csv")


class TestAllocateSlots:
    def test_uniform_demand_uniform_result(self):
        demand = torch.ones(10)
        slots = allocate_slots_per_layer(demand, total_budget=100, num_experts=64, num_buckets=4)
        assert sum(slots) == 100
        assert len(set(slots)) == 1
        assert slots[0] == 10

    def test_total_budget_preserved(self):
        demand = torch.tensor([3.0, 1.0, 2.0, 5.0, 4.0, 1.5, 2.5, 3.5])
        slots = allocate_slots_per_layer(demand, total_budget=80, num_experts=64, num_buckets=4)
        assert sum(slots) == 80

    def test_clamped_to_num_experts(self):
        demand = torch.tensor([100.0, 1.0])
        slots = allocate_slots_per_layer(demand, total_budget=100, num_experts=32, num_buckets=4)
        assert all(s <= 32 for s in slots)
        assert all(s >= 1 for s in slots)

    def test_min_slots_respected(self):
        demand = torch.tensor([0.01, 100.0])
        slots = allocate_slots_per_layer(
            demand, total_budget=20, num_experts=64, num_buckets=4, min_slots=2,
        )
        assert all(s >= 2 for s in slots)

    def test_higher_demand_gets_more(self):
        demand = torch.tensor([10.0, 5.0, 5.0, 5.0])
        slots = allocate_slots_per_layer(demand, total_budget=100, num_experts=64, num_buckets=4)
        assert slots[0] >= slots[1]

    def test_bucket_count_limits_distinct_values(self):
        demand = torch.arange(1.0, 49.0)
        slots = allocate_slots_per_layer(demand, total_budget=48 * 20, num_experts=64, num_buckets=3)
        distinct = len(set(slots))
        # Bucketing targets num_buckets centers; residual redistribution may add
        # ±1 variants, so allow up to 2*num_buckets distinct values.
        assert distinct <= 6

    def test_max_bucket_ratio_enforced(self):
        demand = torch.tensor([1.0, 10.0, 1.0, 10.0, 1.0, 10.0])
        slots = allocate_slots_per_layer(
            demand, total_budget=120, num_experts=64, num_buckets=4, max_bucket_ratio=1.5,
        )
        assert max(slots) / min(slots) <= 1.5 + 0.1

    def test_empty_demand(self):
        demand = torch.empty(0)
        slots = allocate_slots_per_layer(demand, total_budget=0, num_experts=64, num_buckets=4)
        assert slots == []

    def test_single_layer(self):
        demand = torch.tensor([5.0])
        slots = allocate_slots_per_layer(demand, total_budget=20, num_experts=64, num_buckets=4)
        assert slots == [20]

    def test_realistic_48_layers(self):
        torch.manual_seed(42)
        demand = torch.rand(48) * 30 + 15
        total_budget = 48 * 20
        slots = allocate_slots_per_layer(
            demand, total_budget=total_budget, num_experts=64, num_buckets=4,
            max_bucket_ratio=2.0,
        )
        assert len(slots) == 48
        assert sum(slots) == total_budget
        assert all(1 <= s <= 64 for s in slots)
        distinct = len(set(slots))
        # Bucketing targets num_buckets centers; residual redistribution may
        # add ±1 variants, so allow up to 2*num_buckets distinct values.
        assert distinct <= 8
        assert max(slots) / min(slots) <= 2.0 + 0.15
