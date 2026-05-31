import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from nanovllm.scheduling.cache_strategy import LFURankGuardStrategy
from nanovllm.scheduling.draft_reroute_profile import (
    DraftRerouteProfile,
    load_draft_reroute_profile,
    save_draft_reroute_profile,
    seed_lfu_rank_guard_from_profile,
)
from nanovllm.utils.heterogeneous_loader import HeterogeneousModelLoader


def _profile_tensors(num_layers: int = 2, num_experts: int = 4) -> dict[str, torch.Tensor]:
    return {
        "cond_sim": torch.ones(num_layers, num_experts, num_experts, dtype=torch.float16),
        "skip_err": torch.ones(num_layers, num_experts, dtype=torch.float16),
        "sim": torch.eye(num_experts, dtype=torch.float32).repeat(num_layers, 1, 1),
        "sens": torch.linspace(0.1, 0.9, num_layers, dtype=torch.float32),
        "act_freq": torch.tensor(
            [
                [0.20, 0.10, 0.60, 0.10],
                [0.05, 0.70, 0.20, 0.05],
            ],
            dtype=torch.float32,
        )[:num_layers, :num_experts],
    }


class TestDraftRerouteProfile(unittest.TestCase):
    def test_full_safetensors_profile_loads_all_tensors_and_metadata(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profile.safetensors"
            save_file(
                _profile_tensors(),
                str(path),
                metadata={
                    "format_version": "2",
                    "num_layers": "2",
                    "num_experts": "4",
                    "top_k": "2",
                    "model_type": "qwen3_moe",
                },
            )

            profile = load_draft_reroute_profile(
                str(path),
                num_experts=4,
                expected_layers=2,
                expected_top_k=2,
            )

        self.assertFalse(profile.is_legacy)
        self.assertEqual(profile.metadata["format_version"], "2")
        self.assertEqual(tuple(profile.cond_sim.shape), (2, 4, 4))
        self.assertEqual(tuple(profile.skip_err.shape), (2, 4))
        self.assertEqual(tuple(profile.sim.shape), (2, 4, 4))
        self.assertEqual(tuple(profile.sens.shape), (2,))
        self.assertEqual(tuple(profile.act_freq.shape), (2, 4))
        self.assertEqual(profile.cond_sim.dtype, torch.float32)
        self.assertTrue(profile.cond_sim.is_contiguous())

    def test_legacy_pt_profile_loads_only_similarity_tensors(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy.pt"
            torch.save(
                {
                    "cond_sim": torch.ones(2, 4, 4, dtype=torch.float16),
                    "skip_err": torch.ones(2, 4, dtype=torch.float16),
                },
                path,
            )

            profile = load_draft_reroute_profile(str(path), num_experts=4)

        self.assertTrue(profile.is_legacy)
        self.assertIsNone(profile.sim)
        self.assertIsNone(profile.sens)
        self.assertIsNone(profile.act_freq)
        self.assertEqual(tuple(profile.cond_sim.shape), (2, 4, 4))
        self.assertEqual(profile.skip_err.dtype, torch.float32)

    def test_full_pt_profile_round_trips_new_schema(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profile.pt"
            save_draft_reroute_profile(
                str(path),
                tensors=_profile_tensors(),
                metadata={
                    "format_version": "2",
                    "num_layers": "2",
                    "num_experts": "4",
                    "top_k": "2",
                },
            )

            profile = load_draft_reroute_profile(
                str(path),
                num_experts=4,
                expected_layers=2,
                expected_top_k=2,
            )

        self.assertFalse(profile.is_legacy)
        self.assertEqual(tuple(profile.act_freq.shape), (2, 4))
        self.assertEqual(tuple(profile.sim.shape), (2, 4, 4))

    def test_profile_loader_rejects_metadata_mismatch(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profile.safetensors"
            save_file(
                _profile_tensors(),
                str(path),
                metadata={
                    "format_version": "2",
                    "num_layers": "2",
                    "num_experts": "5",
                    "top_k": "2",
                },
            )

            with self.assertRaisesRegex(ValueError, "num_experts"):
                load_draft_reroute_profile(str(path), num_experts=4, expected_layers=2)

    def test_initial_placement_uses_act_freq_and_seeds_cache_stats(self):
        profile = DraftRerouteProfile(
            cond_sim=None,
            skip_err=None,
            sim=None,
            sens=None,
            act_freq=torch.tensor(
                [
                    [0.20, 0.10, 0.60, 0.10],
                    [0.05, 0.70, 0.20, 0.05],
                ],
                dtype=torch.float32,
            ),
            metadata={},
            is_legacy=False,
        )
        loader = HeterogeneousModelLoader(
            self._config(slots_per_layer=2),
            draft_reroute_profile=profile,
        )
        cpu_pool = self._cpu_pool(layer_indices=(0, 3), num_experts=4)

        layer_caches = loader._init_layer_caches(cpu_pool)
        loader._load_initial_placement(layer_caches, cpu_pool)

        self.assertEqual(layer_caches[0].slot_to_expert, [2, 0])
        self.assertEqual(layer_caches[3].slot_to_expert, [1, 2])
        self.assertEqual(layer_caches[0].access_count, [200, 100, 600, 100])
        self.assertEqual(layer_caches[3].access_count, [50, 700, 200, 50])
        self.assertAlmostEqual(layer_caches[0].access_score_sum[2], 600.0)
        self.assertEqual(layer_caches[0].last_access_step, [-1, -1, -1, -1])

    def test_initial_placement_without_act_freq_keeps_expert_id_order(self):
        loader = HeterogeneousModelLoader(self._config(slots_per_layer=2))
        cpu_pool = self._cpu_pool(layer_indices=(0,), num_experts=4)

        layer_caches = loader._init_layer_caches(cpu_pool)
        loader._load_initial_placement(layer_caches, cpu_pool)

        self.assertEqual(layer_caches[0].slot_to_expert, [0, 1])
        self.assertEqual(layer_caches[0].access_count, [0, 0, 0, 0])

    def test_lfu_rank_guard_seed_uses_activation_frequency_times_topk(self):
        profile = DraftRerouteProfile(
            cond_sim=None,
            skip_err=None,
            sim=None,
            sens=None,
            act_freq=torch.tensor([[0.10, 0.20, 0.30, 0.40]], dtype=torch.float32),
            metadata={},
            is_legacy=False,
        )
        strategy = LFURankGuardStrategy(num_experts=4)

        seed_lfu_rank_guard_from_profile(strategy, profile, layer_indices=[7], top_k=4)

        self.assertEqual(strategy.get_rank_scores(7), [0.4, 0.8, 1.2, 1.6])

    @staticmethod
    def _config(slots_per_layer: int):
        return SimpleNamespace(
            hf_config=SimpleNamespace(torch_dtype=torch.float32),
            cpu_expert_pin_memory=False,
            heterogeneous_slots_per_layer=slots_per_layer,
            prefetch_staging_slots_per_layer=0,
            spec_enable_prefetch=False,
        )

    @staticmethod
    def _cpu_pool(layer_indices: tuple[int, ...], num_experts: int):
        pool = {}
        for layer_idx in layer_indices:
            pool[layer_idx] = {}
            for expert_idx in range(num_experts):
                pool[layer_idx][expert_idx] = {
                    "gate_up": torch.full((4, 4), float(expert_idx)),
                    "down": torch.full((4, 2), float(expert_idx)),
                }
        return pool


if __name__ == "__main__":
    unittest.main()
