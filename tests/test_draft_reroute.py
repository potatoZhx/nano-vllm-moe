import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from nanovllm.expert.cache import LayerExpertCache
from nanovllm.expert.placement import build_cached_draft_plan_gpu
from nanovllm.scheduling.draft_reroute import (
    BOUNDED_CACHE_BIAS,
    DROP_MISS,
    ENTROPY_CACHE_BIAS,
    PUBLIC_DRAFT_REROUTE_POLICIES,
    ROUND_ROBIN,
    SIMILARITY_REPLACE,
    SOURCE_V2_EQUIVALENT_POLICIES,
    SOURCE_V2_TO_PUBLIC_POLICY,
    DraftReroutePolicy,
    load_draft_reroute_artifact,
)


class TestDraftReroutePolicy(unittest.TestCase):
    def _cache(self) -> LayerExpertCache:
        cache = LayerExpertCache(
            num_experts=8,
            slots_per_layer=3,
            gate_up_shape=(4, 4),
            down_shape=(4, 2),
            device=torch.device("cpu"),
            dtype=torch.float32,
            cpu_expert_pool={},
        )
        weight = torch.zeros(4, 4)
        down = torch.zeros(4, 2)
        for slot, expert in enumerate([0, 1, 2]):
            cache.put_to_slot(slot, expert, weight, down)
        return cache

    def _inputs(self):
        selected = torch.tensor([[0, 4, 2, 5]], dtype=torch.int64)
        weights = torch.tensor([[0.4, 0.3, 0.2, 0.1]], dtype=torch.float32)
        logits = torch.tensor([[0.4, -1.5, 0.2, -4.0, 0.3, 0.1, -3.0, -2.0]])
        probs = torch.softmax(logits, dim=-1)
        return logits, probs, selected, weights

    def _policy(self, name: str, **kwargs) -> DraftReroutePolicy:
        cache = self._cache()
        return DraftReroutePolicy(
            policy=name,
            num_experts=8,
            top_k=4,
            cached_expert_mask=cache.get_cached_expert_mask(),
            slot_to_expert_lut=cache.get_slot_to_expert_lut(),
            **kwargs,
        )

    def _aggregate(self, ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        dense = torch.zeros(ids.shape[0], 8, dtype=weights.dtype)
        dense.scatter_add_(1, ids, weights)
        return dense

    def _finalize_reference(self, ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        ids = ids.clone()
        weights = weights.clone()
        empty = weights.sum(-1) == 0
        ids[empty, 0] = 0
        weights[empty, 0] = 1.0
        weights /= weights.sum(-1, keepdim=True).clamp_min(1e-9)
        return self._aggregate(ids, weights)

    def _v2_reference(
        self,
        policy: str,
        logits: torch.Tensor,
        selected: torch.Tensor,
        weights: torch.Tensor,
        *,
        cond_sim: torch.Tensor | None = None,
        skip_err: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cache_mask = self._cache().get_cached_expert_mask()
        hit = cache_mask[selected]
        miss_rate = (~hit).float().mean(-1)
        ids = selected.clone()

        if policy == DROP_MISS:
            return self._finalize_reference(ids, weights * hit.float())

        if policy in (ENTROPY_CACHE_BIAS, BOUNDED_CACHE_BIAS):
            gate = ((miss_rate - 0.25) / (0.50 - 0.25)).clamp(0.0, 1.0)
            probs = torch.softmax(logits.float(), dim=-1)
            entropy = -(probs * (probs + 1e-9).log()).sum(-1)
            if policy == ENTROPY_CACHE_BIAS:
                ent_scale = ((entropy - math.log(8) * 0.25) / (math.log(8) * 0.50)).clamp(0.0, 1.0)
                gamma = 4.0 * (0.2 + 0.8 * ent_scale) * gate
                biased = logits.float() + gamma.unsqueeze(-1) * cache_mask.float().unsqueeze(0)
                ids = biased.topk(4, dim=-1).indices
                for row in range(ids.shape[0]):
                    top1 = int(selected[row, 0])
                    if not bool(cache_mask[top1]) and top1 not in ids[row].tolist():
                        ids[row, -1] = top1
            else:
                gamma = 4.0 * gate * (entropy / math.log(8)).clamp(0.0, 1.0)
                pool_ids = logits.topk(8, dim=-1).indices
                pool = torch.zeros_like(logits, dtype=torch.bool)
                pool.scatter_(1, pool_ids, True)
                biased = logits.float() + gamma.unsqueeze(-1) * (pool & cache_mask.unsqueeze(0)).float()
                ids = biased.topk(4, dim=-1).indices
                for row in range(ids.shape[0]):
                    if float(gamma[row]) < 1e-6:
                        ids[row] = selected[row]
                        continue
                    candidate_set = set(ids[row].tolist())
                    displaced = sum(
                        float(weights[row, route])
                        for route, expert in enumerate(selected[row].tolist())
                        if expert not in candidate_set
                    )
                    if displaced > 0.20:
                        ids[row] = selected[row]
            new_weights = torch.softmax(logits.gather(-1, ids).float(), dim=-1)
            return self._finalize_reference(ids, new_weights * cache_mask[ids].float())

        assert policy == SIMILARITY_REPLACE
        assert cond_sim is not None and skip_err is not None
        masked_sim = cond_sim.clone()
        masked_sim.masked_fill_(~cache_mask.unsqueeze(0), -1e9)
        masked_sim.fill_diagonal_(-1e9)
        best_sim, best_sub = masked_sim.max(-1)
        ids = selected.clone()
        out_weights = weights.clone()
        contrib = weights * skip_err[selected]
        low_contrib = contrib < 0.10 * contrib.mean(-1, keepdim=True)
        for row in range(ids.shape[0]):
            topk_set = set(selected[row].tolist())
            gate = max(0.0, float((miss_rate[row] - 0.25) / (1.0 - 0.25)))
            for route, expert in enumerate(selected[row].tolist()):
                if bool(hit[row, route]):
                    continue
                sub = int(best_sub[expert])
                if (
                    gate < 1e-6
                    or bool(low_contrib[row, route])
                    or sub in topk_set
                    or float(best_sim[expert]) < 0.40
                ):
                    out_weights[row, route] = 0.0
                else:
                    ids[row, route] = sub
                    out_weights[row, route] = weights[row, route] * best_sim[expert] * gate
        return self._finalize_reference(ids, out_weights)

    def test_public_names_replace_experiment_labels(self):
        self.assertEqual(
            PUBLIC_DRAFT_REROUTE_POLICIES,
            (
                ROUND_ROBIN,
                DROP_MISS,
                ENTROPY_CACHE_BIAS,
                BOUNDED_CACHE_BIAS,
                SIMILARITY_REPLACE,
            ),
        )
        self.assertNotIn("Alg2_PostSub", PUBLIC_DRAFT_REROUTE_POLICIES)
        self.assertEqual(
            SOURCE_V2_TO_PUBLIC_POLICY,
            {
                "SkipAll": DROP_MISS,
                "Alg2_v2": ENTROPY_CACHE_BIAS,
                "HybridCP_v2": BOUNDED_CACHE_BIAS,
                "PostSub_v2": SIMILARITY_REPLACE,
            },
        )
        self.assertEqual(SOURCE_V2_EQUIVALENT_POLICIES["Alg2_PostSub"], ENTROPY_CACHE_BIAS)

    def test_drop_miss_zeros_uncached_routes_and_maps_them_to_fixed_fallback(self):
        logits, probs, selected, weights = self._inputs()
        out_ids, out_weights = self._policy(DROP_MISS)(
            logits, probs, selected, weights, torch.float32
        )

        self.assertEqual(out_ids.tolist(), [[0, 0, 2, 0]])
        self.assertTrue(torch.allclose(out_weights, torch.tensor([[2 / 3, 0.0, 1 / 3, 0.0]])))

    def test_entropy_cache_bias_outputs_only_cached_nonzero_routes(self):
        logits = torch.tensor([[0.0, -0.1, -0.2, 1.2, 1.1, 1.0, 0.9, 0.8]])
        probs = torch.softmax(logits, dim=-1)
        weights, selected = torch.topk(probs, 4, dim=-1)
        out_ids, out_weights = self._policy(ENTROPY_CACHE_BIAS)(
            logits, probs, selected, weights, torch.float32
        )
        cache_mask = self._cache().get_cached_expert_mask()

        self.assertEqual(out_ids.shape, selected.shape)
        self.assertTrue(cache_mask.index_select(0, out_ids.reshape(-1)).all().item())
        self.assertTrue(torch.allclose(out_weights.sum(-1), torch.ones(1)))

    def test_bounded_cache_bias_keeps_output_cache_valid(self):
        logits = torch.tensor([[0.0, -0.1, -0.2, 1.2, 1.1, 1.0, 0.9, 0.8]])
        probs = torch.softmax(logits, dim=-1)
        weights, selected = torch.topk(probs, 4, dim=-1)
        out_ids, out_weights = self._policy(BOUNDED_CACHE_BIAS)(
            logits, probs, selected, weights, torch.float32
        )
        cache_mask = self._cache().get_cached_expert_mask()

        self.assertTrue(cache_mask.index_select(0, out_ids.reshape(-1)).all().item())
        self.assertTrue(torch.allclose(out_weights.sum(-1), torch.ones(1)))

    def test_similarity_replace_uses_v2_gate_floor_and_scaled_weight(self):
        logits, probs, selected, weights = self._inputs()
        selected = torch.tensor([[0, 4, 5, 6]], dtype=torch.int64)
        weights = torch.tensor([[0.4, 0.3, 0.2, 0.1]], dtype=torch.float32)
        cond_sim = torch.zeros(8, 8, dtype=torch.float32)
        cond_sim[4, 1] = 0.8
        cond_sim[5, 2] = 0.3
        cond_sim[6, 2] = 0.2
        skip_err = torch.ones(8, dtype=torch.float32)

        out_ids, out_weights = self._policy(
            SIMILARITY_REPLACE,
            cond_sim=cond_sim,
            skip_err=skip_err,
        )(logits, probs, selected, weights, torch.float32)

        expected = torch.tensor([[0.4 / 0.56, 0.16 / 0.56, 0.0, 0.0]])
        self.assertEqual(out_ids.tolist(), [[0, 1, 0, 0]])
        self.assertTrue(torch.allclose(out_weights, expected, atol=1e-6))

    def test_cached_direct_plan_does_not_build_round_robin_substitution(self):
        cache = self._cache()
        selected = torch.tensor([[0, 1, 2, 0]], dtype=torch.int64)
        weights = torch.tensor([[0.5, 0.25, 0.2, 0.05]], dtype=torch.float32)
        plan = build_cached_draft_plan_gpu(
            layer_idx=0,
            selected_experts=selected,
            routing_weights=weights,
            expert_cache=cache,
        )

        self.assertIsNone(plan.substitution_lut)
        self.assertTrue(torch.equal(plan.flat_selected_effective, selected.reshape(-1)))
        self.assertIsNone(plan.cpu_route_indices)
        self.assertEqual(plan.gpu_route_indices.numel(), selected.numel())

    def test_similarity_artifact_loader_validates_and_normalizes_tensors(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "reroute.pt"
            torch.save(
                {
                    "cond_sim": torch.ones(2, 8, 8, dtype=torch.float16),
                    "skip_err": torch.ones(2, 8, dtype=torch.float16),
                },
                path,
            )
            loaded = load_draft_reroute_artifact(str(path), num_experts=8)

        self.assertEqual(loaded["cond_sim"].dtype, torch.float32)
        self.assertEqual(loaded["skip_err"].dtype, torch.float32)
        self.assertTrue(loaded["cond_sim"].is_contiguous())

    def test_vectorized_public_policies_match_v2_weight_semantics(self):
        logits = torch.tensor(
            [
                [1.0, 0.1, -0.2, 2.0, 1.8, 1.7, 1.6, 1.5],
                [2.0, 1.9, 1.8, 1.7, 1.6, 0.1, -0.1, -0.2],
                [-1.0, -1.1, -1.2, 2.4, 2.3, 2.2, 2.1, 2.0],
            ],
            dtype=torch.float32,
        )
        probs = torch.softmax(logits, dim=-1)
        weights, selected = probs.topk(4, dim=-1)
        cond_sim = torch.tensor(
            [
                [1.0, 0.2, 0.1, 0.6, 0.5, 0.1, 0.2, 0.3],
                [0.2, 1.0, 0.1, 0.2, 0.3, 0.4, 0.1, 0.1],
                [0.3, 0.2, 1.0, 0.1, 0.1, 0.3, 0.4, 0.2],
                [0.8, 0.2, 0.1, 1.0, 0.2, 0.3, 0.2, 0.1],
                [0.1, 0.75, 0.3, 0.2, 1.0, 0.2, 0.1, 0.2],
                [0.2, 0.1, 0.70, 0.1, 0.2, 1.0, 0.1, 0.2],
                [0.6, 0.1, 0.2, 0.1, 0.2, 0.3, 1.0, 0.2],
                [0.1, 0.65, 0.2, 0.3, 0.2, 0.1, 0.2, 1.0],
            ],
            dtype=torch.float32,
        )
        skip_err = torch.tensor([1.0, 0.8, 0.9, 1.1, 0.7, 1.2, 0.6, 1.0])

        for policy in (DROP_MISS, ENTROPY_CACHE_BIAS, BOUNDED_CACHE_BIAS, SIMILARITY_REPLACE):
            kwargs = {"cond_sim": cond_sim, "skip_err": skip_err} if policy == SIMILARITY_REPLACE else {}
            ids, actual_weights = self._policy(policy, **kwargs)(
                logits, probs, selected, weights, torch.float32
            )
            actual = self._aggregate(ids, actual_weights)
            expected = self._v2_reference(
                policy,
                logits,
                selected,
                weights,
                cond_sim=cond_sim,
                skip_err=skip_err,
            )
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6), policy)


if __name__ == "__main__":
    unittest.main()
