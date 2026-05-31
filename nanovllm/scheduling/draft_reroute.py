from __future__ import annotations

import math
from functools import partial

import torch
from torch import nn

from nanovllm.scheduling.draft_reroute_profile import load_draft_reroute_profile


ROUND_ROBIN = "round_robin"
DROP_MISS = "drop_miss"
ENTROPY_CACHE_BIAS = "entropy_cache_bias"
BOUNDED_CACHE_BIAS = "bounded_cache_bias"
SIMILARITY_REPLACE = "similarity_replace"

PUBLIC_DRAFT_REROUTE_POLICIES = (
    ROUND_ROBIN,
    DROP_MISS,
    ENTROPY_CACHE_BIAS,
    BOUNDED_CACHE_BIAS,
    SIMILARITY_REPLACE,
)
SOURCE_V2_TO_PUBLIC_POLICY = {
    "SkipAll": DROP_MISS,
    "Alg2_v2": ENTROPY_CACHE_BIAS,
    "HybridCP_v2": BOUNDED_CACHE_BIAS,
    "PostSub_v2": SIMILARITY_REPLACE,
}
# Alg2_v2 sets every residual miss weight to zero before PostSub_v2 runs.
# Therefore its composite output is exactly Alg2_v2 after final normalization.
SOURCE_V2_EQUIVALENT_POLICIES = {"Alg2_PostSub": ENTROPY_CACHE_BIAS}

MISS_GATE = 0.25
SIM_FLOOR = 0.40
GAMMA0 = 4.0
J_FACTOR = 3


def load_draft_reroute_artifact(
    path: str,
    *,
    num_experts: int,
) -> dict[str, torch.Tensor]:
    """Load the offline v2 tensors required by similarity_replace."""
    return load_draft_reroute_profile(path, num_experts=num_experts).similarity_artifact()


class DraftReroutePolicy(nn.Module):
    """Fixed-shape top_c=0 draft rerouting operations suitable for CUDA Graph."""

    def __init__(
        self,
        *,
        policy: str,
        num_experts: int,
        top_k: int,
        cached_expert_mask: torch.Tensor,
        slot_to_expert_lut: torch.Tensor,
        cond_sim: torch.Tensor | None = None,
        skip_err: torch.Tensor | None = None,
        gamma0: float = GAMMA0,
        miss_gate: float = MISS_GATE,
        sim_floor: float = SIM_FLOOR,
        j_factor: int = J_FACTOR,
        tau_dev: float = 0.20,
        contrib_thresh: float = 0.10,
    ) -> None:
        super().__init__()
        if policy not in PUBLIC_DRAFT_REROUTE_POLICIES or policy == ROUND_ROBIN:
            raise ValueError(f"unsupported active draft reroute policy: {policy}")
        if policy == SIMILARITY_REPLACE and (cond_sim is None or skip_err is None):
            raise ValueError("similarity_replace requires cond_sim and skip_err calibration tensors")
        self.policy = policy
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.gamma0 = float(gamma0)
        self.miss_gate = float(miss_gate)
        self.sim_floor = float(sim_floor)
        self.j_factor = int(j_factor)
        self.tau_dev = float(tau_dev)
        self.contrib_thresh = float(contrib_thresh)
        self.register_buffer("cached_expert_mask", cached_expert_mask, persistent=False)
        self.register_buffer("slot_to_expert_lut", slot_to_expert_lut, persistent=False)
        route_ids = torch.arange(self.top_k, device=cached_expert_mask.device).view(1, -1)
        self.register_buffer("first_route_mask", route_ids.eq(0), persistent=False)
        self.register_buffer("last_route_mask", route_ids.eq(self.top_k - 1), persistent=False)
        self.register_buffer(
            "cond_sim",
            None if cond_sim is None else cond_sim.float().to(cached_expert_mask.device).contiguous(),
            persistent=False,
        )
        self.register_buffer(
            "skip_err",
            None if skip_err is None else skip_err.float().to(cached_expert_mask.device).contiguous(),
            persistent=False,
        )

    def _hit_mask(self, selected_experts: torch.Tensor) -> torch.Tensor:
        return self.cached_expert_mask.index_select(0, selected_experts.reshape(-1)).view_as(selected_experts)

    def _fallback_expert(self) -> torch.Tensor:
        sentinel = torch.full_like(self.slot_to_expert_lut, self.num_experts)
        return torch.where(self.slot_to_expert_lut.ge(0), self.slot_to_expert_lut, sentinel).amin().clamp_max(
            self.num_experts - 1
        )

    def _finalize(
        self,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fallback = self._fallback_expert()
        positive = routing_weights.gt(0)
        final_ids = torch.where(positive, selected_experts, fallback)
        row_sum = routing_weights.sum(dim=-1, keepdim=True)
        empty = row_sum.le(0)
        inject = empty & self.first_route_mask
        final_weights = torch.where(inject, torch.ones_like(routing_weights), routing_weights)
        normalizer = torch.where(empty, torch.ones_like(row_sum), row_sum).clamp_min(1e-9)
        final_weights = final_weights / normalizer
        return final_ids.to(torch.int64), final_weights.to(output_dtype)

    def _miss_gate(self, hit_mask: torch.Tensor, upper: float = 0.50) -> torch.Tensor:
        miss_rate = (~hit_mask).float().mean(dim=-1)
        return ((miss_rate - self.miss_gate) / max(upper - self.miss_gate, 1e-6)).clamp(0.0, 1.0)

    def _entropy_cache_bias(
        self,
        router_logits: torch.Tensor,
        router_probs: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hit_mask = self._hit_mask(selected_experts)
        gate = self._miss_gate(hit_mask)
        probs = router_probs.float()
        entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1)
        tau_low = math.log(self.num_experts) * 0.25
        tau_high = math.log(self.num_experts) * 0.75
        entropy_scale = ((entropy - tau_low) / max(tau_high - tau_low, 1e-6)).clamp(0.0, 1.0)
        gamma = self.gamma0 * (0.2 + 0.8 * entropy_scale) * gate
        biased_logits = router_logits.float() + gamma.unsqueeze(-1) * self.cached_expert_mask.unsqueeze(0)
        rerouted = torch.topk(biased_logits, self.top_k, dim=-1).indices

        original_top1 = selected_experts[:, :1]
        original_top1_missed = ~hit_mask[:, :1]
        displaced = ~rerouted.eq(original_top1).any(dim=-1, keepdim=True)
        protect = original_top1_missed & displaced & self.last_route_mask
        rerouted = torch.where(protect, original_top1.expand_as(rerouted), rerouted)

        # Finalization renormalizes retained routes, so gathered full-router
        # probabilities are equivalent to a second softmax over selected logits.
        weights = router_probs.float().gather(-1, rerouted)
        weights = weights * self._hit_mask(rerouted)
        return rerouted, weights

    def _bounded_cache_bias(
        self,
        router_logits: torch.Tensor,
        router_probs: torch.Tensor,
        selected_experts: torch.Tensor,
        selected_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hit_mask = self._hit_mask(selected_experts)
        gate = self._miss_gate(hit_mask)
        probs = router_probs.float()
        entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1)
        gamma = self.gamma0 * gate * (entropy / math.log(self.num_experts)).clamp(0.0, 1.0)
        pool_size = min(self.top_k * self.j_factor, self.num_experts)
        pool_ids = torch.topk(router_logits, pool_size, dim=-1).indices
        pool = torch.zeros_like(router_logits, dtype=torch.bool)
        pool.scatter_(1, pool_ids, True)
        bias_mask = pool & self.cached_expert_mask.unsqueeze(0)
        biased_logits = router_logits.float() + gamma.unsqueeze(-1) * bias_mask
        candidate = torch.topk(biased_logits, self.top_k, dim=-1).indices

        retained = selected_experts.unsqueeze(-1).eq(candidate.unsqueeze(-2)).any(dim=-1)
        displaced_weight = torch.where(
            retained,
            torch.zeros_like(selected_weights.float()),
            selected_weights.float(),
        ).sum(dim=-1)
        revert = gamma.le(1e-6) | displaced_weight.gt(self.tau_dev)
        rerouted = torch.where(revert.unsqueeze(-1), selected_experts, candidate)
        weights = router_probs.float().gather(-1, rerouted)
        weights = weights * self._hit_mask(rerouted)
        return rerouted, weights

    def _similarity_replace(
        self,
        selected_experts: torch.Tensor,
        selected_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.cond_sim is not None and self.skip_err is not None
        hit_mask = self._hit_mask(selected_experts)
        miss_rate = (~hit_mask).float().mean(dim=-1)
        gate = ((miss_rate - self.miss_gate) / max(1.0 - self.miss_gate, 1e-6)).clamp(0.0, 1.0)

        sim = self.cond_sim.index_select(0, selected_experts.reshape(-1)).view(
            selected_experts.shape[0],
            selected_experts.shape[1],
            self.num_experts,
        )
        sim = sim.masked_fill(~self.cached_expert_mask.view(1, 1, -1), -1e9)
        best_sim, best_sub = sim.max(dim=-1)
        already_selected = best_sub.unsqueeze(-1).eq(selected_experts.unsqueeze(1)).any(dim=-1)
        onorm = self.skip_err.index_select(0, selected_experts.reshape(-1)).view_as(selected_weights)
        contrib = selected_weights.float() * onorm
        low_contrib = contrib < self.contrib_thresh * contrib.mean(dim=-1, keepdim=True)
        replace = (
            (~hit_mask)
            & gate.unsqueeze(-1).gt(0)
            & (~low_contrib)
            & best_sim.ge(self.sim_floor)
            & (~already_selected)
        )
        rerouted = torch.where(replace, best_sub, selected_experts)
        substitute_weights = selected_weights.float() * best_sim * gate.unsqueeze(-1)
        weights = torch.where(
            hit_mask,
            selected_weights.float(),
            torch.where(replace, substitute_weights, torch.zeros_like(substitute_weights)),
        )
        return rerouted, weights

    @partial(torch.compile, fullgraph=True, mode="max-autotune-no-cudagraphs")
    def forward(
        self,
        router_logits: torch.Tensor,
        router_probs: torch.Tensor,
        selected_experts: torch.Tensor,
        selected_weights: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.policy == DROP_MISS:
            rerouted = selected_experts
            weights = selected_weights.float() * self._hit_mask(selected_experts)
        elif self.policy == ENTROPY_CACHE_BIAS:
            rerouted, weights = self._entropy_cache_bias(router_logits, router_probs, selected_experts)
        elif self.policy == BOUNDED_CACHE_BIAS:
            rerouted, weights = self._bounded_cache_bias(
                router_logits, router_probs, selected_experts, selected_weights
            )
        else:
            rerouted, weights = self._similarity_replace(selected_experts, selected_weights)
        return self._finalize(rerouted, weights, output_dtype)
