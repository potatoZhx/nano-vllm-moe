"""Qwen/Qwen3 MoE ExpertSubsetInference with random_cache mode.

This module is intentionally focused on Qwen/Qwen3 MoE blocks. It wraps every
`layer.mlp` that has `gate` and `experts`, records prefill expert activations in
standard mode, builds a per-layer expert cache by LFU/LRU, and during decode
replaces out-of-cache routed experts with random experts from the highest-scored
cached candidates.

Deployment note:
  The wrapper stores routing metadata as GPU tensors. Only the offline collector
  converts metadata to CPU lists for JSONL serialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RandomCacheConfig:
    cache_ratio: float = 0.5
    cache_policy: str = "lfu"  # "lfu" or "lru"
    cache_topc_ratio: float = 0.5
    collect_prefill: bool = False


def _as_float_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


class QwenRandomCacheMoeWrapper(nn.Module):
    """Qwen/Qwen3 MoE wrapper supporting standard and random_cache modes."""

    def __init__(
        self,
        original_block: nn.Module,
        layer_idx: int,
        mode: str = "standard",
        cache_ratio: float = 0.5,
        cache_policy: str = "lfu",
        cache_topc_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        self.original_block = original_block
        self.layer_idx = layer_idx
        self.mode = mode
        self.random_cache = RandomCacheConfig(
            cache_ratio=cache_ratio,
            cache_policy=cache_policy,
            cache_topc_ratio=cache_topc_ratio,
            collect_prefill=False,
        )

        self.config = getattr(original_block, "config", None)
        self.gate = original_block.gate
        self.experts = original_block.experts
        self.shared_expert = getattr(original_block, "shared_expert", None)
        self.shared_expert_gate = getattr(original_block, "shared_expert_gate", None)

        self.num_experts = getattr(original_block, "num_experts", None)
        if self.num_experts is None and self.config is not None:
            self.num_experts = getattr(self.config, "num_experts", None)
        if self.num_experts is None and self.config is not None:
            self.num_experts = getattr(self.config, "n_routed_experts", None)
        if self.num_experts is None:
            self.num_experts = len(self.experts)

        self.top_k = getattr(original_block, "top_k", None)
        if self.top_k is None and self.config is not None:
            self.top_k = getattr(self.config, "num_experts_per_tok", None)
        if self.top_k is None and self.config is not None:
            self.top_k = getattr(self.config, "num_experts_per_token", None)
        if self.top_k is None:
            # Qwen3-30B-A3B commonly uses 8 routed experts per token.
            self.top_k = 8

        dev = self._infer_device()
        self.register_buffer("prefill_counts", torch.zeros(self.num_experts, dtype=torch.float32, device=dev), persistent=False)
        self.register_buffer("prefill_last_pos", torch.full((self.num_experts,), -1.0, dtype=torch.float32, device=dev), persistent=False)
        self.register_buffer("cache_ids", torch.empty(0, dtype=torch.long, device=dev), persistent=False)
        self.cache_built = False

        self.last_metadata: Dict[str, torch.Tensor | int | float | str] = {}

    def _infer_device(self) -> torch.device:
        for p in self.original_block.parameters(recurse=True):
            return p.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reset_prefill_stats(self) -> None:
        self.prefill_counts.zero_()
        self.prefill_last_pos.fill_(-1.0)
        self.cache_ids = torch.empty(0, dtype=torch.long, device=self.prefill_counts.device)
        self.cache_built = False
        self.last_metadata = {}

    def set_prefill_collection(self, enabled: bool) -> None:
        self.random_cache.collect_prefill = enabled

    def configure_random_cache(self, cache_ratio: float, cache_policy: str, cache_topc_ratio: float) -> None:
        if not (0.0 < cache_ratio <= 1.0):
            raise ValueError(f"cache_ratio must be in (0,1], got {cache_ratio}")
        if not (0.0 < cache_topc_ratio <= 1.0):
            raise ValueError(f"cache_topc_ratio must be in (0,1], got {cache_topc_ratio}")
        if cache_policy not in {"lfu", "lru"}:
            raise ValueError(f"cache_policy must be 'lfu' or 'lru', got {cache_policy}")
        self.random_cache.cache_ratio = float(cache_ratio)
        self.random_cache.cache_policy = cache_policy
        self.random_cache.cache_topc_ratio = float(cache_topc_ratio)

    @torch.no_grad()
    def build_expert_cache(self) -> torch.Tensor:
        cache_size = max(1, int(round(self.num_experts * self.random_cache.cache_ratio)))
        cache_size = min(cache_size, self.num_experts)

        if self.random_cache.cache_policy == "lfu":
            score = self.prefill_counts
        elif self.random_cache.cache_policy == "lru":
            # The first prefill token has the smallest position and is least recent.
            score = self.prefill_last_pos
        else:
            raise ValueError(f"Unsupported cache policy: {self.random_cache.cache_policy}")

        # Stable fallback: if many experts are never activated, argsort still fills
        # remaining cache slots deterministically by index order among equal scores.
        self.cache_ids = torch.argsort(score, descending=True)[:cache_size].to(torch.long)
        self.cache_built = True
        return self.cache_ids

    @torch.no_grad()
    def _update_prefill_stats(self, topk_indices: torch.Tensor, batch_size: int, seq_len: int) -> None:
        # topk_indices: [batch_size * seq_len, top_k]
        idx = topk_indices.reshape(-1).to(self.prefill_counts.device)
        ones = torch.ones_like(idx, dtype=torch.float32, device=self.prefill_counts.device)
        self.prefill_counts.scatter_add_(0, idx, ones)

        token_pos = torch.arange(seq_len, device=self.prefill_counts.device, dtype=torch.float32).repeat(batch_size)
        pos_values = token_pos[:, None].expand(batch_size * seq_len, self.top_k).reshape(-1)
        # PyTorch >= 1.12 supports scatter_reduce_. This repository targets torch 2.7.
        self.prefill_last_pos.scatter_reduce_(0, idx, pos_values, reduce="amax", include_self=True)

    def _topk_route(self, router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        topk_logits, topk_indices = torch.topk(router_logits, self.top_k, dim=-1)
        weights = F.softmax(_as_float_tensor(topk_logits), dim=-1).to(router_logits.dtype)
        return topk_indices, weights

    def _random_cache_select(self, router_logits: torch.Tensor, original_indices: torch.Tensor) -> torch.Tensor:
        if not self.cache_built or self.cache_ids.numel() == 0:
            self.build_expert_cache()

        n_tokens = original_indices.size(0)
        new_indices = original_indices.clone()
        cache_ids = self.cache_ids.to(router_logits.device)
        cache_size = cache_ids.numel()
        candidate_size = max(1, int(round(cache_size * self.random_cache.cache_topc_ratio)))
        candidate_size = min(candidate_size, cache_size)

        in_cache = torch.zeros(self.num_experts, dtype=torch.bool, device=router_logits.device)
        in_cache[cache_ids] = True

        for t in range(n_tokens):
            used = set()
            # Cache candidates are the top-c scored cached experts for this token.
            cache_scores = router_logits[t, cache_ids]
            cand_local = torch.topk(cache_scores, candidate_size, dim=-1).indices
            candidates = cache_ids[cand_local]

            for j in range(self.top_k):
                orig = int(original_indices[t, j].item())
                if bool(in_cache[orig].item()) and orig not in used:
                    used.add(orig)
                    continue

                eligible: List[int] = [int(x.item()) for x in candidates if int(x.item()) not in used]
                if eligible:
                    pick_idx = int(torch.randint(0, len(eligible), (1,), device=router_logits.device).item())
                    repl = eligible[pick_idx]
                else:
                    # Extremely rare: all top-c cached candidates are already used.
                    # Fall back to the best cached expert, even if it duplicates.
                    repl = int(candidates[0].item())
                new_indices[t, j] = repl
                used.add(repl)
        return new_indices

    def _process_selection(self, router_logits: torch.Tensor, batch_size: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        original_indices, original_weights = self._topk_route(router_logits)
        final_indices = original_indices

        if self.mode == "standard":
            if self.random_cache.collect_prefill:
                self._update_prefill_stats(original_indices.detach(), batch_size=batch_size, seq_len=seq_len)
        elif self.mode == "random_cache":
            final_indices = self._random_cache_select(router_logits, original_indices)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        final_logits = router_logits.gather(1, final_indices)
        final_weights = F.softmax(_as_float_tensor(final_logits), dim=-1).to(router_logits.dtype)
        replacement_mask = final_indices.ne(original_indices)

        self.last_metadata = {
            "layer_idx": self.layer_idx,
            "mode": self.mode,
            "original_ids": original_indices.detach(),
            "original_weights": original_weights.detach(),
            "modified_ids": final_indices.detach(),
            "modified_weights": final_weights.detach(),
            "replacement_mask": replacement_mask.detach(),
            "cache_ids": self.cache_ids.detach() if self.cache_ids.numel() else torch.empty(0, dtype=torch.long, device=router_logits.device),
            "cache_ratio": self.random_cache.cache_ratio,
            "cache_policy": self.random_cache.cache_policy,
            "cache_topc_ratio": self.random_cache.cache_topc_ratio,
        }
        return final_indices, final_weights

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, hidden_dim)
        router_logits = self.gate(hidden_flat)
        if isinstance(router_logits, tuple):
            router_logits = router_logits[0]
        if router_logits.dim() != 2:
            router_logits = router_logits.reshape(-1, self.num_experts)

        final_indices, final_weights = self._process_selection(router_logits, batch_size=batch_size, seq_len=seq_len)

        final_hidden = torch.zeros(
            (batch_size * seq_len, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        expert_mask = F.one_hot(final_indices, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False)

        for expert_idx_tensor in expert_hit:
            expert_idx = int(expert_idx_tensor[0].item())
            if expert_idx >= len(self.experts):
                continue
            mask_slice = expert_mask[expert_idx]
            topk_pos, token_pos = torch.where(mask_slice)
            if token_pos.numel() == 0:
                continue
            expert_out = self.experts[expert_idx](hidden_flat[token_pos])
            weights = final_weights[token_pos, topk_pos].unsqueeze(1)
            final_hidden.index_add_(0, token_pos, (expert_out * weights).to(hidden_states.dtype))

        if self.shared_expert is not None:
            shared_out = self.shared_expert(hidden_flat)
            if self.shared_expert_gate is not None:
                shared_out = torch.sigmoid(self.shared_expert_gate(hidden_flat)) * shared_out
            final_hidden = final_hidden + shared_out.to(final_hidden.dtype)

        return final_hidden.reshape(batch_size, seq_len, hidden_dim)


def apply_random_cache_wrapper_to_qwen(
    model: nn.Module,
    mode: str = "standard",
    cache_ratio: float = 0.5,
    cache_policy: str = "lfu",
    cache_topc_ratio: float = 0.5,
) -> nn.Module:
    """Replace Qwen/Qwen3 MoE blocks with QwenRandomCacheMoeWrapper."""
    count = 0
    for layer_idx, layer in enumerate(model.model.layers):
        if not hasattr(layer, "mlp"):
            continue
        mlp = layer.mlp
        if isinstance(mlp, QwenRandomCacheMoeWrapper):
            mlp.mode = mode
            mlp.configure_random_cache(cache_ratio, cache_policy, cache_topc_ratio)
            count += 1
            continue
        if hasattr(mlp, "gate") and hasattr(mlp, "experts"):
            layer.mlp = QwenRandomCacheMoeWrapper(
                mlp,
                layer_idx=layer_idx,
                mode=mode,
                cache_ratio=cache_ratio,
                cache_policy=cache_policy,
                cache_topc_ratio=cache_topc_ratio,
            )
            count += 1
    print(
        f"Applied QwenRandomCacheMoeWrapper to {count} MoE layers. "
        f"mode={mode}, cache_ratio={cache_ratio}, policy={cache_policy}, topc={cache_topc_ratio}"
    )
    return model


def iter_wrapped_moe_layers(model: nn.Module):
    for layer in model.model.layers:
        if hasattr(layer, "mlp") and isinstance(layer.mlp, QwenRandomCacheMoeWrapper):
            yield layer.mlp


def set_moe_mode(model: nn.Module, mode: str) -> None:
    for mlp in iter_wrapped_moe_layers(model):
        mlp.mode = mode


def reset_prefill_cache_state(model: nn.Module) -> None:
    for mlp in iter_wrapped_moe_layers(model):
        mlp.reset_prefill_stats()


def enable_prefill_collection(model: nn.Module, enabled: bool) -> None:
    for mlp in iter_wrapped_moe_layers(model):
        mlp.set_prefill_collection(enabled)


def build_all_expert_caches(model: nn.Module, cache_ratio: float = 0.5, cache_policy: str = "lfu", cache_topc_ratio: float = 0.5) -> None:
    for mlp in iter_wrapped_moe_layers(model):
        mlp.configure_random_cache(cache_ratio, cache_policy, cache_topc_ratio)
        mlp.build_expert_cache()


def _tensor_to_list(x: torch.Tensor, first_token_only: bool = True):
    if first_token_only and x.dim() >= 2:
        x = x[:1]
    return x.detach().cpu().tolist()


def collect_moe_metadata(model: nn.Module, first_token_only: bool = True) -> List[Dict]:
    """Collect per-layer metadata as CPU lists for offline JSONL storage."""
    metas: List[Dict] = []
    for mlp in iter_wrapped_moe_layers(model):
        m = mlp.last_metadata
        if not m:
            continue
        out = {
            "layer_idx": int(m["layer_idx"]),
            "mode": str(m["mode"]),
            "original_ids": _tensor_to_list(m["original_ids"], first_token_only=first_token_only),
            "original_weights": _tensor_to_list(m["original_weights"], first_token_only=first_token_only),
            "modified_ids": _tensor_to_list(m["modified_ids"], first_token_only=first_token_only),
            "modified_weights": _tensor_to_list(m["modified_weights"], first_token_only=first_token_only),
            "replacement_mask": _tensor_to_list(m["replacement_mask"], first_token_only=first_token_only),
            "cache_size": int(m["cache_ids"].numel()) if isinstance(m["cache_ids"], torch.Tensor) else 0,
            "cache_ratio": float(m["cache_ratio"]),
            "cache_policy": str(m["cache_policy"]),
            "cache_topc_ratio": float(m["cache_topc_ratio"]),
        }
        metas.append(out)
    metas.sort(key=lambda x: x["layer_idx"])
    return metas


@torch.no_grad()
def summarize_cache(model: nn.Module) -> List[Dict]:
    summary = []
    for mlp in iter_wrapped_moe_layers(model):
        summary.append(
            {
                "layer_idx": mlp.layer_idx,
                "cache_size": int(mlp.cache_ids.numel()),
                "cache_policy": mlp.random_cache.cache_policy,
                "cache_ratio": mlp.random_cache.cache_ratio,
                "activated_experts": int((mlp.prefill_counts > 0).sum().item()),
            }
        )
    return summary
