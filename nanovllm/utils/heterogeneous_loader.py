from __future__ import annotations

import os
from glob import glob
from typing import TYPE_CHECKING

import torch
from safetensors import safe_open

from nanovllm.config import Config
from nanovllm.expert.cache import LayerExpertCache
from nanovllm.expert.cpu_weights import CpuExpertWeights
from nanovllm.scheduling.draft_reroute_profile import DraftRerouteProfile
from nanovllm.utils.loader import default_weight_loader

if TYPE_CHECKING:
    from nanovllm.models.qwen3_moe import Qwen3MoeForCausalLM


class HeterogeneousModelLoader:
    """Load non-expert weights to GPU and route expert weights through CPU pool."""

    def __init__(
        self,
        config: Config,
        draft_reroute_profile: DraftRerouteProfile | None = None,
    ):
        self.config = config
        self.hf_config = config.hf_config
        self.pin_memory = config.cpu_expert_pin_memory
        self.draft_reroute_profile = draft_reroute_profile

    def load(
        self,
        model: Qwen3MoeForCausalLM,
        path: str,
    ) -> tuple[dict[int, LayerExpertCache], dict[int, dict[int, dict[str, torch.Tensor]]]]:
        self._load_non_expert_weights(model, path)
        cpu_pool = self._load_expert_weights_to_cpu(path)
        layer_caches = self._init_layer_caches(cpu_pool)
        self._load_initial_placement(layer_caches, cpu_pool)
        torch.cuda.synchronize()
        return layer_caches, cpu_pool

    def _load_non_expert_weights(self, model: Qwen3MoeForCausalLM, path: str) -> None:
        packed_modules_mapping = model.packed_modules_mapping
        for file in glob(os.path.join(path, "*.safetensors")):
            with safe_open(file, "pt", "cpu") as f:
                for orig_name in f.keys():
                    if "mlp.experts" in orig_name:
                        continue
                    weight_tensor = f.get_tensor(orig_name)
                    weight_name = orig_name
                    is_loaded = False

                    for key in packed_modules_mapping:
                        if key in weight_name:
                            param_name, shard_id = packed_modules_mapping[key]
                            param_name = weight_name.replace(key, param_name)
                            param = model.get_parameter(param_name)
                            weight_loader = getattr(param, "weight_loader")
                            weight_loader(param, weight_tensor, shard_id)
                            is_loaded = True
                            break

                    if not is_loaded:
                        param = model.get_parameter(weight_name)
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, weight_tensor)

    def _load_expert_weights_to_cpu(
        self,
        path: str,
    ) -> dict[int, dict[int, dict[str, torch.Tensor]]]:
        # layer_idx -> expert_idx -> {"gate_up": Tensor, "down": Tensor}
        cpu_pool: dict[int, dict[int, dict[str, torch.Tensor]]] = {}
        pending_gate: dict[tuple[int, int], torch.Tensor] = {}
        pending_up: dict[tuple[int, int], torch.Tensor] = {}

        for file in glob(os.path.join(path, "*.safetensors")):
            with safe_open(file, "pt", "cpu") as f:
                for weight_name in f.keys():
                    if "mlp.experts" not in weight_name:
                        continue
                    weight = f.get_tensor(weight_name)
                    layer_idx = self._parse_layer_idx(weight_name)
                    expert_idx = self._parse_expert_idx(weight_name)
                    key = (layer_idx, expert_idx)
                    cpu_pool.setdefault(layer_idx, {}).setdefault(expert_idx, {})

                    if "down_proj" in weight_name:
                        cpu_pool[layer_idx][expert_idx]["down"] = self._to_cpu(
                            weight,
                            dtype=self._target_expert_dtype(),
                        )
                    elif "gate_proj" in weight_name:
                        pending_gate[key] = weight
                    elif "up_proj" in weight_name:
                        pending_up[key] = weight

        for key, gate in pending_gate.items():
            up = pending_up[key]
            gate_up = torch.cat([gate, up], dim=0)
            layer_idx, expert_idx = key
            gate_up_cpu = self._to_cpu(gate_up, dtype=self._target_expert_dtype())
            down_cpu = cpu_pool[layer_idx][expert_idx]["down"]
            packed = CpuExpertWeights(
                expert_idx=expert_idx,
                gate_up=gate_up_cpu,
                down=down_cpu,
                dtype=self._target_expert_dtype(),
            )
            packed.validate()
            cpu_pool[layer_idx][expert_idx]["gate_up"] = gate_up_cpu
            cpu_pool[layer_idx][expert_idx]["packed"] = packed

        return cpu_pool

    def _init_layer_caches(
        self,
        cpu_pool: dict[int, dict[int, dict[str, torch.Tensor]]],
    ) -> dict[int, LayerExpertCache]:
        base_slots = self.config.heterogeneous_slots_per_layer
        allocation_mode = getattr(self.config, "heterogeneous_slot_allocation", "uniform")
        sorted_layers = sorted(cpu_pool.keys())

        per_layer_slots = None
        if allocation_mode == "profile_weighted":
            per_layer_slots = self._compute_profile_weighted_slots(sorted_layers, cpu_pool, base_slots)

        layer_caches: dict[int, LayerExpertCache] = {}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for layer_idx in sorted_layers:
            experts = cpu_pool[layer_idx]
            num_experts = len(experts)
            if per_layer_slots is not None:
                slots = min(per_layer_slots[layer_idx], num_experts)
            else:
                slots = num_experts if base_slots <= 0 else min(base_slots, num_experts)
            sample = next(iter(experts.values()))
            layer_caches[layer_idx] = LayerExpertCache(
                num_experts=num_experts,
                slots_per_layer=slots,
                gate_up_shape=tuple(sample["gate_up"].shape),
                down_shape=tuple(sample["down"].shape),
                device=device,
                dtype=self.hf_config.torch_dtype,
                cpu_expert_pool=experts,
                staging_slots_per_layer=getattr(self.config, "prefetch_staging_slots_per_layer", 0),
                enable_prefetch=bool(getattr(self.config, "spec_enable_prefetch", False)),
            )
        return layer_caches

    def _compute_profile_weighted_slots(
        self,
        sorted_layers: list[int],
        cpu_pool: dict[int, dict[int, dict[str, torch.Tensor]]],
        base_slots_per_layer: int,
    ) -> dict[int, int]:
        from nanovllm.expert.slot_allocation import (
            allocate_slots_per_layer,
            compute_layer_demand_from_act_freq,
            compute_layer_demand_from_csv,
        )

        num_experts = len(next(iter(cpu_pool.values())))
        num_layers = len(sorted_layers)
        effective_base = num_experts if base_slots_per_layer <= 0 else min(base_slots_per_layer, num_experts)
        total_budget = effective_base * num_layers

        csv_path = getattr(self.config, "heterogeneous_slot_profile_csv", "")
        num_buckets = getattr(self.config, "heterogeneous_slot_buckets", 4)
        max_ratio = getattr(self.config, "heterogeneous_slot_max_bucket_ratio", 2.0)

        demand = None
        source = "none"
        if csv_path:
            try:
                demand = compute_layer_demand_from_csv(csv_path)
                source = f"csv({csv_path})"
            except Exception as e:
                print(f"[slot_allocation] WARNING: failed to load CSV: {e}", flush=True)

        if demand is None and self.draft_reroute_profile is not None:
            act_freq = self.draft_reroute_profile.act_freq
            if act_freq is not None:
                demand = compute_layer_demand_from_act_freq(act_freq)
                source = "act_freq"

        if demand is None:
            print(
                "[slot_allocation] WARNING: no profile data available, "
                "falling back to uniform allocation",
                flush=True,
            )
            return {layer_idx: effective_base for layer_idx in sorted_layers}

        if demand.shape[0] < num_layers:
            print(
                f"[slot_allocation] WARNING: demand has {demand.shape[0]} layers "
                f"but model has {num_layers} MoE layers, falling back to uniform",
                flush=True,
            )
            return {layer_idx: effective_base for layer_idx in sorted_layers}

        demand = demand[:num_layers]

        slots_list = allocate_slots_per_layer(
            demand=demand,
            total_budget=total_budget,
            num_experts=num_experts,
            num_buckets=num_buckets,
            max_bucket_ratio=max_ratio,
        )

        result = {}
        for i, layer_idx in enumerate(sorted_layers):
            result[layer_idx] = slots_list[i]

        distinct = sorted(set(slots_list))
        print(
            f"[slot_allocation] profile_weighted (source={source}): "
            f"total_budget={total_budget} buckets_used={len(distinct)} "
            f"bucket_values={distinct} "
            f"min={min(slots_list)} max={max(slots_list)} "
            f"sum={sum(slots_list)}",
            flush=True,
        )
        return result

    def _load_initial_placement(
        self,
        layer_caches: dict[int, LayerExpertCache],
        cpu_pool: dict[int, dict[int, dict[str, torch.Tensor]]],
    ) -> None:
        # First step default: S=N, so this maps expert i -> slot i where possible.
        for profile_row, layer_idx in enumerate(sorted(layer_caches)):
            cache = layer_caches[layer_idx]
            expert_ids = sorted(cpu_pool[layer_idx].keys())
            initial_experts = self._initial_experts_for_layer(profile_row, expert_ids, cache.num_slots)
            self._seed_cache_stats_from_profile(cache, profile_row, expert_ids)
            for slot_idx, expert_idx in enumerate(initial_experts):
                params = cpu_pool[layer_idx][expert_idx]
                cache.put_to_slot(slot_idx, expert_idx, params["gate_up"], params["down"])

    def _initial_experts_for_layer(
        self,
        profile_row: int,
        expert_ids: list[int],
        num_slots: int,
    ) -> list[int]:
        act_freq = None if self.draft_reroute_profile is None else self.draft_reroute_profile.act_freq
        if act_freq is None or profile_row >= int(act_freq.shape[0]):
            return expert_ids[:num_slots]
        ranked = sorted(
            expert_ids,
            key=lambda expert_idx: (-float(act_freq[profile_row, expert_idx]), int(expert_idx)),
        )
        return ranked[:num_slots]

    def _seed_cache_stats_from_profile(
        self,
        cache: LayerExpertCache,
        profile_row: int,
        expert_ids: list[int],
    ) -> None:
        act_freq = None if self.draft_reroute_profile is None else self.draft_reroute_profile.act_freq
        if act_freq is None or profile_row >= int(act_freq.shape[0]):
            return
        for expert_idx in expert_ids:
            if not (0 <= int(expert_idx) < cache.num_experts):
                continue
            freq = float(act_freq[profile_row, expert_idx])
            cache.access_count[expert_idx] = int(round(freq * 1000.0))
            cache.access_score_sum[expert_idx] = float(cache.access_count[expert_idx])

    def _to_cpu(self, x: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        if dtype is None:
            dtype = self._target_expert_dtype()
        x = x.to(device="cpu", dtype=dtype).contiguous()
        return x.pin_memory() if self.pin_memory else x

    def _target_expert_dtype(self) -> torch.dtype:
        dtype = getattr(self.hf_config, "torch_dtype", None)
        return dtype if isinstance(dtype, torch.dtype) else torch.get_default_dtype()

    @staticmethod
    def _parse_layer_idx(weight_name: str) -> int:
        parts = weight_name.split(".")
        return int(parts[parts.index("layers") + 1])

    @staticmethod
    def _parse_expert_idx(weight_name: str) -> int:
        parts = weight_name.split(".")
        return int(parts[parts.index("experts") + 1])
