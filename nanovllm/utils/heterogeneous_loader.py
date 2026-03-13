from __future__ import annotations

import os
from glob import glob

import torch
from safetensors import safe_open

from nanovllm.config import Config
from nanovllm.expert.cache import LayerExpertCache
from nanovllm.models.qwen3_moe import Qwen3MoeForCausalLM
from nanovllm.utils.loader import default_weight_loader


class HeterogeneousModelLoader:
    """Load non-expert weights to GPU and route expert weights through CPU pool."""

    def __init__(self, config: Config):
        self.config = config
        self.hf_config = config.hf_config
        self.pin_memory = config.cpu_expert_pin_memory

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
                        cpu_pool[layer_idx][expert_idx]["down"] = self._to_cpu(weight)
                    elif "gate_proj" in weight_name:
                        pending_gate[key] = weight
                    elif "up_proj" in weight_name:
                        pending_up[key] = weight

        for key, gate in pending_gate.items():
            up = pending_up[key]
            gate_up = torch.cat([gate, up], dim=0)
            layer_idx, expert_idx = key
            cpu_pool[layer_idx][expert_idx]["gate_up"] = self._to_cpu(gate_up)

        return cpu_pool

    def _init_layer_caches(
        self,
        cpu_pool: dict[int, dict[int, dict[str, torch.Tensor]]],
    ) -> dict[int, LayerExpertCache]:
        slots_per_layer = self.config.heterogeneous_slots_per_layer
        layer_caches: dict[int, LayerExpertCache] = {}
        for layer_idx, experts in cpu_pool.items():
            num_experts = len(experts)
            slots = num_experts if slots_per_layer <= 0 else min(slots_per_layer, num_experts)
            sample = next(iter(experts.values()))
            layer_caches[layer_idx] = LayerExpertCache(
                num_experts=num_experts,
                slots_per_layer=slots,
                gate_up_shape=tuple(sample["gate_up"].shape),
                down_shape=tuple(sample["down"].shape),
                device=torch.device("cuda"),
                dtype=self.hf_config.torch_dtype,
                cpu_expert_pool=experts,
            )
        return layer_caches

    def _load_initial_placement(
        self,
        layer_caches: dict[int, LayerExpertCache],
        cpu_pool: dict[int, dict[int, dict[str, torch.Tensor]]],
    ) -> None:
        # First step default: S=N, so this maps expert i -> slot i where possible.
        for layer_idx, cache in layer_caches.items():
            expert_ids = sorted(cpu_pool[layer_idx].keys())
            for slot_idx, expert_idx in enumerate(expert_ids[: cache.num_slots]):
                params = cpu_pool[layer_idx][expert_idx]
                cache.put_to_slot(slot_idx, expert_idx, params["gate_up"], params["down"])

    def _to_cpu(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to("cpu")
        return x.pin_memory() if self.pin_memory else x

    @staticmethod
    def _parse_layer_idx(weight_name: str) -> int:
        parts = weight_name.split(".")
        return int(parts[parts.index("layers") + 1])

    @staticmethod
    def _parse_expert_idx(weight_name: str) -> int:
        parts = weight_name.split(".")
        return int(parts[parts.index("experts") + 1])
