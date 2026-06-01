from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


PROFILE_TENSOR_KEYS = ("cond_sim", "skip_err", "sim", "sens", "act_freq")
PROFILE_FORMAT_VERSION = "2"


@dataclass(frozen=True)
class DraftRerouteProfile:
    cond_sim: torch.Tensor | None
    skip_err: torch.Tensor | None
    sim: torch.Tensor | None
    sens: torch.Tensor | None
    act_freq: torch.Tensor | None
    metadata: dict[str, str]
    is_legacy: bool = False

    def similarity_artifact(self) -> dict[str, torch.Tensor]:
        if self.cond_sim is None or self.skip_err is None:
            raise ValueError("draft reroute profile must include cond_sim and skip_err")
        return {"cond_sim": self.cond_sim, "skip_err": self.skip_err}


def load_draft_reroute_profile(
    path: str,
    *,
    num_experts: int,
    expected_layers: int | None = None,
    expected_top_k: int | None = None,
    hf_config: object | None = None,
) -> DraftRerouteProfile:
    artifact_path = Path(path)
    tensors, metadata, legacy = _read_profile_payload(artifact_path)
    metadata = {str(k): str(v) for k, v in metadata.items()}

    _validate_metadata(
        metadata,
        num_experts=num_experts,
        expected_layers=expected_layers,
        expected_top_k=expected_top_k,
        hf_config=hf_config,
    )

    normalized = {
        key: _normalize_tensor(key, tensors.get(key))
        for key in PROFILE_TENSOR_KEYS
    }
    _validate_tensors(
        normalized,
        metadata,
        num_experts=num_experts,
        expected_layers=expected_layers,
    )

    return DraftRerouteProfile(
        cond_sim=normalized["cond_sim"],
        skip_err=normalized["skip_err"],
        sim=normalized["sim"],
        sens=normalized["sens"],
        act_freq=normalized["act_freq"],
        metadata=metadata,
        is_legacy=legacy,
    )


def save_draft_reroute_profile(
    path: str,
    *,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, object],
) -> None:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_tensors = {
        key: tensor.detach().cpu().float().contiguous()
        for key, tensor in tensors.items()
        if key in PROFILE_TENSOR_KEYS and isinstance(tensor, torch.Tensor)
    }
    normalized_metadata = {str(k): str(v) for k, v in metadata.items()}
    normalized_metadata.setdefault("format_version", PROFILE_FORMAT_VERSION)

    if artifact_path.suffix == ".safetensors":
        from safetensors.torch import save_file

        save_file(normalized_tensors, str(artifact_path), metadata=normalized_metadata)
    else:
        torch.save(
            {"metadata": normalized_metadata, "tensors": normalized_tensors},
            artifact_path,
        )


def seed_lfu_rank_guard_from_profile(
    strategy: object,
    profile: DraftRerouteProfile | None,
    *,
    layer_indices: Iterable[int],
    top_k: int,
) -> None:
    if not bool(getattr(strategy, "profile_seed_enabled", True)):
        return
    if profile is None or profile.act_freq is None:
        return
    for profile_row, layer_idx in enumerate(layer_indices):
        if profile_row >= int(profile.act_freq.shape[0]):
            break
        scores = [
            round(float(value) * int(top_k), 6)
            for value in profile.act_freq[profile_row].tolist()
        ]
        set_rank_scores = getattr(strategy, "set_rank_scores", None)
        if callable(set_rank_scores):
            set_rank_scores(int(layer_idx), scores)


def _read_profile_payload(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, object], bool]:
    if path.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        return tensors, metadata, False

    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("draft reroute profile must contain a tensor dictionary")

    if isinstance(payload.get("tensors"), dict):
        tensors = payload["tensors"]
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("draft reroute profile metadata must be a dictionary")
        return tensors, metadata, False

    return payload, {}, True


def _normalize_tensor(key: str, tensor: object) -> torch.Tensor | None:
    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"draft reroute profile field {key} must be a tensor")
    return tensor.float().cpu().contiguous()


def _validate_metadata(
    metadata: dict[str, str],
    *,
    num_experts: int,
    expected_layers: int | None,
    expected_top_k: int | None,
    hf_config: object | None,
) -> None:
    if "num_experts" in metadata and int(metadata["num_experts"]) != int(num_experts):
        raise ValueError(
            f"draft reroute profile num_experts={metadata['num_experts']} "
            f"does not match runtime num_experts={num_experts}"
        )
    if expected_layers is not None and "num_layers" in metadata and int(metadata["num_layers"]) != int(expected_layers):
        raise ValueError(
            f"draft reroute profile num_layers={metadata['num_layers']} "
            f"does not match expected_layers={expected_layers}"
        )
    if expected_top_k is not None and "top_k" in metadata and int(metadata["top_k"]) != int(expected_top_k):
        raise ValueError(
            f"draft reroute profile top_k={metadata['top_k']} "
            f"does not match expected_top_k={expected_top_k}"
        )
    if hf_config is not None and "model_type" in metadata:
        runtime_model_type = getattr(hf_config, "model_type", None)
        if runtime_model_type is not None and str(runtime_model_type) != metadata["model_type"]:
            raise ValueError(
                f"draft reroute profile model_type={metadata['model_type']} "
                f"does not match runtime model_type={runtime_model_type}"
            )
    if hf_config is not None and "hidden_size" in metadata:
        runtime_hidden_size = getattr(hf_config, "hidden_size", None)
        if runtime_hidden_size is not None and int(metadata["hidden_size"]) != int(runtime_hidden_size):
            raise ValueError(
                f"draft reroute profile hidden_size={metadata['hidden_size']} "
                f"does not match runtime hidden_size={runtime_hidden_size}"
            )


def _validate_tensors(
    tensors: dict[str, torch.Tensor | None],
    metadata: dict[str, str],
    *,
    num_experts: int,
    expected_layers: int | None,
) -> None:
    layer_count = expected_layers
    if layer_count is None and "num_layers" in metadata:
        layer_count = int(metadata["num_layers"])

    _validate_pair_tensor("cond_sim", tensors["cond_sim"], num_experts, layer_count)
    _validate_pair_tensor("sim", tensors["sim"], num_experts, layer_count)
    _validate_layer_expert_tensor("skip_err", tensors["skip_err"], num_experts, layer_count)
    _validate_layer_expert_tensor("act_freq", tensors["act_freq"], num_experts, layer_count)
    _validate_layer_tensor("sens", tensors["sens"], layer_count)

    cond_sim = tensors["cond_sim"]
    skip_err = tensors["skip_err"]
    if (cond_sim is None) != (skip_err is None):
        raise ValueError("draft reroute profile cond_sim and skip_err must be provided together")
    if cond_sim is not None and skip_err is not None and int(cond_sim.shape[0]) != int(skip_err.shape[0]):
        raise ValueError("draft reroute profile skip_err layer count does not match cond_sim")


def _validate_pair_tensor(
    name: str,
    tensor: torch.Tensor | None,
    num_experts: int,
    layer_count: int | None,
) -> None:
    if tensor is None:
        return
    if tensor.ndim != 3:
        raise ValueError(f"draft reroute profile {name} must have shape [layers, experts, experts]")
    if tuple(tensor.shape[1:]) != (int(num_experts), int(num_experts)):
        raise ValueError(f"draft reroute profile {name} expert dimensions do not match num_experts={num_experts}")
    if layer_count is not None and int(tensor.shape[0]) != int(layer_count):
        raise ValueError(f"draft reroute profile {name} layer count does not match num_layers={layer_count}")


def _validate_layer_expert_tensor(
    name: str,
    tensor: torch.Tensor | None,
    num_experts: int,
    layer_count: int | None,
) -> None:
    if tensor is None:
        return
    if tensor.ndim != 2:
        raise ValueError(f"draft reroute profile {name} must have shape [layers, experts]")
    if int(tensor.shape[1]) != int(num_experts):
        raise ValueError(f"draft reroute profile {name} expert dimension does not match num_experts={num_experts}")
    if layer_count is not None and int(tensor.shape[0]) != int(layer_count):
        raise ValueError(f"draft reroute profile {name} layer count does not match num_layers={layer_count}")


def _validate_layer_tensor(
    name: str,
    tensor: torch.Tensor | None,
    layer_count: int | None,
) -> None:
    if tensor is None:
        return
    if tensor.ndim != 1:
        raise ValueError(f"draft reroute profile {name} must have shape [layers]")
    if layer_count is not None and int(tensor.shape[0]) != int(layer_count):
        raise ValueError(f"draft reroute profile {name} layer count does not match num_layers={layer_count}")
