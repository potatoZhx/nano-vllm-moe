from __future__ import annotations

import os
from dataclasses import dataclass
from time import perf_counter

import torch

from nanovllm.expert.cache import LayerExpertCache
from nanovllm.scheduling.draft_scheduler import DraftScheduler


@dataclass
class MoEExecutionPlan:
    layer_idx: int
    gpu_route_indices: torch.Tensor
    gpu_m_sizes: torch.Tensor | None
    cpu_route_indices: torch.Tensor | None
    cpu_task_expert_ids: torch.Tensor | None
    cpu_task_offsets: torch.Tensor | None
    cpu_task_expert_ids_host: list[int] | None = None
    cpu_task_offsets_host: list[int] | None = None
    flat_selected_original: torch.Tensor | None = None
    flat_selected_effective: torch.Tensor | None = None
    gpu_route_weights: torch.Tensor | None = None
    cpu_graph_enabled: bool = False
    cpu_graph_async: bool = False
    substitution_lut: torch.Tensor | None = None
    gpu_route_mask: torch.Tensor | None = None
    cpu_route_mask: torch.Tensor | None = None

    @property
    def m_sizes(self) -> torch.Tensor | None:
        # Backward-compatible alias used by existing call sites/tests.
        return self.gpu_m_sizes

    @property
    def substitution_map(self) -> dict[int, int]:
        # Kept only for compatibility/debug access. Hot path should use substitution_lut.
        if self.substitution_lut is None:
            return {}
        lut_cpu = self.substitution_lut.detach().to("cpu")
        idx = torch.nonzero(lut_cpu != torch.arange(lut_cpu.numel(), dtype=lut_cpu.dtype), as_tuple=False).flatten()
        out: dict[int, int] = {}
        for i in idx.tolist():
            out[int(i)] = int(lut_cpu[i].item())
        return out


@dataclass
class VerifyCacheFillResult:
    promoted_expert_ids: list[int]
    cpu_expert_ids: list[int]
    evicted_expert_ids: list[int]
    skipped_pending_count: int
    transfer_ms: float

    @property
    def promoted_expert_count(self) -> int:
        return len(self.promoted_expert_ids)

    @property
    def cpu_expert_count(self) -> int:
        return len(self.cpu_expert_ids)


def _build_grouped_layout(
    gpu_slots: torch.Tensor,
    gpu_route_indices: torch.Tensor,
    num_slots: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Group routes by slot id with deterministic tie-break on original route index.
    # This keeps intra-slot order stable across runs and matches route-major semantics.
    route_order = torch.argsort(gpu_route_indices, stable=True)
    slots_by_route = gpu_slots.index_select(0, route_order)
    slot_order = torch.argsort(slots_by_route, stable=True)
    final_order = route_order.index_select(0, slot_order)
    sorted_slots = gpu_slots.index_select(0, final_order)
    sorted_gpu_route_indices = gpu_route_indices.index_select(0, final_order)
    # Use fixed-shape scatter_add to keep graph capture friendly on some runtimes.
    m_sizes = torch.zeros(num_slots, dtype=torch.int32, device=sorted_slots.device)
    ones = torch.ones_like(sorted_slots, dtype=torch.int32)
    m_sizes.scatter_add_(0, sorted_slots.to(torch.int64), ones)
    return m_sizes, sorted_gpu_route_indices


def _build_grouped_layout_masked(
    gpu_slots: torch.Tensor,
    gpu_route_indices: torch.Tensor,
    route_mask: torch.Tensor,
    num_slots: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Fixed-shape layout for graph capture: active routes are grouped first by
    # slot, inactive routes remain in the returned permutation tail with zero
    # contribution to m_sizes.
    inactive_slot = torch.full_like(gpu_slots, int(num_slots))
    sort_slots = torch.where(route_mask, gpu_slots, inactive_slot)
    route_order = torch.argsort(gpu_route_indices, stable=True)
    slots_by_route = sort_slots.index_select(0, route_order)
    slot_order = torch.argsort(slots_by_route, stable=True)
    final_order = route_order.index_select(0, slot_order)
    sorted_slots = sort_slots.index_select(0, final_order)
    sorted_gpu_route_indices = gpu_route_indices.index_select(0, final_order)

    active_counts = route_mask.index_select(0, final_order).to(torch.int32)
    scatter_slots = torch.where(
        sorted_slots < int(num_slots),
        sorted_slots,
        torch.zeros_like(sorted_slots),
    )
    m_sizes = torch.zeros(num_slots, dtype=torch.int32, device=sorted_slots.device)
    m_sizes.scatter_add_(0, scatter_slots.to(torch.int64), active_counts)
    return m_sizes, sorted_gpu_route_indices


def _verify_compact_gpu_routes_enabled() -> bool:
    return os.getenv("NANOVLLM_VERIFY_COMPACT_GPU_ROUTES", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _select_verify_cache_fill_slot(
    expert_cache: LayerExpertCache,
    active_expert_ids: set[int],
) -> tuple[int | None, int, int]:
    skipped_pending_slots: set[int] = set()
    for slot_idx, slot_expert in enumerate(expert_cache.slot_to_expert):
        if expert_cache.is_active_slot_pending(slot_idx):
            skipped_pending_slots.add(slot_idx)
            continue
        if int(slot_expert) < 0:
            return slot_idx, int(slot_expert), len(skipped_pending_slots)

    best_slot: int | None = None
    best_prev_expert = -1
    best_last_access: int | None = None
    for slot_idx, slot_expert in enumerate(expert_cache.slot_to_expert):
        if expert_cache.is_active_slot_pending(slot_idx):
            skipped_pending_slots.add(slot_idx)
            continue
        prev_expert = int(slot_expert)
        if prev_expert in active_expert_ids:
            continue
        last_access = int(expert_cache.last_access_step[prev_expert]) if prev_expert >= 0 else -1
        if best_last_access is None or last_access < best_last_access:
            best_slot = slot_idx
            best_prev_expert = prev_expert
            best_last_access = last_access

    return best_slot, best_prev_expert, len(skipped_pending_slots)


def apply_verify_cache_fill_policy(
    layer_idx: int,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_cache: LayerExpertCache,
    step_id: int,
    profile: dict[str, float] | None = None,
) -> VerifyCacheFillResult:
    """Promote verify miss experts into active cache slots before planning.

    The policy keeps active cached experts resident. Miss experts with the fewest
    active routes remain on CPU only when the active unique set exceeds cache
    capacity.
    """
    _ = layer_idx, routing_weights, step_id
    flat_selected = selected_experts.reshape(-1).to(torch.int64)
    if flat_selected.numel() == 0:
        return VerifyCacheFillResult([], [], [], 0, 0.0)

    flat_cpu = flat_selected.detach().to(device=torch.device("cpu"), dtype=torch.int64)
    unique_ids, counts = torch.unique(flat_cpu, sorted=True, return_counts=True)
    route_counts = {int(eid): int(count) for eid, count in zip(unique_ids.tolist(), counts.tolist())}
    active_expert_ids = set(route_counts)

    miss_ids = [expert_idx for expert_idx in sorted(active_expert_ids) if not expert_cache.is_cached_cpu(expert_idx)]
    if not miss_ids:
        if profile is not None:
            profile["verify_cache_fill_promoted_expert_count"] = float(
                profile.get("verify_cache_fill_promoted_expert_count", 0.0)
            )
            profile["verify_cache_fill_cpu_expert_count"] = float(
                profile.get("verify_cache_fill_cpu_expert_count", 0.0)
            )
        return VerifyCacheFillResult([], [], [], 0, 0.0)

    overflow = max(0, len(active_expert_ids) - int(expert_cache.num_slots))
    cpu_expert_ids = set()
    if overflow > 0:
        ranked_cpu = sorted(miss_ids, key=lambda expert_idx: (route_counts[expert_idx], expert_idx))
        cpu_expert_ids.update(ranked_cpu[:overflow])
    promote_ids = sorted(
        (expert_idx for expert_idx in miss_ids if expert_idx not in cpu_expert_ids),
        key=lambda expert_idx: (-route_counts[expert_idx], expert_idx),
    )

    promoted: list[int] = []
    evicted: list[int] = []
    skipped_pending_count = 0
    transfer_t0 = perf_counter()
    for expert_idx in promote_ids:
        slot_idx, prev_expert, skipped = _select_verify_cache_fill_slot(expert_cache, active_expert_ids)
        skipped_pending_count += skipped
        if slot_idx is None:
            cpu_expert_ids.add(expert_idx)
            continue
        params = expert_cache.get_cpu_expert_weights(expert_idx)
        if params is None or "gate_up" not in params or "down" not in params:
            raise RuntimeError(f"Missing CPU expert weights for verify cache fill expert {expert_idx}")
        expert_cache.put_to_slot(slot_idx, expert_idx, params["gate_up"], params["down"])
        promoted.append(expert_idx)
        if prev_expert >= 0 and prev_expert != expert_idx:
            evicted.append(prev_expert)
    transfer_ms = (perf_counter() - transfer_t0) * 1000.0 if promoted else 0.0

    cpu_sorted = sorted(cpu_expert_ids)
    if profile is not None:
        profile["verify_cache_fill_promoted_expert_count"] = float(
            profile.get("verify_cache_fill_promoted_expert_count", 0.0) + len(promoted)
        )
        profile["verify_cache_fill_cpu_expert_count"] = float(
            profile.get("verify_cache_fill_cpu_expert_count", 0.0) + len(cpu_sorted)
        )
        profile["verify_cache_fill_evicted_expert_count"] = float(
            profile.get("verify_cache_fill_evicted_expert_count", 0.0) + len(evicted)
        )
        profile["verify_cache_fill_skipped_pending_count"] = float(
            profile.get("verify_cache_fill_skipped_pending_count", 0.0) + skipped_pending_count
        )
        profile["verify_cache_fill_transfer_ms"] = float(
            profile.get("verify_cache_fill_transfer_ms", 0.0) + transfer_ms
        )

    return VerifyCacheFillResult(
        promoted_expert_ids=promoted,
        cpu_expert_ids=cpu_sorted,
        evicted_expert_ids=evicted,
        skipped_pending_count=skipped_pending_count,
        transfer_ms=transfer_ms,
    )


def collect_cache_fill_no_cpu_expert_ids(
    selected_experts: torch.Tensor,
    expert_cache: LayerExpertCache,
    profile: dict[str, float] | None = None,
) -> tuple[list[int], list[int]]:
    """Return active expert ids and currently-missing active expert ids.

    The hot path keeps all route metadata on the device and transfers only two
    fixed-size bitsets to the host: active experts and miss experts.
    """
    flat_selected = selected_experts.reshape(-1).to(torch.int64)
    if flat_selected.numel() == 0:
        if profile is not None:
            profile["verify_cache_fill_no_cpu_active_expert_count"] = float(
                profile.get("verify_cache_fill_no_cpu_active_expert_count", 0.0)
            )
            profile["verify_cache_fill_no_cpu_miss_expert_count"] = float(
                profile.get("verify_cache_fill_no_cpu_miss_expert_count", 0.0)
            )
        return [], []

    t0 = perf_counter()
    flags_device, flags_host = expert_cache.get_cache_fill_no_cpu_flag_buffers()
    active_flags = flags_device[0]
    miss_flags = flags_device[1]
    active_flags.zero_()
    miss_flags.zero_()

    active_flags.scatter_(0, flat_selected, torch.ones_like(flat_selected, dtype=torch.bool))
    slots = expert_cache.expert_to_slot_lut.index_select(0, flat_selected)
    miss_routes = slots.lt(0)
    miss_flags.scatter_(0, flat_selected, miss_routes)

    flags_host.copy_(flags_device, non_blocking=flags_device.is_cuda)
    if flags_device.is_cuda:
        torch.cuda.current_stream(flags_device.device).synchronize()
    active_ids = torch.nonzero(flags_host[0], as_tuple=False).flatten().tolist()
    miss_ids = torch.nonzero(flags_host[1], as_tuple=False).flatten().tolist()
    elapsed_ms = (perf_counter() - t0) * 1000.0

    if profile is not None:
        profile["verify_cache_fill_no_cpu_flag_ms"] = float(
            profile.get("verify_cache_fill_no_cpu_flag_ms", 0.0) + elapsed_ms
        )
        profile["verify_cache_fill_no_cpu_active_expert_count"] = float(
            profile.get("verify_cache_fill_no_cpu_active_expert_count", 0.0) + len(active_ids)
        )
        profile["verify_cache_fill_no_cpu_miss_expert_count"] = float(
            profile.get("verify_cache_fill_no_cpu_miss_expert_count", 0.0) + len(miss_ids)
        )
    return [int(x) for x in active_ids], [int(x) for x in miss_ids]


def _select_cache_fill_no_cpu_slots(
    expert_cache: LayerExpertCache,
    active_expert_ids: set[int],
    needed_count: int,
    cache_strategy: str,
) -> tuple[list[tuple[int, int]], int]:
    skipped_pending_count = 0
    empty_slots: list[tuple[int, int]] = []
    victim_slots: list[tuple[tuple[int, int, int], int, int]] = []
    use_lfu = str(cache_strategy).strip().lower() == "lfu"
    for slot_idx, slot_expert in enumerate(expert_cache.slot_to_expert):
        if expert_cache.is_active_slot_pending(slot_idx):
            skipped_pending_count += 1
            continue
        prev_expert = int(slot_expert)
        if prev_expert < 0:
            empty_slots.append((slot_idx, prev_expert))
            continue
        if prev_expert in active_expert_ids:
            continue
        primary = int(expert_cache.access_count[prev_expert]) if use_lfu else int(expert_cache.last_access_step[prev_expert])
        secondary = int(expert_cache.last_access_step[prev_expert]) if use_lfu else int(expert_cache.access_count[prev_expert])
        victim_slots.append(((primary, secondary, slot_idx), slot_idx, prev_expert))

    selected: list[tuple[int, int]] = empty_slots[:needed_count]
    if len(selected) < needed_count:
        victim_slots.sort(key=lambda item: item[0])
        selected.extend((slot_idx, prev_expert) for _, slot_idx, prev_expert in victim_slots[: needed_count - len(selected)])
    return selected, skipped_pending_count


def apply_verify_cache_fill_no_cpu_policy_ids(
    layer_idx: int,
    active_expert_ids: list[int],
    miss_expert_ids: list[int],
    expert_cache: LayerExpertCache,
    step_id: int,
    cache_strategy: str = "lru",
    profile: dict[str, float] | None = None,
) -> VerifyCacheFillResult:
    """Promote miss experts from compact id lists for cache_fill_no_cpu verify."""
    _ = layer_idx, step_id
    active_ids = sorted({int(eid) for eid in active_expert_ids if 0 <= int(eid) < expert_cache.num_experts})
    miss_ids = sorted(
        {
            int(eid)
            for eid in miss_expert_ids
            if 0 <= int(eid) < expert_cache.num_experts and not expert_cache.is_cached_cpu(int(eid))
        }
    )

    if len(active_ids) > int(expert_cache.num_slots):
        if profile is not None:
            profile["verify_cache_fill_promoted_expert_count"] = float(
                profile.get("verify_cache_fill_promoted_expert_count", 0.0)
            )
            profile["verify_cache_fill_cpu_expert_count"] = float(
                profile.get("verify_cache_fill_cpu_expert_count", 0.0) + len(miss_ids)
            )
            profile["verify_cache_fill_no_cpu_remaining_miss_count"] = float(
                profile.get("verify_cache_fill_no_cpu_remaining_miss_count", 0.0) + len(miss_ids)
            )
            profile["verify_cache_fill_no_cpu_remaining_miss_expert_count"] = float(
                profile.get("verify_cache_fill_no_cpu_remaining_miss_expert_count", 0.0) + len(miss_ids)
            )
            profile["verify_cache_fill_no_cpu_remaining_miss_route_count"] = float(
                profile.get("verify_cache_fill_no_cpu_remaining_miss_route_count", 0.0)
            )
            profile["verify_cache_fill_no_cpu_fallback_count"] = float(
                profile.get("verify_cache_fill_no_cpu_fallback_count", 0.0) + 1.0
            )
        return VerifyCacheFillResult([], miss_ids, [], 0, 0.0)

    if not miss_ids:
        if profile is not None:
            profile["verify_cache_fill_promoted_expert_count"] = float(
                profile.get("verify_cache_fill_promoted_expert_count", 0.0)
            )
            profile["verify_cache_fill_cpu_expert_count"] = float(
                profile.get("verify_cache_fill_cpu_expert_count", 0.0)
            )
            profile["verify_cache_fill_no_cpu_remaining_miss_count"] = float(
                profile.get("verify_cache_fill_no_cpu_remaining_miss_count", 0.0)
            )
            profile["verify_cache_fill_no_cpu_remaining_miss_expert_count"] = float(
                profile.get("verify_cache_fill_no_cpu_remaining_miss_expert_count", 0.0)
            )
            profile["verify_cache_fill_no_cpu_remaining_miss_route_count"] = float(
                profile.get("verify_cache_fill_no_cpu_remaining_miss_route_count", 0.0)
            )
            profile["verify_cache_fill_no_cpu_fallback_count"] = float(
                profile.get("verify_cache_fill_no_cpu_fallback_count", 0.0)
            )
        return VerifyCacheFillResult([], [], [], 0, 0.0)

    slots, skipped_pending_count = _select_cache_fill_no_cpu_slots(
        expert_cache=expert_cache,
        active_expert_ids=set(active_ids),
        needed_count=len(miss_ids),
        cache_strategy=cache_strategy,
    )
    if len(slots) < len(miss_ids):
        if profile is not None:
            profile["verify_cache_fill_promoted_expert_count"] = float(
                profile.get("verify_cache_fill_promoted_expert_count", 0.0)
            )
            profile["verify_cache_fill_cpu_expert_count"] = float(
                profile.get("verify_cache_fill_cpu_expert_count", 0.0) + len(miss_ids)
            )
            profile["verify_cache_fill_skipped_pending_count"] = float(
                profile.get("verify_cache_fill_skipped_pending_count", 0.0) + skipped_pending_count
            )
            profile["verify_cache_fill_no_cpu_remaining_miss_count"] = float(
                profile.get("verify_cache_fill_no_cpu_remaining_miss_count", 0.0) + len(miss_ids)
            )
            profile["verify_cache_fill_no_cpu_remaining_miss_expert_count"] = float(
                profile.get("verify_cache_fill_no_cpu_remaining_miss_expert_count", 0.0) + len(miss_ids)
            )
            profile["verify_cache_fill_no_cpu_remaining_miss_route_count"] = float(
                profile.get("verify_cache_fill_no_cpu_remaining_miss_route_count", 0.0)
            )
            profile["verify_cache_fill_no_cpu_fallback_count"] = float(
                profile.get("verify_cache_fill_no_cpu_fallback_count", 0.0) + 1.0
            )
        return VerifyCacheFillResult([], miss_ids, [], skipped_pending_count, 0.0)

    promoted: list[int] = []
    evicted: list[int] = []
    transfer_t0 = perf_counter()
    for expert_idx, (slot_idx, prev_expert) in zip(miss_ids, slots):
        params = expert_cache.get_cpu_expert_weights(expert_idx)
        if params is None or "gate_up" not in params or "down" not in params:
            raise RuntimeError(f"Missing CPU expert weights for verify cache fill expert {expert_idx}")
        expert_cache.put_to_slot(slot_idx, expert_idx, params["gate_up"], params["down"])
        promoted.append(expert_idx)
        if prev_expert >= 0 and prev_expert != expert_idx:
            evicted.append(prev_expert)
    transfer_ms = (perf_counter() - transfer_t0) * 1000.0 if promoted else 0.0

    if profile is not None:
        profile["verify_cache_fill_promoted_expert_count"] = float(
            profile.get("verify_cache_fill_promoted_expert_count", 0.0) + len(promoted)
        )
        profile["verify_cache_fill_cpu_expert_count"] = float(
            profile.get("verify_cache_fill_cpu_expert_count", 0.0)
        )
        profile["verify_cache_fill_evicted_expert_count"] = float(
            profile.get("verify_cache_fill_evicted_expert_count", 0.0) + len(evicted)
        )
        profile["verify_cache_fill_skipped_pending_count"] = float(
            profile.get("verify_cache_fill_skipped_pending_count", 0.0) + skipped_pending_count
        )
        profile["verify_cache_fill_transfer_ms"] = float(
            profile.get("verify_cache_fill_transfer_ms", 0.0) + transfer_ms
        )
        profile["verify_cache_fill_no_cpu_remaining_miss_count"] = float(
            profile.get("verify_cache_fill_no_cpu_remaining_miss_count", 0.0)
        )
        profile["verify_cache_fill_no_cpu_remaining_miss_expert_count"] = float(
            profile.get("verify_cache_fill_no_cpu_remaining_miss_expert_count", 0.0)
        )
        profile["verify_cache_fill_no_cpu_remaining_miss_route_count"] = float(
            profile.get("verify_cache_fill_no_cpu_remaining_miss_route_count", 0.0)
        )
        profile["verify_cache_fill_no_cpu_fallback_count"] = float(
            profile.get("verify_cache_fill_no_cpu_fallback_count", 0.0)
        )

    return VerifyCacheFillResult(
        promoted_expert_ids=promoted,
        cpu_expert_ids=[],
        evicted_expert_ids=evicted,
        skipped_pending_count=skipped_pending_count,
        transfer_ms=transfer_ms,
    )


def _build_topc0_substitution_lut(
    num_experts: int,
    cached_expert_mask: torch.Tensor,
    slot_to_expert_lut: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Build a deterministic substitution table for draft top_c=0.

    Strategy:
    - Cached experts keep identity mapping.
    - Uncached experts map to cached experts via round-robin over cache slots.
    - This produces fixed-shape tensors and avoids host-side branching in hot path.
    """
    expert_ids = torch.arange(num_experts, dtype=torch.int64, device=device)
    if slot_to_expert_lut.numel() == 0:
        return expert_ids

    rr_slot = torch.remainder(expert_ids, int(slot_to_expert_lut.numel()))
    fallback_experts = slot_to_expert_lut.index_select(0, rr_slot)
    # Defensive fallback: if a slot is not populated, keep identity for that id.
    fallback_experts = torch.where(fallback_experts >= 0, fallback_experts, expert_ids)
    return torch.where(cached_expert_mask, expert_ids, fallback_experts.to(torch.int64))


def _build_simple_scheduler_substitution_lut_static(
    need_substitution_mask: torch.Tensor,
    cached_expert_mask: torch.Tensor,
    num_experts: int,
    device: torch.device,
) -> torch.Tensor:
    """Capture-safe equivalent of SimpleDraftScheduler.build_substitution_lut_gpu.

    SimpleDraftScheduler maps the sorted expert ids that need substitution to
    sorted cached expert ids in round-robin order. The eager implementation uses
    nonzero() to compact those two sets; this version keeps all tensors at
    num_experts shape so it can run inside CUDA graph capture.
    """
    expert_ids = torch.arange(num_experts, dtype=torch.int64, device=device)
    sentinel = torch.full((num_experts,), num_experts, dtype=torch.int64, device=device)

    cached_sorted = torch.sort(torch.where(cached_expert_mask, expert_ids, sentinel)).values
    cached_count = cached_expert_mask.to(torch.int64).sum().clamp_min(1)
    rr_rank = torch.remainder(expert_ids, cached_count)
    substitute_by_rank = cached_sorted.index_select(0, rr_rank)

    need_rank = torch.cumsum(need_substitution_mask.to(torch.int64), dim=0) - 1
    substitutes = substitute_by_rank.index_select(0, need_rank.clamp_min(0))
    has_cached = cached_expert_mask.any()
    return torch.where(need_substitution_mask & has_cached, substitutes, expert_ids)


def _build_cpu_task_layout(
    flat_selected_original: torch.Tensor,
    cpu_route_indices: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, list[int] | None, list[int] | None]:
    if cpu_route_indices.numel() == 0:
        return None, None, None, None, None

    cpu_experts = flat_selected_original.index_select(0, cpu_route_indices)
    route_order = torch.argsort(cpu_route_indices, stable=True)
    experts_by_route = cpu_experts.index_select(0, route_order)
    expert_order = torch.argsort(experts_by_route, stable=True)
    final_order = route_order.index_select(0, expert_order)
    sorted_experts = cpu_experts.index_select(0, final_order)
    sorted_route_indices = cpu_route_indices.index_select(0, final_order)
    task_expert_ids, counts = torch.unique_consecutive(sorted_experts, return_counts=True)
    task_offsets = torch.zeros(
        task_expert_ids.numel() + 1,
        dtype=torch.int64,
        device=cpu_route_indices.device,
    )
    task_offsets[1:] = torch.cumsum(counts.to(torch.int64), dim=0)
    ids_cpu = task_expert_ids.to(torch.int64)
    # Pre-extract host metadata to avoid GPU->CPU transfer in hot path.
    ids_host = [int(x) for x in ids_cpu.detach().to("cpu", non_blocking=False).tolist()]
    offsets_host = [int(x) for x in task_offsets.detach().to("cpu", non_blocking=False).tolist()]
    return sorted_route_indices, ids_cpu, task_offsets, ids_host, offsets_host


def _build_uncached_expert_mask_static(
    flat_selected: torch.Tensor,
    uncached_route_mask: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    counts = torch.zeros((num_experts,), dtype=torch.int32, device=flat_selected.device)
    counts.scatter_add_(
        0,
        flat_selected.to(torch.int64),
        uncached_route_mask.to(torch.int32),
    )
    return counts.gt(0)


def _select_cpu_experts_topc_static(
    uncached_expert_mask: torch.Tensor,
    routing_weights_flat: torch.Tensor,
    selected_experts_flat: torch.Tensor,
    top_c: int,
    active_route_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    num_experts = int(uncached_expert_mask.numel())
    out = torch.zeros((num_experts,), dtype=torch.bool, device=uncached_expert_mask.device)
    if top_c <= 0 or num_experts <= 0:
        return out

    candidate_route_mask = uncached_expert_mask.index_select(0, selected_experts_flat.to(torch.int64))
    if active_route_mask is not None:
        candidate_route_mask = candidate_route_mask & active_route_mask
    score = torch.zeros((num_experts,), dtype=torch.float32, device=uncached_expert_mask.device)
    score.scatter_add_(
        0,
        selected_experts_flat.to(torch.int64),
        torch.where(candidate_route_mask, routing_weights_flat.float(), torch.zeros_like(routing_weights_flat.float())),
    )

    neg = torch.full_like(score, -float("inf"))
    masked_score = torch.where(uncached_expert_mask, score, neg)
    pick_n = min(int(top_c), num_experts)
    ranked = torch.argsort(masked_score, descending=True, stable=True)
    picked_ids = ranked[:pick_n]
    picked_score = masked_score.index_select(0, picked_ids)
    picked_valid = torch.isfinite(picked_score)
    out.scatter_(0, picked_ids.to(torch.int64), picked_valid)
    return out


def _flatten_experts(selected_experts: torch.Tensor) -> torch.Tensor:
    return selected_experts.reshape(-1).to(torch.int64)


def _flatten_weights(routing_weights: torch.Tensor) -> torch.Tensor:
    return routing_weights.reshape(-1).float()


def flatten_selected_and_weights(
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _flatten_experts(selected_experts), _flatten_weights(routing_weights)


def build_runtime_meta_view(
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return selected_experts, routing_weights


def build_prefill_plan(
    layer_idx: int,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_cache: LayerExpertCache,
    num_experts: int,
) -> MoEExecutionPlan:
    return build_prefill_plan_gpu(
        layer_idx=layer_idx,
        selected_experts=selected_experts,
        routing_weights=routing_weights,
        expert_cache=expert_cache,
        num_experts=num_experts,
    )


def build_verify_plan(
    layer_idx: int,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_cache: LayerExpertCache,
    num_experts: int,
) -> MoEExecutionPlan:
    return build_verify_plan_gpu(
        layer_idx=layer_idx,
        selected_experts=selected_experts,
        routing_weights=routing_weights,
        expert_cache=expert_cache,
        num_experts=num_experts,
    )


def build_verify_graph_safe_plan_gpu(
    layer_idx: int,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_cache: LayerExpertCache,
    num_experts: int,
) -> MoEExecutionPlan:
    """Graph-safe verify plan: all tensor ops, no torch.nonzero or .item().

    GPU processes ALL routes (uncached routes mapped to cached slots via
    substitution LUT with zeroed weights).  kt_direct handles miss experts
    (skips GPU-cached ones via gpu_expert_mask_cpu).
    """
    flat_selected = _flatten_experts(selected_experts)
    flat_weights = _flatten_weights(routing_weights)
    device = flat_selected.device

    cached_expert_mask = expert_cache.get_cached_expert_mask()
    slot_to_expert_lut = expert_cache.get_slot_to_expert_lut()

    substitution_lut = _build_topc0_substitution_lut(
        num_experts=num_experts,
        cached_expert_mask=cached_expert_mask,
        slot_to_expert_lut=slot_to_expert_lut,
        device=device,
    )
    flat_effective = substitution_lut.index_select(0, flat_selected)

    uncached_route_mask = ~cached_expert_mask.index_select(0, flat_selected)
    gpu_slots = expert_cache.expert_to_slot_lut.index_select(0, flat_effective)
    gpu_route_indices = torch.arange(flat_selected.numel(), dtype=torch.int64, device=device)
    if _verify_compact_gpu_routes_enabled():
        m_sizes, gpu_route_indices = _build_grouped_layout_masked(
            gpu_slots,
            gpu_route_indices,
            ~uncached_route_mask,
            expert_cache.num_slots,
        )
    else:
        m_sizes, gpu_route_indices = _build_grouped_layout(
            gpu_slots, gpu_route_indices, expert_cache.num_slots,
        )
    gpu_route_weights = torch.where(
        uncached_route_mask,
        torch.zeros_like(flat_weights),
        flat_weights,
    )

    return MoEExecutionPlan(
        layer_idx=layer_idx,
        gpu_route_indices=gpu_route_indices,
        gpu_m_sizes=m_sizes,
        cpu_route_indices=None,
        cpu_task_expert_ids=None,
        cpu_task_offsets=None,
        flat_selected_original=flat_selected,
        flat_selected_effective=flat_effective,
        gpu_route_weights=gpu_route_weights,
        cpu_graph_enabled=True,
        substitution_lut=substitution_lut,
        gpu_route_mask=torch.ones_like(flat_selected, dtype=torch.bool),
        cpu_route_mask=uncached_route_mask,
    )


def build_prefill_plan_gpu(
    layer_idx: int,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_cache: LayerExpertCache,
    num_experts: int,
) -> MoEExecutionPlan:
    flat_selected = _flatten_experts(selected_experts)
    _ = num_experts
    slot_indices, gpu_route_mask = expert_cache.remap_experts_to_slots(flat_selected)
    gpu_route_indices = torch.nonzero(gpu_route_mask, as_tuple=False).flatten()

    if gpu_route_indices.numel() > 0:
        gpu_slots = slot_indices.index_select(0, gpu_route_indices)
        m_sizes, gpu_route_indices = _build_grouped_layout(
            gpu_slots,
            gpu_route_indices,
            expert_cache.num_slots,
        )
    else:
        m_sizes = None

    cpu_route_mask = ~gpu_route_mask
    cpu_route_indices_raw = torch.nonzero(cpu_route_mask, as_tuple=False).flatten()
    cpu_route_indices, cpu_task_expert_ids, cpu_task_offsets, ids_host, offsets_host = _build_cpu_task_layout(
        flat_selected,
        cpu_route_indices_raw,
    )

    return MoEExecutionPlan(
        layer_idx=layer_idx,
        gpu_route_indices=gpu_route_indices,
        gpu_m_sizes=m_sizes,
        cpu_route_indices=cpu_route_indices,
        cpu_task_expert_ids=cpu_task_expert_ids,
        cpu_task_offsets=cpu_task_offsets,
        cpu_task_expert_ids_host=ids_host,
        cpu_task_offsets_host=offsets_host,
        flat_selected_original=flat_selected,
        flat_selected_effective=flat_selected,
        substitution_lut=None,
        gpu_route_mask=gpu_route_mask,
        cpu_route_mask=cpu_route_mask,
    )


def build_cache_fill_no_cpu_verify_plan_gpu(
    layer_idx: int,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_cache: LayerExpertCache,
    num_experts: int,
    profile: dict[str, float] | None = None,
    check_remaining_misses: bool = True,
) -> MoEExecutionPlan:
    """Build a verify plan optimized for cache-fill cases that become all-GPU.

    `cache_fill_no_cpu` expects the demand-load step to make every active route
    resident before expert compute. When that invariant holds, this avoids the
    generic prefill/verify CPU-route layout and returns a GPU-only plan. If the
    invariant is violated, keep correctness by falling back to the generic plan
    and record how many miss experts/routes remained after cache-fill.
    """
    flat_selected = _flatten_experts(selected_experts)
    _ = routing_weights, num_experts
    gpu_slots = expert_cache.expert_to_slot_lut.index_select(0, flat_selected)
    remaining_route_count = 0
    remaining_expert_count = 0
    if check_remaining_misses:
        remaining_miss_mask = gpu_slots.lt(0)
        remaining_route_count = int(remaining_miss_mask.sum().item()) if flat_selected.numel() > 0 else 0
        if remaining_route_count > 0:
            remaining_expert_count = int(torch.unique(flat_selected[remaining_miss_mask]).numel())

    if profile is not None and check_remaining_misses:
        profile["verify_cache_fill_no_cpu_remaining_miss_count"] = float(
            profile.get("verify_cache_fill_no_cpu_remaining_miss_count", 0.0) + remaining_expert_count
        )
        profile["verify_cache_fill_no_cpu_remaining_miss_expert_count"] = float(
            profile.get("verify_cache_fill_no_cpu_remaining_miss_expert_count", 0.0) + remaining_expert_count
        )
        profile["verify_cache_fill_no_cpu_remaining_miss_route_count"] = float(
            profile.get("verify_cache_fill_no_cpu_remaining_miss_route_count", 0.0) + remaining_route_count
        )

    if check_remaining_misses and remaining_route_count > 0:
        if profile is not None:
            profile["verify_cache_fill_no_cpu_fallback_count"] = float(
                profile.get("verify_cache_fill_no_cpu_fallback_count", 0.0) + 1.0
            )
        return build_prefill_plan_gpu(
            layer_idx=layer_idx,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            expert_cache=expert_cache,
            num_experts=num_experts,
        )

    if profile is not None and check_remaining_misses:
        profile["verify_cache_fill_no_cpu_fallback_count"] = float(
            profile.get("verify_cache_fill_no_cpu_fallback_count", 0.0)
        )

    gpu_route_indices = torch.arange(flat_selected.numel(), dtype=torch.int64, device=flat_selected.device)
    if gpu_route_indices.numel() > 0:
        m_sizes, gpu_route_indices = _build_grouped_layout(
            gpu_slots,
            gpu_route_indices,
            expert_cache.num_slots,
        )
    else:
        m_sizes = None
    return MoEExecutionPlan(
        layer_idx=layer_idx,
        gpu_route_indices=gpu_route_indices,
        gpu_m_sizes=m_sizes,
        cpu_route_indices=None,
        cpu_task_expert_ids=None,
        cpu_task_offsets=None,
        cpu_task_expert_ids_host=None,
        cpu_task_offsets_host=None,
        flat_selected_original=flat_selected,
        flat_selected_effective=flat_selected,
        substitution_lut=None,
        gpu_route_mask=None,
        cpu_route_mask=None,
    )


def build_draft_plan(
    layer_idx: int,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_cache: LayerExpertCache,
    draft_scheduler: DraftScheduler,
    num_experts: int,
    top_c: int,
    graph_safe_cpu: bool = False,
    graph_async_cpu: bool = False,
    active_token_mask: torch.Tensor | None = None,
) -> MoEExecutionPlan:
    return build_draft_plan_gpu(
        layer_idx=layer_idx,
        selected_experts=selected_experts,
        routing_weights=routing_weights,
        expert_cache=expert_cache,
        draft_scheduler=draft_scheduler,
        num_experts=num_experts,
        top_c=top_c,
        graph_safe_cpu=graph_safe_cpu,
        graph_async_cpu=graph_async_cpu,
        active_token_mask=active_token_mask,
    )


def build_draft_plan_gpu(
    layer_idx: int,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_cache: LayerExpertCache,
    draft_scheduler: DraftScheduler,
    num_experts: int,
    top_c: int,
    graph_safe_cpu: bool = False,
    graph_async_cpu: bool = False,
    active_token_mask: torch.Tensor | None = None,
) -> MoEExecutionPlan:
    flat_selected = _flatten_experts(selected_experts)
    flat_weights = _flatten_weights(routing_weights)
    device = flat_selected.device

    cached_expert_mask = expert_cache.get_cached_expert_mask()
    active_route_mask = None
    if graph_safe_cpu and active_token_mask is not None:
        active_tokens = active_token_mask.reshape(-1).to(device=device, dtype=torch.bool)
        routes_per_token = int(flat_selected.numel() // max(1, active_tokens.numel()))
        active_route_mask = active_tokens.repeat_interleave(routes_per_token)

    # Unified top_c=0 path (no CPU execution): map uncached experts to cached experts
    # through substitution LUT, then run fully on GPU.
    if top_c <= 0:
        substitution_lut = _build_topc0_substitution_lut(
            num_experts=num_experts,
            cached_expert_mask=cached_expert_mask,
            slot_to_expert_lut=expert_cache.get_slot_to_expert_lut(),
            device=device,
        )
        flat_effective = substitution_lut.index_select(0, flat_selected)
        gpu_slots = expert_cache.expert_to_slot_lut.index_select(0, flat_effective)
        gpu_route_indices = torch.arange(flat_selected.numel(), dtype=torch.int64, device=device)
        m_sizes, gpu_route_indices = _build_grouped_layout(
            gpu_slots,
            gpu_route_indices,
            expert_cache.num_slots,
        )
        gpu_route_mask = torch.ones_like(flat_selected, dtype=torch.bool)
        cpu_route_mask = torch.zeros_like(flat_selected, dtype=torch.bool)
        return MoEExecutionPlan(
            layer_idx=layer_idx,
            gpu_route_indices=gpu_route_indices,
            gpu_m_sizes=m_sizes,
            cpu_route_indices=None,
            cpu_task_expert_ids=None,
            cpu_task_offsets=None,
            cpu_task_expert_ids_host=None,
            cpu_task_offsets_host=None,
            flat_selected_original=flat_selected,
            flat_selected_effective=flat_effective,
            gpu_route_weights=None,
            cpu_graph_enabled=False,
            cpu_graph_async=False,
            substitution_lut=substitution_lut,
            gpu_route_mask=gpu_route_mask,
            cpu_route_mask=cpu_route_mask,
        )

    slot_indices, gpu_mask = expert_cache.remap_experts_to_slots(flat_selected)
    all_uncached_route_mask = ~gpu_mask
    uncached_route_mask = all_uncached_route_mask
    if active_route_mask is not None:
        uncached_route_mask = uncached_route_mask & active_route_mask
    if graph_safe_cpu:
        uncached_expert_mask = _build_uncached_expert_mask_static(
            flat_selected=flat_selected,
            uncached_route_mask=uncached_route_mask,
            num_experts=num_experts,
        )
    else:
        uncached_expert_mask = torch.zeros((num_experts,), dtype=torch.bool, device=device)
        uncached_expert_ids = torch.unique(flat_selected[uncached_route_mask])
        uncached_expert_mask[uncached_expert_ids] = True

    if graph_safe_cpu:
        cpu_expert_mask = _select_cpu_experts_topc_static(
            uncached_expert_mask=uncached_expert_mask,
            routing_weights_flat=flat_weights,
            selected_experts_flat=flat_selected,
            top_c=top_c,
            active_route_mask=active_route_mask,
        )
    else:
        cpu_expert_mask = draft_scheduler.select_cpu_experts_gpu(
            uncached_expert_mask=uncached_expert_mask,
            routing_weights_flat=flat_weights,
            selected_experts_flat=flat_selected,
            top_c=top_c,
        )
    need_substitution_mask = uncached_expert_mask & (~cpu_expert_mask)
    if graph_safe_cpu:
        substitution_lut = _build_simple_scheduler_substitution_lut_static(
            need_substitution_mask=need_substitution_mask,
            cached_expert_mask=cached_expert_mask,
            num_experts=num_experts,
            device=device,
        )
        all_uncached_expert_mask = _build_uncached_expert_mask_static(
            flat_selected=flat_selected,
            uncached_route_mask=all_uncached_route_mask,
            num_experts=num_experts,
        )
        base_substitution_lut = _build_simple_scheduler_substitution_lut_static(
            need_substitution_mask=all_uncached_expert_mask,
            cached_expert_mask=cached_expert_mask,
            num_experts=num_experts,
            device=device,
        )
    else:
        base_substitution_lut = None
        substitution_lut = draft_scheduler.build_substitution_lut_gpu(
            cpu_expert_mask=need_substitution_mask,
            cached_expert_mask=cached_expert_mask,
            num_experts=num_experts,
            device=device,
        )

    flat_effective = substitution_lut.index_select(0, flat_selected)
    selected_cpu_mask = cpu_expert_mask.index_select(0, flat_selected)
    if active_route_mask is not None:
        selected_cpu_mask = selected_cpu_mask & active_route_mask

    if graph_safe_cpu:
        if graph_async_cpu:
            topc0_lut = _build_topc0_substitution_lut(
                num_experts=num_experts,
                cached_expert_mask=cached_expert_mask,
                slot_to_expert_lut=expert_cache.get_slot_to_expert_lut(),
                device=device,
            )
            flat_gpu_effective = topc0_lut.index_select(0, flat_selected)
            gpu_slots = expert_cache.expert_to_slot_lut.index_select(0, flat_gpu_effective)
            gpu_route_indices = torch.arange(flat_selected.numel(), dtype=torch.int64, device=device)
            m_sizes, gpu_route_indices = _build_grouped_layout(
                gpu_slots,
                gpu_route_indices,
                expert_cache.num_slots,
            )
            gpu_route_mask = torch.ones_like(flat_selected, dtype=torch.bool)
            cpu_route_mask = selected_cpu_mask
            cpu_route_indices = torch.arange(flat_selected.numel(), dtype=torch.int64, device=device)
            return MoEExecutionPlan(
                layer_idx=layer_idx,
                gpu_route_indices=gpu_route_indices,
                gpu_m_sizes=m_sizes,
                cpu_route_indices=cpu_route_indices,
                cpu_task_expert_ids=None,
                cpu_task_offsets=None,
                cpu_task_expert_ids_host=None,
                cpu_task_offsets_host=None,
                flat_selected_original=flat_selected,
                flat_selected_effective=flat_gpu_effective,
                gpu_route_weights=None,
                cpu_graph_enabled=True,
                cpu_graph_async=True,
                substitution_lut=topc0_lut,
                gpu_route_mask=gpu_route_mask,
                cpu_route_mask=cpu_route_mask,
            )
        if base_substitution_lut is None:
            base_substitution_lut = _build_topc0_substitution_lut(
                num_experts=num_experts,
                cached_expert_mask=cached_expert_mask,
                slot_to_expert_lut=expert_cache.get_slot_to_expert_lut(),
                device=device,
            )
        cpu_gpu_fallback = base_substitution_lut.index_select(0, flat_selected)
        dummy_gpu_mask = selected_cpu_mask
        if active_route_mask is not None:
            dummy_gpu_mask = dummy_gpu_mask | (~active_route_mask)
        flat_gpu_effective = torch.where(dummy_gpu_mask, cpu_gpu_fallback, flat_effective)
        gpu_slots = expert_cache.expert_to_slot_lut.index_select(0, flat_gpu_effective)
        gpu_route_indices = torch.arange(flat_selected.numel(), dtype=torch.int64, device=device)
        m_sizes, gpu_route_indices = _build_grouped_layout(
            gpu_slots,
            gpu_route_indices,
            expert_cache.num_slots,
        )
        gpu_route_weights = torch.where(
            dummy_gpu_mask,
            torch.zeros_like(routing_weights.reshape(-1)),
            routing_weights.reshape(-1),
        )
        gpu_route_mask = torch.ones_like(flat_selected, dtype=torch.bool)
        cpu_route_mask = selected_cpu_mask
        cpu_route_indices = torch.arange(flat_selected.numel(), dtype=torch.int64, device=device)
        return MoEExecutionPlan(
            layer_idx=layer_idx,
            gpu_route_indices=gpu_route_indices,
            gpu_m_sizes=m_sizes,
            cpu_route_indices=cpu_route_indices,
            cpu_task_expert_ids=None,
            cpu_task_offsets=None,
            cpu_task_expert_ids_host=None,
            cpu_task_offsets_host=None,
            flat_selected_original=flat_selected,
            flat_selected_effective=flat_gpu_effective,
            gpu_route_weights=gpu_route_weights,
            cpu_graph_enabled=True,
            cpu_graph_async=False,
            substitution_lut=substitution_lut,
            gpu_route_mask=gpu_route_mask,
            cpu_route_mask=cpu_route_mask,
        )

    slot_eff, gpu_mask_eff = expert_cache.remap_experts_to_slots(flat_effective)
    gpu_route_mask = gpu_mask_eff & (~selected_cpu_mask)
    cpu_route_mask = ~gpu_route_mask

    gpu_route_indices = torch.nonzero(gpu_route_mask, as_tuple=False).flatten()
    if gpu_route_indices.numel() > 0:
        gpu_slots = slot_eff.index_select(0, gpu_route_indices)
        m_sizes, gpu_route_indices = _build_grouped_layout(
            gpu_slots,
            gpu_route_indices,
            expert_cache.num_slots,
        )
    else:
        m_sizes = None

    cpu_route_indices_raw = torch.nonzero(cpu_route_mask, as_tuple=False).flatten()
    cpu_route_indices, cpu_task_expert_ids, cpu_task_offsets, ids_host, offsets_host = _build_cpu_task_layout(
        flat_selected,
        cpu_route_indices_raw,
    )

    return MoEExecutionPlan(
        layer_idx=layer_idx,
        gpu_route_indices=gpu_route_indices,
        gpu_m_sizes=m_sizes,
        cpu_route_indices=cpu_route_indices,
        cpu_task_expert_ids=cpu_task_expert_ids,
        cpu_task_offsets=cpu_task_offsets,
        cpu_task_expert_ids_host=ids_host,
        cpu_task_offsets_host=offsets_host,
        flat_selected_original=flat_selected,
        flat_selected_effective=flat_effective,
        gpu_route_weights=None,
        cpu_graph_enabled=False,
        substitution_lut=substitution_lut,
        gpu_route_mask=gpu_route_mask,
        cpu_route_mask=cpu_route_mask,
    )


def build_cached_draft_plan_gpu(
    layer_idx: int,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_cache: LayerExpertCache,
) -> MoEExecutionPlan:
    """Build a fixed-route GPU plan once rerouting has produced cache-valid ids."""
    flat_selected = _flatten_experts(selected_experts)
    _ = routing_weights
    gpu_slots = expert_cache.expert_to_slot_lut.index_select(0, flat_selected)
    gpu_route_indices = torch.arange(flat_selected.numel(), dtype=torch.int64, device=flat_selected.device)
    m_sizes, gpu_route_indices = _build_grouped_layout(
        gpu_slots,
        gpu_route_indices,
        expert_cache.num_slots,
    )
    return MoEExecutionPlan(
        layer_idx=layer_idx,
        gpu_route_indices=gpu_route_indices,
        gpu_m_sizes=m_sizes,
        cpu_route_indices=None,
        cpu_task_expert_ids=None,
        cpu_task_offsets=None,
        cpu_task_expert_ids_host=None,
        cpu_task_offsets_host=None,
        flat_selected_original=flat_selected,
        flat_selected_effective=flat_selected,
        gpu_route_weights=None,
        cpu_graph_enabled=False,
        cpu_graph_async=False,
        substitution_lut=None,
        gpu_route_mask=torch.ones_like(flat_selected, dtype=torch.bool),
        cpu_route_mask=torch.zeros_like(flat_selected, dtype=torch.bool),
    )


def build_verify_plan_gpu(
    layer_idx: int,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_cache: LayerExpertCache,
    num_experts: int,
) -> MoEExecutionPlan:
    return build_prefill_plan_gpu(
        layer_idx=layer_idx,
        selected_experts=selected_experts,
        routing_weights=routing_weights,
        expert_cache=expert_cache,
        num_experts=num_experts,
    )


def build_moe_execution_plan(
    selected_experts: torch.Tensor,
    expert_cache: LayerExpertCache,
) -> MoEExecutionPlan:
    """Backward-compatible wrapper for prefill/standard heterogeneous planning."""
    fake_weights = torch.ones_like(selected_experts, dtype=torch.float32)
    return build_prefill_plan_gpu(
        layer_idx=-1,
        selected_experts=selected_experts,
        routing_weights=fake_weights,
        expert_cache=expert_cache,
        num_experts=expert_cache.num_experts,
    )
