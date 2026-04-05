import argparse
import json
import time
from statistics import median

import torch

from nanovllm.expert.cache import LayerExpertCache
from nanovllm.expert.placement import build_prefill_plan_gpu
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.fuse_moe.heterogeneous import heterogeneous_moe_forward


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = int(round((len(arr) - 1) * q))
    return float(arr[idx])


def _build_cpu_pool(num_experts: int, hidden_size: int, intermediate_size: int) -> dict[int, dict[str, torch.Tensor]]:
    pool: dict[int, dict[str, torch.Tensor]] = {}
    for eid in range(num_experts):
        gate_up = torch.randn(intermediate_size * 2, hidden_size, dtype=torch.float32)
        down = torch.randn(hidden_size, intermediate_size, dtype=torch.float32)
        pool[eid] = {"gate_up": gate_up, "down": down}
    return pool


def _build_cache(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    cached_expert_count: int,
    device: torch.device,
    cpu_pool: dict[int, dict[str, torch.Tensor]],
) -> LayerExpertCache:
    slots = max(1, cached_expert_count)
    cache = LayerExpertCache(
        num_experts=num_experts,
        slots_per_layer=slots,
        gate_up_shape=(intermediate_size * 2, hidden_size),
        down_shape=(hidden_size, intermediate_size),
        device=device,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
        cpu_expert_pool=cpu_pool,
    )

    if cached_expert_count == 0:
        cache.expert_to_slot_lut.fill_(-1)
        cache.cached_expert_mask.fill_(False)
        cache.slot_to_expert = [-1] * cache.num_slots
        cache.expert_to_slot.clear()
        return cache

    for slot_idx in range(cached_expert_count):
        params = cpu_pool[slot_idx]
        cache.put_to_slot(slot_idx, slot_idx, params["gate_up"], params["down"])
    return cache


def _build_controlled_routes(num_tokens: int, top_k: int, num_experts: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    total = num_tokens * top_k
    flat = torch.arange(total, device=device, dtype=torch.int64) % num_experts
    selected = flat.view(num_tokens, top_k)
    routing_weights = torch.rand(num_tokens, top_k, device=device, dtype=torch.float32)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    return selected, routing_weights


def run_once(
    hidden_states: torch.Tensor,
    selected: torch.Tensor,
    routing_weights: torch.Tensor,
    cache: LayerExpertCache,
    cpu_pool: dict[int, dict[str, torch.Tensor]],
    act_fn: SiluAndMul,
) -> tuple[torch.Tensor, dict]:
    plan = build_prefill_plan_gpu(
        layer_idx=0,
        selected_experts=selected,
        routing_weights=routing_weights,
        expert_cache=cache,
        num_experts=cache.num_experts,
    )
    profile: dict[str, float] = {}
    out = heterogeneous_moe_forward(
        hidden_states=hidden_states,
        selected_experts=selected,
        routing_weights=routing_weights.to(hidden_states.dtype),
        expert_cache=cache,
        cpu_expert_pool=cpu_pool,
        act_fn=act_fn,
        plan=plan,
        cpu_expert_execution_enabled=True,
        cpu_expert_parallel_mode="serial",
        cpu_expert_num_threads=1,
        profile=profile,
    )

    total_routes = float(selected.numel())
    cpu_routes = float(plan.cpu_route_indices.numel()) if plan.cpu_route_indices is not None else 0.0
    flat_weights = routing_weights.reshape(-1)
    cpu_mass = 0.0
    if plan.cpu_route_indices is not None and plan.cpu_route_indices.numel() > 0:
        cpu_mass = float(flat_weights.index_select(0, plan.cpu_route_indices).sum().item())
    total_mass = float(flat_weights.sum().item())

    profile["cpu_route_ratio"] = cpu_routes / total_routes if total_routes > 0 else 0.0
    profile["cpu_weight_mass_ratio"] = cpu_mass / total_mass if total_mass > 0 else 0.0
    profile["activated_expert_set_size"] = float(torch.unique(selected).numel())
    profile["realized_cpu_expert_count"] = (
        float(plan.cpu_task_expert_ids.numel()) if plan.cpu_task_expert_ids is not None else 0.0
    )
    return out, profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-layer synthetic MoE CPU/GPU parallel benchmark")
    parser.add_argument("--token-sizes", type=str, default="64,256")
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--intermediate-size", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    token_sizes = [int(x) for x in args.token_sizes.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    act_fn = SiluAndMul()
    cpu_pool = _build_cpu_pool(args.num_experts, args.hidden_size, args.intermediate_size)

    rows: list[dict] = []
    for num_tokens in token_sizes:
        hidden_states = torch.randn(num_tokens, args.hidden_size, device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32)
        selected, routing_weights = _build_controlled_routes(num_tokens, args.top_k, args.num_experts, device)

        for target_cpu_count in range(0, args.num_experts + 1):
            cached_count = max(0, args.num_experts - target_cpu_count)
            cache = _build_cache(
                num_experts=args.num_experts,
                hidden_size=args.hidden_size,
                intermediate_size=args.intermediate_size,
                cached_expert_count=cached_count,
                device=device,
                cpu_pool=cpu_pool,
            )

            for _ in range(args.warmup):
                _ = run_once(hidden_states, selected, routing_weights, cache, cpu_pool, act_fn)
            if device.type == "cuda":
                torch.cuda.synchronize()

            latencies: list[float] = []
            profile_acc: dict[str, float] = {}
            for _ in range(args.repeat):
                t0 = time.perf_counter()
                out, profile = run_once(hidden_states, selected, routing_weights, cache, cpu_pool, act_fn)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000.0)
                _ = float(out.sum().item())
                for key, value in profile.items():
                    profile_acc[key] = float(profile_acc.get(key, 0.0) + value)

            avg_profile = {k: v / max(1, args.repeat) for k, v in profile_acc.items()}
            row = {
                "num_tokens": num_tokens,
                "target_cpu_expert_count": target_cpu_count,
                "latency_ms_p50": median(latencies),
                "latency_ms_p90": _percentile(latencies, 0.9),
                "latency_ms_p99": _percentile(latencies, 0.99),
            }
            row.update(avg_profile)
            rows.append(row)

    report = {
        "config": {
            "token_sizes": token_sizes,
            "num_experts": args.num_experts,
            "top_k": args.top_k,
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "device": str(device),
        },
        "results": rows,
    }

    text = json.dumps(report, ensure_ascii=True, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main()
