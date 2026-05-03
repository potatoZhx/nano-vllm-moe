from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import torch

from nanovllm.expert.cache import LayerExpertCache
from nanovllm.expert.cpu_weights import CpuExpertWeights
from nanovllm.expert.placement import build_prefill_plan_gpu
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.fuse_moe.cpu_backend import TorchPackedCpuMoeBackend
from nanovllm.layers.fuse_moe.heterogeneous import heterogeneous_moe_forward


def _parse_list(raw: str, cast):
    return [cast(x) for x in raw.split(",") if x]


def _make_case(
    *,
    num_tokens: int,
    top_k: int,
    cpu_route_ratio: float,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: torch.dtype,
    device: torch.device,
):
    cpu_pool: dict[int, dict[str, object]] = {}
    for expert_idx in range(num_experts):
        gate_up = (torch.randn(intermediate_size * 2, hidden_size, dtype=dtype) * 0.02).contiguous()
        down = (torch.randn(hidden_size, intermediate_size, dtype=dtype) * 0.02).contiguous()
        packed = CpuExpertWeights(expert_idx=expert_idx, gate_up=gate_up, down=down, dtype=dtype)
        packed.validate()
        cpu_pool[expert_idx] = {"gate_up": gate_up, "down": down, "packed": packed}

    cached_experts = max(1, min(num_experts - 1, round(num_experts * (1.0 - cpu_route_ratio))))
    cache = LayerExpertCache(
        num_experts=num_experts,
        slots_per_layer=cached_experts,
        gate_up_shape=(intermediate_size * 2, hidden_size),
        down_shape=(hidden_size, intermediate_size),
        device=device,
        dtype=dtype,
        cpu_expert_pool=cpu_pool,
    )
    for slot in range(cache.num_slots):
        params = cpu_pool[slot]
        cache.put_to_slot(slot, slot, params["gate_up"], params["down"])

    num_routes = num_tokens * top_k
    num_cpu_routes = int(round(num_routes * cpu_route_ratio))
    cpu_expert_ids = torch.arange(cached_experts, num_experts, device=device, dtype=torch.int64)
    gpu_expert_ids = torch.arange(cached_experts, device=device, dtype=torch.int64)
    route_experts = torch.empty(num_routes, device=device, dtype=torch.int64)
    if num_cpu_routes > 0:
        route_experts[:num_cpu_routes] = cpu_expert_ids[torch.arange(num_cpu_routes, device=device) % cpu_expert_ids.numel()]
    if num_cpu_routes < num_routes:
        gpu_count = num_routes - num_cpu_routes
        route_experts[num_cpu_routes:] = gpu_expert_ids[torch.arange(gpu_count, device=device) % gpu_expert_ids.numel()]

    selected_experts = route_experts.view(num_tokens, top_k)
    hidden_states = (torch.randn(num_tokens, hidden_size, device=device, dtype=dtype) * 0.1).contiguous()
    routing_weights = torch.rand(num_tokens, top_k, device=device, dtype=torch.float32)
    routing_weights = (routing_weights / routing_weights.sum(dim=-1, keepdim=True)).to(dtype)
    plan = build_prefill_plan_gpu(
        layer_idx=0,
        selected_experts=selected_experts,
        routing_weights=routing_weights,
        expert_cache=cache,
        num_experts=num_experts,
    )
    return cpu_pool, cache, hidden_states, selected_experts, routing_weights, plan


def _run_backend(
    *,
    backend_name: str,
    cpu_pool,
    cache,
    hidden_states,
    selected_experts,
    routing_weights,
    plan,
    iterations: int,
    warmup: int,
    packed_min_routes: int,
):
    cpu_backend = None
    if backend_name == "torch_packed":
        cpu_backend = TorchPackedCpuMoeBackend(
            layer_idx=0,
            cpu_expert_pool=cpu_pool,
            max_routes=selected_experts.numel(),
            strict_dtype=True,
        )
    elif backend_name != "torch":
        raise ValueError(f"Unsupported backend: {backend_name}")

    act_fn = SiluAndMul()
    elapsed_ms = []
    profiles: list[dict[str, float]] = []
    last = None
    for step in range(warmup + iterations):
        profile: dict[str, float] = {}
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        last = heterogeneous_moe_forward(
            hidden_states=hidden_states,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            expert_cache=cache,
            cpu_expert_pool=cpu_pool,
            act_fn=act_fn,
            plan=plan,
            cpu_expert_execution_enabled=True,
            cpu_backend=cpu_backend,
            cpu_backend_min_routes=packed_min_routes,
            profile=profile,
        )
        torch.cuda.synchronize()
        if step >= warmup:
            elapsed_ms.append((time.perf_counter() - t0) * 1000.0)
            profiles.append(profile)

    assert last is not None
    mean_profile = {
        key: statistics.mean(float(p.get(key, 0.0)) for p in profiles)
        for key in [
            "cpu_prepare_ms",
            "cpu_compute_ms",
            "cpu_to_gpu_merge_ms",
            "gpu_gather_ms",
            "gpu_compute_ms",
            "scatter_ms",
        ]
    }
    return last, statistics.mean(elapsed_ms), mean_profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", action="append", choices=["torch", "torch_packed"], default=None)
    parser.add_argument("--tokens", default="1,8,32,128")
    parser.add_argument("--cpu-route-ratio", default="0.25,0.5,0.75")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--intermediate-size", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--packed-min-routes", type=int, default=32)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("bench_cpu_moe_backend.py requires CUDA")

    torch.manual_seed(1234)
    backends = args.backend or ["torch", "torch_packed"]
    rows = []
    for tokens in _parse_list(args.tokens, int):
        for ratio in _parse_list(args.cpu_route_ratio, float):
            cpu_pool, cache, hidden_states, selected_experts, routing_weights, plan = _make_case(
                num_tokens=tokens,
                top_k=args.top_k,
                cpu_route_ratio=ratio,
                num_experts=args.num_experts,
                hidden_size=args.hidden_size,
                intermediate_size=args.intermediate_size,
                dtype=torch.bfloat16,
                device=torch.device("cuda"),
            )
            ref_out = None
            for backend in backends:
                out, decode_ms, prof = _run_backend(
                    backend_name=backend,
                    cpu_pool=cpu_pool,
                    cache=cache,
                    hidden_states=hidden_states,
                    selected_experts=selected_experts,
                    routing_weights=routing_weights,
                    plan=plan,
                    iterations=args.iterations,
                    warmup=args.warmup,
                    packed_min_routes=args.packed_min_routes,
                )
                if ref_out is None:
                    ref_out = out.detach()
                    max_abs = 0.0
                    max_rel = 0.0
                else:
                    diff = (ref_out.float() - out.float()).abs()
                    max_abs = float(diff.max().item())
                    max_rel = float((diff / ref_out.float().abs().clamp_min(1e-5)).max().item())
                row = {
                    "backend": backend,
                    "mode": "block",
                    "batch_tokens": tokens,
                    "top_k": args.top_k,
                    "cpu_route_ratio": ratio,
                    "activated_cpu_experts": int(plan.cpu_task_expert_ids.numel()) if plan.cpu_task_expert_ids is not None else 0,
                    "decode_forward_ms": decode_ms,
                    "tokens_per_sec": float(tokens / (decode_ms / 1000.0)) if decode_ms > 0 else 0.0,
                    "max_abs": max_abs,
                    "max_rel": max_rel,
                }
                row.update(prof)
                rows.append(row)
                print(row)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
