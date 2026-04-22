#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

import torch

from nanovllm.expert.cache import LayerExpertCache
from nanovllm.expert.prefetcher import PrefetchRuntime
from nanovllm.expert.runtime_meta import LayerRuntimeMetaCPU
from nanovllm.scheduling.cache_strategy import create_cache_strategy
from nanovllm.scheduling.prefetch_strategy import create_prefetch_strategy


def make_cfg(enabled: bool) -> SimpleNamespace:
    step_budget = 8 if enabled else 0
    max_inflight = 16 if enabled else 0
    return SimpleNamespace(
        prefetch_step_budget=step_budget,
        prefetch_max_inflight=max_inflight,
        cache_eviction_budget_per_step=4,
        prefetch_verify_wait_ms=0.2 if enabled else 0.0,
        prefetch_source_weight_prefill=1.0,
        prefetch_source_weight_verify=1.2,
        prefetch_source_weight_draft=1.5,
        prefetch_activation_count_weight=0.1,
        prefetch_age_penalty=0.01,
        prefetch_history_decay=0.9,
        prefetch_history_ttl_steps=32,
        prefetch_global_queue_capacity=4096,
        prefetch_use_prefill_history=True,
        prefetch_use_verify_history=True,
        prefetch_use_draft_live=True,
    )


def build_runtime(cfg: SimpleNamespace, layers: int, experts: int, hidden: int) -> PrefetchRuntime:
    layer_caches = {}
    cpu_pool = {}
    for layer_idx in range(layers):
        pool = {}
        for expert_idx in range(experts):
            pool[expert_idx] = {
                "gate_up": torch.randn(hidden * 2, hidden, dtype=torch.float32),
                "down": torch.randn(hidden, hidden, dtype=torch.float32),
            }
        cache = LayerExpertCache(
            num_experts=experts,
            slots_per_layer=max(1, experts // 4),
            gate_up_shape=(hidden * 2, hidden),
            down_shape=(hidden, hidden),
            device=torch.device("cpu"),
            dtype=torch.float32,
            cpu_expert_pool=pool,
            staging_slots_per_layer=4,
            enable_prefetch=True,
        )
        for slot_idx in range(cache.num_slots):
            cache.put_to_slot(slot_idx, slot_idx, pool[slot_idx]["gate_up"], pool[slot_idx]["down"])
        layer_caches[layer_idx] = cache
        cpu_pool[layer_idx] = pool

    return PrefetchRuntime(
        config=cfg,
        layer_caches=layer_caches,
        cpu_expert_pool=cpu_pool,
        cache_strategy=create_cache_strategy("lru"),
        prefetch_strategy=create_prefetch_strategy("history_window", cfg),
        runtime_meta_recorder=SimpleNamespace(),
    )


def simulate(runtime: PrefetchRuntime, steps: int, layers: int, experts: int, tokens: int, top_k: int, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    step_ms = []
    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        runtime_meta = {}
        for layer_idx in range(layers):
            selected = torch.randint(0, experts, (tokens, top_k), generator=g, dtype=torch.int64)
            weights = torch.rand((tokens, top_k), generator=g, dtype=torch.float32)
            weights = weights / weights.sum(dim=-1, keepdim=True)
            runtime_meta[layer_idx] = LayerRuntimeMetaCPU(
                step_id=step,
                mode="draft",
                layer_idx=layer_idx,
                token_count=tokens,
                selected_experts=selected,
                routing_weights=weights,
            )
        runtime.observe_draft(runtime_meta, step_id=step)
        runtime.submit_from_global_queue(step_id=step, phase="before_draft")
        runtime.publish_ready(step_id=step)
        runtime.wait_for_verify(step_id=step, timeout_ms=float(runtime.config.prefetch_verify_wait_ms))
        step_ms.append((time.perf_counter() - t0) * 1000.0)

    profile = runtime.get_profile(reset=False)
    return {
        "avg_step_ms": float(mean(step_ms) if step_ms else 0.0),
        "profile": profile,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare prefetch enabled vs disabled runtime overhead")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    runtime_on = build_runtime(make_cfg(True), args.layers, args.experts, args.hidden)
    runtime_off = build_runtime(make_cfg(False), args.layers, args.experts, args.hidden)

    on = simulate(runtime_on, args.steps, args.layers, args.experts, args.tokens, args.top_k, args.seed)
    off = simulate(runtime_off, args.steps, args.layers, args.experts, args.tokens, args.top_k, args.seed)

    result = {
        "benchmark": "phase3_prefetch_onoff_compare",
        "enabled": on,
        "disabled": off,
        "delta_avg_step_ms": float(on["avg_step_ms"] - off["avg_step_ms"]),
        "params": {
            "steps": args.steps,
            "layers": args.layers,
            "experts": args.experts,
            "tokens": args.tokens,
            "top_k": args.top_k,
            "hidden": args.hidden,
        },
    }

    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
