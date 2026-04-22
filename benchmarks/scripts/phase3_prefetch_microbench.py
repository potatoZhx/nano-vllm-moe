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
from nanovllm.expert.runtime_meta import LayerRuntimeMetaCPU, ModelRuntimeMetaRecorder
from nanovllm.scheduling.cache_strategy import create_cache_strategy
from nanovllm.scheduling.prefetch_strategy import create_prefetch_strategy


def build_config() -> SimpleNamespace:
    return SimpleNamespace(
        prefetch_step_budget=8,
        prefetch_max_inflight=16,
        cache_eviction_budget_per_step=4,
        prefetch_verify_wait_ms=0.0,
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


def build_runtime(
    num_layers: int,
    num_experts: int,
    slots: int,
    staging_slots: int,
    hidden: int,
) -> tuple[PrefetchRuntime, dict[int, LayerExpertCache]]:
    cfg = build_config()
    layer_caches: dict[int, LayerExpertCache] = {}
    cpu_pool: dict[int, dict[int, dict[str, torch.Tensor]]] = {}

    for layer_idx in range(num_layers):
        pool: dict[int, dict[str, torch.Tensor]] = {}
        for expert_idx in range(num_experts):
            pool[expert_idx] = {
                "gate_up": torch.randn(hidden * 2, hidden, dtype=torch.float32),
                "down": torch.randn(hidden, hidden, dtype=torch.float32),
            }
        cache = LayerExpertCache(
            num_experts=num_experts,
            slots_per_layer=slots,
            gate_up_shape=(hidden * 2, hidden),
            down_shape=(hidden, hidden),
            device=torch.device("cpu"),
            dtype=torch.float32,
            cpu_expert_pool=pool,
            staging_slots_per_layer=staging_slots,
            enable_prefetch=True,
        )
        for slot_idx in range(slots):
            cache.put_to_slot(slot_idx, slot_idx, pool[slot_idx]["gate_up"], pool[slot_idx]["down"])
        layer_caches[layer_idx] = cache
        cpu_pool[layer_idx] = pool

    runtime = PrefetchRuntime(
        config=cfg,
        layer_caches=layer_caches,
        cpu_expert_pool=cpu_pool,
        cache_strategy=create_cache_strategy("lru"),
        prefetch_strategy=create_prefetch_strategy("history_window", cfg),
        runtime_meta_recorder=SimpleNamespace(),
    )
    return runtime, layer_caches


def run_runtime_bench(args) -> dict:
    runtime, _ = build_runtime(
        num_layers=args.layers,
        num_experts=args.experts,
        slots=args.slots,
        staging_slots=args.staging_slots,
        hidden=args.hidden,
    )
    g = torch.Generator().manual_seed(args.seed)

    observe_ms = []
    submit_ms = []
    publish_ms = []

    for step in range(1, args.steps + 1):
        runtime_meta = {}
        for layer_idx in range(args.layers):
            selected = torch.randint(
                low=0,
                high=args.experts,
                size=(args.tokens, args.top_k),
                generator=g,
                dtype=torch.int64,
            )
            weights = torch.rand((args.tokens, args.top_k), generator=g, dtype=torch.float32)
            weights = weights / weights.sum(dim=-1, keepdim=True)
            runtime_meta[layer_idx] = LayerRuntimeMetaCPU(
                step_id=step,
                mode="draft",
                layer_idx=layer_idx,
                token_count=args.tokens,
                selected_experts=selected,
                routing_weights=weights,
            )

        t0 = time.perf_counter()
        runtime.observe_draft(runtime_meta, step_id=step)
        observe_ms.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        runtime.submit_from_global_queue(step_id=step, phase="before_draft")
        submit_ms.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        runtime.publish_ready(step_id=step)
        publish_ms.append((time.perf_counter() - t0) * 1000.0)

    profile = runtime.get_profile(reset=False)
    return {
        "observe_avg_ms": float(mean(observe_ms) if observe_ms else 0.0),
        "submit_avg_ms": float(mean(submit_ms) if submit_ms else 0.0),
        "publish_avg_ms": float(mean(publish_ms) if publish_ms else 0.0),
        "profile": profile,
    }


def run_recorder_bench(args) -> dict:
    recorder = ModelRuntimeMetaRecorder(
        config=SimpleNamespace(),
        hf_config=SimpleNamespace(num_hidden_layers=args.layers, num_experts_per_tok=args.top_k),
    )
    g = torch.Generator().manual_seed(args.seed)

    times = []
    for step in range(args.steps):
        t0 = time.perf_counter()
        recorder.arm(
            mode="draft",
            step_id=step,
            token_capacity=args.tokens,
            logical_token_count=args.tokens,
        )
        for layer_idx in range(args.layers):
            selected = torch.randint(
                low=0,
                high=args.experts,
                size=(args.tokens, args.top_k),
                generator=g,
                dtype=torch.int64,
            )
            weights = torch.rand((args.tokens, args.top_k), generator=g, dtype=torch.float32)
            recorder.record_layer(layer_idx, selected, weights)
        handle = recorder.offload_async(stream=None)
        _ = recorder.collect(handle, wait=True)
        recorder.reset()
        times.append((time.perf_counter() - t0) * 1000.0)

    return {
        "record_collect_avg_ms": float(mean(times) if times else 0.0),
        "buffer_bytes": int(handle.buffer_bytes if handle is not None else 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 prefetch micro benchmark")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--staging-slots", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = {
        "benchmark": "phase3_prefetch_microbench",
        "runtime": run_runtime_bench(args),
        "recorder": run_recorder_bench(args),
        "params": {
            "steps": args.steps,
            "layers": args.layers,
            "experts": args.experts,
            "slots": args.slots,
            "staging_slots": args.staging_slots,
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
