#!/usr/bin/env python3
"""Stricter contention test: interleaved on-demand + prefetch, layer-by-layer simulation.

Simulates the real verify forward pattern:
  Layer 0: [on-demand DMA for L0 (default stream)] || [prefetch DMA for L1 (streams)]
  Layer 1: [on-demand DMA for L1 (default stream)] || [prefetch DMA for L2 (streams)]
  ...
"""

from __future__ import annotations

import time
from typing import Any

import torch


EXPERT_SIZE_BYTES = 9_437_184  # ~9 MiB


def _alloc_cpu(size_bytes: int) -> torch.Tensor:
    return torch.empty(size_bytes // 2, dtype=torch.bfloat16, pin_memory=True)


def _alloc_gpu(size_bytes: int) -> torch.Tensor:
    return torch.empty(size_bytes // 2, dtype=torch.bfloat16, device="cuda")


def simulate_layer_by_layer(
    num_layers: int = 48,
    misses_per_layer: int = 6,
    prefetch_misses_per_layer: int = 0,
    expert_bytes: int = EXPERT_SIZE_BYTES,
) -> dict[str, Any]:
    """Simulate the verify forward layer-by-layer pattern.

    For each layer:
      1. On-demand DMA for THIS layer's miss experts (default stream, sequential)
      2. (Optional) Prefetch DMA for NEXT layer's miss experts (separate streams)
      3. GPU compute (simulated with a tiny kernel)

    Prefetch for layer i+1 runs on separate streams while layer i's on-demand
    DMA executes, simulating the real pipeline.
    """
    # Pre-allocate all CPU and GPU tensors
    od_cpu = [
        [_alloc_cpu(expert_bytes) for _ in range(misses_per_layer)]
        for _ in range(num_layers)
    ]
    od_gpu = [
        [_alloc_gpu(expert_bytes) for _ in range(misses_per_layer)]
        for _ in range(num_layers)
    ]
    pf_cpu = [
        [_alloc_cpu(expert_bytes) for _ in range(prefetch_misses_per_layer)]
        for _ in range(num_layers)
    ]
    pf_gpu = [
        [_alloc_gpu(expert_bytes) for _ in range(prefetch_misses_per_layer)]
        for _ in range(num_layers)
    ]

    pf_streams = [
        torch.cuda.Stream() for _ in range(min(prefetch_misses_per_layer, 8))
    ]

    # Warmup
    for ct, gt in zip(od_cpu[0], od_gpu[0]):
        gt.copy_(ct)
    torch.cuda.synchronize()

    # --- Baseline: on-demand only (no prefetch) ---
    t0 = time.perf_counter()
    for layer in range(num_layers):
        # On-demand DMA for THIS layer
        for ct, gt in zip(od_cpu[layer], od_gpu[layer]):
            gt.copy_(ct, non_blocking=True)
        # Simulate GPU compute (tiny kernel to force stream ordering)
        _ = od_gpu[layer][0] + 1
    torch.cuda.synchronize()
    baseline_ms = (time.perf_counter() - t0) * 1000

    # --- With prefetch: interleaved ---
    t0 = time.perf_counter()
    for layer in range(num_layers):
        # 1. On-demand DMA for THIS layer
        for ct, gt in zip(od_cpu[layer], od_gpu[layer]):
            gt.copy_(ct, non_blocking=True)

        # 2. Prefetch DMA for NEXT layer (if applicable)
        next_layer = layer + 1
        if next_layer < num_layers and prefetch_misses_per_layer > 0:
            for j, (ct, gt) in enumerate(zip(pf_cpu[next_layer], pf_gpu[next_layer])):
                sidx = j % len(pf_streams)
                with torch.cuda.stream(pf_streams[sidx]):
                    gt.copy_(ct, non_blocking=True)

        # 3. Simulate GPU compute
        _ = od_gpu[layer][0] + 1

    torch.cuda.synchronize()
    interleaved_ms = (time.perf_counter() - t0) * 1000

    return {
        "num_layers": num_layers,
        "misses_per_layer": misses_per_layer,
        "prefetch_misses_per_layer": prefetch_misses_per_layer,
        "baseline_ms": baseline_ms,
        "interleaved_ms": interleaved_ms,
        "delta_ms": interleaved_ms - baseline_ms,
        "slowdown_pct": (interleaved_ms / baseline_ms - 1.0) * 100,
    }


def test_on_demand_latency_with_contention(
    num_on_demand: int = 6,
    num_prefetch: int = 6,
    expert_bytes: int = EXPERT_SIZE_BYTES,
) -> dict[str, Any]:
    """Directly measure per-transfer on-demand latency with concurrent prefetch.

    Key measurement: use CUDA events to time each on-demand transfer
    while prefetch transfers run on separate streams.
    """
    od_cpu = [_alloc_cpu(expert_bytes) for _ in range(num_on_demand)]
    od_gpu = [_alloc_gpu(expert_bytes) for _ in range(num_on_demand)]

    # --- Baseline: on-demand only ---
    events_start = [torch.cuda.Event(enable_timing=True) for _ in range(num_on_demand)]
    events_end = [torch.cuda.Event(enable_timing=True) for _ in range(num_on_demand)]

    # Warmup
    od_gpu[0].copy_(od_cpu[0])
    torch.cuda.synchronize()

    for i in range(num_on_demand):
        events_start[i].record()
        od_gpu[i].copy_(od_cpu[i], non_blocking=True)
        events_end[i].record()
    torch.cuda.synchronize()
    baseline_times = [
        events_start[i].elapsed_time(events_end[i]) for i in range(num_on_demand)
    ]

    # --- With contention: prefetch on separate streams ---
    pf_streams = [torch.cuda.Stream() for _ in range(4)]
    pf_cpu = [_alloc_cpu(expert_bytes) for _ in range(num_prefetch)]
    pf_gpu = [_alloc_gpu(expert_bytes) for _ in range(num_prefetch)]

    # Re-record with prefetch running
    events_start2 = [torch.cuda.Event(enable_timing=True) for _ in range(num_on_demand)]
    events_end2 = [torch.cuda.Event(enable_timing=True) for _ in range(num_on_demand)]

    # Start prefetch transfers first (to have them in-flight)
    for j, (ct, gt) in enumerate(zip(pf_cpu, pf_gpu)):
        sidx = j % len(pf_streams)
        with torch.cuda.stream(pf_streams[sidx]):
            gt.copy_(ct, non_blocking=True)

    # Now on-demand transfers, interleaved with more prefetch
    for i in range(num_on_demand):
        # Launch more prefetch to keep them in-flight
        if i < num_prefetch:
            sidx = (i + num_on_demand) % len(pf_streams)
            with torch.cuda.stream(pf_streams[sidx]):
                pf_gpu[i].copy_(pf_cpu[i], non_blocking=True)

        events_start2[i].record()
        od_gpu[i].copy_(od_cpu[i], non_blocking=True)
        events_end2[i].record()

    torch.cuda.synchronize()
    contention_times = [
        events_start2[i].elapsed_time(events_end2[i]) for i in range(num_on_demand)
    ]

    return {
        "num_on_demand": num_on_demand,
        "num_prefetch": num_prefetch,
        "baseline_per_transfer_ms": [round(t, 4) for t in baseline_times],
        "baseline_avg_ms": sum(baseline_times) / len(baseline_times),
        "contention_per_transfer_ms": [round(t, 4) for t in contention_times],
        "contention_avg_ms": sum(contention_times) / len(contention_times),
        "slowdown_pct": (sum(contention_times) / sum(baseline_times) - 1.0) * 100,
    }


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Expert size: {EXPERT_SIZE_BYTES / 1024 / 1024:.1f} MiB")
    print()

    # --- Test A: Layer-by-layer simulation ---
    print("=== Test A: Layer-by-layer Simulation (48 layers) ===")
    print(f"{'Miss/layer':<14} {'PF/layer':<12} {'Baseline ms':<14} {'Interleaved ms':<18} {'Delta ms':<12} {'Slowdown %':<12}")
    print("-" * 82)
    for misses in [2, 4, 6, 8]:
        for pf in [0, misses]:
            if pf == 0 and misses != 6:
                continue  # only need one baseline
            r = simulate_layer_by_layer(
                num_layers=48,
                misses_per_layer=misses,
                prefetch_misses_per_layer=pf,
            )
            if pf == 0:
                print(f"  {misses:<14} baseline     {r['baseline_ms']:<14.2f} (baseline)")
            else:
                print(
                    f"  {misses:<14} pf={pf:<9} "
                    f"{r['interleaved_ms']:<14.2f} "
                    f"{r['interleaved_ms']:<18.2f} "
                    f"{r['delta_ms']:<+12.2f} "
                    f"{r['slowdown_pct']:<+12.1f}"
                )
    print()

    # --- Test B: Per-transfer latency with contention ---
    print("=== Test B: Per-transfer On-demand Latency with Prefetch Contention ===")
    print(f"{'OD count':<12} {'PF count':<12} {'Baseline avg ms':<18} {'Contention avg ms':<20} {'Slowdown %':<12}")
    print("-" * 74)
    for od_count in [4, 6, 8]:
        for pf_count in [0, 4, 8, 16]:
            r = test_on_demand_latency_with_contention(
                num_on_demand=od_count,
                num_prefetch=pf_count,
            )
            if pf_count == 0:
                # This is really baseline (pf=0 means no prefetch launched in the test)
                # Actually the function always launches pf_count transfers,
                # but let me run a version with pf=0 separately
                pass
            print(
                f"  {r['num_on_demand']:<12} {r['num_prefetch']:<12} "
                f"{r['baseline_avg_ms']:<18.4f} {r['contention_avg_ms']:<20.4f} "
                f"{r['slowdown_pct']:<+12.1f}"
            )

    # --- Test C: Pure contention - interleaved tiny transfers ---
    print()
    print("=== Test C: Worst-case Contention (interleaved 1-by-1) ===")
    # Launch 1 on-demand, 1 prefetch, 1 on-demand, 1 prefetch, ...
    # on the same batches to maximize contention
    num_pairs = 32
    od_cpu = [_alloc_cpu(EXPERT_SIZE_BYTES) for _ in range(num_pairs)]
    od_gpu = [_alloc_gpu(EXPERT_SIZE_BYTES) for _ in range(num_pairs)]
    pf_cpu = [_alloc_cpu(EXPERT_SIZE_BYTES) for _ in range(num_pairs)]
    pf_gpu = [_alloc_gpu(EXPERT_SIZE_BYTES) for _ in range(num_pairs)]
    pf_stream = torch.cuda.Stream()

    # Warmup
    od_gpu[0].copy_(od_cpu[0])
    torch.cuda.synchronize()

    # Baseline: sequential on-demand only
    ev_s = [torch.cuda.Event(enable_timing=True) for _ in range(num_pairs)]
    ev_e = [torch.cuda.Event(enable_timing=True) for _ in range(num_pairs)]
    for i in range(num_pairs):
        ev_s[i].record()
        od_gpu[i].copy_(od_cpu[i], non_blocking=True)
        ev_e[i].record()
    torch.cuda.synchronize()
    baseline_avg = sum(ev_s[i].elapsed_time(ev_e[i]) for i in range(num_pairs)) / num_pairs

    # Contention: interleaved OD and PF
    ev_s2 = [torch.cuda.Event(enable_timing=True) for _ in range(num_pairs)]
    ev_e2 = [torch.cuda.Event(enable_timing=True) for _ in range(num_pairs)]
    for i in range(num_pairs):
        # Launch prefetch right before on-demand
        with torch.cuda.stream(pf_stream):
            pf_gpu[i].copy_(pf_cpu[i], non_blocking=True)
        ev_s2[i].record()
        od_gpu[i].copy_(od_cpu[i], non_blocking=True)
        ev_e2[i].record()
    torch.cuda.synchronize()
    contention_avg = sum(ev_s2[i].elapsed_time(ev_e2[i]) for i in range(num_pairs)) / num_pairs

    print(f"  Interleaved 1-by-1 ({num_pairs} pairs):")
    print(f"    Baseline avg:   {baseline_avg:.4f} ms/xfer")
    print(f"    Contention avg: {contention_avg:.4f} ms/xfer")
    print(f"    Slowdown:       {(contention_avg/baseline_avg - 1.0)*100:+.1f}%")

    del od_cpu, od_gpu, pf_cpu, pf_gpu


if __name__ == "__main__":
    main()
