#!/usr/bin/env python3
"""Measure PCIe bandwidth and contention between on-demand and prefetch DMA.

Tests:
  1. Single-stream bandwidth baseline
  2. Multi-stream concurrent bandwidth (saturation point)
  3. On-demand + prefetch contention: does parallel prefetch hurt on-demand latency?
"""

from __future__ import annotations

import time
from typing import Any

import torch


EXPERT_SIZE_BYTES = 9_437_184  # ~9 MiB per expert (bf16)
EXPERT_ELEMENTS = EXPERT_SIZE_BYTES // 2  # bf16 = 2 bytes/element


def _alloc_cpu_pinned(size_bytes: int) -> torch.Tensor:
    """Allocate a pinned CPU tensor of the given size in bytes."""
    n_elem = size_bytes // 2
    return torch.empty(n_elem, dtype=torch.bfloat16, pin_memory=True)


def _alloc_gpu(size_bytes: int) -> torch.Tensor:
    n_elem = size_bytes // 2
    return torch.empty(n_elem, dtype=torch.bfloat16, device="cuda")


def test_single_stream_bandwidth(
    num_transfers: int = 64,
    expert_bytes: int = EXPERT_SIZE_BYTES,
) -> dict[str, Any]:
    """Baseline: sequential DMA on a single stream."""
    cpu_tensors = [_alloc_cpu_pinned(expert_bytes) for _ in range(num_transfers)]
    gpu_tensors = [_alloc_gpu(expert_bytes) for _ in range(num_transfers)]

    # Warmup
    for ct, gt in zip(cpu_tensors[:4], gpu_tensors[:4]):
        gt.copy_(ct)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for ct, gt in zip(cpu_tensors, gpu_tensors):
        gt.copy_(ct, non_blocking=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_bytes = num_transfers * expert_bytes
    bandwidth_gibs = total_bytes / elapsed / (1024**3)

    del cpu_tensors, gpu_tensors
    return {
        "num_transfers": num_transfers,
        "expert_bytes": expert_bytes,
        "total_bytes": total_bytes,
        "elapsed_ms": elapsed * 1000,
        "bandwidth_gibs": bandwidth_gibs,
        "per_transfer_ms": elapsed * 1000 / num_transfers,
    }


def test_multi_stream_bandwidth(
    num_streams: int,
    transfers_per_stream: int = 16,
    expert_bytes: int = EXPERT_SIZE_BYTES,
) -> dict[str, Any]:
    """Launch N streams doing DMA in parallel, measure total throughput."""
    streams = [torch.cuda.Stream() for _ in range(num_streams)]
    cpu_tensors = [
        [_alloc_cpu_pinned(expert_bytes) for _ in range(transfers_per_stream)]
        for _ in range(num_streams)
    ]
    gpu_tensors = [
        [_alloc_gpu(expert_bytes) for _ in range(transfers_per_stream)]
        for _ in range(num_streams)
    ]

    # Warmup: all streams
    for s_idx in range(num_streams):
        with torch.cuda.stream(streams[s_idx]):
            for ct, gt in zip(cpu_tensors[s_idx][:2], gpu_tensors[s_idx][:2]):
                gt.copy_(ct, non_blocking=True)
    torch.cuda.synchronize()

    # Timed run: launch all streams concurrently
    t0 = time.perf_counter()
    for s_idx in range(num_streams):
        with torch.cuda.stream(streams[s_idx]):
            for ct, gt in zip(cpu_tensors[s_idx], gpu_tensors[s_idx]):
                gt.copy_(ct, non_blocking=True)
    torch.cuda.synchronize()  # wait for ALL streams
    elapsed = time.perf_counter() - t0

    total_transfers = num_streams * transfers_per_stream
    total_bytes = total_transfers * expert_bytes
    bandwidth_gibs = total_bytes / elapsed / (1024**3)

    del cpu_tensors, gpu_tensors, streams
    return {
        "num_streams": num_streams,
        "transfers_per_stream": transfers_per_stream,
        "total_transfers": total_transfers,
        "total_bytes": total_bytes,
        "elapsed_ms": elapsed * 1000,
        "bandwidth_gibs": bandwidth_gibs,
        "per_transfer_ms": elapsed * 1000 / total_transfers,
    }


def test_contention(
    on_demand_transfers: int = 64,
    prefetch_streams: int = 4,
    prefetch_transfers_per_stream: int = 8,
    expert_bytes: int = EXPERT_SIZE_BYTES,
) -> dict[str, Any]:
    """Simulate on-demand (default stream) + prefetch (separate streams) DMA.

    Measures whether prefetch DMA on separate streams increases
    the completion time of on-demand DMA on the default stream.
    """
    # --- Baseline: on-demand only ---
    od_cpu = [_alloc_cpu_pinned(expert_bytes) for _ in range(on_demand_transfers)]
    od_gpu = [_alloc_gpu(expert_bytes) for _ in range(on_demand_transfers)]

    # Warmup
    for ct, gt in zip(od_cpu[:4], od_gpu[:4]):
        gt.copy_(ct)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for ct, gt in zip(od_cpu, od_gpu):
        gt.copy_(ct, non_blocking=True)
    torch.cuda.synchronize()
    baseline_ms = (time.perf_counter() - t0) * 1000

    # --- Contention: on-demand + prefetch ---
    pf_streams = [torch.cuda.Stream() for _ in range(prefetch_streams)]
    pf_cpu = [
        [_alloc_cpu_pinned(expert_bytes) for _ in range(prefetch_transfers_per_stream)]
        for _ in range(prefetch_streams)
    ]
    pf_gpu = [
        [_alloc_gpu(expert_bytes) for _ in range(prefetch_transfers_per_stream)]
        for _ in range(prefetch_streams)
    ]

    # Re-allocate on-demand tensors to avoid cache effects
    od_cpu2 = [_alloc_cpu_pinned(expert_bytes) for _ in range(on_demand_transfers)]
    od_gpu2 = [_alloc_gpu(expert_bytes) for _ in range(on_demand_transfers)]

    # Warmup
    for ct, gt in zip(od_cpu2[:4], od_gpu2[:4]):
        gt.copy_(ct)
    torch.cuda.synchronize()

    # Interleave: on-demand (default stream) + prefetch (separate streams) concurrently
    t0 = time.perf_counter()

    # Launch prefetch first to simulate "running ahead"
    for s_idx in range(prefetch_streams):
        with torch.cuda.stream(pf_streams[s_idx]):
            for ct, gt in zip(pf_cpu[s_idx], pf_gpu[s_idx]):
                gt.copy_(ct, non_blocking=True)

    # Then on-demand on default stream
    for ct, gt in zip(od_cpu2, od_gpu2):
        gt.copy_(ct, non_blocking=True)

    torch.cuda.synchronize()
    contention_ms = (time.perf_counter() - t0) * 1000

    # Also measure just the on-demand portion (default stream only)
    # by using a CUDA event
    od_cpu3 = [_alloc_cpu_pinned(expert_bytes) for _ in range(on_demand_transfers)]
    od_gpu3 = [_alloc_gpu(expert_bytes) for _ in range(on_demand_transfers)]

    # Launch prefetch on separate streams
    for s_idx in range(prefetch_streams):
        with torch.cuda.stream(pf_streams[s_idx]):
            for ct, gt in zip(pf_cpu[s_idx], pf_gpu[s_idx]):
                gt.copy_(ct, non_blocking=True)

    # On-demand on default stream, timed with CUDA events
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for ct, gt in zip(od_cpu3, od_gpu3):
        gt.copy_(ct, non_blocking=True)
    end_ev.record()
    torch.cuda.synchronize()
    od_with_pf_ms = start_ev.elapsed_time(end_ev)

    del od_cpu, od_gpu, od_cpu2, od_gpu2, od_cpu3, od_gpu3
    del pf_cpu, pf_gpu, pf_streams

    return {
        "on_demand_transfers": on_demand_transfers,
        "prefetch_streams": prefetch_streams,
        "prefetch_transfers_per_stream": prefetch_transfers_per_stream,
        "total_prefetch_transfers": prefetch_streams * prefetch_transfers_per_stream,
        "baseline_od_only_ms": baseline_ms,
        "contention_total_ms": contention_ms,
        "od_with_prefetch_ms": od_with_pf_ms,
        "od_slowdown_pct": (od_with_pf_ms / baseline_ms - 1.0) * 100
        if baseline_ms > 0
        else 0.0,
    }


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    device = torch.cuda.get_device_name(0)
    print(f"GPU: {device}")
    print(f"Expert size: {EXPERT_SIZE_BYTES / 1024 / 1024:.1f} MiB (bf16)")
    print()

    # --- Test 1: Single stream bandwidth ---
    print("=== Test 1: Single-stream Bandwidth ===")
    r = test_single_stream_bandwidth(num_transfers=64)
    print(f"  {r['num_transfers']} transfers × {r['expert_bytes']/1024/1024:.0f} MiB = {r['total_bytes']/1024/1024:.0f} MiB")
    print(f"  Time: {r['elapsed_ms']:.1f} ms  ({r['per_transfer_ms']:.3f} ms/xfer)")
    print(f"  Bandwidth: {r['bandwidth_gibs']:.2f} GiB/s")
    print()

    # --- Test 2: Multi-stream bandwidth scaling ---
    print("=== Test 2: Multi-stream Bandwidth Scaling ===")
    print(f"{'Streams':<8} {'Total MiB':<12} {'Time ms':<12} {'ms/xfer':<12} {'GiB/s':<10}")
    print("-" * 54)
    results_multi = {}
    for n in [1, 2, 4, 8, 16]:
        r = test_multi_stream_bandwidth(num_streams=n, transfers_per_stream=8)
        results_multi[n] = r
        print(
            f"  {r['num_streams']:<8} "
            f"{r['total_bytes']/1024/1024:<12.0f} "
            f"{r['elapsed_ms']:<12.2f} "
            f"{r['per_transfer_ms']:<12.4f} "
            f"{r['bandwidth_gibs']:<10.2f}"
        )
    print()

    # --- Test 3: Contention test ---
    print("=== Test 3: On-demand + Prefetch Contention ===")
    print(f"{'PF streams':<12} {'OD only ms':<14} {'OD w/PF ms':<14} {'Slowdown %':<12}")
    print("-" * 52)
    for pf_streams in [1, 2, 4, 8]:
        r = test_contention(
            on_demand_transfers=32,
            prefetch_streams=pf_streams,
            prefetch_transfers_per_stream=8,
        )
        print(
            f"  {r['prefetch_streams']:<12} "
            f"{r['baseline_od_only_ms']:<14.2f} "
            f"{r['od_with_prefetch_ms']:<14.2f} "
            f"{r['od_slowdown_pct']:<+12.1f}"
        )
    print()

    # --- Test 4: Heavy contention (max stress) ---
    print("=== Test 4: Heavy Contention (many prefetch streams) ===")
    for pf_streams in [1, 2, 4, 8, 12, 16]:
        r = test_contention(
            on_demand_transfers=32,
            prefetch_streams=pf_streams,
            prefetch_transfers_per_stream=4,
        )
        print(
            f"  pf_streams={r['prefetch_streams']:<4} "
            f"od_only={r['baseline_od_only_ms']:.2f}ms "
            f"od_w_pf={r['od_with_prefetch_ms']:.2f}ms "
            f"slowdown={r['od_slowdown_pct']:+.1f}% "
            f"total_pf={r['total_prefetch_transfers']}"
        )


if __name__ == "__main__":
    main()
