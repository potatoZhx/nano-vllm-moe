#!/usr/bin/env python3
"""Microbenchmark KTransformers cpuinfer_ext.moe.MOE on Qwen3-30B-A3B MoE shapes."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch


QWEN3_EXPERT_NUM = 128
QWEN3_HIDDEN_SIZE = 2048
QWEN3_INTERMEDIATE_SIZE = 768
QWEN3_TOPK = 8
QWEN3_LAYERS = 48

GGML_TYPE_BF16 = 30
GGML_TYPE_F16 = 1


def _add_ktransformers_paths(root: str) -> None:
    root_path = Path(root).expanduser().resolve()
    candidates = [
        root_path / "build" / "lib.linux-x86_64-cpython-312",
        root_path,
    ]
    for path in candidates:
        if path.exists():
            sys.path.insert(0, str(path))


def _system_info() -> dict[str, object]:
    info: dict[str, object] = {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return info


def _split_threads(total: int, pools: int) -> list[int]:
    base, remainder = divmod(int(total), int(pools))
    return [base + (1 if pool_idx < remainder else 0) for pool_idx in range(pools)]


def _parse_numa_nodes(value: str, pools: int) -> list[int]:
    if value:
        nodes = [int(item) for item in value.split(",") if item.strip()]
    else:
        nodes = list(range(int(pools)))
    if len(nodes) != int(pools):
        raise ValueError("--numa-nodes length must equal --threadpool-count")
    return nodes


def _make_cpuinfer(cpuinfer_ext, threads: int, threadpool_count: int, numa_nodes: str):
    pools = int(threadpool_count)
    if pools == 1 and not numa_nodes:
        return cpuinfer_ext.CPUInfer(int(threads))
    worker_config = cpuinfer_ext.WorkerPoolConfig()
    worker_config.subpool_count = pools
    worker_config.subpool_numa_map = _parse_numa_nodes(numa_nodes, pools)
    worker_config.subpool_thread_count = _split_threads(int(threads), pools)
    return cpuinfer_ext.CPUInfer(worker_config)


def _random_expert_ids(
    pool_size: int,
    qlen: int,
    topk: int,
    expert_num: int,
    cpu_route_fraction: float,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(int(seed))
    expert_ids = (
        torch.rand(pool_size * qlen, expert_num, generator=generator)
        .argsort(dim=-1)[:, :topk]
        .reshape(pool_size, qlen, topk)
        .contiguous()
    )
    keep_count = round(int(qlen) * int(topk) * float(cpu_route_fraction))
    if keep_count >= int(qlen) * int(topk):
        return expert_ids
    for pool_idx in range(int(pool_size)):
        flat_ids = expert_ids[pool_idx].view(-1)
        keep = torch.randperm(flat_ids.numel(), generator=generator)[:keep_count]
        mask = torch.ones(flat_ids.numel(), dtype=torch.bool)
        mask[keep] = False
        flat_ids[mask] = -1
    return expert_ids


def _make_forward_task(moe, qlen_tensor, topk, expert_ids, weights, input_tensor, output_tensor):
    try:
        return moe.forward(
            qlen_tensor.data_ptr(),
            topk,
            expert_ids.data_ptr(),
            weights.data_ptr(),
            input_tensor.data_ptr(),
            output_tensor.data_ptr(),
            False,
        )
    except TypeError:
        return moe.forward(
            qlen_tensor.data_ptr(),
            topk,
            expert_ids.data_ptr(),
            weights.data_ptr(),
            input_tensor.data_ptr(),
            output_tensor.data_ptr(),
        )


def _load_moes(
    cpuinfer_ext,
    cpuinfer,
    layer_num: int,
    group_min_len: int,
    group_max_len: int,
    m_block: int,
    weight_dtype: str,
):
    torch_weight_dtype = torch.float16 if weight_dtype == "f16" else torch.bfloat16
    ggml_weight_type = GGML_TYPE_F16 if weight_dtype == "f16" else GGML_TYPE_BF16
    moes = []
    weights_keepalive = []
    for layer_idx in range(layer_num):
        gate_w = torch.randn(
            QWEN3_EXPERT_NUM,
            QWEN3_INTERMEDIATE_SIZE,
            QWEN3_HIDDEN_SIZE,
            dtype=torch_weight_dtype,
        ).contiguous()
        up_w = torch.randn(
            QWEN3_EXPERT_NUM,
            QWEN3_INTERMEDIATE_SIZE,
            QWEN3_HIDDEN_SIZE,
            dtype=torch_weight_dtype,
        ).contiguous()
        down_w = torch.randn(
            QWEN3_EXPERT_NUM,
            QWEN3_HIDDEN_SIZE,
            QWEN3_INTERMEDIATE_SIZE,
            dtype=torch_weight_dtype,
        ).contiguous()

        config = cpuinfer_ext.moe.MOEConfig(
            QWEN3_EXPERT_NUM,
            QWEN3_TOPK,
            QWEN3_HIDDEN_SIZE,
            QWEN3_INTERMEDIATE_SIZE,
        )
        config.layer_idx = layer_idx
        config.pool = cpuinfer.backend_
        config.group_min_len = int(group_min_len)
        config.group_max_len = int(group_max_len)
        config.m_block = int(m_block)
        config.gate_proj = gate_w.data_ptr()
        config.up_proj = up_w.data_ptr()
        config.down_proj = down_w.data_ptr()
        config.gate_type = ggml_weight_type
        config.up_type = ggml_weight_type
        config.down_type = ggml_weight_type
        config.hidden_type = GGML_TYPE_BF16

        moe = cpuinfer_ext.moe.MOE(config)
        cpuinfer.submit(moe.load_weights())
        cpuinfer.sync()
        moes.append(moe)
        weights_keepalive.append((gate_w, up_w, down_w))
    return moes, weights_keepalive


def bench_one(cpuinfer_ext, args, qlen: int) -> dict[str, object]:
    cpuinfer = _make_cpuinfer(
        cpuinfer_ext,
        args.threads,
        args.threadpool_count,
        args.numa_nodes,
    )
    moes, weights_keepalive = _load_moes(
        cpuinfer_ext,
        cpuinfer,
        args.layer_num,
        args.group_min_len,
        args.group_max_len,
        args.m_block,
        args.weight_dtype,
    )
    _ = weights_keepalive

    pool_size = max(args.warmup, args.iters, 256)
    expert_ids = _random_expert_ids(
        pool_size,
        qlen,
        QWEN3_TOPK,
        QWEN3_EXPERT_NUM,
        args.cpu_route_fraction,
        args.seed,
    )
    routing_weights = torch.rand(pool_size, qlen, QWEN3_TOPK, dtype=torch.float32).contiguous()
    input_tensor = torch.randn(
        args.layer_num,
        qlen,
        QWEN3_HIDDEN_SIZE,
        dtype=torch.bfloat16,
    ).contiguous()
    output_tensor = torch.empty_like(input_tensor)
    qlen_tensor = torch.tensor([qlen], dtype=torch.int32)

    for i in range(args.warmup):
        layer_idx = i % args.layer_num
        pool_idx = i % pool_size
        task = _make_forward_task(
            moes[layer_idx],
            qlen_tensor,
            QWEN3_TOPK,
            expert_ids[pool_idx],
            routing_weights[pool_idx],
            input_tensor[layer_idx],
            output_tensor[layer_idx],
        )
        cpuinfer.submit(task)
        cpuinfer.sync()

    lat_us = []
    t_total_start = time.perf_counter()
    for i in range(args.iters):
        layer_idx = i % args.layer_num
        pool_idx = i % pool_size
        t0 = time.perf_counter()
        task = _make_forward_task(
            moes[layer_idx],
            qlen_tensor,
            QWEN3_TOPK,
            expert_ids[pool_idx],
            routing_weights[pool_idx],
            input_tensor[layer_idx],
            output_tensor[layer_idx],
        )
        cpuinfer.submit(task)
        cpuinfer.sync()
        lat_us.append((time.perf_counter() - t0) * 1e6)
    total_s = time.perf_counter() - t_total_start

    lat_sorted = sorted(lat_us)
    routes_per_call = round(qlen * QWEN3_TOPK * args.cpu_route_fraction)
    avg_us = sum(lat_us) / len(lat_us)
    flops_per_call = 3 * 2 * routes_per_call * QWEN3_HIDDEN_SIZE * QWEN3_INTERMEDIATE_SIZE
    return {
        "system": "ktransformers_cpuinfer_ext_moe",
        "backend": "llamafile",
        "dtype": args.weight_dtype,
        "weight_dtype": args.weight_dtype,
        "hidden_dtype": "bf16",
        "qlen": qlen,
        "threads": args.threads,
        "threadpool_count": args.threadpool_count,
        "numa_nodes": _parse_numa_nodes(args.numa_nodes, args.threadpool_count),
        "layer_num": args.layer_num,
        "warmup_iters": args.warmup,
        "iters": args.iters,
        "expert_num": QWEN3_EXPERT_NUM,
        "topk": QWEN3_TOPK,
        "hidden_size": QWEN3_HIDDEN_SIZE,
        "intermediate_size": QWEN3_INTERMEDIATE_SIZE,
        "routes_per_call": routes_per_call,
        "cpu_route_fraction": args.cpu_route_fraction,
        "avg_latency_us": avg_us,
        "p50_latency_us": lat_sorted[len(lat_sorted) // 2],
        "p95_latency_us": lat_sorted[int(len(lat_sorted) * 0.95)],
        "min_latency_us": min(lat_us),
        "avg_us_per_route": avg_us / routes_per_call if routes_per_call else None,
        "avg_ms_per_48_layers": avg_us * QWEN3_LAYERS / 1000.0,
        "tokens_per_s_one_layer": qlen * args.iters / total_s,
        "tflops": flops_per_call * args.iters / total_s / 1e12,
        "total_time_s": total_s,
        "group_min_len": args.group_min_len,
        "group_max_len": args.group_max_len,
        "m_block": args.m_block,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ktransformers-root", default="/home/linke/ktransformers")
    parser.add_argument("--qlen", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--threadpool-count", type=int, default=1)
    parser.add_argument("--numa-nodes", default="")
    parser.add_argument("--weight-dtype", choices=("bf16", "f16"), default="bf16")
    parser.add_argument("--cpu-route-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--layer-num", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--group-min-len", type=int, default=10)
    parser.add_argument("--group-max-len", type=int, default=1024)
    parser.add_argument("--m-block", type=int, default=4)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.threads < args.threadpool_count:
        raise ValueError("--threads must be >= --threadpool-count")
    if not 0.0 <= args.cpu_route_fraction <= 1.0:
        raise ValueError("--cpu-route-fraction must be in [0, 1]")
    _parse_numa_nodes(args.numa_nodes, args.threadpool_count)
    _add_ktransformers_paths(args.ktransformers_root)
    import cpuinfer_ext

    print(json.dumps({"system_info": _system_info(), "torch": torch.__version__}, ensure_ascii=False))
    rows = []
    for qlen in args.qlen:
        row = bench_one(cpuinfer_ext, args, qlen)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
