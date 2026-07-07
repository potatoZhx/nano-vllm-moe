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


def _make_cpuinfer(cpuinfer_ext, threads: int):
    return cpuinfer_ext.CPUInfer(int(threads))


def _random_expert_ids(pool_size: int, qlen: int, topk: int, expert_num: int) -> torch.Tensor:
    return (
        torch.rand(pool_size * qlen, expert_num)
        .argsort(dim=-1)[:, :topk]
        .reshape(pool_size, qlen, topk)
        .contiguous()
    )


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


def _load_moes(cpuinfer_ext, cpuinfer, layer_num: int, group_min_len: int, group_max_len: int):
    moes = []
    weights_keepalive = []
    for layer_idx in range(layer_num):
        gate_w = torch.randn(
            QWEN3_EXPERT_NUM,
            QWEN3_INTERMEDIATE_SIZE,
            QWEN3_HIDDEN_SIZE,
            dtype=torch.bfloat16,
        ).contiguous()
        up_w = torch.randn(
            QWEN3_EXPERT_NUM,
            QWEN3_INTERMEDIATE_SIZE,
            QWEN3_HIDDEN_SIZE,
            dtype=torch.bfloat16,
        ).contiguous()
        down_w = torch.randn(
            QWEN3_EXPERT_NUM,
            QWEN3_HIDDEN_SIZE,
            QWEN3_INTERMEDIATE_SIZE,
            dtype=torch.bfloat16,
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
        config.gate_proj = gate_w.data_ptr()
        config.up_proj = up_w.data_ptr()
        config.down_proj = down_w.data_ptr()
        config.gate_type = GGML_TYPE_BF16
        config.up_type = GGML_TYPE_BF16
        config.down_type = GGML_TYPE_BF16
        config.hidden_type = GGML_TYPE_BF16

        moe = cpuinfer_ext.moe.MOE(config)
        cpuinfer.submit(moe.load_weights())
        cpuinfer.sync()
        moes.append(moe)
        weights_keepalive.append((gate_w, up_w, down_w))
    return moes, weights_keepalive


def bench_one(cpuinfer_ext, args, qlen: int) -> dict[str, object]:
    cpuinfer = _make_cpuinfer(cpuinfer_ext, args.threads)
    moes, weights_keepalive = _load_moes(
        cpuinfer_ext,
        cpuinfer,
        args.layer_num,
        args.group_min_len,
        args.group_max_len,
    )
    _ = weights_keepalive

    pool_size = max(args.warmup, args.iters, 256)
    expert_ids = _random_expert_ids(pool_size, qlen, QWEN3_TOPK, QWEN3_EXPERT_NUM)
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
    routes_per_call = qlen * QWEN3_TOPK
    avg_us = sum(lat_us) / len(lat_us)
    flops_per_call = 3 * 2 * qlen * QWEN3_TOPK * QWEN3_HIDDEN_SIZE * QWEN3_INTERMEDIATE_SIZE
    return {
        "system": "ktransformers_cpuinfer_ext_moe",
        "backend": "llamafile",
        "dtype": "bf16",
        "qlen": qlen,
        "threads": args.threads,
        "layer_num": args.layer_num,
        "warmup_iters": args.warmup,
        "iters": args.iters,
        "expert_num": QWEN3_EXPERT_NUM,
        "topk": QWEN3_TOPK,
        "hidden_size": QWEN3_HIDDEN_SIZE,
        "intermediate_size": QWEN3_INTERMEDIATE_SIZE,
        "routes_per_call": routes_per_call,
        "avg_latency_us": avg_us,
        "p50_latency_us": lat_sorted[len(lat_sorted) // 2],
        "p95_latency_us": lat_sorted[int(len(lat_sorted) * 0.95)],
        "min_latency_us": min(lat_us),
        "avg_us_per_route": avg_us / routes_per_call,
        "avg_ms_per_48_layers": avg_us * QWEN3_LAYERS / 1000.0,
        "tokens_per_s_one_layer": qlen * args.iters / total_s,
        "tflops": flops_per_call * args.iters / total_s / 1e12,
        "total_time_s": total_s,
        "group_min_len": args.group_min_len,
        "group_max_len": args.group_max_len,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ktransformers-root", default="/home/linke/ktransformers")
    parser.add_argument("--qlen", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--layer-num", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--group-min-len", type=int, default=10)
    parser.add_argument("--group-max-len", type=int, default=1024)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
