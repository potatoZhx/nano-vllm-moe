#!/usr/bin/env python3
"""
KTransformers CPU BF16 Expert 推理速度基准测试
目标模型: Qwen3-30B-A3B

Qwen3-30B-A3B MoE 架构参数:
  - expert_num       : 128  (每层共 128 个专家)
  - hidden_size      : 2048
  - intermediate_size: 768  (每个专家的 FFN 中间维度)
  - num_experts_per_tok: 8  (每个 token 激活 8 个专家)
  - num_layers       : 48   (本脚本抽样测量若干层)

支持的 backend:
  - AMXBF16  : Intel AMX BF16 (需要 Sapphire Rapids 及以上 CPU)
  - AMXInt8  : Intel AMX Int8 (在线量化自 BF16 权重)
  - llamafile: 原始 llamafile 后端 (对比基准)

依赖:
  pip install kt-kernel  # 或从 kt-kernel/ 目录执行 pip install .

最简运行（默认 AVX2BF16，decode+prefill 两档）
python bench_qwen3_30b_a3b_cpu_experts.py --threads <你的物理核心数>

同时对比 decode (qlen=1) 和 prefill (qlen=128)
python bench_qwen3_30b_a3b_cpu_experts.py \
    --backend AVX2BF16 \
    --threads 16 \
    --qlen 1 128 \
    --output result.jsonl
"""

import argparse
import os
import sys
import time
import json
import platform
import subprocess
from typing import Optional

import torch
from tqdm import tqdm

# ─── 尝试导入 kt_kernel ────────────────────────────────────────────────────────
try:
    from kt_kernel import kt_kernel_ext
except ImportError:
    # 如果从源码目录运行，先尝试 build/ 路径
    _build_dir = os.path.join(os.path.dirname(__file__), "build")
    sys.path.insert(0, _build_dir)
    try:
        from kt_kernel import kt_kernel_ext
    except ImportError:
        print("[ERROR] 无法导入 kt_kernel_ext。")
        print("  请先进入 kt-kernel/ 目录执行: pip install .")
        sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# Qwen3-30B-A3B 模型参数 (与 Hugging Face config.json 对应)
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_EXPERT_NUM        = 128   # num_experts
MODEL_HIDDEN_SIZE       = 2048  # hidden_size
MODEL_INTERMEDIATE_SIZE = 768   # moe_intermediate_size
MODEL_NUM_EXPERTS_PER_TOK = 8   # num_experts_per_tok
MODEL_NUM_LAYERS        = 48    # num_hidden_layers

# ═══════════════════════════════════════════════════════════════════════════════
# 默认测试超参数
# ═══════════════════════════════════════════════════════════════════════════════
DEFAULT_THREADS      = 32   # CPU 线程数（建议 = 物理核心数）
DEFAULT_SUBPOOL      = 1    # NUMA 子池数量
DEFAULT_QLEN_DECODE  = 1    # Decode 阶段：1 token/step
DEFAULT_QLEN_PREFILL = 128  # Prefill 阶段：128 token/step
DEFAULT_LAYER_NUM    = 4    # 参与测试的层数（随机权重，用于测速）
DEFAULT_WARMUP       = 200
DEFAULT_ITERS        = 2000
DEFAULT_MAX_LEN      = MODEL_NUM_EXPERTS_PER_TOK * 512  # 内部缓冲区上限


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════════

def build_cpuinfer(total_threads: int, num_subpools: int) -> kt_kernel_ext.CPUInfer:
    """构建 CPUInfer 实例（含 NUMA 子池配置）。"""
    base   = total_threads // num_subpools
    remain = total_threads %  num_subpools
    counts = [base + (1 if i < remain else 0) for i in range(num_subpools)]

    cfg = kt_kernel_ext.WorkerPoolConfig()
    cfg.subpool_count        = num_subpools
    cfg.subpool_numa_map     = list(range(num_subpools))
    cfg.subpool_thread_count = counts
    return kt_kernel_ext.CPUInfer(cfg)


def get_system_info() -> dict:
    info = {"platform": platform.platform(), "cpu_count": os.cpu_count()}
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    if os.path.exists("/proc/meminfo"):
        with open("/proc/meminfo") as f:
            for line in f:
                if "MemTotal" in line:
                    kb = float(line.split(":", 1)[1].split()[0])
                    info["mem_gb"] = round(kb / 1024 / 1024, 1)
                    break
    # 检测 AMX 支持
    try:
        flags = subprocess.check_output(["grep", "-m1", "flags", "/proc/cpuinfo"],
                                        stderr=subprocess.DEVNULL).decode()
        info["amx_bf16"] = "amx-bf16" in flags
        info["amx_int8"] = "amx-int8" in flags
        info["avx512"]   = "avx512f" in flags
    except Exception:
        info["amx_bf16"] = info["amx_int8"] = info["avx512"] = "unknown"
    return info


def make_moe(cpuinfer, config: kt_kernel_ext.moe.MOEConfig,
             backend: str) -> object:
    """
    根据 backend 字符串创建并加载 MoE 对象。

    AVX2BF16 → kt_kernel_ext.moe.AVX2BF16_MOE  ← 本机 AVX2+BF16 路径
    AMXBF16  → kt_kernel_ext.moe.AMXBF16_MOE
    AMXInt8  → kt_kernel_ext.moe.AMXInt8_MOE
    llamafile→ kt_kernel_ext.moe.MOE
    """
    if backend == "AVX2BF16":
        moe = kt_kernel_ext.moe.AVX2BF16_MOE(config)
    elif backend == "AMXBF16":
        moe = kt_kernel_ext.moe.AMXBF16_MOE(config)
    elif backend == "AMXInt8":
        moe = kt_kernel_ext.moe.AMXInt8_MOE(config)
    elif backend == "llamafile":
        moe = kt_kernel_ext.moe.MOE(config)
    else:
        raise ValueError(f"未知 backend: {backend}，可选: AVX2BF16 / AMXBF16 / AMXInt8 / llamafile")

    cpuinfer.submit(moe.load_weights_task())
    cpuinfer.sync()
    return moe


def random_expert_ids(batch_size: int, num_experts_per_tok: int,
                      expert_num: int) -> torch.Tensor:
    """生成随机 topK 专家 ID（模拟 router 输出）。"""
    return (
        torch.rand(batch_size, expert_num)
        .argsort(dim=-1)[:, :num_experts_per_tok]
        .contiguous()
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 核心基准函数
# ═══════════════════════════════════════════════════════════════════════════════

def bench(
    backend: str,
    qlen: int,
    threads: int,
    subpool: int,
    layer_num: int,
    warmup: int,
    iters: int,
    max_len: int,
    show_progress: bool = True,
) -> dict:
    """
    对 CPU BF16 专家执行基准测试。

    返回包含吞吐量、带宽、FLOPS 的字典。
    """

    cpuinfer = build_cpuinfer(threads, subpool)

    # ── 初始化权重张量（随机，模拟真实模型权重形状）──────────────────────────
    # Qwen3-30B-A3B 的每个专家权重形状:
    #   gate_proj : [intermediate_size, hidden_size] = [768, 2048]
    #   up_proj   : [intermediate_size, hidden_size] = [768, 2048]
    #   down_proj : [hidden_size, intermediate_size] = [2048, 768]
    E  = MODEL_EXPERT_NUM
    H  = MODEL_HIDDEN_SIZE
    I  = MODEL_INTERMEDIATE_SIZE
    K  = MODEL_NUM_EXPERTS_PER_TOK

    physical_to_logical = torch.arange(E, dtype=torch.int64).contiguous()

    moes      = []
    gate_ws, up_ws, down_ws = [], [], []

    print(f"\n[{backend}] 正在初始化 {layer_num} 层随机权重 ...")
    for layer_idx in range(layer_num):
        gate_w = torch.randn(E, I, H, dtype=torch.float32).contiguous()
        up_w   = torch.randn(E, I, H, dtype=torch.float32).contiguous()
        down_w = torch.randn(E, H, I, dtype=torch.float32).contiguous()

        config = kt_kernel_ext.moe.MOEConfig(E, K, H, I, 0)
        config.max_len                = max_len
        config.gate_proj              = gate_w.data_ptr()
        config.up_proj                = up_w.data_ptr()
        config.down_proj              = down_w.data_ptr()
        config.pool                   = cpuinfer.backend_
        config.physical_to_logical_map = physical_to_logical.data_ptr()

        moe = make_moe(cpuinfer, config, backend)
        moes.append(moe)
        gate_ws.append(gate_w)
        up_ws.append(up_w)
        down_ws.append(down_w)

    # ── 准备输入张量 ──────────────────────────────────────────────────────────
    # 预生成多组 expert_ids / weights 以避免测试循环中重复分配
    pool_size  = max(warmup, iters, 1000)
    expert_ids = random_expert_ids(pool_size, K, E)          # [pool, K]
    weights    = torch.rand(pool_size, 1, K, dtype=torch.float32).contiguous()

    input_tensor  = torch.randn(layer_num, qlen, H, dtype=torch.bfloat16).contiguous()
    output_tensor = torch.empty(layer_num, qlen, H, dtype=torch.bfloat16).contiguous()
    bsz_tensor    = torch.tensor([qlen], dtype=torch.int32)

    # ── Warm-up ───────────────────────────────────────────────────────────────
    print(f"[{backend}] Warm-up ({warmup} 次迭代) ...")
    for i in tqdm(range(warmup), desc="Warm-up", disable=not show_progress):
        idx = i % layer_num
        pool_i = i % pool_size
        cpuinfer.submit(
            moes[idx].forward_task(
                bsz_tensor.data_ptr(),
                K,
                expert_ids[pool_i].data_ptr(),
                weights[pool_i].data_ptr(),
                input_tensor[idx].data_ptr(),
                output_tensor[idx].data_ptr(),
                False,
            )
        )
        cpuinfer.sync()

    # ── 正式测量 ──────────────────────────────────────────────────────────────
    print(f"[{backend}] 测试 ({iters} 次迭代) ...")
    latencies_us = []
    t_total_start = time.perf_counter()

    for i in tqdm(range(iters), desc="Bench", disable=not show_progress):
        idx    = i % layer_num
        pool_i = i % pool_size
        t0 = time.perf_counter()
        cpuinfer.submit(
            moes[idx].forward_task(
                bsz_tensor.data_ptr(),
                K,
                expert_ids[pool_i].data_ptr(),
                weights[pool_i].data_ptr(),
                input_tensor[idx].data_ptr(),
                output_tensor[idx].data_ptr(),
                False,
            )
        )
        cpuinfer.sync()
        t1 = time.perf_counter()
        latencies_us.append((t1 - t0) * 1e6)

    t_total = time.perf_counter() - t_total_start

    # ── 计算性能指标 ──────────────────────────────────────────────────────────
    # 每次 forward 的计算量：3 个矩阵乘 × topK 专家
    #   gate: qlen × H × I × 2 FLOP
    #   up  : qlen × H × I × 2 FLOP
    #   down: qlen × I × H × 2 FLOP
    total_flops_per_iter = 3 * 2 * qlen * K * H * I  # FLOPs

    # BF16 每元素 2 字节；读一次权重，激活 topK 专家
    bytes_per_elem = {"AVX2BF16": 2.0, "AMXBF16": 2.0, "AMXInt8": 1.0, "llamafile": 2.0}.get(backend, 2.0)
    weight_bytes   = K * (H * I + H * I + I * H) * bytes_per_elem * qlen

    avg_us   = sum(latencies_us) / len(latencies_us)
    p50_us   = sorted(latencies_us)[len(latencies_us) // 2]
    p95_us   = sorted(latencies_us)[int(len(latencies_us) * 0.95)]
    min_us   = min(latencies_us)
    bandwidth_gbs = weight_bytes * iters / t_total / 1e9
    tflops        = total_flops_per_iter * iters / t_total / 1e12

    result = {
        "backend"       : backend,
        "qlen"          : qlen,
        "threads"       : threads,
        "subpool"       : subpool,
        "expert_num"    : E,
        "hidden_size"   : H,
        "intermediate_size": I,
        "num_experts_per_tok": K,
        "layer_num"     : layer_num,
        "warmup_iters"  : warmup,
        "test_iters"    : iters,
        "total_time_s"  : round(t_total, 4),
        "avg_latency_us": round(avg_us, 2),
        "p50_latency_us": round(p50_us, 2),
        "p95_latency_us": round(p95_us, 2),
        "min_latency_us": round(min_us, 2),
        "bandwidth_GBs" : round(bandwidth_gbs, 3),
        "TFLOPS"        : round(tflops, 4),
        "tokens_per_s"  : round(qlen * iters / t_total, 2),
    }
    return result


def print_result(r: dict):
    phase = "Prefill" if r["qlen"] > 1 else "Decode"
    print(f"\n{'='*60}")
    print(f"  [{r['backend']}]  {phase} (qlen={r['qlen']})")
    print(f"{'='*60}")
    print(f"  专家数 / topK        : {r['expert_num']} / {r['num_experts_per_tok']}")
    print(f"  hidden_size          : {r['hidden_size']}")
    print(f"  intermediate_size    : {r['intermediate_size']}")
    print(f"  线程 / NUMA子池      : {r['threads']} / {r['subpool']}")
    print(f"  平均延迟             : {r['avg_latency_us']:.1f} µs")
    print(f"  P50 / P95 延迟       : {r['p50_latency_us']:.1f} / {r['p95_latency_us']:.1f} µs")
    print(f"  最低延迟             : {r['min_latency_us']:.1f} µs")
    print(f"  内存带宽             : {r['bandwidth_GBs']:.2f} GB/s")
    print(f"  算力利用             : {r['TFLOPS']:.4f} TFLOPS")
    print(f"  生成吞吐             : {r['tokens_per_s']:.1f} tokens/s")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="KTransformers CPU BF16 Expert 基准 — Qwen3-30B-A3B"
    )
    p.add_argument("--backend",   nargs="+", default=["AVX2BF16"],
                   help="测试的 backend，可多选: AVX2BF16 AMXBF16 AMXInt8 llamafile")
    p.add_argument("--threads",   type=int, default=DEFAULT_THREADS,
                   help="CPU 总线程数（建议设为物理核心数）")
    p.add_argument("--subpool",   type=int, default=DEFAULT_SUBPOOL,
                   help="NUMA 子池数量（双路服务器通常设 2）")
    p.add_argument("--qlen",      type=int, nargs="+",
                   default=[DEFAULT_QLEN_DECODE, DEFAULT_QLEN_PREFILL],
                   help="测试的序列长度列表，1=decode，>1=prefill")
    p.add_argument("--layer-num", type=int, default=DEFAULT_LAYER_NUM,
                   help="并发测试的层数（随机权重）")
    p.add_argument("--warmup",    type=int, default=DEFAULT_WARMUP)
    p.add_argument("--iters",     type=int, default=DEFAULT_ITERS)
    p.add_argument("--max-len",   type=int, default=DEFAULT_MAX_LEN)
    p.add_argument("--no-progress", action="store_true",
                   help="关闭 tqdm 进度条")
    p.add_argument("--output",    type=str, default=None,
                   help="将结果追加写入此 JSONL 文件")
    return p.parse_args()


def main():
    args = parse_args()

    print("\n" + "="*60)
    print("  KTransformers CPU Expert 推理基准")
    print("  模型: Qwen3-30B-A3B")
    print("="*60)

    sysinfo = get_system_info()
    print(f"\n系统信息:")
    for k, v in sysinfo.items():
        print(f"  {k:20s}: {v}")

    if not sysinfo.get("amx_bf16"):
        print("\n[WARNING] 当前 CPU 不支持 amx-bf16 指令集。")
        print("          AMXBF16/AMXInt8 backend 可能回退到 AVX-512 或报错。")

    all_results = []
    for backend in args.backend:
        for ql in args.qlen:
            try:
                r = bench(
                    backend      = backend,
                    qlen         = ql,
                    threads      = args.threads,
                    subpool      = args.subpool,
                    layer_num    = args.layer_num,
                    warmup       = args.warmup,
                    iters        = args.iters,
                    max_len      = args.max_len,
                    show_progress= not args.no_progress,
                )
                r.update({"system": sysinfo,
                           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})
                all_results.append(r)
                print_result(r)
            except Exception as e:
                print(f"\n[ERROR] backend={backend}, qlen={ql}: {e}")
                import traceback; traceback.print_exc()

    if args.output:
        with open(args.output, "a") as f:
            for r in all_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n结果已追加写入: {args.output}")

    # ── 汇总对比表 ────────────────────────────────────────────────────────────
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"  汇总对比")
        print(f"{'='*70}")
        print(f"  {'Backend':<12} {'qlen':>6} {'Avg(µs)':>10} {'BW(GB/s)':>10} {'TFLOPS':>8} {'tok/s':>8}")
        print(f"  {'-'*62}")
        for r in all_results:
            print(f"  {r['backend']:<12} {r['qlen']:>6} "
                  f"{r['avg_latency_us']:>10.1f} "
                  f"{r['bandwidth_GBs']:>10.2f} "
                  f"{r['TFLOPS']:>8.4f} "
                  f"{r['tokens_per_s']:>8.1f}")


if __name__ == "__main__":
    main()
