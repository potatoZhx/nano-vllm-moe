# 31. t16 CPU tail 分解与 ghost16 否决点

日期：2026-08-22

## 一句话总结

当前公平 t16 配置的低扰动 profile 显示 CPUInfer exposed sync 是最大热点；但把 ghost
保护从 8/8 扩到 16/16 后，单请求 TPOT 从 52.566 回退到 **55.559 ms/token（+5.69%）**，
因此不新增 preset，继续保留 ghost8。

## 当前 t16 latency 分解

分析配置与当前推荐 preset 一致：

```text
k2_dynamic_f16_3080_active14_phase1_recent_b1_ghost8_lutfuse
```

资源固定为 16 total CPUInfer threads、双 NUMA 2 x 8；MMLU-Pro validation sample 0、
seed 20260719、temperature 0.6、固定生成 512 token。`latency_breakdown_profile` 使用 CUDA
event 并在自然同步/请求结束后汇总，不在每个算子后强制同步。

| 分项 | 请求累计时间 | 占 verify stream | 占 decode |
|:---|---:|---:|---:|
| verify stream | 20.551 s | 100% | 74.4% |
| CPUInfer sync | **12.626 s** | **61.4%** | **45.7%** |
| CPU→GPU output copy | 0.216 s | 1.0% | 0.8% |
| prefetch 暴露 wait（buffer reuse + final drain） | 0.024 s | 0.1% | 0.1% |

profile 请求本身为 54.118 ms/token，只用于热点排序。结论很清楚：copy、LUT 和 prefetch
显式 wait 已不是首要成本；下一阶段应减少 CPU miss routes，或降低同源 CPUInfer 的实际
计算/同步尾部。

结果目录：

```text
results/analysis_t16_b1_ghost8_lutfuse_latency_20260822/
```

## 当前 t16 lifecycle

独立的 transfer-aware profile 记录：

- 2,618 次 publication；phase1/verify/draft 为 269/1620/729；
- 首次消费 2,420 次；三类首次消费率为 87.36%/90.99%/97.53%；
- 1,985 次 eviction 中 1,386 次后来重载；
- eviction→reload 在 8 steps 内 367 次，9–16 steps 又有 206 次；
- ghost8 命中 367 次、避免 88 次 victim，safety valve 为 0；
- 实际执行 CPU routes 为 109,794，说明 CPU tail 仍有足够的算法优化空间。

结果目录：

```text
results/analysis_t16_b1_ghost8_lutfuse_lifecycle_20260822/
```

## ghost16 一请求门禁

候选只用命令行把 `predictive_ghost_window_steps` 和
`predictive_ghost_protect_steps` 从 8 改为 16；所有资源、算法和 profiling-off 口径均与
当前公平最佳相同。

| 配置 | TPOT | decode tok/s | steps | mean step wall | p50/p90/p95/max |
|:---|---:|---:|---:|---:|:---|
| ghost8 + fusion | **52.566035 ms** | **19.024** | 265 | **101.363 ms** | 94.486/134.610/141.977/187.491 ms |
| ghost16 + fusion | 55.559044 ms | 17.999 | 265 | 107.135 ms | 99.681/141.697/150.550/201.860 ms |
| 变化 | **+2.993 ms / +5.69%** | -5.39% | 0 | **+5.69%** | 全部分位数回退 |

两条请求 output digest 不同，但 decode steps 恰好相同，且 TPOT、平均 step wall 和所有
分位数一致回退。更长窗口能识别更多重载，不代表保留这些 expert 的机会成本为正；16-step
保护会挤出更近期/更高价值的 resident。候选结果位于：

```text
results/tpot_t16_b1_ghost16_lutfuse_20260822/
```

## 决定

- 不新增 ghost16 preset，不改运行时代码；
- 当前推荐和所有 fallback 保持不变，继续使用 ghost8；
- 不再盲扫 ghost TTL；下一候选必须直接面向 CPU miss routes/CPUInfer exposed tail；
- 后续所有 profile 与 TPOT 继续锁定 16 total threads。
