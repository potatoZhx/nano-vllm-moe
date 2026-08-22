# 33. t16 CPUInfer native 分项与静态任务划分否决

日期：2026-08-22

## 一句话总结

同源 KT llamafile F16 native profile 证明 gate/up GEMM 占 **65–67%**、down GEMM 占
**约 30%**；静态任务划分虽使孤立 qlen2/3 算子改善 4.17%/0.91%，端到端却从
**52.566035** 回退到 **53.638205 ms/token（+2.04%）**，因此不启用该开关。

## 边界与资源

Nano 当前并不是另写了一套 CPU expert kernel，而是直接使用本地 KTransformers 构建的
`cpuinfer_ext` 与 llamafile MoE。KT 工作树已有用户未提交修改，本轮没有编辑、覆盖或提交
这些文件，也没有替换正式 `.so`。native instrumentation 仅在 `/tmp` 下从当前源码只读
构建；编译和基准的 CPU 并行度/CPUInfer 线程都限制为 16。

分析形状与当前 Nano 路径一致：Qwen3-30B-A3B、F16 expert、BF16 hidden、qlen2/3、
约 50% routes 留在 CPU、`group_min_len=1`、`m_block=32`、双 NUMA 2 x 8。

## Native phase profile

`FORWARD_TIME_PROFILE` 的短微基准按每个 NUMA half 记录阶段；qlen2/3 分别得到 50/49 条
可解析记录（qlen3 有一条并行 stdout 交织，不影响阶段排序）。instrumentation 有打印扰动，
所以只使用阶段占比，不把绝对耗时当正式性能数据。

| qlen | prepare | input copy | gate/up + activation/quantize-down | down GEMM | weighted merge |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.73% | 3.10% | **65.56%** | **29.72%** | 0.70% |
| 3 | 0.56% | 1.63% | **67.06%** | **30.11%** | 0.51% |

因此复用一次小 input memcpy、减少 Python wrapper 或单独优化 merge 都不是高收益路线；
95% 以上时间已经在三次大权重 GEMM 及其相邻 activation/quantize-down。真正有量级潜力的
native 方案必须减少权重带宽/算术，或减少送入 CPU 的 routes。

## Dynamic/static 微基准

KT worker 已提供 `DISABLE_DYNAMIC_SCHEDULING=1`：把每个 GEMM block 的原子
work-stealing 改成按 worker 静态连续分片。使用正式 `.so`、同一 seed、30 warmup + 300
iterations：

| qlen | 调度 | mean | p50 | p95 | mean 变化 |
|---:|:---|---:|---:|---:|---:|
| 2 | dynamic | 1009.443 us | 1017.047 us | 1047.011 us | baseline |
| 2 | static | **967.383 us** | **979.429 us** | **1010.733 us** | **-4.17%** |
| 3 | dynamic | 1450.684 us | 1469.024 us | 1517.139 us | baseline |
| 3 | static | **1437.495 us** | **1459.667 us** | **1509.886 us** | **-0.91%** |

两个 shape 的 mean/p50/p95 均改善，因此进入一条真实请求门禁。

## 公平一请求 TPOT

端到端候选只在进程环境增加 `DISABLE_DYNAMIC_SCHEDULING=1`，继续使用当前推荐
`k2_dynamic_f16_3080_active14_phase1_recent_b1_ghost8_lutfuse`。MMLU-Pro sample 0、
seed 20260719、temperature 0.6、固定输出 512 token、profile 全关、single-weight F16、
GPU 0.98/KV 1536 和 CPUInfer 2 x 8 均不变。

| 配置 | TPOT | decode tok/s | decode | rounds | mean round | p50/p90/p95/max |
|:---|---:|---:|---:|---:|---:|:---|
| dynamic 当前最佳 | **52.566035 ms** | **19.024** | 26.861 s | 265 | **101.363 ms** | 94.486/134.610/141.977/187.491 ms |
| static candidate | 53.638205 ms | 18.643 | 27.409 s | 268 | 102.273 ms | 94.840/133.807/140.304/179.163 ms |
| 变化 | **+1.072170 ms / +2.04%** | -2.00% | +0.548 s | +3 | **+0.90%** | mixed |

候选生成 512/512 token、fixed-length validation 通过。sampling digest 不同且 rounds 多 3，
但 TPOT 与 mean round wall 同时回退，未形成可保留的端到端收益。结果：

```text
results/tpot_t16_b1_ghost8_lutfuse_static_schedule_20260822/
```

## 决定

- 不设置 `DISABLE_DYNAMIC_SCHEDULING`，正式路径继续用 dynamic work-stealing；
- 不改 Nano 或 KT runtime，不新增 preset；
- 不再优先扫描 m_block、group_min 或 worker scheduling 小参数；
- operator 后续仅保留两类高价值路线：带正确性/质量门禁的低带宽 packed kernel，或减少
  CPU routes；鉴于量化改动会改变数值，下一项优先做 source/rank-aware cache admission。
