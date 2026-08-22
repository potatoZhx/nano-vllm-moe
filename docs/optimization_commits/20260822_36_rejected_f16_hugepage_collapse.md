# 36. 精确 F16 CPUInfer 大页折叠否决

日期：2026-08-22

## 一句话总结

保持 F16 权重与 2 x 8 CPUInfer 资源不变时，2 MiB 大页使单层 qlen2 微基准出现约
1.5% 的顺序校正后 p50 信号，但完整请求实际覆盖约 31.78 GiB 后 TPOT 仍从
**52.566035** 回退到 **52.677165 ms/token（+0.21%）**；候选代码、CLI 和临时脚本均撤销。

## 约束与候选

本轮严格遵守不压缩权重的红线：CPUInfer 继续直接读取原始 F16 single-weight buffer，
没有量化、重排、近似或额外权重副本。候选只对每个 NUMA shard 的 2 MiB 对齐内部区间
设置 `MADV_HUGEPAGE`，并尝试 Linux 6.8 的 `MADV_COLLAPSE`。资源继续固定为：

- CPUInfer total threads = 16；
- 双 NUMA `[0:8] [1:8]`；
- canonical dynamic draft、active14、phase1 b1、ghost8 和 fused LUT 全部不变；
- MMLU-Pro validation sample 0、seed 20260719、temperature 0.6、512 固定输出。

## 精确 F16 微基准

单层 Qwen3-30B-A3B 形状、F16 weights/BF16 hidden、约 50% CPU routes、
`group_min_len=1`、`m_block=32`。每个 shape 120 warmup + 1200 iterations；另跑不折叠的
before/after control，扣除第二轮天然变快的顺序效应。

| shape | paired mean | paired p50 | control mean | control p50 | 顺序校正后 p50 |
|:---|---:|---:|---:|---:|---:|
| qlen2 / 8 CPU routes | 1021.095 -> 980.208 us | 1007.758 -> 989.319 us | -1.10% | -0.31% | **约 -1.52%** |
| qlen3 / 12 CPU routes | 1415.129 -> 1407.410 us | 1431.916 -> 1425.754 us | -0.11% | -0.16% | **约 -0.27%** |

`smaps` 验证单层六个 shard 的 1.113 GiB 对齐内部区间全部成为 `AnonHugePages`，因此
微基准比较的确测到了大页，而不是只测 `madvise` 调用。

## 完整模型可用性处理

整段同步折叠在完整模型加载内暴露了微基准没有覆盖的内存压力和 VMA 状态：

1. 临时 F16 pack 尚未释放时，整段调用返回 `ENOMEM`；
2. 改到释放临时 pack 后，整段调用返回 `EAGAIN`；
3. 改为逐 2 MiB 后，已有约 21 GiB 成功，但后续遇到 `EINVAL`；
4. 最终 best-effort 仅跳过内核明确的 `EAGAIN/EINVAL/ENOMEM`，模型可用，并在请求前通过
   `/proc/<pid>/smaps_rollup` 实测 `AnonHugePages=33,320,960 KiB`，约 **31.78 GiB**。

前三次都在初始化阶段失败，没有生成 token，也不计作 TPOT 验证。第四次完成了规定的
一条真实请求。

## 公平一请求 TPOT

| 配置 | TPOT | decode tok/s | decode | rounds | mean round | p50/p90/p95/max |
|:---|---:|---:|---:|---:|---:|:---|
| canonical t16 | **52.566035 ms** | **19.024** | 26.861 s | 265 | **101.363 ms** | 94.486/134.610/141.977/187.491 ms |
| F16 hugepage best-effort | 52.677165 ms | 18.984 | 26.918 s | 263 | 102.350 ms | 95.433/133.918/142.918/210.795 ms |
| 变化 | **+0.111130 ms / +0.21%** | -0.21% | +0.057 s | -2 | **+0.97%** | mixed，max 更差 |

候选完成 512/512 token，fixed-length validation 通过。输出 digest 因 sampling 轨迹不同为
`fcc01f052f5baf5fff92cf3c2ca2463acbf2305eb2b8ad643ac09c0dde40e6a1`；TPOT 和 mean
round wall 同时回退，所以不能把较少的两个 rounds 解释为算子收益。

成功结果：

```text
results/tpot_t16_b1_ghost8_lutfuse_hugepage_best_effort_20260822/
```

三个初始化失败结果也分别保存在：

```text
results/tpot_t16_b1_ghost8_lutfuse_hugecollapse_20260822/
results/tpot_t16_b1_ghost8_lutfuse_hugecollapse_aftertrim_20260822/
results/tpot_t16_b1_ghost8_lutfuse_hugepage_partial_20260822/
```

## 决定

- 不保留 hugepage runtime、CLI、config 或测试，临时微基准脚本已删除；
- 不再尝试对已 first-touch 的 54 GiB native weights 做 post-hoc 同步页折叠；
- canonical 继续是
  `k2_dynamic_f16_3080_active14_phase1_recent_b1_ghost8_lutfuse`，最低点仍为
  **52.566035 ms/token**；
- 如果未来修改同源 C++ allocator，只能低优先级分析“分配时对齐并在 first-touch 前设置
  THP”，且必须先证明不会破坏 NUMA 页归属或增加启动失败，不能复用本轮 post-hoc 方案。
