# 32. t16 动态 draft 门限 0.95 否决

日期：2026-08-22

## 一句话总结

在当前公平 t16 最优配置上，把动态 K1→K2 门限从 0.97 放宽到 0.95 后，单请求
TPOT 从 **52.566035** 回退到 **53.994229 ms/token（+2.72%）**；不新增 preset，
继续永久保留 0.97 `first_increase` 动态配置。

## 为什么测试 0.95

先对当前 t16 lifecycle trace 做只读分析。270 个 speculative rounds 中，245 轮实际
K1、24 轮实际 K2、1 轮因输出预算只剩一个 token 而 K0。当前 0.97 门限选中的 24 个
K2 round 里：

- 第一个 draft 为 24/24 接受；第二个 draft 为 23/24 接受；
- K2 组内累计 71 个输出 token，`step_wall/output=49.358 ms`；
- 现有 K1→K2 判定仍有选择性，不是 fixed K2。

未进入 K2 的 round 中，`predicted alpha` 位于 `[0.95, 0.97)` 的 13 轮首 draft 为
13/13 接受。相比直接放宽到 0.90，0.95 没有纳入 trace 中已出现首 draft 失败的区间，
因而是本轮风险最低、最接近现有算法框架的候选。需要强调：首 draft 已接受并不能提供
同一 round 的 alpha2 反事实，也不能证明额外 qlen3 verify、accept 和 cache 外部性有净收益，
所以仍必须以 production-off 端到端请求门禁。

只读 trace：

```text
results/analysis_t16_b1_ghost8_lutfuse_lifecycle_20260822/
```

## 公平一请求门禁

候选沿用：

```text
k2_dynamic_f16_3080_active14_phase1_recent_b1_ghost8_lutfuse
```

仅通过 `--draft-tpot-td-ms 95` 把 `td/tv` 从 `97/100` 改为 `95/100`。其它设置完全
一致：MMLU-Pro validation sample 0、107-token prompt、seed 20260719、temperature 0.6、
固定输出 512 token、single-weight llamafile F16、profile 全关、CUDA 0、GPU utilization
0.98，以及 CPUInfer 双 NUMA 2 x 8（总计 16 threads）。

| 配置 | TPOT | decode tok/s | decode | steps | mean step | p50/p90/p95/max |
|:---|---:|---:|---:|---:|---:|:---|
| 0.97 当前最佳 | **52.566035 ms** | **19.024** | 26.861 s | 265 | **101.363 ms** | 94.486/134.610/141.977/187.491 ms |
| 0.95 候选 | 53.994229 ms | 18.520 | 27.591 s | 263 | 104.909 ms | 98.919/135.215/143.776/201.591 ms |
| 变化 | **+1.428194 ms / +2.72%** | -2.65% | +0.730 s | -2 | **+3.50%** | p50/p90/p95/max 全部回退 |

候选生成 512/512 token，`output_fixed_length_ok=true` 且 validation error 为空；不同
sampling digest 不用于逐 token 等同性判断。结果：

```text
results/tpot_t16_b1_ghost8_lutfuse_threshold095_20260822/
```

## 决定与后续门禁

- 不新增 0.95 preset，不改运行时代码；现有 0.97 动态配置和全部 fallback 不变；
- 0.98 在历史 t32 请求回退，0.95 又在公平 t16 请求回退，因此不再把静态门限扫描当作
  动态长度优化；
- 下一版动态算法若继续，必须在 K1 时显式估计“额外 K2 draft + qlen3 verify +
  cache/prefetch 外部性”相对 K1 的边际 TPOT，而不能只把 alpha1 当作 alpha2 的代理；
- 下一项按收益预期回到最大热点：t16 CPUInfer exposed sync/native operator 分解与优化。
