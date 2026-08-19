# 回退 verify histogram NumPy collect

## 决定

回退 `2654eb4` 对 production `histogram_kt_hybrid` collect 的 NumPy 稀疏提取，恢复
`cd2cb06` 已使用的逐层 PyTorch `nonzero/index_select/clone` 路径。原因不是正确性失败，
而是两套相同请求的端到端 TPOT 均为负收益。

## 端到端证据

固定 MMLU-Pro validation 第 0 条、seed 20260719、512 输出、temperature 0.6、F16
single-weight 和 2 x 8 NUMA CPUInfer；所有 profiling 关闭：

| preset | 回退目标状态 `cd2cb06` | NumPy 候选 `2654eb4` | 候选变化 |
|:---|---:|---:|---:|
| `k2_dynamic_f16_3080_active14_phase1_recent` | **60.156 ms/token** | 65.567 ms/token | +8.99% |
| `k2_dynamic_f16_3080_active14` | **62.637 ms/token** | 68.960 ms/token | +10.10% |

回退后的运行时代码与已完成一条正式 active14 请求的 `cd2cb06` 完全相同，因此
62.637 ms/token 就是本回退版本的单请求 TPOT 验证。它相对 `28ca880` 的 70.663
ms/token 为正收益，但没有超过同日复跑的全局锚点 `296cf59`（56.846 ms/token）。

## 保留内容

- 保留 `tests/test_prefetch_runtime_meta.py` 的 pooled host-buffer 独立性断言；原 PyTorch
  路径使用 `clone()`，同样满足不被 buffer 复用污染的不变量。
- 不回退 draft `histogram` 的 NumPy collect；本次失败仅覆盖 verify
  `histogram_kt_hybrid`。
- 不改变 route mask、动态 draft length、prefetch/cache 或任何 preset。
- `2654eb4` 的文档和微基准继续保留，明确标记为端到端未通过的负结果。

## 一句话总结

verify NumPy collect 的孤立微基准没有兑现为 TPOT，故恢复已实测更快的 PyTorch hybrid
collect，并保留 route-mask 与现有动态最优配置。
