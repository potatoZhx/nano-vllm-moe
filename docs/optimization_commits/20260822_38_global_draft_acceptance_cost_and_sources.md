# 38. 动态 draft 接受率、边际成本与预取来源全局分析

日期：2026-08-22

## 一句话总结

当前动态 K1/K2 的 draft 总接受率为 **82.70%**，保守门限选出的 K2 第二 token
条件接受率为 **89.47%**，且低扰动 trace 中新增一次 draft 的边际调用约
**19.46 ms**；K2 已经有效摊薄 verify，但现有数据还不足以证明 K3 有净收益，下一项仍应
先在固定 vpb2 内给高精度的 draft-live 互补候选保留一个位置，再单独评估动态 K3。

## 范围与口径

本轮只读取已有的 16-thread、single-weight llamafile F16、active14、动态 K1/K2 分析
profile，不修改运行时算法，也没有运行新的端到端优化验证。主要 trace 为：

```text
results/analysis_t16_b1_ghost8_lutfuse_latency_20260822/
results/analysis_t16_b1_ghost8_lutfuse_lifecycle_20260822/
results/analysis_t16_b1_ghost8_lutfuse_admission_shadow_20260822/
```

三个 profile 开启的 telemetry 不同，输出轨迹也不同，因此合并值用于估计范围和选择性，
不能替代 profile-off TPOT 门禁。当前 production 最优仍是
`k2_dynamic_f16_3080_active14_phase1_recent_b1_ghost8_lutfuse` 的
**52.566035 ms/token**；权重保持原始 F16/BF16 表示，CPUInfer 始终为 16 threads。

## 1. draft 接受率

| trace | draft tokens | accepted | 总接受率 | 实际 K2 | K2 第二 token 接受 |
|:---|---:|---:|---:|---:|---:|
| latency | 295 | 246 | 83.39% | 31 | 27/31 = 87.10% |
| lifecycle | 293 | 241 | 82.25% | 24 | 23/24 = 95.83% |
| admission shadow | 302 | 249 | 82.45% | 40 | 35/40 = 87.50% |
| **合并** | **890** | **736** | **82.70%** | **95** | **85/95 = 89.47%** |

合并后的 795 个非 terminal speculative rounds 中：

- K1 组为 700 轮，第一 token 接受 `556/700 = 79.43%`；
- K2 组为 95 轮，第一 token `95/95` 全接受，第二 token 条件接受
  `85/95 = 89.47%`，所以 K2 token 级接受率为 `180/190 = 94.74%`；
- 全部 round 的第一 token 接受率为 `651/795 = 81.89%`。

这说明 0.97 控制器确实筛中了高价值 K2 状态，不能把总体 82.70% 直接当作继续 K3 的
条件概率；K3 的边际收益取决于在“前两个 draft 都值得保留”条件下尚未观测的 alpha3。

## 2. 一次 draft 的可见成本

使用 instrumentation 最轻的 latency trace：

| 指标 | 数值 |
|:---|---:|
| 295 次 `run_draft` 模型调用均值 | **20.869 ms/call** |
| 其中 model core 均值 | **17.909 ms/call** |
| 第一个 draft 调用 p50 / p90 | 20.853 / 21.054 ms |
| 第二个 draft 的边际调用均值 / p50 / p90 | **19.456 / 19.526 / 19.813 ms** |
| 完整 K1 draft phase 均值 | **22.068 ms/round** |
| 完整 K2 draft phase 均值 | **42.001 ms/round** |

因此“draft 一步开销”按模型可见 forward 应报告约 **20.87 ms**，而评估 K2/K3 的新增
一步时应使用已经 warm 的边际值约 **19.46 ms**。lifecycle/admission profile 因额外
metadata 和 shadow 记录分别提高到约 26.17/24.40 ms/call，不应当作 production 成本。

## 3. verify history 与 draft predict 的准确度

“准确度”必须拆成原始 expert-set route recall 和经过当前排序后的候选 precision。

### 3.1 同样每层 8 个 expert 的原始 route recall

| 预测源 | 被预测的 verify routes | route recall | 备注 |
|:---|---:|---:|:---|
| 上一 verify 最后 token → 当前 verify 第一 token | 100,224 | **46.295%** | 每层固定 8 个 recent experts |
| 上一 verify 全 token 并集 → 当前第一 token | 100,224 | **56.844%** | 每层平均 12.834 个 experts，不再同预算 |
| draft token → verify 对应 token，全部位置 | 115,968 | **43.296%** | 每层固定 8 个 draft experts |
| draft position 1 → verify position 1 | 100,608 | **43.807%** | K1/K2 合并 |
| draft position 2 → verify position 2 | 15,360 | **39.954%** | 仅 40 个 K2 shadow rounds |

在同样 8 experts/layer 的公平口径上，recent verify history 的 `46.30%` 略高于 draft 的
`43.30%`。第二个 draft 的 route recall 没有因长度增加而提高，反而从 position 1 的
`43.81%` 降至 `39.95%`；因此“更长 draft 自动带来更准确路由预测”没有现有证据。

### 3.2 当前 verify 边界候选的 precision

只看 `source=verify_segment` 的候选，并与同一 verify step 的真实 routes 对齐：

| 合并后的 candidate source | 候选数 | 至少命中一次 | candidate precision | route demand | CPU route demand |
|:---|---:|---:|---:|---:|---:|
| `verify_history` | 1,572 | 908 | **57.761%** | 1,462 | 503 |
| `draft_live` | 953 | 928 | **97.377%** | 1,412 | 1,412 |

这里的 `draft_live` 是跨 source 去重后仍未被 history 覆盖的“边际 draft 独有候选”，不是
全部 draft experts，所以它的 97.38% 与上一节的 43.30% recall 不矛盾。当前 vpb2 的
1,572 个实际 submit 全被 priority 更高的 history 占据；953 个 draft-live 候选虽然排在
后面，但其中 928 个会在当前 step 使用，且 1,412 条对应 routes 最终全部走 CPU。这正是
“保留 history 覆盖，同时给 draft 一个互补位置”的证据。

candidate precision 也不能当作全局 recall：两种 source 的 route demand 分别只占全部
216,576 routes 的 0.675% 和 0.652%。它们衡量的是固定极小 admission budget 中每个位置
是否值得，而不是完整路由可预测程度。

## 4. 与 CPU 路由全局信息的关系

同一 admission shadow 有 262 次 verify、564 个 verified tokens、48 个 MoE layers：

- CPU routes 为 `111,869 / 216,576 = 51.653%`；
- 平均每次 verify、每层 **8.895 条 CPU routes**，平均每个 verified token、每层
  **4.132/8 条 CPU routes**；
- 平均每次 verify、每层有 **7.410 个 unique CPU experts**。

CPU route 归因中，71.802% 没进入候选集，27.123% 进入宽候选但没有准入；已提交仍走 CPU
和使用前换出合计只有 1.075%。因此 CPU 比例高的首因是覆盖/排序和仅 10.94% 的 active
slot 容量，不是预取窗口来不及。增加 K 可能同时增加新位置的预测信息和已有候选的领先
时间，但在当前 draft-live 甚至拿不到 vpb2 位置时，先增加计算来制造更多预测的兑现率较低。

## 5. 增加 draft 长度能否摊薄 verify

### 5.1 K2 已经证明“有机会”

低扰动 latency trace 的观察值为：

| 实际 K | rounds | output/round | round wall | wall/output | verify-ready/output |
|---:|---:|---:|---:|---:|---:|
| K1 | 233 | 1.807 | 100.261 ms | 55.489 ms | 43.273 ms |
| K2 | 31 | 2.871 | 134.919 ms | **46.994 ms** | **32.363 ms** |

被 0.97 门限选中的 K2 组，wall/output 比 K1 观察组低 15.31%，verify-ready/output 低
25.21%。这确实展示了“一次 verify 产出更多 token”的摊薄效果；但两组由控制器选择、
上下文不同，不能把差值当作 K2 相对同状态 K1 的严格因果收益。

### 5.2 K3 有小而真实的机会，但缺少决定性变量

把现有 95 个 K2 trace 离线代入同一个 `td/tv=97/100 first_increase` 公式，如果仅把
`Kmax` 放宽到 3，只有 **12/95 = 12.63%** 的 K2 rounds 会继续 K3，占全部 795 rounds 的
**1.51%**；这 12 轮的前两个 draft 恰好全部真实接受。因此该路线不是 fixed K3，而会只在
极高置信状态尝试，方向上合理。

以 latency K2 的 `134.919 ms/round` 和 `2.871 output/round` 为参考：

- 只计算新增 draft `19.456 ms`，K3 新增一个输出的概率至少要约 **41.4%** 才持平；
- 若保守地再计入当前 qlen2→qlen3 的 verify-ready 增量 `14.724 ms`，并把它作为
  qlen3→qlen4 的代理，临界概率升至约 **72.7%**。

真实 K3 还缺少 alpha3、qlen4 verify/accept 成本和第三位置 route recall，所以 41.4%--72.7%
只是边际 break-even 范围，不是新阈值。当前 active14 preset 只 capture qlen2/3 graph，
增加 K3 还需要 qlen4 graph；active14 只有 6 个 KV blocks/1536-token 容量，新增 graph 的
显存与 warmup 风险也明显高于单纯改变控制器参数。

早期 fixed-K 单请求 screen 中 K2/K3 为 85.613/83.984 ms/token，说明 K3 曾比 K2 略能
摊薄 verify；但两者都慢于 K1 的 75.959，而且当时是 segment12/active12/vpb4 的旧路径，
不能作为当前动态 K3 的验收结果。

## 6. 下一步优先级

本轮全局信息没有推翻上一轮选择，反而把理由收紧为：

1. **先做固定 vpb2 的 source-aware 1+1**：每个同时存在两种 source 的 verify boundary
   至多给一个 `draft_live` 位置，另一个保留 `verify_history`；不增加 draft forward、H2D
   数量或 graph 显存，直接利用 97.38% precision 的边际 draft 候选。
2. **之后再单独做动态 K3**：永久保留当前 K1/K2 最优 preset，新建独立 Kmax3 preset；
   先确认 qlen4 graph 容量，再只允许现有 first-increase 选中的极高置信 rounds 进入 K3，
   用一条同口径 16-thread 请求验收 TPOT，无正收益则只保留本文分析。

不能先把 history 全换成 draft：公平 raw recall 中 history 仍略强，而且 draft-live 的高
precision 是“与 history 去重后的互补价值”。也不能直接扩大 vpb2：历史 budget sweep 已
证明额外 transfer/cache churn 会回退。

本提交只记录分析结论，不实施上述任一优化；开始新的运行时方向前仍需用户确认。
