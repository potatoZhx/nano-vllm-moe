# 37. 准入 shadow 与逐层 CPU route 归因

日期：2026-08-22

## 一句话总结

新增只在 transfer-aware profile 开启时工作的 source/rank/victim shadow；真实 t16 请求显示
平均每次 verify 每层有 **8.895 条 CPU routes、7.410 个实际 CPU experts**，CPU route 高的
主因是候选覆盖不足而非预取太晚或错误换出，因此不启用过拟合的净收益准入门槛。

## 配置与资源

分析与 production-off 门禁均使用当前推荐 preset：

```text
k2_dynamic_f16_3080_active14_phase1_recent_b1_ghost8_lutfuse
```

- MMLU-Pro validation sample 0，seed 20260719，temperature 0.6，固定生成 512 tokens；
- single-weight、精确 F16 权重，没有量化或压缩；
- CPUInfer 严格为 16 total threads：2 个 NUMA subpool × 8 threads；
- profile 只用于归因；TPOT 门禁关闭 collect/engine/transfer profile。

分析结果：

```text
results/analysis_t16_b1_ghost8_lutfuse_admission_shadow_20260822/
results/analysis_t16_b1_ghost8_lutfuse_admission_shadow_20260822/cpu_route_causes.json
```

生产门禁结果：

```text
results/tpot_t16_b1_ghost8_lutfuse_admission_shadow_20260822/
```

## “平均每层 CPU 计算数量”的两个口径

本次 trace 有 262 次 verify、48 个 MoE layers、564 个 verified tokens；每个 token 每层有
8 条 expert routes。执行总量为 216,576 routes，其中 111,869 条进入 CPU。

| 指标 | 数值 | 含义 |
|:---|---:|:---|
| CPU route ratio | **51.653%** | 111,869 / 216,576 |
| 每次 verify、每层 CPU routes | **8.895** | 动态 K 下，一次 verify 每层实际进入 CPU 的 route 数 |
| 每次 verify、每层 CPU experts | **7.410** | 合并同 expert routes 后，每层实际发给 CPUInfer 的不同 expert 数 |
| 每个 verified token、每层 CPU routes | **4.132 / 8** | 排除动态 K 导致的单次 verify token 数变化 |

这里的“CPU experts”比 CPU routes 更接近 CPUInfer 的小批量计算次数；二者不能混用。
CPU 最重的 layer 是 0/1，分别为 13.565/12.485 routes per verify；最低的 layer 18 为
6.011。逐层完整数据在 `cpu_route_causes.json`。

## 为什么 CPU route 占比高

### 1. 首要原因：预测/候选覆盖不足

按每个 verify round 的候选、submit、evict 与真实执行 route 对齐，111,869 条 CPU routes
可分为：

| 归因 | CPU routes | 占 CPU routes | 解释 |
|:---|---:|---:|:---|
| 未进入本轮候选集 | **80,324** | **71.802%** | 当前预测和历史均未把它提升为可预取候选；这是覆盖缺口上限 |
| 在宽候选集但未准入 | 30,342 | 27.123% | 主要是低 rank 候选超过当前有限预算；不能等同为增加 budget 就能获益 |
| 已 submit 仍走 CPU | 912 | **0.815%** | 预取窗口、ready/publication 时机不足的上限 |
| 本轮换出后又使用 | 291 | **0.260%** | 可直接归因于当前轮错误替换/冲突的上限 |

对确实存在 draft 对照的 115,968 条 routes 再直接比较 expert set：draft 只覆盖
50,210 条，recall 为 **43.296%**；这部分的 60,163 条 CPU routes 中有 43,679 条
（**72.601%**）对应 draft 未预测 expert。这个独立统计支持“预测覆盖是主因”，而不是
仅由候选过滤规则造成的假象。

### 2. 缓存容量和 expert 分布给出了较高底噪

当前只有 672 个 active slots，对 48 × 128 个 expert 权重是 10.94% 的驻留比例。保持当前
逐层 slot 数，使用整条 trace 的未来全局频率静态保留每层 top-S expert，CPU route ratio
仍为 **44.974%**。这不是可实现的动态最优下界，但说明大部分 51.653% 来自小容量面对
分散路由分布；当前算法相对该静态未来信息参考只剩约 6.68 percentage points。

### 3. 不是预取窗口不足，也不是准入替换主导

已 submit 仍走 CPU 和本轮换出后马上再用合计只有 **1.075%** 的 CPU routes，因此把重点
放在更早发同一批 candidate，或者只调整 victim 选择，不能解释当前约一半 routes 走 CPU。
扩大 budget 也不是直接答案：历史 vpb4 比 vpb2 慢、vpb1 和 2/2/1 同样回退，额外 H2D、
cache churn 和 overlap 会抵消多加载的命中。

## 净收益准入 shadow 结论

对每次实际替换，在未来 8 steps 比较 incoming 和 victim 的真实 route demand：

| source | 有 victim 的 submit | incoming 更有价值 | victim 更有价值 | 完美未来 guard 最多挽回 routes |
|:---|---:|---:|---:|---:|
| phase1 | 262 | 162 | 45 | 87 |
| verify segment | 1,572 | 1,064 | 269 | 561 |
| draft segment | 728 | 548 | 74 | 139 |
| 合计 | 2,562 | 1,774 | 388 | **787** |

即使使用不可实现的未来真值拒绝每一次负收益替换，上限也只有 CPU routes 的 **0.704%**。
更重要的是，负/正样本的 candidate priority、rank、victim recency 和 lifetime access count
高度重叠；单请求阈值只能筛出很小且明显过拟合的子集。基于这些证据：

- 不新增运行时 admission guard，不改变 canonical preset；
- 保留 shadow 字段和 `scripts/analyze_cpu_route_causes.py`，用于后续 predictor 候选的统一归因；
- 下一步应优先提高 top-rank 候选覆盖和排序精度，而不是再扫 ghost TTL、LRU/LFU 权重或
  全局 prefetch budget。

## 下一候选的离线排序证据

当前 786 个 verify segment 边界共有 1,572 个实际 submit；由于跨 source 的 priority 尺度
不一致，实际被选中的 1,572 个候选全部来自 `verify_history`，`draft_live` 没有进入这条
vpb2 路径。在保持 submit 数、H2D 字节数和 vpb2 完全不变的 shadow 中，只按当前 verify
step 的真实 route demand 比较：

| 固定 vpb2 排序 | 当前 step route demand | 命中候选数 |
|:---|---:|---:|
| 当前 priority（全部 verify history） | 1,462 | 908 |
| 每个同时有两种 source 的边界保留 1 个 draft + 1 个 history | 1,703 | 1,078 |
| draft-live 优先，剩余位置再由 history 填充 | **1,914** | **1,243** |
| 候选池未来真值 top2 oracle | 2,142 | — |

因此下一个更有根据的候选不是加 budget，而是在固定 vpb2 中校准 source：保守版本每个边界
至多保留一个 live-draft 位置，仍保留一个 recent-history 位置。该 shadow 只衡量当前 step
需求，没有重放 cache/victim/overlap，不能代替 TPOT；开始实现前仍需单独获得用户确认。

## TPOT 门禁

profile-off 请求为 **54.367700 ms/token**，参考最低点为 52.566035 ms/token（+3.43%）；
但本次输出有 276 个 decode steps，参考为 265，digest 也不同。平均 step wall 反而从
101.363 降至 100.659 ms（-0.69%）。因此该数值只能证明完整请求可用，不能把随机路径变化
误判为 shadow 的运行时回退或收益；shadow 数据收集在 production-off 时不执行，且没有
启用任何新的 cache/admission 决策。

## 验证

```text
python -m pytest -q \
  tests/test_predictive_prefetch.py \
  tests/test_prefetch_runtime.py \
  tests/test_analyze_cpu_route_causes.py

36 passed
```
