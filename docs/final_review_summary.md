# 最终审核报告

**Date:** 2026-05-29  
**Project:** nano-vllm-moe — MoE Speculative Decoding with Expert Offloading  
**Reviewer:** Claude (Cowork Session)

---

## 一、交付物清单

| 文件 | 类型 | 状态 | 大小 |
|------|------|------|------|
| `docs/system_design_report.md` | 系统设计报告 | ✅ 完整 | 31K |
| `docs/post_experiment_optimization.md` | 实验后优化设计 | ✅ 完整 | 30K |
| `pre_exps/cache_dual_objective_eval.py` | E1 实验脚本 | ✅ 完整 | 29K |
| `pre_exps/top12_protection_eval.py` | E2 实验脚本 | ✅ 完整 | 37K |
| `pre_exps/dynamic_k_analysis.py` | E3 实验脚本 | ✅ 完整 | 37K |
| `pre_exps/prefetch_coverage_analysis.py` | E4 实验脚本 | ✅ 完整 | 30K |
| `pre_exps/alpha_prediction_eval.py` | E5 实验脚本 | ✅ 完整 | 38K |
| `pre_exps/routing_entropy_analysis.py` | Routing 分析脚本 | ✅ 完整 | 28K |
| `docs/reroute_full_validation_20260528.md` | 完整验证报告 | ✅ 完整 | 23K |
| `atc/` | ATC 论文草稿 | ⚠️ 含 placeholder | — |

---

## 二、核心实验结论一致性验证

### 2.1 Rerouting 算法验证（reroute_full_validation）

实际测量值（Standard Sampling，自然语言任务最相关）：

| Policy | r=0.25 l=128 | r=0.25 l=512 | Draft 开销 |
|--------|-------------|-------------|-----------|
| `round_robin` (baseline) | 0.7248 | 0.3039 | 19.7 ms |
| `drop_miss` (SkipAll) | 0.1476 | 0.1498 | 18.9 ms |
| **`entropy_cache_bias`** | **0.9655** | **0.9278** | 19.4 ms |
| `bounded_cache_bias` | 0.9412 | 0.8558 | 20.2 ms |
| `similarity_replace` | 0.8271 | 0.7702 | 18.8 ms |

**结论：`entropy_cache_bias`（= Alg2_v2）在低缓存比（25%）下表现最优，acceptance rate 比 baseline 高 +0.24～+0.62，draft 开销可忽略（+0.25 ms）。**

### 2.2 Pre-Experiments 结论（E1–E5）

| 实验 | 核心数据 | 设计决策 | 一致性 |
|------|---------|---------|--------|
| E1: 双目标缓存 | HitScore-SubstScore ρ=0.78；LFU hit rate @r=0.25: 75.14% | 放弃 EvictCost-aware 策略，使用纯 LFU | ✅ |
| E2: Top-1/2 保护 | reroute 保护 +0.2pp @r=0.25；cache pin 无效 | 实现 reroute 保护；LFU-RankGuard 取代 pin | ✅ |
| E3: Dynamic K | 简化模型低估 T_stall 8×；K*=12 不可信 | 用 E4 覆盖率信号替代；用 profiling-based controller | ✅ |
| E4: 预取覆盖 | r=0.25 时 K*=1（覆盖率过低）；m=1→2 sweet spot | K*=1 at r=0.25；profiling-based Dynamic K | ✅ |
| E5: α 预测 | RMSE=0.03～0.28；MLP 无显著优势 | 初期使用解析模型 α̂=α₀·exp(-λE(k)) | ✅ |

### 2.3 ATC 论文与实验数据一致性

论文 `evaluation.tex` 的 ablation table 已使用实际数据：
- Full sysname seq_α = **0.731**（来自 E3/E5 的 Alg2_v2 @r=0.25）
- −Entropy cache bias (SkipAll) seq_α = **0.636**（来自 E3/E5 的 SkipAll @r=0.25）

其余性能数字（tok/s、speedup倍数）标记为 `\placeholder{}`，需要真实系统 benchmark 后填入。

---

## 三、文档间一致性检查

### 3.1 设计演进主线

```
system_design_report.md          →  post_experiment_optimization.md
(理论设计，含 5 个验证实验设计)         (基于 E1-E5 实验结果的修正版)

主要修正：
├── 放弃 EvictCost-aware 双目标 (E1)
├── 将 cache pin 改为 LFU-RankGuard (E2)
├── 将静态 K 改为 profiling-based Dynamic K (E3+E4矛盾)
├── 确认解析 α 模型足够 (E5)
└── 新增算法耦合分析 (§6)
```

### 3.2 已发现的小问题

| 问题 | 严重性 | 说明 |
|------|--------|------|
| ATC paper 写 RTX 4090，但实验在 A100-SXM4-80GB 上运行 | ⚠️ 需修正 | `implementation.tex` L37-38 需更新硬件描述 |
| E4 的 prefetch rate 上限 m=4 可能偏保守 | ℹ️ 已记录 | `post_experiment_optimization.md` §2修正2 已说明 |
| reroute validation 使用合成 prompt（filler text），接受率虚高 | ℹ️ 已记录 | 报告 §6.4-6.5 已明确说明，相对排名仍有效 |
| E3 简化 T_stall 模型与 E4 差 8× | ℹ️ 已记录 | `post_experiment_optimization.md` §4.1 已分析 |

---

## 四、核心设计决策总结

本项目为 Qwen3-30B-A3B（N=128, k=8, 48 MoE layers）在单 GPU + Host DRAM 场景下设计并验证了以下优化方案：

### 推荐配置（基于实验验证）

```yaml
# Rerouting
rerouting_algorithm: "entropy_cache_bias"   # +24~62% acceptance vs round_robin @r=0.25
rerouting_gamma0: 4.0
rerouting_miss_low: 0.25
rerouting_miss_high: 0.50
rerouting_top12_protect: true               # +0.2pp, 近零开销

# Cache
cache_strategy: "lfu_rankguard"             # LFU + top-1/2 驱逐保护
rank_guard_threshold: 0.15                  # 保护约 42/128 高频专家

# Dynamic K
dynamic_k_enabled: true
dynamic_k_max: 8
dynamic_k_level1_crit_threshold: 0.3
dynamic_k_level1_miss_stop: 0.6
dynamic_k_level1_miss_safe: 0.15
dynamic_k_profiling_window: 50             # profiling-based，非静态模型

# Prefetch
prefetch_rate_m: 2                          # sweet spot: m=1→2 收益最大
```

### 放弃的设计点（有实验依据）

| 方案 | 放弃原因 |
|------|---------|
| EvictCost-aware 双目标缓存 | E1: ρ=0.78，LFU 全面优于 Joint 策略 |
| Cache Pinning | E2: LFU 已天然覆盖 top-1/2，pin 无额外效果 |
| Dynamic K MLP 升级 | E5: MLP RMSE 仅比解析模型低 0.016，不值得复杂度 |
| 离线 T_stall 建模 | E3 vs E4: 差 8×，只有 profiling 才可信 |

---

## 五、后续工作

按优先级排序（见 `post_experiment_optimization.md` §7）：

**Phase 1（P0，基础集成）**
- [ ] Metadata 双轨记录（flat_selected_original 传递正确性）
- [ ] LFU-RankGuard 实现（`cache_strategy.py`）
- [ ] Level-1 Dynamic K 阈值 early-stop（`spec_engine.py`）
- [ ] Top-1/2 reroute 保护集成（`placement.py`）

**Phase 2（P1，核心算法）**
- [ ] Alg2_v2 routing 集成到 `build_draft_plan_gpu()`
- [ ] StepProfile 收集与 ProfilingDynamicKController
- [ ] 端到端 benchmark（对比 baseline 的 tok/s 和 speedup）

**Phase 3（P2，在线优化）**
- [ ] Online α calibration（EMA curve_fit）
- [ ] Verify 层间预取优化
- [ ] ATC paper placeholder 填充（需真实 benchmark 数据）

---

## 六、审核结论

**所有核心交付物已完成，文档间逻辑一致，实验设计与结论链路完整。**

主要待办：
1. ATC 论文硬件描述与实际使用的 A100 对齐
2. 真实系统 benchmark 后填充论文 evaluation 中的 tok/s 数字
3. 按 Phase 1→2→3 路线图推进实现

---

*最终审核完成时间：2026-05-29*
