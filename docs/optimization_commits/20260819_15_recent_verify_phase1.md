# Prefer recent verify routes for predictive phase-1

日期：2026-08-19

## 一句话总结

新增独立 `k2_dynamic_f16_3080_active14_phase1_recent` preset，让每轮 phase1 优先使用
上一轮 verify 的近期 route index，而不是反复扫描 lifetime frequency；analysis-only
请求中 phase1 首次消费率从 11.66% 提高到 72.92%，未消费换出字节减少 70.15%。

## 保留边界

原 `k2_dynamic_f16_3080_active14` 不变，新增配置默认值也是
`predictive_phase1_recent_verify=false`。只有显式选择新 preset 或 CLI：

```text
--predictive-phase1-recent-verify true
```

才启用新选择器。两个既有动态最优 preset、K1/K2 predictor、0.97 `first_increase`、
standard sampling、cache slots、victim rule、transfer budget 和 verify/draft prefetch 均未
改变。

## 根因证据

提交 `0a40f64` 的 source lifecycle telemetry 在同一 MMLU-Pro validation 第 0 条、512
固定输出、active14、transfer-aware profile 中记录：

- phase1 每轮固定提交 4 个，268 轮共 1072 次；
- 只有 125 个 publication 首次消费，命中率 11.66%；
- 1012 个已换出 publication 中 938 个从未消费；
- 未消费换出为 8.24 GiB；
- 1072 次提交仅覆盖 104 个 layer-expert，90.3% 是重复换入；
- 重复换入 step gap 的中位数为 5；
- rank 1--4 首次消费率分别约 11.2%、8.6%、9.7%、12.3%，不存在“保留 top-1”这种
  单调截断点。

因此问题不是 budget 尾部质量，而是 lifetime access frequency 缺少近期条件：少数历史
高频 miss 在 last segment 被不断换入、很快换出，却与下一轮真实 route 相关性很低。

## 算法修改

phase1 仍只面向 draft 的最后一个 segment，仍最多提交 `predictive_phase1_budget` 个：

1. opt-in 模式从 `verify_segment_index` 取目标 layer range 的 top candidates；
2. priority 继承 index 已有的 verify score、activation count、age penalty 和 TTL；
3. index 有数据时只使用 recent verify candidates，不用 lifetime frequency 补低质量尾部；
4. 第一轮或 index 为空时回退原 frequency selector；
5. 复用 `candidates_for_layer_range(max_candidates=budget)` 的 ranked cache，避免每轮扫描
   last segment 的全部 CPU expert pool。

新增 profile counter：recent candidate 数、recent round 数、frequency fallback round 数。

## Analysis-only A/B

两次均为同一个 MMLU-Pro sample、相同 seed、512 输出、active14、engine profile、
transfer-aware profile。它们是调度分析实验，不是 optimization validation；带 profile 的
TPOT 不作为正式收益。

| 指标 | 原 frequency phase1 | recent verify phase1 | 变化 |
|:---|---:|---:|---:|
| phase1 published | 1072 | 1056 | -1.49% |
| phase1 first-consumed | 125 | 770 | +645 |
| phase1 首次消费率 | 11.66% | **72.92%** | **6.25x** |
| phase1 evicted-without-consume | 938 | 280 | -70.15% count |
| phase1 未消费换出字节 | 8.24 GiB | **2.46 GiB** | **-70.15%** |
| 全部 prefetch publication | 3508 | 3423 | -2.42% |
| verify execution CPU routes / active routes | 52.12% | **49.23%** | **-2.89 pp / -5.55% relative** |
| `run_draft_prefetch_before` / phase1 round | 13.57 ms | **9.27 ms** | **-31.68% CPU 毛路径** |
| ambiguous lifecycle attribution | 4 | 6 | 均低于 0.2% publication |

新策略 264 个 phase1 round 全部使用 recent candidates，frequency fallback 为 0；这包括
warmup 后保留的 verify history。`verify_segment`/`draft_segment_indexed` publication 数和
decode round 数随 stochastic 执行略有变化，因此不能把 CPU-route 或 profile wall 差值
解释成严格反事实；但 phase1 自身的 6.25x 首次消费率和 70.15% 无效字节下降直接由新
source lifecycle 统计支持。

结果目录：

- frequency baseline：`results/analysis_prefetch_source_rank_mmlu0_20260819/`；
- recent candidate：`results/analysis_phase1_recent_mmlu0_20260819/`。

候选 profile TPOT 为 61.505 ms，baseline 为 66.608 ms；二者都开启重 instrumentation，
按用户要求不作为端到端优化验收，也不更新 59.701 ms 正式最佳点。

## 启动 OOM 与修复

新增 preset 第一次运行时遗漏在 `NANOVLLM_GROUPED_GEMM_FIXED_QWEN3=1` 的 preset
白名单，意外触发 Triton autotuner；10 GiB GPU 在其 256 MiB benchmark cache 申请处
OOM。把新 preset 加入与 active14 相同的 fixed-kernel 环境后正常运行，并增加配置测试
锁定该继承关系。OOM 发生在策略执行前，没有产生可分析结果。

## 测试

覆盖：

- 默认 Config 与原 active14 preset 保持 `recent_verify=false`；
- 新 preset 只增加 opt-in flag，并继承 active14 的 cache/dynamic/kernel 环境；
- recent verify candidate 会覆盖 lifetime frequency 更高但更旧的 candidate；
- verify index 为空时精确回退原 frequency selector；
- predictive runtime、benchmark parser/config 和相邻 prefetch 路径回归。

本提交不删除旧策略；若未来恢复无 profile 的正式验证，新 preset 可单独验收、保留或
回退，不影响当前已保留动态最优。
