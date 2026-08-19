# b1 prefetch lifecycle 与 verify budget 否决点

## b1 analysis profile

为避免在 b1 的 53.726 ms/token 后继续盲扫，使用相同 MMLU-Pro 第 0 条、512 token、
t32/双 NUMA 跑了一条 `collect_profile + engine_profile + transfer_aware_profile` 分析请求。
instrumentation 下 TPOT 为 62.012 ms，不作为正式性能结果；profile 位于
`results/analysis_phase1_recent_t32_b1_mmlu0_20260819/`。

按 publication 到首次消费的 lifecycle 统计：

| source | publications | first-consumed | 首次消费率 |
|:---|---:|---:|---:|
| predictive phase1（b1） | 260 | 212 | **81.54%** |
| verify segment（3 段 x vpb2） | 1560 | 1393 | **89.29%** |
| draft segment indexed | 711 | 681 | **95.78%** |

phase1 的最高排名候选仍有明显净价值，与 budget0 回退一致。verify 按 16-layer segment
拆分后，首次消费率依次为 92.31%（480/520）、88.27%（459/520）、87.31%
（454/520）。最后一段浪费略高，因此产生“全局 vpb1”和“仅最后段减为 1”两个候选。

其它关键分析量：phase1 260 个 rounds 全部使用 recent index、frequency fallback=0；
verify/draft/phase1 分别提交 1560/711/260 个 expert；没有 late transfer 或 timeout。

## 否决 1：全局 verify budget 1

| 配置 | TPOT | decode rounds | mean round wall | validation |
|:---|---:|---:|---:|:---|
| b1 + vpb2 | **53.726 ms** | 261 | **105.187 ms** | 512 token，valid |
| b1 + vpb1 | 56.203 ms | 270 | 106.370 ms | 512 token，valid |
| 变化 | **+2.478 ms / +4.61%** | +9 | +1.183 ms / +1.12% | 无错误 |

输出从第 3 个 token 起分叉，但 TPOT、rounds 与每轮成本都回退。第二 verify candidate
不是可以整体删除的低质量尾部，继续保留全局 vpb2。结果目录：
`results/tpot_phase1_b1_vpb1_20260819/`。

## 否决 2：per-segment 2/2/1

临时加入默认空列表的 per-segment 配置，仅把消费率最低的最后一段从 2 降到 1：

| 配置 | TPOT | decode rounds | mean round wall | validation |
|:---|---:|---:|---:|:---|
| b1 + 2/2/2 | **53.726 ms** | 261 | **105.187 ms** | 512 token，valid |
| b1 + 2/2/1 | 54.449 ms | 257 | 108.263 ms | 512 token，valid |
| 变化 | **+0.724 ms / +1.35%** | -4 | **+3.076 ms / +2.92%** | 无错误 |

候选虽少 4 个 rounds，平均 round wall 明显变慢；输出从第 63 token 起分叉。按门禁
完全撤销 Config/CLI/runtime/prefetcher/test 的 per-segment 实现，不保留未获收益的运行时
复杂度。结果目录：`results/tpot_phase1_b1_vpb221_20260819/`。

## 当前结论与下一证据点

- phase1 budget1 与 verify vpb2 是当前相邻扫描保留点；
- 单看 source 或 segment 消费率不足以安全删除第二候选；
- profile 中同步 boundary submit 在 instrumentation 下仍有明显可见成本，因此又测试了
  当前低预算路径能否由已有 async boundary worker 隐藏；结果见下节；
- 所有后续运行时改动继续以一条 production-off 真实请求 TPOT 作保留门禁。

## 否决 3：async verify boundary worker

临时增加默认关闭的 benchmark 开关，复用已有 worker，把 boundary ranking/transfer
submit 从主线程移出；候选 summary 确认
`NANOVLLM_VERIFY_BOUNDARY_PREFETCH_ASYNC=1`：

| 配置 | TPOT | decode rounds | mean round wall | validation |
|:---|---:|---:|---:|:---|
| b1 同步 boundary | **53.726 ms** | 261 | **105.187 ms** | 512 token，valid |
| b1 async boundary | 57.504 ms | 267 | 110.054 ms | 512 token，valid |
| 变化 | **+3.778 ms / +7.03%** | +6 | **+4.867 ms / +4.63%** | 无错误 |

输出从第 63 token 起分叉，但 TPOT、rounds 与 round wall 全部回退。worker queue、drain、
锁以及同一 H2D/copy resource 的竞争没有隐藏提交，反而放大关键路径；这与旧路径的 async
负结果一致。显式开关、metadata 字段和测试已全部撤销，继续固定 async=0。结果目录：
`results/tpot_phase1_b1_verify_async_20260819/`。

## 一句话总结

b1 profile 证明 phase1/verify/draft 首次消费率依次为 81.5%/89.3%/95.8%；verify 全局
vpb1、分段 2/2/1 和 async boundary 均回退，因此保持 b1+vpb2+sync，并撤销候选代码。
