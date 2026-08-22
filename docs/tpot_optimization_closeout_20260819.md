# TPOT 优化路线最终收尾

日期：2026-08-19；最后更新：2026-08-22

硬件：RTX 3080 10 GiB，2 x Xeon Gold 5218R

模型：Qwen3-30B-A3B

范围：从 single-weight 到 recent phase1 budget1、三个被否决的 verify 候选，以及
随后保留的 ghost8 cache 保护

> 本文是本轮路线的权威收尾入口。较早的
> [`tpot_optimization_final_review_20260813.md`](tpot_optimization_final_review_20260813.md)
> 保留完整历史推导；其中后半部分出现的“当前”只表示当时快照。

> **2026-08-22 公平性更正：** baseline 只允许 16 total CPU threads。此前 t32 路线同时
> 改变算法与资源，旧数值只能作为历史探索，不能再作为公平性能证据。全部内置优化 preset
> 已统一到 16 threads，并在完全相同资源下重跑累计链；详细证据见
> [`20260822_30_uniform_t16_fairness_revalidation.md`](optimization_commits/20260822_30_uniform_t16_fairness_revalidation.md)。

## 1. 最终结论

当前基于同资源一请求门禁推荐的 preset 是：

```text
k2_dynamic_f16_3080_active14_phase1_recent_b1_ghost8_lutfuse
```

2026-08-22 production-off 公平复测中，所有候选均使用 16 total CPUInfer threads、双
NUMA 2 x 8。budget4/budget2/budget1/ghost8/LUT fusion 依次为 60.909540、55.043059、
54.627249、53.336804、**52.566035 ms/token**，每个相邻点均为正，累计改善 **13.70%**。
因此 52.566035 是当前同资源、同门禁已测最低点；旧 53.725789/t32 不再称为公平最佳。
配置保留动态 draft 长度框架，并采用：

- single-weight、llamafile F16 CPU expert、BF16 hidden；
- workload-sized active14 cache，GPU memory utilization 0.98；
- 动态 K1/K2、`first_increase`、静态 `td/tv=97/100`；
- segment 16、verify prefetch budget 2、staging 0；
- recent-verify phase1，只提交最高排名的 1 个候选；
- 对 8 steps 内被换出又重载的 expert 提供 8-step ghost 保护；
- 预热的 fused cache LUT update，单次 publication 只发出一个映射 kernel；
- CPUInfer 16 total threads，2 个 NUMA pool，各 8 threads；
- warmup 1024 tokens。

它没有覆盖旧配置。ghost8 是只关闭 LUT fusion 的直接 fallback，原 b1 同时关闭 ghost 和
fusion；以下 preset 也仍可独立选择：budget2、
budget4/recent、active14，以及可容纳完整 8192 context 的
`k2_dynamic_f16_3080`。本次实测请求的物理
KV capacity 为 1536 tokens，足以覆盖 107-token prompt + 512-token output，但不应把
workload-sized b1 当作任意长上下文的默认配置。

### 1.1 59.701 所在提交，以及后续是否建立在其上

**59.701 ms/token 位于提交 `296cf59`：**

```text
296cf59 perf(moe): skip unused legacy mask copies
```

该提交移除了 legacy llamafile F16 后端从不读取的 host expert-mask D2H 刷新；48 个
MoE layer 的 draft/verify CUDA graph 都因此删除了冗余 memcpy node。它从
65.517 降到 59.701 ms/token（-8.88%），直接证据见
[`20260818_11_skip_legacy_mask_d2h.md`](optimization_commits/20260818_11_skip_legacy_mask_d2h.md)。

`git merge-base --is-ancestor 296cf59 HEAD` 返回成功，所以从 `9eea588` 到当前 HEAD 的
所有后续工作都**累计包含** `296cf59`，并非绕开 59.701 的另一条分支。不过“包含旧优化”
与“能把一次随机请求的数值逐提交相减”不是一回事：输出在中途会分叉，旧提交在同日复跑
也从历史 59.701 变为 56.846 ms/token。

后续收益应这样判读：

| 累计节点 | TPOT | 与直接对照 | 是否超过可比锚点 |
|:---|---:|:---|:---|
| `296cf59` 历史请求 | **59.701 ms** | 相对优化前 -8.88% | 当时超过 KT |
| `296cf59` 2026-08-19 同日复跑 | **56.846 ms** | 同一旧提交的漂移锚点 | 当日全局锚点 |
| route-mask 复用 `cd2cb06` | 62.637 ms | 相对直接前序 70.663，-11.36% | 未超过 56.846 |
| 关闭 production consumption `ac254df` | 60.526 ms | 相对 62.637，-3.37% | 未超过 56.846 |
| recent phase1 | 60.420 ms | 相对 active14 60.526，-0.175% | 未超过 56.846 |
| recent + t32（历史、资源不公平） | 59.673 ms | 相对 t16 60.420，-1.24% | 仅作历史探索 |
| phase1 budget2 `1602a85`（历史 t32） | 55.169 ms | 相对 budget4 59.673，-7.55% | 算法收益需由 t16 复测确认 |
| phase1 budget1 `7c5e139`（历史 t32） | 53.726 ms | 相对 b2 -2.62%，相对 b4 -9.97% | 旧资源口径的低点 |
| ghost8（2026-08-21，历史 t32） | 54.393 ms | 相对同日 b1 57.294，-5.06% | 旧资源口径的正收益门禁 |
| ghost8 + LUT fusion（历史 t32） | 54.236 ms | 相对同日 ghost8 -0.29% | 旧资源口径的正收益门禁 |
| budget4→b2→b1（2026-08-22，公平 t16） | 60.910→55.043→54.627 ms | 相邻 -9.63%、-0.76% | 算法收益在同资源下全部复现 |
| ghost8→LUT fusion（2026-08-22，公平 t16） | 53.337→**52.566 ms** | 相邻 -2.36%、-1.45% | **当前同资源已测最低点** |

所以答案是：**后续路线确实以 `296cf59` 为累计基础，并最终拿到了额外收益。**在修正
资源口径后，budget2、budget1、ghost8、LUT fusion 又在同一 16-thread 链上逐项通过门禁；
当前 52.566 比历史 59.701 数值低约 11.95%，但跨日期随机请求仍不作严格隔离归因。

## 2. 测量口径与证据边界

正式保留门禁使用 MMLU-Pro validation 第 0 条、seed 20260719、固定输出 512 tokens、
temperature 0.6、single-weight F16，关闭 collect/engine/transfer profile。每个运行时候选
至少跑一条真实请求，只有正收益且 validation 通过的可用版本才保留；负收益候选撤销代码，
但保留结果与文档。

单请求仍有三个限制：

1. speculative sampling 轨迹可能在几十个 token 后分叉，decode rounds 也会变化；
2. CPU NUMA、cache、温度和后台状态会造成跨日/跨进程漂移；
3. instrumentation profile 明显放大 TPOT，只用于定位，不能替代 production-off 数值。

因此当前 52.566 是“同资源已测点中的单请求最优”，不是跨数据集统计最优或硬件理论下界。
更强的发布结论仍需同 sampling 参数、多个独立 seed、MMLU-Pro/MT-Bench/HumanEval holdout
的配对复测。

正式 budget1 artifact 的文件名和 `optimized_config` 字段仍显示父 preset `..._b2`，因为
候选先以 `manual_overrides.predictive_phase1_budget=1` 完成门禁，确认正收益后才由
`7c5e139` 固化为独立 `..._b1` preset。artifact 的实际 metadata 明确记录 budget1；随后
b1 preset 的测试与 analysis profile 也锁定相同配置，不是误把 budget2 结果改名。

## 3. 与最新 KTransformers 的关系

最新 KT suite 位于：

```text
../ktransformers/benchmark_outputs/
  ktransformers_qwen3_cpu_experts_tpot_suite_20260818-064511/
```

其中 graph-replay/model-forward-only 均值为：MMLU-Pro 66.028、MT-Bench 66.161、
HumanEval 66.148 ms/token；MMLU-Pro 第 0 条为 71.553 ms/token。当前 Nano t16/fusion 的
52.566 是完整 `llm.step()` TPOT，数值上比 KT 最低 suite mean 低约 20.39%。但两边
prompt formatting、sampling、seed、EOS 和计时边界并不严格相同，因此这里只能作保守的
跨系统参考，不能当逐 token A/B。

KT 该配置确实使用 BF16 CPU expert；Nano 当前使用同源 KTransformers CPUInfer/
llamafile 算子的 F16 expert。exact-route、双 NUMA 微基准中 F16/BF16 五个点的绝对差异
都不超过 0.61%，所以“KT 用 BF16”不是性能差距的充分解释。更关键的结构差异是：

- Nano 用 active GPU expert cache 接走大约一半 route，KT 测试配置的 expert op 主要在 CPU；
- Nano 用动态 K1/K2 speculative decode、cache/prefetch 和 single-weight；
- Nano 已删除冗余 mask copy、诊断扫描并减少低价值 phase1 transfer；
- Nano 的 qlen 主要为 2/3，而 KT 逐 token 路径主要是 qlen1/全 8 route CPU；
- 两边 wrapper、graph 边界和采样控制不同，不能只看一个 CPU kernel 的 dtype。

所以“换成 KT 同源 CPU 算子”本身已经完成；下一阶段的主要空间在 CPU 任务组织、
cache/prefetch 决策和 CPU/GPU exposed tail，而不是再做一次同名算子替换。

## 4. 完整提交、文档与一句话总结

下表覆盖从 `8943dd4` 到本文更新提交的完整路线，包括运行时优化、分析基础设施、回退和
文档补记。

| commit | 类型 / 文档 | 一句话总结 |
|:---|:---|:---|
| `8943dd4` | perf / [01](optimization_commits/20260812_01_single_weight.md) | CPUInfer 复用单份 expert 权重，消除双份权重的内存与换页压力。 |
| `8a90da6` | fix / [04](optimization_commits/20260812_04_repeat_engine_cleanup.md) | 重复 benchmark 间释放 engine/CUDA 状态，恢复多次测量可信度。 |
| `4123895` | perf / [02](optimization_commits/20260812_02_k1_f16_vpb2.md) | 固化 F16、K1、verify budget2 的首个本机优化 preset。 |
| `c07ed02` | correctness / [03](optimization_commits/20260812_03_sampling_alignment.md) | 对齐 top-k/top-p speculative sampling 语义。 |
| `d79e12b` | docs / [KT 对照](tpot_vs_ktransformers_20260812.md) | 汇总第一阶段 KT 对照、profile 和负结果。 |
| `3475ea7` | perf / [05](optimization_commits/20260812_05_segment16.md) | segment 12→16，三轮均值改善 1.84%。 |
| `2868f7e` | perf / [06](optimization_commits/20260813_06_reclaim_staging_cache.md) | 回收未使用 staging slot 为 active cache，三 seed 均正、均值改善 5.11%。 |
| `cc37b2f` | perf / [07](optimization_commits/20260813_07_fixed_decode_grouped_gemm.md) | decode qlen 使用固定 grouped-GEMM 配置，均值改善 1.60%。 |
| `714cec5` | perf / [08](optimization_commits/20260813_08_workload_sized_warmup.md) | warmup 与 max context 解耦，释放 workload-sized KV/active-cache 空间。 |
| `74a6521` | perf / [09](optimization_commits/20260813_09_dynamic_k1_k2_tpot.md) | 保留 full-context-safe 动态 K1/K2、0.97 `first_increase` preset。 |
| `8b82ce3` | perf / [10](optimization_commits/20260813_10_dynamic_active14.md) | 保留 workload-sized active14 动态配置，三轮均值 63.713 ms。 |
| `dbe6b25` | docs / [历史综述](tpot_optimization_final_review_20260813.md) | 首次全局审计 profile 与后续路线。 |
| `af3de30` | docs / [历史综述](tpot_optimization_final_review_20260813.md) | 深化机会排序与证据边界。 |
| `296cf59` | perf / [11](optimization_commits/20260818_11_skip_legacy_mask_d2h.md) | 删除 legacy F16 未使用 mask D2H，59.701 ms 并首次超过最新 KT 参考。 |
| `104e7eb` | docs / [历史综述](tpot_optimization_final_review_20260813.md) | 记录 KT 最新 suite、计时口径和 crossover。 |
| `9eea588` | perf / [12](optimization_commits/20260819_12_skip_unprofiled_draft_m3.md) | profile-off 跳过纯诊断 draft M3 统计；当时无独立 production TPOT A/B。 |
| `461165c` | perf / [13](optimization_commits/20260819_13_numpy_draft_histogram_collect.md) | 稀疏 NumPy view 加速 draft histogram；当时无独立 production TPOT A/B。 |
| `4175186` | docs / [历史综述](tpot_optimization_final_review_20260813.md) | 收束 metadata/cache 审计。 |
| `0a40f64` | telemetry / [14](optimization_commits/20260819_14_prefetch_source_lifecycle_telemetry.md) | 加入 source residency/首次消费/换出归因，production-off 无统计负担。 |
| `41008d4` | perf / [15](optimization_commits/20260819_15_recent_verify_phase1.md) | phase1 改用 recent verify route，首次消费率提高 6.25x。 |
| `dc2c2d4` | docs / [15](optimization_commits/20260819_15_recent_verify_phase1.md) | 补齐 recent phase1 分析证据。 |
| `28ca880` | analysis / [16](optimization_commits/20260819_16_cpuinfer_precision_numa_analysis.md) | 补齐 F16/BF16、route density 和双 NUMA 同口径微基准。 |
| `cd2cb06` | perf / [17](optimization_commits/20260819_17_reuse_verify_cpu_route_mask.md) | 复用 verify CPU route mask，两套请求相对直接前序均正。 |
| `2654eb4` | rejected / [18](optimization_commits/20260819_18_verify_histogram_numpy_collect.md) | verify NumPy collect 微基准快但端到端回退。 |
| `f844475` | revert / [19](optimization_commits/20260819_19_revert_verify_histogram_numpy_collect.md) | 恢复端到端更快的 PyTorch verify collect。 |
| `1699642` | docs / [18](optimization_commits/20260819_18_verify_histogram_numpy_collect.md) | 补记正式 rollback 验证。 |
| `ac254df` | perf / [20](optimization_commits/20260819_20_skip_production_verify_consumption.md) | production-off 跳过 consumption 诊断 map，单请求改善 3.37%。 |
| `74cb7f2` | docs / [20](optimization_commits/20260819_20_skip_production_verify_consumption.md) | 补齐 production fast path 证据。 |
| `da55218` | rejected docs / [21](optimization_commits/20260819_21_rejected_skip_verify_meta_profile.md) | 删除 verify metadata 聚合回退 15.48%，候选代码撤销。 |
| `2bf1efc` | docs / [15](optimization_commits/20260819_15_recent_verify_phase1.md) | recent phase1 正式单请求改善 0.175%。 |
| `b8be0aa` | historical perf / [22](optimization_commits/20260819_22_cpuinfer_t32.md) | 当时保留双 NUMA 2x16/t32；现因资源不公平降级为历史探索。 |
| `425a612` | historical docs / [22](optimization_commits/20260819_22_cpuinfer_t32.md) | 记录旧 t32 验收；结论已由公平 t16 复测取代。 |
| `d14f133` | rejected docs / [22](optimization_commits/20260819_22_cpuinfer_t32.md) | t28 比 t32 回退 1.55%，不新增 preset。 |
| `597f1e6` | rejected docs / [23](optimization_commits/20260819_23_rejected_cpuinfer_groupmin2.md) | group_min2 微基准局部快但 TPOT 回退 1.37%，代码撤销。 |
| `1602a85` | perf / [24](optimization_commits/20260819_24_phase1_budget2.md) | phase1 budget4→2，TPOT 55.169 ms，刷新当时最佳。 |
| `7c5e139` | perf / [25](optimization_commits/20260819_25_phase1_budget1.md) | phase1 budget2→1，TPOT 53.726 ms，刷新当前最佳。 |
| `e8f549e` | rejected docs / [25](optimization_commits/20260819_25_phase1_budget1.md) | budget0 回退 2.83%，证明最高排名 phase1 候选仍有净价值。 |
| `7acb865` | analysis docs / [26](optimization_commits/20260819_26_b1_prefetch_lifecycle_and_verify_budget.md) | 记录 b1 lifecycle 及 vpb1、2/2/1 两个负结果。 |
| `1290bee` | rejected docs / [26](optimization_commits/20260819_26_b1_prefetch_lifecycle_and_verify_budget.md) | async verify boundary 回退 7.03%，开关与实现完全撤销。 |
| 本次更新 | perf / [27](optimization_commits/20260821_27_predictive_ghost8.md) | 8/8 短期 ghost 保护同日一请求改善 5.06%，原 b1 保留为 fallback。 |
| 本次更新 | perf / [28](optimization_commits/20260821_28_fused_cache_lut_updates.md) | 预热并禁止动态索引 specialization 的 LUT fusion 再改善 0.29%。 |
| 本次更新 | analysis / [29](optimization_commits/20260821_29_publish_batching_shadow.md) | LUT fusion 后跨层 publication batching 的理想上界不到 0.1% TPOT，调整后续排序。 |
| 本次更新 | fairness/perf / [30](optimization_commits/20260822_30_uniform_t16_fairness_revalidation.md) | 全部 preset 固定 t16；同资源累计链逐项为正，最终 52.566 ms/token。 |
| 本次更新 | rejected/analysis / [31](optimization_commits/20260822_31_rejected_ghost16_after_t16_profiles.md) | t16 profile 定位 CPUInfer sync；ghost16 回退 5.69%，不新增 preset。 |
| 本次更新 | rejected/analysis / [32](optimization_commits/20260822_32_rejected_dynamic_threshold095_t16.md) | 公平 t16 下动态门限 0.97→0.95 回退 2.72%；保留 0.97，不再盲扫静态门限。 |
| 本次更新 | rejected/analysis / [33](optimization_commits/20260822_33_rejected_cpuinfer_static_schedule.md) | native 分项定位三次 GEMM；static 微基准正、端到端回退 2.04%，不启用。 |
| 本次更新 | rejected/analysis / [34](optimization_commits/20260822_34_rejected_lru_frequency_tiebreak.md) | lifetime frequency 破 LRU 同分使公平 t16 TPOT 回退 5.42%，候选代码撤销。 |
| 本次更新 | constraint / [35](optimization_commits/20260822_35_uncompressed_weight_constraint.md) | 撤回全部压缩权重路线；后续只做保持 F16/BF16 表示的等价优化。 |

每个实际保留的运行时版本都在同一提交中包含文档，或紧随一个 docs-only 补记提交。
`9eea588`、`461165c` 只能声明分析/微基准收益，不能追溯性声称独立 TPOT 收益；
`0a40f64` 是 telemetry，不是性能提交。

## 5. Profile、分析与结果目录总索引

### 5.1 single-weight、基础 preset 与 GPU/cache

| 证据 | 内容 | 结论 |
|:---|:---|:---|
| `results/single_weight_*` | single/double weight、F16 K1、vpb、sampling、batch 2/3/5 | single-weight 可用；F16 K1/vpb2 是后续起点。 |
| `results/single_weight_f16_k1_vpb2_512_repeats3/` | 第一阶段三轮 512-token | K1 F16 vpb2 均值约 65.443 ms。 |
| `results/single_weight_f16_k1_vpb2_segment{8,16}_*` | segment 8/16 | segment16 降低 boundary 次数并保留。 |
| `results/*staging0*`、`results/active14_*` | staging 回收、active12/13/14 | workload-sized active14 最优；长上下文另用 safe preset。 |
| [05](optimization_commits/20260812_05_segment16.md)–[10](optimization_commits/20260813_10_dynamic_active14.md) | 每一步正式数据和 fallback | 构成 `296cf59` 之前的累计主线。 |

### 5.2 动态 draft 长度

| 证据 | 内容 | 当前结论 |
|:---|:---|:---|
| `results/dynamic_tpot_k2_threshold{090,095,097}_*` | 动态 K1/K2 门限扫描 | 0.97 是已测保留点。 |
| `results/dynamic_k2_active14_threshold097_gmu098_512_r{0,1,2}/` | active14 三轮正式请求 | 均值 63.713 ms，证明 K2 可达且动态框架可用。 |
| `results/dynamic_k2_best_draft_op_profile_64/` | draft per-op instrumentation | 只看热点占比，不作 TPOT 数值。 |
| `results/tpot_phase1_recent_t32_threshold098_20260819/` | 历史 t32 的 0.98 候选 | 61.020 vs 59.673 ms，回退 2.26%；只支持保留 0.97 的历史判断。 |
| [`draft_tpot_stop_policy_analysis.md`](optimize_ops/draft_tpot_stop_policy_analysis.md) | stop-policy 全史 | 下一步必须估计 K2 相对 K1 的边际收益，而非继续盲扫单门限。 |

### 5.3 CPUInfer 与 KT 同源算子

| 证据 | 内容 | 当前结论 |
|:---|:---|:---|
| `results/analysis_cpuinfer_precision_numa_20260819/` | F16/BF16、单/双 NUMA、稀疏 route | dtype 差异 ≤0.61%；双 NUMA 有效。 |
| `results/analysis_cpuinfer_threads_20260819/` | t12–t40 初筛 | 历史微基准曾指向 t32，但超过 baseline 资源上限，不再产生可保留 preset。 |
| `results/analysis_cpuinfer_nano_groupmin_20260819/` | exact-Nano group_min | group_min2 只改善 qlen1，端到端被否决。 |
| `results/analysis_cpuinfer_nano_mblock_20260819/` | m_block 1–128、qlen1/2/3 | m4 与当前 m32 差异落在噪声内，无端到端候选。 |
| `results/tpot_phase1_recent_t{28,32}_20260819/` | t28/t32 一请求 | 历史线程探索；均超过公平上限，不作为当前证据。 |
| [`verify_cpuinfer_overlap_ktransformers_report.md`](optimize_ops/verify_cpuinfer_overlap_ktransformers_report.md) | CPU/GPU overlap 与 segment 尾部 | CPU work 和 exposed sync tail 仍是主要优化对象。 |

### 5.4 metadata、route-mask 与 prefetch lifecycle

| 证据 | 内容 | 当前结论 |
|:---|:---|:---|
| `results/tpot_ab_active14_formal_20260819/` | `296cf59` 同日旧提交锚点 | 56.846 ms，说明历史单点存在漂移。 |
| `results/tpot_ab_route_metadata_20260819/` | route mask 与 verify collect | route-mask 正；NumPy verify collect 负。 |
| `results/tpot_active14_skip_consumed_diag_20260819/` | production consumption fast path | 60.526 ms，相对直接前序 -3.37%。 |
| `results/tpot_active14_skip_verify_meta_profile_20260819/` | 删除 metadata 聚合 | 回退 15.48%；说明存在隐含 pacing/时序影响。 |
| `results/analysis_prefetch_source_{lifecycle,rank}_mmlu0_20260819/` | source/rank 驻留、消费与换出 | recent phase1 的依据；消费率不是充分优化目标。 |
| `results/analysis_phase1_recent_mmlu0_20260819/` | frequency vs recent phase1 | 首次消费率 11.66%→72.92%，无效换出字节降 70.15%。 |
| `results/analysis_phase1_recent_t32_b1_mmlu0_20260819/` | 历史 t32 b1 instrumented profile | phase1/verify/draft 首次消费率 81.54%/89.29%/95.78%；结构性假设需 t16 profile 复核。 |

### 5.5 b1 相邻点与最终边界

| 结果目录 | TPOT | 决定 |
|:---|---:|:---|
| `results/tpot_phase1_recent_t32_budget2_20260819/` | 55.169 ms | 历史 t32 b2 门禁；当前由公平 t16 b2 取代。 |
| `results/tpot_phase1_recent_t32_budget1_20260819/` | 53.726 ms | 历史 t32 b1 门禁；不再是公平最佳。 |
| `results/tpot_phase1_recent_t32_budget0_20260819/` | 55.248 ms | budget0 否决。 |
| `results/tpot_phase1_b1_vpb1_20260819/` | 56.203 ms | 全局 vpb1 否决。 |
| `results/tpot_phase1_b1_vpb221_20260819/` | 54.449 ms | 分段 2/2/1 否决并删除实现。 |
| `results/tpot_phase1_b1_verify_async_20260819/` | 57.504 ms | async boundary 否决并删除实现。 |
| `results/tpot_phase1_b1_baseline_20260821/` | 57.294 ms | ghost8 的同日直接 baseline。 |
| `results/tpot_phase1_b1_ghost8_20260821/` | **54.393 ms** | 同日改善 5.06%，保留独立 preset。 |
| `results/tpot_phase1_b1_ghost8_lutfuse_20260821/` | 59.467 ms | 未正确预热产生 611.618 ms JIT 尖峰，否决该实现。 |
| `results/tpot_phase1_b1_ghost8_lutfuse_prewarm_20260821/` | **54.236 ms** | 修正版相对 ghost8 再改善 0.29%，保留独立 preset。 |
| `results/tpot_t16_fair_recent_b4_20260822/` | 60.910 ms | 当前公平 budget4 锚点。 |
| `results/tpot_t16_fair_recent_b2_20260822/` | 55.043 ms | 相对公平 b4 改善 9.63%。 |
| `results/tpot_t16_fair_b1_20260822/` | 54.627 ms | 相对公平 b2 改善 0.76%。 |
| `results/tpot_t16_fair_b1_ghost8_20260822/` | 53.337 ms | 相对公平 b1 改善 2.36%。 |
| `results/tpot_t16_fair_b1_ghost8_lutfuse_20260822/` | **52.566 ms** | 当前同资源已测最低点。 |
| `results/analysis_t16_b1_ghost8_lutfuse_{latency,lifecycle}_20260822/` | 54.118/57.128 ms | 只作低扰动热点和 lifecycle 分析，不作正式 TPOT。 |
| `results/tpot_t16_b1_ghost16_lutfuse_20260822/` | 55.559 ms | 相对 ghost8/fusion 回退 5.69%，不新增 preset。 |
| `results/tpot_t16_b1_ghost8_lutfuse_threshold095_20260822/` | 53.994 ms | 动态门限 0.97→0.95 回退 2.72%，不新增 preset。 |
| `results/tpot_t16_b1_ghost8_lutfuse_static_schedule_20260822/` | 53.638 ms | static worker scheduling 微基准正、TPOT 回退 2.04%，不启用。 |
| `results/tpot_t16_b1_ghost8_lutfuse_lrufreq_20260822/` | 55.417 ms | LRU 同龄时按 lifetime frequency 破同分回退 5.42%，实现撤销。 |

b1 profile 共记录 2531 次 source-tracked publication；verify/draft/phase1 submit 为
1560/711/260，没有 late transfer 或 timeout。verify 三段首次消费率为
92.31%/88.27%/87.31%，但直接削减尾段预算仍回退，说明“是否被消费”没有计入预取
带来的等待规避、cache 保护和时序价值，不能直接作为删除规则。

### 5.6 收尾阶段的只读 trace 再分析

文档初稿完成后，又对上述 b1 profile 做了只读聚合，不运行新候选、不改 runtime。这里
使用 `model_verify_call_records[*].metadata_layer_execution_*`；顶层 graph-capture profile
counter 会混入 replay/capture 的零值，不适合计算 route ratio。

**动态长度实际行为：**260 个 speculative rounds 中，209 轮停在 K1，51 轮进入 K2。
K2 组首 token 实际接受 50/51（98.04%），第二 token 接受 43/51（84.31%）；K1 组产生
367 个输出 token，K2 组产生 144 个。在 instrumentation 下，两组的
`(draft + verify) / produced token` 分别为 63.154 和 55.776 ms。这是被控制器选择后的
条件分组而非反事实 A/B，但证明当前 0.97 确实把 K2 集中在高接受率轮次。

同时，instrumentation 下 K1/K2 组平均 draft 总耗时为 26.73/54.96 ms，verify 为
84.17/102.53 ms；preset 中 `td=97, tv=100` 显然不是逐调用实测成本，而是把 K1→K2
门限调到 alpha1≈0.97 的策略参数。当前 `first_increase` 在执行完 K1 后用 T1 与 T0
决定是否执行 K2，无法提前看到 alpha2；执行完 K2 后再产生 stop signal 已不改变
Kmax2 的行为。这是下一代动态控制器最明确的建模缺口。

**cache oracle：**260 个 verify records 共 219264 条 route，实际 CPU route 为 112284，
即 **51.21%**；单轮 CPU route 数与 verify `total_ms` 的 Pearson 相关系数为 **0.840**。
在保留当前每层 slot 数的前提下，使用整条 trace 的未来信息静态保留每层 top-S expert，
理论 CPU route ratio 为 44.53%；再允许 672 slots 跨层自由贪心分配，才进一步到 43.97%。

这个 oracle 不能直接实现，也没有计入 draft/prefetch 时序，但它给出两个重要边界：

- admission/retention/prediction 的 trace 上限约为 6.68 个 route percentage points；
- 其中仅靠层间重分配的额外上限约为 0.55 points，不能按“当前 miss 高就多给 slot”盲调。

例如 layer0 虽有 20 slots，CPU route ratio 仍为 79.09%，但其 expert 分布很分散，oracle
边际分配反而不会继续给它 slot。下一 allocator 应按“下一 slot 能减少多少 exposed CPU
tail”分配，而不是按 route 总量或 miss ratio 分配。

**transfer/churn：**每个 expert 为 9 MiB，2531 次 submit 约传输 **22.24 GiB**。其中只有
1242 个不同的 `(layer, expert)`，50.93% 的 submit 是该请求内再次传过的对象。更直接的
坏 churn 信号是：232 次“未消费即换出”中有 176 次（75.86%）后来又被传回，重传间隔
中位数仅 11 steps。不同 source 的时序也明显不同：draft/verify/phase1 从 submit 到 ready
平均约 13.52/14.20/25.95 ms，首次消费平均滞后 0.59/2.62/4.64 steps。

因此最值得先做的不是全局减 budget，而是 source/rank-aware admission、短期 ghost
protection 和按预期首次使用时刻排队。8/8 ghost protection 已在 2026-08-21 以独立
preset 落地：同日 b1 57.294、ghost8 54.393 ms/token（-5.06%）。随后 fused LUT commit
把 ghost8 进一步降到 54.236 ms/token（-0.29%）；未预热版本的 JIT 尖峰已记录。原始
trace 还建议评估连续 expert copy、event 和 publication 批量化；LUT 落地后需按新的单次
commit 成本重新计算其上界。

LUT fusion 后又按 ticket submit step 做了 publication grouping shadow：2531 次 publication
最多可归并为 571 个跨层组，但 `(submit step, layer)` 上 2411/2531 是 singleton。按已测
13.565 µs/fused commit 计算，把全部跨层 publication 理想压到 571 次也只省约
26.59 ms/request，即 0.052 ms/token、不到当前 TPOT 的 0.1%，且尚未计入共享二维 LUT、
pointer batch 和延迟 expert 可见性的成本。因此 batching 已降到 CPUInfer native 分解和
source/rank admission 之后。

## 6. 已排除或暂不应重试的方向

- phase1 budget0、verify vpb1、verify 2/2/1、async boundary：均有当前 b1 一请求负结果。
- dynamic threshold 0.98：相对 0.97 回退 2.26%。
- CPUInfer t28、group_min2：分别回退 1.55% 和 1.37%。
- CPUInfer static worker scheduling：qlen2/3 微基准改善 4.17%/0.91%，但公平 t16 TPOT
  回退 2.04%；不能用孤立 GEMM 结果替代端到端 overlap 门禁。
- lifetime frequency 打破 LRU 同一步并列：公平 t16 TPOT 回退 5.42%；离线 choice-difference
  上界不是 miss/tail 收益，不再尝试 LFU/LRU 简单混合或静态加权。
- verify NumPy collect：孤立函数变快但两套端到端均回退，已经 revert。
- 直接删除 verify metadata 聚合：回退 15.48%，先解释 pacing 再改。
- m_block 盲扫：exact-Nano qlen1/2/3 没有超出噪声的候选。
- ghost16：公平 t16 下与 ghost8/fusion 相比回退 5.69%，不再盲扫更长 TTL。
- dynamic threshold 0.95：公平 t16 下相对 0.97 回退 2.72%；结合历史 0.98 负结果，
  不再继续静态门限扫描。
- 仅因 KT YAML 是 BF16 就切回 BF16：同 route 微基准不支持。
- Q8/Q4/INT8/FP8 等压缩或量化权重：用户明确禁止；没有运行时代码或 preset，后续不再评估。
- 全量迁移 KTransformers runtime：Nano 已复用其 CPUInfer 核心，架构迁移风险高且不是
  当前差距的主要来源。

## 7. 全局后续优化机会

### P0：先把下一次收益变成可归因、可泛化的收益

1. **建立稳定 holdout 门禁。** 固定双方 sampling/chat template，至少 3 个独立 seed，
   覆盖 MMLU-Pro、MT-Bench、HumanEval 和不同 prompt/output 长度；同时报告 TPOT、rounds、
   round wall、acceptance 和输出 digest。单请求仍可作每轮快速门禁，holdout 用于发布。
2. **从 CPUInfer native 分项进入精确 F16 算子优化。** exact-current qlen2/3 profile 已证明
   gate/up 占 65–67%、down 约 30%，input copy/merge 仅 1–4%；static scheduling 又在真实请求
   回退 2.04%。权重压缩已明确禁止，后续只允许等价的软件预取、NUMA-local 分块、减少
   中间 buffer 写回或 activation/down 融合；不再扫描 dtype、group_min、m_block 或 worker
   scheduling。
3. **深化 rank/source-aware admission。** 8/8 ghost 已保守落地；简单 lifetime frequency
   破 LRU 同分已回退 5.42%，证明 choice-difference 不能代表收益。下一版必须按 source+rank
   直接预测 next reuse / CPU tail saved，并减去 victim reload、publication 和 overlap 成本。
   不再扫描更长 ghost TTL 或 LFU/LRU 静态混合。
4. **有条件地批量化 transfer/publish。** LUT commit 已融合；当前仍有 2531 次、约
   22.24 GiB publication，但只读上界表明跨层 commit 合并不到 0.1% TPOT。只有 native
   profile 证明 event/H2D 控制面仍显著，且能保持每个 expert 独立可见时刻时，才评估
   NUMA-local pinned packing ring 或一次提交 2–8 个 expert。
5. **谨慎重拟合 active-cache 层间分配。** 在固定 672 slots 下，用下一 slot 的 miss-tail
   边际收益分配，并用 holdout 防止只拟合 sample0。当前未来信息 oracle 表明层间重分配只比
   保持当前 per-layer 数量多降低约 0.55 route points，所以它应与 admission 联合做，不能
   根据 layer miss ratio 单独扫描。
6. **显式化 metadata pacing。** 先记录 worker queue depth、submit→start、start→publish、
   boundary wait 和 stream event；确认 15.48% 负结果来自何种时序，再考虑 C++/批处理替代聚合。

### P1：在现有动态框架上继续扩展

7. **重写 K2 边际决策而不是继续扫静态门限。** 公平 t16 trace 中现有 K2 的第二 draft
   接受 23/24，但把门限从 0.97 放宽到 0.95 仍端到端回退 2.72%，证明 alpha1/接受率不足以
   表达额外 qlen3 verify 与 cache/prefetch 外部性。下一版应在 K1 记录或 shadow 预测 alpha2、
   额外 draft cost、verify qlen3 cost、cache/prefetch credit，直接最小化 K2 相对 K1 的边际
   TPOT；保留现有 0.97 preset 为 fallback。
8. **GPU 小 M MoE/route 融合。** 对 qlen1/2/3 专门融合 reroute/LUT/scatter/reduce、GPU
   cached expert 计算和 CPU result add，减少 workspace 与小 kernel launch。
9. **保持权重格式的 cache/算子融合。** 不允许通过 INT8/Q8/Q4/FP8 增加 cache 容量；只在
   原始 F16/BF16 表示下评估 route/LUT/scatter/reduce 与 expert kernel 的等价融合。
10. **attention/sampling 小核融合。** qk-norm、RoPE、QKV/KV-store 与 sampler 是 CPU MoE
    尾部压低后会显现的固定成本，可由 CUDA graph/event profile 再排序。
11. **自动选择 workload-sized 与 full-context-safe preset。** 按 prompt + max output +
    graph/KV padding 预估容量，能安全容纳时选 b1，否则退回完整上下文动态 preset。

### P2：需要更大改动或不同 workload 才值得启动

12. **并发/批处理专用策略。** 当前结论只适用于单请求；batch 下 CPU route 合并、cache
    共享、prefetch budget 和 dynamic K 都应重新拟合。
13. **更强 draft/tree speculative。** EAGLE、Medusa 或 tree verification 可能突破 K1/K2
    局部最优，但属于模型/图结构级路线，需单独质量与显存预算。
14. **完整 heterogeneous runtime 迁移。** 仅当 native phase profile 证明 Nano wrapper
    仍有大块不可消除开销时，再分阶段迁移 KT scheduler/cache，而不是整体替换。

优先级的核心判断是：当前 adjacent 参数点已经连续出现负收益；下一阶段最可能的增益来自
减少 CPU exposed tail、批量化 prefetch 控制面，以及让动态 K/prefetch 使用真实边际价值。
继续扫描小整数预算或单一阈值的预期收益已经较低。

## 8. 收尾状态

- 同资源推荐：`k2_dynamic_f16_3080_active14_phase1_recent_b1_ghost8_lutfuse`；ghost8
  是关闭 fusion 的直接 fallback，原 b1 同时关闭 ghost 和 fusion。
- 公平 t16 已测最低点：52.566035 ms/token；b4→b2→b1→ghost8→LUT fusion 的相邻改善
  分别为 9.63%、0.76%、2.36%、1.45%，五条均通过 512-token validation。
- 59.701 所在提交：`296cf59`；当前路线完整继承它。
- b1 实现提交：`7c5e139`；其后的三个 verify 候选均被否决且代码已撤销；ghost8 在独立
  preset 中累计继承 b1。
- 动态 draft 的历史最优、active14、recent/b4、b2 与 full-context-safe preset 全部保留；
  旧 t32 名称仅作命令兼容 alias，实际也固定为 16 threads。
- 权重格式红线：禁止任何 Q8/Q4/INT8/FP8 压缩或量化；canonical 保持 F16 single-weight。
- 本文之后不再把旧综述中的 59.701 或 63.713 称为“当前最终状态”。
