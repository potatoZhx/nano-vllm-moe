# TPOT 优化收尾审计与后续路线图

日期：2026-08-13  
硬件：RTX 3080 10 GiB，2 x Xeon Gold 5218R  
模型：Qwen3-30B-A3B  
主要指标：单请求、固定输出长度的 mean TPOT/TBT

## 1. 最终状态

当前保留两个动态长度 preset：

| preset | 用途 | 512-token TPOT（3 个独立进程） | mean / pop std | KV 容量 |
|:---|:---|:---|:---|---:|
| `k2_dynamic_f16_3080` | 8192-context-safe | 66.031 / 68.461 / 65.199 ms | 66.564 / 1.384 ms | 9728 tokens |
| `k2_dynamic_f16_3080_active14` | workload-sized 当前实测最优 | 63.930 / 61.600 / 65.609 ms | **63.713 / 1.644 ms** | 1536 tokens |

两者都保留现有 acceptance predictor + TPOT `first_increase` 框架，参数为
`Kmax=2`、`min_steps=1`、`td/tv=97/100`。在 K1 后继续到 K2 的条件是
`predicted_alpha >= 0.97`。active14 三轮共 12 次 K2，其中 9 次 full accept、3 次
partial accept、无 zero accept；并非伪装成动态模式的 fixed K1。

此前 fixed K1/active12 的三轮均值是 `65.443 ms`。active14 动态配置相对它快
2.64%，相对 active12 动态快 4.28%。但 active14 同时改变了 expert cache，尚无
active14/0.98 fixed K1 的同布局三 seed 对照，所以只能得出“active14 动态配置是当前
端到端实测最优”，不能把全部 2.64% 因果收益归给动态算法。

与 KTransformers 的三个参考点相比，`63.713 ms`：

- 比最可靠 BF16 `122.35 ms` 低 47.9%，速度为 1.92x；
- 比历史最好 F16 `81.90 ms` 低 22.2%，速度为 1.29x；
- 比本日同 prompt F16 `125.886 ms` 低 49.4%，速度为 1.98x。

这些参考的 dtype、线程绑定、prompt 或采样轨迹不完全一致。最保守的跨实现结论是：
当前 nano workload-sized 最优明显超过已留存的 KTransformers 端到端参考；它不是逐
token、同 RNG、同 kernel 的严格配对试验。

## 2. 证据等级与“最优”结论

本轮证据按以下顺序使用：

1. 三个独立进程的 512-token 完整 decode 墙钟 TPOT、固定长度和 validation；
2. 256-token 单点筛选，只用于淘汰候选或决定是否进入三轮复测；
3. instrumentation-off 的 phase/profile counter；
4. CUDA event / torch profiler，只用于定位，不用于性能宣称；
5. microbenchmark，只说明算子边界，不替代端到端结果。

目前能证明的是经验局部最优，而不是数学或硬件全局最优：

- 在已测固定 K1..K6 中，K1 最好；
- 在现有 predictor/TPOT 框架已测 `Kmax` 和 0.90/0.95/0.97 门限中，Kmax2/0.97
  最好且跨 seed 保持真实动态；
- 在已测 active12/13/14 中，active14 workload-sized 配置最好；
- 当前结果没有覆盖所有 prompt 长度、采样分布、并发度、cache ratio、CPU/GPU
  kernel、量化格式或 speculative 算法，因此不能宣称全局最优。

若未来要证明“接近硬件最优”，还需建立 mandatory attention、GPU cached MoE、CPU
miss MoE、KV 带宽和 graph launch 的 roofline 下界，并用 native CPUInfer timing
证明实际路径与下界的差距；现有 profile 还不足以完成这项证明。

## 3. 本轮提交、文档与一句话总结

以下是从 single-weight 开始的完整提交账本。每个性能或功能提交都有对应文档；
`d79e12b` 是中途证据汇总，不是新的运行时优化。

| commit | 文档 | 一句话总结 |
|:---|:---|:---|
| `8943dd4` | `optimization_commits/20260812_01_single_weight.md` | CPUInfer 复用单份 expert 权重，消除双份权重导致的换页/内存压力。 |
| `8a90da6` | `optimization_commits/20260812_04_repeat_engine_cleanup.md` | 重复 benchmark 间释放 engine/CUDA 引用，保证多次均值不会被资源泄漏破坏。 |
| `4123895` | `optimization_commits/20260812_02_k1_f16_vpb2.md` | 固化 F16、K1、verify prefetch budget 2 的首个本机低 TPOT preset。 |
| `c07ed02` | `optimization_commits/20260812_03_sampling_alignment.md` | 对齐 top-k/top-p speculative sampling 语义，属于比较正确性而非正 TPOT 收益。 |
| `d79e12b` | `tpot_vs_ktransformers_20260812.md` | 汇总 KTransformers 对照、profile 证据、口径和已测负结果。 |
| `3475ea7` | `optimization_commits/20260812_05_segment16.md` | segment 12 改为 16，减少 boundary/prefetch 次数，三轮均值改善 1.84%。 |
| `2868f7e` | `optimization_commits/20260813_06_reclaim_staging_cache.md` | 回收未使用的 staging2 为 active12，三 seed 全正、均值改善 5.11%。 |
| `cc37b2f` | `optimization_commits/20260813_07_fixed_decode_grouped_gemm.md` | 为 decode qlen 固定选择 grouped GEMM 配置，均值改善 1.60% 并降低启动耗时。 |
| `714cec5` | `optimization_commits/20260813_08_workload_sized_warmup.md` | synthetic warmup 从 max context 解耦，降低峰值并把 active12 KV 从 8 增至 49 blocks。 |
| `74a6521` | `optimization_commits/20260813_09_dynamic_k1_k2_tpot.md` | 保留 full-context-safe K1/K2、0.97 门限动态配置，较旧动态改善 20.4%。 |
| `8b82ce3` | `optimization_commits/20260813_10_dynamic_active14.md` | 新增 active14/0.98 workload-sized 动态最优，三轮均值达到 63.713 ms。 |

相对较早的核心历史提交也应保留在理解调用链时使用：

| commit(s) | 一句话总结 | 主要分析文档 |
|:---|:---|:---|
| `31f918f`, `8500f98`, `6b6a623`, `6707ffc` | 建立 kt-hybrid verify CUDA graph、segment graph 及修复。 | `verify_optimization_summary.md` |
| `a960165`, `f3938a1`, `d155c5a`, `04a7f66` | 建立 dual-queue/predictive prefetch、adaptive 逻辑和消费侧 profile。 | `verify_per_op_call_chain_breakdown.md` |
| `ce8f556`, `39fcaae`, `25b3c1e` | 引入 acceptance predictor、alpha stop 和动态 TPOT stop。 | `draft_tpot_stop_policy_analysis.md` |
| `2575cb8` | 引入按层不同 slot 分配。 | `qwen3_cpu_experts_slot_buckets_call_chain.md` |
| `d015727` | 去同步 metadata readback、延后 metadata、优化 verify boundary。 | `verify_prefetch_off_metadata_profile_report.md` |
| `2283643`, `30eed35` | 增加 draft per-op 分析并优化 draft plan/schedule。 | `draft_per_op_call_chain_breakdown.md` |
| `944c1e0`, `aaab02a`, `f90db15` | 演进 verify cost model、multi-horizon 和 transfer-aware early stop。 | `verify_time_cost_model_tpot.md` |
| `455c4cf` | 增加 3080 batch 路线和本机基线。 | `tpot_vs_ktransformers_20260812.md` |

## 4. Profile 与分析文档索引

| 文档 | 覆盖内容 | 当前应继承的结论 |
|:---|:---|:---|
| `optimize_ops/qwen3_cpu_experts_slot_buckets_call_chain.md` | Nano 与 KT 从 loader 到 attention/MoE/CPUInfer/graph 的完整调用链 | 两个系统边界不同；局部复用 CPU kernel 比全量算子替换风险小。 |
| `optimize_ops/verify_prefetch_off_metadata_profile_report.md` | metadata `.cpu()` 同步、offload、prefetch on/off | 同步 readback 曾是大热点；关闭后剩余关键路径进入 graph/CPUInfer。 |
| `optimize_ops/verify_cpuinfer_overlap_ktransformers_report.md` | segment event、CPU route 数、CPU/GPU overlap、KT microbench | CPU/GPU 已有 overlap，但 GPU 路径太短，段尾仍等待 CPUInfer。 |
| `optimize_ops/verify_per_op_call_chain_breakdown.md` | verify graph 内逐 op CUDA event | 历史 K4 case 的最大真实项为 `kt.cpuinfer_sync`，copy 与 task create 很小。 |
| `optimize_ops/draft_per_op_call_chain_breakdown.md` | draft graph 内外、attention、MoE、plan、metadata | draft 同时受 GPU MoE、attention、小 kernel/plan 和 graph 外调度影响。 |
| `optimize_ops/verify_time_cost_model_tpot.md` | verify demand/cache/transfer simulator 和 latency model | 旧高 K 模型的主要误差来自 predicted demand + simulated cache，非 bucket base。 |
| `optimize_ops/draft_tpot_stop_policy_analysis.md` | 静态、history、dynamic、transfer-aware stop 策略 | 高 K 上过早停止增加 verify 次数；动态策略必须用真实 TPOT 与多 seed 验收。 |
| `optimize_ops/verify_optimization_summary.md` | verify 优化总史、复现和历史 K6/K12 结果 | 历史最优依赖不同 GPU/cache/backend，不能覆盖本机 F16 K1 结论。 |
| `optimize_ops/ktransformers_integration_strategy.md` | 全替换与迁移两条架构路线 | 短期继续 Nano `kt_direct` 局部优化；长期迁移应分阶段实现 heterogeneous MoE。 |
| `tpot_vs_ktransformers_20260812.md` | 本机最终对照、所有主要筛选、batch 与口径 | 是当前总体结果的主入口。 |

## 5. 热点的统一解释

### 5.1 当前 active14 端到端 profile

三次 512-token profile 均值：

| 指标 | active12 动态 | active14 动态 | 变化 |
|:---|---:|---:|---:|
| TPOT | 66.564 ms | 63.713 ms | -4.28% |
| draft forward / call | 30.777 ms | 30.546 ms | -0.231 ms |
| verify forward / call | 78.360 ms | 77.230 ms | -1.129 ms |
| realized CPU experts | 57.403 | 56.153 | -1.250 |
| CPU route ratio | 0.9068 | 0.8952 | -0.0116 |
| prefetch submits / MoE profile call | 77.65 | 75.35 | -2.30 |

更多常驻专家减少 CPU route 和预取工作，draft/verify 都略降。`cpu_compute_ms` 是异步
累计 profile 值，不可直接与 wall time 相加，也不能用其单项波动否定端到端结果。

### 5.2 当前动态 draft 的 per-op profile

`results/dynamic_k2_best_draft_op_profile_64/` 在 active12 动态路径上开启了
`NANOVLLM_DRAFT_OP_EVENT_TIMING=1`。该模式插入大量 CUDA event，TPOT 被放大到
87.804 ms，只能看占比。按 35 次 draft call 汇总：

| op | 约 ms/draft call |
|:---|---:|
| `layer.total` | 21.277 |
| `layer.attention` | 6.899 |
| `moe.gpu_gate_up` | 3.446 |
| `moe.gpu_down` | 1.924 |
| `moe.draft_reroute` | 0.835 |
| `moe.runtime_metadata_record` | 0.728 |
| `moe.draft_feature_record` | 0.375 |

独立 microbench 中 token features 约 0.312 ms，predictor eager 约 1.664 ms，predictor
CUDA graph 约 0.220 ms，features + graph predictor 约 0.351 ms。由此可见 predictor
热路径有优化空间，但通常只是每 draft call 数个零点几毫秒，不是当前最大的数量级。

### 5.3 历史 verify 深度 profile

历史 K4/cache0.3125 profile 中 verify wall 为 118.692 ms/call，segment CUDA event
116.849 ms；`kt.cpuinfer_sync` 为 76.792 ms/verify，GPU gate/up 11.579 ms、down
4.772 ms、attention 5.218 ms，CPU-to-GPU output copy 只有 0.360 ms。正确解释是：
`cpuinfer_sync` 是 GPU overlap 后仍暴露的 CPU 尾巴，不是完整 CPU compute；不能再与
GPU MoE 取 `max()` 或简单相加。

该数字来自旧 backend/cache/K4，不可直接套到当前 active14 F16/K1-K2；但调用链结论
仍成立：verify 的高潜力方向是减少 CPU miss 或加速 CPU expert，而不是只优化 output
copy。当前路径需要重新做 native CPUInfer 分项计时才能得到可用于定量决策的新占比。

### 5.4 动态长度的实际收益边界

active14 三轮共有 888 个 speculative round（含 2 个 terminal K0），只有 12 个 K2，
约占正常策略 round 的 1.35%。这说明 0.97 门限在本 workload 上已经非常保守：

- 再调阈值只能影响极少 round，单靠参数微调很难产生大幅收益；
- 放宽阈值会迅速增加 qlen=3 verify，固定 K 扫描已经证明这通常不划算；
- predictor/feature 即使完全免费，按约 0.7 ms/draft call 粗略上界，也只是每输出
  token 约 0.4 ms，且“完全免费”并不现实；
- 真正的大收益更可能来自降低每次 draft/verify 成本，或让更长 draft 的 acceptance
  与单位成本同时发生结构性改善。

## 6. 已排除或暂不采用的路线

| 路线 | 结果 | 决策 |
|:---|:---|:---|
| fixed K1..K6 | 75.959 / 85.613 / 83.984 / 90.728 / 109.542 / 103.703 ms（早期同路径 256-token） | 当前 F16 单请求工作点 K1 最好。 |
| 旧动态 Kmax6、19/80 | 83.627 ms，平均 K=2.45 | 阈值太松，已由 Kmax2/0.97 替代。 |
| 0.90 / 0.95 / 0.97 | 68.612 / 69.322 / 66.245 ms 单点 | 0.97 更符合固定 K 证据，并通过三 seed。 |
| predictor Kmax1 | 70.426 ms vs 同日 fixed 66.396 ms | 说明预测热路径有成本；单点差异不是严格因果值。 |
| history-only streak2 / streak8 | 74.905 / 67.869 ms | 不符合继续沿用现有框架的要求且无正收益，已回退、未提交。 |
| LM head + features + predictor tail graph | 73.415 vs 66.245 ms | 改变 RNG/采样轨迹且负收益，已回退、未提交。 |
| active13 | 67.178 vs active12 66.245 ms | 离散缓存点负收益。 |
| active14/0.97 | 65.385 ms，但 KV 仅 512 tokens | 不能完成 67+512 workload，不作为可用 preset。 |
| 额外 qlen=3 graph + 0.996 memory | warmup 后 OOM | 显存边界不允许。 |
| verify prefetch budget 0/1/2/3/4 | 85.317 / 78.949 / 72.317 / 76.313 / 75.959 ms | budget 2 最好。 |
| segment 8 | 76.669 ms | boundary 过多。 |
| rank multiplier 2 | 三轮均值 70.830 vs rank1 70.092 ms | 无正收益。 |
| LFU cache | 86.113 ms | 当前 workload 不如 LRU/profile-weighted 组合。 |
| capture 仅 bs1,2 | 71.646 ms | 无正收益。 |
| top-k20/top-p0.95 | 81.960 ms | 功能已支持，但该采样条件不是性能最优。 |

## 7. 后续优化机会：优先级总表

这里的“潜力”是依据现有 profile 的工程判断，不是尚未测量的性能承诺。

| 优先级 | 方向 | 潜力 | 风险/成本 | 首个验收 |
|:---:|:---|:---|:---|:---|
| P0 | active14 fixed K1 同布局因果对照 | 决策价值高，开发成本几乎为零 | 仅实验成本 | 3 seed，与 active14 dynamic 配对。 |
| P0 | CPUInfer qlen2/3 F16 native 分项计时与专用 kernel | 高 | C++/NUMA/kernel 中高风险 | queue/GEMM/activation/merge 分项 + 3 seed TPOT。 |
| P0 | 压缩 GPU cached expert 权重以同时增加 active experts 和 KV | 高 | 数值、kernel 与加载器高风险 | logits/acceptance 正确；active 数增加；完整 512/8192 容量分档。 |
| P0 | 根据请求长度自动选择 active12-safe / active14-fast 内存档 | 中高 | 配置与容量校验低风险 | 运行前证明 KV capacity >= prompt+max output+graph padding。 |
| P1 | verify qlen2/3 grouped GEMM 专用配置/融合 | 中 | Triton autotune 与回归中风险 | kernel microbench 后必须过端到端 3 seed。 |
| P1 | `reroute + LUT + grouped plan` 单 kernel 融合 | 中 | graph-safe CUDA/Triton 中风险 | per-op event 降低且 TPOT 正收益。 |
| P1 | metadata/feature 四次小 copy 打包、静态 workspace | 低到中 | predictor 输入兼容风险 | predictor 输入 bitwise/容差一致，正常 profile 无事件开销。 |
| P1 | prefetch ranking 缓存、ring descriptor、批量 transfer enqueue | 中 | cache 时序和 acceptance 风险 | late/timeout 不增，CPU route 和 TPOT 同时下降。 |
| P1 | CPU affinity、NUMA 页归属、threadpool 1/2 sweep | 低到中 | 低代码风险，环境敏感 | 独立进程 5 seed，报告 p50/p95 和 NUMA 配置。 |
| P2 | predictor 校准与按 workload/position 的 threshold | 低（当前 workload）/跨 workload 中 | 过拟合风险 | deterministic trace replay + 未见 prompt holdout。 |
| P2 | draft tail 只捕获纯特征/predictor且保持 sampler RNG 次序 | 低 | 极易改变采样轨迹 | 固定随机数输入下 token/概率一致后再测。 |
| P2 | q/k norm、RoPE、split 小 kernel 融合 | 低到中 | attention 正确性风险 | attention 单测、长上下文数值和 TPOT。 |
| P2 | 自动不等长 segment（按预计 CPU tail 划分） | 中 | graph bucket 数和显存风险 | segment event 尾差下降且 graph 全命中。 |
| P2 | CPU expert INT8/INT4/FP8 或 weight packing | 高 | 精度与新 kernel 高风险 | perplexity/logit/acceptance + CPU native throughput + TPOT。 |
| P3 | 更便宜/更准确的 draft model、early-exit/self-spec | 很高 | 算法与训练/权重大改 | acceptance、draft cost、verify rounds共同进入 TPOT 模型。 |
| P3 | EAGLE/Medusa/tree speculative decoding | 很高 | 大规模重写、graph 与采样复杂 | exact sampling 正确性及多 workload TPOT。 |
| P3 | 把 heterogeneous MoE/spec 链迁入 KTransformers | 架构价值，短期 TPOT 不确定 | 很高 | 按 MVP -> spec -> prefetch -> graph 分阶段。 |

## 8. 后续方向的具体分析

### 8.1 首先隔离 active14 与动态策略的贡献

下一阶段第一件事应是用 `cache_ratio=0.109375`、GPU utilization 0.98、相同 F16、
segment16、vpb2、warmup1024，关闭 predictor 并固定 K1，跑相同 seeds。可能出现三种
结果：

- fixed active14 更快：说明当前动态 preset 应保留为“动态最优”，总体默认应另设
  active14 fixed；后续动态工作重点是削减 predictor 成本。
- 两者相当：说明 K2 的少量收益大致偿还 predictor 成本，当前 0.97 已接近局部平衡。
- dynamic 更快：才有直接证据说明 rare high-alpha K2 在同布局上产生净收益。

### 8.2 内存压缩比继续加 cache ratio 更有战略价值

active12 到 active14 只把 CPU route ratio 从 0.9068 降到 0.8952，就带来 4.28% 动态
均值改善，但代价是 KV 从 9728 tokens 降到 1536 tokens。继续裸增 active experts
已经没有显存空间；应改变每个 cached expert 的字节数，而不是继续挤 KV：

1. GPU cache 权重 FP8/INT8，CPU 主权重仍保留 F16；
2. 只量化最占空间的 gate/up/down 权重，route/gate 保持高精度；
3. 加载或换入时一次性 dequant/pack 到 kernel 所需布局，避免每 token host 转换；
4. 分别建立 full-context-safe 与 workload-sized 档位。

若数值误差改变 draft 分布或 acceptance，必须重新比较最终输出质量和标准 speculative
sampling，而不能只看 kernel tok/s。

### 8.3 CPU expert 是 verify 的最大结构性机会

旧 verify profile 指向 `kt.cpuinfer_sync`，当前 active14 仍有约 89.5% CPU route ratio。
建议先补 native timing，再决定 kernel 方向：

1. 记录 queue wait、expert grouping、gate/up、SiLU/mul、down、merge、output ready；
2. 按 qlen2 与 qlen3 分桶，避免均值掩盖短 shape；
3. 检查 F16 实际是否走最合适 ISA，以及 NUMA worker 是否读取本地权重页；
4. 针对单/少 token expert 合并 task，减少 per-expert dispatch；
5. 固定 shape、rolling pinned buffer 和预打包权重；
6. 比较 1/2 threadpool、每 NUMA 线程数、绑核和 first-touch。

output copy 只有历史约 0.36 ms/verify，不应先花精力做 copy 微优化。

### 8.4 动态长度算法本身的下一阶段

当前阈值已把错误 K2 压到零，但使用静态 `td=97,tv=100`。后续仍基于现框架时，可按
以下顺序扩展：

1. 用实际 active14 的 rolling draft/verify EMA 替代静态绝对成本，但保留 0.97 作为
   conservative prior；
2. 模型输入加入当前 KV length、verify graph bucket、CPU route/active cache 摘要；
3. 对 alpha 做按 draft position 和 workload 的 calibration，优化 expected accepted
   output，而非只拟合单 token label；
4. 决策直接比较 `E[TPOT(K1)]` 与 `E[TPOT(K2)]`，把 predictor 自身成本计入；
5. 用 holdout prompt 和 deterministic random stream 做离线 threshold sweep；
6. 在线仍 fail-safe 到 K1，并保留 full-context-safe preset。

由于本 workload K2 只占约 1.35%，在不改变 draft/verify cost 或 acceptance 的前提下，
这一方向预期是小收益。若希望“大幅”提升，必须引入更便宜且更准确的 draft 表达，或
多分支/tree speculative，而不是继续细调 0.97 的第二位小数。

### 8.5 GPU graph 内的中等收益组合

当前 draft per-op 中 attention 6.899 ms、GPU cached MoE 5.370 ms，reroute + metadata +
features 约 1.938 ms/call。建议拆成可回退的小提交：

1. 固定 qlen1/2/3 的 grouped GEMM configs，逐 shape 建 benchmark；
2. 融合 reroute、slot LUT、route sort/group offsets；
3. 合并 metadata/feature buffer 写入，减少 48 层小 kernel/copy；
4. 静态预分配 plan workspace，取消热路径清零/分配；
5. 再考虑 q/k norm + RoPE/split 融合。

每项都必须在 instrumentation-off 正常路径复测。op-event 自身将 TPOT 从正常水平放大
到 87.804 ms，不能用 profile-on 数字验收优化。

### 8.6 Prefetch/cache 调度

已知 budget2、segment16、rank1、LRU/profile-weighted 是当前离散筛选最优，但仍可改进
“选择质量”和“提交成本”：

- 缓存每层/segment 的候选排序，只在 route distribution 显著改变时更新；
- metadata descriptor 使用预分配 ring，只提交 index/range；
- 多 expert H2D 合批，减少 event/stream 管理；
- 以 verify CPU tail 为目标保护热专家，而不是只最大化 draft hit；
- 将请求长度纳入 cache/KV 内存分配，自动选择 active12/14；
- 记录 consumed/late/timeout 与 saved CPU routes 的因果比率，避免“提交更多就是更好”。

### 8.7 大改算法与架构

如果局部 kernel 优化进入平台期，真正可能跨越数量级的路线是：

- 训练或抽取更小 draft model，使 K2+ 的 acceptance 提高且 draft cost 降低；
- early-exit/self-spec，共享主模型前层或隐藏状态；
- EAGLE/Medusa 类多 token head；
- tree speculative，一次 verify 多分支候选；
- CPU expert 量化/稀疏化；
- 把 heterogeneous MoE 功能逐阶段迁入 KTransformers。

这些都必须保持 exact/standard sampling 语义或明确标注近似模式，且开发成本显著高于
当前配置/算子级优化。

## 9. 下一阶段建议执行顺序

1. active14 fixed K1 三 seed，隔离动态策略净收益。
2. current-path native CPUInfer qlen2/3 profile；并行做 affinity/NUMA 低成本 sweep。
3. 评估 GPU cached expert FP8/INT8 的容量、误差与 kernel 可行性。
4. 做 verify qlen2/3 kernel 和 reroute/plan 融合，每个正收益单独提交并补文档。
5. 再优化 predictor feature/metadata；不要再次融合会改变 sampler RNG 次序的 tail。
6. 若以上进入平台期，再决定训练型 draft 或 tree speculative 大改。

每个候选继续遵守当前验收规则：先短筛，正收益才进入至少三个独立进程 512-token
复测；提交中同时包含代码、配置/测试、命令、TPOT 前后、正确性和结果路径。负收益只
写入分析，不进入运行时代码提交。

## 10. 最终复现入口

workload-sized 当前最优：

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
  /home/edge/.conda/envs/nano_moe/bin/python \
  scripts/bench_eval_workload_tpot.py \
  --request-mode per_layer_slots \
  --model-path /home/edge/models/Qwen3-30B-A3B \
  --output-dir results/final_dynamic_active14_512 \
  --optimized-config k2_dynamic_f16_3080_active14 \
  --output-lens 512 --temperature 0.6 \
  --kt-llamafile-extension-path \
    /home/edge/zx/ktransformers/build/lib.linux-x86_64-cpython-312/cpuinfer_ext.cpython-312-x86_64-linux-gnu.so \
  --kt-single-weight true --collect-profile true \
  --save-profile-json true --save-token-ids true --fail-fast true
```

若请求总长度可能超过 1536 tokens，必须改用：

```text
--optimized-config k2_dynamic_f16_3080
```

最终结果目录与逐轮 JSON：

- `results/dynamic_k2_active14_threshold097_gmu098_512_r{0,1,2}/`
- `results/dynamic_tpot_k2_threshold097_gmu097_512_r{0,1,2}/`
- `results/active12_ctx8192_fixedgemm_512_repeats3{,_r2}/`

