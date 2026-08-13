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
| P0 | pinned packing ring + 合批 H2D/publish/LUT 更新 | 高；现有换入主机毛开销很大 | cache 一致性、NUMA 与并发中风险 | 先做逐 expert 生命周期计时，再按小提交验证 enqueue/publish 与 TPOT。 |
| P0 | CPUInfer qlen2/3 F16 native 分项计时与专用 kernel | 高 | C++/NUMA/kernel 中高风险 | queue/GEMM/activation/merge 分项 + 3 seed TPOT。 |
| P0 | 压缩 GPU cached expert 权重以同时增加 active experts 和 KV | 高 | 数值、kernel 与加载器高风险 | logits/acceptance 正确；active 数增加；完整 512/8192 容量分档。 |
| P0 | 根据请求长度自动选择 active12-safe / active14-fast 内存档 | 中高 | 配置与容量校验低风险 | 运行前证明 KV capacity >= prompt+max output+graph padding。 |
| P1 | 标准 speculative acceptance 的 GPU 化/融合 | 上限约 0.683 ms/token | RNG、top-k/top-p 与 residual sampling 语义中风险 | 分布/固定随机流测试；只回传最终 accept length/token。 |
| P1 | legacy llamafile 跳过冗余 expert-mask D2H | 低但置信度高 | 后端分支不变量低风险 | mask/route 单测，确认 legacy 构造器确实不读 host mask。 |
| P1 | verify qlen2/3 grouped GEMM 专用配置/融合 | 中 | Triton autotune 与回归中风险 | kernel microbench 后必须过端到端 3 seed。 |
| P1 | `reroute + LUT + grouped plan` 单 kernel 融合 | 中 | graph-safe CUDA/Triton 中风险 | per-op event 降低且 TPOT 正收益。 |
| P1 | metadata/feature 四次小 copy 打包、静态 workspace | 低到中 | predictor 输入兼容风险 | predictor 输入 bitwise/容差一致，正常 profile 无事件开销。 |
| P1 | prefetch ranking 缓存、ring descriptor、批量 transfer enqueue | 中 | cache 时序和 acceptance 风险 | late/timeout 不增，CPU route 和 TPOT 同时下降。 |
| P1 | 按“每 9 MiB 可减少的 CPU 尾部”重做逐层 slot/初始 placement | 中高 | 需要先补可信逐层 route telemetry | holdout prompt 上 CPU tail/cache miss/TPOT 同时改善。 |
| P1 | CPU affinity、NUMA 页归属、threadpool 1/2 sweep | 低到中 | 低代码风险，环境敏感 | 独立进程 5 seed，报告 p50/p95 和 NUMA 配置。 |
| P2 | 直接比较 `E[TPOT(K2)]` 与 `E[TPOT(K1)]` 的动态控制器 | 当前 workload 小，跨 workload 中 | 需要未执行 K2 的反事实 alpha2/cost | shadow trace + holdout；保留 0.97/K1 fail-safe。 |
| P2 | predictor 校准与按 workload/position 的 threshold | 低（当前 workload）/跨 workload 中 | 过拟合风险 | deterministic trace replay + 未见 prompt holdout。 |
| P2 | draft tail 只捕获纯特征/predictor且保持 sampler RNG 次序 | 低 | 极易改变采样轨迹 | 固定随机数输入下 token/概率一致后再测。 |
| P2 | q/k norm、RoPE、split 小 kernel 融合 | 低到中 | attention 正确性风险 | attention 单测、长上下文数值和 TPOT。 |
| P2 | graph route accumulation、CPU output add 与 workspace 融合 | 中 | 48 层 graph-safe kernel 中风险 | 去掉 zero/index_copy/reduce 节点，数值一致。 |
| P2 | production fast path：关闭 trace/profile/mode 遍历 | 低到中 | 可观测性回归低风险 | profile-on/off 功能一致，最终生产口径单独报告。 |
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

1. 恢复验证后的第一个控制实验仍是 active14 fixed K1 三 seed，用于隔离动态策略净收益。
2. 工程优化先量化并削减 prefetch 的 sharded/pageable enqueue 与逐 expert publish；优先
   评估短生命周期 pinned packing ring、批量 LUT/mask 更新和 source 消费价值。
3. 做 current-path native CPUInfer qlen2/3 分项 profile；再按证据优化 kernel、task、
   affinity/NUMA 和 threadpool。
4. 评估 GPU cached expert INT8/其他压缩格式的容量、误差与小 M kernel 可行性。
5. 再做 verify qlen2/3 GPU kernel、route accumulation、reroute/plan 和 acceptance 融合，
   每个正收益独立提交并补文档。
6. 收集停止点 shadow alpha2 后，把动态决策升级为 K2-vs-K1；不要只微调 0.97 小数位。
7. 若以上进入平台期，再决定训练型 draft 或 tree speculative 大改。

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

## 11. 第二次全局审计：范围与冻结边界

在 `8b82ce3` 保留动态最优、`dbe6b25` 完成第一版收尾文档后，按要求停止新增优化
验证。以下结论仅来自：

- 对现有 active14 三轮 JSON/step trace 的离线统计；
- 对 active12 的 0.90/0.95/0.97 已有 trace 的交叉统计；
- 对 speculative、acceptance、prefetch/cache、kt_direct、MoE、attention、graph、
  sampler、KV 分配和 benchmark 路径的静态代码审计；
- 模型 config 的确定性字节/FLOP 计算。

本次没有生成新的 TPOT 点，没有运行新的候选端到端验证，也没有把静态推断写成性能
收益。后文所有“上限”都区分为端到端硬上限、调用路径毛开销或理论容量，不可混用。

## 12. 当前 63.713 ms 的精确预算

active14 三轮 step trace 每轮各覆盖 511 个 speculative 输出；每次请求最开始的 1 个
token 不在该 trace 中。按每轮 trace 输出归一化：

| 阶段 | r0 | r1 | r2 | 三轮均值 | 占 63.713 ms |
|:---|---:|---:|---:|---:|---:|
| verify model call | 44.934 | 43.587 | 45.686 | **44.736 ms/token** | **70.21%** |
| draft total | 18.248 | 17.312 | 19.205 | **18.255 ms/token** | **28.65%** |
| standard acceptance | 0.708 | 0.661 | 0.680 | **0.683 ms/token** | **1.07%** |
| verify-prefetch 外层 call | 0.331 | 0.289 | 0.320 | 0.313 ms/token | 0.49% |

`verify-prefetch` 已位于 draft/verify 调用链内，不能再与前三项相加。前三项合计约
99.94%，说明后续排序应首先回答“降低 verify 还是 draft 内部的哪一项”，而不是继续
猜测 scheduler 外层存在大块未解释时间。

可安全使用的收益边界只有以下几类：

| 方向 | 当前证据允许的上限解释 |
|:---|:---|
| 删除整个 verify | 44.736 ms/token、70.21%的绝对硬上限；现实不可删除，只能优化其中一部分。 |
| 删除整个 draft | 18.255 ms/token、28.65%的绝对硬上限；删除后也失去 speculative 收益。 |
| acceptance 融合 | 0.683 ms/token、1.07%的直接硬上限。 |
| prefetch enqueue/publish | 只有调用路径毛时间，因异步重叠和嵌套计时，不能直接换算 TPOT。 |
| cache/量化/新算法 | 间接改变 CPU route、KV 或 round 数，没有现成单项硬上限，必须重测。 |

按实际选择的 K 分桶，已有 trace 给出：

| 实际 K | rounds | 输出数 | round TPOT | 平均 step | 平均 draft | 平均 verify | 平均 accept |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0（terminal） | 2 | 2 | 104.197 | 104.197 | 0.026 | 103.566 | 0.578 |
| 1 | 874 | 1498 | 63.829 | 109.400 | 31.312 | 76.897 | 1.174 |
| 2 | 12 | 33 | **54.646** | 150.276 | 51.518 | 97.038 | 1.700 |

少量被选中的 K2 round 是局部高价值样本，而不是纯开销；但这仍不能替代同布局 fixed
K1 的完整反事实，因为 predictor 成本发生在所有 K1 round。

## 13. 离线重分析得到的新结论

### 13.1 Prefetch 的瓶颈不只是 PCIe，而是 pageable NUMA 分片的提交与发布

active14 三轮每请求均值：

| 指标 | 每请求 | 每输出 token / 每 expert |
|:---|---:|---:|
| expert transfer submit/completed/published | 3617 / 3617 / 3617 | 7.064 expert/token |
| 传输字节 | 3617 x 9 MiB = 31.79 GiB | **63.58 MiB/token** |
| predictive / verify-segment / draft-segment submit | 1181 / 1776 / 660 | 32.7% / 49.1% / 18.2% |
| late / timeout | 0 / 0 | 所有已提交项最终完成并发布 |
| transfer enqueue 毛时间 | 10.101 s | 2.791 ms/expert |
| completion latency 毛时间 | 65.700 s | 18.164 ms/expert，包含排队和异步重叠 |
| draft segment submit visible time | 2.583 s | 5.045 ms/token |
| verify segment submit visible time | 5.671 s | 11.076 ms/token |
| publish call time | 12.385 s | 24.190 ms/token；3.425 ms/expert |
| 真正显式 prefetch wait | 0.121 s | 0.235 ms/token |

`visible time` 包含 `transfer enqueue`，而 enqueue、publish 又可能和 GPU/CPU 工作重叠，
因此不能把 5.045、11.076、24.190 相加后声称可以直接减少 40.3 ms TPOT。它们能证明
的是：主机线程在关键调用链花了大量时间发起/发布换入，而不是只在 PCIe event 上等待。

根因在当前 single-weight 设计中很具体：每个 expert 为 9 MiB；双 NUMA shard 的
`NumaShardedExpertTensor.copy_to()` 会把 gate、up、down 拆成约 6 次 H2D `copy_`。
主权重默认不整体 pin（61 GiB 全量 pin 的启动代价不可接受），而 publish 又为一个
expert 分别更新 Python dict/list、GPU `slot_to_expert_lut`、`expert_to_slot_lut` 和两个
mask。3617 次换入会放大 CUDA API、pageable staging 和微小设备赋值开销。

由此得到比“再调 prefetch budget”更直接的 P0 实现路线：

1. 保持 single-weight 的静态内存优势，只增加 2--8 个 expert 的短生命周期 pinned
   packing ring；由 NUMA 本地 CPU worker 把 6 个 shard 拼到连续 ring slot；
2. 每个 expert 用 1--2 次连续 H2D，或把多个 expert 合成一个 descriptor/batch；
3. publish 先在 host 事务表中批量完成一致性检查，再用一个 GPU kernel 更新所有 LUT/
   mask；避免每 expert 多次设备标量赋值；
4. 将 query-ready 扫描改成固定 ring head/tail 或 completion queue，避免反复遍历
   `inflight` dict；
5. 新增按 source 的“发布后命中/挽救 CPU route/随后被淘汰”计数。当前所有 transfer
   都成功不等于都值得传，尤其 predictive 与 verify source 尚无完整消费归因。

### 13.2 Standard acceptance 是确定的小热点，硬上限约 1.07%

`StandardSamplingAcceptance.accept()` 当前会为 target 和 draft logits 都构造完整
151936-vocab 概率张量，并在每个 draft token 上用 `.item()` 执行 GPU→CPU 判定；
`_sample_from_probs()` 还会分别同步 total、isfinite 和最终 multinomial token。当前
active14 profile 已直接量到 0.683 ms/token，所以即使完全删除该阶段，端到端硬上限也
只有 1.07%；现实收益必然更小。

合理实现不是改变 speculative sampling，而是：

- 用 fused GPU kernel 计算 draft token 的 `p/q`、accept bits 和首个 reject；
- 只将最终 accept length/next token 一次性回传；
- 对未拒绝路径避免 materialize 不需要的 residual row；拒绝时再生成
  `max(p-q, 0)` 并采样；
- top-k/top-p、temperature、residual distribution 和 RNG 消耗顺序必须分别测试。

LM head 每行是 `151936 x 2048`，约 0.622 GFLOP，F16/BF16 权重 593.5 MiB。长期可以
探索 fused logits filtering/sampling 或量化 LM head，但此前 tail graph 已改变 RNG/轨迹
并产生负收益，因此不能简单重做相同融合。

### 13.3 当前动态控制器判断的是 K1 对 K0，不是 K2 对 K1

`first_increase` 的静态模型为：

```text
T(K) = (K * td + tv) / (1 + alpha1 + alpha1*alpha2 + ...)
```

在 `td=97, tv=100, min_steps=1` 下，执行完 K1 后比较 `T1 > T0`，其代数条件恰好是
`alpha1 < 0.97`。如果不满足便继续执行 K2。因此 0.97 是“K1 相对无 draft 是否值得”
的阈值，却被用作“现在是否值得再付出 K2”的代理。原则上第二个问题应直接比较：

```text
E[TPOT(K2) | current state] < E[TPOT(K1) | current state]
```

它需要未来第二 token 的条件 acceptance `alpha2`、第二次 draft 的边际成本和 qlen3
相对 qlen2 的 verify 增量。用当前 K1/K2 平均成本粗算，若 `alpha1 ~= 0.98`，K2 要优于
K1 大约需要 `alpha2 >= 0.75`。这只是用不同样本均值推导的决策量级，不是新阈值；
恰好当前 12 个 K2 中第二 token 接受 9 次（75%），更说明决策处在边界，必须获得
同状态反事实后再改算法。

active14 的第一步 predictor 离线校准：

| 项目 | 数值 |
|:---|---:|
| 样本数 | 886 |
| 平均预测 alpha1 | 0.6215 |
| 实际第一 token acceptance | 0.7178 |
| Brier score | 0.1988 |
| predicted >= 0.95 / 0.97 / 0.98 | 34 / 12 / 6 |

模型总体偏保守。0.95--0.97 区间已有 22 个 K1 样本且第一 token 全接受，但它们因为
停止而没有第二 token 的 `alpha2` 与 qlen3 成本，不能据此下调门限。对所有既有 active12
0.90/0.95/0.97 与 active14/0.97 的 47 个 K2 混合统计为：

| predicted alpha1 | K2 数 | full / partial / zero | round TPOT |
|:---|---:|:---:|---:|
| [0.90, 0.925) | 6 | 3 / 3 / 0 | 64.166 ms |
| [0.925, 0.95) | 5 | 2 / 1 / 2 | 78.957 ms |
| [0.95, 0.97) | 5 | 3 / 2 / 0 | 65.064 ms |
| [0.97, 0.98) | 12 | 9 / 3 / 0 | 54.964 ms |
| [0.98, 1.001) | 19 | 15 / 4 / 0 | 54.942 ms |

它混合了 cache、seed 和运行条件，只支持“0.97 是现有数据中合理的保守局部点”，不
支持跨 workload 的全局阈值。下一代仍应基于现框架，但增加 shadow/counterfactual：

1. 在不改变线上 K1 输出的独立 shadow 请求或离线 replay 中补采停止点的 alpha2；
2. 分别学习 `D1`、边际 `D2`、`V2`、`V3` 和 acceptance/prefetch 成本；
3. 用 first-step feature、route/cache 摘要、KV length 和历史条件 acceptance 预测 alpha2；
4. 直接比较 K2/K1，低置信度或分布外样本 fail-safe 到现有 K1；
5. 0.97 preset 永久保留，任何新控制器都作为独立 preset/独立提交。

### 13.4 显存的精确交换率

由模型 config 可直接计算：

| 对象 | F16/BF16 大小 |
|:---|---:|
| 单 expert 的 gate + up + down | **9 MiB** |
| 48 层每层平均增加 1 个 active slot | **432 MiB** |
| active12（576 slots） | 5184 MiB |
| active14（672 slots） | 6048 MiB |
| 两档差值 | 864 MiB |
| KV 每 token（48 层，K/V，4 heads x 128） | 96 KiB |
| KV block（256 tokens） | **24 MiB** |

active14 的 6 blocks 是 144 MiB/1536 tokens；active12 的 38 blocks 是 912 MiB/9728
tokens。差值与 864 MiB expert cache 增量及 `gpu_memory_utilization` 0.97→0.98 基本吻合。

因此：

- 只把 KV 变成 1 byte 格式，在其他内存不变时 active14 约从 1536 增到 3072 tokens，
  仍不足 8192；
- 把 active14 GPU expert cache 从 2 byte 压到 1 byte，理论释放约 3024 MiB，可大幅
  增加 KV 或 active slots，但必须有 decode 小 M 友好的 dequant/GEMM；
- RTX 3080 上 INT8 比 FP8 更贴近原生硬件能力；格式选择应由 qlen1/2/3 kernel 与数值
  共同决定，不能只按字节数；
- graph pool、临时 route buffer、LM head 和 warmup peak 也占显存，任何理论 slot/block
  数都必须在 capture 完成后重新做显存 census。

## 14. 全调用链剩余优化点

### 14.1 Cache placement 与预取价值模型

当前 `compute_layer_demand_from_csv()` 会把所有 segment size 的
`unique_expert_count_mean` 等权平均；初始 expert ranking 只使用 draft reroute profile
的 `act_freq`，缺失时直接选 expert id 0..N。它没有直接优化当前 prompt、K1/K2、verify
CPU 尾部或每 9 MiB 的收益。

后续目标函数应改成每层每 slot 的 marginal value：

```text
saved exposed CPU tail + avoided transfer/publish cost - displaced-hot-expert cost
--------------------------------------------------------------------------
                              9 MiB
```

可采用“静态核心 + prompt-conditioned 尾部”：prefill 后根据该请求的 route posterior
调整少量 slot，而不是全量 cache 抖动。当前最新 async profile 中若干
`model_layer_i_active_routes_sum` 为 0、CPU routes 却非零，逐层比率不可信；在 placement
优化前必须先补低开销、无同步的逐层 active/CPU route telemetry。

### 14.2 CPUInfer 与 NUMA

verify 占 70.21%，历史深 profile 又显示 overlap 后的 `kt.cpuinfer_sync` 最大，因此
CPU expert 仍是结构性主线。需要的不是又一个总 `cpu_compute_ms`，而是 qlen2/3 的：

- queue wait、task create、expert grouping；
- gate/up GEMM、SiLU/mul、down GEMM、merge；
- 两个 NUMA pool 各自的本地/远端读和负载不均；
- 从 submit 到 GPU 真正 wait 的 exposed tail。

在此基础上再决定 task 合并、权重 packing、m-block、线程池和 affinity。当前 legacy
llamafile 构造的 `MOEConfig` 不接收 `gpu_expert_mask_cpu` 指针，但
`begin_forward_graph_verify()` 每层仍 `_refresh_gpu_expert_mask()`，产生 48 次 128-bool
D2H，并在 GPU 上先把 cached route 改为 -1。应在后端不变量测试后为 legacy 分支跳过
冗余 mask D2H；历史整个 `kt.cpu_prepare_copies` 约 0.966 ms/verify，所以它是低风险小
收益，不应取代 native CPU kernel 主线。

### 14.3 Verify/draft GPU graph 与 MoE

剩余候选按依赖关系排序：

1. qlen1/2/3 grouped GEMM 应分别固定/穷举，而不是只用 `M<=32` 一个 decode 档；真实 M
   还受 route grouping 影响，应按 layer/active-route 分布加权；
2. `forward_verify_kt_hybrid` 每层建立/清零 route buffer，随后 `index_copy + view/sum`，
   再加 CPU output；可融合 down weighting、scatter/reduce 和 CPU add，使用持久 workspace；
3. top-c substitution LUT、arange、route sort/group offsets 可合并为 graph-safe plan kernel；
4. q/k RMSNorm 当前经历 float cast、square、mean、rsqrt、cast、weight 多个 op；RoPE 又
   chunk/cat/cast。自定义 Triton kernel 可融合，但其收益必须由 attention per-op 上限先
   证明；
5. verify bucket 2/3 共用 graph pool，但 K2 仅 1.35% rounds。先记录每次 capture 后的
   allocated/reserved 增量，再评估 lazy qlen3 graph、共享 workspace 或 graph 压缩；不可
   未测便让 qlen3 落回 eager。

### 14.4 Runtime、profile 与 Python 控制面

当前最终 benchmark 开启 `collect_profile=true`。静态审计发现：

- 每个 speculative step 都构建完整 dict/list step trace；
- `run_verify()` 即使走结构上已固定的 kt-hybrid graph，也遍历 48 个 MoE 层设置
  verify/normal mode；
- `get_and_reset_heterogeneous_profile()` 在 verify 路径未完全由 `profile_enabled` 守卫，
  会遍历 48 层；
- graph 内还维护 CPU/active route count 等主要用于 profile 的小 kernel。
- benchmark metrics 中仍有 provisional steady-draft gate `21 ms` 的 TODO；它应参数化并
  只用于分析筛选，不能成为 runtime 优化正确性的隐含条件。

应提供功能完全相同的 production fast path：profile/trace 关闭时不创建这些对象、
不遍历层、不执行纯观测 kernel；profile-on 仍保留完整证据。它只能在未来单独报告
instrumentation-off 的生产 TPOT，不能拿来悄悄替换本文 profile-on 基线。

### 14.5 Attention、dense、sampling 和调度

完整审计未发现一个可绕过的单一大项，但有以下组合机会：

- fused QKV split + q/k norm + RoPE，减少 48 层短 qlen 的 memory pass/graph nodes；
- KV store 与 attention 的接口融合，或只为 qlen1/2/3 做专用 decode kernel；
- LM head INT8/weight-only、分块 logits 与 fused sampling，减少 593.5 MiB 权重读取；
- standard acceptance 与普通 sampler 统一随机数生成接口，避免多套 full-vocab 分布；
- 单请求时复用 input/position/context pinned buffers，减少每轮 Python list→tensor；
- KV block 从 256 变小主要改善并发/碎片，不会减少单请求 579-token 的实际 KV 字节，
  优先级低于 expert cache 压缩。

每项都必须先用无扰动 event/graph node 计数给出上限。当前 verify/draft 大头在 MoE 与
prefetch/CPU 路径，不能因为这些融合容易编码就错误置顶。

### 14.6 Workload、并发与验收口径

本文最优只覆盖单请求、约 67-token prompt、512 输出、temperature 0.6。下列维度尚未被
当前最优证明覆盖，后续不能用单点默认值代替：

- prompt 短/中/长与接近 1536/8192 容量边界；
- 64/256/512/1024 输出长度，以及 EOS-on 的真实长度分布；
- greedy、temperature、top-k/top-p；
- batch/continuous batching 和多租户并发；
- 重复 prompt/prefix cache、不同语言和代码/数学 route 分布；
- TPOT p50/p95/p99、TTFT、吞吐和每 token 能耗，而不只是 mean TPOT。

并发下的最优很可能不同：更大的 grouped GEMM M 能提高 GPU 利用率，CPUInfer task 也
更易合批，但 KV 和 graph workspace 会挤压 expert slots；prefetch source 还可能跨请求
互相驱逐。应建立 workload matrix 后自动选择 memory preset、Kmax、graph buckets 和
prefetch budget，而不是把 active14 设为所有请求的无条件默认。

正确性分三层验收：单元级 tensor/logits 与 cache 不变量；固定随机流的 speculative
distribution/RNG 行为；holdout prompt 的质量与输出长度。exact sampling 优化不能只用
一次 token digest 相同作为统计正确性的充分条件，也不能只看分布测试而忽略工程回归。

### 14.7 系统与硬件层

低代码成本但环境敏感的变量应独立于算法提交管理：

- CPUInfer worker/OMP 线程数、物理核绑定、双 NUMA first-touch、内存交错/本地分配；
- CPU governor、turbo、GPU application clocks、温度和功耗稳定性；
- transparent huge pages/显式 huge pages 对 61 GiB 权重 TLB 的影响；
- pageable、`cudaHostRegister` 局部注册与小型 pinned ring 的启动/稳态交换；
- CUDA/PyTorch/Triton/FlashAttention/编译器版本，以及 graph capture 后的显存碎片；
- PCIe 链路宽度/速率、IOMMU 与跨 NUMA GPU 归属。

这些配置可能带来可观收益，但可移植性低。文档必须记录完整硬件拓扑、版本、affinity、
clock/thermal 状态，并在独立进程多 seed 下报告；环境调优不应混入 kernel 代码提交。

### 14.8 算法级上限

如果 P0/P1 降低了每轮固定成本，K2/K3 会自然变得更有价值；届时应重新训练/校准现有
predictor，而不是沿用旧成本下的阈值。再往后依次是：

1. 更便宜的 self-spec/early-exit draft；
2. 多 token head（Medusa/EAGLE 类）；
3. tree speculative，一次 verify 多分支；
4. CPU expert INT8/稀疏化；
5. 将 Nano 的 heterogeneous MoE、spec、prefetch 和 graph 分阶段迁入 KTransformers。

这些路线可能获得数量级更大的收益，但 exact sampling、模型权重/训练、graph bucket 和
显存都要重做，不能与当前“调现有动态框架参数”混为一个提交。

## 15. 后续实验矩阵与提交边界

用户恢复优化验证后，建议严格按以下依赖执行；本轮未执行这些实验：

| 阶段 | 只回答的问题 | 最小实验 | 进入下一阶段条件 |
|:---|:---|:---|:---|
| A | dynamic 的净收益是多少 | active14 fixed K1，同 seed/布局 3 轮 | 得到 predictor+rare K2 的因果差值 |
| B | prefetch 主机毛时间中多少在关键路径 | enqueue/publish/no-op shadow、CUDA/CPU timeline | 证明可兑现上限并锁定 packing/publish 子项 |
| C | 哪个 transfer source 真正挽救 CPU route | source-tagged consume/evict telemetry | saved-tail/byte 为正，冷换入不过量 |
| D | 当前 CPU 尾部在哪里 | native qlen2/3 分项 + NUMA/affinity | 找到占 exposed tail 的首要 kernel/queue |
| E | 小 kernel/GPU graph 上限 | instrumentation-off node/event census | 单项上限足够覆盖开发与回归风险 |
| F | 新动态 K2-vs-K1 控制器 | shadow alpha2/cost + holdout replay | 期望收益稳定且 fail-safe 不差于 0.97 |
| G | 权重量化/新 speculative 算法 | correctness/quality 后再 TPOT | 独立 preset，不覆盖精确基线 |

提交规则继续保持：

- `single-weight` 已有独立提交；
- 每个正收益、可用、可回退的优化独立提交；
- 同一提交补齐设计、命令/环境、基线与候选 TPOT、correctness、结果路径和一句话总结；
- 纯 telemetry 若是后续优化的必要前置，可独立提交，但不得写成性能收益；
- 负收益/无结论只进入分析文档，不把实验代码残留在 runtime；
- 动态算法无论后续结果如何，都保留 `k2_dynamic_f16_3080` 与
  `k2_dynamic_f16_3080_active14` 两个当前最优配置。

## 16. 收尾判断

当前路线已经达到“明显超过留存 KTransformers 参考”的工程目标，并保留了 full-context
safe 与 workload-sized 两个动态最优；但没有足够证据宣称硬件全局最优。下一轮最可能
产生真实大收益的顺序，已由本次全局审计修正为：

1. **先消除 single-weight NUMA 分片换入的逐 expert 提交/发布毛开销；**
2. **再定位并加速 qlen2/3 CPUInfer exposed tail；**
3. **用压缩 expert cache 同时改善 active slots 与 KV 容量；**
4. **补 fixed K1 因果对照与 shadow alpha2，再把动态决策改为 K2-vs-K1；**
5. **最后处理 acceptance、graph 小 kernel、norm/RoPE 与 production profile fast path。**

其中第 1 项是第二次审计新增的最高价值结论：显式 wait 很小不代表 prefetch 免费，当前
真正昂贵的是主机提交、pageable/sharded copy 和逐 expert publish；后续优化应围绕“每
次换入的可兑现 CPU route 尾部收益”，而不是围绕“提交数越多越好”。
