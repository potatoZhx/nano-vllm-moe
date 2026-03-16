# 0
## 0.1 d
[prompt]
请仔细阅读demo_analysis.md和migration_design.md文档理解设计意图和实现的架构细节等；phase1_cpu_gpu_basic_heterogeneous_report.md中是Phase 1：基础异构推理的实现情况；现在请你思考Phase 2：投机解码的实现方案，包含实现目标、修改的具体内容、需要新增或修改的接口及其功能与设计、测试方案等，越详细越好；现在，如果有任何疑问，请先和我交流
[response]
我会先把这三份文档和仓库记忆做一次对齐阅读，提取 Phase 2 真正需要承接的约束与边界，再给你一个可落地的详细实现方案，并把我当前存在的不确定点单独列出来和你确认。

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/docs/demo_analysis.md#1-1)，行 1 到 260

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/docs/migration_design.md#1-1)，行 1 到 320

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/docs/summary/phase1_cpu_gpu_basic_heterogeneous_report.md#1-1)，行 1 到 260

读取内存 nano-vllm-moe-facts.md

我已经对齐了三份设计文档和仓库记忆，下一步会快速扫一遍当前代码里和 Phase 2 直接相关的现状（是否已有 speculative 框架、Sequence/BlockManager 可扩展点、placement 的 draft 分支），然后给你一个“可直接开工”的详细方案与待确认问题清单。

搜索匹配 `**/nano-vllm-moe/nanovllm/**/speculative/*.py` 的文件，无匹配项

搜索正则表达式 `enable_speculative|speculative|draft|verify|accept_draft|start_draft|rollback_draft|run_draft|run_verify` (`**/nano-vllm-moe/nanovllm/**/*.py`)，无结果

搜索正则表达式 `build_draft_plan|MoEExecutionPlan|substitution_map|draft_scheduler` (`**/nano-vllm-moe/nanovllm/**/*.py`)，3 个结果

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/nanovllm)

搜索正则表达式 `enable_speculative|speculative|draft|verify|accept_draft|start_draft|rollback_draft|run_draft|run_verify` (`**/nanovllm/**/*.py`)，无结果

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/nanovllm/engine)

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/nanovllm/scheduling)

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/nanovllm/engine/speculative)

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/nanovllm/engine/llm_engine.py)

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/nanovllm/engine/model_runner.py)

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/nanovllm/engine/sequence.py)

我确认到代码里已经有 Phase 1 的异构主路径，但投机解码相关代码几乎还没接入主循环；我再读取调度与块管理代码，明确 Phase 2 的最小侵入改造点后给你完整方案。

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/nanovllm/config.py)

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/nanovllm/expert/placement.py)

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/nanovllm/layers/fuse_moe/heterogeneous.py)

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/nanovllm/engine/block_manager.py)

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/nanovllm/engine/scheduler.py)

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/examples)

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/benchmarks)

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/nanovllm/expert/cache.py)

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/nanovllm/models/qwen3_moe.py#1-1)，行 1 到 420

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/nanovllm/models/qwen3_moe.py#420-420)，行 420 到 505

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/nanovllm/utils/heterogeneous_loader.py)

我已经完成现状核对：Phase 1 异构链路已接通，但主引擎还没有 speculative 分支，speculative 目录目前也没有实现文件。接下来我会给出一个按“先全 GPU speculative、再接异构 draft/verify”的 Phase 2 详细落地蓝图，并把需要你先拍板的关键问题单独列出来。你这个问题提得非常好，我先按你的要求“先交流疑问”，再给出我基于当前代码状态的 Phase 2 详细方案草案，方便你直接拍板。

我已完成文档与代码对齐阅读，关键现状是：

1. Phase 1 已经具备异构 MoE 主路径，入口与执行链路在 nanovllm/engine/model_runner.py、nanovllm/models/qwen3_moe.py、nanovllm/layers/fuse_moe/heterogeneous.py。
2. 当前主循环还没有 speculative 分支，nanovllm/engine/llm_engine.py 的 step 仍是单一 run。
3. speculative 目录已存在但尚无实现文件，nanovllm/engine/speculative。
4. Sequence 和 BlockManager 目前没有 draft 生命周期接口，nanovllm/engine/sequence.py、nanovllm/engine/block_manager.py。
5. placement 目前只有基础执行计划，还没有 draft substitution 语义，nanovllm/expert/placement.py。

**先确认的疑问**
1. Phase 2 你希望一次到位“异构 speculative 全量能力”，还是按设计文档建议先做全 GPU speculative 再接异构 draft？
2. 验收优先级是“先正确性和稳定性”还是“先吞吐提升”？这会影响 verify skip 和异步传输默认是否开启。
3. 采样模式是否要在 Phase 2 同时完整支持 greedy 与 sampling 两条接受策略，还是先只做 greedy 保证闭环？
4. 你是否同意 Phase 2 默认禁用 CUDA Graph（仅在 speculative 路径禁用），保留标准路径 graph 不变？
5. 你希望先支持单请求 speculative，还是直接支持连续批处理下的多序列 speculative 调度？
6. draft 期间预取触发点你更倾向哪一种先落地：按 step 触发（实现简单）还是按 layer 触发（更高潜力）？
7. 是否接受 Phase 2 默认使用 SimpleDraftScheduler 和 LRU，Adaptive 策略放到 Phase 3？
8. 你希望 Phase 2 的 benchmark 口径严格沿用迁移文档中的 C1/C2/C3，还是先跑最小子集做快速迭代？

下面是我建议的 Phase 2 方案草案，你确认上面问题后我可以据此收敛成最终实施清单。

---

**Phase 2 实现目标**
1. 在现有引擎内新增 draft-verify-accept 循环，不破坏标准 decode 路径。
2. 完成 KV cache 的 draft 回滚与 verify 替换闭环。
3. 在 MoE 层支持 draft placement 的三分流：GPU 命中、CPU top-c、GPU 替代。
4. 支持 verify skip 的严格条件优化。
5. 提供可插拔接受策略与 draft 调度策略接口，保证后续 Phase 3 可扩展。
6. 给出完整回归测试与性能基准框架，保证每次改动可量化。

---

**具体改动清单**

1. 引擎主循环接入 speculative
- 修改 nanovllm/engine/llm_engine.py
- 新增配置判断：
  1. prefill 阶段走现有 run
  2. decode 阶段若 enable_speculative 为真，走 spec_engine.speculative_step
  3. 否则仍走 run
- 设计要点：
  1. 保持 Scheduler 现有 schedule 和 postprocess 契约不破坏
  2. speculative_step 返回 token 列表与当前 postprocess 兼容

2. 新增 speculative 核心模块
- 新增 nanovllm/engine/speculative/spec_engine.py
- 新增 nanovllm/engine/speculative/acceptance.py
- 可选新增 nanovllm/engine/speculative/spec_scheduler.py
- 核心接口：
  1. speculative_step(seqs) -> accepted token ids
  2. draft(seqs) -> draft_tokens_per_seq, draft_activations
  3. verify(seqs, draft_tokens) -> verify_logits
  4. accept(seqs, draft_tokens, verify_logits) -> num_accepted per seq
- 设计要点：
  1. verify 输入拼接规则与文档一致：last_accepted + draft_tokens
  2. accept 对 greedy 与 sampling 分支清晰分离
  3. verify skip 只在严格条件满足时触发

3. Sequence 增强 draft 生命周期
- 修改 nanovllm/engine/sequence.py
- 新增字段：
  1. is_drafting
  2. draft_token_ids
  3. draft_start_num_tokens
- 新增方法：
  1. start_draft
  2. append_draft_token
  3. rollback_draft_view
  4. accept_draft
- 设计要点：
  1. token_ids 作为权威序列，draft 期间可先追加
  2. accept 后裁剪未接受的 draft 尾部
  3. 序列状态机与 FINISHED 状态不冲突

4. BlockManager 支持 draft rollback/accept
- 修改 nanovllm/engine/block_manager.py
- 新增接口：
  1. start_draft(seq)
  2. append_draft_token(seq)
  3. rollback_draft(seq)
  4. accept_draft(seq, num_accepted)
- 功能语义：
  1. start_draft 记录 block_table 与 token 计数快照
  2. rollback_draft 回退到 draft 前
  3. verify 按 prefill-like 重新建立对应 KV
  4. accept_draft 保留已接受段并释放冗余 blocks
- 风险点：
  1. 引用计数 ref_count 要严格对称
  2. preemption 与 draft 状态并发要加保护

5. ModelRunner 增加 draft/verify 执行接口
- 修改 nanovllm/engine/model_runner.py
- 新增方法：
  1. run_draft
  2. run_verify
  3. run_standard 保留现有 run 语义
- 设计要点：
  1. draft 执行时需要返回激活统计供预取调度
  2. verify 强制走无替换 placement
  3. speculative 路径默认 eager，避免 graph capture 冲突

6. Placement 扩展为 prefill 与 draft 双计划
- 修改 nanovllm/expert/placement.py
- 新增数据结构：
  1. MoEExecutionPlan 增加 substitution_map、selected_experts、routing_weights、cpu_expert_indices
- 新增接口：
  1. build_prefill_plan
  2. build_draft_plan
- build_draft_plan 逻辑：
  1. 命中 cache 的专家 -> GPU
  2. 缺失专家中选 top-c -> CPU
  3. 剩余缺失专家 -> substitute 到已缓存专家
  4. 构建 slot 级 m_sizes 与 route 索引

7. MoE 执行内核接入 substitution 与 CPU/GPU 并行
- 修改 nanovllm/layers/fuse_moe/heterogeneous.py
- 新增能力：
  1. draft 路径先对 selected_experts 做 substitution remap
  2. CPU path 支持 ThreadPool 并行专家计算
  3. GPU path 与 CPU path 结果统一 scatter 聚合
- 设计要点：
  1. 尽量保持当前张量化路径，避免回退到 Python 循环热点
  2. 先保证正确性，再做融合优化

8. Draft 调度与缓存策略接口落地
- 新增 nanovllm/scheduling/draft_scheduler.py
- 新增 nanovllm/scheduling/cache_strategy.py
- 新增 nanovllm/scheduling/prefetch_strategy.py
- 接口职责：
  1. select_cpu_experts
  2. select_gpu_substitutes
  3. select_experts_to_transfer
- Phase 2 建议默认实现：
  1. SimpleDraftScheduler
  2. LRUCacheStrategy
  3. SimplePrefetchStrategy

9. Config 扩展
- 修改 nanovllm/config.py
- 新增字段：
  1. enable_speculative
  2. max_draft_tokens
  3. draft_top_c
  4. acceptance_strategy
  5. acceptance_threshold
  6. prefetch_strategy
  7. cache_strategy
  8. draft_scheduler
- 设计要点：
  1. 兼容旧参数，默认关闭 speculative
  2. 参数分组清晰，便于命令行映射

---

**Phase 2 推荐实施顺序**
1. 2A 最小闭环：全 GPU speculative，不做替换与 CPU 路径，打通 draft-verify-accept 与 KV 生命周期。
2. 2B 接 draft placement：引入 top-c CPU 与 substitution_map，先同步传输。
3. 2C 加异步预取：加入 transfer stream 与 complete_transfers 轮询。
4. 2D verify skip：加严格条件门控与统计计数器。
5. 2E 策略插件化与参数化：把 simple 实现挂到接口工厂。

---

**测试方案**

1. 单元测试
- 建议新增目录：tests/speculative
- 用例组：
  1. Sequence draft 生命周期状态迁移
  2. BlockManager rollback 与 accept 的 block/ref_count 正确性
  3. build_draft_plan 的分流正确性
  4. acceptance greedy 与 sampling 逻辑
  5. verify skip 触发条件

2. 集成测试
- 场景矩阵：
  1. baseline standard decode
  2. speculative 全 GPU
  3. speculative 异构 top-c=0
  4. speculative 异构 top-c>0
  5. speculative 异构 + substitution 禁用
  6. speculative 异构 + prefetch 禁用
- 关键断言：
  1. greedy 下与标准路径 token 精确一致
  2. sampling 下概率接受行为合法且无崩溃
  3. 长序列下无 KV 泄漏和 block 泄漏

3. 性能测试
- 复用你已有 benchmark 习惯，补齐 C1/C2/C3
- 记录指标：
  1. tokens per second
  2. TTFT, TPOT
  3. draft, verify, accept 分项耗时
  4. cache hit rate, evictions, prefetch late
  5. accept rate 与 avg accepted tokens
- 稳定性口径：
  1. warmup 至少 2 次
  2. 正式至少 5 次
  3. 抖动超 10% 自动复跑

---

**我建议你先拍板的决策**
1. 是否按 2A 到 2E 分阶段推进，而不是一次性全开。
2. Phase 2 是否先只保 greedy 完整正确性，再补 sampling acceptance。
3. draft 预取触发点先按 step 还是直接按 layer。
4. 默认是否启用 verify skip。

你确认这 4 点后，我可以下一步直接输出“逐文件改动设计稿”，包括每个文件新增函数签名、输入输出、状态变更时序图、以及首批测试用例清单。

## 0.1
[prompt]
请先仔细阅读demo_analysis.md和migration_design.md文档理解设计意图和实现的架构细节等；phase1_cpu_gpu_basic_heterogeneous_report.md中是Phase 1：基础异构推理的实现情况；现在请你思考Phase 2：投机解码，再次仔细阅读migration_design文档相关章节（比如4.6，4.7，4.8，4.9，4.10等;ps,4.10仅需要实现最简单的方案即可，会在之后的phase重新优化），设计实现方案，包含实现目标、修改的具体内容、需要新增或修改的接口及其功能与设计（包含简洁的伪代码）、测试方案等，越详细越好；设计实现优先考虑正确性；设计文档写入nano-vllm-moe/docs/phase2_design.md；现在，如果有任何疑问，请先和我交流
[response]
我会先完整读取你提到的三份文档，重点核对 migration_design 中 4.6 到 4.10 的细节，然后基于现有 Phase 1 状态输出一份偏正确性优先、可直接落地的 Phase 2 设计文档到目标路径。接下来我先收集原文上下文并整理约束清单。

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/docs/demo_analysis.md#1-1)，行 1 到 260

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/docs/migration_design.md#1-1)，行 1 到 320

读取内存 nano-vllm-moe-facts.md

我已经拿到核心背景和仓库事实了，接下来会继续读取 migration_design 的后半部分（尤其 4.6~4.10 与测试章节）以及 Phase 1 报告原文，确保新文档和当前实现状态严格对齐。

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/docs/migration_design.md#321-321)，行 321 到 760

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/docs/migration_design.md#761-761)，行 761 到 1260

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/docs/summary/phase1_cpu_gpu_basic_heterogeneous_report.md#1-1)，行 1 到 260

我已经拿到迁移文档的关键章节原文。下一步我会快速扫描 nano-vllm-moe 当前代码里是否已有 speculative 相关骨架，确保 Phase 2 设计里的接口名与实际代码结构一致、可直接对接。

搜索与 `**/nano-vllm-moe/nanovllm/engine/**` 匹配的文件，5 匹配项

搜索正则表达式 `speculative|draft|accept_draft|run_draft|run_verify|AcceptanceStrategy|start_draft|rollback_draft` (`**/nano-vllm-moe/nanovllm/**/*.py`)，无结果

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/nanovllm)

搜索正则表达式 `speculative|draft|accept_draft|run_draft|run_verify|AcceptanceStrategy|start_draft|rollback_draft` (`**/nanovllm/**/*.py`)，无结果

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/nanovllm/expert)

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/nanovllm/scheduling)

我发现当前仓库里还没有明显的 speculative 代码命名痕迹，下一步会直接读现有核心实现文件（llm_engine、model_runner、sequence、block_manager、placement 等），用真实代码结构来约束 Phase 2 设计接口。

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/nanovllm/engine/model_runner.py)

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/nanovllm/engine/sequence.py)

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/nanovllm/engine/llm_engine.py)

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/nanovllm/config.py)

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/nanovllm/engine/block_manager.py)

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/nanovllm/engine/scheduler.py)

我已经确认当前代码尚未接入 Phase 2 的 speculative 主链路。现在我会读取异构相关的现有实现（cache、placement、heterogeneous forward、qwen3_moe），让 Phase 2 文档中的接口定义完全贴合现状。

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/nanovllm/expert/cache.py)

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/nanovllm/expert/placement.py)

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/nanovllm/layers/fuse_moe/heterogeneous.py)

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/docs)

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/nanovllm/models/qwen3_moe.py#1-1)，行 1 到 460

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/nanovllm/models/qwen3_moe.py#461-461)，行 461 到 505

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/docs/phase2_design.md)

我已经完成架构对齐，确认当前代码只有 Phase 1 异构执行能力，Phase 2 仍是空白状态。接下来我会直接写入一份可落地的 Phase 2 设计文档，按“正确性优先”给出分文件改动、接口定义、伪代码和测试矩阵。

正在生成 [](file:///zx_data1/sparsity/nano-vllm-moe/docs/phase2_design.md) 中的修补程序(576 行)


```

```


读取 [](file:///zx_data1/sparsity/nano-vllm-moe/docs/phase2_design.md#1-1)，行 1 到 260

已完成阅读与设计，并将 Phase 2 方案写入：
[ phase2_design.md ](nano-vllm-moe/docs/phase2_design.md)

本次文档内容覆盖了你要求的重点，且按“正确性优先”展开，包含：

1. 实现目标与非目标  
2. 与当前 Phase 1 实现的差距分析  
3. Phase 2 总体架构（Draft-Verify-Accept）  
4. 分文件改造清单（新增与修改）  
5. 配置项设计与约束  
6. 关键接口与功能设计（Sequence、BlockManager、ModelRunner、SpeculativeEngine、DraftScheduler、AcceptanceStrategy、Placement），并附简洁伪代码  
7. 与 LLMEngine/Scheduler 的集成方式  
8. 正确性约束与状态一致性要求  
9. 测试方案（单元、集成、回归、指标）  
10. 分步落地计划、风险与规避  
11. 与 migration_design 4.6-4.10 的逐点映射  
12. Phase 2 完成判定标准

当前没有阻塞性疑问；如果你愿意，我下一步可以直接按这份文档开始 Phase 2 代码实现，并先从最关键的状态机与 KV 回滚原语落地。

Made changes.

## 0.2
#### [prompt]
为了对比效果（推理速度以及内容正确性等），需要保留standard（原始nano-vllm实现），heter（异构推理基础推理路径），spec（异构路径基础上实现的speculative）三种mode；为了对比纯路径实现带来的差异，当前所有的对比实验都采用S=N的全量expert在GPU上的配置，但heter和spec不可以为S=N做任何特殊优化；heter和spec下，若没有激活的expert落在cpu，默认跳过相关操作；请先写好单元测试，在实现的过程中每一步都运行测试确保正确性；实现基本路径后先写好benchmark测试，对比三种模式的速度和输出正确性（对齐standard）；在文档中添加优化部分作为本阶段的最后实现步骤（比如kv cache管理引入“Ghost Block / 软释放” 机制。Draft 回滚时，不真正 free_block，而是将其标记为当前 Sequence 的复用池。Verify 阶段直接覆盖写入这些现成的 Block，省去分配开销。）；batch draft时，先实现所有序列相同draft长度，在优化章节补充批次内序列使用不同draft停止长度/策略的设计；请先根据以上建议修改phase2_design；如有任何问题或目前设计有不合理的地方，请先提出

#### [response]
我会先在设计文档中增补你提到的关键约束与流程顺序，重点覆盖三种模式并存、S=N 对比口径、先单测后实现再基准、以及优化章节中的 Ghost Block 与可变 draft 长度设计。接下来直接修改文档并回读确认。

正在生成 [](file:///zx_data1/sparsity/nano-vllm-moe/docs/phase2_design.md) 中的修补程序(321 行)


```

```


读取 [](file:///zx_data1/sparsity/nano-vllm-moe/docs/phase2_design.md#1-1)，行 1 到 340

读取 [](file:///zx_data1/sparsity/nano-vllm-moe/docs/phase2_design.md#341-341)，行 341 到 725

我再做一次小修订，把文档里的编号重复和一处优先级冲突点写得更严谨，避免后续按文档执行时出现歧义。随后我会把需要你确认的两个设计问题列出来。

正在生成 [](file:///zx_data1/sparsity/nano-vllm-moe/docs/phase2_design.md) 中的修补程序(37 行)


```

```
已按你的建议完成文档更新，主要新增了这些关键点：

1. 明确三种模式并存与对比目标  
   - standard / heter / spec 三模式统一纳入设计与评测。  
2. 明确 S=N 对比口径与约束  
   - 当前对比统一使用 S=N。  
   - heter/spec 不允许为了 S=N 增加任何特判优化路径。  
3. 明确 CPU 路径空任务处理  
   - heter/spec 下若无 expert 落到 CPU，CPU 相关操作直接跳过。  
4. 调整实施顺序为测试优先  
   - 先写单元测试，再实现，每一步实现后立即跑测试。  
5. 增加三模式 benchmark 章节  
   - 明确速度对比与输出对齐 standard 的要求。  
6. 增加优化章节并作为最后步骤  
   - 补充 Ghost Block / 软释放方案。  
   - 补充 batch 内不同 draft 停止长度策略设计（先设计后续实现，主路径先统一 draft 长度）。

已更新文件：  
[ phase2_design.md ](nano-vllm-moe/docs/phase2_design.md)

需要你先确认的两点（避免后续实现歧义）：

1. 模式配置优先级  
   - 我在文档里定义为 inference_mode 优先，旧字段只做兼容校验与自动推导。  
   - 你是否同意这个优先级策略？

2. 输出对齐口径  
   - 目前文档写的是 greedy 下 heter/spec 都要 token 级对齐 standard。  
   - 你是否希望把 heter 在 greedy 下也要求严格逐 token 对齐（和 spec 同级），还是只要求 spec 对齐 standard、heter 只要求“无明显语义偏差”？

你确认后，我就按这版文档进入“先写测试”的实现阶段。

Made changes.

## 1
#### [prompt]
确认：1.同意 2.目前的nano-vllm-moe中sampling_params似乎不支持确定性推理（temperature=0.0），如果能实现完全确定性推理的话就要求token级对齐，否则只要求无明显语义偏差即可
另外，CPU 直接跳过暂不实现，推迟到优化的step中
以上没有其他疑问的话，请开始实现，注意nano-vllm-moe环境为moe_spec（conda run -n moe_spec/conda activate moe_spec）

#### [response]



# skills
- nano-vllm-moe环境为moe_spec（conda run -n moe_spec/conda activate moe_spec）
- 实现时为了优化推理速度，尽量考虑不使用Python循环的实现