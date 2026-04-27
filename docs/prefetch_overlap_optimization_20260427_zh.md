# 预取重叠优化记录（扩展版，2026-04-27）

## 1. 背景、目标与阅读指引

本文总结了 `nano-vllm-moe` 中 Phase 3 推测预取路径近期的优化工作。该说明的原始简版记录了主要结论，但对不熟悉代码库的读者来说过于压缩。本扩展版会以足够细节解释系统与每一步优化，使新读者能够理解：

1. 推测预取流水线试图实现什么
2. 运行时开销最初来自哪里
3. 每一步优化改了什么
4. 哪些改动被接受并保留
5. 哪些改动虽然尝试过但因削弱正确性而被拒绝

我们希望验证的高层性能判断是：

1. 预取的运行时元数据导出不应阻塞 draft decode
2. 预取提交与执行应与后续 GPU draft 计算重叠
3. 当重叠有效时，预取成本应主要被隐藏，而不是在 decode 关键路径上可见

具体性能目标是：

1. 在 `S = N` 下，启用 Phase 3 运行 draft forward，并使其收敛到标准 CUDA graph decode
2. 在 `S != N` 下，以 `cache ratio = 75%` 和 `50%` 评估真实缓存压力
3. 识别重叠何时不再充分，以及预取成本何时变为可见

本轮全程遵守的工程规则非常严格：

1. 任何加速都必须以确定性行为仍成立为前提
2. 异步优化必须审查竞态条件、缺失同步、陈旧元数据读取和顺序变化
3. 直接 token 级一致性检查被视为最终正确性门禁

## 2. 范围与相关代码路径

本文中的实现与 profiling 工作主要涉及以下文件：

1. [nanovllm/engine/model_runner.py](/home/mumura/moe_spec/nano-vllm-moe/nanovllm/engine/model_runner.py:1)
2. [nanovllm/expert/runtime_meta.py](/home/mumura/moe_spec/nano-vllm-moe/nanovllm/expert/runtime_meta.py:1)
3. [nanovllm/expert/prefetcher.py](/home/mumura/moe_spec/nano-vllm-moe/nanovllm/expert/prefetcher.py:1)
4. [nanovllm/expert/cache.py](/home/mumura/moe_spec/nano-vllm-moe/nanovllm/expert/cache.py:1)
5. [nanovllm/engine/speculative/spec_engine.py](/home/mumura/moe_spec/nano-vllm-moe/nanovllm/engine/speculative/spec_engine.py:1)
6. [nanovllm/layers/layernorm.py](/home/mumura/moe_spec/nano-vllm-moe/nanovllm/layers/layernorm.py:1)
7. [examples/benchmarks/draft_standard_decode_forward_bench.py](/home/mumura/moe_spec/nano-vllm-moe/examples/benchmarks/draft_standard_decode_forward_bench.py:1)
8. [examples/heterogeneous_benchmark_case.py](/home/mumura/moe_spec/nano-vllm-moe/examples/heterogeneous_benchmark_case.py:1)

这些文件覆盖了系统的三个不同层面：

1. 模型执行与推测调度
2. 元数据导出、CPU 侧分析与预取排队
3. 基准测试、报告与确定性校验

## 3. 面向新读者的系统走读

### 3.1 标准 decode 路径

标准 decode 路径是最简单的基线。一个 decode step 执行一次 forward 并返回下一个 token。在代码库中，执行最终会流经 model runner，它会启动 decode kernel，并在启用时使用 CUDA graph replay 以获得稳定低开销执行。

概念上，标准 decode 如下：

1. 构建 decode 输入
2. 在 GPU 上运行一次 decode forward
3. 采样下一个 token
4. 更新 KV/cache 状态

这条路径是我们“最小控制面开销”的参考，因为它不需要推测 draft 记账、verify 同步或预取编排。

### 3.2 推测 decode 路径

推测 decode 会增加额外阶段。主推测循环围绕以下阶段组织：

1. `draft`
2. `rollback`
3. `verify`
4. `accept`

在实践中，draft 阶段会向前预测多个 token，verify 阶段用完整模型路径复核，随后引擎根据一致性接受或拒绝 draft token。

这意味着推测 decode 面临两类性能挑战：

1. draft 与 verify 的原始 GPU 计算量
2. 维持 expert cache 足够“热”的控制面工作，使后续 draft/verify 不因 expert 缺失而停顿

### 3.3 Phase 3 prefetch 在哪里发挥作用

Phase 3 prefetch 是控制面路径，目标是在所需 expert 变成延迟关键之前将其搬入 GPU cache。

高层流水线是：

1. 一个 draft/verify step 在 GPU 上产出路由元数据
2. 这些元数据被导出到 host
3. host 侧逻辑判断哪些 expert 已缓存、哪些值得预取
4. 候选被插入全局 warm-start 队列
5. 后台 prefetch 为有界数量的 expert 提交拷贝
6. 拷贝完成后，expert 从 staging 被“发布”到 active cache 状态
7. 后续 draft/verify 步骤即可使用这些 expert，而无需等待冷加载

关键细节是：元数据导出和预取调度并不是我们关心的实际张量计算，它们是支持性工作。因此如果它们落在关键路径上，就会变成纯开销。

### 3.4 为什么 `S = N` 与 `S != N` 都有价值

本文使用两种缓存设置，因为它们回答的是不同问题：

1. `S = N` 是控制路径对齐场景。它并非用于模拟真实缓存压力，而是用于回答：当所有 expert 实际上都可用时，推测预取机制本身引入了多少开销？
2. 在 `75%` 和 `50%` cache ratio 下的 `S != N` 是受压行为场景。它回答：当存在真实预取工作时，有多少成本仍被重叠隐藏，多少变成了可见延迟？

这一点很重要，因为某个机制在 `S = N` 下可能看起来非常好，但在真实传输与 publish 事件存在时仍可能成为瓶颈。

## 4. 实验环境与资源工作流

本轮使用了 `cluster-compute-workflow` skill，并且该 skill 也被更新，以便未来运行遵循同一资源选择规则。

更新后的流程是：

1. 若给定 `jobid`，先检查该 job 的节点
2. 若该节点仍有空闲可见 GPU，则复用该节点并绑定到真正空闲的可见设备
3. 若节点已满，则不再依赖该 job，申请新的单 GPU A100 job
4. 若未给定 `jobid`，自动申请新的 A100 job
5. partition、节点家族、GPU 类型、可见设备布局、conda 环境和 benchmark 参数尽量与此前 profiling 运行保持一致

skill 文件位于：

1. [/home/mumura/.codex/skills/cluster-compute-workflow/SKILL.md](/home/mumura/.codex/skills/cluster-compute-workflow/SKILL.md:1)

用于 publish-fast 调查的最新运行为：

1. `jobid=19597`
2. 节点 `gpu15-A100-E2-3U`
3. `CUDA_VISIBLE_DEVICES=7`
4. conda 环境 `nano_moe`

相关日志：

1. [job19597_publish_fastpath_pytest_20260427_151857.log](/home/mumura/moe_spec/logs/job19597_publish_fastpath_pytest_20260427_151857.log)
2. [job19597_publish_fastpath_batch_20260427_151930.log](/home/mumura/moe_spec/logs/job19597_publish_fastpath_batch_20260427_151930.log)
3. [job19597_publishfast_tokencheck_20260427_153620.log](/home/mumura/moe_spec/logs/job19597_publishfast_tokencheck_20260427_153620.log)

此前被接受的 host-buffer-pool 基线来自：

1. [job19053_hostpool_pytest_20260425_075448.log](/home/mumura/moe_spec/logs/job19053_hostpool_pytest_20260425_075448.log)
2. [job19053_hostpool_smoke_20260425_075521.log](/home/mumura/moe_spec/logs/job19053_hostpool_smoke_20260425_075521.log)
3. [job19053_hostpool_batch_20260425_080103.log](/home/mumura/moe_spec/logs/job19053_hostpool_batch_20260425_080103.log)
4. [job19053_cache75_tokencheck_20260425_081756.log](/home/mumura/moe_spec/logs/job19053_cache75_tokencheck_20260425_081756.log)

## 5. Benchmark 方法、Profiling 方法与正确性门禁

### 5.1 Benchmark 脚本与通用参数

主要报告脚本是：

1. [draft_standard_decode_forward_bench.py](/home/mumura/moe_spec/nano-vllm-moe/examples/benchmarks/draft_standard_decode_forward_bench.py:1)

单 case 运行与 token 级验证也由以下脚本产出：

1. [heterogeneous_benchmark_case.py](/home/mumura/moe_spec/nano-vllm-moe/examples/heterogeneous_benchmark_case.py:1)

本文对比中使用的通用运行参数为：

1. `num_seqs = 1`
2. `input_len = 24`
3. `output_len = 12`
4. `max_num_batched_tokens = 512`
5. `max_num_seqs = 32`
6. `max_model_len = 512`
7. `max_draft_tokens = 4`
8. `draft_top_c = 0`
9. `temperature = 0.0`
10. `engine_profile = true`
11. `engine_profile_cuda_sync = true`
12. `spec_enable_prefetch = true`
13. `prefetch_verify_wait_ms = 1.0`
14. `prefetch_step_budget = 4`
15. `prefetch_max_inflight = 8`
16. `prefetch_staging_slots_per_layer = 2`

### 5.2 为什么有多种计时口径

本工作使用了多种计时视角，它们回答的是不同问题：

1. `draft_forward_ms` 回答端到端 draft 延迟
2. `run_model_decode_ms_per_call` 隔离 runner 内部 decode forward 路径
3. `metadata_collect_ms_per_call` 衡量导出并物化路由元数据的成本
4. `metadata_observe_ms_per_call` 衡量 host 侧分析与队列更新成本
5. `submit_after_ms_per_call` 衡量 collect 之后仍可能泄漏到关键路径上的提交工作
6. `publish_ms` 衡量将 staged expert 变为 active routing 可见的成本
7. `prefetch_wait_ms` 衡量 verify/draft 仍需等待 ready 工作变得可用的时长
8. `prefetch_async_hidden_ratio` 估计异步 worker 周转中有多少被有效 GPU 计算重叠隐藏

关键点在于：仅仅 `publish_ms` 更低或 `prefetch_wait_ms` 更低，并不保证端到端更好。优化可能改变时序，而时序变化会影响推测轨迹或正确性。

### 5.3 每次重要改动都使用的正确性检查

本系列中每个非平凡优化都按三层正确性标准评估：

1. 针对改动组件的定向 pytest
2. `temperature=0.0` 下的 benchmark 级确定性 digest 一致性
3. 使用 `--return-token-ids true` 的直接 token 级比较

直接 token 级比较是最强信号。benchmark mismatch 说明某些内容变了，但 token-level mismatch 说明实际生成序列已经发散。这就是 publish-fast 路径被拒绝的原因，尽管某些汇总计时看起来有希望。

## 6. 起点：主重叠优化前系统是什么样子

在已接受的重叠优化落地前，预取路径更像串行控制面扩展，而非真正重叠的流水线。

### 6.1 原始行为

最初一个 draft step 的粗略执行模式是：

1. 完成 draft GPU 计算
2. 从 GPU recorder buffer 导出元数据
3. 在 host 侧物化元数据
4. 同步分析这些元数据
5. 更新访问统计和预取候选
6. 再继续后续 draft/verify 工作

也就是说，draft step 本身虽然已经在做有价值的 GPU 计算，但元数据路径仍处理得过于同步且保守。

### 6.2 原始局限

原始路径存在几个问题：

1. 想启动下一轮 decode 工作的控制线程，还必须等待元数据导出与处理
2. 元数据物化包含了比必要更多的拷贝与转换
3. 单一 host 侧元数据槽造成连续 step 之间的人为串行化
4. profiling 本身不足以判断时间是“暴露”还是仅“延后”
5. cache 状态观测与队列更新成本仍足以主导 `S = N`

### 6.3 为什么这不够

这种设计与 Phase 3 的预期目的相矛盾。prefetch 的存在是为了帮助未来计算，所以它应主要运行在未来计算“下方”。如果下一步 draft 需要等待前一步元数据收集，说明 prefetch 在重叠这件事上已经失败。

起始痛点最清晰的测量是：

1. [draft_standard_decode_forward_sn_prefetch_opt2_20260424.json](/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/draft_standard_decode_forward_sn_prefetch_opt2_20260424.json)
2. `draft / standard = 1.825x`
3. `metadata_observe ~= 6.34 ms/call`

对于理想情况下在 `S = N` 近乎零可见成本的控制路径来说，这个开销过高。

## 7. 优化 0：通过修复 RMSNorm 重新编译来稳定 Profiling 面

这项优化本身不是 prefetch 优化，但它是获得可信测量的必要前提。

### 7.1 优化前

系统使用了一个编译后的 RMSNorm 路径，它同时见到了 decode 和 verify/prefill 输入。这些输入 rank 不同：

1. decode 常用 2D tensor
2. verify 与 prefill 使用 3D tensor

由于同一路径见到两种 rank 模式，`torch._dynamo` 会反复重建图或触发 `recompile_limit`。这带来两个问题：

1. 与我们要研究的 prefetch 机制无关的不稳定运行时开销
2. graph capture 行为更难预测，使 standard-vs-draft 对比噪声更大

### 7.2 优化后

[layernorm.py](/home/mumura/moe_spec/nano-vllm-moe/nanovllm/layers/layernorm.py:1) 中的修复将 2D 和 3D 行为拆分，不再依赖一个编译路径同时服务两者。意图很直接：

1. 保持 decode 执行 shape 稳定
2. 保持 verify/prefill 执行 shape 稳定
3. 阻止 rank 多态污染 benchmark

### 7.3 为什么它改善了系统

改动后：

1. `torch._dynamo recompile_limit` 警告消失
2. CUDA graph capture 再次稳定
3. 后续重叠测量受无关编译抖动干扰更少

这项改动应被视为“测量卫生”。没有它，后续优化数字会把“真实 prefetch 开销”与“shape 重新编译噪声”混在一起。

## 8. 优化 1：引入异步元数据 Worker

这是朝向真实重叠的第一步结构性改造。

### 8.1 优化前

在异步 worker 之前，元数据处理与主 draft 控制流耦合太紧。希望持续发射 decode 工作的主线程还要承担大量 CPU 侧杂务：

1. 完成从 GPU 导出元数据
2. 收集为 CPU 侧对象
3. 运行观测逻辑
4. 更新预取队列
5. 有时还要推送更多工作后才能返回计算路径

这意味着即使 GPU 还有后续 draft 工作，CPU 记账也会拖慢流水线。

### 8.2 旧设计的局限

旧设计有三个核心局限：

1. 元数据导出与元数据解释没有逻辑分离
2. 没有独立 worker 吸收 host 侧观测延迟
3. profiling 不能清晰量化 worker 周转有多少被 GPU 计算隐藏

结果就是很难区分“真实依赖”和“偶发串行化”。

### 8.3 优化后

[model_runner.py](/home/mumura/moe_spec/nano-vllm-moe/nanovllm/engine/model_runner.py:1) 中被接受的 async-worker 设计引入了：

1. 后台 metadata worker 线程
2. 从 draft 路径到 worker 的异步交接队列
3. outstanding 工作的显式跟踪
4. device-buffer reuse wait 与 host-buffer reuse wait 的分离记账
5. 面向重叠的 profile 指标，例如 `prefetch_async_hidden_ms`、`prefetch_async_hidden_ratio` 和 `prefetch_async_exposed_wait_ms`

新的预期流程变为：

1. draft 完成后将运行时元数据写入 recorder buffer
2. 发起元数据导出
3. draft 控制路径把条目交给 worker
4. 主路径继续推进未来 draft 计算
5. worker 收集元数据、观测访问模式、更新预取状态，并在需要时触发后续提交

### 8.4 为什么它改善了系统

改进来自把串行依赖改成生产者/消费者关系。主线程不再必须先完成元数据解释才能发起后续有价值工作。

这不会神奇地消除全部开销。worker 仍要做同样的逻辑工作。但它改变了这些工作的时间位置：

1. 优化前，元数据处理更常直接落在关键路径上
2. 优化后，其中大部分可在后续 step 的 GPU draft 计算进行时并行执行

### 8.5 正确性保护

由于引入并发，正确性依赖更严格的所有权与 drain 规则：

1. 在前一次导出尚未安全脱离之前，GPU recorder buffer 不能复用
2. 在 worker 条目仍引用时，host metadata buffer 不能复用
3. verify 必须在等待 prefetch ready 前显式 drain 异步元数据，否则 verify 可能消费陈旧队列状态

这些保护被编码在 buffer reuse 记账和 verify 侧 drain 路径中。

### 8.6 测得效果

第一版 async-worker 结果仍显示 host 开销偏高：

1. [draft_standard_decode_forward_sn_prefetch_async_final_20260425.json](/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/draft_standard_decode_forward_sn_prefetch_async_final_20260425.json)
2. `draft / standard = 1.787x`
3. `metadata_collect ~= 7.35 ms/call`
4. `prefetch_async_hidden_ratio ~= 0.916`

这在结构上是进展，但绝对成本仍不够理想。它证明了重叠计量有价值，同时也暴露出元数据物化本身仍过重。

## 9. 优化 2：分离 Device Buffer 生命周期与 Host Buffer 生命周期

该优化旨在消除伪依赖。

### 9.1 优化前

在更细粒度复用记账之前，系统对元数据导出生命周期处理得过于保守。实际存在两类不同资源：

1. GPU step 写入的 device 侧 recorder buffer
2. 异步 worker 后续读取的 host 侧 buffer 或对象

如果不分离这两者生命周期，系统就可能比必要等待更久。例如，后续 draft step 可能无法复用 recorder buffer，尽管数据已安全转移出去。

### 9.2 旧设计的局限

旧方法容易累积不必要等待，原因在于：

1. 复用所有权过于粗粒度
2. 代码并未总是区分 GPU 侧安全与 host 侧安全
3. 等待可能被归因到错误资源，从而导致错误优化方向

### 9.3 优化后

[model_runner.py](/home/mumura/moe_spec/nano-vllm-moe/nanovllm/engine/model_runner.py:1) 中被接受的改动将记账拆分为：

1. device-buffer reuse wait
2. host-buffer reuse wait

这听起来很小，但概念上很关键，意味着：

1. draft 路径只等待真正尚不安全复用的资源
2. profiling 可以指出真实瓶颈是 GPU recorder 压力还是 host metadata 压力
3. 后续优化可精准攻击正确瓶颈，而不是模糊聚合值

### 9.4 为什么它改善了系统

收益不仅是时间下降，更是可诊断性提升。等待拆分后，host 侧复用压力成为清晰重要来源，值得单独修复。这直接催生了下一节 host-buffer-pool 改动。

## 10. 优化 3：Host Metadata Buffer Pool

这是整个重叠优化系列中最重要的已接受优化。

### 10.1 优化前

最初元数据导出路径本质上像单车道 host staging 区。即使 GPU draft 计算与 worker 处理概念上已解耦，单一 host 元数据槽仍会产生背压：

1. step A 把元数据导出到 host 槽
2. worker 在处理期间仍占用该槽
3. step B 想导出新的元数据
4. step B 必须等待 step A 槽位释放

这形成了典型的生产者/消费者瓶颈。GPU 可能已准备前进，但生产者因只有一个 host staging 车道无法继续交接。

### 10.2 为什么这不够

这种设计使重叠变得脆弱：

1. 即使很短的 worker 延迟也可能阻塞下一次导出
2. 这类阻塞会显示为“metadata 开销”，尽管部分只是槽位争用
3. 延迟对 host 调度噪声更敏感

这正是 `S = N` 下最该避免的串行化，因为该场景本应几乎没有真实预取工作。

### 10.3 优化后

[runtime_meta.py](/home/mumura/moe_spec/nano-vllm-moe/nanovllm/expert/runtime_meta.py:1) 与 [model_runner.py](/home/mumura/moe_spec/nano-vllm-moe/nanovllm/engine/model_runner.py:1) 中被接受的 host-buffer-pool 设计做了三件事：

1. host metadata 存储从单槽改为小型池
2. 每个 offload handle 记录其持有的 host slot
3. host slot 的复用只阻塞仍在处理中那一个具体槽位

这意味着后续 draft step 不必等待全部 host 处理完成，只需从池中拿到另一个可用槽位。

### 10.4 为什么它改善了系统

这与图形和数据处理中的多缓冲流水线生效原理一致：

1. 一个 buffer 可以被填充
2. 另一个可以被处理
3. 第三个已经为下一步准备

生产者与消费者不再争同一内存槽，缓冲池吸收了 worker 完成时间的正常抖动。

### 10.5 正确性保护

由于池化复用在所有权不清时也会引入 bug，设计中通过 offload handle 显式存储槽位所有权：

1. handle 不能无声迁移到其他槽
2. 槽位在 worker 条目退役前不会回收
3. worker 与主线程按槽位 ID 达成一致，而非间接推断

这很重要，因为陈旧槽位复用会造成灾难且难调试：可能静默混入两个 decode step 的元数据。

### 10.6 测得效果

这项优化带来了 `S = N` 最大的已接受提升：

1. [draft_standard_decode_forward_sn_prefetch_hostpool_20260425.json](/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/draft_standard_decode_forward_sn_prefetch_hostpool_20260425.json)
2. `standard decode forward = 15.92 ms`
3. `draft forward = 20.59 ms`
4. `draft / standard = 1.293x`
5. `metadata_collect ~= 2.55 ms/call`
6. `metadata_observe ~= 0.50 ms/call`
7. `metadata_buffer_reuse_wait ~= 0.01 ms/call`
8. `prefetch_async_hidden_ratio ~= 0.896`

相比更早的 observation-heavy 已接受基线：

1. [draft_standard_decode_forward_sn_prefetch_obsopt2_20260424.json](/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/draft_standard_decode_forward_sn_prefetch_obsopt2_20260424.json)
2. `draft / standard = 1.502x`
3. `metadata_collect ~= 2.88 ms/call`
4. `metadata_observe ~= 1.53 ms/call`

这不是“稍微快一点”，而是瓶颈形态发生了变化。host metadata 复用不再是一阶问题。

## 11. 优化 4：减少不必要的 CPU 物化与转换

该优化聚焦让每个元数据条目处理更便宜。

### 11.1 优化前

在小聚合路径细化之前，元数据处理做了超过必要的工作：

1. 即使只需聚合信息，也会创建额外 host clone
2. 某些路径存在可避免的 CPU 转换
3. cache 观测逻辑对小元数据场景处理过于通用，常见 decode case 为灵活性付出了额外成本

也就是说，即便已有异步处理，每个工作项本身仍偏重。

### 11.2 为什么这不够

异步执行能隐藏延迟，但不能让工作项“免费”。若每个条目都做了不必要 clone 或转换，worker 仍会消耗 CPU 周期和内存带宽，并长时间占据 host 槽位，这会直接降低系统可达成的重叠程度。

### 11.3 优化后

[runtime_meta.py](/home/mumura/moe_spec/nano-vllm-moe/nanovllm/expert/runtime_meta.py:1)、[prefetcher.py](/home/mumura/moe_spec/nano-vllm-moe/nanovllm/expert/prefetcher.py:1) 与 [cache.py](/home/mumura/moe_spec/nano-vllm-moe/nanovllm/expert/cache.py:1) 中被接受的改动做了两件事：

1. 对小聚合场景，`collect()` 可在 host view 上操作，而不是急切 clone 全量 per-token 元数据载荷
2. 观测路径在信息已经以可用 CPU 形式存在时，避免不必要的 `.to(device="cpu")` 或等价 host 物化

### 11.4 为什么它改善了系统

这从两个维度降低了开销：

1. 每个元数据条目的数据移动更少
2. 每个元数据条目的 worker 占用时间更短

这很关键，因为 worker 路径不只是计算，还包含内存拷贝、数据布局转换和队列更新。缩小载荷并减少转换会同时加速这三部分。

### 11.5 正确性保护

该优化之所以安全，是因为它不改变元数据语义，只改变表示与拷贝方式：

1. 仍观测相同路由决策
2. 仍记录相同访问统计
3. 仍执行相同下游队列更新逻辑

因此关键正确性问题不是路由语义等价性，而是 host view 与 clone 的生命周期安全。这也是 buffer-pool 所有权规则仍然关键的原因。

## 12. 优化 5：利用 CPU Cache-State 查询降低 Observation 成本

该优化瞄准了早期主要可见成本 `metadata_observe`。

### 12.1 优化前

早期观测路径需要判断选中的 expert 是否已缓存。朴素做法通常需要搬运或检查比必要更广的缓存状态表示。

实践中，这意味着观测路径为回答一个简单问题付出了过高成本：

1. 该 expert 是否已经驻留可用？
2. 若已驻留，是否还需要将其纳入预取？

### 12.2 为什么这不够

当答案常常是“已缓存”时，系统应以低成本快速拒绝。如果代码先支付重状态物化成本，再得出“无需操作”，就等于为发现“没有工作”做了昂贵工作。

### 12.3 优化后

被接受的优化把 cache-state 检查转向更轻量的 CPU 侧查询，复用运行时已维护的 cache 元数据，而不是强制更广状态搬运。实践上意味着：

1. 使用轻量的 cache 驻留信息
2. 更早过滤已缓存 expert
3. 仅在存在真实预取工作时进入更重的队列更新路径

### 12.4 为什么它改善了系统

这让 `metadata_observe` 变成更具选择性的流水线：

1. 对已缓存 expert 做廉价拒绝
2. 仅对真正有价值的预取候选走昂贵路径

这正是 `S = N` 场景应有行为，因为该场景几乎没有有价值预取工作，观测路径应当很小。

### 12.5 测得效果

2026-04-24 的 observation-heavy 已接受基线是：

1. [draft_standard_decode_forward_sn_prefetch_obsopt2_20260424.json](/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/draft_standard_decode_forward_sn_prefetch_obsopt2_20260424.json)
2. `metadata_observe ~= 1.53 ms/call`

后来的 hostpool 基线是：

1. [draft_standard_decode_forward_sn_prefetch_hostpool_20260425.json](/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/draft_standard_decode_forward_sn_prefetch_hostpool_20260425.json)
2. `metadata_observe ~= 0.50 ms/call`

该下降并非单行代码带来，但这类 cache-aware 观测优化是后续 hostpool 版本能把 `observe` 压到不再主导可见项的重要原因之一。

## 13. 已接受优化的整体效果

到 hostpool 设计就位时，系统已发生了定性变化。

### 13.1 优化系列之前

主重叠优化之前：

1. 元数据处理过于串行
2. buffer 复用产生了人为等待
3. 观测逻辑每步做了过多工作
4. 缺少足够 instrumentation 区分隐藏时间与暴露时间

### 13.2 已接受优化之后

已接受改动之后：

1. 元数据导出与观测从主 draft 线程解耦
2. 池化 host buffer 消除了大部分 host 侧交接争用
3. 小元数据条目物化更便宜
4. 观测可更廉价过滤已缓存 expert
5. 重叠由显式指标衡量，而非间接推断

### 13.3 实际含义

在当前已接受实现下：

1. 元数据导出已不再是首要瓶颈
2. host-buffer 复用几乎完全被隐藏
3. 剩余可见延迟已下移到 `submit_after`、`publish`、verify 可见性以及 `route/plan/gpu_compute` 的固有成本

这是一个重要架构收益，因为它表明优化前沿已移动，团队不再需要猜时间去哪了。

## 14. Phase-3 基线与性能时间线

下方时间线包含多个中间状态测量。这些结果并非完全苹果对苹果，因为代码结构与 profiling 字段随时间演进，但它们仍展示了瓶颈如何迁移。

### 14.1 `S = N` 演进

| 阶段 | 结果文件 | Draft / Standard | 主要观察 |
| --- | --- | ---: | --- |
| Phase 3 default | `draft_standard_decode_forward_phase3_default_final2_20260424.json` | `1.184x` | 启用特性的早期 Phase 3 基线 |
| Prefetch opt2 | `draft_standard_decode_forward_sn_prefetch_opt2_20260424.json` | `1.825x` | 观测路径主导，`S = N` 明显过慢 |
| Observe opt2 | `draft_standard_decode_forward_sn_prefetch_obsopt2_20260424.json` | `1.502x` | 观测成本下降，但仍过于可见 |
| Async worker | `draft_standard_decode_forward_sn_prefetch_async_final_20260425.json` | `1.787x` | 结构改善，但 collect 仍过重 |
| Host buffer pool | `draft_standard_decode_forward_sn_prefetch_hostpool_20260425.json` | `1.293x` | 目前最佳已接受 `S = N` 结果 |
| Publish-fast attempt | `draft_standard_decode_forward_sn_publishfast_20260427.json` | `1.413x` | 因后续正确性失败未被接受 |

关键解读是：

1. 最持久的 `S = N` 改善来自元数据导出“低成本+异步化”
2. 单独降低 publish 成本并不会自动改善 `S = N`
3. 当 `observe` 与 host 复用已大多被隐藏后，下一个可见瓶颈会迁移

### 14.2 `S != N` 的已接受 hostpool 基线

| Cache ratio | 结果文件 | Draft / Standard | Hidden ratio | Host reuse wait | 主要暴露项 |
| --- | --- | ---: | ---: | ---: | --- |
| 75% | `draft_standard_decode_forward_cache75_hostpool_summary_20260425.json` | `1.569x` | `0.975` | `0.011 ms/call` | `submit_after`, `publish`, `prefetch_wait` |
| 50% | `draft_standard_decode_forward_cache50_hostpool_20260425.json` | `1.804x` | `0.994` | `0.008 ms/call` | `submit_after`, `publish`, `prefetch_wait` |

这些数字很关键，因为它们显示即便在真实缓存压力下，元数据导出重叠仍然良好。较低 cache ratio 下延迟仍上升，说明可见成本已迁移到更后段预取阶段，而非导出阶段。

## 15. 优化 6：2026-04-27 的 Publish-Fast 尝试

本节有意写得更细，因为它是本轮中“局部看起来合理、整体却因正确性被否决”的最佳案例。

### 15.1 为什么 `publish` 成为下一个目标

在 hostpool 改动后，`S != N` 的下一个可见成本已不再是元数据收集与 host 槽位争用，而是：

1. `submit_after`
2. `publish_ms`
3. `prefetch_wait_ms`

这一点在以下结果中很清楚：

1. [draft_standard_decode_forward_cache75_hostpool_summary_20260425.json](/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/draft_standard_decode_forward_cache75_hostpool_summary_20260425.json)
2. [draft_standard_decode_forward_cache50_hostpool_20260425.json](/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/draft_standard_decode_forward_cache50_hostpool_20260425.json)

直觉很直接：若元数据已基本被隐藏，剩余暴露时间很可能花在把“staging 拷贝完成”转化为“active cache 状态可见 expert”上。

### 15.2 优化前

在尝试 publish-fast 前，`publish_ready()` 采用保守路径，在评估 victim-selection 状态时依赖 cache snapshot。该路径的工程优势明确：

1. 更易推理
2. 将 publish 逻辑与 live 可变 cache 状态分离
3. 降低读取部分更新驻留元数据的风险

但其性能代价也明确：

1. 获取 snapshot 会引入拷贝或物化成本
2. publish 侧逻辑反复支付该成本
3. 当元数据导出本身已优化后，该成本更易暴露

### 15.3 为什么原设计不够

snapshot 路径安全但在真实缓存压力下不够便宜。在已接受 hostpool 基线中，前段开销已被压低，因此 snapshot 成本不再被其他成本遮蔽，成为合理下一个目标。

### 15.4 优化尝试后

该 fast path 尝试将目标从“基于 snapshot 计算 publish 决策”改为“直接基于 live cache 元数据计算，不做 snapshot clone”。涉及实现：

1. `nanovllm/scheduling/cache_strategy.py`
2. `nanovllm/expert/cache.py`
3. `nanovllm/expert/prefetcher.py`

预期机制是：

1. 直接从 cache 读取 victim-selection 元数据
2. 避免每次 publish 的 snapshot clone
3. 保持相同 eviction-policy 语义
4. 保持相同 staging-to-active publish 状态转换

### 15.5 为什么它看起来有希望

从纯微优化视角，这正是我们想要的变化：

1. 更少拷贝
2. 更低 publish 侧 CPU 工作量
3. 更低 publish 延迟
4. 更低 expert 可用前等待时间

首轮计时数字确实朝该方向移动。

### 15.6 在 cache ratio 75% 下的性能影响

已接受 hostpool 基线：

1. raw file: [draft_standard_decode_forward_cache75_hostpool_raw_20260425/repeat_00_spec.json](/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/draft_standard_decode_forward_cache75_hostpool_raw_20260425/repeat_00_spec.json)
2. `draft_forward_ms = 24.18`
3. `submit_after_total = 18.34 ms`
4. `publish_ms = 12.23 ms`
5. `prefetch_wait_ms = 4.48 ms`

尝试的 publish-fast 路径：

1. raw file: [draft_standard_decode_forward_cache75_publishfast_raw_20260427/repeat_00_spec.json](/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/draft_standard_decode_forward_cache75_publishfast_raw_20260427/repeat_00_spec.json)
2. summary file: [cache75_publishfast_summary_20260427.json](/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/cache75_publishfast_summary_20260427.json)
3. `draft_forward_ms = 24.76`
4. `submit_after_total = 20.35 ms`
5. `publish_ms = 10.21 ms`
6. `prefetch_wait_ms = 3.63 ms`

含义是：

1. publish 变便宜了
2. wait-for-prefetch 也变便宜了
3. 但总 draft 延迟没有改善

因此即便在正确性检查前，这已是一个警讯：该优化改变了系统行为，其复杂性超出局部微优化。

### 15.7 在 cache ratio 50% 下的性能影响

已接受 hostpool 基线：

1. raw file: [draft_standard_decode_forward_cache50_hostpool_raw_20260425/repeat_00_spec.json](/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/draft_standard_decode_forward_cache50_hostpool_raw_20260425/repeat_00_spec.json)
2. `draft_forward_ms = 26.92`
3. `submit_after_total = 48.72 ms`
4. `publish_ms = 15.64 ms`
5. `prefetch_wait_ms = 5.56 ms`

尝试的 publish-fast 路径：

1. raw file: [draft_standard_decode_forward_cache50_publishfast_raw_20260427/repeat_00_spec.json](/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/draft_standard_decode_forward_cache50_publishfast_raw_20260427/repeat_00_spec.json)
2. summary file: [cache50_publishfast_summary_20260427.json](/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/cache50_publishfast_summary_20260427.json)
3. `draft_forward_ms = 28.37`
4. `submit_after_total = 39.12 ms`
5. `publish_ms = 9.87 ms`
6. `prefetch_wait_ms = 3.46 ms`

同样，局部 publish 指标改善了，但端到端 draft 没有改善。这强烈暗示优化影响了时序敏感行为，而不只是削减了无效周期。

### 15.8 正确性验证与失败

尝试路径通过了定向单测：

1. cache-strategy tests
2. prefetch runtime tests
3. metadata recorder tests
4. benchmark reporting tests

Pytest 日志：

1. [job19597_publish_fastpath_pytest_20260427_151857.log](/home/mumura/moe_spec/logs/job19597_publish_fastpath_pytest_20260427_151857.log)

但确定性运行时验证暴露了问题：

1. benchmark 汇总在 `cache ratio 75%` 报告了确定性 mismatch
2. 一个 `cache ratio 50%` benchmark batch 也出现 mismatch
3. `cache ratio 75%` 的直接 token 级复跑确认了真实发散

相关文件：

1. [cache75_standard_publishfast_tokencheck_20260427.json](/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/cache75_standard_publishfast_tokencheck_20260427.json)
2. [cache75_spec_publishfast_tokencheck_20260427.json](/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/cache75_spec_publishfast_tokencheck_20260427.json)
3. [cache50_standard_publishfast_tokencheck_20260427.json](/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/cache50_standard_publishfast_tokencheck_20260427.json)
4. [cache50_spec_publishfast_tokencheck_20260427.json](/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/cache50_spec_publishfast_tokencheck_20260427.json)

决定性证据是 `cache75` token 发散：

1. standard tokens: `4710, 16141, 1447, 32313, 11, 773, 358, 1184, 311, 7071, 700, 3170`
2. spec tokens: `576, 4226, 1265, 387, 304, 6364, 11, 323, 279, 2790, 3084, 1265`

这不是小的统计波动，而是生成轨迹发生了变化。

### 15.9 为什么优化被拒绝

publish-fast 尝试被拒绝并回退，原因是：

1. 在 `cache ratio 75%` 下 token 级正确性失败
2. 失败恰好发生在时序与可见顺序敏感的系统区域
3. 局部计时改进不足以支撑语义不确定性风险

### 15.10 最可能解释

精确 bug 机制仍需更深调查，但问题类别很清楚：

1. snapshot 路径可能隐含了比 live-cache 路径更强的一致性边界
2. 去掉该边界可能改变 expert 对后续 routing 或 verify 逻辑变为可见的时机
3. 更快的 host 路径可能改变了“权重拷贝完成、元数据提交、后续消费者读取”之间的交错

这就是为什么 publish 优化必须被当作正确性敏感的调度变更，而非廉价局部清理。

## 16. 系统当前状态

### 16.1 已经工作良好的部分

在 hostpool 基线对应的已接受实现下：

1. 元数据导出大体已与后续 draft 计算重叠
2. host-buffer 复用不再是有意义的可见瓶颈
3. `S = N` 相比早期版本明显更接近标准 decode
4. `75%` 与 `50%` cache-ratio 运行中，异步工作仍有较高 hidden ratio

最重要的已接受结果是：

1. [draft_standard_decode_forward_sn_prefetch_hostpool_20260425.json](/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/draft_standard_decode_forward_sn_prefetch_hostpool_20260425.json)
2. `draft / standard = 1.293x`

### 16.2 仍然暴露的部分

当前主要可见项是：

1. `route`
2. `plan`
3. `submit_after`
4. `publish_ms`
5. `prefetch_wait_ms`
6. 较低 cache ratio 下的 verify 侧 readiness 成本

换言之，剩余工作已不再是“如何便宜导出元数据”，而是“如何在不破坏语义的前提下，让预取完成在正确时机可见”。

### 16.3 为什么 `submit_after` 与 `publish` 是下一个前沿

当系统已能很好隐藏元数据导出后，publish 成为下一个不可回避的边界：

1. 仅拷贝到 staging 的 expert 仍不可直接使用
2. 必须有显式 publish/activation 步骤将其变为可见
3. 这个可见性边界对正确性至关重要

因此，任何降低 `submit_after` 或 `publish` 的尝试都必须保持：

1. 拷贝完成与可见化之间的顺序
2. active-slot 元数据与实际权重的一致性
3. 后续 draft/verify 对“resident”语义的预期

## 17. 风险与当前限制

### 17.1 正确性敏感性

publish-fast 失败表明该子系统对细微时序变化非常敏感。这意味着未来优化需要比“策略不变、拷贝更少”更强的论证。

### 17.2 Benchmark mismatch 不是最终结论

benchmark 级确定性 mismatch 很有用，但不足以单独定性 bug。更强检查是直接 token 一致性。这也是本文区分“观察到 benchmark mismatch”与“确认 token 级发散”的原因。

### 17.3 更低 cache ratio 会自然扰动推测轨迹

在 `75%`，尤其 `50%` cache ratio 下，系统执行真实 prefetch、真实 staging 和真实 publish。这会改变细粒度时序，即便 benchmark 设置一致，不同运行的 profile 计数也可能无法一一对齐。这使正确性门禁更重要，而不是更次要。

### 17.4 Verify 仍然昂贵

尽管本文聚焦 draft 侧重叠，verify 仍是端到端推测延迟中的大项。这意味着某些 draft 侧收益在完整 step 层面可能被部分掩盖，除非后续也优化 verify。

## 18. 建议下一步

### 18.1 更细地拆分 `submit_after` 与 `publish` 埋点

下一步有价值的 profiling 改进是把 publish 拆成更小子阶段：

1. ready polling
2. candidate selection
3. victim selection
4. staging-to-active transfer 或 activation
5. metadata commit
6. 向后续消费者的 visibility handoff

如果不拆分，就只能知道 `publish_ms` 很贵，却不知道具体哪个子阶段在主导。

### 18.2 将可见性语义作为一等 API 边界

被拒绝的 publish-fast 尝试提示系统需要更明确的状态模型，定义 expert 何时被认为：

1. 已拷贝
2. 已 staged
3. 已 ready
4. 已 published
5. 对 draft 合法可见
6. 对 verify 合法可见

让这些状态更清晰，可能有助于后续更安全优化。

### 18.3 仅在语义完全一致时继续压缩元数据载荷

元数据导出已不是最大瓶颈，但载荷大小或表示方式可能仍有安全收益。前提是这些收益不改变队列更新顺序或下游解释语义。

### 18.4 研究 expert 数量上限与可用重叠裕量的关系

长期控制策略可能应当是定量化的：

1. 测量 prefetch transfer 与 publish 时间随 expert 数量变化
2. 将其与可用 draft-compute slack 对比
3. 设置提交上限，使 `prefetch_time <= hidden_slack`

这是迈向“按构造隐藏 prefetch”而非“尽力隐藏”的自然下一步。

### 18.5 若 publish 持续脆弱，可考虑更大架构改造

若 publish 优化持续呈现“昂贵且正确性敏感”，可能需要更大重构：

1. 显式版本化 cache 可见性
2. 拆分“拷贝完成”与“routing 可见”
3. 在 staging 与 active 状态之间施加强 barrier
4. 让 verify 只消费完全 publish 的 generation
5. 重构 victim selection，使其对可变 live 状态依赖更小

## 19. 总结

本轮优化的已接受结果是：

1. 保留 cluster-compute skill 的资源选择流程更新
2. 保留已接受的 RMSNorm 稳定化、异步 metadata worker、buffer 生命周期分离、host-buffer pool 与元数据物化压缩改动
3. 拒绝 2026-04-27 的 publish-fast 路径，因为它虽改善了局部 publish 指标，但未通过最终正确性门禁

实践结论是：

1. 元数据导出与 host-buffer 复用已基本不再是首要问题
2. 系统行为已更接近预期重叠设计
3. 下一真正优化前沿是 `submit_after` 与 `publish`
4. 该前沿对正确性敏感，应作为调度语义工作处理，而不仅是微优化
