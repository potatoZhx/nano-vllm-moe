# Phase2 投机解码框架实现报告（review + 设计对照）

## 1. 审查范围与方法

### 1.1 审查范围

本报告覆盖以下内容：

1. 现有 speculative 解码实现的事实正确性与逻辑正确性。
2. 代码质量与实现模式（分层、可扩展性、接口设计、可测试性）。
3. 与两份设计文档的一致性：
   - docs/phase2_design.md
   - docs/migration_design.md
4. 边界条件、异常处理、兼容性、性能、并发相关风险。
5. 下一步整改与实现规划。

### 1.2 审查依据（核心代码）

1. 引擎与流程：
   - nanovllm/engine/llm_engine.py
   - nanovllm/engine/model_runner.py
   - nanovllm/engine/speculative/spec_engine.py
2. 调度与状态：
   - nanovllm/engine/scheduler.py
   - nanovllm/engine/block_manager.py
   - nanovllm/engine/sequence.py
3. MoE 路由与执行：
   - nanovllm/expert/placement.py
   - nanovllm/layers/fuse_moe/heterogeneous.py
   - nanovllm/models/qwen3_moe.py
   - nanovllm/expert/cache.py
   - nanovllm/utils/heterogeneous_loader.py
4. 配置与脚本：
   - nanovllm/config.py
   - examples/three_mode_speed_compare.py
   - examples/heterogeneous_benchmark_case.py
5. 测试：
   - tests/test_spec_engine_flow.py
   - tests/test_model_runner_spec_modes.py
   - tests/test_placement_spec.py
   - tests/test_block_manager_draft.py
   - tests/test_scheduler_draft_kv.py
   - 其余 mode/config/acceptance/scheduler 基础测试

### 1.3 审查结论（摘要）

1. 现有实现已经形成可运行的 Phase2 基线，主链路 Draft -> Rollback -> Verify -> Accept 可执行，且在近期 S=N smoke 条件下可达到与 standard 的 deterministic 对齐。
2. 与设计文档相比，当前实现属于“可运行基线完成、策略化与并发优化未完成”的状态。
3. 仍存在若干设计偏差：CPU fallback 实现与设计并行模型不一致、部分统计语义存在偏差、migration 模块未补齐。
4. 报告第 6 章给出按优先级排序的整改建议与下一阶段计划。

---

## 2. 现阶段实现总览

### 2.1 架构状态

1. 三模式入口（standard/heter/spec）已打通：
   - Config 侧推导与约束：nanovllm/config.py
   - step 分流：nanovllm/engine/llm_engine.py
2. speculative 状态机已打通：
   - SpeculativeEngine 主循环：nanovllm/engine/speculative/spec_engine.py
3. draft/verify 执行模式已从模型层可切换：
   - ModelRunner 模式切换：nanovllm/engine/model_runner.py
   - Qwen3 MoE block 执行模式：nanovllm/models/qwen3_moe.py
4. placement 已分化 prefill plan 与 draft plan：
   - nanovllm/expert/placement.py
5. heterogeneous 执行层支持 plan 注入：
   - nanovllm/layers/fuse_moe/heterogeneous.py
6. 基准与 profile 可用：
   - 三模式脚本：examples/three_mode_speed_compare.py
   - 单 case 执行脚本：examples/heterogeneous_benchmark_case.py

### 2.2 主要流程（spec decode）

1. LLMEngine.step 在 spec 分支调用 SpeculativeEngine.speculative_step。
2. speculative_step 对当前 decode batch：
   - start_draft + draft KV 起点标记
   - 按 _budget_draft_steps 执行 run_draft 循环
   - rollback draft KV + rollback token
   - 构造 verify 输入（last accepted + draft tokens）并 run_verify 一次性前向
   - 逐序列比较 draft 与 verify 轨迹，计算 keep_after_start
   - accept_draft_kv，更新 token_ids/num_tokens/last_token
3. 结束条件由 _maybe_mark_finished 处理（eos / max_tokens）。

### 2.3 核心算法要点（当前实现）

1. draft 步数预算：
   - min(max_draft_tokens, remaining - 1)
   - 预留 1 个 verify-next token 位置
2. verify 语义：
   - run_verify 返回每个 verify query 位置的 argmax 轨迹
   - next_token 由 clamped keep_after_start 决定
3. profile：
   - spec profile：draft/verify/accept 等阶段
   - engine profile：step + runner/model/spec 汇总
4. 批内策略：
   - 仍为统一 draft 上限（未实现批内不同长度终止）

---

## 3. 修改内容章节（文件/API/算法）

本章节按模块说明当前阶段主要改造点与功能。

### 3.1 配置与模式

1. 文件：nanovllm/config.py
2. 主要字段/API：
   - inference_mode
   - enable_heterogeneous / enable_speculative 兼容推导
   - max_draft_tokens, draft_top_c
   - acceptance_strategy, acceptance_threshold
   - draft_scheduler
   - spec_verify_eager, spec_enable_prefetch
   - spec_profile, engine_profile, engine_profile_cuda_sync
3. 算法/行为：
   - 提供 standard/heter/spec 模式统一入口
   - 对不合法组合做 assert（如 spec 且 heterogeneous=False）

### 3.2 Sequence/BlockManager/Scheduler draft 生命周期

1. 文件：
   - nanovllm/engine/sequence.py
   - nanovllm/engine/block_manager.py
   - nanovllm/engine/scheduler.py
2. 主要字段/API：
   - Sequence: start_draft, append_draft_token, rollback_tokens_to_draft_start, finish_draft
   - BlockManager: start_draft, append_draft_token, rollback_draft, accept_draft
   - Scheduler: start_draft_kv, append_draft_kv, rollback_draft_kv, accept_draft_kv
3. 算法/行为：
   - token 与 KV 的 draft 生命周期分离管理
   - rollback/accept 按目标 token 长度收缩 block_table

### 3.3 SpeculativeEngine（状态机）

1. 文件：nanovllm/engine/speculative/spec_engine.py
2. 主要 API：
   - speculative_step
   - _budget_draft_steps
   - get_profile
3. 算法/行为：
   - Draft -> Rollback -> Verify -> Accept
   - verify 一次性前向
   - max_tokens 预算 clamp
   - 输出 per-step 轨迹：draft_steps_per_step 与 step_traces

### 3.4 ModelRunner（draft/verify 运行路径）

1. 文件：nanovllm/engine/model_runner.py
2. 主要 API：
   - run_draft
   - run_verify
   - _set_speculative_execution_mode
3. 算法/行为：
   - run_draft 在 mode=draft 下调用 run decode
   - run_verify 在 mode=verify 下执行 prefill-like 全 query logits 计算
   - profile 记录 prepare/run/sample/draft/verify 细分

### 3.5 MoE 路由计划与执行

1. 文件：
   - nanovllm/expert/placement.py
   - nanovllm/layers/fuse_moe/heterogeneous.py
   - nanovllm/models/qwen3_moe.py
2. 主要 API：
   - build_prefill_plan
   - build_draft_plan
   - heterogeneous_moe_forward(..., plan=None)
   - Qwen3MoeHeterogeneousSparseMoeBlock.set_speculative_execution
3. 算法/行为：
   - draft 模式支持 top_c CPU + substitution_map
   - verify 模式走 prefill plan（无替换）
   - forward 层支持外部 plan 注入

### 3.6 异构加载与缓存

1. 文件：
   - nanovllm/utils/heterogeneous_loader.py
   - nanovllm/expert/cache.py
2. 主要 API：
   - HeterogeneousModelLoader.load
   - LayerExpertCache.remap_experts_to_slots
3. 算法/行为：
   - 非 expert 权重常规加载
   - expert 权重 CPU pool + per-layer slot cache
   - slots_per_layer<=0 时按 S=N 初始化

### 3.7 benchmark 与可观测性

1. 文件：
   - examples/three_mode_speed_compare.py
   - examples/heterogeneous_benchmark_case.py
2. 主要 API：
   - run_case
   - token_alignment
   - summarize
3. 算法/行为：
   - 三模式独立子进程运行
   - 支持 deterministic token 对齐报告
   - warmup 后 reset profile，避免污染计时
   - 参数改为 input-len/output-len（固定长度）

### 3.8 测试覆盖（当前）

1. 流程测试：test_spec_engine_flow.py
2. runner 模式切换：test_model_runner_spec_modes.py
3. placement：test_placement_spec.py
4. block 生命周期：test_block_manager_draft.py
5. scheduler 包装：test_scheduler_draft_kv.py
6. 配置与分发：test_mode_config.py, test_llm_engine_mode_dispatch.py

---

## 4. 设计文档关键约束与验收点对比（逐点映射）

以下对比基于 phase2_design 与 migration_design 的关键设计点。

### 4.1 设计点 1：三模式统一入口与兼容性

- 设计点：三模式模式化切换
- 文档要求：支持 standard/heter/spec，非法组合 fail-fast，保留兼容推导
- 代码实现位置：nanovllm/config.py；nanovllm/engine/llm_engine.py
- 实现说明：Config 完成 inference_mode 推导与断言，LLMEngine.step 依 mode 分流
- 结论：符合
- 依据：config 中 valid_modes 与 spec 约束；step 中 spec 分支

### 4.2 设计点 2：SpeculativeEngine 状态机主链路

- 设计点：Draft -> Verify -> Accept 主循环
- 文档要求：decode 阶段进入 speculative_step，流程完整可运行
- 代码实现位置：nanovllm/engine/speculative/spec_engine.py
- 实现说明：状态机完整，含 rollback、verify、accept 与结束判定
- 结论：符合
- 依据：speculative_step 主体实现

### 4.3 设计点 3：KV draft 生命周期原语

- 设计点：start/append/rollback/accept draft KV
- 文档要求：BlockManager/Scheduler 提供一致接口
- 代码实现位置：nanovllm/engine/block_manager.py；nanovllm/engine/scheduler.py
- 实现说明：四个原语均存在并贯穿 spec 流程
- 结论：符合
- 依据：对应函数已实现并被 SpeculativeEngine 调用

### 4.4 设计点 4：run_verify 一次性 verify 语义

- 设计点：verify 一次性前向全部新增 token
- 文档要求：prefill-like verify，不是逐 token verify
- 代码实现位置：nanovllm/engine/model_runner.py
- 实现说明：run_verify 构造 full hidden_states logits 并按 verify_lengths 切分
- 结论：符合
- 依据：run_verify 中 F.linear(hidden_states, lm_head.weight) + offset 切分

### 4.5 设计点 5：run_draft/run_verify 与 MoE 计划分流

- 设计点：draft 与 verify 使用不同计划构建
- 文档要求：draft 用 build_draft_plan，verify 用 build_prefill_plan
- 代码实现位置：nanovllm/models/qwen3_moe.py；nanovllm/expert/placement.py
- 实现说明：Qwen3MoeHeterogeneousSparseMoeBlock 在 execution_mode 下切换 draft/prefill plan
- 结论：部分符合
- 依据：分流已存在；但 runner 仍通过 mode 全局切换，没有更细粒度 per-layer/per-step 策略注入

### 4.6 设计点 6：接受策略可插拔

- 设计点：AcceptanceStrategy 接口与策略选择
- 文档要求：greedy/standard 按配置接入主链
- 代码实现位置：nanovllm/engine/speculative/acceptance.py；nanovllm/config.py；nanovllm/engine/speculative/spec_engine.py
- 实现说明：策略类通过工厂创建并接入 SpeculativeEngine，接受阶段调用 strategy.accept；当前默认策略保持 greedy。
- 结论：符合
- 依据：spec_engine 在初始化时按 config.acceptance_strategy 构造策略对象并在 accept 阶段调用

### 4.7 设计点 7：DraftScheduler 可插拔

- 设计点：draft scheduler 通过配置选择实现
- 文档要求：draft_scheduler 字段驱动策略工厂
- 代码实现位置：nanovllm/config.py；nanovllm/engine/model_runner.py；nanovllm/scheduling/draft_scheduler.py
- 实现说明：通过 create_draft_scheduler(config.draft_scheduler) 工厂化创建，默认仍为 simple。
- 结论：符合
- 依据：ModelRunner 使用配置驱动工厂创建 draft scheduler

### 4.8 设计点 8：heterogeneous forward 可接收外部 plan

- 设计点：算子层支持 plan 注入
- 文档要求：plan 可外部传入，内部构建仅兼容分支
- 代码实现位置：nanovllm/layers/fuse_moe/heterogeneous.py
- 实现说明：函数签名含 plan=None，未传时回退 build_moe_execution_plan
- 结论：符合
- 依据：heterogeneous_moe_forward 参数与分支逻辑

### 4.9 设计点 9：S=N 约束

- 设计点：S=N 评测不允许专门特判分支
- 文档要求：heter/spec 仍走完整计划路径
- 代码实现位置：nanovllm/utils/heterogeneous_loader.py；nanovllm/expert/placement.py；nanovllm/models/qwen3_moe.py
- 实现说明：slots<=0 映射到 num_experts，不回退 standard；仍经过 plan 与 heterogeneous forward
- 结论：符合
- 依据：slot 初始化与 plan 调用链

### 4.10 设计点 10：sampling 模式流程稳定

- 设计点：sampling 下 speculative 流程可用
- 文档要求：流程稳定、无状态错乱
- 代码实现位置：nanovllm/engine/speculative/spec_engine.py
- 实现说明：当前 sampling 模式直接 fallback 到普通 decode，不走 draft/verify
- 结论：部分符合
- 依据：speculative_step 中 temperature>0 直接 return model_runner.run

### 4.11 设计点 11：预取与策略模块（migration）

- 设计点：prefetcher/spec_scheduler/cache_strategy/prefetch_strategy
- 文档要求：模块化保留并逐步接入
- 代码实现位置：仓库未找到对应模块
- 实现说明：当前仅有 draft_scheduler/simple，其余模块未落地
- 结论：不符合
- 依据：文件缺失；运行链无调用

### 4.12 设计点 12：指标与日志

- 设计点：draft_rounds/accept_rate/verify_time 等观测
- 文档要求：可用于性能与正确性分析
- 代码实现位置：nanovllm/engine/speculative/spec_engine.py；examples/three_mode_speed_compare.py
- 实现说明：已有 run_draft_calls/run_verify_calls/accept_rate/step_traces 等，满足基础观测
- 结论：部分符合
- 依据：spec profile 字段较全，但与文档字段命名并不完全一致，且统计口径尚未统一到多轮报告

---

## 5. 设计未覆盖但代码新增行为

1. benchmark 参数语义调整：从 min/max 输入输出长度改为固定 input-len/output-len。
2. 引擎级 profile 汇总：engine_profile 与 model/spec 前缀合并字段。
3. spec per-step 轨迹：draft_steps_per_step 与 step_traces。
4. warmup profile 清零机制：基准计时前 reset，避免统计污染。

这些行为对调试有明显帮助，但不属于原设计文档明确条目，后续建议在设计文档补充“可观测性扩展”小节统一口径。

---

## 6. 问题清单（按优先级）

### P1-1 CPU fallback 与文档设计不一致（缺少并行执行模型）

- 问题：CPU fallback 采用逐 expert Python 循环并在热路径执行权重 to(device)
- 对应设计要求：migration 设计建议 CPU 路径并行执行（ThreadPool/批处理）
- 代码位置：nanovllm/layers/fuse_moe/heterogeneous.py
- 风险说明：
  1. uncached 场景吞吐显著下降。
  2. Python 循环导致高调度开销，且与“减少 Python 循环”的性能目标冲突。
- 建议修改：
  1. 对 CPU 路径按 expert 分组后并行计算（线程池或批处理）。
  2. 减少每步重复权重搬运，可考虑缓存或异步预取。

### P1-2 spec 分支 num_tokens 统计口径失真

- 问题：spec 分支固定返回 num_tokens = -len(seqs)，未反映每轮可能接受多个 token
- 对应设计要求：指标应可解释并可比
- 代码位置：nanovllm/engine/llm_engine.py
- 风险说明：
  1. decode throughput 展示与真实推进 token 数偏离。
  2. profile/benchmark 对比可能出现误判。
- 建议修改：
  1. 由 speculative_step 返回每序列净推进 token 数并汇总。
  2. 在 step 中使用真实 token 增量更新统计。

### P1-3 sampling 模式直接回退 decode，未执行 spec 主链

- 问题：temperature>0 时 speculative_step 直接走普通 run
- 对应设计要求：sampling 模式流程稳定（至少应有明确策略定义）
- 代码位置：nanovllm/engine/speculative/spec_engine.py
- 风险说明：
  1. 用户以 spec 模式运行 sampling 时，行为与预期不一致。
  2. 无法评估 sampling 下 speculative 策略效果。
- 建议修改：
  1. 明确文档：当前 phase2 仅保证 greedy，sampling 默认降级。
  2. 或实现 sampling 下 acceptance 策略并开放开关。

### P1-4 migration 规划模块缺失

- 问题：spec_scheduler/prefetcher/cache_strategy/prefetch_strategy 未落地
- 对应设计要求：migration 文档中的模块化扩展接口
- 代码位置：nanovllm 目录下无对应文件
- 风险说明：
  1. 后续性能优化路径缺少承载模块，改造成本上升。
  2. 文档与实现偏差持续扩大。
- 建议修改：
  1. 先补空实现与接口，再逐步接线。
  2. 报告中标明“接口就绪/算法未启用”状态。

### P2-1 profile 字段与文档口径未统一

- 问题：现有字段命名与设计文档示例字段不一致
- 对应设计要求：统一指标与日志口径
- 代码位置：nanovllm/engine/speculative/spec_engine.py；examples/three_mode_speed_compare.py
- 风险说明：多版本结果难横向对比，自动分析脚本适配成本高。
- 建议修改：建立统一 profile schema（版本号 + 必选字段 + 可选扩展字段）。

### P2-2 运行时并发资源命名冲突风险

- 问题：SharedMemory 名称固定为 nanovllm
- 对应设计要求：并发与稳定性
- 代码位置：nanovllm/engine/model_runner.py
- 风险说明：多实例并行 benchmark 时可能冲突。
- 建议修改：加随机后缀或基于 dist_port 生成唯一 shm 名。

---

## 7. 边界条件、异常处理、兼容性、性能、权限、并发专项结论

### 7.1 边界条件

1. max_tokens 边界：已处理 keep_after_start clamp，基本正确。
2. draft_steps=0 场景：可运行，但建议补回归用例验证 next token 选择与完成判定。
3. 多序列 offset：run_verify 按 verify_lengths 线性切分，需持续以多序列 case 压测。

### 7.2 异常处理

1. mode 不合法组合有 assert，较清晰。
2. draft 执行缺少对 verify_traces=None 或长度异常的显式保护，建议增加 fail-fast 断言。

### 7.3 兼容性

1. inference_mode 与 legacy flag 兼容推导已实现。
2. benchmark 参数已从 min/max 迁移到 fixed len，旧命令需要同步更新文档。

### 7.4 性能

1. spec 主耗时仍集中在 run_draft 与 run_verify 前向，符合预期。
2. CPU fallback 当前实现仍有明显 Python 循环与搬运开销。
3. engine_profile 已可支撑归因，但统计口径仍待统一。

### 7.5 权限与并发

1. 未发现权限相关高风险。
2. 多实例并发下固定 shm 名称存在潜在冲突风险。

---

## 8. 下一步实现规划（建议执行顺序）

### 8.1 第一优先级（正确性与接口一致性）

1. 修正 spec 分支 token 统计口径（P1）。
2. 在工厂创建路径补充配置值合法性测试（greedy/standard/simple）。

### 8.2 第二优先级（性能主路径）

1. CPU fallback 并行化与搬运优化（P1）。
2. 评估 draft 纯 GPU 调度与 CUDA graph（你已明确此方向）。
3. 保持 S=N 无特判前提下，减少 Python 级循环与对象操作。

### 8.3 第三优先级（migration 模块补齐）

1. 补齐 prefetcher/spec_scheduler/cache_strategy/prefetch_strategy 框架文件。
2. 先空实现 + 接口打桩，再逐步替换算法。

### 8.4 第四优先级（文档与测试完善）

1. 更新 docs/phase2_design.md 与 docs/migration_design.md 的“当前实现状态”附录。
2. 新增测试：
   - acceptance strategy 参数化
   - spec 分支真实 token 增量统计
   - draft_steps=0 / 多序列 verify offset 强化用例
   - 多实例 shm 命名冲突防回归

---

## 9. 最终结论

1. 现有 Phase2 实现已经达到“可运行、可对齐、可观测”的基线目标。
2. 但从设计一致性和工程可扩展性角度看，仍有关键缺口：策略可插拔未闭环、部分 migration 模块未落地、CPU 路径性能模型偏保守。
3. 建议先完成 P0/P1 问题，再进入下一阶段性能优化（CPU 并行、draft 纯 GPU、CUDA graph 验证），以避免后续在不稳定接口上反复返工。
