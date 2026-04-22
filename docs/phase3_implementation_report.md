# Phase 3 实施与评估报告

## 1. 目标与结论

本次实现围绕 `docs/phase3_design.md` 的 baseline 主线完成了端到端落地，核心结果如下：

1. 完成 `Approach 1`（staging memory + replay-boundary publish）实现。
2. 完成 global warm-start/history queue（含 `prefill_history` / `verify_history` / `draft_live` 三来源）。
3. 完成 runtime metadata recorder，并接入 prefill / draft / verify，包含 draft graph capture 的 recorder arm。
4. 完成 `ModelRunner` / `SpeculativeEngine` / `Qwen3Moe` / `LLMEngine` 的联动改造。
5. 完成 cache strategy / prefetch strategy 可插拔骨架。
6. 完成覆盖配置、cache lifecycle、queue、runtime、engine 联动、verify feedback 的功能测试。
7. 完成两个 benchmark 脚本，并生成结果文件用于性能分析。


## 2. 计算环境与执行流程

按个人 cluster workflow 执行了计算节点校验（`jobid=17242`）：

1. `squeue -u $USER` 与 `sinfo` 预检查通过。
2. 通过 `validate_on_compute.sh --jobid 17242` 成功进入计算节点。
3. 关键校验结果：
   - `hostname=gpu15-A100-E2-3U`
   - `CONDA_DEFAULT_ENV=nano_moe`
   - `Python 3.12.13`
   - `torch 2.8.0+cu128`
   - `cuda_available=True`
   - `SRUN_STATUS=0`
4. 日志落盘：`/home/mumura/cluster_logs/cluster_compute_personal_20260417_232906.log`


## 3. 实现内容（按设计特性分组）

### 3.1 Feature 1：Config 扩展与顶层装配

已实现：

1. 在 `nanovllm/config.py` 新增 prefetch/cache 相关配置字段与校验。
2. 在 `ModelRunner.__init__` 新增：
   - `cache_strategy` / `prefetch_strategy` 构建
   - `runtime_meta_recorder` / `prefetch_runtime` 构建
   - `_prefetch_step_id` 与 `_next_prefetch_step_id()`
3. prefetch 关闭时仍保持 Phase 2 路径可运行。


### 3.2 Feature 2：Runtime Metadata 与 graph-safe 导出

已实现：

1. 新增 `nanovllm/expert/runtime_meta.py`：
   - `LayerRuntimeMetaCPU`
   - `RuntimeMetaOffloadHandle`
   - `ModelRuntimeMetaRecorder`
2. 在 `Qwen3MoeHeterogeneousSparseMoeBlock.forward()` top-k 后记录 metadata。
3. 在 `Qwen3MoeForCausalLM` 增加 `set_runtime_meta_recorder()` 模型级 setter。
4. 在 `ModelRunner.capture_draft_cudagraph()` 增加 draft bucket 的 recorder `arm/reset`。


### 3.3 Feature 3：LayerExpertCache staging lifecycle

已实现：

1. `LayerExpertCache` 新增：
   - access 统计：`last_access_step` / `access_count` / `access_score_sum`
   - generation：`slot_generation`
   - staging buffer 与 slot state
2. 新增 lifecycle dataclass：
   - `StagingReservation`
   - `PublishedExpert`
   - `LayerCacheSnapshot`
3. 新增 staging 生命周期方法：
   - `reserve_staging_slot`
   - `begin_async_put_to_staging`
   - `mark_staging_ready`
   - `publish_ready_staging_to_active`
   - `commit_published_expert`
   - `cancel_staging_reservation`
4. `put_to_slot()` 保持兼容并保留原用途。


### 3.4 Feature 4 + 5：Global Queue 与 PrefetchRuntime

已实现：

1. 新增 `nanovllm/expert/prefetcher.py`：
   - `PrefetchCandidate`
   - `PrefetchTicket`
   - `GlobalWarmStartQueue`
   - `PrefetchRuntime`
2. queue 能力：
   - 去重（key=`(layer_idx, expert_idx)`）
   - 衰减更新
   - TTL 清理
   - priority 排序
3. runtime 能力：
   - `observe_prefill/observe_draft/observe_verify`
   - `submit_from_global_queue`
   - `poll_ready_tickets`
   - `publish_ready`
   - `wait_for_verify`
   - `record_verify_consumed`
   - profile 原始计数器


### 3.5 Feature 6：Strategy 层

已实现：

1. `nanovllm/scheduling/cache_strategy.py`
   - `LRUCacheStrategy`
   - `LFUCacheStrategy`
   - `AdaptiveCacheStrategy`（当前占位回落到 LRU）
2. `nanovllm/scheduling/prefetch_strategy.py`
   - `NoopPrefetchStrategy`
   - `HistoryWindowPrefetchStrategy`
3. `nanovllm/scheduling/__init__.py` 新增导出。


### 3.6 Feature 7 + 8 + 9：ModelRunner / SpecEngine / Verify Feedback 联动

已实现：

1. `ModelRunner.run()`：prefill recorder arm/offload/observe。
2. `ModelRunner.run_draft()`：
   - replay 前 `publish_ready + submit(before_draft)`
   - replay 后 `observe_draft + submit(after_draft)`
   - 返回 `{"prefetch_step_id": step_id}`
3. 新增 `ModelRunner.wait_prefetch_for_verify()`。
4. `ModelRunner.run_verify()`：
   - verify recorder
   - `observe_verify`
   - `record_verify_consumed`
   - `submit(after_verify)`
5. `SpeculativeEngine` 在 verify 前接入 `wait_prefetch_for_verify`。


### 3.7 Feature 10：DraftScheduler/placement 最小改造

已实现：

1. `placement.py` 新增 helper：
   - `flatten_selected_and_weights`
   - `build_runtime_meta_view`
2. `draft_scheduler.py` 新增：
   - `DraftSchedulerContext`
   - `build_draft_scheduler_context`
3. `MoEExecutionPlan` dataclass 未膨胀，保持紧凑。


### 3.8 Feature 11：Profiling 与 Canonical Alias

已实现：

1. `PrefetchRuntime.get_profile()` 输出 prefetch 关键统计字段。
2. `ModelRunner.get_profile()` 合并 runtime profile。
3. `LLMEngine.get_profile()` 新增 canonical alias：
   - `prefetch_submit_count`
   - `prefetch_completed_count`
   - `prefetch_wait_ms`
   - `prefetch_consumed_count`
   - `publish_count`


## 4. 测试设计与执行结果

### 4.1 新增测试文件

新增（功能测试）：

1. `tests/test_config_prefetch.py`
2. `tests/test_cache_strategy.py`
3. `tests/test_prefetch_strategy.py`
4. `tests/test_expert_cache_staging.py`
5. `tests/test_expert_cache_generation.py`
6. `tests/test_prefetch_global_queue.py`
7. `tests/test_prefetch_runtime_meta.py`
8. `tests/test_prefetch_runtime.py`
9. `tests/test_prefetch_wait.py`
10. `tests/test_model_runner_prefetch.py`
11. `tests/test_spec_engine_prefetch.py`
12. `tests/test_verify_feedback.py`

更新（回归兼容）：

1. `tests/test_model_runner_spec_modes.py`
2. `tests/test_spec_engine_flow.py`
3. `tests/test_draft_scheduler.py`
4. `tests/test_placement_spec.py`


### 4.2 分阶段验证记录

第一次收集阶段出现 10 个收集错误（两类问题）：

1. `ModelRuntimeMetaRecorder` 注解前向引用导致 `NameError`
2. `nanovllm/scheduling` 缺少 `__init__.py` 导致导入 `KeyError`

修复后复测：

1. Phase 3 相关测试集：`39 passed in 8.95s`
2. 相关回归集（`llm_engine/spec_engine/draft_cuda_graph`）：`14 passed in 8.77s`

说明：验证是分阶段执行并即时修复，而不是最后一次性收敛。


## 5. Benchmark 方案与结果

### 5.1 Benchmark 脚本

新增：

1. `benchmarks/scripts/phase3_prefetch_microbench.py`
2. `benchmarks/scripts/phase3_prefetch_onoff_compare.py`

产出结果：

1. `benchmarks/results/phase3_prefetch_microbench.json`
2. `benchmarks/results/phase3_prefetch_onoff_compare.json`


### 5.2 方法学

#### A. `phase3_prefetch_microbench`

目标：度量 runtime 关键环节平均耗时与行为计数。

1. synthetic trace 驱动 8 层、32 experts、top-k=2。
2. 每 step 执行 `observe_draft -> submit -> publish`。
3. 统计 `observe_avg_ms` / `submit_avg_ms` / `publish_avg_ms`。
4. 另测 recorder `record+offload+collect` 平均耗时。

结果（当前运行）：

1. `observe_avg_ms = 3.2014`
2. `submit_avg_ms = 1.7904`
3. `publish_avg_ms = 0.3386`
4. `record_collect_avg_ms = 0.5410`
5. `prefetch_submit_count = 332`
6. `prefetch_completed_count = 332`
7. `publish_count = 320`

#### B. `phase3_prefetch_onoff_compare`

目标：评估 prefetch 开启 vs 关闭的运行时开销差。

1. 统一 synthetic trace。
2. `enabled`: 正常 budget 与 wait。
3. `disabled`: `prefetch_step_budget=0`，近似关停 prefetch 提交流程。
4. 比较 `avg_step_ms` 与 profile 差异。

结果（当前运行）：

1. `enabled.avg_step_ms = 6.3659`
2. `disabled.avg_step_ms = 3.4136`
3. `delta_avg_step_ms = +2.9523`

解释：这是 CPU synthetic 微基准，不代表真实 GPU 大模型端到端吞吐；其主要价值是验证功能路径、行为计数、以及开销分布。


## 6. 与原设计的偏差说明

本次偏差主要集中在“第一版可运行优先”：

1. `adaptive` cache strategy 目前为占位实现，回落到 LRU。
2. `prefetch_consumed_count` 使用“recent publish + verify 激活交集”的 baseline 近似定义。
3. `wait_for_verify()` 采用轻量轮询实现（短 sleep），未引入更复杂事件聚合器。
4. benchmark 目前为 synthetic microbench，尚未接入完整真实模型吞吐 A/B 自动化。
5. 设计文档中的附录优化（Approach2/3、request-scoped queue、predictive verify-prefetch）未进入本次主实现（与文档 baseline 分层一致）。


## 7. 风险、限制与后续工作

### 7.1 已知限制

1. 当前 queue 为 global 级别，跨 request 可能存在工作集互相污染。
2. microbench 的性能结论不能直接外推到真实服务流量。
3. `prefetch_wait_ms` 的收益依赖真实 H2D 与 verify 负载，需在真实模型链路复核。

### 7.2 风险

1. 如果 staging slots 配置过小，会限制提交并降低命中提升空间。
2. publish budget 过低会导致 ready ticket 堆积，过高可能增加抖动。

### 7.3 建议的下一步

1. 增加真实模型 A/B benchmark（phase2_post vs phase3 baseline）并纳入 CI 报告。
2. 引入 request-scoped queue 做公平性和局部性对照实验。
3. 在真实 workload 上调参：
   - `prefetch_step_budget`
   - `prefetch_max_inflight`
   - `cache_eviction_budget_per_step`
   - `prefetch_verify_wait_ms`
4. 评估是否推进附录路径（segmented graph / predictive verify-prefetch）。


## 8. 变更文件总览（本次实现）

核心代码：

1. `nanovllm/config.py`
2. `nanovllm/expert/cache.py`
3. `nanovllm/expert/runtime_meta.py`（新增）
4. `nanovllm/expert/prefetcher.py`（新增）
5. `nanovllm/scheduling/cache_strategy.py`（新增）
6. `nanovllm/scheduling/prefetch_strategy.py`（新增）
7. `nanovllm/scheduling/__init__.py`（新增）
8. `nanovllm/expert/placement.py`
9. `nanovllm/scheduling/draft_scheduler.py`
10. `nanovllm/models/qwen3_moe.py`
11. `nanovllm/engine/model_runner.py`
12. `nanovllm/engine/speculative/spec_engine.py`
13. `nanovllm/engine/llm_engine.py`
14. `nanovllm/utils/heterogeneous_loader.py`

测试与 benchmark：

1. `tests/test_config_prefetch.py`（新增）
2. `tests/test_cache_strategy.py`（新增）
3. `tests/test_prefetch_strategy.py`（新增）
4. `tests/test_expert_cache_staging.py`（新增）
5. `tests/test_expert_cache_generation.py`（新增）
6. `tests/test_prefetch_global_queue.py`（新增）
7. `tests/test_prefetch_runtime_meta.py`（新增）
8. `tests/test_prefetch_runtime.py`（新增）
9. `tests/test_prefetch_wait.py`（新增）
10. `tests/test_model_runner_prefetch.py`（新增）
11. `tests/test_spec_engine_prefetch.py`（新增）
12. `tests/test_verify_feedback.py`（新增）
13. `tests/test_model_runner_spec_modes.py`（更新）
14. `tests/test_spec_engine_flow.py`（更新）
15. `tests/test_draft_scheduler.py`（更新）
16. `tests/test_placement_spec.py`（更新）
17. `benchmarks/scripts/phase3_prefetch_microbench.py`（新增）
18. `benchmarks/scripts/phase3_prefetch_onoff_compare.py`（新增）
19. `benchmarks/results/phase3_prefetch_microbench.json`（新增结果）
20. `benchmarks/results/phase3_prefetch_onoff_compare.json`（新增结果）


## 9. 真实模型 E2E 复核（job17242）

本节为后续补充的真实模型端到端复核结果，重点覆盖 `docs/phase3_design.md` 中对 baseline 可运行性、ablation、profiling 完整性的要求。

### 9.1 复核工况

1. Slurm 运行上下文：`jobid=17242`（A100, `gpu15`）
2. 入口脚本：`benchmarks/scripts/phase3_real_e2e_orchestrator.py`
3. 最终结果文件：`benchmarks/results/phase3_real_e2e_orchestrator_job17242_v7_gpu.json`
4. 评测矩阵：
   - 模式对照：`standard` / `heter` / `spec_prefetch_off`
   - prefetch 全开：`spec_prefetch_on_full`
   - ablation：`draft_live` / `verify_history` / `prefill_history` / `wait=0` / `cache=lfu`

### 9.2 关键修复闭环

本轮真实模型复核中，修复了 4 类真实运行缺陷：

1. **runtime metadata host mirror 设备错误**
   - 症状：`Only dense CPU tensors can be pinned`
   - 修复：host mirror 显式分配在 CPU。
   - 文件：`nanovllm/expert/runtime_meta.py`

2. **global queue 设备泄漏导致 scatter_add 设备不一致**
   - 症状：`index is on cpu, different from other tensors on cuda:0`
   - 修复：`GlobalWarmStartQueue.update_from_runtime_meta()` 聚合路径显式固定 CPU device。
   - 文件：`nanovllm/expert/prefetcher.py`

3. **draft graph + recorder 的 capture-safe 标量写入问题**
   - 症状：`CUDA error: operation not permitted when stream is capturing`
   - 根因：capturing 流中对 CUDA tensor 做 host 标量写入。
   - 修复：改为 device-to-device 写入 `token_count_capture_value`。
   - 文件：`nanovllm/expert/runtime_meta.py`

4. **inference tensor 生命周期与 run_draft 语义不一致**
   - 症状：`Inplace update to inference tensor outside InferenceMode`
   - 修复：`run_draft()` 统一进入 `@torch.inference_mode()`。
   - 文件：`nanovllm/engine/model_runner.py`

### 9.3 最终矩阵结果（v7）

1. 结果总览：`9/9` case 全部成功。
2. `spec_enable_prefetch=true` 家族全部可运行，含 5 个 ablation。
3. `graph_hit_rate`：spec 相关 case 均为 `1.0`，未出现 graph hit 回退。
4. `draft_live_prefetch_submit_count`：
   - full case 为非零（17）
   - `ablate_draft_live` 为 0（符合开关语义）
5. 关键 profile 字段（phase3 新增）已通过 canonical alias 暴露到顶层：
   - `prefetch_submit_count`
   - `prefetch_completed_count`
   - `prefetch_late_count`
   - `prefetch_wait_ms`
   - `prefetch_consumed_count`
   - `prefetch_timeout_count`
   - `publish_count`
   - `publish_ms`
   - `metadata_offload_ms`
   - `metadata_offload_bytes`
   - `history_prefetch_submit_count`
   - `verify_history_prefetch_submit_count`
   - `draft_live_prefetch_submit_count`
   - `verify_ready_before_wait_count`
   - `verify_ready_after_wait_count`

### 9.4 Expectation Gap（设计预期 vs 实测）

#### A. 已满足

1. **Phase 3A~3D 可运行性验收**：通过。spec/prefetch 全链路在真实模型可执行。
2. **Phase 3C `draft_live` 指标**：通过。full case 非零，ablation 为零。
3. **graph-safe 约束**：通过。spec 家族 graph hit rate 保持 1.0。
4. **Feature 11 observability**：通过。关键 prefetch profile 字段已对齐顶层 canonical 输出。

#### B. 存在偏差

1. **吞吐预期偏差（主要 gap）**：
   - `spec_prefetch_on_full` 相对 `spec_prefetch_off` 输出吞吐下降约 `52.56%`（本次工况）。
   - 说明 baseline prefetch 当前在该负载/参数下净收益为负，需要进一步参数调优与策略优化。

2. **输出一致性偏差（digest）**：
   - 各模式与 standard baseline 的 token digest 未对齐。
   - 该行为在 heterogeneous/spec 语义下可出现，但若目标是严格 token-level 一致，还需额外收敛。

3. **CPU 并行分解指标缺口**：
   - spec prefetch-on case 仍缺 `cpu_prepare_ms/cpu_compute_ms/cpu_to_gpu_merge_ms`。
   - 原因是本次工况未打开 CPU-GPU parallel 执行路径；该三项属于并行路径专属指标。

## 10. 下一步建议（针对剩余 gap）

1. 增加一个 `cpu_gpu_parallel_execution_enabled=true` 的真实模型补充 case，补齐 CPU 并行分解指标。
2. 以 `spec_prefetch_off` 为性能基线，做参数网格调优：
   - `prefetch_step_budget`
   - `prefetch_max_inflight`
   - `cache_eviction_budget_per_step`
   - `prefetch_verify_wait_ms`
3. 对 digest 差异做可解释拆分：
   - 固定随机性与温度设置
   - 对比 verify trace / acceptance trace
   - 定位差异是否来自路由波动还是采样路径。
