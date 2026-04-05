# Phase2 Post 实现对照评审与修复记录（2026-04-05）

## 1. 评审范围与依据

本评审对照以下设计文档执行：

1. docs/phase2_post.md
2. docs/phase2_design.md
3. docs/migration_design.md

本次评审覆盖以下代码与产物：

1. 核心实现：nanovllm/config.py、nanovllm/engine/model_runner.py、nanovllm/engine/speculative/spec_engine.py、nanovllm/expert/placement.py、nanovllm/scheduling/draft_scheduler.py、nanovllm/layers/fuse_moe/heterogeneous.py、nanovllm/models/qwen3_moe.py、nanovllm/expert/cache.py
2. benchmark与脚本：examples/heterogeneous_benchmark_case.py、examples/three_mode_speed_compare.py、examples/benchmarks/moe_single_layer_cpu_gpu_parallel_bench.py、examples/benchmarks/spec_verify_cpu_ratio_bench.py
3. 测试：tests 全量
4. 结果文件：benchmarks/results/three_mode_smoke_profile_avg.json、benchmarks/results/three_mode_smoke_profile_avg_fix_spec.json、benchmarks/results/moe_single_layer_cpu_gpu_parallel_bench_smoke.json

## 2. 总体结论

当前实现已完成从 Phase2 baseline 向 phase2_post 主干设计的大部分关键迁移，尤其在以下方面达到设计目标：

1. Draft graph 正确性：不再复用 standard decode graph，具备 draft-mode 专属 graph 路径与 graph-safe 回退。
2. placement/build GPU-first：核心构建路径已转为设备侧张量流程，减少 host crossing。
3. 真实 CPU expert 执行首版：已具备 serial + batched per-expert 路径，并保留 legacy fallback。
4. profile/benchmark 统一字段：关键分段指标已可输出与归因。
5. spec 关键问题已修复：deterministic 对齐恢复通过，吞吐显著回升。

同时，仍有阶段性未完成项：

1. CPU/GPU same-layer overlap 尚未形成真实并行（目前逻辑仍偏串行聚合）。
2. cpu_expert_parallel_mode=expert_parallel 仍未落地线程池/任务并行执行。
3. P3 大规模 S<N 曲线和端到端多档结果尚未系统产出。

## 3. 对照 phase2_post 逐阶段状态

| 阶段 | 设计目标 | 当前状态 | 结论 |
|---|---|---|---|
| P0 | 统一 profile/benchmark 字段 | route/plan/gpu/cpu/scatter/spec/draft/verify/graph/cpu ratio 等字段已接入并可在 engine_profile 中聚合 | 基本完成 |
| P1 | 真实 CPU expert execution | 已有 real_cpu_execution 分支（CPU上 F.linear，按 expert 批处理），可开关；legacy fallback 保留 | 完成首版 |
| P2 | CPU/GPU 同层并行 | 尚未形成真正 overlap，当前 CPU 路径仍在同一调用链串行执行 | 未完成 |
| P3 | S<N benchmark主分析 | 已新增单层与spec verify脚本，但系统化多档实验和瓶颈地图未完整执行 | 部分完成 |
| P4 | speculative placement/build 全GPU化 | plan 数据结构 device-first，scheduler 提供 GPU mask/LUT，S=N 快路径已加 | 基本完成 |
| P5 | Draft CUDA Graph 自动优先 | 已实现 draft 专属 graph capture/replay，条件受 enforce_eager 与 draft_top_c 控制，模板未命中回退 eager | 完成首版 |

## 4. 当前实现内容梳理

### 4.1 配置与模式

在 nanovllm/config.py 中新增并约束了以下关键开关：

1. cpu_expert_execution_enabled
2. cpu_expert_num_threads
3. cpu_expert_parallel_mode
4. draft_cuda_graph_enabled
5. draft_cuda_graph_max_bs
6. draft_cuda_graph_bucket_steps
7. perf_profile_level
8. gpu_plan_builder_enabled / gpu_plan_builder_fallback（保留为后续扩展位）

### 4.2 placement/build GPU-first

nanovllm/expert/placement.py 已完成以下演进：

1. MoEExecutionPlan 改为 device-first 结构（gpu_route_indices、gpu_m_sizes、cpu_route_indices、cpu_task_expert_ids、cpu_task_offsets、substitution_lut、route masks）。
2. build_prefill_plan_gpu、build_draft_plan_gpu、build_verify_plan_gpu 已形成统一入口。
3. Draft 路径已使用 GPU mask/LUT 进行 cpu_expert 选择与 substitution。
4. 新增 S=N 快路径，all-cached 时短路复杂 draft 规划。

### 4.3 scheduler 与缓存

1. nanovllm/scheduling/draft_scheduler.py 已新增 GPU 接口：select_cpu_experts_gpu、build_substitution_lut_gpu。
2. nanovllm/expert/cache.py 新增 cached_expert_mask，支持在设备侧判断 cached 集合，减少 host dict 依赖。

### 4.4 真实 CPU 执行与 profile

nanovllm/layers/fuse_moe/heterogeneous.py：

1. 支持 real_cpu_expert_execution（serial + batched per-expert）。
2. 保留 legacy_gpu_fallback 作为兜底。
3. 输出分段 profile：gpu_gather_ms、gpu_compute_ms、cpu_prepare_ms、cpu_compute_ms、cpu_to_gpu_merge_ms、scatter_ms。

### 4.5 模型与Runner链路

1. nanovllm/models/qwen3_moe.py：draft/verify 使用 GPU-first plan；聚合每层 profile（route_ms、plan_ms、cpu_route_ratio 等）。
2. nanovllm/engine/model_runner.py：
   - Draft 模式下独立 graph 策略，不复用标准 decode graph。
   - capture_draft_cudagraph 与 replay 路径已接入。
   - graph_hit_rate、graph_replay_count 与 MoE聚合指标可输出。

### 4.6 benchmark脚本

1. 已新增单层脚本：examples/benchmarks/moe_single_layer_cpu_gpu_parallel_bench.py
2. 已新增spec ratio脚本：examples/benchmarks/spec_verify_cpu_ratio_bench.py
3. three_mode_speed_compare.py / heterogeneous_benchmark_case.py 支持 gpu_memory_utilization 参数透传。

## 5. spec 性能下降与 deterministic 失配的调试/修复过程

### 5.1 问题现象（修复前）

来自 benchmarks/results/three_mode_smoke_profile_avg.json：

1. spec throughput_output_tok_s = 4.7132
2. spec_vs_standard ratio = 0.4795
3. deterministic 对齐：spec_vs_standard exact_match = false
4. spec model_plan_ms = 287.9890
5. spec model_run_draft_total_ms = 522.4685

在 S=N 场景下，这一结果不符合预期（不应出现明显失配和过度降速）。

### 5.2 根因定位

定位到两个直接根因：

1. speculative KV 追加时序与 decode 约定不一致
   - draft迭代中 reserve/append 顺序导致 block_table 与 token 进展错位风险。
2. draft substitution 语义集合错误
   - substitution 应作用于 uncached 且非 cpu_selected 的 need_substitution 集合，而不是 cpu_selected 集合本身。
   - 同时 S=N 缺少 all-cached 快路径，造成不必要 plan 开销。

### 5.3 修复动作

1. 调整 spec_engine 中 draft KV 追加时序：
   - 仅在 draft 迭代间预留下一步 slot。
   - verify 重建阶段首 token 复用已预留 slot，后续 token 再追加。
2. 修复 placement 中 substitution 目标集合：
   - 改为 need_substitution_mask = uncached_expert_mask & (~cpu_expert_mask)
3. 增加 S=N 快路径：
   - all-cached 时直接生成简化 plan，跳过不必要分支。
4. 补充测试：
   - tests/test_spec_engine_flow.py
   - tests/test_placement_spec.py

### 5.4 修复验证（修复后）

来自 benchmarks/results/three_mode_smoke_profile_avg_fix_spec.json：

1. spec throughput_output_tok_s = 9.2621（较修复前显著提升）
2. spec_vs_standard ratio = 0.8620（由 0.4795 提升）
3. deterministic 对齐：spec_vs_standard exact_match = true
4. spec model_plan_ms = 50.6407（由 287.9890 降低）
5. spec model_run_draft_total_ms = 204.2947（由 522.4685 降低）

同时，tests 全量通过（32 passed）。

## 6. 当前风险与未完成项

1. P2 并行语义风险：
   - real CPU path 与 GPU path 仍未形成可证实的 same-layer overlap。
2. expert_parallel 模式未真正实现：
   - cpu_expert_parallel_mode=expert_parallel 目前仍复用 serial 路径。
3. graph命中率风险：
   - 当前 smoke 主要在 enforce_eager=true 下验证逻辑，graph_hit_rate 尚未形成稳定的业务侧命中统计。
4. P3 数据完整性不足：
   - 单层脚本可运行，但大样本多档统计（含置信区间、重复实验）尚未形成固定报告模板。

## 7. 下一步实施计划（下一章）

本章节作为后续执行的唯一入口，按优先级推进。

### 7.1 N1：完成 P2（CPU/GPU 同层并行）

目标：在同层 forward 中实现可观测 overlap，而非串行。

实施项：

1. 将 CPU executor 拆分为异步任务接口（future/worker pool）。
2. GPU 路径与 CPU 路径并行发起，merge 前同步。
3. 新增 tests/test_cpu_gpu_parallel_moe.py，验证数值一致性与并行语义。
4. profile 增加 overlap 可证实字段（如 cpu_wait_ms、gpu_wait_ms 或 overlap_ratio）。

验收：

1. 单层 synthetic routing 下结果与串行参考一致。
2. profile 证实 CPU/GPU 计算段存在重叠。

### 7.2 N2：完成 cpu_expert_parallel_mode=expert_parallel

目标：把 expert_parallel 从配置占位变为真实能力。

实施项：

1. 新增 nanovllm/expert/cpu_executor.py（线程池/任务切片）。
2. 增加线程数与 PyTorch 线程设置的防过度超订阅保护。
3. 分别评估 serial 与 expert_parallel 的拐点范围。

验收：

1. expert_parallel 在中高 cpu_route_ratio 下相对 serial 出现可重复收益。
2. 结果在不同 repeat 中波动可控。

### 7.3 N3：完成 P3 主报告（S<N）

目标：形成可复现、可归因的 S<N 主性能报告。

实施项：

1. 单层 benchmark：
   - activated_expert_set_size 固定为 8，扫描 realized_cpu_expert_count=0..8
   - 至少两档 token size，warmup/repeat 固定
2. 端到端 benchmark：
   - ratio=25%/50%/75%
   - 同时记录 set ratio、route ratio、weight mass ratio
3. 输出标准化报告到 docs/summary 下新文件。

验收：

1. 得到 CPU 瓶颈拐点与端到端瓶颈地图。
2. 报告可直接支撑下一轮优化决策。

### 7.4 N4：Draft CUDA Graph 命中率与模板策略收敛

目标：在业务可用设置下提升 graph_hit_rate，并确保 miss 时回退稳定。

实施项：

1. 扩展 draft bucket 策略并记录 hit/miss 原因。
2. 增加 enforce_eager=false + draft_top_c=0 的系统化回归脚本。
3. 在 docs/summary 形成 graph 策略小结与建议默认参数。

验收：

1. deterministic 条件下 eager/graph 一致。
2. graph 命中率在目标工作负载上可复现。

---

本报告结论：当前实现已达到“P1首版 + P4基本完成 + P5首版完成 + spec关键问题修复完成”的阶段目标，下一步应优先补齐 P2 并行与 P3 系统化 S<N 报告，以完成 phase2_post 的主闭环。