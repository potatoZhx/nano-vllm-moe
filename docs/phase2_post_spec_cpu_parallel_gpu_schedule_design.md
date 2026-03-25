# Phase 2+ 详细设计：投机解码后续增强（CPU 专家并行 + 纯 GPU 调度 + Draft CUDA Graph）

## 1. 目标与范围

本文档是对 [docs/migration_design.md](docs/migration_design.md) 与 [docs/phase2_design.md](docs/phase2_design.md) 的后续扩展设计，目标是在现有 speculative 基线可运行前提下，补齐以下能力：

1. CPU 专家计算并行增强
2. CPU/GPU 专家计算真正并行
3. 纯 GPU 调度可行性评估与实现方案
4. 当 draft_top_c=0 时，Draft 路径支持 CUDA Graph
5. 完整实验设计与验收口径

说明：
1. 本文档只定义设计与实验方案，不包含代码变更。
2. 本阶段默认保持当前 S=N 验证口径作为主验收场景。

---

## 2. 当前现状与问题定义

基于现有实现（Speculative 主链路已打通）可归纳为：

1. Spec 语义链路已可运行，但 CPU 专家路径仍偏“功能可用”，算子与调度并行度不足。
2. heter/draft/verify 的关键调度仍有 CPU 参与（尤其在 route 解析、plan 构建、索引整理阶段）。
3. Draft 路径默认 eager，尚未具备可捕获、可复用的 CUDA Graph 快路径。
4. 在混合 CPU/GPU 专家场景下，缺少针对 verify 的专项性能数据闭环。

因此需要新增“性能导向但不破坏语义”的设计层。

---

## 3. 设计总览

新增三条主线：

1. 计算主线：CPU 专家算子优化 + CPU/GPU 双路径并行
2. 调度主线：route 到 plan 的纯 GPU 化（能纯 GPU 则纯 GPU，不能则分层降级）
3. 执行主线：draft_top_c=0 时切换到 Draft CUDA Graph

分期建议：

1. Phase A: 可观测性先行（分段计时、计数、trace 稳定化）
2. Phase B: CPU 专家算子与并行改造
3. Phase C: 纯 GPU 调度路径落地（heter/draft/verify）
4. Phase D: draft_top_c=0 的 CUDA Graph 方案
5. Phase E: 统一基准、回归与验收

---

## 4. CPU 专家计算并行设计

## 4.1 目标

1. 降低 CPU 专家路径单 token 延迟
2. 避免 Python 循环成为瓶颈
3. 在同一层内实现 CPU 与 GPU 专家并行执行，并最终聚合

## 4.2 CPU 专家算子优化

现状痛点：
1. 逐 expert/逐 token 的 Python 循环组织计算，调度开销高。
2. 小 batch 下重复 tensor 切片与拼接开销大。

优化策略：

1. 路由重排向量化
- 将 cpu_route_indices 按 expert 分组时，避免 Python 逐元素聚类。
- 采用张量化排序键：expert_id 为主键，token_id 为次键，一次排序得到连续段。

2. 分段 batched GEMM
- 对同 expert 的 token 子集构造连续视图，执行批量线性计算。
- gate_up 与 down 保持与 GPU 路径一致的融合顺序，减少中间写回。

3. CPU 线程池与算子亲和
- 引入固定线程池（避免每步创建销毁）。
- 分配规则按 expert 段粒度，减少锁竞争。
- 使用 pinned host buffer 做中间缓存，降低跨设备聚合时的隐式拷贝抖动。

4. 缓冲区复用
- 预分配 CPU 输出缓冲（按最大 token 段长度），避免每步 malloc/free。

算法草图：

1. 输入：hidden_states, cpu_route_indices, expert_ids, routing_weights
2. 生成按 expert 连续的 index 段
3. 对每段执行 fused gate_up/down（CPU 实现）
4. 乘 routing_weights 后写入 cpu_output_buffer
5. 与 gpu_output 做最终聚合

复杂度变化（定性）：
1. 由 Python 循环主导转为张量与算子主导
2. 调度复杂度不变，但常数项显著下降

## 4.3 CPU/GPU 专家并行执行

目标：
1. GPU 专家路径和 CPU 专家路径在同一层并行推进。
2. 聚合点前只做必要同步。

执行模型：

1. GPU stream 执行 GPU route（fused_moe kernel）
2. CPU 线程池执行 CPU route（batched linear）
3. 聚合阶段等待两侧完成后一次 index_add

同步约束：
1. 不在 CPU 分支中阻塞 GPU stream
2. 聚合前只做一次同步屏障

异常策略：
1. CPU route 为空时快速跳过 CPU 分支
2. GPU route 为空时快速跳过 GPU 分支
3. 两侧皆空直接返回零增量

---

## 5. 验证与测速设计（CPU 比例实验）

## 5.1 单层 MoE 延迟曲线实验

目标：
1. 固定总激活专家数为 8
2. 扫描 CPU 计算专家数量从 0 到 10
3. 记录单层前向延迟

实验变量：
1. cpu_expert_count in [0, 1, 2, ..., 10]
2. top_k 固定
3. token 数固定（建议两档：小 batch 与中 batch）

记录字段：
1. latency_ms_p50/p90/p99
2. cpu_compute_ratio = cpu_routes / total_routes
3. gpu_compute_ratio = gpu_routes / total_routes
4. cpu_queue_wait_ms（若线程池排队）
5. cpu_compute_ms, gpu_compute_ms, merge_ms

输出：
1. CSV + JSON（便于画曲线）
2. 延迟-CPU比例曲线图

### 5.1.1 实施形态（已确认）

已确认采用“单独测试文件 + 单层 MoE 定向压测”的方式，不走完整端到端生成链路。

建议新增独立脚本：
1. examples/benchmarks/moe_single_layer_cpu_gpu_parallel_bench.py

测试流程：
1. 加载模型并完成权重初始化。
2. 仅选定一个目标层（layer_idx）进行循环测试。
3. 构造固定 hidden_states 与固定路由权重输入。
4. 强制注入可控的 selected_experts（总激活专家数固定为 8）。
5. 扫描 cpu_expert_count=0..10，执行单层 forward 并计时。

路由控制方案（优先级顺序）：
1. 方案 A（推荐）：在目标层 MoE block 增加测试专用 hook，直接覆盖 selected_experts/routing_weights。
2. 方案 B（备选）：截获 router 输出后原地改写 selected_experts，再继续走 plan 构建与执行。

方案选择原则：
1. 若能稳定注入 selected_experts，优先方案 A。
2. 若模型结构不便注入 hook，则使用方案 B。

对比口径：
1. hook/改写只改变路由输入，不改变后续执行路径。
2. 仍使用真实 heterogeneous_moe_forward，以保证计时有工程意义。

### 5.1.2 强制路由生成规则（已确认）

固定约束：
1. 每 token 激活专家数保持为 top_k（模型配置）。
2. 全样本“专家集合规模”固定为 8。
3. 在该 8 个专家中，按实验档位指定 CPU 专家数量（0 到 10；超过 8 时按 8 截断并记录 realized 值）。

记录字段补充：
1. requested_cpu_expert_count
2. realized_cpu_expert_count
3. activated_expert_set_size
4. activated_cpu_expert_ratio_set = realized_cpu_expert_count / activated_expert_set_size

### 5.1.3 输出与复现实验约束

1. 每档位至少预热 10 次，正式计时 100 次。
2. 输出 p50/p90/p99 与均值、标准差。
3. 固定随机种子，保存本次 selected_experts 样本到 JSON，便于复跑。
4. 默认在 CUDA_VISIBLE_DEVICES=3 上运行。

## 5.2 Verify 场景 CPU 比例实验

目标：
1. 评估 verify 真实带 CPU 专家的开销
2. 比例档位：25%, 50%, 75% 的专家在 CPU（不在 GPU cache）

实验设置：
1. speculative 打开
2. verify 仍保持一次性验证新增轨迹（prefill-like）
3. 每档位至少重复 5 次，取中位数并给抖动范围

记录字段：
1. verify_ms
2. draft_ms
3. spec_step_ms
4. accepted_tokens_total, accept_rate
5. cpu_expert_ratio_realized_set（按专家集合占比计算）

验收建议：
1. 各档位均需保持 deterministic 对齐（temperature=0）
2. verify_ms 随 CPU 比例上升应可解释（线性或次线性，需结合 profile）

比例定义（已确认）：
1. 25%/50%/75% 按“专家集合占比”定义，而非路由命中条目占比。
2. 即先确定本层可用专家集合 E，再按比例将 |E_cpu| 设为 0.25|E|、0.5|E|、0.75|E|。
3. 每轮记录 realized_set_ratio，若因约束导致偏差需输出偏差值。

---

## 6. 纯 GPU 调度可行性评估与方案

## 6.1 评估结论

结论分层：

1. 可纯 GPU 的部分
- flatten route
- expert remap（lut/index_select）
- gpu_mask 构建
- route indices 生成
- m_sizes 统计（bincount）
- draft substitution map 应用（若映射表驻留 GPU）

2. 难以纯 GPU 的部分（需要策略化）
- 复杂全局策略（跨层历史、prefetch 优先级）天然更适合 CPU 控制面
- 与 CPU expert 线程池协调的执行决策最终仍需主机侧提交

3. 设计选择
- 数据面纯 GPU：每步 route->plan 主路径尽可能 GPU 化
- 控制面保留 CPU：低频策略更新、阈值更新、日志与诊断

## 6.2 纯 GPU 调度架构

新增概念：
1. DevicePlanBuilder
- 输入 GPU 张量（selected_experts, routing_weights）
- 输出 GPU 计划张量（gpu_routes, cpu_routes, m_sizes, effective_experts）

2. HostControlPlane
- 仅处理低频策略参数
- 每 N 步刷新一次策略快照到 GPU 常量缓冲

heter/draft/verify 三阶段映射：
1. heter: build_prefill_plan_gpu
2. draft: build_draft_plan_gpu（含 top_c 与 substitution）
3. verify: build_verify_plan_gpu（无替换，保证精确）

失败降级机制：
1. 若 GPU plan builder 不满足约束（例如出现不支持形态）
2. 自动降级到现有 CPU plan builder，并打点记录 fallback_count

## 6.3 纯 GPU 可行性边界

1. 当存在 CPU expert 路径时，“完全零主机参与”不现实，但“调度计算纯 GPU”可实现。
2. 当 draft_top_c=0 且无 CPU expert 参与时，可接近端到端 GPU-only。

---

## 7. Draft CUDA Graph 设计（draft_top_c=0）

## 7.1 设计前提

启用条件：
1. draft_top_c == 0
2. Draft 路径不触发 CPU expert
3. shape 满足 capture 约束（batch、seq 增量、max_draft_tokens 模板化）

禁用条件：
1. 出现 CPU route
2. 动态 shape 超出图模板
3. 触发不支持算子或外部分配

## 7.2 执行流程

1. 预热阶段
- 对固定模板进行 capture
- 缓存 graph handle（按关键维度分桶）

2. Draft 执行阶段
- 命中模板则 replay
- 未命中则 eager 执行并可选补抓图

3. Verify 阶段
- 保持现有 verify 语义，不在本子阶段强制 graph 化

图模板键建议：
1. batch_size
2. draft_steps（或上限模板）
3. hidden_size/top_k
4. device id

首版范围（已确认）：
1. 仅覆盖 num_seqs=1 的稳定模板。
2. 多序列模板放入下一阶段扩展。

## 7.3 正确性约束

1. graph 仅替换执行方式，不改变 draft/verify/accept 语义
2. deterministic 场景输出必须与 eager 一致
3. profile 需记录 graph_hit_rate 与 graph_replay_count

---

## 8. 统一实验与验收计划

## 8.1 基础验收

1. 三模式均可运行：standard/heter/spec
2. temperature=0 下 token 对齐：
- heter vs standard
- spec vs standard

## 8.2 CPU 专家并行验收

1. 单层 0~10 CPU 专家延迟曲线可生成
2. verify 场景 25/50/75% CPU 比例数据完整
3. profile 字段完整且可复现

## 8.3 纯 GPU 调度验收

1. heter/draft/verify 三阶段输出 gpu_plan_builder 命中率
2. fallback_count 可观测
3. 与现有路径结果一致

## 8.4 Draft CUDA Graph 验收

1. draft_top_c=0 条件下支持 capture/replay
2. 输出 graph_hit_rate, replay_count
3. 与 eager 结果一致
4. 输出吞吐对比：
- draft eager tok/s
- draft graph tok/s
- 提升比

---

## 9. 新增配置项（设计级）

建议在配置层新增如下字段（仅设计，不代表已实现）：

1. cpu_expert_parallel_enabled: bool
2. cpu_expert_num_threads: int
3. cpu_expert_batch_fusion: bool
4. gpu_plan_builder_enabled: bool
5. gpu_plan_builder_fallback: bool
6. draft_cuda_graph_enabled: bool
7. draft_cuda_graph_max_bs: int
8. draft_cuda_graph_bucket_steps: list[int]
9. perf_profile_level: str  # basic | detailed

---

## 10. 风险与缓解

1. 风险：GPU 调度内核引入额外 kernel launch，反而拖慢小 batch
- 缓解：小 batch 自动走简化路径，记录分段开销

2. 风险：CPU/GPU 并行引入同步点抖动
- 缓解：聚合前单点同步，减少细粒度 barrier

3. 风险：CUDA Graph 模板爆炸
- 缓解：按 bucket 约束模板数，超出回退 eager

4. 风险：多进程实验环境抖动导致结论失真
- 缓解：固定空闲 GPU、重复跑取中位数、异常立即复跑

---

## 11. 已确认口径

1. 单层 0~10 CPU 专家实验采用独立测试文件，优先通过 hook 直接指定激活结果；若实现受限，允许截获并改写路由结果。
2. verify 25%/50%/75% 比例按“专家集合占比”定义。
3. Draft CUDA Graph 首版仅覆盖 num_seqs=1。

---

## 12. 实施顺序（仅计划）

1. P0: 增强 profile 与实验脚本字段，先拿到可解释数据
2. P1: CPU 专家算子向量化 + 线程池并行
3. P2: CPU/GPU 同层并行与聚合同步优化
4. P3: DevicePlanBuilder（heter -> draft -> verify）
5. P4: draft_top_c=0 的 CUDA Graph
6. P5: 全量 benchmark 与回归，形成阶段报告

---

## 13. 与现有设计文档关系

1. [docs/migration_design.md](docs/migration_design.md)
- 本文档细化了其中“CPU-GPU 混合执行、Draft Scheduler、调度与加速”的实现级方案与实验口径。

2. [docs/phase2_design.md](docs/phase2_design.md)
- 该文档把 Draft/Verify/CUDA Graph 列为后续阶段；本文档即对应后续阶段的详细落地计划。
