# Phase 1 实现报告：nano-vllm-moe CPU-GPU 基础异构推理

## 1. 文档目的与范围

本报告总结 nano-vllm-moe 在 Phase 1（基础异构推理，不含 speculative draft-verify）阶段的实际落地实现，覆盖：

1. 设计目标与约束。
2. 已落地架构与代码实现。
3. 关键执行流程（加载、路由、执行、聚合）。
4. 本阶段性能优化清单（逐项说明：功能、思路、算法、复杂度）。
5. 当前性能瓶颈复盘。
6. 下一阶段可行优化（重点包括融合内核设计与并行化方案）。

说明：本报告聚焦“已实现且可运行”的 Phase 1 主路径（enable_heterogeneous 打开后走异构 MoE 路径）。

---

## 2. Phase 1 目标与验收口径

### 2.1 目标

1. 在不重写主干引擎的前提下，实现 MoE 层 CPU-GPU 异构执行能力。
2. 在 S=N（每层 slot 数等于该层 expert 数）场景稳定跑通，避免重复显存占用导致 OOM。
3. 保持标准路径可回退，异构路径可开关。
4. 建立可复现实验脚本，具备吞吐与正确性对比能力。

### 2.2 约束

1. 保留现有 grouped GEMM（Triton）作为 GPU expert 计算主算子。
2. 尽量避免改动 attention/KV cache/scheduler 主流程。
3. 以最小侵入集成到已有 ModelRunner + Qwen3Moe 模型结构中。

### 2.3 口径

1. 端到端：standard vs heterogeneous 吞吐比（output_tps / total_tps）。
2. 算子级：remap / plan / gather / fused gate_up / fused down / scatter 分项耗时。
3. 正确性：token 级 exact match rate + 文本样例对照。

---

## 3. 已落地架构总览（实现态）

### 3.1 配置与入口

在 Config 中新增异构开关与 slot 配置：

1. enable_heterogeneous
2. heterogeneous_slots_per_layer
3. cpu_expert_pin_memory

运行入口中通过 ModelRunner 注入：

1. 若 enable_heterogeneous=False，走原始全 GPU 加载与标准 MoE。
2. 若 enable_heterogeneous=True，调用 HeterogeneousModelLoader，构建每层 LayerExpertCache 与 cpu_expert_pool，再将其挂载到 MoE block。

### 3.2 关键模块职责

1. nanovllm/utils/heterogeneous_loader.py
- 拆分非 expert 与 expert 权重加载路径。
- 非 expert 权重直接加载到 GPU 参数。
- expert 权重加载到 CPU pool（可 pin memory），并初始化 layer slot cache。

2. nanovllm/expert/cache.py
- 每层 LayerExpertCache，维护：
  - gate_up_buffer: [S, N, K]
  - down_buffer: [S, N, K]
  - slot_to_expert / expert_to_slot / expert_to_slot_lut
- 提供 remap_experts_to_slots（专家索引 remap）与 get_layer_buffers。

3. nanovllm/expert/placement.py
- 基于 selected_experts 构建 MoEExecutionPlan：
  - gpu_route_indices
  - cpu_route_indices
  - m_sizes
- 负责路由 token 的 GPU/CPU 分流与 slot 分组布局。

4. nanovllm/layers/fuse_moe/heterogeneous.py
- 异构 MoE forward 主逻辑：
  - GPU 路径：gather -> fused gate_up -> fused down -> weighted scatter
  - CPU fallback 路径：按 expert 分组 linear 计算并回写

5. nanovllm/models/qwen3_moe.py
- 新增 Qwen3MoeHeterogeneousSparseMoeBlock。
- 在 decoder layer 构建时，基于 config.enable_heterogeneous 选择标准块或异构块。
- 通过 enable_heterogeneous_mode 注入 layer cache + cpu pool。

---

## 4. 端到端执行流程（Phase 1）

## 4.1 初始化与加载

1. ModelRunner 初始化模型结构。
2. 异构模式下，HeterogeneousModelLoader 执行：
- Step A: 非 expert 权重加载到 GPU。
- Step B: expert 权重加载到 CPU pool。
- Step C: 为每层创建 LayerExpertCache（S 由 heterogeneous_slots_per_layer 决定；若 <=0 则默认 S=N）。
- Step D: 初始 placement 将 expert 写入 slot buffer（S=N 时为 expert i -> slot i）。

3. 模型层对象通过 enable_heterogeneous_mode 绑定对应 layer cache + cpu pool。

## 4.2 推理阶段（单层 MoE）

1. Router 计算 selected_experts 与 routing_weights。
2. 在 heterogeneous_moe_forward 中 flatten 路由结果。
3. build_moe_execution_plan：
- remap_experts_to_slots
- 构造 gpu_route_indices
- 计算 m_sizes（按 slot 计数）
- 计算 cpu_route_indices（若有）

4. GPU 路径执行：
- gpu_token_indices = gpu_route_indices // top_k
- gather hidden_states[gpu_token_indices]
- fused_moe_linear(gate_up)
- fused_moe_linear(down)
- 乘路由权重
- output.index_add_ scatter 回 token 维

5. CPU fallback（当前 S=N 通常不触发）：
- 对 cpu_route_indices 按 expert 分组计算 F.linear
- 加权后 index_add_ 回 output

---

## 5. 关键数据结构与复杂度模型

定义：

1. M：当前层 token 数。
2. k：top-k（num_experts_per_tok）。
3. N = M*k：展开后的路由条目数。
4. S：该层 GPU slot 数（S<=E，S=N 场景下 S=E）。

核心对象：

1. selected_experts: [M, k] -> flatten 后 [N]
2. gpu_route_indices / cpu_route_indices: [Ng] / [Nc]，Ng+Nc=N
3. m_sizes: [S]

时间复杂度（当前实现层面）：

1. remap（LUT index_select）：O(N)
2. gpu mask nonzero：O(N)
3. slot sort（用于 grouped layout）：O(Ng log Ng)
4. bincount：O(Ng + S)
5. gather/scatter：O(Ng * hidden_size) 的内存搬运型开销
6. fused gate_up/down：与 token 分组和矩阵维度相关，近似 O(GEMM)

---

## 6. 本阶段优化清单（逐项详解）

以下为已在 Phase 1 内实施的优化与重构，按“功能 -> 思路 -> 实现算法 -> 复杂度变化 -> 价值”说明。

### 6.1 优化 A：异构初始化形态修复（避免 full expert + slot 双占用）

1. 功能
- 解决异构模式初始化阶段 OOM 风险。

2. 优化思路
- 异构块不再构建 full expert GPU 参数副本。
- expert 权重走 CPU pool + slot buffer 的单一路径。

3. 实现算法
- 模型构建时选择 Qwen3MoeHeterogeneousSparseMoeBlock。
- 加载器只将非 expert 权重加载为模型参数；expert 权重进入 cpu_expert_pool。
- 初始 placement 将 expert 写入每层 slot buffer。

4. 复杂度分析
- 初始化时间复杂度仍受权重读取主导，量级不变。
- 显存复杂度从 O(E * layers * expert_size + slot_size) 收敛到 O(S * layers * expert_size)，S=N 时仍避免了参数与 cache 双份驻留。

5. 价值
- 消除双份占用导致的启动 OOM。
- 为后续 S<E 缓存策略打下统一内存模型。

### 6.2 优化 B：remap 从 Python 循环切换为 LUT 张量索引

1. 功能
- 降低 remap 路径的 Python 解释器开销和分支开销。

2. 优化思路
- 使用 expert_to_slot_lut + index_select 一次性映射。

3. 实现算法
- 预构建长度为 num_experts 的 LUT（未缓存置 -1）。
- 对 flat_selected 直接 index_select 得到 slot_indices。
- gpu_mask = slot_indices >= 0。

4. 复杂度分析
- 优化前：Python 循环近似 O(N)，但常数项高且不可并行。
- 优化后：Tensor 化 O(N)，常数项显著降低，GPU 端并行执行。

5. 价值
- remap 分项耗时显著下降并稳定。

### 6.3 优化 C：plan 输出 route indices，减少下游重复构造

1. 功能
- 避免 heterogeneous forward 内重复 mask/filter/index 构建。

2. 优化思路
- plan 阶段直接产出 gpu_route_indices 与 cpu_route_indices。
- forward 直接消费 route index。

3. 实现算法
- build_moe_execution_plan 内：
  - nonzero(gpu_mask) 得 gpu_route_indices
  - nonzero(~gpu_mask) 得 cpu_route_indices
  - 仅在 GPU 路由存在时再做 slot 分组与 m_sizes

4. 复杂度分析
- 总量级仍为 O(N)+O(Ng log Ng)，但避免重复 passes，减少中间张量创建。

5. 价值
- 降低 plan+forward 之间的冗余内存流量与 kernel 数。

### 6.4 优化 D：去除 inv_sort 回排链路，改为 route 索引直接聚合

1. 功能
- 简化排序后结果回排逻辑，减少一次额外重排。

2. 优化思路
- 统一以排序后 route 顺序进入 fused 与 scatter，不再构造 inv_sort_idx 回排到原次序。

3. 实现算法
- 使用 gpu_route_indices 推导 token 索引与权重。
- fused 输出直接按同序乘权重，index_add_ 聚合。

4. 复杂度分析
- 删除一次 O(Ng) 重排与对应内存读写。

5. 价值
- 降低 memory-bound 子步骤耗时和显存带宽压力。

### 6.5 优化 E：plan 内去 .item() 等同步热点

1. 功能
- 避免 host-device 同步导致的流水阻塞。

2. 优化思路
- 不在热路径读取标量到 CPU，保持张量逻辑闭环。

3. 实现算法
- 使用张量判断、numel 分支，避免频繁 Python 标量提取。

4. 复杂度分析
- 渐近复杂度不变，关键在降低同步 stall。

5. 价值
- 让多子算子更连贯地在 GPU stream 上排队执行。

### 6.6 优化 F：benchmark/debug 脚本标准化与可解释输出

1. 功能
- 提升回归数据可信度、可比性与可读性。

2. 优化思路
- 用自然语言 prompt 代替纯随机 token。
- 输出文本对照、token 摘要、exact match 统计。
- robust benchmark 采用 ABAB/BABA 交替顺序减少顺序偏差。

3. 实现算法
- speed_compare 使用子进程隔离执行标准/异构。
- 增加多轮重复并计算 mean/median/std/p90。

4. 复杂度分析
- 统计侧开销 O(R)（R 为重复轮数），不影响推理核心复杂度。

5. 价值
- 便于识别真实性能变化与正确性退化。

### 6.7 优化 G：LLMEngine.exit 幂等化

1. 功能
- 解决 atexit 双调用导致的退出噪声与潜在异常。

2. 优化思路
- exit 前判断 model_runner 是否已释放。

3. 复杂度分析
- O(1) 守卫逻辑。

4. 价值
- 基准脚本多进程执行更稳定，减少非业务噪声。

---

## 7. 当前性能状态（Phase 1 收敛结果）

根据最近同配置回归（S=N，slots=0 默认映射为每层全部 expert）可观察到：

1. 端到端异构吞吐约为标准路径的 0.82x 左右（约 -17.5%）。
2. debug 分项中，plan 与 fused 子项处于同一量级，scatter 次之，remap 已明显压缩。
3. CPU fallback 在 S=N 基本为 0，不是主要瓶颈。

结论：

1. Phase 1 已完成“可运行 + 可测 + 可优化”的基础异构框架。
2. 当前瓶颈已从“功能正确性/初始化 OOM”转向“路由整理与数据重排开销”。

---

## 8. 为什么 plan 会很贵（即使逻辑已精简）

这部分是 Phase 1 复盘中的关键认识。

1. plan 不是“几行 Python”，而是多段 GPU 内存操作的组合：
- nonzero
- index_select
- sort
- bincount

2. 这些操作具有典型 memory-bound 特征：
- 算术强度低，受限于全局内存读写/重排。
- 小批高频调用时，kernel launch 常数项占比很高。

3. sort 的复杂度为 O(Ng log Ng)，在每层每步都重复。

4. 与 GEMM 的“高 FLOPs 高吞吐”不同，plan/gather/scatter 属于“低 FLOPs 高搬运”，因此耗时可接近甚至显著可见。

---

## 9. Phase 1 进一步优化方向（重点：融合与并行）

以下是 CPU-GPU 基础异构在下一阶段最有收益的优化设计，按优先级给出。

## 9.1 方向 1：路由 plan 融合核（替代 nonzero+sort+bincount）

1. 目标
- 将当前多 kernel 的路由整理压缩为单/少数 kernel。

2. 设计
- 使用线性分桶（histogram + prefix-sum + scatter）直接构建：
  - m_sizes[S]
  - gpu_route_indices（按 slot 分桶）
  - cpu_route_indices（可选）

3. 算法
- Pass 1: 统计各 slot 计数（histogram）
- Pass 2: 前缀和得到每 slot 写入偏移
- Pass 3: 按偏移把 route 写入连续缓冲

4. 复杂度
- 目标从 O(Ng log Ng) 降到 O(N)

5. 预期收益
- 降低 plan 时间与 kernel 启动数量。
- 减少中间张量与显存读写。

## 9.2 方向 2：plan + gather 融合

1. 目标
- 避免先产 route 再二次 gather hidden 的双遍历。

2. 设计
- 在分桶写 route 时直接写 packed_hidden（或 packed token idx + weight）。

3. 算法
- route kernel 输出结构化 packed buffers：
  - packed_hidden
  - packed_weights
  - m_sizes

4. 复杂度
- 渐近 O(N) 不变，减少一次 O(Ng * hidden_size) 额外访存 pass。

5. 风险
- 需要注意 packed buffer 对齐、dtype、跨层复用管理。

## 9.3 方向 3：down + scatter 融合（Epilogue 聚合）

1. 目标
- 消除独立 index_add_ scatter kernel。

2. 设计
- 在 down 的 kernel epilogue 阶段直接根据 token index 做写回/归约。

3. 算法
- block 内先局部归约，再原子写回 output，减少全局原子冲突。

4. 复杂度
- 仍为 O(Ng * hidden_size) 级，但可显著降低常数项和原子冲突成本。

5. 风险
- kernel 实现复杂度高，需要较强数值一致性验证。

## 9.4 方向 4：工作区预分配与内存池化

1. 目标
- 降低每步临时张量创建和 allocator 抖动。

2. 设计
- 预分配 route/gather/scatter 所需 buffer（按 max token 上界）。
- 每步仅复用与覆盖。

3. 复杂度
- 渐近复杂度不变，降低分配常数项。

## 9.5 方向 5：多 stream 并行与流水重叠

1. 目标
- 让可并行子步骤重叠执行，提高设备利用率。

2. 设计
- stream_plan：路由整理
- stream_gemm：fused 计算
- stream_aux：可选 CPU fallback 数据拷贝
- 用事件依赖管理同步点，仅在聚合前栅栏

3. 注意
- 过度并行会触发 L2/DRAM 竞争，需通过 profiler 评估收益拐点。

## 9.6 方向 6：形状稳定场景下引入图捕获

1. 目标
- 降低高频小 kernel 的 launch overhead。

2. 设计
- 对稳定 batch/token 形状建立若干图模板。
- 变长场景走 eager 回退。

3. 风险
- 动态形状较多时图复用率不足，维护复杂度上升。

## 9.7 方向 7：CPU 路径并行化（针对 S<E 阶段）

1. 场景
- S=N 时 CPU fallback 很少，收益有限。
- 当 S<E 且 fallback 出现时，该方向价值显著。

2. 设计
- CPU expert 批化 + 多线程执行 + 异步 H2D 回传。
- 将 CPU 路径结果与 GPU 路径在最终聚合前同步。

3. 复杂度
- 逻辑复杂度提升，但可把 CPU 变为“并行补偿路径”，避免串行拖慢。

---

## 10. 建议的下一阶段实施顺序

建议按收益/改造风险比排序：

1. 第一优先：实现路由 plan 融合核（线性分桶，替代 sort 流程）。
2. 第二优先：推进 plan+gather 融合，减少一次大规模内存搬运。
3. 第三优先：引入工作区预分配，稳定延迟。
4. 第四优先：探索 down+scatter 融合（高收益高复杂度）。
5. 第五优先：在稳定形状下尝试图捕获策略。

---

## 11. 验证与风险控制建议

1. 正确性验证
- token 级一致性（greedy）
- 文本语义一致性（采样）
- 分层输出误差统计（可选）

2. 性能验证
- 固定随机种子 + ABAB/BABA 顺序
- 至少 5 次重复，报告 mean/median/std/p90
- 分项耗时与端到端吞吐同时看，避免局部优化破坏全局

3. 风险点
- 融合核引入后调试难度显著增加
- 原子写回与并行归约可能带来数值漂移
- 多 stream 设计不当会引入额外同步点，反而退化

---

## 12. Phase 1 小结

Phase 1 已完成“基础 CPU-GPU 异构推理”的核心闭环：

1. 从权重加载、模型结构、执行计划到异构 forward 全链路可运行。
2. 修复了初始化形态导致的 OOM 风险。
3. 通过多轮张量化与路径简化，已将瓶颈集中到可定位、可工程化优化的路由/重排环节。
4. 当前性能差距已可量化，下一阶段重点应转向融合内核与流水并行，以进一步逼近标准路径吞吐。

该阶段为后续 speculative、预取与高级调度策略提供了稳定底座。

---

## 13. 与 migration_design 的对照实现状态（4.1 / 4.2 / 4.3 / 4.9）

本章针对 `docs/migration_design.md` 中的关键设计段落，给出三类结论：

1. 已实现的内容。
2. 已实现且在 Phase 1 中重构优化后的内容（与原设计存在实现差异，但目标一致）。
3. 未实现内容（保留到后续 Phase）。

### 13.1 4.1 Config 扩展 对照

已实现：

1. `enable_heterogeneous`（已接入配置与运行入口）。
2. `heterogeneous_slots_per_layer`（已用于层级 slot 数控制；`<=0` 走 S=N）。
3. `cpu_expert_pin_memory`（已用于 CPU expert 权重 pin memory 开关）。

重构优化实现：

1. Phase 1 采用“最小配置闭包”，只保留异构基础运行所需字段，避免在未落地 speculative/策略模块时引入过多配置噪声。
2. slot 计算不走 `gpu_memory_limit_gb` 自动估算，而是直接由 `heterogeneous_slots_per_layer` 驱动，便于 S=N 基线验证。

未实现：

1. `gpu_memory_limit_gb`、`expert_gpu_memory_gb`（自动显存预算与 slot 自动规划未落地）。
2. `expert_placement_config`（YAML 初始放置策略未落地）。
3. speculative 与策略相关配置：`enable_speculative`、`max_draft_tokens`、`draft_top_c`、`acceptance_threshold`、`acceptance_strategy`、`cache_strategy`、`prefetch_strategy`、`draft_scheduler`。

### 13.2 4.2 异构参数加载器 对照

已实现：

1. 异构加载器 `HeterogeneousModelLoader` 已落地，并由 `ModelRunner` 调用。
2. 非 expert 权重与 expert 权重分流加载：
  - 非 expert 权重加载到模型参数（GPU）。
  - expert 权重加载到 CPU pool（支持 pin memory）。
3. 已实现 gate/up 合并为 `gate_up` 的存储形态，并为每层构建 cache。
4. 已实现初始 slot 填充流程并在加载后同步。

重构优化实现：

1. 设计文档中的“全局 ExpertCacheManager”在 Phase 1 实现为“每层一个 `LayerExpertCache`”：
  - 返回值为 `dict[layer_idx, LayerExpertCache] + nested cpu_pool`。
  - 该结构更贴合当前模型层级注入方式，减少跨层调度复杂度。
2. 初始放置策略采用“按 expert id 排序后顺序写 slot”，而不是 YAML/随机混合策略：
  - 可重复性更好。
  - 便于做 S=N 公平基线。
3. `slots_per_layer <= 0` 直接映射为 S=N，优先保证“先跑通 + 先测准”。

未实现：

1. `compute_slots_per_layer` 的显存预算推导链路未落地。
2. YAML placement 解析与按层随机补齐未落地。
3. `cache_expert(layer_idx, expert_idx)` 这种“按需装载 + 替换策略驱动”的运行期加载在 Phase 1 基线中未启用（当前更偏静态 S=N 预装载）。

### 13.3 4.3 Expert Cache Manager 对照

已实现：

1. 分层固定 slot buffer 的核心思想已实现。
2. 每层维护 `slot_to_expert / expert_to_slot` 映射。
3. 已实现 remap 功能，且采用 LUT 张量索引优化（`expert_to_slot_lut + index_select`）。
4. 已实现 buffer 获取接口并与 fused MoE 执行路径打通。

重构优化实现：

1. 文档中的 `ExpertCacheManager`（含跨层策略、异步传输队列）在 Phase 1 落地为轻量 `LayerExpertCache`：
  - 把关注点聚焦到“单层缓存 + remap + fused 执行必需接口”。
  - 先消除路径复杂性，再迭代策略层。
2. remap 从字典/循环思路演进为 LUT 张量化，复杂度保持 O(N) 但常数显著下降。
3. 与 placement 联动时，plan 已简化为 `gpu_route_indices + cpu_route_indices + m_sizes`，避免冗余索引链。

未实现：

1. 缓存替换策略框架（LRU/LFU/Adaptive）未落地。
2. 异步 `cache_expert_async`、`transfer_stream`、`complete_transfers` 未落地。
3. `get_cached_experts`、`evict_from_slot`、`build_m_sizes(layer_idx, ...)` 等策略化接口未完整落地。
4. S<E 运行期动态换入换出尚未进入 Phase 1 基线。

### 13.4 4.9 ModelRunner 扩展 对照

已实现：

1. `ModelRunner` 已支持异构模式初始化：
  - `enable_heterogeneous` 下走异构加载器。
  - 将 layer cache 与 cpu pool 注入模型。
2. 标准运行主循环 (`run`) 保持兼容，异构路径通过模型内部 MoE block forward 生效。

重构优化实现：

1. 文档中的 `run_draft / run_verify / run_standard` 三分路径，在 Phase 1 被有意延后。
2. 当前采用“Runner 不扩散、MoE block 内聚”的改造策略：
  - 优点：对原引擎侵入更小，回归风险更低。
  - 代价：speculative 相关控制点尚未在 Runner 层展开。
3. CUDA Graph 相关行为维持原逻辑，异构基线主要在 eager 配置下评估。

未实现：

1. `run_draft`、`run_verify`、`run_standard` 专用接口未落地。
2. Runner 层逐层显式构建 draft/prefill placement 的流程未落地。
3. 与 speculative engine 的完整集成（accept/verify 循环）未落地。

### 13.5 补充：4.4/4.5（与本次对照主题强相关）

虽然本节重点要求 4.1/4.2/4.3/4.9，但 Phase 1 的性能收敛主要发生在 4.4/4.5 对应实现，补充如下：

已实现：

1. `MoEExecutionPlan` 已落地（当前字段为 `gpu_route_indices`、`cpu_route_indices`、`m_sizes`）。
2. 异构 forward 的 GPU 路径与 CPU fallback 路径均已落地并可跑通。

重构优化实现：

1. 相比设计中的 `gpu_sort_idx/gpu_inv_sort_idx/gpu_token_map` 全字段方案，Phase 1 改为更轻量 route-index 驱动方案。
2. 去掉 inv-sort 回排与重复索引构造，减少 O(Ng) 额外内存重排。

未实现：

1. 设计中的 CPU ThreadPool 并行执行框架在 S=N 基线场景下未作为主要路径落地。
2. plan/gather/scatter 融合内核与深度流水并行仍属于后续优化项。

### 13.6 对照结论

1. Phase 1 已完整实现“基础异构推理闭环”（配置开关、异构加载、层级 cache、placement、异构 forward、benchmark）。
2. 与 migration_design 相比，当前实现是“收敛版最小可行子集”：
  - 保留核心主干。
  - 延后 speculative、策略框架、动态换入换出。
3. 已完成的“重新修改优化”主要集中在两条主线：
  - 数据结构轻量化（layer cache + route index plan）。
  - 路径张量化（LUT remap、减少回排和重复索引）。
4. 未实现内容与后续 Phase 目标一致，不属于偏离设计，而是分阶段落地取舍。