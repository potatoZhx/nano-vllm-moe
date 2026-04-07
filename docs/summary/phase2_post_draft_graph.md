# Phase2 Post: Draft CUDA Graph 完整实现总结（统一 top_c=0 路径，无 S=N 特判）

## 1. 目标与结论

本文档记录 Phase2 Post 中 Draft CUDA Graph 的最终实现形态、约束、验证方法与后续扩展入口。

本次实现的核心目标：

1. 废弃仅用于测速对齐的 S=N 特判路径。
2. 将 Draft CUDA Graph 的可用路径统一为 top_c=0：
	1. Draft 阶段不允许 CPU expert 执行。
	2. 对所有原本应落在 CPU 的激活专家执行 GPU 替换。
3. 在验收场景 S=N 下，虽然实际替换数为 0，但仍走同一 top_c=0 代码路径。
4. 在真实模型 workload 下验证：
	1. graph 可启用并 replay；
	2. deterministic 与 standard 对齐；
	3. draft forward 与 standard decode forward 性能对齐。

结论：目标已达成。

---

## 2. 关键实现变更

### 2.1 路径统一：移除 S=N 特判，固定走 top_c=0 替换逻辑

文件：`nanovllm/expert/placement.py`

实现要点：

1. 删除 `expert_cache.num_slots >= num_experts` 的 S=N 早返回特判。
2. 在 `build_draft_plan_gpu(..., top_c, ...)` 中新增统一分支：
	1. `top_c <= 0` 时构建 substitution LUT；
	2. 将 `flat_selected_original` 映射到 `flat_selected_effective`；
	3. 所有 route 走 GPU，`cpu_route_indices=None`。

这意味着：

1. S=N 只是在该统一路径下“替换数=0”的特例。
2. S<N 且 top_c=0 时，uncached 专家会被替换到 cached 专家后再执行。

### 2.2 专家替换策略：查表 + 轮转（简单实现）

文件：`nanovllm/expert/placement.py`

新增 `_build_topc0_substitution_lut(...)`：

1. 输入：`cached_expert_mask`、`slot_to_expert_lut`、`num_experts`。
2. 输出：长度为 `num_experts` 的 `substitution_lut`。
3. 映射规则：
	1. cached 专家：identity 映射（`lut[e]=e`）；
	2. uncached 专家：按 expert_id 对 slot 数取模，轮转映射到 cached slot 对应 expert。

性质：

1. 确定性（相同 cache 状态 + 输入下映射稳定）。
2. 简单可复现。
3. 可直接替换为更复杂策略（如权重质量优先、最近访问频率优先）。

### 2.3 为替换查表提供设备侧缓存映射

文件：`nanovllm/expert/cache.py`

新增：

1. `slot_to_expert_lut`（device tensor）。
2. `get_slot_to_expert_lut()` 访问接口。

并在 `put_to_slot(...)` 中同步更新该 LUT。

目的：

1. top_c=0 替换路径可完全在设备侧查表。
2. 降低 host 参与和 graph capture 风险。

### 2.4 Graph 兼容性修复（与本路径相关）

文件：`nanovllm/expert/placement.py`

1. 将 grouped 计数从 `bincount` 改为固定形状 `scatter_add_`。
2. 避免在 capture 中触发已知不稳定算子。

文件：`nanovllm/engine/model_runner.py`

1. `spec` 模式不再捕获 standard decode graph（仅捕获 draft graph）。
2. standard graph 不可用时，decode 自动回退 eager，避免错误 replay。

---

## 3. top_c=0 下支持 CUDA Graph 的替换策略约束（必须满足）

以下约束是“替换策略可用于 CUDA Graph”的硬条件：

1. 无 CPU expert 执行：
	1. top_c=0 语义下 `cpu_route_indices` 必须为空。
2. 映射闭包在 cached 集合内：
	1. 对任意被替换 expert `e`，`lut[e]` 必须指向当前缓存中的合法 expert。
3. 确定性与可复现：
	1. 相同输入/缓存状态必须得到一致 LUT。
4. 固定形状优先：
	1. 避免依赖数据值引起的动态输出形状算子。
	2. 避免 Python 侧基于 CUDA tensor 值的分支判断。
5. 回退语义清晰：
	1. 若配置不满足 graph 条件（如 `enforce_eager=true`、`draft_top_c!=0`），必须安全回退 eager。
6. 验证一致性：
	1. temperature=0 下，graph 与 eager、spec 与 standard 必须可对齐。

---

## 4. 测试与验收

### 4.1 单元/回归测试

1. `tests/test_placement_spec.py`
	1. 验证 top_c=0 下 S<N 会发生替换且无 CPU route。
	2. 验证 S=N 下仍走 top_c=0 路径但 LUT 为 identity（替换数为 0）。
2. `tests/test_draft_cuda_graph.py`
	1. 验证 graph policy 与 fallback 行为。
3. `tests/test_draft_standard_decode_forward_bench.py`
	1. 验证 benchmark 指标抽取与 graph 校验逻辑。
4. `tests/test_draft_cuda_graph_real_world.py`
	1. 真实模型场景下验证 graph 启用、正确性、速度对齐区间。

### 4.2 真实场景 benchmark（S=N 验收口径）

命令（moe_spec，真实模型）：

1. 运行脚本 `examples/benchmarks/draft_standard_decode_forward_bench.py`
2. 关键参数：
	1. `--slots-per-layer 0`（S=N）
	2. `--draft-top-c 0`
	3. `--enforce-eager false`

结果文件：

1. `benchmarks/results/draft_standard_decode_forward_real_graph_compare_topc0_unified.json`

本次结果（中位数）：

1. standard decode forward：13.236 ms，75.555 tok/s
2. draft forward：16.151 ms，61.925 tok/s
3. 比值：
	1. 时延比 1.220
	2. 吞吐比 0.820
4. deterministic：`exact_match=true`

解释：

1. Draft 仍有 speculative 链路附加开销，通常不会完全等于 standard decode。
2. 当前对齐指标保持在预期阈值内，并且 graph replay 与正确性均通过。

---

## 5. 给后续开发者的接力说明

如果你第一次接触该项目，可按以下顺序继续：

1. 阅读路径：
	1. `nanovllm/engine/model_runner.py`：graph 捕获/重放与 policy。
	2. `nanovllm/models/qwen3_moe.py`：draft/verify/normal 三模式调用点。
	3. `nanovllm/expert/placement.py`：top_c=0 替换与 route 规划。
	4. `nanovllm/expert/cache.py`：LUT 与 cache 元数据。
2. 本地验证基线：
	1. 跑 `tests/test_placement_spec.py`。
	2. 跑 `tests/test_draft_cuda_graph_real_world.py`（需开启环境变量）。
3. 若要替换策略升级（例如质量感知替换）：
	1. 保留 top_c=0 “无 CPU route”语义；
	2. 保留 LUT 形状稳定和 deterministic；
	3. 先在 S=N 与 S<N 两类场景做 deterministic 验证，再比较性能。

---

## 6. 当前已知边界

1. 当前 top_c=0 替换策略采用简单轮转，不保证最优质量/最优性能。
2. 若未来引入更复杂策略（history/weight-aware），必须严格遵守第 3 章约束。
3. top_c>0 路径仍可演进，但不在本次 CUDA Graph 主验收范围内。

---

## 7. 最终状态

1. 已废弃 S=N 特判实现。
2. 已建立统一 top_c=0 draft graph 路径。
3. 在 S=N 验收场景下：
	1. 同一路径替换数量自然为 0；
	2. 结果与 standard deterministic 对齐；
	3. draft forward 与 standard decode forward 达到可接受对齐范围。
