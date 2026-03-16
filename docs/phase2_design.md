# Phase 2 设计文档：基于 Phase 1 的投机解码（Draft-Verify-Accept）落地方案

## 1. 目标与边界

### 1.1 实现目标（正确性优先）

1. 在现有 Phase 1 CPU-GPU 异构 MoE 推理基础上，新增可运行的投机解码主链路：Draft -> Verify -> Accept。
2. 保证语义正确：最终输出必须等价于以 Verify 路径逐步生成的结果（greedy 场景严格对齐）。
3. 保留并统一支持三种 mode：
	- standard：原始 nano-vllm 路径。
	- heter：异构基础推理路径（无 speculative）。
	- spec：异构路径基础上的 speculative 路径。
4. 支持多序列 batch，支持已有调度器的 prefill/decode 生命周期，不破坏现有标准路径。
5. 保留策略扩展点，但 Phase 2 只实现最简单、稳定、可验证的策略（对应 migration_design 中 4.10 的最简实现要求）。

### 1.2 非目标（留到后续 Phase）

1. 不在 Phase 2 追求最大吞吐，不做复杂预取算法优化。
2. 不实现 verify 跳过优化（可留接口，默认关闭）。
3. 不在异构 speculative 路径启用 CUDA Graph。
4. 不实现复杂自适应接受策略，仅实现标准策略与 greedy 精确比对。
5. 本阶段先实现 batch 内统一 draft 长度，不在主路径实现批次内不同 draft 停止策略。

### 1.3 成功标准

1. 三种 mode 可独立运行，且对外接口一致。
2. Greedy 模式下，heter 与 spec 输出与 standard 输出 token 逐位一致（同输入、同随机种子）。
3. Sampling 模式下，流程稳定、无状态错乱、无 KV 泄漏。
4. Draft/Verify 轮次中的 KV 回滚与保留逻辑正确（block 数与 token 数一致）。
5. 代码具备可插拔接口，后续 Phase 可替换调度/预取/接受策略。
6. 基准评测默认在 S=N 配置下完成，heter/spec 不允许因 S=N 添加特判优化分支。

---

## 2. 现状与差距

基于当前仓库实现（Phase 1）：

1. 已有异构 MoE 前向：
	- `nanovllm/layers/fuse_moe/heterogeneous.py`
	- `nanovllm/expert/cache.py`
	- `nanovllm/expert/placement.py`
2. 已有全局调度与 decode 循环：
	- `nanovllm/engine/llm_engine.py`
	- `nanovllm/engine/scheduler.py`
	- `nanovllm/engine/model_runner.py`
3. 缺失 speculative 核心能力：
	- Sequence 无 draft 状态字段。
	- BlockManager 无 start_draft/rollback_draft/accept_draft。
	- ModelRunner 无 run_draft/run_verify。
	- Engine 无 SpeculativeEngine。
	- 无 AcceptanceStrategy 与 DraftScheduler（简单版）。

当前需要新增的对比与约束能力：

1. 统一 mode 配置与入口，保证 standard/heter/spec 可一键切换。
2. 明确对比实验口径：当前统一使用 S=N（全量 expert 在 GPU），用于观察纯路径开销差异。
3. 在 heter/spec 路径中禁止对 S=N 做专门优化（例如直接回退 standard 或跳过 plan 逻辑）。

---

## 3. 总体架构（Phase 2）

Phase 2 在现有链路上新增一个轻量控制层：

1. decode 阶段进入 `SpeculativeEngine.speculative_step()`。
2. `speculative_step` 内执行三段：
	- Draft：用替换+top-c CPU 的近似执行快速出草稿 token。
	- Verify：用无替换的完整执行对草稿校验。
	- Accept：按策略接收部分草稿，并提交下一个 verify token。
3. KV 生命周期由 BlockManager 增加 draft 状态原语保证一致。

mode 选择规则：

1. standard：完全沿用现有 run 流程。
2. heter：沿用 Phase 1 异构 MoE 流程，不进入 speculative 状态机。
3. spec：仅 decode 阶段进入 SpeculativeEngine；prefill 与 heter 一致。

数据流（单轮）：

1. `start_draft(seq)` 标记起点。
2. Draft 追加 K 个 token（临时 KV）。
3. `rollback_draft(seq)` 回退到起点。
4. Verify 一次处理 `[last_token_before_draft + draft_tokens]`，写入全精度 KV。
5. `accept_draft(seq, M)` 保留前 M 个草稿对应 KV，丢弃其余。
6. 追加 verify 的 next token，完成本轮。

---

## 4. 分文件改造清单

## 4.1 新增文件

1. `nanovllm/engine/speculative/__init__.py`
	- 导出 `SpeculativeEngine`、`AcceptanceStrategy` 构造器。

2. `nanovllm/engine/speculative/spec_engine.py`
	- Draft-Verify-Accept 主循环。
	- 输入：当前 decode 批次 `seqs`。
	- 输出：本轮应提交给 scheduler.postprocess 的 token 列表。

3. `nanovllm/engine/speculative/acceptance.py`
	- 定义接受策略接口。
	- 实现 `GreedyAcceptance` 与 `StandardAcceptance`（最简）。

4. `nanovllm/scheduling/draft_scheduler.py`
	- 定义 Draft 调度接口。
	- 实现 `SimpleDraftScheduler`（最简）：
	  - CPU 选分数最高 top-c 未缓存 expert。
	  - 替代 expert 从本层已缓存 expert 中按顺序/随机选。
	  - 传输选择先不做复杂策略，按激活频次排序。

## 4.2 修改文件

1. `nanovllm/config.py`
	- 新增 speculative 配置项。

2. `nanovllm/engine/sequence.py`
	- 新增 draft 状态字段和管理方法。

3. `nanovllm/engine/block_manager.py`
	- 新增 draft KV 起点记录、回滚、接受接口。

4. `nanovllm/engine/scheduler.py`
	- 增加 draft 相关包装调用，统一管理 block 变化。

5. `nanovllm/engine/model_runner.py`
	- 新增 `run_draft`、`run_verify`。
	- 新增 placement 构建入口（draft 与 verify 区分）。

6. `nanovllm/expert/placement.py`
	- 扩展为支持 `build_prefill_plan` 与 `build_draft_plan`。
	- 在 draft plan 中加入 substitution map 与 CPU/GPU 分配。

7. `nanovllm/layers/fuse_moe/heterogeneous.py`
	- 增加可选参数：接受 `MoEExecutionPlan`（含替换映射和 CPU/GPU 路由），避免在 forward 内重复推断。

8. `nanovllm/models/qwen3_moe.py`
	- MoE block 增加“外部注入 routing/plan”的执行入口（用于 run_draft/run_verify 分流）。

9. `nanovllm/engine/llm_engine.py`
	- 在 decode 路径根据 mode 切换到 standard/heter/spec。

10. `examples/`（新增或改造 benchmark 脚本）
	- 增加统一三模式对比脚本，输出吞吐与正确性对齐结果。

11. `tests/`（新增单元测试与集成测试）
	- 先测试后实现，且每步实现后立即回归测试。

## 4.3 mode 配置与兼容矩阵

建议新增统一字段：

1. `inference_mode: str = "standard"`，可选 `standard | heter | spec`。
2. 保留 `enable_heterogeneous` 与 `enable_speculative` 作为兼容字段，但内部优先使用 `inference_mode`。

兼容规则：

1. 当 `inference_mode=standard`：忽略异构与 speculative 配置。
2. 当 `inference_mode=heter`：要求异构配置可用，speculative 配置不生效。
3. 当 `inference_mode=spec`：要求异构可用；speculative 状态机开启。
4. 非法组合（如 spec 但异构关闭）直接报错。

优先级规则：

1. 若显式指定 `inference_mode`，则 `enable_heterogeneous`/`enable_speculative` 仅作一致性校验，不再单独决定路径。
2. 若未指定 `inference_mode`（兼容旧调用），则按旧字段推导：
	- `enable_speculative=True` -> `spec`
	- `enable_speculative=False` 且 `enable_heterogeneous=True` -> `heter`
	- 否则 -> `standard`

---

## 5. 配置设计

在 `Config` 中新增：

```python
inference_mode: str = "standard"         # standard | heter | spec
enable_speculative: bool = False
max_draft_tokens: int = 8
draft_top_c: int = 2
acceptance_strategy: str = "greedy"   # greedy | standard
acceptance_threshold: float = 0.7
draft_scheduler: str = "simple"
spec_verify_eager: bool = True         # verify 强制 eager
spec_enable_prefetch: bool = False     # Phase 2 默认关闭复杂预取
```

约束：

1. `inference_mode=spec` 时必须 `enable_heterogeneous=True`。
2. `draft_top_c <= num_experts_per_tok`。
3. `max_draft_tokens >= 1`。
4. benchmark 默认配置统一为 `heterogeneous_slots_per_layer=0`（即 S=N），用于路径差异对比。
5. 在 heter/spec 中不得出现 `if S == N` 的功能分叉或快速路径。

---

## 6. 核心接口设计（含简洁伪代码）

## 6.1 Sequence 扩展

新增字段：

1. `draft_token_ids: list[int]`
2. `is_drafting: bool`
3. `_draft_start_num_tokens: int`

新增方法：

```python
def start_draft(self):
	 self.draft_token_ids = []
	 self._draft_start_num_tokens = self.num_tokens
	 self.is_drafting = True

def append_draft_token(self, token_id: int):
	 self.draft_token_ids.append(token_id)
	 self.append_token(token_id)

def finish_draft(self):
	 self.is_drafting = False

def rollback_tokens_to_draft_start(self):
	 self.token_ids = self.token_ids[:self._draft_start_num_tokens]
	 self.num_tokens = self._draft_start_num_tokens
	 self.last_token = self.token_ids[-1]
```

设计要点：

1. `append_draft_token` 会临时污染主 token 序列，这是预期行为。
2. Verify 前必须显式 rollback，保证 verify 从正确前缀重算。

## 6.2 BlockManager 扩展

新增接口：

```python
def start_draft(self, seq):
	 seq._draft_start_num_tokens = seq.num_tokens
	 seq._draft_start_num_blocks = len(seq.block_table)

def append_draft_token(self, seq):
	 self.may_append(seq)

def rollback_draft(self, seq):
	 target = seq._draft_start_num_tokens
	 while seq.num_blocks > ceil_div(target, self.block_size):
		  freed = seq.block_table.pop()
		  self.blocks[freed].ref_count -= 1
		  if self.blocks[freed].ref_count == 0:
				self._deallocate_block(freed)
	 seq.num_tokens = target

def accept_draft(self, seq, num_accepted):
	 target = seq._draft_start_num_tokens + num_accepted
	 while seq.num_blocks > ceil_div(target, self.block_size):
		  freed = seq.block_table.pop()
		  self.blocks[freed].ref_count -= 1
		  if self.blocks[freed].ref_count == 0:
				self._deallocate_block(freed)
	 seq.num_tokens = target
```

正确性要点：

1. `rollback_draft` 和 `accept_draft` 都必须维护 block 引用计数一致。
2. 调用后 `seq.last_token` 需与 `seq.token_ids[-1]` 同步。

## 6.3 DraftScheduler（最简实现）

接口：

```python
class DraftScheduler(ABC):
	 def select_cpu_experts(uncached_experts, routing_weights, selected_experts, top_c): ...
	 def select_gpu_substitutes(need_substitution, cached_experts, all_experts): ...
	 def select_experts_to_transfer(recent_activations, cached_experts, cache_capacity): ...
```

Simple 策略伪代码：

```python
def select_cpu_experts(...):
	 score = aggregate_score_per_expert(selected_experts, routing_weights)
	 return top_c_experts_in_uncached_by_score(score)

def select_gpu_substitutes(need_substitution, cached_experts, ...):
	 # Phase 2: 最简单，循环复用 cached_experts
	 mapping = {}
	 cached = sorted(list(cached_experts))
	 for i, e in enumerate(need_substitution):
		  mapping[e] = cached[i % len(cached)]
	 return mapping
```

说明：

1. 替代策略只保证流程闭环，不保证质量最优。
2. 后续 Phase 再替换为相似度或历史共现策略。

## 6.4 Placement 扩展

从 Phase 1 的 `build_moe_execution_plan` 扩展为两条路径：

1. `build_prefill_plan`：无替换，GPU 命中走 GPU，未命中走 CPU。
2. `build_draft_plan`：
	- 未命中集合中选 top-c 走 CPU。
	- 其余未命中进行替换并映射到 GPU slot。

新增数据结构建议：

```python
@dataclass
class MoEExecutionPlan:
	 gpu_route_indices: Tensor
	 cpu_route_indices: Tensor | None
	 m_sizes: Tensor | None
	 substitution_map: dict[int, int]
	 flat_selected_original: Tensor
	 flat_selected_effective: Tensor
```

draft plan 伪代码：

```python
flat = selected_experts.reshape(-1)
slot_idx, gpu_mask = cache.remap_experts_to_slots(flat)
uncached = unique(flat[~gpu_mask])
cpu_set = scheduler.select_cpu_experts(uncached, routing_weights, flat, top_c)
need_sub = uncached - cpu_set
sub_map = scheduler.select_gpu_substitutes(need_sub, cached_experts, all_experts)

flat_eff = apply_substitution(flat, sub_map)
slot_eff, gpu_mask_eff = cache.remap_experts_to_slots(flat_eff)
gpu_routes = nonzero(gpu_mask_eff)
cpu_routes = nonzero((~gpu_mask_eff) | isin(flat, cpu_set))
m_sizes = bincount(slot_eff[gpu_routes])
```

正确性注意：

1. 替换只能替成本层已缓存 expert。
2. CPU 集合中的 route 不能被替换覆盖。
3. `flat_selected_original` 需保留，供统计与调试对照。
4. 当某层无 CPU expert 被选中（常见于 S=N）时，CPU 路径直接跳过，不执行任何空转逻辑。（先不实现）

## 6.5 heterogeneous_moe_forward 扩展

当前函数在内部自行 build plan。Phase 2 建议改为：

1. 若传入 `plan`，直接执行。
2. 若未传入，回退到 Phase 1 兼容模式。

伪代码：

```python
def heterogeneous_moe_forward(..., plan=None):
	 if plan is None:
		  plan = build_prefill_plan(...)
	 run_gpu_path(plan)
	 run_cpu_path(plan)
	 return output
```

好处：

1. Draft/Verify 在 ModelRunner 层决定策略，不把调度逻辑塞进算子层。
2. 降低重复计算与状态歧义。

## 6.6 ModelRunner 扩展

新增接口：

```python
def run_draft(self, seqs, input_ids, positions, draft_scheduler):
	 # decode-like，逐层 routing -> build_draft_plan -> heterogeneous_moe_forward(plan)
	 # 返回 logits + activations

def run_verify(self, seqs, input_ids, positions):
	 # prefill-like，routing -> build_prefill_plan（无替换）
	 # 返回 verify logits（按 seq 切分）
```

关键设计：

1. Draft 与 Verify 都走 eager。
2. Verify 使用完整路由，不调用替换逻辑。
3. 返回激活摘要（如每层 top experts）供预取器使用。

## 6.7 AcceptanceStrategy（最简）

接口：

```python
class AcceptanceStrategy(ABC):
	 def accept(self, draft_tokens, verify_logits, temperature) -> dict: ...
```

实现：

1. `GreedyAcceptance`：逐位对比 `draft_token` 与 `argmax(verify_logits[i])`。
2. `StandardAcceptance`：若 `P_verify(draft_token) >= threshold` 则接受，否则停止。

greedy 伪代码：

```python
num = 0
for i in range(len(draft_tokens)):
	 if draft_tokens[i] == argmax(verify_logits[i]):
		  num += 1
	 else:
		  break
next_token = argmax(verify_logits[num])
return {"num_accepted": num, "next_token": next_token}
```

## 6.8 SpeculativeEngine 主循环

关键职责：

1. 管理单轮 draft/verify/accept 的状态机。
2. 调用 scheduler 的 draft KV 接口。
3. 组织 ModelRunner 的 draft/verify 前向。

伪代码（简化）：

```python
def speculative_step(seqs):
	 for seq in seqs:
		  seq.start_draft()
		  scheduler.start_draft_kv(seq)

	 draft_tokens = _draft_loop(seqs)

	 for seq in seqs:
		  scheduler.rollback_draft_kv(seq)
		  seq.rollback_tokens_to_draft_start()

	 verify_logits = _run_verify(seqs, draft_tokens)

	 token_ids = []
	 for seq in seqs:
		  result = acceptance.accept(...)
		  scheduler.accept_draft_kv(seq, result.num_accepted)
		  seq.apply_accept_result(result)
		  token_ids.append(result.next_token)
	 return token_ids
```

约束：

1. 每轮必须确保 rollback 在 verify 之前执行。
2. 任何异常都要 fail-fast，避免 block_table 与 token_ids 不一致。
3. batch draft 先实现统一 draft 长度（所有序列同一轮同样的 max_draft_tokens 截止条件）。

---

## 7. 与 LLMEngine/Scheduler 的集成方式

## 7.1 LLMEngine.step 改造

逻辑：

1. prefill 不变。
2. decode 时按 `inference_mode` 分流：
	- standard：`model_runner.run(seqs, False)`
	- heter：`model_runner.run(seqs, False)`（异构 MoE 已在模型层注入）
	- spec：`spec_engine.speculative_step(seqs)`

伪代码：

```python
seqs, is_prefill = scheduler.schedule()
if is_prefill:
	 token_ids = model_runner.call("run", seqs, True)
elif config.inference_mode == "spec":
	 token_ids = spec_engine.speculative_step(seqs)
else:
	 token_ids = model_runner.call("run", seqs, False)
scheduler.postprocess(seqs, token_ids)
```

## 7.2 Scheduler 扩展点

新增包装方法，内部委派 `block_manager`：

1. `start_draft_kv(seq)`
2. `append_draft_kv(seq)`
3. `rollback_draft_kv(seq)`
4. `accept_draft_kv(seq, num_accepted)`

这样可以避免 SpeculativeEngine 直接操作 BlockManager 内部细节。

---

## 8. 正确性优先的实现细节约束

1. 顺序约束
	- 先 `seq.start_draft + start_draft_kv`，后 draft token 追加。
	- draft 结束必须 rollback（token 与 KV 双回滚）后再 verify。
2. 一致性约束
	- `len(seq.token_ids) == seq.num_tokens` 始终成立。
	- `seq.last_token == seq.token_ids[-1]` 始终成立。
	- `seq.num_blocks` 与 `block_table` 长度一致。
3. 边界约束
	- draft 提前遇到 EOS 时，仍进行 verify/accept 流程，避免状态分叉。
	- `num_accepted` 可为 0；此时仍需生成 verify next token。
4. 并发约束
	- Phase 2 不引入复杂 transfer 并发，先保证单 stream 逻辑正确。
5. 回退约束
	- 任一 speculative 关键路径报错时可配置降级到标准 decode（建议加开关）。
6. 空 CPU 路径约束(先不实现)
	- heter/spec 下若当前批次没有激活 expert 落在 CPU，CPU 路径必须直接跳过（不做占位计算）。
7. S=N 约束
	- 即使 S=N，也必须走 heter/spec 的完整规划与执行路径，不得特判绕过。

---

## 9. 测试方案（以正确性为主）

测试执行原则（强制）：

1. 先写测试，再实现对应功能。
2. 每完成一个实现步骤，立即运行受影响测试并记录结果。
3. 若新增实现导致旧测试失败，先修复再进入下一步。

## 9.1 单元测试

1. Sequence draft 状态
	- `start_draft -> append_draft_token -> rollback_tokens_to_draft_start`。
	- 检查 token_ids、num_tokens、last_token。

2. BlockManager draft 生命周期
	- 构造跨 block 边界 case，验证 `rollback_draft` 与 `accept_draft` 后 free/used/ref_count。

3. DraftScheduler simple
	- 给定固定 selected_experts/routing_weights，验证 top-c 选择确定性。

4. Placement draft/prefill
	- 验证替换映射后 gpu/cpu route 划分与 `m_sizes` 正确。

5. AcceptanceStrategy
	- greedy：逐位匹配中断位置正确。
	- standard：阈值比较正确。

6. mode 分发测试
	- 验证 standard/heter/spec 三种 mode 的入口分发正确。

7. S=N 路径一致性测试
	- 在 S=N 下验证 heter/spec 不触发特判分支（可通过 hook 统计关键函数调用次数）。

8. 空 CPU 路径测试（先不实现）
	- 构造全命中场景，断言 CPU fallback 路径不执行。

## 9.2 集成测试

1. 单序列 greedy 对齐
	- 关闭采样，speculative 与标准 decode 输出应完全一致。

2. 多序列 batch
	- 不同 prompt 长度同时 decode，确保每个 seq 的 draft/verify 状态互不污染。

3. 含 CPU fallback 场景
	- 人工降低 slots 触发 uncached expert，验证不会崩溃且输出合理。

4. 长序列稳定性
	- 多轮 draft-verify 后检查无内存泄漏、无 block_table 累积错误。

## 9.3 回归测试

1. `enable_speculative=False` 时，性能与行为不变。
2. `enable_heterogeneous=False` 且启用 speculative 时，明确报错或自动关闭（二选一，推荐报错）。

3. 三模式输出对齐
	- greedy 下 heter 与 spec 对齐 standard。
	- 采样下检查行为稳定（不要求 token 完全一致）。

## 9.4 指标与日志

建议落盘字段：

1. `draft_rounds`
2. `avg_draft_tokens`
3. `accept_rate`
4. `avg_accepted_tokens_per_round`
5. `verify_time_s`
6. `rollback_count`

补充对比字段：

1. `mode`
2. `match_rate_vs_standard`
3. `exact_match_vs_standard`
4. `cpu_path_executed_ratio`
5. `s_equals_n`

## 9.5 benchmark 方案（先于优化实现）

实现基本路径后立即补齐 benchmark，覆盖三模式：

1. Case A：standard
2. Case B：heter（S=N）
3. Case C：spec（S=N）

评测要求：

1. 统一输入、统一随机种子、统一采样参数。
2. 每个 case 至少 warmup 2 次，正式重复至少 5 次，报告 mean/std。
3. 输出文本与 token 结果保存，和 standard 对齐比较。
4. 明确记录是否出现 CPU 路径执行；若无 CPU 激活，相关步骤应被跳过。（先不实现）

---

## 10. 分阶段落地计划（Phase 2 内）

0. Step 0：先补齐测试骨架
	- 先写 Sequence/BlockManager/Mode 分发/Acceptance 的单元测试。

1. Step 1：数据结构与状态机
	- 改 `Sequence`、`BlockManager`、`Scheduler`。
   - 完成后立即运行对应单元测试。
2. Step 2：策略与计划构建
	- 新增 `DraftScheduler`、`AcceptanceStrategy`。
	- 扩展 `placement.py`。
   - 完成后运行 placement 与策略单测。
3. Step 3：ModelRunner 双路径
	- 实现 `run_draft`、`run_verify`。
   - 完成后运行集成 smoke 测试。
4. Step 4：SpeculativeEngine 与 LLMEngine 集成
	- 接入 `step()`。
   - 完成后运行三模式端到端回归。
5. Step 5：Benchmark 对比
	- 实现并运行三模式 benchmark（速度+正确性对齐）。
6. Step 6：优化章节中的增强实现
	- 最后实现 Ghost Block/软释放等优化项，并重新回归与 benchmark。

---

## 11. 风险清单与规避

1. 风险：KV 回滚错误导致后续 attention 污染。
	- 规避：增加严格断言与 block 引用计数检查测试。

2. 风险：draft token 回滚与 sequence 状态不同步。
	- 规避：所有 token 回滚集中在 Sequence 原语中处理，禁止外部直接切片。

3. 风险：替换映射引入越界或空缓存替换。
	- 规避：`select_gpu_substitutes` 必须校验 cached_experts 非空，否则强制走 CPU。

4. 风险：多序列下 verify logits 对齐错位。
	- 规避：verify 输入构造按 seq 记录 offset，accept 阶段按 offset 取 logits。

5. 风险：策略实现影响可复现性。
	- 规避：SimpleDraftScheduler 默认 deterministic（固定排序+固定 tie-break）。

6. 风险：S=N 场景被误加特判导致对比失真。
	- 规避：代码审查 + 单测断言关键路径调用。

7. 风险：CPU 路径无任务时仍有额外同步开销。
	- 规避：显式 fast skip（仅跳过空任务，不做 S=N 特判优化）。

---

## 12. 优化章节（本阶段最后实现步骤）

本节中的优化不进入基本正确性主路径，放在 Step 6 实现。

### 12.1 Ghost Block / 软释放（KV 管理优化）

目标：减少 Draft 回滚到 Verify 重放期间的 block 分配/释放开销。

设计：

1. Draft 回滚时，不立即将回滚出的 block 归还全局 free 队列。
2. 将其放入当前 Sequence 的 ghost 池（可复用池）。
3. Verify 阶段优先从 ghost 池复用 block 并覆盖写入。
4. 当 Sequence 结束或 ghost 池超过上限时，再真正 free。

核心接口建议：

1. `BlockManager.rollback_draft_to_ghost(seq)`
2. `BlockManager.allocate_for_verify_with_ghost(seq)`
3. `BlockManager.flush_ghost_blocks(seq)`

风险与约束：

1. ghost block 必须严格绑定序列，禁止跨序列复用。
2. preemption 或 sequence 迁移时必须先 flush。
3. 必须保证 hash/prefix cache 语义不被 ghost 机制破坏。

### 12.2 batch 内不同 draft 停止长度策略（设计，后续实现）

当前主路径：

1. batch 内所有序列统一 draft 轮次上限与停止条件。

优化设计：

1. 支持每个序列独立的 draft 停止条件：
   - 命中 EOS。
   - 接受率历史过低提前停止。
   - 达到序列级 max_draft_tokens。
2. batch 仍维持向量化执行，但在每步按活跃 mask 过滤。
3. verify 输入构造按序列长度对齐，避免 logits 错位。

---

## 13. 与 migration_design 4.6-4.10 的映射

1. 4.6 SpeculativeEngine：本方案完整覆盖，先实现最小稳定主循环。
2. 4.7 KV Cache Draft/Verify：本方案通过 BlockManager 原语精确落地。
3. 4.8 Sequence 扩展：本方案定义了最小必需字段与方法。
4. 4.9 ModelRunner 扩展：新增 run_draft/run_verify，并与现有异构 MoE 对接。
5. 4.10 调度策略接口：保留完整接口，仅落地 simple 版本，满足“先正确后优化”。

---

## 14. Phase 2 完成判定

满足以下条件即可判定 Phase 2 完成：

1. 三模式可执行并可复现实验。
2. 投机解码主链路可执行，支持 batch decode（先统一 draft 长度）。
3. greedy 下 heter/spec token 级对齐 standard。
4. draft/verify KV 生命周期测试全部通过。
5. 已完成三模式 benchmark（速度与正确性结果产出）。
6. 代码中策略接口可替换，且 simple 实现稳定。

以上设计保证以最小侵入方式在现有 Phase 1 基础上建立可靠的 Phase 2 基线，为后续预取优化、接受策略优化和性能调优提供稳定起点。
