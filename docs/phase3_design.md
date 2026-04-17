# Phase 3 详细设计：双阶段 Prefetch、Global Warm-start/History Queue 与高级优化路线

> 本文档是 Phase 3 的主设计文档，目标是在当前 `Phase 2 / phase2_post` 已有实现之上，补齐：
>
> 1. `draft` 阶段的异步 prefetch  
> 2. `prefill / verify` 观测驱动的 global warm-start/history queue  
> 3. `LayerExpertCache` 的 staging lifecycle  
> 4. 精确可解释的 profiling / benchmark 合约  
> 5. post-baseline 优化路径（Approach 2 / segmented graph / request-scoped queue / predictive verify-prefetch）
>
> 本文档风格与 [migration_design.md](./migration_design.md) 对齐，但粒度更细：  
> 对每个 feature 都给出实现目标、设计原则、完整算法、代码级接口、关键逻辑块、需要修改的文件以及实现步骤。

---

## 0. 设计原则

1. **以当前仓库实现为地面真相**：设计必须服从当前调用图、现有类接口、graph capture 边界、profile 字段与测试结构。
2. **verify 必须保持全精度**：Phase 3 只能改变 verify 命中 GPU cache 的概率，不能改变 verify 的路由、替换、acceptance 语义。
3. **baseline 先追求可实现与可验证**：第一轮实现只采用 `Approach 1 = staging memory + replay-boundary publish`。
4. **不删除已有设计面**：现有 Phase 3 文档里已经引入的 `global warm-start/history queue` 设计保留，不做概念删减，只把它写完整、写可实现。
5. **MoEExecutionPlan 保持紧凑**：`nanovllm/expert/placement.py` 里的 `MoEExecutionPlan` 继续只保存执行计划最小张量，不塞 prefetch runtime 状态。
6. **graph-safe 优先**：draft CUDA graph 主路径中不引入 host sync、不依赖 mid-replay Python callback、不在图内执行 CPU 规划。
7. **基线与优化阶段明确分离**：主文只写 baseline 必做设计；更激进的方案移入附录，并明确前置条件、收益假设与风险。
8. **profiling 必须回答问题**：必须能直接回答“是否提交、是否按时到达、是否被消费、是否值得”。
9. **实现按阶段推进**：先做可闭环版本，再做局部性优化和更细粒度 overlap。

---

## 1. 当前仓库事实与 Phase 3 约束

本节只陈述仓库中已经存在、Phase 3 不能违背的事实。

## 1.1 当前主链路

```text
LLMEngine
  -> SpeculativeEngine
    -> ModelRunner
      -> Qwen3MoeForCausalLM
        -> Qwen3MoeDecoderLayer
          -> self_attn
          -> mlp (Qwen3MoeHeterogeneousSparseMoeBlock)
```

spec 模式下：

1. `SpeculativeEngine.speculative_step()` 执行 `Draft -> Rollback -> Verify -> Accept`
2. `ModelRunner.run_draft()` 走 decode 路径，并在 block 内执行 `draft` plan
3. `ModelRunner.run_verify()` 走 prefill-like forward，重算 verify trace
4. 当前 `run_verify()` 是一次完整 `self.model(...)` 调用，`ModelRunner` 只有在整次 verify forward 结束后才重新获得控制权

## 1.2 关键模块事实

### 1.2.1 `nanovllm/expert/cache.py`

当前 `LayerExpertCache` 只有：

1. `num_slots`
2. `gate_up_buffer`
3. `down_buffer`
4. `slot_to_expert`
5. `expert_to_slot`
6. `expert_to_slot_lut`
7. `slot_to_expert_lut`
8. `cached_expert_mask`
9. 同步 `put_to_slot()`

当前 **没有**：

1. inflight / ready / published 生命周期
2. staging buffers
3. generation 保护
4. access 统计
5. victim 选择策略状态

### 1.2.2 `nanovllm/utils/heterogeneous_loader.py`

CPU expert pool 当前真实结构是：

```python
dict[layer_idx][expert_idx]["gate_up" | "down"]
```

其中：

1. `gate_up = torch.cat([gate_proj, up_proj], dim=0)`
2. `down = down_proj`
3. 不存在 `"gate_proj" / "up_proj" / "down_proj"` 三分字段接口

### 1.2.3 `nanovllm/scheduling/draft_scheduler.py`

当前只有 `SimpleDraftScheduler`，现有接口为：

1. `select_cpu_experts()`
2. `select_gpu_substitutes()`
3. `select_experts_to_transfer()`
4. `select_cpu_experts_gpu()`
5. `build_substitution_lut_gpu()`

Phase 3 不能直接删掉这些兼容接口。

### 1.2.4 `nanovllm/expert/placement.py`

当前 `MoEExecutionPlan` 包含：

1. `layer_idx`
2. `gpu_route_indices`
3. `gpu_m_sizes`
4. `cpu_route_indices`
5. `cpu_task_expert_ids`
6. `cpu_task_offsets`
7. `flat_selected_original`
8. `flat_selected_effective`
9. `substitution_lut`
10. `gpu_route_mask`
11. `cpu_route_mask`

Phase 3 不应把 runtime prefetch 统计塞进这个 dataclass。

### 1.2.5 `nanovllm/engine/model_runner.py`

当前关键事实：

1. `run_draft(self, seqs) -> tuple[list[int], list]`
2. `run_verify(self, seqs, verify_lengths) -> list[list[int]]`
3. `run_model()` 在 decode 下会选择：
   - eager
   - standard graph
   - draft graph
4. draft CUDA graph 只在 `draft_top_c == 0` 时启用
5. `run_draft()` 当前只在进入和退出 draft mode 时切换模式，中途不 regain control

### 1.2.6 `nanovllm/engine/speculative/spec_engine.py`

当前 spec 语义：

1. 只在 greedy / deterministic 路径上真正走 speculative
2. sampling (`temperature > 0`) 回退到普通 decode
3. verify 用 `run_verify()` 做全精度重算
4. verify 之前有一个清晰的 host-side 边界：`prepare_verify -> run_verify`

### 1.2.7 `nanovllm/models/qwen3_moe.py`

当前 `Qwen3MoeHeterogeneousSparseMoeBlock.forward()`：

1. 先算 `router_logits -> routing_weights -> selected_experts`
2. 根据 `execution_mode` 构建 `draft` 或 `verify` plan
3. 调用 `heterogeneous_moe_forward(...)`
4. 只保留 `_last_profile`
5. 没有导出 runtime metadata 的机制

### 1.2.8 `nanovllm/engine/llm_engine.py`

当前 phase2_post canonical alias 的统一出口在 `LLMEngine.get_profile()`，不是 `ModelRunner.get_profile()`。

现有 canonical 字段包括：

1. `route_ms`
2. `plan_ms`
3. `gpu_gather_ms`
4. `gpu_compute_ms`
5. `cpu_prepare_ms`
6. `cpu_compute_ms`
7. `cpu_to_gpu_merge_ms`
8. `scatter_ms`
9. `draft_ms`
10. `verify_ms`
11. `spec_step_ms`
12. `graph_hit_rate`
13. `graph_replay_count`
14. `cpu_route_ratio`
15. `cpu_weight_mass_ratio`
16. `activated_expert_set_size`
17. `realized_cpu_expert_count`

## 1.3 Phase 3 的额外需求

1. draft 第一轮 replay 结束前，CPU 无法从本轮 draft 获得 routing metadata，因此如果没有额外候选来源，第一轮 replay 期间 transfer 带宽会闲置。
2. expert activation 往往是偏态分布，同一 request 在 prefill / verify / draft 之间可能重复激活相同 expert。
3. 因此 baseline 就要保留并实现完整的 **global warm-start/history queue**：
   - 从 `prefill_history`
   - 从 `verify_history`
   - 从 `draft_live`
   三类来源持续积累候选。
4. baseline 先使用全局队列，request-scoped queue 放到 post-baseline 阶段，以便做对照实验。

---

## 2. Phase 3 范围划分

## 2.1 baseline（主文必须落地）

baseline 只实现以下内容：

1. `Approach 1`：staging memory + replay-boundary publish
2. 完整的 `global warm-start/history queue`
3. draft graph-safe metadata 导出
4. prefill / verify 观测驱动的 history 入队
5. draft replay 前 / 后的 best-effort transfer 发射
6. verify 前的 budgeted wait
7. verify 后的精确反馈入队
8. cache / prefetch strategy skeleton
9. observability 与 benchmark 指标

## 2.2 不进入 baseline 主线的优化设计（附录）

1. `Approach 2`：图内 per-layer GPU->CPU signaling
2. `Approach 3`：segmented graph
3. `Request-scoped warm-start/history queue`
4. `Predictive verify-prefetch`

---

## 3. 总体架构与 ownership

## 3.1 baseline ownership

baseline 中 ownership 固定如下：

1. `ModelRunner`
   - 持有 `PrefetchRuntime`
   - 持有 runtime meta recorder / host-pinned buffers
   - 持有 transfer stream / metadata stream
   - 负责在 prefill / draft / verify 边界调用 runtime
2. `Qwen3MoeHeterogeneousSparseMoeBlock`
   - 只负责导出 runtime metadata
   - 不直接发射 prefetch transfer
   - 不直接维护全局队列
3. `PrefetchRuntime`
   - 持有 global warm-start/history queue
   - 管理 inflight ticket、staging publish、dispatch 预算
4. `SpeculativeEngine`
   - 不做 queue 逻辑
   - 只在 `draft -> verify` 边界调用 `wait_prefetch_for_verify`

这样做的原因：

1. 避免把 prefetch 规划逻辑耦合进 model block
2. 避免 baseline 依赖 mid-forward callback
3. 避免让 verify “当前层 metadata -> 同一 verify 立即 transfer” 这种当前仓库无法承载的语义进入 baseline

## 3.2 baseline 端到端数据流

### 3.2.1 prefill

```text
ModelRunner.run(..., is_prefill=True)
  -> arm runtime-meta recorder(mode="prefill")
  -> self.model(...)
  -> collect runtime meta
  -> PrefetchRuntime.observe_prefill(...)
  -> global queue 更新 prefill_history 候选
```

### 3.2.2 draft（单次 replay）

```text
run_draft()
  -> PrefetchRuntime.publish_ready_before_draft()
  -> PrefetchRuntime.submit_from_global_queue(phase="before_draft")
  -> arm runtime-meta recorder(mode="draft")
  -> draft replay / eager draft
  -> offload runtime meta to host
  -> PrefetchRuntime.observe_draft(...)
  -> global queue 更新 draft_live 候选
  -> PrefetchRuntime.submit_from_global_queue(phase="after_draft")
```

注意：

1. baseline 中当前 replay 产生的 draft metadata 不要求在同一 replay 内生效
2. 但 replay 前先发射的 warm-start transfers 可以与 replay 计算重叠
3. replay 后新产生的 draft_live 候选用于下一轮 draft / 当前 verify 前的 transfer

### 3.2.3 verify

```text
SpeculativeEngine
  -> wait_prefetch_for_verify()
  -> run_verify()
      -> arm runtime-meta recorder(mode="verify")
      -> self.model(...)
      -> collect verify runtime meta
      -> PrefetchRuntime.observe_verify(...)
      -> global queue 更新 verify_history 候选
  -> accept
```

baseline 中 verify 的作用是：

1. verify 前可做 bounded wait，提高命中率
2. verify 后把精确 route 信息反馈给全局队列

baseline **不宣称**：

1. 当前 verify forward 内立刻 overlap transfer
2. 当前 verify 层内“边算边替换”

---

## 4. Feature 1：Config 扩展与顶层对象装配

## 4.1 实现目标

1. 为 baseline prefetch 加入明确的配置面。
2. 在 `ModelRunner.__init__()` 中装配 `PrefetchRuntime`、recorder、strategy。
3. 保持旧配置可继续工作，不要求用户必须打开 prefetch。

## 4.2 设计原则

1. 复用现有 `spec_enable_prefetch` 作为总开关。
2. 新字段都必须有保守默认值。
3. 关闭 prefetch 时，行为应退化为当前 Phase 2。

## 4.3 详细设计

### 4.3.1 `Config` 新增字段

在 [nanovllm/config.py](../nanovllm/config.py) 的 `Config` 中新增：

```python
@dataclass
class Config:
    # ... existing fields ...
    spec_enable_prefetch: bool = False

    # Cache / prefetch strategy
    cache_strategy: str = "lru"                     # "lru" | "lfu" | "adaptive"
    prefetch_strategy: str = "history_window"      # "noop" | "history_window"

    # Staging / dispatch budget
    prefetch_staging_slots_per_layer: int = 2
    prefetch_max_inflight: int = 8
    prefetch_step_budget: int = 4
    cache_eviction_budget_per_step: int = 2
    prefetch_verify_wait_ms: float = 0.0

    # Global queue behavior
    prefetch_global_queue_capacity: int = 4096
    prefetch_history_decay: float = 0.9
    prefetch_history_ttl_steps: int = 64
    prefetch_source_weight_prefill: float = 1.0
    prefetch_source_weight_verify: float = 1.2
    prefetch_source_weight_draft: float = 1.5
    prefetch_activation_count_weight: float = 0.1
    prefetch_age_penalty: float = 0.02

    # Source gates
    prefetch_use_prefill_history: bool = True
    prefetch_use_verify_history: bool = True
    prefetch_use_draft_live: bool = True

    # Runtime integration
    prefetch_runtime_mode: str = "baseline_staging"   # baseline_staging only in mainline
```

### 4.3.2 `Config.__post_init__()` 校验

新增校验：

```python
def __post_init__(self):
    # ... existing checks ...
    assert self.cache_strategy in {"lru", "lfu", "adaptive"}
    assert self.prefetch_strategy in {"noop", "history_window"}
    assert self.prefetch_staging_slots_per_layer >= 0
    assert self.prefetch_max_inflight >= 0
    assert self.prefetch_step_budget >= 0
    assert self.cache_eviction_budget_per_step >= 0
    assert self.prefetch_verify_wait_ms >= 0.0
    assert self.prefetch_global_queue_capacity >= 0
    assert 0.0 <= self.prefetch_history_decay <= 1.0
    assert self.prefetch_history_ttl_steps >= 1
    assert self.prefetch_runtime_mode == "baseline_staging"
```

### 4.3.3 `ModelRunner.__init__()` 装配对象

新增顶层装配：

```python
class ModelRunner:
    def __init__(self, config: Config, rank: int, events):
        # ... existing init ...
        self.cache_strategy = create_cache_strategy(config.cache_strategy)
        self.prefetch_strategy = create_prefetch_strategy(config.prefetch_strategy, config)
        self.runtime_meta_recorder = None
        self.prefetch_runtime = None
        self._prefetch_step_id = 0

        if config.enable_heterogeneous and config.spec_enable_prefetch:
            self.runtime_meta_recorder = ModelRuntimeMetaRecorder(
                config=config,
                hf_config=config.hf_config,
            )
            self.prefetch_runtime = PrefetchRuntime(
                config=config,
                layer_caches=self.layer_caches,
                cpu_expert_pool=self.cpu_expert_pool,
                cache_strategy=self.cache_strategy,
                prefetch_strategy=self.prefetch_strategy,
                runtime_meta_recorder=self.runtime_meta_recorder,
            )
            if hasattr(self.model, "set_runtime_meta_recorder"):
                self.model.set_runtime_meta_recorder(self.runtime_meta_recorder)
```

### 4.3.4 `_next_prefetch_step_id()`

新增辅助方法：

```python
class ModelRunner:
    def _next_prefetch_step_id(self) -> int:
        self._prefetch_step_id += 1
        return self._prefetch_step_id
```

## 4.4 需要修改的文件

1. `nanovllm/config.py`
2. `nanovllm/engine/model_runner.py`
3. `nanovllm/scheduling/cache_strategy.py`（新增）
4. `nanovllm/scheduling/prefetch_strategy.py`（新增）
5. `nanovllm/expert/prefetcher.py`（新增）
6. `tests/test_config_prefetch.py`（新增）

## 4.5 实现步骤

1. 先加配置字段与校验。
2. 再实现 `create_cache_strategy()` / `create_prefetch_strategy()` skeleton。
3. 最后在 `ModelRunner.__init__()` 中装配 runtime，但先不接入具体调度逻辑。

---

## 5. Feature 2：Runtime Metadata 模型与 graph-safe 导出

## 5.1 实现目标

1. 在 prefill / draft / verify 三条路径上统一导出“最小必要 routing metadata”。
2. draft graph 路径必须 graph-safe。
3. metadata 必须足够驱动：
   - 全局队列更新
   - draft_live 候选生成
   - prefill/verify 历史积累

## 5.2 设计原则

1. baseline 不导出执行结果张量，只导出 routing metadata。
2. 只导出 Phase 3 真实需要的最小字段。
3. draft graph 路径中不做动态分配、不做 Python list append。

## 5.3 详细设计

### 5.3.1 CPU 侧元数据结构

新增 `nanovllm/expert/runtime_meta.py`：

```python
from dataclasses import dataclass

@dataclass
class LayerRuntimeMetaCPU:
    step_id: int
    mode: str                                  # "prefill" | "draft" | "verify"
    layer_idx: int
    token_count: int
    selected_experts: torch.Tensor             # CPU int64, shape [T, K]
    routing_weights: torch.Tensor              # CPU float32, shape [T, K]


@dataclass
class RuntimeMetaOffloadHandle:
    step_id: int
    mode: str
    event: torch.cuda.Event
    token_capacity: int
    logical_token_count: int
```

字段选择说明：

1. `selected_experts`：决定“哪些 expert 被激活”
2. `routing_weights`：决定历史 priority / score_sum
3. `token_count`：告诉 CPU 本层本次有多少有效 token 行

baseline **不需要**额外导出：

1. executed expert outputs
2. per-token hidden states
3. `gpu_route_mask` / `cpu_route_mask`
4. `flat_selected_effective`

因为全局 warm-start/history queue 只需要“原始 expert 激活及权重”。

### 5.3.2 recorder 结构

```python
class ModelRuntimeMetaRecorder:
    def __init__(self, config: Config, hf_config):
        self.config = config
        self.num_layers = hf_config.num_hidden_layers
        self.top_k = hf_config.num_experts_per_tok
        self.device_buffers = {}     # key=(mode, bucket_or_capacity)
        self.host_buffers = {}       # key=(mode, bucket_or_capacity)
        self.active_key = None
        self.active_step_id = -1
        self.active_mode = "idle"
        self.active_logical_token_count = 0

    def arm(
        self,
        mode: str,
        step_id: int,
        token_capacity: int,
        logical_token_count: int | None = None,
    ) -> None: ...
    def record_layer(
        self,
        layer_idx: int,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> None: ...
    def offload_async(
        self,
        stream: torch.cuda.Stream,
    ) -> RuntimeMetaOffloadHandle | None: ...
    def collect(
        self,
        handle: RuntimeMetaOffloadHandle,
        wait: bool = False,
    ) -> dict[int, LayerRuntimeMetaCPU] | None: ...
    def reset(self) -> None: ...
```

`arm()` 的语义：

1. `token_capacity`：本次 recorder buffer 的容量
2. `logical_token_count`：本次真实有效 token 数；若为 `None`，默认等于 `token_capacity`

### 5.3.3 设备侧 buffer 形状

对每个 `(mode, token_capacity)` 预分配：

```python
selected_experts_device: torch.Tensor   # [num_layers, token_capacity, top_k], int64
routing_weights_device: torch.Tensor    # [num_layers, token_capacity, top_k], float32
token_count_device: torch.Tensor        # [num_layers], int32
```

其中：

1. `token_capacity` 是 buffer / graph bucket 容量
2. `logical_token_count` 是本次真实有效 token 数
3. draft graph replay 时，这两个值可能不同；collector 在 `collect()` 时必须用 `min(recorded_token_count, logical_token_count)` 截断掉 bucket padding

对应 host-pinned mirror：

```python
selected_experts_host: torch.Tensor
routing_weights_host: torch.Tensor
token_count_host: torch.Tensor
```

### 5.3.4 `record_layer()` 算法

`Qwen3MoeHeterogeneousSparseMoeBlock.forward()` 在完成 `topk` 后立刻记录：

```python
def record_layer(self, layer_idx, selected_experts, routing_weights):
    # selected_experts: [T, K]
    # routing_weights: [T, K]
    if self.active_key is None:
        return

    token_count = int(selected_experts.size(0))
    dev = self.device_buffers[self.active_key]

    # graph-safe：只做 preallocated tensor 的切片 copy
    dev["token_count"][layer_idx] = token_count
    dev["selected_experts"][layer_idx, :token_count].copy_(
        selected_experts.to(torch.int64),
        non_blocking=True,
    )
    dev["routing_weights"][layer_idx, :token_count].copy_(
        routing_weights.float(),
        non_blocking=True,
    )
```

为什么在 `topk` 后记录：

1. 对 prefetch 来说，最关键的是“原始路由选择”
2. baseline 不依赖 plan 后的 gpu/cpu 分裂信息
3. 记录点足够早，且不会改变 verify 语义

### 5.3.5 `offload_async()` 算法

```python
def offload_async(self, stream):
    if self.active_key is None:
        return None

    dev = self.device_buffers[self.active_key]
    host = self.host_buffers[self.active_key]
    handle = RuntimeMetaOffloadHandle(
        step_id=self.active_step_id,
        mode=self.active_mode,
        event=torch.cuda.Event(blocking=False),
        token_capacity=host["selected_experts"].size(1),
        logical_token_count=self.active_logical_token_count,
    )

    with torch.cuda.stream(stream):
        host["token_count"].copy_(dev["token_count"], non_blocking=True)
        host["selected_experts"].copy_(dev["selected_experts"], non_blocking=True)
        host["routing_weights"].copy_(dev["routing_weights"], non_blocking=True)
        handle.event.record(stream)
    return handle
```

### 5.3.6 `collect()` 算法

```python
def collect(self, handle, wait=False):
    if handle is None:
        return None
    if not wait and not handle.event.query():
        return None
    if wait:
        handle.event.synchronize()

    host = self.host_buffers[(handle.mode, handle.token_capacity)]
    out = {}
    token_counts = host["token_count"]
    for layer_idx in range(self.num_layers):
        token_count = int(token_counts[layer_idx].item())
        token_count = min(token_count, handle.logical_token_count)
        if token_count <= 0:
            continue
        out[layer_idx] = LayerRuntimeMetaCPU(
            step_id=handle.step_id,
            mode=handle.mode,
            layer_idx=layer_idx,
            token_count=token_count,
            selected_experts=host["selected_experts"][layer_idx, :token_count].clone(),
            routing_weights=host["routing_weights"][layer_idx, :token_count].clone(),
        )
    return out
```

### 5.3.7 `Qwen3MoeHeterogeneousSparseMoeBlock` 修改

新增字段与 setter：

```python
class Qwen3MoeHeterogeneousSparseMoeBlock(nn.Module):
    def __init__(...):
        # ... existing fields ...
        self.runtime_meta_recorder: ModelRuntimeMetaRecorder | None = None

    def set_runtime_meta_recorder(
        self,
        recorder: ModelRuntimeMetaRecorder | None,
    ) -> None:
        self.runtime_meta_recorder = recorder
```

在 `forward()` 中加入：

```python
def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    # ... router -> topk ...
    if self.runtime_meta_recorder is not None:
        # 这里只记录原始路由，不记录执行结果
        self.runtime_meta_recorder.record_layer(
            layer_idx=self.layer_idx,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
        )
    # ... build plan + heterogeneous_moe_forward ...
```

### 5.3.8 `Qwen3MoeForCausalLM` 修改

新增 model 级 setter：

```python
class Qwen3MoeForCausalLM(nn.Module):
    def set_runtime_meta_recorder(
        self,
        recorder: ModelRuntimeMetaRecorder | None,
    ) -> None:
        for layer in self.model.layers:
            if isinstance(layer.mlp, Qwen3MoeHeterogeneousSparseMoeBlock):
                layer.mlp.set_runtime_meta_recorder(recorder)
```

### 5.3.9 `capture_draft_cudagraph()` 集成

`ModelRunner.capture_draft_cudagraph()` 在 capture 前需要为每个 draft bucket 预热 recorder：

```python
def capture_draft_cudagraph(self):
    # ... existing logic ...
    for bs in reversed(self.draft_graph_bs):
        if self.runtime_meta_recorder is not None:
            self.runtime_meta_recorder.arm(
                mode="draft",
                step_id=-1,         # capture 阶段无真实 step_id
                token_capacity=bs,
                logical_token_count=bs,
            )
        # warmup
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
        with torch.cuda.graph(graph, self.draft_graph_pool):
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
```

`record_layer()` 在 capture 阶段只会把 tensor op capture 进去，不触发 host planning。

## 5.4 需要修改的文件

1. `nanovllm/expert/runtime_meta.py`（新增）
2. `nanovllm/models/qwen3_moe.py`
3. `nanovllm/engine/model_runner.py`
4. `tests/test_prefetch_runtime_meta.py`（新增）
5. `tests/test_draft_cuda_graph.py`
6. `tests/test_draft_cuda_graph_real_world.py`

## 5.5 实现步骤

1. 先实现 eager path 的 recorder。
2. 再接入 prefill / verify。
3. 最后接入 draft graph bucket 的 arm / capture / offload。

---

## 6. Feature 3：LayerExpertCache staging lifecycle

## 6.1 实现目标

1. 给当前 `LayerExpertCache` 增加 staging memory 与 inflight lifecycle。
2. 保持 active cache 在 replay 内不可变。
3. 只在 replay 边界 publish。

## 6.2 设计原则

1. baseline 采用 **separate staging memory**，不引入 hidden active slots。
2. 兼容现有 `put_to_slot()` 和当前加载路径。
3. publish 必须有 generation 保护。

## 6.3 详细设计

### 6.3.1 `LayerExpertCache.__init__()` 扩展

修改 [nanovllm/expert/cache.py](../nanovllm/expert/cache.py)：

```python
class LayerExpertCache:
    def __init__(
        self,
        num_experts: int,
        slots_per_layer: int,
        gate_up_shape: tuple[int, int],
        down_shape: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
        cpu_expert_pool: dict[int, dict[str, torch.Tensor]] | None = None,
        staging_slots_per_layer: int = 0,
        enable_prefetch: bool = False,
    ) -> None:
        # existing active buffers
        self.num_experts = num_experts
        self.num_slots = max(1, min(slots_per_layer, self.num_experts))
        self.cpu_expert_pool = cpu_expert_pool or {}

        self.gate_up_buffer = ...
        self.down_buffer = ...
        self.slot_to_expert = ...
        self.expert_to_slot = ...
        self.expert_to_slot_lut = ...
        self.slot_to_expert_lut = ...
        self.cached_expert_mask = ...

        # new access stats (CPU-side book-keeping)
        self.last_access_step = [-1] * self.num_experts
        self.access_count = [0] * self.num_experts
        self.access_score_sum = [0.0] * self.num_experts

        # new generation state for active slots
        self.slot_generation = [0] * self.num_slots

        # staging
        self.enable_prefetch = bool(enable_prefetch)
        self.num_staging_slots = int(staging_slots_per_layer) if enable_prefetch else 0
        if self.num_staging_slots > 0:
            self.staging_gate_up_buffer = torch.empty(...)
            self.staging_down_buffer = torch.empty(...)
        else:
            self.staging_gate_up_buffer = None
            self.staging_down_buffer = None

        self.staging_slot_state = [0] * self.num_staging_slots         # 0=free,1=inflight,2=ready
        self.staging_slot_to_expert = [-1] * self.num_staging_slots
        self.staging_slot_generation = [0] * self.num_staging_slots
```

### 6.3.2 新增辅助 dataclass

同文件新增：

```python
@dataclass
class StagingReservation:
    layer_idx: int
    staging_slot_idx: int
    expert_idx: int
    generation: int


@dataclass
class PublishedExpert:
    layer_idx: int
    expert_idx: int
    active_slot_idx: int
    staging_slot_idx: int
    generation: int
```

### 6.3.3 `mark_access()`

```python
def mark_access(
    self,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor | None,
    step_id: int,
) -> None:
    unique_ids, inverse = torch.unique(expert_ids.reshape(-1).to(torch.int64), return_inverse=True)
    score_sum = None
    if routing_weights is not None:
        flat_weights = routing_weights.reshape(-1).float()
        score_sum = torch.zeros((unique_ids.numel(),), dtype=torch.float32, device=flat_weights.device)
        score_sum.scatter_add_(0, inverse, flat_weights)

    for i, expert_idx in enumerate(unique_ids.tolist()):
        self.last_access_step[expert_idx] = step_id
        self.access_count[expert_idx] += 1
        if score_sum is not None:
            self.access_score_sum[expert_idx] += float(score_sum[i].item())
```

### 6.3.4 `snapshot()`

```python
@dataclass
class LayerCacheSnapshot:
    layer_idx: int
    cached_expert_mask: torch.Tensor
    expert_to_slot_lut: torch.Tensor
    slot_to_expert_lut: torch.Tensor
    last_access_step: list[int]
    access_count: list[int]
    access_score_sum: list[float]
    slot_generation: list[int]


def snapshot(self, layer_idx: int) -> LayerCacheSnapshot:
    return LayerCacheSnapshot(
        layer_idx=layer_idx,
        cached_expert_mask=self.cached_expert_mask.clone(),
        expert_to_slot_lut=self.expert_to_slot_lut.clone(),
        slot_to_expert_lut=self.slot_to_expert_lut.clone(),
        last_access_step=list(self.last_access_step),
        access_count=list(self.access_count),
        access_score_sum=list(self.access_score_sum),
        slot_generation=list(self.slot_generation),
    )
```

### 6.3.5 `reserve_staging_slot()`

```python
def reserve_staging_slot(
    self,
    expert_idx: int,
) -> StagingReservation | None:
    if self.num_staging_slots <= 0:
        return None
    for slot_idx in range(self.num_staging_slots):
        if self.staging_slot_state[slot_idx] == 0:
            self.staging_slot_state[slot_idx] = 1
            self.staging_slot_to_expert[slot_idx] = expert_idx
            self.staging_slot_generation[slot_idx] += 1
            return StagingReservation(
                layer_idx=-1,   # caller fill
                staging_slot_idx=slot_idx,
                expert_idx=expert_idx,
                generation=self.staging_slot_generation[slot_idx],
            )
    return None
```

### 6.3.6 `begin_async_put_to_staging()`

```python
def begin_async_put_to_staging(
    self,
    reservation: StagingReservation,
    gate_up_cpu: torch.Tensor,
    down_cpu: torch.Tensor,
    stream: torch.cuda.Stream,
) -> torch.cuda.Event:
    idx = reservation.staging_slot_idx
    with torch.cuda.stream(stream):
        self.staging_gate_up_buffer[idx].copy_(gate_up_cpu, non_blocking=True)
        self.staging_down_buffer[idx].copy_(down_cpu, non_blocking=True)
        event = torch.cuda.Event(blocking=False)
        event.record(stream)
    return event
```

### 6.3.7 `mark_staging_ready()`

```python
def mark_staging_ready(self, reservation: StagingReservation) -> bool:
    idx = reservation.staging_slot_idx
    if self.staging_slot_state[idx] != 1:
        return False
    if self.staging_slot_generation[idx] != reservation.generation:
        return False
    self.staging_slot_state[idx] = 2
    return True
```

### 6.3.8 `publish_ready_staging_to_active()`

```python
def publish_ready_staging_to_active(
    self,
    reservation: StagingReservation,
    active_slot_idx: int,
    stream: torch.cuda.Stream,
) -> PublishedExpert | None:
    s = reservation.staging_slot_idx
    if self.staging_slot_state[s] != 2:
        return None
    if self.staging_slot_generation[s] != reservation.generation:
        return None

    with torch.cuda.stream(stream):
        self.gate_up_buffer[active_slot_idx].copy_(self.staging_gate_up_buffer[s], non_blocking=True)
        self.down_buffer[active_slot_idx].copy_(self.staging_down_buffer[s], non_blocking=True)

    return PublishedExpert(
        layer_idx=reservation.layer_idx,
        expert_idx=reservation.expert_idx,
        active_slot_idx=active_slot_idx,
        staging_slot_idx=s,
        generation=reservation.generation,
    )
```

### 6.3.9 `commit_published_expert()`

`commit_published_expert()` 必须只在 publish copy 对 active slot 已经完成后调用：

```python
def commit_published_expert(
    self,
    published: PublishedExpert,
) -> None:
    slot_idx = published.active_slot_idx
    expert_idx = published.expert_idx

    prev_expert = self.slot_to_expert[slot_idx]
    if prev_expert >= 0 and prev_expert in self.expert_to_slot:
        del self.expert_to_slot[prev_expert]
        self.expert_to_slot_lut[prev_expert] = -1
        self.cached_expert_mask[prev_expert] = False

    self.slot_to_expert[slot_idx] = expert_idx
    self.slot_to_expert_lut[slot_idx] = expert_idx
    self.expert_to_slot[expert_idx] = slot_idx
    self.expert_to_slot_lut[expert_idx] = slot_idx
    self.cached_expert_mask[expert_idx] = True
    self.slot_generation[slot_idx] += 1

    # release staging slot
    s = published.staging_slot_idx
    self.staging_slot_state[s] = 0
    self.staging_slot_to_expert[s] = -1
```

### 6.3.10 兼容 `put_to_slot()`

现有 `put_to_slot()` 不删除，改为兼容 wrapper：

```python
def put_to_slot(self, slot_idx, expert_idx, gate_up_cpu, down_cpu):
    # 仍保留给：
    # 1) 初始 placement
    # 2) prefetch 关闭时的直接加载
    prev_expert = self.slot_to_expert[slot_idx]
    # ... existing logic ...
```

## 6.4 publish 的精确时机

baseline 中 publish 只允许在以下 host-side 边界调用：

1. `run_draft()` 开始前
2. `wait_prefetch_for_verify()` 内
3. `run_verify()` 结束后、下一次 draft 前

baseline 不允许：

1. 在 draft replay 中途 publish active slots
2. 在 verify monolithic forward 中途 publish active slots

## 6.5 需要修改的文件

1. `nanovllm/expert/cache.py`
2. `nanovllm/utils/heterogeneous_loader.py`
3. `nanovllm/expert/prefetcher.py`
4. `tests/test_expert_cache_staging.py`（新增）
5. `tests/test_expert_cache_generation.py`（新增）

## 6.6 实现步骤

1. 先给 cache 加 access 统计与 snapshot。
2. 再加 staging buffers 与 reservation lifecycle。
3. 最后接 publish 与 generation 保护。

---

## 7. Feature 4：Global Warm-start/History Queue

## 7.1 实现目标

1. 保留并完整实现现有文档中的 global warm-start/history queue 设计。
2. 在第一轮 draft replay 之前，就能根据 prefill / verify 历史发射 transfer。
3. 在 baseline 中使用单个全局队列，后续再把 request-scoped 版本放到优化阶段。

## 7.2 设计原则

1. baseline 不做 request-scoped 隔离。
2. 但来源必须区分：
   - `prefill_history`
   - `verify_history`
   - `draft_live`
3. 全局队列不是简单 FIFO，而是基于 score / activation / recency 的可重排候选池。
4. 队列项必须可去重、可衰减、可淘汰。

## 7.3 详细设计

### 7.3.1 队列项定义

新增 `nanovllm/expert/prefetcher.py`：

```python
@dataclass
class PrefetchCandidate:
    layer_idx: int
    expert_idx: int
    source: str                      # "prefill_history" | "verify_history" | "draft_live"
    score_sum: float
    activation_count: int
    first_seen_step: int
    last_seen_step: int
    priority: float
```

### 7.3.2 队列内部结构

```python
class GlobalWarmStartQueue:
    def __init__(self, config: Config):
        self.config = config
        self.entries: dict[tuple[int, int], PrefetchCandidate] = {}
```

key 为 `(layer_idx, expert_idx)`。  
这意味着同一 expert 的不同来源观测会合并到同一个条目中，而不是重复排队。

### 7.3.3 priority 公式

```python
def compute_priority(
    source: str,
    score_sum: float,
    activation_count: int,
    age: int,
    config: Config,
) -> float:
    source_weight = {
        "prefill_history": config.prefetch_source_weight_prefill,
        "verify_history": config.prefetch_source_weight_verify,
        "draft_live": config.prefetch_source_weight_draft,
    }[source]

    return (
        source_weight * score_sum
        + config.prefetch_activation_count_weight * activation_count
        - config.prefetch_age_penalty * age
    )
```

### 7.3.4 从一批 runtime metadata 更新全局队列

```python
def update_from_runtime_meta(
    self,
    runtime_meta: dict[int, LayerRuntimeMetaCPU],
    source: str,
    step_id: int,
    layer_caches: dict[int, LayerExpertCache],
) -> None:
    if runtime_meta is None:
        return

    for layer_idx, meta in runtime_meta.items():
        cache = layer_caches[layer_idx]
        flat_experts = meta.selected_experts.reshape(-1).to(torch.int64)
        flat_weights = meta.routing_weights.reshape(-1).float()

        unique_ids, inverse = torch.unique(flat_experts, return_inverse=True)
        score_sum = torch.zeros((unique_ids.numel(),), dtype=torch.float32)
        score_sum.scatter_add_(0, inverse.cpu(), flat_weights.cpu())
        counts = torch.zeros((unique_ids.numel(),), dtype=torch.int64)
        counts.scatter_add_(0, inverse.cpu(), torch.ones_like(inverse.cpu(), dtype=torch.int64))

        cached_mask = cache.get_cached_expert_mask().cpu().index_select(0, unique_ids.cpu())
        for i, expert_idx in enumerate(unique_ids.tolist()):
            if bool(cached_mask[i]):
                # 已在 active cache，不进入队列
                continue

            key = (layer_idx, expert_idx)
            new_score = float(score_sum[i].item())
            new_count = int(counts[i].item())

            if key in self.entries:
                entry = self.entries[key]
                entry.score_sum = self.config.prefetch_history_decay * entry.score_sum + new_score
                entry.activation_count = int(
                    round(self.config.prefetch_history_decay * entry.activation_count)
                ) + new_count
                entry.last_seen_step = step_id
                # 使用最近一次来源覆盖 source，便于解释当前优先级
                entry.source = source
            else:
                entry = PrefetchCandidate(
                    layer_idx=layer_idx,
                    expert_idx=expert_idx,
                    source=source,
                    score_sum=new_score,
                    activation_count=new_count,
                    first_seen_step=step_id,
                    last_seen_step=step_id,
                    priority=0.0,
                )
                self.entries[key] = entry

            age = max(0, step_id - entry.last_seen_step)
            entry.priority = compute_priority(
                source=entry.source,
                score_sum=entry.score_sum,
                activation_count=entry.activation_count,
                age=age,
                config=self.config,
            )
```

### 7.3.5 老化与清理

```python
def prune(
    self,
    step_id: int,
    layer_caches: dict[int, LayerExpertCache],
) -> None:
    stale_keys = []
    for key, entry in self.entries.items():
        layer_idx, expert_idx = key
        cache = layer_caches[layer_idx]
        if step_id - entry.last_seen_step > self.config.prefetch_history_ttl_steps:
            stale_keys.append(key)
            continue
        if bool(cache.get_cached_expert_mask()[expert_idx].item()):
            stale_keys.append(key)
            continue
    for key in stale_keys:
        self.entries.pop(key, None)

    if len(self.entries) > self.config.prefetch_global_queue_capacity:
        ranked = sorted(self.entries.items(), key=lambda kv: kv[1].priority, reverse=True)
        self.entries = dict(ranked[: self.config.prefetch_global_queue_capacity])
```

### 7.3.6 取出 dispatch 候选

```python
def ranked_candidates(
    self,
    step_id: int,
    layer_caches: dict[int, LayerExpertCache],
    inflight_keys: set[tuple[int, int]],
) -> list[PrefetchCandidate]:
    self.prune(step_id, layer_caches)
    ranked = []
    for key, entry in self.entries.items():
        layer_idx, expert_idx = key
        if key in inflight_keys:
            continue
        if bool(layer_caches[layer_idx].get_cached_expert_mask()[expert_idx].item()):
            continue
        age = max(0, step_id - entry.last_seen_step)
        entry.priority = compute_priority(
            source=entry.source,
            score_sum=entry.score_sum,
            activation_count=entry.activation_count,
            age=age,
            config=self.config,
        )
        ranked.append(entry)
    ranked.sort(key=lambda x: (-x.priority, x.layer_idx, x.expert_idx))
    return ranked
```

### 7.3.7 为什么 baseline 保留 global queue

这里显式保留，不删不弱化：

1. 它是第一轮 draft replay 前能利用 prefill / verify 历史的唯一 baseline 机制
2. 它提供后续 request-scoped queue 的对照组
3. 它让 Phase 3 第一版不需要先处理 request 局部性与跨 request 公平性问题

## 7.4 需要修改的文件

1. `nanovllm/expert/prefetcher.py`
2. `nanovllm/engine/model_runner.py`
3. `nanovllm/engine/speculative/spec_engine.py`
4. `tests/test_prefetch_global_queue.py`（新增）

## 7.5 实现步骤

1. 先实现 `PrefetchCandidate` / `GlobalWarmStartQueue`。
2. 再接入 `prefill_history` 与 `verify_history`。
3. 最后接入 `draft_live` 并完成 priority 排序。

---

## 8. Feature 5：PrefetchRuntime 与完整调度算法

## 8.1 实现目标

1. 把 queue 管理、transfer、publish、wait 全部集中在一个 runtime。
2. 让 baseline 在没有 mid-replay Python control 的情况下仍能 overlap：
   - replay 前先发射一批 transfer
   - replay 期间 transfer 在独立 stream 上继续
3. 对 draft / verify / prefill 三类观察统一入队。

## 8.2 设计原则

1. baseline 不依赖后台线程；所有 runtime 动作都由 `ModelRunner` 在边界驱动。
2. `submit` 永远 non-blocking。
3. `wait` 只在 verify 前允许，且受 budget 控制。
4. `publish` 只在 replay 边界执行。

## 8.3 详细设计

### 8.3.1 ticket 定义

```python
@dataclass
class PrefetchTicket:
    step_id: int
    layer_idx: int
    expert_idx: int
    source: str
    staging_slot_idx: int
    staging_generation: int
    submit_ts_ms: float
    ready_event: torch.cuda.Event
```

### 8.3.2 `PrefetchRuntime` 字段

```python
class PrefetchRuntime:
    def __init__(
        self,
        config: Config,
        layer_caches: dict[int, LayerExpertCache],
        cpu_expert_pool: dict[int, dict[int, dict[str, torch.Tensor]]],
        cache_strategy: CacheStrategy,
        prefetch_strategy: PrefetchStrategy,
        runtime_meta_recorder: ModelRuntimeMetaRecorder,
    ):
        self.config = config
        self.layer_caches = layer_caches
        self.cpu_expert_pool = cpu_expert_pool
        self.cache_strategy = cache_strategy
        self.prefetch_strategy = prefetch_strategy
        self.runtime_meta_recorder = runtime_meta_recorder

        self.global_queue = GlobalWarmStartQueue(config)
        self.transfer_stream = torch.cuda.Stream()
        self.metadata_stream = torch.cuda.Stream()
        self.publish_stream = torch.cuda.Stream()

        self.inflight: dict[tuple[int, int], PrefetchTicket] = {}
        self._profile = defaultdict(float)
```

### 8.3.3 从 runtime metadata 生成 / 更新候选

```python
def observe_runtime_meta(
    self,
    runtime_meta: dict[int, LayerRuntimeMetaCPU] | None,
    source: str,
    step_id: int,
) -> None:
    if runtime_meta is None:
        return
    self.global_queue.update_from_runtime_meta(
        runtime_meta=runtime_meta,
        source=source,
        step_id=step_id,
        layer_caches=self.layer_caches,
    )
```

### 8.3.4 dispatch 主算法

```python
def submit_from_global_queue(
    self,
    step_id: int,
    phase: str,   # "before_draft" | "after_draft" | "after_verify"
) -> int:
    if self.config.prefetch_step_budget <= 0:
        return 0

    inflight_keys = set(self.inflight.keys())
    ranked = self.global_queue.ranked_candidates(
        step_id=step_id,
        layer_caches=self.layer_caches,
        inflight_keys=inflight_keys,
    )

    submitted = 0
    max_submit = max(0, self.config.prefetch_step_budget)
    inflight_budget = max(0, self.config.prefetch_max_inflight - len(self.inflight))
    dispatch_budget = min(max_submit, inflight_budget)

    for candidate in ranked:
        if submitted >= dispatch_budget:
            break

        layer_idx = candidate.layer_idx
        expert_idx = candidate.expert_idx
        key = (layer_idx, expert_idx)
        cache = self.layer_caches[layer_idx]

        if bool(cache.get_cached_expert_mask()[expert_idx].item()):
            continue
        if key in self.inflight:
            continue

        reservation = cache.reserve_staging_slot(expert_idx)
        if reservation is None:
            continue
        reservation.layer_idx = layer_idx

        weights = self.cpu_expert_pool[layer_idx].get(expert_idx)
        if weights is None:
            continue

        ready_event = cache.begin_async_put_to_staging(
            reservation=reservation,
            gate_up_cpu=weights["gate_up"],
            down_cpu=weights["down"],
            stream=self.transfer_stream,
        )

        ticket = PrefetchTicket(
            step_id=step_id,
            layer_idx=layer_idx,
            expert_idx=expert_idx,
            source=candidate.source,
            staging_slot_idx=reservation.staging_slot_idx,
            staging_generation=reservation.generation,
            submit_ts_ms=time.perf_counter() * 1000.0,
            ready_event=ready_event,
        )
        self.inflight[key] = ticket
        submitted += 1

        self._profile["prefetch_submit_count"] += 1
        if candidate.source == "prefill_history":
            self._profile["history_prefetch_submit_count"] += 1
        elif candidate.source == "verify_history":
            self._profile["verify_history_prefetch_submit_count"] += 1
        elif candidate.source == "draft_live":
            self._profile["draft_live_prefetch_submit_count"] += 1

    return submitted
```

### 8.3.5 polling ready ticket

```python
def poll_ready_tickets(self) -> list[PrefetchTicket]:
    ready = []
    for key, ticket in list(self.inflight.items()):
        if ticket.ready_event.query():
            cache = self.layer_caches[ticket.layer_idx]
            ok = cache.mark_staging_ready(
                StagingReservation(
                    layer_idx=ticket.layer_idx,
                    staging_slot_idx=ticket.staging_slot_idx,
                    expert_idx=ticket.expert_idx,
                    generation=ticket.staging_generation,
                )
            )
            if ok:
                ready.append(ticket)
                self._profile["prefetch_completed_count"] += 1
            else:
                # generation mismatch / state mismatch -> stale completion
                self.inflight.pop(key, None)
    return ready
```

### 8.3.6 publish 主算法

```python
def publish_ready(
    self,
    step_id: int,
    max_publish: int | None = None,
) -> int:
    ready = self.poll_ready_tickets()
    if not ready:
        return 0

    publish_budget = self.config.cache_eviction_budget_per_step
    if max_publish is not None:
        publish_budget = min(publish_budget, max_publish)

    published = 0
    for ticket in ready:
        if published >= publish_budget:
            break

        cache = self.layer_caches[ticket.layer_idx]
        snapshot = cache.snapshot(layer_idx=ticket.layer_idx)
        victim_slot = self.cache_strategy.select_victim_slot(
            snapshot=snapshot,
            incoming_expert_idx=ticket.expert_idx,
            step_id=step_id,
        )
        if victim_slot is None:
            continue

        reservation = StagingReservation(
            layer_idx=ticket.layer_idx,
            staging_slot_idx=ticket.staging_slot_idx,
            expert_idx=ticket.expert_idx,
            generation=ticket.staging_generation,
        )

        published_item = cache.publish_ready_staging_to_active(
            reservation=reservation,
            active_slot_idx=victim_slot,
            stream=self.publish_stream,
        )
        if published_item is None:
            self.inflight.pop((ticket.layer_idx, ticket.expert_idx), None)
            continue

        # 只等待 publish_stream，不做全设备同步
        torch.cuda.current_stream().wait_stream(self.publish_stream)
        cache.commit_published_expert(published_item)
        self.inflight.pop((ticket.layer_idx, ticket.expert_idx), None)
        published += 1

        self._profile["publish_count"] += 1

    return published
```

### 8.3.7 verify 前 wait 算法

```python
def wait_for_verify(
    self,
    step_id: int,
    timeout_ms: float,
) -> None:
    if timeout_ms <= 0.0:
        self.publish_ready(step_id=step_id)
        return

    t0 = time.perf_counter()
    self.publish_ready(step_id=step_id)
    self._profile["verify_ready_before_wait_count"] += self._count_ready_relevant_experts()

    deadline = t0 + timeout_ms / 1000.0
    while time.perf_counter() < deadline:
        published = self.publish_ready(step_id=step_id)
        if published > 0:
            break
        if not self.inflight:
            break
        time.sleep(0.0002)

    self._profile["prefetch_wait_ms"] += (time.perf_counter() - t0) * 1000.0
    self._profile["verify_ready_after_wait_count"] += self._count_ready_relevant_experts()
```

这里 `_count_ready_relevant_experts()` 在 baseline 可以先实现为：

```python
def _count_ready_relevant_experts(self) -> int:
    count = 0
    for ticket in self.inflight.values():
        if ticket.ready_event.query():
            count += 1
    return count
```

也就是“当前 inflight 中已经 ready 但尚未 publish 的 ticket 数”；后续再升级为 verify-required 精确子集。

### 8.3.8 prefill / draft / verify 三个入口

```python
def observe_prefill(self, runtime_meta, step_id):
    if self.config.prefetch_use_prefill_history:
        self.observe_runtime_meta(runtime_meta, source="prefill_history", step_id=step_id)

def observe_draft(self, runtime_meta, step_id):
    if self.config.prefetch_use_draft_live:
        self.observe_runtime_meta(runtime_meta, source="draft_live", step_id=step_id)

def observe_verify(self, runtime_meta, step_id):
    if self.config.prefetch_use_verify_history:
        self.observe_runtime_meta(runtime_meta, source="verify_history", step_id=step_id)
```

### 8.3.9 为什么 baseline 不需要后台线程

因为 baseline 的 overlap 机制是：

1. draft replay 前，从全局队列先提交一批 H2D transfer
2. 这些 transfer 在 `transfer_stream` 上与 replay 并行进行
3. 当前 replay 结束后，再把新 `draft_live` 候选入队，并为下一轮提交

也就是说 baseline 的重叠来自：

1. `warm-start/history` 提前准备
2. 独立 transfer stream
3. replay-boundary publish

而不是来自：

1. replay 中途 host 回调
2. per-layer CPU planning

## 8.4 需要修改的文件

1. `nanovllm/expert/prefetcher.py`
2. `nanovllm/scheduling/cache_strategy.py`
3. `nanovllm/scheduling/prefetch_strategy.py`
4. `tests/test_prefetch_runtime.py`（新增）
5. `tests/test_prefetch_wait.py`（新增）

## 8.5 实现步骤

1. 先实现 `observe_*` 与 `submit_from_global_queue()`。
2. 再实现 `poll_ready_tickets()` / `publish_ready()`。
3. 最后加 `wait_for_verify()` 与 profile 统计。

---

## 9. Feature 6：Cache Strategy 与 Prefetch Strategy

## 9.1 实现目标

1. 把“淘汰谁”和“优先预取谁”从 runtime 里拆出来。
2. baseline 先提供可运行的默认策略；adaptive 只留接口。

## 9.2 详细设计

### 9.2.1 `cache_strategy.py`

新增 `nanovllm/scheduling/cache_strategy.py`：

```python
from abc import ABC, abstractmethod

class CacheStrategy(ABC):
    @abstractmethod
    def select_victim_slot(
        self,
        snapshot: LayerCacheSnapshot,
        incoming_expert_idx: int,
        step_id: int,
    ) -> int | None:
        raise NotImplementedError


class LRUCacheStrategy(CacheStrategy):
    def select_victim_slot(self, snapshot, incoming_expert_idx, step_id):
        best_slot = None
        best_age = None
        for slot_idx, expert_idx in enumerate(snapshot.slot_to_expert_lut.tolist()):
            if expert_idx < 0:
                return slot_idx
            age = snapshot.last_access_step[expert_idx]
            if best_age is None or age < best_age:
                best_age = age
                best_slot = slot_idx
        return best_slot


class LFUCacheStrategy(CacheStrategy):
    def select_victim_slot(self, snapshot, incoming_expert_idx, step_id):
        best_slot = None
        best_count = None
        for slot_idx, expert_idx in enumerate(snapshot.slot_to_expert_lut.tolist()):
            if expert_idx < 0:
                return slot_idx
            cnt = snapshot.access_count[expert_idx]
            if best_count is None or cnt < best_count:
                best_count = cnt
                best_slot = slot_idx
        return best_slot


def create_cache_strategy(name: str) -> CacheStrategy:
    normalized = name.strip().lower()
    if normalized == "lru":
        return LRUCacheStrategy()
    if normalized == "lfu":
        return LFUCacheStrategy()
    if normalized == "adaptive":
        return LRUCacheStrategy()  # 占位，后续替换
    raise ValueError(...)
```

### 9.2.2 `prefetch_strategy.py`

baseline 中 prefetch strategy 的责任不是“发 copy”，而是“对全局队列候选进行可插拔重排/裁剪”：

```python
class PrefetchStrategy(ABC):
    @abstractmethod
    def rank(
        self,
        candidates: list[PrefetchCandidate],
        step_id: int,
    ) -> list[PrefetchCandidate]:
        raise NotImplementedError


class NoopPrefetchStrategy(PrefetchStrategy):
    def rank(self, candidates, step_id):
        return candidates


class HistoryWindowPrefetchStrategy(PrefetchStrategy):
    def __init__(self, config: Config):
        self.config = config

    def rank(self, candidates, step_id):
        fresh = [
            c for c in candidates
            if step_id - c.last_seen_step <= self.config.prefetch_history_ttl_steps
        ]
        fresh.sort(key=lambda c: (-c.priority, c.layer_idx, c.expert_idx))
        return fresh


def create_prefetch_strategy(name: str, config: Config) -> PrefetchStrategy:
    normalized = name.strip().lower()
    if normalized == "noop":
        return NoopPrefetchStrategy()
    if normalized == "history_window":
        return HistoryWindowPrefetchStrategy(config)
    raise ValueError(...)
```

`PrefetchRuntime.submit_from_global_queue()` 在 `ranked_candidates()` 之后接一层：

```python
ranked = self.global_queue.ranked_candidates(...)
ranked = self.prefetch_strategy.rank(ranked, step_id=step_id)
```

## 9.3 需要修改的文件

1. `nanovllm/scheduling/cache_strategy.py`（新增）
2. `nanovllm/scheduling/prefetch_strategy.py`（新增）
3. `nanovllm/expert/prefetcher.py`
4. `tests/test_cache_strategy.py`（新增）
5. `tests/test_prefetch_strategy.py`（新增）

## 9.4 实现步骤

1. 先实现 `LRUCacheStrategy`。
2. 再实现 `LFUCacheStrategy`。
3. 最后接 `HistoryWindowPrefetchStrategy`。

---

## 10. Feature 7：ModelRunner 级联动与具体函数改造

## 10.1 实现目标

1. 明确每个入口函数如何与 runtime 交互。
2. 给出可直接落地的函数签名与关键逻辑块。

## 10.2 详细设计

### 10.2.1 `ModelRunner.run()`

当前签名不变：

```python
def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
```

关键改造：

```python
def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
    step_id = None
    input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
    if self.prefetch_runtime is not None and is_prefill:
        step_id = self._next_prefetch_step_id()
        self.runtime_meta_recorder.arm(
            mode="prefill",
            step_id=step_id,
            token_capacity=int(input_ids.numel()),
            logical_token_count=int(input_ids.numel()),
        )

    # existing body continues with the already prepared input_ids / positions
    token_ids = ...

    if self.prefetch_runtime is not None and is_prefill:
        handle = self.runtime_meta_recorder.offload_async(self.prefetch_runtime.metadata_stream)
        runtime_meta = self.runtime_meta_recorder.collect(handle, wait=True)
        self.prefetch_runtime.observe_prefill(runtime_meta, step_id=step_id)

    return token_ids
```

实现时应把上面的逻辑嵌回当前 `run()` 的真实 prepare / sample / run_model 顺序中，不要再额外复制一套 prepare。

### 10.2.2 `ModelRunner.run_draft()`

把签名从：

```python
def run_draft(self, seqs: list[Sequence]) -> tuple[list[int], list]:
```

改为：

```python
def run_draft(self, seqs: list[Sequence]) -> tuple[list[int], dict[str, object]]:
```

返回值第二项用于把 `prefetch_step_id` 传回 `SpeculativeEngine`。

详细逻辑：

```python
def run_draft(self, seqs: list[Sequence]) -> tuple[list[int], dict[str, object]]:
    t0 = perf_counter()
    step_id = self._next_prefetch_step_id()

    self._set_speculative_execution_mode("draft")
    self._decode_graph_policy = "draft"
    try:
        if self.prefetch_runtime is not None:
            draft_capacity = len(seqs)
            if self._can_use_draft_cudagraph(len(seqs)):
                draft_capacity = next(x for x in self.draft_graph_bs if x >= len(seqs))

            # 1) 把上一轮已完成的 staging expert publish 到 active cache
            self.prefetch_runtime.publish_ready(step_id=step_id)

            # 2) 利用 global warm-start/history queue 在 replay 前先发射一批 transfer
            self.prefetch_runtime.submit_from_global_queue(
                step_id=step_id,
                phase="before_draft",
            )

            # 3) arm 本轮 draft runtime metadata recorder
            self.runtime_meta_recorder.arm(
                mode="draft",
                step_id=step_id,
                token_capacity=draft_capacity,
                logical_token_count=len(seqs),
            )

        token_ids = self.run(seqs, False)

        if self.prefetch_runtime is not None:
            # 4) replay 结束后异步拷 metadata 到 host
            handle = self.runtime_meta_recorder.offload_async(self.prefetch_runtime.metadata_stream)
            runtime_meta = self.runtime_meta_recorder.collect(handle, wait=True)

            # 5) 更新 draft_live 候选
            self.prefetch_runtime.observe_draft(runtime_meta, step_id=step_id)

            # 6) 为 verify / 下一轮 draft 再提交一批 transfer
            self.prefetch_runtime.submit_from_global_queue(
                step_id=step_id,
                phase="after_draft",
            )

        return token_ids, {"prefetch_step_id": step_id}
    finally:
        self._decode_graph_policy = "standard"
        self._set_speculative_execution_mode("normal")
        # existing profile aggregation...
```

### 10.2.3 `ModelRunner.wait_prefetch_for_verify()`

新增：

```python
def wait_prefetch_for_verify(
    self,
    step_id: int,
) -> dict[str, float]:
    if self.prefetch_runtime is None:
        return {}

    t0 = perf_counter()
    self.prefetch_runtime.wait_for_verify(
        step_id=step_id,
        timeout_ms=self.config.prefetch_verify_wait_ms,
    )
    return {
        "verify_prefetch_wait_ms": (perf_counter() - t0) * 1000.0,
    }
```

### 10.2.4 `ModelRunner.run_verify()`

签名保持：

```python
def run_verify(self, seqs: list[Sequence], verify_lengths: list[int]) -> list[list[int]]:
```

关键改造：

```python
def run_verify(self, seqs, verify_lengths):
    total_t0 = perf_counter()
    step_id = self._next_prefetch_step_id()
    self._set_speculative_execution_mode("verify")

    input_ids, positions = self.prepare_prefill(seqs)

    if self.prefetch_runtime is not None:
        self.runtime_meta_recorder.arm(
            mode="verify",
            step_id=step_id,
            token_capacity=int(input_ids.numel()),
            logical_token_count=int(input_ids.numel()),
        )

    try:
        hidden_states = self.model(input_ids, positions)
        # existing profile path...
    finally:
        self._set_speculative_execution_mode("normal")

    if self.prefetch_runtime is not None:
        handle = self.runtime_meta_recorder.offload_async(self.prefetch_runtime.metadata_stream)
        runtime_meta = self.runtime_meta_recorder.collect(handle, wait=True)
        self.prefetch_runtime.observe_verify(runtime_meta, step_id=step_id)
        self.prefetch_runtime.submit_from_global_queue(
            step_id=step_id,
            phase="after_verify",
        )

    # existing verify trace extraction...
    return verify_tokens_per_seq
```

### 10.2.5 `ModelRunner.get_profile()`

不改变 canonical alias ownership，只增加原始 model-level 统计：

```python
def get_profile(self, reset: bool = False) -> dict:
    out = # existing model profile
    if self.prefetch_runtime is not None:
        out.update(self.prefetch_runtime.get_profile(reset=False))
    # canonical aliases 仍由 LLMEngine.get_profile 暴露
    return out
```

## 10.3 需要修改的文件

1. `nanovllm/engine/model_runner.py`
2. `nanovllm/models/qwen3_moe.py`
3. `nanovllm/expert/prefetcher.py`
4. `tests/test_model_runner_prefetch.py`（新增）

## 10.4 实现步骤

1. 先接 `run_draft()`。
2. 再接 `wait_prefetch_for_verify()`。
3. 最后接 `run()` 的 prefill 观测与 `run_verify()` 的 verify 反馈。

---

## 11. Feature 8：SpeculativeEngine 级联动

## 11.1 实现目标

1. 在不改写 speculative 主逻辑的前提下，把 verify 前 wait 接入正确边界。
2. 让 `run_draft()` 返回的 `prefetch_step_id` 能传递到 wait 阶段。

## 11.2 详细设计

修改 [nanovllm/engine/speculative/spec_engine.py](../nanovllm/engine/speculative/spec_engine.py)：

### 11.2.1 draft loop

当前：

```python
draft_result = self.model_runner.call("run_draft", seqs)
if isinstance(draft_result, tuple):
    token_ids = draft_result[0]
else:
    token_ids = draft_result
```

修改为：

```python
draft_prefetch_state = None
draft_result = self.model_runner.call("run_draft", seqs)
if isinstance(draft_result, tuple):
    token_ids = draft_result[0]
    if len(draft_result) > 1 and isinstance(draft_result[1], dict):
        draft_prefetch_state = draft_result[1]
else:
    token_ids = draft_result
```

### 11.2.2 verify 前 wait

在 `prepare_verify` 完成、`run_verify` 调用前插入：

```python
if draft_prefetch_state is not None and "prefetch_step_id" in draft_prefetch_state:
    wait_prof = self.model_runner.call(
        "wait_prefetch_for_verify",
        draft_prefetch_state["prefetch_step_id"],
    )
    if self.profile_enabled and wait_prof:
        for key, value in wait_prof.items():
            self._profile[key] += float(value)
```

为什么放在这里：

1. 这是 verify 前最后一个稳定 host-side 边界
2. 不改变 verify forward 语义
3. 与当前 `run_verify()` 的 monolithic 结构兼容

## 11.3 需要修改的文件

1. `nanovllm/engine/speculative/spec_engine.py`
2. `tests/test_spec_engine_prefetch.py`（新增）

## 11.4 实现步骤

1. 先接 `draft_prefetch_state` 的透传。
2. 再接 verify 前 wait。
3. 最后补充 spec-level profile。

---

## 12. Feature 9：Verify Feedback Path（baseline 精确版本）

## 12.1 实现目标

1. 利用 verify 的精确路由结果提高后续 step 的 warm-start 质量。
2. 不要求“当前 verify forward 内立即 overlap transfer”。

## 12.2 设计原则

1. baseline 中 verify 反馈的重点是“提升下一步”，不是“抢当前步”。
2. 由于当前 `run_verify()` 是 monolithic forward，baseline 不引入 mid-forward hook。
3. 如果将来要做同一 verify forward 内 overlap，放到附录。

## 12.3 详细设计

verify 反馈流程：

1. `run_verify()` 开始前 `arm(mode="verify")`
2. 各层 block 在 eager verify forward 中记录 routing metadata
3. verify forward 完成后 `offload_async()` + `collect(wait=True)`
4. `PrefetchRuntime.observe_verify(runtime_meta, step_id)`
5. verify_history 候选更新进 global queue
6. `submit_from_global_queue(phase="after_verify")` 为下一次 draft 做准备

这条路径的价值：

1. verify 是全精度观测，优先级高于 draft_live
2. 它能持续提高未来第一轮 draft replay 的 warm-start 质量

## 12.4 需要修改的文件

1. `nanovllm/engine/model_runner.py`
2. `nanovllm/models/qwen3_moe.py`
3. `nanovllm/expert/prefetcher.py`
4. `tests/test_verify_feedback.py`（新增）

## 12.5 实现步骤

1. 先导出 verify runtime meta。
2. 再接 global queue 更新。
3. 最后接 `after_verify` 的 dispatch。

---

## 13. Feature 10：DraftScheduler 与 placement 的最小改造

## 13.1 实现目标

1. 保持现有 draft / verify 执行语义不变。
2. 仅添加支撑 metadata / access 统计 / helper 的最小改动。

## 13.2 详细设计

### 13.2.1 `placement.py`

`MoEExecutionPlan` 不增字段。  
只新增两个 helper：

```python
def flatten_selected_and_weights(
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return selected_experts.reshape(-1).to(torch.int64), routing_weights.reshape(-1).float()


def build_runtime_meta_view(
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # helper for recorder / profile code paths
    return selected_experts, routing_weights
```

### 13.2.2 `draft_scheduler.py`

当前接口全部保留。  
只新增一个不破坏现有调用面的统一 helper：

```python
@dataclass
class DraftSchedulerContext:
    uncached_expert_mask: torch.Tensor
    routing_weights_flat: torch.Tensor
    selected_experts_flat: torch.Tensor
    top_c: int


def build_draft_scheduler_context(
    uncached_expert_mask: torch.Tensor,
    routing_weights_flat: torch.Tensor,
    selected_experts_flat: torch.Tensor,
    top_c: int,
) -> DraftSchedulerContext:
    return DraftSchedulerContext(...)
```

baseline 不强制重写现有 scheduler 逻辑；这个 helper 只是为了后续扩展。

## 13.3 需要修改的文件

1. `nanovllm/expert/placement.py`
2. `nanovllm/scheduling/draft_scheduler.py`
3. `tests/test_draft_scheduler.py`

## 13.4 实现步骤

1. 先加 helper，不改 hot path。
2. 后续需要时再让 prefetch strategy 复用这些 helper。

---

## 14. Feature 11：Profiling、Observability 与 Benchmark 合约

## 14.1 实现目标

1. 让 profile 能准确回答 prefetch 是否有效。
2. 不回退 phase2_post 已有 canonical 字段。
3. 区分 global queue 的三类来源收益。

## 14.2 详细设计

### 14.2.1 新增原始统计字段

`PrefetchRuntime.get_profile()` 返回：

```python
{
    "prefetch_submit_count": ...,
    "prefetch_completed_count": ...,
    "prefetch_late_count": ...,
    "prefetch_wait_ms": ...,
    "prefetch_consumed_count": ...,
    "prefetch_timeout_count": ...,
    "publish_count": ...,
    "publish_ms": ...,
    "metadata_offload_ms": ...,
    "metadata_offload_bytes": ...,
    "history_prefetch_submit_count": ...,
    "verify_history_prefetch_submit_count": ...,
    "draft_live_prefetch_submit_count": ...,
    "verify_ready_before_wait_count": ...,
    "verify_ready_after_wait_count": ...,
}
```

### 14.2.2 “consumed” 的 baseline 定义

baseline 中可实现的 `prefetch_consumed_count` 定义为：

1. expert 在 verify 开始前已经在 active cache 中
2. 且 verify runtime meta 显示该 expert 确实在 verify 中被激活

实现方式：

1. `run_verify()` 结束后读取 verify runtime meta
2. 对每层 unique verify experts 做一次 `cached_expert_mask` 交集统计
3. 如果该 expert 来自最近一次 publish，可记作 consumed

这一定义已经足以回答“prefetch 是否被 verify 真正消费”。

### 14.2.3 `LLMEngine.get_profile()` canonical alias

继续由 [nanovllm/engine/llm_engine.py](../nanovllm/engine/llm_engine.py) 统一暴露。  
新增 alias：

```python
canonical_map.update({
    "prefetch_submit_count": "model_prefetch_submit_count",
    "prefetch_completed_count": "model_prefetch_completed_count",
    "prefetch_wait_ms": "model_prefetch_wait_ms",
    "prefetch_consumed_count": "model_prefetch_consumed_count",
    "publish_count": "model_publish_count",
})
```

### 14.2.4 benchmark 必须回答的问题

Phase 3 基线 benchmark 报告必须能回答：

1. 第一轮 draft replay 前是否已经利用 global warm-start/history queue 发射 transfer？
2. 多少 transfer 在 replay 期间完成？
3. verify 前 wait 是否提高了 ready 命中？
4. `prefill_history / verify_history / draft_live` 哪类来源更有效？
5. publish 开销有多大？
6. metadata offload 是否影响 graph hit rate？

## 14.3 需要修改的文件

1. `nanovllm/expert/prefetcher.py`
2. `nanovllm/engine/model_runner.py`
3. `nanovllm/engine/speculative/spec_engine.py`
4. `nanovllm/engine/llm_engine.py`
5. benchmark 脚本与报告脚本

## 14.4 实现步骤

1. 先加 runtime 原始计数器。
2. 再加 `ModelRunner.get_profile()` 汇总。
3. 最后在 `LLMEngine.get_profile()` 暴露 canonical alias。

---

## 15. 实现阶段拆分

## 15.1 Phase 3A：配置、recorder、cache lifecycle skeleton

1. `Config` 扩展
2. `ModelRuntimeMetaRecorder`
3. `LayerExpertCache` 的 access 统计 / staging skeleton
4. `CacheStrategy` / `PrefetchStrategy` skeleton

验收标准：

1. prefetch 开关关闭时不改变现有行为
2. draft graph 仍可 capture / replay

## 15.2 Phase 3B：global warm-start/history queue + PrefetchRuntime

1. `GlobalWarmStartQueue`
2. `PrefetchRuntime.submit_from_global_queue()`
3. prefill / verify 历史入队
4. draft replay 前 submit

验收标准：

1. 第一轮 draft replay 前能看到 `prefetch_submit_count > 0`
2. 不影响现有 correctness

## 15.3 Phase 3C：draft metadata offload + draft_live

1. draft recorder 接入 graph
2. `run_draft()` 的 metadata offload
3. `draft_live` 入队
4. replay 后追加 dispatch

验收标准：

1. graph hit rate 不回退明显
2. `draft_live_prefetch_submit_count` 非零

## 15.4 Phase 3D：publish / wait / verify feedback

1. `poll_ready_tickets()`
2. `publish_ready()`
3. `wait_prefetch_for_verify()`
4. `run_verify()` 的 verify_history 反馈

验收标准：

1. verify 前 wait 可工作
2. `publish_count` / `prefetch_consumed_count` 有意义

## 15.5 Phase 3E：observability 与 benchmark 收敛

1. canonical alias 完整
2. benchmark 报告输出
3. 与 phase2_post 对照实验

---

## 16. 测试计划

## 16.1 单元测试

1. `tests/test_expert_cache_staging.py`
   - reserve -> inflight -> ready -> publish -> release 正确
2. `tests/test_expert_cache_generation.py`
   - generation mismatch 时丢弃 completion
3. `tests/test_prefetch_global_queue.py`
   - 入队、去重、衰减、TTL、排序正确
4. `tests/test_prefetch_runtime.py`
   - submit / poll / publish / wait 正确
5. `tests/test_prefetch_runtime_meta.py`
   - recorder eager / graph path 形状与收集逻辑正确
6. `tests/test_cache_strategy.py`
   - LRU / LFU victim 正确

## 16.2 集成测试

1. `tests/test_model_runner_prefetch.py`
   - `run_draft()` 前 submit、后 observe_draft 路径正确
2. `tests/test_spec_engine_prefetch.py`
   - verify 前 wait 在 speculative loop 中被调用
3. `tests/test_verify_feedback.py`
   - verify_history 能进入全局队列
4. `tests/test_draft_cuda_graph_real_world.py`
   - recorder 接入后 graph replay 不退化

## 16.3 回归测试

1. prefetch 关闭时 Phase 2 行为不变
2. verify token trace 不变
3. `MoEExecutionPlan` 结构不变
4. `LLMEngine.get_profile()` 旧 alias 不丢失

## 16.4 必测失败场景

1. staging slot 用尽
2. inflight ticket 晚到
3. publish 时 victim slot generation 变化
4. verify 前 wait 超时
5. graph bucket 改变后 recorder token capacity 切换
6. cache 中 expert 已存在但队列仍残留旧条目

---

## 17. 附录 A：Approach 2 图内 per-layer GPU->CPU signaling

> 本节是优化阶段设计，不属于 baseline 第一轮实现。

## 17.1 背景

在 draft graph 中，如果每层都能把 frontier metadata 发回 CPU，那么 CPU 就能在收到 layer `i` 信号后认为：

1. 当前 replay 中 layers `<= i-1` 的 active slots 已安全
2. 可以更早地发射 transfer 或更早替换

## 17.2 可能实现

```python
class FrontierSignalSink:
    def on_layer_complete(
        self,
        layer_idx: int,
        step_id: int,
        token_count: int,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> None:
        ...
```

但这条路径需要额外验证：

1. 当前 PyTorch + CUDA graph 是否稳定支持图内固定形状 D2H frontier signal
2. CPU reaction latency 是否足够低
3. 相比 baseline staging 是否真的更快

## 17.3 与 ownership 的关系

如果后续证明把 frontier signal 消费责任部分下沉到 model/block 内能显著简化实现，则可在这一附录路径中讨论 ownership 变化；baseline 仍保持 `ModelRunner` 持有主 runtime。

---

## 18. 附录 B：Approach 3 segmented graph

> 本节是优化阶段设计，不属于 baseline 第一轮实现。

思路：

1. 把 whole-model draft graph 切分为多个 segment
2. 在 segment 边界获得更细粒度 publish / planning 安全点
3. 为 Approach 2 与 predictive verify-prefetch 提供更明确的执行边界

风险：

1. graph 命中率下降
2. bucket 管理更复杂
3. 需要重新评估 capture/replay 开销

---

## 19. 附录 C：Request-scoped warm-start/history queue

> 本节是 post-baseline 优化设计，不属于 baseline 第一轮实现。

优化目标：

1. 用 request-scoped / sequence-scoped 队列替代 baseline global queue
2. 避免不同 request 的工作集互相污染
3. 让 warm-start 命中更容易解释

基本结构：

```python
class RequestScopedWarmStartQueue:
    per_seq_entries: dict[int, dict[tuple[int, int], PrefetchCandidate]]
```

调度时只合并当前 `speculative_step(seqs)` 中的序列对应队列。

---

## 20. 附录 D：Predictive verify-prefetch

> 本节是最终阶段优化设计，不属于 baseline 第一轮实现。

目标：

1. 在 verify 中利用跨层预测，预取未来层可能需要的 expert
2. 把当前 baseline 的“verify 后反馈”升级为“verify 中预测”

实现前提：

1. 先稳定 baseline staging lifecycle
2. 再稳定 request-scoped queue 或更细粒度 frontier
3. 最后评估 predictive path 是否值得引入

---

## 21. 结论

Phase 3 的 baseline 实现应当是：

1. 保留并完整实现 **global warm-start/history queue**
2. 采用 **Approach 1：staging memory + replay-boundary publish**
3. 在 prefill / verify / draft 三个阶段统一收集 routing metadata
4. 由 `ModelRunner` 持有 `PrefetchRuntime`，在边界驱动 submit / publish / wait
5. draft replay 前先利用 warm-start candidates 发射 transfer，使 replay 期间带宽不空闲
6. verify 后把精确路由结果继续反馈到全局队列

post-baseline 优化再逐步引入：

1. Approach 2 图内 per-layer signaling
2. segmented graph
3. request-scoped queue
4. predictive verify-prefetch

这样分层以后，主设计可直接落地，优化设计也保留了清晰的演进空间。
