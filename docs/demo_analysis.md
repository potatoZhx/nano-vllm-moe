# on_device_sd/demo 项目分析文档

> 目标：完整梳理 on_device_sd/demo 项目的设计意图、架构、算法、实现细节与待优化接口，为基于 nano-vllm-moe 的改造提供参考。

---

## 1. 项目目标

构建一个面向**端侧单卡**（single GPU + host CPU）的 **MoE 大模型推理系统**，核心思路是：

| 维度 | 说明 |
|------|------|
| 目标模型 | Qwen3-30B-A3B-Base（48 层，128 routed experts / 层，top-8 路由，无 shared expert） |
| 部署环境 | 单机单卡，GPU 显存不足以放下全部 expert |
| 核心 idea | **CPU-GPU 异构协同推理** + **基于 Expert 替换的投机解码（Speculative Decoding）** |
| 性能关键 | 减少 CPU 参与量、异步传输、Fused MoE kernel、Expert cache 命中率 |

> **关于 Shared Expert**：Qwen3-30B-A3B-Base 的 config.json 中 `num_shared_experts` 未设置（默认为 0），运行日志也确认 `shared_experts=0`。但 demo 代码的 `ParameterLoader` 已完整实现了 shared expert 的加载逻辑（独立命名空间方式如 DeepSeek-V2、experts 列表前缀方式、shared_expert_gate 等），改造后的系统也需要兼容带 shared expert 的模型（如 DeepSeek-V2/V3、Qwen2-MoE 等）。

### 1.1 核心创新点

1. **Expert 替换投机解码**：Draft 阶段不传输缺失 expert，而是用 GPU 上已缓存的 expert 替代，显著减少 CPU-GPU 数据搬运
2. **Top-c CPU 执行**：Draft 阶段允许少量（top-c，默认 c=2）高激活分数 expert 在 CPU 执行，与 GPU 并行
3. **Expert 预取调度**：根据激活历史，在 Draft 期间异步将下一轮可能用到的 expert 传输到 GPU
4. **Verify 阶段全精度校验**：使用完整路由（无替换）做 verify，通过接受策略过滤 draft token

---

## 2. 系统架构

### 2.1 整体分层

```
┌──────────────────────────────────────────────────────────┐
│                    User API Layer                         │
│  MoEInferenceEngine: generate() / submit() / get_result()│
└──────────────────────────┬───────────────────────────────┘
                           │
┌──────────────────────────┴───────────────────────────────┐
│               Execution Orchestrator                     │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐ │
│  │  Standard    │ │  Speculative │ │  Continuous Batch │ │
│  │  Decode      │ │  (Draft +    │ │  Engine           │ │
│  │  Engine      │ │   Verify)    │ │  (CBScheduler +   │ │
│  └──────────────┘ └──────────────┘ │   CBExecutor)     │ │
│                                     └──────────────────┘ │
└──────────────────────────┬───────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
┌─────────────────┐ ┌───────────────┐ ┌──────────────┐
│ Memory Mgmt     │ │ Scheduling    │ │ Model Runner │
│ - ParameterLoader│ │ - Prefetcher  │ │ (Qwen3       │
│ - ExpertCache   │ │ - DraftSched  │ │  ModelRunner) │
│ - PagedKVCache  │ │ - CacheStrat  │ │              │
└─────────────────┘ └───────────────┘ └──────────────┘
         │                 │                 │
         └─────────────────┼─────────────────┘
                           │
                ┌──────────┴──────────┐
                │                     │
                ▼                     ▼
         ┌──────────────┐    ┌──────────────┐
         │  GPU Memory  │    │  CPU Memory  │
         │ - Static     │    │ - Expert     │
         │   Params     │    │   Pool       │
         │ - Expert     │    │ (pin_memory) │
         │   Cache      │    │              │
         │ - KV Cache   │    │              │
         └──────────────┘    └──────────────┘
```

### 2.2 关键模块

| 模块 | 位置 | 职责 |
|------|------|------|
| **ModelRunner** | `src/core/model_runner.py` | 抽象接口：embed / forward_attention / route_experts / forward_moe / compute_logits |
| **Qwen3ModelRunner** | `src/model/qwen3_runner.py` | 具体实现，支持 CPU/GPU 混合 expert 执行，可选 fused MoE |
| **CBExecutor** | `src/execution/cb_executor.py` | 执行器：prefill / decode_standard / decode_speculative |
| **CBScheduler** | `src/execution/cb_scheduler.py` | 调度器：prefill vs decode 调度，KV cache 分配 |
| **ParameterLoader** | `src/memory/parameter_loader.py` | 参数加载：静态→GPU，shared expert→GPU（若有），routed expert→CPU，可配置部分→GPU |
| **ExpertCache** | `src/memory/expert_cache.py` | GPU expert 缓存管理：LRU 替换、异步预取、pinned expert |
| **PagedKVCache** | `src/memory/paged_kv_cache.py` | 分页 KV cache，支持 draft/verify 状态切换 |
| **DraftScheduler** | `src/scheduling/draft_schduler.py` | Draft 调度策略：top-c CPU 选择、GPU 替代、传输决策 |
| **ExpertPrefetcher** | `src/scheduling/prefetcher.py` | Expert 预取：预测下层激活 expert，异步传输 |
| **AcceptanceStrategy** | `src/execution/acceptance_strategy.py` | 投机采样接受策略：Standard / Adaptive |

---

## 3. 核心数据结构

### 3.1 ExpertID 与 ExpertPlacement

```python
@dataclass
class ExpertID:
    layer_idx: int
    expert_idx: int   # 负数表示 shared expert

@dataclass
class ExpertPlacement:
    gpu_expert_params: Dict[int, Dict[str, Tensor]]  # expert_idx → {gate_proj, up_proj, down_proj}
    cpu_expert_params: Dict[int, Dict[str, Tensor]]
    substitution_map: Dict[int, int]                  # orig_expert_idx → substitute_expert_idx
    routing_result: Optional[RoutingResult]
```

**这是系统最核心的中间结构**：引擎（CBExecutor）根据路由结果和缓存状态构建 ExpertPlacement，ModelRunner 据此执行 MoE 计算。

### 3.2 RoutingResult

```python
@dataclass
class RoutingResult:
    layer_idx: int
    topk_indices: Tensor    # [num_tokens, top_k]
    topk_scores: Tensor     # [num_tokens, top_k]
    activated_expert_ids: Set[ExpertID]
```

路由计算与 expert 执行**分离**，允许引擎在两者之间插入调度逻辑。

### 3.3 AttentionOutput

```python
@dataclass
class AttentionOutput:
    hidden_states: Tensor       # attn 后 + residual
    post_attn_normed: Tensor    # 用于 routing 和 MoE 计算
    residual: Tensor            # MoE 之前的 residual
```

---

## 4. 核心算法

### 4.1 参数加载策略

```
加载流程：
1. 静态参数（embedding, attention, layernorm, router, lm_head）→ GPU
2. Shared expert 参数 → GPU（pinned，不可驱逐）
   - 当前目标模型 Qwen3-30B-A3B 无 shared expert（num_shared_experts=0）
   - 但系统需兼容带 shared expert 的模型（如 DeepSeek-V2/V3）
   - 支持两种加载方式：
     a. 独立命名空间：mlp.shared_expert.{proj} / mlp.shared_experts.{idx}.{proj}
     b. experts 列表前缀：experts 列表的前 num_shared_experts 个
   - 如有 shared_expert_gate，也加载到 GPU
3. 所有 routed expert 参数 → CPU（pin_memory 加速后续传输）
4. 根据 placement_config，将部分 routed expert 复制到 GPU
   - 若某层全部 expert 都在 GPU，构建 fused layout（[E, O, I] 连续张量）
```

### 4.2 Prefill 阶段

```
For each layer:
  1. forward_attention(hidden_states, kv_cache, positions)
  2. route_experts(post_attn_normed) → RoutingResult
  3. prefetcher.on_layer_complete(layer_idx, activations)  // 预取下层
  4. build_prefill_placement(routing_result, expert_cache, parameter_loader)
     → ExpertPlacement（优先 GPU cache，否则 CPU）
  5. forward_moe(attn_output, expert_placement)
```

### 4.3 Standard Decode

单步自回归解码，每步一个 token，使用 `build_prefill_placement` 构建 placement。

### 4.4 Speculative Decode（核心）

```
┌─ Draft Phase ─────────────────────────────────────────────────┐
│ For step in range(max_draft_tokens):                          │
│   1. forward_attention (decode mode)                          │
│   2. route_experts → RoutingResult                            │
│   3. build_draft_placement:                                   │
│      a. 分离 GPU 可用 vs 不可用 expert                          │
│      b. 不可用中选 <= top-c → CPU 执行                              │
│      c. 其余不可用 → 从 GPU cache 选替代 expert                   │
│   4. forward_moe with placement (CPU 并行 ThreadPoolExecutor)  │
│   5. sample → draft_token                                     │
│   6. _schedule_expert_transfers(step_activations)             │
│      → 异步将高频 expert 从 CPU 传输到 GPU cache                 │
│      触发时机有两种策略：                                         │
│      a. 每层 route results 出来后立即启动（优选，更细粒度，          │
│         可与当前层 MoE 计算 overlap）                             │
│      b. 每个 draft step 完成后启动（当前实现）                     │
│      当前代码在每个 step 结束后调用，改造目标倾向方案 a             │
│   7. expert_cache.complete_ready_transfers()                  │
└───────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─ Verify Phase ────────────────────────────────────────────────┐
│ 1. 构造 verify 输入：last_token + draft_tokens                 │
│ 2. _forward_verify (类似 prefill，使用 build_prefill_placement) │
│    → 全精度路由，无替换                                          │
│ 3. 输出 verify_logits_map[seq_id]                             │
└───────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─ Accept Phase ────────────────────────────────────────────────┐
│ For each seq:                                                 │
│   if greedy (do_sample=False):                                │
│     逐位比较 draft_token vs verify_argmax                      │
│     第一个不匹配处停止，accept 前面的 + verify 的下一个 token      │
│   else:                                                       │
│     AcceptanceStrategy.accept(draft_tokens, verify_logits)    │
│     - Standard: P_verify(token) >= threshold                  │
│     - Adaptive: 动态调整 threshold                             │
│   kv_cache.accept_draft(seq_id, num_accepted)                 │
└───────────────────────────────────────────────────────────────┘
```

#### 4.4.1 build_draft_placement 详细逻辑

```python
def build_draft_placement(routing_result, expert_cache, parameter_loader,
                          draft_scheduler, top_c, num_experts):
    # 1. 检查哪些 activated expert 在 GPU 上可用
    gpu_available = {eid for eid in activated if _get_gpu_params(eid) is not None}
    missing_gpu = activated - gpu_available

    # 2. 从 missing 中选 top-c 个在 CPU 执行（按激活分数排序）; 
    # ps:这里允许等于top_c，比如剩余不足top-c
    cpu_expert_ids = draft_scheduler.select_cpu_experts(activations, top_c)

    # 3. 剩余 missing（既不在 GPU 也不在 CPU 执行的）→ 找 GPU 替代
    # 小于1则跳过替换
    needs_substitution = activated - gpu_available - cpu_expert_ids
    substitution_map = draft_scheduler.select_gpu_substitutes(
        requested=needs_substitution, cached=gpu_available, all=all_layer_experts)

    # 4. 构建最终的 gpu_expert_params 和 cpu_expert_params
    return ExpertPlacement(gpu_params, cpu_params, substitution_map, routing_result)
```

#### 4.4.2 Expert 替换执行

在 `Qwen3ModelRunner._execute_moe_with_placement` 中：
1. 将 `topk_indices` 中被替换的 expert_idx 映射到 substitute_idx
2. 构建 GPU/CPU 任务列表
3. **并行执行**：CPU task 使用 `ThreadPoolExecutor`，GPU task 使用 fused MoE 或逐 expert forward
4. 结果按 routing weights 加权累加

#### 4.4.3 Verify 跳过优化

当满足以下条件时，可跳过 verify 直接接受所有 draft token：
- 所有 routed expert 都在 GPU 上（无替换发生）
- `draft_top_c == 0`（无 CPU 执行）
- 所有 sequence 使用 greedy decoding

### 4.5 Expert 预取调度

#### Prefill/Verify 期间的预取

`ExpertPrefetcher` 使用可插拔的 `PrefetchStrategy`：
- **SimplePrefetchStrategy**：预测下一层激活与当前层相同的 expert（时序局部性）
- **HistoryBasedPrefetchStrategy**：基于历史共现矩阵预测

#### Draft 期间的传输调度

`select_experts_to_prefetch` 函数：

- 统计所有 draft step 的 expert 激活频率和分数
- 排除已缓存和正在传输的 expert
- 按 `count × avg_score` 排序，选取 top 若干个异步传输

**传输数量预估**：系统在推理引擎启动前，需根据以下信息估算每次可传输的 expert 数量上限：
- **Expert 传输速度**：单个 expert 参数量 × dtype 字节数 / PCIe 带宽（可通过 benchmark 测定实际 CPU→GPU 传输吞吐）
- **计算时长窗口**：一层 MoE 计算时长（若按每层触发）或一个 draft step 的全部层计算时长（若按 step 触发）
- 由此可计算出在一个计算窗口内可完成的异步传输数量 `max_prefetch_per_window = floor(compute_time / transfer_time_per_expert)`
- 该值作为 `max_prefetch_per_step`（或 `max_prefetch_per_layer`）的默认配置，避免过量传输导致带宽竞争或阻塞计算

### 4.6 Expert Cache 管理

`ExpertCache` 使用 `CacheReplacementStrategy`（LRU/LFU/Adaptive）：
- `put` / `get`：同步缓存操作
- `prefetch_async`：使用 CUDA stream 异步传输
- `complete_ready_transfers`：轮询完成的异步传输
- pinned expert（shared expert）不可驱逐

### 4.7 KV Cache 与 Draft/Verify 切换

每一轮 draft-verify 的 KV cache 生命周期如下：

```
假设上一轮 prefill/verify 结束后，KV cache 中已有 token 序列 [t0, t1, ..., tN]
其中 tN 是上一轮生成的最后一个 token（即本轮 draft 的输入）

┌─ Draft Phase ──────────────────────────────────────────────────┐
│ start_draft(seq_id)：记录 draft_start = N+1                     │
│ 自回归生成 draft tokens [d0, d1, ..., dK-1]                      │
│ KV cache 临时扩展为 [t0..tN, d0..dK-1]                           │
└───────────────────────────────────────────────────────────────┘
                          │
                          ▼ Draft 结束，丢弃 draft 阶段的 KV cache
                            KV cache 回退到 [t0..tN]
                          │
┌─ Verify Phase ─────────────────────────────────────────────────┐
│ 输入：[tN] + [d0, d1, ..., dK-1]（上一轮最后 token + draft tokens）│
│ 基于 [t0..tN-1] 的已有 KV cache，以 prefill 模式                  │
│ 一次性处理 K+1 个 token，生成 verify_logits                       │
│ KV cache 扩展为 [t0..tN, d0..dK-1]                               │
└───────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─ Accept Phase ─────────────────────────────────────────────────┐
│ 比较 draft tokens 与 verify logits                               │
│ 假设接受了 [d0..dM-1]（M 个 token）                               │
│ accept_draft(seq_id, M)：                                       │
│   - 保留 KV cache [t0..tN, d0..dM-1]                            │
│   - 丢弃 [dM..dK-1] 对应的 KV 条目（释放多余 block）              │
│ verify 在位置 M 的 logits 对应的 argmax/sample 即为                │
│   本轮 verify 新生成的 token tN+M+1                               │
│ 将 tN+M+1 追加到序列，KV cache 最终为 [t0..tN, d0..dM-1, tN+M+1] │
└───────────────────────────────────────────────────────────────┘
```

关键 API：
- `start_draft(seq_id)`：标记进入 draft 模式，记录 `draft_start_num_tokens`
- `append_token(seq_id)`：draft/verify 期间正常追加 KV 条目
- `accept_draft(seq_id, num_accepted)`：回退到 `draft_start + num_accepted` 位置，释放多余 block

**核心要点**：draft 结束后，其 KV cache 完全丢弃；verify 是在 draft 之前的 KV cache 基础上重新推理 `[last_token_before_draft + draft_tokens]`，因此 verify 生成的 KV cache 才是正确的。最终只保留被接受 token 对应的 verify KV cache。

---

## 5. 模型层实现

### 5.1 Decoder Layer

```python
class Qwen3DecoderLayer(nn.Module):
    - input_layernorm (RMSNorm)
    - self_attn (Qwen3Attention: flash_attn)
    - post_attention_layernorm (RMSNorm)
    - mlp (Qwen3MoELayer: gate + experts)
```

### 5.2 MoE Gate

```python
class Qwen3MoEGate(nn.Module):
    gate: Linear(hidden_size, num_experts, bias=False)
    forward: router_logits → topk → softmax normalize
```

### 5.3 Expert Forward

```python
def expert_forward_with_weights(x, gate_weight, up_weight, down_weight):
    return down_proj @ silu(gate_proj @ x) * (up_proj @ x)
```

### 5.4 Fused MoE（可选）

尝试使用 nanovllm 的 `fused_moe_linear`（Triton grouped GEMM kernel）：
1. 将同层全部 expert 权重拼成 `[E, O, I]` 张量
2. 按 expert 排序 token
3. 计算 `m_sizes`（每个 expert 的 token 数）
4. 调用 grouped GEMM：gate_out, up_out, down_out
5. 加权累加

---

## 6. 待优化接口（改造时需保留）

| 接口 | 位置 | 说明 |
|------|------|------|
| **PrefetchStrategy** | `scheduling/base_scheduler.py` | 可插拔的预取预测策略，可替换为更先进的模型 |
| **DraftSchedulingStrategy** | `scheduling/base_scheduler.py` | Draft 调度策略：CPU expert 选择、GPU 替代选择、传输决策、verify 触发条件 |
| **CacheReplacementStrategy** | `scheduling/cache_strategy.py` | Expert cache 替换策略：LRU / LFU / Adaptive / Predictive |
| **AcceptanceStrategy** | `execution/acceptance_strategy.py` | 投机采样接受策略：Standard / Adaptive |
| **select_experts_to_prefetch** | `execution/prefetch_selector.py` | Draft 期间选择哪些 expert 异步传输 |
| **top-c 参数** | 配置项 `draft_top_c` | CPU 执行 expert 数量上限 |
| **max_draft_tokens** | 配置项 | 每轮 draft 最大 token 数 |
| **Operator** | `src/operators/` | CPU/GPU/传输算子抽象层 |

---

## 7. 当前实现的局限性

1. **性能瓶颈**：逐 expert 循环调用 `expert_forward_with_weights`，无法利用 fused kernel 的并行优势
2. **KV Cache**：自行实现的 PagedKVCache 不如 flash_attn 的 paged attention 高效
3. **Attention**：自行实现，缺少 CUDA graph 支持
4. **内存管理**：expert 传输使用 `tensor.to()` 而非预分配 buffer
5. **参数加载**：不支持 packed modules（QKV merge、gate_up merge）
6. **无 Tensor Parallelism**：单卡实现【不需要实现】
7. **无 CUDA Graph**：decode 阶段无法利用 CUDA graph 减少 launch overhead
8. **纯 GPU 场景下也比 nano-vllm 慢 10 倍**：说明基础推理框架本身开销巨大

---

## 8. 总结

on_device_sd/demo 的**核心算法价值**在于：
1. 路由-调度-执行分离架构（RoutingResult → ExpertPlacement → forward_moe）
2. Draft 阶段的 expert 替换策略（减少 CPU 参与和数据搬运）
3. CPU-GPU 并行执行（ThreadPoolExecutor + CUDA stream）
4. 基于激活历史的 expert 预取与缓存管理
5. 投机解码的 KV cache 生命周期管理（start_draft / accept_draft）

**基础推理框架**（attention、KV cache、CUDA graph、weight loading）应该复用 nano-vllm-moe 的高性能实现。
