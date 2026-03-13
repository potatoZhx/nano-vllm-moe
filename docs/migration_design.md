# 基于 nano-vllm-moe 实现 CPU-GPU 异构 MoE 推理设计文档

> 在 nano-vllm-moe 的高性能推理框架上，实现 on_device_sd/demo 的核心功能：  
> CPU-GPU 异构 Expert 执行 + 基于 Expert 替换的投机解码（Speculative Decoding）。

---

## 0. 设计原则

1. **复用 nano-vllm-moe 的高性能基础设施**：flash_attn、Triton grouped GEMM、Paged KV Cache、CUDA graph（标准路径）、Scheduler、Block Manager
2. **最小化侵入性**：通过插件式扩展（新增模块 + hook）而非大规模重写实现
3. **保留优化接口**：所有策略（预取、调度、缓存替换、接受策略）均为可插拔接口
4. **渐进式实现**：先实现全 GPU 的投机解码，再加入 CPU-GPU 异构

---

## 1. nano-vllm-moe 当前架构速览

```
LLMEngine
  ├── Tokenizer
  ├── Scheduler (prefill / decode 调度, block 分配)
  └── ModelRunner
        ├── Model (Qwen3MoeForCausalLM)
        │     ├── Qwen3MoeModel
        │     │     ├── embed_tokens
        │     │     ├── layers[] → Qwen3MoeDecoderLayer
        │     │     │     ├── self_attn (Qwen3MoeAttention, flash_attn)
        │     │     │     ├── input_layernorm / post_attention_layernorm
        │     │     │     └── mlp (Qwen3MoeBlock / Qwen3MoeMLP)
        │     │     │           ├── gate (RowParallelLinear → router)
        │     │     │           ├── gate_up_proj (MergedColumnParallelFusedMoeLinear)
        │     │     │           └── down_proj (RowParallelFusedMoeLinear)
        │     │     └── norm
        │     └── lm_head
        ├── Sampler
        ├── KV Cache (flat tensor, block-managed)
        └── CUDA Graph (decode path)
```

### 关键特点

| 特性 | 实现方式 |
|------|---------|
| MoE 执行 | Triton grouped GEMM (fused_moe_linear)，所有 expert 权重 [E,N,K] 在 GPU |
| KV Cache | 分页管理（BlockManager），flash_attn paged attention |
| Attention | flash_attn_varlen_func (prefill) / flash_attn_with_kvcache (decode) |
| Decode 加速 | CUDA Graph capture |
| 调度 | Prefill 优先，decode batch，preemption |
| 生成循环 | `step()` = schedule → prepare → run_model → sample → postprocess |

---

## 2. 改造目标概述

在 nano-vllm-moe 上新增以下能力：

| 功能 | 说明 |
|------|------|
| **异构参数加载** | 静态参数→GPU，Expert 权重→CPU (pinned)，部分 Expert→GPU Cache |
| **Expert Cache** | GPU Expert 缓存管理，支持 LRU 替换、异步预取 |
| **路由-调度分离** | 路由计算后，由调度器决定每个 Expert 的执行位置 |
| **CPU-GPU 混合 MoE 执行** | GPU 使用 fused kernel，CPU 使用标准 forward，并行执行 |
| **投机解码 (Draft-Verify)** | Draft 阶段使用 Expert 替换加速，Verify 阶段全精度校验 |
| **Draft Scheduler** | 选择 top-c CPU Expert、GPU 替代、传输决策 |
| **Expert 预取** | 基于激活历史的异步 Expert 传输 |
| **KV Cache Draft/Verify 管理** | 支持 Draft KV 回滚和 Verify KV 替换 |

---

## 3. 整体架构设计

### 3.1 新增模块布局

```
nanovllm/
├── engine/
│   ├── llm_engine.py          # 修改：支持 speculative mode
│   ├── model_runner.py        # 修改：增加异构 forward 路径
│   ├── scheduler.py           # 修改：增加 draft/verify 调度
│   ├── block_manager.py       # 修改：支持 draft KV 回滚
│   ├── sequence.py            # 修改：增加 draft 状态管理
│   └── speculative/           # **新增**
│       ├── __init__.py
│       ├── spec_engine.py     # 投机解码引擎 (Draft-Verify 循环)
│       ├── spec_scheduler.py  # 投机解码 Sequence 调度
│       └── acceptance.py      # 接受策略接口 + 实现
│
├── expert/                    # **新增**（运行时 Expert 管理）
│   ├── __init__.py
│   ├── cache.py               # ExpertCacheManager（GPU Expert 缓存）
│   ├── placement.py           # ExpertPlacement 构建（draft/prefill）
│   └── prefetcher.py          # Expert 预取调度
│
├── layers/
│   ├── fuse_moe/
│   │   ├── heterogeneous.py   # **新增**：异构 MoE forward（CPU+GPU 混合执行）
│   │   └── ...                # 现有 Triton kernel 不修改
│   └── ...
│
├── utils/
│   ├── loader.py              # 现有：通用模型加载
│   └── heterogeneous_loader.py # **新增**：异构参数加载器
│
├── models/
│   ├── qwen3_moe.py           # 修改：MoE block 支持异构模式
│   └── ...
│
├── scheduling/                # **新增**
│   ├── __init__.py
│   ├── draft_scheduler.py     # Draft 调度策略接口 + 实现
│   ├── cache_strategy.py      # Expert Cache 替换策略
│   └── prefetch_strategy.py   # Expert 预取策略
│
└── config.py                  # 修改：增加异构推理配置
```

### 3.2 数据流

```
┌──────────────────────────────────────────────────────────────┐
│                    LLMEngine.generate()                       │
│  1. add_request → tokenize → Sequence                        │
│  2. Loop: step()                                             │
└────────────────────────┬─────────────────────────────────────┘
                         │
                    ┌────┴────┐
                    │ 标准模式？│
                    └────┬────┘
             ┌───────────┴───────────┐
             │ Yes                   │ No (Speculative)
             ▼                       ▼
    ┌────────────────┐     ┌──────────────────────┐
    │ 标准 Decode     │     │  SpeculativeEngine   │
    │ (现有路径)      │     │  ┌──────────────────┐│
    │                │     │  │  Draft Loop       ││
    │                │     │  │  (max_draft_tokens)││
    │                │     │  └────────┬─────────┘│
    │                │     │           ▼           │
    │                │     │  ┌──────────────────┐│
    │                │     │  │  Verify           ││
    │                │     │  └────────┬─────────┘│
    │                │     │           ▼           │
    │                │     │  ┌──────────────────┐│
    │                │     │  │  Accept           ││
    │                │     │  └──────────────────┘│
    │                │     └──────────────────────┘
    └────────────────┘
             │                       │
             └───────────┬───────────┘
                         ▼
            ┌────────────────────────┐
            │   ModelRunner.run()    │
            │   (异构 forward 路径)  │
            └────────────┬───────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
  ┌────────────┐  ┌────────────┐  ┌────────────┐
  │ Attention  │  │ Router     │  │ MoE Exec   │
  │ (flash_attn│  │ (gate proj)│  │ (heterogen)│
  │  不修改)    │  │            │  │            │
  └────────────┘  └────────────┘  └────────────┘
                                       │
                              ┌────────┴────────┐
                              │                 │
                              ▼                 ▼
                       ┌────────────┐    ┌────────────┐
                       │ GPU Path   │    │ CPU Path   │
                       │ (Triton    │    │ (per-expert │
                       │  grouped   │    │  forward +  │
                       │  GEMM)     │    │  ThreadPool)│
                       └────────────┘    └────────────┘
```

---

## 4. 各模块详细设计

### 4.1 Config 扩展

```python
@dataclass
class Config:
    # ... 现有字段 ...

    # === 异构推理配置 ===
    enable_heterogeneous: bool = False    # 启用 CPU-GPU 异构模式
    gpu_memory_limit_gb: float = 0.0      # 可用 GPU 显存上限（GB）；0=使用实际可用显存
                                          # 用于在大显存机器上模拟消费级显卡
    expert_gpu_memory_gb: float = 0.0     # Expert Cache 可用 GPU 内存（0=自动计算）
    expert_placement_config: str | None = None  # Expert 初始放置配置文件路径；
                                                # None 则每层固定数量随机选取
    cpu_expert_pin_memory: bool = True    # CPU Expert 使用 pinned memory

    # === 投机解码配置 ===
    enable_speculative: bool = False
    max_draft_tokens: int = 8
    draft_top_c: int = 2                 # CPU 执行 Expert 数上限
    acceptance_threshold: float = 0.7
    acceptance_strategy: str = "standard" # "standard" | "adaptive"

    # === 策略配置（可插拔接口）===
    cache_strategy: str = "lru"           # "lru" | "lfu" | "adaptive"
    prefetch_strategy: str = "simple"     # "simple" | "history"
    draft_scheduler: str = "simple"       # "simple" | "adaptive"
```

### 4.2 异构参数加载器 (`utils/heterogeneous_loader.py`)

#### 为什么不放在 `expert/` 下

`expert/` 包聚焦**运行时** Expert 管理（cache、placement、prefetch），而参数加载是**初始化阶段**的一次性操作，与 `ModelRunner.__init__()` 的流程紧密耦合。nano-vllm-moe 现有的加载入口在 `utils/loader.py`，异构加载器作为其扩展放在 `utils/heterogeneous_loader.py` 更自然。

#### 职责

在原有 `load_model()` 基础上，实现分层加载：
1. 静态参数（attention, embedding, layernorm, router, lm_head, shared expert 若有）→ GPU（复用现有 `load_model` 逻辑）
2. 所有 routed Expert 权重 → CPU pinned memory（`cpu_expert_pool`）
3. 初始化 ExpertCacheManager（分层 slot buffer），根据配置将部分 Expert 从 CPU 加载到 GPU slot 中

#### 与现有代码的关系

现有加载链路：

```
ModelRunner.__init__()
  → load_model(self.model, config.model)              # utils/loader.py
    → Qwen3MoeForCausalLM.load_model(path)            # models/qwen3_moe.py
      → 遍历 safetensors，全部权重 → GPU
```

异构模式下的加载链路：

```
ModelRunner.__init__()
  → heterogeneous_load_model(self.model, config)       # utils/heterogeneous_loader.py
    → Step 1: _load_non_expert_weights(model, path)    # 非 Expert 权重 → GPU
    → Step 2: _load_expert_weights_to_cpu(path)        # Expert 权重 → CPU pinned
    → Step 3: _init_expert_cache(model, config, cpu_pool)  # 创建分层 slot buffer
    → Step 4: _load_initial_placement(config, cache, cpu_pool)  # 初始 Expert → slot
    → 返回 (cpu_expert_pool, expert_cache)
```

#### 关键接口

```python
class HeterogeneousModelLoader:
    """
    异构参数加载器。
    替代标准 load_model，将 Expert 权重分流到 CPU 和 GPU slot buffer。
    """

    def __init__(self, config: Config):
        self.config = config
        self.hf_config = config.hf_config
        self.packed_modules_mapping = Qwen3MoeForCausalLM.packed_modules_mapping

    def load(
        self,
        model: Qwen3MoeForCausalLM,
        path: str,
    ) -> Tuple[Dict[Tuple[int,int], Dict[str, Tensor]], ExpertCacheManager]:
        """
        完整的异构加载流程。

        Returns:
            cpu_expert_pool: {(layer_idx, expert_idx): {"gate_up": Tensor, "down": Tensor}}
            expert_cache: ExpertCacheManager（已完成初始 Expert 加载）
        """
        # Step 1: 非 Expert 权重 → GPU
        self._load_non_expert_weights(model, path)

        # Step 2: Expert 权重 → CPU pinned memory
        cpu_expert_pool = self._load_expert_weights_to_cpu(path)

        # Step 3: 创建 ExpertCacheManager（分配分层 slot buffer）
        expert_cache = self._init_expert_cache(model, cpu_expert_pool)

        # Step 4: 将初始 Expert 从 CPU 加载到 GPU slot
        self._load_initial_placement(expert_cache, cpu_expert_pool)

        return cpu_expert_pool, expert_cache
```

#### Step 1: 非 Expert 权重加载

复用 `Qwen3MoeForCausalLM.load_model()` 的解析逻辑，但跳过 `mlp.experts.*` 权重。

```python
    def _load_non_expert_weights(self, model: Qwen3MoeForCausalLM, path: str):
        """
        加载非 Expert 权重到 GPU：
        - attention (qkv_proj, o_proj)
        - embed_tokens, lm_head
        - layernorm (input_layernorm, post_attention_layernorm, norm)
        - router gate (mlp.gate)
        - shared expert（若存在：mlp.shared_expert.* / mlp.shared_experts.*）

        实现方式：遍历 safetensors 文件，对于每个权重：
        - 如果是 "mlp.experts." 开头的 → 跳过（由 Step 2 处理）
        - 否则 → 使用现有 packed_modules_mapping 和 weight_loader 加载到 GPU
        """
        for file in glob(os.path.join(path, "*.safetensors")):
            with safe_open(file, "pt", "cpu") as f:
                for weight_name in f.keys():
                    if "mlp.experts" in weight_name:
                        continue  # Expert 权重由 Step 2 处理

                    weight_tensor = f.get_tensor(weight_name)

                    # 复用现有 packed_modules + weight_loader 逻辑
                    is_loaded = False
                    for k, (v, shard_id) in self.packed_modules_mapping.items():
                        if k in weight_name:
                            param_name = weight_name.replace(k, v)
                            param = model.get_parameter(param_name)
                            param.weight_loader(param, weight_tensor, shard_id)
                            is_loaded = True
                            break
                    if not is_loaded:
                        param = model.get_parameter(weight_name)
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, weight_tensor)
```

#### Step 2: Expert 权重加载到 CPU

```python
    def _load_expert_weights_to_cpu(
        self, path: str
    ) -> Dict[Tuple[int, int], Dict[str, Tensor]]:
        """
        加载所有 routed Expert 权重到 CPU pinned memory。

        解析 safetensors 中 "mlp.experts.{expert_idx}.{gate/up/down}_proj.weight"，
        需要处理 gate_proj + up_proj 的合并（与 MergedColumnParallelFusedMoeLinear 一致）。

        Returns:
            cpu_expert_pool: {
                (layer_idx, expert_idx): {
                    "gate_up": Tensor [2*intermediate, hidden],  # gate+up 合并
                    "down":    Tensor [hidden, intermediate],
                }
            }
        """
        cpu_pool: Dict[Tuple[int,int], Dict[str, Tensor]] = {}
        # 临时存储未合并的 gate/up
        pending_gate: Dict[Tuple[int,int], Tensor] = {}
        pending_up: Dict[Tuple[int,int], Tensor] = {}

        for file in glob(os.path.join(path, "*.safetensors")):
            with safe_open(file, "pt", "cpu") as f:
                for weight_name in f.keys():
                    if "mlp.experts" not in weight_name:
                        continue

                    weight_tensor = f.get_tensor(weight_name)
                    layer_idx = self._parse_layer_idx(weight_name)
                    expert_idx = self._parse_expert_idx(weight_name)
                    key = (layer_idx, expert_idx)

                    if key not in cpu_pool:
                        cpu_pool[key] = {}

                    if "down_proj" in weight_name:
                        cpu_pool[key]["down"] = weight_tensor.pin_memory()
                    elif "gate_proj" in weight_name:
                        pending_gate[key] = weight_tensor
                    elif "up_proj" in weight_name:
                        pending_up[key] = weight_tensor

        # 合并 gate + up → gate_up（与 MergedColumnParallelFusedMoeLinear 一致）
        for key in pending_gate:
            gate = pending_gate[key]
            up = pending_up[key]
            gate_up = torch.cat([gate, up], dim=0)  # [2*intermediate, hidden]
            cpu_pool[key]["gate_up"] = gate_up.pin_memory()

        return cpu_pool

    @staticmethod
    def _parse_layer_idx(weight_name: str) -> int:
        """从 'model.layers.{idx}.mlp.experts...' 中提取 layer_idx"""
        parts = weight_name.split(".")
        return int(parts[parts.index("layers") + 1])

    @staticmethod
    def _parse_expert_idx(weight_name: str) -> int:
        """从 '...mlp.experts.{idx}...' 中提取 expert_idx"""
        parts = weight_name.split(".")
        return int(parts[parts.index("experts") + 1])
```

#### Step 3: 创建 ExpertCacheManager

```python
    def _init_expert_cache(
        self,
        model: Qwen3MoeForCausalLM,
        cpu_expert_pool: Dict,
    ) -> ExpertCacheManager:
        """
        根据可用 GPU 显存创建分层 slot buffer。

        1. 计算 slots_per_layer（见 §5.3）
        2. 为每层分配 gate_up buffer [S, 2*inter, hidden] + down buffer [S, hidden, inter]
        3. 创建空的 slot_to_expert / expert_to_slot 映射
        """
        model_config = self.hf_config
        slots_per_layer = compute_slots_per_layer(self.config, model_config)

        expert_cache = ExpertCacheManager(
            model_config=model_config,
            config=self.config,
            cpu_expert_pool=cpu_expert_pool,
            slots_per_layer=slots_per_layer,
        )
        return expert_cache
```

#### Step 4: 初始 Expert 加载到 GPU Slot

```python
    def _load_initial_placement(
        self,
        expert_cache: ExpertCacheManager,
        cpu_expert_pool: Dict,
    ):
        """
        将初始 Expert 从 CPU 加载到 GPU slot buffer。

        读取策略：
        1. config.expert_placement_config 存在 → 从 YAML 文件读取
        2. 否则 → 每层随机选取 S 个 expert

        对于每个 (layer_idx, expert_idx)：
          expert_cache.cache_expert(layer_idx, expert_idx)
            → cpu_pool 中取出权重
            → copy_ 到 slot buffer 对应位置
            → 更新映射表
        """
        placement = self._resolve_placement(expert_cache.slots_per_layer)

        for layer_idx in range(self.hf_config.num_hidden_layers):
            expert_indices = placement.get(layer_idx, [])
            for expert_idx in expert_indices:
                expert_cache.cache_expert(layer_idx, expert_idx)

        torch.cuda.synchronize()

    def _resolve_placement(self, slots_per_layer: int) -> Dict[int, List[int]]:
        """
        解析 Expert 初始放置配置。

        Returns:
            {layer_idx: [expert_indices]}  每层需要初始加载的 expert 列表
        """
        if self.config.expert_placement_config:
            import yaml
            with open(self.config.expert_placement_config) as f:
                raw = yaml.safe_load(f)
            placement = {}
            for layer_idx in range(self.hf_config.num_hidden_layers):
                if layer_idx in raw.get("placement", {}):
                    indices = raw["placement"][layer_idx][:slots_per_layer]
                else:
                    indices = random.sample(
                        range(self.hf_config.num_experts), slots_per_layer
                    )
                placement[layer_idx] = indices
            return placement
        else:
            return {
                layer_idx: random.sample(
                    range(self.hf_config.num_experts), slots_per_layer
                )
                for layer_idx in range(self.hf_config.num_hidden_layers)
            }
```

配置文件格式示例（YAML）：
```yaml
# expert_placement.yaml
placement:
  0: [0, 3, 7, 12, 50, 100, 115, 127]
  1: [1, 5, 9, 20, 60, 88, 110, 125]
  # 未指定的层使用随机选取
```

#### ModelRunner 集成

```python
# engine/model_runner.py 中 __init__ 的修改
class ModelRunner:
    def __init__(self, config, rank, event):
        ...
        self.model = self.MODEL_TYPE_DICT[hf_config.model_type](hf_config)

        if config.enable_heterogeneous:
            from nanovllm.utils.heterogeneous_loader import HeterogeneousModelLoader
            loader = HeterogeneousModelLoader(config)
            self.cpu_expert_pool, self.expert_cache = loader.load(self.model, config.model)
        else:
            load_model(self.model, config.model)
            self.cpu_expert_pool = None
            self.expert_cache = None

        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        ...
```

#### Expert 权重格式说明

nano-vllm-moe 中 Expert 权重的组织方式（与 `cpu_expert_pool` 的对应关系）：

```
safetensors 文件中:
  model.layers.{L}.mlp.experts.{E}.gate_proj.weight  → [intermediate, hidden]
  model.layers.{L}.mlp.experts.{E}.up_proj.weight    → [intermediate, hidden]
  model.layers.{L}.mlp.experts.{E}.down_proj.weight  → [hidden, intermediate]

cpu_expert_pool[(L, E)]:
  "gate_up" → cat(gate_proj, up_proj, dim=0) → [2*intermediate, hidden]
  "down"    → down_proj                      → [hidden, intermediate]

ExpertCacheManager slot buffer:
  gate_up_buffers[L]: [S, 2*intermediate, hidden]   # slot buffer
  down_buffers[L]:    [S, hidden, intermediate]      # slot buffer

加载到 slot:
  gate_up_buffers[L][slot_idx].copy_(cpu_pool[(L, E)]["gate_up"])
  down_buffers[L][slot_idx].copy_(cpu_pool[(L, E)]["down"])
```

> **注意**：nano-vllm-moe 原始代码中 `MergedColumnParallelFusedMoeLinear` 的 `weight` 形状为 `[E, 2*intermediate/tp_size, hidden]`，`RowParallelFusedMoeLinear` 的 `weight` 形状为 `[E, hidden, intermediate/tp_size]`。当前单卡场景 `tp_size=1`，因此 `cpu_expert_pool` 中存储的维度与 slot buffer 完全一致。

#### Routed Expert 的 GPU 存储方式

初始加载到 GPU 的 routed expert **直接写入 ExpertCacheManager 的分层 slot buffer** 中，作为普通 cached 状态（可被后续驱逐替换）。这样做的好处是：
- Expert 权重始终在分层连续 buffer `[S, N, K]` 中，可直接被 fused MoE kernel 使用
- 避免了静态参数与 cache buffer 之间的数据不一致问题
- 统一了 expert 的管理路径（加载、驱逐、替换均通过 ExpertCacheManager）

### 4.3 Expert Cache Manager (`expert/cache.py`)

#### 核心设计：分层固定 Slot

每个 MoE 层维护一个固定大小的 slot buffer `[S, N, K]`（S = 该层的 slot 数量），用于存放当前在 GPU 上的 expert 权重。这是整个异构推理系统的关键设计：

```
Layer 0:  [slot_0, slot_1, ..., slot_{S-1}]  gate_up: [S, 2*inter, hidden]  down: [S, hidden, inter]
Layer 1:  [slot_0, slot_1, ..., slot_{S-1}]
  ...
Layer 47: [slot_0, slot_1, ..., slot_{S-1}]
```

- 每个 slot 可存放该层任意一个 expert
- 当路由选中的 expert 不在任何 slot 中时，驱逐一个 slot 中的 expert，替换加载新 expert
- Buffer 在引擎启动时一次性分配，运行期间不增减
- Fused MoE kernel 直接使用 `[S, N, K]` buffer，S 是 `NUM_EXPERTS` 参数，m_sizes 长度为 S

**为何不使用 `[E, N, K]`（E=128 全量 buffer）**：
- 对于 Qwen3-30B-A3B（E=128, hidden=2048, intermediate=768, bf16），每层 gate_up+down 完整 buffer ≈ 128 × (2×768×2048 + 2048×768) × 2 bytes ≈ 0.9 GB
- 48 层共 ≈ 43 GB，**超出消费级显卡显存**
- 分层固定 slot（如 S=16）时，48 层仅需 ≈ 5.4 GB

#### Slot 数量计算

```python
def compute_slots_per_layer(config, model_config) -> int:
    """
    根据可用 GPU 显存计算每层 slot 数量。

    available = gpu_memory_limit_gb（或实际可用显存）
                - 静态参数内存（attention, embed, norm, router, lm_head, shared expert）
                - KV Cache 预留
                - 系统开销预留
    per_expert_size = (gate_up_weight_size + down_weight_size) × dtype_bytes
    total_expert_budget = available / per_expert_size
    slots_per_layer = total_expert_budget // num_moe_layers
    """
```

#### 关键接口

```python
class ExpertCacheManager:
    """
    分层固定 Slot 的 GPU Expert 缓存管理器。
    """

    def __init__(self, model_config, config, cpu_expert_pool):
        self.slots_per_layer = compute_slots_per_layer(config, model_config)
        self.num_layers = model_config.num_hidden_layers
        self.cpu_expert_pool = cpu_expert_pool
        self.strategy = create_cache_strategy(config.cache_strategy)
        self.transfer_stream = torch.cuda.Stream()

        # 分层 slot buffer（一次性分配连续内存）
        # gate_up_buffers[layer_idx]: Tensor [S, 2*intermediate, hidden]
        # down_buffers[layer_idx]:    Tensor [S, hidden, intermediate]
        self.gate_up_buffers: Dict[int, Tensor] = {}
        self.down_buffers: Dict[int, Tensor] = {}

        # 分层映射表
        # slot_to_expert[layer_idx][slot_idx] = expert_idx 或 -1（空 slot）
        # expert_to_slot[layer_idx][expert_idx] = slot_idx 或 None
        self.slot_to_expert: Dict[int, list[int]] = {}
        self.expert_to_slot: Dict[int, Dict[int, int]] = {}

    def is_cached(self, layer_idx: int, expert_idx: int) -> bool:
        """检查 Expert 是否在该层的某个 slot 中"""
        return expert_idx in self.expert_to_slot[layer_idx]

    def get_cached_experts(self, layer_idx: int) -> set[int]:
        """获取指定层所有已缓存的 Expert 索引"""

    def get_slot_idx(self, layer_idx: int, expert_idx: int) -> int | None:
        """获取 Expert 对应的 slot index"""
        return self.expert_to_slot[layer_idx].get(expert_idx)

    def cache_expert(self, layer_idx: int, expert_idx: int) -> int:
        """
        将 Expert 从 CPU 加载到该层的一个 slot（同步）。
        如果所有 slot 已满，根据策略驱逐一个。
        返回分配到的 slot_idx。
        """

    def cache_expert_async(self, layer_idx: int, expert_idx: int):
        """异步加载 Expert 到 slot（使用 transfer_stream）"""

    def evict_from_slot(self, layer_idx: int, slot_idx: int):
        """驱逐指定 slot 中的 expert（只更新映射，不清零 buffer）"""

    def get_layer_buffers(self, layer_idx: int) -> Tuple[Tensor, Tensor]:
        """获取指定层的 slot buffer（gate_up_proj, down_proj），用于传入 fused kernel"""

    def complete_transfers(self) -> list:
        """
        轮询并完成待定的异步传输。
        - 检查 transfer_stream 上的 pending events
        - 对已完成的传输，更新 slot_to_expert / expert_to_slot 映射
        - 返回已完成传输的 (layer_idx, expert_idx) 列表
        """

    def build_m_sizes(self, layer_idx: int, selected_experts: Tensor,
                      routing_weights: Tensor) -> Tensor:
        """
        根据当前层的 slot 映射，将路由结果映射为 m_sizes[S]。
        未缓存的 expert 对应的 token 不计入（由 CPU 路径处理或替换）。
        """
```

#### 缓存操作详细流程

```python
def cache_expert(self, layer_idx, expert_idx):
    slot_idx = self._find_free_slot(layer_idx)
    if slot_idx is None:
        # 所有 slot 已满，驱逐一个
        victim_slot = self.strategy.select_victim(
            layer_idx, self.slot_to_expert[layer_idx]
        )
        self.evict_from_slot(layer_idx, victim_slot)
        slot_idx = victim_slot

    # 从 CPU 拷贝到 slot buffer 的对应位置
    cpu_gate_up = self.cpu_expert_pool[(layer_idx, expert_idx)]["gate_up"]
    cpu_down = self.cpu_expert_pool[(layer_idx, expert_idx)]["down"]
    self.gate_up_buffers[layer_idx][slot_idx].copy_(cpu_gate_up, non_blocking=True)
    self.down_buffers[layer_idx][slot_idx].copy_(cpu_down, non_blocking=True)

    # 更新映射
    self.slot_to_expert[layer_idx][slot_idx] = expert_idx
    self.expert_to_slot[layer_idx][expert_idx] = slot_idx
    self.strategy.on_access((layer_idx, expert_idx))
    return slot_idx
```

**优势**：
- 每层 buffer 连续，可直接传入 Triton grouped GEMM kernel（`[S, N, K]`）
- Cache 命中时 0 额外开销（权重已在正确 slot 位置）
- Slot 数量固定，无运行时内存分配/释放
- 驱逐仅更新映射表，不清零 buffer（新 expert 会覆盖旧数据）
- Kernel 中 m_sizes=0 的 slot 自动跳过

### 4.4 Expert Placement 构建 (`expert/placement.py`)

#### 核心数据结构

```python
@dataclass
class MoEExecutionPlan:
    """MoE 层的执行计划（基于分层 slot buffer 设计）"""
    layer_idx: int

    # GPU 执行信息（基于分层 slot buffer [S, N, K]）
    gpu_slot_indices: Tensor        # [M * topk] remapped slot index（-1 表示非 GPU 执行）
    gpu_mask: Tensor                # [M * topk] bool，True 表示由 GPU 执行
    gpu_m_sizes: Tensor             # [S] 每个 slot 的 token 数（未使用的 slot 为 0）
    gpu_sort_idx: Tensor            # 按 slot 排序的索引
    gpu_inv_sort_idx: Tensor        # 反排序索引
    gpu_token_map: Tensor           # GPU 结果到 output 的映射
    gpu_weights: Tensor             # GPU 路径的路由权重
    gpu_token_count: int            # GPU 处理的 token 总数

    # CPU 执行信息
    cpu_expert_indices: list[int]   # 在 CPU 执行的 expert 原始索引
    cpu_token_map: Tensor           # CPU 结果到 output 的映射
    cpu_weights: Tensor             # CPU 路径的路由权重
    cpu_token_count: int            # CPU 处理的 token 总数

    # 替换映射（draft 阶段使用）
    substitution_map: dict[int, int]  # {原 expert_idx: 替代 expert_idx（已缓存在同层 slot 中）}

    # 原始路由信息（用于激活统计和预取决策）
    selected_experts: Tensor        # [M * topk] 原始 expert index（替换前）
    routing_weights: Tensor         # [M, topk] 原始路由权重
```

#### 构建流程

```python
def build_prefill_plan(
    layer_idx: int,
    selected_experts: Tensor,       # [M * topk]
    routing_weights: Tensor,        # [M, topk]
    expert_cache: ExpertCacheManager,
    cpu_expert_pool: dict,
    num_experts: int,
) -> MoEExecutionPlan:
    """
    Prefill/Verify 阶段：
    - 缓存命中 → GPU 执行
    - 缓存未命中 → CPU 执行（无替换）
    """

def build_draft_plan(
    layer_idx: int,
    selected_experts: Tensor,
    routing_weights: Tensor,
    expert_cache: ExpertCacheManager,
    cpu_expert_pool: dict,
    draft_scheduler: DraftScheduler,
    num_experts: int,
    top_c: int,
) -> MoEExecutionPlan:
    """
    Draft 阶段：
    1. 检查哪些 expert 在 GPU cache 中
    2. 未缓存的 expert 中选 top-c 个在 CPU 执行
    3. 其余未缓存 expert → 用 GPU cache 中的 expert 替代
    4. 构建 MoEExecutionPlan
    """
```

### 4.5 异构 MoE Forward (`layers/fuse_moe/heterogeneous.py`)

这是改造中最关键的模块，实现 CPU-GPU 混合 Expert 执行。详细的 slot 映射和 kernel 适配见 §5。

```python
def heterogeneous_moe_forward(
    hidden_states: Tensor,          # [M, hidden_dim]
    plan: MoEExecutionPlan,
    expert_cache: ExpertCacheManager,
    cpu_expert_pool: dict,
    layer_idx: int,
    num_selected: int,              # top-k
) -> Tensor:
    """
    CPU-GPU 混合 MoE Forward。

    GPU 路径：使用分层 slot buffer [S, N, K] + Triton grouped GEMM kernel
    CPU 路径：per-expert forward + ThreadPoolExecutor
    两路并行执行，在结果聚合时同步。
    """
    M, D = hidden_states.shape
    output = torch.zeros(M, D, device=hidden_states.device, dtype=hidden_states.dtype)

    # === GPU 路径：Fused Kernel via Slot Buffer ===
    if plan.gpu_token_count > 0:
        expanded = hidden_states.repeat_interleave(num_selected, dim=0)
        gpu_expanded = expanded[plan.gpu_sort_idx]

        gate_up_buf, down_buf = expert_cache.get_layer_buffers(layer_idx)
        gate_up_out = fused_moe_linear(gpu_expanded, gate_up_buf, plan.gpu_m_sizes)
        expert_out = fused_moe_linear(act_fn(gate_up_out), down_buf, plan.gpu_m_sizes)

        expert_out = expert_out[plan.gpu_inv_sort_idx]
        _scatter_weighted_add(output, expert_out, plan.gpu_token_map, plan.gpu_weights)

    # === CPU 路径：并行 Per-Expert Forward ===
    if plan.cpu_token_count > 0:
        cpu_results = _cpu_parallel_forward(
            hidden_states, plan, cpu_expert_pool, layer_idx
        )
        _scatter_weighted_add(output, cpu_results, plan.cpu_token_map, plan.cpu_weights)

    return output
```

> **GPU 路径要点**：`gate_up_buf` 和 `down_buf` 是 ExpertCacheManager 管理的分层连续 buffer `[S, N, K]`，`plan.gpu_m_sizes` 长度为 S（slot 数），`plan.gpu_sort_idx` / `gpu_inv_sort_idx` 基于 slot index 排序。详见 §5.4-§5.6。

#### CPU Parallel Forward 详细设计

```python
def _cpu_parallel_forward(hidden_states, plan, cpu_expert_pool, layer_idx):
    """
    在 CPU 上并行执行 top-c Expert。
    使用 ThreadPoolExecutor + pinned memory 实现 CPU 多线程计算 + GPU 异步传输。
    """
    results = {}

    def _execute_one_expert(expert_idx, token_indices, weights):
        params = cpu_expert_pool[(layer_idx, expert_idx)]
        x = hidden_states[token_indices].to("cpu", non_blocking=True)
        torch.cuda.current_stream().synchronize()

        gate_out = F.linear(x, params["gate_proj"])
        up_out = F.linear(x, params["up_proj"])
        inter = F.silu(gate_out) * up_out
        out = F.linear(inter, params["down_proj"])

        out_gpu = out.pin_memory().to(hidden_states.device, non_blocking=True)
        return expert_idx, token_indices, weights, out_gpu

    with ThreadPoolExecutor(max_workers=min(len(plan.cpu_expert_indices), 4)) as executor:
        futures = [
            executor.submit(_execute_one_expert, eid, tidx, w)
            for eid, tidx, w in plan.iter_cpu_tasks()
        ]
        for future in futures:
            eid, tidx, w, out = future.result()
            results[eid] = (tidx, w, out)

    return results
```

### 4.6 投机解码引擎 (`engine/speculative/spec_engine.py`)

#### 核心流程

```python
class SpeculativeEngine:
    """
    管理 Draft-Verify 循环。
    嵌入到 LLMEngine.step() 的 decode 路径中。
    """

    def __init__(self, model_runner, config, expert_cache, scheduler):
        self.model_runner = model_runner
        self.config = config
        self.expert_cache = expert_cache
        self.draft_scheduler = create_draft_scheduler(config.draft_scheduler)
        self.acceptance_strategy = create_acceptance_strategy(config)
        self.prefetcher = create_prefetcher(config, expert_cache)
        self.max_draft_tokens = config.max_draft_tokens

    def speculative_step(self, seqs: list[Sequence]) -> list[int]:
        """
        一次完整的 Draft-Verify 步骤。

        Returns:
            每个 seq 本轮接受的 token 数
        """
        # Phase 1: Draft（内含异步预取，边 draft 边预取 verify 所需 expert）
        draft_tokens, draft_activations = self._draft(seqs)

        # Phase 2: Verify
        verify_logits = self._verify(seqs, draft_tokens)

        # Phase 3: Accept
        accepted_counts = self._accept(seqs, draft_tokens, verify_logits)

        return accepted_counts
```

#### Draft 阶段

异步预取发生在 draft 循环**内部**：每个 step（或每层）路由结果出来后，根据激活信息预测 verify 阶段可能需要的 expert，发起异步 CPU→GPU 传输。理想情况下，这些传输在 verify 的第一层 attention 结束前完成。

```python
def _draft(self, seqs):
    draft_tokens_per_seq = {seq.seq_id: [] for seq in seqs}
    all_activations = []

    for step in range(self.max_draft_tokens):
        active_seqs = [s for s in seqs if not s.is_draft_finished]
        if not active_seqs:
            break

        # 准备输入（上一步的 draft token 或 last accepted token）
        input_ids, positions = self._prepare_draft_input(active_seqs)

        # Forward（异构模式）
        logits, step_activations = self.model_runner.run_draft(
            active_seqs, input_ids, positions, self.expert_cache, self.draft_scheduler
        )

        all_activations.extend(step_activations)

        # 根据 draft 激活预测 verify 所需 expert，发起异步传输
        # 这里预取的是 verify 阶段（全精度执行）可能需要但当前不在 GPU cache 中的 expert
        self._schedule_prefetch_for_verify(all_activations)

        # 完成已就绪的异步传输（更新 slot 映射）
        self.expert_cache.complete_transfers()

        # Sample
        token_ids = self.model_runner.sampler(logits, temperatures)

        # Append draft tokens
        for seq, token_id in zip(active_seqs, token_ids):
            seq.append_draft_token(token_id)
            self.scheduler.append_draft_kv(seq)

    # Draft 完成后，同步等待所有待定传输完成（可选策略：
    # 也可以在 verify 的第一层 attention 期间再 sync）
    self.expert_cache.complete_transfers()

    return draft_tokens_per_seq, all_activations
```

> **关于 `complete_transfers()`**：该方法轮询 `transfer_stream` 上的 pending events，检查哪些异步 CPU→GPU 传输已完成。对已完成的传输，更新 slot_to_expert / expert_to_slot 映射表，使后续的 `is_cached()` 查询能感知到新加载的 expert。未完成的传输继续在后台执行，不阻塞当前 CUDA stream 上的 GPU 计算。

> **传输与计算的冲突处理**：异步传输使用独立的 `transfer_stream`，与 GPU 计算所在的 default stream 并行。但 PCIe 带宽共享可能影响 GPU 计算性能。简化实现方案：可在每个 draft step 结束时或 draft 全部结束后调用 `transfer_stream.synchronize()` 等待传输完成，避免传输与计算在 PCIe 上竞争。

#### Verify 阶段

```python
def _verify(self, seqs, draft_tokens):
    # 构造 verify 输入：last_accepted_token + draft_tokens
    input_ids, positions = self._prepare_verify_input(seqs, draft_tokens)

    # Forward（使用 prefill-like 模式，无替换）
    logits = self.model_runner.run_verify(
        seqs, input_ids, positions, self.expert_cache
    )

    return logits
```

#### Accept 阶段

```python
def _accept(self, seqs, draft_tokens, verify_logits):
    accepted_counts = []
    for seq in seqs:
        seq_draft = draft_tokens[seq.seq_id]
        seq_logits = verify_logits[seq.seq_id]

        if seq.is_greedy:
            # Argmax 比较
            verify_argmax = seq_logits.argmax(dim=-1)
            num_accepted = 0
            for i, (draft_tok, verify_tok) in enumerate(zip(seq_draft, verify_argmax)):
                if draft_tok != verify_tok:
                    break
                num_accepted += 1
            # 额外接受 verify 的下一个 token
            if num_accepted < len(verify_argmax):
                seq.append_token(verify_argmax[num_accepted].item())
        else:
            # 使用可插拔的 AcceptanceStrategy
            result = self.acceptance_strategy.accept(seq_draft, seq_logits, seq.temperature)
            num_accepted = result["num_accepted"]

        # 更新 Sequence 状态和 KV Cache
        seq.accept_draft(num_accepted)
        self.scheduler.accept_draft_kv(seq, num_accepted)
        accepted_counts.append(num_accepted)

    return accepted_counts
```

### 4.7 KV Cache Draft/Verify 支持

每一轮 draft-verify 的 KV cache 生命周期如下：

```
假设上一轮 prefill/verify 结束后，KV cache 中已有 token 序列 [t0, t1, ..., tN]
其中 tN 是上一轮生成的最后一个 token（即本轮 draft 的输入）

┌─ Draft Phase ──────────────────────────────────────────────────┐
│ start_draft(seq)：记录 draft_start = N+1                        │
│ 自回归生成 draft tokens [d0, d1, ..., dK-1]                      │
│ KV cache 临时扩展为 [t0..tN, d0..dK-1]                           │
└───────────────────────────────────────────────────────────────┘
                          │
                          ▼ Draft 结束，回滚 draft 阶段的 KV cache
                            KV cache 回退到 [t0..tN]
                          │
┌─ Verify Phase ─────────────────────────────────────────────────┐
│ 输入：[tN] + [d0, d1, ..., dK-1]（上一轮最后 token + draft tokens）│
│ 基于 [t0..tN-1] 的已有 KV cache，以 prefill 模式                  │
│ 一次性处理 K+1 个 token，生成 verify_logits                       │
│ KV cache 扩展为 [t0..tN, d0..dK-1]                               │
│ （此时 KV 是 verify 的全精度 attention 生成的，与 draft 不同）       │
└───────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─ Accept Phase ─────────────────────────────────────────────────┐
│ 假设接受了 [d0..dM-1]（M 个 token）                               │
│ accept_draft(seq, M)：                                          │
│   - 保留 verify KV cache [t0..tN, d0..dM-1]                     │
│   - 丢弃 [dM..dK-1] 对应的 KV 条目（释放多余 block）              │
│ verify 在位置 M 的 logits argmax/sample 即为新生成的 token          │
│ 将其追加到序列，KV cache 最终为 [t0..tN, d0..dM-1, new_token]     │
└───────────────────────────────────────────────────────────────┘
```

**核心要点**：draft 的 KV cache 在 draft 结束后完全丢弃；verify 基于 draft 之前的 KV cache 重新推理，生成正确的 KV；最终只保留被接受 token 对应的 verify KV cache。

#### BlockManager 扩展

```python
class BlockManager:
    # ... 现有代码 ...

    def start_draft(self, seq: Sequence):
        """记录 draft 开始前的 block 状态，用于 draft 结束后回滚"""
        seq._draft_start_num_tokens = seq.num_tokens
        seq._draft_start_num_blocks = len(seq.block_table)
        seq._draft_start_last_block_tokens = seq.last_block_num_tokens

    def append_draft_token(self, seq: Sequence):
        """为 draft token 分配 KV cache slot（可能分配新 block）"""
        self.may_append(seq)

    def rollback_draft(self, seq: Sequence):
        """
        Draft 结束后，回滚所有 draft token 的 KV 条目。
        KV cache 恢复到 draft 开始前的状态。
        Verify 阶段将在此基础上重新写入。
        """
        target_tokens = seq._draft_start_num_tokens
        while seq.num_blocks > (target_tokens + self.block_size - 1) // self.block_size:
            freed_block = seq.block_table.pop()
            self.free_block(freed_block)
        seq.num_tokens = target_tokens

    def accept_draft(self, seq: Sequence, num_accepted: int):
        """
        Verify 完成后，保留前 num_accepted 个 token 的 KV 条目。
        丢弃剩余 verify KV 条目，释放多余 block。
        """
        target_tokens = seq._draft_start_num_tokens + num_accepted
        while seq.num_blocks > (target_tokens + self.block_size - 1) // self.block_size:
            freed_block = seq.block_table.pop()
            self.free_block(freed_block)
        seq.num_tokens = target_tokens
```

### 4.8 Sequence 扩展

```python
class Sequence:
    # ... 现有字段 ...

    # === Draft 相关 ===
    draft_token_ids: list[int]
    _draft_start_num_tokens: int
    is_drafting: bool

    def start_draft(self):
        self.draft_token_ids = []
        self._draft_start_num_tokens = self.num_tokens
        self.is_drafting = True

    def append_draft_token(self, token_id: int):
        self.draft_token_ids.append(token_id)
        self.append_token(token_id)  # 同时更新主 token 序列

    def accept_draft(self, num_accepted: int):
        """接受前 num_accepted 个 draft token"""
        # 回滚到 draft 起点 + accepted
        target = self._draft_start_num_tokens + num_accepted
        self.token_ids = self.token_ids[:target]
        self.num_tokens = target
        self.last_token = self.token_ids[-1]
        self.draft_token_ids = []
        self.is_drafting = False

    @property
    def is_draft_finished(self):
        return not self.is_drafting or self.is_finished
```

### 4.9 ModelRunner 扩展

```python
class ModelRunner:
    # ... 现有代码 ...

    def run_draft(self, seqs, input_ids, positions, expert_cache, draft_scheduler):
        """
        Draft forward：使用异构 MoE 执行（含 expert 替换）。
        不使用 CUDA Graph（因为执行路径动态变化）。
        """
        # 设置 Context
        self.prepare_decode(seqs)

        # 逐层 forward，在 MoE 层插入 placement 逻辑
        hidden_states = self.model.model.embed_tokens(input_ids)
        activations = []

        for layer_idx, layer in enumerate(self.model.model.layers):
            # Attention（使用标准 flash_attn 路径）
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            hidden_states = layer.self_attn(positions, hidden_states)
            hidden_states = residual + hidden_states

            # MoE：路由-调度-异构执行
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)

            if isinstance(layer.mlp, Qwen3MoeBlock):
                # 路由
                router_logits = layer.mlp.gate(hidden_states)
                routing_weights, selected_experts = self._compute_routing(
                    router_logits, layer.mlp.num_selected, layer.mlp.norm_topk_prob
                )
                # 构建执行计划
                plan = build_draft_plan(
                    layer_idx, selected_experts, routing_weights,
                    expert_cache, self.cpu_expert_pool,
                    draft_scheduler, layer.mlp.num_experts,
                    self.config.draft_top_c,
                )
                activations.append((layer_idx, plan))
                # 异构执行
                hidden_states = heterogeneous_moe_forward(
                    hidden_states, plan, expert_cache, self.cpu_expert_pool,
                    layer_idx, layer.mlp.num_selected,
                )
            else:
                hidden_states = layer.mlp(hidden_states)

            hidden_states = residual + hidden_states

        hidden_states = self.model.model.norm(hidden_states)
        logits = self.model.compute_logits(hidden_states)

        reset_context()
        return logits, activations

    def run_verify(self, seqs, input_ids, positions, expert_cache):
        """
        Verify forward：类似 prefill，使用完整路由（无替换）。
        """
        # 类似 run_draft 但使用 build_prefill_plan（无替换）
        ...

    def run_standard(self, seqs, is_prefill):
        """
        标准路径：所有 expert 在 GPU。
        使用 CUDA Graph 加速（decode），原有逻辑不变。
        """
        return self.run(seqs, is_prefill)
```

### 4.10 调度策略接口

#### Draft Scheduler (`scheduling/draft_scheduler.py`)

```python
class DraftScheduler(ABC):
    """Draft 阶段调度策略接口"""

    @abstractmethod
    def select_cpu_experts(
        self,
        uncached_experts: list[int],
        routing_weights: Tensor,
        selected_experts: Tensor,
        top_c: int,
    ) -> list[int]:
        """从未缓存的 expert 中选择 top-c 个在 CPU 执行"""

    @abstractmethod
    def select_gpu_substitutes(
        self,
        need_substitution: list[int],
        cached_experts: set[int],
        all_experts: list[int],
    ) -> dict[int, int]:
        """为需要替换的 expert 选择 GPU cache 中的替代"""

    @abstractmethod
    def select_experts_to_transfer(
        self,
        recent_activations: list,
        cached_experts: set,
        cache_capacity: int,
    ) -> list[tuple[int, int]]:
        """选择应异步传输到 GPU 的 expert"""


class SimpleDraftScheduler(DraftScheduler):
    """
    简单策略：
    - CPU：选激活分数最高的 top-c
    - GPU 替代：随机选 cached expert（接口预留，可替换为基于相似度的策略）
    - 传输：按激活频率排序
    """
```

#### Cache Replacement Strategy (`scheduling/cache_strategy.py`)

```python
class CacheReplacementStrategy(ABC):
    @abstractmethod
    def select_victim(self, cached: list, pinned: set) -> tuple[int, int] | None:
        """选择驱逐对象"""

    @abstractmethod
    def on_access(self, key): ...
    @abstractmethod
    def on_insert(self, key): ...
    @abstractmethod
    def on_evict(self, key): ...

class LRUCacheStrategy(CacheReplacementStrategy): ...
class LFUCacheStrategy(CacheReplacementStrategy): ...
class AdaptiveCacheStrategy(CacheReplacementStrategy): ...
```

#### Prefetch Strategy (`scheduling/prefetch_strategy.py`)

```python
class PrefetchStrategy(ABC):
    @abstractmethod
    def predict_next_experts(
        self, layer_idx: int, current_activations: list
    ) -> list[tuple[int, int]]:
        """预测下一层可能激活的 expert"""

class SimplePrefetchStrategy(PrefetchStrategy):
    """时序局部性：预测下层激活与当前层相同 index 的 expert"""

class HistoryPrefetchStrategy(PrefetchStrategy):
    """基于历史共现矩阵预测"""
```

#### Acceptance Strategy (`engine/speculative/acceptance.py`)

```python
class AcceptanceStrategy(ABC):
    @abstractmethod
    def accept(self, draft_tokens: Tensor, verify_logits: Tensor,
               temperature: float) -> dict:
        """返回 {num_accepted, accepted_tokens, ...}"""

class StandardAcceptance(AcceptanceStrategy):
    """P_verify(token) >= threshold"""

class AdaptiveAcceptance(AcceptanceStrategy):
    """动态调整 threshold"""
```

---

## 5. Fused MoE Kernel 适配（重点）

### 5.1 问题分析

nano-vllm-moe 的 `_grouped_gemm_forward_kernel` 要求权重为连续的 `[E, N, K]` 张量，`m_sizes` 为长度 E 的整数向量。kernel 中 `NUM_EXPERTS` 是 `tl.constexpr`（编译时常量），遍历每个 expert 并跳过 `m_size = 0` 的 expert。

在异构模式下，每层仅有部分 expert（S 个）在 GPU 上。核心问题是：如何让 fused kernel 高效地处理这 S 个 expert。

### 5.2 方案：分层固定 Slot Buffer

根据可用 GPU 显存，为每个 MoE 层分配固定数量 S 的 expert slot，形成连续 buffer `[S, N, K]`，直接作为 fused kernel 的权重输入。

```
┌── Layer 0 ──────────────────────────────────────────────────┐
│ gate_up_buffer: [S, 2*intermediate, hidden]  （连续 GPU 内存）│
│ down_buffer:    [S, hidden, intermediate]    （连续 GPU 内存）│
│                                                              │
│ slot_to_expert: [expert_42, expert_7, ..., expert_99]        │
│ expert_to_slot: {42→0, 7→1, ..., 99→S-1}                    │
└──────────────────────────────────────────────────────────────┘
│  同理 Layer 1 .. Layer 47                                     │
```

**核心思路**：
1. 路由结果中的 `selected_experts`（原始 expert index）通过 `expert_to_slot` 映射为 `slot_index`（0..S-1）
2. 构建长度为 S 的 `m_sizes`，未缓存的 expert 不映射（由 CPU 路径处理或替换为已缓存 expert 的 slot）
3. 调用现有 grouped GEMM kernel，传入 `[S, N, K]` buffer 和 `m_sizes[S]`

**为何不使用全量 `[E, N, K]` buffer**：
- 对于 Qwen3-30B-A3B（E=128, hidden=2048, moe_intermediate=768, bf16），每层 gate_up + down 完整 buffer ≈ 128 × (2×768×2048 + 2048×768) × 2 ≈ 0.9 GB
- 48 层共 ≈ 43 GB，超出消费级显卡（如 24GB RTX 4090）的可用显存
- 分层 S slot（如 S=16），48 层仅需 ≈ 5.4 GB

### 5.3 Slot 数量计算

```python
def compute_slots_per_layer(config, model_config) -> int:
    """根据 gpu_memory_limit_gb 或实际可用显存计算每层可用 slot 数。"""

    if config.gpu_memory_limit_gb > 0:
        total_gpu = config.gpu_memory_limit_gb
    else:
        total_gpu = torch.cuda.get_device_properties(0).total_mem / (1024**3)

    # 扣除非 Expert 参数内存
    static_mem = estimate_static_params_memory(model_config)  # attention, embed, norm, router, lm_head, shared expert
    kv_cache_mem = estimate_kv_cache_memory(config, model_config)
    overhead = 1.0  # GB，系统开销预留

    available = total_gpu - static_mem - kv_cache_mem - overhead
    available = max(available, 0)

    # 计算单个 expert 权重大小
    per_expert_bytes = calc_expert_weight_size(model_config)  # gate_up + down
    total_expert_slots = int(available * (1024**3) / per_expert_bytes)

    slots_per_layer = total_expert_slots // model_config.num_hidden_layers
    slots_per_layer = max(slots_per_layer, model_config.num_experts_per_tok)  # 至少能放 top-k 个
    slots_per_layer = min(slots_per_layer, model_config.num_experts)           # 不超过总 expert 数

    return slots_per_layer
```

### 5.4 Triton Kernel 适配

**关键发现**：现有 `_grouped_gemm_forward_kernel` **不需要修改**。

```python
# kernel 关键循环（triton）
for expert_idx in range(NUM_EXPERTS):       # NUM_EXPERTS = S（slot 数）
    m_size = tl.load(m_sizes_ptr + expert_idx).to(tl.int32)
    if m_size > 0:
        # ... 处理这个 slot 的 tokens ...
```

kernel 天然跳过 `m_size = 0` 的 slot。我们只需要：
1. 传入 `[S, N, K]` slot buffer（而非 `[E, N, K]`）
2. 传入长度为 S 的 `m_sizes`（对应每个 slot 的 token 数）
3. `NUM_EXPERTS = S` 是编译时常量——由于所有层共享相同的 S，kernel **只需编译一次**

> **关于 kernel 重编译**：`NUM_EXPERTS` 作为 `tl.constexpr` 参数会影响 Triton autotuning cache key。但因为我们固定了所有层的 S 值，且 N、K 等维度也固定，kernel 只需在首次调用时编译一次。如果未来允许不同层不同 S，则会产生多次编译（但数量有限，可接受）。

### 5.5 Expert Index Remapping

路由产生的 `selected_experts` 是原始 expert index（0..E-1），需要映射为 slot index（0..S-1）才能被 fused kernel 使用。

```python
def remap_experts_to_slots(
    selected_experts: Tensor,        # [M * topk], 原始 expert index
    expert_to_slot: dict[int, int],  # 该层的 expert→slot 映射
    substitution_map: dict[int, int] | None = None,  # draft 阶段的替换映射
) -> tuple[Tensor, Tensor]:
    """
    将原始 expert index 映射为 slot index。

    对于 draft 阶段：
      - 已缓存的 expert → 直接映射到 slot
      - 被替换的 expert → 先通过 substitution_map 映射到替代 expert，再映射到 slot
      - CPU 执行的 expert → 标记为 -1（由 CPU 路径处理）

    Returns:
        slot_indices: [M * topk], slot index（-1 表示非 GPU 执行）
        gpu_mask: [M * topk], bool, True 表示由 GPU 执行
    """
    if substitution_map:
        selected_experts = apply_substitution(selected_experts, substitution_map)

    slot_indices = torch.full_like(selected_experts, -1)
    gpu_mask = torch.zeros_like(selected_experts, dtype=torch.bool)

    for orig_idx, slot_idx in expert_to_slot.items():
        mask = (selected_experts == orig_idx)
        slot_indices[mask] = slot_idx
        gpu_mask[mask] = True

    return slot_indices, gpu_mask
```

构建 `m_sizes`：

```python
def build_slot_m_sizes(
    slot_indices: Tensor,   # [M * topk], remapped slot indices
    gpu_mask: Tensor,       # [M * topk], bool
    num_slots: int,         # S
) -> tuple[Tensor, Tensor, Tensor]:
    """
    构建 fused kernel 所需的 m_sizes[S] 和排序索引。
    仅统计 gpu_mask=True 的 token。
    """
    gpu_slots = slot_indices[gpu_mask]
    return get_expert_counts_and_idx(gpu_slots, num_slots)
```

### 5.6 异构 MoE Forward 完整流程

```python
def heterogeneous_moe_forward(
    hidden_states: Tensor,         # [M, hidden_dim]
    plan: MoEExecutionPlan,
    expert_cache: ExpertCacheManager,
    cpu_expert_pool: dict,
    layer_idx: int,
    num_selected: int,             # top-k
) -> Tensor:
    M, D = hidden_states.shape
    output = torch.zeros(M, D, device=hidden_states.device, dtype=hidden_states.dtype)

    # === GPU 路径：Fused Kernel via Slot Buffer ===
    if plan.gpu_token_count > 0:
        # 1. expand + sort（仅 GPU token）
        expanded = hidden_states.repeat_interleave(num_selected, dim=0)
        gpu_expanded = expanded[plan.gpu_sort_idx]

        # 2. fused kernel 使用 slot buffer
        gate_up_buf, down_buf = expert_cache.get_layer_buffers(layer_idx)
        gate_up_out = fused_moe_linear(gpu_expanded, gate_up_buf, plan.gpu_m_sizes)
        expert_out = fused_moe_linear(act_fn(gate_up_out), down_buf, plan.gpu_m_sizes)

        # 3. 反排序 + 加权聚合
        expert_out = expert_out[plan.gpu_inv_sort_idx]
        _scatter_weighted_add(output, expert_out, plan.gpu_token_map, plan.gpu_weights)

    # === CPU 路径：并行 Per-Expert Forward ===
    if plan.cpu_token_count > 0:
        cpu_results = _cpu_parallel_forward(hidden_states, plan, cpu_expert_pool, layer_idx)
        _scatter_weighted_add(output, cpu_results, plan.cpu_token_map, plan.cpu_weights)

    return output
```

### 5.7 Expert 替换与 Slot 的关系

Draft 阶段的 expert 替换在 slot 方案下更加自然：

```
路由选中 expert_42（不在 cache 中）
               │
               ▼
   substitution_map: {42 → 7}  （expert_7 在 slot_1 中）
               │
               ▼
   slot_indices 中 expert_42 的位置被映射为 slot_1
   fused kernel 使用 slot_1 的权重计算
```

替换逻辑只需要确保 `substitution_map` 中的替代 expert 必须是当前层 slot 中已缓存的 expert。这在分层 cache 中是天然满足的——每层只能从自己的 slot 中选替代。

### 5.8 CPU-GPU 并行执行时序

```
Timeline (单层 MoE):
─────────────────────────────────────────────────────────────────
GPU Thread:  ┌─ Attention ──┐ ┌─ Route ─┐ ┌─ Fused GEMM ──────┐
             └──────────────┘ └────┬────┘ └────────────────────┘
                                   │                              ▲
                          build_plan                              │
                                   │                              │
CPU Thread:                        └─────┌─ Expert Forward ─────┐│
(ThreadPool)                             └──────────────────────┘│
                                                                  │
Transfer:    ┌─ Async prefetch (transfer_stream) ────────────────┤
(for verify) └───────────────────────────────────────────────────┘│
                                                                   │
             ────────────────────────────────────── sync ───────── ★
             （聚合 GPU + CPU 结果）
```

GPU fused kernel 和 CPU expert forward **并行**执行。CPU 路径使用 ThreadPoolExecutor，在独立线程中执行 CPU 矩阵运算，不阻塞 GPU CUDA stream。同步点在两路结果聚合时。

同时，`transfer_stream` 上的异步预取（为后续 verify 准备 expert）也在后台进行，与 GPU 计算和 CPU 计算三路并行。

---

## 6. 生成循环集成

### 6.1 LLMEngine.step() 修改

```python
def step(self):
    seqs, is_prefill = self.scheduler.schedule()

    if is_prefill:
        token_ids = self.model_runner.call("run", seqs, True)
    elif self.config.enable_speculative and not is_prefill:
        # 投机解码路径
        token_ids = self.spec_engine.speculative_step(seqs)
    else:
        # 标准 decode 路径（可用于全 GPU 场景或不启用投机解码时）
        token_ids = self.model_runner.call("run", seqs, False)

    self.scheduler.postprocess(seqs, token_ids)
    ...
```

### 6.2 Verify 跳过优化

当所有条件满足时可跳过 verify：
1. 所有 routed expert 都在 GPU cache 中（无替换发生）
2. `draft_top_c == 0`（无 CPU 执行）
3. 所有 sequence 使用 greedy decoding

此时 draft 输出与标准 decode 完全一致，可直接接受所有 draft token。

---

## 7. 实现路线图

### Phase 1：基础异构推理（不含投机解码）

| 步骤 | 内容 | 涉及文件 |
|------|------|---------|
| 1.1 | Config 扩展 | `config.py` |
| 1.2 | 异构参数加载器 | `utils/heterogeneous_loader.py` |
| 1.3 | Expert Cache Manager | `expert/cache.py` |
| 1.4 | MoEExecutionPlan + placement 构建 | `expert/placement.py` |
| 1.5 | 异构 MoE forward (heterogeneous.py) | `layers/fuse_moe/heterogeneous.py` |
| 1.6 | ModelRunner 异构 forward 路径 | `engine/model_runner.py` |
| 1.7 | 调度策略框架 | `scheduling/*.py` |
| 1.8 | 集成测试 | `examples/` |

### Phase 2：投机解码

| 步骤 | 内容 | 涉及文件 |
|------|------|---------|
| 2.1 | Sequence draft 状态扩展 | `engine/sequence.py` |
| 2.2 | BlockManager draft/verify 支持 | `engine/block_manager.py` |
| 2.3 | AcceptanceStrategy 框架 | `engine/speculative/acceptance.py` |
| 2.4 | SpeculativeEngine 核心循环 | `engine/speculative/spec_engine.py` |
| 2.5 | LLMEngine 集成 | `engine/llm_engine.py` |
| 2.6 | Verify 跳过优化 | `engine/speculative/spec_engine.py` |
| 2.7 | 端到端测试 | `examples/` |

### Phase 3：预取与高级策略

| 步骤 | 内容 | 涉及文件 |
|------|------|---------|
| 3.1 | Expert Prefetcher | `expert/prefetcher.py` |
| 3.2 | Draft Scheduler 高级策略 | `scheduling/draft_scheduler.py` |
| 3.3 | Adaptive Cache Strategy | `scheduling/cache_strategy.py` |
| 3.4 | History-based Prefetch | `scheduling/prefetch_strategy.py` |
| 3.5 | 性能基准测试 | `benchmarks/` |

### Phase 4：高级优化

| 步骤 | 内容 |
|------|------|
| 4.1 | CPU Expert 执行优化（torch.compile、NUMA affinity） |
| 4.2 | Draft-Verify KV Cache 增量更新（避免全量回滚） |

---

## 8. 关键设计决策与权衡

### 8.1 为何在 MoE block 层面而非 model 层面拆分

**方案 A**：在 model 整体 forward 外拆分（demo 的做法）
- 优势：完全解耦路由和执行
- 劣势：无法利用 nano-vllm-moe 的 CUDA graph、Context 管理等基础设施

**方案 B（采用）**：在 MoE block 层面拆分
- 保留 decoder layer 的整体结构不变
- 仅在 MoE block 的 forward 中插入 placement 逻辑
- Attention、LayerNorm 等完全复用现有路径

### 8.2 CUDA Graph 兼容性

| 场景 | CUDA Graph |
|------|-----------|
| 全 GPU 标准 decode | ✅ 使用（现有路径不变） |
| 异构 decode (draft) | ❌ 禁用（CPU 路径不可 capture） |
| 异构 decode (verify) | ❌ 禁用（类似 prefill） |

通过 `enforce_eager=True` 或自动检测来禁用异构路径的 CUDA graph。

### 8.3 Fused vs Non-fused 切换

```python
# 在 MoE block forward 中
if self.heterogeneous_mode:
    # 使用异构执行路径
    return heterogeneous_moe_forward(hidden_states, plan, expert_cache, ...)
else:
    # 使用原有 fused 路径（全 GPU）
    return self._original_fused_forward(hidden_states)
```

---

## 9. 与 demo 的功能对照表

| demo 功能 | nano-vllm-moe 改造方案 | 状态 |
|-----------|----------------------|------|
| ParameterLoader | `utils/heterogeneous_loader.py` - HeterogeneousModelLoader | 新增 |
| ExpertCache | `expert/cache.py` - ExpertCacheManager | 新增 |
| ModelRunner 抽象 | 直接修改 `engine/model_runner.py` | 修改 |
| build_prefill_placement | `expert/placement.py` - build_prefill_plan | 新增 |
| build_draft_placement | `expert/placement.py` - build_draft_plan | 新增 |
| CBExecutor._forward_draft | `engine/model_runner.py` - run_draft | 修改 |
| CBExecutor._forward_verify | `engine/model_runner.py` - run_verify | 修改 |
| CBExecutor._execute_accept | `engine/speculative/spec_engine.py` | 新增 |
| DraftScheduler | `scheduling/draft_scheduler.py` | 新增 |
| ExpertPrefetcher | `expert/prefetcher.py` | 新增 |
| AcceptanceStrategy | `engine/speculative/acceptance.py` | 新增 |
| CacheReplacementStrategy | `scheduling/cache_strategy.py` | 新增 |
| PrefetchStrategy | `scheduling/prefetch_strategy.py` | 新增 |
| PagedKVCache draft/verify | `engine/block_manager.py` 扩展 | 修改 |
| Sequence draft 状态 | `engine/sequence.py` 扩展 | 修改 |
| CPU-GPU 并行 expert | `layers/fuse_moe/heterogeneous.py` | 新增 |
| Fused MoE + Expert Cache | ExpertCacheManager 分层固定 slot buffer `[S,N,K]` | 新增 |

---

## 10. 测试策略

### 10.1 单元测试

- Expert Cache：load/evict/prefetch
- Placement 构建：draft / prefill / 替换映射正确性
- Heterogeneous forward：GPU-only / CPU-only / 混合
- KV Cache rollback：draft token 正确回滚
- Acceptance strategy：greedy / sampling 模式

### 10.2 集成测试

- 全 GPU 标准 decode（回归测试）
- 全 GPU 投机解码（verify 跳过）
- 部分 GPU 投机解码（含 CPU 执行 + expert 替换）
- 多 sequence batch
- 长序列生成

### 10.3 性能基准

- vs nano-vllm-moe 原生全 GPU decode（tokens/s）
- vs on_device_sd/demo 同配置（tokens/s）
- Expert cache hit rate
- CPU/GPU 执行时间占比
- 投机解码接受率

---

## 11. Benchmark 规范（新增）

本节定义可复现的性能评估口径，确保改造过程中每个阶段都能判断“是否真实提速”。

### 11.1 Benchmark 目标

1. 验证迁移后在纯 GPU 场景不退化，尽量贴近现有 nano-vllm-moe 基线。
2. 量化异构执行（CPU-GPU）带来的开销与收益边界。
3. 量化 speculative（draft-verify）在不同接受率条件下的净收益。
4. 量化预取和 cache 策略对传输隐藏效果的影响。

### 11.2 统一实验条件

1. 固定模型：`Qwen3-30B-A3B-Base`。
2. 固定采样参数：`temperature=0`（greedy）和 `temperature=0.6`（sampling）各一组。
3. 固定 prompt 集：短（64 tokens）、中（512 tokens）、长（2048 tokens）三档。
4. 固定 batch：`bs=1, 2, 4, 8`。
5. 每组实验：`warmup >= 2`，正式重复 `>= 5` 次，报告均值/标准差。
6. 计时口径必须 CUDA 同步：阶段结束前调用 `torch.cuda.synchronize()`。

### 11.3 评测矩阵

| Case ID | 模式 | 说明 | 目的 |
|--------|------|------|------|
| A1 | baseline-gpu | 原生 nano-vllm-moe，标准 decode | GPU 基线 |
| A2 | migrated-gpu | 改造后关闭异构/投机 | 框架迁移开销 |
| B1 | hetero-standard | 异构 prefill/decode，无 speculative | 异构本体开销 |
| C1 | hetero-spec | 异构 + speculative（draft_top_c=2） | 端到端收益 |
| C2 | hetero-spec-no-prefetch | 关闭预取，仅替换 + verify | 评估预取价值 |
| C3 | hetero-spec-no-substitute | 关闭替换，仅 CPU 执行缺失 expert | 评估替换价值 |

### 11.4 指标定义

1. 端到端吞吐：`tokens/s`（prefill+decode 全流程）。
2. TTFT：首 token 延迟（ms）。
3. TPOT：decode 每 token 延迟（ms/token）。
4. 分阶段耗时：`prefill_s`, `draft_s`, `verify_s`, `accept_s`。
5. 算子分解：`attn_s`, `route_s`, `moe_gpu_s`, `moe_cpu_s`, `transfer_wait_s`, `scheduler_s`。
6. 缓存指标：`expert_cache_hit_rate`, `evictions`, `prefetch_completed`, `prefetch_late`。
7. speculative 指标：`accept_rate`, `avg_accepted_tokens_per_round`, `verify_rounds`。

### 11.5 结果判定门槛（阶段性）

1. Phase 1 完成门槛：A2 相比 A1 的吞吐不低于 `0.9x`。
2. Phase 2 完成门槛：C1 相比 B1 的吞吐提升至少 `1.1x`（greedy，短中 prompt）。
3. Phase 3 完成门槛：C1 相比 C2 提升至少 `1.05x`，且 `prefetch_late` 显著下降。
4. 稳定性门槛：同组 5 次运行标准差不超过均值的 `10%`。

### 11.6 传输预算估算（用于默认预取上限）

令单 expert 传输时间为 $T_{copy}$，单层（或单 draft step）计算窗口时间为 $T_{win}$，则预取上限：

$$
N_{prefetch,max} = \left\lfloor \frac{T_{win}}{T_{copy}} \right\rfloor
$$

其中：

1. $T_{copy}$ 通过启动时 micro-benchmark 实测（CPU pinned -> GPU，按专家权重大小）。
2. $T_{win}$ 可选“按层窗口”或“按 step 窗口”；默认按层窗口更细粒度。
3. 实际调度使用 `min(N_prefetch,max, cache_free_slots, strategy_quota)`。

### 11.7 输出格式与落盘

建议每次实验输出 JSONL，字段至少包含：

```json
{
    "case_id": "C1",
    "model": "Qwen3-30B-A3B-Base",
    "batch_size": 4,
    "prompt_len": 512,
    "max_new_tokens": 128,
    "tokens_per_sec": 0.0,
    "ttft_ms": 0.0,
    "tpot_ms": 0.0,
    "prefill_s": 0.0,
    "draft_s": 0.0,
    "verify_s": 0.0,
    "accept_s": 0.0,
    "attn_s": 0.0,
    "moe_gpu_s": 0.0,
    "moe_cpu_s": 0.0,
    "transfer_wait_s": 0.0,
    "expert_cache_hit_rate": 0.0,
    "accept_rate": 0.0
}
```

推荐目录：`benchmarks/results/YYYYMMDD/*.jsonl`。

### 11.8 速度变化预期（工程估计）

1. A2 vs A1：理论接近持平，允许轻微回归（<=10%）。
2. B1 vs A2：通常下降（异构开销），取决于 cache 命中率与传输重叠。
3. C1 vs B1：在高接受率和高重叠下应提升；低接受率时可能持平或略降。

> 以上为预期区间，不作为最终结论；必须以 11.2~11.7 的实测结果为准。


