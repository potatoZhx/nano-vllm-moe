# nano-vllm-moe 接入 KTransformers CPU BF16 MoE 算子的第一阶段实现方案

> 目标：在 `nano-vllm-moe` 中新增一个基于 KTransformers `kt-kernel` 的 CPU MoE backend，使 decode / 小 `qlen` 场景优先走 KTransformers 的 AVX512 / AVX2 BF16 算子；允许 `kt-kernel` 每层持有一份 MoE expert packed 权重副本；所有权重加载与 packing 必须在推理前完成，forward 热路径不得触发 `load_weights` 或 safetensors 二次读取。

---

## 0. 结论

建议实现一个新的 backend：

```text
nanovllm/layers/fuse_moe/kt_direct_backend.py
```

它不要复用当前 `kt_backend.py` 的 shared-wrapper + `weight_path` 方案，而是直接调用 KTransformers 的 C++ extension：

```python
kt_kernel_ext.moe.MOEConfig
kt_kernel_ext.moe.AMXBF16_MOE
kt_kernel_ext.moe.AVX2BF16_MOE
kt_kernel_ext.CPUInfer
kt_kernel_ext.WorkerPoolConfig
```

核心原则：

```text
nano-vllm-moe 负责：
  1. 正常加载模型权重；
  2. 管理 GPU expert slots / expert_cache；
  3. 保留 raw CPU expert weights，继续服务 GPU slot copy；
  4. 构造 selected_experts / routing_weights / heterogeneous plan。

kt-kernel 负责：
  1. 在模型初始化阶段，从 nano 已加载的 CPU expert tensors 拷贝并 pack 一份内部 MoE 权重；
  2. decode / 小 qlen 时执行 AVX512/AVX2 BF16 MoE fused operator；
  3. 返回 per-token CPU partial output，直接加到最终 output。
```

第一阶段允许的额外内存：

```text
nano raw CPU expert weights
+ kt-kernel per-layer packed CPU expert weights
```

第一阶段禁止的事情：

```text
forward / decode 热路径中：
  - 不允许从 safetensors 读权重；
  - 不允许调用 KTMoEWrapper.load_weights；
  - 不允许按 layer 切换重新 pack；
  - 不允许 Python 循环逐 expert 调 F.linear 作为主路径。
```

---

## 1. 当前代码问题与改造动机

### 1.1 当前 `kt_backend.py` 的问题

当前 `nanovllm/layers/fuse_moe/kt_backend.py` 是一个 optional kt-kernel backend，但它的架构是：

```text
module-level shared singleton:
  _shared_kt_wrapper

每层 forward 前：
  _ensure_layer_weights(layer_idx)
    -> wrapper.load_weights(...)
```

文件开头注释明确说明：

```text
Uses a single NativeMoEWrapper for all layers to avoid creating 48
AMX contexts (which causes a segfault). Before each layer's forward
call, weights are reloaded for that layer.
```

这对功能验证可以接受，但对性能不可接受。特别是 decode 场景，`qlen=1` 时真正计算量较小，任何 forward 热路径里的 load / pack / layer switching 都会压倒 AVX512/AVX2 kernel 的收益。

### 1.2 当前 PyTorch CPU backend 的瓶颈

当前 `cpu_backend.py` 的 CPU path 本质上是：

```python
gate_up = F.linear(hidden_cpu[start:end], gate_up_weight)
out = F.linear(act_fn(gate_up), down_weight)
out.mul_(weights_cpu[start:end].unsqueeze(-1))
```

它按 expert/task 循环执行小 GEMM。decode / speculative verify 中，很多 expert 只命中 1~2 个 token，这类小 M GEMM 在 PyTorch/MKL/ThreadPool 组合下调度开销显著，且难以和 MoE route grouping / weighted merge 做深度融合。

### 1.3 为什么要接 kt-kernel 的底层 MoE operator

KTransformers BF16 MoE operator 已经把以下流程融合在一个 C++ operator 中：

```text
expert grouping
+ gate GEMM
+ up GEMM
+ SwiGLU
+ down GEMM
+ routing weight merge
```

更重要的是，它在 BF16 MoE 内部针对 workload 自动分流：

```text
大 qlen / prefill    -> AMX mat_mul
小 qlen / decode     -> AVX512 vec_mul 或 AVX2 fallback
```

这正好匹配本项目的需求：**主要提升 decode / 小 qlen CPU expert fallback**。

---

## 2. 设计目标

### 2.1 必须达成

1. `cpu_expert_backend="kt_direct"` 时启用新 backend。
2. 不再使用 `KTMoEWrapper(weight_path=...)` 的文件加载路径。
3. 不再使用当前 `kt_backend.py` 的 `_ensure_layer_weights()` 重载路径。
4. 每个 MoE layer 在推理前完成一次 kt packed 权重初始化。
5. forward 热路径只执行：
   - hidden / top-k ids / top-k weights 拷贝到 pinned CPU buffer；
   - `moe.forward_task(...)`；
   - CPUInfer / CUDA stream 同步；
   - CPU output copy 回 GPU；
   - 与 GPU expert output 合并。
6. 保留 nano-vllm-moe 现有 `cpu_expert_pool`，继续兼容 GPU slots。
7. `kt-kernel` 可以持有一份 packed MoE expert 权重副本。
8. decode / 小 `qlen` 默认走 KTransformers AVX512 BF16 path；无 AVX512 BF16 时允许 AVX2 fallback。
9. 结果语义与现有 heterogeneous MoE 一致。

### 2.2 第一阶段不做

1. 不删除 nano raw CPU expert weights。
2. 不把 kt packed weights 反向作为 GPU slot 的权重来源。
3. 不复制 KTransformers C++ kernel 到 nano-vllm-moe 仓库。
4. 不优化 prefill AMX 大 batch 场景作为第一优先级。
5. 不支持 CUDA graph capture 中的 CPU sidecar 作为第一阶段目标。
6. 不支持 INT4/FP8/MXFP4；第一阶段只做 BF16。

---

## 3. 总体架构

### 3.1 新增组件

```text
nanovllm/layers/fuse_moe/kt_direct_backend.py

  KtDirectGlobalRuntime
    - 单例
    - 持有 kt_kernel_ext.CPUInfer
    - 持有 WorkerPoolConfig
    - 负责 CPU feature/backend selection

  KtDirectLayerState
    - 每层一个
    - 持有 kt MOE object: AMXBF16_MOE 或 AVX2BF16_MOE
    - 持有 raw tensor 引用，防止 data_ptr 悬空
    - 持有 gpu_expert_mask CPU pinned tensor
    - 持有 physical_to_logical tensor
    - 初始化阶段执行 load_weights_task 一次

  KtDirectCpuMoeBackend
    - 对接 heterogeneous_moe_forward 的 CPU backend 接口
    - forward 返回 CpuMoeResult
    - outputs_cpu.shape == hidden_states.shape
      表示 kt-kernel 已经返回 per-token aggregated partial output
```

### 3.2 数据流

```text
模型加载 / enable_heterogeneous 阶段
  |
  |-- nano-vllm-moe 原有 loader 加载 expert 权重到 cpu_expert_pool
  |       gate_up: [2I, H]
  |       down:    [H, I]
  |
  |-- KtDirectCpuMoeBackend 初始化
          |
          |-- 从 cpu_expert_pool 拿 tensor
          |-- split gate_up -> gate / up
          |-- 构造 gate_ptrs / up_ptrs / down_ptrs
          |-- 构造 MOEConfig
          |-- 创建 AMXBF16_MOE 或 AVX2BF16_MOE
          |-- cpu_infer.submit(moe.load_weights_task(...))
          |-- cpu_infer.sync()
          |
          `-- kt-kernel 内部持有 packed weights

decode forward
  |
  |-- routing: selected_experts / routing_weights
  |-- heterogeneous_moe_forward
  |-- GPU cached experts 走原路径
  |-- CPU missed experts 走 KtDirectCpuMoeBackend
          |
          |-- submit_forward
          |-- moe.forward_task
          |-- sync_forward
          `-- outputs_cpu: [num_tokens, hidden_size]
  |
  `-- output.add_(cpu_partial)
```

---

## 4. 和当前 heterogeneous merge 语义的兼容

当前 `heterogeneous.py` 已经有特殊逻辑：

```python
# kt_kernel returns per-token output, not per-route
if cpu_outputs.shape == output.shape:
    output.add_(cpu_outputs.to(dtype=output.dtype, device=device))
else:
    route_buffer.index_copy_(...)
```

因此新 backend 可以直接返回：

```python
CpuMoeResult(
    token_indices=torch.empty(0, dtype=torch.int64, device="cpu"),
    outputs_cpu=cpu_partial,   # shape == hidden_states.shape
    prep_ms=...,
    compute_ms=...,
)
```

这样不需要改 `_accumulate_mixed_routes_deterministic()` 的主要逻辑。

---

## 5. 权重生命周期设计

### 5.1 第一阶段权重副本策略

第一阶段采用“双 CPU 表示”：

```text
A. nano raw CPU weights
   - 来源：nano-vllm-moe 现有加载流程
   - 作用：
       1. 继续服务 GPU fallback workspace / GPU slot copy；
       2. 作为 kt-kernel 初始化阶段的输入；
       3. 保持现有功能兼容。

B. kt-kernel packed CPU weights
   - 来源：初始化阶段读取 nano raw tensor 的 data_ptr
   - 作用：
       1. decode / 小 qlen AVX512/AVX2 BF16 MoE operator；
       2. 避免 PyTorch F.linear 小 GEMM；
       3. 避免 forward 热路径重新 packing。
```

### 5.2 为什么 packed 副本不可避免

KTransformers 的 BF16 MoE kernel 不是直接使用 PyTorch row-major 权重进行普通 GEMM。它在 `load_weights_task` 中会把 per-expert gate/up/down 权重转换为内部 BufferB / NUMA TP 布局，以适配 AMX/AVX512/AVX2 kernel 的访存模式。

因此第一阶段允许：

```text
raw row-major weights + packed kt weights
```

但不允许：

```text
raw row-major weights + safetensors 二次加载 raw weights + packed kt weights
```

### 5.3 tensor 引用保活

因为 `MOEConfig.gate_projs/up_projs/down_projs` 传的是裸 `data_ptr()`，Python 侧必须保存 tensor 引用：

```python
self._weight_refs: list[torch.Tensor] = []
```

对于每个 expert：

```python
gate_up = params["gate_up"]
down = params["down"]

gate = gate_up[:I, :].contiguous()
up = gate_up[I:, :].contiguous()
down = down.contiguous()

self._weight_refs.extend([gate, up, down])
```

注意：

1. 如果 `gate_up[:I, :]` 是 view，不应直接传 view 的 pointer，必须 `.contiguous()`。
2. 如果 nano 的 `gate_up` 本身已经按 `[gate, up]` 顺序 contiguous，可考虑后续优化为零拷贝切片，但第一阶段以正确性和稳定性优先。
3. `down` 的 shape 必须与 kt-kernel 预期一致，通常是 `[hidden_size, intermediate_size]`。
4. 权重 dtype 必须是 `torch.bfloat16`。

---

## 6. backend selection：优先使用 decode 友好的路径

### 6.1 默认选择

第一阶段建议提供如下配置：

```text
--cpu-expert-backend kt_direct
--kt-direct-backend auto|amx_bf16|avx2_bf16
```

默认：

```text
auto
```

选择逻辑：

```python
if forced == "avx2_bf16":
    use AVX2BF16_MOE
elif forced == "amx_bf16":
    use AMXBF16_MOE
else:
    if AMXBF16_MOE is available:
        use AMXBF16_MOE
    elif AVX2BF16_MOE is available:
        use AVX2BF16_MOE
    else:
        raise RuntimeError
```

### 6.2 为什么 decode 仍可用 AMXBF16_MOE

即使构造的是 `AMXBF16_MOE`，KTransformers BF16 MoE 内部在小 qlen / decode 时会进入 `vec_mul`，实际走 AVX512 BF16 vector path，而不是 AMX tile path。

因此：

```text
AMXBF16_MOE 不等于 decode 一定使用 AMX。
对于 qlen=1，它会走 KTransformers 的 AVX512 vec path。
```

如果目标机器没有 AVX512 BF16 或 AMX build 不稳定，则用 `AVX2BF16_MOE`。

### 6.3 推荐的第一阶段默认值

对 Ice Lake / 无 AMX 的机器：

```text
kt_direct_backend=avx2_bf16
```

对 Sapphire Rapids / Emerald Rapids：

```text
kt_direct_backend=auto
```

如果发现 AMX context / destructor 稳定性问题，但仍希望 decode 跑 AVX512，可新增一个 KTransformers 侧的 `AVX512BF16_MOE` 显式 binding；若当前 kt-kernel 未暴露单独 AVX512BF16_MOE，则先使用 `AMXBF16_MOE` 的 vec path 或 AVX2 fallback。

---

## 7. 新文件设计：`kt_direct_backend.py`

### 7.1 文件位置

```text
nanovllm/layers/fuse_moe/kt_direct_backend.py
```

### 7.2 public class

```python
class KtDirectCpuMoeBackend:
    def __init__(
        self,
        *,
        layer_idx: int,
        cpu_expert_pool: dict[int, dict[str, torch.Tensor]],
        max_routes: int,
        moe_intermediate_size: int,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        gpu_expert_mask: torch.Tensor,
        kt_num_threads: int = 0,
        kt_threadpool_count: int = 1,
        kt_chunked_prefill_size: int = 4096,
        kt_direct_backend: str = "auto",
        kt_numa_nodes: list[int] | None = None,
        strict_dtype: bool = True,
    ) -> None:
        ...
```

### 7.3 forward signature

保持和现有 CPU backend 兼容：

```python
@torch.no_grad()
def forward(
    self,
    *,
    hidden_states: torch.Tensor,
    flat_weights: torch.Tensor,
    top_k: int,
    cpu_indices: torch.Tensor,
    cpu_task_expert_ids: torch.Tensor,
    cpu_task_offsets: torch.Tensor,
    act_fn: Callable[[torch.Tensor], torch.Tensor],
    parallel_mode: str = "serial",
    num_threads: int = 4,
    cpu_task_expert_ids_host: list[int] | None = None,
    cpu_task_offsets_host: list[int] | None = None,
    selected_experts: torch.Tensor | None = None,
    routing_weights: torch.Tensor | None = None,
) -> CpuMoeResult:
    ...
```

注意：`act_fn`、`parallel_mode`、`cpu_task_*` 在 kt-direct 中不参与计算，但为了兼容 `heterogeneous_moe_forward` 保留参数。

---

## 8. 运行时单例：`KtDirectGlobalRuntime`

### 8.1 作用

KTransformers `CPUInfer` / worker pool 应该全局共享，不应每层创建一个 worker pool。

```python
class KtDirectGlobalRuntime:
    _instance: "KtDirectGlobalRuntime | None" = None

    def __init__(
        self,
        kt_num_threads: int,
        kt_threadpool_count: int,
        kt_numa_nodes: list[int] | None,
    ):
        self.kt_kernel_ext = ...
        self.moe_mod = ...
        self.cpu_infer = ...
```

### 8.2 初始化 worker pool

伪代码：

```python
from kt_kernel import kt_kernel_ext

worker_config = kt_kernel_ext.WorkerPoolConfig()
worker_config.subpool_count = kt_threadpool_count

if kt_numa_nodes is None:
    worker_config.subpool_numa_map = list(range(kt_threadpool_count))
else:
    worker_config.subpool_numa_map = kt_numa_nodes

worker_config.subpool_thread_count = split_threads(
    kt_num_threads,
    kt_threadpool_count,
)

cpu_infer = kt_kernel_ext.CPUInfer(worker_config)
```

线程切分：

```python
def split_threads(total: int, groups: int) -> list[int]:
    base = total // groups
    rem = total % groups
    return [base + (1 if i < rem else 0) for i in range(groups)]
```

### 8.3 不要强制限制 16 线程

当前 `kt_backend.py` 有：

```python
kt_num_threads = min(kt_num_threads, 16)
```

第一阶段建议删除这个限制，改成：

```text
默认使用 os.sched_getaffinity(0) 的可用核数；
允许用户通过配置显式限制；
不要在 backend 内部静默 clamp 到 16。
```

---

## 9. 每层 state：`KtDirectLayerState`

### 9.1 数据成员

```python
@dataclass
class KtDirectLayerState:
    layer_idx: int
    moe: object
    gpu_expert_mask_cpu: torch.Tensor
    physical_to_logical: torch.Tensor
    weight_refs: list[torch.Tensor]
    output_buffers: ...
    loaded: bool = False
```

### 9.2 构造权重指针表

kt-kernel 的 BF16 MoE `MOEConfig` 需要：

```text
gate_projs: list[list[int]]
up_projs:   list[list[int]]
down_projs: list[list[int]]
```

外层 list 对应 NUMA/TP，第一阶段可以每个 NUMA 传同一组 raw tensor pointers，让 kt-kernel 在 load 阶段为每个 TP 做内部切分/packing。

伪代码：

```python
def build_bf16_weight_ptrs(
    cpu_expert_pool: dict[int, dict[str, torch.Tensor]],
    num_experts: int,
    intermediate_size: int,
    hidden_size: int,
    threadpool_count: int,
    strict_dtype: bool,
) -> tuple[list[list[int]], list[list[int]], list[list[int]], list[torch.Tensor]]:
    gate_ptrs_1numa = []
    up_ptrs_1numa = []
    down_ptrs_1numa = []
    refs = []

    for eid in range(num_experts):
        params = cpu_expert_pool.get(eid)
        if params is None:
            # GPU-resident expert 也可以在 CPU pool 里不存在。
            # 但 kt-kernel 的 load_weights 需要处理 gpu_experts_mask。
            # 最稳妥：要求所有 expert 都有 raw CPU copy；
            # 若不是，则为 GPU expert 填 dummy bf16 tensor。
            gate, up, down = make_dummy_weights(...)
        else:
            gate_up = params.get("gate_up")
            down_w = params.get("down")

            if gate_up is None or down_w is None:
                packed = params.get("packed")
                if packed is not None:
                    gate_up = packed.gate_up
                    down_w = packed.down

            validate_shape_dtype(gate_up, down_w)

            gate = gate_up[:intermediate_size, :].contiguous()
            up = gate_up[intermediate_size:, :].contiguous()
            down = down_w.contiguous()

        refs.extend([gate, up, down])
        gate_ptrs_1numa.append(gate.data_ptr())
        up_ptrs_1numa.append(up.data_ptr())
        down_ptrs_1numa.append(down.data_ptr())

    gate_ptrs = [gate_ptrs_1numa for _ in range(threadpool_count)]
    up_ptrs = [up_ptrs_1numa for _ in range(threadpool_count)]
    down_ptrs = [down_ptrs_1numa for _ in range(threadpool_count)]

    return gate_ptrs, up_ptrs, down_ptrs, refs
```

### 9.3 是否必须所有 expert 都在 cpu_expert_pool 中

建议第一阶段要求：

```text
cpu_expert_pool 必须包含该层所有 expert 的 raw CPU 权重。
```

理由：

1. 简化 kt-kernel load。
2. 简化 gpu_expert_mask 切换。
3. 为后续 GPU slot refill 保留统一 CPU source。
4. 避免给 GPU expert dummy pointer 后 kt-kernel 内部仍误读。

如果当前 nano-vllm-moe 为节省内存只保留 CPU miss experts，则需要在 loader 阶段增加配置：

```text
--cpu-expert-keep-all-raw-weights-for-kt-direct true
```

或在 kt_direct 初始化时对缺失 expert 报错。

### 9.4 MOEConfig 构造

伪代码：

```python
from kt_kernel_ext.moe import MOEConfig
import kt_kernel_ext.moe as kt_moe

moe_config = MOEConfig(
    num_experts,
    num_experts_per_tok,
    hidden_size,
    moe_intermediate_size,
    gpu_expert_mask_cpu.data_ptr(),
)

moe_config.layer_idx = layer_idx
moe_config.pool = runtime.cpu_infer.backend_
moe_config.max_len = kt_chunked_prefill_size

moe_config.gate_proj = 0
moe_config.up_proj = 0
moe_config.down_proj = 0

moe_config.gate_projs = gate_ptrs
moe_config.up_projs = up_ptrs
moe_config.down_projs = down_ptrs

# BF16 没有 scale。
moe_config.gate_scales = [[0] * num_experts for _ in range(threadpool_count)]
moe_config.up_scales = [[0] * num_experts for _ in range(threadpool_count)]
moe_config.down_scales = [[0] * num_experts for _ in range(threadpool_count)]

moe_config.load = False
moe_config.save = False
```

创建 C++ MoE object：

```python
if backend_cls == "amx_bf16":
    moe = kt_moe.AMXBF16_MOE(moe_config)
elif backend_cls == "avx2_bf16":
    moe = kt_moe.AVX2BF16_MOE(moe_config)
else:
    raise RuntimeError(...)
```

加载 / pack：

```python
physical_to_logical = torch.arange(num_experts, dtype=torch.int64, device="cpu")
runtime.cpu_infer.submit(moe.load_weights_task(physical_to_logical.data_ptr()))
runtime.cpu_infer.sync()
```

### 9.5 load 完成时间

必须在以下阶段之一完成：

1. `Qwen3MoeHeterogeneousSparseMoeBlock.enable_heterogeneous(...)` 阶段；
2. 模型全部权重加载完毕后的 explicit warmup 阶段；
3. server 开始接收请求前的 `prepare_inference()` 阶段。

禁止 lazy load 到第一轮 decode forward。

---

## 10. forward 热路径设计

### 10.1 首选：复用 KTransformers BaseMoEWrapper 的 buffer 思路

KTransformers `BaseMoEWrapper.submit_forward()` 的核心逻辑是：

```text
hidden_states copy to pinned CPU input buffer
topk ids copy to pinned CPU id buffer
routing weights copy to pinned CPU weight buffer
cpu_infer.submit_with_cuda_stream(cuda_stream, moe.forward_task(...))
sync_with_cuda_stream
output_cpu copy back to GPU output buffer
```

新 backend 可以复刻这段逻辑，避免依赖 `KTMoEWrapper` 对象。

### 10.2 自己维护 buffer cache

新增：

```python
class KtDirectCPUBuffer:
    capture_bs: list[int] = []
    capture_buffers: dict[tuple[int, int, torch.dtype, torch.device], tuple] = {}
    temp_key = None
    temp_buffer = None
    buffer_depth = 2
```

每个 buffer tuple：

```text
input_tensor_cpu:
  list[torch.Tensor] of shape [batch_size, hidden_size], dtype bf16, pin_memory=True

topk_ids_cpu:
  list[torch.Tensor] of shape [batch_size, top_k], dtype int64, pin_memory=True

weights_cpu:
  list[torch.Tensor] of shape [batch_size, top_k], dtype float32 or bf16, pin_memory=True

output_cpu:
  list[torch.Tensor] of shape [batch_size, hidden_size], dtype bf16, pin_memory=True

bsz_tensor_cpu:
  list[torch.Tensor] of shape [1], dtype int32, pin_memory=True

output_gpu:
  list[torch.Tensor] of shape [batch_size, hidden_size], dtype hidden_states.dtype, device=hidden_states.device
```

decode 场景 `batch_size` 经常是 1、2、4、8、16，可以预注册 capture batch sizes：

```text
--kt-direct-capture-bs 1,2,4,8,16,32
```

### 10.3 forward 伪代码

```python
@torch.no_grad()
def forward(...):
    prep_t0 = perf_counter()

    assert selected_experts is not None
    assert routing_weights is not None

    flat_hidden = hidden_states.view(-1, hidden_states.shape[-1]).contiguous()
    batch_size = flat_hidden.shape[0]

    topk_ids = selected_experts.contiguous().to(torch.int64)
    topk_w = routing_weights.contiguous()
    if topk_w.dtype != torch.float32:
        # kt BaseMoEWrapper 使用 float32 weights_cpu。
        # 第一阶段建议保持 float32，避免 BF16 route 权重误差扩大。
        topk_w_for_cpu = topk_w.to(torch.float32)
    else:
        topk_w_for_cpu = topk_w

    buffers = KtDirectCPUBuffer.get_buffer(
        hidden_states=flat_hidden,
        top_k=self.num_experts_per_tok,
        prefer_weight_dtype=torch.float32,
    )

    slot = self.layer_idx % KtDirectCPUBuffer.buffer_depth

    input_cpu[slot].copy_(flat_hidden, non_blocking=True)
    topk_ids_cpu[slot].copy_(topk_ids, non_blocking=True)
    weights_cpu[slot].copy_(topk_w_for_cpu, non_blocking=True)

    prep_ms = (perf_counter() - prep_t0) * 1000

    compute_t0 = perf_counter()

    stream = torch.cuda.current_stream(hidden_states.device).cuda_stream

    self.runtime.cpu_infer.submit_with_cuda_stream(
        stream,
        self.moe.forward_task(
            bsz_tensor_cpu[slot].data_ptr(),
            self.num_experts_per_tok,
            topk_ids_cpu[slot].data_ptr(),
            weights_cpu[slot].data_ptr(),
            input_cpu[slot].data_ptr(),
            output_cpu[slot].data_ptr(),
            False,  # incremental
        ),
    )

    self.runtime.cpu_infer.sync_with_cuda_stream(stream, 0)

    output_gpu[slot].copy_(output_cpu[slot], non_blocking=True)

    compute_ms = (perf_counter() - compute_t0) * 1000

    return CpuMoeResult(
        token_indices=torch.empty(0, dtype=torch.int64, device="cpu"),
        outputs_cpu=output_gpu[slot],
        prep_ms=prep_ms,
        compute_ms=compute_ms,
    )
```

说明：

1. `outputs_cpu` 字段名沿用 `CpuMoeResult`，但实际可以是 GPU tensor。当前 merge 逻辑会 `.to(device=device)`，所以不会破坏语义。
2. 如果需要严格字段语义，可新增 `outputs` 字段，但第一阶段不建议大改数据结构。
3. `incremental=False`，第一阶段不启用 deferred expert。

### 10.4 GPU expert mask 更新问题

`gpu_expert_mask` 表示哪些 expert 由 GPU cache 处理，kt-kernel 应跳过它们。

问题：nano-vllm-moe 的 GPU expert cache 可能动态变化。

第一阶段可选两种策略：

#### 策略 A：固定 GPU mask

初始化 backend 时快照：

```python
gpu_mask = expert_cache.get_cached_expert_mask()
```

之后不动态变更。

适用：

```text
verify / serving 阶段 GPU experts 相对稳定；
CPU fallback 负责所有非 cached experts。
```

优点：简单稳定。  
缺点：GPU cache 动态替换后，kt mask 可能过期。

#### 策略 B：forward 前更新 pinned mask

因为 `MOEConfig` 持有的是 `gpu_expert_mask_cpu.data_ptr()`，只要 tensor 本身不被替换，可以更新内容：

```python
self.gpu_expert_mask_cpu.copy_(
    expert_cache.get_cached_expert_mask().to("cpu", dtype=torch.bool),
    non_blocking=False,
)
```

但不要重新构造 MOE object。

推荐第一阶段采用策略 B，但加上版本号避免每次 copy：

```python
cache_version = expert_cache.version
if cache_version != self._gpu_mask_version:
    self.gpu_expert_mask_cpu.copy_(...)
    self._gpu_mask_version = cache_version
```

如果 `LayerExpertCache` 当前没有 version，第一阶段可以在 `heterogeneous_moe_forward` 外层每步更新，后续再优化。

---

## 11. `qwen3_moe.py` 改造

### 11.1 import

当前：

```python
from nanovllm.layers.fuse_moe.kt_backend import KtKernelCpuMoeBackend
```

新增：

```python
from nanovllm.layers.fuse_moe.kt_direct_backend import KtDirectCpuMoeBackend
```

### 11.2 enable_heterogeneous 增加 backend 分支

当前有：

```python
elif cpu_expert_backend == "kt_kernel":
    ...
    self.cpu_backend = KtKernelCpuMoeBackend(...)
```

新增：

```python
elif cpu_expert_backend == "kt_direct":
    hidden_size = expert_cache.gate_up_buffer.shape[2]
    moe_int_size = expert_cache.gate_up_buffer.shape[1] // 2

    gpu_mask = torch.zeros(self.num_experts, dtype=torch.bool)
    expert_cache_snapshot = expert_cache.get_cached_expert_mask()
    gpu_mask.copy_(expert_cache_snapshot.detach().to("cpu", dtype=torch.bool))

    self.cpu_backend = KtDirectCpuMoeBackend(
        layer_idx=self.layer_idx,
        cpu_expert_pool=cpu_expert_pool,
        max_routes=cpu_expert_workspace_max_routes,
        moe_intermediate_size=moe_int_size,
        hidden_size=hidden_size,
        num_experts=self.num_experts,
        num_experts_per_tok=self.num_selected,
        gpu_expert_mask=gpu_mask,
        kt_num_threads=kt_num_threads,
        kt_threadpool_count=kt_threadpool_count,
        kt_chunked_prefill_size=kt_chunked_prefill_size,
        kt_direct_backend=kt_method,  # 复用原参数或新增参数
        strict_dtype=cpu_expert_strict_dtype,
    )
```

建议把 `kt_method` 语义改名：

```text
kt_method:
  - BF16
  - BF16_AVX2

kt_direct_backend:
  - auto
  - amx_bf16
  - avx2_bf16
```

为了兼容 CLI，可以先复用：

```python
if kt_method in {"BF16", "auto"}:
    kt_direct_backend = "auto"
elif kt_method in {"BF16_AVX2", "AVX2", "avx2_bf16"}:
    kt_direct_backend = "avx2_bf16"
```

---

## 12. `heterogeneous.py` 改造

### 12.1 CPU backend 类型注解

当前类型写死为：

```python
cpu_backend: TorchPackedCpuMoeBackend | None = None
```

建议改成 Protocol 或 Any：

```python
from typing import Protocol

class CpuMoeBackendProtocol(Protocol):
    def forward(...) -> CpuMoeResult:
        ...
```

或简单改为：

```python
cpu_backend: object | None = None
```

### 12.2 使用 selected_experts / routing_weights

在 parallel path 中，当前调用 active CPU backend 时已经传入：

```python
selected_experts=selected_experts,
routing_weights=routing_weights,
```

标准 CPU path 中部分位置未传这两个参数。建议统一传入，确保 kt-direct 总能拿到完整 top-k，而不是只拿 cpu_indices。

当前非 parallel CPU path：

```python
cpu_result = active_cpu_backend.forward(
    hidden_states=hidden_states,
    flat_weights=flat_weights,
    top_k=top_k,
    cpu_indices=cpu_indices,
    cpu_task_expert_ids=plan.cpu_task_expert_ids,
    cpu_task_offsets=plan.cpu_task_offsets,
    act_fn=act_fn,
    parallel_mode=cpu_expert_parallel_mode,
    num_threads=cpu_expert_num_threads,
    cpu_task_expert_ids_host=plan.cpu_task_expert_ids_host,
    cpu_task_offsets_host=plan.cpu_task_offsets_host,
)
```

改成：

```python
cpu_result = active_cpu_backend.forward(
    hidden_states=hidden_states,
    flat_weights=flat_weights,
    top_k=top_k,
    cpu_indices=cpu_indices,
    cpu_task_expert_ids=plan.cpu_task_expert_ids,
    cpu_task_offsets=plan.cpu_task_offsets,
    act_fn=act_fn,
    parallel_mode=cpu_expert_parallel_mode,
    num_threads=cpu_expert_num_threads,
    cpu_task_expert_ids_host=plan.cpu_task_expert_ids_host,
    cpu_task_offsets_host=plan.cpu_task_offsets_host,
    selected_experts=selected_experts,
    routing_weights=routing_weights,
)
```

原因：kt-kernel 的 MoE forward 接收完整 `[tokens, top_k]` expert id 与 routing weights，并通过 `gpu_expert_mask` 决定哪些 expert 跳过。只传 CPU route indices 会破坏 kt-kernel 的 per-token weighted merge 语义。

### 12.3 merge 不需要大改

已有逻辑可以兼容：

```python
if cpu_outputs.shape == output.shape:
    output.add_(cpu_outputs.to(dtype=output.dtype, device=device))
```

但建议补充注释：

```python
# KtDirectCpuMoeBackend returns per-token accumulated CPU contribution.
# It already includes routing weights and only covers CPU experts
# according to gpu_expert_mask.
```

---

## 13. CLI / config 改造

新增或复用参数：

```text
--cpu-expert-backend kt_direct

--kt-direct-backend auto|amx_bf16|avx2_bf16
--kt-num-threads 0
--kt-threadpool-count 1
--kt-numa-nodes ""              # 例如 "0,1"
--kt-chunked-prefill-size 4096
--kt-capture-bs "1,2,4,8,16,32"
--kt-direct-init-at-load true
--kt-direct-strict-dtype true
--kt-direct-require-all-experts true
```

建议第一阶段默认：

```text
kt_direct_init_at_load=true
kt_direct_strict_dtype=true
kt_direct_require_all_experts=true
```

---

## 14. `pyproject.toml` / 依赖

第一阶段建议把 `kt-kernel` 作为 optional dependency，而不是默认强依赖：

```toml
[project.optional-dependencies]
kt = [
  "kt-kernel>=0.6.1",
]
```

如果 `kt-kernel` 还没有稳定 PyPI wheel，文档中要求用户按 KTransformers 指南本地安装，并确保：

```python
import kt_kernel
from kt_kernel import kt_kernel_ext
from kt_kernel_ext import moe
```

可用。

启动时检测：

```python
try:
    import kt_kernel
    from kt_kernel import kt_kernel_ext
    import kt_kernel_ext.moe as kt_moe
except Exception as e:
    raise RuntimeError(
        "cpu_expert_backend='kt_direct' requires kt-kernel / kt_kernel_ext. "
        "Install KTransformers kt-kernel first."
    ) from e
```

---

## 15. 关键伪代码

### 15.1 backend class skeleton

```python
# nanovllm/layers/fuse_moe/kt_direct_backend.py

from __future__ import annotations

import os
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import torch

from nanovllm.layers.fuse_moe.cpu_backend import CpuMoeResult


def _available_cores() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 4


def _split_threads(total: int, groups: int) -> list[int]:
    base = total // groups
    rem = total % groups
    return [base + (1 if i < rem else 0) for i in range(groups)]


class KtDirectGlobalRuntime:
    _instance = None

    @classmethod
    def get(
        cls,
        *,
        kt_num_threads: int,
        kt_threadpool_count: int,
        kt_numa_nodes: list[int] | None,
    ) -> "KtDirectGlobalRuntime":
        if cls._instance is None:
            cls._instance = cls(
                kt_num_threads=kt_num_threads,
                kt_threadpool_count=kt_threadpool_count,
                kt_numa_nodes=kt_numa_nodes,
            )
        return cls._instance

    def __init__(
        self,
        *,
        kt_num_threads: int,
        kt_threadpool_count: int,
        kt_numa_nodes: list[int] | None,
    ) -> None:
        import kt_kernel
        from kt_kernel import kt_kernel_ext
        import kt_kernel_ext.moe as kt_moe

        self.kt_kernel = kt_kernel
        self.kt_kernel_ext = kt_kernel_ext
        self.kt_moe = kt_moe

        if kt_num_threads <= 0:
            kt_num_threads = _available_cores()
        kt_threadpool_count = max(1, int(kt_threadpool_count))

        worker_config = kt_kernel_ext.WorkerPoolConfig()
        worker_config.subpool_count = kt_threadpool_count

        if kt_numa_nodes is None:
            worker_config.subpool_numa_map = list(range(kt_threadpool_count))
        else:
            if len(kt_numa_nodes) != kt_threadpool_count:
                raise ValueError("kt_numa_nodes length must equal kt_threadpool_count")
            worker_config.subpool_numa_map = [int(x) for x in kt_numa_nodes]

        worker_config.subpool_thread_count = _split_threads(
            int(kt_num_threads),
            kt_threadpool_count,
        )

        self.cpu_infer = kt_kernel_ext.CPUInfer(worker_config)
        self.kt_num_threads = int(kt_num_threads)
        self.kt_threadpool_count = int(kt_threadpool_count)
```

### 15.2 backend selection

```python
def _select_kt_bf16_moe_class(kt_moe, backend: str):
    backend = backend.strip().lower()

    amx_cls = getattr(kt_moe, "AMXBF16_MOE", None)
    avx2_cls = getattr(kt_moe, "AVX2BF16_MOE", None)

    if backend in {"auto", "bf16", ""}:
        if amx_cls is not None:
            return amx_cls, "amx_bf16"
        if avx2_cls is not None:
            return avx2_cls, "avx2_bf16"
        raise RuntimeError("No KTransformers BF16 MoE backend available.")

    if backend in {"amx", "amx_bf16"}:
        if amx_cls is None:
            raise RuntimeError("AMXBF16_MOE is not available in kt_kernel_ext.")
        return amx_cls, "amx_bf16"

    if backend in {"avx2", "avx2_bf16", "bf16_avx2"}:
        if avx2_cls is None:
            raise RuntimeError("AVX2BF16_MOE is not available in kt_kernel_ext.")
        return avx2_cls, "avx2_bf16"

    raise ValueError(f"Unsupported kt_direct_backend={backend!r}")
```

### 15.3 build weight refs / pointers

```python
def _get_raw_gate_up_down(params: dict, strict_dtype: bool):
    if "gate_up" in params and "down" in params:
        gate_up = params["gate_up"]
        down = params["down"]
    elif params.get("packed") is not None:
        gate_up = params["packed"].gate_up
        down = params["packed"].down
    else:
        raise RuntimeError("Missing gate_up/down weights")

    if strict_dtype and (gate_up.dtype != torch.bfloat16 or down.dtype != torch.bfloat16):
        raise RuntimeError(
            f"kt_direct requires BF16 weights, got gate_up={gate_up.dtype}, down={down.dtype}"
        )

    if gate_up.dtype != torch.bfloat16:
        gate_up = gate_up.to(torch.bfloat16)
    if down.dtype != torch.bfloat16:
        down = down.to(torch.bfloat16)

    if gate_up.device.type != "cpu":
        gate_up = gate_up.cpu()
    if down.device.type != "cpu":
        down = down.cpu()

    return gate_up, down


def _build_ptrs_from_cpu_pool(
    *,
    cpu_expert_pool,
    num_experts,
    hidden_size,
    intermediate_size,
    threadpool_count,
    strict_dtype,
):
    gate_ptrs_1 = []
    up_ptrs_1 = []
    down_ptrs_1 = []
    refs = []

    for eid in range(num_experts):
        params = cpu_expert_pool.get(eid)
        if params is None:
            raise RuntimeError(
                f"kt_direct requires all experts in cpu_expert_pool; missing expert {eid}"
            )

        gate_up, down_w = _get_raw_gate_up_down(params, strict_dtype)

        expected_gate_up_shape = (intermediate_size * 2, hidden_size)
        expected_down_shape = (hidden_size, intermediate_size)

        if tuple(gate_up.shape) != expected_gate_up_shape:
            raise RuntimeError(
                f"expert {eid} gate_up shape mismatch: got {tuple(gate_up.shape)}, "
                f"expected {expected_gate_up_shape}"
            )
        if tuple(down_w.shape) != expected_down_shape:
            raise RuntimeError(
                f"expert {eid} down shape mismatch: got {tuple(down_w.shape)}, "
                f"expected {expected_down_shape}"
            )

        gate = gate_up[:intermediate_size, :].contiguous()
        up = gate_up[intermediate_size:, :].contiguous()
        down = down_w.contiguous()

        refs.extend([gate, up, down])
        gate_ptrs_1.append(gate.data_ptr())
        up_ptrs_1.append(up.data_ptr())
        down_ptrs_1.append(down.data_ptr())

    gate_ptrs = [list(gate_ptrs_1) for _ in range(threadpool_count)]
    up_ptrs = [list(up_ptrs_1) for _ in range(threadpool_count)]
    down_ptrs = [list(down_ptrs_1) for _ in range(threadpool_count)]

    return gate_ptrs, up_ptrs, down_ptrs, refs
```

### 15.4 layer init

```python
class KtDirectCpuMoeBackend:
    def __init__(...):
        self.layer_idx = int(layer_idx)
        self.num_experts = int(num_experts)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(moe_intermediate_size)

        self.runtime = KtDirectGlobalRuntime.get(
            kt_num_threads=kt_num_threads,
            kt_threadpool_count=kt_threadpool_count,
            kt_numa_nodes=kt_numa_nodes,
        )

        self.gpu_expert_mask_cpu = torch.empty(
            self.num_experts,
            dtype=torch.bool,
            device="cpu",
            pin_memory=True,
        )
        self.gpu_expert_mask_cpu.copy_(gpu_expert_mask.detach().to("cpu", dtype=torch.bool))

        gate_ptrs, up_ptrs, down_ptrs, refs = _build_ptrs_from_cpu_pool(
            cpu_expert_pool=cpu_expert_pool,
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            threadpool_count=self.runtime.kt_threadpool_count,
            strict_dtype=strict_dtype,
        )
        self._weight_refs = refs

        MOEConfig = self.runtime.kt_moe.MOEConfig
        moe_config = MOEConfig(
            self.num_experts,
            self.num_experts_per_tok,
            self.hidden_size,
            self.intermediate_size,
            self.gpu_expert_mask_cpu.data_ptr(),
        )
        moe_config.layer_idx = self.layer_idx
        moe_config.pool = self.runtime.cpu_infer.backend_
        moe_config.max_len = int(kt_chunked_prefill_size)
        moe_config.gate_proj = 0
        moe_config.up_proj = 0
        moe_config.down_proj = 0
        moe_config.gate_projs = gate_ptrs
        moe_config.up_projs = up_ptrs
        moe_config.down_projs = down_ptrs
        moe_config.gate_scales = [[0] * self.num_experts for _ in range(self.runtime.kt_threadpool_count)]
        moe_config.up_scales = [[0] * self.num_experts for _ in range(self.runtime.kt_threadpool_count)]
        moe_config.down_scales = [[0] * self.num_experts for _ in range(self.runtime.kt_threadpool_count)]

        moe_cls, selected_backend = _select_kt_bf16_moe_class(
            self.runtime.kt_moe,
            kt_direct_backend,
        )
        self.kt_selected_backend = selected_backend
        self.moe = moe_cls(moe_config)

        self.physical_to_logical = torch.arange(
            self.num_experts,
            dtype=torch.int64,
            device="cpu",
        )

        # One-time pack/load before inference.
        self.runtime.cpu_infer.submit(
            self.moe.load_weights_task(self.physical_to_logical.data_ptr())
        )
        self.runtime.cpu_infer.sync()
        self.loaded = True
```

---

## 16. 性能路径说明

### 16.1 decode / 小 qlen

当 `batch_size` 很小、`qlen=1` 或 speculative verify 只处理少量 token 时：

```text
KtDirectCpuMoeBackend.forward
  -> moe.forward_task
  -> KTransformers BF16 MoE
  -> qlen 阈值判断
  -> vec_mul
  -> AVX512 BF16 / AVX2 BF16 path
```

这避免了：

```text
Python per-expert loop
ThreadPoolExecutor scheduling
F.linear small GEMM
per-route CPU output scatter
```

### 16.2 GPU/CPU overlap

当前 `heterogeneous_moe_forward` 已经有 `can_overlap_cpu_gpu` 逻辑：

```text
GPU cached expert path on separate CUDA stream
CPU backend forward in parallel
merge output
```

kt-direct forward 使用 `CPUInfer.submit_with_cuda_stream`，可以和当前 overlap 机制兼容。

第一阶段建议：

```text
--cpu-gpu-parallel-execution-enabled auto
```

并通过 profile 观察：

```text
cpu_prepare_ms
cpu_compute_ms
gpu_gather_ms
gpu_compute_ms
parallel_wall_ms
gpu_wait_ms
cpu_wait_ms
```

### 16.3 pinned buffer 预分配

decode 常见 batch size 应预分配：

```text
1,2,4,8,16,32
```

避免每层/每步创建 pinned memory。

---

## 17. 正确性验证

### 17.1 单层数值对齐

测试输入：

```text
num_tokens = 1, 2, 4, 8
top_k = model config
dtype = bf16
```

对比：

```text
TorchPackedCpuMoeBackend
vs
KtDirectCpuMoeBackend
```

需要设置：

```text
gpu_expert_mask = 全 False
```

这样所有 expert 都由 CPU backend 处理。

验收：

```text
max_abs_error <= 2e-2
mean_abs_error <= 2e-3
```

BF16 + 不同归约顺序会有轻微误差，不应要求 bitwise identical。

### 17.2 heterogeneous 语义对齐

构造：

```text
部分 expert cached on GPU
部分 expert on CPU
selected_experts 覆盖 GPU/CPU 混合
```

对比：

```text
原 heterogeneous torch CPU path
vs
kt_direct path
```

验收：

```text
输出 shape 一致
无 NaN/Inf
误差在 BF16 容忍范围内
```

### 17.3 GPU mask 动态变化

如果采用 mask 动态更新，测试：

```text
step 1: expert 0,1 cached
step 2: expert 2,3 cached
```

确保 kt_direct 只计算 CPU experts，不重复计算 GPU experts。

### 17.4 多层 smoke test

模型加载后：

```text
所有 MoE layer 初始化 kt_direct backend
所有 layer 完成 load_weights_task
发起 1 条 decode 请求
确认 forward 中没有调用 load_weights
```

建议在 backend 中加入 debug counter：

```python
self.load_count += 1
self.forward_count += 1
```

断言：

```text
load_count == 1 per layer
forward_count >= 1
```

---

## 18. 性能 benchmark 方案

### 18.1 micro benchmark

固定单层：

```text
hidden_size = model hidden
intermediate_size = model moe intermediate
num_experts = model num experts
top_k = model top-k
batch_size = 1,2,4,8,16,32
CPU expert ratio = 25%,50%,75%,100%
```

对比：

```text
torch_packed
fused
kt_direct_auto
kt_direct_avx2
```

记录：

```text
cpu_prepare_ms
cpu_compute_ms
total_cpu_ms
tokens/s
```

### 18.2 end-to-end decode benchmark

场景：

```text
prompt len fixed
decode tokens = 128 / 256
batch size = 1 / 4 / 8
GPU cache size 固定
CPU miss ratio 固定
```

记录：

```text
ttft
decode tok/s
per-layer cpu_compute_ms
parallel_wall_ms
GPU wait / CPU wait
```

### 18.3 关键观察指标

若 kt-direct 有效，应看到：

```text
cpu_compute_ms 明显下降
Python CPU backend fallback count 下降
decode tok/s 上升
load_weights 不出现在 forward profile 中
```

若 `cpu_prepare_ms` 成为瓶颈，需要继续优化：

```text
topk ids / weights pinned copy
mask update
output CPU->GPU copy
```

---

## 19. 风险与缓解

### 19.1 AMX context / destructor segfault

当前 `kt_backend.py` 使用 shared wrapper 是为了避免多 AMX context 崩溃。

第一阶段规避策略：

1. 直接用 `kt_kernel_ext.moe.AMXBF16_MOE/AVX2BF16_MOE`，不使用 `KTMoEWrapper`。
2. 全局共享一个 `CPUInfer`。
3. 每层创建一个 `moe` object，但不频繁析构。
4. 进程生命周期内保留 `moe` object。
5. 如果仍有析构问题，在 manager 中维护 `_zombie_moes`，进程结束统一释放或不主动释放。

### 19.2 内存增加

第一阶段内存增加约为：

```text
每个 CPU expert:
  gate: I * H * 2 bytes
  up:   I * H * 2 bytes
  down: H * I * 2 bytes
  raw subtotal ≈ 6 * H * I bytes

kt packed copy 约同量级，可能有 padding / NUMA TP 额外开销。
```

如果内存超限：

1. 只对 CPU fallback 会用到的 layer 启用 kt_direct；
2. 减少保留的 CPU raw weights；
3. 第二阶段让 kt packed weights 反向服务 GPU slot refill；
4. 引入 per-layer lazy init，但必须在服务开始前 warmup，不在请求内执行。

### 19.3 GPU cache 动态 mask 不一致

如果 GPU expert cache 是动态 LRU，kt 的 `gpu_expert_mask_cpu` 必须随 cache 更新。

缓解：

1. `LayerExpertCache` 增加 version；
2. backend forward 前比较 version；
3. 仅 version 变化时更新 pinned mask；
4. mask tensor 地址不变，避免重建 MOE object。

### 19.4 权重 shape / gate-up 顺序错误

Qwen3 MoE 中 gate/up 是 merged column projection。必须确认 `gate_up` 的前半是 gate，后半是 up，与 `SiluAndMul` 的语义一致。

测试：

```text
用单 expert、单 token、固定 selected_experts 测试 kt_direct vs torch path。
```

### 19.5 route 权重 dtype

KTransformers `weights_cpu` 当前常用 float32 pinned buffer。第一阶段建议将 routing weights copy 为 float32，避免 BF16 route weight 造成误差放大。

### 19.6 CUDA stream 同步错误

禁止在 forward 中直接 `torch.cuda.synchronize()`。应使用：

```text
cpu_infer.submit_with_cuda_stream
cpu_infer.sync_with_cuda_stream
```

让 CPU task 挂接到当前 CUDA stream。

---

## 20. 实施步骤

### Step 1：新增 `kt_direct_backend.py`

完成：

```text
KtDirectGlobalRuntime
KtDirectCPUBuffer
KtDirectCpuMoeBackend
```

先实现：

```text
初始化时 pack weights
forward 同步执行
返回 per-token output
```

### Step 2：接入 `qwen3_moe.py`

新增 backend 分支：

```text
cpu_expert_backend == "kt_direct"
```

保留旧的：

```text
torch
torch_packed
fused
kt_kernel
```

### Step 3：修改 `heterogeneous.py`

统一把 `selected_experts` 和 `routing_weights` 传给 CPU backend，尤其是非 parallel path。

### Step 4：增加配置

增加：

```text
kt_direct_backend
kt_numa_nodes
kt_capture_bs
kt_direct_require_all_experts
```

或复用已有 `kt_*` 参数。

### Step 5：权重加载阶段 warmup

确保所有 MoE layer 在服务开始前构造 `KtDirectCpuMoeBackend` 并完成 `load_weights_task`。

### Step 6：测试

完成：

```text
unit numerical test
single-layer benchmark
end-to-end decode benchmark
cache mask dynamic test
```

### Step 7：默认开关策略

初期默认关闭：

```text
cpu_expert_backend=torch
```

实验启用：

```text
cpu_expert_backend=kt_direct
kt_direct_backend=auto
```

---

## 21. 建议提交拆分

### PR 1：backend skeleton + dependency guard

包含：

```text
kt_direct_backend.py
runtime import guard
无 qwen3 接入
```

### PR 2：qwen3_moe 接入 + config

包含：

```text
cpu_expert_backend="kt_direct"
配置项
初始化 load
```

### PR 3：heterogeneous forward 传参修复

包含：

```text
所有 active_cpu_backend.forward 调用都传 selected_experts/routing_weights
```

### PR 4：测试与 benchmark

包含：

```text
数值测试
micro benchmark
profile counters
```

### PR 5：动态 GPU mask version 优化

包含：

```text
LayerExpertCache.version
KtDirectCpuMoeBackend.update_gpu_mask_if_needed
```

---

## 22. 第一阶段验收标准

### 功能验收

```text
[ ] cpu_expert_backend=kt_direct 可启动
[ ] 所有 MoE layer 初始化时完成 kt load_weights_task
[ ] decode forward 不调用 load_weights
[ ] GPU slots 仍可从 nano raw CPU weights 加载
[ ] heterogeneous output 与 torch CPU path BF16 误差可接受
[ ] GPU cache mask 不导致 CPU/GPU 重复计算
```

### 性能验收

```text
[ ] decode batch=1/4/8 时 cpu_compute_ms 低于 torch_packed/fused backend
[ ] CPU route 数较少时无明显 Python 调度开销
[ ] per-layer forward 中没有 safetensors read / kt pack
[ ] 端到端 decode tok/s 有可测提升
```

### 稳定性验收

```text
[ ] 多层模型启动不 segfault
[ ] 多轮请求不出现 data_ptr 悬空
[ ] 进程退出不因 kt MOE object destructor 崩溃
[ ] 无 pinned memory 泄漏
```

---

## 23. 源码依据路径

本方案依据以下源码结构设计：

```text
nano-vllm-moe:
  nanovllm/layers/fuse_moe/kt_backend.py
    - 当前 shared-wrapper + per-layer reload 方案

  nanovllm/layers/fuse_moe/cpu_backend.py
    - TorchPackedCpuMoeBackend / FusedTorchCpuMoeBackend
    - CpuMoeResult 数据结构
    - 当前 PyTorch F.linear 小 GEMM CPU path

  nanovllm/layers/fuse_moe/heterogeneous.py
    - heterogeneous_moe_forward
    - GPU cached expert path
    - CPU backend forward 接入点
    - cpu_outputs.shape == output.shape 时直接 output.add_ 的兼容逻辑

  nanovllm/models/qwen3_moe.py
    - Qwen3MoeHeterogeneousSparseMoeBlock
    - enable_heterogeneous
    - cpu_expert_backend 分支

KTransformers:
  kt-kernel/python/experts_base.py
    - BaseMoEWrapper
    - CPUInfer singleton
    - KExpertsCPUBuffer
    - submit_forward / sync_forward

  kt-kernel/python/utils/amx.py
    - AMXBF16_MOE / AVX2BF16_MOE symbol selection
    - Native BF16 MoE wrapper 思路

  kt-kernel/ext_bindings.cpp
    - MOEConfig
    - AMXBF16_MOE / AVX2BF16_MOE pybind
    - load_weights_task / forward_task

  kt-kernel/operators/amx/bf16-moe.hpp
    - BF16 MoE fused operator
    - qlen 阈值分流 mat_mul / vec_mul

  kt-kernel/operators/avx2/bf16-moe.hpp
    - AVX2 BF16 fallback
```

---

## 25. 相关参数：

示例：

```bash
  --kt-direct-backend auto \
  --kt-num-threads 12 \
  --kt-threadpool-count 1 \
  --kt-capture-bs 1,2,4,8,16,32 \
  --cpu-gpu-parallel-execution-enabled auto
```

如果机器无 AMX 或 AMX context 不稳定：

```bash
--kt-direct-backend avx2_bf16
```

---

## 26. 最小可行实现判断

第一阶段最小可行实现只需要做到：

```text
1. 新增 kt_direct_backend.py；
2. 从 cpu_expert_pool 构造 gate/up/down data_ptr；
3. 每层初始化时调用 load_weights_task 一次；
4. forward 使用 moe.forward_task；
5. qwen3_moe.py 增加 cpu_expert_backend="kt_direct"；
6. heterogeneous.py 确保传入 selected_experts/routing_weights；
7. 保持 nano raw CPU weights 用于 GPU slots。
```

只要这 7 点完成，就能验证核心目标：

```text
decode / 小 qlen CPU expert fallback 是否能从 PyTorch 小 GEMM 切换到 KTransformers AVX512/AVX2 BF16 MoE 算子。
```
