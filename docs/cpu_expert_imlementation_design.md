# nano-vllm-moe CPU MoE Expert 优化实现文档

本文档完整回顾前面对话、附件方案、KTransformers/kt-kernel 调研结果，并给出一个可实施、可回退、逐步测试的优化路线。核心原则是：

> **2026-05-08 修订提示**：A100 节点 `gpu11-A100-E1-3U` 上的实测表明，本文后续 Phase 4 中把 kt-kernel BF16 作为优先 backend 的路线不可靠。未强制 AVX2 BF16 class 时 kt-kernel 会选择不受当前 CPU 支持的 AMX BF16 路径并在 forward 中 `Illegal instruction`；即使强制 AVX2，当前 nano-vllm-moe 的 shared-wrapper layer reload 端到端 spec smoke 仍会在 `NativeMoEWrapper.load_weights()` 中 segfault。后续实现应先参考 `docs/cpu_expert_ktransformers_operator_research_20260508.md`，把 kt-kernel BF16 降级为实验 backend，并优先验证 KTransformers legacy CPU expert operator 或 nano-local C++ backend。

1. **正确性优先**：默认只实施不改变推理数值语义的优化。量化、expert deferral、近似替换、预测性跳过等可能影响精度的方案全部放入实验分支。
2. **兼容性优先**：不绕开 nano-vllm-moe 现有 heterogeneous MoE 架构，不重写 router、GPU fused MoE、speculative engine、prefetch runtime。
3. **逐项优化、逐项测试**：每个优化点完成后必须跑 correctness、regression、microbenchmark 和 end-to-end benchmark，不能一次改很多再统一测试。
4. **优先复用成熟 backend**：kt-kernel 已经提供 `KTMoEWrapper`、CPUInfer、NUMA threadpool、pinned buffer、AMX/AVX/AVX2 多变体自动选择，优先作为可选 backend 接入，而不是第一阶段复制其 C++ 内核源码。kt-kernel 明确暴露 `KTMoEWrapper`，并通过 `_cpu_detect` 自动选择 AMX、AVX512、AVX2 等变体。

------

## 0. 当前实现架构回顾

当前 nano-vllm-moe 的 CPU-GPU heterogeneous MoE 路径主要由以下模块构成：

| 模块                                        | 作用                                                         |
| ------------------------------------------- | ------------------------------------------------------------ |
| `nanovllm/utils/heterogeneous_loader.py`    | 非 expert 权重加载到 GPU；expert 权重加载到 CPU pool；初始化每层 GPU expert cache。 |
| `nanovllm/expert/cache.py`                  | 每层固定 GPU expert slot cache，维护 expert-to-slot / slot-to-expert LUT、staging slot、prefetch 状态。 |
| `nanovllm/expert/placement.py`              | 根据 selected experts 和 GPU cache，生成 `MoEExecutionPlan`，拆分 GPU routes 和 CPU routes，并将 CPU routes 按 expert 分组。 |
| `nanovllm/layers/fuse_moe/heterogeneous.py` | 实际执行 heterogeneous MoE：GPU cached expert path、CPU expert fallback path、CPU/GPU overlap、profile。 |
| `nanovllm/layers/fuse_moe/grouped_gemm.py`  | GPU fused MoE grouped GEMM，要求 CUDA contiguous input/weights/m_sizes。 |
| `nanovllm/config.py`                        | 已有 heterogeneous、CPU expert execution、prefetch、profile 等配置项。 |
| `nanovllm/engine/model_runner.py`           | 初始化 heterogeneous loader、layer caches、CPU expert pool、prefetch runtime，并下发 heterogeneous mode。 |

附件方案中有不少优化点与这些方向一致，但它假设存在 `nanovllm/layers/moe.py` 和通用 `MoELayer`，这与当前 nano-vllm-moe 的真实代码结构不一致；因此本文档会把附件中的方案适配到现有 `heterogeneous.py / placement.py / cache.py / heterogeneous_loader.py / model_runner.py` 这条路径中。

------

# 1. 总体实现顺序

推荐顺序如下：

| 阶段         | 优化项                                                       | 是否默认启用 | 正确性风险 | 实现优先级 |
| ------------ | ------------------------------------------------------------ | ------------ | ---------- | ---------- |
| Phase 0      | 建立 baseline、profile、correctness harness                  | 是           | 无         | P0         |
| Phase 1      | 移除 CPU 循环中可避免的 expert weight dtype cast；load-time contiguous/precast | 是           | 低         | P0         |
| Phase 2      | `torch_packed` CPU backend：单 buffer 输出、减少 per-expert merge/H2D | 可选启用     | 低         | P0         |
| Phase 3      | CPU-GPU 数据传输异步化 + pinned workspace 复用               | 可选启用     | 中         | P1         |
| Phase 4      | kt-kernel BF16 backend 接入                                  | 可选启用     | 中         | P1         |
| Phase 5      | kt-kernel 模式下去除重复 CPU expert pool                     | 可选启用     | 中         | P1         |
| Phase 6      | NUMA 感知线程池、CPU 线程绑定、物理核心配置                  | 可选启用     | 低         | P2         |
| Phase 7      | exact dynamic expert placement / hotness cache 策略增强      | 可选启用     | 中         | P2         |
| Phase 8      | 自研 C++ reference backend / oneDNN backend                  | 可选启用     | 中         | P3         |
| Phase 9      | AMX/AVX512 custom backend                                    | 可选启用     | 中高       | P4         |
| Experimental | INT8/FP8/INT4 量化、expert deferral、approximate prefetch    | 默认禁用     | 高         | 实验       |

其中 Phase 0 到 Phase 4 是最值得优先做的部分。

------

# 2. Phase 0：Baseline、profile 与 correctness harness

## 步骤 1：原始实现问题分析

当前优化讨论不能只看单个函数，需要先建立稳定 baseline。当前 `heterogeneous_moe_forward()` 已有 profile 字段，例如：

```text
cpu_prepare_ms
cpu_compute_ms
cpu_to_gpu_merge_ms
gpu_compute_ms
scatter_ms
parallel_wall_ms
cpu_route_ratio
cpu_weight_mass_ratio
realized_cpu_expert_count
```

这些最终通过 engine/model runner profile 聚合。`LLMEngine.get_profile()` 已经会把 model/spec/engine profile 合并，并暴露 `cpu_route_ratio`、`cpu_compute_ms` 等 canonical key。

问题是：如果没有固定 benchmark harness，后续每个优化是否有效无法判断。

## 步骤 2：设计 baseline 测试方法

需要三层测试：

1. **MoE block-level correctness**
   - 固定 `hidden_states`
   - 固定 `selected_experts`
   - 固定 `routing_weights`
   - 比较 CPU backend 修改前后输出。
2. **单层 heterogeneous MoE microbenchmark**
   - 直接测 `heterogeneous_moe_forward()`
   - 控制 CPU route ratio、activated expert 数、batch/token 数。
3. **端到端 generation benchmark**
   - 使用固定 prompt 集
   - 比较 tokens/s、decode latency、profile counters。

## 步骤 3：代码级修改方案

新增：

```text
tests/test_cpu_moe_correctness.py
benchmarks/bench_cpu_moe_backend.py
benchmarks/bench_heterogeneous_end2end.py
```

### `tests/test_cpu_moe_correctness.py`

核心检查函数：

```python
import torch


def assert_close_moe_output(ref, test, *, name: str):
    ref_f = ref.float()
    test_f = test.float()
    max_abs = (ref_f - test_f).abs().max().item()
    denom = ref_f.abs().clamp_min(1e-5)
    max_rel = ((ref_f - test_f).abs() / denom).max().item()

    assert max_abs < 5e-2, f"{name}: max_abs too large: {max_abs}"
    assert max_rel < 5e-2, f"{name}: max_rel too large: {max_rel}"
```

BF16 路径建议先用较宽容阈值，之后根据实际误差收紧。

### `benchmarks/bench_cpu_moe_backend.py`

输出 CSV：

```text
backend,mode,batch_tokens,top_k,cpu_route_ratio,activated_cpu_experts,
cpu_prepare_ms,cpu_compute_ms,cpu_to_gpu_merge_ms,gpu_compute_ms,
decode_forward_ms,tokens_per_sec,max_abs,max_rel
```

## 步骤 4：测试与数据分析

每次优化后都要对比：

| 指标                  | 解释                            | 必须满足                     |
| --------------------- | ------------------------------- | ---------------------------- |
| `max_abs_error`       | 输出最大绝对误差                | BF16 exact path 不得显著恶化 |
| `max_rel_error`       | 输出最大相对误差                | 不得显著恶化                 |
| token agreement       | 固定 seed 下生成 token 是否一致 | exact path 应高度一致        |
| `cpu_compute_ms`      | CPU expert 计算时间             | 优化项相关                   |
| `cpu_prepare_ms`      | D2H/gather/metadata 准备时间    | 异步和 buffer 相关           |
| `cpu_to_gpu_merge_ms` | CPU output 回传和 merge         | buffer/merge 相关            |
| `decode_forward_ms`   | decode 单步延迟                 | end-to-end 目标              |
| `tokens/s`            | 吞吐                            | end-to-end 目标              |

------

# 3. Phase 1：移除可避免的 expert weight dtype cast 与 load-time contiguous/precast

## 步骤 1：源码分析

当前 `_compute_real_cpu_expert_outputs()` 中的 `run_task()` 会执行：

```python
gate_up_weight = params["gate_up"].to(dtype=compute_dtype)
down_weight = params["down"].to(dtype=compute_dtype)
```

更准确地说：

- 如果 `params["gate_up"].dtype == compute_dtype`，`Tensor.to(dtype=...)` 通常不会复制 storage，但仍有 Python 调用开销。
- 如果 dtype 不一致，会在每个 expert task 中重复 cast，并分配新 tensor。
- `HeterogeneousModelLoader._to_cpu()` 当前只做 `x.to("cpu")` 和可选 `pin_memory()`，没有强制 dtype、contiguous、packed metadata。

因此 Phase 1 的目标不是声称“每次一定有大规模 cast”，而是：

1. 在 loader 阶段保证 expert weight dtype 与模型 dtype 一致。
2. 在 hot path 中移除不必要 `.to(dtype=...)`。
3. 如果 dtype 不一致，fail-fast，而不是静默重复转换。

## 步骤 2：优化设计

新增数据结构：

```python
@dataclass
class CpuExpertWeights:
    expert_idx: int
    gate_up: torch.Tensor
    down: torch.Tensor
    dtype: torch.dtype
```

加载时：

```text
safetensors expert weight
-> CPU
-> target dtype
-> contiguous
-> optional pinned memory
-> CpuExpertWeights
```

运行时：

```text
run_task()
-> 直接读取 packed.gate_up / packed.down
-> strict dtype check
-> 不再 per-task .to(dtype)
```

## 步骤 3：代码级修改方案

### 新增文件：`nanovllm/expert/cpu_weights.py`

```python
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class CpuExpertWeights:
    expert_idx: int
    gate_up: torch.Tensor
    down: torch.Tensor
    dtype: torch.dtype

    def validate(self) -> None:
        if self.gate_up.device.type != "cpu":
            raise ValueError("gate_up must be on CPU")
        if self.down.device.type != "cpu":
            raise ValueError("down must be on CPU")
        if not self.gate_up.is_contiguous():
            raise ValueError("gate_up must be contiguous")
        if not self.down.is_contiguous():
            raise ValueError("down must be contiguous")
        if self.gate_up.dtype != self.dtype:
            raise ValueError(f"gate_up dtype mismatch: {self.gate_up.dtype} != {self.dtype}")
        if self.down.dtype != self.dtype:
            raise ValueError(f"down dtype mismatch: {self.down.dtype} != {self.dtype}")
```

### 修改：`nanovllm/utils/heterogeneous_loader.py`

导入：

```python
from nanovllm.expert.cpu_weights import CpuExpertWeights
```

修改 `_to_cpu()`：

```python
def _to_cpu(
    self,
    x: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if dtype is None:
        dtype = self.hf_config.torch_dtype
    x = x.to(device="cpu", dtype=dtype).contiguous()
    return x.pin_memory() if self.pin_memory else x
```

修改 down weight 处理：

```python
if "down_proj" in weight_name:
    cpu_pool[layer_idx][expert_idx]["down"] = self._to_cpu(
        weight,
        dtype=self.hf_config.torch_dtype,
    )
```

修改 gate/up 合并处：

```python
for key, gate in pending_gate.items():
    up = pending_up[key]
    gate_up = torch.cat([gate, up], dim=0)
    layer_idx, expert_idx = key

    gate_up_cpu = self._to_cpu(gate_up, dtype=self.hf_config.torch_dtype)
    down_cpu = cpu_pool[layer_idx][expert_idx]["down"]

    packed = CpuExpertWeights(
        expert_idx=expert_idx,
        gate_up=gate_up_cpu,
        down=down_cpu,
        dtype=self.hf_config.torch_dtype,
    )
    packed.validate()

    cpu_pool[layer_idx][expert_idx]["gate_up"] = gate_up_cpu
    cpu_pool[layer_idx][expert_idx]["packed"] = packed
```

### 修改：`nanovllm/layers/fuse_moe/heterogeneous.py`

在 `_compute_real_cpu_expert_outputs()` 的 `run_task()` 中替换：

```python
gate_up_weight = params["gate_up"].to(dtype=compute_dtype)
down_weight = params["down"].to(dtype=compute_dtype)
```

为：

```python
packed = params.get("packed", None)
if packed is not None:
    gate_up_weight = packed.gate_up
    down_weight = packed.down
    if gate_up_weight.dtype != compute_dtype or down_weight.dtype != compute_dtype:
        raise RuntimeError(
            f"CPU expert {expert_idx} dtype mismatch: "
            f"gate_up={gate_up_weight.dtype}, down={down_weight.dtype}, "
            f"expected={compute_dtype}"
        )
else:
    gate_up_weight = params["gate_up"]
    down_weight = params["down"]
    if gate_up_weight.dtype != compute_dtype:
        gate_up_weight = gate_up_weight.to(dtype=compute_dtype)
    if down_weight.dtype != compute_dtype:
        down_weight = down_weight.to(dtype=compute_dtype)
```

## 步骤 4：测试与数据分析

### 单元测试

新增测试：

```python
def test_cpu_expert_weights_dtype_is_precise():
    # load a tiny fake expert
    # assert packed.gate_up.dtype == hf_config.torch_dtype
    # assert packed.down.dtype == hf_config.torch_dtype
    # assert contiguous
```

### 回归测试

跑原 backend：

```bash
python benchmarks/bench_cpu_moe_backend.py \
  --backend torch \
  --before-after phase1 \
  --dump-csv phase1.csv
```

必须满足：

| 指标             | 判定                                      |
| ---------------- | ----------------------------------------- |
| correctness      | 输出误差与 baseline 一致                  |
| `cpu_compute_ms` | 不应变差；若原本 dtype 不一致，应明显下降 |
| RAM              | 不应明显增加                              |
| fallback         | 移除 `packed` 字段后仍能走旧路径          |

------

# 4. Phase 2：`torch_packed` CPU backend，减少 per-expert merge 和 H2D

## 步骤 1：源码分析

当前 `_compute_real_cpu_expert_outputs()` 返回：

```python
token_indices_chunks: list[torch.Tensor]
cpu_outputs: list[torch.Tensor]
```

然后 `_merge_real_cpu_outputs()` 对每个 expert chunk 执行：

```python
out = cpu_out.to(device=output_device, dtype=output_dtype, non_blocking=False)
output.index_add_(0, tokens, out)
```

问题：

1. 每个 expert 一次 CPU->GPU copy。
2. 每个 expert 一次 `index_add_`。
3. Python list 管理多个小 tensor。
4. decode 小 batch 下 Python overhead 明显。

## 步骤 2：优化设计

新增 backend 抽象：

```text
CpuMoeBackend.forward()
-> 返回单个 outputs_cpu: [num_cpu_routes, hidden_size]
-> 返回 token_indices: [num_cpu_routes]
```

merge 阶段：

```python
cpu_out_gpu = outputs_cpu.to(device="cuda", non_blocking=True)
output.index_add_(0, token_indices, cpu_out_gpu)
```

这不改变计算语义，只改变中间 buffer 形态。

## 步骤 3：代码级修改方案

### 新增文件：`nanovllm/layers/fuse_moe/cpu_backend.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F

from nanovllm.layers.activation import SiluAndMul


@dataclass
class CpuMoeResult:
    token_indices: torch.Tensor
    outputs_cpu: torch.Tensor
    prep_ms: float
    compute_ms: float


class TorchPackedCpuMoeBackend:
    def __init__(
        self,
        *,
        layer_idx: int,
        cpu_expert_pool: dict[int, dict[str, torch.Tensor]],
        max_routes: int,
        strict_dtype: bool = True,
    ) -> None:
        self.layer_idx = int(layer_idx)
        self.cpu_expert_pool = cpu_expert_pool
        self.max_routes = int(max_routes)
        self.strict_dtype = bool(strict_dtype)

    def _get_weights(
        self,
        expert_idx: int,
        compute_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.cpu_expert_pool.get(int(expert_idx))
        if params is None:
            raise RuntimeError(f"Missing CPU expert weights for expert {expert_idx}")

        packed = params.get("packed", None)
        if packed is not None:
            gate_up = packed.gate_up
            down = packed.down
        else:
            gate_up = params["gate_up"]
            down = params["down"]

        if self.strict_dtype:
            if gate_up.dtype != compute_dtype or down.dtype != compute_dtype:
                raise RuntimeError(
                    f"CPU expert {expert_idx} dtype mismatch: "
                    f"gate_up={gate_up.dtype}, down={down.dtype}, expected={compute_dtype}"
                )
        else:
            if gate_up.dtype != compute_dtype:
                gate_up = gate_up.to(dtype=compute_dtype)
            if down.dtype != compute_dtype:
                down = down.to(dtype=compute_dtype)

        return gate_up, down

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
        act_fn: SiluAndMul,
        parallel_mode: str = "serial",
        num_threads: int = 4,
    ) -> CpuMoeResult:
        prep_t0 = perf_counter()
        compute_dtype = hidden_states.dtype
        num_routes = int(cpu_indices.numel())

        if num_routes > self.max_routes:
            raise RuntimeError(
                f"CPU MoE routes {num_routes} exceed max_routes={self.max_routes}"
            )

        token_indices = torch.div(cpu_indices, top_k, rounding_mode="floor")

        hidden_cpu = hidden_states.index_select(0, token_indices).to(
            "cpu", dtype=compute_dtype, non_blocking=False
        )
        weights_cpu = flat_weights.index_select(0, cpu_indices).to(
            "cpu", dtype=compute_dtype, non_blocking=False
        )

        outputs_cpu = torch.empty(
            (num_routes, hidden_states.shape[-1]),
            device="cpu",
            dtype=compute_dtype,
            pin_memory=True,
        )

        task_offsets = [int(x) for x in cpu_task_offsets.detach().to("cpu").tolist()]
        task_expert_ids = [int(x) for x in cpu_task_expert_ids.detach().to("cpu").tolist()]
        prep_ms = (perf_counter() - prep_t0) * 1000.0

        def run_task(task_idx: int) -> None:
            start = task_offsets[task_idx]
            end = task_offsets[task_idx + 1]
            if start >= end:
                return

            expert_idx = task_expert_ids[task_idx]
            gate_up_weight, down_weight = self._get_weights(expert_idx, compute_dtype)

            h = hidden_cpu[start:end]
            w = weights_cpu[start:end]

            gate_up = F.linear(h, gate_up_weight)
            out = F.linear(act_fn(gate_up), down_weight)
            out.mul_(w.unsqueeze(-1))
            outputs_cpu[start:end].copy_(out)

        compute_t0 = perf_counter()
        num_tasks = len(task_expert_ids)
        if parallel_mode == "expert_parallel" and num_threads > 1 and num_tasks > 1:
            with ThreadPoolExecutor(max_workers=min(num_threads, num_tasks)) as pool:
                list(pool.map(run_task, range(num_tasks)))
        else:
            for i in range(num_tasks):
                run_task(i)

        compute_ms = (perf_counter() - compute_t0) * 1000.0

        return CpuMoeResult(
            token_indices=token_indices,
            outputs_cpu=outputs_cpu,
            prep_ms=prep_ms,
            compute_ms=compute_ms,
        )
```

### 修改：`nanovllm/config.py`

新增：

```python
cpu_expert_backend: str = "torch"  # torch | torch_packed | kt_kernel
cpu_expert_workspace_max_routes: int = 8192
cpu_expert_strict_dtype: bool = True
```

校验：

```python
assert self.cpu_expert_backend in {"torch", "torch_packed", "kt_kernel"}
assert self.cpu_expert_workspace_max_routes >= 1
```

### 修改：`heterogeneous_moe_forward()`

新增参数：

```python
cpu_backend: object | None = None
```

替换 CPU execution 分支：

```python
if has_cpu_work:
    if cpu_expert_execution_enabled and cpu_backend is not None:
        result = cpu_backend.forward(
            hidden_states=hidden_states,
            flat_weights=flat_weights,
            top_k=top_k,
            cpu_indices=cpu_indices,
            cpu_task_expert_ids=plan.cpu_task_expert_ids,
            cpu_task_offsets=plan.cpu_task_offsets,
            act_fn=act_fn,
            parallel_mode=cpu_expert_parallel_mode,
            num_threads=cpu_expert_num_threads,
        )

        merge_t0 = perf_counter()
        cpu_out_gpu = result.outputs_cpu.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
            non_blocking=False,
        )
        output.index_add_(0, result.token_indices, cpu_out_gpu)
        merge_ms = (perf_counter() - merge_t0) * 1000.0

        _prof_add(profile, "cpu_prepare_ms", result.prep_ms / 1000.0)
        _prof_add(profile, "cpu_compute_ms", result.compute_ms / 1000.0)
        _prof_add(profile, "cpu_to_gpu_merge_ms", merge_ms / 1000.0)
    else:
        # 原逻辑不变
```

## 步骤 4：测试与数据分析

### 单元测试

- same route plan
- compare old torch backend vs `torch_packed`
- exact same dtype
- same `output.index_add_` semantics

### 集成测试

运行：

```bash
python benchmarks/bench_cpu_moe_backend.py \
  --backend torch \
  --backend torch_packed \
  --cpu-route-ratio 0.25,0.5,0.75 \
  --tokens 1,8,32,128 \
  --output phase2.csv
```

### 判定标准

| 指标                  | 预期                             |
| --------------------- | -------------------------------- |
| correctness           | 与原 torch backend 一致          |
| `cpu_to_gpu_merge_ms` | 下降，尤其 CPU experts 多时      |
| `cpu_compute_ms`      | 基本持平                         |
| `decode_forward_ms`   | CPU route ratio 高时下降         |
| regression            | CPU route ratio 低时不能明显变慢 |

------

# 5. Phase 3：CPU-GPU 数据传输异步化与缓冲区复用

## 步骤 1：源码分析

当前 CPU path 中：

```python
hidden_cpu = hidden_states.index_select(...).to("cpu", non_blocking=False)
weights_cpu = flat_weights.index_select(...).to("cpu", non_blocking=False)
```

问题：

1. `non_blocking=False`，同步 D2H。
2. 每次 forward 分配新的 CPU tensor。
3. CPU output H2D 也是同步。
4. 即使已有 CPU/GPU overlap 分支，CPU prepare 仍可能阻塞主线程。

## 步骤 2：优化设计

分两级实现。

### Phase 3A：安全低风险版

保留 `index_select()` 产生 GPU 临时 tensor，但把 D2H/H2D 改为 pinned workspace copy：

```text
hidden_gpu = hidden_states.index_select(...)
workspace.hidden_cpu[:R].copy_(hidden_gpu, non_blocking=True)
```

优点：改动小。
缺点：仍有 GPU 临时 allocation。

### Phase 3B：高级版

自写 CUDA gather kernel，直接把 selected hidden states 写入 GPU staging buffer 或直接 D2H 到 pinned buffer。Phase 3B 放后面，不作为第一实现目标。

## 步骤 3：代码级修改方案

### 新增文件：`nanovllm/layers/fuse_moe/cpu_workspace.py`

```python
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class CpuMoeWorkspace:
    max_routes: int
    hidden_size: int
    dtype: torch.dtype

    hidden_cpu: torch.Tensor
    weights_cpu: torch.Tensor
    outputs_cpu: torch.Tensor

    @classmethod
    def create(cls, max_routes: int, hidden_size: int, dtype: torch.dtype):
        return cls(
            max_routes=max_routes,
            hidden_size=hidden_size,
            dtype=dtype,
            hidden_cpu=torch.empty(
                (max_routes, hidden_size),
                device="cpu",
                dtype=dtype,
                pin_memory=True,
            ),
            weights_cpu=torch.empty(
                (max_routes,),
                device="cpu",
                dtype=dtype,
                pin_memory=True,
            ),
            outputs_cpu=torch.empty(
                (max_routes, hidden_size),
                device="cpu",
                dtype=dtype,
                pin_memory=True,
            ),
        )
```

### 修改 `TorchPackedCpuMoeBackend`

初始化时创建 workspace：

```python
self.workspace: CpuMoeWorkspace | None = None
```

forward 中：

```python
if self.workspace is None or self.workspace.dtype != compute_dtype:
    self.workspace = CpuMoeWorkspace.create(
        max_routes=self.max_routes,
        hidden_size=hidden_states.shape[-1],
        dtype=compute_dtype,
    )
ws = self.workspace
```

替换：

```python
hidden_cpu = hidden_states.index_select(...).to("cpu", ...)
```

为：

```python
hidden_gpu = hidden_states.index_select(0, token_indices)
weights_gpu = flat_weights.index_select(0, cpu_indices)

hidden_cpu = ws.hidden_cpu[:num_routes]
weights_cpu = ws.weights_cpu[:num_routes]
outputs_cpu = ws.outputs_cpu[:num_routes]

hidden_cpu.copy_(hidden_gpu, non_blocking=True)
weights_cpu.copy_(weights_gpu, non_blocking=True)

# CPU 读 hidden_cpu 前必须同步当前 CUDA stream
torch.cuda.current_stream(hidden_states.device).synchronize()
```

这一步虽然还有 synchronize，但 CPU tensor 不再重复分配，且后续可以把同步移动到 CPU worker 中。

### 进一步异步 worker 版

新增：

```python
d2h_event = torch.cuda.Event(blocking=False)
d2h_event.record(torch.cuda.current_stream(hidden_states.device))
```

CPU worker 中：

```python
d2h_event.synchronize()
run_cpu_compute()
```

这允许主线程继续提交 GPU cached expert path。

## 步骤 4：测试与数据分析

### 单元测试

- workspace overflow test
- dtype change test
- repeated forward reuse test
- different route count test

### 性能测试

重点观察：

| 指标                      | 预期           |
| ------------------------- | -------------- |
| `cpu_prepare_ms`          | 下降或稳定     |
| allocation count          | 下降           |
| `parallel_wall_ms`        | overlap 时下降 |
| `gpu_wait_ms/cpu_wait_ms` | 更接近均衡     |
| correctness               | 不变           |

------

# 6. Phase 4：kt-kernel BF16 backend 接入

## 步骤 1：源码分析

当前 nano-vllm-moe 的 CPU expert compute 是 PyTorch `F.linear()`，即使 Phase 2/3 优化 buffer，也没有改变 CPU GEMM/GEMV kernel 本身。

kt-kernel 已提供：

- `KTMoEWrapper`
- AMX/AVX/AVX2 自动选择
- `CPUInfer`
- pinned CPU buffer
- `submit_forward()` / `sync_forward()`
- BF16/native/AMXINT8/AMXINT4/LLAMAFILE 等 backend

`BaseMoEWrapper.submit_forward()` 会把 hidden/topk/weights 拷贝到 pinned CPU buffer，然后通过 `CPUInfer.submit_with_cuda_stream()` 提交 `moe.forward_task(...)`；`sync_forward()` 等待并把 output copy 回 GPU。

## 步骤 2：优化设计

第一阶段只接 `BF16` exact backend：

```text
nano router/topk/placement 不变
GPU cached experts 继续由 nano Triton grouped GEMM 处理
CPU uncached experts 改由 kt-kernel BF16 处理
kt-kernel max_deferred_experts_per_token = 0
```

关键是 `gpu_experts_mask`：

- `True`：expert 在 nano GPU cache 中，由 nano GPU path 处理。
- `False`：expert 不在 GPU cache 中，由 kt-kernel CPU backend 处理。

## 步骤 3：代码级修改方案

### 新增配置：`nanovllm/config.py`

```python
kt_method: str = "BF16"
kt_weight_path: str | None = None
kt_cpuinfer_threads: int = 0
kt_threadpool_count: int = 1
kt_chunked_prefill_size: int = 4096
kt_max_deferred_experts_per_token: int = 0
kt_numa_nodes: list[int] | None = None
```

校验：

```python
assert self.kt_max_deferred_experts_per_token == 0, (
    "Exact mode requires kt_max_deferred_experts_per_token=0"
)
```

### 新增文件：`nanovllm/layers/fuse_moe/kt_kernel_backend.py`

```python
from __future__ import annotations

import torch


class KtKernelCpuMoeBackend:
    def __init__(
        self,
        *,
        layer_idx: int,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_size: int,
        moe_intermediate_size: int,
        gpu_experts_mask: torch.Tensor,
        weight_path: str,
        method: str,
        cpuinfer_threads: int,
        threadpool_count: int,
        chunked_prefill_size: int,
        max_deferred_experts_per_token: int = 0,
        numa_nodes: list[int] | None = None,
    ) -> None:
        if max_deferred_experts_per_token != 0:
            raise ValueError("Exact kt-kernel backend requires max_deferred_experts_per_token=0")

        try:
            from kt_kernel import KTMoEWrapper
        except Exception as exc:
            raise RuntimeError("Please install kt-kernel: pip install kt-kernel") from exc

        self.layer_idx = int(layer_idx)
        self.num_experts = int(num_experts)

        self.gpu_experts_mask = torch.empty(
            (num_experts,),
            dtype=torch.bool,
            device="cpu",
            pin_memory=True,
        )
        self.gpu_experts_mask.copy_(gpu_experts_mask.detach().to("cpu", dtype=torch.bool))

        self.wrapper = KTMoEWrapper(
            layer_idx=layer_idx,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            gpu_experts_mask=self.gpu_experts_mask,
            cpuinfer_threads=cpuinfer_threads,
            threadpool_count=threadpool_count,
            weight_path=weight_path,
            chunked_prefill_size=chunked_prefill_size,
            cpu_save=False,
            max_deferred_experts_per_token=0,
            method=method,
            numa_nodes=numa_nodes,
            mode="inference",
        )

        physical_to_logical = torch.arange(
            num_experts,
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        )
        self.wrapper.load_weights(physical_to_logical)

    @torch.no_grad()
    def update_gpu_experts_mask(self, cached_expert_mask: torch.Tensor) -> None:
        mask_cpu = cached_expert_mask.detach().to("cpu", dtype=torch.bool)
        if not torch.equal(mask_cpu, self.gpu_experts_mask):
            self.gpu_experts_mask.copy_(mask_cpu)

    @torch.no_grad()
    def submit(
        self,
        *,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        cached_expert_mask: torch.Tensor,
    ) -> None:
        self.update_gpu_experts_mask(cached_expert_mask)
        stream = torch.cuda.current_stream(hidden_states.device).cuda_stream

        self.wrapper.submit_forward(
            hidden_states.contiguous(),
            selected_experts.to(torch.long).contiguous(),
            routing_weights.contiguous(),
            stream,
        )

    @torch.no_grad()
    def sync(self, *, hidden_states: torch.Tensor) -> torch.Tensor:
        stream = torch.cuda.current_stream(hidden_states.device).cuda_stream
        return self.wrapper.sync_forward(hidden_states.contiguous(), stream)

    @torch.no_grad()
    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        cached_expert_mask: torch.Tensor,
    ) -> torch.Tensor:
        self.submit(
            hidden_states=hidden_states,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            cached_expert_mask=cached_expert_mask,
        )
        return self.sync(hidden_states=hidden_states)
```

### 修改 `heterogeneous_moe_forward()`

增加参数：

```python
kt_cpu_backend: object | None = None
```

在 GPU path 前提交：

```python
kt_submitted = False
if has_cpu_work and kt_cpu_backend is not None:
    t0 = perf_counter()
    kt_cpu_backend.submit(
        hidden_states=hidden_states,
        selected_experts=selected_experts,
        routing_weights=routing_weights,
        cached_expert_mask=expert_cache.get_cached_expert_mask(),
    )
    kt_submitted = True
    _prof_add(profile, "kt_submit_ms", perf_counter() - t0)
```

GPU path 保持不变。GPU path 后：

```python
if kt_submitted:
    t0 = perf_counter()
    cpu_partial = kt_cpu_backend.sync(hidden_states=hidden_states)
    output.add_(cpu_partial.to(device=output.device, dtype=output.dtype))
    _prof_add(profile, "kt_sync_merge_ms", perf_counter() - t0)
elif has_cpu_work:
    # 原 torch CPU path
```

## 步骤 4：测试与数据分析

### 正确性测试

比较：

```text
backend=torch
backend=kt_kernel, kt_method=BF16
```

必须测试：

1. CPU-only routes。
2. GPU-only routes。
3. GPU+CPU mixed routes。
4. dynamic GPU cache mask 改变。
5. top-k > 1。
6. repeated decode steps。

### 兼容性测试

- `cpu_expert_backend="torch"` 完全保持原逻辑。
- 未安装 `kt-kernel` 时，只有显式启用 `kt_kernel` 才报错。
- `kt_kernel` 失败时不能 silent fallback，避免性能结果误判。

### 性能测试

重点看：

| 指标                              | 预期                                         |
| --------------------------------- | -------------------------------------------- |
| `cpu_compute_ms`                  | 明显下降，尤其 CPU route 多时                |
| `kt_submit_ms + kt_sync_merge_ms` | 应低于原 `cpu_prepare + cpu_compute + merge` |
| end-to-end decode                 | CPU route ratio 高时提升                     |
| CPU route ratio 低                | 不应明显变差                                 |

------

# 7. Phase 5：kt-kernel 模式下去除重复 CPU expert pool

## 步骤 1：源码分析

如果 Phase 4 按最小接入方式实现，会出现两份 CPU expert weights：

1. nano `HeterogeneousModelLoader` 加载的 `cpu_expert_pool`
2. kt-kernel 自己从 `kt_weight_path` 加载的 CPU weights

这在大模型上会显著增加 RAM。

## 步骤 2：优化设计

当：

```python
cpu_expert_backend == "kt_kernel"
```

时：

- 不构建完整 `cpu_expert_pool`
- GPU cache 初始化仍需要读取一部分 experts 到 GPU slot
- kt-kernel 持有 CPU compute weights

流程：

```text
load non-expert weights -> GPU
for initial GPU cached experts:
    read expert weights from safetensors
    put_to_slot()
    release temporary tensor
kt-kernel:
    load all CPU compute weights internally
```

## 步骤 3：代码级修改方案

### 修改：`HeterogeneousModelLoader.load()`

增加分支：

```python
if getattr(self.config, "cpu_expert_backend", "torch") == "kt_kernel":
    self._load_non_expert_weights(model, path)
    layer_caches = self._init_layer_caches_lightweight(path)
    self._load_initial_placement_lightweight(layer_caches, path)
    cpu_pool = {}
    torch.cuda.synchronize()
    return layer_caches, cpu_pool
```

新增：

```python
def _init_layer_caches_lightweight(self, path: str) -> dict[int, LayerExpertCache]:
    # 从 config/hf_config 获取 num_experts、moe_intermediate_size、hidden_size
    # 不加载完整 expert pool，只构造 shape
```

如果 shape 不易从 config 直接拿，则读取第一个 expert tensor 作为 sample，读取完释放。

### 注意事项

- 如果现有 prefetch runtime 依赖 `cpu_expert_pool` 做 GPU cache refill，Phase 5 必须先禁用 dynamic prefetch 或改为 safetensors 按需读取。
- Phase 5 不应该和 Phase 4 同时实现。先证明 kt-kernel backend 有收益，再做内存优化。

## 步骤 4：测试与数据分析

| 测试                          | 目标                                         |
| ----------------------------- | -------------------------------------------- |
| RAM usage before/after        | 验证 CPU RAM 显著下降                        |
| initial GPU cache correctness | 初始 cached experts 权重正确                 |
| kt-kernel forward correctness | CPU experts 输出正确                         |
| prefetch disabled test        | 确认关闭 prefetch 不影响普通 heter mode      |
| prefetch enabled test         | 若开启，必须确认不会访问空 `cpu_expert_pool` |

------

# 8. Phase 6：NUMA 感知、CPU 线程绑定、物理核心配置

## 步骤 1：源码分析

当前 `cpu_expert_parallel_mode="expert_parallel"` 使用 Python `ThreadPoolExecutor`。它没有：

- 物理核心识别
- NUMA node 分配
- thread affinity
- 避免 hyperthread oversubscription
- CPU memory locality

kt-kernel 已有 `WorkerPoolConfig.subpool_count/subpool_numa_map/subpool_thread_count`，并由 `BaseMoEWrapper._get_cpu_infer()` 构造 CPUInfer singleton。

## 步骤 2：优化设计

分两条路径：

### torch backend

新增可选 thread pinning：

```text
cpu_thread_affinity = none | compact | scatter
cpu_numa_policy = none | local | interleave
```

但 Python `ThreadPoolExecutor` 很难可靠绑定每个 worker。这个路径收益有限。

### kt-kernel backend

优先使用：

```python
kt_cpuinfer_threads = physical_core_count
kt_threadpool_count = numa_node_count
kt_numa_nodes = [...]
```

不要自己写 NUMA allocator，先复用 kt-kernel。

## 步骤 3：代码级修改方案

### `config.py`

```python
cpu_thread_affinity: str = "none"  # none | compact | scatter
cpu_numa_policy: str = "none"      # none | kt_kernel
```

校验：

```python
assert self.cpu_thread_affinity in {"none", "compact", "scatter"}
assert self.cpu_numa_policy in {"none", "kt_kernel"}
```

### 自动建议函数

新增：

```text
nanovllm/utils/cpu_topology.py
import os
import subprocess


def detect_physical_cores() -> int:
    try:
        out = subprocess.check_output(["lscpu"]).decode()
        cpus = threads_per_core = None
        for line in out.splitlines():
            if line.startswith("CPU(s):"):
                cpus = int(line.split(":")[1].strip())
            if line.startswith("Thread(s) per core:"):
                threads_per_core = int(line.split(":")[1].strip())
        if cpus and threads_per_core:
            return cpus // threads_per_core
    except Exception:
        pass
    return os.cpu_count() or 1


def detect_numa_nodes() -> int:
    try:
        out = subprocess.check_output(["lscpu"]).decode()
        for line in out.splitlines():
            if line.startswith("NUMA node(s):"):
                return int(line.split(":")[1].strip())
    except Exception:
        pass
    return 1
```

在 `ModelRunner` 初始化 kt 参数时：

```python
if config.kt_cpuinfer_threads <= 0:
    config.kt_cpuinfer_threads = detect_physical_cores()
if config.kt_threadpool_count <= 0:
    config.kt_threadpool_count = detect_numa_nodes()
```

## 步骤 4：测试与数据分析

运行：

```bash
numactl --hardware
numastat -p $(pgrep -f nano)
perf stat -e task-clock,context-switches,cpu-migrations python benchmarks/bench_heterogeneous_end2end.py
```

比较：

| 配置                             | 指标                       |
| -------------------------------- | -------------------------- |
| `kt_threadpool_count=1`          | baseline                   |
| `kt_threadpool_count=numa_nodes` | NUMA subpool               |
| hyperthreads vs physical cores   | `cpu_compute_ms`、tokens/s |

判定：

- 单 socket：收益可能接近 0。
- 双 socket：如果 CPU expert compute 占比高，应看到 `cpu_compute_ms` 或 end-to-end latency 改善。

------

# 9. Phase 7：exact dynamic expert placement / hotness cache 策略增强

## 步骤 1：源码分析

附件方案提出 `ExpertHotnessCache` 和 residual prefetch predictor。问题是 nano-vllm-moe 已经有：

- `LayerExpertCache.mark_access`
- `access_count`
- `access_score_sum`
- `last_access_step`
- `cache_strategy`
- `prefetch_strategy`
- `PrefetchRuntime`

新建第二套 cache 会破坏兼容性。`LayerExpertCache` 已维护 GPU cache slot 与 expert ID 的映射，并支持 staging/publish。

## 步骤 2：优化设计

只做 exact placement：

```text
观测路由频率 / routing weight mass
选择 hot experts
将 hot experts prefetch 到 GPU cache
CPU experts 仍精确计算
不做 expert substitution
不做 deferral
不做跳过
```

增强现有策略，而不是新建 dispatcher。

## 步骤 3：代码级修改方案

### 修改 `cache_strategy`

新增策略：

```text
score = alpha * activation_count + beta * routing_weight_sum - gamma * age
```

在现有 `cache_strategy` 中新增：

```python
class WeightedHotnessCacheStrategy:
    def score(self, snapshot, expert_idx, step_id):
        age = max(0, step_id - snapshot.last_access_step[expert_idx])
        return (
            self.activation_weight * snapshot.access_count[expert_idx]
            + self.score_weight * snapshot.access_score_sum[expert_idx]
            - self.age_penalty * age
        )
```

### 修改 `config.py`

```python
cache_strategy: str = "lru"  # existing; add weighted_hotness
cache_hotness_activation_weight: float = 1.0
cache_hotness_score_weight: float = 1.0
cache_hotness_age_penalty: float = 0.01
```

## 步骤 4：测试与数据分析

| 测试                         | 指标                |
| ---------------------------- | ------------------- |
| fixed prompt repeated decode | GPU expert hit rate |
| random prompts               | no regression       |
| long-context prefill         | prefetch late count |
| speculative mode             | verify correctness  |

重点 profile：

```text
cpu_route_ratio
cpu_weight_mass_ratio
realized_cpu_expert_count
prefetch_submit_count
prefetch_completed_count
prefetch_late_count
publish_count
publish_ms
```

正确性要求：

- 不能 substitute expert。
- 不能使用 stale GPU cache slot。
- staging publish 必须保证 event 完成后再更新 LUT。

------

# 10. Phase 8：自研 C++ reference backend / oneDNN backend

## 步骤 1：源码分析

kt-kernel 是最佳短期方案，但长期可能有以下原因需要自研：

- 想避免外部依赖。
- 只需要 BF16 exact，不需要量化。
- 想深度控制 buffer/layout/profile。
- kt-kernel 接口不满足 nano 特定 route layout。

## 步骤 2：优化设计

先做 C++ reference backend，不追求性能：

```text
input:
    hidden_cpu [R, H]
    route_weights_cpu [R]
    task_expert_ids [T]
    task_offsets [T+1]
    per-expert gate_up/down pointers

for task in experts:
    compute gate_up
    silu_and_mul
    compute down
    multiply route weight
    write output_cpu
```

再做 oneDNN backend：

```text
M_e 小：继续 torch/small kernel
M_e 大：oneDNN matmul
```

## 步骤 3：代码级修改方案

新增：

```text
nanovllm/csrc/cpu_moe/cpu_moe_ref.cpp
nanovllm/layers/fuse_moe/cpu_ext.py
```

第一版用 `torch.utils.cpp_extension.load()` JIT 编译，不改 pyproject。

回退机制：

```text
cpu_expert_backend="cpp_ref"
import/compile 失败 -> 明确报错
默认 backend 仍是 torch
```

## 步骤 4：测试与数据分析

C++ reference 只要求 correctness：

| 指标             | 要求                  |
| ---------------- | --------------------- |
| max_abs/max_rel  | 与 torch backend 接近 |
| cpu_compute_ms   | 不要求快              |
| memory leak      | 无                    |
| repeated forward | 稳定                  |

oneDNN backend 才要求性能提升。

------

# 11. Phase 9：AMX/AVX512 custom backend

## 步骤 1：源码分析

附件提出分层 SIMD kernel：AMX、AVX512、AVX2。方向正确，但第一阶段自写成本极高。KTransformers 的 AMX 文件显示，其 FP8/AMX MoE operator 依赖 CRTP base、packed BufferB、group scale、work stealing pool、NUMA subpool，并不是单个 kernel 函数。

## 步骤 2：优化设计

仅当 kt-kernel 无法满足时再做。

原则：

1. 权重 load-time packing。
2. 不在 forward 时搬运/合并 expert weights。
3. 不 padding M 维度作为默认策略。
4. 小 M 用 GEMV-like kernel。
5. 大 M 用 AMX/BRGEMM-like kernel。
6. tail 在 kernel 内处理。

## 步骤 3：代码级修改方案

新增 layout：

```python
@dataclass
class PackedCpuExpertWeights:
    expert_idx: int
    gate_packed: object
    up_packed: object
    down_packed: object
    layout: str
    dtype: torch.dtype
```

新增 backend：

```text
nanovllm/layers/fuse_moe/cpu_backend_amx.py
nanovllm/csrc/cpu_moe/amx_bf16/*
```

但不建议进入近期实现计划。

## 步骤 4：测试与数据分析

需要独立 microbench：

```text
M_e = 1, 2, 4, 8, 16, 32, 64, 128
H = hidden_size
I = moe_intermediate_size
dtype = bf16
```

指标：

```text
GFLOP/s
cpu_compute_ms
LLC miss
memory bandwidth
thread scaling
```

------

# 12. Experimental：附件中高风险优化项处理

## 12.1 Expert Deferral

附件中已标注废弃。结论：默认禁止。

原因：

- MoE 层输出必须在后续 residual/attention/MLP 使用前完整可见。
- deferred expert 如果跨层重叠，需要严格证明数学等价。
- KTransformers 的 deferral 可能是系统级流水线优化，但 nano-vllm-moe 中先不引入。

配置：

```python
kt_max_deferred_experts_per_token = 0
```

若未来实验：

```python
experimental_allow_approximate_moe = True
```

必须与 exact mode 隔离。

## 12.2 INT8 / FP8 / INT4 量化

默认禁止作为 exact 优化。

可选 benchmark：

```text
kt_method=AMXINT8
kt_method=AMXINT4
kt_method=FP8
```

必须单独跑：

- perplexity delta
- generation token agreement
- 下游任务分数
- 长文本稳定性

## 12.3 Residual prefetch predictor

不直接引入新 `ResidualPrefetchPredictor`。如要实现，应作为现有 `prefetch_strategy` 的新策略，而不是新建 `nanovllm/cpu_moe/prefetch.py`。

## 12.4 vLLM miss expert buffer

仅适用于“CPU 存储、GPU 计算”的 miss expert copy 模式。若采用 kt-kernel CPU compute，优先级低。可作为未来 GPU cache refill 的优化。

------

# 13. 每个阶段的回退机制

| 阶段         | 新 flag                             | 默认值  | 回退方式                           |
| ------------ | ----------------------------------- | ------- | ---------------------------------- |
| Phase 1      | 无或 `cpu_expert_strict_dtype`      | strict  | 移除 `packed` 字段走旧 dict tensor |
| Phase 2      | `cpu_expert_backend=torch_packed`   | `torch` | 改回 `torch`                       |
| Phase 3      | `cpu_expert_async_transfer_enabled` | false   | 关闭异步 workspace                 |
| Phase 4      | `cpu_expert_backend=kt_kernel`      | `torch` | 改回 `torch`                       |
| Phase 5      | `kt_avoid_duplicate_cpu_pool`       | false   | 恢复完整 CPU pool                  |
| Phase 6      | `cpu_numa_policy`                   | none    | 关闭 NUMA                          |
| Phase 7      | `cache_strategy=weighted_hotness`   | lru     | 改回 lru                           |
| Phase 8      | `cpu_expert_backend=cpp_ref`        | torch   | 改回 torch                         |
| Experimental | `experimental_*`                    | false   | 全部关闭                           |

------

# 14. 基准测试总方案

## 14.1 测试矩阵

每个阶段单独测试：

```text
backend:
    torch
    torch_packed
    kt_kernel_BF16
    cpp_ref
    oneDNN/AMX experimental

mode:
    heter
    spec, 但先关闭 approximate deferral

GPU expert cache ratio:
    0%, 25%, 50%, 75%, 100%

batch / tokens:
    decode batch 1, 4, 16, 64
    prefill tokens 512, 2048, 8192

CPU:
    single socket
    dual socket if available
```

## 14.2 输出数据模板

每次 benchmark 输出：

| field                     | 说明                             |
| ------------------------- | -------------------------------- |
| git_commit                | 当前 commit                      |
| backend                   | torch / torch_packed / kt_kernel |
| model                     | 模型路径                         |
| dtype                     | bf16/fp16                        |
| cpu_threads               | CPU 线程数                       |
| numa_nodes                | NUMA 设置                        |
| gpu_slots_per_layer       | GPU expert cache slots           |
| decode_forward_ms         | decode 单步                      |
| prefill_forward_ms        | prefill                          |
| tokens_per_sec            | 端到端吞吐                       |
| cpu_route_ratio           | CPU route 占比                   |
| realized_cpu_expert_count | CPU expert 数                    |
| cpu_prepare_ms            | CPU prepare                      |
| cpu_compute_ms            | CPU compute                      |
| cpu_to_gpu_merge_ms       | CPU output merge                 |
| gpu_compute_ms            | GPU expert compute               |
| max_abs_error             | correctness                      |
| max_rel_error             | correctness                      |
| token_agreement           | generation correctness           |

## 14.3 数据分析方法

每个阶段都生成对比：

```text
speedup_cpu_compute = baseline_cpu_compute_ms / optimized_cpu_compute_ms
speedup_decode = baseline_decode_forward_ms / optimized_decode_forward_ms
merge_reduction = 1 - optimized_merge_ms / baseline_merge_ms
```

必须分 CPU route ratio 分析：

| CPU route ratio | 预期                                |
| --------------- | ----------------------------------- |
| 0%              | 不应有显著回归                      |
| 10%             | 优化收益可能小                      |
| 50%             | CPU backend 优化应明显              |
| 90%             | kt-kernel / packed backend 应最明显 |

------

# 15. 推荐工作计划

## 第 1 步：Phase 0 + Phase 1

目标：

```text
建立测试基线
修复 expert weight dtype/precast 问题
不改变计算流
```

交付：

```text
tests/test_cpu_moe_correctness.py
benchmarks/bench_cpu_moe_backend.py
CpuExpertWeights
loader precast/contiguous
hot path strict dtype check
```

## 第 2 步：Phase 2

目标：

```text
torch_packed backend
单 buffer CPU output
单次 H2D merge
```

交付：

```text
cpu_backend.py
Config.cpu_expert_backend
heterogeneous.py backend hook
```

## 第 3 步：Phase 3

目标：

```text
pinned workspace reuse
非阻塞 copy 初步实现
```

交付：

```text
cpu_workspace.py
workspace overflow check
profile: workspace_reuse_count
```

## 第 4 步：Phase 4

目标：

```text
接入 kt-kernel BF16 exact backend
```

交付：

```text
kt_kernel_backend.py
Config kt_* fields
ModelRunner 参数下发
heterogeneous.py submit/sync path
```

## 第 5 步：Phase 5

目标：

```text
kt-kernel backend 去重 CPU expert pool
降低 RAM
```

交付：

```text
heterogeneous_loader lightweight mode
initial GPU cache loading without full cpu_pool
prefetch compatibility check
```

## 第 6 步：Phase 6 + Phase 7

目标：

```text
NUMA/thread tuning
exact hotness cache strategy
```

交付：

```text
cpu_topology.py
weighted_hotness cache strategy
benchmark on single/dual NUMA
```

## 第 7 步：Experimental

仅在 exact backend 稳定后进行：

```text
AMXINT8/INT4
FP8
deferral
custom AMX/AVX
residual prefetch
```

------

# 16. 最终总结

最优实现路线不是一次性重写 CPU MoE，而是：

```text
先保证可测：
    baseline + correctness + profile

再做无风险优化：
    expert weight precast/contiguous
    torch_packed 单 buffer merge
    pinned workspace

再接成熟 backend：
    kt-kernel BF16 exact backend

再做系统优化：
    去重 CPU expert pool
    NUMA/threadpool
    exact dynamic placement

最后才做实验项：
    quantization
    deferral
    custom AMX/AVX
```

其中最值得优先投入的是 **Phase 1 到 Phase 4**。它们满足：

1. **正确性风险最低**：不改变 expert 选择、不替换 expert、不量化、不跳过计算。
2. **兼容性最好**：保留现有 `placement.py`、`LayerExpertCache`、GPU fused MoE、profile、engine API。
3. **性能收益路径清晰**：先减少 Python/merge/alloc 开销，再用 kt-kernel 替换 PyTorch CPU expert compute。
4. **回退简单**：全部通过 `cpu_expert_backend` 和相关 flags 控制，默认仍可回到当前 torch path。
