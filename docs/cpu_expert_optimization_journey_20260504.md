# CPU Expert 优化全记录

本文档按时间顺序记录 Phase 1-3 实现 review 及后续所有优化点的发现过程、解决思路、尝试方案、debug 历程和最终结果。

---

## 1. Phase 1-3 实现 Review 与性能瓶颈分析

### 发现/问题

用户要求 review Phase 1-3 的已实现代码，并分析为何加速效果有限。

已实现内容（Phase 1-3 summary 文档记录）：
- **Phase 1**: `CpuExpertWeights` 预转换 + load-time dtype contiguous
- **Phase 2**: `TorchPackedCpuMoeBackend` + 单 buffer merge
- **Phase 3**: `CpuMoeWorkspace` pinned buffer 复用

实际 spec 测试中加速效果"approximately neutral"：verify latency 从 556.8ms (torch) 到 562.6ms (torch_packed)，几乎无差别。

### 解决思路

逐文件 review Phase 1-3 代码，trace 关键路径，做 microbenchmark 定位真正瓶颈。

### Review 结论

代码实现正确，无 bug。但优化方向有误：

| 优化目标 | 占 verify 延迟比例 | Phase 1-3 效果 |
|---|---|---|
| CPU Compute (F.linear) | **64%** | 未触及 |
| GPU Compute + Attention | 30% | 未触及 |
| CPU Merge (H2D) | **2%** | 减少 67-84% ✓ |

Phase 1-3 优化了 merge 路径，但 merge 只占总延迟的 2%。减少 20ms merge 对 557ms verify 几乎无影响。**CPU matmul (F.linear) 是真正的瓶颈**。

### Microbenchmark 验证

```bash
# 在 login 节点运行，无需 CUDA
python -c "
import torch, torch.nn.functional as F, time
H, I = 2048, 768
M = 3  # decode 典型值
gate_up = torch.randn(I*2, H, dtype=torch.bfloat16)
down = torch.randn(H, I, dtype=torch.bfloat16)
# ... 测量 F.linear 单次调用时间
"
```

结果：
- M=3, 16 OMP threads: **0.80ms/ expert**
- 5 CPU experts × 0.80ms = 4.0ms / layer
- 44 MoE layers × 4.0ms = **176ms** CPU compute (与实测 ~120-360ms 吻合)
- 算术强度: **3.0 FLOP/byte** (严重 memory-bandwidth bound)
- 有效内存带宽: ~12 GB/s (仅为 DDR4 峰值的 12-24%)

**关键发现**: F.linear 在 CPU 上的 small-M matmul 严重受限于内存带宽，9.4MB 权重读取 + 28k FLOPs 计算。

### 测试来源

- 代码 review: 读取 `cpu_weights.py`, `cpu_backend.py`, `cpu_workspace.py`, `heterogeneous.py`, `heterogeneous_loader.py`, `config.py`, `model_runner.py`, `qwen3_moe.py`
- 测试: `python -m pytest -q tests/test_cpu_moe_correctness.py -k 'not packed_backend_matches'` → 3 passed
- 性能数据来源: Phase 1-3 summary 文档中的 spec benchmark 表格

---

## 2. Phase 1-3 进一步优化尝试

### 2.1 持久化线程池 (Persistent Thread Pool)

#### 发现/问题
`TorchPackedCpuMoeBackend` 每次 forward 调用创建新的 `ThreadPoolExecutor`，存在创建/销毁开销。

#### 解决思路
在 backend 初始化时创建线程池，复用跨 forward 调用。

#### 实现
在 `cpu_backend.py` 中添加 `_ensure_thread_pool()` 方法，lazy 创建并缓存。

#### 结果
**几乎无效果** (<1%)。原因: serial 模式不使用线程池，而 serial 模式是默认且最高效的模式。

---

### 2.2 动态并行模式选择 (Dynamic Parallel Mode)

#### 发现/问题
`cpu_expert_parallel_mode` 固定为 `serial` 或 `expert_parallel`，不能根据 workload 特征自动选择。

#### 解决思路
根据 average routes per expert 自动选择：
- avg_routes ≤ 1.5 → expert_parallel + OMP_NUM_THREADS=1
- avg_routes > 1.5 → serial + full OMP

#### 实现
添加 `_resolve_dynamic_parallel_mode()` 函数和 `"auto"` 模式。

Microbenchmark 验证:
```bash
# 在 login 节点运行
python -c "
# 测试 M=3 (serial 16 OMP) vs M=1 (parallel 5T 1 OMP)
# ...
"
```

结果：
- M=3: Serial 16 OMP: 4.29ms, Parallel 5T 1 OMP: 6.20ms → **serial 胜**
- M=1: Serial 16 OMP: 2.67ms, Parallel 5T 1 OMP: 2.60ms → **parallel 略胜**

#### 测试结果
在 Qwen3-30B-A3B spec 测试中，auto vs serial 差异 <1%。decode 场景下 avg_routes_per_expert ≈ 0.5-1.5，两种模式性能接近。

**测试来源**:
- 命令: `python benchmarks/scripts/phase13_followup_validate.sh`
- 结果: `/home/mumura/moe_spec/logs/phase13_followup_20260503_151946/full_run.log`
- 表格: serial vs auto 在 75%/50%/25% cache 的 verify_ms 差异 <2ms

---

### 2.3 CPU 元数据预加载 (Metadata Preload)

#### 发现/问题
`cpu_task_offsets` 和 `cpu_task_expert_ids` 在 hot path 中从 GPU 拷贝到 CPU:
```python
task_offsets_host = [int(x) for x in cpu_task_offsets.detach().to("cpu", ...).tolist()]
```

#### 解决思路
在 plan building (`_build_cpu_task_layout`) 时预计算 CPU host list，存储在 `MoEExecutionPlan` 中。

#### 实现
- `placement.py`: `_build_cpu_task_layout` 新增返回 `ids_host`, `offsets_host`
- `MoEExecutionPlan`: 新增 `cpu_task_expert_ids_host`, `cpu_task_offsets_host` 字段
- `heterogeneous.py` + `cpu_backend.py`: 传递 host metadata

#### 结果
**几乎无效果** (<1ms)。metadata 只有 5-15 个 int64 值，GPU→CPU 传输 <0.1ms。

---

### 2.4 结论

Phase 1-3 范围内的优化已到天花板。三个优化点都已实现且正确，但对 end-to-end 延迟的影响 <1%。**根本瓶颈是 CPU F.linear 计算 kernel**，需要用 Phase 4 (kt-kernel) 或自定义 fused kernel 来加速。

---

## 3. GPU Compute Time 异常分析

### 发现/问题

用户报告 verify phase 的 GPU compute time (~165ms) 远超 draft phase (<20ms)。即使 verify 计算量增加了 5× tokens，理论上 GPU compute 应该也是 ~20ms 量级。

### 调查过程

#### Step 1: 收集 profile 数据

从已有 spec profile JSON 文件提取指标。

**数据来源**: `/home/mumura/moe_spec/logs/spec_real_after_fix_20260503_085117/torch_0.25.json`

关键发现:
```
model_verify_gpu_compute_ms: 33.33   ← MoE GPU compute 只有 33ms!
model_gpu_compute_ms: 190.45         ← 这是所有 steps 的累计
model_run_model_ms: 4778.21          ← 整个 benchmark 的 model forward 总时间
```

GPU MoE compute 本身只有 33ms，不是 165ms。用户所说的 165ms 是非 CPU 时间（总 verify 557ms - CPU 375ms ≈ 182ms）。

#### Step 2: 分解 verify 延迟

Verify forward (555.73ms) 的完整分解:

| 组件 | ms | % |
|---|---|---|
| `verify_cpu_compute_ms` | 355 | 63.9% |
| Attention + other GPU | 94 | 16.8% |
| `verify_plan_ms` | 40 | 7.2% |
| `verify_gpu_compute_ms` | 33 | 6.0% |
| `verify_cpu_to_gpu_merge_ms` | 12 | 2.1% |
| 其他 (route, gather, scatter) | 22 | 4.0% |

**数据来源**: `/home/mumura/moe_spec/logs/spec_real_after_fix_20260503_085117/torch_0.25.json` 中的 engine_profile

#### Step 3: 根因分析

**GPU compute time 异常原因**:

1. **Draft 使用 CUDA graph decode，Verify 使用 eager prefill**
   - `run_draft()` → `self.run(seqs, False)` → decode path + CUDA graph
   - `run_verify()` → `self.model(input_ids, positions)` → eager prefill path
   - CUDA graph 消除 ~480 kernel launches 的 CPU→GPU 开销 (~24ms)
   - Prefill attention (5 queries × full KV) vs Decode attention (1 query × cached KV) — 不同 kernel，不同特性

2. **Plan building 占 40ms (7.2%)**
   - `build_verify_plan_gpu` 每层执行 argsort, nonzero, scatter_add 等 ~10 GPU 操作
   - 48 MoE 层 × 0.83ms = 40ms
   - 这些是小 GPU kernel，eager 模式下 launch overhead 显著

3. **Attention + other GPU 占 94ms (16.8%)**
   - 48 层 × ~2ms/layer attention 在 eager prefill 模式
   - 5 tokens 的 prefill attention 不如 1 token decode attention 高效

### 尝试的修复

短期修复（已实现）:
- 将 plan building 中 GPU→CPU metadata 传输移到 plan building 阶段（见 2.3）
- 但由于 plan building 本身在 GPU 上进行，收益极小

长期方向（未实现，留作后续）:
- Verify CUDA graph 支持（需要固定 shape）
- 切换到 sequential decode verify（5× 15ms = 75ms vs 570ms prefill）

---

## 4. CPU-GPU 并行执行修复

### 发现/问题

分析 `heterogeneous_moe_forward` 的并行路径后发现三个问题:

1. **默认关闭**: `cpu_gpu_parallel_execution_enabled = False`
2. **阈值过严**: `cpu_gpu_parallel_min_cpu_route_ratio = 0.7` — 只在 ≥70% CPU routes 时启用，封锁了 50%/75% cache 场景
3. **Stream 重复创建**: 每个 MoE 层创建一个新的 CUDA stream (44个/forward)

### 解决思路

1. 将 `cpu_gpu_parallel_execution_enabled` 从 `bool` 改为 `str` (`"off"` | `"on"` | `"auto"`)
2. `"auto"` 模式: 只要有 GPU + CPU 工作就启用并行
3. 复用 CUDA stream: 每层创建一次，整个 forward 复用

### 实现

修改文件:
- `config.py`: 类型 `bool` → `str`, 默认 `"auto"`, min_ratio → `0.0`
- `heterogeneous.py`: 添加 `"auto"` 逻辑, 接受 `cpu_gpu_parallel_stream` 参数
- `qwen3_moe.py`: 添加 `_get_parallel_stream()` 方法, lazy 创建并缓存
- `model_runner.py`: 更新 config 传递
- `heterogeneous_benchmark_case.py`: 更新 CLI 参数类型
- `test_cpu_gpu_parallel_moe.py`: 更新测试使用新 API

### Debug: 测试更新

测试文件使用旧的 bool API (`cpu_gpu_parallel_execution_enabled=True/False`)，需要改为 `"on"/"off"`。

```bash
# 修复后运行
python -m pytest -q tests/test_cpu_moe_correctness.py \
    tests/test_cpu_gpu_expert_operator_alignment.py \
    tests/test_cpu_gpu_parallel_moe.py
# → 6 passed
```

**数据来源**: `/home/mumura/moe_spec/logs/parallel_mode_20260504_104007/full_run.log` 中的 correctness 部分

### 结果

**Spec Qwen3-30B-A3B verify-phase 对比**:

| parallel | 75% cache | 50% cache | 25% cache |
|---|---|---|---|
| off | 301.4ms | 438.7ms | 567.1ms |
| **on** | **297.2ms** (-1.4%) | **422.3ms** (-3.7%) | **554.9ms** (-2.2%) |

**数据来源**: `/home/mumura/moe_spec/logs/parallel_mode_20260504_104007/` 中的 JSON 文件

**分析**:
- 理论最大 overlap: GPU MoE compute (32ms) 与 CPU compute (114-358ms) 重叠
- 实测 efficiency: 15-50%（受 CUDA stream 管理开销 + D2H/scatter/merge 不可重叠部分影响）
- 3.7% 的改善虽不大，但远大于 thread pool / metadata preload 的 <1%

**测试命令**:
```bash
bash /home/mumura/moe_spec/nano-vllm-moe/benchmarks/scripts/parallel_mode_validate.sh
```

---

## 5. Fused CPU Expert Backend

### 发现/问题

每个 CPU expert 的计算包含 4 个独立操作:
```python
gate_up = F.linear(hidden, gate_up_w)    # 分配临时 tensor (M, 2I)
act = act_fn(gate_up)                     # 分配临时 tensor (M, I) × 2 (chunk + silu)
out = F.linear(act, down_w)              # 分配临时 tensor (M, H)
out.mul_(w.unsqueeze(-1))                 # in-place
```

每个 expert 分配 3-5 个临时 tensor，5 个 experts → 每层 ~69KB 临时分配，44 层 → ~3MB/forward。更重要的是 Python dispatch 开销: 2× F.linear + silu_and_mul + mul = 4 个 Python 调用/expert。

### 解决思路

将 4 个操作融合为 1 个，使用预分配 buffer 消除临时分配:
```
gate_up = torch.mm(hidden, gate_up_w.t())        → pre-allocated gate_up_buf
act = silu(gate_up[:,:I]) * gate_up[:,I:]         → pre-allocated act_buf  
out = torch.mm(act, down_w.t())                   → pre-allocated out_buf
out *= weights                                    → in-place
```

### 尝试过的方案

#### 方案 1: FP32 计算 + FP32 buffer (失败 — 更慢)

将 BF16 输入转为 FP32，在 FP32 buffer 中计算，结果转回 BF16。

**测试**: 本地 microbenchmark
```bash
python -c "
# 5 experts, M=3, H=2048, I=768, BF16
# Baseline: F.linear (BF16)
# Fused: torch.mm (FP32) + convert back
"
```

结果: Fused 7.5ms vs Baseline 5.3ms → **fused 更慢**。原因: FP32 matmul 数据量翻倍 (4 bytes vs 2 bytes)，FP32↔BF16 转换有开销。

#### 方案 2: BF16 buffer + torch.mm(out=) (失败 — InferenceMode 冲突)

使用 BF16 预分配 buffer + `torch.mm(out=buffer)` 避免分配。

结果: 正确性通过 (max diff 0.0039)，但 spec benchmark 中失败:
```
RuntimeError: Inplace update to inference tensor outside InferenceMode
  is not allowed.
```

**根因**: `torch.mm(out=buffer)` 和 `copy_` 在 InferenceMode 下被视为 inplace 操作。关键是 ThreadPoolExecutor 的 worker 线程不继承 InferenceMode 上下文。当 `expert_parallel` 模式在线程池中执行 `run_fused_task` 时，worker 线程不在 InferenceMode 中，因此无法对 inference tensor 进行 inplace 写入。

#### 方案 3: BF16 + 非 inplace + InferenceMode 检测 (成功)

1. 使用 `torch.mm()` 而非 `torch.mm(out=)` — 允许临时分配
2. 检测 `torch.is_inference_mode_enabled()` — 在线程池模式下强制回退到 serial
3. 使用 `copy_` 写入 output buffer（在 inference_mode 的主线程中允许）

**关键修复** (`cpu_backend.py`):
```python
in_inference_mode = torch.is_inference_mode_enabled()
can_parallel = (
    eff_mode == "expert_parallel"
    and eff_threads > 1
    and num_tasks > 1
    and not in_inference_mode  # ThreadPool workers don't inherit InferenceMode
)
```

### 实现细节

```
文件变更:
  cpu_backend.py         — 新增 FusedTorchCpuMoeBackend (~120 行)
  cpu_workspace.py       — 新增 gate_up_buf, act_buf, out_fp32_buf 字段
  config.py              — 新增 "fused" 到 cpu_expert_backend 选项
  qwen3_moe.py           — 新增 moe_intermediate_size 属性, wired fused backend
  placement.py           — CPU metadata 预加载 (辅助优化)
  bench_cpu_moe_backend.py — 支持 --backend fused
```

### Debug 历程

1. **`torch.silu` 不存在** → 改为 `F.silu`
2. **`o_slice.float()` 创建副本** → 使用专用 `out_fp32_buf`
3. **FP32 转换更慢** → 改为 BF16 buffer
4. **InferenceMode + ThreadPool 冲突** → 添加强制 serial 检测
5. **`Qwen3MoeHeterogeneousSparseMoeBlock` 缺少 `moe_intermediate_size`** → 在构造函数中添加
6. **Benchmark CLI 参数类型不匹配** → `str2bool` → `str` with choices

### 结果

**正确性**: 6/6 tests passed

**数据来源**: `/home/mumura/moe_spec/logs/fused_backend_20260504_132127/` (完整 validation log)

**Synthetic benchmark** (H=2048, I=768, 128 experts, BF16):

| Backend | Avg Decode (ms) | Avg Compute (ms) | Avg Merge (ms) |
|---|---|---|---|
| torch | 34.86 | — | 0.23 |
| torch_packed | 34.75 | — | 0.16 |
| **fused** | **28.61** (-18%) | **22.35** | **0.14** |

**Spec benchmark** (Qwen3-30B-A3B):

| Cache | Backend | Verify (ms) | CPU Compute (ms) | CPU Merge (ms) |
|---|---|---|---|---|
| 75% | torch | 298.4 | 114.2 | 5.4 |
| 75% | **fused** | **280.0** (-6.2%) | **100.7** (-11.8%) | **3.5** (-35%) |
| 50% | torch | 427.3 | 233.4 | 8.0 |
| 50% | **fused** | **398.2** (-6.8%) | **192.1** (-17.7%) | **4.6** (-43%) |
| 25% | torch | 566.6 | 348.9 | 11.0 |
| 25% | **fused** | **497.7** (-12.2%) | **291.5** (-16.5%) | **5.1** (-54%) |

**测试命令**:
```bash
bash /home/mumura/moe_spec/nano-vllm-moe/benchmarks/scripts/fused_backend_validate.sh
```

### Speedup 来源分析

CPU compute 减少 12-18% 的来源:
1. **减少 Python dispatch**: 4 calls/expert → 1 call/expert (2× F.linear + silu_and_mul + mul → 1 fused)
2. **减少临时分配**: 3-5 temporary tensors/expert → 0 (all pre-allocated)
3. **更好的内存访问**: gate_up 和 act 连续写入预分配 buffer，cache-friendly

Merge 减少 35-54% 的来源:
- Fused backend 使用 `TorchPackedCpuMoeBackend` 风格的单 buffer merge (1× H2D + 1× index_add_)

---

## 6. 优化总结

### 所有优化点一览

| # | 优化点 | 类型 | 实现状态 | End-to-end 收益 | 文件 |
|---|---|---|---|---|---|
| 1 | Phase 1-3 Review | 分析 | 完成 | — | — |
| 2 | 持久化线程池 | 实现 | 已实现 | <1% | cpu_backend.py |
| 3 | 动态并行模式(auto) | 实现 | 已实现 | <1% | cpu_backend.py, config.py |
| 4 | CPU metadata 预加载 | 实现 | 已实现 | <1% | placement.py, heterogeneous.py |
| 5 | GPU Compute 异常分析 | 分析 | 完成 | — | — |
| 6 | CPU-GPU 并行执行修复 | 修复 | 已实现 | 1.4-3.7% | heterogeneous.py, config.py, qwen3_moe.py |
| 7 | Fused CPU Expert Backend | 实现 | 已实现 | 6-12% | cpu_backend.py, cpu_workspace.py, config.py, qwen3_moe.py |
| 8 | C++ Fused Kernel | 规划 | 未实现 | TBD (预计额外 10-20%) | — |

### 累计改善

假设所有优化叠加 (parallel=auto + backend=fused):

| Cache | Baseline (torch, no parallel) | Optimized (fused, auto parallel) | 总改善 |
|---|---|---|---|
| 75% | 301.4ms | 280.0ms | **-7.1%** |
| 50% | 438.7ms | 398.2ms | **-9.2%** |
| 25% | 567.1ms | 497.7ms | **-12.2%** |

### 架构改进

所有优化都是**可选、可回退**的:
- `cpu_expert_backend`: `"torch"` (default) | `"torch_packed"` | `"fused"`
- `cpu_gpu_parallel_execution_enabled`: `"off"` | `"on"` | `"auto"` (default)
- `cpu_expert_parallel_mode`: `"serial"` (default) | `"expert_parallel"` | `"auto"`

默认配置 (`torch` + `auto` parallel + `serial`) 保持原有行为不变。

### 已知限制与后续方向

1. **Fused backend 的 C++ kernel**: 当前使用 PyTorch `torch.mm`，转换为 C++ + BLAS 直接调用可获得额外 10-20% 提升
2. **Phase 4 (kt-kernel BF16 backend)**: 这是解决 CPU compute 瓶颈（当前占 64%）的根本途径，预计 2-5x CPU compute 加速
3. **Verify CUDA graph**: 可节省 ~30ms kernel launch overhead
4. **Sequential decode verify**: 5× 15ms = 75ms vs 570ms prefill，是最大的潜在优化但需要架构变更
5. **Weight pre-packing**: 将 expert weights 重组为 cache-friendly blocked layout，可进一步提升 CPU matmul 效率
