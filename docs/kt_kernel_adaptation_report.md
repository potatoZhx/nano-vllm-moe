# kt-kernel CPU Backend 适配总结报告

## 1. 实验目标

将 kt-kernel (v0.6.1) 作为可选 CPU expert backend 集成到 nano-vllm-moe 的 heterogeneous MoE 推理管线中，在 A100 (Ice Lake) 和 RTX4090 (Sapphire Rapids) 集群节点上测试 spec 推理性能，对比 PyTorch MKL baseline 和 fused backend。

## 2. 实现架构

### 2.1 整体设计

采用**模块级共享单例 NativeMoEWrapper + 按层切换权重**的架构：

- 全局只创建一个 `NativeMoEWrapper`（避免 48 个 AMX C++ context 的 segfault）
- 每个 MoE 层的 `KtKernelCpuMoeBackend` 是轻量级句柄，指向共享 wrapper
- 在 `forward()` 前调用 `_ensure_layer_weights(layer_idx)` 切换当前层的权重

### 2.2 文件变更

| 文件 | 变更类型 | 说明 |
|---|---|---|
| `nanovllm/layers/fuse_moe/kt_backend.py` | **新增** | kt-kernel backend 主体 (~260行) |
| `nanovllm/config.py` | 修改 | 新增 `"kt_kernel"` 后端选项和 kt 参数 |
| `nanovllm/models/qwen3_moe.py` | 修改 | 导入 kt backend, `moe_intermediate_size` 属性 |
| `nanovllm/engine/model_runner.py` | 修改 | 传递 kt 配置参数 |
| `nanovllm/layers/fuse_moe/heterogeneous.py` | 修改 | per-token output merge, `GpuFallbackWorkspace` 类 |
| `benchmarks/scripts/spec_bench_runner.py` | 修改 | 支持 `--backends kt_kernel` |
| `examples/heterogeneous_benchmark_kt.py` | **新增** | kt_kernel spec benchmark 脚本 |

### 2.3 关键代码片段

**共享单例 wrapper 创建** (`kt_backend.py:90-140`):

```python
_shared_kt_wrapper = None      # NativeMoEWrapper (全局唯一)
_shared_wrapper_layer = -1     # 当前加载的层
_zombie_moes: list = []        # 防止 C++ destructor 崩溃

def _init_shared_wrapper(*, num_experts, ..., weight_path, ...):
    global _shared_kt_wrapper
    if _shared_kt_wrapper is not None:
        return
    import kt_kernel
    # Ice Lake 检测: 无 AMX 硬件则强制 AVX2
    import kt_kernel.utils.amx as amx_mod
    if not _HAS_AMX_HARDWARE and amx_mod._HAS_BF16_SUPPORT:
        amx_mod._HAS_BF16_SUPPORT = False
    _shared_kt_wrapper = kt_kernel.KTMoEWrapper(...)
```

**按层切换权重** (`kt_backend.py:143-150`):

```python
def _ensure_layer_weights(layer_idx: int):
    global _shared_wrapper_layer
    if _shared_wrapper_layer == layer_idx:
        return
    wrapper = _shared_kt_wrapper
    # Zombie 旧 MOE: 其 C++ destructor 在 Ice Lake 上崩溃
    if hasattr(wrapper, "moe") and wrapper.moe is not None:
        _zombie_moes.append(wrapper.moe)
        wrapper.moe = None
    wrapper.layer_idx = layer_idx
    wrapper.load_weights(_shared_physical_to_logical)
    _shared_wrapper_layer = layer_idx
```

## 3. 遇到的问题和解决方案

### 3.1 AMX_BF16_MOE 在 Ice Lake 上 Illegal Instruction

**现象**: A100 节点 (Xeon 8358P, Ice Lake) 有 AVX512_BF16 但无 AMX。kt-kernel 检测 `_HAS_BF16_SUPPORT=True` 后创建 `AMXBF16_MOE`，C++ 代码包含 AMX 指令导致 SIGILL。

**根因**: kt-kernel 0.6.1 代码中 BF16 只有两条路径：
```python
if _HAS_BF16_SUPPORT:    # AMX (需要 Sapphire Rapids+)
    self.moe = AMXBF16_MOE(config)
else:                     # AVX2 (所有 x86-64)
    self.moe = AVX2BF16_MOE(config)
```
没有独立的 AVX512_BF16 (无 AMX) 路径。

**解决**: 在 `_init_shared_wrapper` 中通过读取 `/proc/cpuinfo` 检测 AMX 硬件，若无则 monkey-patch `_HAS_BF16_SUPPORT = False` 强制 AVX2 fallback。

### 3.2 C++ MOE 对象创建第二个实例时 Segfault

**现象**: 创建第二个 `NativeMoEWrapper` 时 segfault (exit 139)。AMX_BF16_MOE 和 AVX2_BF16_MOE 都有此问题。

**根因**: kt-kernel 内部 C++ 对象 (AMX context / CPUInfer worker pool) 不支持多实例并发。每个 `NativeMoEWrapper` 创建独立 `AMXBF16_MOE` C++ 对象，第二个构造函数触发内存冲突。

**解决**: 采用**共享单例 wrapper** 架构——全局只创建一个 `NativeMoEWrapper`，按层调用 `load_weights` 切换权重。旧 C++ MOE 对象放入 `_zombie_moes` 列表防止 Python GC 触发 C++ destructor（destructor 也崩溃）。

### 3.3 `import kt_kernel` 时机导致 C++ 初始化冲突

**现象**: 在 nano-vllm import chain 中提前 `import kt_kernel.utils.amx`（用于 AMX 检测），导致后续 `import kt_kernel` 时 C++ 状态不一致，`cpp_load_weights` segfault。

**根因**: kt-kernel 的 C++ 扩展在首次 import 时初始化。分步 import (`utils.amx` 先于 `kt_kernel`) 导致部分初始化状态。

**解决**: 不在模块顶层 import kt_kernel。AMX 硬件检测改用 `/proc/cpuinfo` 读取。仅在 `_init_shared_wrapper` 内部（创建 wrapper 前一刻）才 `import kt_kernel`。

### 3.4 kt-kernel 输出格式与 heterogeneous_moe_forward 不兼容

**现象**: `_accumulate_mixed_routes_deterministic` 中 `index_copy_(): Number of indices (59) should be equal to source.size(dim) (9)`。

**根因**: kt-kernel 的 `forward()` 返回 per-token 累加输出 (shape `[num_tokens, hidden_size]`)，而 heterogeneous_moe_forward 期望 per-route 输出 (shape `[num_cpu_routes, hidden_size]`)。

**解决**: 在 `_accumulate_mixed_routes_deterministic` 中添加检测：若 `cpu_outputs.shape == output.shape` (per-token)，直接 `output.add_()`。

### 3.5 Profile 数据采集

**现象**: `get_profile()` 返回空数据。

**根因**: `LLMEngine.get_profile()` 返回的是展平 dict（`model_verify_cpu_compute_ms`），而非嵌套在 `engine_profile` 下。

**解决**: 直接读取 `profile.get("model_verify_cpu_compute_ms", 0)`。

### 3.6 `dist.init_process_group` 重复调用

**现象**: 同一进程多次创建 `LLMEngine` 时报 `ValueError: trying to initialize the default process group twice!`。

**解决**: 每个 backend 变体使用独立 Python 进程运行（通过 subprocess 或多次 srun）。

## 4. 测试实验记录

### 4.1 环境探索

| 测试脚本 | 命令 | 目的 | 结果 |
|---|---|---|---|
| `explore_kt.py` | `python3 explore_kt.py` | 探索 kt-kernel API | 发现 KTMoEWrapper, NativeMoEWrapper, AMX_BF16_MOE 等类 |
| `explore_kt2.py` | `python3 explore_kt2.py` | 探索 wrapper 方法 | 发现 forward, submit_forward, sync_forward, load_weights |
| `test_kt_loader.py` | `python3 test_kt_loader.py` | 探索 safetensors loader | 发现 BF16SafeTensorLoader, load_experts(base_key) |

### 4.2 单层功能验证

| 测试脚本 | 命令 | 节点 | 结果 |
|---|---|---|---|
| `test_kt_minimal.py` | `srun --jobid=20908 python3 test_kt_minimal.py` | gpu16 (A100) | AMX crash (SIGILL) |
| `test_kt_forward.py` | `srun --jobid=20908 python3 test_kt_forward.py` | gpu16 (A100) | Forward OK, crash at cleanup |
| `test_kt_tensors2.py` | `srun --jobid=20908 python3 test_kt_tensors2.py` | gpu18 (A100) | load_weights_from_tensors NotImplementedError |
| `test_kt_amx.py` | `srun --jobid=20911 python3 test_kt_amx.py` | gpu8 (RTX4090) | AMX works! forward OK, correct output |

### 4.3 单层性能 Microbenchmark

| 测试脚本 | 命令 | 节点 | 关键结果 |
|---|---|---|---|
| `test_kt_bench.py` | `srun --jobid=20911 python3 test_kt_bench.py` | gpu8 (RTX4090) | AMX: 19.70ms vs PyTorch: 1.99ms (5 tokens) |
| `test_kt_bench2.py` | `srun --jobid=20911 python3 test_kt_bench2.py` | gpu8 (RTX4090) | Speedup scales with tokens: 0.02x@1 → 0.60x@512 |
| `test_kt_amx_scale.py` | `srun --jobid=20911 python3 test_kt_amx_scale.py` | gpu8 (RTX4090) | AMX: 0.15x@1tok → 0.60x@256tok |
| `test_kt_final.py` | `srun --jobid=20914 python3 test_kt_final.py` | gpu8 (RTX4090) | **Proper CPU affinity**: kt=22.5ms vs torch=31.5ms @256tok (1.4x) |

**关键发现**: kt-kernel 需要 32 核 slurm 分配才能获得正确的 CPU pinning。8 核分配时 "Core X inside NUMA node 0 not found" 导致性能崩溃。

### 4.4 多层的功能验证

| 测试脚本 | 命令 | 节点 | 结果 |
|---|---|---|---|
| `test_kt_amx.py` (修改) | zombie + shared wrapper | gpu8 (RTX4090) | **3 layers OK** (exit 0) |
| 手动测试 | AVX2 + zombie + 3 layers | gpu11 (A100) | **3 layers OK** (exit 0) |
| 手动测试 | AMX + zombie + 3 layers | gpu8 (RTX4090) | **3 layers OK** (exit 0) |

### 4.5 全模型加载和 Spec 推理

| 测试脚本 | 命令 | 节点 | 关键结果 |
|---|---|---|---|
| `test_kt_switch.py` | `srun python3 test_kt_switch.py` | gpu8 (RTX4090) | **48 layers loaded OK**, generation OK |
| `test_kt_steady.py` | `srun python3 test_kt_steady.py` | gpu8 (RTX4090) | verify_cpu=280.3ms, first=1.1s, steady=0.5s |
| `run_one_kt.py` | 3-way comparison | gpu8 (RTX4090) | torch=110.4ms, kt_amx=106.7ms, kt_avx2=113.0ms |

**结果来源文件**:
- `/home/mumura/moe_spec/logs/kt_3way/spec_torch.json`
- `/home/mumura/moe_spec/logs/kt_3way/spec_kt_amx.json`
- `/home/mumura/moe_spec/logs/kt_3way/spec_kt_avx2.json`

## 5. 最终性能对比

### 5.1 Spec 推理 (Qwen3-30B-A3B, RTX4090 Sapphire Rapids, 32 slots)

| backend | cpu/call | verify/call | 1st call | 2nd call (steady) |
|---|---|---|---|---|
| **torch** (MKL AVX512_BF16) | 110.4ms | 321.1ms | 1.3s | 0.6s |
| **kt_amx** (AMX_BF16_MOE) | 106.7ms | 311.0ms | 1.4s | 0.6s |
| **kt_avx2** (AVX2_BF16_MOE forced) | 113.0ms | 323.5ms | 1.4s | 0.6s |

### 5.2 A100 节点 (Ice Lake, 无 AMX)

kt-kernel 在 Ice Lake 上不稳定：`AMXBF16_MOE` 崩溃 (SIGILL)，`AVX2BF16_MOE` 第二个实例 segfault。单层测 AVX2 比 PyTorch MKL 慢 2-50x。**不可用**。

### 5.3 与 fused backend 对比

| backend | A100 verify 加速 | RTX4090 verify 加速 | 稳定性 |
|---|---|---|---|
| **fused** | 18-25% | 18-25% | 稳定 |
| kt_kernel | 不可用 | ~3% | 需 workaround |

## 6. 结论

1. **kt-kernel 成功集成为可选 backend** (`cpu_expert_backend="kt_kernel"`)，在 Sapphire Rapids (RTX4090) 上通过 48 层全模型 spec 推理测试。

2. **spec decode 场景下 kt-kernel 与 PyTorch MKL 性能几乎相同** (差异 <5%)。CPU compute 在 small-M 时 memory-bandwidth bound (~9.4MB/expert)，算子差异被带宽限制抹平。

3. **kt-kernel 的 AMX 路径需 Sapphire Rapids+ 且 32+ 核 slurm 分配**才能获得正确的 CPU pinning。

4. **kt-kernel 在 Ice Lake (A100 节点) 上不可用**：AvX2 fallback 比 MKL 慢 2-50x 且多实例 segfault。

5. **推荐方案**: `cpu_expert_backend="fused"` + `cpu_gpu_parallel_execution_enabled="auto"` 在两种节点上均提供稳定 18-25% verify 加速。
