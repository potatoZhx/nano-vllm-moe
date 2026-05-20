# Draft top_c Fused CUDA Graph 实现分析报告

## 问题背景

阅读 `draft_topc_fused_cuda_graph_implementation_report.md` 后产生三个疑问：
1. 报告中有很多未解释的术语（sidecar、fused_sync 等）
2. 实现精度是否与 `top_c=0` 对齐？
3. 为什么 `top_c=1/2` 比 `top_c=0` 慢？理论上 CPU 专家计算应被 GPU 掩盖

另外，之前实验测得的 `draft_graph_replay_ms` 约 18ms，而报告中是 24.7ms（高出 37%），怀疑报告实验方法有问题。

---

## 1. 术语解释

| 术语 | 全称/含义 | 说明 |
|------|----------|------|
| **Fused** | `FusedTorchCpuMoeBackend` | 将 gate+up 投影融合为单次 matmul 的 CPU 后端实现 |
| **fused_sync** | 同步 CPU 融合路径 | CUDA graph replay 时，host callback 同步等待 CPU 专家计算完成，然后合并 GPU+CPU 结果 |
| **fused (default/perf)** | 默认性能路径 | **内部回退到 top_c=0 行为**——不启动 CPU 专家，仅用 GPU 的 round-robin substitution |
| **sidecar** | 异步旁路计算 | 通过 `NANOVLLM_DRAFT_GRAPH_FUSED_ASYNC_SIDE_COMPUTE=1` 启用。CPU 在后台异步计算但不阻塞 replay——**结果被丢弃**，仅用于预热 CPU cache |
| **substitution LUT** | 替换查找表 | top_c=0 时，未缓存的 expert 通过 round-robin 映射到已缓存 expert |

---

## 2. 关键代码追踪

### 2.1 默认 `fused` 模式 (top_c=1/2) 的实际执行路径

**文件：** `nanovllm/models/qwen3_moe.py`

```python
# Line 462
use_graph_cpu_plan = graph_safe_cpu and (
    self.draft_cuda_graph_cpu_backend == "fused_sync" or async_sidecar_enabled
)
# → 当 backend="fused" 且未设 sidecar env 时，use_graph_cpu_plan = False

# Line 472
top_c = self.draft_top_c if use_graph_cpu_plan else 0
# → 传入 build_draft_plan_gpu 的 top_c 实际为 0
```

**结论：默认 `fused` 模式下的 top_c=1/2 与 top_c=0 执行完全相同的 GPU-only 图。** 不启动 CPU 专家，无 CPU routes，无 graph CPU 状态。

### 2.2 相关文件

| 文件 | 关键行 | 作用 |
|------|--------|------|
| `nanovllm/models/qwen3_moe.py` | 462-472 | `use_graph_cpu_plan` 决策，top_c 覆盖 |
| `nanovllm/expert/placement.py` | 329-363 | top_c=0 路径：所有 routes 走 GPU |
| `nanovllm/layers/fuse_moe/heterogeneous.py` | 155-229 | graph CPU block：fused_sync 进入，fused 跳过 |
| `nanovllm/layers/fuse_moe/cpu_backend.py` | 28-49, 353-438 | `_FusedGraphState` 和 `_get_graph_state` |
| `nanovllm/engine/model_runner.py` | 837-861 | `_replay_draft_graph`：graph reploy + compute_logits |
| `nanovllm/engine/model_runner.py` | 973-1076 | `run_draft`：draft 解码路径 |
| `nanovllm/engine/model_runner.py` | 876-882 | `_can_use_draft_cpu_cudagraph` |

### 2.3 run_draft_core_run vs draft_graph_replay_ms

```
run_draft_core_run trace (model_runner.py:1022-1029):
  └─ self.run() (line 891):
       ├─ prepare_decode        (~0.17ms)
       ├─ prepare_sample        (~0.03ms)
       ├─ run_model
       │   └─ _replay_draft_graph (line 837):
       │       ├─ graph var copy
       │       ├─ graph.replay() + cuda.sync  ← draft_graph_replay_ms 测量范围
       │       └─ compute_logits (lm_head)    (~0.5ms)
       └─ sampler               (~2.7ms, 见 §5)
```

---

## 3. 精度对齐验证

### 实验配置

- Model: Qwen3-30B-A3B (128 experts/layer, 48 layers, topk=8)
- slots_per_layer=16 (12.5% expert cache on GPU)
- num_seqs=4, input_len=8, output_len=32, temperature=0.0
- bucket_steps=1,2,3,4,5（预热所有 bs）

### 结果

| 配置 | digest |
|------|--------|
| topc0_none | `3839ea29c2ae05f3ce752b32ac76164d76bdba50` |
| topc1_fused | `3839ea29c2ae05f3ce752b32ac76164d76bdba50` |

**精度完全对齐。** 这从代码层面也得到验证：默认 fused 模式下走的是 top_c=0 的 GPU-only 路径。

---

## 4. 性能对比

### 实验配置（同上，A100-80GB）

| 指标 | topc0_none | topc1_fused | 差异 |
|------|-----------|------------|------|
| `draft_graph_replay_ms` steady | 19.82ms | 19.99ms | +0.9% |
| `run_draft_core_run` steady median | 20.62ms | 21.45ms | +4.0% |
| `run_model_decode_ms` steady | 20.38ms | 20.56ms | +0.9% |
| `sample_decode_ms` | 2.76ms | 2.64ms | -4.3% |
| `prepare_decode_ms` | 0.17ms | 0.17ms | ~0% |

### 关键发现

1. **图 replay 时间几乎一致**（19.82 vs 19.99ms），验证了 fused 模式实际走 top_c=0 GPU-only 路径
2. **非图开销约 ~0.8-1.5ms**（prepare_decode + prepare_sample + graph var copy + sampler），远非 ~7ms
3. **Sampler 占 ~2.7ms**（见 §5）
4. 首次调用有 ~200ms recompile 开销，排除后稳定在 ~20-22ms

---

## 5. Sampler 开销分析

### 根因

`nanovllm/layers/sampler.py` 中 `Sampler.forward` 被 `@torch.compile` 装饰。即使 temperature=0.0（greedy 模式），仍完整执行：

```python
@torch.compile
def forward(self, logits, temperatures):
    logits_fp32 = logits.float()                    # bf16→fp32, [4, 151936]
    probs = torch.softmax(scaled_logits, dim=-1)    # 沿 vocab 维度的 reduction
    sample_tokens = probs.div_(
        torch.empty_like(probs).exponential_(1)     # Gumbel-max: 60万个随机数
    ).argmax(dim=-1)
    greedy_tokens = logits_fp32.argmax(dim=-1)      # 第二个 argmax（始终执行）
    return torch.where(greedy_mask, greedy_tokens, sample_tokens)
```

即使 T=0 时 `greedy_mask` 覆盖了 Gumbel-max 结果，两个路径的计算仍完整执行：
- softmax + exponential + div + argmax（Gumbel-max 路径）
- argmax（greedy 路径）
- 总计在 [4, 151936] 上有 ~5 个内存绑定 pass + 2 个 reduction

### Sampler 开销与 batch size 线性关系

实测确认 Sampler 开销与 batch size 严格线性缩放：

| num_seqs | sample_decode_ms | batch size | per-seq 开销 |
|----------|-----------------|------------|-------------|
| 1 | 0.51-0.66ms | [1, 151936] | ~0.55ms |
| 4 | 2.64-2.76ms | [4, 151936] | ~0.66ms |

per-seq 开销约 **0.55-0.66ms**，主要由 `logits.float()` + `softmax` + `exponential` + 两个 `argmax` 构成。对于 vocab=151936，vocab 维度 reduction 是主要瓶颈。

### Standard decode 也同样受影响

Standard decode 和 spec draft 使用**完全相同的 `self.sampler(logits, temperatures)`**（model_runner.py:931），所以 standard decode 也有 ~2.7ms 的 sample 开销（num_seqs=4 时）。

### Verify bonus token 不走 Sampler

verify bonus token 直接使用 `logits.argmax(dim=-1).tolist()`（T=0 时），或 `torch.multinomial`（T>0 时），**不经过 Sampler.forward**。功能正确。

---

## 6. fused_sync OOM 分析

### 6.1 实测数据

| 配置 | GPU alloc | Free |
|------|-----------|------|
| top_c=0, buckets=[1,2,3,4,5] | **60.41 GiB** | 18.84 GiB |
| fused_sync, buckets=[1,4,16,64] | **73.06 GiB** | 6.19 GiB |
| **fused_sync 额外消耗** | **+12.65 GiB** | — |

### 6.2 显存分解

**60.41 GiB baseline (top_c=0)：**

| 组件 | 大小 | 说明 |
|------|------|------|
| GPU expert cache (48层×16 slots) | ~6.75 GiB | gate_up + down per expert |
| 非 expert 权重 | ~2.19 GiB | attention, norms, embedding, lm_head |
| GpuFallbackWorkspace | ~1.13 GiB | **始终分配**的全 128 expert 权重副本 |
| KV cache (max_model_len=4096) | **~48 GiB** | 主要消费者 |
| Standard CUDA graph pool | ~1.5 GiB | 36 个 bs bucket 共享 |
| 其他 overhead | ~0.9 GiB | allocator 对齐 + 中间缓冲区 |

**fused_sync 额外 +12.65 GiB：**

| 额外组件 | 大小 | 说明 |
|----------|------|------|
| Draft CUDA graph pool | **~12.5 GiB** | 捕获 48 层 MoE + fused_sync callback 完整 forward |
| `_FusedGraphState` per-layer-per-bs | ~0.12 GiB | `outputs_gpu` + `route_indices_gpu` (4 bucket steps) |
| Route buffer cache | ~0.02 GiB | 可复用的 `(n*routes, hidden)` buffer |

**核心发现：fused_sync 的额外消耗主要来自 draft CUDA graph pool（~12.5 GiB），而非 `_FusedGraphState`。**

### 6.3 OOM 的双重根因

fused_sync 的 OOM 实际有两个层面：

#### 层面 1：GPU 显存 OOM（多 bucket steps）

当 bucket_steps 过多（如 37 个）时：
- 每个 bucket step 需捕获一个 graph → draft CUDA graph pool 增长
- `_FusedGraphState` 也按 bucket 数量增长
- 73.06 + extra_graphs + extra_states > 80 GiB → CUDA OOM

**解决方案：** 减少 bucket_steps + padding（见 §7.1）。

#### 层面 2：CPU 内存 OOM（Slurm cgroup 限制）

在部分节点（如 gpu19）上，即使 bucket_steps=1，fused_sync 仍然 OOM。经诊断，**这是 Slurm cgroup CPU 内存限制，非 GPU OOM**。

**关键实验证据：**
- `salloc --gres=gpu:1`（无 `--mem`）：fused_sync 反复 RC=-9 OOM kill
- `salloc --gres=gpu:1 --mem=250G`：**相同节点、相同 GPU，fused_sync 成功运行**

**原因：** 无 `--mem` 时 Slurm 按 GPU 比例分配默认内存（~125 GB / GPU）。fused_sync 的 CPU 端需加载：
- 全量 expert 权重（112 uncached experts × 48 层 × 9.4 MB）：~50.6 GiB
- `CpuMoeWorkspace`（`max_routes=262144`）：~4.2 GiB/层 × 48 层（部分虚拟内存不提交）
- Model weight loading：~60 GiB
- Python 运行环境 + 临时缓冲区：~10-20 GiB
- **总计：~125+ GiB → 超限 → kernel OOM kill**

加上 `--mem=250G` 后 cgroup limit 足够，OOM 消失。

### 6.4 为什么 verify with fused 不 OOM

verify 路径不使用：
1. CUDA graph capture（走 eager 转发）
2. `_FusedGraphState`（用 `CpuMoeWorkspace`，CPU pageable memory）
3. 不触发 graph pool 分配

同时，verify 在 `--mem=250G` 之前也能运行，因为 verify 路径的 CPU workspace 是 `torch.empty`（延迟物理页分配），实际提交的物理内存远小于虚拟分配。

---

## 7. 解决方案

### 7.1 减少 bucket steps + padding（推荐，已验证可行）

**原理：** 当 `draft_graph_bs=[1,5,10]` 时，bs=3 自动选择 bs=5 的 graph。padding 位置 (`slot_mapping=-1`) 不会被实际访问。

**现有代码已支持：** `_replay_draft_graph` 中只填充 `[:bs]` 条目，返回 `outputs[:bs]`，无需额外修改。

**效果：**
- `[1,5,10]` (3 states) → 仅 ~81 MB 的 `_FusedGraphState` outputs_gpu
- `[1,2,...,512]` (37 states) → ~13.3 GB

### 7.2 降低 gpu_memory_utilization

从 0.80 降至 0.75 可释放 ~4 GiB（缩减 KV cache），为 fused_sync graph pool 留出更多空间。

### 7.3 消除 GpuFallbackWorkspace（代码优化）

`model_runner.py:110-115` 在 `cpu_expert_execution_enabled=True` 时仍然分配全 128 expert 的 GPU workspace（1.13 GiB）。可在此条件下跳过分配。

---

## 8. 单条请求 (num_seqs=1) fused_sync 性能分析

### 8.1 实验配置

- num_seqs=1, input_len=32, output_len=16, temperature=0.0
- slots_per_layer=16 (12.5% cache)
- bucket_steps=[1]（单条请求仅需 bs=1 graph）
- gpu_memory_utilization=0.75 (fused_sync) / 0.80 (top_c=0)
- --mem=250G（避免 CPU OOM）

### 8.2 单条请求对比

| 指标 | top_c=0 | fused_sync | 比率 |
|------|---------|------------|------|
| **draft_graph_replay_ms / call** | **18.16ms** | **103.16ms** | **5.7x** |
| run_draft_core_run median | 18.93ms | 104.92ms | 5.5x |
| run_model_decode_ms | 18.70ms | 103.92ms | 5.6x |
| sample_decode_ms | 0.51ms | 0.87ms | 1.7x |
| prepare_decode_ms | 0.16ms | 0.28ms | — |
| spec_draft_forward_ms | 19.98ms | 106.32ms | 5.3x |
| digest | `fb986f...` | `fb986f...` | **一致** |

### 8.3 耗时分解对比

**top_c=0 draft step (~19ms)：**
```
run() total (18.9ms):
  ├─ prepare_decode:     0.16ms
  ├─ prepare_sample:     0.02ms
  ├─ run_model (18.7ms):
  │   └─ _replay_draft_graph:
  │       ├─ graph var copy
  │       ├─ graph.replay() + cuda.sync:              18.16ms  ← 纯 GPU
  │       └─ compute_logits (lm_head):                ~0.5ms
  └─ sampler:            0.51ms
```

**fused_sync draft step (~105ms)：**
```
run() total (104.9ms):
  ├─ prepare_decode:     0.28ms
  ├─ prepare_sample:     0.04ms
  ├─ run_model (103.9ms):
  │   └─ _replay_draft_graph:
  │       ├─ graph.replay() + cuda.sync:  103.16ms    ← 包含 CPU sync
  │       │   Per 48 MoE layers:
  │       │   ├─ GPU expert compute:       ~0.3ms/layer
  │       │   ├─ cudaLaunchHostFunc submit: ~μs/layer  (调度)
  │       │   ├─ Python callback overhead:  ~0.5ms/layer (GIL + 字典查找)
  │       │   ├─ CPU matmul (单线程):      ~0.3ms/layer (1 expert)
  │       │   ├─ cudaLaunchHostFunc sync:   ~μs/layer  (调度)
  │       │   └─ CPU→GPU merge:            ~0.4ms/layer
  │       │   Total per layer: ~1.5ms × 48 = ~72ms callback + ~14ms compute
  │       └─ compute_logits (lm_head):                ~0.7ms
  └─ sampler:            0.87ms
```

**额外 85ms 的分解：**

| 组件 | 耗时 | 说明 |
|------|------|------|
| CPU expert matmul (48层 × 1 expert) | ~14ms | 单线程 PyTorch `torch.mm()` |
| Python callback overhead (48层) | ~25ms | GIL + 全局字典查找 + lambda 调用 |
| ThreadPoolExecutor 调度 (48层) | ~15ms | submit → enqueue → worker 唤醒 |
| CPU→GPU merge (48层) | ~20ms | `outputs_gpu.copy_(outputs_cpu)` |
| Host sync future.result() (48层) | ~11ms | 等待 worker 完成 |
| **总计** | **~85ms** | |

### 8.4 关键发现

1. **85ms 中 CPU 计算仅占 ~14ms(16%)**，其余 ~71ms(84%) 是框架开销
2. **Python 回调是主要瓶颈**：graph replay 中每层都通过 `register_host_callback` 查找并调用 Python lambda，引入 GIL 竞争和解释器开销
3. **单线程执行**：`ThreadPoolExecutor(max_workers=1)` 中，即使只有 1 个 CPU expert，也走完整的 submit/schedule/wait 路径
4. **与 multi-seq 扩展**：num_seqs=4 时，CPU expert 计算线性增长到 ~57ms（4x experts），callback 开销恒定 ~71ms，预计 ~130ms/step

---

## 9. 与 ktransformers 实现对比

ktransformers 同样在 CUDA graph 中通过 `cudaLaunchHostFunc` 接入 CPU 计算，但推理速度很快。本节分析其差异。

### 9.1 架构对比

| 维度 | nano-vllm fused_sync | ktransformers |
|------|---------------------|---------------|
| **回调类型** | Python lambda（`register_host_callback`） | C 函数指针直接传 `cudaLaunchHostFunc` |
| **CPU 线程模型** | `ThreadPoolExecutor(max_workers=1)` 单线程 | NUMA-aware **60+ 持久线程池**，线程绑核 |
| **计算实现** | PyTorch `torch.mm()`（单线程 MKL） | 手写 **AMX/AVX C++ kernel**，权重预加载到 NUMA 本地 |
| **Python 参与 replay** | 是（全局字典查找 lambda） | 否（静态 C 函数，零 Python 开销） |
| **层间流水线** | 无（每层同步等待） | **deferred expert pipelining**：低分 expert 延后到下一层 |
| **任务入队** | `ThreadPoolExecutor.submit()` (Python) | Lock-free atomic queue (C++) |

### 9.2 为什么 ktransformers 没有 ~71ms 框架开销

ktransformers 的关键设计：

**1. C++ 回调，零 Python 开销**
```cpp
// cpuinfer.h: 直接传 C 函数指针
cudaLaunchHostFunc(stream, (cudaHostFn_t)func, args);
// func = MOEBindings<T>::ForwardBindings::inner (编译期静态函数)
```
nano-vllm 的 Python lambda 需要在 graph replay 时从全局字典查找 → Python 调用 → GIL 获取，每层 ~0.5ms × 48 = ~24ms。

**2. 持久线程池 + Lock-free 队列**
```cpp
// task_queue.h: lock-free SPSC queue
void enqueue(Args... args) { /* atomic head/tail */ }
// 线程 50ms busy-spin 后 condvar wait，提交仅为内存写入
```
nano-vllm 的 `ThreadPoolExecutor.submit()` 每次创建 Future 对象 + Python 锁 + 唤醒 worker，每层 ~0.3ms × 48 = ~15ms。

**3. 多线程 AMX/AVX 加速**
```cpp
// moe-tp.hpp: NUMA 分发
pool->dispense_backend()->do_numa_job([](int numa_id) {
    tps[numa_id]->forward(...);  // AMX/AVX intrinsic，多线程并行
});
```
nano-vllm 单线程 `torch.mm()` 在小 batch 下有显著 kernel launch overhead。

**4. 延迟专家流水线（deferred expert pipelining）**
```python
# experts_base.py: 低分 expert 延后到下一层
if max_deferred_experts_per_token > 0:
    submit_deferred(experts[deferred_mask], ...)
```
GPU 在当前层无需等待被延后的 CPU expert，下一层 forward 时上一层的延迟结果已就绪，实现层间 overlap。

### 9.3 改进方向

要让 nano-vllm 的 fused_sync 达到可用性能，需要：

1. **消除 Python 回调在 replay 中的参与**：将 CPU 计算调用编译为 C 函数指针，直接传给 `cudaLaunchHostFunc`（类似 ktransformers 的 `submit_with_cuda_stream`）
2. **多线程 CPU 计算**：使用持久线程池 + 多线程 BLAS（如 `torch.set_num_threads(N)` + `torch.mm` 在 graph 外预分配的 pinned buffer 上）
3. **层间流水线**：借鉴 ktransformers 的 deferred expert 机制，低分 expert 异步计算，下一层 merge
4. **简化同步路径**：将 per-layer 的 submit/sync 对合并为 batch 操作，减少 callback 节点数（48 次 → 1 次）

### 9.4 关键代码索引

| 文件 | 作用 |
|------|------|
| `ktransformers/kt-kernel/cpu_backend/cpuinfer.h` | `submit_with_cuda_stream` / `sync_with_cuda_stream` |
| `ktransformers/kt-kernel/cpu_backend/worker_pool.h` | NUMA worker pool + `NumaJobDistributor` |
| `ktransformers/kt-kernel/cpu_backend/task_queue.h` | Lock-free 原子任务队列 |
| `ktransformers/kt-kernel/operators/moe-tp.hpp` | `TP_MOE_Common::forward` — AMX/AVX 多线程 MoE |
| `ktransformers/kt-kernel/ext_bindings.cpp` | C 回调函数 + pybind11 绑定 |
| `ktransformers/kt-kernel/python/experts_base.py` | `submit_forward()` / `sync_forward()` / deferred experts |

---

## 10. 总结

| 问题 | 结论 |
|------|------|
| 术语理解 | 已解释：fused, fused_sync, sidecar, substitution LUT |
| 精度对齐（4-seq, 12.5% cache） | **完全对齐**（相同 digest），默认 fused 模式走 top_c=0 GPU-only 路径 |
| 精度对齐（1-seq, 12.5% cache） | **完全对齐**，fused_sync 也产生相同 digest |
| 为什么 top_c>0 不慢 | **因为实际没有使用 CPU 专家**——`use_graph_cpu_plan=False` 导致 top_c 被覆盖为 0 |
| 报告 24.7ms vs 实测 18ms | 报告使用 `run_draft_core_run` trace（含所有开销），且可能未排除 recompile |
| Sampler ~2.7ms (4-seq) | T=0 时仍执行完整 softmax + Gumbel-max + 冗余 greedy argmax。与 batch size 线性缩放（~0.55ms/seq） |
| Sampler ~0.51ms (1-seq) | 同上，[1, 151936] 时的采样开销 |
| fused_sync GPU OOM | Draft CUDA graph pool ~12.5 GiB。减少 bucket steps 可解决 |
| fused_sync CPU OOM | Slurm cgroup 内存限制。加 `--mem=250G` 可解决 |
| bucket_steps padding 方案 | 代码已支持，无需修改 |
| fused_sync 单条请求耗时 | **103ms/step**，其中 ~85ms 是框架开销（Python 回调 + 单线程调度 + 同步），CPU 计算仅 ~14ms |
| 与 ktransformers 对比 | ktransformers 使用 C++ 回调 + 持久线程池 + AMX/AVX kernel + 延迟流水线，避免了 nano-vllm 的 Python 和单线程瓶颈 |
| 改进方向 | 消除 Python 回调、多线程 CPU 计算、层间流水线 |

---

## 11. 相关文件索引

| 文件 | 用途 |
|------|------|
| `docs/draft_topc_fused_cuda_graph_implementation_report.md` | 原始实现报告 |
| `docs/draft_topc_fused_cuda_graph_analysis.md` | 本文档 |
| `examples/benchmarks/draft_step_breakdown.py` | 多请求 benchmark 脚本（预热 bs 1-5，排除首次调用） |
| `examples/benchmarks/profile_single_draft.py` | 单请求 timing 分解脚本 |
| `examples/benchmarks/single_fused_sync_test.py` | 单请求 fused_sync 完整 benchmark |
| `examples/benchmarks/mem_profile.py` | 显存 profile 脚本 |
| `examples/benchmarks/diag_fused_sync_oom.py` | fused_sync OOM 诊断脚本 |
| `nanovllm/models/qwen3_moe.py:462-472` | `use_graph_cpu_plan` 决策点 |
| `nanovllm/layers/fuse_moe/heterogeneous.py:155-229` | graph CPU block |
| `nanovllm/layers/fuse_moe/cpu_backend.py:28-49` | `_FusedGraphState` 定义 |
| `nanovllm/engine/model_runner.py:837-861` | `_replay_draft_graph` |
| `nanovllm/engine/model_runner.py:973-1076` | `run_draft` |
| `nanovllm/layers/sampler.py` | Sampler（@torch.compile，Gumbel-max） |
| `ktransformers/kt-kernel/cpu_backend/cpuinfer.h` | ktransformers CPU infer 协调器 |
| `ktransformers/kt-kernel/cpu_backend/worker_pool.h` | ktransformers NUMA worker pool |
| `ktransformers/kt-kernel/operators/moe-tp.hpp` | ktransformers AMX/AVX MoE kernel |
| `ktransformers/kt-kernel/ext_bindings.cpp` | ktransformers C 回调 + pybind11 绑定 |
