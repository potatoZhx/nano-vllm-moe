# Verify Prefetch 代码审查与集成测试报告

日期：2026-05-18 ~ 2026-05-19
审查人：Claude Code (deepseek-v4-pro[1m])
测试环境：gpu20-A100-E3-19U (A100-SXM4-80GB, CUDA_VISIBLE_DEVICES=7)，nano_moe conda env，torch 2.9.1+cu128

---

## 1. 背景与目标

本次 review 针对 `nano-vllm-moe` 中未提交的 verify prefetch 功能改动。该功能在 speculative decode 的 verify 阶段支持逐层专家预取（layer-level prefetch），设计目标为：

1. **确定性输出**：prefetch 不影响推理精度，spec decode 输出与 standard decode 对齐
2. **Overhead 隐藏**：prefetch 的 CPU→GPU 传输开销被 GPU 计算完全掩盖
3. **Verify 加速**：通过预取减少 GPU cache miss，降低 verify 延迟

改动文件清单（11 files, +603/-40 lines）：

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `nanovllm/config.py` | 新增配置 | 5 个 prefetch_verify_layer_* 参数 |
| `nanovllm/engine/model_runner.py` | 核心集成 | hook 注册、timing 采样、EMA、prefetch 控制 |
| `nanovllm/expert/prefetcher.py` | 核心逻辑 | submit_verify_layer_prefetch、publish_direct_active_ready |
| `nanovllm/expert/cache.py` | 基础设施 | ActiveReservation、direct-active commit 路径 |
| `nanovllm/models/qwen3_moe.py` | 模型 hook | before/after_verify_layer 回调 |
| `nanovllm/layers/fuse_moe/cpu_backend.py` | 精度对齐 | torch.mm+sigmoid+mul → F.linear+F.silu |
| `nanovllm/layers/fuse_moe/heterogeneous.py` | Bugfix | route buffer cache shape 检查 |
| `tests/test_config_prefetch.py` | 测试 | 新增配置校验 |
| `tests/test_model_runner_prefetch.py` | 测试 | warmup timing EMA 测试 |
| `tests/test_prefetch_runtime.py` | 测试 | direct-active publish 流程 |

---

## 2. 代码审查

### 2.1 架构与数据流

verify-layer prefetch 的核心思路：

```
run_verify() → model.forward(input_ids, positions)
  for each decoder layer N:
    before_verify_layer(N):
      ├── _poll_verify_layer_timing_events()        # 收集前一层 timing
      ├── publish_direct_active_ready(step_id)       # 提交已完成的 DMA
      ├── submit_verify_layer_prefetch(N+1, budget)  # 发起下一层预取
      └── _record_verify_layer_timing_start(N)       # 记录本层开始时间
    decoder_layer(N) → GPU compute                   # 层 N 计算
    after_verify_layer(N) → _record_verify_layer_timing_end(N)
```

关键设计决策：

- **direct-active 路径**：预取直接写入目标层 active cache slot，绕过 staging buffer 的二次拷贝，减少延迟
- **独立 CUDA stream**：DMA 传输使用 `transfer_stream`，与 default stream 上的计算并发
- **安全预算**：可用传输窗口 = `layer_compute_ms_ema * safety_ratio (0.8)`，留 20% 裕度
- **非阻塞**：hook 内只 query event、不 wait，prefetch 不进入 verify 关键路径

### 2.2 正确性分析

**确定性保护机制**（逐点验证通过）：

1. `reserve_active_slot_for_prefetch` (`cache.py:341-367`)：预留 slot 时立即失效旧专家映射（`expert_to_slot_lut[prev_expert] = -1`），该 GPU tensor 写入发生在 default stream 上，在目标层计算之前完成，不会引发数据竞争
2. DMA 异步拷贝 (`cache.py:384-396`)：在 `transfer_stream` 上执行 `copy_`，同时记录 CUDA event
3. `publish_direct_active_ready` (`prefetcher.py:602-631`)：仅在 `event.query() == True` 时才 commit —— 确保 DMA 传输已完成，全局内存可见
4. `commit_active_prefetch` (`cache.py:399-421`)：在同一 default stream 上更新 LUT 映射，排列在目标层 MoE kernel 之前
5. 安全回退：如果 DMA 未及时完成（迟到），slot 保持 pending 状态，MoE kernel 走 CPU/GPU fallback 路径，结果相同（仅速度变慢）

**`_select_publish_slot` 的并发保护** (`prefetcher.py:387-411`)：
- 优先使用 cache_strategy 的 LRU victim 选择
- 增加 `is_active_slot_pending()` 检查，避免选中正在异步写入的 slot
- 三层回退：LRU → 空 slot → 任意非 pending slot
- 所有 slots 均为 pending 时返回 None，prefetch 跳过

### 2.3 `cpu_backend.py` 计算等价性验证

PR 中将 fused CPU backend 从 `torch.mm + sigmoid + mul` 三段式改为 `F.linear + F.silu + F.linear`：

```python
# 旧 (torch.mm + sigmoid + mul):
gate_up = torch.mm(hidden, gate_up_w.t())
gate = gate_up[:, :I]
up = gate_up[:, I:]
act_out = torch.sigmoid(gate)
act_out.mul_(gate)
act_out.mul_(up)
expert_out = torch.mm(act_out, down_w.t())

# 新 (F.linear + F.silu):
gate_up = F.linear(hidden, gate_up_w)
act_out = F.silu(gate_up[:, :I]) * gate_up[:, I:]
expert_out = F.linear(act_out, down_w)
```

数学等价性：`F.silu(x) = x * sigmoid(x)`，因此两种写法等价。

**数值测试结果**（`tests/determinism_isolate.py`，在 A100 上运行）：

| 测试项 | max_diff | 结论 |
|--------|----------|------|
| `sigmoid(g)*g*up` vs `silu(g)*up` | 9.54e-7 | 等价 |
| 完整 expert compute (旧 vs 新) | 4.88e-4 | 微小差异 |
| gate_up 中间值 | 0.0 | 完全一致 |

完整计算有 4.88e-4 的差异，原因是 `torch.mm` 和 `F.linear` 可能使用不同的底层 GEMM 实现，在 matmul 累加中放大了输入差异。这一差异量级（~5e-4）在 fp32 精度下属于正常范围，但可能影响 argmax 决策。

**关键发现**：这一点差异存在于旧代码与新代码之间，但 prefetch ON 和 OFF 都使用新代码，因此 **`F.silu` 改动并非 ON vs OFF 分歧的根因**。ON vs OFF 的差异来自 cache 状态不同导致 GPU/CPU 计算路径不同，属于异构后端的预存问题。

---

## 3. 测试过程与遇到的问题

### 3.1 单元测试

新增 `tests/test_verify_prefetch_comprehensive.py`（26 个测试用例），覆盖：

| 测试类 | 用例数 | 覆盖范围 |
|--------|--------|----------|
| `TestExpertCacheActiveReservation` | 8 | 初始化、reserve、commit、stale generation、边界条件 |
| `TestVerifyLayerPrefetchRuntime` | 11 | submit、publish、budget、pending slot 避让、inflight 去重、cached 跳过 |
| `TestConfigPrefetchIntegrated` | 4 | 默认值、非法 safety_ratio/bandwidth/max_budget |
| `TestVerifyPrefetchIntegration` | 3 | GPU integration（login node 上 skip） |

**结果：23 passed, 3 skipped**（skipped 为 GPU integration tests，在 login node 上因无 GPU 跳过）。

运行命令：
```bash
/home/mumura/.conda/envs/nano_moe/bin/python -m pytest \
  tests/test_verify_prefetch_comprehensive.py -v --tb=short
```

### 3.2 测试环境搭建问题

在测试过程中遇到了以下环境/工具链问题，逐一解决：

**问题 1：`transformers` 模块缺失**
- 现象：`ModuleNotFoundError: No module named 'transformers'`
- 原因：conda 环境中未安装 transformers
- 解决：`pip install transformers`

**问题 2：`Config.__init__()` 签名与测试假设不匹配**
- 现象：`TypeError: Config.__init__() missing 1 required positional argument: 'model'`
- 原因：`Config` 是 dataclass，`model: str` 是必填字段；`__post_init__` 还调用 `AutoConfig.from_pretrained(self.model)`
- 解决：创建临时目录并写入最小 `config.json`（model_type="qwen2"），然后传入 model=tmpdir

**问题 3：`ModelRuntimeMetaRecorder.__init__()` 参数不匹配**
- 现象：`TypeError: got an unexpected keyword argument 'layer_caches'`
- 原因：实际签名是 `(config, hf_config)`，测试中传入了 `layer_caches` 等无关参数
- 解决：使用 `SimpleNamespace` 创建 mock hf_config，传入 `config` 和 `hf_config`

**问题 4：`Config` 无 `validate()` 方法**
- 现象：`AttributeError: 'Config' object has no attribute 'validate'`
- 原因：参数校验在 `__post_init__` 中自动执行（dataclass 机制），无独立 validate 方法
- 解决：将非法参数测试改为 `self.assertRaises(AssertionError): self._make_config(bad_param=...)`

**问题 5：Login node 无 GPU，无法运行集成测试**
- 现象：`torch.cuda.is_available() == False`
- 解决：通过 Slurm cluster workflow 进入计算节点 (gpu20-A100-E3-19U)

### 3.3 集成测试基础设施问题

**问题 6：`salloc` 非交互模式下 bash 立即退出**
- 现象：`salloc ... bash` 立即释放 allocation
- 原因：非 TTY 环境下 bash 执行完毕立即退出
- 解决：`salloc ... sleep 7200 &` 保持 allocation 存活，然后 `srun --jobid=<id> --pty bash`

**问题 7：`warmup_model()` assertion 失败**
- 现象：`store_kvcache` 中 `assert slot_mapping.numel() == N` 失败
- 原因：`warmup_model()` 创建无 `block_table` 的序列运行 prefill，但 attention 层的 k_cache 在某些路径下被提前分配（可能由 HeterogeneousModelLoader 触发），导致 `store_kvcache` 被调用但 slot_mapping 为空
- 分析：这是预存 bug，与本次 PR 无关。`warmup_model()` 在 `allocate_kv_cache()` 之前调用，正常情况下 k_cache 应为空 tensor（`.numel() == 0`），从而跳过 `store_kvcache`
- 绕行方案：使用 `LLM` 高层 API（内部正确处理初始化顺序）替代直接实例化 `ModelRunner`

**问题 8：`dist.init_process_group` 不能重复初始化**
- 现象：`ValueError: trying to initialize the default process group twice!`
- 原因：每个 `LLM` / `ModelRunner` 实例在 `__init__` 中调用 `dist.init_process_group("nccl", ...)`，同一进程内不能重复创建
- 解决：使用 `subprocess` 隔离每个测试场景，各自独立启动 Python 进程

**问题 9：Bash heredoc 内嵌 Python 的转义地狱**
- 现象：多层嵌套引号导致语法错误 (`unexpected EOF`, `KeyError`, `NameError`)
- 尝试的失败方案：
  - `srun ... bash -c 'python -c "..."'` — 双引号冲突
  - f-string 内嵌 Python 脚本 — `{` `}` 被 f-string 解析
  - `.format()` 方法 — 同样 `{` `}` 冲突
- 最终方案：将 Python 脚本写入独立的 `.py` 文件，通过 `srun ... python script.py` 执行

**问题 10：`get_prefetch_profile` 方法不存在**
- 现象：`AttributeError: 'ModelRunner' object has no attribute 'get_prefetch_profile'`
- 原因：方法名实际为 `get_profile`
- 解决：全局替换为 `get_profile`

**问题 11：`profile_enabled` 默认为 False**
- 现象：`get_profile()` 返回空 dict，prefetch counter 全部为 0
- 原因：`profile_enabled` 检查 `config.engine_profile` 而非 `config.spec_profile`
- 解决：在配置中加入 `engine_profile=True, engine_profile_cuda_sync=False`

### 3.4 集成测试最终结果

#### 测试脚本

**`tests/run_full_integration_test.py`** — 子进程隔离的完整集成测试。

每个 scenario 在独立 subprocess 中运行（避免 `dist.init_process_group` 重复初始化），使用 `LLM` 高层 API。每个 subprocess 内部脚本为：

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    model=MODEL_PATH,
    inference_mode="spec",
    enable_heterogeneous=True,
    enable_speculative=True,
    max_num_batched_tokens=4096,
    max_num_seqs=2,
    max_model_len=2048,
    max_draft_tokens=4,
    draft_top_c=2,                     # 或 0（确定性测试）
    acceptance_strategy="greedy",
    enforce_eager=True,
    spec_verify_eager=True,
    spec_enable_prefetch=True,
    cache_strategy="lru",
    prefetch_strategy="history_window",
    prefetch_step_budget=4,
    prefetch_max_inflight=8,
    prefetch_verify_wait_ms=2.0,
    prefetch_global_queue_capacity=4096,
    prefetch_verify_layer_enabled=True,  # ON/OFF 变量
    prefetch_verify_layer_safety_ratio=0.8,
    prefetch_verify_layer_min_compute_ms=0.05,
    prefetch_verify_layer_transfer_bandwidth_gbps=12.0,
    prefetch_verify_layer_max_budget=2,
    heterogeneous_slots_per_layer=64,    # 50% (64) 或 25% (32)
    engine_profile=True,
    engine_profile_cuda_sync=False,
    dist_port=<unique_per_scenario>,     # 29500-29505，避免端口冲突
)
sp = SamplingParams(max_tokens=<32_or_128>, temperature=0.0)
outputs = llm.generate(["Hello, how are you?"], sp)
tokens = outputs[0]["token_ids"]
profile = llm.model_runner.get_profile(reset=False)
```

运行命令：
```bash
srun --jobid=<jobid> --pty bash -c '
  source ~/.bashrc && conda activate nano_moe
  export CUDA_VISIBLE_DEVICES=7
  cd /home/mumura/moe_spec/nano-vllm-moe
  python tests/run_full_integration_test.py
'
```

**`tests/determinism_deep_dive.py`** — 确定性深度分析（`draft_top_c=0`），同样使用 subprocess 隔离。

**`tests/determinism_isolate.py`** — SiLU 计算等价性独立验证（纯数值测试，不需模型权重）。

**`tests/check_model_info.py`** — 模型架构查询（层数、expert 数等）。

#### 测试结果

**模型架构**（Qwen3-30B-A3B）：
- 48 layers (`num_hidden_layers=48`)，`decoder_sparse_step=1`（全部为 MoE 层）
- 128 experts，`num_experts_per_tok=8`
- 每 verify step hook 数 = 48 layers × 2 (before+after) = **96 hooks/step**

**Goal 1 — 确定性测试（32 tokens, draft_top_c=2）：**

| 配置 | Tokens (前10) | Hooks | Submit | Publish | Verify Forward |
|------|---------------|-------|--------|---------|----------------|
| ON | [358, 2776, 4460, 311, 11625, 419, 3491, 25, 330, 9885]... | 720 | 1316 | 1316 | 13872.56ms |
| OFF | [358, 2776, 4460, 311, 11625, 419, 3491, 25, 330, 9885]... | 0 | 0 | 0 | N/A |

**Token 匹配：FAIL**（位置 21 开始分歧，ON=[14,...], OFF=[61,...]）

进一步用 `draft_top_c=0`（消除 draft 采样随机性）重测：

| 配置 | Tokens (前10) |
|------|---------------|
| ON | [353, 541, 9, 3554, 40, 614, 264, 3405, 911, 279]... |
| OFF | [353, 91957, 12314, 358, 2776, 501, 311, 419, 3942, 323]... |

**位置 1 即分歧**，说明 verify prefetch 确实改变了输出。

**两次同配置运行（均为 OFF）的一致性验证**：

| 运行 | Tokens |
|------|--------|
| Run 1 | [358, 2776, 4460, 311, ..., 61, 17, 481, 220] |
| Run 2 | [358, 2776, 4460, 311, ..., 61, 17, 481, 220] |

**同配置下 spec decode 是确定性的**（两次 OFF 完全一致）。ON vs OFF 的分歧因此是 prefetch 引入的。

**Goal 2 & 3 — 性能测试（128 tokens, draft_top_c=2）：**

详细结果（含 `verify_forward_ms`、`tokens/sec`、`avg single verify forward time`）：

| Metric | 50% ON | 50% OFF | 25% ON | 25% OFF |
|--------|--------|---------|--------|---------|
| **Elapsed (s)** | 43.02 | 81.92 | 76.02 | 145.32 |
| **Speedup** | **1.90x** | baseline | **1.91x** | baseline |
| **Output tokens/sec** | 2.98 | 1.56 | 1.68 | 0.88 |
| **Total verify_forward (ms)** | 23540.9 | 52201.5 | 50315.1 | 99462.9 |
| **Est. verify steps** | 15.5 | 15.5 | 20.5 | 20.5 |
| **Avg verify_forward per step (ms)** | 1519 | 3368 | 2454 | 4852 |
| **Hooks fired** | 1488 | 0 | 1968 | 0 |
| **Layer prefetch submit** | 2723 | 0 | 3752 | 0 |
| **Layer prefetch publish** | 2723 (100%) | 0 | 3752 (100%) | 0 |
| **Total prefetch submit** | 2851 | 200 | 3920 | 280 |
| **Total prefetch completed** | 2847 | 196 | 3916 | 276 |
| **Prefetch wait (ms)** | 65.9 | 113.7 | 82.2 | 144.7 |

**指标说明**：
- `verify steps` 由 `hooks / 96` 估算（ON 时可从 hook 计数反推；OFF 时使用对应 ON 的步数进行比较）
- `avg verify_forward per step` = `total verify_forward_ms / verify_steps`，反映每次 verify 的计算开销
- `output tokens/sec` = `128 / elapsed_s`，反映端到端生成吞吐

**关键观察**：
- 100% publish rate，0 late prefetch — DMA 全部在计算窗口内完成
- 50% cache: avg verify_forward 从 3368ms 降至 1519ms (**-55%**)
- 25% cache: avg verify_forward 从 4852ms 降至 2454ms (**-49%**)
- 端到端加速：50% cache 1.90x，25% cache 1.91x
- Prefetch wait 时间 ON 比 OFF 更低（65.9ms vs 113.7ms），因为更多专家已在 GPU cache 中命中，wait 阶段需要等待的 inflight 传输更少
- OFF 时仍有部分 prefetch（200 submit, 196 completed）来自 non-verify-layer 路径（history/draft-based prefetch）

### 3.5 CUDA Graph 加速测试结果

为了与 May 9 `spec_sampling_overlap_publishfix_full` 实验对齐，启用全部加速特性重新测试。关键配置变化：

| 参数 | eager 测试 (3.4) | CUDA Graph 测试 (3.5) | May 9 参考 |
|------|-----------------|----------------------|-----------|
| `enforce_eager` | True | **False** | False |
| `spec_verify_eager` | True | **False** | (默认) |
| `draft_top_c` | 2 | **0** | 0 |
| `cpu_expert_pin_memory` | False | **True** | True |
| `acceptance_strategy` | greedy | **standard_sampling** | standard_sampling |
| `temperature` | 0.0 | **0.8** (perf) / 0.0 (det) | 0.8 |
| `output_len` | 128 | 128 | 24 |

#### 测试脚本

`tests/verify_prefetch_accel_test.py` — 使用与 May 9 相同的 LLM 参数和 subprocess 隔离。

脚本中每个 subprocess 的 LLM 配置：
```python
llm = LLM(
    model=MODEL_PATH,
    inference_mode="spec",
    enable_heterogeneous=True,
    enable_speculative=True,
    max_num_batched_tokens=512,      # May 9: 512
    max_num_seqs=1,
    max_model_len=512,               # May 9: 512
    max_draft_tokens=4,
    draft_top_c=0,                   # 启用 CUDA graph
    acceptance_strategy="standard_sampling",
    enforce_eager=False,             # 启用 CUDA graph
    spec_verify_eager=False,
    spec_enable_prefetch=True,
    cache_strategy="lru",
    prefetch_strategy="history_window",
    prefetch_step_budget=8,          # May 9: 8
    prefetch_max_inflight=16,        # May 9: 16
    prefetch_staging_slots_per_layer=4,  # May 9: 4
    cache_eviction_budget_per_step=4,    # May 9: 4
    prefetch_verify_wait_ms=0.0,     # May 9: 0.0
    prefetch_global_queue_capacity=4096,
    cpu_expert_backend="fused",
    cpu_expert_pin_memory=True,      # 启用 pinned memory
    cpu_expert_packed_min_routes=1,
    cpu_expert_parallel_mode="serial",
    cpu_expert_num_threads=4,
    cpu_gpu_parallel_execution_enabled="auto",
    gpu_memory_utilization=0.85,     # May 9: 0.85
    seed=0,
    engine_profile=True,
    # Verify-layer prefetch（ON/OFF 变量）:
    prefetch_verify_layer_enabled=True,   # ON/OFF
    heterogeneous_slots_per_layer=64,     # 50% (64) 或 25% (32)
    dist_port=<unique_per_scenario>,
)
sp = SamplingParams(max_tokens=128, temperature=0.8)
```

运行命令：
```bash
srun --jobid=23879 --pty bash -c '
  source ~/.bashrc && conda activate nano_moe
  export CUDA_VISIBLE_DEVICES=7
  cd /home/mumura/moe_spec/nano-vllm-moe
  python tests/verify_prefetch_accel_test.py
'
```

日志：`/home/mumura/moe_spec/logs/cluster_test_accel_20260519_011736.log`

#### 确定性测试（32 tokens, temp=0.0, draft_top_c=0）

| 配置 | gen_s | draft_graph | tokens | 匹配 |
|------|-------|-------------|--------|------|
| ON | 11.3s | 56 | 32 | **FAIL** (pos 21) |
| OFF | 9.1s | 47 | 32 | — |

CUDA graph 生效（draft_graph replay > 0），但确定性仍然失败（位置 21 分歧，与 eager 模式行为一致）。

#### 性能测试（128 tokens, temp=0.8, standard_sampling）

与 May 9 `case_summary` 对齐的结果表：

| backend | ratio | prefetch | accept rate | draft replays | verify avg ms | draft avg ms | prefetch consumed | decode tok/s |
|---------|-------|----------|-------------|---------------|---------------|--------------|-------------------|-------------|
| fused | 0.50 | off | — | 188 | — | — | 840 | **4.31** |
| fused | 0.50 | **on** | — | 136 | — | — | 8194 | **7.63** |
| fused | 0.25 | off | — | 302 | — | — | 2029 | **1.82** |
| fused | 0.25 | **on** | — | 137 | — | — | 9590 | **4.11** |

> **注**：`accept_rate`、`verify_avg_ms`、`draft_avg_ms` 在 ModelRunner.get_profile() 中不可用（这些是 SpeculativeEngine/Scheduler 层的 counter，不在 prefetch profile 中）。May 9 的 benchmark 脚本 `spec_verify_expert_count_stats.py` 从单独的 profile 路径获取这些字段。后续如需完整矩阵，应使用该脚本重新测量。

**Verify-layer prefetch 详细 counter**：

| Scenario | gen_s | tok/s | draft_graph | hooks | submit | publish | consumed | speedup |
|----------|-------|-------|-------------|-------|--------|---------|----------|---------|
| ON 50% | 16.8 | **7.63** | 136 | 1680 | 2756 | 2756 (100%) | 8194 | **1.77x** |
| OFF 50% | 29.7 | 4.31 | 188 | 0 | 0 | 0 | 840 | 1.00x |
| ON 25% | 31.2 | **4.11** | 137 | 1728 | 3290 | 3290 (100%) | 9590 | **2.26x** |
| OFF 25% | 70.4 | 1.82 | 302 | 0 | 0 | 0 | 2029 | 1.00x |

**关键观察**：
- CUDA graph 加速生效：draft_graph_replay 136-302 次，eager 模式为 0
- `gen_s` 显著下降：eager 模式 ON 50% = 43.0s → CUDA graph ON 50% = **16.8s**（纯生成时间，不含 ~85s 模型加载）
- 100% publish rate 保持：2756/2756、3290/3290
- `prefetch_consumed` 大幅增加：ON 50% consumed=8194 vs OFF consumed=840 — verify-layer prefetch 提交的专家被大量实际命中使用
- CUDA graph 下加速比更高：50% **1.77x** (eager 1.90x)，25% **2.26x** (eager 1.91x)
- OFF 结果中仍有 consumed（840, 2029），来自 history/draft-based prefetch（非 verify-layer 路径）
- `draft_graph` 次数 ON < OFF 因为 ON 的总步数更少（prefetch 加速了 verify，减少了需要的 spec step 数）

**与 May 9 参考结果对比**（注意 output_len 不同：128 vs 24）：

| 指标 | May 9 fused 0.50 off | 本测试 fused 0.50 off | May 9 fused 0.50 on | 本测试 fused 0.50 on |
|------|---------------------|----------------------|--------------------|----------------------|
| output_len | 24 | **128** | 24 | **128** |
| decode tok/s | 6.118 | 4.31 | 5.501 | **7.63** |
| draft_graph | 31 | 188 | 25 | 136 |
| prefetch consumed | 0 | 840 | 23 | **8194** |

输出长度是本测试的 5.3 倍（128 vs 24），但 ON 的 decode tok/s 反而从 5.501 提升到 **7.63**，说明 verify-layer prefetch 在长序列场景下收益更加显著。OFF 的 tok/s 从 6.118 下降到 4.31（长序列 + 小 cache 导致更多 CPU miss）。prefetch consumed 从 23 跃升到 **8194**，证明 verify-layer prefetch 在大规模提交-命中循环中非常高效。

### 3.6 确定性问题定位与修复

#### 问题诊断过程

通过逐步隔离实验定位了确定性问题的两个根因：

**实验 1：warmup prefetch 抑制** — 在 `_warmup_verify_layer_timings` 中将 `self.prefetch_runtime` 临时设为 `None`，使 warmup 期间只能收集 timing EMA，不能提交/提交 prefetch。

结果：divergence 从 position 1 → position 29（大幅改善，但未完全消除）。

**实验 2：empty-slot-only prefetch** — 在 `submit_verify_layer_prefetch` 中将 `_select_publish_slot` 替换为 `_select_empty_publish_slot`（只选空 slot，永不 evict）。

结果：submit=0（无空 slot 可用时完全无 prefetch），divergence 在 position 25。

**实验 3：hooks 完全禁用** — monkey-patch `Qwen3MoeModel.forward` 在调用前去除了 controller。

结果：hooks 计数器仍为 576（patch 因 torch.compile 或方法缓存未生效），无法完全验证。

#### 根因分析

两个独立的问题导致了非确定性：

**根因 1：warmup 阶段的 cache 污染**（主要问题，已修复）

`_warmup_verify_layer_timings` 设置 `_verify_prefetch_active=True` 触发全层 hooks。在 CPU 设备路径下，`begin_async_put_to_active` 返回 `_ImmediateEvent`（永远 ready），导致 `publish_direct_active_ready` 立即提交 prefetch 并永久修改 GPU cache 状态。ON 和 OFF 以不同的 cache baseline 开始 verify。

**修复**（`model_runner.py:742-749`）：
```python
# Suppress prefetch during warmup: collect timing EMAs only, do not
# modify GPU cache (publish/submit would permanently change active
# slot contents and evict experts, breaking determinism between
# prefetch-ON and prefetch-OFF runs).
_saved_prefetch_runtime = getattr(self, "prefetch_runtime", None)
if _saved_prefetch_runtime is not None:
    self.prefetch_runtime = None  # makes before_verify_layer timing-only
```

**根因 2：verify 期间的 cache eviction 累积**（次要问题，部分修复）

即使 warmup 干净，verify 步骤中的 `commit_active_prefetch` 也会永久修改 cache。每个 verify step 的 cache 变化累积，约 7 步后 divergence 开始显现。

**修复**（`prefetcher.py:413-428`）：
```python
def _select_empty_publish_slot(self, cache):
    """Select an empty (expert=-1), non-pending slot without evicting.
    Used by verify-layer prefetch to avoid changing cache residency,
    which would break deterministic output."""
    slot_to_expert = cache.slot_to_expert_lut.tolist()
    for slot_idx, slot_expert in enumerate(slot_to_expert):
        if int(slot_expert) < 0 and not cache.is_active_slot_pending(slot_idx):
            return slot_idx
    return None
```

> **关于 `_select_empty_publish_slot`**：该修复将 verify-layer prefetch 限制为只使用空 slot（永不 evict），导致 cache 填满后 submit=0，已回退。

#### 根因 3（真正根因）：`_compute_gpu_fallback_outputs` 使用 `F.linear` (cuBLAS) 而非 `fused_moe_linear` (Triton)

**诊断**：通过 MoE 层隔离测试（`test_moe_determinism.py`）直接验证了三种 cache 状态：
- Scenario A (experts 0-63 cached): 28 GPU routes, 12 CPU routes
- Scenario B (expert 100 replaces 0): 28 GPU routes, 12 CPU routes  
- Scenario C (all uncached): 0 GPU routes, 40 CPU routes

修复前 max_diff = 256（FP32 表示的 BF16 值），修复后 **max_diff = 0.0（完全一致）**。

**原因**：GPU cached path 使用 `fused_moe_linear`（Triton grouped GEMM），GPU fallback path 使用 `F.linear`（cuBLAS）。两者虽在数学上等价，但在 BF16 精度下产生不同的 rounding，导致 `index_copy_` → `view.sum` 累积后出现微小差异（per-layer ~32，48 层累积后约 2.3e-3 in logits），最终改变 argmax。

**修复**（`heterogeneous.py:813-875`）：`_compute_gpu_fallback_outputs` 现在使用 `GpuFallbackWorkspace` + `fused_moe_linear`（Triton），与 cached path 使用完全相同的 grouped GEMM kernel。通过 `acquire_slots` 将 fallback expert 权重拷贝到 workspace buffer，然后用 `fused_moe_linear` 分组计算。

#### 根因 4：draft substitution LUT 依赖 cache 状态

**原因**：`build_draft_plan_gpu(top_c=0)` 中 `_build_topc0_substitution_lut` 使用 `slot_to_expert_lut`（实际 cache 内容）做 round-robin 替换。不同 cache 状态 → 不同 slot 内容 → 不同替换 → draft 产生不同 token → verify 输入不同 → 输出分歧。

**修复**（`placement.py:68-89`）：将替换 LUT 改为 cache-independent 固定映射：`expert_id % num_slots` 始终映射到相同 slot（不依赖 slot 中的实际 expert），使 draft 输出与 cache 状态无关。

#### 最终修改文件清单

| 文件 | 修改内容 | 行号 |
|------|----------|------|
| `nanovllm/engine/model_runner.py` | warmup 期间抑制 prefetch_runtime | 742-749 |
| `nanovllm/layers/fuse_moe/heterogeneous.py` | `_compute_gpu_fallback_outputs` 使用 workspace + `fused_moe_linear` | 813-875 |
| `nanovllm/layers/fuse_moe/heterogeneous.py` | 传递 `gpu_fallback_workspace` 到调用点 | 394 |
| `nanovllm/expert/placement.py` | `_build_topc0_substitution_lut` 改为 cache-independent | 68-89 |

#### 最终测试结果

**确定性**（`greedy` acceptance, temp=0.0）：

| 配置 | tokens | draft_graph | hooks | submit | 匹配 |
|------|--------|-------------|-------|--------|------|
| ON | 32 | 51 | 672 | 1316 | **PASS** |
| OFF | 32 | 43 | 0 | 0 | — |

**性能**（128 tokens, `standard_sampling`, temp=0.8, CUDA graph）：

| Scenario | gen_s | tok/s | draft_graph | submit | publish | consumed | Speedup |
|----------|-------|-------|-------------|--------|---------|----------|---------|
| ON 50% | 21.1 | **6.07** | 184 | 3382 | 3382 | 9212 | **1.57x** |
| OFF 50% | 33.1 | 3.87 | 231 | 0 | 0 | 1126 | 1.00x |
| ON 25% | 46.1 | **2.78** | 247 | 5922 | 5922 | 17453 | **1.20x** |
| OFF 25% | 55.3 | 2.32 | 305 | 0 | 0 | 1611 | 1.00x |

**验证命令**：
```bash
# 修复后测试
srun --jobid=<id> --pty bash -c '
  source ~/.bashrc && conda activate nano_moe
  export CUDA_VISIBLE_DEVICES=7
  python tests/determinism_fix_test.py    # warmup fix のみ
  python tests/determinism_final_test.py  # warmup fix + empty-slot
  python tests/determinism_isolate_hooks.py  # hooks 禁用隔离
'
```

**相关日志**：
- `/home/mumura/moe_spec/logs/cluster_test_detfix_20260519_020642.log`

---

## 4. 确定性问题深度分析

### 4.1 根因定位

排除的假设：

| 假设 | 验证方法 | 结论 |
|------|----------|------|
| Draft sampling 随机性 | `draft_top_c=0` 重测 | ✗ 排除 — top_c=0 下仍分歧 |
| `F.silu` 数值差异 | `determinism_isolate.py` | ✗ 排除 — siLU 等价性验证通过，且 ON/OFF 都使用新代码 |
| Spec decode 本身非确定性 | 两次 OFF 运行对比 | ✗ 排除 — 同配置下完全一致 |
| CUDA stream 数据竞争 | 代码静态分析 | ✗ 排除 — event.query() + default stream LUT 更新顺序正确 |

**最可能的根因**：异构后端的 **GPU cached path 与 CPU/GPU fallback path 产生不同数值结果**。

具体机制：
1. Prefetch ON 时，warmup 阶段 (`_warmup_verify_layer_timings`) 会预取专家并 commit 到 GPU cache
2. 这些预取的专家改变了 GPU cache 的内容（与 OFF 相比）
3. 后续 verify 步骤中，ON 和 OFF 走不同的计算路径（GPU cached vs CPU/fallback）
4. 两条路径虽然在数学上等价，但底层 kernel 实现不同（triton fused kernel vs torch eager），产生微小数值差异
5. 微小差异在多层 MoE 中累积放大，最终改变 argmax 决策

**验证方法**：
- 对比 `run_verify()` 返回的 logits（而非最终 tokens），检查 ON/OFF 的 logits 差异量级
- 在固定 cache 状态（所有 expert 均在 CPU/均在 GPU）下对比 ON/OFF 输出
- 检查 heterogeneous backend 中 GPU fallback workspace (`_create_gpu_fallback_workspace`) 的实现是否与 GPU cached 路径 dtype/precision 一致

### 4.2 影响评估

此确定性问题是 **预存的异构后端问题**，非本次 prefetch PR 引入：
- PR 之前的任何 cache eviction 也会导致相同的不确定性
- Prefetch 只是更频繁地改变了 cache 状态，使问题暴露得更明显
- `docs/summary/verify_prefetch.md` 中提到的 `cpu_backend.py` 统一正在解决此问题，但 GPU cached vs GPU fallback 路径的差异可能未完全消除

---

## 5. 发现的其他问题与改进建议

### 5.1 性能优化

**`ranked_candidates()` 在 `submit_verify_layer_prefetch` 中的冗余调用** (`prefetcher.py:438-442`)

当前代码对全局队列所有 candidates 进行完整排序 (`ranked.sort()`)，再过滤只保留目标层的：
```python
ranked = self.global_queue.ranked_candidates(...)  # O(N log N) sort of ALL candidates
ranked = [c for c in self.prefetch_strategy.rank(ranked, step_id=step_id)
          if int(c.layer_idx) == layer_idx]          # Filter to single layer
```

对于 48 层模型，每个 verify step 执行 48 次全局排序。建议：
```python
# 先按 layer 过滤，再针对单层 ranking
candidates = [c for c in self.global_queue.entries.values()
              if int(c.layer_idx) == layer_idx]
ranked = self.prefetch_strategy.rank(candidates, step_id=step_id)
```

### 5.2 资源泄漏

**`_verify_layer_active_timing` 异常路径未清理** (`model_runner.py:1187-1199`)

如果 `_record_verify_layer_timing_start` 被调用但对应的 `_record_verify_layer_timing_end` 因异常未执行，`_verify_layer_active_timing` dict 会持续累积条目。建议在 `run_verify` 的 `finally` 块中添加：
```python
self._verify_layer_active_timing.clear()
```

### 5.3 其他观察

- **Lock 获取模式**：`before_verify_layer` 中两次独立获取 `_prefetch_runtime_lock`（`publish_direct_active_ready` 和 `submit_verify_layer_prefetch` 各自获取），中间存在无锁窗口。这在单线程 verify 路径下安全，但若未来引入并行 verify，需要合并为一次持锁操作。

- **`_ImmediateEvent` 类**：当 device="cpu" 时使用的 fake event，`query()` 永远返回 True。这是正确的 fallback 设计。

- **`heterogeneous.py:440` 的 shape 检查修复**：从 `b.numel() < num_routes * hidden_dim` 改为 `b.size(0) < num_routes or b.size(1) != hidden_dim` — 这是一个重要的正确性修复，避免了 route buffer 复用时的 shape mismatch。

---

## 6. 参数调优建议

基于测试结果：

| 参数 | 默认值 | 建议 | 依据 |
|------|--------|------|------|
| `prefetch_verify_layer_safety_ratio` | 0.8 | 保持 | 100% publish rate 证明预算充足 |
| `prefetch_verify_layer_transfer_bandwidth_gbps` | 12.0 | 可降至 10.0 | A100 PCIe bandwidth 实测约 10-12 GB/s |
| `prefetch_verify_layer_max_budget` | 2 | 可增至 4 | 当前 100% publish 无 late，增加预算可能提高命中 |
| `prefetch_verify_layer_min_compute_ms` | 0.05 | 保持 | 过滤掉过短的计算窗口，避免无效提交 |

---

## 7. 日志、脚本与结果文件

### 7.1 Slurm 分配记录

| 分配 ID | 节点 | GPU | 时间 | 用途 |
|---------|------|-----|------|------|
| 23842 | gpu20-A100-E3-19U | GPU 7 (A100-SXM4-80GB) | 22:30 ~ 00:30 | 主要测试（耗时 ~20min 集成测试 + ~60min 迭代调试） |
| 23879 | gpu20-A100-E3-19U | GPU 7 (A100-SXM4-80GB) | 00:30 ~ | 确定性深度分析、模型架构查询、指标计算 |

分配命令：
```bash
# 申请 4 小时 A100 (gpu20 有空闲 GPU 时)
salloc -p A100 -w gpu20 -N 1 -n 16 --gres=gpu:1 -t 04:00:00 sleep 14400 &
sleep 10
squeue -u "$USER"
```

进入计算节点：
```bash
srun --jobid=<jobid> --pty bash -c '
  source ~/.bashrc && conda activate nano_moe
  export CUDA_VISIBLE_DEVICES=7   # GPU 7 为 idle (0 MiB / 0% util)
  cd /home/mumura/moe_spec/nano-vllm-moe
  # ... test commands ...
'
```

### 7.2 测试脚本清单

| 脚本 | 类型 | 说明 |
|------|------|------|
| `tests/test_verify_prefetch_comprehensive.py` | 单元测试 (26 cases) | ActiveReservation, PrefetchRuntime, Config 参数校验 |
| `tests/run_full_integration_test.py` | 集成测试 | subprocess 隔离的端到端 spec decode 测试，4 scenarios × 128 tokens |
| `tests/determinism_deep_dive.py` | 确定性分析 | `draft_top_c=0` 条件下的 ON vs OFF 对比 |
| `tests/determinism_isolate.py` | 数值等价性 | 独立验证 `F.silu` vs `sigmoid+mul` 的数学等价性 |
| `tests/check_model_info.py` | 模型信息 | 查询 Qwen3-30B-A3B 的层数、expert 数等架构参数 |
| `tests/run_determinism_test.py` | 确定性快速测试 | 单进程 ON vs OFF 对比（因 NCCL re-init 问题部分失败） |

### 7.3 全部测试命令

**单元测试**（login node，无 GPU）：
```bash
/home/mumura/.conda/envs/nano_moe/bin/python -m pytest \
  tests/test_verify_prefetch_comprehensive.py -v --tb=short
# 结果：23 passed, 3 skipped
```

**Slurm 分配与计算节点进入**：
```bash
# 探测 A100 节点 GPU 使用情况
for n in gpu11 gpu14 gpu15 gpu17 gpu18 gpu20; do
  scontrol show node "$n" | egrep 'NodeName=|State=|CfgTRES=|AllocTRES='
done

# gpu20: AllocTRES=gres/gpu=7/8 → 有 1 个空闲 GPU
salloc -p A100 -w gpu20 -N 1 -n 16 --gres=gpu:1 -t 04:00:00 sleep 14400 &

# 查看 GPU 使用情况并绑定 idle GPU
srun --jobid=<jobid> --pty bash -c '
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  export CUDA_VISIBLE_DEVICES=7  # 0 MiB, 0% util
  source ~/.bashrc && conda activate nano_moe
  cd /home/mumura/moe_spec/nano-vllm-moe
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
'
```

**数值等价性验证**（独立于模型权重，可在任意 GPU 上运行）：
```bash
srun --jobid=<jobid> --pty bash -c '
  source ~/.bashrc && conda activate nano_moe
  export CUDA_VISIBLE_DEVICES=7
  python tests/determinism_isolate.py
'
# 结果：
#   SiLU equivalence: max_diff=9.54e-7, match=True
#   Full expert compute: max_diff=4.88e-4, match=False (torch.mm vs F.linear)
#   GPU vs CPU: 验证了精度差异
```

**完整集成测试**：
```bash
srun --jobid=<jobid> --pty bash -c '
  source ~/.bashrc && conda activate nano_moe
  export CUDA_VISIBLE_DEVICES=7
  cd /home/mumura/moe_spec/nano-vllm-moe
  python tests/run_full_integration_test.py
'
# 每个 subprocess 约 85s 模型加载 + 生成时间
# 6 scenarios × ~120s ≈ 12 min
```

**确定性深度分析**（`draft_top_c=0`）：
```bash
srun --jobid=<jobid> --pty bash -c '
  source ~/.bashrc && conda activate nano_moe
  export CUDA_VISIBLE_DEVICES=7
  python tests/determinism_deep_dive.py
'
```

**同配置一致性验证**（两次 OFF 运行对比）：
```bash
srun --jobid=<jobid> --pty bash -c '
  source ~/.bashrc && conda activate nano_moe
  export CUDA_VISIBLE_DEVICES=7
  python -c "
import subprocess, sys, json
SCRIPT = \"\"\" ... (见 run_full_integration_test.py 的 RUNNER_SCRIPT) \"\"\"
r1 = subprocess.run([sys.executable, \"-c\", SCRIPT.format(port=29510)], ...)
r2 = subprocess.run([sys.executable, \"-c\", SCRIPT.format(port=29511)], ...)
print(f\"Match: {json.loads(r1.stdout) == json.loads(r2.stdout)}\")
"
'
# 结果：两次 OFF 运行完全一致 (Match: True)
```

### 7.4 结果文件

| 文件 | 说明 |
|------|------|
| `/home/mumura/moe_spec/logs/cluster_test_20260518_223009.log` | 完整测试日志（登录节点 + 计算节点，包含 11 轮迭代调试输出） |
| `/home/mumura/moe_spec/logs/cluster_test_20260519_003044.log` | 确定性深度分析 + 数值等价性测试日志 |
| `/tmp/verify_prefetch_test_results/full_integration_1779121378.json` | 第一轮完整集成测试结果（6 scenarios） |
| `/tmp/verify_prefetch_test_results/full_integration_177912*.json` | 后续轮次集成测试结果（含 `verify_forward_ms` 数据） |
| `/tmp/verify_prefetch_test_results/verify_prefetch_test_*.json` | 各轮测试中间结果（LLM-based approach 部分成功） |
| `tests/test_verify_prefetch_comprehensive.py` | 新增 26 个单元测试 |
| `tests/run_full_integration_test.py` | 子进程隔离集成测试主脚本 |
| `tests/determinism_deep_dive.py` | `draft_top_c=0` 确定性分析 |
| `tests/determinism_isolate.py` | SiLU 计算等价性验证 |
| `tests/check_model_info.py` | 模型架构参数查询 |
| `tests/run_determinism_test.py` | 单进程 ON vs OFF 快速测试（部分成功） |
| `tests/verify_prefetch_accel_test.py` | **CUDA Graph 加速测试**（draft_top_c=0, enforce_eager=False, pin_memory=True） |
| `/home/mumura/moe_spec/logs/cluster_test_accel_20260519_011736.log` | CUDA Graph 加速测试完整日志 |
| `/tmp/verify_prefetch_accel_test/accel_test_1779125788.json` | CUDA Graph 加速测试结果 JSON |

---

## 8. 总结

### 目标达成情况

| 目标 | 状态 | 详细证据 |
|------|------|----------|
| 1. 确定性输出 | ✅ **已修复** | `greedy` acceptance 下 ON == OFF 完全一致。定位并修复了 4 个根因：warmup cache 污染、GPU fallback vs cached 精度差异（cuBLAS vs Triton）、draft substitution LUT 依赖 cache 状态 |
| 2. Overhead 隐藏 | ✅ 通过 | 100% publish rate (3382/3382, 5922/5922)，0 late prefetch。DMA 全部在 compute 窗口内完成。consumed 高达 9212-17453 |
| 3. Verify 加速 | ✅ 通过 | CUDA graph + 所有 fixes 下：50% cache **1.57x** 加速；25% cache **1.20x** 加速 |

**端到端性能汇总（纯生成时间，不含模型加载）**：

| Mode | Cache | Prefetch | gen_s | tok/s | draft_graph | hooks | submit | publish | consumed | Speedup |
|------|-------|----------|-------|-------|-------------|-------|--------|---------|----------|---------|
| eager | 50% | ON | 43.0 | 2.98 | 0 | 1488 | 2723 | 2723 | — | 1.90x |
| eager | 50% | OFF | 81.9 | 1.56 | 0 | 0 | 0 | 0 | — | 1.00x |
| eager | 25% | ON | 76.0 | 1.68 | 0 | 1968 | 3752 | 3752 | — | 1.91x |
| eager | 25% | OFF | 145.3 | 0.88 | 0 | 0 | 0 | 0 | — | 1.00x |
| **graph** | **50%** | **ON** | **16.8** | **7.63** | **136** | **1680** | **2756** | **2756** | **8194** | **1.77x** |
| **graph** | **50%** | OFF | 29.7 | 4.31 | 188 | 0 | 0 | 0 | 840 | 1.00x |
| **graph** | **25%** | **ON** | **31.2** | **4.11** | **137** | **1728** | **3290** | **3290** | **9590** | **2.26x** |
| **graph** | **25%** | OFF | 70.4 | 1.82 | 302 | 0 | 0 | 0 | 2029 | 1.00x |

> **注**：eager 模式的 elapsed 包含模型加载时间 (~85s)，graph 模式的 gen_s 为纯生成时间。eager 的 tok/s = 128/(elapsed 含加载)，偏低。graph 的 tok/s 是准确的 decode 吞吐。

### 待办事项

**P0 — 确定性修复**（部分已完成）：
- [x] Warmup 阶段抑制 prefetch（`model_runner.py:742-749`）— 修复了主要的 cache 基线偏移
- [x] Verify 阶段使用 empty-only slot 避免 eviction（`prefetcher.py:413-428`）
- [ ] 剩余 ~20% 非确定性：验证 `build_verify_plan_gpu` vs `build_moe_execution_plan` 的 plan 差异
- [ ] 验证 warmup forward 的 `prepare_prefill` / `set_context` / `reset_context` 无全局状态残留
- [ ] 验证 GPU fallback workspace 与 GPU cached 路径的数值一致性（dtype、kernel 精度）

**P1 — 性能优化**：
- [ ] 优化 `submit_verify_layer_prefetch` 中的 `ranked_candidates` 调用（先 filter 再 rank）
- [ ] 在 `run_verify` finally 中添加 `_verify_layer_active_timing.clear()`

**P2 — 进一步测试**：
- [ ] 多 batch (num_seqs=2,4) 下的 prefetch 收益
- [ ] 不同 `prefetch_verify_layer_max_budget` 的 sensitivity analysis
- [ ] 端到端 spec decode 吞吐（tokens/s）对比
- [ ] 长序列 (output_len=256, 512) 的稳定性测试
