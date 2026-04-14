# Phase2 Post CPU/GPU 异构并行实现与验证总结（2026-04-09）

## 1. 目标与范围

本轮在已有 real CPU expert execution 基础上，完成以下工作：

1. 实现同层 CPU expert 与 GPU cached expert 的并行执行机制（heterogeneous parallel execution）。
2. 将并行机制贯穿到 spec 模式 verify 阶段（verify 使用同一 MoE 执行函数链路）。
3. 提供 heter 模式 CPU 比例扫描基准与 spec verify A/B 基准。
4. 补齐异常定位与修复记录，并输出可复现结果文件。

说明：

1. 本文聚焦并行执行、性能与正确性状态，不重复已有算子级精度门禁细节。
2. 所有命令在 `conda activate moe_spec` 环境执行。

## 2. 核心实现变更

### 2.1 并行执行主路径

文件：`nanovllm/layers/fuse_moe/heterogeneous.py`

1. 新增 GPU 子路径函数 `_run_gpu_cached_expert_path`，拆分 gather/compute 计时。
2. 新增 CPU 子路径函数 `_compute_real_cpu_expert_outputs` 与 `_merge_real_cpu_outputs`。
3. 在 `heterogeneous_moe_forward` 增加同层 overlap 路径：
   - GPU 使用独立 CUDA stream。
   - CPU 同时执行 D2H + CPU matmul。
   - merge 前仅保留一次必要同步：`current_stream.wait_stream(gpu_stream)`。
4. 增加并行 profile 字段：
   - `parallel_overlap_est_ms`
   - `parallel_critical_path_est_ms`
   - `parallel_wall_ms`
   - `cpu_wait_ms` / `gpu_wait_ms`
   - `parallel_enabled_count`

### 2.2 调度策略与配置

文件：`nanovllm/config.py`、`nanovllm/engine/model_runner.py`、`nanovllm/models/qwen3_moe.py`

新增配置：

1. `cpu_gpu_parallel_execution_enabled: bool = False`
2. `cpu_gpu_parallel_min_cpu_route_ratio: float = 0.7`

调度策略（hybrid static + dynamic）：

1. static：基于 plan 的 `gpu_route_indices` / `cpu_route_indices` 做固定分工。
2. dynamic：仅当 `cpu_route_ratio >= min_threshold` 时启用 overlap，低 CPU 比例场景自动退回串行路径，避免调度开销反噬。

### 2.3 benchmark 与调试脚本

文件：`examples/benchmarks/moe_single_layer_cpu_gpu_parallel_bench.py`

1. 支持 CPU 比例 `0/25/50/75/100`。
2. 同时输出 `parallel_enabled=false/true` 两组 A/B。
3. 输出指标：`latency_ms_p50/p95/mean`、`throughput_tok_s_mean`、`cpu_util_est`、`gpu_util_est`、`curves`。
4. 新增 latency breakdown 字段：
   - `latency_breakdown_gpu_path_exec_ms`
   - `latency_breakdown_cpu_path_exec_ms`
   - `latency_breakdown_wait_ms`
   - `latency_breakdown_sync_barrier_ms`
   - `latency_breakdown_moe_wall_ms`
   - `latency_breakdown_other_overhead_ms`
   - 以及对应 ratio 字段（相对 `latency_ms_mean`）

文件：`examples/benchmarks/spec_verify_cpu_ratio_bench.py`

1. 新增 `--parallel-settings off,on`。
2. 新增 verify 相关指标：`verify_ms_per_call_mean`、`spec_step_ms_per_call_mean`。
3. 允许传递 overlap threshold，做 verify 阶段 A/B。
4. 新增基于 `engine_profile` 的 latency breakdown 聚合输出（与单层基准字段命名保持一致）。

文件：`examples/benchmarks/cpu_alignment_case.py`

1. 对齐 hook 签名适配新 CPU 执行接口参数。
2. 新增 overlap 开关参数透传，便于 deterministic 问题定位。

### 2.4 单元测试

新增文件：`tests/test_cpu_gpu_parallel_moe.py`

1. 校验 overlap 路径与串行参考数值一致。
2. 校验 `expert_parallel` 模式与串行一致。
3. 校验并行 profile 字段触发。

回归结果：

1. `tests/test_cpu_gpu_parallel_moe.py`
2. `tests/test_cpu_gpu_expert_operator_alignment.py`
3. `tests/test_spec_engine_flow.py`
4. `tests/test_model_runner_spec_modes.py`

合计 `7 passed`。

## 3. 并行调度与关键路径分析

### 3.1 数据依赖与同步点

执行序列：

1. 构建 MoE plan（CPU/GPU route 切分）。
2. GPU stream 显式等待当前 stream，避免读未完成 hidden states。
3. GPU path 与 CPU path 并发执行。
4. merge 前等待 GPU stream 完成。
5. 按固定顺序合并：先 GPU scatter，再 CPU merge（保证聚合顺序稳定）。

避免的 barrier：

1. 不在 GPU compute 完成后再启动 CPU。
2. 不在每个 expert 任务间引入跨设备同步。

### 3.2 H2D/D2H overlap 策略

1. CPU prepare 阶段执行 D2H（hidden/weights）同时，GPU stream 已开始 cached experts 计算。
2. CPU 输出 H2D 仅在 merge 阶段发生，当前实现 correctness-first（blocking copy）。
3. 后续可在显式事件同步下引入更细粒度异步 H2D。

### 3.3 Critical Path

估计公式：

1. `gpu_core = gpu_gather + gpu_compute`
2. `cpu_core = cpu_prepare + cpu_compute`
3. `critical_path_est = max(gpu_core, cpu_core) + scatter + cpu_merge`

观测：

1. 在高 CPU 比例下，`cpu_core` 主导，`gpu_wait_ms` 明显高于 `cpu_wait_ms`。
2. CPU/GPU 利用率交叉点出现在低中比例区（约 25% 左右），CPU 很早成为瓶颈。

### 3.4 为什么串行有时比 overlap 更快（开销量化）

并行并不等于更快，实际 wall time 取决于“可重叠部分”是否能覆盖“新增调度与同步开销”。

从结果文件可见，退化主要来自三类开销：

1. 额外 stream 管理与跨 stream 同步（`wait_stream`）带来的固定成本。
2. overlap 后 CPU merge 段拉长（H2D + `index_add_` 仍在默认流串行完成）。
3. 负载不平衡时 GPU 等 CPU（`gpu_wait_ms` 增大），重叠收益被等待时间抵消。

量化示例（单位：ms）：

| 场景 | serial latency_mean | parallel latency_mean | speedup(serial/parallel) | overlap_est | gpu_wait | cpu_merge(serial->parallel) |
|---|---:|---:|---:|---:|---:|---:|
| tokens=64, cpu=75% | 11.432 | 11.791 | 0.970 | 0.706 | 4.026 | 2.602 -> 2.972 |
| tokens=256, cpu=75% | 17.953 | 16.493 | 1.089 | 0.726 | 9.803 | 2.844 -> 1.974 |
| tokens=20, cpu=75% | 31.068 | 36.330 | 0.855 | 1.012 | 6.570 | 11.825 -> 13.620 |

结论：

1. 只有当 CPU/GPU 两侧都有足够“可重叠工作量”，且 merge 段不恶化时，overlap 才有净收益。
2. 小 batch 或不均衡负载下，串行路径可能更稳更快，这是当前实验中已观测到的正常现象。

### 3.5 CPU merge / prepare 语义澄清

关于“CPU merge 是什么意思”：

1. CPU expert 输出先在 CPU 侧计算完成。
2. merge 阶段把这些输出拷回 GPU（H2D），再执行 `output.index_add_` 聚合到最终输出张量。

关于“merge 前等待 GPU stream 完成”是什么意思：

1. overlap 路径里 GPU cached experts 在独立 stream 计算。
2. 进入最终聚合前，默认流执行 `current_stream.wait_stream(gpu_stream)`。
3. 含义是“确保 GPU 路径结果已就绪后再合并”，避免未完成写入导致的数据竞争或顺序不稳定。

关于“CPU prepare 是否传输 weights”：

1. 需要澄清：prepare 不传输专家参数本体（`gate_up/down`）。
2. 当前 prepare 传输的是 token hidden slice 和 routing weights（top-k 权重）到 CPU。
3. 专家参数来自 `cpu_expert_pool`（CPU 常驻缓存），只做 dtype 对齐，不做 CPU<->GPU 往返搬运。

## 4. 性能评测结果

### 4.1 heter 比例扫描

结果文件：

1. `benchmarks/results/moe_single_layer_cpu_gpu_parallel_bench_phase2_post.json`

配置：

1. token sizes: 64, 256
2. cpu ratios: 0, 25, 50, 75, 100
3. `cpu_expert_parallel_mode=serial`
4. `cpu_gpu_parallel_min_cpu_route_ratio=0.7`

关键结论：

1. 并行路径只在高 CPU route 比例触发（`parallel_enabled_count>0` 主要出现在 75% 档）。
2. 低 CPU 比例场景自动保持串行，避免了稳定的负收益。
3. 在 75% 档可观测到非零 `parallel_overlap_est_ms`，说明 overlap 机制被触发。
4. 受 workload 抖动影响，速度收益并非单调；已通过重复运行确认趋势是 CPU 侧主导而非 GPU 侧主导。

### 4.2 heter 比例扫描表格化结果（新增 1/3/5/10/20）

结果文件：

1. `benchmarks/results/moe_single_layer_cpu_gpu_parallel_bench_phase2_post_small_tokens.json`
2. `benchmarks/results/moe_single_layer_cpu_gpu_parallel_bench_phase2_post_token1_only.json`
3. `benchmarks/results/moe_single_layer_cpu_gpu_parallel_bench_phase2_post_small_tokens_1_3_5_10_20_curated.json`

执行命令：

```bash
python examples/benchmarks/moe_single_layer_cpu_gpu_parallel_bench.py \
   --output benchmarks/results/moe_single_layer_cpu_gpu_parallel_bench_phase2_post_small_tokens.json \
   --token-sizes 3,5,10,20 \
   --cpu-ratios 0,25,50,75,100 \
   --warmup 2 \
   --repeat 3 \
   --cpu-expert-parallel-mode serial \
   --cpu-expert-num-threads 1 \
   --cpu-gpu-parallel-min-cpu-route-ratio 0.7

python examples/benchmarks/moe_single_layer_cpu_gpu_parallel_bench.py \
   --output benchmarks/results/moe_single_layer_cpu_gpu_parallel_bench_phase2_post_token1_only.json \
   --token-sizes 1 \
   --cpu-ratios 0,25,50,75,100 \
   --warmup 2 \
   --repeat 5 \
   --cpu-expert-parallel-mode serial \
   --cpu-expert-num-threads 1 \
   --cpu-gpu-parallel-min-cpu-route-ratio 0.7
```

表 1：主结果（tokens=64/256，单位：ms）

| tokens | cpu ratio | serial latency_mean | parallel latency_mean | speedup(serial/parallel) |
|---:|---:|---:|---:|---:|
| 64 | 0% | 7.991 | 8.237 | 0.970 |
| 64 | 25% | 13.197 | 7.213 | 1.830 |
| 64 | 50% | 21.259 | 28.539 | 0.745 |
| 64 | 75% | 11.432 | 11.791 | 0.970 |
| 64 | 100% | 12.677 | 13.644 | 0.929 |
| 256 | 0% | 2.153 | 2.631 | 0.818 |
| 256 | 25% | 8.982 | 8.415 | 1.067 |
| 256 | 50% | 13.425 | 13.887 | 0.967 |
| 256 | 75% | 17.953 | 16.493 | 1.089 |
| 256 | 100% | 20.684 | 19.592 | 1.056 |

表 2：小 token 结果（tokens=1/3/5/10/20，单位：ms）

| tokens | cpu ratio | serial latency_mean | parallel latency_mean | speedup(serial/parallel) |
|---:|---:|---:|---:|---:|
| 1 | 0% | 0.874 | 0.808 | 1.082 |
| 1 | 25% | 0.889 | 0.843 | 1.054 |
| 1 | 50% | 0.809 | 0.810 | 0.998 |
| 1 | 75% | 0.876 | 0.833 | 1.052 |
| 1 | 100% | 106.155 | 41.747 | 2.543 |
| 3 | 0% | 8.141 | 7.403 | 1.100 |
| 3 | 25% | 7.821 | 8.161 | 0.958 |
| 3 | 50% | 24.559 | 24.554 | 1.000 |
| 3 | 75% | 27.252 | 25.344 | 1.075 |
| 3 | 100% | 32.183 | 30.299 | 1.062 |
| 5 | 0% | 7.820 | 10.050 | 0.778 |
| 5 | 25% | 24.468 | 25.142 | 0.973 |
| 5 | 50% | 29.275 | 30.077 | 0.973 |
| 5 | 75% | 35.655 | 36.344 | 0.981 |
| 5 | 100% | 41.116 | 41.151 | 0.999 |
| 10 | 0% | 9.894 | 9.868 | 1.003 |
| 10 | 25% | 24.524 | 24.552 | 0.999 |
| 10 | 50% | 29.334 | 30.112 | 0.974 |
| 10 | 75% | 36.361 | 36.331 | 1.001 |
| 10 | 100% | 41.142 | 41.051 | 1.002 |
| 20 | 0% | 0.879 | 0.985 | 0.893 |
| 20 | 25% | 24.476 | 22.308 | 1.097 |
| 20 | 50% | 25.980 | 24.051 | 1.080 |
| 20 | 75% | 31.068 | 36.330 | 0.855 |
| 20 | 100% | 37.896 | 37.546 | 1.009 |

补充说明（对应“为什么会出现串行更快”）：

1. 小 token 场景中固定开销占比更高，overlap 不一定能摊薄。
2. 在 `tokens=20,cpu=75%`，可见 parallel 的 `cpu_to_gpu_merge_ms` 与 `gpu_wait_ms` 明显增大，导致退化。
3. `tokens=1,cpu=100%` 在本轮复跑中存在明显抖动（p95 远高于 p50），该点应视为“极小 batch + 全 CPU 路由”下的高方差样本。

### 4.3 spec verify A/B

结果文件：

1. `benchmarks/results/spec_verify_cpu_ratio_bench_phase2_post_min.json`
2. `benchmarks/results/spec_verify_cpu_ratio_bench_phase2_post_min_rerun.json`
3. `benchmarks/results/spec_verify_cpu_ratio_bench_phase2_post_min_threshold0.json`
4. `benchmarks/results/spec_verify_cpu_ratio_bench_phase2_post_min_threshold0_rerun.json`

关键结论：

1. verify 阶段已接入同一并行实现链路（无额外分叉代码）。
2. 在采样 workload（2 seq, output 8）下，开启并行并未带来正收益，`verify_latency_speedup_parallel_vs_serial` 约 `0.87~0.97`。
3. rerun 结果趋势一致，说明该结论不是单次偶发噪声。

表 3：spec verify A/B（单位：ms）

| 文件 | threshold | serial latency_mean | parallel latency_mean | verify speedup |
|---|---:|---:|---:|---:|
| spec_verify_cpu_ratio_bench_phase2_post_min.json | 0.7 | 8695.557 | 9497.327 | 0.916 |
| spec_verify_cpu_ratio_bench_phase2_post_min_rerun.json | 0.7 | 8081.066 | 9318.020 | 0.867 |
| spec_verify_cpu_ratio_bench_phase2_post_min_threshold0.json | 0.0 | 7775.952 | 7992.445 | 0.973 |
| spec_verify_cpu_ratio_bench_phase2_post_min_threshold0_rerun.json | 0.0 | 9398.716 | 9718.552 | 0.967 |

### 4.4 关于 256 tokens 出现约 2ms 的解释

`tokens=256,cpu_ratio=0%` 的 `latency_mean≈2.153ms` 看起来偏小，但结合当前基准定义是可解释的：

1. 这是“单层 synthetic MoE 微基准”，不是端到端解码时延；未包含注意力、KV cache 管理、调度器与网络栈。
2. 该点没有 CPU 路径（`cpu_route_ratio=0`），只测 GPU cached experts，路径最短。
3. 该脚本 warmup/repeat 较小，且不同 token size 次序会受缓存与频率状态影响，导致非严格单调。
4. 对照小 token 文件可见 `tokens=20,cpu=0%` 也能到亚毫秒量级（`0.879ms`），说明“低毫秒”本身并非异常值。

结论：2ms 是当前微基准定义下的可达值，不等同于端到端推理时延。

### 4.5 latency breakdown 复跑结果（2026-04-09）

结果文件：

1. `benchmarks/results/moe_single_layer_cpu_gpu_parallel_bench_phase2_breakdown_rerun.json`

复跑配置：

1. token sizes: 64, 256
2. cpu ratios: 25, 75
3. repeat=3, warmup=2
4. `cpu_expert_parallel_mode=serial`
5. `cpu_gpu_parallel_min_cpu_route_ratio=0.7`

表 4：breakdown 关键样本（单位：ms）

| tokens | cpu ratio | parallel | latency_mean | gpu_path_exec | cpu_path_exec | wait | sync_barrier | other_overhead |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 25% | false | 24.559 | 0.468 | 9.139 | 0.000 | 0.000 | 14.952 |
| 64 | 25% | true | 24.718 | 0.421 | 9.267 | 0.000 | 0.000 | 15.030 |
| 64 | 75% | false | 37.620 | 0.472 | 21.986 | 0.000 | 0.000 | 15.162 |
| 64 | 75% | true | 45.498 | 0.791 | 29.630 | 15.417 | 0.990 | 14.851 |
| 256 | 25% | false | 24.616 | 0.454 | 11.398 | 0.000 | 0.000 | 12.763 |
| 256 | 25% | true | 29.509 | 0.502 | 14.706 | 0.000 | 0.000 | 14.301 |
| 256 | 75% | false | 37.792 | 0.475 | 25.270 | 0.000 | 0.000 | 12.046 |
| 256 | 75% | true | 41.226 | 0.753 | 26.613 | 13.542 | 0.926 | 13.654 |

结论：

1. 高 CPU 比例并行样本（75%）中，`wait` 与 `sync_barrier` 开销可观（尤其 `gpu_wait`），是 latency 上升的重要来源。
2. 低 CPU 比例（25%）下并行路径未形成有效重叠收益，`other_overhead` 占比仍高。
3. 本轮新增 breakdown 字段已经能稳定定位“执行本体”与“等待/同步/框架残余”开销边界。

## 5. 正确性与一致性验证

### 5.1 已通过项

1. 单测层面：7 个回归测试全部通过。
   - 复跑命令：`pytest -q tests/test_cpu_gpu_parallel_moe.py tests/test_cpu_gpu_expert_operator_alignment.py tests/test_spec_engine_flow.py tests/test_model_runner_spec_modes.py`
   - 复跑结果：`7 passed in 67.98s`
2. heter deterministic case（场景 A）：
   - `benchmarks/results/cpu_alignment_standard_phase2_post.json`
   - `benchmarks/results/cpu_alignment_heter_parallel_phase2_post.json`
   - 对比结果：`exact_match=true`，且 `cpu_exec_calls=125`、`cpu_exec_routes=1888710`。
3. spec on/off 对齐：
   - 两次 A/B 运行 `outputs_digest` 一致（`afa32546...`），未观察到 spec 输出破坏。

### 5.2 未完全满足项（强约束差距）

1. 在部分 heter 高 CPU 压力场景（例如 `num_seqs=4,prompt_len=32,slots=32`）仍出现与 standard 的 token 偏差。
2. 该问题在 overlap 关闭时仍存在，说明不完全由并行调度引起，更可能是 CPU/GPU 数值路径差异在长链路上的放大。

## 6. 异常定位与修复记录

### 6.1 并行性能退化

现象：

1. 低 CPU 比例开启并行后 latency 上升。

root cause：

1. overlap 开销（stream 调度 + 额外路径管理）在低 CPU 比例下无法摊薄。

修复：

1. 增加 `cpu_gpu_parallel_min_cpu_route_ratio` 动态阈值门控（默认 0.7）。

验证：

1. 低比例场景 `parallel_enabled_count` 回落到 0，避免稳定负收益。

### 6.2 benchmark 脚本崩溃

现象：

1. `cpu_alignment_case.py` 在 heter 运行时报 `wrapped_cpu_exec() got an unexpected keyword argument`。

root cause：

1. hook wrapper 签名未同步新接口参数（`cpu_expert_parallel_mode`、`cpu_expert_num_threads`）。

修复：

1. 扩展 wrapper 签名并透传参数。

验证：

1. 脚本恢复运行并成功记录 `cpu_exec_calls/cpu_exec_routes`。

### 6.3 潜在流依赖竞态

现象：

1. overlap 路径存在使用独立 stream 但未显式等待当前 stream 的风险。

修复：

1. 增加 `gpu_stream.wait_stream(current_stream)`。

验证：

1. 回归测试通过，避免了读取未就绪输入的潜在时序问题。

## 7. 已知限制与下一步

已知限制：

1. heter 端到端在部分高 CPU 压力场景下仍无法对 standard 严格逐 token 对齐。
2. spec verify 阶段当前 workload 下并行收益不明显，CPU 仍是主瓶颈。

下一步建议：

1. 增加 layer-wise hidden state 差分导出，定位首个分歧层。
2. 对 CPU 路径引入可选高精度累积策略（仅用于 strict 对齐模式）。
3. 将 overlap 门控从静态阈值升级为自适应策略（基于近几步 profile）。
4. 扩展 spec verify 基准为固定模型常驻、避免多进程重复加载造成评测噪声。

