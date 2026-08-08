# Qwen3-30B-A3B 累积式 TPOT Breakdown：实现与 GPU Smoke 实验记录

日期：2026-07-27  
实验状态：P0-Eager、修复后的 P0+Graph、P1–P5 和 R-Eager 均已完成真实 GPU 单请求 smoke；80-request validation 尚未运行。

## 1. 实验目的

本实验用于拆解 Qwen3-30B-A3B 异构推理与投机解码系统中各项机制对 TPOT（Time Per Output Token）的影响。主累积链为：

```text
P0 Exact Hetero AR + Graph
  → P1 + Drafter
  → P2 + Rerouter
  → P3 + Segment Graph
  → P4 + Predictive Prefetch
  → P5 + Transfer-aware Early Stop
```

另外设置两个不进入累积收益计算的 eager 参考：

- `P0-Eager`：P0 的 exact heterogeneous autoregressive eager 版本，用于直接验证 P0 标准 CUDA Graph。
- `R-Eager`：与 P3 机制相同但关闭所有 graph，用于验证 speculative segment graph。

本轮工作的重点还包括：

1. 给 benchmark 增加运行模式和 graph/prefetch 参数透传。
2. 提供固定、不可任意拼装的 breakdown 编排脚本。
3. 运行真实 GPU smoke，验证每个配置实际激活了目标机制。
4. 定位并修复 P0 heterogeneous standard CUDA Graph capture 失败。
5. 验证 P0-Eager 与 P0+Graph 的 exact 输出一致性。

## 2. 机制口径与实验顺序

### 2.1 P0 没有运行时专家换入换出

P0 的 GPU experts 只在模型加载时按 `profile_weighted` 静态放置。运行时 cache miss 直接交给 KTransformers `kt_direct` CPU backend 计算：

```text
GPU-resident route → GPU grouped MoE
GPU cache miss     → KT-direct CPU MoE
```

`spec_verify_miss_policy=cpu` 不会把 miss expert 传回 GPU，因此 P0 不存在动态 prefetch、promotion 或 eviction；`cache_strategy=lru` 对 P0 的静态放置没有运行时作用。

### 2.2 Segment Graph 必须先于 Predictive Prefetch

Predictive prefetch 使用 `draft_segment_indexed`。当 `segment_size=48` 时，它可以退化为单个全模型 segment，但没有正常的层间 prefetch window，无法代表完整的 segment-aware prefetch。

因此累积顺序采用：

```text
Monolithic Graph → Segment Graph → Predictive Prefetch
```

而不是先打开 prefetch 再切 graph。

### 2.3 Predictive Prefetch 与 protected eviction 是一个机制包

当前 `PredictivePrefetchRuntime` 无条件保护本轮刚载入的 experts，代码中没有“predictive prefetch + 纯 LRU eviction”的独立开关。因此：

- P4 表示 predictive prefetch 与其内建 protected eviction 的组合。
- 本轮没有虚构独立的 “+ Protected Evictor” 柱。
- 本轮没有修改 predictive eviction 算法或 `nanovllm/config.py`。

## 3. 实验环境与公共配置

### 3.1 硬件与软件

| 项目 | 值 |
|---|---|
| 模型 | `/data1/models/Qwen3-30B-A3B` |
| GPU | NVIDIA GeForce RTX 4090，实际使用物理 GPU 2 |
| CPU affinity | `taskset --cpu-list 64-96` |
| Python | `/home/linke/miniconda3/envs/nano_moe/bin/python` |
| CPU MoE backend | `kt_direct` |
| KT kernel | `avx2_bf16` |
| KT threads | 16 |
| 数据集 | MT-Bench `/data1/datasets/mt_bench/question.jsonl` |
| Smoke 请求 | 1 条，实际 question id 81 |
| 输出长度 | 固定 512 tokens，`ignore_eos=true` |
| 随机种子 | `20260719` |
| Temperature | 0.8 |
| Acceptance | `standard_sampling` |

### 3.2 专家缓存与模型参数

| 参数 | 值 |
|---|---|
| GPU expert cache ratio | 0.3125 |
| 放置策略 | `profile_weighted` |
| 总 GPU expert slots | 1920 |
| 每层 slot bucket | 34、35、41、42、48、55、56 |
| Slot profile | `pre_exps/exp_and_figs/unique/unique_count_plot_summary_n1024.csv` |
| Cache metric | `lru` |
| Fixed speculative K | 12 |
| Segment size | monolithic 为 48；segment graph 为 12 |
| GPU memory utilization | 0.99 |

### 3.3 测量定义

主指标为 pooled TPOT：

```text
pooled TPOT (ms/token)
  = 1000 × Σ decode_sec / Σ(generated_output_tokens - 1)
```

编排脚本还会计算：

- 请求级 TPOT p50、p90。
- 以请求为 bootstrap 单元的 10,000 次 95% CI。
- CSV、JSON 和不依赖 matplotlib 的 SVG 图。

本报告当前只有每配置 1 条 smoke 请求，所以：

- smoke TPOT 只用于机制激活和初步性能 sanity check。
- 单请求 bootstrap 区间退化为单点，没有统计解释力。
- 不能用当前结果替代计划中的 80-request validation。

## 4. 每个实验配置的含义

| ID | 图中标签 | 推理模式 | Graph | 运行时传输 | Draft/Reroute/Stop |
|---|---|---|---|---|---|
| P0-Eager | P0 Eager Ref. | Exact heterogeneous AR | 全 eager | 无 | 无 drafter |
| P0 | Exact Hetero AR + Graph | Exact heterogeneous AR | Standard CUDA Graph | 无 | 无 drafter |
| P1 | + Drafter | Spec fixed K=12 | Draft/verify monolithic graph | 无 | `round_robin` substitute |
| P2 | + Rerouter | Spec fixed K=12 | Monolithic graph | 无 | `entropy_cache_bias` |
| R-Eager | Eager Ref. | 与 P3 相同 | 全 eager | 无 | `entropy_cache_bias` |
| P3 | + Segment Graph | Spec fixed K=12 | 4 × 12-layer draft/verify graph | 无 | `entropy_cache_bias` |
| P4 | + Predictive Prefetch | Spec fixed K=12 | 4 × 12-layer graph | Predictive prefetch + protected eviction | `entropy_cache_bias` |
| P5 | + Early Stop (Ours) | Spec adaptive K=6–12 | 4 × 12-layer graph | 与 P4 相同 | `transfer_aware_step` |

### 4.1 P0-Eager：exact heterogeneous eager 参考

配置重点：

```text
inference_mode=heter
enforce_eager=true
draft_cuda_graph_enabled=false
verify_cuda_graph=false
spec_enable_prefetch=false
prefetch_runtime_kind=legacy
prefetch_runtime_mode=baseline_staging
segment_size=48
```

它与 P0 使用相同静态 expert placement、router、GPU experts 和 KT-direct CPU miss backend，唯一关键差异是关闭 standard CUDA Graph。

### 4.2 P0：exact heterogeneous autoregressive + standard graph

配置重点：

```text
inference_mode=heter
enforce_eager=false
draft_cuda_graph_enabled=false
verify_cuda_graph=false
spec_enable_prefetch=false
```

这里的 graph 是普通 autoregressive decode 的 standard CUDA Graph，不是 speculative draft/verify graph。每个 decode token 应 replay 一次全模型 graph；GPU cache miss 仍由 KT-direct CPU 精确计算。

### 4.3 P1：加入原始 fixed-K drafter

配置重点：

```text
inference_mode=spec
max_draft_tokens=12
draft_stop_policy=none
draft_reroute_policy=round_robin
prefetch_runtime_kind=legacy
prefetch_runtime_mode=baseline_staging
segment_size=48
draft_cuda_graph_enabled=true
verify_cuda_graph=true
spec_enable_prefetch=false
```

`segment_size=48` 产生一个完整模型 draft graph 和一个完整模型 verify graph。该阶段只测 drafter，不发生 runtime expert transfer。

### 4.4 P2：加入 entropy/cache-aware rerouter

P2 与 P1 唯一的主要机制差异是：

```text
draft_reroute_policy=entropy_cache_bias
```

它用当前 rerouter 替换原始 `round_robin` substitute；仍使用 fixed K=12、monolithic graph，且关闭 prefetch。

### 4.5 R-Eager：P3 的 eager 参考

配置重点：

```text
inference_mode=spec
enforce_eager=true
draft_cuda_graph_enabled=false
verify_cuda_graph=false
prefetch_runtime_mode=draft_segment_indexed
segment_size=12
draft_reroute_policy=entropy_cache_bias
spec_enable_prefetch=false
```

R-Eager 不进入主累积链，只用于与 P3 对比 graph 开销。

### 4.6 P3：加入 12-layer Segment Graph

P3 与 R-Eager 使用相同的 speculative/rerouter/segment 配置，但打开 draft/verify graph：

```text
enforce_eager=false
draft_cuda_graph_enabled=true
verify_cuda_graph=true
prefetch_runtime_kind=legacy
prefetch_runtime_mode=draft_segment_indexed
segment_size=12
spec_enable_prefetch=false
```

Qwen3-30B-A3B 的 48 层被划分为 4 个 12-layer segment。prefetch 仍关闭，因此该柱只测 graph 分段。

### 4.7 P4：加入 Predictive Prefetch

P4 在 P3 基础上首次打开运行时专家传输：

```text
spec_enable_prefetch=true
prefetch_runtime_kind=predictive
prefetch_runtime_mode=draft_segment_indexed
segment_size=12
```

底层 access metric 为 LRU，但 `PredictivePrefetchRuntime` 同时保护本轮刚载入的 experts，因此 P4 是 predictive prefetch 与 protected eviction 的组合。

### 4.8 P5：加入 transfer-aware early stop

P5 是当前完整的 `k12_transfer_step` 机制：

```text
draft_stop_policy=tpot
draft_tpot_cost_model=history
draft_tpot_stop_rule=transfer_aware_step
draft_tpot_min_steps=6
draft_tpot_stop_margin=0.0
draft_tpot_lookahead_cache_credit_ms_per_step=0.0
draft_tpot_verify_model_mode=active
draft_tpot_verify_model_path=results/transfer_v3_artifact_20260719/verify_cost_v3.json
draft_tpot_uncertainty_scale=0.0
acceptance_predictor_enabled=true
```

它保留 P4 的 predictive prefetch 和 segment graph，动态选择 K=6–12。固定 512-token 输出的最后一轮可能因剩余 token budget 被截短到 K<6；这种 terminal clipping 不属于 early-stop policy 的决策。

## 5. 代码更改

### 5.1 Benchmark 参数透传

文件：[scripts/bench_eval_workload_tpot.py](../scripts/bench_eval_workload_tpot.py)

新增或透传的 CLI：

```text
--inference-mode heter|spec
--spec-enable-prefetch true|false
--enforce-eager true|false
--draft-cuda-graph-enabled true|false
--verify-cuda-graph true|false
--prefetch-runtime-kind legacy|predictive|dual_queue
--prefetch-runtime-mode baseline_staging|draft_direct_active|draft_segment_indexed
```

`create_llm()` 不再硬编码这些值，而是按 resolved CLI 创建引擎：

```python
mode = args.inference_mode

inference_mode=mode
enable_heterogeneous=mode in {"heter", "spec"}
enable_speculative=mode == "spec"
enforce_eager=args.enforce_eager
spec_enable_prefetch=args.spec_enable_prefetch
prefetch_runtime_mode=args.prefetch_runtime_mode
prefetch_runtime_kind=args.prefetch_runtime_kind
draft_cuda_graph_enabled=args.draft_cuda_graph_enabled
verify_cuda_graph=args.verify_cuda_graph
```

同时完成：

- 将 resolved runtime config 写入 `summary.json` metadata。
- 禁止 heter 模式打开 speculative prefetch、predictor 或 early stop。
- 限定 `spec_enable_prefetch=true` 只能用于 spec 模式。
- 对 `transfer_aware_step` 严格校验 predictive runtime、segment=12、verify graph 和 active v3 artifact。
- eager 配置必须显式关闭 draft/verify graph。

### 5.2 固定 breakdown 编排器

文件：[scripts/run_tpot_performance_breakdown.py](../scripts/run_tpot_performance_breakdown.py)

主要能力：

- 固定 P0-Eager、P0–P5、R-Eager manifest，避免生成不可解释的任意组合。
- 支持 `--phase smoke|validation|all`。
- 支持 `--stage`、`--resume`、`--dry-run`、`--print-commands`。
- 每个配置使用独立进程、端口、日志和结果目录。
- `all` 模式要求全部 smoke 通过后才运行 validation。
- validation 预期 80/80 请求，每条固定生成 512 tokens。
- 自动检查 graph replay、transfer、runtime class、early stop 和 resolved config。
- 自动聚合 pooled TPOT、p50/p90、request bootstrap 95% CI。
- 自动生成 JSON、CSV 和 SVG。

新增 P0-Eager 后：

- `P0-Eager` 被放在 P0 左侧，使用灰色参考柱。
- P0 标签改为 `Exact Hetero AR + Graph`。
- 图中增加 `P0 Graph vs Eager` 括号。
- P0-Eager 验收要求所有 graph replay 为 0，且 KT-direct CPU routes 非零。
- P0 验收要求 standard graph 和 exact KT hybrid graph replay 均非零。

### 5.3 P0 standard CUDA Graph 修复

涉及文件：

- [nanovllm/engine/model_runner.py](../nanovllm/engine/model_runner.py)
- [nanovllm/models/qwen3_moe.py](../nanovllm/models/qwen3_moe.py)

#### 初次失败

原 P0 在 standard graph capture 中调用普通 heterogeneous MoE plan builder：

```text
build_moe_execution_plan
  → build_prefill_plan_gpu
  → torch.nonzero(gpu_route_mask)
```

`torch.nonzero` 产生数据相关动态 shape，CUDA stream capture 不允许该操作。实际错误：

```text
torch.AcceleratorError:
CUDA error: operation not permitted when stream is capturing
```

原始失败栈见：

- [P0 初次失败日志](../results/tpot_performance_breakdown/logs/smoke/p0.log)
- 关键位置：`nanovllm/expert/placement.py:705`

#### 修复策略

修复复用了已有的 exact、graph-capturable KT-direct hybrid 语义：

1. Router 仍计算原始 top-k experts 和 weights。
2. `build_verify_graph_safe_plan_gpu()` 为所有 routes 建立固定形状 GPU layout。
3. GPU cache hit 使用真实 GPU expert slot 和原始 weight。
4. GPU cache miss 在 GPU 侧映射到合法替代 slot，但对应 weight 置零。
5. 同一批 cache misses 由 KT-direct CPU backend 精确计算。
6. GPU hit 输出与 CPU miss 输出相加，保持 exact heterogeneous 结果。

P0 standard graph capture 现在仅在以下组合下启用 hybrid 路径：

```text
inference_mode=heter
enable_heterogeneous=true
cpu_expert_execution_enabled=true
cpu_expert_backend=kt_direct
spec_verify_miss_policy=cpu
```

若 heterogeneous standard graph 使用其他未支持的组合，引擎会给出明确错误，而不是进入动态 plan 后在 capture 中失败。

#### Profile 证据

每个 MoE 层增加固定形状 device counters：

```text
_graph_kt_cpu_route_count
_graph_kt_active_route_count
```

它们在 graph 内更新，replay 后聚合到 engine profile，用于证明：

- standard graph 确实 replay。
- KT-direct hybrid graph 确实 replay。
- CPU miss routes 确实发生。

这避免在 graph capture 本体中执行 `.item()`、`torch.nonzero()` 等 host/dynamic 操作。

### 5.4 明确未修改的部分

本轮没有修改：

- `nanovllm/config.py`
- Predictive prefetch eviction selector
- `_round_loaded` protected eviction 语义
- Transfer-aware v3 artifact

如果未来要把 P4 拆成 “Prefetch + LRU” 与 “+ Protected Evictor”，需要另行增加 `predictive_eviction_protection_enabled`，不属于本轮实验。

## 6. 实验过程

### 6.1 静态检查与命令预览

执行：

```bash
/home/linke/miniconda3/envs/nano_moe/bin/python -m py_compile \
  nanovllm/models/qwen3_moe.py \
  nanovllm/engine/model_runner.py \
  scripts/bench_eval_workload_tpot.py \
  scripts/run_tpot_performance_breakdown.py

git diff --check

/home/linke/miniconda3/envs/nano_moe/bin/python \
  scripts/run_tpot_performance_breakdown.py \
  --phase smoke \
  --stage p0_eager,p0 \
  --output-dir results/tpot_performance_breakdown_p0_graph_fix \
  --dry-run \
  --print-commands
```

运行前通过 `nvidia-smi` 确认 GPU 2 空闲。

### 6.2 第一次完整 smoke

实际命令：

```bash
/home/linke/miniconda3/envs/nano_moe/bin/python \
  scripts/run_tpot_performance_breakdown.py \
  --phase smoke \
  --output-dir results/tpot_performance_breakdown
```

运行时间：

```text
2026-07-27 13:17:50 UTC
  → 2026-07-27 13:42:06 UTC
```

结果：

- P1、P2、R-Eager、P3、P4、P5 通过。
- 原 P0 在 standard CUDA Graph capture 阶段失败。
- 失败原因是 `build_prefill_plan_gpu()` 中的动态 `torch.nonzero()`。

原始状态与结果：

- [第一次 run_status.json](../results/tpot_performance_breakdown/run_status.json)
- [第一次 smoke breakdown.json](../results/tpot_performance_breakdown/smoke/breakdown.json)

### 6.3 增加 P0-Eager 并修复 P0 Graph

完成以下工作：

1. 在固定 manifest 中增加 `p0_eager`。
2. 保持原 P0 为累积链的 graph baseline。
3. standard graph capture 改用 exact KT-direct hybrid 路径。
4. 增加 hybrid route counters 和机制验收。
5. 将新实验输出写入独立目录，避免覆盖第一次完整 smoke。

### 6.4 P0-Eager 与 P0+Graph 定向真实 GPU smoke

实际命令：

```bash
/home/linke/miniconda3/envs/nano_moe/bin/python \
  scripts/run_tpot_performance_breakdown.py \
  --phase smoke \
  --stage p0_eager,p0 \
  --output-dir results/tpot_performance_breakdown_p0_graph_fix
```

运行时间：

```text
2026-07-27 13:55:21 UTC
  → 2026-07-27 14:01:14 UTC
```

两项均通过，且进程退出后 GPU 0–3 均回到约 1 MiB 空闲占用。

结果：

- [修复后 run_status.json](../results/tpot_performance_breakdown_p0_graph_fix/run_status.json)
- [修复后 breakdown.json](../results/tpot_performance_breakdown_p0_graph_fix/smoke/breakdown.json)
- [修复后 breakdown.svg](../results/tpot_performance_breakdown_p0_graph_fix/smoke/breakdown.svg)

### 6.5 输出一致性检查

P0-Eager 与 P0+Graph 使用同一请求、同一 seed 和同一 sampling 配置：

```text
P0-Eager digest:
5b3dd3fd59ed26c326df81cba8ac9e679d01f203d73697f2ada039536c6abf16

P0+Graph digest:
5b3dd3fd59ed26c326df81cba8ac9e679d01f203d73697f2ada039536c6abf16
```

两者 512 个 generated token IDs 完全相同，且 `cpu_routes_sum` 都是 86,559。这证明：

- Graph 修复没有改变 router 或 CPU miss 语义。
- Graph 与 eager 执行了相同的 exact heterogeneous 路由。

### 6.6 回归测试

执行：

```bash
/home/linke/miniconda3/envs/nano_moe/bin/python -m pytest -q \
  tests/test_verify_cuda_graph_kt_hybrid.py \
  tests/test_model_runner_spec_modes.py \
  tests/test_bench_eval_workload_tpot.py
```

结果：

```text
50 passed in 3.89s
```

最后使用 `--resume` 重新校验已有结果并生成更新后的聚合 SVG：

```bash
/home/linke/miniconda3/envs/nano_moe/bin/python \
  scripts/run_tpot_performance_breakdown.py \
  --phase smoke \
  --stage p0_eager,p0 \
  --output-dir results/tpot_performance_breakdown_p0_graph_fix \
  --resume
```

### 6.7 Verify expert 统计补采

日期：2026-07-28。

原始 R-Eager、P4、P5 profile 已保存逐层 active/CPU expert
metadata；P1、P2、P3 当时关闭了 prefetch runtime，因此 graph 内的
expert metadata recorder 没有创建，对应 profile 字段为 0。这里的 0
表示“未采集”，不能解释为“没有 CPU experts”。

为补齐 P1、P2、P3，额外运行了 profile-only metadata shadow：

```text
spec_enable_prefetch=true
prefetch_runtime_kind=legacy
NANOVLLM_PREFETCH_SKIP_OBSERVE=1
```

该组合只创建 `ModelRuntimeMetaRecorder`，但跳过 runtime observe，
不会产生 prefetch 请求或修改 cache。每个 shadow run 均要求：

```text
output digest == 原正式 smoke digest
verify call count == 原正式 smoke
verify token count == 原正式 smoke
prefetch submit/ready/publish/consume == 0
```

三个 shadow run 均通过：

| 阶段 | Digest 匹配 | Verify calls | Verify tokens | Transfer |
|---|---:|---:|---:|---:|
| P1 | Yes | 163 | 2,083 | 0 |
| P2 | Yes | 157 | 2,027 | 0 |
| P3 | Yes | 157 | 2,027 | 0 |

补采结果目录：

- [P1 metadata shadow](../results/tpot_breakdown_stats_profile/p1_metadata_shadow/summary.json)
- [P2 metadata shadow](../results/tpot_breakdown_stats_profile/p2_metadata_shadow/summary.json)
- [P3 metadata shadow](../results/tpot_breakdown_stats_profile/p3_metadata_shadow/summary.json)

shadow run 的诊断开销会影响 TPOT 和 latency，因此：

- expert 数量使用 shadow metadata。
- latency 始终使用原始正式 smoke 的 `model_verify_call_records`。
- shadow run 的 TPOT 不进入任何性能表或收益计算。

## 7. 运行命令

### 7.1 推荐入口

运行全部配置的 smoke：

```bash
/home/linke/miniconda3/envs/nano_moe/bin/python \
  scripts/run_tpot_performance_breakdown.py \
  --phase smoke \
  --output-dir results/tpot_performance_breakdown
```

运行 smoke，通过后自动运行全部 80-request validation：

```bash
/home/linke/miniconda3/envs/nano_moe/bin/python \
  scripts/run_tpot_performance_breakdown.py \
  --phase all \
  --output-dir results/tpot_performance_breakdown
```

仅运行 validation：

```bash
/home/linke/miniconda3/envs/nano_moe/bin/python \
  scripts/run_tpot_performance_breakdown.py \
  --phase validation \
  --output-dir results/tpot_performance_breakdown
```

注意：本报告没有执行后两条 validation 命令。

### 7.2 单独运行每个配置

通用形式：

```bash
PY=/home/linke/miniconda3/envs/nano_moe/bin/python

"$PY" scripts/run_tpot_performance_breakdown.py \
  --phase smoke \
  --stage <STAGE_ID> \
  --output-dir results/tpot_performance_breakdown
```

每个配置的命令：

```bash
# P0-Eager
"$PY" scripts/run_tpot_performance_breakdown.py --phase smoke \
  --stage p0_eager --output-dir results/tpot_performance_breakdown

# P0+Graph
"$PY" scripts/run_tpot_performance_breakdown.py --phase smoke \
  --stage p0 --output-dir results/tpot_performance_breakdown

# P1 + Drafter
"$PY" scripts/run_tpot_performance_breakdown.py --phase smoke \
  --stage p1 --output-dir results/tpot_performance_breakdown

# P2 + Rerouter
"$PY" scripts/run_tpot_performance_breakdown.py --phase smoke \
  --stage p2 --output-dir results/tpot_performance_breakdown

# R-Eager
"$PY" scripts/run_tpot_performance_breakdown.py --phase smoke \
  --stage r_eager --output-dir results/tpot_performance_breakdown

# P3 + Segment Graph
"$PY" scripts/run_tpot_performance_breakdown.py --phase smoke \
  --stage p3 --output-dir results/tpot_performance_breakdown

# P4 + Predictive Prefetch
"$PY" scripts/run_tpot_performance_breakdown.py --phase smoke \
  --stage p4 --output-dir results/tpot_performance_breakdown

# P5 + Early Stop
"$PY" scripts/run_tpot_performance_breakdown.py --phase smoke \
  --stage p5 --output-dir results/tpot_performance_breakdown
```

编排脚本会将 `<STAGE_ID>` 展开为固定 manifest。查看每个配置的完整 benchmark 命令：

```bash
"$PY" scripts/run_tpot_performance_breakdown.py \
  --phase smoke \
  --stage all \
  --output-dir results/tpot_performance_breakdown \
  --dry-run \
  --print-commands
```

每次真实运行的完整 argv、端口、日志和结果目录也保存在对应的 `run_status.json` 中。

### 7.3 公共 benchmark 命令

编排器为每个 stage 生成的公共部分如下：

```bash
CUDA_VISIBLE_DEVICES=2 \
taskset --cpu-list 64-96 \
/home/linke/miniconda3/envs/nano_moe/bin/python \
scripts/bench_eval_workload_tpot.py \
  --model-path /data1/models/Qwen3-30B-A3B \
  --dataset mt_bench \
  --mt-bench-path /data1/datasets/mt_bench/question.jsonl \
  --request-mode dataset \
  --num-samples 1 \
  --optimized-config k12_transfer_step \
  --cache-ratios 0.3125 \
  --output-lens 512 \
  --max-draft-tokens-values 12 \
  --allocation-modes profile_weighted \
  --slot-buckets 4 \
  --slot-max-bucket-ratio 2.0 \
  --slot-profile-csv pre_exps/exp_and_figs/unique/unique_count_plot_summary_n1024.csv \
  --kt-num-threads 16 \
  --kt-direct-backend avx2_bf16 \
  --verify-cuda-graph-bucket-steps 5,7,8,9,10,11,12,13 \
  --verify-prefetch-max-per-boundary 4 \
  --verify-prefetch-rank-multiplier 1 \
  --gpu-memory-utilization 0.99 \
  --temperature 0.8 \
  --acceptance-strategy standard_sampling \
  --decode-driver generate \
  --collect-profile true \
  --engine-profile true \
  --engine-profile-cuda-sync false \
  --save-profile-json true \
  --save-token-ids true \
  --save-text true \
  --reset-profile-after-warmup true \
  --reset-seed-after-warmup true \
  --reset-profile-before-request true \
  --repeats 1 \
  --skip-existing false \
  --fail-fast true \
  --fail-on-output-validation-error true \
  --seed 20260719 \
  --cache-strategy lru
```

Smoke 使用 `--num-samples 1 --engine-profile true`；validation 使用 `--num-samples all --engine-profile false`。

## 8. Smoke 结果

### 8.1 TPOT 汇总

下表将两次真实 GPU smoke 合并展示：

- P0-Eager、P0 来自修复后定向运行。
- P1–P5、R-Eager 来自第一次完整运行。

| 配置 | 状态 | TPOT (ms/token) | 说明 |
|---|---:|---:|---|
| P0-Eager | Passed | 119.025 | exact heterogeneous eager 参考 |
| P0+Graph | Passed | 30.569 | 修复后的 standard CUDA Graph |
| P1 | Passed | 93.138 | fixed-K round-robin drafter |
| P2 | Passed | 92.389 | fixed-K entropy/cache rerouter |
| R-Eager | Passed | 392.247 | speculative segment 配置的 eager 参考 |
| P3 | Passed | 92.284 | 4 × 12-layer segment graph |
| P4 | Passed | 37.204 | predictive prefetch + protected eviction |
| P5 | Passed | 34.705 | P4 + transfer-aware early stop |

单请求 smoke 的观察值：

- P0 standard graph 相对 P0-Eager 的 TPOT 降低 74.32%，约 3.89×；需要 validation 才能形成正式结论。
- P3 相对 R-Eager 的 TPOT 降低 76.47%，约 4.25×。
- P1 → P2：降低 0.80%。
- P2 → P3：降低 0.11%。
- P3 → P4：降低 59.69%。
- P4 → P5：降低 6.72%。

这些百分比只描述当前一条 smoke 请求，不表示全数据集的稳定收益。

### 8.2 机制验收

| 配置 | 关键机制证据 |
|---|---|
| P0-Eager | graph replay=0；CPU routes=86,559；transfer=0；cache mutation=0 |
| P0+Graph | standard replay=511；KT hybrid replay=511；CPU routes=86,559；transfer=0；cache mutation=0 |
| P1 | draft calls/replays=1,920；verify calls/replays=163；segment replay=0；transfer=0 |
| P2 | draft calls/replays=1,870；verify calls/replays=157；segment replay=0；transfer=0 |
| R-Eager | 所有 graph replay=0；transfer=0 |
| P3 | draft segment replay=7,480=4×1,870；verify segment replay=628=4×157；transfer=0 |
| P4 | runtime=`PredictivePrefetchRuntime`；submit=8,889；ready=8,888；publish=8,888；consume=23,480 |
| P5 | runtime=`PredictivePrefetchRuntime`；submit=7,504；ready=7,501；publish=7,501；consume=23,673；early stop=94 |

P5 的 policy-controlled draft 长度：

```text
min K = 6
max K = 7
terminal clipped K = 3
resolved config mismatches = 0
```

terminal K=3 是固定 512-token 输出末尾的剩余预算裁剪，不是违反 `min K=6` 的 early-stop 决策。

### 8.3 每次 Verify 的 CPU/GPU Experts 与 Latency

#### 统计口径

这里的一个 expert 表示一个 `(verify call, layer, expert_id)` 的
logical expert-layer instance：

- 同一 expert 在同一层的一次 verify 中即使接收多个 token routes，
  也只计数一次。
- 同一 expert 出现在不同层时分别计数，因为它们对应不同层的独立权重和计算。
- `active_experts` 是该 verify 中 48 层所有被 router 选中的唯一 experts
  数量之和。
- `cpu_experts` 是其中实际 cache miss、由 KT-direct CPU 计算的唯一
  experts 数量之和。
- `gpu_experts = active_experts - cpu_experts`，表示由驻留 GPU experts
  承担的 logical experts。

计算公式：

```text
avg_cpu_experts_per_verify
  = Σ verify_realized_cpu_expert_count_sum / verify_call_count

avg_gpu_experts_per_verify
  = (Σ verify_activated_expert_set_size_sum
     - Σ verify_realized_cpu_expert_count_sum)
    / verify_call_count

avg_*_experts_per_layer_per_verify
  = avg_*_experts_per_verify / 48
```

KT hybrid graph 为保持固定 shape，会让 CPU miss routes 在 GPU grouped
GEMM 中经过零权重 substitute slots。本表统计 logical GPU-hit experts，
不把这些零权重 placeholder 当作 GPU-computed experts。

Latency 使用原正式 smoke 中每条 `model_verify_call_records`：

- `forward latency`：模型 verify forward、LM head 及 rank gather 的
  `forward_ms`。
- `E2E latency`：从进入 `run_verify` 到输出拆分、metadata/prefetch
  finalize 完成的 `total_ms`。

#### 每阶段平均值

| 阶段 | Verify calls | Tokens/verify | CPU experts/verify | GPU experts/verify | Total experts/verify | Forward latency (ms) | E2E latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0-Eager | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| P0+Graph | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| P1 | 163 | 12.779 | 713.429 | 926.810 | 1,640.239 | 2.426 | 3.161 |
| P2 | 157 | 12.911 | 730.465 | 933.376 | 1,663.841 | 2.055 | 2.771 |
| R-Eager | 157 | 12.911 | 730.465 | 933.376 | 1,663.841 | 193.257 | 193.868 |
| P3 | 157 | 12.911 | 730.465 | 933.376 | 1,663.841 | 2.229 | 2.949 |
| P4 | 65 | 12.892 | 297.477 | 1,337.815 | 1,635.292 | 54.211 | 69.869 |
| P5 | 95 | 7.432 | 185.768 | 1,109.263 | 1,295.032 | 45.138 | 59.600 |

P0-Eager 和 P0+Graph 是 exact autoregressive decode，没有 speculative
`run_verify`，因此 verify 相关统计必须为 N/A。它们对应的 AR decode
step latency 为：

| 阶段 | Decode steps | Mean (ms) | p50 (ms) | p90 (ms) |
|---|---:|---:|---:|---:|
| P0-Eager | 511 | 119.025 | 118.821 | 122.544 |
| P0+Graph | 511 | 30.569 | 29.894 | 35.784 |

#### 每层归一化 Expert 数量

为了避免 48 层求和的数值过大，下表同时给出每层、每次 verify 的平均值：

| 阶段 | CPU experts/layer/verify | GPU experts/layer/verify | Total experts/layer/verify |
|---|---:|---:|---:|
| P1 | 14.863 | 19.309 | 34.172 |
| P2 | 15.218 | 19.445 | 34.663 |
| R-Eager | 15.218 | 19.445 | 34.663 |
| P3 | 15.218 | 19.445 | 34.663 |
| P4 | 6.197 | 27.871 | 34.069 |
| P5 | 3.870 | 23.110 | 26.980 |

P2、R-Eager、P3 的 expert 数量完全相同，不是近似值：

```text
active experts total = 261,223
CPU experts total    = 114,683
GPU experts total    = 146,540
```

三者使用相同 seed、输出 digest、verify calls/tokens 和静态 cache，
仅执行方式不同（monolithic graph、eager、segment graph）。这也验证了
graph 切分没有改变 logical expert routing。

P4/P5 中 CPU experts 显著减少，是当前 verify 时刻实际 cache status 的
统计结果，反映 predictive prefetch 后更多 active experts 已驻留 GPU。
P5 的 total experts/verify 进一步降低，主要因为 early stop 将平均
verify 长度从约 12.9 tokens 降至 7.43 tokens。

#### Latency 分布与限制

| 阶段 | Forward p50 (ms) | Forward p90 (ms) | E2E p50 (ms) | E2E p90 (ms) |
|---|---:|---:|---:|---:|
| P1 | 1.700 | 2.504 | 2.388 | 3.341 |
| P2 | 1.700 | 2.459 | 2.401 | 3.148 |
| R-Eager | 190.966 | 217.313 | 191.580 | 217.923 |
| P3 | 1.841 | 2.630 | 2.542 | 3.371 |
| P4 | 53.777 | 64.691 | 67.706 | 86.103 |
| P5 | 43.035 | 55.718 | 56.291 | 74.706 |

本次 smoke 使用 `engine_profile_cuda_sync=false`。因此以上 latency 是
正式 run 中记录的 host-observed 区间：

- 不会为了 profile 在每个区间额外执行全局 CUDA synchronize。
- CUDA graph 的异步 device completion 可能部分落到后续同步点。
- P4/P5 的 E2E 数值包含 metadata/prefetch finalize，不能直接解释为纯
  MoE kernel latency。

这些数值适合解释本次执行路径和相对开销，但若需要严格 device
component latency，应另行运行 CUDA-event latency breakdown；不能把
本表当作纯 GPU kernel 时间。

### 8.4 每个结果的验证文件

P0：

- [P0-Eager mechanism validation](../results/tpot_performance_breakdown_p0_graph_fix/smoke/p0_eager_exact_hetero_ar/mechanism_validation.json)
- [P0+Graph mechanism validation](../results/tpot_performance_breakdown_p0_graph_fix/smoke/p0_exact_hetero_ar/mechanism_validation.json)

其余配置：

- [P1 mechanism validation](../results/tpot_performance_breakdown/smoke/p1_drafter/mechanism_validation.json)
- [P2 mechanism validation](../results/tpot_performance_breakdown/smoke/p2_rerouter/mechanism_validation.json)
- [R-Eager mechanism validation](../results/tpot_performance_breakdown/smoke/r_eager/mechanism_validation.json)
- [P3 mechanism validation](../results/tpot_performance_breakdown/smoke/p3_segment_graph/mechanism_validation.json)
- [P4 mechanism validation](../results/tpot_performance_breakdown/smoke/p4_predictive_prefetch/mechanism_validation.json)
- [P5 mechanism validation](../results/tpot_performance_breakdown/smoke/p5_early_stop_full/mechanism_validation.json)

## 9. 结论与后续工作

### 9.1 已确认

1. 固定 manifest 能正确区分 exact AR、drafter、rerouter、segment graph、predictive prefetch 和 early stop。
2. P0 原始失败来自 heterogeneous dynamic route planning，而不是 KT-direct backend 本身。
3. 修复后的 standard graph 使用固定形状 exact KT-direct hybrid 路径，完成了 511 次 decode replay。
4. P0-Eager 与 P0+Graph 的 token IDs、digest 和 CPU route 数完全一致。
5. P1/P2 是 monolithic graph，P3 是每 forward 四段 graph。
6. P4/P5 确实激活 `PredictivePrefetchRuntime` 和运行时 transfer。
7. P5 确实发生 transfer-aware early stop，policy K 保持在 6–12 范围内。

### 9.2 尚未完成

80-request validation 尚未运行，因此当前不能给出：

- 稳定 pooled TPOT。
- 有意义的 request-level p50/p90。
- 有意义的 bootstrap 95% CI。
- 可用于论文或正式报告的累积收益百分比。

正式 validation 应使用：

```bash
/home/linke/miniconda3/envs/nano_moe/bin/python \
  scripts/run_tpot_performance_breakdown.py \
  --phase all \
  --output-dir results/tpot_performance_breakdown
```

验收要求每个配置：

```text
80/80 requests passed
512 generated tokens per request
no output validation error
```

TPOT 不要求随 P0→P5 严格单调；最终结论应以 validation 的 pooled TPOT 和 request bootstrap CI 为准。
