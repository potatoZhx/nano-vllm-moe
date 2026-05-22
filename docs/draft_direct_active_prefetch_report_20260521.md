# Draft Direct-Active Prefetch Report - 2026-05-21

## 结论

本次实现新增了一个可切换的 draft prefetch runtime：

- `prefetch_runtime_mode="baseline_staging"`：默认值，保持已实现的 Approach 1 staging memory + replay-boundary publish，不改变原有功能。
- `prefetch_runtime_mode="draft_direct_active"`：draft CUDA Graph 仍负责产生 gating metadata，但 draft prefetch 不再写 staging buffer，而是在安全 frontier 后直接写 GPU active expert cache。CPU 仍负责 GPU expert cache 的选择、替换和状态管理；H2D expert copy、cache 替换和 publish 都不放进 CUDA Graph。

最终选择的路径是 Approach 3 的 segment frontier direct-active prefetch，并保留 Approach 2 的 `layer` granularity 作为实验开关。默认 segment 大小从细粒度的 4 层调整为 12 层，因为 A100 实测显示 4 层 segment 的边界数量过多，metadata enqueue 和 host buffer 复用等待会反过来破坏 draft graph 的收益。

当前实现已满足“draft CUDA Graph 中支持 prefetch”的功能目标：segment graph replay 期间会在每个 segment 结束后 offload 该 segment 的 metadata，并用 `frontier_layer_idx=segment_end-1` 提交 direct-active prefetch。正确性方面，已加入 stale metadata submit guard 和下一次 draft replay 前 direct-active drain，避免异步 prefetch 在 graph replay 仍可能读取的 layer 上替换 active expert。

性能上，Approach 1 的 publish/metadata 路径确实是瓶颈之一；direct-active segment 模式已经移除了 draft-selected expert 的 staging publish-back，但 `<3 ms` 完全隐藏目标还没有彻底达成。最好的当前配置是 `draft_prefetch_segment_size=12`：draft metadata enqueue 和 host buffer wait 已降到低位，但 CPU queue ranking/direct-active submit 仍有约 12.9 ms/call 的 worker-side visible accounting，需要继续优化候选队列和 submit 节流。

## Approach 1 瓶颈

Approach 1 的安全性来自 staging buffer：draft prefetch 先写 staging slot，等 replay boundary 后再 publish 到真正的 active expert cache。代价是每个有用 expert 多了一次 publish-back 路径，并且 metadata observe、global queue ranking、staging submit、publish scan 都在 replay 外形成 CPU 侧压力。

A100 hot-cache baseline 的形状如下：

| Metric | baseline_staging hot |
|---|---:|
| draft graph hit rate | 1.0 |
| draft graph replay/call | 20.62 ms |
| draft forward/call | 44.27 ms |
| draft metadata collect/call | 8.93 ms |
| draft metadata observe/call | 3.00 ms |
| draft submit-after/call | 10.90 ms |
| total publish | 35.70 ms |
| staging publish | 5.25 ms |
| async hidden ratio | 93.85% |

这说明 graph replay 本身不是主要退化点。draft replay 约 20 ms/call，瓶颈主要在 replay 外的 metadata 消费、queue scheduling、staging submit/publish 和 verify-layer direct-active publish scan。

## Approach 2 和 Approach 3

Approach 2，layer frontier：

- layer `i` replay 完成后，`<= i` 的 layer active expert cache 可以被替换。
- `> i` 的未来 layer 仍被保护，避免与当前 draft graph 后续 replay 冲突。
- overlap 机会最大，但每层一个边界会产生大量 Python 调度、metadata offload 和 queue submit 压力。

Approach 3，segment frontier：

- 多个 layer 组成一个 segment，segment replay 完成后只允许替换已越过 frontier 的 layer。
- 安全性接近 layer frontier，但边界数显著更少。
- 本实现把 `layer` 视为 segment size 1，用于验证性能包络；默认使用更粗的 12-layer segment。

选择 segment-frontier 的原因是：prefetch 需要的资源主要是 CPU 侧 expert 选择/cache 管理和 PCIe 传输，CUDA Graph 只需要提供可异步消费的 metadata。segment frontier 可以把 metadata offload 和 H2D 提前到 draft replay 中间，同时避免替换未来 layer 仍会读取的 active expert。

## 实现内容

- `nanovllm/config.py`
  - 新增 `prefetch_runtime_mode`，默认 `baseline_staging`。
  - 新增 `draft_prefetch_frontier_granularity`、`draft_prefetch_segment_size`、`draft_prefetch_visible_budget_ms`、`draft_prefetch_min_per_boundary`、`draft_prefetch_max_per_boundary`。
  - 新增 `prefetch_metadata_host_buffer_pool_size` 和 `draft_prefetch_segment_host_buffer_pool_size`。后者为 0 时自动按 segment 数扩展 draft host metadata buffer pool。

- `nanovllm/models/qwen3_moe.py`
  - 增加 `forward_layers(...)` 和 `forward_draft_segment(...)`，允许捕获 decoder layer range 的 CUDA Graph。
  - full forward 路径保持等价。

- `nanovllm/engine/model_runner.py`
  - 增加 segmented draft CUDA Graph capture/replay。
  - segment replay 后只 offload `[layer_start, layer_end)` metadata。
  - metadata item 携带 `frontier_layer_idx`，draft direct-active submit 只允许替换 frontier 后方 layer。
  - stale draft metadata submit 会跳过，下一次 draft replay 前 drain 已完成的 direct-active prefetch，避免 cache 替换和 graph replay 冲突。

- `nanovllm/expert/runtime_meta.py`
  - `offload_async()` 支持 partial layer range。
  - draft segment/layer 模式下自动扩大 host metadata buffer pool，减少 replay 中等待 host slot 复用。

- `nanovllm/expert/prefetcher.py`
  - 增加 `submit_draft_direct_active_prefetch(...)`，直接写 active expert cache slot，不使用 staging buffer。
  - global queue candidate ranking 支持 `max_layer_idx`，在排序前剪掉未来 layer，避免 segment frontier 下先排序再丢弃。
  - 增加 direct-active draft submit/ready/publish/consume、drain、frontier、budget、visible overhead 等 profile counters。
  - adaptive budget 会根据 visible overhead 在 `min/max_per_boundary` 之间调整，`get_profile(reset=True)` 会重置预算，避免 warmup 污染测量。

- `examples/heterogeneous_benchmark_case.py`
- `examples/benchmarks/draft_standard_decode_forward_bench.py`
  - 增加新参数透传和 profile 字段。

## 正确性修复

在 A100 上，早期 `segment + direct_active` 曾触发 device-side assert。根因是 async metadata worker 可能在 draft replay 之后才消费某个 segment 的 metadata，然后提交 direct-active cache 替换；此时替换可能和下一次 draft graph replay 读取同一 layer active expert 发生冲突。

修复策略：

- metadata submit 阶段检查 `item.step_id == _active_draft_prefetch_step_id`。如果 draft metadata 已经过期，只 observe 队列，不再提交 direct-active prefetch。
- 每次 `run_draft()` arm/replay draft graph 前调用 `drain_direct_active_ready(step_id=step_id)`，把上一轮已经 ready 的 direct-active ticket publish 完，避免 replay 期间 cache slot 处于 pending/替换中。
- segment frontier 只允许替换 `layer_idx <= frontier_layer_idx` 的 layer。

修复后，small guarded case 和 full guarded case 都能在 A100 上通过，draft graph hit rate 保持 1.0。

## A100 Benchmark

环境：

- Node examples: `gpu15-A100-E2-3U`, `gpu11-A100-E1-3U`
- GPU: NVIDIA A100-SXM4-80GB
- Conda: `nano_moe`
- Torch: `2.9.1+cu128`
- Model: `/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B`
- Case: spec-only, cache slots/layer 50, `draft_top_c=0`, draft CUDA Graph enabled, batch 4, output 8, temperature 0.8.

Important caveat: end-to-end elapsed and digest are not clean A/B metrics here because temperature 0.8 changes acceptance traces. The most useful numbers are graph hit rate, graph replay/call, metadata enqueue/wait, submit-after, and direct-active counters.

| Metric | baseline staging | direct iteration old | segment 4 pre-opt | segment 4 optimized | segment 12 optimized |
|---|---:|---:|---:|---:|---:|
| graph hit rate | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| draft calls | 11 | 10 | 11 | 9 | 6 |
| draft graph replay/call | 20.62 ms | 20.09 ms | 54.23 ms | 24.33 ms | 23.07 ms |
| segment graph replays/call | 0 | 0 | 12 | 12 | 4 |
| segment metadata enqueue/call | 0 | 0 | 4.59 ms | 3.41 ms | 1.11 ms |
| draft host buffer wait/call | 2.44 ms | 3.44 ms | 43.36 ms | 2.27 ms | 0.60 ms |
| draft metadata collect/call | 8.93 ms | 9.95 ms | 8.18 ms | 7.79 ms | 7.15 ms |
| draft metadata observe/call | 3.00 ms | 3.72 ms | 3.56 ms | 2.98 ms | 3.19 ms |
| draft submit-after/call | 10.90 ms | 3.47 ms | 22.39 ms | 16.78 ms | 12.89 ms |
| draft direct-active submits | 0 | 3 | 24 | 13 | 9 |
| draft direct-active consumed hits | 0 | 33 | 84 | 25 | 13 |
| frontier skips | 0 | 0 | 738 | 0 | 0 |
| draft direct-active drain/call | 0 | 0 | 0.26 ms | 0.22 ms | 0.19 ms |
| async hidden ratio | 93.85% | 94.25% | 74.57% | 99.51% | 98.86% |

Benchmark artifacts:

- `benchmarks/results/spec_prefetch_baseline_staging_hot_20260521.json`
- `benchmarks/results/spec_prefetch_draft_direct_active_20260521.json`
- `benchmarks/results/spec_prefetch_draft_direct_active_segment_guarded_20260521.json`
- `benchmarks/results/spec_prefetch_draft_direct_active_segment_optimized_20260521.json`
- `benchmarks/results/spec_prefetch_draft_direct_active_segment12_20260521.json`
- `/home/mumura/moe_spec/logs/draft_direct_active_segment_guarded_full_20260521_220953.log`
- `/home/mumura/moe_spec/logs/draft_direct_active_segment_optimized_20260521_222954.log`
- `/home/mumura/moe_spec/logs/draft_direct_active_segment12_20260521_223254.log`

## 性能分析

1. Approach 1 的 replay 本身没有明显退化。

   baseline 的 draft graph replay/call 是 20.62 ms，direct iteration old 是 20.09 ms。瓶颈不在 CUDA Graph replay，而在 graph 外 metadata 和 publish。

2. iteration-level direct-active 能证明 staging publish-back 可以被绕开，但 overlap 不够。

   old direct-active iteration run 把 draft submit-after/call 从 10.90 ms 降到 3.47 ms，graph replay 不退化。但它只能在完整 replay boundary 后提交，不能把 H2D 更早塞进 draft replay 中间。

3. naive 4-layer segment 边界过多。

   pre-opt 的 4-layer segment 每次 draft call 有 12 个 segment replay 和 12 次 metadata enqueue，host buffer wait/call 高达 43.36 ms，async hidden ratio 降到 74.57%。这不是 CUDA Graph 被 prefetch 破坏，而是 Python 边界和 host buffer 池不够导致主路径等待。

4. frontier prefilter 和自动 host buffer pool 扩展有效。

   optimized 4-layer segment 将 frontier skip 从 738 降到 0，将 draft host buffer wait/call 从 43.36 ms 降到 2.27 ms，async hidden ratio 提升到 99.51%。这说明 metadata offload 本身可以不破坏 CUDA Graph，但边界数量仍然太多。

5. 当前推荐配置是 12-layer segment。

   `draft_prefetch_segment_size=12` 把 segment boundary 从 12 个/call 降到 4 个/call，segment metadata enqueue/call 降到 1.11 ms，host buffer wait/call 降到 0.60 ms，draft graph replay/call 为 23.07 ms。相比 baseline full draft graph 的 20.62 ms，graph 层面的额外成本约 2.45 ms/call，已经接近 `<3 ms` 目标。

6. 未达标部分在 CPU queue submit，而不是 graph replay。

   segment 12 的 draft direct-active visible overhead 仍约 12.88 ms/call，主要来自 async worker 中候选队列排序、active slot reservation、budget 判断和 submit。虽然大部分被 async hidden ratio 覆盖，但 worker backlog 会带来 stale submit skip，说明继续增加 prefetch 数量并不一定有收益。

## 当前默认和使用方式

推荐 opt-in 配置：

```python
LLM(
    ...,
    spec_enable_prefetch=True,
    prefetch_runtime_mode="draft_direct_active",
    draft_prefetch_frontier_granularity="segment",
    draft_prefetch_segment_size=12,
    draft_prefetch_visible_budget_ms=3.0,
    draft_prefetch_min_per_boundary=0,
    draft_prefetch_max_per_boundary=4,
)
```

实验选项：

- `draft_prefetch_frontier_granularity="iteration"`：最低 boundary overhead，但没有 draft replay 内 overlap。
- `draft_prefetch_frontier_granularity="segment"`：当前推荐路径。
- `draft_prefetch_frontier_granularity="layer"`：用于测量最大 overlap 上限，不建议作为默认。
- `draft_prefetch_segment_host_buffer_pool_size=0`：自动按 segment 数扩展 host buffer pool。显式大于 0 时使用用户给定目标。

## 验证

本地验证命令：

```bash
eval "$(conda shell.bash hook)" && conda activate nano_moe
python -m py_compile nanovllm/config.py nanovllm/expert/runtime_meta.py nanovllm/expert/prefetcher.py nanovllm/models/qwen3_moe.py nanovllm/engine/model_runner.py examples/heterogeneous_benchmark_case.py examples/benchmarks/draft_standard_decode_forward_bench.py
python -m unittest tests/test_prefetch_runtime_meta.py tests/test_prefetch_runtime.py tests/test_config_prefetch.py tests/test_model_runner_prefetch.py tests/test_draft_standard_decode_forward_bench.py tests/test_draft_cuda_graph.py
python -m unittest tests.test_verify_prefetch_comprehensive.TestExpertCacheActiveReservation tests.test_verify_prefetch_comprehensive.TestVerifyLayerPrefetchRuntime tests.test_verify_prefetch_comprehensive.TestConfigPrefetchIntegrated tests/test_verify_feedback.py
git diff --check
```

结果：

- py_compile passed。
- 56 个 targeted metadata/runtime/config/model-runner/benchmark/draft-graph tests passed，2 个 CUDA-only tests skipped。
- 24 个 verify-prefetch tests passed。
- `git diff --check` passed。
- verify-prefetch 测试中仍有 pre-existing temporary config file ResourceWarning，不影响结果。

## 后续优化

1. 建 per-layer candidate queue 或 frontier-aware heap，避免每个 segment boundary 重新扫全局队列。
2. 给 draft direct-active submit 加 worker backlog 感知：当 metadata queue depth 或 stale skip 升高时，把 adaptive budget 更快降到 0 或暂停若干 boundary。
3. 将 segment metadata observe 和 submit 解耦：当前 stale metadata 仍会更新 global queue，但 submit 会跳过；可以进一步合并同一 draft step 的多个 segment update，降低 worker item 数。
4. 单独拆分 verify-layer direct-active publish counters 和 draft direct-active counters，避免 benchmark dashboard 把两条路径的 publish scan 混在一起。
5. 在 temperature 0 或固定 acceptance trace 下补一组 cleaner A/B，专门比较 standard decode CUDA Graph、baseline staging draft graph、segment 12 draft graph。
