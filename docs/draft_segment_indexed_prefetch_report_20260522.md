# Draft Segment Indexed Prefetch Implementation Report

本文记录 `prefetch_runtime_mode="draft_segment_indexed"` 的实现。它是新的可切换功能，不改变旧的 `baseline_staging` 和 `draft_direct_active` 路径。

## 术语

- `draft forward`：speculative decoding 中 draft model 的一次完整 forward 流程，包含 CUDA graph replay、metadata offload、sampler 等主路径开销。
- `standard decode`：非 speculative decoding 的普通 decode forward。
- `metadata`：MoE router 产生的每层 expert id、routing weight、token count，用于判断哪些 expert 值得预取。
- `prefetch`：提前把 CPU 上的 expert 权重通过 PCIe H2D copy 拷到 GPU expert cache。
- `segment`：一段连续 decoder layer。默认 segment size 为 12，48 层模型会拆成 4 个 segment。
- `frontier`：当前 draft CUDA graph replay 已经完成的最大 layer。frontier 之前的 layer 本轮不会再被 replay 读取，因此可以安全替换这些 layer 的 GPU expert cache。
- `candidate`：一个可预取的 `(layer_idx, expert_idx)`。
- `victim slot`：GPU expert cache 中被选中替换的 slot。
- `deferred mapping`：先把 expert 数据拷进 active cache slot，但延迟更新 GPU LUT/mask，直到 copy ready 且处在安全边界。

## 实现概览

新路径的调用链如下：

```text
ModelRunner.run_draft()
  runtime_meta_recorder.arm(mode="draft")
  PrefetchRuntime.begin_draft_iteration(step_id)
  run()
    _replay_draft_segment_graph()
      replay segment graph
      _enqueue_draft_segment_metadata(frontier)

metadata worker:
  collect histogram metadata
  PrefetchRuntime.observe_draft()
    update draft per-segment index
  submit_draft_segment_indexed_prefetch(frontier)
    read current frontier segment candidates
    select victim from CPU cache state
    reserve active slot without publishing GPU mapping
    enqueue H2D expert copy

verify wait / next draft:
  publish_direct_active_ready()
    commit deferred GPU LUT/mask mapping after copy ready
```

和旧 `draft_direct_active` 的关键区别：

- metadata 在 GPU 上预聚合成 per-layer histogram，不再由 CPU 对小 tensor 做 `torch.unique/scatter_add`。
- draft candidate 不进入全局队列，而是进入当前 spec iteration 的 per-segment index。
- prefill/verify metadata 仍作为长期准确频率来源，进入 long-term per-segment index。
- victim selection 使用 CPU 侧 list 状态，不调用 `cache.snapshot()`，避免 GPU tensor clone 和 `.tolist()` 同步。
- draft metadata 不丢弃。若 metadata 错过 prefetch 窗口，仍然 observe 统计，只跳过不安全的 submit，并记录 missed-window counter。

## 主要代码变更

- `Config` 新增 opt-in runtime mode：`draft_segment_indexed`。
- `ModelRuntimeMetaRecorder` 在新 mode 的 draft 路径使用 histogram metadata buffer：
  - `activation_count[layer, expert]`
  - `score_sum[layer, expert]`
  - `token_count[layer]`
- `PrefetchRuntime` 新增：
  - `SegmentCandidateIndex`，按 segment 保存候选。
  - `long_term_segment_index`，接收 prefill/verify history。
  - `draft_segment_index`，只接收当前 spec iteration 的 draft metadata。
  - `submit_draft_segment_indexed_prefetch()`，只扫描 frontier 对应 segment。
- `LayerExpertCache` 新增 deferred active prefetch lifecycle：
  - `reserve_active_slot_for_prefetch_deferred()`
  - `commit_deferred_active_prefetch()`
  - `cancel_deferred_active_prefetch()`

## 正确性与安全边界

deferred mapping 的时间线：

```text
T0: segment N replay 完成
T1: CPU 选择 segment N 的 candidate 和 victim slot
T2: H2D copy 写入 victim slot 的 weight buffer
T3: copy ready 前，GPU LUT/mask 仍指向旧 expert
T4: copy ready 后，publish_direct_active_ready() 更新 GPU LUT/mask
T5: 下一次可能消费该 layer 的 graph/verify 才能看到新 expert
```

这样做避免了“copy 还没完成但 GPU mapping 已经指向新 expert”的错误，也避免了在 draft graph 内做 cache replacement。

## Profile Counters

新增或扩展的关键 counters：

- `draft_segment_indexed_prefetch_visible_overhead_ms`
- `draft_segment_indexed_rank_ms`
- `draft_segment_indexed_victim_select_ms`
- `draft_segment_indexed_prefetch_est_transfer_ms`
- `draft_segment_indexed_missed_prefetch_window_count`
- `segment_index_aggregate_ms`
- `segment_index_filter_ms`
- `segment_index_entry_update_ms`
- `run_draft_missed_prefetch_window_count`

这些 counters 用于拆分 CPU 调度开销、候选索引开销、victim selection 开销和真实 H2D 估算成本。

## 测试结果

已运行：

```text
python -m pytest -q \
  tests/test_config_prefetch.py \
  tests/test_prefetch_runtime_meta.py \
  tests/test_expert_cache_staging.py \
  tests/test_prefetch_runtime.py \
  tests/test_prefetch_global_queue.py \
  tests/test_prefetch_strategy.py \
  tests/test_prefetch_wait.py \
  tests/test_model_runner_prefetch.py \
  tests/test_spec_engine_prefetch.py \
  tests/test_draft_cuda_graph.py \
  tests/test_draft_cuda_graph_real_world.py \
  tests/test_mode_config.py

59 passed, 3 skipped

python -m pytest -q \
  tests/test_prefetch_runtime_meta.py::TestPrefetchRuntimeMeta::test_histogram_record_layer_is_capture_safe

1 passed
```

## A100 Benchmark 验证

环境：

- A100 单卡，`CUDA_VISIBLE_DEVICES=1`。
- `num_seqs=1`，`input_len=24`，`output_len=8`，`max_model_len=128`。
- `temperature=0.0`，`enforce_eager=false`，开启 engine profile 和 CUDA sync profile。
- spec 参数：`max_draft_tokens=4`，`draft_top_c=0`，segment size 12。

结果文件：

- `benchmarks/results/draft_segment_indexed_bs1_20260522_231506/standard_cuda_graph.json`
- `benchmarks/results/draft_segment_indexed_bs1_20260522_231506/spec_draft_direct_active_segment12.json`
- `benchmarks/results/draft_segment_indexed_bs1_20260522_231506/spec_draft_segment_indexed_segment12.json`

核心结果：

| 指标 | standard CUDA graph | draft_direct_active segment12 | draft_segment_indexed segment12 |
| --- | ---: | ---: | ---: |
| 输出 digest | `7d829b...bb14` | `ebe069...beea` | `7d829b...bb14` |
| decode/draft forward summary | 13.38 ms | 26.19 ms | 24.35 ms |
| run model decode/draft total | 12.72 ms/call | 26.17 ms/call | 24.33 ms/call |
| graph replay | 12.21 ms/call | 17.70 ms/call | 18.77 ms/call |
| sampler | 0.46 ms/call | 0.52 ms/call | 0.53 ms/call |
| draft metadata offload | N/A | 6.93 ms/call | 9.48 ms/call |
| draft prefetch before | N/A | 4.74 ms/call | 0.79 ms/call |
| draft submit after | N/A | 12.60 ms/call | 10.20 ms/call |
| visible draft prefetch overhead | N/A | 12.58 ms/call | 10.14 ms/call |

解读：

- 新路径功能正确性更好：`draft_segment_indexed` 的输出 digest 与 standard decode 一致，旧 `draft_direct_active` 本次不一致。
- 新路径的完整 draft forward 为 24.35 ms，比旧 direct-active 的 26.19 ms 快 1.84 ms。
- 目标是完整 draft forward 与 standard CUDA graph decode forward 的差距 `<5ms`。本次差距为 `24.35 - 13.38 = 10.97 ms`，仍未达标。
- sampler 不是瓶颈。indexed 路径 sampler 为 0.53 ms/call，只比 standard 的 0.46 ms/call 高 0.07 ms/call。
- 当前最大剩余问题是 CPU 调度和 submit 路径仍太重。indexed 路径 `visible draft prefetch overhead` 为 10.14 ms/call，其中 candidate ranking 为 4.89 ms/call，victim selection 只有 0.05 ms/call，说明 CPU 侧 cache victim 备份已经压低了替换选择开销，下一步应继续压缩候选排序和提交流程。
- indexed 路径没有丢 metadata：`stale_metadata_observe_count=0`，`missed_prefetch_window_count=0`。这说明当前 segment 边界消费流程没有窗口错过，但代价是 worker/submit 仍占用了过多 CPU 时间。
- graph replay 本身也需要继续压缩。indexed 路径 graph replay 为 18.77 ms/call，比 standard 的 12.21 ms/call 高 6.56 ms/call，也比旧 direct-active 高 1.08 ms/call。这部分可能来自 draft segment graph 被拆成 4 段 replay、segment metadata enqueue、以及 histogram scatter_add 进入 graph 后的额外 kernel。

## 后续优化方向

按本次 profile，后续优化优先级如下：

- candidate ranking 从每次 Python 排序改为固定大小 top-k heap 或预排序 bucket，只在分数变化的 segment 增量更新。
- submit path 合并 candidate scan、pending check、budget check 和 reservation，减少 Python 函数调用和锁/状态表访问次数。
- draft histogram metadata 继续保留，但要减少每个 segment replay 后的 enqueue 数量和 host buffer 轮转开销。
- 评估 segment size 24 或 layer group 合并，减少一次 draft forward 中的 graph replay 边界数量。
- 将 `score_sum/count` 的 CPU collect 后处理改为只扫描非零 expert 或 GPU 预生成 compact top-k metadata，减少 histogram 全 expert 行传输和 CPU `nonzero/index_select` 开销。
