# Draft Direct-Active Prefetch Report - 2026-05-21

## 结论

本文说明一个新的、默认关闭的 draft prefetch runtime。这里的 draft 指 speculative decoding 里的草稿模型前向；prefetch 指提前把后续可能用到的 expert 权重从 CPU 传到 GPU；runtime 指运行时调度逻辑。

新增的运行模式是：

- `prefetch_runtime_mode="baseline_staging"`：默认模式，保持已有的 Approach 1。staging 是临时缓冲区；这个模式先把预取的 expert 写到 staging buffer，等安全点再 publish 到真正的 GPU expert cache。
- `prefetch_runtime_mode="draft_direct_active"`：新模式。direct-active 表示预取完成后直接写 GPU active expert cache，也就是模型实际读取的 GPU expert 缓存槽，不再经过 staging buffer。

本次最终采用 Approach 3，也就是 segment frontier direct-active prefetch。segment 是一组连续 decoder layer；frontier 是安全边界，表示当前 draft replay 已经越过的最后一层。只有 `layer_idx <= frontier_layer_idx` 的 layer 才允许被 prefetch 替换，因为这些 layer 在本次 draft replay 后续不会再被读取。

CUDA Graph 是 CUDA 的图捕获和重放机制，用于减少 kernel launch 开销。当前实现不把 CPU 侧 GPU expert cache 管理、H2D copy、cache 替换放进 CUDA Graph。H2D 是 host-to-device copy，即 CPU 到 GPU 的数据传输。CUDA Graph 只负责执行 draft forward，并在 MoE gate 处把 gating metadata 记录下来。gating metadata 是每层 MoE gate 选出的 expert id、routing weight 和 token count，CPU prefetch 逻辑用它判断哪些 expert 值得预取。

当前功能状态：

- 默认 `baseline_staging` 不变。
- opt-in `draft_direct_active` 已支持 draft CUDA Graph 内 segment prefetch。
- 默认 `draft_prefetch_segment_size=12`，因为 A100 实测显示 4-layer segment 的边界太多。
- 正确性上已加入 stale metadata guard 和 replay 前 drain，避免异步 prefetch 与 draft graph replay 读同一 GPU active slot 冲突。
- 性能上，segment 12 的 graph 层额外开销约 `2.45 ms/call`，接近 `<3 ms` 目标；但 CPU submit-after 仍约 `12.89 ms/call`，说明后续瓶颈在 CPU 候选队列和 direct-active 提交流程。

## 术语表

| 术语 | 含义 |
|---|---|
| draft | speculative decoding 中先尝试生成候选 token 的草稿前向路径。 |
| verify | 对 draft token 做验证的主模型前向路径。 |
| prefetch | 提前把可能会用到的 CPU expert 权重传到 GPU，减少真正执行时等待。 |
| expert | MoE 模型里的专家子网络。Qwen3-MoE 每层会从多个 expert 中选择一部分执行。 |
| GPU expert cache | GPU 上保存 expert 权重的缓存。GPU 显存放不下全部 expert 时，只缓存一部分。 |
| active expert cache | 模型前向真正会读取的 GPU expert cache 槽。 |
| staging buffer | 临时 GPU 缓冲区。Approach 1 先写这里，再 publish 到 active cache。 |
| publish | 把已经准备好的 expert 标记为可被模型使用；Approach 1 还包含从 staging buffer 替换到 active cache 的动作。 |
| direct-active | 直接预取到 active expert cache，跳过 staging buffer。 |
| CUDA Graph | CUDA 的 graph capture/replay 机制。capture 是捕获一段固定形状的 GPU 执行图，replay 是后续重放它。 |
| replay boundary | 一次 CUDA Graph replay 结束后的安全点。Approach 1 只在这里 publish。 |
| layer frontier | 以单个 layer 为单位的安全边界。layer `i` replay 完后，`<= i` 的 layer 可以替换。 |
| segment frontier | 以一组 layer 为单位的安全边界。segment replay 完后，该 segment 及之前的 layer 可以替换。 |
| metadata offload | 把 GPU 上记录的 gating metadata 异步拷贝到 CPU。 |
| async worker | 后台 CPU 线程，负责收集 metadata、更新候选队列、提交 prefetch。 |
| host buffer | CPU 侧 metadata 缓冲区。这里通常是 pinned host memory，便于异步 D2H copy。 |
| D2H | device-to-host copy，即 GPU 到 CPU 的数据传输。metadata offload 用 D2H。 |
| H2D | host-to-device copy，即 CPU 到 GPU 的数据传输。expert prefetch 用 H2D。 |
| arm | 本代码里的函数含义是“准备 metadata recorder”：设置 step id、mode、token capacity，并清空 token_count，让后续 layer 可以写入 metadata。不是 CUDA Graph capture。 |
| drain | 等待或处理已经 ready 的 prefetch ticket。这里用于在下一次 draft replay 前发布上一轮已完成的 direct-active copy。 |
| ready event | CUDA event 或 CPU immediate event，用来表示一次异步 copy 是否完成。 |
| ticket | 一次 prefetch 请求的运行时记录，包含 layer、expert、目标 slot、event、source 等信息。 |
| stale metadata | 过期 metadata。比如 worker 到很晚才处理上一个 draft step 的 segment metadata，此时它不能再提交会影响当前 replay 的 cache 替换。 |
| adaptive budget | 自适应预算。根据可见开销动态调整每个边界最多提交多少 prefetch。 |
| visible overhead | 在当前线程或 worker 统计中能直接看到的耗时；不代表全部都会暴露到端到端延迟，但会反映 CPU 压力。 |

## 当前执行流程

### Approach 1 baseline

Approach 1 是当前默认路径。它的执行流程是：

```text
run_draft()
  arm metadata recorder
  replay full draft CUDA Graph
    each MoE layer writes gating metadata to GPU metadata buffer
  offload full draft metadata to CPU host buffer
  async worker collect metadata
  async worker update global candidate queue
  async worker submit prefetch to staging buffer

later, at replay boundary / verify wait:
  publish ready staging experts into active expert cache
```

它安全但多了一段 staging publish-back 成本。这里的 publish-back 指把 staging buffer 里的 expert 变成 active expert cache 的可用 expert。

### 当前 direct-active segment path

新路径把一个完整 draft CUDA Graph 拆成多个 segment graph。每个 segment graph 只包含一段 decoder layer。以 48 层模型、`draft_prefetch_segment_size=12` 为例，一次 draft forward 有 4 个 segment：

```text
segment 0: layer  0..11
segment 1: layer 12..23
segment 2: layer 24..35
segment 3: layer 36..47
```

主调用链如下：

```text
SpecEngine.step()
  ModelRunner.run_draft()
    _flush_pending_prefetch_metadata(block=False)
    _set_speculative_execution_mode("draft")
    drain_direct_active_ready()
    _wait_for_prefetch_device_reuse()
    runtime_meta_recorder.arm(mode="draft")
    run()
      _replay_draft_segment_graph()
        replay segment 0 CUDA Graph
        _enqueue_draft_segment_metadata(layer 0..12, frontier=11)
        replay segment 1 CUDA Graph
        _enqueue_draft_segment_metadata(layer 12..24, frontier=23)
        replay segment 2 CUDA Graph
        _enqueue_draft_segment_metadata(layer 24..36, frontier=35)
        replay segment 3 CUDA Graph
        _enqueue_draft_segment_metadata(layer 36..48, frontier=47)
    runtime_meta_recorder.reset()
    _flush_pending_prefetch_metadata(block=False)

metadata async worker, running concurrently:
  wait metadata D2H event
  collect CPU metadata
  observe metadata and update global candidate queue
  submit_draft_direct_active_prefetch(frontier_layer_idx)
    rank candidates with layer_idx <= frontier_layer_idx
    reserve active cache slot
    launch H2D expert copy
```

### 时序图

下面的图展示 segment prefetch 如何试图和后续 segment replay 重叠：

```text
main CUDA stream:
  replay seg0 | replay seg1 | replay seg2 | replay seg3 | logits
       |            |            |            |
       v            v            v            v
metadata stream:
  D2H meta0   D2H meta1   D2H meta2   D2H meta3
       |            |            |            |
       v            v            v            v
CPU async worker:
  collect0 -> queue update -> prefetch layer <= 11
               collect1 -> queue update -> prefetch layer <= 23
                            collect2 -> queue update -> prefetch layer <= 35
                                         collect3 -> queue update -> prefetch layer <= 47
H2D transfer stream:
  copy selected experts directly into active cache slots
```

关键约束是：当 main CUDA stream 还没有 replay 某个未来 layer 时，prefetch 不能替换那个 layer 可能读取的 active expert cache。frontier 就是用来表达这个约束的。

## Approach 1 瓶颈

Approach 1 的安全性来自 staging buffer：draft prefetch 先写 staging slot，等 replay boundary 后再 publish 到真正的 active expert cache。代价是每个有用 expert 多了一次 staging 到 active 的发布路径，并且 metadata observe、global queue ranking、staging submit、publish scan 都在 replay 外形成 CPU 压力。

A100 hot-cache baseline 的形状如下：

| Metric | baseline_staging hot |
|---|---:|
| draft graph hit rate | 1.0 |
| draft graph replay/call | 20.62 ms |
| draft forward/call | 44.29 ms |
| draft metadata collect/call | 8.93 ms |
| draft metadata observe/call | 3.00 ms |
| draft submit-after/call | 10.90 ms |
| total publish | 35.70 ms |
| staging publish | 5.25 ms |
| async hidden ratio | 93.85% |

这说明 CUDA Graph replay 本身不是主要退化点。draft replay 约 20 ms/call，瓶颈主要在 replay 外的 metadata 消费、queue scheduling、staging submit/publish 和 verify-layer direct-active publish scan。

## Approach 2 和 Approach 3

Approach 2 是 layer frontier：

- layer `i` replay 完成后，`<= i` 的 layer active expert cache 可以被替换。
- `> i` 的未来 layer 仍被保护，避免和当前 draft graph 后续 replay 冲突。
- overlap 机会最大，但每层一个边界会产生很多 Python 调度、metadata offload 和 queue submit 压力。

Approach 3 是 segment frontier：

- 多个 layer 组成一个 segment。
- segment replay 完成后，该 segment 及之前 layer 的 active expert cache 可以被替换。
- 安全性接近 layer frontier，但边界数显著更少。

当前实现选择 segment frontier，并把 `layer` 作为实验开关保留。`draft_prefetch_frontier_granularity="layer"` 等价于 segment size 1。默认使用 12-layer segment，是因为 A100 实测显示 4-layer segment 的边界数太多。

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
  - stale draft metadata submit 会跳过，下一次 draft replay 前 drain 已完成的 direct-active prefetch。

- `nanovllm/expert/runtime_meta.py`
  - `offload_async()` 支持 partial layer range。
  - draft segment/layer 模式下自动扩大 host metadata buffer pool，减少 replay 中等待 host slot 复用。

- `nanovllm/expert/prefetcher.py`
  - 增加 `submit_draft_direct_active_prefetch(...)`，直接写 active expert cache slot，不使用 staging buffer。
  - global queue candidate ranking 支持 `max_layer_idx`，在排序前剪掉未来 layer，避免 segment frontier 下先排序再丢弃。
  - 增加 direct-active draft submit/ready/publish/consume、drain、frontier、budget、visible overhead 等 profile counters。
  - adaptive budget 会根据 visible overhead 在 `min/max_per_boundary` 之间调整，`get_profile(reset=True)` 会重置预算，避免 warmup 污染测量。

## 正确性修复

早期 `segment + direct_active` 在 A100 上触发过 device-side assert。device-side assert 是 GPU kernel 内部断言失败，通常表现为后续 CUDA 调用报错。这里的问题不是 graph replay 本身错了，而是 cache 替换的时序错了。

### 出错时间线

假设一次 draft 有 4 个 segment：

```text
T0 main stream replay segment 0, layer 0..11
T1 main stream enqueue metadata for segment 0, frontier=11
T2 main stream replay segment 1, layer 12..23
T3 async worker is busy, metadata segment 0 has not been processed
T4 main stream finishes this draft call
T5 next draft call starts, layer 0..11 may be read again by a new CUDA Graph replay
T6 async worker finally processes old segment 0 metadata
T7 worker submits direct-active prefetch and replaces layer 0 active cache slot
T8 current draft graph replay may still need the old layer 0 slot
T9 graph reads a slot whose expert index/state was changed underneath it
```

直白地说，metadata 是异步处理的。segment 0 的 metadata 可能很晚才被 worker 消费。如果这时已经进入下一次 draft replay，worker 再根据旧 metadata 替换 layer 0 的 active cache，就可能把当前 graph 正在读或即将读的 expert 换掉。

Approach 1 不会遇到这个问题，因为它先写 staging buffer，active cache 只在 replay boundary publish。direct-active 没有 staging buffer，所以必须显式保护 active cache 替换时机。

### 修复方法

修复由三层保护组成：

```text
保护 1: stale submit guard
  如果 metadata item 的 step_id 不是当前 active draft step，
  worker 只 observe metadata 更新队列，不提交 direct-active prefetch。

保护 2: draft replay 前 drain
  每次 run_draft() 准备 replay 前，先处理上一轮已经 ready 的 direct-active ticket。
  这样 replay 开始时不会留下正在 pending 的 active cache 替换。

保护 3: frontier filter
  submit_draft_direct_active_prefetch() 只允许 layer_idx <= frontier_layer_idx。
  当前 segment 之后的未来 layer 不会被替换。
```

对应到时间线：

```text
old behavior:
  old metadata arrives late -> submit prefetch -> replace active slot during next replay -> unsafe

new behavior:
  old metadata arrives late -> step_id mismatch -> skip direct-active submit -> safe

new draft call starts:
  drain ready direct-active tickets -> arm metadata recorder -> replay graph -> safe
```

`arm metadata recorder` 的含义是准备记录 metadata：设置 mode、step id、token capacity，清空 token_count。它不提交 prefetch，也不修改 active expert cache。

修复后，small guarded case 和 full guarded case 都能在 A100 上通过，draft graph hit rate 保持 1.0。

## A100 Benchmark

环境：

- GPU: NVIDIA A100-SXM4-80GB
- Conda: `nano_moe`
- Torch: `2.9.1+cu128`
- Model: `/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B`
- Case: spec-only, cache slots/layer 50, `draft_top_c=0`, draft CUDA Graph enabled, batch 4, output 8, temperature 0.8.

这里的 draft calls 是本次生成中调用 `run_draft()` 的次数。不同配置的 draft calls 不同，不代表某个配置一定更快或更慢。原因是 benchmark 使用 `temperature=0.8`，sampling 和 speculative acceptance trace 会变化；不同 prefetch/cache 状态也可能造成输出 digest 和接受 token 序列不同，进而让某些序列更早完成或更少进入 draft loop。因此下表主要看 per-call 指标，不把端到端 elapsed 当严格 A/B。

draft forward 是 spec profile 中的 `spec_draft_forward_ms`，表示每次 draft call 的平均总耗时，包含 graph replay、metadata enqueue、prefetch-before/drain 等 run_draft 路径开销。表中也列出 graph replay，因为它更接近 CUDA Graph 本身的成本。

| Metric | baseline staging | direct iteration old | segment 4 pre-opt | segment 4 optimized | segment 12 optimized |
|---|---:|---:|---:|---:|---:|
| graph hit rate | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| draft calls | 11 | 10 | 11 | 9 | 6 |
| draft forward/call | 44.29 ms | 45.72 ms | 87.33 ms | 79.17 ms | 67.56 ms |
| draft graph replay/call | 20.62 ms | 20.09 ms | 54.23 ms | 24.33 ms | 23.07 ms |
| segment graph replays/call | 0 | 0 | 12 | 12 | 4 |
| segment metadata enqueue/call | 0 | 0 | 4.59 ms | 3.41 ms | 1.11 ms |
| draft host buffer wait/call | 2.44 ms | 3.44 ms | 43.36 ms | 2.27 ms | 0.60 ms |
| draft metadata collect/call | 8.93 ms | 9.95 ms | 8.18 ms | 7.79 ms | 7.15 ms |
| draft metadata observe/call | 3.00 ms | 3.72 ms | 3.56 ms | 2.98 ms | 3.19 ms |
| draft submit-after/call | 10.90 ms | 3.47 ms | 22.39 ms | 16.78 ms | 12.89 ms |
| draft direct-active visible overhead/call | 0 | 3.43 ms | 22.25 ms | 16.76 ms | 12.88 ms |
| draft direct-active submits | 0 | 3 | 24 | 13 | 9 |
| draft direct-active consumed hits | 0 | 33 | 84 | 25 | 13 |
| frontier skips | 0 | 0 | 738 | 0 | 0 |
| async queue depth max | 4 | 4 | 4 | 14 | 6 |
| stale submit skips | 0 | 0 | 2 | 80 | 17 |
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

### 1. Approach 1 的 replay 本身没有明显退化

baseline 的 draft graph replay/call 是 `20.62 ms`，direct iteration old 是 `20.09 ms`。这说明 CUDA Graph replay 本身稳定。baseline 的 draft forward/call 是 `44.29 ms`，比 replay 多出的部分来自 graph 外 metadata offload、metadata collect、queue submit 和 publish。

### 2. iteration-level direct-active 证明 staging publish-back 可以绕开，但 overlap 不够

iteration-level 表示仍等完整 draft replay 结束后才提交 direct-active prefetch。它把 draft submit-after/call 从 `10.90 ms` 降到 `3.47 ms`，graph replay 不退化。但它不能把 H2D expert copy 提前到 draft replay 中间，所以隐藏能力有限。

### 3. naive 4-layer segment 的“边界”到底是什么

这里的边界不是模型语义上的新层，而是每个 segment replay 结束后的 CPU 调度点。48 层模型、segment size 4 时，每次 draft call 有 12 个边界。每个边界会触发这些操作：

```text
1. graph.replay() for this segment
2. acquire host metadata buffer slot
3. offload_async(): D2H copy token_count / selected_experts / routing_weights
4. enqueue metadata item to async worker
5. worker collect(): 等 D2H event ready 后把 CPU metadata 聚合出来
6. worker observe(): 更新 layer cache access 统计和 global candidate queue
7. worker submit_after(): 根据 frontier 选择 candidate 并提交 direct-active H2D
```

pre-opt 4-layer segment 的主要开销：

| Component | pre-opt segment 4 |
|---|---:|
| segment graph replays/call | 12 |
| segment metadata enqueue/call | 4.59 ms |
| draft host buffer wait/call | 43.36 ms |
| draft submit-after/call | 22.39 ms |
| draft forward/call | 87.33 ms |

host buffer wait 高的原因是：默认只有 3 个 host buffer slot，但每次 draft call 会快速产生 12 个 metadata item。async worker 来不及处理时，主线程想复用 host buffer 就必须等待，导致原本应该异步隐藏的 metadata offload 变成主路径等待。

优化后的变化：

- 自动把 draft segment host buffer pool 扩到 `segment_count + 2`，上限 64。
- 在 global queue ranking 前用 `max_layer_idx=frontier` 剪掉未来 layer，避免先排序再丢弃。

结果是 4-layer optimized 的 host buffer wait/call 从 `43.36 ms` 降到 `2.27 ms`，frontier skips 从 `738` 降到 `0`。这说明 metadata offload 本身可以不破坏 CUDA Graph；真正的问题是边界数太多时 CPU 调度点过密。

### 4. 为什么推荐 12-layer segment

segment size 12 时，48 层模型每次 draft call 只有 4 个边界：

```text
layer 0..11   boundary 1
layer 12..23  boundary 2
layer 24..35  boundary 3
layer 36..47  boundary 4
```

它的效果：

- segment metadata enqueue/call 从 4-layer optimized 的 `3.41 ms` 降到 `1.11 ms`。
- host buffer wait/call 从 `2.27 ms` 降到 `0.60 ms`。
- graph replay/call 是 `23.07 ms`，相比 baseline full draft graph 的 `20.62 ms`，graph 层额外开销约 `2.45 ms/call`。

也就是说，12-layer segment 更接近目标：让 CUDA Graph 继续高效 replay，同时给 prefetch 留出中间 overlap 机会。

### 5. “CPU queue submit”具体是什么，为什么能到 12.88 ms/call

文档里的 CPU queue submit 指 worker 处理完 metadata 后调用 `submit_draft_direct_active_prefetch(...)` 的过程。它不是单一一次 H2D copy，而是一串 CPU 操作：

```text
submit_draft_direct_active_prefetch()
  build inflight_keys
  global_queue.ranked_candidates()
    prune stale entries
    iterate queue entries
    filter inflight / cached / layer_idx > frontier
    recompute priority
    sort candidates
  prefetch_strategy.rank()
    filter TTL
    sort again by priority / layer / expert
  for candidate in ranked:
    check dispatch budget and adaptive budget
    lookup layer cache
    check expert already cached
    lookup CPU expert weights
    estimate H2D transfer time
    cache.snapshot()
    cache_strategy.select_victim_slot()
    reserve_active_slot_for_prefetch()
    begin_async_put_to_active()
    create PrefetchTicket and update counters
```

segment 12 的 `draft direct-active visible overhead/call = 12.88 ms`，但其中真正的 H2D 估算总量只有 `7.08 ms total`，平均约 `1.18 ms/call`。其余主要是 CPU 上的 queue scan/sort、cache snapshot/victim selection、reservation、ticket bookkeeping，以及 worker backlog 造成的重复调度。

为什么会这么高：

- 每个 draft call 还有 4 个 segment boundary，每个 boundary 都可能触发一次 submit-after。
- global queue 是全局候选集合，不是 per-layer frontier heap。即使加了 `max_layer_idx` 剪枝，仍要遍历和排序候选。
- prefetch budget 很小，实际 segment 12 只提交了 9 个 draft direct-active prefetch，但每个边界仍会付出候选选择成本。
- async queue depth max 是 6，stale submit skips 是 17，说明 worker 有积压。积压后，一些 metadata 到达 submit 阶段时已经过期，只能跳过 direct-active submit；这些 item 仍然消耗了 collect/observe 的 CPU 时间。

所以当前未达标的主要原因不是 CUDA Graph 不能承载 prefetch，也不是 PCIe copy 本身太慢，而是 CPU 侧候选队列和提交路径还不够轻。

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
