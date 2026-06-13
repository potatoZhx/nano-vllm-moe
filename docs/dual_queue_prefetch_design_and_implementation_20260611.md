# Dual-Queue Segment Prefetch 设计与实现

更新日期：2026-06-12

本文说明 speculative decoding 下 `dual_queue` expert prefetch 模式的设计、
实现、调用链、时序、数据流、评分与淘汰策略，以及测试和性能测试方法。

## 1. 目标与边界

`dual_queue` 的目标是在 draft segment graph 和 verify segment graph 之间建立
两类互相隔离的 expert 候选来源：

- `draft_predict`：当前 speculative round 内，draft 原始路由产生的短期预测。
- `ground_truth`：prefill 和 verify 真实路由产生的跨 round 历史。

它解决三个问题：

1. 第一轮 draft 尚无本轮 draft 路由时，使用历史真实路由完成 warm start。
2. 后续 draft 和 verify 使用本轮 draft 原始路由预测下一个 segment。
3. metadata offload、CPU 队列更新和 expert H2D 都不阻塞主计算路径。

本模式明确采用 best-effort 语义：

- 不等待 metadata worker drain 来获得“完整队列”。
- 不等待未完成的 expert H2D。
- 目标 segment 开始时，只发布已经完成 H2D 的 expert。
- 未在目标窗口内完成的传输标记为 expired，完成后释放 staging slot，不再写入 active cache。
- verify round 结束时仍未完成的本轮传输全部丢弃。

因此，”在 segment i 期间更新 segment i+1 cache”在实现上表示：

- segment i 的 metadata 到达 metadata worker 后，异步提交 H2D 直接写入
  active slot（direct_active 模式，slot 处于 pending 状态）；
- segment i+1 开始前查询完成事件；
- 已完成才原子式 commit 到 active cache（更新 LUT）；
- 未完成则保留 pending 状态，不等待。

## 2. 启用条件

核心配置位于 `nanovllm/config.py`：

```python
prefetch_runtime_kind = "dual_queue"
dual_queue_segment_size = 12
dual_queue_ground_truth_decay = 0.9
dual_queue_ground_truth_ttl_rounds = 64
dual_queue_ground_truth_count_weight = 0.1
dual_queue_budget_safety_ratio = 0.8
dual_queue_segment_time_ema_alpha = 0.2
```

`dual_queue_segment_size` 默认每个 segment 12 层。启用后该值同时覆盖：

```text
draft_prefetch_segment_size
verify_prefetch_segment_size
```

这样 draft 和 verify 的同一个 `segment_id` 始终表示相同 layer 范围。

配置初始化还会强制：

```text
prefetch_runtime_mode = draft_segment_indexed
draft_prefetch_frontier_granularity = segment
draft_cuda_graph_enabled = true
verify_cuda_graph = true
verify_cuda_graph_kt_hybrid = true
cpu_expert_backend = kt_direct
spec_verify_miss_policy = cpu
enforce_eager = false
prefetch_staging_slots_per_layer >= 1
num_hidden_layers > dual_queue_segment_size
```

最后一项保证至少存在两个 segment，否则没有跨 segment overlap 窗口。

## 3. 总体架构

```text
Qwen3 MoE routing on GPU
        |
        | record_layer()
        v
ModelRuntimeMetaRecorder
  draft:  [layer, expert] score_sum
  verify: [layer, expert] score_sum + activation_count + status
        |
        | segment-scoped async D2H
        v
metadata worker queue
        |
        | observe_draft / observe_verify / observe_prefill
        v
+---------------- DualQueuePrefetchRuntime ----------------+
|                                                         |
|  draft_predict_index       ground_truth_index            |
|  current round only        persistent history            |
|  raw score sum             normalized score/count        |
|  cleared after verify      decay + TTL                   |
|           \                    /                         |
|            \ ranked_for_range(segment)                   |
|             v                                           |
|        submit_segment_prefetch()                         |
|             |  [metadata worker thread]                  |
|             v                                           |
|  shared inflight tickets + transfer stream pool          |
|             |                                           |
|             v                                           |
|        direct_active: H2D -> active slot (deferred)      |
|             | ready event queried at segment boundary    |
|             v                                           |
|        commit_deferred_active_prefetch()                  |
|             v                                           |
|        active expert cache                               |
+---------------------------------------------------------+
```

这里有三种不同含义的“队列”：

1. 候选队列：`draft_predict_index` 和 `ground_truth_index`，两者完全分离。
2. metadata worker queue：异步处理 D2H metadata handle 的统一工作队列。
3. transfer inflight：两类候选提交后共享的 `inflight[(layer, expert)]`。

选择 `dual_queue` 后，不再更新或消费 legacy/predictive 的 global queue、
long-term index 或原有 `draft_segment_index`。父类对象仍为兼容性而存在，但
dual_queue 的 `observe_*` 和 segment hook 只操作新的两个 index。

## 4. 核心数据结构

### 4.1 DualQueueEntry

定义于 `nanovllm/expert/prefetcher.py`：

```python
@dataclass
class DualQueueEntry:
    layer_idx: int
    expert_idx: int
    score_sum: float
    activation_count: float
    last_seen_round: int
```

- draft entry 使用 `score_sum`，`activation_count` 固定为 0。
- ground-truth entry 同时维护归一化 score 和 activation count。
- `last_seen_round` 用于 decay、age 和 TTL。

### 4.2 DualQueueSegmentIndex

内部组织形式：

```text
entries_by_segment:
    segment_id ->
        (layer_idx, expert_idx) -> DualQueueEntry
```

segment 映射：

```text
segment_id = layer_idx // dual_queue_segment_size
```

选择 candidate 时必须同时满足：

- expert 的 layer 位于指定 `[layer_start, layer_end)`。
- expert 不在 active GPU cache。
- expert 不处于 cache pending 状态。
- `(layer, expert)` 不在统一 inflight 中。
- ground-truth entry 未超过 TTL。

最后按以下顺序排序：

```text
priority 降序，layer_idx 升序，expert_idx 升序
```

### 4.3 PrefetchTicket

两类 queue 的传输共享父类 `PrefetchTicket`：

```text
step_id
layer_idx, expert_idx
source
staging_slot_idx, staging_generation
submit_ts_ms
ready_event
segment_id
num_bytes
transfer_enqueue_ms
transfer_stream_idx
```

`segment_id` 是 ticket 的消费 deadline。只有对应 target segment 开始时，
该 ticket 才会被尝试发布。

`source` 取值：

```text
dual_queue_draft_predict
dual_queue_ground_truth
```

### 4.4 Round 状态

`DualQueuePrefetchRuntime` 维护：

```text
_round_id
_round_active
_draft_forward_index
_round_protected[layer] -> set(expert)
_expired_tickets -> set[(layer, expert)]
_segment_compute_ms[phase][segment]
_expert_transfer_ms
```

`_draft_forward_index == 0` 表示当前 round 的第一次 `draft_forward`。

## 5. Metadata 设计

### 5.1 Draft metadata

draft 在 GPU 上维护固定大小：

```text
score_sum:   float32[num_layers, num_experts]
token_count: int32[num_layers]
```

记录公式为：

```text
active_score_sum[layer, expert]
    = sum(routing_score[token, route])
      for every original route selecting that expert
```

例如 expert `x` 被两个 token 选中，原始 routing score 为 `x1`、`x2`：

```text
active_score_sum[x] = x1 + x2
```

MoE forward 中的记录位置在 draft reroute 之前：

```text
router softmax
  -> top-k original selected_experts/routing_weights
  -> runtime_meta_recorder.record_layer()
  -> draft reroute policy
  -> actual draft execution plan
```

因此 `draft_predict` 使用的是 draft 原始路由，而不是 reroute 后实际执行路由。

这种 `[layer, expert]` 固定向量有两个目的：

- CUDA graph capture 不依赖动态 expert 数量。
- D2H 大小只与 segment layer 数和 expert 数有关，不随 token/top-k 展开。

segment graph replay 后仅卸载当前 segment 的 layer slice。

### 5.2 Verify metadata

verify kt-hybrid graph 使用 histogram 格式：

```text
score_sum:        float32[num_layers, num_experts]
activation_count: int32[num_layers, num_experts]
expert_status:    int8[num_layers, num_experts]
token_count:      int32[num_layers]
```

`expert_status` 用于 verify hit/miss 统计；ground-truth 更新主要使用
`score_sum`、`activation_count` 和 `token_count`。

verify metadata 来自真实 verify top-k 路由，因此属于 `ground_truth`。
prefill metadata 同样调用 `observe_prefill()` 更新该 index。

### 5.3 异步 offload

调用链：

```text
segment graph replay
  -> ModelRuntimeMetaRecorder.offload_async(layer range)
  -> RuntimeMetaOffloadHandle
  -> ModelRunner._enqueue_prefetch_metadata()
  -> metadata worker
  -> ModelRunner._process_prefetch_metadata_item()
  -> collect(wait=False)
  -> prefetch_runtime.observe_draft/observe_verify/observe_prefill
```

offload 使用独立 metadata CUDA stream 和 pinned host buffer pool。

dual_queue 不在 draft/verify 阶段调用 metadata worker drain：

- `_flush_pending_prefetch_metadata(block=False)`。
- profile 收集也使用非阻塞 poll。
- host buffer 全部占用且无法扩容时，当前 metadata sample 直接 drop。

drop 计入：

```text
dual_queue_metadata_host_buffer_drop_count
```

这意味着候选 index 可能只包含部分已到达 metadata，提交决策仍立即基于当前
可见数据执行。

## 6. Queue 更新与分数

### 6.1 draft_predict

`draft_predict_index` 设置 `history=False`。

同一 round 内，同一 `(layer, expert)` 多次出现时：

```text
score_sum <- score_sum + current_active_score_sum
last_seen_round <- current_round
```

优先级：

```text
draft_priority = score_sum
```

不做 token_count 归一化，不做跨 round decay。原因是它表达当前 round 内
draft 对 verify 和后续 draft 的直接预测强度。

每次 draft metadata 到达后，该 metadata 中激活的 expert 同时加入：

```text
_round_protected[layer]
```

round 生命周期：

```text
第一次 draft_forward 开始 -> clear draft_predict，round_id += 1
后续 draft_forward        -> 累积更新
verify 结束               -> clear draft_predict 和 round protection
```

verify 结束后的迟到 draft metadata 会因 step 不再属于
`_active_draft_iteration_steps` 而被丢弃，并计入
`dual_queue_stale_draft_metadata_count`。

### 6.2 ground_truth

`ground_truth_index` 设置 `history=True`，跨 round 保留。

每个 metadata sample 先按 token 数归一化：

```text
sample_score = score_sum / token_count
sample_count = activation_count / token_count
```

若 entry 距离上次激活已跨过 `gap` 个 round，先对存量衰减：

```text
historical_score <- historical_score * decay^gap
historical_count <- historical_count * decay^gap
```

再合并当前 sample：

```text
historical_score <- historical_score + sample_score
historical_count <- historical_count + sample_count
last_seen_round  <- current_round
```

选择时再次根据当前 age 计算优先级：

```text
age = current_round - last_seen_round

ground_truth_priority
    = decay^age
      * (historical_score
         + count_weight * historical_count)
```

参数：

```text
decay        = dual_queue_ground_truth_decay
count_weight = dual_queue_ground_truth_count_weight
```

这里 score 表示路由概率质量，count 表示被选中的频率。两者结合可以避免：

- 仅按 count 选择大量低分路由；
- 仅按 score 忽略稳定高频 expert。

### 6.3 TTL

```text
age > dual_queue_ground_truth_ttl_rounds
```

时 entry 被惰性删除。删除发生在该 segment 下次执行 `ranked_for_range()` 时，
避免每个 round 全量扫描所有 layer/expert。

### 6.4 Cache residency 的处理

queue 更新时不删除已在 GPU cache 的 expert。过滤发生在提交时。

这样某 expert 当前在 cache 中时仍保留历史；之后被淘汰时，它可以重新成为
ground-truth candidate。

## 7. Prefetch 时序

以下使用用户视角的 1-based segment 编号。代码内部是 0-based。

设共有 `n` 个 segment。

### 7.1 当前 round 的第一次 draft_forward

关键路径（graph replay thread）和 metadata worker thread 并行执行：

```text
[graph replay thread]              [metadata worker thread]

segment 1 start:
    publish_direct_active_ready()
segment 1 compute
segment 1 end:
    offload_async(segment 1)  -->  observe_draft(segment 1)
                                   submit_deferred_draft_segment():
                                     submit GT(segment 2)
                                     submit GT(segment 3)  [segment 0 是首段]

segment 2 start:
    publish_direct_active_ready()
segment 2 compute
segment 2 end:
    offload_async(segment 2)  -->  observe_draft(segment 2)
                                   submit_deferred_draft_segment():
                                     submit GT(segment 4)

...

segment n-1 start:
    publish_direct_active_ready()
segment n-1 compute
segment n-1 end:
    offload_async(segment n-1) --> observe_draft(segment n-1)
                                   submit_deferred_draft_segment():
                                     submit DP(segment 1)

segment n start:
    publish_direct_active_ready()
segment n compute
segment n end:
    offload_async(segment n)  -->  observe_draft(segment n)
                                   submit_deferred_draft_segment():
                                     submit DP(segment 2)
```

DP 表示 `draft_predict`，GT 表示 `ground_truth`。

关键路径上只有 `publish_direct_active_ready()`（遍历 inflight tickets，对
已完成的执行 `commit_deferred_active_prefetch()`）。所有 prefetch submission
（含 victim 选择、active slot 预留、H2D 提交）都在 metadata worker thread
中异步执行。

`async offload/update DP` 与 submit 解耦。metadata worker 如果
尚未完成，submit 使用当前已经可见的 DP 内容；不会为了等待刚结束 segment 的
完整 metadata 而阻塞。

代码对应：

```text
on_draft_segment_start()           -> publish_direct_active_ready()  [graph replay thread]
submit_deferred_draft_segment()    -> submit_segment_prefetch()      [metadata worker thread]
```

第一轮的调度规则（由 `submit_deferred_draft_segment()` 实现）：

```text
segment 1 metadata arrives -> GT segment 2, GT segment 3
segment i metadata arrives -> GT segment i+2, i <= n-2
segment n-1 metadata arrives -> DP segment 1
segment n metadata arrives   -> DP segment 2
```

### 7.2 第 2 到第 k 次 draft_forward

每个 segment 开始时（graph replay thread）：

```text
publish_direct_active_ready()     [发布已完成 ticket]
```

metadata worker 在处理完该 segment metadata 后：

```text
submit_deferred_draft_segment():
    submit DP((segment + 1) % n)
```

即 `target = (segment + 1) % n`。

### 7.3 Verify

每个 verify segment 开始时（graph replay thread）：

```text
publish_direct_active_ready()     [发布已完成 ticket]
```

metadata worker 在处理完该 segment metadata 后：

```text
submit_deferred_verify_segment():
    segment 1~n-1 -> submit DP((segment + 1) % n)
    segment n     -> submit GT(segment 1)
```

所以：

```text
segment 1 metadata -> prefetch DP segment 2, 更新 GT segment 1
...
segment n-1 metadata -> prefetch DP segment n, 更新 GT segment n-1
segment n metadata -> prefetch GT segment 1, 更新 GT segment n
```

segment 1~n-1 使用 DP 的原因：这些 segment 在准备下一轮 draft 的后续
segment，draft_predict 保留了本轮 draft 原始路由的短期预测，用于预取与
当前 draft 路由相关性最高的 expert。

最后一个 segment n 使用 GT 的原因：该 segment 结束后 draft_predict 将被
清空，此时预取的目标是下一轮第一个 draft segment。ground_truth 保留了跨
round 的真实路由历史，是为全新 draft round cold start 提供候选的唯一来源。

每个 segment graph replay 结束后异步卸载真实路由 metadata，更新对应
ground-truth segment。

最后一个 verify segment 结束后：

1. 再尝试发布目标为 segment 1 的已完成 ticket。
2. 将仍未完成的 dual_queue ticket 标记为 round-end discard。
3. 清空 `draft_predict`。
4. 清空本 round 的 victim protection。
5. 结束 round。

没有 verify segment graph 的 fallback 路径会调用 `discard_verify_round()`，
避免上一条 graph 路径中提交的 ticket 在错误边界发布。

## 8. Transfer 与发布状态机

### 8.1 提交（direct_active 模式）

`submit_segment_prefetch()` 使用 direct_active 模式，在提交时即选择 victim
并预留 active slot：

1. 根据 source 选择 DP 或 GT index。
2. 读取当前 phase 的自动预算。
3. 预算再受 `prefetch_max_inflight - len(inflight)` 限制。
4. 仅在目标 layer range 内排序候选。
5. 过滤 cached、pending、inflight 和无 CPU 权重的 expert。
6. 调用 `_select_dual_queue_victim()` 选择 active slot。
7. 调用 `reserve_active_slot_for_prefetch_deferred()` 预留 active slot，
   将该 slot 标记为 pending（deferred 模式）。
8. 在 transfer stream 上以 `direct_active=True` 异步复制 `gate_up` 和
   `down` 直接写入 active slot buffer。
9. 创建带 target `segment_id` 的 `PrefetchTicket`，记录
   `active_slot_idx`、`active_generation`、`active_slot_prev_expert`。

两类来源共享 transfer stream pool，stream 使用 round-robin 分配。

与原 staging 模式的区别：不申请 staging slot、不做 staging→active D2D 拷贝、
不调用 `_finalize_publish()` 进行 publish stream 同步。

### 8.2 目标边界发布（direct_active）

`publish_segment_ready()` 委托给 `publish_direct_active_ready()`：

```text
遍历 inflight tickets:
  ready_event.query() == false -> 保留，继续等待
  ready_event.query() == true:
    non-deferred source -> commit_active_prefetch()
    deferred source     -> commit_deferred_active_prefetch()
    commit 成功 -> 从 inflight 移除，计入 published
    commit 失败 -> cancel_deferred_active_prefetch()（仅 deferred），计入 late
```

non-deferred source（`draft_direct_active`、`verify_layer_predict`、
`predictive_phase1`）使用 `reserve_active_slot_for_prefetch()` 在预留时
即清理旧 expert，必须与 `commit_active_prefetch()` 配对。

deferred source（`draft_segment_indexed`、`verify_segment`、
`dual_queue_draft_predict`、`dual_queue_ground_truth`）使用
`reserve_active_slot_for_prefetch_deferred()` 保留旧 expert 在 slot 中，
必须与 `commit_deferred_active_prefetch()` 配对，后者验证 `prev_expert`
和 `generation` 一致性后才原子更新 LUT。

两种 commit 操作都只需原子更新 cache LUT，无需额外数据拷贝或 stream 同步。

### 8.3 迟到传输

未在 target segment 到达前完成的 ticket 保留在 inflight 中，active slot 处于
pending 状态。后续 `publish_direct_active_ready()` 调用仍会尝试发布。

`_reap_expired_tickets()` 在 round 结束时清理残留 ticket：

- ticket 为 direct_active：调用 `cancel_deferred_active_prefetch()` 释放
  pending slot，恢复原 expert。
- ticket 为 staging（兼容路径）：调用 `cancel_staging_reservation()` 释放
  staging slot。

不会把迟到的数据发布到错误的 segment，也不会同步等待。

### 8.4 direct_active 模式的正确性保证

H2D 直接写入 active slot 而非 staging buffer，通过以下机制保证安全：

- `reserve_active_slot_for_prefetch_deferred()` 将 slot 标记为 pending，
  segment graph replay 中的 cache lookup 不会选中 pending slot 中的旧数据。
- `commit_deferred_active_prefetch()` 在 H2D 完成后原子更新 LUT，验证
  `prev_expert` 和 `generation` 一致性。
- cancel 时恢复 slot 到原状态（prev_expert），不丢失已有 expert。
- 由于 prefetch submission 从关键路径移至 metadata worker 线程（见第 19.2 节），
  victim 选择和 H2D 提交不阻塞 graph replay。

## 9. Victim 选择与保护

### 9.1 空 slot 优先

存在非 pending 空 active slot 时直接使用，不进行淘汰。

### 9.2 淘汰分数

无空 slot 时，对 resident expert 计算：

```text
(ground_truth_priority, last_access_step, slot_idx)
```

按升序选择 victim：

- ground-truth 分数越低越先淘汰；
- 分数相同时，越久未访问越先淘汰；
- 最后用 slot index 保证确定性。

即使当前 prefetch 来源是 DP，victim 的长期价值仍由 GT 分数衡量。

### 9.3 Round protection

DP metadata 中本 round 激活过的 expert 被加入 `_round_protected`。

当 source 为 `dual_queue_draft_predict` 时，victim 选择跳过这些 expert。
目的在于避免：

```text
本 round 已激活 expert
  -> 被 DP prefetch 淘汰
  -> 后续 draft/verify 又需要
  -> 再次传入
```

从而减少一轮内相同 expert 的反复换入换出。

GT prefetch 不受 round protection 限制。第一轮从 GT 取 candidate 时，旧的
DP 消费窗口已经结束，因此允许 GT 淘汰本轮曾被 DP 保护的 expert。

### 9.4 Boundary protection

同一次 `publish_segment_ready()` 中刚发布的 expert 会加入临时
`boundary_protected`。该保护对 DP 和 GT 都生效，防止一个较大的 boundary
budget 在同一批发布中立即淘汰前面刚发布的 expert。

若所有可选 slot 都受保护：

```text
dual_queue_all_slots_protected_count += 1
```

当前 staging reservation 被取消，不阻塞计算。

## 10. 自动预算标定

初始化 `LLMEngine` 时，dual_queue 自动执行一次短 warm generation：

```text
LLMEngine._calibrate_dual_queue_prefetch()
  -> ModelRunner.begin_dual_queue_calibration()
  -> generate(one token prompt, max_draft_tokens + 2)
  -> collect draft/verify per-segment CUDA event timings
  -> ModelRunner.finalize_dual_queue_calibration()
  -> calibrate one expert H2D
  -> compute phase budgets
```

标定期间 `submit_segment_prefetch()` 返回 0，只记录 segment 时间，不污染
实际 prefetch。

expert transfer 选择一个代表 expert，复制 `gate_up + down` 多次，并使用样本
最大值作为保守的：

```text
expert_transfer_ms
```

每个 phase/segment 的计算时间使用 EMA：

```text
t_ema = (1 - alpha) * t_old + alpha * t_new
```

预算使用该 phase 所有 segment 的最短时间：

```text
budget_phase
    = floor(
        safety_ratio
        * min(segment_compute_ms[phase])
        / expert_transfer_ms
      )
```

并限制为：

```text
0 <= budget_phase <= prefetch_max_inflight
```

最短 segment、最大 transfer 样本和 safety ratio 共同使预算偏保守。
运行中获得新的 segment timing 后会继续更新 EMA 并重算预算。

标定完成会自动打印：

```text
[dual_queue] calibrated
draft_prefetch_max_per_boundary=...
verify_prefetch_max_per_boundary=...
expert_transfer_ms=...
draft_min_segment_ms=...
verify_min_segment_ms=...
```

`draft_prefetch_min_per_boundary` 和 `verify_prefetch_min_per_boundary` 在本模式下
不强制抬高自动预算；性能 bench 将它们设为 0。

## 11. 主要调用链

### 11.1 初始化

```text
Config.__post_init__()
  -> 统一 draft/verify segment size
  -> 强制 graph + kt_direct + cpu miss policy

ModelRunner.__init__()
  -> runtime factory["dual_queue"]
  -> DualQueuePrefetchRuntime(...)

LLMEngine.__init__()
  -> _calibrate_dual_queue_prefetch()
```

### 11.2 Draft

```text
SpeculativeEngine
  -> ModelRunner.run_draft()
  -> begin_draft_iteration(step_id)
  -> ModelRunner.run()
  -> _replay_draft_segment_graph()
       for each segment:
         on_draft_segment_start()
           -> publish_direct_active_ready()       [轻量: 仅发布已完成 ticket]
         graph.replay()
         record segment timing (perf_counter after calibration)
         _enqueue_draft_segment_metadata()
           -> offload_async(layer range)
           -> submit_after_phase="after_draft_segment"
           -> metadata worker
             -> observe_draft()
             -> _submit_prefetch_after_metadata()
               -> submit_deferred_draft_segment()
                 -> submit_segment_prefetch()
```

关键变化：`on_draft_segment_start()` 仅调用 `publish_direct_active_ready()`
发布上一 segment 传输中已完成的 ticket。prefetch submission（含 victim 选择、
active slot 预留和 H2D 提交）全部由 metadata worker 线程中的
`submit_deferred_draft_segment()` 异步执行，不阻塞 graph replay。

`on_draft_segment_end()` 返回 0，不执行任何操作。

### 11.3 Verify

```text
SpeculativeEngine
  -> ModelRunner.run_verify()
  -> _run_verify_with_kt_hybrid_segment_graph()
       arm("verify_kt_hybrid")
       for each segment:
         on_verify_segment_start()
           -> publish_direct_active_ready()       [轻量: 仅发布已完成 ticket]
         graph.replay()
         record segment timing
         _enqueue_verify_segment_metadata()
           -> offload_async(layer range)
           -> submit_after_phase="after_verify_segment"
           -> metadata worker
             -> observe_verify()
             -> _submit_prefetch_after_metadata()
               -> submit_deferred_verify_segment()
                 -> submit_segment_prefetch()
       complete_verify_round()
```

verify 路径同理：`on_verify_segment_start()` 仅做发布，submit 移至
metadata worker 线程。

### 11.4 Queue selection 到传输（direct_active）

```text
submit_segment_prefetch()              [metadata worker thread]
  -> DualQueueSegmentIndex.ranked_for_range()
  -> _select_dual_queue_victim()
  -> LayerExpertCache.reserve_active_slot_for_prefetch_deferred()
  -> PrefetchRuntime._begin_prefetch_transfer(direct_active=True)
  -> LayerExpertCache.begin_async_put_to_active()
  -> PrefetchTicket(direct_active=True)
  -> inflight[(layer, expert)]
```

### 11.5 发布（direct_active）

```text
publish_direct_active_ready()          [graph replay thread, segment boundary]
  -> ready_event.query()
  -> LayerExpertCache.commit_deferred_active_prefetch()
  -> 移除 inflight entry
```

不再使用 `mark_staging_ready()`、`publish_ready_staging_to_active()`、
`_finalize_publish()` 和 publish stream 同步。

## 12. 与已有模式隔离

模式选择由 `prefetch_runtime_kind` 完成：

```text
legacy
predictive
dual_queue
```

隔离点：

- runtime factory 为 dual_queue 创建独立 `DualQueuePrefetchRuntime`。
- dual_queue 覆盖 `observe_prefill/observe_draft/observe_verify`。
- dual_queue 的 draft/verify segment hooks 不调用 predictive API。
- metadata item 使用 `submit_after_phase="after_draft_segment"` 或
  `"after_verify_segment"`，由 `_submit_prefetch_after_metadata()` 中的
  dual_queue 分支分发到 `submit_deferred_draft_segment()` 或
  `submit_deferred_verify_segment()`，不触发旧模式提交。
- dual_queue 不执行 predictive phase-1。
- source 名称独立，profile 可按来源拆分。
- 原有模式的默认配置和行为不变。

共享部分仅包括：

- `LayerExpertCache`。
- staging slot。
- transfer/publish stream 基础设施。
- metadata recorder 和 worker 框架。
- 通用 profile counter。

## 13. Profile 指标

核心 dual_queue 指标：

```text
model_dual_queue_draft_budget
model_dual_queue_verify_budget
model_dual_queue_expert_transfer_ms
model_dual_queue_draft_predict_size
model_dual_queue_ground_truth_size
model_dual_queue_target_miss_count
model_dual_queue_round_end_discard_count
model_dual_queue_expired_transfer_count
model_dual_queue_stale_draft_metadata_count
model_dual_queue_round_clear_count
model_dual_queue_all_slots_protected_count
model_dual_queue_metadata_host_buffer_drop_count
```

按 source 拆分：

```text
model_prefetch_submit_count_by_source
model_prefetch_completed_count_by_source
model_prefetch_published_count_by_source
model_prefetch_late_count_by_source
model_prefetch_submitted_bytes_by_source
model_prefetch_completed_bytes_by_source
model_prefetch_published_bytes_by_source
model_prefetch_late_bytes_by_source
```

关键解释：

- `target_miss_count`：目标 segment 开始时 H2D 尚未完成。
- `round_end_discard_count`：verify 结束时仍在 inflight 的本轮 ticket。
- `expired_transfer_count`：expired ticket 后续完成并已释放 staging 的数量。
- `late_count`：完成但未能发布，或超过 target deadline 的传输。
- `metadata_host_buffer_drop_count`：metadata host pool 满时被跳过的 sample。
- `all_slots_protected_count`：有 ready candidate，但没有可用 victim。

对于 best-effort 设计，`target_miss` 或 `round_end_discard` 非零不是正确性错误，
但说明预算、stream 数、staging 数或 PCIe 竞争需要调优。

`model_prefetch_async_drain_wait_ms` 在正常 dual_queue decode 中应接近 0。

## 14. 实现文件总结

### `nanovllm/config.py`

- 新增 dual_queue 参数。
- 默认 segment size 为 12。
- 统一 draft/verify segment graph 配置。
- 校验 graph、kt_direct、CPU miss fallback 和 staging 前提。

### `nanovllm/expert/runtime_meta.py`

- draft dual_queue 使用固定 `[layer, expert] score_sum`。
- verify 使用 count/score/status histogram。
- 支持按 segment layer range offload。
- host buffer pool 按 segment 数自动扩容。

### `nanovllm/models/qwen3_moe.py`

- draft 在 reroute 前记录原始 top-k 路由。
- verify kt-hybrid 记录真实 top-k 路由和 cache miss status。

### `nanovllm/expert/prefetcher.py`

- 新增 `DualQueueEntry`、`DualQueueSegmentIndex`。
- 新增 `DualQueuePrefetchRuntime`。
- 实现 DP/GT 更新、排序、TTL、预算、victim protection。
- 实现 target-boundary publish 和 late/round-end discard。
- 扩展 source 级 transfer/profile 指标。

### `nanovllm/engine/model_runner.py`

- runtime factory 接入 dual_queue。
- draft/verify segment graph 接入 start/end hook。
- metadata offload 与 prefetch submit 解耦。
- dual_queue host buffer 无可用 slot 时直接 drop。
- profile 收集不 drain metadata worker。
- verify graph fallback 时丢弃当前 round transfer。
- 记录 segment CUDA event 时间并更新预算。

### `nanovllm/engine/llm_engine.py`

- 初始化时执行自动标定。
- 回写并打印 draft/verify boundary budget。

### `benchmarks/scripts/spec_verify_expert_count_stats.py`

- 单用例 CLI 暴露 dual_queue 全部关键参数。
- case JSON 增加 dual_queue 预算、queue、deadline、source 和 byte 指标。

### `scripts/bench_dual_queue_prefetch.py`

- 新增独立性能测试入口。
- 不修改已有 `bench_verify_segment_graph.py` 的用例和输出。
- 默认执行 `dual_queue,predictive` 同配置对照。
- 支持 output length、cache ratio、draft K、segment size 和 repeat 矩阵。

## 15. 测试

### 15.1 单元测试覆盖

`tests/test_dual_queue_prefetch.py` 覆盖：

- draft score sum 累积。
- stale draft metadata 丢弃。
- GT token normalization、decay、priority 和 TTL。
- 仅选择指定 segment layer range。
- DP victim round protection。
- GT source 解除 round protection。
- ready target 发布。
- unready target expire，不修改 active cache。
- 同 boundary 刚发布 expert 保护。
- round-end discard。
- verify graph fallback discard。
- 保守预算计算。
- 固定大小 draft metadata。
- 第一轮 GT 到 DP 的调度时序。
- 后续 draft 和 verify 的环形 next-segment 调度。

`tests/test_config_predictive_prefetch.py` 覆盖：

- dual_queue 默认 12 层 segment。
- 用户 segment size 同步到 draft/verify。
- 强制 graph、kt_direct 和 miss policy 的配置行为。

`tests/test_spec_verify_expert_count_stats.py` 覆盖：

- 单用例 parser 的 dual_queue 参数。
- source 级 dual_queue summary。

`tests/test_dual_queue_bench.py` 覆盖：

- benchmark case matrix。
- DP/GT source 指标提取。
- dual_queue/predictive 同配置配对和 digest 对比。

### 15.2 运行测试

```bash
cd /home/linke/nano-vllm-moe

conda run -n nano_moe python -m pytest -q \
  tests/test_dual_queue_prefetch.py \
  tests/test_config_predictive_prefetch.py \
  tests/test_spec_verify_expert_count_stats.py \
  tests/test_dual_queue_bench.py
```

静态检查：

```bash
conda run -n nano_moe python -m py_compile \
  scripts/bench_dual_queue_prefetch.py \
  benchmarks/scripts/spec_verify_expert_count_stats.py
```

## 16. 性能测试

### 16.1 Benchmark 设计

新脚本：

```text
scripts/bench_dual_queue_prefetch.py
```

它复用：

```text
benchmarks/scripts/spec_verify_expert_count_stats.py --single-case
```

每个 case 启动独立子进程，保存独立 JSON 和 log，避免模型/runtime 状态跨 case
污染。

匹配维度：

```text
output_len
cache_ratio
max_draft_tokens
segment_size
repeat
```

默认 runtime：

```text
dual_queue,predictive
```

同一组维度下输出：

- output token throughput。
- draft/verify forward 平均时间。
- route hit rate。
- acceptance rate。
- 输出 digest 是否一致。
- dual_queue 自动预算。
- DP/GT submit、completed、published、late。
- publish ratio。
- target miss、round discard、metadata drop。

### 16.2 最小 smoke bench

```bash
cd /home/linke/nano-vllm-moe

conda run -n nano_moe python scripts/bench_dual_queue_prefetch.py \
  --output-dir results/dual_queue_smoke \
  --model-path /data1/models/Qwen3-30B-A3B \
  --profile-artifact results/reroute_impl_20260531/offline_profile_20260531_203257.safetensors \
  --runtime-kinds dual_queue \
  --output-lens 32 \
  --cache-ratios 0.25 \
  --max-draft-tokens-values 4 \
  --segment-sizes 12 \
  --repeats 1 \
  --gpu-memory-utilization 0.99 \
  --kt-num-threads 32
```

### 16.3 dual_queue 与 predictive 对照

```bash
conda run -n nano_moe 
rm -rf results/dual_queue_vs_predictive_mixed
CUDA_VISIBLE_DEVICES=1 python scripts/bench_dual_queue_prefetch.py \
  --output-dir results/dual_queue_vs_predictive_mixed \
  --model-path /data1/models/Qwen3-30B-A3B \
  --profile-artifact results/reroute_impl_20260531/offline_profile_20260531_203257.safetensors \
  --runtime-kinds dual_queue,predictive \
  --secondary-index-weight 1 \
  --output-lens 128,512,1024,2048,4096,8092 \
  --cache-ratios 0.25,0.3125 \
  --max-draft-tokens-values 3,8 \
  --segment-sizes 12 \
  --repeats 1 \
  --gpu-memory-utilization 0.99 \
  --kt-num-threads 32
```

测试其他 segment size：

```bash
--segment-sizes 6,8,12
```

### 16.4 调优参数

```text
--ground-truth-decay
--ground-truth-ttl-rounds
--ground-truth-count-weight
--budget-safety-ratio
--segment-time-ema-alpha
--prefetch-max-inflight
--prefetch-transfer-stream-count
--prefetch-staging-slots-per-layer
--prefetch-metadata-host-buffer-pool-size
--draft-prefetch-max-per-boundary
--verify-prefetch-max-per-boundary
```

最后两个参数是标定无 timing sample 时的 fallback/cap 初值；完成标定后由自动预算
覆盖。

### 16.5 输出文件

```text
<output-dir>/
  dual_queue_prompt.txt
  <case-name>.json
  <case-name>.log
  summary.json
  summary.md
```

可使用：

```text
--report-doc docs/res/dual_queue_benchmark.md
```

额外写一份 Markdown 汇总。

`--skip-existing true` 会复用已完成 case JSON；修改代码或参数后需要使用新的
output directory，或者显式设置：

```text
--skip-existing false
```

## 17. 结果判读

优先检查：

1. `digest_match`：同 seed、同配置下应一致。
2. `verify_segment_graph_replays`：确认实际走 segment graph。
3. 自动打印的 draft/verify budget 是否大于 0 且明显小于 inflight 上限。
4. `dual_publish_ratio = published / submitted`。
5. `target_miss_count` 和 `round_end_discard_count`。
6. route hit rate、draft/verify 时间和最终 tok/s。

典型调优方向：

```text
target miss 高:
  降低 budget_safety_ratio
  降低 max_per_boundary fallback
  检查 PCIe 与 CPU pinned memory
  避免 transfer stream 过多造成带宽竞争

metadata drop 高:
  增大 metadata host buffer pool
  检查 metadata worker CPU 调度

all slots protected 高:
  增加 active cache slots
  减小 DP boundary budget
  检查本 round 激活集合是否接近 cache 容量

published 高但 hit rate 无提升:
  调整 GT decay/count weight/TTL
  检查 DP 原始路由与 verify 路由相关性
  检查 victim churn
```

不能仅追求 submit 数量。对该设计更重要的是：

```text
deadline 内 published
  -> 在目标真实路由中命中
  -> 降低 CPU miss 路由
  -> 不增加可见等待
  -> 提升端到端 tok/s
```

## 18. 当前限制

- 当前只支持 draft 和 verify 都使用一致 segment size。
- verify dual_queue 路径依赖 kt-direct hybrid segment graph。
- metadata 和 transfer 都是 best-effort，系统负载高时允许丢 sample/transfer。
- GT history 按 round 而非 wall-clock ageing。
- 自动预算以单 expert 代表传输和最短 segment 为依据，未显式建模多 stream
  并发后的 PCIe 饱和；因此仍需通过 benchmark 校验。
- `prefetch_max_inflight` 是全局上限，DP 和 GT 不预留独立配额。
- direct_active 模式下 active slot 被预留为 pending 后不可被其他 prefetch
  选作 victim，某些 layer 高并发 prefetch 时可能受 slot 数限制。

这些限制都是性能退化路径，不改变 verify 的数值语义：未及时完成的 prefetch
不会被强行发布，miss expert 仍由 `spec_verify_miss_policy=cpu` 执行。

## 19. 性能回归分析与优化（2026-06-12）

### 19.0 问题描述

初始实现在 benchmark 中与 `predictive` 模式对比出现全面倒退：

| 指标 | dual_queue | predictive | 差异 |
|---|---|---|---|
| tok/s | 18.9 | 23.0 | -18% |
| draft_forward_ms | 24.8 | 19.0 | +31% |
| verify_forward_ms | 85.3 | 72.4 | +18% |
| hit_rate | 0.818 | 0.865 | -5.4pp |
| accept_rate | 0.822 | 0.867 | -4.5pp |
| budget (D/V) | 6/16 | adaptive ~16 | draft 预算严重不足 |

benchmark 使用配置：

```text
model: Qwen3-30B-A3B
output_len: 512, cache_ratio: 0.25, max_draft_tokens: 4
segment_size: 12, budget_safety_ratio: 0.6
```

### 19.1 根因 A：关键路径开销（~6ms/draft segment）

初始实现中 `on_draft_segment_start()` 和 `on_draft_segment_end()` 在 graph
replay 循环内、`_prefetch_runtime_lock` 下同步执行以下操作：

**A.1 同步 segment hooks**

每次 segment boundary 调用：

- `publish_segment_ready()`（`prefetcher.py:3145-3209`）：遍历所有 inflight
  tickets，逐个调用 `ticket.ready_event.query()`（CPU-GPU 同步），选择
  victim，通过 `publish_ready_staging_to_active()` 执行 staging→active D2D。
- `submit_segment_prefetch()`（`prefetcher.py:3020-3110`）：调用
  `ranked_for_range()` 对所有候选按 exponential decay 排序，预留 staging
  slot，发起 H2D 传输。
- `_finalize_publish()`（`prefetcher.py:2026-2029`）：调用
  `torch.cuda.current_stream().wait_stream(self.publish_stream)` — **每个
  已发布 expert 都触发一次 CUDA stream 同步**。

`predictive` 在 draft replay 循环中跳过全部上述操作。它使用
`_enqueue_draft_segment_metadata()` 将 segment metadata 异步传递给
metadata worker thread，由 `submit_draft_segment_indexed_prefetch()` 在
关键路径之外执行。

**A.2 Staging buffer 架构 vs direct_active**

初始 dual_queue 使用 staging：

```text
H2D -> staging buffer -> publish_segment_ready() 检查 event
  -> staging→active D2D -> _finalize_publish() stream 同步
```

predictive 使用 `direct_active=True`：

```text
H2D -> active slot (deferred) -> commit 原子更新 LUT
```

staging 路径每个 expert 多一次 D2D 拷贝和一次 publish stream 同步。

**A.3 per-segment CUDA timing instrumentation**

`_start_dual_queue_segment_timing()`（`model_runner.py:1555-1600`）在每个
segment 创建 2 个 CUDA event，`_poll_dual_queue_segment_timings(block=False)`
查询它们。即使标定已完成，这些操作仍在每个 segment 执行（~0.5-1ms 开销）。

**A.4 每 segment 重算预算**

`record_segment_compute_ms()` 在标定完成后每个 segment 都调用
`_recompute_prefetch_budgets()`，涉及遍历所有 segment 计时并重算预算。

### 19.2 优化方案：关键路径开销消除

**A.1 segment hooks 移至 metadata worker**

移除 graph replay 循环中的同步 `submit_segment_prefetch()` 调用。

- `on_draft_segment_start()`：仅调用 `publish_direct_active_ready()`
  （轻量：遍历 inflight tickets 执行 commit）。
- `on_draft_segment_end()`：返回 0，不执行任何操作。
- `on_verify_segment_start()`：仅调用 `publish_direct_active_ready()`。

prefetch submission 由以下路径异步执行：

```text
_enqueue_draft_segment_metadata()
  submit_after_phase="after_draft_segment"
  -> metadata worker
    -> _submit_prefetch_after_metadata()
      -> submit_deferred_draft_segment()
        -> submit_segment_prefetch()
```

verify 路径同理使用 `submit_after_phase="after_verify_segment"` 和
`submit_deferred_verify_segment()`。

新增 `_segment_id_for_frontier()` helper 根据 `frontier_layer_idx` 反查
segment_id，因为 metadata worker 不持有 segment_id，只知道 frontier layer。

**A.2 切换至 direct_active 模式**

`submit_segment_prefetch()` 改为在提交时执行 victim 选择（非发布时）：

1. 调用 `_select_dual_queue_victim()` 获取 active slot。
2. 调用 `reserve_active_slot_for_prefetch_deferred()` 将 slot 标记为 pending。
3. 以 `direct_active=True` 发起 H2D，直接写入 active slot buffer。
4. ticket 记录 `active_slot_idx`、`active_generation`、`active_slot_prev_expert`。

发布路径使用 `publish_direct_active_ready()`：

- `ready_event.query()` 成功 → `commit_deferred_active_prefetch()` 原子更新 LUT。
- 不需要 staging slot、D2D 拷贝、publish stream、`_finalize_publish()`。

`_reap_expired_tickets()` 相应更新：direct_active ticket 使用
`cancel_deferred_active_prefetch()` 取消。

**A.3 timing instrumentation（保留 CUDA event）**

`_start_dual_queue_segment_timing()` 始终使用 CUDA event，不改用
`perf_counter()`。原因是 `graph.replay()` 异步返回，`perf_counter()` 只能
测量 CPU enqueue 时间（~0.01ms），无法反映实际 GPU 计算时间（~4ms）。budget
公式依赖准确的 segment 计算时间，错误的低值会导致 budget 为 0。CUDA event
的创建/record/query 开销约 40μs/segment，相比 graph replay 本身可忽略。

**A.4 节流预算重算**

`record_segment_compute_ms()`：标定完成后每 8 个 segment 才调用一次
`_recompute_prefetch_budgets()`，而非每个 segment。通过
`_budget_recompute_interval` 和 `_budget_recompute_counter` 控制。

### 19.3 Python 微优化

**C.1 缓存配置查找**

`DualQueueSegmentIndex.__init__()` 将 `_decay`、`_count_weight`、`_ttl` 缓存
为实例属性，替代 `_priority()` 和 `ranked_for_range()` 中的 `getattr()` 调用。

**C.2 预计算 decay 幂表**

```python
self._decay_table = [self._decay ** i for i in range(self._ttl + 2)]
```

`_priority()` 使用 `self._decay_table[min(age, self._ttl + 1)]` 替代
`decay ** age` 实时计算。

**C.3 消除集合构造**

- `submit_segment_prefetch()` 直接传递 `self.inflight` dict 给
  `ranked_for_range()` 的 `inflight_keys` 参数（`dict` 支持 `in` 操作），
  替代 `set(self.inflight.keys())` 的复制开销。
- `_select_dual_queue_victim()` 使用 `frozenset()` 作为空集合哨兵，替代
  `set()` 的重复构造。

**C.4 EMA alpha 缓存**

`DualQueuePrefetchRuntime.__init__()` 将 `_ema_alpha` 缓存为实例属性，替代
`record_segment_compute_ms()` 中每次调用时的 `getattr()` 查找。

## 20. 实现变更总结（2026-06-12）

### 20.1 `nanovllm/expert/prefetcher.py`

变更概要：

- `DualQueueSegmentIndex.__init__()`：新增 `_decay`、`_count_weight`、`_ttl`、
  `_decay_table` 缓存属性。
- `DualQueueSegmentIndex._priority()`：使用 decay table lookup 和缓存属性替代
  `getattr()` 和实时幂运算。
- `DualQueueSegmentIndex.ranked_for_range()`：`inflight_keys` 参数类型扩展为
  `set | dict`，使用缓存 `_ttl`。
- `DualQueueSegmentIndex.update()`：使用缓存 `self._decay`。
- `DualQueuePrefetchRuntime.__init__()`：新增 `_ema_alpha`、
  `_budget_recompute_interval`、`_budget_recompute_counter`。
- `record_segment_compute_ms()`：使用缓存 `_ema_alpha`，节流预算重算至每 8 次。
- `_select_dual_queue_victim()`：使用 `frozenset()` 哨兵。
- `submit_segment_prefetch()`：改为 direct_active 模式 — 提交时选择 victim、
  `reserve_active_slot_for_prefetch_deferred()`、`direct_active=True`。不再
  申请 staging slot。profile counter 从 `staging_prefetch_submit_count` 改为
  `direct_active_prefetch_submit_count`。
- `_reap_expired_tickets()`：处理 direct_active ticket 使用
  `cancel_deferred_active_prefetch()`，兼容 staging ticket。
- `publish_segment_ready()`：委托给 `publish_direct_active_ready()`。
- `on_draft_segment_start()`：简化为仅调用 `publish_direct_active_ready()`。
- `on_draft_segment_end()`：返回 0。
- `on_verify_segment_start()`：简化为仅调用 `publish_direct_active_ready()`。
- 新增 `submit_deferred_draft_segment()`：metadata worker 调用，包含完整的
  first-draft GT/DP 调度和后续 draft DP 调度逻辑。
- 新增 `submit_deferred_verify_segment()`：metadata worker 调用，执行 verify
  阶段的 next-segment 调度。segment 1~n-1 使用 DP，最后一个 segment n
  使用 GT 预取下一轮第一个 draft segment。
- 新增 `_segment_id_for_frontier()`：根据 frontier_layer_idx 反查 segment_id。

### 20.2 `nanovllm/engine/model_runner.py`

变更概要：

- `_submit_prefetch_after_metadata()`：新增 dual_queue 分支，在 metadata
  worker thread 中根据 mode 调用 `submit_deferred_draft_segment()` 或
  `submit_deferred_verify_segment()`。
- `_enqueue_draft_segment_metadata()`：`submit_after_phase` 从 `None`（dual_queue
  时）改为无条件使用 `"after_draft_segment"`。
- `_replay_draft_segment_graph()`：移除 `on_draft_segment_end()` 调用。缓存
  `dual_queue = self._dual_queue_prefetch_enabled()` 避免重复方法调用。
- `_start_dual_queue_segment_timing()`：始终使用 CUDA event（`perf_counter()`
  无法测量异步 graph replay 的 GPU 时间，会导致 budget=0）。
- `_enqueue_verify_segment_metadata()`：dual_queue 时设置
  `submit_after_phase="after_verify_segment"` 和 `frontier_layer_idx`。

### 20.3 `tests/test_dual_queue_prefetch.py`

变更概要：

- `test_unready_target_is_discarded_without_cache_mutation` →
  `test_unready_target_remains_inflight_without_cache_mutation`：
  改为验证 direct_active 语义 — unready ticket 保留在 inflight（不移至
  expired），ready 后调用 `publish_direct_active_ready()` 完成发布。
- `test_same_boundary_does_not_evict_just_published_expert`：
  断言从 `submitted==2` 改为 `submitted==1`（direct_active 提交时选择 victim
  并标记 pending，第二个 candidate 无可用 slot）。发布使用
  `publish_direct_active_ready()`。
- `test_first_draft_uses_ground_truth_then_draft_predict`：
  调用 `submit_deferred_draft_segment()` 替代 `on_draft_segment_start()` /
  `on_draft_segment_end()`，传递 `frontier_layer_idx`。
- `test_later_draft_and_verify_prefetch_next_segment`：
  调用 `submit_deferred_draft_segment()` / `submit_deferred_verify_segment()`
  替代直接调用 segment hooks。验证最后一个 verify segment 使用 GT。
- 新增 `test_verify_non_last_segment_uses_draft_predict`：
  验证 verify segment 1~n-1 使用 DP，最后一个 segment 使用 GT。

## 21. 测试验证

### 21.1 单元测试

```bash
cd /home/linke/nano-vllm-moe

conda run -n nano_moe python -m pytest -q \
  tests/test_dual_queue_prefetch.py \
  tests/test_config_predictive_prefetch.py \
  tests/test_dual_queue_bench.py
```

当前通过 28 个测试，覆盖：

- direct_active 模式下的提交、发布、取消语义。
- unready ticket 保留在 inflight 而非移至 expired。
- submit 时 victim 选择导致 pending slot 不可被后续 candidate 使用。
- deferred draft/verify segment 调度时序。
- round-end discard 对 direct_active ticket 的清理。
- 所有原有覆盖点（decay、TTL、protection、budget 等）。

### 21.2 静态编译检查

```bash
conda run -n nano_moe python -m py_compile nanovllm/expert/prefetcher.py
conda run -n nano_moe python -m py_compile nanovllm/engine/model_runner.py
```

### 21.3 完整测试矩阵

```bash
conda run -n nano_moe python -m pytest -q \
  tests/test_dual_queue_prefetch.py \
  tests/test_config_predictive_prefetch.py \
  tests/test_spec_verify_expert_count_stats.py \
  tests/test_dual_queue_bench.py
```

## 22. 优化后性能测试

### 22.1 Smoke test

```bash
cd /home/linke/nano-vllm-moe

conda run -n nano_moe python scripts/bench_dual_queue_prefetch.py \
  --output-dir results/dual_queue_opt_smoke \
  --model-path /data1/models/Qwen3-30B-A3B \
  --profile-artifact results/reroute_impl_20260531/offline_profile_20260531_203257.safetensors \
  --runtime-kinds dual_queue \
  --output-lens 32 \
  --cache-ratios 0.25 \
  --max-draft-tokens-values 4 \
  --segment-sizes 12 \
  --repeats 1 \
  --gpu-memory-utilization 0.99 \
  --kt-num-threads 32
```

### 22.2 优化前后 A/B 对照

```bash
conda run -n nano_moe python scripts/bench_dual_queue_prefetch.py \
  --output-dir results/dual_queue_opt_vs_predictive \
  --model-path /data1/models/Qwen3-30B-A3B \
  --profile-artifact results/reroute_impl_20260531/offline_profile_20260531_203257.safetensors \
  --runtime-kinds dual_queue,predictive \
  --output-lens 128,512 \
  --cache-ratios 0.25,0.3125,0.50 \
  --max-draft-tokens-values 4,8 \
  --segment-sizes 12 \
  --repeats 3 \
  --gpu-memory-utilization 0.99 \
  --kt-num-threads 32
```

### 22.3 预期改善

优化预期效果（需 benchmark 验证）：

| 优化项 | 预期影响 |
|---|---|
| segment hooks 移至 metadata worker | draft_forward_ms 下降 ~5-6ms（消除同步 submit + publish） |
| 切换 direct_active | 消除 staging D2D + publish stream 同步开销 |
| 节流预算重算（每 8 segment） | 减少 per-segment CPU 开销 |
| Python 微优化 | 减少 `_priority()` / `ranked_for_range()` 热路径延迟 |

注意：以上优化仅消除关键路径开销（Phase A + C）。hit_rate 改善（Phase B：
合并双 index 查询、添加 phase-1 cold start、优化 victim 策略、修正 draft
budget 不足）尚未实施，将根据 benchmark 结果决定是否执行。

### 22.4 关键验证指标

优化后重点检查：

1. `draft_forward_ms`：是否从 ~24.8ms 下降至接近 predictive 的 ~19ms。
2. `verify_forward_ms`：是否从 ~85.3ms 下降至接近 predictive 的 ~72ms。
3. `tok/s`：端到端吞吐是否回升。
4. `hit_rate`：在相同 budget 下命中率是否因传输及时性改善而提升。
5. `digest_match`：同 seed 同配置下输出应一致。
6. `dual_queue_draft_budget` / `dual_queue_verify_budget`：自动标定预算是否合理。
7. `direct_active_prefetch_submit_count`：确认使用 direct_active 路径（不再有
   `staging_prefetch_submit_count`）。
