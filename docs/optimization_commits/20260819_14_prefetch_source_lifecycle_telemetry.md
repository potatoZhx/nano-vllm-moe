# Prefetch source lifecycle telemetry

日期：2026-08-19

## 一句话总结

在 profile 模式下按 prefetch source 记录 publication residency 的首次消费、重复消费、
换出、未消费换出、驻留步数和替换来源，为 admission/budget 优化提供可归因证据；
production/profile-off 路径不保留或扫描这些状态。

## 背景

既有 profile 已能按 source 统计 submit/completed/published/bytes，但
`record_verify_consumed()` 只有总消费和少数硬编码 source counter。尤其缺少
`predictive_phase1`、`verify_segment` 的统一消费与换出数据，无法回答：

- 哪类预取在被淘汰前真正命中过；
- 哪类预取完成 H2D 后从未被使用；
- 一个 publication 等待多少 step 才首次消费；
- 哪个 replacement source 驱逐了哪个 resident source。

此外，非 deferred direct-active reservation 在 submit 时已经移除 victim，而 staging
与 deferred 路径在成功 publish 时才移除 victim；只在一个位置观察 cache 会混淆两种
生命周期。

## 实现

`PrefetchRuntime` 新增 profile-only `PrefetchResidency`：

- publish 时记录 source、publish step、bytes 和单调 sequence；
- non-deferred victim 在 submit 时闭合 residency；
- staging/deferred victim 在成功 publish 时闭合 residency；
- external cache mutation 在消费观察后清理并标成 `external` replacement；
- verify metadata 优先使用 graph 执行时捕获的 `expert_status` 判断真实 GPU-cache hit，
  缺失时才退回处理时的 cache 状态；
- segment metadata 非 deferred 模式下，profile 会让每个 segment 都记录消费；默认 deferred
  metadata 本来就是一次覆盖全部层的 handle；
- 同 step 已闭合 residency 优先于当前 active residency，用于处理异步 metadata 晚于后续
  segment replacement 到达的情况，并单独记录歧义数。

新增 JSON 字段包括：

- `prefetch_consumed_count_by_source`；
- `prefetch_first_consumed_count_by_source` / `..._bytes_by_source`；
- `prefetch_first_consume_latency_steps_by_source` / `..._max_by_source`；
- `prefetch_evicted_count_by_source` / `..._bytes_by_source`；
- `prefetch_evicted_after_consume_count_by_source`；
- `prefetch_evicted_without_consume_count_by_source` / `..._bytes_by_source`；
- `prefetch_residence_steps_by_source`；
- `prefetch_resident_count_by_source` / `..._consumed_count_by_source`；
- `prefetch_eviction_count_by_source_pair`；
- tracked/closed/active residency 数及 ambiguous/unattributed/republish 诊断数。

`transfer_aware_profile` 额外在 lifecycle event list 写入 `first_consume` 与 `evict` 事件。

## 不变量与开销边界

- 不修改 candidate index、priority、victim、budget、admission、H2D 或 publish 决策；
- 不修改 dynamic K、acceptance predictor 或 sampling；
- 仅 `engine_profile`、`spec_profile` 或 `transfer_aware_profile` 开启时创建 residency；
- profile-off 仍保留旧的 `_recent_published` 行为，但 residency 容器保持空，输出中不出现
  新字段；
- profile reset 会清空 residency 区间，warmup 前的 cache resident 不混入测量区间的
  published/consumed denominator。

这是分析基础设施提交，不宣称 TPOT 正收益，也没有运行端到端优化验证。

## 测试

执行：

```text
PYTHONPATH=. /home/edge/.conda/envs/nano_moe/bin/python -m pytest -q \
  tests/test_prefetch_runtime.py \
  tests/test_predictive_prefetch.py \
  tests/test_dual_queue_prefetch.py
```

结果：相关 prefetch runtime 测试通过。另单独验证：

- profile-on 的 prefill publication 能正确记为首次消费；
- 被下一 publication 驱逐后记为 consumed-before-evict；
- 外部替换未消费 resident 后记为 evicted-without-consume；
- profile-off 不产生 residency 或新 profile 字段；
- verify 非最后 segment 只在 lifecycle profile 开启时记录消费。

`tests/test_verify_segment_graph.py` 全文件另有一个既有失败：测试把
`runtime_meta_recorder=None`，但仍期待 `_enqueue_verify_segment_metadata()` 调用；失败路径
与本提交 diff 无关。相关 `TestEnqueueVerifySegmentMetadata` 用例单独通过。

## 下一步

用一次 analysis-only 数据集请求采集新字段，按 source 计算：

```text
first-consumed publications / published publications
evicted-without-consume bytes / published bytes
first-consume latency / residence steps
source A -> source B replacement matrix
```

只有这些指标证明某个 source 稳定低价值后，才实现独立、可回退的 admission/budget 优化。
