# 29. Publication batching 只读机会评估

日期：2026-08-21

## 一句话总结

LUT fusion 落地后，跨层 publication 再批处理的理想 launch 上界不到 0.1% TPOT，且会
改变 expert 可见时刻，因此从下一运行时候选降级为 CPUInfer native 分解之后的方向。

## 证据

只读分析现有 b1 lifecycle artifact：

```text
results/analysis_phase1_recent_t32_b1_mmlu0_20260819/
```

其中有 2531 次 publication，来源为 verify/draft/phase1 = 1560/711/260。按 ticket 的
submit step 分组：

- 571 组，平均 4.43 experts/组，最大 7；
- 只有 12 组是 singleton，其余 2519 次 publication 属于多 expert submit step；
- 但按 `(submit step, layer)` 分组后有 2471 组，其中 2411 组只有一个 expert，只有
  60 组能在同一 layer 直接合并两个 commit；
- `direct_active_prefetch_ready_scan_count=2531` 与 publish count 完全相同，instrumented
  请求没有表现出同一 ticket 被反复 event-query 的扫描浪费。

`step_id` 是 ticket 的 submit step，不是 publication 调用的 wall-clock batch id，因此
571 只能作为最乐观分组上界，不能声称当前 runtime 真能形成 571 个 ready batch。

## 收益上界

已保留 fused commit 的隔离中位数是 0.013565 ms/launch。即使忽略所有语义和实现成本，
把 2531 个 publication kernel 理想压到 571 个：

```text
(2531 - 571) * 0.013565 ms = 26.59 ms/request
26.59 / 511 = 0.052 ms/output-token
```

相对约 54.2 ms/token 不到 0.1%。实际收益还要扣除 batch 索引准备、跨层 pointer/storage
组织和新 kernel 成本。由于 2411/2531 publication 在 `(submit step, layer)` 上是 singleton，
简单的 per-layer batching 基本无效；真正合并必须把 48 层 LUT 改为共享二维 storage，或
传递跨层 pointer 数组。

共享 H2D completion event 也不能免费获得：一个 transfer stream 上较早完成的 expert 会
被迫等待组内最后一个 copy，可能增加 CPU fallback 和 exposed tail。此前 async boundary、
metadata fast-path、未正确预热的 LUT fusion 都证明微小的 pacing 变化能造成远大于 launch
节省的回退。

## 决定与新排序

本次不产生运行时候选，也不运行新的 TPOT：

1. 先做 exact current CPUInfer 的 queue/compute/output-copy/exposed-sync 分解；
2. 再做 source/rank-aware admission，目标是减少真实 CPU tail 和无效 transfer；
3. publication/event/H2D batching 仅在 native profile 证明控制面重新成为显著占比，或能
   保持每个 expert 独立可见时刻时再启动。

这不是永久否决 batching，而是用 LUT fusion 后的新成本上界纠正优先级。
