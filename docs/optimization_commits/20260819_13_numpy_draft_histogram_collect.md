# 2026-08-19：降低 draft histogram metadata 主机解析开销

## 一句话总结

保持 draft metadata 的 expert id、routing score 和 activation count 完全不变，使用 NumPy 直接解析已在 CPU 上的稀疏 histogram row，减少每层 PyTorch dispatcher 调用。

## 问题

active14 的 predictive prefetch 使用 16-layer draft segment。GPU 把每层 128 个
expert 的 `activation_count` 和 `score_sum` 异步复制到 pinned host buffer 后，
`ModelRuntimeMetaRecorder.collect()` 原来对每层依次执行：

```text
torch.nonzero -> index_select(count) -> to(int64)
              -> index_select(score) -> to(float32)
              -> to(expert_ids, int64)
```

这些 tensor 已经位于 CPU，行宽仅 128；此时主要成本不是数据量，而是对每个
16-layer metadata item 重复进入 PyTorch dispatcher。

现有代表 profile：

- `metadata_offload_draft_count=900`；
- `metadata_offload_draft_bytes=14,803,200`，即每个 16-layer item
  `16,448 bytes`；
- `metadata_offload_collect_ms=5,681.230 ms`。该 profile 值包含后台线程调度和
  重叠影响，不能直接当作同步 wall-clock 相加，但说明解析路径会被高频调用。

## 修改

仅对标准 `metadata_format="histogram"` 使用新的 CPU fast path：

1. 在 layer loop 外取得 token-count、activation-count 和 score-sum 的 NumPy
   零拷贝视图；
2. 每层用 `np.flatnonzero` 找到激活 expert；
3. 只复制非零 count/score，并用 `torch.from_numpy` 构造下游现有接口需要的
   CPU tensor。

以下路径没有改变：

- `histogram_kt_hybrid`；
- transfer-aware raw metadata；
- GPU 端 histogram record 和 D2H offload；
- `SegmentCandidateIndex` 的 priority、排序和 cache/prefetch 决策；
- 动态 K1/K2 draft-length 策略及其最优保留配置。

`np.flatnonzero` 与原 `torch.nonzero` 都按 expert index 升序返回；count 和 score
的 dtype 仍分别为 `torch.int64`、`torch.float32`。

## 分析实验

遵照“停止优化验证实验”的要求，没有运行端到端数据集验证。本实验直接构造真实
`ModelRuntimeMetaRecorder`：48 层、128 experts、top-k 8；每次 handle 收集一个
16-layer segment；旧逻辑和新逻辑各重复 1,000 次。

| 解析方法 | median | mean | p95 |
|---|---:|---:|---:|
| 原逐层 PyTorch 路径 | 0.387347 ms | 0.393815 ms | 0.420099 ms |
| NumPy CPU row fast path | 0.130504 ms | 0.131757 ms | 0.139963 ms |

均值减少约 **66.5%**，即 `0.262058 ms/item`。按已有 900 个 draft item
外推，约避免 `235.9 ms/request` 的主机解析工作，512-token 请求约
`0.461 ms/output-token`。这是 CPU 工作量分析估算；metadata worker 和模型计算
存在重叠，因此不宣称它会等比例转化为端到端 TPOT。

同一脚本逐层比较了新旧三个 tensor：

- `aggregated_expert_ids`；
- `aggregated_score_sum`；
- `aggregated_activation_count`。

全部 `torch.equal`，因此下游 prefetch candidate 输入没有变化。

## 测试

```text
PYTHONPATH=. /home/edge/.conda/envs/nano_moe/bin/python -m pytest -q \
  tests/test_prefetch_runtime_meta.py \
  tests/test_predictive_prefetch.py \
  tests/test_prefetch_runtime.py \
  tests/test_model_runner_prefetch.py \
  tests/test_spec_engine_prefetch.py \
  tests/test_dual_queue_prefetch.py \
  tests/test_verify_cuda_graph_kt_hybrid.py

101 passed in 6.16s
```

## 后续方向

主机解析已经明显收窄，但当前每个 draft forward 仍产生三个 segment metadata item。
后续优先分析：

1. candidate index 更新和 rank-cache rebuild 是否可批量化；
2. metadata host-buffer reuse 等待是否来自 worker/GIL，而不是 D2H 带宽；
3. prefetch publish/consume 比率能否驱动 admission gate，减少低价值 H2D；
4. BF16 KT 原生路径与 Nano `llamafile_bf16` 在完全相同 qlen/route pattern 下的
   wrapper 成本。
