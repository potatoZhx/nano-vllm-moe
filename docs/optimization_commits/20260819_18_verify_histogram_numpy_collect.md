# Verify histogram NumPy collect 快路径

## 问题

runtime metadata 已在 GPU 侧聚合成每层 128-expert histogram，并异步拷贝到 pinned host
buffer。draft 的 `histogram` collect 已使用 NumPy 提取非零 expert；verify 的
`histogram_kt_hybrid` 却仍为 48 层逐层调用：

- `torch.nonzero`；
- `index_select`；
- 多次 `.to(dtype/device)`；
- status/activation 的 torch 规约与 clone。

这些张量都已在 CPU，且每层通常只有约 8--12 个非零 expert，开销主要是 PyTorch
dispatcher，而不是数据处理。

## 改动

- `histogram` 与 `histogram_kt_hybrid` 共用一次 `.numpy()` host-buffer view；
- 每层用 `np.flatnonzero` 提取 active expert；
- counts/score 仅切出非零行，再用 `torch.from_numpy` 构造结果；
- `active_count` 直接使用非零项数，`miss_count` 只在 active expert 上统计 status=2；
- `expert_status` 与可选 `execution_activation_count` 显式 `.copy()`，保持 host buffer
  pool 复用后的结果独立性；
- raw metadata、transfer-aware profiling、perfect-match trace 和 GPU record/offload 路径
  均不改变。

本改动只优化 CPU readback，不修改 cache access、prefetch queue、dynamic draft length、
route status 或 optimized preset。

## Analysis-only 微基准

合成 Qwen3-30B-A3B verify metadata：48 layers、128 experts/layer、每层 12 个 active
expert、score/status 均开启。每轮连续 collect 200 次，15 轮取中位数：

| 实现 | ms/collect | 相对变化 |
|:---|---:|---:|
| 逐层 PyTorch | 2.750641 | baseline |
| NumPy sparse row | 0.585856 | -78.7% |
| 绝对减少 | 2.164785 | - |

该结果测量的是 production `histogram_kt_hybrid` readback CPU 工作。此前 source lifecycle
分析开启了 `transfer_aware_profile`，会强制使用 raw route metadata，因此那次 profile 的
13.70 ms/verify collect 不在本优化覆盖范围，不能拿来外推收益。

按用户要求没有运行新的端到端优化验证；正式最优仍为既有 `59.701 ms/token`，动态最优
preset 保持不变。

## 正确性与测试

新增 host buffer 复用回归断言：collect 后清空 pooled status/execution-count buffer，已返回
metadata 保持原值。现有测试还覆盖：

- logical token 与 padding execution count 分离；
- padding-only expert 不计入 active/miss；
- score/status 与 execution histogram；
- CUDA Graph capture-safe record/offload；
- omit-score/status 和 raw transfer-aware 兼容路径。

```bash
conda run -n nano_moe pytest -q \
  tests/test_prefetch_runtime_meta.py \
  tests/test_verify_cuda_graph_kt_hybrid.py
```

结果：`41 passed`。

## 一句话总结

让 verify histogram 复用 draft 已验证的 NumPy 稀疏提取方式，把 48 层 metadata collect
CPU 开销降低 78.7%，并保持 pooled buffer 的独立复制语义。
