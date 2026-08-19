# Verify histogram NumPy collect 快路径（端到端未通过，已回退）

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

微基准只测孤立 CPU readback；恢复端到端验收后，该实现没有通过正收益门禁，见下节。

## 单请求 TPOT 与回退决定

同一 MMLU-Pro validation 第 0 条、seed 20260719、512 固定输出、temperature 0.6、
single-weight F16、2 x 8 CPUInfer，各版本运行一次；engine/transfer/profile timing 全关。

| preset | route-mask 前序 `cd2cb06` | 加入本改动 `2654eb4` | 变化 | decode rounds |
|:---|---:|---:|---:|---:|
| `...active14_phase1_recent` | 60.156 ms | 65.567 ms | **+8.99%** | 260 -> 266 |
| `...active14` | 62.637 ms | 68.960 ms | **+10.10%** | 271 -> 267 |

两条输出都通过 512-token validation，但 stochastic token 轨迹分别在第 60/64 个 token
起分叉。active14 候选虽然还少执行 4 个 decode round，总 decode 仍多 3.231 秒，因而
不能用“候选多跑了”解释回退。两套真实请求都为负收益，孤立 collect 的 78.7% 微基准
没有兑现为端到端收益。

结论：`2654eb4` 仅保留为被否决候选的历史证据；当前分支通过后续
`f844475` 恢复原 PyTorch verify hybrid collect。dynamic preset、draft NumPy
collect 与 route-mask 优化均不回退。结果目录为
`results/tpot_ab_route_metadata_20260819/` 和
`results/tpot_ab_active14_formal_20260819/`。

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

verify NumPy collect 虽把孤立 CPU 微基准降低 78.7%，但两套单请求 TPOT 分别回退
8.99%/10.10%，因此按端到端正收益门禁撤销。
