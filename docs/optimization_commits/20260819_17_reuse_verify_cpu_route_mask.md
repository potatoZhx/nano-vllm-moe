# Verify 复用 CPU route mask

## 问题

`build_verify_graph_safe_plan_gpu()` 已经为每层计算了 `plan.cpu_route_mask`：它精确表示
当前 top-k 中哪些 route 未命中 GPU expert cache。legacy llamafile backend 在提交
CPUInfer 前却没有复用它，而是再次执行：

1. `gpu_expert_mask.index_select(topk_ids)`；
2. `topk_ids.masked_fill(cached_routes, -1)`。

因此 CUDA Graph 每层多了一次完全重复的 cache-mask gather。48 层 verify 会重复 48 次。

## 改动

- `KtDirectCpuMoeBackend.begin_forward_graph_verify()` 新增可选
  `cpu_route_mask`；
- KT hybrid verify 将已经生成的 `plan.cpu_route_mask` 传入 backend；
- legacy llamafile 使用一次 `torch.where(cpu_route_mask, topk_ids, -1)` 直接生成
  CPUInfer route ids；
- 未传 mask 的 eager/兼容调用继续走原来的 expert-mask fallback；
- `include_gpu_cached_routes=True` 的全 CPU route 特例仍忽略 mask，语义不变；
- 非 legacy AVX2/AMX backend 继续由 native config 消费 GPU mask，不改变路径。

本改动不调整动态 draft length、acceptance、cache admission、prefetch budget 或任何最优
preset。

## Analysis-only 微基准

在 RTX 3080 上分别捕获 48 层旧/新 route-mask 子图，每次 replay 等价一次 verify 的
mask 生成，3000 replay × 9 轮取中位数：

| bucket | 旧路径 µs/48 layers | 复用 mask µs/48 layers | 减少 |
|---:|---:|---:|---:|
| 2 | 271.641 | 91.392 | 66.4% / 180.249 µs |
| 3 | 156.376 | 91.218 | 41.7% / 65.159 µs |

绝对收益约 0.065--0.180 ms/verify，仅说明重复 graph kernel 被消除。按用户要求没有运行
新的端到端优化验证，因此不把该微基准外推为 TPOT 收益，也不更新正式最优结果。

## 正确性

- route-mask 复用与原 cache-mask fallback 生成完全相同的 `-1` route；
- 增加 mask 元素数不匹配的显式错误；
- legacy F16/BF16 权重和 BF16 hidden/output 语义不变；
- 原有 CUDA Graph begin/finish、双 buffer slot 与 model dispatch 测试继续通过。

## 验证

```bash
conda run -n nano_moe pytest -q \
  tests/test_kt_direct_backend.py \
  tests/test_verify_cuda_graph_kt_hybrid.py
```

结果：`43 passed`。

## 一句话总结

直接复用 verify plan 已有的 CPU route mask，消除 legacy CPUInfer 提交前每层一次重复的
cache-mask gather kernel。
