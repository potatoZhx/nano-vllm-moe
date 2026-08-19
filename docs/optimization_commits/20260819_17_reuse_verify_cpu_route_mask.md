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

绝对收益约 0.065--0.180 ms/verify，仅说明重复 graph kernel 被消除，不能直接外推成
端到端 TPOT。

## 单请求 TPOT 验收

恢复端到端验收后，使用同一个 MMLU-Pro validation 第 0 条、seed 20260719、固定生成
512 token、temperature 0.6、single-weight F16 CPUInfer 2 x 8 threads，各版本运行一次。
所有 engine/transfer/profile timing 均关闭。

| preset | 直接前序 | route-mask 复用 | 变化 | decode rounds |
|:---|---:|---:|---:|---:|
| `k2_dynamic_f16_3080_active14_phase1_recent` | 65.072 ms | **60.156 ms** | **-7.55%** | 270 -> 260 |
| `k2_dynamic_f16_3080_active14` | 70.663 ms | **62.637 ms** | **-11.36%** | 284 -> 271 |

两条候选都通过 512-token 固定长度和输出 validation，并相对各自直接前序取得正收益，
所以按单请求门禁保留本提交。但随机 sampling 轨迹分别从第 64/75 个 token 起分叉，候选
也分别少执行 10/13 个 verify round；微基准所示的实现净收益只有约
0.065--0.180 ms/verify，不能把端到端的全部 7.55%/11.36% 归因于 mask kernel。

同日复跑全局锚点 `296cf59 + active14` 为 56.846 ms/token，故本提交没有刷新全局最佳
commit。正式结果目录为 `results/tpot_ab_route_metadata_20260819/` 与
`results/tpot_ab_active14_formal_20260819/`。

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
cache-mask gather kernel；两套单请求均优于直接前序，但没有超过全局锚点。
