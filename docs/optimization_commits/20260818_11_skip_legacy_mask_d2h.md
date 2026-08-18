# Skip unused legacy host-mask refresh

## 目标与不变量

本优化保留现有 `k2_dynamic_f16_3080_active14` 的 acceptance predictor、TPOT
`first_increase` 动态 K1/K2、predictive prefetch、LRU cache 和 standard sampling
语义，只移除 legacy llamafile F16 CPU backend 不消费的一条数据传输。

`KtDirectCpuMoeBackend` 有两种不同的不变量：

- native `kt_direct` 在构造 `MOEConfig` 时传入 `gpu_expert_mask_cpu.data_ptr()`，CPU
  kernel 依赖最新 host mask，必须在 forward 前刷新；
- legacy `llamafile_f16` 的 `MOEConfig` 没有 mask 指针，cached route 已由
  `_cpu_topk_ids()` 在 GPU 上改为 `-1`。此前仍在每个 MoE layer、每次 draft/verify
  forward 中把 128 个 bool 从 GPU 复制到未被读取的 host tensor。

因此新 helper 只为 native backend 刷新 host mask；legacy 路径跳过该 copy。Qwen3
共有 48 个 MoE layer，这会从每次 draft 和 verify CUDA Graph 中分别删除 48 个冗余
D2H memcpy node，不改变 CPUInfer 收到的 expert id、routing weight 或模型权重。

## 单请求验收

遵循当前节时约束，只选择 MMLU-Pro validation 的第 0 条请求，各运行一次。两次均为：

- RTX 3080 GPU 0，Qwen3-30B-A3B；
- `k2_dynamic_f16_3080_active14`，672 个 expert slot，1536-token 物理 KV；
- raw prompt 107 token，固定生成 512 token，`ignore_eos=true`；
- temperature 0.6，runtime seed 20260719；
- profile/debug timing 全关；
- single-weight + llamafile F16，CPUInfer 16 threads / 2 NUMA pools。

| 版本 | TPOT | decode rounds | mean round wall | 正确性 |
|:---|---:|---:|---:|:---|
| 优化前 | 65.517 ms/token | 269 | 124.458 ms | 512 token，valid |
| 跳过 legacy mask D2H | **59.701 ms/token** | 273 | **111.748 ms** | 512 token，valid |
| 变化 | **-5.816 ms / -8.88%** | +4 | -12.711 ms | 无 validation error |

候选实际多执行 4 个 speculative round，仍取得 8.88% 端到端收益，因此改善不是由
更少 round 或偶然更高 acceptance 造成。sampling 输出 digest 从
`58f2071c...` 变为 `e65e4f00...`；移除 graph memcpy 改变了异步 cache/prefetch 的
完成时序，CPU/GPU F16 expert 路径不要求 bitwise 一致，故 stochastic sampling 轨迹
可以变化。两次都通过固定长度、重复 token run 和 validation 检查。

结果目录：

- 基线：`results/goal_active14_mmlu0_20260818/`
- 候选：`results/goal_active14_skip_mask_d2h_mmlu0_20260818/`

## 最新 KTransformers 对照

KTransformers 最新完整 suite 位于
`../ktransformers/benchmark_outputs/ktransformers_qwen3_cpu_experts_tpot_suite_20260818-064511/`。
Nano 候选是完整 `llm.step()` 墙钟；KT headline 是 graph replay model-forward-only，
不包含 sampling。即便采用对 Nano 更严格的 KT 口径：

| KT 参考 | KT | Nano | Nano 领先 |
|:---|---:|---:|---:|
| 同源第 0 条，forward-only | 71.553 ms | 59.701 ms | **16.56%** |
| 三数据集 suite forward-only mean 中最低值 | 66.028 ms | 59.701 ms | **9.58%** |
| 同源第 0 条，完整墙钟近似 | 74.320 ms | 59.701 ms | **19.67%** |
| MMLU-Pro suite，完整墙钟近似均值 | 67.688 ms | 59.701 ms | **11.80%** |

跨实现仍有 chat template、EOS、dtype 和 sampling 参数差异；但 Nano 的完整解码已经比
KT 最低的 forward-only suite mean 快 9.58%，所以结论不依赖更宽松的计时口径。

## 测试

```bash
PYTHONPATH=. /home/edge/.conda/envs/nano_moe/bin/python -m pytest -q \
  tests/test_kt_direct_backend.py \
  tests/test_verify_cuda_graph_kt_hybrid.py
```

结果：`43 passed`。

新增测试分别锁定：legacy backend 不刷新未使用的 host mask，native backend 仍执行
刷新。性能验收命令与此前 active14 文档相同，仅改为 dataset mode、`--num-samples 1`
和上面的结果目录。

## 一句话总结

删除 legacy F16 每层未被 CPUInfer 消费的 mask D2H graph node，使代表性 MMLU 请求
TPOT 从 65.517 降至 59.701 ms，并以完整解码口径明显超过最新 KTransformers。
