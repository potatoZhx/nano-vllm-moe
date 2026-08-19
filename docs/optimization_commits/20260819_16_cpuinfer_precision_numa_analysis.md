# CPUInfer 精度/NUMA/稀疏 route 分析基准

## 目的

最新 KTransformers suite 与 Nano 都使用 KTransformers 构建的
`cpuinfer_ext`/llamafile MoE，但两者不是同一精度与 route 形态：

- KT：BF16 expert、BF16 hidden、qlen=1、8 条 route 全部在 CPU；
- Nano：F16 expert、BF16 hidden、qlen 主要为 2/3，约 49% route 在 CPU；
- 两者均为 16 threads、2 个 NUMA subpool，NUMA0/1 各 8 threads。

原脚本只能构造 BF16 expert，且只能使用 `CPUInfer(threads)` 的隐式单池接口，无法做
同口径归因。本提交只增强分析工具，不改变 Nano runtime 或任何 optimized preset。

## 改动

`scripts/bench_ktransformers_cpuinfer_qwen3_moe.py` 新增：

- `--weight-dtype {bf16,f16}`，hidden/output 始终明确记录为 BF16；
- `--threadpool-count` 与 `--numa-nodes`，使用与 runtime 相同的
  `WorkerPoolConfig`；
- `--cpu-route-fraction`，以 `-1` 模拟被 GPU cache 接管的 route；
- `--seed`，固定 expert 与 route-mask 样本；
- 结果中记录 weight/hidden dtype、NUMA layout 和实际 CPU route 数；
- FLOP 统计只计算未被屏蔽的 CPU route。

默认参数保持原脚本行为：BF16、单 threadpool、全部 route 在 CPU。

## Analysis-only 结果

环境：Xeon Gold 5218R、RTX 3080、16 CPUInfer threads、2 subpools
`[numa0:8, numa1:8]`、`group_min_len=1`、`m_block=32`。结果保存在未纳入 Git 的
`results/analysis_cpuinfer_precision_numa_20260819/`。

| qlen | CPU route | BF16 µs/layer | F16 µs/layer | F16 相对 BF16 |
|---:|---:|---:|---:|---:|
| 1 | 8/8 | 991.253 | 996.069 | +0.49% |
| 2 | 8/16 | 984.342 | 978.333 | -0.61% |
| 3 | 12/24 | 1410.885 | 1417.812 | +0.49% |
| 5 | 20/40 | 2241.737 | 2235.458 | -0.28% |
| 7 | 28/56 | 3001.813 | 2986.199 | -0.52% |

五个点的绝对差异均不超过 0.61%。因此在本机 AVX2/llamafile 路径上，KT 使用 BF16
并不能解释其端到端差异；F16/BF16 都是 16-bit 权重，当前瓶颈主要由 route 数、qlen、
NUMA merge 与 CPU/GPU 同步决定。

当前 active14 profile 的真实 bucket2/3 分别是 7.87/11.85 CPU routes/layer，与本基准
的 8/12 routes 对齐。F16 原生串行估算为 46.96/68.05 ms/48 layers，而 profile 中
model forward 为 58.16/73.95 ms，说明 wrapper 不是数量级瓶颈，CPU expert work 才是
verify 的主体。

这些都是分析微基准，不是端到端优化验证，不更新 `59.701 ms/token` 的正式最优记录。

## 验证

```bash
conda run -n nano_moe pytest -q \
  tests/test_bench_ktransformers_cpuinfer_qwen3_moe.py
```

结果：`4 passed`。

## 一句话总结

补齐同 route、同双 NUMA 的 F16/BF16 CPUInfer 基准，并确认精度 specialization 不是当前
Nano/KT 性能差距的主因。
