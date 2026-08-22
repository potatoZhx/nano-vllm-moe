# 否决 CPUInfer `group_min_len=2`

> **公平性更正（2026-08-22）：** 本页使用的 t32 资源超过 16-thread baseline 上限，
> 因此只保留为历史负结果，不能据此推荐线程配置。当前统一 t16 口径见
> [`20260822_30_uniform_t16_fairness_revalidation.md`](20260822_30_uniform_t16_fairness_revalidation.md)。

## 假设

Nano 的 legacy llamafile CPUInfer 当前固定 `m_block=32`、`group_min_len=1`，因此 qlen1
也走 `forward_many`。KTransformers 同源 C++ 算子在 `qlen < group_min_len` 时改走
`forward_one`；本轮只考察把阈值改成 2，即 qlen1 走单 token 路径，而 qlen2/3 verify
仍走 grouped 路径。动态 K、cache、prefetch、route、权重精度和线程拓扑均不改变。

## exact-Nano CPUInfer 微基准

环境为 F16 expert / BF16 hidden、32 total threads、2 x 16 NUMA pools、`m_block=32`、
50 warmup + 500 iterations、50% CPU routes。结果位于未纳入 Git 的
`results/analysis_cpuinfer_nano_groupmin_20260819/`：

| qlen | group_min=1 µs/route | group_min=2 µs/route | 变化 |
|---:|---:|---:|---:|
| 1 | 127.026 | **119.237** | **-6.13%** |
| 2 | 119.211 | **118.763** | -0.38% |
| 3 | 117.028 | **116.550** | -0.41% |

孤立算子支持候选：qlen1 明显更快，qlen2/3 没有实质回退。因此临时加入默认仍为 1 的
配置贯通和单元测试，再按要求跑一条真实请求。该临时代码在负收益确认后已全部撤销。

## 一条真实请求 TPOT 门禁

配置为当前推荐 `k2_dynamic_f16_3080_active14_phase1_recent_t32`、MMLU-Pro validation
第 0 条、seed 20260719、512 固定输出、temperature 0.6、single-weight F16，且所有
profiling 关闭：

| CPUInfer 阈值 | TPOT | decode rounds | mean round wall | validation |
|:---|---:|---:|---:|:---|
| group_min=1（现有 t32） | **59.673 ms** | 261 | 116.830 ms | 512 token，valid |
| group_min=2 | 60.487 ms | 271 | **114.056 ms** | 512 token，valid |
| 变化 | **+0.815 ms / +1.37%** | +10 | -2.775 ms | 无错误 |

候选每轮实际更快，但从第 76 个输出 token 起与基线分叉，最终多 10 个 speculative
rounds；在当前 stochastic sampling 的单请求门禁下，总 TPOT 是负收益。不能把 1.37%
全部因果归给算子本身，但也没有足够端到端证据保留新参数或新 preset。

候选结果位于 `results/tpot_phase1_recent_t32_groupmin2_20260819/`，基线位于
`results/tpot_phase1_recent_t32_20260819/`。

## 决定

- 完全撤销 `kt_llamafile_group_min_len` 的 Config/CLI/model/backend 贯通和测试；
- 保持现有 runtime `group_min_len=1`，保留 t32/t16/active14 及全部动态长度 fallback；
- 后续 CPUInfer 调优优先扫不改变算术路径的线程、affinity/task/queue 参数；若修改
  `m_block` 或 forward-one/many 选择，仍必须以一条真实请求 TPOT 决定是否保留。

## 一句话总结

`group_min_len=2` 虽使 exact-Nano qlen1 微基准快 6.13%，真实请求却从 59.673 回退到
60.487 ms/token，因此代码全部撤销，仅保留负结果。
