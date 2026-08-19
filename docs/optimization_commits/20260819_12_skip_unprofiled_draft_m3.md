# 2026-08-19：正式运行跳过 draft-M3 诊断计算

## 一句话总结

在不改变 metadata、prefetch 候选、cache victim 或动态 draft 长度策略的前提下，正式运行不再为仅供 profile 展示的 draft-M3 完美命中率逐层执行 `torch.unique` 和 Python cache 查询。

## 背景与边界

最新 KTransformers 对照使用
`Qwen3Moe-serve-bf16-cpu-experts.yaml`：专家权重和 hidden 均为 BF16。当前
Nano 最优 preset `k2_dynamic_f16_3080_active14` 使用 KT 构建出的同源
`cpuinfer_ext`/llamafile 算子家族，但 `llamafile_f16` 仅表示专家权重为
FP16，hidden 仍为 BF16。因此两者是同源实现的不同权重精度实例，不能把
“同源算子”误写成“完全相同的 kernel 配置”。

本提交不切换专家精度，也不修改以下生产决策：

- `PredictivePrefetchRuntime.observe_draft()` 的 metadata 接收；
- `draft_segment_index` 的更新、排序和候选集合；
- cache 保护、victim 选择和 H2D 提交；
- `k2_dynamic_f16_3080_active14` 的动态 K1/K2 参数和保留配置。

## 问题

`_record_draft_m3_cache_hits()` 用于统计“一个 draft forward 的所有层是否
都命中 GPU expert cache”。它会对每层 routed experts 执行
`torch.unique(...).tolist()`，再逐 expert 调用 `cache.is_cached_cpu()`。
这些结果只写入 `draft_m3_*` profile 字段，不参与任何运行时决策，但此前在
`engine_profile=False`、`spec_profile=False`、
`transfer_aware_profile=False` 的正式 TPOT 路径中仍会执行。

现有 active14 profile 每个请求约有 900 个 draft metadata item，因此这段
单次很小的诊断工作会在长请求中重复放大。

## 修改

`PrefetchRuntime` 新增 `_diagnostic_profile_enabled`，仅在以下任一模式开启时
收集 draft-M3：

- `engine_profile`；
- `spec_profile`；
- `transfer_aware_profile`。

正式运行会同时跳过 M3 step-start 状态和逐层 M3 cache-hit 扫描。profile
开启时保留原有字段和语义。

## 分析实验

遵照“停止优化验证实验”的要求，本提交没有再运行端到端数据集验证，只做
不生成模型输出的 CPU 合成 metadata 分析。

环境模拟 48 层、每层 14 个 cache slot；单个 metadata item 覆盖 16 层，
每层 8 个聚合 routed experts；每种模式重复 500 次：

| 模式 | median | mean | p95 |
|---|---:|---:|---:|
| profile 开启，保留 M3 | 0.233783 ms | 0.235227 ms | 0.238135 ms |
| profile 关闭，跳过 M3 | 0.000217 ms | 0.000222 ms | 0.000234 ms |

按已有 profile 的约 900 个 draft metadata item 外推，可避免约
`211.5 ms/request` 的纯诊断 CPU 工作，折合 512-token 请求约
`0.413 ms/output-token`。这是分析估算，不替代端到端 TPOT 结果，也不宣称
等比例转化为 wall-clock 收益，因为 metadata worker 与 GPU/CPU compute
存在重叠。

同一实验还比较了 profile 开/关后的 `draft_segment_index`：两边候选 key
完全一致，均为 114 个。

## 测试

```text
PYTHONPATH=. /home/edge/.conda/envs/nano_moe/bin/python -m pytest -q \
  tests/test_predictive_prefetch.py \
  tests/test_prefetch_runtime.py \
  tests/test_prefetch_runtime_meta.py \
  tests/test_model_runner_prefetch.py \
  tests/test_spec_engine_prefetch.py

48 passed in 5.97s
```

新增测试覆盖：

- profile 开启时 `draft_m3_*` 仍按原语义记录；
- profile 关闭时不调用 `torch.unique`，也不残留 M3 round 状态。

## 后续方向

下一步继续区分两类成本：

1. 用 Nano 已有 `llamafile_bf16` 做 KT 原生 BF16 同精度调用链分析，避免把
   FP16/BF16 差异混入框架开销；
2. 审计 `SegmentCandidateIndex`、transfer lifecycle 和 metadata timing 中
   仍无条件执行的 profile-only 计时/计数；
3. 用已有 profile 定量评估 prefetch admission、cache churn 和已发布 expert
   的真实消费收益，再决定是否修改算法。
