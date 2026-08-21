# 28. 融合 cache LUT publication 更新

日期：2026-08-21

## 一句话总结

把每次 active-cache 映射提交的 3–5 个 CUDA 标量更新融合为一个预热 Triton kernel，
ghost8 同日一请求 TPOT 从 54.393 降到 54.236 ms/token（-0.29%）。

## 动机

b1 instrumented profile 记录 2531 次 publication，累计 `publish_ms=7888 ms`。profile
会放大绝对时间，不能直接当 production 收益，但当前每次 commit 的控制面工作是明确的：

- eager reservation 需要清除 `expert_to_slot_lut`、`slot_to_expert_lut` 和 cached mask；
- eager publication 再写回三个映射；
- deferred publication 一次完成旧 expert 清除和新 expert 发布，共 5 个标量写。

这些 Python 标量赋值分别发出很小的 CUDA kernel。它们不传输 expert 权重，却会与
2531 次约 22.24 GiB H2D publication 共享 launch/stream 调度路径。

## 隔离微基准

RTX 3080、5000 次循环、每组 5 次取中位数：

| 操作 | PyTorch 标量更新 | 融合 kernel | 加速 |
|:---|---:|---:|---:|
| deferred commit（5 writes） | 0.049835 ms | 0.013565 ms | 3.67x |
| eager unmap（3 writes） | 0.030148 ms | 0.012749 ms | 2.36x |

该结果只证明减少 launch 可行；是否改善端到端仍由一请求 TPOT 决定。

## 实现与 JIT 陷阱

新增 opt-in `fused_cache_lut_updates`，只在 CUDA cache 上启用；CPU cache 和开关关闭的
全部旧 preset 继续走原来的内联 PyTorch 更新。保留的独立 preset 为：

```text
k2_dynamic_f16_3080_active14_phase1_recent_t32_b1_ghost8_lutfuse
```

第一次实现虽然融合了 kernel，但把 expert/slot 索引作为普通 Python integer 传给 Triton。
Triton 会按整数值/对齐继续生成 specialization；短 warmup 没有覆盖真实请求中的索引组合，
导致 timed decode 出现 JIT 尖峰：

| 候选 | TPOT | step p50 | step max | 决定 |
|:---|---:|---:|---:|:---|
| ghost8 baseline | 54.392560 ms | 100.339 ms | 195.287 ms | 对照 |
| 未正确预热的 LUT fusion | 59.466996 ms | 96.209 ms | **611.618 ms** | 否决该实现 |

修正版对 `previous_expert/expert_idx/slot_idx` 设置 `do_not_specialize`，只保留语义上固定的
`evict_previous=true/false` 两个编译变体，并在模型加载阶段用 scratch LUT 显式编译、启动
unmap 和两种 commit。预热后的真实动态索引首次调用约 0.12–0.25 ms，不再发生几十到
数百毫秒的 JIT。

## 修正版同日一请求门禁

共同口径：MMLU-Pro validation sample 0、seed 20260719、temperature 0.6、107-token
prompt、固定生成 512 token、single-weight llamafile F16、动态 K1/K2、profile 全关。

| 配置 | TPOT | decode tok/s | decode steps | 平均 step wall | 校验 |
|:---|---:|---:|---:|---:|:---:|
| ghost8 | 54.392560 ms | 18.385 | 263 | 105.683 ms | 512/512 |
| ghost8 + LUT fusion/prewarm | **54.235843 ms** | **18.438** | 269 | **103.028 ms** | 512/512 |

TPOT 改善 **0.29%**，decode throughput 提高 **0.29%**。候选虽然多 6 个 decode steps，
平均 step wall 仍低 2.51%，最终 decode time 27.795→27.715 s。两次 output digest 不同，
所以 0.29% 仍是按既定规则保留的一请求端到端门禁，不是隔离因果估计。

结果目录：

```text
results/tpot_phase1_b1_ghost8_20260821/
results/tpot_phase1_b1_ghost8_lutfuse_20260821/             # 未正确预热，否决
results/tpot_phase1_b1_ghost8_lutfuse_prewarm_20260821/     # 保留实现
```

## 验证与 fallback

- 57 项 config/cache/predictive 单元测试通过；
- CUDA eager、deferred、dynamic-index LUT 生命周期通过；
- 512-token fixed-length/output validation 通过；
- `git diff --check` 通过。

原 `..._ghost8` 是关闭 fusion 的直接 fallback，原 b1 则同时关闭 ghost 与 fusion；历史
budget2、active14 和 full-context-safe 动态 preset 均保留。

## 后续

LUT launch 已压缩，下一项不再继续微调该 kernel。剩余高预期收益控制面是把同一 drain
中的 event query/publication 按 ready batch 处理，或在不额外打包 9 MiB expert 权重的前提下
批量提交 H2D copy；必须显式记录 queue/pacing，避免重现 metadata fast-path 和未预热
fusion 的时序回退。
