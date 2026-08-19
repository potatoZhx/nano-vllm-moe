# Production-off 跳过 verify consumption 诊断

## 问题

提交 `0a40f64` 已把 source residency 生命周期对象放在 profile gate 后，但 production
仍在每轮 verify 的最后一个 metadata item 调用 `record_verify_consumed()`：

- 逐层把 active expert tensor 转为 Python list；
- 逐 expert 查询执行时 status 和当前 cache；
- 查 `_recent_published`、按 source 累加消费 counter；
- 扫描 TTL 并清理 recent-publication map。

`_recent_published`、消费 counter 和 source residency 只由 `get_profile()` 输出；全仓搜索
确认它们不参与 prefetch admission/ranking、cache victim、transfer budget 或动态 draft
length。production 还会在每次 publication 维护这两张只供上述统计使用的 map。

## 改动

- `_record_prefetch_residency_published()` 在 diagnostic profile 关闭时不再维护
  `_recent_published` / `_recent_published_source`；
- `record_verify_consumed()` 增加 production-off 快速返回；
- deferred verify metadata 只在 diagnostic/source-lifecycle profile 开启时请求 consumption
  统计，避免 worker 进入无效函数；
- engine/spec/transfer profile 开启时保留完整 per-source counter、TTL 和 residency 归因；
- 不改变 verify metadata collect/observe、recent verify candidate index、cache/prefetch、
  route mask、sampling 或 K1/K2 动态长度算法。

## 分析微基准

合成 48 层 verify metadata、每层 12 个 active expert，production-off，11 轮各 200 次调用
取中位数：

| 状态 | `record_verify_consumed` ms/call |
|:---|---:|
| 前序 `cd2cb06` 路径 | 1.852132 |
| production-off 快速返回 | 0.000132 |

该微基准只说明被删除的诊断 CPU 工作，不替代下面的真实请求 TPOT。

## 单请求 TPOT 验收

同一 MMLU-Pro validation 第 0 条、seed 20260719、512 固定输出、temperature 0.6、
`k2_dynamic_f16_3080_active14`、single-weight F16、2 x 8 NUMA CPUInfer，所有 profiling
关闭：

| 版本 | TPOT | decode rounds | mean round wall | validation |
|:---|---:|---:|---:|:---|
| 回退后基线（`cd2cb06` runtime） | 62.637 ms | 271 | 118.108 ms | 512 token，valid |
| 跳过 production consumption | **60.526 ms** | 264 | **117.154 ms** | 512 token，valid |
| 变化 | **-2.111 ms / -3.37%** | -7 | **-0.954 ms** | 无错误 |

随机 sampling 输出从第 131 token 起分叉，所以不能把全部 TPOT 差值归因于本改动；但候选
不仅 verify rounds 更少，平均每轮也降低 0.954 ms，与跳过诊断扫描的方向和量级一致。
因此按一条真实请求正收益门禁保留。它仍未超过同日全局锚点 `296cf59` 的
56.846 ms/token。

结果目录：
`results/tpot_active14_skip_consumed_diag_20260819/`；基线位于
`results/tpot_ab_active14_formal_20260819/route_cd2cb06/`。

## 测试

```bash
/home/edge/.conda/envs/nano_moe/bin/python -m pytest -q \
  tests/test_prefetch_runtime.py \
  tests/test_verify_feedback.py \
  tests/test_verify_segment_graph.py::TestEnqueueVerifySegmentMetadata
```

结果：`17 passed`。完整 `tests/test_verify_segment_graph.py` 另有一个既有失败：
`TestVerifySegmentReplay` 将 `runtime_meta_recorder=None` 却断言 metadata enqueue 两次；
已在未修改的 `cd2cb06` worktree 单独复现，不属于本改动。

## 一句话总结

关闭 profile 时不再维护和扫描只供诊断的 verify consumption 状态，使代表性单请求 TPOT
从 62.637 降至 60.526 ms/token，同时完整保留 profile-on 分析能力。
