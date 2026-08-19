# 否决 production-off verify metadata profile 聚合删除

## 候选

`ModelRunner._record_verify_metadata_profile_from_runtime_meta()` 的最终写入位于
`profile_enabled` 分支，但此前无论 profile 是否开启都会先遍历 verify metadata，统计
logical/execution route、CPU miss、active expert 和逐层 row。候选在函数入口为
production-off 增加快速返回；profile-on 路径保持原样。

静态依赖上，这些聚合值只进入 profile 输出，不直接参与 prefetch/cache 或动态 draft
length。但函数位于异步 metadata worker 的 `collect -> profile aggregate -> observe_verify`
顺序中，执行耗时也会改变 `observe_verify`、candidate index 更新和后续 prefetch 的时序。

## 微基准

合成 48 层、每层 12 个 active expert 的 verify metadata，11 轮各 100 次调用：

| 状态 | ms/call |
|:---|---:|
| 原 production-off 聚合 | 5.805192 |
| 入口快速返回 | 0.000200 |

孤立函数确实消除了 CPU 工作，但这不足以证明异步系统端到端更快。

## 单请求 TPOT：负收益

同一 MMLU-Pro validation 第 0 条、seed 20260719、512 固定输出、temperature 0.6、
active14、single-weight F16、2 x 8 CPUInfer，所有 profiling 关闭：

| 版本 | TPOT | decode rounds | mean round wall | validation |
|:---|---:|---:|---:|:---|
| 保留基线 `ac254df` | **60.526 ms** | 264 | **117.154 ms** | 512 token，valid |
| 跳过 profile 聚合候选 | 69.893 ms | 274 | 130.348 ms | 512 token，valid |
| 候选变化 | **+9.367 ms / +15.48%** | +10 | +13.193 ms | 无错误 |

token 轨迹从第 60 token 起分叉。候选既增加 10 个 speculative round，也显著增加平均
round wall；因此不能把结果解释成纯采样噪声，也不能用 5.8 ms 孤立微基准宣称收益。
结果目录：`results/tpot_active14_skip_verify_meta_profile_20260819/`。

## 决定与后续边界

- 候选运行时代码和测试已完全撤销，不产生性能提交；当前分支仍为 `ac254df` 行为。
- 不应再直接删除该聚合。它虽然不写 production profile，却对 async worker 相位具有隐含
  pacing 作用。
- 若以后继续优化，应先 profile metadata queue turnaround、observe/publish 时点、cache
  replacement 和 transfer completion，再把“统计 CPU 工作”与“必要 pacing”拆开；可尝试
  显式延迟/批处理，而不是无条件快速返回。
- 动态长度 preset、route-mask 和 production consumption gate 均保持不变。

## 一句话总结

verify metadata profile 聚合看似是 5.8 ms dead work，但直接删除使 TPOT 回退 15.48%，
暴露出异步 observe/prefetch 的隐含时序依赖，因此否决并完整撤销。
