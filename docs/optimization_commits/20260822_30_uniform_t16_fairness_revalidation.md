# 30. 统一 16-thread 资源并重跑累计优化链

日期：2026-08-22

## 一句话总结

把所有内置优化 preset 统一为与 baseline 相同的 16 total CPUInfer threads（双 NUMA
2 x 8），重新跑出的 b4→b2→b1→ghost8→LUT fusion 链每一步都为正，最终达到
**52.566 ms/token**；旧 t32 数据保留为历史探索，但不再作为公平性能证据。

## 公平性修正

此前 `...recent_t32...` 系列使用 32 total threads，而可比 baseline 使用 16 total
threads。算法和 CPU 资源同时变化，无法把差值公平归因于算法，因此：

- 所有内置优化 preset 现在都固定 `kt_num_threads=16`；
- 当前拓扑固定为 `kt_threadpool_count=2`、`kt_numa_nodes=0,1`，即每池 8 workers；
- 新增不含 `t32` 的 canonical preset 名称；旧名称只作为命令兼容 alias，实际同样解析为 16；
- 任何内置优化 preset 若被命令行手工改成非 16 threads，配置校验会直接拒绝；
- 三个 benchmark 脚本中的活动示例也从 32 改成 16，避免复制旧命令产生不公平结果。

`optimized_config=none` 仍允许通用/分析脚本自行设置资源；正式 baseline 与优化门禁必须显式
使用相同的 16-thread 口径。

## 同资源一请求重测

共同口径：MMLU-Pro validation sample 0、seed 20260719、temperature 0.6、107-token
prompt、固定生成 512 token、single-weight llamafile F16、动态 K1/K2、所有 profile 关闭、
RTX 3080 device 0。五条运行的 metadata 均为 16 threads、2 pools、NUMA 0/1，CPUInfer
启动日志均显示 `[0:8] [1:8]`；输出均为 512/512 且 validation 通过。

| 累计配置 | canonical preset | TPOT | decode tok/s | steps | mean step wall | 相对前序 |
|:---|:---|---:|---:|---:|---:|---:|
| recent / budget4 | `...phase1_recent` | 60.909540 ms | 16.418 | 276 | 112.771 ms | baseline |
| budget2 | `...phase1_recent_b2` | 55.043059 ms | 18.168 | 265 | 106.140 ms | **-9.63%** |
| budget1 | `...phase1_recent_b1` | 54.627249 ms | 18.306 | 269 | 103.771 ms | **-0.76%** |
| b1 + ghost8 | `...phase1_recent_b1_ghost8` | 53.336804 ms | 18.749 | 264 | 103.239 ms | **-2.36%** |
| ghost8 + LUT fusion | `...phase1_recent_b1_ghost8_lutfuse` | **52.566035 ms** | **19.024** | 265 | **101.363 ms** | **-1.45%** |

从 budget4 到最终配置累计改善 **13.70%**。五次随机采样的 output digest 不同，所以上述
数值仍是一请求可用性/正收益门禁，不是严格同轨迹的隔离因果估计；但在完全一致的 CPU
资源下，每一个相邻保留点都取得正收益，足以支持当前逐版本保留顺序。

## 结果目录

- `results/tpot_t16_fair_recent_b4_20260822/`
- `results/tpot_t16_fair_recent_b2_20260822/`
- `results/tpot_t16_fair_b1_20260822/`
- `results/tpot_t16_fair_b1_ghost8_20260822/`
- `results/tpot_t16_fair_b1_ghost8_lutfuse_20260822/`

最终请求的 step wall p50/p90/p95/max 为 94.486/134.610/141.977/187.491 ms，
`outputs_digest=8d5cf1678e560c01fca372a80da9407525cc761174ee0de4d2f88b06b18702ca`。

## 历史边界与 KT 对照

2026-08-19/21 的 32-thread 数据（包括 53.726、54.393、54.236 ms/token）仍用于说明当时
的探索过程，但不能再称为与 16-thread baseline 公平的“当前最佳”。新的同资源已测最低点
是 **52.566 ms/token**。

相对最新 KT suite 最低 workload mean 66.028 ms/token，当前数值低 20.39%；相对 KT
MMLU-Pro sample 0 的 71.553 ms/token，低 26.54%。两边 prompt formatting、sampling、
seed、EOS 和计时边界仍未完全配平，所以该比较只是跨系统参考，不是严格逐 token A/B。

## 验证

```text
PYTHONPATH=. /home/edge/.conda/envs/nano_moe/bin/python -m pytest -q \
  tests/test_eval_tpot_config.py \
  tests/test_config_predictive_prefetch.py \
  tests/test_predictive_prefetch.py \
  tests/test_cache_lut.py \
  tests/test_expert_cache_staging.py
```

共 59 项通过。新增测试遍历全部内置 preset，锁定 total threads=16，并覆盖旧 t32 alias
降级到 16 以及手工覆盖为 32 被拒绝。

## 当前保留顺序

当前推荐：

```text
k2_dynamic_f16_3080_active14_phase1_recent_b1_ghost8_lutfuse
```

直接 fallback 依次为 ghost8、b1、b2、budget4；它们共享完全相同的 16-thread 资源，只改变
算法开关。下一项优化也必须从这个 canonical t16 preset 出发，并继续执行一条 512-token
真实请求 TPOT 门禁。
