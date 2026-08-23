# 39. verify 边界 source-aware 1+1 预取

日期：2026-08-23

## 一句话总结

在固定 vpb2 内为 `draft_live` 保留一个互补位置、同时保留一个 `verify_history` 位置，
公平 16-thread 单请求 TPOT 从 **52.566035** 降至 **51.793570 ms/token（-1.47%）**；
保留独立 `source11` preset，并永久保留原 lutfuse 最优作为直接 fallback。

## 选择依据

上一轮 admission shadow 显示，当前 786 个 verify segment 边界实际提交的 1,572 个候选
全部来自 `verify_history`，`draft_live` 没有进入 vpb2。固定同样两个位置的离线重排为：

| vpb2 排序 | 当前 step route demand | 命中候选数 |
|:---|---:|---:|
| 当前 global priority（两个 history） | 1,462 | 908 |
| **一个 history + 一个 draft-live** | **1,703** | **1,078** |

去重后仍只属于 `draft_live` 的 953 个边际候选中，928 个会在当前 verify step 使用，
candidate precision 为 97.377%。这使 1+1 比扩大 budget 更保守：它不增加 draft forward、
H2D 次数、expert cache slots 或 CUDA graph 显存，只改变既有两个 dispatch 位置的 source
构成。详细预测口径见
`20260822_38_global_draft_acceptance_cost_and_sources.md`。

## 实现

- 新增默认关闭的 `verify_prefetch_draft_reserve`，默认值 0 保持全部既有 preset 的
  global-priority 顺序；
- `source_aware_verify_candidates()` 只在同一 verify boundary 同时存在 `draft_live` 和
  `verify_history` 时生效；在 dispatch prefix 中放入最高优先级 history 和最高优先级
  marginal draft，剩余候选保持原顺序作为 cached/pending/reservation 失败时的 fallback；
- 新增独立 preset：

```text
k2_dynamic_f16_3080_active14_phase1_recent_b1_ghost8_lutfuse_source11
```

  它只在原 lutfuse preset 上设置 `verify_prefetch_draft_reserve=1`。原
  `...ghost8_lutfuse` 仍解析为 0，没有被覆盖；
- metadata 记录 reserve 值；通过 `object.__new__` 构造的轻量测试 runtime 在 telemetry
  字段缺失时安全视为 profile-off，不改变正式 runtime 行为。

本改动不改变 speculative sampling、动态 K1/K2、0.97 `first_increase`、vpb2 总预算、
cache victim 规则、权重内容或精度。

## 公平单请求 TPOT 门禁

共同口径为 MMLU-Pro validation sample 0、107-token prompt、seed 20260719、temperature
0.6、固定生成 512 token、single-weight llamafile F16、profile 全关、CUDA 0、active14，
以及 CPUInfer 双 NUMA `2 x 8 = 16` total threads。候选 metadata 明确记录：

```text
kt_num_threads=16
kt_threadpool_count=2
kt_numa_nodes=[0, 1]
kt_direct_backend=llamafile_f16
kt_single_weight=true
verify_prefetch_max_per_boundary=2
verify_prefetch_draft_reserve=1
```

| 配置 | TPOT | decode tok/s | decode | steps | mean step | p50/p90/p95/max |
|:---|---:|---:|---:|---:|---:|:---|
| 原 lutfuse 最低点 | 52.566035 ms | 19.024 | 26.861 s | 265 | **101.363 ms** | 94.486/134.610/141.977/187.491 ms |
| **source-aware 1+1** | **51.793570 ms** | **19.307** | **26.467 s** | **254** | 104.199 ms | 97.350/135.509/142.503/186.617 ms |
| 变化 | **-0.772465 ms / -1.47%** | **+1.49%** | **-0.395 s** | **-11 / -4.15%** | +2.80% | p50/p90/p95 +3.03/+0.67/+0.37%，max -0.47% |

候选生成 512/512 token，`output_fixed_length_ok=true`、validation error 为空。结果目录：

```text
results/tpot_t16_b1_ghost8_lutfuse_source11_20260823/
```

本次 digest 与历史最低点不同，且 mean step 反而增加 2.80%；观察到的 TPOT 改善来自
decode rounds 从 265 降到 254，而不是单轮变快。因此这条一请求结果满足既定的正收益可用
门禁并支持保留独立 preset，但不能把 1.47% 全部解释为 source 排序的严格因果加速。若未来
跨 workload 使用，应继续以旧 lutfuse preset 作为低风险 fallback。

## 复现

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
  /home/edge/.conda/envs/nano_moe/bin/python \
  scripts/bench_eval_workload_tpot.py \
  --request-mode dataset --dataset mmlu_pro --num-samples 1 \
  --model-path /home/edge/models/Qwen3-30B-A3B \
  --mmlu-pro-path \
  results/k2_dynamic_f16_3080_eval_workloads_20260817/inputs/mmlu_pro_validation.jsonl \
  --output-dir results/tpot_t16_b1_ghost8_lutfuse_source11_20260823 \
  --optimized-config \
  k2_dynamic_f16_3080_active14_phase1_recent_b1_ghost8_lutfuse_source11 \
  --output-lens 512 --temperature 0.6 \
  --kt-llamafile-extension-path \
  /home/edge/zx/ktransformers/build/lib.linux-x86_64-cpython-312/cpuinfer_ext.cpython-312-x86_64-linux-gnu.so \
  --kt-single-weight true --decode-driver generate \
  --collect-profile false --engine-profile false \
  --engine-profile-cuda-sync false --verify-cost-model-profile false \
  --transfer-aware-profile false --save-profile-json false \
  --save-token-ids true --save-text true --reset-seed-after-warmup true \
  --seed 20260719 --fail-fast true \
  --fail-on-output-validation-error true
```

## 测试

```text
python -m pytest -q \
  tests/test_verify_segment_graph.py \
  tests/test_eval_tpot_config.py \
  tests/test_config_predictive_prefetch.py \
  tests/test_predictive_prefetch.py \
  tests/test_cache_lut.py \
  tests/test_expert_cache_staging.py

103 passed
```

测试覆盖默认关闭时顺序完全不变、单一 source 不重排、vpb2 前缀为一个 history 加一个
draft、真实 submit source 为 1+1、独立 preset 参数以及全部 built-in preset 仍严格为
16 total threads。

## 决定

- 保留本实现和独立 `source11` preset，作为新的同资源一请求最低点；
- 保留原 `...ghost8_lutfuse`、ghost8、b1、b2 和 budget4 全部 fallback；
- 不增加 prefetch budget，不压缩或转换权重；
- 下一优化方向在开始前重新汇总候选、按预期收益排序并等待用户确认。
