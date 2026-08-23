# 40. 否决 verify draft-first 双位置预取

日期：2026-08-23

## 一句话总结

把固定 vpb2 的两个位置都优先分配给 `draft_live`，公平 16-thread 单请求 TPOT 从
**51.793570** 回退到 **56.910824 ms/token（+9.88%）**；候选运行时代码和 preset 已完整
撤销，继续保留 history + draft 1+1 的 `source11` 最优配置。

## 选择依据与离线上界

上一轮固定 vpb2 的 source-aware 1+1 已把当前 verify step 的 route demand 从 1,462 提高到
1,703。本轮只改变这两个 dispatch 位置的 source 构成，离线重排结果为：

| vpb2 排序 | 当前 step route demand | 命中候选数 |
|:---|---:|---:|
| 当前 global priority（两个 history） | 1,462 | 908 |
| 已保留 source-aware 1+1 | 1,703 | 1,078 |
| 本轮 draft-first（最多两个 draft） | **1,914** | **1,243** |
| 无预算候选池 oracle | 2,142 | — |

只属于 `draft_live` 的边际候选对当前 step 有 97.377% candidate precision，因此 draft-first
是当时固定 vpb2 内最高的当前-step 离线覆盖候选。它不增加 prefetch budget、cache slots、
H2D dispatch 上限或 draft forward，也不改变权重表示。

## 已尝试实现与撤销

- 扩展 `source_aware_verify_candidates()`，允许 `verify_prefetch_draft_reserve=2`；
- dispatch prefix 先放最多两个最高优先级 `draft_live`，draft 不足时再由
  `verify_history` 和原 global-priority 顺序补齐；
- 新增独立 `..._draftfirst` preset，继承 source11 的全部配置，只把 reserve 从 1 改成 2；
- 增加 helper、真实 submit 和 preset 回归测试。

端到端门禁为负后，上述 helper 语义、preset 和测试均已完整撤销。保留实现仍将 reserve
限制为 0 或 1，从而确保 source-aware 模式至少留下一个 history 位置。工作树中不存在
`draftfirst` preset 或 reserve=2 的运行时路径。

## 公平单请求 TPOT 门禁

共同口径为 MMLU-Pro validation sample 0、107-token prompt、seed 20260719、temperature
0.6、固定生成 512 token、single-weight llamafile F16、profile 全关、CUDA 0、active14，
CPUInfer 双 NUMA `2 x 8 = 16` total threads，以及固定 verify prefetch budget 2。候选
metadata 明确记录：

```text
kt_num_threads=16
kt_threadpool_count=2
kt_numa_nodes=[0, 1]
kt_direct_backend=llamafile_f16
kt_single_weight=true
verify_prefetch_max_per_boundary=2
verify_prefetch_draft_reserve=2
```

| 配置 | TPOT | decode tok/s | decode | steps | mean step | p50/p90/p95/max |
|:---|---:|---:|---:|---:|---:|:---|
| **保留 source-aware 1+1** | **51.793570 ms** | **19.307** | **26.467 s** | **254** | **104.199 ms** | **97.350/135.509/142.503/186.617 ms** |
| draft-first | 56.910824 ms | 17.571 | 29.081 s | 266 | 109.329 ms | 101.971/146.350/153.592/180.167 ms |
| draft-first 变化 | **+5.117254 ms / +9.88%** | **-8.99%** | **+2.615 s / +9.88%** | **+12 / +4.72%** | **+4.92%** | p50/p90/p95 +4.75/+8.00/+7.78%，max -3.46% |

候选生成 512/512 token，`output_fixed_length_ok=true`、validation error 为空；输出 digest 为
`28cf5b8c863ee8b1accb34ddceac3f04ee3ed703f75f49abb5d3c8da165bf183`。结果目录：

```text
results/tpot_t16_b1_ghost8_lutfuse_draftfirst_20260823/
```

## 为什么离线覆盖增加却端到端回退

离线指标只统计“本轮候选是否会被当前 verify step 使用”，没有计入候选进入 active cache
后的驻留寿命、victim 选择、跨步复用、transfer 与 CPU/GPU overlap。draft-first 完全挤掉
history 后，不仅 decode steps 增加 4.72%，单步 mean、p50、p90 和 p95 也全部变差；因此
9.88% 回退不是只由随机轨迹多出若干 rounds 造成，而是同时出现了广泛的单步时延损失。

已有全局 trace 中，verify history 对 verify expert-set 的 recall 为 46.30%，draft predict
为 43.30%。97.377% 的 draft 边际 candidate precision 证明 draft 是高价值的互补 source，
却不能证明它可以替代 history：history 提供的是 draft 当前 token 没覆盖的跨步稳定信息。
本轮 digest 与 source11 不同，仍不能把差值严格拆成单一算法因果；但 9.88% 的大幅回退与
单步分位数一致恶化足以否决该候选。

## 复现

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
  /home/edge/.conda/envs/nano_moe/bin/python \
  scripts/bench_eval_workload_tpot.py \
  --request-mode dataset --dataset mmlu_pro --num-samples 1 \
  --model-path /home/edge/models/Qwen3-30B-A3B \
  --mmlu-pro-path \
  results/k2_dynamic_f16_3080_eval_workloads_20260817/inputs/mmlu_pro_validation.jsonl \
  --output-dir results/tpot_t16_b1_ghost8_lutfuse_draftfirst_20260823 \
  --optimized-config \
  k2_dynamic_f16_3080_active14_phase1_recent_b1_ghost8_lutfuse_draftfirst \
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

该命令用于记录已撤销候选，当前 HEAD 不再提供 `..._draftfirst` preset。

## 测试

候选实现门禁前完整相关测试为 `106 passed`。撤销后对保留运行时重新执行同一组测试：

```text
103 passed in 7.88s
```

它验证 source11、所有 fallback 和 16-thread 资源约束未被候选残留污染。

## 决定

- 不保留 draft-first 运行时代码或 preset，只提交本负结果文档；
- `k2_dynamic_f16_3080_active14_phase1_recent_b1_ghost8_lutfuse_source11` 继续作为当前最低点；
- 原 lutfuse、ghost8、b1、b2、budget4、active14 和 full-context-safe 动态配置全部保留；
- 不增加预算，不修改或压缩权重；后续 source 优化必须保留 history，或使用能显式估计
  cache 驻留和 exposed tail 的条件 gate，而不能只按当前-step candidate precision 替换。
