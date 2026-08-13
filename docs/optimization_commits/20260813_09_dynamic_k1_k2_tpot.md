# Retained best dynamic draft length: conservative TPOT K1/K2

## 目标与算法边界

本优化保留现有的 acceptance predictor + TPOT `first_increase` 框架，没有用另一套
controller 替代动态算法。当前 F16/active12 工作点的固定 K 扫描证明 K1 最优，旧动态
配置却使用 `Kmax=6, td/tv=19/80`。在 K1 决策点，静态模型只有在
`predicted alpha < td/tv` 时停止；旧阈值 0.2375 因而把平均 K 推到 2.45。

新动态 preset `k2_dynamic_f16_3080` 使用：

- `Kmax=2`、`min_steps=1`、`first_increase`；
- `td=97, tv=100`，即 K1 后仅当 predicted alpha `>=0.97` 才进入 K2；
- 原 acceptance predictor、standard speculative sampling 和 TPOT 公式不变；
- F16/single-weight、active12/staging0、segment16、vpb2、fixed grouped GEMM；
- verify graph buckets `2,3`；`gpu_memory_utilization=0.97` 为第二个 graph bucket
  留出空间，KV 仍为 38 blocks / 9728 tokens，大于完整 `max_model_len=8192`。

公式上，K1 候选为 `(97+100)/(1+alpha)`，T0 为 `100`，所以继续条件正好是
`alpha >= 0.97`。它不是把动态模式强制退化成 fixed K1：三次 512-token 测试均实际
出现 K2。

## 参数筛选（256-token）

| 配置 | TPOT | 实际策略行为 |
|:---|---:|:---|
| 旧动态 Kmax6，阈值 19/80 | 83.627 ms | 平均 K=2.45 |
| predictor Kmax1 成本对照 | 70.426 ms | 始终 K1 |
| Kmax2，阈值 0.90 | 68.612 ms | K1=122, K2=22 |
| Kmax2，阈值 0.95 | 69.322 ms | K1=148, K2=2（单次波动较大） |
| **Kmax2，阈值 0.97** | **66.245 ms** | **K1=147, K2=2；K2 均全接受** |

0.95 和 0.97 单次都只有两轮 K2，不能用单次差异证明 0.97 本身更快；选择 0.97 的
依据是更保守、与 K1 最优的算子实测一致，然后用三次 512-token 结果验收整体配置。

## 三次独立 512-token 验收

| repeat/seed | TPOT | actual K1 | actual K2 | K2 acceptance |
|---:|---:|---:|---:|:---|
| 0 | 66.031 ms | 307 | 1 | 1 token |
| 1 | 68.461 ms | 307 | 4 | 4/4 full |
| 2 | 65.199 ms | 292 | 4 | 3 full, 1 partial |
| **mean** | **66.564 ms** | | | |

Population std 为 `1.384 ms`。三轮均生成恰好 512 token，
`output_fixed_length_ok=true`、`output_validation_error=""`。repeat 0/2 各有一个
terminal K0，是剩余 output budget 已不足以再放一个 draft+verify token，不是策略选择。

相对旧动态 `83.627`，动态最优降低 **20.4%**；相对当前 fixed K1 三次均值
`65.443` 仅慢 **1.71%**。因此它达到“接近当前总体最优”，但没有证据取代 fixed K1
作为默认总体最优。它作为单独 preset 永久保留，便于动态长度研究和不同 workload
复测。

结果：

- `results/dynamic_tpot_k2_threshold{090,095,097}_gmu097_256/`
- `results/dynamic_tpot_k2_threshold097_gmu097_512_r{0,1,2}/`
- predictor K1 对照：`results/dynamic_current_predictor_overhead_k1_256/`
- 旧动态：`results/single_weight_f16_k6_dynamic_default_256/`

## 复现与测试

每个 repeat 使用独立进程；把 `<R>` 分别替换为 `0,1,2`：

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
  /home/edge/.conda/envs/nano_moe/bin/python \
  scripts/bench_eval_workload_tpot.py \
  --request-mode per_layer_slots \
  --model-path /home/edge/models/Qwen3-30B-A3B \
  --output-dir results/dynamic_k2_best_512_r<R> \
  --optimized-config k2_dynamic_f16_3080 \
  --output-lens 512 --temperature 0.6 \
  --repeat-index-offset <R> \
  --kt-llamafile-extension-path \
    /home/edge/zx/ktransformers/build/lib.linux-x86_64-cpython-312/cpuinfer_ext.cpython-312-x86_64-linux-gnu.so \
  --kt-single-weight true \
  --collect-profile true --save-profile-json true \
  --save-token-ids true --fail-fast true
```

```bash
PYTHONPATH=. /home/edge/.conda/envs/nano_moe/bin/python -m pytest -q \
  tests/test_eval_tpot_config.py \
  tests/test_bench_eval_workload_tpot.py \
  tests/test_spec_engine_flow.py \
  tests/test_grouped_gemm_fixed_config.py
```

结果：`49 passed`。测试覆盖 preset 的完整参数、显式 cost override、predictor 启用、
verify bucket 覆盖和 fixed grouped GEMM 环境。
