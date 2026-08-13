# Workload-sized dynamic K1/K2 active14 optimum

## 目标与边界

本优化继续使用已有 acceptance predictor + TPOT `first_increase` 动态长度框架，
不改变采样语义、预测器或停止公式。在已经保留的 full-context-safe active12 配置上，
利用 workload-sized warmup 释放出的显存，把每层常驻专家从 12 增至 14，并把
`gpu_memory_utilization` 从 0.97 调到 0.98。动态参数仍为 `Kmax=2`、
`td/tv=97/100`、`min_steps=1`。

新 preset 为 `k2_dynamic_f16_3080_active14`。原来的
`k2_dynamic_f16_3080` 保持不变，继续作为可覆盖 8192-token KV 的安全版本；新
preset 是当前实测 workload-sized 动态 TPOT 最优。

## 筛选

同一 seed、256 输出 token 的单点筛选：

| active experts | GPU utilization | KV capacity | TPOT |
|---:|---:|---:|---:|
| 12 | 0.97 | 9728 tokens | 66.245 ms |
| 13 | 0.97 | 5120 tokens | 67.178 ms |
| 14 | 0.97 | 512 tokens | 65.385 ms |

active13 为负收益。active14/0.97 虽更快，但 512-token KV 无法覆盖 67-token prompt
加 512-token 输出，不能作为可用的 512-token 配置。把 utilization 调至 0.98 后获得
6 个 256-token block，即 1536-token KV，足够本次 579-token 序列。这里必须注意：
CLI 的 `max_model_len=8192` 仍是模型上限，物理 KV 容量只有 1536 token；该 preset
不能冒充完整 8192-context 配置。

## 三次独立 512-token 验收

| seed | TPOT | actual K1 | actual K2 | K2 acceptance |
|---:|---:|---:|---:|:---|
| 0 | 63.930 ms | 286 | 7 | 4 full，3 partial |
| 1 | 61.600 ms | 294 | 1 | 1 full |
| 2 | 65.609 ms | 294 | 4 | 4 full |
| **mean** | **63.713 ms** | | | |

Population std 为 `1.644 ms`。12 次 K2 中 9 次 full accept、3 次 partial accept、
没有 zero accept，证明策略真实动态且保守门限有效。三轮都生成恰好 512 token，
`fixed_length_ok=true`、validation error 为空。seed 1/2 各有一个 terminal K0，是
剩余输出预算不足一个 draft+verify token，并非策略选择。

相对原 active12 动态均值 `66.564 ms` 改善 **4.28%**；相对此前 fixed K1 总体
最优均值 `65.443 ms` 改善 **2.64%**。它也比 KTransformers 历史 F16
`81.90 ms` 快 **22.2%（1.29x）**，比可靠 BF16 `122.35 ms` 快
**47.9%（1.92x）**。

三轮 profile 均值从 active12 到 active14 的变化为：draft forward
`30.777 -> 30.546 ms/call`，verify forward `78.360 -> 77.230 ms/call`，
realized CPU experts `57.403 -> 56.153`，CPU route ratio
`0.9068 -> 0.8952`，prefetch submits/MoE profile call `77.65 -> 75.35`。
CPU compute 是异步累计值，不能把其单项变化直接解释成墙钟收益；可确认的机制是更多
常驻专家减少了 CPU route 与预取工作，并在端到端 TPOT 上三轮均值为正。

结果：

- `results/dynamic_k2_active13_threshold097_256/`
- `results/dynamic_k2_active14_threshold097_256/`
- `results/dynamic_k2_active14_threshold097_gmu098_512_r{0,1,2}/`

## 复现与测试

把 `<R>` 替换为 `0,1,2`，每轮使用独立进程：

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
  /home/edge/.conda/envs/nano_moe/bin/python \
  scripts/bench_eval_workload_tpot.py \
  --request-mode per_layer_slots \
  --model-path /home/edge/models/Qwen3-30B-A3B \
  --output-dir results/dynamic_active14_512_r<R> \
  --optimized-config k2_dynamic_f16_3080_active14 \
  --output-lens 512 --temperature 0.6 --repeat-index-offset <R> \
  --kt-llamafile-extension-path \
    /home/edge/zx/ktransformers/build/lib.linux-x86_64-cpython-312/cpuinfer_ext.cpython-312-x86_64-linux-gnu.so \
  --kt-single-weight true --collect-profile true \
  --save-profile-json true --save-token-ids true --fail-fast true
```
