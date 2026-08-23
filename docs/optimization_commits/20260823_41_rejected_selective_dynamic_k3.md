# 41. 否决选择性动态 K3

日期：2026-08-23

## 一句话总结

在当前 `source11` 最优上只把 Kmax 2→3、增加 qlen4 verify graph，让现有
`first_increase` 选择高置信 K3，公平 16-thread 单请求 TPOT 仍从 **51.793570** 回退到
**52.733818 ms/token（+1.82%）**；候选 preset 与测试已完整撤销，K1/K2 最优配置继续保留。

## 选择依据

此前 t16 trace 把 Kmax 离线放宽到 3 时，只有 12/95 个 K2 rounds 会继续 K3，占全部
rounds 的 1.51%，而这 12 轮的前两个 draft 全部真实接受。早期旧路径 fixed K3 也曾比
fixed K2 快约 1.9%，因此这是比固定 K3 更保守、且完全沿用当前动态长度框架的候选。

已有成本边界同时说明风险：新增一个 warm draft 约 19.46 ms；再计入 qlen3→qlen4 verify
增量后，第三 draft 的条件接受率需要约 41.4%--72.7% 才可能持平。因此本轮不修改静态
`td/tv=97/100` 或 0.97 隐含门限，只检验现有 `first_increase` 的自然 K3 扩展。

## 已尝试实现与撤销

新增独立候选：

```text
k3_dynamic_f16_3080_active14_phase1_recent_b1_ghost8_lutfuse_source11
```

它逐项继承当前最佳 `...source11`，仅修改：

```text
max_draft_tokens_values=3
verify_cuda_graph_bucket_steps=2,3,4
```

现有控制器在获得第一步 alpha 后比较 T(1)/T(0)，获得第二步 alpha 后比较 T(2)/T(1)；
只有后者仍不增加时才执行第三个 draft。qlen4 graph 与 qlen2/3 共用 graph pool，避免 K3
轮次回退 eager。测试锁定 Kmax、graph buckets、F16、source11、cache/prefetch 继承关系和
16-thread 资源限制。

端到端门禁为负后，候选 preset、choice、固定 grouped-GEMM 环境映射及新增测试均已完整
撤销；没有修改通用 `first_increase` 控制器。当前 HEAD 不提供该 preset。

## 公平单请求 TPOT 门禁

共同口径为 MMLU-Pro validation sample 0、107-token prompt、seed 20260719、temperature
0.6、固定生成 512 token、single-weight llamafile F16、profile 全关、CUDA 0、active14、
vpb2/source11，以及 CPUInfer 双 NUMA `2 x 8 = 16` total threads。候选 metadata 明确记录：

```text
kt_num_threads=16
kt_threadpool_count=2
kt_numa_nodes=[0, 1]
kt_direct_backend=llamafile_f16
kt_single_weight=true
max_draft_tokens=3
verify_cuda_graph_bucket_steps=[2, 3, 4]
verify_prefetch_max_per_boundary=2
verify_prefetch_draft_reserve=1
```

qlen4 graph capture 成功；active14 仍分配 6 个 256-token KV blocks，即 1536-token capacity，
没有为了 K3 减少 GPU expert cache、KV 容量或改变资源。

| 配置 | TPOT | decode tok/s | decode | steps | mean step | p50/p90/p95/max |
|:---|---:|---:|---:|---:|---:|:---|
| **保留 source11 K1/K2** | **51.793570 ms** | **19.307** | **26.467 s** | 254 | **104.199 ms** | **97.350/135.509/142.503/186.617 ms** |
| 选择性动态 K3 | 52.733818 ms | 18.963 | 26.947 s | **248** | 108.657 ms | 99.368/143.940/160.152/203.706 ms |
| K3 变化 | **+0.940248 ms / +1.82%** | **-1.78%** | **+0.480 s / +1.82%** | **-6 / -2.36%** | **+4.28%** | p50/p90/p95/max +2.07/+6.22/+12.38/+9.16% |

候选生成 512/512 token，`output_fixed_length_ok=true`、validation error 为空；输出 digest 为
`04d49ce94d6ff97c84c1dd9819604da6ed91cc611f07d0f4a263a28e16a93e34`。正式结果目录：

```text
results/tpot_t16_b1_ghost8_lutfuse_source11_selectivek3_20260823/
```

decode rounds 的确减少，但 mean 和全部 tail 分位数的增加更大，最终 decode 时间增加
0.480 s。候选与 source11 digest 不同，不能把 1.82% 严格拆成单一参数因果；不过减少 rounds
仍无法抵消广泛的单步回退，已足以否决本候选。

## analysis-only K 分布与边际成本

production-off 门禁按约定不保存 profile，无法从其 artifact 还原逐轮 K。撤销前另跑一条
`collect_profile=true`、`engine_profile=false`、同步 instrumentation 全关的 analysis-only
trace；它只用于确认控制器行为，不替代上面的性能门禁。目录：

```text
results/analysis_t16_b1_ghost8_lutfuse_source11_selectivek3_20260823/
```

该 trace 的 253 个 speculative rounds 中，K1/K2/K3 为 200/40/12，另有最后一个剩余预算
不足的 K0；K3 占 4.74%，高于旧 trace 离线预测的 1.51%，所以候选并非“没有触发”。接受
prefix 分布为：

| 实际 K | rounds | 接受 prefix 分布 | output/round | round wall | wall/output |
|---:|---:|:---|---:|---:|---:|
| K1 | 200 | 0: 51，1: 149 | 1.745 | 99.364 ms | 56.942 ms |
| K2 | 40 | 1: 5，2: 35 | 2.875 | 131.402 ms | **45.705 ms** |
| K3 | 12 | 1: 1，3: 11 | **3.833** | 177.838 ms | 46.392 ms |

K3 的第三 token 在 12 轮中接受 11 次（91.67%），且前两个都接受时第三个为 11/11；回退
因此也不能归因于第三 token 接受率不足。K3 三次 draft call 平均为 20.435/20.523/
23.164 ms，总 draft 64.122 ms；verify-ready 为 111.349 ms。相对 K2 观察组，K3 每轮多
46.436 ms、平均多 0.958 output，边际约 48.45 ms/output，高于 K2 组已有的 45.705
ms/output，因此即使接受率很高也略微拉高该高置信组的单位成本。

这些 K 组由控制器选择且 analysis digest 与正式门禁不同，不是同状态 K2/K3 反事实 A/B；
但它清楚表明剩余缺口是第三 draft + qlen4 的真实边际成本，而不是触发率或接受率。继续扫
K3 静态门限只能减少触发，无法在执行第三 draft 之前观察 alpha3；下一代动态 K 若重启，
需要预先预测 alpha3、qlen4 verify cost 和 cache/prefetch credit，或先降低 qlen4/native
执行成本。

## 复现

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
  /home/edge/.conda/envs/nano_moe/bin/python \
  scripts/bench_eval_workload_tpot.py \
  --request-mode dataset --dataset mmlu_pro --num-samples 1 \
  --model-path /home/edge/models/Qwen3-30B-A3B \
  --mmlu-pro-path \
  results/k2_dynamic_f16_3080_eval_workloads_20260817/inputs/mmlu_pro_validation.jsonl \
  --output-dir \
  results/tpot_t16_b1_ghost8_lutfuse_source11_selectivek3_20260823 \
  --optimized-config \
  k3_dynamic_f16_3080_active14_phase1_recent_b1_ghost8_lutfuse_source11 \
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

该命令用于记录已撤销候选，当前 HEAD 不再提供该 preset。

## 测试

候选实现时相关测试为 `133 passed in 7.17s`。撤销后重新运行同一组测试：

```text
131 passed in 7.14s
```

它验证 source11、全部动态 K1/K2 fallback 和 16-thread 资源约束未被候选残留污染。

## 决定

- 不保留选择性 K3 preset 或 qlen4 graph 扩展，只提交本负结果与 analysis；
- 当前推荐仍为
  `k2_dynamic_f16_3080_active14_phase1_recent_b1_ghost8_lutfuse_source11`，TPOT
  51.793570 ms/token；
- 保留所有既有动态 draft 最优/fallback，不增加预算，不修改或压缩权重；
- 不再盲扫 K3 静态阈值；只有得到执行前 alpha3/qlen4 边际成本预测，或显著降低第三步
  native 成本后，才重新评估 K3。
