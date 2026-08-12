# RTX 3080 低缓存比例 TPOT 定位与优化

> **2026-08-12 更新：本文的 taskset、K6 最优和“100 ms 不可达”结论已被后续
> 受控实验推翻，请勿继续使用本文末尾命令。** 外层 `taskset` 会与 CPUInfer 的
> NUMA worker 绑核冲突；改用 KTransformers 本机修复后的 legacy kernel、取消外层
> 绑核并重新筛选后，K3/512-token TPOT 为 `109.946 ms`。true batch 的 exact F16
> 路线也已在 B2/B3/B5 的平均 TPOT/吞吐上达到或超过 KT 基线。完整证据、限制和新
> 命令见 `docs/tpot_vs_ktransformers_20260812.md`。

日期：2026-08-11  
分支：`refactor`

## 结论

原命令的首要问题不是随机抖动，而是 verify CUDA Graph bucket 少了一个
token：固定 `K=6` 的每轮 verify 实际处理 `6 + 1 = 7` 个 token，但命令只
捕获了 `3,4,5,6`。因此 213/216 个 verify 调用静默退回 eager，修正为包含
7 后，512-token 单请求 TPOT 从 398.234 ms 降到 270.350 ms，改善 32.1%。
关闭 profiling 后的最终 512-token 验收为 **258.413 ms/token**（181 decode
steps，512/512 tokens 输出有效），相对错误 bucket profile 改善 35.1%。

修正 Graph 后，剩余瓶颈是 7.5% GPU expert cache 下的大量 CPU expert 计算，
而不是 PCIe 传输：完整 latency breakdown 中 verify CPUInfer 的暴露时间为
186.951 ms/output-token，占 69.2%；传输只有 0.249 ms/token。当前机器的
10 GiB RTX 3080 无法容纳与 4090 实验相同的 31.25% expert cache，且两台
机器的 CPU/PCIe 也不同，因此历史 4090 结果不能作为仅更换 GPU 的等配置
对照。

## 实验环境

| 项目 | 当前服务器 | 仓库内历史 4090 smoke |
|---|---|---|
| GPU | RTX 3080 10 GiB | RTX 4090 24 GiB |
| PCIe | Gen3 x16（GPU 1） | 文档未记录 |
| CPU | 2 × Xeon Gold 5218R，20 物理核/socket | 文档未记录型号 |
| CPU ISA | AVX2/AVX-512，无 AMX/BF16 CPU flag | 未记录 |
| expert cache ratio | 0.075，共 480 slots | 0.3125，共 1920 slots |
| KT backend | `avx2_bf16` | `avx2_bf16` |
| 请求 | per-layer-slots，67 input + 512 output | MT-Bench，512 output |

历史结果见 `docs/tpot_performance_breakdown_smoke_20260727.md`。其中 P4/P5
约为 37/34 ms，但 cache ratio、请求、CPU 和机制均不相同，不能据此断言
3080 单卡 GPU 算力造成了全部 3–5 倍差距。

## 根因一：verify bucket 的 off-by-one

投机轮次会把 `len(draft_tokens) + 1` 写入 `verify_lengths`。Graph fast path
只有在至少一个 bucket 不小于该长度时才可用，因此固定 K 的配置约束是：

```text
max(verify_cuda_graph_bucket_steps) >= max_draft_tokens + 1
```

错误配置 profiling：

| 指标 | `3,4,5,6`（错误） | `3,4,5,6,7`（修正） |
|---|---:|---:|
| 输出有效性 | 512/512，有效 | 512/512，有效 |
| TPOT | 398.234 ms | 270.350 ms |
| decode steps | 216 | 179 |
| acceptance rate | 0.229 | 0.310 |
| verify graph hits | 3/216 | 179/179 |
| segment graph replays | 12 | 716（179 × 4） |
| GPU utilization（观察值） | 约 4% | 约 60% |

两次采样轨迹不同，所以 acceptance/decode steps 会带入随机差异；但 graph
计数直接证明了配置错误，且修正后的每一次 verify 都走 graph。

为防止同类问题再次发生，benchmark 配置解析现在会拒绝未覆盖 `K+1` 的
bucket，并给出会退回 eager 的明确错误；默认 bucket 也从 `3,5,8,12`
改为 `3,5,8,13`，覆盖默认 K=12。

## 根因二：低缓存比例下的 CPU expert 路径

修正 bucket 后的累积 latency breakdown：

| 每个输出 token 的暴露延迟 | ms | 占比 |
|---|---:|---:|
| Draft GPU | 40.573 | 15.0% |
| Draft transfer | 0.072 | 0.0% |
| Draft other | 23.954 | 8.9% |
| Verify GPU | 11.327 | 4.2% |
| Verify CPUInfer | 186.951 | 69.2% |
| Verify transfer | 0.249 | 0.1% |
| Verify other | 7.169 | 2.7% |
| Residual | 0.055 | 0.0% |
| **总计** | **270.350** | **100%** |

运行中累计观察到 292,444 个 verify cache misses、164,749 个 realized CPU
experts，CPU route ratio 约为 0.611。GPU power/utilization 在错误 graph 时很
低；修正后 GPU 被有效利用，但 CPUInfer 仍决定 critical path。PCIe copy
仅占 0.1%，所以继续微调异步 copy 不可能把 270 ms 降到 100 ms。

一个直接的下界是：即使把除 verify CPUInfer 外的所有成本都降为零，当前
执行轨迹仍需 186.951 ms/token。因此在 cache ratio=0.075、当前 CPU kernel
和线程配置不变时，`TPOT < 100 ms` 在理论上不可达。

## 已验证的替代方案

### Exact heterogeneous autoregressive

关闭 speculative 路径、运行 exact heter AR graph 得到 408.370 ms/token，
比修正后的 K=6 更慢。原因是 batch=1 时每 token 都要进行大量 CPU expert
计算；它没有 drafter 的 amortization，不能解决低缓存比例瓶颈。

### Draft K 筛选

为避免每个 K 重载约 125 GiB host weights，使用同一 engine 对 128-token
请求做方向性筛选。每个 K 都生成了固定 128 tokens 且通过输出校验：

| K | TPOT | decode steps |
|---:|---:|---:|
| 1 | 347.725 ms | 74 |
| 2 | 252.599 ms | 58 |
| 3 | 254.201 ms | 56 |
| 4 | 253.526 ms | 54 |
| 6 | **227.936 ms** | 41 |

K=6 仍是当前配置的最佳点。较小 K 减少每轮 verify 工作，但需要更多 verify
轮次，CPU 启动/同步成本抵消了收益。由于 stochastic sampling 和 engine
cache 状态会随 case 演化，此表用于筛选，不作为严格置信区间。

### KT 线程数

把 CPU affinity 从双 NUMA 各 8 个物理核扩到各 16 个，并把
`--kt-num-threads` 从 16 提高到 32 后，K=6/128-token TPOT 为 243.669 ms；
它只需 37 个 decode steps，却比 16-thread 筛选的 41 steps / 227.936 ms
更慢。不同 sampling 轨迹不适合直接比较 TPOT 百分比，但“更少 step、更多
总时间”足以说明单 step 已退化。该 AVX2 kernel 在当前双 socket 平台上受
内存带宽和线程同步约束，继续增加线程不是有效优化，推荐保留总 16 线程、
两个 subpool 各 8 线程。

## 优化建议

1. 所有 K=6 运行必须使用包含 7 的 verify bucket，例如 `3,4,5,6,7`。
2. 使用两个 NUMA subpool，并让 taskset 与 `--kt-numa-nodes 0,1` 一致；不要
   把双 socket 内存上的约 125 GiB expert weights 只交给一个 NUMA pool。
3. 保持总 16 个 KT threads 和 K=6；32 threads、K=1–4 和 exact AR 均已
   实测更慢。
4. 若必须达到 100 ms，需要改变主要约束，而不是只调 Graph：提高 GPU
   expert cache、使用更快的 CPU MoE kernel/AMX 平台、量化 CPU experts，或
   换到能容纳更多 experts 的 24 GiB GPU。按测得下界，CPU 路径自身至少
   需要 1.87 倍加速才可能刚好越过 100 ms；实际还需削减约 83 ms 的其他
   开销，而当前端到端 TPOT 则需约 2.70 倍加速。

## 最终验收命令

这台机器只有 CPU 0–79；原来的 `taskset --cpu-list 64-96` 实际只会留下
64–79，并且都属于 NUMA 1 的 SMT sibling。下面使用两个 socket 各 8 个
物理核，与两个 KT subpool 对齐：

```bash
NANOVLLM_VERIFY_PREFETCH_RANK_MULTIPLIER=1 \
NANOVLLM_VERIFY_DEFER_SEGMENT_METADATA=1 \
NANOVLLM_VERIFY_BOUNDARY_PREFETCH_ASYNC=0 \
CUDA_VISIBLE_DEVICES=1 \
taskset --cpu-list 0-7,20-27 \
conda run --no-capture-output -n nano_moe \
python scripts/bench_eval_workload_tpot.py \
  --request-mode per_layer_slots \
  --model-path /home/edge/models/Qwen3-30B-A3B \
  --output-dir results/final_k6_r075_bucket7_3080 \
  --optimized-config k6_decode \
  --cache-ratios 0.075 \
  --output-lens 512 \
  --max-draft-tokens-values 6 \
  --segment-sizes 12 \
  --allocation-modes profile_weighted \
  --slot-buckets 4 \
  --slot-max-bucket-ratio 2.0 \
  --slot-profile-csv pre_exps/exp_and_figs/unique/unique_count_plot_summary_n1024.csv \
  --cpu-expert-pin-memory false \
  --kt-num-threads 16 \
  --kt-threadpool-count 2 \
  --kt-numa-nodes 0,1 \
  --kt-direct-backend avx2_bf16 \
  --verify-cuda-graph-bucket-steps 3,4,5,6,7 \
  --verify-prefetch-max-per-boundary 4 \
  --verify-prefetch-rank-multiplier 1 \
  --draft-stop-policy none \
  --acceptance-predictor-enabled false \
  --gpu-memory-utilization 0.996 \
  --temperature 0.8 \
  --acceptance-strategy standard_sampling \
  --decode-driver generate \
  --reuse-engine-across-draft-lengths true \
  --collect-profile false \
  --save-token-ids true \
  --save-text true \
  --reset-seed-after-warmup true \
  --skip-existing false \
  --fail-fast true \
  --seed 20260719 \
  --dist-port-base 38170
```

最终结果：decode 132.049 s，TPOT 258.413 ms，3.870 output token/s，输出
digest 为 `8ef0c43eaa313ac8535f10511755a0d033bf92d27fb7fd853635adc6548402c0`。

## 结果位置

- 错误 bucket profile：`results/profile_k6_r075_3080/`
- 修正 bucket profile：`results/profile_k6_r075_bucket7_3080/`
- exact AR profile：`results/profile_heter_ar_r075_3080/`
- K 筛选：`results/screen_k_r075_3080/`
- 32-thread 对照：`results/screen_k6_threads32_r075_3080/`
- 最终无 profiling 验收：`results/final_k6_r075_bucket7_3080/`

`results/` 默认不进入源码提交；文档保留了可复查的关键指标和配置依据。
