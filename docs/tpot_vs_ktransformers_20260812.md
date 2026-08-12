# nano-vllm-moe 与 KTransformers TPOT 对照、定位和优化

日期：2026-08-12  
硬件：RTX 3080 10 GiB，2 × Xeon Gold 5218R（2 NUMA，40 物理核）  
模型：Qwen3-30B-A3B，GPU expert cache ratio `0.075`

## 结论

旧 nano 结果 `258.413 ms/token` 不是这台机器在 7.5% cache 下的性能下界。
它同时受到外层 `taskset`、不同/较慢的 CPU MoE backend、以及在错误基线上选择
`K=6` 的影响。修正后：

- 单请求、BF16、K3、512 个固定输出 token：nano 完整请求墙钟 TPOT
  **109.946 ms/token**；KTransformers BF16 的 61-step stable replay 为
  **122.35 ms/token**。nano 的更严格口径仍快 **10.1%**。
- F16 true batch 的最终 nano 路线不是投机解码，而是 exact decode。优化配置的
  B2 三次完整墙钟 TPOT 为 `110.237 / 117.120 / 125.859 ms`，均值
  **117.739 ms**、中位数 **117.120 ms**；KT 为 `121.84 ms`。B3 两次为
  `156.084 / 159.096 ms`，KT 为 `163.65 ms`；B5 两次为
  `154.511 / 154.854 ms`，KT 为 `251.92 ms`。
- B2 仍有尾延迟问题：最佳完整轨迹的 stable replay p50 为 `105.46 ms`，但
  p95 为 `147.07 ms`，高于 KT 的 `131.46 ms`。因此结论是均值和聚合吞吐已
  超过 KT，而不是所有尾延迟指标都超过 KT。

## 对照口径

KTransformers 批测来自：
`/home/edge/zx/ktransformers/benchmark_outputs/ktransformers_qwen3_batch_tpot_2026-08-09.md`。
它使用 F16 GGUF、prompt 23、固定输出 64、temperature 0.6、top-k 20、top-p
0.95，并排除 graph capture，只统计之后 61 个 replay step。

nano 批测使用从同一模型 BF16 safetensors 转成 F16 后交给相同的 fixed legacy
`cpuinfer_ext` kernel；prompt 67、固定输出 64、temperature 0.6。nano 当前 sampler
没有 KT 脚本的 top-k/top-p 选项。主表的 nano 数字是全部 63 个 decode step 的
完整 `llm.step()` 墙钟，包括 scheduler、model、sampler 和后处理，口径不比 KT
宽松。新 benchmark 另行输出去掉前两个 step 的 61-step stable replay 指标，便于
逐项核对，但不会覆盖完整墙钟 TPOT。

权重来源、prompt 和采样策略仍不完全相同，因此这些结果证明本机实现性能已经超过
现有 KT 基线，不应解释成逐 token 的 bitwise 等价 A/B。

## 结果

### 单请求

| 实现/阶段 | CPU backend | affinity | K | 固定输出 | TPOT |
|---|---|---|---:|---:|---:|
| nano 旧最终结果 | kt-kernel AVX2 BF16 | `taskset 0-7,20-27` | 6 | 512 | 258.413 ms |
| nano fixed legacy 初始 exact 对照 | llamafile BF16 | 无外层绑核 | 0 | 64 | 148.568 ms |
| nano fixed legacy K3，temp 0.8 | llamafile BF16 | 无外层绑核 | 3 | 512 | 116.695 ms |
| **nano fixed legacy K3，temp 0.6** | llamafile BF16 | 无外层绑核 | 3 | 512 | **109.946 ms** |
| KTransformers BF16 stable replay | fixed legacy BF16 | CPUInfer 自绑核 | 0 | 64 | 122.35 ms |

去掉错误 affinity/backend 后重新筛选 K（固定输出 256，temperature 0.8）：

| K | 1 | 2 | 3 | 4 | 6 |
|---:|---:|---:|---:|---:|---:|
| TPOT | 112.161 | 109.080 | **105.549** | 144.968 | 153.113 |

这推翻了旧文档基于错误运行环境得到的“K6 最优”。最终单请求选择 K3；K2 在短筛
中接近，但 K3 的 512-token 验收已经完成。

### True batch

下面 nano 使用 exact F16 路线。nano 数字为完整 63-step 墙钟；KT 数字为其文档的
61-step stable replay。`aggregate tok/s = B / step_TPOT`。

| B | nano TPOT（复测） | nano 代表 aggregate tok/s | KT TPOT | KT aggregate tok/s | 结论 |
|---:|---:|---:|---:|---:|---|
| 2 | 110.237 / 117.120 / 125.859 ms | 18.14（110.237 ms 轮） | 121.84 ms | 16.41 | nano 三轮均值快 3.4%，但有跨轮波动 |
| 3 | 156.084 / 159.096 ms | 19.22（156.084 ms 轮） | 163.65 ms | 18.33 | nano 快 4.6%（较好一轮） |
| 5 | 154.511 / 154.854 ms | 32.36（154.511 ms 轮） | 251.92 ms | 19.85 | nano 快 38.7% |

所有 nano 行都验证了每个请求正好生成 64 token；B2/B3/B5 的代表输出分别包含
128/192/320 个生成 token，没有 EOS 提前停止。代表 digest：

- B2：`874f5e9ba069c3df18844524023462b74bfc088b792836f5aa367bea448f80be`
- B3：`1ddb8bb6b20850dd07e6c7cc5e1e25f360056eb764cd9b67e198532e66619ae3`
- B5：`6620fcc9f782dc372abb20c67460fe037990e31da2c862490302db7c3e9a1d7a`

## 根因和代码改动

### 1. 外层 `taskset` 与 CPUInfer 的 NUMA 绑核冲突

同一 fixed legacy BF16 单层 kernel 的受控测试：CPUInfer 自己管理两个 NUMA pool
时约 `0.61 ms/layer`；再套旧命令的 `taskset` 后约 `6.3 ms/layer`，慢约 10 倍。
端到端 exact 对照为 `148.568 ms/token`（无 taskset）对 `529.620 ms/token`
（有 taskset）。

CPUInfer 已使用 hwloc/NUMA 将 worker 绑定到指定节点。限制整个 Python 进程会同时
限制 CUDA callback 主线程和 worker 可用 CPU，并破坏其内部布局。代码现在发现受限
process affinity 时会明确告警。最终命令不使用 `taskset`。

### 2. nano 原 backend 与 KT 实际 kernel 不同

nano 原先使用 `kt-kernel 0.6.4` 的 `AVX2BF16_MOE`，KT 本机有效结果使用修过两个
数值 bug 的 legacy `MOE`。新增 `llamafile_bf16` 和 `llamafile_f16` backend，动态
加载 KT 本机重编译的扩展：

```text
/home/edge/zx/ktransformers/build/lib.linux-x86_64-cpython-312/
cpuinfer_ext.cpython-312-x86_64-linux-gnu.so
```

backend 会按 expert-major 的 gate/up/down 布局一次性打包并调用 legacy
`load_weights()`；热路径只提交 `forward()`。F16 模式把 CPU expert 权重转换为 F16，
hidden/output 保持 BF16，与 KT 的 F16 expert kernel 路径对齐。小型真实扩展数值测试：

- BF16：相对 PyTorch reference 最大绝对误差约 `9e-8`；
- F16 权重/BF16 activation：最大绝对误差 `3.08e-7`，平均 `4.05e-8`。

### 3. 单请求和 batch 不能共用同一个投机策略

单请求 K3 的 verify qlen 为 `K+1=4`，一次 CPU callback 摊销更多 token，因此优于
exact。true batch 时 verify qlen 变成 `B × (K+1)`；K3 在 B2/B3/B5 会把一次 verify
扩大到 8/12/20 token，同时 acceptance 并没有补偿额外 CPU expert 计算。实测 K1
和 K3 batch 均更慢，所以 batch 切到 exact AR。

### 4. legacy grouped path 和 B2 全 CPU 路由

KT 默认 `group_min_len=10`，qlen 2/3/5 都逐 token 执行。单层微基准把阈值改为 1：

| qlen | group=10 | group=1 | 改善 |
|---:|---:|---:|---:|
| 2 | 2.052 ms | 1.895 ms | 7.7% |
| 3 | 3.044 ms | 2.671 ms | 12.3% |
| 5 | 5.023 ms | 4.144 ms | 17.5% |

`m_block=4/8/16/32` 的 1000-iteration 筛选差异约 1%，没有可靠收益，因此保留已有
端到端验证配置。对于 exact F16 的 qlen=2，混合 `-1`（GPU cached routes）与 CPU
routes 会破坏 grouped fast path；CPUInfer 已持有全量专家，所以 B2 改为 CPU 计算完整
top-k，并跳过重复 GPU expert 计算。B3/B5 继续使用 hybrid 路径。

### 5. true-batch graph 和 benchmark 正确性

- standard/draft CUDA graph capture 过去只有 `1,2,4,...`，`max_num_seqs=5` 会静默
  eager；现在总是补上 `max_bs` 尾项。
- draft reroute 的有界 shape/slot-width specialization 会超过 Dynamo 默认 8 次重编译
  限制；现在按即将捕获的 shape 数设置 cache 上限。
- `kt_capture_bs` 自动包含所有 verify bucket，避免 CUDA graph host callback 持有已被
  替换的 pinned buffer 指针。
- 新增 `scripts/bench_eval_true_batch_tpot.py`，在同一 engine 中提交真正的 B 个并发
  request；逐行报告 request-visible TPOT、aggregate TPOT/吞吐、step p50/p95、完整
  63-step trace和61-step stable replay，并验证每行固定输出长度。
- batch 形状先预热 32 token；通用 allocation warmup 的 token budget 限制到
  `max_model_len`，避免为 decode benchmark 构造无关的两条 8192-token fallback tensor
  而 OOM。

## 推荐命令

不要在下面命令外层添加 `taskset`。

### 单请求 K3/BF16

```bash
cd /home/edge/zx/nano-vllm-moe

PYTHONPATH=. /home/edge/.conda/envs/nano_moe/bin/python \
  scripts/bench_eval_workload_tpot.py \
  --request-mode per_layer_slots \
  --model-path /home/edge/models/Qwen3-30B-A3B \
  --output-dir results/llamafile_k3_r075_bucket4_t06_512 \
  --optimized-config k3_3080 \
  --output-lens 512 \
  --temperature 0.6 \
  --kt-direct-backend llamafile_bf16 \
  --kt-llamafile-extension-path /home/edge/zx/ktransformers/build/lib.linux-x86_64-cpython-312/cpuinfer_ext.cpython-312-x86_64-linux-gnu.so \
  --collect-profile false --save-token-ids true --fail-fast true
```

### B2/B3/B5 exact F16

```bash
cd /home/edge/zx/nano-vllm-moe

PYTHONPATH=. /home/edge/.conda/envs/nano_moe/bin/python \
  scripts/bench_eval_true_batch_tpot.py \
  --request-mode per_layer_slots \
  --model-path /home/edge/models/Qwen3-30B-A3B \
  --output-dir results/llamafile_f16_final_batch_2_3_5 \
  --optimized-config k3_3080 \
  --inference-mode heter \
  --spec-enable-prefetch false \
  --draft-cuda-graph-enabled false \
  --verify-cuda-graph false \
  --output-lens 64 --batch-sizes 2,3,5 \
  --kt-capture-bs 1,2,4,8,16,32 \
  --kt-direct-backend llamafile_f16 \
  --kt-llamafile-extension-path /home/edge/zx/ktransformers/build/lib.linux-x86_64-cpython-312/cpuinfer_ext.cpython-312-x86_64-linux-gnu.so \
  --gpu-memory-utilization 0.999 --temperature 0.6 \
  --collect-profile false --save-token-ids true --fail-fast true
```

## 结果位置

- 单请求最终：`results/llamafile_k3_r075_bucket4_t06_512/`
- B2 带完整 step trace：`results/llamafile_f16_group1_cpuall_q2_trace16/`
- B2/B3/B5 合并验收：
  `results/llamafile_f16_group1_cpuall_q2_final_batch_2_3_5/`
- grouped-path B3/B5 复测：
  `results/llamafile_f16_group1_exact_r075_t06_true_batch_2_3_5/`
- `m_block` 微筛：`results/cpuinfer_mblock_{4,8,16,32}_group1.jsonl`
