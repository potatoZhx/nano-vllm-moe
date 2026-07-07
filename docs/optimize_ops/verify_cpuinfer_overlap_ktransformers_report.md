# Verify CPUInfer Timing, Overlap, and KTransformers Comparison

日期：2026-07-06

## 背景

本轮目标：

1. 增加 Verify 路径 CPUInfer 相关计时，并统计 CPU expert 计算数量。
2. 通过调节 draft 长度控制 verify 的 CPU expert 工作量，观察 CPUInfer/关键路径变化。
3. 检查 CPU 和 GPU 计算是否充分重叠。
4. 对照 KTransformers，在相近 CPU expert 工作量下比较 Nano 使用的 kt 算子时间是否合理。

## 代码改动

### Segment CUDA Event Timing

文件：`nanovllm/engine/model_runner.py`

新增环境变量：

```bash
NANOVLLM_VERIFY_SEGMENT_CUDA_EVENT_TIMING=1
```

作用：

- 即使 prefetch runtime 关闭，也在 verify segment graph replay 前后记录 CUDA event。
- profile 输出：
  - `model_verify_segment_cuda_event_ms`
  - `model_verify_segment_cuda_event_count`
  - `model_verify_segment_{i}_cuda_event_ms`
- 这个计时覆盖 CUDA stream 上 segment graph 的完成时间，包括 CUDA kernels、CPUInfer host dependency、CPU/GPU copy/merge 依赖。
- 相比 `NANOVLLM_VERIFY_BREAKDOWN_SYNC=1`，event timing 不在每段后强制同步，扰动更小。

兼容性修复：

- 当 prefetch runtime 是 `PredictivePrefetchRuntime` 时，不调用只有 `DualQueuePrefetchRuntime` 才有的 `record_segment_compute_ms`。

### Benchmark 输出字段

文件：

- `scripts/bench_segment_graph_no_prefetch.py`
- `scripts/bench_verify_boundary_overhead.py`

新增输出：

- `verify_cpu_routes_per_call`
- `verify_realized_cpu_expert_count_per_call`
- `verify_cpu_compute_ms_per_call`
- `verify_cpu_compute_ms_per_route`
- `verify_gpu_compute_ms_per_call`
- `verify_segment_cuda_event_ms_per_call`
- `verify_segment_cuda_event_ms_per_segment`
- `verify_cpu_routes_source`
- `verify_cpu_compute_source`

注意：

- `verify_cpu_routes_source=verify_profile` 才表示来自 verify metadata/readback 的 replay 计数。
- `verify_cpu_routes_source=aggregate_fallback` 表示 fallback 到 aggregate/capture profile，不可当作真实 replay route 计数。
- `verify_cpu_compute_source=aggregate_fallback` 时，`cpu_compute_ms` 不是 native CPUInfer replay 计时，只能作为 profile 代理量。

## 实验 1：关闭 prefetch/metadata，只测 verify segment 关键路径

命令：

```bash
source /home/linke/miniconda3/etc/profile.d/conda.sh
conda activate nano_moe
cd /home/linke/nano-vllm-moe
rm -rf results/verify_cpuinfer_sweep_smoke

NANOVLLM_VERIFY_DISABLE_RUNTIME_METADATA=1 \
NANOVLLM_VERIFY_SKIP_SYNC_METADATA_READBACK=1 \
NANOVLLM_VERIFY_SEGMENT_CUDA_EVENT_TIMING=1 \
CUDA_VISIBLE_DEVICES=2 \
python scripts/bench_segment_graph_no_prefetch.py \
  --output-dir results/verify_cpuinfer_sweep_smoke \
  --gpu-memory-utilization 0.99 \
  --cache-ratios 0.3125 \
  --output-lens 32 \
  --max-draft-tokens-values 2,4,8,12 \
  --segment-sizes 12 \
  --verify-max-cpu-routes-per-layer 0 \
  --kt-num-threads 16 \
  --case-timeout-sec 2400
```

有效结果：

| K | verify ms | verify tokens/call | segment event/call ms | segment event/segment ms | gpu compute/call ms | route source | cpu compute source |
|---:|---:|---:|---:|---:|---:|:---|:---|
| 2 | 84.000 | 2.93 | 82.387 | 20.597 | 1.440 | aggregate_fallback | aggregate_fallback |
| 4 | 107.946 | 4.77 | 106.261 | 26.565 | 1.678 | aggregate_fallback | aggregate_fallback |
| 8 | 150.969 | 8.33 | 149.238 | 37.309 | 2.404 | aggregate_fallback | aggregate_fallback |

K=12 结果不作为结论：

- 默认 verify graph buckets 是 `3,5,8,12`。
- K=12 时 verify token 可能是 K+1=13，部分路径不再是同一个 graph replay 口径。

结论：

- `segment event/call` 与 `verify ms` 基本贴合，说明关闭 prefetch 和 metadata 后，剩余 latency 主要在 segment graph replay 内。
- `gpu compute/call` 只有 1.4-2.4 ms，远小于 82-149 ms 的 segment completion。
- 因此关键路径不是 GPU grouped GEMM 或 attention，而是 graph replay 内的 CPUInfer host dependency/CPU output copy/merge 等等待。
- 这组实验的 CPU route 和 CPU compute 字段是 aggregate fallback，不能用于绝对 CPUInfer 计数。

## 实验 2：开启 metadata readback，只用于真实 CPU route/expert 计数

命令：

```bash
source /home/linke/miniconda3/etc/profile.d/conda.sh
conda activate nano_moe
cd /home/linke/nano-vllm-moe
rm -rf results/verify_cpuinfer_count_with_metadata

NANOVLLM_VERIFY_SEGMENT_CUDA_EVENT_TIMING=1 \
CUDA_VISIBLE_DEVICES=2 \
python scripts/bench_verify_boundary_overhead.py \
  --output-dir results/verify_cpuinfer_count_with_metadata \
  --modes verify_prefetch_off \
  --gpu-memory-utilization 0.99 \
  --cache-ratios 0.3125 \
  --output-lens 32 \
  --verify-off-max-draft-tokens-values 2,4,8 \
  --segment-sizes 12 \
  --allocation-mode profile_weighted \
  --slot-buckets 4 \
  --slot-max-bucket-ratio 2.0 \
  --slot-profile-csv pre_exps/exp_and_figs/unique/unique_count_plot_summary_n1024.csv \
  --kt-num-threads 16 \
  --case-timeout-sec 2400
```

结果：

| K | verify ms | verify tokens/call | CPU routes/call | CPU experts/call | segment event/call ms | hit | accept | route source | cpu compute source |
|---:|---:|---:|---:|---:|---:|---:|---:|:---|:---|
| 2 | 80.899 | 2.92 | 518.8 | 374.2 | 71.011 | 0.5368 | 0.8261 | verify_profile | aggregate_fallback |
| 4 | 104.159 | 4.43 | 733.4 | 447.6 | 93.669 | 0.5687 | 1.0000 | verify_profile | aggregate_fallback |
| 8 | 97.410 | 6.20 | 978.0 | 511.6 | 88.249 | 0.5892 | 1.0000 | verify_profile | aggregate_fallback |

解释：

- 这组的 CPU route/expert 计数来自 verify metadata readback，可信。
- `verify ms` 含 metadata readback 同步开销，不作为纯 verify latency。
- `segment event/call` 只包 segment graph replay，可与 CPU route/expert 计数配对。
- K=2 到 K=4，CPU routes/call 增加约 41%，segment event/call 增加约 32%。
- K=8 的总 CPU routes/call 继续增加，但 segment event/call 低于 K=4，说明短 output_len 下关键路径还受每段/每层 expert 分布、cache hit、acceptance 和专家复用影响，不能只按总 routes 做线性预测。

## 实验 3：route cap sweep

命令：

```bash
source /home/linke/miniconda3/etc/profile.d/conda.sh
conda activate nano_moe
cd /home/linke/nano-vllm-moe
rm -rf results/verify_cpuinfer_cap_sweep

NANOVLLM_VERIFY_DISABLE_RUNTIME_METADATA=1 \
NANOVLLM_VERIFY_SKIP_SYNC_METADATA_READBACK=1 \
NANOVLLM_VERIFY_SEGMENT_CUDA_EVENT_TIMING=1 \
CUDA_VISIBLE_DEVICES=2 \
python scripts/bench_segment_graph_no_prefetch.py \
  --output-dir results/verify_cpuinfer_cap_sweep \
  --gpu-memory-utilization 0.99 \
  --cache-ratios 0.3125 \
  --output-lens 32 \
  --max-draft-tokens-values 8 \
  --segment-sizes 12 \
  --verify-max-cpu-routes-per-layer 4,8,16 \
  --kt-num-threads 16 \
  --case-timeout-sec 2400
```

结果：

| cpu cap | verify ms | reported routes/call | reported experts/call | segment event/call ms | route source | accept |
|---:|---:|---:|---:|---:|:---|---:|
| 4 | 111.496 | 14.8 | 5.5 | 109.807 | aggregate_fallback | 0.2045 |
| 8 | 121.281 | 64.0 | 16.2 | 119.464 | aggregate_fallback | 0.5682 |
| 16 | 119.414 | 48.0 | 8.4 | 117.746 | aggregate_fallback | 0.1471 |

结论：

- 这个 monkey patch cap 对 graph replay 的真实 CPUInfer work 不能作为强证据。
- reported routes/expert 来源是 aggregate fallback，且 acceptance/输出路径被 cap 改变。
- `segment event/call` 未随 reported routes 显著下降，说明该 cap 实验更适合暴露 profile 计数限制，不适合证明 CPUInfer 缩放。

## CPU/GPU Overlap 判断

调用链：

```text
run_verify
  -> _run_verify_with_kt_hybrid_segment_graph
     -> segment graph replay
        -> Qwen3MoeDecoderLayer.forward_verify_kt_hybrid
           -> Qwen3MoeHeterogeneousSparseMoeBlock.forward_verify_kt_hybrid
              -> kt_direct.begin_forward_graph_verify
                 -> CPUInfer.submit_with_cuda_stream(...)
              -> GPU cached experts grouped GEMM
              -> kt_direct.finish_forward_graph_verify
                 -> CPUInfer.sync_with_cuda_stream(...)
                 -> CPU output copy/merge to CUDA
```

实测信号：

- K=2/4/8 no-metadata run：
  - segment event/call = 82.4/106.3/149.2 ms。
  - gpu compute/call = 1.4/1.7/2.4 ms。
  - verify ms 几乎等于 segment event/call。
- torch profiler 旧结果中，visible CUDA kernels 约 25 ms/verify，而 segment completion 约 83 ms/verify。

判断：

- GPU grouped GEMM 和 attention 不是剩余 verify latency 主体。
- CPU/GPU 有提交层面的 overlap：CPUInfer 通过 `submit_with_cuda_stream` 进入 CUDA stream dependency，GPU cached expert kernel 同段执行。
- 但 overlap 不充分隐藏 CPUInfer，因为段尾 `sync_with_cuda_stream`/CPU output merge 是每个 MoE segment 的硬依赖。
- 当前关键路径接近“每段最慢 CPUInfer 层/CPU output path 的完成时间累加”，而不是 GPU kernel 时间。

## KTransformers 对比

### KTransformers 端到端结果

已有结果：

```text
/home/linke/ktransformers/benchmark_outputs/ktransformers_qwen3_cpu_experts_tpot_20260630-040705/summary.csv
```

关键数据：

| system | CPUInfer threads | decode graph replay TPOT | decode tok/s |
|:---|---:|---:|---:|
| KTransformers Qwen3 CPU experts | 16 | 49.501 ms/token | 20.202 |

Qwen3-30B-A3B config：

| key | value |
|:---|---:|
| num_hidden_layers | 48 |
| num_experts_per_tok | 8 |
| num_experts | 128 |

粗略换算：

- 每 decode token 约 `48 * 8 = 384` CPU expert routes。
- 49.501 ms/token 折合约 `0.129 ms/route`。

限制：

- KTransformers 这个端到端 TPOT 包含 attention、linear、sampling 外的 model graph replay 开销、CPU/GPU copy。
- KTransformers YAML 使用 `KExpertsCPU`，调用链是 `CPUInfer -> TP_MOE/llamafile_sgemm`。
- Nano 当前 kt_direct 是 `kt_kernel_ext.moe.AVX2BF16_MOE/AMXBF16_MOE forward_task`。
- 所以这个对比只能作为端到端 sanity check，不能当作同一 native kernel 的 apples-to-apples 对比。

### Nano kt_kernel_ext microbench

命令：

```bash
source /home/linke/miniconda3/etc/profile.d/conda.sh
conda activate nano_moe
cd /home/linke/nano-vllm-moe
rm -f results/kt_cpu_expert_microbench_16t.jsonl

python scripts/bench_qwen3_30b_a3b_cpu_experts.py \
  --backend AVX2BF16 \
  --threads 16 \
  --subpool 1 \
  --qlen 1 \
  --layer-num 1 \
  --warmup 50 \
  --iters 300 \
  --no-progress \
  --output results/kt_cpu_expert_microbench_16t.jsonl

python scripts/bench_qwen3_30b_a3b_cpu_experts.py \
  --backend AVX2BF16 \
  --threads 16 \
  --subpool 1 \
  --qlen 4 8 \
  --layer-num 1 \
  --warmup 30 \
  --iters 150 \
  --no-progress \
  --output results/kt_cpu_expert_microbench_16t.jsonl
```

结果：

| qlen | avg latency per MoE layer | p50 | p95 | token/s | per-route estimate |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.926 ms | 0.926 ms | 0.952 ms | 1079.0 | 0.116 ms |
| 4 | 3.160 ms | 3.186 ms | 3.421 ms | 1265.1 | 0.099 ms |
| 8 | 5.961 ms | 5.955 ms | 6.382 ms | 1341.6 | 0.093 ms |

与 verify 对照：

- Metadata-count K=2/4/8 的 CPU routes/call 为 519/733/978，segment event/call 为 71/94/88 ms。
- 若直接用 segment event/routes 粗算，约 0.09-0.14 ms/route，和 microbench qlen=1/4/8 的 0.093-0.116 ms/route 同量级。
- 因此 Nano verify 使用的 kt 算子时间没有明显偏离 standalone kt_kernel_ext microbench。
- 但 verify segment event 包含 GPU kernels、CPUInfer stream dependency、CPU output copy/merge、graph replay 依赖，仍不是 native CPUInfer compute-only。

## 关键瓶颈

1. 每段 graph replay 的 CPUInfer stream dependency。
   - GPU kernel 很小，segment completion 很大。
   - `sync_with_cuda_stream` 和 CPU output copy/merge 是关键路径硬依赖。

2. CPU route/expert 分布不均。
   - 总 routes 增加不必然线性增加 segment event。
   - 真正决定段尾等待的是每段最慢层、每层 activated experts 数、每个 expert 的 token 分布。

3. Python/profile 层计数不足。
   - metadata disabled 后没有真实 replay route/expert 计数。
   - `model_cpu_compute_ms` 会退回 aggregate/capture profile，不能作为 replay CPUInfer native timing。

## 优化建议

### 必须补的测量

1. 在 kt_kernel_ext/CPUInfer native 层增加 task 计时：
   - queue wait
   - forward_task compute
   - per-layer task start/end
   - output copy/merge wait

2. 增加低扰动 runtime CPU work counter：
   - 只记录 per-segment/per-layer `cpu_route_count`、`cpu_expert_count`、max expert token count。
   - 用 tiny device counter + async D2H，避免当前 full metadata readback。

3. 在 summary 中区分：
   - graph replay segment event
   - visible CUDA kernel
   - CPUInfer native compute
   - stream wait/copy/merge

### 可优化方向

1. 降低段尾 CPUInfer 暴露时间。
   - 减少每段覆盖层数，降低最慢段等待，但会增加 graph replay 次数和边界开销。
   - 根据 CPU routes/expert 分布做不等长 segment，而不是固定 12 层。

2. 改善 CPU expert batching。
   - 对小 qlen/少 token expert，优先减少 task dispatch 和 per-expert overhead。
   - 参考 KTransformers 的 `forward_one/forward_many` 分界思想，但 Nano 使用的是 kt_kernel_ext AVX2/AMX BF16 backend，需要在对应 backend 内实现或验证。

3. 减少 CPU output copy/merge 暴露。
   - 复用 pinned buffer。
   - 检查 CPU output to CUDA 是否可以延迟到更靠近消费点，或与下一层 GPU attention/linear 重叠。
   - 避免在 Python 主线程做同步 metadata 读回。

4. 用 runtime route distribution 驱动 cache/prefetch。
   - 当前 verify prefetch off 下 CPU route miss 仍多。
   - 若能用 draft metadata 预测 verify segment 的最慢 CPU expert 层，应优先预取这些专家，而不是只优化总 submit 数。

## 当前结论

- 关闭 verify prefetch 和 metadata 后，verify 剩余 latency 的主体在 segment graph replay 内。
- GPU kernels 不是主瓶颈；CPUInfer/stream dependency/CPU output path 是关键路径。
- CPU/GPU 有 overlap，但 CPUInfer 没被 GPU 计算隐藏，因为 GPU path 太短，段尾必须等待 CPU path。
- Nano kt_direct 的 standalone AVX2BF16 MoE microbench 与 verify segment event 粗略换算同量级，没有看到明显慢于 KTransformers 参考的证据。
- 精确拆分 CPUInfer compute、queue wait、copy/merge 仍需要 kt_kernel_ext native 计时或低扰动 runtime counters。
