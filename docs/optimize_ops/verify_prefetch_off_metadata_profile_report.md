# Verify Prefetch-Off Metadata Profile Report

本文整理 `nano-vllm-moe` 中 verify prefetch off 场景的 profile 命令、过程、结果、调用链归因、与 draft/KTransformers 的对比，以及后续优化建议。

## 结论摘要

- trace 中 `aten::to -> aten::_to_copy -> aten::copy_ -> cudaMemcpyAsync` 的 65ms 级 CPU 时间已经通过深度 profile 精确定位到 verify 后的同步 metadata 读回：
  `status_cpu = dev["expert_status"].cpu()`。
- 这个 `.cpu()` 不是 CPU expert GEMM 计算本身，而是一个同步点：它等待前序 verify segment graph、kt_direct CPUInfer、D2H/H2D 依赖完成后再把 `expert_status` 拉回 CPU。
- 关闭同步读回后，torch profiler trace 里的 `cudaMemcpyAsync` CPU 时间从约 65ms 降到约 0.236ms；no-profiler benchmark 中 verify latency 从 102.264ms/call 降到 80.908ms/call。
- 关闭同步读回不能简单按 65-88ms 逐项相减，因为原同步点也给 async metadata worker/prefetch 提供了等待窗口；去掉后部分等待会转移到后台 event wait，且 cache/prefetch 时序会变化。
- verify metadata 本身的 enqueue/collect/observe 是几毫秒级；完全关闭 verify runtime metadata 后，verify latency 进一步到 77.466ms/call，说明 metadata offload 本身可省的端到端收益约 3-5ms，本轮主要瓶颈仍是同步边界和 CPU/GPU 依赖等待。
- draft metadata 开销小的主要原因是 draft 只做异步 segment metadata offload；verify 原先还在关键路径上同步读取 `expert_status`/`activation_count` 并逐层填 `_last_profile`。

## 相关代码开关

本轮增加的可回退开关：

```bash
# 只关闭 verify 末尾同步 profile 读回，保留 verify metadata async offload。
NANOVLLM_VERIFY_SKIP_SYNC_METADATA_READBACK=1

# verify metadata 轻量化：省掉 score_sum 和 expert_status，保留 activation_count。
NANOVLLM_VERIFY_METADATA_LIGHTWEIGHT=1

# 分别省掉 score_sum 或 expert_status。
NANOVLLM_VERIFY_METADATA_OMIT_SCORE_SUM=1
NANOVLLM_VERIFY_METADATA_OMIT_STATUS=1

# 完全关闭 verify runtime metadata：graph capture/replay 不记录 metadata，也不 enqueue verify metadata offload。
NANOVLLM_VERIFY_DISABLE_RUNTIME_METADATA=1

# 同义诊断开关。
NANOVLLM_VERIFY_SKIP_METADATA_OFFLOAD=1

# 捕获 verify/draft torch profiler trace。
NANOVLLM_VERIFY_TORCH_PROFILE_DIR=...
NANOVLLM_DRAFT_TORCH_PROFILE_DIR=...

# 打开 verify segment graph replay 和 metadata readback 的 record_function/counter。
NANOVLLM_VERIFY_DEEP_PROFILE=1
NANOVLLM_VERIFY_DEEP_PROFILE_SYNC=1
```

相关文件：

- `nanovllm/engine/model_runner.py`
  - verify/draft torch profiler trace 导出
  - verify sync metadata readback skip
  - verify runtime metadata disable
  - verify segment graph replay enqueue counters
- `nanovllm/expert/runtime_meta.py`
  - verify metadata lightweight / omit score / omit status
  - histogram metadata collect/offload 对缺失 `score_sum` 的兼容
- `scripts/bench_verify_boundary_overhead.py`
  - verify boundary prefetch on/off 对照脚本

## Profile 命令

### 1. 原始 per-layer slot benchmark

```bash
source /home/linke/miniconda3/etc/profile.d/conda.sh
conda activate nano_moe
cd /home/linke/nano-vllm-moe

rm -rf results/per_layer_slots_bench
CUDA_VISIBLE_DEVICES=2 python scripts/bench_per_layer_slots.py \
  --output-dir results/per_layer_slots_bench \
  --gpu-memory-utilization 0.99 \
  --cache-ratios 0.3125 \
  --output-lens 512 \
  --max-draft-tokens-values 12 \
  --segment-sizes 12 \
  --allocation-modes profile_weighted \
  --slot-buckets 4 \
  --slot-max-bucket-ratio 2.0 \
  --slot-profile-csv pre_exps/exp_and_figs/unique/unique_count_plot_summary_n1024.csv \
  --kt-num-threads 16
```

结果目录：`results/per_layer_slots_bench`

### 2. 补 verify bucket 13，避免 K=12 时 verify tokens=13 fallback

```bash
source /home/linke/miniconda3/etc/profile.d/conda.sh
conda activate nano_moe
cd /home/linke/nano-vllm-moe

rm -rf results/per_layer_slots_bench_verify13
CUDA_VISIBLE_DEVICES=2 python scripts/bench_per_layer_slots.py \
  --output-dir results/per_layer_slots_bench_verify13 \
  --gpu-memory-utilization 0.99 \
  --cache-ratios 0.3125 \
  --output-lens 512 \
  --max-draft-tokens-values 12 \
  --segment-sizes 12 \
  --allocation-modes profile_weighted \
  --slot-buckets 4 \
  --slot-max-bucket-ratio 2.0 \
  --slot-profile-csv pre_exps/exp_and_figs/unique/unique_count_plot_summary_n1024.csv \
  --kt-num-threads 16 \
  --verify-cuda-graph-bucket-steps 3,5,7,10,13
```

结果目录：`results/per_layer_slots_bench_verify13`

### 3. Verify boundary prefetch on/off smoke

```bash
source /home/linke/miniconda3/etc/profile.d/conda.sh
conda activate nano_moe
cd /home/linke/nano-vllm-moe

rm -rf results/verify_boundary_overhead_smoke
CUDA_VISIBLE_DEVICES=2 python scripts/bench_verify_boundary_overhead.py \
  --output-dir results/verify_boundary_overhead_smoke \
  --modes verify_prefetch_on,verify_prefetch_off \
  --output-lens 32 \
  --cache-ratios 0.3125 \
  --prefetch-on-max-draft-tokens-values 4 \
  --verify-off-max-draft-tokens-values 4 \
  --segment-sizes 12 \
  --gpu-memory-utilization 0.99 \
  --kt-num-threads 16 \
  --case-timeout-sec 1800
```

结果目录：`results/verify_boundary_overhead_smoke`

### 4. Verify prefetch off，torch profiler 初始 trace

```bash
source /home/linke/miniconda3/etc/profile.d/conda.sh
conda activate nano_moe
cd /home/linke/nano-vllm-moe

rm -rf results/verify_prefetch_off_profile
CUDA_VISIBLE_DEVICES=2 \
NANOVLLM_VERIFY_TORCH_PROFILE_DIR=results/verify_prefetch_off_profile/torch_profile/verify \
python scripts/bench_verify_boundary_overhead.py \
  --output-dir results/verify_prefetch_off_profile \
  --modes verify_prefetch_off \
  --output-lens 64 \
  --cache-ratios 0.3125 \
  --verify-off-max-draft-tokens-values 4 \
  --segment-sizes 12 \
  --gpu-memory-utilization 0.99 \
  --kt-num-threads 16 \
  --case-timeout-sec 1800
```

结果目录：`results/verify_prefetch_off_profile`

### 5. Verify prefetch off，深度 profile 精确定位同步读回

```bash
source /home/linke/miniconda3/etc/profile.d/conda.sh
conda activate nano_moe
cd /home/linke/nano-vllm-moe

rm -rf results/verify_prefetch_off_deep_profile
CUDA_VISIBLE_DEVICES=2 \
NANOVLLM_VERIFY_DEEP_PROFILE=1 \
NANOVLLM_VERIFY_TORCH_PROFILE_DIR=results/verify_prefetch_off_deep_profile/torch_profile/verify \
python scripts/bench_verify_boundary_overhead.py \
  --output-dir results/verify_prefetch_off_deep_profile \
  --modes verify_prefetch_off \
  --output-lens 64 \
  --cache-ratios 0.3125 \
  --verify-off-max-draft-tokens-values 4 \
  --segment-sizes 12 \
  --gpu-memory-utilization 0.99 \
  --kt-num-threads 16 \
  --case-timeout-sec 1800
```

结果目录：`results/verify_prefetch_off_deep_profile`

### 6. 关闭同步读回，带 profiler 验证 trace

```bash
source /home/linke/miniconda3/etc/profile.d/conda.sh
conda activate nano_moe
cd /home/linke/nano-vllm-moe

rm -rf results/verify_prefetch_off_skip_readback
CUDA_VISIBLE_DEVICES=2 \
NANOVLLM_VERIFY_SKIP_SYNC_METADATA_READBACK=1 \
NANOVLLM_VERIFY_DEEP_PROFILE=1 \
NANOVLLM_VERIFY_TORCH_PROFILE_DIR=results/verify_prefetch_off_skip_readback/torch_profile/verify \
python scripts/bench_verify_boundary_overhead.py \
  --output-dir results/verify_prefetch_off_skip_readback \
  --modes verify_prefetch_off \
  --output-lens 64 \
  --cache-ratios 0.3125 \
  --verify-off-max-draft-tokens-values 4 \
  --segment-sizes 12 \
  --gpu-memory-utilization 0.99 \
  --kt-num-threads 16 \
  --case-timeout-sec 1800
```

结果目录：`results/verify_prefetch_off_skip_readback`

### 7. no-profiler 基线，用于真实 latency 对照

```bash
source /home/linke/miniconda3/etc/profile.d/conda.sh
conda activate nano_moe
cd /home/linke/nano-vllm-moe

rm -rf results/verify_prefetch_off_baseline_noprof
CUDA_VISIBLE_DEVICES=2 python scripts/bench_verify_boundary_overhead.py \
  --output-dir results/verify_prefetch_off_baseline_noprof \
  --modes verify_prefetch_off \
  --output-lens 64 \
  --cache-ratios 0.3125 \
  --verify-off-max-draft-tokens-values 4 \
  --segment-sizes 12 \
  --gpu-memory-utilization 0.99 \
  --kt-num-threads 16 \
  --case-timeout-sec 1800
```

结果目录：`results/verify_prefetch_off_baseline_noprof`

### 8. no-profiler，只关闭同步读回

```bash
source /home/linke/miniconda3/etc/profile.d/conda.sh
conda activate nano_moe
cd /home/linke/nano-vllm-moe

rm -rf results/verify_prefetch_off_skip_readback_noprof
CUDA_VISIBLE_DEVICES=2 \
NANOVLLM_VERIFY_SKIP_SYNC_METADATA_READBACK=1 \
python scripts/bench_verify_boundary_overhead.py \
  --output-dir results/verify_prefetch_off_skip_readback_noprof \
  --modes verify_prefetch_off \
  --output-lens 64 \
  --cache-ratios 0.3125 \
  --verify-off-max-draft-tokens-values 4 \
  --segment-sizes 12 \
  --gpu-memory-utilization 0.99 \
  --kt-num-threads 16 \
  --case-timeout-sec 1800
```

结果目录：`results/verify_prefetch_off_skip_readback_noprof`

### 9. no-profiler，关闭同步读回 + verify metadata 轻量化

```bash
source /home/linke/miniconda3/etc/profile.d/conda.sh
conda activate nano_moe
cd /home/linke/nano-vllm-moe

rm -rf results/verify_prefetch_off_lightweight_noprof
CUDA_VISIBLE_DEVICES=2 \
NANOVLLM_VERIFY_SKIP_SYNC_METADATA_READBACK=1 \
NANOVLLM_VERIFY_METADATA_LIGHTWEIGHT=1 \
python scripts/bench_verify_boundary_overhead.py \
  --output-dir results/verify_prefetch_off_lightweight_noprof \
  --modes verify_prefetch_off \
  --output-lens 64 \
  --cache-ratios 0.3125 \
  --verify-off-max-draft-tokens-values 4 \
  --segment-sizes 12 \
  --gpu-memory-utilization 0.99 \
  --kt-num-threads 16 \
  --case-timeout-sec 1800
```

结果目录：`results/verify_prefetch_off_lightweight_noprof`

### 10. no-profiler，完全关闭 verify runtime metadata 下界

```bash
source /home/linke/miniconda3/etc/profile.d/conda.sh
conda activate nano_moe
cd /home/linke/nano-vllm-moe

rm -rf results/verify_prefetch_off_no_verify_metadata_noprof
CUDA_VISIBLE_DEVICES=2 \
NANOVLLM_VERIFY_DISABLE_RUNTIME_METADATA=1 \
NANOVLLM_VERIFY_SKIP_SYNC_METADATA_READBACK=1 \
python scripts/bench_verify_boundary_overhead.py \
  --output-dir results/verify_prefetch_off_no_verify_metadata_noprof \
  --modes verify_prefetch_off \
  --output-lens 64 \
  --cache-ratios 0.3125 \
  --verify-off-max-draft-tokens-values 4 \
  --segment-sizes 12 \
  --gpu-memory-utilization 0.99 \
  --kt-num-threads 16 \
  --case-timeout-sec 1800
```

结果目录：`results/verify_prefetch_off_no_verify_metadata_noprof`

### 11. 结果汇总脚本

```bash
cd /home/linke/nano-vllm-moe
python - <<'PY'
import json

cases = [
    ("baseline", "results/verify_prefetch_off_baseline_noprof"),
    ("skip_readback", "results/verify_prefetch_off_skip_readback_noprof"),
    ("skip+light_meta", "results/verify_prefetch_off_lightweight_noprof"),
    ("disable_verify_meta", "results/verify_prefetch_off_no_verify_metadata_noprof"),
]
print("| case | verify ms | tok/s | calls | accept | hit | status .cpu | graph enq | meta enq | meta wait | collect | observe |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for label, base in cases:
    raw = json.load(open(f"{base}/verify_prefetch_off_seg12_ratio3125_l64_k4_r0.json"))
    s = raw["summary"]
    ep = raw["engine_profile"]
    vc = s["verify_calls"]
    def per(key):
        return float(ep.get(key, 0.0) or 0.0) / vc if vc else 0.0
    print(
        f"| {label} | {s['verify_forward_ms_avg']:.3f} | {s['throughput_output_tok_s']:.3f} | "
        f"{vc} | {s['acceptance']['acceptance_rate']:.4f} | {float(s['cache'].get('true_route_hit_rate') or 0.0):.4f} | "
        f"{per('model_verify_metadata_status_cpu_ms'):.3f} | "
        f"{per('model_verify_segment_graph_replay_enqueue_ms'):.3f} | "
        f"{per('model_verify_segment_metadata_enqueue_ms'):.3f} | "
        f"{per('model_run_verify_kt_hybrid_metadata_wait_ms'):.3f} | "
        f"{per('model_run_verify_kt_hybrid_metadata_collect_ms'):.3f} | "
        f"{per('model_run_verify_kt_hybrid_metadata_observe_ms'):.3f} |"
    )
PY
```

## 结果汇总

### 原始长输出实验

| case | verify ms | draft ms | calls | acceptance | hit | 备注 |
|---|---:|---:|---:|---:|---:|---|
| `results/per_layer_slots_bench` | 123.423 | 20.937 | 67 | 0.8147 | 0.9109 | 原始命令，verify bucket 缺 13，有 eager fallback |
| `results/per_layer_slots_bench_verify13` | 101.825 | 19.735 | 77 | 0.6966 | 0.9153 | 加 bucket 13，segment graph coverage 到 1.0 |

原始命令里 K=12 时 verify token 数可能为 K+1=13，但 bucket 只有 `3,5,8,12`，导致部分 verify 落到 eager fallback。加入 13 后 fallback 消失，verify avg 明显下降；但 acceptance/calls 改变，所以只能说明 graph coverage 问题真实存在，不作为严格 apples-to-apples latency 对照。

### Boundary prefetch smoke

| case | verify ms | draft ms | calls | acceptance | hit | boundary submit/call | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| prefetch on | 132.783 | 20.251 | 7 | 1.0000 | 0.6885 | 约 7.5 | boundary prefetch 有明显可见开销 |
| prefetch off | 87.555 | 20.770 | 7 | 1.0000 | 0.5679 | 0 | 即使 CPU route work 不完全一致，关 boundary prefetch 后 verify 大幅下降 |

### Verify prefetch off 关键实验

| case | verify ms | tok/s | calls | accept | hit | status `.cpu()` ms/verify | profile loop ms/verify | graph enqueue ms/verify | verify meta enqueue ms/verify | verify meta wait ms/verify | collect ms/verify | observe ms/verify |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline no-prof | 102.264 | 17.252 | 14 | 0.9423 | 0.6190 | 88.158 | 8.243 | 3.053 | 0.462 | 26.533 | 7.840 | 3.723 |
| skip readback no-prof | 80.908 | 17.722 | 16 | 0.8545 | 0.3951 | 0.000 | 0.000 | 2.848 | 0.469 | 75.033 | 1.861 | 2.121 |
| skip + light meta no-prof | 92.334 | 18.013 | 14 | 0.9423 | 0.3951 | 0.000 | 0.000 | 2.776 | 0.405 | 85.697 | 1.295 | 3.128 |
| disable verify metadata no-prof | 77.466 | 18.576 | 15 | 0.9231 | 0.3951 | 0.000 | 0.000 | 2.131 | 0.000 | 0.000 | 0.000 | 0.000 |

解释：

- baseline 中 `status .cpu()` 计数器为 88.158ms/verify，`profile_loop` 为 8.243ms/verify。
- skip readback 直接消除了这两项，verify 从 102.264ms 降到 80.908ms。
- skip readback 后 metadata wait 变大，说明原 `.cpu()` 同步点被移除后，等待没有凭空消失，而是更多暴露在 async metadata event wait 或后续依赖中。
- lightweight metadata 明确降低了 metadata collect/enqueue/graph enqueue，但单次端到端不稳定且未优于 skip readback；这说明字节量/collect 不是主瓶颈。
- disable verify metadata 是下界：verify metadata 相关 enqueue/wait/collect/observe 全部归零，verify 最低到 77.466ms。

## Torch Profiler 关键发现

深度 profile 的 trace 中，加入 record_function 后能精确看到：

```text
nanovllm::verify.metadata_status_cpu
  -> aten::to
  -> aten::_to_copy
  -> aten::copy_
  -> cudaMemcpyAsync
```

在 `results/verify_prefetch_off_deep_profile` 中：

- `model_verify_metadata_status_cpu_ms = 1080.727`，14 calls，平均 `77.195ms/verify`。
- `model_verify_metadata_activation_cpu_ms = 2.490`，平均 `0.178ms/verify`。
- `model_verify_metadata_profile_loop_ms = 104.437`，平均 `7.460ms/verify`。
- torch profiler 单次 trace 中 `nanovllm::verify.metadata_status_cpu` 约 65ms，对应 `cudaMemcpyAsync` CPU self time 约 65ms。

关闭同步读回后，在 `results/verify_prefetch_off_skip_readback` 中：

- `model_verify_metadata_profile_readback_skipped_count = 16`。
- trace 中不再出现 `nanovllm::verify.metadata_status_cpu`。
- `cudaMemcpyAsync` CPU self time 从约 65ms 降到约 0.236ms。
- profiler 中仍可能看到 `cudaDeviceSynchronize` 或 `cudaEventSynchronize` 的大时间，但这是 profiler 导出 trace 时显式同步或 async event wait，不再是 verify forward 内的同步 metadata `.cpu()` 读回。

## Verify 调用链和开销对应

verify segment graph 主链：

```text
SpeculativeEngine.speculative_step
  -> ModelRunner.run_verify
    -> prepare verify input_ids / positions
    -> _run_verify_with_kt_hybrid_segment_graph
      -> copy input/context tensors into verify_graph_vars
      -> runtime_meta_recorder.arm(mode="verify_kt_hybrid")
      -> for each segment:
         -> prefetch_runtime.on_verify_segment_start / publish_direct_active_ready
         -> graph.replay()
            -> Qwen3MoeForCausalLM.forward_verify_kt_hybrid_segment
              -> Qwen3MoeDecoderLayer.forward_verify_kt_hybrid
                -> attention / residual / norm
                -> Qwen3MoeHeterogeneousSparseMoeBlock.forward_verify_kt_hybrid
                  -> gate linear
                  -> softmax + topk
                  -> build_verify_graph_safe_plan_gpu
                  -> runtime_meta_recorder.record_layer
                  -> kt_direct begin_forward_graph_verify
                  -> GPU cached/substitution fused_moe_linear grouped GEMM
                  -> kt_direct finish_forward_graph_verify
                  -> merge GPU route output + CPU output
         -> runtime_meta_recorder.offload_async(...)
         -> enqueue metadata worker item
      -> optional synchronous _last_profile metadata readback
      -> lm_head F.linear
```

开销对应：

- `graph.replay()` enqueue：`model_verify_segment_graph_replay_enqueue_ms`，约 2-4ms/verify，注意这是 CPU launch/enqueue 时间，不是 graph 内所有 CUDA/CPUInfer 完成时间。
- GPU cached expert grouped GEMM：属于 CUDA kernels。文档里单独列出 grouped GEMM 是因为按 kernel 名 `_grouped_gemm_forward_kernel` 聚合，它是 CUDA kernel 子集，不是 CUDA kernel 之外的另一类执行。
- kt_direct CPU expert：在 CUDA graph 中通过 `CPUInfer.submit_with_cuda_stream` / `sync_with_cuda_stream` 参与流依赖，Python 级 profiler 不能直接把 C++ worker 内部每个 AMX/AVX GEMM 切开；需要 kt extension 内部计时、`perf`、VTune 或 Nsight Systems 才能细分。
- verify metadata async offload：`model_verify_segment_metadata_enqueue_ms`，约 0.4-0.6ms/verify；D2H event wait 计入 `model_run_verify_kt_hybrid_metadata_wait_ms`。
- metadata collect/observe：`model_run_verify_kt_hybrid_metadata_collect_ms` / `observe_ms`，baseline 约 7.8ms/3.7ms，轻量化后 collect 可降到约 1.3ms。
- 同步 `_last_profile` 读回：`model_verify_metadata_status_cpu_ms` / `activation_cpu_ms` / `profile_loop_ms`。这是已确认的最大单点同步开销。

## Draft 和 Verify Metadata Offload 对比

Draft path：

```text
ModelRunner.run_draft
  -> draft segment graph replay
  -> _enqueue_draft_segment_metadata
     -> runtime_meta_recorder.offload_async(...)
     -> enqueue async metadata worker
  -> no synchronous .cpu() readback in critical path
```

Verify path 原始实现：

```text
ModelRunner._run_verify_with_kt_hybrid_segment_graph
  -> verify segment graph replay
  -> _enqueue_verify_segment_metadata
     -> runtime_meta_recorder.offload_async(...)
     -> enqueue async metadata worker
  -> status_cpu = dev["expert_status"].cpu()
  -> act_count_cpu = dev["activation_count"].cpu()
  -> loop all MoE layers and set mlp._last_profile
```

关键差异：

- draft 只有异步 offload；verify 额外有同步读回和逐层 profile loop。
- draft histogram 主要服务后续 prefetch decision；verify metadata 还携带 `expert_status`，用于 hit/miss profile。
- verify `expert_status` 对 async observe/调度路径不是核心输入，更多服务 `_last_profile` 统计。
- verify `score_sum` 影响候选 ranking，但可在诊断/轻量化模式下用 activation count 近似替代。

因此 draft metadata 看起来开销小，不是因为 draft 没有 metadata，而是它没有在主线程关键路径上同步读回 device tensor。

## 与 KTransformers 调用链对比

参考 `docs/optimize_ops/qwen3_cpu_experts_slot_buckets_call_chain.md`：

KTransformers CPU expert 路径：

```text
KExpertsCPU.forward
  -> copy hidden/topk/weights to pinned CPU buffer
  -> CPUInfer.submit / submit_with_cuda_stream
  -> cpuinfer_ext.moe.MOE.forward
  -> CPUInfer.sync / sync_with_cuda_stream
  -> output CPU -> CUDA copy
```

Nano verify kt-hybrid 路径：

```text
forward_verify_kt_hybrid
  -> GPU cached routes: Triton grouped GEMM
  -> CPU miss routes: kt_direct CPUInfer forward_task
  -> CPUInfer sync_with_cuda_stream
  -> merge GPU route output + kt_output
```

可借鉴点：

- CPUInfer submit/sync 应尽量通过 CUDA stream dependency 管理，不要在 Python 主线程额外 `.cpu()` 读回状态。
- pinned buffer 和 CUDA stream fence 的生命周期要保持异步，避免 metadata/event wait 在 verify critical path 上变成强同步。
- CPU expert 内部更细的优化需要在 kt extension/CPUInfer 层加计时，而不是依赖 Python torch profiler。

不可直接套用点：

- KTransformers 是所有 routed experts 都在 CPU 执行；Nano 是 GPU cache hit grouped GEMM + CPU miss kt_direct 混合。
- Nano 的 latency 还受 slot cache 命中率、prefetch 时序、graph-safe substitution、metadata/prefetch worker 的影响。
- 因此 KTransformers 的 expert grouping/forward_many 优化思路有参考价值，但不能直接解释 Nano verify 的 `.cpu()` metadata 同步问题。

## 优化建议

### 立即可用

1. 默认关闭 verify 同步 profile 读回：

```bash
NANOVLLM_VERIFY_SKIP_SYNC_METADATA_READBACK=1
```

收益：去掉关键路径 `.cpu()` 同步点；no-profiler 短测从 102.264ms 降到 80.908ms。

代价：`verify_cpu_routes_sum`、`pre_transfer_cache_miss_sum` 等依赖 `_last_profile` 的同步统计会变成 0 或不可用。可改为从 async metadata worker 汇总非阻塞统计。

2. 保留 `NANOVLLM_VERIFY_DISABLE_RUNTIME_METADATA=1` 作为诊断/下界开关：

```bash
NANOVLLM_VERIFY_DISABLE_RUNTIME_METADATA=1 \
NANOVLLM_VERIFY_SKIP_SYNC_METADATA_READBACK=1
```

收益：verify metadata graph/offload 完全归零，短测下界 77.466ms。

代价：verify history metadata 不再更新，可能影响后续 prefetch/cache 决策；不建议直接作为默认生产策略，除非 verify history 不参与调度。

### Metadata offload 本身

1. 将 `_last_profile` 从同步读回改为 async worker 汇总。
   - async worker 已有 `activation_count` 和可选 `expert_status` host buffer。
   - 可以在 worker collect 后累计 miss/active/cpu route counters。
   - 主线程只读上一轮已完成统计，避免等待当前 verify。

2. 减少不必要字段。
   - 如果只用于调度，verify 可以只传 `activation_count`。
   - `score_sum` 可以按需关闭，用 activation count 近似 score。
   - `expert_status` 只有需要实时 hit/miss profile 时才传。

3. 分层/分段保留当前已实现策略。
   - 当前 offload 已按 segment layer range copy，不是全 48 层每次都复制。
   - 继续避免 full metadata copy。

4. 对 metadata worker 做 best-effort。
   - host buffer 不可用或 event 未 ready 时应 drop/sample，而不是阻塞主线程。
   - 对 verify history 这类预测信号，可接受滞后一轮或采样。

### 异步执行与计算隐藏

1. 不在 verify forward 内做任何当前轮 device-to-host 同步。
   - 当前轮 metadata 最早用于下一轮/后续 prefetch，允许滞后。
   - `.cpu()`、`.item()`、`torch.nonzero` on CUDA tensor 都要避免在主线程关键路径出现。

2. 将 event wait 移出 critical path。
   - async metadata worker 可以等待，但主线程不应等待 worker 清空。
   - host buffer pool 需要足够大，避免 buffer reuse wait 变相同步。

3. 对 verify boundary prefetch 使用更严格 budget。
   - boundary smoke 表明 verify boundary prefetch on 会显著增加 verify latency。
   - 如果边界预取不能稳定降低 miss route，应该降低 `verify_prefetch_visible_budget_ms` 或禁用 verify boundary prefetch，只保留 draft-side prefetch。

4. 对 kt_direct CPUInfer 增加 native timing。
   - Python profiler 看不到 CPUInfer worker 内部 AMX/AVX GEMM 分解。
   - 建议在 kt extension 中统计 `submit`, queue wait, gate/up GEMM, activation, down GEMM, merge, output copy。

### 调度/缓存层

1. 区分诊断统计和调度必要 metadata。
   - `_last_profile` 属于诊断统计，不应强制同步当前 verify。
   - prefetch ranking 必要 metadata 应尽量小、异步、可丢弃。

2. Verify history 可以降频。
   - 例如每 N 个 verify 才 offload 一次 verify metadata。
   - 或只在 acceptance/cpu miss 明显异常时采样。

3. 避免 profile 影响 benchmark。
   - torch profiler 会引入 `torch.cuda.synchronize()`，不能用 profiler run 的 `verify_forward_ms_avg` 当最终性能数字。
   - 性能结论以 no-profiler run 为准，trace 只用于归因。

## 进一步拆解：关闭 verify prefetch 和 metadata 后的剩余 latency

在 `verify_prefetch_off + NANOVLLM_VERIFY_DISABLE_RUNTIME_METADATA=1 + NANOVLLM_VERIFY_SKIP_SYNC_METADATA_READBACK=1` 下，verify 仍约 77ms/call。为拆解这部分剩余开销，增加了诊断开关：

```bash
NANOVLLM_VERIFY_BREAKDOWN_SYNC=1
```

该开关会在 verify graph setup、每个 segment graph replay、lm_head 后同步一次，并记录：

```text
verify_segment_graph_setup_enqueue_ms
verify_segment_graph_setup_sync_ms
verify_segment_{0..3}_prefetch_hook_ms
verify_segment_{0..3}_graph_replay_enqueue_ms
verify_segment_{0..3}_graph_replay_sync_ms
verify_segment_{0..3}_boundary_submit_ms
verify_lm_head_enqueue_ms
verify_lm_head_sync_ms
```

### 诊断命令

torch profiler trace：

```bash
cd /home/linke/nano-vllm-moe
rm -rf results/verify_prefetch_off_no_meta_torch_profile
CUDA_VISIBLE_DEVICES=2 \
NANOVLLM_VERIFY_DISABLE_RUNTIME_METADATA=1 \
NANOVLLM_VERIFY_SKIP_SYNC_METADATA_READBACK=1 \
NANOVLLM_VERIFY_DEEP_PROFILE=1 \
NANOVLLM_VERIFY_TORCH_PROFILE_DIR=results/verify_prefetch_off_no_meta_torch_profile/torch_profile/verify \
python scripts/bench_verify_boundary_overhead.py \
  --output-dir results/verify_prefetch_off_no_meta_torch_profile \
  --modes verify_prefetch_off \
  --output-lens 64 \
  --cache-ratios 0.3125 \
  --verify-off-max-draft-tokens-values 4 \
  --segment-sizes 12 \
  --gpu-memory-utilization 0.99 \
  --kt-num-threads 16 \
  --case-timeout-sec 1800
```

per-segment sync breakdown：

```bash
cd /home/linke/nano-vllm-moe
rm -rf results/verify_prefetch_off_no_meta_breakdown_sync
CUDA_VISIBLE_DEVICES=2 \
NANOVLLM_VERIFY_DISABLE_RUNTIME_METADATA=1 \
NANOVLLM_VERIFY_SKIP_SYNC_METADATA_READBACK=1 \
NANOVLLM_VERIFY_BREAKDOWN_SYNC=1 \
python scripts/bench_verify_boundary_overhead.py \
  --output-dir results/verify_prefetch_off_no_meta_breakdown_sync \
  --modes verify_prefetch_off \
  --output-lens 64 \
  --cache-ratios 0.3125 \
  --verify-off-max-draft-tokens-values 4 \
  --segment-sizes 12 \
  --gpu-memory-utilization 0.99 \
  --kt-num-threads 16 \
  --case-timeout-sec 1800
```

### 阶段 breakdown 结果

正常 no-profiler 下界：

| case | verify ms | calls | acceptance | hit | 备注 |
|---|---:|---:|---:|---:|---|
| `results/verify_prefetch_off_no_verify_metadata_noprof` | 77.466 | 15 | 0.9231 | 0.3951 | 无 verify prefetch boundary submit，无 verify metadata offload |

per-segment sync 诊断结果：

| stage | ms/verify |
|---|---:|
| `verify_prepare_prefill_ms` | 0.164 |
| graph setup enqueue | 0.112 |
| graph setup sync | 0.017 |
| segment prefetch hook total | 0.043 |
| segment boundary submit total | 0.051 |
| segment graph replay enqueue total | 2.507 |
| segment graph replay sync total | 83.275 |
| lm_head enqueue | 0.220 |
| lm_head sync | 0.649 |
| run_verify total | 88.011 |

按 segment 拆分：

| segment | prefetch hook | graph enqueue | graph sync | boundary submit |
|---|---:|---:|---:|---:|
| 0 | 0.008 | 0.618 | 23.798 | 0.017 |
| 1 | 0.012 | 0.622 | 16.659 | 0.011 |
| 2 | 0.011 | 0.632 | 20.463 | 0.011 |
| 3 | 0.012 | 0.635 | 22.356 | 0.012 |

解释：

- `NANOVLLM_VERIFY_BREAKDOWN_SYNC=1` 会破坏部分 overlap，所以总 verify 从 77.466ms 增到 88.082ms；该 run 用于归因，不用于最终性能数字。
- setup、prefetch hook、boundary submit、lm_head 都是小项。
- 关键路径几乎全部在 4 个 verify segment graph 的 completion wait 上：约 83.3ms/verify。

### CUDA kernel 可见部分

torch profiler trace 中，单个 profiled verify 的 CUDA kernel 总时长约 24.9ms，粗分类如下：

| bucket | count | total ms |
|---|---:|---:|
| GPU cached expert grouped GEMM (Triton) | 96 | 15.277 |
| linear/gate/lm_head CUTLASS GEMM | 145 | 3.554 |
| other CUDA kernels | 1974 | 2.488 |
| routing topk/sort | 192 | 0.981 |
| routing/merge scatter-gather-index | 865 | 0.976 |
| RMSNorm/reductions | 675 | 0.949 |
| memcpy other | 298 | 0.320 |
| FlashAttention decode | 48 | 0.293 |
| memcpy DtoH | 192 | 0.218 |
| memcpy HtoD | 48 | 0.114 |
| softmax | 48 | 0.068 |
| activation silu/mul | 48 | 0.055 |
| KV cache store | 48 | 0.052 |

按 profiler 的 segment GPU annotation 粗分，每段普通 CUDA kernel 约 6ms：

| segment | GPU annotation ms | visible CUDA kernel ms | non-kernel gap ms |
|---|---:|---:|---:|
| 0 | 20.047 | 6.070 | 13.977 |
| 1 | 14.562 | 5.928 | 8.635 |
| 2 | 31.591 | 5.964 | 25.627 |
| 3 | 51.399 | 6.310 | 45.089 |

profiler 会放大后段等待，不能把这里的 segment duration 当性能数字；但它清楚说明：可见 CUDA kernel 不是 77ms 剩余 latency 的主体。

### 剩余关键路径归因

关闭 verify prefetch 和 metadata 后，关键路径主要是：

```text
verify segment graph replay
  -> 12 layers per segment
     -> attention / norm / router / plan kernels
     -> GPU cached expert grouped GEMM, visible CUDA kernel
     -> kt_direct begin_forward_graph_verify
        -> hidden/topk/weights copied to pinned CPU buffers
        -> CPUInfer.submit_with_cuda_stream(...)
     -> GPU cached/substitution MoE kernels overlap CPU work
     -> kt_direct finish_forward_graph_verify
        -> CPUInfer.sync_with_cuda_stream(...)
        -> output_cpu pinned buffer copy back to GPU output buffer
     -> merge CPU/GPU route outputs
```

可见 CUDA kernels 约 25ms/verify，而 per-segment sync 显示 segment completion 约 83ms/verify。差值约 55-60ms/verify，最可能来自 kt_direct CPUInfer 路径和 CUDA stream/graph host dependency：

- CPU miss expert AMX/AVX BF16 MoE 计算。
- CPUInfer queue/sync wait。
- pinned CPU input/topk/weight/output buffer copy 与 stream fence。
- `sync_with_cuda_stream` 和 graph host node 对 PyTorch profiler 的不可见部分。

这也解释了为什么消除 prefetch 和 metadata offload 后下降不够显著：剩余瓶颈已经转移到 verify graph 内部的 hybrid MoE，尤其是 CPU miss routes 的 kt_direct critical path，而不是外层 prefetch/metadata。

## 后续建议实验

1. 跑更长输出长度验证稳定性：

```bash
CUDA_VISIBLE_DEVICES=2 \
NANOVLLM_VERIFY_SKIP_SYNC_METADATA_READBACK=1 \
python scripts/bench_verify_boundary_overhead.py \
  --output-dir results/verify_prefetch_off_skip_readback_l512 \
  --modes verify_prefetch_off \
  --output-lens 512 \
  --cache-ratios 0.3125 \
  --verify-off-max-draft-tokens-values 4,8,12 \
  --segment-sizes 12 \
  --gpu-memory-utilization 0.99 \
  --kt-num-threads 16 \
  --case-timeout-sec 1800
```

2. 分别测试只省 `score_sum` 和只省 `expert_status`：

```bash
NANOVLLM_VERIFY_SKIP_SYNC_METADATA_READBACK=1 \
NANOVLLM_VERIFY_METADATA_OMIT_SCORE_SUM=1 \
...

NANOVLLM_VERIFY_SKIP_SYNC_METADATA_READBACK=1 \
NANOVLLM_VERIFY_METADATA_OMIT_STATUS=1 \
...
```

3. 增大 verify metadata host buffer pool，确认 event wait 是否来自 reuse/backpressure：

```bash
python scripts/bench_verify_boundary_overhead.py \
  ... \
  --prefetch-metadata-host-buffer-pool-size 8
```

4. 增加 kt_direct native timers，拆分 CPU expert 内部：

```text
begin_forward_graph_verify
  -> H2D/D2H/pinned copy wait
  -> CPUInfer queue wait
  -> gate/up GEMM
  -> activation
  -> down GEMM
  -> output merge/copy
  -> sync_with_cuda_stream wait
```

## 验证

语法检查：

```bash
cd /home/linke/nano-vllm-moe
/home/linke/miniconda3/envs/nano_moe/bin/python -m py_compile \
  nanovllm/engine/model_runner.py \
  nanovllm/expert/runtime_meta.py \
  scripts/bench_verify_boundary_overhead.py \
  benchmarks/scripts/spec_verify_expert_count_stats.py \
  scripts/bench_per_layer_slots.py
```

已通过。
