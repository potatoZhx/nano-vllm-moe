# ktransformers vs nano-vllm-moe Spec Profile and Optimization Analysis

Date: 2026-06-17

Repos:

- `ktransformers`: `/home/linke/ktransformers`, conda env `ktransformers`
- `nano-vllm-moe`: `/home/linke/nano-vllm-moe`, conda env `nano_moe`

Model:

- `/data1/models/Qwen3-30B-A3B`
- `num_hidden_layers=48`
- `num_experts=128`
- `num_experts_per_tok=8`
- `hidden_size=2048`
- `moe_intermediate_size=768`
- BF16

This document records the profiling process, measurements, analysis, and optimization direction for comparing:

- ktransformers decode with all MoE experts on CPU through `KExpertsCPU`.
- nano-vllm-moe speculative draft/verify with GPU expert cache, CPU miss experts through `kt_direct`, predictive prefetch, and segment CUDA graph.

The important rule used during this investigation: a claim is treated as confirmed only when a matching test/profile was run. Code-reading-only explanations are marked as hypotheses or next profiling targets.

## 1. Executive Summary

Measured facts:

- ktransformers CPU-expert single-token decode graph replay is about `33-35 ms/token`.
- nano-vllm-moe draft forward is about `19-20 ms`, with draft graph replay itself already about `15.9-16.1 ms`.
- nano-vllm-moe verify forward is about `90-99 ms` when verify graph hits.
- nano-vllm-moe predOFF verify was previously `150-170 ms` mostly because verify graph buckets did not include verify length `16`; adding bucket `16` reduced a short predOFF case from `161.09 ms` to `91.99 ms`.
- CPU miss expert compute is not the dominant cost in the measured nano eager per-layer profile: kt_direct CPU compute summed to about `2.05 ms` per verify forward, while GPU MoE execution/event time and route/plan/non-model work dominated.

Main interpretation:

- Comparing ktransformers `33 ms` against nano verify `90 ms` is not an apples-to-apples comparison. ktransformers is a single-token decode path. nano verify is multi-token, prefill-like verify, and often has tens to over a hundred active routes per layer.
- The premise "only 3-6 experts per layer are on CPU, so verify should be faster than ktransformers all-CPU experts" is incomplete. In nano verify, CPU miss experts are only one part of the cost. GPU cached experts, route planning, mixed CPU/GPU merge, metadata/prefetch, lm_head/non-model work, and multi-token verify all remain.
- The best engineering path is to optimize nano-vllm-moe spec by importing ktransformers hot-path ideas. Porting nano spec into ktransformers is higher risk and would duplicate the already-working spec engine, predictor, KV rollback, cache, prefetch, and acceptance logic.

## 2. Profiling Methodology

The investigation used three levels of measurement.

### 2.1 End-to-end benchmark probes

These runs isolate configuration-level effects such as CUDA graph bucket coverage, segment size, predictor mode, and kt_direct threadpool settings.

Representative output paths:

- `/tmp/nano_plan_probe_bucket12_off`
- `/tmp/nano_plan_probe_bucket16_off`
- `/tmp/nano_plan_probe_bucket16_off_pool2`
- `/tmp/nano_plan_probe_bucket16_on`
- `/tmp/nano_plan_probe_seg24_on`
- `/tmp/nano_impl_validate_auto_bucket`

### 2.2 Engine counters and per-layer Python hooks

nano-vllm-moe exposes engine counters through `LLM.get_profile()` and per-layer hooks through `scripts/verify_kt_direct_profile.py`.

Important limitation:

- Python per-layer hooks do not run during CUDA graph replay. They run during graph capture or eager execution, not every replay.
- Therefore:
  - Graph-path profiling uses engine counters and torch profiler / nsys.
  - Per-layer Python decomposition uses `--verify-cuda-graph false` as an eager proxy.

Representative output paths:

- Graph-path profile: `/tmp/nano_impl_verify_kt_direct_profile_graph.json`
- Eager per-layer profile: `/tmp/nano_impl_verify_kt_direct_profile_eager.json`
- Torch profiler trace: `/tmp/nano_impl_verify_kt_direct_profile_trace.json`

### 2.3 PyTorch profiler / professional profilers

Used in this round:

- PyTorch profiler for ktransformers and nano.
- Chrome trace export for both sides.

Professional profiler status:

- `nsys`: `/usr/local/cuda-13.0/bin/nsys`, used for CUDA/NVTX timeline analysis.
- `ncu`: `/usr/local/cuda-13.0/bin/ncu`, available but not yet needed for per-kernel source-level analysis.

Rationale:

- PyTorch profiler and engine counters confirmed bucket fallback, graph-hit behavior, draft graph replay cost, and the fact that Python hooks do not decompose graph replay.
- nsys was then used to separate CUDA graph launches, CUDA event waits, H2D transfers, and NVTX-tagged draft/verify phases on one timeline.

Recommended next nsys command for nano graph verify:

```bash
cd /home/linke/nano-vllm-moe
conda run -n nano_moe nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --force-overwrite=true \
  --output=/tmp/nano_verify_graph_nsys \
  python scripts/verify_kt_direct_profile.py \
    --output /tmp/nano_verify_graph_nsys.json \
    --output-len 64 \
    --cache-ratio 0.3125 \
    --max-draft-tokens 15 \
    --gpu-memory-utilization 0.99 \
    --max-model-len 8192 \
    --kt-num-threads 32 \
    --verify-cuda-graph true \
    --verify-cuda-graph-bucket-steps 3,5,8,12 \
    --auto-extend-verify-cuda-graph-buckets true \
    --engine-profile-cuda-sync false
```

Recommended next nsys command for ktransformers:

```bash
cd /home/linke/ktransformers
conda run -n ktransformers nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --force-overwrite=true \
  --output=/tmp/ktransformers_cpu_experts_nsys \
  python sosp25-ae/ktransformers-utils/bench_qwen3_cpu_experts_once.py \
    --model-path /data1/models/Qwen3-30B-A3B \
    --max-new-tokens 32 \
    --cpu-infer-threads 32 \
    --profile-output /tmp/ktransformers_cpu_experts_trace.json
```

## 3. ktransformers Call Chain

Command shape:

```bash
cd /home/linke/ktransformers
conda run -n ktransformers python sosp25-ae/ktransformers-utils/bench_qwen3_cpu_experts_once.py \
  --model-path /data1/models/Qwen3-30B-A3B \
  --max-new-tokens 64 \
  --sample \
  --temperature 0.6 \
  --top-k 20 \
  --top-p 0.95 \
  --cpu-infer-threads 32 \
  --profile \
  --system-profile
```

Optimization rule:

```yaml
model.layers.*.mlp.experts -> KTransformersExpertsV2
prefill_op: KExpertsCPU
generate_op: KExpertsCPU
out_device: cuda
```

Forward chain:

1. Build `Qwen3MoeForCausalLM` on meta device.
2. Load model through `optimize_and_load_gguf`.
3. Replace MoE experts with `KTransformersExpertsV2` / `KExpertsCPU`.
4. Prefill:
   - Embedding and dense layers use CUDA.
   - Experts are on CPU.
5. Decode:
   - First decode steps run/capture CUDA graph.
   - Later tokens replay graph through `CUDAGraphRunner`.
6. Per decoder layer:
   - Attention and dense projections: CUDA.
   - Router gate/topk: CUDA.
   - Expert compute: `KExpertsCPU` submits CPUInfer task.
   - CPUInfer writes CPU output, then output is copied back to CUDA.
7. lm_head and token selection run after model forward.

Important implementation properties:

- `KExpertsCPU` uses static pinned CPU buffers.
- CPUInfer is submitted/synced with CUDA stream.
- In the reported 32-thread run, CPUInfer used two NUMA pools:

```text
WorkerPool[...] 2 subpools, [numa:threads][0:16] [1:16]
```

Measured ktransformers results:

| Metric | Value |
|---|---:|
| TTFT | `619.31 ms` to `0.628 s` |
| Graph replay TPOT | `33.05 ms/token` |
| Profile graph replay avg | `33.69 ms/token` |
| Decode avg including warmup/capture | `37.92 ms/token` |
| PyTorch self CUDA total | `372.611 ms` |
| PyTorch self CPU total | `1.682 s` |

Short validation run after adding profile export:

```bash
cd /home/linke/ktransformers
conda run -n ktransformers python sosp25-ae/ktransformers-utils/bench_qwen3_cpu_experts_once.py \
  --model-path /data1/models/Qwen3-30B-A3B \
  --max-new-tokens 8 \
  --cpu-infer-threads 32 \
  --profile \
  --profile-output /tmp/ktransformers_impl_profile_trace.json \
  --profile-row-limit 10
```

Short run output:

| Metric | Value |
|---|---:|
| Graph replay avg | `35.28 ms/token` |
| Trace | `/tmp/ktransformers_impl_profile_trace.json` |
| Trace size | `57 MB` |

PyTorch profiler observations:

- CPU top:
  - `cudaDeviceSynchronize`
  - `cudaGraphLaunch`
  - `cudaLaunchKernel`
  - `aten::copy_`
  - `aten::mm`
- CUDA top:
  - GEMV-like CUDA kernels
  - elementwise kernels
  - reduction kernels
  - small `aten::mm`

Limitation:

- PyTorch profiler does not fully break down native CPUInfer worker compute. ktransformers' CPU expert time is mostly hidden behind synchronization and native worker activity. This is why nsys/perf-style timeline is needed for exact CPUInfer wait attribution.

## 4. nano-vllm-moe Spec Call Chain

Command shape from the original benchmark:

```bash
cd /home/linke/nano-vllm-moe
conda run -n nano_moe python scripts/bench_acceptance_predictor.py \
  --output-dir results/acc_predictor_tpot_bench \
  --acceptance-predictor-path random_cache_srdp_scripts-1/res/run_20260614_133025 \
  --gpu-memory-utilization 0.99 \
  --cache-ratios 0.3125 \
  --output-lens 512,4096 \
  --max-draft-tokens-values 15 \
  --repeats 3 \
  --segment-sizes 12 \
  --predictor-modes on,off \
  --draft-stop-policy tpot \
  --draft-tpot-td-ms 19 \
  --draft-tpot-tv-ms 80 \
  --kt-num-threads 32
```

Actual wrapper chain:

1. `scripts/bench_acceptance_predictor.py`
2. Calls `benchmarks/scripts/spec_verify_expert_count_stats.py --single-case`
3. Creates `LLM(... inference_mode="spec", enable_heterogeneous=True, enable_speculative=True, ...)`
4. `LLMEngine.step`
5. `SpeculativeEngine.speculative_step`
6. Repeated draft calls followed by one verify call.

### 4.1 Draft forward chain

Spec draft loop:

1. `SpeculativeEngine.speculative_step`
2. `model_runner.run_draft`
3. `model_runner.run(..., is_prefill=False)`
4. `prepare_decode`
5. `prepare_sample`
6. `run_model`
7. `_replay_draft_segment_graph`
8. `model.compute_logits`
9. Acceptance predictor tail graph when enabled
10. sampler

Segment graph behavior:

- With 48 layers and segment size 12, draft uses 4 segment graph replays per draft token.
- Between segments it can offload routing metadata and enqueue predictive prefetch.

### 4.2 Verify forward chain

Verify loop:

1. `SpeculativeEngine.speculative_step`
2. Build verify input: original token plus draft tokens.
3. `model_runner.run_verify`
4. `prepare_prefill`
5. `_run_verify_with_kt_hybrid_segment_graph` when enabled and bucket exists.
6. For each segment:
   - replay segment CUDA graph
   - GPU cached expert path runs inside model forward
   - CPU miss expert path uses kt_direct / CPUInfer
   - mixed route output is merged
   - metadata is offloaded
   - next segment prefetch may be submitted
7. lm_head and acceptance logic run after verify forward.

Important difference from ktransformers:

- ktransformers decode computes one token.
- nano verify computes multiple tokens: verify length is `draft_len + 1`.
- nano verify has both GPU cached routes and CPU miss routes.
- nano verify includes route planning, mixed CPU/GPU output merge, cache/prefetch metadata, and speculative acceptance machinery.

## 5. Test Chronology and Results

### 5.1 Verify bucket fallback

Hypothesis:

- predOFF with max draft tokens `15` frequently has verify length `16`.
- Default buckets `3,5,8,12` cannot capture/replay verify length `16`.
- Therefore predOFF falls back to eager verify and becomes much slower.

Test 1:

```bash
conda run -n nano_moe python scripts/bench_acceptance_predictor.py \
  --output-dir /tmp/nano_plan_probe_bucket12_off \
  --cache-ratios 0.3125 \
  --output-lens 128 \
  --max-draft-tokens-values 15 \
  --repeats 1 \
  --segment-sizes 12 \
  --predictor-modes off \
  --verify-cuda-graph-bucket-steps 3,5,8,12 \
  --kt-num-threads 32 \
  --skip-existing false
```

Test 2:

```bash
conda run -n nano_moe python scripts/bench_acceptance_predictor.py \
  --output-dir /tmp/nano_plan_probe_bucket16_off \
  --cache-ratios 0.3125 \
  --output-lens 128 \
  --max-draft-tokens-values 15 \
  --repeats 1 \
  --segment-sizes 12 \
  --predictor-modes off \
  --verify-cuda-graph-bucket-steps 3,5,8,12,16 \
  --kt-num-threads 32 \
  --skip-existing false
```

Result:

| Case | Buckets | Verify graph replay | Draft ms | Verify ms | Hit | Accept | Miss/L | Active/L |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| predOFF | `3,5,8,12` | `1/21` | `19.30` | `161.09` | `0.7275` | `0.3442` | `32.60` | `119.64` |
| predOFF | `3,5,8,12,16` | `14/14` | `19.23` | `91.99` | `0.8139` | `0.5765` | `20.84` | `112.00` |

Confirmed:

- Missing bucket `16` caused most predOFF verify calls to miss CUDA graph.
- Adding `16` reduced verify from `161.09 ms` to `91.99 ms`.
- This is a confirmed configuration bug/benchmark artifact.

Implemented fix:

- `bench_acceptance_predictor.py` and `spec_verify_expert_count_stats.py` now auto-extend verify CUDA graph buckets with `max_draft_tokens + 1`.
- Validation output: `/tmp/nano_impl_validate_auto_bucket/summary.json`

Validation:

| Metric | Value |
|---|---:|
| Effective buckets | `[3, 5, 8, 12, 16]` |
| Verify graph replay rate | `100%` |
| Draft graph replay | `15.93 ms/fwd` |
| Verify | `98.56 ms` |

### 5.2 Predictor ON graph-hit verify

Command:

```bash
conda run -n nano_moe python scripts/bench_acceptance_predictor.py \
  --output-dir /tmp/nano_plan_probe_bucket16_on \
  --cache-ratios 0.3125 \
  --output-lens 128 \
  --max-draft-tokens-values 15 \
  --repeats 1 \
  --segment-sizes 12 \
  --predictor-modes on \
  --draft-stop-policy tpot \
  --draft-tpot-td-ms 19 \
  --draft-tpot-tv-ms 80 \
  --verify-cuda-graph-bucket-steps 3,5,8,12,16 \
  --kt-num-threads 32 \
  --skip-existing false
```

Result:

| Case | Verify graph replay | Draft ms | Verify ms | Hit | Accept | Miss/L | Active/L |
|---|---:|---:|---:|---:|---:|---:|---:|
| predON bucket16 | `20/20` | `20.02` | `90.23` | `0.8256` | `0.7483` | `10.83` | `62.10` |

Confirmed:

- Even with verify graph hit and fewer active/miss routes than predOFF, verify remains about `90 ms`.
- Therefore bucket fallback is not the only bottleneck.

### 5.3 kt_direct threadpool / NUMA endpoint probe

Command changed from one pool to two pools:

```bash
--kt-threadpool-count 2 --kt-numa-nodes 0,1
```

Result:

| Case | WorkerPool | Draft ms | Verify ms | Hit | Accept | Miss/L | Active/L |
|---|---|---:|---:|---:|---:|---:|---:|
| pool1 | `[0:32]` | `19.23` | `91.99` | `0.8139` | `0.5765` | `20.84` | `112.00` |
| pool2 | `[0:16] [1:16]` | `19.23` | `96.43` | `0.8357` | `0.5161` | `19.06` | `116.00` |

Confirmed:

- The endpoint run did not improve simply by switching to two pools.

Not confirmed:

- This does not prove NUMA is irrelevant, because route load and acceptance differed.
- A fixed-routing CPUInfer microbenchmark is still needed to test NUMA fairly.

### 5.4 Segment size 12 vs 24

Hypothesis:

- Reducing verify/draft segment count from 4 to 2 might reduce graph replay overhead and metadata/prefetch overhead.

Result:

| Case | Segment | Draft ms | Verify ms | Accept | Hit | Miss/L | Active/L | Prefetch visible/call |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| predON | 12 | `20.02` | `90.23` | `0.7483` | `0.8256` | `10.83` | `62.10` | `7.83 ms` |
| predON | 24 | `20.47` | `90.55` | `0.8917` | `0.7224` | `14.80` | `53.33` | `5.78 ms` |

Confirmed:

- Segment 24 reduced visible prefetch/metadata overhead.
- Segment 24 did not reduce end-to-end draft or verify latency in this endpoint test.

Not confirmed:

- A full-model graph might still help, but this must be tested separately. The simple "fewer segments is enough" hypothesis is not supported by this test.

### 5.5 nano graph-path profile

Command:

```bash
conda run -n nano_moe python scripts/verify_kt_direct_profile.py \
  --output /tmp/nano_impl_verify_kt_direct_profile_graph.json \
  --output-len 8 \
  --cache-ratio 0.3125 \
  --max-draft-tokens 15 \
  --gpu-memory-utilization 0.99 \
  --max-model-len 8192 \
  --kt-num-threads 32 \
  --verify-cuda-graph true \
  --verify-cuda-graph-bucket-steps 3,5,8,12 \
  --auto-extend-verify-cuda-graph-buckets true \
  --sync-layer-timing true \
  --engine-profile-cuda-sync true
```

Result:

| Metric | Value |
|---|---:|
| Verify calls | `6` |
| Verify forward avg | `90.94 ms` |
| Draft forward avg | `22.71 ms` |

Observation:

- Python per-layer hooks did not fire during graph replay. This is expected: graph replay bypasses Python.
- For graph-path internals, use torch profiler trace or nsys.

### 5.6 nano eager per-layer profile

Command:

```bash
conda run -n nano_moe python scripts/verify_kt_direct_profile.py \
  --output /tmp/nano_impl_verify_kt_direct_profile_eager.json \
  --output-len 8 \
  --cache-ratio 0.3125 \
  --max-draft-tokens 15 \
  --gpu-memory-utilization 0.99 \
  --max-model-len 8192 \
  --kt-num-threads 32 \
  --verify-cuda-graph false \
  --verify-cuda-graph-bucket-steps 3,5,8,12 \
  --auto-extend-verify-cuda-graph-buckets true \
  --sync-layer-timing true \
  --engine-profile-cuda-sync true
```

This is not the target graph performance. It is a decomposition run.

Result:

| Metric | Value |
|---|---:|
| Verify calls | `7` |
| Verify forward avg | `366.55 ms` |
| Draft forward avg | `24.25 ms` |
| MoE wall per verify forward | `123.31 ms` |
| Non-MoE decoder estimate | `12.9 ms` |
| Non-model estimate | `230.3 ms` |

Per-verify MoE budget from eager proxy:

| Component | ms |
|---|---:|
| Route CPU | `6.07` |
| Plan CPU | `30.60` |
| CPU prepare | `4.51` |
| kt_direct CPU compute | `2.05` |
| GPU gather | `2.61` |
| GPU compute | `20.07` |
| Scatter | `4.67` |
| Merge | `4.72` |
| GPU wall by CUDA events | `120.14` |
| CPU overhead | `1.76` |
| MoE total wall | `123.31` |

Aggregate per-layer facts:

| Metric | Avg |
|---|---:|
| CPU routes/layer | `12.89` |
| GPU routes/layer | `19.11` |
| Pre-transfer hit rate | `0.601` |
| kt_direct CPU compute/layer | `0.043 ms` |
| kt_direct CPU prepare/layer | `0.094 ms` |
| Layer MoE GPU wall/layer | `2.503 ms` |
| Plan/layer | `0.638 ms` |
| Route/layer | `0.126 ms` |

Confirmed from eager proxy:

- kt_direct CPU compute is small in this profile.
- GPU cached expert path and plan/route work are much larger than CPU compute.
- The mixed MoE path is materially more complex than ktransformers' all-CPU expert path.

Limit:

- Eager decomposition should not be numerically equated to graph latency. It identifies cost categories, not exact graph replay percentages.

## 6. Latency Breakdown Answers

### 6.1 ktransformers latency breakdown

Measured latency:

| Stage | Measured value |
|---|---:|
| TTFT | `~619-628 ms` |
| Decode graph replay | `33.05-33.69 ms/token` |
| Short profile graph replay | `35.28 ms/token` |
| PyTorch visible CUDA total in system profile | `372.61 ms` |
| PyTorch visible CUDA per graph token, rough | `~8-9 ms/token` |

Breakdown interpretation:

- CUDA visible work is mainly small GEMV, elementwise, reduction, copy, and `aten::mm` kernels.
- CPUInfer native expert compute is not exposed as PyTorch CPU ops.
- The remaining time after visible CUDA kernels is dominated by CPUInfer expert compute, CPU/GPU synchronization, and CPU output copy-back.
- ktransformers' hot path is simple:
  - one token
  - top-8 experts per layer
  - all selected experts computed by CPUInfer
  - no GPU expert cache split
  - no CPU/GPU route merge
  - no predictive prefetch metadata loop
  - no multi-token verify acceptance loop

### 6.2 nano-vllm-moe draft latency breakdown

Measured values:

| Case | Draft total | Draft graph replay | Non-graph residual |
|---|---:|---:|---:|
| auto-bucket predOFF short | `19.26 ms` | `15.93 ms` | `~3.33 ms` |
| predON bucket16 short | `20.02 ms` | `16.14 ms` | `~3.88 ms` |
| seg24 predON short | `20.47 ms` | `16.10 ms` | `~4.37 ms` |

Draft breakdown:

- `~16 ms` is already inside graph replay.
- The residual `~3-4 ms` includes prepare/sample/predictor/prefetch bookkeeping and graph boundaries.
- Segment 24 did not improve total draft time even though it reduced segment count. Therefore draft optimization must reduce graph-body compute and captured operations, not only graph launch count.

Answer for target `10-11 ms`:

- Current nano draft cannot reach `10-11 ms` through Python/launch overhead cleanup alone, because graph replay itself is already `~16 ms`.
- The next test must compare nano draft graph kernels against the A100 ktransformers full-GPU graph path with torch profiler or nsys.

### 6.3 nano-vllm-moe verify latency breakdown

Graph-path measured values:

| Case | Verify graph hit | Verify total |
|---|---:|---:|
| predOFF bucket12 | `1/21` | `161.09 ms` |
| predOFF bucket16 | `14/14` | `91.99 ms` |
| predON bucket16 | `20/20` | `90.23 ms` |
| auto-bucket validation | `7/7` | `98.56 ms` |
| graph profile | graph path | `90.94 ms` |

Graph-path available breakdown:

| Component | Measured value |
|---|---:|
| Verify total | `~90-99 ms` |
| Metadata enqueue | `~0.7 ms/call` |
| Visible prefetch overhead | `~8 ms/call` |
| Exact graph-internal per-layer split | requires torch trace/nsys |

Eager per-layer proxy:

| Component | Value |
|---|---:|
| Verify total eager | `366.55 ms` |
| MoE wall | `123.31 ms` |
| GPU wall inside MoE | `120.14 ms` |
| Plan | `30.60 ms` |
| Route | `6.07 ms` |
| kt_direct CPU compute | `2.05 ms` |
| CPU prepare | `4.51 ms` |
| Merge | `4.72 ms` |

Key interpretation:

- The measured CPU expert compute is too small to explain verify latency.
- The verified bottleneck direction is GPU cached MoE / graph internal work / plan+route / non-model work, not raw kt_direct CPU expert compute.
- The exact graph-path breakdown must be done with torch profiler trace or nsys because Python hooks do not execute on graph replay.

## 7. Why nano-vllm-moe Is Slower Than ktransformers

### 7.1 Workload mismatch

ktransformers number:

- Single-token decode.
- One forward produces one token.
- Per layer, top-8 experts for one token.

nano verify:

- Multi-token verify.
- Verify length is `draft_len + 1`.
- predON short probe had `active/L=62.10`.
- predOFF bucket16 short probe had `active/L=112.00`.

This alone makes the comparison unfair. nano verify is doing far more active route work per layer than ktransformers single-token decode.

### 7.2 Confirmed graph fallback artifact

Default wrapper buckets were `3,5,8,12`. With `max_draft_tokens=15`, verify len can be `16`.

Measured effect:

- `161.09 ms` -> `91.99 ms` after adding bucket `16`.

This has been fixed by auto-extending buckets.

### 7.3 nano hybrid MoE path is more complex

ktransformers:

- CPUInfer computes all selected experts.
- One output path.
- No GPU expert cache hit/miss split.
- No mixed CPU/GPU route merge.

nano:

- GPU cached experts are computed on GPU.
- CPU miss experts are computed through kt_direct.
- Route planning decides GPU vs CPU routes.
- Outputs are merged back deterministically.
- Runtime metadata is recorded and offloaded.
- Predictive prefetch interacts with segment boundaries.

Measured eager proxy shows this complexity:

- Plan: `30.60 ms`
- Route: `6.07 ms`
- Merge: `4.72 ms`
- GPU MoE event wall: `120.14 ms`
- CPU compute: only `2.05 ms`

### 7.4 Draft graph body is already above target

Measured:

- draft graph replay `~16 ms`.

Therefore:

- To reach `10-11 ms`, optimize graph-body kernels and captured graph structure, not just Python overhead.

### 7.5 Segment count is not the only problem

Measured:

- Segment 12 verify: `90.23 ms`
- Segment 24 verify: `90.55 ms`

Segment 24 reduced visible prefetch overhead but did not improve total latency.

Therefore:

- Full graph may still help, but the simple segment-count hypothesis is not enough.

### 7.6 NUMA is not confirmed as endpoint bottleneck

Measured:

- pool1 `[0:32]`: `91.99 ms`
- pool2 `[0:16] [1:16]`: `96.43 ms`

This endpoint test did not show improvement. A fixed-routing CPUInfer microbenchmark is still needed before making NUMA changes a primary optimization.

## 8. How to Optimize nano-vllm-moe Toward the Targets

Targets:

- draft_forward near ktransformers full-GPU CUDA graph on A100: `10-11 ms`
- verify_forward near ktransformers CPU experts: `30-35 ms`

### 8.1 First: enforce correct benchmark conditions

Already implemented:

- Auto-extend verify graph buckets to include `max_draft_tokens + 1`.
- Report `vgraph %`.
- Do not compare verify latency unless `vgraph %` is near `100%`.

Required for future runs:

- Run service-latency mode with `--engine-profile-cuda-sync false`.
- Run separate profiler mode with syncs and torch/nsys traces.
- Report verify latency by verify length bucket.

### 8.2 Draft optimization plan

Measured starting point:

- Draft total: `19-20 ms`
- Draft graph replay: `15.9-16.1 ms`

Therefore, maximum theoretical gain from non-graph cleanup is only about `3-4 ms`. To reach `10-11 ms`, graph-body work must drop by about `5-6 ms`.

Priority tests and changes:

1. Profile nano draft graph with nsys.
   - Identify whether time is in attention, dense linears, GPU MoE, router/topk, lm_head, graph gaps, or memcpy.
   - Add NVTX ranges around draft segment replay, logits, predictor tail, sampler, and prefetch metadata.

2. Compare nano draft kernels against ktransformers full-GPU graph on A100.
   - If ktransformers uses faster dense/GEMV kernels, port or wrap those kernels into nano.
   - If nano uses less efficient small-batch MoE kernels, replace the draft GPU MoE path for batch=1/top-k with a specialized CUDA graph-friendly kernel.

3. Capture more of the draft tail.
   - Avoid eager lm_head/predictor/sampler boundaries where possible.
   - Avoid duplicate lm_head work for predictor features and sampling.
   - Use static logits/token buffers.

4. Reduce prepare/decode overhead.
   - Reuse static CUDA and pinned host buffers.
   - Avoid per-token small tensor allocations/copies in `prepare_decode`.
   - Ensure all shapes match graph buckets and avoid hidden fallbacks.

5. Re-test full-model draft graph.
   - Segment 24 was not enough.
   - A true full draft graph or a graph with deferred metadata offload may still help, but must be measured.

Expected path to `10-11 ms`:

- This likely requires kernel-level parity with the faster ktransformers full-GPU path, not just scheduling cleanup.

### 8.3 Verify optimization plan

Measured starting point:

- Verify graph path: `90-99 ms`
- Eager proxy shows CPU expert compute is not the bottleneck.

Important feasibility constraint:

- ktransformers `33 ms` is single-token decode.
- nano verify often verifies multiple tokens.
- If average verify length stays around `7-8`, a `30-35 ms` verify target is unlikely without major batching/kernel improvements. The target is more plausible if verify length is usually `1-3`, or if multi-token verify becomes much more efficient than current path.

Priority tests and changes:

1. Report verify latency by verify length.
   - Required table: verify len -> calls, active/L, miss/L, graph hit, verify ms.
   - This determines whether `30-35 ms` is feasible under current TPOT policy.

2. nsys graph-path verify.
   - Python hooks cannot decompose graph replay.
   - Use nsys to identify:
     - CUDA graph replay duration
     - CPUInfer wait points
     - GPU MoE kernels
     - lm_head kernels
     - memcpy H2D/D2H
     - gaps between segment graphs

3. Move lm_head into verify bucket graph.
   - Current verify graph returns hidden and then does additional non-model work.
   - Fixed verify buckets make lm_head capture possible.
   - This must be tested because lm_head cost can be significant for multi-token verify.

4. Reduce route/plan overhead.
   - Eager proxy plan cost was `30.60 ms`.
   - Move route/plan into static graph buffers where possible.
   - Reuse plan buffers.
   - Avoid dynamic Python-side planning in verify.

5. Optimize mixed MoE merge.
   - Preallocate route scratch/merge buffers.
   - Avoid per-layer zeroing/allocation patterns.
   - Fuse GPU cached output and CPU miss output accumulation where possible.

6. Compact CPU miss kt_direct input.
   - Current kt_direct path should be tested for how much full-route metadata it copies/filters.
   - If full top-k arrays are copied while only miss routes are needed, add compact miss-route task submission.
   - This is lower priority than GPU MoE/plan unless nsys shows CPU transfer/wait is larger than the eager proxy suggests.

7. Add adaptive verify MoE mode.
   - For small verify length and high mixed-path overhead, test a ktransformers-style all-CPU expert path.
   - For large verify length or high GPU cache hit, keep hybrid GPU/CPU.
   - Select path by measured cost model: verify length, active routes, miss routes, cache hit rate, and graph bucket.

8. Tune TPOT policy using measured costs.
   - `--draft-tpot-td-ms` and `--draft-tpot-tv-ms` do not cap performance, but they drive draft length.
   - Once real draft/verify costs are updated, retune TPOT policy to avoid verify lengths that make `30-35 ms` impossible.

Expected path to `30-35 ms`:

- First eliminate graph fallback: already done.
- Then reduce verify length or make verify length distribution explicit.
- Then use nsys to attack the graph-internal dominant cost.
- A direct `90 -> 35 ms` improvement without changing average verify length or kernels is not supported by current measurements.

## 9. Which Direction Is Better?

### Option A: optimize nano-vllm-moe spec using ktransformers ideas

Recommended.

Reasons:

- nano already has the spec engine:
  - draft/verify loop
  - KV rollback/accept
  - standard sampling acceptance
  - TPOT stop policy
  - acceptance predictor
  - expert cache
  - predictive prefetch
  - heterogeneous CPU/GPU expert placement
- The current bottlenecks are inside nano's hot path, not absence of spec logic.
- ktransformers contributes useful implementation ideas:
  - static pinned CPU buffers
  - CPUInfer stream submit/sync pattern
  - simple all-CPU expert path
  - NUMA-aware worker pool
  - faster CUDA graph decode path
- These ideas can be imported into nano incrementally and verified with existing spec correctness tests.

### Option B: port nano-vllm-moe spec into ktransformers

Not recommended as first path.

Reasons:

- It requires re-implementing or porting:
  - speculative scheduler
  - draft KV lifecycle
  - verify trace construction
  - acceptance logic
  - sampling acceptance semantics
  - acceptance predictor state
  - expert cache and eviction protection
  - predictive prefetch metadata
  - mixed CPU/GPU expert fallback policy
- It has high correctness risk.
- It delays the main profiling question: which kernels/stages actually consume the `90 ms` verify graph path.

When Option B becomes reasonable:

- If nsys proves that ktransformers' model/layer implementation has a fundamental kernel advantage that cannot be ported cleanly to nano.
- If the long-term product target is the ktransformers server rather than nano.
- If a minimal ktransformers spec proof-of-concept shows the same draft/verify semantics with materially lower latency under a fair multi-token verify workload.

Current recommendation:

- Optimize nano-vllm-moe spec first.
- Use ktransformers as a source of hot-path techniques and a profiling baseline.
- Revisit porting only after fair multi-token ktransformers verify-like experiments.

## 10. Concrete Next Work Items

1. Add NVTX ranges to nano draft/verify graph paths:
   - draft segment replay
   - verify segment replay
   - lm_head
   - sampler/acceptance
   - CPUInfer begin/finish
   - metadata offload
   - prefetch submit/publish

2. Run nsys graph profiles for:
   - nano draft graph
   - nano verify graph
   - ktransformers CPU-expert decode graph
   - ktransformers full-GPU graph on A100 if available

3. Add verify-length histogram to benchmark summary:
   - verify len
   - calls
   - verify ms
   - active/L
   - miss/L
   - graph hit

4. Implement and test lm_head-in-verify-graph.

5. Implement and test static route/merge scratch buffers.

6. Implement compact miss-route kt_direct submission only if nsys confirms metadata/copy/CPU wait is significant.

7. Build a fair ktransformers multi-token verify-like benchmark before making claims that ktransformers can do nano verify in `30-35 ms`.

## 11. Code Changes Made During This Investigation

nano-vllm-moe:

- `scripts/bench_acceptance_predictor.py`
  - auto-extend verify graph buckets
  - report effective buckets, graph hit rate, and profile breakdown fields
  - add `--verify-segment-size-override` for measuring verify graph bucket size independently from draft prefetch segment size
  - report both full hybrid graph replay and segmented hybrid graph replay counts
- `benchmarks/scripts/spec_verify_expert_count_stats.py`
  - auto-extend verify graph buckets
  - add `--engine-profile-cuda-sync`
- `scripts/verify_kt_direct_profile.py`
  - enable engine/spec profile
  - add torch profiler trace export
  - add `--cuda-profiler-range`
  - add `--nvtx-ranges`
  - handle CUDA graph replay with no Python per-layer events
  - fix flat `LLM.get_profile()` parsing
- `nanovllm/engine/model_runner.py`
  - add optional NVTX ranges around draft/verify graph, metadata, prefetch, and lm_head stages
  - fix verify segment metadata `is_last_segment` accounting
- `nanovllm/config.py`
  - add `heterogeneous_slots_per_layer_list`
  - add default-off `NANOVLLM_DUAL_QUEUE_DECOUPLE_VERIFY_SEGMENT=1` gate for experiments that decouple `verify_prefetch_segment_size` from `dual_queue_segment_size`
- `nanovllm/expert/runtime_meta.py`
  - size verify metadata host-buffer pools from logical metadata segment boundaries for the default-off full-graph metadata experiment
- `nanovllm/utils/heterogeneous_loader.py`
  - support per-layer expert-cache slot counts when explicitly configured
- `nanovllm/expert/cache.py`
  - add experimental host-only deferred prefetch commit and vectorized device LUT/mask update helpers
- `nanovllm/expert/prefetcher.py`
  - add experimental `NANOVLLM_BATCH_DEFERRED_PUBLISH=1` path for batching deferred direct-active publish device updates

ktransformers:

- `sosp25-ae/ktransformers-utils/bench_qwen3_cpu_experts_once.py`
  - add `--profile-output`
  - include PyTorch profiler top tables in `RESULT_JSON`

Validation:

```bash
cd /home/linke/nano-vllm-moe
python -m py_compile scripts/bench_acceptance_predictor.py \
  benchmarks/scripts/spec_verify_expert_count_stats.py \
  scripts/verify_kt_direct_profile.py \
  nanovllm/engine/model_runner.py \
  nanovllm/expert/cache.py \
  nanovllm/expert/prefetcher.py

cd /home/linke/ktransformers
python -m py_compile sosp25-ae/ktransformers-utils/bench_qwen3_cpu_experts_once.py
```

Both passed.

## 12. Follow-up Experiments After Initial Report

### 12.1 nsys range profiling for nano verify graph

I added `--cuda-profiler-range` to `scripts/verify_kt_direct_profile.py` so `nsys` can capture only the measured benchmark `llm.generate()` range instead of model loading.

Legacy-profile command shape:

```bash
/usr/local/cuda-13.0/bin/nsys profile \
  --force-overwrite=true \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --sample=none \
  -o /tmp/nano_verify_graph_nsys \
  python scripts/verify_kt_direct_profile.py \
    --output /tmp/nano_nsys_verify_profile.json \
    --output-len 16 \
    --cache-ratio 0.3125 \
    --max-draft-tokens 15 \
    --temperature 0.0 \
    --gpu-memory-utilization 0.99 \
    --cpu-expert-backend kt_direct \
    --spec-verify-miss-policy cpu \
    --prefetch-enabled true \
    --kt-num-threads 32 \
    --verify-cuda-graph true \
    --verify-cuda-graph-bucket-steps 3,5,8,12 \
    --auto-extend-verify-cuda-graph-buckets true \
    --engine-profile-cuda-sync true \
    --cuda-profiler-range true
```

Aligned predictive-profile command used the same runtime mode as `bench_acceptance_predictor.py`:

```bash
python scripts/verify_kt_direct_profile.py \
  --output /tmp/nano_nsys_predictive_profile.json \
  --output-len 16 \
  --cache-ratio 0.3125 \
  --max-draft-tokens 15 \
  --temperature 0.0 \
  --gpu-memory-utilization 0.99 \
  --cpu-expert-backend kt_direct \
  --spec-verify-miss-policy cpu \
  --prefetch-enabled true \
  --prefetch-runtime-mode draft_segment_indexed \
  --prefetch-runtime-kind predictive \
  --dual-queue-segment-size 12 \
  --draft-prefetch-segment-size 12 \
  --verify-prefetch-segment-size 12 \
  --prefetch-step-budget 16 \
  --prefetch-max-inflight 16 \
  --prefetch-transfer-stream-count 1 \
  --prefetch-metadata-host-buffer-pool-size 3 \
  --draft-prefetch-max-per-boundary 16 \
  --verify-prefetch-max-per-boundary 16 \
  --kt-num-threads 32 \
  --verify-cuda-graph true \
  --verify-cuda-graph-bucket-steps 3,5,8,12 \
  --auto-extend-verify-cuda-graph-buckets true \
  --engine-profile-cuda-sync true \
  --cuda-profiler-range true
```

Results:

| profile | output tokens | verify calls | draft ms | verify ms | GPU kernel total | H2D total | graph trace total |
|---|---:|---:|---:|---:|---:|---:|---:|
| legacy | 16 | 12 | 20.31 | 104.63 | 125.6 ms | 619.2 ms | 1101.5 ms |
| predictive-aligned | 16 | 6 | 21.89 | 107.12 | 96.9 ms | 525.5 ms | 395.3 ms |

The predictive trace showed large expert-weight H2D copies:

- `6,291,456 B` gate/up packed tensor copies.
- `3,145,728 B` down tensor copies.
- Predictive-aligned counts: 1224 pairs on stream 7 and 250 pairs on stream 16.
- All kernels and CUDA graph trace events were on stream 7.

Caution:

- The `llm.generate()` range still includes prefill plus decode, so total H2D cannot be assigned wholly to `verify_forward`.
- However, 492 large H2D copies in the predictive trace fell within the CUDA graph trace span, totaling 87.93 ms and 2214 MiB.
- This confirms that expert transfers are present in the decode graph window and are not just model-load noise.

Interpretation:

- The grouped-GEMM kernels are not the only dominant cost.
- There is meaningful stream-level interaction among CUDA graph replay, expert transfer, CPUInfer synchronization, and metadata/prefetch work.
- Future nsys work should add NVTX ranges around draft replay, verify replay, lm_head, acceptance, prefetch submit/publish, metadata offload, and CPUInfer submit/sync to separate draft vs verify precisely.

### 12.2 Route-buffer reuse test

Hypothesis:

- `forward_verify_kt_hybrid()` allocated a new `route_buffer = torch.zeros(...)` per MoE layer.
- Reusing the existing route-buffer cache might reduce verify replay latency.

Change tested:

- Temporarily replaced the per-call `torch.zeros` allocation with `_get_route_buffer_cache().get(...)`.

Microbenchmark:

```python
num_tokens = 16
top_k = 8
hidden_dim = 2048
num_routes = 128
```

Result:

| path | avg event time |
|---|---:|
| `torch.zeros` path | 0.0163 ms |
| cached buffer path | 0.0171 ms |

The outputs matched exactly (`max_abs_diff = 0.0`), but the cache path was slightly slower. The change was reverted. Route-buffer allocation is not a meaningful verify bottleneck under graph replay.

### 12.3 Segment verify metadata `is_last_segment` fix

Observation:

- `_run_verify_with_kt_hybrid_segment_graph()` computed:

```python
is_last = seg_idx == num_segments - 1
```

- But it passed `is_last_segment=True` for every segment when enqueueing verify metadata.

Fix:

```python
is_last_segment=is_last
```

Deterministic A/B command:

```bash
python scripts/bench_acceptance_predictor.py \
  --output-dir /tmp/nano_islast_fix_t0 \
  --acceptance-predictor-path random_cache_srdp_scripts-1/res/run_20260614_133025 \
  --gpu-memory-utilization 0.99 \
  --cache-ratios 0.3125 \
  --output-lens 64 \
  --max-draft-tokens-values 15 \
  --repeats 1 \
  --segment-sizes 12 \
  --predictor-modes off \
  --draft-stop-policy tpot \
  --draft-tpot-td-ms 19 \
  --draft-tpot-tv-ms 80 \
  --kt-num-threads 32 \
  --temperature 0.0
```

Result:

| case | digest | draft ms | verify ms | verify visible prefetch | submit | consumed | accept | miss/L | active/L |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | `29b537...2161` | 19.239 | 100.120 | 8.673 | 1517 | 3601 | 0.6154 | 22.815 | 98.0 |
| fixed | `29b537...2161` | 19.296 | 100.028 | 8.483 | 1517 | 1413 | 0.6154 | 22.815 | 98.0 |

Interpretation:

- Correctness-sensitive outputs matched exactly.
- The fix eliminated inflated verify-consumed accounting.
- Latency improvement was only ~0.1 ms for this run, so this is a correctness/accounting fix, not the main performance lever.

### 12.4 Per-layer variable expert-cache slots

I added explicit support for `heterogeneous_slots_per_layer_list`:

- `nanovllm/config.py`
- `nanovllm/utils/heterogeneous_loader.py`
- `benchmarks/scripts/spec_verify_expert_count_stats.py`
- `scripts/bench_acceptance_predictor.py`

Default behavior is unchanged when the list is empty.

Candidate allocation was derived from offline profile `act_freq` with the same total slot budget as fixed 40 slots/layer:

```text
59,53,52,55,47,42,31,31,37,35,41,48,42,50,39,28,
35,35,27,28,34,33,36,41,37,43,37,33,37,41,28,28,
33,35,35,49,39,48,40,40,41,50,45,42,43,37,44,56
```

Budget and theoretical profile coverage:

| allocation | total slots | min | max | offline top-slot coverage |
|---|---:|---:|---:|---:|
| fixed 40 | 1920 | 40 | 40 | 42.6091 |
| variable | 1920 | 27 | 59 | 43.0039 |

The expected coverage gain is only 0.3948 across all 48 layers, under 1% relative to the aggregate coverage.

Attempted benchmark:

```bash
python scripts/bench_acceptance_predictor.py \
  --output-dir /tmp/nano_varslots_t0 \
  --cache-ratios 0.3125 \
  --slots-per-layer-list "$SLOTS" \
  --output-lens 64 \
  --max-draft-tokens-values 15 \
  --predictor-modes off \
  --temperature 0.0 \
  --kt-num-threads 32
```

Result:

- The run was interrupted after more than 5 minutes.
- GPU utilization was 0%.
- The process was busy in torch inductor compile workers.

Interpretation:

- Arbitrary per-layer slot counts create many distinct fused-MoE shapes.
- Cold compile/capture cost becomes unacceptable.
- If per-layer slots are pursued, the slot counts should be bucketed to a small set of shapes, for example 32/40/48/56, and tested separately.
- Given the offline coverage gain is under 1%, same-budget per-layer redistribution is unlikely to close the 100 ms to 35 ms verify gap.

### 12.5 NVTX-separated nsys profile

I added optional NVTX ranges to the nano hot path:

- `nano:draft_segment_graph_total`
- `nano:draft_segment_graph_replay:{segment}:{layers}`
- `nano:draft_segment_metadata:{segment}:{layers}`
- `nano:draft_lm_head`
- `nano:run_verify_total`
- `nano:verify_segment_graph_total`
- `nano:verify_segment_graph_replay:{segment}:{layers}`
- `nano:verify_segment_prefetch_start:{segment}:{layers}`
- `nano:verify_segment_prefetch_submit:{segment}:{layers}`
- `nano:verify_segment_metadata:{segment}:{layers}`
- `nano:verify_lm_head`

The switch is off by default and enabled through:

```bash
--nvtx-ranges true
```

or:

```bash
NANOVLLM_NVTX_RANGES=1
```

Aligned command:

```bash
conda run -n nano_moe -- /usr/local/cuda-13.0/bin/nsys profile \
  --force-overwrite=true \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --sample=none \
  --output=/tmp/nano_nvtx_predictive \
  python scripts/verify_kt_direct_profile.py \
    --output /tmp/nano_nvtx_predictive.json \
    --output-len 16 \
    --cache-ratio 0.3125 \
    --max-draft-tokens 15 \
    --temperature 0.0 \
    --gpu-memory-utilization 0.99 \
    --max-model-len 8192 \
    --cpu-expert-backend kt_direct \
    --spec-verify-miss-policy cpu \
    --prefetch-enabled true \
    --prefetch-runtime-mode draft_segment_indexed \
    --prefetch-runtime-kind predictive \
    --dual-queue-segment-size 12 \
    --draft-prefetch-segment-size 12 \
    --verify-prefetch-segment-size 12 \
    --prefetch-step-budget 16 \
    --prefetch-max-inflight 16 \
    --prefetch-transfer-stream-count 1 \
    --prefetch-metadata-host-buffer-pool-size 3 \
    --draft-prefetch-max-per-boundary 16 \
    --verify-prefetch-max-per-boundary 16 \
    --kt-num-threads 32 \
    --verify-cuda-graph true \
    --verify-cuda-graph-bucket-steps 3,5,8,12 \
    --auto-extend-verify-cuda-graph-buckets true \
    --engine-profile-cuda-sync true \
    --cuda-profiler-range true \
    --nvtx-ranges true
```

Result:

| Metric | Value |
|---|---:|
| Trace | `/tmp/nano_nvtx_predictive.nsys-rep` |
| SQLite export | `/tmp/nano_nvtx_predictive.sqlite` |
| Output JSON | `/tmp/nano_nvtx_predictive.json` |
| Output tokens | `16` |
| Verify calls | `6` |
| Draft forward | `21.346 ms` |
| Verify forward | `106.264 ms` |

NVTX range summary:

| Range | Count | Total ms | Avg ms |
|---|---:|---:|---:|
| `nano:draft_segment_graph_total` | 33 | `543.093` | `16.457` |
| `nano:run_verify_total` | 3 | `285.781` | `95.260` |
| `nano:verify_segment_graph_total` | 3 | `277.462` | `92.487` |
| `nano:verify_segment_prefetch_start:*` | 12 | `176.076` | `14.673` |
| `nano:verify_segment_graph_replay:*` | 12 | `17.797` | `1.483` |
| `nano:verify_segment_prefetch_submit:*` | 12 | `25.797` | `2.150` |
| `nano:verify_lm_head` | 3 | `0.449` | `0.150` |

Overlap facts from SQLite:

| NVTX range | CUDA graph trace overlap | H2D overlap | Main runtime symptoms |
|---|---:|---:|---|
| draft total | `526.055 ms` | `84.813 ms` | `cudaDeviceSynchronize=480.197 ms`, `cudaEventSynchronize=402.466 ms` |
| verify total | `243.710 ms` | `64.867 ms` | `cudaEventSynchronize=215.621 ms`, `cudaStreamSynchronize=161.020 ms` |
| verify segment prefetch start | `158.348 ms` | `36.340 ms` | `675 cudaStreamSynchronize`, `675 cudaMemcpyAsync`, `9 cudaEventSynchronize` |
| verify segment prefetch submit | `25.797 ms` | `16.666 ms` | `12 cudaEventSynchronize`, `360 cudaMemcpyAsync` |

Memcpy sizes inside `verify_segment_prefetch_start`:

| Copy | Count | Bytes each | Event ms |
|---|---:|---:|---:|
| H2D expert gate/up | 100 | `6,291,456` | `23.836` |
| H2D expert down | 106 | `3,145,728` | `12.753` |
| H2D tiny LUT writes | 405 | `8` | `0.135` |
| H2D tiny mask writes | 270 | `1` | `0.090` |

Confirmed:

- `verify_lm_head` is not a meaningful bottleneck in this trace (`~0.15 ms/call`).
- `verify_segment_graph_replay:*` CPU enqueue ranges are short; the long time is not Python merely calling `graph.replay()`.
- A large part of verify wall time appears as CUDA event/stream synchronization around graph replay and direct-active prefetch publishing/submission.
- Direct-active publishing caused hundreds of tiny H2D scalar writes and stream synchronizations in the critical verify window.

Important nuance:

- The long `verify_segment_prefetch_start` ranges overlap heavily with CUDA graph trace. Some of the time is hidden under previous graph execution, but the range still sits on the verify segment boundary and contributes to serialized scheduling when it extends past the overlapped compute window.

### 12.6 Batched deferred-publish experiment

Hypothesis:

- `publish_direct_active_ready()` commits each deferred direct-active prefetch by writing several CUDA scalar LUT/mask entries:
  - `slot_to_expert_lut[slot]`
  - `expert_to_slot_lut[expert]`
  - `cached_expert_mask[expert]`
  - plus old-expert clears.
- nsys showed this as hundreds of tiny H2D copies and `cudaStreamSynchronize` calls.
- Batching device LUT/mask updates per layer should reduce launch/sync overhead.

Implementation:

- Added `LayerExpertCache.commit_deferred_active_prefetch_host_only()`.
- Added `LayerExpertCache.apply_deferred_active_prefetch_device_updates()`.
- `PrefetchRuntime.publish_direct_active_ready()` can batch deferred device updates when:

```bash
NANOVLLM_BATCH_DEFERRED_PUBLISH=1
```

Default is off, because the first validation changed deterministic output digest.

Default-path guard after adding the switch:

| Case | Digest | Draft ms | Verify ms | Accept | Hit |
|---|---|---:|---:|---:|---:|
| previous `is_last` fixed | `29b537...2161` | `19.296` | `100.028` | `0.6154` | `0.7672` |
| current default, batch switch off | `29b537...2161` | `19.240` | `109.505` | `0.6154` | `0.7672` |

This confirms the experimental switch does not change default semantics. The verify latency difference in the default rerun is not interpreted as an optimization because digest/cache stats matched while wall time varied.

Initial incorrect batching:

- First attempt wrote all new expert mappings and then cleared all previous experts.
- Deterministic benchmark output digest changed.
- Cause: within one layer, an expert can be both a newly mapped expert and a later evicted previous expert. Bulk "set all new, clear all old" does not preserve scalar commit ordering.
- Fix: build vectorized device updates from the final host-side `slot_to_expert` and `expert_to_slot` state after all host commits.

Deterministic endpoint test after the fix:

```bash
conda run -n nano_moe python scripts/bench_acceptance_predictor.py \
  --output-dir /tmp/nano_batch_publish_finalstate_t0 \
  --acceptance-predictor-path random_cache_srdp_scripts-1/res/run_20260614_133025 \
  --gpu-memory-utilization 0.99 \
  --cache-ratios 0.3125 \
  --output-lens 64 \
  --max-draft-tokens-values 15 \
  --repeats 1 \
  --segment-sizes 12 \
  --predictor-modes off \
  --draft-stop-policy tpot \
  --draft-tpot-td-ms 19 \
  --draft-tpot-tv-ms 80 \
  --kt-num-threads 32 \
  --temperature 0.0 \
  --skip-existing false
```

Result compared with previous fixed baseline:

| Case | Digest | Draft ms | Verify ms | Verify profile ms | Prefetch visible | Submit | Publish | Consumed | Accept | Hit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| previous `is_last` fixed | `29b537...2161` | `19.296` | `100.028` | `98.401` | `8.483` | 1517 | 1517 | 1413 | `0.6154` | `0.7672` |
| batched final-state | `45d6d9...7814` | `19.615` | `97.853` | `95.870` | `6.730` | 1250 | 1250 | 1257 | `0.6706` | `0.7241` |

Interpretation:

- The optimization reduced verify latency in this endpoint run by about `2.2 ms` versus the previous fixed baseline.
- It also changed digest and cache/prefetch statistics, so it is not yet accepted as a default-preserving optimization.
- The digest change is likely caused by changed cache hit/miss timing and CPU-vs-GPU numerical differences under the TPOT policy. This needs a stricter semantic validation before enabling by default.

nsys comparison with `NANOVLLM_BATCH_DEFERRED_PUBLISH=1`:

| Metric | Before | Batched | Change |
|---|---:|---:|---:|
| Verify forward in short nsys JSON | `106.264 ms` | `91.867 ms` | `-14.397 ms` |
| `verify_segment_graph_total` range | `277.462 ms` | `265.858 ms` | `-11.604 ms` |
| `run_verify_total` range | `285.781 ms` | `275.602 ms` | `-10.179 ms` |
| `cudaStreamSynchronize` inside verify segment total | 681 calls / `160.285 ms` | 271 calls / `171.767 ms` | fewer calls, similar wait wall |
| `cudaMemcpyAsync` inside verify segment total | 1107 calls / `45.907 ms` | 565 calls / `35.176 ms` | fewer copies and lower memcpy time |
| tiny 8-byte H2D in verify prefetch start | 405 calls | 84 calls | `-79%` |
| tiny 1-byte H2D in verify prefetch start | 270 calls | replaced mostly by batched 2/8/16-byte copies | fewer scalar writes |

Confirmed:

- The scalar-LUT-write hypothesis is real: batching greatly reduced tiny H2D copies and copy call count.
- The improvement is not enough by itself: verify is still around `90-100 ms`, far from `<40 ms`.
- Because output digest changed in the deterministic endpoint run, this optimization remains experimental and disabled by default.

### 12.7 Draft nosync and sampler experiments

I ran default-path endpoint benchmarks with `--engine-profile-cuda-sync false` to separate CUDA graph enqueue cost from the first mandatory synchronization point.

Temperature `0.0` command:

```bash
conda run -n nano_moe python scripts/bench_acceptance_predictor.py \
  --output-dir /tmp/nano_nosync_default_t0 \
  --acceptance-predictor-path random_cache_srdp_scripts-1/res/run_20260614_133025 \
  --gpu-memory-utilization 0.99 \
  --cache-ratios 0.3125 \
  --output-lens 64 \
  --max-draft-tokens-values 15 \
  --repeats 1 \
  --segment-sizes 12 \
  --predictor-modes off \
  --draft-stop-policy tpot \
  --draft-tpot-td-ms 19 \
  --draft-tpot-tv-ms 80 \
  --kt-num-threads 32 \
  --temperature 0.0 \
  --engine-profile-cuda-sync false \
  --skip-existing false
```

Temperature `0.8` command used the same flags without `--temperature`.

Results:

| Case | Draft total | Draft graph replay/enqueue | Draft sample | Verify | Accept | Hit |
|---|---:|---:|---:|---:|---:|---:|
| temp `0.0`, nosync | `19.350 ms` | `5.193 ms` | `11.405 ms` | `108.604 ms` | `0.6154` | `0.7505` |
| temp `0.8`, nosync | `19.186 ms` | `4.881 ms` | `11.708 ms` | `114.660 ms` | `0.5773` | `0.7747` |

Interpretation:

- With profile CUDA sync disabled, `draft_graph_replay_ms_per_forward` becomes `~5 ms`, but `draft_total` remains `~19 ms`.
- The large `draft_sample_ms_per_forward` is not pure sampler kernel time. It includes the first hard D2H synchronization (`.tolist()`) needed to return the sampled draft token to the CPU-side speculative loop. That synchronization waits for previously enqueued graph/lm_head/sampler work.
- Therefore, draft cannot be reduced to `~12 ms` just by removing Python graph replay overhead or replacing the sampler math. The bigger design issue is per-token CPU token handoff in the draft loop.

I also tested a sampler rewrite:

- Existing sampler computes `argmax(softmax(logits / T) / Exp(1))`.
- Mathematically equivalent candidate computes `argmax(logits / T - log(Exp(1)))`, avoiding full softmax.

Temperature `0.8` result:

| Case | Draft total | Draft graph | Draft sample | Verify | Accept | Tok/s |
|---|---:|---:|---:|---:|---:|---:|
| baseline sampler | `19.186 ms` | `4.881 ms` | `11.708 ms` | `114.660 ms` | `0.5773` | `16.557` |
| exp-trick sampler | `19.336 ms` | `4.819 ms` | `11.782 ms` | `100.784 ms` | `0.3759` | `12.912` |

Conclusion:

- The sampler rewrite did not reduce draft sample time.
- It changed the random token sequence and lowered acceptance in this run, hurting throughput.
- The change was reverted.

Torch profiler evidence for temp `0.8`:

| Top item | Observation |
|---|---|
| `cudaMemcpyAsync` | `401.098 ms` self CPU over the short run |
| `cudaStreamSynchronize` | `275.279 ms` self CPU |
| `Memcpy HtoD` | `599.631 ms` self CUDA, mostly expert prefetch traffic |
| `aten::copy_` | `540.078 ms` self CUDA |
| `_grouped_gemm_forward_kernel` | `176.809 ms` self CUDA |

This reinforces that draft/verify timing is dominated by synchronization and transfer interactions, not just the math inside sampler kernels.


### 12.8 Transfer stream count experiment

I tested whether increasing the number of async H2D transfer streams improves the default temp `0.8` nosync endpoint case.

Commands used the same base flags as `/tmp/nano_nosync_default_temp08`, changing only `--prefetch-transfer-stream-count`.

| Streams | Draft total | Draft graph enqueue | Draft sample/sync | Verify total | Verify profile | Verify prefetch visible | Accept | Hit | Tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `19.186 ms` | `4.881 ms` | `11.708 ms` | `114.660 ms` | `111.820 ms` | `8.023 ms` | `0.5773` | `0.7747` | `16.557` |
| 2 | `19.413 ms` | `5.219 ms` | `11.426 ms` | `111.543 ms` | `108.163 ms` | `9.106 ms` | `0.5670` | `0.7750` | `16.055` |
| 4 | `19.479 ms` | `5.186 ms` | `11.378 ms` | `113.821 ms` | `110.880 ms` | `9.622 ms` | `0.5714` | `0.7622` | `16.347` |

Interpretation:

- More transfer streams did not move draft toward `10-12 ms`.
- Verify improved slightly at 2 streams but not enough to affect the conclusion, and throughput/acceptance did not improve.
- The current bottleneck is therefore not simply insufficient transfer-stream parallelism. It remains the synchronization and metadata/publish behavior around direct-active prefetch, plus the total H2D/prefetch workload itself.

### 12.9 Device LUT scalar-copy experiment

Hypothesis:

- The scalar CUDA LUT/mask writes inside `commit_deferred_active_prefetch()` are part of the tiny H2D and synchronization cost observed in nsys.
- Replacing those writes with device-resident one-element copy sources may reduce H2D metadata overhead while preserving per-ticket commit order.

Implementation status:

- Added default-off env switch `NANOVLLM_DEVICE_LUT_SCALAR_COPY=1`.
- Default-off behavior was validated against the previous deterministic temp `0.0` output-len `64` case.

Default-off validation:

| Case | Digest | Draft | Verify | Verify profile | Prefetch visible | Accept | Hit | Submit/Publish/Consumed |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|
| previous default guard | `29b537...2161` | `19.240 ms` | `109.505 ms` | `107.889 ms` | `8.417 ms` | `0.6154` | `0.7672` | 1517 / 1517 / 1413 |
| new code, switch off | `29b537...2161` | `19.377 ms` | `117.685 ms` | `116.001 ms` | `8.821 ms` | `0.6154` | `0.7672` | 1517 / 1517 / 1413 |

Switch-on result:

| Case | Digest | Draft | Verify | Verify profile | Prefetch visible | Accept | Hit | Submit/Publish/Consumed |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|
| `NANOVLLM_DEVICE_LUT_SCALAR_COPY=1` | `45d6d9...7814` | `19.281 ms` | `97.478 ms` | `95.517 ms` | `5.756 ms` | `0.6477` | `0.7377` | 1257 / 1257 / 1306 |

Interpretation:

- The switch reduced visible prefetch/publish overhead and verify latency in this run.
- It changed digest and cache trajectory, matching the earlier batched publish digest family (`45d6d9...7814`).
- This confirms the metadata publish cost is real, but it also confirms that the current cache/prefetch algorithm is timing-dependent. Faster publish changes which transfers are visible to later draft/verify steps, changing CPU/GPU execution paths and output under deterministic decoding.
- Therefore this remains an experimental profiler switch, not a default-preserving optimization.

### 12.10 Prefetch budget and verify-length controls

I ran additional endpoint controls to separate three effects:

- the cost of prefetch/publish itself;
- the benefit of higher cache hit rate;
- verify length / segment boundary effects.

All rows below are temp `0.0`, output-len `64`, predOFF.

| Case | K | Segment | Digest | Draft | Verify | Verify profile | Prefetch visible | Accept | Hit | Miss routes/layer | Active routes/layer | Verify prefetch/forward | Tok/s |
|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default 16/16 budget | 15 | 12 | `29b537...2161` | `19.377` | `117.685` | `116.001` | `8.821` | `0.6154` | `0.7672` | `22.82` | `98.00` | `60.0` | `16.939` |
| default 16/16 budget | 15 | 24 | `29b537...2161` | `19.605` | `122.636` | `120.954` | `6.662` | `0.7500` | `0.6275` | `34.91` | `93.71` | `30.0` | `18.724` |
| no prefetch budget | 15 | 12 | `45d6d9...7814` | `18.000` | `138.790` | `137.948` | `0.000` | `0.6154` | `0.3931` | `59.47` | `98.00` | `0.0` | `16.851` |
| 8/8 budget | 15 | 12 | `45d6d9...7814` | `18.514` | `88.216` | `86.955` | `6.538` | `0.4821` | `0.7334` | `25.81` | `96.80` | `31.7` | `15.781` |
| K=4, segment=4 | 4 | 4 | `45d6d9...7814` | `18.923` | `127.075` | `125.579` | `17.141` | `0.8909` | `0.8459` | `5.67` | `36.80` | `179.4` | `16.116` |
| K=4, segment=12 | 4 | 12 | `29b537...2161` | `19.481` | `73.394` | `71.850` | `7.258` | `0.8909` | `0.7740` | `8.32` | `36.80` | `60.0` | `19.515` |

Interpretation:

- Removing prefetch does not make verify fast. It removes visible prefetch overhead, but hit rate collapses and verify worsens to `138.8 ms`.
- 8/8 budget reduces single-call verify to `88.2 ms`, but lowers acceptance and end-to-end throughput. It is not a good default optimization.
- K=15/segment=24 reduces boundary prefetch work, but hit rate drops and verify worsens to `122.6 ms`.
- K=4/segment=4 shows that small segments can be worse because segment-boundary prefetch work explodes.
- K=4/segment=12 is the cleanest short-verify comparison: active routes/layer drop from `98.0` to `36.8`, but verify is still `73.4 ms`. That means the target `25-35 ms` cannot be reached by reducing CPU miss expert count alone; fixed graph/layer cost, GPU cached MoE work, and publish/prefetch synchronization remain large.

### 12.11 Draft CPU handoff confirmation

Relevant code path:

- `SpecEngine` repeatedly calls `ModelRunner.run_draft()`.
- `run_draft()` calls `self.run(seqs, False, ...)`.
- `run()` performs `logits = self.run_model(...)`, then `token_ids = self.sampler(logits, temperatures).tolist()`.
- The returned Python token IDs are then used by the speculative loop to update sequence state and decide the next draft step.

Measured evidence:

- With profile CUDA sync disabled, draft graph enqueue/replay measured only `~4.9-5.2 ms`, but draft total stayed `~19.2 ms`.
- With profile CUDA sync enabled, draft graph replay measured `~15.9-16.0 ms` and sampler itself was only `~0.8-0.9 ms`.
- Therefore the `~11.4-11.8 ms` nosync sampler interval is not pure sampler kernel cost; it is the first hard GPU-to-CPU synchronization point after graph/lm_head/sampler work has been enqueued.
- A mathematically equivalent sampler rewrite did not reduce draft total.
- Disabling prefetch budget reduced draft only to `18.0 ms`, still far above `10-12 ms`.

Interpretation:

- Current draft is a CPU-driven one-token loop. CUDA graph capture covers the model body for one step, but the sampled token is synchronized to CPU every step.
- Reaching `10-12 ms` requires materially faster captured graph-body kernels. A multi-step GPU-resident draft loop would also remove CPU handoff overhead, but by itself it is not enough while one-step graph body is already `~16 ms`.

### 12.12 Graph/eager verify body diagnostics

I added more diagnostics to separate graph replay behavior from Python per-layer decomposition.

Correct endpoint-like graph path:

```bash
python scripts/verify_kt_direct_profile.py \
  --output /tmp/nano_graph_predictive_cpu_t0.json \
  --output-len 16 \
  --cache-ratio 0.3125 \
  --max-draft-tokens 15 \
  --verify-cuda-graph true \
  --prefetch-runtime-mode draft_segment_indexed \
  --prefetch-runtime-kind predictive \
  --spec-verify-miss-policy cpu \
  --engine-profile-cuda-sync false
```

Result:

- Python per-layer hooks did not run, confirming CUDA graph replay.
- Verify forward averaged `92.356 ms`.
- Draft forward averaged `21.257 ms` in this short profile run.

Torch profiler on the same graph shape (`/tmp/nano_graph_predictive_cpu_torch_t0.json`, trace `/tmp/nano_graph_predictive_cpu_torch_trace.json`) showed:

| CUDA item | Self CUDA |
|---|---:|
| H2D pinned copies | `498.887 ms` |
| `aten::copy_` | `453.736 ms` |
| `_grouped_gemm_forward_kernel` | `160.250 ms` |
| GEMV kernels | `42.044 ms` |
| `aten::mm` | `29.150 ms` |

Interpretation:

- On the graph path, prefetch H2D traffic and GPU grouped-GEMM work dominate the measured CUDA time.
- This is consistent with nsys: direct-active prefetch/publish and H2D transfer are real costs, but simply reducing prefetch hurts cache hit rate and can worsen verify.

Eager per-layer decomposition (`/tmp/nano_eager_breakdown_k15_t0.json`, output-len `32`, K=15, cache ratio `0.3125`) measured:

| Component | Per-verify sum |
|---|---:|
| Route CPU | `6.239 ms` |
| Plan CPU | `29.285 ms` |
| CPU prepare | `4.465 ms` |
| kt_direct CPU compute | `2.028 ms` |
| GPU exec CUDA event sum | `140.854 ms` |
| MoE CPU wall sum | `142.757 ms` |

Near all-CPU-miss diagnostic (`/tmp/nano_eager_cpu_heavy_slot1_t0.json`, slots/layer `1`, no prefetch, K=4):

| Metric | Value |
|---|---:|
| Cache hit rate | `1.8%` |
| Verify forward | `198.652 ms` |
| Draft forward | `13.450 ms` |
| kt_direct CPU compute sum | `2.064 ms` |
| GPU exec CUDA event sum | `134.679 ms` |
| Plan CPU sum | `23.440 ms` |

Interpretation:

- kt_direct CPU compute remains small even when almost every route is a CPU miss.
- The expensive part is still the heterogeneous/nano MoE execution plan and GPU-side kernels around routing, scatter/merge, fallback/cache routing, and attention/model body.
- Therefore "more CPU experts" or "fewer CPU experts" alone does not explain nor solve the gap to ktransformers.

Skip-observe diagnostic (`NANOVLLM_PREFETCH_SKIP_OBSERVE=1`, `/tmp/nano_skip_observe_t0/summary.json`) measured:

- Verify forward `122.946 ms`.
- Prefetch visible `0.052 ms`.
- Hit rate `0.3932`.

This shows metadata observe is not the main bottleneck. Removing it collapses future cache quality and verify remains high.

### 12.13 Cache strategy controls

I tested existing cache eviction strategies as default-preserving scheduling candidates under the same temp `0.0`, output-len `64`, K=15, segment=12 setup.

| Strategy | Draft | Verify | Hit | Accept | Tok/s | Submit/Publish/Consumed |
|---|---:|---:|---:|---:|---:|---:|
| `lru` baseline | `19.377 ms` | `117.685 ms` | `0.7672` | `0.6154` | `16.939` | 1517 / 1517 / 1413 |
| `lfu_rankguard` | `19.307 ms` | `113.163 ms` | `0.7157` | `0.4060` | `13.355` | 2140 / 2140 / 1567 |
| `lfu` | `19.279 ms` | `126.775 ms` | `0.7065` | `0.4661` | `14.245` | 1902 / 1902 / 1393 |

Interpretation:

- `lfu_rankguard` slightly lowered verify in one run but hurt hit rate, acceptance, and throughput.
- `lfu` was worse on verify, hit rate, acceptance, and throughput.
- Existing cache strategies do not provide a safe optimization path to the target.

### 12.14 Current Updated Bottleneck View

The current evidence points to these priorities:

1. Draft total remains `~19 ms` in the target path. The one-step captured graph body is already `~16 ms`, so closing to `10-12 ms` requires reducing graph-body kernels/captured work, not just sampler or Python handoff tweaks.
2. Verify remains far above `25-35 ms` even after reducing verify length: K=4/segment=12 still measured `73.4 ms`, while full K=15/segment=12 default measured `~100-118 ms` depending on run.
3. Verify bottleneck is not kt_direct CPU compute. Eager decomposition repeatedly measured kt_direct compute around `2 ms/verify`; GPU execution and plan dominate.
4. Prefetch is a tradeoff, not pure overhead: disabling/skipping it reduces visible prefetch work but collapses hit rate and leaves verify high.
5. Same-budget per-layer slots and existing LFU/rankguard cache strategies are not promising in current tests.
6. Segment metadata accounting had a real bug, now fixed, but it did not explain the latency gap.
7. `verify_lm_head` is measured at only `~0.15 ms/call`; it is not the main verify lever.
8. The next high-value changes are:
   - make direct-active publish batching semantically safe or move publish off the segment boundary;
   - reduce required expert H2D traffic without sacrificing hit rate;
   - replace nano's current graph-safe verify MoE plan/GPU kernels with a more ktransformers-like fused expert path;
   - reduce draft graph-body kernels enough that a GPU-resident draft loop would matter.

### 12.15 Verify compact GPU-route experiment

Question:

- Does nano verify still spend GPU grouped-GEMM work on CPU miss routes?

Code finding:

- `build_verify_graph_safe_plan_gpu()` originally mapped all routes through a substitution LUT and always set:
  - `gpu_route_indices = arange(num_routes)`
  - `m_sizes = grouped_layout(all routes)`
  - miss-route `gpu_route_weights = 0`
- `forward_verify_kt_hybrid()` then ran both `fused_moe_linear()` calls on `gpu_hidden` for all routes. CPU miss routes were also computed by `kt_direct` and added later.

Experiment:

- Added default-off env gate `NANOVLLM_VERIFY_COMPACT_GPU_ROUTES=1`.
- In the gated path, grouped-GEMM `m_sizes` counts only cached routes; CPU miss routes remain in the fixed-shape permutation tail with zero contribution.
- Added zero-mask protection after the first and second grouped GEMM to avoid `torch.empty()` tail rows affecting activation/scatter.

Endpoint command shape:

```bash
CUDA_VISIBLE_DEVICES=0 NANOVLLM_VERIFY_COMPACT_GPU_ROUTES=1 \
conda run -n nano_moe python scripts/bench_acceptance_predictor.py \
  --output-dir /tmp/nano_verify_compact_gpu_routes_t0 \
  --acceptance-predictor-path random_cache_srdp_scripts-1/res/run_20260614_133025 \
  --gpu-memory-utilization 0.99 \
  --cache-ratios 0.3125 \
  --output-lens 64 \
  --max-draft-tokens-values 15 \
  --repeats 1 \
  --segment-sizes 12 \
  --predictor-modes off \
  --draft-stop-policy tpot \
  --draft-tpot-td-ms 19 \
  --draft-tpot-tv-ms 80 \
  --kt-num-threads 32 \
  --temperature 0.0 \
  --skip-existing false
```

Endpoint result:

| Mode | Digest | Draft | Verify | Verify profile | Hit | Accept | vgraph |
|---|---|---:|---:|---:|---:|---:|---:|
| default, env off | `29b53727...2161` | `19.381 ms` | `118.411 ms` | `116.746 ms` | `0.7677` | `0.6154` | `100%` |
| compact, env on | `29b53727...2161` | `19.307 ms` | `112.153 ms` | `110.524 ms` | `0.7672` | `0.6154` | `100%` |

This confirms the all-route GPU work is real enough to affect endpoint latency, and the gated path preserved deterministic output in this temp-0 run.

However, torch profiler on the graph path showed the current compact implementation is not yet a mature optimization:

| Profile | Sync mode | Verify calls | Verify avg | H2D pinned | `aten::copy_` CUDA | `_grouped_gemm_forward_kernel` |
|---|---|---:|---:|---:|---:|---:|
| default graph profile | `engine_profile_cuda_sync=false` | `6` | `107.672 ms` | `498.887 ms` | `453.736 ms` | `160.250 ms` |
| compact graph profile | `engine_profile_cuda_sync=false` | `6` | `119.666 ms` | `669.479 ms` | `570.474 ms` | `318.186 ms` |

Interpretation:

- The code-level diagnosis is confirmed: default verify plans send CPU miss routes through the GPU grouped-GEMM path with zero weights.
- The simple fixed-shape compact patch is insufficient. It adds extra masking/sorting/copy graph work and did not reduce profiler-visible grouped-GEMM time under the comparable torch profile.
- A real optimization must compact active GPU routes end to end, not only adjust `m_sizes`:
  - graph bucket by active-route capacity;
  - avoid activation/scatter over inactive tail rows;
  - preallocate compact route buffers;
  - avoid extra `where`, sort, and copy kernels;
  - keep deterministic route order and digest equality.

Therefore this experiment is useful evidence, but it is not a final performance path.

### 12.16 All-CPU verify experiment

Question:

- Can nano verify match ktransformers by making verify experts all go through `kt_direct`/CPUInfer and skipping GPU expert GEMM?

Experiment:

- Added default-off env gate `NANOVLLM_VERIFY_ALL_CPU_EXPERTS=1`.
- In this path:
  - verify still runs router/topk on GPU;
  - `KtDirectCpuMoeBackend.begin_forward_graph_verify(..., force_all_cpu=True)` fills `gpu_expert_mask_cpu` with `False`;
  - CPUInfer computes all selected top-k experts;
  - nano skips GPU expert grouped GEMM and route-buffer merge for verify;
  - runtime metadata records all routes as CPU/miss.

Validation:

- `python -m py_compile` passed for touched files.
- `pytest tests/test_kt_direct_backend.py tests/test_verify_cuda_graph_kt_hybrid.py -q` passed: `38 passed`.

Endpoint result (`/tmp/nano_verify_all_cpu_t0/summary.json`, temp `0.0`, output-len `64`, K=15, segment=12):

| Mode | Digest | Draft | Verify | Hit | Accept | vgraph |
|---|---|---:|---:|---:|---:|---:|
| default baseline | `29b53727...2161` | `19.381 ms` | `118.411 ms` | `0.7677` | `0.6154` | `100%` |
| all-CPU verify | `2b5b5861...12a7` | `19.446 ms` | `141.598 ms` | `0.0000` | `0.5670` | `100%` |

Interpretation:

- Directly making nano verify all-CPU is worse, not better.
- The ktransformers `33 ms` number is a one-token decode path. In this nano endpoint, all-CPU verify asks CPUInfer to compute multi-token verify work; it also marks every route as miss, which changes cache/prefetch trajectory and output digest.
- This rejects the naive porting idea "just use ktransformers CPU expert execution for the whole nano verify."
- A useful all-CPU path would need an adaptive, bucketed cost model and should only be selected for cases where measured verify length/route count make it cheaper than hybrid. The tested K=15 endpoint is not such a case.

### 12.17 Verify length bucket instrumentation

Question:

- Is the `~118 ms` verify average hiding a small-length bucket that is already close to the `<40 ms` target?

Implementation:

- Added `ModelRunner._verify_length_buckets`, reported through `engine_profile["verify_length_buckets"]`.
- The benchmark summary now preserves and prints:
  - verify length;
  - call count;
  - average verify forward ms;
  - average token count;
  - graph hit rate;
  - active routes/layer;
  - miss routes/layer.

Validation:

- `python -m py_compile` passed for `model_runner.py` and `bench_acceptance_predictor.py`.
- `pytest tests/test_kt_direct_backend.py tests/test_verify_cuda_graph_kt_hybrid.py -q` passed: `38 passed`.
- Default endpoint rerun preserved digest `29b53727...2161`.

Default endpoint result (`/tmp/nano_verify_length_buckets_default_t0/summary.json`, temp `0.0`, output-len `64`, K=15, segment=12):

| Verify len | Calls | Verify ms | Token count | Graph | Active/L | Miss/L |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1 | `73.447` | `4.00` | `100%` | `32.00` | `3.60` |
| 14 | 1 | `121.159` | `14.00` | `100%` | `112.00` | `21.17` |
| 16 | 5 | `125.801` | `16.00` | `100%` | `128.00` | `31.55` |

Interpretation:

- Even verify length `4` is `73.4 ms`; therefore the `<40 ms` target is not reachable by merely nudging TPOT to slightly shorter draft lengths.
- The measured latency scales partly with active routes, but the length-4 bucket is still far above target, which points to fixed per-verify/per-layer overhead and current graph/hybrid MoE structure.
- Future optimization should first target the length-4 bucket. If length-4 cannot approach `<40 ms`, length-14/16 cannot reach the target without a deeper execution rewrite.

### 12.18 Decoupled verify segment / full verify graph experiment

Question:

- Does nano verify spend a material amount of time replaying and stitching small segment graphs?
- Can a full verify graph reduce latency without changing the draft segment size that drives predictive prefetch?

Implementation:

- Added a default-off config gate:
  - `NANOVLLM_DUAL_QUEUE_DECOUPLE_VERIFY_SEGMENT=1`
  - This prevents `prefetch_runtime_kind=dual_queue` from forcibly setting `verify_prefetch_segment_size = dual_queue_segment_size`.
- Added benchmark wrapper argument:
  - `--verify-segment-size-override`
  - When the override differs from `--segment-sizes`, the child process sets `NANOVLLM_DUAL_QUEUE_DECOUPLE_VERIFY_SEGMENT=1` and passes the larger verify segment size.
- Fixed graph replay accounting so summaries distinguish:
  - `verify_hybrid_graph_replays`: full verify graph replay;
  - `verify_segment_graph_replays`: segmented verify graph replay;
  - `verify_graph_replay_rate`: total graph hit rate.

Direct short-prompt A/B:

| Case | Digest | Verify avg | L3 bucket | L16 bucket | Full graph | Segment graph |
|---|---:|---:|---:|---:|---:|---:|
| vseg forced to 12 by config | `13e18f2e...26a7b` | `112.016 ms` | `67.921 ms` | `116.715 ms` | `0` | `9` |
| decoupled vseg48 | `13e18f2e...26a7b` | `92.301 ms` | `37.441 ms` | `96.268 ms` | `11` | `0` |

Official benchmark-prompt comparison:

| Case | Digest | Draft avg | Verify avg | Verify profile | Hit | Accept | Graph path | Bucket highlights |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| seg12/vseg12 default | `29b53727...2161` | `19.344 ms` | `119.301 ms` | `117.659 ms` | `0.7672` | `0.6154` | segment graph `100%` | L4 `73.447 ms`, L16 `125.801 ms` |
| seg24/vseg24 | `45d6d91f...7814` | `20.584 ms` | `125.727 ms` | `124.986 ms` | `0.6852` | `0.5437` | segment graph `100%` | L14 `123.104 ms`, L16 `125.300 ms` |
| seg12/vseg48 decoupled | `45d6d91f...7814` | `20.584 ms` | `85.350 ms` | `83.874 ms` | `0.6994` | `0.4661` | full graph `100%` | L14 `85.606 ms`, L16 `83.626 ms` |

Interpretation:

- Full verify graph is a real latency lever:
  - direct short prompt improved `112.0 -> 92.3 ms`;
  - official benchmark prompt improved `119.3 -> 85.4 ms` against the default segment-12 baseline.
- The short-prompt length-3 bucket hit `37.4 ms`, which is the first measured nano verify bucket inside the desired `25-35 ms` neighborhood. It is not enough because long buckets still remain `~84-96 ms`.
- The official benchmark digest and acceptance trajectory changed under seg12/vseg48. Therefore this is not a safe default optimization yet.
- Segment size is entangled with cache/prefetch trajectory. A larger verify graph changes when verify metadata and prefetch updates are published, which can change future route hits and accepted tokens even at temperature `0.0`.
- This experiment narrows the optimization target:
  - first make full-verify-graph semantics and cache/prefetch publication match default segment-12 behavior, or move segment metadata/prefetch work off the critical path while keeping the same publication order;
  - then optimize the graph-internal MoE route/GEMM/H2D work that keeps long buckets above `80 ms`.

### 12.19 Full-graph follow-up A/B experiments

Question:

- Can full verify graph keep its latency benefit while restoring the default segment-12 cache/prefetch trajectory?
- Can the existing compact GPU-route experiment reduce full-graph MoE work?

Experiment A: full graph plus segment metadata lifecycle.

Implementation:

- Added default-off `NANOVLLM_VERIFY_FULL_GRAPH_SEGMENT_METADATA=1`.
- In full verify graph mode, it offloads verify metadata by logical metadata boundaries, defaulting to `dual_queue_segment_size=12`, even when the graph replay shape uses vseg48.
- It also calls `complete_verify_round()` instead of the previous full-graph `discard_verify_round()`.
- `ModelRuntimeMetaRecorder` now sizes the host-buffer pool from the logical metadata segment size under this experiment, so all segment metadata handles can be queued.

Command shape:

```bash
NANOVLLM_VERIFY_FULL_GRAPH_SEGMENT_METADATA=1 \
python scripts/bench_acceptance_predictor.py \
  --output-dir /tmp/nano_verify_fullgraph_segmeta_t0 \
  --output-lens 64 \
  --segment-sizes 12 \
  --verify-segment-size-override 48 \
  --predictor-modes off \
  ...
```

Result:

| Case | Digest | Draft avg | Verify avg | Verify profile | Hit | Accept | Miss/L | Active/L | Buckets |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| full graph baseline seg12/vseg48 | `45d6d91f...7814` | `20.584 ms` | `85.350 ms` | `83.874 ms` | `0.6994` | `0.4661` | `33.66` | `112.00` | L14 `85.606`, L16 `83.626` |
| full graph + segment metadata | `247f3474...b264` | `19.295 ms` | `105.004 ms` | `104.253 ms` | `0.6459` | `0.3955` | `37.09` | `104.73` | L4 `75.116`, L12 `89.859`, L16 `109.695` |

Additional profile counters:

| Counter | Value |
|---|---:|
| `model_verify_full_graph_segment_metadata_count` | `10` |
| `model_verify_full_graph_segment_metadata_segments` | `40` |
| `model_verify_segment_metadata_enqueue_count` | `40` |
| `model_dual_queue_round_end_discard_count` | `0` |
| `model_dual_queue_verify_phase_submit_count` | `0` |

Interpretation:

- Segment metadata alone does not restore default digest or acceptance trajectory.
- It also worsens verify latency from `85.35 ms` to `105.00 ms`.
- The missing piece is not just metadata publication order. The default segmented verify graph also runs `on_verify_segment_start()` and verify-segment prefetch between graph replays, changing cache residency before later verify layers.
- Full graph removes those inter-segment prefetch/publish windows. That is part of its speedup and also part of its semantic/cache trajectory change.

Experiment B: full graph plus existing compact GPU-route path.

Command shape:

```bash
NANOVLLM_VERIFY_COMPACT_GPU_ROUTES=1 \
python scripts/bench_acceptance_predictor.py \
  --output-dir /tmp/nano_verify_fullgraph_compact_t0 \
  --output-lens 64 \
  --segment-sizes 12 \
  --verify-segment-size-override 48 \
  --predictor-modes off \
  ...
```

Result:

| Case | Digest | Draft avg | Verify avg | Verify profile | Hit | Accept | Miss/L | Active/L | Buckets |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| full graph baseline seg12/vseg48 | `45d6d91f...7814` | `20.584 ms` | `85.350 ms` | `83.874 ms` | `0.6994` | `0.4661` | `33.66` | `112.00` | L14 `85.606`, L16 `83.626` |
| full graph + compact GPU routes | `f847d907...4ec` | `20.625 ms` | `90.891 ms` | `89.569 ms` | `0.7147` | `0.3586` | `29.67` | `104.00` | L5 `60.202`, L7 `96.394`, L16 `92.073` |

Interpretation:

- The existing compact GPU-route implementation is not a full-graph optimization. It worsened `85.35 -> 90.89 ms`.
- This agrees with the earlier torch-profiler observation that the current compact path adds extra masking/sorting/copy work and does not reduce the dominant graph body enough.
- A useful compact route optimization must be redesigned as a true end-to-end compact route kernel/data layout, not a fixed-shape post-filter around the current grouped-GEMM path.

Conclusion from 12.19:

- Do not pursue "full graph + delayed segmented metadata" as the main path.
- Do not promote the current compact GPU-route experiment.
- The next useful optimization needs a lower-level MoE execution change:
  - replace nano's graph-safe hybrid MoE grouped-GEMM/route layout with a ktransformers-style or custom compact route operator;
  - or port the relevant ktransformers full-GPU/CPUInfer expert operator into nano and compare per-kernel latency under the same verify buckets.

## 13. Final Answers to the Four Questions

### 13.1 Latency breakdown: ktransformers vs nano draft/verify

ktransformers is a decode baseline, not a speculative draft/verify engine. Its measured hot path is:

| ktransformers stage | Measured latency / attribution |
|---|---:|
| TTFT | `~619-628 ms` |
| decode graph replay TPOT | `33.05-33.69 ms/token` |
| short profile replay | `35.28 ms/token` |
| PyTorch-visible CUDA, rough per graph token | `~8-9 ms/token` |
| hidden/native part | CPUInfer expert compute + CPU/GPU sync + copy-back |

ktransformers call-chain summary:

1. CUDA graph replays one decode token.
2. Attention, dense projections, router/topk, lm_head are CUDA-side.
3. All selected top-8 experts per layer run through `KExpertsCPU`/CPUInfer.
4. There is no GPU expert cache split, no mixed GPU/CPU route merge, no predictive expert prefetch loop, and no multi-token verify acceptance loop.

nano draft measured hot path:

| nano draft component | Representative value |
|---|---:|
| draft total, temp-0 endpoint | `19.38 ms/forward` |
| draft CUDA graph replay | `15.99 ms/forward` |
| prepare decode | `0.11 ms/forward` |
| sample | `0.84-0.86 ms/forward` |
| remaining residual | `~2.4 ms/forward` |

Important draft conclusion:

- `--draft-tpot-td-ms 19` does not cap performance.
- The draft graph body itself is already about `16 ms`, so draft cannot reach `10-11 ms` through Python/sampler cleanup alone.

nano verify measured hot path:

| nano verify component / case | Representative value |
|---|---:|
| default temp-0 endpoint, K=15, segment=12 | `118.41 ms/call` |
| profile forward inside that call | `116.75 ms/call` |
| metadata enqueue | `~0.75 ms/call` |
| visible prefetch overhead | `~8.34 ms/call` |
| active routes per layer | `~98` |
| miss routes per layer | `~22.8` |
| verify graph hit | `100%` |
| K=4, segment=12 temp-0 | `73.39 ms/call` |
| device-LUT scalar-copy experimental path | `97.48 ms/call`, but digest/cache trajectory changed |
| compact GPU-route experimental path | `112.15 ms/call`, digest preserved in temp-0 endpoint |
| all-CPU verify experimental path | `141.60 ms/call`, digest changed |
| decoupled full verify graph, direct short prompt | `92.30 ms/call`, digest preserved for that prompt |
| decoupled full verify graph, official prompt | `85.35 ms/call`, digest/acceptance trajectory changed |
| full verify graph + segmented metadata experiment | `105.00 ms/call`, digest/acceptance trajectory changed |
| full verify graph + current compact GPU-route experiment | `90.89 ms/call`, digest/acceptance trajectory changed |
| verify len 4 bucket, default endpoint | `73.45 ms/call` |
| verify len 3 bucket, decoupled full graph short prompt | `37.44 ms/call` |
| verify len 16 bucket, default endpoint | `125.80 ms/call` |
| verify len 16 bucket, decoupled full graph official prompt | `83.63 ms/call` |

Eager decomposition, used only for attribution because graph replay bypasses Python hooks:

| Eager verify proxy component | Per-verify sum |
|---|---:|
| route CPU | `6.239 ms` |
| plan CPU | `29.285 ms` |
| CPU prepare | `4.465 ms` |
| kt_direct CPU compute | `2.028 ms` |
| GPU exec CUDA event sum | `140.854 ms` |
| MoE CPU wall sum | `142.757 ms` |

Graph-path torch profiler:

| CUDA category | Observed total |
|---|---:|
| H2D pinned copies | `498-669 ms` across profile window |
| `aten::copy_` CUDA | `453-570 ms` |
| `_grouped_gemm_forward_kernel` | `160-318 ms` |
| GEMV kernels | `42-89 ms` |
| `aten::mm` | `29-44 ms` |

The exact graph-internal per-layer split needs nsys/NVTX or a graph-aware probe; Python hooks cannot decompose replay.

### 13.2 Why nano is much slower

Confirmed reasons:

1. The workload is not equivalent.
   - ktransformers measures one-token decode.
   - nano verify validates `draft_len + 1` tokens and often sees `~98` active routes/layer in the tested endpoint.

2. nano verify does substantial GPU work even when some experts are on CPU.
   - kt_direct CPU compute measured only `~2 ms` per verify in eager decomposition.
   - GPU execution, plan/route, H2D/copy, and merge dominate.

3. The graph-safe verify plan computes too broad a GPU route set.
   - Code inspection showed all routes go through GPU grouped GEMM with zeroed weights for miss routes.
   - The compact-route endpoint experiment preserved digest and reduced verify by `~6.3 ms`, confirming the effect exists.
   - The current simple compact implementation is not sufficient because profiler-visible copy/GEMM work increased under torch profiler.

4. All-CPU verify is not a drop-in fix.
   - The all-CPU experiment worsened verify to `141.60 ms`, changed digest, and dropped route hit rate to `0`.
   - ktransformers' `33 ms` CPU expert result is one-token decode, not multi-token nano verify.

5. Predictive prefetch is a tradeoff, not pure overhead.
   - Disabling or skipping prefetch lowered visible prefetch work but collapsed hit rate and worsened verify.
   - Device-LUT scalar-copy and batched publish improved verify, but changed digest/cache trajectory, so they are not safe defaults yet.

6. Draft is already graph-bound.
   - Draft total is `~19 ms`; graph replay is `~16 ms`.
   - The performance gap to a `10-11 ms` A100 full-GPU graph path is mostly inside captured kernels/work, not Python launch overhead.

7. Earlier very high nano verify numbers had a confirmed graph-bucket artifact.
   - Missing verify bucket `16` caused graph fallback and `~161 ms` verify.
   - Auto-extending buckets fixed that class of error, but graph-hit verify still remains far above the target.

8. Verify segment replay and metadata publication are also part of the gap.
   - Decoupling verify segment size and replaying a full verify graph reduced official-prompt verify from `119.30 ms` to `85.35 ms`.
   - The same change altered digest/acceptance trajectory in the official benchmark, so the benefit is real but not yet safe as a default.
   - This shows nano pays a fixed segment/metadata cost that ktransformers' one-token decode path does not have.

9. The full-graph semantic difference is not solved by delayed segmented metadata alone.
   - Full graph plus segment metadata worsened verify to `105.00 ms` and still changed digest.
   - Therefore the default segmented path's inter-segment prefetch/publish windows are part of the observed cache trajectory.
   - Preserving exact behavior while using one full graph likely requires either replay-safe in-graph segment events or accepting that full graph is a different cache schedule.

10. The current compact GPU-route experiment is not the needed operator replacement.
    - Full graph plus compact GPU routes worsened verify to `90.89 ms`.
    - A useful compact route optimization must remove work end to end, not add graph-side filtering around the existing layout.

### 13.3 How to optimize nano toward draft 10-15ms and verify 25-35ms

Draft target:

- Near-term `15 ms` is plausible only if graph body is reduced or the measurement platform is faster than the current RTX 4090 environment.
- `10-11 ms` on A100 requires kernel/graph parity with the known faster full-GPU graph path, not just cleanup.

Draft optimization actions:

1. Use nsys on draft graph replay with NVTX ranges around attention, router/topk, MoE, lm_head, sampler, predictor, and prefetch metadata.
2. Compare nano draft graph kernels with the A100 ktransformers full-GPU graph path:
   - dense/GEMV kernels;
   - batch-1/top-k MoE kernels;
   - topk/router kernels;
   - lm_head/sampler boundary.
3. Replace small-batch GPU MoE with a specialized batch-1/top-8 CUDA graph-friendly kernel if grouped GEMM is the difference.
4. Capture or fuse the draft tail:
   - static logits buffers;
   - avoid duplicate lm_head/predictor work;
   - avoid CPU-visible sampler sync until absolutely needed.
5. Reuse all decode/prefetch metadata buffers and remove small per-token allocations/copies.

Verify target:

- `25-35 ms` is not supported by the current K=15/segment=12 workload without a substantial execution rewrite.
- With current endpoint, verify length and active-route count are too high: K=4/segment=12 measured `73.39 ms`, and the default endpoint's actual verify-length-4 bucket measured `73.45 ms`.
- Decoupled full verify graph proved one important fixed cost: official-prompt verify improved to `85.35 ms`, and a short-prompt length-3 bucket reached `37.44 ms`.
- To reach `25-35 ms`, nano must keep that full-graph style benefit while preserving default cache/prefetch trajectory, and then replace or rewrite the graph-internal hybrid MoE path.

Verify optimization actions:

1. Treat full verify graph as a performance probe, not a default-safe fix yet.
   - Segment metadata replay alone failed (`105.00 ms`, digest changed).
   - To make it semantically safe, the default segment prefetch/publish lifecycle must be reproduced, not only the metadata offload order.
   - If exact cache schedule must be preserved, this may require graph-safe segment events or a different prefetch design that is mathematically/cache deterministic across graph shapes.
2. Replace the current graph-safe hybrid MoE plan:
   - do not run CPU miss routes through GPU GEMM;
   - compact active GPU routes end to end by bucket;
   - avoid activation/scatter on inactive tail rows;
   - keep deterministic route order and digest equality.
   - The existing `NANOVLLM_VERIFY_COMPACT_GPU_ROUTES=1` experiment is insufficient because it worsened full-graph verify to `90.89 ms`.
3. Import or wrap ktransformers-style expert execution where profiling proves it helps:
   - `KExpertsCPU`/CPUInfer submit/sync model;
   - static pinned input/output buffers;
   - NUMA-aware worker ownership;
   - batch-1/top-8 kernels from the faster ktransformers full-GPU path if the grouped-GEMM profile remains dominant.
4. Move plan/route/merge into static graph-friendly buffers:
   - preallocated route index buffers;
   - preallocated merge buffers;
   - no per-layer `zeros`, `sort`, `where`, or `index_copy_` churn unless profiler proves it is cheap.
5. Make prefetch publish semantically stable:
   - batched publish and device-LUT scalar copy both showed latency upside;
   - they must preserve cache trajectory/digest before becoming defaults.
6. Reduce critical-path H2D:
   - nsys/torch profiler confirmed large expert-weight H2D inside the decode/profile window;
   - prefetch must be earlier, less bursty, and not synchronized with verify graph replay.
7. Move or capture verify lm_head only after measuring per bucket.
   - Current measured `verify_lm_head` was small in one profile, so this is secondary unless a bucket-specific profile shows otherwise.
8. Keep adaptive verify expert mode as a later measured policy:
   - the tested all-CPU verify path was worse at `141.60 ms`, so it is not a default replacement;
   - it may still be useful for very short verify buckets only if a per-bucket A/B beats the hybrid path and preserves digest.

### 13.4 Which implementation direction is better

Recommended path:

- Optimize nano-vllm-moe spec using ktransformers hot-path techniques.

Reason:

- nano already owns the hard speculative-decoding semantics:
  - draft/verify scheduling;
  - KV accept/rollback;
  - standard sampling acceptance;
  - TPOT stop policy;
  - acceptance predictor;
  - expert cache;
  - predictive prefetch;
  - mixed CPU/GPU expert placement.
- The measured bottlenecks are inside nano's hot path and can be attacked incrementally with digest/latency tests.
- ktransformers should be used as a source of proven implementation ideas:
  - `KExpertsCPU`/CPUInfer submission model;
  - pinned static CPU buffers;
  - NUMA-aware worker pool;
  - simpler all-CPU expert decode path;
  - faster full-GPU graph kernels where portable.

Not recommended as first path:

- Port nano-vllm-moe spec into ktransformers.

Reason:

- It requires reimplementing the entire speculative control plane in a different engine before solving the proven bottlenecks.
- Correctness risk is high: sampling acceptance and KV rollback bugs can look like performance gains if not caught by deterministic digest and quality tests.
- It delays direct work on the measured latency sources: graph-body draft kernels, verify route/plan/GPU work, H2D/prefetch, and mixed merge.

When porting into ktransformers becomes justified:

- nsys proves ktransformers has a non-portable model/layer kernel advantage that nano cannot reasonably import;
- the target serving stack must be ktransformers for product reasons;
- a minimal ktransformers spec proof-of-concept matches nano's draft/verify semantics and shows a fair multi-token verify advantage, not only single-token decode advantage.

Current decision:

- Continue optimizing nano-vllm-moe spec first, but be more aggressive than parameter tuning:
  - make decoupled/full verify graph semantically safe;
  - replace nano's hot MoE kernels or expert execution path with ktransformers-derived operators where per-kernel A/B proves a win;
  - preserve nano's existing spec correctness harness, digest checks, KV rollback, and predictor flow.
- Treat "implement nano spec inside ktransformers" as the fallback after operator replacement is tested:
  - it is justified if ktransformers' faster decode path depends on non-portable graph/model structure;
  - it should start as a minimal proof-of-concept that reproduces nano's draft/verify outputs and acceptance decisions on fixed prompts before any performance claim.
