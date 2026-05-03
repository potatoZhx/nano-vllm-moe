# Verify Stage Breakdown and Optimization Report

Date: 2026-04-28

## 1. Experiment Setup

Repository: `/home/mumura/moe_spec/nano-vllm-moe`

Model: `/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B`

Compute: Slurm A100 partition, `gpu15-A100-E2-3U`, selected physical GPU 4 after runtime probe (`0 MiB / 0% util` before run). Log:

- `/home/mumura/moe_spec/logs/verify_breakdown_profile_20260428_152603.log`

Benchmark config:

- `num_seqs=1`, `input_len=24`, `output_len=12`
- `max_draft_tokens=4`, `draft_top_c=0`
- `enforce_eager=false`, `engine_profile_cuda_sync=true`
- `cpu_expert_execution_enabled=true`
- `cpu_gpu_parallel_execution_enabled=true`
- `cpu_gpu_parallel_min_cpu_route_ratio=0.0`
- `spec_enable_prefetch=true`, `prefetch_verify_wait_ms=1.0`

Result artifacts:

- Summary JSON: `/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/verify_breakdown_profile_20260428_152603.json`
- Raw per-case outputs: `/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/verify_breakdown_profile_20260428_152603_raw`
- Torch profiler trace: `/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/verify_breakdown_profile_20260428_152603_torchprof/cache100_fastpath_on_torch_profile/verify_forward_rank0.json`
- Torch profiler summary: `/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/verify_breakdown_profile_20260428_152603_torchprof/cache100_fastpath_on_torch_profile/verify_forward_rank0_summary.json`

Validation:

- Compute-node pytest: `tests/test_placement_spec.py tests/test_model_runner_spec_modes.py`, 9 passed.
- Deterministic digest: all spec cases matched standard output digest `01f742981c472604cec6231307387904c81f13013c3522b8946bffe69456886f`.

## 2. Code Evidence

Current graph policy:

- `ModelRunner.run_model()` uses eager whenever `is_prefill` is true. Standard graph replay only applies to decode (`is_prefill=false`) and draft graph is a decode-mode policy.
- `run_verify()` explicitly calls `prepare_prefill(seqs)` and then directly calls `self.model(input_ids, positions)`, because verify needs logits for every queried token position, while `compute_logits()` would slice to last-token logits.
- Therefore current verify cannot hit either `standard_graph_replay_count` or `draft_graph_replay_count`.

Relevant code:

- `nanovllm/engine/model_runner.py:751-764`
- `nanovllm/engine/model_runner.py:766-802`
- `nanovllm/engine/model_runner.py:1027-1103`
- `nanovllm/engine/model_runner.py:1163-1195`

Verify routing path:

- `build_verify_plan_gpu()` previously delegated to `build_prefill_plan_gpu()`, which always remaps cache slots, builds GPU route indices, builds `cpu_route_mask`, and constructs CPU task layout.
- I added an S=N all-cached fast path guarded by `NANOVLLM_VERIFY_ALL_CACHED_FASTPATH`. It is enabled by default and can be disabled with `NANOVLLM_VERIFY_ALL_CACHED_FASTPATH=0`.

Relevant code:

- `nanovllm/expert/placement.py:150-191`
- `nanovllm/expert/placement.py:319-360`

Additional instrumentation:

- `run_verify()` now aggregates per-verify MoE profile keys as `verify_*`, so route/plan/GPU/CPU/merge/parallel-wall data are not mixed with draft/prefill.
- Optional torch profiler capture is controlled by `NANOVLLM_VERIFY_TORCH_PROFILE_DIR`; it captures only the first verify forward and is disabled by default.

## 3. Verify Cost Breakdown

Per-call median from one repeat per case. `metadata_verify_ms_total` is total over all verify calls in the case, not per call.

| case | verify ms/call | forward | route | plan | GPU compute | CPU compute | merge | parallel wall | CPU route ratio | metadata verify total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cache100 fastpath off | 214.8 | 213.1 | 9.2 | 22.8 | 28.7 | 0.0 | 0.0 | 0.0 | 0.000 | 263.9 |
| cache100 fastpath on | 185.3 | 183.6 | 8.0 | 14.2 | 26.7 | 0.0 | 0.0 | 0.0 | 0.000 | 229.7 |
| cache75 baseline | 489.8 | 487.3 | 11.0 | 45.4 | 36.6 | 159.5 | 13.2 | 238.7 | 0.240 | 264.8 |
| cache50 baseline | 575.6 | 573.4 | 12.4 | 47.1 | 36.3 | 283.5 | 30.6 | 390.1 | 0.478 | 56.1 |

Interpretation:

- 100% cache: no CPU expert path, no publish, no offloaded expert compute. Main cost is eager full-model verify forward. MoE GPU compute is only about 26-29 ms/call; route+plan+scatter is about 27-37 ms/call. The remaining large portion is transformer/attention/LM head eager scheduling and synchronization overhead.
- 75% cache: CPU route ratio is about 24%. CPU compute becomes the main incremental bottleneck. `parallel_wall` is 238.7 ms/call, with GPU waiting about 130.0 ms/call for CPU-side work.
- 50% cache: CPU route ratio is about 48%. CPU compute dominates more strongly. `parallel_wall` is 390.1 ms/call, with GPU waiting about 250.3 ms/call.
- Bottleneck changes with cache ratio: S=N is graph/eager dispatch dominated; S<N is CPU expert critical path dominated.

## 4. S=N High Verify Latency

S=N facts from cache100 fastpath-on:

- `verify_cpu_route_ratio=0.0`
- `verify_cpu_compute_ms_per_call=0.0`
- `verify_cpu_to_gpu_merge_ms_per_call=0.0`
- `publish_ms_total=0.0`
- `standard_graph_replay_count=0`
- `draft_graph_replay_count=8`
- Standard graph decode in the same script has `standard_graph_replay_count=11`, `graph_hit_rate=1.0`, elapsed 270.0 ms for 12 output tokens.

Torch profiler evidence for first S=N verify forward:

- Verify tokens in captured call: 3
- Kernel events: 3898
- CUDA kernel duration sum: 24.16 ms
- CUDA memcpy duration sum: 5.31 ms
- CUDA runtime duration sum: 32.29 ms
- `cudaLaunchKernel` count from key averages: 3562, CPU self time 22.84 ms
- Top kernels are grouped GEMM / BF16 GEMM kernels; each grouped GEMM is mostly around 80-137 us.

Conclusion:

- The S=N verify path is not slow because of CPU experts or expert weight movement.
- It is slow because verify is a small-token prefill-style eager forward. It launches thousands of small kernels and pays Python/PyTorch/CUDA runtime scheduling, graph-miss, and explicit end-of-forward synchronization costs.
- The actual MoE GPU compute is a minority of verify latency, so optimizing only expert compute will not fix S=N.

## 5. Optimization Attempts

### 5.1 S=N Verify Plan Fast Path

Change:

- If `expert_cache.num_slots >= num_experts`, `build_verify_plan_gpu()` now constructs an all-GPU plan directly.
- It skips `remap_experts_to_slots()`, `nonzero(~gpu_mask)`, and CPU task layout construction.
- Correctness guard: CPU route mask is all false, CPU task fields are `None`, and output digest matches standard.

Result:

| metric | fastpath off | fastpath on | delta |
|---|---:|---:|---:|
| end-to-end elapsed | 1102.5 ms | 958.8 ms | -13.0% |
| verify ms/call | 214.8 | 185.3 | -13.7% |
| verify forward ms/call | 213.1 | 183.6 | -13.8% |
| verify plan ms/call | 22.8 | 14.2 | -37.8% |

Assessment:

- Effective but not sufficient. It removes avoidable CPU-layout planning in S=N, but most latency remains outside plan construction.

### 5.2 Disable Verify-History Prefetch Feedback

Change:

- Set `prefetch_use_verify_history=false`.
- This removes verify-history queue updates and mark-access work from the verify metadata path.

Result:

| cache ratio | baseline verify ms/call | no verify-history verify ms/call | delta | end-to-end effect |
|---|---:|---:|---:|---:|
| 75% | 489.8 | 379.7 | -22.5% | elapsed improved 3197.6 -> 2602.0 ms |
| 50% | 575.6 | 478.6 | -16.8% | elapsed worsened 6419.6 -> 9721.8 ms |

Assessment:

- This is a verify-local optimization, but it is not universally safe as a global default. At 50%, verify itself improves, but full generation worsens, likely because verify-history feedback helps future prefetch/cache state enough to offset its own metadata cost.
- Recommended action: make verify-history feedback adaptive, not simply disabled. Disable it only when its observed queue/update cost exceeds later prefetch benefit.

## 6. CUDA Graph Feasibility for Verify

Recommended graph granularity:

- Capture verify forward from after `prepare_prefill()`/context construction to before argmax/metadata offload.
- Include embedding, attention, MoE route/plan if graph-safe, MoE GPU kernels, final norm, and LM head.
- Keep acceptance, sequence mutation, draft KV rollback/accept, prefetch publish, metadata offload/observe, and cache update outside the graph.

Bucket strategy:

- Start with verify-token buckets `{1, 2, 4, 8, 16, 20}`.
- For each bucket, allocate static `input_ids`, `positions`, `cu_seqlens_q`, `cu_seqlens_k`, `slot_mapping`, `block_tables`, routing scratch, expert route buffers, output hidden/logit buffers.
- Pad to bucket length and mask unused token rows before accepting logits.

Can likely be captured:

- S=N verify forward with all experts cached and fixed bucket shape.
- Attention prefill kernels if context tensors point to fixed-address buffers.
- Fused/grouped MoE GPU execution if plan builder avoids graph-unsafe dynamic allocation and CPU branching.
- LM head and logits output into static buffer.

Difficult to capture:

- CPU expert execution and CPU/GPU heterogeneous overlap.
- Dynamic route count and `torch.nonzero`, `torch.unique`, dynamic sort layouts when shapes and allocations vary.
- Async prefetch publish, staging-to-active expert weight copy, and expert cache state mutation.
- Metadata offload / runtime meta observation, because it copies dynamic routing metadata to host and updates Python queues.

Hybrid graph plan:

- Phase A outside graph: build verify batch, choose bucket, copy/pad input/context into static buffers, publish any ready prefetched experts, and validate graph eligibility.
- Phase B inside graph: S=N or all-GPU verify forward replay.
- Phase C outside graph: slice real logits, argmax/acceptance, metadata offload, cache/prefetch feedback.
- For S<N, use partial graph only for non-MoE transformer/attention blocks if MoE CPU path remains dynamic. A better medium-term design is to split MoE into graphable GPU expert part plus outside-graph CPU merge.

Correctness requirements:

- Graph replay must validate a cache generation vector or a stable all-cached condition before replay.
- Static expert slot mapping must not change between capture and replay. If any active slot generation changes, invalidate or recapture the graph.
- Routing outputs must be checked against eager for a deterministic batch before enabling graph path.
- Padding must not affect logits for real token positions.
- All prefetch publish events must be complete before graph replay reads expert buffers.

## 7. ktransformers CUDA Graph Mechanisms

Sources:

- GitHub repository: https://github.com/kvcache-ai/ktransformers
- Docs: https://ktransformers.net/docs
- CPU-GPU expert scheduling tutorial: https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/experts-sched-Tutorial.md
- Balance serve graph runner code: https://github.com/kvcache-ai/ktransformers/blob/main/archive/ktransformers/server/balance_serve/inference/model_runner.py
- CPU expert graph-aware code: https://github.com/kvcache-ai/ktransformers/blob/main/archive/ktransformers/operators/experts.py
- Legacy CUDA graph runner: https://github.com/kvcache-ai/ktransformers/blob/main/archive/ktransformers/util/cuda_graph_runner.py

Findings:

- ktransformers uses bucketed graph capture. It builds `self.cuda_graphs`, one `torch.cuda.CUDAGraph()` per bucket, and chooses the smallest graph bucket that can hold current `num_tokens`.
- It keeps static GPU buffers for graph replay: `features_buf`, `page_idx_buf`, `page_offset_buf`, `bsz_tensor_buf`, `num_tokens_tensor_buf`, plus model output buffers.
- Before replay it updates static buffers and FlashInfer attention planning state, then calls `graph.replay()`.
- For decode split, it only uses graph when `is_prefill == False`; prefill remains outside graph.
- CPU expert integration uses pinned CPU buffers indexed by `cuda_graph_idx`: input activations, expert ids, weights, output CPU, and output GPU map. During graph capture it detects stream capture and submits CPU inference with the CUDA stream handle, then syncs and copies CPU output to the static GPU output map.
- Expert routing remains dynamic input to CPU kernels, but the memory addresses are static because the route tensors are copied into pinned per-bucket buffers before CPU inference.

Reusable ideas:

- Bucketed verify graphs by token count.
- Static per-bucket buffers for input, KV/page/slot metadata, routing, and output.
- A graph index passed through attention and MoE code so each bucket uses its own static buffers.
- Separate graph replay from dynamic request/acceptance logic.
- Generation/cache version checks before replay.

Incompatibilities with current nano-vllm-moe:

- Current verify uses `prepare_prefill()` tensors allocated per call, not static per-bucket context buffers.
- Current verify MoE plan uses dynamic PyTorch ops (`nonzero`, sorting, route layout construction) and Python-side CPU task layout.
- Current CPU expert path copies activations to CPU and schedules Python/ThreadPool work; it is not graph-capture-safe.
- Current prefetch runtime mutates active expert cache state outside a versioned graph contract.

Required module/interface changes:

- `ModelRunner`: add `capture_verify_cudagraph()` and `_replay_verify_graph()`, analogous to draft graph but with prefill context buffers and verify buckets.
- `Config`: add `verify_cuda_graph_enabled`, `verify_cuda_graph_bucket_steps`, `verify_cuda_graph_max_tokens`.
- `placement.py`: add graph-safe all-GPU verify plan path that uses fixed-size route buffers and avoids dynamic CPU route construction.
- `LayerExpertCache`: expose slot generation snapshots and a cheap validation method for graph eligibility.
- `runtime_meta_recorder/prefetcher`: separate graph replay from metadata collection; record routing metadata outside graph or into static device buffers copied after replay.

## 8. Priority Recommendations

1. Implement S=N verify CUDA Graph first. This is the cleanest target: no CPU expert path, stable expert weights, and measured bottleneck is launch/scheduling overhead.
2. Keep the S=N verify plan fast path. It is correct and reduced verify by 13.7% in the A/B run.
3. Add verify graph buckets `{1,2,4,8,16,20}` with static context buffers and generation checks.
4. Make verify-history feedback adaptive. It helps verify-local latency when disabled, but cache50 shows full-generation regression.
5. For S<N, optimize CPU expert critical path before graphing heterogeneous verify: CPU compute and GPU wait dominate at 75%/50%.
6. Once S=N graph is stable, evaluate hybrid graph for S<N by graphing non-MoE transformer blocks and keeping CPU expert/merge outside graph.
