# Verify Unified Follow-up Report (2026-04-28)

## Scope

This follow-up reran verify profiling under the new constraints:

- no `S = N` special-case path
- verify-history feedback must stay enabled
- optimization must preserve the unified verify path
- correctness is higher priority than latency

I kept the profiling/instrumentation improvements, but **did not keep any runtime optimization in the final worktree** because every latency-changing attempt either regressed `S < N`, changed output digests, or both.

## Landed Code Changes

These changes remain in the worktree because they are measurement-only or correctness-neutral:

- `nanovllm/engine/model_runner.py`
  - add per-verify MoE profile accumulation via `verify_*` counters
  - add optional one-shot verify `torch.profiler` capture via `NANOVLLM_VERIFY_TORCH_PROFILE_DIR`
- `examples/benchmarks/verify_breakdown_profile.py`
  - unified benchmark harness for `100% / 75% / 50%` cache-ratio verify profiling
  - remove the old `S=N` fastpath A/B cases
  - add finer verify metadata counters to the output summary

Attempted runtime optimizations were all reverted before finishing the turn.

## Correctness Status

Compute-node validation on `nano_moe`:

- smoke after final revert:
  - `tests/test_placement_spec.py`
  - `tests/test_model_runner_spec_modes.py`
  - `tests/test_prefetch_runtime_meta.py`
  - `tests/test_prefetch_runtime.py`
  - `tests/test_prefetch_global_queue.py`
  - result: `17 passed`
- broader validation during experiment runs:
  - above tests plus `tests/test_cpu_gpu_parallel_moe.py`
  - result: `18 passed`

Runtime optimization attempts were rejected because speculative end-to-end output digests drifted in `75%` and/or `50%` cases.

## Experiment Artifacts

Baseline unified-path profile:

- JSON: `/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/verify_breakdown_unified_baseline_20260428_193124.json`
- log: `/home/mumura/moe_spec/logs/verify_unified_baseline_20260428_193124.log`

Failed attempt 1:

- JSON: `/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/verify_breakdown_unified_opt_20260428_194640.json`
- log: `/home/mumura/moe_spec/logs/verify_unified_opt_20260428_194640.log`

Failed attempt 2:

- JSON: `/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/verify_breakdown_unified_opt2_20260428_195701.json`
- log: `/home/mumura/moe_spec/logs/verify_unified_opt2_20260428_195701.log`

Rejected plan-only candidate:

- JSON: `/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/verify_breakdown_unified_final_20260428_201140.json`
- log: `/home/mumura/moe_spec/logs/verify_unified_final_20260428_201140.log`

Torch profiler trace for verify eager path:

- `/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/verify_breakdown_unified_final_20260428_201140_torchprof/cache100_torch_profile/verify_forward_rank0.json`

## Unified Baseline Breakdown

### 100% cache ratio

- `verify_ms_per_call = 165.1 ms`
- `verify_plan_ms_per_call = 17.0 ms`
- `verify_gpu_compute_ms_per_call = 22.6 ms`
- `metadata_collect_ms_per_call = 61.6 ms`
- `cpu_compute_ms_per_call = 0.0 ms`

Main contributors:

- eager verify forward dispatch / launch / sync
- verify metadata collect/offload on host
- plan construction

### 75% cache ratio

- `verify_ms_per_call = 343.8 ms`
- `verify_plan_ms_per_call = 32.2 ms`
- `verify_cpu_compute_ms_per_call = 116.7 ms`
- `metadata_collect_ms_per_call = 54.3 ms`
- `submit_after_ms_per_call = 2.1 ms`
- `publish_ms_total = 11.2 ms`

Main contributors:

- CPU expert compute
- verify metadata collect/offload
- plan construction
- GPU waits on CPU path

### 50% cache ratio

- `verify_ms_per_call = 419.1 ms`
- `verify_plan_ms_per_call = 33.8 ms`
- `verify_cpu_compute_ms_per_call = 190.7 ms`
- `metadata_collect_ms_per_call = 32.8 ms`
- `submit_after_ms_per_call = 3.7 ms`
- `publish_ms_total = 21.9 ms`

Main contributors:

- CPU expert compute dominates
- CPU->GPU merge cost is non-trivial
- verify metadata collect plus publish/submit_after still visible

## Optimization Attempts

### Attempt 1: small-meta collect acceleration + publish snapshot fastpath + merged CPU output

Code was applied transiently and then reverted.

Observed effect:

- `100%`: verify `165.1 -> 161.6 ms`, plan `17.0 -> 14.9 ms`, metadata collect `61.6 -> 22.7 ms`
- `75%`: verify `343.8 -> 424.5 ms`, CPU compute `116.7 -> 205.0 ms`, publish `11.2 -> 38.8 ms`
- `50%`: verify `419.1 -> 531.3 ms`, CPU compute `190.7 -> 271.4 ms`, submit_after `3.7 -> 14.1 ms`

Correctness result:

- `75%` output digest drifted from baseline
- `50%` path also showed unstable behavior across attempts

Conclusion:

- not acceptable
- reverted

### Attempt 2: revert merged CPU scatter, keep metadata/publish speedups and threadpool reuse

Observed effect:

- metadata collect stayed much lower than baseline
- CPU compute still regressed
- `75%` and `50%` digests still drifted in benchmark runs

Conclusion:

- latency reduction in feedback path changed speculative cache timing enough to alter which experts were CPU/GPU-resident later
- numerics became unstable at the sequence-output level
- reverted

### Attempt 3: only all-GPU-route plan shortcut inside unified planner

Observed effect:

- no stable win versus baseline
- `75%` benchmark digest still drifted on a noisy node

Conclusion:

- not merged

## Current Conclusions

### Plan overhead

- I did not find a latency optimization that could be kept with enough confidence.
- The best apparent plan reduction came from shortcuts that did not survive correctness screening.
- Baseline `plan_ms` remains a real cost:
  - `100%`: `17.0 ms/call`
  - `75%`: `32.2 ms/call`
  - `50%`: `33.8 ms/call`

### CPU path

- CPU expert compute is still the dominant `S < N` bottleneck.
- Safe micro-optimizations in Python did not help enough.
- More aggressive changes must preserve deterministic accumulation order and avoid changing whether later steps run CPU or GPU due to earlier feedback timing.

### Verify-history feedback

- Feedback overhead is large enough to optimize.
- But every successful-latency attempt also changed speculative cache timing and destabilized end-to-end outputs.
- The dominant visible baseline feedback sub-costs are:
  - `metadata_collect_ms_per_call`
  - `submit_after_ms_per_call`
  - `publish_ms_total`

## Remaining Bottlenecks

- `100%`: verify eager forward overhead, plus metadata collect
- `75%`: CPU expert compute, metadata collect, GPU wait on CPU path
- `50%`: CPU expert compute, CPU->GPU merge, metadata collect, publish/submit_after

## Recommended Next Steps

1. Add a stricter correctness harness for speculative runs.
   - Compare per-step accepted tokens and verify token IDs, not just final sequence digest.
   - Record cache residency and CPU/GPU route ratio per verify step.

2. Isolate feedback optimization from cache-visibility timing.
   - If metadata processing is sped up, delay cache-state publication to the same semantic boundary as baseline.
   - This avoids turning a profiling optimization into a behavior change.

3. Revisit CPU path only with deterministic accumulation preserved.
   - Candidate directions:
     - grouped CPU expert kernel or fused batched CPU MLP
     - stable per-expert accumulation order
     - pinned-buffer and overlap changes only if publication timing remains unchanged

4. Prioritize verify CUDA Graph on the pure GPU path.
   - The earlier evidence still holds: eager verify has large non-compute overhead even without CPU fallback.
   - This remains the cleanest high-upside path that is less likely to perturb speculative cache state.
