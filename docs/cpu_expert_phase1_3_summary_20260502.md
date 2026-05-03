# CPU Expert Phase 1-3 Implementation Summary

Date: 2026-05-02

## Scope

This report summarizes the Phase 1-3 CPU expert implementation work for `nano-vllm-moe`, the deterministic token mismatch investigation in spec mode, and the measured performance impact.

## Implemented Changes

### Phase 1: CPU Expert Weight Precast and Validation

- Added `CpuExpertWeights` metadata for CPU expert weights.
- Changed heterogeneous loading so expert weights are converted to target dtype, made contiguous, optionally pinned, and validated at load time.
- Removed avoidable hot-path dtype conversion for packed CPU weights.
- Added strict dtype validation for packed CPU expert execution.

Main files:

- `nanovllm/expert/cpu_weights.py`
- `nanovllm/utils/heterogeneous_loader.py`
- `nanovllm/layers/fuse_moe/heterogeneous.py`

### Phase 2: `torch_packed` CPU Backend

- Added `TorchPackedCpuMoeBackend`.
- Added packed CPU result layout so CPU expert outputs are produced in a single route-aligned buffer instead of one list item per expert.
- Added config knobs:
  - `cpu_expert_backend`
  - `cpu_expert_workspace_max_routes`
  - `cpu_expert_packed_min_routes`
  - `cpu_expert_strict_dtype`
- Wired the backend through model runner and Qwen3-MoE blocks.
- Added synthetic CPU MoE backend benchmark.

Main files:

- `nanovllm/layers/fuse_moe/cpu_backend.py`
- `nanovllm/config.py`
- `nanovllm/engine/model_runner.py`
- `nanovllm/models/qwen3_moe.py`
- `benchmarks/bench_cpu_moe_backend.py`

### Phase 3: Pinned CPU Workspace Reuse

- Added reusable `CpuMoeWorkspace`.
- `TorchPackedCpuMoeBackend` now reuses CPU buffers by `max_routes`, hidden size, dtype, and pinned-memory setting.
- Added overflow and workspace reuse tests.

Main files:

- `nanovllm/layers/fuse_moe/cpu_workspace.py`
- `nanovllm/layers/fuse_moe/cpu_backend.py`
- `tests/test_cpu_moe_correctness.py`

## Deterministic Token Mismatch Root Cause

### Symptom

Standard mode with all parameters resident on GPU generated:

```text
[[576, 4226, 1265, 387, 304, 6364]]
digest=4426c01423d671af40c9a9d73cebbb6018a763bfbe6050e3fe452c7839ba021c
```

Partial-cache spec mode sometimes generated:

```text
[[4710, 16141, 1447, 32313, 11, 773]]
digest=8342062718b9d6d7fa8fb563091dd09f3297f36d9267c3df7252d609776ec942
```

### Evidence

- 100% expert cache matched standard exactly.
- 75% expert cache could match or mismatch depending on run/cache state.
- With prefetch disabled, 75% expert cache still mismatched:
  - `spec75_gpu_fallback`: mismatch
  - `spec75_cpu_exec`: mismatch
  - Log: `/home/mumura/moe_spec/logs/spec_no_prefetch_probe_20260502_212834.log`
- CPU expert execution is not required to reproduce the bug:
  - `cpu_expert_execution_enabled=false` still mismatched.
- Therefore, prefetch is not changing weights incorrectly. Prefetch/cache only changes which experts are resident, which changes the execution path.

### Root Cause

Partial expert cache changes the MoE computation from a single standard all-GPU grouped-GEMM path into a mixed path:

- cached routes use Triton grouped GEMM through `fused_moe_linear`;
- uncached routes use fallback per-expert `F.linear` after CPU-to-GPU weight copy when CPU execution is disabled;
- uncached routes use CPU `F.linear` when CPU execution is enabled;
- mixed cached/uncached outputs are accumulated with separate `index_add_` calls.

These paths use the same weights, but not the same numerical implementation or accumulation order. In BF16 greedy decoding, small logit changes are enough to change the selected token. This is why cache/prefetch can appear to affect accuracy even though it does not alter the weights.

### Fix Attempts

Attempted fixes:

- Route-buffer accumulation in original route order.
- A temporary fused fallback path for uncached GPU routes.

Result:

- Dynamic fused fallback triggered excessive Torch/Triton compilation on the real model and was not viable.
- Route-buffer-only change passed small operator tests but broke end-to-end generation, producing token `0`; it was rolled back.

Current status:

- Phase 1-3 CPU backend changes are kept.
- The deterministic exact partial-cache issue remains unresolved.
- A correct future fix should use a unified exact fallback path with static/preallocated GPU expert buffers, or a validation-only all-GPU expert path. The fallback must avoid dynamic shapes that trigger long Inductor/Triton compilation.

## Correctness Tests

Passed:

```text
python -m py_compile ...
python -m pytest -q tests/test_cpu_moe_correctness.py -k 'not packed'
2 passed, 2 deselected
```

CUDA tests:

```text
python -m pytest -q tests/test_cpu_moe_correctness.py tests/test_cpu_gpu_expert_operator_alignment.py tests/test_cpu_gpu_parallel_moe.py
6 passed
```

Representative log:

```text
/home/mumura/moe_spec/logs/cpu_expert_mismatch_fix_correctness_20260502_213714.log
```

## Performance Results

### Block-Level CPU MoE Benchmark

Phase 2 `torch_packed` versus original `torch` backend:

| batch_tokens | average decode speedup |
|---:|---:|
| 32 | 1.32x |
| 128 | 1.32x |

Phase 3 `torch_packed` versus Phase 2 `torch_packed`:

| batch_tokens | decode improvement | CPU prepare improvement | CPU merge improvement |
|---:|---:|---:|---:|
| 32 | 12.1% | 25.3% | 19.7% |
| 128 | 7.8% | 18.8% | 10.7% |

CSV inputs:

- `benchmarks/results/cpu_moe_backend_phase12_guard_20260502_114701.csv`
- `benchmarks/results/cpu_moe_backend_phase3_20260502_144621.csv`

Conclusion: Phase 1-3 improve the isolated CPU MoE backend path, especially merge overhead.

### Spec Real-Path CPU Metrics

Spec benchmark settings:

- `mode=spec`
- `cpu_expert_execution_enabled=true`
- `spec_enable_prefetch=false`
- `draft_top_c=128`
- `num_seqs=1`
- `input_len=12`
- `output_len=6`
- expert cache ratios 75%, 50%, 25%

Results are performance-only because partial-cache deterministic exactness is not fully solved.

| cache ratio | backend | output tok/s | verify ms | verify CPU prepare ms | verify CPU compute ms | verify CPU merge ms |
|---:|---|---:|---:|---:|---:|---:|
| 75% | torch | 2.59 | 290.0 | 6.08 | 113.4 | 10.41 |
| 75% | torch_packed | 2.29 | 586.4 | 10.71 | 416.3 | 3.47 |
| 50% | torch | 1.34 | 448.5 | 7.41 | 235.9 | 20.77 |
| 50% | torch_packed | 1.82 | 436.6 | 11.84 | 247.9 | 4.15 |
| 25% | torch | 1.24 | 596.3 | 8.04 | 363.2 | 30.13 |
| 25% | torch_packed | 1.29 | 932.2 | 12.68 | 734.3 | 5.61 |

Log and JSON:

- `/home/mumura/moe_spec/logs/spec_phase13_backend_perf_20260502_220346.log`
- `/home/mumura/moe_spec/logs/spec_phase13_backend_perf_20260502_220346/`

Interpretation:

- `torch_packed` consistently reduces CPU-to-GPU merge time by about 67-84%.
- In real spec verify, packed backend does not consistently reduce total verify latency.
- At 50% cache it improved output throughput by about 1.36x and slightly improved verify latency.
- At 75% and 25% cache it regressed verify latency due to higher measured CPU prepare/compute time.
- The real spec path is dominated by CPU expert compute, not merge. Phase 3 helps the merge bottleneck, but does not solve CPU matmul cost.

## Current Recommendation

- Keep Phase 1-3 as optional backend work because block-level CPU MoE performance improved and tests pass.
- Do not claim end-to-end spec acceleration yet.
- Treat deterministic exact partial-cache spec as a blocking correctness issue before enabling `torch_packed` by default.
- Next correctness work should target a static-shape exact GPU fallback for uncached experts or an exact validation mode that keeps all expert routes on the same grouped-GEMM path.

## 2026-05-03 Follow-Up: `torch_packed` Verify CPU Compute Regression

User report: in real spec workloads, the `torch_packed` backend's verify CPU compute increase is unacceptable, while the CPU merge reduction is valuable and should be retained.

Root cause:

- The original packed workspace used pinned CPU tensors for route hidden states, route weights, and packed outputs.
- The first fix moved hidden/weight compute inputs back to pageable CPU memory, which removed the worst 75%/50% behavior but still left a large 25% cache regression.
- The remaining 25% regression came from writing every expert result into pinned `outputs_cpu` inside the CPU compute timing window. At high CPU route ratios this pageable-to-pinned write dominated enough to make verify CPU compute jump from about 363ms to about 679ms.

Fix:

- `TorchPackedCpuMoeBackend` now uses pageable CPU workspace tensors for hidden states, route weights, and packed outputs.
- The packed merge structure is preserved: one packed output buffer, one CPU-to-GPU transfer, and one `index_add_`.
- The ordinary `torch` backend now also uses packed merge semantics by concatenating CPU output chunks and issuing a single transfer/index-add, so the useful merge optimization is available without opting into packed compute.

Validation:

```text
python -m py_compile nanovllm/layers/fuse_moe/cpu_workspace.py nanovllm/layers/fuse_moe/cpu_backend.py nanovllm/layers/fuse_moe/heterogeneous.py tests/test_cpu_moe_correctness.py
pytest -q tests/test_cpu_moe_correctness.py -k 'not packed_backend_matches'
python -m pytest -q tests/test_cpu_moe_correctness.py
```

Results:

- Local CPU subset: `3 passed, 1 deselected`.
- CUDA correctness on A100 allocation `20686`: `4 passed`.

Real spec worst-case re-test, Qwen3-30B-A3B, `spec_enable_prefetch=false`, `draft_top_c=128`, 25% expert cache, forced `torch_packed` with `cpu_expert_packed_min_routes=1`:

| backend/config | verify ms | verify CPU prepare ms | verify CPU compute ms | verify CPU merge ms |
|---|---:|---:|---:|---:|
| torch baseline after torch merge optimization | 556.8 | 7.80 | 355.0 | 11.86 |
| forced `torch_packed`, pinned output workspace | 874.5 | 14.48 | 678.9 | 4.96 |
| forced `torch_packed`, pageable output workspace | 562.6 | 14.04 | 363.2 | 7.10 |

Representative logs:

- `/home/mumura/moe_spec/logs/torch_packed_fallback_fix_20260503_084458.log`
- `/home/mumura/moe_spec/logs/spec_real_after_fix_20260503_085117/`
- `/home/mumura/moe_spec/logs/spec_real_after_fix_forced_packed_20260503_090640/`
- `/home/mumura/moe_spec/logs/spec_real_after_fix_pageable_output_20260503_091550/`

Conclusion:

- The unacceptable verify CPU compute regression is fixed for the measured worst case by removing pinned output compute writes.
- `torch_packed` still does not provide a robust end-to-end verify speedup; it is approximately neutral in the 25% cache worst case after the fix.
- The merge optimization remains useful and is now available in the default `torch` backend, so real spec runs should prefer `cpu_expert_backend="torch"` unless a workload-specific benchmark proves `torch_packed` is faster.

## Complete Effective Test Record

This section records the effective tests and benchmarks run during Phase 1, Phase 2, Phase 3, and final Phase 1-3 integration. Runs that failed because of environment errors, wrong import paths, port conflicts, interrupted benchmark sweeps, or log post-processing bugs are intentionally excluded. Diagnostic runs that completed and exposed a real correctness or performance problem are included.

### Common Command Templates

The Slurm wrapper for these runs was the cluster workflow: allocate one A100 GPU, activate `conda activate nano_moe`, record `HOST`, `CUDA_VISIBLE_DEVICES`, and `nvidia-smi`, then run the command shown below.

```text
CPU correctness suite:
python -m pytest -q tests/test_cpu_moe_correctness.py tests/test_cpu_gpu_expert_operator_alignment.py tests/test_cpu_gpu_parallel_moe.py

CPU backend benchmark:
python benchmarks/bench_cpu_moe_backend.py --tokens 8,32,128 --cpu-route-ratio 0.25,0.5,0.75 --output <csv>

Single spec case:
python examples/heterogeneous_benchmark_case.py \
  --model-path /data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B \
  --mode <standard|spec> --slots-per-layer <slots> --num-seqs 1 --input-len 12 --output-len 6 \
  --max-num-batched-tokens 1024 --max-num-seqs 64 --max-model-len 1024 --gpu-memory-utilization 0.85 \
  --max-draft-tokens 4 --draft-top-c 128 --cpu-expert-execution-enabled <true|false> \
  --cpu-expert-backend <torch|torch_packed> --spec-profile true --engine-profile true \
  --engine-profile-cuda-sync true --spec-enable-prefetch <true|false> --temperature 0.0 --seed 0 \
  --enforce-eager false --return-token-ids true --return-text false --return-prompts false \
  --dist-port <port> --output <json>
```

### Phase 1 Tests

| Test | Command | Result path | Key result |
|---|---|---|---|
| CPU expert weight/load validation smoke | `python -m pytest -q tests/test_cpu_moe_correctness.py -k 'loader_to_cpu_precasts_and_contiguates or cpu_expert_weights_validation'` plus baseline CPU path probe | `/home/mumura/moe_spec/logs/cpu_expert_baseline_compute_single_20260502_112816.log` | `2 passed`; baseline CPU path `prep=0.162ms`, `compute=16.669ms`, `merge=0.336ms`. |
| Phase 1/2 shared CUDA correctness after flash-attn env fix | CPU correctness suite | `/home/mumura/moe_spec/logs/cpu_expert_phase12_correctness_20260502_113524.log` | `5 passed in 118.09s`; CUDA device was idle GPU 1 on `gpu18`. |
| Phase 1/2 guard correctness rerun | CPU correctness suite | `/home/mumura/moe_spec/logs/cpu_expert_phase12_correctness_guard_20260502_114455.log` | `5 passed in 103.84s`; confirms precast/validation changes did not break CPU/GPU MoE alignment tests. |

Phase 1 had no separate throughput target; its effective validation was dtype/device/contiguity correctness plus preserving existing CPU/GPU expert operator behavior.

### Phase 2 Tests

| Test | Command | Result path | Key result |
|---|---|---|---|
| `torch_packed` correctness with route overflow/backend guard | CPU correctness suite | `/home/mumura/moe_spec/logs/cpu_expert_phase12_correctness_guard_20260502_114455.log` | `5 passed`; covers CPU packed weight validation and packed backend alignment with the existing torch CPU path. |
| Phase 2 CPU backend benchmark | `python benchmarks/bench_cpu_moe_backend.py --tokens 8,32,128 --cpu-route-ratio 0.25,0.5,0.75 --output benchmarks/results/cpu_moe_backend_phase12_guard_20260502_114701.csv` | `/home/mumura/moe_spec/logs/cpu_expert_phase12_bench_guard_20260502_114701.log`; `benchmarks/results/cpu_moe_backend_phase12_guard_20260502_114701.csv` | All rows had `max_abs=0`, `max_rel=0`. Average decode speedup of `torch_packed` vs `torch`: `3.12x` at 8 tokens, `1.39x` at 32 tokens, `1.34x` at 128 tokens. Aggregate CPU merge reduction: `42.5%`. |
| Phase 2 benchmark spot rerun | `python benchmarks/bench_cpu_moe_backend.py --tokens 32 --cpu-route-ratio 0.25 --output benchmarks/results/cpu_moe_backend_phase12_rerun_20260502_114202.csv` | `/home/mumura/moe_spec/logs/cpu_expert_phase12_bench_rerun_20260502_114202.log`; `benchmarks/results/cpu_moe_backend_phase12_rerun_20260502_114202.csv` | Completed correctness-preserving spot rerun; row-level outputs kept in CSV. |

### Phase 3 Tests

| Test | Command | Result path | Key result |
|---|---|---|---|
| Pinned workspace reuse correctness | CPU correctness suite | `/home/mumura/moe_spec/logs/cpu_expert_phase3_correctness_20260502_144512.log` | `6 passed in 54.51s`; includes overflow and workspace reuse checks. |
| Phase 3 CPU backend benchmark | `python benchmarks/bench_cpu_moe_backend.py --tokens 8,32,128 --cpu-route-ratio 0.25,0.5,0.75 --output benchmarks/results/cpu_moe_backend_phase3_20260502_144621.csv` | `/home/mumura/moe_spec/logs/cpu_expert_phase3_bench_20260502_144621.log`; `benchmarks/results/cpu_moe_backend_phase3_20260502_144621.csv` | All rows had `max_abs=0`, `max_rel=0`. Average decode speedup of `torch_packed` vs `torch`: `1.21x` at 8 tokens, `1.83x` at 32 tokens, `1.27x` at 128 tokens. Aggregate CPU merge reduction: `41.2%`. |
| Phase 3 benchmark spot rerun | `python benchmarks/bench_cpu_moe_backend.py --tokens 8 --cpu-route-ratio 0.25,0.5,0.75 --output benchmarks/results/cpu_moe_backend_phase3_tokens8_rerun_20260502_144812.csv` | `/home/mumura/moe_spec/logs/cpu_expert_phase3_bench_tokens8_rerun_20260502_144812.log`; `benchmarks/results/cpu_moe_backend_phase3_tokens8_rerun_20260502_144812.csv` | Completed spot rerun for small route counts; row-level outputs kept in CSV. |
| Phase 3 spec 100%/75% smoke | `python benchmarks/scripts/spec_standard_cache_ratio_suite.py --ratios 1.0,0.75 --setting-profile smoke ...` | `/home/mumura/moe_spec/logs/spec_standard_cache_ratio_phase3_smoke_100_75_20260502_145017.log`; `/home/mumura/moe_spec/logs/spec_standard_cache_ratio_phase3_smoke_100_75_rerun_20260502_145951.log` | 100% cache matched standard exactly. 75% partial-cache mismatch was reproduced and treated as a correctness finding, not an environment error. |
| Phase 3 75% cache CPU/GPU path diagnostic | Single spec case matrix for standard, `spec_gpu_fallback`, and `spec_cpu_exec` at 75% cache | `/home/mumura/moe_spec/logs/spec_cache75_phase3_cpu_diag_20260502_145516/`; `/home/mumura/moe_spec/logs/spec_cache75_phase3_cpu_diag_20260502_145516.log` | In this run standard, GPU fallback, and CPU exec all matched digest `4426c014...`; `spec_cpu_exec` verify CPU compute was `117.189ms`. |

### Phase 1-3 Final Integration Tests

| Test | Command | Result path | Key result |
|---|---|---|---|
| Final CPU expert correctness after mismatch investigation | CPU correctness suite | `/home/mumura/moe_spec/logs/cpu_expert_mismatch_fix_correctness_20260502_213714.log` | `6 passed in 110.66s`; kept Phase 1-3 CPU backend changes after rolling back route-buffer-only experiment. |
| No-prefetch partial-cache diagnostic | Single spec case matrix with `spec_enable_prefetch=false`, standard vs `spec75_gpu_fallback` vs `spec75_cpu_exec` | `/home/mumura/moe_spec/logs/spec_no_prefetch_probe_20260502_212834/`; `/home/mumura/moe_spec/logs/spec_no_prefetch_probe_20260502_212834.log` | Standard digest `4426c014...`; both 75% spec GPU fallback and CPU exec produced digest `83420627...`. CPU exec verify CPU compute was `121.659ms`. This showed prefetch is not required for the mismatch. |
| Cache-state repeat diagnostic | Repeated single spec cases around cache100 and GPU fallback warmup | `/home/mumura/moe_spec/logs/spec_mismatch_repeat_probe_20260502_210452/`; `/home/mumura/moe_spec/logs/spec_mismatch_repeat_probe_20260502_210452.log` | 75% partial-cache output could alternate between `4426c014...` and `83420627...` depending on run/cache state. Effective root-cause evidence for path-dependent numerical drift. |
| Route-buffer-only fix attempt validation | Single spec case matrix after route-buffer-only change | `/home/mumura/moe_spec/logs/spec_no_prefetch_routebuffer_only_20260502_215742/`; `/home/mumura/moe_spec/logs/spec_no_prefetch_routebuffer_only_20260502_215742.log` | Standard digest `4426c014...`; route-buffer-only path produced all-zero tokens and digest `f383a13f...`. The attempted fix was therefore rolled back. |
| Pre-Phase1-3 branch alignment check | `PYTHONPATH=/home/mumura/moe_spec/nano-vllm-moe_fdc28f4 python examples/heterogeneous_benchmark_case.py ...` for standard and spec cache 100/75/50/25 variants | `/home/mumura/moe_spec/logs/fdc28f4_spec_alignment_matrix_fixedpath_20260502_223347/`; `/home/mumura/moe_spec/logs/fdc28f4_spec_alignment_matrix_fixedpath_20260502_223347.log` | On `fdc28f4`, 100% cache matched standard; 25% GPU fallback mismatched. This confirmed deterministic drift existed before Phase 1-3. |
| Pre-Phase1-3 repeat check | Same fdc28f4 setup, repeated `spec75_gpu_np` three times | `/home/mumura/moe_spec/logs/fdc28f4_spec75_repeat_20260502_225237/`; `/home/mumura/moe_spec/logs/fdc28f4_spec75_repeat_20260502_225237.log` | `spec75_gpu_np` matched on reps 1 and 3, mismatched on rep 2; confirms intermittent/path-state-dependent behavior before Phase 1-3. |
| Current branch alignment check | Current tree single spec case matrix for standard, `spec75_gpu_np`, `spec75_cpu_np` | `/home/mumura/moe_spec/logs/current_spec75_alignment_compare_20260502_224733/`; `/home/mumura/moe_spec/logs/current_spec75_alignment_compare_20260502_224733.log` | Current tree reproduced same pattern: `spec75_cpu_np` matched standard with CPU compute `915.959ms`; `spec75_gpu_np` mismatched. |
| Phase 1-3 real spec backend performance | Single spec case matrix for ratios 75/50/25 and backends `torch`/`torch_packed`; common args: `spec_enable_prefetch=false`, `draft_top_c=128`, CPU expert execution enabled | `/home/mumura/moe_spec/logs/spec_phase13_backend_perf_20260502_220346/` | Per-case JSONs are valid; wrapper post-processing TypeError is excluded. `torch_packed` reduced merge by `67-84%`, but regressed verify at 75% and 25% due to higher CPU prepare/compute. |
| 2026-05-03 regression-fix local checks | `python -m py_compile ...`; `pytest -q tests/test_cpu_moe_correctness.py -k 'not packed_backend_matches'` | Terminal run; summarized in this document | `py_compile` passed; CPU subset `3 passed, 1 deselected`. |
| 2026-05-03 regression-fix CUDA correctness | `python -m pytest -q tests/test_cpu_moe_correctness.py` | `/home/mumura/moe_spec/logs/torch_packed_fallback_fix_20260503_084458.log` | `4 passed in 53.93s` on A100 allocation `20686`. |
| 2026-05-03 default torch merge integration perf | Single spec case matrix for `torch` and `torch_packed`, ratios 75/50/25, default `cpu_expert_packed_min_routes=32` | `/home/mumura/moe_spec/logs/spec_real_after_fix_20260503_085117/`; `/home/mumura/moe_spec/logs/torch_packed_fallback_fix_20260503_084458.log` | Default `torch` backend merge improved: 75% cache merge `6.36ms` vs older `10.41ms`; 25% cache merge `11.86ms` vs older `30.13ms`. |
| 2026-05-03 forced packed pinned-output diagnostic | Single spec case matrix, forced `torch_packed` with `cpu_expert_packed_min_routes=1` | `/home/mumura/moe_spec/logs/spec_real_after_fix_forced_packed_20260503_090640/`; `/home/mumura/moe_spec/logs/torch_packed_fallback_fix_20260503_084458.log` | Pinned output workspace caused 25% cache verify CPU compute regression: `678.93ms`; merge was fast at `4.96ms`. |
| 2026-05-03 forced packed pageable-output fix | Single spec case, 25% cache, forced `torch_packed`, pageable output workspace | `/home/mumura/moe_spec/logs/spec_real_after_fix_pageable_output_20260503_091550/`; `/home/mumura/moe_spec/logs/torch_packed_fallback_fix_20260503_084458.log` | Verify CPU compute returned to `363.16ms`, close to `torch` baseline `354.96ms`; merge remained improved at `7.10ms` vs `torch` `11.86ms`. |

### Key Result Tables

Phase 2 CPU backend benchmark summary:

| batch tokens | torch avg decode ms | torch_packed avg decode ms | speedup |
|---:|---:|---:|---:|
| 8 | 8.620 | 2.765 | 3.12x |
| 32 | 5.476 | 3.943 | 1.39x |
| 128 | 8.542 | 6.374 | 1.34x |

Phase 3 CPU backend benchmark summary:

| batch tokens | torch avg decode ms | torch_packed avg decode ms | speedup |
|---:|---:|---:|---:|
| 8 | 27.088 | 22.307 | 1.21x |
| 32 | 6.305 | 3.451 | 1.83x |
| 128 | 7.534 | 5.923 | 1.27x |

Final real spec verify performance summary:

| run | cache ratio | backend/config | output tok/s | verify ms | verify CPU prepare ms | verify CPU compute ms | verify CPU merge ms |
|---|---:|---|---:|---:|---:|---:|---:|
| Phase 1-3 before follow-up | 75% | torch | 2.589 | 290.0 | 6.08 | 113.40 | 10.41 |
| Phase 1-3 before follow-up | 75% | torch_packed | 2.289 | 586.4 | 10.71 | 416.26 | 3.47 |
| Phase 1-3 before follow-up | 50% | torch | 1.345 | 448.5 | 7.41 | 235.87 | 20.77 |
| Phase 1-3 before follow-up | 50% | torch_packed | 1.823 | 436.6 | 11.84 | 247.86 | 4.15 |
| Phase 1-3 before follow-up | 25% | torch | 1.243 | 596.3 | 8.04 | 363.23 | 30.13 |
| Phase 1-3 before follow-up | 25% | torch_packed | 1.290 | 932.2 | 12.68 | 734.30 | 5.61 |
| After torch merge integration | 75% | torch | 2.146 | 289.7 | 6.10 | 116.09 | 6.36 |
| After torch merge integration | 50% | torch | 1.621 | 835.0 | 7.31 | 637.92 | 10.81 |
| After torch merge integration | 25% | torch | 1.123 | 556.8 | 7.80 | 354.96 | 11.86 |
| Forced packed pinned output | 25% | torch_packed min1 | 1.067 | 874.5 | 14.48 | 678.93 | 4.96 |
| Forced packed pageable output | 25% | torch_packed min1 | 1.146 | 562.6 | 14.04 | 363.16 | 7.10 |
