# Predictive Prefetch Validation Report

Date: 2026-06-01

## 1. Experiment Design

### 1.1 Objectives

1. Validate `prefetch_runtime_kind=predictive` against legacy segment-indexed prefetch.
2. Confirm generated text quality under the meaningful reroute prompt.
3. Compare acceptance rate, cache hit rate, throughput, draft/verify forward time, graph replays, and prefetch counters.
4. Report M3-style `perfect_fraction` and `step0_perfect_fraction`.

### 1.2 Test Matrix

| Dimension | Values |
|---|---|
| Runtime kinds | `legacy`, `predictive` |
| Cache ratios | 0.25, 0.50, 0.75 |
| Output lengths | 128, 512 tokens |
| max_draft_tokens | ratio 0.25 -> 2, ratio 0.50 -> 6, ratio 0.75 -> 8 |
| Reroute policy | `entropy_cache_bias` |
| Draft settings | `draft_top_c=0`, draft CUDA graph enabled |
| Prefetch settings | `spec_enable_prefetch=true`, `prefetch_runtime_mode=draft_segment_indexed` |
| Expert cache strategy | `lru`, seeded by offline profile |
| Verify miss policy | `cache_fill` |

### 1.3 Common Settings

```
Model:        /data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B
Profile:      results/reroute_impl_20260531/offline_profile_20260531_203257.safetensors
num_seqs:     1
max_model_len:2048
draft_top_c:  0
draft graph:  enabled
CPU backend:  fused
workspace:    cpu_expert_workspace_max_routes=16384
```

Metric definitions:

- `perfect_fraction`: fraction of draft forwards where every recorded MoE layer had zero CPU experts.
- `step0_perfect_fraction`: same metric restricted to the first draft forward of each speculative step.
- `prefetch submit/done/used`: benchmark `submit_count/completed_count/consumed_count`.

### 1.4 Script And Output

- Matrix runner: `scripts/predictive_prefetch_validation.py`
- Batch wrapper: `scripts/run_predictive_prefetch_matrix.sbatch`
- Per-case backend: `benchmarks/scripts/spec_verify_expert_count_stats.py --single-case`
- Full matrix result directory:
  `results/predictive_prefetch_validation_20260601_222120/`
- Batch log:
  `/home/mumura/moe_spec/logs/predictive_prefetch_matrix_20260601_222120_28464.log`
- Slurm job:
  `28464`

### 1.5 Current Run Status

The full matrix job has been submitted but is pending due to Slurm priority.

```
Job:              28464
State:            PENDING
Reason:           Priority
Requested time:   02:00:00
Scheduler start:  2026-06-02 23:00:46 CST
Expected finish:  approximately 2026-06-03 00:30 CST if it starts on schedule
Walltime end:     2026-06-03 01:00:46 CST
```

The batch job will overwrite/update this document with final result tables when the matrix completes.

---

## 2. Code Review And Fixes

Implemented before launching the matrix:

1. Same-round async draft metadata remains valid after `end_draft_iteration` until the next draft round begins.
2. Predictive Phase 2 maps segment-0 frontier prefetches to the last segment, matching Part B.
3. Failed predictive reservations roll back temporary round-protection entries.
4. Benchmark CLIs now expose predictive runtime knobs and preserve them in result metadata.
5. Benchmark summaries include predictive prefetch counters.
6. M3 perfect-fraction counters are collected from draft runtime metadata, so they work with draft CUDA graph replay.

## 3. Validation Commands

Targeted tests:

```bash
source /opt/Software/Anaconda3/etc/profile.d/conda.sh
conda activate nano_moe
python -m pytest -q \
  tests/test_predictive_prefetch.py \
  tests/test_predictive_prefetch_cli_args.py \
  tests/test_spec_verify_expert_count_stats.py \
  tests/test_config_prefetch.py \
  tests/test_config_predictive_prefetch.py
```

Result:

```
52 passed
```

Broader related suite:

```bash
python -m pytest -q \
  tests/test_prefetch_runtime.py \
  tests/test_prefetch_wait.py \
  tests/test_prefetch_global_queue.py \
  tests/test_prefetch_strategy.py \
  tests/test_prefetch_runtime_meta.py \
  tests/test_model_runner_prefetch.py \
  tests/test_verify_prefetch_comprehensive.py \
  tests/test_verify_prefetch_integration.py \
  tests/test_expert_cache_staging.py \
  tests/test_expert_cache_generation.py \
  tests/test_spec_engine_prefetch.py \
  tests/test_spec_engine_basic.py \
  tests/test_spec_engine_flow.py \
  tests/test_block_manager_draft.py \
  tests/test_placement_spec.py \
  tests/test_draft_standard_decode_forward_bench.py \
  tests/test_spec_verify_expert_count_stats.py \
  tests/test_mode_config.py \
  tests/test_config_prefetch.py \
  tests/test_llm_engine_mode_dispatch.py \
  tests/test_model_runner_spec_modes.py \
  tests/test_model_runner_cache_strategy.py \
  tests/test_predictive_prefetch.py \
  tests/test_predictive_prefetch_cli_args.py \
  tests/test_config_predictive_prefetch.py
```

Result:

```
149 passed, 7 skipped
```

Syntax checks:

```bash
python -m py_compile \
  nanovllm/expert/prefetcher.py \
  examples/heterogeneous_benchmark_case.py \
  examples/benchmarks/draft_standard_decode_forward_bench.py \
  benchmarks/scripts/spec_verify_expert_count_stats.py \
  scripts/predictive_prefetch_validation.py
```

Result: OK.

## 4. Lightweight Functional Test

Lightweight predictive smoke:

```
Result dir: results/predictive_prefetch_light_m3_20260601_221749/
Log:        /home/mumura/moe_spec/logs/predictive_prefetch_light_m3_20260601_221749.log
```

Observed summary:

| kind | out | ratio | K | accept | route hit | output tok/s | draft ms | verify ms | graph | perfect | step0 perfect | text |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| predictive | 8 | 0.50 | 2 | 0.8000 | 0.9169 | 2.496 | 21.669 | 191.565 | 5 | 0.0000 | 0.0000 | ok |

M3 counters were present in the JSON:

```
group_count=5
step0_group_count=1
source=runtime_metadata
```

## 5. Results

Full matrix results are pending Slurm execution.

## 6. Conclusion

The code fixes, unit tests, and lightweight predictive smoke are complete. The full matrix is queued as Slurm job `28464`; this document will be updated automatically by the batch job after completion.
