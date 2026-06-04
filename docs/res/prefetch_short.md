# Predictive Prefetch Validation Report

Date: 2026-06-02

## 1. Experiment Design

### 1.1 Objectives

1. Validate `prefetch_runtime_kind=predictive` against legacy segment-indexed prefetch.
2. Confirm generated text quality under the meaningful reroute prompt.
3. Compare acceptance rate, cache hit rate, throughput, draft/verify forward time, graph replays, and prefetch counters.
4. Report M3-style `perfect_fraction` and `step0_perfect_fraction`.

### 1.2 Test Matrix

| Dimension | Values |
|---|---|
| Runtime kinds | `legacy, predictive` |
| Cache ratios | `0.25, 0.5, 0.75, 0.25, 0.5, 0.75, 0.25, 0.5, 0.75, 0.25, 0.5, 0.75` |
| Output lengths | `128, 512` |
| max_draft_tokens | ratio 0.25 -> 2, ratio 0.50 -> 6, ratio 0.75 -> 8 |
| Reroute policy | `entropy_cache_bias` |
| Prefetch mode | `draft_segment_indexed` |
| Expert cache strategy | `lru` seeded by offline profile |
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

### 1.4 Run Status

- status: `completed`
- slurm_job_id: ``
- eta: ``
- output_dir: `/home/mumura/moe_spec/nano-vllm-moe/results/predictive_prefetch_validation_direct_20260602_003130`

---

## 2. Code Review And Fixes

Implemented fixes before benchmarking:

1. Same-round async draft metadata remains valid after `end_draft_iteration` until the next draft round begins.
2. Predictive Phase 2 maps segment-0 frontier prefetches to the last segment, matching Part B.
3. Failed predictive reservations roll back temporary round-protection entries.
4. Benchmark CLIs now expose predictive runtime knobs and preserve them in result metadata.
5. Benchmark summaries include predictive prefetch counters and M3 perfect-fraction metrics.

## 3. Results

| kind | out | ratio | K | accept | route hit | weight hit | output tok/s | draft ms | verify ms | graph | prefetch submit/done/used | phase1 | verify-layer | perfect | step0 perfect | text |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|---:|---:|---:|---:|:---|
| legacy | 128 | 0.25 | 2 | 0.8404 | 0.9887 | 0.9892 | 5.401 | 21.505 | 348.438 | 94 | 1747/1749/8299 | 0 | 1254/5991 | 0.0000 | 0.0000 | ok |
| legacy | 128 | 0.50 | 6 | 0.8548 | 0.9853 | 0.9864 | 9.192 | 32.678 | 307.889 | 124 | 1059/1063/3336 | 0 | 709/2547 | 0.0323 | 0.0000 | ok |
| legacy | 128 | 0.75 | 8 | 1.0000 | 0.9899 | 0.9907 | 16.128 | 23.937 | 256.029 | 112 | 466/470/1641 | 0 | 298/1120 | 0.1607 | 0.0000 | ok |
| legacy | 512 | 0.25 | 2 | 0.8238 | 0.9971 | 0.9973 | 6.957 | 24.320 | 312.179 | 386 | 4081/4083/23146 | 0 | 1262/6263 | 0.0000 | 0.0000 | ok |
| legacy | 512 | 0.50 | 6 | 0.8921 | 0.9961 | 0.9964 | 18.573 | 21.766 | 178.790 | 482 | 2676/2680/7823 | 0 | 710/2396 | 0.0477 | 0.0000 | ok |
| legacy | 512 | 0.75 | 8 | 0.9848 | 0.9973 | 0.9975 | 26.541 | 21.124 | 143.035 | 460 | 765/769/2034 | 0 | 301/1096 | 0.4413 | 0.0000 | ok |
| predictive | 128 | 0.25 | 2 | 0.7525 | 0.9891 | 0.9895 | 7.753 | 21.565 | 224.250 | 101 | 2625/2625/11994 | 204 | 1719/8556 | 0.0000 | 0.0000 | ok |
| predictive | 128 | 0.50 | 6 | 0.8760 | 0.9849 | 0.9860 | 16.239 | 19.777 | 164.153 | 121 | 796/796/2790 | 84 | 254/1158 | 0.0331 | 0.0000 | ok |
| predictive | 128 | 0.75 | 8 | 0.9412 | 0.9890 | 0.9898 | 13.654 | 33.937 | 255.411 | 119 | 510/510/1599 | 60 | 128/516 | 0.1597 | 0.0000 | ok |
| predictive | 512 | 0.25 | 2 | 0.8377 | 0.9971 | 0.9972 | 7.301 | 25.381 | 295.723 | 382 | 11644/11644/55296 | 764 | 8386/41191 | 0.0000 | 0.0000 | ok |
| predictive | 512 | 0.50 | 6 | 0.9435 | 0.9957 | 0.9961 | 21.051 | 19.655 | 172.931 | 460 | 2594/2594/6145 | 308 | 680/2428 | 0.0217 | 0.0000 | ok |
| predictive | 512 | 0.75 | 8 | 0.9396 | 0.9971 | 0.9973 | 19.336 | 25.548 | 208.381 | 480 | 911/911/1907 | 240 | 219/700 | 0.4667 | 0.0000 | ok |

## 4. Predictive Delta Versus Legacy

Positive throughput, acceptance, cache hit, and perfect-fraction deltas are improvements. Negative draft/verify ms deltas are improvements.

| out | ratio | accept delta | route-hit delta | tok/s delta | draft ms delta | verify ms delta | perfect delta | step0 perfect delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.25 | -0.0880 | +0.0004 | +2.352 | +0.060 | -124.188 | +0.0000 | +0.0000 |
| 128 | 0.50 | +0.0212 | -0.0004 | +7.047 | -12.901 | -143.736 | +0.0008 | +0.0000 |
| 128 | 0.75 | -0.0588 | -0.0009 | -2.474 | +9.999 | -0.617 | -0.0011 | +0.0000 |
| 512 | 0.25 | +0.0139 | -0.0001 | +0.344 | +1.061 | -16.456 | +0.0000 | +0.0000 |
| 512 | 0.50 | +0.0514 | -0.0003 | +2.478 | -2.110 | -5.859 | -0.0260 | +0.0000 |
| 512 | 0.75 | -0.0452 | -0.0002 | -7.205 | +4.424 | +65.346 | +0.0254 | +0.0000 |

## 5. Text Quality

All completed cases passed the automated text-quality guard.

## 6. Conclusion

The benchmark completed; use the tables above for the legacy-vs-predictive comparison.
