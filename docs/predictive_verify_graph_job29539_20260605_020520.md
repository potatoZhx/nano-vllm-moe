# Predictive Prefetch Validation Report

Date: 2026-06-05

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
| Cache ratios | `0.75, 0.75` |
| Output lengths | `512` |
| max_draft_tokens | ratio 0.25 -> 2, ratio 0.50 -> 6, ratio 0.75 -> 10 |
| Reroute policy | `entropy_cache_bias` |
| Prefetch mode | `draft_segment_indexed` |
| Expert cache strategy | `lru` seeded by offline profile |
| Verify miss policy | `cache_fill_no_cpu` |
| Verify CUDA graph | `True` |
| Verify graph buckets | `4, 8, 12, 16` |

### 1.3 Common Settings

```
Model:        /data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B
Profile:      results/reroute_impl_20260531/offline_profile_20260531_203257.safetensors
num_seqs:     1
max_model_len:2048
draft_top_c:  0
draft graph:  enabled
verify graph: True
CPU backend:  fused
workspace:    cpu_expert_workspace_max_routes=32768
```

Metric definitions:

- `true hit`: cache hit rate measured BEFORE `cache_fill` transfers miss experts into cache slots.
- `post-xfer hit`: cache hit rate measured AFTER transfers (legacy metric, artificially high).
- `miss/layer`: average number of miss routes per MoE layer forward (pre-transfer).
- `active/layer`: average number of active routes per MoE layer forward (= tokens * top_k).
- `perfect_fraction`: fraction of draft forwards where every recorded MoE layer had zero CPU experts.
- `step0_perfect_fraction`: same metric restricted to the first draft forward of each speculative step.
- `prefetch submit/done/used`: benchmark `submit_count/completed_count/consumed_count`.

### 1.4 Run Status

- status: `completed`
- slurm_job_id: `29539`
- eta: ``
- output_dir: `/home/mumura/moe_spec/nano-vllm-moe/results/predictive_verify_graph_job29539_20260605_020520`

---

## 2. Code Review And Fixes

Implemented fixes before benchmarking:

1. Same-round async draft metadata remains valid after `end_draft_iteration` until the next draft round begins.
2. Predictive Phase 2 maps segment-0 frontier prefetches to the last segment, matching Part B.
3. Failed predictive reservations roll back temporary round-protection entries.
4. Benchmark CLIs now expose predictive runtime knobs and preserve them in result metadata.
5. Benchmark summaries include predictive prefetch counters and M3 perfect-fraction metrics.

## 3. Results

| kind | out | ratio | K | accept | true hit | post-xfer hit | miss/layer | active/layer | weight hit | output tok/s | draft ms | verify ms | verify graph | verify replay/fallback | draft graph | prefetch submit/done/used | phase1 | verify-layer | perfect | step0 perfect | text |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---|---:|:---|---:|---:|---:|---:|:---|
| legacy | 512 | 0.75 | 10 | 0.8982 | 0.8344 | 0.8344 | 0.00 | 0.00 | 0.8467 | 28.014 | 23.386 | 93.785 | on | 52/2496/0 | 511 | 1012/1012/0 | 0 | 247/0 | 0.4501 | 0.0000 | ok |
| predictive | 512 | 0.75 | 10 | 0.8878 | 0.8190 | 0.8190 | 0.00 | 0.00 | 0.8323 | 32.659 | 19.768 | 81.648 | on | 52/2496/0 | 517 | 994/994/0 | 208 | 58/0 | 0.2147 | 0.0000 | ok |

## 4. Predictive Delta Versus Legacy

Positive throughput, acceptance, cache hit, and perfect-fraction deltas are improvements. Negative draft/verify ms deltas are improvements.

| out | ratio | accept delta | true-hit delta | miss/layer delta | tok/s delta | draft ms delta | verify ms delta | perfect delta | step0 perfect delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.75 | -0.0104 | -0.0153 | +0.00 | +4.645 | -3.618 | -12.137 | -0.2354 | +0.0000 |

## 5. Text Quality

All completed cases passed the automated text-quality guard.

## 6. Conclusion

The benchmark completed; use the tables above for the legacy-vs-predictive comparison.
