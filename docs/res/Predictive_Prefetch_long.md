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
| Cache ratios | `0.25, 0.5, 0.75, 0.25, 0.5, 0.75, 0.25, 0.5, 0.75, 0.25, 0.5, 0.75, 0.25, 0.5, 0.75, 0.25, 0.5, 0.75` |
| Output lengths | `1024, 4096, 8192` |
| max_draft_tokens | ratio 0.25 -> 2, ratio 0.50 -> 6, ratio 0.75 -> 8 |
| Reroute policy | `entropy_cache_bias` |
| Prefetch mode | `draft_segment_indexed` |
| Expert cache strategy | `lru` seeded by offline profile |
| Verify miss policy | `cache_fill` |

### 1.3 Common Settings

```
Model:        /zx_data1/models/Qwen--Qwen3-30B-A3B-Base
Profile:      results/reroute_impl_20260531/offline_profile_20260531_203257.safetensors
num_seqs:     1
max_model_len:9216
draft_top_c:  0
draft graph:  enabled
CPU backend:  fused
workspace:    cpu_expert_workspace_max_routes=1638400
```

Metric definitions:

- `perfect_fraction`: fraction of draft forwards where every recorded MoE layer had zero CPU experts.
- `step0_perfect_fraction`: same metric restricted to the first draft forward of each speculative step.
- `prefetch submit/done/used`: benchmark `submit_count/completed_count/consumed_count`.

### 1.4 Run Status

- status: `completed`
- slurm_job_id: ``
- eta: ``
- output_dir: `/zx_data1/sparsity/nano-vllm-moe/results/predictive_prefetch_validation_direct_20260602_030904`

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
| legacy | 1024 | 0.25 | 2 | 0.8554 | 0.9984 | 0.9985 | 7.871 | 21.503 | 291.223 | 754 | 4506/4506/24795 | 0 | 1157/6105 | 0.0000 | 0.0000 | repeated_12gram |
| legacy | 1024 | 0.50 | 6 | 0.9134 | 0.9978 | 0.9980 | 18.369 | 22.231 | 205.772 | 947 | 4333/4333/14086 | 0 | 589/2183 | 0.0084 | 0.0000 | ok |
| legacy | 1024 | 0.75 | 8 | 0.9034 | 0.9987 | 0.9988 | 20.613 | 24.661 | 188.903 | 994 | 1723/1723/4304 | 0 | 282/1182 | 0.4135 | 0.0000 | ok |
| legacy | 4096 | 0.25 | 2 | 0.8111 | 0.9996 | 0.9996 | 8.590 | 22.340 | 256.323 | 3123 | 14897/14897/88087 | 0 | 1156/5649 | 0.0000 | 0.0000 | repeated_12gram |
| legacy | 4096 | 0.50 | 6 | 0.9331 | 0.9994 | 0.9995 | 19.841 | 21.956 | 194.479 | 3723 | 15096/15096/48772 | 0 | 598/2184 | 0.0099 | 0.0000 | ok |
| legacy | 4096 | 0.75 | 8 | 0.9417 | 0.9996 | 0.9997 | 22.531 | 23.540 | 183.486 | 3839 | 7045/7045/18817 | 0 | 289/1214 | 0.3313 | 0.0000 | repeated_12gram |
| legacy | 8192 | 0.25 | 2 | 0.8995 | 0.9998 | 0.9998 | 9.241 | 21.541 | 255.804 | 5852 | 27379/27379/160956 | 0 | 1167/5778 | 0.0002 | 0.0000 | repeated_12gram |
| legacy | 8192 | 0.50 | 6 | 0.9075 | 0.9997 | 0.9997 | 18.971 | 22.066 | 201.859 | 7625 | 30629/30629/102570 | 0 | 603/2055 | 0.0079 | 0.0000 | repeated_12gram |
| legacy | 8192 | 0.75 | 8 | 0.9682 | 0.9998 | 0.9998 | 21.616 | 25.020 | 197.473 | 7492 | 16621/16621/47846 | 0 | 284/1144 | 0.2700 | 0.0000 | repeated_12gram |
| predictive | 1024 | 0.25 | 2 | 0.8532 | 0.9984 | 0.9984 | 8.385 | 23.636 | 266.835 | 756 | 19956/19956/93581 | 1512 | 16662/81125 | 0.0000 | 0.0000 | ok |
| predictive | 1024 | 0.50 | 6 | 0.9085 | 0.9978 | 0.9979 | 19.061 | 22.138 | 192.955 | 951 | 4955/4955/12441 | 636 | 2221/7184 | 0.0105 | 0.0000 | ok |
| predictive | 1024 | 0.75 | 8 | 0.9318 | 0.9984 | 0.9985 | 21.026 | 24.435 | 194.382 | 968 | 1543/1543/2515 | 484 | 436/1199 | 0.4380 | 0.0000 | ok |
| predictive | 4096 | 0.25 | 2 | 0.7912 | 0.9996 | 0.9996 | 8.043 | 22.857 | 270.918 | 3171 | 82145/82145/391559 | 6344 | 67941/339253 | 0.0000 | 0.0000 | ok |
| predictive | 4096 | 0.50 | 6 | 0.9154 | 0.9994 | 0.9995 | 17.667 | 23.598 | 219.246 | 3784 | 28615/28615/75437 | 2524 | 14668/43622 | 0.0063 | 0.0000 | repeated_12gram |
| predictive | 4096 | 0.75 | 8 | 0.9421 | 0.9996 | 0.9996 | 21.981 | 24.424 | 185.687 | 3837 | 6519/6519/11080 | 1920 | 1427/3836 | 0.3784 | 0.0000 | repeated_12gram |
| predictive | 8192 | 0.25 | 2 | 0.8462 | 0.9998 | 0.9998 | 8.574 | 22.814 | 264.601 | 6084 | 156177/156177/735967 | 12168 | 129282/630376 | 0.0002 | 0.0000 | repeated_12gram |
| predictive | 8192 | 0.50 | 6 | 0.9422 | 0.9997 | 0.9997 | 19.581 | 23.506 | 193.503 | 7386 | 33463/33463/76441 | 4924 | 13950/41132 | 0.0445 | 0.0000 | ok |
| predictive | 8192 | 0.75 | 8 | 0.9673 | 0.9998 | 0.9998 | 22.150 | 24.521 | 192.079 | 7498 | 17259/17259/36163 | 3752 | 5080/14755 | 0.3081 | 0.0000 | repeated_12gram |

## 4. Predictive Delta Versus Legacy

Positive throughput, acceptance, cache hit, and perfect-fraction deltas are improvements. Negative draft/verify ms deltas are improvements.

| out | ratio | accept delta | route-hit delta | tok/s delta | draft ms delta | verify ms delta | perfect delta | step0 perfect delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 0.25 | -0.0023 | -0.0000 | +0.515 | +2.133 | -24.388 | +0.0000 | +0.0000 |
| 1024 | 0.50 | -0.0049 | -0.0001 | +0.692 | -0.093 | -12.817 | +0.0021 | +0.0000 |
| 1024 | 0.75 | +0.0284 | -0.0002 | +0.413 | -0.226 | +5.479 | +0.0245 | +0.0000 |
| 4096 | 0.25 | -0.0198 | -0.0000 | -0.548 | +0.517 | +14.595 | +0.0000 | +0.0000 |
| 4096 | 0.50 | -0.0177 | -0.0000 | -2.174 | +1.642 | +24.766 | -0.0036 | +0.0000 |
| 4096 | 0.75 | +0.0005 | -0.0000 | -0.550 | +0.885 | +2.200 | +0.0471 | +0.0000 |
| 8192 | 0.25 | -0.0534 | +0.0000 | -0.667 | +1.273 | +8.797 | -0.0000 | +0.0000 |
| 8192 | 0.50 | +0.0346 | -0.0000 | +0.610 | +1.440 | -8.356 | +0.0367 | +0.0000 |
| 8192 | 0.75 | -0.0009 | -0.0000 | +0.534 | -0.499 | -5.393 | +0.0381 | +0.0000 |

## 5. Text Quality

10 cases failed the text-quality guard.

## 6. Conclusion

The benchmark completed; use the tables above for the legacy-vs-predictive comparison.
