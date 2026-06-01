# LFU RankGuard Expert Cache Validation Report

Date: 2026-06-01

## 1. Experiment Design

### 1.1 Objectives

1. Audit the newly integrated expert cache algorithm from
   `docs/impl_lfu_rankguard.md` before running a full benchmark.
2. Test old and new cache strategies under the reroute setup requested for
   section 9/10 style validation in `docs/reroute_full_validation_20260528.md`.
3. Verify generated text quality for every benchmark case: no malformed text,
   no replacement characters, and no obvious repeated-line or repeated-phrase
   failure.
4. Compare acceptance rate, expert-cache route hit rate, and output throughput
   across output length and cache-ratio settings.
5. If the implementation audit exposes a correctness issue or an improvement
   opportunity, fix it and include the improved algorithm in the same matrix.

### 1.2 Test Matrix

| Dimension | Values |
|---|---|
| Cache strategies | `lfu`, `lfu_rankguard`, `lfu_rankguard_online` |
| Cache ratios | 0.25, 0.50, 0.75 |
| Output lengths | 128, 512 tokens |
| Reroute policy | `entropy_cache_bias` |
| Acceptance strategy | `standard_sampling`, `temperature=0.8` |
| Draft settings | `max_draft_tokens=8`, `draft_top_c=0`, draft CUDA graph enabled |
| Prefetch settings | `prefetch_enabled=true`, `prefetch_runtime_mode=draft_segment_indexed` |
| Total valid cases | 18 = 3 strategies x 3 cache ratios x 2 output lengths |

### 1.3 Common Settings

```
Model:        Qwen3-30B-A3B
Weight:       /data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B
Profile:      results/reroute_impl_20260531/offline_profile_20260531_203257.safetensors
num_seqs:     1
max_model_len: 2048
max_draft_tokens: 8
draft_top_c:  0
reroute:      entropy_cache_bias
prefetch:     enabled
runtime mode: draft_segment_indexed
draft graph:  enabled
CPU backend:  fused
workspace:    cpu_expert_workspace_max_routes=16384
```

Metric definitions:

- `accept`: benchmark-reported speculative acceptance rate.
- `cache hit`: route-level expert cache hit rate, computed as
  `1 - model_cpu_route_ratio`.
- `output tok/s`: end-to-end generated output tokens per second.
- `decode tok/s`: decode-phase generated output tokens per second.
- `graph replays`: draft CUDA graph replay count. Non-zero values confirm that
  the draft graph path was active.

### 1.4 Script And Output

- Matrix runner: `scripts/lfu_rankguard_cache_validation.py`
- Per-case backend: `benchmarks/scripts/spec_verify_expert_count_stats.py --single-case`
- Baseline/default RankGuard results:
  `results/lfu_rankguard_cache_20260601_024816/`
- Online RankGuard results:
  `results/lfu_rankguard_online_cache_20260601_141640/`
- Logs:
  - `/home/mumura/moe_spec/logs/lfu_rankguard_cache_20260601_024816.log`
  - `/home/mumura/moe_spec/logs/lfu_rankguard_online_resume_20260601_141640.log`

Each case runs in a separate subprocess because `LLM()` initializes the default
distributed process group. Reusing the same Python process for all matrix cells
would hit the same process-group reinitialization issue documented in
`docs/reroute_full_validation_20260528.md`.

---

## 2. Algorithm Implementation Audit

### 2.1 Source Files Audited And Updated

| File | Role |
|---|---|
| `nanovllm/scheduling/cache_strategy.py` | Cache strategy implementations and factory registration |
| `nanovllm/scheduling/draft_reroute_profile.py` | Offline reroute-profile seed loading |
| `nanovllm/engine/model_runner.py` | Cache strategy construction from `Config` |
| `nanovllm/expert/prefetcher.py` | Verify-routing observation and segment-indexed publish-slot selection |
| `nanovllm/config.py` | Cache strategy config validation |
| `benchmarks/scripts/spec_verify_expert_count_stats.py` | Benchmark flags, cache metrics, generated-text recording |
| `scripts/lfu_rankguard_cache_validation.py` | Matrix runner and text-quality checks |

### 2.2 Original Algorithm: `lfu_rankguard`

`lfu_rankguard` extends LFU eviction with a per-layer expert protection score.
For each layer and cached expert:

```
if slot is empty:
    use the empty slot
elif rank_score(expert) >= threshold:
    skip this expert
else:
    choose the non-protected expert with the smallest LFU access count

if every cached expert is protected:
    fall back to pure LFU
```

The score is:

```
rank_score(j) = 2 * rank1_freq(j) + rank2_freq(j)
```

The default `lfu_rankguard` strategy is seeded from the offline reroute profile,
using activation frequency scaled by top-k as the initial protection signal.
It then updates scores online from raw verify-routing metadata with EMA:

```
score[j] = alpha * score[j] + (1 - alpha) * current_score[j]
```

Default config:

```
rank_guard_threshold = 0.15
rank_guard_ema_alpha = 0.95
```

### 2.3 Correctness Fixes

#### RankGuard score update was tied to verify-history queue

The prefetch runtime previously updated RankGuard scores after a branch that
could return early when `prefetch_use_verify_history=false`. That made the
online protection signal disappear under configurations that disabled the
verify-history queue.

Fix:

- Move `_update_rank_guard_scores(runtime_meta)` before the history-queue branch
  in `PrefetchRuntime.observe_runtime_meta()`.
- Add a regression test proving that scores update even when
  `prefetch_use_verify_history=false`.

#### RankGuard construction did not pass config parameters consistently

`ModelRunner` now uses a small helper to construct both RankGuard variants with
`num_experts`, `rank_guard_threshold`, and `rank_guard_ema_alpha`. This avoids a
factory path where a new strategy name could miss the model-specific expert
count or guard parameters.

#### Benchmark script did not expose required runtime knobs

`benchmarks/scripts/spec_verify_expert_count_stats.py` now accepts and forwards:

- `--prefetch-runtime-mode`
- `--draft-cuda-graph-enabled`
- `--draft-cuda-graph-cpu-backend`
- `--rank-guard-threshold`
- `--rank-guard-ema-alpha`

It also records generated text with UTF-8 JSON output and summarizes cache hit
metrics in a machine-readable form.

### 2.4 New Algorithm: `lfu_rankguard_online`

The benchmark results showed that the offline-seeded guard can over-protect
globally frequent experts and reduce locality for this single-prompt workload.
To test a lower-bias variant, a new strategy was added:

```
lfu_rankguard_online = LFU RankGuard eviction + no offline profile seed
```

Implementation details:

- `LFUOnlineRankGuardStrategy` subclasses `LFURankGuardStrategy`.
- It sets `profile_seed_enabled = False`.
- `seed_lfu_rank_guard_from_profile()` returns early for strategies that disable
  profile seeding.
- The strategy still learns from verify routing online and uses the same
  threshold/EMA eviction guard as `lfu_rankguard`.

This gives a direct comparison between:

- `lfu`: pure LFU baseline.
- `lfu_rankguard`: offline-seeded RankGuard plus online EMA.
- `lfu_rankguard_online`: online-only RankGuard.

---

## 3. Validation Commands

### 3.1 Unit And Compile Checks

Targeted tests were run in the `nano_moe` conda environment:

```bash
python -m unittest \
  tests.test_prefetch_runtime \
  tests.test_draft_reroute_profile \
  tests.test_config_prefetch \
  tests.test_spec_verify_expert_count_stats \
  tests.test_lfu_rankguard_cache_validation \
  tests.test_model_runner_cache_strategy
```

Result:

```
Ran 39 tests
OK
```

Syntax checks:

```bash
python -m py_compile \
  benchmarks/scripts/spec_verify_expert_count_stats.py \
  scripts/lfu_rankguard_cache_validation.py \
  nanovllm/config.py \
  nanovllm/engine/model_runner.py \
  nanovllm/expert/prefetcher.py \
  nanovllm/scheduling/cache_strategy.py \
  nanovllm/scheduling/draft_reroute_profile.py \
  tests/test_lfu_rankguard_cache_validation.py \
  tests/test_model_runner_cache_strategy.py
```

Result: OK.

### 3.2 Matrix Commands

Baseline LFU plus offline-seeded RankGuard:

```bash
python scripts/lfu_rankguard_cache_validation.py \
  --output-dir results/lfu_rankguard_cache_20260601_024816 \
  --dist-port-base 28600 \
  --case-timeout-sec 2400
```

Online-only RankGuard:

```bash
python scripts/lfu_rankguard_cache_validation.py \
  --output-dir results/lfu_rankguard_online_cache_20260601_141640 \
  --cache-strategies lfu_rankguard_online \
  --dist-port-base 28800 \
  --case-timeout-sec 2400
```

### 3.3 Cluster Environment

Baseline/default RankGuard run:

```
Slurm job: 28226
Node:      gpu11-A100-E1-3U
GPU:       NVIDIA A100-SXM4-80GB
CUDA_VISIBLE_DEVICES=0
torch:     2.9.1+cu128
```

Online RankGuard run:

```
Slurm job: 28331
Node:      gpu13-A100-E2-17U
GPU:       NVIDIA A100-SXM4-80GB
CUDA_VISIBLE_DEVICES=1
torch:     2.9.1+cu128
```

Both runs used the `nano_moe` conda environment and verified
`torch.cuda.is_available() == True` and `nanovllm` import success before the
matrix execution.

---

## 4. Results

### 4.1 Full Result Table

| cache | out | ratio | accept | cache hit | output tok/s | decode tok/s | draft ms | verify ms | graph replays | prefetch submit/done/used | text |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|:---|
| lfu | 128 | 0.25 | 0.3815 | 0.6536 | 1.583 | 2.292 | 24.149 | 1554.195 | 249 | 3730/3730/7029 | ok |
| lfu | 128 | 0.50 | 0.6772 | 0.8771 | 4.436 | 7.760 | 22.477 | 644.190 | 158 | 2326/2326/4556 | ok |
| lfu | 128 | 0.75 | 0.8527 | 0.9711 | 9.091 | 17.299 | 22.233 | 262.639 | 129 | 1172/1172/2419 | ok |
| lfu | 512 | 0.25 | 0.4515 | 0.7159 | 2.618 | 2.932 | 23.714 | 1381.526 | 886 | 13036/13036/14455 | ok |
| lfu | 512 | 0.50 | 0.7857 | 0.9209 | 7.335 | 9.173 | 22.772 | 603.211 | 560 | 7743/7743/7272 | ok |
| lfu | 512 | 0.75 | 0.9375 | 0.9887 | 16.629 | 21.413 | 23.026 | 206.790 | 480 | 3108/3108/3340 | ok |
| lfu_rankguard | 128 | 0.25 | 0.3815 | 0.6428 | 1.574 | 2.209 | 24.694 | 1616.573 | 249 | 3696/3696/7440 | ok |
| lfu_rankguard | 128 | 0.50 | 0.6213 | 0.8264 | 3.411 | 5.384 | 22.826 | 902.652 | 169 | 2635/2635/5221 | ok |
| lfu_rankguard | 128 | 0.75 | 0.7730 | 0.9699 | 8.557 | 15.377 | 22.795 | 280.752 | 141 | 1362/1362/2599 | ok |
| lfu_rankguard | 512 | 0.25 | 0.4432 | 0.6457 | 2.269 | 2.524 | 24.539 | 1597.567 | 898 | 13219/13219/15668 | ok |
| lfu_rankguard | 512 | 0.50 | 0.8158 | 0.9238 | 8.007 | 10.189 | 22.479 | 556.147 | 543 | 7586/7586/7532 | ok |
| lfu_rankguard | 512 | 0.75 | 0.8851 | 0.9903 | 16.835 | 21.741 | 21.950 | 191.690 | 505 | 2657/2657/3051 | ok |
| lfu_rankguard_online | 128 | 0.25 | 0.4145 | 0.6274 | 2.302 | 3.151 | 23.939 | 1164.391 | 234 | 3499/3499/7362 | ok |
| lfu_rankguard_online | 128 | 0.50 | 0.7105 | 0.8918 | 7.922 | 9.915 | 22.857 | 492.687 | 152 | 1905/1905/4798 | ok |
| lfu_rankguard_online | 128 | 0.75 | 0.8527 | 0.9697 | 8.174 | 13.272 | 23.764 | 382.680 | 129 | 1016/1016/2614 | ok |
| lfu_rankguard_online | 512 | 0.25 | 0.6177 | 0.7643 | 5.130 | 5.412 | 23.624 | 907.375 | 688 | 10303/10303/16915 | ok |
| lfu_rankguard_online | 512 | 0.50 | 0.8189 | 0.9449 | 13.483 | 14.859 | 22.546 | 322.945 | 541 | 4855/4855/8277 | ok |
| lfu_rankguard_online | 512 | 0.75 | 0.9108 | 0.9903 | 13.507 | 19.773 | 23.196 | 228.273 | 493 | 1846/1846/3960 | ok |

### 4.2 Delta Versus LFU

Positive values are improvements over the same output length and cache ratio
under pure LFU.

| cache | out | ratio | accept delta | cache-hit delta | output tok/s delta |
|:---|---:|---:|---:|---:|---:|
| lfu_rankguard | 128 | 0.25 | +0.0000 | -0.0108 | -0.009 |
| lfu_rankguard | 128 | 0.50 | -0.0559 | -0.0507 | -1.025 |
| lfu_rankguard | 128 | 0.75 | -0.0797 | -0.0012 | -0.533 |
| lfu_rankguard | 512 | 0.25 | -0.0083 | -0.0702 | -0.349 |
| lfu_rankguard | 512 | 0.50 | +0.0301 | +0.0029 | +0.673 |
| lfu_rankguard | 512 | 0.75 | -0.0524 | +0.0017 | +0.206 |
| lfu_rankguard_online | 128 | 0.25 | +0.0330 | -0.0262 | +0.719 |
| lfu_rankguard_online | 128 | 0.50 | +0.0333 | +0.0148 | +3.486 |
| lfu_rankguard_online | 128 | 0.75 | +0.0000 | -0.0014 | -0.917 |
| lfu_rankguard_online | 512 | 0.25 | +0.1663 | +0.0484 | +2.512 |
| lfu_rankguard_online | 512 | 0.50 | +0.0331 | +0.0241 | +6.149 |
| lfu_rankguard_online | 512 | 0.75 | -0.0267 | +0.0016 | -3.123 |

### 4.3 Text Quality

All 18 valid matrix cases passed the automated text-quality guard:

- `text_quality_ok=true`
- no replacement characters
- no empty output
- no repeated-line failure
- no repeated 12-gram failure

Manual spot checks were also normal. Example samples:

```
lfu_ratio25_l128:
The efficient inference of MoE models requires careful management of expert
memory access patterns. Recent work has explored techniques like expert
sharding, cache eviction policies, and pre-fetching strategies.
```

```
lfu_rankguard_online_ratio25_l512:
The main problem is how to optimally manage the cache to minimize the number of
expert weight transfers (which are expensive) while ensuring that the necessary
experts are available when needed.
```

The full text for each case is stored in the per-case JSON files under the two
result directories.

---

## 5. Analysis

### 5.1 Offline-Seeded `lfu_rankguard`

The original offline-seeded RankGuard was not a consistent improvement on this
workload.

Observations:

- Acceptance improved only at `out=512, ratio=0.50`.
- Cache hit rate regressed at low cache ratios, especially `out=512,
  ratio=0.25` where hit rate dropped from 0.7159 to 0.6457.
- Throughput improved slightly at `out=512, ratio=0.50/0.75`, but regressed in
  the other four cells.

Interpretation:

The offline profile is useful as a global popularity prior, but the guard can
over-protect experts that are frequent globally while not being optimal for the
current prompt's local routing stream. At lower cache ratios this makes eviction
too conservative and increases route misses.

### 5.2 Online-Only `lfu_rankguard_online`

The online-only variant performed better than the offline-seeded strategy in
this matrix.

Observations:

- Acceptance improved over LFU in 4 of 6 cells and tied in 1 cell.
- The largest acceptance gain was `+0.1663` at `out=512, ratio=0.25`.
- Cache hit improved at `128/0.50`, `512/0.25`, `512/0.50`, and `512/0.75`.
- Output throughput improved strongly at `128/0.50`, `512/0.25`, and
  `512/0.50`, but regressed at the 0.75 cache-ratio cells.

Interpretation:

Starting from LFU and adding only online rank protection avoids the global-prior
over-protection problem. It adapts to the current routing stream and is most
helpful when cache pressure is moderate or high. It is still not a universal
replacement for LFU because high cache ratios already have few misses, and the
extra guard can perturb a near-saturated cache without enough benefit.

### 5.3 CUDA Graph And Prefetch Status

Every case reported non-zero draft graph replays. Prefetch submit/completed
counts also matched in the summary files, so the requested draft CUDA graph path
and `draft_segment_indexed` prefetch path were active during the benchmark.

---

## 6. Conclusion

1. The audited LFU RankGuard implementation had one correctness issue in the
   online-score update path. It has been fixed and covered by a regression test.
2. The original `lfu_rankguard` strategy should not be promoted as a default
   replacement for LFU based on this run. It is only beneficial in a subset of
   the tested cells and regresses cache hit at low cache ratios.
3. The new `lfu_rankguard_online` strategy is the better candidate from this
   validation. It improves acceptance in most cells and gives the strongest
   throughput gains at `cache_ratio=0.25/0.50` for `out=512`.
4. `lfu_rankguard_online` should remain opt-in until repeated across more
   prompts and seeds. The current result is one meaningful-prompt run per matrix
   cell, not a statistically averaged benchmark.
5. All tested outputs were readable and passed the no-garbage/no-repetition text
   checks.
