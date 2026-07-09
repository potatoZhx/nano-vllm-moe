# Draft TPOT Stop Policy Tuning

Date: 2026-07-08

## Goal

Enable `--draft-stop-policy tpot` for the K=12 optimized decode workload, try
parameter tuning, and check whether it can exceed the current best
`decode_tok_s ~= 33`.  Every run was checked for normal output shape so that a
throughput increase cannot come from malformed or duplicated output.

## Benchmark Protocol

All runs used the same base workload:

```bash
conda activate nano_moe
cd /home/linke/nano-vllm-moe

NANOVLLM_VERIFY_PREFETCH_RANK_MULTIPLIER=1 \
NANOVLLM_VERIFY_DEFER_SEGMENT_METADATA=1 \
NANOVLLM_VERIFY_BOUNDARY_PREFETCH_ASYNC=0 \
CUDA_VISIBLE_DEVICES=2 python scripts/bench_eval_workload_tpot.py \
  --request-mode per_layer_slots \
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
  --verify-cuda-graph-bucket-steps 3,5,7,10,13 \
  --verify-prefetch-max-per-boundary 10 \
  --verify-prefetch-rank-multiplier 1 \
  --decode-driver generate \
  --collect-profile true \
  --engine-profile false \
  --save-token-ids true \
  --skip-existing false
```

Only `--output-dir`, `--draft-stop-policy`, `--draft-tpot-td-ms`, and
`--draft-tpot-tv-ms` changed between TPOT runs.

The benchmark harness now also exports:

- `profile_spec_draft_tpot_early_stop_count`
- `profile_spec_draft_alpha_early_stop_count`

Missing `*_count` profile keys are written as `0` in new summaries.

## Commands

Best known `none` baseline:

```bash
... --output-dir results/eval_workload_tpot_k12_generate_driver_profile_l512 \
  --draft-stop-policy none
```

Same baseline rerun with the current harness:

```bash
... --output-dir results/eval_workload_tpot_k12_none_rerun_generate_profile_l512 \
  --draft-stop-policy none
```

Default TPOT:

```bash
... --output-dir results/eval_workload_tpot_k12_tpot_default_generate_profile_l512 \
  --draft-stop-policy tpot \
  --draft-tpot-td-ms 19 \
  --draft-tpot-tv-ms 80
```

Higher verify/draft ratio:

```bash
... --output-dir results/eval_workload_tpot_k12_tpot_td10_tv120_generate_profile_l512 \
  --draft-stop-policy tpot \
  --draft-tpot-td-ms 10 \
  --draft-tpot-tv-ms 120
```

Near-full-K TPOT:

```bash
... --output-dir results/eval_workload_tpot_k12_tpot_td1_tv1000_generate_profile_l512 \
  --draft-stop-policy tpot \
  --draft-tpot-td-ms 1 \
  --draft-tpot-tv-ms 1000
```

No-stop TPOT sanity run:

```bash
... --output-dir results/eval_workload_tpot_k12_tpot_td0_tv1000_generate_profile_l512 \
  --draft-stop-policy tpot \
  --draft-tpot-td-ms 0 \
  --draft-tpot-tv-ms 1000
```

## Results

| run | decode tok/s | profile decode tok/s | e2e tok/s | verify calls | draft calls | avg draft/verify | acceptance | draft ms | verify ms | output check | digest |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|:---|
| none, old best | 33.021 | 33.036 | 30.688 | 48 | 568 | 11.83 | 0.815 | 19.192 | 90.255 | 512 tokens, seq=1, max repeat=1 | `f59581fe...` |
| none, rerun | 30.087 | 30.097 | 28.129 | 53 | 627 | 11.83 | 0.730 | 18.466 | 98.145 | 512 tokens, seq=1, max repeat=1 | `58ba0a53...` |
| tpot 19/80 | 27.180 | 27.190 | 25.579 | 72 | 589 | 8.18 | 0.745 | 18.619 | 104.901 | 512 tokens, seq=1, max repeat=1 | `3fa571df...` |
| tpot 10/120 | 26.846 | 26.857 | 25.287 | 66 | 648 | 9.82 | 0.687 | 18.762 | 99.932 | 512 tokens, seq=1, max repeat=1 | `a7bbc3ce...` |
| tpot 1/1000 | 31.505 | 31.516 | 29.354 | 52 | 611 | 11.75 | 0.751 | 18.356 | 92.329 | 512 tokens, seq=1, max repeat=1 | `f4fa8bf2...` |
| tpot 0/1000 | 27.501 | 27.513 | 25.869 | 58 | 688 | 11.86 | 0.658 | 19.017 | 89.873 | 512 tokens, seq=1, max repeat=1 | `b0fc5637...` |

The output validation fields were normal in all runs:

- `generated_output_tokens = 512`
- `output_sequence_count = 1`
- `output_fixed_length_ok = true`
- `output_validation_error = ""`
- `max_repeated_token_run = 1`

## Trace Observations

For default TPOT, the policy budgeted `853` draft steps but executed only `589`
draft calls.  The average draft length fell from the full-K baseline
`~11.83` to `8.18`, increasing verify calls from `48` in the old best baseline
to `72`.

For `td=10,tv=120`, the saved trace shows:

- actual draft-step histogram:
  `{1:1, 2:2, 3:3, 4:1, 5:1, 6:4, 7:3, 8:1, 9:2, 10:8, 11:6, 12:34}`
- accepted draft-token histogram:
  `{0:5, 1:5, 2:4, 3:6, 4:3, 5:5, 6:2, 7:6, 8:4, 9:3, 10:4, 11:4, 12:15}`
- first TPOT series:
  `[66.110, 51.426, 44.646, 41.108, 38.671, 37.743, 37.055, 36.610, 36.249, 36.274]`

This stop decision is internally consistent with the formula, but it still
loses wall-clock throughput because the extra verify calls dominate the saved
draft work.

For `td=1,tv=1000`, TPOT nearly degenerates to full K:

- actual draft-step histogram:
  `{7:2, 9:1, 12:49}`
- average draft/verify:
  `11.75`

This was the best TPOT run (`31.505` tok/s), but it still did not exceed the
old best `none` run (`33.021` tok/s).

For `td=0,tv=1000`, budgeted and executed draft calls were both `688`, so the
run did not materially early-stop.  It still produced a different output digest
and lower acceptance (`0.658`).  This confirms that a single stochastic sample
is very sensitive to output trajectory and asynchronous/cache timing; a
near-full-K TPOT run should not be claimed as an optimization unless it also
matches or improves verify-call count and output validation remains normal.

## Deeper Root-Cause Follow-Up

Additional diagnostic runs were made on 2026-07-09.  These runs enabled
`NANOVLLM_ACCEPTANCE_TRACE_PROBS=1` to write the standard-sampling ground-truth
per-token acceptance probability `min(1, p/q)` into the speculative step trace.
This diagnostic mode adds GPU-to-CPU scalar reads and should not be used for
throughput claims; it is only for policy analysis.

The benchmark harness was also extended with:

- `--prefetch-runtime-kind {predictive,dual_queue}`;
- `profile_model_verify_kt_hybrid_segment_graph_replay_count`;
- `profile_model_verify_cpu_routes_sum`;
- `profile_model_verify_realized_cpu_expert_count_sum`;
- `profile_model_verify_pre_transfer_cache_miss_sum`;
- `profile_model_verify_pre_transfer_active_count_sum`;
- `profile_model_run_verify_kt_hybrid_metadata_wait_ms`.

### Diagnostic Commands

Full-K `none` diagnostic:

```bash
NANOVLLM_ACCEPTANCE_TRACE_PROBS=1 \
NANOVLLM_VERIFY_PREFETCH_RANK_MULTIPLIER=1 \
NANOVLLM_VERIFY_DEFER_SEGMENT_METADATA=1 \
NANOVLLM_VERIFY_BOUNDARY_PREFETCH_ASYNC=0 \
CUDA_VISIBLE_DEVICES=2 python scripts/bench_eval_workload_tpot.py \
  --request-mode per_layer_slots \
  --output-dir results/eval_workload_tpot_k12_none_diag_accept_trace_l512 \
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
  --verify-cuda-graph-bucket-steps 3,5,7,10,13 \
  --verify-prefetch-max-per-boundary 10 \
  --draft-stop-policy none \
  --verify-prefetch-rank-multiplier 1 \
  --decode-driver generate \
  --collect-profile true \
  --engine-profile true \
  --engine-profile-cuda-sync false \
  --save-profile-json true \
  --save-token-ids true \
  --skip-existing false
```

Default TPOT diagnostic used the same command with:

```bash
--output-dir results/eval_workload_tpot_k12_tpot_default_diag_accept_trace_l512 \
--draft-stop-policy tpot \
--draft-tpot-td-ms 19 \
--draft-tpot-tv-ms 80
```

Two prefetch-budget checks were also run without acceptance-prob tracing:

```bash
# Smaller static verify boundary budget.
--output-dir results/eval_workload_tpot_k12_tpot_default_vpb4_l512 \
--draft-stop-policy tpot \
--draft-tpot-td-ms 19 \
--draft-tpot-tv-ms 80 \
--verify-prefetch-max-per-boundary 4

# Existing history/dynamic budget runtime.
--output-dir results/eval_workload_tpot_k12_tpot_default_dualqueue_l512 \
--draft-stop-policy tpot \
--draft-tpot-td-ms 19 \
--draft-tpot-tv-ms 80 \
--prefetch-runtime-kind dual_queue
```

All diagnostic and prefetch-budget runs produced normal fixed-length output:
`generated_output_tokens=512`, `output_sequence_count=1`,
`output_fixed_length_ok=true`, `output_validation_error=""`, and
`max_repeated_token_run=1`.

### Acceptance-Rate Finding

The original cross-run statement that TPOT lowers acceptance rate is not a
stable root cause.  It was confounded by stochastic output trajectory changes.
In the paired diagnostic runs, TPOT does increase the scalar acceptance rate:

| run | verify calls | draft calls | accepted draft tokens | acceptance | accepted/verify | output/verify |
|:---|---:|---:|---:|---:|---:|---:|
| none diagnostic | 53 | 624 | 458 | 0.734 | 8.64 | 9.66 |
| tpot 19/80 diagnostic | 72 | 549 | 439 | 0.800 | 6.10 | 7.11 |

So the deeper issue is not that shorter drafts intrinsically reduce
`accepted/drafted`.  They do not in this diagnostic run.  The problem is that
TPOT increases the fraction but reduces the absolute number of accepted draft
tokens per verify call.  It therefore needs many more verify rounds to reach
512 output tokens.

Per-position ground truth also matches the intuition that later prefix
acceptance decays:

| position | none true `min(1,p/q)` | none realized prefix accept | tpot true `min(1,p/q)` | tpot realized prefix accept |
|---:|---:|---:|---:|---:|
| 1 | 0.963 | 1.000 | 0.967 | 0.972 |
| 4 | 0.909 | 0.788 | 0.948 | 0.791 |
| 8 | 0.963 | 0.654 | 0.954 | 0.674 |
| 12 | 0.928 | 0.577 | 0.999 | 1.000 |

The high position-12 TPOT value has only 4 samples because TPOT usually stops
before reaching position 12.

### Predictor and TPOT Model Accuracy

The acceptance predictor is pessimistic but is not the only problem:

| run | mean true accept prob | mean predicted alpha | bias |
|:---|---:|---:|---:|
| none diagnostic | 0.932 | 0.902 | -0.030 |
| tpot diagnostic | 0.944 | 0.909 | -0.035 |

On the full-K `none` trace, using the true `min(1,p/q)` probabilities with the
same fixed cost model (`td=19`, `tv=80`) gives:

- true-oracle best K average: `8.12`;
- predicted first-increase TPOT K average: `7.77`;
- predicted decision true-cost / oracle true-cost: `1.096x`;
- full K=12 true-cost / oracle true-cost: `1.180x`;
- true-oracle chose K=12 for only `20/52` full-K steps.

This means that, under the current fixed formula, even perfect acceptance
probabilities would usually choose a shorter draft.  The predictor bias makes
the policy slightly too aggressive, but the larger issue is that the formula's
cost model does not match the real system.

No evidence was found that TPOT is non-unimodal on these traces.  The predicted
series had zero multi-turn direction changes in the diagnostic run.  The
implementation does have a one-token lag: it stops only after computing the
first token that makes predicted TPOT worse, so the current token remains in
verify.  That can waste one marginal draft token, but it does not explain the
main throughput loss because the observed dominant loss is extra verify rounds.

### Verify Optimization and CPU Route Finding

The diagnostic profile confirms that verify optimization was enabled:

| run | graph hit rate | verify KT segment graph replays | verify calls |
|:---|---:|---:|---:|
| none diagnostic | 1.0 | 53 | 53 |
| tpot diagnostic | 1.0 | 72 | 72 |

The reason verify does not speed up proportionally is that early-stop reduces
tokens and routes only modestly, while it worsens CPU expert batching and
exposes more graph-outside metadata wait:

| metric | none diagnostic | tpot 19/80 diagnostic | change |
|:---|---:|---:|---:|
| verify trace tokens | 677 | 621 | -8.3% |
| verify CPU routes | 28,200 | 26,709 | -5.3% |
| realized CPU expert invocations | 16,874 | 17,797 | +5.5% |
| CPU routes / token | 41.65 | 43.01 | +3.3% |
| CPU experts / token | 24.92 | 28.66 | +15.0% |
| verify ms total | 4,615 | 6,956 | +50.7% |
| verify ms / route | 0.164 | 0.260 | +59.1% |
| verify KT metadata wait | 88.8 | 468.5 | +427.5% |
| verify KT metadata wait / call | 1.68 | 6.51 | +288.4% |

The important detail is `realized CPU expert invocations`: TPOT has fewer
routes, but not fewer expert invocations.  Shorter verify segments have less
route aggregation per expert and more verify rounds, so CPUInfer pays more
per-expert/per-call overhead.  This is the missing nonlinearity in the fixed
TPOT model.

The cache/prefetch state also degrades slightly.  The miss-route ratio increased
from `28200/259968 = 10.85%` to `26709/238464 = 11.20%`.  That is not enough by
itself to explain the slowdown, but together with smaller route batches it
raises CPU expert cost per route.

### Prefetch Budget Attempts

Reducing static verify prefetch budget helped but did not close the gap:

| run | decode tok/s | verify calls | draft calls | acceptance | verify ms/call | output/verify |
|:---|---:|---:|---:|---:|---:|---:|
| tpot 19/80, vpb10 | 27.180 | 72 | 589 | 0.745 | 104.9 | 7.11 |
| tpot 19/80, vpb4 | 29.523 | 77 | 541 | 0.802 | 89.3 | 6.65 |
| none old best, vpb10 | 33.021 | 48 | 568 | 0.815 | 90.3 | 10.67 |

`vpb4` lowers per-verify latency close to the full-K baseline, which confirms
that graph-outside/prefetch work was overexposed in the TPOT path.  It still
does not win because it further reduces output tokens per verify and increases
verify calls to 77.

The existing `dual_queue` dynamic budget path was also tested.  It calibrated
to `verify_prefetch_max_per_boundary=16` for all verify segments and was much
slower:

| run | decode tok/s | verify calls | draft calls | verify ms/call | note |
|:---|---:|---:|---:|---:|:---|
| tpot 19/80, dual_queue | 22.926 | 78 | 518 | 111.3 | calibrated verify budget to 16 |

This dynamic policy estimates transfer capacity from segment compute time.  For
short TPOT verify segments that is the wrong signal: it increases transfer
budget instead of capping it by predicted verify length, route density, and
historical realized miss utility.

## Why TPOT Did Not Beat `none`

The root cause is not malformed output and not the Python cost of evaluating
the TPOT formula.  The main issue is that TPOT early stopping is making the
wrong tradeoff for this K=12 workload.

1. The current K=12 path is verify-dominated.  A draft forward call costs about
   `18-19 ms`, while each verify costs about `90-105 ms`.  Reducing draft
   tokens is only useful if it avoids enough rejected work without increasing
   verify iterations.  Default TPOT does the opposite: it shortens drafts and
   increases verify calls.

2. The TPOT decision uses predicted `acceptance_alpha`, not the realized
   acceptance probability.  The predictor is pessimistic by about `0.03-0.035`
   absolute probability on these traces, which makes the early stop slightly
   too aggressive.  However, even true oracle acceptance probabilities still
   prefer average K around `8.1` under the fixed `td=19,tv=80` formula, so
   predictor error is not the only root cause.

3. Static `td_ms` and `tv_ms` do not model the real verify cost.  Verify latency
   includes CPU expert invocation count, route batch density, prefetch waits,
   metadata offload, graph replay, and output trajectory effects.  A single
   fixed `tv_ms` cannot capture the marginal cost of shorter verify segments.

4. Early stopping changes the generation trajectory.  The output digest changes
   between runs.  For this stochastic single-prompt workload, decode tok/s is
   tightly coupled to accepted-token trajectory, not just operator latency.

5. The current implementation can only decide after a draft token has already
   been computed because `acceptance_alpha` is returned by `run_draft`.  The
   first token that makes predicted TPOT worse has already paid its draft cost.
   Dropping it before verify would save only marginal verify work and would not
   fix the dominant problem: TPOT is already under-drafting relative to the
   best `none` run.

## Optimization Assessment

For the current K=12 optimized decode setting, keep:

```bash
--draft-stop-policy none
```

The most favorable TPOT setting tested was `td=1,tv=1000`, which mostly keeps
full-K drafting and reached `31.505` tok/s.  That is still below the old best
`none` result (`33.021` tok/s).  Making TPOT more aggressive hurts because it
adds verify iterations; making TPOT less aggressive converges to `none`.

Potential future improvements before TPOT can be useful:

- add a minimum draft floor, for example do not consider TPOT stop before
  `k >= 10` for K=12;
- use hysteresis, for example stop only when predicted TPOT is worse by a
  margin for two consecutive draft positions;
- replace fixed `td_ms/tv_ms` with a measured rolling cost model that includes
  realized verify length, CPU expert count, prefetch wait, metadata offload,
  and graph bucket;
- calibrate `acceptance_alpha` against realized acceptance and report
  predicted-vs-realized curves per draft position;
- evaluate on multi-sample or deterministic replay, because a single stochastic
  512-token output can shift verify calls by more than the expected gain.

Until those changes exist, TPOT should be treated as diagnostic/tuning
infrastructure rather than the production setting for the K=12 optimized decode
path.

## Dynamic Cost Model Attempt

Date: 2026-07-09

The static TPOT formula assumes:

```text
T(k) = (k * td_ms + tv_ms) / (E[accepted_draft_tokens] + 1)
```

That is too simple for the optimized K=12 verify path because short verify
segments have worse CPU expert batching and expose more graph-outside metadata
wait.  A new, backward-compatible TPOT implementation was added:

- `--draft-tpot-cost-model static|history`
- `--draft-tpot-history-alpha`
- `--draft-tpot-min-steps`
- `--draft-tpot-stop-margin`
- `--draft-tpot-short-verify-penalty-ms`
- `--draft-tpot-verify-cost-floor-ms`
- `--draft-tpot-stop-rule first_increase|best_margin`
- `--verify-prefetch-tpot-dynamic-budget-enabled`
- `--verify-prefetch-tpot-dynamic-budget-token-threshold`
- `--verify-prefetch-tpot-dynamic-budget-small`

The old behavior is unchanged when `--draft-tpot-cost-model static` and
`--draft-tpot-stop-rule first_increase` are used.

### Implementation

`SpeculativeEngine` now keeps online EWMA cost estimates:

- draft cost EWMA from actual `run_draft` latency;
- verify cost EWMA from actual `run_verify` latency;
- optional per-verify-length verify EWMA;
- an optional short-verify penalty:
  `short_penalty_ms * (max_draft_tokens - candidate_draft_len)`;
- an optional verify cost floor, used as an opportunity-cost floor rather than
  a literal measured latency.

The history model uses:

```text
td(k) = draft_latency_ewma
tv(k) = max(observed_verify_ewma_for_len_or_global, verify_cost_floor)
        + short_verify_penalty_ms * (Kmax - k)
```

`best_margin` stops only when the current predicted TPOT is worse than the best
seen TPOT by a margin, and `draft_tpot_min_steps` prevents low-K stops before a
floor.  This addresses local non-monotonicity without changing the default
policy.

`ModelRunner.run_verify` also gained a TPOT-only dynamic prefetch cap.  When
enabled, if verify token count is below a threshold, the current verify round
temporarily lowers `verify_prefetch_max_per_boundary` to a smaller value.  This
targets the observed short-verify metadata/prefetch overhead.

### Commands

Best dynamic-cost run:

```bash
NANOVLLM_VERIFY_PREFETCH_RANK_MULTIPLIER=1 \
NANOVLLM_VERIFY_DEFER_SEGMENT_METADATA=1 \
NANOVLLM_VERIFY_BOUNDARY_PREFETCH_ASYNC=0 \
CUDA_VISIBLE_DEVICES=2 python scripts/bench_eval_workload_tpot.py \
  --request-mode per_layer_slots \
  --output-dir results/eval_workload_tpot_k12_tpot_history_floor200_min10_dynbudget_l512 \
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
  --verify-cuda-graph-bucket-steps 3,5,7,10,13 \
  --verify-prefetch-max-per-boundary 10 \
  --draft-stop-policy tpot \
  --draft-tpot-td-ms 19 \
  --draft-tpot-tv-ms 80 \
  --draft-tpot-cost-model history \
  --draft-tpot-history-alpha 0.2 \
  --draft-tpot-min-steps 10 \
  --draft-tpot-stop-margin 0.02 \
  --draft-tpot-short-verify-penalty-ms 8 \
  --draft-tpot-verify-cost-floor-ms 200 \
  --draft-tpot-stop-rule best_margin \
  --verify-prefetch-tpot-dynamic-budget-enabled true \
  --verify-prefetch-tpot-dynamic-budget-token-threshold 13 \
  --verify-prefetch-tpot-dynamic-budget-small 4 \
  --verify-prefetch-rank-multiplier 1 \
  --prefetch-runtime-kind predictive \
  --decode-driver generate \
  --collect-profile true \
  --engine-profile false \
  --save-token-ids true \
  --skip-existing false
```

Mechanism/profile run for the same configuration:

```bash
... --output-dir results/eval_workload_tpot_k12_tpot_history_floor200_min10_dynbudget_profile_l512 \
  --engine-profile true \
  --save-profile-json true
```

More conservative near-full-K run:

```bash
... --output-dir results/eval_workload_tpot_k12_tpot_history_floor800_min11_dynbudget_l512 \
  --draft-tpot-min-steps 11 \
  --draft-tpot-stop-margin 0.05 \
  --draft-tpot-short-verify-penalty-ms 20 \
  --draft-tpot-verify-cost-floor-ms 800
```

Static vpb4 check with the same history model:

```bash
... --output-dir results/eval_workload_tpot_k12_tpot_history_floor200_min10_vpb4_l512 \
  --verify-prefetch-max-per-boundary 4 \
  --verify-prefetch-tpot-dynamic-budget-enabled false
```

### Results

| run | decode tok/s | verify calls | draft calls | avg draft/verify | acceptance | draft ms/call | verify ms/call | output check |
|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| none old best | 33.021 | 48 | 568 | 11.83 | 0.815 | 19.19 | 90.26 | ok |
| default tpot vpb10 | 27.180 | 72 | 589 | 8.18 | 0.745 | 18.62 | 104.90 | ok |
| default tpot vpb4 | 29.523 | 77 | 541 | 7.03 | 0.802 | 18.67 | 89.32 | ok |
| history floor200/min10/dynbudget | 30.673 | 51 | 593 | 11.63 | 0.776 | 19.24 | 97.25 | ok |
| history floor200/min10/vpb4 | 29.667 | 54 | 613 | 11.35 | 0.746 | 18.68 | 102.04 | ok |
| history floor800/min11/dynbudget | 29.966 | 52 | 614 | 11.81 | 0.749 | 18.81 | 96.45 | ok |

Every run produced normal output:

- 512 generated tokens;
- one sequence;
- `output_fixed_length_ok=true`;
- `output_validation_error=""`;
- `max_repeated_token_run=1`.

The best dynamic-cost model improved default TPOT from `27.180` to `30.673`
decode tok/s and reduced verify calls from `72` to `51`.  It still did not
exceed the old `none` best (`33.021`).

### Mechanism Profile

The profile run confirmed the dynamic pieces were active:

| metric | value |
|:---|---:|
| verify KT segment graph replays | 56 |
| verify calls | 56 |
| dynamic verify budget applied | 56 |
| dynamic budget value sum | 224 |
| average dynamic budget | 4 |
| draft steps histogram | `{9:1, 10:9, 11:1, 12:45}` |
| early-stop steps | 11 |

The history model's cost series used the intended opportunity cost.  For early
steps, K=12 used `verify_ms=200`, while short K candidates paid
`200 + 8 * (12-k)`.

The measured verify EWMA in the profile run ended at `65.1 ms`, while measured
`verify_forward_ms` was `96.4 ms/call`.  This gap shows why an EWMA-only model
is insufficient: the online latency observed by the speculative controller is
not a stable predictor of the realized route/metadata-heavy verify path.  The
explicit floor and short-verify penalty are required to keep TPOT from
under-drafting.

### Stop-Rule Check

Default TPOT traces did not show non-unimodality:

```text
default tpot diagnostic:
  non-unimodal steps = 0 / 72
  recover-after-first-increase = 0
```

The history-cost profile had only one step with more than one direction change:

```text
history-cost profile:
  non-unimodal steps = 1 / 56
  recover-after-first-increase = 1
```

`best_margin` covers this rare local-recovery case, but the data does not show
that non-unimodality is the dominant reason TPOT loses.  The dominant issue is
still that profitable early-stop opportunities are too sparse once verify CPU
expert batching and graph-outside metadata cost are included.

### Dynamic Model Conclusion

The dynamic cost model fixed the most damaging behavior of static TPOT:

- default TPOT under-drafted (`8.18` draft/verify) and caused `72` verify calls;
- dynamic TPOT drafted near full K (`11.63` draft/verify) and needed only `51`
  verify calls;
- draft time per call did not materially increase (`19.24 ms` vs `19.19 ms`
  for none);
- verify per call did not reliably decrease in the conservative model
  (`97.25 ms`), although the aggressive vpb4 TPOT run did reduce verify per call
  (`89.32 ms`) at the cost of too many verify calls (`77`).

This leaves a real tradeoff:

- aggressive TPOT can reduce verify ms/call, but loses because it creates too
  many verify rounds;
- conservative TPOT can reduce verify rounds, but it becomes close to `none`
  and no longer obtains reliable per-call verify savings;
- pushing the model further toward full K simply converges to `none`, with
  stochastic trajectory differences and no consistent speedup.

The practical production recommendation remains `--draft-stop-policy none` for
this K=12 single-sample workload.  TPOT is useful as a diagnostic/tuning
mechanism, but to beat `none` it would need a route-aware predictor available
before the stop decision.  The current draft-side state exposes acceptance
alpha, but not predicted verify CPU routes or realized expert batching cost.
