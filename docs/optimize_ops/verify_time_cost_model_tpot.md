# Verify Time Cost Model for Speculative TPOT

Date: 2026-07-11

## 2026-07-13 Multi-Horizon Update

The promoted sampling model remains valid for the current cache snapshot and
its vpb10 protocol. A new high-K diagnostic identified a separate extrapolation
failure: predicting a later verify boundary while freezing the current cache
state overestimates the CPU-expert workload because intervening draft steps
continue expert prefetch.

With K6 -> K9 and K9 -> K12 three-step projections:

| endpoint comparison | pairs | predicted | reached/actual | mean bias |
|:---|---:|---:|---:|---:|
| K6 causal endpoint -> actual K9 verify | 18 | 111.885 ms | 87.473 ms | +24.411 ms |
| K9 causal endpoint -> actual K12 verify | 10 | 139.377 ms | 112.026 ms | +27.351 ms |

For realized K9 calls, the prediction made after reaching K9 had MAE
`9.794 ms`; the prediction made three draft steps earlier had MAE `27.906 ms`.
Thus CPU experts remain the correct verify-time driver, but a multi-horizon
CPU-expert forecast must include future cache/prefetch state.

The early-stop implementation now separates these quantities in trace data:

- `lookahead_verify_raw_ms`: model result under the current cache snapshot;
- `lookahead_cache_credit_ms`: explicit future-draft overlap correction;
- `lookahead_verify_ms`: adjusted endpoint cost consumed by the policy.

The temporary candidate correction is `8.5 ms` per future draft step and is
disabled by default. It was derived from one mechanism sample and is not part
of the promoted verify artifact. Trace error attribution was also fixed: a
prediction is scored against measured verify only when
`verify_cost_candidate_len == draft_steps_actual`; the last earlier decision
prediction is no longer mislabeled as a Kmax error.

The high-K online candidate reached mean K `8.86` but only `30.686 tok/s`
(`32.589 ms` TPOT), so it does not replace the formal K6/vpb4 baseline of
`33.501 tok/s`. See `draft_tpot_stop_policy_analysis.md` for the policy result.

## Earlier 2026-07-13 Current Status

This section supersedes the production status recorded on 2026-07-11 below.
The old section is retained as the audit trail for the failed first model, not
as the current deployment conclusion.

The measurement boundary and full-bucket CPU-work contract remain unchanged.
Two additional requirements were enforced before promotion:

- `standard_sampling` profiles must measure through the host-observable
  acceptance result; allowing `return_logits` is valid only when the trace has
  a positive acceptance duration, every sequence records sampling mode, and a
  concrete `next_token` was consumed;
- a sampling model is bound to its acceptance strategy and temperature, and is
  rejected at load or request time if the runtime protocol differs.

The base greedy model did not transfer directly to sampling. A frozen,
sampling-specific affine correction was fitted on the first sampling shadow
only, then checked on two CPU-count replays and a new instrumentation-off run.
The fresh deployment shadow passed the predeclared v2 gates:

| metric | result | gate |
|:---|---:|---:|
| calls | 574 | n/a |
| MAE | 5.776 ms | <=7.0 ms |
| P90 absolute error | 10.311 ms | <=14.0 ms |
| ranking accuracy | 0.837 | >=0.82 |
| bucket MAE maximum | 7.250 ms | <=8.0 ms |
| bucket P90 maximum | 16.729 ms | <=17.0 ms |

The measured boundary in that run was `79.655 ms` acceptance-ready,
`75.560 ms` verify stream, and `4.095 ms` outside the stream. Every bucket
`{3,5,7,10,13}` passed. The promoted artifact is:

```text
results/verify_cost_sampling_shadow_20260713/verify_time_cost_model.active.sampling.v2.json
model_id = 63d0b1892bc2bb5d833ddbaaa4a343d8bfe2f39408d34ff7a9db8cc2b7a8bd3e
sha256 = fb385d67168137fe21a65bfd6663596aa749c0bfe89923ce9778d19fda67de16
```

Sampling acceptance alpha was calibrated independently. On 2,230 untouched
validation points, affine calibration improved Brier score by `23.41%` and
changed mean bias from `-0.03212` to `-0.00243`. The calibration is bound to
`standard_sampling` at temperature `0.8`:

```text
results/acceptance_alpha_sampling_calibration_20260713/
  acceptance_alpha_calibration.standard_sampling.json
calibration_id = d1278ab97ac050e7158701650191a5211c151bca202fba56febbc37ac9d1f966
```

Passing the model gates does not imply that the active stop policy improves
TPOT. Two small policy screens both favored fixed K6:

| screen | fixed K6 | best active | active delta |
|:---|---:|---:|---:|
| stage 1 | 31.702 tok/s | 31.329 tok/s | -1.18% |
| stage 2 | 32.688 tok/s | 31.932 tok/s | -2.32% |

The active hot path now defers route-cost prediction until `min_steps` and
does not predict at the final maximum-draft step. This removes predictions
that cannot affect a decision. It was not sufficient to beat fixed K6 on this
workload, so `active` remains opt-in and is not the production recommendation.
The promoted sampling artifact was calibrated and shadowed with vpb10. It must
not be reused for vpb4 active decisions without a new vpb4 calibration and
instrumentation-off shadow.

The production TPOT improvement instead came from the draft/verify fast path:

- fixed `K=6`, which maps the 7-token verify input exactly to bucket 7;
- `draft_stop_policy=none`;
- acceptance predictor disabled because fixed K consumes neither alpha nor the
  draft-route verify proxy;
- verify boundary prefetch budget reduced from 10 to 4.

This fixed policy does not load the verify-time artifact, so its vpb4 result is
independent of the vpb10 model-scope restriction above.

Five paired independent-process seeds at cache ratio `0.3125`, output length
512, temperature `0.8`, and profile instrumentation off produced:

| policy | decode tok/s geomean | mean TPOT |
|:---|---:|---:|
| K6, predictor off, vpb10 | 31.370 | 31.969 ms |
| K6, predictor off, vpb4 | 33.501 | 29.877 ms |

The vpb4 decode-rate geomean improvement is `6.79%`; paired cluster bootstrap
95% CI is `[+2.39%, +11.12%]`. TPOT reduction is `6.26%` on average with 95%
CI `[+2.16%, +10.00%]`. Four of five pairs improved. All ten outputs contained
exactly 512 tokens and the text/token degeneration checks found no failures.
All paired digests differ, so this is an end-to-end deployment comparison, not
a same-token microbenchmark.

Reports:

```text
results/tpot_target_fixed_k_vpb4_predictor_off_20260713/fixed_k_analysis.json
results/tpot_vpb4_final_analysis_20260713/policy_validation.json
```

Use the validated preset for this target:

```bash
python scripts/bench_eval_workload_tpot.py \
  --request-mode per_layer_slots \
  --optimized-config k6_decode \
  --output-lens 512 \
  --output-dir results/tpot_k6_decode \
  --save-token-ids true --save-text true --skip-existing false
```

The preset fixes cache ratio `0.3125`, K6, segment size 12, vpb4,
`avx2_bf16`, graph buckets `3,5,7,10,13`, the `generate` driver, no stop
policy, and predictor off. Explicit CLI values still override the preset.

## Historical 2026-07-11 Status

The production status is intentionally conservative:

- The verify-time model passes when the real CUDA-graph execution workload is
  known: grouped-holdout MAE `4.772 ms`, P90 absolute error `9.075 ms`, and
  ranking accuracy `0.946`.
- The causal draft-route proxy does not pass: MAE `5.968 ms`, P90
  `11.255 ms`, and ranking accuracy `0.838` on its independent holdout.
- The frozen two-stage model also fails an instrumentation-off deployment
  shadow: MAE `6.387 ms`, P90 `12.132 ms`, and ranking accuracy `0.890`.
- `active` mode remains blocked. No TPOT or decode-tokens/s improvement is
  claimed from the new model, and no active policy benchmark was run after the
  failed deployment gate.

This is a useful result: the verify latency relationship is now measured and
modeled reliably, while the remaining failure is isolated to pre-verify CPU
workload prediction and distribution shift rather than hidden by an unsafe
early-stop experiment.

## Correction to the Previous Analysis

The previous version concluded that CPU route counts were weak predictors. That
conclusion used logical-token metadata while verify CUDA graphs and KT CPUInfer
execute the complete bucket, including padding rows. The independent variable
and the timed work therefore described different executions.

The corrected profiler records both:

- logical rows, retained for request semantics and compatibility;
- full bucket execution rows, including padding, used for cost modeling.

For every modeled verify call, the validator requires:

```text
sum(layer, expert, execution_route_count)
    == bucket * top_k * num_layers
```

For this model that is `bucket * 8 * 48`. Any profile that does not satisfy the
identity, has an incomplete layer histogram, or cannot be joined uniquely by
`step_id` and `call_index` is rejected.

## Timing Target

The optimization target is acceptance-ready latency, not the Python return
time of an asynchronously enqueued forward:

```text
verify_accept_ready_ms
    = time immediately before ModelRunner.run_verify
      through host consumption of the verify result by acceptance
```

`ModelRunner.run_verify` also records a CUDA stream event around verify
forward/logits. Events are drained after the measured request; normal cost
profiles do not synchronize on every call.

The clean calibration contains 3,049 valid post-drop calls:

| boundary | mean |
|:---|---:|
| acceptance-ready | 101.182 ms |
| verify stream | 97.981 ms |
| acceptance-ready minus stream | 3.201 ms |

The stream/acceptance-ready correlation is `0.99988`. This establishes that
the variable portion is overwhelmingly in the verify execution stream. The
`acceptance-ready - stream` residual is small, but it is not a pure tail: the
stream event starts inside `run_verify`, so the residual includes both
pre-event preparation and post-event argmax/D2H/acceptance. The model target
intentionally includes all of it.

## Latency Decomposition

### Diagnostic op-event profile

A separate synchronized diagnostic used 6 profiles and 62 post-drop calls at
cache ratio `0.28125`, output length 48, and `K={1,5,12}`. It is excluded from
model fitting and performance claims because event synchronization perturbs
scheduling. Labels are nested and must not be summed as independent wall-time
components.

| nested label or boundary | mean | P90 | corr. acceptance-ready | corr. CPU experts |
|:---|---:|---:|---:|---:|
| acceptance-ready | 108.937 ms | 138.525 ms | 1.000 | n/a |
| verify stream | 106.385 ms | 135.904 ms | n/a | n/a |
| stream-external residual | 2.552 ms | 2.897 ms | n/a | n/a |
| `layer.total` | 94.119 ms | 119.136 ms | 0.995 | 0.933 |
| `layer.moe` | 86.517 ms | 111.727 ms | 0.995 | 0.934 |
| `moe.cpu_sync_copy` | 58.239 ms | 79.465 ms | 0.978 | 0.924 |
| `kt.cpuinfer_sync` | 45.442 ms | 67.216 ms | 0.987 | 0.943 |
| `layer.attention` | 5.367 ms | 5.416 ms | 0.273 | 0.146 |

`kt.cpuinfer_sync` is the exposed wait at the CPU/GPU merge point, not a
measurement of all CPU compute. Even with that limitation, its strong
correlation with both total latency and CPU experts, together with the stable
attention timing, supports CPU expert workload as the primary source of verify
latency variation.

Per-layer CPUInfer wait also increases with execution workload:

| CPU routes in layer | rows | mean sync | P90 sync |
|:---|---:|---:|---:|
| 0 | 3 | 0.049 ms | 0.056 ms |
| 1-3 | 40 | 0.055 ms | 0.063 ms |
| 4-7 | 536 | 0.463 ms | 0.524 ms |
| 8-15 | 1,969 | 0.897 ms | 1.188 ms |
| 16-31 | 396 | 1.831 ms | 2.952 ms |
| >=32 | 32 | 2.337 ms | 2.921 ms |

### Clean same-bucket evidence

The unsynchronized calibration gives the stronger model evidence. Within each
bucket, aggregate execution CPU expert count remains highly correlated with the
acceptance-ready target:

| bucket | calls | target std | corr. CPU routes | corr. CPU experts |
|---:|---:|---:|---:|---:|
| 3 | 1,179 | 17.183 ms | 0.889 | 0.938 |
| 5 | 631 | 20.668 ms | 0.952 | 0.963 |
| 7 | 421 | 24.839 ms | 0.936 | 0.969 |
| 10 | 464 | 29.047 ms | 0.886 | 0.972 |
| 13 | 354 | 31.174 ms | 0.869 | 0.977 |

This rules out the explanation that the overall correlation is only a shared
effect of larger buckets.

## Model

For a verify workload, define:

- `B`: selected CUDA-graph bucket;
- `L`: logical token count;
- `P = B - L`: padding token count;
- `R`: total CPU routes across all bucket rows and all 48 layers;
- `E`: total active CPU experts, summed over layers.

The selected model is a standardized linear model:

```text
x = [one_hot(B, reference=3), L, R, E]
z_i = (x_i - mean_i) / scale_i
verify_ms = max(minimum_ms, intercept + sum_i(beta_i * z_i))
```

Bucket 3 is the reference category. A full bucket one-hot vector plus an
intercept would be singular, and retaining both logical and padding tokens would
be redundant because `B = L + P`. The final design has 8 columns including the
intercept, rank 8, and condition number `24.67`.

The runtime artifact stores feature order, means, scales, coefficients, minimum
latency, bucket list, error statistics, full unknown-row route priors, and a
hardware fingerprint. Runtime loading rejects a fingerprint mismatch.

The reported decomposition is computed by evaluating the same regression with
CPU workload set to zero:

```text
fixed_ms = model(B, L, P, R=0, E=0)
exposed_cpu_ms = max(0, model(B, L, P, R, E) - fixed_ms)
```

On the clean holdout the means are:

- modeled fixed component: `43.994 ms`;
- modeled exposed CPU component: `56.704 ms`;
- exposed component versus actual latency correlation: `0.982`.

These are regression allocations, not physically isolated additive timings.
Only the op-event diagnostic provides operator attribution, and its nested
events are also not additive.

## Clean Calibration

The exclusive run used:

- cache ratios `0.25`, `0.28125`, and `0.3125`, one process per ratio;
- datasets `per_layer_slots`, MT-Bench, HumanEval, and ShareGPT;
- draft limits `K=1..12`, with early stop disabled;
- greedy temperature-zero decoding, fixed output length 64;
- 252 profile files and 3,049 valid post-drop verify calls;
- RTX 4090, AMD EPYC 9554, KT `0.6.2.post4`, 16 threads, AVX2 BF16;
- no concurrent CPU/GPU experiment.

The split is by profile/source, not by individual verify call: 2,449 training
calls and 600 holdout calls. The selected model is refit on all 3,049 calls only
after the fixed holdout comparison.

| candidate | features | ridge | MAE | P90 | R2 | ranking | gate |
|:---|---:|---:|---:|---:|---:|---:|:---:|
| bucket only | 5 | 0 | 15.386 | 32.505 | 0.535 | 0.719 | FAIL |
| global CPU counts | 7 | 0 | 4.772 | 9.075 | 0.965 | 0.946 | PASS |
| per-layer counts | 103 | 10 | 4.819 | 8.835 | 0.965 | 0.945 | PASS |
| per-layer shape | 112 | 10 | 4.886 | 8.407 | 0.966 | 0.944 | PASS |

The simplest passing model is selected. It reduces MAE by `10.614 ms`, or
`69.0%`, relative to bucket-only prediction. More detailed layer features do
not provide a reliable holdout improvement.

Artifact:

```text
results/verify_time_cost_calibration_clean_20260711/verify_time_cost_model.json
model_id = 1bfe24ec40ec5ec224f7043129d3500ad8ee52ec028275a4478ceaca55589841
```

The proxy and shadow runs below were collected immediately before the reference
bucket/padding redundancy was removed, using model id `78fee0...`. The old and
new global-count parameterizations are algebraically equivalent for every valid
workload (`B = L + P`) and have identical holdout predictions. Their failed
proxy calibration execution workloads were also replayed through both models:
2,580 raw calls had maximum prediction delta `8.7e-12 ms`. The failed
proxy/deployment gates are reported as evidence about that collected artifact;
no deployment validation is transferred to the new model id.

## Pre-Verify Workload Prediction

The real execution histogram is only known after verify, so active early stop
needs a causal proxy. The implementation:

1. reads acceptance alpha and original draft routes in one D2H transfer;
2. classifies known draft routes against a snapshot of cached and ready experts;
3. fills the final input and CUDA-graph padding rows from learned per-bucket,
   per-layer expert-route priors;
4. predicts aggregate execution CPU routes and experts;
5. passes those two values to the verified latency model.

The clean proxy calibration used 105 profiles and 2,370 post-drop calls at
output length 96. Feature/ridge selection used a tuning group; final metrics use
a separate grouped holdout of 579 calls.

| result | MAE | P90 | R2 | ranking |
|:---|---:|---:|---:|---:|
| actual workload through frozen base model | 5.239 ms | 9.293 ms | 0.852 | 0.869 |
| predicted workload through frozen base model | 5.968 ms | 11.255 ms | 0.799 | 0.838 |

Workload prediction itself has:

- CPU routes: MAE `31.88`, P90 `69.78`, R2 `0.961`;
- CPU experts: MAE `23.66`, P90 `57.19`, R2 `0.930`.

The proxy adds error, but the oracle row also shows output-horizon/cache-state
distribution shift in the frozen base model. It is therefore incorrect to
attribute the whole failure to route prediction.

### Output-horizon pooling diagnostic

As a predeclared distribution-shift check, the output-64 base calibration and
the output-96 proxy-calibration execution workloads were pooled and refit. The
5,419-call grouped holdout remained strong (MAE `4.745 ms`, P90 `8.421 ms`,
ranking `0.937`). That result alone is not an external validation.

The frozen causal CPU-count predictions from the untouched offset-2 shadow
were then replayed through the pooled base. Replaying them through the original
source model first reproduced every stored prediction exactly (maximum delta
`0.0 ms`), which validates the replay boundary. Changing only the base model
produced:

| external shadow prediction | MAE | P90 | ranking |
|:---|---:|---:|---:|
| original two-stage artifact | 6.387 ms | 12.132 ms | 0.890 |
| pooled-horizon base, frozen proxy counts | 6.263 ms | 11.834 ms | 0.891 |

Bucket 13 still had MAE `7.271 ms` and P90 `15.298 ms`. The pooled model
therefore also fails the fixed gates. This replay is diagnostic and is not an
instrumentation-off execution of the pooled model id, so deployment validation
is explicitly not transferred. The pooled artifact was not promoted.

## Independent Deployment Shadow

The final shadow used the frozen two-stage artifact with workload
instrumentation disabled. It used 105 new profiles, output length 96, and
dataset offset 2, so MT-Bench, HumanEval, and ShareGPT prompts differ from the
calibration prompts. The fixed per-layer synthetic prompt explicitly ignores
dataset offsets.

Overall, 2,381 post-drop calls produced:

| metric | value | gate |
|:---|---:|---:|
| MAE | 6.387 ms | <=5.0 ms |
| P90 absolute error | 12.132 ms | <=10.0 ms |
| ranking | 0.890 | >=0.9 |
| within-source ranking | 0.804 | diagnostic |

Per-bucket results:

| bucket | calls | MAE | P90 | bucket gate |
|---:|---:|---:|---:|:---:|
| 3 | 1,056 | 6.282 | 11.504 | PASS |
| 5 | 542 | 6.292 | 11.716 | PASS |
| 7 | 342 | 6.058 | 12.142 | PASS |
| 10 | 250 | 6.658 | 12.923 | FAIL |
| 13 | 191 | 7.464 | 15.433 | FAIL |

The stream-external residual remains stable at `2.759 ms` mean. The failed
long-bucket errors are therefore not explained by drift in this timing
residual.

Artifact and report:

```text
results/verify_cost_shadow_clean_offset2_20260711/shadow_validation.json
results/verify_cost_shadow_clean_offset2_20260711/verify_time_cost_model.validated.json
deployment_validation.passed = false
```

## Historical Early-Stop Policy Status (2026-07-11)

The original `first_increase` rule is reactive: it only discovers a worse
candidate after paying for that draft step, and then verifies the worse
candidate. A `lookahead` rule was added. It repeats the current alpha as a
one-step persistence estimate and consumes a pre-draft `verify(K+1)` proxy.
If the lookahead value is unavailable, it falls back to `first_increase` rather
than silently drafting the full budget.

This rule is not enabled for production. On 229 full K=12 shadow curves:

- only `27.1%` of model-predicted curves are approximately unimodal;
- runtime-proxy lookahead with 8% margin has model-internal mean regret `12.1%`;
- sweeping margins `0`, `2%`, `5%`, `8%`, and `10%` does not remove the risk.

Those curves use predicted verify cost and observed draft-call cost, not measured
counterfactual TPOT. They can reject an unsafe policy, but cannot establish a
throughput improvement. Because both the model deployment gate and this risk
screen fail, paired active/static/none TPOT experiments were not run.

## Historical Safety Gates (2026-07-11)

`active` runtime loading requires all of:

```text
accuracy_gate_passed == true
proxy_workload_gate_passed == true
deployment_validation.passed == true
deployment_validation.model_id == model_id
```

Shadow mode may load a failed artifact for diagnosis, but never changes the
stop decision. Model identity covers coefficients, feature schema, priors,
fingerprint, and proxy-workload model. Tampering or a hardware mismatch fails
loading.

The fixed gates are:

- training/deployment MAE <= `5 ms`;
- P90 absolute error <= `10 ms`;
- ranking accuracy >= `0.9`;
- per-bucket shadow MAE <= `7.5 ms` and P90 <= `12.5 ms`.

No threshold was relaxed after observing the results.

## Reproduction

Clean base calibration:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n nano_moe python \
  scripts/collect_verify_time_cost_profiles.py \
  --output-dir results/verify_time_cost_calibration_clean_20260711 \
  --output-tokens 64 --num-samples 2 --seed 20260711 \
  --dist-port-base 35000 --no-resume
```

Proxy calibration (exit 1 is the recorded gate failure):

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n nano_moe python \
  scripts/collect_verify_cost_shadow_profiles.py \
  --artifact results/verify_time_cost_calibration_clean_20260711/verify_time_cost_model.json \
  --output-dir results/verify_workload_proxy_clean_20260711 \
  --output-tokens 96 --num-samples 2 --seed 20260712 \
  --dist-port-base 35300 --verify-workload-proxy-calibration --no-resume
```

Independent instrumentation-off shadow (exit 1 is the recorded gate failure):

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n nano_moe python \
  scripts/collect_verify_cost_shadow_profiles.py \
  --artifact results/verify_workload_proxy_clean_20260711/verify_time_cost_model.proxy.json \
  --output-dir results/verify_cost_shadow_clean_offset2_20260711 \
  --output-tokens 96 --num-samples 2 --sample-offset 2 \
  --seed 20260715 --dist-port-base 35600 --no-resume
```

Pooled-horizon diagnostic and untouched-shadow replay (the validator exits 1
because the original and replayed predictions both fail the fixed gate):

```bash
conda run -n nano_moe python scripts/analyze_verify_time_cost_model.py \
  --profiles \
    'results/verify_time_cost_calibration_clean_20260711/**/sample*.json' \
    'results/verify_workload_proxy_clean_20260711/**/sample*.json' \
  --output \
    results/verify_time_cost_calibration_combined_20260711/verify_time_cost_model.json \
  --kt-num-threads 16 --kt-backend avx2_bf16

conda run -n nano_moe python scripts/validate_verify_cost_shadow.py \
  --profiles \
    'results/verify_cost_shadow_clean_offset2_20260711/**/sample*.json' \
  --artifact \
    results/verify_workload_proxy_clean_20260711/verify_time_cost_model.proxy.json \
  --replay-base-artifact \
    results/verify_time_cost_calibration_combined_20260711/verify_time_cost_model.json \
  --output \
    results/verify_cost_shadow_clean_offset2_20260711/shadow_validation_combined_base_replay.json
```

Collectors clear inherited verify debug/sync environment variables and record
the exact commands, git diff hash, hardware, model/profile/dataset hashes, and
measurement contract in their manifests.

## Historical Next Work (Superseded)

The next iteration should not tune the stop rule first. It should:

1. Collect balanced base latency calibration across output horizons and cache
   phases, with horizon/cache-state features and an untouched prompt/seed split;
   naive pooling alone did not pass the external replay.
2. Model the exact cache state after scheduled verify prefetch, not only the
   cache snapshot visible during the draft call.
3. Improve the long-bucket error distribution and validate buckets 10/13
   explicitly.
4. Re-run instrumentation-off shadow with the same fixed gates.
5. Only after all gates pass, run paired `active`, static TPOT, and no-stop
   experiments with identical output validation and clustered confidence
   intervals on decode tokens/s.
