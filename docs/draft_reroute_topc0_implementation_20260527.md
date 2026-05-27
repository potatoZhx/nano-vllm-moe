# Draft Expert Reroute (`top_c=0`) Implementation Report

Date: 2026-05-27

## Scope

This change adapts every distinct `top_c=0` algorithm in
`pre_exps/expert_reroute/draft_decode_eval_v2.py` to the nano-vllm-moe draft
path. Policies are optional and are applied only during draft execution. The
existing behavior remains the default.

Implemented configuration:

```python
draft_top_c=0
draft_reroute_policy="round_robin"          # default, existing implementation
draft_reroute_artifact=""                   # required only by similarity_replace
```

Public policy names:

| Runtime name | v2 source name | Purpose |
|---|---|---|
| `round_robin` | existing nano-vllm-moe baseline | Replace a miss through cache-slot round robin. |
| `drop_miss` | `SkipAll` | Remove every miss contribution and renormalize hits. |
| `entropy_cache_bias` | `Alg2_v2` | Pre-route with entropy-scaled cached-expert bias. |
| `bounded_cache_bias` | `HybridCP_v2` | Restrict cache bias to a router candidate pool and enforce deviation guard. |
| `similarity_replace` | `PostSub_v2` | Substitute qualifying misses through offline conditional similarity. |

`Alg2_PostSub` is intentionally not public. In v2, `Alg2_v2` zeros every
remaining miss weight before `PostSub_v2` receives it. A substitution in stage
two therefore has zero added weight, and final fallback/normalization produces
the same output as `Alg2_v2`. Existing v2 results also reported identical
metrics for the two labels. Its runtime name would only duplicate
`entropy_cache_bias`.

## Baseline Behavior

The previous `top_c=0` path is not a no-op. For every draft MoE layer call,
`build_draft_plan_gpu()`:

1. Builds a length-`N` substitution LUT with `torch.arange`, modulo slot
   mapping, an `index_select` of `slot_to_expert_lut`, and `torch.where`.
2. Looks up the effective expert for every selected route.
3. Looks up each effective expert cache slot and builds the grouped GEMM layout.

Thus `round_robin` has `O(N) + O(T*K)` planning work in addition to common
layout and GEMM work. It remains untouched as the default option to avoid
changing existing outputs or timing when reroute is not enabled.

## Runtime Integration

The policy runs in `Qwen3MoeHeterogeneousSparseMoeBlock.forward()` after raw
router metadata is recorded and before the draft plan is built. This ordering
keeps prefetch observations based on the original unbiased router decision.

Non-default policies return fixed-shape `[T, K]` execution expert IDs and
weights whose nonzero expert IDs are already cached. They enter
`build_cached_draft_plan_gpu()`, which directly maps cached expert IDs to slots
and builds the grouped GPU layout. It does not execute the baseline
round-robin LUT stage.

The route count always remains `T*K`. A dropped route is represented by:

- weight `0`;
- a deterministic valid cached fallback expert ID.

This is numerically equivalent to omitting its contribution. The current fused
path still executes a GEMM row for zero-weight routes because weighting occurs
after expert computation; dynamic removal is deliberately not introduced into
the CUDA Graph path.

## Exact Policy Definitions

Notation:

- `p = softmax(logits)`; original top-k IDs/weights are `(ri, rw)`.
- `cm[e]` indicates whether expert `e` is in the current GPU cache.
- output weights are renormalized per token; an all-zero row routes unit weight
  to the lowest cached expert, matching v2's sorted-cache fallback.
- `MISS_GATE=0.25`, `SIM_FLOOR=0.40`, `GAMMA0=4.0`, `J_FACTOR=3`.

### `drop_miss`

For each original route, keep `rw` if `cm[ri]` is true and set it to zero
otherwise. Renormalize nonzero hit weights. Implementation cost beyond common
planning is an indexed cache mask, multiplication, fallback selection across
`S` cache slots, `where`, and normalization.

### `entropy_cache_bias`

For each token:

1. Compute `miss_rate` over the original top-k and
   `gate = clamp((miss_rate - 0.25) / (0.50 - 0.25), 0, 1)`.
2. Compute router entropy over all experts and entropy scaling with v2
   thresholds `log(N)*0.25` and `log(N)*0.75`.
3. Add `GAMMA0 * (0.2 + 0.8*entropy_scale) * gate` to every cached expert
   logit and select new top-k IDs.
4. If original top-1 was a miss and was displaced, restore it in the final
   selected slot.
5. Compute normalized-equivalent weights from the original unbiased router
   probabilities over the newly selected set; zero remaining misses and
   renormalize. Gathering full-router probabilities is equivalent to a second
   selected-logit softmax after the final retained-route normalization, while
   avoiding that extra softmax in the hot path.

Top-1 protection and miss filtering are vectorized; there is no token loop or
host synchronization.

### `bounded_cache_bias`

For each token:

1. Compute the same miss gate and normalized entropy scale.
2. Form the original router top-`J` pool, `J=min(K*3, N)`.
3. Bias only cached experts within that pool, then select new top-k.
4. Sum original raw router probability for selected experts displaced by the
   candidate set. If the sum exceeds `0.20`, revert the whole row.
5. Weight selected routes from unbiased logits, drop residual misses, and
   renormalize.

The deviation guard deliberately uses raw top-k probabilities, not already
renormalized execution weights, to retain v2 semantics.

### `similarity_replace`

This policy requires offline tensors:

- `cond_sim[L,N,N]`: conditional replacement similarity;
- `skip_err[L,N]`: expected output norm proxy.

For each miss in the original top-k:

1. Require token miss-rate ramp
   `gate=clamp((miss_rate - 0.25)/(1 - 0.25), 0, 1)` to be positive.
2. Require its contribution
   `rw[e] * skip_err[e] >= 0.10 * mean_contribution`.
3. Find the best currently cached substitute from `cond_sim[e]`.
4. Require similarity at least `0.40` and reject a substitute already present
   in the original token top-k.
5. Assign weight `rw[e] * similarity * gate`; otherwise drop the miss.

Unlike a cache-revision table design, implementation gathers only the current
`[T,K,N]` candidate slices and masks them with the live cache tensor. No cache
refresh/version interface is introduced, so it has zero refresh overhead and
cannot become stale after prefetch changes.

The reference script now exports runtime artifacts:

```bash
python pre_exps/expert_reroute/draft_decode_eval_v2.py \
  --model /path/to/model --calibration_artifact /path/to/reroute_v2.pt \
  --calibration_only ...
```

## CUDA Graph Compatibility And Expected Overhead

Every public policy used with `draft_top_c=0` preserves fixed route shapes and
uses only tensor operations already compatible with capture: indexing,
`topk`, `scatter_`, comparisons, reductions, `where`, and normalization. No
policy invokes `.item()`, CPU decisions, dynamic route compaction, allocation
of host metadata, or expert cache mutation while replaying.

| Policy | Additional policy work per layer relative to routing | Expected behavior versus baseline planning |
|---|---|---|
| `round_robin` | Existing `O(N)+O(T*K)` LUT/map work | Reference timing. |
| `drop_miss` | `O(T*K)+O(S)` mask/fallback/normalize; direct plan | May be lower than baseline because it avoids length-`N` LUT generation. |
| `entropy_cache_bias` | entropy `O(T*N)`, cached bias/top-k `O(T*N)`; direct plan | Small fixed GPU work; must be measured against `<3 ms` forward budget. |
| `bounded_cache_bias` | entropy plus candidate-pool mask and deviation guard `O(T*N)`; direct plan | More work than entropy policy; measured before recommendation. |
| `similarity_replace` | gathered candidate similarity `O(T*K*N)`; direct plan | Largest additional vector work; no cache refresh cost. |

These are expectations, not performance claims; measured results are recorded
below.

## Debug And Validation Record

1. Source inspection identified that the existing baseline performs
   round-robin substitution, rather than dropping misses.
2. Initial design considered a cache revision/refresh table for conditional
   similarity. It was removed before implementation: live gathered similarity
   selection is capture-safe, exact under cache changes, and avoids refresh
   overhead entirely.
3. While wiring the model, preserving original v2 probabilities exposed that
   the existing in-place top-k normalization would overwrite raw probabilities.
   The implementation preserves raw values only when a policy is active and
   leaves the default path's original in-place operation unchanged.
4. Attempting `pytest tests` collected an existing executable determinism
   script on the non-GPU login node and failed in NCCL initialization. Targeted
   unit tests are used for the CPU validation pass; real CUDA checks run on the
   A100 allocation.
5. The standard `tests/test_*.py` collection reported one existing CPU-only
   environment failure: `test_moe_determinism` dispatches the fused grouped
   GEMM even when its input tensor is on CPU, while that kernel asserts CUDA
   input. It is unrelated to reroute files and is covered by the A100 checks
   below instead.
6. Artifact generation initially failed because current Transformers stores
   Qwen3 MoE experts as packed tensors instead of iterable expert modules.
   The offline exporter now reconstructs selected unweighted expert outputs
   directly from packed weights. Calibration succeeded and exported tensors;
   `--calibration_only` avoids subsequently entering the reference script's
   legacy wrapper evaluation path, which targets a different Transformers API.

7. The first longer `round_robin` run failed in `BlockManager.may_append()`
   after speculative tokens temporarily completed a partial tail block. The
   retained partial block still carried the full-block prefix hash after
   rollback/partial accept, although subsequent append requires an unhashed
   tail. `rollback_draft()` and `accept_draft()` now invalidate a retained
   hashed partial tail and remove its hash mapping.
8. The next `round_robin` run advanced further and failed in verify prefill
   slot construction. Verify consumes the final proposed draft token as an
   input, while the draft loop had reserved KV storage only through the
   preceding input. `SpeculativeEngine` now reserves one final draft KV slot
   before verify. Both defects reproduce with `draft_reroute_policy=round_robin`;
   they are latent speculative KV boundary bugs exposed by a longer workload,
   not regressions caused by an active reroute policy.
9. The attempted formal command initially failed at `Scheduler.schedule()`.
   This was a benchmark contract defect: `--input-len 128` constructed 128
   textual chunks, tokenizing to 619 tokens; with 256 output tokens it exceeded
   `max_model_len=512`. The benchmark now truncates generated prompts using the
   model tokenizer, records `actual_input_tokens`, and reports effective cache
   ratio from `slots_per_layer / num_experts`.
10. The first semantically correct `entropy_cache_bias` screening improved
    acceptance but cost `+4.379 ms` versus baseline. Removing redundant mask
    conversions and normalization work was insufficient. The policy forward
    is now compiled using the repository's
    `torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")` pattern;
    the compiled CUDA Graph path retains the same output digest and acceptance
    count while meeting the forward budget.

## GPU Validation Results (2026-05-28)

Per current validation priority, output alignment with standard decode was not
used as an acceptance condition in this pass. Measurements use Slurm job
`26765` on node `gpu16`, with assigned visible device `CUDA_VISIBLE_DEVICES=0`
(`NVIDIA A100-SXM4-80GB`, idle at the start of each recorded run). Environment
capture is in:

```text
/home/mumura/moe_spec/logs/reroute_job26765_environment_20260528.log
```

Final formal settings:

```text
num_seqs=1, actual_input_tokens=128, output_len=256, max_draft_tokens=8
temperature=0.8, acceptance_strategy=standard_sampling, seed=0
slots_per_layer=32 / num_experts=128 = 0.25, prefetch=false
draft_top_c=0, enforce_eager=false, max_model_len=512
```

| Policy | Draft graph replays | Accepted / drafted | Acceptance rate | Draft forward avg (ms) | Delta vs baseline (ms) | Elapsed (s) |
|---|---:|---:|---:|---:|---:|---:|
| `round_robin` | 271 | 221 / 271 | 0.815498 | 16.006459 | 0.000000 | 37.825906 |
| `entropy_cache_bias` | 246 | 224 / 246 | 0.910569 | 16.259654 | +0.253195 | 34.412221 |

Conclusion:

- `entropy_cache_bias` exceeds the production `round_robin` acceptance rate
  by `+0.095071` on the formal token-correct run.
- Its draft forward increment is `+0.253195 ms`, below the `<3 ms` goal, while
  CUDA Graph replay remains active.
- The candidate also uses 25 fewer draft graph replays and reduces measured
  elapsed generation time by `3.413684 s` in this single formal run.

Formal artifacts:

```text
results/reroute_impl_20260527/job26765_formal_token_exact_ratio25/round_robin_ratio25.json
results/reroute_impl_20260527/job26765_formal_token_exact_ratio25/entropy_cache_bias_ratio25.json
/home/mumura/moe_spec/logs/reroute_job26765_formal_token_exact_round_robin_ratio25_20260528.log
/home/mumura/moe_spec/logs/reroute_job26765_formal_token_exact_entropy_cache_bias_ratio25_20260528.log
```

Exploratory records retained for debugging:

| Run | Result | Purpose |
|---|---|---|
| Pre-compile `entropy_cache_bias`, legacy generated prompt | acceptance `0.279476`, draft `20.388619 ms`, `+4.378508 ms` | Demonstrated acceptance improvement but failed speed budget. |
| Compiled `entropy_cache_bias`, same legacy generated prompt | acceptance `0.279476`, draft `16.226682 ms`, `+0.216571 ms` | Confirmed compilation retains behavior and eliminates policy overhead. |
| `similarity_replace`, smoke calibration artifact | acceptance `0.099757`, draft `19.526485 ms` | Not a viable candidate with the current smoke artifact. |

The exploratory prompts predate the token-length contract fix and are not used
as the formal acceptance claim.

## Regression Verification (2026-05-28)

The KV fixes are covered by tests for rollback of a speculatively sealed tail,
partial accept followed by append, and verify reservation of the final draft
input. Prompt token-length handling has a separate benchmark unit test.

```bash
python -m pytest -q \
  tests/test_block_manager_draft.py \
  tests/test_scheduler_draft_kv.py \
  tests/test_spec_engine_flow.py \
  tests/test_spec_engine_prefetch.py \
  tests/test_draft_reroute.py \
  tests/test_config_prefetch.py \
  tests/test_placement_spec.py \
  tests/test_draft_standard_decode_forward_bench.py \
  tests/test_model_runner_spec_modes.py \
  tests/test_spec_verify_expert_count_stats.py
# 60 passed
```

## Appendix: Designed `top_c>0` Semantics (Not Implemented)

`top_c>0` is deliberately out of scope for this implementation and validation.
The planned semantics are:

1. Start from original router top-k routes and identify misses.
2. On CPU, execute at most `draft_top_c` distinct original missing experts,
   selected by descending total original routing score.
3. Apply the selected `top_c=0` reroute policy only to remaining miss routes.
4. Preserve the original nonzero CPU-route weights; merge GPU-rerouted and CPU
   contributions in deterministic route order.

For CUDA Graph support this requires fixed-capacity CPU route masks and a
captured CPU/GPU bridge, not dynamic host-side selection or route-count
changes. It must be separately implemented and benchmarked before exposure.
