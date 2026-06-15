# Cost-Aware Dynamic Draft Length (TPOT Stop Policy)

Status: implemented (gated by the acceptance predictor) · Date: 2026-06-15

## 1. Background

nano-vllm's spec engine drafts up to `max_draft_tokens` tokens per speculative step,
verifies them in one pass, and keeps the accepted prefix plus one bonus token. The
optimal draft length is workload-dependent: too short wastes verify capacity, too long
spends draft compute on tokens that will be rejected.

The on-GPU **acceptance predictor**
(`docs/acceptance_predictor_draft_integration.md`) makes a per-draft-token theoretical
acceptance estimate `alpha = Σ_v min(p_target, q_draft)` available inside
`speculative_step`. The first use of that signal (commit `39fcaae`) was a deliberately
crude rule: stop drafting once **every** sequence's `alpha` falls below a fixed
`draft_alpha_stop_threshold` (0.85). That rule ignores the actual economics of
speculative decoding:

- it doesn't weigh draft cost against verify cost, and
- it treats each token independently, whereas acceptance is **cumulative** — a draft
  token only lands if every earlier token in the run was also accepted.

This document describes the replacement: a cost-aware policy that stops drafting at the
length that minimizes expected per-output-token latency (TPOT).

## 2. Goal

Pick the draft length `k` that minimizes expected TPOT, online, per speculative step,
using the predictor's per-step alphas — with negligible overhead and **zero change to
generated tokens** (speculative decoding stays exact; only throughput changes).

Non-goals for this version: estimating the *next* token's alpha before drafting it
(predictive lookahead), and learning `td`/`tv` online (fixed config values for now).

## 3. Design

### 3.1 Cost model

Let `td` = time of one batched draft step, `tv` = time of one batched verify. For a draft
of length `k` with per-token acceptance estimates `[e₁…eₖ]`, the expected number of
accepted draft tokens for a single sequence is the cumulative-product sum:

```
acc_len(k) = Σ_{i=1..k} Π_{j=1..i} e_j
```

(token `i` is accepted only if tokens `1..i` were all accepted). Each spec iteration
produces `acc_len + 1` output tokens (the `+1` is the guaranteed verify/bonus token) at a
cost of `k·td + tv`. Aggregated over the batch of sequences `s`:

```
T(k) = (k·td + tv) / Σ_s (acc_len_s(k) + 1)
```

`T(0) = tv / N` is the no-draft baseline (one bonus token per sequence, `N` sequences).

### 3.2 Why a simple "stop when it stops decreasing" rule works

`T(k)` is **unimodal** in `k`: the numerator grows linearly by `td` each step, while the
denominator grows with diminishing returns (each added term `Π e_j` shrinks since
`e_j ≤ 1`). So `T` decreases, reaches a minimum, then increases — the optimal `k*` is the
first point where it stops decreasing, i.e. where predicted tokens/s peaks.

### 3.3 Policy: reactive lookback

The predictor returns alpha for the token *just* drafted, so the engine evaluates the
rule **after** each draft step (reactive lookback):

```
draft step k → recompute T(k) → if T(k) > T(k-1): stop, else continue
```

This drafts exactly one "extra" step past the minimum before detecting the upturn. That
overshoot is harmless for correctness (extra draft tokens are simply rejected at verify)
and is the accepted tradeoff for v1; a predictive variant that estimates `e_{k+1}` to
avoid the wasted step is possible future work.

The policy is a **no-op** whenever alpha is unavailable (predictor disabled) or doesn't
cover the whole batch — in those cases the loop runs the full `max_draft_tokens` budget,
identical to prior behavior. A minimum of one draft step is naturally guaranteed (`T(0)`
is the baseline; the first iteration always runs before any check).

### 3.4 `td` / `tv` source

Fixed config values for v1 (`draft_tpot_td_ms=19`, `draft_tpot_tv_ms=80`; the suggested
starting point). They can be tuned from a profile run: `get_profile()` already reports
`draft_forward_ms` and `verify_forward_ms`. A measured-EMA mode is deferred.

## 4. Implementation

### 4.1 Configuration — `nanovllm/config.py`

| field | default | meaning |
|---|---|---|
| `draft_stop_policy` | `"tpot"` | `none` / `alpha_threshold` / `tpot` |
| `draft_alpha_stop_threshold` | `0.85` | cutoff for the legacy `alpha_threshold` policy |
| `draft_tpot_td_ms` | `19.0` | fixed per-draft-step time used by `tpot` |
| `draft_tpot_tv_ms` | `80.0` | fixed per-verify time used by `tpot` |

`__post_init__` asserts `draft_stop_policy ∈ {none, alpha_threshold, tpot}`. Since the
policy is a no-op without the predictor (off by default), defaulting to `tpot` only
affects predictor-enabled runs.

### 4.2 Engine — `nanovllm/engine/speculative/spec_engine.py`

- **`expected_tpot_ms(step_alphas, num_seqs, td_ms, tv_ms)`** — pure module-level
  function implementing the §3.1 cost model (`step_alphas` is a list over draft steps of
  per-sequence alpha lists). Used by both the engine and the unit test so the math has a
  single source of truth.
- **`speculative_step`** — initializes `tpot_prev = expected_tpot_ms([], N, td, tv)`
  (= `T(0)`), accumulates `tpot_alpha_history` each draft step, recomputes `T(k)`, and
  breaks once `T(k) > T(k-1)`. The legacy `alpha_threshold` branch is preserved under the
  policy switch.
- **Observability** — `step_trace["draft_steps_actual"]` (chosen `k`),
  `step_trace["draft_stop_policy"]`, `step_trace["draft_tpot"]` (the `T(k)` curve), and
  `_profile["draft_tpot_early_stop_count"]`.

### 4.3 Benchmark / CLI plumbing

`--draft-stop-policy`, `--draft-tpot-td-ms`, `--draft-tpot-tv-ms` are threaded through
`scripts/bench_acceptance_predictor.py` and
`benchmarks/scripts/spec_verify_expert_count_stats.py` (alongside the existing predictor
args), so policies can be swept and compared.

### 4.4 Why generated tokens are unchanged

Speculative decoding is exact: the accepted prefix plus bonus token equal what the target
model would have produced, independent of how many tokens were drafted. Drafting fewer
tokens only reduces tokens-emitted-per-step (throughput), never *which* tokens are
emitted. Hence the on-vs-off `outputs_digest` must still match.

## 5. Tests

### 5.1 Unit test (CPU, no model) — `tests/test_draft_tpot_stop.py`

Validates `expected_tpot_ms` against an independent brute-force `acc_len` reference and
checks the reactive-stop decision against a brute-force argmin of `T(k)`:

- constant-high alpha → drafts the full budget (never stops early);
- decaying alpha → stops at `argmin(T) + 1` (the reactive overshoot);
- very-low alpha → stops after one step (drafting is worse than verify-only);
- cheap `td` → extends draft length even at modest alpha.

```bash
python -m pytest tests/test_draft_tpot_stop.py -v
# or
python -m unittest tests.test_draft_tpot_stop -v
```

(The test logic is CPU-only; it imports `torch` transitively via the engine module, so
run it in the project env.)

### 5.2 Real-model integration test (GPU + model, env-gated)

`tests/test_acceptance_predictor_integration.py` drives the bench with the predictor
on/off and asserts alpha presence/range, alpha in step traces, **digest match**
(predictor must not change outputs), and reports overhead.

```bash
NANOVLLM_RUN_ACCEPTANCE_PREDICTOR_TESTS=1 \
NANOVLLM_REAL_MODEL_PATH=/data1/models/Qwen3-30B-A3B \
NANOVLLM_ACCEPTANCE_PREDICTOR_PATH=random_cache_srdp_scripts-1/res/run_20260614_133025 \
python -m pytest tests/test_acceptance_predictor_integration.py -v -s
```

## 6. Benchmark & comparison commands

### 6.1 Predictor on vs off, TPOT policy active

```bash
conda activate nano_moe
cd /home/linke/nano-vllm-moe
rm -rf results/acc_predictor_bench
python scripts/bench_acceptance_predictor.py \
    --output-dir results/acc_predictor_bench \
    --acceptance-predictor-path random_cache_srdp_scripts-1/res/run_20260614_133025 \
    --gpu-memory-utilization 0.99 \
    --cache-ratios 0.3125 \
    --output-lens 512,4096 \
    --max-draft-tokens-values 6 \
    --segment-sizes 12 \
    --predictor-modes on,off \
    --draft-stop-policy tpot \
    --draft-tpot-td-ms 19 \
    --draft-tpot-tv-ms 80 \
    --kt-num-threads 32
```

Outputs `summary.json` / `summary.md`. Confirm `digest == match` (TPOT stop must not
change outputs), and read `throughput_overhead_pct`, `draft_ms_delta`,
`predicted_alpha_avg`, and the measured acceptance rate from the on/off comparison.

### 6.2 Policy ablation (none vs alpha_threshold vs tpot)

Run the same command three times with `--predictor-modes on` and one policy each, into
separate output dirs, then compare:

```bash
for POL in none alpha_threshold tpot; do
  python scripts/bench_acceptance_predictor.py \
      --output-dir results/acc_pred_${POL} \
      --acceptance-predictor-path random_cache_srdp_scripts-1/res/run_20260614_133025 \
      --gpu-memory-utilization 0.99 \
      --cache-ratios 0.3125 \
      --output-lens 512,4096 \
      --max-draft-tokens-values 6 \
      --segment-sizes 12 \
      --predictor-modes on \
      --draft-stop-policy ${POL} \
      --draft-alpha-stop-threshold 0.85 \
      --draft-tpot-td-ms 19 --draft-tpot-tv-ms 80 \
      --kt-num-threads 32
done
```

Compare across the three runs:

- **end-to-end throughput / TPOT** — `tpot` should match or beat both baselines;
- **`draft_steps_actual` distribution** (from `spec_step_traces`) — average chosen `k`;
- **accepted/drafted ratio** — `accepted_tokens_total / draft_tokens_total`;
- **`draft_tpot_early_stop_count`** — should be `> 0` for the `tpot` policy.

### 6.3 Calibrating `td` / `tv`

Read `draft_forward_ms` and `verify_forward_ms` from a run's engine profile
(`get_profile()` / the per-case `predON_*.json`). If they differ materially from the
defaults (19 / 80 ms), re-run §6.1 with the measured values via `--draft-tpot-td-ms` /
`--draft-tpot-tv-ms`.

## 7. Files changed

- `nanovllm/config.py` — `draft_stop_policy`, `draft_tpot_td_ms`, `draft_tpot_tv_ms` +
  validating assert.
- `nanovllm/engine/speculative/spec_engine.py` — `expected_tpot_ms`, the reactive TPOT
  stop in `speculative_step`, trace/profile fields.
- `scripts/bench_acceptance_predictor.py`,
  `benchmarks/scripts/spec_verify_expert_count_stats.py` — new CLI args + Config wiring.
- `tests/test_draft_tpot_stop.py` — **new** unit test.
- `docs/acceptance_predictor_draft_integration.md` — config table + §9.1 cross-reference.
