# On-GPU Acceptance Predictor in the Draft Segment-Graph Path

Status: implemented (gated, off by default) · Date: 2026-06-14

## 1. Background

`random_cache_srdp_scripts-1/` trains a lightweight branch-encoder MLP
(`AcceptancePredictor`, ~72k params) that predicts the **theoretical acceptance
rate** of a speculative draft token:

```
alpha = sum_v min(p_target(v), q_draft(v))   in [0, 1]
```

i.e. the probability that a draft token sampled from the (cheap, expert-substituted)
draft distribution `q` is accepted when verified against the (full) target
distribution `p`. The predictor consumes cheap, on-the-fly signals available during
draft decode: routing disturbance between original vs draft-modified expert
selection, query-logit statistics, the final hidden state, and a per-sequence decode
history.

The goal of this integration is to run that predictor **inside nano-vllm's draft
path** so the spec engine can use predicted alpha for **adaptive draft length**
(stop drafting once predicted tokens/s peaks), under three hard constraints:

1. on-GPU,
2. without breaking the draft CUDA graphs,
3. without adding CPU↔GPU transfers beyond what already exists.

The predictor was trained for a Qwen3-MoE-30B-A3B-shaped model
(`num_layers=48`, `top_k=8`, `hidden=2048`; see
`random_cache_srdp_scripts-1/res/run_20260614_133025/config.json`).

## 2. Goals & non-goals

- **Goal**: produce a calibrated-shape alpha per draft sub-step, available to the
  spec engine, with negligible latency overhead and zero change to generated tokens.
- **Goal**: keep everything gated; when disabled the draft path is byte-for-byte
  unchanged.
- **Adaptive-draft-length stop policy** (toks/s from alpha): implemented as the
  cost-aware `tpot` policy — see §9.1.
- **Known caveat**: the predictor was trained on the SRDP `random_cache` replacement
  mechanism, which is *not identical* to nano-vllm's reroute/cache-substitution
  draft. Predictions are only as good as that distributional match — calibrate
  before trusting alpha for control (the bench reports `alpha-accept delta`).

## 3. Predictor recap

`AcceptancePredictor` (architecture copied verbatim into
`nanovllm/engine/speculative/acceptance_predictor.py`): five LayerNorm→Linear→SiLU
branch encoders, concatenated to 128, then `128→64→32→1` head with Sigmoid. Each
branch self-normalizes via LayerNorm — **no external normalization needed**.

| feature | dim | meaning |
|---|---|---|
| `route_raw` | 768 | `concat(orig_w[48,8], mod_w[48,8])` flattened — original vs draft-modified top-k softmax weights |
| `route_summary` | 45 | per-layer L1/L2/rep_frac/rep_mass/entropy/top1/margin/max_drop → mean/max/std (30) + early/mid/late block aggregates (15) |
| `token_features` | 10 | top-k(32) draft-logit stats: top1, top2, margin, top5_mean, top1_gap5, std, top1_prob, entropy, eff_vocab, k |
| `hidden` | 2048 | final post-norm hidden state of the draft token |
| `history` | 11 | per-sequence cumulative/EMA(0.3)/max of RSD, replacement-mass, query-entropy + normalized step position + `prefill_len/8096` |

Output: `alpha in (0,1)`. The feature math is reproduced exactly from
`random_cache_srdp_scripts-1/build_random_cache_dataset.py` as pure tensor ops
(`route_features`, `token_features`, history recurrence).

## 4. Semantic mapping SRDP → nano-vllm

| SRDP (training) | nano-vllm draft |
|---|---|
| "standard mode" target `p` / `original_*` | `selected_experts` / `routing_weights` (`qwen3_moe.py` MoE block) |
| "random_cache mode" draft `q` / `modified_*` | `execution_experts` / `execution_weights` (after `draft_reroute_policy`) |
| `replacement_mask` | `execution_experts != selected_experts` per slot |

> The reroute policy must be active (e.g. `entropy_cache_bias`, the default) for the
> disturbance signal to be non-trivial; with no reroute, modified == original and the
> route features degrade to zero disturbance.

## 5. Design

### 5.1 What runs where

```
draft step (per token, per sequence batch):
  [host]  write per-seq history STATE into a static input buffer  (tiny H2D)
  [GPU, captured segment graphs]  transformer layers
        └─ each MoE block writes original+modified routing into [L,max_bs,K] buffers
  [GPU, eager]  LM head (compute_logits)            ← already eager today
  [GPU, eager]  token_features = topk(logits,32)+stats → static buffer (no D2H)
  [GPU, captured TAIL graph]  route_raw/route_summary + history recurrence
                              + predictor MLP → alpha_buf + new history STATE
  [GPU, eager]  sampler(logits).tolist()            ← the one existing D2H sync
  [host]  read alpha + new history STATE (one post-sync D2H), advance host state
```

Three reasons for this split:

- **LM head stays eager** to avoid a `[max_bs, vocab]` (~150 MB) per-bucket logits
  buffer and padding waste; logits already live on GPU.
- **token_features eager but on-GPU** (`topk`+reductions) → staged into a static
  buffer; no CPU↔GPU transfer.
- **The launch-bound part is captured** (the ~14-kernel predictor MLP + the
  route/history reductions), which is exactly where CUDA-graph capture pays off.

### 5.2 History state carry

`history` is a per-sequence recurrence (EMA/cum/max across decode steps). The
recurrence math runs **on-GPU inside the tail graph**, but the *state* is carried
across steps **on the host, keyed by `seq_id`** so it survives batch re-ordering. It
is fed in via a tiny static buffer (`state_in_buf`, H2D) and read back fused with
alpha in a single post-sync D2H. The host-keyed design avoids the correctness bug a
batch-row-indexed GPU-persistent state would have.

- `history[0]` uses a fixed `acceptance_predictor_step_horizon` (live decode length is
  unknown) — the one feature that differs from training-time semantics.
- Rejected speculative steps are not rolled back in v1 (documented approximation;
  history is smoothed).

### 5.3 alpha output

After the tail graph, alpha + new state live in static buffers. They are read right
after the sampler's existing `.tolist()` sync (`model_runner.run`), so the data is
already on-device → the extra `.cpu()` is a near-free copy, **no new sync**. alpha is
threaded back through `run_draft`'s return dict and surfaced by the spec engine.

## 6. Architecture / data structures

### `nanovllm/engine/speculative/acceptance_predictor.py`

- `AcceptancePredictor(nn.Module)` — verbatim architecture (state-dict compatible).
- `PredictorMeta` — dims parsed from `config.json`.
- `load_acceptance_predictor(path, device)` — builds, casts to **fp32 before**
  `load_state_dict` (the runner's default dtype is bf16), `.eval()`, freezes grads.
- `DraftAcceptanceFeatureExtractor` — owns persistent GPU buffers and graph-safe ops:

| buffer | shape | dtype | filled by |
|---|---|---|---|
| `orig_w`, `mod_w` | `[L, max_bs, K]` | fp32 | `record_layer` (inside segment graphs) |
| `orig_ids`, `mod_ids` | `[L, max_bs, K]` | int64 | `record_layer` |
| `token_features_buf` | `[max_bs, 10]` | fp32 | `set_token_features_from_logits` (eager) |
| `state_in_buf` | `[max_bs, 11]` | fp32 | `write_state_in` (H2D from host dict) |
| `state_out_buf` | `[max_bs, 11]` | fp32 | tail graph (`run_predictor`) |
| `alpha_buf` | `[max_bs]` | fp32 | tail graph (`run_predictor`) |
| `_host_state` | dict `seq_id → [11]` | np.fp32 | host carry across steps |

Key methods: `attach(model)`, `record_layer(...)`, `set_token_features_from_logits`,
`run_predictor(bs, hidden)` (the captured tail body), `write_state_in(seqs)`,
`read_outputs(seqs)`, `forget(seq_ids)`.

## 7. Call chain

### Init (`model_runner.ModelRunner.__init__`)
```
load model → (rank 0 & enabled) load_acceptance_predictor()
           → DraftAcceptanceFeatureExtractor(...)
           → extractor.attach(model)   # set_draft_feature_recorder on every MoE block
```

### Capture (`model_runner._capture_draft_segment_cudagraph`, per bucket)
```
capture segment graphs (record_layer copies captured for free)
→ compute_logits(final_hidden)            # eager warmup
→ set_token_features_from_logits(logits)  # eager warmup
→ run_predictor(bs, final_hidden)         # eager warmup
→ torch.cuda.graph(tail): run_predictor(bs, final_hidden)   # CAPTURE
→ draft_tail_graphs[bucket] = tail
```

### Per draft step
```
spec_engine.speculative_step
└─ model_runner.run_draft(seqs)
   ├─ extractor.write_state_in(seqs)                      # H2D state
   ├─ self.run → run_model → _replay_draft_segment_graph
   │     ├─ replay segment graphs (fill routing buffers)
   │     ├─ logits = compute_logits(outputs)              # eager
   │     ├─ extractor.set_token_features_from_logits(logits)
   │     ├─ draft_tail_graphs[bucket].replay()            # alpha + new state
   │     └─ return logits
   │  └─ sampler(logits).tolist()                          # existing D2H sync
   └─ extractor.read_outputs(seqs)                         # alpha + state (post-sync D2H)
      → prefetch_state["acceptance_alpha"]
spec_engine: collect per-seq alpha → step_trace["predicted_alpha"]
spec_engine: forget_acceptance_state(finished_ids)
```

### Per-layer recording (`qwen3_moe.Qwen3MoeHeterogeneousSparseMoeBlock.forward`, draft branch)
```
router → selected_experts/routing_weights (original)
draft_reroute_policy → execution_experts/execution_weights (modified)
draft_feature_recorder.record_layer(layer_idx, selected, routing_w, execution, execution_w)
```

## 8. Overhead

Per draft step (Qwen3-30B-A3B, bs 1–8), captured tail graph:

- predictor MLP: ~0.1 MFLOP/token, launch-bound → **~2–5 µs in replay** (vs
  ~70–140 µs eager — why it is captured).
- per-layer routing writes: folded into the existing segment graphs → a few µs.
- `route_summary` reductions: captured, a few µs.
- `token_features` `topk(32)` over vocab: the one new eager op, ~20–40 µs, dwarfed by
  the LM-head GEMM next to it.
- history update: ~5–8 element-wise ops, captured, <2 µs.
- alpha + state transfer: **0 new syncs** (rides the sampler sync), tiny bytes.

**Net: <1–3% of a draft forward.** The benchmark's `overhead %` column measures the
actual end-to-end throughput delta (predictor-off baseline).

## 9. Configuration

`nanovllm/config.py`:

| field | default | meaning |
|---|---|---|
| `acceptance_predictor_enabled` | `False` | master gate (requires `inference_mode="spec"`) |
| `acceptance_predictor_path` | `""` | dir with `config.json` + `best_model.pth` |
| `acceptance_predictor_step_horizon` | `32` | denominator for `history[0]` |
| `draft_stop_policy` | `"tpot"` | dynamic draft-length stop: `none` / `alpha_threshold` / `tpot` |
| `draft_alpha_stop_threshold` | `0.85` | legacy `alpha_threshold` policy cutoff |
| `draft_tpot_td_ms` | `19.0` | fixed per-draft-step time used by the `tpot` policy |
| `draft_tpot_tv_ms` | `80.0` | fixed per-verify time used by the `tpot` policy |

The predictor only runs in the **draft segment-graph path** (predictive /
draft_segment_indexed prefetch mode). With other draft paths it is a no-op.

### 9.1 Dynamic draft length (`draft_stop_policy="tpot"`)

`spec_engine.expected_tpot_ms` turns the per-step predicted alphas into the expected
per-output-token latency for a draft of length `k`:

```
acc_len(k) = sum_i prod_{j<=i} e_j          # cumulative acceptance, per sequence
T(k) = (k*td + tv) / sum_seq (acc_len_seq(k) + 1)
```

`T(k)` is unimodal in `k`. The draft loop (`SpeculativeEngine.speculative_step`) uses a
**reactive-lookback** rule: draft step `k`, recompute `T(k)`, and stop once `T(k)`
exceeds `T(k-1)` (i.e. predicted tokens/s has peaked). `td`/`tv` are fixed config values
in v1 (a measured-EMA mode is a possible follow-up). The chosen length is exported as
`step_trace["draft_steps_actual"]` and the latency curve as `step_trace["draft_tpot"]`;
`_profile["draft_tpot_early_stop_count"]` counts triggered stops.

## 10. Files changed

- `nanovllm/config.py` — config fields + asserts.
- `nanovllm/engine/speculative/acceptance_predictor.py` — **new** module.
- `nanovllm/models/qwen3_moe.py` — `set_draft_feature_recorder` (block + model) and
  the `record_layer` call in the MoE draft branch.
- `nanovllm/engine/model_runner.py` — load predictor; capture tail graph; replay it;
  stage/read history; thread alpha out of `run_draft`; `forget_acceptance_state`.
- `nanovllm/engine/speculative/spec_engine.py` — collect alpha into step traces;
  forget finished sequences.
- `benchmarks/scripts/spec_verify_expert_count_stats.py` — predictor CLI args +
  Config wiring; `predicted_alpha_*` aggregation in the acceptance summary.
- `scripts/bench_acceptance_predictor.py` — **new** bench (predictor on vs off).
- `tests/test_acceptance_predictor.py`, `tests/test_acceptance_predictor_integration.py`
  — **new** tests.

## 11. Tests

### Unit tests (real components, no mocks; CUDA if available else CPU)
```
python -m pytest tests/test_acceptance_predictor.py -v
# or
python -m unittest tests.test_acceptance_predictor -v
```
Covers: predictor forward shape/range, real-checkpoint load (skipped if the
checkpoint dir is absent), batched feature parity vs the numpy reference
(`route_raw`/`route_summary`/`token_features`), the history recurrence + host carry +
`forget`, fresh-sequence reset, and `replacement_mask` derivation.

### Integration test (real model + GPU, env-gated)
```
NANOVLLM_RUN_ACCEPTANCE_PREDICTOR_TESTS=1 \
NANOVLLM_REAL_MODEL_PATH=/data1/models/Qwen3-30B-A3B \
NANOVLLM_ACCEPTANCE_PREDICTOR_PATH=random_cache_srdp_scripts-1/res/run_20260614_133025 \
python -m pytest tests/test_acceptance_predictor_integration.py -v -s
```
Runs the bench (predictor on/off) and asserts: alpha present & in `[0,1]` in the
profile, alpha in `spec_step_traces`, **digest match** (predictor must not change
generated tokens), and reports overhead. Optional env: `NANOVLLM_ACC_PRED_PROFILE_ARTIFACT`,
`NANOVLLM_ACC_PRED_OUTPUT_LEN`, `NANOVLLM_ACC_PRED_KT_THREADS`, `NANOVLLM_ACC_PRED_GPU_MEM`.

## 12. Benchmark

```
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
    --kt-num-threads 32
```

Outputs `summary.json` / `summary.md` in the output dir. It emits every field the
dual-queue benchmark emits (for direct comparison) plus the acceptance-predictor
columns. The console prints `predict_alpha_avg` per predictor-on case and an
aggregate `predict_alpha_avg=...` line at the end. The on-vs-off comparison reports
`overhead %`, `draft_ms_delta`, `predict_alpha_avg`, measured acceptance, and
`alpha-accept delta`; `digest` must read `match`.
```
predict_alpha_avg=0.7123 (over 384 draft sub-steps)
```
