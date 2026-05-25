# Expert Rerouting Evaluation Suite

Standalone experiments that evaluate the five draft-phase expert rerouting
algorithms (DPERP framework) without depending on nano-vllm-moe or any full
inference system.  Inspired by the `on_device_sd` repository's approach of
running modified-routing forward passes and comparing output logits to an
exact baseline.

---

## Files

| File | Purpose |
|------|---------|
| `expert_rerouting_eval.py` | Main experiment: calibration, evaluation, plotting |
| `test_rerouting.py` | Unit tests for all five algorithms (toy MoE, no GPU required) |

---

## Requirements

```bash
pip install torch transformers datasets matplotlib numpy tqdm
```

Tested with: Python 3.10+, PyTorch ≥ 2.0, Transformers ≥ 4.40.

---

## Quick start

```bash
# Recommended model (same as on_device_sd):
python expert_rerouting_eval.py \
    --model /path/to/Qwen1.5-MoE-A2.7B \
    --cache_ratios 0.125 0.25 0.375 0.5 0.625 0.75 \
    --n_calib 128 --n_eval 128 --seq_len 256 \
    --outdir ./results

# Low-memory sanity check (CPU, shorter sequences):
python expert_rerouting_eval.py \
    --model /path/to/Qwen1.5-MoE-A2.7B \
    --device cpu --n_calib 32 --n_eval 32 --seq_len 64 \
    --outdir ./results_cpu

# Run unit tests (no model or GPU needed):
python test_rerouting.py -v
```

---

## Experiment Design

### Why independent of the full system?

The DPERP objective — maximising accepted draft tokens per second — depends on
**draft output quality** (which algorithms can be measured without PCIe or CPU
fallback) and **system timing** (which requires a full heterogeneous runtime).
Decoupling them lets us iterate on algorithm quality quickly.

The key insight from `on_device_sd`: comparing draft logits to baseline logits
gives the **theoretical token-level acceptance rate**

$$\alpha = \sum_v \min\!\left(p_\text{draft}(v),\; p_\text{target}(v)\right)
         = 1 - \mathrm{TV}(p_\text{draft},\, p_\text{target})$$

without needing to actually run speculative decoding.  This is both faster and
less noisy than empirical sampling.

---

### Phase 1 — Offline Calibration

Runs the model with hooks on every expert and gate layer over `n_calib` chunks.
Builds:

| Artefact | Shape | Used by |
|----------|-------|---------|
| `similarity_table` S | [L, N, N] float16 | Alg1 LUT, Alg3 joint score, Alg5 prior |
| `error_table` D | [L, N, N] float32 | Alg4 budget |
| `sensitivity` ω | [L] float32 | Alg1 adaptive thresholds, Alg4 budget |
| `activation_freq` | [L, N] float32 | Cache warm-start (LFU) |

The similarity S[l, i, j] is the cosine similarity between the **mean expert
outputs** on the calibration set.  For two experts in the same functional group,
this is typically 0.7–0.95; for unrelated experts, 0.1–0.4.

**Output plots:** `similarity_heatmap.png`, `layer_sensitivity.png`

---

### Phase 2 — Algorithm Evaluation

For each `(algorithm, cache_ratio)` pair:

1. Build `SimulatedCache` by caching the `⌊N × ratio⌋` highest-frequency experts
   per layer (LFU warm-start, matching what a real runtime would converge to).
2. Wrap each MoE layer with the rerouting wrapper.
3. For each eval chunk, run:
   - **Baseline forward** (exact routing, all experts available) → logits_baseline
   - **Draft forward** (rerouting applied) → logits_draft
4. Compute α per chunk and average.
5. Restore original modules between chunks.

Algorithm 5 (Bandit) additionally calls `bandit_update()` after each chunk,
simulating online learning across the eval set.

---

### The Five Algorithms

| Algorithm | Timing | Signal | Weight handling |
|-----------|--------|--------|-----------------|
| **Alg1 Skip+SimLUT** | post-routing | offline S, w_e | renormalize |
| **Alg2 EntropyBias** | pre-routing | τ (entropy), cached_mask | original softmax |
| **Alg3 RouterMerge** | post-routing | online z_j + offline S | sim-scaled merge |
| **Alg4 ErrorBudget** | post-routing + terminate | offline D, ω | skip / min-error sub |
| **Alg5 Bandit** | post-routing | EMA α̂ (online) | via Alg3 |

**Two baselines** are included for reference:
- `SkipAll`: zero weight for all miss experts (hardest baseline)
- `RoundRobin`: cycle through cached experts naively

**Expected ordering at cache_ratio = 0.25:**
```
SkipAll < RoundRobin < Alg1 ≈ Alg4 < Alg2 < Alg3 < Alg5 (converged)
```

---

### Phase 3 — Outputs

| File | Contents |
|------|---------|
| `results_summary.csv` | α, std, PPL, PPL gap per (algorithm, cache_ratio) |
| `results_full.json` | Full metrics including per-layer cosine similarity |
| `alpha_vs_cache_ratio.png` | Main result: α curves for all algorithms |
| `ppl_gap_vs_cache_ratio.png` | PPL degradation |
| `layer_cos_sim_cache{r}.png` | Per-layer output similarity at each cache ratio |
| `similarity_heatmap.png` | Expert similarity matrix (sample layers) |
| `layer_sensitivity.png` | Layer sensitivity profile ω |

---

## Key Metrics Explained

**`mean_alpha`** — Theoretical token acceptance rate. The primary metric.
Values closer to 1.0 mean the draft distribution closely matches the target.
α ≥ 0.6 is generally required for positive speculative decoding speedup at
draft depth K=4.

**`ppl_gap`** — Difference in perplexity between rerouted and exact model.
Lower is better; negative values indicate the rerouted model is actually better
calibrated (shouldn't happen for high miss rates).

**`layer_cos_sim`** — Per-layer cosine similarity of MoE block outputs between
draft and exact runs. Low similarity in early layers compounds through the
network, explaining why layer sensitivity ω matters.

---

## Algorithm 5 Convergence Note

The bandit converges over eval chunks in sequence.  With `n_eval=128` chunks of
`seq_len=256`, each (layer, expert_miss, expert_sub) triple sees roughly:

```
128 * 256 * 7_misses / (16_experts * 16_subs) ≈ 89 observations
```

This is sufficient for UCB to select near-optimally.  Plot `mean_alpha` over
chunks (from `results_full.json`) to see the convergence curve.

---

## Extending to Qwen3-30B-A3B

The code is fully generic.  For the 30B model (N=128, k=8):

```bash
python expert_rerouting_eval.py \
    --model /path/to/Qwen3-30B-A3B \
    --cache_ratios 0.0625 0.125 0.25 0.375 0.5 \
    --n_calib 64 --n_eval 64 --seq_len 128 \
    --dtype bfloat16 \
    --outdir ./results_30b
```

Note: building the similarity table requires running all 128 experts per layer,
which is memory-intensive.  Reduce `n_calib` if needed; 32 chunks is sufficient
for a good similarity estimate.

---

## Relationship to nano-vllm-moe

The experiment uses **none** of nano-vllm-moe's infrastructure.  Integration
points when you are ready to test in the full system:

- `build_moe_execution_plan()` in `placement.py` — replace `_build_topc0_substitution_lut`
  with Alg1/3/5 LUT
- `LayerExpertCache.publish_ready_staging_to_active()` — trigger LUT rebuild
- `SpeculativeEngine.accept_phase()` — trigger Alg5 bandit updates
- `ModelRuntimeMetaRecorder` — export full router logits for Alg3 (currently
  exports only top-k)
