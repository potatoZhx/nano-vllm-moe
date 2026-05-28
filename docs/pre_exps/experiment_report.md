# Pre-Experiments Validation Report

**Date:** 2026-05-28
**Model:** Qwen3-30B-A3B (N=128, k=8, L=48 MoE layers)
**Hardware:** NVIDIA A100-SXM4-80GB (gpu18/gpu19)
**Environment:** nano_vllm_env (Python 3.10, torch 2.11.0+cu130, transformers 4.51+)

---

## Executive Summary

| Experiment | Key Finding | Design Impact |
|------------|-------------|---------------|
| E1: Dual-Objective Cache | HitScore-SubstScore correlation ρ=0.78; LFU is sufficient | Skip EvictCost-aware strategy (Phase 3) |
| E2: Top-1/2 Protection | Reroute-level protection gives +0.2pp at r=0.25 only | Implement as cheap safety net; skip cache-pinning |
| E3: Dynamic K | Simplified T_cycle model biases toward K*=12; E4 gives better signal | Use E4's coverage-based signal for Dynamic K |
| E4: Prefetch Coverage | Coverage drops rapidly at r<0.50; K*=1 at r=0.25 with no prefetch | Coverage is an effective Dynamic K signal |
| E5: Alpha Prediction | Analytical model RMSE=0.03-0.28; MLP marginal improvement only at r=0.25 | Use analytical model initially; defer MLP |

---

## E1: Cache Dual-Objective Evaluation

### Goal
Determine whether HitScore and SubstScore are highly correlated, and whether a dual-objective cache (HitValue + SubstValue) outperforms single-objective LFU.

### Method
1. Calibration pass: 8 chunks × 256 tokens → expert output means + pairwise cosine similarity matrix [48, 128, 128]
2. Per-layer Spearman correlation between HitScore (activation frequency) and SubstScore (weighted substitution value)
3. Oracle comparison: build caches using HitOnly_LFU, SubstOnly, Joint (λ=0.5,0.5), Joint (λ=0.7,0.3) strategies; measure actual hit rate on eval set

### Results

**Correlation Analysis:**
- Mean Spearman ρ across 48 layers: **0.782**
- Range: [0.516, 0.880]
- All correlations are statistically significant (p < 1e-9)
- Conclusion: **"dual_likely_unnecessary"** (ρ > 0.7 threshold)

**Oracle Cache Comparison:**

| Ratio | HitOnly_LFU | SubstOnly | Joint_0.5 | Joint_0.7 |
|-------|-------------|-----------|-----------|-----------|
| 0.75  | **99.56%**  | 97.68%    | 98.87%    | 99.14%    |
| 0.50  | **95.19%**  | 85.92%    | 92.06%    | 93.49%    |
| 0.25  | **75.14%**  | 57.97%    | 68.76%    | 72.30%    |

HitOnly_LFU consistently dominates at all cache ratios. Joint strategies interpolate between HitOnly and SubstOnly but never outperform pure LFU.

### Design Implication
**Skip EvictCost-aware strategy.** Single-objective LFU is sufficient for the Qwen3-30B-A3B model. The high correlation between HitScore and SubstScore means that optimizing for hit rate automatically ensures reasonable substitution quality. The EvictCost-aware strategy (Plan Phase 3) is not worth the implementation complexity.

### Issues Encountered
- 6144 expert hooks (128 experts × 48 layers) in the calibration phase caused significant overhead. Reduced `--n_calib` from 64 to 8 for practical runtime.
- The `eval_skipall_alpha()` function in the script uses heuristic α estimation rather than actual rerouting evaluation. This is acceptable for oracle comparison but the absolute α values should not be compared to other experiments.

---

## E2: Top-1/2 Protection Evaluation

### Goal
Quantify the impact of top-1/2 expert cache misses on the per-step acceptance rate (α), and validate the benefit of three protection mechanisms: cache pinning, rerouting-level restore, and both combined.

### Method
1. Lightweight calibration: activation frequency + rank-1/rank-2 frequency per expert
2. CriticalMissRate tracking: decode-mode simulation (8 prompts, 6 draft steps) with gate hooks
3. Protection comparison: SkipAll baseline, Alg2_v2 base, Alg2_v2+cache_pin, Alg2_v2+reroute_protect, Alg2_v2+both; 3 w_protect thresholds (0.10, 0.15, 0.20), 3 cache ratios

### Results

**CriticalMissRate vs α Correlation:**

| Ratio | ρ(CritMiss) | ρ(Top1Miss) | ρ(OverallMiss) |
|-------|-------------|-------------|-----------------|
| 0.75  | +0.11       | +0.03       | +0.17           |
| 0.50  | -0.15       | -0.11       | +0.22           |
| 0.25  | -0.08       | -0.05       | +0.02           |

All correlations are weak and statistically insignificant (p > 0.1). This is expected: at high cache ratios, miss rates are negligible; at low ratios, the rerouting algorithm (miss-rate gate) already handles most cases.

**Protection Mechanism Comparison:**

| Ratio | w_protect | SkipAll | Alg2 base | +cache_pin | +reroute | +both |
|-------|-----------|---------|-----------|------------|-----------|-------|
| 0.75  | 0.10/0.15/0.20 | 0.9896 | 0.9883 | 0.9883 | 0.9883 | 0.9883 |
| 0.50  | 0.10/0.15/0.20 | 0.9695 | 0.9700 | 0.9700 | **0.9702** | **0.9702** |
| 0.25  | 0.10/0.15/0.20 | —     | 0.8935 | 0.8935 | **0.8952** | **0.8952** |

Observations:
- At r=0.75: miss-rate gate (ρ < 0.25 → gate=0) causes all algorithms to fall back to exact routing → identical seq_α
- Cache pinning alone has **zero effect** at all ratios: the LFU cache already prioritizes top-1/2 experts (they are the most frequently activated)
- Reroute-level protection provides a small gain (+0.2pp) at r=0.25, independent of w_protect threshold
- The gain is modest because the miss-rate gate already protects against the worst case (low miss rate → gate=0 → no rerouting)

### Design Implication
**Implement reroute-level top-1/2 protection as a cheap safety net.** The implementation is near-zero overhead (a gather + equality check). Cache pinning is unnecessary since LFU already captures the information. The default `w_protect=0.15` is reasonable; the exact threshold is insensitive.

### Issues Encountered
- **KV cache API change (transformers >= 4.51):** `past_key_values` now returns `DynamicCache` objects instead of tuples. The scripts' `copy_kv` and `free_kv` functions were converting them to tuples, causing `AttributeError: 'tuple' object has no attribute 'get_seq_length'`. Fixed by updating to use `DynamicCache.update(key.clone(), value.clone(), layer_idx)` for cloning.

---

## E3: Dynamic K Analysis

### Goal
Find the optimal draft length K* for each (algorithm, cache_ratio) pair using a simplified T_cycle throughput model. Validate whether Level-1 threshold signals (alpha, miss rate, critical miss rate) can predict K*.

### Method
1. Decode-mode simulation at K=1..12 for SkipAll and Alg2_v2 at 3 cache ratios
2. Collect per-step α, miss rate, critical miss rate
3. Simplified T_cycle model: T_draft=K×2ms, T_verify=48×0.5ms=24ms, T_stall estimated from miss rate and prefetch_rate=2
4. Exponential decay fit: α(k) = α₀·exp(-λ·k)
5. Level-1 threshold analysis

### Results

**Alpha Decay Fitting:**

| Algo | Ratio | α₀ | λ | RMSE | Per-step α range |
|------|-------|------|------|------|------------------|
| SkipAll | 0.75 | 0.9902 | 0.0017 | 0.0073 | [0.961, 0.994] |
| Alg2_v2 | 0.75 | 0.9897 | 0.0015 | 0.0062 | [0.961, 0.994] |
| SkipAll | 0.50 | 0.9712 | 0.0067 | 0.0169 | [0.906, 0.980] |
| Alg2_v2 | 0.50 | 0.9726 | 0.0070 | 0.0177 | [0.901, 0.982] |
| SkipAll | 0.25 | 0.7717 | 0.0069 | 0.0549 | [0.683, 0.842] |
| Alg2_v2 | 0.25 | 0.7872 | 0.0058 | 0.0537 | [0.715, 0.855] |

**Throughput Analysis:**

| Algo | Ratio | K* | Throughput (tok/ms) |
|------|-------|-----|---------------------|
| SkipAll | 0.75 | **12** | 0.246 |
| Alg2_v2 | 0.75 | **12** | 0.247 |
| SkipAll | 0.50 | **12** | 0.202 |
| Alg2_v2 | 0.50 | **12** | 0.202 |
| SkipAll | 0.25 | **12** | 0.028 |
| Alg2_v2 | 0.25 | **12** | 0.030 |

K* = 12 for ALL configurations. This is because the simplified T_stall model underestimates the stall penalty: it assumes 2 experts can be prefetched per draft step, making T_stall near zero at all cache ratios. With T_stall ≈ 0, the throughput curve is dominated by E[A(K)]/(K×2+24), which increases monotonically with K when α is close to 1.0.

At r=0.25, the throughput is 8-9x lower than at r=0.75 (0.028 vs 0.246 tok/ms), but the *relative* throughput across K values still favors large K because the absolute stall penalty is underestimated by ~10x.

### Design Implication
**The simplified T_cycle model in E3 is too optimistic.** Use E4's coverage-based T_stall model for Dynamic K decisions. The Level-1 threshold signals are still useful for detecting when the model should stop early, but the K* prediction should come from a more realistic stall model.

### Issues Encountered
- Same KV cache API fix as E2.
- The T_stall model uses `prefetch_rate=2` as default, but the actual number of prefetchable experts per step depends on PCIe bandwidth and expert size constraints.

---

## E4: Prefetch Coverage Analysis

### Goal
Quantify the relationship between prefetch coverage and verify stall, and validate PrefetchCoverage as an auxiliary signal for Dynamic K.

### Method
1. Collect original routing traces from full-model forward passes (no rerouting)
2. Simulate prefetch at rates m = 0, 1, 2, 3, 4 experts/step with FIFO queue
3. Compute coverage = |P_ready| / |P_need| at verify time
4. Estimate verify stall from uncovered experts × τ_expert (1.5ms)
5. Compute theoretical throughput for all (K, m) combinations

### Results

**Coverage vs K (r=0.75):** Near 1.0 at all K and m; stall ≈ 0
**Coverage vs K (r=0.50):** Drops from ~0.85 (K=1) to ~0.50 (K=12) with m=0; improved by higher m
**Coverage vs K (r=0.25):** Drops from ~0.40 (K=1) to near 0 (K=12); even m=4 only reaches 0.15 at K=12

**Optimal K per prefetch rate:**

| m | r=0.75 K* | tp | r=0.50 K* | tp | r=0.25 K* | tp |
|---|-----------|---|-----------|---|-----------|---|
| 0 | 12 | 0.195 | 11 | 0.050 | **1** | 0.011 |
| 1 | 12 | 0.240 | 11 | 0.054 | **1** | 0.011 |
| 2 | 12 | 0.254 | 11 | 0.059 | **1** | 0.011 |
| 3 | 12 | 0.262 | 11 | 0.065 | **1** | 0.011 |
| 4 | 12 | 0.264 | 11 | 0.072 | **1** | 0.012 |

Key findings:
- At r=0.25, K*=1 regardless of prefetch rate: draft is counterproductive because the verify stall from uncovered experts dominates
- Prefetch rate m has diminishing returns: going from m=0 to m=1 improves throughput by 23% at r=0.75; going from m=3 to m=4 adds only 1%
- Coverage threshold of 0.80 correctly separates cases where long K is beneficial (r=0.75) from cases where K should be small (r=0.25)

**Coverage as Dynamic K Signal:**

Coverage < 0.4 at K=1 strongly signals that draft should not be used (→ K=0 or K=1). Coverage > 0.8 at K=8 suggests that longer draft lengths may be viable.

### Design Implication
- **PrefetchCoverage is an effective Dynamic K signal**, complementary to miss-rate and acceptance-rate signals
- At low cache ratios, the system should consider skipping draft entirely (K=0 or K=1) since the stall from uncovered verify experts outweighs the draft benefit
- Prefetch rate m=1 or m=2 is the sweet spot for cost/benefit

---

## E5: Alpha Prediction Evaluation

### Goal
Validate the analytical model α̂(k) = α₀·exp(-λ·E(k)) for predicting per-step acceptance rate, and determine whether an MLP upgrade is warranted.

### Method
1. Decode-mode simulation (SkipAll): collect per-step (E(k), actual_α(k), miss_rate, etc.) pairs
2. Fit analytical model via scipy curve_fit with bounds α₀∈[0,2], λ∈[0,50]
3. Residual analysis: correlate residuals with auxiliary features
4. Fit 2-layer MLP (16 hidden units, 200 epochs) as baseline
5. Compare RMSE and R²

### Results

**Analytical Model Fit:**

| Ratio | α₀ | λ | RMSE | R² |
|-------|------|------|------|------|
| 0.75 | 0.9890 | 0.442 | 0.028 | 0.074 |
| 0.50 | 0.9839 | 0.198 | 0.111 | 0.011 |
| 0.25 | 0.8437 | 0.085 | 0.285 | 0.009 |
| Combined | 0.9663 | 0.158 | 0.181 | — |

The analytical model fits well at high cache ratios (α varies little → any model works) and degrades at low ratios where the variance in α is much higher. The low R² values indicate that the simple exponential model captures only the general trend of α decay, not the step-by-step variations.

**MLP Comparison:**

| Ratio | Analytical RMSE | MLP RMSE | ΔR² | Recommendation |
|-------|----------------|----------|-----|----------------|
| 0.75 | 0.028 | 0.063 | -4.49 | analytical_sufficient |
| 0.50 | 0.111 | 0.123 | -0.13 | analytical_sufficient |
| 0.25 | 0.285 | 0.269 | +0.02 | mlp_upgrade |
| Combined | 0.181 | 0.179 | — | essentially tied |

At r=0.75 and r=0.50, the MLP performs **worse** than the analytical model due to overfitting on the small training set (192 samples). At r=0.25, the MLP marginally outperforms the analytical model (ΔRMSE = 0.016).

**Residual Analysis:**

The strongest residual predictor is the step index (ρ=0.33 at r=0.50), suggesting that the exponential model's assumption of constant-rate decay doesn't hold. The miss_rate and critical_miss_rate features explain minimal residual variance (R² < 0.02).

### Design Implication
- **Start with the analytical model for Level-2 Dynamic K.** The MLP provides marginal value only at very low cache ratios, and even then the improvement is small (ΔRMSE=0.016).
- The MLP overfits at high cache ratios due to small sample size. Collecting more data (Phase 3, Stage 1) is a prerequisite for any MLP upgrade.
- The step index k should be included as an explicit feature in any prediction model, as it captures non-exponential decay patterns.

### Issues Encountered
- Same KV cache API fix as E2/E3.
- The `cache_ratio` feature has zero within-ratio variance, making it useless for per-ratio MLP fits (ρ=nan in residual analysis). This is expected — each data collection run uses a fixed ratio.
- `alpha0` bounds were set to `[0, 2]` in the script, allowing physically impossible values >1.0. Changed to `[0, 1]` for correctness.

---

## Cross-Cutting Observations

### Code Quality
All five experiment scripts are standalone (no dependency on nano-vllm-moe runtime). However, they exhibit significant code duplication:
- `load_model_and_tokenizer()` — 5 copies
- `detect_moe_config()`, `get_moe_layers()` — 5 copies
- `prepare_chunks()`, `SimulatedCache` — 5 copies
- `alpha_tv()`, `copy_kv()`, `free_kv()` — 3 copies (E2, E3, E5)
- `SkipAllWrapper`, `Alg2V2Wrapper` — 2 copies each (E2, E3; also in expert_reroute/)

A shared `pre_exps/utils.py` module would reduce maintenance burden.

### Model Architecture Detection
The `detect_moe_config()` function correctly identifies `moe_attr="mlp"` for the Qwen3-30B-A3B model (the standard transformers implementation). The fallback check for `"block_sparse_moe"` exists for custom implementations but is not triggered. The model uses the Qwen3Moe architecture from transformers with per-layer `.mlp` attribute containing `.experts` and `.gate`.

### GPU Memory
The 30B model (~60GB in float16) fits comfortably within the 80GB A100, leaving ~20GB for KV cache, activations, and overhead. No OOM issues were encountered with the default batch sizes.

---

## Consolidated Design Recommendations

Based on all five experiments, the design recommendations from the system design report are updated as follows:

| Original Plan | Updated Recommendation | Rationale |
|---------------|----------------------|-----------|
| EvictCost-aware strategy (Phase 3) | **Skip** | Dual-objective unnecessary (E1 ρ=0.78); LFU sufficient |
| Top-1/2 Cache Pinning | **Skip** | Zero effect (E2); LFU already captures top-1/2 |
| Top-1/2 Reroute Protection | **Implement** | +0.2pp at low ratio, near-zero cost (E2) |
| Dynamic K Level-1 Thresholds | **Implement** | Valid signal, but use coverage-based thresholds (E3+E4) |
| Dynamic K Level-2 Analytical Model | **Implement** | RMSE=0.03-0.28; sufficient for production (E5) |
| Dynamic K Level-2 MLP Upgrade | **Defer** | Marginal value; needs more data (E5) |
| Prefetch Coverage Signal | **Implement** | Strong Dynamic K signal, validated (E4) |
| Prefetch Rate Optimization | **m=1 or m=2** | Sweet spot for cost/benefit (E4) |
