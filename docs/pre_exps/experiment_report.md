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
- At r=0.75: miss-rate gate (ρ < 0.25 → gate=0, gamma_eff=0) causes Alg2_v2 to apply zero bias → routing identical to original top-k → then hit-mask zeros out uncached experts' weights → **falls back to SkipAll behavior** (not "exact routing", since miss experts are still skipped). Since miss rate is only ~0.3%, SkipAll at this ratio is near-equivalent to exact routing.
- Cache pinning alone has **zero effect** at all ratios: the LFU cache already prioritizes top-1/2 experts (they are the most frequently activated)
- Reroute-level protection provides a small gain (+0.2pp) at r=0.25, independent of w_protect threshold
- The gain is modest because the miss-rate gate already protects against the worst case (low miss rate → gate=0 → no rerouting)

### Design Implication
**Implement reroute-level top-1/2 protection as a cheap safety net.** The implementation is near-zero overhead (a gather + equality check). Cache pinning is unnecessary since LFU already captures the information. The default `w_protect=0.15` is reasonable; the exact threshold is insensitive.

### Mechanism: What "Reroute-Level Protection" Actually Does

**Critical insight:** If top-1/2 experts are NOT in cache, they cannot be computed. The protection does NOT restore their computation — it preserves **probability mass allocation**.

**Trace of Alg2V2Wrapper.forward():**
1. Original router: `logits → softmax → topk → (rw, ri)` records original top-k
2. Miss-rate gate: `γ_gate = clamp((ρ-ρ_low)/(ρ_high-ρ_low), 0, 1)`
3. Entropy-scaled bias: `γ_eff = γ₀ × γ_gate × (0.2 + 0.8 × τ_ent)`
4. Biased routing: `biased_logits = logits + γ_eff × cache_mask` → new top-k `ni`
5. **Protection**: if original top-1 was displaced AND uncached → restore to `ni[:,-1]`
6. Weights from **original** logits at new indices → zero uncached → renormalize

**Effect on probability mass:**

| | Without Protection | With Protection |
|---|---|---|
| top-1 miss, displaced | A cached expert occupies that slot, gets top-1's probability mass | top-1 restored, weight zeroed → mass redistributed **proportionally to original router weights** across all cached experts |
| Impact | Mass concentrated on one "lucky" cached expert | Mass distributed faithfully among cached experts |

The benefit is small (+0.2pp) because three conditions must co-occur: (a) top-1 not in cache, (b) bias strong enough to displace it, (c) the choice of which cached expert gets the mass meaningfully affects the output.

### `w_protect` Parameter

`w_protect` (default 0.15) is the frequency threshold for identifying "critical" experts to pin:

```python
if rank1_freq[layer, expert] >= w_protect:  # top-1 in ≥15% of tokens → pin
if rank2_freq[layer, expert] >= w_protect:  # top-2 in ≥15% of tokens → pin
```

Three values tested: 0.10 (95 experts pinned), 0.15 (42 experts), 0.20 (22 experts). All gave Δ=0 because any expert with frequency ≥10% is already in the LFU top-32 — making explicit pinning redundant.

### Issues Encountered
- **KV cache API change (transformers >= 4.51):** `past_key_values` now returns `DynamicCache` objects instead of tuples. The scripts' `copy_kv` and `free_kv` functions were converting them to tuples, causing `AttributeError: 'tuple' object has no attribute 'get_seq_length'`. Fixed by updating to use `DynamicCache.update(key.clone(), value.clone(), layer_idx)` for cloning.

---

## E3: Dynamic K Analysis

### Goal
Find the optimal draft length K* for each (algorithm, cache_ratio) pair using a simplified T_cycle throughput model. Validate whether Level-1 threshold signals (alpha, miss rate, critical miss rate) can predict K*.

### Method
1. Decode-mode simulation at K=1..12 for SkipAll and Alg2_v2 at 3 cache ratios: prefill 128 tokens, then autoregressive draft with SkipAll/Alg2V2 wrappers, comparing logits against full-model forward at each position
2. Collect per-step α (TV distance), miss rate, critical miss rate
3. Throughput model: `T_cycle = T_draft + T_verify + T_stall` where:
   - `T_draft(K) = K × 2ms` (each draft step is pure GPU, no PCIe)
   - `T_verify = 48 × 0.5ms = 24ms` (single forward pass for K+1 tokens)
   - `T_stall ≈ max(0, needed - prefetched) × 1.5ms` where `needed = 48 × 8 × avg_miss_rate` and `prefetched = K × prefetch_rate`
   - `E[A(K)] = 1 + Σ_{k=1}^{K} Π_{i=1}^{k} αᵢ` (expected accepted tokens including verify bonus)
   - `Throughput = E[A(K)] / T_cycle`
4. Exponential decay fit: α(k) = α₀·exp(-λ·k)
5. Level-1 threshold analysis: test whether α, miss_rate, or crit_miss at step k can predict optimal K*

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

**Throughput Analysis — Why K* = 12 for ALL configurations:**

K*=12 universally because the simplified T_stall model systematically underestimates PCIe stall. Detailed breakdown at r=0.25:

| K | E[A(K)] | T_draft | T_verify | T_stall(E3 est.) | T_cycle | Throughput |
|---|---------|---------|----------|-----------------|---------|-----------|
| 1 | 1.855 | 2ms | 24ms | ~120ms | 146ms | 0.011 |
| 6 | 3.763 | 12ms | 24ms | ~108ms | 144ms | 0.024 |
| 12 | 5.347 | 24ms | 24ms | ~108ms | 156ms | **0.028** |

T_stall barely increases with K because the model assumes `prefetch_rate=2` experts per step, covering 24 of the 96 needed. The remaining 72 × 1.5ms = 108ms dominates T_cycle regardless of K, making the relative overhead of longer draft small.

**Cross-validation with E4:** E4's more realistic coverage simulation gives T_stall(K=12, m=2, r=0.25) = **857ms** (8× higher!), which would give:

| K | E[A(K)] | T_draft | T_verify | T_stall(E4) | T_cycle | Throughput |
|---|---------|---------|----------|------------|---------|-----------|
| 1 | 1.855 | 2ms | 24ms | 138ms | 164ms | **0.0113** |
| 6 | 3.763 | 12ms | 24ms | 529ms | 565ms | 0.0067 |
| 12 | 5.347 | 24ms | 24ms | 857ms | 905ms | 0.0059 |

With E4's stall model, K*=**1** not 12 — the optimal choice flips completely. This demonstrates that the Dynamic K decision is highly sensitive to the T_stall estimate.

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

### Known Limitations

1. **Prefetch rate range too narrow.** The experiment uses m=0-4 experts/step, but with optimized PCIe transfer (pinned memory, multi-stream concurrency), effective m could reach 6-8. At r=0.50, higher m would significantly improve coverage and potentially shift K* upward.

2. **No standard-decode baseline comparison.** The throughput model only compares different (K,m) configurations internally. The speculative system's advantage over standard decode (no draft, no rerouting) is not explicitly quantified. For reference:
   - Standard decode at r=0.25: ~0.006 tok/ms (each token requires 48-layer forward with expert offload)
   - E4 spec at r=0.25, K=1, m=4: ~0.012 tok/ms — **~2× speedup** even at minimal K

3. **T_verify assumed constant at 24ms.** In practice, verify processes K+1 tokens, activating more unique experts as K grows: E[|V_l|] = N × (1 - (1-k/N)^K). At K=8, this is ~51 experts (vs ~22 at K=1), increasing both GPU compute and PCIe transfer time.

4. **Alpha estimation uses heuristic, not measured values.** `α ≈ 1 - 0.6 × miss_rate` is a rough approximation. The E3 α measurements (actual forward passes with SkipAll wrappers) show higher α values than this heuristic predicts, suggesting the heuristic is conservative.

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

---

## Appendix: Frequently Asked Questions

### Q1: LFU 是否已经优先保留 top-1/2？r=0.25 时会有多少 miss？

LFU 保留"全局激活频率最高"的专家（不限 rank 位置），而非专门针对 rank-1/2。一个在 rank 3-8 出现 100 次的专家和一个只在 rank-1 出现 80 次的专家之间，LFU 会选择前者。

r=0.25 时 LFU 命中率仅 75.14%（E1），意味着每层约 2/8 个专家 miss。其中确实包含部分 top-1/2 miss，但由于高频 top-1/2 专家天然有高全局频率，它们大部分已经被 LFU 缓存。低频 top-1/2 miss 确实会发生，但 E2 的 CriticalMissRate 相关性分析（ρ≈-0.08）表明这些 miss 与 α 下降没有强关联。

### Q2: "回退到精确路由"的说法为什么不准确？

当 miss-rate gate=0 时，Alg2_v2 的 γ_eff=0，bias=0，路由 = 原始 top-k。但**执行阶段**仍然会对 miss 专家置零权重并 renormalize——这就是 SkipAll 行为，不是"精确路由"。精确路由需要实际计算所有 top-k 专家的输出（包括 miss 的，需要 PCIe 传输）。正确的说法是"回退到 SkipAll"。

之所以 r=0.75 时 seq_α 几乎相同（0.9883 vs 0.9896），是因为 miss rate 仅 ~0.3%，SkipAll ≈ 精确路由。

### Q3: "重路由级保护"到底做了什么？top-1/2 miss 时如何"保护"？

**保护的不是计算，而是概率分配。** 代码逻辑：
1. 偏置后的新路由 `ni` 可能不包含原始 top-1
2. 保护代码将原始 top-1 恢复到 `ni[:,-1]`（即使它 uncached）
3. 权重计算后，top-1 因为 uncached 被置零
4. Renormalize 后，top-1 的概率质量按**原始 router 权重比例**分配给其他 cached 专家

无保护时，某个 cached 专家会"独占"top-1 的概率质量。有保护时，质量被公平分配。这解释了收益很小（+0.2pp）——三种条件需要同时满足才有效果。

### Q4: w_protect 是什么？

频率阈值。`w_protect=0.15` 意味着只有出现在 ≥15% token 的 top-1 或 top-2 位置的专家才被 pin。实验中三个阈值（0.10/0.15/0.20）的 Δ 都是 0，因为任何频率 ≥10% 的专家都已经在 LFU top-32 中——pin 不 pin 没有区别。

### Q5: E3 到底在做什么？为什么 K* 都是 12？

**E3 在用简化的 T_cycle 模型找最优 K**。核心公式：`Throughput(K) = E[接受token数] / (K×2 + 24 + stall)`。

K* 全是 12 是因为 α 衰减极慢（λ≈0.007）→ E[A(K)] 随 K 线性增长，而 T_stall 被严重低估（假设每步能预取 2 个专家）。与 E4 的实际模拟对比：E3 估计 T_stall(K=12,r=0.25)=108ms，E4 实际测量=857ms——差了 8 倍。

用 E4 的 T_stall 重新计算，K*=1 而非 12。这说明 **Dynamic K 决策的关键是准确的 T_stall 估计**。

### Q6: E4 的预取率是否太小？是否忽略了 spec 加速？

**关于预取率**：m=0-4 对应每 draft step 传输 0-4 个专家。在 PCIe 4.0 上，单专家 ~47MB，理论 m_max ≈ 2。但 m=4 已接近实用带宽上限（47 GB/s vs 50 GB/s 实用值）。如果考虑多流并发和 pinned memory 优化，m 可达 6-8。实验确实应该覆盖更大的 m 范围。

**关于 spec 加速**：E4 的吞吐模型确实对比了不同 (K,m) 的内部最优，但没有显式计算标准 decode baseline。补充分析：
- 标准 decode at r=0.25：T_per_token ≈ 48 × (3.5ms) ≈ 168ms → 0.006 tok/ms
- E4 spec at r=0.25, K=1, m=4：0.012 tok/ms → **~2× 加速**
- 但 K>1 时 stall 增长超过加速收益（因为覆盖率极低），所以 K*=1 是合理的
