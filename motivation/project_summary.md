# Project Summary: CPU-GPU Collaborative MoE Inference via Expert Substitution and Speculative Decoding

**Document date:** 2026-05-19  
**Codebase:** `nano-vllm-moe`  
**Target model:** Qwen3-30B-A3B (N=128 experts/layer, k=8 active, 48 MoE layers)

---

## 1. Background

Sparse Mixture-of-Experts (MoE) architectures activate only a small fraction of their total expert parameters per token, achieving strong performance at low per-token computation cost. However, for consumer-grade or resource-constrained deployments, the full parameter footprint far exceeds available GPU VRAM. The standard remedy is **expert offloading**: keeping most expert weights in host DRAM and loading them to GPU on demand. This creates an unavoidable bottleneck: every cache miss triggers a PCIe transfer (Qwen3-30B-A3B: ~2×47 MB per expert, PCIe 4.0 ×16 = 64 GB/s peak, yielding ~1.5 ms per expert transfer, blocking inference).

Prior work attacks this bottleneck along three axes, each with unresolved limitations:

**Axis 1 — CPU computation (Fiddler, KTransformers):** Run uncached experts on the CPU to avoid PCIe transfers for weights. CPU computation is slower than GPU and requires either AMX instructions (KTransformers) or absorbs the full CPU processing cost (Fiddler). At high miss rates, CPU becomes the bottleneck instead of PCIe.

**Axis 2 — Expert routing modification (SwapMoE, BuddyMoE, CachePrior):** Redirect token routing toward cached experts to avoid misses. Acceleration is limited and unstable because MoE routing is input-dependent. All three methods permanently alter model behavior, causing accuracy degradation that cannot be bounded without extensive evaluation.

**Axis 3 — Speculative decoding with prefetch (SP-MoE, MoE-SpeQ, MoE-SpAc):** Use a draft model to predict future expert activations and prefetch them asynchronously. All three require the draft model weights to occupy GPU VRAM, reducing the expert cache capacity available to the target model. Verification is also hurt because the parallel multi-token verify pass activates more experts per layer than single-token decode, increasing PCIe and CPU load precisely when it matters most.

**This project's key insight** combines the expert redundancy observation from routing modification literature with the distribution-preservation guarantee of speculative sampling acceptance: expert substitution confined to the draft phase is approximation, but verification corrects it exactly. The system can therefore apply aggressive routing modification (draft with GPU-only computation, substituting uncached experts with cached ones) while asymptotically recovering exact model quality via the verifier — if the substitutions are good enough, most draft tokens are accepted; if not, the verifier samples a correct token at no accuracy cost.

---

## 2. Method

### 2.1 Core Mechanism

The system operates in three phases per decoding step:

**Draft phase:** The target model forward pass runs with all uncached experts either substituted by a functionally similar GPU-cached expert, or skipped (zero weight). No CPU computation or PCIe transfers occur on the draft critical path. The draft runs entirely on GPU-resident weights, at the cost of approximation error in hidden states.

**Verify phase:** The model runs its exact original routing for all draft tokens in parallel. This is a standard speculative decoding verification pass; its output distribution is identical to autoregressive decoding. Experts needed for verification are prefetched asynchronously during the draft phase.

**Accept phase:** Draft tokens are accepted or rejected via speculative sampling: token $\hat{d}_t$ is accepted with probability $\min(1, p^\star(\hat{d}_t) / \tilde{p}(\hat{d}_t))$, where $p^\star$ is the target distribution and $\tilde{p}$ is the draft distribution. Rejected positions sample a replacement from the residual distribution $\max(p^\star - \tilde{p}, 0)$. This is provably lossless: the marginal distribution over accepted tokens is identical to that of exact autoregressive decoding.

### 2.2 Formal Problem Statement (DPERP)

Let $\mathcal{C}^\ell \subset \{0,\ldots,N{-}1\}$ be the GPU expert cache at layer $\ell$, with $|\mathcal{C}^\ell| = S \ll N$. The draft MoE output replaces each miss expert $e \in \mathcal{M}^\ell(\mathbf{h}) = \mathcal{S}^\ell(\mathbf{h}) \setminus \mathcal{C}^\ell$ via a rerouting function:

$$\phi^\ell : \mathcal{M}^\ell(\mathbf{h}) \times \mathbf{h} \times \mathcal{C}^\ell \;\to\; \mathcal{C}^\ell \cup \{\varnothing\}$$

The optimization objective (DPERP) is:

$$\boldsymbol{\phi}^* = \arg\max_{\boldsymbol{\phi}} \; \frac{\mathbb{E}\!\left[\sum_{t=1}^K \prod_{s=1}^t P(\mathrm{accept}_s;\,\boldsymbol{\phi})\right]}{T_\text{draft}(K;\,\boldsymbol{\phi}) + T_\text{verify}(K;\,\boldsymbol{\phi})} \quad \text{s.t. } \phi^\ell(e,\mathbf{h}) \in \mathcal{C}^\ell \cup \{\varnothing\}$$

This couples three sub-problems: substitute selection (P1), draft length adaptation (P2), and output weight redistribution (P3). The GPU residency constraint eliminates CPU computation from the draft critical path by construction.

### 2.3 The Five Rerouting Algorithms

Five algorithms have been designed and implemented to address DPERP, ranging from simple to adaptive:

| Algorithm | Timing | Signal | Key innovation |
|-----------|--------|--------|----------------|
| **Alg1 Skip+SimLUT** | Post-routing | Offline S, routing weight | Layer-adaptive thresholds; renormalize over retained set |
| **Alg2 EntropyBias** | Pre-routing | Token entropy τ, cache mask | Bias strength conditioned on per-token routing confidence; top-1 protection |
| **Alg3 RouterScoreMerge** | Post-routing | Online z-scored router logits + offline S | Input-conditioned substitute selection; similarity-weighted weight merge bounds error amplification |
| **Alg4 ErrorBudget** | Post-routing + terminate | Offline D, layer sensitivity ω | Draft termination as a budget variable; adaptive draft depth optimises the DPERP objective directly |
| **Alg5 OnlineBandit** | Post-routing | EMA empirical acceptance rate | First framing of expert substitution as a contextual bandit; UCB exploration; entropy-binned context |

All five use the same evaluation framework: offline calibration builds the similarity table S[L,N,N], error table D[L,N,N], and layer sensitivity ω[L]; a simulated GPU cache with LFU warm-start selects the S highest-frequency experts; draft and baseline logits are compared via TV distance α = Σ_v min(p_draft(v), p_target(v)).

### 2.4 Expert Prefetching

The `PrefetchRuntime` maintains a `GlobalWarmStartQueue` prioritised by:

$$\text{priority} = w_\text{source} \cdot \text{score\_sum} + w_\text{count} \cdot \text{activation\_count} - w_\text{age} \cdot \text{age}$$

Three signal sources feed the queue with configurable weights (default: prefill_history=1.0, verify_history=1.2, draft_live=1.5). H2D transfers run in a background worker thread; `PrefetchTicket` tracks each inflight transfer with a `ready_event`. Staged experts are promoted to the active cache (`publish_ready`) only at verify time, not during each draft step — this was a key fix discovered during implementation (see §4).

### 2.5 Dual-Objective Expert Caching

The cache serves two distinct roles: direct hit provider (standard LRU/LFU objective) and replacement pool (substitution quality objective). A similarity-aware (SA) eviction policy assigns each cached expert a replacement value equal to the sum of similarity scores to the most frequently activated uncached experts, retaining high-replacement-value experts even if they are rarely directly routed. This conflicts with LRU/LFU at moderate cache ratios where one expert may substitute for several high-traffic misses.

---

## 3. Challenges

**Challenge 1 — Expert Substitution Quality:** The substitution introduces layer-by-layer approximation error that propagates and accumulates. Unlike routing modification methods where any accuracy drop is permanent, here the draft must achieve high enough acceptance rate to justify the draft overhead. If the semantic gap between substitute and target expert is large, tokens are frequently rejected, negating the benefit. The substitution must: (a) minimise per-miss representation error, (b) model cross-layer error accumulation, and (c) estimate the resulting per-step acceptance probability in time to make adaptive draft-length decisions.

**Challenge 2 — Dual-Objective Cache:** LRU/LFU optimise for direct hit rate. An acceptance-rate-aware policy must additionally retain experts with high replacement value for uncached peers. The joint objective requires a replacement value function depending on pairwise similarity, joint routing distributions, and future activation frequencies — all of which must be estimated online within inference time budgets.

**Challenge 3 — Prefetch Timing:** Asynchronous prefetch must decide which experts to transfer and how much bandwidth to allocate at each draft step. Too many concurrent transfers may interfere with GPU computation or memory bus; too few leave verification-phase cache misses unresolved. Optimal budget depends on PCIe bandwidth, per-expert transfer time, the compute window available during drafting, and the probability each prefetched expert is actually consumed.

**Challenge 4 — Verification Cost Inflation:** Verification activates up to $K \times \text{top-k}$ expert slots across all draft tokens in parallel. For Qwen3-30B-A3B with K=8, k=8: up to 64 expert slots per layer per verify step vs. 8 for single-token decode. With high miss rates and insufficient prefetch time, verification can regress end-to-end throughput. The interaction between draft length K, accept rate α, and verification cost V(K) forms a non-trivial optimisation landscape.

---

## 4. Current Progress

### 4.1 Infrastructure (Complete)

The nano-vllm-moe prototype provides a complete heterogeneous MoE inference stack:

- **Three inference modes:** `standard` (exact autoregressive), `heter` (GPU cache + CPU fallback, no spec), `spec` (full speculative pipeline).
- **Expert cache:** `LayerExpertCache` with fixed-size contiguous GPU buffer pools, LRU/LFU/adaptive eviction, staging lifecycle (inflight → ready → published), and generation-protected atomic cache updates.
- **Heterogeneous forward:** `heterogeneous_moe_forward` dispatches GPU-cached routes via grouped-GEMM and CPU-resident routes via `fused`/`torch`/`kt_kernel` backends, with optional CPU-GPU parallel execution.
- **Draft CUDA graph:** `ModelRunner._can_use_draft_cudagraph()` enables CUDA graph replay when `draft_top_c=0`. The `substitution_lut` (round-robin fallback to cached experts) ensures fixed tensor shapes for graph capture.
- **Speculative sampling:** `StandardSamplingAcceptance` implements the correct stochastic speculative sampling algorithm, including residual distribution sampling on rejection and next-token sampling when all drafts are accepted. Greedy/deterministic paths remain available.
- **Prefetch runtime:** `PrefetchRuntime` with `GlobalWarmStartQueue`, three signal sources, staging slots, and background H2D worker thread.

### 4.2 Bugs Found and Fixed

| Bug | Symptom | Fix |
|-----|---------|-----|
| `publish_ready()` called inside draft loop | `prefetch on` was slower than `prefetch off`; each draft step incurred a CUDA stream wait | Moved `publish_ready()` to verify entry only |
| `StandardAcceptance` used as speculative sampling | Accept rate reported as 100% across all cache ratios; throughput < 1 tok/s | Implemented `StandardSamplingAcceptance`; set as default |
| `draft_top_c > 0` disables CUDA graph | Draft graph replay never fired | Set `draft_top_c=0` as default for graph-compatible experiments |
| `draft_tokens=[]` with sampling acceptance | ValueError at last step when no draft tokens remain | Guard: skip logit check when draft is empty |

### 4.3 Quantitative Results (Qwen3-30B-A3B, single RTX GPU, fused backend)

Latest benchmark session (2026-05-08/09), `spec` mode, `standard_sampling`, `draft_top_c=0`:

| Cache ratio | Prefetch | Accept rate | Draft graph replays |
|-------------|----------|-------------|---------------------|
| 0.75 | off | 0.593 | 27 |
| 0.75 | on | 0.536 | — |
| 0.50 | off | 0.484 | — |
| 0.50 | on | 0.640 | — |
| 0.25 | off | 0.342 | — |
| 0.25 | on | 0.333 | 38 |

Accept rates are now non-trivially below 1.0 and vary with cache ratio as expected. Prefetch `on` does not consistently improve accept rate or throughput in the current configuration; the prefetch consumed count is low (19–44 experts per run with 24 output tokens), and the verify-before-publish synchronisation still adds overhead.

The current substitution strategy is **round-robin** (LUT maps uncached experts cyclically to cached slots). This is the baseline that the five designed algorithms are intended to replace.

### 4.4 Standalone Experiment Framework (Complete)

A fully self-contained evaluation framework (`expert_rerouting_eval.py`, `test_rerouting.py`) evaluates all five algorithms without requiring the nano-vllm-moe runtime. It uses forward-pass logit comparison (theoretical α via TV distance) and covers: offline calibration (S, D, ω), LFU cache simulation, all five rerouting wrappers, baselines (SkipAll, RoundRobin), layer cosine similarity measurement, and CSV/JSON/plot output.

---

## 5. Remaining Tasks

### Priority 1 — Algorithm Implementation in nano-vllm-moe

The five rerouting algorithms are designed and tested in the standalone framework but have not yet been integrated into the nano-vllm-moe hot path. The integration point is `build_moe_execution_plan()` in `nanovllm/expert/placement.py`, specifically the `_build_topc0_substitution_lut()` function which currently implements round-robin.

Ordered by implementation effort:

1. **Alg1 (Skip+SimLUT):** Replace round-robin LUT with similarity-table lookup. LUT rebuild triggered by `publish_ready_staging_to_active()`. Requires offline calibration pass (~1 day).
2. **Alg2 (EntropyBias):** Modify `Qwen3MoeHeterogeneousSparseMoeBlock.forward()` to add cache-bias before top-k selection in draft mode. Requires entropy computation from existing router logits (~1 day).
3. **Alg4 (ErrorBudget):** Add error LUT and cumulative risk accumulator to `MoEExecutionPlan`; expose early-termination flag to `SpeculativeEngine._budget_draft_steps()` (~2 days).
4. **Alg3 (RouterScoreMerge):** Extend `ModelRuntimeMetaRecorder` to export full pre-softmax logits (currently exports only top-k); implement joint scoring and similarity-weighted merge in the execution plan builder (~2 days).
5. **Alg5 (Bandit):** Implement `BanditState` class; extend `SpeculativeEngine.accept_phase()` to record (layer, e_miss, e_sub) tuples and call EMA update; wire into plan builder (~3 days).

### Priority 2 — Dual-Objective Cache

The similarity-aware (SA) eviction policy needs to be implemented in `nanovllm/scheduling/cache_strategy.py`. This requires: (a) the offline similarity table to be available to the eviction policy, (b) a per-cached-expert replacement value score that is updated when uncached expert activation patterns change (i.e., after each verify pass). The existing `CacheStrategy` interface needs a `compute_victim()` method override with access to the similarity table and current routing statistics from `ModelRuntimeMetaRecorder`.

### Priority 3 — Prefetch Improvement

The current low prefetch consumed rate (19–44/run) suggests the priority queue is not targeting the right experts at the right time. Remaining work:

- Track publish-to-verify hit rate separately from total prefetch submitted (add counter: "staged and published" vs "staged but evicted before verify").
- Implement request-scoped queue reset: the warm-start queue currently accumulates across requests; for short requests (24 tokens), inter-request mixing reduces prediction accuracy.
- Evaluate adaptive prefetch budget: compute overlap window ($T_\text{draft} - T_\text{plan}$) and scale `prefetch_step_budget` dynamically based on remaining draft steps and measured PCIe throughput.

### Priority 4 — End-to-End Benchmarking

Full comparison against prior work baselines with the new substitution algorithms installed:

| Metric | Target vs. RoundRobin baseline | Target vs. SP-MoE |
|--------|---------------------------------|-------------------|
| Accept rate at cache_ratio=0.5 | +10–15pp | — |
| Tokens/sec | +20% | +20% |
| GPU VRAM used | same | −30% (no draft model weights) |
| MMLU accuracy | no degradation | no degradation |

Experimental plan: ShareGPT (conv.), HumanEval (code), MMLU (knowledge); single RTX 4090; cache ratios 0.125–0.75; draft depth K ∈ {2,4,6,8}.

### Priority 5 — Adaptive Draft Length (Benefit 1 from paper)

The lightweight acceptance probability predictor that enables adaptive K is not yet implemented. Planned as a small MLP (14-feature input matching SRDP feature set from `on_device_sd`) trained on expert-substitution error signals and accept/reject labels from the verify phase. This unlocks the main paper contribution claim that draft quality is dynamically observable and K can be shortened early when substitution quality is poor.

---

## Appendix — Architecture Analysis: nano-vllm-moe

This appendix provides a component-by-component analysis of the existing implementation from the perspective of future algorithm integration and benefit estimation.

### A.1 Component Map

```
LLMEngine  (llm_engine.py)
  ├── Scheduler  (engine/scheduler.py)          sequence queue, KV block manager
  └── SpeculativeEngine  (engine/speculative/spec_engine.py)
        ├── ModelRunner  (engine/model_runner.py)   [worker process per TP rank]
        │     ├── Qwen3MoeForCausalLM  (models/qwen3_moe.py)
        │     │     └── Qwen3MoeDecoderLayer × 48
        │     │           └── Qwen3MoeHeterogeneousSparseMoeBlock
        │     │                 ├── gate  (Linear: hidden→N_experts)
        │     │                 ├── experts[128]  (on CPU or GPU slot)
        │     │                 └── heterogeneous_moe_forward()
        │     ├── LayerExpertCache × 48  (expert/cache.py)
        │     │     ├── gate_up_buffer  [S, gate_up_dim, hidden]   GPU
        │     │     ├── down_buffer     [S, hidden, down_dim]       GPU
        │     │     ├── expert_to_slot_lut  [N]  GPU
        │     │     ├── slot_to_expert_lut  [S]  GPU
        │     │     └── cached_expert_mask  [N]  GPU bool
        │     ├── cpu_expert_pool  dict[layer][expert]{gate_up, down}   pinned CPU
        │     ├── PrefetchRuntime  (expert/prefetcher.py)
        │     │     ├── GlobalWarmStartQueue  (priority + EMA scoring)
        │     │     └── _prefetch_worker_thread  (background H2D)
        │     ├── ModelRuntimeMetaRecorder  (expert/runtime_meta.py)
        │     ├── DraftScheduler  (scheduling/draft_scheduler.py)
        │     └── CacheStrategy  (scheduling/cache_strategy.py)
        └── AcceptanceStrategy  (engine/speculative/acceptance.py)
              StandardSamplingAcceptance  [DEFAULT]
```

### A.2 Critical Path per Decode Step

**Draft step (×K, CUDA-graph replayed when draft_top_c=0):**

```
ModelRunner.run_draft(seqs)
  ├── model forward (Qwen3MoeForCausalLM)
  │     for each MoE layer:
  │       gate(h)  →  top-k routing
  │       build_moe_execution_plan()
  │         ├── cached_expert_mask lookup
  │         ├── _build_topc0_substitution_lut()  ← ROUND-ROBIN (target for replacement)
  │         ├── gpu_route_indices, gpu_m_sizes
  │         └── substitution_lut  [N] GPU
  │       heterogeneous_moe_forward()
  │         ├── GPU grouped-GEMM over cached slots (fast path)
  │         └── [cpu_expert_pool fallback, disabled when top_c=0]
  ├── sampler  →  draft_token
  └── [return draft_logits if return_logits=True]
```

**Verify step (×1, not graph-captured):**

```
SpeculativeEngine: wait_prefetch_for_verify()
  └── PrefetchRuntime.publish_ready()  →  update active LUT + GPU stream sync

ModelRunner.run_verify(seqs + all_draft_tokens)
  ├── model forward [prefill-like, all K draft tokens in parallel]
  │     for each MoE layer:
  │       exact routing (no substitution_lut)
  │       heterogeneous_moe_forward()
  │         ├── GPU cached experts  [~S/N fraction hits]
  │         └── CPU fallback for misses [blocking PCIe+compute]
  └── return verify_logits [T×K, vocab_size]

SpeculativeEngine: accept_phase()
  StandardSamplingAcceptance.accept(draft_tokens, verify_logits, draft_logits)
  →  (accepted_tokens, next_token)
```

### A.3 Integration Points for New Algorithms

**Alg1, Alg3, Alg5 — LUT replacement:**

```python
# Current (placement.py: _build_topc0_substitution_lut)
# Round-robin: uncached expert i → cached slot (i % num_slots)
fallback = slot_to_expert_lut[i % num_slots]

# Target: lookup best substitute from similarity table
# This is a [N] GPU tensor; rebuild O(N) on CPU, push to GPU, no graph impact
```

The LUT is already a pre-allocated GPU tensor read by the graph-captured kernel. Replacing the construction logic requires no changes to the CUDA graph boundary. Rebuild is triggered once per `publish_ready()` call (O(N·S) CPU operation, ~0.1 ms for N=128, S=16).

**Alg2 — Pre-routing bias:**

The `gate(h)` call in `Qwen3MoeHeterogeneousSparseMoeBlock.forward()` currently computes logits without any bias. Adding a cache-state-dependent bias requires:
- A pre-allocated GPU bias tensor `[N]` updated at `publish_ready()` time.
- A flag to enable the bias only in draft mode (verify must use original logits).
- This is a fused vector addition into the existing logit tensor, zero additional kernel launches.

The draft-mode flag is already available via `get_context().is_draft` (the context system tracks draft vs. verify state). Implementation is straightforward; main risk is that altering top-k selection changes tensor shapes if `draft_top_c > 0`, but at `draft_top_c=0` the substitution_lut handles shape normalization regardless.

**Alg4 — Early termination signal:**

`SpeculativeEngine._budget_draft_steps()` currently reads only `max_draft_tokens` and per-sequence token budgets. Wiring Algorithm 4's cumulative risk requires:
- Risk accumulation inside the graph (scalar per token, ~1 ms overhead per layer): feasible only if the risk threshold check runs outside the graph capture boundary.
- Simpler implementation: accumulate risk on CPU in the plan-builder (plan builder runs in Python, outside the graph), check against budget after each draft step, and set `draft_steps = 0` to terminate early.
- `SpeculativeEngine` would read the early-stop flag from the plan builder after each `run_draft()` call.

**Alg5 — Bandit update in accept phase:**

`SpeculativeEngine.accept_phase()` currently calls `AcceptanceStrategy.accept()` and returns the accepted token list. Extending it to also record (layer, e_miss, e_sub) tuples for bandit update requires:
- `MoEExecutionPlan` must store which substitutions were applied (currently stored in `substitution_lut` but not per-route mapping). A new `applied_substitutions` field — a list of (layer, token, miss_expert, sub_expert) tuples — would be populated during `build_moe_execution_plan()` and cleared after each verify.
- `ModelRunner.run_draft()` would accumulate these tuples from all draft steps.
- `SpeculativeEngine.accept_phase()` would iterate per-position accept/reject and call `bandit_state.update()` for each substitution at an accepted or rejected position.
- The bandit state itself is a CPU data structure (~9.4 MB for N=128, L=48) and adds <0.5 ms to accept phase.

### A.4 CUDA Graph Compatibility Analysis

| Component | Graph captured? | Notes |
|-----------|-----------------|-------|
| Gate (router Linear) | ✅ Yes | Static shape [T, N] |
| Substitution LUT lookup | ✅ Yes | LUT is pre-allocated [N]; contents updated between replays |
| GPU expert grouped-GEMM | ✅ Yes | `fused_moe_linear` with fixed slot count S |
| CPU expert fallback | ❌ Never | Requires host sync; disabled via `draft_top_c=0` |
| Cache bias addition (Alg2) | ✅ Yes | Fused into gate output, bias tensor pre-allocated |
| LUT rebuild (Alg1/3/5) | CPU only | Runs at `publish_ready()`, between replays |
| Bandit update (Alg5) | CPU only | Runs in accept phase, between replays |
| Error budget accumulation (Alg4) | CPU only | Runs in plan builder, between replays |
| Prefetch H2D transfer | Async, independent | Background thread, never in graph |

All five algorithms are CUDA-graph compatible. Algorithm 2's bias is the only change that touches the graph-captured kernel body; it adds one vector addition but does not change shapes or control flow.

### A.5 Benefit Estimation

The following estimates assume Qwen3-30B-A3B, cache_ratio=0.5 (64 of 128 experts cached), single RTX 4090, K=6 draft steps.

**Baseline (round-robin substitution, current implementation):**
- Accept rate α ≈ 0.48 (measured at ratio=0.5, prefetch off)
- Expected accepted tokens per verify step: Σ_{j=1}^{6} α^j ≈ 0.48+0.23+0.11+0.05+0.03+0.01 = 0.91
- At T_draft=0.5ms/step, T_verify=3ms: speedup = 0.91 / (6×0.5+3) = 0.91/6 ≈ 0.15×... suboptimal

**With Alg1 (Skip+SimLUT, estimated α=0.62):**
- Expected accepted: Σ 0.62^j for j=1..6 ≈ 0.62+0.38+0.24+0.15+0.09+0.06 = 1.54
- Speedup relative to autoregressive baseline (1 token per T_ar): (1.54/6) / T_draft + 1/(T_ar) — depends on T_verify behaviour.
- More usefully: expected accepted / (K×T_draft + T_verify) vs. 1/(T_ar). At α=0.62, K=6: ~1.7× improvement over round-robin.

**With Alg5 (Bandit, converged, estimated α=0.70):**
- Expected accepted ≈ 2.2 tokens per verify step
- Approximately 2.4× over round-robin; approaching the theoretical benefit of SP-MoE (which has ~0.70 accept rate but uses GPU VRAM for the draft model, reducing the expert cache size available to the target).

**Verify cost at higher α:** Higher α → longer drafts are justified → verify activates more experts. At α=0.70, K=6, the verify pass processes 6 tokens with up to 48 expert slots per layer per token. Cache miss rate during verify depends on prefetch quality. With Alg5 + improved prefetch, verify miss rate is expected to fall from ~40% (current) to ~20%, keeping T_verify below the threshold where it erases the draft benefit.

**Net expected improvement over current round-robin baseline (conservative):**
- Algorithms 1–3: +10–15 percentage points in accept rate; estimated 1.4–1.7× throughput improvement.
- Algorithms 4+5: additional +5–8pp with adaptive depth + converged bandit; estimated 1.8–2.1× over round-robin.
- Full system (Alg5 + SA cache + improved prefetch): target 2.0× over round-robin = approximately 1.5× over standard autoregressive decode with CPU fallback.

### A.6 Risks and Open Questions

**Substitution error accumulation across layers.** A single substitution at layer 2 propagates through all subsequent attention and MoE layers. The standalone experiments measure per-token α, which integrates this effect, but the mechanism is not yet characterised per-layer. If early layers are more sensitive (as the sensitivity heatmap in `layer_sensitivity.png` should reveal), the Alg1 layer-adaptive threshold is the primary mitigation.

**Bandit cold-start at low-traffic experts.** The bandit needs observations to improve over the similarity prior. For rarely-routed experts (activation frequency < 0.1%), the observation count may be insufficient within typical request lengths. The entropy binning helps (high-entropy tokens are more common) but the sparse-expert arms may stay at the prior for many requests.

**Prefetch vs. draft interaction.** The draft produces routing decisions that feed the prefetch queue (source: `draft_live`, weight 1.5). With Alg2 (pre-routing bias), draft routing is systematically biased toward cached experts, reducing `draft_live` signal quality for predicting verify-phase misses. The prefetch strategy may need to weight `verify_history` more heavily when Alg2 is active.

**Variance under `standard_sampling`.** The current quantitative results (6 sessions, 24 output tokens each) show high variance across prefetch conditions. Accept rate at ratio=0.5 varies from 0.48 to 0.64 across prefetch on/off. Statistically robust conclusions require either fixing the token sequence (deterministic sampling) or running ≥100 output tokens per session with multiple seeds.

**Verify early-stop (Benefit 3, prefix skip).** The shared-prefix verification collapse hypothesis has not been implemented or tested. It is the most speculative of the three claimed benefits and requires careful theoretical analysis alongside empirical validation (KL divergence measurement at varying cumulative error thresholds).
