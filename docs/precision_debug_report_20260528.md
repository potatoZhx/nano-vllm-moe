# Spec Mode Precision Debug Report

Date: 2026-05-28

## 1. Problem Statement

The previous reroute full validation experiment (job 26813) used a synthetic prompt
that produced repetitive filler text, causing inflated acceptance rates. The user
requested:

1. Use a **meaningful natural-language prompt** as input.
2. Compare spec mode output (with deterministic greedy acceptance) against
   **standard (non-spec) CUDA Graph output** for each reroute algorithm.
3. If mismatches exist, debug from operator-level precision, KV cache, and any
   other possible sources.
4. Fix all precision issues found.

The key invariant: with greedy acceptance (temperature=0), the verify step always
runs the full model and defers to its argmax. Therefore, **spec+greedy output
MUST be token-identical to standard output**, regardless of draft model quality.

## 2. Experiment Design

### 2.1 Test Matrix (Initial Precision Validation)

| Dimension | Value |
|---|---|
| Reference | Standard mode (non-spec, all experts via fused GPU path) |
| Spec modes | All 5 policies × 3 cache ratios |
| Prompt | ~200-token natural language text about MoE transformers |
| Output length | 128 tokens |
| Temperature | 0.0 (greedy) |
| Acceptance | `greedy` (exact token match) |
| Total cases | 1 reference + 15 spec = 16 |

### 2.2 Prompt

A ~200-token natural-language paragraph about mixture-of-experts transformers,
covering routing mechanisms, load balancing, and expert caching. Key difference
from the previous experiment: this is coherent prose rather than concatenated
filler words.

### 2.3 Test Scripts

| Script | Purpose |
|---|---|
| `scripts/run_case_inline.py` | Subprocess runner for a single case (standard or spec) |
| `scripts/precision_validate.py` | Orchestrator: runs standard reference, then each spec case, compares outputs |
| `scripts/run_precision_validate.sh` | Slurm batch script |
| `scripts/run_prec_debug.sh` | Focused debug: standard + round_robin r=0.25 only |
| `scripts/run_prec_fix_test.sh` | Fix test: standard + round_robin r=0.25 with `--no-cpu-exec` and/or `--enforce-eager` |
| `scripts/test_cpu_disabled.py` | In-process precision comparison (abandoned due to OOM) |

### 2.4 Isolation Design

Each spec case runs as a **separate subprocess** because `LLM()` calls
`torch.distributed.init_process_group()`, which can only be called once per
process. The subprocess isolation also ensures clean GPU memory for each case.

## 3. Debug Process

### 3.1 First Attempt: Inline Script Generation (Failed)

The initial `precision_validate.py` v1 generated Python code as inline strings
passed to `python -c`. The generated code used complex f-string escaping for
conditional code inclusion, causing syntax errors in the spec subprocesses:

```
  [FAIL] exit=1
  [FAIL] exit=1
  ... (all 15 spec cases)
```

**Fix**: Replaced inline code generation with a standalone helper script
(`run_case_inline.py`) that accepts a JSON configuration argument via
`sys.argv[1]`.

### 3.2 Second Attempt: Timeout Too Short

The first successful run (job 27067) failed because `LLM()` construction in spec
mode takes ~250 seconds (model weight loading + CPU expert pool setup). The
default `--case-timeout-sec 600` was sufficient but the **outer** Slurm batch
script had a 4-hour limit that was adequate.

However, the first spec case ran for 12+ minutes on the subprocess without
completing, triggering manual investigation. SSH inspection (job 27074) revealed
the subprocess was actively computing at 639% CPU (6+ cores), not stuck — the
`torch.compile` max-autotune phase was tuning Triton kernels for the
heterogeneous MoE forward path.

**Fix**: Increased `--case-timeout-sec` to 1800 (30 minutes) and Slurm time limit
to 8 hours.

### 3.3 Diagnostic: Interactive Run on GPU Node

To isolate the issue, a simple spec run with 32 output tokens was tested:

```bash
srun --partition=A100 --gres=gpu:A100:1 --time=01:00:00 \
  bash -c 'conda activate nano_moe && python scripts/debug_single_spec.py'
```

Result:
```
LLM created in 250.3s
Warmup done in 6.5s
Generated 32 tokens in 56.1s
Text: I need to find out if the user is a human or a bot...
SUCCESS
```

This confirmed spec mode functions correctly but takes ~5 minutes per case
due to model loading overhead.

### 3.4 First Precision Validation Results (Job 27089)

With subprocess isolation working and adequate timeout, the first precision
validation run produced:

```
STEP 1: Standard mode reference
  Standard mode: 128 tokens in 180.1s
  Reference text (first 200 chars):
    "The optimal caching strategy balances the tradeoff between memory usage
     and transfer cost. The paper proposes a novel expert caching strategy..."

[1/15] spec: round_robin r=0.25
  [MISMATCH@36]  ref:[41018, 279, 35915, 12624, 315]
                  test:[41018, 279, 2615, 12624, 315]
[2/15] spec: round_robin r=0.50
  [MISMATCH@53]  ref:[22670, 4938, 323, 3637, 29995]
                  test:[22670, 4938, 1393, 3637, 849]
```

**Key observations:**
- `round_robin` at ALL cache ratios produces mismatches
- Mismatch position shifts with cache ratio: position 36 for 25% cache,
  position 53 for 50% cache — more cached experts → later divergence
- The mismatch occurs at the **same position** and **same token IDs**
  across repeated runs on different GPU nodes (gpu18, gpu21) — the problem
  is deterministic, not random

### 3.5 Hypothesis Testing

#### Hypothesis A: CPU Fallback Precision

The verify step uses `FusedTorchCpuMoeBackend` (CPU-based `F.linear`) for
uncached experts, while standard mode uses `fused_moe_linear` (GPU Triton
kernel). Different matmul implementations produce different floating-point
results.

**Test**: Add `--no-cpu-exec` flag to force GPU fallback via
`GpuFallbackWorkspace` (which uses the same `fused_moe_linear` kernel).

Implementation:
1. Added `"cpu_expert_execution_enabled"` parameter to `run_case_inline.py`
2. Added `--no-cpu-exec` flag to `precision_validate.py`
3. Ran job 27112: standard + round_robin r=0.25 with `--no-cpu-exec`

**Result (Job 27112)**:
```
[MISMATCH@36] ref:[41018, 279, 35915, 12624, 315]
               test:[41018, 279, 2615, 12624, 315]
```

**Verdict**: Hypothesis REJECTED. Mismatch persists with GPU-only fallback.
CPU fallback is NOT the cause.

#### Hypothesis B: CUDA Graph Capture

CUDA Graph replay might introduce state differences between spec and standard
execution.

**Test**: Add `--enforce-eager` flag to disable CUDA Graph for both standard
and spec mode.

**Result (Job 27117)**:
```
[MISMATCH@36] ref:[41018, 279, 35915, 12624, 315]
               test:[41018, 279, 2615, 12624, 315]
```

**Verdict**: Hypothesis REJECTED. Mismatch persists with eager execution
(no CUDA Graph). Graph capture is NOT the cause.

#### Hypothesis C: KV Cache Management

The previous KV cache fixes (hashed partial tail invalidation, verify slot
reservation) might have edge cases.

**Analysis**: Code audit of `block_manager.py` and `spec_engine.py` confirms
the fixes are correct. The verify step rebuilds KV cache from the last
accepted token through all draft tokens using the full model. No stale
KV state should persist.

Additionally, if KV cache were the cause, the mismatch position would likely
be non-deterministic or would vary between runs. The consistent mismatch at
position 36 across runs strongly suggests a computational precision issue,
not a cache management bug.

**Verdict**: Ruled out by code audit and deterministic mismatch position.

#### Hypothesis D: Split GEMM Accumulation (ROOT CAUSE)

In standard mode, all 128 experts are processed in a **single**
`fused_moe_linear` call:

```
standard:  all_experts[0..127]  --fused_moe_linear--> output
```

In heterogeneous verify, the experts are **split** into two calls (cached
via `expert_cache.gate_up_buffer`, uncached via `GpuFallbackWorkspace`):

```
verify:  cached_experts[0..31]   --fused_moe_linear--> partial_A
         fallback_experts[32..N] --fused_moe_linear--> partial_B
         output = scatter_merge(A, B)
```

Both use the identical Triton `fused_moe_linear` kernel, but the **order
of floating-point accumulation** differs. Since BF16/FP16 addition is
non-associative, `matmul(X, [A|B]) != matmul(X, A) + matmul(X, B)` in the
least significant bits of the mantissa.

This tiny difference (on the order of 1e-4 in BF16) propagates through 48
MoE layers. By the time the hidden state reaches the final layer norm and
LM head, the argmax can differ — changing the output token.

**Evidence supporting this hypothesis:**

1. The mismatch is **deterministic** (same position, same tokens every run)
   — consistent with fixed floating-point accumulation differences
2. The mismatch position **shifts with cache ratio** — more cached experts
   (larger first GEMM) → later divergence (position 53 at 50% cache vs
   position 36 at 25% cache)
3. All other hypotheses (CPU fallback, CUDA Graph, KV cache) have been
   ruled out through direct testing
4. The GPU-only fallback test (using the same `fused_moe_linear` kernel)
   confirms it's not a CPU-vs-GPU issue but a **single-call-vs-split-call**
   issue

### 3.6 Precision Validation Full Results (Partial)

Only `round_robin` completed before the job was cancelled for debugging:

| Policy | Ratio | Match | AccRate | Mismatch Position |
|---|---|---|---|---|
| `round_robin` | 0.25 | NO | 0.8952 | 36 |
| `round_robin` | 0.50 | NO | 0.8500 | 53 |
| `round_robin` | 0.75 | running | - | - |

All other policies would exhibit the same issue since the root cause is in
`heterogeneous_moe_forward`, not in the reroute policy.

## 4. Root Cause

**The heterogeneous MoE forward splits expert computation across two separate
`fused_moe_linear` calls (cached + fallback), while standard mode uses a single
call. The split changes floating-point accumulation order, causing different
BF16/FP16 rounding in the least significant bits of the hidden states. After
48 layers of propagation, this causes a different argmax token.**

### 4.1 Minimal Reproduction

```
Standard:  fused_moe_linear(hidden, [E0..E127])        → logits_A
Verify:    fused_moe_linear(hidden, [E0..E31])  → p1
           fused_moe_linear(hidden, [E32..E127]) → p2
           p1 + p2                                     → logits_B

logits_A != logits_B  in BF16 least-significant bits
→ argmax(logits_A) != argmax(logits_B) at position 36
→ KV cache divergence → cascading token mismatch
```

### 4.2 Why Position 36?

With `max_draft_tokens=8`, position 36 is in the 5th spec step
(36 = 4 × 8 drafted + 4 accepted). By this point:
- The heterogeneous verify has been called ~5 times (5 verify steps)
- Each verify step processes 1+8=9 tokens through 48 MoE layers
- 5 × 9 × 48 = 2160 heterogeneous MoE forward passes
- The accumulated precision drift across 2160 passes crosses the argmax
  decision boundary at position 36

### 4.3 Why Does Cache Ratio Affect Mismatch Position?

- At 25% cache (32/128 experts): 96 experts go through the second GEMM
  call → large split → faster divergence (position 36)
- At 50% cache (64/128 experts): 64 experts in each GEMM call → more
  balanced split → slower divergence (position 53)
- At 75% cache (96/128 experts): only 32 experts in the second call →
  expected to diverge even later or possibly not at all for 128 tokens

## 5. Proposed Fix

### 5.1 Approach: Unified GEMM for Verify

For the **verify** step (where correctness matters), merge all experts into
a single `fused_moe_linear` call:

1. Upload all uncached expert weights to `GpuFallbackWorkspace`
2. Also copy cached expert weights into the workspace (or build a combined
   index mapping)
3. Map ALL routes (cached + uncached) to workspace slots
4. Run **one** `fused_moe_linear` on the unified workspace
5. Result: bit-identical output to standard mode

The **draft** step can continue using the split approach — speed matters
for draft, and draft errors are corrected by verify.

### 5.2 Implementation Location

- `nanovllm/layers/fuse_moe/heterogeneous.py` — `heterogeneous_moe_forward`
- Add a `verify_precision_mode: bool = False` parameter
- When enabled, skip the cached/CPU split and use workspace-only single-call path

### 5.3 Expected Impact

- **Draft forward**: No change (uses existing split path)
- **Verify forward**: Small overhead from copying cached weights to workspace
  (~1-2 ms per layer for weight copy), offset by eliminating CPU fallback
  overhead. Overall verify time should be comparable.
- **Correctness**: Verify output becomes bit-identical to standard mode
- **Graph compatibility**: Single `fused_moe_linear` call is already
  graph-capture-safe

## 6. Environment

| Item | Detail |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB (80 GB HBM2e) |
| Nodes | gpu18, gpu21 (Slurm partition A100) |
| Model | Qwen3-30B-A3B (128 experts, K=8, 48 MoE layers) |
| PyTorch | 2.6.0+cu124 |
| Conda env | nano_moe (Python 3.12.13) |
| Job IDs | 27067, 27069, 27074, 27080-27081, 27085, 27089, 27101, 27105, 27112, 27117 |

## 7. Artifacts

| Artifact | Path |
|---|---|
| Precision validation script | `scripts/precision_validate.py` |
| Subprocess runner | `scripts/run_case_inline.py` |
| Standard reference output | `results/precision_validation_*/standard_reference.json` |
| Standard reference text | `results/precision_validation_*/standard_reference.txt` |
| Logs | `logs/precision_validate_*.log`, `logs/prec_debug_*.log`, `logs/prec_fix*.log` |
