# Spec Mode Code Review Report

**Date**: 2026-05-29  
**Scope**: `nanovllm/` — all speculative decoding related implementation  
**Method**: Static code review (no experiments executed)

---

## Review Scope

The following files were reviewed exhaustively:

| Module | File | Lines |
|--------|------|-------|
| Spec Engine | `engine/speculative/spec_engine.py` | 291 |
| Acceptance | `engine/speculative/acceptance.py` | 194 |
| Model Runner | `engine/model_runner.py` | 1896 |
| LLM Engine | `engine/llm_engine.py` | 236 |
| Scheduler | `engine/scheduler.py` | 84 |
| Block Manager | `engine/block_manager.py` | 154 |
| Sequence | `engine/sequence.py` | 125 |
| Draft Reroute | `scheduling/draft_reroute.py` | 270 |
| Draft Scheduler | `scheduling/draft_scheduler.py` | 190 |
| Cache Strategy | `scheduling/cache_strategy.py` | 274 |
| Prefetch Strategy | `scheduling/prefetch_strategy.py` | 54 |
| Prefetcher | `expert/prefetcher.py` | 1636 |
| Expert Cache | `expert/cache.py` | 517 |
| Config | `config.py` | 174 |
| Sampler | `layers/sampler.py` | 22 |

---

## Summary Table

| Category | Critical | High | Medium | Low |
|----------|----------|------|--------|-----|
| Correctness Bug | 1 | 2 | 1 | — |
| Precision Risk | — | — | 3 | — |
| Efficiency | — | 1 | 2 | 3 |
| Architecture / Design | — | — | 5 | — |
| Edge Case / Boundary | — | — | 3 | — |

**Top priority**: BUG-1 (sampling fallback stalls the engine) — a user-triggerable functional failure.

---

## 1. Correctness Bugs

### BUG-1 [Critical] Sampling fallback never appends tokens — engine stalls permanently

**File**: `engine/speculative/spec_engine.py:79-80`

```python
if has_sampling and not use_sampling_accept:
    return self.model_runner.call("run", seqs, False)
```

**Trigger condition**: Any batch where at least one sequence has `temperature > 0` and the configured `acceptance_strategy` is not a sampling variant (`"greedy"` or `"standard"`).

**Root cause**: This fallback returns raw `token_ids` from `model_runner.run()`. The caller `llm_engine.py:174-183` in spec mode **never calls `scheduler.postprocess()`**:

```python
# llm_engine.py:174-183
elif self.config.inference_mode == "spec":
    token_ids = self.spec_engine.speculative_step(seqs)
    # ← No postprocess call — tokens are never appended to sequences
    outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
    return outputs, num_tokens
```

**Consequences**:
1. Generated tokens are discarded — never appended to `seq.token_ids`.
2. EOS / max_tokens termination is never triggered.
3. Sequences remain in RUNNING state indefinitely.
4. The engine enters an infinite loop, decoding the same last token forever.

**Impact**: This is easily triggered in production: any user who sets `temperature > 0` with the default `acceptance_strategy="standard_sampling"` is safe, but switching to `acceptance_strategy="greedy"` or `"standard"` with sampling requests causes a hard hang.

**Suggested fix**: Either (a) call `self.scheduler.postprocess(seqs, token_ids)` before returning from the fallback path, or (b) handle token appending and finish detection inline:

```python
if has_sampling and not use_sampling_accept:
    token_ids = self.model_runner.call("run", seqs, False)
    # Must replicate postprocess behavior since llm_engine.step() skips it in spec mode.
    for seq, token_id in zip(seqs, token_ids):
        seq.append_token(token_id)
    for seq in seqs:
        self._maybe_mark_finished(seq)
    return [seq.last_token for seq in seqs]
```

---

### BUG-2 [High] `StandardAcceptance` threshold parameter is dead — always behaves as greedy

**Files**: `engine/speculative/acceptance.py:65-107`, `engine/speculative/spec_engine.py:81`

**Root cause chain**:

1. `return_logits` is computed as:
   ```python
   return_logits = bool(has_sampling and use_sampling_accept)  # line 81
   ```
2. When `acceptance_strategy="standard"`, `use_sampling_accept` is `False`, so `return_logits = False`.
3. With `return_logits=False`, `run_verify` returns argmax token IDs (`list[int]`), not logits.
4. In `StandardAcceptance.accept()`, when `verify_data` is not a tensor:
   ```python
   if not isinstance(verify_data, torch.Tensor):
       # Pure argmax comparison — threshold is COMPLETELY IGNORED
       trace = _to_verify_trace(verify_data)
       for i, tok in enumerate(draft_tokens):
           if i >= len(trace) or tok != trace[i]:
               break
           num_accepted += 1
   ```

**Consequence**: A user who configures `acceptance_strategy="standard"` with `acceptance_threshold=0.7` expects probabilistic acceptance (accept if p(token) >= 0.7), but gets exact greedy matching instead. The `threshold` parameter has zero effect.

The threshold-aware branch (lines 92-107 using `probs[i, tok].item() < self.threshold`) is dead code — it requires `verify_data` to be a logits tensor, which never happens for this strategy.

**Suggested fix**: Set `return_logits = True` when `acceptance_strategy == "standard"` as well, or document that "standard" is functionally identical to "greedy" in the current implementation.

---

### BUG-3 [High] `verify_trace_len == 0` path does not roll back tokens — corrupts sequence state

**File**: `engine/speculative/spec_engine.py:229-234`

```python
if verify_trace_len == 0:
    seq.finish_draft()
    self._maybe_mark_finished(seq)
    final_token_ids.append(seq.last_token)
    seq.num_cached_tokens = original_cached_tokens[seq.seq_id]
    continue
```

**Context**: At this point in execution, `seq.token_ids` has already been extended with all draft tokens (lines 163-170, preparing verify input). The normal acceptance path (lines 255-262) overwrites `seq.token_ids` with the correct accepted prefix. But this early-exit path skips that entirely.

**State after this path executes**:
- `seq.token_ids` contains **all unverified draft tokens** — not rolled back.
- `seq.num_tokens` reflects the inflated length.
- `seq.last_token` points to the last draft token (unverified).
- KV cache state (via `accept_draft_kv`) was never called — block table is inconsistent.

**Reachability**: Under normal operation this path is unreachable (verify always returns at least 1 position). However, as defensive code it is actively harmful — if a future change makes `verify_trace_len == 0` possible (e.g., a verify error fallback), it would silently corrupt all affected sequences.

**Suggested fix**:

```python
if verify_trace_len == 0:
    self.scheduler.accept_draft_kv(seq, 0)
    seq.token_ids = base_tokens_map[seq.seq_id]
    seq.num_tokens = len(seq.token_ids)
    seq.last_token = seq.token_ids[-1]
    seq.finish_draft()
    seq.num_cached_tokens = original_cached_tokens[seq.seq_id]
    self._maybe_mark_finished(seq)
    final_token_ids.append(seq.last_token)
    continue
```

---

### BUG-4 [Medium] `reserve_active_slot_for_prefetch` evicts expert before async copy completes — race window

**File**: `expert/cache.py:342-369`

```python
def reserve_active_slot_for_prefetch(self, ...):
    prev_expert = self.slot_to_expert[active_slot_idx]
    if prev_expert >= 0 and prev_expert in self.expert_to_slot:
        del self.expert_to_slot[prev_expert]          # ← Immediate eviction
        self.expert_to_slot_lut[prev_expert] = -1     # ← GPU LUT updated
        self.cached_expert_mask[prev_expert] = False   # ← Mask cleared
    self.slot_to_expert[active_slot_idx] = -1
    self.active_slot_pending_expert[active_slot_idx] = int(expert_idx)
```

The evicted expert is removed from all lookup structures **immediately**, before the replacement async copy even begins. During the transfer window:
- The old expert's weights are **physically present** in the GPU buffer but **logically invisible** (`cached_expert_mask[prev] = False`).
- The new expert is **not yet present** (`active_slot_pending_expert[slot] = new_expert`, but `cached_expert_mask[new] = False`).
- Any MoE routing that queries cache status during this window sees both experts as uncached.

**Contrast with correct pattern**: `reserve_active_slot_for_prefetch_deferred` (line 371) correctly defers eviction — it does not clear `expert_to_slot` or `cached_expert_mask` until `commit_deferred_active_prefetch`. This is used by the `draft_segment_indexed` path.

**Impact**: Affects the `draft_direct_active` and `verify_layer_predict` prefetch paths. During the async transfer, routing may choose suboptimal substitutes for the evicted expert, causing transient accuracy degradation.

**Suggested fix**: Unify both reservation paths to use deferred eviction semantics.

---

## 2. Precision Risks

### PREC-1 [Medium] `_finalize` degrades to single-expert representation when all routes miss cache

**File**: `scheduling/draft_reroute.py:136-144`

```python
row_sum = routing_weights.sum(dim=-1, keepdim=True)
empty = row_sum.le(0)
inject = empty & self.first_route_mask
final_weights = torch.where(inject, torch.ones_like(routing_weights), routing_weights)
normalizer = torch.where(empty, torch.ones_like(row_sum), row_sum).clamp_min(1e-9)
final_weights = final_weights / normalizer
```

When a token's all top-k experts miss the cache (all weights zeroed by `hit_mask`), `_finalize` injects weight 1.0 into the first route pointing to a fallback expert. The token's hidden state is then entirely determined by one potentially unrelated expert.

**Risk scenario**: With aggressive rerouting (high `gamma`, high miss rate), a significant fraction of tokens may hit this fallback, especially during early draft steps before cache warms up.

**Alternative approaches**:
- Preserve the original top-1 expert's contribution even if uncached (accept the latency hit for accuracy).
- Use an identity / zero contribution for fully-missed tokens (passthrough).
- Reduce rerouting aggressiveness when miss rate is very high.

---

### PREC-2 [Medium] `_entropy_cache_bias` can displace original top-1 hit without protection

**File**: `scheduling/draft_reroute.py:165-172`

The protection mechanism only activates when:
1. Original top-1 is a **miss** (`original_top1_missed`), AND
2. It was **displaced** from the rerouted set.

If original top-1 is a **hit** but gets displaced by higher-biased cache hits, there is no protection. Under high entropy with large `gamma`, the bias term can overwhelm the original logit ordering, potentially replacing the model's top-1 choice entirely.

This means the rerouting can change the dominant expert for a token — a stronger intervention than mere substitution of lower-ranked experts.

---

### PREC-3 [Medium] `LFURankGuardStrategy` EMA is too slow relative to protection threshold

**File**: `scheduling/cache_strategy.py:155-169`

With defaults `ema_alpha=0.95`, `protect_threshold=0.15`:

```
Per-step EMA contribution = (1 - alpha) * current[eid]
  = 0.05 * (2.0 / num_tokens)   [for rank-1]
  = 0.05 * 0.0078                [with 256-token batch]
  ≈ 0.0004
```

To reach `protect_threshold = 0.15` from zero, an expert must appear as rank-1 in every verify step for approximately `0.15 / 0.0004 ≈ 375` consecutive steps. The protection is effectively inert during normal operation — by the time an expert accumulates enough score, the workload has likely shifted.

**Suggestion**: Either lower `ema_alpha` (e.g., 0.8) or lower `protect_threshold` (e.g., 0.05), or normalize the contribution by a fixed constant instead of `num_tokens`.

---

## 3. Efficiency Issues

### EFF-1 [High] O(n²) trace lookup in acceptance phase

**File**: `engine/speculative/spec_engine.py:243-253`

```python
for seq_trace in step_trace["sequences"]:
    if seq_trace["seq_id"] == int(seq.seq_id):
        ...
        break
```

This linear scan runs inside the per-sequence acceptance loop, making the overall complexity O(n²) where n is the batch size. With `max_num_seqs=512`, this is up to 262,144 comparisons per step.

**Fix**: Build a `{seq_id: trace_index}` dict before the loop:

```python
seq_trace_map = {t["seq_id"]: t for t in step_trace["sequences"]}
# Then inside the loop:
seq_trace = seq_trace_map[int(seq.seq_id)]
```

---

### EFF-2 [Medium] `_ensure_prefetch_internal_state()` runs ~20 `hasattr` checks on every call

**File**: `engine/model_runner.py:261-301`

This method is called at the entry of almost every prefetch-related method. Each invocation performs approximately 20 `hasattr` checks against `self`. In the draft loop, it may be called 5-10 times per draft step.

**Fix**: Add a single `self._prefetch_state_initialized = True` flag set once in `__init__`, and guard the entire method body:

```python
def _ensure_prefetch_internal_state(self) -> None:
    if getattr(self, "_prefetch_state_initialized", False):
        return
    # ... existing hasattr checks ...
    self._prefetch_state_initialized = True
```

---

### EFF-3 [Medium] Python-level loops over experts in prefetch queue updates

**Files**: `expert/prefetcher.py:119-160` (GlobalWarmStartQueue), `expert/prefetcher.py:282-323` (SegmentCandidateIndex)

Both `update_from_runtime_meta` methods iterate over unique expert IDs in Python:

```python
for expert_idx, new_score, new_count in zip(unique_ids.tolist(), score_sum.tolist(), counts.tolist()):
    expert_idx = int(expert_idx)
    if cache.is_cached_cpu(expert_idx):
        continue
    ...
```

For MoE-128/256 models, this can iterate over 50-100+ experts per layer per step. With 28 layers, that's 1400-2800 Python loop iterations per metadata observation.

**Suggestion**: Batch the `is_cached_cpu` check using `cached_expert_mask` tensor indexing, then process only the uncached subset.

---

### EFF-4 [Low] `speculative_step` copies all token_ids for every sequence

**File**: `engine/speculative/spec_engine.py:161`

```python
base_tokens_map = {seq.seq_id: list(seq.token_ids) for seq in seqs}
```

For sequences with thousands of tokens, this is an O(L × n) copy. The `base_tokens` are only used at line 258 for reconstruction:

```python
seq.token_ids = base_tokens + accepted_draft + [next_token]
```

**Alternative**: Store only the truncation point and reconstruct via slice:

```python
base_len_map = {seq.seq_id: seq._draft_start_num_tokens for seq in seqs}
# At line 258:
seq.token_ids = seq.token_ids[:base_len_map[seq.seq_id]] + accepted_draft + [next_token]
```

However, this requires `seq.token_ids` to not be modified in-place between capture and use. Since `rollback_tokens_to_draft_start()` already restores tokens and then re-appends occur, this optimization needs careful validation.

---

### EFF-5 [Low] `deepcopy(self._step_traces)` on every profile retrieval

**File**: `engine/speculative/spec_engine.py:50`

```python
out["step_traces"] = deepcopy(self._step_traces)
```

`_step_traces` is a list of dicts that grows with every speculative step. `deepcopy` on nested structures is expensive. If profiling is retrieved frequently (e.g., every N steps for logging), this adds non-trivial overhead.

**Alternative**: Move traces to a separate buffer that is swapped atomically on `get_profile(reset=True)`.

---

### EFF-6 [Low / Optimization Opportunity] Mixed-temperature batches lose speculative benefit entirely

**File**: `engine/speculative/spec_engine.py:73-80`

Current logic: if **any** sequence in the batch uses sampling (`temperature > 0`) and the acceptance strategy doesn't support sampling, the **entire batch** falls back to standard decode.

For mixed batches (e.g., 90% greedy, 10% sampling), all greedy sequences lose their speculative speedup.

**Suggestion**: Split the batch — process greedy sequences through the speculative path and sampling sequences through standard decode. This adds scheduling complexity but could significantly improve throughput for mixed workloads.

---

## 4. Architecture & Design

### ARCH-1 `_maybe_mark_finished` duplicates `scheduler.postprocess` responsibilities

**Files**: `engine/speculative/spec_engine.py:277-290` vs `engine/scheduler.py:65-71`

Both implement finish detection (EOS check, max_tokens check) and resource cleanup (block deallocation, running list removal). The conditions differ subtly:

| | spec_engine | scheduler |
|---|---|---|
| max_tokens check | `num_completion_tokens >= max_tokens` | `num_completion_tokens == max_tokens` |
| block deallocation | Conditional on `hasattr(scheduler, "block_manager")` | Direct |
| running removal | Conditional on `hasattr(scheduler, "running")` | Direct |

Any future change to finish conditions (e.g., stop sequences, length penalties) must be synchronized across both implementations.

**Suggestion**: Extract a shared `check_and_mark_finished(seq, eos, scheduler)` utility, or have `speculative_step` delegate finish handling to `scheduler.postprocess`.

---

### ARCH-2 Pervasive `getattr(config, "field", default)` despite well-defined dataclass

Throughout the codebase (especially `model_runner.py` and `prefetcher.py`), config fields are accessed via `getattr` with fallback defaults:

```python
getattr(config, "draft_top_c", 0)
getattr(config, "prefetch_step_budget", 0)
getattr(config, "draft_prefetch_visible_budget_ms", 3.0)
```

`Config` is a `@dataclass` with explicit defaults for every field. `getattr` with defaults:
1. Hides field name typos (no error if misspelled).
2. Prevents IDE navigation and type checking.
3. The fallback default may diverge from the dataclass default.

**Recommendation**: Use direct attribute access (`config.draft_top_c`). If backward compatibility with older Config versions is needed, handle it in `Config.__post_init__` with explicit migration.

---

### ARCH-3 `SegmentCandidateIndex` and `GlobalWarmStartQueue` share ~80% duplicated logic

Both classes contain nearly identical implementations for:
- Runtime meta tensor normalization (device/dtype conversion)
- `torch.unique` + `scatter_add_` aggregation
- Uncached filtering via `cache.is_cached_cpu()`
- Decay + priority computation
- TTL-based pruning

**Suggestion**: Extract a shared `_aggregate_and_filter_runtime_meta(runtime_meta, layer_caches, step_id, source, config)` utility that returns `list[tuple[int, int, float, int]]` (layer_idx, expert_idx, score, count).

---

### ARCH-4 Profile system is unstructured and overly verbose

`PrefetchRuntime.get_profile()` (lines 1512-1635) manually constructs a dict with 100+ keys. `ModelRunner._profile` uses a `defaultdict(float)` that accumulates ad-hoc string keys across dozens of methods. Key naming is inconsistent (e.g., `"draft_direct_active_prefetch_skipped_by_budget_count"` — 54 characters).

Problems:
- No type safety — misspelled keys silently create new counters.
- Hard to discover which counters exist without reading all code.
- get_profile() is 120 lines of boilerplate.

**Suggestion**: Define structured `@dataclass` profile groups:

```python
@dataclass
class DraftPrefetchProfile:
    submit_count: int = 0
    ready_count: int = 0
    publish_count: int = 0
    consumed_count: int = 0
    ...
```

---

### ARCH-5 `DraftReroutePolicy` inherits `nn.Module` without trainable parameters

**File**: `scheduling/draft_reroute.py:71`

`DraftReroutePolicy` extends `nn.Module` solely for `register_buffer` (device tracking). It has no trainable parameters, no `state_dict` requirements, and no gradient flow. The `nn.Module` overhead (parameter registry, hook system, repr) is unnecessary.

Alternative: Use a plain class with explicit `self.device` tracking and `to(device)` method, or use `torch.jit.ScriptModule` if JIT compilation is desired.

---

## 5. Edge Cases & Boundary Conditions

### EDGE-1 `draft_steps == 0` still incurs full speculative overhead

When `_budget_draft_steps` returns 0 (sequence nearly at `max_tokens`), the speculative step still executes:
1. `start_draft()` + `scheduler.start_draft_kv()` — saves draft checkpoint state.
2. Draft loop — zero iterations (no-op).
3. `rollback_draft_kv()` + `rollback_tokens_to_draft_start()` — undoes the checkpoint.
4. Verify preparation — appends 0 draft tokens, sets up 1-token recomputation.
5. `run_verify()` — runs a full prefill-mode forward pass for 1 token.
6. Acceptance — accepts 0 draft tokens, generates 1 next token.

This is functionally equivalent to a standard decode step but goes through 6 phases of overhead (checkpoint, rollback, verify prep, verify forward, acceptance, finish detection).

**Suggestion**: Short-circuit to standard decode when `draft_steps == 0`:

```python
draft_steps = self._budget_draft_steps(seqs)
if draft_steps == 0:
    return self.model_runner.call("run", seqs, False)
    # (With proper token append / finish handling — see BUG-1)
```

---

### EDGE-2 Draft loop does not detect EOS — wastes compute on post-EOS tokens

**File**: `engine/speculative/spec_engine.py:122-148`

The draft loop generates `draft_steps` tokens unconditionally. If a draft token is EOS at step k, steps k+1 through draft_steps-1 generate meaningless tokens that will either be rejected by verify or accepted and immediately trigger finish detection.

For `max_draft_tokens=8`, generating 7 post-EOS tokens wastes 7 draft forward passes and inflates the verify input.

**Suggestion**: Check for EOS after each draft step and break early for that sequence. Since the batch is processed together, maintain a per-sequence `done` mask:

```python
for step_idx in range(draft_steps):
    draft_result = self.model_runner.call("run_draft", seqs, return_logits)
    ...
    for row_idx, (seq, token_id) in enumerate(zip(seqs, token_ids)):
        if seq_done[seq.seq_id]:
            continue
        seq.append_draft_token(token_id)
        draft_tokens_map[seq.seq_id].append(token_id)
        if token_id == self.scheduler.eos and not seq.ignore_eos:
            seq_done[seq.seq_id] = True
```

Note: This requires handling variable-length draft per sequence in the verify phase, adding complexity. A simpler approach is to just break the entire loop if ALL sequences have hit EOS.

---

### EDGE-3 `BlockManager.accept_draft` does not update `num_cached_tokens` — inconsistency window

**File**: `engine/block_manager.py:143-153`

```python
def accept_draft(self, seq: Sequence, num_accepted: int):
    target_tokens = seq._draft_start_num_tokens + num_accepted
    # ... block cleanup ...
    seq.num_tokens = target_tokens
    # ← num_cached_tokens is NOT updated here
```

`num_cached_tokens` is separately restored in `spec_engine.py:262`:

```python
seq.num_cached_tokens = original_cached_tokens[seq.seq_id]
```

Between `accept_draft_kv` (line 255) and the restoration (line 262), `num_cached_tokens` holds whatever value was set during verify preparation (line 176: `seq._draft_start_num_tokens - 1`). If any code between these two points reads `num_cached_tokens`, it gets a stale/incorrect value.

Currently no code reads it in this window, but the fragile ordering dependency is a maintenance hazard.

**Suggestion**: Have `accept_draft_kv` also set `seq.num_cached_tokens` to a consistent value, or document the invariant that `num_cached_tokens` is unreliable during the acceptance phase.

---

## 6. Additional Observations

### Thread safety in `PrefetchRuntime._profile`

`PrefetchRuntime._profile` (a `defaultdict(float)`) is written from both the main thread (during `submit_*` methods under `_prefetch_runtime_lock`) and the background worker (during `_process_prefetch_metadata_item` under the same lock). However, `get_profile()` in `model_runner.py:807-808` reads it under `_prefetch_runtime_lock`, which is correct.

The concern is `spec_engine.py`'s direct writes to `self._profile` (its own defaultdict) — these happen on the main thread without locks, which is fine since `SpeculativeEngine` is single-threaded.

No immediate bug, but the locking discipline is subtle and undocumented.

### `Sequence.__getstate__` / `__setstate__` serialization gap

`Sequence.__setstate__` (line 108-124) reconstructs token_ids only when `num_completion_tokens == 0`. For sequences mid-generation, only `last_token` is preserved, not the full token list. If a sequence is serialized during a speculative step (via shared memory for multi-GPU), the draft token state may be lost.

The `__getstate__` for draft fields (lines 104-107) preserves `draft_token_ids` and `_draft_start_num_tokens`, but if `token_ids` itself was discarded (line 102-103), reconstruction in `__setstate__` produces an empty `token_ids` list with non-zero `num_tokens` — an inconsistent state.

### `DraftReroutePolicy.forward` uses `@torch.compile(fullgraph=True)`

The `fullgraph=True` constraint means dynamic control flow (data-dependent branching) is forbidden. The `if self.policy == ...` dispatch at lines 258-268 is resolved at trace time based on `self.policy` (a string constant), so this is safe. However, if batch size or top_k changes between calls, recompilation may occur. The `mode="max-autotune-no-cudagraphs"` is appropriate for avoiding capture conflicts.

---

## 7. Recommended Fix Priority

### Immediate (before next experiment run)

1. **BUG-1**: Fix sampling fallback to append tokens and check finish. (~10 lines changed in `spec_engine.py`)

### Short-term (before merge to main)

2. **BUG-2**: Either enable `return_logits` for `StandardAcceptance` or document the equivalence with greedy.
3. **BUG-3**: Add proper rollback in the `verify_trace_len == 0` defensive path.
4. **EFF-1**: Replace O(n²) trace lookup with dict.
5. **EDGE-1**: Short-circuit `draft_steps == 0` to standard decode.

### Medium-term (next refactoring pass)

6. **BUG-4**: Unify `reserve_active_slot_for_prefetch` to use deferred eviction.
7. **EFF-2**: Guard `_ensure_prefetch_internal_state` with initialization flag.
8. **ARCH-1**: Extract shared finish-detection utility.
9. **ARCH-2**: Replace `getattr(config, ...)` with direct attribute access.
10. **ARCH-3**: Deduplicate queue/index update logic.

### Low priority (quality of life)

11. **PREC-3**: Tune EMA parameters for rank guard responsiveness.
12. **EFF-3/4/5**: Vectorize Python loops, optimize copies, reduce profile overhead.
13. **ARCH-4/5**: Restructure profile system, simplify DraftReroutePolicy base class.
14. **EDGE-2/3**: Draft EOS detection, block manager num_cached_tokens consistency.
