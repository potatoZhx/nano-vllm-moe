# Segmented Verify CUDA Graph — Implementation Report

## Problem

The monolithic `verify_cuda_graph_kt_hybrid` captures the entire model forward as a single CUDA graph. During replay no Python runs between layers, so:
- Expert prefetch is disabled → cache staleness → lower draft acceptance rate
- More miss experts accumulate → heavier CPU computation

## Solution: Segmented Verify Graph with Inter-Segment Prefetching

Split the verify kt_hybrid graph into N segments. Between each segment's `graph.replay()`, Python runs to:
1. Offload metadata for the just-completed segment (async D→H + observe)
2. Submit prefetch for the NEXT segment (reads from draft-prepared candidate index)

These two operations are fully independent (CPU vs PCIe) and run in parallel.

Circular prefetch: segment N targets segment 0's layers, warming cache for the next speculative round.

---

## Implementation Summary

### Branch

`feature/verify-cuda-graph-kt-hybrid` (extends the monolithic kt_hybrid with segmentation) — 5 core files modified, ~550 insertions.

### Modified Files

| File | Change |
|---|---|
| `nanovllm/config.py` | Added `verify_prefetch_segment_size` (default 12), `verify_prefetch_visible_budget_ms` (12.0), `verify_prefetch_min/max_per_boundary` (0/16); validation in `__post_init__` |
| `nanovllm/models/qwen3_moe.py` | Added `forward_verify_kt_hybrid_segment()` on `Qwen3MoeModel` (layer range iteration) and `Qwen3MoeForCausalLM` (embed + dispatch) |
| `nanovllm/engine/model_runner.py` | Added `_verify_segment_boundaries()`, `_verify_segment_graph_enabled()`, `_capture_verify_cudagraph_kt_hybrid_segments()`, `_run_verify_with_kt_hybrid_segment_graph()`, `_enqueue_verify_segment_metadata()`; modified `_can_use_verify_cudagraph()`, `run_verify()` dispatch, post-verify metadata gating |
| `nanovllm/expert/prefetcher.py` | Added `verify_segment_index` on `PrefetchRuntime`; added `submit_verify_segment_prefetch()` with bandwidth-budgeted submission; updated `observe_verify()` on both `PrefetchRuntime` and `PredictivePrefetchRuntime` to feed the verify segment index |
| `nanovllm/expert/runtime_meta.py` | Updated `target_host_buffer_pool_size()` for `verify_kt_hybrid` mode to size pool from verify segment count |

### How It Works

1. **Segment boundaries**: `_verify_segment_boundaries()` splits `num_hidden_layers` into segments of `verify_prefetch_segment_size`. For 48 layers with segment_size=12: `[(0,12), (12,24), (24,36), (36,48)]`.

2. **Graph capture** (`_capture_verify_cudagraph_kt_hybrid_segments`): Pre-allocates persistent `segment_outputs` buffers. Per bucket (reversed), per segment: warmup forward → `torch.cuda.synchronize()` → capture graph. Segment i reads from `segment_outputs[i-1]`, writes to `segment_outputs[i]`. Metadata recorder armed once (device buffers indexed by layer, disjoint across segments).

3. **Graph replay** (`_run_verify_with_kt_hybrid_segment_graph`): Sequential `graph.replay()` for each segment. After each segment:
   - `publish_direct_active_ready()` — commit completed prefetch transfers
   - `_enqueue_verify_segment_metadata()` — async metadata offload with `submit_after_phase=None` (decoupled from prefetch)
   - `submit_verify_segment_prefetch()` — submit prefetch for NEXT segment (circular via `(seg_idx+1) % N`)

4. **Prefetch submission** (`submit_verify_segment_prefetch`): Merges candidates from `draft_segment_index` (primary, draft-prepared), `verify_segment_index` (secondary), and `long_term_segment_index` (fallback). Budget-limited by `verify_prefetch_visible_budget_ms` and `_estimated_expert_transfer_ms()`.

   The benchmark uses `prefetch_step_budget=16`, allowing about 15 transfers per boundary at the measured ~0.79 ms/expert estimate, or roughly 60 transfers across four verify segments.

5. **Compatibility**: When `verify_prefetch_segment_size >= num_layers` → single segment → `_verify_segment_graph_enabled()` returns False → monolithic graph path unchanged.

### Inter-Segment Prefetch Timeline

```
Segment 0 graph.replay()  →  layers [0, 12)
  ↓
  ┌────────────────────────────────────────────────┐
  │ PARALLEL:                                      │
  │  [CPU] offload metadata [0,12)                 │
  │  [PCIe] submit prefetch → segment 1 experts    │
  └────────────────────────────────────────────────┘
        ↕ both overlap with ↓

Segment 1 graph.replay()  →  layers [12, 24)
  ↓  ...

Segment N-1 graph.replay()  →  layers [36, 48)
  ↓
  ┌────────────────────────────────────────────────┐
  │  [CPU] offload metadata [36,48)                │
  │  [PCIe] submit prefetch → segment 0 (CIRCULAR) │
  └────────────────────────────────────────────────┘
```

### Key Design Decisions

- **Decoupled metadata and prefetch**: Unlike draft where metadata observe gates prefetch submit, verify prefetch reads from the already-populated draft `SegmentCandidateIndex`. Metadata offload updates the index for future use but is not on the critical path.
- **Pre-allocated segment output buffers**: `segment_outputs` are allocated once and reused across all bucket captures and replays (stable addresses for graph replay).
- **kt_direct buffer safety**: `KtDirectCPUBuffer` slot assignment (`layer_idx % buffer_depth`) is sequential within each segment. Segment boundaries sit between graph replays where no kt_direct ops run.

---

## Tests

### Unit Tests

Test file: `tests/test_verify_segment_graph.py`

| Test Class | What It Tests |
|---|---|
| `TestVerifySegmentConfig` | Config field defaults, validation constraints (segment_size >= 1, budget_ms >= 0, max >= min) |
| `TestVerifySegmentBoundaries` | `_verify_segment_boundaries()` for even/uneven/per-layer splits; `_verify_segment_graph_enabled()` gating |
| `TestCanUseVerifyWithSegments` | `_can_use_verify_cudagraph()` checks `verify_kt_hybrid_segment_graphs` when segment-enabled; bucket overflow rejection |
| `TestModelForwardSegment` | `forward_verify_kt_hybrid_segment()` — first segment requires input_ids, subsequent use hidden_states, embed_tokens called correctly |
| `TestVerifySegmentIndex` | `verify_segment_index` creation with correct segment_size, `_segment_id()` mapping, circular target computation |
| `TestEnqueueVerifySegmentMetadata` | `_enqueue_verify_segment_metadata()` — passes correct layer range, `submit_after_phase=None` (decoupled), last segment sets `record_verify_consumed=True` |
| `TestPostVerifyMetadataGating` | Segment graph skips full-model offload; monolithic path does full offload |
| `TestRuntimeMetaPoolSizing` | `target_host_buffer_pool_size("verify_kt_hybrid")` accounts for segment count + headroom |
| `TestExpertStatusAcrossSegments` | Disjoint layer ranges accumulate in same device buffer; segment-scoped `offload_async` works |
| `TestSubmitVerifySegmentPrefetch` | Returns 0 with no candidates; returns 0 when budget exhausted |

### Existing Tests (preserved)

Test file: `tests/test_verify_cuda_graph_kt_hybrid.py` — all existing tests for the monolithic kt_hybrid path remain unchanged.

### Run Tests

```bash
# Segment graph tests (no CUDA required for most)
python -m pytest tests/test_verify_segment_graph.py -v

# Existing kt_hybrid tests
python -m pytest tests/test_verify_cuda_graph_kt_hybrid.py -v

# All together
python -m pytest tests/test_verify_segment_graph.py tests/test_verify_cuda_graph_kt_hybrid.py -v
```

---

## Benchmark

### Script

`scripts/bench_verify_segment_graph.py`

Compares three modes:
1. **eager** — `verify_cuda_graph=false`
2. **kt_hybrid_mono** — `verify_cuda_graph=true`, `verify_prefetch_segment_size=9999` (full model)
3. **kt_hybrid_segN** — `verify_cuda_graph=true`, `verify_prefetch_segment_size=N`

### Metrics

| Metric | Description |
|---|---|
| `acceptance_rate` | Draft token acceptance rate |
| `route_hit_rate` | Fraction of expert routes that hit GPU cache |
| `avg_miss_routes_per_layer` | Average miss route count per layer (duplicated: one token routing to same uncached expert counts multiple times) |
| `avg_miss_unique_experts_per_layer` | Average unique miss expert count per layer (deduplicated) |
| `avg_active_per_layer` | Average active expert count per layer |
| `throughput_output_tok_s` | Output tokens per second |
| `verify_forward_ms_avg` | Average verify forward latency |
| `draft_forward_ms_avg` | Average draft forward latency |
| `verify_segment_prefetch_submit_count` | Total experts submitted for prefetch at verify segment boundaries |
| `verify_segment_prefetch_submit_per_verify` | Average prefetch submissions per verify call |
| `verify_segment_metadata_enqueue_count` | Number of segment metadata offloads |
| `verify_segment_metadata_enqueue_ms` | Total metadata offload time (ms) |
| `direct_active_prefetch_publish_count` | Prefetch transfers published (committed to cache) |
| `verify_kt_hybrid_segment_replay_count` | Number of segmented verify graph replays |

### Delta Table

Each segmented mode is compared against the eager baseline. Key deltas:
- `acceptance_delta`: positive = segmented improves acceptance
- `route_hit_delta`: positive = segmented improves cache hit rate
- `miss_routes_delta`, `miss_unique_delta`: negative = fewer misses
- `throughput_delta`: positive = faster
- `verify_ms_delta`: negative = lower verify latency

### Run Benchmark

```bash
python scripts/bench_verify_segment_graph.py \
    --output-dir results/verify_segment_bench \
    --cache-ratios 0.25,0.50,0.75 \
    --output-lens 128,256 \
    --max-draft-tokens-values 4,8 \
    --segment-sizes 12,24
```

With slurm:

```bash
TS=$(date +%Y%m%d_%H%M%S)
srun --jobid=<JOB_ID> --ntasks=1 bash -c "
  source /opt/Software/Anaconda3/etc/profile.d/conda.sh
  conda activate nano_moe
  cd /home/mumura/moe_spec/nano-vllm-moe/
  python scripts/bench_verify_segment_graph.py \
      --output-dir results/verify_segment_bench_${TS} \
      --cache-ratios 0.25,0.50,0.75 \
      --output-lens 128,256 \
      --max-draft-tokens-values 4,8 \
      --segment-sizes 12,24
"
```

---

## Verification Checklist

1. **Numerical correctness**: temperature=0, compare segmented vs monolithic → logits must match (same graph-safe ops, same kt_direct computation)
2. **Prefetch activity**: `verify_segment_prefetch_submit_count > 0` with segments; `= 0` without
3. **Cache hit rate**: `route_hit_rate` should increase with segments vs monolithic (inter-segment prefetch warms cache)
4. **Draft acceptance**: Should improve with segments (fresher cache at verify time)
5. **No compute blocking**: `verify_segment_metadata_enqueue_ms / enqueue_count < segment_compute_time`
6. **Circular prefetch**: After last segment, prefetch targets segment 0's layer range
7. **Expert status**: `_last_profile` per MoE layer populated correctly (same device buffer, disjoint layer ranges)
8. **Compatibility fallback**: `verify_prefetch_segment_size=9999` → monolithic path → same behavior as before
