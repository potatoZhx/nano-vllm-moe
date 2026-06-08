# Plan: Full CUDA Graph for Verify Path — Hybrid GPU + kt_direct

## Context

nano-vllm-moe's verify path (`--spec-verify-miss-policy cpu` + `--cpu-expert-backend kt_direct`) currently runs eagerly. Each MoE layer involves Python-level plan building, dispatch, and merge. The existing partial verify CUDA graph only captures the prefix (attention+gate+topk), then falls to an eager gap (cache-fill) and eager GPU-only suffix.

**Goal:** Capture the **entire** verify forward (all layers, including MoE expert computation) as a CUDA graph per token-count bucket. GPU computes cached experts (majority), kt_direct CPU computes only uncached miss experts (minority). This matches KTransformers' `cudaLaunchHostFunc`-based approach while preserving GPU cache utilization.

**Approach: Hybrid GPU cached + kt_direct miss** — NOT all-kt. GPU processes all routes via substitution LUT (uncached routes get zero weight), kt_direct handles only miss experts (using `gpu_expert_mask_cpu` to skip GPU-cached ones). This maximizes GPU cache utilization — more GPU experts = faster inference.

**Why fully graph-safe:** The existing draft graph with `graph_safe_cpu=True` proves that plan building (substitution LUT, `argsort`, `scatter_add_`) is capturable using only tensor ops. `expert_to_slot_lut` and `cached_expert_mask` are persistent tensors updated in-place — graph replay reads current values at their stable addresses.

---

## KTransformers CUDA Graph Pattern (Reference)

Source: `kt/ktransformers/kt-sft/csrc/ktransformers_ext/cpu_backend/cpuinfer.h:69-91`

Per MoE layer, all captured as graph nodes:
1. `pinned_input.copy_(gpu_hidden)` — memcpy node
2. `pinned_ids.copy_(expert_ids)` — memcpy node
3. `cudaLaunchHostFunc(stream, submit_task, args)` — host callback: enqueues CPU work
4. `cudaLaunchHostFunc(stream, sync, cpu_infer)` — host callback: GPU waits for CPU
5. `gpu_output.copy_(pinned_output)` — memcpy node

Buffers are pre-allocated pinned memory at stable addresses. Graph replays with fresh data flowing through same pointers.

---

## Changes

### 1. Config — `nanovllm/config.py`

**New fields:**
- `prefetch_draft_layer_enabled: bool = True` — independent draft prefetch switch (split from verify)
- `verify_cuda_graph_kt_hybrid: bool = False` — auto-set flag

**In `__post_init__`:** When `verify_cuda_graph=True` AND `cpu_expert_backend=="kt_direct"` AND `spec_verify_miss_policy=="cpu"`:
- Set `verify_cuda_graph_kt_hybrid = True`
- Force `prefetch_verify_layer_enabled = False` (verify prefetch incompatible with full graph)

### 2. Graph-Safe Verify Plan Builder — `nanovllm/expert/placement.py`

**New function: `build_verify_graph_safe_plan_gpu()`**

Closely follows the `build_draft_plan_gpu(..., graph_safe_cpu=True)` `top_c<=0` path (line 798-832) but adapted for verify:

```python
def build_verify_graph_safe_plan_gpu(layer_idx, selected_experts, routing_weights, expert_cache, num_experts):
    flat_selected = _flatten_experts(selected_experts)   # reshape
    flat_weights = _flatten_weights(routing_weights)      # reshape
    device = flat_selected.device
    cached_expert_mask = expert_cache.get_cached_expert_mask()      # persistent tensor
    slot_to_expert_lut = expert_cache.get_slot_to_expert_lut()      # persistent tensor

    # Substitution LUT: uncached → round-robin cached (all tensor ops)
    # Reuses existing _build_topc0_substitution_lut (line 438)
    substitution_lut = _build_topc0_substitution_lut(num_experts, cached_expert_mask, slot_to_expert_lut, device)
    flat_effective = substitution_lut.index_select(0, flat_selected)

    # GPU layout: all routes → cached expert slots
    gpu_slots = expert_cache.expert_to_slot_lut.index_select(0, flat_effective)
    gpu_route_indices = torch.arange(flat_selected.numel(), dtype=torch.int64, device=device)
    m_sizes, gpu_route_indices = _build_grouped_layout(gpu_slots, gpu_route_indices, expert_cache.num_slots)

    # Masks: which routes are uncached → go to CPU
    uncached_route_mask = ~cached_expert_mask.index_select(0, flat_selected)

    # Zero GPU weights for uncached routes (GPU produces zero for these; kt_direct provides the real output)
    gpu_route_weights = torch.where(uncached_route_mask, torch.zeros_like(flat_weights), flat_weights)

    return MoEExecutionPlan(
        layer_idx=layer_idx,
        gpu_route_indices=gpu_route_indices,
        gpu_m_sizes=m_sizes,
        cpu_route_indices=None,           # kt_direct doesn't use route indices
        cpu_task_expert_ids=None,
        cpu_task_offsets=None,
        flat_selected_original=flat_selected,
        flat_selected_effective=flat_effective,
        gpu_route_weights=gpu_route_weights,
        cpu_graph_enabled=True,
        substitution_lut=substitution_lut,
        gpu_route_mask=torch.ones_like(flat_selected, dtype=torch.bool),
        cpu_route_mask=uncached_route_mask,
    )
```

**All tensor ops — graph-capturable.** No `torch.nonzero`, no `.item()`, no Python branching on tensor values.

### 3. kt_direct Graph Methods — `nanovllm/layers/fuse_moe/kt_direct_backend.py`

**Split the existing `forward()` (line 554) into begin/finish to allow GPU-CPU overlap:**

```python
def begin_forward_graph_verify(self, hidden_states, selected_experts, routing_weights):
    """Submit CPU work to kt_kernel. Call before GPU GEMM for overlap."""
    flat_hidden = hidden_states.view(-1, hidden_states.shape[-1])
    topk_ids = selected_experts.reshape(-1, self.num_experts_per_tok)
    topk_weights = routing_weights.reshape(-1, self.num_experts_per_tok)
    (input_cpu, expert_ids_cpu, routing_weights_cpu,
     output_cpu, batch_size_cpu, output_device) = KtDirectCPUBuffer.get_buffer(flat_hidden, self.num_experts_per_tok)
    slot = self.layer_idx % KtDirectCPUBuffer.buffer_depth
    # Refresh mask: DtoH copy captured as graph memcpy node — reads current cache state at replay
    self._refresh_gpu_expert_mask(non_blocking=True)
    # Copy inputs to pinned buffers (graph memcpy nodes)
    input_cpu[slot].copy_(flat_hidden, non_blocking=True)
    expert_ids_cpu[slot].copy_(topk_ids, non_blocking=True)
    routing_weights_cpu[slot].copy_(topk_weights, non_blocking=True)
    # Create task + submit (cudaLaunchHostFunc → graph host callback node)
    task = self.moe.forward_task(
        batch_size_cpu[slot].data_ptr(), self.num_experts_per_tok,
        expert_ids_cpu[slot].data_ptr(), routing_weights_cpu[slot].data_ptr(),
        input_cpu[slot].data_ptr(), output_cpu[slot].data_ptr(), False,
    )
    stream = torch.cuda.current_stream(flat_hidden.device).cuda_stream
    self.runtime.cpu_infer.submit_with_cuda_stream(stream, task)
    return slot  # caller runs GPU GEMM here for overlap

def finish_forward_graph_verify(self, hidden_states):
    """Sync CPU work + copy output back. Call after GPU GEMM."""
    slot = self.layer_idx % KtDirectCPUBuffer.buffer_depth
    flat_hidden = hidden_states.view(-1, hidden_states.shape[-1])
    (_, _, _, output_cpu, _, output_device) = KtDirectCPUBuffer.get_buffer(flat_hidden, self.num_experts_per_tok)
    stream = torch.cuda.current_stream(flat_hidden.device).cuda_stream
    # Sync: cudaLaunchHostFunc → graph host callback node (GPU waits for CPU completion)
    self.runtime.cpu_infer.sync_with_cuda_stream(stream, 0)
    # Copy result back to GPU (graph memcpy node)
    output_device[slot].copy_(output_cpu[slot], non_blocking=True)
    return output_device[slot]  # per-token output (miss experts only, cached experts zeroed)
```

**Key graph-safety points:**
- `_refresh_gpu_expert_mask(non_blocking=True)`: DtoH copy of `cached_expert_mask` → `gpu_expert_mask_cpu`. Captured as memcpy node. At replay, copies CURRENT mask values. The callback (submit) fires after this copy completes, so kt_kernel sees fresh mask.
- `moe.forward_task(...)`: Python call executes during capture only. Returns task struct with stable buffer pointers. `submit_with_cuda_stream` captures `cudaLaunchHostFunc` with those pointers.
- `sync_with_cuda_stream`: Captured as host callback node. At replay, GPU waits for CPU completion.
- All buffer pointers (`input_cpu[slot].data_ptr()`, etc.) are stable — `KtDirectCPUBuffer.capture_buffers` stores permanent allocations.

### 4. MoE Block — `nanovllm/models/qwen3_moe.py`

**New method on `Qwen3MoeHeterogeneousSparseMoeBlock`:**

```python
def forward_verify_kt_hybrid(self, hidden_states):
    """Graph-capturable verify forward: GPU cached experts + kt_direct miss experts."""
    # 1. Route (graph-safe: gate linear + softmax + topk + normalize)
    router_logits = self.gate(hidden_states)
    router_probs = nn.functional.softmax(router_logits, dim=1, dtype=torch.float32)
    routing_weights, selected_experts = torch.topk(router_probs, self.num_selected, dim=-1)
    if self.norm_topk_prob:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)

    # 2. Graph-safe plan (all tensor ops — see §2)
    plan = build_verify_graph_safe_plan_gpu(
        self.layer_idx, selected_experts, routing_weights, self.expert_cache, self.num_experts)

    # 3. Submit kt_direct CPU work (cudaLaunchHostFunc → graph node)
    self.cpu_backend.begin_forward_graph_verify(hidden_states, selected_experts, routing_weights)

    # 4. GPU GEMM — all routes, uncached routes produce zero (zeroed weights in plan)
    #    Reuses _run_gpu_cached_expert_path pattern from heterogeneous.py:576
    top_k = routing_weights.size(1)
    _, gpu_expert_out, _, _ = _run_gpu_cached_expert_path(
        hidden_states, plan.gpu_route_weights, top_k,
        plan.gpu_route_indices, plan.gpu_m_sizes, self.expert_cache, self.act_fn)

    # 5. Sync kt_direct CPU work (cudaLaunchHostFunc → graph node) + copy output
    kt_output = self.cpu_backend.finish_forward_graph_verify(hidden_states)

    # 6. Merge: GPU route output → per-token sum + kt_direct per-token output
    output = torch.zeros_like(hidden_states)
    num_tokens, hidden_dim = hidden_states.shape
    num_routes = num_tokens * top_k
    route_buffer = _get_route_buffer_cache().get(num_routes, hidden_dim, output.dtype, output.device)
    route_buffer.index_copy_(0, plan.gpu_route_indices.to(torch.int64), gpu_expert_out.to(dtype=output.dtype))
    token_output = route_buffer.view(num_tokens, top_k, hidden_dim).sum(dim=1)
    output.add_(token_output)
    output.add_(kt_output.to(dtype=output.dtype, device=output.device))  # kt_direct per-token contribution
    return output
```

**Pipeline within graph (per MoE layer, captured as sequential graph nodes):**
```
[gate+topk] → [plan build (tensor ops)] → [copy to pinned + submit CPU work]
    → [GPU GEMM: gather+compute+scatter] ← CPU work runs in parallel
    → [sync CPU work] → [copy kt output to GPU] → [merge + residual]
```

**New method on `Qwen3MoeDecoderLayer`:**

```python
def forward_verify_kt_hybrid(self, hidden_states, positions):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(positions, hidden_states)
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp.forward_verify_kt_hybrid(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states
```

**New method on `Qwen3MoeModel`:**

```python
def forward_verify_kt_hybrid_layers(self, hidden_states, position_ids, apply_norm):
    for decoder_layer in self.layers:
        is_moe = isinstance(decoder_layer.mlp, Qwen3MoeHeterogeneousSparseMoeBlock)
        if is_moe:
            hidden_states = decoder_layer.forward_verify_kt_hybrid(hidden_states, position_ids)
        else:
            hidden_states = decoder_layer(hidden_states, position_ids)
    if apply_norm:
        hidden_states = self.norm(hidden_states)
    return hidden_states
```

### 5. Capture & Replay — `nanovllm/engine/model_runner.py`

**Modified: `capture_verify_cudagraph()` (line 1985)**
- Early branch: if `config.verify_cuda_graph_kt_hybrid`, call `_capture_verify_cudagraph_kt_hybrid()` and return

**New: `_capture_verify_cudagraph_kt_hybrid()`**

For each bucket in `verify_cuda_graph_bucket_steps` (reversed):
1. Ensure bucket sizes are in `KtDirectCPUBuffer.capture_bs`
2. Set attention context (cu_seqlens, slot_mapping, block_tables)
3. Warmup: `embed_tokens → forward_verify_kt_hybrid_layers` (populates buffers, warms GPU)
4. `torch.cuda.synchronize()`
5. Capture: `torch.cuda.CUDAGraph()` wrapping `embed_tokens → forward_verify_kt_hybrid_layers`
6. Store in `self.verify_kt_hybrid_graphs[bucket]`
7. Store static buffers in `self.verify_graph_vars`

**One graph per bucket covering the entire model** (all layers). This eliminates ALL per-layer Python overhead during replay.

**Modified: `_can_use_verify_cudagraph()` (line 2103)**
- Add: if `verify_cuda_graph_kt_hybrid`, check `verify_kt_hybrid_graphs` dict

**New: `_run_verify_with_kt_hybrid_graph(input_ids, positions)`**
1. Select bucket ≥ `num_tokens`
2. Copy dynamic inputs into static graph-var buffers (input_ids, positions, cu_seqlens, slot_mapping, block_tables)
3. Set graph context
4. `graph.replay()`
5. Return `hidden_states[:num_tokens]`

**Modified: dispatch in `run_verify()` (line 1654)**
- If `verify_cuda_graph_kt_hybrid` and `_can_use_verify_cudagraph(...)`: call `_run_verify_with_kt_hybrid_graph`

**Modified: verify prefetch gating**
- Add `not config.verify_cuda_graph_kt_hybrid` to verify prefetch condition

### 6. Prefetch Split

Current state: `prefetch_verify_layer_enabled` controls verify-layer prefetch. Draft prefetch is controlled separately by other config fields.

Change: Add `prefetch_draft_layer_enabled: bool = True`. The draft prefetch integration in model_runner should check this flag independently.

When `verify_cuda_graph_kt_hybrid=True`, force `prefetch_verify_layer_enabled=False`. Draft prefetch remains independent.

---

## Buffer Safety

**`KtDirectCPUBuffer.buffer_depth=2`** with `slot = layer_idx % 2`:
- Layer 0→slot 0, layer 1→slot 1, layer 2→slot 0, ...
- `sync_with_cuda_stream` in each layer ensures layer N's CPU work completes before layer N+2 writes to the same slot
- Within a CUDA graph, node execution order matches capture order (sequential on the stream)
- Safe for any number of MoE layers

**Stable pointer invariant:** `KtDirectCPUBuffer.capture_buffers` stores permanent buffers keyed by `(batch_size, hidden_size, top_k, dtype, device)`. `moe.forward_task()` receives raw `data_ptr()` at capture time — these pointers remain valid at replay time.

**`gpu_expert_mask_cpu` freshness:** The DtoH copy `gpu_expert_mask_cpu.copy_(cached_expert_mask)` is captured as a memcpy graph node. At replay, it reads the CURRENT `cached_expert_mask` values (updated in-place by the expert cache between steps). The `cudaLaunchHostFunc` callback fires after this copy completes (stream ordering), so kt_kernel sees the current mask.

**`expert_to_slot_lut` / `cached_expert_mask` freshness:** Both are persistent tensors updated in-place. Graph compute nodes (LUT `index_select`, `torch.where`) read current values at replay time. The plan is correctly rebuilt each replay — no stale routing.

**Padded tokens:** `num_tokens < bucket` → kt_direct computes padding tokens, but only `[:num_tokens]` is used. Minimal waste.

---

## GPU-CPU Overlap Within the Graph

Per MoE layer, the captured graph node sequence is:

```
[plan build tensor ops]
  ↓
[DtoH: cached_expert_mask → gpu_expert_mask_cpu]
[DtoH: hidden → input_cpu, experts → ids_cpu, weights → weights_cpu]
  ↓
[cudaLaunchHostFunc: submit CPU work] ← CPU starts processing
  ↓
[GPU GEMM: gather → fused_moe_linear → scatter] ← runs in parallel with CPU
  ↓
[cudaLaunchHostFunc: sync] ← GPU waits for CPU if not done
  ↓
[HtoD: output_cpu → output_device]
  ↓
[merge: route_buffer sum + kt_output add → output + residual]
```

The GPU GEMM and kt_direct CPU work overlap when both are available. The sync ensures correctness.

---

## Verification

1. **Numerical correctness**: Compare verify eager (`verify_cuda_graph=false`, `spec_verify_miss_policy=cpu`, `cpu_expert_backend=kt_direct`) vs graph replay → logits should match (deterministic routing, same GPU GEMM + kt_direct outputs)
2. **End-to-end spec decoding**: Compare token output with graph on vs off (temperature=0) — should match
3. **Existing path preservation**: `verify_cuda_graph=true` + `cpu_expert_backend!=kt_direct` → falls back to prefix-only graph (unchanged behavior)
4. **Performance**: Measure verify forward latency — graph should eliminate per-layer Python overhead (~30+ layers × plan build + dispatch + merge)
5. **Prefetch independence**: Verify draft prefetch still works when verify prefetch is disabled
6. **Cache state correctness**: Run multiple verify steps, check that `gpu_expert_mask_cpu` reflects current cache state at each replay (not stale capture-time values)

---

## Implementation Summary

### Branch

`feature/verify-cuda-graph-kt-hybrid` — 5 files modified, 361 insertions.

### Modified Files

| File | Change |
|---|---|
| `nanovllm/config.py` | Added `prefetch_draft_layer_enabled`, `verify_cuda_graph_kt_hybrid` fields; auto-set logic in `__post_init__` |
| `nanovllm/expert/placement.py` | Added `build_verify_graph_safe_plan_gpu()` — graph-safe plan builder using substitution LUT |
| `nanovllm/layers/fuse_moe/kt_direct_backend.py` | Added `begin_forward_graph_verify()` / `finish_forward_graph_verify()` — split submit/sync for GPU-CPU overlap |
| `nanovllm/models/qwen3_moe.py` | Added `forward_verify_kt_hybrid` on MoeBlock, DecoderLayer, Model — full hybrid forward path |
| `nanovllm/engine/model_runner.py` | Added `_capture_verify_cudagraph_kt_hybrid()`, `_run_verify_with_kt_hybrid_graph()`, modified dispatch |

### How It Works

1. **Config auto-set**: When `verify_cuda_graph=True` + `cpu_expert_backend=kt_direct` + `spec_verify_miss_policy=cpu`, the engine auto-enables `verify_cuda_graph_kt_hybrid` and disables verify-layer prefetch.

2. **Graph capture**: One CUDA graph per token-count bucket captures the entire model forward (all layers including MoE). Per MoE layer, the graph contains:
   - Graph-safe plan building (substitution LUT, argsort, scatter_add — all tensor ops)
   - `copy_to_pinned` + `cudaLaunchHostFunc(submit)` — CPU work begins
   - GPU GEMM on all routes (uncached routes use substituted experts with zero weight)
   - `cudaLaunchHostFunc(sync)` + `copy_from_pinned` — CPU result collected
   - Merge: GPU per-route output summed per-token + kt_direct per-token output added

3. **Graph replay**: Dynamic inputs (input_ids, positions, cu_seqlens, block_tables) are copied into static graph-var buffers, then `graph.replay()` runs the entire model. At replay, all persistent tensors (`expert_to_slot_lut`, `cached_expert_mask`, `gpu_expert_mask_cpu`) contain current values — no stale data.

4. **GPU-CPU overlap**: Per MoE layer, GPU GEMM and kt_direct CPU computation run in parallel. The `sync_with_cuda_stream` call ensures CPU finishes before merge.

### Tests

Test file: `tests/test_verify_cuda_graph_kt_hybrid.py`

| Test Class | What It Tests |
|---|---|
| `TestConfigAutoSet` | Auto-set of `verify_cuda_graph_kt_hybrid` and `prefetch_verify_layer_enabled` under various config combos |
| `TestBuildVerifyGraphSafePlan` | Plan correctness: cached weights preserved, uncached weights zeroed, substitution LUT maps to cached experts, cpu_route_mask matches uncached, all routes go to GPU |
| `TestKtDirectGraphMethods` | `begin_forward_graph_verify` submits CPU work, `finish_forward_graph_verify` syncs and returns output, buffer slot alternates by layer_idx |
| `TestModelRunnerVerifyDispatch` | `_can_use_verify_cudagraph` returns True/False correctly for kt_hybrid graphs, multi-seq rejection, bucket bounds |
| `TestPlanConsistency` | Graph-safe plan routes all tokens to GPU (4 routes) vs eager plan (only cached routes, 2 routes) |

Run tests:

```bash
python -m pytest tests/test_verify_cuda_graph_kt_hybrid.py -v
```

### Benchmark

Benchmark script: `scripts/bench_verify_kt_hybrid.py`

Compares verify performance with kt_hybrid graph **on** vs **off** across cache ratios, output lengths, and max_draft_tokens.

Metrics: acceptance_rate, route_hit_rate, avg_miss_per_layer, avg_active_per_layer, throughput_output_tok_s, verify_forward_ms_avg, verify_kt_hybrid_replay_count.

Run benchmark:

```bash
python scripts/bench_verify_kt_hybrid.py \
    --output-dir results/kt_hybrid_bench \
    --cache-ratios 0.25,0.50,0.75 \
    --output-lens 128,256 \
    --max-draft-tokens-values 4,8
```

Or with slurm:

```bash
TS=$(date +%Y%m%d_%H%M%S)
srun --jobid=<JOB_ID> --ntasks=1 bash -c "
  source /opt/Software/Anaconda3/etc/profile.d/conda.sh
  conda activate nano_moe
  cd /home/mumura/moe_spec/nano-vllm-moe/
  python scripts/bench_verify_kt_hybrid.py \
      --output-dir results/kt_hybrid_bench_${TS} \
      --cache-ratios 0.25,0.50,0.75 \
      --output-lens 128,256 \
      --max-draft-tokens-values 4,8
"
```

---

## Graph-Internal Stats & Metadata Recording

### Problem

The kt_hybrid CUDA graph captures the entire model forward as one graph. During replay, no Python code executes, so:
- `_last_profile` is never set → profiling keys (miss/layer, active/layer, hit rate) are zero
- `runtime_meta_recorder.record_layer()` is never called → prefetcher metadata pipeline breaks

### Solution: Expert Status Vector (0/1/2) + Histograms

Per MoE layer, three fixed-shape device tensors are written inside the graph using graph-safe tensor ops:

| Buffer | Shape | Dtype | Content | Graph-safe op |
|--------|-------|-------|---------|---------------|
| `expert_status` | `(num_layers, num_experts)` | int8 | 0=inactive, 1=hit, 2=miss | `scatter_` |
| `activation_count` | `(num_layers, num_experts)` | int32 | Route count per expert | `scatter_add_` |
| `score_sum` | `(num_layers, num_experts)` | float32 | Routing weight sum per expert | `scatter_add_` |

Status encoding: `cached_expert_mask.index_select(flat_selected)` → `torch.where(is_cached, 1, 2)` → `scatter_` into `expert_status[layer_idx]`. Same expert from multiple routes writes same value (cached status is per-expert), so duplicate indices are safe. No `active_routes` mask is applied to status (unlike histogram): padding routes write the same per-expert value, and padding-only experts are filtered by `activation_count > 0` at readback.

After replay, derive stats using both `expert_status` and `activation_count` (for route-level counts matching the eager path):
```python
ac = activation_count[layer_idx]        # per-expert route count (0 for padding-only)
s = expert_status[layer_idx]            # per-expert 0/1/2

is_real = ac > 0                        # filter padding-only experts
miss_routes = ac[s == 2].sum()          # route-level miss count (matches eager)
total_routes = ac.sum()                 # total active routes
miss_experts = ((s == 2) & is_real).sum()  # unique miss expert count
```

### Modified Files

| File | Change |
|---|---|
| `nanovllm/expert/runtime_meta.py` | Added `"verify_kt_hybrid"` mode: `expert_status` device/host buffers, `uncached_route_mask` param on `record_layer`, status `scatter_` logic, `"histogram_kt_hybrid"` format in offload/collect, new fields on `LayerRuntimeMetaCPU` |
| `nanovllm/models/qwen3_moe.py` | Added `record_layer()` call in `forward_verify_kt_hybrid` with `uncached_route_mask=plan.cpu_route_mask` |
| `nanovllm/engine/model_runner.py` | Arm recorder during capture/replay with `mode="verify_kt_hybrid"`, post-replay readback of `expert_status` → populate `_last_profile`, route `"verify_kt_hybrid"` in prefetch metadata dispatch |

### Data Flow

```
Capture time:
  arm("verify_kt_hybrid", max_bucket) → allocate device buffers (stable addresses)
  warmup → record_layer() runs eagerly
  capture → record_layer() ops captured into graph

Replay time:
  arm("verify_kt_hybrid", num_tokens) → zero buffers, set token_count_capture_value
  graph.replay() → scatter_/scatter_add_ write current data into device buffers
  readback: expert_status.cpu() + activation_count.cpu() → route-level miss count → _last_profile (for benchmarks)
  offload_async: D→H copy → collect() → LayerRuntimeMetaCPU → prefetcher observe_verify
```

### Memory

~50 KB device + ~50 KB pinned host for 64 layers x 64 experts. Negligible.

### Tests

Added `TestExpertStatusRecording` in `tests/test_verify_cuda_graph_kt_hybrid.py`:
- `test_status_values_hit_and_miss`: Verifies 0/1/2 encoding for mixed routes
- `test_status_all_cached` / `test_status_all_miss`: Edge cases
- `test_derive_stats_from_status`: Verifies activated/miss/hit derivation
- `test_histogram_also_recorded`: Verifies activation_count and score_sum alongside status
- `test_collect_filters_padding_only_experts`: Simulates graph-replay host buffers with padding-only experts, verifies they are excluded from miss_count/active_count via activation_count > 0 filter
- `test_collect_produces_status_fields`: Verifies offload → collect → LayerRuntimeMetaCPU with miss_count, active_count, expert_status
