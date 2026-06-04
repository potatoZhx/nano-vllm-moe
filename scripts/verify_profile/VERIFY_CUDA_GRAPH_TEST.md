# Verify CUDA Graph Test Guide

## Overview

The verify CUDA graph optimization captures prefix operations (attention + gate routing)
as CUDA graphs to reduce kernel launch overhead during speculative decoding verification.
This is controlled by the `verify_cuda_graph` config flag (default: off).

## Prerequisites

- flash-attn >= 2.5.7 (CUDA graph capture support for `flash_attn_varlen_func`)
- CUDA >= 12.0
- Working spec mode configuration with heterogeneous MoE

## Running Tests

### Quick single-case test

```bash
# Run with graph enabled
python scripts/verify_profile/test_verify_cuda_graph.py \
    --model-path /path/to/Qwen3-30B-A3B \
    --verify-cuda-graph-on \
    --output-dir results/verify_graph_test \
    --output-len 512 \
    --cache-ratio 0.75

# Run baseline (no graph) for manual comparison
python scripts/verify_profile/test_verify_cuda_graph.py \
    --model-path /path/to/Qwen3-30B-A3B \
    --output-dir results/verify_graph_test \
    --output-len 512 \
    --cache-ratio 0.75
```

### Automated comparison test (recommended)

Runs both baseline and graph cases sequentially, then generates a comparison table:

```bash
python scripts/verify_profile/test_verify_cuda_graph.py \
    --model-path /path/to/Qwen3-30B-A3B \
    --compare-baseline \
    --output-dir results/verify_graph_test \
    --output-len 512 \
    --cache-ratio 0.75
```

### Using with existing benchmark infrastructure

Pass `verify_cuda_graph=True` to `LLM(...)` in any existing benchmark script:

```python
llm = LLM(
    model_path,
    inference_mode="spec",
    enable_heterogeneous=True,
    enable_speculative=True,
    verify_cuda_graph=True,    # <-- enable verify prefix graph
    # ... other config ...
)
```

## Output Files

The test script generates these files in `--output-dir`:

| File | Content |
|---|---|
| `baseline.json` | Full profile for eager verify path |
| `graph.json` | Full profile for graph verify path |
| `comparison.md` | Side-by-side comparison table (only with `--compare-baseline`) |

## Key Metrics to Observe

### In the comparison table

| Metric | What it means | Expected with graph |
|---|---|---|
| `verify_forward_ms_avg` | Per-call verify forward latency | **15-35% lower** |
| `throughput_output_tok_s` | End-to-end token throughput | Higher |
| `outputs_digest` | SHA256 of output tokens | **Must match baseline** |
| `verify_route_ms_total` | Total gate + softmax + topk time | Near 0 (absorbed by prefix graph) |
| `verify_plan_ms_total` | Total plan building time | Unchanged (stays in eager gap) |
| `verify_gpu_compute_ms_total` | Total GPU expert compute | Unchanged (suffix is eager in Phase 1) |
| `run_verify_total_ms_avg` | Per-call total verify overhead | Lower |

### Correctness check

The most important check is **`outputs_digest`**: baseline and graph cases must produce
identical token sequences. If they differ, the comparison report shows the first
divergence position.

Small floating-point differences are expected but should not affect token selection
with `temperature=0.0` (greedy decoding).

## Troubleshooting

### `CUDA error: operation not permitted during stream capture`

The flash-attn version does not support CUDA graph capture. Either:
- Upgrade flash-attn: `pip install flash-attn>=2.5.7`
- Disable the feature: remove `--verify-cuda-graph-on` or set `verify_cuda_graph=False`

### Token output mismatch (digest differs)

1. Check `max_seqlen_q` / `max_seqlen_k` padding: graph capture bakes in max values
   as constants. If the actual sequence lengths exceed the bucket size, behavior is
   undefined.
2. Run with `--output-len 8` for a minimal test to isolate the issue.
3. Compare token-by-token: the comparison report shows the first divergence position.

### OOM during graph capture

Reduce the number of buckets:

```python
LLM(..., verify_cuda_graph_bucket_steps=[8, 16])
```

Or reduce `--gpu-memory-utilization` to leave more room for graph memory.

### Graph capture hangs or is very slow

- Ensure Triton autotuning has completed (happens during model warmup).
- Ensure `@torch.compile` functions have been warmed up with the target shapes.
- The first capture run does warmup + capture; subsequent buckets reuse the graph pool.

## Configuration Reference

| Config field | Type | Default | Description |
|---|---|---|---|
| `verify_cuda_graph` | bool | `False` | Enable verify prefix CUDA graph |
| `verify_cuda_graph_bucket_steps` | list[int] | `[4, 8, 12, 16]` | Token-count buckets for graph capture |

When `enforce_eager=True`, `verify_cuda_graph` is automatically disabled.
