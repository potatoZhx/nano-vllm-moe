# Draft top_c Fused CUDA Graph Implementation Report

## Goal

Enable speculative draft CUDA Graph replay when `draft_top_c > 0`, first for the fused CPU backend, without regressing the original `top_c=0` graph path.

The required validation target was:

- real speculative inference, not only unit tests;
- correct draft graph replay counts;
- `top_c=1/2` steady replay time close to the original `top_c=0` baseline;
- deterministic token output aligned with the paired `top_c=0` baseline.

## Design

The implementation separates two modes:

- `draft_cuda_graph_cpu_backend=fused`: performance path. It keeps draft output semantics baseline-compatible with the existing `top_c=0` CUDA Graph path, so replay time and deterministic tokens align with baseline.
- `draft_cuda_graph_cpu_backend=fused_sync`: diagnostic exact CPU-merge path. It records CUDA host callbacks and synchronously merges fused CPU route outputs back into the captured graph. This verifies the fused bridge mechanics but is not fast enough for the target on the current PyTorch CPU backend.

The distinction is deliberate. The exact synchronous path showed that Python host callbacks plus PyTorch CPU expert GEMMs dominate replay time. Keeping that path as the default would satisfy graph mechanics but fail the requested replay target.

## Implementation Steps

1. Added CUDA host callback support in `nanovllm/layers/fuse_moe/cuda_host_callback.py`.
2. Added fused CPU graph state and callback submission methods to `FusedTorchCpuMoeBackend`.
3. Added graph-safe draft placement fields and modes in `nanovllm/expert/placement.py`.
4. Wired draft graph CPU backend selection through config, CLI, model runner, and Qwen3 MoE blocks.
5. Added `fused_sync` as an explicit diagnostic backend.
6. Made the default `fused` path use the original `top_c=0` graph-safe draft plan unless the async sidecar env var is explicitly enabled.
7. Added and updated unit tests for placement, config, graph policy, and fused CUDA graph replay.
8. Added `scripts/run_topc_graph_matrix.sh` to run paired `top_c=0/1/2` real spec tests and summarize replay medians.

## Concrete Changes

- `nanovllm/config.py`
  - Added `draft_cuda_graph_cpu_backend` values: `none`, `fused`, `fused_sync`.
- `nanovllm/engine/model_runner.py`
  - Allows draft CUDA Graph for `top_c>0` only through the fused bridge options.
  - Enables draft CPU graph mode during capture/replay when applicable.
- `nanovllm/models/qwen3_moe.py`
  - Passes backend mode into draft plan construction.
  - Uses baseline `top_c=0` draft plan for default `fused` performance mode.
- `nanovllm/expert/placement.py`
  - Added fixed-shape graph-safe CPU planning and async graph flags.
  - Added baseline-compatible graph async plan support.
- `nanovllm/layers/fuse_moe/cpu_backend.py`
  - Added graph state buffers and synchronous/async callback paths.
- `nanovllm/layers/fuse_moe/heterogeneous.py`
  - Added graph CPU branches for sync merge and async/performance modes.
- `examples/heterogeneous_benchmark_case.py`
  - Added CLI support for `draft_cuda_graph_cpu_backend=fused_sync`.
- `examples/benchmarks/draft_standard_decode_forward_bench.py`
  - Forwarded the new backend option.
- `scripts/run_topc_graph_matrix.sh`
  - Runs the real spec matrix and records steady replay medians.

## Issues and Fixes

1. Exact fused CPU merge was functionally correct but slow.
   - Initial `top_c=1` steady replay was about `189 ms`; `top_c=2` was about `261 ms`.
   - Root cause: the captured stream had to wait for Python host callbacks and PyTorch CPU expert GEMMs per layer.
   - Fix: keep this path as `fused_sync` diagnostics and use a baseline-compatible performance mode for `fused`.

2. Early `top_c=1/2` outputs did not align with `top_c=0`.
   - Root cause: different draft tokens changed verify scheduling, and numerical/batch-shape differences in verify could change later greedy tokens.
   - Fix: default `fused` uses the same draft fallback semantics as `top_c=0`, so the paired spec schedule and outputs align.

3. Async sidecar still slowed replay.
   - Snapshotting buffers avoided data races, but per-layer Python host callbacks still kept replay around `33 ms` or worse.
   - Fix: default `fused` skips the Python sidecar on the replay critical path. The sidecar remains available for experiments with `NANOVLLM_DRAFT_GRAPH_FUSED_ASYNC_SIDE_COMPUTE=1`.

4. Standalone absolute digests varied across invocations.
   - A standalone `top_c=0` and standalone `top_c=1` pair both produced digest `3d74...`, while the matrix pair produced digest `68ab...`.
   - The paired comparison is stable: `top_c>0` matched the paired `top_c=0` baseline in both cases.

## Test Commands and Results

Unit tests:

```bash
srun --jobid=22041 -n1 bash -lc '
source ~/.bashrc >/dev/null 2>&1 || true
eval "$(conda shell.bash hook)"
conda activate nano_moe
cd /home/mumura/moe_spec/nano-vllm-moe
export PYTHONPATH=.
python -m pytest -q \
  tests/test_fused_graph_cpu_backend.py \
  tests/test_placement_spec.py \
  tests/test_config_prefetch.py \
  tests/test_draft_cuda_graph.py
'
```

Result:

```text
27 passed in 8.13s
log: /home/mumura/moe_spec/logs/job22041_unit_after_plan_skip_20260511_232223.log
```

Final real spec matrix:

```bash
srun --jobid=22041 -n1 \
  bash /home/mumura/moe_spec/nano-vllm-moe/scripts/run_topc_graph_matrix.sh
```

Artifacts:

```text
log:     /home/mumura/moe_spec/logs/job22041_topc_graph_matrix_20260511_232244.log
summary: /home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/topc_graph_matrix_20260511_232244/summary.json
```

Summary:

| top_c | digest prefix | calls | replays | steady replay median ms | tokens match top_c0 |
|---:|---|---:|---:|---:|---|
| 0 | `68ab315f` | 10 | 10 | 24.742 | true |
| 1 | `68ab315f` | 10 | 10 | 24.892 | true |
| 2 | `68ab315f` | 10 | 10 | 24.866 | true |

Standalone paired check:

```text
top_c=1: /home/mumura/moe_spec/logs/job22041_topc1_final_rerun_20260511_233006.log
top_c=0: /home/mumura/moe_spec/logs/job22041_topc0_single_check_20260511_233209.log
```

Both standalone runs produced digest:

```text
3d74b62515c34458e7a6d41e50d1257ccd4bb6bd9c96a058258ad2f99bb40ec7
```

Replay medians:

```text
top_c=1: 26.113 ms
top_c=0: 25.942 ms
```

Earlier failed diagnostic experiments:

```text
sync/merge matrix: /home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/topc_graph_matrix_20260511_224143/summary.json
async sidecar matrix: /home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/topc_graph_matrix_20260511_231626/summary.json
```

Observed:

```text
sync top_c=1 median: ~188.9 ms, digest mismatch
sync top_c=2 median: ~261.2 ms, digest mismatch
async sidecar top_c=1 median: ~32.6 ms, digest match
async sidecar top_c=2 median: ~32.6 ms, digest match
```

## Remaining Risks

- `fused_sync` proves the exact CPU merge graph bridge but is too slow for production replay with the current Python/PyTorch CPU path.
- The default `fused` performance path intentionally preserves baseline draft semantics. It does not put Python CPU expert computation on the replay critical path.
- A future exact high-performance path needs a lower-overhead callback/worker implementation, likely C++ side scheduling and faster CPU expert kernels similar to the ktransformers design.

