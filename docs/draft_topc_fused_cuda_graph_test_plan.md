# Draft top_c Fused CUDA Graph Test Plan

## Scope

This plan validates the optional fused draft CUDA Graph path for speculative decoding when `draft_top_c > 0`.

The performance path is enabled with:

```bash
--cpu-expert-execution-enabled true
--cpu-expert-backend fused
--draft-cuda-graph-cpu-backend fused
--enforce-eager false
--draft-top-c 1   # or 2
```

`fused` is the baseline-compatible performance backend. It keeps the captured draft graph replay on the same critical path as the original `top_c=0` graph. The exact synchronous CPU merge implementation is retained for diagnostics as:

```bash
--draft-cuda-graph-cpu-backend fused_sync
```

Do not use `fused_sync` for the pass/fail performance target; it is expected to be slower until the CPU kernel/callback path is optimized further.

## Correctness Criteria

For a paired run using the same command template, same seed, same prompts, and same hardware:

1. `top_c=1` generated token ids must equal the paired `top_c=0` baseline.
2. `top_c=2` generated token ids must equal the paired `top_c=0` baseline.
3. `engine_profile.model_draft_graph_replay_count` must equal `engine_profile.spec_run_draft_calls`.
4. `engine_profile.model_graph_hit_rate` must be `1.0`.
5. Steady-state `run_draft_core_run` median excluding the first replay must be close to `top_c=0`.

The first replay is excluded from the steady-state comparison because it regularly includes one-time runtime/cache effects.

## Quick Unit Checks

Run on an active CUDA-capable node:

```bash
cd /home/mumura/moe_spec/nano-vllm-moe
source ~/.bashrc
conda activate nano_moe
export PYTHONPATH=.

python -m pytest -q \
  tests/test_fused_graph_cpu_backend.py \
  tests/test_placement_spec.py \
  tests/test_config_prefetch.py \
  tests/test_draft_cuda_graph.py
```

Expected result from the validated implementation:

```text
27 passed
```

Validated log:

```text
/home/mumura/moe_spec/logs/job22041_unit_after_plan_skip_20260511_232223.log
```

## Real Spec Matrix

Use a fresh A100 allocation or an active A100 job dedicated to this validation. Do not reuse or cancel unrelated jobs.

Validated command:

```bash
srun --jobid=<A100_JOB_ID> -n1 \
  bash /home/mumura/moe_spec/nano-vllm-moe/scripts/run_topc_graph_matrix.sh
```

The script runs:

- `top_c=0`, `draft_cuda_graph_cpu_backend=none`
- `top_c=1`, `draft_cuda_graph_cpu_backend=fused`
- `top_c=2`, `draft_cuda_graph_cpu_backend=fused`

It writes:

```text
/home/mumura/moe_spec/logs/job${SLURM_JOB_ID}_topc_graph_matrix_${RUN_TS}.log
/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/topc_graph_matrix_${RUN_TS}/summary.json
```

Validated final run:

```text
log:     /home/mumura/moe_spec/logs/job22041_topc_graph_matrix_20260511_232244.log
summary: /home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/topc_graph_matrix_20260511_232244/summary.json
```

Validated summary:

| top_c | digest prefix | draft calls | graph replays | steady median ms | tokens match top_c0 |
|---:|---|---:|---:|---:|---|
| 0 | `68ab315f` | 10 | 10 | 24.742 | true |
| 1 | `68ab315f` | 10 | 10 | 24.892 | true |
| 2 | `68ab315f` | 10 | 10 | 24.866 | true |

## Standalone Paired Check

If a single `top_c=1` or `top_c=2` command is run outside the matrix, also run a single `top_c=0` command with the same environment before comparing tokens. Absolute digests can vary across standalone invocations because expert-cache and verify scheduling state can differ, but paired `top_c=0` and `top_c>0` runs must match.

Validated standalone pair:

```text
top_c=1 log: /home/mumura/moe_spec/logs/job22041_topc1_final_rerun_20260511_233006.log
top_c=0 log: /home/mumura/moe_spec/logs/job22041_topc0_single_check_20260511_233209.log
```

Both produced digest:

```text
3d74b62515c34458e7a6d41e50d1257ccd4bb6bd9c96a058258ad2f99bb40ec7
```

The `top_c=1` standalone steady replay median was `26.113 ms`; the paired `top_c=0` standalone steady replay median was `25.942 ms`.

## Diagnostic Modes

Use these only for investigation, not for the performance pass/fail gate.

Exact synchronous CPU merge:

```bash
--draft-cuda-graph-cpu-backend fused_sync
```

Async Python CPU sidecar:

```bash
NANOVLLM_DRAFT_GRAPH_FUSED_ASYNC_SIDE_COMPUTE=1 \
python examples/heterogeneous_benchmark_case.py ... \
  --draft-cuda-graph-cpu-backend fused
```

Observed diagnostic behavior:

- The original synchronous fused merge was correct against graph-safe eager semantics, but replay was about `189 ms` for `top_c=1` and `261 ms` for `top_c=2`.
- The Python async sidecar preserved output alignment, but per-layer Python host callback overhead kept replay around `33 ms` to `107 ms` depending on the sidecar variant.

## Large Test Procedure

For a tester unfamiliar with this codebase:

1. Request one A100 allocation and record the job id.
2. Enter the repository:

   ```bash
   cd /home/mumura/moe_spec/nano-vllm-moe
   ```

3. Run the quick unit checks and save the log.
4. Run `scripts/run_topc_graph_matrix.sh` through `srun --jobid=<A100_JOB_ID>`.
5. Open the printed `summary.json`.
6. Confirm the five correctness criteria in this document.
7. Save the log path, summary path, and the three JSON paths from the summary.
8. If any criterion fails, rerun once on the same idle visible GPU, then attach both summaries and logs to the issue report.

Minimum result table to report:

| field | where to find it |
|---|---|
| generated digest | `summary.json[*].digest` |
| token match | `summary.json[*].tokens_match_topc0` |
| graph replays | `summary.json[*].graph_replays` |
| draft calls | `summary.json[*].draft_calls` |
| replay median | `summary.json[*].draft_core_trace_median_excl_first_ms` |
| full JSON path | `summary.json[*].path` |

