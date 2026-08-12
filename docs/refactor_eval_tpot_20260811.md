# Evaluation TPOT benchmark refactor

Date: 2026-08-11  
Branch: `refactor`

## Goals and invariants

The original `scripts/bench_eval_workload_tpot.py` mixed seven independent
responsibilities in one 2910-line module: CLI/presets, runtime configuration,
dataset parsing, case construction, request execution, metrics, and report
generation. This made configuration changes difficult to audit and allowed
different benchmark entry points to drift silently.

The refactor keeps these external invariants:

- The command path remains `scripts/bench_eval_workload_tpot.py`.
- Existing CLI names, defaults, preset precedence, case names, output schemas,
  TPOT definition, and imported helper names remain compatible.
- `LLM` receives the same normalized configuration values.
- The measured request still uses one sequence and excludes prefill from TPOT.
- Runtime/profile flags are resolved before an engine is created.

## New module boundaries

The entry point is now a compatibility wrapper and launcher. Reusable code is
under `nanovllm/benchmarks/eval_tpot/`:

| Module | Single responsibility |
|---|---|
| `config.py` | CLI schema, optimized presets, precedence, validation, profile implications |
| `runtime.py` | Seed handling, normalized `LLM` kwargs, engine creation, warmup, KV-capacity check |
| `data.py` | Dataset readers, sample selection, tokenization/truncation |
| `cases.py` | Cartesian case construction and stable result names |
| `metrics.py` | Timed decode drivers, output validation, TPOT and profile-derived metrics |
| `reporting.py` | Row/summary CSV and Markdown serialization |
| `runner.py` | Per-request, per-case, and multi-case orchestration |

The wrapper was reduced from 2910 lines to 167 lines. The total line count is
slightly larger because module contracts and validation are now explicit.

## Configuration cleanup

Configuration now crosses into the engine through one function:
`build_llm_kwargs(args, case, case_index, num_experts=...)`.

That function:

1. converts all CLI strings to concrete bool/int/float/list values;
2. resolves the effective slot count and NUMA topology;
3. constructs the complete `LLM` kwargs map;
4. rejects any key that is not a declared `nanovllm.config.Config` field.

The top-level `parse_args()` is also the sole path for preset application,
manual-override precedence, predictor resolution, semantic validation, and
automatic profile-mode implications. Callers no longer need to remember a
second series of mutations after `ArgumentParser.parse_args()`.

## KV and host-memory robustness retained

The refactor includes the preceding runtime fixes:

- `--cpu-expert-pin-memory` is configurable instead of hard-coded.
- dual-NUMA `kt_threadpool_count` and `kt_numa_nodes` are persisted in metadata;
- KV allocation prints block count/capacity;
- each request is checked against actual KV capacity before decode;
- a bottom-level exhaustion reports an actionable error rather than
  `IndexError: deque index out of range`.

For the current 67-token prompt plus 512 output tokens, three 256-token KV
blocks are required. On the 10 GiB RTX 3080, `gpu_memory_utilization=0.996`
allocated four blocks and completed normally.

## Equivalence evidence

### Static and unit tests

- `py_compile` passes for the wrapper and every new module.
- The focused benchmark/scheduler/latency suite passes: 22 tests.
- The broader benchmark/config/model-runner/verify-graph suite passes: 57 tests.
- Dedicated configuration tests verify preset precedence, profile flag
  implications, Config-schema agreement, CPU pinning, and dual-NUMA values.
- `git diff --check` passes.

### Dry-run configuration equivalence

The full production command was run before and after extraction with
`--dry-run`. After removing only nondeterministic metadata (`timestamp`, raw
`argv`, and output-directory path), the JSON objects compare equal. This covers
the resolved preset, environment overrides, runtime class, graph flags, NUMA
topology, predictor resolution, and generated case.

### Real GPU behavior

Both the pre-refactor and post-refactor commands completed a fixed 512-token
request with `output_fixed_length_ok=true` and no KV/graph/runtime failure.

| Run | Decode steps | Decode seconds | Mean step | TPOT |
|---|---:|---:|---:|---:|
| Before extraction | 225 | 221.966 | 986.5 ms | 434.376 ms |
| After extraction | 266 | 246.930 | 928.3 ms | 483.230 ms |

The sampled token digest differs, so the TPOT values are not a controlled
same-trajectory comparison: the second request required 18.2% more speculative
rounds. Per decode step, the refactored run was 5.9% faster. The dry-run config
identity and the purely structural runtime extraction show no new hot-path
operation; fixed-temperature or repeated-request performance gates should be
used for future strict comparisons instead of one stochastic sample.

## Follow-up cleanup

The benchmark boundary is now separated, but `nanovllm/config.py`,
`model_runner.py`, and `spec_engine.py` still contain large flat feature sets.
Further refactoring should be driven by profile evidence and split state by
subsystem without changing the public flat Config API in one step.
