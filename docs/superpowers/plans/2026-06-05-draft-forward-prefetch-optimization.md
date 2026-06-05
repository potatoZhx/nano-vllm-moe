# Draft Forward Prefetch Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure and reduce single-request speculative draft-forward overhead at cache ratios 0.25 and 0.5, while increasing the number of expert weights transferred during the draft GPU window.

**Architecture:** Extend the existing draft-versus-standard benchmark and `PrefetchRuntime` profile rather than introducing a second measurement path. First establish a CUDA Graph baseline, then add ticket-level byte and stream accounting, optimize only the measured candidate-selection bottleneck, and finally test an opt-in fixed CUDA H2D stream pool with deferred cache mapping preserved.

**Tech Stack:** Python, PyTorch CUDA streams/events, nano-vllm-moe CUDA Graph execution, `unittest`, Slurm A100 job 29629.

---

## File Structure

- Modify `examples/benchmarks/draft_standard_decode_forward_bench.py`: calculate per-draft-forward expert counts, bytes, transfer timing, and stream statistics.
- Modify `examples/heterogeneous_benchmark_case.py`: expose the transfer-stream count CLI argument and pass it into `Config`.
- Modify `nanovllm/config.py`: add and validate `prefetch_transfer_stream_count`.
- Modify `nanovllm/expert/prefetcher.py`: record ticket bytes/timing/stream ownership, optimize bounded candidate selection, and own the transfer stream pool.
- Modify `tests/test_config_prefetch.py`: validate stream-count configuration.
- Modify `tests/test_prefetch_runtime.py`: verify per-ticket counters, bounded ranking equivalence, stream selection, and deferred publication behavior.
- Modify `tests/test_draft_standard_decode_forward_bench.py`: verify per-forward metric extraction and repeat summaries.
- Create `scripts/draft_profile/run_draft_forward_profile.py`: run the ratio/stream/segment matrix and save one normalized JSON report.
- Modify `scripts/draft_profile/draft_forward_optimization_report_20260605.md`: append exact commands, raw paths, results, analysis, and keep/revert decisions after every experiment.

### Task 1: Establish The Unmodified A100 Baseline

**Files:**
- Modify: `scripts/draft_profile/draft_forward_optimization_report_20260605.md`
- Output: `results/draft_profile_20260605/baseline_ratio25.json`
- Output: `results/draft_profile_20260605/baseline_ratio50.json`
- Log: `/home/mumura/moe_spec/logs/draft_profile_baseline_*.log`

- [ ] **Step 1: Run the focused CPU test baseline**

Run:

```bash
python -m unittest \
  tests/test_config_prefetch.py \
  tests/test_prefetch_runtime.py \
  tests/test_expert_cache_staging.py \
  tests/test_draft_standard_decode_forward_bench.py
```

Expected: all tests pass before production-code changes.

- [ ] **Step 2: Run cache-ratio 0.25 baseline on job 29629**

Run:

```bash
srun --jobid=29629 --ntasks=1 bash -lc '
source /opt/Software/Anaconda3/etc/profile.d/conda.sh
conda activate nano_moe
export CUDA_VISIBLE_DEVICES=2
cd /home/mumura/moe_spec/nano-vllm-moe
python examples/benchmarks/draft_standard_decode_forward_bench.py \
  --model-path /data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B \
  --slots-per-layer 32 \
  --num-seqs 1 \
  --input-len 64 \
  --output-len 64 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 1 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85 \
  --max-draft-tokens 1 \
  --draft-top-c 0 \
  --draft-reroute-policy entropy_cache_bias \
  --draft-reroute-artifact results/reroute_impl_20260531/offline_profile_20260531_203257.safetensors \
  --draft-cuda-graph-bucket-steps 1 \
  --enforce-eager false \
  --temperature 0.0 \
  --spec-enable-prefetch true \
  --prefetch-runtime-mode draft_segment_indexed \
  --prefetch-runtime-kind predictive \
  --draft-prefetch-segment-size 12 \
  --prefetch-step-budget 4 \
  --prefetch-max-inflight 8 \
  --prefetch-staging-slots-per-layer 0 \
  --prefetch-verify-wait-ms 0 \
  --engine-profile-cuda-sync true \
  --repeats 1 \
  --dist-port-base 30100 \
  --raw-output-dir results/draft_profile_20260605/raw_ratio25 \
  --output results/draft_profile_20260605/baseline_ratio25.json
'
```

Expected: standard and draft CUDA Graph replay counts are positive and deterministic digests match.

- [ ] **Step 3: Run cache-ratio 0.5 baseline**

Repeat Step 2 with:

```text
--slots-per-layer 64
--max-draft-tokens 6
--dist-port-base 30120
--raw-output-dir results/draft_profile_20260605/raw_ratio50
--output results/draft_profile_20260605/baseline_ratio50.json
```

Expected: CUDA Graph and deterministic correctness checks pass.

- [ ] **Step 4: Append baseline evidence to the live report**

Record:

- Git revision and dirty files.
- Exact commands and log paths.
- Standard decode and draft-forward latency.
- Draft graph replay, `run_draft_core_run`, prepare, sampler, mode-set, and prefetch-before latency.
- Metadata collect/observe/submit timings.
- Candidate counts and ranking time.
- Existing submit/completed/published/consumed counts divided by draft calls.

- [ ] **Step 5: Commit only the baseline report update**

```bash
git add scripts/draft_profile/draft_forward_optimization_report_20260605.md
git commit -m "docs: record draft forward baseline"
```

### Task 2: Add Per-Forward Expert And Transfer Metrics

**Files:**
- Modify: `nanovllm/expert/prefetcher.py`
- Modify: `examples/benchmarks/draft_standard_decode_forward_bench.py`
- Modify: `tests/test_prefetch_runtime.py`
- Modify: `tests/test_draft_standard_decode_forward_bench.py`

- [ ] **Step 1: Write failing runtime counter tests**

Add tests that submit and publish one segment-indexed expert and assert:

```python
prof["prefetch_submitted_bytes"] == expected_bytes
prof["prefetch_completed_bytes"] == expected_bytes
prof["prefetch_published_bytes"] == expected_bytes
prof["draft_segment_indexed_prefetch_submitted_bytes"] == expected_bytes
prof["draft_segment_indexed_prefetch_completed_bytes"] == expected_bytes
prof["draft_segment_indexed_prefetch_published_bytes"] == expected_bytes
prof["prefetch_max_inflight_observed"] >= 1
```

`expected_bytes` is:

```python
weights["gate_up"].numel() * weights["gate_up"].element_size() + \
weights["down"].numel() * weights["down"].element_size()
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python -m unittest \
  tests.test_prefetch_runtime.TestPrefetchRuntime.test_draft_segment_indexed_transfer_byte_counters
```

Expected: failure because byte counters do not exist.

- [ ] **Step 3: Extend `PrefetchTicket` and submission accounting**

Add immutable ticket fields:

```python
num_bytes: int = 0
transfer_stream_idx: int = 0
```

Add a helper:

```python
@staticmethod
def _expert_weight_bytes(weights: dict[str, torch.Tensor]) -> int:
    return sum(
        int(weights[name].numel()) * int(weights[name].element_size())
        for name in ("gate_up", "down")
    )
```

At every prefetch submission:

- store `num_bytes` in the ticket;
- increment total and source-specific submitted bytes;
- update `prefetch_max_inflight_observed`.

At ready/publication:

- increment completed bytes once when the ticket event first becomes ready;
- increment published bytes only after cache mapping commit succeeds;
- increment cancelled/late bytes if commit fails.

- [ ] **Step 4: Expose counters from `get_profile()`**

Return:

```text
prefetch_submitted_bytes
prefetch_completed_bytes
prefetch_published_bytes
prefetch_late_bytes
prefetch_max_inflight_observed
draft_segment_indexed_prefetch_submitted_bytes
draft_segment_indexed_prefetch_completed_bytes
draft_segment_indexed_prefetch_published_bytes
```

- [ ] **Step 5: Run runtime tests and verify GREEN**

Run:

```bash
python -m unittest tests/test_prefetch_runtime.py tests/test_expert_cache_staging.py
```

Expected: all tests pass.

- [ ] **Step 6: Write failing benchmark extraction tests**

Add a benchmark fixture with `draft_calls=4`, `submit_count=8`, and
`submitted_bytes=4096`. Assert:

```python
breakdown["prefetch_submitted_experts_per_forward"] == 2.0
breakdown["prefetch_submitted_bytes_per_forward"] == 1024.0
breakdown["prefetch_completed_experts_per_forward"] == expected
breakdown["prefetch_published_experts_per_forward"] == expected
```

- [ ] **Step 7: Run benchmark test and verify RED**

```bash
python -m unittest \
  tests.test_draft_standard_decode_forward_bench.TestDraftStandardDecodeForwardBench.test_extract_prefetch_per_forward_metrics
```

Expected: missing per-forward keys.

- [ ] **Step 8: Implement benchmark extraction**

Add per-draft-forward values under `draft_forward.profile_breakdown` and include
their medians in `summarize_repeats()`. Use `spec_run_draft_calls` as the only
denominator.

- [ ] **Step 9: Run focused tests**

```bash
python -m unittest \
  tests/test_draft_standard_decode_forward_bench.py \
  tests/test_prefetch_runtime.py \
  tests/test_expert_cache_staging.py
```

Expected: all tests pass.

- [ ] **Step 10: Commit metric instrumentation**

```bash
git add \
  nanovllm/expert/prefetcher.py \
  examples/benchmarks/draft_standard_decode_forward_bench.py \
  tests/test_prefetch_runtime.py \
  tests/test_draft_standard_decode_forward_bench.py
git commit -m "perf: measure draft prefetch transfer volume"
```

### Task 3: Add A Reproducible Profile Matrix Runner

**Files:**
- Create: `scripts/draft_profile/run_draft_forward_profile.py`
- Test: `tests/test_draft_profile_runner.py`

- [ ] **Step 1: Write failing command-generation tests**

Test that ratio `0.25` produces 32 slots and draft K 1, while ratio `0.5`
produces 64 slots and draft K 6. Test that every command uses `num_seqs=1`,
CUDA Graph, predictive segment-indexed prefetch, and an explicit stream count.

- [ ] **Step 2: Verify RED**

```bash
python -m unittest tests/test_draft_profile_runner.py
```

Expected: module does not exist.

- [ ] **Step 3: Implement the runner**

The script accepts:

```text
--cache-ratios
--output-len
--repeats
--segment-sizes
--transfer-stream-counts
--dist-port-base
--output-dir
```

It calls `draft_standard_decode_forward_bench.py`, saves one JSON per case, and
writes `summary.json` containing command strings and selected metrics.

- [ ] **Step 4: Verify GREEN**

```bash
python -m unittest tests/test_draft_profile_runner.py
```

Expected: all tests pass without loading the model.

- [ ] **Step 5: Commit the runner**

```bash
git add scripts/draft_profile/run_draft_forward_profile.py tests/test_draft_profile_runner.py
git commit -m "bench: add draft forward profile matrix runner"
```

### Task 4: Optimize Segment Candidate Selection

**Files:**
- Modify: `nanovllm/expert/prefetcher.py`
- Modify: `tests/test_prefetch_runtime.py`

- [ ] **Step 1: Confirm baseline evidence supports this task**

Proceed only when baseline `draft_segment_indexed_rank_ms` or candidate scan
accounts for a material fraction of draft worker turnaround. Otherwise record
the skipped task in the report and continue to Task 5.

- [ ] **Step 2: Write an equivalence test**

Build overlapping long-term and draft segment indexes with:

- duplicate candidates with different priorities;
- cached candidates;
- pending candidates;
- expired candidates;
- enough valid candidates to exceed dispatch budget.

Assert the optimized helper returns the same ordered first `dispatch_budget`
eligible keys as the current full merge-and-sort implementation.

- [ ] **Step 3: Verify RED**

Run the new test and confirm the optimized helper is missing.

- [ ] **Step 4: Implement bounded two-index selection**

Add:

```python
def _rank_segment_candidates(
    self,
    *,
    segment_id: int,
    step_id: int,
    inflight_keys: set[tuple[int, int]],
    candidate_limit: int,
) -> list[PrefetchCandidate]:
```

The helper:

1. Filters stale/cached/pending candidates in each index.
2. Uses `heapq.nlargest` with deterministic key
   `(priority, -layer_idx, -expert_idx)` when the candidate set is larger than
   `candidate_limit`.
3. Merges duplicate `(layer, expert)` keys using the higher priority.
4. Returns deterministic descending priority order.

Use a conservative limit:

```python
candidate_limit = max(dispatch_budget * 8, 32)
```

If fewer than `dispatch_budget` candidates survive weight or victim checks,
fall back once to the full ranking path and record
`draft_segment_indexed_rank_fallback_count`.

- [ ] **Step 5: Run focused tests**

```bash
python -m unittest tests/test_prefetch_runtime.py
```

Expected: full/optimized equivalence and existing lifecycle tests pass.

- [ ] **Step 6: Benchmark ratios 0.25 and 0.5**

Use the Task 3 runner with one transfer stream, segment size 12, output length
64, and three repeats.

Expected retain rule: lower candidate ranking/worker turnaround with no
deterministic mismatch and no draft-forward regression.

- [ ] **Step 7: Keep or revert this optimization**

If retained, commit:

```bash
git add nanovllm/expert/prefetcher.py tests/test_prefetch_runtime.py
git commit -m "perf: bound draft segment candidate ranking"
```

If rejected, remove only Task 4 production/test changes and document the
measured regression.

### Task 5: Implement And Measure The CUDA Transfer Stream Pool

**Files:**
- Modify: `nanovllm/config.py`
- Modify: `nanovllm/expert/prefetcher.py`
- Modify: `examples/heterogeneous_benchmark_case.py`
- Modify: `examples/benchmarks/draft_standard_decode_forward_bench.py`
- Modify: `tests/test_config_prefetch.py`
- Modify: `tests/test_prefetch_runtime.py`
- Modify: `tests/test_draft_standard_decode_forward_bench.py`

- [ ] **Step 1: Write failing configuration tests**

Assert default stream count is 1, counts 2 and 4 are accepted, and 0 is
rejected.

- [ ] **Step 2: Verify RED**

```bash
python -m unittest \
  tests.test_config_prefetch.TestConfigPrefetch.test_prefetch_transfer_stream_count
```

Expected: `Config` has no such field.

- [ ] **Step 3: Add configuration and CLI plumbing**

Add:

```python
prefetch_transfer_stream_count: int = 1
```

Validate `>= 1`, expose
`--prefetch-transfer-stream-count`, and pass it through both benchmark layers.

- [ ] **Step 4: Write failing stream-selection tests**

Patch `torch.cuda.is_available` and `torch.cuda.Stream` with fake stream
objects. Assert:

- count 1 always selects stream 0;
- count 2 selects `0, 1, 0, 1`;
- both tensors of one expert use the same selected stream;
- ticket stores its stream index;
- publication still waits on the ticket event, not a global stream.

- [ ] **Step 5: Verify RED**

Run the new runtime tests and confirm the pool helper is missing.

- [ ] **Step 6: Implement fixed stream-pool ownership**

Replace the single stream field with:

```python
self.transfer_streams = [
    torch.cuda.Stream()
    for _ in range(int(config.prefetch_transfer_stream_count))
] if torch.cuda.is_available() else [None]
self._next_transfer_stream_idx = 0
```

Add:

```python
def _acquire_transfer_stream(self) -> tuple[int, torch.cuda.Stream | None]:
    idx = self._next_transfer_stream_idx
    self._next_transfer_stream_idx = (idx + 1) % len(self.transfer_streams)
    return idx, self.transfer_streams[idx]
```

Acquire exactly once per ticket, pass that stream to
`begin_async_put_to_active` or staging copy, and store its index. Keep the
existing per-ticket ready event and active-slot pending reservation as the
safety boundary.

- [ ] **Step 7: Add per-stream profile counters**

Expose dictionaries with string keys:

```text
prefetch_submit_count_by_stream
prefetch_submitted_bytes_by_stream
prefetch_completed_count_by_stream
prefetch_completed_bytes_by_stream
```

Reset all per-stream dictionaries in `get_profile(reset=True)`.

- [ ] **Step 8: Run focused tests**

```bash
python -m unittest \
  tests/test_config_prefetch.py \
  tests/test_prefetch_runtime.py \
  tests/test_expert_cache_staging.py \
  tests/test_draft_standard_decode_forward_bench.py
```

Expected: all tests pass.

- [ ] **Step 9: Run stream count 1/2/4 A100 matrix**

For cache ratios 0.25 and 0.5:

```text
num_seqs=1
output_len=64
segment_size=12
stream_count=1,2,4
repeats=3
```

Record draft-forward latency, completed/published experts per forward,
completed/published bytes per forward, effective GB/s, exposed drain time,
verify latency, and output throughput.

- [ ] **Step 10: Keep the best stream count or keep count 1**

Retain count greater than one only if it increases completed/published bytes
within the draft window without a repeatable draft, verify, or throughput
regression. The default remains 1 unless both cache ratios support the same
larger count.

- [ ] **Step 11: Commit retained stream-pool implementation**

```bash
git add \
  nanovllm/config.py \
  nanovllm/expert/prefetcher.py \
  examples/heterogeneous_benchmark_case.py \
  examples/benchmarks/draft_standard_decode_forward_bench.py \
  tests/test_config_prefetch.py \
  tests/test_prefetch_runtime.py \
  tests/test_draft_standard_decode_forward_bench.py
git commit -m "perf: add expert prefetch transfer stream pool"
```

### Task 6: Segment And Budget Sweep

**Files:**
- Modify: `scripts/draft_profile/draft_forward_optimization_report_20260605.md`
- Output: `results/draft_profile_20260605/segment_budget_summary.json`

- [ ] **Step 1: Run segment-size controls**

For cache ratios 0.25 and 0.5, run segment sizes 12, 24, and 48 using the
retained stream count.

- [ ] **Step 2: Run budget controls**

For the best segment size, test:

```text
(prefetch_step_budget, prefetch_max_inflight) =
(4, 8), (8, 16), (16, 32)
```

Keep visible transfer budget constant first. Increase it only if budget-stop
counters prevent additional useful copies.

- [ ] **Step 3: Select the retained settings**

Choose settings by:

1. deterministic correctness;
2. draft-forward latency;
3. published bytes and experts per forward;
4. verify and end-to-end throughput.

- [ ] **Step 4: Record all commands and decisions in the live report**

Include rejected variants and the reason each was rejected.

### Task 7: Final 512-Token Verification

**Files:**
- Modify: `scripts/draft_profile/draft_forward_optimization_report_20260605.md`
- Output: `results/draft_profile_20260605/final_ratio25.json`
- Output: `results/draft_profile_20260605/final_ratio50.json`

- [ ] **Step 1: Run the full focused test suite**

```bash
python -m unittest \
  tests/test_config_prefetch.py \
  tests/test_prefetch_runtime.py \
  tests/test_expert_cache_staging.py \
  tests/test_draft_standard_decode_forward_bench.py \
  tests/test_draft_profile_runner.py \
  tests/test_model_runner_prefetch.py \
  tests/test_prefetch_runtime_meta.py
```

Expected: all tests pass.

- [ ] **Step 2: Run final ratio 0.25 benchmark**

Use `num_seqs=1`, `output_len=512`, three independent repetitions, CUDA Graph,
and the retained segment/budget/stream settings.

- [ ] **Step 3: Run final ratio 0.5 benchmark**

Use the same settings except 64 slots per layer and K 6.

- [ ] **Step 4: Compare baseline and final**

Report:

- standard decode forward median;
- draft forward median and ratio to standard;
- every draft timing component;
- experts submitted/completed/published/consumed per forward;
- bytes submitted/completed/published per forward;
- per-stream distribution;
- H2D effective bandwidth and exposed drain;
- verify latency and output throughput;
- deterministic digest alignment.

- [ ] **Step 5: Update report status and commit**

Set report status to completed or explicitly list remaining blockers.

```bash
git add scripts/draft_profile/draft_forward_optimization_report_20260605.md
git commit -m "docs: report draft forward optimization results"
```

