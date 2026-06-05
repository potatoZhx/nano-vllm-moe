# Draft Forward And Prefetch Optimization Report

Date: 2026-06-05

Status: baseline completed; instrumentation in progress

## 1. Objectives

This work compares:

- Standard decode with all model weights resident on GPU and CUDA Graph enabled.
- Speculative `draft_single_forward` with heterogeneous expert caching and draft CUDA Graph enabled.
- Expert prefetch work launched during each draft forward.

The target cache ratios are `0.25` and `0.5`, following
`scripts/small_bench.py`.

Primary goals:

1. Break down the current approximately 20 ms draft-forward latency.
2. Reduce draft-forward overhead toward standard CUDA Graph decode latency.
3. Measure how many experts are submitted, transferred, published, and consumed
   per draft forward.
4. Break down the prefetch path that overlaps GPU computation.
5. Increase useful expert transfer volume during the draft window without
   changing deterministic output.

## 2. Runtime Environment

Requested Slurm allocation:

```text
jobid: 29629
partition: A100
node: gpu15
host: gpu15-A100-E2-3U
CUDA_VISIBLE_DEVICES: 2
GPU: NVIDIA A100-SXM4-80GB
conda environment: nano_moe
torch: 2.9.1+cu128
```

Initial GPU inspection:

```text
visible physical GPU 2: 0 MiB used, 0% utilization
torch.cuda.is_available(): True
torch.cuda.device_count(): 1
```

All compute-node commands must use:

```bash
srun --jobid=29629 --ntasks=1 ...
```

All command output will be captured under:

```text
/home/mumura/moe_spec/logs/
```

## 3. Existing Measurement Infrastructure

The repository already contains:

- `scripts/small_bench.py`
- `examples/benchmarks/draft_standard_decode_forward_bench.py`
- `examples/benchmarks/profile_single_draft.py`
- `examples/benchmarks/draft_step_breakdown.py`
- `tests/test_draft_standard_decode_forward_bench.py`

The existing draft/standard benchmark already reports:

- Standard decode forward latency and CUDA Graph replay count.
- Draft forward latency and CUDA Graph replay count.
- Route, plan, GPU compute, sampler, and model-run timings.
- Draft metadata enqueue, wait, collect, observe, and queue-update timings.
- Prefetch worker turnaround and hidden/exposed overlap.
- Candidate ranking and victim-selection counters.
- Prefetch submit, completion, publication, and consumption counters.

The benchmark will be extended instead of creating a separate timing
definition.

## 4. Current Draft Path

The current draft path is:

```text
ModelRunner.run_draft
  flush ready metadata without blocking
  set speculative draft mode
  begin draft prefetch iteration
  publish ready direct-active transfers
  optionally submit predictive phase-1 prefetch
  wait for metadata buffer reuse
  arm runtime metadata recorder
  run decode preparation
  replay segmented draft CUDA Graph
    replay segment
    enqueue segment routing metadata D2H
  compute logits
  sample token
  enqueue remaining metadata
  restore normal execution mode
```

The asynchronous path is:

```text
metadata stream
  D2H routing metadata

metadata worker
  wait for D2H event
  collect host metadata
  update segment candidate indexes
  rank and filter candidates
  select victim slots
  enqueue expert H2D copies

expert transfer stream
  copy gate_up and down expert weights

later safe boundary
  query transfer event
  publish expert-to-slot mapping
```

## 5. Baseline Matrix

### 5.1 Functional matrix

| Mode | Cache ratio | Prefetch | CUDA Graph |
|---|---:|---:|---:|
| standard | 1.00 | off | on |
| spec | 0.25 | on | draft on |
| spec | 0.50 | on | draft on |

Standard mode must load all experts into GPU memory. Spec mode converts the
ratio to `heterogeneous_slots_per_layer` consistently with
`scripts/small_bench.py`.

### 5.2 Load

All experiments use:

```text
num_seqs=1
```

This isolates single-request fixed overhead, CUDA Graph replay cost, metadata
worker latency, and the number of expert transfers that can finish within one
draft-forward window.

Initial diagnosis uses a short output length. Final retained changes are
retested with `output_len=512`.

### 5.3 Repetition and statistics

- Use at least three fresh process repetitions for final comparisons.
- Report median and individual values.
- Exclude model initialization and explicit warmup from timed counters.
- Preserve CUDA Graph replay validation.
- Use `temperature=0` for deterministic standard/spec alignment.
- Record GPU memory and utilization before every benchmark group.

## 6. Required New Metrics

The following counters will be added or exposed in the benchmark report:

### 6.1 Per-draft-forward expert counts

For each prefetch source and in total:

- submitted experts per draft forward
- completed experts per draft forward
- successfully published experts per draft forward
- consumed experts per draft forward
- late or cancelled experts per draft forward
- unique `(layer, expert)` pairs per draft forward

The denominator is `spec_run_draft_calls`, not speculative outer steps.

### 6.2 Transfer volume

- submitted bytes per draft forward
- completed bytes per draft forward
- published bytes per draft forward
- estimated transfer milliseconds per draft forward
- actual event-based transfer latency distribution where measurable
- effective GB/s for completed transfers

### 6.3 Pipeline timing

- candidate index scan time
- merge/sort time
- cache and pending filtering time
- victim selection time
- slot reservation time
- H2D enqueue CPU time
- H2D event completion latency
- publish scan and mapping update time
- metadata D2H transfer wait
- metadata collect and compacting time
- worker queue wait and end-to-end turnaround
- time hidden behind draft GPU execution
- exposed wait on the next draft or verify boundary

### 6.4 Concurrency

- in-flight expert count over time
- maximum simultaneous transfer tickets
- transfer submissions per segment boundary
- transfer completions before the draft GPU window closes
- per-transfer-stream submitted bytes and completion count

## 7. Optimization Strategy

Optimizations are applied one at a time. Each change must pass deterministic
correctness checks before performance measurement.

### 7.1 Candidate ranking critical path

Current evidence indicates that segment-indexed prefetch repeatedly scans and
sorts a much larger candidate set than the dispatch budget.

Candidate approaches:

1. Maintain a dirty, pre-sorted candidate cache per segment.
2. Merge only the bounded top candidates needed to satisfy dispatch and
   filtering slack.
3. Move sorting from submit time into metadata observation when doing so does
   not extend the main draft path.

The selected implementation must preserve the same ordering rules and avoid
silently dropping a candidate that could become eligible after cache or
pending filtering.

### 7.2 Metadata compacting

The current histogram path scans nonzero experts on CPU after D2H metadata
collection. If this remains significant, test a compact GPU-produced candidate
representation so the CPU worker receives only active expert entries.

This is a second-stage optimization because it changes the metadata contract
and has a larger correctness surface.

### 7.3 Segment boundary count

Test segment sizes that reduce Python and metadata enqueue frequency while
retaining enough draft compute after each boundary to overlap H2D:

```text
segment size: 12, 24, 48
```

`48` is effectively one full-draft boundary and serves as the low-overhead,
low-overlap control.

### 7.4 Prefetch budget and buffer reuse

Measure and tune:

- metadata host buffer pool size
- `prefetch_step_budget`
- `prefetch_max_inflight`
- visible transfer budget
- minimum and maximum submissions per boundary

Budget increases are retained only when they increase completed or consumed
experts without increasing draft latency or unsafe slot contention.

## 8. CUDA Transfer Stream Pool Experiment

### 8.1 Hypothesis

The current expert transfer path uses one CUDA transfer stream. Gate/up and
down copies for different experts are therefore ordered on that stream.

A transfer stream pool may increase the number of experts completed during a
draft window when:

- separate H2D copies can be concurrently scheduled;
- host memory is pinned;
- copies target independent active-cache slots;
- PCIe or NVLink copy engines are not already saturated;
- CPU submission and candidate ranking are no longer the dominant bottleneck.

Multiple streams cannot exceed the physical H2D bandwidth and may regress
performance through extra events, contention, or smaller fragmented copies.
It is therefore an experiment, not an assumed optimization.

### 8.2 Variants

```text
transfer stream count: 1, 2, 4
```

The one-stream variant is the control.

### 8.3 Proposed ownership

- Create a fixed transfer stream pool during prefetch runtime initialization.
- Select a stream by a stable round-robin ticket index or `(layer, expert)`
  hash.
- Keep both expert tensors for one ticket on the same stream.
- Record one ready event after all copies for that ticket.
- Publish the active-cache mapping only after that ticket's event is ready.
- Never allow two pending tickets to write the same active slot.
- Preserve current frontier and stale-step guards.

### 8.4 Correctness hazards

- Two streams writing the same victim slot.
- Publishing a mapping after only one tensor copy completes.
- Reusing a slot while an older stream still owns it.
- Draining one stream while ignoring outstanding work on another.
- CUDA Graph replay reading a slot before a transfer and deferred mapping are
  complete.
- Host pinned-memory or copy-engine bandwidth becoming the real bottleneck.

Focused tests must cover these cases before A100 benchmarking.

### 8.5 Stream-pool decision rule

Retain a stream-pool size greater than one only if repeated measurements show:

- exact deterministic output alignment;
- no slot-generation or pending-state failures;
- no draft-forward latency regression beyond normal noise;
- higher completed or published experts per draft forward;
- higher completed transfer bytes during the measured draft window;
- no equivalent regression in verify latency or total output throughput.

## 9. Correctness Gates

For every retained optimization:

1. Run focused unit tests for the modified prefetch/runtime component.
2. Verify CUDA Graph replay counts are nonzero.
3. Compare standard and spec token digests with `temperature=0`.
4. Exercise cache ratios `0.25` and `0.5`.
5. Exercise the requested single-request load, `num_seqs=1`.
6. Check that prefetch publication never exposes an incomplete expert.
7. Confirm all asynchronous workers drain cleanly at engine shutdown.

## 10. Performance Decision Rules

An optimization is retained when:

- correctness passes;
- draft-forward median improves or remains neutral;
- the result is repeatable across fresh processes;
- useful prefetch throughput improves when that is the optimization target;
- no material verify or end-to-end regression appears.

An optimization that fails correctness or repeatedly regresses the target
workload will be reverted independently and documented here.

## 11. Planned Commands

Exact commands will be added before each run. The common execution wrapper is:

```bash
TS=$(date +%Y%m%d_%H%M%S)
LOG=/home/mumura/moe_spec/logs/draft_profile_${TS}.log
srun --jobid=29629 --ntasks=1 bash -lc '
source /opt/Software/Anaconda3/etc/profile.d/conda.sh
conda activate nano_moe
export CUDA_VISIBLE_DEVICES=2
cd /home/mumura/moe_spec/nano-vllm-moe
python <benchmark-or-test> <arguments>
' 2>&1 | tee "$LOG"
```

The report will record the expanded command, log path, JSON output path, git
revision, dirty files, and result summary for every benchmark group.

## 12. Experiment Log

### 12.1 Focused Test Baseline

Timestamp: 2026-06-05 14:15 CST

Objective: verify the existing prefetch, cache, config, and benchmark tests
before production-code changes.

Code revision:

```text
0a3b9181f2be019404470b1b31946f9a63750ffe
branch: codex/draft-forward-prefetch-opt
```

Command:

```bash
srun --jobid=29629 --ntasks=1 bash -lc '
source /opt/Software/Anaconda3/etc/profile.d/conda.sh
conda activate nano_moe
export CUDA_VISIBLE_DEVICES=2
cd /home/mumura/moe_spec/nano-vllm-moe
python -m unittest \
  tests/test_config_prefetch.py \
  tests/test_prefetch_runtime.py \
  tests/test_expert_cache_staging.py \
  tests/test_draft_standard_decode_forward_bench.py
'
```

Log:

```text
/home/mumura/moe_spec/logs/draft_profile_tests_20260605_141552.log
```

Result:

```text
Ran 42 tests in 0.609s
OK
```

Decision: use this as the correctness baseline.

### 12.2 Cache Ratio 0.25 Baseline

Timestamp: 2026-06-05 14:16 CST

Objective: compare single-request standard decode and predictive
segment-indexed draft forward with 32 GPU expert slots per layer.

Command:

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

Log:

```text
/home/mumura/moe_spec/logs/draft_profile_baseline_ratio25_20260605_141619.log
```

Raw result:

```text
results/draft_profile_20260605/baseline_ratio25.json
results/draft_profile_20260605/raw_ratio25/repeat_00_standard.json
results/draft_profile_20260605/raw_ratio25/repeat_00_spec.json
```

Correctness:

```text
deterministic digest match: true
standard graph replays: 63
draft graph replays: 37
draft segment graph replays: 148
```

Performance:

| Metric | Value |
|---|---:|
| standard decode forward | 14.643 ms |
| draft forward | 32.138 ms |
| draft / standard | 2.195x |
| draft graph replay | 21.340 ms/forward |
| draft core run | 23.433 ms/forward |
| draft prefetch-before | 7.990 ms/forward |
| draft mode set | 0.306 ms/forward |
| segment candidate ranking | 0.200 ms/forward |
| segment victim selection | 0.160 ms/forward |
| segment submit visible overhead | 9.153 ms/forward |
| direct-active publication scan/commit | 5.755 ms/forward |
| segment expert submit | 4.676 experts/forward |
| segment expert ready | 4.676 experts/forward |
| segment expert publish | 4.676 experts/forward |
| segment expert consumption observations | 16.541/forward |
| predictive phase-1 submit | 4.000 experts/forward |
| verify-layer submit | 59.892 experts/draft-forward denominator |
| segment candidates scanned | 55.16/forward |

Analysis:

1. The draft-forward gap is `17.495 ms`; `7.990 ms` is directly exposed
   before draft replay.
2. Full candidate ranking and victim selection together are only
   `0.360 ms/forward`, so the previous candidate-sort hypothesis is not the
   primary bottleneck for this single-request workload.
3. Segment submit visible overhead is `9.153 ms/forward`. After subtracting
   ranking and victim selection, most of the unaccounted time is in cache
   checks, reservation, and expert H2D enqueue.
4. CPU expert weights are currently not pinned. A CUDA `copy_(..., non_blocking=True)`
   from pageable host memory can expose host-side staging/synchronization, so
   multiple CUDA streams alone may not remove this overhead.
5. Existing total prefetch counters mix draft segment, predictive phase-1, and
   verify-layer traffic. Source-specific expert and byte counters are required
   before evaluating transfer-stream concurrency.

Decision: retain the baseline and prioritize fine-grained transfer enqueue and
byte accounting before candidate ranking changes.

### 12.3 Cache Ratio 0.50 Baseline

Timestamp: 2026-06-05 14:19 CST

Objective: repeat the comparison with 64 GPU expert slots per layer and the
`small_bench.py` draft budget of six.

Command: same as Section 12.2 with:

```text
--slots-per-layer 64
--max-draft-tokens 6
--dist-port-base 30120
--raw-output-dir results/draft_profile_20260605/raw_ratio50
--output results/draft_profile_20260605/baseline_ratio50.json
```

Log:

```text
/home/mumura/moe_spec/logs/draft_profile_baseline_ratio50_20260605_141922.log
```

Raw result:

```text
results/draft_profile_20260605/baseline_ratio50.json
results/draft_profile_20260605/raw_ratio50/repeat_00_standard.json
results/draft_profile_20260605/raw_ratio50/repeat_00_spec.json
```

Correctness:

```text
deterministic digest match: true
standard graph replays: 63
draft graph replays: 75
```

Performance:

| Metric | Value |
|---|---:|
| standard decode forward | 14.725 ms |
| draft forward | 25.688 ms |
| draft / standard | 1.745x |
| draft graph replay | 17.318 ms/forward |
| draft core run | 21.453 ms/forward |
| draft prefetch-before | 2.057 ms/forward |
| draft mode set | 1.752 ms/forward |
| segment candidate ranking | 0.318 ms/forward |
| segment victim selection | 0.340 ms/forward |
| segment submit visible overhead | 13.659 ms/forward |
| segment direct-active drain | 0.575 ms/forward |
| direct-active publication scan/commit | 1.463 ms/forward |
| segment expert submit/ready/publish | 3.387 experts/forward |
| segment expert consumption observations | 9.613/forward |
| predictive phase-1 submit | 0.693 experts/forward |
| verify-layer submit | 9.000 experts/draft-forward denominator |
| segment candidates scanned | 41.19/forward |

Analysis:

1. Higher cache coverage reduces the draft gap to `10.963 ms`.
2. Candidate ranking and victim selection remain below `0.7 ms/forward`
   combined.
3. Segment submit visible overhead remains large even though fewer experts are
   submitted, which reinforces the need to split CPU enqueue latency from
   actual asynchronous transfer completion.
4. Draft mode-set time varies significantly between ratios and needs its own
   repeated trace before it is treated as a stable optimization target.
5. Ratio 0.25 and 0.50 use different draft-token budgets, so cross-ratio
   differences describe the requested operating points, not a controlled
   single-variable cache experiment.

Decision: retain the baseline. Do not implement bounded candidate ranking until
new timing confirms it becomes material after transfer enqueue is reduced.

### 12.4 Baseline Root-Cause Summary

The first root-cause hypothesis is:

```text
draft gap
  = segmented draft graph replay overhead
  + logits/sampler and Python run wrapper
  + prefetch-before work
  + pageable-host H2D enqueue/staging cost
  + cache publication and mode transition
```

Evidence against candidate ranking as the immediate priority:

```text
ratio 0.25 ranking + victim: 0.360 ms/forward
ratio 0.50 ranking + victim: 0.658 ms/forward
```

Evidence for instrumenting transfer submission:

```text
ratio 0.25 segment visible submit: 9.153 ms/forward
ratio 0.50 segment visible submit: 13.659 ms/forward
```

Next experiment:

1. Add source-specific submitted/completed/published byte counters.
2. Add reservation and H2D enqueue CPU timings.
3. Add actual ticket completion latency and per-forward metrics.
4. Re-run ratio 0.25 and 0.50 before changing stream count.
5. Test transfer stream counts 1, 2, and 4 only after the single-stream path is
   fully accounted.

Each entry will use this format:

```text
Timestamp:
Objective:
Code revision:
Changed files:
Command:
Log:
Raw result:
Correctness:
Performance:
Analysis:
Decision:
```

## 13. Fine-Grained Prefetch Instrumentation

Timestamp: 2026-06-05 14:25-14:32 CST

Objective:

Split the large segment prefetch visible overhead into source-specific expert
counts, transferred bytes, cache reservation time, CPU-side H2D enqueue time,
and ticket completion latency. Normalize the resulting totals by
`spec_run_draft_calls`.

Changed files:

```text
nanovllm/expert/prefetcher.py
examples/benchmarks/draft_standard_decode_forward_bench.py
tests/test_prefetch_runtime.py
tests/test_draft_standard_decode_forward_bench.py
```

Instrumentation added:

1. Submitted, completed, published, and late bytes.
2. Submitted, completed, published, and late counts grouped by source.
3. Maximum observed number of in-flight expert transfers.
4. Cache reservation CPU time.
5. CPU time spent enqueueing expert H2D copies.
6. Submit-to-event-ready completion latency.
7. Draft-only aggregation over `draft_segment_indexed`,
   `draft_direct_active`, and `predictive_phase1`; verify prefetch is excluded.

TDD RED command:

```bash
srun --jobid=29629 --ntasks=1 bash -lc '
  source /opt/Software/Anaconda3/etc/profile.d/conda.sh
  conda activate nano_moe
  export CUDA_VISIBLE_DEVICES=2
  cd /home/mumura/moe_spec/nano-vllm-moe
  python -m unittest \
    tests/test_prefetch_runtime.py \
    tests/test_draft_standard_decode_forward_bench.py
'
```

RED log:

```text
/home/mumura/moe_spec/logs/draft_profile_metrics_red_20260605_142502.log
```

Expected RED failures:

```text
missing prefetch_submitted_bytes profile field
missing benchmark prefetch_submitted_experts_per_forward field
```

Targeted GREEN log:

```text
/home/mumura/moe_spec/logs/draft_profile_metrics_green_20260605_143113.log
```

Result:

```text
Ran 20 tests in 0.595s
OK
```

Regression command:

```bash
srun --jobid=29629 --ntasks=1 bash -lc '
  source /opt/Software/Anaconda3/etc/profile.d/conda.sh
  conda activate nano_moe
  export CUDA_VISIBLE_DEVICES=2
  cd /home/mumura/moe_spec/nano-vllm-moe
  python -m unittest \
    tests/test_config_prefetch.py \
    tests/test_prefetch_runtime.py \
    tests/test_expert_cache_staging.py \
    tests/test_draft_standard_decode_forward_bench.py
'
```

Regression log:

```text
/home/mumura/moe_spec/logs/draft_profile_metrics_regression_20260605_143131.log
```

Result:

```text
Ran 43 tests in 0.616s
OK
```

Analysis:

The new counters preserve the existing submit/publish behavior while exposing
the distinction between work submitted during draft and verify-side work that
was previously mixed into totals. The next runs will determine whether the
visible overhead is dominated by cache reservation, pageable-host copy enqueue,
or waiting for transfer completion.

## 14. Instrumented Pageable-Host Results

### 14.1 Cache Ratio 0.25

Timestamp: 2026-06-05 14:32 CST

Code revision:

```text
26c5a1f perf: instrument draft expert prefetch
```

The command is identical to Section 12.2 except for:

```text
--dist-port-base 30200
--raw-output-dir results/draft_profile_20260605/raw_instrumented_ratio25
--output results/draft_profile_20260605/instrumented_ratio25.json
```

Log:

```text
/home/mumura/moe_spec/logs/draft_profile_instrumented_ratio25_20260605_143236.log
```

Correctness:

```text
deterministic digest match: true
standard graph replays: 63
draft graph replays: 37
late transfer bytes: 0
```

Performance:

| Metric | Value |
|---|---:|
| standard decode forward | 15.064 ms |
| draft forward | 34.710 ms |
| draft / standard | 2.304x |
| draft graph replay | 21.626 ms/forward |
| draft prefetch-before | 7.095 ms/forward |
| segment submit-after | 18.644 ms/forward |
| segment reservation | 0.034 ms/forward |
| segment H2D enqueue | 17.107 ms/forward |
| segment completion latency sum | 57.480 ms/forward |
| all draft-source submitted/completed/published | 7.378 experts/forward |
| all draft-source submitted/completed/published bytes | 69.631 MB/forward |
| segment submitted experts | 3.378 experts/forward |
| predictive phase-1 submitted experts | 4.000 experts/forward |
| max observed in-flight transfers | 8 |

The completion-latency value is the sum of individual ticket latencies divided
by draft calls, not a wall-clock critical-path duration. For segment tickets,
the mean submit-to-ready latency is approximately:

```text
57.480 / 3.378 = 17.02 ms/expert
```

Analysis:

1. Active-slot reservation is negligible.
2. Segment H2D enqueue alone is `49.3%` of measured draft-forward wall time.
3. Each expert is `9 MiB`; the draft path submits about `66.4 MiB/forward`.
4. The source expert tensors are pageable because
   `cpu_expert_pin_memory=False` in this benchmark path.
5. CUDA `non_blocking=True` does not make pageable-host copies fully
   asynchronous; runtime staging is exposed in the enqueue call.
6. Adding streams before fixing host memory registration cannot remove this
   CPU-side enqueue cost.

### 14.2 Cache Ratio 0.50

Timestamp: 2026-06-05 14:36 CST

The command is identical to Section 12.3 except for:

```text
--dist-port-base 30220
--raw-output-dir results/draft_profile_20260605/raw_instrumented_ratio50
--output results/draft_profile_20260605/instrumented_ratio50.json
```

Log:

```text
/home/mumura/moe_spec/logs/draft_profile_instrumented_ratio50_20260605_143615.log
```

Correctness:

```text
deterministic digest match: true
standard graph replays: 63
draft graph replays: 66
late transfer bytes: 0
```

Performance:

| Metric | Value |
|---|---:|
| standard decode forward | 14.695 ms |
| draft forward | 22.732 ms |
| draft / standard | 1.547x |
| draft graph replay | 16.972 ms/forward |
| draft prefetch-before | 1.915 ms/forward |
| segment submit-after | 9.864 ms/forward |
| all draft-source H2D enqueue | 9.362 ms/forward |
| segment H2D enqueue | 8.580 ms/forward |
| predictive phase-1 H2D enqueue | 0.781 ms/forward |
| segment reservation | 0.029 ms/forward |
| all draft-source completion latency sum | 66.971 ms/forward |
| all draft-source submitted/completed/published | 5.000 experts/forward |
| all draft-source submitted/completed/published bytes | 47.186 MB/forward |
| segment submitted experts | 4.333 experts/forward |
| predictive phase-1 submitted experts | 0.667 experts/forward |
| max observed in-flight transfers | 8 |
| candidate rank + victim selection | 0.823 ms/forward |

The mean submit-to-ready latency across draft-source tickets is:

```text
66.971 / 5.000 = 13.39 ms/expert
```

Analysis:

1. Draft-source enqueue consumes `41.2%` of draft-forward wall time.
2. The measured gap to standard decode is `8.036 ms`, while enqueue is
   `9.362 ms`; part of enqueue is overlapped, but it is still the dominant
   controllable overhead.
3. Candidate ranking and victim selection remain secondary.
4. Both ratios complete and publish every submitted expert, so the immediate
   throughput limit is submission cost rather than ticket cancellation.

Decision:

1. Do not implement bounded candidate ranking at this point.
2. Expose `cpu_expert_pin_memory` in this benchmark and test it before adding
   multiple streams.
3. Test stream counts `1/2/4` only on a pinned-host path; otherwise the
   experiment measures pageable staging rather than CUDA copy concurrency.

## 15. Pinned-Host Control

### 15.1 Cache Ratio 0.50, One Transfer Stream

Timestamp: 2026-06-05 14:40 CST

Code revision:

```text
0c1b21f perf: expose pinned expert prefetch profiling
```

Command change relative to Section 14.2:

```text
--cpu-expert-pin-memory true
--dist-port-base 30240
--raw-output-dir results/draft_profile_20260605/raw_pinned_ratio50_stream1
--output results/draft_profile_20260605/pinned_ratio50_stream1.json
```

Log:

```text
/home/mumura/moe_spec/logs/draft_profile_pinned_ratio50_stream1_20260605_144037.log
```

Correctness:

```text
deterministic digest match: true
standard graph replays: 63
draft graph replays: 71
late transfer bytes: 0
```

Performance comparison:

| Metric | Pageable, stream 1 | Pinned, stream 1 | Change |
|---|---:|---:|---:|
| standard decode forward | 14.695 ms | 14.486 ms | -1.4% |
| draft forward | 22.732 ms | 21.347 ms | -6.1% |
| draft / standard | 1.547x | 1.474x | improved |
| draft graph replay | 16.972 ms | 16.736 ms | -1.4% |
| draft prefetch-before | 1.915 ms | 1.511 ms | -21.1% |
| segment submit-after | 9.864 ms | 3.095 ms | -68.6% |
| all draft-source H2D enqueue | 9.362 ms | 1.505 ms | -83.9% |
| segment H2D enqueue | 8.580 ms | 1.427 ms | -83.4% |
| submitted/completed/published experts | 5.000 | 7.549 | +51.0% |
| submitted/completed/published bytes | 45.0 MiB | 67.9 MiB | +51.0% |
| verify forward | 382.236 ms | 230.617 ms | -39.7% |

Pinned source detail:

```text
predictive_phase1:       48 experts, 0.078 ms enqueue/draft-forward
draft_segment_indexed:  488 experts, 1.427 ms enqueue/draft-forward
verify_layer_predict:   380 experts, excluded from draft-source aggregate
```

Analysis:

1. Full expert pinning removes almost all pageable staging cost and makes H2D
   submission genuinely asynchronous.
2. Reduced enqueue time allows the adaptive segment path to submit and complete
   more experts within the same run.
3. Draft latency improves by less than the enqueue reduction because much of
   the asynchronous worker time was already overlapped with graph replay, and
   the optimized run intentionally transfers 51% more data.
4. The remaining draft gap to standard decode is `6.861 ms`; draft graph replay
   itself remains `3.383 ms` slower than standard graph replay, and wrapper,
   mode-set, prefetch-before, and segment-boundary handling account for most of
   the rest.
5. The one-time model startup cost increases because the CPU expert pool is
   pinned. It is not part of forward latency, but it is an operational tradeoff
   and consumes pinned host memory.

Decision: retain explicit pinned-memory support for the benchmark and use this
run as the stream-pool control.

## 16. Transfer Stream Pool Implementation

Timestamp: 2026-06-05 14:44-14:46 CST

Objective:

Add an opt-in fixed transfer stream pool while preserving one event per expert
ticket and deferred cache publication.

Implementation:

1. Add `Config.prefetch_transfer_stream_count`, default `1`, minimum `1`.
2. Create all transfer streams once during `PrefetchRuntime` initialization.
3. Assign tickets round-robin and keep both expert tensor copies on the same
   stream.
4. Keep `transfer_stream` as an alias for stream 0 for compatibility.
5. Publish a cache mapping only after that ticket's event reports ready.
6. Export submitted experts, bytes, enqueue time, and completion latency by
   stream.
7. Expose `--prefetch-transfer-stream-count` through both benchmark CLIs.

RED log:

```text
/home/mumura/moe_spec/logs/draft_profile_stream_pool_red_20260605_144431.log
```

Expected failures:

```text
Config field missing
CLI arguments missing
round-robin helper missing
benchmark stream metrics missing
```

The first GREEN attempt found a backward-compatibility issue in tests and
legacy callers that construct config-like `SimpleNamespace` objects:

```text
AttributeError: prefetch_transfer_stream_count
```

Compatibility fix:

```python
getattr(config, "prefetch_transfer_stream_count", 1)
```

Final test command:

```bash
srun --jobid=29629 --ntasks=1 bash -lc '
  source /opt/Software/Anaconda3/etc/profile.d/conda.sh
  conda activate nano_moe
  export CUDA_VISIBLE_DEVICES=2
  cd /home/mumura/moe_spec/nano-vllm-moe
  python -m unittest \
    tests/test_config_prefetch.py \
    tests/test_predictive_prefetch_cli_args.py \
    tests/test_prefetch_runtime.py \
    tests/test_expert_cache_staging.py \
    tests/test_draft_standard_decode_forward_bench.py
'
```

GREEN log:

```text
/home/mumura/moe_spec/logs/draft_profile_stream_pool_green_retry_20260605_144613.log
```

Result:

```text
Ran 48 tests in 2.630s
OK
```

The stream-count default remains one. Counts greater than one are retained only
if the A100 pinned-host experiments improve completed bytes or experts per
draft forward without correctness or draft-latency regression.

## 17. Pinned Transfer Stream Matrix, Cache Ratio 0.50

Timestamp: 2026-06-05 14:47-14:52 CST

Common configuration:

```text
num_seqs=1
slots_per_layer=64
max_draft_tokens=6
cpu_expert_pin_memory=true
draft_prefetch_segment_size=12
prefetch_step_budget=4
prefetch_max_inflight=8
```

Logs:

```text
stream 1: /home/mumura/moe_spec/logs/draft_profile_pinned_ratio50_stream1_20260605_144037.log
stream 2: /home/mumura/moe_spec/logs/draft_profile_pinned_ratio50_stream2_20260605_144707.log
stream 4: /home/mumura/moe_spec/logs/draft_profile_pinned_ratio50_stream4_20260605_145011.log
```

Raw reports:

```text
results/draft_profile_20260605/pinned_ratio50_stream1.json
results/draft_profile_20260605/pinned_ratio50_stream2.json
results/draft_profile_20260605/pinned_ratio50_stream4.json
```

All three runs:

```text
deterministic standard/spec digest match: true
standard CUDA Graph replay: positive
draft CUDA Graph replay: positive
late transfer bytes: 0
```

Results:

| Streams | Standard ms | Draft ms | Draft/standard | Draft graph ms | Experts/draft | MiB/draft | Enqueue ms/draft | Verify ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 14.486 | 21.347 | 1.474x | 16.736 | 7.549 | 67.94 | 1.505 | 230.617 |
| 2 | 14.616 | 20.816 | 1.424x | 16.716 | 7.394 | 66.55 | 1.314 | 219.546 |
| 4 | 14.661 | 21.321 | 1.454x | 16.733 | 6.843 | 61.59 | 1.358 | 218.247 |

Stream distribution:

```text
2 streams: 415 / 414 tickets
4 streams: 201 / 201 / 201 / 200 tickets
```

Analysis:

1. Round-robin assignment is balanced.
2. Two streams reduce draft latency by `0.531 ms` relative to the pinned
   single-stream control and reduce enqueue by `0.191 ms/forward`.
3. Four streams regress draft latency by `0.505 ms` relative to two streams and
   transfer fewer experts per draft forward.
4. Multiple streams do not increase transferred expert count under the current
   candidate supply, `prefetch_step_budget=4`, and `prefetch_max_inflight=8`.
5. The A100 H2D engine and PCIe bandwidth are shared; adding streams beyond two
   cannot create additional physical copy bandwidth and adds scheduling/event
   pressure.
6. Expert/forward varies with the cache-dependent speculative trajectory, so
   these single-process runs are directional rather than a low-noise statistical
   estimate. The deterministic final token digest remains exact.

Decision:

- Retain the configurable pool.
- Use `prefetch_transfer_stream_count=2` as the latency-oriented setting.
- Keep the production default at `1` until repeated deployment-level data
  confirms the approximately `0.5 ms` gain.
- Do not use four streams.

## 18. Optimized Cache Ratio 0.25 Validation

Timestamp: 2026-06-05 14:53 CST

Configuration:

```text
slots_per_layer=32
max_draft_tokens=1
num_seqs=1
cpu_expert_pin_memory=true
prefetch_transfer_stream_count=2
draft_prefetch_segment_size=12
prefetch_max_inflight=8
```

Log:

```text
/home/mumura/moe_spec/logs/draft_profile_pinned_ratio25_stream2_20260605_145344.log
```

Raw report:

```text
results/draft_profile_20260605/pinned_ratio25_stream2.json
```

Correctness:

```text
deterministic digest match: true
standard graph replays: 63
draft graph replays: 37
late transfer bytes: 0
stream ticket distribution: 934 / 934
```

Comparison against the instrumented pageable-host run:

| Metric | Pageable, stream 1 | Pinned, stream 2 | Change |
|---|---:|---:|---:|
| standard decode forward | 15.064 ms | 14.707 ms | -2.4% |
| draft forward | 34.710 ms | 22.726 ms | -34.5% |
| draft / standard | 2.304x | 1.545x | improved |
| draft graph replay | 21.626 ms | 17.821 ms | -17.6% |
| draft prefetch-before | 7.095 ms | 2.424 ms | -65.8% |
| segment submit-after | 18.644 ms | 4.859 ms | -73.9% |
| segment H2D enqueue | 17.107 ms | 2.556 ms | -85.1% |
| all draft-source H2D enqueue | not exported | 3.036 ms | n/a |
| completed experts/draft | 7.378 | 12.973 | +75.8% |
| completed MiB/draft | 66.4 MiB | 116.8 MiB | +75.8% |
| verify forward | 595.455 ms | 242.974 ms | -59.2% |

Analysis:

1. The lower cache ratio creates more useful candidates, so removing pageable
   staging exposes a much larger prefetch-throughput gain than ratio 0.50.
2. Two streams stay balanced and complete every submitted ticket.
3. The remaining draft gap is `8.019 ms`; `4.197 ms` of that is the draft graph
   replay difference versus standard, with prefetch-before and segmented
   boundary submission accounting for most of the remaining controllable work.
4. This operating point now transfers approximately `116.8 MiB` during each
   draft forward while keeping draft latency near `22.7 ms`.

## 19. Segment Boundary Compression

Timestamp: 2026-06-05 14:56 CST

Objective:

Reduce graph replay and metadata boundary count by changing the layer segment
size from 12 to 24 while retaining pinned host memory and two transfer streams.

Log:

```text
/home/mumura/moe_spec/logs/draft_profile_pinned_ratio50_stream2_segment24_20260605_145658.log
```

Raw report:

```text
results/draft_profile_20260605/pinned_ratio50_stream2_segment24.json
```

Correctness:

```text
deterministic digest match: true
standard graph replays: 63
draft graph replays: 75
segment graph replays per draft: 2
late transfer bytes: 0
```

Comparison:

| Metric | Segment 12 | Segment 24 | Change |
|---|---:|---:|---:|
| draft forward | 20.816 ms | 20.697 ms | -0.119 ms |
| draft / standard | 1.424x | 1.408x | improved |
| draft graph replay | 16.716 ms | 16.711 ms | neutral |
| metadata enqueue | 0.781 ms | 0.337 ms | -56.8% |
| segment submit-after | 2.781 ms | 2.113 ms | -24.0% |
| completed experts/draft | 7.394 | 5.213 | -29.5% |
| completed MiB/draft | 66.55 MiB | 46.92 MiB | -29.5% |
| verify forward | 219.546 ms | 219.629 ms | neutral |

Analysis:

1. CUDA Graph replay duration is dominated by model work, not graph launch
   count; halving the segment count does not materially reduce replay time.
2. Boundary metadata and submit overhead decrease, but most of that work was
   hidden behind draft compute.
3. The reduced number of prefetch windows materially lowers transfer volume.

Decision: do not use segment size 24 for the throughput-oriented configuration.
Keep segment size 12.

## 20. Expanded Prefetch Supply

Timestamp: 2026-06-05 15:00 CST

Objective:

Determine whether the two-stream path can transfer more experts when software
budgets, rather than stream count, are increased.

Changes from the ratio 0.50 two-stream latency configuration:

```text
prefetch_step_budget: 4 -> 8
draft_prefetch_max_per_boundary: 4 -> 8
prefetch_max_inflight: 8 -> 16
draft_prefetch_visible_budget_ms: 3 -> 6
```

Log:

```text
/home/mumura/moe_spec/logs/draft_profile_pinned_ratio50_stream2_budget8_20260605_150006.log
```

Raw report:

```text
results/draft_profile_20260605/pinned_ratio50_stream2_budget8.json
```

Correctness:

```text
deterministic digest match: true
standard graph replays: 63
draft graph replays: 66
late transfer bytes: 0
max observed in-flight: 16
```

Results:

| Metric | Budget 4 | Budget 8 | Change |
|---|---:|---:|---:|
| draft forward | 20.816 ms | 21.830 ms | +1.014 ms |
| draft / standard | 1.424x | 1.489x | regressed |
| completed experts/draft | 7.394 | 11.667 | +57.8% |
| completed MiB/draft | 66.55 MiB | 105.0 MiB | +57.8% |
| H2D enqueue/draft | 1.314 ms | 2.110 ms | +0.796 ms |
| segment submit-after | 2.781 ms | 4.334 ms | +1.553 ms |
| verify forward | 219.546 ms | 199.785 ms | -9.0% |

Analysis:

1. Stream count was not the limiting factor for expert volume; software
   submission and in-flight budgets were.
2. The larger budget converts approximately `1 ms` of additional draft
   latency into `38.45 MiB` more expert data per draft forward.
3. More prefetched experts reduce verify forward by about `19.8 ms`, so this
   may improve end-to-end speculative-step latency even though draft-only
   latency increases.
4. The choice is workload-policy dependent and should be explicit.

Decision:

- Latency-oriented profile: budget 4, in-flight 8, visible budget 3 ms.
- Transfer/verify-oriented profile: budget 8, in-flight 16, visible budget
  6 ms.
- Do not silently change the global defaults.

## 21. Final Root-Cause And Recommendation

### 21.1 Root cause

The dominant original draft prefetch overhead was not candidate ranking or
cache reservation. It was expert H2D submission from pageable CPU tensors.

At cache ratio 0.50:

```text
pageable draft-source enqueue: 9.362 ms/forward
pinned draft-source enqueue:   1.314 ms/forward
```

At cache ratio 0.25, segment enqueue changed from:

```text
17.107 ms/forward -> 2.556 ms/forward
```

### 21.2 Recommended latency configuration

```text
cpu_expert_pin_memory=true
prefetch_transfer_stream_count=2
draft_prefetch_segment_size=12
prefetch_step_budget=4
draft_prefetch_max_per_boundary=4
prefetch_max_inflight=8
draft_prefetch_visible_budget_ms=3
```

Observed results:

| Cache ratio | Standard decode | Optimized draft | Ratio | Experts/draft | MiB/draft |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 14.707 ms | 22.726 ms | 1.545x | 12.973 | 116.76 |
| 0.50 | 14.616 ms | 20.816 ms | 1.424x | 7.394 | 66.55 |

### 21.3 Recommended transfer-oriented configuration

For ratio 0.50 when verify latency or expert coverage matters more than
draft-only latency:

```text
prefetch_step_budget=8
draft_prefetch_max_per_boundary=8
prefetch_max_inflight=16
draft_prefetch_visible_budget_ms=6
```

Observed:

```text
11.667 experts/draft
105.0 MiB/draft
21.830 ms draft forward
199.785 ms verify forward
```

### 21.4 Remaining draft versus standard gap

For the ratio 0.50 latency configuration, the `6.20 ms` remaining gap includes:

1. Draft graph replay versus standard graph replay: approximately `3.22 ms`.
2. Draft mode transition: approximately `0.99 ms`.
3. Prefetch-before drain, phase-1 submit, and metadata recorder arm:
   approximately `1.33 ms`.
4. Remaining routing, wrapper, publication, and worker interaction overhead.

Segment size 24 proved that graph launch count itself is not the main replay
gap. Closing the remaining difference requires architectural work such as a
global/device-visible execution mode, less Python/GIL interaction around
metadata processing, or a draft graph path with less heterogeneous cache and
reroute overhead. Those changes have a larger correctness blast radius than
the pinned transfer and stream-pool changes implemented here.

## 22. Final Verification

Timestamp: 2026-06-05 15:04 CST

Branch:

```text
codex/draft-forward-prefetch-opt
```

Implementation commits:

```text
26c5a1f perf: instrument draft expert prefetch
0c1b21f perf: expose pinned expert prefetch profiling
a2a1c08 perf: add expert transfer stream pool
```

Command:

```bash
srun --jobid=29629 --ntasks=1 bash -lc '
  source /opt/Software/Anaconda3/etc/profile.d/conda.sh
  conda activate nano_moe
  export CUDA_VISIBLE_DEVICES=2
  cd /home/mumura/moe_spec/nano-vllm-moe
  python -m py_compile \
    nanovllm/config.py \
    nanovllm/expert/prefetcher.py \
    examples/heterogeneous_benchmark_case.py \
    examples/benchmarks/draft_standard_decode_forward_bench.py
  python -m unittest \
    tests/test_config_prefetch.py \
    tests/test_predictive_prefetch_cli_args.py \
    tests/test_prefetch_runtime.py \
    tests/test_expert_cache_staging.py \
    tests/test_draft_standard_decode_forward_bench.py
'
```

Log:

```text
/home/mumura/moe_spec/logs/draft_profile_final_verification_20260605_150453.log
```

Result:

```text
py_compile: OK
Ran 48 tests in 2.775s
OK
git diff --check: OK
```

The pre-existing unrelated worktree changes listed earlier in this report were
left untouched and are not included in the implementation commits.
