# Draft Forward And Prefetch Optimization Report

Date: 2026-06-05

Status: design approved; baseline and implementation pending

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

No benchmark has been run for this optimization series yet.

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
