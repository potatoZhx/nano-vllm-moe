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
