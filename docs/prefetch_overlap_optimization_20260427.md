# Prefetch Overlap Optimization Notes (Expanded, 2026-04-27)

## 1. Background, Goal, and Reading Guide

This document summarizes the recent optimization work on the Phase 3 speculative prefetch path in `nano-vllm-moe`. The original short version of this note captured the main conclusions, but it was too compressed for readers who were not already familiar with the codebase. This expanded version explains the system and each optimization in enough detail that a new reader can understand:

1. what the speculative prefetch pipeline is trying to do
2. where runtime overhead was originally coming from
3. what changed in each optimization step
4. which changes were accepted and kept
5. which changes were tried but rejected because they weakened correctness

The high-level performance claim we wanted to validate was:

1. runtime metadata export for prefetch should not block draft decode
2. prefetch submission and execution should overlap with subsequent GPU draft computation
3. when overlap is effective, prefetch cost should be mostly hidden rather than visible on the decode critical path

The concrete performance targets were:

1. under `S = N`, run draft forward with Phase 3 enabled and make it converge toward standard CUDA graph decode
2. under `S != N`, evaluate realistic cache pressure using `cache ratio = 75%` and `50%`
3. identify the point where overlap stops being sufficient and prefetch cost becomes visible

The main engineering rule for this entire round was strict:

1. no speedup is accepted unless deterministic behavior still holds
2. asynchronous optimizations must be reviewed for race conditions, missing synchronization, stale metadata reads, and ordering changes
3. direct token-level equality checks are treated as the final correctness gate

## 2. Scope and Relevant Code Paths

The implementation and profiling work in this document mainly touched the following files:

1. [nanovllm/engine/model_runner.py](/home/mumura/moe_spec/nano-vllm-moe/nanovllm/engine/model_runner.py:1)
2. [nanovllm/expert/runtime_meta.py](/home/mumura/moe_spec/nano-vllm-moe/nanovllm/expert/runtime_meta.py:1)
3. [nanovllm/expert/prefetcher.py](/home/mumura/moe_spec/nano-vllm-moe/nanovllm/expert/prefetcher.py:1)
4. [nanovllm/expert/cache.py](/home/mumura/moe_spec/nano-vllm-moe/nanovllm/expert/cache.py:1)
5. [nanovllm/engine/speculative/spec_engine.py](/home/mumura/moe_spec/nano-vllm-moe/nanovllm/engine/speculative/spec_engine.py:1)
6. [nanovllm/layers/layernorm.py](/home/mumura/moe_spec/nano-vllm-moe/nanovllm/layers/layernorm.py:1)
7. [examples/benchmarks/draft_standard_decode_forward_bench.py](/home/mumura/moe_spec/nano-vllm-moe/examples/benchmarks/draft_standard_decode_forward_bench.py:1)
8. [examples/heterogeneous_benchmark_case.py](/home/mumura/moe_spec/nano-vllm-moe/examples/heterogeneous_benchmark_case.py:1)

These files cover three different layers of the system:

1. model execution and speculative scheduling
2. metadata export, CPU-side analysis, and prefetch queueing
3. benchmarking, reporting, and deterministic validation

## 3. System Walkthrough for Readers New to the Project

### 3.1 Standard decode path

The standard decode path is the simplest baseline. A decode step executes one forward pass and returns the next token. In the codebase, the execution eventually flows through the model runner, which launches the decode kernels and, when enabled, uses CUDA graph replay for stable low-overhead execution.

Conceptually, standard decode looks like this:

1. build decode inputs
2. run one decode forward on GPU
3. sample the next token
4. update KV/cache state

This path is our reference for "minimum control-plane overhead" because it does not need speculative draft bookkeeping, verify synchronization, or prefetch orchestration.

### 3.2 Speculative decode path

Speculative decode adds extra stages. The main speculative loop is organized around:

1. `draft`
2. `rollback`
3. `verify`
4. `accept`

In practice the draft stage predicts several tokens ahead, the verify stage re-checks them with the full model path, and the engine accepts or rejects the draft tokens based on agreement.

That means speculative decode has two different performance challenges:

1. the raw GPU work of draft and verify
2. the control-plane work needed to keep expert cache state warm enough so that future draft and verify steps do not stall on missing experts

### 3.3 Where Phase 3 prefetch fits

Phase 3 prefetch is the control-plane path that tries to move needed experts into GPU cache before they become latency-critical.

At a high level, the pipeline is:

1. a draft/verify step produces routing metadata on GPU
2. that metadata is exported to host
3. host-side logic decides which experts are already cached and which are likely worth prefetching
4. candidates are inserted into a global warm-start queue
5. background prefetch submits copies for a bounded number of experts
6. when those copies complete, experts are "published" from staging to active cache state
7. later draft/verify steps can use those experts without waiting for a cold load

The important detail is that metadata export and prefetch scheduling are not the actual tensor compute we care about. They are supporting work. Therefore, if they execute on the critical path, they become pure overhead.

### 3.4 Why `S = N` and `S != N` are both useful

This document uses two different cache settings because they answer different questions:

1. `S = N` is a control-path alignment case. It is not meant to model real cache pressure. It is used to ask: when all experts are effectively available, how much overhead does speculative prefetch machinery itself introduce?
2. `S != N` at `75%` and `50%` cache ratio is a behavior-under-pressure case. It asks: once actual prefetch work exists, how much of it remains hidden by overlap, and how much becomes visible latency?

This distinction is important because a mechanism can look excellent under `S = N` but still become a bottleneck once there are real transfers and publish events.

## 4. Experimental Environment and Resource Workflow

This round used the `cluster-compute-workflow` skill, and that skill was also updated so future runs follow the same resource-selection rules.

The updated workflow is:

1. if a `jobid` is given, inspect that job's node first
2. if the node still has an idle visible GPU, reuse that node and bind to a truly idle visible device
3. if the node is full, stop relying on that job and allocate a fresh one-GPU A100 job
4. if no `jobid` is given, automatically allocate a fresh A100 job
5. keep partition, node family, GPU type, visible-device layout, conda environment, and benchmark knobs as close as possible to earlier profiling runs

The skill file is:

1. [/home/mumura/.codex/skills/cluster-compute-workflow/SKILL.md](/home/mumura/.codex/skills/cluster-compute-workflow/SKILL.md:1)

The latest run used for the publish-fast investigation was:

1. `jobid=19597`
2. node `gpu15-A100-E2-3U`
3. `CUDA_VISIBLE_DEVICES=7`
4. conda env `nano_moe`

Relevant logs:

1. [job19597_publish_fastpath_pytest_20260427_151857.log](/home/mumura/moe_spec/logs/job19597_publish_fastpath_pytest_20260427_151857.log)
2. [job19597_publish_fastpath_batch_20260427_151930.log](/home/mumura/moe_spec/logs/job19597_publish_fastpath_batch_20260427_151930.log)
3. [job19597_publishfast_tokencheck_20260427_153620.log](/home/mumura/moe_spec/logs/job19597_publishfast_tokencheck_20260427_153620.log)

The earlier accepted host-buffer-pool baseline came from:

1. [job19053_hostpool_pytest_20260425_075448.log](/home/mumura/moe_spec/logs/job19053_hostpool_pytest_20260425_075448.log)
2. [job19053_hostpool_smoke_20260425_075521.log](/home/mumura/moe_spec/logs/job19053_hostpool_smoke_20260425_075521.log)
3. [job19053_hostpool_batch_20260425_080103.log](/home/mumura/moe_spec/logs/job19053_hostpool_batch_20260425_080103.log)
4. [job19053_cache75_tokencheck_20260425_081756.log](/home/mumura/moe_spec/logs/job19053_cache75_tokencheck_20260425_081756.log)

## 5. Benchmark Method, Profiling Method, and Correctness Gates

### 5.1 Benchmark scripts and common knobs

The primary reporting script is:

1. [draft_standard_decode_forward_bench.py](/home/mumura/moe_spec/nano-vllm-moe/examples/benchmarks/draft_standard_decode_forward_bench.py:1)

Single-case runs and token-level verification were also produced with:

1. [heterogeneous_benchmark_case.py](/home/mumura/moe_spec/nano-vllm-moe/examples/heterogeneous_benchmark_case.py:1)

The common runtime knobs used for the comparisons in this document were:

1. `num_seqs = 1`
2. `input_len = 24`
3. `output_len = 12`
4. `max_num_batched_tokens = 512`
5. `max_num_seqs = 32`
6. `max_model_len = 512`
7. `max_draft_tokens = 4`
8. `draft_top_c = 0`
9. `temperature = 0.0`
10. `engine_profile = true`
11. `engine_profile_cuda_sync = true`
12. `spec_enable_prefetch = true`
13. `prefetch_verify_wait_ms = 1.0`
14. `prefetch_step_budget = 4`
15. `prefetch_max_inflight = 8`
16. `prefetch_staging_slots_per_layer = 2`

### 5.2 Why there are several kinds of timing

This work uses several different timing views, and they answer different questions:

1. `draft_forward_ms` answers end-to-end draft latency
2. `run_model_decode_ms_per_call` isolates the decode forward path inside the runner
3. `metadata_collect_ms_per_call` measures the cost of exporting and materializing routing metadata
4. `metadata_observe_ms_per_call` measures host-side analysis and queue update
5. `submit_after_ms_per_call` measures post-collect submission work that could still leak onto the critical path
6. `publish_ms` measures the cost of making staged experts visible to active routing
7. `prefetch_wait_ms` measures how long verify/draft still had to wait for ready work to become usable
8. `prefetch_async_hidden_ratio` estimates how much of the async worker turnaround was overlapped by useful GPU compute

The key point is that lower `publish_ms` or lower `prefetch_wait_ms` alone does not guarantee a better end-to-end result. The optimization can change sequencing, and sequencing changes can affect the speculative trajectory or correctness.

### 5.3 Correctness checks used in every serious change

Every nontrivial optimization in this series was judged against three layers of correctness:

1. targeted pytest for the modified components
2. benchmark-level deterministic digest equality under `temperature=0.0`
3. direct token-level comparison using `--return-token-ids true`

The direct token-level compare is the strongest signal. A benchmark mismatch indicates something changed, but a token-level mismatch shows that the actual generated sequence diverged. That is the reason the publish-fast path was rejected even though some summary timing numbers looked promising.

## 6. Starting Point: What the System Looked Like Before the Main Overlap Work

Before the accepted overlap optimizations landed, the prefetch path behaved more like a serial control-plane extension than a genuinely overlapped pipeline.

### 6.1 Original behavior

Originally, the rough execution pattern for one draft step was:

1. finish the draft GPU work
2. export metadata from GPU recorder buffers
3. materialize metadata on host
4. synchronously analyze it
5. update access statistics and prefetch candidates
6. continue to later draft/verify work

In other words, the draft step itself was already doing useful GPU computation, but the metadata path was still handled too synchronously and too conservatively.

### 6.2 Original limitations

The original path had several problems:

1. the same control thread that wanted to launch the next decode work also had to wait for metadata export and processing
2. metadata materialization used more copying and conversion than necessary
3. a single host-side metadata slot created artificial serialization between consecutive steps
4. the profiling itself was not rich enough to tell whether time was exposed or merely delayed
5. cache-state observation and queue update were still expensive enough to dominate `S = N`

### 6.3 Why this was insufficient

This design contradicted the intended purpose of Phase 3. Prefetch exists to help future compute, so it should mostly run "under" future compute. If the next draft step is delayed by metadata collection from the previous one, then prefetch has already lost the overlap game.

The clearest measurement of the starting pain point was:

1. [draft_standard_decode_forward_sn_prefetch_opt2_20260424.json](/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/draft_standard_decode_forward_sn_prefetch_opt2_20260424.json)
2. `draft / standard = 1.825x`
3. `metadata_observe ~= 6.34 ms/call`

For a control path that ideally should have near-zero visible cost in `S = N`, this was too expensive.

## 7. Optimization 0: Stabilizing the Profiling Surface by Fixing RMSNorm Recompilation

This optimization is not a prefetch optimization by itself, but it was necessary for trustworthy measurements.

### 7.1 Before optimization

The system used a compiled RMSNorm path that saw both decode and verify/prefill inputs. Those inputs had different ranks:

1. decode commonly used 2D tensors
2. verify and prefill used 3D tensors

Because the same compiled path saw both rank patterns, `torch._dynamo` repeatedly rebuilt the graph or hit `recompile_limit`. This caused two problems:

1. unstable runtime overhead unrelated to the prefetch mechanisms we wanted to study
2. less predictable graph-capture behavior, which made standard-vs-draft comparisons noisier

### 7.2 After optimization

The fix in [layernorm.py](/home/mumura/moe_spec/nano-vllm-moe/nanovllm/layers/layernorm.py:1) split the 2D and 3D behavior instead of relying on one compiled path to serve both. The intent was simple:

1. keep decode execution shape-stable
2. keep verify/prefill execution shape-stable
3. stop letting rank polymorphism contaminate the benchmark

### 7.3 Why it improved the system

After the change:

1. `torch._dynamo recompile_limit` warnings disappeared
2. CUDA graph capture became stable again
3. later overlap measurements were less distorted by unrelated compile jitter

This change should be thought of as measurement hygiene. Without it, later optimization numbers would have mixed "real prefetch overhead" with "shape recompilation noise".

## 8. Optimization 1: Introducing an Async Metadata Worker

This was the first structural step toward real overlap.

### 8.1 Before optimization

Before the async worker, metadata handling was too tightly coupled to the main draft control flow. The main thread that wanted to keep issuing decode work also had to do a lot of CPU-side housekeeping:

1. finish exporting metadata from GPU
2. collect it into CPU-side objects
3. run observation logic
4. update prefetch queues
5. sometimes push more work before returning to the compute path

This meant that even if the GPU had more draft work available, CPU bookkeeping could hold the pipeline back.

### 8.2 Limitations of the old design

The old design had three core limitations:

1. metadata export was not logically separated from metadata interpretation
2. there was no independent worker to absorb host-side observation latency
3. the profiling did not clearly quantify how much worker turnaround was hidden by GPU compute

As a result, it was hard to distinguish "real dependency" from "incidental serialization".

### 8.3 After optimization

The accepted async-worker design in [model_runner.py](/home/mumura/moe_spec/nano-vllm-moe/nanovllm/engine/model_runner.py:1) introduced:

1. a background metadata worker thread
2. an async handoff queue from the draft path to the worker
3. explicit tracking of outstanding work
4. separate accounting for device-buffer reuse wait and host-buffer reuse wait
5. overlap-oriented profile metrics such as `prefetch_async_hidden_ms`, `prefetch_async_hidden_ratio`, and `prefetch_async_exposed_wait_ms`

The new intended flow became:

1. draft finishes writing runtime metadata to a recorder buffer
2. metadata export is initiated
3. the draft control path hands an item to the worker
4. the main path continues toward future draft compute
5. the worker collects metadata, observes access patterns, updates prefetch state, and optionally triggers follow-up submission

### 8.4 Why it improved the system

The improvement came from changing a serial dependence into a producer/consumer relationship. The main thread no longer needed to sit and interpret metadata before launching subsequent useful work.

This did not magically remove all overhead. The worker still had to do the same logical work. But it changed where that work sat in time:

1. before optimization, metadata handling was more often directly paid on the critical path
2. after optimization, a large part of it could run while GPU draft compute from later steps was already in flight

### 8.5 Correctness protection

Because this introduced concurrency, correctness depended on stricter ownership and drain rules:

1. GPU recorder buffers cannot be reused until the previous export has safely detached from them
2. host metadata buffers cannot be reused while a worker item still references them
3. verify must explicitly drain async metadata before it waits on prefetch readiness, otherwise verify could consume stale queue state

These protections were encoded in the buffer-reuse bookkeeping and the verify-side drain path.

### 8.6 Measured effect

The first async-worker result still showed too much host overhead:

1. [draft_standard_decode_forward_sn_prefetch_async_final_20260425.json](/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/draft_standard_decode_forward_sn_prefetch_async_final_20260425.json)
2. `draft / standard = 1.787x`
3. `metadata_collect ~= 7.35 ms/call`
4. `prefetch_async_hidden_ratio ~= 0.916`

This was progress in structure, but not yet good enough in absolute cost. It proved that overlap accounting was useful, but it also revealed that metadata materialization itself was still too expensive.

## 9. Optimization 2: Separating Device-Buffer Lifetime from Host-Buffer Lifetime

This optimization was about removing false dependencies.

### 9.1 Before optimization

Before the more careful reuse accounting, the system treated metadata-export lifetime too conservatively. In practice there are two different resources:

1. the device-side recorder buffer that the GPU step writes into
2. the host-side buffer or object that the async worker later reads from

If these lifetimes are not separated, the system can end up waiting longer than necessary. For example, a later draft step may be prevented from reusing a recorder buffer even though the data has already been safely transferred out of it.

### 9.2 Limitations of the old design

The old approach made it too easy to accumulate unnecessary waits because:

1. reuse ownership was coarse
2. the code did not always distinguish GPU-side safety from host-side safety
3. waits could be attributed to the wrong resource and therefore optimized badly

### 9.3 After optimization

The accepted changes in [model_runner.py](/home/mumura/moe_spec/nano-vllm-moe/nanovllm/engine/model_runner.py:1) split the bookkeeping into:

1. device-buffer reuse wait
2. host-buffer reuse wait

This sounds small, but it matters a lot conceptually. It means:

1. the draft path waits for exactly the resource that is still unsafe to reuse
2. profiling can say whether the real bottleneck is GPU recorder pressure or host metadata pressure
3. later optimizations can attack the right bottleneck instead of a blurred aggregate

### 9.4 Why it improved the system

The benefit was not only lower time but better diagnosability. Once the waits were separated, it became clear that host-side reuse pressure was an important source of serialization and deserved its own fix. That directly motivated the host-buffer-pool change described next.

## 10. Optimization 3: Host Metadata Buffer Pool

This was the most important accepted optimization in the whole overlap series.

### 10.1 Before optimization

Originally the metadata export path effectively behaved like a single-lane host staging area. Even if GPU draft compute and worker processing were conceptually decoupled, a single host metadata slot still forced backpressure:

1. step A exports metadata into the host slot
2. the worker still owns that slot while processing it
3. step B wants to export new metadata
4. step B must wait until step A's slot is released

This creates a classic producer/consumer bottleneck. The GPU may be ready to move on, but the producer cannot hand off the next item because there is only one host staging lane.

### 10.2 Why this was insufficient

This design made overlap brittle:

1. even a short worker delay could stall the next export
2. the stall looked like "metadata overhead" even though it was partly just slot contention
3. the latency became sensitive to host scheduling noise

This is exactly the kind of avoidable serialization that shows up strongly in `S = N`, where there should be little real work to do.

### 10.3 After optimization

The accepted host-buffer-pool design in [runtime_meta.py](/home/mumura/moe_spec/nano-vllm-moe/nanovllm/expert/runtime_meta.py:1) and [model_runner.py](/home/mumura/moe_spec/nano-vllm-moe/nanovllm/engine/model_runner.py:1) changed the design in three ways:

1. host metadata storage became a small pool rather than a single slot
2. each offload handle records which host slot it owns
3. host-slot reuse is only blocked for the specific slot still being processed

That means a later draft step does not have to wait for all host processing to finish. It only has to acquire another available slot from the pool.

### 10.4 Why it improved the system

This works for the same reason that multi-buffered pipelines work in graphics and data processing:

1. one buffer can be filled
2. another can be processed
3. a third can already be prepared for the next step

The producer and consumer no longer fight over one memory slot. The pool absorbs normal jitter in worker completion time.

### 10.5 Correctness protection

Because pooled reuse can also introduce bugs if ownership is loose, the design explicitly stores slot ownership in the offload handle:

1. a handle cannot silently migrate to another slot
2. a slot is not recycled until the worker item retires
3. the worker and main thread agree on the slot identity rather than inferring it indirectly

This is important because stale-slot reuse would be catastrophic but hard to debug: it could silently mix metadata from two different decode steps.

### 10.6 Measured effect

This optimization produced the biggest accepted `S = N` improvement:

1. [draft_standard_decode_forward_sn_prefetch_hostpool_20260425.json](/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/draft_standard_decode_forward_sn_prefetch_hostpool_20260425.json)
2. `standard decode forward = 15.92 ms`
3. `draft forward = 20.59 ms`
4. `draft / standard = 1.293x`
5. `metadata_collect ~= 2.55 ms/call`
6. `metadata_observe ~= 0.50 ms/call`
7. `metadata_buffer_reuse_wait ~= 0.01 ms/call`
8. `prefetch_async_hidden_ratio ~= 0.896`

Compared with the earlier accepted observation-heavy baseline:

1. [draft_standard_decode_forward_sn_prefetch_obsopt2_20260424.json](/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/draft_standard_decode_forward_sn_prefetch_obsopt2_20260424.json)
2. `draft / standard = 1.502x`
3. `metadata_collect ~= 2.88 ms/call`
4. `metadata_observe ~= 1.53 ms/call`

The improvement was not only "a little faster". It changed the shape of the bottleneck. Host metadata reuse stopped being a first-order problem.

## 11. Optimization 4: Reducing Unnecessary CPU Materialization and Conversion

This optimization focused on making each metadata item cheaper to process.

### 11.1 Before optimization

Before the refinement of the small aggregated path, metadata handling was doing more work than strictly necessary:

1. extra host clones were created even when only aggregated information was needed
2. some paths performed avoidable CPU conversions
3. cache observation logic handled small metadata cases too generically, which meant paying for flexibility that the common decode case did not need

In other words, even after introducing asynchronous processing, each work item was still relatively heavy.

### 11.2 Why this was insufficient

Async execution can hide latency, but it cannot make a work item free. If each item performs unnecessary cloning or conversion, the worker still burns CPU cycles, burns memory bandwidth, and keeps host slots occupied longer. That directly reduces the amount of overlap the system can achieve.

### 11.3 After optimization

The accepted changes across [runtime_meta.py](/home/mumura/moe_spec/nano-vllm-moe/nanovllm/expert/runtime_meta.py:1), [prefetcher.py](/home/mumura/moe_spec/nano-vllm-moe/nanovllm/expert/prefetcher.py:1), and [cache.py](/home/mumura/moe_spec/nano-vllm-moe/nanovllm/expert/cache.py:1) did two things:

1. for small aggregated cases, `collect()` can operate on a host view rather than eagerly cloning the full per-token metadata payload
2. the observation path avoids unnecessary `.to(device="cpu")` or equivalent host materialization work when the information is already available in a usable CPU form

### 11.4 Why it improved the system

This reduced overhead in two different ways:

1. less data movement per metadata item
2. shorter worker occupancy per metadata item

That matters because the worker path is not just compute. It is part memory-copy, part data-layout conversion, and part queue update. Shrinking the payload and reducing conversions speeds up all three.

### 11.5 Correctness protection

This optimization was safe because it did not change the meaning of the metadata, only how it was represented and copied:

1. the same routing decisions are still observed
2. the same access statistics are still recorded
3. the same downstream queue-update logic still runs

The key correctness question was therefore not semantic equivalence of routing, but lifetime safety of host views versus clones. That is why the buffer-pool ownership rules remained essential.

## 12. Optimization 5: Using CPU Cache-State Queries to Cut Observation Cost

This optimization targeted `metadata_observe`, which had shown up as a major visible cost early in the process.

### 12.1 Before optimization

The earlier observation path needed to determine whether selected experts were already cached. A naive way to do that is to move or inspect a broader cache-state representation than is really necessary.

In practice, this meant the observation path was paying too much just to answer a simple question:

1. is this expert already resident and usable?
2. if it is already resident, do we need to consider it for prefetch at all?

### 12.2 Why this was insufficient

When the answer is often "already cached", the system should reject those experts cheaply. If the code first pays for a heavy state-materialization step and only then decides "nothing to do", it is doing expensive work to discover the absence of work.

### 12.3 After optimization

The accepted optimization moved the cache-state check toward a cheaper CPU-side lookup using cache metadata that was already maintained for the runtime, rather than forcing broader state movement. In practice this meant:

1. use lightweight cache residency information
2. filter out already-cached experts earlier
3. only do more expensive queue-update work when there is real prefetch work to schedule

### 12.4 Why it improved the system

This turned `metadata_observe` into a more selective pipeline:

1. cheap reject for already-cached experts
2. expensive path only for genuinely useful prefetch candidates

That is exactly the right behavior for `S = N`, where there is almost no useful prefetch work to do. The observation path should become very small in that regime.

### 12.5 Measured effect

The observation-heavy accepted baseline on 2026-04-24 had:

1. [draft_standard_decode_forward_sn_prefetch_obsopt2_20260424.json](/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/draft_standard_decode_forward_sn_prefetch_obsopt2_20260424.json)
2. `metadata_observe ~= 1.53 ms/call`

The later hostpool baseline had:

1. [draft_standard_decode_forward_sn_prefetch_hostpool_20260425.json](/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/draft_standard_decode_forward_sn_prefetch_hostpool_20260425.json)
2. `metadata_observe ~= 0.50 ms/call`

That reduction was not caused by a single code line, but this earlier cache-aware observation work was part of the reason the later hostpool version could keep `observe` small enough that it was no longer the dominant visible term.

## 13. What the Accepted Optimizations Achieved Together

By the time the accepted hostpool design was in place, the system had changed qualitatively.

### 13.1 Before the optimization series

Before the main overlap work:

1. metadata handling behaved too serially
2. buffer reuse created artificial waiting
3. observation logic did too much work per step
4. there was not enough instrumentation to tell hidden time from exposed time

### 13.2 After the accepted optimizations

After the accepted changes:

1. metadata export and observation were decoupled from the main draft thread
2. pooled host buffers removed most host-side handoff contention
3. small metadata items were cheaper to materialize
4. observation filtered cached experts more cheaply
5. overlap was explicitly measured instead of inferred indirectly

### 13.3 What this means in practical terms

Under the accepted implementation:

1. metadata export is no longer the first bottleneck to attack
2. host-buffer reuse is nearly fully hidden
3. the remaining visible latency has shifted downstream to `submit_after`, `publish`, verify visibility, and the intrinsic `route/plan/gpu_compute` cost

This is a meaningful architectural win because it tells us the optimization frontier has moved. The team is no longer guessing where the time went.

## 14. Phase-3 Baseline and Performance Timeline

The timeline below includes measurements from different intermediate states. These are not all perfectly apples-to-apples because code structure and profiling fields evolved over time, but they still show how the bottleneck moved.

### 14.1 `S = N` progression

| Stage | Result file | Draft / Standard | Main observation |
| --- | --- | ---: | --- |
| Phase 3 default | `draft_standard_decode_forward_phase3_default_final2_20260424.json` | `1.184x` | early Phase 3 baseline with feature enabled |
| Prefetch opt2 | `draft_standard_decode_forward_sn_prefetch_opt2_20260424.json` | `1.825x` | observation path dominated and made `S = N` clearly too slow |
| Observe opt2 | `draft_standard_decode_forward_sn_prefetch_obsopt2_20260424.json` | `1.502x` | observation cost reduced, but still too visible |
| Async worker | `draft_standard_decode_forward_sn_prefetch_async_final_20260425.json` | `1.787x` | structure improved, but collection remained too heavy |
| Host buffer pool | `draft_standard_decode_forward_sn_prefetch_hostpool_20260425.json` | `1.293x` | best accepted `S = N` result so far |
| Publish-fast attempt | `draft_standard_decode_forward_sn_publishfast_20260427.json` | `1.413x` | not accepted because correctness later failed |

The key interpretation is:

1. the biggest durable `S = N` win came from making metadata export cheap and asynchronous
2. reducing publish cost alone did not automatically improve `S = N`
3. once `observe` and host reuse were mostly hidden, the next visible bottlenecks moved elsewhere

### 14.2 `S != N` accepted hostpool baselines

| Cache ratio | Result file | Draft / Standard | Hidden ratio | Host reuse wait | Main exposed terms |
| --- | --- | ---: | ---: | ---: | --- |
| 75% | `draft_standard_decode_forward_cache75_hostpool_summary_20260425.json` | `1.569x` | `0.975` | `0.011 ms/call` | `submit_after`, `publish`, `prefetch_wait` |
| 50% | `draft_standard_decode_forward_cache50_hostpool_20260425.json` | `1.804x` | `0.994` | `0.008 ms/call` | `submit_after`, `publish`, `prefetch_wait` |

These numbers are important because they show that overlap of metadata export remained good even under real cache pressure. The fact that latency still rose at lower cache ratio means the exposed cost had moved to later prefetch stages rather than earlier export stages.

## 15. Optimization 6: Publish-Fast Attempt on 2026-04-27

This section is intentionally detailed because it is the best example in this round of an optimization that looked reasonable in isolation but still failed correctness.

### 15.1 Why `publish` became the next target

After the hostpool changes, the next visible costs under `S != N` were no longer metadata collection and host-slot contention. They were:

1. `submit_after`
2. `publish_ms`
3. `prefetch_wait_ms`

This showed up clearly in:

1. [draft_standard_decode_forward_cache75_hostpool_summary_20260425.json](/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/draft_standard_decode_forward_cache75_hostpool_summary_20260425.json)
2. [draft_standard_decode_forward_cache50_hostpool_20260425.json](/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/draft_standard_decode_forward_cache50_hostpool_20260425.json)

The intuition was straightforward. If metadata is already mostly hidden, then the remaining exposed time is probably spent turning "copy completed in staging" into "expert is now visible in active cache state".

### 15.2 Before optimization

Before the attempted publish-fast path, `publish_ready()` used a conservative approach that relied on a cache snapshot when evaluating victim-selection state. That approach had a clear engineering advantage:

1. it was easier to reason about
2. it separated publish logic from live mutable cache state
3. it reduced the risk of reading partially updated residency metadata

But it also had performance costs:

1. taking a snapshot introduces copying or materialization work
2. publish-side logic pays that cost repeatedly
3. the cost becomes visible once metadata export itself has already been optimized

### 15.3 Why the original design was insufficient

The snapshot-based design was safe but not cheap enough under real cache pressure. In the accepted hostpool baseline, the system had already pushed earlier overhead down, so the snapshot cost no longer hid behind anything else. That made it a reasonable next target.

### 15.4 After optimization attempt

The attempted fast path changed the design goal from "compute publish decisions from a snapshot" to "compute them directly from live cache metadata without snapshot cloning". The implementation touched:

1. `nanovllm/scheduling/cache_strategy.py`
2. `nanovllm/expert/cache.py`
3. `nanovllm/expert/prefetcher.py`

The intended mechanism was:

1. read victim-selection metadata directly from the cache
2. avoid per-publish snapshot cloning
3. preserve the same eviction-policy semantics
4. keep the same staging-to-active publish transition

### 15.5 Why it looked promising

From a pure micro-optimization perspective, this looked like exactly the kind of change we should want:

1. less copying
2. less publish-side CPU work
3. lower publish latency
4. lower wait before experts become usable

And the first timing numbers did indeed move in that direction.

### 15.6 Performance effect at cache ratio 75%

Accepted hostpool baseline:

1. raw file: [draft_standard_decode_forward_cache75_hostpool_raw_20260425/repeat_00_spec.json](/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/draft_standard_decode_forward_cache75_hostpool_raw_20260425/repeat_00_spec.json)
2. `draft_forward_ms = 24.18`
3. `submit_after_total = 18.34 ms`
4. `publish_ms = 12.23 ms`
5. `prefetch_wait_ms = 4.48 ms`

Attempted publish-fast path:

1. raw file: [draft_standard_decode_forward_cache75_publishfast_raw_20260427/repeat_00_spec.json](/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/draft_standard_decode_forward_cache75_publishfast_raw_20260427/repeat_00_spec.json)
2. summary file: [cache75_publishfast_summary_20260427.json](/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/cache75_publishfast_summary_20260427.json)
3. `draft_forward_ms = 24.76`
4. `submit_after_total = 20.35 ms`
5. `publish_ms = 10.21 ms`
6. `prefetch_wait_ms = 3.63 ms`

What this means:

1. publish became cheaper
2. wait-for-prefetch also became cheaper
3. but total draft latency did not improve

So even before correctness was checked, this was already a warning sign that the optimization was changing system behavior in a more complicated way than a local micro-optimization.

### 15.7 Performance effect at cache ratio 50%

Accepted hostpool baseline:

1. raw file: [draft_standard_decode_forward_cache50_hostpool_raw_20260425/repeat_00_spec.json](/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/draft_standard_decode_forward_cache50_hostpool_raw_20260425/repeat_00_spec.json)
2. `draft_forward_ms = 26.92`
3. `submit_after_total = 48.72 ms`
4. `publish_ms = 15.64 ms`
5. `prefetch_wait_ms = 5.56 ms`

Attempted publish-fast path:

1. raw file: [draft_standard_decode_forward_cache50_publishfast_raw_20260427/repeat_00_spec.json](/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/draft_standard_decode_forward_cache50_publishfast_raw_20260427/repeat_00_spec.json)
2. summary file: [cache50_publishfast_summary_20260427.json](/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/cache50_publishfast_summary_20260427.json)
3. `draft_forward_ms = 28.37`
4. `submit_after_total = 39.12 ms`
5. `publish_ms = 9.87 ms`
6. `prefetch_wait_ms = 3.46 ms`

Again, the local publish metrics improved, but end-to-end draft did not. That strongly suggested the optimization was affecting timing-sensitive behavior, not just shaving wasted cycles.

### 15.8 Correctness validation and failure

The attempted path passed targeted unit tests:

1. cache-strategy tests
2. prefetch runtime tests
3. metadata recorder tests
4. benchmark reporting tests

Pytest log:

1. [job19597_publish_fastpath_pytest_20260427_151857.log](/home/mumura/moe_spec/logs/job19597_publish_fastpath_pytest_20260427_151857.log)

However, deterministic runtime validation exposed problems:

1. the benchmark summary reported deterministic mismatches at `cache ratio 75%`
2. one `cache ratio 50%` benchmark batch also showed a mismatch
3. a direct token-level rerun at `cache ratio 75%` confirmed real divergence

Relevant files:

1. [cache75_standard_publishfast_tokencheck_20260427.json](/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/cache75_standard_publishfast_tokencheck_20260427.json)
2. [cache75_spec_publishfast_tokencheck_20260427.json](/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/cache75_spec_publishfast_tokencheck_20260427.json)
3. [cache50_standard_publishfast_tokencheck_20260427.json](/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/cache50_standard_publishfast_tokencheck_20260427.json)
4. [cache50_spec_publishfast_tokencheck_20260427.json](/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/cache50_spec_publishfast_tokencheck_20260427.json)

The decisive evidence was the `cache75` token divergence:

1. standard tokens: `4710, 16141, 1447, 32313, 11, 773, 358, 1184, 311, 7071, 700, 3170`
2. spec tokens: `576, 4226, 1265, 387, 304, 6364, 11, 323, 279, 2790, 3084, 1265`

This is not a small statistical wobble. It is a different generation trajectory.

### 15.9 Why the optimization was rejected

The publish-fast attempt was rejected and reverted because:

1. token-level correctness failed at `cache ratio 75%`
2. the failure happened in exactly the part of the system where timing and visibility order matter
3. the local timing improvement was not enough to justify living with uncertain semantics

### 15.10 Most likely interpretation

The exact bug mechanism still needs deeper investigation, but the likely class of problem is clear:

1. the snapshot path implicitly provided a stronger consistency boundary than the live-cache path
2. removing that boundary may have changed when an expert became visible to later routing or verify logic
3. a faster host path may have altered the interleaving between weight-copy completion, metadata commit, and later consumers

This is why publish optimization must be treated as a correctness-sensitive scheduling change rather than a cheap local cleanup.

## 16. Current State of the System

### 16.1 What is already working well

Under the accepted implementation as of the hostpool baseline:

1. metadata export is largely overlapped with later draft computation
2. host-buffer reuse is no longer a meaningful visible bottleneck
3. `S = N` has moved much closer to standard decode than earlier versions
4. `75%` and `50%` cache-ratio runs still show high hidden ratios for async work

The most important accepted result is:

1. [draft_standard_decode_forward_sn_prefetch_hostpool_20260425.json](/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/draft_standard_decode_forward_sn_prefetch_hostpool_20260425.json)
2. `draft / standard = 1.293x`

### 16.2 What is still exposed

The major visible terms now are:

1. `route`
2. `plan`
3. `submit_after`
4. `publish_ms`
5. `prefetch_wait_ms`
6. verify-side readiness cost at lower cache ratios

In other words, the remaining work is no longer "how do we export metadata cheaply?" It is "how do we make prefetch completion visible at the right time without breaking semantics?"

### 16.3 Why `submit_after` and `publish` are the next frontier

Once the system already hides metadata export well, publish becomes the next unavoidable boundary:

1. an expert that is only copied into staging is not yet safe to use
2. some explicit publish or activation step must make it visible
3. that visibility boundary is correctness-critical

Therefore, any attempt to reduce `submit_after` or `publish` must preserve:

1. ordering between copy completion and visibility
2. consistency between active-slot metadata and actual weights
3. the expectations of later draft and verify steps about what "resident" means

## 17. Risks and Current Limitations

### 17.1 Correctness sensitivity

The publish-fast failure showed that this subsystem is sensitive to subtle timing changes. That means future optimizations need stronger reasoning than "same eviction policy, less copying".

### 17.2 Benchmark mismatch is not the last word

A benchmark-level deterministic mismatch is useful, but it is not sufficient by itself to classify a bug. The stronger check is direct token equality. That is why the document distinguishes between "benchmark mismatch observed" and "token-level divergence confirmed".

### 17.3 Lower cache ratios naturally perturb the speculative trajectory

At `75%` and especially `50%` cache ratio, the system performs real prefetch, real staging, and real publish. This can alter detailed timing enough that profile counters may not line up one-for-one across runs even if the benchmark settings match. That makes correctness gating more important, not less.

### 17.4 Verify remains expensive

Even though this document focuses on draft-side overlap, verify is still a very large component of end-to-end speculative latency. That means some draft-side wins may be partially masked at the full-step level unless verify is also addressed later.

## 18. Recommended Next Steps

### 18.1 Instrument `submit_after` and `publish` more finely

The next useful profiling improvement is to split publish into smaller internal phases:

1. ready polling
2. candidate selection
3. victim selection
4. staging-to-active transfer or activation
5. metadata commit
6. visibility handoff to later consumers

Without this breakdown, it is too easy to know that `publish_ms` is expensive without knowing which substage actually owns the cost.

### 18.2 Treat visibility semantics as a first-class API boundary

The rejected publish-fast attempt suggests that the system needs a more explicit model of when an expert is considered:

1. copied
2. staged
3. ready
4. published
5. legally visible to draft
6. legally visible to verify

Making these states clearer could allow safer optimization later.

### 18.3 Continue reducing metadata payload only if semantics stay identical

Metadata export is no longer the largest bottleneck, but there may still be safe wins in payload size or representation. Those wins are only worth taking if they do not change queue-update ordering or downstream interpretation.

### 18.4 Study expert-count limits versus available overlap slack

The long-term control strategy should probably be quantitative:

1. measure prefetch transfer and publish time as a function of expert count
2. compare that with the available draft-compute slack
3. cap submission such that `prefetch_time <= hidden_slack`

That is the natural next step toward a policy that hides prefetch by construction rather than by best effort.

### 18.5 Consider larger architectural changes if publish remains fragile

If publish optimization continues to be both expensive and correctness-sensitive, a larger refactor may be required:

1. version cache visibility explicitly
2. split "copy complete" from "routing-visible"
3. enforce stronger barriers between staging and active state
4. let verify consume only fully published generations
5. redesign victim selection so it is less dependent on mutable live state

## 19. Summary

The accepted outcome of this optimization round is:

1. keep the resource-selection workflow update in the cluster-compute skill
2. keep the accepted RMSNorm stabilization, async metadata worker, buffer-lifetime separation, host-buffer pool, and reduced metadata-materialization changes
3. reject the 2026-04-27 publish-fast path because it improved local publish metrics but failed the final correctness gate

The practical conclusion is:

1. metadata export and host-buffer reuse have largely stopped being the primary problem
2. the system now behaves much closer to the intended overlap design
3. the next real optimization frontier is `submit_after` and `publish`
4. that frontier is correctness-sensitive and should be treated as scheduling semantics work, not just micro-optimization
