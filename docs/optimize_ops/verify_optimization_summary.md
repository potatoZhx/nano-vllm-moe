# Verify Optimization Summary

Date: 2026-07-07

This document summarizes the verify latency investigation, the code changes
that were made, the rollback knobs, the validation results, and the optimized
benchmark script.

For the later `draft-stop-policy=tpot` investigation and per-verify cost model,
see `docs/optimize_ops/verify_time_cost_model_tpot.md`.

## 2026-07-13 High-K Follow-Up

The five-seed K6/vpb4 result below remains the current formal best, but it is
now treated as a baseline rather than the target design. A new
`k12_bucket_stop` path implements high maximum draft length with decisions only
at graph-efficient K6/K9/K12 boundaries.

The first high-K online candidate achieved mean K `8.86` and reduced verify
calls from 86 to 64, but reached only `30.686 tok/s` (`32.589 ms` TPOT). The
extra draft work exceeded the saved verify work. The key verify-model finding
is that a K6-time forecast for K9 overpredicted actual K9 verify by `24.411 ms`
on 18 paired rounds because it froze cache state across three draft/prefetch
steps. Current-state verify prediction remains valid; multi-horizon cache
evolution is the missing model term.

The implementation now includes boundary endpoint lookahead, reachable-only
cost references, deferred model calls, explicit cache-warmup credit, correct
candidate-matched trace errors, and a token-feature hot-path change that avoids
full-vocabulary FP32 materialization. The latter has feature-equivalence tests
but has not been performance measured.

No large high-K sweep was run. The accepted formal number is still the K6/vpb4
five-seed result, while `k12_bucket_stop` remains an experimental configuration.
The reported absolute decode rate uses 511 inter-token intervals for a
512-token output because the excluded prefill step emits the first token. The
saved timing fields allowed the earlier all-token denominator to be corrected
without rerunning performance experiments; relative comparisons are unchanged.

## Earlier 2026-07-13 TPOT Fast-Path Result

The current production recommendation supersedes the older K4/K12 tuning
recommendations later in this document for the exact `cache_ratio=0.3125`,
512-token, sampling workload.

The verify-time work now has two separate outcomes:

1. The sampling-specific CPU-expert verify-time model passed its fresh
   instrumentation-off v2 shadow: MAE `5.776 ms`, P90 `10.311 ms`, ranking
   `0.837`, with every graph bucket passing its fixed gate.
2. The active early-stop policy did not beat fixed K6. Its best two screens
   were `-1.18%` and `-2.32%` versus their fixed baselines, so model promotion
   did not automatically promote the policy.

The sampling model is validated for the vpb10 collection protocol. The vpb4
production result below uses no verify-time model; using active mode with vpb4
would require a separate vpb4 shadow gate.

For production TPOT, fixed K6 removes the early-stop control cost and aligns
verify length 7 with graph bucket 7. The fixed policy does not consume
acceptance alpha or draft-route cost predictions, so its predictor is disabled.
Reducing `verify_prefetch_max_per_boundary` from 10 to 4 then limits
best-effort transfers that otherwise contend with the short verify critical
path.

The frozen five-seed comparison is:

| setting | decode tok/s geomean | mean TPOT | quality failures |
|:---|---:|---:|---:|
| K6, predictor off, vpb10 | 31.370 | 31.969 ms | 0/5 |
| K6, predictor off, vpb4 | 33.501 | 29.877 ms | 0/5 |

The paired improvement is `6.79%` geomean with bootstrap 95% CI
`[+2.39%, +11.12%]`. A three-seed full K sweep also selected K6 under vpb4:
K6 reached `34.273 tok/s`, versus `33.137` for K4 and `31.586` for K9.

Run the accepted configuration with:

```bash
python scripts/bench_eval_workload_tpot.py \
  --request-mode per_layer_slots \
  --optimized-config k6_decode \
  --output-lens 512 \
  --output-dir results/tpot_k6_decode \
  --save-token-ids true --save-text true --skip-existing false
```

`acceptance_predictor_enabled` is now automatic in the workload benchmark:
fixed `stop_policy=none` disables it, while TPOT/alpha/verify-proxy policies
enable it. An explicit attempt to disable a predictor required by a policy
fails before model load. The policy collector applies the same rule per
variant, preventing the fixed baseline from paying predictor overhead.

## 2026-07-11 Verify-Time Model Correction

The earlier TPOT analysis in this document and the previous version of the
linked note counted logical verify rows. CUDA-graph verify and KT CPUInfer
execute the complete bucket, including padding rows. Conclusions that CPU
routes were weak predictors were therefore based on a mismatched workload and
timing target and are superseded by the clean analysis in
[`verify_time_cost_model_tpot.md`](verify_time_cost_model_tpot.md).

Current validated status:

| stage | data | result | status |
|:---|:---|:---|:---:|
| real execution workload -> acceptance-ready verify time | 3,049 calls, grouped holdout | MAE 4.772 ms, P90 9.075 ms, ranking 0.946 | PASS |
| draft-route proxy -> execution workload -> verify time | 579-call grouped holdout | MAE 5.968 ms, P90 11.255 ms, ranking 0.838 | FAIL |
| instrumentation-off, offset-2 deployment shadow | 2,381 calls | MAE 6.387 ms, P90 12.132 ms, ranking 0.890 | FAIL |

The clean model shows that verify latency is strongly related to full-bucket
CPU expert workload. Within a fixed bucket, CPU expert count has correlation
`0.938-0.977` with acceptance-ready latency. A diagnostic op-event run also
shows `kt.cpuinfer_sync` correlation `0.987` with acceptance-ready latency,
while attention is nearly constant (mean 5.367 ms, P90 5.416 ms). Op events are
nested and synchronized, so they are diagnostic only and are not used to fit
the model.

Production `active` mode remains disabled because the causal workload proxy and
the independent shadow did not pass the fixed gates. No new decode-tokens/s
improvement is claimed. The historical verify and K=12 benchmark tables below
remain useful as records of those specific runs, but they must not be used as
evidence that the new verify-cost model is deployable.

Pooling the output-64 and output-96 calibration workloads did not resolve the
failure: replaying the frozen causal CPU-count predictions on the untouched
shadow changed MAE/P90 only from `6.387/12.132 ms` to `6.263/11.834 ms`, with
ranking `0.891`. The replay reproduced the original stored predictions with
maximum delta `0.0 ms`; it remains diagnostic and transfers no deployment gate.

## Target

The optimized verify target is:

- `verify_forward_ms_avg`: 50-80 ms/call
- `decode_phase_output_tok_s`: 25-40 output tokens/s
- graph-external overhead outside verify CUDA graph replay: about 3 ms/call

The validated optimized benchmark keeps the same feature path as
`scripts/bench_per_layer_slots.py`: profile-weighted slot allocation, slot
buckets, runtime prefetch, metadata offload, kt_direct CPU experts, verify CUDA
graph, and CPU miss policy.  The default optimized draft length is `K=4`.
Use `--max-draft-tokens-values 12` to reproduce the original command's draft
length; this is useful as a stress test but was not the setting that reached
the 50-80 ms verify target.

## Baseline Findings

The first precise breakdown was collected with verify prefetch and runtime
metadata disabled to remove boundary overhead.  That case showed:

| item | ms/verify | notes |
|:---|---:|:---|
| verify avg | 118.692 | wall time per verify call |
| segment CUDA event | 116.849 | CUDA graph replay critical path |
| `layer.total` | 110.720 | nested event total, not wall-additive |
| `layer.moe` | 103.363 | MoE dominates verify |
| `kt.cpuinfer_sync` | 76.792 | CPUInfer wait tail at the MoE merge point |
| `moe.gpu_gate_up` | 11.579 | GPU cached expert grouped GEMM |
| `moe.gpu_down` | 4.772 | GPU cached expert grouped GEMM |
| `layer.attention` | 5.218 | attention path |
| `kt.output_cpu_to_gpu_copy` | 0.360 | not a material bottleneck |

This resolved two earlier ambiguities:

- `kt.cpuinfer_sync` is the remaining wait at the CPU/GPU merge point, not the
  complete CPUInfer compute time.
- GPU MoE kernels can overlap part of CPUInfer, but there is not enough GPU
  work to hide the CPUInfer tail.  That is why verify time is much larger than
  `attention + visible GPU MoE`.

After turning verify prefetch and metadata back on, CPU miss routes dropped and
graph replay improved, but graph-external overhead became visible:

| item | prefetch off / metadata off | prefetch on / metadata on |
|:---|---:|---:|
| verify avg | 118.692 | 91.872 |
| segment CUDA event | 116.849 | 70.428 |
| `layer.moe` | 103.363 | 56.633 |
| `kt.cpuinfer_sync` | 76.792 | 25.711 |
| CPU routes/call | fallback only | 506.1 |
| graph-external gap | 1.843 | 21.443 |

Conclusion: prefetch and metadata are useful because they reduce CPUInfer wait,
but the old verify boundary ranking/submit and metadata profile path consumed
too much host-visible time.

## Call Chain And Cost Mapping

The relevant verify path is:

```text
ModelRunner.run_verify
  -> _execute_verify_forward
     -> _run_verify_with_kt_hybrid_segment_graph
        -> for each verify segment:
           -> prefetch_runtime.on_verify_layer_start / publish ready
           -> CUDAGraph.replay
              -> Qwen3MoeForCausalLM.forward_verify_kt_hybrid_segment
                 -> Qwen3MoeModel.forward_verify_kt_hybrid_segment
                    -> Qwen3MoeDecoderLayer.forward_verify_kt_hybrid
                       -> Qwen3MoeHeterogeneousSparseMoeBlock.forward_verify_kt_hybrid
                          -> router gate / topk / plan
                          -> KTDirectMoEBackend.begin_forward_graph_verify
                          -> GPU cached expert gate/up/down
                          -> kt_direct CPUInfer submit/copy/sync
                          -> accumulate CPU and GPU outputs
           -> submit_verify_segment_prefetch for next segment
        -> deferred verify metadata offload
```

Current optimized cost mapping:

| path component | representative profile field | role |
|:---|:---|:---|
| graph replay wall | `verify_segment_cuda_event_ms` | critical graph path |
| graph-external gap | `verify_forward_ms_avg - verify_segment_cuda_event_ms/call` | host/prefetch/metadata outside graph |
| boundary submit | `verify_segment_boundary_submit_ms` | visible next-segment prefetch work |
| candidate ranking | `verify_segment_prefetch_rank_ms` | candidate scan/merge/rank for verify prefetch |
| H2D enqueue | `verify_segment_prefetch_transfer_enqueue_ms` | host enqueue of prefetch transfers |
| metadata enqueue | `verify_deferred_segment_metadata_enqueue_total_ms` | D2H metadata offload scheduling |
| async metadata profile | `verify_metadata_profile_async_loop_ms` | worker-side route/expert stats |
| per-layer CPU experts | `verify_layer_{idx}_realized_cpu_expert_count_sum` | real CPUInfer active miss experts |
| per-layer CPU routes | `verify_layer_{idx}_cpu_routes_sum` | CPUInfer routes |

## Code Changes

| area | files | change | why |
|:---|:---|:---|:---|
| sync metadata readback | `nanovllm/engine/model_runner.py` | synchronous verify metadata profile readback is no longer the default | removes the earlier 65-77 ms host-side sync/readback path from normal verify |
| deferred segment metadata | `nanovllm/engine/model_runner.py` | `NANOVLLM_VERIFY_DEFER_SEGMENT_METADATA` defaults to enabled | avoids per-segment metadata D2H/enqueue work blocking the next replay |
| async metadata stats | `nanovllm/engine/model_runner.py` | metadata worker aggregates verify route/expert counters from runtime metadata | preserves observability without forcing main-thread readback |
| verify rank cache | `nanovllm/expert/prefetcher.py` | `SegmentCandidateIndex` maintains per-segment ranked candidates and dirty segments | moves most candidate sorting out of verify boundary submit |
| verify rank limit | `nanovllm/expert/prefetcher.py` | `NANOVLLM_VERIFY_PREFETCH_RANK_MULTIPLIER=1` by default | scans only the top dispatch-budget candidates per index instead of full ranking |
| boundary budget | `nanovllm/config.py`, benchmark scripts | `verify_prefetch_max_per_boundary` default lowered to `4` | keeps H2D volume and boundary submit under the graph-external gap target |
| per-layer CPU expert stats | `model_runner.py`, `qwen3_moe.py`, benchmark scripts | records 48-layer CPU expert/route arrays and exports CSV | allows latency changes to be tied to exact CPUInfer workload per layer |
| per-op event profile | `nanovllm/utils/verify_op_events.py`, `model_runner.py` | CUDA event labels for verify ops | used for diagnosis only; not enabled in performance benchmark |
| optimized benchmark wrapper | `scripts/bench_optimized_verify_perf.py` | runs `bench_per_layer_slots.py` with optimized knobs and target checks | one command validates verify latency and decode throughput |

Benchmark scripts that now understand per-layer CPU expert stats include:

- `scripts/bench_per_layer_slots.py`
- `scripts/bench_verify_boundary_overhead.py`
- `scripts/bench_segment_graph_no_prefetch.py`

The default verify prefetch budget was also aligned in:

- `scripts/bench_acceptance_predictor.py`
- `scripts/bench_eval_workload_tpot.py`
- `scripts/bench_dual_queue_prefetch.py`
- `scripts/bench_verify_segment_graph.py`
- `benchmarks/scripts/spec_verify_expert_count_stats.py`

## Rollback And Profile Knobs

| knob | effect |
|:---|:---|
| `NANOVLLM_VERIFY_SYNC_METADATA_PROFILE_READBACK=1` | restore synchronous verify metadata profile readback for debugging |
| `NANOVLLM_VERIFY_DEFER_SEGMENT_METADATA=0` | offload metadata after each segment instead of once per verify |
| `NANOVLLM_VERIFY_PREFETCH_RANK_MULTIPLIER=0` | disable top-N rank limiting and use full candidate ranking |
| `--verify-prefetch-max-per-boundary 16` | approximate old high-volume boundary prefetch behavior |
| `NANOVLLM_VERIFY_BOUNDARY_PREFETCH_ASYNC=1` | test async boundary submit; not default because it did not improve the measured critical path |
| `NANOVLLM_VERIFY_OP_EVENT_TIMING=1` | enable per-op CUDA event profiling; this is a profiling path, not a performance run |
| `NANOVLLM_VERIFY_DISABLE_RUNTIME_METADATA=1` or `NANOVLLM_VERIFY_SKIP_METADATA_OFFLOAD=1` | disable verify runtime metadata offload |

## Validation Results

All rows below use:

- cache ratio `0.3125`
- segment size `12`
- allocation mode `profile_weighted`
- slot buckets `4`
- slot max bucket ratio `2.0`
- slot profile CSV `pre_exps/exp_and_figs/unique/unique_count_plot_summary_n1024.csv`
- kt threads `16`
- verify CUDA graph buckets `3,5,7,10,13`
- `verify_prefetch_max_per_boundary=4`
- `NANOVLLM_VERIFY_PREFETCH_RANK_MULTIPLIER=1`
- `NANOVLLM_VERIFY_DEFER_SEGMENT_METADATA=1`

| run | output len | repeats | verify ms | decode tok/s | segment event | graph gap | boundary | rank | H2D enqueue | CPU routes/call | CPU experts/call | hit | accept |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `verify_rank_mult1_budget4_event_k4_l32` | 32 | 1 | 74.901 | n/a | 71.870 | 3.031 | 1.947 | 0.250 | 0.928 | 615.1 | 397.6 | 0.6104 | 0.7857 |
| `verify_rank_mult1_budget4_event_k4_l128_r3` mean | 128 | 3 | 66.218 | 29.761 | 63.081 | 3.137 | 2.058 | 0.288 | 0.980 | 439.5 | 302.3 | 0.755 | 0.882 |
| `verify_rank_mult1_budget4_event_k4_l512_r1` | 512 | 1 | 54.604 | 31.956 | 51.466 | 3.137 | 2.364 | 0.324 | 1.082 | 213.6 | 161.2 | 0.885 | 0.864 |

The 512-token optimized run meets both targets:

- verify: `54.604 ms`, inside the 50-80 ms target
- decode: `31.956 tok/s`, inside the 25-40 tok/s target
- graph-external gap: `3.137 ms`, matching the about-3 ms target

The long-output result is faster because cache hit rate rises over time and CPU
routes drop from about 439.5/call in the 128-token mean to 213.6/call at
512 tokens.

The optimized wrapper script was also run directly:

```bash
CUDA_VISIBLE_DEVICES=2 conda run -n nano_moe python scripts/bench_optimized_verify_perf.py \
  --output-dir results/optimized_verify_perf_k4_l512 \
  --bench-timeout-sec 3600 \
  --case-timeout-sec 3000
```

Result:

| output dir | verify ms | decode tok/s | total tok/s | segment event | graph gap | boundary | rank | H2D enqueue | H2D MB/call | CPU routes/call | CPU experts/call | hit | accept | target |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| `results/optimized_verify_perf_k4_l512` | 55.292 | 32.589 | 30.301 | 51.814 | 3.478 | 2.868 | 0.422 | 1.331 | 150.995 | 208.0 | 156.1 | 0.8903 | 0.8944 | PASS |

Generated artifacts:

- `results/optimized_verify_perf_k4_l512/optimized_verify_summary.md`
- `results/optimized_verify_perf_k4_l512/optimized_verify_summary.json`
- `results/optimized_verify_perf_k4_l512/per_layer_cpu_experts.csv`

## Why The Optimizations Work

The main win does not come from copying CPU expert outputs faster.  The CPU
output copy was already small.  The win comes from changing the tradeoff between
CPUInfer routes and graph-external overhead:

1. Verify prefetch and metadata reduce active CPU miss routes, shrinking the
   `kt.cpuinfer_sync` tail inside graph replay.
2. Deferred metadata and async profile aggregation remove the earlier synchronous
   metadata readback/profile loop from the main verify path.
3. Ranked candidate cache and top-N rank limiting reduce boundary ranking from
   multi-millisecond full scans to sub-millisecond ranking in the optimized path.
4. Boundary budget `4` limits H2D volume.  A larger budget lowers CPU routes but
   can slow graph replay through transfer/copy-engine contention; a smaller
   budget saves boundary time but lets CPUInfer grow again.

This leaves the remaining bottleneck mostly inside graph replay: CPUInfer wait
at each MoE merge point plus the non-hidden GPU MoE/attention work.

## Optimized Benchmark Script

Run the optimized default:

```bash
conda activate nano_moe
cd /home/linke/nano-vllm-moe
rm -rf results/optimized_verify_perf
CUDA_VISIBLE_DEVICES=2 python scripts/bench_optimized_verify_perf.py \
  --output-dir results/optimized_verify_perf
```

The defaults expand to the feature-equivalent per-layer-slots configuration:

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/bench_optimized_verify_perf.py \
  --output-dir results/optimized_verify_perf \
  --gpu-memory-utilization 0.99 \
  --cache-ratios 0.3125 \
  --output-lens 512 \
  --max-draft-tokens-values 4 \
  --segment-sizes 12 \
  --allocation-modes profile_weighted \
  --slot-buckets 4 \
  --slot-max-bucket-ratio 2.0 \
  --slot-profile-csv pre_exps/exp_and_figs/unique/unique_count_plot_summary_n1024.csv \
  --kt-num-threads 16 \
  --verify-cuda-graph-bucket-steps 3,5,7,10,13 \
  --verify-prefetch-max-per-boundary 4 \
  --verify-prefetch-visible-budget-ms 12.0
```

Script outputs:

- `summary.json`: raw `bench_per_layer_slots.py` summary
- `summary.md`: raw per-layer slot benchmark report
- `per_layer_cpu_experts.csv`: one row per layer per case
- `optimized_verify_summary.json`: optimized target summary
- `optimized_verify_summary.md`: compact pass/fail report
- `optimized_verify_command.txt`: exact command used

To stress the original draft length while keeping the optimized feature path:

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/bench_optimized_verify_perf.py \
  --output-dir results/optimized_verify_perf_k12 \
  --max-draft-tokens-values 12 \
  --fail-on-target-miss false
```

Use `--skip-existing true` to summarize existing case JSON files without
rerunning them.

## K=12 Decode Tuning

The original K=12-style run was tuned separately because its best prefetch and
draft-stop settings differ from the K=4 latency target.  The K=12 sweep used the
same base feature path as `bench_per_layer_slots.py`:

- output len `512`
- cache ratio `0.3125`
- segment size `12`
- allocation mode `profile_weighted`
- slot buckets `4`
- slot profile CSV `pre_exps/exp_and_figs/unique/unique_count_plot_summary_n1024.csv`
- kt threads `16`
- verify CUDA graph buckets `3,5,7,10,13`
- `NANOVLLM_VERIFY_PREFETCH_RANK_MULTIPLIER=1`
- `NANOVLLM_VERIFY_DEFER_SEGMENT_METADATA=1`

### K=12 Sweep Results

| run | draft stop | verify budget | rank mult | verify ms | decode tok/s | total tok/s | accept | hit | verify calls | segment event | gap | boundary | H2D MB/call | submit/call | CPU routes/call | CPU experts/call |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `tpot_b4` | `tpot` | 4 | 1 | 97.610 | 25.748 | 24.300 | 0.7270 | 0.8672 | 77 | 94.145 | 3.465 | 2.557 | 151.0 | 16.0 | 446.2 | 279.7 |
| `tpot_b6` | `tpot` | 6 | 1 | 86.296 | 30.002 | 28.072 | 0.8330 | 0.8844 | 72 | 82.448 | 3.848 | 3.130 | 226.5 | 24.0 | 369.4 | 238.2 |
| `tpot_b8` | `tpot` | 8 | 1 | 85.550 | 30.967 | 28.907 | 0.8299 | 0.8963 | 67 | 81.625 | 3.925 | 3.463 | 302.0 | 32.0 | 357.9 | 228.5 |
| `tpot_b10` | `tpot` | 10 | 1 | 90.845 | 27.001 | 25.432 | 0.7247 | 0.9041 | 74 | 86.154 | 4.691 | 4.201 | 377.5 | 40.0 | 336.8 | 222.4 |
| `tpot_b8_r2` | `tpot` | 8 | 2 | 91.087 | 30.639 | 28.638 | 0.8441 | 0.8927 | 67 | 87.158 | 3.929 | 3.599 | 302.0 | 32.0 | 364.7 | 239.3 |
| `none_b6` | `none` | 6 | 1 | 74.881 | 28.961 | 27.159 | 0.6796 | 0.8902 | 57 | 71.465 | 3.416 | 2.792 | 226.5 | 24.0 | 536.2 | 300.4 |
| `none_b8` | `none` | 8 | 1 | 75.744 | 31.258 | 29.177 | 0.7403 | 0.8913 | 52 | 71.781 | 3.964 | 3.554 | 302.0 | 32.0 | 539.3 | 310.8 |
| `none_b10` | `none` | 10 | 1 | 79.088 | 33.017 | 30.682 | 0.8007 | 0.8969 | 49 | 74.580 | 4.508 | 4.257 | 377.1 | 40.0 | 505.8 | 294.6 |
| `none_b12` | `none` | 12 | 1 | 93.001 | 30.930 | 28.866 | 0.7925 | 0.8976 | 49 | 87.391 | 5.610 | 5.536 | 453.0 | 48.0 | 507.4 | 294.2 |

Best observed K=12 setting:

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/bench_optimized_verify_perf.py \
  --output-dir results/optimized_verify_perf_k12_budget10_stopnone_l512 \
  --max-draft-tokens-values 12 \
  --verify-prefetch-max-per-boundary 10 \
  --draft-stop-policy none \
  --fail-on-target-miss false
```

Result:

- `decode_phase_output_tok_s = 33.017`
- `throughput_output_tok_s = 30.682`
- `verify_forward_ms_avg = 79.088`
- `verify_segment_cuda_event_ms_per_call = 74.580`
- `graph_external_gap_ms_per_call = 4.508`
- `verify_prefetch_submitted_mb_per_call = 377.1`
- `verify_cpu_routes_per_call = 505.8`
- `verify_cpu_experts_per_call = 294.6`

Interpretation:

- For K=12, the default `tpot` stop policy drafted only about 8 tokens per
  verify round in the best `tpot_b8` run.  For this prompt and cache profile,
  forcing full K reduced verify calls enough to improve decode throughput.
- `verify_prefetch_max_per_boundary=10` is the best observed H2D/CPUInfer tradeoff.
  Budget 8 leaves more CPUInfer work and lower acceptance; budget 12 increases
  H2D/graph-external pressure and pushes verify back to 93 ms.
- `NANOVLLM_VERIFY_PREFETCH_RANK_MULTIPLIER=2` did not help K=12 in this sweep:
  it raised rank overhead and did not reduce CPU routes.
- The result is a single 512-token run.  For release-level confidence, rerun the
  recommended setting with `--repeats 3`.

## TPOT Workload Metric Alignment

Follow-up date: 2026-07-08.

The earlier K=12 result `decode_phase_output_tok_s = 33.017` came from
`bench_per_layer_slots.py` / `bench_optimized_verify_perf.py`.  That metric is:

```text
generated_output_tokens / (engine_profile["spec_spec_step_ms"] / 1000)
```

It is not the same as `scripts/bench_eval_workload_tpot.py`'s wall-clock TPOT
metric, which times each outer `LLM.step()` decode call:

```text
generated_output_tokens / sum(decode_step_wall_seconds)
```

The per-layer JSON also reports an end-to-end generation throughput:

```text
throughput_output_tok_s = generated_output_tokens / elapsed_sec
```

For the K=12 best observed run:

| source | output tok/s metric | value | verify/spec steps | digest |
|:---|:---|---:|---:|:---|
| `optimized_verify_perf_k12_budget10_stopnone_l512` | `decode_phase_output_tok_s` | 33.017 | 49 | `0e72ada9...` |
| same JSON | `throughput_output_tok_s` | 30.682 | 49 | `0e72ada9...` |
| current per-layer rerun with segment event timing | `decode_phase_output_tok_s` | 34.566 | 49 | `0e72ada9...` |
| same JSON | `throughput_output_tok_s` | 32.013 | 49 | `0e72ada9...` |
| original TPOT workload run | wall decode tok/s | 27.882 | 58 | `252ec251...` |

The `27.882` result was therefore not a direct regression from `33.017`.
It used a different generated-token trajectory: 58 speculative verify steps
instead of 49 and a different output digest.  With stochastic sampling, small
differences in warmup prompt, prompt bytes, profile reset timing, and prefetch
state can change the accepted draft lengths and therefore total verify calls.

### TPOT Harness Fixes

`scripts/bench_eval_workload_tpot.py` was updated to make these differences
visible and tunable:

- records `decode_step_wall_ms_sum/mean/p50/p90/max`;
- records `runtime_seed`;
- records `decode_driver`, `output_sequence_count`, `output_fixed_length_ok`,
  `output_validation_error`, and `max_repeated_token_run` so throughput results
  can be checked against malformed or duplicated output;
- supports `--decode-driver generate` to use the same `LLM.generate` driver as
  `bench_per_layer_slots.py` while timing each internal `step()` call with a
  hook;
- supports `--collect-profile` to write derived `profile_*` counters outside
  the timed request;
- supports `--engine-profile` and `--engine-profile-cuda-sync` explicitly;
- keeps `spec_profile=False` during workload TPOT so profile collection does
  not switch to a diagnostic path;
- uses the same default warmup prompt as the per-layer runner:
  `Warmup request for verify layer profile.`;
- calls `llm.get_profile(reset=True)` after warmup by default via
  `--reset-profile-after-warmup true`;
- keeps `--reset-seed-after-warmup false` by default to preserve the old
  per-layer runner behavior;
- keeps `--reset-profile-before-request false` by default so diagnostic profile
  collection does not add an extra pre-request drain/reset beyond the warmup
  reset;
- fails by default on fixed-output validation errors via
  `--fail-on-output-validation-error true`;
- exposes and passes the prefetch/rank/history parameters that the single-case
  runner passes explicitly;
- uses `PER_LAYER_SLOTS_PROMPT_TEXT + "\n"` for `--request-mode per_layer_slots`
  so the prompt token IDs match `bench_per_layer_slots.py`'s prompt file.

### Reproduction Commands

Per-layer-driver wall-clock TPOT run with profile readout:

```bash
NANOVLLM_VERIFY_PREFETCH_RANK_MULTIPLIER=1 \
NANOVLLM_VERIFY_DEFER_SEGMENT_METADATA=1 \
NANOVLLM_VERIFY_BOUNDARY_PREFETCH_ASYNC=0 \
CUDA_VISIBLE_DEVICES=2 python scripts/bench_eval_workload_tpot.py \
  --request-mode per_layer_slots \
  --output-dir results/eval_workload_tpot_k12_generate_driver_profile_l512 \
  --gpu-memory-utilization 0.99 \
  --cache-ratios 0.3125 \
  --output-lens 512 \
  --max-draft-tokens-values 12 \
  --segment-sizes 12 \
  --allocation-modes profile_weighted \
  --slot-buckets 4 \
  --slot-max-bucket-ratio 2.0 \
  --slot-profile-csv pre_exps/exp_and_figs/unique/unique_count_plot_summary_n1024.csv \
  --kt-num-threads 16 \
  --verify-cuda-graph-bucket-steps 3,5,7,10,13 \
  --verify-prefetch-max-per-boundary 10 \
  --draft-stop-policy none \
  --verify-prefetch-rank-multiplier 1 \
  --decode-driver generate \
  --collect-profile true \
  --engine-profile false \
  --save-token-ids true \
  --skip-existing false
```

Pure fast-path wall-clock TPOT run without profile readout:

```bash
NANOVLLM_VERIFY_PREFETCH_RANK_MULTIPLIER=1 \
NANOVLLM_VERIFY_DEFER_SEGMENT_METADATA=1 \
NANOVLLM_VERIFY_BOUNDARY_PREFETCH_ASYNC=0 \
CUDA_VISIBLE_DEVICES=2 python scripts/bench_eval_workload_tpot.py \
  --request-mode per_layer_slots \
  --output-dir results/eval_workload_tpot_k12_generate_driver_l512 \
  --gpu-memory-utilization 0.99 \
  --cache-ratios 0.3125 \
  --output-lens 512 \
  --max-draft-tokens-values 12 \
  --segment-sizes 12 \
  --allocation-modes profile_weighted \
  --slot-buckets 4 \
  --slot-max-bucket-ratio 2.0 \
  --slot-profile-csv pre_exps/exp_and_figs/unique/unique_count_plot_summary_n1024.csv \
  --kt-num-threads 16 \
  --verify-cuda-graph-bucket-steps 3,5,7,10,13 \
  --verify-prefetch-max-per-boundary 10 \
  --draft-stop-policy none \
  --verify-prefetch-rank-multiplier 1 \
  --decode-driver generate \
  --collect-profile false \
  --engine-profile false \
  --save-token-ids true \
  --skip-existing false
```

### Observed TPOT Runs

| run | driver | collect profile | wall decode tok/s | profile decode tok/s | throughput tok/s | steps | output check | digest |
|:---|:---|:---:|---:|---:|---:|---:|:---|:---|
| old TPOT | `step` | no | 27.882 | n/a | 26.227 | 58 | legacy/no check | `252ec251...` |
| prompt bytes aligned | `step` | no | 27.855 | n/a | 26.161 | 57 | legacy/no check | `1a197911...` |
| generate driver, no profile | `generate` | no | 30.390 | n/a | 28.415 | 52 | `seqs=1`, fixed length ok, max repeat 1 | `cde69cd6...` |
| generate driver, profile readout | `generate` | yes | 33.021 | 33.036 | 30.688 | 48 | `seqs=1`, fixed length ok, max repeat 1 | `f59581fe...` |

The generate-driver profile-readout run meets the original comparison target:

- wall-clock decode tok/s: `33.021`
- previous per-layer `decode_phase_output_tok_s`: `33.017`
- wall/profile gap: `7.190 ms` total over the 512-token decode phase
- output length: `512`
- output sequence count: `1`
- `output_fixed_length_ok = true`
- `output_validation_error = ""`
- `max_repeated_token_run = 1`

The remaining difference to the current per-layer rerun
(`decode_phase_output_tok_s = 34.566`) is mainly trajectory and per-run
variation: this TPOT run used 48 decode steps with a different digest, while
the per-layer rerun used 49 verify calls and a different output digest.  For
wall-clock workload reporting, the achieved comparison should be made against
the old K=12 target that produced the `33.017` question and against the
per-layer end-to-end throughput (`30.682`), both of which are matched or
slightly exceeded by this TPOT run.

### Draft TPOT Stop Policy Follow-Up

`--draft-stop-policy tpot` was tested separately for K=12 with default,
higher verify/draft ratio, near-full-K, and no-stop sanity configurations.
None exceeded the best `none` run.  The best TPOT result was `31.505`
decode tok/s (`td=1,tv=1000`), while the current best `none` run remains
`33.021`.  Details, commands, output validation checks, and root-cause analysis
are in
[`draft_tpot_stop_policy_analysis.md`](draft_tpot_stop_policy_analysis.md).

### Practical Conclusion

Use `decode_phase_output_tok_s` only when comparing engine-internal speculative
decode phase timing across `bench_per_layer_slots.py` / optimized verify runs.
Use TPOT wall decode tok/s for workload-facing latency, but always compare
alongside `outputs_digest`, `decode_steps`, prompt/warmup metadata, and the
output validation fields.  The corrected TPOT harness can now reproduce the
old K=12 `decode_phase_output_tok_s = 33.017` target as a wall-clock decode
measurement without relying on malformed output.
