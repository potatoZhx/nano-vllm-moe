# Spec Verify Miss Expert Cache-Fill Implementation Plan

## Summary

Add a verify miss policy that synchronously promotes verify miss experts into
the GPU expert cache before building the verify execution plan, leaving experts
on CPU only when the active unique expert count exceeds available cache slots.
The implemented default is `cache_fill`; the legacy `cpu` behavior remains
available explicitly through `spec_verify_miss_policy="cpu"`.

Status: implemented and validated on A100 job 28363.

## Implementation Changes

- [x] Add `spec_verify_miss_policy` to public config and benchmark CLI, with allowed values `cpu` and `cache_fill`; default is `cache_fill`.
- [x] In verify mode, before `build_verify_plan_gpu()`, run a cache-fill planner when `spec_verify_miss_policy="cache_fill"`.
- [x] Cache-fill planner behavior:
  - Count per-expert active routes in the current verify layer.
  - Keep already-cached active experts in cache.
  - If active unique experts fit in cache, promote all miss experts.
  - If active unique experts exceed cache slots, keep the lowest-route-count miss experts on CPU and promote the rest.
  - Prefer empty non-pending slots; otherwise evict inactive non-pending slots using LRU-compatible `last_access_step`.
  - Do not evict active experts or write pending active slots.
- [x] Add profile counters for promoted experts, CPU experts left after policy, evictions, skipped pending candidates, and transfer time.
- [x] Add `scripts/verify_miss_policy_validation.py` to compare `cpu` vs `cache_fill` under the requested meaningful-prompt matrix.

## Test Plan

- [x] Unit tests for planner behavior:
  - all active miss experts fit and are promoted
  - active unique experts exceed slots and lowest-count miss experts stay on CPU
  - active cached experts are never evicted
  - pending slots are skipped
- [x] Config/parser tests for `spec_verify_miss_policy`.
- [x] Smoke commands:

```bash
eval "$(conda shell.bash hook)" && conda activate nano_moe
cd /home/mumura/moe_spec/nano-vllm-moe
python -m unittest tests.test_placement_spec tests.test_config_prefetch tests.test_spec_verify_expert_count_stats
python -m py_compile nanovllm/config.py nanovllm/models/qwen3_moe.py nanovllm/expert/placement.py benchmarks/scripts/spec_verify_expert_count_stats.py scripts/verify_miss_policy_validation.py
```

## Benchmark Plan

- [x] Compare `spec_verify_miss_policy={cpu,cache_fill}`.
- [x] Matrix: output length `{128,512}` x cache ratio `{0.25,0.5,0.75}`.
- [x] Ratio-specific draft lengths: `0.25 -> 3`, `0.5 -> 5`, `0.75 -> 7`.
- [x] Common settings: `draft_reroute_policy=entropy_cache_bias`, `draft_top_c=0`, draft CUDA graph enabled, `spec_enable_prefetch=true`, `prefetch_runtime_mode=draft_segment_indexed`, `cache_strategy=lru`, offline profile `results/reroute_impl_20260531/offline_profile_20260531_203257.safetensors`, CPU backend `fused`, `standard_sampling`, `temperature=0.8`.
- [x] Report acceptance rate, cache hit rate, output throughput, draft/verify forward time, graph replays, prefetch submit/done/used, and generated text quality.

Full benchmark output:

- Summary: `results/verify_miss_policy_full_20260601_162526/summary.md`
- JSON: `results/verify_miss_policy_full_20260601_162526/summary.json`
- Log: `/home/mumura/moe_spec/logs/verify_miss_policy_full_20260601_162526.log`

## Assumptions

- `token最少` means the fewest active routes for that expert in the current verify layer.
- Default behavior changes to `spec_verify_miss_policy="cache_fill"` 
