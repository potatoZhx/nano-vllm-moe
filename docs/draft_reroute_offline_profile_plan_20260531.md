# Draft Reroute Offline Profile Integration Plan

**Goal:** Implement full offline profile export from `pre_exps/expert_reroute/draft_decode_eval_v2.py` and online loading/use in `nano-vllm-moe`.

**Architecture:** Add a reusable profile schema/loader, make `scripts/offline_profile.py` the unified offline profile export script, and wire the loaded profile into heterogeneous expert cache initialization and draft reroute policies. Keep old `cond_sim/skip_err` artifacts compatible while making the new safetensors artifact the default.

**Tech Stack:** Python, PyTorch, safetensors, nano-vllm-moe heterogeneous MoE cache/runtime, Slurm A100 validation via `$cluster-compute-workflow`.

---

## Summary

The current runtime only loads `cond_sim` and `skip_err`, and only for `similarity_replace`. This plan expands profile support to all reusable v2 calibration outputs:

- `cond_sim[L,N,N]`: conditional replacement similarity, used by `similarity_replace`.
- `skip_err[L,N]`: skip/output norm proxy, used by `similarity_replace`.
- `sim[L,N,N]`: combined expert similarity, exported for downstream runtime/analysis use.
- `sens[L]`: layer sensitivity, exported for downstream runtime/analysis use.
- `act_freq[L,N]`: activation frequency, used online for initial expert cache placement and LFU/RankGuard prior.

New default artifact format is `.safetensors` with metadata. Old `.pt` artifacts containing only `cond_sim/skip_err` remain loadable for `similarity_replace`, but do not affect cache initialization because they lack `act_freq`.

## Key Implementation Changes

### 1. Profile Schema And Loader

- [ ] Create `nanovllm/scheduling/draft_reroute_profile.py`.
- [ ] Define a small immutable profile container with fields:
  - `cond_sim: torch.Tensor | None`
  - `skip_err: torch.Tensor | None`
  - `sim: torch.Tensor | None`
  - `sens: torch.Tensor | None`
  - `act_freq: torch.Tensor | None`
  - `metadata: dict[str, str]`
  - `is_legacy: bool`
- [ ] Implement `load_draft_reroute_profile(path, *, num_experts, expected_layers=None, expected_top_k=None, hf_config=None)`.
- [ ] Loader behavior:
  - `.safetensors`: use `safetensors.torch.load_file` plus metadata.
  - `.pt` new format: accept `{"metadata": dict, "tensors": dict}`.
  - `.pt` legacy format: accept flat tensor dict with `cond_sim` and `skip_err`.
  - Convert loaded tensors to CPU float32 contiguous tensors.
  - Validate tensor ranks and expert dimensions.
  - If `expected_layers` is provided, require all present layer-shaped tensors to match it.
  - If metadata includes `num_experts`, `num_layers`, or `top_k`, require them to match runtime values.
- [ ] Move `load_draft_reroute_artifact` in `nanovllm/scheduling/draft_reroute.py` to call the new loader and return only the old dict shape for existing callers until all call sites are updated.

### 2. Unified Offline Profile Script

- [ ] Create `scripts/offline_profile.py`.
- [ ] Reuse calibration functions from `pre_exps/expert_reroute/draft_decode_eval_v2.py`:
  - `load_model_and_tokenizer`
  - `detect_moe_config`
  - `prepare_chunks`
  - `calibrate`
- [ ] CLI arguments:
  - `--model`
  - `--data-file`
  - `--output`
  - `--n-calib`
  - `--seq-len`
  - `--device`
  - `--dtype`
  - `--seed`
- [ ] Export tensors:
  - `cond_sim`
  - `skip_err`
  - `sim`
  - `sens`
  - `act_freq`
- [ ] Export metadata:
  - `format_version=2`
  - `num_layers`
  - `num_experts`
  - `top_k`
  - `model_type`
  - `hidden_size`
  - `source_model`
- [ ] If `--output` ends with `.safetensors`, use `safetensors.torch.save_file(tensors, metadata=metadata)`.
- [ ] If `--output` ends with `.pt`, use `torch.save({"metadata": metadata, "tensors": tensors}, output)`.
- [ ] Update `pre_exps/expert_reroute/draft_decode_eval_v2.py --calibration_artifact` to emit the same full schema instead of only `cond_sim/skip_err`.

### 3. Online Loading And Use

- [ ] In `nanovllm/engine/model_runner.py`, load `draft_reroute_artifact` whenever the path is non-empty, not only when policy is `similarity_replace`.
- [ ] Keep `similarity_replace` validation strict: if that policy is selected, loaded profile must include both `cond_sim` and `skip_err`.
- [ ] Pass the loaded profile into `HeterogeneousModelLoader`.
- [ ] In `nanovllm/utils/heterogeneous_loader.py`, use `profile.act_freq` for initial active expert placement:
  - For each real transformer layer, map to its MoE-layer profile row.
  - Select top `cache.num_slots` experts by descending `act_freq`.
  - Break ties by ascending expert id for deterministic placement.
  - If `act_freq` is absent or invalid for the layer, keep current expert-id-order placement.
- [ ] Seed cache counters when `act_freq` is present:
  - `access_count[e] = round(act_freq[layer,e] * 1000)`.
  - `access_score_sum[e] = act_freq[layer,e] * 1000.0`.
  - Leave `last_access_step[e] = -1` so LRU behavior is not changed by offline profile.
- [ ] If `cache_strategy == "lfu_rankguard"`, initialize rank scores from `act_freq`:
  - `rank_score[e] = act_freq[layer,e] * top_k`.
  - Use real transformer `layer_idx` as the strategy key.
- [ ] In `Qwen3MoeForCausalLM.enable_heterogeneous_mode`, pass per-MoE-layer `cond_sim/skip_err` from the profile into `DraftReroutePolicy`, preserving existing policy behavior.

### 4. Public Interface And Compatibility

- [ ] Keep the existing config field `draft_reroute_artifact`; it now means “offline profile artifact”.
- [ ] Do not introduce `scripts/export_draft_reroute_profile.py`; the unified script name is `scripts/offline_profile.py`.
- [ ] Keep `Config` validation:
  - `similarity_replace` requires `draft_reroute_artifact`.
  - non-round-robin reroute requires `draft_top_c == 0`.
- [ ] Non-`similarity_replace` policies may provide `draft_reroute_artifact`; only `act_freq` affects runtime behavior.
- [ ] No artifact means current runtime behavior remains unchanged.
- [ ] Legacy `cond_sim/skip_err` artifacts remain usable for `similarity_replace`.

## Test Plan

- [ ] Add profile loader unit tests in `tests/test_draft_reroute.py`:
  - full `.safetensors` schema loads all five tensors and metadata.
  - full `.pt` schema loads all five tensors and metadata.
  - legacy flat `.pt` with only `cond_sim/skip_err` loads as `is_legacy=True`.
  - invalid expert dimensions reject with `ValueError`.
  - metadata mismatch for `num_experts`, `num_layers`, or `top_k` rejects with `ValueError`.
- [ ] Add runtime policy test:
  - `similarity_replace` loaded from new profile produces the same output as the existing direct `cond_sim/skip_err` test.
- [ ] Add loader/cache placement tests:
  - with `act_freq`, initial cache uses top-frequency experts per layer.
  - without `act_freq`, initial cache uses current expert-id order.
  - seeded `access_count` and `access_score_sum` match the fixed `1000` scale.
- [ ] Add LFU/RankGuard test:
  - loaded `act_freq` initializes rank scores as `act_freq * top_k`.
- [ ] Run local targeted tests:
  - `python -m pytest tests/test_draft_reroute.py tests/test_config_prefetch.py tests/test_cache_strategy.py -q`
  - If the login-node Python/Torch NCCL issue appears, record the error and continue with cluster validation.
- [ ] Validate on A100 using `$cluster-compute-workflow`:
  - Reuse an active allocation if available; otherwise request one `salloc -p A100 -N 1 -n 16 --gres=gpu:1 -t 02:00:00 bash`.
  - Enter compute shell, activate `nano_moe`, log `HOST`, `CONDA_DEFAULT_ENV`, `CUDA_VISIBLE_DEVICES`, and `nvidia-smi`.
  - Export smoke profile:
    ```bash
    python scripts/offline_profile.py \
      --model /path/to/model \
      --data-file pre_exps/wikitext2_test.txt \
      --output /tmp/reroute_profile_smoke.safetensors \
      --n-calib 1 \
      --seq-len 128 \
      --device cuda \
      --dtype bfloat16
    ```
  - Run online smoke with `draft_reroute_policy="similarity_replace"` and the smoke profile.
  - Run online smoke with `draft_reroute_policy="entropy_cache_bias"` and the same smoke profile to verify `act_freq` can be used without requiring the similarity path.
  - Save all logs under `/home/mumura/moe_spec/logs`.

## Assumptions And Non-Goals

- “所有离线 profile” means the stable `Calib` outputs from `draft_decode_eval_v2.py`: `sim`, `cond_sim`, `skip_err`, `sens`, and `act_freq`.
- Calibration accumulators inside `calibrate()` are not persisted unless they correspond to a `Calib` output field.
- `act_freq` automatically affects online cache initialization whenever a valid profile path is provided.
- The plan does not implement cache revision or precomputed `best_substitute/best_similarity` refresh tables; runtime continues to use live cache mask gather for `similarity_replace`.
- The plan does not change reroute numerical semantics or add support for `draft_top_c > 0`.
