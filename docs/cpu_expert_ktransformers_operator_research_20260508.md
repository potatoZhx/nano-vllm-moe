# CPU Expert Backend: kt-kernel Failure and KTransformers Operator Path

Date: 2026-05-08
Hardware tested: `gpu11-A100-E1-3U`, Slurm job `21462`, `CUDA_VISIBLE_DEVICES=5`
Runtime: `nano_moe`, PyTorch `2.9.1+cu128`, kt-kernel `0.6.1.post1`

## Summary

The current kt-kernel backend should not be treated as a reliable implementation path for nano-vllm-moe on the tested A100 node.

Two isolated `KTMoEWrapper` instances can be created, loaded, and used for small forwards when the runtime is forced to the AVX2 BF16 class. However, the real nano-vllm-moe `cpu_expert_backend=kt_kernel` path still segfaults during multi-layer execution when the shared wrapper reloads layer weights inside a model forward.

This means the design in `docs/cpu_expert_imlementation_design.md` needs to be revised: kt-kernel BF16 can remain an experimental probe, but it should not be the primary Phase 4 backend.

## Reproduction Results

### 1. Direct unpatched kt-kernel BF16 is unsafe

Command shape:

```bash
KT_KERNEL_CPU_VARIANT=avx2 python benchmarks/scripts/repro_kt_kernel_multi_instance.py \
  --mode direct_multi_wrapper \
  --weight-path /data1/group_*/mumura/models/Qwen--Qwen3-30B-A3B \
  --layers 0 1 \
  --cpuinfer-threads 16 \
  --threadpool-count 1 \
  --forward-after-load \
  --batch-tokens 4
```

Result:

- Exit status: `132`
- Failure: `Illegal instruction`
- First forward crashes before reaching the second wrapper.
- Cause: the installed single-variant kt-kernel `.so` exposes `AMXBF16_MOE` even though this node has no `amx_bf16` CPU flag. `NativeMoEWrapper` therefore chooses the AMX class unless Python monkey-patches `_HAS_BF16_SUPPORT=False`.

Log:

- `/home/mumura/moe_spec/logs/kt_multi_instance_no_force_20260508_193839.log`

### 2. Forced AVX2 BF16 direct wrappers do not reproduce the simple two-instance crash

Command shape:

```bash
KT_KERNEL_CPU_VARIANT=avx2 python benchmarks/scripts/repro_kt_kernel_multi_instance.py \
  --mode direct_multi_wrapper \
  --weight-path /data1/group_*/mumura/models/Qwen--Qwen3-30B-A3B \
  --layers 0 1 \
  --cpuinfer-threads 16 \
  --threadpool-count 1 \
  --force-avx2-bf16-class \
  --forward-after-load \
  --batch-tokens 4
```

Result:

- Exit status: `0`
- Two `NativeMoEWrapper` instances were created.
- Both layers loaded through `AVX2_BF16_MOE_TP`.
- Both forwards returned finite BF16 output.

Logs:

- `/home/mumura/moe_spec/logs/kt_multi_instance_direct_20260508_193706.log`
- `/home/mumura/moe_spec/logs/kt_multi_instance_forward_20260508_193751.log`

### 3. Current nano-vllm-moe kt-kernel backend still segfaults in real execution

Command shape:

```bash
KT_KERNEL_CPU_VARIANT=avx2 python examples/heterogeneous_benchmark_case.py \
  --model-path /data1/group_*/mumura/models/Qwen--Qwen3-30B-A3B \
  --mode spec \
  --slots-per-layer 32 \
  --num-seqs 1 \
  --input-len 4 \
  --output-len 2 \
  --max-model-len 128 \
  --cpu-expert-execution-enabled true \
  --cpu-expert-backend kt_kernel \
  --enforce-eager true
```

Result:

- Exit status: `139`
- Failure: segmentation fault.
- Stack top:
  - `kt_kernel/utils/amx.py:632 load_weights`
  - `nanovllm/layers/fuse_moe/kt_backend.py:158 _ensure_layer_weights`
  - `nanovllm/layers/fuse_moe/kt_backend.py:266 forward`

Log:

- `/home/mumura/moe_spec/logs/kt_e2e_smoke_20260508_193932.log`

The synthetic nano backend check passed for two layer handles, but the full model path fails because it repeatedly reloads layer weights in the shared wrapper during real multi-layer forward execution. That reloading path is too fragile for production use.

## What "KTransformers CPU Expert Operator" Means Now

The local KTransformers checkout has changed shape:

- Top-level `ktransformers.py` is now a lightweight package entry.
- The maintained inference path is `kt-kernel`.
- The older injected-operator framework is archived under `archive/ktransformers/`.

Relevant archived operator classes:

- `archive/ktransformers/operators/experts.py::KExpertsCPU`
- `archive/ktransformers/operators/experts.py::KTransformersExperts`
- `archive/ktransformers/util/custom_loader.py::SafeTensorLoader`
- `archive/ktransformers/operators/cpuinfer.py::CPUInfer`

The archived `KExpertsCPU` supports:

- `backend="llamafile"` via `cpuinfer_ext.moe.MOE`
- `backend="AMXBF16"` via `cpuinfer_ext.moe.AMXBF16_MOE`
- async-ish CUDA stream submission in some decode/cuda-graph branches
- safetensors and GGUF loading through KTransformers loaders

But it is not immediately usable in the current environment:

- `cpuinfer_ext` is not installed in `nano_moe`.
- `KTransformersOps` is also not available as an importable extension.
- Building the archived framework is a separate dependency path from current `ktransformers -> kt-kernel`.

## Recommended Integration Design

Do not inject KTransformers modules into Qwen3Moe. Instead, add a nano-native backend that implements the existing `CpuMoeResult` contract:

```text
nanovllm/layers/fuse_moe/ktransformers_backend.py
  KTransformersCpuMoeBackend.forward(...) -> CpuMoeResult
```

The backend should own only CPU expert compute. Routing, GPU expert cache, prefetch, and deterministic accumulation should stay in nano-vllm-moe.

### Key API Adaptation

KTransformers `KExpertsCPU.forward(input_tensor, expert_ids, weights)` returns per-token aggregated output. nano-vllm-moe currently expects per-route CPU outputs aligned with `cpu_indices`.

The cleanest bridge is to run the KTransformers operator as a top-1 route backend:

```text
cpu_indices -> route_token_indices = cpu_indices // original_top_k
route_hidden = hidden_states[route_token_indices]
route_expert_ids = selected_experts.view(-1)[cpu_indices].view(num_cpu_routes, 1)
route_weights = routing_weights.view(-1)[cpu_indices].view(num_cpu_routes, 1)
KExpertsCPU configured with num_experts_per_tok=1
output shape: [num_cpu_routes, hidden_size]
```

This avoids computing GPU-cached experts on CPU and matches nano-vllm-moe's existing deterministic route-buffer accumulation.

Avoid using full `top_k` with zeroed GPU routes unless debugging correctness; it wastes CPU compute on GPU-cached routes and may hide performance problems.

### Loader Strategy

Prefer one of these two approaches:

1. Use nano's existing `cpu_expert_pool` to avoid another model-weight loader.
   - Convert each layer's `CpuExpertWeights` into the pointer format expected by a small KTransformers-compatible wrapper.
   - This keeps one source of truth for heterogeneous loading and cache refill.

2. If reusing archived KTransformers loaders first, use `SafeTensorLoader.load_experts("model.layers.{i}.mlp.experts")`.
   - It already supports Qwen3 safetensors names.
   - It converts BF16 tensors to `uint16` numpy arrays and returns GGML type metadata.
   - This is easiest for a proof of concept but duplicates CPU expert weights outside nano's loader.

### Required Dependency Work

Before coding the backend, the legacy operator must be made importable:

```text
python -c "import cpuinfer_ext, KTransformersOps"
```

This currently fails in `nano_moe`. The viable options are:

1. Build archived KTransformers C++ extensions into `nano_moe`.
2. Vendor only the minimal `cpuinfer_ext.moe` pieces required by `KExpertsCPU`.
3. Treat archived `KExpertsCPU` as reference code and implement a nano-local C++ backend with the same top-1 route contract.

Option 1 is fastest for research. Option 3 is cleaner for long-term maintainability.

## Practical Recommendation

Short term:

- Mark `cpu_expert_backend=kt_kernel` as experimental/unsafe on Ice Lake-class nodes.
- Keep the reproduction script in `benchmarks/scripts/repro_kt_kernel_multi_instance.py`.
- Do not build more optimization work on shared-wrapper layer reloading.

Next implementation step:

- Build or expose legacy `cpuinfer_ext` in the `nano_moe` environment.
- Prototype `KTransformersCpuMoeBackend` in top-1 route mode.
- Validate only block-level correctness first, then the same minimal end-to-end smoke that currently segfaults for kt-kernel.

