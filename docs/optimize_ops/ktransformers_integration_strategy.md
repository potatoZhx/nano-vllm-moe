# Nano-vLLM-MoE and KTransformers Integration Strategy

本文比较两条路线：

1. 将 `nano-vllm-moe` 的算子全部替换成 `ktransformers` 的算子。
2. 将 `nano-vllm-moe` 这条 slot-bucket/spec/CPU expert benchmark 的调用链和功能实现到 `ktransformers` 中。

结论先行：严格意义上的“全量替换 Nano 算子”为低可行性，不建议作为主线；更好的路线是在 KTransformers 内实现 Nano 这条 benchmark 的功能子集，并分阶段把 slot cache、heterogeneous MoE、speculative draft/verify、predictive prefetch 和 CUDA graph 加进去。若短期目标只是优化 CPU expert kernel，则最好继续在 Nano 现有 `kt_direct` 路径上迭代，不做全量算子替换。

## 当前系统边界

### Nano-vLLM-MoE 当前命令依赖的核心能力

目标命令：

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/bench_per_layer_slots.py \
  --output-dir results/per_layer_slots_bench \
  --gpu-memory-utilization 0.99 \
  --cache-ratios 0.3125 \
  --output-lens 512 \
  --max-draft-tokens-values 12 \
  --segment-sizes 12 \
  --allocation-modes profile_weighted \
  --slot-buckets 4 \
  --slot-max-bucket-ratio 2.0 \
  --slot-profile-csv pre_exps/exp_and_figs/unique/unique_count_plot_summary_n1024.csv \
  --kt-num-threads 16
```

它不是单个 MoE kernel benchmark，而是一条完整推理链：

```text
scripts/bench_per_layer_slots.py
  -> benchmarks/scripts/spec_verify_expert_count_stats.py --single-case
  -> LLM(... inference_mode="spec", enable_heterogeneous=True)
  -> HeterogeneousModelLoader
     -> non-expert safetensors -> GPU
     -> expert safetensors -> CPU pool
     -> profile_weighted slot allocation
     -> LayerExpertCache GPU buffers
  -> ModelRunner
     -> PrefetchRuntime(predictive)
     -> capture draft CUDA graphs
     -> capture verify kt-hybrid segment CUDA graphs
  -> SpeculativeEngine
     -> run_draft up to 12 tokens
     -> rollback and run_verify
     -> acceptance strategy
```

算子层面，它同时依赖：

- Nano attention: `store_kvcache` Triton kernel、`flash_attn_varlen_func`、`flash_attn_with_kvcache`。
- Nano linear: `F.linear`，含 tensor parallel shard loader 和 `RowParallelLinear` 的 `dist.all_reduce`。
- Nano heterogeneous MoE: `LayerExpertCache`、LUT、route plan、GPU cached route grouped GEMM、CPU miss route backend。
- Nano GPU expert kernel: `fused_moe_linear -> grouped_gemm_forward -> Triton tl.dot`。
- Nano kt_direct CPU expert backend: `KtDirectCpuMoeBackend -> kt_kernel_ext.CPUInfer -> AMXBF16_MOE/AVX2BF16_MOE forward_task`。
- Nano graph-safe verify: `forward_verify_kt_hybrid`，GPU cached/substitution routes 和 CPU per-token output 合并。
- Nano prefetch/runtime metadata: predictive prefetch、segment boundary、late transfer discard、publish/consume。

### KTransformers 当前命令依赖的核心能力

KTransformers 的 Qwen3 CPU expert benchmark 当前路径：

```text
bench_qwen3_cpu_experts_once.py
  -> Qwen3MoeForCausalLM(meta)
  -> optimize_and_load_gguf(...)
  -> YAML inject
     -> KTransformersLinear / KLinearTorch
     -> KQwen3MoeSparseMoeBlockV2
     -> KTransformersExpertsV2 / KExpertsCPU
  -> StaticCache
  -> prefill
  -> decode loop
  -> CUDAGraphRunner capture/replay after step 2
```

KTransformers 的优势在：

- 注入式 operator framework：`BaseInjectedModule`、YAML rules、`InferenceState`。
- `GGUFLoader/SafeTensorLoader` 和按模块加载权重。
- `KExpertsCPU` 已能把 Qwen3 MoE expert 放在 CPU 上，通过 `CPUInfer -> TP_MOE<LLAMA_MOE_TP> -> llamafile_sgemm` 执行。
- 单 token decode CUDA graph runner 已存在。

KTransformers 当前缺口：

- 没有 Nano 的 per-layer GPU expert slot cache。
- 没有 profile-weighted slot bucket 分配。
- 没有 Nano 的 heterogeneous route plan 和 GPU cached route Triton grouped GEMM。
- 没有等价的 speculative draft/verify engine、acceptance predictor、verify segment graph、predictive prefetch。
- KTransformers model forward 是 HF-style batch/sequence/cache 接口；Nano runner 是 vLLM-like flattened token + paged KV/context 接口。

## 路线 A：将 Nano 算子全部替换成 KTransformers 算子

### 可行性判断

严格全量替换的可行性低。原因不是单个 operator 不能调用，而是两个系统的 operator 边界不一致：

| 模块 | Nano 当前实现 | KTransformers 对应实现 | 替换难度 | 主要问题 |
|---|---|---|---:|---|
| Linear | `F.linear` + TP shard loader | `KTransformersLinear -> KLinearTorch/KLinear*` | 中 | 权重布局、TP shard、`F.linear` vs `x @ weight`、`RowParallelLinear` all-reduce 语义不同 |
| Attention | paged KV + FlashAttention varlen/kvcache | HF attention + `StaticCache`，部分模型支持 flash-attn | 高 | cache layout、`slot_mapping/block_tables/cu_seqlens` 接口不兼容 |
| RMSNorm | torch ops | KTransformers 无明显更优替代 | 低价值 | 替换收益很小 |
| MoE routing | `F.linear -> softmax -> topk -> route plan` | `KQwen3MoeSparseMoeBlockV2` routing | 中 | routing 本身可复用，但后续 slot/plan 语义不同 |
| GPU cached MoE | Triton grouped GEMM | KTransformers 主要是 CPU expert | 高 | KTransformers 没有 LayerExpertCache/GPU cached routes |
| CPU miss MoE | `kt_direct` per-token miss output | `KExpertsCPU` all selected experts CPU output | 中高 | 可用 `-1` mask 跳过 cached routes，但需要新 wrapper |
| CUDA graph | draft/verify segment graphs | 单 decode graph runner | 高 | graph input buffers、segment boundaries、CPU/GPU overlap 不同 |
| Prefetch | predictive runtime | 无 | 高 | 需要 runtime metadata、transfer queues、publish/consume |
| Sampler | `torch.compile` sampler | benchmark-local sampler | 低 | 容易替换，但不是瓶颈 |

因此，“全部替换”会把 Nano 的主要优势：paged attention、slot cache、speculative engine、verify hybrid graph、predictive prefetch 都拆掉或重写。最终工作量接近把 KTransformers 的执行模型塞进 Nano，而不是简单换算子。

### 可执行方案 A

如果仍要走路线 A，应把“全量替换”拆成可验证的阶段，不应一次性替换所有 operator。

#### A0. 定义替换范围和验收指标

先明确目标是：

- 只替换 CPU expert backend；
- 替换 MoE block；
- 替换所有 linear/MoE；
- 还是连 attention/cache/graph 都替换。

建议不要把 attention/cache 纳入第一阶段。验收指标：

- 同 prompt 下 logits top-k 一致或误差在 bf16 容忍范围内。
- `route_hit_rate`、`cpu_routes_sum`、`verify_kt_hybrid_segment_graph_replay_count` 等 profiling 字段不退化。
- 吞吐和 TPOT 与原 Nano baseline 对齐。

#### A1. 先做 KTransformers CPU expert adapter

目标：只替换 Nano 的 CPU miss route backend，保留 Nano 的 `LayerExpertCache`、route plan、GPU grouped GEMM 和 prefetch。

新增一个 Nano backend，例如：

```text
nanovllm/layers/fuse_moe/ktransformers_cpu_backend.py
```

职责：

```text
KTransformersCpuMoeBackend
  -> 用 KTransformers SafeTensorLoader/GGUFLoader 适配 expert weights
  -> 构造 KExpertsCPU 或 KTransformersExpertsV2
  -> forward(hidden_states, selected_experts, routing_weights, plan)
     -> cached routes 对应的 expert id 改成 -1 或 weight 置 0
     -> miss routes 保留原 expert id/weight
     -> 调 KExpertsCPU.forward(...)
     -> 返回 per-token CPU miss output
```

这个阶段有一个可利用点：KTransformers 的 C++ `LLAMA_MOE_TP::forward_one/forward_many` 中已经跳过 `expert_ids == -1` 的 route。因此理论上可以把 GPU cache 命中的 route mask 掉，只让 KExpertsCPU 计算 miss routes。

关键风险：

- `KExpertsCPU` 期望的 weight loader、key 命名、weight layout 和 Nano safetensors loader 不一致。
- `KExpertsCPU.forward` 的 capture 分支和 Nano verify graph 的 buffer 生命周期不同。
- Nano 当前 `kt_direct` 已经是专门为 graph verify 设计的 per-token CPU miss backend，替换成 `KExpertsCPU` 可能更慢或更难 capture。

#### A2. 替换 MoE block 内 CPU/GPU merge 的一部分

在 `Qwen3MoeHeterogeneousSparseMoeBlock.forward` 和 `forward_verify_kt_hybrid` 增加 backend option：

```text
cpu_expert_backend in {"torch", "torch_packed", "fused", "kt_kernel", "kt_direct", "ktransformers_cpu"}
```

保留 Nano 的 route planning：

```text
build_prefill_plan_gpu
build_verify_plan_gpu
build_verify_graph_safe_plan_gpu
```

仅把 CPU miss output 的计算交给 KTransformers backend。

#### A3. 尝试 linear adapter，但不建议作为主优化路径

若必须替换 linear，可做 adapter：

```text
Nano ColumnParallelLinear/RowParallelLinear
  -> KTransformersLinearAdapter
     -> 用 Nano 已加载好的 shard weight 初始化 KLinearTorch
     -> 保持 TP shard shape
     -> RowParallelLinear 仍由 Nano 负责 all_reduce
```

必须加 per-layer numerical tests，因为 KTransformers `KLinearTorch.forward` 是 `x @ self.weight`，Nano `F.linear` 是 `x @ weight.T`。KTransformers loader 可能已经把 GGUF 权重整理成转置布局；直接复用 Nano safetensor weight 很容易方向错。

#### A4. 不建议替换 Nano attention

Nano attention 与 engine/context 强耦合：

```text
get_context()
  -> slot_mapping
  -> block_tables
  -> cu_seqlens_q/cu_seqlens_k
  -> flash_attn_varlen_func / flash_attn_with_kvcache
```

KTransformers `StaticCache` 是另一套 cache 抽象。替换 attention 意味着 ModelRunner、KV cache allocator、Sequence rollback、graph capture 都要变。这个阶段会变成重写 Nano engine，收益不明确。

### 路线 A 的阶段性结论

路线 A 只有“局部替换 CPU expert backend”值得尝试。严格全量替换不可取：

- 技术风险高；
- 性能收益不清晰；
- 容易破坏 Nano 这条 benchmark 真正关注的 slot cache/prefetch/spec/graph 行为；
- 很多 KTransformers operator 的边界是 HF 注入式模型，不适合直接嵌入 Nano 的 vLLM-like runner。

## 路线 B：将 Nano 命令的调用链和功能实现到 KTransformers

### 可行性判断

路线 B 的可行性中等，明显优于路线 A。它的核心思路不是替换 KTransformers 现有 linear/attention，而是在 KTransformers 的 Qwen3 model 和 `KExpertsCPU/CPUInfer` 基础上，加上 Nano 这条命令需要的 heterogeneous slot cache 和 speculative runner。

更准确地说，路线 B 应该分为两个目标等级：

- MVP 目标：在 KTransformers 中跑通 `profile_weighted slot cache + GPU cached routes + CPU miss routes + single-sequence generation`，先不要求 predictive prefetch 和 segment CUDA graph 完全等价。
- 完整目标：复现 Nano 命令的 `spec draft/verify + max_draft_tokens=12 + predictive prefetch + verify kt-hybrid segment graph + profile JSON/log`。

MVP 可控；完整目标工作量较大，但比路线 A 更清晰，因为 KTransformers 的 CPU expert kernel、模型加载和 Qwen3 MoE operator 已经存在。

### 可执行方案 B

#### B0. 增加 KTransformers benchmark CLI

新增脚本，例如：

```text
/home/linke/ktransformers/sosp25-ae/ktransformers-utils/bench_qwen3_slot_buckets_spec.py
```

CLI 对齐 Nano 命令中的关键参数：

```text
--model-path
--cache-ratio
--slots-per-layer
--slot-allocation profile_weighted
--slot-buckets
--slot-max-bucket-ratio
--slot-profile-csv
--max-new-tokens / --output-len
--max-draft-tokens
--segment-size
--prefetch-enabled
--prefetch-runtime-kind predictive
--verify-cuda-graph
--verify-cuda-graph-bucket-steps
--cpu-infer-threads
```

第一阶段可以把 `--prefetch-enabled`、`--verify-cuda-graph` 接住但标记为 unsupported 或 eager fallback，避免 CLI 先失配。

#### B1. Port slot allocation 和 cache 数据结构

从 Nano 移植或重写以下逻辑到 KTransformers：

```text
slot_allocation.py
  -> compute_layer_demand_from_csv
  -> allocate_slots_per_layer

expert_slot_cache.py
  -> LayerExpertCache
  -> gate_up_buffer/down_buffer
  -> expert_to_slot_lut
  -> slot_to_expert_lut
  -> cached_expert_mask
  -> put_to_slot
```

KTransformers 侧需要新增一个 expert weight store：

```text
KTransformersExpertWeightStore
  -> 从 GGUFLoader/SafeTensorLoader 读取每层每个 expert 的 gate/up/down
  -> 规范化成 GPU grouped GEMM 需要的布局:
     gate_up: [2 * intermediate_size, hidden_size]
     down:    [hidden_size, intermediate_size]
  -> 保留 CPU backend 所需的原始 pointers/loader 状态
```

这里要特别验证 weight layout。Nano 的 Triton grouped GEMM 假设 `w` 是 `[expert, N, K]`，kernel 内做 `tl.dot(x, w.T)`；KTransformers loader 中的线性权重可能已经是适配 `x @ weight` 的布局，不能直接假设等价。

#### B2. Port route planning 和 GPU grouped GEMM

移植 Nano 的：

```text
placement.py
  -> build_prefill_plan_gpu
  -> build_verify_plan_gpu
  -> build_verify_graph_safe_plan_gpu

fuse_moe/grouped_gemm.py
  -> grouped_gemm_forward
  -> _grouped_gemm_forward_kernel

fuse_moe/functional.py
  -> fused_moe_linear
```

MVP 中只需要：

```text
selected_experts/routing_weights
  -> remap_experts_to_slots
  -> split GPU cached routes and CPU miss routes
  -> gpu_m_sizes
  -> route_buffer merge
```

#### B3. 新增 KTransformers heterogeneous Qwen3 MoE operator

新增 operator，例如：

```text
ktransformers/operators/heterogeneous_experts.py
  -> KQwen3MoeHeterogeneousSparseMoeBlock
```

替换 YAML：

```yaml
- match:
    name: "^model\\.layers\\..*\\.mlp$"
    class: ktransformers.models.modeling_qwen3_moe.Qwen3MoeSparseMoeBlock
  replace:
    class: ktransformers.operators.heterogeneous_experts.KQwen3MoeHeterogeneousSparseMoeBlock
    kwargs:
      generate_device: "cuda"
      prefill_device: "cuda"
      slot_allocation: "profile_weighted"
      slot_buckets: 4
      slot_max_bucket_ratio: 2.0
```

operator forward：

```text
KQwen3MoeHeterogeneousSparseMoeBlock.forward
  -> flatten hidden_states
  -> router_logits = self.gate(hidden_states)
  -> softmax + topk
  -> plan = build_*_plan_gpu(...)
  -> GPU cached path:
     -> gather hidden by route
     -> fused_moe_linear(gate_up_buffer)
     -> SiluAndMul
     -> fused_moe_linear(down_buffer)
     -> multiply route weights
     -> route_buffer.index_copy_/sum
  -> CPU miss path:
     Option 1: use KExpertsCPU with cached routes set to -1
     Option 2: port Nano kt_direct backend into KTransformers
  -> return GPU output + CPU miss output
```

建议先选 Option 1，因为 KTransformers 已有 `KExpertsCPU` 和 CPUInfer 生命周期。等 MVP 跑通后，再评估是否需要 Option 2 复现 Nano 的 `kt_direct`。

#### B4. 实现 KTransformers single-sequence speculative runner

不要一开始接 KTransformers server。先做脚本级 runner，复用当前 `bench_qwen3_cpu_experts_once.py` 的模式：

```text
load tokenizer/config/model
optimize_and_load_gguf with new heterogeneous YAML
create StaticCache
prefill prompt
for each output token:
  draft phase:
    run model repeatedly up to max_draft_tokens=12
    record draft token ids/logits
  rollback StaticCache sequence length to verify start
  verify phase:
    run target model over prefix token + draft tokens
  acceptance:
    compare draft token with verify distribution
    accept/reject
  update StaticCache and generated ids
```

`StaticCache.change_seq_length(bias)` 已支持通过 bias 改变各层 `past_tokens`，因此可以实现 rollback；KV 内容不需要清零，后续相同位置会被覆盖。需要新增严格测试确保 rollback 后 logits 与不经 draft 的 baseline 一致。

#### B5. Port predictive prefetch

在 B1-B4 跑通后再移植 prefetch：

```text
runtime_meta_recorder
  -> record selected_experts/routing_weights/miss mask per layer

predictive prefetch runtime
  -> history_window demand
  -> transfer budget
  -> staging slots
  -> publish_direct_active_ready
  -> discard late transfers
```

KTransformers 侧可以先实现 layer-level prefetch，再实现 segment-level prefetch。不要一开始复制 Nano 的所有策略参数；先覆盖目标命令实际使用的：

```text
prefetch_runtime_kind=predictive
prefetch_runtime_mode=draft_segment_indexed
prefetch_strategy=history_window
segment_size=12
```

#### B6. Port CUDA graph segment replay

最后实现 graph：

```text
capture draft graphs:
  -> bucket by draft token count
  -> static input token/position/cache_position buffers

capture verify kt-hybrid segment graphs:
  -> bucket steps [3, 5, 8, 12]
  -> split 48 layers into segment_size=12 boundaries
  -> forward layers [start, end)
  -> CPUInfer submit before GPU grouped GEMM
  -> CPUInfer sync after GPU grouped GEMM
```

KTransformers Qwen3 model 当前没有 Nano 的：

```text
forward_draft_segment
forward_verify_kt_hybrid_segment
forward_verify_kt_hybrid_layers
```

因此需要新增 layer-range forward API：

```text
model.forward_layers(
  hidden_states,
  start_layer,
  end_layer,
  cache_position,
  position_ids,
  past_key_values,
  mode,
)
```

这个 API 是实现 segment graph 和 inter-segment prefetch 的关键。

#### B7. 对齐输出和 profiling

为了让结果能和 Nano 命令比较，KTransformers 脚本需要输出：

```text
case config
generated_output_tokens
throughput_output_tok_s
draft_forward_ms_avg
verify_forward_ms_avg
acceptance_rate
route_hit_rate
avg_miss_routes_per_layer
verify_segment_graph_replays
prefetch submit/completed/late/consumed
per-layer route/plan/gpu/cpu timing
```

## 执行方案 Review

### 路线 A 的主要问题

1. “全量替换”目标过宽，且会破坏现有 benchmark 的核心语义。

   Nano 这条命令关注的是 heterogeneous slot cache、prefetch 和 speculative verify，而不是单纯把 `F.linear` 换成某个 KTransformers linear。替换 attention/cache/graph 后，benchmark 可能已经不再测同一个东西。

2. KTransformers operator 是注入式 HF 模型边界，Nano 是 runner/context 边界。

   KTransformers operator 依赖 `BaseInjectedModule`、`GGUFLoader`、`InferenceState`、YAML rule；Nano operator 依赖 `Config`、`LayerExpertCache`、`get_context()`、paged KV、route plan。直接替换会产生大量 adapter glue。

3. Linear 替换收益低但风险真实。

   Nano 的 `F.linear` 已经是稳定路径；KTransformers `KLinearTorch` 的权重布局与 loader 有绑定关系。替换 linear 后性能不一定提升，还要处理 TP shard 和 all-reduce。

4. Attention 替换风险最高。

   Nano 使用 paged KV + FlashAttention kvcache；KTransformers benchmark 使用 `StaticCache`。这不是同一个 operator 签名，替换会牵连 scheduler、cache allocator 和 graph capture。

5. CPU expert 局部替换已有更直接替代方案。

   Nano 当前命令已经使用 `cpu_expert_backend=kt_direct` 和 `kt_num_threads=16`。若目标是优化 CPU expert kernel，更应该沿这个后端改 kernel 或 buffer/overlap，而不是把整个 Nano operator stack 换掉。

### 路线 B 的主要问题

1. 完整复现 Nano 命令仍然是大工程。

   MVP 只需 heterogeneous MoE + single-sequence runner；完整版本还需要 speculative acceptance、predictive prefetch、segment graph 和 profiling parity。

2. 需要在 KTransformers 里引入 Nano 风格的 slot/cache/planning 代码。

   这会产生一套新子系统，需要决定是复制代码、抽公共包，还是只写最小实现。复制最快，但长期维护成本高。

3. `KExpertsCPU` 用作 CPU miss backend 需要验证。

   用 `-1` mask 跳过 GPU cached routes 是可行方向，但必须验证 C++ `forward_one/forward_many`、capture 分支、routing weight 和 output accumulation 在所有 batch size 下行为一致。

4. 完整 `kt_direct` parity 可能仍要移植 Nano backend。

   Nano 当前命令使用的是 `kt_direct`，不是 KTransformers `KExpertsCPU`。如果性能对齐要求严格，最终可能需要把 `KtDirectCpuMoeBackend` 或等价 C++ extension 接入 KTransformers。

5. Segment graph 需要改 KTransformers model forward API。

   当前 KTransformers runner 是整模型 forward；Nano 的 verify graph 是按 layer segment replay。要实现 prefetch overlap，必须新增 layer-range forward，这部分需要小心维护 KV cache、position embedding 和 residual/norm 状态。

## 推荐路线

推荐路线 B，但分阶段执行；不推荐路线 A 的严格全量替换。

理由：

- 路线 B 保留 KTransformers 已经稳定的 Qwen3 model loading、YAML injection、`KExpertsCPU/CPUInfer` 和 decode graph 基础，只把缺失的 slot-bucket/spec 功能补进去。
- 路线 A 会把 Nano 现有成熟的 runner、paged attention、heterogeneous cache 和 graph-safe verify 全部置于风险中，替换收益不明确。
- 路线 B 的每个阶段都有可单独验收的里程碑：slot allocation、GPU cached MoE、CPU miss MoE、spec draft/verify、prefetch、segment graph。
- 如果最终目标是优化 KTransformers CPU expert kernel，路线 B 能直接在 KTransformers 环境里测；如果目标是优化 Nano 命令，现有 Nano `kt_direct` 路径已经是更短路径。

## 建议实施顺序

### 第一阶段：KTransformers heterogeneous MoE MVP

目标：不做 speculative，不做 prefetch，不做 segment graph，只跑通：

```text
profile_weighted slot allocation
LayerExpertCache
GPU cached route Triton grouped GEMM
CPU miss route via KExpertsCPU with -1 route mask
single-sequence decode
```

验收：

- 与 all-CPU KTransformers baseline logits 接近。
- `cache_ratio=0.3125` 时每层 base slots 为 40，profile-weighted 分配总 budget 保持 `40 * 48`。
- route hit/miss 统计正确。
- CPU miss output 与纯 CPU expert output 的 masked-route 对照一致。

### 第二阶段：speculative draft/verify eager

目标：实现 Nano 命令的 decode 语义，但先不捕获 CUDA graph。

```text
draft up to max_draft_tokens=12
StaticCache rollback
verify drafted tokens
acceptance strategy
JSON profiling
```

验收：

- greedy 或固定 seed sampling 下输出可复现。
- rollback 后 verify logits 与无 draft baseline 对齐。
- acceptance rate、draft/verify latency 可输出。

### 第三阶段：predictive prefetch

目标：移植目标命令使用的 `predictive + history_window + segment_size=12`。

验收：

- prefetch submit/completed/late/consumed 计数正确。
- route hit rate 相比无 prefetch 有提升或行为符合预期。
- late prefetch 不阻塞 compute。

### 第四阶段：verify kt-hybrid segment CUDA graph

目标：实现与 Nano 近似的：

```text
verify_cuda_graph_bucket_steps=[3,5,8,12]
segment_size=12
layer-range forward
CPUInfer submit/sync 与 GPU grouped GEMM overlap
```

验收：

- graph replay count 与 verify call/segment 数匹配。
- graph replay logits 与 eager verify 对齐。
- CPU/GPU overlap 时间可从 profiler 观察。

## 最终判断

如果只能选一条主线：

```text
选择路线 B：把 Nano 这条命令的功能分阶段实现到 KTransformers。
```

如果目标是近期产出可运行性能数据：

```text
先不要做任何全量迁移。
继续使用 Nano 现有命令和 kt_direct 后端，
只针对 CPU expert kernel、route planning、prefetch overlap 做局部优化。
```

如果目标是长期统一到 KTransformers：

```text
先做 B1-B3 的 KTransformers heterogeneous MoE MVP，
确认 slot cache + CPU miss backend 能工作；
再决定是否继续投入 speculative/prefetch/segment graph。
```
