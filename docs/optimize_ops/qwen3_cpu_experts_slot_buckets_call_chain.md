# Qwen3 CPU Experts and Slot-Bucket Benchmark Call Chains

本文分析两条 benchmark 命令的调用链，重点落到线性层、attention、MoE 路由、GPU cached expert、CPU expert 后端和 CUDA graph replay 的算子边界。

## 范围和关键参数

第一条命令在 `/home/linke/ktransformers` 下运行：

```bash
source /home/linke/miniconda3/etc/profile.d/conda.sh
conda activate ktransformers
cd /home/linke/ktransformers

rm -f /tmp/ktransformers_qwen3_cpu_experts_16threads.log

PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python sosp25-ae/ktransformers-utils/bench_qwen3_cpu_experts_once.py \
  --model-path /data1/models/Qwen3-30B-A3B \
  --max-new-tokens 64 \
  --sample \
  --temperature 0.6 \
  --top-k 20 \
  --top-p 0.95 \
  --cpu-infer-threads 16 \
  2>&1 | tee /tmp/ktransformers_qwen3_cpu_experts_16threads.log
```

有效的 KTransformers `CPUInfer` 线程配置按 `--cpu-infer-threads 16` 计算。

第二条命令在 `/home/linke/nano-vllm-moe` 下运行：

```bash
source /home/linke/miniconda3/etc/profile.d/conda.sh
conda activate nano_moe
cd /home/linke/nano-vllm-moe

rm -rf results/per_layer_slots_bench

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

`/data1/models/Qwen3-30B-A3B/config.json` 中的关键 MoE 维度是 `num_hidden_layers=48`、`num_experts=128`、`num_experts_per_tok=8`、`hidden_size=2048`、`moe_intermediate_size=768`。因此第二条命令里 `cache-ratio=0.3125` 的基础 budget 是每层 `round(128 * 0.3125) = 40` 个 GPU expert slots；`profile_weighted + 4 buckets + max bucket ratio 2.0` 会在总 slot budget 约束下按层重分配。

## 1. KTransformers CPU Experts Once

入口脚本是 `/home/linke/ktransformers/sosp25-ae/ktransformers-utils/bench_qwen3_cpu_experts_once.py`。

### 初始化和模块替换

主流程：

```text
main
  -> argparse
  -> Config().cpu_infer = 16
  -> AutoTokenizer.from_pretrained(model_path)
  -> AutoConfig.from_pretrained(model_path)
  -> torch.set_default_dtype(config.torch_dtype or bf16)
  -> with torch.device("meta"): Qwen3MoeForCausalLM(config)
  -> optimize_and_load_gguf(model, rule_path, model_path, config)
  -> patch_qwen3_attention_compat(model)
  -> StaticCache(...)
```

默认 `rule_path` 是 `sosp25-ae/ktransformers-utils/Qwen3Moe-serve-bf16-cpu-experts.yaml`。这份 YAML 决定了算子替换：

- `lm_head` 和 `model.layers.*` 中非 `shared_expert_gate` 的 `torch.nn.Linear` 替换成 `KTransformersLinear`，prefill/generate 都使用 `KLinearTorch`，设备为 CUDA。
- `model.layers.*.mlp` 的 `Qwen3MoeSparseMoeBlock` 替换成 `KQwen3MoeSparseMoeBlockV2`。
- `model.layers.*.mlp.experts` 替换成 `KTransformersExpertsV2`，prefill/generate 都使用 `KExpertsCPU`，expert 权重在 CPU，输出回 CUDA。
- `model.embed_tokens` 保持默认 embedding，但 prefill/generate device 是 CPU。
- dense/shared MLP 的 `Qwen3MoeMLP` 替换成 `KQwen2MoeMLP`，设备为 CUDA。

### Prefill 调用链

脚本先用 tokenizer 生成 prompt token，然后显式在 CPU 上做 embedding，再把 embedding 搬到首层 attention 所在 GPU：

```text
tokenizer.apply_chat_template
  -> input_ids
  -> model.model.embed_tokens(input_ids.to("cpu"))
  -> .to(input_device)
  -> model(inputs_embeds=..., cache_position=..., past_key_values=StaticCache, use_cache=True)
```

模型 forward 链：

```text
Qwen3MoeForCausalLM.forward
  -> Qwen3MoeModel.forward
    -> for each Qwen3MoeDecoderLayer.forward
      -> input_layernorm
      -> Qwen3MoeAttention.forward
      -> residual add
      -> post_attention_layernorm
      -> KQwen3MoeSparseMoeBlockV2.forward
      -> residual add
  -> final norm
  -> lm_head
```

Attention 内部主要算子：

```text
q_proj/k_proj/v_proj: KTransformersLinear -> KLinearTorch.forward -> x @ weight
q_norm/k_norm: RMSNorm torch ops
RoPE: cos/sin + rotate/mul/add
KV cache update: StaticCache.update
attention:
  -> repeat_kv
  -> torch.matmul(query, key.transpose)
  -> add attention_mask
  -> softmax(..., dtype=float32).to(query.dtype)
  -> dropout
  -> torch.matmul(attn_weights, value)
o_proj: KTransformersLinear -> KLinearTorch.forward -> x @ weight
```

`KLinearTorch.forward` 的实际算子很直接：

```text
x.to(device=self.device, dtype=self.dtype)
  -> x @ self.weight
  -> optional add bias
  -> x.to(dtype=original_dtype, device=original_device)
```

### Decode 和 CUDA Graph

prefill 后，脚本进入 `max_new_tokens` decode loop。第 2 个 decode step 默认触发 CUDA graph capture：

```text
for step in range(1, max_new_tokens):
  if step == 2 and not disable_cuda_graph:
    CUDAGraphRunner.capture(model, cur_token, position_ids, cache_position, past_key_values, ...)

  if graph_runner is None:
    embed cur_token on CPU
    model(...)
  else:
    graph_runner(cur_token, position_ids, cache_position)

  past_key_values.change_seq_length(1)
  pick_next_token(logits)
```

采样算子由脚本本身执行：

```text
logits[:, -1, :].float()
  -> divide by temperature
  -> torch.topk for top-k
  -> masked_fill below kth score
  -> torch.sort for top-p
  -> torch.softmax
  -> torch.cumsum
  -> scatter top-p removal mask
  -> masked_fill
  -> torch.softmax
  -> torch.multinomial
```

### CPU Expert MoE 调用链

每层 MoE 的核心调用链：

```text
KQwen3MoeSparseMoeBlockV2.forward
  -> hidden_states.view(-1, hidden_size)
  -> router_logits = self.gate(hidden_states)
       self.gate is KTransformersLinear -> KLinearTorch -> x @ weight
  -> routing_weights = softmax(router_logits, dim=1, dtype=float32)
  -> routing_weights, selected_experts = torch.topk(..., top_k=8)
  -> optional top-k probability normalization
  -> self.experts(...)
       KTransformersExpertsV2.forward
         -> KExpertsCPU.forward
           -> CPU/GPU pinned buffer copy
           -> CPUInfer.submit(...) or submit_with_cuda_stream(...)
           -> cpuinfer_ext.moe.MOE.forward(...)
           -> CPUInfer.sync(...) or sync_with_cuda_stream(...)
           -> output CPU -> CUDA copy
```

非 graph eager 分支中，`KExpertsCPU.forward` 会把 `input_tensor`、`expert_ids`、`weights` 转成 contiguous CPU tensor，提交：

```text
self.moe.forward(
  bsz_tensor.data_ptr(),
  expert_ids.size(1),
  expert_ids.data_ptr(),
  weights.data_ptr(),
  input_tensor.data_ptr(),
  output.data_ptr(),
  incremental=False,
)
```

CUDA graph capture/replay 分支中，`KQwen3MoeSparseMoeBlockV2.forward` 会优先走：

```text
KExpertsCPU.submit_for_one_decode(...)
  -> copy hidden/topk/weights/bsz to pinned CPU rolling buffers
  -> CPUInfer.submit_with_cuda_stream(current_cuda_stream, self.moe.forward(...))

KExpertsCPU.sync_for_one_decode(...)
  -> CPUInfer.sync_with_cuda_stream(current_cuda_stream)
  -> output_cpu pinned buffer copy to CUDA output buffer
```

### C++ CPU Expert Kernel 边界

`KExpertsCPU.load` 构造 `MOEConfig`，默认 backend 是 `llamafile`：

```text
MOEConfig(
  n_routed_experts=128,
  top_k=8,
  hidden_size=2048,
  intermediate_size=768,
)
  -> pool = CPUInfer.cpuinfer.backend_
  -> group_min_len = 10
  -> group_max_len = 1024
  -> gate/up/down weight pointers
  -> hidden_type = bf16
  -> self.moe = MOE(moe_config)
  -> CPUInfer.submit(self.moe.load_weights())
```

pybind 侧把 `moe.MOE` 绑定到 `TP_MOE<LLAMA_MOE_TP>`。运行时 C++ 链路是：

```text
cpuinfer_ext.CPUInfer.submit / submit_with_cuda_stream
  -> cpuinfer task queue / CUDA stream fence
  -> TP_MOE<LLAMA_MOE_TP>::forward
    -> read qlen from qlen_ptr
    -> pool->dispense_backend()->do_numa_job(...)
      -> LLAMA_MOE_TP::forward on each NUMA partition
        -> if qlen < group_min_len: forward_one per token
        -> else: forward_many in grouped token batches
    -> merge_results(...)
```

`LLAMA_MOE_TP` 的实际 MoE 算子：

```text
route grouping by expert
  -> gate projection: llamafile_sgemm
  -> up projection: llamafile_sgemm
  -> activation: act_fn(gate) * up
  -> convert intermediate to down input dtype if needed
  -> down projection: llamafile_sgemm
  -> multiply by routing weight
  -> accumulate per-token hidden output
```

因此这条命令的 expert 热点不是 CUDA fused MoE，而是 CPU worker pool 内的 `llamafile_sgemm(gate/up/down)`、路由权重乘加、CPU/GPU buffer copy 和 CUDA-stream 同步。

## 2. Nano-vLLM-MoE Per-Layer Slot Buckets

入口脚本是 `/home/linke/nano-vllm-moe/scripts/bench_per_layer_slots.py`。它本身不直接跑模型，而是生成单个 case，并用子进程调用 `/home/linke/nano-vllm-moe/benchmarks/scripts/spec_verify_expert_count_stats.py --single-case`。

### 子进程参数展开

本命令只生成一个 case：

```text
allocation_mode = profile_weighted
cache_ratio = 0.3125
output_len = 512
max_draft_tokens = 12
segment_size = 12
repeat = 0
```

`bench_per_layer_slots.py` 展开的核心子进程参数包括：

```text
--single-case
--model-path /data1/models/Qwen3-30B-A3B
--cache-ratio 0.3125
--slots-per-layer 0
--slot-allocation profile_weighted
--slot-buckets 4
--slot-max-bucket-ratio 2.0
--slot-profile-csv pre_exps/exp_and_figs/unique/unique_count_plot_summary_n1024.csv
--num-seqs 1
--input-len 1
--output-len 512
--max-draft-tokens 12
--prefetch-enabled true
--prefetch-runtime-mode draft_segment_indexed
--prefetch-runtime-kind predictive
--dual-queue-segment-size 12
--draft-cuda-graph-enabled true
--draft-cuda-graph-cpu-backend none
--draft-prefetch-segment-size 12
--verify-cuda-graph true
--verify-cuda-graph-bucket-steps 3,5,8,12
--verify-prefetch-segment-size 12
--spec-verify-miss-policy cpu
--cpu-expert-execution-enabled true
--cpu-expert-backend kt_direct
--cpu-expert-pin-memory true
--cpu-expert-packed-min-routes 1
--cpu-expert-parallel-mode serial
--kt-num-threads 16
```

单例实验的 stdout/stderr 会写入 `results/per_layer_slots_bench/<case>.log`，原始 JSON 写入 `results/per_layer_slots_bench/<case>.json`，最后再汇总成表格和 summary 文件。

### LLM 构造和 slot-bucket 分配

`spec_verify_expert_count_stats.py` 的 `run_single_case` 先读取 HF config：

```text
num_experts = 128
slots_per_layer argument = 0
slots = round(num_experts * cache_ratio) = 40
effective_cache_ratio = 40 / 128 = 0.3125
```

然后构造：

```text
LLM(
  model_path,
  inference_mode="spec",
  enable_heterogeneous=True,
  enable_speculative=True,
  heterogeneous_slots_per_layer=40,
  heterogeneous_slot_allocation="profile_weighted",
  heterogeneous_slot_buckets=4,
  heterogeneous_slot_max_bucket_ratio=2.0,
  heterogeneous_slot_profile_csv=...,
  max_draft_tokens=12,
  spec_verify_miss_policy="cpu",
  cpu_expert_backend="kt_direct",
  kt_num_threads=16,
  spec_enable_prefetch=True,
  prefetch_runtime_kind="predictive",
  draft_cuda_graph_enabled=True,
  verify_cuda_graph=True,
  verify_cuda_graph_bucket_steps=[3, 5, 8, 12],
  verify_prefetch_segment_size=12,
)
```

`nanovllm/config.py` 中还有一个重要联动：当 `verify_cuda_graph=True`、`cpu_expert_backend="kt_direct"`、`spec_verify_miss_policy="cpu"` 同时成立时，会打开 `verify_cuda_graph_kt_hybrid=True`。因此 verify 阶段不是普通 eager MoE，而是 graph-capturable 的 GPU cached experts + kt_direct CPU miss experts 混合路径。

加载器调用链：

```text
ModelRunner.__init__
  -> HeterogeneousModelLoader.load
    -> _load_non_expert_weights: 非 expert 权重加载到 GPU
    -> _load_expert_weights_to_cpu: expert gate/up/down 留在 CPU
       gate_up = cat([gate, up], dim=0)
    -> _init_layer_caches
       -> _compute_profile_weighted_slots
          -> compute_layer_demand_from_csv(slot_profile_csv)
          -> allocate_slots_per_layer(
               demand,
               total_budget=40 * 48,
               num_experts=128,
               num_buckets=4,
               max_bucket_ratio=2.0,
             )
       -> LayerExpertCache per layer
    -> _load_initial_placement
       -> LayerExpertCache.put_to_slot(...)
```

`LayerExpertCache` 的 GPU 常驻结构：

```text
gate_up_buffer: [num_slots_for_layer, 2 * intermediate_size, hidden_size]
down_buffer:    [num_slots_for_layer, hidden_size, intermediate_size]
expert_to_slot_lut
slot_to_expert_lut
cached_expert_mask
optional staging buffers for prefetch
```

`slot-buckets=4` 只约束每层 slot 数被量化到最多 4 个 bucket；实际每层 slots 仍要满足总 budget、`max_bucket_ratio=2.0` 和 `[min_slots, num_experts]` clamp。它影响的是每层 GPU expert cache 容量、LUT 命中率和 GPU/CPU route 分流，而不是直接改变单个 expert MLP 的矩阵尺寸。

### Engine、Speculative Decode 和 Graph

生成调用链：

```text
spec_verify_expert_count_stats.run_single_case
  -> llm.generate(prompts, sampling_params, use_tqdm=False)
    -> LLMEngine.generate
      -> scheduler.add(...)
      -> while unfinished:
        -> LLMEngine.step
          -> prefill: ModelRunner.run(seqs, is_prefill=True)
          -> decode: SpeculativeEngine.speculative_step(seqs)
```

Speculative step：

```text
SpeculativeEngine.speculative_step
  -> draft loop, up to max_draft_tokens=12
     -> ModelRunner.run_draft(seqs, return_logits=...)
  -> rollback to verify start
  -> prepare verify inputs: accepted prefix token + drafted tokens
  -> optional wait_prefetch_for_verify
  -> ModelRunner.run_verify(seqs, verify_lengths, return_logits=True)
  -> acceptance strategy updates sequences and KV state
```

`ModelRunner.__init__` 会在 warmup/KV cache 初始化后 capture：

```text
capture_draft_cudagraph()
  -> _capture_draft_segment_cudagraph(...) when segment graph enabled

capture_verify_cudagraph()
  -> _capture_verify_cudagraph_kt_hybrid()
    -> _capture_verify_cudagraph_kt_hybrid_segments()
```

verify runtime 使用 segment graph：

```text
ModelRunner.run_verify
  -> _can_use_verify_cudagraph(...)
  -> _run_verify_with_kt_hybrid_segment_graph(input_ids, positions, step_id)
    -> select bucket by num verify tokens, bucket in [3, 5, 8, 12]
    -> copy input_ids/positions/context tensors into graph vars
    -> for each segment boundary:
       -> predictive prefetch runtime publishes ready transfers
       -> optional submit next segment prefetch
       -> graph.replay()
       -> record runtime metadata for next prefetch decision
```

`segment_size=12` 表示 draft/verify graph 和 predictive prefetch 都按 12 层左右的 segment 组织。对于 48 层 Qwen3-MoE，通常会形成 4 个 layer segments。

### Nano Attention 和 Linear 算子

Nano 的 Qwen3 model 在 `/home/linke/nano-vllm-moe/nanovllm/models/qwen3_moe.py`。非 MoE block 的主干算子是：

```text
Qwen3MoeForCausalLM.forward / forward_*_segment
  -> embedding
  -> for each Qwen3MoeDecoderLayer
     -> RMSNorm
     -> Qwen3MoeAttention.forward
        -> QKVParallelLinear -> F.linear
        -> split q/k/v
        -> q_norm/k_norm -> RMSNorm torch ops
        -> rotary embedding
        -> Attention.forward
           -> store_kvcache Triton kernel
           -> prefill: flash_attn_varlen_func
           -> decode/verify: flash_attn_with_kvcache
        -> RowParallelLinear -> F.linear
     -> residual add
     -> RMSNorm
     -> Qwen3MoeHeterogeneousSparseMoeBlock
  -> final norm
  -> lm_head: F.linear(hidden_states, lm_head.weight)
```

RMSNorm 算子是常规 torch 路径：

```text
x.float()
  -> pow(2)
  -> mean(dim=-1, keepdim=True)
  -> rsqrt
  -> multiply by weight
  -> cast back to input dtype
```

### Heterogeneous MoE Eager/Draft 路径

MoE block 常规 `forward` 路径：

```text
Qwen3MoeHeterogeneousSparseMoeBlock.forward
  -> router_logits = self.gate(hidden_states)
       RowParallelLinear -> F.linear
  -> router_probs = softmax(router_logits, dim=1, dtype=float32)
  -> routing_weights, selected_experts = torch.topk(router_probs, top_k=8)
  -> optional top-k probability normalization
  -> plan selection by mode:
       draft: build_cached_draft_plan_gpu or build_draft_plan_gpu
       verify eager + miss_policy=cpu: build_verify_plan_gpu
       normal/prefill: build_prefill_plan_gpu
  -> heterogeneous_moe_forward(...)
```

`build_*_plan_gpu` 的关键 GPU planning 算子：

```text
expert_cache.remap_experts_to_slots(selected_experts)
cached_expert_mask lookup
nonzero / mask split into GPU routes and CPU routes
argsort stable by expert/slot
scatter_add_ into per-slot m_sizes
build cpu_task_expert_ids and cpu_task_offsets for CPU routes
```

GPU cached expert path：

```text
_run_gpu_cached_expert_path
  -> gpu_token_indices = gpu_route_indices // top_k
  -> gpu_hidden = hidden_states[gpu_token_indices]
  -> gpu_weights = flat_weights.index_select(0, gpu_route_indices)
  -> gate_up_buffer, down_buffer = expert_cache.get_layer_buffers()
  -> fused_moe_linear(gpu_hidden, gate_up_buffer, gpu_m_sizes)
       -> grouped_gemm_forward
       -> Triton _grouped_gemm_forward_kernel
       -> tl.dot(x, weight.T)
  -> SiluAndMul activation
  -> fused_moe_linear(activated, down_buffer, gpu_m_sizes)
       -> Triton grouped GEMM
  -> gpu_expert_out *= gpu_weights.unsqueeze(-1)
  -> route_buffer.index_copy_ / view / sum for per-token accumulation
```

CPU route 在普通 `heterogeneous_moe_forward` 中会选择 active CPU backend。此命令启用的是 `kt_direct`，所以 CPU miss routes 通过 `KtDirectCpuMoeBackend.forward`，不是 Python `F.linear` fallback。

### Verify KT Hybrid Graph MoE 路径

在本命令的 verify CUDA graph 中，MoE 不走普通 `forward`，而走：

```text
Qwen3MoeForCausalLM.forward_verify_kt_hybrid_segment
  -> Qwen3MoeModel.forward_verify_kt_hybrid_segment
    -> Qwen3MoeDecoderLayer.forward_verify_kt_hybrid
      -> attention/residual/norm
      -> Qwen3MoeHeterogeneousSparseMoeBlock.forward_verify_kt_hybrid
```

`forward_verify_kt_hybrid` 的算子顺序：

```text
router_logits = self.gate(hidden_states)          # F.linear
router_probs = softmax(router_logits, dim=1)
routing_weights, selected_experts = topk(...)
optional normalize routing_weights

plan = build_verify_graph_safe_plan_gpu(...)
  -> cached routes remain mapped to real GPU slots
  -> uncached routes are represented in cpu_route_mask
  -> graph-safe substitution LUT supplies GPU-side placeholder routes
  -> weights for CPU-selected placeholder routes are zeroed on GPU side

cpu_backend.begin_forward_graph_verify(hidden_states, selected_experts, routing_weights)
  -> copy hidden/topk/weights into pinned CPU buffers
  -> kt_kernel_ext MOE forward_task(...)
  -> CPUInfer.submit_with_cuda_stream(current CUDA stream, task)

GPU cached/substitution path overlaps CPU work:
  -> gpu_token_indices = gpu_route_indices // top_k
  -> gpu_hidden = hidden_states[gpu_token_indices]
  -> gpu_weights = plan.gpu_route_weights.index_select(...)
  -> fused_moe_linear(gpu_hidden, gate_up_buffer, plan.gpu_m_sizes)
  -> SiluAndMul
  -> fused_moe_linear(..., down_buffer, plan.gpu_m_sizes)
  -> multiply by gpu_weights

kt_output = cpu_backend.finish_forward_graph_verify(hidden_states)
  -> CPUInfer.sync_with_cuda_stream(current CUDA stream)
  -> output_cpu pinned buffer copy to CUDA buffer

merge:
  -> route_buffer = zeros([num_tokens * top_k, hidden_size])
  -> route_buffer.index_copy_(0, gpu_route_indices, gpu_expert_out)
  -> token_output = route_buffer.view(num_tokens, top_k, hidden_size).sum(dim=1)
  -> output = token_output + kt_output
```

这里 `kt_output` 是 per-token CPU MoE 输出，已经在 kt_direct kernel 内按 top-k routing weight 聚合；GPU 路径仍是 route-major 输出再 `sum(dim=1)`。

### kt_direct CPU 后端

`kt_direct` 后端在 `/home/linke/nano-vllm-moe/nanovllm/layers/fuse_moe/kt_direct_backend.py`。

全局 runtime：

```text
KtDirectGlobalRuntime.get
  -> resolve kt_threadpool_count and kt_num_threads
     kt_threadpool_count default 1
     kt_num_threads from command = 16
  -> WorkerPoolConfig
  -> kt_kernel_ext.CPUInfer(worker_config)
```

后端初始化：

```text
KtDirectCpuMoeBackend.__init__
  -> _build_bf16_weight_ptrs
     packed gate_up CPU tensor split into gate and up pointers
     down pointers collected
  -> _select_kt_bf16_moe_class
     backend=auto:
       use AMXBF16_MOE if CPU supports amx_bf16 and extension has it
       else use AVX2BF16_MOE if CPU supports avx2 and extension has it
  -> MOEConfig(
       num_experts=128,
       top_k=8,
       hidden_size=2048,
       intermediate_size=768,
       gpu_expert_mask pointer,
     )
  -> self.moe = selected_moe_class(moe_config)
  -> CPUInfer.submit(self.moe.load_weights_task(physical_to_logical.data_ptr()))
  -> CPUInfer.sync()
```

普通 eager CPU route：

```text
KtDirectCpuMoeBackend.forward
  -> KtDirectCPUBuffer.get_buffer(...)
  -> refresh GPU expert mask
  -> copy hidden/topk/routing_weights to pinned CPU
  -> task = self.moe.forward_task(
       batch_size_ptr,
       top_k,
       expert_ids_ptr,
       routing_weights_ptr,
       input_ptr,
       output_ptr,
       incremental=False,
     )
  -> runtime.cpu_infer.submit_with_cuda_stream(current_cuda_stream, task)
  -> runtime.cpu_infer.sync_with_cuda_stream(current_cuda_stream)
  -> output_cpu -> output_device copy
  -> return per-token output on GPU
```

graph verify CPU route：

```text
begin_forward_graph_verify
  -> copy hidden/topk/routing_weights to pinned CPU buffer slot
  -> task = self.moe.forward_task(...)
  -> CPUInfer.submit_with_cuda_stream(current_cuda_stream, task)

finish_forward_graph_verify
  -> CPUInfer.sync_with_cuda_stream(current_cuda_stream)
  -> output_cpu[slot] copy to output_device[slot]
```

和 KTransformers 的 CPU expert 相比，Nano 的 kt_direct 只处理 GPU cache miss routes，并且专门服务 heterogeneous cache + graph-safe verify merge；KTransformers 命令则把所有 routed experts 都放在 CPU backend 上执行。

### Nano 采样算子

`nanovllm/layers/sampler.py` 使用 `torch.compile` 包装采样函数，主要算子是：

```text
logits.float()
  -> optional temperature scaling
  -> softmax
  -> exponential sampling trick: probs / exponential noise
  -> argmax
  -> torch.where for deterministic/temperature cases
```

## 3. 两条命令的算子级对比

### Expert 放置策略

KTransformers 命令：

```text
所有 routed experts 在 CPU
router/gate linear 在 CUDA
MoE MLP gate/up/down 在 CPU llamafile_sgemm
每层 expert 输出再回 CUDA
```

Nano slot-bucket 命令：

```text
每层只缓存一部分 expert 到 GPU slots
命中 GPU cache 的 routes 用 Triton grouped GEMM
miss routes 通过 kt_direct CPUInfer 执行
verify graph 使用 graph-safe substitution + CPU per-token delta/output merge
prefetch runtime 尝试在 draft/verify segment 间搬运 expert cache
```

### 主要热点算子

KTransformers：

- Router: `KLinearTorch -> x @ weight`，然后 `softmax + topk`。
- Attention: Q/K/V/O projection matmul、RoPE、`torch.matmul` attention、softmax、KV cache update。
- CPU MoE: `CPUInfer -> TP_MOE<LLAMA_MOE_TP> -> LLAMA_MOE_TP -> llamafile_sgemm(gate/up/down)`。
- 同步和拷贝: pinned CPU buffers、CPUInfer sync、CPU output to CUDA。
- Sampling: top-k/top-p sort、softmax、multinomial。

Nano：

- Router: `RowParallelLinear -> F.linear`，然后 `softmax + topk`。
- Planning: cache LUT lookup、mask/nonzero、stable argsort、scatter_add_、m_sizes 构造。
- GPU cached MoE: `fused_moe_linear -> Triton grouped_gemm_forward -> tl.dot`，两次 GEMM，中间 `SiluAndMul`。
- CPU miss MoE: `KtDirectCpuMoeBackend -> kt_kernel_ext CPUInfer -> AMXBF16_MOE/AVX2BF16_MOE forward_task`。
- Attention: `store_kvcache` Triton kernel、FlashAttention varlen/kvcache kernels。
- Graph/prefetch: draft segment graph replay、verify kt-hybrid segment graph replay、predictive prefetch H2D transfer 和 publish/consume。

### 优化含义

- 如果目标是优化 KTransformers 命令，优先看 CPU expert 的 `llamafile_sgemm`、`forward_one/forward_many` 分界、CPUInfer 线程/NUMA 配置、pinned buffer copy 和 graph replay 里的 `submit_for_one_decode/sync_for_one_decode`。
- 如果目标是优化 Nano slot-bucket 命令，热点分散在 route planning、GPU grouped GEMM、kt_direct CPU miss kernel、CPU/GPU overlap、slot 分配导致的 miss route 数，以及 prefetch 是否及时命中。
- `slot-buckets` 本身不会改变 expert MLP 的数学算子；它通过改变每层 GPU cache 容量影响 `cached_expert_mask`、`gpu_m_sizes`、CPU route 数、prefetch 压力和 verify graph 中 CPU/GPU merge 的规模。
- 第二条命令的 `--kt-num-threads 16` 作用在 kt_direct `CPUInfer` worker pool；第一条命令的有效 CPUInfer 线程数是 `--cpu-infer-threads 16`。
