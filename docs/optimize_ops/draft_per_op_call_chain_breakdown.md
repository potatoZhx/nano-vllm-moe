# Draft Per-Op Call-Chain Breakdown

日期：2026-07-07

## 结论摘要

本轮参考 `verify_per_op_call_chain_breakdown.md` 的粒度，对 draft 路径增加 segment CUDA event 和 per-op CUDA event profile。计时口径分两类：

- 正常 segment-only profile：不在每个 segment 后强制同步，用来判断真实 wall latency、graph 内/外边界。
- op-event profile：在 CUDA graph capture 时插入 `torch.cuda.Event(enable_timing=True, external=True)`，replay 后收集逐 op elapsed time；该路径会在每个 segment 后同步，只用于分析 graph 内占比。

当前 benchmark case：

| 项 | 值 |
|:---|---:|
| output len | 32 |
| max draft tokens K | 4 |
| draft calls | 28 |
| segment size | 12 |
| draft segments/call | 4 |
| cache ratio | 0.3125 |
| slot allocation | `profile_weighted` |
| slot buckets | 4 |
| kt threads | 16 |

关键结果：

| 项 | ms/draft call |
|:---|---:|
| draft wall, normal path | 20.553 |
| draft segment CUDA event, normal path | 16.057 |
| graph 外暴露 gap | 4.496 |
| draft wall, op-event path | 34.821 |
| draft segment CUDA event, op-event path | 22.803 |
| op-event sync overhead | 16.273 |
| op-event 到 normal segment 缩放系数 | 0.7042 |

结论：

1. draft 的主耗时在 CUDA graph 内，正常路径 graph 内 segment event 为 `16.057 ms/call`，约占 draft wall 的 `78.1%`。
2. graph 外暴露 gap 为 `4.496 ms/call`，主要来自 prefetch 前置准备、draft graph 前后的 tail/logits/sampler/acceptance predictor，以及 metadata/prefetch enqueue 的可见开销。
3. graph 内主瓶颈是 GPU MoE 和 attention。按 op-event 缩放到正常 segment 估计，`layer.moe ~= 10.254 ms/call`，`layer.attention ~= 4.040 ms/call`。
4. graph 内非模型计算开销不可忽略：`moe.plan ~= 1.171 ms/call`、`moe.softmax_topk ~= 0.514 ms/call`、`moe.draft_reroute ~= 0.484 ms/call`、`moe.runtime_metadata_record ~= 0.438 ms/call`、`moe.draft_feature_record ~= 0.239 ms/call`。这部分合计已经接近 3 ms/call，是 draft 后续优化的重点。
5. 与 verify 不同，draft 当前没有 `CPUInfer.sync_with_cuda_stream` 这类 graph 内长等待；draft 的核心问题不是 CPU expert 等待，而是 GPU cached experts 的 grouped GEMM、routing/plan 构建、metadata/feature 记录和 graph 外 orchestration 开销。

## 本轮新增 profile 能力

新增/扩展的插桩：

- `nanovllm/utils/verify_op_events.py`
  - 增加 `phase` 字段，支持 `phase="draft"`。
  - 增加 `NANOVLLM_DRAFT_OP_EVENT_TIMING`。
- `nanovllm/engine/model_runner.py`
  - capture draft segment CUDA graph 时用 `verify_op_capture_context(..., phase="draft")` 记录 graph 内 event。
  - replay draft segment graph 后收集 `draft_op_event_records`。
  - `get_profile()` 输出 `draft_op_event_records`。
- `nanovllm/models/qwen3_moe.py`
  - 对 draft normal `Qwen3MoeDecoderLayer.forward()` 增加 `layer.*` 级别 event。
  - 对 `Qwen3MoeHeterogeneousSparseMoeBlock.forward()` 增加 router/topk/metadata/reroute/plan/heterogeneous forward event。
- `nanovllm/layers/fuse_moe/heterogeneous.py`
  - 对 GPU cached expert path 增加 gather/gate_up/down/weight_mul/accumulate/zero event。
- `scripts/analyze_draft_op_breakdown.py`
  - 汇总 `draft_op_event_records`，生成 markdown/json/csv。

生成的结果文件：

- `results/draft_segment_event_k4_l32/profile_weighted_seg12_ratio3125_l32_k4_r0.json`
- `results/draft_op_event_k4_l32/profile_weighted_seg12_ratio3125_l32_k4_r0.json`
- `results/draft_op_event_k4_l32/draft_op_event_records.csv`
- `results/draft_op_event_k4_l32/draft_op_breakdown_summary.md`
- `results/draft_op_event_k4_l32/draft_op_breakdown_summary.json`

## Profile 命令

### 正常 segment-only profile

```bash
source /home/linke/miniconda3/etc/profile.d/conda.sh
conda activate nano_moe
cd /home/linke/nano-vllm-moe
rm -rf results/draft_segment_event_k4_l32

NANOVLLM_DRAFT_SEGMENT_CUDA_EVENT_TIMING=1 \
NANOVLLM_VERIFY_PREFETCH_RANK_MULTIPLIER=1 \
NANOVLLM_VERIFY_DEFER_SEGMENT_METADATA=1 \
NANOVLLM_VERIFY_BOUNDARY_PREFETCH_ASYNC=0 \
CUDA_VISIBLE_DEVICES=2 \
python scripts/bench_per_layer_slots.py \
  --output-dir results/draft_segment_event_k4_l32 \
  --gpu-memory-utilization 0.99 \
  --cache-ratios 0.3125 \
  --output-lens 32 \
  --max-draft-tokens-values 4 \
  --segment-sizes 12 \
  --allocation-modes profile_weighted \
  --slot-buckets 4 \
  --slot-max-bucket-ratio 2.0 \
  --slot-profile-csv pre_exps/exp_and_figs/unique/unique_count_plot_summary_n1024.csv \
  --kt-num-threads 16 \
  --verify-cuda-graph-bucket-steps 3,5,7,10,13 \
  --skip-existing false \
  --case-timeout-sec 3000
```

### Draft per-op event profile

```bash
source /home/linke/miniconda3/etc/profile.d/conda.sh
conda activate nano_moe
cd /home/linke/nano-vllm-moe
rm -rf results/draft_op_event_k4_l32

NANOVLLM_DRAFT_OP_EVENT_TIMING=1 \
NANOVLLM_DRAFT_SEGMENT_CUDA_EVENT_TIMING=1 \
NANOVLLM_VERIFY_PREFETCH_RANK_MULTIPLIER=1 \
NANOVLLM_VERIFY_DEFER_SEGMENT_METADATA=1 \
NANOVLLM_VERIFY_BOUNDARY_PREFETCH_ASYNC=0 \
CUDA_VISIBLE_DEVICES=2 \
python scripts/bench_per_layer_slots.py \
  --output-dir results/draft_op_event_k4_l32 \
  --gpu-memory-utilization 0.99 \
  --cache-ratios 0.3125 \
  --output-lens 32 \
  --max-draft-tokens-values 4 \
  --segment-sizes 12 \
  --allocation-modes profile_weighted \
  --slot-buckets 4 \
  --slot-max-bucket-ratio 2.0 \
  --slot-profile-csv pre_exps/exp_and_figs/unique/unique_count_plot_summary_n1024.csv \
  --kt-num-threads 16 \
  --verify-cuda-graph-bucket-steps 3,5,7,10,13 \
  --skip-existing false \
  --case-timeout-sec 3000
```

### Breakdown 汇总

```bash
python scripts/analyze_draft_op_breakdown.py \
  --op-json results/draft_op_event_k4_l32/profile_weighted_seg12_ratio3125_l32_k4_r0.json \
  --segment-json results/draft_segment_event_k4_l32/profile_weighted_seg12_ratio3125_l32_k4_r0.json \
  --output-dir results/draft_op_event_k4_l32
```

## Draft 完整调用链

顶层 speculative decode 调用链：

```text
SpeculativeEngine.speculative_step
  -> ModelRunner.run_draft(seqs)
     -> _ensure_prefetch_internal_state()
     -> _next_prefetch_step_id()
     -> _flush_pending_prefetch_metadata(block=False)
     -> _set_speculative_execution_mode("draft")
     -> model.set_draft_cpu_graph_mode(True/False)
     -> _decode_graph_policy = "draft"
     -> prefetch_runtime.begin_draft_iteration(step_id)
     -> prefetch_runtime.drain_direct_active_ready(step_id)
     -> prefetch_runtime.maybe_submit_phase1(step_id)
     -> _wait_for_prefetch_device_reuse(mode="draft")
     -> runtime_meta_recorder.arm(mode="draft")
     -> acceptance_extractor.write_state_in(seqs)
     -> ModelRunner.run(seqs, is_prefill=False)
     -> acceptance_extractor.read_outputs(seqs)
     -> runtime_meta_recorder.reset()
     -> _flush_pending_prefetch_metadata(block=False)
```

decode graph dispatch 链：

```text
ModelRunner.run
  -> _prepare_decode / set_context
  -> _run_model_with_graph_policy("draft")
  -> _replay_draft_graph
  -> _replay_draft_segment_graph(input_ids, positions)
```

draft segment CUDA graph replay 链：

```text
_replay_draft_segment_graph
  -> copy input_ids / positions / slot_mapping / context_lens / block_tables
  -> for segment_id in 0..3:
       -> prefetch_runtime.on_draft_segment_start(...)
       -> graph.replay()
          -> Qwen3MoeForCausalLM.forward_draft_segment(...)
             -> first segment only: embed_tokens(input_ids)
             -> Qwen3MoeModel.forward_layers(layer_start, layer_end)
                -> Qwen3MoeDecoderLayer.forward
                   -> input_layernorm
                   -> Qwen3MoeAttention.forward
                      -> qkv_proj
                      -> qkv_split
                      -> q_norm
                      -> k_norm
                      -> rope
                      -> flash_attn_with_kvcache
                      -> o_proj
                   -> attention residual add
                   -> post_attention_layernorm
                   -> Qwen3MoeHeterogeneousSparseMoeBlock.forward
                      -> router_gate
                      -> softmax + topk + topk renorm + dtype cast
                      -> runtime_metadata_record
                      -> draft_reroute_policy
                      -> draft_feature_recorder.record_layer
                      -> build_cached_draft_plan_gpu / build_draft_plan_gpu
                      -> heterogeneous_moe_forward
                         -> output_zero
                         -> _run_gpu_cached_expert_path
                            -> gpu_gather
                            -> fused_moe_linear(gate/up)
                            -> SiluAndMul
                            -> fused_moe_linear(down)
                            -> route weight mul
                         -> accumulate route output
                   -> MoE residual add
             -> last segment only: final_norm
       -> _enqueue_draft_segment_metadata(...)
       -> _poll_dual_queue_segment_timings(block=False)
  -> final_hidden = graph_vars["outputs"][:bs]
  -> model.compute_logits(final_hidden)
  -> acceptance_extractor.set_token_features_from_logits(logits)
  -> draft_tail_graph.replay()
  -> sampler(logits, temperatures)
```

这里有两个边界需要分清：

- `graph.replay()` 内部是 CUDA graph captured 的模型 segment，包括每层 attention/MoE 和最后一个 segment 的 `final_norm`。
- `graph.replay()` 外部仍有 Python/host orchestration、prefetch/metadata enqueue、logits、acceptance predictor tail graph、sampler、acceptance output readback 等工作。

## Graph 外 latency breakdown

正常 segment-only profile 是 graph 外分析的依据：

| item | ms/call | 说明 |
|:---|---:|:---|
| draft wall | 20.553 | `draft_forward_ms_avg` |
| draft segment CUDA event | 16.057 | 4 个 draft segment graph replay 的 CUDA elapsed |
| graph 外暴露 gap | 4.496 | wall - segment CUDA event |
| run_draft mode set | 0.373 | execution mode / graph policy / step 状态 |
| run_draft prefetch before | 1.156 | drain active ready、phase1 submit、device reuse wait、metadata recorder arm |
| run_draft core_run | 18.642 | 包含 graph replay、logits、tail graph、sampler、acceptance readback |
| core_run - segment event | 2.585 | segment graph 外但仍在 `self.run()` 关键路径上的部分 |
| sample decode | 0.548 | sampler 调用 |
| prepare sample decode | 0.021 | temperature 准备 |
| draft metadata enqueue | 0.417 | 每 segment 后 enqueue metadata |
| draft prefetch visible overhead | 1.846 | draft segment-indexed prefetch 可见开销 |
| draft prefetch rank | 0.405 | ranking/选择开销 |
| draft prefetch transfer enqueue | 0.929 | H2D transfer enqueue 开销 |
| metadata collect worker | 1.221 | worker/async 侧 metadata collect，不可直接加到 wall |
| metadata observe worker | 1.971 | worker/async 侧 observe，不可直接加到 wall |
| run_draft submit_after worker | 1.878 | worker/async 侧 submit after，不可直接加到 wall |

关键解释：

- `graph 外暴露 gap = 4.496 ms/call` 是最可信的关键路径上界。
- `metadata collect/observe/submit_after` 是异步 worker 统计，它们会消耗 CPU、锁和 copy 资源，但不能逐项加到 `draft wall`；真实暴露部分已经体现在 4.496 ms gap 中。
- `core_run - segment event = 2.585 ms/call` 是 graph 外最重要的一块，包含 input/static buffer staging、graph replay enqueue、final logits、acceptance predictor tail graph、sampler 和 readback。
- draft 的 graph 外开销已经接近 verify 文档里希望的 `~3 ms` 目标，但仍比目标高约 `1.5 ms/call`。

### Graph 外暴露延迟来源

按调用链对应到关键路径：

| 调用链位置 | 暴露/相关开销 | 主要来源 |
|:---|---:|:---|
| `_set_speculative_execution_mode` / graph policy | 0.373 | Python 状态切换、模型 execution mode 分发 |
| `prefetch_runtime.drain_direct_active_ready` / `maybe_submit_phase1` / `_wait_for_prefetch_device_reuse` / `runtime_meta_recorder.arm` | 1.156 | prefetch 队列清理、冷启动提交、buffer/device reuse 防护、metadata recorder arm |
| per-segment `_enqueue_draft_segment_metadata` | 0.417 | 每个 draft call 4 次 metadata enqueue |
| segment-indexed prefetch ranking | 0.405 | 边界专家候选选择/ranking |
| segment-indexed transfer enqueue | 0.929 | expert H2D copy enqueue 和相关 stream/event 管理 |
| logits + acceptance tail + sampler | 至少 0.55，合计包含在 2.585 | LM head、acceptance feature、tail graph replay、sampler 同步/输出 |

## Graph 内逐算子 breakdown

op-event profile 的 segment CUDA event 为 `22.803 ms/call`，正常 segment-only profile 为 `16.057 ms/call`，所以表中 `scaled` 按 `0.7042` 缩放到正常 segment 时间。`raw` 反映 profile 插桩路径中的真实 event elapsed，`scaled` 用来估计正常路径占比。

### 顶层 graph 内层级

| 层级 | raw ms/call | scaled ms/call | 说明 |
|:---|---:|---:|:---|
| `layer.total` | 22.544 | 15.875 | 48 层 decoder layer 总和 |
| `model.final_norm` | 0.012 | 0.009 | 最后一个 segment 的 final norm |
| segment 内未标注 gap | - | 0.174 | embedding、output slice/store、event 误差等 |
| normal segment CUDA event | - | 16.057 | 正常路径 graph 内总时间 |

### Decoder layer breakdown

| op label | raw ms/call | scaled ms/call | avg ms/op | p95 ms/op | 说明 |
|:---|---:|---:|---:|---:|:---|
| `layer.input_layernorm` | 0.575 | 0.405 | 0.0120 | 0.0143 | RMSNorm |
| `layer.attention` | 5.738 | 4.040 | 0.1195 | 0.1679 | attention block |
| `layer.attn_residual_add` | 0.210 | 0.148 | 0.0044 | 0.0082 | residual add |
| `layer.post_attention_layernorm` | 0.575 | 0.405 | 0.0120 | 0.0143 | RMSNorm |
| `layer.moe` | 14.563 | 10.254 | 0.3034 | 0.3860 | MoE block |
| `layer.moe_residual_add` | 0.201 | 0.142 | 0.0042 | 0.0082 | residual add |
| layer 未标注 gap | 0.682 | 0.480 | - | - | Python wrapper/event nesting/未标注小 op |

### Attention breakdown

| op label | raw ms/call | scaled ms/call | avg ms/op | p95 ms/op | 说明 |
|:---|---:|---:|---:|---:|:---|
| `attn.qkv_proj` | 1.347 | 0.949 | 0.0281 | 0.0338 | QKV projection |
| `attn.qkv_split` | 0.095 | 0.067 | 0.0020 | 0.0041 | q/k/v split/view |
| `attn.q_norm` | 0.619 | 0.436 | 0.0129 | 0.0174 | q norm |
| `attn.k_norm` | 0.597 | 0.421 | 0.0124 | 0.0154 | k norm |
| `attn.rope` | 0.271 | 0.191 | 0.0057 | 0.0102 | RoPE |
| `attn.flash_kvcache` | 0.950 | 0.669 | 0.0198 | 0.0225 | flash attention with KV cache |
| `attn.o_proj` | 1.085 | 0.764 | 0.0226 | 0.0266 | output projection |
| attention 未标注 gap | 0.773 | 0.544 | - | - | wrapper/小 kernel/event nesting |

### MoE breakdown

| op label | raw ms/call | scaled ms/call | avg ms/op | p95 ms/op | 说明 |
|:---|---:|---:|---:|---:|:---|
| `moe.router_gate` | 0.277 | 0.195 | 0.0058 | 0.0102 | router linear |
| `moe.softmax_topk` | 0.731 | 0.514 | 0.0152 | 0.0195 | softmax、topk、renorm、dtype cast |
| `moe.runtime_metadata_record` | 0.621 | 0.438 | 0.0129 | 0.0174 | prefetch metadata record |
| `moe.draft_reroute` | 0.687 | 0.484 | 0.0143 | 0.0184 | draft reroute policy |
| `moe.draft_feature_record` | 0.340 | 0.239 | 0.0071 | 0.0113 | acceptance predictor feature record |
| `moe.plan` | 1.663 | 1.171 | 0.0347 | 0.0379 | cached draft plan build |
| `moe.heterogeneous_forward` | 9.453 | 6.657 | 0.1969 | 0.2365 | GPU cached expert execution |
| MoE 未标注 gap | 0.790 | 0.556 | - | - | plan/forward wrapper、小 kernel/event nesting |

### Heterogeneous GPU expert path breakdown

| op label | raw ms/call | scaled ms/call | avg ms/op | p95 ms/op | 说明 |
|:---|---:|---:|---:|---:|:---|
| `moe.output_zero` | 0.206 | 0.145 | 0.0043 | 0.0082 | 初始化输出 buffer |
| `moe.gpu_gather` | 0.333 | 0.235 | 0.0069 | 0.0102 | route token gather + weight gather |
| `moe.gpu_gate_up` | 5.222 | 3.677 | 0.1088 | 0.1198 | grouped GEMM gate/up |
| `moe.gpu_down` | 2.330 | 1.641 | 0.0486 | 0.0543 | grouped GEMM down |
| `moe.gpu_weight_mul` | 0.258 | 0.182 | 0.0054 | 0.0092 | route weight multiply |
| `moe.accumulate` | 0.419 | 0.295 | 0.0087 | 0.0113 | route output scatter/reduce |
| heterogeneous 未标注 gap | 0.685 | 0.482 | - | - | activation、buffer access、wrapper 小开销 |

### Per-segment breakdown

以下为 op-event raw ms/call，4 个 segment 比较均衡：

| segment | layer.total | layer.moe | attention | hetero | gate_up | down | plan | reroute | metadata_record |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5.482 | 3.569 | 1.387 | 2.344 | 1.314 | 0.590 | 0.413 | 0.169 | 0.150 |
| 1 | 5.630 | 3.632 | 1.438 | 2.350 | 1.297 | 0.578 | 0.415 | 0.172 | 0.156 |
| 2 | 5.713 | 3.687 | 1.455 | 2.388 | 1.313 | 0.584 | 0.416 | 0.173 | 0.158 |
| 3 | 5.719 | 3.675 | 1.459 | 2.372 | 1.298 | 0.579 | 0.420 | 0.174 | 0.158 |

## Graph 外优化分析

### 从操作本身降低开销

1. 降低 `run_draft_prefetch_before`。
   - 当前暴露约 `1.156 ms/call`。
   - 主要动作是 drain、phase1 submit、device reuse wait、metadata recorder arm。
   - 优化方向：把无 work 时的 drain 变成 fast-path；`maybe_submit_phase1` 避免持锁做复杂选择；device reuse wait 只在确有 buffer hazard 时执行；`runtime_meta_recorder.arm` 尽量只更新少量 device scalar/host state。

2. 合并 per-segment metadata enqueue。
   - 当前 `_enqueue_draft_segment_metadata` 约 `0.417 ms/call`，每个 draft call 有 4 个 segment。
   - 如果 prefetch 不依赖精确 segment 边界，可以把 4 次 enqueue 合并成 1 次 call-level enqueue。
   - 如果必须保留边界，可以把 metadata descriptor 预分配为 ring buffer，只写 index/step/layer range，避免每 segment 构造 Python 对象。

3. 优化 segment-indexed prefetch ranking/transfer enqueue。
   - 当前 visible overhead `1.846 ms/call`，其中 rank `0.405 ms/call`，transfer enqueue `0.929 ms/call`。
   - ranking 可借鉴 verify 的 top-N/cache 思路：缓存每层/segment 的候选 expert 列表，只在 routing 分布明显变化时重排。
   - transfer enqueue 应减少小 copy 和 stream/event 管理次数，按 layer/segment 合批提交，避免每个 expert 单独 enqueue。

4. 收缩 logits/tail/sampler 开销。
   - `sample_decode` 为 `0.548 ms/call`，`core_run - segment event` 为 `2.585 ms/call`。
   - 优化方向：把 `compute_logits + acceptance feature + tail graph` 尽量 capture 到同一个 graph/tail graph；sampler 避免额外 host sync；acceptance alpha/readback 借 sampler 已有同步点读取，避免新增同步。

5. 减少 mode switch 和 Python 状态更新。
   - `run_draft_mode_set` 为 `0.373 ms/call`。
   - 可把 draft/verify mode 的 per-layer Python set 调整为模型级轻量 flag 或预绑定 graph policy，避免每 draft call 递归设置大量 module 属性。

### 用异步并行掩盖开销

1. 把 phase1 prefetch 提前到上一轮 verify 或上一轮 draft tail。
   - 当前 `maybe_submit_phase1` 在 `run_draft_prefetch_before` 中触发，天然暴露在 draft graph 之前。
   - 若可以在上一轮 token 接受结果明确前做保守预取，phase1 H2D 可与上一轮 verify/draft 尾部重叠。

2. metadata collect/observe 必须保持后台化。
   - profile 显示 worker 侧 collect `1.221 ms/call`、observe `1.971 ms/call`，这些不应进入主线程关键路径。
   - 需要检查 `_flush_pending_prefetch_metadata(block=False)` 和锁粒度，避免在下一次 `run_draft` 前把 worker backlog 转成主线程等待。

3. segment N 的 metadata 应驱动 segment N+2/N+3 的 prefetch。
   - 现在 per-segment metadata enqueue 在 `graph.replay()` 后执行。
   - 如果 H2D copy 能在下一段 GPU compute 期间完成，transfer enqueue 的 `0.929 ms/call` 就不应暴露。
   - 需要减少 `on_draft_segment_start`/metadata enqueue 与 graph replay 的串行依赖，只保留必要的 event dependency。

4. 避免 prefetch runtime 全局锁包住重操作。
   - `prefetch_runtime` 多处通过 `_prefetch_runtime_lock` 保护。
   - ranking、候选集合构造、统计 observe 不应在锁内执行；锁内只更新队列指针和状态位。

5. 利用 sampler 同步点聚合 readback。
   - acceptance predictor 的 `read_outputs` 在 sampler 已经 forced sync 后执行。
   - metadata readback、acceptance alpha、必要统计应尽量共用这个同步点，避免额外 D2H 同步。

目标拆解：

- 当前 graph 外 gap：`4.496 ms/call`。
- 短期目标：降到 `~3 ms/call`。
- 优先级：先消掉 `run_draft_prefetch_before` 中的暴露等待和 per-segment metadata enqueue，再 capture/合并 logits-tail-sampler。

## Graph 内优化分析

### 模型计算算子

1. `moe.gpu_gate_up` / `moe.gpu_down`
   - scaled 合计 `5.318 ms/call`，是 graph 内最大计算项。
   - 这是 GPU cached experts grouped GEMM，不能消除，只能优化 kernel 和布局。
   - 优化方向：
     - 保证 route 按 slot/expert 连续分组，降低 grouped GEMM 小 group 数量。
     - 复用 plan 产生的 grouped layout，避免 gather 与 GEMM layout 不一致。
     - 针对 draft batch/token shape 做专门 bucket，不让小 batch 走通用 grouped GEMM 配置。
     - 检查 `fused_moe_linear` 是否已经使用最优 BF16/FP16 kernel 和 workspace。

2. attention block
   - `layer.attention` scaled `4.040 ms/call`。
   - qkv/o projection scaled 合计 `1.713 ms/call`，flash KV cache `0.669 ms/call`，q/k norm 合计 `0.856 ms/call`。
   - 优化方向：
     - q/k norm 与 qkv split/RoPE 的小 kernel 融合。
     - 复查 flash attention KV cache kernel 是否对 draft token count/bucket shape 最优。
     - 如果模型精度允许，尝试 projection + norm 的融合路径，但收益可能小于 MoE plan/reroute。

### 非模型计算和路由算子

1. `moe.plan`
   - scaled `1.171 ms/call`，是最大的非模型单项。
   - 当前每层构建 cached draft plan，包括 remap、route grouping、buffer/slot layout。
   - 优化方向：
     - 将 reroute 后 expert id 到 slot 的 LUT lookup、grouped layout 构造和 route index 生成融合为一个 CUDA kernel。
     - 对固定 K/topK/slot bucket 预生成静态 workspace，避免每层重复分配/清零。
     - 对 `profile_weighted` slot buckets 引入 per-layer hot expert bitmap，直接快速判断 all-cached 路径。
     - 如果 draft reroute policy 固定，可把 reroute 和 plan 合并，输出最终 grouped layout。

2. `moe.softmax_topk`
   - scaled `0.514 ms/call`。
   - 包含 softmax、topk、topk renorm、dtype cast。
   - 优化方向：
     - 融合 softmax/topk/renorm，避免完整 router_probs 写回再 topk。
     - 对 topK=8 使用专门 topk kernel。
     - 若 draft 允许近似路由，可评估只对 top candidates 做 partial softmax；但这会影响 draft acceptance，需要验证 tok/s。

3. `moe.draft_reroute`
   - scaled `0.484 ms/call`。
   - 这是 draft 语义引入的额外非模型计算。
   - 优化方向：
     - 将 policy 变成 table-driven LUT，避免每层动态计算。
     - 与 `moe.plan` 融合，直接输出 execution expert slots 和 weights。
     - 如果 reroute 只用于 cache hit，可提前用 per-layer cached expert mask 过滤，减少无效 reroute。

4. `moe.runtime_metadata_record`
   - scaled `0.438 ms/call`。
   - 用于后续 prefetch，但在 graph 内占用真实 stream 时间。
   - 优化方向：
     - 只记录 prefetch 真正需要的字段，例如 expert ids/counts，不记录完整 routing weights。
     - 对 draft 可按 segment 汇总，不必每层写完整 metadata。
     - 使用压缩 dtype/int16 expert id 和固定 ring buffer，减少写带宽和 descriptor 维护。

5. `moe.draft_feature_record`
   - scaled `0.239 ms/call`。
   - acceptance predictor 需要 original vs reroute top-k features。
   - 优化方向：
     - 只保留 predictor 实际读取的 feature，减少 selected/execution 两套完整记录。
     - 将 feature record 与 reroute/plan 的输出写合并。
     - 若 predictor 对部分层贡献低，可评估 layer sampling 或降频记录。

6. `moe.gpu_gather` / `moe.gpu_weight_mul` / `moe.accumulate` / `moe.output_zero`
   - scaled 合计 `0.857 ms/call`。
   - 这些都是非 GEMM 的数据搬运/散射/初始化。
   - 优化方向：
     - 将 `gpu_weight_mul` 融入 down GEMM epilogue 或 accumulate kernel。
     - 避免 `output_zero` 清空完整 hidden output，改为按 route 覆写并在 accumulate 中处理初始化。
     - 将 gather 输入 packing 与 grouped GEMM 前处理融合，减少一次中间 tensor 写读。
     - 针对 topK=8 做固定形状 scatter/reduce kernel，替代通用 `index_copy`/`index_add` 组合。

7. layernorm/residual 小 kernel
   - input/post RMSNorm scaled 合计 `0.810 ms/call`，residual add 合计 `0.290 ms/call`。
   - 单项不大，但数量多。
   - 优化方向：优先级低于 MoE plan/reroute；可在后续尝试 residual add + RMSNorm 融合或持久 buffer 减少写回。

### Graph 内优先级

按收益/风险排序：

1. `moe.plan` 与 `moe.draft_reroute` 融合：潜在收益约 `1.0-1.6 ms/call`，属于非模型计算，风险中等。
2. `runtime_metadata_record` / `draft_feature_record` 精简：潜在收益约 `0.3-0.7 ms/call`，需要确认 prefetch/predictor 所需字段。
3. gather/weight/accumulate/zero 融合：潜在收益约 `0.4-0.8 ms/call`，需要写专用 CUDA kernel。
4. grouped GEMM layout/kernel 调优：理论收益最大，但需要 kernel 层验证，且可能受 expert route 分布限制。
5. attention 小 kernel 融合：收益可能 `0.3-0.8 ms/call`，但不是 draft 特有瓶颈。

## 与 Verify 的差异

| 维度 | verify | draft |
|:---|:---|:---|
| graph 内最大瓶颈 | `kt.cpuinfer_sync`，CPUInfer 未完全被 GPU work 掩盖 | GPU cached expert grouped GEMM + routing/plan |
| CPU expert | verify 有 CPU routes 和 CPU output copy/accumulate | 当前 draft 主要走 reroute 后 GPU cached experts |
| metadata/offload | verify 中曾被怀疑是大头，但关闭后下降有限 | draft metadata record/worker 有开销，但主要以 graph 内小 kernel和 graph 外 enqueue/observe 形式出现 |
| graph 外目标 | 希望降到 `~3 ms` | 当前 `4.496 ms`，离目标约 `1.5 ms` |
| 最值得先做的非模型优化 | CPUInfer overlap/segment pipeline | draft reroute + plan + metadata/feature record 融合/精简 |

因此，verify 的 ktransformers 优化思路不能直接套到 draft CPUInfer 上，因为 draft 的长尾不是 CPUInfer sync；但可以借鉴两个思想：

1. 把路由/plan 的 Python/通用算子开销变成固定 shape、固定 workspace、table-driven 的低开销路径。
2. 把 H2D/metadata/observe 都安排到计算可覆盖的窗口中，主线程只做 enqueue，不做等待和重计算。

## 下一步建议

1. 先做 graph 外优化，把 `4.496 ms/call` 降到 `~3 ms/call`：
   - 对 `run_draft_prefetch_before` 增加子项计时，拆出 drain、phase1、device reuse wait、recorder arm。
   - 合并 `_enqueue_draft_segment_metadata` 或改为 ring descriptor。
   - 检查 `_prefetch_runtime_lock` 粒度，确保 ranking/observe 不在锁内。

2. 再做 graph 内非模型优化：
   - 将 `moe.draft_reroute + moe.plan` 作为第一阶段融合目标。
   - 精简 `runtime_metadata_record` 和 `draft_feature_record`，减少 graph 内写带宽和小 kernel。
   - 把 `gpu_weight_mul + accumulate` 合并，避免多次 route output 读写。

3. 每个优化都用同一套 profile 验证：
   - 正常 segment-only profile 判断真实 draft wall 和 graph 外 gap。
   - op-event profile 判断 graph 内算子占比是否下降。
   - 保持 `output-lens=32, K=4` 的小样本做快迭代，再用 K=12 decode tok/s benchmark 验证吞吐收益。
