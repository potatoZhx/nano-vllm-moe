# Verify Per-Op Call-Chain Breakdown

日期：2026-07-06

## 结论摘要

这次重新检查后，之前报告里的两个字段不能作为关键结论：

- `verify_cpu_compute_source=aggregate_fallback`：verify segment graph replay 不会重新进入 Python 的 `forward_verify_kt_hybrid()` / `consume_profile()` 路径；关闭 metadata 后，replay 时没有真实 `verify_cpu_compute_ms`，脚本只能 fallback 到 aggregate/capture profile。这个字段不能解释 replay CPUInfer 时间。
- `verify_gpu_compute_ms_per_call` 只有个位数：它来自 Python `perf_counter()` 包住 GPU MoE 操作 enqueue 的路径，且没有 CUDA 同步；它不是 GPU kernel elapsed time。

新的 per-op CUDA event 计时显示，关闭 verify prefetch 和 runtime metadata 后，verify 主要仍卡在 graph replay 内的 CPUInfer stream dependency：

| 项 | 数值 |
|:---|---:|
| verify calls | 13 |
| verify forward avg | 118.692 ms/call |
| segment CUDA event | 116.849 ms/verify |
| segment CUDA event | 29.212 ms/segment |
| per-op records | 18,109 |
| per-op errors | 0 |

关键路径中最大的真实 op 是：

| op label | ms/verify | avg ms/op | 说明 |
|:---|---:|---:|:---|
| `kt.cpuinfer_sync` | 76.792 | 1.5998 | `CPUInfer.sync_with_cuda_stream` 等待 CPU MoE worker 完成 |
| `moe.cpu_sync_copy` | 77.400 | 1.6125 | 包住 `kt.cpuinfer_sync` + CPU output copy |
| `moe.gpu_gate_up` | 11.579 | 0.2412 | GPU cached experts gate/up grouped GEMM |
| `moe.gpu_down` | 4.772 | 0.0994 | GPU cached experts down grouped GEMM |
| `layer.attention` | 5.218 | 0.1087 | attention 子图总和 |

因此当前瓶颈不是 metadata offload、CPU->GPU output copy 或 GPU grouped GEMM，而是每层 MoE 尾部必须等待 CPUInfer 完成。`kt.output_cpu_to_gpu_copy` 只有 0.360 ms/verify，几乎不是问题。

## 为什么不是 `kt.cpuinfer_sync + attention`

不能用 `76.792 + 5.218 = 82.010 ms` 来估计 verify wall time，因为 `kt.cpuinfer_sync` 不是 CPUInfer 的完整计算时间，而是 GPU path 已经执行之后，`sync_with_cuda_stream` 处仍然没有被 GPU work 隐藏掉的 CPUInfer 等待尾巴。

CUDA graph 中每层 MoE 的顺序大致是：

```text
router / softmax / topk / plan
  -> CPUInfer submit host node
     CPU worker starts asynchronously
  -> GPU cached experts gather + gate/up + down + weight
     overlaps with CPU worker
  -> CPUInfer sync host node
     waits only for remaining CPU worker time
  -> CPU output copy + CPU/GPU output accumulate
```

所以如果把 `kt.cpuinfer_sync` 当成 CPU compute，再认为 GPU MoE 被 CPU 覆盖，就会少算两类时间：

1. sync 之前的串行 stream 时间：router、plan、CPU submit、GPU gather/GEMM/weight 等。
2. sync 之后和 segment 外层的时间：CPU output copy/merge、layernorm/residual、segment output store、graph 外 host overhead 等。

按本次 per-op event，`verify avg = 118.692 ms/call` 可以对齐为：

| 层级 | 构成 | ms/verify |
|:---|:---|---:|
| 用户提出的下界 | `kt.cpuinfer_sync + attention + GPU gate/up + GPU down` | 98.361 |
| MoE 内未计入项 | router、softmax/topk、plan、CPU submit、GPU gather、weight、CPU output wrapper、accumulate、MoE 未标注间隙 | +10.220 |
| layer 非 MoE 项 | input/post layernorm、residual add、layer 未标注间隙 | +2.140 |
| decoder layers total | 48 层 `layer.total` | 110.720 |
| segment graph 外层未标注项 | first segment embedding、segment output slice store、final norm、graph boundary gap | +6.129 |
| segment CUDA event | 4 个 verify segment graph replay | 116.849 |
| graph 外 host/engine overhead | prepare/setup/lm_head enqueue、graph launch/profile bookkeeping 等 | +1.843 |
| verify wall | `verify_forward_ms_avg` | 118.692 |

其中最关键的是：GPU MoE 的确隐藏了一部分 CPUInfer 计算，但这个“已隐藏部分”已经体现为 `kt.cpuinfer_sync` 比完整 CPUInfer compute 更小。也就是说，`kt.cpuinfer_sync` 是 overlap 之后剩下的等待时间，不是可以再拿来和 GPU MoE 做“二选一 max”的完整 CPU 路径。

## 本轮改动

新增 replay per-op event timing：

- `nanovllm/utils/verify_op_events.py`
- `nanovllm/models/qwen3_moe.py`
- `nanovllm/layers/fuse_moe/kt_direct_backend.py`
- `nanovllm/engine/model_runner.py`

核心机制：

- 在 CUDA graph capture 时，用 `torch.cuda.Event(enable_timing=True, external=True)` 包住关键 op。
- 在 graph replay 后收集 event elapsed time。
- 这些 event 是 CUDA stream 时间，能覆盖 graph 内 CUDA kernel 和 host callback dependency，包括 `CPUInfer.sync_with_cuda_stream` 的等待。
- 采集时会在每个 segment replay 后同步，所以这是 profile-only 路径，不应替代普通 benchmark latency。

新增 KTransformers 对照脚本：

- `scripts/bench_ktransformers_cpuinfer_qwen3_moe.py`

该脚本直接调用 KTransformers `cpuinfer_ext.moe.MOE`，Qwen3-30B-A3B MoE shape：`E=128, H=2048, I=768, topK=8`。

## Profile 命令

### Nano verify per-op

```bash
source /home/linke/miniconda3/etc/profile.d/conda.sh
conda activate nano_moe
cd /home/linke/nano-vllm-moe
rm -rf results/verify_op_event_k4_l32

NANOVLLM_VERIFY_DISABLE_RUNTIME_METADATA=1 \
NANOVLLM_VERIFY_SKIP_SYNC_METADATA_READBACK=1 \
NANOVLLM_VERIFY_SEGMENT_CUDA_EVENT_TIMING=1 \
NANOVLLM_VERIFY_OP_EVENT_TIMING=1 \
CUDA_VISIBLE_DEVICES=2 \
python scripts/bench_segment_graph_no_prefetch.py \
  --output-dir results/verify_op_event_k4_l32 \
  --gpu-memory-utilization 0.99 \
  --cache-ratios 0.3125 \
  --output-lens 32 \
  --max-draft-tokens-values 4 \
  --segment-sizes 12 \
  --verify-max-cpu-routes-per-layer 0 \
  --kt-num-threads 16 \
  --case-timeout-sec 2400
```

结果文件：

- `results/verify_op_event_k4_l32/noseg_seg12_ratio3125_l32_k4_cpucap0_r0.json`
- `results/verify_op_event_k4_l32/verify_op_event_records.csv`
- `results/verify_op_event_k4_l32/verify_op_breakdown_summary.md`
- `results/verify_op_event_k4_l32/verify_op_breakdown_summary.json`

`verify_op_event_records.csv` 是逐 op 调用粒度，字段为 `step_id,bucket,segment,token_count,layer_idx,label,elapsed_ms,error`。

### KTransformers qlen 对照

```bash
source /home/linke/miniconda3/etc/profile.d/conda.sh
conda activate ktransformers
cd /home/linke/nano-vllm-moe
rm -f results/ktransformers_cpuinfer_qwen3_moe_qlen_16t.jsonl

python scripts/bench_ktransformers_cpuinfer_qwen3_moe.py \
  --qlen 2 4 8 \
  --threads 16 \
  --layer-num 1 \
  --warmup 50 \
  --iters 1000 \
  --output results/ktransformers_cpuinfer_qwen3_moe_qlen_16t.jsonl
```

### Nano kt-kernel qlen 对照

```bash
source /home/linke/miniconda3/etc/profile.d/conda.sh
conda activate nano_moe
cd /home/linke/nano-vllm-moe

python scripts/bench_qwen3_30b_a3b_cpu_experts.py \
  --backend AVX2BF16 \
  --threads 16 \
  --qlen 2 4 8 \
  --layer-num 1 \
  --warmup 50 \
  --iters 1000 \
  --output results/nano_kt_kernel_qwen3_moe_qlen_16t.jsonl \
  --no-progress
```

## Verify 调用链与开销对应

当前 profile case：

- output len: 32
- max draft tokens: 4
- segment size: 12
- cache ratio: 0.3125
- verify prefetch: off
- runtime metadata offload/readback: off
- kt threads: 16

调用链：

```text
ModelRunner.run_verify
  -> prepare_prefill
  -> _execute_verify_forward
     -> _run_verify_with_kt_hybrid_segment_graph
        -> for each segment: graph.replay()
           -> Qwen3MoeForCausalLM.forward_verify_kt_hybrid_segment
              -> Qwen3MoeModel.forward_verify_kt_hybrid_segment
                 -> Qwen3MoeDecoderLayer.forward_verify_kt_hybrid
                    -> input_layernorm
                    -> Qwen3MoeAttention.forward
                    -> post_attention_layernorm
                    -> Qwen3MoeHeterogeneousSparseMoeBlock.forward_verify_kt_hybrid
                       -> router gate / softmax / topk / plan
                       -> KTDirectMoEBackend.begin_forward_graph_verify
                          -> CPU buffer copies
                          -> CPUInfer forward task create
                          -> CPUInfer.submit_with_cuda_stream
                       -> GPU cached experts gather / gate_up / down / weight_mul
                       -> KTDirectMoEBackend.finish_forward_graph_verify
                          -> CPUInfer.sync_with_cuda_stream
                          -> CPU output copy to GPU
                       -> accumulate CPU and GPU route outputs
                    -> residual add
              -> final_norm on last segment
     -> lm_head F.linear(hidden, lm_head.weight)
```

Function/op 对应表：

| 调用链节点 | event label | ms/verify | 说明 |
|:---|:---|---:|:---|
| all decoder layers | `layer.total` | 110.720 | 48 layers，嵌套总和，不与子项相加 |
| attention block | `layer.attention` | 5.218 | Q/K/V、RoPE、Flash KV cache、O projection |
| MoE block | `layer.moe` | 103.363 | router + CPU/GPU experts + merge |
| router linear | `moe.router_gate` | 0.695 | `self.gate(hidden_states)` |
| softmax/topk | `moe.softmax_topk` | 0.799 | route probability 和 selected experts |
| verify plan | `moe.plan` | 2.046 | CPU/GPU route partition |
| CPU pinned copies | `kt.cpu_prepare_copies` | 0.966 | hidden/topk/weights -> CPUInfer buffers |
| task object | `kt.forward_task_create` | 0.077 | `moe.forward_task(...)` |
| CPU submit | `kt.cpuinfer_submit` | 2.483 | `submit_with_cuda_stream` |
| GPU gather | `moe.gpu_gather` | 0.498 | gather GPU route inputs |
| GPU gate/up GEMM | `moe.gpu_gate_up` | 11.579 | GPU grouped GEMM，属于 CUDA kernel |
| GPU down GEMM | `moe.gpu_down` | 4.772 | GPU grouped GEMM，属于 CUDA kernel |
| GPU weight mul | `moe.gpu_weight_mul` | 0.274 | route weights |
| CPUInfer wait | `kt.cpuinfer_sync` | 76.792 | CPU worker 完成前的 stream dependency |
| CPU output copy | `kt.output_cpu_to_gpu_copy` | 0.360 | CPU output -> GPU |
| CPU sync/copy wrapper | `moe.cpu_sync_copy` | 77.400 | `kt.cpuinfer_sync` + output copy |
| accumulate | `moe.accumulate` | 0.609 | CPU/GPU route output 合并 |
| final norm | `model.final_norm` | 0.013 | last segment norm |

注意：`moe.gpu_gate_up` 和 `moe.gpu_down` 被单独列出，是因为这里按逻辑 op label 拆分；它们本质上仍是 CUDA kernels，不是 CPU 操作。

## Per-Op 结果

Top labels：

| label | count | total ms | ms/verify | avg ms/op | p95 | max |
|:---|---:|---:|---:|---:|---:|---:|
| `layer.total` | 624 | 1439.359 | 110.720 | 2.3067 | 3.1048 | 9.9451 |
| `layer.moe` | 624 | 1343.714 | 103.363 | 2.1534 | 2.9501 | 9.7935 |
| `moe.cpu_sync_copy` | 624 | 1006.195 | 77.400 | 1.6125 | 2.3155 | 9.2245 |
| `kt.cpuinfer_sync` | 624 | 998.296 | 76.792 | 1.5998 | 2.3020 | 9.2160 |
| `moe.gpu_gate_up` | 624 | 150.526 | 11.579 | 0.2412 | 0.3267 | 0.3994 |
| `layer.attention` | 624 | 67.831 | 5.218 | 0.1087 | 0.1096 | 0.1331 |
| `moe.gpu_down` | 624 | 62.042 | 4.772 | 0.0994 | 0.1208 | 0.1608 |
| `moe.cpu_submit` | 624 | 49.830 | 3.833 | 0.0799 | 0.0866 | 0.1302 |
| `kt.cpuinfer_submit` | 624 | 32.285 | 2.483 | 0.0517 | 0.0602 | 0.1000 |
| `moe.plan` | 624 | 26.592 | 2.046 | 0.0426 | 0.0440 | 0.1014 |

Per verify call critical labels：

| label | avg ms/call | min | p50 | max |
|:---|---:|---:|---:|---:|
| `layer.total` | 110.720 | 70.698 | 114.452 | 127.823 |
| `layer.moe` | 103.363 | 63.376 | 107.092 | 120.497 |
| `kt.cpuinfer_sync` | 76.792 | 37.538 | 79.606 | 92.113 |
| `moe.cpu_sync_copy` | 77.400 | 38.114 | 80.240 | 92.706 |
| `moe.gpu_gate_up` | 11.579 | 10.293 | 11.243 | 12.966 |
| `moe.gpu_down` | 4.772 | 4.180 | 4.739 | 5.252 |
| `layer.attention` | 5.218 | 5.190 | 5.215 | 5.272 |
| `moe.plan` | 2.046 | 1.938 | 2.037 | 2.260 |
| `moe.cpu_submit` | 3.833 | 3.759 | 3.840 | 3.879 |

Segment distribution：

| segment | `layer.total` ms/verify | `kt.cpuinfer_sync` ms/verify | `moe.gpu_gate_up` ms/verify | `layer.attention` ms/verify |
|---:|---:|---:|---:|---:|
| 0 | 28.249 | 19.519 | 3.053 | 1.307 |
| 1 | 26.800 | 18.402 | 2.848 | 1.306 |
| 2 | 26.383 | 18.100 | 2.763 | 1.299 |
| 3 | 29.287 | 20.771 | 2.915 | 1.306 |

解读：

- 4 个 segment 的 CPUInfer wait 都在 18-21 ms/verify，整体较均衡。
- segment 3 略高，是本 case 的最慢 segment。
- 单层 `kt.cpuinfer_sync` 有长尾，最大约 9.2 ms，会拉高单次 verify max。
- `verify_op_event_ms` 是嵌套 event 的总和，不能当作 wall time；wall time 应看 segment CUDA event 或 `layer.total` 这类外层 event。

## KTransformers qlen=2/4/8 对照

KTransformers `cpuinfer_ext.moe.MOE`，BF16，llamafile backend，16 threads：

| qlen | routes/call | avg ms/layer | p50 ms | p95 ms | us/route | 48-layer serial estimate |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 16 | 1.507 | 1.508 | 1.539 | 94.16 | 72.31 ms |
| 4 | 32 | 3.009 | 3.011 | 3.054 | 94.02 | 144.41 ms |
| 8 | 64 | 6.018 | 6.018 | 6.097 | 94.03 | 288.86 ms |

Nano `kt_kernel_ext` AVX2BF16，16 threads：

| qlen | routes/call | avg ms/layer | p50 ms | p95 ms | us/route |
|---:|---:|---:|---:|---:|---:|
| 2 | 16 | 1.691 | 1.719 | 1.773 | 105.70 |
| 4 | 32 | 3.199 | 3.203 | 3.438 | 99.96 |
| 8 | 64 | 6.031 | 6.030 | 6.501 | 94.23 |

对比：

- KTransformers qlen 2/4/8 基本严格线性，约 94 us/route。
- Nano kt-kernel qlen 4/8 与 KTransformers 在同量级，qlen=8 几乎一致；qlen=2 略慢。
- 当前 verify case 的 route 数由于关闭 metadata 只能看到 `aggregate_fallback`，不能当作 replay 精确计数。粗略值约 `29.3 routes/layer/verify`，等价 qlen 约 3.66。
- 若按 KTransformers 94 us/route 估算，48 层纯 CPU serial 约 `29.3 * 94 us * 48 = 132 ms`；Nano replay 里 `kt.cpuinfer_sync` wait 是 76.8 ms/verify。这个差异不表示 KTransformers 慢或 Nano 计数矛盾，因为：
  - `kt.cpuinfer_sync` 是剩余等待时间，不是完整 CPU compute time。
  - CPUInfer submit 后与 GPU cached expert GEMM 有一段 overlap。
  - 关闭 metadata 后 CPU route count 不是 replay 精确值。
  - 每层 route 分布不均，critical path 由最慢层/segment 决定，不等同总 routes 线性和。

整体看，Nano 当前使用的 kt CPU expert 算子本身没有明显异常慢；verify latency 的问题主要是 CPUInfer work 处在每层 MoE 的硬同步点上。

## 可优化项

### 低收益或不是瓶颈

- `kt.output_cpu_to_gpu_copy`：0.360 ms/verify。即使完全消除，收益也很小。
- `kt.cpu_prepare_copies`：0.966 ms/verify。可优化但不是主瓶颈。
- `kt.forward_task_create`：0.077 ms/verify。几乎可忽略。
- `router_gate + softmax_topk + plan`：约 3.54 ms/verify。可做融合或减少 plan 开销，但不是主因。
- GPU cached expert grouped GEMM：`gate_up + down + weight/gather` 约 17.1 ms/verify，小于 CPUInfer wait。

### 主要优化方向

1. 降低 CPU routes
   - 提高 GPU expert cache hit：更好的 slot allocation、profile weighted、verify 前热 expert 保护。
   - 针对 verify 的短 qlen 分布优化 cache，而不是只看 draft/prefill 统计。
   - route cap 会改变 acceptance/正确性路径，不能作为默认优化，只能作为实验上界。

2. 降低 CPUInfer 每 route 时间
   - 当前 qlen=2/4/8 仍近似线性，说明小 batch 没吃到明显 grouped/batched GEMM 收益。
   - 可以评估 kt-kernel/KTransformers 对 qlen<10 的 grouped path 或专门小 qlen kernel。
   - 检查 `kt_threadpool_count` / NUMA / CPU affinity。KTransformers 端到端 run 使用 `cpu_affinity=32-47`；当前 Nano 命令没有 pin CPU，可能引入 worker tail。

3. 减少同步点数量或隐藏同步
   - 当前每层 MoE 都有 `CPUInfer.sync_with_cuda_stream`，48 层累计成主路径。
   - Nano 已经使用 `submit_with_cuda_stream`，能隐藏一部分 CPUInfer 在 GPU cached expert GEMM 后面，但 GPU work 只有约 17 ms/verify，无法隐藏 76 ms 的 CPU wait。
   - 真正的大收益需要把 CPU task 更早提交，或者把低优先级 CPU routes 延迟/分层处理；但 exact verify 语义要求每层输出在下一层前完整合并，不能简单跨层推迟。

4. 借鉴 KTransformers
   - 已可借鉴：rolling pinned buffers、CUDA graph host dependency、CPU affinity/NUMA 配置、固定 shape buffer。
   - Nano kt-direct 已经具备 `submit_with_cuda_stream`/`sync_with_cuda_stream` 的基本机制，缺的不是“是否使用 host callback”，而是 CPU work 太大且同步点太密。
   - KTransformers 的 deferred expert 机制需要谨慎；它不是可以直接搬到 exact verify 的无损优化，除非证明延迟输出不会改变下一层输入或引入可接受的近似策略。

## 下一步建议

优先实验：

1. 固定 CPU affinity 重跑 per-op verify，比较 `kt.cpuinfer_sync` 的 p95/max 是否下降。
2. 对 `kt_threadpool_count=1/2`、NUMA nodes 做 sweep，看 per-layer `kt.cpuinfer_sync` 是否改善。
3. 在 metadata 开启的小样本上同时记录真实 CPU routes，再与 per-op event 对齐，建立 `routes/layer -> kt.cpuinfer_sync` 的回归。
4. 给 kt C++ worker 增加内部计时，拆到 expert grouping、gate/up/down GEMM、activation、merge；CUDA event 只能看到 host callback wait，不能看到 CPUInfer worker 内部每个 GEMM。

## Prefetch + Metadata On

用户指出 metadata offload 和 prefetch 会减少 CPU experts，因此又补跑了同一小 case，但这次打开完整 metadata 和 prefetch：

```bash
source /home/linke/miniconda3/etc/profile.d/conda.sh
conda activate nano_moe
cd /home/linke/nano-vllm-moe
rm -rf results/verify_op_event_prefetch_on_k4_l32

NANOVLLM_VERIFY_SEGMENT_CUDA_EVENT_TIMING=1 \
NANOVLLM_VERIFY_OP_EVENT_TIMING=1 \
CUDA_VISIBLE_DEVICES=2 \
python scripts/bench_verify_boundary_overhead.py \
  --output-dir results/verify_op_event_prefetch_on_k4_l32 \
  --modes verify_prefetch_on \
  --gpu-memory-utilization 0.99 \
  --cache-ratios 0.3125 \
  --output-lens 32 \
  --prefetch-on-max-draft-tokens-values 4 \
  --segment-sizes 12 \
  --allocation-mode profile_weighted \
  --slot-buckets 4 \
  --slot-max-bucket-ratio 2.0 \
  --slot-profile-csv pre_exps/exp_and_figs/unique/unique_count_plot_summary_n1024.csv \
  --kt-num-threads 16 \
  --acceptance-predictor-enabled false \
  --draft-stop-policy none \
  --case-timeout-sec 2400
```

结果文件：

- `results/verify_op_event_prefetch_on_k4_l32/verify_prefetch_on_seg12_ratio3125_l32_k4_r0.json`
- `results/verify_op_event_prefetch_on_k4_l32/verify_op_event_records.csv`
- `results/verify_op_event_prefetch_on_k4_l32/verify_op_breakdown_summary.md`
- `results/verify_op_event_prefetch_on_k4_l32/verify_op_breakdown_summary.json`

### Summary

| metric | no prefetch / metadata off | prefetch + metadata on |
|:---|---:|---:|
| verify avg | 118.692 ms | 91.872 ms |
| segment event | 116.849 ms/verify | 70.428 ms/verify |
| segment event | 29.212 ms/segment | 17.607 ms/segment |
| layer.total | 110.720 ms/verify | 64.084 ms/verify |
| layer.moe | 103.363 ms/verify | 56.633 ms/verify |
| `kt.cpuinfer_sync` | 76.792 ms/verify | 25.711 ms/verify |
| route hit | 0.2902 | 0.7241 |
| CPU routes/call | fallback, 1404.8 | verify_profile, 506.1 |
| CPU routes/layer/call | fallback, 29.27 | verify_profile, 10.54 |
| CPU experts/call | fallback, 161.2 | verify_profile, 330.2 |
| op records | 18,109 | 12,969 |

这里 prefetch+metadata-on 的 CPU route/expert 计数来自 `verify_profile`，可信；no-prefetch/metadata-off 的 CPU routes 是 `aggregate_fallback`，只能做参考，不应作为真实 replay route count。

关键变化：

- `kt.cpuinfer_sync` 从 76.792 降到 25.711 ms/verify，说明 prefetch/metadata 确实减少了 verify CPU miss route 后的 CPUInfer 等待。
- `layer.moe` 从 103.363 降到 56.633 ms/verify。
- segment graph 内时间从 116.849 降到 70.428 ms/verify，下降 46.421 ms。
- verify wall 只从 118.692 降到 91.872 ms，下降 26.820 ms。差值主要被 prefetch/metadata 的 graph 外开销吃掉。

### MoE Breakdown

prefetch + metadata on 的 MoE 内顺序分解：

```text
layer.moe 56.633
≈ router_gate 0.700
+ softmax_topk 0.810
+ plan 2.052
+ runtime_metadata_record 0.850
+ cpu_submit 4.032
+ gpu_gather 0.502
+ gpu_gate_up 12.013
+ gpu_down 4.908
+ gpu_weight_mul 0.283
+ cpu_sync_copy 28.916
+ accumulate 0.610
+ uninstrumented gap 0.957
```

`cpu_submit` 可继续拆成：

```text
cpu_submit 4.032
≈ kt.cpu_prepare_copies 0.986
+ kt.forward_task_create 0.083
+ kt.cpuinfer_submit 2.635
+ wrapper gap 0.328
```

`cpu_sync_copy` 可继续拆成：

```text
cpu_sync_copy 28.916
≈ kt.cpuinfer_sync 25.711
+ kt.output_cpu_to_gpu_copy 2.976
+ wrapper gap 0.229
```

与 metadata off 的 case 相比，`kt.output_cpu_to_gpu_copy` 从 0.360 ms/verify 升到 2.976 ms/verify。原因是 prefetch 后 CPU route 数减少但 realized CPU expert set 仍较多，且 metadata/prefetch-on case 的 route/expert 分布不同；不过主项仍是 `kt.cpuinfer_sync`。

### Layer And Segment Breakdown

```text
layer.total 64.084
≈ input_layernorm 0.656
+ attention 5.293
+ attn_residual_add 0.185
+ post_attention_layernorm 0.630
+ layer.moe 56.633
+ moe_residual_add 0.184
+ layer gap 0.502
```

```text
segment event 70.428
≈ layer.total 64.084
+ segment external gap 6.345

verify avg 91.872
≈ segment event 70.428
+ graph-external host/metadata/prefetch gap 21.443
```

graph-external gap 的主要可见项：

| item | ms/verify |
|:---|---:|
| verify boundary prefetch submit | 8.148 |
| verify segment prefetch hook | 4.283 |
| verify metadata profile loop | 5.324 |
| verify prepare prefill | 0.137 |
| verify graph setup enqueue | 0.106 |
| lm_head enqueue | 0.148 |
| metadata status+activation CPU readback | 0.085 |
| remaining/unattributed host gap | about 3.212 |

注意：metadata offload worker profile 里还有 `collect/observe/async_turnaround` 等字段，但其中相当一部分是异步/隐藏时间，不能直接累加到 verify critical path。

### Prefetch / Metadata Stats

| metric | value |
|:---|---:|
| verify calls | 9 |
| verify tokens/call | 4.78 |
| verify CPU routes/call | 506.1 |
| verify CPU routes/layer/call | 10.54 |
| realized CPU experts/call | 330.2 |
| realized CPU experts/layer/call | 6.88 |
| verify pre-transfer active routes/call | 1834.7 |
| verify segment prefetch submits/call | 46.6 |
| verify boundary visible overhead/call | 8.108 ms |
| verify candidate rank/call | 2.057 ms |
| verify submitted MB/call | 439.4 MB |
| prefetch late count | 0 |

结论：

- prefetch/metadata 明显降低了 segment 内 CPUInfer wait，是有效的。
- 当前 prefetch-on latency 没有按同幅度下降，是因为 verify boundary prefetch ranking/submit 和 metadata profile/readback 带来约 21 ms/verify 的 graph 外开销。
- 下一步优化重点应从纯 CPUInfer 转向两条线并行：
  1. 继续降低 CPU routes，尤其降低 `kt.cpuinfer_sync`。
  2. 减少 verify boundary prefetch 和 metadata profile loop 的主线程可见开销，把 ranking/observe/readback 尽量异步化或延后到不阻塞 verify 的位置。

## Graph-External Optimization

### Code Changes

本轮针对 graph 外开销做了三类修改：

1. 关闭默认同步 metadata profile readback。
   - 原来的 verify 末尾会在主线程执行 `expert_status.cpu()`、`activation_count.cpu()` 和 Python layer loop。
   - 现在默认不走这条同步路径，只在 `NANOVLLM_VERIFY_SYNC_METADATA_PROFILE_READBACK=1` 时回退。
   - profile 统计改由 metadata worker 在 host buffer 上汇总，并写入 `verify_cpu_routes_sum`、`verify_realized_cpu_expert_count_sum` 等 verify-specific counters。

2. verify segment metadata 默认延后到整个 verify graph 结束后一次性 offload。
   - 回退开关：`NANOVLLM_VERIFY_DEFER_SEGMENT_METADATA=0`。
   - 目的不是减少 metadata 总工作量，而是避免每个 segment 后的 D2H metadata copy 和 observe worker 与下一段 verify graph/CPUInfer 抢资源。

3. verify boundary prefetch 细分计时并限流。
   - 新增 ranking scan/sort、filter、victim select、reservation、transfer enqueue、bookkeeping 计时。
   - `verify_prefetch_max_per_boundary` 默认从 16 调到 6。
   - 增加了 `NANOVLLM_VERIFY_BOUNDARY_PREFETCH_ASYNC=1` 的后台 submit 实验路径，但实测没有成为默认，因为它不能稳定降低 wall time，且大预算时会放大 H2D 与 graph 的资源竞争。

### Commands

真实 wall profile 关闭 per-op event sync，只保留 segment CUDA event：

```bash
source /home/linke/miniconda3/etc/profile.d/conda.sh
conda activate nano_moe
cd /home/linke/nano-vllm-moe

NANOVLLM_VERIFY_SEGMENT_CUDA_EVENT_TIMING=1 \
NANOVLLM_VERIFY_BOUNDARY_PREFETCH_ASYNC=0 \
CUDA_VISIBLE_DEVICES=2 \
python scripts/bench_verify_boundary_overhead.py \
  --output-dir results/verify_external_budget6_breakdown_k4_l32 \
  --modes verify_prefetch_on \
  --gpu-memory-utilization 0.99 \
  --cache-ratios 0.3125 \
  --output-lens 32 \
  --prefetch-on-max-draft-tokens-values 4 \
  --segment-sizes 12 \
  --allocation-mode profile_weighted \
  --slot-buckets 4 \
  --slot-max-bucket-ratio 2.0 \
  --slot-profile-csv pre_exps/exp_and_figs/unique/unique_count_plot_summary_n1024.csv \
  --kt-num-threads 16 \
  --acceptance-predictor-enabled false \
  --draft-stop-policy none \
  --verify-prefetch-max-per-boundary 4 \
  --case-timeout-sec 2400
```

结果文件：

- `results/verify_external_budget6_breakdown_k4_l32/summary.md`
- `results/verify_external_budget6_breakdown_k4_l32/summary.json`
- `results/verify_external_budget6_breakdown_k4_l32/verify_prefetch_on_seg12_ratio3125_l32_k4_r0.json`

### Boundary And Metadata Breakdown

早期最佳单次观测是 `verify_external_budget6_breakdown_k4_l32`：

```text
verify avg 71.079 ms/call
≈ segment event 67.611
+ graph-external gap 3.468
```

boundary prefetch 可见开销：

```text
verify boundary 4.145 ms/call
≈ rank_scan 1.628
+ rank_sort 0.217
+ filter 0.281
+ victim_select 0.384
+ reservation 0.032
+ transfer_enqueue 1.246
+ bookkeeping 0.127
+ remaining gap 0.230
```

metadata offload/observe 分解：

```text
metadata enqueue 0.132 ms/call
metadata readback wait 12.454 ms/call   # worker/hidden; not主线程关键路径
metadata collect 5.927 ms/call          # worker/hidden
metadata observe 2.635 ms/call          # worker/hidden
≈ observe_call 2.173
+ record_consumed 0.458
metadata async profile loop 2.230 ms/call
sync profile loop 0.000 ms/call
```

`readback wait/collect/observe/profile_async_loop` 是 metadata worker 上的时间；它们会占用 CPU/GPU copy resource，但不应直接加到主线程 graph-external gap。优化后真正的同步 metadata profile loop 已经消失。

### Budget Sweep

| run | verify | segment event | gap | CPU routes/call | submit/call | MB/call | boundary | metadata observe | hit |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| per-op sync, budget 16 | 91.872 | 70.428 | 21.443 | 506.1 | 46.6 | 439.4 | 8.108 | 0.000 | 0.7241 |
| no per-op sync, budget 16, segment metadata | 101.159 | 93.781 | 7.379 | 468.6 | 60.0 | 566.2 | 8.374 | 23.904 | 0.7497 |
| deferred metadata, budget 16 | 114.220 | 108.441 | 5.780 | 461.3 | 60.0 | 566.2 | 7.491 | 2.199 | 0.7543 |
| deferred metadata, budget 8 | 82.089 | 77.994 | 4.094 | 582.9 | 32.0 | 302.0 | 4.994 | 3.143 | 0.6886 |
| deferred metadata, budget 6 | 71.079 | 67.611 | 3.468 | 554.8 | 24.0 | 226.5 | 4.145 | 2.635 | 0.7111 |
| deferred metadata, budget 6, async submit | 72.028 | 68.456 | 3.571 | 554.8 | 24.0 | 226.5 | 9.382 | 2.951 | 0.7111 |
| deferred metadata, budget 4 | 95.542 | 92.446 | 3.096 | 638.4 | 16.0 | 151.0 | 3.870 | 2.148 | 0.6500 |
| rank cache + multiplier=1 + budget 4 | 74.901 | 71.870 | 3.031 | 615.1 | 16.0 | 151.0 | 1.947 | 5.781 | 0.6104 |

解释：

- per-op event profile 会在每个 segment 后同步 GPU，只适合做 graph 内算子分解；它会改变 graph 外工作和 graph compute 的重叠关系。
- budget 16 提交 566 MB/call，虽然 CPU routes 少，但 H2D prefetch 和 metadata worker 会拖慢 graph 内 CPUInfer/GPU kernels。
- budget 4 把 graph 外 gap 压到约 3 ms，但 CPU routes 增到 638/call，CPUInfer 反噬，verify 变慢。
- rank cache + multiplier=1 后，budget 4 成为当前短测最佳折中点：H2D 约 151 MB/call，CPU routes 约 615/call，verify 约 74.9 ms，graph-external gap 约 3.03 ms。
- async boundary submit 没有带来收益；它把 submit 从主线程移走，但仍要 enqueue 同样的 H2D，且需要 drain/线程调度，实测略慢。

### Current Bottlenecks

1. 主要瓶颈已经不是同步 metadata readback；该路径已基本消除。
2. 当前关键约束是 prefetch H2D volume 与 CPU routes 的 tradeoff：
   - prefetch 太多：H2D/copy engine 和 worker 竞争拖慢 graph。
   - prefetch 太少：CPU routes 增加，`kt.cpuinfer_sync` 变大。
3. graph-external gap 已从约 21.4 ms 降到约 3.0 ms 的量级；当前短测达到 verify 约 74.9 ms，但还需要 repeat>=3 和更长 output_len 验证稳定性。

### Next Optimization

- 做 repeat>=3 的 budget=4 复测，并用 output_len=128/512 验证 graph-external gap 约 3 ms 是否稳定。
- 对 verify prefetch 做 adaptive budget：按最近 segment event / CPU routes / H2D MB 自动在 3/4/6 之间切换，而不是固定预算。
- metadata observe 可以继续减负：`observe_call` 约 2.17 ms/call，`record_consumed` 约 0.46 ms/call；这部分已经不在主线程，但仍会占 CPU，可进一步移到更低优先级或 verify 后的 decode/draft 间隙。

### Per-Layer CPU Expert Count Instrumentation

已增加每层 CPU 计算 expert 数量统计，口径是 verify replay 中实际走 CPUInfer 的 active miss experts：

```text
expert_status == 2 and activation_count > 0
```

新增 engine profile counter：

- `verify_layer_{idx}_realized_cpu_expert_count_sum`
- `verify_layer_{idx}_cpu_routes_sum`
- `verify_layer_{idx}_active_expert_count_sum`
- `verify_layer_{idx}_active_routes_sum`
- `verify_layer_{idx}_moe_profile_count`

`scripts/bench_per_layer_slots.py`、`scripts/bench_verify_boundary_overhead.py` 和 `scripts/bench_segment_graph_no_prefetch.py` 会把这些 counter 汇总为：

- `summary.json` 中的 48 层数组：`verify_layer_realized_cpu_expert_count_per_call`、`verify_layer_cpu_routes_per_call` 等。
- `per_layer_cpu_experts.csv`：每个 case 每层一行，便于画图或和 `kt.cpuinfer_sync` 做相关性分析。
- `summary.md` 的 top layers 表：只展示 CPU experts/call 最高的层。

### Verify Boundary Rank Cache Optimization

新增 verify boundary ranking 优化：

- 默认启用 `NANOVLLM_VERIFY_PREFETCH_RANK_MULTIPLIER=1`，即每个 index 只向 submit 路径返回 `dispatch_budget` 个候选。
- `NANOVLLM_VERIFY_PREFETCH_RANK_MULTIPLIER=0` 可回退到全量 ranking。
- `SegmentCandidateIndex` 维护每个 segment 的排序缓存，并在 metadata observe/update 阶段重建，把排序成本从 verify boundary 主线程移到 metadata worker。
- 新增 profile 字段：`verify_segment_prefetch_rank_limit_sum`、`verify_segment_prefetch_rank_limited_count`、`run_verify_kt_hybrid_metadata_segment_index_rank_cache_rebuild_ms`。

短测命令：

```bash
CUDA_VISIBLE_DEVICES=2 conda run -n nano_moe python scripts/bench_verify_boundary_overhead.py \
  --output-dir results/verify_rank_rebuilt_event_k4_l32 \
  --gpu-memory-utilization 0.99 \
  --cache-ratios 0.3125 \
  --output-lens 32 \
  --modes verify_prefetch_on \
  --prefetch-on-max-draft-tokens-values 4 \
  --segment-sizes 12 \
  --allocation-mode profile_weighted \
  --slot-buckets 4 \
  --slot-max-bucket-ratio 2.0 \
  --slot-profile-csv pre_exps/exp_and_figs/unique/unique_count_plot_summary_n1024.csv \
  --kt-num-threads 16 \
  --verify-cuda-graph-bucket-steps 3,5
```

同配置短测结果：

| run | verify | segment event | gap | boundary | rank | rank scan | rank sort | transfer enqueue | metadata observe | candidates/submit |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full ranking | 715.440 | 0.000 | n/a | 5.147 | 2.293 | 1.989 | 0.298 | 1.641 | 1.841 | 305.3 |
| top-N only | 631.220 | 626.961 | 4.259 | 4.960 | 2.161 | 2.119 | 0.040 | 1.590 | 1.823 | 48.0 |
| sorted cache in submit | 845.770 | 841.458 | 4.312 | 4.520 | 1.839 | 1.810 | 0.026 | 1.452 | 1.813 | 48.0 |
| rebuild in metadata worker | 811.708 | 806.950 | 4.758 | 3.594 | 0.812 | 0.764 | 0.044 | 1.535 | 2.900 | 48.0 |
| rebuild + multiplier=1 | 722.313 | 717.969 | 4.344 | 3.142 | 0.374 | 0.350 | 0.021 | 1.542 | 2.620 | 12.0 |
| rebuild + multiplier=1 + budget=4 | 74.901 | 71.870 | 3.031 | 1.947 | 0.250 | 0.235 | 0.012 | 0.928 | 5.781 | 8.0 |

结论：

- 候选数量从约 305/submit 降到约 12/submit。
- `rank/call` 从 2.29 ms 降到 0.37 ms，verify boundary visible 从 5.15 ms 降到 3.14 ms。
- 进一步把 verify boundary budget 从 6 降到 4 后，submit/call 从 24 降到 16，H2D 从 226 MB/call 降到 151 MB/call，boundary visible 降到 1.95 ms，graph-external gap 降到 3.03 ms。
- 排序重建成本被转移到 metadata observe/worker，`metadata observe` 从约 1.8 ms 增到约 2.9 ms；这不是主线程 graph-external gap，但会占用 CPU。
- 当前短测已达到约 3 ms graph-external gap，但还需要更长 output_len 和 repeat 验证稳定性。剩余主线程可见开销主要是 `transfer_enqueue`、victim/filter/bookkeeping，以及非逐项归因 gap。

### Stability Validation

使用当前默认配置验证：

```text
verify_prefetch_max_per_boundary = 4
NANOVLLM_VERIFY_PREFETCH_RANK_MULTIPLIER = 1
NANOVLLM_VERIFY_SEGMENT_CUDA_EVENT_TIMING = 1
K = 4
segment_size = 12
cache_ratio = 0.3125
```

结果：

| run | repeats | verify mean | verify min/max | segment mean | gap mean | gap min/max | boundary mean | rank mean | transfer enqueue mean | CPU routes/call | CPU experts/call | hit | accept |
|:---|---:|---:|:---|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| output_len=128 | 3 | 66.218 | 63.185/69.846 | 63.081 | 3.137 | 3.015/3.239 | 2.058 | 0.288 | 0.980 | 439.5 | 302.3 | 0.755 | 0.882 |
| output_len=512 | 1 | 54.604 | 54.604/54.604 | 51.466 | 3.137 | 3.137/3.137 | 2.344 | 0.324 | 1.082 | 213.6 | 161.2 | 0.885 | 0.864 |

结论：

- `output_len=128` 的 3 次 repeat 均稳定低于 70 ms/verify，graph-external gap 最大 3.239 ms。
- `output_len=512` 单次验证更低，verify 54.604 ms，gap 3.137 ms。长输出后 cache 命中率升到 0.885，CPU routes/call 降到 213.6。
- 当前 `~75 ms verify` 和 `~3 ms graph-external gap` 目标在 K=4、budget=4 配置下已被 128 repeat=3 与 512 single-case 支持；仍建议再跑 output_len=512 repeat>=3 作为最终稳定性证明。

## Optimization Summary And Repro Script

完整的优化变更、回退开关、验证结果和目标测试脚本已整理到：

- `docs/optimize_ops/verify_optimization_summary.md`
- `scripts/bench_optimized_verify_perf.py`

默认目标测试命令：

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/bench_optimized_verify_perf.py \
  --output-dir results/optimized_verify_perf
```

该脚本复用 `scripts/bench_per_layer_slots.py` 的功能路径，并默认使用优化后的
`K=4`、`verify_prefetch_max_per_boundary=4`、
`NANOVLLM_VERIFY_PREFETCH_RANK_MULTIPLIER=1` 和 deferred verify metadata。
它会输出 `optimized_verify_summary.{json,md}`，并检查 verify 50-80 ms 与
decode 25-40 tok/s 目标。
