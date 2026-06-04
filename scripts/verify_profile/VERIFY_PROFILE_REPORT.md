# Verify Profile Report

本文档记录 `logs/predictive_prefetch_matrix_direct_20260602_003130.log` 中
`legacy_ratio75_l512_k8` 的 verify 性能拆解过程、复现实验脚本、运行方法和结论。

## 背景

原始矩阵日志中最低的 `verify_ms` 仍然很高：

```text
legacy_ratio75_l512_k8:
  accept=0.9848 hit=0.9973 tok/s=26.541 draft_ms=21.124 verify_ms=143.035
```

该 case 配置为：

- runtime kind: `legacy`
- cache ratio: `0.75`
- output len: `512`
- max draft tokens: `8`
- prefetch runtime mode: `draft_segment_indexed`
- verify miss policy: `cache_fill`
- CPU expert backend: `fused`

## 脚本

本目录包含本次分析用到的可复跑脚本：

- `run_verify_profile_matrix.sh`: 入口脚本。进入指定 Slurm allocation，记录环境，运行 probe/direct/对照 case，生成结果目录。
- `run_direct_verify_case.py`: 不安装 benchmark monkeypatch 的直接路径 case runner，用于测当前 verify 路径。
- `summarize_verify_profile.py`: 汇总 result directory 下的 JSON，生成 case 对比和当前路径分解表。
- `parse_torch_trace.py`: 解析 `NANOVLLM_VERIFY_TORCH_PROFILE_DIR` 导出的 PyTorch Chrome trace。

## 运行方法

从 login 节点运行，复用已有 allocation，不会释放资源：

```bash
cd /home/mumura/moe_spec/nano-vllm-moe
bash scripts/verify_profile/run_verify_profile_matrix.sh 29309
```

脚本会在 compute node 中执行：

1. `source ~/.bashrc && conda activate nano_moe`
2. 打印 `hostname`、`CUDA_VISIBLE_DEVICES`、`nvidia-smi`、git sha/status。
3. 运行两个 benchmark probe case：
   - `probe_sync`: `spec_verify_expert_count_stats.py --sync-layer-timing true`
   - `probe_nosync`: `spec_verify_expert_count_stats.py --sync-layer-timing false`
4. 运行五个直接路径 case：
   - `direct_prefetch_vlayer_on`: 当前路径，prefetch on，verify-layer prefetch callback on，`cache_fill`
   - `direct_prefetch_vlayer_off`: 关闭 verify-layer callback
   - `direct_prefetch_off`: 关闭 prefetch，仍使用 `cache_fill`
   - `direct_cpu_policy_prefetch_off`: 关闭 prefetch，verify miss 直接 CPU 执行
   - `direct_torchprof_l512`: 当前路径并抓一次 verify forward torch profiler trace
5. 生成：
   - `summary_table.md`
   - `torch_trace_summary.md`

本次实际运行输出：

- log: `/home/mumura/moe_spec/logs/verify_breakdown_job29309_run_20260604_190717.log`
- result dir: `/home/mumura/moe_spec/nano-vllm-moe/results/verify_breakdown_job29309_20260604_190721`
- current path JSON: `direct_prefetch_vlayer_on.json`

## 当前路径直接记账

当前路径使用 `direct_prefetch_vlayer_on`，即不安装 benchmark probe monkeypatch，仅保留线上配置：
prefetch on、verify-layer callback on、`spec_verify_miss_policy=cache_fill`。

结果：

- verify calls: `58`
- verify tokens per call: `8.91`
- verify forward: `146.933 ms/call`
- throughput: `26.45 output tok/s`

| bucket | ms / verify call | percent |
|---|---:|---:|
| route | 7.20 | 4.9% |
| MoE plan | 32.65 | 22.2% |
| GPU gather | 2.82 | 1.9% |
| GPU expert compute | 26.55 | 18.1% |
| CPU expert compute | 0.00 | 0.0% |
| CPU merge | 0.00 | 0.0% |
| scatter | 4.77 | 3.2% |
| cache-fill transfer | 4.99 | 3.4% |
| forward residual | 67.95 | 46.2% |

这里的 `verify_ms` 来自 `SpeculativeEngine` 包住 `ModelRunner.run_verify(...)` 的前台耗时。
在当前 case 中，`model_run_verify_total_ms` 和 `spec_run_verify_infer_ms_total`
只差约 `0.22 ms/call`，所以瓶颈主要在 `run_verify` 前台路径内。

## 对照实验

| case | verify ms/call | calls | tok/call | throughput tok/s | plan | GPU expert | CPU expert | cache-fill transfer | residual | 说明 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `direct_prefetch_vlayer_on` | 146.9 | 58 | 8.91 | 26.45 | 32.7 | 26.6 | 0.0 | 5.0 | 68.0 | 当前路径 |
| `direct_prefetch_vlayer_off` | 139.5 | 60 | 8.92 | 26.15 | 32.9 | 26.4 | 0.0 | 5.1 | 60.3 | 关闭 verify-layer callback |
| `direct_prefetch_off` | 170.1 | 60 | 8.88 | 25.87 | 69.0 | 26.4 | 0.0 | 40.1 | 19.8 | 关闭 prefetch，仍 `cache_fill` |
| `direct_cpu_policy_prefetch_off` | 366.1 | 82 | 8.95 | 11.74 | 41.4 | 32.0 | 173.5 | 0.0 | 84.5 | miss 直接 CPU 执行 |

`probe_sync` 和 `probe_nosync` 结果：

- `probe_sync`: `151.830 ms/call`
- `probe_nosync`: `148.710 ms/call`

逐层 `torch.cuda.synchronize()` probe 只带来约 `3 ms/call` 差异，因此原始 `143 ms`
不是 validation 脚本同步插桩造成的假象。

## 代码路径解释

verify 模式 MoE 路径在 `nanovllm/models/qwen3_moe.py`：

1. gate + softmax + topk 计入 `route_ms`。
2. `cache_fill` 模式先统计 pre-transfer miss：
   `selected_experts.reshape(-1).detach().to(cpu)` + `torch.unique(...)`。
3. 调用 `apply_verify_cache_fill_policy(...)` 晋升 miss expert。
4. 调用 `build_verify_plan_gpu(...)`，它当前直接复用 `build_prefill_plan_gpu(...)`。
5. `heterogeneous_moe_forward(...)` 执行 GPU expert path、scatter，以及必要时 CPU expert path。

`build_prefill_plan_gpu(...)` 当前会执行：

- `expert_cache.remap_experts_to_slots(flat_selected)`
- `torch.nonzero(gpu_route_mask)`
- 按 route 和 slot 两次 `argsort`
- `scatter_add` 生成 `m_sizes`
- `torch.nonzero(cpu_route_mask)`
- CPU task layout 构造

在当前高命中 case 中，`cache_fill` 后 CPU work 为 0，但 plan 仍每层走通用规划流程，
因此 `MoE plan ~= 32.65 ms/call` 成为最大可归因单项。

## Torch profiler 说明

`direct_torchprof_l512` 会导出：

```text
direct_torchprof_l512_torch_profile/verify_forward_rank0.json
direct_torchprof_l512_torch_profile/verify_forward_rank0_summary.json
```

本环境中 `key_averages()` 的 CUDA self time 字段为 0，所以 `parse_torch_trace.py`
直接解析 Chrome trace 的 `traceEvents`。该 profiler case 因为首次 verify forward 触发 Dynamo/Inductor
编译和 trace 记录，整体 `verify_ms` 不能直接作为正常运行耗时，但可用于定位 kernel/runtime/memcpy 形态。

## 结论

1. 当前高命中 verify 路径的主要瓶颈不是 CPU fallback。`cache_fill` 后 `CPU expert compute=0`，
   `cache-fill transfer` 约 `5 ms/call`。
2. `MoE plan` 是最大可归因单项，约 `32.65 ms/call`，占 `22.2%`。
3. `GPU expert compute` 约 `26.55 ms/call`，占 `18.1%`，是实算成本。
4. `forward residual` 约 `67.95 ms/call`，占 `46.2%`，主要来自非 MoE 的 transformer 层、
   LM head、kernel launch、Python/调度和未细分同步。
5. verify-layer callback 前台成本约 `7.4 ms/call`，有优化空间。
6. 关闭 prefetch 会把 `verify_ms` 从 `146.9` 拉到 `170.1 ms/call`，
   主要因为 `cache-fill transfer` 从约 `5.0` 升到 `40.1 ms/call`，且 `plan` 明显变重。
7. miss 直接 CPU 执行不可接受：`verify_ms=366.1 ms/call`，其中 CPU expert compute 约
   `173.5 ms/call`。

## 优化优先级

1. **Verify CUDA Graph**
   - 覆盖主形状 `verify_len ~= K + 1 = 9`，以及尾部短 verify。
   - 目标是降低 `forward residual` 中的 launch/调度/固定 transformer 成本。

2. **Verify 全 GPU fast plan**
   - `cache_fill` 后如果没有 CPU route，走精简 plan。
   - 跳过 CPU task layout、host list 生成和不必要的 `nonzero(~gpu_mask)`。
   - 可参考 `build_cached_draft_plan_gpu(...)` 的固定 GPU route 方案。

3. **GPU 化 verify miss 统计**
   - 避免每层 `selected_experts.detach().to(cpu) + torch.unique`。
   - 用 GPU mask/scatter 统计 miss 和 active expert，减少 CPU 往返和潜在同步。

4. **降低 verify-layer prefetch callback 前台开销**
   - 减少每层锁、scan 和 publish ready。
   - 只在 frontier 非空时提交下一层 prefetch。
   - 考虑按多层批量发布 ready。

5. **保留 `cache_fill` 策略**
   - 对照实验显示 CPU miss policy 会显著退化，不应作为性能优化方向。
