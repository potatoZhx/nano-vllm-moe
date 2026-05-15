# nano-vllm-moe speculative sampling / prefetch session report

时间范围：2026-05-08 至 2026-05-09，Asia/Shanghai。  
工作目录：`/home/mumura/moe_spec/nano-vllm-moe`。

## 1. 实验目标

本 session 的目标来自以下连续需求：

1. 复核异常结果：`logs/spec_verify_expert_count_stats_20260508_213958/summary.md` 中 cache ratio 0.75/0.50/0.25 的 accept rate 都是 100%，output 甚至不到 1 tok/s，且 CPU experts=1 时 CPU compute 出现 13.562 ms，明显异常。
2. 后端采用优化后的 `FusedTorchCpuMoeBackend`，并保留与 torch backend 的对比。
3. `draft_top_c` 设置为 `0`，让 draft decode 能走 CUDA graph。
4. 不破坏已有 deterministic 接收算法，新增一个可选的标准投机采样接收算法，并设为默认。
5. 调整 prefetch，让 CPU->GPU transfer 尽量和计算重叠；发现 prefetch on 时 draft 反而变慢后，继续定位并修正主路径 publish 开销。
6. 重复 cache ratio 0.75/0.50/0.25、prefetch off/on、fused/torch backend 的统计实验，并输出 verify 阶段单层总 expert / CPU expert 的频率与耗时统计。

## 2. 主要实现

### 2.1 标准投机采样接收算法

修改文件：

- `nanovllm/engine/speculative/acceptance.py`
- `nanovllm/engine/speculative/spec_engine.py`
- `nanovllm/engine/model_runner.py`
- `nanovllm/config.py`
- `tests/test_acceptance.py`
- `tests/test_spec_engine_flow.py`

具体实现：

- 保留原有 `GreedyAcceptance` 和 `StandardAcceptance`。
- `StandardAcceptance` 的 deterministic 行为不改：
  - verify trace/list 输入时仍按 argmax trace 做精确前缀匹配。
  - logits 输入时仍按 target prob threshold 做 deterministic acceptance。
- 新增 `StandardSamplingAcceptance`：
  - 使用 target logits `p` 和 draft logits `q`。
  - 对每个 draft token 按 `min(1, p(x_i) / q(x_i))` 随机接收。
  - reject 时从 residual distribution `max(p - q, 0)` 采样 replacement token。
  - 所有 draft token accepted 时，从 target verify 的 next position 采样 next token。
  - `temperature <= 1e-10` 或没有 logits 时回退到 greedy deterministic 路径。
- `create_acceptance_strategy()` 增加别名：
  - `standard_sampling`
  - `sampling`
  - `spec_sampling`
- `Config.acceptance_strategy` 默认改为 `standard_sampling`。
- `SpeculativeEngine` 在 sampling 模式下不再回退普通 decode，而是：
  - draft 阶段请求并保存 draft logits。
  - verify 阶段请求 target logits。
  - 将 draft tokens、target logits、draft logits 交给 `StandardSamplingAcceptance`。
- `ModelRunner.run(..., return_logits=True)` 支持返回 `(token_ids, logits)`。
- `ModelRunner.run_draft(..., return_logits=True)` 支持返回 `(token_ids, {"prefetch_step_id": step_id}, draft_logits)`。
- `ModelRunner.run_verify(..., return_logits=True)` 支持按 sequence 返回 verify logits slices。

### 2.2 draft CUDA graph

相关文件：

- `benchmarks/scripts/spec_verify_expert_count_stats.py`
- `nanovllm/engine/model_runner.py`

具体实现/实验设置：

- 实验默认 `--draft-top-c 0`。
- `ModelRunner._can_use_draft_cudagraph()` 要求 `draft_top_c == 0` 才启用 draft graph replay。
- 结果中最终完整矩阵所有 case 都有 draft graph replay，例如 fused 0.75 off 为 27 次、fused 0.25 off 为 38 次。

### 2.3 CPU expert backend 和 benchmark 脚本

相关文件：

- `nanovllm/layers/fuse_moe/cpu_backend.py`
- `tests/test_cpu_moe_correctness.py`
- `benchmarks/scripts/spec_verify_expert_count_stats.py`

具体实现/设置：

- `FusedTorchCpuMoeBackend` 路径使用预分配 workspace，减少临时 tensor 分配。
- benchmark 支持 `--cpu-expert-backends fused,torch`。
- 本 session 后续默认重点看 fused；torch 对比仍在最终完整矩阵中保留，因为脚本已启动且用户允许正常跑完。

### 2.4 prefetch 与 transfer overlap

相关文件：

- `nanovllm/engine/model_runner.py`
- `nanovllm/expert/prefetcher.py`
- `benchmarks/scripts/spec_verify_expert_count_stats.py`
- `tests/test_model_runner_prefetch.py`

具体实现：

- benchmark 默认 `--cpu-expert-pin-memory true`，因为非 pinned CPU tensor 无法可靠实现异步 H2D overlap。
- `wait_prefetch_for_verify()` 不再强制 drain metadata worker，而是 non-blocking flush，仅消费已经 ready 的 prefetch：
  - `metadata_drain_ms = 0.0`
  - `_flush_pending_prefetch_metadata(block=False)`
  - `prefetch_runtime.wait_for_verify(... timeout_ms=prefetch_verify_wait_ms)`
- 初版修正后发现 prefetch on 的 draft avg 仍高，定位到 `run_draft()` 每个 draft step 前调用 `publish_ready()`：
  - staging->active GPU copy + `current_stream.wait_stream(publish_stream)` 进入 draft 主路径。
- 最终修正：去掉 draft 前 `publish_ready()`，保留 verify 前 publish。
- verify 安全性：
  - `SpeculativeEngine` 在 `run_verify()` 前调用 `wait_prefetch_for_verify()`。
  - `PrefetchRuntime.wait_for_verify()` 第一件事仍是 `publish_ready()`。
  - `publish_ready()` 中 `_finalize_publish()` 会执行 `torch.cuda.current_stream().wait_stream(self.publish_stream)` 后才 `commit_published_expert()` 更新 active LUT。
  - 因此 verify 不会看到“LUT 已更新但 active weight copy 未完成”的状态；未完成的 prefetch 只会继续走 CPU/uncached 路径。

## 3. 遇到的问题和解决方案

### 3.1 accept rate 100% 异常

现象：

- 旧结果 `/home/mumura/moe_spec/logs/spec_verify_expert_count_stats_20260508_213958/summary.md` 中所有 ratio accept rate 都为 1.0。
- 同时 output/e2e tok/s 很低，verify avg ms 高达数秒。

处理：

- 接入真正的 `standard_sampling`，不再用 deterministic trace/threshold 代替 stochastic speculative sampling。
- 最终结果中 accept rate 不再是 100%，例如 final fused：
  - ratio 0.75 off/on：0.5926 / 0.5357
  - ratio 0.50 off/on：0.4839 / 0.6400
  - ratio 0.25 off/on：0.3421 / 0.3333

### 3.2 draft top_c 导致 CUDA graph 不生效

现象：

- 用户指出 draft `top_c` 需要为 0，否则 draft CUDA graph 加速无法开启。

处理：

- 所有后续实验显式设置 `--draft-top-c 0`。
- final summary 中 draft graph replays 均为非零。

### 3.3 standard_sampling 首次 smoke 缺少 draft logits

现象：

- 两次 smoke 在运行到最后一个 speculative step 时失败：
  - `/home/mumura/moe_spec/logs/spec_sampling_overlap_smoke_20260508_232640.log`
  - `/home/mumura/moe_spec/logs/spec_sampling_overlap_smoke_20260508_233045.log`
- 报错：
  - `ValueError: standard_sampling speculative acceptance requires draft logits`

根因：

- 当剩余生成预算只够 verify-next、`draft_tokens=[]` 时，接收算法不应该要求 draft logits。

解决：

- `StandardSamplingAcceptance.accept()` 增加 no-draft 分支：
  - 直接从 target verify logits 的第 0 行采样 next token。
  - 返回 `num_accepted=0`、`rejected=False`。
- 增加测试 `test_standard_sampling_samples_verify_token_without_drafts`。

### 3.4 prefetch on 使 draft 变慢

现象：

- 修正 sampling 后的完整矩阵 `/home/mumura/moe_spec/logs/spec_sampling_overlap_full_20260508_233603/summary.md` 中 prefetch on 的 draft avg 明显高于 off：
  - fused 0.75：18.289 ms -> 21.375 ms
  - fused 0.50：18.747 ms -> 21.702 ms
  - fused 0.25：18.624 ms -> 22.238 ms

根因：

- `run_draft()` 在 draft 前执行 `publish_ready()`，每个 draft step 会把 ready staging slot publish 到 active cache。
- 这部分包含 staging->active copy 和 stream wait，直接进入 draft 主路径。

解决：

- 去掉 draft 前 `publish_ready()`。
- 保留 verify 前 `publish_ready()`，保证 verify 可安全消费 ready prefetch。
- publish-fix A/B 结果：
  - 路径：`/home/mumura/moe_spec/logs/spec_sampling_overlap_publishfix_ab_20260509_000219/summary.md`
  - fused 0.75 on：
    - `model_run_draft_prefetch_before_ms` 从修正前 47.970 ms 降到 2.719 ms。
    - `model_publish_count` 从 120 降到 28。
    - `model_publish_ms` 从 56.524 ms 降到 14.453 ms。
    - draft avg 从 21.375 ms 降到 18.972 ms，接近 off 的 17.947 ms。

## 4. 测试脚本、命令和结果路径

### 4.1 本地单元测试和编译检查

脚本/文件：

- `tests/test_acceptance.py`
- `tests/test_spec_engine_basic.py`
- `tests/test_spec_engine_flow.py`
- `tests/test_spec_engine_prefetch.py`
- `tests/test_model_runner_spec_modes.py`
- `tests/test_model_runner_prefetch.py`
- `nanovllm/engine/speculative/acceptance.py`
- `nanovllm/engine/speculative/spec_engine.py`
- `nanovllm/engine/model_runner.py`
- `benchmarks/scripts/spec_verify_expert_count_stats.py`

命令：

```bash
python -m pytest tests/test_acceptance.py tests/test_spec_engine_basic.py tests/test_spec_engine_flow.py tests/test_spec_engine_prefetch.py tests/test_model_runner_spec_modes.py tests/test_model_runner_prefetch.py -q
python -m py_compile nanovllm/engine/speculative/acceptance.py nanovllm/engine/speculative/spec_engine.py nanovllm/engine/model_runner.py benchmarks/scripts/spec_verify_expert_count_stats.py
```

结果：

- `22 passed in 10.45s`
- `py_compile` 无错误。

结果文件：

- `/home/mumura/moe_spec/logs/session_validation_20260509_003802.log`

### 4.2 初始异常结果复核

脚本：

- `benchmarks/scripts/spec_verify_expert_count_stats.py`

原实验命令来自 summary metadata：

```bash
python benchmarks/scripts/spec_verify_expert_count_stats.py \
  --model-path /data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B \
  --output-dir /home/mumura/moe_spec/logs/spec_verify_expert_count_stats_20260508_213958 \
  --cache-ratios 0.75,0.50,0.25 \
  --prefetch-order off,on \
  --num-seqs 1 --input-len 12 --output-len 24 \
  --max-draft-tokens 4 --draft-top-c 128 \
  --temperature 0.0 --acceptance-strategy standard \
  --cpu-expert-backend torch \
  --cpu-expert-packed-min-routes 1 \
  --cpu-gpu-parallel-execution-enabled auto \
  --prefetch-verify-wait-ms 1.0 \
  --prefetch-step-budget 4 \
  --prefetch-max-inflight 8 \
  --prefetch-staging-slots-per-layer 2 \
  --dist-port-base 27500 \
  --case-timeout-sec 1800
```

结果文件：

- `/home/mumura/moe_spec/logs/spec_verify_expert_count_stats_20260508_213958/summary.md`
- `/home/mumura/moe_spec/logs/spec_verify_expert_count_stats_20260508_213958/summary.json`

重要结果：

- accept rate 全部为 1.0。
- e2e tok/s 约 0.802、0.814、0.418、0.406、0.306、0.307。
- verify avg ms 约 2400 至 7487 ms。
- 这是后续修正的基线异常。

### 4.3 deterministic + top_c=0 + fused/torch 对比

脚本：

- `benchmarks/scripts/spec_verify_expert_count_stats.py`

命令来自 summary metadata：

```bash
python benchmarks/scripts/spec_verify_expert_count_stats.py \
  --model-path /data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B \
  --output-dir /home/mumura/moe_spec/logs/spec_verify_expert_count_stats_topc0_fused_torch_20260508_222733 \
  --cache-ratios 0.75,0.50,0.25 \
  --prefetch-order off,on \
  --cpu-expert-backends fused,torch \
  --num-seqs 1 --input-len 12 --output-len 24 \
  --max-draft-tokens 4 --draft-top-c 0 \
  --temperature 0.0 --acceptance-strategy standard \
  --acceptance-threshold 0.7 \
  --cpu-expert-packed-min-routes 1 \
  --cpu-expert-parallel-mode serial \
  --cpu-expert-num-threads 4 \
  --cpu-gpu-parallel-execution-enabled auto \
  --max-num-batched-tokens 512 --max-num-seqs 1 --max-model-len 512 \
  --gpu-memory-utilization 0.85 --enforce-eager false \
  --prefetch-verify-wait-ms 1.0 \
  --prefetch-step-budget 4 \
  --prefetch-max-inflight 8 \
  --prefetch-staging-slots-per-layer 2 \
  --seed 0 --sync-layer-timing true \
  --dist-port-base 28700 \
  --case-timeout-sec 1800
```

结果文件：

- `/home/mumura/moe_spec/logs/spec_verify_expert_count_stats_topc0_fused_torch_20260508_222733/summary.md`
- `/home/mumura/moe_spec/logs/spec_verify_expert_count_stats_topc0_fused_torch_20260508_222733/summary.json`

重要结果：

- draft graph replay 生效。
- accept rate 已不再全为 1.0，但这仍是 deterministic `standard`，不是标准 speculative sampling。
- fused 在大多数 case 上优于 torch。

### 4.4 failed smoke：错误参数

脚本：

- `benchmarks/scripts/spec_verify_expert_count_stats.py`

命令摘要：

```bash
srun --jobid=21487 ... python benchmarks/scripts/spec_verify_expert_count_stats.py \
  --model-path /data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B \
  --output-dir /home/mumura/moe_spec/logs/spec_sampling_overlap_smoke_20260508_232623 \
  --cache-ratios 0.75 \
  --prefetch-order on \
  --cpu-expert-backends fused \
  --output-len 8 \
  --draft-top-c 0 \
  --temperature 0.8 \
  --acceptance-strategy standard_sampling \
  --cpu-expert-pin-memory true
```

结果文件：

- `/home/mumura/moe_spec/logs/spec_sampling_overlap_smoke_20260508_232623.log`

重要结果：

- 参数错误：suite 模式 `--prefetch-order` 只接受 `off,on` 或 `on,off`。
- 之后改用 `--single-case --prefetch-enabled true`。

### 4.5 failed smoke：no-draft 边界条件缺少 draft logits

脚本：

- `benchmarks/scripts/spec_verify_expert_count_stats.py`

命令摘要：

```bash
srun --jobid=21487 ... python benchmarks/scripts/spec_verify_expert_count_stats.py \
  --single-case \
  --model-path /data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B \
  --output /home/mumura/moe_spec/logs/spec_sampling_overlap_smoke_20260508_232640/case.json \
  --cache-ratio 0.75 \
  --prefetch-enabled true \
  --cpu-expert-backend fused \
  --output-len 8 \
  --max-draft-tokens 4 \
  --draft-top-c 0 \
  --temperature 0.8 \
  --acceptance-strategy standard_sampling \
  --cpu-expert-pin-memory true \
  --prefetch-verify-wait-ms 0.0 \
  --prefetch-step-budget 8 \
  --prefetch-max-inflight 16 \
  --prefetch-staging-slots-per-layer 4
```

结果文件：

- `/home/mumura/moe_spec/logs/spec_sampling_overlap_smoke_20260508_232640.log`
- `/home/mumura/moe_spec/logs/spec_sampling_overlap_smoke_20260508_233045.log`

重要结果：

- 两次均触发：
  - `ValueError: standard_sampling speculative acceptance requires draft logits`
- 修复 no-draft 分支后通过。

### 4.6 successful smoke：standard_sampling + pinned prefetch + fused

脚本：

- `benchmarks/scripts/spec_verify_expert_count_stats.py`

命令：

```bash
srun --jobid=21487 -N1 -n1 --cpus-per-task=16 bash -lc '
source ~/.bashrc
conda activate nano_moe
cd /home/mumura/moe_spec/nano-vllm-moe
export CUDA_VISIBLE_DEVICES=5
python benchmarks/scripts/spec_verify_expert_count_stats.py \
  --single-case \
  --model-path /data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B \
  --output /home/mumura/moe_spec/logs/spec_sampling_overlap_smoke_20260508_233352/case.json \
  --cache-ratio 0.75 \
  --prefetch-enabled true \
  --cpu-expert-backend fused \
  --num-seqs 1 --input-len 12 --output-len 8 \
  --max-draft-tokens 4 --draft-top-c 0 \
  --temperature 0.8 \
  --acceptance-strategy standard_sampling \
  --cpu-expert-pin-memory true \
  --cpu-expert-packed-min-routes 1 \
  --cpu-expert-parallel-mode serial \
  --cpu-expert-num-threads 4 \
  --cpu-gpu-parallel-execution-enabled auto \
  --max-num-batched-tokens 512 --max-num-seqs 1 --max-model-len 512 \
  --gpu-memory-utilization 0.85 --enforce-eager false \
  --prefetch-verify-wait-ms 0.0 \
  --prefetch-step-budget 8 \
  --prefetch-max-inflight 16 \
  --prefetch-staging-slots-per-layer 4 \
  --seed 0 --sync-layer-timing true \
  --dist-port 29102 \
  --case-timeout-sec 900'
```

结果文件：

- `/home/mumura/moe_spec/logs/spec_sampling_overlap_smoke_20260508_233352.log`
- `/home/mumura/moe_spec/logs/spec_sampling_overlap_smoke_20260508_233352/case.json`

重要结果：

- generated output tokens：8。
- decode phase tok/s：3.742。
- draft graph replay：13。
- prefetch submit/completed/consumed：34/36/19。
- accept rate：0.1538。
- rejection rate per step：0.8。

### 4.7 standard_sampling 完整矩阵，publish 修正前

脚本：

- `benchmarks/scripts/spec_verify_expert_count_stats.py`

命令：

```bash
srun --jobid=21487 -N1 -n1 --cpus-per-task=16 bash -lc '
source ~/.bashrc
conda activate nano_moe
cd /home/mumura/moe_spec/nano-vllm-moe
export CUDA_VISIBLE_DEVICES=5
python benchmarks/scripts/spec_verify_expert_count_stats.py \
  --model-path /data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B \
  --output-dir /home/mumura/moe_spec/logs/spec_sampling_overlap_full_20260508_233603 \
  --cache-ratios 0.75,0.50,0.25 \
  --prefetch-order off,on \
  --cpu-expert-backends fused,torch \
  --num-seqs 1 --input-len 12 --output-len 24 \
  --max-draft-tokens 4 --draft-top-c 0 \
  --temperature 0.8 \
  --acceptance-strategy standard_sampling \
  --cpu-expert-pin-memory true \
  --cpu-expert-packed-min-routes 1 \
  --cpu-expert-parallel-mode serial \
  --cpu-expert-num-threads 4 \
  --cpu-gpu-parallel-execution-enabled auto \
  --max-num-batched-tokens 512 --max-num-seqs 1 --max-model-len 512 \
  --gpu-memory-utilization 0.85 --enforce-eager false \
  --prefetch-verify-wait-ms 0.0 \
  --prefetch-step-budget 8 \
  --prefetch-max-inflight 16 \
  --prefetch-staging-slots-per-layer 4 \
  --cache-eviction-budget-per-step 4 \
  --seed 0 --sync-layer-timing true \
  --dist-port-base 29200 \
  --case-timeout-sec 900'
```

结果文件：

- `/home/mumura/moe_spec/logs/spec_sampling_overlap_full_20260508_233603.log`
- `/home/mumura/moe_spec/logs/spec_sampling_overlap_full_20260508_233603/summary.md`
- `/home/mumura/moe_spec/logs/spec_sampling_overlap_full_20260508_233603/summary.json`
- 单 case JSON/log 在同目录下，例如 `fused_ratio75_prefetch_on.json`。

重要结果：

- accept rate 不再是 100%。
- draft graph replay 生效。
- 但 prefetch on draft avg 仍明显上升：
  - fused 0.75：18.289 -> 21.375 ms
  - fused 0.50：18.747 -> 21.702 ms
  - fused 0.25：18.624 -> 22.238 ms
- 定位到 draft 前 publish 开销：
  - fused 0.75 on：`model_publish_count=120`，`model_publish_ms=56.524`，`model_run_draft_prefetch_before_ms=47.970`。

### 4.8 publish-fix A/B

脚本：

- `benchmarks/scripts/spec_verify_expert_count_stats.py`

命令：

```bash
srun --jobid=21487 -N1 -n1 --cpus-per-task=16 bash -lc '
source ~/.bashrc
conda activate nano_moe
cd /home/mumura/moe_spec/nano-vllm-moe
export CUDA_VISIBLE_DEVICES=5
python benchmarks/scripts/spec_verify_expert_count_stats.py \
  --model-path /data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B \
  --output-dir /home/mumura/moe_spec/logs/spec_sampling_overlap_publishfix_ab_20260509_000219 \
  --cache-ratios 0.75 \
  --prefetch-order off,on \
  --cpu-expert-backends fused \
  --num-seqs 1 --input-len 12 --output-len 24 \
  --max-draft-tokens 4 --draft-top-c 0 \
  --temperature 0.8 \
  --acceptance-strategy standard_sampling \
  --cpu-expert-pin-memory true \
  --cpu-expert-packed-min-routes 1 \
  --cpu-expert-parallel-mode serial \
  --cpu-expert-num-threads 4 \
  --cpu-gpu-parallel-execution-enabled auto \
  --max-num-batched-tokens 512 --max-num-seqs 1 --max-model-len 512 \
  --gpu-memory-utilization 0.85 --enforce-eager false \
  --prefetch-verify-wait-ms 0.0 \
  --prefetch-step-budget 8 \
  --prefetch-max-inflight 16 \
  --prefetch-staging-slots-per-layer 4 \
  --cache-eviction-budget-per-step 4 \
  --seed 0 --sync-layer-timing true \
  --dist-port-base 29400 \
  --case-timeout-sec 900'
```

结果文件：

- `/home/mumura/moe_spec/logs/spec_sampling_overlap_publishfix_ab_20260509_000219.log`
- `/home/mumura/moe_spec/logs/spec_sampling_overlap_publishfix_ab_20260509_000219/summary.md`
- `/home/mumura/moe_spec/logs/spec_sampling_overlap_publishfix_ab_20260509_000219/summary.json`

重要结果：

- fused 0.75 off/on：
  - off decode tok/s 9.288，draft avg 17.947 ms。
  - on decode tok/s 7.465，draft avg 18.972 ms。
- publish 主路径开销明显下降：
  - `model_run_draft_prefetch_before_ms=2.719`
  - `model_publish_count=28`
  - `model_publish_ms=14.453`
- 说明 draft 前 publish 被移出后，draft 计算基本恢复。

### 4.9 最终完整矩阵

脚本：

- `benchmarks/scripts/spec_verify_expert_count_stats.py`

命令：

```bash
srun --jobid=21503 -N1 -n1 --cpus-per-task=16 bash -lc '
source ~/.bashrc
conda activate nano_moe
cd /home/mumura/moe_spec/nano-vllm-moe
export CUDA_VISIBLE_DEVICES=4
python benchmarks/scripts/spec_verify_expert_count_stats.py \
  --model-path /data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B \
  --output-dir /home/mumura/moe_spec/logs/spec_sampling_overlap_publishfix_full_20260509_001330 \
  --cache-ratios 0.75,0.50,0.25 \
  --prefetch-order off,on \
  --cpu-expert-backends fused,torch \
  --num-seqs 1 --input-len 12 --output-len 24 \
  --max-draft-tokens 4 --draft-top-c 0 \
  --temperature 0.8 \
  --acceptance-strategy standard_sampling \
  --cpu-expert-pin-memory true \
  --cpu-expert-packed-min-routes 1 \
  --cpu-expert-parallel-mode serial \
  --cpu-expert-num-threads 4 \
  --cpu-gpu-parallel-execution-enabled auto \
  --max-num-batched-tokens 512 --max-num-seqs 1 --max-model-len 512 \
  --gpu-memory-utilization 0.85 --enforce-eager false \
  --prefetch-verify-wait-ms 0.0 \
  --prefetch-step-budget 8 \
  --prefetch-max-inflight 16 \
  --prefetch-staging-slots-per-layer 4 \
  --cache-eviction-budget-per-step 4 \
  --seed 0 --sync-layer-timing true \
  --dist-port-base 29500 \
  --case-timeout-sec 900'
```

结果文件：

- `/home/mumura/moe_spec/logs/spec_sampling_overlap_publishfix_full_20260509_001330.log`
- `/home/mumura/moe_spec/logs/spec_sampling_overlap_publishfix_full_20260509_001330/summary.md`
- `/home/mumura/moe_spec/logs/spec_sampling_overlap_publishfix_full_20260509_001330/summary.json`
- 单 case JSON/log 在同目录下，例如：
  - `/home/mumura/moe_spec/logs/spec_sampling_overlap_publishfix_full_20260509_001330/fused_ratio75_prefetch_off.json`
  - `/home/mumura/moe_spec/logs/spec_sampling_overlap_publishfix_full_20260509_001330/fused_ratio75_prefetch_on.json`

最终 case summary：

| backend | ratio | prefetch | accept rate | rejects | drafted/accepted | draft replays | verify avg ms | draft avg ms | prefetch consumed | decode tok/s | e2e tok/s |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fused | 0.75 | off | 0.5926 | 3 | 27/16 | 27 | 300.282 | 18.191 | 0 | 9.225 | 6.547 |
| fused | 0.75 | on | 0.5357 | 4 | 28/15 | 28 | 340.836 | 18.908 | 19 | 7.263 | 5.445 |
| fused | 0.50 | off | 0.4839 | 6 | 31/15 | 31 | 415.675 | 18.944 | 0 | 6.118 | 4.086 |
| fused | 0.50 | on | 0.6400 | 5 | 25/16 | 25 | 546.491 | 19.874 | 23 | 5.501 | 3.795 |
| fused | 0.25 | off | 0.3421 | 8 | 38/13 | 38 | 540.646 | 18.991 | 0 | 3.908 | 2.709 |
| fused | 0.25 | on | 0.3333 | 9 | 36/12 | 36 | 579.671 | 20.266 | 44 | 3.349 | 2.399 |
| torch | 0.75 | off | 0.3714 | 7 | 35/13 | 35 | 371.300 | 18.562 | 0 | 5.484 | 3.931 |
| torch | 0.75 | on | 0.5926 | 3 | 27/16 | 27 | 874.082 | 20.897 | 31 | 3.570 | 2.937 |
| torch | 0.50 | off | 0.3611 | 9 | 36/13 | 36 | 602.662 | 19.100 | 0 | 3.568 | 2.719 |
| torch | 0.50 | on | 0.5714 | 4 | 28/16 | 28 | 925.838 | 20.093 | 30 | 3.390 | 2.591 |
| torch | 0.25 | off | 0.1296 | 14 | 54/7 | 54 | 593.634 | 18.961 | 0 | 2.277 | 1.740 |
| torch | 0.25 | on | 0.2927 | 9 | 41/12 | 41 | 708.536 | 20.115 | 47 | 2.762 | 1.652 |

重要结论：

- true sampling 接收已生效，accept rate 不再固定 100%。
- `draft_top_c=0` 下 draft graph replay 生效。
- fused backend 明显优于 torch backend：
  - fused off decode tok/s：9.225 / 6.118 / 3.908。
  - torch off decode tok/s：5.484 / 3.568 / 2.277。
- final fused CPU experts=1 的 CPU compute 不再出现 13 ms 量级：
  - ratio 0.75 off：avg 0.512 ms，min 0.333 ms，max 0.701 ms，freq 3。
  - ratio 0.75 on：avg 0.356 ms，min 0.307 ms，max 0.717 ms，freq 19。
- prefetch on 仍未稳定提升 fused decode tok/s：
  - 0.75：9.225 -> 7.263
  - 0.50：6.118 -> 5.501
  - 0.25：3.908 -> 3.349
- 但 publish-fix 后 draft prefetch-before 开销已显著下降：
  - 修正前 fused on publish counts：120 / 160 / 180。
  - 修正后 fused on publish counts：28 / 28 / 40。
  - 修正前 fused on draft avg：21.375 / 21.702 / 22.238 ms。
  - 修正后 fused on draft avg：18.908 / 19.874 / 20.266 ms。

## 5. 当前结论和后续建议

1. 原始 100% accept rate 异常已解决。最终 `standard_sampling` 下 accept rate 随 ratio、采样路径变化，不再固定为 1.0。
2. deterministic 算法保留为可选项，没有被覆盖。
3. `standard_sampling` 已设为默认 acceptance strategy。
4. `draft_top_c=0` 能正常启用 draft CUDA graph。
5. `FusedTorchCpuMoeBackend` 是后续默认 backend 的合理选择；最终完整矩阵仍保留 torch 对比，结果支持 fused 更优。
6. CPU experts=1 的异常 13 ms 在 final fused 结果中没有复现，最终为 0.3-0.7 ms 量级。
7. prefetch 的 transfer/metadata 已尽量从 draft 主路径移开，但 final prefetch on 对 fused decode tok/s 仍未带来稳定提升。当前剩余问题更可能是：
   - verify 前 publish 仍会引入同步开销；
   - prefetch 命中量不足，final fused consumed 仅 19/23/44；
   - prefetch 改变 cache residency，可能使 verify 的 CPU expert 分布与 off case 不完全可比；
   - 单 prompt / 24 output tokens 的样本量偏小，sampling 随机性会影响 verify calls 和 accept count。

建议后续如果继续优化 prefetch：

- 将 active cache publish 做成更细粒度的预算控制或只 publish 当前 verify 高概率会用到的 experts。
- 在 summary 中额外记录 publish 前后 active cache 命中率、staging ready 未发布数量、verify 实际命中 published expert 的比例。
- 固定采样 token trace 或增加多 seed 重复，减少 `standard_sampling` 随机性对 off/on 的可比性影响。
- 后续默认只跑 fused backend，除非需要做回归对照。
