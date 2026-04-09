# Phase2 Post CPU 算子实现与精度修复交接文档（2026-04-07）

## 1. Goal And Scope

本阶段目标是把 MoE 的 CPU 专家执行路径从“可运行”推进到“可验证、可交付”的状态，重点包含两部分：

1. CPU 算子实现闭环：确保真实 CPU 专家路径被启用、可统计、可与 GPU 参考路径对照。
2. 精度修复与门禁：在相同输入、相同权重下量化 CPU/GPU 算子误差，建立阈值化验收机制，避免后续优化回归。

范围说明：

1. 本文聚焦 CPU 专家算子路径与算子级精度，不覆盖 speculative 接受率逻辑细节。
2. 保留并记录端到端高 CPU 比例下仍存在的已知极端差异（slots=4 单点 token 偏差），作为后续专项处理项。

## 2. Exact Code Paths Changed

本轮与 CPU 算子实现/精度修复直接相关的代码路径如下：

1. nanovllm/layers/fuse_moe/heterogeneous.py
	- 函数 `_run_real_cpu_expert_execution`：
	- CPU 计算 dtype 改为跟随 `hidden_states.dtype`（避免 CPU/GPU dtype 路径不一致）。
	- GPU->CPU 拷贝改为阻塞：`non_blocking=False`。
	- CPU->GPU 合并拷贝改为阻塞：`non_blocking=False`。
	- 目的：优先 correctness，消除异步拷贝潜在竞态导致的数值漂移。

2. examples/benchmarks/cpu_alignment_case.py
	- 新增标准/异构分进程 deterministic 对齐测试脚本。
	- 支持 `--remap-cache-high-ids` 强制触发 CPU 路由。
	- 通过 hook 记录真实 CPU 执行计数：`cpu_exec_calls`、`cpu_exec_routes`。

3. examples/benchmarks/cpu_gpu_expert_operator_alignment.py
	- 新增算子级 CPU/GPU 对齐基准。
	- 对比 `_run_real_cpu_expert_execution` 与 `_run_legacy_gpu_fallback` 在同输入同权重下的输出误差。
	- 内置阈值门禁：`max_rel_l2` 与 `max_mean_abs`。

4. tests/test_cpu_gpu_expert_operator_alignment.py
	- 新增单测，固定随机种子验证 CPU/GPU 专家算子误差阈值。
	- 阈值断言：`max_abs <= 1e-4`、`mean_abs <= 1e-6`、`rel_l2 <= 5e-4`。

## 3. Algorithm/Strategy Details And Constraints

### 3.1 CPU 算子执行策略

1. 路由组织方式：按 expert 分组执行（`cpu_task_expert_ids + cpu_task_offsets`），单 expert 内 batched matmul。
2. 算子序列：
	- `gate_up = F.linear(hidden_cpu, gate_up_weight)`
	- `cpu_out = F.linear(act_fn(gate_up), down_weight)`
	- `cpu_out *= routing_weights`
	- `output.index_add_` 回写到 GPU 输出缓冲。
3. 实现目标：保持与 GPU fallback 等价的数学表达式与路由权重应用顺序。

### 3.2 精度修复策略

1. dtype 对齐：CPU 计算路径改为使用 `hidden_states.dtype`，避免 bf16 主路径被 CPU float32 路径放大不一致。
2. 拷贝同步：
	- D2H/H2D 都改为 blocking copy。
	- 目的不是提速，而是先排除异步可见性/时序问题，确保结果稳定可复现。

### 3.3 约束与前提

1. 运行环境统一使用 `conda activate moe_spec`。
2. 算子级对齐基准要求 CUDA 可用（脚本与测试均显式检查）。
3. 端到端 deterministic 对齐结果受模型路由分布、CPU 比例和累积误差放大影响，不等价于单算子误差门禁。

## 4. Validation Commands And Acceptance Criteria

### 4.1 算子级精度门禁（必须通过）

命令：

1. `conda activate moe_spec && pytest -q tests/test_cpu_gpu_expert_operator_alignment.py`
2. `conda activate moe_spec && python examples/benchmarks/cpu_gpu_expert_operator_alignment.py --batch-sizes 64,128 --hidden-size 512 --intermediate-size 1024 --dtype bfloat16 --weight-scale 0.02 --max-rel-l2 5e-4 --max-mean-abs 1e-5 --seed 0 --output benchmarks/results/cpu_gpu_expert_operator_alignment_bf16.json`
3. `conda activate moe_spec && python examples/benchmarks/cpu_gpu_expert_operator_alignment.py --batch-sizes 64,128 --hidden-size 512 --intermediate-size 1024 --dtype float32 --weight-scale 0.02 --max-rel-l2 1e-5 --max-mean-abs 1e-6 --seed 0 --output benchmarks/results/cpu_gpu_expert_operator_alignment_fp32.json`

验收标准：

1. 单测必须通过。
2. 基准脚本 summary 必须 `passed=true`。
3. bf16 与 fp32 两档都需满足各自阈值。

### 4.2 端到端 CPU 比例 deterministic 对齐（阶段性验证）

命令参考：

1. `conda activate moe_spec && python examples/benchmarks/cpu_alignment_case.py ... --mode standard ...`
2. `conda activate moe_spec && python examples/benchmarks/cpu_alignment_case.py ... --mode heter --slots-per-layer {32,16,8,4} --cpu-expert-execution-enabled true --remap-cache-high-ids true ...`

验收标准（当前阶段）：

1. 需证明 CPU 路径真实执行：`cpu_exec_calls > 0` 且 `cpu_exec_routes > 0`。
2. 在中高比例（32/16/8）达到 exact match；最极端比例（4）作为已知限制持续跟踪。

## 5. Measured Results With Artifact Paths

### 5.1 算子级误差结果

1. 文件：benchmarks/results/cpu_gpu_expert_operator_alignment_bf16.json
	- worst_rel_l2: 4.1290651219266403e-04
	- worst_mean_abs: 7.319354011769974e-09
	- passed: true

2. 文件：benchmarks/results/cpu_gpu_expert_operator_alignment_fp32.json
	- worst_rel_l2: 5.47863968103648e-07
	- worst_mean_abs: 1.4558958205679318e-10
	- passed: true

3. 单测结果：
	- `tests/test_cpu_gpu_expert_operator_alignment.py` 通过（1 passed）

结论：

1. 在同输入同权重的专家算子对比中，CPU 路径误差受控并满足阈值门禁。
2. 算子级不存在需要立即修复的精度错误。

### 5.2 端到端 CPU 比例对齐结果

文件：benchmarks/results/cpu_alignment_text_summary_fix4.json

1. slots=32：exact_match=true，cpu_exec_calls=432，cpu_exec_routes=4747331
2. slots=16：exact_match=true，cpu_exec_calls=432，cpu_exec_routes=5614100
3. slots=8：exact_match=true，cpu_exec_calls=432，cpu_exec_routes=6087626
4. slots=4：exact_match=false，首个差异 `seq_idx=3, token_pos=5, standard=17, heter=19`，cpu_exec_calls=432，cpu_exec_routes=6259191

## 6. Known Limits And Explicit Next Steps

### 6.1 已知限制

1. 极端高 CPU 比例（slots=4）下，端到端仍有单点 token 分歧。
2. 该分歧不等价于算子实现错误：算子级门禁已通过，问题更可能来自多层累计误差放大、路径顺序与聚合细节耦合。

### 6.2 后续工程动作（供接手工程师继续）

1. 做首个分歧点逐层定位：
	- 在分歧步导出 layer-wise hidden states，比较 CPU path 与 legacy GPU fallback 的每层输出差异。
2. 强化端到端精度门禁：
	- 在 `cpu_alignment_case.py` 增加按层误差统计输出，形成自动归因报告。
3. 评估精度与性能折中策略：
	- 在高 CPU 比例下对比两种策略：
	- 策略 A：保持 blocking copy（当前 correctness-first 基线）。
	- 策略 B：引入显式同步点后恢复部分异步拷贝，验证能否在不破坏精度下恢复吞吐。
4. 纳入 CI 门禁建议：
	- 保留 `tests/test_cpu_gpu_expert_operator_alignment.py` 作为必跑。
	- 将 `examples/benchmarks/cpu_gpu_expert_operator_alignment.py` 纳入 nightly 精度巡检，阈值失败即阻断。

