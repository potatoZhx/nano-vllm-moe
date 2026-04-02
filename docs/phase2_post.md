# Phase 2 后续详细设计：真实 CPU Expert 执行、S<N 验证、Draft 纯 GPU 与自动 CUDA Graph

## 1. 背景与目标

本文档用于替代旧版后续草案 [docs/phase2_post_spec_cpu_parallel_gpu_schedule_design.md](./phase2_post_spec_cpu_parallel_gpu_schedule_design.md)，作为 Phase 2 之后的主设计与执行计划。旧文档保留作为历史参考，但后续实现以本文档为准。

本文档的目标不是重复“Phase 2 已经跑通”，而是明确下一阶段真正要完成的事情：

1. 把当前 uncached fallback 改造成真实 CPU expert 执行路径。
2. 让 CPU experts 与 GPU cached experts 在同层真正并行。
3. 在真实 CPU 路径落地后，把主性能验证从 `S=N` 转到 `S<N`。
4. 把 draft / verify 热路径中的 Python object crossing 清理到足够低，为纯 GPU plan 与 CUDA Graph 铺路。
5. 在 `enforce_eager=False` 且 `draft_top_c=0` 时，Draft CUDA Graph 默认自动启用，而不是只作为手动实验功能。

本文档风格参考 [docs/phase2_design.md](./phase2_design.md) 与 [docs/migration_design.md](./migration_design.md)：除了阶段目标外，还给出需要修改的文件、核心接口变化与伪代码。

---

## 2. 当前实现状态与关键事实

## 2.1 来自 Phase 1 的事实

根据 [docs/summary/phase1_cpu_gpu_basic_heterogeneous_report.md](./summary/phase1_cpu_gpu_basic_heterogeneous_report.md) 与 [docs/steps/phase1_steps.md](./steps/phase1_steps.md)：

1. `S=N` 已经建立为有效、公平的基线验证口径。
2. 在 `S=N` 场景下，heter 路径吞吐约为 standard 的 `0.82x`，代表性结果约 `-17.5%`。
3. `S=N` 下 CPU fallback 基本不触发，不是主瓶颈。
4. 当前主要开销集中在：
   - `plan`
   - gather / reorder
   - fragmented routing 下 fused kernel 利用率
   - `scatter/index_add`

因此，`S=N` 的定位是“验证完整 heter/spec 主链路自身的开销”，而不是未来真实部署目标。

## 2.2 来自 Phase 2 的事实

根据 [docs/summary/phase2_speculative_decoding_report.md](./summary/phase2_speculative_decoding_report.md) 与当前代码实现：

1. speculative baseline 已跑通，`Draft -> Rollback -> Verify -> Accept` 主链路可运行。
2. deterministic 条件下已具备 standard / heter / spec 的对齐基础。
3. draft / verify planning 已部分张量化，但仍有 Python `list` / `dict` / `.tolist()` 等对象热路径。
4. 当前 decode CUDA Graph 已在 standard decode 路径存在，但 draft-mode 没有独立 graph 语义。

## 2.3 当前代码的真实状态

当前代码中必须明确区分“CPU pool”与“真实 CPU 执行”：

1. 目前 uncached expert 权重存放在 `cpu_expert_pool`。
2. 但在执行时，这些权重会被搬到 GPU，再在 GPU 上做 `F.linear`。
3. 因此，当前 fallback 的本质是：
   - host 驻留权重
   - Python 控制分组
   - host-to-device 权重搬运
   - 最终 GPU 计算
4. 这不是“真实 CPU expert 执行”，也不是未来 `S<N` 的目标性能模型。

这个事实决定了两点：

1. 在真实 CPU 执行落地前，主验收仍必须以 `S=N` 为主。
2. `S<N` 当前只能做 exploratory measurement，不能作为本阶段主结论。

---

## 3. 为什么当前主验收仍然是 S=N

## 3.1 `S=N` 的角色

`S=N` 表示每层 GPU slot 覆盖全部 expert。其作用是：

1. 让 heter/spec 走完整的 plan 与 heterogeneous forward 主链路。
2. 禁止通过 `if S == N` 回退到 standard。
3. 在不引入未完成 CPU 执行模型的情况下，公平测量异构路径自身的控制与执行开销。

## 3.2 为什么现在不能把 `S<N` 作为主结论口径

真实 CPU expert 执行尚未实现前，`S<N` 下的 uncached 路径仍然混入了：

1. Python expert 分组与控制成本。
2. host-to-device 权重搬运成本。
3. GPU 侧执行而非 CPU 侧执行。

因此，当前 `S<N` 的数值不能直接解释为未来真实 CPU/GPU 混合执行的性能结论。

## 3.3 阶段切换规则

本文档明确采用以下切换规则：

1. 在真实 CPU expert execution 完成并通过 correctness 验证前，主验收仍为 `S=N`。
2. 当真实 CPU expert execution 与 CPU/GPU same-layer overlap 都完成后，主性能实验转入 `S<N`。
3. `S=N` 在后续仍保留，用于持续验证 heter/spec 主链路开销没有被新优化破坏。

---

## 4. 下一阶段的真实目标：从 “uncached fallback” 走向 “real CPU execution”

后续阶段统一把系统区分为三种执行形态。

## 4.1 形态 A：`S=N` 验证形态

特征：

1. 全部 expert 都驻留在 GPU slot。
2. heter/spec 仍经过完整 plan 与 heterogeneous forward。
3. 禁止通过检测 `S=N` 回退到 standard。

用途：

1. 验证 heter/spec 主链路 correctness。
2. 测量异构主链路控制开销。
3. 作为后续 CPU / draft 优化的公平 A/B 基线。

## 4.2 形态 B：当前 uncached-expert fallback 形态

特征：

1. expert 权重存放在 CPU pool。
2. uncached expert 执行时把权重搬到 GPU。
3. expert 计算主要在 GPU 上完成。
4. Python 热路径参与仍然较多。

用途：

1. 当前代码下的功能正确性兜底。
2. 真实 CPU 执行落地前的过渡路径。

限制：

1. 不能视作“真实 CPU expert execution”。
2. 不能作为未来 `S<N` 的目标性能模型。

## 4.3 形态 C：未来 real CPU expert execution 形态

目标特征：

1. expert matmul 真正在 CPU 上执行。
2. GPU cached experts 与 CPU experts 同层并行推进。
3. 两侧输出统一在聚合点合并。
4. `S<N` 的主性能结论只在该形态完成后才成立。

---

## 5. 分阶段设计

## 5.1 P0：可观测性与 benchmark 闭环优先

### 5.1.1 目标

在所有性能优化前，先统一 profile 与 benchmark 字段，保证后续结论可解释、可复现、可归因。

### 5.1.2 必须具备的 profile / benchmark 字段

1. `route_ms`
2. `plan_ms`
3. `gpu_gather_ms`
4. `gpu_compute_ms`
5. `cpu_prepare_ms`
6. `cpu_compute_ms`
7. `cpu_to_gpu_merge_ms`
8. `scatter_ms`
9. `draft_ms`
10. `verify_ms`
11. `spec_step_ms`
12. `graph_hit_rate`
13. `graph_replay_count`
14. `cpu_route_ratio`
15. `cpu_weight_mass_ratio`
16. `activated_expert_set_size`
17. `realized_cpu_expert_count`

### 5.1.3 需要修改的文件

1. `nanovllm/engine/model_runner.py`
2. `nanovllm/engine/speculative/spec_engine.py`
3. `nanovllm/layers/fuse_moe/heterogeneous.py`
4. `nanovllm/expert/placement.py`
5. `examples/heterogeneous_debug_profile.py`
6. `examples/heterogeneous_benchmark_case.py`
7. 新增 `examples/benchmarks/` 下的专项 benchmark 脚本

### 5.1.4 设计要点

1. 分段计时优先用 lightweight timer，必要时 CUDA 段做显式同步。
2. 统一由 `ModelRunner` 与 `SpeculativeEngine` 聚合 profile，避免 benchmark 脚本自行拼字段。
3. `heterogeneous_moe_forward` 输出结构中补充分段统计，便于 single-layer 与 end-to-end 复用。

### 5.1.5 伪代码

```python
def heterogeneous_moe_forward(..., profile: dict | None = None):
    t0 = now()
    selected_experts, routing_weights = route_tokens(...)
    prof_add(profile, "route_ms", now() - t0)

    t0 = now()
    plan = build_or_use_plan(...)
    prof_add(profile, "plan_ms", now() - t0)

    t0 = now()
    gpu_out = run_gpu_path(...)
    prof_add(profile, "gpu_compute_ms", now() - t0)

    t0 = now()
    cpu_out = run_cpu_path(...)
    prof_add(profile, "cpu_compute_ms", now() - t0)

    t0 = now()
    output = merge_outputs(gpu_out, cpu_out)
    prof_add(profile, "cpu_to_gpu_merge_ms", now() - t0)

    return output, profile
```

### 5.1.6 出口条件

1. heter / draft / verify 都能输出统一 profile 字段。
2. benchmark 结果可直接归因到 route / plan / gpu / cpu / merge / scatter。
3. 后续任何性能结论不再只基于总时延。

---

## 5.2 P1：真实 CPU expert execution

## 5.2.1 目标

把当前“CPU 存权重、GPU 做 matmul”的 fallback，改造成“CPU 真正执行 expert 计算”的路径。

### 5.2.2 需要修改的文件

1. `nanovllm/config.py`
2. `nanovllm/layers/fuse_moe/heterogeneous.py`
3. `nanovllm/expert/placement.py`
4. `nanovllm/utils/heterogeneous_loader.py`
5. 如需要，新增 `nanovllm/expert/cpu_executor.py`
6. 新增或补强 tests：
   - `tests/test_placement_spec.py`
   - `tests/test_model_runner_spec_modes.py`
   - 新增 `tests/test_cpu_expert_execution.py`

### 5.2.3 配置设计

在 `Config` 中建议新增：

```python
cpu_expert_execution_enabled: bool = False
cpu_expert_num_threads: int = 4
cpu_expert_parallel_mode: str = "serial"   # serial | expert_parallel
perf_profile_level: str = "basic"          # basic | detailed
```

约束：

1. `cpu_expert_num_threads >= 1`
2. `cpu_expert_parallel_mode in {"serial", "expert_parallel"}`

### 5.2.4 执行模型

1. CPU expert 权重保持在 CPU。
2. routed entries 按 expert id 分组。
3. CPU output buffer 与 workspace 复用。
4. 禁止逐 token Python 热路径循环。
5. 若首版保留逐 expert 循环，则每个循环体必须是 batched CPU matmul，而不是 token 级细粒度操作。

### 5.2.5 接口调整

建议把当前 CPU fallback 细化成两个显式分支：

1. `legacy_gpu_fallback`
2. `real_cpu_execution`

伪代码：

```python
def run_cpu_path(hidden_states, plan, cpu_expert_pool, config):
    if plan.cpu_route_indices is None or plan.cpu_route_indices.numel() == 0:
        return None

    if not config.cpu_expert_execution_enabled:
        return run_legacy_gpu_fallback(hidden_states, plan, cpu_expert_pool)

    return run_real_cpu_expert_execution(hidden_states, plan, cpu_expert_pool, config)
```

### 5.2.6 CPU executor 建议伪代码

```python
def run_real_cpu_expert_execution(hidden_states, plan, cpu_expert_pool, config):
    cpu_tasks = build_cpu_tasks(plan)   # expert_id -> token_indices / weights
    cpu_hidden = hidden_states.index_select(0, cpu_tasks.all_token_indices).to("cpu")

    if config.cpu_expert_parallel_mode == "serial":
        outputs = [run_one_cpu_expert(task, cpu_hidden, cpu_expert_pool) for task in cpu_tasks]
    else:
        outputs = thread_pool_map(run_one_cpu_expert, cpu_tasks, num_threads=config.cpu_expert_num_threads)

    cpu_out = assemble_cpu_outputs(outputs)
    return cpu_out.to(hidden_states.device, non_blocking=True)

def run_one_cpu_expert(task, cpu_hidden, cpu_expert_pool):
    params = cpu_expert_pool[task.expert_id]
    x = cpu_hidden.index_select(0, task.local_rows)
    gate_up = F.linear(x, params["gate_up"])
    out = F.linear(act_fn(gate_up), params["down"])
    out.mul_(task.weights.unsqueeze(-1))
    return task.token_indices, out
```

### 5.2.7 实现注意

1. CPU output 应尽量按 token contiguous buffer 回填，避免大量 Python list 拼接。
2. CPU 路径与 GPU 路径都应输出统一聚合前表示，减少 merge 分支复杂度。
3. 保留旧 fallback 作为兜底路径，直到 parity 建立完成。

### 5.2.8 出口条件

1. 单层 controlled routing 下，真实 CPU expert 输出与参考实现一致。
2. 在禁用 legacy GPU fallback 时，真实 CPU expert 路径可独立跑通。
3. 旧 fallback 仍然保留，不破坏现有功能。

---

## 5.3 P2：CPU/GPU 同层并行执行

## 5.3.1 目标

让 GPU cached experts 与 CPU experts 真正 same-layer overlap，而不是仅把两条路径都做成更复杂的串行逻辑。

### 5.3.2 需要修改的文件

1. `nanovllm/layers/fuse_moe/heterogeneous.py`
2. 如新增则 `nanovllm/expert/cpu_executor.py`
3. `tests/test_model_runner_spec_modes.py`
4. 新增 `tests/test_cpu_gpu_parallel_moe.py`

### 5.3.3 执行语义

1. GPU cached experts 在 CUDA stream 上执行。
2. CPU experts 在 CPU worker / thread pool 上执行。
3. 只在 merge 点前做一次必要同步。
4. 若 CPU 路径为空，直接跳过。
5. 若 GPU 路径为空，直接跳过。
6. 混合结果必须与串行参考一致。

### 5.3.4 伪代码

```python
def heterogeneous_moe_forward(...):
    gpu_future = None
    cpu_future = None

    if has_gpu_work(plan):
        gpu_future = launch_gpu_path_on_stream(hidden_states, plan, expert_cache)

    if has_cpu_work(plan):
        cpu_future = launch_cpu_executor(hidden_states, plan, cpu_expert_pool, config)

    gpu_out = wait_gpu_if_needed(gpu_future)
    cpu_out = wait_cpu_if_needed(cpu_future)

    return merge_gpu_cpu_outputs(gpu_out, cpu_out, plan)
```

### 5.3.5 合并策略

建议统一成“先得到 token-aligned partial output，再一次性 merge”：

```python
def merge_gpu_cpu_outputs(gpu_out, cpu_out, plan):
    output = zeros_like_hidden(...)
    if gpu_out is not None:
        output.add_(gpu_out)
    if cpu_out is not None:
        output.add_(cpu_out)
    return output
```

### 5.3.6 验证策略

1. 先做单层 synthetic routing。
2. 先验证结果正确，再观察 overlap 是否真的存在。
3. 不先对端到端收益做承诺。

### 5.3.7 出口条件

1. mixed CPU/GPU aggregation 与串行参考一致。
2. profile 能分离 CPU compute、GPU compute 与 merge。
3. profile 能证明 overlap 真实发生。

---

## 5.4 P3：`S<N` benchmark 与瓶颈分析

只有在 P1/P2 完成后，`S<N` benchmark 才转为主性能分析阶段。

## 5.4.1 Benchmark 组 A：单层受控 MoE benchmark

### 目标

1. 固定激活 expert 集合。
2. 扫描真实 CPU expert 数量。
3. 找出 CPU 成为瓶颈的拐点。

### 需要新增或修改的文件

1. 新增 `examples/benchmarks/moe_single_layer_cpu_gpu_parallel_bench.py`
2. 如需注入路由，修改 `nanovllm/models/qwen3_moe.py`
3. 可选补强 `examples/heterogeneous_debug_profile.py`

### 规则

1. 若激活 expert 集合大小固定为 `8`，则主扫描范围为 `0..8`。
2. `>8` 只作为非法输入或鲁棒性测试，不进入主性能曲线。
3. 至少覆盖两档 token 数规模。

### 伪代码

```python
for realized_cpu_expert_count in range(0, 9):
    selected_experts, routing_weights = build_controlled_routes(
        activated_expert_set_size=8,
        cpu_expert_count=realized_cpu_expert_count,
    )
    for _ in range(warmup):
        run_one_layer_forward(...)
    stats = timed_repeat(run_one_layer_forward, repeat=10)
    save_result(stats)
```

### 记录字段

1. `latency_ms_p50/p90/p99`
2. `cpu_route_ratio`
3. `cpu_weight_mass_ratio`
4. `cpu_prepare_ms`
5. `cpu_compute_ms`
6. `gpu_compute_ms`
7. `cpu_to_gpu_merge_ms`
8. `scatter_ms`
9. `activated_expert_set_size`
10. `realized_cpu_expert_count`

## 5.4.2 Benchmark 组 B：端到端 verify/spec benchmark

### 目标

在真实 speculative 场景下评估 CPU expert 占比变化对 verify/spec 的影响。

### 需要修改的文件

1. `examples/heterogeneous_benchmark_case.py`
2. 或新增 `examples/benchmarks/spec_verify_cpu_ratio_bench.py`
3. `nanovllm/engine/speculative/spec_engine.py`
4. `nanovllm/engine/model_runner.py`

### 档位

1. `25%`
2. `50%`
3. `75%`

### 记录要求

必须同时记录：

1. expert-set ratio
2. routed-entry ratio
3. routing-weight mass ratio

不允许只凭 set ratio 解释性能。

### 出口条件

1. 单层曲线能展示 CPU experts 增加时的拐点位置。
2. 端到端数据能区分 set ratio 与真实 route 压力。
3. 形成第一版 `S<N` 瓶颈地图。

---

## 5.5 P4：speculative 模式下的 placement/build 全 GPU 化

P4 的目标不是狭义的“draft 纯 GPU”，而是把 speculative 模式下 placement.py 中所有 build 相关热路径，统一改造成以 GPU 为主的数据面实现。也就是说，在 heter / draft / verify 三类 speculative 相关 plan 构建中：

1. 中间张量默认留在 GPU。
2. CPU/GPU expert 集合划分在 GPU 上完成。
3. GPU 计shid算所需的 route / slot / m_sizes / effective expert 等准备数据，直接在 GPU 上生成并直接用于后续 GPU 执行。
4. 只有 CPU expert 执行确实需要的数据，才从 GPU 回传到 CPU。

因此，P4 的真正名称应理解为：`speculative 模式下 placement/build 全 GPU 化`。

## 5.5.1 当前问题

当前 `nanovllm/expert/placement.py` 的 build 路径虽然已经使用了一部分 GPU tensor 操作，但仍然存在明显的 host crossing 和 Python object 依赖：

1. `build_draft_plan()` 中存在 `torch.unique(...).tolist()`，会把 uncached expert 集合拉回 host。
2. `DraftScheduler.select_cpu_experts()` 返回 Python `list[int]`。
3. `DraftScheduler.select_gpu_substitutes()` 返回 Python `dict[int, int]`。
4. `MoEExecutionPlan.substitution_map` 仍然是 Python 容器。
5. `cached_experts = set(expert_cache.expert_to_slot.keys())` 依赖 host 侧映射。
6. `need_substitution` / `cpu_experts` / `substitution_map` 等中间决策都在 host 上做，再写回 GPU 张量。

这些问题的结果是：

1. speculative 模式下 plan build 会发生不必要的 GPU -> CPU -> GPU 往返。
2. GPU 路径虽然最终执行在设备上，但必须等待 host 决策完成。
3. CPU 路径所需数据与 GPU 路径所需数据没有清晰拆分，导致“所有中间状态都经过 host 一遍”。
4. P5 的 Draft CUDA Graph 会被 host crossing 破坏可捕获性。

## 5.5.2 P4 的设计目标

P4 要实现的是一个更清晰的两层分工：

1. 数据面：全部在 GPU 上完成 route -> placement -> plan 的主流程。
2. 控制面：只保留低频策略参数、fallback 判定和调试输出在 CPU。

具体目标：

1. `build_prefill_plan`、`build_draft_plan`、后续需要新增的 `build_verify_plan`，都提供 GPU-first 版本。
2. GPU expert 集合、CPU expert 集合、substitution、effective selected experts、route indices、m_sizes，全部在 GPU 上生成。
3. GPU 路径不等待 CPU 上的中间变量构造完成。
4. 只在 CPU expert 执行真正开始前，把 CPU 分支所需的最小任务描述传回 CPU。
5. `heter` / `draft` / `verify` 三类 plan 尽可能共享统一张量接口。

## 5.5.3 改造后的数据流

改造后，speculative 模式下的 plan/build 数据流应如下：

1. `selected_experts` / `routing_weights` 在 GPU 上产生。
2. `expert_cache.expert_to_slot_lut` 在 GPU 上做 remap，得到 `slot_indices` 与 `gpu_mask`。
3. 在 GPU 上决定：
   - 哪些 routed experts 进入 GPU 集合
   - 哪些 routed experts 进入 CPU 集合
   - 哪些 uncached experts 需要 substitution
4. 在 GPU 上生成：
   - `flat_selected_effective`
   - `gpu_route_indices`
   - `cpu_route_indices`
   - `gpu_m_sizes`
   - `cpu_task_offsets` / `cpu_task_expert_ids` / `cpu_task_route_indices`
5. GPU 路径直接消费 `gpu_route_indices + gpu_m_sizes` 执行，不等待 CPU。
6. 只有 CPU executor 需要的最小任务描述张量，才在必要时回传 CPU。

也就是说，后续不再允许“先把中间 expert 集合传回 CPU 决策，再把结果写回 GPU”这种 build 流程。

## 5.5.4 需要修改的文件

核心代码文件：

1. `nanovllm/expert/placement.py`
2. `nanovllm/scheduling/draft_scheduler.py`
3. `nanovllm/expert/cache.py`
4. `nanovllm/models/qwen3_moe.py`
5. `nanovllm/layers/fuse_moe/heterogeneous.py`
6. 如需把 GPU build 从 placement 中拆分，新增 `nanovllm/expert/device_plan_builder.py`

相关调用与验证文件：

1. `nanovllm/engine/model_runner.py`
2. `tests/test_draft_scheduler.py`
3. `tests/test_model_runner_spec_modes.py`
4. 新增 `tests/test_device_plan_builder.py`
5. 新增 `tests/test_spec_plan_gpu_build.py`

## 5.5.5 需要调整的数据接口

### A. `LayerExpertCache`

当前 `LayerExpertCache` 已经有 `expert_to_slot_lut`，这是 P4 的基础。需要进一步明确它既是 remap 输入，也是 build 阶段的 GPU 数据源。

建议新增或强化接口：

```python
class LayerExpertCache:
    expert_to_slot_lut: Tensor          # [num_experts], on device
    cached_expert_mask: Tensor          # [num_experts], bool, on device

    def remap_experts_to_slots(self, selected_experts) -> tuple[Tensor, Tensor]:
        ...

    def get_cached_expert_mask(self) -> Tensor:
        return self.cached_expert_mask
```

其中 `cached_expert_mask` 用于避免从 `expert_to_slot` 这个 host dict 推导 cached 集合。

### B. `DraftScheduler`

当前接口是 host-oriented，需要改成 device-friendly 版本。建议增加新的 GPU 侧接口，并把旧接口保留为 fallback：

```python
class DraftScheduler(ABC):
    def select_cpu_experts_gpu(
        self,
        uncached_expert_mask: Tensor,     # [num_experts], bool
        routing_weights_flat: Tensor,     # [M * top_k]
        selected_experts_flat: Tensor,    # [M * top_k]
        top_c: int,
    ) -> Tensor:                          # [num_experts], bool
        ...

    def build_substitution_lut_gpu(
        self,
        cpu_expert_mask: Tensor,          # [num_experts], bool
        cached_expert_mask: Tensor,       # [num_experts], bool
        num_experts: int,
        device: torch.device,
    ) -> Tensor:                          # [num_experts], int64
        ...
```

目标不是让 scheduler 直接返回 Python `list/dict`，而是直接返回 GPU 可消费的 mask 和 LUT。

### C. `MoEExecutionPlan`

当前 plan 结构过于简化，且仍保留 Python 容器字段。P4 后建议扩展为完整的 device-first 结构：

```python
@dataclass
class MoEExecutionPlan:
    layer_idx: int

    gpu_route_indices: Tensor            # [R_gpu]
    gpu_m_sizes: Tensor | None           # [num_slots]

    cpu_route_indices: Tensor | None     # [R_cpu]
    cpu_task_expert_ids: Tensor | None   # [num_cpu_tasks]
    cpu_task_offsets: Tensor | None      # [num_cpu_tasks + 1]

    flat_selected_original: Tensor       # [M * top_k]
    flat_selected_effective: Tensor      # [M * top_k]
    substitution_lut: Tensor | None      # [num_experts]

    gpu_route_mask: Tensor | None        # [M * top_k], bool
    cpu_route_mask: Tensor | None        # [M * top_k], bool
```

说明：

1. `gpu_route_indices + gpu_m_sizes` 直接给 GPU 执行路径使用。
2. `cpu_task_expert_ids + cpu_task_offsets + cpu_route_indices` 用于 CPU executor 构建最小任务描述。
3. `substitution_lut` 在 debug / verify / graph 场景下都比 Python dict 更稳定。

## 5.5.6 `placement.py` 的详细改造方案

P4 后，`placement.py` 不再是“CPU 辅助做张量操作”的文件，而是 speculative 模式下的设备侧 plan builder。

建议重构为以下结构：

```python
def build_prefill_plan_gpu(...): ...
def build_draft_plan_gpu(...): ...
def build_verify_plan_gpu(...): ...

def _build_grouped_layout_gpu(...): ...
def _build_cpu_task_layout_gpu(...): ...
def _apply_substitution_lut_gpu(...): ...
```

旧接口如 `build_prefill_plan` / `build_draft_plan` 可以暂时保留为 wrapper，但内部应优先调用 GPU 版本。

### 关键子过程 1：GPU 上决定 CPU / GPU expert 集合

```python
def select_execution_sets_gpu(
    flat_selected: Tensor,              # [R]
    flat_weights: Tensor,               # [R]
    expert_cache: LayerExpertCache,
    draft_scheduler: DraftScheduler,
    top_c: int,
    num_experts: int,
):
    cached_expert_mask = expert_cache.get_cached_expert_mask()          # [E]
    selected_expert_mask = bincount(flat_selected, minlength=num_experts) > 0
    uncached_expert_mask = selected_expert_mask & (~cached_expert_mask)

    cpu_expert_mask = draft_scheduler.select_cpu_experts_gpu(
        uncached_expert_mask=uncached_expert_mask,
        routing_weights_flat=flat_weights,
        selected_experts_flat=flat_selected,
        top_c=top_c,
    )

    substitution_lut = draft_scheduler.build_substitution_lut_gpu(
        cpu_expert_mask=cpu_expert_mask,
        cached_expert_mask=cached_expert_mask,
        num_experts=num_experts,
        device=flat_selected.device,
    )
    return cpu_expert_mask, substitution_lut
```

### 关键子过程 2：GPU 上生成 effective selected experts

```python
def apply_substitution_lut_gpu(flat_selected, substitution_lut, cpu_expert_mask):
    flat_effective = substitution_lut.index_select(0, flat_selected)
    selected_cpu_mask = cpu_expert_mask.index_select(0, flat_selected)
    return flat_effective, selected_cpu_mask
```

### 关键子过程 3：GPU 上生成 GPU route / CPU route

```python
def build_route_layouts_gpu(flat_selected, flat_effective, cpu_expert_mask, expert_cache):
    slot_eff, gpu_mask_eff = expert_cache.remap_experts_to_slots(flat_effective)
    selected_cpu_mask = cpu_expert_mask.index_select(0, flat_selected)

    gpu_route_mask = gpu_mask_eff & (~selected_cpu_mask)
    cpu_route_mask = ~gpu_route_mask

    gpu_route_indices = nonzero(gpu_route_mask).flatten()
    cpu_route_indices = nonzero(cpu_route_mask).flatten()

    if gpu_route_indices.numel() > 0:
        gpu_slots = slot_eff.index_select(0, gpu_route_indices)
        gpu_m_sizes, gpu_route_indices = build_grouped_layout_gpu(gpu_slots, gpu_route_indices, expert_cache.num_slots)
    else:
        gpu_m_sizes = None

    cpu_task_expert_ids, cpu_task_offsets = build_cpu_task_layout_gpu(flat_selected, cpu_route_indices)
    return gpu_route_indices, gpu_m_sizes, cpu_route_indices, cpu_task_expert_ids, cpu_task_offsets
```

### 关键子过程 4：只把 CPU executor 必需数据回传 CPU

```python
def materialize_cpu_tasks_for_host(plan, flat_weights):
    if plan.cpu_route_indices is None or plan.cpu_route_indices.numel() == 0:
        return None

    return {
        "route_indices_cpu": plan.cpu_route_indices.to("cpu", non_blocking=True),
        "task_expert_ids_cpu": plan.cpu_task_expert_ids.to("cpu", non_blocking=True),
        "task_offsets_cpu": plan.cpu_task_offsets.to("cpu", non_blocking=True),
        "weights_cpu": flat_weights.index_select(0, plan.cpu_route_indices).to("cpu", non_blocking=True),
    }
```

这里的重点是：

1. 不是把所有中间变量都带回 CPU。
2. 只把 CPU executor 真正要消费的 route/task 描述回传 CPU。
3. GPU 路径不需要等待这些 host 数据构造完成。

## 5.5.7 `heterogeneous.py` 的接口调整

`heterogeneous_moe_forward()` 需要配合新的 device-first plan：

1. GPU 路径直接消费 `gpu_route_indices + gpu_m_sizes`。
2. CPU 路径只消费 `cpu_route_indices + cpu_task_expert_ids + cpu_task_offsets`。
3. 不再在 `heterogeneous.py` 内部重新做 host 侧 expert 集合解析。

建议伪代码：

```python
def heterogeneous_moe_forward(..., plan: MoEExecutionPlan):
    output = zeros_like(hidden_states)

    if plan.gpu_route_indices is not None and plan.gpu_route_indices.numel() > 0:
        gpu_out = run_gpu_path_with_device_plan(hidden_states, routing_weights, plan, expert_cache)
        output.add_(gpu_out)

    if plan.cpu_route_indices is not None and plan.cpu_route_indices.numel() > 0:
        cpu_tasks = materialize_cpu_tasks_for_host(plan, routing_weights.reshape(-1))
        cpu_out = run_cpu_path_with_task_layout(hidden_states, cpu_tasks, cpu_expert_pool)
        output.add_(cpu_out)

    return output
```

## 5.5.8 `qwen3_moe.py` 与调用链调整

当前 `Qwen3MoeHeterogeneousSparseMoeBlock.forward()` 在 draft / verify 时直接调用 `build_draft_plan()` / `build_prefill_plan()`。P4 后建议把调用逻辑改为：

1. `normal` -> `build_prefill_plan_gpu`
2. `draft` -> `build_draft_plan_gpu`
3. `verify` -> `build_verify_plan_gpu`

伪代码：

```python
if self.execution_mode == "draft":
    plan = build_draft_plan_gpu(...)
elif self.execution_mode == "verify":
    plan = build_verify_plan_gpu(...)
else:
    plan = build_prefill_plan_gpu(...)

return heterogeneous_moe_forward(..., plan=plan)
```

这样可以保证 speculative 模式下所有 build 逻辑都统一走 GPU-first 路径，而不是 normal / draft / verify 各自混用不同风格的 CPU/GPU build。

## 5.5.9 测试与验收

需要补充以下测试：

1. `tests/test_device_plan_builder.py`
   - 对比 `build_prefill_plan_gpu` / `build_draft_plan_gpu` / `build_verify_plan_gpu` 与旧逻辑输出一致。
2. `tests/test_spec_plan_gpu_build.py`
   - 确认 draft/verify 下不发生 Python `list/dict` 驱动的核心 build 分叉。
3. `tests/test_draft_scheduler.py`
   - 覆盖 GPU mask / LUT 版本接口。
4. `tests/test_model_runner_spec_modes.py`
   - 确认 speculative 模式下 plan build 结果可被后续执行链直接消费。

验收条件：

1. speculative 模式下 placement/build 主路径不再依赖 `.tolist()`、Python `list`、Python `dict`。
2. GPU 执行所需 plan 数据全部在 GPU 上直接生成并直接消费。
3. 只有 CPU executor 必需的数据才会回传 CPU。
4. `heter` / `draft` / `verify` 三类 plan 的数据接口统一到同一套 device-first 结构。
5. P5 Draft CUDA Graph 的 capture 路径不再被 placement/build 中的 host crossing 阻塞。

---

## 5.6 P5：Draft CUDA Graph

## 5.6.1 启用策略

在以下条件同时成立时，Draft CUDA Graph 自动启用：

1. `enforce_eager == False`
2. `draft_top_c == 0`

也就是说，只要用户没有强制 eager，且 Draft 不涉及 CPU expert 执行，系统默认优先尝试 graph 路径。

## 5.6.2 设计原则

1. Draft CUDA Graph 自动启用，但必须允许模板未命中时自动回退 eager。
2. Draft graph 是 draft-mode 专属 graph，不允许隐式复用 standard decode graph。
3. graph 支持范围由模板命中决定，而不是由“只支持单序列”人为限制。
4. 只要 `enforce_eager=False` 且 `draft_top_c=0`，就应进入“优先 graph、失败回退 eager”的自动模式。

## 5.6.3 需要修改的文件

1. `nanovllm/config.py`
2. `nanovllm/engine/model_runner.py`
3. `nanovllm/models/qwen3_moe.py`
4. `nanovllm/engine/speculative/spec_engine.py`
5. `nanovllm/utils/context.py`
6. 相关 tests：
   - `tests/test_model_runner_spec_modes.py`
   - 新增 `tests/test_draft_cuda_graph.py`

## 5.6.4 配置设计

建议在 `Config` 中补充：

```python
draft_cuda_graph_enabled: bool = True
draft_cuda_graph_max_bs: int = 512
draft_cuda_graph_bucket_steps: list[int] = [1, 2, 4, 8]
```

语义：

1. `draft_cuda_graph_enabled` 表示是否允许自动 graph。
2. 真正的运行时启用条件为：
   - `draft_cuda_graph_enabled`
   - `not enforce_eager`
   - `draft_top_c == 0`

## 5.6.5 关键接口设计

### ModelRunner

新增 draft graph 能力：

```python
def run_draft(self, seqs):
    self._set_speculative_execution_mode("draft")
    try:
        if self._can_use_draft_cudagraph(seqs):
            return self.run_draft_graph(seqs)
        return self.run(seqs, False)
    finally:
        self._set_speculative_execution_mode("normal")
```

新增判断函数：

```python
def _can_use_draft_cudagraph(self, seqs):
    return (
        self.config.draft_cuda_graph_enabled
        and (not self.enforce_eager)
        and getattr(self.config, "draft_top_c", 0) == 0
        and self._draft_graph_template_supported(seqs)
    )
```

### Draft graph capture / replay

建议在 `ModelRunner` 中增加独立的 draft graph cache：

```python
self.draft_graphs = {}
self.draft_graph_vars = {}
```

伪代码：

```python
def capture_draft_cudagraph(self):
    for bucket in self.draft_graph_buckets:
        graph = torch.cuda.CUDAGraph()
        vars = allocate_draft_graph_buffers(bucket)
        self._set_speculative_execution_mode("draft")
        with torch.cuda.graph(graph):
            vars["outputs"][:] = self.model(vars["input_ids"], vars["positions"])
        self._set_speculative_execution_mode("normal")
        self.draft_graphs[bucket] = graph
        self.draft_graph_vars[bucket] = vars

def run_draft_graph(self, seqs):
    bucket = select_draft_bucket(seqs)
    vars = self.draft_graph_vars[bucket]
    fill_draft_graph_inputs(vars, seqs)
    self.draft_graphs[bucket].replay()
    return self.model.compute_logits(vars["outputs"][:batch_size])
```

## 5.6.6 为什么必须是 draft-mode 专属 graph

当前代码中，`run_draft()` 是先切换模型到 `"draft"` 模式，再走 decode 路径。Draft mode 下 MoE block 会调用 draft-specific plan 逻辑，而 standard decode graph 捕获的是 normal mode 行为。

因此：

1. standard decode graph 不能直接视作 draft graph。
2. Draft graph 必须在 draft-mode 下单独 capture。
3. 若模板未命中或不支持，则自动回退 eager。

## 5.6.7 模板策略

模板 key 建议至少包含：

1. `batch_size bucket`
2. `max_num_blocks / block_tables width bucket`
3. `draft step bucket`
4. `device id`

注意：

1. 这里不把 `num_seqs=1` 作为首版限制。
2. 多序列支持由 bucket 命中情况决定，而不是功能层面禁用。

## 5.6.8 出口条件

1. 当 `enforce_eager=False` 且 `draft_top_c=0` 时，Draft 自动优先尝试 graph。
2. 模板命中时 replay；模板未命中时自动回退 eager。
3. deterministic 条件下 eager 与 graph 输出一致。
4. benchmark 输出 `graph_hit_rate` 与 `graph_replay_count`。

---

## 6. 基准与验收口径

## 6.1 `S=N` 正确性验收

1. `standard / heter / spec` 在 deterministic 条件下 token 对齐。
2. 不允许对 heter/spec 主路径做 `S=N` 特判绕过。

## 6.2 真实 CPU 执行正确性验收

1. 单层 controlled routing 下，CPU expert 输出与参考实现误差在容忍范围内。
2. mixed CPU/GPU same-layer aggregation 与串行参考一致。

## 6.3 单层时延验收

1. CPU expert 数量主扫描范围为 `0..8`。
2. 至少覆盖两档 token 数规模。
3. 预热与正式计时次数固定并写入 benchmark 脚本。

建议默认：

1. warmup `3` 次
2. timed run `10` 次

## 6.4 端到端 spec/verify 验收

1. 测试 CPU expert-set ratio `25% / 50% / 75%`。
2. 必须同步记录：
   - set ratio
   - route ratio
   - weight-mass ratio

## 6.5 Draft CUDA Graph 验收

1. 当 `enforce_eager=False` 且 `draft_top_c=0` 时，自动优先尝试 draft graph。
2. eager 与 graph 在 deterministic 条件下输出一致。
3. 模板未命中时自动回退 eager。
4. 输出 `graph_hit_rate` 与 `graph_replay_count`。
5. 只对命中的模板 bucket 声明 graph 性能结果。

---

## 7. 性能预期与风险

## 7.1 保守性能预期

### P1 / P2：真实 CPU execution + CPU/GPU 并行

预期：

1. 在 CPU-heavy fallback 场景下可能带来较大收益。
2. 对当前 `S=N` 主验收 benchmark 的直接影响有限，因为 `S=N` 下 CPU 路径本来就接近不触发。

### P4：draft 纯 GPU plan 清理

预期：

1. 独立时延收益通常是小到中等。
2. 更重要的是清理接口形态，为 graph 化与减少 host 参与打基础。

### P5：Draft CUDA Graph

预期：

1. 对 draft decode launch overhead 有直接帮助。
2. 端到端收益仍会被 verify / accept 的 eager 段限制。
3. 真正收益大小取决于模板命中率，而不是是否单序列。

### 非目标

1. 不承诺 P1 完成后 `S<N` 立即优于 `S=N`。
2. 第一目标是 correctness、稳定性与可解释 scaling。

## 7.2 主要风险

1. 风险：真实 CPU 执行引入额外 host 调度开销，收益不明显。
   - 缓解：先单层 benchmark，再决定并行粒度。
2. 风险：CPU/GPU 并行实现引入额外同步，overlap 不成立。
   - 缓解：profile 必须验证 overlap，而不是只看总时间。
3. 风险：`S<N` 只看 set ratio，误判瓶颈来源。
   - 缓解：同步记录 route ratio 与 weight-mass ratio。
4. 风险：Draft graph 错误复用 standard decode graph。
   - 缓解：draft-mode 下独立 capture / replay。
5. 风险：graph 模板过多导致维护复杂度上升。
   - 缓解：采用 bucket 化模板与自动 eager fallback。

---

## 8. 配置项建议

建议新增或标准化如下配置项：

```python
cpu_expert_execution_enabled: bool = False
cpu_expert_num_threads: int = 4
cpu_expert_parallel_mode: str = "serial"   # serial | expert_parallel

gpu_plan_builder_enabled: bool = False
gpu_plan_builder_fallback: bool = True

draft_cuda_graph_enabled: bool = True
draft_cuda_graph_max_bs: int = 512
draft_cuda_graph_bucket_steps: list[int] = [1, 2, 4, 8]

perf_profile_level: str = "basic"          # basic | detailed
```

约束：

1. 本文档不额外引入更复杂的策略配置，除非实现顺序确实需要。
2. 遵循“先最小闭包，后策略扩展”的原则。

---

## 9. 实施顺序与阶段出口条件

建议按以下顺序推进：

1. `P0`：统一 profile 与 benchmark 字段。
2. `P1`：实现真实 CPU expert execution。
3. `P2`：实现 CPU/GPU same-layer overlap。
4. `P3`：展开 `S<N` 单层与端到端 benchmark。
5. `P4`：清理 draft 纯 GPU planning 接口与数据结构。
6. `P5`：实现 automatic Draft CUDA Graph。

阶段切换规则：

1. 未完成 `P1`，不把 `S<N` 作为主性能结论来源。
2. 未完成 `P4`，Draft graph 仍可能受 Python/host crossing 限制。
3. `P5` 完成后，`enforce_eager=False` 且 `draft_top_c=0` 时，Draft graph 为自动优先路径。

---

## 10. 与已有文档的关系

1. [docs/migration_design.md](./migration_design.md)
   - 提供总体迁移方向与长期架构目标。
   - 本文档对其在当前代码基线上的后续落地顺序进行重排与收敛。

2. [docs/phase2_design.md](./phase2_design.md)
   - 定义了 speculative baseline 与 `S=N` 约束。
   - 本文档延续其结论，并明确 `S=N` 只是当前主验收基线，而不是长期目标。

3. [docs/phase2_post_spec_cpu_parallel_gpu_schedule_design.md](./phase2_post_spec_cpu_parallel_gpu_schedule_design.md)
   - 作为旧版草案保留。
   - 本文档对其做了三点关键修正：
     - 明确当前 fallback 不是 real CPU execution
     - 明确 `S=N -> S<N` 的阶段切换条件
     - 明确 Draft CUDA Graph 在 `enforce_eager=False` 且 `draft_top_c=0` 时应自动启用，且不以 `num_seqs=1` 作为功能限制

4. [docs/summary/phase1_cpu_gpu_basic_heterogeneous_report.md](./summary/phase1_cpu_gpu_basic_heterogeneous_report.md)
   - 提供 `S=N` 基线与当前瓶颈证据。

5. [docs/summary/phase2_speculative_decoding_report.md](./summary/phase2_speculative_decoding_report.md)
   - 提供 speculative baseline 当前状态与下一阶段缺口。

综上，本文档的定位是把当前代码事实、已有 benchmark 结论与下一阶段工程工作整合为一个可执行的详细设计与计划文档。
