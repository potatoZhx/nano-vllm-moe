# Predictive Prefetcher 重设计方案（Plan）

**Project:** nano-vllm-moe
**Target Model:** Qwen3-30B-A3B (N=128, k=8, 48 MoE layers)
**Hardware:** Single GPU + Host DRAM, PCIe 4.0 ×16
**Date:** 2026-06-01
**前置文档:** `post_experiment_optimization.md` §5, `system_design_report.md`
**状态:** 设计讨论中（本文档用于锚定现状观察 + 新方案,后续迭代具体算法与实现）

---

## 0. 本文档目的

1. 固化我们对**现有 prefetcher 行为**的观察结论（来自 `post_experiment_optimization.md` §5 及后续 code review 讨论）。
2. 记录**新 prefetcher 方案**的初步设计，作为可替换现有 prefetcher 的 option。
3. 标注需要进一步讨论/确认的开放问题，作为下一轮算法与实现细化的输入。

**核心约束**：新 prefetcher 作为一个**可选项**接入，开启时替换现有 prefetcher；关闭时，现有所有实现路径必须**逐字节不变**（零影响）。

---

## Part A — 现有 Prefetcher 行为观察

### A.1 总体架构

预取系统由 `PrefetchRuntime`（`nanovllm/expert/prefetcher.py`）与 `ModelRunner`（`nanovllm/engine/model_runner.py`）协作完成，是一条**异步流水线**：

```
forward 计算（读 cached_expert_mask 判命中 / reroute）        ← 主线程，实时
  → offload_async() metadata D2H 拷贝                          ← 异步入队
  → prefetch worker 线程:
       collect() → observe_*() → mark_access() + 队列更新       ← 频率统计 & 候选更新
       _submit_prefetch_after_metadata() → H2D 专家传输          ← 发起预取
```

### A.2 三个信息源（observe 入口）

| 信息源 | 入口方法 | 数据内容 | 门控开关 |
|--------|---------|---------|---------|
| Prefill routing | `observe_prefill()` | prefill 阶段 router 输出 | `prefetch_use_prefill_history` |
| Draft routing | `observe_draft()` | draft 每步 forward 的 gating metadata | `prefetch_use_draft_live` |
| Verify routing | `observe_verify()` | verify 阶段 router 输出 + rank_guard 更新 | `prefetch_use_verify_history` |

### A.3 关键观察：`observe_*()` 内部做了两件正交的事

`observe_runtime_meta()`（prefetcher.py:475）对每批 metadata 做**两次独立累加**：

| | `mark_access()` → `cache.access_count` | `update_from_runtime_meta()` → 预取队列 |
|---|---|---|
| 用途 | LFU 驱逐：选**已缓存**专家踢出 | 预取排序：选**未缓存**专家搬入 |
| 范围 | 全部 128 专家（cached + uncached） | 仅 uncached（cached 被过滤） |
| 衰减 | **无衰减**，`+= 1` 单调累加 | 有 decay，`decay*old + new`（近期加权） |
| 语义 | 激活**频率**（不是命中数） | 候选优先级（含 age/TTL/priority） |

当前优先级函数（prefetcher.py:47）：

```
priority = source_weight × score_sum + activation_count_weight × activation_count − age_penalty × age
```

### A.4 关键观察：频率统计是延迟的、异步的

- `access_count += 1` **不在 forward 路径**发生，唯一写入点是 worker 线程里的 `mark_access`（cache.py:169 / 199）。
- 计算时的 cache 命中判定（实时）用的是 `cached_expert_mask` / `expert_to_slot_lut`，**不碰** `access_count`。
- flush 是 `block=False` 机会式处理，频率更新可能**滞后数个 step**。
- 后果 1：LFU 驱逐基于"滞后的频率"，topic 切换时反应慢一拍。
- 后果 2：`access_count` 是"激活频率"，不是"命中率"；当前没有任何 per-expert 命中/未命中计数器。

### A.5 关键观察：冗余与耦合

- **冗余**：`mark_access` 与队列更新在同一 worker 线程、对**同一批 offload 后的 metadata** 各做一遍 `torch.unique + scatter_add`。"哪个 uncached 专家热"这一信号，cache 统计里已有，队列又重算并存一份。两者时机一致，合并无时序障碍。
- **耦合**：`mark_access`（LFU 信号）与队列更新被绑在同一 observe 调用里，且共享 `prefetch_use_*_history` 开关。后果是无法"用 prefill 喂 LFU 但不喂预取队列"——两个本应正交的东西被耦合了。

### A.6 现有两阶段预取流程

**Draft 阶段**（三种模式，由 `prefetch_runtime_mode` 选择）：

| 模式 | 方法 | 特点 |
|------|------|------|
| `baseline_staging` | `submit_from_global_queue()` | 预取到 staging buffer，安全点 publish |
| `draft_direct_active` | `submit_draft_direct_active_prefetch()` | 直接进 active slot，frontier 约束保护未执行层 |
| `draft_segment_indexed` | `submit_draft_segment_indexed_prefetch()` | 推荐。按 segment 边界提交，合并 long_term + draft 两个 SegmentIndex |

约束：frontier 保护（仅预取 `layer ≤ frontier`）、adaptive budget（按可见开销自适应）、stale metadata guard。

**Verify 阶段**（`before_verify_layer()` hook，model_runner.py:1452）：

```
before_verify_layer(layer_idx=l)
  → publish_direct_active_ready()
  → available_ms = EMA(layer_compute_ms) × safety_ratio
  → submit_verify_layer_prefetch(target_layer_idx=l+1, available_ms=...)   # 只预取下一层
```

候选来自 `GlobalWarmStartQueue`（混合 draft/prefill/verify 历史），过滤 `layer == l+1`。

### A.7 现有设计的局限（与新方案动机相关）

1. 优先级未区分 top-1/2 重要性（score_sum 一视同仁）。
2. Verify 层预取候选来自混合队列；draft 当轮 per-layer 预测在 `end_draft_iteration()` 时被清空丢弃。
3. 频率统计与预取候选共享数据源、互相纠缠（A.5）。

---

## Part B — 新 Prefetcher 方案设计

### B.0 最主要目标与核心约束

**主要目标**：利用 **draft 预测信息** 降低 **verify 期间的 cache miss**。draft 阶段把 verify 将要用到的（draft 预测的）miss expert 提前搬进 GPU。

**核心约束——预取不与 draft 计算冲突**：
- **冲突的定义**：prefetch 替换第 i 层的 expert cache slot，而第 i 层在**本次 draft forward** 中正要/正在被计算。
- 每层有独立的 `LayerExpertCache`，因此"预取 segment s 的 expert"只触碰 segment s 各层的 cache，只可能与 segment s 的计算冲突。

### B.1 设计原则

1. **可替换、零影响**：作为独立 option 接入；关闭时现有路径不变。
2. **数据源彻底分离**（解决 A.5 的冗余与耦合）：
   - **Expert cache 的频率统计**：**仅**记录 prefill/verify 的 **ground-truth** 激活频率。
   - **预取队列**：**仅**由 draft 预测更新，不再持有 prefill/verify 来源信息。
3. **段级安全流水线**：用"已算完的段对预取是安全的"这一性质，把预取交错进 draft 计算而不引发冲突（见 B.3 安全规则）。

> 术语：以下假设单次 forward 被切成 n 个 segment，编号 `segment 0 … segment n-1`（沿用现有 `draft_prefetch_segment_size`，如 48 层 / 12 = 4 段）。

### B.2 数据源分离后的两套统计

| 结构 | 数据来源 | 用途 |
|------|---------|------|
| `cache.access_count`（ground-truth 频率） | **仅** prefill + verify | ① LFU 驱逐 ② 新方案 draft 冷启动（阶段 1）的频率查询 |
| 预取队列（draft 预测） | **仅** draft routing | draft/verify 阶段预取候选排序，每轮 draft 内累积 |

> 关于"draft 预测"：项目**暂无对 draft experts 的预测机制**。这里的"预测"= 把**已观测的 draft routing**（draft forward 的实际 gating metadata）当作启发式预测器使用，其有效性依赖两个先验：① **draft↔verify 路由一致性**（draft 选中的专家 verify 大概率也选）；② **token 激活局部性**（相近 token 的专家激活有重合，motivation 实验结论）。并非训练得到的预测模型。

### B.3 Draft 预取：两阶段

#### 阶段 1 — 冷启动（首个 draft step 的 segment 0）

时机：prefill/verify 之后的**第一步 draft**，正在计算 **segment 0**。此时**没有任何 draft 预测信息**。

流程：
1. Prefetcher 向 expert cache 发送：
   - **估计预取数量** = segment 0 计算时间能够掩盖的传输数量。
   - **目标 segment** = **`segment n-1`**（仅此一段）。
2. Expert cache 返回：`segment n-1` 内、**miss（未缓存）** 且 **ground-truth 激活频率最高** 的 expert id 列表。
3. Prefetcher 在 segment 0 计算期间预取这些 expert。

要点：
- **为什么只取 `segment n-1`**：segment 0 计算时，`n-1` 在本次 forward 最后才被读，runway 最长，最不易冲突——与阶段 2 的 segment 0 特例一致。限定单段以**保证安全性**（不向 `1…n-2` 这些更早被读的段写入，避免传输超时导致冲突）。
- 冷启动期没有 draft 预测，只能退回到 cache 的 ground-truth 频率作为先验——复用 B.2 中 cache 专属的频率统计。
- **取舍**：仅 `n-1` 一段会限制可预取的 expert 数量（至多一段的 miss 数）。若估计预取量 > `n-1` miss 数，则 segment 0 的窗口未被充分利用。是否接受此保守取舍，或允许溢出到 `n-2`（牺牲一点安全裕度）见 D.8。

#### 阶段 2 — draft 期间其余 segment（预测驱动 + 安全规则）

时机：draft 期间 segment 0 之后的各 segment（含非首步 draft 的所有 segment）。

**安全规则**（intra-step，约束在本次 forward 内）：

- **保守模式（容忍预取超时）**：计算 segment i 时，预取 **segment `0 … i-1`** 的 experts。理由：segment `0…i-1` 在本次 forward 已计算完毕，本次 forward 结束前不会再被读取；即使预取估计不准、传输未在 segment i 的计算窗口内完成，也**不会冲突**。
- **segment 0 特例（非首步 draft）**：计算 segment 0 时，`0…i-1` 为空，故预取 **segment `n-1`** 的 experts——它在本次 forward 最后才被读，runway 最长，最不易冲突。
- **紧模式（预取保证在 segment i 计算窗口内完成）**：约束放松为"只要预取的**不是 segment i**（当前正在计算的段）的 experts 即可"——因为传输在 segment i 内即完成，后续 `i+1…n-1` 被读到时 slot 已稳定。

**预取内容**：受安全规则约束的目标段内、由 **draft 预测** 指示的 miss experts。计算 segment i 时，队列中已含本轮所有先前 draft step 的预测 + 本次 forward 中 segment `0…i-1` 的最新预测，覆盖目标段绰绰有余。

队列生命周期：
- 预取队列**只保留每轮 draft 的预测信息**。"每轮"= 上一次 verify/prefill 结束 到 下一次 verify 开始之前的**所有 draft step**。
- 即队列在一轮内跨 draft step 累积 draft 预测，不混入 prefill/verify。

### B.4 Verify 预取

- 计算第 `i` 层时，依据 **draft 预取队列** 预取第 `i+1` 层的 experts。
- 预取数量受约束：不能阻塞下一层（`i+1`）计算 → 复用现有 `available_ms` 时间预算机制。
- **更保守的窗口约束**：verify 的 demand-load（cache_fill，发生在每层 MoE 段）会与预取**竞争 PCIe 通信资源**。为给 demand-load 让路，verify 预取的传输窗口应**尽量限制在 attention 计算期间**（即只用层内 attention 段的 compute 时间作预算，MoE 段不发起预取传输）。详见 E.13。
- **verify 完成后，预取队列清空**（一轮 draft 预测的生命周期到此结束）。

### B.5 时序示意

记 `S0..S(n-1)` 为本次 forward 的 segment；预取目标段按 B.3 安全规则（保守模式）标注。

```
[prefill / 上一轮 verify 结束]  → cache.access_count 含最新 ground-truth 频率
        │
        ▼
Draft step 0 (首步):
  S0 计算  ──(阶段1: 查 cache 频率, 预取 S1..S(n-1) 的高频 miss expert)──►
  S1 计算  ──(阶段2: 预取 S0 的预测 miss)──►            # S0 已算完, 安全
  S2 计算  ──(阶段2: 预取 S0..S1 的预测 miss)──►
  ...
  S(n-1) 计算 ──(阶段2: 预取 S0..S(n-2) 的预测 miss)──►  → 队列累积本步预测
Draft step 1..K (非首步):
  S0 计算  ──(阶段2 特例: 预取 S(n-1) 的预测 miss)──►     # runway 最长, 安全
  S1 计算  ──(阶段2: 预取 S0 的预测 miss)──►
  ...
        │  draft 队列 = 本轮所有 draft step 的预测累积 (不含 prefill/verify)
        ▼
Verify:
  layer i 计算 ──(依据 draft 队列, 预取 layer i+1, 不阻塞下一层)──►
  ...
  [verify 完成] → 清空预取队列；cache.access_count 追加本轮 verify ground-truth
```

> 注：阶段 2 预取的是"已算完段"的预测 miss expert，其消费者是 **verify**（以及下一 draft step 的同段计算），而非本次 forward 后续段——这与"用 draft 预测降低 verify miss"的主目标一致（B.0）。

### B.6 Scope 与副作用

**1. Draft 自身的 cache miss 不在本方案改善范围内（可接受）**
- 安全规则使本次 forward 的后续段 `i…n-1` 无法被预取（会冲突），且项目暂无 draft expert 预测机制。
- 因此 draft 自身的 miss 仍由 **reroute** 兜底，本方案不直接削减它。这是明确接受的取舍——prefetch 专注服务 verify。

**2. Draft cache 保护属于 cache 策略层、与本 prefetcher 正交，且暂无证据有效**
- 可在 expert cache 策略中基于经验保护 draft 质量（如保护 top-1/2 激活），但当前测试显示此类策略相比 **LFU 无明显改善**。
- 故本方案**不依赖** draft cache 保护带来收益；是否启用由 cache 策略独立决定（§2.2 RankGuard）。

**3. 副作用（正向，待测）：verify 导向的预取可能顺带改善 draft cache**
- 依据 motivation 实验"相近 token 的 expert 激活有重合"：
  - **Intra-round**：draft step t 在已算完段预取的专家，step t+1 重新计算同段时（相近 token）大概率仍需要 → 命中。
  - **Inter-round**：本轮为 verify 预热的专家，下一轮 draft 从已接受 token 续写（相近 token）→ 命中。
- 即本方案虽不显式为 draft 预取，token 局部性可能让 draft cache 命中率被动受益。
- **风险**：预取按 LFU 选 victim 驱逐，若驱逐了 draft 当前需要、又与 verify 预测不重合的专家，可能反而拉低 draft 命中。局部性强时净效应应为正，但需实测（见 D.7）。

### B.7 单轮预取保护（防止单轮内反复换入换出）

**动机**：一轮内（上一次 verify/prefill 结束 → 下一次 verify 开始）跨多个 draft step、多个 segment 边界反复提交预取。若不加约束，某个本轮刚预取进来的 expert 可能在同一轮稍后又被另一次预取的 victim 选择驱逐，下一 step 再次需要时又得重新搬入——产生**单轮内的抖动（thrashing）**，浪费 PCIe 带宽且抵消预取收益。

**机制**：对**本轮已预取进 cache 的 expert** 施加"单轮不被驱逐"保护。

```
_round_loaded: dict[layer_idx, set[expert_id]]   # 本轮已预取的专家
```

- **写入**：每次预取成功 reserve active slot 时，将 `(layer, expert)` 记入 `_round_loaded`。
- **victim 选择**：后续预取选 victim slot 时，跳过 `_round_loaded[layer]` 中的 expert。
- **安全阀**：若某层所有候选 slot 都被本轮保护（无可驱逐项），回退到纯 LFU（在全部 slot 上选最低频），避免死锁——与 §2.2 RankGuard 的 all-protected fallback 同构。
- **清空**：verify 完成、预取队列清空时，同时清空 `_round_loaded`（保护仅限本轮）。

**与其他保护的关系**：
- 本保护是**临时的、轮级的**（防抖动），与 §2.2 LFU-RankGuard 的**长期 top-1/2 保护**正交，可叠加。
- `_round_loaded` 中的 expert 已在 cache 内，因此天然不会再作为预取候选（被"is_cached"过滤），无需在队列侧额外处理——保护只作用于 **victim 选择侧**。

**待定**（见 D.9）：是否也保护本轮 **demand-load**（forward 计算时按需载入）的 expert，而不仅是预取载入的。

### B.8 预取队列更新算法（draft 预测打分）

队列只由 draft routing 更新（B.2），对每个候选 `(layer, expert)` 维护一个**优先级分数**，排序越靠前越优先预取。打分综合三个信号（与你的设想一致）：

| 信号 | 含义 | 方向 |
|------|------|------|
| **topk rank** | draft 预测中该 expert 的 topk 位次（rank 0 = top-1） | 越靠前 → 越高 |
| **routing score** | draft router 给该 expert 的权重 | 越高 → 越高 |
| **draft 激活频率** | 本轮 draft 期间该 expert 被预测激活的次数 | 越高 → 越高 |

**推荐形式——统一累加（单次累加自然融合三信号）**：

对本轮每一次 draft 激活（某 token 在某 step 选中 expert j，topk 位次 r，routing 权重 w）：

```
priority[j] += w × rank_factor(r)
```

其中 `rank_factor(r)` 是位次的递减函数，两种候选：
- 线性：`rank_factor(r) = (k − r) / k`
- 谐波（更强调 top-1）：`rank_factor(r) = 1 / (r + 1)`

这一形式的好处：**频率被求和项数自然编码**（激活越多 → 累加项越多 → 分数越高），无需单列频率项。即一个公式同时吃下 rank、score、freq 三者。

**可选——显式加权变体**（便于单独调参）：

```
priority[j] = α · best_rank_factor(j) + β · norm(score_sum[j]) + γ · norm(freq[j])
```

`best_rank_factor` 取本轮最佳（最小）rank 的 factor。`α,β,γ` 可调；norm 为轮内归一化。代价是多维调参。

**跨 step 时效（可选）**：是否对较早 draft step 的贡献做轻衰减，偏向最近 step（其 token 离 verify 关心的 token 更近，局部性更强）：

```
priority[j] = Σ_step decay^(cur_step − step) × Σ_token [ w × rank_factor(r) ]
```

默认 `decay = 1.0`（轮内不衰减，等权累加）；若实测显示近 step 预测更准，再调小。

**字段**：队列每项维护 `priority`（上述累加值）、`best_rank`（tie-break / 可用于 B.7 之外的 boost）、`last_step`（衰减用）。排序键 `(-priority, layer, expert)`。已缓存项被过滤（不进候选）。

---

## Part C — 实现计划（高层，细节待讨论）

### C.1 零影响接入机制（确定方案：子类化 + 工厂选择）

**目标**：新 prefetcher 开启时替换、关闭时现有路径**逐字节不变**。

**核心洞察**：`model_runner` 与预取系统的所有交互都是**通过方法名调用** `self.prefetch_runtime.<method>(...)`（鸭子类型），唯一按字符串分派的是 `_prefetch_runtime_mode()` 驱动的几个 gate。因此只要新实现**保持相同的方法签名表面**，且让那几个 gate 正确触发，`model_runner` 的方法体几乎无需改动。

**方案：`PredictivePrefetchRuntime(PrefetchRuntime)` 子类化**

```
class PredictivePrefetchRuntime(PrefetchRuntime):
    # 继承全部基础设施：CUDA streams、inflight 管理、
    # reserve/publish active slot、metadata offload、profiling
    # 仅 override 行为不同的方法（见下）
```

**override 的方法（新逻辑全部封装在子类内）**：

| 方法 | 新逻辑 |
|------|--------|
| `observe_prefill` / `observe_verify` | 只更新 `cache.access_count`（ground-truth 频率），**不**喂队列（解开 A.5 耦合） |
| `observe_draft` | 只更新 draft 预测队列（B.8 打分），**不**写 access_count |
| `begin_draft_iteration` | 轮开始：初始化 `_round_loaded`（B.7）、阶段标志 |
| `end_draft_iteration` / verify 后 | 清空队列 + `_round_loaded`（轮结束） |
| `submit_draft_segment_indexed_prefetch` | 两阶段预取（B.3）：阶段 1 查 cache 频率取 `n-1`，阶段 2 按安全规则预取 `0…i-1` |
| `submit_verify_layer_prefetch` | 依据 draft 队列预取 `layer+1`（B.4） |
| `_select_publish_slot` / `_select_publish_slot_cpu` | victim 选择叠加 B.7 单轮保护（跳过 `_round_loaded`，all-protected 回退 LFU） |

**`model_runner` 的唯一改动 —— `__init__` 工厂选择**：

```python
# 新增 config: prefetch_runtime_kind: "legacy"（默认）| "predictive"
kind = getattr(config, "prefetch_runtime_kind", "legacy")
PrefetchCls = PredictivePrefetchRuntime if kind == "predictive" else PrefetchRuntime
self.prefetch_runtime = PrefetchCls(...)   # 构造参数不变
```

**让现有 gate 正确触发**：predictive kind 复用 `prefetch_runtime_mode = "draft_segment_indexed"`，使以下 gate 自然命中（均已存在，无需改）：
- `run_draft`（model_runner.py:1280）→ `begin_draft_iteration`
- `wait_prefetch_for_verify`（:1394）→ `end_draft_iteration`
- `_submit_prefetch_after_metadata`（:371）→ `submit_draft_segment_indexed_prefetch`
- `before_verify_layer`（:1476）→ `submit_verify_layer_prefetch`

**零影响保证**：
- 旧类 `PrefetchRuntime` 与所有现有方法体**不改一行**。
- `model_runner` 仅 `__init__` 增加工厂分支；`kind != "predictive"` 时构造的对象、走的路径与现状完全相同。
- 新 config 默认 `"legacy"`，现有部署行为不变。

**待确认**（见 D.6）：复用 `"draft_segment_indexed"` mode 字符串以触发 gate 是否会引入语义混淆；若介意，替代做法是在那 4 个 gate 处加 `kind == "predictive"` 的显式分支（更清晰，但需在 4 处各加一行 guard，旧分支仍保持原样）。

### C.2 Expert cache 新增查询（additive，不改现有方法）

需要一个新方法：给定 segment（layer）区间 + 数量 budget，返回该区间内 **未缓存** 且 **ground-truth 频率最高** 的 expert id（阶段 1 用）。

- 与现有 `select_victim_slot`（选最低频已缓存）相反，是选最高频未缓存。
- 实现为 cache 上的新方法，不触碰现有 `mark_access` / `select_victim_slot` 语义。

### C.3 频率统计改造（仅在新 kind 生效）

- 新 kind 下：`mark_access` 仅由 prefill/verify observe 触发；draft observe **不**写 `access_count`，只更新预取队列。
- 旧 kind 下：维持现状（draft 也写 access_count）。
- 注意：这要求解开 A.5 的耦合——在新 kind 中把"喂 LFU"与"喂队列"拆成两条独立路径。

### C.4 复用的现有基础设施

- Segment 边界：`_segment_id_for_layer`、`draft_prefetch_segment_size`。
- Verify 逐层 hook：`before_verify_layer` / `after_verify_layer` + `available_ms` 预算。
- 异步流水线：metadata offload worker、H2D transfer stream、`reserve_active_slot_for_prefetch*`。
- 队列清空：`begin_draft_iteration` / `end_draft_iteration` 的生命周期钩子。

### C.5 Profiling

沿用 `PrefetchStepMetrics`（覆盖率、stall），并新增：
- 阶段 1 冷启动预取命中率（预取的高频 miss 中有多少被本轮 verify 真正消费）。
- 阶段 2 预测命中率（draft 预测 vs verify ground-truth）。
- 队列纯 draft 来源后的候选质量对比（vs 旧混合队列）。

---

## Part D — 开放问题（下一轮讨论输入）

> 以下是我在整理时发现、需要你确认的点；我没有擅自改你的设计，先标出来。

### D.1 阶段 2 的索引语义 — 已澄清（intra-step + 安全规则）

结论（见 B.3）：约束是**本次 forward 内的安全性**，预取目标段为 `0…i-1`（保守模式），principle 是"已算完的段写入安全"。已据此修正 B.3/B.5。剩余细化项：
- **紧/保守模式的判定阈值**：如何判断某次预取"能保证在 segment i 计算窗口内完成"，从而启用紧模式（可预取 `i` 以外的任意段）？需要一个传输时间 vs segment 计算时间的在线估计（见 D.3）。
- **per-segment budget 分配**：在保守模式下，`0…i-1` 随 i 增大而变宽，预算如何在这些段间分配（偏向高 rank / 高频段？）。

### D.2 阶段 1 的 ground-truth 时效

首个 draft step 紧接 verify 之后，而 verify 的 metadata offload + observe 是**异步**的（A.4）。查询 cache 频率时，**刚结束的 verify 是否已计入 access_count**？
- 若未计入，阶段 1 看到的是"上上轮"的频率，需评估影响或在新 kind 下对 verify metadata 做同步 flush。

### D.3 "估计预取数量"的计算

阶段 1 的"segment 0 计算能掩盖的传输数量"如何估？
- 候选：`floor(EMA(segment0_compute_ms) / EMA(per_expert_transfer_ms))`，复用现有 verify-layer 的 `available_ms` 思路。

### D.4 阶段 2 队列更新与传输的时序（覆盖率问题，非正确性）

segment i-1 的 routing 要先 offload→collect→observe 才能进队列，而这是异步的。安全规则已保证正确性（即便最新预测没赶上也不会冲突），但**覆盖率**仍受影响：若 segment i-1 的预测还没进队列，segment i 时对 `i-1` 段的预取就只能依赖更旧的预测（上一 step 的同段）。需要量化这个 lag，并决定是否为新 kind 设计更紧的 draft metadata 路径（甚至 segment 边界处同步 flush）。

### D.5 与 rerouting / RankGuard 的交互

- 新方案下 cache 频率仅来自 ground-truth，LFU 驱逐信号更"干净"，但也更稀疏（draft 不再贡献）。是否影响驱逐质量？
- top-1/2 保护（§2.2 RankGuard）依赖 rank_scores（来自 verify）——与新方案的 ground-truth-only 频率天然一致，可直接复用。

### D.6 零影响接入 — 已确定（子类化 + 工厂），剩余一处待拍板

机制见 C.1（`PredictivePrefetchRuntime(PrefetchRuntime)` 子类化，`model_runner.__init__` 工厂选择）。剩余待拍板：
- **gate 触发方式二选一**：(a) 复用 `prefetch_runtime_mode="draft_segment_indexed"` 触发现有 gate（`model_runner` 零改动，但 mode 字符串语义略混）；(b) 在 4 个 gate 处加显式 `kind == "predictive"` guard（更清晰，但加 4 行）。倾向 (a)，除非介意语义。

### D.7 验证 verify 导向预取对 draft cache 的副作用（B.6.3）

需要量化 token 局部性带来的 draft 被动收益，并排除 draft 命中回退：
- 指标：开启/关闭新 prefetcher 下的 **draft per-layer 命中率**、**verify miss 覆盖率**、端到端 accept rate。
- 关键对照：draft 命中率是否因 verify 导向的预取/驱逐而**下降**（B.6.3 风险）。
- 若局部性收益成立，可考虑在 victim 选择中加入"近期 draft 使用过"的轻量保护，避免驱逐 draft 仍需的专家（与 B.7 / D.9 相关）。

### D.8 阶段 1 仅取 `segment n-1` 的保守性（B.3 阶段 1 取舍）

仅预取 `n-1` 一段保证安全，但若"估计预取量 > `n-1` miss 数"，segment 0 的计算窗口未被充分利用。
- 选项 1：接受保守（安全优先）。
- 选项 2：允许溢出到 `n-2`（甚至 `n-Δ`），用一点安全裕度换取吞吐——需评估 `n-2` 在 segment 0 计算窗口内被读到的概率（取决于各 segment 计算耗时分布）。
- 需要 segment 计算耗时的 EMA 数据来判断溢出是否安全（与 D.3 共用估计器）。

### D.9 单轮保护是否覆盖 demand-load — 已查证并定论（见 E.7-C）

结论：verify 的 cache_fill 走**独立** victim 选择器、不经 prefetcher，且**每层独立 cache** 天然隔离了 cache_fill 与跨层预取。采用 **Impl A**（round 保护只在 prefetcher 内，cache_fill 不感知）；无需改 `placement.py`。详见 E.7-C。

---

## Part E — 保守模式 v1 实现计划（已确认）

### E.0 已确认的设计决策

| # | 决策点 | 选择 | 影响 |
|---|--------|------|------|
| 1 | 保守模式范围 | **v1 全量保守模式** | segment 0 预取 segment n-1；segment i（i≥1）预取 segment `0…i-1`。**含阶段 1 冷启动**（见决策 5、E.9）。 |
| 2 | B.8 队列打分 | **score_sum 代理** | 复用聚合元数据的 `score_sum`（routing 权重衰减累加）+ `activation_count` 排序，零额外开销。显式 topk-rank 留作后续（E.6）。 |
| 3 | verify 预取 | **启用，但更保守** | 队列生命周期改为 draft→verify 存活、下轮 begin 才清空；override `submit_verify_layer_prefetch` 从 draft 队列取候选。**预取窗口尽量限制在 attention 计算期间**，为 MoE 段的 demand-load 让出通信资源（本轮 point 1）。见 B.4 / E.13。 |
| 4 | B.7 保护范围 | **draft+verify 的所有预取驱逐**；rankguard 保护与单轮保护**叠加**（不绕过 rankguard，E.7-A）；`on_verify_layer_start(i)` 释放第 i 层（hygiene，非正确性必需，E.7-C） | 见 E.7 的 cache_fill 隔离分析 |
| 5 | 阶段 1 冷启动 | **v1 做，best-effort 重叠** | 上一轮 prefill/verify 更新完即可启动；prefill/verify/segment-graph replay 一完成就开下一段计算，metadata 卸载 / cache 更新 / prefetch 等非 GPU 同步操作全部在 GPU 计算时并行尽力而为。eager LUT 下其 intra-forward 可见性为 best-effort（见 E.9）。 |
| 6 | GPU LUT 更新 | **v1 不动（保持现有 eager）** | 批量更新移入后续（E.6 / E.8）。v1 沿用现有 `commit_*` 的 eager 逐 expert 写，不新增 LUT-apply 集成点（本轮 point 2、point 4）。 |
| 7 | 紧模式 | **不实现** | 紧模式会导致 LUT 频繁更新；保守模式天然规避。 |
| 8 | 动态 K 兼容 | **必须兼容** | K 后续将动态化（每步 draft forward 后决定是否 verify）。所有设计不得依赖固定 K。见 E.10。 |
| 9 | 关键路径开销 | **< 3ms / forward** | 任何实现不得在关键路径增加 ≥3ms 开销。 |
| 10 | victim 选择/submit 位置 | **全程计算重叠、不在主 forward 关键路径** | draft 段预取本就在 worker 线程；verify 预取在 `before_verify_layer`（主线程）但须 **sync-free**，借异步 GPU 让 CPU 跑在 GPU 前面实现重叠（本轮 point 3）。见 E.11。 |

### E.1 关键实现洞察（决定了最小改动面）

1. **Draft 预取无需 override**：现有 `submit_draft_segment_indexed_prefetch` 已是"每段算完后预取该段"——结构上即保守模式安全规则，且迭代 `(long_term_segment_index, draft_segment_index)` 两个索引。只要不给 long_term 喂 prefill/verify，它**天然 draft-only**。
2. **B.7 保护完全塞进两个 victim 选择器**：`_select_publish_slot_cpu`（draft 段预取用）和 `_select_publish_slot`（verify 预取用）都已拿到 incoming 的 `layer_idx`/`expert_idx`。在其中①记录 `_round_loaded`②跳过受保护 victim，即可同时完成"记录"和"保护"，**无需 override 150 行的 submit 方法**。
3. **数据源分离靠 observe override + 空 long_term**：observe_draft 只喂 draft 队列、observe_prefill/verify 只写 cache 频率；long_term 永不被喂 → draft submit 自动 draft-only。

### E.2 改动清单（最小集）

**使用说明 —— 单参数切换新旧实现**：

| `prefetch_runtime_kind` | 效果 |
|---|---|
| `"legacy"`（默认） | 原 `PrefetchRuntime`，行为逐字节不变 |
| `"predictive"` | 新 `PredictivePrefetchRuntime`（保守模式） |

```python
# 启用新实现
config = Config(..., prefetch_runtime_kind="predictive")
# 切回原实现：改回 "legacy" 或删除该项（默认即 legacy）
```

- **只需这一个参数**：工厂（`model_runner.__init__`）按它选类；predictive 时 `__post_init__` 自动强制 `prefetch_runtime_mode="draft_segment_indexed"`（无需手动配）。
- **前提**：预取本身已启用，即 `spec_enable_prefetch=True` 且 `inference_mode=="spec"`（任何预取的共同前提，非 predictive 特有）。
- **可选调参**（均有默认值，不设也能跑）：`prefetch_verify_attention_ratio=0.3`（E.13，verify 预取 attention 窗口占比）、`predictive_phase1_budget=4`（E.9，阶段 1 冷启动预取专家数）。

**配置改动（`config.py`）** — 已落地：
- 新增 `prefetch_runtime_kind: str = "legacy"`（取值 `legacy`|`predictive`）。
- 新增 `prefetch_verify_attention_ratio: float = 0.3`、`predictive_phase1_budget: int = 4`。
- `__post_init__`：predictive 时强制 `prefetch_runtime_mode = "draft_segment_indexed"`，使 model_runner 现有 gate 触发；并校验新参数取值。

**新增类（`prefetcher.py`）** — `PredictivePrefetchRuntime(PrefetchRuntime)`：

| override / 新增 | 行为 |
|------|------|
| `_segment_indexed_enabled()` | 恒 `True` |
| `observe_prefill` | 仅 `mark_access`（cache 频率），不喂队列；忽略 legacy `prefetch_use_*` 开关 |
| `observe_verify` | 仅 `mark_access` + `_update_rank_guard_scores`，不喂队列 |
| `observe_draft` | 仅更新 `draft_segment_index`（严格按 `_active_draft_iteration_steps` 门控），**不** `mark_access` |
| `begin_draft_iteration` | 轮起（`not _draft_iteration_open`）时 `clear()` draft 队列 + `_round_loaded`；置 open、记录 step；**置 `_phase1_pending=True`**（不在此提交阶段 1，因其在 drain 之前，E.9.1） |
| `maybe_submit_phase1`（新增） | drain 之后被调；若 `_phase1_pending` 则 `submit_phase1_prefetch` 并清标志（每轮一次，E.9.1） |
| `end_draft_iteration` | **不清 draft 队列**（留给 verify）；仅 `_active_draft_iteration_steps.clear()` + open=False |
| `submit_verify_layer_prefetch` | 从 `draft_segment_index`（target 段、过滤 `layer==target`）取候选，按 `available_ms` 预算提交；预算尽量收敛到 attention 窗口（决策 3 / E.13）；victim 走受保护选择器；**sync-free** |
| `_select_publish_slot_cpu` / `_select_publish_slot` | 委托 `_select_protected_victim`：记录 incoming 到 `_round_loaded`；victim 跳过谓词 = `round_protected(_round_loaded) OR rankguard.is_protected`（**叠加**，不绕过 rankguard）；度量按 strategy 选 LFU(access_count)/LRU(last_access_step)；全保护安全阀 → 无视两种保护选最低频（决策 4 + E.7-A/B） |
| `submit_phase1_prefetch`（新增） | 阶段 1 冷启动：查 cache ground-truth 频率取 `segment n-1` 高频 miss，**non-deferred** reserve（避免 buffer 竞争，E.9.2），async 提交到 transfer_stream |
| `on_verify_layer_start(layer_idx)` | 释放 `_round_loaded[layer_idx]`（hygiene） |

**集成（`model_runner.py`）** — **3 处**，对 legacy 均零影响（工厂分支 + guarded getattr）：
- `__init__` 工厂：`PrefetchCls = PredictivePrefetchRuntime if kind=="predictive" else PrefetchRuntime`。
- `before_verify_layer`：guarded 调用 `getattr(prefetch_runtime, "on_verify_layer_start", None)`，legacy 无此方法 → no-op。
- `run_draft`（drain 之后、`self.run` 之前）：guarded 调用 `getattr(prefetch_runtime, "maybe_submit_phase1", None)`，触发阶段 1（E.9.1），legacy 无此方法 → no-op。
- v1 **不新增 LUT-apply 集成点**（决策 6）；预取可见性沿用现有 `commit_*`/`publish_*` 的 eager 写。

### E.3 生命周期（v1）

```
spec_step:
  run_draft × K (动态 K, 见 E.10):
   每 draft step:
     begin_draft_iteration(sx)   → 轮起(首步): clear draft 队列 + _round_loaded; 置 _phase1_pending
     drain_direct_active_ready   → 同步屏障(commit 上一步在途预取)
     maybe_submit_phase1(sx)     → 首步: 阶段1 non-deferred 预取 segment n-1 (与随后 forward 重叠, E.9)
     self.run() = segment 0..n-1 计算 (此时 phase-1 H2D 并行):
        每段算完(worker 线程): observe_draft 累积 draft 队列(只此轮)
                              submit_draft_segment_indexed_prefetch (阶段2: 预取已算完段 0..i-1)
                                 victim 选择器: 记录 _round_loaded[layer]+=incoming, 跳过 (round OR rankguard) 保护
        预取可见性: 现有 commit_*/publish_* eager 写 LUT (v1 不动 LUT, 决策 6)
  end_draft_iteration        → 仅置 open=False；draft 队列与 _round_loaded 保留
  wait_prefetch_for_verify   → publish/wait
  run_verify (逐层 i):
     before_verify_layer(i):
        on_verify_layer_start(i)      → 释放 _round_loaded[i] (hygiene; cache_fill 本不感知, E.7-C)
        submit_verify_layer_prefetch(target=i+1)  → 从 draft 队列取 i+1 候选, victim 受保护
  [next spec_step] begin_draft_iteration → clear 队列 + _round_loaded（上一轮残留在此统一清）
```

### E.4 最小开销要点

- victim 选择器：遍历 `cache.slot_to_expert`（Python list，长度 = num_slots，几十量级）+ `_round_loaded[layer]` 集合查表 + rankguard `is_protected`（dict/list 查表），O(num_slots)，无张量操作、无 snapshot 拷贝。
- `_round_loaded`：`defaultdict(set)`，每轮 `clear()` + 每层 `pop()`，无累积增长。
- observe 数据分离：prefill/verify 走 `observe_runtime_meta(update_global_queue=False, segment_index=None)`（仅 mark_access）；draft 直接调 `draft_segment_index.update_from_runtime_meta`，均复用现有聚合路径，无新增张量计算。
- victim 选择/submit 全程 **sync-free**：draft 段预取在 worker 线程（天然离开关键路径）；verify 预取在主线程但不触发 GPU 同步，CPU 跑在异步 GPU 之前 → 与计算重叠（决策 10）。
- 无新增 CUDA 同步、无新增 stream（阶段 1/预取传输复用 transfer_stream，best-effort 与计算重叠）；v1 LUT 沿用现有 eager 写，不新增 LUT 操作。

### E.5 v1 验证点

- 正确性：legacy（默认 kind）路径逐字节不变（工厂分支 + 全 guarded hook 之外无改动）。
- predictive 开启：阶段 1（segment n-1 冷启动）+ 阶段 2（已算完段）预取命中、verify 层预取从 draft 队列产出候选、`_round_loaded`+rankguard 叠加防抖动生效（对照 thrash 计数）。
- access_count 仅含 prefill+verify ground-truth（draft 无贡献）。
- 关键路径净增 < 3ms/forward（E.11）；动态 K 下生命周期正确（E.10）。

### E.6 后续设计（已规划，暂不实现）

| 项 | 设计要点 | 依赖/触发 |
|----|---------|----------|
| **GPU LUT 批量更新**（决策 6 移此） | 把现有逐 expert 标量 GPU 写合并为受控同步点的单次散写（`index_put_` / staging-LUT 整体 copy），降低小 kernel 启动开销。详见 E.8 | 仅当实测 eager LUT 超 3ms 预算时启用 |
| 显式 topk-rank 打分（B.8 原式） | `priority += w × rank_factor(r)`，需原始非聚合 `selected_experts` 按列号取 rank | 走更重 metadata 路径，先评估 score_sum 代理效果再决定 |
| 紧模式（B.3 紧模式） | **明确放弃**（决策 7）：会导致 LUT 频繁更新 | — |
| 阶段 1 intra-forward 可见性强化 | 在 segment n-1 计算前确保阶段 1 预取 LUT 可见（需 LUT 批量更新 + segment-graph 边界 apply） | 依赖 GPU LUT 批量更新；v1 退化为 best-effort（E.9） |

### E.7 已查证的代码事实与结论

- **A. rankguard 实现方式（已查证 `cache_strategy.py`）**：`LFURankGuardStrategy` 不是"加权"，而是 victim 选择时**直接跳过** `is_protected(layer, expert)`（rank_score ≥ threshold 的 top-1/2），全保护才回退纯 LFU。
  → **结论**：受保护 victim 选择器把**单轮保护**与 **rankguard 保护**两个 skip 谓词**叠加**（`round OR rankguard`），不绕过 rankguard（修正先前误判）。LRU 配置则用 last_access_step。
- **B. 安全阀（已确认）**：某层所有非空 slot 都被保护时，回退"无视两种保护、选最低频/最久未用"驱逐，**保证预取推进**（决策 4 of 本轮）。
- **C. demand-load 路径（已查证 `placement.py`）**：verify 的 `apply_verify_cache_fill_policy` 有**独立的** `_select_verify_cache_fill_slot`，**不经 prefetcher 的 victim 选择器、不看 `_round_loaded`**；其 victim = 跳过 `active_expert_ids`（当前层需要的）+ 跳过 pending + 按 last_access。
  → **关键推论（per-layer cache 隔离）**：每层独立 `LayerExpertCache`。cache_fill 在第 j 层只动第 j 层 cache；prefetcher 为第 L≠j 层预取的 round-loaded expert 在不同 cache 里，**cache_fill 碰不到**。同层 L 内，round-loaded 要么是 verify 真需要的（在 `active_expert_ids`，cache_fill 本就跳过），要么是误预测的（本就该释放）。
  → **结论：采用 Impl A** —— round 保护**只**活在 prefetcher victim 选择器内，cache_fill 不感知它；**无需改 `placement.py`**（对 legacy 零影响）。`on_verify_layer_start(i)` 释放对正确性非必需（隔离已保证），保留为廉价 hygiene。

### E.8 GPU LUT 批量更新 —— **v1 不实现（移入 E.6 后续）**

> 决策 6 / 本轮 point 2、point 4：v1 **不改 LUT 更新机制**，沿用现有 eager 写；以下为后续实现保留的设计。

**现状（已查证 `cache.py`）**：GPU LUT（`expert_to_slot_lut` / `slot_to_expert_lut` / `cached_expert_mask`，均为 GPU 张量）由 `put_to_slot` / `commit_active_prefetch` / `commit_deferred_active_prefetch` 做**逐元素标量 GPU 写**（eager）：每 commit 一个 expert ≈ 3 个标量张量赋值（清除被驱逐项 + 写入新项）。频繁 commit → 频繁小 kernel 启动。

**后续批量方案（暂不实现）**：
- prefetch 传输完成后不立即写 GPU LUT，把 slot→expert 变更累积到 CPU 侧 pending 列表，在**受控同步点**一次性批量写（`index_put_` / staging-LUT 整体 copy）。
- 受控点候选：阶段 1 → segment n-1 计算前；阶段 2 → draft 结束 / verify 前。
- 正确性约束：LUT 必须在消费该 expert 的 kernel 之前可见。
- 触发条件：仅当实测 eager LUT 开销逼近/超过 3ms 预算时才启用（先测后定）。

### E.9 阶段 1 冷启动（决策 5，v1 纳入，eager LUT 下 best-effort）

- **触发**：上一轮 prefill/verify 更新完 cache 频率后、本轮首个 draft step。
- **数据**：查 cache `access_count`（仅 prefill/verify ground-truth）取 `segment n-1` 内 uncached 且频率最高的 expert（budget = `predictive_phase1_budget`，默认 4）。
- **重叠原则**：阶段 1 的 cache-freq 扫描（CPU）+ async H2D（transfer_stream）**与 segment 0 计算并行**，不插入 GPU 同步、不阻塞。
- **扫描开销**：segment n-1 约 `segment_size` 层 × ≤128 expert，保持 top-budget，sub-ms CPU。

**E.9.1 提交时机 —— 必须在 drain 之后、forward 之前**

`run_draft` 主线程顺序（model_runner，行号示意）：

```
begin_draft_iteration(step_id)        # 标记轮边界 (在 drain 之前!)
drain_direct_active_ready(step_id)    # 同步屏障: synchronize() 等所有在途 direct-active 传输 + commit
arm(...)                              # 准备 metadata 捕获
self.run(...)                         # forward = segment 0..n-1 计算
```

- ❌ **不能在 `begin_draft_iteration` 提交阶段 1**：它在 drain **之前**，紧随的 drain 会 `synchronize()` 阻塞等待阶段 1 的 H2D（budget×~1.18ms），既无法与 segment 0 重叠，又违反非阻塞/<3ms。
- ✅ **正确位置：drain 之后、`self.run` 之前**。本步 drain 不碰它 → async H2D 与 segment 0..n-1 计算并行 → 下一步 drain / verify 时再 commit。
- **实现（3 函数分工）**：
  - `begin_draft_iteration`：轮起时只置 `self._phase1_pending = True`（不提交）。
  - `maybe_submit_phase1(step_id)`（新，薄包装）：每 draft step 在 drain 后被调一次；若 `_phase1_pending` 则 `submit_phase1_prefetch` 并清标志（每轮一次）。
  - `submit_phase1_prefetch(step_id)`：扫频率 + reserve + async H2D。
- **集成点**：model_runner `run_draft` 在 drain 之后、`self.run` 之前加一行 guarded 调用 `getattr(prefetch_runtime, "maybe_submit_phase1", None)`（legacy 无此方法 → no-op）。→ model_runner 集成点共 **3 处**。

**E.9.2 reserve 模式 —— 必须 non-deferred（避免 buffer 竞争）**

阶段 1 写的是 **segment n-1 的 active slot**，而 segment n-1 **本 forward 还要被计算**：

- ❌ **deferred reserve 不安全**：deferred 不清 LUT，slot V 仍指向旧专家 W；但 `begin_async_put_to_active` 已在 transfer_stream 覆盖 `buffer[V]`。本 forward 算 segment n-1 若读 slot V（取 W 权重）→ 与传输**数据竞争**。（deferred 只对"已算完段"安全，故现有阶段 2 段预取用它。）
- ✅ **non-deferred reserve**（`reserve_active_slot_for_prefetch`，同 verify-layer 预取）：reserve 时立刻把 slot V 从 LUT 摘除（置 -1）→ 本 forward segment n-1 **无专家映射到 V → 无人读 `buffer[V]` → 无竞争**；旧专家 W 立即驱逐（若 n-1 需要则 miss，cache_fill/reroute 兜底）；E 在下一步 drain 时 commit 可见。
- ticket `source` 用非 `"draft_segment_indexed"` 值（路由到 `commit_active_prefetch`，与 non-deferred 配对）。
- **代价**：可能提前驱逐一个低频 W（budget 小、选最低频，影响有限）。

**E.9.3 LUT 可见性（v1 eager，best-effort）**：阶段 1 预取经现有 `publish_*`/`commit_*` eager 写 LUT；**v1 不保证 segment n-1 在本 forward 内即可见**——通常在下一步 drain / verify 前 commit，退化为"为 verify / 下一 draft step 预热"（与阶段 2 同价值）。intra-forward 强可见性留作后续（E.6，依赖 LUT 批量 + segment 边界 apply）。

### E.10 动态 K 兼容性（决策 8）

- 当前 v1 生命周期**已 K-无关**：以 `begin_draft_iteration`（每 draft step）/ `end_draft_iteration`（verify 前）事件为界，不假设固定 K。
- 队列在 round 内持续累积、verify 前不清空 → verify 可在**任意** draft step 后启动并立即取到当前 draft 预测。
- `_round_loaded` 同理按事件清理。
- 实现红线：禁止任何"预知 K"或"按固定 K 预分配/调度"的逻辑。

### E.11 关键路径开销预算（决策 9 + 10，< 3ms/forward）

| 操作 | 位置 | 预估 | 控制手段 |
|------|------|------|---------|
| draft 段预取 victim 选择/submit | **worker 线程** | O(num_slots) Python | 天然离开主 forward 关键路径 |
| observe 数据分离 | worker 线程 | 复用聚合路径 | 不在主 forward 关键路径 |
| verify 预取 victim 选择/submit | `before_verify_layer`（主线程） | O(num_slots) Python | **sync-free**，CPU 跑在异步 GPU 前 → 与计算重叠（决策 10） |
| 阶段 1 freq 扫描 | 首 draft forward 启动 | sub-ms CPU，与计算重叠 | best-effort、bounded budget、sync-free |
| GPU LUT 写 | commit（**v1 沿用 eager**） | 逐 expert 标量，需实测 | 超预算 → 启用 E.6/E.8 批量 |

- **决策 10 红线**：predictive 在主 forward 路径（含 `before_verify_layer`）的所有新增 CPU 工作必须 **sync-free**，使其与异步 GPU 计算重叠；不得引入 `cuda.synchronize` / `.item()` / `.cpu()` 等强制同步。
- 验证：开启 predictive 后对比 per-forward wall time，确保关键路径净增 < 3ms。

### E.12 仍需讨论的点 —— 本轮已清零

- **A.（已定）**：v1 不动 LUT 更新 → 无 LUT-apply 集成点。model_runner 改动 **3 处**（工厂 + `on_verify_layer_start` + `maybe_submit_phase1`），全 guarded、legacy 零影响。阶段 1 须在 drain 后提交（E.9.1），故需 `run_draft` 内的第 3 个 hook。
- **B.（已定，本轮 point 4）**：LUT 选型问题搁置——v1 保持 eager，先测开销；超预算才上批量（E.6/E.8）。
- **C.（已定，本轮 point 2/4）**：阶段 1 intra-forward 可见性依赖 LUT 批量 + segment 边界，v1 不做 → 阶段 1 在 v1 退化为 best-effort（E.9）。是否恒用 segment graph 不再阻塞 v1。
- **D. demand-load 资源竞争（已定，本轮 point 1）**：见 E.13。

### E.13 verify 预取的 attention 窗口约束（本轮 point 1）

**动机**：verify 每层 = attention 段 + MoE 段；demand-load（cache_fill）发生在 **MoE 段**并占用 PCIe。verify 预取的 H2D 若与 MoE 段的 demand-load 同时进行，会**争抢 PCIe 带宽**，拖慢关键路径上的 demand-load。

**目标**：verify 预取的传输窗口**尽量限制在 attention 段计算期间**，MoE 段不发起预取传输，把通信资源让给 demand-load。

**实现（v1 = 方案 A，保守 fraction 近似）**：
- 把 `submit_verify_layer_prefetch` 的时间预算从"整层 compute 的 EMA × safety_ratio"再乘一个 **attention 占比系数** `prefetch_verify_attention_ratio`（默认 ≈0.3），即 `available_ms = EMA(layer_compute_ms) × safety_ratio × attention_ratio`。预取量自然受限、传输更可能落在 attention 段内。
- 触发时机沿用现有 `before_verify_layer`（层计算前，近似 attention 起点）；不在 MoE 段追加提交。
- 零侵入：不新增计时 hook、不触碰 model 代码。

**后续（方案 B，精确分段）**：新增 attention/MoE 分段计时 hook，用真实 attention 段 EMA 作预算。更准但需在 forward 内加计时点。留作后续。

### E.14 测试计划

> 目标：覆盖 legacy 零影响 + predictive 各组件正确性 + 开销红线。建议落在 `tests/`（如 `tests/test_predictive_prefetch.py`），多数可用轻量 fake（小 `LayerExpertCache` + 假 `cpu_expert_pool` + 假 metadata）单测，无需真实模型。

**A. 配置与零影响**
1. `test_config_predictive_forces_segment_indexed`：`prefetch_runtime_kind="predictive"` → `__post_init__` 后 `prefetch_runtime_mode=="draft_segment_indexed"`；非法 kind / ratio>1 / 负 budget 触发 assert。
2. `test_factory_selects_class`：kind=predictive → `model_runner.prefetch_runtime` 是 `PredictivePrefetchRuntime`；kind=legacy → `PrefetchRuntime`（精确类型，非子类）。
3. `test_legacy_unchanged`：kind=legacy 时 `observe_*` / 队列 / victim 行为与改动前一致（黄金值或 mock 调用序列对比）。

**B. 数据源分离（B.2）**
4. `test_observe_prefill_verify_only_mark_access`：喂 prefill/verify metadata → `cache.access_count` 增加，draft 队列（`draft_segment_index`/`global_queue`）**为空**。
5. `test_observe_draft_only_queue`：喂 draft metadata（step 在 `_active_draft_iteration_steps`）→ `draft_segment_index` 有候选，`access_count` **不变**。
6. `test_observe_draft_stale_dropped`：step 不在 active steps → 队列不更新，`predictive_draft_stale_observe_count` +1。

**C. 单轮保护 + rankguard 叠加（B.7 / E.7-A/B）**
7. `test_round_protect_skips_loaded`：先经 victim 选择器载入 expert E（进 `_round_loaded`）→ 后续 victim 选择跳过 E 的 slot。
8. `test_rankguard_composed`：`cache_strategy=lfu_rankguard` + 某 expert `is_protected` → victim 跳过它（即使最低频）。
9. `test_safety_valve_all_protected`：所有非空 slot 都被 (round∪rankguard) 保护 → 返回最低频 slot（不返回 None，预取仍推进）。
10. `test_empty_slot_preferred`：存在空 slot → 直接返回空 slot，不驱逐。
11. `test_on_verify_layer_start_releases`：`on_verify_layer_start(i)` 后 `_round_loaded[i]` 清空。

**D. 队列生命周期（E.3）**
12. `test_queue_persists_through_verify`：begin→K×draft 累积→end_draft_iteration 后队列**仍非空**（draft→verify 存活）；下一轮 begin 后清空。
13. `test_round_loaded_reset_next_round`：`_round_loaded` 在下一轮 begin 清空、verify 期间保留。

**E. 阶段 1 冷启动（E.9）**
14. `test_phase1_fires_once_per_round`：begin 置 `_phase1_pending`；`maybe_submit_phase1` 首调提交、再调 no-op；下一轮重新 arm。
15. `test_phase1_targets_segment_n_minus_1`：仅对最后一个 segment 的层、uncached、按 `access_count` 降序、≤budget 提交。
16. `test_phase1_non_deferred_reserve`：phase-1 ticket `source=="predictive_phase1"`、走 `reserve_active_slot_for_prefetch`（非 deferred）、publish 经 `commit_active_prefetch`（mock 断言调用）。
17. `test_phase1_after_drain_not_in_begin`：`begin_draft_iteration` 不提交预取（inflight 不变）；`maybe_submit_phase1` 才提交。

**F. verify 预取（B.4 / E.13）**
18. `test_verify_prefetch_from_draft_queue`：候选来自 `draft_segment_index`（过滤 `layer==target`），`global_queue` 空也能产出。
19. `test_verify_prefetch_attention_ratio`：`available_ms` 被乘以 `prefetch_verify_attention_ratio`；ratio→0 时不提交。

**G. 动态 K 兼容（E.10）**
20. `test_dynamic_k_lifecycle`：K=1 与 K=5 两种，begin/end 事件驱动的队列/保护行为一致，无固定 K 假设。

**H. 开销红线（E.11，决策 9/10）**
21. `test_victim_selection_no_sync`（可选，需 GPU）：predictive victim/submit 路径不触发 `cuda.synchronize`/`.item()`（用 patch 计数或 stream 探针）。
22. `test_overhead_budget`（集成，需 GPU/模型）：predictive vs legacy per-forward wall time 净增 < 3ms。标注为 slow/opt-in。

---

## 附：备选方案（来自 §5 讨论，暂不纳入主线）

**基于层间相似性的 verify 预取**：verify 第 l 层完成后，用 l 层精确路由 + 层间 Jaccard 相关性（≈0.442）预测 l+1…l+Δ 层专家。
- 现状：尚未研究层间相似性预取的准确度，作为备选保留。
- 与本方案关系：本方案 verify 预取依赖 **draft 预测队列**（B.4），而非层间相似性；两者可作为 verify 预取的不同信号源对比评估。
