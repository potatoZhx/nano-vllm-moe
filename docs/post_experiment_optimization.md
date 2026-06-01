# 基于验证实验的算法优化设计

**Project:** nano-vllm-moe  
**Target Model:** Qwen3-30B-A3B (N=128, k=8, 48 MoE layers)  
**Hardware:** Single GPU + Host DRAM, PCIe 4.0 ×16  
**Date:** 2026-05-28  
**前置文档:** `system_design_report.md`, `experiment_report.md`

---

## 1. 实验结论综述与修正

### 1.1 实验结果总表

| 实验 | 核心发现 | 可信度 | 设计决策 |
|------|---------|--------|---------|
| E1: 双目标缓存 | HitScore-SubstScore ρ=0.78；LFU 全面优于 Joint 策略 | **高** | 放弃 EvictCost-aware 策略 |
| E2: Top-1/2 保护 | Cache pin 无效（LFU 天然覆盖）；reroute 保护仅 +0.2pp | **需修正** | 见 §2 详细分析 |
| E3: 动态 K | 简化 T_cycle 模型低估 T_stall 8 倍，K*=12 不可信 | **低** | 用 E4 覆盖率信号替代 |
| E4: 预取覆盖 | r=0.25 时 K*=1（覆盖率太低）；覆盖率是有效的 Dynamic K 信号 | **中** | 见 §4 修正分析 |
| E5: α 预测 | 解析模型 RMSE=0.03-0.28；MLP 无显著优势 | **高** | 初始使用解析模型 |

### 1.2 需要修正的两个关键结论

**修正 1: Top-1/2 保护实验的局限性**

E2 报告中 reroute 保护仅 +0.2pp 的原因并非保护机制本身无效，而是实验存在一个设计问题：保护机制将 top-1 专家恢复到 routing 列表中，但由于该专家 uncached，权重仍被置零。实际上保护的是"概率质量重分配方式"，而非"top-1 专家的计算"。这个机制本身是正确的，但收益度量取决于权重分配改变引起的 logit 偏移，在 SkipAll 的框架内天然受限。

然而，从系统设计角度，需要关注的核心问题是：**LFU 是否在所有 cache ratio 下都能保证 top-1/2 专家驻留？** E2 的数据显示 r=0.25 时 cache pin 无效（∵ LFU 已覆盖），但这是基于离线校准的静态 LFU 初始化。在线 LFU/LRU 在实际运行中由于 access pattern 漂移，可能在某些时段驱逐 top-1/2 专家。已有实验（v2 rerouting eval 和 routing analysis）充分证明了 top-1/2 专家对生成质量的决定性作用，因此只要 LFU 不能**保证**在所有运行时态下保护 top-1/2，就需要缓存层面的保护机制。

**修正 2: 预取覆盖率模拟的简化程度**

E4 使用的预取模型（FIFO 队列、每步 m 个专家、1-step 延迟）过度简化了实际系统。nano-vllm-moe 的预取分为 draft 阶段预取（`submit_draft_direct_active_prefetch`）和 verify 阶段预取（利用已完成 verify 层的路由信息预取后续层）。实际系统中还涉及 PCIe 传输流水线、pinned memory、inflight 管理等。E4 的定性结论（覆盖率是有效信号）成立，但定量结论（K*=1 at r=0.25）可能过于悲观，因为没有考虑 verify 阶段的层间预取。

---

## 2. 缓存策略优化

### 2.1 确认：LFU 作为基础驱逐策略

E1 以压倒性的证据表明，在 Qwen3-30B-A3B 上 HitScore 和 SubstScore 高度正相关（ρ=0.78），纯 LFU 在所有 cache ratio 下的 hit rate 均高于任何 Joint 策略。**放弃 EvictCost-aware 双目标策略**。

保留 LFU 的 access 统计数据来源设计不变：

```
mark_access() 数据来源 = flat_selected_original （= verify routing 预测）
```

这保证驱逐决策反映目标模型（verify）偏好而非 draft 的修改路由。

### 2.2 Top-1/2 缓存保护：LFU-RankGuard 策略

**动机**

虽然 E2 的静态分析表明 LFU 初始化已覆盖 top-1/2，但在线运行中 LFU 可能因以下原因驱逐 top-1/2 专家：

1. **分布漂移**：生成长序列时，topic 切换导致某些 top-1/2 专家的近期 access count 下降
2. **LRU 退化**：当系统使用 LRU（而非 LFU）时，上一轮 verify 未激活的 top-1/2 专家立即成为驱逐候选
3. **Prefetch 挤压**：draft 期间预取新专家需要腾出 slot，在 cache 满时必须驱逐某个已有专家

以上任何一种情况发生时，top-1/2 miss 会导致 verify 阶段关键概率质量丢失（top-1 通常承载 30-50% 权重），产生不成比例的质量损失。

**设计方案：LFU-RankGuard**

在现有 `LFUCacheStrategy.select_victim_slot()` 的基础上增加一个轻量保护层：

```python
class LFURankGuardStrategy(CacheStrategy):
    """LFU with top-1/2 eviction protection.
    
    Never evict an expert whose rank1_score + rank2_score exceeds protect_threshold,
    unless all non-protected experts have been evicted (safety valve).
    """
    def __init__(self, rank_scores: dict[int, dict[int, float]] | None = None,
                 protect_threshold: float = 0.10):
        """
        rank_scores: {layer_idx: {expert_id: score}} where score = 
            2 * rank1_freq + rank2_freq (calibration-time, EMA-updated online)
        protect_threshold: experts with score >= threshold are protected
        """
        self.rank_scores = rank_scores or {}
        self.protect_threshold = protect_threshold
    
    def select_victim_slot(self, snapshot, incoming_expert_idx, step_id):
        # Phase 1: try to evict among non-protected, lowest access count
        best_slot = None
        best_count = None
        all_protected = True
        
        for slot_idx, expert_idx in enumerate(snapshot.slot_to_expert_lut.tolist()):
            if expert_idx < 0:
                return slot_idx  # empty slot
            
            layer_scores = self.rank_scores.get(snapshot.layer_idx, {})
            is_protected = layer_scores.get(expert_idx, 0.0) >= self.protect_threshold
            
            if not is_protected:
                all_protected = False
                cnt = snapshot.access_count[expert_idx]
                if best_count is None or cnt < best_count:
                    best_count = cnt
                    best_slot = slot_idx
        
        if best_slot is not None:
            return best_slot
        
        # Phase 2: safety valve — all are protected, fall back to pure LFU
        if all_protected:
            return self._fallback_lfu(snapshot)
        
        return best_slot
    
    def _fallback_lfu(self, snapshot):
        best_slot = None
        best_count = None
        for slot_idx, expert_idx in enumerate(snapshot.slot_to_expert_lut.tolist()):
            if expert_idx < 0:
                return slot_idx
            cnt = snapshot.access_count[expert_idx]
            if best_count is None or cnt < best_count:
                best_count = cnt
                best_slot = slot_idx
        return best_slot
```

**rank_scores 的计算和维护**

- 离线校准：`rank_score(j) = 2 * rank1_freq(j) + rank2_freq(j)`
- 在线 EMA 更新：每轮 verify 后，根据实际 top-k routing 的 rank 信息更新：
  ```python
  rank_score[j] = α_ema * rank_score[j] + (1 - α_ema) * current_rank_score[j]
  ```
  其中 `α_ema = 0.95`，`current_rank_score` 来自本轮 verify 的 routing

**protect_threshold 选择**

根据 E2 校准数据：
- `threshold = 0.10`：约 95 个 expert 被保护（过多，cache 压力大）
- `threshold = 0.15`：约 42 个 expert 被保护
- `threshold = 0.20`：约 22 个 expert 被保护

推荐 `threshold = 0.15`：在 r=0.25 时（cache_size=32），约 42 个 protected expert 中只有一部分同时存在于 cache，不会阻碍正常驱逐。关键是保护 cache 中已有的 top-1/2 专家不被替换。

**开销**

- 内存：每层 128 个 float32 rank_score = 512 bytes × 48 layers ≈ 24 KB（可忽略）
- 计算：每次驱逐决策增加一次 `rank_scores.get()` 查表 ≈ 0.001ms
- EMA 更新：每轮 verify 后遍历 top-k routing ≈ 0.01ms

**部署位置**

修改 `nanovllm/scheduling/cache_strategy.py`：新增 `LFURankGuardStrategy` 类，通过 `create_cache_strategy("lfu_rankguard")` 创建。rank_scores 在 calibration 阶段由 `PrefetchRuntime` 初始化，在线通过 `observe_runtime_meta` 的 verify 路径更新。

### 2.3 验证方案

无需独立的离线实验——在实际系统中 A/B 对比：

1. 记录 baseline LFU 的 verify hit rate 和 top-1/2 miss 事件
2. 切换到 LFU-RankGuard，记录同样指标
3. 重点观察长序列生成（>200 tokens）中 top-1/2 miss 是否减少

---

## 3. Rerouting 算法优化

### 3.1 确认：Alg2_v2 作为首选 rerouting 算法

v2 实验数据（report §2.11）清楚表明 Alg2_v2 在 r=0.25 时比 SkipAll 高出 9.5pp（0.7309 vs 0.6364）。E3/E5 的 per-step α 数据进一步证实 Alg2_v2 在所有 cache ratio 下均优于或等于 SkipAll。

**推荐的 rerouting 选择策略**：

| 运行时 miss rate ρ | 算法 | 理由 |
|---|---|---|
| ρ < 0.25 | SkipAll（自动） | miss-rate gate 置零 → Alg2_v2 退化为 SkipAll |
| 0.25 ≤ ρ ≤ 0.50 | Alg2_v2 渐进激活 | gate 线性开启 bias |
| ρ > 0.50 | Alg2_v2 全开 | bias 最大化避免 miss |

这已经由 miss-rate gate 自动处理，无需额外逻辑。

### 3.2 Top-1/2 Rerouting 保护修正

E2 中 reroute 保护的 +0.2pp 收益看似微小，但其机制是正确的：在偏置后路由中恢复被挤出的 top-1（即便 uncached 也恢复到路由列表以影响权重分配）。实际收益小的原因是 SkipAll 框架下 uncached 专家权重被清零，保护仅影响 renormalize 后的比例分配。

**优化方向**：保留现有 reroute 级 top-1/2 保护（近零开销），但更重要的保护应在缓存层面（§2.2 LFU-RankGuard）。两层保护的职责划分：

- **缓存保护**（主动防御）：确保 top-1/2 高频专家尽可能驻留 GPU，从根本上避免 miss
- **Reroute 保护**（被动兜底）：当 miss 不可避免时，保证权重分配的公平性

### 3.3 当前 Alg2_v2 的实现映射

在 nano-vllm-moe 中，Alg2_v2 需要集成到 `nanovllm/expert/placement.py` 的 `build_draft_plan_gpu()` 路径中：

```
当前流程：
  build_draft_plan_gpu() 
    → _build_topc0_substitution_lut()  # round-robin 替换
    → MoEExecutionPlan(flat_selected_original=..., flat_selected_effective=...)

优化后流程：
  build_draft_plan_gpu()
    → _build_alg2v2_routing()          # 偏置后 top-k + SkipAll fallback
    → MoEExecutionPlan(flat_selected_original=原始路由, flat_selected_effective=偏置后路由)
```

`_build_alg2v2_routing()` 的输入：原始 router logits、cache mask、miss-rate gate 参数（离线标定）。输出：修改后的 (weights, indices)。整个操作在 GPU 上完成，开销约 0.1ms/layer。

---

## 4. Dynamic K 优化

### 4.1 问题重新定位

Dynamic K 的决策质量取决于 T_stall 估计的准确性，而 T_stall 高度依赖预取的实际表现，这在离线仿真中很难精确模拟。

### 4.2 新方案：基于离线 Profiling 的 Dynamic K

放弃试图在离线仿真中精确建模 T_stall 的思路，改为**在实际系统运行中收集 profiling 数据，基于历史统计做 Dynamic K 决策**。

**架构：DynamicKController**

```
DynamicKController
├── Level-0: 静态上界（配置参数，默认 K_max=8）
├── Level-1: 阈值拦截（always-on，≈0 开销）
│   ├── Signal A: CriticalMissRate > θ_crit → STOP
│   ├── Signal B: ρ_miss > θ_stop → STOP
│   └── Signal C: ρ_miss < θ_safe → CONTINUE
└── Level-2: Profiling-based 自适应（每 N 轮更新）
    ├── Throughput LUT[cache_regime][algo] → K_opt
    └── Online calibration via moving-window profiling
```

**Level-1 阈值信号设计**（保留自 system_design_report，经 E3 验证有效）

```python
def level1_decision(critical_miss_rate, miss_rate, step_k):
    if critical_miss_rate > 0.3:
        return "STOP"    # top-1/2 大面积 miss，继续 draft 无益
    if miss_rate > 0.6:
        return "STOP"    # 超过 60% miss → rerouting 质量不保
    if miss_rate < 0.15:
        return "CONTINUE"  # 几乎全命中，继续 draft 安全
    return "EVAL_LEVEL2"   # 中间区域，交 Level-2 判断
```

**Level-2 Profiling-Based 自适应（核心创新）**

核心思想：不预测 T_stall，而是直接从最近 W 轮的实际 speculative step 中测量有效吞吐，然后选择最优 K。

```python
class ProfilingDynamicKController:
    """
    Maintains a moving window of recent speculative step profiles.
    Periodically recalculates optimal K from actual measurements.
    """
    
    def __init__(self, k_max=8, window_size=50, update_interval=10):
        self.k_max = k_max
        self.window_size = window_size
        self.update_interval = update_interval
        
        # Ring buffer of recent step profiles
        self.profiles = deque(maxlen=window_size)
        self.current_k = k_max  # start conservatively
        self.step_count = 0
        
        # Cache regime detection
        self._miss_rate_ema = 0.0
        self._ema_alpha = 0.1
    
    def record_step(self, profile: StepProfile):
        """Called after each speculative step with actual measurements."""
        self.profiles.append(profile)
        self.step_count += 1
        
        # Update miss rate EMA
        self._miss_rate_ema = (self._ema_alpha * profile.avg_miss_rate 
                               + (1 - self._ema_alpha) * self._miss_rate_ema)
        
        # Periodic K optimization
        if self.step_count % self.update_interval == 0:
            self._update_optimal_k()
    
    def get_draft_steps(self) -> int:
        """Called before each draft phase to get recommended K."""
        return self.current_k
    
    def _update_optimal_k(self):
        if len(self.profiles) < 10:
            return
        
        # Group profiles by draft_steps (K) used
        by_k = defaultdict(list)
        for p in self.profiles:
            by_k[p.draft_steps].append(p)
        
        # Calculate effective throughput for each K
        throughput_by_k = {}
        for k, profs in by_k.items():
            # effective throughput = mean(accepted_tokens) / mean(t_cycle)
            mean_accepted = np.mean([p.accepted_tokens for p in profs])
            mean_cycle = np.mean([p.t_cycle_ms for p in profs])
            if mean_cycle > 0:
                throughput_by_k[k] = mean_accepted / mean_cycle
        
        if not throughput_by_k:
            return
        
        # Exploration: if some K values haven't been tried recently, try them
        untried = set(range(1, self.k_max + 1)) - set(throughput_by_k.keys())
        if untried and random.random() < 0.1:
            # 10% exploration probability
            self.current_k = random.choice(list(untried))
            return
        
        # Exploitation: pick best K
        self.current_k = max(throughput_by_k, key=throughput_by_k.get)
```

**StepProfile 数据结构**（从 spec_engine 收集）

```python
@dataclass
class StepProfile:
    draft_steps: int            # K used
    accepted_tokens: int        # actual accepted count
    t_cycle_ms: float          # total wall time: draft + rollback + verify + accept
    t_draft_ms: float          # draft phase wall time
    t_verify_ms: float         # verify phase wall time
    t_stall_ms: float          # PCIe stall (verify - pure GPU compute)
    avg_miss_rate: float       # average per-step miss rate across layers
    critical_miss_rate: float  # top-1/2 miss fraction
    prefetch_coverage: float   # fraction of verify experts that were prefetched
```

`spec_engine.py` 已经有 `_profile` 和 `_step_traces` 机制（见 `speculative_step()` 中的 timing 记录），只需要扩展记录更多字段。

### 4.3 Exploration-Exploitation 策略

纯 exploitation（总是选最优 K）的问题是：环境变化后（如 topic 切换导致 cache miss pattern 改变）无法发现新的最优 K。

**ε-greedy with decay**：

```python
exploration_rate = max(0.05, 0.2 * (0.99 ** step_count))
```

- 初始 20% 探索（快速覆盖 K=1..8）
- 衰减到 5% 稳态探索
- 探索时均匀随机选择 K ∈ [1, K_max]

**Regime-Adaptive 增强**：

当检测到 cache regime 发生显著变化时（miss_rate EMA 突变 > 0.1），重置 exploration_rate 到 0.2，快速重新标定最优 K：

```python
if abs(self._miss_rate_ema - self._last_regime_miss_rate) > 0.1:
    self._exploration_rate = 0.2  # reset exploration
    self._last_regime_miss_rate = self._miss_rate_ema
```

### 4.4 Prefetch Coverage 作为辅助信号

E4 验证了 PrefetchCoverage 是有效的 Dynamic K 信号，但简化模型的定量结论不够准确。在 profiling-based 方案中，PrefetchCoverage 可以作为 Level-1.5 的快速信号：

```python
def level1_5_coverage_check(self, current_step_k):
    """Quick coverage estimate from recent profiles."""
    recent = [p for p in self.profiles[-20:]]
    if not recent:
        return "CONTINUE"
    
    avg_coverage = np.mean([p.prefetch_coverage for p in recent])
    
    if avg_coverage < 0.3 and current_step_k > 2:
        return "REDUCE_K"  # coverage too low, shorten draft
    if avg_coverage > 0.8 and current_step_k < self.k_max:
        return "MAY_INCREASE_K"  # coverage good, can try longer
    return "CONTINUE"
```

### 4.5 部署路径

**Phase 0：集成 Level-1 阈值信号**

修改 `spec_engine.py` 的 `speculative_step()` 循环：

```python
# 现有代码：
for step in range(draft_steps):
    # ... draft forward ...

# 修改后：
for step in range(draft_steps):
    # ... draft forward ...
    
    # Level-1 check (near-zero overhead)
    if self.dynamic_k_controller.should_stop(step, step_miss_rate, step_crit_miss):
        break
```

`step_miss_rate` 和 `step_crit_miss` 可以从 `MoEExecutionPlan.flat_selected_original` 和 cache 状态直接计算。

**Phase 1：集成 Profiling Controller**

在 `speculative_step()` 返回前记录 `StepProfile`，传给 `ProfilingDynamicKController.record_step()`。下一轮 `_budget_draft_steps()` 从 controller 获取推荐 K：

```python
def _budget_draft_steps(self, seqs) -> int:
    limits = [self.max_draft_tokens]
    # ... existing budget logic ...
    
    # Dynamic K from profiling
    profiling_k = self.dynamic_k_controller.get_draft_steps()
    limits.append(profiling_k)
    
    return min(limits)
```

**Phase 2：Online α Calibration**

E5 验证的解析模型 α̂(k) = α₀·exp(-λ·E(k)) 可用于 Level-2 的边际决策。α₀ 和 λ 从 profiling 窗口中在线 EMA 拟合：

```python
def _online_fit_alpha(self):
    """Fit α₀, λ from recent accept/reject data."""
    if len(self._alpha_samples) < 20:
        return
    
    E_vals = np.array([s.cumulative_error for s in self._alpha_samples])
    alpha_vals = np.array([s.actual_alpha for s in self._alpha_samples])
    
    try:
        popt, _ = curve_fit(lambda x, a0, lam: a0 * np.exp(-lam * x),
                            E_vals, alpha_vals,
                            p0=[self._alpha0, self._lambda],
                            bounds=([0, 0], [1, 50]))
        self._alpha0, self._lambda = float(popt[0]), float(popt[1])
    except:
        pass  # keep previous parameters
```

---

## 5. 预取调度优化

### 5.1 当前预取架构梳理

nano-vllm-moe 的预取系统由 `PrefetchRuntime`（`prefetcher.py`）和 `ModelRunner`（`model_runner.py`）协作完成，具备完整的两阶段异步预取能力。

#### 5.1.1 信息源

预取候选队列的信息来源有三个，分别通过 `observe_*()` 方法注入 `GlobalWarmStartQueue`（及 `SegmentCandidateIndex`）：

| 信息源 | 入口方法 | 数据内容 | 目标队列 |
|--------|---------|---------|---------|
| Prefill routing | `observe_prefill()` | prefill 阶段 router 输出的 expert ids + routing weights | GlobalWarmStartQueue + long_term SegmentIndex |
| Draft routing | `observe_draft()` | draft 阶段每步 forward 的 gating metadata | draft SegmentIndex（segment_indexed 模式）或 GlobalWarmStartQueue |
| Verify routing | `observe_verify()` | verify 阶段 router 输出 + rank_guard scores 更新 | GlobalWarmStartQueue + long_term SegmentIndex |

三个信息源通过 `source_weight` 配置各自的优先级贡献权重（`prefetch_source_weight_prefill/draft/verify`）。

#### 5.1.2 候选优先级计算

当前 `compute_priority()` 公式（`prefetcher.py:47`）：

```
priority = source_weight × score_sum + activation_count_weight × activation_count − age_penalty × age
```

其中：
- `score_sum`：routing weight 的 decay 累加（`decay * old_score + new_score`），反映专家的激活强度
- `activation_count`：出现次数的 decay 累加，反映激活频率
- `age`：`current_step - last_update_step`，惩罚长时间未被观察到的候选

#### 5.1.3 两阶段预取流程

**Draft 阶段预取**

Draft 阶段有三种预取模式（由 `prefetch_runtime_mode` 配置）：

| 模式 | 方法 | 特点 |
|------|------|------|
| `baseline_staging` | `submit_from_global_queue()` | 预取到 staging buffer，后续在安全点 publish 到 active cache |
| `draft_direct_active` | `submit_draft_direct_active_prefetch()` | 直接预取到 active cache slot，有 frontier 约束保护未执行层 |
| `draft_segment_indexed` | `submit_draft_segment_indexed_prefetch()` | 推荐模式。按 segment 边界提交，合并 long_term + draft 两个 SegmentIndex 的候选 |

Draft 预取的关键约束：
- **Frontier 保护**：仅预取 `layer_idx ≤ frontier_layer_idx` 的专家，避免替换后续 segment replay 需要读取的 active slot
- **Adaptive budget**：根据可见开销（visible_overhead_ms）自适应调整每次 boundary 提交数量
- **Stale metadata guard**：metadata 的 step_id ≠ 当前 active draft step_id 时跳过 direct-active 提交

整体时序：
```
run_draft()
  → begin_draft_iteration()                      # 清空 draft SegmentIndex
  → arm() metadata recorder                      # 准备捕获 gating metadata
  → model forward (MoE layers execute)            # GPU 写入 routing metadata
  → offload_async() → 异步 D2H copy              # metadata 从 GPU 到 pinned CPU
  → prefetch worker thread:
      → collect() → observe_draft()               # 聚合 metadata → 更新候选队列
      → _submit_prefetch_after_metadata()          # 排序候选 → 发起 H2D 专家传输
```

**Verify 阶段预取**

Verify 阶段的预取发生在**逐层执行时**，通过 `model_runner.before_verify_layer()` hook 触发：

```
before_verify_layer(layer_idx=l)
  → publish_direct_active_ready()           # 完成已就绪的 direct-active 传输
  → 计算 available_ms = EMA(layer_compute_ms) × safety_ratio
  → submit_verify_layer_prefetch(
        target_layer_idx=l+1,               # 只预取下一层
        available_ms=available_ms            # 时间预算 ≤ 当前层计算时间
    )
```

Verify 层预取的特点：
- **时间预算驱动**：available_ms 来自当前层计算耗时的 EMA 估计 × 0.8 安全系数
- **单层范围**：`submit_verify_layer_prefetch()` 仅过滤 `layer_idx == target_layer_idx` 的候选
- **候选来源**：排序后的 `GlobalWarmStartQueue`，即依赖 draft/prefill/verify 历史的混合信号
- **无独立信息源**：verify 层预取不使用当前 verify 正在执行的层的精确路由来预测后续层

### 5.2 优化分析

#### 5.2.1 当前预取系统的优势与局限

**已有的优势：**

1. 异步流水线架构成熟：metadata offload（D2H）、候选排序、H2D 传输完全在后台 worker thread 完成
2. Segment-indexed 模式能按 segment 粒度精准投递，避免一次性预取浪费带宽
3. Verify 层预取已实现 `before_verify_layer` hook + 时间预算机制，框架完整

**需要关注的局限：**

1. **优先级函数未区分 top-1/2 重要性**：当前 `compute_priority()` 纯粹基于 score_sum（routing weight 累加）和 activation_count。LFU-RankGuard（§2.2）的 `rank_scores` 信息未反映到预取优先级中。一个 top-1 高频专家和一个 top-8 高频专家在预取优先级上没有区别，但 miss 造成的质量损失差异巨大。

2. **Verify 层预取的候选池未充分利用 draft original routing**：`submit_verify_layer_prefetch()` 从 `GlobalWarmStartQueue` 中取候选，这个队列已经包含了 draft routing 的历史信息（通过 `observe_draft()`）。但问题在于 GlobalWarmStartQueue 是跨层混合的队列，verify 层预取过滤 `layer_idx == target` 后剩余的候选可能很少。如果 draft metadata 的 observe 延迟较大或被 stale guard 跳过，target_layer 的候选可能不完整。

3. **Draft SegmentIndex 与 Verify 层预取的信息断层**：在 `draft_segment_indexed` 模式中，draft routing 进入 `draft_segment_index`，但 verify 开始前 `end_draft_iteration()` 会清空 draft_segment_index。Verify 层预取只能依赖 GlobalWarmStartQueue 和 long_term_segment_index 中的残留信息，**draft 当前轮次的 per-layer routing 预测被丢弃了**。

#### 5.2.2 优化方向 1：Top-1/2 Boost 优先级

在 `compute_priority()` 中引入 `rank_score` 提升因子，使 top-1/2 高频专家在预取排序中获得优先权：

```python
def compute_priority(source, score_sum, activation_count, age, config,
                     rank_score=0.0, rank_boost_threshold=0.15):
    source_weight = { ... }.get(source, 1.0)
    base = (
        source_weight * float(score_sum)
        + float(config.prefetch_activation_count_weight) * float(activation_count)
        - float(config.prefetch_age_penalty) * float(age)
    )
    # top-1/2 boost
    if rank_score >= rank_boost_threshold:
        base *= 1.5
    return base
```

这与 §2.2 LFU-RankGuard 协同：RankGuard 保护 cache 中已有的 top-1/2 不被驱逐，Top-1/2 Boost 确保未缓存的 top-1/2 优先被预取。两者共同形成"优先进、不轻出"的 top-1/2 保护闭环。

**实现位置**：修改 `prefetcher.py:compute_priority()`，`rank_score` 从 `LFURankGuardStrategy.rank_scores` 查表获取。

**开销**：每个候选增加一次 dict 查表，≈0 开销。

#### 5.2.3 优化方向 2：保留 Draft Per-Layer Routing 信息到 Verify 阶段

当前 `end_draft_iteration()` 清空 `draft_segment_index`，导致 verify 阶段无法利用 draft 当前轮次的 per-layer 精确路由信息。

**方案**：在 `end_draft_iteration()` 中将 draft_segment_index 的候选 merge 到 long_term_segment_index（或一个专用的 verify_hint 缓存），而非直接丢弃。

```python
def end_draft_iteration(self):
    if self._draft_iteration_open and self.draft_segment_index is not None:
        # 将 draft 候选合入 long_term，供 verify 层预取使用
        self.long_term_segment_index.merge_from(self.draft_segment_index)
    self.draft_segment_index.clear()
    self._active_draft_iteration_steps.clear()
    self._draft_iteration_open = False
```

这样 verify 层预取通过 GlobalWarmStartQueue（已含 draft observe 数据）+ 增强后的 long_term_segment_index 能获得更完整的 per-layer 候选。

**前提验证**：需要确认 draft routing 对 verify routing 的预测准确性。现有 `record_verify_consumed()` 已经记录了 verify 实际消费了哪些预取专家，可以从 profile 中提取 draft→verify 命中率作为验证。

#### 5.2.4 优化方向 3：Verify 层预取的层序紧迫性

当前 `submit_verify_layer_prefetch()` 只预取 `target_layer_idx = current_layer + 1`（单层向前看）。如果当前层计算耗时足够长且 PCIe 带宽有余，可以扩展到 `current_layer + 1 ... current_layer + Δ`。

但这个优化需要谨慎评估：
- verify 每层计算耗时约 0.5-1ms（取决于 token 数），PCIe 传输一个专家约 1.18ms
- 在 available_ms 内最多传输 0-1 个专家，lookahead 增大不会增加实际传输量
- 更大的 lookahead 主要价值是**选择更紧急的候选**（如 l+2 层有一个 top-1 miss，比 l+1 层的 top-7 miss 更值得预取）

**方案**：将 `submit_verify_layer_prefetch()` 的过滤条件从 `layer_idx == target` 改为 `target ≤ layer_idx ≤ target + Δ`，在排序时加入层序紧迫性因子：

```python
# 在 ranked_candidates 过滤后，按 (priority × urgency) 重排
urgency = 1.0 / (candidate.layer_idx - current_layer + 1)
effective_priority = candidate.priority * urgency
```

Δ 建议设为 2-3（对应 2-3 层 lookahead），过大无实际收益。

### 5.3 Prefetch Profiling 指标

为了支撑 Dynamic K 决策，预取系统需要向 `StepProfile` 输出以下指标：

```python
@dataclass  
class PrefetchStepMetrics:
    draft_prefetch_submitted: int   # draft 阶段提交的预取请求数
    draft_prefetch_completed: int   # draft 结束时已完成的预取数
    verify_miss_total: int          # verify 阶段实际 miss 总数
    verify_miss_covered: int        # verify miss 中被预取覆盖的数量
    verify_stall_ms: float          # verify 阶段的 PCIe 等待时间
    
    @property
    def coverage(self) -> float:
        return self.verify_miss_covered / max(1, self.verify_miss_total)
```

这些指标大部分已可从 `PrefetchRuntime._profile` 中提取：

| 指标 | 现有 profile key | 说明 |
|------|-----------------|------|
| draft_prefetch_submitted | `draft_*_prefetch_submit_count` | 已有 |
| draft_prefetch_completed | `direct_active_prefetch_publish_count` | 已有 |
| verify_miss_covered | `verify_layer_prefetch_consumed_count` | 已有（`record_verify_consumed` 路径） |
| verify_miss_total | — | 需新增：verify 阶段每层 uncached expert 计数 |
| verify_stall_ms | — | 需新增：verify 计算中等待 PCIe 的时间 |

`verify_miss_total` 和 `verify_stall_ms` 需要在 `run_verify()` 路径中新增采集点，与 `before_verify_layer()` / `after_verify_layer()` hook 结合。

---

## 6. 算法间耦合的修正分析

### 6.1 验证实验暴露的耦合问题

E3 vs E4 的巨大分歧（K*=12 vs K*=1 at r=0.25）源于 **Dynamic K ⟷ Prefetch** 的强耦合：

```
Draft 更长 → 预取窗口更大 → 覆盖率更高 → verify stall 更小 → 理论上 K 应更长
但同时：
Draft 更长 → verify 需要更多 unique experts → 需求增长快于预取 → 覆盖率反而下降
```

E3 假设预取率固定（每步 m 个），没有捕捉到"需求增长快于供给"的动态。E4 虽然模拟了这个效应，但没有考虑 verify 阶段的层间预取，高估了 stall。

**真实的最优 K 在两者之间**，且高度依赖实际硬件的 PCIe 带宽利用率。这进一步证实了 §4 中 profiling-based Dynamic K 的必要性——只有实际运行数据才能准确反映这个耦合。

### 6.2 闭环优化路径

```
Alg2_v2 rerouting
    ↓ 减少 draft miss → 更高 α → 允许更长 K
    ↓ 产生 original routing metadata（不受 bias 影响）
    ↓
Prefetch (draft 阶段)
    ↓ 利用 original routing 预取 verify 专家
    ↓ 预取优先级受 LFU-RankGuard 的 rank_scores 指导
    ↓
Dynamic K Controller (Level-1 + profiling)
    ↓ 基于实际 accept rate + coverage + stall 选择 K
    ↓ K 决定 draft 步数 → 影响预取窗口
    ↓
LFU-RankGuard Cache
    ↓ 驱逐决策基于 verify routing access 统计
    ↓ Top-1/2 保护确保关键专家驻留
    ↓ cache 内容影响下一轮 rerouting 的 miss rate
    ↓
→ 回到 Alg2_v2 rerouting
```

---

## 7. 实现路线图（修正版）

### Phase 1: 基础集成（1-2 周）

**P0 优先级：**

- [ ] Metadata 双轨记录：确认 `flat_selected_original` 在所有 draft 路径中正确传递
- [ ] LFU-RankGuard 策略：实现并替代当前 `AdaptiveCacheStrategy`
- [ ] Level-1 Dynamic K 阈值：在 `speculative_step()` 中加入 early-stop 检查
- [ ] Top-1/2 reroute 保护：集成到 `build_draft_plan_gpu()` 的替换 LUT 构建中

### Phase 2: Alg2_v2 集成 + Profiling（2 周）

**P1 优先级：**

- [ ] Alg2_v2 routing wrapper：在 `placement.py` 中实现偏置后 top-k + SkipAll 残余处理
- [ ] StepProfile 收集：扩展 `spec_engine.py` 的 profile 记录
- [ ] PrefetchStepMetrics 聚合：从 `PrefetchRuntime._profile` 中提取覆盖率指标
- [ ] ProfilingDynamicKController：实现并挂接到 `_budget_draft_steps()`

### Phase 3: 在线优化（1-2 周）

**P2 优先级：**

- [ ] Online α calibration：在 profiling controller 中集成 E5 的解析模型
- [ ] Verify 层间预取优化：利用 draft original routing 的 per-layer 信息
- [ ] rank_scores 在线 EMA 更新
- [ ] 端到端 benchmark：对比 baseline（LRU + 静态 K + round-robin 替换）与优化后系统

### 优先级排序（修正版）

| 优先级 | 设计点 | 理由 | 实验依据 |
|--------|--------|------|---------|
| P0 | Metadata 双轨记录 | 预取和缓存策略的正确性基础 | 架构设计 |
| P0 | LFU-RankGuard | 保护 top-1/2 驻留（缓存层面） | E2 + 先验知识 |
| P0 | Level-1 Dynamic K 阈值 | 近零开销，拦截不利长 draft | E3 |
| P1 | Alg2_v2 routing 集成 | r≤0.50 时 seq_α 提升 3-9.5pp | v2 实验 |
| P1 | Profiling Dynamic K | 唯一能准确捕捉真实 T_stall 的方案 | E3 vs E4 矛盾 |
| P2 | Online α calibration | 边际 K 决策的优化 | E5 |
| P2 | Verify 层间预取 | 提高覆盖率的关键 | E4 局限性分析 |

---

## 8. 待解决问题与后续实验

### 8.1 Profiling Dynamic K 的冷启动

系统启动初期没有 profiling 数据，需要合理的默认策略：

- **方案 A：保守启动**，K=1 或 K=2，快速收集数据后提升
- **方案 B：离线标定启动**，使用 E3/E4 的（粗略）数据作为初始 LUT，在线修正

推荐方案 B：E3 虽然不够精确，但给出了不同 cache ratio 下 K 的相对大小关系（r=0.75 → 大 K, r=0.25 → 小 K），可作为初始启发。

### 8.2 Batch 场景下的 Dynamic K

当前分析均基于 batch_size=1。在 batch 推理中，不同 sequence 可能处于不同的 cache regime：

- 方案：使用 batch 中最保守的 K（即 min(K_opt per sequence)）
- 问题：这会导致高 cache ratio 的 sequence 被拖累
- 后续研究：是否需要 per-sequence K 调度

### 8.3 Alg2_v2 超参数的在线自适应

当前 γ₀=4.0, ρ_low=0.25, ρ_high=0.50 来自 v2 实验的离线标定。这些参数是否需要在线调整？

- γ₀ 控制 bias 强度：过大 → 路由偏离原始 → α 下降；过小 → miss 减少不够
- 可以将 γ₀ 纳入 profiling 的搜索空间（离散化为 {2, 4, 6, 8}）
- 但这会增加 exploration 的维度，前期建议固定 γ₀=4.0

### 8.4 需要在系统中验证的关键假设

| 假设 | 来源 | 验证方法 |
|------|------|---------|
| LFU 在线运行时可能驱逐 top-1/2 | 设计分析 | 记录 verify top-1/2 miss 事件频率 |
| Alg2_v2 的 miss-rate gate 在实际 KV cache 条件下仍有效 | v2 仿真 | 对比仿真 α 与实际 accept rate |
| Profiling Dynamic K 收敛速度 | 算法设计 | 记录 K 选择的稳定性和吞吐趋势 |
| Verify 层间预取能显著提升覆盖率 | E4 局限性分析 | 对比开启/关闭 verify 预取的 stall 时间 |

---

## 附录 A: 代码修改清单

### A.1 新增文件

```
nanovllm/scheduling/cache_strategy.py   # 新增 LFURankGuardStrategy 类
nanovllm/scheduling/dynamic_k.py        # 新增 DynamicKController 模块
nanovllm/scheduling/step_profile.py     # 新增 StepProfile / PrefetchStepMetrics
```

### A.2 修改文件

```
nanovllm/engine/speculative/spec_engine.py
  - _budget_draft_steps(): 集成 DynamicKController
  - speculative_step(): 增加 Level-1 early-stop + StepProfile 记录

nanovllm/expert/placement.py
  - build_draft_plan_gpu(): 替换 round-robin LUT 为 Alg2_v2 routing
  - 新增 _build_alg2v2_routing() 

nanovllm/expert/prefetcher.py
  - observe_runtime_meta(): 更新 rank_scores EMA
  - submit_draft_direct_active_prefetch(): 使用优化的优先级函数
  - 新增 PrefetchStepMetrics 输出

nanovllm/scheduling/cache_strategy.py
  - create_cache_strategy(): 注册 "lfu_rankguard"
```

### A.3 配置参数

```yaml
# cache_strategy
cache_strategy: "lfu_rankguard"       # 新增选项
rank_guard_threshold: 0.15             # top-1/2 保护阈值
rank_guard_ema_alpha: 0.95             # EMA 更新系数

# dynamic_k
dynamic_k_enabled: true
dynamic_k_max: 8
dynamic_k_level1_crit_threshold: 0.3
dynamic_k_level1_miss_stop: 0.6
dynamic_k_level1_miss_safe: 0.15
dynamic_k_profiling_window: 50
dynamic_k_profiling_update_interval: 10
dynamic_k_exploration_rate: 0.2
dynamic_k_exploration_decay: 0.99

# rerouting (Alg2_v2)
rerouting_algorithm: "alg2_v2"         # "skipall" | "alg2_v2"
rerouting_gamma0: 4.0
rerouting_miss_low: 0.25
rerouting_miss_high: 0.50
rerouting_top12_protect: true
```
