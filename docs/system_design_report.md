# Expert-Substitution Speculative MoE Inference: System Design Report

**Project:** nano-vllm-moe  
**Target Model:** Qwen3-30B-A3B (N=128 experts/layer, k=8 active, 48 MoE layers)  
**Hardware:** Single GPU + Host DRAM, PCIe 4.0 ×16  
**Date:** 2026-05-28

---

## 1. Background & Motivation

### 1.1 问题背景

稀疏 MoE 架构在消费级硬件上的核心瓶颈是 **Expert Offloading**。以 Qwen3-30B-A3B 为例：单个专家约 47 MB，PCIe 4.0 ×16 理论峰值 64 GB/s，传输耗时约 1.5 ms。在 cache ratio=0.25 时，每层平均约 6 个 miss，48 层的单次 decode step 的 PCIe 阻塞时间远超 GPU 计算时间。

### 1.2 本课题的核心思路

结合两个关键洞察设计 **Draft-Phase Expert Rerouting**：

1. **Expert 冗余性**：MoE 路由具有容错性，top-k 之外的 cached 专家在特定条件下可作为近似替代
2. **投机采样无损性**：speculative decoding 的 accept/reject 机制保证最终输出分布等价

由此实现 **三阶段流水线**：

- **Draft 阶段**：所有 cache miss 专家通过 rerouting 近似处理（替换或跳过），forward 完全在 GPU cached 权重上执行，消除 PCIe 传输
- **Verify 阶段**：使用完整原始路由精确计算，accept/reject 保证输出正确性
- **Prefetch**：draft 阶段提供时间窗口，异步预取 verify 所需专家

### 1.3 系统架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                    Speculative Engine                            │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │ Draft K  │──>│ Rollback │──>│  Verify  │──>│  Accept  │    │
│  │ steps    │   │ KV cache │   │ K+1 fwd  │   │ /Reject  │    │
│  └────┬─────┘   └──────────┘   └────┬─────┘   └──────────┘    │
│       │                              │                          │
│  ┌────▼──────────────────────────────▼─────┐                   │
│  │           Runtime Metadata              │                   │
│  │  • original routing (= verify 预测)      │                   │
│  │  • effective routing (= draft 执行)      │                   │
│  └────┬────────────────────────┬───────────┘                   │
│       │                        │                                │
│  ┌────▼─────┐            ┌────▼──────┐                         │
│  │ Prefetch │            │   Cache   │                         │
│  │ Runtime  │            │  Strategy │                         │
│  │ (预取队列) │            │  (驱逐决策) │                         │
│  └──────────┘            └───────────┘                         │
│                                                                 │
│  ┌───────────────┐   ┌───────────────────┐                     │
│  │ Dynamic K     │   │ Expert Rerouting  │                     │
│  │ Controller    │   │ Algorithms        │                     │
│  └───────────────┘   └───────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

本报告系统化地定义和分析四个耦合子问题：**Expert Rerouting**、**Cache Strategy**、**Prefetch Scheduling**、**Dynamic Draft Length**。

---

## 2. Expert Rerouting 算法

### 2.1 问题定义 (DPERP)

**符号定义**：

| 符号 | 含义 |
|------|------|
| $\mathcal{S}^\ell(\mathbf{h})$ | 第 $\ell$ 层 top-k 路由选择（原始） |
| $\mathcal{C}^\ell$ | GPU cache 驻留专家集（$\|\mathcal{C}^\ell\| = S$） |
| $\mathcal{H}^\ell = \mathcal{S}^\ell \cap \mathcal{C}^\ell$ | 命中集 |
| $\mathcal{M}^\ell = \mathcal{S}^\ell \setminus \mathcal{C}^\ell$ | 缺失集 |
| $\phi^\ell$ | 重路由函数：$\mathcal{M}^\ell \to \mathcal{C}^\ell \cup \{\varnothing\}$ |

**DPERP 目标**：最大化有效吞吐

$$\boldsymbol{\phi}^* = \arg\max_{\boldsymbol{\phi}} \frac{\mathbb{E}\left[\sum_{t=1}^K \prod_{s=1}^t P(\text{accept}_s; \boldsymbol{\phi})\right]}{T_\text{draft}(K; \boldsymbol{\phi}) + T_\text{verify}(K; \boldsymbol{\phi})}$$

**GPU Residency 约束**：$\forall e \in \mathcal{M}^\ell: \phi^\ell(e) \in \mathcal{C}^\ell \cup \{\varnothing\}$

### 2.2 校准数据

所有算法的离线部分依赖对校准集 $\mathcal{D}$（64 × 256-token chunks, WikiText-2）的统计量：

| 统计量 | 形状 | 含义 |
|--------|------|------|
| $S^\ell_\text{cos}$ | $[L, N, N]$ | 专家均值输出余弦相似度 |
| $S^\ell_\text{cond}$ | $[L, N, N]$ | 条件替换误差相似度 $\exp(-D_\text{cond}(e,j))$ |
| $S^\ell_\text{coact}$ | $[L, N, N]$ | 共激活频率 $P(j \in \mathcal{S} \mid e \in \mathcal{S})$ |
| $S^\ell_\text{corr}$ | $[L, N, N]$ | Router logit Pearson 相关 |
| $D^\ell_\text{skip}$ | $[L, N]$ | 跳过误差 $\mathbb{E}[\|E_e(h)\|_2/\|h\|_2]$ |
| $\omega^\ell$ | $[L]$ | 层敏感度（routing weight 方差归一化） |
| $f^\ell$ | $[L, N]$ | 激活频率 |
| $f^\ell_\text{rank1}$ | $[L, N]$ | Top-1 位置出现频率 |
| $f^\ell_\text{rank2}$ | $[L, N]$ | Top-2 位置出现频率 |

**关键经验发现**（v1 实验）：
- Off-diagonal 余弦相似度均值仅 **0.055**，约 90% 的 expert pair 低于 0.40
- 路由熵均值 3.967（= 81.8% × log(128)），全程平坦，不随上下文变化
- 连续 token 共享约 4/8 个 top-k 专家（话题连贯性，非分布集中）

### 2.3 公共基础：scatter_add 聚合

所有算法共享同一 forward 实现，通过 scatter_add 消除 v1 中的 double-counting 问题：

```
1. gate(h) → logits [T, N], softmax → top-k → (router_weights, router_indices)
2. _reroute() → (final_weights [T, k], final_indices [T, k])
3. 权重缓冲区 wb[T, N] = 0
   for slot in range(k):
       wb.scatter_add_(dim=1, index=fi[:,slot:slot+1], src=fw[:,slot:slot+1])
4. for ei in wb.any(dim=0).nonzero():  // 每个专家恰好调用一次
       out += experts[ei](h[mask_ei]) * w_ei
```

### 2.4 Miss-Rate Gate（所有算法共享）

v1 的核心教训：r=0.75 时所有替换算法不如 SkipAll。原因是低 miss rate 时（约 2/8 miss），miss 专家权重极低（各 6-8%），任何替代品的 K/V 向量都是 router 不认可的，8 步累积导致 seq_α 崩溃。

**Miss-rate gate** 从根本上解决此问题：

$$\gamma_\text{gate}(\rho) = \text{clamp}\!\left(\frac{\rho - \rho_\text{low}}{\rho_\text{high} - \rho_\text{low}}, 0, 1\right)$$

- $\rho < \rho_\text{low}$（默认 0.25）：gate = 0，退化为精确路由/SkipAll
- $\rho > \rho_\text{high}$（默认 0.50）：gate = 1，算法全开

### 2.5 Top-1/2 路由保护

Top-1/2 专家对生成质量极为关键。Router 的 top-1 通常占总权重 30-50%，top-2 占 15-25%，两者合计超过 50% 的 hidden state 贡献。任何对 top-1/2 的近似误差被 attention 固化后会持续传播。

**三层保护设计**：

**层 1: Rerouting 保护**

- **Top-1 绝对保护**：无论偏置/替换结果如何，原始 top-1 专家的 slot 不可被修改。若 top-1 不在 cache 中，该 slot 走 SkipAll（而非低质量替代）。理由：用错误替代品"替代"top-1 的危害远大于直接 skip 再 renormalize。
- **Top-2 条件保护**：若 top-2 的原始权重 $w_2 > w_\text{protect}$（默认 0.15），同样享受保护。

**层 2: Cache 驱逐保护**

在 EvictCost 公式中增加 routing-rank 保护项：

$$\text{EvictCost}(v) = \lambda_1 \cdot p(v) \cdot \tau_e + \lambda_2 \cdot \Delta\text{SubstValue}(v) + \mu_\text{rank} \cdot f_\text{top12}(v)$$

其中 $f_\text{top12}(v) = 2 f_\text{rank1}(v) + f_\text{rank2}(v)$，从校准集离线统计，在线 EMA 更新。

**层 3: Prefetch 优先级提升**

Top-1/2 高频专家的预取优先级乘以放大因子 $\beta_\text{rank}$（默认 1.5），确保 verify 阶段 cache 中优先包含这些关键专家。

### 2.6 Algorithm 0: SkipAll（基线）

Miss 专家权重置零，hit 专家做 renormalize：

$$\tilde{w}_i = \frac{w_i}{\sum_{j \in \mathcal{H}^\ell} w_j} \cdot \mathbf{1}[i \in \mathcal{H}^\ell]$$

**特性**：hidden state 方向几乎不变（仅幅度放大约 $1/(1-w_\text{miss})$），KV cache 零污染。在高 cache ratio 时最优。

### 2.7 Algorithm 1: Alg2_v2（熵条件预路由偏置 + Miss-Rate Gate）

**核心思想**：在 top-k 选择前对 cached 专家的 logit 加偏置，使 router 主动选择 cached 专家，从根本上减少 miss。不引入外来 K/V 表示，KV cache 污染最低。

**正式表述**：

有效偏置强度（结合 miss-rate gate 和路由熵调制）：

$$\gamma_\text{eff}(\mathbf{h}) = \gamma_0 \cdot \gamma_\text{gate}(\rho_\text{miss}) \cdot \left(0.2 + 0.8 \cdot \frac{\tau - \tau_\text{low}}{\tau_\text{high} - \tau_\text{low}}\right)$$

偏置后 top-k：

$$\tilde{g}_i^\ell = g_i^\ell(\mathbf{h}) + \gamma_\text{eff} \cdot \mathbf{1}[i \in \mathcal{C}^\ell]$$
$$\widetilde{\mathcal{S}}^\ell = \text{top-}k(\tilde{g}^\ell)$$

**关键设计**：
- 权重从**原始** logit 计算（保证权重比例反映模型真实偏好）
- Top-1/2 保护（见 §2.5）
- 残余 miss 走 SkipAll，不做后路由替换
- r=0.75 时 gate=0，完全退化为精确路由

**超参数**：$\gamma_0 = 4.0$，$\rho_\text{low} = 0.25$，$\rho_\text{high} = 0.50$

### 2.8 Algorithm 2: HybridCP_v2（有界候选池偏置 + Deviation Guard）

在 Alg2_v2 基础上增加两个安全约束：

1. **候选池约束**：偏置只施加给 top-$J$（$J = 3k$）候选内的 cached 专家，防止 router 完全不考虑的专家被强推入 top-k
2. **偏差守卫**：计算被挤出的原始 top-k 专家总权重 $\Delta_\text{route}$，若超过 $\tau_\text{dev} = 0.20$ 则放弃偏置

**候选池约束的影响**：减少了偏置覆盖面 → 残余 miss 比 Alg2_v2 更多 → 对 SubstValue 的需求更高（见 §3 分析）。

### 2.9 Algorithm 3: PostSub_v2（后路由替换 + Sim Floor）

**核心思想**：保持原始路由不变，仅在执行时用最佳 cached 替代品替换 miss 专家。

替换条件（所有条件必须同时满足）：

$$\phi^\ell(e) = j^* \iff \underbrace{\rho_\text{miss} > \rho_\text{low}}_\text{gate open} \wedge \underbrace{c_e \geq \theta_c \cdot \bar{c}}_\text{contribution OK} \wedge \underbrace{S^\ell(e, j^*) \geq \sigma_\text{floor}}_\text{sim floor} \wedge \underbrace{j^* \notin \mathcal{T}(\mathbf{h})}_\text{no double count}$$

**sim_floor 的经验基础**：off-diagonal 相似度均值 0.055，约 90% pair 低于 0.40。$\sigma_\text{floor} = 0.40$ 导致绝大多数替换退化为 SkipAll。PostSub_v2 仅在高相似度 pair 上做精确替换。

**与预取的良好交互**：原始路由被保留 → runtime_meta 直接反映 verify 需求 → 预取信号无偏。

### 2.10 Algorithm 4: Alg2_PostSub（两阶段组合）

- **阶段一**：Alg2_v2 先通过偏置减少 miss
- **阶段二**：对残余 miss，PostSub_v2 处理

**v2 实验结果**：Alg2_PostSub 的 seq_α 与 Alg2_v2 完全相同，说明阶段一已解决绝大多数 miss，阶段二几乎未触发。

### 2.11 v2 实验结果

```
Algorithm               r=0.750  r=0.500  r=0.250
─────────────────────────────────────────────────
SkipAll                 0.9706   0.9018   0.6364
Alg2_v2                 0.9700   0.9037   0.7309
HybridCP_v2             0.9701   0.9041   0.6929
PostSub_v2              0.9701   0.9019   0.6991
Alg2_PostSub            0.9700   0.9037   0.7309
```

**关键结论**：
- Miss-rate gate 使所有算法在 r=0.75 时与 SkipAll 持平（gate=0）
- Alg2_v2 在 r=0.25 时比 SkipAll 高 9.5 pp
- HybridCP_v2 的候选池约束导致其在 r=0.25 时落后 Alg2_v2 3.8 pp

### 2.12 算法对比总结

| | SkipAll | Alg2_v2 | HybridCP_v2 | PostSub_v2 | Alg2_PostSub |
|---|---|---|---|---|---|
| 路由时机 | Post | Pre | Pre | Post | Pre + Post |
| KV 污染 | 无 | 无 | 无 | sim_floor 保护 | 最小 |
| 高 ratio 行为 | 最优 | = 精确路由 | = 精确路由 | = SkipAll | = 精确路由 |
| 预取信号质量 | 好 | 好（用原始路由） | 好 | 最好（路由未修改） | 好 |
| 对 SubstValue 需求 | 无 | 低 | 中 | 高（但 sim_floor 限制） | 低 |

---

## 3. Cache Strategy（缓存策略）

### 3.1 问题定义：双目标优化

**传统缓存**仅优化命中率。在 expert-substitution speculative decoding 中，缓存承担**第二角色**：当 draft 遇到 miss 时，缓存池的内容决定了替代品质量。

**联合目标**：

$$\max_{\{\mathcal{C}_l\}} \sum_{l=1}^L \left[ \lambda_1 \cdot \text{HitValue}_l(\mathcal{C}_l) + \lambda_2 \cdot \text{SubstValue}_l(\mathcal{C}_l) \right]$$

**HitValue**（verify 命中价值）：

$$\text{HitValue}_l(\mathcal{C}_l) = \sum_{j \in \mathcal{C}_l} p_l(j) \cdot \tau_e$$

**SubstValue**（draft 替代价值）：

$$\text{SubstValue}_l(\mathcal{C}_l) = \sum_{j \notin \mathcal{C}_l} p_l(j) \cdot \max_{i \in \mathcal{C}_l} M_l(i, j)$$

### 3.2 双目标的必要性：算法条件化分析

不同 rerouting 算法对 SubstValue 的需求不同。定义**残余 miss 条件下的 SubstValue**：

$$\text{SubstValue}_\text{residual}(\mathcal{C}_l; \phi) = \sum_{j \notin \mathcal{C}_l} \rho_\text{residual}(j, \mathcal{C}_l; \phi) \cdot p_l(j) \cdot \max_{i \in \mathcal{C}_l} M_l(i, j)$$

其中 $\rho_\text{residual}(j, \mathcal{C}_l; \phi)$ 是专家 $j$ 在 rerouting 后仍然 miss 的概率：

| 算法 | $\rho_\text{residual}$ | 对 SubstValue 需求 |
|------|----------------------|-------------------|
| SkipAll / PostSub_v2 | 1（所有 miss 都暴露） | 名义上高，但 sim_floor 限制了实际价值 |
| Alg2_v2 | 低（偏置覆盖面广） | 低 |
| HybridCP_v2 | 中（候选池约束限制覆盖面） | 中 |

**核心未验证假设**：在 Qwen3-30B-A3B 上，HitScore 和 SubstScore 是否高度相关？若高度相关（Spearman ρ > 0.7），双目标退化为单目标，LFU 已近最优。→ 需要 **实验 E1** 验证。

### 3.3 缓存 access 统计的数据来源

**设计原则**：缓存驱逐决策应基于**目标模型**（verify）的路由偏好，而非 draft 的修改路由。

| 操作 | 数据来源 | 理由 |
|------|----------|------|
| `mark_access()` 更新 LRU/LFU | 原始路由（= verify 路由预测） | 驱逐决策反映目标模型偏好 |
| 预取队列优先级 | 原始路由 | 预取目标是 verify 需要的专家 |
| Draft 实际执行 | 修改后路由（effective） | GPU 计算 |

实现上，`observe_draft()` 传入的 runtime_meta 应记录原始路由（`flat_selected_original`），而非执行路由。当前代码中 `placement.py` 已区分 `flat_selected_original` 和 `flat_selected_effective`。

**Access 权重分级**：

| 来源 | access 增量权重 | 理由 |
|------|----------------|------|
| verify routing | 1.0 | 精确的目标模型路由 |
| prefill routing | 1.0 | 精确路由 |
| draft original routing | 0.5-0.8 | 原始路由但 hidden state 有偏差 |

### 3.4 驱逐策略

**方案 A: 纯 LFU/LRU**（当前实现）

```python
class LRUCacheStrategy:
    def select_victim_slot(self, snapshot, incoming_expert_idx, step_id):
        # 选择 last_access_step 最小的 slot
```

- 优势：简单，零额外开销
- 劣势：不感知 SubstValue 和 routing rank

**方案 B: EvictCost-aware**（双目标 + Top-1/2 保护）

$$\text{EvictCost}(v) = \lambda_1 \cdot p(v) \cdot \tau_e + \lambda_2 \cdot \Delta\text{SubstValue}(v) + \mu_\text{rank} \cdot f_\text{top12}(v)$$

需要维护：
- $p(v)$：在线 EMA 更新的激活频率
- $\Delta\text{SubstValue}(v)$：通过 top-2 替代者索引降至 $O(N)$
- $f_\text{top12}(v)$：离线统计 + 在线 EMA

**方案 C: 分阶段 $\lambda$ 调节**

$$\lambda_1(k) = \lambda_1^\text{base} + (\lambda_1^\text{verify} - \lambda_1^\text{base}) \cdot k/K$$

Draft 前期偏向保护 SubstValue，越接近 verify 越偏向 HitValue。

**开销分析**：方案 B 相比方案 A 增加的开销是每次驱逐决策的 $O(N)$ 扫描（计算 $\Delta\text{SubstValue}$），N=128 时约 0.01ms，可忽略。是否值得取决于 SubstValue 的实际收益 → **实验 E1**。

### 3.5 Draft 期间的预取与缓存交互

**设计约束**：Draft 期间不能冻结 cache 驱逐——预取是重要优化，必须在 draft 时间窗口内完成。

**安全保证**：

1. **时序安全**：预取的 `commit` 发生在 draft forward 的 segment 边界之间（`submit_draft_direct_active_prefetch` 在边界调用），不会在 forward 执行中途改变 cache 内容
2. **Pin 保护**：刚预取进来的专家加 min-residence 保护，直到 verify 使用
3. **驱逐基于 verify routing**：LRU/LFU 的 access 统计来自原始路由，不会因偏置保护错误的专家

---

## 4. Prefetch Scheduling（预取调度）

### 4.1 问题背景

Draft 阶段的 GPU 计算提供了异步预取 verify 所需专家的时间窗口。核心挑战：verify 所需的精确专家集合在 verify 执行前未知，需要预测。

### 4.2 两阶段预取模型

**Draft 阶段预取**：每个 draft segment 边界发起，利用 draft forward 的计算时间异步传输。

**Verify 阶段预取**：利用当前层 verify 计算时间，预取后续层专家。

**统一约束**：

$$\text{completion}(\mathcal{P}(t)) \leq \text{deadline}(t)$$

Draft 阶段 deadline = 下一步 draft forward 开始时间；Verify 阶段 deadline = 目标层 verify 开始时间。

### 4.3 预取信号来源与质量

| 信息源 | 可用阶段 | 质量 | 备注 |
|--------|----------|------|------|
| Draft 原始路由 | Draft 中 | 高 | Rerouting 不改变原始路由 metadata |
| Prefill 路由历史 | 全程 | 中 | 基线频率先验 |
| Verify 路由历史 | Verify 后 | 最高 | 精确目标路由 |
| 已执行 verify 层路由 | Verify 逐层中 | 高 | 相邻层强相关 |

**Rerouting 算法对预取信号的影响**：

所有 v2 算法（Alg2_v2、HybridCP_v2、PostSub_v2）的重路由发生在原始 router 计算之后。原始路由 metadata（`flat_selected_original`）始终可用于预取队列更新，不受 rerouting 影响。

预取队列更新逻辑（`GlobalWarmStartQueue.update_from_runtime_meta`）过滤已 cached 专家，只保留 uncached 候选。使用原始路由时，uncached 候选精确反映 verify 可能的 miss。

### 4.4 Verify 阶段专家激增

Verify 并行处理 K 个 token，激活的唯一专家数随 K 增长：

$$\mathbb{E}[|V_l|] = N \cdot (1 - (1 - \text{top}_k/N)^K)$$

| K | $\mathbb{E}[\|V_l\|]$ | 超出 cache (S=32) |
|---|----------------------|-------------------|
| 3 | 22.5 | 0 |
| 5 | 35.1 | 3.1 |
| 8 | 50.6 | 18.6 |

**预取可行性条件**：

$$|\mathcal{P}| \cdot \tau_e \leq K \cdot \bar{T}_\text{draft}$$

当 $\bar{T}_\text{draft} / \tau_e < 1$（传输一个专家慢于一步 draft 计算），预取带宽是严格瓶颈。这是限制 K 的物理约束之一。

### 4.5 预取优先级

$$\text{Priority}(l, j) = \hat{q}_l(j) \times c_l(j) \times f_\text{urgency}(l) \times \beta_\text{rank}(j)$$

其中：
- $\hat{q}_l(j)$：预测 verify 激活概率
- $c_l(j) = \tau_e$：miss 代价
- $f_\text{urgency}(l) = (L-l+1)/L$：前层优先
- $\beta_\text{rank}(j)$：Top-1/2 高频专家放大因子

---

## 5. Dynamic Draft Length（动态 Draft 长度）

### 5.1 问题定义

在每步 draft forward 完成后，决策 $d_k \in \{\text{CONTINUE}, \text{STOP}\}$。

**边际决策准则**：

$$\Delta G(k) = \underbrace{\Delta\alpha(k) \cdot \bar{T}_\text{decode}}_\text{节省时间} - \underbrace{T_\text{draft}(k+1)}_\text{draft 成本} - \underbrace{\Delta T_\text{stall}(k)}_\text{verify stall 增量}$$

$\Delta G(k) < 0$ 时应停止。

### 5.2 两级信号架构

#### Level-1：阈值信号（always-on，≈0 开销）

目标：零开销地拦截明显应该 early-stop 的情况。

**信号 A：Top-1/2 Critical Miss Rate**

$$\text{CriticalMissRate}(k) = \frac{1}{L}\sum_{l=1}^L \mathbf{1}[\text{top-1 or top-2 of layer } l \text{ is miss}]$$

开销：一次 gather 操作 ≈ 0.01ms。

**信号 B：Per-step 平均 miss rate**

$$\bar{\rho}_\text{miss}(k) = \frac{1}{L}\sum_{l=1}^L \frac{|\mathcal{M}^\ell(\mathbf{h}_k)|}{k_\text{top}}$$

已被 miss-rate gate 计算，零额外开销。

**信号 C：Miss rate 趋势**

连续 2 步 miss rate 上升 → cache 状态恶化信号。

**Level-1 决策逻辑**：

```
if CriticalMissRate(k) > θ_crit (default 0.3):
    trigger Level-2
elif ρ_miss(k) > θ_stop (default 0.6):
    trigger Level-2
elif ρ_miss(k) < θ_safe (default 0.15):
    CONTINUE  // 几乎全部命中
else:
    if k % 2 == 0: trigger Level-2  // 中间区域周期评估
```

#### Level-2：接受率预测（按需触发）

**方案 A：解析模型**（推荐初始方案）

定义层级累积误差：

$$E(k) = \sum_{l=1}^L \sum_{k'=1}^k \gamma^{k-k'} \cdot \epsilon_l(k') \cdot (1-\bar{M}_l(k'))$$

其中 $\epsilon_l(k') = |\mathcal{M}^\ell|/k_\text{top}$ 为 miss rate，$\bar{M}_l$ 为替代平均相似度，$\gamma \in (0,1)$ 为折扣因子。

接受率估计：

$$\hat{\alpha}(k) \approx \alpha_0 \cdot \exp(-\lambda \cdot E(k))$$

$\alpha_0, \lambda$ 从最近 N 轮实际 accept/reject 结果在线 EMA 拟合。

开销：累积 $E(k)$ 每步加一个标量 + 指数运算 ≈ 0.001ms。

**方案 B：小 MLP**（数据充足后升级）

输入特征（≈10 维）：当前步 k、累积误差 E(k)、CriticalMissRate、$\bar{\rho}_\text{miss}$、miss rate 趋势（差分）、预取覆盖率、cache ratio、近期 seq_α 滑动平均。

结构：2 层 MLP，隐藏维 16，ReLU + sigmoid。参数量 ≈ 432（< 2KB），CPU 推理 ≈ 0.005ms。

**方案 C：RNN/GRU**（不推荐）

序列极短（K ≤ 8），RNN 序列建模优势无法体现。训练更复杂，冷启动问题更严重。

### 5.3 预取覆盖率信号

$$\text{PrefetchCoverage}(K) = \frac{|\mathcal{P}_\text{ready}|}{|\hat{\mathcal{P}}_\text{need}(K)|}$$

纳入 Level-2 的 $\Delta G(k)$ 计算：

$$\Delta T_\text{stall}(k) = \max\!\left(0, \left[\Delta|\mathcal{P}_\text{need}|(k) - \Delta|\mathcal{P}_\text{prefetchable}|(k)\right] \cdot \tau_e\right)$$

**覆盖率高（> 0.8）** → verify stall 小 → 允许更长 K  
**覆盖率低（< 0.4）** → verify 大量 stall → 即使 α 可以也应缩短 K

### 5.4 渐进实现策略

| 阶段 | 方案 | 依赖 | 预期收益 |
|------|------|------|----------|
| 0 | Level-1 阈值 + 解析模型 Level-2 | 无 | 捕捉大部分动态 K 收益 |
| 1 | 收集 (features, actual_α) 数据 | 阶段 0 运行数据 | 验证 MLP 是否有价值 |
| 2 | 小 MLP 替换解析模型（如验证有价值） | 阶段 1 数据 | 边际改善 |

### 5.5 收益估计

动态 K 的最大收益在低 cache ratio + 长默认 K 时：

- 固定 K=8, r=0.25, Alg2_v2: seq_α=0.73 → 后几步边际接受率低
- 假设最优 K=5，节省 3 步 draft + 对应 verify 负担
- 每个额外 verify token 增加的 PCIe stall ≈ 5-15ms（取决于预取覆盖率）
- 3 个 token × 10ms ≈ 30ms，在 T_cycle ≈ 100-300ms 中 → **10-20% 吞吐改善**

→ 需要 **实验 E3** 在仿真中验证不同 K 的最优点。

---

## 6. 四大子问题的耦合关系

```
Rerouting 算法 → 决定 draft routing metadata (original vs effective)
  ├──→ Prefetch: 原始路由用于预测 verify 需求
  ├──→ Cache: 原始路由用于 access 统计，驱逐基于 verify 偏好
  └──→ Dynamic K: miss rate / 替代质量信号用于 early-stop

Cache Strategy → 决定 C^ℓ 内容
  ├──→ Rerouting: 决定可用替代品集合和 miss/hit 集合
  ├──→ Prefetch: 驱逐策略决定预取的写入目标
  └──→ Dynamic K: cache 质量影响接受率

Prefetch → 改变 C^ℓ 内容（异步）
  ├──→ Cache: 预取成功后 commit 到 active cache
  ├──→ Verify: 预取覆盖率决定 verify stall
  └──→ Dynamic K: 预取进度影响 verify stall 估计

Dynamic K → 决定 draft 步数
  ├──→ Prefetch: K 决定预取时间窗口大小
  ├──→ Verify: K 决定 verify 处理的 token 数和 miss 总量
  └──→ Rerouting: 更长 K → 更多 KV cache 误差累积
```

**Alg2_v2 的正反馈闭环**：偏置减少 miss → draft 快 → 更多 draft 步 → 更长预取窗口 → verify hit rate 提高。闭环完整的前提是预取使用原始路由 metadata。

---

## 7. 验证实验设计

以下实验设计为独立的验证实验，不依赖完整系统实现，参考 `pre_exps/` 目录的风格。
### 实验settings
` model_path: "/data1/group_谈海生/mumura/


### 实验 E1: 双目标缓存必要性验证

**目的**：验证 HitScore 和 SubstScore 是否高度相关，判断双目标优化是否必要。

**方法**（4 步）：
1. 跑一次校准 forward pass，收集每层专家的激活频率 $p_l(j)$ 和输出均值向量，构建相似度矩阵 $M_l$
2. 计算每个专家的 HitScore = $p_l(j)$ 和 SubstScore = $\sum_{i \neq j} p_l(i) \cdot M_l(j,i)$
3. 对每层画散点图，计算 Spearman ρ
4. Oracle 对比：分别用 HitScore-only、SubstScore-only、joint（$0.5 H + 0.5 S$）选 $S_l$ 个专家入 cache，在 rerouting eval 中比较 seq_α

**扩展维度**：
- 按 cache ratio（0.25, 0.50, 0.75）分别测试
- 按 rerouting 算法条件化：增加 ResidualSubstScore($\phi$)
- 增加第四种策略：algorithm-aware joint

**预期结果**：
- 若 Spearman ρ > 0.7 且 Oracle 差异 < 1pp → 单目标 LFU 足够
- 若 ρ < 0.5 且 joint Oracle 优于 single > 2pp → 双目标有价值

**脚本**：`pre_exps/cache_dual_objective_eval.py`

### 实验 E2: Top-1/2 保护机制验证

**目的**：量化 top-1/2 miss 对接受率的影响，验证保护机制的收益。

**方法**：
1. 在 decode-mode 仿真中，记录每步每层的 top-1/2 路由结果和 cache 状态
2. 计算 CriticalMissRate(k) 与 per-step α 的相关性
3. 对比有/无 top-1/2 保护的 Alg2_v2 seq_α
4. 分析 top-1/2 miss 事件后的 α decay 速率

**扩展维度**：
- 不同 $w_\text{protect}$ 阈值（0.10, 0.15, 0.20）
- 不同 cache ratio

**预期结果**：
- CriticalMissRate 与 α 强负相关
- 有保护的 Alg2_v2 在低 cache ratio 时 seq_α 提升 1-3pp

**脚本**：`pre_exps/top12_protection_eval.py`

### 实验 E3: 动态 K 最优点分析

**目的**：在仿真中找到不同算法×cache ratio 下的最优 draft 长度。

**方法**：
1. 对每个（算法, cache_ratio）组合，跑 K=1..12 的 decode-mode 仿真
2. 计算每个 K 的 seq_α、alpha_decay、per-step 边际接受概率
3. 用简化 $T_\text{cycle}$ 模型（参数化 draft/verify/stall 耗时）计算理论吞吐
4. 找到每个配置的 $K^*$
5. 测试 Level-1 阈值信号能否在 $K^*$ 附近触发 stop

**时延参数**（Qwen3-30B-A3B on PCIe 4.0）：
- $T_\text{draft}(1)$ ≈ 2ms（纯 GPU, rerouting 消除 PCIe）
- $\tau_e$ ≈ 1.5ms（单专家传输）
- $T_\text{verify\_layer}$ ≈ 0.5ms（单层 GPU 计算）

**预期结果**：
- r=0.75: $K^* \geq 8$（α 衰减极慢）
- r=0.50: $K^* \approx 5-7$
- r=0.25: $K^* \approx 3-5$（Alg2_v2）或 2-4（SkipAll）

**脚本**：`pre_exps/dynamic_k_analysis.py`

### 实验 E4: 预取覆盖率与 verify stall 关系

**目的**：量化预取覆盖率对 verify 阶段的影响，验证覆盖率作为动态 K 信号的有效性。

**方法**：
1. 在 decode-mode 仿真中，每步 draft 记录原始路由产生的 verify 候选集
2. 模拟不同预取速率（每步预取 m=0,1,2,3 个专家）下的覆盖率
3. 基于覆盖率估计 verify stall
4. 计算不同 (K, m) 组合的理论吞吐

**预期结果**：
- 覆盖率 > 0.8 时 verify stall < 10% of T_cycle → K 可以拉长
- 覆盖率 < 0.4 时 verify stall 主导 → K 应缩短
- 预取覆盖率是有效的 Dynamic K 辅助信号

**脚本**：`pre_exps/prefetch_coverage_analysis.py`

### 实验 E5: 解析模型 vs 实际接受率的拟合精度

**目的**：验证 Level-2 解析模型 $\hat{\alpha}(k) = \alpha_0 \cdot \exp(-\lambda E(k))$ 的预测精度，判断是否需要 MLP 升级。

**方法**：
1. 在 decode-mode 仿真中收集 (E(k), actual_α(k)) 数据对
2. 拟合 $\alpha_0, \lambda$，计算 RMSE
3. 分析残差与额外特征的相关性（cache ratio, prefetch coverage, CriticalMissRate）
4. 若残差 R² 提升 > 0.1，建议 MLP 升级

**预期结果**：
- 解析模型 RMSE < 0.05 时已足够好
- 最有价值的额外特征是 CriticalMissRate

**脚本**：`pre_exps/alpha_prediction_eval.py`

---

## 8. 实现路线图

### Phase 1: 验证实验（2 周）

- [ ] E1: 双目标缓存验证 → 决定是否实现 EvictCost-aware 策略
- [ ] E2: Top-1/2 保护验证 → 确定保护阈值
- [ ] E3: 动态 K 最优点 → 确定 Level-1 阈值参数

### Phase 2: 核心集成（2 周）

- [ ] Metadata 双轨记录：original vs effective routing
- [ ] Top-1/2 保护集成到 rerouting wrapper
- [ ] Cache access 统计改为基于 original routing
- [ ] Dynamic K Level-1 阈值信号

### Phase 3: 优化（2 周）

- [ ] E4/E5 结果驱动：预取覆盖率信号、Level-2 模型选择
- [ ] EvictCost-aware 策略（若 E1 证明必要）
- [ ] 端到端 benchmark

### 优先级排序

| 优先级 | 设计点 | 理由 |
|--------|--------|------|
| P0 | Metadata 双轨记录 | 预取和缓存策略的正确性基础 |
| P0 | Top-1/2 Rerouting 保护 | 生成质量直接影响 |
| P1 | Dynamic K Level-1 阈值 | 近零开销，拦截不利长 draft |
| P1 | Cache access 基于 original routing | 驱逐决策与 verify 对齐 |
| P2 | Dynamic K Level-2 解析模型 | 适中开销，优化边际 K |
| P2 | E1 驱动的 EvictCost-aware 策略 | 需先验证必要性 |
| P3 | Dynamic K Level-2 MLP 升级 | 需运行数据，解析模型可能够用 |
