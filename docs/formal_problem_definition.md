# 专家缓存与专家预取：形式化问题定义与分析

## 1. 符号系统与基本设定

### 1.1 模型结构

| 符号 | 含义 |
|------|------|
| $L$ | MoE 层总数（索引 $l \in [L]$） |
| $N$ | 每层专家总数 |
| $\text{top}_k$ | 每层每 token 激活的专家数 |
| $S_l$ | 第 $l$ 层 GPU cache 的 slot 数量（$S_l \ll N$） |
| $\mathcal{C}_l^{(t)}$ | 第 $t$ 步时第 $l$ 层缓存驻留的专家集合, $|\mathcal{C}_l^{(t)}| \leq S_l$ |
| $e_{l,j}$ | 第 $l$ 层第 $j$ 号专家 |
| $w_{\text{size}}$ | 单个专家的参数量（字节），假设各专家同构 |

### 1.2 推理流程

| 符号 | 含义 |
|------|------|
| $K$ | 单轮 draft 的 token 数（可为动态值 $K^{(t)}$） |
| $\mathbf{x}^{(t)}$ | 第 $t$ 步输入 token |
| $R_l(\mathbf{x})$ | 原始 router 在第 $l$ 层对 token $\mathbf{x}$ 的 top-$k$ 选择结果集合 |
| $\hat{R}_l(\mathbf{x})$ | draft 阶段经路由修改后的实际执行专家集合 |
| $\sigma_l(j \mid \mathbf{x})$ | router softmax 对第 $l$ 层专家 $j$ 的 routing weight |

### 1.3 硬件参数

| 符号 | 含义 |
|------|------|
| $B_{\text{PCIe}}$ | PCIe 单向有效带宽 (bytes/s) |
| $\tau_e = w_{\text{size}} / B_{\text{PCIe}}$ | 单个专家的 PCIe 传输时延 |
| $T_{\text{draft}}(k)$ | 第 $k$ 步 draft forward 的 GPU 计算耗时 |
| $T_{\text{verify}}(K)$ | 对 $K$ 个 draft token 执行 verify 的总耗时 |
| $T_{\text{cpu}}$ | 单个专家在 CPU 上的执行耗时 |

---

## 2. 专家缓存双目标优化

### 2.1 问题背景

传统专家缓存（如 LRU/LFU）单一优化 **命中率**：使尽可能多的被路由选中的专家已驻留 GPU cache。在标准 MoE 推理中，这是唯一目标——cache miss 意味着 PCIe 传输阻塞或 CPU 计算回退。

在 expert-substitution speculative decoding 中，缓存承担了 **第二角色**：当 draft 阶段遇到 cache miss 时，系统不执行 PCIe 传输，而是从缓存池中选择一个替代专家近似执行。因此缓存内容不仅决定命中率，还决定了 **替代池的质量**——当 miss 发生时，缓存中是否存在能高质量近似目标专家的替代选项。

### 2.2 形式化问题定义

**输入**：
- 当前步 $t$ 各层缓存状态 $\{\mathcal{C}_l^{(t)}\}_{l=1}^L$
- 各层 slot 容量 $\{S_l\}_{l=1}^L$
- 专家间成对相似度矩阵 $\mathbf{M}_l \in \mathbb{R}^{N \times N}$，其中 $M_l(i,j)$ 表示专家 $e_{l,i}$ 替代 $e_{l,j}$ 时的质量保持度（$M_l(j,j)=1$）
- 专家激活概率分布 $p_l(j) = \Pr[j \in R_l(\mathbf{x})]$（可从历史统计或先验估计获得）
- 未来 $H$ 步的 draft + verify 执行计划

**决策变量**：每步结束时的缓存驱逐/加载决策，即 $\mathcal{C}_l^{(t+1)} \subseteq [N]$, $|\mathcal{C}_l^{(t+1)}| \leq S_l$。

**目标函数**：最大化期望加速比，可分解为两个子目标的加权组合：

$$\max_{\{\mathcal{C}_l\}} \sum_{l=1}^L \Big[ \underbrace{\lambda_1 \cdot \text{HitValue}_l(\mathcal{C}_l)}_{\text{目标1: 命中价值}} + \underbrace{\lambda_2 \cdot \text{SubstValue}_l(\mathcal{C}_l)}_{\text{目标2: 替代价值}} \Big]$$

其中：

**目标 1: 命中价值** — 衡量缓存对 verify 阶段 PCIe 传输消除的贡献（verify 使用原始路由，cache hit 直接避免传输）：

$$\text{HitValue}_l(\mathcal{C}_l) = \sum_{j \in \mathcal{C}_l} p_l(j) \cdot \tau_e$$

**目标 2: 替代价值** — 衡量缓存作为 draft 阶段替代池时，对 draft token 接受率的贡献：

$$\text{SubstValue}_l(\mathcal{C}_l) = \sum_{j \notin \mathcal{C}_l} p_l(j) \cdot \max_{i \in \mathcal{C}_l} M_l(i, j)$$

即对于每个可能的未命中专家 $j$，缓存中最佳替代者对其的近似质量，按 $j$ 的激活概率加权。

### 2.3 目标冲突分析

两个目标之间存在结构性张力：

**冲突场景 1：低频但高替代价值专家**。
设专家 $e_{l,a}$ 自身激活频率很低（$p_l(a) \approx 0$），但 $M_l(a, j)$ 对多个高频未命中专家 $j$ 均较高。LRU/LFU 会将其作为驱逐候选（对 HitValue 贡献近零），但保留它对 SubstValue 的边际贡献可能很大。

**冲突场景 2：高频专家的机会成本**。
设 slot 已满，需要在保留高频专家 $e_{l,b}$（$p_l(b)$ 大）与低频但高替代价值专家 $e_{l,a}$ 之间选择。保留 $b$ 对 HitValue 贡献 $p_l(b) \cdot \tau_e$，驱逐 $a$ 的 SubstValue 损失为 $\sum_{j \notin \mathcal{C}_l} p_l(j) \cdot [\max_{i \in \mathcal{C}_l} M_l(i,j) - \max_{i \in \mathcal{C}_l \setminus \{a\}} M_l(i,j)]$，即 $a$ 作为最佳替代者时的不可替代价值。

**冲突场景 3：Draft 与 Verify 的异步需求分歧**。
Draft 阶段缓存内容决定替代质量（SubstValue），verify 阶段缓存内容决定命中率（HitValue）。两个阶段之间存在时间差——draft 执行期间可以通过 prefetch 改变缓存内容。因此缓存策略需要 **跨时相规划**：draft 开始时的缓存偏向 SubstValue，draft 期间通过 prefetch 逐步将缓存转向 HitValue。

#### 2.3.1 目标冲突的经验性验证

上述冲突场景是否在实际模型中显著存在，是一个关键的经验性问题。存在两种对立假设：

**假设 A（重合）**：高频激活的专家往往也是高替代价值的——因为 router 将它们路由给大量不同 token，说明其功能覆盖面广，自然也容易近似其他专家。若此假设成立，双目标优化退化为单目标，LFU 已接近最优。

**假设 B（不重合）**：高频专家是"专科医生"——被频繁激活恰恰因为高度专业化，处理特定类型 token；而高替代价值专家是"全科医生"——功能更泛化、与多个专家输出空间重叠。高频 $\neq$ 泛化。

**初步判断**：在专家数少（如 Mixtral 8 experts/layer）时两者可能高度重合，但在细粒度 MoE（如 Qwen3 128 experts/layer, top-8）时分离度会显著增大——专家越多、top-k 选择越稀疏，功能分化越强，"全科"与"专科"的区分越明显。

**验证实验设计**：

**第一步：构建每层相似度矩阵 $M_l$**。跑 calibration dataset（如 C4 的 1000 条样本），对每层每对 $(i,j)$ 专家，收集所有被 router 选中 expert $j$ 的 token 的 hidden states，分别过 expert $i$ 和 expert $j$，计算输出的 cosine similarity 均值，即得 $M_l(i,j)$。

**第二步：计算每个专家的两个指标**。

- HitScore：$p_l(j)$，即该专家的激活频率（从同一 calibration set 统计）
- SubstScore：$\sum_{i \neq j} p_l(i) \cdot M_l(j, i)$，即该专家作为替代者时对所有其他专家的加权替代价值

**第三步：相关性分析**。对每层画 HitScore vs SubstScore 散点图，计算 Spearman rank correlation $\rho$。$\rho > 0.7$ 支持假设 A；弱相关或负相关支持假设 B。

**第四步：Oracle 对比实验**。分别用 HitScore-only、SubstScore-only、以及 joint 排序来选择 $S_l$ 个专家放入 cache，对比三种策略下的 draft 接受率。若 joint 显著优于两个 single-objective 策略，即证明双目标的必要性。

### 2.4 复杂性分析

**定理 1（NP-hard 性）**：给定专家相似度矩阵和激活概率分布，最大化 $\text{HitValue} + \text{SubstValue}$ 的联合目标在 slot 容量约束下是 NP-hard 问题。

*论证*：当 $\lambda_2 = 0$ 时退化为 weighted maximum coverage 问题（已知 NP-hard）。非零 $\lambda_2$ 引入的 SubstValue 项具有 submodular 结构（证明见下），但两个 submodular 函数的加权和在一般情况下不保持 submodular 性，因此贪心算法不能保证 $(1 - 1/e)$ 近似比。

**命题 1（SubstValue 的 Submodularity）**：$\text{SubstValue}_l(\mathcal{C}_l)$ 关于 $\mathcal{C}_l$ 是 submodular 函数。

*证明思路*：对任意 $\mathcal{A} \subseteq \mathcal{B} \subseteq [N]$ 和 $e \notin \mathcal{B}$，将 $e$ 加入 $\mathcal{A}$ 对 SubstValue 的边际贡献 $\geq$ 加入 $\mathcal{B}$ 的边际贡献。因为 $\max_{i \in \mathcal{A} \cup \{e\}} M_l(i,j) - \max_{i \in \mathcal{A}} M_l(i,j) \geq \max_{i \in \mathcal{B} \cup \{e\}} M_l(i,j) - \max_{i \in \mathcal{B}} M_l(i,j)$（max 的边际递减）。

**命题 2（HitValue 的 Modularity）**：$\text{HitValue}_l(\mathcal{C}_l)$ 是 modular 函数（各元素独立贡献，无边际递减）。

结合命题 1 和 2，联合目标 $\lambda_1 \cdot \text{HitValue} + \lambda_2 \cdot \text{SubstValue}$ 是一个 modular 函数与 submodular 函数的非负加权和，仍然是 submodular 的（非负加权和保持 submodularity）。因此对于 **静态** 的从空集构建缓存问题，贪心算法可以在每步选择边际贡献最大的专家，获得 $(1-1/e)$ 近似比。

**注意**：上述近似保证仅适用于一次性选择 $S_l$ 个专家的 batch 设定。在推理时的在线增量设定中（每步只替换少量专家），需要通过在线 submodular 优化框架（如 sliding-window greedy）获得类似保证，但会引入额外的 regret 项。

### 2.5 在线近似求解框架

在推理时限下，无法每步从头求解全局最优。需要在线增量策略：

**边际驱逐评分**：当需要驱逐某个专家 $e_{l,v}$ 为新专家腾出空间时，评估其驱逐代价：

$$\text{EvictCost}(v) = \underbrace{\lambda_1 \cdot p_l(v) \cdot \tau_e}_{\text{Hit 损失}} + \underbrace{\lambda_2 \cdot \Delta\text{SubstValue}_l(v)}_{\text{替代池损失}} + \underbrace{\mu \cdot \mathbb{1}[v \in \text{ProtectedSet}]}_{\text{保护惩罚}}$$

其中

$$\Delta\text{SubstValue}_l(v) = \sum_{j \notin \mathcal{C}_l} p_l(j) \cdot \max\Big(0,\; M_l(v,j) - \max_{i \in \mathcal{C}_l \setminus \{v\}} M_l(i,j)\Big)$$

即 $v$ 作为不可替代最佳替代者时的边际贡献。$\mu$ 为保护惩罚系数（详见 §2.6.2），ProtectedSet 为受保护专家集合。选择 $\text{EvictCost}$ 最小的 slot 作为驱逐候选。

**计算复杂度**：朴素计算 $\Delta\text{SubstValue}_l(v)$ 需要 $O(N \cdot S_l)$ 时间（对每个未缓存专家检查所有缓存 slot）。可通过维护 **top-2 替代者索引** 降至 $O(N)$：对每个未缓存专家 $j$，记录缓存中相似度最高和次高的两个替代者。仅当驱逐候选恰好是某个 $j$ 的最佳替代者时，才需要退化到次高替代者。

### 2.6 $\lambda_1, \lambda_2$ 自适应调节与缓存保护机制

#### 2.6.1 分阶段权重调节

权重应随推理阶段动态变化。Draft 阶段和 verify 阶段对缓存的需求存在本质差异：

- **Draft prefetch 阶段**（draft 执行中的预取决策）：缓存的角色主要是替代池，$\lambda_2$ 应更大。但预取的 **选择** 偏向 HitValue（为 verify 命中做准备），而 **驱逐策略** 偏向保护 SubstValue（不破坏 draft 替代质量）。
- **Verify prefetch 阶段**（verify 逐层执行中的预取）：需要的是即将被 verify 使用的精确专家，$\lambda_1$ 应压倒性地大。

为避免阶段切换的突变引起抖动（如 draft 后期刚预取的 verify 候选被 SubstValue 驱逐策略挤出），采用 **渐进线性插值**：

$$\lambda_1(k) = \lambda_1^{\text{base}} + (\lambda_1^{\text{verify}} - \lambda_1^{\text{base}}) \cdot \frac{k}{K}$$

即越接近 verify 开始，驱逐策略越偏向保护高命中率专家。进入 verify 阶段后固定为 $\lambda_1^{\text{verify}}$。

此外采用 **EMA 反馈微调** 作为补充：若连续几步 draft 接受率下降，微增 $\lambda_2$；若 verify stall 增加，微增 $\lambda_1$。相比 bandit 类方法（如 EXP3），EMA 反馈在 MoE 推理的短时间尺度（每轮仅 $K$ 步 draft）上收敛更快且可解释性更好。

#### 2.6.2 缓存保护机制

为防止高价值专家被不当驱逐，设计三层保护：

**Soft Protection（驱逐惩罚）**。不做硬分区，而是在 EvictCost 中为已识别的高价值专家加 protection bonus $\mu$（见 §2.5 公式）。ProtectedSet 定义为 top-$P$ HitScore 和 top-$P$ SubstScore 的并集。$\mu$ 取值应足够大使被保护专家仅在"万不得已"时才被驱逐，但不至于使 cache 完全冻结。

**Pin 机制（预取保护）**。刚预取进来的专家存在被后续预取操作驱逐的 thrashing 风险——尚未被 verify 使用就被换出。解决方案：

- 给刚预取的专家加 **min-residence 保护**：预取专家 $e$ 的目标为 verify 第 $l$ 层，则 pin 它直到 verify 执行到第 $l$ 层（或超过 $\delta$ 步）。
- 在 pin 期间，该 slot 不可被其他预取或驱逐选为 victim。
- 已有实现基础：`active_slot_pending_expert` 已保护传输中的专家，需要扩展为"传输完成 → pin until consumed"语义。

**Per-Layer 预取上限**。对同一层的预取数量施加上限，防止单层过度预取导致其他层 cache 空间被侵占：

$$B_l^{\text{prefetch}} = \max(0, \; \mathbb{E}[|V_l|] - |\mathcal{C}_l \cap V_l|) \approx \max(0, \; K \cdot \text{top}_k \cdot (1 - \text{hit\_rate}_l))$$

### 2.7 开放问题

**跨层联合优化 vs 逐层独立优化**。上述定义是 per-layer 的。实际中 draft 的接受率由所有层的联合近似质量决定（误差跨层累积），是否需要跨层协调缓存策略（如在前几层保留更高替代质量、后几层侧重命中率）？误差累积意味着前几层替代质量的边际重要性更大。这引入 $O(L \cdot S_l)$ 的决策空间。

**相似度矩阵 $\mathbf{M}_l$ 的获取与维护**。离线计算稳定但无法适应输入分布变化；在线估计适应性强但引入额外计算和统计噪声。混合方案（离线初始化 + 在线 EMA 修正）需要量化两部分的贡献权重和更新频率。从形式化分析看，SubstValue 的计算对 $M_l$ 绝对精度的敏感度有限——驱逐决策只需要 top-2 排序正确性，因此粗粒度估计可能已经够用。

---

## 3. 专家预取

### 3.1 问题背景

在本课题的 draft-verify 框架中，预取的核心目标是：**利用 draft 阶段的 GPU 计算时间窗口，异步将 verify 所需专家从 CPU 内存传输至 GPU cache**，使 verify 执行时 cache miss 尽可能少。

关键约束：
1. Draft 阶段的 GPU 计算与 PCIe 传输可并行，但 PCIe 带宽有限
2. Verify 所需的精确专家集合在 verify 执行前未知（取决于 verify 的原始路由结果）
3. 预取必须在 draft 开始时或 draft 步间边界发起，基于对 verify 激活模式的预测
4. 预取过多无用专家浪费带宽和 cache 空间，预取不足导致 verify 阶段 PCIe stall

### 3.2 两阶段预取模型

预取分为两个时序阶段，共享一个核心硬约束：**传输绝不阻塞计算**。

#### 3.2.1 Draft 阶段预取

**决策窗口**：每个 draft segment 的边界（如每 12 层为一个 segment，记 segment 大小为 $s$）。

**目标**：利用 draft forward 的 GPU 计算时间窗口，异步传输 verify 可能需要的专家。

**核心约束——不阻塞下一个 draft forward**。设在决策点 $t$ 发起 $m$ 个新传输，已有 $n_{\text{inflight}}$ 个传输在队列中，各自剩余完成时间为 $\{r_1, \ldots, r_{n_{\text{inflight}}}\}$，下一个 draft forward 的计算窗口为 $T_{\text{next}}$。则串行传输约束为：

$$\max(r_1, \ldots, r_{n_{\text{inflight}}}) + m \cdot \tau_e \leq T_{\text{next}}$$

等价地，可预取数量上限：

$$m \leq \Big\lfloor \frac{T_{\text{next}} - \text{remaining}(\text{inflight})}{\tau_e} \Big\rfloor$$

其中 $\text{remaining}(\text{inflight})$ 为当前传输队列的最大剩余完成时间。

如果估计不精确，需留安全余量：实际预取预算 = 理论预算 × $(1 - \epsilon)$，$\epsilon$ 为安全系数。

#### 3.2.2 Verify 阶段预取

**决策窗口**：每层 verify 计算开始时。

**目标**：利用当前层 verify 的计算时间，预取后续层 verify 需要的专家。

**核心约束——预取必须在目标层开始计算前完成**。设当前正在执行第 $l$ 层 verify，预取目标为第 $l+d$ 层的专家（$d \geq 1$），可用传输窗口为从现在到第 $l+d$ 层 verify 开始执行之间的计算时间：

$$n_{\text{prefetch}} \cdot \tau_e \leq \sum_{l'=l}^{l+d-1} T_{\text{verify\_layer}}(l')$$

其中 $T_{\text{verify\_layer}}(l')$ 为第 $l'$ 层 verify forward 的计算耗时。

**紧迫度差异**：前面的层紧迫度更高（verify 按层顺序执行，若前面层 miss 则无时间补救），因此在 verify 阶段预取应优先预取 $d$ 小的层。

#### 3.2.3 统一约束形式

两个阶段的约束可统一表述为：在任何决策点 $t$，预取调度器发起的所有传输 $\mathcal{P}(t)$，必须满足：

$$\text{completion}(\mathcal{P}(t)) \leq \text{deadline}(t)$$

其中 deadline 在 draft 阶段为下一个 draft forward 的开始时间，在 verify 阶段为目标层 verify 的开始时间。

### 3.3 预测不确定性建模

预取的核心难点在于 $V_l^{(t)}$（verify 在第 $l$ 层实际激活的专家集合）是未知的。可用信息包括：

**信息源 1: Draft 阶段路由观测**。Draft 的第 $k$ 步在第 $l$ 层的路由选择 $\hat{R}_l(\mathbf{x}_{t+k})$（经路由修改后）及原始 router logits $\sigma_l(j \mid \mathbf{x}_{t+k})$。由于 draft 使用了替代专家，hidden state 与真实前向传播有偏差，但 router logits 仍提供有价值信号。

**信息源 2: 历史统计**。Prefill 和之前 verify 阶段的精确路由记录。可估计每层专家的基础激活概率 $\hat{p}_l(j)$。

**信息源 3: Draft-Verify 路由相关性**。经验上，draft 和 verify 在相同 token 位置上的路由选择有较高重合度（这是 speculative decoding 在 MoE 上有效的前提之一）。设 $\rho_l$ 为第 $l$ 层 draft-verify 路由一致率：

$$\rho_l = \mathbb{E}\Big[\frac{|R_l(\mathbf{x}) \cap \hat{R}_l(\mathbf{x})|}{|R_l(\mathbf{x})|}\Big]$$

**信息源 4（仅 Verify 阶段预取可用）: 已执行 Verify 层的精确路由**。在 verify 逐层执行中，已经通过的层的精确路由统计是下一层的强预测信号——相邻层的激活模式往往高度相关。

**预测模型**。对 verify 中第 $l$ 层专家 $j$ 被激活的概率估计：

$$\hat{q}_l(j) = \alpha \cdot \mathbb{1}[j \in \hat{R}_l(\cdot)] + \beta \cdot \hat{p}_l(j) + \gamma \cdot \bar{\sigma}_l(j)$$

其中 $\bar{\sigma}_l(j)$ 是 draft 过程中观测到的平均 routing weight，$\alpha, \beta, \gamma$ 为可调参数。

由于 verify 并行处理 $K$ 个 token，第 $l$ 层的**唯一**激活专家数期望为：

$$\mathbb{E}[|V_l|] = N \cdot \Big(1 - \prod_{k=1}^{K}(1 - \hat{q}_l(j)^{(k)})\Big) \approx N \cdot \Big(1 - (1 - \bar{q}_l)^{K \cdot \text{top}_k}\Big)$$

这是 verify 阶段专家激增问题（Challenge 3.2）的数学基础——$K$ 越大，唯一激活专家数越多，cache miss 概率越高。

### 3.4 Verify 阶段专家激增问题（Challenge 3.2）

**定量分析**。设各层各专家的激活概率为均匀分布 $p = \text{top}_k / N$（最坏情况估计），则 verify 在一层中激活的唯一专家数期望为：

$$\mathbb{E}[|V_l|] = N \cdot \big(1 - (1 - p)^K\big) = N \cdot \big(1 - (1 - \text{top}_k/N)^K\big)$$

当 $N = 128, \text{top}_k = 8, S_l = 32$ 时：

| $K$ | $\mathbb{E}[\|V_l\|]$ | 超出 cache 容量的期望 miss 数 |
|-----|----------------------|---------------------------|
| 1   | 8                    | 0 (若 cache 命中率 ≥ 25%)  |
| 3   | 22.5                 | 0                          |
| 5   | 35.1                 | 3.1                        |
| 8   | 50.6                 | 18.6                       |
| 12  | 66.0                 | 34.0                       |

可见 $K \geq 5$ 时 verify 层的 cache miss 开始显著增长。即使 prefetch 完美预测，PCIe 传输时间也可能超出 draft 窗口。

**可行性条件**。设 draft 窗口为 $W_{\text{draft}} = K \cdot \bar{T}_{\text{draft}}$，需要预取的专家数为 $|\mathcal{P}|$，串行传输总时间为 $|\mathcal{P}| \cdot \tau_e$。预取完全覆盖的必要条件：

$$|\mathcal{P}| \cdot \tau_e \leq W_{\text{draft}} = K \cdot \bar{T}_{\text{draft}}$$

$$\Rightarrow |\mathcal{P}| \leq K \cdot \bar{T}_{\text{draft}} / \tau_e$$

其中 $\bar{T}_{\text{draft}} / \tau_e$ 是"每步 draft 计算时间内可传输的专家数"，记为 **传输-计算比** $\eta$。当 $\eta < 1$（即传输一个专家慢于一步 draft 计算）时，预取带宽是严格瓶颈。

### 3.5 预取候选选择策略

给定预取预算 $B$（由 §3.2 的约束计算得出），从候选集中选择 top-$B$ 个专家。

**贪心优先级排序**。定义专家 $(l, j)$ 的预取优先级：

$$\text{Priority}(l, j) = \underbrace{\hat{q}_l(j)}_{\text{被 verify 激活概率}} \times \underbrace{c_l(j)}_{\text{miss 代价}} \times \underbrace{f_{\text{urgency}}(l)}_{\text{紧迫度}}$$

其中各项的设计：

- **$\hat{q}_l(j)$**：预测 expert $j$ 在 verify 第 $l$ 层被激活的概率。信息源按阶段不同：
  - Draft 阶段预取：主要用 draft 路由观测（router logits 中 $j$ 的权重），加历史频率的贝叶斯先验
  - Verify 阶段预取：已执行 verify 层的精确路由统计更可靠（相邻层激活模式高度相关），加 draft 阶段同层 logits

- **$c_l(j)$**：miss 代价。基本为 $\tau_e$（PCIe 同步传输时间）或 $T_{\text{cpu}}$（CPU 回退时间），取决于 verify 阶段的 fallback 策略。

- **$f_{\text{urgency}}(l)$**：紧迫度。
  - Draft 阶段：verify 的前几层稍高（verify 按层顺序执行，前层 miss 无法补救），$f_{\text{urgency}}(l) = (L - l + 1) / L$
  - Verify 阶段：$f_{\text{urgency}}(l) = 1 / (l - l_{\text{current}})$（指数衰减，越近越紧迫）

**Per-Layer 上限约束**。防止同层过度预取导致浪费：同一层内的候选按 Priority 排序后只取前 $B_l^{\text{prefetch}}$ 个（见 §2.6.2）。

### 3.6 预取-Draft 长度耦合（与动态 Draft 长度的交互）

预取策略与动态 draft 长度决策存在双向耦合：

**正向影响（$K \to$ 预取）**：
- 更长的 $K$ 提供更大的传输窗口 $W_{\text{draft}}$，允许预取更多专家
- 但更长的 $K$ 也使 verify 激活的唯一专家数增长，**需要**预取的专家更多
- 存在一个均衡点 $K^*$：当 $K > K^*$ 时，verify miss 增速超过可用传输窗口增速

**反向影响（预取 $\to K$）**：
- 当预取命中率高时，可容忍更长 $K$（verify stall 被预取消解）
- 当预取命中率低时，$K$ 应更短以减少 verify 的 miss 总量

**联合优化形式**。设 $\alpha(K)$ 为 $K$ 步 draft 后的期望接受 token 数（来自 acceptance 策略和 draft 近似质量），$T_{\text{total}}(K)$ 为一轮 spec decoding 的总时延：

$$T_{\text{total}}(K) = \underbrace{\sum_{k=1}^K T_{\text{draft}}(k)}_{\text{draft 计算}} + \underbrace{T_{\text{verify}}(K) + T_{\text{stall}}(K, \mathcal{P})}_{\text{verify 及其 stall}} + T_{\text{accept}}$$

有效吞吐为：

$$\text{Throughput}(K) = \frac{\alpha(K) + 1}{T_{\text{total}}(K)}$$

最优 $K^*$ 满足：

$$K^* = \arg\max_{K \geq 1} \frac{\alpha(K) + 1}{T_{\text{total}}(K)}$$

### 3.7 预取方案分析总结

整合以上形式化，预取子系统的完整决策流程为：

1. **Draft segment 边界**（每 $s$ 层 draft 计算完成后）：
   - 收集本 segment 路由元数据，更新候选优先级
   - 计算传输队列剩余时间和下一个 draft forward 窗口，得出可用预取预算
   - 按 Priority 选择候选，发起异步传输（保证不阻塞下一步 draft）

2. **Draft 结束 / Verify 开始前**：
   - 等待进行中传输完成（bounded wait）
   - 将已完成的预取 commit 到 active cache

3. **Verify 逐层执行中**：
   - 对每层，检查 cache 状态；miss 的专家走 CPU 回退或同步传输
   - 当前层 verify 执行期间并行预取后续层专家（verify-layer prefetch），受目标层 deadline 约束

---

## 4. 动态 Draft 长度决策

### 4.1 问题定义

在每步 draft forward $k$ 完成后（$k = 1, 2, \ldots, K_{\max}$），系统需做出二元决策：

$$d_k \in \{\text{CONTINUE}, \text{STOP}\}$$

如果 $d_k = \text{STOP}$，则 $K = k$，进入 verify 阶段。

**决策所需信息**（在 $d_k$ 时刻可观测）：
- 已完成的 $k$ 步 draft 的路由选择 $\{\hat{R}_l^{(1)}, \ldots, \hat{R}_l^{(k)}\}_{l=1}^L$
- 每步每层的替代状况：哪些专家被 skip/替代，替代质量 $M_l(i,j)$
- 当前预取进度：$\mathcal{P}_{\text{inflight}}$（进行中）、$\mathcal{P}_{\text{ready}}$（已完成）
- 历史接受率统计

### 4.2 边际决策准则

在第 $k$ 步继续 draft 的净收益为：

$$\Delta G(k) = \underbrace{\Delta\alpha(k) \cdot \bar{T}_{\text{decode}}}_{\text{额外接受 token 节省的时间}} - \underbrace{T_{\text{draft}}(k+1)}_{\text{额外 draft 成本}} - \underbrace{\Delta T_{\text{stall}}(k)}_{\text{verify stall 增量}}$$

其中：
- $\Delta\alpha(k) = \alpha(k+1) - \alpha(k)$：第 $k+1$ 步 draft token 的边际接受概率
- $\bar{T}_{\text{decode}}$：标准 decode 一步的基线时延（用于换算加速收益）
- $\Delta T_{\text{stall}}(k)$：因多一个 draft token 导致 verify 多激活的专家引起的额外 stall

**停止条件**：当 $\Delta G(k) < 0$ 时应停止。

### 4.3 边际接受率估计

第 $k+1$ 步 draft token 的边际接受概率需要从 draft 阶段可观测量中估计。定义层级近似误差度量：

$$\epsilon_l(k) = \frac{|\{j \in R_l(\mathbf{x}_{k}) : j \notin \mathcal{C}_l\}|}{|R_l(\mathbf{x}_k)|}$$

即第 $l$ 层第 $k$ 步中被替代的专家比例。加权累积误差：

$$E(k) = \sum_{l=1}^L \sum_{k'=1}^{k} \gamma^{k-k'} \cdot \epsilon_l(k') \cdot (1 - \bar{M}_l(k'))$$

其中 $\bar{M}_l(k')$ 是第 $k'$ 步中替代专家的平均相似度，$\gamma \in (0,1)$ 为误差累积折扣因子（近期层误差影响更大）。

接受率可建模为 $E(k)$ 的递减函数：

$$\hat{\alpha}(k) \approx \alpha_0 \cdot \exp(-\lambda \cdot E(k))$$

其中 $\alpha_0$ 和 $\lambda$ 从在线历史拟合。

### 4.4 Verify Stall 增量估计

当从 $K=k$ 增加到 $K=k+1$ 时，verify 额外需要的专家数量可估计为：

$$\Delta|\mathcal{P}_{\text{need}}|(k) \approx \sum_{l=1}^L \sum_{j=1}^N \hat{q}_l(j) \cdot \prod_{k'=1}^{k}(1 - \hat{q}_l^{(k')}(j))$$

即在第 $k+1$ 个 token 中新引入的、之前 $k$ 个 token 未激活过的专家数。对应 stall 增量：

$$\Delta T_{\text{stall}}(k) = \max\Big(0, \; [\Delta|\mathcal{P}_{\text{need}}|(k) - \Delta|\mathcal{P}_{\text{prefetchable}}|(k)] \cdot \tau_e\Big)$$

其中 $\Delta|\mathcal{P}_{\text{prefetchable}}|(k)$ 是额外一步 draft 提供的传输窗口内可额外预取的专家数。

### 4.5 与 MoE-SpeQ Amortization Roofline Model 的对比与综合

#### 4.5.1 MoE-SpeQ 的 Speculative Governor

MoE-SpeQ（Hwang et al., 2024）提出 Amortization Roofline Model 来动态调节 draft 长度。其核心框架：

**吞吐目标**：$\Theta(k) = k_{\text{accept}}(k) \;/\; T_{\text{cycle}}(k)$

其中 $k_{\text{accept}}(k) = \sum_{i=1}^k \prod_{j=1}^i p_j$（链式条件接受概率），$p_j$ 为第 $j$ 个 draft token 的条件接受率，通过 warm-up 期间经验测量并持续以 EMA 更新。

**时延模型**：$T_{\text{cycle}}(k) = \max(T_{\text{draft}}(k), T_{\text{pcie,init}}) + T_{\text{pcie,new}}(k) + T_{\text{verify}}(k+1)$

其中 $T_{\text{pcie,new}}(k)$ 通过分析 Expert Lookahead Buffer（ELB）与当前 cache 状态计算。ELB 由 INT4 量化 draft model 精确预测每层每 token 的 expert ID，与 target model 路由一致率约 90.9%。

**优化**：$k^* = \arg\max_{k \in [k_{\min}, k_{\text{SLO}}]} \Theta(k)$，其中 $k_{\text{SLO}}$ 由离线 TTFT profiling 确定的上界。

#### 4.5.2 MoE-SpeQ 方案的局限性（在本课题框架下）

**局限 1：依赖独立 draft model 的高保真预测**。MoE-SpeQ 的 $p_i$ 估计来自 INT4 量化模型（与 target 路由 ~90% 一致）。本课题 draft 使用专家替代，hidden state 在 draft 过程中偏离真实前向传播——draft 的 $p_i$ 比量化 draft 更不稳定，特别是 $k$ 较大时累积误差更显著。因此 MoE-SpeQ 的 EMA 估计在本场景下存在 bias。

**局限 2：$T_{\text{pcie,new}}(k)$ 估计方式不适用**。MoE-SpeQ 通过分析 ELB（draft model 给出的精确 expert ID 预测）来估计 verify 需要加载的新专家数。本课题的 draft 路由被替代修改过，不能直接作为 verify 路由的精确预测。

**局限 3：不感知替代质量**。MoE-SpeQ 的 $p_i$ 只反映 token 级接受概率，不区分"接受率高因为量化误差小"还是"接受率高因为 cache hit 率碰巧高"。本课题可利用层级替代质量信息——具体哪些层做了替代、替代相似度多少——做更精细的逐步预测。

#### 4.5.3 本课题方案的比较优势

**优势 1：替代质量的实时可观测性**。每步 draft 结束后，$\epsilon_l(k)$ 和 $\bar{M}_l(k)$ 是直接可观测的（不需要 EMA 历史），可实时估计误差累积 $E(k)$——这是量化 draft model 无法提供的信号。

**优势 2：更简洁的 $T_{\text{cycle}}$ 结构**。Draft 阶段完全不产生 PCIe 传输（cache miss 被替代），无 $T_{\text{pcie,init}}$ 和 $T_{\text{pcie,new}}$ 的复杂重叠建模：

$$T_{\text{cycle}}(K) = \sum_{k=1}^K T_{\text{draft}}(k) + T_{\text{verify}}(K) + T_{\text{stall}}(K)$$

其中 $T_{\text{stall}}$ 可在 draft 期间被预取渐进消减。

**优势 3：预取进度感知**。每步 draft 结束时已知 $|\mathcal{P}_{\text{ready}}|$（成功预取的专家数），可更准确估计 $T_{\text{stall}}$——预取进度好则容忍更长 $K$，预取滞后则提前停止。

#### 4.5.4 综合优化方案

借鉴 MoE-SpeQ 的 roofline 框架作为宏观结构，用替代质量感知信号替换其 acceptance probability 估计：

**1. 保留 Throughput 最优化的总体框架**：

$$K^* = \arg\max_K \frac{\alpha(K)+1}{T_{\text{total}}(K)}$$

**2. 替换 $\alpha(K)$ 的估计方法**（不用 EMA chain-rule $p_i$，用实时累积替代误差模型）：

$$\hat{\alpha}(K) = \alpha_0 \cdot \exp\Big(-\lambda \sum_{k=1}^K \sum_l \epsilon_l(k)(1 - \bar{M}_l(k))\Big)$$

$\epsilon_l(k)$ 和 $\bar{M}_l(k)$ 为每步 draft 结束后的实时观测值。

**3. 替换 $T_{\text{stall}}$ 的估计方法**（不用 ELB 的精确 expert ID，用概率预测减预取进度）：

$$\hat{T}_{\text{stall}}(K) = \sum_l \max\big(0, \; \hat{N}_{\text{miss},l}(K) - |\mathcal{P}_{\text{ready},l}|\big) \cdot \tau_e$$

**4. 加入预取进度反馈**（MoE-SpeQ 没有的信号）：在每步 draft 的 $\Delta G(k)$ 计算中纳入当前预取进度 $|\mathcal{P}_{\text{ready}}|$。

**5. 复用 SLO-based bounding**：用离线 profiling 确定 $K_{\max}$，在线搜索范围限定在 $[1, K_{\max}]$。

**6. 两层决策机制**：roofline 提供宏观 $K$ 范围约束（哪些 $K$ 值物理上不可能有收益），$\Delta G(k)$ 的逐步边际准则在此范围内做 early-stop 微调。

---

## 5. 三大子问题的耦合关系

上述三个子问题（缓存策略、预取调度、动态 draft 长度）并非独立，存在以下耦合：

**缓存 ↔ 预取**：缓存驱逐策略决定了预取的写入目标（需要驱逐哪个 slot），同时预取的成功率影响缓存状态演化。一个好的缓存策略应该为即将到来的预取"腾出空间"，而非等到预取到达时才临时驱逐。分阶段 $\lambda$ 调节（§2.6.1）与 pin 机制（§2.6.2）是协调两者的关键。

**缓存 ↔ Draft 长度**：缓存内容决定 draft 的替代质量，从而影响接受率 $\alpha(K)$。更好的替代质量允许更长的 $K$，而更长的 $K$ 意味着更多 verify miss，反过来对缓存命中率提出更高要求。

**预取 ↔ Draft 长度**：draft 长度决定预取窗口大小，预取成功率影响 verify stall 从而影响最优 draft 长度。这是最直接的耦合——动态 $K$ 决策需要感知预取进度（§4.5.4），预取策略需要知道 $K$ 来规划传输预算。

**全局视角**：理想的联合优化是一个 **在线随机控制问题**，状态空间包含缓存内容、预取进度、draft 步数，动作空间包含驱逐选择、预取选择和 continue/stop 决策。由于状态空间巨大且转移概率（路由随机性）难以精确建模，实际需要将其分解为松耦合的子问题，通过共享信息（路由统计、缓存状态快照、预取进度）实现近似协调。

---

## 6. 与现有方法的差异化定位

| 维度 | CachePrior / SwapMoE / BuddyMoE | MoE-SpeQ / SP-MoE | 本课题 |
|------|-----------------------------------|---------------------|--------|
| 精度保证 | 永久偏差（设计层面 trade-off） | 理论无损（speculative sampling） | 理论无损（speculative sampling） |
| Draft model | 无 | 独立小模型或量化模型（占用 GPU 内存） | **自身 + 路由修改**（零额外 GPU 内存） |
| 缓存目标 | 仅命中率 | 仅命中率 | **命中率 + 替代价值联合优化**（§2） |
| 缓存保护 | 无 | 无 | **分阶段 $\lambda$ + soft protection + pin**（§2.6） |
| Draft 长度 | 不适用 | Amortization Roofline（EMA $p_i$） | **Roofline + 替代质量感知 + 预取进度反馈**（§4.5.4） |
| 预取约束 | 不适用 | ELB 精确预测 + 三阶段管理 | **两阶段不阻塞约束 + 概率预测 + 紧迫度排序**（§3.2-3.5） |
| Verify stall 处理 | 不适用 | 依赖预取窗口，但 draft model 压缩了 cache 预算 | 全 cache 预算用于预取，draft 零 PCIe 阻塞 |
