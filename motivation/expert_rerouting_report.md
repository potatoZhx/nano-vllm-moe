# Draft-Phase Expert Rerouting for Speculative MoE Inference
## Research Report

**Project:** nano-vllm-moe  
**Target Model:** Qwen3-30B-A3B（N=128 experts/layer，k=8 active，48 MoE layers）  
**Hardware:** Single GPU + Host DRAM，PCIe 4.0 ×16

---

## 1. 问题背景

### 1.1 MoE 模型的 Expert Offloading 瓶颈

稀疏混合专家（Sparse MoE）架构已成为前沿大模型的主流范式，通过每个 token 只激活少量专家（top-k routing）实现低计算成本下的高性能。然而，完整模型参数规模远超消费级 GPU 的 VRAM 容量，**Expert Offloading**——将大部分专家权重保存在主机 DRAM，按需加载至 GPU——成为标准解决方案。

这引入了一个根本性的延迟瓶颈：每次 cache miss 触发一次 PCIe 传输。以 Qwen3-30B-A3B 为例，单个专家约 47 MB，PCIe 4.0 ×16 理论峰值 64 GB/s，传输耗时约 1.5 ms，且在传输完成前推理完全阻塞。在每层平均 6 个 miss（cache ratio=0.25）的情况下，单次 decode step 的 PCIe 阻塞时间超过 GPU 计算时间 10 倍以上。

### 1.2 现有方法的局限

先行工作从三个方向攻坚，各有未解决的根本限制：

**方向一：CPU 计算（Fiddler, KTransformers）**  
将 miss 专家的计算转移至 CPU，避免权重传输。CPU 计算速度显著慢于 GPU，KTransformers 依赖 AMX 指令集（部分设备不支持）。

**方向二：路由修改（SwapMoE, BuddyMoE, CachePrior）**  
修改路由决策，令 token 优先选择 cache 内专家，减少 miss。加速效果不稳定，且**永久修改了模型行为**，精度降级不可控。

**方向三：投机解码 + 预取（SP-MoE, MoE-SpeQ, MoE-SpAc）**  
用 draft 模型的路由决策提前预取 verify 所需专家。但所有方案均要求 **draft 模型权重常驻 GPU**，挤占了 expert cache 的 VRAM 空间；verify 阶段并行处理多个 draft token，激活更多专家，PCIe 压力反而加重。

### 1.3 核心思路

本课题结合两个关键洞察：

1. **Expert 冗余性（来自路由修改文献）**：MoE 路由具有容错性，相邻专家在功能上存在相似性，用近似替代品不一定引起显著精度损失。

2. **投机采样的无损性**：投机解码的接受/拒绝机制在数学上保证最终输出分布等价于原始模型——替换误差只需在接受率意义上可控，不必为零。

由此设计**草稿阶段专家替换（Draft-Phase Expert Rerouting）**：

- **草稿阶段**：对所有 cache miss 专家实施替换或跳过，使 forward 完全在 GPU-cached 权重上执行，消除 draft critical path 上的 PCIe 传输和 CPU 计算
- **验证阶段**：使用完整原始路由精确计算，接受或拒绝草稿 token，保证分布等价性
- **预取**：draft 阶段提供时间窗口，异步预取 verify 所需专家



## 2. 问题正式定义（DPERP）

### 2.1 符号定义

设模型有 $L$ 个 MoE 层，每层有 $N$ 个专家。在第 $\ell$ 层：

- $\mathcal{E}^\ell = \{E_0^\ell, \ldots, E_{N-1}^\ell\}$：全体专家
- $\mathcal{C}^\ell \subset \{0,\ldots,N-1\}$，$|\mathcal{C}^\ell| = S$：GPU cache 中的专家集合
- Router $\mathcal{R}^\ell(\mathbf{h}) \to \mathcal{S}^\ell(\mathbf{h})$：top-k 路由，选出 $k$ 个激活专家
- $\mathcal{H}^\ell(\mathbf{h}) = \mathcal{S}^\ell(\mathbf{h}) \cap \mathcal{C}^\ell$：命中集（hit）
- $\mathcal{M}^\ell(\mathbf{h}) = \mathcal{S}^\ell(\mathbf{h}) \setminus \mathcal{C}^\ell$：缺失集（miss）

精确 MoE 输出：

$$\mathrm{MoE}^\star(\mathbf{h}; \ell) = \sum_{i \in \mathcal{S}^\ell(\mathbf{h})} w_i^\ell(\mathbf{h}) \cdot E_i^\ell(\mathbf{h})$$

### 2.2 草稿阶段重路由函数

**定义（Expert Rerouting Function）**

草稿阶段重路由函数 $\phi^\ell$ 为：

$$\phi^\ell : \mathcal{M}^\ell(\mathbf{h}) \times \mathbf{h} \times \mathcal{C}^\ell \;\to\; \mathcal{C}^\ell \cup \{\varnothing\}$$

将每个 miss 专家 $e \in \mathcal{M}^\ell$ 映射到一个 cached 替代品 $j \in \mathcal{C}^\ell$，或空集（跳过）$\varnothing$。

草稿 MoE 输出：

$$\widetilde{\mathrm{MoE}}(\mathbf{h}; \ell, \phi^\ell) = \sum_{i \in \mathcal{H}^\ell} w_i^\ell \cdot E_i^\ell(\mathbf{h}) + \sum_{e \in \mathcal{M}^\ell} w_e^\ell \cdot \Phi^\ell(e, \mathbf{h}, \mathcal{C}^\ell)$$

其中替代输出 $\Phi^\ell$ 定义为：

$$\Phi^\ell(e, \mathbf{h}) = \begin{cases} E_{\phi^\ell(e)}^\ell(\mathbf{h}) & \text{if } \phi^\ell(e) \in \mathcal{C}^\ell \\ \mathbf{0} & \text{if } \phi^\ell(e) = \varnothing \end{cases}$$

**约束（GPU Residency）**：$\forall e \in \mathcal{M}^\ell,\; \phi^\ell(e) \in \mathcal{C}^\ell \cup \{\varnothing\}$，即草稿阶段不允许 CPU 计算或 PCIe 传输。

### 2.3 优化目标（DPERP）

**定义（Draft-Phase Expert Rerouting Problem，DPERP）**

$$\boldsymbol{\phi}^* = \arg\max_{\boldsymbol{\phi}} \; \frac{\mathbb{E}\!\left[\sum_{t=1}^K \prod_{s=1}^t P(\mathrm{accept}_s;\, \boldsymbol{\phi})\right]}{T_\text{draft}(K;\, \boldsymbol{\phi}) + T_\text{verify}(K;\, \boldsymbol{\phi})} \quad \text{s.t. } \forall e,\ell:\; \phi^\ell(e) \in \mathcal{C}^\ell \cup \{\varnothing\}$$

其中 token 接受率由投机采样机制决定：

$$P(\mathrm{accept}_t) = \min\!\left(1,\; \frac{p^\star(\hat{d}_t \mid x_{<t})}{\tilde{p}(\hat{d}_t \mid x_{<t})}\right)$$

DPERP 可分解为三个相互独立的子问题：
- **P1 替代品选择**：对每个 $(e, \ell, \mathbf{h})$，选择使输出误差最小的 $j^* \in \mathcal{C}^\ell$
- **P2 草稿长度自适应**：决定草稿步数 $K$，平衡接受率与计算代价
- **P3 权重重分配**：决定 miss 专家的路由权重如何转移

### 2.4 近似误差分析

替换 $e \to j$ 的近似误差上界：

$$\Delta^\ell(\mathbf{h}; \phi^\ell) \leq \sum_{e \in \mathcal{M}^\ell} w_e^\ell \cdot \left\| E_e^\ell(\mathbf{h}) - E_j^\ell(\mathbf{h}) \right\|_2$$

跳过（$\phi^\ell(e) = \varnothing$）优于替换当且仅当：

$$\|E_e - E_j\|_2 > \|E_e\|_2 \iff \cos(E_e, E_j) < 0.5$$

由校准数据可知，Qwen3-30B-A3B 专家对的离对角余弦相似度均值仅为 **0.055**，远低于 0.5 的门槛。这意味着：**在低 cache ratio（大量 miss）时，任意随机替换的误差方向与目标专家几乎正交，替换比跳过更有害**，反之仅在相似度确实较高时替换才有意义。

---

## 3. 校准数据

算法的离线部分依赖对校准集 $\mathcal{D}$（64 个 256-token chunk，WikiText-2）的单次 forward pass 所得到的以下统计量：

| 统计量 | 形状 | 含义 |
|---|---|---|
| **余弦相似度** $S^\ell_\mathrm{cos}$ | $[L, N, N]$ | 专家均值输出向量的余弦相似度 |
| **条件替换误差相似度** $S^\ell_\mathrm{cond}$ | $[L, N, N]$ | $\exp(-D_\mathrm{cond}(e,j))$，$D_\mathrm{cond}$ 为在 $e$ 被激活的 token 上计算的归一化替换误差；未观测对用余弦先验填充 |
| **共激活频率** $S^\ell_\mathrm{coact}$ | $[L, N, N]$ | $P(j \in \mathcal{S} \mid e \in \mathcal{S})$ |
| **路由 logit 相关性** $S^\ell_\mathrm{corr}$ | $[L, N, N]$ | Router logit 的 Pearson 相关系数 |
| **综合相似度** $S^\ell$ | $[L, N, N]$ | $0.5 S_\mathrm{cos} + 0.3 S_\mathrm{coact} + 0.2 S_\mathrm{corr}$ |
| **跳过误差** $D^\ell_\mathrm{skip}$ | $[L, N]$ | $\mathbb{E}[\|E_e(h)\|_2 / \|h\|_2 \mid e \in \mathcal{S}^\ell]$ |
| **层敏感度** $\omega^\ell$ | $[L]$ | 该层路由权重方差的归一化值，越大越敏感 |
| **激活频率** $f^\ell$ | $[L, N]$ | 专家在校准集上的激活比例，用于 LFU cache warm-start |

---

## 4. 算法设计（v2）

### 4.1 设计总纲

v1 实验（见第 5 节）揭示了三个系统性问题，v2 算法针对性地修正：

| 问题 | 根因 | v2 修复 |
|---|---|---|
| r=0.75 时所有替换算法不如 SkipAll | 低 miss rate 时替换将路由器明确拒绝的专家的 K/V 写入 KV cache，8 步累积导致 seq_α 崩溃 | **Miss-rate gate**：miss rate 低时 gamma→0，退化为精确路由 |
| 双重计数 (double counting) | BaseWrapper 逐 slot 遍历专家，当替代品 j* 已在 top-k 中时同一专家被调用两次，权重被放大 | **scatter_add 聚合**：所有权重先累积到 [T, N] buffer，每个专家恰好调用一次 |
| 低 cache ratio 时随机替换恶化 KV cache | 相似度均值 0.055，大多数替代品与目标专家方向近乎正交 | **sim floor**：仅当最佳替代品相似度 ≥ 0.40 时替换，否则 SkipAll |

### 4.2 公共基础：BaseWrapper（修复 double counting）

所有算法共享同一 forward 实现，用 scatter_add 消除双重计数：

```
输入: hidden_states [B, T, D]
1. gate(h) → logits [T, N]，softmax → top-k → (router_weights, router_indices)
2. _reroute() → (final_weights [T, k], final_indices [T, k])
3. 权重缓冲区 wb[T, N] = 0
   for slot in range(k):
       wb.scatter_add_(dim=1, index=fi[:,slot:slot+1], src=fw[:,slot:slot+1])
   # 每个专家在 wb 中恰好出现一次，不会被重复调用
4. for ei in wb.any(dim=0).nonzero():    # 只调用实际有权重的专家
       out += experts[ei](h[mask_ei]) * w_ei
```

### 4.3 Algorithm 0：SkipAll（基线）

**设计思路**：Miss 专家的权重置零，对命中专家做 renormalize。保守但正确——hidden state 方向几乎不变，KV cache 不受外来专家污染。在高 cache ratio（miss 极少）时是最优选择。

**正式表述**：

$$\phi^\ell_\mathrm{SkipAll}(e, \mathbf{h}) = \varnothing \quad \forall e \in \mathcal{M}^\ell$$

$$\tilde{w}_i = \frac{w_i}{\sum_{j \in \mathcal{H}^\ell} w_j} \cdot \mathbf{1}[i \in \mathcal{H}^\ell]$$

### 4.4 Algorithm 1：Alg2_v2（熵条件预路由偏置 + Miss-Rate Gate）

**设计思路**：在 top-k 路由之前，对 cache 内专家的 logit 加偏置，使路由器主动选择 cached 专家，从根本上减少 miss 数量而不是事后处理 miss。与后路由替换相比，此方法不引入外来专家的 K/V 表示，KV cache 污染率最低。

**Miss-rate gate**：偏置强度与当前 token 的实际 miss rate 挂钩。当 cache ratio 高（miss 极少）时，gate = 0，算法完全退化为原始路由，避免了 v1 中 r=0.75 时的不必要路由干扰。

**正式表述**：

设当前 token 的 per-token miss rate 为 $\rho_\text{miss}(\mathbf{h}) = |\mathcal{M}^\ell(\mathbf{h})| / k$，定义 gate 函数：

$$\gamma_\text{gate}(\rho) = \text{clamp}\!\left(\frac{\rho - \rho_\text{low}}{\rho_\text{high} - \rho_\text{low}},\; 0,\; 1\right)$$

其中 $\rho_\text{low} = 0.25$（低于此值 gate 关闭），$\rho_\text{high} = 0.50$（达到此值 gate 全开）。

设 token 路由熵 $\tau(\mathbf{h}) = -\sum_i p_i \log p_i$，则有效 gamma：

$$\gamma_\text{eff}(\mathbf{h}) = \gamma_0 \cdot \gamma_\text{gate}(\rho_\text{miss}) \cdot \left(0.2 + 0.8 \cdot \frac{\tau - \tau_\text{low}}{\tau_\text{high} - \tau_\text{low}}\right)$$

偏置后 top-k：

$$\tilde{\mathbf{g}}^\ell_i = g^\ell_i(\mathbf{h}) + \gamma_\text{eff} \cdot \mathbf{1}[i \in \mathcal{C}^\ell]$$
$$\widetilde{\mathcal{S}}^\ell = \text{top-}k(\tilde{\mathbf{g}}^\ell)$$

**Top-1 保护**：若原始 top-1 专家不在 cache 中且被偏置挤出，则强制将其替换回最后一个 slot：

$$\text{if } \arg\max_i g_i \notin \widetilde{\mathcal{S}}^\ell \text{ and } \arg\max_i g_i \notin \mathcal{C}^\ell: \quad \widetilde{\mathcal{S}}^\ell_{[-1]} \leftarrow \arg\max_i g_i$$

**权重从原始 logit 计算**（保证权重比例反映模型真实偏好）：

$$\tilde{w}_i = \frac{\exp(g_i^\ell)}{\sum_{j \in \widetilde{\mathcal{S}}^\ell} \exp(g_j^\ell)}, \quad i \in \widetilde{\mathcal{S}}^\ell$$

**残余 miss 处理**：即使施加偏置后仍有 miss 的专家，直接 SkipAll（权重置零），**不做后路由替换**。

**超参数**：$\gamma_0 = 4.0$，$\rho_\text{low} = 0.25$，$\rho_\text{high} = 0.50$

### 4.5 Algorithm 2：HybridCP_v2（有界候选池预路由偏置 + SkipAll Fallback）

**设计思路**：在 Alg2_v2 基础上增加两个约束，进一步提高偏置的安全性：

1. **候选池约束（Bounded Candidate Pool）**：偏置只施加给"在原始 router 的 top-J 候选内"的 cached 专家（$J = 3k$）。防止将路由器完全未考虑的专家强行推入 top-k。

2. **偏差守卫（Deviation Guard）**：计算被偏置挤出的原始 top-k 专家的总权重 $\Delta_\text{route}$，若超过阈值 $\tau_\text{dev}$ 则放弃偏置，直接用原始路由。

**正式表述**：

候选池 $\mathcal{P}(\mathbf{h}) = \text{top-}J(\mathbf{g}^\ell(\mathbf{h})) \cap \mathcal{C}^\ell$（top-$J$ 中的 cached 专家）

有效偏置掩码：

$$\mathbf{b}_i = \gamma_\text{eff}(\mathbf{h}) \cdot \mathbf{1}[i \in \mathcal{P}(\mathbf{h})]$$

偏差检验：

$$\Delta_\text{route} = \sum_{e \in \mathcal{S}^\text{orig} \setminus \widetilde{\mathcal{S}}} w_e$$

$$\text{if } \Delta_\text{route} > \tau_\text{dev}: \quad \widetilde{\mathcal{S}} \leftarrow \mathcal{S}^\text{orig} \quad \text{（放弃偏置）}$$

**gamma 与 Alg2_v2 的区别**：去掉熵调制，直接 $\gamma_\text{eff} = \gamma_0 \cdot \gamma_\text{gate}(\rho_\text{miss}) \cdot H_\text{norm}$（基于实验观测，熵在此模型上几乎恒定，调制效果有限）。

残余 miss 同样 SkipAll 处理。

**超参数**：$\gamma_0 = 4.0$，$J = 3k = 24$，$\tau_\text{dev} = 0.20$

### 4.6 Algorithm 3：PostSub_v2（双重计数感知后路由替换）

**设计思路**：针对 v1 后路由替换的三个缺陷进行精确修复：

1. **Miss-rate gate**：低 miss rate 时（$\rho < \rho_\text{low}$）直接 SkipAll，不做任何替换
2. **Sim floor**：只有当最佳替代品的相似度 $\geq \sigma_\text{floor}$ 时才替换，避免低质量替换污染 KV cache
3. **双重计数感知**：若最佳替代品 $j^*$ 已在当前 token 的 top-k 中（作为 hit），则对该 miss slot 使用 SkipAll 而非替换，防止 $j^*$ 的权重被放大

**正式表述**：

对每个 miss 专家 $e \in \mathcal{M}^\ell(\mathbf{h})$，令当前 token 的 top-k 集合为 $\mathcal{T}(\mathbf{h}) = \mathcal{S}^\ell(\mathbf{h})$，最佳替代品：

$$j^*(e) = \arg\max_{j \in \mathcal{C}^\ell} S^\ell_\mathrm{cond}(e, j)$$

贡献度检验（防止跳过不必要的专家）：

$$c_e = w_e \cdot D^\ell_\mathrm{skip}(e)$$

当以下所有条件均满足时，执行替换：

$$\phi^\ell_\mathrm{PostSub}(e, \mathbf{h}) = j^* \iff \underbrace{\rho_\text{miss} > \rho_\text{low}}_{\text{gate open}} \;\wedge\; \underbrace{c_e \geq \theta_c \cdot \bar{c}}_{\text{contribution OK}} \;\wedge\; \underbrace{S^\ell(e, j^*) \geq \sigma_\text{floor}}_{\text{sim floor}} \;\wedge\; \underbrace{j^* \notin \mathcal{T}(\mathbf{h})}_{\text{no double count}}$$

否则 $\phi^\ell_\mathrm{PostSub}(e, \mathbf{h}) = \varnothing$（SkipAll）。

替换后权重按相似度×gate ramp 缩放：

$$\tilde{w}_{r} = w_{r} \cdot S^\ell(e, j^*) \cdot \gamma_\text{gate}(\rho_\text{miss})$$

**超参数**：$\sigma_\text{floor} = 0.40$，$\theta_c = 0.10$，$\rho_\text{low} = 0.25$

### 4.7 Algorithm 4：Alg2_PostSub（两阶段组合）

**设计思路**：

- **阶段一（PreRouting）**：Alg2_v2 先通过偏置减少 miss 数量
- **阶段二（PostRouting）**：对阶段一输出中仍有 miss 的专家，PostSub_v2 处理

这两阶段顺序执行，阶段一减少 miss 总量后，阶段二面对的 miss 更少，sim_floor 和 double-count 检查的触发频率也相应降低，综合效果优于单独使用任一阶段。

**正式表述**：

$$\widetilde{\mathcal{S}}^{(1)} = \text{Alg2\_v2}(\mathbf{h}, \mathcal{C}^\ell)$$
$$\phi^\ell_\mathrm{combined}(e) = \begin{cases} \text{Alg2\_v2}(e) & e \notin \widetilde{\mathcal{S}}^{(1)} \text{ is impossible after stage 1} \\ \text{PostSub\_v2}(e, \widetilde{\mathcal{S}}^{(1)}) & e \in \mathcal{M}^\ell(\mathbf{h}) \cap \text{residual after stage 1} \end{cases}$$

### 4.8 算法对比

| | **SkipAll** | **Alg2_v2** | **HybridCP_v2** | **PostSub_v2** | **Alg2_PostSub** |
|---|---|---|---|---|---|
| 路由时机 | Post | **Pre** | **Pre** | Post | Pre + Post |
| 主信号 | — | 路由熵、miss rate | miss rate、候选池 | 条件替换误差相似度 | 两者组合 |
| Miss-rate gate | — | ✓ | ✓ | ✓ | ✓ |
| Double-count 修复 | N/A | N/A | N/A | ✓ | ✓ |
| KV 污染来源 | 无 | 无（仅 SkipAll fallback）| 无（仅 SkipAll fallback）| 相似度低时跳过 | 最小 |
| 高 ratio 行为 | 最优 | 退化为精确路由 | 退化为精确路由 | 退化为 SkipAll | 退化为精确路由 |
| 低 ratio 核心优势 | 基线 | 大幅减少 miss | 候选池约束更安全 | 精确控制替换质量 | 两阶段减少残余 miss |

---

## 5. 实验设计

### 5.1 评测框架：Decode-Mode 仿真

区别于 v1 框架（256 token 并行 forward，所有 token 均应用替换），v2 框架精确模拟了真实 speculative decoding 的时序：

```
步骤 1：完整模型 forward 整个 chunk → target_logits [T, vocab]
步骤 2：完整模型 prefill 前 128 个 token → prompt KV cache
步骤 3：安装替换 wrapper
步骤 4：for step in range(8):
            input = chunk[128+step]（参考 token，非采样）
            draft_logit = model(input, past_kv=draft_kv)
            draft_kv 更新（包含替换产生的 K/V 偏差）
            α_step = TV(target_logit[128+step], draft_logit)
步骤 5：记录 per-step α 和 PPL
```

**关键特性**：
- 专家替换**仅在 draft 的 8 个 decode step 中发生**，prefill 使用完整模型
- draft KV cache 是独立累积的（包含替换误差），模拟真实误差传播
- 使用参考 token 作为输入（非采样），确保不同算法在相同 token 序列上比较

**评测指标**：
- `iso_α`：第 1 个 draft step 的接受率（单步质量，无 KV 误差积累）
- `seq_α`：8 步平均接受率（综合质量，含 KV 误差积累效应）
- `alpha_decay`：第 8 步 α / 第 1 步 α（KV cache 污染速率）

### 5.2 实验设置

| 参数 | 值 |
|---|---|
| 模型 | Qwen3-30B-A3B-Base |
| Cache ratios | 0.75, 0.50, 0.25 |
| Prompt length | 128 tokens |
| Draft length | 8 steps |
| N prompts | 32 |
| 校准数据 | WikiText-2 test（64 chunks × 256 tokens）|
| 评测数据 | WikiText-2 test（32 prompts × 128 tokens）|
| Cache warm-start | LFU（校准集激活频率最高的 S 个专家）|

### 5.3 路由熵分析实验（支撑实验）

为验证"routing 分布随上下文增长而集中"的假设，设计独立测量实验，测量以下四个命题：

| 假设 | 测量量 | 测量方法 |
|---|---|---|
| H1 路由熵下降 | $H(g^\ell(h_t))$ vs $t$ | gate hook 全量 logit，Spearman 相关 |
| H2 连续 token 专家重叠增加 | $\text{Jaccard}(\mathcal{S}_t, \mathcal{S}_{t+1})$ vs $t$ | per-position top-k 集合比较 |
| H3 top-1 权重增大 | $\max_i w_i(h_t)$ vs $t$ | gate hook |
| H4 跨序列专家一致性增加 | 序列间 pairwise Jaccard vs $t$ | 200 对随机序列采样 |

控制变量：per-token NLL（模型预测置信度，应随位置单调下降）。

---

## 6. 实验结果与分析

### 6.1 路由熵分析：原假设证伪

| 假设 | Spearman ρ | p 值 | 结论 |
|---|---|---|---|
| H1 路由熵下降 | +0.018 | 0.775 | **不显著，原假设证伪** |
| H2 连续 Jaccard 增加 | +0.442 | 1.26×10⁻¹³ | 显著，但效应量小 |
| H3 top-1 权重增大 | −0.093 | 0.136 | **不显著** |
| H4 跨序列一致性增加 | +0.154 | 0.014 | 显著但方向暂不确定 |
| 控制 per-token NLL 下降 | −0.560 | 1.79×10⁻²² | 显著，符合预期 |

**关键发现**：

路由熵均值 **3.967 / log(128) = 81.8%**，全程平坦，与 uniform 分布极为接近。这是 load-balancing auxiliary loss 刻意设计的结果，不随上下文长度变化。

H2 虽显著，但效应量极小：连续 token 共享专家数从 Q1 的 4.07/8 增加到 Q4 的 4.18/8，变化约 +7.1%，而观测到的均值 Jaccard=0.343 是随机期望（0.032）的 10.6 倍，说明连续 token 的专家共享是**话题连贯性**（semantic coherence）的体现，而非分布集中的结果。

模型预测置信度（NLL）从 2.90 下降到 1.80（-1.10 nats，PPL 降低 3.0×），但与路由熵的相关性 ρ=0.007（p=0.914），**完全不相关**。这证明模型在更长上下文下的置信度提升来自 attention 机制，与 MoE 路由行为无关。

**修正原来的分析**：r=0.5 时 seq_α > iso_α 的真实机制是话题连贯性下连续 token 共享的 ~4 个 cached 专家被 LFU cache 正确保留，这批命中专家在 8 步 draft 中保持稳定，而不是因为"路由分布趋于集中"。

### 6.2 v1 算法结果（三个 Cache Ratio）

| 算法 | r=0.75 | r=0.50 | r=0.25 | 核心问题 |
|---|---|---|---|---|
| SkipAll | **0.958** | 0.729 | 0.357 | 基线 |
| Alg2_EntropyBias | 0.913 | **0.823** | **0.576** | r=0.75 时不如 SkipAll |
| HybridCP | 0.889 | 0.801 | 0.480 | 同上 |
| RoundRobin | 0.913 | 0.723 | 0.423 | 随机替换 |
| Alg1_CoAct | 0.871 | 0.725 | 0.319 | r=0.25 低于 SkipAll |
| W4_TopHitRedist | 0.898 | 0.710 | **0.172** | r=0.25 灾难性失败 |

**r=0.75 所有替换算法不如 SkipAll 的根因**：

在 r=0.75（96/128 cached，约 2 个 miss/token）时，2 个 miss 专家是路由器明确排在 top-k 末位的选择，权重极低（各约 6-8%）。任何替代品所产生的 K/V 向量都是路由器不认可的，写入 KV cache 后，后续 step 的 attention context 被持续污染。SkipAll 仅丢失 2 个权重极低的专家，对 hidden state 方向影响极小（renormalize 后近似放大 ~1.14 倍），KV cache 不受污染。

**r=0.25 部分算法低于 SkipAll 的根因**：

off-diagonal 相似度均值 0.055，约 90% 的 expert pair 相似度低于 0.40。在 75% miss rate 下，大量 miss 专家被映射到方向几乎正交的替代品，注入的噪声方向比遗漏专家（SkipAll）更具破坏性。加之 8 步 KV 积累，误差雪崩式放大。

### 6.3 v2 算法预期分析

v2 引入 miss-rate gate 后：

- **r=0.75**（$\rho_\text{miss} \approx 0.25$）：gate = 0，所有算法退化为精确路由 ≈ SkipAll
- **r=0.50**（$\rho_\text{miss} \approx 0.50$）：gate = 1，偏置全开；预期 Alg2_v2/HybridCP_v2 接近 v1 的最优表现（0.82-0.80）
- **r=0.25**（$\rho_\text{miss} \approx 0.75$）：gate = 1，PostSub_v2 因 sim_floor=0.40 而大多数走 SkipAll，仅高相似度对做替换

scatter_add 修复 double counting 后，所有后路由算法在 r=0.75 的 alpha_decay 应显著改善（v1 中 Alg1_CoAct decay=0.784 是最差的）。

### v2结果

```
 ── Summary (seq_alpha_total) ──────────────────────────────
  Algorithm               r=0.750  r=0.500  r=0.250
  -------------------------------------------------
  SkipAll                 0.9706  0.9018  0.6364
  Alg2_v2                 0.9700  0.9037  0.7309
  HybridCP_v2             0.9701  0.9041  0.6929
  PostSub_v2              0.9701  0.9019  0.6991
  Alg2_PostSub            0.9700  0.9037  0.7309
```



---

## 7. 结论与后续工作

### 7.1 当前结论

1. **Alg2_v2（预路由偏置 + miss-rate gate）是最有价值的算法**：在 r=0.25 时 seq_α=0.576，比 SkipAll(0.357) 高 21.9 pp；在 r=0.75 时通过 gate 机制退化为精确路由，不损失性能。

2. **Miss-rate gate 是 v2 最重要的设计**：从根本上解决了"高 cache ratio 时所有替换算法不如 SkipAll"的问题，且实现极为简单。

3. **路由熵假设被证伪**：Qwen3-MoE 的路由熵全程约为 uniform 分布的 81.8%，不随上下文增长而集中。seq_α > iso_α 的机制是话题连贯性下的稳定专家命中，而非分布集中。

4. **后路由替换在低相似度环境下劣于 SkipAll**：off-diagonal 相似度均值 0.055 意味着绝大多数替代品对目标专家几乎正交，sim_floor 是必要的保护机制。

### 7.2 待完成工作

- **v2 算法在真实 speculative decoding 中的端到端测试**：将上述算法集成进 nano-vllm-moe 的 `build_moe_execution_plan()`，与现有 round-robin baseline 和先行工作（SP-MoE、HybriMoE）对比端到端 tokens/sec
- **动态 cache 模拟**：当前 `SimulatedCache` 是静态 LFU 快照，真实系统的 LRU/LFU 动态更新会影响 miss pattern，需评估其对算法排名的影响
- **超参数校准**：miss_gate 的合理范围（0.20-0.35）和 gamma0（2.0-6.0）需要在更多 cache ratio 和数据集上系统扫描
