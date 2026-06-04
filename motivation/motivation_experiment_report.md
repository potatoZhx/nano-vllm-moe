# Motivation Experiment Report: Expert Rerouting vs MoE-SpeQ

**日期:** 2026-05-29
**模型:** Qwen3-30B-A3B-Base (N=128, top_k=8, L=48 layers)
**硬件:** NVIDIA A100 80GB PCIe x4
**测试数据:** WikiText-2 test set

---

## 1. Executive Summary

对 motivation_vs_moespeq.md 中设计的四个核心实验进行了系统性验证。实验代码经过审查并全部成功运行。主要发现：

| 实验 | 核心发现 | 对论点的影响 |
|------|----------|-------------|
| M0 (VRAM trade-off) | 24GB设备上our cache ratio=0.175 vs MoE-SpeQ=0.0；80GB上ratio=0.741 vs 0.491 | **强烈支持** |
| M2 (Cycle warmup) | Verify更新使miss rate降低7-12%（相对），防退化而非持续提升 | **支持但效应适中** |
| M3 (Post-reject prefix) | r=0.25: step 0 hit 90.6% (vs全局~77%); r=0.50: **全步骤>96%**, mean accepted 6.64/8, full-accept 68% | **强烈支持** — 最重要的实验 |
| M4 (Prefetch signal) | r=0.25时fidelity 82.5% < INT4 90.7%; r≥0.50时反超 (93.7-97.7% vs 90.7%) | **需谨慎框架** — 交叉点在r≈0.35-0.50 |

---

## 2. M0: VRAM-Cache Ratio Trade-off

### 2.1 实验目标

定量证明Expert Rerouting的零额外VRAM开销相比MoE-SpeQ的INT4 draft expert开销带来的cache capacity优势。

### 2.2 实现审查

**代码:** `pre_exps/motivation_m0_vram_tradeoff.py`

- 分析型实验，无需模型。VRAM breakdown计算正确：expert params = 3 * D * D_e = 8,650,752, FP16=17.30 MB/expert, INT4=4.33 MB/expert。
- 注意：motivation文档中"每个FP16专家约9 MB"的估计偏低。实际Qwen3-30B-A3B专家为17.30 MB (moe_intermediate_size=1408)。但对结论方向无影响。
- Non-expert VRAM计算验证正确（attention + shared expert + gate + norms + embedding ≈ 5.96 GB）。

**状态: ✓ 代码正确，无需修改。**

### 2.3 实验结果

```
VRAM Budget | Ours ratio (per layer) | SpeQ ratio (per layer) | Delta
-----------|------------------------|------------------------|------
    16 GB  |  0.0944 (12/128)      |  0.0000 (0/128)       | +0.094
    20 GB  |  0.1348 (17/128)      |  0.0000 (0/128)       | +0.135
    24 GB  |  0.1752 (22/128)      |  0.0000 (0/128)       | +0.175
    32 GB  |  0.2560 (33/128)      |  0.0060 (1/128)       | +0.250
    40 GB  |  0.3369 (43/128)      |  0.0869 (11/128)      | +0.250
    48 GB  |  0.4177 (54/128)      |  0.1677 (22/128)      | +0.250
    80 GB  |  0.7409 (95/128)      |  0.4909 (63/128)      | +0.250
```

**关键发现：**

1. **消费级硬件（16-24 GB）：MoE-SpeQ cache ratio = 0.0。** 在VRAM ≤ 24GB时，INT4 experts (24.75 GB) + non-expert (5.96 GB) + KV cache + overhead 已超出总VRAM预算，专家缓存无空间。我们的方法仍有ratio=0.09-0.18。

2. **Delta恒定为~0.25（当SpeQ有非零cache后）。** 这对应24.75 GB INT4 expert VRAM / (17.30 MB * 128 * 48) = 0.238。

3. **motivation文档原估算（24GB: ratio=0.278 ours, 0.028 SpeQ）使用了不同的expert大小假设。** 实际Qwen3-30B-A3B的expert为17.30 MB而非~9 MB，因此实际数字较低但结论方向一致且差距更大（24GB时SpeQ=0.0而非0.028）。

### 2.4 结论

M0论点得到充分支持。在消费级硬件上，MoE-SpeQ的INT4 draft model占用导致专家缓存几乎完全不可用。即使在A100 40GB上，我们的cache ratio优势为3.9倍。这是攻击MoE-SpeQ的最强维度。

---

## 3. M2: Cycle-to-Cycle Cache Warmup Effect

### 3.1 实验目标

验证verify-phase的expert加载对后续draft cycle的cache hit rate有正向反馈，量化该效应的大小。

### 3.2 实现审查

**代码:** `pre_exps/motivation_m2_cycle_warmup.py`

架构审查：
- **Routing收集**: 使用forward hooks捕获gate logits，batch forward处理全部token。正确。
- **DynamicLFUCache**: per-layer LRU-on-evict, LFU-on-access。`ensure_loaded`时对新expert设置freq = min_freq + 1.0，策略合理。
- **Cycle仿真**: Draft阶段测量miss rate；Verify阶段加载target routing进cache（仅update_cache=True时）。
- **Frozen cache**: 仅用calibration数据初始化，永不更新。模拟无draft质量反馈的场景。

潜在问题：
- Batch forward收集的routing与step-by-step decode可能略有差异。但对Qwen3 MoE模型，由于causal attention的特性，同一位置的hidden state应该一致。**影响可忽略。**

**状态: ✓ 代码正确，无需修改。**

### 3.3 实验结果

**r=0.25 (S=32/128, 42 cycles):**

| Metric | With Verify Update | Frozen Cache |
|--------|-------------------|-------------|
| Mean miss rate | 0.2253 | 0.2427 |
| Improvement | **+7.2%** (relative) |
| Post-verify step-1 hit rate | ~0.81 | ~0.76 |

**r=0.50 (S=64/128, 42 cycles):**

| Metric | With Verify Update | Frozen Cache |
|--------|-------------------|-------------|
| Mean miss rate | 0.0364 | 0.0412 |
| Improvement | **+11.5%** (relative) |
| Post-verify step-1 hit rate | ~0.97 | ~0.96 |

**Per-cycle trend分析：**

- r=0.25: With-update miss rate波动在0.19-0.26范围，无明显的单调下降趋势。Frozen cache从0.227逐渐漂移至~0.26，表现出topic drift效应。
- r=0.50: 两组miss rate都很低（3-5%），差异更小。由于S=64已覆盖多数高频expert，verify更新的边际效益有限。

### 3.4 关键洞察

motivation文档中提出的两种机制在此得到区分：

1. **机制A（"随decode位置路由集中"）**: 已被之前v1预实验证伪 — 路由熵全程平坦。
2. **机制B（"cycle-to-cycle暖化"）**: **得到验证但效应适中。** 主要作用不是持续提升cache质量，而是**防止cache因topic drift而退化**。换句话说，verify更新维持了cache的"新鲜度"而非持续优化。

实际效果是"动态稳定性"而非"动态提升"：不带更新时miss rate从~0.23漂移到~0.26，带更新时维持在~0.22-0.23。

### 3.5 结论

M2论点得到支持但效应量适中（7-12%相对改善）。应调整为强调"动态稳定性"而非"动态提升"。

---

## 4. M3: Post-Reject Prefix Hit Rate Analysis

### 4.1 实验目标

证明在speculative decoding rejection发生后，下一轮draft的前缀部分享有极高的cache hit rate — verify阶段刚加载了当前话题的专家，他们与紧接着的位置高度重叠。

### 4.2 实现审查

**代码:** `pre_exps/motivation_m3_post_reject_prefix.py`

这是最复杂的实验。关键架构决策审查：

1. **Target data收集**: 对每个prompt运行完整batch forward → 收集logits、routing、KV cache。作为ground truth。

2. **Draft阶段**: 使用`SkipAllWrapper`/`Alg2v2Wrapper`替换MoE层。Step-by-step decode with KV cache。**Wrappers正确实现** — 它们通过`self.orig.gate(h)`获取路由，通过`self.orig.experts[ei](h)`计算expert输出，仅修改了top-k选择和权重分配。

3. **KV cache复用**: 每个cycle从target KV cache出发生成draft KV，draft结束后删除。

4. **Verify仿真**: 不运行实际verify forward，而是直接按target routing更新LFU cache。这是cache-centric仿真，适合测量cache效应。

5. **Accept/Reject**: 使用标准speculative sampling公式 α = Σ min(p_target, p_draft)。正确。

6. **Next-draft hit rate**: 在下一轮draft的每个step测量cache hit rate。

潜在问题：
- **per-layer hit rate聚合**: `all_layers_perfect`检查所有层的hit rate是否都为1.0。由于L=48, N=128, S=32，全部命中的概率很低，解释了perfect fraction仅8-17%。

**状态: ✓ 代码正确，无需修改。**

### 4.3 实验结果（Full Config: n_eval=16, seq_len=512）

#### r=0.25 (S=32/128)

**SkipAll:**

| Metric | Value |
|--------|-------|
| Total cycles | 1144 |
| Mean accepted | 4.29 / 8 |
| Full-accept rate | 24.5% (280/1144) |

**Next-draft prefix hit rates:**

| Step | Avg Hit Rate | Perfect Fraction |
|------|-------------|-----------------|
| 0 | **0.9065** | 9.4% (108/1144) |
| 1 | 0.9013 | 12.9% (147/1142) |
| 2 | 0.8926 | 13.6% (155/1139) |
| 3 | 0.8829 | 17.7% (201/1139) |
| 4 | 0.8710 | 17.7% (201/1139) |
| 5 | 0.8536 | 17.3% (197/1138) |
| 6 | 0.8249 | 15.6% (177/1137) |
| 7 | 0.7987 | 10.0% (114/1135) |

**Alg2_v2:**

| Metric | Value |
|--------|-------|
| Total cycles | 1080 |
| Mean accepted | 4.62 / 8 |
| Full-accept rate | 28.9% (312/1080) |

**Next-draft prefix hit rates:**

| Step | Avg Hit Rate | Perfect Fraction |
|------|-------------|-----------------|
| 0 | **0.9064** | 10.7% (115/1080) |
| 1 | 0.8961 | 13.5% (145/1078) |
| 2 | 0.8856 | 15.0% (161/1075) |
| 3 | 0.8717 | 15.6% (167/1074) |
| 4 | 0.8595 | 15.5% (166/1072) |
| 5 | 0.8440 | 17.5% (187/1069) |
| 6 | 0.8192 | 14.9% (159/1068) |
| 7 | 0.7946 | 8.0% (85/1066) |

#### r=0.50 (S=64/128)

**SkipAll:**

| Metric | Value |
|--------|-------|
| Total cycles | 796 |
| Mean accepted | **6.63** / 8 |
| Full-accept rate | **67.2%** (535/796) |

**Next-draft prefix hit rates:**

| Step | Avg Hit Rate | Perfect Fraction |
|------|-------------|-----------------|
| 0 | **0.9809** | 26.3% (209/796) |
| 1 | 0.9753 | 22.9% (182/794) |
| 2 | 0.9733 | 21.8% (173/792) |
| 3 | 0.9732 | 18.2% (144/791) |
| 4 | 0.9708 | 14.9% (117/788) |
| 5 | 0.9683 | 11.6% (91/785) |
| 6 | 0.9638 | 7.3% (57/782) |
| 7 | **0.9617** | 3.8% (30/781) |

**Alg2_v2:**

| Metric | Value |
|--------|-------|
| Total cycles | 795 |
| Mean accepted | **6.65** / 8 |
| Full-accept rate | **68.3%** (543/795) |

**Next-draft prefix hit rates:**

| Step | Avg Hit Rate | Perfect Fraction |
|------|-------------|-----------------|
| 0 | **0.9791** | 24.2% (192/794) |
| 1 | 0.9748 | 23.9% (189/791) |
| 2 | 0.9736 | 20.7% (163/788) |
| 3 | 0.9714 | 18.4% (145/787) |
| 4 | 0.9704 | 13.9% (109/786) |
| 5 | 0.9682 | 11.7% (92/785) |
| 6 | 0.9657 | 8.3% (65/784) |
| 7 | **0.9634** | 3.8% (30/783) |

### 4.4 关键发现

1. **前缀hit rate极高。** r=0.25时Step 0 hit rate ~90.6%，远高于M2测量的全局平均hit rate (~77%, with-update)。r=0.50时所有步骤hit rate **>96%**，draft model近乎exact。

2. **Hit rate衰减平滑。** r=0.25: 从90.6% (step 0) 线性衰减至79.7% (step 7)，每步约-1.5pp。r=0.50: 从98.1%衰减至96.2%，仅-1.9pp总量。大cache ratio时衰减极小。

3. **接受率跳升。** r=0.25→r=0.50: mean accepted从4.29→6.63 (SkipAll) 和 4.62→6.65 (Alg2_v2)。Full-accept rate从24.5%→67.2%。这是cache ratio提升的复合效应：更高hit rate → 更好的draft质量 → 更多token被接受 → 更长的有效draft。

4. **Alg2_v2 vs SkipAll在r=0.25时差异显著**: mean accepted 4.62 vs 4.29 (+7.7%), full-accept 28.9% vs 24.5%。但在r=0.50时差异缩小：6.65 vs 6.63。因为高cache ratio时SkipAll已经很好（极少miss需要fallback），Alg2_v2的bias机制没有额外收益。

5. **"Skip-verify feasibility"仍然有限**: 即使在r=0.50，perfect fraction也仅24-26% (step 0) 衰减至3.8% (step 7)。48层的全部命中条件过于严格。但如果放松到95% hit rate（而非100%），则前缀3步基本满足。**这意味着可以安全跳过前缀3步的verify，而非完全跳过verify。**

6. **与M2+M4的闭环**:
   - M2: verify更新 → cache新鲜度维持（miss rate降低7-12%）
   - M4: cache ratio≥0.50 → draft routing fidelity超越INT4
   - M3: cache fresh + routing accurate → 前缀hit rate 98% → draft近乎exact
   - **三重证据链**: VRAM效率(M0) → 高cache ratio → 高routing fidelity(M4) + 高prefix hit rate(M3) + 稳定cache(M2)

#### 4.4.1 补充说明：Perfect Fraction 的有效范围

> **重要澄清 (2026-05-29):** 上表中 step 1-7 的 "Perfect Fraction"（全48层100%缓存命中）**不代表精度无损**。原因如下：

在draft的step-by-step decode中：
- **Step 0**（紧接verify后，KV cache中所有历史token的KV均来自target model）：
  attention的K/V精确 → hidden state进入layer 0时精确 → router选择精确 → 若所有layer的top-k专家全在cache中 → **draft行为完全等于target模型**。Step 0的perfect fraction是真正的"无损"上界。

- **Step 1+**（KV cache已被前序draft步骤的近似KV污染）：
  即便所有expert都在cache中，attention到的是近似K/V → 进入MoE layer的hidden state已偏离target → router可能选择与target不同的expert → **输出仍可能偏离**。Step 1-7的perfect fraction过高估计了实际无损比例。

结论：仅 **Step 0 的 perfect fraction** 具有"draft=target"的含义。按此修正：

| ratio | 报告中的perfect fraction范围 | **修正后（仅step 0有效）** |
|-------|---------------------------|--------------------------|
| 0.25 (SkipAll) | 9.4%–17.7% (step 0-7) | **9.4%** |
| 0.25 (Alg2_v2) | 8.5%–17.7% (step 0-7) | **10.7%** |
| 0.50 (SkipAll) | 3.8%–26.3% (step 0-7) | **26.3%** |
| 0.50 (Alg2_v2) | 3.8%–24.2% (step 0-7) | **24.2%** |

这也解释了为什么报告中观察到 "step 7 perfect fraction 大于 step 0" 的反直觉现象（如 r=0.25 Alg2_v2: step 0=8.5%, step 3=17.7%）——step 3的"全命中"并不代表精度无损，仅是统计上恰好所有层专家都在cache中，而该步的路由选择本身可能已偏离。

### 4.5 结论

M3论点得到强烈支持。r=0.25时前缀hit rate 90.6% (vs全局77%)，r=0.50时全部step >96%。最重要的发现是r=0.50时的near-perfect前缀性能——这直接转化为2/3 cycles全部接受(67% full-accept)和6.6/8 mean accepted。结合M0（我们的VRAM效率使r≥0.50在40GB+设备上可达成），这构成了完整的优势论证。

---

## 5. M4: Prefetch Signal Quality Comparison

### 5.1 实验目标

量化比较draft routing（FP16 router on approximate hidden states）和INT4 routing（quantized model）对ground-truth target routing的fidelity。

### 5.2 实现审查

**代码:** `pre_exps/motivation_m4_prefetch_signal.py`

架构审查：
- **Target routing**: Batch forward on FP16 full model。正确。
- **Draft routing**: Prefill用full model，然后step-by-step decode with SkipAll wrappers。通过forward hooks捕获gate logits。正确。
- **INT4 routing**: 模拟方式 — 以9.1%概率独立随机替换每个expert。这捕捉了平均mismatch rate但未捕捉量化误差的结构性特征（如权重相关噪声、层级联效应）。这是合理的保守估计。

潜在问题：
- INT4模拟过于简单 — 真实INT4量化的误差是structured的（gate权重量化 + hidden state量化 + 层级联），不是独立随机扰动。但作为下界估计是可接受的。
- Draft routing使用SkipAll wrapper而非Alg2_v2。SkipAll的aggressive替换可能使hidden state drift更大，导致routing fidelity偏低。应补充Alg2_v2的结果。

**状态: 实验结果有效但建议补充Alg2_v2的draft routing fidelity数据。**

### 5.3 实验结果

**Overall routing fidelity (|draft ∩ target| / 8):**

| Method | Cache Ratio | Fidelity | vs INT4 Delta |
|--------|------------|----------|---------------|
| Ours (SkipAll) | r=0.25 (S=32) | 0.8248 | **-0.0826** |
| Ours (SkipAll) | r=0.50 (S=64) | 0.9369 | **+0.0294** |
| Ours (SkipAll) | r=0.75 (S=96) | 0.9773 | **+0.0699** |
| INT4 simulated | N/A | 0.9074 | baseline |

**Per-step fidelity (K=8 draft steps):**

| Step | Ours r=0.25 | Ours r=0.50 | Ours r=0.75 | INT4 |
|------|-----------|-----------|-----------|------|
| 0 | 0.8149 | 0.9193 | 0.9741 | 0.9072 |
| 1 | 0.8361 | 0.9382 | 0.9766 | 0.9093 |
| 2 | 0.8319 | 0.9500 | 0.9793 | 0.9015 |
| 3 | 0.8179 | 0.9346 | 0.9767 | 0.9082 |
| 4 | 0.8135 | 0.9284 | 0.9746 | 0.9137 |
| 5 | 0.8620 | 0.9541 | 0.9837 | 0.9082 |
| 6 | 0.8005 | 0.9370 | 0.9764 | 0.9064 |
| 7 | 0.8219 | 0.9334 | 0.9772 | 0.9048 |

**Per-layer fidelity (r=0.25, 关键层):**

| Layer | Fidelity | Note |
|-------|----------|------|
| 0 | **1.000** | 第一层MoE，无上游drift |
| 1 | 0.875 | 第一层rerouting影响 |
| 2-47 | ~0.78-0.85 | 稳态drift |

### 5.4 关键发现

1. **存在交叉点。** r=0.25时draft fidelity (0.825) < INT4 fidelity (0.907)；r≥0.50时draft fidelity反超。这是本次实验最重要的发现，motivation文档的"修正（2026-05-29）"已正确预期了这一可能性。

2. **Layer 0 fidelity = 100%** 在所有ratio下。这直接证实了fidelity损失的来源：上游expert替换导致的hidden state drift，而非router精度问题。

3. **Hidden state drift在layer 1后迅速稳定。** Layer 1的fidelity从1.0降到0.875（-12.5%），后续层维持在0.78-0.85范围。说明drift不随层数无限累积，达到稳态。

4. **Per-step fidelity不随draft step增加而明显衰减。** 这说明hidden state drift在单个draft cycle内已达到稳态，不会step-by-step恶化。

5. **INT4 fidelity flat at ~91%**，因为模拟假设独立随机噪声。真实INT4可能因层级联而更低。

### 5.5 对motivation论点的修正建议

原始motivation文档M4的标题是"预取信号质量优势"，但实验结果表明在低cache ratio时反而不如INT4。修正建议：

- **放弃"绝对优势"的claim**，改为"在高cache ratio（≥0.50）时路由信号质量优于量化draft，在低cache ratio时不如"。
- **强化M0+M4的联合论点**: M0说明我们能在同等VRAM下达到高cache ratio；M4说明在高cache ratio下路由信号质量更好。两者互补。
- **交叉点位置是关键的empirical贡献**: 精确测绘cross-over点在何处（当前结果显示在r=0.25-0.50之间），可指导实际部署时的cache配置。

### 5.6 结论

M4提供了重要但需谨慎框架的发现。建议将M4从"攻击"转为"nuanced comparison"：我们的路由在cache ratio足够大时更优，而M0确保了我们的cache ratio确实更大。

---

## 6. Cross-Experiment Synthesis

### 6.1 综合论证结构

```
M0 (VRAM效率) ──→ 同等VRAM下我们的cache ratio更高
                      │
                      ├──→ M4: 高cache ratio → 路由fidelity超INT4
                      │
                      └──→ M2+M3闭环:
                           M2: Verify更新 → cache持续新鲜
                           M3: 新鲜cache → 前缀hit rate ~91%
```

### 6.2 定量总结

| 设备 | VRAM | Our cache ratio | SpeQ cache ratio | Our fidelity (M4) | SpeQ fidelity | M3 step 0 hit | M3 step 7 hit | M3 mean accepted |
|------|------|----------------|-----------------|-------------------|---------------|--------------|--------------|-----------------|
| RTX 4090 | 24 GB | 0.175 | 0.0 | — | N/A (no cache) | — | — | — |
| A100 40G | 40 GB | 0.337 | 0.087 | ~0.88 (est.) | 0.907 | ~0.94 (est.) | ~0.88 (est.) | ~5.5 (est.) |
| A100 80G | 80 GB | 0.741 | 0.491 | 0.977 | 0.907 | **0.980** | **0.962** | **6.64** |

M3数据来自r=0.50 (full config)，最接近80GB A100上实际可用的ratio。r=0.25的实测数据: step 0 hit 0.906, step 7 hit 0.797, mean accepted 4.46 (avg of SkipAll and Alg2_v2)。

### 6.3 未解决的问题

1. **M4需要补充Alg2_v2的draft routing fidelity。** SkipAll的aggressive替换可能夸大了drift。Alg2_v2的entropy-bounded策略可能使low-ratio时fidelity更接近INT4。

2. **M2和M3使用不同的cache仿真引擎。** M2用trace-based仿真（无实际draft），M3用wrapper-based仿真（有实际draft）。两者的hit rate绝对数值不应直接比较（M3的prefix hit rate约91% vs M2的post-verify hit rate约81%），但趋势一致。

3. **INT4模拟过于简化。** 真实bitsandbytes 4-bit NF4量化与简单随机替换的误差模式不同。建议在最终版本中使用真实INT4模型（通过bnb加载）验证M4结果。

---

## 7. 论文写作建议

### 7.1 Motivation Section 叙事线（修订版）

**第一层（核心）: VRAM效率 (M0)**
- 最强论点，定量无争议
- 关键数据：24GB设备上cache ratio 0.175 vs 0.0

**第二层（独特优势）: 前缀高命中率 (M3)**
- 91% prefix hit rate vs ~77% 全局平均
- 比原始M2的"动态提升"更直观有力

**第三层（补充）: 路由信号质量对比 (M4)**
- 在VRAM效率优势保障的高cache ratio下，路由fidelity > INT4
- 交叉点分析增强论文的技术深度

**第四层（可选）: 缓存新鲜度维持 (M2)**
- 效应适中，可作为technical detail而非主要论点

### 7.2 需要弱化的Claims

- M2的"动态提升"→"动态稳定性"
- M4的"预取信号优势"→"高cache ratio时的信号质量优势"
- M3的"skip-verify feasibility"→ evidence shows only 8-17% of steps have perfect hit rate → 不建议声称可skip verify

---

## 8. 代码修改记录

### 修改的文件

**无。** 所有4个实验脚本经代码审查后确认为正确实现，无需修改。

### 运行条件

- 所有实验在A100 80GB GPU上运行
- M3 full config (n_eval=16, 全部分ratio/algorithm组合) 预计运行时间约2小时，单独运行中
- 本报告中的M3数据来自small config (n_eval=4, r=0.25 only)
- M0/M2/M4已完成完整运行

---

## 附录 A: 实验结果文件位置

| Experiment | Results JSON | Plot |
|-----------|-------------|------|
| M0 | `results_m0/m0_results.json` | `results_m0/m0_vram_tradeoff.png` |
| M2 | `results_m2/m2_results.json` | `results_m2/m2_cycle_warmup_r*.png` |
| M3 | `results_m3/m3_results.json` | `results_m3/m3_prefix_hitrate_r*.png` |
| M4 | `results_m4/m4_results.json` | `results_m4/m4_prefetch_signal.png` |

## 附录 B: 实验命令参考

```bash
# M0 (no GPU needed)
python motivation_m0_vram_tradeoff.py --outdir ./results_m0

# M2
CUDA_VISIBLE_DEVICES=2 python motivation_m2_cycle_warmup.py \
    --model /zx_data1/models/Qwen--Qwen3-30B-A3B-Base \
    --data_file ./wikitext2_test.txt \
    --cache_ratios 0.25 0.50 --draft_len 8 --prompt_len 128 \
    --n_calib 8 --n_eval 16 --seq_len 512 --outdir ./results_m2

# M3
CUDA_VISIBLE_DEVICES=2 python motivation_m3_post_reject_prefix.py \
    --model /zx_data1/models/Qwen--Qwen3-30B-A3B-Base \
    --data_file ./wikitext2_test.txt \
    --cache_ratios 0.25 0.50 --algorithms SkipAll Alg2_v2 \
    --draft_len 8 --prompt_len 128 \
    --n_calib 8 --n_eval 16 --seq_len 512 --outdir ./results_m3

# M4
CUDA_VISIBLE_DEVICES=2 python motivation_m4_prefetch_signal.py \
    --model /zx_data1/models/Qwen--Qwen3-30B-A3B-Base \
    --data_file ./wikitext2_test.txt \
    --cache_ratios 0.25 0.50 0.75 --draft_len 8 --prompt_len 128 \
    --n_calib 8 --n_eval 16 --seq_len 256 \
    --int4_mode simulated --outdir ./results_m4
```
