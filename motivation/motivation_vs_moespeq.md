# Expert Rerouting vs MoE-SpeQ: Motivation 分析与实验设计

**日期:** 2026-05-28

---

## 1. 两种方案的核心架构对比

| 维度 | MoE-SpeQ | Expert Rerouting (Ours) |
|------|----------|------------------------|
| Draft 模型来源 | INT4 量化的目标模型 | 目标模型本身 + 专家替换/跳过 |
| Draft 额外 VRAM | INT4 专家权重（占用显著 VRAM） | **零额外开销** |
| 参数共享 | 非专家参数 + KV cache | 全部参数 + KV cache + 专家缓存 |
| Draft 路由准确性 | ~90.9% expert fidelity（量化误差） | 路由器精确（FP16），仅执行层近似 |
| Draft 加速手段 | 量化 kernel (Marlin + fuseMoE) | CUDA graph replay（零 CPU 计算/PCIe） |
| 对目标模型的要求 | 目标必须是 FP16（才能做 INT4 draft） | **无限制**（FP16/INT8/INT4 均可） |
| Draft 质量随时间变化 | 固定（量化误差是静态的） | **动态提升**（缓存更新 → miss rate 下降） |
| Verify 后的恢复 | 无特殊优势 | 前缀高命中率复用 |

---

## 2. Motivation 分析：可攻击 MoE-SpeQ 的维度

### M0: VRAM 节省 → 更大专家缓存（已知，最核心）

**论点**：MoE-SpeQ 的 INT4 专家权重占用大量 VRAM，这些 VRAM 本可用于缓存更多 FP16 专家，直接降低 miss rate。

**定量估算（Qwen3-30B-A3B）**：

- 总参数 30B，其中专家参数约 27B（54 GB FP16）
- INT4 专家：约 **13.5 GB**
- 每个 FP16 专家约 9 MB（per layer）
- 13.5 GB 可额外缓存 **~32 experts/layer**，即 cache ratio 提升 **+0.25**

**RTX 4090 (24 GB) 预算分析**：

| | 可用于缓存的 VRAM | 每层缓存数 | Cache ratio |
|--|-------------------|-----------|-------------|
| Ours | ~15 GB | ~36 | 0.278 |
| MoE-SpeQ | ~1.5 GB | ~4 | 0.028 |

这不是微小差异——在 24GB 设备上，MoE-SpeQ 几乎无法维持有意义的专家缓存。即使在 40GB A100 上（MoE-SpeQ 的评估硬件），INT4 权重仍然挤占了大量缓存空间。

**攻击力度**：⭐⭐⭐⭐⭐（定量可证、无争议）

### M1: 对量化目标模型的兼容性（已知）

**论点**：MoE-SpeQ 要求目标模型为 FP16，以便 INT4 draft 保持足够精度。如果目标模型本身已经是 INT4/INT8（这是消费级部署的常见选择），则无法再对 draft 做进一步量化。

Expert Rerouting 是正交于量化的——无论目标模型是 FP16/INT8/INT4，rerouting 的 GPU residency 约束和 speculative sampling 保证均成立。这意味着可以将量化和 rerouting 组合使用，获得双重收益。

**攻击力度**：⭐⭐⭐⭐（对实际部署场景影响大）

### M2: Draft Model 动态提升（用户提出）

**论点**：由于 draft model 与目标模型共享专家缓存，随着 verify 阶段加载正确专家进缓存，draft model 的质量会逐步提升。

**前置实验结论与修正**：

v1 预实验证实了 H2：连续 token 共享约 4/8 个 top-k 专家（Jaccard=0.343，远高于随机期望 0.032），但**效应量随 decode 位置增加很小**（从 Q1 的 4.07/8 到 Q4 的 4.18/8，仅 +2.7%）。路由熵全程平坦（3.967，= 81.8% × log(128)），这是 load-balancing loss 的设计结果。

因此，"随 decode 位置增加接受率持续提升"的强版本论点在 Qwen3-30B-A3B 上**不成立**。但需要区分两个不同的机制：

**机制 A（弱，位置效应）**：随 decode 位置增长，路由分布趋于集中 → 已被证伪。

**机制 B（强，cycle-to-cycle 效应）**：每轮 verify 加载正确专家进缓存 → 下一轮 draft 的 cache 状态更匹配当前话题 → miss rate 下降。这个机制不依赖路由分布集中，而是依赖 **verify 的缓存更新对 LFU/LRU 的正向反馈**。

机制 B 的关键洞察：连续 token 共享 ~4/8 专家（话题连贯性），verify 阶段精确加载了当前话题需要的专家，这些专家被 LFU 正确保留，在下一个 draft cycle 中自然命中。这是一个 **cycle 级别的暖化效应**，不是 token 级别的。

**与 MoE-SpeQ 的对比**：MoE-SpeQ 的量化误差是固定的——不会因为 decode 进行而改善。MoE-SpeQ 也有缓存暖化效应（通过 ELB prefetch），但 prefetch 的准确性受限于 INT4 路由误差（~9% mismatch）。

**修正（2026-05-29）**：之前的表述"我们的原始路由 metadata 是精确的"不准确。Expert rerouting 后 hidden state 已偏离目标模型，下游层的路由器虽然使用 FP16 权重（无量化噪声），但其**输入**（hidden state）是近似的。因此我们的路由误差来源是 hidden state drift（间接，随 cache ratio 增大而减小），而 MoE-SpeQ 的误差来源是 gate 权重量化 + hidden state 量化（直接，固定）。两者的误差大小需要通过 M4 实验量化比较。

**攻击力度**：⭐⭐⭐（机制 B 有效但需要精心设计实验与 MoE-SpeQ 区分）

### M3: 拒绝后前缀高命中率复用（用户提出）

**论点**：假设 draft 长度 K=10，在位置 j=7 被拒绝。Verify 阶段已精确加载了位置 1-8 所需的全部专家。下一轮 draft 从位置 8 开始，前几步的专家需求与刚 verify 加载的专家高度重叠（因为话题连贯性），因此前缀部分的 cache hit rate 极高，draft model 近似等于原始模型。

**理论分析**：

设 verify 加载了 K 个 token 的专家集合 $V = \bigcup_{t=1}^{K} \mathcal{S}^\ell(h_t)$。下一轮 draft 的第 1 步需要 $\mathcal{S}^\ell(h_{K+1})$。根据 H2 结论，$|\mathcal{S}^\ell(h_{K+1}) \cap \mathcal{S}^\ell(h_K)| \approx 4$（top-k=8），即约 50% 直接命中上一个 token 的专家。但 $V$ 包含了 K 个 token 的**全部**唯一专家，命中率应远高于 50%。

对 Qwen3-30B-A3B (N=128, k=8)，K 个 token 激活的唯一专家数：

$$\mathbb{E}[|V|] = 128 \times (1 - (1 - 8/128)^K)$$

| K | E[\|V\|] | 若 cache S=32, hit rate |
|---|--------|------------------------|
| 3 | 22.5 | min(22.5, 32)/22.5 = 100% |
| 5 | 35.1 | 32/35.1 = 91.2% |
| 8 | 50.6 | 32/50.6 = 63.2% |

加上 LFU 保留的高频专家（未被 verify 覆盖的部分），实际命中率更高。关键是**verify 恰好加载了语义相关的专家**。

**前缀跳过验证（Verify Prefix Skip）的可能性**：

如果前缀 m 步的 draft 输出与精确输出几乎一致（因为几乎零 miss），理论上可以跳过这 m 步的 verify。但这需要严格的理论保证（需要证明 KL 散度 < ε），在 project_summary 中被列为 "Benefit 3, 最投机的声明"。

**更保守但可靠的论点**：不声称可以跳过 verify，而是强调前缀高命中率 → 前缀 draft 质量高 → 整体接受率提升 → 有效吞吐改善。这与 MoE-SpeQ 形成对比：MoE-SpeQ 的每一步 draft 都有固定的 ~9% 路由误差，无法利用 verify 的缓存更新效应。

**攻击力度**：⭐⭐⭐⭐（机制清晰，可实验验证，与 M2 形成闭环论证）

### M4: 预取信号质量优势（新增）

**论点**：Expert Rerouting 的路由决策由 FP16 路由器在 draft hidden state 上计算。MoE-SpeQ 的 ELB 基于 INT4 模型的路由，有 ~9% 的 mismatch。

**修正（2026-05-29）**：之前声称"原始路由无误差"不成立。Expert rerouting 后 hidden state 偏离目标模型，FP16 路由器在近似 hidden state 上的 top-k 选择与目标模型的选择存在差异。两种方法的误差来源不同：

- MoE-SpeQ: 路由误差 = INT4 gate 权重量化噪声 + INT4 hidden state 量化偏差（两者叠加，且逐层级联）
- Ours: 路由误差 = FP16 路由器在近似 hidden state 上的偏差（仅 hidden state drift，无权重量化噪声）

**影响分析**：

预取的价值取决于预测准确性。设 K 步 draft 后需要 verify 的专家总集合为 $V_\text{need}$，预取集合为 $V_\text{prefetch}$：

- MoE-SpeQ: $|V_\text{prefetch} \cap V_\text{need}| / |V_\text{need}| \leq 0.909^K$（逐层 mismatch 累积）
- Ours: 取决于 cache ratio 和 rerouting 算法带来的 hidden state drift 大小

MoE-SpeQ 的误差来源是双重的（量化权重 + 量化 hidden state），且一旦某层路由出错，后续层 hidden state 偏移导致**误差级联**。我们的误差仅来自 hidden state drift，在高 cache ratio 时（miss 少，drift 小）应显著优于 INT4 routing；在低 cache ratio 时差距需实验验证。

**攻击力度**：⭐⭐⭐（与 M0 互补，需要 M4 实验量化两者的实际差异）

### M5: 细粒度 MoE 的扩展性优势（新增）

**论点**：MoE-SpeQ 在 Qwen1.5-MoE (N=60) 和 DeepSeek-V2-Lite (N=64) 上评估。Qwen3-30B-A3B 有 N=128 experts/layer。随着 N 增大：

1. INT4 专家总量线性增长 → VRAM 压力更大
2. 路由空间更大 → 量化路由误差更容易导致 mismatch
3. 但 rerouting 的替代池也更大 → 找到高相似度替代品的概率更高

MoE-SpeQ 在论文中承认："MoE-SpeQ's potential is still constrained by the draft model's memory overhead"。在 N=128 的细粒度 MoE 上，这个约束更加严重。

**攻击力度**：⭐⭐⭐（趋势性论点，增强 M0）

### M6: CUDA Graph 兼容性与 Draft 速度（新增）

**论点**：Expert Rerouting 在 `draft_top_c=0` 时支持完整 CUDA graph replay，draft 每步只需 ~0.5ms（project_summary 估算）。MoE-SpeQ 需要运行量化 kernel（fuseMoE），虽然快于 FP16 但仍有 kernel launch overhead，且 ablation 显示去掉 fused kernel 后性能降至 68.2%。

更重要的是：MoE-SpeQ 的 draft 阶段仍需要加载 INT4 专家（如果不全在 GPU），或者在全 GPU 时受限于量化 kernel 的算术密度（MoE-SpeQ 论文中提到 fine-grained MoE 场景下 Marlin 甚至慢于 FP16 PyTorch）。

CUDA graph replay 消除了所有 kernel launch overhead 和 Python 层调度开销，在 decode 阶段（batch=1）的优势尤为明显。

**攻击力度**：⭐⭐⭐（系统层面优势，但需要端到端数字支撑）

### M7: Draft 质量的在线可观测性（用户的 Motivation 1，暂不实验）

Expert Rerouting 的 draft 质量信号（miss rate, 替代相似度, 累积误差）是**透明可计算的**——每一步 draft 的误差来源完全已知（哪个专家被替换、用了什么替代品、相似度多少）。这使得动态 K 和 early-stop 决策可以基于精确信号。

MoE-SpeQ 的 Amortization Roofline Model 依赖离线 profiling 和 EMA 更新的接受率估计，缺乏 per-step 的误差分解能力。

---

## 3. Motivation 优先级与论文叙事建议

### 论文 Motivation Section 建议叙事线

**第一层（核心，1-2 段）**：VRAM 效率优势 (M0)

> "MoE-SpeQ 等基于量化 draft 模型的方案需要在 GPU 上存储 INT4 专家权重，占用宝贵的 VRAM 空间。对于 Qwen3-30B-A3B (N=128, 48 layers)，INT4 专家约占 13.5 GB。在消费级 RTX 4090 (24 GB) 上，这使得可用于专家缓存的空间从 ~15 GB 降至 ~1.5 GB，cache ratio 从 0.28 降至 0.03——几乎没有有效缓存。我们的方法不引入额外显存开销，全部 VRAM 用于专家缓存。"

**第二层（独特优势，1-2 段）**：动态提升与前缀复用闭环 (M2+M3)

> "更进一步，我们观察到 self-speculative 架构具有独特的正反馈效应：verify 阶段精确加载的专家直接更新了 draft model 的专家缓存，使下一轮 draft 的 miss rate 下降。尤其在拒绝发生后，verify 已加载的专家为下一轮 draft 的前缀提供了高命中率，前缀部分的 draft 质量接近精确模型。这种 cycle-to-cycle 的缓存暖化效应在量化 draft 模型中不存在——量化误差是固定的、不随推理进行而改善。"

**第三层（补充，简述）**：量化兼容性 (M1) + 预取信号质量 (M4)

> "此外，该方法与量化正交（目标模型可以是任意精度），且保留精确的原始路由 metadata 用于预取调度，避免了量化路由的 ~9% mismatch 对预取准确性的损害。"

---

## 4. 实验设计

### 实验 M0-Exp: VRAM-Cache Ratio Trade-off

**目的**：定量证明 VRAM 节省转化为更高 cache ratio 和更高接受率。

**方法**：
1. 在 standalone eval framework 中，对不同 cache ratio 运行 rerouting eval
2. 计算 MoE-SpeQ 在同等总 VRAM 预算下能达到的 cache ratio（扣除 INT4 专家显存）
3. 将两者的 seq_α 绘制在同一张图上，x 轴为总 VRAM 预算

**具体步骤**：
```
For VRAM_budget in [16, 20, 24, 32, 40] GB:
    # Ours
    S_ours = (VRAM_budget - VRAM_non_expert - VRAM_kv - VRAM_overhead) / expert_size_fp16 / L
    ratio_ours = S_ours / N
    α_ours = eval_rerouting(ratio=ratio_ours, alg='Alg2_v2')
    
    # MoE-SpeQ (需要 INT4 expert VRAM)
    S_speq = max(0, (VRAM_budget - VRAM_non_expert - VRAM_kv - VRAM_overhead - VRAM_int4_experts)) / expert_size_fp16 / L
    ratio_speq = S_speq / N
    α_speq = 0.909  # MoE-SpeQ 报告的 expert fidelity, 作为接受率上界
    # 或用 eval_rerouting(ratio=ratio_speq, alg='SkipAll') 作为下界参考
```

**输出**：VRAM budget vs seq_α 曲线图，展示在 24-32 GB 设备上我们的方法优势最大。

**预期结论**：在 24 GB 设备上，MoE-SpeQ 几乎无可用缓存（ratio < 0.05），而我们的方法达到 ratio ~0.28，seq_α 差异 > 20pp。

**脚本位置**：`pre_exps/vram_tradeoff_eval.py`

---

### 实验 M2-Exp: Cycle-to-Cycle 缓存暖化效应

**目的**：证明 verify 的专家加载对后续 draft cycle 有正向反馈，且这个效应在量化 draft 中不存在。

**方法**：

**实验 M2a：per-cycle miss rate 趋势**

在 decode-mode 仿真中，模拟多轮 draft-verify cycle：
1. 初始化 LFU cache（基于 prefill 频率）
2. 每轮 draft K 步，记录每步的 miss rate
3. Verify：精确计算原始路由，更新 cache（LFU mark_access）
4. 记录每轮的平均 miss rate 和 seq_α

**关键对比维度**：
- "With verify cache update"：verify 后按原始路由更新 LFU
- "Without verify cache update"（冻结 cache）：模拟静态 draft 模型（类似 MoE-SpeQ 的行为——其 INT4 精度不会因 verify 而改善）

```python
for cycle in range(num_cycles):
    # Draft phase
    miss_rates = []
    for step in range(K):
        token_routing = get_original_routing(hidden_states[pos])
        miss_rate = count_misses(token_routing, cache) / k_active
        miss_rates.append(miss_rate)
        pos += 1
    
    log_cycle_miss_rate(cycle, mean(miss_rates))
    
    # Verify phase - update cache
    if WITH_VERIFY_UPDATE:
        for step in range(K):
            verify_routing = get_original_routing(hidden_states_verify[pos-K+step])
            cache.mark_access(verify_routing)  # LFU update
            cache.load_missing(verify_routing)  # prefetch simulation
```

**预期结果**：
- With verify update: per-cycle miss rate 在前 5-10 个 cycle 显著下降后稳定
- Without update: miss rate 保持平坦或波动
- 差异在 r=0.25-0.50 时最显著

**实验 M2b：连续 token 的专家缓存命中率**

```python
# 在每轮 verify 完成后，测量下一轮 draft 第 1 步的 cache hit rate
for cycle in range(num_cycles):
    verify_and_update_cache(...)
    
    # 下一轮 draft 第 1 步
    next_routing = get_original_routing(hidden_states[next_pos])
    hit_rate_step1 = count_hits(next_routing, cache) / k_active
    log(cycle, hit_rate_step1)
```

**预期结果**：下一轮 draft 第 1 步的 hit rate 显著高于整体平均 hit rate（因为 verify 刚加载了语义相关专家）。

**脚本位置**：`pre_exps/cycle_warmup_eval.py`

---

### 实验 M3-Exp: 拒绝后前缀命中率分析

**目的**：证明在 reject 发生后，下一轮 draft 的前缀有极高 cache hit rate。

**方法**：

**实验 M3a：post-reject 前缀命中率**

```python
for cycle in range(num_cycles):
    # Draft K steps
    draft_tokens, draft_routings = run_draft(K)
    
    # Simulate verify + accept/reject
    reject_pos = simulate_accept_reject(draft_tokens, target_logits)
    
    # Verify loads experts for positions 0..reject_pos+1
    experts_loaded = set()
    for t in range(reject_pos + 2):  # +1 for bonus token
        verify_routing = get_target_routing(t)
        for expert in verify_routing:
            cache.ensure_loaded(expert)
            experts_loaded.add(expert)
    
    # Next draft starts from reject_pos + 1
    # Measure hit rate for first m steps of next draft
    for m in range(K):
        next_routing = get_original_routing(next_pos + m)
        hit_rate = count_hits(next_routing, cache) / k_active
        log(m, hit_rate)
```

**预期结果**：
- 前 1-3 步 hit rate > 0.8（甚至 > 0.9）
- 后续步骤逐步回到平均水平
- 这与 MoE-SpeQ 形成对比：其每步 draft 都有固定 ~9% 路由误差

**实验 M3b：前缀命中率 vs 接受率提升**

将 M3a 的 per-step hit rate 转换为 per-step 的 α 估计，构建"前缀加速"的完整论证：

```python
# 对比两种场景的 seq_α
# 场景 1: 每轮 draft 用全新随机 cache（模拟无复用）
# 场景 2: 每轮 draft 继承 verify 更新后的 cache（我们的方法）
for scenario in ['fresh_cache', 'verify_inherited']:
    for cycle in range(num_cycles):
        if scenario == 'fresh_cache':
            cache = build_lfu_cache(global_freq)  # reset
        
        seq_alpha = run_rerouting_eval(cache, alg='Alg2_v2')
        log(scenario, cycle, seq_alpha)
```

**预期结果**：verify_inherited 场景的 seq_α 比 fresh_cache 高 3-8pp（在 r=0.25-0.50 时）。

**脚本位置**：`pre_exps/post_reject_prefix_eval.py`

---

### 实验 M4-Exp: 预取信号质量对比

**目的**：量化比较 draft routing（FP16 router on approximate hidden state）和 INT4 routing 的 fidelity 差异。

**修正（2026-05-29）**：之前的表述暗示我们的路由是"精确的"，实际不然。Draft routing 在近似 hidden state 上计算，存在 drift 误差。实验目标改为**测量两种误差源的实际大小**。

**方法**：

```python
for each prompt:
    # 1. Ground truth: FP16 full model routing
    target_routing = collect_routing(fp16_model, prompt)
    
    # 2. Draft routing: FP16 router on rerouted hidden states
    draft_routing = collect_routing(model_with_rerouting, prompt)
    
    # 3. INT4 routing: 通过 bitsandbytes 4-bit 量化或模拟噪声
    int4_routing = collect_routing(int4_model, prompt)  # or simulated
    
    # 4. 计算 fidelity
    draft_fidelity = |draft_routing ∩ target_routing| / top_k
    int4_fidelity  = |int4_routing  ∩ target_routing| / top_k
```

**INT4 模型来源**（3种选择，按优先级）：
1. bitsandbytes 4-bit (NF4) 量化加载 (`--int4_mode bnb`)
2. AutoGPTQ / AWQ 量化（如有预量化模型）
3. 模拟噪声：校准到 MoE-SpeQ 报告的 ~9.1% mismatch (`--int4_mode simulated`)

**预期结论**：在高 cache ratio（≥0.50）时，draft fidelity 显著高于 INT4 fidelity（hidden state drift 小）；在低 cache ratio（≤0.25）时差距缩小，需实验确认。

**脚本位置**：`pre_exps/prefetch_signal_quality_eval.py`

---

## 5. 实验优先级排序

| 优先级 | 实验 | 理由 | 工作量 |
|--------|------|------|--------|
| **P0** | M0-Exp (VRAM trade-off) | 最核心论点，定量无争议 | 0.5 天 |
| **P0** | M3-Exp (前缀命中率) | 独特优势，直观有力 | 1 天 |
| **P1** | M2-Exp (cycle 暖化) | 与 M3 形成闭环论证 | 1 天 |
| **P2** | M4-Exp (预取信号) | 补充论点，需模拟量化误差 | 1 天 |

---

## 6. 潜在反驳与应对

### 反驳 1: "MoE-SpeQ 有 90%+ 接受率，你们的只有 ~73%"

**应对**：
1. 接受率比较不能脱离 VRAM 预算——MoE-SpeQ 在 A100-40G 上评估，我们在 RTX 4090-24G 的约束下。在**同等 VRAM 预算下**比较才公平。
2. 我们当前 73% 是 Alg2_v2 在 r=0.25 的结果。M0-Exp 将展示在同等 VRAM 下我们的 cache ratio 更高 → seq_α 更高。
3. MoE-SpeQ 的 90% 接受率是在 Qwen1.5-MoE (N=60) 上测的，不是 N=128 的细粒度模型。

### 反驳 2: "量化 draft 模型的路由更接近目标，因为它经过完整 forward"

**应对**：
1. 我们的 draft 也经过完整 forward（所有层），只是 miss 专家被替换。路由器本身是精确的 FP16。
2. 量化模型的路由误差来自权重量化对 gate logit 的扰动，这是**每一层、每一步都存在的系统性误差**。我们的误差来自 hidden state 偏差，但在高 cache ratio 时偏差极小（miss 专家权重通常很低）。

### 反驳 3: "前缀复用的收益很小"

**应对**：通过 M3-Exp 直接测量。即使前缀只有 2-3 步是高命中的，在 K=8 的 draft 中也代表 25-37% 的步骤近乎精确，对 seq_α 的提升是可测量的。

---

## 7. 总结

对 MoE-SpeQ 的攻击维度按重要性排序：

1. **VRAM 效率** (M0)：最核心、最不可争辩的优势。量化 draft 模型占用的 VRAM 在消费级硬件上是致命的。
2. **前缀复用闭环** (M3+M2)：self-speculative 的独特结构性优势——verify 直接更新 draft 的工作集。
3. **量化兼容性** (M1)：扩大适用场景。
4. **预取信号质量** (M4)：精确 metadata 的间接但重要的系统级优势。
5. **细粒度 MoE 扩展** (M5)：趋势性论点，随 N 增大优势扩大。
6. **CUDA graph 加速** (M6)：系统实现层面的优势。
7. **Draft 质量可观测** (M7)：暂不作为 motivation 实验，留给后续 Dynamic K 工作。
