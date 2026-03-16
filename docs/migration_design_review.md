# migration_design.md 设计评审报告

> 基于 LLM/MoE 推理、系统优化、PyTorch/Triton/CUDA/通信等专业知识，对《基于 nano-vllm-moe 实现 CPU-GPU 异构 MoE 推理设计文档》的事实性与合理性评审。

---

## 一、事实性错误与修正

### 1.1 CPU 路径参数键名与计算方式（§4.5）

**错误**：文档 §4.5 中 `_cpu_parallel_forward` 伪代码使用：

```python
params = cpu_expert_pool[(layer_idx, expert_idx)]
gate_out = F.linear(x, params["gate_proj"])
up_out = F.linear(x, params["up_proj"])
inter = F.silu(gate_out) * up_out
out = F.linear(inter, params["down_proj"])
```

**事实**：

- 异构加载器（§4.2 Step 2 及现有 `heterogeneous_loader.py`）将 expert 权重以 **合并后的 `gate_up`** 和 **`down`** 存入 CPU pool，键名为 `"gate_up"` 与 `"down"`，不存在 `"gate_proj"` / `"up_proj"` / `"down_proj"`。
- 正确计算应为：用 `gate_up` 做一次 linear 得到 `[B, 2*intermediate]`，再拆成 gate/up，做 SiLU(gate)*up，再与 `down` 做 linear。

**建议修正**（与现有 `heterogeneous.py` 及 loader 一致）：

```python
params = cpu_expert_pool[(layer_idx, expert_idx)]
gate_up = params["gate_up"]   # [2*intermediate, hidden]
down = params["down"]         # [hidden, intermediate]
gate_up_out = F.linear(x, gate_up)  # [B, 2*intermediate]
gate_out, up_out = gate_up_out.chunk(2, dim=-1)
inter = F.silu(gate_out) * up_out
out = F.linear(inter, down)
```

文档中所有出现 `params["gate_proj"]` / `params["up_proj"]` / `params["down_proj"]` 的地方应统一改为上述 `gate_up` + `down` 的用法。

---

### 1.2 cpu_expert_pool 的键类型（与实现一致即可）

文档中写的是 `Dict[Tuple[int,int], Dict[str, Tensor]]`，即 `(layer_idx, expert_idx)` 作为键；当前代码实现为 `dict[layer_idx][expert_idx]`（嵌套 dict）。两种方式在逻辑上等价，实现时只需二选一并统一即可，无需改设计目标，但文档若与现有实现一致，建议在「与现有代码的关系」处注明：返回结构可为「按 layer 分组的嵌套 dict」，以便与 `heterogeneous_loader.py` 对齐。

---

### 1.3 Triton kernel 的 E 与权重形状

文档描述「权重为连续的 `[E, N, K]` 张量，`m_sizes` 为长度 E」；在异构方案中改为「传入 `[S, N, K]` slot buffer，`m_sizes` 长度为 S」。  
与 `grouped_gemm.py` 一致：`grouped_gemm_forward` 中 `E, N, _ = w.shape`，kernel 的 `NUM_EXPERTS` 即传入权重的第一维。因此用 S 作为第一维、`m_sizes.numel() == S` 在语义和实现上均正确，无需修改，仅需在文档中明确：此处「E」在异构模式下即「S（slot 数）」，避免读者与「总 expert 数 E=128」混淆。

---

### 1.4 Autotune 与 `NUM_EXPERTS`（S）的关系

文档称「NUM_EXPERTS = S 是编译时常量」「所有层共享相同 S，kernel 只需编译一次」。  
`autotuning.py` 中 `get_autotune_keys()` 包含 `"NUM_EXPERTS"`，因此 **S 不同会触发 Triton 重新编译/autotune**。若未来支持「每层不同 S」，则每层一种 S 会对应一次编译；当前设计「所有层同一 S」的表述正确，只需在文档中补充一句：若改为每层不同 slot 数，需接受多份 kernel 编译（数量=不同 S 的个数）。

---

## 二、设计合理性评估

### 2.1 整体架构与复用策略

- **复用 nano-vllm-moe 基础设施**（flash_attn、Triton grouped GEMM、Paged KV、Scheduler、Block Manager）并采用「MoE block 内插 placement + 异构 forward」而非整模型拆分的方案，与文档 §8.1 一致，能保留 CUDA Graph（标准路径）、Context 管理等，**合理且可落地**。
- **分层固定 Slot、每层 `[S, N, K]` buffer** 与现有 `_grouped_gemm_forward_kernel` 的按 expert 维遍历、跳过 `m_size=0` 完全匹配，无需改 kernel，**设计正确**。
- **Draft 用 Expert 替换、Verify 用全精度无替换** 的投机解码语义清晰，与常见 speculative decoding 一致，**合理**。

### 2.2 数据流与职责划分

- 数据流图（§3.2）中「标准 / Speculative 分支 → ModelRunner.run() → Attention / Router / MoE(heterogen) → GPU path + CPU path」与模块划分一致。
- Expert 预取在 draft 循环内、基于激活预测 verify 所需 expert，并在 `complete_transfers()` 中同步映射表，逻辑自洽；文档已说明传输与计算的 PCIe 竞争及用 `transfer_stream.synchronize()` 的简化方案，**合理**。

### 2.3 KV Cache Draft/Verify 生命周期

- Draft 阶段扩展 KV、Draft 结束后回滚、Verify 用「last_accepted + draft_tokens」做一次 prefill-like 得到 verify KV、Accept 后只保留「accepted」对应的 verify KV，**与标准 speculative 语义一致**。
- 唯一需要显式写清的一点：**new_token（从 verify 在位置 M 的 logits 采样得到）的 KV 尚未写入**。通常做法是在同一 step 内再跑一次 decode（仅输入 new_token）以写入其 KV，或由下一 step 的 decode 自然写入。建议在 §4.7 或 §6.1 增加一句：「接受后序列为 [..., dM-1, new_token]；new_token 的 KV 需在本 step 内补一次 decode 或在下一 step 的 decode 中写入，以保证 BlockManager/KV 状态一致。」

### 2.4 配置与策略可插拔

- `cache_strategy` / `prefetch_strategy` / `draft_scheduler` / `acceptance_strategy` 等以配置项+接口形式存在，便于后续换 LRU/LFU/adaptive 等，**设计合理**。

---

## 三、有效性评估

### 3.1 能达成的目标

- **显存约束下运行大 MoE**：通过「静态+少量 S 的 expert 在 GPU、其余在 CPU」和分层 slot，在 24GB 级显卡上跑 Qwen3-30B-A3B 级别模型在工程上可行。
- **正确性**：Draft 用替换只影响 draft 质量；Verify 全精度、无替换，保证输出与全量 GPU 一致；Accept 逻辑与常见实现一致。
- **性能潜力**：预取与 CPU-GPU 并行可部分掩盖传输；投机解码在接受率足够高时有吞吐收益。文档中 Benchmark 与「Phase 2 完成门槛 C1 vs B1 ≥1.1x」等设定有助于验证有效性。

### 3.2 风险与依赖

- **PCIe 带宽**：CPU↔GPU 拷贝与 GPU 计算共享带宽，高负载时可能成为瓶颈；文档已提及，实现时需保留「按层/按 step 控制并发传输量」的开关（如 §11.6 的预取上限）。
- **CPU 路径延迟**：若 `draft_top_c` 或 cache miss 导致较多 expert 在 CPU 上执行，CPU 延迟会直接拉高 draft/verify 时延；文档中 Phase 4 的「CPU Expert 执行优化（torch.compile、NUMA）」是必要的后续方向。

---

## 四、需要优化与补充的部分

### 4.1 实现与接口层面

1. **MoEExecutionPlan 与现有实现的差异**  
   文档中的 `MoEExecutionPlan` 字段较多（如 `gpu_sort_idx`、`gpu_inv_sort_idx`、`gpu_token_map`、`gpu_weights`、`cpu_expert_indices`、`cpu_token_map`、`cpu_weights`、`substitution_map` 等），而当前 `placement.py` 仅包含 `gpu_route_indices`、`cpu_route_indices`、`m_sizes`。  
   **建议**：在实现路线图（§7）或 placement 小节中明确：先实现「无替换、无 draft」的 prefill/verify 最小 plan（与现有 MoEExecutionPlan 对齐），再迭代增加 draft 的 substitution、权重聚合与 scatter 所需字段，避免一次性实现所有字段导致接口反复变更。

2. **Draft 阶段 substitution 的语义**  
   文档明确「未缓存 expert → 用 GPU cache 中某 expert 替代」，但未规定替代策略（随机、按相似度、按 slot 负载等）。  
   **建议**：在 §4.10 Draft Scheduler 或 §5.7 中简短约定默认策略（如「当前实现：从同层已缓存 expert 中按路由权重或随机选替代」），并注明对 acceptance 的影响（替换会降低 draft 质量，可能降低接受率）。

3. **Verify 跳过条件（§6.2）**  
   「所有 routed expert 在 GPU + draft_top_c==0 + greedy」时可跳过 verify 的优化是合理的。  
   **建议**：在 BlockManager/Sequence 或 spec_engine 中明确「跳过 verify 时，如何复用 draft 的 KV 而非丢弃」（若 draft KV 与全精度一致则可直接保留，否则需在文档中说明当前实现为「不跳过时一律回滚 draft KV」）。

### 4.2 性能与可观测性

4. **分阶段计时与算子分解**  
   §11.4 已定义 `attn_s`、`route_s`、`moe_gpu_s`、`moe_cpu_s`、`transfer_wait_s` 等，对定位瓶颈很重要。  
   **建议**：在 §4.9 ModelRunner 或 spec_engine 中约定轻量打点方式（如 CUDA events 或 torch.cuda.synchronize() 包裹的关键段），便于与 11.4 指标一一对应。

5. **预取上限与传输预算**  
   §11.6 的 \(N_{prefetch,max} = \lfloor T_{win} / T_{copy} \rfloor\) 和「按层窗口」合理。  
   **建议**：在 prefetcher 接口或配置中显式加入「每层/每 step 最大并发预取数」和「是否使用 micro-benchmark 的 \(T_{copy}\)」，避免预取过猛占满 PCIe 导致 GPU 计算变慢。

### 4.3 正确性与边界情况

6. **多 Sequence 与 preemption**  
   文档提到「多 sequence batch」和 scheduler 的 preemption，但未详细说明在 speculative 路径下，若某 sequence 被抢占或延迟，draft/verify 的 batch 如何与 scheduler 的 seq 列表对齐。  
   **建议**：在 §4.6 或 §6.1 中简要说明：speculative_step 接收的 `seqs` 与 scheduler 当前调度的 decode batch 一致，且 draft/verify 期间不改变 batch 组成；若存在 preemption，需在 scheduler 层保证同一 step 内不混入新加入的 sequence（或明确约定行为）。

7. **Greedy 与 sampling 的 Accept 表述**  
   §4.6 Accept 阶段中，greedy 用 argmax 比较、sampling 用 AcceptanceStrategy，逻辑正确。可补充一句：sampling 时「接受」的定义（如 P_verify(token) ≥ threshold 或基于 verify 的采样与 draft 一致）与 `acceptance_strategy` 的实现保持一致，避免后续实现与文档语义不一致。

### 4.4 文档与实现一致性

8. **BlockManager 的 `rollback_draft` / `accept_draft`**  
   文档用 `seq.num_tokens`、`seq.block_table`、`seq.num_blocks` 等；若现有 BlockManager 使用 `num_blocks` 或 token 计数方式不同，实现时需按实际字段做等价操作（例如用 block 数量反推 token 数），并在文档中注明「与当前 BlockManager 的字段对应关系」。

9. **Sequence 的 `append_draft_token`**  
   文档写「同时更新主 token 序列」即 `append_token(token_id)`。若主序列和 KV 的「逻辑长度」由 BlockManager 管理，需明确：draft 期间主序列仅用于 draft 下一步输入，真正提交的是 accept_draft 时的截断与 new_token 追加，避免「主序列」与「KV 可见长度」不一致。

---

## 五、小结

| 维度           | 结论 |
|----------------|------|
| 事实性         | 需修正 §4.5 CPU 路径参数键名与 gate_up 计算方式；其余与代码/内核行为一致，仅建议补充 S 与 E 的用语、autotune 与 S 的关系。 |
| 合理性         | 分层 Slot、Draft-Verify-Accept、路由-调度分离、策略可插拔等设计合理；KV 生命周期需补一句 new_token 的 KV 写入时机。 |
| 有效性         | 能支撑显存受限下的 MoE 推理与投机解码；性能依赖预取与接受率，需控制 PCIe 与 CPU 路径延迟。 |
| 建议优化       | 明确 MoEExecutionPlan 的渐进实现、substitution 默认策略、预取上限与打点、多 seq/preemption 与 speculative 的约定、BlockManager/Sequence 字段与文档的对应关系。 |

按上述修正与补充后，该设计可直接作为实现与评审的基准文档使用。
