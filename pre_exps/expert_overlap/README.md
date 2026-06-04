# MoE Expert Activation Overlap Analyzer

测试 **Qwen/Qwen3-30B-A3B** MoE 模型在推理（decode）阶段，相邻 token 序列段之间激活专家集合的重叠度。

## 实验方法

```
Prompt  ──prefill──▶ [不记录]
                     │
                     ▼
         ┌─── Segment 1 ───┐  ┌─── Segment 2 ───┐  ┌─── Segment 3 ───┐
Decode:  │ tok1 tok2 ... tokN│  │ tok1 tok2 ... tokN│  │ tok1 tok2 ... tokN│  ...
         └──────────────────┘  └──────────────────┘  └──────────────────┘
              ↓                      ↓                      ↓
         每层记录 N 步               每层记录 N 步            每层记录 N 步
         激活的专家集合(union)        激活的专家集合(union)     激活的专家集合(union)
              │                      │                      │
              └──────┬───────────────┘                      │
                     ↓                                      │
              计算 Jaccard(Seg1, Seg2)                      │
                            └──────────┬────────────────────┘
                                       ↓
                                计算 Jaccard(Seg2, Seg3)
```

对于每个请求（prompt）：

1. **Prefill** — 编码 prompt，不记录专家激活
2. **Decode** — 逐 token 自回归生成，每层通过 gate hook 记录每个 token 激活的 top-k 专家
3. **分段** — 将 decode 步骤切分为长度为 `n` 的段（segment）
4. **计算重叠** — 每层取每段内所有 token 激活专家的并集（union），计算相邻段的 Jaccard 相似度

重复多个 prompt，汇总统计。

## Qwen3-30B-A3B 架构参数

| 参数 | 值 |
|------|-----|
| 总参数量 | 30.5B |
| 激活参数量 | 3.3B / token |
| Decoder 层数 | 48 |
| 每层专家数 | 128 |
| 每 token 路由专家数 | 8 (top-8) |
| 注意力头 | 32 query + 4 KV (GQA) |
| 上下文长度 | 32K (YaRN可扩展到131K) |

## 安装依赖

```bash
pip install torch transformers accelerate numpy

# 如果需要量化加载（推荐，可大幅减少显存）
pip install bitsandbytes

# 如果需要可视化
pip install matplotlib
```

## 显存需求

| 加载方式 | 显存需求 | 命令行参数 |
|---------|---------|-----------|
| BF16/FP16 | ~61 GB | (默认) |
| 8-bit 量化 | ~31 GB | `--load_in_8bit` |
| 4-bit 量化 | ~16 GB | `--load_in_4bit` |

## 使用方法

### 基本用法

```bash
# 默认配置：segment_length=32, 8 segments, 10 prompts
python test_moe_expert_overlap.py

# 4-bit 量化（单卡 24GB 即可运行）
python test_moe_expert_overlap.py --load_in_4bit

# 关闭 thinking 模式（避免生成过长的思考链）
python test_moe_expert_overlap.py --load_in_4bit --disable_thinking
```

### 自定义参数

```bash
# 不同 segment 长度
python test_moe_expert_overlap.py --segment_length 64 --num_segments 6

# 同时测试多种 segment 长度
python test_moe_expert_overlap.py \
    --segment_length 16 \
    --extra_segment_lengths 32 64 128 \
    --num_segments 8

# 少量 prompt 快速测试
python test_moe_expert_overlap.py --num_prompts 3 --load_in_4bit

# 使用自定义 prompt
python test_moe_expert_overlap.py --prompts_file my_prompts.json

# 指定输出文件
python test_moe_expert_overlap.py --output results_v1.json
```

### 可视化

```bash
# 交互显示
python visualize_overlap.py expert_overlap_results.json

# 保存为图片
python visualize_overlap.py expert_overlap_results.json --save overlap.png

# 附带 per-prompt 热力图
python visualize_overlap.py expert_overlap_results.json \
    --save overlap.png --heatmap --heatmap_save heatmap.png
```

## 输出指标

### 1. Jaccard 相似度（核心指标）

$$\text{Jaccard}(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

其中 A、B 是相邻两段内某层激活专家的并集。

- **≈ 1.0**: 相邻段激活几乎完全相同的专家 → 高时间局部性
- **≈ 0.5**: 中等重叠
- **≈ 0.0**: 完全不同的专家

### 2. Overlap 系数

$$\text{Overlap}(A, B) = \frac{|A \cap B|}{\min(|A|, |B|)}$$

### 3. Token 级别专家持续率

连续 token 之间共享专家的比例，反映逐步的专家切换频率。

### 4. 每段激活专家数

反映每段 N 步 decode 后共涉及多少个不同专家（上限 128）。

## 输出文件结构 (JSON)

```json
{
  "config": {
    "model": "Qwen/Qwen3-30B-A3B",
    "num_experts": 128,
    "top_k": 8,
    "num_moe_layers": 48,
    "segment_lengths": [32],
    "num_segments": 8,
    ...
  },
  "per_prompt": [
    {
      "prompt": "...",
      "tokens_decoded": 256,
      "segment_analyses": {
        "32": {
          "0": {
            "jaccard_mean": 0.72,
            "jaccard_std": 0.05,
            "intersection_size_mean": 45.2,
            "segment_expert_count_mean": 52.3,
            ...
          },
          ...
        }
      }
    }
  ],
  "global_summary": {
    "seg_len_32": {
      "per_layer": { ... },
      "overall_jaccard_mean": 0.68,
      "overall_jaccard_std": 0.12
    },
    "token_persistence": { ... }
  }
}
```

## 实验意义

专家激活的时间局部性直接影响 MoE 推理系统的优化策略：

- **高重叠 (Jaccard > 0.7)** → 专家缓存/预取策略有效，可显著减少分布式推理中的专家加载开销
- **中等重叠 (0.3~0.7)** → 部分缓存有效，需要结合预测机制
- **低重叠 (< 0.3)** → 专家调度需要更激进的预取或全量常驻策略

## 技术细节

### Hook 机制

脚本通过 PyTorch 的 `register_forward_hook` 注入到每层 MoE gate/router 模块：

- **transformers ≥ v5.0**: gate 是 `Qwen3MoeTopKRouter`，forward 返回 `(logits, scores, indices)`，直接取 `indices`
- **transformers < v5.0**: gate 是 `nn.Linear`，返回 raw logits，脚本自行做 top-k

### KV Cache

Decode 使用 KV cache，每步只处理新 token。Hook 在每步触发一次，记录该 token 在各层的专家选择。

### 兼容性

- 自动检测 MoE 层（跳过 `mlp_only_layers` 指定的 dense 层）
- 自动检测 gate 类型（Router vs Linear）
- 支持量化加载（4-bit / 8-bit via bitsandbytes）
