# LFU-RankGuard 缓存策略集成实现报告

**Date:** 2026-05-29  
**设计来源:** `docs/post_experiment_optimization.md` §2  
**目标:** 在现有缓存驱逐框架中新增 LFU-RankGuard 策略，保护 top-1/2 高频专家不被意外驱逐

---

## 1. 变更总览

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `nanovllm/scheduling/cache_strategy.py` | 新增类 + 扩展工厂 | 新增 `LFURankGuardStrategy` 类，注册 `"lfu_rankguard"` 到 `create_cache_strategy` |
| `nanovllm/config.py` | 新增配置项 | 新增 `rank_guard_threshold`、`rank_guard_ema_alpha`，扩展 `cache_strategy` 验证 |
| `nanovllm/engine/model_runner.py` | 条件分支 | 当策略为 `lfu_rankguard` 时传入 `num_experts`、阈值、EMA 系数 |
| `nanovllm/expert/prefetcher.py` | 新增方法 + 扩展 | 新增 `_update_rank_guard_scores`；`observe_verify` 挂钩 EMA 更新；`_select_publish_slot_cpu` 增加 rank guard 分支 |
| `nanovllm/scheduling/__init__.py` | 导出 | 新增 `LFURankGuardStrategy` 导出 |

**零侵入性保证：** 所有现有策略（LRU / LFU / Adaptive）的代码、行为、接口均未改动。`create_cache_strategy` 签名扩展为 `(name, **kwargs)` ——无 kwargs 时行为与原函数完全一致。

---

## 2. 核心算法：LFURankGuardStrategy

### 2.1 驱逐决策（`select_victim_slot`）

```
Phase 1: 遍历所有 cache slot
  ├─ 空 slot (expert < 0) → 立即返回
  ├─ 受保护 expert (rank_score ≥ threshold) → 跳过
  └─ 非保护 expert → 按 access_count 选最小者 (LFU)

Phase 2: 安全阀
  └─ 若所有 cached expert 均受保护 → 回退到纯 LFU（不考虑保护）
```

开销：每次驱逐决策增加一次 `dict.get()` + 浮点比较，约 0.001ms。

### 2.2 rank_scores 计算

**公式：** `rank_score(j) = 2 × rank1_freq(j) + rank2_freq(j)`

其中 `rank1_freq` 和 `rank2_freq` 分别是专家在 verify routing 中作为 rank-1（第一列）和 rank-2（第二列）出现的频率。

**在线 EMA 更新：**
```python
score[j] = α × score[j] + (1 - α) × current_score[j]
```
- 默认 `α = 0.95`
- 每轮 verify 后从 routing metadata 中提取 `selected_experts`（shape `[T, topk]`）
- 仅在 raw `selected_experts` 可用时更新（aggregated metadata 丢失了 rank 顺序信息）

### 2.3 快速路径：`select_victim_slot_cpu`

为 segment-indexed prefetch 的热路径提供无 snapshot 开销的版本：

```python
select_victim_slot_cpu(slot_to_expert, access_count, layer_idx, is_pending_fn) -> int | None
```

直接操作 list 而非 snapshot，避免 tensor clone 开销。与 `select_victim_slot` 逻辑完全一致。

---

## 3. 集成点详细说明

### 3.1 Config 新增参数

```python
rank_guard_threshold: float = 0.15   # rank_score ≥ 此值的 expert 受保护
rank_guard_ema_alpha: float = 0.95   # EMA 平滑系数（越大越平滑）
```

验证约束：`0.0 ≤ threshold ≤ 1.0`，`0.0 < ema_alpha ≤ 1.0`

### 3.2 model_runner.py 初始化

```python
if config.cache_strategy == "lfu_rankguard":
    self.cache_strategy = create_cache_strategy(
        config.cache_strategy,
        num_experts=int(getattr(config.hf_config, "num_experts", 128)),
        protect_threshold=config.rank_guard_threshold,
        ema_alpha=config.rank_guard_ema_alpha,
    )
```

`num_experts` 从 HuggingFace config 中读取，确保与模型一致。

### 3.3 PrefetchRuntime 集成

**EMA 更新路径：**

```
observe_verify()
  → observe_runtime_meta()          # 原有：mark_access + queue update
  → _update_rank_guard_scores()     # 新增：EMA 更新 rank_scores
      └─ 遍历每层 meta
         └─ 若有 raw selected_experts → update_rank_scores_from_routing()
```

**驱逐路径：**

`_select_publish_slot()`（用于 staging publish、direct-active prefetch、verify-layer prefetch）
  → 调用 `self.cache_strategy.select_victim_slot(snapshot)` → 自动适配 rank guard

`_select_publish_slot_cpu()`（用于 segment-indexed prefetch）
  → 检测 `strategy_name == "lfu_rankguard"` 且 cache_strategy 为 `LFURankGuardStrategy` 实例
  → 调用 `cache_strategy.select_victim_slot_cpu()`（无 snapshot 快速路径）
  → 需要 `layer_idx` 参数 → 在调用点 `submit_draft_segment_indexed_prefetch` 中已传入

### 3.4 与设计文档的差异

| 设计文档描述 | 实际实现 | 差异原因 |
|-------------|---------|---------|
| `rank_scores` 存为 `dict[int, dict[int, float]]` | 存为 `dict[int, list[float]]` | list 按 expert_idx 索引，O(1) 查找且内存更紧凑 |
| 修改 `AdaptiveCacheStrategy` | 新增独立类 | "不改动现有实现"的要求 |
| `PrefetchRuntime` 初始化 rank_scores | 初始为空，在线 EMA 填充 | 离线校准数据可通过 `load_rank_scores_dict()` 手动加载 |
| aggregated metadata 也用于 EMA 更新 | 仅 raw `selected_experts` 可用时更新 | aggregated 格式丢失 rank 顺序信息，无法准确计算 rank1/rank2 频率 |

---

## 4. 使用方式

### 4.1 基本启用

```python
config = Config(
    model="/path/to/Qwen3-30B-A3B",
    cache_strategy="lfu_rankguard",       # 启用 rank guard
    rank_guard_threshold=0.15,            # 保护阈值
    rank_guard_ema_alpha=0.95,            # EMA 系数
    # ... 其他配置
)
```

### 4.2 加载离线校准数据

在 `PrefetchRuntime` 初始化后：

```python
from nanovllm.scheduling.cache_strategy import LFURankGuardStrategy

if isinstance(model_runner.cache_strategy, LFURankGuardStrategy):
    calibration_data = load_calibration(...)  # {layer_idx: [score_per_expert]}
    model_runner.cache_strategy.load_rank_scores_dict(calibration_data)
```

### 4.3 阈值选择建议（基于 E2 校准数据）

| threshold | 每层受保护 expert 数 | 适用场景 |
|-----------|---------------------|---------|
| 0.10 | ~95 | 过于保守，cache 压力大 |
| **0.15** | ~42 | **推荐默认值** |
| 0.20 | ~22 | 激进，仅保护最核心 expert |

---

## 5. 内存与性能开销

- **内存：** 每层 128 × float64（Python list）≈ 1 KB × 48 层 ≈ 48 KB（可忽略）
- **驱逐决策：** 增加 `dict.get()` + `list[]` + float 比较 ≈ 0.001ms/次
- **EMA 更新：** 每轮 verify 遍历 topk 列 + 128-expert 列表更新 ≈ 0.01ms/层 × 48 层 ≈ 0.5ms/轮
- **无运行时 tensor 分配**，无 GPU 内存消耗

---

## 6. 验证清单

- [x] `create_cache_strategy("lru")` / `"lfu"` / `"adaptive"` 行为不变
- [x] `create_cache_strategy("lfu_rankguard")` 正确创建实例
- [x] 无 rank_scores 时退化为纯 LFU
- [x] 受保护 expert 被跳过，选择最低 access_count 的非保护 expert
- [x] 所有 expert 均受保护时安全阀回退到纯 LFU
- [x] 空 slot 立即返回
- [x] EMA 更新数值正确
- [x] `is_protected` 边界条件（无 scores 的层、越界 expert_idx）
- [x] `select_victim_slot_cpu` 与 `select_victim_slot` 行为一致
- [x] `load_rank_scores_dict` 正确加载
- [x] Config 新字段验证约束正确
- [x] 所有修改文件 AST 解析无误
