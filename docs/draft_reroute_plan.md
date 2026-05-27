
  # Draft Expert Reroute V2 CUDA Graph 集成与验证计划

  ## Summary

  - 仅实现 draft_top_c=0 的 reroute；draft_top_c>0 未来方案写入主文档附录，本次不实现或测试。
  - 当前生产 baseline 不是零成本路径：它在每层 replay 内构建 [num_experts] round-robin substitution
    LUT，再对原始 routes 查表。新策略应替代该 substitution 阶段，而不是叠加其上。
  - 默认 draft_reroute_policy=baseline 保持现有行为和代码路径；公开的新策略必须支持 CUDA Graph，且
    相对当前 baseline 的 draft_graph_replay 与 draft_forward 中位数增量均 <3.0 ms。

  ## Public Interfaces

  - 新增配置/CLI：
      - draft_reroute_policy: baseline | skip_all_v2 | alg2_v2 | hybrid_cp_v2 | post_sub_v2，以及经
        测试决定保留的其他 option。
      - draft_reroute_artifact_path: post_sub_v2 等依赖校准表策略所需的 safetensors 文件。
      - draft_reroute_miss_gate=0.25
      - draft_reroute_sim_floor=0.40
  - 配置约束：draft_reroute_policy != "baseline" 与 draft_top_c > 0 同时出现时直接报错，并指向文档
    附录。
  - Alg2_PostSub 不预先作为公开 option：先验证其是否与 alg2_v2 在规范化 execution routes/weights 及
    MoE 输出上等价；等价则只记录算法消去结论，不暴露重复配置；不等价则加入公开候选与完整测试矩阵。
  - 调试过程中发现的其他改进策略可以作为新增 option 候选；必须记录精确差异并通过同一验收门槛。

  ## Runtime Architecture

  - 在 draft MoE router 产生原始 selected_experts/routing_weights 后、执行 plan 构建前插入
    DraftReroutePolicy。
  - Prefetch/runtime metadata 继续记录原始 router demand；只有实际 MoE 执行使用 policy 输出的
    execution_selected/execution_weights。
  - 保留现有 baseline 分支不变：其 round-robin LUT 逻辑仍作为性能和接受率对照。
  - 为非 baseline 策略增加 build_cached_draft_plan_gpu(...) 路径：输入保证所有物理 route 都能映射到
    cached slot，直接建立 grouped layout，跳过 baseline 的 [N] substitution LUT 构建。
  - Graph 固定保持 tokens * top_k 条物理 routes。策略中的“skip miss”表示权重为零；该零权重 route 绑
    定确定性的 cached fallback slot 以满足现有 graph/layout 约束。首版仍会为该物理 route 执行
    grouped GEMM，权重在输出合并前归零；减少零权重 GEMM 属于后续可评估优化 option。
  - 若多个非零 route 最终指向同一 expert，以固定 K x K 比较/归并将权重合入首个 slot，其余重复 slot
    置零并绑定 fallback，精确对应原型的 scatter-add 语义。

  ## Baseline And Policy Costs

| Policy         | Graph 内替换行为                                                                                                                                                                       | 相对当前 baseline 的主要差异                                                     | 预期开销                                              |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------- |
| `baseline`     | 每层创建 `[N]` LUT：cached 为 identity，uncached `e` 映射到 `slot_to_expert_lut[e % slots]`；随后对 `T*K` routes 查表                                                                             | 当前实现                                                                    | 对照值；特有成本为 `O(N) + O(T*K)` substitution            |
| `skip_all_v2`  | `hit = cached_mask[original_selected]`；miss 权重置零；hit 权重重归一化；全 miss 回退到一个 cached expert                                                                                            | 用 `O(T*K)` mask / where / reduce 替代 baseline 的 `[N]` LUT 与 route lookup | plan 部分可能低于 baseline；首版 GEMM 数不减少                 |
| `alg2_v2`      | 复用 router 完整概率计算 entropy；由原始 top-k hit 计算 miss ramp；对 cached experts 施加 entropy-scaled bias；执行 biased top-k；向量化 top-1 protection；按原始 logits 重算所选权重；residual miss 走 SkipAll        | 不再执行 round-robin LUT，但增加 `[T,N]` bias / reduction 与第二次 top-k            | 主要新增成本为第二次 top-k；应通过融合 / 编译保持 `<3 ms`             |
| `hybrid_cp_v2` | 计算 miss ramp / entropy；取原始 top-J，`J=min(3K,N)`；只对 `cached ∩ top-J` 加 bias；按被替换原始权重总量执行向量化 deviation guard；residual miss 走 SkipAll                                                 | 替代 baseline，并比 `alg2_v2` 多 top-J 与 guard                                | 预计是最重的预路由策略；超门槛则不得作为验收通过 option                   |
| `post_sub_v2`  | 对原始 miss gather `best_substitute[e]` / `best_similarity[e]`；应用 miss gate、contribution threshold、sim floor 和 original-top-k collision guard；合格 miss 替换并按 `best_sim * gate` 缩放，否则置零 | replay 内仅 `O(T*K)` gather / mask / reduce，替代 baseline LUT               | 固定 cache 下预期接近 `skip_all_v2`；cache refresh 开销单独报告 |
| `Alg2_PostSub` | 先按原型执行 `alg2_v2`，再对阶段一输出执行 `post_sub_v2`；零 route canonicalize 到统一 fallback                                                                                                        | 用于等价性判定，不默认公开                                                           | 若数值等价于 `alg2_v2` 则运行时删除，无额外成本                     |


  模型目标规模为 N=128、K=8、L=48、测试 num_seqs=1；以上开销判断为实现指导，最终结论以 A100 实测为
  准。

  ## Calibration And Cache Refresh

  - 新增离线校准导出脚本，产出 safetensors artifact：cond_sim[L,N,N]、skip_err[L,N]，以及格式版本、
    模型标识、层数、expert 数、top-k metadata。
  - post_sub_v2 启用时在初始化校验 artifact 与当前模型一致，并分配每层持久 device buffers：
    best_substitute[N]、best_similarity[N]、fallback_cached_expert[1]。
  - 缓存映射提交变化时递增 layer cache revision；在下一次 draft replay 前，仅为 revision 改变的层在
    replay 外刷新 fallback scalar，且对 post_sub_v2 刷新 best-sub buffers。
  - skip_all_v2、alg2_v2、hybrid_cp_v2 graph 内直接读取当前 cache mask；仅依赖公共 fallback state
    的轻量更新，不刷新相似度表。
  - 新增 profile 字段并纳入报告：
      - draft_reroute_refresh_count
      - draft_reroute_refresh_layers_total
      - draft_reroute_refresh_ms_total
      - draft_reroute_refresh_visible_ms_total
      - 按 policy 汇总的 replay、draft-forward 和 refresh 中位数
  - 固定缓存性能测试中 refresh 应为初始化外零增量；prefetch 兼容测试中单独报告 refresh 对用户可见时
    延的贡献。

  ## Implementation Areas

  - 新增独立 policy/calibration 模块，封装策略张量计算、artifact 加载、cache revision state 和
    refresh。
  - 修改 heterogeneous MoE draft 接点，保存完整 router probabilities 供 entropy 策略复用，并将 raw
    metadata 与 execution routes 分离。
  - 扩展 placement 层，新增已 cache-valid 的 graph-safe plan 构建入口；不修改默认 baseline 语义。
  - 扩展配置、benchmark CLI、profile 汇总与结果 JSON，使 policy、refresh 和接受率均可复现比较。

  ## Validation

  - Reference 单测：对每个算法以小张量逐项比对 v2 语义，包括 miss gate、entropy bias、top-1
    protection、candidate pool、deviation guard、sim floor、contribution skip、collision guard、全
    miss fallback 与重复 route 归并。
  - 去重测试：比较 Alg2_PostSub 与 alg2_v2 的 canonical execution routes/weights 和 MoE 输出；由结
    果决定是否保留组合 option。
  - 配置/artifact/cache 测试：metadata mismatch 拒绝、cache revision 只在映射变化后触发刷新、非
    baseline 与 top_c>0 被拒绝。
  - CUDA Graph 正确性：每个最终公开策略使用 num_seqs=1、draft_top_c=0、temperature=0、greedy、
    slots_per_layer=96/64/32，要求 graph replay count > 0 且 standard graph == spec graph == spec
    eager token 输出一致。
  - 性能矩阵：prefetch 关闭；每个公开策略和未修改 baseline 在 slots_per_layer=96/64/32 下至少运行 3
    次，比较：
      - model_draft_graph_replay_ms / spec_run_draft_calls
      - spec_run_draft_infer_ms_total / spec_run_draft_calls
        每项中位数相对 baseline 增量必须 <3.0 ms。
  - 接受率矩阵：num_seqs=1、max_draft_tokens=8、standard_sampling、temperature=0.8、seed=0、
    input_len=128、output_len=256、prefetch 关闭，在 75%/50%/25% cache 下报告
    accepted_draft_tokens_total / drafted_tokens_total 及相对 baseline 增量。
  - Prefetch 冒烟：对性能合格且接受率最佳的策略开启当前 prefetch runtime，复测 graph replay、确定性
    行为、接受率与 refresh visible overhead。
  - GPU 条件：真实测试运行在任一单张空闲 A100 上即可；每次记录 GPU 占用、环境、命令、JSON 输出和日
    志路径，不要求同一 allocation 或同一物理 GPU。

  ## Documentation

  - 新增完整集成报告，记录所有算法的精确定义、原型对应关系、向量化实现、CUDA Graph 兼容分析、
    baseline 查表语义、预期/实测 overhead、artifact 与 refresh 设计、debug 过程、失败实现、优化迭
    代、测试命令、日志和最终对比表。
  - 在该报告附录记录 draft_top_c>0 未来方案：先按原始路由分数选择最多 top_c 个 miss 由 CPU 精确计算
    并锁定，reroute 只处理剩余 miss；说明现有 exact CPU graph bridge 的性能阻塞以及本次不实现、不验
    收该组合。
  - 对实现过程中新增的任何优化 option，文档必须明确其相对 v2 的行为变化、提出原因、实测质量收益与性
    能结果。

# Draft Expert Reroute 适配计划（`top_c=0`）

日期：2026-05-27

## 1. 目标与边界

本计划将 `pre_exps/expert_reroute/draft_decode_eval_v2.py` 中所有具有独立
行为的 cache-miss reroute 算法接入 `nano-vllm-moe` 的 speculative draft
前向，并满足以下约束：

1. 仅实现并验证 `draft_top_c=0` 的公开 reroute 选项。
2. 所有公开选项必须支持 CUDA Graph capture/replay，不引入 host decision、
   动态 route 数量或 replay 期间的 cache 更新。
3. 当前生产默认路径保持不变；不显式启用 reroute 时仍使用现有 baseline。
4. 正确性优先，其次验证接受率和性能。任一策略若无法满足 CUDA Graph
   或单次 draft forward 相对 baseline 增量 `<3 ms`，不得作为推荐选项。
5. 性能/接受率实验使用 `num_seqs=1`，在单张空闲 A100 上完成。
6. `draft_top_c>0` 仅记录方案，不在本阶段实现或测试。

算法适配以 v2 的 **cache-miss reroute 模型** 为语义参考。生产 baseline
不是 v2 的 `SkipAll`：它有独立的 round-robin miss replacement，因此实验
中需要同时保留生产 baseline 与 v2 算法的区别。

## 2. 公开接口与命名

新增可选配置：

```python
draft_top_c = 0
draft_reroute_policy = "round_robin"
draft_reroute_artifact = ""
```

公开运行时命名如下：

| 公开名称 | v2 名称 | 行为 |
|---|---|---|
| `round_robin` | 现有生产 baseline | miss 按 cache slot 轮转替换；默认不变 |
| `drop_miss` | `SkipAll` | miss 权重置零，仅保留 hit 贡献 |
| `entropy_cache_bias` | `Alg2_v2` | 以熵和 miss rate 控制 cached-expert logit bias |
| `bounded_cache_bias` | `HybridCP_v2` | 仅在原 router 候选池内 bias，并限制偏离 |
| `similarity_replace` | `PostSub_v2` | 以离线 conditional similarity 替换合格 miss |

`Alg2_PostSub` 不作为公开选项。v2 的第一阶段 `Alg2_v2` 已将所有仍然
miss 的 route 权重置为零；第二阶段只可能处理这些零权重 route，因此
其最终按 expert 聚合后的权重与 `Alg2_v2` 完全相同。适配验证中仍需用
参考测试确认这一等价性；若参考脚本或语义后续改变导致不等价，再恢复为
独立策略。

## 3. 生产 Baseline 的实际行为

当前 `top_c=0` draft baseline 并非无操作，也不是直接跳过 miss。对每个
MoE layer forward，`build_draft_plan_gpu()` 执行：

1. 建立长度为 `N` 的 substitution LUT：
   `cached(e) ? e : slot_to_expert_lut[e % num_slots]`。
2. 对原始 selected expert IDs 进行一次 LUT lookup，得到 cache-valid ID。
3. 将 cache-valid ID 映射到 slot，并构建 grouped GEMM layout。

因此 `round_robin` 自身已经包含 `O(N) + O(T*K)` 的 GPU tensor 操作成本。
新策略不是在“零开销”基线之上叠加计算：启用策略后会跳过该 substitution
LUT，并将已经 cache-valid 的 fixed-shape route 直接交给 grouped layout。

## 4. 固定形状执行语义

符号定义：

- `logits`: router logits，形状 `[T, N]`。
- `p = softmax(logits)`。
- `(rw, ri) = topk(p, K)`：原始 router top-k 权重与 expert IDs。
- `cm[e]`：当前 expert 是否位于 GPU cache 的 live mask。
- `MISS_GATE=0.25`、`SIM_FLOOR=0.40`、`GAMMA0=4.0`、
  `J_FACTOR=3`。

所有公开策略保持输出形状为 `[T, K]`。这对 CUDA Graph 是必要的：
不能根据 miss 数量动态压缩 route 或修改 GEMM layout 的容量。

对于 `drop_miss` 等语义上“跳过”的 route：

1. 将它的执行权重设为 `0`，从数值上移除贡献。
2. 将其物理 expert ID 映射到确定性的有效 cached slot，以保证固定布局
   中不存在无效 cache lookup。

这不意味着策略重新引入了 miss 的贡献。当前 fused path 在 expert 输出后
才乘 routing weight；因此零权重 route 可能仍执行一个 GEMM row，但其贡献
严格为零。动态删除该 row 会破坏图内固定形状，不在本阶段采用。

若某 token 的所有 route 权重都被清零，按照 v2 语义将其第一条 route
设为当前最小 cached expert、权重设为 `1`；随后所有 token 均做逐行
归一化。

## 5. 各策略精确定义与实现方式

### 5.1 `drop_miss`

v2 语义：

```text
fw = rw * cm[ri]
fi = ri
fw = normalize_with_empty_row_fallback(fw)
```

实现：

- `index_select` 读取 `[T, K]` hit mask；
- 权重与 mask 相乘；
- 零权重 ID 仅在执行计划中替换为有效 fallback expert；
- 使用 cache-valid direct plan，不构建 round-robin LUT。

CUDA Graph 分析：

- 全部为固定形状 tensor op；
- 无 `.item()`、CPU loop 或动态 compaction；
- 额外操作仅为 mask、`where`、reduce/normalize 和 `O(S)` fallback
  selection；
- 由于移除 baseline 的 `O(N)` LUT 构建，预计可能等于或快于 baseline。

### 5.2 `entropy_cache_bias`

v2 语义：

```text
miss_rate = mean(~cm[ri])
gate = clamp((miss_rate - 0.25) / (0.50 - 0.25), 0, 1)
H = -sum(p * log(p))
entropy_scale = clamp((H - log(N)*0.25) / (log(N)*0.50), 0, 1)
gamma = GAMMA0 * (0.2 + 0.8*entropy_scale) * gate
biased_logits = logits + gamma * cm
ni = topk(biased_logits, K)
protect original top-1 miss if it was displaced
nw = softmax(gather(logits, ni))
nw = nw * cm[ni]
```

实现：

- 保存 active policy 所需的未归一化原始 `rw`，不改变默认路径已有的
  in-place normalize 行为；
- 熵、gate、cached bias、top-k protection、residual miss drop 均以
  batched tensor op 实现；
- top-1 protection 使用比较与 `where` 替代 v2 中逐 token Python 循环；
- 最终进入 cache-valid direct plan。

CUDA Graph 分析：

- 固定执行 `softmax/reduction/topk/gather/where`；
- 无 host 同步；
- 相对 `drop_miss` 多一次全 expert 熵计算和 biased top-k，
  预计为小量 `O(T*N)` GPU 成本，必须实测 `<3 ms`。

### 5.3 `bounded_cache_bias`

v2 语义：

```text
gate = clamp((miss_rate - 0.25) / (0.50 - 0.25), 0, 1)
gamma = GAMMA0 * gate * clamp(entropy / log(N), 0, 1)
pool = topk(logits, J), J = min(K * J_FACTOR, N)
biased_logits = logits + gamma * (cm & pool)
ni = topk(biased_logits, K)
if sum(raw rw for displaced original routes) > 0.20:
    ni = ri
nw = softmax(gather(logits, ni)) * cm[ni]
```

实现：

- 以 `scatter_` 构建固定 `[T, N]` candidate-pool mask；
- 以 broadcast comparison 判断原始 route 是否仍在 candidate top-k 中，
  计算 displaced raw router weight；
- 以 `where` 对整行执行 deviation revert；
- 保持 v2 的 raw probability 语义，不使用已归一化执行权重代替；
- 输出进入 cache-valid direct plan。

CUDA Graph 分析：

- 所有操作为固定 tensor op；
- 比 `entropy_cache_bias` 增加 top-`J`、pool mask 和 displacement
  reduction；
- 预计是 bias 类策略中开销最大的一个，需以实际 draft forward
  数据决定是否推荐。

### 5.4 `similarity_replace`

所需离线 artifact：

```text
cond_sim[L, N, N]  # miss expert 到替代 expert 的 conditional similarity
skip_err[L, N]     # 预估输出范数/贡献尺度
```

v2 语义（对每个原始 miss route `e`）：

```text
gate = clamp((miss_rate - 0.25) / (1 - 0.25), 0, 1)
contribution = rw[e] * skip_err[e]
candidate = argmax_j cond_sim[e, j], where cm[j] is true and j != e

replace iff:
  gate > 0
  contribution >= 0.10 * mean_contribution
  cond_sim[e, candidate] >= 0.40
  candidate not in original top-k IDs

replacement weight = rw[e] * cond_sim[e, candidate] * gate
otherwise miss weight = 0
```

实现：

- artifact 在初始化时加载并按 layer 放到设备上；
- 每次 forward 从 `cond_sim` gather 当前 `[T, K, N]` 行；
- 使用 **live** `cm` mask 当前可选 cached experts，再对末维 `max`；
- 以 batched comparison 实现 original-top-k duplicate rejection；
- hit route 保持原权重，未满足条件的 miss route 置零；
- 输出进入 cache-valid direct plan。

缓存更新接口决策：

- 不增加 cache revision/refresh 接口。
- 原因是缓存集合可能由已有 prefetch/placement 改变；预先缓存 substitution
  map 需要在每次变化时重建或验证，增加 refresh 时间且容易过期。
- 本方案始终使用 live cache mask，refresh overhead 为 `0`。
- 代价是每个 forward 执行 `[T, K, N]` gather/masked-max；在
  `num_seqs=1` 的 draft 场景规模有限，但它是四个新策略中预计开销最大
  的策略，必须实测预算。

CUDA Graph 分析：

- artifact 是持久设备 tensor，cache mask 是已存在的 live device tensor；
- gather、mask、max、compare 和 normalize 都可 capture；
- 不读取 CPU 状态、不分支创建新执行形状；
- graph 兼容，但开销通过性能测试而非推断确认。

## 6. 代码落点

| 文件 | 计划改动 |
|---|---|
| `nanovllm/scheduling/draft_reroute.py` | 策略常量、artifact loader、四种 active policy 的向量化实现 |
| `nanovllm/config.py` | `draft_reroute_policy` 与 `draft_reroute_artifact` 配置校验；限制非 baseline 策略仅适用于 `top_c=0` |
| `nanovllm/models/qwen3_moe.py` | 在 draft 路由后、执行计划前调用策略；baseline 保持原路径 |
| `nanovllm/expert/placement.py` | 增加已 cache-valid fixed routes 的 direct GPU draft plan |
| `nanovllm/engine/model_runner.py` | 初始化并向各 MoE layer 注入策略/artifact |
| `examples/heterogeneous_benchmark_case.py` | 暴露 benchmark CLI 选项 |
| `examples/benchmarks/draft_standard_decode_forward_bench.py` | 透传 policy/artifact，用于 draft forward 测量 |
| `benchmarks/scripts/spec_verify_expert_count_stats.py` | 透传策略并汇总接受率/graph replay/耗时 |
| `pre_exps/expert_reroute/draft_decode_eval_v2.py` | 导出 `cond_sim` 和 `skip_err` artifact，用于 `similarity_replace` |
| `tests/test_draft_reroute.py` | 算法语义、映射与 direct-plan 测试 |
| `tests/test_config_prefetch.py` | 配置兼容和非法组合测试 |

关键调用顺序：

1. Router 产生原始 `(logits, p, ri, rw)`。
2. 记录基于原始 router 的 runtime metadata，避免 reroute 污染 cache
   观测和后续分析。
3. 仅当处于 draft 且 policy 非 `round_robin` 时，执行 reroute policy。
4. active policy 输出 cache-valid fixed routes，走 direct draft plan；
   `round_robin` 严格走现有 substitution LUT 路径。
5. MoE forward 在固定 shape 下执行。

## 7. CUDA Graph 与开销验收

所有公开策略的实现规则：

1. 禁止 `.item()`、`.tolist()`、Python 逐 token/route 控制流进入运行
   forward。
2. 禁止根据 miss 数量执行 `nonzero` route compaction 或创建变长 plan。
3. 禁止 replay 期间 CPU 计算替代 expert 或更新 cache。
4. 允许 graph 内固定 tensor 算子，例如 `topk`、`index_select`、
   `scatter_`、`gather`、`where`、reduction 和 normalize。

预期开销排序：

| 策略 | 相对 baseline 替换掉的操作 | 新增主要操作 | 预期 |
|---|---|---|---|
| `drop_miss` | `O(N)` round-robin LUT | `O(T*K)+O(S)` mask/normalize | 可能更快或持平 |
| `entropy_cache_bias` | `O(N)` LUT | `O(T*N)` entropy/bias/top-k | 小幅增加 |
| `bounded_cache_bias` | `O(N)` LUT | `O(T*N)` pool/bias/deviation | 高于 entropy |
| `similarity_replace` | `O(N)` LUT | `O(T*K*N)` gather/max | 预计最大 |

验收标准：

- 每个公开非 baseline policy 的 CUDA Graph draft replay count 必须大于零。
- 在相同参数下，策略的 graph 与其 eager 运行必须在确定性采样中 token
  一致。
- 策略的 `draft_forward_ms_avg - round_robin_draft_forward_ms_avg < 3 ms`。
- 不满足预算的策略保留为实验实现或移除公开暴露，不作为可启用推荐路径。

## 8. 正确性与性能测试矩阵

### 8.1 CPU/单元级算法语义

验证内容：

1. 公开名称与 v2 名称映射正确。
2. `Alg2_PostSub` 与 `entropy_cache_bias` 的按 expert 聚合权重等价。
3. 四个 active policy 的向量化输出，按 expert 聚合后与逐项 v2 参考
   计算一致。
4. 每个 active policy 的非零 route 都落在当前 cache 中。
5. direct plan 不生成 round-robin substitution LUT。
6. artifact shape/dtype/layer 数校验正确。

测试命令：

```bash
python -m pytest -q \
  tests/test_draft_reroute.py \
  tests/test_config_prefetch.py \
  tests/test_placement_spec.py \
  tests/test_draft_cuda_graph.py \
  tests/test_draft_standard_decode_forward_bench.py \
  tests/test_model_runner_spec_modes.py
```

### 8.2 A100 CUDA Graph 正确性

公共设置：

```text
num_seqs=1
temperature=0
draft_top_c=0
slots_per_layer=32             # 25% cache, N=128
max_draft_tokens=4
spec_enable_prefetch=false
enforce_eager=false            # CUDA Graph
policies=round_robin,drop_miss,entropy_cache_bias,bounded_cache_bias,similarity_replace
```

步骤：

1. 运行 `standard` CUDA Graph，记录确定性 token/digest。
2. 运行默认 `spec round_robin` CUDA Graph，确认默认路径可 replay。
3. 若 `round_robin` 与 `standard` 不一致，先用同参数
   `spec round_robin --enforce-eager=true` 诊断：
   - graph 与 eager 不一致表示 graph 正确性阻塞，停止策略比较并修复；
   - graph 与 eager 一致但均不同于 standard，记录为已有 spec/baseline
     的严格 standard 对齐限制，不将其错误归因于新 reroute。
4. 对每个 active policy 分别运行 graph 与 eager，校验确定性 token 对齐
   并记录 graph replay count。

### 8.3 接受率与 draft forward 性能

在完全相同的 A100、prompt、cache ratio 与运行参数下，对
`round_robin` 及四个 active policy 分别记录：

```text
accepted_draft_tokens_total
spec_draft_tokens_total
acceptance_rate
model_draft_graph_replay_count
draft_forward_ms_avg
delta_draft_forward_ms_vs_round_robin
```

判定：

- 接受率目标是至少有一个 v2 策略相对当前 `round_robin` baseline 提升；
- 所有可对外启用策略必须保留 graph replay；
- 任一策略的 draft forward 增量达到或超过 `3 ms` 时，在结论中明确
  标记为不满足性能目标，不以默认或推荐 option 暴露。

测试原始文件与汇总写入：

```text
results/reroute_impl_20260527/
docs/draft_reroute_topc0_implementation_20260527.md
```

## 9. 调试与记录要求

实现过程必须记录以下信息：

1. baseline 实际 substitution 算法及其已有开销，避免将其误认为零开销
   或 `SkipAll`。
2. 算法从 v2 Python 控制流改写为 fixed-shape tensor op 时的逐项语义
   对照。
3. artifact 生成、packed MoE 权重/API 差异及修复过程。
4. CUDA Graph capture/replay 失败的调用点、原因与修复。
5. 确定性输出若未与 standard 对齐，应区分已有 baseline/spec 问题与
   active reroute 引入的问题。
6. 所有测试命令、A100 可见设备、原始 JSON/log 位置、接受率和耗时
   矩阵。

实施结果和调试日志收敛到实现报告：

```text
docs/draft_reroute_topc0_implementation_20260527.md
```

## 10. 2026-05-28 执行进展与验收结果

本轮按当前指令暂不以 standard/spec 输出对齐作为阻塞条件，优先验证
CUDA Graph draft forward 开销与相对生产 `round_robin` baseline 的实际
接受率。

### 10.1 先决修复

在 A100 长序列测试中，`round_robin` baseline 即可复现两处 speculative
KV 边界缺陷，因此它们不是 active reroute 引入的问题：

1. 临时 draft token 将尾 block 填满并生成 hash 后，rollback 或 partial
   accept 保留了一个实际已变回 partial 的 hashed block。修复为在保留
   partial tail 时清除 hash 及 hash 映射。
2. verify 会消费最后一个 proposed draft token 作为输入，但原逻辑没有
   为该最终输入预留 KV slot。修复为进入 verify 前追加最后一次 slot
   reservation。

此外，benchmark 的 `--input-len` 原先按文本片段生成：请求值 `128`
实际 token 数为 `619`，加上 `output_len=256` 后超过
`max_model_len=512` 并触发 scheduler 断言。现已改为经过实际 tokenizer
截断到准确的输入 token 数，并在 JSON 中写入 `actual_input_tokens`；
显式 `slots_per_layer` 运行也改为记录有效 cache ratio。

### 10.2 性能修复

`entropy_cache_bias` 的初始正确实现能够提升接受率，但每层 entropy/bias
逐算子 replay 使 draft forward 超出 `3 ms` 增量门槛。保持 v2 语义不变
的修复包括：

- 合并最终归一化的重复 reduction，去除无需的 bool-to-float 中间张量；
- 利用最终 retained-route 归一化，将 selected-logit 二次 softmax 等价
  替换为已存在 full-router probabilities 的 gather；
- 使用仓库已有的
  `torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")`
  模式融合 policy forward，再由外层 CUDA Graph 捕获执行。

### 10.3 正式验收数据

运行环境：Slurm job `26765`，`gpu16`，分配卡
`CUDA_VISIBLE_DEVICES=0`（A100-SXM4-80GB，运行开始时空闲）。

公共设置：

```text
actual_input_tokens=128, output_len=256, max_draft_tokens=8, num_seqs=1
temperature=0.8, acceptance_strategy=standard_sampling, seed=0
slots_per_layer=32 (25% of 128 experts), prefetch=false, draft_top_c=0
enforce_eager=false, max_model_len=512
```

| Policy | Accepted / Drafted | Acceptance rate | Draft forward avg (ms) | Delta vs `round_robin` (ms) | Graph replays |
|---|---:|---:|---:|---:|---:|
| `round_robin` | 221 / 271 | 0.815498 | 16.006459 | 0.000000 | 271 |
| `entropy_cache_bias` | 224 / 246 | 0.910569 | 16.259654 | +0.253195 | 246 |

结论：`entropy_cache_bias` 已实际超过生产 `round_robin` baseline 的接受率
（`+0.095071`），同时 draft forward 增量 `+0.253195 ms < 3 ms`，且
CUDA Graph replay 生效。本轮所要求的速度与至少一个算法接受率提升目标
均达到。

原始结果和日志：

```text
results/reroute_impl_20260527/job26765_formal_token_exact_ratio25/round_robin_ratio25.json
results/reroute_impl_20260527/job26765_formal_token_exact_ratio25/entropy_cache_bias_ratio25.json
/home/mumura/moe_spec/logs/reroute_job26765_formal_token_exact_round_robin_ratio25_20260528.log
/home/mumura/moe_spec/logs/reroute_job26765_formal_token_exact_entropy_cache_bias_ratio25_20260528.log
```

回归验证：

```text
KV/spec/reroute/config/placement/model-runner and benchmark focused suite: 60 passed
```

## 附录 A：`draft_top_c>0 + reroute` 设计（本阶段不实现）

语义要求：

1. 从原始 router top-k route 出发识别 misses。
2. CPU 最多计算 `draft_top_c` 个 **原始路由匹配** 的 distinct miss
   experts，按对应原始 routing score 总和从高到低选择。
3. 对剩余未计算的 miss routes 应用所选 `top_c=0` reroute policy。
4. CPU 执行 route 保留原始非零权重；reroute 与 CPU 输出在原始 route
   次序中确定性合并。

未来实现若要保持 CUDA Graph，可采用：

- 固定容量 CPU-route mask 与任务 buffer；
- 图内固定形状 GPU reroute buffer；
- 已验证的 captured CPU/GPU bridge；
- 不依赖 Python 动态选取或变长 route plan 的 device-side selection。

该组合会新增 CPU 任务调度、CPU/GPU merge 以及 graph bridge 成本，必须
单独评估正确性、接受率和 `<3 ms` 性能约束；在完成验证前不对外开放。
