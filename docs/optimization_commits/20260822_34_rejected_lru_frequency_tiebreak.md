# 34. t16 LRU 同步频率破同分否决

日期：2026-08-22

## 一句话总结

在 LRU 主顺序完全不变的前提下，用 lifetime access count 打破同一步访问的并列，公平
16-thread 一请求 TPOT 从 **52.566035** 回退到 **55.416990 ms/token（+5.42%）**；候选
实现、CLI 和测试已全部撤回，当前 ghost8 + LUT fusion preset 保持不变。

## 候选动机与边界

t16 lifecycle trace 的离线扫描发现，2618 次 replacement 中约 602 次存在“另一个 resident
expert 的 `last_access_step` 相同、但 `access_count` 更低”。当前 LRU 在这种情况下按 slot
index 破同分，因此这 602 次只是一个值得验证的选择差异上界，不代表 602 次可避免 miss。

候选规则刻意保持保守：

- `last_access_step` 仍是唯一主排序，较老 expert 永远先淘汰；
- 只有两个 expert 的最后访问 model step 完全相同，才优先淘汰 lifetime count 更低者；
- 不改变 phase1 budget1、ghost8、LUT fusion、prefetch budget、cache 容量或 CPU/GPU route；
- 开关默认关闭，原 preset 在候选测试期间也保持原行为。

定向测试覆盖开关默认关闭、同龄时频率破同分、LRU 主顺序不可被频率覆盖、配置透传与
全部内置 preset 的 16-thread 约束，共 **72 项通过**。

## 公平一请求 TPOT

共同口径为 MMLU-Pro validation sample 0、seed 20260719、temperature 0.6、107-token prompt、
固定输出 512 token、profile 全关、single-weight F16、GPU 0.98/KV 1536。CPUInfer 启动日志
明确显示 `2 subpools, [0:8] [1:8]`，即 total 16 threads；候选唯一变量是上述破同分规则。

| 配置 | TPOT | decode tok/s | prefill | decode | rounds | mean round | p50/p90/p95/max |
|:---|---:|---:|---:|---:|---:|---:|:---|
| 当前最佳 | **52.566035 ms** | **19.024** | 10.443 s | 26.861 s | 265 | **101.363 ms** | 94.486/134.610/141.977/187.491 ms |
| frequency tie-break | 55.416990 ms | 18.045 | 12.361 s | 28.318 s | 265 | 106.861 ms | 100.087/139.470/147.009/208.270 ms |
| 变化 | **+2.850955 ms / +5.42%** | -5.14% | +1.919 s | +1.457 s | 0 | **+5.42%** | +5.93%/+3.61%/+3.54%/+11.08% |

候选生成 512/512 token、fixed-length validation 通过。两次 sampling digest 不同，但 decode
rounds 同为 265，且 mean、p50、p90、p95、max round wall 全部回退，没有可保留的性能信号。
结果目录：

```text
results/tpot_t16_b1_ghost8_lutfuse_lrufreq_20260822/
```

候选 digest：
`ee58d122317f3068b29d106c4f4eb56ecd2a0473a96d56444cfce41f68cb0ee0`。

## 原因判断与决定

离线的 602 次只说明“选择会不同”，没有计算被淘汰 expert 的下一次使用距离、reload 成本、
source、rank 或 CPU exposed tail。相同 `last_access_step` 往往表示 expert 在同一个 route group
共同使用；lifetime 高频不等于近期仍会复用，反而会把旧阶段积累的历史热度带入当前局部。
slot-index 稳定顺序也可能隐含 publication/route 的局部性，不能当作纯随机噪声删除。

因此：

- 不保留 frequency tie-break 的 runtime、CLI 或 preset；
- 不再尝试 lifetime LFU/LRU 的简单加权或静态破同分扫描；
- source/rank-aware admission 仍保留为方向，但必须直接预测 next reuse / CPU tail saved，并减去
  victim reload 与 publication 成本，不能再以 access count 作为无成本代理；
- 当前推荐继续是
  `k2_dynamic_f16_3080_active14_phase1_recent_b1_ghost8_lutfuse`，公平最低点仍为
  **52.566035 ms/token**。
