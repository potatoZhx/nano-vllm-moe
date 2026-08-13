# Reclaim unused staging slots as active expert cache

## 发现

最终 preset 使用 `prefetch_runtime_mode=draft_segment_indexed`。该路径通过
`reserve_active_slot_for_prefetch_deferred()` 和 `begin_async_put_to_active()` 直接把
权重传到 active victim slot；它不调用 staging reservation/copy。但是通用默认值仍为
每层 2 个 staging slots，因此 48 层常驻 96 个从不使用的完整 expert buffers。

本优化把每层布局从等价的 `active10 + staging2` 改为 `active12 + staging0`：

- `cache_ratio`: `0.075`（取整每层基准 10）→ `0.09375`（12/128）；
- profile-weighted active budget: 480 → 576 slots；
- `prefetch_staging_slots_per_layer`: 2 → 0；
- direct-active prefetch 算法、transfer budget、K、segment 和 sampling 均不变。

这不是额外堆显存：它把不参与当前 runtime 的 staging 容量重新分配给有效 cache。
实测 KV cache 反而从 3 blocks（768 tokens）增加到 8 blocks（2048 tokens），表明去掉
staging 还释放了额外 allocator/graph 余量。

## 容量边界

256-token 筛选：

| 布局 | TPOT (ms/token) | step ms/verify | realized CPU experts |
|:---|---:|---:|---:|
| active10 + staging2 | 70.942 | 118.189 | 58.73 |
| active12 + staging0 | **69.462** | **114.973** | **57.17** |

下一整数档 active13/staging0（624 active slots）在初始化 warmup 申请 224 MiB fallback
tensor 时 OOM，当时 GPU0 仅余 197 MiB。因此 active12 是当前实现和 warmup 口径下的
最大安全整数档，不报告不可启动配置的性能。

## 三次 512-token 配对复测

| repeat | active10/staging2 | active12/staging0 | delta (ms) |
|---:|---:|---:|---:|
| 0 | 72.631 | 69.914 | −2.717 |
| 1 | 70.026 | 64.984 | −5.042 |
| 2 | 67.620 | 64.623 | −2.996 |
| mean | **70.092** | **66.507** | **−3.585** |

三 seed 全部正收益，均值降低 **5.11%**（1.054×）。三轮均固定生成 512 token 并通过
长度验证。相对 KTransformers BF16 122.35 ms 快 45.6%（1.84×）；相对当日同 prompt
F16 stable replay 125.886 ms 快 47.2%（1.89×）；相对历史 F16 最好值 81.90 ms 仍快
18.8%（1.23×）。

结果目录：

- `results/single_weight_f16_k1_vpb2_seg16_active12_staging0_256/`
- `results/single_weight_f16_k1_vpb2_seg16_active12_staging0_512_repeats3/`
- `results/single_weight_f16_k1_vpb2_seg16_active12_staging0_512_repeats3_r2/`
- OOM 边界：`results/single_weight_f16_k1_vpb2_seg16_active13_staging0_256/`
