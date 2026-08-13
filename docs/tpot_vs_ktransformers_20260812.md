# nano-vllm-moe 与 KTransformers TPOT 对照、定位和优化

日期：2026-08-12  
硬件：RTX 3080 10 GiB，2 × Xeon Gold 5218R（2 NUMA，40 物理核）  
模型：Qwen3-30B-A3B，最终 GPU expert cache ratio `0.09375`

## 结论

旧 nano 结果 `258.413 ms/token` 不是这台机器在 7.5% cache 下的性能下界。
它同时受到外层 `taskset`、不同/较慢的 CPU MoE backend、以及在错误基线上选择
`K=6` 的影响。修正后：

- 当前单请求最终配置为 **single-weight + llamafile F16 + fixed K1 +
  verify prefetch budget 2 + segment16 + active12/staging0**。三次独立 engine、每次
  512 个固定输出 token 的完整 decode 墙钟 TPOT 为
  fixed grouped GEMM 后为 `63.432 / 66.085 / 66.811 ms/token`，均值
  **65.443 ms/token**；相对同布局 autotune 均值 `66.507 ms/token` 降低 1.60%。
  相对 KTransformers 最可靠的 BF16 `122.35 ms/token` 快 **46.5%（1.87×）**；
  相对其历史 F16 `81.90 ms/token` 快 **20.1%（1.25×）**；相对本页此前
  single-weight BF16 K3 的 `91.287 ms/token` 再快 **28.3%（1.39×）**。F16 KT
  行的线程数、
  prompt 和原始日志
  留存方式不同，因此同 dtype 百分比是当前最严格参考，不是逐 token 配对 A/B。
- 单请求、BF16、K3、512 个固定输出 token：nano 完整请求墙钟 TPOT
  **109.946 ms/token**；KTransformers BF16 的 61-step stable replay 为
  **122.35 ms/token**。这是较早的同 dtype 结果；nano 的更严格口径仍快
  **10.1%**。
- F16 true batch 的最终 nano 路线不是投机解码，而是 exact decode。优化配置的
  B2 三次完整墙钟 TPOT 为 `110.237 / 117.120 / 125.859 ms`，均值
  **117.739 ms**、中位数 **117.120 ms**；KT 为 `121.84 ms`。B3 两次为
  `156.084 / 159.096 ms`，KT 为 `163.65 ms`；B5 两次为
  `154.511 / 154.854 ms`，KT 为 `251.92 ms`。
- B2 仍有尾延迟问题：最佳完整轨迹的 stable replay p50 为 `105.46 ms`，但
  p95 为 `147.07 ms`，高于 KT 的 `131.46 ms`。因此结论是均值和聚合吞吐已
  超过 KT，而不是所有尾延迟指标都超过 KT。

## F16 K1 与动态 draft-length 复测

single-weight 消除换页临界点后，重新在当前 fixed legacy 后端上筛选了 K，而不是
沿用旧 AVX2/高 cache 实验得到的动态长度结论。所有筛选行共用一个已加载 engine、
同一 prompt、temperature `0.6`、固定输出 256 token，并开启轻量 profile：

| fixed K | TPOT (ms/token) | acceptance | draft ms/call | verify ms/call |
|---:|---:|---:|---:|---:|
| **1** | **75.959** | 0.705 | 32.68 | 94.79 |
| 2 | 85.613 | 0.522 | 27.39 | 115.85 |
| 3 | 83.984 | 0.513 | 25.20 | 132.75 |
| 4 | 90.728 | 0.443 | 25.16 | 144.86 |
| 5 | 109.542 | 0.322 | 25.32 | 152.98 |
| 6 | 103.703 | 0.354 | 24.25 | 172.93 |

F16 使这台 Cascade Lake 主机上的 legacy CPU expert kernel 明显快于 BF16；同时 K1
把 verify 固定在 `qlen=2` 的 grouped fast path。K 增大后，单次 draft 虽略快，
但 acceptance 降低且 verify 随 qlen 增长，节省的 verify 轮次无法偿还额外计算。

### 现有动态长度配置是否有收益

用当前 F16/single-weight 路线复测了现有静态 TPOT stop：`Kmax=6`、`td=19 ms`、
`tv=80 ms`、`min_steps=1`、`first_increase`。结果为 **83.627 ms/token**；112 个
verify round 中 111 个触发提前停止，预算 draft step 为 668、实际执行 274，平均
实际 K 为 2.45。它确实避免了 K5/K6 的最坏区域，但仍比同口径 fixed K1 的
`75.959 ms/token` 慢 **10.1%**。

原因不是动态逻辑没有生效，而是 predictor 将工作点放在 K2/K3；实测 verify 已从
K1 的 `94.79 ms/call` 增至 `121.34 ms/call`，acceptance 又从 `0.705` 降至
`0.522`。旧动态 history 模型在 K12 实验中也因增加 verify 次数而未超过 fixed K；
当前新后端上的直接复测进一步说明，生产配置应使用 fixed K1。把动态策略调到总是
K1 只能退化成 fixed K1，不能构成额外优化。

### Verify prefetch budget 筛选

固定 F16/K1 后筛选每个 boundary 的 prefetch budget。关闭 prefetch 会降低 draft
计算量，但 acceptance 从约 0.70 降到 0.39，净 TPOT 变差；预算 2 是离散筛选最优点：

| prefetch budget | TPOT (ms/token) | acceptance | prefetch submits |
|---:|---:|---:|---:|
| off | 85.317 | 0.393 | 0 |
| 1 | 78.949 | 0.671 | 1717 |
| **2** | **72.317** | **0.705** | 2284 |
| 3 | 76.313 | 0.604 | 3047 |
| 4 | 75.959 | 0.705 | 3484 |

这些是 256-token 筛选值。budget 2 的第一次 512-token 验收为
**73.064 ms/token**；完整 decode 为 315 个 speculative round，draft/verify 分别为
`32.85/83.89 ms/call`，生成恰好 512 token，`max_repeated_token_run=1`，digest 为
`211a059ddcd22624b3caa78d0eb7f9f7e01c5ea21cc6c0b8abc1e0eba12396f9`。

随后使用最终 `k1_f16_3080` preset 做三次独立 engine 复测，得到
`69.682 / 73.837 / 70.689 ms/token`，均值 **71.403 ms/token**。原 benchmark 在
同一进程执行第二个 repeat 时暴露 CUDA OOM：`atexit` bound callback 和
`SpeculativeEngine` 仍持有旧 rank-0 ModelRunner。清除这些引用、GC 并释放 CUDA
cache 后，第二、三次 engine 均能完整重建。修复只保证重复实验可靠，不计作单次
TPOT 性能收益。

使用本地 Qwen3 tokenizer 解码后，文本正常解释 speculative draft/verify、MoE
routing metadata、bounded transfer budget 和 cache eviction；无 replacement
character 或无意义重复。与本文其他 sampling 实验一样，不同 draft/prefetch 轨迹会
改变 RNG 消耗，因此正确性验收不是 token digest 跨配置 bitwise 相同，而是 exact
acceptance 语义、长度、数值测试与文本检查。

上述组合已固化为 `--optimized-config k1_f16_3080`，避免再次误用历史 K3/BF16
preset。结果位于：

- `results/single_weight_f16_k1_6_screen_256/`
- `results/single_weight_f16_k6_dynamic_default_256/`
- `results/single_weight_f16_k1_no_prefetch_256/`
- `results/single_weight_f16_k1_vpb{1,2,3}_256/`
- `results/single_weight_f16_k1_vpb2_512/`
- `results/single_weight_f16_k1_vpb2_512_repeats3/`
- `results/single_weight_f16_k1_vpb2_segment16_512_repeats3{,_r2}/`
- `results/single_weight_f16_k1_vpb2_seg16_active12_staging0_512_repeats3{,_r2}/`

### Segment boundary 筛选

对 48 层 forward 的 segment 8/12/16 做筛选。segment 8 为 `76.669 ms/token`，
增加 graph/prefetch boundary 且为负收益；segment 16 的 256-token TPOT 为
`70.942 ms/token`，per-verify step 从 segment 12 的 `122.899 ms` 降到
`118.189 ms`，prefetch submit 从 2284 降到 1892。三次 512-token 配对中，
segment 16 为 `72.631 / 70.026 / 67.620 ms/token`，均值 **70.092 ms/token**；
segment 12 为 `69.682 / 73.837 / 70.689 ms/token`，均值 **71.403 ms/token**。
均值改善 1.84%，但 repeat 0 回退 2.949 ms，因此这里只认定为小幅正收益，而非稳定
大幅提升。最终 preset 已改用 segment 16。

### 回收未使用 staging 为 active cache

`draft_segment_indexed` runtime 的热路径只做 direct-active prefetch，不使用通用默认的
2 个 staging slots。最终布局因此从每层 `active10 + staging2` 改为
`active12 + staging0`，profile-weighted active budget 从 480 增到 576；KV capacity
还从 3 blocks 增至 8 blocks。三次 512-token 配对全部正收益：新布局
`69.914 / 64.984 / 64.623 ms/token`，均值 **66.507 ms/token**；旧布局
`72.631 / 70.026 / 67.620 ms/token`，均值 **70.092 ms/token**，降低 5.11%。
active13/staging0 在 warmup 需要额外 224 MiB 时 OOM，因此 active12 是当前可用容量
上限。完整证据见 `docs/optimization_commits/20260813_06_reclaim_staging_cache.md`。

### Decode-aware fixed grouped GEMM

原 Triton autotune key 不含 route 数 `M`，会把大 prefill 选出的 config 复用于
`M<=16` decode；autotune 本身还临时申请 256 MiB cache-flush tensor。对 Qwen3 两个
projection shape 独立测得 decode tiling 并直调底层 JIT 后，active12/ctx8192 三次均值
从 66.507 降到 **65.443 ms/token**（1.60%），并消除 256 MiB 峰值。逐 seed 只有一个
更快，故只视为小幅 mean 优化；完整配置、正确性和不利数据见
`docs/optimization_commits/20260813_07_fixed_decode_grouped_gemm.md`。

### Workload-sized CUDA memory warmup

旧初始化按 `max_model_len=8192` 做一条 8192-token synthetic prefill，远大于本测试的
67-token prompt，并把该瞬时峰值用于 KV cache sizing。新增独立
`warmup_model_tokens` 后，最终 preset 用 1024-token 峰值测量但仍保留
`max_model_len=8192`。active12 的 KV capacity 从 8 blocks / 2048 tokens 增到
49 blocks / 12544 tokens；256-token 筛选从 67.881 降至 66.396 ms/token，生成长度
与输出校验通过。单次 TPOT 不并入最终三次均值；该配置只适用于最大未分块 prefill
不超过 1024 token 的工作负载。完整证据见
`docs/optimization_commits/20260813_08_workload_sized_warmup.md`。

### Top-k/top-p 与当前 KT F16 复测

为了缩小采样口径差异，nano 新增了 `SamplingParams.top_k/top_p` 及 benchmark
`--top-k/--top-p`。普通 sampler 和 speculative acceptance 的 target `p`、draft
`q` 都使用相同的 temperature → top-k → top-p 分布，reject 时仍按
`max(p-q, 0)` 重采样，因此不是只改 draft token、却继续用旧 q 验证的近似实现。

当前 K1/vpb2 加 `top-k=20, top-p=0.95` 的 256-token TPOT 为
**81.960 ms/token**，慢于无过滤的 `72.317 ms/token`。acceptance 从 `0.705`
降到 `0.555`，decode round 从 150 增到 164；过滤/acceptance 自身也增加少量开销。
所以最终性能 preset 保持 `top_k=0, top_p=1.0`，但现在可以显式复现 KT 的采样
设置。结果位于 `results/single_weight_f16_k1_vpb2_topk20p095_256/`。

此外在同一天直接重跑 KT F16 GGUF：使用相同问题文本、temperature `0.6`、
top-k 20、top-p 0.95、16 CPUInfer 线程；KT 的 chat/tokenizer 包装后 prompt 为
79 token（nano raw tokenizer 为 67），生成 64 token。61-step stable graph replay
为 **125.886 ms/token**，p50 `101.326 ms`、p95 `269.842 ms`；nano 最终
`65.443 ms/token` 的三次均值相比快 **48.0%（1.92×）**。这个当前复测比历史 F16
`81.90 ms/token` 慢得多且尾延迟较大，因此正文仍同时报告历史 81.90 这一更保守
参考；无论使用哪个 F16 KT 基线，nano 当前结果都更快。

## Single-weight 更新

`single-weight` 分支消除了 legacy llamafile `kt_direct` 路径的两份常驻 CPU
expert 权重。旧路径在 nano 的 `cpu_expert_pool` 中保留 expert-major raw tensor，
`cpuinfer_ext.moe.MOE.load_weights()` 又把同一层复制到 NUMA-local gate/up/down
buffer。新路径仍使用完全相同的 NUMA-local buffer 和 forward kernel，但在
`load_weights()` 完成后通过只读地址接口为这些 native buffer 创建 non-owning
PyTorch view，并原地替换 raw pool：

```text
旧：raw cpu_expert_pool + CPUInfer NUMA-local weights
新：CPUInfer NUMA-local weights <- zero-copy views <- GPU cache/prefetch
```

gate/up 按 intermediate row、down 按 intermediate column 组合成逻辑 tensor；GPU
cache 初始放置、direct-active prefetch、staging prefetch 和 fallback workspace 都能
从同一组 NUMA shard 复制。forward 热路径以及 CPUInfer 的内存布局、线程池、GEMM
调用均未改变。初始化时仍存在一层大小的临时 expert-major tensor，但每层接管后立即
释放，不再有第二套全模型常驻副本。

### 先澄清：这不是“Torch 文件 + GGUF 文件”各存一份

原先的“双份”现象确实来自 GPU expert cache 和 kt-kernel 对权重源的不同要求，
但更准确的说法是：一份是 nano 保留的 **Torch 逻辑 tensor**，另一份是
CPUInfer 为 legacy llamafile kernel 分配的 **native NUMA buffer**。本次测试的权重从
Qwen3 的 BF16 `safetensors` 读入，运行时并没有再打开或保留一个 `.gguf`
文件。

`MOEConfig.gate_type/up_type/down_type` 用的 `30` (BF16) 和 `1` (F16) 是
**GGML dtype 枚举值**，它们只告诉 kernel 每个元素如何解码和计算。GGUF 则是带
metadata 和多个 tensor 的文件容器。“使用 GGML type”不等于“内存里还放着一份
GGUF 文件”。在这条 BF16/F16 legacy 路径上，native buffer 实际是每元素 2 byte、
block size 1 的连续行主序数组，所以能安全重建为 Torch 视图。

旧路径的完整生命周期是：

1. `heterogeneous_loader.py` 从 safetensors 读取每个 expert 的 gate/up/down，
   把 gate 和 up 合并为 `[2I, H]` 的 BF16 `gate_up`，并在
   `cpu_expert_pool` 中保留 `[2I,H]` 和 `[H,I]` 两个 Torch tensor。GPU
   cache/prefetch 以它们作为回源。
2. legacy backend 为当前层临时 stack 出 expert-major gate/up/down；F16 模式
   还会在这一步从 BF16 转为 F16，然后把三个 `data_ptr()` 传给
   `cpuinfer_ext`。
3. `LLAMA_MOE_TP::load_weights()` 在每个 NUMA pool 内自己分配 gate/up/down
   buffer，并用 `memcpy` 按 intermediate 维切片。这些数组才是 CPU forward
   每层 GEMM 直接读取的权重。
4. 临时 stack 可以释放，但步骤 1 的全模型 Torch pool 还必须为 GPU cache
   refill 存活，因而与步骤 3 的 native buffer 形成两份常驻 CPU 权重。

### Single-weight 如何同时满足两个消费者

native 补丁只增加 `get_weight_ptrs()`，返回每个 NUMA 分片的 gate/up/down 首地址，
没有改动 `load_weights()`、`forward()` 或 llamafile GEMM。Python 端在 load/sync 完成后：

1. 用 `ctypes.from_address` + `torch.frombuffer` 在 native 地址上建立
   **non-owning view**；view 不分配、不复制权重，native buffer 的真实所有者仍是
   `MOE` 对象。代码只把它们当作只读源。
2. 设 expert 数为 `E`、intermediate 为 `I`、hidden 为 `H`、NUMA pool 数为
   `P`。每个 pool 的 gate/up view 为 `[E,I/P,H]`，down view 为
   `[E,H,I/P]`。`NumaShardedExpertTensor` 对外呈现逻辑 `[2I,H]` 和
   `[H,I]`，复制到 GPU slot 时再把 gate/up 的 row shard 和 down 的 column
   shard 写入对应 slice。GPU 端的 slot 仍是原来的连续 Torch tensor，fused GPU
   MoE 无需感知 NUMA 分片。
3. 原地 `clear()` 每个 pool entry，用上述逻辑视图替换 raw tensor。
   `LayerExpertCache` 和所有 prefetch runtime 持有的是同一个 dict，因此不会留下
   隐藏的 raw 引用。之后 CPU kernel 直接读 native buffer，GPU cache 也从同一
   native buffer 回源。

因此 `single-weight` 指的是“只有一份全量常驻 **CPU master expert
weights**”。有限大小的 GPU expert cache 必然还有它自己的 slot 副本，这是缓存本身，
不是被消除的第二份全量 CPU 权重。

这一方法有明确边界：它目前只用于本页验收的 legacy
`llamafile_bf16`/`llamafile_f16` 路径。如果是 Q4/K-block 量化权重，或 AMX/AVX
kernel 内部的 `BufferB` 打包布局，native bytes 不再等价于 `[2I,H]`/`[H,I]`
逻辑 tensor，不能直接用同一方法作为 GPU cache 源。那些后端要么需要可逆
unpack view，要么仍需要单独的 GPU-friendly 权重。

F16 还有一个精度细节：旧路径的 CPU kernel 使用 F16 native 权重，GPU cache
却从原 BF16 pool 填充。single-weight 后 F16 native 成为唯一 master，GPU slot 填充时再由
F16 转为 GPU slot 的 BF16。这使 GPU 副本源自同一份 F16 master，但也意味着
F16 single/double 不应期待 token digest 位级完全相同；正确性依靠数值对照、长度
验证和语义检查，而不是 bitwise 等价。

### 内存验收

同一 Qwen3-30B-A3B、同一 patched `cpuinfer_ext`、同一 K3/BF16 64-token 命令，
以 `/proc/<pid>/status` 每 0.5 秒采样：

| 模式 | Max RSS | 观测到的权重加载后 RSS | swap | 说明 |
|---|---:|---:|---:|---|
| `--kt-single-weight false` | 121.40 GiB | 约 118 GiB | 8 GiB 基本打满 | raw + native 两份 |
| `--kt-single-weight true` | 78.77 GiB | 55.68 GiB | 约 0.6 GiB | native 一份；Max RSS 含加载临时量 |

Max RSS 降低 **42.63 GiB（35.1%）**。Max RSS 不会正好减半，因为它还包含非
expert 参数、CUDA/graph host buffer、safetensors 映射和单层临时 packing。更直接的
所有权验证中，每个 `cpu_expert_pool` 条目都已变为 native NUMA shard view，raw
tensor 被原地移除；小模型实测 logical pool bytes 与 native weight bytes 完全相等，
接管前的 raw 地址与 CPUInfer 新分配的 native 地址不同，接管后 pool view 的
storage 地址则直接落在 native buffer 内。

### 数值正确性

- 双 NUMA BF16/F16 小模型分别逐元素还原 native weights，确认 gate/up 的行切片和
  down 的列切片按逻辑布局组合无误；相对 PyTorch reference 的最大绝对误差为
  `1.16e-9` / `1.43e-9`。
- BF16 和 F16 全模型验收 Max RSS 分别为 78.77 GiB 和 78.84 GiB。每个
  `cpu_expert_pool` entry 在接管后只包含 `NumaShardedExpertTensor`，不再包含 raw
  Torch storage。

### 为什么减少权重后性能反而提升

single-weight 没有让矩阵乘的 FLOP 变少，也没有替换 CPU kernel。它的直接性能收益是
消除了这台 125 GiB 主机上的内存压力和 paging cliff：

- double-weight 的 Max RSS 达 121.40 GiB，同时 8 GiB swap 基本打满。除权重外，
  进程和系统还需要 CUDA graph host buffer、page cache、Python/allocator metadata 和
  CPUInfer scratch space。因此这不是“内存刚好够用”，而是会让权重页、临时页和
  file-backed page 反复被回收/换入的临界状态。CPU MoE 每个 token 都扫描大量 expert
  weights，对页缺失和内存带宽停顿很敏感，所以少数换页就能把某些 step 放大成
  秒级尾延迟。
- single-weight 权重加载后曾观测到 55.68 GiB RSS，给工作集留出了数十 GiB 空间。
  CPU forward 仍读它原来的 NUMA-local buffer，但这些 hot page 现在更可能保持 resident；
  GPU cache refill 也从同一份 native 工作集读取，不再唤醒另一组 cold raw page。
- GPU cache 仍复制相同数量的权重 byte，而且分片 view 会把一次逻辑填充分成若干
  slice copy。所以“少一份权重”不等于“每 token 少一次 H2D”；本机的主收益是
  避免 swap/page reclaim，不是 zero-copy 到 GPU。

最直接的相同命令 A/B 是 64-token BF16 K3，除 `--kt-single-weight` 外参数一致：

| 模式 | TPOT | decode step p50 | decode step p95 | 解读 |
|---|---:|---:|---:|---|
| double | 273.210 ms/token | 369.124 ms | 3254.783 ms | 已进入换页尾延迟区，不代表 kernel 算力 |
| single | 115.872 ms/token | 314.040 ms | 526.155 ms | 去掉秒级 paging spike 后恢复正常范围 |

这个 57.6% 差值不应外推成“single-weight 总能使 kernel 快 57.6%”；它表明的是旧布局
在本机已经越过内存容量临界点。更长的 K3/BF16 512-token single 运行为
**91.287 ms/token**，相对本页较早的 `109.946 ms/token` 快 17.0%，但这两轮的
seed、命令别名和当时系统状态不完全相同，因而 17.0% 只是端到端观测，不是纯净
single-weight 微基准。

F16 的 B2/B3/B5 完整 63-step TPOT 为 **105.937 / 156.797 / 148.826 ms**。
相对本页旧代表轮次 `110.237 / 156.084 / 154.511 ms`，分别快 3.9%、慢 0.46%、
快 3.7%。B3 没有提升，说明在未触发严重换页时，single-weight 的收益应理解为内存
稳定性改善，而不是对每个 batch shape 的固定算术加速。本分支的 grouped path、
`group_min_len=1`、B2 全 CPU routing 和 batch exact decode 选择是另外的性能改动，不是
single-weight 指针视图本身的因果。

### 直接阅读模型输出

为了不把“长度正确”误当成“模型输出正常”，本次使用模型目录中的 Qwen3 tokenizer
对保存的 token ID 重新解码（`skip_special_tokens=False`），并人工阅读 BF16 的
512-token 完整文本以及 F16 B2/B3/B5 的全部 10 条 64-token 文本。测试 prompt 要求
解释 sparse MoE 中 speculative decoding 与 expert prefetch 的重叠。

BF16 512-token 输出开头为：

> Okay, so I need to explain how speculative decoding can overlap expert prefetch with draft and verify segment computation while maintaining exact verification semantics.

后文持续解释 draft/verifier、sparse router、预测下一 token 的 expert 并与计算重叠，
语法连贯且主题未偏离。它有两个属于生成设置/内容质量而不是权重损坏的特征：

- 输出使用“Okay, so I need to...”式的自我分析，前半段用较多 token 复述问题，
  还没来得及展开 bounded budget/eviction 的完整细节就到达 512-token 上限。
- 文本说 expert 可能从“disk or another memory”获取，对一般架构说得通，但本实现
  具体是从 host RAM 的 native NUMA buffer 填充 GPU cache。这是回答精确度问题，
  不是乱码或数值崩坏。

F16 的 10 条 batch 输出都是同一问题的合理英文回答开头。例如 B2 第 1 条为：

> This is a complex question that requires an in-depth technical explanation. Let me break it down step by step.

B5 第 1 条则直接从 sparse MoE 只激活部分 expert 开始解释。所有文本都没有乱码、
Unicode replacement character、异常控制字符或无意义循环。因为 benchmark 强制每请求
64 token 且 ignore EOS，多数样本在句子中间结束，这是 `stopped_by=max_output_tokens`
的预期截断，不是推理异常。

机械检查与人工阅读一致：BF16 为 1 条 512 token，F16 B2/B3/B5 为 2/3/5 条、
每条 64 token；`output_fixed_length_ok=true`、`output_validation_error=""`、
`max_repeated_token_run=1`。结论是当前 single-weight 实现没有表现出权重错位、dtype
误解码或内存生命期损坏常见的语义崩坏；输出正常有逻辑，但固定长度 raw
prompt benchmark 本身不是对回答完整度的高质量评测。

结果位于：

- `results/single_weight_k3_r075_t06_512/`
- `results/single_weight_f16_seed20260719_batch_2_3_5/`
- 内存 A/B：`results/single_weight_ab_{double,single}_64/`

### cpuinfer_ext 补丁

native 扩展需要暴露三个内部权重地址，但不改变其计算代码。补丁随 nano 仓库放在
`patches/ktransformers-single-weight.patch`：

```bash
cd /home/edge/zx/ktransformers
git apply /home/edge/zx/nano-vllm-moe/patches/ktransformers-single-weight.patch
# 按原 cpuinfer_ext 构建流程重编译；本机增量构建示例：
cmake --build csrc/ktransformers_ext/build --target cpuinfer_ext --parallel 16
```

`kt_single_weight` 默认为 `true`。如果扩展没有 `MOE.get_weight_ptrs()`，初始化会
fail fast 并提示应用补丁；需要临时使用旧扩展时可显式传
`--kt-single-weight false`，但会恢复双份权重。

## 对照口径

KTransformers 批测来自：
`/home/edge/zx/ktransformers/benchmark_outputs/ktransformers_qwen3_batch_tpot_2026-08-09.md`。
它使用 F16 GGUF、prompt 23、固定输出 64、temperature 0.6、top-k 20、top-p
0.95，并排除 graph capture，只统计之后 61 个 replay step。

nano 批测使用从同一模型 BF16 safetensors 转成 F16 后交给相同的 fixed legacy
`cpuinfer_ext` kernel；prompt 67、固定输出 64、temperature 0.6。nano 当前 sampler
没有 KT 脚本的 top-k/top-p 选项。主表的 nano 数字是全部 63 个 decode step 的
完整 `llm.step()` 墙钟，包括 scheduler、model、sampler 和后处理，口径不比 KT
宽松。新 benchmark 另行输出去掉前两个 step 的 61-step stable replay 指标，便于
逐项核对，但不会覆盖完整墙钟 TPOT。

权重来源、prompt 和采样策略仍不完全相同，因此这些结果证明本机实现性能已经超过
现有 KT 基线，不应解释成逐 token 的 bitwise 等价 A/B。

## 结果

### 单请求

| 实现/阶段 | CPU backend | affinity | K | 固定输出 | TPOT |
|---|---|---|---:|---:|---:|
| nano 旧最终结果 | kt-kernel AVX2 BF16 | `taskset 0-7,20-27` | 6 | 512 | 258.413 ms |
| nano fixed legacy 初始 exact 对照 | llamafile BF16 | 无外层绑核 | 0 | 64 | 148.568 ms |
| nano fixed legacy K3，temp 0.8 | llamafile BF16 | 无外层绑核 | 3 | 512 | 116.695 ms |
| **nano fixed legacy K3，temp 0.6** | llamafile BF16 | 无外层绑核 | 3 | 512 | **109.946 ms** |
| nano single-weight K3，temp 0.6 | llamafile BF16 | 无外层绑核 | 3 | 512 | 91.287 ms |
| **nano fixed-GEMM K1/vpb2/seg16/active12，temp 0.6，3-run mean** | **llamafile F16** | 无外层绑核 | **1** | **512** | **65.443 ms** |
| KTransformers BF16 stable replay | fixed legacy BF16 | CPUInfer 自绑核 | 0 | 64 | 122.35 ms |
| KTransformers 当日 F16 stable replay | fixed legacy F16 | CPUInfer 16 线程 | 0 | 64 | 125.886 ms |
| KTransformers 历史 F16 stable replay | fixed legacy F16 | CPUInfer 20 线程 | 0 | 64 | 81.90 ms |

去掉错误 affinity/backend 后重新筛选 K（固定输出 256，temperature 0.8）：

| K | 1 | 2 | 3 | 4 | 6 |
|---:|---:|---:|---:|---:|---:|
| TPOT | 112.161 | 109.080 | **105.549** | 144.968 | 153.113 |

这推翻了旧文档基于错误运行环境得到的“K6 最优”。最终单请求选择 K3；K2 在短筛
中接近，但 K3 的 512-token 验收已经完成。

### True batch

下面 nano 使用 exact F16 路线。nano 数字为完整 63-step 墙钟；KT 数字为其文档的
61-step stable replay。`aggregate tok/s = B / step_TPOT`。

| B | nano TPOT（复测） | nano 代表 aggregate tok/s | KT TPOT | KT aggregate tok/s | 结论 |
|---:|---:|---:|---:|---:|---|
| 2 | 110.237 / 117.120 / 125.859 ms | 18.14（110.237 ms 轮） | 121.84 ms | 16.41 | nano 三轮均值快 3.4%，但有跨轮波动 |
| 3 | 156.084 / 159.096 ms | 19.22（156.084 ms 轮） | 163.65 ms | 18.33 | nano 快 4.6%（较好一轮） |
| 5 | 154.511 / 154.854 ms | 32.36（154.511 ms 轮） | 251.92 ms | 19.85 | nano 快 38.7% |

所有 nano 行都验证了每个请求正好生成 64 token；B2/B3/B5 的代表输出分别包含
128/192/320 个生成 token，没有 EOS 提前停止。代表 digest：

- B2：`874f5e9ba069c3df18844524023462b74bfc088b792836f5aa367bea448f80be`
- B3：`1ddb8bb6b20850dd07e6c7cc5e1e25f360056eb764cd9b67e198532e66619ae3`
- B5：`6620fcc9f782dc372abb20c67460fe037990e31da2c862490302db7c3e9a1d7a`

## 根因和代码改动

### 1. 外层 `taskset` 与 CPUInfer 的 NUMA 绑核冲突

同一 fixed legacy BF16 单层 kernel 的受控测试：CPUInfer 自己管理两个 NUMA pool
时约 `0.61 ms/layer`；再套旧命令的 `taskset` 后约 `6.3 ms/layer`，慢约 10 倍。
端到端 exact 对照为 `148.568 ms/token`（无 taskset）对 `529.620 ms/token`
（有 taskset）。

CPUInfer 已使用 hwloc/NUMA 将 worker 绑定到指定节点。限制整个 Python 进程会同时
限制 CUDA callback 主线程和 worker 可用 CPU，并破坏其内部布局。代码现在发现受限
process affinity 时会明确告警。最终命令不使用 `taskset`。

### 2. nano 原 backend 与 KT 实际 kernel 不同

nano 原先使用 `kt-kernel 0.6.4` 的 `AVX2BF16_MOE`，KT 本机有效结果使用修过两个
数值 bug 的 legacy `MOE`。新增 `llamafile_bf16` 和 `llamafile_f16` backend，动态
加载 KT 本机重编译的扩展：

```text
/home/edge/zx/ktransformers/build/lib.linux-x86_64-cpython-312/
cpuinfer_ext.cpython-312-x86_64-linux-gnu.so
```

backend 会按 expert-major 的 gate/up/down 布局一次性打包并调用 legacy
`load_weights()`；热路径只提交 `forward()`。F16 模式把 CPU expert 权重转换为 F16，
hidden/output 保持 BF16，与 KT 的 F16 expert kernel 路径对齐。小型真实扩展数值测试：

- BF16：相对 PyTorch reference 最大绝对误差约 `9e-8`；
- F16 权重/BF16 activation：最大绝对误差 `3.08e-7`，平均 `4.05e-8`。

### 3. 单请求和 batch 不能共用同一个投机策略

单请求 K3 的 verify qlen 为 `K+1=4`，一次 CPU callback 摊销更多 token，因此优于
exact。true batch 时 verify qlen 变成 `B × (K+1)`；K3 在 B2/B3/B5 会把一次 verify
扩大到 8/12/20 token，同时 acceptance 并没有补偿额外 CPU expert 计算。实测 K1
和 K3 batch 均更慢，所以 batch 切到 exact AR。

### 4. legacy grouped path 和 B2 全 CPU 路由

KT 默认 `group_min_len=10`，qlen 2/3/5 都逐 token 执行。单层微基准把阈值改为 1：

| qlen | group=10 | group=1 | 改善 |
|---:|---:|---:|---:|
| 2 | 2.052 ms | 1.895 ms | 7.7% |
| 3 | 3.044 ms | 2.671 ms | 12.3% |
| 5 | 5.023 ms | 4.144 ms | 17.5% |

`m_block=4/8/16/32` 的 1000-iteration 筛选差异约 1%，没有可靠收益，因此保留已有
端到端验证配置。对于 exact F16 的 qlen=2，混合 `-1`（GPU cached routes）与 CPU
routes 会破坏 grouped fast path；CPUInfer 已持有全量专家，所以 B2 改为 CPU 计算完整
top-k，并跳过重复 GPU expert 计算。B3/B5 继续使用 hybrid 路径。

### 5. true-batch graph 和 benchmark 正确性

- standard/draft CUDA graph capture 过去只有 `1,2,4,...`，`max_num_seqs=5` 会静默
  eager；现在总是补上 `max_bs` 尾项。
- draft reroute 的有界 shape/slot-width specialization 会超过 Dynamo 默认 8 次重编译
  限制；现在按即将捕获的 shape 数设置 cache 上限。
- `kt_capture_bs` 自动包含所有 verify bucket，避免 CUDA graph host callback 持有已被
  替换的 pinned buffer 指针。
- 新增 `scripts/bench_eval_true_batch_tpot.py`，在同一 engine 中提交真正的 B 个并发
  request；逐行报告 request-visible TPOT、aggregate TPOT/吞吐、step p50/p95、完整
  63-step trace和61-step stable replay，并验证每行固定输出长度。
- batch 形状先预热 32 token；通用 allocation warmup 的 token budget 限制到
  `max_model_len`，避免为 decode benchmark 构造无关的两条 8192-token fallback tensor
  而 OOM。

## 推荐命令

不要在下面命令外层添加 `taskset`。

### 单请求最终 F16/K1/vpb2

```bash
cd /home/edge/zx/nano-vllm-moe

PYTHONPATH=. /home/edge/.conda/envs/nano_moe/bin/python \
  scripts/bench_eval_workload_tpot.py \
  --request-mode per_layer_slots \
  --model-path /home/edge/models/Qwen3-30B-A3B \
  --output-dir results/single_weight_f16_k1_vpb2_512_repeats3 \
  --optimized-config k1_f16_3080 \
  --output-lens 512 \
  --repeats 3 \
  --temperature 0.6 \
  --kt-llamafile-extension-path /home/edge/zx/ktransformers/build/lib.linux-x86_64-cpython-312/cpuinfer_ext.cpython-312-x86_64-linux-gnu.so \
  --kt-single-weight true \
  --collect-profile true --save-token-ids true --fail-fast true
```

### 单请求历史 K3/BF16 对照

```bash
cd /home/edge/zx/nano-vllm-moe

PYTHONPATH=. /home/edge/.conda/envs/nano_moe/bin/python \
  scripts/bench_eval_workload_tpot.py \
  --request-mode per_layer_slots \
  --model-path /home/edge/models/Qwen3-30B-A3B \
  --output-dir results/llamafile_k3_r075_bucket4_t06_512 \
  --optimized-config k3_3080 \
  --output-lens 512 \
  --temperature 0.6 \
  --kt-direct-backend llamafile_bf16 \
  --kt-llamafile-extension-path /home/edge/zx/ktransformers/build/lib.linux-x86_64-cpython-312/cpuinfer_ext.cpython-312-x86_64-linux-gnu.so \
  --kt-single-weight true \
  --collect-profile false --save-token-ids true --fail-fast true
```

### B2/B3/B5 exact F16

```bash
cd /home/edge/zx/nano-vllm-moe

PYTHONPATH=. /home/edge/.conda/envs/nano_moe/bin/python \
  scripts/bench_eval_true_batch_tpot.py \
  --request-mode per_layer_slots \
  --model-path /home/edge/models/Qwen3-30B-A3B \
  --output-dir results/llamafile_f16_final_batch_2_3_5 \
  --optimized-config k3_3080 \
  --inference-mode heter \
  --spec-enable-prefetch false \
  --draft-cuda-graph-enabled false \
  --verify-cuda-graph false \
  --output-lens 64 --batch-sizes 2,3,5 \
  --kt-capture-bs 1,2,4,8,16,32 \
  --kt-direct-backend llamafile_f16 \
  --kt-llamafile-extension-path /home/edge/zx/ktransformers/build/lib.linux-x86_64-cpython-312/cpuinfer_ext.cpython-312-x86_64-linux-gnu.so \
  --kt-single-weight true \
  --gpu-memory-utilization 0.999 --temperature 0.6 \
  --collect-profile false --save-token-ids true --fail-fast true
```

## 结果位置

- 单请求最终三次复测：`results/single_weight_f16_k1_vpb2_512_repeats3/`
- 单请求首次 512-token profile：`results/single_weight_f16_k1_vpb2_512/`
- 单请求历史 K3/BF16：`results/llamafile_k3_r075_bucket4_t06_512/`
- B2 带完整 step trace：`results/llamafile_f16_group1_cpuall_q2_trace16/`
- B2/B3/B5 合并验收：
  `results/llamafile_f16_group1_cpuall_q2_final_batch_2_3_5/`
- grouped-path B3/B5 复测：
  `results/llamafile_f16_group1_exact_r075_t06_true_batch_2_3_5/`
- `m_block` 微筛：`results/cpuinfer_mblock_{4,8,16,32}_group1.jsonl`

## 优化提交索引

每个提交均附带独立文档，记录改动边界、实验口径、前后 TPOT、验证和结果目录：

- `8943dd4`：single-weight CPU expert storage；
  `docs/optimization_commits/20260812_01_single_weight.md`。
- `8a90da6`：重复 engine CUDA 生命周期修复；
  `docs/optimization_commits/20260812_04_repeat_engine_cleanup.md`。
- `4123895`：F16/K1/vpb2 正收益 preset 与三次均值；
  `docs/optimization_commits/20260812_02_k1_f16_vpb2.md`。
- `c07ed02`：top-k/top-p 精确 speculative sampling 对齐；
  `docs/optimization_commits/20260812_03_sampling_alignment.md`。该提交是比较正确性功能，
  其过滤配置实测为负收益，未纳入最终性能 preset。
- 本提交：segment 16 boundary schedule；
  `docs/optimization_commits/20260812_05_segment16.md`。
- 本提交：回收未使用 staging slots 为 active cache；
  `docs/optimization_commits/20260813_06_reclaim_staging_cache.md`。
- 本提交：decode-aware Qwen3 fixed grouped GEMM；
  `docs/optimization_commits/20260813_07_fixed_decode_grouped_gemm.md`。
- 本提交：workload-sized CUDA memory warmup；
  `docs/optimization_commits/20260813_08_workload_sized_warmup.md`。
