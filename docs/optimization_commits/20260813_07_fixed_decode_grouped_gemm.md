# Fixed decode-aware Qwen3 grouped GEMM dispatch

## 根因

GPU cached experts 的 gate-up/down 都使用 Triton grouped GEMM。原 autotune key 默认
只有 `(N,K,NUM_EXPERTS)`，不含总 route 数 `M`；初始化的大 synthetic prefill 因此
选出一个大 M config，decode 的 `M<=16` 后续直接复用它。独立 autotune 证明工作点
不同：

| projection | M | best config `(BM,BN,BK,warps,stages)` |
|:---|---:|:---|
| gate-up `(N,K)=(1536,2048)` | 1024 | `(64,64,64,4,5)` |
| gate-up | 16 | `(16,64,64,4,4)` |
| down `(2048,768)` | 1024 | `(32,128,64,8,4)` |
| down | 16 | `(16,128,64,4,3)` |

新路径通过 Autotuner 的底层 JIT kernel 对 Qwen3 两种固定 projection shape 做大/小 M
dispatch。它不复制 kernel，也不改 accumulation dtype/order。非 Qwen shape 或未启用环境
变量时仍使用通用 autotuner。

## 正确性与资源

- 底层 fixed JIT 与逐 expert BF16 matmul 在实际 gate-up shape 上 bitwise 相同：
  `max_abs=0, mean_abs=0`；端到端测试也覆盖固定 dispatch。
- 避免 Triton `do_bench()` 的固定 256 MiB L2 cache-flush tensor；这消除了 active14
  初始化时“仅余 233 MiB、申请 256 MiB”的 OOM。
- active12/ctx8192 的 256-token 启动墙钟约从 196 s 降到 118 s；TPOT 从 69.462 降到
  67.881 ms/token。

## 三次 512-token 配对复测（active12/ctx8192）

| repeat | autotune | fixed dispatch | delta (ms) |
|---:|---:|---:|---:|
| 0 | 69.914 | 63.432 | −6.482 |
| 1 | 64.984 | 66.085 | +1.101 |
| 2 | 64.623 | 66.811 | +2.188 |
| mean | **66.507** | **65.443** | **−1.064** |

均值降低 **1.60%**，population std 从 2.414 降到 1.452 ms。逐 seed 并非一致正收益，
因此这只是小幅均值优化；其确定性收益是正确的 decode tiling、低启动开销和消除
autotune 峰值。三轮均固定生成 512 token 并通过长度验证。

结果：`results/active12_ctx8192_fixedgemm_{256,512_repeats3}/` 与
`results/active12_ctx8192_fixedgemm_512_repeats3_r2/`。
