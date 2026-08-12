# Single-weight CPU expert storage

## 改动

legacy llamafile/CPUInfer 在 `load_weights()` 后已经持有按 NUMA 分片的完整专家权重。此前
nano-vllm-moe 仍保留一份 expert-major PyTorch CPU tensor，形成两份原始 CPU 权重。
本改动给 ktransformers 的 `cpuinfer_ext` 增加只读权重指针接口，并在 nano-vllm-moe 中：

- 用 `NumaShardedExpertTensor` 对 native NUMA buffer 建立非 owning view；
- CPUInfer 完成加载后原地替换 expert pool，释放旧 raw tensor；
- GPU cache fill/prefetch 直接从 NUMA shards 拷贝并恢复逻辑 gate/up/down 顺序；
- 保留 `--kt-single-weight false` 兼容路径。

这项优化的主要收益是消除第二份 CPU expert raw weights，使 125 GiB 双路机器能够稳定运行
F16/BF16 CPUInfer + GPU expert cache，而不是依赖 swap 或偶然的 allocator 状态。

## 验证

- 单元测试：`test_cpu_expert_weights.py` 和 `test_kt_direct_backend.py`，15 项通过；测试覆盖
  shard 顺序、native pointer alias、旧 tensor 释放以及 GPU cache copy。
- BF16/K3、单请求、512 输出：`91.287 ms/token`。
- 结果：`results/single_weight_k3_r075_t06_512/`（主报告记录同口径结果）。
- native 扩展补丁：`patches/ktransformers-single-weight.patch`。

TPOT 定义始终为 `decode_wall_sec / (generated_output_tokens - 1)`；模型为
Qwen3-30B-A3B，硬件为 2 x RTX 3080 10 GiB、2 x Xeon Gold 5218R。
