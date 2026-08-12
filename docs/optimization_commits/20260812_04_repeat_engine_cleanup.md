# Benchmark repeated-engine CUDA cleanup

## 问题

`--repeats > 1` 每个 repeat 新建 LLMEngine。旧 `atexit` bound callback 和
`SpeculativeEngine.model_runner` 仍持有 rank-0 ModelRunner，导致模型权重与 CUDA graphs
在 `llm.exit()` 后没有释放。首个 512-token repeat 完成后，第二次构造会在仅申请
20 MiB 时 CUDA OOM。

## 修复

LLMEngine.exit 现在注销 atexit callback，清除 model runner/spec engine/scheduler/process
引用，运行 GC，并调用 `torch.cuda.empty_cache()`；退出保持幂等。

## 验证

- 新增生命周期单元测试；连同相关 spec/sampling 测试共 36 项通过。
- 修复前：repeat 0 得到 `69.682 ms/token`，repeat 1 构造时 OOM。
- 修复后：同一 Python 进程继续完成 repeat 1 `73.837` 和 repeat 2 `70.689 ms/token`，
  最终生成 `results/single_weight_f16_k1_vpb2_512_repeats3/summary.json`。

该修复不宣称改变单请求 TPOT；它消除了重复实验的资源泄漏和选择性报告风险，使均值可复现。
