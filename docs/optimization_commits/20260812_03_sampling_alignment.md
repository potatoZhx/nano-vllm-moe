# top-k/top-p benchmark and speculative-sampling alignment

## 改动

为与 ktransformers 的 `top_k=20, top_p=0.95` 口径对齐，SamplingParams、普通 sampler、
speculative acceptance 和 benchmark CLI 现在完整传递 top-k/top-p。standard speculative
sampling 对 target distribution `p` 与 draft distribution `q` 应用相同过滤，并继续使用
精确的 `min(1, p/q)` 接受率和 residual distribution。

无过滤的默认配置仍走原有 compiled fast path，不承担排序/散射开销。

## 验证和结论

- 相关采样、acceptance、spec flow、benchmark 配置测试共 72 项通过。
- F16/K1/vpb2、256 输出、`top_k=20, top_p=0.95`：`81.960 ms/token`，结果目录
  `results/single_weight_f16_k1_vpb2_topk20p095_256/`。
- 同阶段无过滤结果为 `72.317 ms/token`；过滤使 acceptance 从约 `170 ms` 增至
  `361 ms`（全请求累计），因此该对齐配置不是性能 preset，默认继续使用
  `top_k=0, top_p=1.0`。

这是一项比较正确性/功能提交，不把负收益结果包装成优化。它使后续 nano-vllm-moe 与
ktransformers 的 sampling 参数可以显式对齐。
