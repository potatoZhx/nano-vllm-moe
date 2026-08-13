# Workload-sized CUDA memory warmup

## 改动

原 `ModelRunner.warmup_model()` 总是按 `max_model_len` 构造合成 prefill；TPOT
benchmark 的 `max_model_len=8192`、batch=1 因而在启动时运行一条 8192-token
prefill。实际测试输入只有 67 token，这个临时峰值既不代表 steady decode，也会挤掉
KV cache 和 GPU expert cache。

新增 `Config.warmup_model_tokens` / `--warmup-model-tokens`：

- `0` 完整保留旧行为；
- 正值限制用于 CUDA 峰值测量的**总 synthetic prefill token budget**；
- `k1_f16_3080` 使用 1024，仍是实测 67-token prefill 的 15.3 倍；
- `max_model_len` 仍为 8192，没有通过缩短 decode context 获得结果。

该值必须不小于部署中最大的未分块 prefill；长 prompt 服务不应盲目复用 1024 preset。

## 资源与性能证据

同为 active12/staging0、ctx8192、fixed grouped GEMM：

| warmup tokens | KV blocks / capacity | 256-token TPOT | 固定长度校验 |
|---:|---:|---:|:---:|
| legacy 8192 | 8 / 2048 tokens | 67.881 ms | pass |
| **1024** | **49 / 12544 tokens** | **66.396 ms** | **pass** |

单次筛选 TPOT 降低 2.19%，但这不是多 seed 证据，因此不把它计入最终 512-token
均值；确定性收益是 synthetic warmup 峰值降低、KV capacity 增加，以及为后续增加
active expert cache 留出空间。输出固定为 256 token、无 validation error。

结果：

- `results/active12_ctx8192_fixedgemm_256/`
- `results/active12_ctx8192_warmup1024_fixedgemm_256/`

## 验证

```bash
PYTHONPATH=. /home/edge/.conda/envs/nano_moe/bin/python -m pytest -q \
  tests/test_model_warmup_token_budget.py \
  tests/test_eval_tpot_config.py \
  tests/test_grouped_gemm_fixed_config.py
```

结果：`16 passed`。测试覆盖显式总 token budget、`0` 的 legacy shape、preset 值和
CLI 显式覆盖；另执行 `python -m compileall -q nanovllm`。
