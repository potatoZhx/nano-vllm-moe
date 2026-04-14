## 4. 全量结果表格（脚本自动生成）

### 4.1 Alignment（正确性与 profile 观测）

| Case | cpu_exec_routes | model_cpu_route_ratio | wait_ms(model_cpu_wait+model_gpu_wait) | 与 standard token mismatch |
|---|---|---|---|---|
| standard | 0 | 0.0000 | 0.000 | 0 |
| heter_serial | 346062 | 0.0000 | 0.000 | 0 |
| heter_parallel | 8192 | 0.0000 | 35263.029 | 1 |

`heter_parallel` 的首个差异位点：`seq=1, token_pos=3`，`17 -> 16`。

### 4.2 单层 MoE（token=64，按 cpu ratio）

文件：`moe_single_layer_cpu_gpu_parallel_bench_phase2_post_rerun_job15779_idlegpu3.json`

| cpu_ratio(%) | serial_ms | parallel_ms | speedup(serial/parallel) | parallel_wait_ms | parallel_cpu_route_ratio |
|---|---|---|---|---|---|
| 0 | 0.807 | 0.718 | 1.1244 | 0.000 | 0.0000 |
| 25 | 75.275 | 63.619 | 1.1832 | 0.000 | 0.2500 |
| 50 | 115.495 | 4.676 | 24.6990 | 0.000 | 0.5000 |
| 75 | 6.450 | 7.512 | 0.8587 | 4.732 | 0.7500 |
| 100 | 8.317 | 8.041 | 1.0344 | 0.000 | 1.0000 |

### 4.3 单层 MoE（token=256，按 cpu ratio）

同文件：`moe_single_layer_cpu_gpu_parallel_bench_phase2_post_rerun_job15779_idlegpu3.json`

| cpu_ratio(%) | serial_ms | parallel_ms | speedup(serial/parallel) | parallel_wait_ms | parallel_cpu_route_ratio |
|---|---|---|---|---|---|
| 0 | 0.774 | 0.752 | 1.0292 | 0.000 | 0.0000 |
| 25 | 5.792 | 5.664 | 1.0226 | 0.000 | 0.2500 |
| 50 | 12.782 | 12.432 | 1.0281 | 0.000 | 0.5000 |
| 75 | 16.202 | 16.630 | 0.9742 | 13.701 | 0.7500 |
| 100 | 20.270 | 20.313 | 0.9978 | 0.000 | 1.0000 |

### 4.4 Small tokens（1/3/5/10/20）speedup 矩阵

文件：`moe_single_layer_cpu_gpu_parallel_bench_phase2_breakdown_small_tokens_1_3_5_10_20_rerun_job15779_idlegpu3.json`

speedup 定义为 `serial_latency / parallel_latency`，`>1` 表示并行更快。

| token_size | cpu0% | cpu25% | cpu50% | cpu75% | cpu100% |
|---|---|---|---|---|---|
| 1 | 1.1113 | 1.0658 | 1.0504 | 1.0393 | 47.5381 |
| 3 | 1.0446 | 1.0198 | 0.9929 | 0.9928 | 1.0021 |
| 5 | 1.0171 | 1.0058 | 0.9982 | 1.0081 | 1.0353 |
| 10 | 1.0270 | 1.0306 | 0.9945 | 0.8847 | 0.9735 |
| 20 | 1.0296 | 1.0105 | 0.9682 | 0.9099 | 1.0134 |

Small tokens 全量明细（逐 token_size x cpu_ratio）：

| token_size | cpu_ratio(%) | serial_ms | parallel_ms | speedup(serial/parallel) | parallel_wait_ms | serial_cpu_route_ratio | parallel_cpu_route_ratio |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 0.805 | 0.725 | 1.1113 | 0.000 | 0.0000 | 0.0000 |
| 1 | 25 | 0.775 | 0.727 | 1.0658 | 0.000 | 0.0000 | 0.0000 |
| 1 | 50 | 0.779 | 0.742 | 1.0504 | 0.000 | 0.0000 | 0.0000 |
| 1 | 75 | 0.752 | 0.724 | 1.0393 | 0.000 | 0.0000 | 0.0000 |
| 1 | 100 | 59.615 | 1.254 | 47.5381 | 0.000 | 1.0000 | 1.0000 |
| 3 | 0 | 0.786 | 0.753 | 1.0446 | 0.000 | 0.0000 | 0.0000 |
| 3 | 25 | 0.746 | 0.732 | 1.0198 | 0.000 | 0.0000 | 0.0000 |
| 3 | 50 | 1.799 | 1.812 | 0.9929 | 0.000 | 0.3333 | 0.3333 |
| 3 | 75 | 2.208 | 2.223 | 0.9928 | 0.000 | 0.6667 | 0.6667 |
| 3 | 100 | 2.064 | 2.060 | 1.0021 | 0.000 | 1.0000 | 1.0000 |
| 5 | 0 | 0.757 | 0.744 | 1.0171 | 0.000 | 0.0000 | 0.0000 |
| 5 | 25 | 1.790 | 1.780 | 1.0058 | 0.000 | 0.2000 | 0.2000 |
| 5 | 50 | 2.238 | 2.242 | 0.9982 | 0.000 | 0.4000 | 0.4000 |
| 5 | 75 | 2.688 | 2.666 | 1.0081 | 0.000 | 0.6000 | 0.6000 |
| 5 | 100 | 2.756 | 2.662 | 1.0353 | 0.000 | 1.0000 | 1.0000 |
| 10 | 0 | 0.794 | 0.773 | 1.0270 | 0.000 | 0.0000 | 0.0000 |
| 10 | 25 | 1.929 | 1.872 | 1.0306 | 0.000 | 0.2000 | 0.2000 |
| 10 | 50 | 2.395 | 2.408 | 0.9945 | 0.000 | 0.4000 | 0.4000 |
| 10 | 75 | 2.999 | 3.390 | 0.8847 | 1.064 | 0.7000 | 0.7000 |
| 10 | 100 | 3.056 | 3.139 | 0.9735 | 0.000 | 1.0000 | 1.0000 |
| 20 | 0 | 0.778 | 0.756 | 1.0296 | 0.000 | 0.0000 | 0.0000 |
| 20 | 25 | 2.123 | 2.101 | 1.0105 | 0.000 | 0.2500 | 0.2500 |
| 20 | 50 | 2.902 | 2.998 | 0.9682 | 0.000 | 0.5000 | 0.5000 |
| 20 | 75 | 3.752 | 4.123 | 0.9099 | 1.773 | 0.7500 | 0.7500 |
| 20 | 100 | 3.940 | 3.888 | 1.0134 | 0.000 | 1.0000 | 1.0000 |

### 4.5 Spec verify（多 cpu ratio 对比，min 配置）

文件：`spec_verify_cpu_ratio_bench_phase2_post_min_job15779_idlegpu2.json`

| cpu_ratio(%) | serial_ms | parallel_ms | speedup(serial/parallel) | parallel_wait_ms | parallel_cpu_route_ratio |
|---|---|---|---|---|---|
| 25 | 4724.404 | 5182.593 | 0.9116 | 0.000 | 0.1581 |
| 50 | 8272.390 | 8251.743 | 1.0025 | 3.938 | 0.2521 |
| 75 | 12671.358 | 15319.176 | 0.8272 | 8314.784 | 0.3253 |

### 4.6 Spec verify（idlegpu3 收尾四文件）

| file | cpu_ratio(%) | serial_ms | parallel_ms | speedup(serial/parallel) | parallel_wait_ms |
|---|---|---|---|---|---|
| phase2_post_min | 75 | 11482.260 | 17540.444 | 0.6546 | 8377.441 |
| phase2_post_min_rerun | 75 | 10854.890 | 16986.853 | 0.6390 | 8378.790 |
| phase2_post_min_threshold0 | 75 | 14034.496 | 15902.121 | 0.8826 | 9648.829 |
| phase2_post_min_threshold0_rerun | 75 | 16340.677 | 11795.609 | 1.3853 | 6977.884 |

### 4.7 真实模型 cpugpuparallel（历史基线）

文件：`moe_real_model_cpu_gpu_parallel_bench_phase2_post_job15779_idlegpu3.json`

| cpu_ratio(%) | serial_ms | parallel_ms | speedup(serial/parallel) | parallel_wait_ms | parallel_cpu_route_ratio | parallel_cpu_path_exec_ms |
|---|---|---|---|---|---|---|
| 75 | 7842.393 | 9787.886 | 0.8012 | 6422.365 | 0.0000 | 8400.843 |

### 4.8 真实模型小请求补跑（job15932，带时间戳）

| file | num_seqs | input_len | output_len | serial_ms | parallel_ms | speedup(serial/parallel) | parallel_wait_ms |
|---|---|---|---|---|---|---|---|
| moe_real_model_cpu_gpu_parallel_bench_phase2_post_smallreq_1x8to1_job15932_20260414_142506.json | 1 | 8 | 1 | 8700.487 | 8162.611 | 1.0659 | 6527.363 |
| moe_real_model_cpu_gpu_parallel_bench_phase2_post_smallreq_5x5to1_job15932_20260414_150734.json | 5 | 5 | 1 | 33365.838 | 31488.684 | 1.0596 | 26992.481 |

涉及文件：
- `moe_real_model_cpu_gpu_parallel_bench_phase2_post_smallreq_1x8to1_job15932_20260414_142506.json`
- `moe_real_model_cpu_gpu_parallel_bench_phase2_post_smallreq_5x5to1_job15932_20260414_150734.json`

### 4.9 异常点跨文件稳定性复核（自动统计）

说明：统计同类 rerun 文件中的 `latency_ms_mean`，用于判断异常点是否稳定复现。

| point | samples | min_ms | median_ms | max_ms | max_from_file |
|---|---|---|---|---|---|
| token64_cpu25_serial | 4 | 35.694 | 65.543 | 75.275 | moe_single_layer_cpu_gpu_parallel_bench_phase2_post_rerun_job15779_idlegpu3.json |
| token64_cpu25_parallel | 4 | 3.035 | 42.536 | 63.619 | moe_single_layer_cpu_gpu_parallel_bench_phase2_post_rerun_job15779_idlegpu3.json |
| token64_cpu50_serial | 4 | 4.514 | 10.154 | 115.495 | moe_single_layer_cpu_gpu_parallel_bench_phase2_post_rerun_job15779_idlegpu3.json |
| token64_cpu50_parallel | 4 | 4.388 | 4.650 | 4.677 | moe_single_layer_cpu_gpu_parallel_bench_phase2_post_rerun_job15714_idlegpu4.json |
| token1_cpu100_serial | 4 | 1.259 | 34.373 | 61.605 | moe_single_layer_cpu_gpu_parallel_bench_phase2_breakdown_small_tokens_1_3_5_10_20_rerun_job15779_idlegpu2.json |
| token1_cpu100_parallel | 4 | 1.237 | 5.223 | 59.304 | moe_single_layer_cpu_gpu_parallel_bench_phase2_breakdown_small_tokens_1_3_5_10_20_rerun_job15779_idlegpu2.json |

### 4.10 JSON 源文件覆盖统计（完整性校验）

说明：列出 `benchmarks/results` 下参与整理的 JSON 文件及其 `results/curves` 行数。

| file | results_rows | curves_rows |
|---|---|---|
| _smoke_moe.json | 2 | 1 |
| _smoke_real_model_parallel_job15714_idlegpu4.json | 1 | 0 |
| cpu_alignment_heter_parallel_phase2_post_rerun_job15304_idlegpu0.json | 0 | 0 |
| cpu_alignment_heter_parallel_phase2_post_rerun_job15714_idlegpu4.json | 0 | 0 |
| cpu_alignment_heter_parallel_phase2_post_rerun_job15779_idlegpu2.json | 0 | 0 |
| cpu_alignment_heter_parallel_phase2_post_rerun_job15779_idlegpu3.json | 0 | 0 |
| cpu_alignment_heter_serial_phase2_post_rerun_job15304_idlegpu0.json | 0 | 0 |
| cpu_alignment_heter_serial_phase2_post_rerun_job15714_idlegpu4.json | 0 | 0 |
| cpu_alignment_heter_serial_phase2_post_rerun_job15779_idlegpu2.json | 0 | 0 |
| cpu_alignment_heter_serial_phase2_post_rerun_job15779_idlegpu3.json | 0 | 0 |
| cpu_alignment_standard_phase2_post_rerun_job15304_idlegpu0.json | 0 | 0 |
| cpu_alignment_standard_phase2_post_rerun_job15714_idlegpu4.json | 0 | 0 |
| cpu_alignment_standard_phase2_post_rerun_job15779_idlegpu2.json | 0 | 0 |
| cpu_alignment_standard_phase2_post_rerun_job15779_idlegpu3.json | 0 | 0 |
| moe_real_model_cpu_gpu_parallel_bench_phase2_post_job15779_idlegpu3.json | 2 | 1 |
| moe_real_model_cpu_gpu_parallel_bench_phase2_post_smallreq_1x8to1_job15932_20260414_142506.json | 2 | 1 |
| moe_real_model_cpu_gpu_parallel_bench_phase2_post_smallreq_5x5to1_job15932_20260414_150734.json | 2 | 1 |
| moe_single_layer_cpu_gpu_parallel_bench_phase2_breakdown_rerun_idle_gpu0.json | 8 | 4 |
| moe_single_layer_cpu_gpu_parallel_bench_phase2_breakdown_rerun_job15304_idlegpu0.json | 8 | 4 |
| moe_single_layer_cpu_gpu_parallel_bench_phase2_breakdown_rerun_job15714_idlegpu4.json | 20 | 10 |
| moe_single_layer_cpu_gpu_parallel_bench_phase2_breakdown_rerun_job15779_idlegpu2.json | 20 | 10 |
| moe_single_layer_cpu_gpu_parallel_bench_phase2_breakdown_rerun_job15779_idlegpu3.json | 20 | 10 |
| moe_single_layer_cpu_gpu_parallel_bench_phase2_breakdown_small_tokens_1_3_5_10_20_rerun_job15304_idlegpu0.json | 50 | 25 |
| moe_single_layer_cpu_gpu_parallel_bench_phase2_breakdown_small_tokens_1_3_5_10_20_rerun_job15714_idlegpu4.json | 50 | 25 |
| moe_single_layer_cpu_gpu_parallel_bench_phase2_breakdown_small_tokens_1_3_5_10_20_rerun_job15779_idlegpu2.json | 50 | 25 |
| moe_single_layer_cpu_gpu_parallel_bench_phase2_breakdown_small_tokens_1_3_5_10_20_rerun_job15779_idlegpu3.json | 50 | 25 |
| moe_single_layer_cpu_gpu_parallel_bench_phase2_breakdown_small_tokens_rerun_idle_gpu0.json | 20 | 10 |
| moe_single_layer_cpu_gpu_parallel_bench_phase2_post_rerun_idle_gpu0.json | 20 | 10 |
| moe_single_layer_cpu_gpu_parallel_bench_phase2_post_rerun_job15304_idlegpu0.json | 20 | 10 |
| moe_single_layer_cpu_gpu_parallel_bench_phase2_post_rerun_job15714_idlegpu4.json | 20 | 10 |
| moe_single_layer_cpu_gpu_parallel_bench_phase2_post_rerun_job15779_idlegpu2.json | 20 | 10 |
| moe_single_layer_cpu_gpu_parallel_bench_phase2_post_rerun_job15779_idlegpu3.json | 20 | 10 |
| moe_single_layer_cpu_gpu_parallel_bench_phase2_post_small_tokens_1_3_5_10_20_curated_rerun_idle_gpu0.json | 50 | 0 |
| moe_single_layer_cpu_gpu_parallel_bench_phase2_post_small_tokens_1_3_5_10_20_curated_rerun_job15304_idlegpu0.json | 50 | 25 |
| moe_single_layer_cpu_gpu_parallel_bench_phase2_post_small_tokens_rerun_idle_gpu0.json | 40 | 20 |
| moe_single_layer_cpu_gpu_parallel_bench_phase2_post_small_tokens_rerun_job15304_idlegpu0.json | 40 | 20 |
| moe_single_layer_cpu_gpu_parallel_bench_phase2_post_token1_only_rerun_idle_gpu0.json | 10 | 5 |
| moe_single_layer_cpu_gpu_parallel_bench_phase2_post_token1_only_rerun_job15304_idlegpu0.json | 10 | 5 |
| spec_verify_cpu_ratio_bench_phase2_post_min_job15714_idlegpu4.json | 2 | 1 |
| spec_verify_cpu_ratio_bench_phase2_post_min_job15779_idlegpu2.json | 6 | 3 |
| spec_verify_cpu_ratio_bench_phase2_post_min_job15779_idlegpu3.json | 2 | 1 |
| spec_verify_cpu_ratio_bench_phase2_post_min_rerun_job15714_idlegpu4.json | 2 | 1 |
| spec_verify_cpu_ratio_bench_phase2_post_min_rerun_job15779_idlegpu3.json | 2 | 1 |
| spec_verify_cpu_ratio_bench_phase2_post_min_threshold0_job15714_idlegpu4.json | 2 | 1 |
| spec_verify_cpu_ratio_bench_phase2_post_min_threshold0_job15779_idlegpu3.json | 2 | 1 |
| spec_verify_cpu_ratio_bench_phase2_post_min_threshold0_rerun_job15714_idlegpu4.json | 2 | 1 |
| spec_verify_cpu_ratio_bench_phase2_post_min_threshold0_rerun_job15779_idlegpu3.json | 2 | 1 |
