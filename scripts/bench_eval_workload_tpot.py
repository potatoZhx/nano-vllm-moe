#!/usr/bin/env python3
"""Benchmark batch-size-1 TPOT on the evaluation-plan workloads.

This script uses the same nano-vllm-moe runtime configuration as
``scripts/bench_per_layer_slots.py`` but runs real workload prompts from:

* ShareGPT
* MT-Bench
* HumanEval
* MMLU-Pro
* the exact single prompt used by ``scripts/bench_per_layer_slots.py``

It intentionally does not install layer probes and does not enable
``engine_profile``/``spec_profile``. By default requests run with
``ignore_eos=False`` and stop on EOS. For legacy compatibility,
``--output-lens N`` with ``N > 0`` generates exactly like the older benchmark
path: it sets the maximum output tokens and ignores EOS. TPOT is measured with
wall-clock decode time by excluding the initial prefill step and dividing by
``generated_output_tokens - 1`` inter-token intervals. The default
``--decode-driver step`` drives ``LLM.step()`` directly; ``--decode-driver
generate`` uses the same ``LLM.generate`` driver as ``bench_per_layer_slots.py``
and times each internal step with a hook.

Example:

Validated fixed-K decode configuration for the 0.3125-cache, 512-token target:

    CUDA_VISIBLE_DEVICES=2 python scripts/bench_eval_workload_tpot.py \
        --request-mode per_layer_slots \
        --optimized-config k6_decode \
        --output-lens 512 \
        --output-dir results/eval_workload_tpot_k6_vpb4 \
        --save-token-ids true \
        --save-text true \
        --skip-existing false

    NANOVLLM_VERIFY_PREFETCH_RANK_MULTIPLIER=1 \
    NANOVLLM_VERIFY_DEFER_SEGMENT_METADATA=1 \
    NANOVLLM_VERIFY_BOUNDARY_PREFETCH_ASYNC=0 \
    CUDA_VISIBLE_DEVICES=2 python scripts/bench_eval_workload_tpot.py \
        --dataset mt_bench \
        --num-samples 0 \
        --output-dir results/eval_workload_tpot \
        --gpu-memory-utilization 0.99 \
        --cache-ratios 0.3125 \
        --max-output-tokens 0 \
        --max-draft-tokens-values 12 \
        --segment-sizes 12 \
        --allocation-modes profile_weighted \
        --slot-buckets 4 \
        --slot-max-bucket-ratio 2.0 \
        --slot-profile-csv pre_exps/exp_and_figs/unique/unique_count_plot_summary_n1024.csv \
        --kt-num-threads 16 \
        --verify-cuda-graph-bucket-steps 3,5,7,10,13 \
        --verify-prefetch-max-per-boundary 10 \
        --draft-stop-policy none \
        --verify-prefetch-rank-multiplier 1


        CUDA_VISIBLE_DEVICES=3 python scripts/bench_eval_workload_tpot.py \
    --request-mode per_layer_slots \
    --output-dir results/eval_workload_tpot_slots_prompt \
    --gpu-memory-utilization 0.99 \
    --cache-ratios 0.3125 \
    --output-lens 512 \
    --max-draft-tokens-values 12 \
    --segment-sizes 12 \
    --allocation-modes profile_weighted \
    --slot-buckets 4 \
    --slot-max-bucket-ratio 2.0 \
    --slot-profile-csv pre_exps/exp_and_figs/unique/unique_count_plot_summary_n1024.csv \
    --kt-num-threads 16

Legacy K=12 per-layer-slots prompt benchmark:

    NANOVLLM_VERIFY_PREFETCH_RANK_MULTIPLIER=1 \
    NANOVLLM_VERIFY_DEFER_SEGMENT_METADATA=1 \
    NANOVLLM_VERIFY_BOUNDARY_PREFETCH_ASYNC=0 \
    CUDA_VISIBLE_DEVICES=2 python scripts/bench_eval_workload_tpot.py \
        --request-mode per_layer_slots \
        --output-dir results/eval_workload_tpot_k12_optimized \
        --gpu-memory-utilization 0.99 \
        --cache-ratios 0.3125 \
        --output-lens 512 \
        --max-draft-tokens-values 12 \
        --segment-sizes 12 \
        --allocation-modes profile_weighted \
        --slot-buckets 4 \
        --slot-max-bucket-ratio 2.0 \
        --slot-profile-csv pre_exps/exp_and_figs/unique/unique_count_plot_summary_n1024.csv \
        --kt-num-threads 16 \
        --verify-cuda-graph-bucket-steps 3,5,7,10,13 \
        --verify-prefetch-max-per-boundary 10 \
        --draft-stop-policy none \
        --verify-prefetch-rank-multiplier 1 \
        --decode-driver generate \
        --collect-profile true \
        --save-token-ids true
"""
from __future__ import annotations

import sys

from nanovllm.benchmarks.eval_tpot.runtime import (
    create_llm,
    parse_csv as _parse_csv,
    reset_runtime_seed,
    resolved_runtime_config,
    runtime_seed,
    validate_kv_cache_capacity,
    warmup_llm,
)
from nanovllm.benchmarks.eval_tpot.config import (
    DATASET_CHOICES,
    DEFAULT_PREDICTOR_PATH,
    DEFAULT_PROFILE,
    DEFAULT_WARMUP_PROMPT,
    MODEL_PATH,
    OPTIMIZED_CONFIG_CHOICES,
    OPTIMIZED_CONFIG_PRESETS,
    REQUEST_MODE_CHOICES,
    TRANSFER_AWARE_V3_ARTIFACT,
    _num_samples_label,
    _parse_allocation_modes,
    apply_optimized_config,
    build_parser,
    configure_optimized_env,
    parse_args as parse_benchmark_args,
    parse_num_samples,
    resolve_acceptance_predictor,
    str2bool,
    validate_runtime_config,
)
from nanovllm.benchmarks.eval_tpot.cases import build_cases, case_name
from nanovllm.benchmarks.eval_tpot.data import (
    PromptSample,
    _effective_request_mode,
    _effective_sample_offset,
    _requested_datasets,
    load_dataset_samples,
    prepare_prompt_tokens,
    select_samples,
)
from nanovllm.benchmarks.eval_tpot.metrics import (
    TPOT_DEFINITION,
    collect_profile_metrics,
    finalize_prompt_result,
    grouped_summaries,
    reset_llm_profile,
    run_prompt,
    run_prompt_generate,
    steady_draft_call_stats,
)
from nanovllm.benchmarks.eval_tpot.reporting import (
    flatten_rows,
    write_csv,
    write_markdown_report,
    write_summary_csv,
)
from nanovllm.benchmarks.eval_tpot.runner import run, run_case

def main() -> None:
    run(parse_benchmark_args(sys.argv[1:]))


if __name__ == "__main__":
    main()
