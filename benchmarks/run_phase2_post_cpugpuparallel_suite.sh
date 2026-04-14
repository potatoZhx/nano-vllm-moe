#!/usr/bin/env bash

set -e
set -o pipefail

MODEL_PATH="/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B"
RESULT_DIR="benchmarks/results"
JOB_TAG="${JOB_TAG:-job15779_idlegpu0}"
GPU_ID="${GPU_ID:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_ID}"

mkdir -p "$RESULT_DIR"

echo "=== START $(date -Is) ==="
echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader

echo "--- pytest ---"
pytest -q \
  tests/test_cpu_gpu_parallel_moe.py \
  tests/test_cpu_gpu_expert_operator_alignment.py \
  tests/test_spec_engine_flow.py \
  tests/test_model_runner_spec_modes.py

echo "--- cpu alignment standard ---"
python examples/benchmarks/cpu_alignment_case.py \
  --model-path "$MODEL_PATH" \
  --mode standard \
  --slots-per-layer 32 \
  --cpu-expert-execution-enabled false \
  --cpu-gpu-parallel-execution-enabled false \
  --num-seqs 4 \
  --prompt-len 32 \
  --prompt-kind text \
  --max-tokens 8 \
  --seed 0 \
  --max-model-len 256 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.9 \
  --dist-port 29210 \
  --output "$RESULT_DIR/cpu_alignment_standard_phase2_post_rerun_${JOB_TAG}.json"

echo "--- cpu alignment heter serial ---"
python examples/benchmarks/cpu_alignment_case.py \
  --model-path "$MODEL_PATH" \
  --mode heter \
  --slots-per-layer 32 \
  --cpu-expert-execution-enabled true \
  --cpu-expert-num-threads 1 \
  --cpu-gpu-parallel-execution-enabled false \
  --cpu-gpu-parallel-min-cpu-route-ratio 0.0 \
  --remap-cache-high-ids true \
  --num-seqs 4 \
  --prompt-len 32 \
  --prompt-kind text \
  --max-tokens 8 \
  --seed 0 \
  --max-model-len 256 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.9 \
  --dist-port 29211 \
  --output "$RESULT_DIR/cpu_alignment_heter_serial_phase2_post_rerun_${JOB_TAG}.json"

echo "--- cpu alignment heter parallel ---"
python examples/benchmarks/cpu_alignment_case.py \
  --model-path "$MODEL_PATH" \
  --mode heter \
  --slots-per-layer 32 \
  --cpu-expert-execution-enabled true \
  --cpu-expert-num-threads 1 \
  --cpu-gpu-parallel-execution-enabled true \
  --cpu-gpu-parallel-min-cpu-route-ratio 0.0 \
  --remap-cache-high-ids true \
  --num-seqs 4 \
  --prompt-len 32 \
  --prompt-kind text \
  --max-tokens 8 \
  --seed 0 \
  --max-model-len 256 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.9 \
  --dist-port 29212 \
  --output "$RESULT_DIR/cpu_alignment_heter_parallel_phase2_post_rerun_${JOB_TAG}.json"

echo "--- moe single layer post rerun ---"
python examples/benchmarks/moe_single_layer_cpu_gpu_parallel_bench.py \
  --output "$RESULT_DIR/moe_single_layer_cpu_gpu_parallel_bench_phase2_post_rerun_${JOB_TAG}.json" \
  --token-sizes 64,256 \
  --cpu-ratios 0,25,50,75,100 \
  --warmup 2 \
  --repeat 3 \
  --cpu-expert-parallel-mode serial \
  --cpu-expert-num-threads 1 \
  --cpu-gpu-parallel-min-cpu-route-ratio 0.7

echo "--- moe single layer breakdown rerun ---"
python examples/benchmarks/moe_single_layer_cpu_gpu_parallel_bench.py \
  --output "$RESULT_DIR/moe_single_layer_cpu_gpu_parallel_bench_phase2_breakdown_rerun_${JOB_TAG}.json" \
  --token-sizes 64,256 \
  --cpu-ratios 0,25,50,75,100 \
  --warmup 2 \
  --repeat 3 \
  --cpu-expert-parallel-mode serial \
  --cpu-expert-num-threads 1 \
  --cpu-gpu-parallel-min-cpu-route-ratio 0.7

echo "--- moe single layer small-token breakdown rerun ---"
python examples/benchmarks/moe_single_layer_cpu_gpu_parallel_bench.py \
  --output "$RESULT_DIR/moe_single_layer_cpu_gpu_parallel_bench_phase2_breakdown_small_tokens_1_3_5_10_20_rerun_${JOB_TAG}.json" \
  --token-sizes 1,3,5,10,20 \
  --cpu-ratios 0,25,50,75,100 \
  --warmup 2 \
  --repeat 3 \
  --cpu-expert-parallel-mode serial \
  --cpu-expert-num-threads 1 \
  --cpu-gpu-parallel-min-cpu-route-ratio 0.7

echo "--- spec verify ratio threshold0.7 ---"
python examples/benchmarks/spec_verify_cpu_ratio_bench.py \
  --model-path "$MODEL_PATH" \
  --cpu-ratios 25,50,75 \
  --parallel-settings off,on \
  --cpu-expert-parallel-mode serial \
  --cpu-expert-num-threads 1 \
  --cpu-gpu-parallel-min-cpu-route-ratio 0.7 \
  --num-seqs 2 \
  --input-len 32 \
  --output-len 8 \
  --max-model-len 256 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 16 \
  --repeat 2 \
  --dist-port-base 29300 \
  --seed 0 \
  --temperature 0.0 \
  --enforce-eager true \
  --engine-profile true \
  --engine-profile-cuda-sync true \
  --output "$RESULT_DIR/spec_verify_cpu_ratio_bench_phase2_post_min_${JOB_TAG}.json"

echo "--- spec verify ratio threshold0.7 rerun ---"
python examples/benchmarks/spec_verify_cpu_ratio_bench.py \
  --model-path "$MODEL_PATH" \
  --cpu-ratios 25,50,75 \
  --parallel-settings off,on \
  --cpu-expert-parallel-mode serial \
  --cpu-expert-num-threads 1 \
  --cpu-gpu-parallel-min-cpu-route-ratio 0.7 \
  --num-seqs 2 \
  --input-len 32 \
  --output-len 8 \
  --max-model-len 256 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 16 \
  --repeat 2 \
  --dist-port-base 29330 \
  --seed 0 \
  --temperature 0.0 \
  --enforce-eager true \
  --engine-profile true \
  --engine-profile-cuda-sync true \
  --output "$RESULT_DIR/spec_verify_cpu_ratio_bench_phase2_post_min_rerun_${JOB_TAG}.json"

echo "--- spec verify ratio threshold0.0 ---"
python examples/benchmarks/spec_verify_cpu_ratio_bench.py \
  --model-path "$MODEL_PATH" \
  --cpu-ratios 25,50,75 \
  --parallel-settings off,on \
  --cpu-expert-parallel-mode serial \
  --cpu-expert-num-threads 1 \
  --cpu-gpu-parallel-min-cpu-route-ratio 0.0 \
  --num-seqs 2 \
  --input-len 32 \
  --output-len 8 \
  --max-model-len 256 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 16 \
  --repeat 2 \
  --dist-port-base 29360 \
  --seed 0 \
  --temperature 0.0 \
  --enforce-eager true \
  --engine-profile true \
  --engine-profile-cuda-sync true \
  --output "$RESULT_DIR/spec_verify_cpu_ratio_bench_phase2_post_min_threshold0_${JOB_TAG}.json"

echo "--- spec verify ratio threshold0.0 rerun ---"
python examples/benchmarks/spec_verify_cpu_ratio_bench.py \
  --model-path "$MODEL_PATH" \
  --cpu-ratios 25,50,75 \
  --parallel-settings off,on \
  --cpu-expert-parallel-mode serial \
  --cpu-expert-num-threads 1 \
  --cpu-gpu-parallel-min-cpu-route-ratio 0.0 \
  --num-seqs 2 \
  --input-len 32 \
  --output-len 8 \
  --max-model-len 256 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 16 \
  --repeat 2 \
  --dist-port-base 29390 \
  --seed 0 \
  --temperature 0.0 \
  --enforce-eager true \
  --engine-profile true \
  --engine-profile-cuda-sync true \
  --output "$RESULT_DIR/spec_verify_cpu_ratio_bench_phase2_post_min_threshold0_rerun_${JOB_TAG}.json"

echo "--- real model cpugpu parallel bench ---"
python examples/benchmarks/moe_real_model_cpu_gpu_parallel_bench.py \
  --model-path "$MODEL_PATH" \
  --cpu-ratios 25,50,75 \
  --parallel-settings off,on \
  --cpu-expert-parallel-mode serial \
  --cpu-expert-num-threads 1 \
  --cpu-gpu-parallel-min-cpu-route-ratio 0.7 \
  --num-seqs 2 \
  --input-len 32 \
  --output-len 8 \
  --max-model-len 256 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 16 \
  --repeat 2 \
  --dist-port-base 29440 \
  --seed 0 \
  --temperature 0.0 \
  --enforce-eager true \
  --engine-profile true \
  --engine-profile-cuda-sync true \
  --output "$RESULT_DIR/moe_real_model_cpu_gpu_parallel_bench_phase2_post_${JOB_TAG}.json"

echo "=== DONE $(date -Is) ==="