#!/usr/bin/env bash
set -euo pipefail

# Test CPU-GPU parallel execution mode: off vs auto vs on
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="/home/mumura/moe_spec/nano-vllm-moe"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/home/mumura/moe_spec/logs/parallel_mode_${TIMESTAMP}"
RES_DIR="${LOG_DIR}/results"
mkdir -p "${LOG_DIR}" "${RES_DIR}"

cd "${REPO}"
MODEL_PATH="/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B"

echo "=== CPU-GPU Parallel Execution Mode Validation ===" | tee "${LOG_DIR}/summary.log"
echo "Timestamp: ${TIMESTAMP}" | tee -a "${LOG_DIR}/summary.log"
echo "" | tee -a "${LOG_DIR}/summary.log"

# --------------------------------------------------
# Test 1: Correctness
# --------------------------------------------------
echo "--- Test 1: Correctness ---" | tee -a "${LOG_DIR}/summary.log"
python -m pytest -q \
    tests/test_cpu_moe_correctness.py \
    tests/test_cpu_gpu_expert_operator_alignment.py \
    tests/test_cpu_gpu_parallel_moe.py \
    2>&1 | tee "${LOG_DIR}/correctness.log"

PASSED=$(tail -1 "${LOG_DIR}/correctness.log" | grep -oP '\d+(?= passed)' || echo "0")
echo "Correctness: ${PASSED} passed" | tee -a "${LOG_DIR}/summary.log"
echo "" | tee -a "${LOG_DIR}/summary.log"

# --------------------------------------------------
# Test 2: Spec benchmark - parallel=off
# --------------------------------------------------
echo "--- Test 2: Spec benchmark (parallel=off) ---" | tee -a "${LOG_DIR}/summary.log"

for ratio in 0.75 0.5 0.25; do
    slots=$(python -c "print(int(128 * ${ratio}))")
    echo "  ratio=${ratio} slots=${slots} parallel=off" | tee -a "${LOG_DIR}/summary.log"

    python examples/heterogeneous_benchmark_case.py \
        --model-path "${MODEL_PATH}" \
        --mode spec \
        --slots-per-layer "${slots}" \
        --num-seqs 1 --input-len 12 --output-len 6 \
        --max-num-batched-tokens 1024 --max-num-seqs 64 --max-model-len 1024 \
        --gpu-memory-utilization 0.85 \
        --max-draft-tokens 4 --draft-top-c 128 \
        --cpu-expert-execution-enabled true \
        --cpu-expert-backend torch \
        --cpu-expert-packed-min-routes 32 \
        --cpu-gpu-parallel-execution-enabled off \
        --spec-profile true --engine-profile true --engine-profile-cuda-sync true \
        --spec-enable-prefetch false \
        --temperature 0.0 --seed 0 --enforce-eager false \
        --return-token-ids true --return-text false --return-prompts false \
        --dist-port 12345 \
        --output "${RES_DIR}/parallel_off_${ratio}.json" \
        2>&1 | tee "${LOG_DIR}/parallel_off_${ratio}.log"
done

# --------------------------------------------------
# Test 3: Spec benchmark - parallel=auto
# --------------------------------------------------
echo "--- Test 3: Spec benchmark (parallel=auto) ---" | tee -a "${LOG_DIR}/summary.log"

for ratio in 0.75 0.5 0.25; do
    slots=$(python -c "print(int(128 * ${ratio}))")
    echo "  ratio=${ratio} slots=${slots} parallel=auto" | tee -a "${LOG_DIR}/summary.log"

    python examples/heterogeneous_benchmark_case.py \
        --model-path "${MODEL_PATH}" \
        --mode spec \
        --slots-per-layer "${slots}" \
        --num-seqs 1 --input-len 12 --output-len 6 \
        --max-num-batched-tokens 1024 --max-num-seqs 64 --max-model-len 1024 \
        --gpu-memory-utilization 0.85 \
        --max-draft-tokens 4 --draft-top-c 128 \
        --cpu-expert-execution-enabled true \
        --cpu-expert-backend torch \
        --cpu-expert-packed-min-routes 32 \
        --cpu-gpu-parallel-execution-enabled auto \
        --spec-profile true --engine-profile true --engine-profile-cuda-sync true \
        --spec-enable-prefetch false \
        --temperature 0.0 --seed 0 --enforce-eager false \
        --return-token-ids true --return-text false --return-prompts false \
        --dist-port 12346 \
        --output "${RES_DIR}/parallel_auto_${ratio}.json" \
        2>&1 | tee "${LOG_DIR}/parallel_auto_${ratio}.log"
done

# --------------------------------------------------
# Test 4: Spec benchmark - parallel=on (force all)
# --------------------------------------------------
echo "--- Test 4: Spec benchmark (parallel=on) ---" | tee -a "${LOG_DIR}/summary.log"

for ratio in 0.75 0.5 0.25; do
    slots=$(python -c "print(int(128 * ${ratio}))")
    echo "  ratio=${ratio} slots=${slots} parallel=on" | tee -a "${LOG_DIR}/summary.log"

    python examples/heterogeneous_benchmark_case.py \
        --model-path "${MODEL_PATH}" \
        --mode spec \
        --slots-per-layer "${slots}" \
        --num-seqs 1 --input-len 12 --output-len 6 \
        --max-num-batched-tokens 1024 --max-num-seqs 64 --max-model-len 1024 \
        --gpu-memory-utilization 0.85 \
        --max-draft-tokens 4 --draft-top-c 128 \
        --cpu-expert-execution-enabled true \
        --cpu-expert-backend torch \
        --cpu-expert-packed-min-routes 32 \
        --cpu-gpu-parallel-execution-enabled on \
        --spec-profile true --engine-profile true --engine-profile-cuda-sync true \
        --spec-enable-prefetch false \
        --temperature 0.0 --seed 0 --enforce-eager false \
        --return-token-ids true --return-text false --return-prompts false \
        --dist-port 12347 \
        --output "${RES_DIR}/parallel_on_${ratio}.json" \
        2>&1 | tee "${LOG_DIR}/parallel_on_${ratio}.log"
done

# --------------------------------------------------
# Summary
# --------------------------------------------------
echo "" | tee -a "${LOG_DIR}/summary.log"
echo "=== Summary ===" | tee -a "${LOG_DIR}/summary.log"
echo "mode | ratio | verify_ms | cpu_comp_ms | gpu_moe_ms | parallel_wall_ms | parallel_enabled_count | tok_s" | tee -a "${LOG_DIR}/summary.log"
echo "---|---|---|---|---|---|---|---" | tee -a "${LOG_DIR}/summary.log"

for mode in off auto on; do
    for ratio in 0.75 0.5 0.25; do
        json="${RES_DIR}/parallel_${mode}_${ratio}.json"
        if [ -f "${json}" ]; then
            python3 -c "
import json
d = json.load(open('${json}'))
ep = d.get('engine_profile', {})
verify = ep.get('model_run_verify_total_ms', 0)
cpu_comp = ep.get('model_verify_cpu_compute_ms', 0)
gpu_moe = ep.get('model_verify_gpu_compute_ms', 0)
par_wall = ep.get('model_verify_parallel_wall_ms', 0)
par_count = ep.get('model_parallel_enabled_count', 0)
tok_s = d.get('throughput_output_tok_s', 0)
digest = d.get('outputs_digest','')[:16]
print(f'${mode} | ${ratio} | {verify:.1f} | {cpu_comp:.1f} | {gpu_moe:.1f} | {par_wall:.1f} | {par_count:.0f} | {tok_s:.3f} | {digest}')
" 2>&1 | tee -a "${LOG_DIR}/summary.log"
        fi
    done
done

echo "" | tee -a "${LOG_DIR}/summary.log"
echo "Full logs: ${LOG_DIR}" | tee -a "${LOG_DIR}/summary.log"
