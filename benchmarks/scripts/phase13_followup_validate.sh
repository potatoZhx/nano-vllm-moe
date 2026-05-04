#!/usr/bin/env bash
set -euo pipefail

# Phase 1-3 follow-up optimization validation
# Tests: correctness, synthetic benchmark, spec benchmark

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="/home/mumura/moe_spec/nano-vllm-moe"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/home/mumura/moe_spec/logs/phase13_followup_${TIMESTAMP}"
RES_DIR="${LOG_DIR}/results"
mkdir -p "${LOG_DIR}" "${RES_DIR}"

cd "${REPO}"

echo "=== Phase 1-3 Follow-up Validation ===" | tee "${LOG_DIR}/summary.log"
echo "Timestamp: ${TIMESTAMP}" | tee -a "${LOG_DIR}/summary.log"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo 'N/A')" | tee -a "${LOG_DIR}/summary.log"
echo "" | tee -a "${LOG_DIR}/summary.log"

# --------------------------------------------------
# Test 1: CUDA Correctness
# --------------------------------------------------
echo "--- Test 1: CUDA Correctness ---" | tee -a "${LOG_DIR}/summary.log"
python -m pytest -q \
    tests/test_cpu_moe_correctness.py \
    tests/test_cpu_gpu_expert_operator_alignment.py \
    tests/test_cpu_gpu_parallel_moe.py \
    2>&1 | tee "${LOG_DIR}/correctness.log"

CORRECT_PASSED=$(grep -c 'passed' "${LOG_DIR}/correctness.log" || echo "0")
echo "Correctness: ${CORRECT_PASSED} passed tests" | tee -a "${LOG_DIR}/summary.log"
echo "" | tee -a "${LOG_DIR}/summary.log"

# --------------------------------------------------
# Test 2: Synthetic CPU Backend Benchmark
# --------------------------------------------------
echo "--- Test 2: Synthetic Benchmark ---" | tee -a "${LOG_DIR}/summary.log"

MODEL_PATH="/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B"
HIDDEN_SIZE=2048
INTERMEDIATE_SIZE=768
NUM_EXPERTS=128

python benchmarks/bench_cpu_moe_backend.py \
    --tokens 1,8,32,128 \
    --cpu-route-ratio 0.25,0.5,0.75 \
    --hidden-size ${HIDDEN_SIZE} \
    --intermediate-size ${INTERMEDIATE_SIZE} \
    --num-experts ${NUM_EXPERTS} \
    --iterations 8 \
    --warmup 3 \
    --output "${RES_DIR}/bench_synthetic.csv" \
    2>&1 | tee "${LOG_DIR}/bench_synthetic.log"

python -c "
import csv
rows = list(csv.DictReader(open('${RES_DIR}/bench_synthetic.csv')))
print(f'Synthetic benchmark: {len(rows)} rows')
for backend in ['torch', 'torch_packed']:
    backend_rows = [r for r in rows if r['backend'] == backend]
    if backend_rows:
        avg_decode = sum(float(r['decode_forward_ms']) for r in backend_rows) / len(backend_rows)
        avg_merge = sum(float(r['cpu_to_gpu_merge_ms']) for r in backend_rows) / len(backend_rows)
        print(f'  {backend}: avg decode={avg_decode:.3f}ms, avg merge={avg_merge:.3f}ms')
" 2>&1 | tee -a "${LOG_DIR}/summary.log"

echo "" | tee -a "${LOG_DIR}/summary.log"

# --------------------------------------------------
# Test 3: Spec Benchmark - Baseline (serial)
# --------------------------------------------------
echo "--- Test 3: Spec Benchmark (serial mode) ---" | tee -a "${LOG_DIR}/summary.log"

for ratio in 0.75 0.5 0.25; do
    slots=$(python -c "print(int(128 * ${ratio}))")
    echo "  ratio=${ratio} slots=${slots} backend=torch mode=serial" | tee -a "${LOG_DIR}/summary.log"

    python examples/heterogeneous_benchmark_case.py \
        --model-path "${MODEL_PATH}" \
        --mode spec \
        --slots-per-layer "${slots}" \
        --num-seqs 1 \
        --input-len 12 \
        --output-len 6 \
        --max-num-batched-tokens 1024 \
        --max-num-seqs 64 \
        --max-model-len 1024 \
        --gpu-memory-utilization 0.85 \
        --max-draft-tokens 4 \
        --draft-top-c 128 \
        --cpu-expert-execution-enabled true \
        --cpu-expert-backend torch \
        --cpu-expert-parallel-mode serial \
        --cpu-expert-packed-min-routes 32 \
        --spec-profile true \
        --engine-profile true \
        --engine-profile-cuda-sync true \
        --spec-enable-prefetch false \
        --temperature 0.0 \
        --seed 0 \
        --enforce-eager false \
        --return-token-ids true \
        --return-text false \
        --return-prompts false \
        --dist-port 12345 \
        --output "${RES_DIR}/spec_serial_${ratio}.json" \
        2>&1 | tee "${LOG_DIR}/spec_serial_${ratio}.log"
done

# --------------------------------------------------
# Test 4: Spec Benchmark - Auto mode (dynamic parallel)
# --------------------------------------------------
echo "--- Test 4: Spec Benchmark (auto mode) ---" | tee -a "${LOG_DIR}/summary.log"

for ratio in 0.75 0.5 0.25; do
    slots=$(python -c "print(int(128 * ${ratio}))")
    echo "  ratio=${ratio} slots=${slots} backend=torch mode=auto" | tee -a "${LOG_DIR}/summary.log"

    python examples/heterogeneous_benchmark_case.py \
        --model-path "${MODEL_PATH}" \
        --mode spec \
        --slots-per-layer "${slots}" \
        --num-seqs 1 \
        --input-len 12 \
        --output-len 6 \
        --max-num-batched-tokens 1024 \
        --max-num-seqs 64 \
        --max-model-len 1024 \
        --gpu-memory-utilization 0.85 \
        --max-draft-tokens 4 \
        --draft-top-c 128 \
        --cpu-expert-execution-enabled true \
        --cpu-expert-backend torch \
        --cpu-expert-parallel-mode auto \
        --cpu-expert-packed-min-routes 32 \
        --spec-profile true \
        --engine-profile true \
        --engine-profile-cuda-sync true \
        --spec-enable-prefetch false \
        --temperature 0.0 \
        --seed 0 \
        --enforce-eager false \
        --return-token-ids true \
        --return-text false \
        --return-prompts false \
        --dist-port 12346 \
        --output "${RES_DIR}/spec_auto_${ratio}.json" \
        2>&1 | tee "${LOG_DIR}/spec_auto_${ratio}.log"
done

echo "" | tee -a "${LOG_DIR}/summary.log"
echo "=== Summary of Results ===" | tee -a "${LOG_DIR}/summary.log"

for ratio in 0.75 0.5 0.25; do
    echo "--- Ratio ${ratio} ---" | tee -a "${LOG_DIR}/summary.log"
    for mode in serial auto; do
        json="${RES_DIR}/spec_${mode}_${ratio}.json"
        if [ -f "${json}" ]; then
            python -c "
import json
d = json.load(open('${json}'))
ep = d.get('engine_profile', {})
verify_ms = ep.get('model_run_verify_total_ms', 0)
cpu_compute = ep.get('model_verify_cpu_compute_ms', 0)
cpu_merge = ep.get('model_verify_cpu_to_gpu_merge_ms', 0)
gpu_moe = ep.get('model_verify_gpu_compute_ms', 0)
plan_ms = ep.get('model_verify_plan_ms', 0)
tok_s = d.get('throughput_output_tok_s', 0)
digest = d.get('outputs_digest', 'N/A')[:16]
print(f'  {mode}: verify={verify_ms:.1f}ms cpu_comp={cpu_compute:.1f}ms cpu_merge={cpu_merge:.1f}ms gpu_moe={gpu_moe:.1f}ms plan={plan_ms:.1f}ms tok/s={tok_s:.3f} digest={digest}...')
" 2>&1 | tee -a "${LOG_DIR}/summary.log"
        fi
    done
done

echo "" | tee -a "${LOG_DIR}/summary.log"
echo "Full logs: ${LOG_DIR}" | tee -a "${LOG_DIR}/summary.log"
echo "Done."
