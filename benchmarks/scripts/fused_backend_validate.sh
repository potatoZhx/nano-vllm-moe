#!/usr/bin/env bash
set -euo pipefail

REPO="/home/mumura/moe_spec/nano-vllm-moe"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/home/mumura/moe_spec/logs/fused_backend_${TIMESTAMP}"
RES_DIR="${LOG_DIR}/results"
mkdir -p "${LOG_DIR}" "${RES_DIR}"
cd "${REPO}"
MODEL_PATH="/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B"

echo "=== Fused CPU Backend Validation ===" | tee "${LOG_DIR}/summary.log"
echo "Timestamp: ${TIMESTAMP}" | tee -a "${LOG_DIR}/summary.log"

# Test 1: Correctness
echo "--- Test 1: Correctness ---" | tee -a "${LOG_DIR}/summary.log"
python -m pytest -q \
    tests/test_cpu_moe_correctness.py \
    tests/test_cpu_gpu_expert_operator_alignment.py \
    tests/test_cpu_gpu_parallel_moe.py \
    2>&1 | tee "${LOG_DIR}/correctness.log"
PASSED=$(tail -1 "${LOG_DIR}/correctness.log" | grep -oP '\d+(?= passed)' || echo "0")
echo "Correctness: ${PASSED} passed" | tee -a "${LOG_DIR}/summary.log"

# Test 2: Synthetic benchmark (all 3 backends)
echo "--- Test 2: Synthetic Benchmark ---" | tee -a "${LOG_DIR}/summary.log"
python benchmarks/bench_cpu_moe_backend.py \
    --tokens 1,8,32,128 --cpu-route-ratio 0.25,0.5,0.75 \
    --hidden-size 2048 --intermediate-size 768 --num-experts 128 \
    --backend torch --backend torch_packed --backend fused \
    --iterations 8 --warmup 3 --packed-min-routes 1 \
    --output "${RES_DIR}/bench_synthetic.csv" \
    2>&1 | tee "${LOG_DIR}/bench_synthetic.log"

python3 -c "
import csv
rows = list(csv.DictReader(open('${RES_DIR}/bench_synthetic.csv')))
print(f'Synthetic benchmark: {len(rows)} rows')
for backend in ['torch', 'torch_packed', 'fused']:
    br = [r for r in rows if r['backend'] == backend]
    if br:
        d = sum(float(r['decode_forward_ms']) for r in br) / len(br)
        m = sum(float(r['cpu_to_gpu_merge_ms']) for r in br) / len(br)
        c = sum(float(r['cpu_compute_ms']) for r in br) / len(br)
        print(f'  {backend:15s}: decode={d:.2f}ms merge={m:.2f}ms compute={c:.2f}ms')
" 2>&1 | tee -a "${LOG_DIR}/summary.log"

# Test 3: Spec benchmarks (torch vs fused)
echo "--- Test 3: Spec Benchmark ---" | tee -a "${LOG_DIR}/summary.log"
for backend in torch fused; do
    for ratio in 0.75 0.5 0.25; do
        slots=$(python -c "print(int(128 * ${ratio}))")
        echo "  ratio=${ratio} slots=${slots} backend=${backend}" | tee -a "${LOG_DIR}/summary.log"
        python examples/heterogeneous_benchmark_case.py \
            --model-path "${MODEL_PATH}" --mode spec \
            --slots-per-layer "${slots}" \
            --num-seqs 1 --input-len 12 --output-len 6 \
            --max-num-batched-tokens 1024 --max-num-seqs 64 --max-model-len 1024 \
            --gpu-memory-utilization 0.85 --max-draft-tokens 4 --draft-top-c 128 \
            --cpu-expert-execution-enabled true --cpu-expert-backend ${backend} \
            --cpu-expert-packed-min-routes 1 \
            --cpu-gpu-parallel-execution-enabled auto \
            --spec-profile true --engine-profile true --engine-profile-cuda-sync true \
            --spec-enable-prefetch false --temperature 0.0 --seed 0 --enforce-eager false \
            --return-token-ids true --return-text false --return-prompts false \
            --dist-port 12345 --output "${RES_DIR}/spec_${backend}_${ratio}.json" \
            2>&1 | tee "${LOG_DIR}/spec_${backend}_${ratio}.log"
    done
done

# Summary
echo "" | tee -a "${LOG_DIR}/summary.log"
echo "=== Summary ===" | tee -a "${LOG_DIR}/summary.log"
echo "backend | ratio | verify_ms | cpu_comp_ms | cpu_merge_ms | tok_s | digest" | tee -a "${LOG_DIR}/summary.log"
echo "---|---|---|---|---|---|---" | tee -a "${LOG_DIR}/summary.log"
for backend in torch fused; do
    for ratio in 0.75 0.5 0.25; do
        json="${RES_DIR}/spec_${backend}_${ratio}.json"
        if [ -f "${json}" ]; then
            python3 -c "
import json
d = json.load(open('${json}'))
ep = d.get('engine_profile', {})
v = ep.get('model_run_verify_total_ms', 0)
c = ep.get('model_verify_cpu_compute_ms', 0)
m = ep.get('model_verify_cpu_to_gpu_merge_ms', 0)
t = d.get('throughput_output_tok_s', 0)
dig = d.get('outputs_digest','')[:16]
print(f'${backend} | ${ratio} | {v:.1f} | {c:.1f} | {m:.1f} | {t:.3f} | {dig}')
" 2>&1 | tee -a "${LOG_DIR}/summary.log"
        fi
    done
done
echo "Full logs: ${LOG_DIR}"
