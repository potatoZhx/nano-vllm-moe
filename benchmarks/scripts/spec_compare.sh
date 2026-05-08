#!/usr/bin/env bash
# Spec benchmark comparison: torch vs fused on RTX4090 (24GB)
set -euo pipefail
REPO="/home/mumura/moe_spec/nano-vllm-moe"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/home/mumura/moe_spec/logs/spec_compare_${TIMESTAMP}"
RES_DIR="${LOG_DIR}/results"
mkdir -p "${LOG_DIR}" "${RES_DIR}"
cd "${REPO}"
MODEL="/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B"

echo "=== Spec Comparison ===" | tee "${LOG_DIR}/summary.log"
echo "Timestamp: ${TIMESTAMP}" | tee -a "${LOG_DIR}/summary.log"
echo "Host: $(hostname)" | tee -a "${LOG_DIR}/summary.log"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | head -2 | tee -a "${LOG_DIR}/summary.log"

# Only test safe ratios on 24GB
for ratio in 0.125 0.0625; do
    slots=$(python3 -c "print(int(128 * ${ratio}))")
    echo "--- ratio=${ratio} slots=${slots} ---" | tee -a "${LOG_DIR}/summary.log"

    for backend in torch fused; do
        out="${RES_DIR}/spec_${backend}_${ratio}.json"
        echo "  backend=${backend}" | tee -a "${LOG_DIR}/summary.log"
        python examples/heterogeneous_benchmark_case.py \
            --model-path "${MODEL}" --mode spec \
            --slots-per-layer "${slots}" \
            --num-seqs 1 --input-len 12 --output-len 6 \
            --max-num-batched-tokens 1024 --max-num-seqs 64 --max-model-len 1024 \
            --gpu-memory-utilization 0.75 --max-draft-tokens 4 --draft-top-c 128 \
            --cpu-expert-execution-enabled true --cpu-expert-backend "${backend}" \
            --cpu-expert-packed-min-routes 1 \
            --cpu-gpu-parallel-execution-enabled auto \
            --spec-profile true --engine-profile true --engine-profile-cuda-sync true \
            --spec-enable-prefetch false --temperature 0.0 --seed 0 --enforce-eager false \
            --return-token-ids true --return-text false --return-prompts false \
            --dist-port 12345 --output "${out}" \
            2>&1 | tee "${LOG_DIR}/run_${backend}_${ratio}.log"
        echo "  done, exit=$?" | tee -a "${LOG_DIR}/summary.log"
    done
done

echo "" | tee -a "${LOG_DIR}/summary.log"
echo "=== Results ===" | tee -a "${LOG_DIR}/summary.log"
echo "backend | ratio | verify_ms | cpu_comp_ms | merge_ms | gpu_moe_ms | tok_s | digest" | tee -a "${LOG_DIR}/summary.log"
echo "---|---|---|---|---|---|---|---" | tee -a "${LOG_DIR}/summary.log"
for backend in torch fused; do
    for ratio in 0.125 0.0625; do
        json="${RES_DIR}/spec_${backend}_${ratio}.json"
        if [ -f "${json}" ]; then
            python3 -c "
import json
d=json.load(open('${json}'))
ep=d.get('engine_profile',{})
print('${backend} | ${ratio} | {:.1f} | {:.1f} | {:.1f} | {:.1f} | {:.3f} | {}'.format(
    ep.get('model_run_verify_total_ms',0),
    ep.get('model_verify_cpu_compute_ms',0),
    ep.get('model_verify_cpu_to_gpu_merge_ms',0),
    ep.get('model_verify_gpu_compute_ms',0),
    d.get('throughput_output_tok_s',0),
    d.get('outputs_digest','')[:16]))
" | tee -a "${LOG_DIR}/summary.log"
        fi
    done
done
echo "Full logs: ${LOG_DIR}" | tee -a "${LOG_DIR}/summary.log"
