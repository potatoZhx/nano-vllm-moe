#!/usr/bin/env bash
# Full spec benchmark: torch vs fused vs kt_kernel on A100 (80GB)
set -uo pipefail
REPO="/home/mumura/moe_spec/nano-vllm-moe"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/home/mumura/moe_spec/logs/full_spec_${TIMESTAMP}"
RES_DIR="${LOG_DIR}/results"
mkdir -p "${LOG_DIR}" "${RES_DIR}"
cd "${REPO}"
MODEL="/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B"

echo "=== Full Spec Comparison ===" | tee "${LOG_DIR}/summary.log"
echo "Timestamp: ${TIMESTAMP}" | tee -a "${LOG_DIR}/summary.log"
echo "Host: $(hostname)" | tee -a "${LOG_DIR}/summary.log"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | head -2 | tee -a "${LOG_DIR}/summary.log"

COMMON="--num-seqs 1 --input-len 12 --output-len 6 \
    --max-num-batched-tokens 1024 --max-num-seqs 64 --max-model-len 1024 \
    --gpu-memory-utilization 0.85 --max-draft-tokens 4 --draft-top-c 128 \
    --cpu-expert-execution-enabled true --cpu-expert-packed-min-routes 1 \
    --cpu-gpu-parallel-execution-enabled auto \
    --spec-profile true --engine-profile true --engine-profile-cuda-sync true \
    --spec-enable-prefetch false --temperature 0.0 --seed 0 --enforce-eager false \
    --return-token-ids true --return-text false --return-prompts false"

for ratio in 0.75 0.5 0.25; do
    slots=$(python3 -c "print(int(128 * ${ratio}))")
    echo "--- ratio=${ratio} slots=${slots} ---" | tee -a "${LOG_DIR}/summary.log"
    for backend in torch fused; do
        out="${RES_DIR}/spec_${backend}_${ratio}.json"
        echo "  backend=${backend}" | tee -a "${LOG_DIR}/summary.log"
        python examples/heterogeneous_benchmark_case.py \
            --model-path "${MODEL}" --mode spec \
            --slots-per-layer "${slots}" \
            --cpu-expert-backend "${backend}" \
            --dist-port 12345 --output "${out}" \
            ${COMMON} \
            2>&1 | tee "${LOG_DIR}/run_${backend}_${ratio}.log"
        echo "  exit=$?" | tee -a "${LOG_DIR}/summary.log"
    done
done

echo "" | tee -a "${LOG_DIR}/summary.log"
echo "=== Results ===" | tee -a "${LOG_DIR}/summary.log"
echo "backend | ratio | verify_ms | cpu_comp_ms | merge_ms | gpu_moe_ms | tok_s" | tee -a "${LOG_DIR}/summary.log"
echo "---|---|---|---|---|---|---" | tee -a "${LOG_DIR}/summary.log"
for f in ${RES_DIR}/*.json; do
    if [ -f "$f" ]; then
        python3 -c "
import json
d=json.load(open('${f}'))
ep=d.get('engine_profile',{})
name='${f}'.split('/')[-1].replace('.json','').replace('spec_','')
print('{} | {:.1f} | {:.1f} | {:.1f} | {:.1f} | {:.3f}'.format(
    name,
    ep.get('model_run_verify_total_ms',0),
    ep.get('model_verify_cpu_compute_ms',0),
    ep.get('model_verify_cpu_to_gpu_merge_ms',0),
    ep.get('model_verify_gpu_compute_ms',0),
    d.get('throughput_output_tok_s',0)))
" | tee -a "${LOG_DIR}/summary.log"
    fi
done
echo "Full logs: ${LOG_DIR}" | tee -a "${LOG_DIR}/summary.log"
