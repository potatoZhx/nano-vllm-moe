#!/usr/bin/env bash
# Full spec benchmark: torch vs fused on RTX4090 (24GB, AMX CPU)
# Fixed VRAM: reduced KV cache to fit 25% expert cache
set -uo pipefail  # no -e: handle errors explicitly
REPO="/home/mumura/moe_spec/nano-vllm-moe"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/home/mumura/moe_spec/logs/kt_spec_${TIMESTAMP}"
RES_DIR="${LOG_DIR}/results"
mkdir -p "${LOG_DIR}" "${RES_DIR}"
cd "${REPO}"
MODEL="/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B"
# Use python from PATH (set by conda activate)

echo "=== kt-kernel Full Spec Benchmark ===" | tee "${LOG_DIR}/summary.log"
echo "Timestamp: ${TIMESTAMP}" | tee -a "${LOG_DIR}/summary.log"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -2 | tee -a "${LOG_DIR}/summary.log"

# Fixed VRAM params for 24GB:
# expert cache (25% = 32 slots): 48*32*9.44MB = 14.5GB
# non-expert weights: ~3GB
# KV cache (4seqs*512len): ~200MB
# Total: ~18GB, fits in 24GB
COMMON="--num-seqs 1 --input-len 12 --output-len 6 \
    --max-num-batched-tokens 512 --max-num-seqs 4 --max-model-len 512 \
    --gpu-memory-utilization 0.85 --max-draft-tokens 4 --draft-top-c 128 \
    --cpu-expert-execution-enabled true --cpu-expert-packed-min-routes 1 \
    --cpu-gpu-parallel-execution-enabled auto \
    --spec-profile true --engine-profile true --engine-profile-cuda-sync true \
    --spec-enable-prefetch false --temperature 0.0 --seed 0 --enforce-eager false \
    --return-token-ids true --return-text false --return-prompts false"

for ratio in 0.25 0.125; do
    slots=$(python -c "print(int(128 * ${ratio}))")
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
echo "backend | ratio | verify_ms | cpu_comp_ms | merge_ms | gpu_moe_ms | tok_s | digest" | tee -a "${LOG_DIR}/summary.log"
echo "---|---|---|---|---|---|---|---" | tee -a "${LOG_DIR}/summary.log"
for f in ${RES_DIR}/*.json; do
    if [ -f "$f" ]; then
        python -c "
import json
d=json.load(open('${f}'))
ep=d.get('engine_profile',{})
name='${f}'.split('/')[-1].replace('.json','').replace('spec_','')
print('{} | {:.1f} | {:.1f} | {:.1f} | {:.1f} | {:.3f} | {}'.format(
    name,
    ep.get('model_run_verify_total_ms',0),
    ep.get('model_verify_cpu_compute_ms',0),
    ep.get('model_verify_cpu_to_gpu_merge_ms',0),
    ep.get('model_verify_gpu_compute_ms',0),
    d.get('throughput_output_tok_s',0),
    d.get('outputs_digest','')[:16]))
" | tee -a "${LOG_DIR}/summary.log"
    fi
done
echo "Full logs: ${LOG_DIR}" | tee -a "${LOG_DIR}/summary.log"
