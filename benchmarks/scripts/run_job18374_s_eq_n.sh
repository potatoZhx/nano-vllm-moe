#!/usr/bin/env bash
set -euo pipefail
cd /home/mumura/moe_spec/nano-vllm-moe
TS=$(date +%Y%m%d_%H%M%S)
OUT_JSON="benchmarks/results/phase3_real_e2e_orchestrator_job18374_s_eq_n_${TS}.json"
OUT_LOG="benchmarks/results/phase3_real_e2e_orchestrator_job18374_s_eq_n_${TS}.log"
MODEL_PATH="/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B"
mkdir -p /home/mumura/.cache/triton_tmp /home/mumura/.cache/triton_cache
echo "OUT_JSON=$OUT_JSON"
echo "OUT_LOG=$OUT_LOG"
srun --jobid 18374 --overlap --ntasks=1 bash -lc "
  set -e
  source /opt/Software/Anaconda3/etc/profile.d/conda.sh
  conda activate nano_moe
  export TMPDIR=/home/mumura/.cache/triton_tmp
  export TRITON_CACHE_DIR=/home/mumura/.cache/triton_cache
  cd /home/mumura/moe_spec/nano-vllm-moe
  PYTHONPATH=. python benchmarks/scripts/phase3_real_e2e_orchestrator.py \
    --model-path '$MODEL_PATH' \
    --num-seqs 2 \
    --input-len 32 \
    --output-len 8 \
    --max-num-batched-tokens 1024 \
    --max-num-seqs 32 \
    --max-model-len 512 \
    --gpu-memory-utilization 0.99 \
    --slots-per-layer 128 \
    --prefetch-wait-ms 1.0 \
    --base-dist-port 29900 \
    --output '$OUT_JSON' \
    2>&1 | tee '$OUT_LOG'
"
echo "DONE_JSON=$OUT_JSON"
echo "DONE_LOG=$OUT_LOG"
