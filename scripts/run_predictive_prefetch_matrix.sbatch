#!/usr/bin/env bash
#SBATCH -p A100
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --gres=gpu:1
#SBATCH -t 06:00:00
#SBATCH -J predictive_prefetch_matrix

set -eo pipefail

REPO_ROOT=${REPO_ROOT:-/home/mumura/moe_spec/nano-vllm-moe}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT/results/predictive_prefetch_validation_${SLURM_JOB_ID:-manual}"}
REPORT_DOC=${REPORT_DOC:-"$REPO_ROOT/docs/predictive_prefetch_validation_20260601.md"}
DIST_PORT_BASE=${DIST_PORT_BASE:-29600}

source /opt/Software/Anaconda3/etc/profile.d/conda.sh
conda activate nano_moe

cd "$REPO_ROOT"

echo "timestamp=$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "job_id=${SLURM_JOB_ID:-}"
echo "host=$(hostname)"
echo "pwd=$(pwd)"
echo "git_sha=$(git rev-parse HEAD)"
echo "git_status_short=$(git status --short | wc -l)"
echo "conda_env=${CONDA_DEFAULT_ENV:-}"
echo "initial_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
PY

nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "keeping_slurm_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
elif command -v nvidia-smi >/dev/null 2>&1; then
  BEST_GPU=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3); print $2, $3, $1}' \
    | sort -n -k1,1 -k2,2 \
    | head -n 1 \
    | awk '{print $3}')
  if [ -n "${BEST_GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$BEST_GPU"
  fi
fi
echo "selected_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true

mkdir -p "$OUTPUT_DIR"
python scripts/predictive_prefetch_validation.py \
  --output-dir "$OUTPUT_DIR" \
  --report-doc "$REPORT_DOC" \
  --dist-port-base "$DIST_PORT_BASE" \
  --case-timeout-sec "${CASE_TIMEOUT_SEC:-2400}"
