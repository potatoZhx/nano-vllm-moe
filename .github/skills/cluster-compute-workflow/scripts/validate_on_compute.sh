#!/usr/bin/env bash

set -o pipefail

ROOT_DIR="/home/mumura/moe_spec"
LOG_DIR="${ROOT_DIR}/logs"
REQUESTED_JOBID="${1:-15299}"

mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/cluster_compute_skill_${TS}.log"

{
  echo "=== cluster compute validation start $(date -Is) ==="
  echo "login hostname=$(hostname)"
  echo "requested jobid=${REQUESTED_JOBID}"

  echo "=== squeue -u ${USER} ==="
  squeue -u "${USER}"

  TARGET_JOBID="${REQUESTED_JOBID}"
  if [[ -z "$(squeue -h -j "${TARGET_JOBID}" -o "%A" || true)" ]]; then
    FALLBACK_JOBID="$(squeue -h -u "${USER}" -o "%A" | head -n 1 || true)"
    if [[ -n "${FALLBACK_JOBID}" ]]; then
      echo "requested jobid not found, fallback to ${FALLBACK_JOBID}"
      TARGET_JOBID="${FALLBACK_JOBID}"
    else
      echo "no active job found for user ${USER}; submit an A100 job first"
      echo "example: sbatch --partition=A100 --gres=gpu:a100:1 --cpus-per-task=8 --mem=64G --time=02:00:00 --wrap 'sleep infinity'"
      exit 2
    fi
  fi

  echo "=== entering compute node with jobid=${TARGET_JOBID} ==="
  SRUN_STATUS=0
  srun --jobid="${TARGET_JOBID}" --pty bash -lc '
    set -o pipefail
    echo "compute hostname=$(hostname)"
    source ~/.bashrc || true
    conda activate nano_moe
    echo "CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"
    echo "python=$(which python)"
    python -V
    cd /home/mumura/moe_spec/nano-vllm-moe

    python - <<"PY"
import torch
import nanovllm
from nanovllm.sampling_params import SamplingParams

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
sp = SamplingParams(max_tokens=16, temperature=0.8, ignore_eos=False)
print("sampling_params_ok", isinstance(sp, SamplingParams), sp)
print("nanovllm_import_ok", nanovllm.__name__)
PY
  ' || SRUN_STATUS=$?

  echo "SRUN_STATUS=${SRUN_STATUS}"
  echo "=== cluster compute validation end $(date -Is) ==="

  if [[ "${SRUN_STATUS}" -ne 0 ]]; then
    exit "${SRUN_STATUS}"
  fi
} 2>&1 | tee "${LOG_FILE}"

echo "LOG_FILE=${LOG_FILE}"
