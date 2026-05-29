#!/bin/bash
#SBATCH --job-name=prec_val
#SBATCH --partition=A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
#SBATCH --time=08:00:00
#SBATCH --output=/home/mumura/moe_spec/logs/precision_validate_%j.log
#SBATCH --error=/home/mumura/moe_spec/logs/precision_validate_%j.err

set -e
echo "=== Job: $SLURM_JOB_ID Node: $SLURM_NODELIST ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source ~/.bashrc
conda activate nano_moe
cd /home/mumura/moe_spec/nano-vllm-moe

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="results/precision_validation_${TIMESTAMP}"

echo "=== Starting precision validation ==="
echo "Output: $OUTDIR"
echo "Start: $(date)"

python scripts/precision_validate.py \
    --model-path /data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B \
    --calibration-artifact results/reroute_impl_20260527/calibration/v2_calibration_smoke.pt \
    --output-dir "$OUTDIR" \
    --output-len 128 \
    --max-model-len 2048 \
    --cpu-expert-pin-memory \
    --case-timeout-sec 1800

echo "=== Done: $(date) ==="
echo "Results: $OUTDIR"
