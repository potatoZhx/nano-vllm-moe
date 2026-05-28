#!/bin/bash
#SBATCH --job-name=reroute_val
#SBATCH --partition=A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
#SBATCH --time=04:00:00
#SBATCH --output=/home/mumura/moe_spec/logs/reroute_full_validation_%j.log
#SBATCH --error=/home/mumura/moe_spec/logs/reroute_full_validation_%j.err

set -e

echo "=== Environment ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

source ~/.bashrc
conda activate nano_moe

cd /home/mumura/moe_spec/nano-vllm-moe

echo "=== Python ==="
which python
python --version
echo ""

echo "=== PyTorch GPU Check ==="
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="results/reroute_full_validation_${TIMESTAMP}"

echo "=== Starting Full Validation ==="
echo "Output dir: $OUTDIR"
echo "Start time: $(date)"
echo ""

python scripts/reroute_full_validation.py \
    --model-path /data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B \
    --calibration-artifact results/reroute_impl_20260527/calibration/v2_calibration_smoke.pt \
    --output-dir "$OUTDIR" \
    --input-len 128 \
    --max-draft-tokens 8 \
    --draft-top-c 0 \
    --max-model-len 2048 \
    --seed 0 \
    --temperature 0.8 \
    --cpu-expert-pin-memory

echo ""
echo "=== Done ==="
echo "End time: $(date)"
echo "Output dir: $OUTDIR"
