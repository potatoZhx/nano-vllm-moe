#!/bin/bash
#SBATCH --job-name=reroute_meaning
#SBATCH --partition=A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/mumura/moe_spec/logs/reroute_meaningful_%j.log
#SBATCH --error=/home/mumura/moe_spec/logs/reroute_meaningful_%j.err

source ~/.bashrc
conda activate nano_moe
cd /home/mumura/moe_spec/nano-vllm-moe

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="results/reroute_meaningful_${TIMESTAMP}"

echo "=== Reroute Validation with Meaningful Prompt ==="
echo "Job: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Output: $OUTDIR"
echo "Start: $(date)"

python scripts/reroute_meaningful_prompt.py \
    --model-path /data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B \
    --calibration-artifact results/reroute_impl_20260527/calibration/v2_calibration_smoke.pt \
    --output-dir "$OUTDIR" \
    --max-model-len 2048 \
    --seed 0 \
    --temperature 0.8 \
    --cpu-expert-pin-memory \
    --case-timeout-sec 1800

echo ""
echo "=== Done: $(date) ==="
echo "Results: $OUTDIR"
