#!/bin/bash
#SBATCH --partition=A100
#SBATCH --gres=gpu:A100:1
#SBATCH --time=02:00:00
#SBATCH --job-name=prec_fix
#SBATCH --output=/home/mumura/moe_spec/logs/prec_fix_%j.log

source ~/.bashrc
conda activate nano_moe
cd /home/mumura/moe_spec/nano-vllm-moe

# Test with GPU-only fallback (--no-cpu-exec)
python scripts/precision_validate.py \
    --model-path /data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B \
    --calibration-artifact results/reroute_impl_20260527/calibration/v2_calibration_smoke.pt \
    --output-dir results/precision_fix_gpuonly \
    --output-len 128 \
    --max-model-len 2048 \
    --only-policy round_robin \
    --only-ratio 0.25 \
    --cpu-expert-pin-memory \
    --no-cpu-exec \
    --enforce-eager \
    --case-timeout-sec 1800
