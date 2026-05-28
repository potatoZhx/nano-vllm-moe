#!/usr/bin/env bash
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --job-name=pre_exps
#SBATCH --output=/home/mumura/moe_spec/logs/pre_exps_%j.out
#SBATCH --error=/home/mumura/moe_spec/logs/pre_exps_%j.err

set -eo pipefail

MODEL_PATH="/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B"
DATA_FILE="/home/mumura/moe_spec/nano-vllm-moe/pre_exps/wikitext2_test.txt"
SCRIPT_DIR="/home/mumura/moe_spec/nano-vllm-moe/pre_exps"
RESULTS_BASE="/home/mumura/moe_spec/nano-vllm-moe/results"

echo "=== Experiment run start $(date -Is) ==="
echo "HOST=$(hostname)"
echo "SLURM_JOBID=${SLURM_JOB_ID}"

source ~/.bashrc
conda activate nano_vllm_env

echo "CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"
echo "python=$(which python)"
python -V

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

cd /home/mumura/moe_spec/nano-vllm-moe

# Verify environment
python -c "
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('cuda_device_count', torch.cuda.device_count())
if torch.cuda.is_available():
    print('gpu_name', torch.cuda.get_device_name(0))
    print('gpu_mem_gb', torch.cuda.get_device_properties(0).total_memory / 1e9)
"

echo "=== Environment OK ==="

# Common args
COMMON_ARGS="--model $MODEL_PATH --data_file $DATA_FILE --device cuda"

# ── E1: Cache Dual-Objective Evaluation ──────────────────────────────────────
echo ""
echo "============================================================"
echo "  E1: Cache Dual-Objective Evaluation"
echo "============================================================"
echo "Started at $(date)"

python "$SCRIPT_DIR/cache_dual_objective_eval.py" \
    $COMMON_ARGS \
    --cache_ratios 0.75 0.50 0.25 \
    --n_calib 8 --n_eval 16 \
    --outdir "${RESULTS_BASE}/results_e1" \
    2>&1
echo "E1 completed at $(date)"

# ── E2: Top-1/2 Protection Evaluation ────────────────────────────────────────
echo ""
echo "============================================================"
echo "  E2: Top-1/2 Protection Evaluation"
echo "============================================================"
echo "Started at $(date)"

python "$SCRIPT_DIR/top12_protection_eval.py" \
    $COMMON_ARGS \
    --cache_ratios 0.75 0.50 0.25 \
    --w_protects 0.10 0.15 0.20 \
    --n_calib 8 --n_eval 8 \
    --draft_len 6 --prompt_len 128 \
    --outdir "${RESULTS_BASE}/results_e2" \
    2>&1
echo "E2 completed at $(date)"

# ── E3: Dynamic K Analysis ───────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  E3: Dynamic K Analysis"
echo "============================================================"
echo "Started at $(date)"

python "$SCRIPT_DIR/dynamic_k_analysis.py" \
    $COMMON_ARGS \
    --cache_ratios 0.75 0.50 0.25 \
    --k_max 12 --n_calib 8 --n_eval 16 \
    --prompt_len 128 \
    --outdir "${RESULTS_BASE}/results_e3" \
    2>&1
echo "E3 completed at $(date)"

# ── E4: Prefetch Coverage Analysis ───────────────────────────────────────────
echo ""
echo "============================================================"
echo "  E4: Prefetch Coverage Analysis"
echo "============================================================"
echo "Started at $(date)"

python "$SCRIPT_DIR/prefetch_coverage_analysis.py" \
    $COMMON_ARGS \
    --cache_ratios 0.75 0.50 0.25 \
    --prefetch_rates 0 1 2 3 4 \
    --k_max 12 --n_calib 8 --n_eval 16 \
    --prompt_len 128 \
    --outdir "${RESULTS_BASE}/results_e4" \
    2>&1
echo "E4 completed at $(date)"

# ── E5: Alpha Prediction Evaluation ──────────────────────────────────────────
echo ""
echo "============================================================"
echo "  E5: Alpha Prediction Evaluation"
echo "============================================================"
echo "Started at $(date)"

python "$SCRIPT_DIR/alpha_prediction_eval.py" \
    $COMMON_ARGS \
    --cache_ratios 0.75 0.50 0.25 \
    --k_max 12 --n_calib 8 --n_eval 16 \
    --prompt_len 128 \
    --outdir "${RESULTS_BASE}/results_e5" \
    2>&1
echo "E5 completed at $(date)"

echo ""
echo "=== All experiments completed $(date -Is) ==="
