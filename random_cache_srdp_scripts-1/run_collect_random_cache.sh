#!/bin/bash
#SBATCH -J rc_accept_collect
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -p A800
#SBATCH -w gpu5
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

echo "Running on host: $(hostname)"
echo "Starting time: $(date)"

module load cuda/12.9.1 || true
module load Anaconda3/2025.06 || true

cd ~/MOE_SD
source .venv/bin/activate
mkdir -p logs
export OMP_NUM_THREADS=4

# Train split example: Wiki random_cache.
python -u random_cache_srdp/collect_random_cache_acceptance.py \
  --dataset wiki \
  --model-path /data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B \
  --wiki-jsonl /data2/group_谈海生/mumura/dynamick/predictor/filtered_wikitext/train_articles_qwen3.jsonl \
  --output-dir /data2/group_谈海生/mumura/dynamick/predictor/random_cache_runs \
  --cache-policy lfu \
  --cache-ratio 0.5 \
  --cache-topc-ratio 0.5 \
  --decode-steps 20 \
  --min-prefill-n 8 \
  --max-prefill-n 1024 \
  --reserve-tokens 5 \
  --max-samples 300

# Test split example: MTBench random_cache. Uncomment when collecting test data.
# python -u random_cache_srdp/collect_random_cache_acceptance.py \
#   --dataset mtbench \
#   --model-path /data2/group_谈海生/lagin/models/Qwen3-30B-A3B-Base \
#   --output-dir /data2/group_谈海生/lagin/data/Sd_Data/data/random_cache_runs \
#   --cache-policy lfu \
#   --cache-ratio 0.5 \
#   --cache-topc-ratio 0.5 \
#   --decode-steps 20 \
#   --min-prefill-n 8 \
#   --max-prefill-n 1024 \
#   --max-samples 300

echo "Finished at: $(date)"
