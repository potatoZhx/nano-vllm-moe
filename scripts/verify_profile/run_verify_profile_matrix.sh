#!/usr/bin/env bash
set -euo pipefail

JOB_ID="${1:-29309}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_ROOT="${LOG_ROOT:-/home/mumura/moe_spec/logs}"
mkdir -p "${LOG_ROOT}"
LOG="${LOG_ROOT}/verify_breakdown_job${JOB_ID}_run_$(date +%Y%m%d_%H%M%S).log"

export JOB_ID
export REPO_ROOT

srun --jobid="${JOB_ID}" bash <<'REMOTE' 2>&1 | tee "${LOG}"
source ~/.bashrc
conda activate nano_moe
set -eo pipefail

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT="${REPO_ROOT}/results/verify_breakdown_job${JOB_ID}_${TS}"
PROMPT="${REPO_ROOT}/results/predictive_prefetch_validation_direct_20260602_003130/meaningful_prompt.txt"
mkdir -p "${OUT}"

printf "timestamp=%s\n" "$(date "+%F %T %Z")"
printf "job_id=%s\n" "${JOB_ID}"
printf "host=%s\n" "$(hostname)"
printf "pwd=%s\n" "$(pwd)"
printf "git_sha=%s\n" "$(git rev-parse HEAD)"
printf "git_status_short=%s\n" "$(git status --short | wc -l)"
printf "conda_env=%s\n" "${CONDA_DEFAULT_ENV}"
printf "CUDA_VISIBLE_DEVICES=%s\n" "${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
printf "output_dir=%s\n" "${OUT}"

COMMON=(
  --single-case
  --model-path /data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B
  --prompt-text-file "${PROMPT}"
  --cache-ratio 0.75
  --slots-per-layer 0
  --prefetch-enabled true
  --prefetch-runtime-mode draft_segment_indexed
  --prefetch-runtime-kind legacy
  --prefetch-verify-attention-ratio 0.3
  --predictive-phase1-budget 4
  --num-seqs 1
  --input-len 1
  --output-len 512
  --max-draft-tokens 8
  --draft-top-c 0
  --draft-reroute-policy entropy_cache_bias
  --draft-reroute-artifact results/reroute_impl_20260531/offline_profile_20260531_203257.safetensors
  --temperature 0.8
  --acceptance-strategy standard_sampling
  --acceptance-threshold 0.7
  --spec-verify-miss-policy cache_fill
  --cache-strategy lru
  --cpu-expert-backend fused
  --cpu-expert-pin-memory true
  --cpu-expert-workspace-max-routes 16384
  --cpu-expert-packed-min-routes 1
  --cpu-expert-parallel-mode serial
  --cpu-expert-num-threads 4
  --cpu-gpu-parallel-execution-enabled auto
  --cpu-gpu-parallel-min-cpu-route-ratio 0.0
  --max-num-batched-tokens 16384
  --max-num-seqs 1
  --max-model-len 2048
  --gpu-memory-utilization 0.90
  --enforce-eager false
  --draft-cuda-graph-enabled true
  --draft-cuda-graph-cpu-backend none
  --prefetch-verify-wait-ms 0.0
  --prefetch-step-budget 4
  --prefetch-max-inflight 8
  --prefetch-staging-slots-per-layer 2
  --cache-eviction-budget-per-step 2
  --prefetch-global-queue-capacity 4096
  --prefetch-use-prefill-history true
  --prefetch-use-verify-history true
  --prefetch-use-draft-live true
  --seed 0
  --case-timeout-sec 2400
)

run_probe() {
  local name="$1"
  local port="$2"
  local sync_flag="$3"
  printf "[%s] probe %s start sync_layer_timing=%s\n" "$(date +%T)" "${name}" "${sync_flag}"
  python benchmarks/scripts/spec_verify_expert_count_stats.py \
    "${COMMON[@]}" \
    --output "${OUT}/${name}.json" \
    --dist-port "${port}" \
    --sync-layer-timing "${sync_flag}" \
    > "${OUT}/${name}.stdout" 2>&1
  local status=$?
  printf "[%s] probe %s exit=%s\n" "$(date +%T)" "${name}" "${status}"
  tail -n 40 "${OUT}/${name}.stdout"
  return "${status}"
}

run_direct() {
  local name="$1"
  local port="$2"
  local prefetch_enabled="$3"
  local verify_layer_enabled="$4"
  local miss_policy="$5"
  local torch_profile="$6"
  printf "[%s] direct %s start prefetch=%s verify_layer=%s miss_policy=%s torch_profile=%s\n" \
    "$(date +%T)" "${name}" "${prefetch_enabled}" "${verify_layer_enabled}" "${miss_policy}" "${torch_profile}"
  python scripts/verify_profile/run_direct_verify_case.py \
    --name "${name}" \
    --output-dir "${OUT}" \
    --prompt-text-file "${PROMPT}" \
    --dist-port "${port}" \
    --prefetch-enabled "${prefetch_enabled}" \
    --prefetch-verify-layer-enabled "${verify_layer_enabled}" \
    --spec-verify-miss-policy "${miss_policy}" \
    --torch-profile "${torch_profile}" \
    > "${OUT}/${name}.stdout" 2>&1
  local status=$?
  printf "[%s] direct %s exit=%s\n" "$(date +%T)" "${name}" "${status}"
  tail -n 80 "${OUT}/${name}.stdout"
  return "${status}"
}

run_probe probe_sync 19931 true
run_probe probe_nosync 19932 false
run_direct direct_prefetch_vlayer_on 19941 true true cache_fill false
run_direct direct_cache_fill_no_cpu 19946 true true cache_fill_no_cpu false
run_direct direct_prefetch_vlayer_off 19942 true false cache_fill false
run_direct direct_prefetch_off 19943 false false cache_fill false
run_direct direct_cpu_policy_prefetch_off 19944 false false cpu false
run_direct direct_torchprof_l512 19945 true true cache_fill true

python scripts/verify_profile/summarize_verify_profile.py "${OUT}" --output "${OUT}/summary_table.md"
if [ -f "${OUT}/direct_torchprof_l512_torch_profile/verify_forward_rank0.json" ]; then
  python scripts/verify_profile/parse_torch_trace.py \
    "${OUT}/direct_torchprof_l512_torch_profile/verify_forward_rank0.json" \
    --output "${OUT}/torch_trace_summary.md"
fi

printf "FINAL_OUTPUT_DIR=%s\n" "${OUT}"
REMOTE

printf 'LOG=%s\n' "${LOG}"
