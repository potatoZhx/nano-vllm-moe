#!/usr/bin/env bash
set -eo pipefail

source ~/.bashrc >/dev/null 2>&1 || true
eval "$(conda shell.bash hook)"
conda activate nano_moe

cd /home/mumura/moe_spec/nano-vllm-moe
export PYTHONPATH=.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
OUTDIR="${OUTDIR:-/home/mumura/moe_spec/nano-vllm-moe/benchmarks/results/topc_graph_matrix_${RUN_TS}}"
LOG="${LOG:-/home/mumura/moe_spec/logs/job${SLURM_JOB_ID:-unknown}_topc_graph_matrix_${RUN_TS}.log}"
DIST_PORT_BASE="${DIST_PORT_BASE:-$((41000 + (RANDOM % 10000)))}"
mkdir -p "$OUTDIR" /home/mumura/moe_spec/logs

run_case() {
  local top_c="$1"
  local backend="$2"
  local port="$3"
  local out="$OUTDIR/spec_topc${top_c}_graph.json"

  echo "CASE_START top_c=${top_c} backend=${backend} out=${out}"
  python examples/heterogeneous_benchmark_case.py \
    --model-path /data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B \
    --mode spec \
    --slots-per-layer 64 \
    --num-seqs 4 \
    --input-len 32 \
    --output-len 8 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 8 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.85 \
    --max-draft-tokens 4 \
    --draft-top-c "$top_c" \
    --draft-cuda-graph-bucket-steps 1,2,4 \
    --cpu-expert-execution-enabled true \
    --cpu-expert-backend fused \
    --draft-cuda-graph-cpu-backend "$backend" \
    --cpu-expert-workspace-max-routes 32768 \
    --cpu-expert-packed-min-routes 1 \
    --cpu-expert-parallel-mode serial \
    --cpu-expert-num-threads 4 \
    --cpu-gpu-parallel-execution-enabled off \
    --spec-enable-prefetch false \
    --enforce-eager false \
    --engine-profile true \
    --engine-profile-cuda-sync true \
    --return-token-ids true \
    --return-text false \
    --return-prompts false \
    --dist-port "$port" \
    --output "$out"
  echo "CASE_DONE top_c=${top_c}"
}

{
  echo "LOG=$LOG"
  echo "OUTDIR=$OUTDIR"
  echo "DIST_PORT_BASE=$DIST_PORT_BASE"
  echo "HOST=$(hostname)"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

  run_case 0 none "$DIST_PORT_BASE"
  run_case 1 fused "$((DIST_PORT_BASE + 1))"
  run_case 2 fused "$((DIST_PORT_BASE + 2))"

  python - "$OUTDIR" <<'PY'
import json
import sys
import statistics
from pathlib import Path

outdir = Path(sys.argv[1])
rows = []
base = None
for top_c in (0, 1, 2):
    path = outdir / f"spec_topc{top_c}_graph.json"
    data = json.loads(path.read_text())
    profile = data.get("engine_profile") or {}
    draft_calls = float(profile.get("spec_run_draft_calls", profile.get("model_run_draft_count", 0)) or 0)
    graph_replays = float(profile.get("model_draft_graph_replay_count", 0) or 0)
    draft_ms_total = float(profile.get("spec_run_draft_infer_ms_total", 0) or 0)
    trace_ms = [
        float(event.get("dur", 0.0)) / 1000.0
        for event in profile.get("model_prefetch_trace_events", [])
        if event.get("name") == "run_draft_core_run"
    ]
    steady_trace_ms = trace_ms[1:] if len(trace_ms) > 1 else trace_ms
    row = {
        "top_c": top_c,
        "path": str(path),
        "digest": data.get("outputs_digest"),
        "generated_token_ids": data.get("generated_token_ids"),
        "draft_calls": draft_calls,
        "graph_replays": graph_replays,
        "draft_ms_total": draft_ms_total,
        "draft_ms_per_call": draft_ms_total / draft_calls if draft_calls else 0.0,
        "draft_core_trace_ms": trace_ms,
        "draft_core_trace_median_excl_first_ms": (
            statistics.median(steady_trace_ms) if steady_trace_ms else 0.0
        ),
        "draft_core_trace_mean_excl_first_ms": (
            statistics.mean(steady_trace_ms) if steady_trace_ms else 0.0
        ),
        "graph_hit_rate": profile.get("model_graph_hit_rate", profile.get("graph_hit_rate")),
        "draft_steps_per_step": profile.get("spec_draft_steps_per_step"),
        "cpu_graph_async_count": profile.get("model_cpu_graph_async_count", profile.get("cpu_graph_async_count")),
    }
    if base is None:
        base = row
    row["tokens_match_topc0"] = row["generated_token_ids"] == base["generated_token_ids"]
    row["digest_match_topc0"] = row["digest"] == base["digest"]
    rows.append(row)

summary_path = outdir / "summary.json"
summary_path.write_text(json.dumps(rows, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
print(json.dumps(rows, ensure_ascii=True, indent=2))
print(f"SUMMARY={summary_path}")
PY
} 2>&1 | tee "$LOG"
