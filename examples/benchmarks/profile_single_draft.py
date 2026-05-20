"""Profile single draft forward with full timing breakdown.

Tests both top_c=0 (baseline) and fused_sync with num_seqs=1.
bucket_steps=[1] — minimal config, single request never needs bs>1 in draft.
"""
import json, os, statistics, subprocess, sys

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "heterogeneous_benchmark_case.py")
MODEL = "/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B"
OUTDIR = "/home/mumura/moe_spec/tmp/single_draft_profile"
os.makedirs(OUTDIR, exist_ok=True)
SLOTS = 16


def run_one(label, top_c, backend, extra_args=None):
    outpath = os.path.join(OUTDIR, f"{label}.json")
    cmd = [
        sys.executable, SCRIPT,
        "--model-path", MODEL,
        "--mode", "spec",
        "--slots-per-layer", str(SLOTS),
        "--num-seqs", "1",              # single request
        "--input-len", "32",
        "--output-len", "16",
        "--max-num-batched-tokens", "16384",
        "--max-num-seqs", "512",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.80",
        "--max-draft-tokens", "4",
        "--draft-top-c", str(top_c),
        "--draft-cuda-graph-bucket-steps", "1",
        "--cpu-expert-execution-enabled", "true",
        "--cpu-expert-backend", "fused",
        "--cpu-expert-workspace-max-routes", "262144",
        "--draft-cuda-graph-cpu-backend", backend,
        "--enforce-eager", "false",
        "--spec-enable-prefetch", "false",
        "--engine-profile", "true",
        "--engine-profile-cuda-sync", "true",
        "--seed", "42",
        "--temperature", "0.0",
        "--output", outpath,
        "--dist-port", "5100",
    ]
    if extra_args:
        cmd.extend(extra_args)
    print(f"\n{'='*60}", flush=True)
    print(f"Running {label}: top_c={top_c} backend={backend}", flush=True)
    print(f"{'='*60}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"})
    if result.returncode != 0:
        print(f"RC={result.returncode}", flush=True)
        for line in result.stderr.strip().split("\n")[-20:]:
            print(f"  ERR: {line}", flush=True)
        return None
    data = None
    for line in result.stdout.strip().split("\n"):
        try:
            d = json.loads(line)
            if isinstance(d, dict):
                data = d
        except (json.JSONDecodeError, TypeError):
            pass
    if data is None and os.path.exists(outpath):
        with open(outpath) as f:
            data = json.load(f)
    return data


def analyze(data, label):
    if data is None:
        print(f"\n=== {label}: FAILED ===", flush=True)
        return None

    prof = data.get("engine_profile") or {}
    if not prof:
        print(f"\n=== {label}: NO PROFILE ===", flush=True)
        return None

    dc = max(int(prof.get("model_decode_count", 0)), 1)
    dr = max(int(prof.get("model_draft_graph_replay_count", 0)), 1)
    drd = max(int(prof.get("spec_run_draft_calls", 0)), 1)
    dgrs = max(int(prof.get("model_run_draft_count", 0)), 1)

    n_excl = 1  # exclude first call

    def avg_excl(total_key, count_key=None, count=None):
        c = count if count is not None else max(int(prof.get(count_key, 0)), 1)
        total = float(prof.get(total_key, 0))
        c_steady = max(c - n_excl, 1)
        return total / max(c, 1), total / c_steady if c_steady > 0 else 0

    print(f"\n=== {label} ===", flush=True)
    print(f"  counts: decode={dc} draft_replays={dr} draft_calls={drd} run_draft={dgrs}", flush=True)

    dgr_total = float(prof.get("model_draft_graph_replay_ms", 0))
    dgr_avg = dgr_total / max(dr, 1)
    # steady = exclude first call (handled by warmup now, but still)
    print(f"\n  --- Core Timing (ms) ---", flush=True)
    print(f"  draft_graph_replay:      total={dgr_total:.1f}  avg/call={dgr_avg:.3f}", flush=True)

    # run_model_decode breakdown
    rmd_total = float(prof.get("model_run_model_decode_ms", 0))
    print(f"  run_model_decode:        avg={rmd_total/max(dc,1):.3f}", flush=True)

    # prepare
    pd_avg, _ = avg_excl("model_prepare_decode_ms", count=dc)
    ps_avg, _ = avg_excl("model_prepare_sample_decode_ms", count=dc)
    print(f"  prepare_decode:          avg={pd_avg:.3f}", flush=True)
    print(f"  prepare_sample:          avg={ps_avg:.3f}", flush=True)

    # sample
    sm_avg, _ = avg_excl("model_sample_decode_ms", count=dc)
    print(f"  sample_decode:           avg={sm_avg:.3f}", flush=True)

    # Standard graph replay for comparison
    sgr_total = float(prof.get("model_standard_graph_replay_ms", 0))
    sgc = max(int(prof.get("model_standard_graph_replay_count", 1)), 1)
    print(f"  standard_graph_replay:   avg={sgr_total/max(sgc,1):.3f}", flush=True)

    # run_draft_core_run trace median
    traces = list(prof.get("model_prefetch_trace_events", []))
    core_runs = [float(e["dur"]) / 1000.0 for e in traces
                 if isinstance(e, dict) and e.get("name") == "run_draft_core_run"]
    if core_runs:
        print(f"\n  --- run_draft_core_run trace (n={len(core_runs)}) ---", flush=True)
        print(f"    all:    median={statistics.median(core_runs):.3f} mean={statistics.mean(core_runs):.3f}",
              flush=True)
        if len(core_runs) > n_excl:
            steady = core_runs[n_excl:]
            print(f"    steady: median={statistics.median(steady):.3f} mean={statistics.mean(steady):.3f}",
                  flush=True)
        vals = [f"{v:.2f}" for v in core_runs]
        if len(vals) > 24:
            vals = vals[:4] + ["..."] + vals[-12:]
        print(f"    values: [{', '.join(vals)}]", flush=True)

    # draft-specific keys
    print(f"\n  --- Spec Draft ---", flush=True)
    for k in ["spec_draft_forward_ms", "spec_draft_loop_ms", "spec_run_draft_infer_ms_total",
              "spec_run_draft_calls", "spec_draft_steps_total", "spec_accept_ms",
              "model_run_draft_mode_set_ms", "model_run_draft_total_ms"]:
        v = prof.get(k)
        if v is not None:
            print(f"    {k}: {v}", flush=True)

    # MoE profile: CPU vs GPU
    print(f"\n  --- MoE Per-Call ---", flush=True)
    mc = max(int(prof.get("model_moe_profile_count", 1)), 1)
    for k in ["model_gpu_compute_ms", "model_cpu_compute_ms", "model_cpu_prepare_ms",
              "model_cpu_to_gpu_merge_ms", "model_route_ms", "model_scatter_ms",
              "model_gpu_gather_ms", "model_plan_ms",
              "model_parallel_wall_ms", "model_parallel_critical_path_est_ms"]:
        v = prof.get(k)
        if v is not None:
            print(f"    {k}: total={v:.1f}  avg={v/max(mc,1):.3f}", flush=True)

    digest = data.get("outputs_digest", "")
    if digest:
        print(f"\n  digest: {str(digest)[:40]}", flush=True)

    return {
        "label": label,
        "dgr_avg": dgr_avg,
        "dgr_total": dgr_total,
        "digest": str(digest)[:16],
    }


def main():
    results = []
    for label, top_c, backend in [
        ("topc0_none", 0, "none"),
        ("topc1_fused_sync", 1, "fused_sync"),
    ]:
        data = run_one(label, top_c, backend)
        r = analyze(data, label)
        if r:
            results.append(r)

    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Config':<25s} {'dgr_avg(ms)':>12s} {'dgr_total':>10s} {'digest':>16s}", flush=True)
    print("-" * 70, flush=True)
    for r in results:
        print(f"{r['label']:<25s} {r['dgr_avg']:12.3f} {r['dgr_total']:10.1f} {r['digest']:>16s}", flush=True)

    print(f"\nResults dir: {OUTDIR}", flush=True)


if __name__ == "__main__":
    main()
