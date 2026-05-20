"""Single-seq fused_sync draft forward profiling.

Runs fused_sync with num_seqs=1, bucket_steps=[1], mem_util=0.75.
Reports all timing components.
"""
import json, os, statistics, subprocess, sys

MODEL = "/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B"
OUTDIR = "/home/mumura/moe_spec/tmp/single_draft_profile"
os.makedirs(OUTDIR, exist_ok=True)
BENCH = os.path.join(os.path.dirname(__file__), "..", "heterogeneous_benchmark_case.py")


def run_one(label, top_c, backend, mem_util, bucket_steps, num_seqs, port):
    outpath = os.path.join(OUTDIR, f"{label}.json")
    cmd = [
        sys.executable, BENCH,
        "--model-path", MODEL,
        "--mode", "spec",
        "--slots-per-layer", "16",
        "--num-seqs", str(num_seqs),
        "--input-len", "32",
        "--output-len", "16",
        "--max-num-batched-tokens", "16384",
        "--max-num-seqs", "512",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", str(mem_util),
        "--max-draft-tokens", "4",
        "--draft-top-c", str(top_c),
        "--draft-cuda-graph-bucket-steps", bucket_steps,
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
        "--dist-port", str(port),
    ]
    print(f"Running {label} (top_c={top_c} backend={backend} mem={mem_util} bs={bucket_steps})", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"})
    if result.returncode != 0:
        print(f"RC={result.returncode}", flush=True)
        for line in result.stderr.strip().split("\n")[-15:]:
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

    n_excl = 1

    print(f"\n=== {label} ===", flush=True)
    print(f"  counts: decode={dc} draft_replays={dr} draft_calls={drd} run_draft={dgrs}", flush=True)

    dgr_total = float(prof.get("model_draft_graph_replay_ms", 0))
    print(f"\n  --- Draft Step Timing (ms, per call) ---", flush=True)
    print(f"  draft_graph_replay:       total={dgr_total:.1f}  avg={dgr_total/max(dr,1):.3f}", flush=True)

    rdc_total = float(prof.get("model_run_draft_core_run_ms", 0))
    rmd_total = float(prof.get("model_run_model_decode_ms", 0))
    sm_total = float(prof.get("model_sample_decode_ms", 0))
    pd_total = float(prof.get("model_prepare_decode_ms", 0))
    ps_total = float(prof.get("model_prepare_sample_decode_ms", 0))
    mdms_total = float(prof.get("model_run_draft_mode_set_ms", 0))
    rdt_total = float(prof.get("model_run_draft_total_ms", 0))

    print(f"  run_draft_core_run:       total={rdc_total:.1f}  avg={rdc_total/max(dc,1):.3f}", flush=True)
    print(f"  run_model_decode:         avg={rmd_total/max(dc,1):.3f}", flush=True)
    print(f"  sample_decode:            avg={sm_total/max(dc,1):.3f}", flush=True)
    print(f"  prepare_decode:           avg={pd_total/max(dc,1):.3f}", flush=True)
    print(f"  prepare_sample:           avg={ps_total/max(dc,1):.3f}", flush=True)
    print(f"  mode_set:                 avg={mdms_total/max(dc,1):.3f}", flush=True)
    print(f"  run_draft_total:          avg={rdt_total/max(dc,1):.3f}", flush=True)

    # Trace events
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
        vals = [f"{v:.1f}" for v in core_runs]
        if len(vals) > 24:
            vals = vals[:4] + ["..."] + vals[-12:]
        print(f"    values: [{', '.join(vals)}]", flush=True)

    # MoE profile
    mc = max(int(prof.get("model_moe_profile_count", 1)), 1)
    print(f"\n  --- MoE Per Call (avg over {mc} profiles) ---", flush=True)
    for k in ["model_gpu_compute_ms", "model_cpu_compute_ms", "model_cpu_prepare_ms",
              "model_cpu_to_gpu_merge_ms", "model_route_ms", "model_scatter_ms",
              "model_gpu_gather_ms", "model_plan_ms",
              "model_parallel_wall_ms", "model_parallel_critical_path_est_ms"]:
        v = prof.get(k)
        if v is not None:
            print(f"    {k}: total={float(v):.1f}  avg={float(v)/max(mc,1):.3f}", flush=True)

    # Spec draft specifics
    print(f"\n  --- Spec Draft Higher-Level ---", flush=True)
    for k in ["spec_draft_forward_ms", "spec_draft_loop_ms", "spec_run_draft_infer_ms_total",
              "spec_accept_ms", "spec_accepted_tokens_total", "spec_draft_tokens_total"]:
        v = prof.get(k)
        if v is not None:
            print(f"    {k}: {v}", flush=True)

    digest = data.get("outputs_digest", "")
    if digest:
        print(f"\n  digest: {str(digest)[:40]}", flush=True)

    return {
        "label": label,
        "dgr_avg": dgr_total / max(dr, 1),
        "core_median": statistics.median(core_runs[1:]) if len(core_runs) > 1 else 0,
        "digest": str(digest)[:16],
    }


def main():
    results = []

    # Test 1: top_c=0 baseline (single-seq)
    data = run_one("topc0_none_1seq", 0, "none", 0.80, "1", 1, 5100)
    r = analyze(data, "topc0_none_1seq")
    if r:
        results.append(r)

    # Test 2: fused_sync (single-seq, lower mem to avoid OOM)
    data = run_one("fused_sync_1seq", 1, "fused_sync", 0.75, "1", 1, 5110)
    r = analyze(data, "fused_sync_1seq")
    if r:
        results.append(r)

    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Config':<25s} {'dgr_avg':>10s} {'core_med':>10s} {'digest':>16s}", flush=True)
    print("-" * 65, flush=True)
    for r in results:
        print(f"{r['label']:<25s} {r['dgr_avg']:10.3f} {r['core_med']:10.3f} {r['digest']:>16s}", flush=True)

    print(f"\nResults dir: {OUTDIR}", flush=True)


if __name__ == "__main__":
    main()
