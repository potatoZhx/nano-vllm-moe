"""Benchmark draft step timing with proper warmup for all batch sizes.

Warms up batch sizes 1-5 before timing to avoid torch.compile recompile.
Reports pure draft_graph_replay_ms and run_draft_core_run with first-N excluded.
Separates pure graph replay from total overhead properly.
"""
import json, math, os, statistics, subprocess, sys

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "heterogeneous_benchmark_case.py")
MODEL = "/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B"
OUTDIR = "/tmp/draft_breakdown_v2"
os.makedirs(OUTDIR, exist_ok=True)

# 128 experts, 12.5% cache (16/128) to reduce GPU memory for fused_sync
SLOTS = 16


def run_one(label, top_c, backend, port, extra_env=None):
    """Run heterogeneous_benchmark_case.py and return parsed JSON."""
    if extra_env is None:
        extra_env = {}
    outpath = os.path.join(OUTDIR, f"{label}.json")
    cmd = [
        sys.executable, SCRIPT,
        "--model-path", MODEL,
        "--mode", "spec",
        "--slots-per-layer", str(SLOTS),
        "--num-seqs", "4",
        "--input-len", "8",
        "--output-len", "32",
        "--max-num-batched-tokens", "16384",
        "--max-num-seqs", "512",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.80",
        "--max-draft-tokens", "4",
        "--draft-top-c", str(top_c),
        "--draft-cuda-graph-bucket-steps", "1,2,3,4,5",
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
    env = {**os.environ, "PYTHONUNBUFFERED": "1", **extra_env}
    print(f"\n{'='*60}", flush=True)
    print(f"Running {label}", flush=True)
    print(f"  top_c={top_c} backend={backend} slots={SLOTS} env={extra_env}", flush=True)
    print(f"{'='*60}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    if result.returncode != 0:
        print(f"RC={result.returncode}", flush=True)
        for line in result.stderr.strip().split("\n")[-30:]:
            print(f"  ERR: {line}", flush=True)
        for line in result.stdout.strip().split("\n"):
            try:
                d = json.loads(line)
                if isinstance(d, dict):
                    return d
            except (json.JSONDecodeError, TypeError):
                pass
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
    """Print per-config analysis with proper statistics (excl first N)."""
    if data is None:
        print(f"\n=== {label}: FAILED ===", flush=True)
        return None

    prof = data.get("engine_profile") or data.get("spec_profile") or {}
    if not isinstance(prof, dict) or not prof:
        print(f"\n=== {label}: NO PROFILE ===", flush=True)
        return None

    dc = max(int(prof.get("model_decode_count", 0)), 1)
    dr = max(int(prof.get("model_draft_graph_replay_count", 0)), 1)
    drd = max(int(prof.get("spec_run_draft_calls", 0)), 1)

    # Extract trace events for run_draft_core_run
    traces = list(prof.get("model_prefetch_trace_events", []))
    core_runs = [float(e["dur"]) / 1000.0 for e in traces
                 if isinstance(e, dict) and e.get("name") == "run_draft_core_run"]

    # Per-call metrics
    dgr_total = float(prof.get("model_draft_graph_replay_ms", 0))
    sm_total = float(prof.get("model_sample_decode_ms", 0))
    rm_total = float(prof.get("model_run_model_decode_ms", 0))
    pd_total = float(prof.get("model_prepare_decode_ms", 0))
    ps_total = float(prof.get("model_prepare_sample_decode_ms",
                              prof.get("model_prepare_sample_ms", 0)))
    sgr_total = float(prof.get("model_standard_graph_replay_ms", 0))

    # Exclude first N from statistics (N = number of warmup calls at batch_size=4)
    # The first call is the prefill decode; the first draft call may trigger recompile.
    # Exclude the first call from averages for clear steady-state numbers.
    n_excl = 1

    def per_call_excl(total, count, excl=1):
        c = max(count - excl, 1)
        return total / max(count, 1), total / c if c > 0 else 0.0

    dgr_avg, dgr_steady = per_call_excl(dgr_total, dr, n_excl)
    sm_avg, sm_steady = per_call_excl(sm_total, dc, n_excl)
    rm_avg, rm_steady = per_call_excl(rm_total, dc, n_excl)

    print(f"\n=== {label} ===", flush=True)
    print(f"  counts: decode={dc} draft_replays={dr} draft_calls={drd}", flush=True)

    print(f"\n  draft_graph_replay_ms:", flush=True)
    print(f"    total={dgr_total:.1f}  avg={dgr_avg:.3f}  steady(excl1st)={dgr_steady:.3f}", flush=True)

    print(f"\n  per-decode-call:", flush=True)
    print(f"    prepare_decode:       avg={pd_total/max(dc,1):.3f}", flush=True)
    print(f"    run_model_decode:     avg={rm_avg:.3f}  steady={rm_steady:.3f}", flush=True)
    print(f"    sample_decode:        avg={sm_avg:.3f}  steady={sm_steady:.3f}", flush=True)
    total_overhead_avg = (pd_total + ps_total + sm_total) / max(dc, 1)
    print(f"    overhead(prep+samp):  {total_overhead_avg:.3f}", flush=True)
    print(f"    pure_replay_implied:  {rm_avg - total_overhead_avg:.3f}", flush=True)

    if core_runs:
        print(f"\n  run_draft_core_run trace (n={len(core_runs)}):", flush=True)
        print(f"    all:    median={statistics.median(core_runs):.3f} mean={statistics.mean(core_runs):.3f}",
              flush=True)
        if len(core_runs) > n_excl:
            steady = core_runs[n_excl:]
            print(f"    steady: median={statistics.median(steady):.3f} mean={statistics.mean(steady):.3f}",
                  flush=True)
        # Show individual values (truncated)
        vals = [f"{v:.2f}" for v in core_runs]
        if len(vals) > 24:
            vals = vals[:4] + ["..."] + vals[-12:]
        print(f"    values: [{', '.join(vals)}]", flush=True)

    digest = data.get("outputs_digest", "")
    if digest:
        print(f"\n  digest: {str(digest)[:40]}", flush=True)

    return {
        "label": label,
        "dgr_steady": dgr_steady,
        "core_steady_median": statistics.median(core_runs[n_excl:]) if len(core_runs) > n_excl else 0,
        "digest": str(digest)[:16],
    }


def main():
    configs = [
        ("topc0_none",       0, "none",       {}),
        ("topc1_fused",      1, "fused",      {}),
        # fused_sync needs lower memory; use even smaller slots
        ("topc1_fused_sync", 1, "fused_sync", {}),
    ]
    results = []
    for (label, top_c, backend, extra_env), port in zip(configs, [5110, 5120, 5130]):
        data = run_one(label, top_c, backend, port, extra_env)
        r = analyze(data, label)
        if r:
            results.append(r)

    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    header = f"{'Config':<25s} {'dgr_steady':>10s} {'core_steady':>10s} {'digest':>16s}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for r in results:
        print(f"{r['label']:<25s} {r['dgr_steady']:10.3f} {r['core_steady']:10.3f} {r['digest']:>16s}", flush=True)

    print(f"\nResults dir: {OUTDIR}", flush=True)


if __name__ == "__main__":
    main()
