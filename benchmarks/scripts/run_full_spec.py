#!/usr/bin/env python3
"""Comprehensive spec benchmark with detailed statistics."""
import subprocess, json, os, sys, time, argparse

SCRIPT = os.path.join(os.path.dirname(__file__), "../../examples/heterogeneous_benchmark_case.py")

def run_one(model, backend, slots, dist_port, outdir):
    out = os.path.join(outdir, f"spec_{backend}_{slots}.json")
    cmd = [
        sys.executable, SCRIPT,
        "--model-path", model, "--mode", "spec",
        "--slots-per-layer", str(slots),
        "--num-seqs", "1", "--input-len", "12", "--output-len", "6",
        "--max-num-batched-tokens", "256", "--max-num-seqs", "1", "--max-model-len", "128",
        "--gpu-memory-utilization", "0.85", "--max-draft-tokens", "4", "--draft-top-c", "128",
        "--cpu-expert-execution-enabled", "true", "--cpu-expert-backend", backend,
        "--cpu-expert-packed-min-routes", "1", "--cpu-gpu-parallel-execution-enabled", "auto",
        "--spec-profile", "true", "--engine-profile", "true", "--engine-profile-cuda-sync", "true",
        "--spec-enable-prefetch", "false", "--temperature", "0.0", "--seed", "0", "--enforce-eager", "false",
        "--return-token-ids", "true", "--return-text", "false", "--return-prompts", "false",
        "--dist-port", str(dist_port), "--output", out,
    ]
    print(f"  [{time.strftime('%H:%M:%S')}] backend={backend} slots={slots}")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                       cwd=os.path.join(os.path.dirname(__file__), "../.."))
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"  FAILED ({dt:.0f}s, exit={r.returncode})")
        if r.stderr:
            print(f"  stderr: {r.stderr[-300:]}")
        return None
    print(f"  OK ({dt:.0f}s)")
    return out


def extract_stats(json_path):
    with open(json_path) as f:
        d = json.load(f)
    ep = d.get("engine_profile", {})
    n = ep.get("model_moe_profile_count", 1) or 1
    return {
        "verify_ms": ep.get("model_run_verify_total_ms", 0),
        "cpu_compute_ms": ep.get("model_verify_cpu_compute_ms", 0),
        "cpu_merge_ms": ep.get("model_verify_cpu_to_gpu_merge_ms", 0),
        "gpu_moe_ms": ep.get("model_verify_gpu_compute_ms", 0),
        "plan_ms": ep.get("model_verify_plan_ms", 0),
        "route_ms": ep.get("model_verify_route_ms", 0),
        "cpu_prepare_ms": ep.get("model_verify_cpu_prepare_ms", 0),
        "tok_s": d.get("throughput_output_tok_s", 0),
        "digest": d.get("outputs_digest", "")[:16],
        # Per-layer means
        "act_experts_mean": ep.get("model_activated_expert_set_size_sum", 0) / n,
        "cpu_experts_mean": ep.get("model_realized_cpu_expert_count_sum", 0) / n,
        "cpu_route_ratio_mean": ep.get("model_cpu_route_ratio_sum", 0) / n,
        # Prefill/verify split
        "prefill_ms": d.get("engine_profile", {}).get("prefill_forward_ms", 0) if "engine_profile" in d else ep.get("prefill_forward_ms", 0),
        "verify_ms_from_spec": d.get("engine_profile", {}).get("verify_forward_ms", 0) if "engine_profile" in d else ep.get("verify_forward_ms", 0),
        "draft_ms": d.get("engine_profile", {}).get("draft_forward_ms", 0) if "engine_profile" in d else ep.get("draft_forward_ms", 0),
        "spec_layer_count": int(n),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/tmp/qwen3_model")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--slots", default="40,32,24,16,8")
    parser.add_argument("--backends", default="torch,fused")
    parser.add_argument("--dist-port", type=int, default=12345)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    slots = [int(x) for x in args.slots.split(",")]
    backends = args.backends.split(",")

    print(f"Model: {args.model}")
    print(f"Slots: {slots}")
    print(f"Backends: {backends}")
    print()

    results = []
    port = args.dist_port
    for s in slots:
        for b in backends:
            out = run_one(args.model, b, s, port, args.outdir)
            port += 1
            if out:
                stats = extract_stats(out)
                stats["backend"] = b
                stats["slots"] = s
                results.append(stats)
                print(f"    verify={stats['verify_ms']:.1f}ms cpu={stats['cpu_compute_ms']:.1f}ms "
                      f"act_exp={stats['act_experts_mean']:.1f} cpu_exp={stats['cpu_experts_mean']:.1f} "
                      f"cpu_ratio={stats['cpu_route_ratio_mean']:.3f}")

    # Summary table
    print(f"\n{'='*100}")
    print(f"{'backend':8s} {'slots':>5s} {'verify':>8s} {'cpu_ms':>8s} {'merge':>8s} {'gpu_ms':>8s} "
          f"{'act_exp':>8s} {'cpu_exp':>8s} {'cpu_ratio':>9s} {'tok_s':>7s}")
    print(f"{'-'*100}")
    for r in results:
        print(f"{r['backend']:8s} {r['slots']:5d} {r['verify_ms']:8.1f} {r['cpu_compute_ms']:8.1f} "
              f"{r['cpu_merge_ms']:8.1f} {r['gpu_moe_ms']:8.1f} "
              f"{r['act_experts_mean']:8.1f} {r['cpu_experts_mean']:8.1f} "
              f"{r['cpu_route_ratio_mean']:9.3f} {r['tok_s']:7.3f}")

    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {args.outdir}")


if __name__ == "__main__":
    main()
