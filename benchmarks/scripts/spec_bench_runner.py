#!/usr/bin/env python3
"""Run spec benchmarks and output results as JSON."""
import sys, json, os, time, argparse
import torch

def run_one_case(model_path, backend, slots, dist_port, output_path):
    """Run one spec benchmark case."""
    # Use subprocess to get a clean process
    import subprocess
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "../../examples/heterogeneous_benchmark_case.py"),
        "--model-path", model_path,
        "--mode", "spec",
        "--slots-per-layer", str(slots),
        "--num-seqs", "1",
        "--input-len", "12",
        "--output-len", "6",
        "--max-num-batched-tokens", "256",
        "--max-num-seqs", "1",
        "--max-model-len", "256",
        "--gpu-memory-utilization", "0.85",
        "--max-draft-tokens", "4",
        "--draft-top-c", "128",
        "--cpu-expert-execution-enabled", "true",
        "--cpu-expert-backend", backend,
        "--cpu-expert-packed-min-routes", "1",
        "--cpu-gpu-parallel-execution-enabled", "auto",
        "--spec-profile", "true",
        "--engine-profile", "true",
        "--engine-profile-cuda-sync", "true",
        "--spec-enable-prefetch", "false",
        "--temperature", "0.0",
        "--seed", "0",
        "--enforce-eager", "false",
        "--return-token-ids", "true",
        "--return-text", "false",
        "--return-prompts", "false",
        "--dist-port", str(dist_port),
        "--output", output_path,
    ]
    print(f"  Running: backend={backend} slots={slots}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200,
                            cwd=os.path.join(os.path.dirname(__file__), "../.."))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  FAILED (exit={result.returncode}, {elapsed:.0f}s)")
        print(f"  stderr: {result.stderr[-500:]}")
        return None

    print(f"  Done ({elapsed:.0f}s)")

    # Parse output JSON
    if os.path.exists(output_path):
        with open(output_path) as f:
            data = json.load(f)
        ep = data.get("engine_profile", {})
        return {
            "backend": backend,
            "slots": slots,
            "ratio": slots / 128.0,
            "verify_ms": ep.get("model_run_verify_total_ms", 0),
            "cpu_compute_ms": ep.get("model_verify_cpu_compute_ms", 0),
            "cpu_merge_ms": ep.get("model_verify_cpu_to_gpu_merge_ms", 0),
            "gpu_moe_ms": ep.get("model_verify_gpu_compute_ms", 0),
            "plan_ms": ep.get("model_verify_plan_ms", 0),
            "tok_s": data.get("throughput_output_tok_s", 0),
            "digest": data.get("outputs_digest", "")[:16],
            "elapsed_sec": elapsed,
        }
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--slots", default="96,64,32", help="comma-separated slot counts")
    parser.add_argument("--backends", default="torch,fused", help="comma-separated backends")
    parser.add_argument("--dist-port", type=int, default=12345)
    args = parser.parse_args()

    slots_list = [int(x) for x in args.slots.split(",")]
    backends = args.backends.split(",")
    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    print(f"=== Spec Benchmark Runner ===")
    print(f"Model: {args.model_path}")
    print(f"Slots: {slots_list}")
    print(f"Backends: {backends}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    for slots in slots_list:
        for backend in backends:
            out = os.path.join(args.output_dir, f"spec_{backend}_{slots}.json")
            dist_port = args.dist_port + len(results)
            r = run_one_case(args.model_path, backend, slots, dist_port, out)
            if r:
                results.append(r)
                print(f"  verify={r['verify_ms']:.1f}ms cpu_comp={r['cpu_compute_ms']:.1f}ms "
                      f"merge={r['cpu_merge_ms']:.1f}ms gpu_moe={r['gpu_moe_ms']:.1f}ms "
                      f"tok_s={r['tok_s']:.3f} digest={r['digest']}")
                torch.cuda.empty_cache()

    # Summary
    print(f"\n=== Summary ===")
    print(f"{'backend':10s} {'ratio':>6s} {'verify_ms':>10s} {'cpu_ms':>10s} {'merge_ms':>10s} {'gpu_ms':>10s} {'tok_s':>8s}")
    print("-" * 70)
    for r in results:
        print(f"{r['backend']:10s} {r['ratio']:6.2f} {r['verify_ms']:10.1f} {r['cpu_compute_ms']:10.1f} "
              f"{r['cpu_merge_ms']:10.1f} {r['gpu_moe_ms']:10.1f} {r['tok_s']:8.3f}")

    # Save summary
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {args.output_dir}")


if __name__ == "__main__":
    main()
