#!/usr/bin/env python3
"""
Full integration test for verify-layer prefetch using process isolation.
Each scenario runs in its own subprocess to avoid NCCL re-init issues.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MODEL_PATH = "/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B"
RESULT_DIR = Path("/tmp/verify_prefetch_test_results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Script template for a single run
RUNNER_SCRIPT = """
import json, os, sys, time, torch

MODEL_PATH = {model_path!r}
PREFETCH_ENABLED = {prefetch_enabled}
SLOTS = {slots}
MAX_TOKENS = {max_tokens}
DIST_PORT = {dist_port}

from nanovllm import LLM, SamplingParams

t0 = time.perf_counter()
llm = LLM(
    model=MODEL_PATH,
    inference_mode="spec",
    enable_heterogeneous=True,
    enable_speculative=True,
    max_num_batched_tokens=4096,
    max_num_seqs=2,
    max_model_len=2048,
    max_draft_tokens=4,
    draft_top_c=2,
    acceptance_strategy="greedy",
    enforce_eager=True,
    spec_verify_eager=True,
    spec_enable_prefetch=True,
    cache_strategy="lru",
    prefetch_strategy="history_window",
    prefetch_step_budget=4,
    prefetch_max_inflight=8,
    prefetch_verify_wait_ms=2.0,
    prefetch_global_queue_capacity=4096,
    prefetch_verify_layer_enabled=PREFETCH_ENABLED,
    prefetch_verify_layer_safety_ratio=0.8,
    prefetch_verify_layer_min_compute_ms=0.05,
    prefetch_verify_layer_transfer_bandwidth_gbps=12.0,
    prefetch_verify_layer_max_budget=2,
    heterogeneous_slots_per_layer=SLOTS,
    engine_profile=True,
    engine_profile_cuda_sync=False,
    perf_profile_level="basic",
    cpu_expert_backend="fused",
    dist_port=DIST_PORT,
)
sp = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)
outputs = llm.generate(["Hello, how are you?"], sp)
elapsed = time.perf_counter() - t0
tokens = outputs[0]["token_ids"]
profile = llm.model_runner.get_profile(reset=False)

result = {{
    "tokens": tokens,
    "elapsed_s": elapsed,
    "prefetch_enabled": PREFETCH_ENABLED,
    "slots": SLOTS,
    "hook_count": profile.get("verify_layer_prefetch_hook_count", 0),
    "submit_count": profile.get("verify_layer_prefetch_submit_count", 0),
    "publish_count": profile.get("verify_layer_prefetch_publish_count", 0),
    "budget_stop_count": profile.get("verify_layer_prefetch_budget_stop_count", 0),
    "verify_forward_ms": profile.get("verify_forward_ms", 0),
    "prefetch_submit_total": profile.get("prefetch_submit_count", 0),
    "prefetch_completed_total": profile.get("prefetch_completed_count", 0),
    "prefetch_wait_ms": profile.get("prefetch_wait_ms", 0),
}}

del llm
torch.cuda.empty_cache()
print("JSON_RESULT: " + json.dumps(result, default=str, ensure_ascii=False))
sys.exit(0)
"""


def run_one(prefetch_enabled, slots, max_tokens, dist_port):
    """Run a single scenario in a subprocess."""
    script = RUNNER_SCRIPT.format(
        model_path=MODEL_PATH,
        prefetch_enabled=prefetch_enabled,
        slots=slots,
        max_tokens=max_tokens,
        dist_port=dist_port,
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=600,
        env=os.environ.copy(),
    )
    if proc.returncode != 0:
        err = proc.stderr[-500:] if proc.stderr else "(no stderr)"
        raise RuntimeError(f"Subprocess failed (port={dist_port}): {err}")

    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("JSON_RESULT:"):
            return json.loads(line[len("JSON_RESULT:"):])

    raise RuntimeError(f"No JSON_RESULT in output (port={dist_port})")


def main():
    print("=" * 60)
    print("Verify Prefetch Integration Test (process-isolated)")
    print(f"Model: {MODEL_PATH}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print("=" * 60)

    results = {}

    # Goal 1: Determinism
    print("\n--- Goal 1: Determinism (prefetch ON vs OFF) ---")
    try:
        print("Running prefetch ON (port=29500)...", flush=True)
        r_on = run_one(prefetch_enabled=True, slots=64, max_tokens=32, dist_port=29500)
        print(f"  Tokens: {r_on['tokens'][:10]}...")
        print(f"  Hook count: {r_on['hook_count']}")
        print(f"  Submit: {r_on['submit_count']}, Publish: {r_on['publish_count']}")
        print(f"  Verify forward: {r_on['verify_forward_ms']:.2f}ms")

        print("Running prefetch OFF (port=29501)...", flush=True)
        r_off = run_one(prefetch_enabled=False, slots=64, max_tokens=32, dist_port=29501)
        print(f"  Tokens: {r_off['tokens'][:10]}...")
        print(f"  Hook count: {r_off['hook_count']}")

        match = r_on["tokens"] == r_off["tokens"]
        results["determinism"] = {
            "deterministic": match,
            "tokens_on": r_on["tokens"],
            "tokens_off": r_off["tokens"],
        }
        print(f"  Deterministic: {'PASS' if match else 'FAIL'}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        results["determinism"] = {"deterministic": False, "error": str(e)}

    # Goal 2 & 3: Performance at 50% and 25% cache
    print("\n--- Goal 2 & 3: Performance profiles ---")
    scenarios = [
        ("prefetch_on_50pct", True, 64, 29502),
        ("prefetch_off_50pct", False, 64, 29503),
        ("prefetch_on_25pct", True, 32, 29504),
        ("prefetch_off_25pct", False, 32, 29505),
    ]
    for name, enabled, slots, port in scenarios:
        try:
            print(f"Running {name} (port={port})...", flush=True)
            r = run_one(prefetch_enabled=enabled, slots=slots, max_tokens=128, dist_port=port)
            results[name] = r
            print(f"  Tokens: {len(r['tokens'])}, Elapsed: {r['elapsed_s']:.2f}s")
            print(f"  Hooks: {r['hook_count']}, Submit: {r['submit_count']}, Publish: {r['publish_count']}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            results[name] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    det = results.get("determinism", {})
    print(f"\nGoal 1 - Determinism: {'PASS' if det.get('deterministic') else 'FAIL'}")
    if not det.get("deterministic"):
        print(f"  ON:  {det.get('tokens_on', [])[:10]}...")
        print(f"  OFF: {det.get('tokens_off', [])[:10]}...")

    for ratio in ["50pct", "25pct"]:
        r_on = results.get(f"prefetch_on_{ratio}", {})
        r_off = results.get(f"prefetch_off_{ratio}", {})
        print(f"\n--- {ratio} cache ---")
        if "elapsed_s" in r_on:
            print(f"  ON:  {r_on['elapsed_s']:.2f}s, hooks={r_on['hook_count']}, submit={r_on['submit_count']}, publish={r_on['publish_count']}")
        else:
            print(f"  ON: ERROR - {r_on.get('error', 'unknown')}")
        if "elapsed_s" in r_off:
            print(f"  OFF: {r_off['elapsed_s']:.2f}s")
        else:
            print(f"  OFF: ERROR - {r_off.get('error', 'unknown')}")
        if "elapsed_s" in r_on and "elapsed_s" in r_off and r_off["elapsed_s"] > 0:
            speedup = r_off["elapsed_s"] / r_on["elapsed_s"]
            print(f"  Speedup: {speedup:.3f}x")

    # Save
    result_path = RESULT_DIR / f"full_integration_{int(time.time())}.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    print(f"\nResults saved to: {result_path}")

    return 0 if det.get("deterministic") else 1


if __name__ == "__main__":
    import torch
    sys.exit(main())
