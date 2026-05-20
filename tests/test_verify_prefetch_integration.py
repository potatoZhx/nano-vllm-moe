#!/usr/bin/env python3
"""Integration test for verify-layer prefetch using subprocess isolation.

Each scenario runs in its own subprocess (as distributed init can't be repeated).
Tests:
1. Deterministic output (prefetch ON vs OFF produce same output)
2. Prefetch overhead profile
3. Prefetch acceleration

Scenarios: expert cache ratio 50% and 25%, num_seqs=1, output_len=128
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MODEL_PATH = "/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B"
RESULT_DIR = Path("/tmp/verify_prefetch_test_results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def run_scenario(description, heterogeneous_slots_per_layer, prefetch_verify_layer_enabled,
                 max_tokens=128, temperature=0.0, prompt="Hello, how are you?",
                 dist_port=29500):
    """Run a single spec-decode test in a subprocess and return profile + tokens."""
    script = f"""
import json
import os
import time
import torch

os.environ.setdefault("NANOVLLM_PROFILE_ENABLED", "1")

from nanovllm import LLM, SamplingParams

llm = LLM(
    model="{MODEL_PATH}",
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
    prefetch_verify_layer_enabled={prefetch_verify_layer_enabled},
    prefetch_verify_layer_safety_ratio=0.8,
    prefetch_verify_layer_min_compute_ms=0.05,
    prefetch_verify_layer_transfer_bandwidth_gbps=12.0,
    prefetch_verify_layer_max_budget=2,
    heterogeneous_slots_per_layer={heterogeneous_slots_per_layer},
    engine_profile=True,
    engine_profile_cuda_sync=False,
    perf_profile_level="basic",
    cpu_expert_backend="fused",
    dist_port={dist_port},
)

sp = SamplingParams(max_tokens={max_tokens}, temperature={temperature})
t0 = time.perf_counter()
outputs = llm.generate([{prompt!r}], sp)
elapsed = time.perf_counter() - t0

tokens = outputs[0].get("token_ids", [])

profile = llm.model_runner.get_profile(reset=False) if hasattr(llm, "model_runner") else {{}}

result = {{
    "description": {description!r},
    "elapsed_s": elapsed,
    "num_tokens": len(tokens),
    "tokens": tokens,
    "profile": profile,
}}
print("JSON_RESULT:", json.dumps(result, default=str, ensure_ascii=False))
"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "7"
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=600,
        env=env,
    )

    if proc.returncode != 0:
        print(f"  STDERR: {proc.stderr[:500]}")
        raise RuntimeError(f"Subprocess failed: {proc.stderr[-200:]}")

    for line in proc.stdout.splitlines():
        if line.startswith("JSON_RESULT:"):
            return json.loads(line[len("JSON_RESULT:"):].strip())

    raise RuntimeError(f"No JSON_RESULT found in output. stdout preview:\n{proc.stdout[:500]}")


def main():
    print("=" * 60)
    print("Verify Prefetch Comprehensive Integration Test")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")

    results = {}

    # ----------------------------------------------------------------
    # GOAL 1: Determinism
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("GOAL 1: Determinism Test")
    print("=" * 60)

    try:
        print("Running with prefetch ON (dist_port=29500)...")
        result_on = run_scenario(
            "determinism_on", heterogeneous_slots_per_layer=64,
            prefetch_verify_layer_enabled=True, max_tokens=32,
            temperature=0.0, dist_port=29500,
        )
        print(f"  Tokens ON ({len(result_on['tokens'])}): {result_on['tokens'][:15]}...")
        print(f"  Elapsed: {result_on['elapsed_s']:.2f}s")

        print("Running with prefetch OFF (dist_port=29501)...")
        result_off = run_scenario(
            "determinism_off", heterogeneous_slots_per_layer=64,
            prefetch_verify_layer_enabled=False, max_tokens=32,
            temperature=0.0, dist_port=29501,
        )
        print(f"  Tokens OFF ({len(result_off['tokens'])}): {result_off['tokens'][:15]}...")
        print(f"  Elapsed: {result_off['elapsed_s']:.2f}s")

        match = result_on["tokens"] == result_off["tokens"]
        print(f"  Token match: {match}")
        results["determinism"] = {
            "deterministic": match,
            "tokens_on": result_on["tokens"],
            "tokens_off": result_off["tokens"],
        }
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        results["determinism"] = {"deterministic": False, "error": str(e)}

    # ----------------------------------------------------------------
    # GOAL 2 & 3: Performance Profiles
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("GOAL 2 & 3: Performance Profiles")
    print("=" * 60)

    scenarios = [
        ("prefetch_on_50pct", True, 64, 29502),
        ("prefetch_off_50pct", False, 64, 29503),
        ("prefetch_on_25pct", True, 32, 29504),
        ("prefetch_off_25pct", False, 32, 29505),
    ]

    for name, prefetch_enabled, slots, port in scenarios:
        print(f"\n--- {name} (slots={slots}, port={port}) ---")
        try:
            result = run_scenario(
                name, heterogeneous_slots_per_layer=slots,
                prefetch_verify_layer_enabled=prefetch_enabled,
                max_tokens=128, temperature=0.0,
                prompt="Write a brief explanation of GPU computing in AI.",
                dist_port=port,
            )
            profile = result.get("profile", {})
            key_metrics = {
                "elapsed_s": result["elapsed_s"],
                "num_tokens": result["num_tokens"],
                "verify_layer_prefetch_hook_count": profile.get("verify_layer_prefetch_hook_count", 0),
                "verify_layer_prefetch_submit_count": profile.get("verify_layer_prefetch_submit_count", 0),
                "verify_layer_prefetch_publish_count": profile.get("verify_layer_prefetch_publish_count", 0),
                "prefetch_submit_count": profile.get("prefetch_submit_count", 0),
                "prefetch_completed_count": profile.get("prefetch_completed_count", 0),
                "prefetch_wait_ms": profile.get("prefetch_wait_ms", 0),
                "verify_forward_ms": profile.get("verify_forward_ms", 0),
            }
            print(f"  Elapsed: {key_metrics['elapsed_s']:.2f}s")
            print(f"  Tokens: {key_metrics['num_tokens']}")
            print(f"  Hook count: {key_metrics['verify_layer_prefetch_hook_count']}")
            print(f"  Layer prefetch submit: {key_metrics['verify_layer_prefetch_submit_count']}")
            print(f"  Layer prefetch publish: {key_metrics['verify_layer_prefetch_publish_count']}")
            print(f"  All submit: {key_metrics['prefetch_submit_count']}")
            print(f"  All completed: {key_metrics['prefetch_completed_count']}")
            results[name] = key_metrics
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            results[name] = {"error": str(e)}

    # ----------------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    det = results.get("determinism", {})
    print(f"\nGoal 1 - Determinism: {'PASS' if det.get('deterministic') else 'FAIL'}")
    if not det.get("deterministic"):
        print(f"  Details: {det.get('error', 'token mismatch')}")
        print(f"  Tokens ON:  {det.get('tokens_on', [])[:10]}...")
        print(f"  Tokens OFF: {det.get('tokens_off', [])[:10]}...")

    for ratio in ["50pct", "25pct"]:
        r_on = results.get(f"prefetch_on_{ratio}", {})
        r_off = results.get(f"prefetch_off_{ratio}", {})
        if "elapsed_s" in r_on and "elapsed_s" in r_off:
            speedup = r_off["elapsed_s"] / r_on["elapsed_s"] if r_on["elapsed_s"] > 0 else 1.0
            print(f"\n--- {ratio} cache ratio ---")
            print(f"  Prefetch ON:  {r_on['elapsed_s']:.2f}s, {r_on.get('num_tokens', 0)} tokens")
            print(f"  Prefetch OFF: {r_off['elapsed_s']:.2f}s, {r_off.get('num_tokens', 0)} tokens")
            print(f"  Speedup: {speedup:.3f}x")
            print(f"  Layer prefetch submit: {r_on.get('verify_layer_prefetch_submit_count', 0)}")
            print(f"  Layer prefetch publish: {r_on.get('verify_layer_prefetch_publish_count', 0)}")
            print(f"  Total prefetch submit: {r_on.get('prefetch_submit_count', 0)}")
            print(f"  Total prefetch completed: {r_on.get('prefetch_completed_count', 0)}")

    # Save
    result_path = RESULT_DIR / f"verify_prefetch_test_{int(time.time())}.json"
    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable[k] = {kk: vv for kk, vv in v.items()
                             if isinstance(vv, (str, int, float, bool, list))}
        else:
            serializable[k] = str(v)
    with open(result_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str, ensure_ascii=False)
    print(f"\nResults saved to: {result_path}")

    if det.get("deterministic", False):
        print("\nALL CHECKS PASSED")
        return 0
    else:
        print("\nFAIL: Determinism check failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
