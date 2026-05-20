#!/usr/bin/env python3
"""Verify prefetch test with CUDA graph and all acceleration features enabled.
Matches May 9 spec_sampling_overlap_publishfix_full configuration.
Tests determinism (ON vs OFF token match) at temperature=0.0,
then performance at temperature=0.8 with 128-token output.
"""
import json
import os
import subprocess
import sys
import time
import torch  # noqa: F401 — needed for cleanup

MODEL_PATH = "/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B"
OUTPUT_DIR = "/tmp/verify_prefetch_accel_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_KWARGS = dict(
    model=MODEL_PATH,
    inference_mode="spec",
    enable_heterogeneous=True,
    enable_speculative=True,
    max_num_batched_tokens=512,
    max_num_seqs=1,
    max_model_len=512,
    max_draft_tokens=4,
    draft_top_c=0,
    acceptance_strategy="standard_sampling",
    enforce_eager=False,
    spec_verify_eager=False,
    spec_enable_prefetch=True,
    cache_strategy="lru",
    prefetch_strategy="history_window",
    prefetch_step_budget=8,
    prefetch_max_inflight=16,
    prefetch_staging_slots_per_layer=4,
    cache_eviction_budget_per_step=4,
    prefetch_verify_wait_ms=0.0,
    prefetch_global_queue_capacity=4096,
    cpu_expert_backend="fused",
    cpu_expert_pin_memory=True,
    cpu_expert_packed_min_routes=1,
    cpu_expert_parallel_mode="serial",
    cpu_expert_num_threads=4,
    cpu_gpu_parallel_execution_enabled="auto",
    gpu_memory_utilization=0.85,
    engine_profile=True,
    engine_profile_cuda_sync=False,
    perf_profile_level="basic",
    seed=0,
)

def _make_script(prefetch_verify_layer_enabled, heterogeneous_slots_per_layer,
                 max_tokens, temperature, dist_port):
    """Generate a self-contained Python script for subprocess execution."""
    kwargs = dict(BASE_KWARGS)
    kwargs["prefetch_verify_layer_enabled"] = prefetch_verify_layer_enabled
    kwargs["heterogeneous_slots_per_layer"] = heterogeneous_slots_per_layer
    kwargs["dist_port"] = dist_port

    # Build LLM(...) argument string using repr() for correct Python literals
    llm_args = ",\n    ".join(f"{k}={v!r}" for k, v in kwargs.items())

    return f'''
import json, sys, time, torch

from nanovllm import LLM, SamplingParams

t0 = time.perf_counter()
llm = LLM(
    {llm_args}
)
init_s = time.perf_counter() - t0
sp = SamplingParams(max_tokens={max_tokens}, temperature={temperature})
t1 = time.perf_counter()
outputs = llm.generate(["Hello, how are you?"], sp)
gen_s = time.perf_counter() - t1
elapsed_s = time.perf_counter() - t0

tokens = outputs[0]["token_ids"]
profile = llm.model_runner.get_profile(reset=False)

# Collect key profile counters
result = {{
    "tokens": tokens,
    "init_s": round(init_s, 3),
    "gen_s": round(gen_s, 3),
    "elapsed_s": round(elapsed_s, 3),
    "num_tokens": len(tokens),
    # Spec profile counters
    "draft_graph_replay_count": profile.get("draft_graph_replay_count", 0),
    "verify_graph_replay_count": profile.get("verify_graph_replay_count", 0),
    "decode_tok_per_s": profile.get("decode_tok_per_s", 0.0),
    "e2e_tok_per_s": profile.get("e2e_tok_per_s", 0.0),
    # Verify prefetch counters
    "verify_layer_prefetch_hook_count": profile.get("verify_layer_prefetch_hook_count", 0),
    "verify_layer_prefetch_submit_count": profile.get("verify_layer_prefetch_submit_count", 0),
    "verify_layer_prefetch_publish_count": profile.get("verify_layer_prefetch_publish_count", 0),
    "verify_layer_prefetch_budget_stop_count": profile.get("verify_layer_prefetch_budget_stop_count", 0),
    # Prefetch system counters
    "prefetch_submit_count": profile.get("prefetch_submit_count", 0),
    "prefetch_completed_count": profile.get("prefetch_completed_count", 0),
    "prefetch_consumed_count": profile.get("prefetch_consumed_count", 0),
    "prefetch_late_count": profile.get("prefetch_late_count", 0),
    "publish_count": profile.get("publish_count", 0),
    "publish_ms": profile.get("publish_ms", 0),
    "prefetch_wait_ms": profile.get("prefetch_wait_ms", 0),
    # Timing
    "draft_avg_ms": profile.get("draft_avg_ms", 0),
    "verify_avg_ms": profile.get("verify_avg_ms", 0),
    "verify_forward_ms": profile.get("verify_forward_ms", 0),
    "draft_forward_ms": profile.get("draft_forward_ms", 0),
    # Spec stats
    "accept_rate": profile.get("accept_rate", 0),
    "num_spec_steps": profile.get("num_spec_steps", 0),
    "draft_count": profile.get("draft_count", 0),
}}

del llm
torch.cuda.empty_cache()
print("RESULT: " + json.dumps(result, default=str, ensure_ascii=False))
sys.exit(0)
'''


def run_one(name, prefetch_on, slots, max_tokens, temperature, port):
    script = _make_script(prefetch_on, slots, max_tokens, temperature, port)
    print(f"  Running {name} (port={port})...", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=1200,
        env=os.environ.copy(),
    )
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        err = proc.stderr[-800:] if proc.stderr else "(no stderr)"
        return {"error": err, "elapsed": elapsed}
    for line in proc.stdout.splitlines():
        if line.strip().startswith("RESULT: "):
            result = json.loads(line.strip()[len("RESULT: "):])
            result["wall_clock_s"] = round(elapsed, 1)
            return result
    return {"error": "no RESULT line", "stdout": proc.stdout[:500]}


def main():
    print("=" * 60)
    print("Verify Prefetch Test (CUDA Graph + Acceleration ON)")
    print(f"Model: {MODEL_PATH}")
    print(f"Settings: draft_top_c=0, enforce_eager=False, pin_memory=True")
    print("=" * 60)

    results = {}

    # ---- Goal 1: Determinism (temp=0.0, greedy acceptance) ----
    print("\n--- Goal 1: Determinism (32 tokens, temp=0.0) ---")
    # Use acceptance_strategy=greedy for deterministic comparison
    # We need to override just the acceptance strategy for this test
    # Actually, let's use standard_sampling with temp=0.0 which falls back to greedy

    # For determinism: compare ON vs OFF
    r_on = run_one("det_on", True, 64, 32, 0.0, 29600)
    r_off = run_one("det_off", False, 64, 32, 0.0, 29601)

    if "error" not in r_on and "error" not in r_off:
        t_on = r_on["tokens"]
        t_off = r_off["tokens"]
        match = t_on == t_off
        results["determinism"] = {
            "deterministic": match,
            "tokens_on": t_on,
            "tokens_off": t_off,
            "gen_s_on": r_on.get("gen_s", 0),
            "gen_s_off": r_off.get("gen_s", 0),
            "draft_graph_replay_on": r_on.get("draft_graph_replay_count", 0),
            "draft_graph_replay_off": r_off.get("draft_graph_replay_count", 0),
        }
        print(f"  ON:  {len(t_on)} tokens, gen={r_on.get('gen_s', 0):.1f}s, "
              f"draft_graph={r_on.get('draft_graph_replay_count', 0)}")
        print(f"  OFF: {len(t_off)} tokens, gen={r_off.get('gen_s', 0):.1f}s, "
              f"draft_graph={r_off.get('draft_graph_replay_count', 0)}")
        print(f"  Token match: {match}")
        if not match:
            for i, (a, b) in enumerate(zip(t_on, t_off)):
                if a != b:
                    print(f"  First diff at pos {i}: ON={a}, OFF={b}")
                    break
    else:
        results["determinism"] = {"error_on": r_on.get("error", ""),
                                  "error_off": r_off.get("error", "")}

    # ---- Goal 2&3: Performance (temp=0.8, 128 tokens) ----
    print("\n--- Performance (128 tokens, temp=0.8, standard_sampling) ---")
    scenarios = [
        ("prefetch_on_50pct", True, 64, 29602),
        ("prefetch_off_50pct", False, 64, 29603),
        ("prefetch_on_25pct", True, 32, 29604),
        ("prefetch_off_25pct", False, 32, 29605),
    ]
    for name, on, slots, port in scenarios:
        print(f"\n  {name} (slots={slots})...")
        r = run_one(name, on, slots, 128, 0.8, port)
        if "error" in r:
            print(f"    ERROR: {r['error'][:200]}")
        else:
            print(f"    gen={r.get('gen_s', 0):.1f}s, tok/s(gen)={128/r['gen_s']:.2f}"
                  if r.get('gen_s', 0) > 0 else f"    gen=0")
            print(f"    draft_graph={r.get('draft_graph_replay_count', 0)}, "
                  f"verify_graph={r.get('verify_graph_replay_count', 0)}")
            print(f"    verify_avg={r.get('verify_avg_ms', 0):.1f}ms, "
                  f"draft_avg={r.get('draft_avg_ms', 0):.1f}ms")
            print(f"    accept_rate={r.get('accept_rate', 0):.4f}")
            print(f"    vf_hooks={r.get('verify_layer_prefetch_hook_count', 0)}, "
                  f"submit={r.get('verify_layer_prefetch_submit_count', 0)}, "
                  f"publish={r.get('verify_layer_prefetch_publish_count', 0)}")
            print(f"    prefetch_consumed={r.get('prefetch_consumed_count', 0)}")
            # Compute derived metrics
            gen_s = r.get('gen_s', 0)
            if gen_s > 0:
                r['decode_tok_per_s'] = round(128 / gen_s, 3)
        results[name] = r
        torch.cuda.empty_cache()  # parent process cleanup

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (CUDA Graph + Acceleration ON)")
    print("=" * 60)

    det = results.get("determinism", {})
    print(f"\nGoal 1 - Determinism: {'PASS' if det.get('deterministic') else 'FAIL'}")
    if det.get("draft_graph_replay_on", 0) > 0:
        print(f"  CUDA graph active: YES (replay={det['draft_graph_replay_on']})")

    print(f"\n{'Scenario':<25} {'gen_s':>8} {'tok/s':>8} {'draft_g':>7} {'verify_avg':>10} {'draft_avg':>9} {'accept':>7} {'hooks':>7} {'submit':>7} {'publish':>7} {'consumed':>9}")
    print("-" * 110)
    for name, on, slots, port in scenarios:
        r = results.get(name, {})
        if "error" not in r:
            gen_s = r.get('gen_s', 0)
            tok_per_s = 128 / gen_s if gen_s > 0 else 0
            print(f"{name:<25} {gen_s:>8.2f} {tok_per_s:>8.2f} "
                  f"{r.get('draft_graph_replay_count',0):>7} "
                  f"{r.get('verify_avg_ms',0):>10.1f} "
                  f"{r.get('draft_avg_ms',0):>9.1f} "
                  f"{r.get('accept_rate',0):>7.4f} "
                  f"{r.get('verify_layer_prefetch_hook_count',0):>7} "
                  f"{r.get('verify_layer_prefetch_submit_count',0):>7} "
                  f"{r.get('verify_layer_prefetch_publish_count',0):>7} "
                  f"{r.get('prefetch_consumed_count',0):>9}")

    # Save
    result_path = os.path.join(OUTPUT_DIR, f"accel_test_{int(time.time())}.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    print(f"\nResults saved to: {result_path}")
    return 0 if det.get("deterministic", True) else 1


if __name__ == "__main__":
    sys.exit(main())
