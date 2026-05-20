#!/usr/bin/env python3
"""Determinism test with CUDA graph disabled (enforce_eager=True)."""
import json, os, subprocess, sys, time

MODEL_PATH = "/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B"

def make_script(prefetch_enabled, dist_port):
    return f'''
import json, sys, time, torch
MODEL_PATH = {MODEL_PATH!r}
PREFETCH_ENABLED = {prefetch_enabled}
DIST_PORT = {dist_port}
from nanovllm import LLM, SamplingParams
t0 = time.perf_counter()
llm = LLM(
    model=MODEL_PATH, inference_mode="spec", enable_heterogeneous=True,
    enable_speculative=True, max_num_batched_tokens=512, max_num_seqs=1,
    max_model_len=512, max_draft_tokens=4, draft_top_c=0,
    acceptance_strategy="greedy", enforce_eager=True, spec_verify_eager=True,
    spec_enable_prefetch=True, cache_strategy="lru", prefetch_strategy="history_window",
    prefetch_step_budget=4, prefetch_max_inflight=8, prefetch_verify_wait_ms=2.0,
    prefetch_verify_layer_enabled=PREFETCH_ENABLED,
    heterogeneous_slots_per_layer=64, engine_profile=True,
    cpu_expert_backend="fused", cpu_expert_pin_memory=True,
    gpu_memory_utilization=0.85, seed=0, dist_port=DIST_PORT)
init_s = time.perf_counter() - t0
sp = SamplingParams(max_tokens=32, temperature=0.0)
t1 = time.perf_counter()
outputs = llm.generate(["Hello, how are you?"], sp)
gen_s = time.perf_counter() - t1
tokens = outputs[0]["token_ids"]
p = llm.model_runner.get_profile(reset=False)
result = {{
    "tokens": tokens, "gen_s": round(gen_s, 2),
    "vf_hooks": p.get("verify_layer_prefetch_hook_count",0),
    "vf_submit": p.get("verify_layer_prefetch_submit_count",0),
    "draft_graph": p.get("draft_graph_replay_count",0),
}}
del llm; torch.cuda.empty_cache()
print("RESULT: " + json.dumps(result))
'''

def run_one(name, enabled, port):
    print(f"  Running {name}...", flush=True)
    proc = subprocess.run([sys.executable, "-c", make_script(enabled, port)],
                         capture_output=True, text=True, timeout=600, env=os.environ.copy())
    if proc.returncode != 0: return {"error": proc.stderr[-500:]}
    for line in proc.stdout.splitlines():
        if line.strip().startswith("RESULT: "):
            return json.loads(line.strip()[len("RESULT: "):])
    return {"error": "no RESULT"}

print("=" * 60)
print("Determinism Test: enforce_eager=True (no CUDA graph)")
print("=" * 60)

r_on = run_one("ON", True, 29660)
r_off = run_one("OFF", False, 29661)

if "error" in r_on or "error" in r_off:
    print(f"ERR ON: {r_on.get('error','')[:300]}")
    print(f"ERR OFF: {r_off.get('error','')[:300]}")
    sys.exit(1)

t_on = r_on["tokens"]; t_off = r_off["tokens"]
match = t_on == t_off

print(f"\nON:  {len(t_on)} t, hooks={r_on['vf_hooks']}, submit={r_on['vf_submit']}, graph={r_on['draft_graph']}")
print(f"OFF: {len(t_off)} t, hooks={r_off['vf_hooks']}, graph={r_off['draft_graph']}")
print(f"DETERMINISM: {'PASS' if match else 'FAIL'}")
if not match:
    for i,(a,b) in enumerate(zip(t_on, t_off)):
        if a!=b: print(f"  First diff pos {i}: ON={a} OFF={b}"); break

sys.exit(0 if match else 1)
