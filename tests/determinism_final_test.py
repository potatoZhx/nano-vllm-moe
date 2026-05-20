#!/usr/bin/env python3
"""Final determinism test with both fixes applied."""
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
    acceptance_strategy="greedy", enforce_eager=False, spec_verify_eager=False,
    spec_enable_prefetch=True, cache_strategy="lru", prefetch_strategy="history_window",
    prefetch_step_budget=8, prefetch_max_inflight=16, prefetch_staging_slots_per_layer=4,
    cache_eviction_budget_per_step=4, prefetch_verify_wait_ms=0.0,
    prefetch_global_queue_capacity=4096,
    prefetch_verify_layer_enabled=PREFETCH_ENABLED,
    heterogeneous_slots_per_layer=64, engine_profile=True,
    cpu_expert_backend="fused", cpu_expert_pin_memory=True,
    cpu_expert_packed_min_routes=1, cpu_expert_parallel_mode="serial",
    cpu_expert_num_threads=4, cpu_gpu_parallel_execution_enabled="auto",
    gpu_memory_utilization=0.85, seed=0, dist_port=DIST_PORT)
init_s = time.perf_counter() - t0
sp = SamplingParams(max_tokens=32, temperature=0.0)
t1 = time.perf_counter()
outputs = llm.generate(["Hello, how are you?"], sp)
gen_s = time.perf_counter() - t1
tokens = outputs[0]["token_ids"]
profile = llm.model_runner.get_profile(reset=False)
prefetch_hooks = profile.get("verify_layer_prefetch_hook_count", 0)
prefetch_submit = profile.get("verify_layer_prefetch_submit_count", 0)
prefetch_publish = profile.get("verify_layer_prefetch_publish_count", 0)
draft_graph = profile.get("draft_graph_replay_count", 0)
warmup_count = profile.get("verify_layer_timing_warmup_count", 0)
result = {{
    "tokens": tokens, "gen_s": round(gen_s, 2), "init_s": round(init_s, 1),
    "num_tokens": len(tokens), "draft_graph": draft_graph,
    "hooks": prefetch_hooks, "submit": prefetch_submit,
    "publish": prefetch_publish, "warmup": warmup_count,
}}
del llm; torch.cuda.empty_cache()
print("RESULT: " + json.dumps(result, default=str))
'''

def run_one(name, enabled, port):
    print(f"  Running {name} (port={port})...", flush=True)
    proc = subprocess.run([sys.executable, "-c", make_script(enabled, port)],
                         capture_output=True, text=True, timeout=600, env=os.environ.copy())
    if proc.returncode != 0:
        return {"error": proc.stderr[-500:]}
    for line in proc.stdout.splitlines():
        if line.strip().startswith("RESULT: "):
            return json.loads(line.strip()[len("RESULT: "):])
    return {"error": "no RESULT"}

print("=" * 60)
print("Final Determinism Test (warmup fix + empty-slot prefetch)")
print("=" * 60)

r_on = run_one("PREFETCH ON", True, 29630)
r_off = run_one("PREFETCH OFF", False, 29631)

if "error" in r_on or "error" in r_off:
    print(f"ERROR ON: {r_on.get('error', 'OK')[:300]}")
    print(f"ERROR OFF: {r_off.get('error', 'OK')[:300]}")
    sys.exit(1)

t_on = r_on["tokens"]
t_off = r_off["tokens"]
match = t_on == t_off

print(f"\nON:  {len(t_on)} tokens, gen={r_on['gen_s']}s, hooks={r_on['hooks']}, "
      f"submit={r_on['submit']}, publish={r_on['publish']}, warmup={r_on['warmup']}")
print(f"OFF: {len(t_off)} tokens, gen={r_off['gen_s']}s, hooks={r_off['hooks']}")
print(f"\nDETERMINISM: {'PASS' if match else 'FAIL'}")
if match:
    print(f"  Tokens ON:  {t_on}")
    print(f"  Tokens OFF: {t_off}")
else:
    for i, (a, b) in enumerate(zip(t_on, t_off)):
        if a != b:
            print(f"  First diff at pos {i}: ON={a}, OFF={b}")
            break
    print(f"  ON[:15]:  {t_on[:15]}")
    print(f"  OFF[:15]: {t_off[:15]}")

sys.exit(0 if match else 1)
