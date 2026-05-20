#!/usr/bin/env python3
"""Compare raw verify logits between ON and OFF for a SINGLE verify step."""
import json, os, subprocess, sys, time

MODEL_PATH = "/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B"

SCRIPT = '''
import json, sys, time, torch
MODEL_PATH = {model_path!r}
PREFETCH_ENABLED = {prefetch_enabled}
DIST_PORT = {dist_port}
from nanovllm import LLM, SamplingParams

# Use temperature=0.0, greedy acceptance to eliminate sampling randomness
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
sp = SamplingParams(max_tokens=32, temperature=0.0)
outputs = llm.generate(["Hello, how are you?"], sp)
tokens = outputs[0]["token_ids"]

# Get profile stats
p = llm.model_runner.get_profile(reset=False)
result = {{
    "tokens": tokens, "num_tokens": len(tokens),
    "vf_hooks": p.get("verify_layer_prefetch_hook_count", 0),
    "vf_submit": p.get("verify_layer_prefetch_submit_count", 0),
    "vf_publish": p.get("verify_layer_prefetch_publish_count", 0),
    "draft_graph": p.get("draft_graph_replay_count", 0),
    "warmup": p.get("verify_layer_timing_warmup_count", 0),
}}
del llm; torch.cuda.empty_cache()
print("RESULT: " + json.dumps(result, default=str))
'''

def run_one(name, enabled, port):
    print(f"  Running {name} (port={port})...", flush=True)
    proc = subprocess.run([sys.executable, "-c", SCRIPT.format(model_path=MODEL_PATH, prefetch_enabled=enabled, dist_port=port)],
                         capture_output=True, text=True, timeout=600, env=os.environ.copy())
    if proc.returncode != 0:
        return {"error": proc.stderr[-500:]}
    for line in proc.stdout.splitlines():
        if line.strip().startswith("RESULT: "):
            return json.loads(line.strip()[len("RESULT: "):])
    return {"error": "no RESULT"}

print("=" * 60)
print("Single verify step logits comparison")
print("=" * 60)

r_on = run_one("ON", True, 29650)
r_off = run_one("OFF", False, 29651)

if "error" in r_on or "error" in r_off:
    print(f"ERR ON: {r_on.get('error','')[:300]}")
    print(f"ERR OFF: {r_off.get('error','')[:300]}")
    sys.exit(1)

t_on = r_on["tokens"]
t_off = r_off["tokens"]
match = t_on == t_off

print(f"\nON:  {len(t_on)} tokens, hooks={r_on['vf_hooks']}, submit={r_on['vf_submit']}, draft_graph={r_on['draft_graph']}, warmup={r_on['warmup']}")
print(f"OFF: {len(t_off)} tokens, hooks={r_off['vf_hooks']}, draft_graph={r_off['draft_graph']}")
print(f"Match: {match}")
if not match:
    for i, (a, b) in enumerate(zip(t_on, t_off)):
        if a != b:
            print(f"  First diff at pos {i}: ON={a}, OFF={b}")
            break
    # Find first matching prefix length
    prefix = 0
    for a, b in zip(t_on, t_off):
        if a == b: prefix += 1
        else: break
    print(f"  Matching prefix: {prefix}/{len(t_on)} tokens")

sys.exit(0 if match else 1)
