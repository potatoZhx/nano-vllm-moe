#!/usr/bin/env python3
"""Quick determinism check via subprocesses."""
import subprocess, sys, json

SCRIPT = """
import json, sys, time, torch
from nanovllm import LLM, SamplingParams
llm = LLM(model="/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B", inference_mode="spec", enable_heterogeneous=True,
    enable_speculative=True, max_num_batched_tokens=512, max_num_seqs=1,
    max_model_len=512, max_draft_tokens=4, draft_top_c=0,
    acceptance_strategy="greedy", enforce_eager=False, spec_verify_eager=False,
    spec_enable_prefetch=True, cache_strategy="lru", prefetch_strategy="history_window",
    prefetch_step_budget=8, prefetch_max_inflight=16, prefetch_staging_slots_per_layer=4,
    cache_eviction_budget_per_step=4, prefetch_verify_wait_ms=0.0,
    prefetch_verify_layer_enabled=PREFETCH_ON,
    heterogeneous_slots_per_layer=64, engine_profile=True,
    cpu_expert_backend="fused", cpu_expert_pin_memory=True,
    cpu_expert_packed_min_routes=1, cpu_expert_parallel_mode="serial",
    cpu_expert_num_threads=4, cpu_gpu_parallel_execution_enabled="auto",
    gpu_memory_utilization=0.85, seed=0, dist_port=PORT)
sp = SamplingParams(max_tokens=32, temperature=0.0)
outputs = llm.generate(["Hello"], sp)
tokens = outputs[0]["token_ids"]
p = llm.model_runner.get_profile(reset=False)
r = {"tokens": tokens, "hooks": p.get("verify_layer_prefetch_hook_count",0),
     "submit": p.get("verify_layer_prefetch_submit_count",0)}
del llm; torch.cuda.empty_cache()
print("R:", json.dumps(r))
"""

r1 = subprocess.run([sys.executable, "-c", SCRIPT.replace("PREFETCH_ON", "True").replace("PORT", "29730")],
                   capture_output=True, text=True, timeout=600)
r2 = subprocess.run([sys.executable, "-c", SCRIPT.replace("PREFETCH_ON", "False").replace("PORT", "29731")],
                   capture_output=True, text=True, timeout=600)

d1 = json.loads([l for l in r1.stdout.splitlines() if l.startswith("R:")][0][3:])
d2 = json.loads([l for l in r2.stdout.splitlines() if l.startswith("R:")][0][3:])
print(f"ON:  {d1['tokens'][:10]}... hooks={d1['hooks']} submit={d1['submit']}")
print(f"OFF: {d2['tokens'][:10]}... hooks={d2['hooks']}")
match = d1["tokens"] == d2["tokens"]
print(f"MATCH: {match}")
if not match:
    for i,(a,b) in enumerate(zip(d1["tokens"],d2["tokens"])):
        if a!=b: print(f"  Diff pos {i}: ON={a} OFF={b}"); break
sys.exit(0 if match else 1)
