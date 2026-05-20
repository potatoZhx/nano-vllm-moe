#!/usr/bin/env python3
"""Deep dive determinism test comparing verify logits directly."""
import json
import subprocess
import sys

MODEL_PATH = "/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B"

def _make_script(model_path, enabled, port):
    return f"""
import json, sys, torch
MODEL_PATH = {model_path!r}
from nanovllm import LLM, SamplingParams
llm = LLM(model=MODEL_PATH, inference_mode="spec", enable_heterogeneous=True,
    enable_speculative=True, max_num_batched_tokens=4096, max_num_seqs=2,
    max_model_len=2048, max_draft_tokens=4, draft_top_c=0,
    acceptance_strategy="greedy", enforce_eager=True, spec_verify_eager=True,
    spec_enable_prefetch=True, cache_strategy="lru", prefetch_strategy="history_window",
    prefetch_step_budget=4, prefetch_max_inflight=8, prefetch_verify_wait_ms=2.0,
    prefetch_global_queue_capacity=4096,
    prefetch_verify_layer_enabled={enabled},
    heterogeneous_slots_per_layer=64, engine_profile=True,
    dist_port={port})
sp = SamplingParams(max_tokens=32, temperature=0.0)
outputs = llm.generate(["Hi"], sp)
tokens = outputs[0]["token_ids"]
prefetch_hooks = llm.model_runner.get_profile(reset=False).get("verify_layer_prefetch_hook_count", 0)
print("RESULT: " + json.dumps({{"tokens": tokens, "hooks": prefetch_hooks}}))
"""


def run_one(enabled, port):
    code = _make_script(MODEL_PATH, enabled, port)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        print(f"FAILED (port={port}): {proc.stderr[-300:]}")
        return None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("RESULT: "):
            return json.loads(line[len("RESULT: "):])
    print(f"No RESULT line (port={port}): stdout={proc.stdout[:200]}")
    return None


r_on = run_one(True, 29540)
r_off = run_one(False, 29541)

if r_on and r_off:
    t_on = r_on["tokens"]
    t_off = r_off["tokens"]
    print(f"ON  ({len(t_on)} tokens, {r_on['hooks']} hooks): {t_on[:20]}...")
    print(f"OFF ({len(t_off)} tokens, {r_off['hooks']} hooks): {t_off[:20]}...")
    match = t_on == t_off
    print(f"\nDETERMINISM: {'PASS' if match else 'FAIL'}")
    if not match:
        for i, (a, b) in enumerate(zip(t_on, t_off)):
            if a != b:
                print(f"  First diff at pos {i}: ON={a}, OFF={b}")
                break
    sys.exit(0 if match else 1)
else:
    print("FAILED: one or both runs returned None")
    sys.exit(1)
