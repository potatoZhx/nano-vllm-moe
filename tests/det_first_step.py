#!/usr/bin/env python3
"""Compare first-step verify logits between ON and OFF."""
import subprocess, sys, json

SCRIPT = """
import json, sys, time, torch
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.sequence import Sequence
from nanovllm.config import Config
from multiprocessing import Event

cfg = Config(
    model="/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B",
    inference_mode="spec", enable_heterogeneous=True, enable_speculative=True,
    max_num_batched_tokens=512, max_num_seqs=1, max_model_len=512,
    max_draft_tokens=4, draft_top_c=0, acceptance_strategy="greedy",
    enforce_eager=False, spec_verify_eager=False, spec_enable_prefetch=True,
    cache_strategy="lru", prefetch_strategy="history_window",
    prefetch_step_budget=8, prefetch_max_inflight=16, prefetch_staging_slots_per_layer=4,
    cache_eviction_budget_per_step=4, prefetch_verify_wait_ms=0.0,
    prefetch_verify_layer_enabled=PREFETCH_ON, heterogeneous_slots_per_layer=64,
    engine_profile=True, cpu_expert_backend="fused", cpu_expert_pin_memory=True,
    gpu_memory_utilization=0.85, seed=0, dist_port=PORT)
runner = ModelRunner(cfg, rank=0, event=Event())
runner.profile_enabled = True

# Run a single verify step with just 1 token (no draft)
seq = Sequence([0]*8 + [1])  # prompt + 1 continuation
result = runner.run_verify([seq], verify_lengths=[2], return_logits=True)
logits = torch.stack(result)
runner.get_profile(reset=True)
del runner; torch.cuda.empty_cache()
print("L:", json.dumps(logits[0,:10].argmax(dim=-1).tolist()))
"""

r1 = subprocess.run([sys.executable, "-c", SCRIPT.replace("PREFETCH_ON", "True").replace("PORT", "29740")],
                   capture_output=True, text=True, timeout=600)
r2 = subprocess.run([sys.executable, "-c", SCRIPT.replace("PREFETCH_ON", "False").replace("PORT", "29741")],
                   capture_output=True, text=True, timeout=600)

import re
l1 = json.loads([l for l in r1.stdout.splitlines() if l.startswith("L:")][0][3:])
l2 = json.loads([l for l in r2.stdout.splitlines() if l.startswith("L:")][0][3:])
print(f"ON tokens:  {l1}")
print(f"OFF tokens: {l2}")
print(f"Match: {l1 == l2}")
