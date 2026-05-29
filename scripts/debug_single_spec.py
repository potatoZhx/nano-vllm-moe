#!/usr/bin/env python3
"""Debug: run single spec case interactively to find the hang."""
import sys, json, time, torch
sys.path.insert(0, "/home/mumura/moe_spec/nano-vllm-moe")
from nanovllm import LLM, SamplingParams
from transformers import AutoConfig

model_path = "/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B"
hf_config = AutoConfig.from_pretrained(model_path)
num_experts = int(getattr(hf_config, "num_experts", 128))
slots = max(1, int(round(num_experts * 0.25)))

print(f"num_experts={num_experts} slots={slots}", flush=True)
print("Creating LLM (spec mode)...", flush=True)
t0 = time.time()
llm = LLM(
    model_path, dist_port=26699,
    enforce_eager=False,
    max_num_batched_tokens=8192, max_num_seqs=1, max_model_len=2048,
    gpu_memory_utilization=0.85,
    inference_mode="spec", enable_heterogeneous=True, enable_speculative=True,
    heterogeneous_slots_per_layer=slots,
    max_draft_tokens=8, draft_top_c=0,
    draft_reroute_policy="round_robin",
    acceptance_strategy="greedy",
    cpu_expert_execution_enabled=True, cpu_expert_pin_memory=True,
    cpu_expert_backend="fused",
    cpu_expert_workspace_max_routes=16384, cpu_expert_packed_min_routes=1,
    cpu_expert_parallel_mode="serial", cpu_expert_num_threads=4,
    spec_profile=True, engine_profile=True, engine_profile_cuda_sync=True,
    spec_enable_prefetch=False,
)
print(f"LLM created in {time.time()-t0:.1f}s", flush=True)

prompt_text = "Hello, this is a short test of expert routing."
prompt_ids = llm.tokenizer.encode(prompt_text)
print(f"Prompt tokens: {len(prompt_ids)}", flush=True)

print("Warmup...", flush=True)
t1 = time.time()
llm.generate([prompt_ids], SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=4), use_tqdm=False)
print(f"Warmup done in {time.time()-t1:.1f}s", flush=True)

print("Main generation (32 tokens)...", flush=True)
t2 = time.time()
outputs = llm.generate([prompt_ids], SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=32), use_tqdm=False)
elapsed = time.time() - t2
token_ids = outputs[0]["token_ids"]
text = outputs[0]["text"]
print(f"Generated {len(token_ids)} tokens in {elapsed:.1f}s", flush=True)
print(f"Text: {text[:300]}", flush=True)
llm.exit()
print("SUCCESS", flush=True)
