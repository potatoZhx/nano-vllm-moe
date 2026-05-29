#!/usr/bin/env python3
"""Test: does disabling CPU expert execution fix the precision mismatch?"""
import sys, json, time, torch
sys.path.insert(0, "/home/mumura/moe_spec/nano-vllm-moe")
from nanovllm import LLM, SamplingParams

model_path = "/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B"
prompt_text = (
    "A mixture-of-experts (MoE) transformer differs from a standard dense transformer "
    "primarily in its feed-forward layers. In a dense transformer, every token activates "
    "all parameters in each feed-forward block."
)

def run_standard():
    print("=== STANDARD ===", flush=True)
    t0 = time.time()
    llm = LLM(model_path, dist_port=26750, enforce_eager=False,
              max_num_batched_tokens=8192, max_num_seqs=1, max_model_len=2048,
              gpu_memory_utilization=0.85,
              inference_mode="standard", enable_heterogeneous=False, enable_speculative=False)
    print(f"  Init: {time.time()-t0:.1f}s", flush=True)
    prompt_ids = llm.tokenizer.encode(prompt_text)
    llm.generate([prompt_ids], SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=4), use_tqdm=False)
    t1 = time.time()
    outputs = llm.generate([prompt_ids], SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=64), use_tqdm=False)
    print(f"  Gen: {time.time()-t1:.1f}s  tokens={len(outputs[0]['token_ids'])}", flush=True)
    llm.exit()
    torch.cuda.empty_cache()
    return outputs[0]["token_ids"]

def run_spec(cpu_enabled, label):
    print(f"\n=== SPEC (cpu_exec={cpu_enabled}): {label} ===", flush=True)
    t0 = time.time()
    llm = LLM(model_path, dist_port=26760 + (0 if cpu_enabled else 100), enforce_eager=False,
              max_num_batched_tokens=8192, max_num_seqs=1, max_model_len=2048,
              gpu_memory_utilization=0.85,
              inference_mode="spec", enable_heterogeneous=True, enable_speculative=True,
              heterogeneous_slots_per_layer=32,
              max_draft_tokens=8, draft_top_c=0,
              draft_reroute_policy="round_robin",
              acceptance_strategy="greedy",
              cpu_expert_execution_enabled=cpu_enabled,
              cpu_expert_pin_memory=True,
              cpu_expert_backend="fused",
              cpu_expert_workspace_max_routes=16384,
              cpu_expert_packed_min_routes=1,
              cpu_expert_parallel_mode="serial", cpu_expert_num_threads=4,
              spec_profile=True, engine_profile=True, engine_profile_cuda_sync=True,
              spec_enable_prefetch=False)
    print(f"  Init: {time.time()-t0:.1f}s", flush=True)
    prompt_ids = llm.tokenizer.encode(prompt_text)
    llm.generate([prompt_ids], SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=4), use_tqdm=False)
    t1 = time.time()
    outputs = llm.generate([prompt_ids], SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=64), use_tqdm=False)
    print(f"  Gen: {time.time()-t1:.1f}s  tokens={len(outputs[0]['token_ids'])}", flush=True)
    llm.exit()
    torch.cuda.empty_cache()
    return outputs[0]["token_ids"]

ref = run_standard()
spec_cpu = run_spec(cpu_enabled=True, label="CPU fallback")
spec_gpu = run_spec(cpu_enabled=False, label="GPU fallback only")

print("\n=== RESULTS ===")
for name, ids in [("STANDARD", ref), ("SPEC+CPU", spec_cpu), ("SPEC+GPU", spec_gpu)]:
    match = ids == ref
    if not match:
        for i in range(min(len(ids), len(ref))):
            if ids[i] != ref[i]:
                print(f"{name}: MISMATCH@{i} (out of {len(ids)})")
                break
        else:
            print(f"{name}: LEN_MISMATCH ({len(ids)} vs {len(ref)})")
    else:
        print(f"{name}: MATCH")
