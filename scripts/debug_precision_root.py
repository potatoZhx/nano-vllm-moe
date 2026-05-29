#!/usr/bin/env python3
"""Debug: isolate precision difference source."""
import sys, time, torch
sys.path.insert(0, "/home/mumura/moe_spec/nano-vllm-moe")
from nanovllm import LLM, SamplingParams

model_path = "/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B"
prompt_text = "Hello world, this is a precision test."

def run_and_get_tokens(config_name, extra_config):
    """Run LLM with given config and return token_ids."""
    print(f"\n=== {config_name} ===", flush=True)
    t0 = time.time()
    llm = LLM(
        model_path, dist_port=26700 + hash(config_name) % 1000,
        enforce_eager=False, max_num_batched_tokens=8192, max_num_seqs=1,
        max_model_len=2048, gpu_memory_utilization=0.85,
        **extra_config
    )
    print(f"  LLM created in {time.time()-t0:.1f}s", flush=True)
    prompt_ids = llm.tokenizer.encode(prompt_text)
    llm.generate([prompt_ids], SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=4), use_tqdm=False)
    t1 = time.time()
    outputs = llm.generate([prompt_ids], SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=32), use_tqdm=False)
    dt = time.time() - t1
    token_ids = outputs[0]["token_ids"]
    print(f"  Generated {len(token_ids)} tokens in {dt:.1f}s", flush=True)
    llm.exit()
    # Force GPU memory cleanup between cases
    import gc; gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.1f}GB allocated, "
          f"{torch.cuda.memory_reserved()/1e9:.1f}GB reserved", flush=True)
    return token_ids

# 1. Standard mode - ground truth
ref_ids = run_and_get_tokens("STANDARD", {
    "inference_mode": "standard", "enable_heterogeneous": False,
    "enable_speculative": False,
})

# 2. Spec mode with CPU fallback (current config)
spec_cpu_ids = run_and_get_tokens("SPEC + CPU FALLBACK", {
    "inference_mode": "spec", "enable_heterogeneous": True,
    "enable_speculative": True,
    "heterogeneous_slots_per_layer": 32,
    "max_draft_tokens": 8, "draft_top_c": 0,
    "draft_reroute_policy": "round_robin",
    "acceptance_strategy": "greedy",
    "cpu_expert_execution_enabled": True,
    "cpu_expert_pin_memory": True,
    "cpu_expert_backend": "fused",
    "cpu_expert_workspace_max_routes": 16384,
    "cpu_expert_packed_min_routes": 1,
    "cpu_expert_parallel_mode": "serial",
    "cpu_expert_num_threads": 4,
    "spec_profile": True, "engine_profile": True,
    "engine_profile_cuda_sync": True,
    "spec_enable_prefetch": False,
})

# 3. Spec mode WITHOUT CPU execution (GPU fallback only)
spec_gpu_ids = run_and_get_tokens("SPEC + GPU FALLBACK ONLY", {
    "inference_mode": "spec", "enable_heterogeneous": True,
    "enable_speculative": True,
    "heterogeneous_slots_per_layer": 32,
    "max_draft_tokens": 8, "draft_top_c": 0,
    "draft_reroute_policy": "round_robin",
    "acceptance_strategy": "greedy",
    "cpu_expert_execution_enabled": False,   # <-- DISABLED
    "cpu_expert_pin_memory": True,
    "cpu_expert_backend": "fused",
    "cpu_expert_workspace_max_routes": 16384,
    "cpu_expert_packed_min_routes": 1,
    "cpu_expert_parallel_mode": "serial",
    "cpu_expert_num_threads": 4,
    "spec_profile": True, "engine_profile": True,
    "engine_profile_cuda_sync": True,
    "spec_enable_prefetch": False,
})

print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)
for name, ids in [("STANDARD", ref_ids), ("SPEC+CPU", spec_cpu_ids), ("SPEC+GPU", spec_gpu_ids)]:
    match_cpu = ids == spec_cpu_ids
    match_ref = ids == ref_ids
    print(f"{name:<20} len={len(ids):>3} match_ref={match_ref} match_spec_cpu={match_cpu}")
    if ids != ref_ids:
        for i in range(min(len(ids), len(ref_ids))):
            if ids[i] != ref_ids[i]:
                print(f"  First diff at pos {i}: {ids[max(0,i-2):i+3]} vs {ref_ids[max(0,i-2):i+3]}")
                break
