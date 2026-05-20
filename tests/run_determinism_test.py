#!/usr/bin/env python3
"""Quick determinism test: prefetch ON vs OFF producing same output."""
import json, os, time, torch

MODEL_PATH = "/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B"

from nanovllm import LLM, SamplingParams

def run_one(label, prefetch_enabled, dist_port):
    print(f"\n=== {label} ===", flush=True)
    llm = LLM(
        model=MODEL_PATH,
        inference_mode="spec",
        enable_heterogeneous=True,
        enable_speculative=True,
        max_num_batched_tokens=4096,
        max_num_seqs=2,
        max_model_len=2048,
        max_draft_tokens=4,
        draft_top_c=2,
        acceptance_strategy="greedy",
        enforce_eager=True,
        spec_verify_eager=True,
        spec_enable_prefetch=True,
        cache_strategy="lru",
        prefetch_strategy="history_window",
        prefetch_step_budget=4,
        prefetch_max_inflight=8,
        prefetch_verify_wait_ms=2.0,
        prefetch_global_queue_capacity=4096,
        prefetch_verify_layer_enabled=prefetch_enabled,
        prefetch_verify_layer_safety_ratio=0.8,
        prefetch_verify_layer_min_compute_ms=0.05,
        prefetch_verify_layer_transfer_bandwidth_gbps=12.0,
        prefetch_verify_layer_max_budget=2,
        heterogeneous_slots_per_layer=64,
        engine_profile=True,
        engine_profile_cuda_sync=False,
        perf_profile_level="basic",
        cpu_expert_backend="fused",
        dist_port=dist_port,
    )
    sp = SamplingParams(max_tokens=32, temperature=0.0)
    outputs = llm.generate(["Hello, how are you?"], sp)
    tokens = outputs[0]["token_ids"]
    profile = llm.model_runner.get_profile(reset=False)

    print(f"Tokens ({len(tokens)}): {tokens[:20]}...")
    print(f"verify_layer_prefetch_hook_count: {profile.get('verify_layer_prefetch_hook_count', 'N/A')}")
    print(f"verify_layer_prefetch_submit_count: {profile.get('verify_layer_prefetch_submit_count', 'N/A')}")
    print(f"verify_layer_prefetch_publish_count: {profile.get('verify_layer_prefetch_publish_count', 'N/A')}")
    print(f"verify_forward_ms: {profile.get('verify_forward_ms', 0):.2f}")

    del llm
    torch.cuda.empty_cache()
    return tokens, profile

# Run both
tokens_on, prof_on = run_one("PREFETCH ON", True, 29500)
tokens_off, prof_off = run_one("PREFETCH OFF", False, 29501)

match = tokens_on == tokens_off
print(f"\n{'='*40}")
print(f"DETERMINISM: {'PASS' if match else 'FAIL'}")
print(f"Tokens ON:  {tokens_on[:10]}...")
print(f"Tokens OFF: {tokens_off[:10]}...")
print(f"Match: {match}")
print(f"Prefetch hooks ON:  {prof_on.get('verify_layer_prefetch_hook_count', 'N/A')}")
print(f"Prefetch hooks OFF: {prof_off.get('verify_layer_prefetch_hook_count', 'N/A')}")
