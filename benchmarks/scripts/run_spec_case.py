#!/usr/bin/env python3
"""Run a single spec benchmark case and print key metrics."""
import sys, json, os, time
from nanovllm.config import Config

def main():
    model = sys.argv[1]       # model path
    backend = sys.argv[2]     # torch | fused | kt_kernel
    slots = int(sys.argv[3])  # slots per layer
    out = sys.argv[4]         # output json path

    config_kwargs = {
        "model": model,
        "inference_mode": "spec",
        "enable_heterogeneous": True,
        "enable_speculative": True,
        "heterogeneous_slots_per_layer": slots,
        "max_num_batched_tokens": 512,
        "max_num_seqs": 4,
        "max_model_len": 512,
        "gpu_memory_utilization": 0.85,
        "max_draft_tokens": 4,
        "draft_top_c": 128,
        "cpu_expert_execution_enabled": True,
        "cpu_expert_backend": backend,
        "cpu_expert_packed_min_routes": 1,
        "cpu_expert_parallel_mode": "serial",
        "cpu_gpu_parallel_execution_enabled": "auto",
        "spec_profile": True,
        "engine_profile": True,
        "engine_profile_cuda_sync": True,
        "spec_enable_prefetch": False,
        "enforce_eager": False,
        "dist_port": 12345,
    }

    t0 = time.time()
    config = Config(model, **config_kwargs)

    from nanovllm.engine.llm_engine import LLMEngine
    from nanovllm.sampling_params import SamplingParams

    llm = LLMEngine(config)
    sp = SamplingParams(temperature=0.0, max_tokens=6, ignore_eos=False)

    # Warmup
    prompt = "Warmup request for benchmark."
    for _ in range(2):
        llm.generate([prompt], sp)

    # Benchmark
    t1 = time.time()
    result = llm.generate([prompt], sp)
    t2 = time.time()

    token_ids = result[0].token_ids if result else []
    digest = __import__("hashlib").sha256(
        ",".join(str(t) for t in token_ids).encode()
    ).hexdigest()

    profile = llm.get_profile(reset=False)
    ep = profile.get("engine_profile", {})

    metrics = {
        "backend": backend,
        "slots": slots,
        "verify_ms": ep.get("model_run_verify_total_ms", 0),
        "cpu_compute_ms": ep.get("model_verify_cpu_compute_ms", 0),
        "cpu_merge_ms": ep.get("model_verify_cpu_to_gpu_merge_ms", 0),
        "gpu_moe_ms": ep.get("model_verify_gpu_compute_ms", 0),
        "plan_ms": ep.get("model_verify_plan_ms", 0),
        "cpu_route_ratio": ep.get("cpu_route_ratio", 0),
        "tokens": token_ids,
        "digest": digest[:16],
        "total_sec": t2 - t1,
    }
    json.dump(metrics, open(out, "w"), indent=2)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
