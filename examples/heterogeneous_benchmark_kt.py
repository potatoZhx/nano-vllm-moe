#!/usr/bin/env python3
"""Like heterogeneous_benchmark_case.py but switches to kt_kernel after loading."""
import sys, os, json, time, argparse, hashlib, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.llm_engine import LLMEngine

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--mode", default="spec")
    p.add_argument("--slots-per-layer", type=int, default=32)
    p.add_argument("--num-seqs", type=int, default=1)
    p.add_argument("--input-len", type=int, default=12)
    p.add_argument("--output-len", type=int, default=6)
    p.add_argument("--max-num-batched-tokens", type=int, default=256)
    p.add_argument("--max-num-seqs", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=128)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-draft-tokens", type=int, default=4)
    p.add_argument("--draft-top-c", type=int, default=128)
    p.add_argument("--cpu-expert-backend", default="kt_kernel")
    p.add_argument("--spec-enable-prefetch", type=lambda x: x.lower() == "true", default=False)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--enforce-eager", default=False)
    p.add_argument("--dist-port", type=int, default=12345)
    p.add_argument("--output", default=None)
    p.add_argument("--profile", action="store_true", default=True)
    args = p.parse_args()

    MODEL = args.model_path
    total_t0 = time.time()

    # Load with torch backend first
    llm = LLMEngine(MODEL,
        inference_mode=args.mode, enable_heterogeneous=True,
        heterogeneous_slots_per_layer=args.slots_per_layer,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs, max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager, gpu_memory_utilization=args.gpu_memory_utilization,
        max_draft_tokens=args.max_draft_tokens, draft_top_c=args.draft_top_c,
        cpu_expert_execution_enabled=True, cpu_expert_backend="torch",
        spec_enable_prefetch=args.spec_enable_prefetch,
        dist_port=args.dist_port,
    )

    # Switch to kt_kernel
    from nanovllm.layers.fuse_moe.kt_backend import KtKernelCpuMoeBackend
    model = llm.model_runner.model
    switched = 0
    for layer in model.model.layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not hasattr(mlp, "cpu_expert_pool") or mlp.cpu_expert_pool is None:
            continue
        if not hasattr(mlp, "moe_intermediate_size"):
            continue
        gm = torch.zeros(mlp.num_experts, dtype=torch.bool)
        gm[:args.slots_per_layer] = True
        mlp.cpu_backend = KtKernelCpuMoeBackend(
            layer_idx=mlp.layer_idx,
            moe_intermediate_size=mlp.moe_intermediate_size,
            hidden_size=2048, num_experts=128, num_experts_per_tok=8,
            gpu_expert_mask=gm, weight_path=MODEL,
        )
        mlp.cpu_expert_backend_name = "kt_kernel"
        switched += 1

    # Benchmark
    sp = SamplingParams(temperature=args.temperature, max_tokens=args.output_len, ignore_eos=False)
    prompt = "Hello world, this is a benchmark test."
    _ = llm.generate([prompt], sp)  # warmup

    t1 = time.time()
    result = llm.generate([prompt], sp)
    gen_s = time.time() - t1

    if isinstance(result, dict):
        token_ids = result.get("token_ids", [])
    elif result and hasattr(result[0], "token_ids"):
        token_ids = result[0].token_ids
    else:
        token_ids = []
    digest = hashlib.sha256(",".join(str(t) for t in token_ids).encode()).hexdigest()
    profile = llm.get_profile(reset=False)
    ep = profile.get("engine_profile", {})

    out = {
        "backend": "kt_kernel", "slots": args.slots_per_layer, "switched_layers": switched,
        "total_s": time.time() - total_t0, "gen_s": gen_s,
        "verify_ms": ep.get("model_run_verify_total_ms", 0),
        "cpu_compute_ms": ep.get("model_verify_cpu_compute_ms", 0),
        "cpu_merge_ms": ep.get("model_verify_cpu_to_gpu_merge_ms", 0),
        "gpu_moe_ms": ep.get("model_verify_gpu_compute_ms", 0),
        "plan_ms": ep.get("model_verify_plan_ms", 0),
        "prefill_ms": ep.get("prefill_forward_ms", 0),
        "verify_forward_ms": ep.get("verify_forward_ms", 0),
        "draft_forward_ms": ep.get("draft_forward_ms", 0),
        "tok_s": profile.get("throughput_output_tok_s", 0),
        "tokens": token_ids, "digest": digest[:16],
    }
    if args.output:
        json.dump(out, open(args.output, "w"), indent=2)
    print(json.dumps(out, indent=2))
    del llm; torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
