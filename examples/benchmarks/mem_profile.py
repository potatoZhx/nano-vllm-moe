"""Profile GPU memory with different configurations."""
import sys, torch

def gpu_status(label):
    alloc = torch.cuda.memory_allocated() / 1024**3
    res = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free = total - alloc
    print(f"  [{label}] GPU alloc={alloc:.2f}GiB res={res:.2f}GiB total={total:.0f}GiB free={free:.2f}GiB", flush=True)
    return alloc, res, free

def main():
    mem_util = float(sys.argv[1]) if len(sys.argv) > 1 else 0.80
    backend = sys.argv[2] if len(sys.argv) > 2 else "none"
    bucket_steps = sys.argv[3] if len(sys.argv) > 3 else "1,2,3,4,5"
    steps_list = [int(x) for x in bucket_steps.split(",")]

    print(f"mem_util={mem_util} backend={backend} bucket_steps={steps_list}", flush=True)

    from nanovllm import LLM, SamplingParams

    print("Creating LLM...", flush=True)
    llm = LLM(
        "/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B",
        dist_port=5111,
        gpu_memory_utilization=mem_util,
        inference_mode="spec",
        enable_heterogeneous=True,
        enable_speculative=True,
        draft_top_c=1 if backend != "none" else 0,
        max_draft_tokens=4,
        slots_per_layer=16,
        cpu_expert_execution_enabled=True,
        cpu_expert_backend="fused",
        cpu_expert_workspace_max_routes=262144,
        draft_cuda_graph_cpu_backend=backend,
        draft_cuda_graph_bucket_steps=steps_list,
        spec_enable_prefetch=False,
        engine_profile=True,
        enforce_eager=False,
    )
    a, r, f = gpu_status("after_llm")

    print("Warmup...", flush=True)
    llm.generate(
        ["warmup prompt "],
        SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True),
        use_tqdm=False,
    )
    gpu_status("after_warmup")

    print("Benchmark...", flush=True)
    llm.generate(
        ["test prompt "] * 4,
        SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True),
        use_tqdm=False,
    )
    gpu_status("after_bench")

    print("DONE", flush=True)
    llm.exit()


if __name__ == "__main__":
    main()
