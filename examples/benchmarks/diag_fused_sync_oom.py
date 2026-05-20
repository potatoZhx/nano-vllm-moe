"""Diagnose fused_sync OOM — probe memory at each stage with lower mem util."""
import gc, os, subprocess, sys, torch

os.environ["PYTHONUNBUFFERED"] = "1"

def mem(label=""):
    rss = subprocess.check_output(["grep", "VmRSS", "/proc/self/status"]).decode().strip()
    gpu_alloc = torch.cuda.memory_allocated() / 1024**3
    gpu_res = torch.cuda.memory_reserved() / 1024**3
    print(f"  [{label}] CPU: {rss}  GPU: {gpu_alloc:.1f}/{gpu_res:.1f} GiB", flush=True)


def main():
    mem_util = float(sys.argv[1]) if len(sys.argv) > 1 else 0.80
    bucket_steps = sys.argv[2] if len(sys.argv) > 2 else "1,4,16,64,256,512"
    print(f"gpu_memory_utilization={mem_util} bucket_steps={bucket_steps}", flush=True)

    from nanovllm import LLM, SamplingParams

    print("Creating LLM...", flush=True)
    llm = LLM(
        "/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B",
        dist_port=5107,
        max_num_batched_tokens=16384,
        max_num_seqs=512,
        max_model_len=4096,
        gpu_memory_utilization=mem_util,
        inference_mode="spec",
        enable_heterogeneous=True,
        enable_speculative=True,
        draft_top_c=1,
        max_draft_tokens=4,
        slots_per_layer=16,
        cpu_expert_execution_enabled=True,
        cpu_expert_backend="fused",
        cpu_expert_workspace_max_routes=262144,
        draft_cuda_graph_cpu_backend="fused_sync",
        draft_cuda_graph_bucket_steps=bucket_steps,
        spec_enable_prefetch=False,
        engine_profile=True,
        enforce_eager=False,
    )
    mem("after_llm")

    print("Warmup...", flush=True)
    llm.generate(["warmup prompt "],
                 SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True),
                 use_tqdm=False)
    mem("after_warmup")

    print("Benchmark...", flush=True)
    llm.generate(["test prompt "] * 4,
                 SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True),
                 use_tqdm=False)
    mem("after_bench")

    print("DONE", flush=True)
    llm.exit()


if __name__ == "__main__":
    main()
