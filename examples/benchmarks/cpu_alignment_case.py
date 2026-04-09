import argparse
import json
import random
from pathlib import Path

from nanovllm import LLM, SamplingParams
import nanovllm.layers.fuse_moe.heterogeneous as hetero_mod


def str2bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def build_prompts(num_seqs: int, prompt_len: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    prompts = []
    for _ in range(num_seqs):
        prompts.append([rng.randint(100, 120000) for _ in range(prompt_len)])
    return prompts


def build_text_prompts(num_seqs: int, prompt_len: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    base = [
        "Explain sparse MoE routing with a concrete deterministic example.",
        "Describe prefill vs decode and why their throughput is different.",
        "Summarize CPU/GPU mixed expert execution and potential trade-offs.",
        "List practical checks for deterministic inference regression debugging.",
    ]
    prompts: list[str] = []
    for i in range(num_seqs):
        words = []
        while len(words) < prompt_len:
            sentence = base[i % len(base)]
            sentence += f" Context-{rng.randint(0, 999)}."
            words.extend(sentence.split())
        prompts.append(" ".join(words[:prompt_len]))
    return prompts


def remap_cache_to_high_experts(llm: LLM) -> None:
    for layer in llm.model_runner.model.model.layers:
        mlp = getattr(layer, "mlp", None)
        if not hasattr(mlp, "expert_cache") or mlp.expert_cache is None:
            continue
        cache = mlp.expert_cache
        pool = mlp.cpu_expert_pool
        high_ids = list(range(cache.num_experts - 1, cache.num_experts - cache.num_slots - 1, -1))
        for slot_idx, expert_idx in enumerate(high_ids):
            params = pool[expert_idx]
            cache.put_to_slot(slot_idx, expert_idx, params["gate_up"], params["down"])


def run_case(args: argparse.Namespace) -> dict:
    if args.prompt_kind == "text":
        prompts = build_text_prompts(args.num_seqs, args.prompt_len, args.seed)
    else:
        prompts = build_prompts(args.num_seqs, args.prompt_len, args.seed)
    sp = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=args.max_tokens)

    cpu_counter = {"calls": 0, "routes": 0}
    orig_cpu_exec = hetero_mod._run_real_cpu_expert_execution

    def wrapped_cpu_exec(
        hidden_states,
        output,
        flat_weights,
        top_k,
        cpu_indices,
        cpu_task_expert_ids,
        cpu_task_offsets,
        flat_selected_original,
        cpu_expert_pool,
        act_fn,
        cpu_expert_parallel_mode="serial",
        cpu_expert_num_threads=4,
    ):
        cpu_counter["calls"] += 1
        cpu_counter["routes"] += int(cpu_indices.numel())
        return orig_cpu_exec(
            hidden_states,
            output,
            flat_weights,
            top_k,
            cpu_indices,
            cpu_task_expert_ids,
            cpu_task_offsets,
            flat_selected_original,
            cpu_expert_pool,
            act_fn,
            cpu_expert_parallel_mode=cpu_expert_parallel_mode,
            cpu_expert_num_threads=cpu_expert_num_threads,
        )

    if args.mode == "heter":
        hetero_mod._run_real_cpu_expert_execution = wrapped_cpu_exec

    try:
        llm = LLM(
            args.model_path,
            inference_mode=args.mode,
            enable_heterogeneous=(args.mode == "heter"),
            enable_speculative=False,
            heterogeneous_slots_per_layer=args.slots_per_layer,
            cpu_expert_execution_enabled=args.cpu_expert_execution_enabled,
            cpu_expert_parallel_mode="serial",
            cpu_expert_num_threads=args.cpu_expert_num_threads,
            cpu_gpu_parallel_execution_enabled=args.cpu_gpu_parallel_execution_enabled,
            cpu_gpu_parallel_min_cpu_route_ratio=args.cpu_gpu_parallel_min_cpu_route_ratio,
            enforce_eager=True,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            engine_profile=True,
            dist_port=args.dist_port,
        )

        if args.mode == "heter" and args.remap_cache_high_ids:
            remap_cache_to_high_experts(llm)

        outputs = llm.generate(prompts, sp, use_tqdm=False)
        profile = llm.get_profile(reset=True)
        llm.exit()
    finally:
        hetero_mod._run_real_cpu_expert_execution = orig_cpu_exec

    return {
        "mode": args.mode,
        "slots_per_layer": args.slots_per_layer,
        "seed": args.seed,
        "num_seqs": args.num_seqs,
        "prompt_len": args.prompt_len,
        "max_tokens": args.max_tokens,
        "generated_token_ids": [x["token_ids"] for x in outputs],
        "cpu_exec_calls": cpu_counter["calls"],
        "cpu_exec_routes": cpu_counter["routes"],
        "engine_profile": profile,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one deterministic alignment case for standard/heter mode")
    p.add_argument("--model-path", required=True)
    p.add_argument("--mode", choices=["standard", "heter"], required=True)
    p.add_argument("--slots-per-layer", type=int, default=0)
    p.add_argument("--cpu-expert-execution-enabled", type=str2bool, default=False)
    p.add_argument("--cpu-expert-num-threads", type=int, default=4)
    p.add_argument("--cpu-gpu-parallel-execution-enabled", type=str2bool, default=True)
    p.add_argument("--cpu-gpu-parallel-min-cpu-route-ratio", type=float, default=0.7)
    p.add_argument("--remap-cache-high-ids", type=str2bool, default=False)
    p.add_argument("--num-seqs", type=int, default=4)
    p.add_argument("--prompt-len", type=int, default=96)
    p.add_argument("--prompt-kind", choices=["random_ids", "text"], default="random_ids")
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-model-len", type=int, default=256)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.99)
    p.add_argument("--dist-port", type=int, required=True)
    p.add_argument("--output", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = run_case(args)
    text = json.dumps(report, ensure_ascii=True)
    print(text)
    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
