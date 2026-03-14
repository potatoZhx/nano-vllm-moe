import argparse
import json
import time
from collections import defaultdict
from random import Random

import torch

from nanovllm import LLM, SamplingParams


GREEDY_LIKE_TEMPERATURE = 1e-5


def str2bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def _ms(start_event: torch.cuda.Event, end_event: torch.cuda.Event) -> float:
    return float(start_event.elapsed_time(end_event))


def patch_standard(stats: dict):
    import nanovllm.models.qwen3_moe as qwen3_moe

    orig_forward = qwen3_moe.Qwen3MoeFusedSparseMoeBlock.forward

    def wrapped_forward(self, hidden_states):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        out = orig_forward(self, hidden_states)
        e1.record()
        torch.cuda.synchronize()
        stats["standard_moe_forward_ms"] += _ms(e0, e1)
        stats["standard_moe_calls"] += 1
        return out

    qwen3_moe.Qwen3MoeFusedSparseMoeBlock.forward = wrapped_forward


def patch_heterogeneous(stats: dict):
    import nanovllm.models.qwen3_moe as qwen3_moe
    import nanovllm.layers.fuse_moe.heterogeneous as hetero
    import nanovllm.expert.cache as cache_mod

    orig_remap = cache_mod.LayerExpertCache.remap_experts_to_slots

    def remap_wrapped(self, selected_experts):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        out = orig_remap(self, selected_experts)
        e1.record()
        torch.cuda.synchronize()
        stats["hetero_remap_ms"] += _ms(e0, e1)
        stats["hetero_remap_calls"] += 1
        return out

    def _record_cuda_time(name: str, fn):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        out = fn()
        e1.record()
        torch.cuda.synchronize()
        stats[name] += _ms(e0, e1)
        return out

    def hetero_wrapped(hidden_states, selected_experts, routing_weights, expert_cache, cpu_expert_pool, act_fn):
        M, _ = hidden_states.shape
        top_k = routing_weights.size(1)

        flat_selected = selected_experts.reshape(-1)
        flat_weights = routing_weights.reshape(-1)
        stats["hetero_total_routed_tokens"] += int(flat_selected.numel())

        expanded_hidden = _record_cuda_time(
            "hetero_expand_ms",
            lambda: hidden_states.repeat_interleave(top_k, dim=0),
        )
        token_indices = torch.arange(M, device=hidden_states.device, dtype=torch.int64).repeat_interleave(top_k)
        output = torch.zeros_like(hidden_states)

        plan = _record_cuda_time(
            "hetero_plan_ms",
            lambda: hetero.build_moe_execution_plan(flat_selected, expert_cache),
        )

        if plan.gpu_slots.numel() > 0:
            gpu_mask = plan.gpu_mask
            gpu_hidden = _record_cuda_time(
                "hetero_gpu_gather_ms",
                lambda: expanded_hidden[gpu_mask][plan.sort_idx],
            )
            gpu_token_indices = token_indices[gpu_mask][plan.sort_idx]
            gpu_weights = flat_weights[gpu_mask][plan.sort_idx]
            gate_up_buffer, down_buffer = expert_cache.get_layer_buffers()

            gate_up = _record_cuda_time(
                "hetero_fused_gate_up_ms",
                lambda: hetero.fused_moe_linear(gpu_hidden, gate_up_buffer, plan.m_sizes),
            )
            gpu_expert_out = _record_cuda_time(
                "hetero_fused_down_ms",
                lambda: hetero.fused_moe_linear(act_fn(gate_up), down_buffer, plan.m_sizes),
            )
            _record_cuda_time(
                "hetero_scatter_ms",
                lambda: output.index_add_(0, gpu_token_indices, gpu_expert_out * gpu_weights.unsqueeze(-1)),
            )

        if plan.gpu_slots.numel() < flat_selected.numel():
            cpu_indices = torch.nonzero(~plan.gpu_mask, as_tuple=False).flatten()
        else:
            cpu_indices = None

        if cpu_indices is not None and cpu_indices.numel() > 0:
            stats["hetero_cpu_fallback_calls"] += 1
            stats["hetero_cpu_tokens"] += int(cpu_indices.numel())
            # Keep fallback path behavior identical to implementation.
            if cpu_expert_pool is None:
                raise RuntimeError("Missing cpu_expert_pool for uncached expert fallback.")
            cpu_hidden = expanded_hidden[cpu_indices]
            cpu_experts = flat_selected[cpu_indices]
            cpu_token_indices = token_indices[cpu_indices]
            cpu_weights = flat_weights[cpu_indices]
            for expert_idx in cpu_experts.unique().tolist():
                expert_mask = cpu_experts == expert_idx
                h = cpu_hidden[expert_mask]
                params = cpu_expert_pool[expert_idx]
                gate_up_weight = params["gate_up"].to(device=hidden_states.device, dtype=hidden_states.dtype, non_blocking=True)
                down_weight = params["down"].to(device=hidden_states.device, dtype=hidden_states.dtype, non_blocking=True)
                gate_up = torch.nn.functional.linear(h, gate_up_weight)
                out = torch.nn.functional.linear(act_fn(gate_up), down_weight)
                out = out * cpu_weights[expert_mask].unsqueeze(-1)
                output.index_add_(0, cpu_token_indices[expert_mask], out)

        stats["hetero_moe_calls"] += 1
        return output

    orig_block_forward = qwen3_moe.Qwen3MoeHeterogeneousSparseMoeBlock.forward

    def block_forward_wrapped(self, hidden_states):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        out = orig_block_forward(self, hidden_states)
        e1.record()
        torch.cuda.synchronize()
        stats["hetero_block_forward_ms"] += _ms(e0, e1)
        stats["hetero_block_calls"] += 1
        return out

    cache_mod.LayerExpertCache.remap_experts_to_slots = remap_wrapped
    hetero.heterogeneous_moe_forward = hetero_wrapped
    qwen3_moe.heterogeneous_moe_forward = hetero_wrapped
    qwen3_moe.Qwen3MoeHeterogeneousSparseMoeBlock.forward = block_forward_wrapped


def run_case(args: argparse.Namespace, enable_heterogeneous: bool) -> tuple[dict, dict]:
    rng = Random(args.seed)
    stats = defaultdict(float)
    if enable_heterogeneous:
        patch_heterogeneous(stats)
    else:
        patch_standard(stats)

    llm = LLM(
        args.model_path,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        enable_heterogeneous=enable_heterogeneous,
        heterogeneous_slots_per_layer=args.slots_per_layer,
    )

    prompts = [
        [rng.randint(0, 10000) for _ in range(rng.randint(args.min_input_len, args.max_input_len))]
        for _ in range(args.num_seqs)
    ]
    sampling_params = [
        SamplingParams(
            temperature=GREEDY_LIKE_TEMPERATURE,
            ignore_eos=True,
            max_tokens=rng.randint(args.min_output_len, args.max_output_len),
        )
        for _ in range(args.num_seqs)
    ]

    # Warmup and reset stats to profile only measured run.
    llm.generate([[1, 2, 3, 4]], SamplingParams(temperature=GREEDY_LIKE_TEMPERATURE, max_tokens=4), use_tqdm=False)
    stats.clear()

    t0 = time.time()
    llm.generate(prompts, sampling_params, use_tqdm=False)
    elapsed = time.time() - t0
    llm.exit()

    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    base = {
        "enable_heterogeneous": enable_heterogeneous,
        "total_tokens": total_tokens,
        "elapsed_sec": elapsed,
        "throughput_tok_s": total_tokens / elapsed,
    }
    return base, dict(stats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug profile for standard vs heterogeneous path")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--enable-heterogeneous", type=str2bool, required=True)
    parser.add_argument("--slots-per-layer", type=int, default=0)
    parser.add_argument("--num-seqs", type=int, default=64)
    parser.add_argument("--min-input-len", type=int, default=64)
    parser.add_argument("--max-input-len", type=int, default=512)
    parser.add_argument("--min-output-len", type=int, default=32)
    parser.add_argument("--max-output-len", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enforce-eager", type=str2bool, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base, profile = run_case(args, enable_heterogeneous=args.enable_heterogeneous)
    report = {
        "enable_heterogeneous": args.enable_heterogeneous,
        "base": base,
        "profile": profile,
    }
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
