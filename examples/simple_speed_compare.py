"""
运行示例：
1) 复用 simple_inference 风格的单条/批量推理示例；
2) 输出 nano-vllm 的 single/batch 推理速度（tokens/s）；
3) 方便与 on_device_sd/demo/examples/simple_speculative_speed_compare.py 对比。

示例：
CUDA_VISIBLE_DEVICES=3 python /zx_data1/sparsity/nano-vllm-moe/examples/simple_speed_compare.py \
  --model-path /zx_data1/models/Qwen--Qwen3-30B-A3B-Base \
  --max-new-tokens 50 \
  --enforce-eager \
  --warmup 1
"""

import argparse
import random
import time
from typing import Dict, List

import numpy as np
import torch

from nanovllm import LLM, SamplingParams


PROMPTS = [
    "Once upon a time",
    "In a distant future, humans",
    "Write a short poem about the sea",
]


def _count_input_tokens(llm: LLM, prompt: str) -> int:
    return len(llm.tokenizer.encode(prompt))


def _extract_ttft_tpot(
    debug: Dict[str, object] | None,
    total_time_s: float,
    output_tokens: int,
) -> Dict[str, float | None]:
    if not debug:
        return {"ttft_s": None, "tpot_s": None}

    step_records = debug.get("step_records", [])
    if not isinstance(step_records, list) or not step_records:
        return {"ttft_s": None, "tpot_s": None}

    cumulative_s = 0.0
    ttft_s: float | None = None
    for rec in step_records:
        step_wall_s = float(rec.get("step_wall_s", 0.0))
        cumulative_s += step_wall_s
        phase = rec.get("phase", "")
        if phase == "decode":
            ttft_s = cumulative_s
            break

    if ttft_s is None:
        return {"ttft_s": None, "tpot_s": None}

    if output_tokens <= 1:
        tpot_s = None
    else:
        tpot_s = max(total_time_s - ttft_s, 0.0) / float(output_tokens - 1)

    return {"ttft_s": ttft_s, "tpot_s": tpot_s}


def _format_latency_metric(value: float | None) -> str:
    if value is None:
        return "N/A (enable --debug-timing)"
    return f"{value:.4f}s"


def _run_once(
    llm: LLM,
    prompt: str,
    sampling_params: SamplingParams,
    debug_timing: bool,
) -> Dict[str, object]:
    input_tokens = _count_input_tokens(llm, prompt)
    start = time.perf_counter()
    if debug_timing:
        outputs, debug_stats = llm.generate([prompt], sampling_params, use_tqdm=False, return_debug_stats=True)
        output = outputs[0]
    else:
        output = llm.generate([prompt], sampling_params, use_tqdm=False)[0]
        debug_stats = None
    elapsed = time.perf_counter() - start
    token_ids = output["token_ids"]
    output_tokens = len(token_ids)
    latency = _extract_ttft_tpot(debug_stats, elapsed, output_tokens)
    return {
        "time": elapsed,
        "input_tokens": input_tokens,
        "token_ids": token_ids,
        "text": output["text"],
        "tokens": output_tokens,
        "input_tps": input_tokens / max(elapsed, 1e-8),
        "output_tps": output_tokens / max(elapsed, 1e-8),
        "total_tps": (input_tokens + output_tokens) / max(elapsed, 1e-8),
        "ttft_s": latency["ttft_s"],
        "tpot_s": latency["tpot_s"],
        "debug": debug_stats,
    }


def _run_batch(
    llm: LLM,
    prompts: List[str],
    sampling_params: SamplingParams,
    debug_timing: bool,
) -> Dict[str, object]:
    input_token_counts = [_count_input_tokens(llm, prompt) for prompt in prompts]
    total_input_tokens = sum(input_token_counts)
    batch_sampling_params = [sampling_params] * len(prompts)
    start = time.perf_counter()
    if debug_timing:
        outputs, debug_stats = llm.generate(prompts, batch_sampling_params, use_tqdm=False, return_debug_stats=True)
    else:
        outputs = llm.generate(prompts, batch_sampling_params, use_tqdm=False)
        debug_stats = None
    elapsed = time.perf_counter() - start

    token_counts = [len(item["token_ids"]) for item in outputs]
    total_tokens = sum(token_counts)
    latency = _extract_ttft_tpot(debug_stats, elapsed, total_tokens)
    return {
        "time": elapsed,
        "outputs": outputs,
        "input_token_counts": input_token_counts,
        "input_tokens": total_input_tokens,
        "token_counts": token_counts,
        "tokens": total_tokens,
        "input_tps": total_input_tokens / max(elapsed, 1e-8),
        "output_tps": total_tokens / max(elapsed, 1e-8),
        "total_tps": (total_input_tokens + total_tokens) / max(elapsed, 1e-8),
        "ttft_s": latency["ttft_s"],
        "tpot_s": latency["tpot_s"],
        "debug": debug_stats,
    }


def _print_debug_timing(tag: str, debug: Dict[str, object] | None) -> None:
    if not debug:
        return
    print(f"\n[{tag}] DEBUG TIMING")
    for phase in ["prefill", "decode"]:
        data = debug.get(phase, {})
        if not data:
            continue
        print(
            f"[{tag}][{phase}] steps={int(data.get('steps', 0))}, "
            f"schedule={data.get('schedule_s', 0.0):.4f}s, "
            f"prepare={data.get('prepare_s', 0.0):.4f}s, "
            f"prepare_sample={data.get('prepare_sample_s', 0.0):.4f}s, "
            f"model={data.get('model_s', 0.0):.4f}s, "
            f"model_embed={data.get('model_embed_s', 0.0):.4f}s, "
            f"model_attn={data.get('model_attn_s', 0.0):.4f}s, "
            f"model_moe={data.get('model_moe_s', 0.0):.4f}s, "
            f"model_norm={data.get('model_norm_s', 0.0):.4f}s, "
            f"sample={data.get('sample_s', 0.0):.4f}s, "
            f"reset_context={data.get('reset_context_s', 0.0):.4f}s, "
            f"postprocess={data.get('postprocess_s', 0.0):.4f}s"
        )

        layer_attn: List[float] = []
        layer_moe: List[float] = []
        for rec in debug.get("step_records", []):
            if rec.get("phase") != phase:
                continue
            modules = rec.get("model_modules", {})
            per_layer = modules.get("per_layer", [])
            if not per_layer:
                continue
            if not layer_attn:
                layer_attn = [0.0] * len(per_layer)
                layer_moe = [0.0] * len(per_layer)
            for idx, item in enumerate(per_layer):
                layer_attn[idx] += float(item.get("attn_s", 0.0))
                layer_moe[idx] += float(item.get("moe_s", 0.0))

        if layer_attn:
            top_attn = sorted(enumerate(layer_attn), key=lambda x: x[1], reverse=True)[:5]
            top_moe = sorted(enumerate(layer_moe), key=lambda x: x[1], reverse=True)[:5]
            print(
                f"[{tag}][{phase}] top_attn_layers="
                + ", ".join([f"L{i}:{v:.4f}s" for i, v in top_attn])
            )
            print(
                f"[{tag}][{phase}] top_moe_layers="
                + ", ".join([f"L{i}:{v:.4f}s" for i, v in top_moe])
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="nano-vllm speed comparison example")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/zx_data1/models/Qwen--Qwen3-30B-A3B-Base",
        help="HuggingFace model path",
    )
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--warmup", type=int, default=1, help="warmup runs")
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="ignore eos and always try to generate max_new_tokens",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enforce-eager", action="store_true", help="disable cuda graph")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--debug-timing", action="store_true")
    parser.add_argument("--debug-sync-cuda", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this example")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        ignore_eos=args.ignore_eos,
    )

    print("=" * 80)
    print("NANO-VLLM SPEED EXAMPLE")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Prompts: {len(PROMPTS)}")
    print(
        f"max_new_tokens={args.max_new_tokens}, temperature={args.temperature}, "
        f"ignore_eos={args.ignore_eos}, warmup={args.warmup}"
    )
    print(
        f"tensor_parallel_size={args.tensor_parallel_size}, enforce_eager={args.enforce_eager}, "
        f"max_model_len={args.max_model_len}, max_num_seqs={args.max_num_seqs}, "
        f"max_num_batched_tokens={args.max_num_batched_tokens}"
    )

    llm = LLM(
        args.model_path,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )
    if args.debug_timing:
        llm.set_debug_timing(True, sync_cuda=args.debug_sync_cuda)

    for _ in range(max(0, args.warmup)):
        _ = llm.generate([PROMPTS[0]], sampling_params, use_tqdm=False)

    print("\n" + "-" * 80)
    print("Running SINGLE request")
    print("-" * 80)
    single_stats = _run_once(llm, PROMPTS[0], sampling_params, args.debug_timing)
    print(f"[Single] Generated {single_stats['tokens']} tokens")
    print(f"[Single] Output IDs: {single_stats['token_ids']}")
    print(f"[Single] Decoded: {single_stats['text']}")

    print("\n" + "-" * 80)
    print("Running BATCH requests")
    print("-" * 80)
    batch_stats = _run_batch(llm, PROMPTS, sampling_params, args.debug_timing)
    for idx, output in enumerate(batch_stats["outputs"]):
        print(f"[Batch Prompt {idx}] Generated {batch_stats['token_counts'][idx]} tokens")
        print(f"[Batch Prompt {idx}] Output IDs: {output['token_ids']}")
        print(f"[Batch Prompt {idx}] Decoded: {output['text']}")

    print("\n" + "=" * 80)
    print("SPEED SUMMARY (NANO-VLLM)")
    print("=" * 80)
    print(
        "[Single] "
        f"input_tokens={single_stats['input_tokens']}, "
        f"tokens={single_stats['tokens']}, "
        f"time={single_stats['time']:.4f}s, "
        f"input_throughput={single_stats['input_tps']:.2f} tok/s, "
        f"output_throughput={single_stats['output_tps']:.2f} tok/s, "
        f"total_throughput={single_stats['total_tps']:.2f} tok/s, "
        f"ttft={_format_latency_metric(single_stats['ttft_s'])}, "
        f"tpot={_format_latency_metric(single_stats['tpot_s'])}"
    )
    print(
        "[Batch]  "
        f"input_tokens={batch_stats['input_tokens']}, "
        f"tokens={batch_stats['tokens']}, "
        f"time={batch_stats['time']:.4f}s, "
        f"input_throughput={batch_stats['input_tps']:.2f} tok/s, "
        f"output_throughput={batch_stats['output_tps']:.2f} tok/s, "
        f"total_throughput={batch_stats['total_tps']:.2f} tok/s, "
        f"ttft={_format_latency_metric(batch_stats['ttft_s'])}, "
        f"tpot={_format_latency_metric(batch_stats['tpot_s'])}"
    )
    if args.debug_timing:
        _print_debug_timing("Single", single_stats.get("debug"))
        _print_debug_timing("Batch", batch_stats.get("debug"))


if __name__ == "__main__":
    main()