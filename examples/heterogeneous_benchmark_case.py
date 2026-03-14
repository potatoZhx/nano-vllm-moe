import argparse
import json
import time
import hashlib
from random import Random

import torch

from nanovllm import LLM, SamplingParams


BASE_PROMPTS = [
    "Summarize the key reasons why sparse MoE models can improve inference efficiency compared with dense models.",
    "Write a short explanation of how top-k routing works in mixture-of-experts layers, with one simple example.",
    "Given a deployment with limited GPU memory, propose a practical strategy to balance latency and memory usage.",
    "Explain the difference between prefill and decode phases in LLM inference and why throughput differs.",
    "List three risks when optimizing kernels for speed and how to validate model correctness after optimization.",
    "Provide a concise troubleshooting checklist for unexpected throughput regressions after code refactoring.",
    "Describe how caching expert weights on GPU can reduce transfer overhead in heterogeneous execution.",
    "Explain why deterministic seeds are important in benchmark experiments and what they do not guarantee.",
]


def _build_meaningful_prompts(args: argparse.Namespace, rng: Random) -> list[str]:
    prompts: list[str] = []
    for i in range(args.num_seqs):
        target_words = rng.randint(args.min_input_len, args.max_input_len)
        base = BASE_PROMPTS[i % len(BASE_PROMPTS)]
        context_lines = []
        context_words = 0
        step = 0
        while context_words < target_words:
            score = (i * 37 + step * 11 + args.seed) % 101
            line = (
                f"Observation {step + 1}: request batch {i % 8}, load score {score}, "
                f"focus on stable quality under performance tuning."
            )
            context_lines.append(line)
            context_words += len(line.split())
            step += 1
        prompt = (
            f"Task:\n{base}\n\n"
            f"Context:\n{' '.join(context_lines)}\n\n"
            "Please answer with clear reasoning and a brief conclusion."
        )
        prompts.append(prompt)
    return prompts


def str2bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def _hash_token_ids(token_ids: list[int]) -> str:
    # Use a stable digest to compare outputs across runs without dumping huge payloads.
    payload = ",".join(str(token) for token in token_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_many_token_ids(seqs: list[list[int]]) -> str:
    per_seq = [_hash_token_ids(seq) for seq in seqs]
    return hashlib.sha256("|".join(per_seq).encode("utf-8")).hexdigest()


def _count_prompt_tokens(llm: LLM, prompts: list[str]) -> int:
    return sum(len(llm.tokenizer.encode(prompt)) for prompt in prompts)


def run_case(args: argparse.Namespace) -> dict:
    rng = Random(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    llm = LLM(
        args.model_path,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        enable_heterogeneous=args.enable_heterogeneous,
        heterogeneous_slots_per_layer=args.slots_per_layer,
    )

    prompts = _build_meaningful_prompts(args, rng)
    sampling_params = [
        SamplingParams(
            temperature=args.temperature,
            ignore_eos=True,
            max_tokens=rng.randint(args.min_output_len, args.max_output_len),
        )
        for _ in range(args.num_seqs)
    ]

    llm.generate(["Warmup request for benchmark."], SamplingParams(temperature=args.temperature, max_tokens=4), use_tqdm=False)

    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    elapsed = time.time() - t0
    llm.exit()

    input_tokens = _count_prompt_tokens(llm, prompts)
    target_output_tokens = sum(sp.max_tokens for sp in sampling_params)
    generated_token_ids = [output["token_ids"] for output in outputs]
    generated_texts = [output["text"] for output in outputs]
    generated_output_tokens = sum(len(token_ids) for token_ids in generated_token_ids)
    processed_tokens = input_tokens + generated_output_tokens

    result = {
        "enable_heterogeneous": args.enable_heterogeneous,
        "slots_per_layer": args.slots_per_layer,
        "num_seqs": args.num_seqs,
        "min_input_len": args.min_input_len,
        "max_input_len": args.max_input_len,
        "min_output_len": args.min_output_len,
        "max_output_len": args.max_output_len,
        "seed": args.seed,
        "input_tokens": input_tokens,
        "target_output_tokens": target_output_tokens,
        "generated_output_tokens": generated_output_tokens,
        "processed_tokens": processed_tokens,
        "elapsed_sec": elapsed,
        "throughput_output_tok_s": generated_output_tokens / elapsed,
        "throughput_total_tok_s": processed_tokens / elapsed,
        "outputs_digest": _hash_many_token_ids(generated_token_ids),
    }

    if args.return_token_ids:
        result["generated_token_ids"] = generated_token_ids
    if args.return_text:
        result["generated_texts"] = generated_texts
    if args.return_prompts:
        result["prompts"] = prompts

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single benchmark case for standard or heterogeneous path.")
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
    parser.add_argument("--temperature", type=float, default=1e-5)
    parser.add_argument("--enforce-eager", type=str2bool, default=True)
    parser.add_argument("--return-token-ids", type=str2bool, default=False)
    parser.add_argument("--return-text", type=str2bool, default=True)
    parser.add_argument("--return-prompts", type=str2bool, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_case(args)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
