import argparse
import json
import time
from random import Random

from nanovllm import LLM, SamplingParams


GREEDY_LIKE_TEMPERATURE = 1e-5


def str2bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def run_case(args: argparse.Namespace) -> dict:
    rng = Random(args.seed)

    llm = LLM(
        args.model_path,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        enable_heterogeneous=args.enable_heterogeneous,
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

    llm.generate(
        [[1, 2, 3, 4]],
        SamplingParams(temperature=GREEDY_LIKE_TEMPERATURE, max_tokens=4),
        use_tqdm=False,
    )

    t0 = time.time()
    llm.generate(prompts, sampling_params, use_tqdm=False)
    elapsed = time.time() - t0
    llm.exit()

    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / elapsed
    return {
        "enable_heterogeneous": args.enable_heterogeneous,
        "slots_per_layer": args.slots_per_layer,
        "num_seqs": args.num_seqs,
        "min_input_len": args.min_input_len,
        "max_input_len": args.max_input_len,
        "min_output_len": args.min_output_len,
        "max_output_len": args.max_output_len,
        "seed": args.seed,
        "total_tokens": total_tokens,
        "elapsed_sec": elapsed,
        "throughput_tok_s": throughput,
    }


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
    parser.add_argument("--enforce-eager", type=str2bool, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_case(args)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
