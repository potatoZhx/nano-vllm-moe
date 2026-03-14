import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

'''
单案例（标准）
python examples/heterogeneous_benchmark_case.py --model-path /zx_data1/models/Qwen--Qwen3-30B-A3B-Base --enable-heterogeneous false --slots-per-layer 0

单案例（异构，默认 S=N）
python examples/heterogeneous_benchmark_case.py --model-path /zx_data1/models/Qwen--Qwen3-30B-A3B-Base --enable-heterogeneous true --slots-per-layer 0

自动对比并输出统计
python examples/heterogeneous_speed_compare.py --model-path /zx_data1/models/Qwen--Qwen3-30B-A3B-Base --slots-per-layer 0 --result-json benchmarks/results/hetero_compare.json

python examples/heterogeneous_speed_compare.py --model-path /zx_data1/models/Qwen--Qwen3-30B-A3B-Base --slots-per-layer 0 --enable-robust-benchmark true --robust-repeat 7
'''


def str2bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def run_one_case(case_script: Path, args: argparse.Namespace, enable_heterogeneous: bool) -> dict:
    cmd = [
        sys.executable,
        str(case_script),
        "--model-path",
        args.model_path,
        "--enable-heterogeneous",
        str(enable_heterogeneous).lower(),
        "--slots-per-layer",
        str(args.slots_per_layer),
        "--num-seqs",
        str(args.num_seqs),
        "--min-input-len",
        str(args.min_input_len),
        "--max-input-len",
        str(args.max_input_len),
        "--min-output-len",
        str(args.min_output_len),
        "--max-output-len",
        str(args.max_output_len),
        "--max-model-len",
        str(args.max_model_len),
        "--seed",
        str(args.seed),
        "--temperature",
        str(args.temperature),
        "--enforce-eager",
        str(args.enforce_eager).lower(),
        "--return-token-ids",
        str(args.check_correctness).lower(),
        "--return-text",
        str(args.show_text_outputs).lower(),
        "--return-prompts",
        str(args.show_text_outputs).lower(),
    ]

    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError(f"Case failed: enable_heterogeneous={enable_heterogeneous}")

    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("No output from case script")
    return json.loads(lines[-1])


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    pos = (len(sorted_values) - 1) * q
    left = int(math.floor(pos))
    right = int(math.ceil(pos))
    if left == right:
        return sorted_values[left]
    weight = pos - left
    return sorted_values[left] * (1.0 - weight) + sorted_values[right] * weight


def _build_metrics(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p90": float("nan"),
        }
    count = len(values)
    mean = sum(values) / count
    median = _percentile(values, 0.5)
    variance = sum((x - mean) ** 2 for x in values) / count
    std = variance ** 0.5
    return {
        "count": count,
        "mean": mean,
        "median": median,
        "std": std,
        "min": min(values),
        "max": max(values),
        "p90": _percentile(values, 0.9),
    }


def run_robust_compare(case_script: Path, args: argparse.Namespace) -> tuple[dict, dict, dict]:
    pairs = []
    for i in range(args.robust_repeat):
        # Alternate execution order (ABAB/BABA) to reduce warm-up and order bias.
        run_standard_first = (i % 2 == 0)
        if run_standard_first:
            standard = run_one_case(case_script, args, enable_heterogeneous=False)
            heterogeneous = run_one_case(case_script, args, enable_heterogeneous=True)
            order = "standard_then_heterogeneous"
        else:
            heterogeneous = run_one_case(case_script, args, enable_heterogeneous=True)
            standard = run_one_case(case_script, args, enable_heterogeneous=False)
            order = "heterogeneous_then_standard"

        ratio_output = heterogeneous["throughput_output_tok_s"] / standard["throughput_output_tok_s"]
        ratio_total = heterogeneous["throughput_total_tok_s"] / standard["throughput_total_tok_s"]
        pairs.append(
            {
                "iter": i,
                "order": order,
                "standard": standard,
                "heterogeneous": heterogeneous,
                "ratio_output_tok_s_hetero_vs_standard": ratio_output,
                "ratio_total_tok_s_hetero_vs_standard": ratio_total,
                "delta_output_tok_s_percent": (ratio_output - 1.0) * 100.0,
                "delta_total_tok_s_percent": (ratio_total - 1.0) * 100.0,
            }
        )

    ratio_output_values = [pair["ratio_output_tok_s_hetero_vs_standard"] for pair in pairs]
    ratio_total_values = [pair["ratio_total_tok_s_hetero_vs_standard"] for pair in pairs]
    robust_summary = {
        "enabled": True,
        "repeat": args.robust_repeat,
        "ratio_output_tok_s_hetero_vs_standard": _build_metrics(ratio_output_values),
        "ratio_total_tok_s_hetero_vs_standard": _build_metrics(ratio_total_values),
        "delta_output_tok_s_percent": _build_metrics([(x - 1.0) * 100.0 for x in ratio_output_values]),
        "delta_total_tok_s_percent": _build_metrics([(x - 1.0) * 100.0 for x in ratio_total_values]),
    }

    # Keep one representative run for correctness/text outputs and top-level compatibility.
    representative = pairs[0]
    return representative["standard"], representative["heterogeneous"], {
        "summary": robust_summary,
        "runs": pairs,
    }


def summarize_correctness(standard: dict, heterogeneous: dict, max_mismatches: int) -> dict:
    std_ids = standard.get("generated_token_ids") or []
    het_ids = heterogeneous.get("generated_token_ids") or []

    if len(std_ids) != len(het_ids):
        return {
            "checked": True,
            "num_sequences": len(std_ids),
            "num_heterogeneous_sequences": len(het_ids),
            "exact_match": False,
            "exact_match_rate": 0.0,
            "mismatches": [
                {
                    "seq_idx": -1,
                    "reason": "sequence_count_mismatch",
                }
            ],
        }

    mismatches = []
    matched = 0
    for seq_idx, (std_seq, het_seq) in enumerate(zip(std_ids, het_ids)):
        if std_seq == het_seq:
            matched += 1
            continue

        first_diff = -1
        std_tok_at_diff = None
        het_tok_at_diff = None
        for token_pos, (std_tok, het_tok) in enumerate(zip(std_seq, het_seq)):
            if std_tok != het_tok:
                first_diff = token_pos
                std_tok_at_diff = std_tok
                het_tok_at_diff = het_tok
                break
        if first_diff < 0 and len(std_seq) != len(het_seq):
            first_diff = min(len(std_seq), len(het_seq))

        std_seq_digest = hashlib.sha256(",".join(str(token) for token in std_seq).encode("utf-8")).hexdigest()
        het_seq_digest = hashlib.sha256(",".join(str(token) for token in het_seq).encode("utf-8")).hexdigest()

        if len(mismatches) < max_mismatches:
            mismatches.append(
                {
                    "seq_idx": seq_idx,
                    "std_len": len(std_seq),
                    "het_len": len(het_seq),
                    "first_diff_token_pos": first_diff,
                    "std_token_at_diff": std_tok_at_diff,
                    "het_token_at_diff": het_tok_at_diff,
                    "std_seq_digest": std_seq_digest,
                    "het_seq_digest": het_seq_digest,
                }
            )

    total = len(std_ids)
    exact_rate = (matched / total) if total > 0 else 1.0
    return {
        "checked": True,
        "num_sequences": total,
        "exact_match": matched == total,
        "exact_match_rate": exact_rate,
        "matched_sequences": matched,
        "mismatched_sequences": total - matched,
        "mismatches": mismatches,
    }


def _short_text(text: str, max_len: int = 180) -> str:
    flat = " ".join(text.strip().split())
    if len(flat) <= max_len:
        return flat
    return flat[: max_len - 3] + "..."


def build_qualitative_samples(standard: dict, heterogeneous: dict, max_examples: int) -> list[dict]:
    prompts = standard.get("prompts") or []
    std_texts = standard.get("generated_texts") or []
    het_texts = heterogeneous.get("generated_texts") or []
    count = min(len(prompts), len(std_texts), len(het_texts), max_examples)
    samples = []
    for i in range(count):
        samples.append(
            {
                "seq_idx": i,
                "prompt": prompts[i],
                "standard_text": std_texts[i],
                "heterogeneous_text": het_texts[i],
                "text_exact_match": std_texts[i] == het_texts[i],
            }
        )
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standard vs heterogeneous benchmark in isolated processes.")
    parser.add_argument("--model-path", default=os.path.expanduser("/zx_data1/models/Qwen--Qwen3-30B-A3B-Base"))
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
    parser.add_argument("--check-correctness", type=str2bool, default=True)
    parser.add_argument("--max-mismatches", type=int, default=5)
    parser.add_argument("--show-text-outputs", type=str2bool, default=True)
    parser.add_argument("--max-text-examples", type=int, default=3)
    parser.add_argument("--enable-robust-benchmark", type=str2bool, default=False)
    parser.add_argument("--robust-repeat", type=int, default=5)
    parser.add_argument("--result-json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_script = Path(__file__).with_name("heterogeneous_benchmark_case.py")

    if args.robust_repeat < 1:
        raise ValueError("--robust-repeat must be >= 1")

    robust_detail = {"summary": {"enabled": False}}
    if args.enable_robust_benchmark:
        standard, heterogeneous, robust_detail = run_robust_compare(case_script, args)
    else:
        standard = run_one_case(case_script, args, enable_heterogeneous=False)
        heterogeneous = run_one_case(case_script, args, enable_heterogeneous=True)

    ratio_output = heterogeneous["throughput_output_tok_s"] / standard["throughput_output_tok_s"]
    ratio_total = heterogeneous["throughput_total_tok_s"] / standard["throughput_total_tok_s"]
    delta_output_percent = (ratio_output - 1.0) * 100.0
    delta_total_percent = (ratio_total - 1.0) * 100.0
    correctness = summarize_correctness(standard, heterogeneous, max_mismatches=args.max_mismatches) if args.check_correctness else {"checked": False}
    qualitative_samples = build_qualitative_samples(standard, heterogeneous, max_examples=args.max_text_examples) if args.show_text_outputs else []

    report = {
        "standard": standard,
        "heterogeneous": heterogeneous,
        "ratio_output_tok_s_hetero_vs_standard": ratio_output,
        "delta_output_tok_s_percent": delta_output_percent,
        "ratio_total_tok_s_hetero_vs_standard": ratio_total,
        "delta_total_tok_s_percent": delta_total_percent,
        "correctness": correctness,
        "qualitative_samples": qualitative_samples,
        "robust_benchmark": robust_detail,
    }

    print("=== Standard Path ===")
    print(
        f"input={standard['input_tokens']}, output={standard['generated_output_tokens']}, "
        f"processed={standard['processed_tokens']}, time={standard['elapsed_sec']:.3f}s, "
        f"output_tps={standard['throughput_output_tok_s']:.2f}, total_tps={standard['throughput_total_tok_s']:.2f}"
    )
    print("=== Heterogeneous Path (S=N by default when slots=0) ===")
    print(
        f"input={heterogeneous['input_tokens']}, output={heterogeneous['generated_output_tokens']}, "
        f"processed={heterogeneous['processed_tokens']}, time={heterogeneous['elapsed_sec']:.3f}s, "
        f"output_tps={heterogeneous['throughput_output_tok_s']:.2f}, total_tps={heterogeneous['throughput_total_tok_s']:.2f}"
    )
    print("=== Delta ===")
    print(
        f"output_tps_ratio={ratio_output:.4f} ({delta_output_percent:+.2f}%), "
        f"total_tps_ratio={ratio_total:.4f} ({delta_total_percent:+.2f}%)"
    )
    if args.enable_robust_benchmark:
        robust_summary = robust_detail["summary"]
        out_stat = robust_summary["ratio_output_tok_s_hetero_vs_standard"]
        tot_stat = robust_summary["ratio_total_tok_s_hetero_vs_standard"]
        print("=== Robust Summary ===")
        print(
            "output_tps_ratio "
            f"median={out_stat['median']:.4f}, mean={out_stat['mean']:.4f}, "
            f"std={out_stat['std']:.4f}, p90={out_stat['p90']:.4f}, "
            f"min={out_stat['min']:.4f}, max={out_stat['max']:.4f}, n={out_stat['count']}"
        )
        print(
            "total_tps_ratio "
            f"median={tot_stat['median']:.4f}, mean={tot_stat['mean']:.4f}, "
            f"std={tot_stat['std']:.4f}, p90={tot_stat['p90']:.4f}, "
            f"min={tot_stat['min']:.4f}, max={tot_stat['max']:.4f}, n={tot_stat['count']}"
        )
    if args.check_correctness:
        print("=== Correctness ===")
        print(
            f"exact_match={correctness['exact_match']}, "
            f"exact_match_rate={correctness['exact_match_rate']:.4f}, "
            f"matched={correctness['matched_sequences']}/{correctness['num_sequences']}"
        )

    if args.show_text_outputs and qualitative_samples:
        print("=== Text Samples ===")
        for sample in qualitative_samples:
            print(f"[seq {sample['seq_idx']}] match={sample['text_exact_match']}")
            print(f"prompt: {_short_text(sample['prompt'])}")
            print(f"standard: {_short_text(sample['standard_text'])}")
            print(f"heterogeneous: {_short_text(sample['heterogeneous_text'])}")

    if args.result_json:
        with open(args.result_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=True, indent=2)
        print(f"Saved report to: {args.result_json}")


if __name__ == "__main__":
    main()
