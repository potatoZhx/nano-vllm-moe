import argparse
import json
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
        "--enforce-eager",
        str(args.enforce_eager).lower(),
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
    parser.add_argument("--enforce-eager", type=str2bool, default=True)
    parser.add_argument("--result-json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_script = Path(__file__).with_name("heterogeneous_benchmark_case.py")

    standard = run_one_case(case_script, args, enable_heterogeneous=False)
    heterogeneous = run_one_case(case_script, args, enable_heterogeneous=True)

    ratio = heterogeneous["throughput_tok_s"] / standard["throughput_tok_s"]
    delta_percent = (ratio - 1.0) * 100.0

    report = {
        "standard": standard,
        "heterogeneous": heterogeneous,
        "ratio_hetero_vs_standard": ratio,
        "delta_percent": delta_percent,
    }

    print("=== Standard Path ===")
    print(
        f"tokens={standard['total_tokens']}, time={standard['elapsed_sec']:.3f}s, "
        f"throughput={standard['throughput_tok_s']:.2f} tok/s"
    )
    print("=== Heterogeneous Path (S=N by default when slots=0) ===")
    print(
        f"tokens={heterogeneous['total_tokens']}, time={heterogeneous['elapsed_sec']:.3f}s, "
        f"throughput={heterogeneous['throughput_tok_s']:.2f} tok/s"
    )
    print("=== Delta ===")
    print(f"ratio(hetero/standard)={ratio:.4f}, delta={delta_percent:+.2f}%")

    if args.result_json:
        with open(args.result_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=True, indent=2)
        print(f"Saved report to: {args.result_json}")


if __name__ == "__main__":
    main()
