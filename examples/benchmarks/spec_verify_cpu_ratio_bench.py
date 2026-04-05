import argparse
import json
import subprocess
import sys
from pathlib import Path


def str2bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spec/verify benchmark by CPU expert-set ratios")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--cpu-ratios", type=str, default="25,50,75")
    parser.add_argument("--num-seqs", type=int, default=8)
    parser.add_argument("--input-len", type=int, default=64)
    parser.add_argument("--output-len", type=int, default=24)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-draft-tokens", type=int, default=8)
    parser.add_argument("--dist-port-base", type=int, default=28600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--enforce-eager", type=str2bool, default=True)
    parser.add_argument("--engine-profile", type=str2bool, default=True)
    parser.add_argument("--engine-profile-cuda-sync", type=str2bool, default=True)
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def run_case(case_script: Path, args: argparse.Namespace, slots_per_layer: int, dist_port: int) -> dict:
    cmd = [
        sys.executable,
        str(case_script),
        "--model-path",
        args.model_path,
        "--mode",
        "spec",
        "--slots-per-layer",
        str(slots_per_layer),
        "--num-seqs",
        str(args.num_seqs),
        "--input-len",
        str(args.input_len),
        "--output-len",
        str(args.output_len),
        "--max-model-len",
        str(args.max_model_len),
        "--max-draft-tokens",
        str(args.max_draft_tokens),
        "--dist-port",
        str(dist_port),
        "--seed",
        str(args.seed),
        "--temperature",
        str(args.temperature),
        "--enforce-eager",
        str(args.enforce_eager).lower(),
        "--engine-profile",
        str(args.engine_profile).lower(),
        "--engine-profile-cuda-sync",
        str(args.engine_profile_cuda_sync).lower(),
        "--return-token-ids",
        "false",
        "--return-text",
        "false",
        "--return-prompts",
        "false",
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Benchmark case failed (slots={slots_per_layer}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("No JSON output from heterogeneous_benchmark_case.py")
    return json.loads(lines[-1])


def main() -> None:
    args = parse_args()
    ratios = [int(x) for x in args.cpu_ratios.split(",") if x.strip()]
    case_script = Path(__file__).resolve().parents[1] / "heterogeneous_benchmark_case.py"

    rows = []
    for idx, ratio in enumerate(ratios):
        cpu_ratio = max(0.0, min(1.0, ratio / 100.0))
        slots = int(round(args.num_experts * (1.0 - cpu_ratio)))
        result = run_case(case_script, args, slots_per_layer=slots, dist_port=args.dist_port_base + idx)
        profile = result.get("engine_profile", {})
        rows.append(
            {
                "cpu_expert_set_ratio": cpu_ratio,
                "slots_per_layer": slots,
                "throughput_output_tok_s": result.get("throughput_output_tok_s", 0.0),
                "throughput_total_tok_s": result.get("throughput_total_tok_s", 0.0),
                "cpu_route_ratio": profile.get("model_cpu_route_ratio", 0.0),
                "cpu_weight_mass_ratio": profile.get("model_cpu_weight_mass_ratio", 0.0),
                "activated_expert_set_size": profile.get("model_activated_expert_set_size", 0.0),
                "realized_cpu_expert_count": profile.get("model_realized_cpu_expert_count", 0.0),
                "verify_ms": profile.get("spec_verify_ms", 0.0),
                "spec_step_ms": profile.get("spec_spec_step_ms", 0.0),
            }
        )

    report = {
        "config": {
            "cpu_ratios": ratios,
            "num_experts": args.num_experts,
            "num_seqs": args.num_seqs,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "enforce_eager": args.enforce_eager,
        },
        "results": rows,
    }

    text = json.dumps(report, ensure_ascii=True, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main()
