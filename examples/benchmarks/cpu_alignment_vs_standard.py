import argparse
import json
import subprocess
import sys
from pathlib import Path

from transformers import AutoConfig


def str2bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic alignment check: standard vs heter with CPU execution")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--cpu-ratios", type=str, default="25,50,75,90")
    parser.add_argument("--num-seqs", type=int, default=2)
    parser.add_argument("--input-len", type=int, default=8)
    parser.add_argument("--output-len", type=int, default=6)
    parser.add_argument("--max-model-len", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--enforce-eager", type=str2bool, default=True)
    parser.add_argument("--dist-port-base", type=int, default=29900)
    parser.add_argument("--cpu-expert-parallel-mode", type=str, default="serial")
    parser.add_argument("--cpu-expert-num-threads", type=int, default=4)
    parser.add_argument("--output", type=str, default="benchmarks/results/cpu_alignment_vs_standard_deterministic.json")
    return parser.parse_args()


def run_case(case_script: Path, args: argparse.Namespace, mode: str, slots_per_layer: int, dist_port: int) -> dict:
    cmd = [
        sys.executable,
        str(case_script),
        "--model-path",
        args.model_path,
        "--mode",
        mode,
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
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-draft-tokens",
        "2",
        "--dist-port",
        str(dist_port),
        "--seed",
        str(args.seed),
        "--temperature",
        str(args.temperature),
        "--enforce-eager",
        str(args.enforce_eager).lower(),
        "--engine-profile",
        "true",
        "--engine-profile-cuda-sync",
        "true",
        "--cpu-expert-execution-enabled",
        "true" if mode != "standard" else "false",
        "--cpu-expert-parallel-mode",
        args.cpu_expert_parallel_mode,
        "--cpu-expert-num-threads",
        str(args.cpu_expert_num_threads),
        "--return-token-ids",
        "true",
        "--return-text",
        "false",
        "--return-prompts",
        "false",
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Case failed: mode={mode}, slots={slots_per_layer}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"No output from case script for mode={mode}")
    return json.loads(lines[-1])


def main() -> None:
    args = parse_args()
    case_script = Path(__file__).resolve().parents[1] / "heterogeneous_benchmark_case.py"
    num_experts = int(getattr(AutoConfig.from_pretrained(args.model_path), "num_experts"))
    ratios = [int(x) for x in args.cpu_ratios.split(",") if x.strip()]

    standard = run_case(case_script, args, mode="standard", slots_per_layer=0, dist_port=args.dist_port_base)
    std_tokens = standard.get("generated_token_ids") or []

    rows = []
    for i, ratio_int in enumerate(ratios, start=1):
        cpu_ratio = max(0.0, min(1.0, ratio_int / 100.0))
        slots = max(1, int(round(num_experts * (1.0 - cpu_ratio))))
        heter = run_case(case_script, args, mode="heter", slots_per_layer=slots, dist_port=args.dist_port_base + i)
        heter_tokens = heter.get("generated_token_ids") or []
        per_seq_match = [a == b for a, b in zip(std_tokens, heter_tokens)]
        profile = heter.get("engine_profile") or {}
        rows.append(
            {
                "cpu_ratio_target": cpu_ratio,
                "slots_per_layer": slots,
                "sequence_exact_match": per_seq_match,
                "all_exact_match": bool(per_seq_match) and all(per_seq_match),
                "model_cpu_route_ratio": profile.get("model_cpu_route_ratio", 0.0),
                "model_cpu_weight_mass_ratio": profile.get("model_cpu_weight_mass_ratio", 0.0),
                "model_realized_cpu_expert_count": profile.get("model_realized_cpu_expert_count", 0.0),
                "throughput_output_tok_s": heter.get("throughput_output_tok_s", 0.0),
            }
        )

    report = {
        "model_path": args.model_path,
        "num_experts": num_experts,
        "num_seqs": args.num_seqs,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "temperature": args.temperature,
        "enforce_eager": args.enforce_eager,
        "cpu_expert_execution_enabled": True,
        "cpu_expert_parallel_mode": args.cpu_expert_parallel_mode,
        "cpu_expert_num_threads": args.cpu_expert_num_threads,
        "baseline_standard_outputs_digest": standard.get("outputs_digest"),
        "results": rows,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    print(f"Saved report to: {out}")


if __name__ == "__main__":
    main()
