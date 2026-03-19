import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# CUDA_VISIBLE_DEVICES=2 conda run -n moe_spec python examples/three_mode_speed_compare.py --model-path /zx_data1/models/Qwen--Qwen3-30B-A3B-Base --slots-per-layer 0 --num-seqs 1 --min-input-len 8 --max-input-len 12 --min-output-len 4 --max-output-len 6 --temperature 0.0 --enforce-eager true --check-correctness true --max-draft-tokens 4 --dist-port-base 27420 --result-json benchmarks/results/three_mode_smoke_profile_avg.json


def str2bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def run_case(case_script: Path, args: argparse.Namespace, mode: str) -> dict:
    mode_offset = {"standard": 0, "heter": 1, "spec": 2}[mode]
    dist_port = args.dist_port_base + mode_offset

    cmd = [
        sys.executable,
        str(case_script),
        "--model-path",
        args.model_path,
        "--mode",
        mode,
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
        "--spec-profile",
        str(args.spec_profile).lower(),
        "--return-token-ids",
        str(args.check_correctness).lower(),
        "--return-text",
        str(args.return_text).lower(),
        "--return-prompts",
        str(args.return_text).lower(),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError(f"Case failed: mode={mode}")

    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"No output from case script: mode={mode}")
    return json.loads(lines[-1])


def token_alignment(reference: dict, target: dict) -> dict:
    ref = reference.get("generated_token_ids") or []
    cur = target.get("generated_token_ids") or []
    if len(ref) != len(cur):
        return {
            "sequence_count_match": False,
            "exact_match": False,
            "exact_match_rate": 0.0,
            "matched_sequences": 0,
            "total_sequences": len(ref),
        }

    matched = 0
    for a, b in zip(ref, cur):
        if a == b:
            matched += 1
    total = len(ref)
    return {
        "sequence_count_match": True,
        "exact_match": matched == total,
        "exact_match_rate": (matched / total) if total else 1.0,
        "matched_sequences": matched,
        "total_sequences": total,
    }


def summarize(standard: dict, heter: dict, spec: dict, deterministic: bool) -> dict:
    def ratio(a: float, b: float) -> float:
        return a / b

    summary = {
        "deterministic_check": deterministic,
        "ratio_output_tps": {
            "heter_vs_standard": ratio(heter["throughput_output_tok_s"], standard["throughput_output_tok_s"]),
            "spec_vs_standard": ratio(spec["throughput_output_tok_s"], standard["throughput_output_tok_s"]),
        },
        "ratio_total_tps": {
            "heter_vs_standard": ratio(heter["throughput_total_tok_s"], standard["throughput_total_tok_s"]),
            "spec_vs_standard": ratio(spec["throughput_total_tok_s"], standard["throughput_total_tok_s"]),
        },
    }

    if deterministic:
        summary["alignment"] = {
            "heter_vs_standard": token_alignment(standard, heter),
            "spec_vs_standard": token_alignment(standard, spec),
        }
    else:
        summary["alignment"] = {
            "heter_vs_standard": "non-deterministic run; use qualitative comparison",
            "spec_vs_standard": "non-deterministic run; use qualitative comparison",
        }

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare standard/heter/spec modes on speed and correctness.")
    parser.add_argument("--model-path", default=os.path.expanduser("/zx_data1/models/Qwen--Qwen3-30B-A3B-Base"))
    parser.add_argument("--slots-per-layer", type=int, default=0)
    parser.add_argument("--num-seqs", type=int, default=8)
    parser.add_argument("--min-input-len", type=int, default=32)
    parser.add_argument("--max-input-len", type=int, default=96)
    parser.add_argument("--min-output-len", type=int, default=16)
    parser.add_argument("--max-output-len", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-draft-tokens", type=int, default=8)
    parser.add_argument("--dist-port-base", type=int, default=26000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--enforce-eager", type=str2bool, default=True)
    parser.add_argument("--spec-profile", type=str2bool, default=True)
    parser.add_argument("--check-correctness", type=str2bool, default=True)
    parser.add_argument("--return-text", type=str2bool, default=False)
    parser.add_argument("--result-json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_script = Path(__file__).with_name("heterogeneous_benchmark_case.py")

    standard = run_case(case_script, args, mode="standard")
    heter = run_case(case_script, args, mode="heter")
    spec = run_case(case_script, args, mode="spec")

    deterministic = args.temperature <= 1e-10 and args.check_correctness
    report = {
        "standard": standard,
        "heter": heter,
        "spec": spec,
        "summary": summarize(standard, heter, spec, deterministic=deterministic),
    }

    print("=== Throughput (output tok/s) ===")
    print(f"standard={standard['throughput_output_tok_s']:.2f}")
    print(f"heter={heter['throughput_output_tok_s']:.2f}")
    print(f"spec={spec['throughput_output_tok_s']:.2f}")
    print("=== Ratios vs standard ===")
    print(f"heter_output_ratio={report['summary']['ratio_output_tps']['heter_vs_standard']:.4f}")
    print(f"spec_output_ratio={report['summary']['ratio_output_tps']['spec_vs_standard']:.4f}")

    if deterministic:
        a = report["summary"]["alignment"]
        print("=== Deterministic Token Alignment ===")
        print(
            f"heter exact={a['heter_vs_standard']['exact_match']} "
            f"rate={a['heter_vs_standard']['exact_match_rate']:.4f}"
        )
        print(
            f"spec exact={a['spec_vs_standard']['exact_match']} "
            f"rate={a['spec_vs_standard']['exact_match_rate']:.4f}"
        )

    spec_profile = report["spec"].get("spec_profile")
    if isinstance(spec_profile, dict) and spec_profile:
        run_draft_calls = float(spec_profile.get("run_draft_calls", 0.0))
        run_verify_calls = float(spec_profile.get("run_verify_calls", 0.0))
        draft_infer_total_ms = float(spec_profile.get("run_draft_infer_ms_total", spec_profile.get("draft_loop_ms", 0.0)))
        verify_infer_total_ms = float(spec_profile.get("run_verify_infer_ms_total", spec_profile.get("verify_ms", 0.0)))
        draft_single_ms = (draft_infer_total_ms / run_draft_calls) if run_draft_calls > 0 else 0.0
        verify_single_ms = (verify_infer_total_ms / run_verify_calls) if run_verify_calls > 0 else 0.0

        print("=== Spec Profile (ms) ===")
        print(f"spec_step_ms={spec_profile.get('spec_step_ms', 0.0):.2f}")
        print(f"draft_loop_ms={spec_profile.get('draft_loop_ms', 0.0):.2f}")
        print(f"draft_single_infer_ms={draft_single_ms:.2f}")
        print(f"run_draft_infer_ms_total={draft_infer_total_ms:.2f}")
        print(f"prepare_verify_ms={spec_profile.get('prepare_verify_ms', 0.0):.2f}")
        print(f"verify_ms={spec_profile.get('verify_ms', 0.0):.2f}")
        print(f"verify_single_infer_ms={verify_single_ms:.2f}")
        print(f"run_verify_infer_ms_total={verify_infer_total_ms:.2f}")
        print(f"accept_ms={spec_profile.get('accept_ms', 0.0):.2f}")
        draft_total = float(spec_profile.get("draft_tokens_total", 0.0))
        accepted_total = float(spec_profile.get("accepted_tokens_total", 0.0))
        accept_rate = (accepted_total / draft_total) if draft_total > 0 else 0.0
        print(f"draft_tokens_total={int(draft_total)}")
        print(f"accepted_tokens_total={int(accepted_total)}")
        print(f"accept_rate={accept_rate:.4f}")

    if args.result_json:
        with open(args.result_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=True, indent=2)
        print(f"Saved report to: {args.result_json}")


if __name__ == "__main__":
    main()
