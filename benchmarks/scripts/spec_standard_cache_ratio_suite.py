from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from transformers import AutoConfig


def str2bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


@dataclass(frozen=True)
class SpecSetting:
    name: str
    args: tuple[str, ...]


VERIFY_CPU_KEYS = [
    "verify_cpu_route_ratio",
    "verify_cpu_weight_mass_ratio",
    "verify_cpu_prepare_ms",
    "verify_cpu_compute_ms",
    "verify_cpu_to_gpu_merge_ms",
    "verify_realized_cpu_expert_count",
    "verify_prefetch_wait_ms",
    "verify_verify_prefetch_wait_ms",
    "verify_prefetch_submit_count",
    "verify_prefetch_completed_count",
    "verify_prefetch_late_count",
    "verify_prefetch_consumed_count",
    "verify_prefetch_timeout_count",
    "verify_ready_before_wait_count",
    "verify_ready_after_wait_count",
]


def _extract_last_json(stdout: str) -> dict:
    text = stdout.strip()
    if not text:
        raise RuntimeError("No JSON output found in subprocess stdout")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
    raise RuntimeError(f"Unable to parse JSON from stdout:\n{stdout}")


def _slots_for_ratio(model_path: str, ratio: float) -> int:
    if ratio >= 0.999:
        return 0
    cfg = AutoConfig.from_pretrained(model_path)
    num_experts = int(getattr(cfg, "num_experts"))
    return max(1, min(num_experts, round(num_experts * ratio)))


def _effective_draft_top_c(model_path: str, requested: int) -> int:
    if requested >= 0:
        return requested
    cfg = AutoConfig.from_pretrained(model_path)
    return int(getattr(cfg, "num_experts"))


def _run_case(
    *,
    case_script: Path,
    args: argparse.Namespace,
    mode: str,
    slots_per_layer: int,
    dist_port: int,
    extra_args: tuple[str, ...] = (),
) -> dict:
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
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-draft-tokens",
        str(args.max_draft_tokens),
        "--draft-top-c",
        str(_effective_draft_top_c(args.model_path, args.draft_top_c)),
        "--cpu-expert-execution-enabled",
        "true" if mode == "spec" else "false",
        "--seed",
        str(args.seed),
        "--temperature",
        str(args.temperature),
        "--enforce-eager",
        "false",
        "--spec-profile",
        "true",
        "--engine-profile",
        "true",
        "--engine-profile-cuda-sync",
        "true",
        "--spec-enable-prefetch",
        "true" if mode == "spec" else "false",
        "--prefetch-verify-wait-ms",
        str(args.prefetch_verify_wait_ms),
        "--return-token-ids",
        "true",
        "--return-text",
        "false",
        "--return-prompts",
        "false",
        "--dist-port",
        str(dist_port),
    ]
    cmd.extend(extra_args)

    env = os.environ.copy()
    proc = subprocess.run(
        cmd,
        cwd=case_script.parents[1],
        text=True,
        capture_output=True,
        check=False,
        timeout=args.case_timeout_sec,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Case failed: mode={mode}, slots={slots_per_layer}, port={dist_port}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return _extract_last_json(proc.stdout)


def _setting_matrix(profile: str) -> list[SpecSetting]:
    full = SpecSetting("full_prefetch", ())
    if profile == "smoke":
        return [full]
    return [
        full,
        SpecSetting("no_prefill_history", ("--prefetch-use-prefill-history", "false")),
        SpecSetting("no_verify_history", ("--prefetch-use-verify-history", "false")),
        SpecSetting("no_draft_live", ("--prefetch-use-draft-live", "false")),
        SpecSetting("zero_verify_wait", ("--prefetch-verify-wait-ms", "0.0")),
    ]


def _compare_tokens(standard: dict, target: dict, *, temperature: float) -> dict:
    standard_ids = standard.get("generated_token_ids") or []
    target_ids = target.get("generated_token_ids") or []
    exact = standard_ids == target_ids
    if temperature <= 1e-10 and not exact:
        raise RuntimeError(
            "Deterministic token mismatch against standard mode: "
            f"standard_digest={standard.get('outputs_digest')}, target_digest={target.get('outputs_digest')}"
        )
    return {
        "exact_match": exact,
        "standard_digest": standard.get("outputs_digest"),
        "target_digest": target.get("outputs_digest"),
    }


def _profile_float(profile: dict, key: str) -> float:
    candidates = [key]
    if key.startswith("verify_"):
        candidates.append(f"model_{key}")
    for candidate in candidates:
        if candidate in profile:
            return float(profile.get(candidate, 0.0))
    return 0.0


def _flatten_row(
    *,
    standard: dict,
    spec: dict,
    cache_ratio: float,
    slots_per_layer: int,
    setting: str,
    temperature: float,
) -> dict:
    alignment = _compare_tokens(standard, spec, temperature=temperature)
    engine_profile = spec.get("engine_profile") or {}
    spec_profile = spec.get("spec_profile") or {}
    row = {
        "setting": setting,
        "cache_ratio": cache_ratio,
        "slots_per_layer": slots_per_layer,
        "standard_output_tok_s": float(standard.get("throughput_output_tok_s", 0.0)),
        "spec_output_tok_s": float(spec.get("throughput_output_tok_s", 0.0)),
        "spec_vs_standard_output_ratio": (
            float(spec.get("throughput_output_tok_s", 0.0)) / float(standard.get("throughput_output_tok_s", 1.0))
        ),
        "standard_total_tok_s": float(standard.get("throughput_total_tok_s", 0.0)),
        "spec_total_tok_s": float(spec.get("throughput_total_tok_s", 0.0)),
        "spec_vs_standard_total_ratio": (
            float(spec.get("throughput_total_tok_s", 0.0)) / float(standard.get("throughput_total_tok_s", 1.0))
        ),
        "token_exact_match": alignment["exact_match"],
        "standard_digest": alignment["standard_digest"],
        "spec_digest": alignment["target_digest"],
        "spec_step_ms": float(spec_profile.get("spec_step_ms", 0.0)),
        "verify_ms": float(spec_profile.get("verify_ms", 0.0)),
        "run_verify_calls": float(spec_profile.get("run_verify_calls", 0.0)),
    }
    for key in VERIFY_CPU_KEYS:
        row[key] = _profile_float(engine_profile, key)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate standard CUDA graph vs spec+prefetch across expert cache ratios.")
    parser.add_argument("--model-path", default="/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B")
    parser.add_argument("--ratios", default="1.0,0.75,0.5,0.25")
    parser.add_argument("--setting-profile", choices=["smoke", "full"], default="full")
    parser.add_argument("--num-seqs", type=int, default=1)
    parser.add_argument("--input-len", type=int, default=12)
    parser.add_argument("--output-len", type=int, default=6)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-draft-tokens", type=int, default=4)
    parser.add_argument("--draft-top-c", type=int, default=-1, help="Use -1 to select all uncached experts for exact deterministic spec.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prefetch-verify-wait-ms", type=float, default=1.0)
    parser.add_argument("--dist-port-base", type=int, default=31000)
    parser.add_argument("--case-timeout-sec", type=int, default=2400)
    parser.add_argument("--output-json", type=Path, default=Path("benchmarks/results/spec_standard_cache_ratio_suite.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("benchmarks/results/spec_standard_cache_ratio_suite.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    case_script = repo_root / "examples" / "heterogeneous_benchmark_case.py"
    ratios = [float(x) for x in args.ratios.split(",") if x]
    settings = _setting_matrix(args.setting_profile)

    standard = _run_case(
        case_script=case_script,
        args=args,
        mode="standard",
        slots_per_layer=0,
        dist_port=args.dist_port_base,
    )

    rows = []
    cases = []
    next_port = args.dist_port_base + 1
    for ratio in ratios:
        slots = _slots_for_ratio(args.model_path, ratio)
        for setting in settings:
            spec = _run_case(
                case_script=case_script,
                args=args,
                mode="spec",
                slots_per_layer=slots,
                dist_port=next_port,
                extra_args=setting.args,
            )
            next_port += 1
            row = _flatten_row(
                standard=standard,
                spec=spec,
                cache_ratio=ratio,
                slots_per_layer=slots,
                setting=setting.name,
                temperature=args.temperature,
            )
            rows.append(row)
            cases.append({"row": row, "spec": spec})
            print(json.dumps(row, ensure_ascii=True))

    report = {"standard": standard, "cases": cases, "rows": rows}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=True, indent=2)
        f.write("\n")
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved JSON: {args.output_json}")
    print(f"Saved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
