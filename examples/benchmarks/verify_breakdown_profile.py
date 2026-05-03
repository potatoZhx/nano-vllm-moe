import argparse
import json
import os
import statistics
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


def _extract_json_stdout(stdout: str) -> dict:
    text = stdout.strip()
    if not text:
        raise RuntimeError("No JSON output")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _case_env(base_env: dict[str, str], case: dict, args: argparse.Namespace) -> dict[str, str]:
    env = dict(base_env)
    if case.get("torch_profile"):
        env["NANOVLLM_VERIFY_TORCH_PROFILE_DIR"] = str(args.profile_dir / case["name"])
    return env


def _run_case(case_script: Path, args: argparse.Namespace, case: dict, dist_port: int, output_path: Path) -> dict:
    cmd = [
        sys.executable,
        str(case_script),
        "--model-path",
        args.model_path,
        "--mode",
        case.get("mode", "spec"),
        "--slots-per-layer",
        str(case["slots"]),
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
        str(args.draft_top_c),
        "--cpu-expert-execution-enabled",
        str(args.cpu_expert_execution_enabled).lower(),
        "--cpu-expert-parallel-mode",
        args.cpu_expert_parallel_mode,
        "--cpu-expert-num-threads",
        str(args.cpu_expert_num_threads),
        "--cpu-gpu-parallel-execution-enabled",
        str(args.cpu_gpu_parallel_execution_enabled).lower(),
        "--cpu-gpu-parallel-min-cpu-route-ratio",
        str(args.cpu_gpu_parallel_min_cpu_route_ratio),
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
        str(args.engine_profile_cuda_sync).lower(),
        "--spec-enable-prefetch",
        str(case.get("spec_enable_prefetch", args.spec_enable_prefetch)).lower(),
        "--prefetch-verify-wait-ms",
        str(case.get("prefetch_verify_wait_ms", args.prefetch_verify_wait_ms)),
        "--prefetch-step-budget",
        str(args.prefetch_step_budget),
        "--prefetch-max-inflight",
        str(args.prefetch_max_inflight),
        "--prefetch-staging-slots-per-layer",
        str(args.prefetch_staging_slots_per_layer),
        "--prefetch-use-verify-history",
        str(case.get("prefetch_use_verify_history", True)).lower(),
        "--return-token-ids",
        str(args.return_token_ids).lower(),
        "--return-text",
        "false",
        "--return-prompts",
        "false",
        "--output",
        str(output_path),
    ]
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=False,
        timeout=int(args.case_timeout_sec),
        env=_case_env(os.environ, case, args),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Case failed: {case['name']} port={dist_port}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return _extract_json_stdout(proc.stdout)


def _profile_value(profile: dict, key: str) -> float:
    return float(profile.get(key, 0.0))


def _summarize_result(case: dict, repeats: list[dict]) -> dict:
    profiles = [(r.get("engine_profile") or {}) for r in repeats]
    verify_calls = [_profile_value(p, "spec_run_verify_calls") for p in profiles]
    verify_call_den = [v if v > 0 else 1.0 for v in verify_calls]
    moe_counts = [_profile_value(p, "model_verify_moe_profile_count") for p in profiles]

    def per_call(key: str) -> float:
        return _median([_safe_div(_profile_value(p, key), den) for p, den in zip(profiles, verify_call_den)])

    def per_moe(key: str) -> float:
        return _median([_safe_div(_profile_value(p, key), den) for p, den in zip(profiles, moe_counts) if den > 0])

    def raw_med(key: str) -> float:
        return _median([_profile_value(p, key) for p in profiles])

    cpu_route_ratio = _median(
        [
            _safe_div(_profile_value(p, "model_verify_cpu_route_ratio_sum"), _profile_value(p, "model_verify_moe_profile_count"))
            for p in profiles
            if _profile_value(p, "model_verify_moe_profile_count") > 0
        ]
    )
    cpu_weight_mass_ratio = _median(
        [
            _safe_div(
                _profile_value(p, "model_verify_cpu_weight_mass_ratio_sum"),
                _profile_value(p, "model_verify_moe_profile_count"),
            )
            for p in profiles
            if _profile_value(p, "model_verify_moe_profile_count") > 0
        ]
    )
    elapsed_ms = _median([float(r.get("elapsed_sec", 0.0)) * 1000.0 for r in repeats])
    throughput = _median([float(r.get("throughput_output_tok_s", 0.0)) for r in repeats])
    verify_ms = _median(
        [
            _safe_div(_profile_value(p, "spec_verify_ms"), den)
            for p, den in zip(profiles, verify_call_den)
        ]
    )
    verify_tokens = _median(
        [
            _safe_div(_profile_value(p, "model_verify_tokens_in_total"), den)
            for p, den in zip(profiles, verify_call_den)
        ]
    )

    profile_summary = {
        "elapsed_ms_median": elapsed_ms,
        "throughput_output_tok_s_median": throughput,
        "verify_calls_median": _median(verify_calls),
        "verify_tokens_per_call_median": verify_tokens,
        "spec_verify_ms_per_call": verify_ms,
        "run_verify_total_ms_per_call": per_call("model_run_verify_total_ms"),
        "verify_prepare_prefill_ms_per_call": per_call("model_verify_prepare_prefill_ms"),
        "verify_forward_ms_per_call": per_call("model_verify_forward_ms"),
        "verify_route_ms_per_call": per_call("model_verify_route_ms"),
        "verify_plan_ms_per_call": per_call("model_verify_plan_ms"),
        "verify_gpu_gather_ms_per_call": per_call("model_verify_gpu_gather_ms"),
        "verify_gpu_compute_ms_per_call": per_call("model_verify_gpu_compute_ms"),
        "verify_scatter_ms_per_call": per_call("model_verify_scatter_ms"),
        "verify_cpu_prepare_ms_per_call": per_call("model_verify_cpu_prepare_ms"),
        "verify_cpu_compute_ms_per_call": per_call("model_verify_cpu_compute_ms"),
        "verify_cpu_to_gpu_merge_ms_per_call": per_call("model_verify_cpu_to_gpu_merge_ms"),
        "verify_parallel_wall_ms_per_call": per_call("model_verify_parallel_wall_ms"),
        "verify_cpu_wait_ms_per_call": per_call("model_verify_cpu_wait_ms"),
        "verify_gpu_wait_ms_per_call": per_call("model_verify_gpu_wait_ms"),
        "verify_cpu_route_ratio": cpu_route_ratio,
        "verify_cpu_weight_mass_ratio": cpu_weight_mass_ratio,
        "verify_route_ms_per_moe": per_moe("model_verify_route_ms"),
        "verify_plan_ms_per_moe": per_moe("model_verify_plan_ms"),
        "verify_gpu_compute_ms_per_moe": per_moe("model_verify_gpu_compute_ms"),
        "verify_cpu_compute_ms_per_moe": per_moe("model_verify_cpu_compute_ms"),
        "prefetch_wait_ms_total": raw_med("model_prefetch_wait_ms"),
        "publish_ms_total": raw_med("model_publish_ms"),
        "metadata_offload_verify_ms_total": raw_med("model_metadata_offload_verify_ms"),
        "metadata_offload_verify_count": raw_med("model_metadata_offload_verify_count"),
        "run_verify_metadata_enqueue_ms_per_call": per_call("model_run_verify_metadata_enqueue_ms"),
        "run_verify_metadata_wait_ms_per_call": per_call("model_run_verify_metadata_wait_ms"),
        "run_verify_metadata_collect_ms_per_call": per_call("model_run_verify_metadata_collect_ms"),
        "run_verify_metadata_observe_ms_per_call": per_call("model_run_verify_metadata_observe_ms"),
        "run_verify_metadata_mark_access_ms_per_call": per_call("model_run_verify_metadata_mark_access_ms"),
        "run_verify_metadata_queue_update_ms_per_call": per_call("model_run_verify_metadata_queue_update_ms"),
        "run_verify_metadata_queue_aggregate_ms_per_call": per_call("model_run_verify_metadata_queue_aggregate_ms"),
        "run_verify_metadata_queue_filter_ms_per_call": per_call("model_run_verify_metadata_queue_filter_ms"),
        "run_verify_metadata_queue_entry_update_ms_per_call": per_call("model_run_verify_metadata_queue_entry_update_ms"),
        "run_verify_submit_after_ms_per_call": per_call("model_run_verify_submit_after_ms"),
        "publish_snapshot_ms_total": raw_med("model_publish_snapshot_ms"),
        "publish_select_ms_total": raw_med("model_publish_select_ms"),
        "publish_finalize_ms_total": raw_med("model_publish_finalize_ms"),
        "standard_graph_replay_count": raw_med("model_standard_graph_replay_count"),
        "draft_graph_replay_count": raw_med("model_draft_graph_replay_count"),
        "graph_hit_rate": raw_med("model_graph_hit_rate"),
    }
    return {
        "case": case,
        "summary": profile_summary,
        "output_digests": [r.get("outputs_digest", "") for r in repeats],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify-stage breakdown and optimization experiment")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--raw-output-dir", required=True)
    parser.add_argument("--profile-dir", required=True, type=Path)
    parser.add_argument("--num-seqs", type=int, default=1)
    parser.add_argument("--input-len", type=int, default=24)
    parser.add_argument("--output-len", type=int, default=12)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.98)
    parser.add_argument("--max-draft-tokens", type=int, default=4)
    parser.add_argument("--draft-top-c", type=int, default=0)
    parser.add_argument("--cpu-expert-execution-enabled", type=str2bool, default=True)
    parser.add_argument("--cpu-expert-parallel-mode", type=str, default="expert_parallel")
    parser.add_argument("--cpu-expert-num-threads", type=int, default=4)
    parser.add_argument("--cpu-gpu-parallel-execution-enabled", type=str2bool, default=True)
    parser.add_argument("--cpu-gpu-parallel-min-cpu-route-ratio", type=float, default=0.0)
    parser.add_argument("--dist-port-base", type=int, default=30700)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--enforce-eager", type=str2bool, default=False)
    parser.add_argument("--engine-profile-cuda-sync", type=str2bool, default=True)
    parser.add_argument("--spec-enable-prefetch", type=str2bool, default=True)
    parser.add_argument("--prefetch-verify-wait-ms", type=float, default=1.0)
    parser.add_argument("--prefetch-step-budget", type=int, default=4)
    parser.add_argument("--prefetch-max-inflight", type=int, default=8)
    parser.add_argument("--prefetch-staging-slots-per-layer", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--case-timeout-sec", type=int, default=1800)
    parser.add_argument("--return-token-ids", type=str2bool, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_script = Path(__file__).resolve().parents[1] / "heterogeneous_benchmark_case.py"
    raw_dir = Path(args.raw_output_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    args.profile_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        {"name": "standard_graph_decode", "mode": "standard", "slots": 0, "cache_ratio": 1.0, "spec_enable_prefetch": False},
        {"name": "cache100_baseline", "mode": "spec", "slots": 0, "cache_ratio": 1.0},
        {
            "name": "cache100_torch_profile",
            "mode": "spec",
            "slots": 0,
            "cache_ratio": 1.0,
            "torch_profile": True,
        },
        {"name": "cache75_baseline", "mode": "spec", "slots": 96, "cache_ratio": 0.75},
        {"name": "cache50_baseline", "mode": "spec", "slots": 64, "cache_ratio": 0.50},
    ]

    rows = []
    port = int(args.dist_port_base)
    for case in cases:
        repeats = []
        for repeat_idx in range(int(args.repeats)):
            output_path = raw_dir / f"{case['name']}_repeat{repeat_idx:02d}.json"
            result = _run_case(case_script, args, case, port, output_path)
            repeats.append(result)
            port += 1
        rows.append(_summarize_result(case, repeats))

    report = {
        "config": {
            "model_path": args.model_path,
            "num_seqs": args.num_seqs,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "max_draft_tokens": args.max_draft_tokens,
            "draft_top_c": args.draft_top_c,
            "repeats": args.repeats,
            "engine_profile_cuda_sync": args.engine_profile_cuda_sync,
            "cpu_expert_execution_enabled": args.cpu_expert_execution_enabled,
            "cpu_gpu_parallel_execution_enabled": args.cpu_gpu_parallel_execution_enabled,
            "cpu_gpu_parallel_min_cpu_route_ratio": args.cpu_gpu_parallel_min_cpu_route_ratio,
            "prefetch_verify_wait_ms": args.prefetch_verify_wait_ms,
            "prefetch_step_budget": args.prefetch_step_budget,
            "prefetch_max_inflight": args.prefetch_max_inflight,
            "prefetch_staging_slots_per_layer": args.prefetch_staging_slots_per_layer,
            "raw_output_dir": str(raw_dir),
            "profile_dir": str(args.profile_dir),
        },
        "rows": rows,
    }
    text = json.dumps(report, ensure_ascii=True, indent=2)
    print(text)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
