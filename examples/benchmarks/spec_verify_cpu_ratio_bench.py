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
    parser.add_argument("--cpu-ratios", type=str, default="0,25,50,75,100")
    parser.add_argument("--parallel-settings", type=str, default="off,on")
    parser.add_argument("--cpu-expert-parallel-mode", type=str, default="serial")
    parser.add_argument("--cpu-expert-num-threads", type=int, default=4)
    parser.add_argument("--cpu-gpu-parallel-min-cpu-route-ratio", type=float, default=0.7)
    parser.add_argument("--num-seqs", type=int, default=8)
    parser.add_argument("--input-len", type=int, default=64)
    parser.add_argument("--output-len", type=int, default=24)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-draft-tokens", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--dist-port-base", type=int, default=28600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--enforce-eager", type=str2bool, default=True)
    parser.add_argument("--engine-profile", type=str2bool, default=True)
    parser.add_argument("--engine-profile-cuda-sync", type=str2bool, default=True)
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = int(round((len(arr) - 1) * q))
    return float(arr[idx])


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _extract_latency_breakdown_from_engine_profile(profile: dict) -> dict[str, float]:
    gpu_path_exec_ms = float(
        profile.get("model_gpu_gather_ms", 0.0)
        + profile.get("model_gpu_compute_ms", 0.0)
        + profile.get("model_scatter_ms", 0.0)
    )
    cpu_path_exec_ms = float(
        profile.get("model_cpu_prepare_ms", 0.0)
        + profile.get("model_cpu_compute_ms", 0.0)
        + profile.get("model_cpu_to_gpu_merge_ms", 0.0)
    )
    wait_ms = float(profile.get("model_cpu_wait_ms", 0.0) + profile.get("model_gpu_wait_ms", 0.0))
    parallel_wall_ms = float(profile.get("model_parallel_wall_ms", 0.0))
    parallel_critical_path_est_ms = float(profile.get("model_parallel_critical_path_est_ms", 0.0))
    sync_barrier_ms = float(max(0.0, parallel_wall_ms - parallel_critical_path_est_ms))

    if parallel_wall_ms > 0.0:
        moe_wall_ms = parallel_wall_ms
    else:
        moe_wall_ms = gpu_path_exec_ms + cpu_path_exec_ms

    return {
        "gpu_path_exec_ms": gpu_path_exec_ms,
        "cpu_path_exec_ms": cpu_path_exec_ms,
        "wait_ms": wait_ms,
        "sync_barrier_ms": sync_barrier_ms,
        "moe_wall_ms": moe_wall_ms,
    }


def run_case(
    case_script: Path,
    args: argparse.Namespace,
    slots_per_layer: int,
    dist_port: int,
    cpu_gpu_parallel_execution_enabled: bool,
) -> dict:
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
        "--cpu-expert-execution-enabled",
        "true",
        "--cpu-expert-parallel-mode",
        args.cpu_expert_parallel_mode,
        "--cpu-expert-num-threads",
        str(args.cpu_expert_num_threads),
        "--cpu-gpu-parallel-execution-enabled",
        str(cpu_gpu_parallel_execution_enabled).lower(),
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
    parallel_tokens = [x.strip().lower() for x in args.parallel_settings.split(",") if x.strip()]
    parallel_settings: list[bool] = []
    for token in parallel_tokens:
        if token in {"on", "true", "1"}:
            parallel_settings.append(True)
        elif token in {"off", "false", "0"}:
            parallel_settings.append(False)
        else:
            raise ValueError(f"Invalid parallel setting: {token}")
    parallel_settings = list(dict.fromkeys(parallel_settings))

    case_script = Path(__file__).resolve().parents[1] / "heterogeneous_benchmark_case.py"

    rows = []
    curves = []
    case_idx = 0
    for ratio in ratios:
        cpu_ratio = max(0.0, min(1.0, ratio / 100.0))
        slots = int(round(args.num_experts * (1.0 - cpu_ratio)))
        per_mode: dict[bool, dict] = {}

        for parallel_enabled in parallel_settings:
            latencies: list[float] = []
            throughput_output: list[float] = []
            throughput_total: list[float] = []
            verify_per_call: list[float] = []
            spec_step_per_call: list[float] = []
            cpu_route_ratios: list[float] = []
            cpu_weight_mass_ratios: list[float] = []
            gpu_path_exec_ms_list: list[float] = []
            cpu_path_exec_ms_list: list[float] = []
            wait_ms_list: list[float] = []
            sync_barrier_ms_list: list[float] = []
            moe_wall_ms_list: list[float] = []

            for _ in range(args.repeat):
                result = run_case(
                    case_script,
                    args,
                    slots_per_layer=slots,
                    dist_port=args.dist_port_base + case_idx,
                    cpu_gpu_parallel_execution_enabled=parallel_enabled,
                )
                case_idx += 1
                profile = result.get("engine_profile", {})

                elapsed_sec = float(result.get("elapsed_sec", 0.0))
                latencies.append(elapsed_sec * 1000.0)
                throughput_output.append(float(result.get("throughput_output_tok_s", 0.0)))
                throughput_total.append(float(result.get("throughput_total_tok_s", 0.0)))

                verify_ms = float(profile.get("spec_verify_ms", 0.0))
                verify_count = max(1.0, float(profile.get("spec_run_verify_calls", 0.0)))
                spec_step_ms = float(profile.get("spec_spec_step_ms", 0.0))
                spec_step_count = max(1.0, float(profile.get("spec_spec_step_count", 0.0)))
                verify_per_call.append(verify_ms / verify_count)
                spec_step_per_call.append(spec_step_ms / spec_step_count)
                cpu_route_ratios.append(float(profile.get("model_cpu_route_ratio", 0.0)))
                cpu_weight_mass_ratios.append(float(profile.get("model_cpu_weight_mass_ratio", 0.0)))

                breakdown = _extract_latency_breakdown_from_engine_profile(profile)
                gpu_path_exec_ms_list.append(float(breakdown["gpu_path_exec_ms"]))
                cpu_path_exec_ms_list.append(float(breakdown["cpu_path_exec_ms"]))
                wait_ms_list.append(float(breakdown["wait_ms"]))
                sync_barrier_ms_list.append(float(breakdown["sync_barrier_ms"]))
                moe_wall_ms_list.append(float(breakdown["moe_wall_ms"]))

            latency_mean = _mean(latencies)
            gpu_path_exec_ms_mean = _mean(gpu_path_exec_ms_list)
            cpu_path_exec_ms_mean = _mean(cpu_path_exec_ms_list)
            wait_ms_mean = _mean(wait_ms_list)
            sync_barrier_ms_mean = _mean(sync_barrier_ms_list)
            moe_wall_ms_mean = _mean(moe_wall_ms_list)
            other_overhead_ms_mean = max(0.0, latency_mean - moe_wall_ms_mean)
            denom = latency_mean if latency_mean > 0 else 1.0

            row = {
                "cpu_expert_set_ratio": cpu_ratio,
                "slots_per_layer": slots,
                "parallel_enabled": bool(parallel_enabled),
                "latency_ms_p50": _percentile(latencies, 0.5),
                "latency_ms_p95": _percentile(latencies, 0.95),
                "latency_ms_mean": latency_mean,
                "throughput_output_tok_s_mean": _mean(throughput_output),
                "throughput_total_tok_s_mean": _mean(throughput_total),
                "verify_ms_per_call_mean": _mean(verify_per_call),
                "spec_step_ms_per_call_mean": _mean(spec_step_per_call),
                "cpu_route_ratio": _mean(cpu_route_ratios),
                "cpu_weight_mass_ratio": _mean(cpu_weight_mass_ratios),
                "latency_breakdown_gpu_path_exec_ms": gpu_path_exec_ms_mean,
                "latency_breakdown_cpu_path_exec_ms": cpu_path_exec_ms_mean,
                "latency_breakdown_wait_ms": wait_ms_mean,
                "latency_breakdown_sync_barrier_ms": sync_barrier_ms_mean,
                "latency_breakdown_moe_wall_ms": moe_wall_ms_mean,
                "latency_breakdown_other_overhead_ms": other_overhead_ms_mean,
                "latency_breakdown_gpu_path_ratio": gpu_path_exec_ms_mean / denom,
                "latency_breakdown_cpu_path_ratio": cpu_path_exec_ms_mean / denom,
                "latency_breakdown_wait_ratio": wait_ms_mean / denom,
                "latency_breakdown_sync_barrier_ratio": sync_barrier_ms_mean / denom,
                "latency_breakdown_other_overhead_ratio": other_overhead_ms_mean / denom,
            }
            rows.append(row)
            per_mode[parallel_enabled] = row

        if False in per_mode and True in per_mode:
            serial_mean = float(per_mode[False]["latency_ms_mean"])
            parallel_mean = float(per_mode[True]["latency_ms_mean"])
            curves.append(
                {
                    "cpu_expert_set_ratio": cpu_ratio,
                    "verify_latency_speedup_parallel_vs_serial": (
                        serial_mean / parallel_mean if parallel_mean > 0 else 0.0
                    ),
                }
            )

    report = {
        "config": {
            "cpu_ratios": ratios,
            "num_experts": args.num_experts,
            "parallel_settings": parallel_settings,
            "cpu_expert_parallel_mode": args.cpu_expert_parallel_mode,
            "cpu_expert_num_threads": args.cpu_expert_num_threads,
            "cpu_gpu_parallel_min_cpu_route_ratio": args.cpu_gpu_parallel_min_cpu_route_ratio,
            "num_seqs": args.num_seqs,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "enforce_eager": args.enforce_eager,
            "repeat": args.repeat,
        },
        "results": rows,
        "curves": curves,
    }

    text = json.dumps(report, ensure_ascii=True, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main()
