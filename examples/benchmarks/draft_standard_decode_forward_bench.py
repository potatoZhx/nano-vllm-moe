import argparse
import json
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


def _extract_last_json_line(stdout: str) -> dict:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("No output from heterogeneous_benchmark_case.py")
    return json.loads(lines[-1])


class _PortInUseError(RuntimeError):
    pass


def _classify_failure(stderr: str) -> str:
    if "EADDRINUSE" in stderr:
        return "port_in_use"
    if "OutOfMemoryError" in stderr or "CUDA out of memory" in stderr:
        return "oom"
    if "operation not permitted when stream is capturing" in stderr:
        return "graph_capture_unsafe_op"
    if "CUDA error" in stderr and "capture" in stderr.lower():
        return "graph_capture_cuda_error"
    return "unknown"


def run_case(
    case_script: Path,
    args: argparse.Namespace,
    mode: str,
    dist_port: int,
) -> tuple[dict, int]:
    max_retry = max(0, int(getattr(args, "port_retry", 8)))
    timeout_sec = max(1, int(getattr(args, "case_timeout_sec", 1800)))
    last_error: Exception | None = None

    for retry in range(max_retry + 1):
        current_port = dist_port + retry
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
            "--dist-port",
            str(current_port),
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
            "--return-token-ids",
            "false",
            "--return-text",
            "false",
            "--return-prompts",
            "false",
        ]
        try:
            proc = subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=timeout_sec)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Case timeout: mode={mode}, dist_port={current_port}, timeout_sec={timeout_sec}. "
                "This usually means model load/inference stalled under current resource pressure."
            ) from exc

        if proc.returncode == 0:
            return _extract_last_json_line(proc.stdout), current_port

        failure_kind = _classify_failure(proc.stderr)
        if failure_kind == "port_in_use" and retry < max_retry:
            last_error = _PortInUseError(
                f"Case failed: mode={mode}, dist_port={current_port} (EADDRINUSE), retrying next port"
            )
            continue

        reason_hint = {
            "oom": "GPU out of memory. Please choose a less occupied GPU (e.g. CUDA_VISIBLE_DEVICES=3) or reduce model load pressure.",
            "graph_capture_unsafe_op": "Graph capture hit a non-capture-safe op. Draft/standard CUDA graph cannot be enabled under current kernels/ops.",
            "graph_capture_cuda_error": "CUDA graph capture failed. Check graph-safe constraints (draft_top_c=0, no CPU route in draft, fixed template shapes).",
            "unknown": "See STDERR details below.",
        }[failure_kind]
        raise RuntimeError(
            f"Case failed: mode={mode}, dist_port={current_port}, failure_kind={failure_kind}. {reason_hint}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    if last_error is not None:
        raise RuntimeError(str(last_error))
    raise RuntimeError("Case failed before execution")


def extract_standard_decode_metrics(case_result: dict) -> dict:
    profile = case_result.get("engine_profile") or {}
    decode_runner_ms = float(profile.get("decode_runner_ms", 0.0))
    decode_step_count = int(profile.get("decode_step_count", 0))

    forward_ms = (decode_runner_ms / decode_step_count) if decode_step_count > 0 else 0.0
    tokens_per_forward = float(case_result.get("num_seqs", 0))
    forward_tok_s = (tokens_per_forward * 1000.0 / forward_ms) if forward_ms > 0 else 0.0

    return {
        "decode_runner_ms_total": decode_runner_ms,
        "decode_step_count": decode_step_count,
        "forward_ms": forward_ms,
        "tokens_per_forward": tokens_per_forward,
        "forward_tok_s": forward_tok_s,
    }


def extract_draft_forward_metrics(case_result: dict) -> dict:
    profile = case_result.get("engine_profile") or {}
    draft_infer_ms_total = float(profile.get("spec_run_draft_infer_ms_total", 0.0))
    draft_calls = int(profile.get("spec_run_draft_calls", 0))
    draft_tokens_total = float(profile.get("spec_draft_tokens_total", 0.0))

    forward_ms = (draft_infer_ms_total / draft_calls) if draft_calls > 0 else 0.0
    if draft_calls > 0 and draft_tokens_total > 0:
        tokens_per_forward = draft_tokens_total / draft_calls
    else:
        tokens_per_forward = float(case_result.get("num_seqs", 0))
    forward_tok_s = (tokens_per_forward * 1000.0 / forward_ms) if forward_ms > 0 else 0.0

    return {
        "draft_infer_ms_total": draft_infer_ms_total,
        "draft_calls": draft_calls,
        "draft_tokens_total": draft_tokens_total,
        "forward_ms": forward_ms,
        "tokens_per_forward": tokens_per_forward,
        "forward_tok_s": forward_tok_s,
    }


def validate_cuda_graph_usage(standard_result: dict, spec_result: dict, enforce_eager: bool) -> None:
    if enforce_eager:
        raise RuntimeError("This benchmark requires CUDA Graph. Please set --enforce-eager false.")

    standard_profile = standard_result.get("engine_profile") or {}
    spec_profile = spec_result.get("engine_profile") or {}

    standard_replays = int(standard_profile.get("model_standard_graph_replay_count", 0))
    draft_replays = int(spec_profile.get("model_draft_graph_replay_count", 0))

    if standard_replays <= 0:
        raise RuntimeError(
            "Standard decode CUDA Graph was not replayed. "
            "Possible causes: graph capture failed, decode path fell back to eager, or batch shape missed templates."
        )
    if draft_replays <= 0:
        raise RuntimeError(
            "Draft CUDA Graph was not replayed under draft_top_c=0. "
            "Possible causes: draft graph capture failed, policy gate blocked graph, or draft batch missed templates."
        )


def validate_deterministic_alignment(standard_result: dict, spec_result: dict, temperature: float) -> bool:
    if temperature > 0.0:
        return True
    standard_digest = standard_result.get("outputs_digest", "")
    spec_digest = spec_result.get("outputs_digest", "")
    exact_match = bool(standard_digest) and (standard_digest == spec_digest)
    if not exact_match:
        raise RuntimeError(
            "Deterministic mismatch between standard and spec outputs under temperature=0. "
            "This indicates potential correctness regression in draft/verify path."
        )
    return exact_match


def summarize_repeats(rows: list[dict]) -> dict:
    standard_ms = [float(row["standard_decode"]["forward_ms"]) for row in rows]
    draft_ms = [float(row["draft_forward"]["forward_ms"]) for row in rows]
    standard_tok_s = [float(row["standard_decode"]["forward_tok_s"]) for row in rows]
    draft_tok_s = [float(row["draft_forward"]["forward_tok_s"]) for row in rows]

    std_ms_med = statistics.median(standard_ms) if standard_ms else 0.0
    draft_ms_med = statistics.median(draft_ms) if draft_ms else 0.0
    std_tps_med = statistics.median(standard_tok_s) if standard_tok_s else 0.0
    draft_tps_med = statistics.median(draft_tok_s) if draft_tok_s else 0.0

    return {
        "standard_decode_forward_ms_median": std_ms_med,
        "draft_forward_ms_median": draft_ms_med,
        "draft_over_standard_ms_ratio": (draft_ms_med / std_ms_med) if std_ms_med > 0 else 0.0,
        "standard_decode_forward_tok_s_median": std_tps_med,
        "draft_forward_tok_s_median": draft_tps_med,
        "draft_over_standard_tok_s_ratio": (draft_tps_med / std_tps_med) if std_tps_med > 0 else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark: standard decode forward vs spec draft forward")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--slots-per-layer", type=int, default=0)
    parser.add_argument("--num-seqs", type=int, default=8)
    parser.add_argument("--input-len", type=int, default=64)
    parser.add_argument("--output-len", type=int, default=24)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.99)
    parser.add_argument("--max-draft-tokens", type=int, default=4)
    parser.add_argument("--draft-top-c", type=int, default=0)
    parser.add_argument("--dist-port-base", type=int, default=29100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--enforce-eager", type=str2bool, default=False)
    parser.add_argument("--engine-profile-cuda-sync", type=str2bool, default=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--port-retry", type=int, default=8)
    parser.add_argument("--case-timeout-sec", type=int, default=1800)
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be >= 1")

    case_script = Path(__file__).resolve().parents[1] / "heterogeneous_benchmark_case.py"
    rows: list[dict] = []

    for i in range(args.repeats):
        base_port = args.dist_port_base + (i * 2)
        standard_result, standard_port = run_case(
            case_script,
            args,
            mode="standard",
            dist_port=base_port,
        )
        spec_result, spec_port = run_case(
            case_script,
            args,
            mode="spec",
            dist_port=base_port + 1,
        )
        validate_cuda_graph_usage(standard_result, spec_result, enforce_eager=bool(args.enforce_eager))
        exact_match = validate_deterministic_alignment(
            standard_result,
            spec_result,
            temperature=float(args.temperature),
        )

        rows.append(
            {
                "repeat_index": i,
                "standard_dist_port": int(standard_port),
                "spec_dist_port": int(spec_port),
                "deterministic_exact_match": bool(exact_match),
                "standard_outputs_digest": standard_result.get("outputs_digest", ""),
                "spec_outputs_digest": spec_result.get("outputs_digest", ""),
                "standard_decode": extract_standard_decode_metrics(standard_result),
                "draft_forward": extract_draft_forward_metrics(spec_result),
                "standard_throughput_output_tok_s": float(standard_result.get("throughput_output_tok_s", 0.0)),
                "spec_throughput_output_tok_s": float(spec_result.get("throughput_output_tok_s", 0.0)),
            }
        )

    report = {
        "config": {
            "model_path": args.model_path,
            "slots_per_layer": args.slots_per_layer,
            "num_seqs": args.num_seqs,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_draft_tokens": args.max_draft_tokens,
            "draft_top_c": args.draft_top_c,
            "seed": args.seed,
            "temperature": args.temperature,
            "enforce_eager": args.enforce_eager,
            "engine_profile_cuda_sync": args.engine_profile_cuda_sync,
            "repeats": args.repeats,
            "port_retry": args.port_retry,
            "case_timeout_sec": args.case_timeout_sec,
        },
        "summary": summarize_repeats(rows),
        "repeats": rows,
    }

    text = json.dumps(report, ensure_ascii=True, indent=2)
    print(text)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"Saved report to: {out_path}")


if __name__ == "__main__":
    main()