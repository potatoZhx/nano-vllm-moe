#!/usr/bin/env python3
"""Run the K12 single-request and MT-Bench latency-breakdown experiment."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from latency_breakdown import (
    aggregate_breakdowns,
    build_request_breakdown,
    flatten_request,
    render_markdown,
    write_csv,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = REPO_ROOT / "scripts" / "bench_eval_workload_tpot.py"
MODEL_PATH = Path("/data1/models/Qwen3-30B-A3B")
MT_BENCH_PATH = Path("/data1/datasets/mt_bench/question.jsonl")
VERIFY_COST_PATH = (
    REPO_ROOT
    / "results/transfer_v3_artifact_20260719/verify_cost_v3.json"
)
REFERENCE_ROWS = (
    REPO_ROOT
    / "results/transfer_v3_active_screen_u000_20260719"
    / "datasets/mt_bench/rows.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results/k12_latency_breakdown"
EXPECTED_OUTPUT_TOKENS = 512
EXPECTED_MT_BENCH_REQUESTS = 80
DEFAULT_CPU_COUNT = 33
SEED = 20260719


def _run_text(command: list[str]) -> str:
    return subprocess.check_output(
        command, text=True, stderr=subprocess.STDOUT
    )


def _gpu_inventory() -> list[dict[str, Any]]:
    output = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,memory.used,memory.total,"
            "utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    processes: set[str] = set()
    try:
        process_output = _run_text(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ]
        )
        for line in process_output.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if fields and fields[0]:
                processes.add(fields[0])
    except subprocess.CalledProcessError:
        pass

    rows = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            continue
        rows.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "bus_id": fields[2],
                "memory_used_mib": int(fields[3]),
                "memory_total_mib": int(fields[4]),
                "utilization": int(fields[5]),
                "has_compute_process": fields[1] in processes,
            }
        )
    return rows


def choose_gpu(requested: str) -> dict[str, Any]:
    inventory = _gpu_inventory()
    if not inventory:
        raise RuntimeError("nvidia-smi returned no GPUs")
    if requested != "auto":
        index = int(requested)
        matches = [row for row in inventory if row["index"] == index]
        if not matches:
            raise ValueError(f"GPU {index} does not exist")
        selected = matches[0]
        if selected["has_compute_process"]:
            raise RuntimeError(f"GPU {index} has an active compute process")
        return selected
    candidates = [
        row
        for row in inventory
        if not row["has_compute_process"]
        and row["memory_used_mib"] <= 512
        and row["utilization"] <= 5
    ]
    if not candidates:
        rendered = ", ".join(
            f"gpu{row['index']}:mem={row['memory_used_mib']}MiB,"
            f"util={row['utilization']}%,proc={row['has_compute_process']}"
            for row in inventory
        )
        raise RuntimeError(f"no idle GPU is available ({rendered})")
    return min(
        candidates,
        key=lambda row: (
            row["memory_used_mib"],
            row["utilization"],
            row["index"],
        ),
    )


def gpu_numa_node(gpu: dict[str, Any]) -> int:
    bus_id = str(gpu["bus_id"])
    domain, bus, device = bus_id.split(":", 2)
    normalized_bus_id = (
        f"{int(domain, 16):04x}:{bus.lower()}:{device.lower()}"
    )
    candidates = [
        Path("/sys/bus/pci/devices") / bus_id.lower(),
        Path("/sys/bus/pci/devices") / normalized_bus_id,
    ]
    for path in candidates:
        numa_path = path / "numa_node"
        if numa_path.is_file():
            node = int(numa_path.read_text(encoding="utf-8").strip())
            if node >= 0:
                return node
    raise RuntimeError(
        f"cannot resolve NUMA node for GPU PCI bus {bus_id}"
    )


def _read_cpu_busy() -> dict[int, tuple[int, int]]:
    rows: dict[int, tuple[int, int]] = {}
    with Path("/proc/stat").open(encoding="utf-8") as handle:
        for line in handle:
            match = re.match(r"cpu(\d+)\s+(.+)", line)
            if not match:
                continue
            values = [int(value) for value in match.group(2).split()]
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            rows[int(match.group(1))] = (sum(values), idle)
    return rows


def _cpu_utilization_sample() -> dict[int, float]:
    before = _read_cpu_busy()
    time.sleep(0.2)
    after = _read_cpu_busy()
    result = {}
    for cpu, (total_after, idle_after) in after.items():
        total_before, idle_before = before.get(
            cpu, (total_after, idle_after)
        )
        total_delta = max(1, total_after - total_before)
        idle_delta = max(0, idle_after - idle_before)
        result[cpu] = 1.0 - idle_delta / total_delta
    return result


def choose_cpu_list(numa_node: int, requested: str) -> str:
    if requested != "auto":
        completed = subprocess.run(
            ["taskset", "--cpu-list", requested, "true"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode:
            raise ValueError(
                f"invalid CPU list {requested!r}: "
                f"{completed.stderr.strip()}"
            )
        return requested

    output = _run_text(["lscpu", "-p=CPU,CORE,SOCKET,NODE"])
    representatives: dict[tuple[int, int], int] = {}
    for line in output.splitlines():
        if not line or line.startswith("#"):
            continue
        cpu, core, socket, node = (
            int(value) for value in line.split(",")
        )
        if node != numa_node:
            continue
        representatives.setdefault((socket, core), cpu)
    if len(representatives) < DEFAULT_CPU_COUNT:
        raise RuntimeError(
            f"NUMA node {numa_node} has only "
            f"{len(representatives)} visible physical cores"
        )
    utilization = _cpu_utilization_sample()
    selected = sorted(
        representatives.values(),
        key=lambda cpu: (utilization.get(cpu, 1.0), cpu),
    )[:DEFAULT_CPU_COUNT]
    return ",".join(str(cpu) for cpu in sorted(selected))


def python_executable() -> str:
    configured = Path(
        "/home/linke/miniconda3/envs/nano_moe/bin/python"
    )
    return str(configured if configured.is_file() else Path(sys.executable))


def benchmark_command(
    *,
    phase: str,
    result_dir: Path,
    cpu_list: str,
    port: int,
    resume: bool,
) -> list[str]:
    num_samples = "1" if phase == "single" else "all"
    return [
        "taskset",
        "--cpu-list",
        cpu_list,
        python_executable(),
        str(BENCHMARK),
        "--model-path",
        str(MODEL_PATH),
        "--dataset",
        "mt_bench",
        "--mt-bench-path",
        str(MT_BENCH_PATH),
        "--request-mode",
        "dataset",
        "--num-samples",
        num_samples,
        "--optimized-config",
        "k12_transfer_step",
        "--draft-tpot-verify-model-path",
        str(VERIFY_COST_PATH),
        "--output-lens",
        str(EXPECTED_OUTPUT_TOKENS),
        "--gpu-memory-utilization",
        "0.99",
        "--temperature",
        "0.8",
        "--acceptance-strategy",
        "standard_sampling",
        "--decode-driver",
        "generate",
        "--latency-breakdown-profile",
        "true",
        "--collect-profile",
        "true",
        "--engine-profile",
        "true",
        "--engine-profile-cuda-sync",
        "false",
        "--save-profile-json",
        "true",
        "--save-token-ids",
        "true",
        "--save-text",
        "true",
        "--reset-profile-after-warmup",
        "true",
        "--reset-profile-before-request",
        "true",
        "--reset-seed-after-warmup",
        "true",
        "--repeats",
        "1",
        "--seed",
        str(SEED),
        "--dist-port-base",
        str(port),
        "--skip-existing",
        "true" if resume else "false",
        "--fail-fast",
        "true",
        "--fail-on-output-validation-error",
        "true",
        "--output-dir",
        str(result_dir),
    ]


def run_logged(
    command: list[str],
    *,
    gpu_index: int,
    log_path: Path,
    dry_run: bool,
) -> int:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    env["PYTHONPATH"] = (
        str(REPO_ROOT)
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    rendered = (
        f"CUDA_VISIBLE_DEVICES={gpu_index} "
        + shlex.join(command)
    )
    print(f"$ {rendered}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log_path.write_text(rendered + "\n", encoding="utf-8")
        return 0
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"$ {rendered}\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            print(line, end="", flush=True)
        return process.wait()


def load_reference_digests() -> dict[str, str]:
    if not REFERENCE_ROWS.is_file():
        return {}
    with REFERENCE_ROWS.open(encoding="utf-8", newline="") as handle:
        return {
            str(row.get("sample_id", "")): str(
                row.get("outputs_digest", "")
            )
            for row in csv.DictReader(handle)
            if row.get("status") == "ok"
            and row.get("sample_id")
            and row.get("outputs_digest")
        }


def resolve_profile_path(
    result_dir: Path, configured: str
) -> Path:
    path = Path(configured)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if path.is_file():
        return path
    matches = sorted(result_dir.glob("*_profiles/sample*.json"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"profile JSON does not exist: {path}")


def analyze_phase(
    *,
    phase: str,
    result_dir: Path,
    analysis_dir: Path,
) -> dict[str, Any]:
    summary_path = result_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"benchmark did not produce {summary_path}"
        )
    benchmark_summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )
    rows = [
        row
        for row in benchmark_summary.get("rows", [])
        if isinstance(row, dict) and row.get("status") == "ok"
    ]
    expected = 1 if phase == "single" else EXPECTED_MT_BENCH_REQUESTS
    if len(rows) != expected:
        raise RuntimeError(
            f"{phase} produced {len(rows)} successful rows, expected {expected}"
        )

    references = load_reference_digests()
    requests = []
    for row in rows:
        profile_path = resolve_profile_path(
            result_dir, str(row.get("profile_json", ""))
        )
        profile = json.loads(
            profile_path.read_text(encoding="utf-8")
        )
        breakdown = build_request_breakdown(row, profile)
        reference_digest = references.get(
            str(row.get("sample_id", ""))
        )
        breakdown["reference_digest"] = reference_digest or ""
        breakdown["digest_matches_reference"] = (
            not reference_digest
            or reference_digest
            == str(row.get("outputs_digest", ""))
        )
        if not breakdown["digest_matches_reference"]:
            breakdown["warnings"].append(
                "output digest differs from the historical K12 reference; "
                "temperature=0.8 plus timing-sensitive dynamic draft length "
                "changes RNG consumption, so the digest is diagnostic rather "
                "than a correctness gate"
            )
        if int(row.get("generated_output_tokens", 0) or 0) != (
            EXPECTED_OUTPUT_TOKENS
        ):
            breakdown["errors"].append(
                f"generated_output_tokens="
                f"{row.get('generated_output_tokens')} expected="
                f"{EXPECTED_OUTPUT_TOKENS}"
            )
            breakdown["passed"] = False
        breakdown["profile_json"] = str(profile_path)
        requests.append(breakdown)

    aggregate = aggregate_breakdowns(requests)
    aggregate["phase"] = phase
    aggregate["benchmark_summary_json"] = str(summary_path)
    aggregate["requests"] = requests
    write_json(analysis_dir / "requests.json", requests)
    write_csv(
        analysis_dir / "requests.csv",
        [flatten_request(item) for item in requests],
    )
    write_json(analysis_dir / "summary.json", aggregate)
    (analysis_dir / "summary.md").write_text(
        render_markdown(aggregate), encoding="utf-8"
    )
    return aggregate


def phase_complete(output_dir: Path, phase: str) -> bool:
    path = output_dir / phase / "analysis" / "summary.json"
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(value.get("passed"))


def execute_phase(
    *,
    phase: str,
    output_dir: Path,
    gpu: dict[str, Any],
    cpu_list: str,
    port: int,
    resume: bool,
    dry_run: bool,
) -> dict[str, Any]:
    phase_dir = output_dir / phase
    if resume and phase_complete(output_dir, phase):
        print(f"{phase}: existing passing result reused", flush=True)
        return json.loads(
            (
                phase_dir / "analysis" / "summary.json"
            ).read_text(encoding="utf-8")
        )
    result_dir = phase_dir / "benchmark"
    command = benchmark_command(
        phase=phase,
        result_dir=result_dir,
        cpu_list=cpu_list,
        port=port,
        resume=resume,
    )
    write_json(
        phase_dir / "manifest.json",
        {
            "phase": phase,
            "gpu": gpu,
            "cpu_list": cpu_list,
            "command": command,
            "started_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
        },
    )
    return_code = run_logged(
        command,
        gpu_index=int(gpu["index"]),
        log_path=phase_dir / "run.log",
        dry_run=dry_run,
    )
    if dry_run:
        return {"phase": phase, "passed": True, "dry_run": True}
    if return_code:
        raise RuntimeError(
            f"{phase} benchmark exited with code {return_code}"
        )
    aggregate = analyze_phase(
        phase=phase,
        result_dir=result_dir,
        analysis_dir=phase_dir / "analysis",
    )
    print(
        f"{phase}: {'PASS' if aggregate['passed'] else 'FAIL'}; "
        f"TPOT={aggregate['pooled_per_token_ms']['tpot']:.4f} ms/token; "
        f"report={phase_dir / 'analysis' / 'summary.md'}",
        flush=True,
    )
    return aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed k12_transfer_step TPOT latency breakdown. "
            "Phase all gates full MT-Bench on the single-request result."
        )
    )
    parser.add_argument(
        "--phase",
        choices=["single", "mt_bench", "all"],
        default="all",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default="auto",
        help="Physical GPU index or auto.",
    )
    parser.add_argument(
        "--cpu-list",
        default="auto",
        help="taskset CPU list or auto for idle same-NUMA physical cores.",
    )
    parser.add_argument("--dist-port-base", type=int, default=37980)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    for required in (
        BENCHMARK,
        MODEL_PATH,
        MT_BENCH_PATH,
        VERIFY_COST_PATH,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    gpu = choose_gpu(str(args.cuda_visible_devices))
    numa_node = gpu_numa_node(gpu)
    cpu_list = choose_cpu_list(numa_node, str(args.cpu_list))
    gpu["numa_node"] = numa_node
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "run_manifest.json",
        {
            "gpu": gpu,
            "cpu_list": cpu_list,
            "phase": args.phase,
            "resume": bool(args.resume),
            "seed": SEED,
        },
    )
    print(
        f"selected GPU {gpu['index']} (NUMA {numa_node}); "
        f"CPU list {cpu_list}",
        flush=True,
    )

    phases = (
        ["single", "mt_bench"]
        if args.phase == "all"
        else [args.phase]
    )
    for index, phase in enumerate(phases):
        result = execute_phase(
            phase=phase,
            output_dir=args.output_dir,
            gpu=gpu,
            cpu_list=cpu_list,
            port=int(args.dist_port_base) + index,
            resume=bool(args.resume),
            dry_run=bool(args.dry_run),
        )
        if phase == "single" and not bool(result.get("passed")):
            print(
                "single-request gate failed; MT-Bench was not started",
                file=sys.stderr,
            )
            return 1
        if not bool(result.get("passed")):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
