#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CASE_ORDER = [
    "direct_prefetch_vlayer_on",
    "direct_cache_fill_no_cpu",
    "direct_prefetch_vlayer_off",
    "direct_prefetch_off",
    "direct_cpu_policy_prefetch_off",
    "probe_sync",
    "probe_nosync",
    "direct_torchprof_l512",
]


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _profile_value(data: dict[str, Any], key: str) -> float:
    return float(data.get("engine_profile", {}).get(key, 0.0) or 0.0)


def _calls(data: dict[str, Any]) -> float:
    return float(data.get("summary", {}).get("verify_calls", 0.0) or data.get("engine_profile", {}).get("spec_run_verify_calls", 0.0) or 0.0)


def _verify_avg(data: dict[str, Any]) -> float:
    summary = data.get("summary", {})
    if "verify_forward_ms_avg" in summary:
        return float(summary["verify_forward_ms_avg"])
    return float(data.get("engine_profile", {}).get("verify_forward_ms", 0.0) or 0.0)


def _case_row(name: str, data: dict[str, Any]) -> dict[str, float | str]:
    calls = _calls(data)
    verify = _verify_avg(data)
    tokens_per_call = float(data.get("summary", {}).get("verify_tokens_per_call", 0.0) or 0.0)
    if tokens_per_call <= 0.0:
        tokens_per_call = _safe_div(_profile_value(data, "model_verify_tokens_in_total"), calls)
    components = {
        "route": _safe_div(_profile_value(data, "model_verify_route_ms"), calls),
        "plan": _safe_div(_profile_value(data, "model_verify_plan_ms"), calls),
        "gpu_gather": _safe_div(_profile_value(data, "model_verify_gpu_gather_ms"), calls),
        "gpu_compute": _safe_div(_profile_value(data, "model_verify_gpu_compute_ms"), calls),
        "cpu_compute": _safe_div(_profile_value(data, "model_verify_cpu_compute_ms"), calls),
        "cpu_merge": _safe_div(_profile_value(data, "model_verify_cpu_to_gpu_merge_ms"), calls),
        "scatter": _safe_div(_profile_value(data, "model_verify_scatter_ms"), calls),
        "cache_fill": _safe_div(_profile_value(data, "model_verify_cache_fill_transfer_ms"), calls),
        "no_cpu_remaining_miss": _safe_div(
            _profile_value(data, "model_verify_cache_fill_no_cpu_remaining_miss_count"),
            calls,
        ),
    }
    residual = verify - sum(components.values())
    return {
        "name": name,
        "verify": verify,
        "calls": calls,
        "tokens_per_call": tokens_per_call,
        "throughput": float(data.get("throughput_output_tok_s", 0.0) or 0.0),
        "residual": residual,
        **components,
    }


def summarize(result_dir: Path) -> str:
    rows = []
    for name in CASE_ORDER:
        path = result_dir / f"{name}.json"
        if path.exists():
            rows.append(_case_row(name, _load(path)))

    lines = [
        f"# Verify Profile Summary: `{result_dir}`",
        "",
        "## Case Comparison",
        "",
        "| case | verify ms/call | calls | tok/call | tok/s | plan | GPU expert | CPU expert | cache-fill | no-cpu miss/call | residual |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['name']} | "
            f"{row['verify']:.3f} | "
            f"{row['calls']:.0f} | "
            f"{row['tokens_per_call']:.2f} | "
            f"{row['throughput']:.2f} | "
            f"{row['plan']:.2f} | "
            f"{row['gpu_compute']:.2f} | "
            f"{row['cpu_compute']:.2f} | "
            f"{row['cache_fill']:.2f} | "
            f"{row['no_cpu_remaining_miss']:.2f} | "
            f"{row['residual']:.2f} |"
        )

    current = next((row for row in rows if row["name"] == "direct_prefetch_vlayer_on"), None)
    if current is not None:
        verify = float(current["verify"])
        lines.extend(
            [
                "",
                "## Current Path Breakdown",
                "",
                "| bucket | ms / verify call | percent |",
                "|---|---:|---:|",
            ]
        )
        for key, label in [
            ("route", "route"),
            ("plan", "MoE plan"),
            ("gpu_gather", "GPU gather"),
            ("gpu_compute", "GPU expert compute"),
            ("cpu_compute", "CPU expert compute"),
            ("cpu_merge", "CPU merge"),
            ("scatter", "scatter"),
            ("cache_fill", "cache-fill transfer"),
            ("residual", "forward residual"),
        ]:
            value = float(current[key])
            lines.append(f"| {label} | {value:.3f} | {_safe_div(value, verify) * 100.0:.1f}% |")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize verify-profile result JSON files.")
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    text = summarize(args.result_dir)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
