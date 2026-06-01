#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROMPT_TEXT = (
    "A mixture-of-experts (MoE) transformer differs from a standard dense transformer "
    "primarily in its feed-forward layers. In a dense transformer, every token activates "
    "all parameters in each feed-forward block. In an MoE transformer, each token is "
    "routed to only a small subset of expert sub-networks. This conditional computation "
    "allows MoE models to scale to much larger parameter counts without proportionally "
    "increasing the FLOPs per token.\n\n"
    "The routing mechanism typically uses a learned gating network that produces a "
    "probability distribution over experts for each token. The top-K experts are selected "
    "and their outputs are weighted by the routing probabilities. The key challenge is "
    "load balancing: if all tokens route to the same few experts, those experts become "
    "bottlenecks while others sit idle. Auxiliary loss terms penalize imbalanced routing "
    "during training.\n\n"
    "During inference, expert caching becomes critical for deployment efficiency. "
    "Since GPU memory is limited, only a subset of expert weights can be kept in GPU "
    "memory at any time. The remaining experts reside in CPU memory and must be "
    "transferred to GPU when needed. This heterogeneous execution model trades "
    "memory capacity for transfer latency."
)


MAX_DRAFT_BY_RATIO = {
    0.25: 3,
    0.5: 5,
    0.75: 7,
}


def assess_text_quality(text: str) -> dict[str, Any]:
    reasons: list[str] = []
    stripped = text.strip()
    if not stripped:
        reasons.append("empty")
    if "\ufffd" in text:
        reasons.append("replacement_char")
    if any((ord(ch) < 32 and ch not in "\n\r\t") for ch in text):
        reasons.append("control_char")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 4:
        most_common = max(lines.count(line) for line in set(lines))
        if most_common / len(lines) >= 0.5:
            reasons.append("repeated_lines")

    words = text.lower().split()
    if len(words) >= 80:
        window = 12
        grams = [" ".join(words[i : i + window]) for i in range(0, len(words) - window + 1)]
        if grams:
            max_repeat = max(grams.count(gram) for gram in set(grams))
            if max_repeat >= 3:
                reasons.append("repeated_12gram")

    return {"ok": not reasons, "reasons": reasons}


def _ratio_key(ratio: float) -> float:
    return round(float(ratio), 2)


def _case_name(policy: str, out_len: int, ratio: float) -> str:
    return f"{policy}_ratio{int(round(ratio * 100))}_l{out_len}"


def _row_from_raw(name: str, raw: dict[str, Any], quality: dict[str, Any]) -> dict[str, Any]:
    summary = raw.get("summary", {})
    case = raw.get("case", {})
    ep = raw.get("engine_profile", {})
    acceptance = summary.get("acceptance", {})
    cache = summary.get("cache", {})
    cuda_graph = summary.get("cuda_graph", {})
    prefetch = summary.get("prefetch", {})
    verify_cache_fill = summary.get("verify_cache_fill", {})
    return {
        "name": name,
        "spec_verify_miss_policy": case.get("spec_verify_miss_policy", "cpu"),
        "cache_ratio": case.get("cache_ratio"),
        "output_len": case.get("output_len"),
        "max_draft_tokens": case.get("max_draft_tokens"),
        "acceptance_rate": acceptance.get("acceptance_rate", 0.0),
        "cache_route_hit_rate": cache.get("route_hit_rate", 0.0),
        "cache_weight_hit_rate": cache.get("weight_hit_rate", 0.0),
        "activated_expert_set_size": cache.get("activated_expert_set_size", 0.0),
        "realized_cpu_expert_count": cache.get("realized_cpu_expert_count", 0.0),
        "throughput_output_tok_s": summary.get("throughput_output_tok_s", 0.0),
        "decode_phase_output_tok_s": summary.get("decode_phase_output_tok_s", 0.0),
        "draft_forward_ms_avg": summary.get("draft_forward_ms_avg", 0.0),
        "verify_forward_ms_avg": summary.get("verify_forward_ms_avg", 0.0),
        "draft_graph_replay_count": cuda_graph.get("draft_replay_count", 0),
        "total_graph_replay_count": cuda_graph.get("total_replay_count", 0),
        "graph_hit_rate": cuda_graph.get("hit_rate", 0.0),
        "prefetch_submit_count": prefetch.get("submit_count", 0),
        "prefetch_completed_count": prefetch.get("completed_count", 0),
        "prefetch_consumed_count": prefetch.get("consumed_count", 0),
        "draft_segment_indexed_submit_count": int(ep.get("model_draft_segment_indexed_prefetch_submit_count", 0) or 0),
        "draft_segment_indexed_success_count": int(ep.get("model_draft_segment_indexed_prefetch_success_count", 0) or 0),
        "draft_segment_indexed_consumed_count": int(ep.get("model_draft_segment_indexed_prefetch_consumed_count", 0) or 0),
        "verify_layer_prefetch_submit_count": int(ep.get("model_verify_layer_prefetch_submit_count", 0) or 0),
        "verify_layer_prefetch_done_count": int(ep.get("model_verify_layer_prefetch_publish_count", 0) or 0),
        "verify_layer_prefetch_used_count": int(ep.get("model_verify_layer_prefetch_consumed_count", 0) or 0),
        "verify_cache_fill_promoted_expert_count": verify_cache_fill.get("promoted_expert_count", 0),
        "verify_cache_fill_cpu_expert_count": verify_cache_fill.get("cpu_expert_count", 0),
        "verify_cache_fill_evicted_expert_count": verify_cache_fill.get("evicted_expert_count", 0),
        "verify_cache_fill_transfer_ms_total": verify_cache_fill.get("transfer_ms_total", 0.0),
        "elapsed_sec": raw.get("elapsed_sec", 0.0),
        "generated_output_tokens": raw.get("generated_output_tokens", 0),
        "outputs_digest": raw.get("outputs_digest", ""),
        "text_quality_ok": quality["ok"],
        "text_quality_reasons": quality["reasons"],
        "generated_text_sample": "\n".join(str(x) for x in raw.get("generated_text", []))[:1200],
    }


def run_case(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    prompt_file: Path,
    policy: str,
    out_len: int,
    ratio: float,
    case_index: int,
) -> dict[str, Any]:
    name = _case_name(policy, out_len, ratio)
    outdir = Path(args.output_dir)
    case_json = outdir / f"{name}.json"
    case_log = outdir / f"{name}.log"
    script_path = repo_root / "benchmarks" / "scripts" / "spec_verify_expert_count_stats.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    max_draft_tokens = MAX_DRAFT_BY_RATIO[_ratio_key(ratio)]

    cmd = [
        sys.executable,
        str(script_path),
        "--single-case",
        "--model-path",
        args.model_path,
        "--prompt-text-file",
        str(prompt_file),
        "--cache-ratio",
        str(ratio),
        "--slots-per-layer",
        "0",
        "--prefetch-enabled",
        "true",
        "--prefetch-runtime-mode",
        "draft_segment_indexed",
        "--prefetch-staging-slots-per-layer",
        "0",
        "--output",
        str(case_json),
        "--dist-port",
        str(args.dist_port_base + case_index),
        "--num-seqs",
        "1",
        "--input-len",
        "0",
        "--output-len",
        str(out_len),
        "--max-draft-tokens",
        str(max_draft_tokens),
        "--draft-top-c",
        "0",
        "--draft-reroute-policy",
        "entropy_cache_bias",
        "--draft-reroute-artifact",
        args.profile_artifact,
        "--temperature",
        str(args.temperature),
        "--acceptance-strategy",
        args.acceptance_strategy,
        "--acceptance-threshold",
        str(args.acceptance_threshold),
        "--cache-strategy",
        "lru",
        "--spec-verify-miss-policy",
        policy,
        "--cpu-expert-backend",
        "fused",
        "--cpu-expert-pin-memory",
        str(args.cpu_expert_pin_memory).lower(),
        "--cpu-expert-workspace-max-routes",
        str(args.cpu_expert_workspace_max_routes),
        "--cpu-expert-packed-min-routes",
        "1",
        "--cpu-expert-parallel-mode",
        "serial",
        "--cpu-expert-num-threads",
        str(args.cpu_expert_num_threads),
        "--max-num-batched-tokens",
        "8192",
        "--max-num-seqs",
        "1",
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--enforce-eager",
        "false",
        "--draft-cuda-graph-enabled",
        "true",
        "--draft-cuda-graph-cpu-backend",
        "none",
        "--prefetch-verify-wait-ms",
        str(args.prefetch_verify_wait_ms),
        "--prefetch-step-budget",
        str(args.prefetch_step_budget),
        "--prefetch-max-inflight",
        str(args.prefetch_max_inflight),
        "--cache-eviction-budget-per-step",
        str(args.cache_eviction_budget_per_step),
        "--prefetch-global-queue-capacity",
        str(args.prefetch_global_queue_capacity),
        "--prefetch-use-prefill-history",
        "true",
        "--prefetch-use-verify-history",
        "true",
        "--prefetch-use-draft-live",
        "true",
        "--seed",
        str(args.seed),
        "--sync-layer-timing",
        "true",
    ]

    print(f"[{case_index + 1}] {name}", flush=True)
    started = time.time()
    with case_log.open("w", encoding="utf-8") as log_f:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.case_timeout_sec,
        )
    elapsed = time.time() - started
    if proc.returncode != 0:
        tail = case_log.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"{name} failed after {elapsed:.1f}s with exit={proc.returncode}\n{tail}")

    raw = json.loads(case_json.read_text(encoding="utf-8"))
    generated_texts = raw.get("generated_text") or []
    quality = assess_text_quality("\n".join(str(x) for x in generated_texts))
    row = _row_from_raw(name, raw, quality)
    print(
        f"  accept={row['acceptance_rate']:.4f} hit={row['cache_route_hit_rate']:.4f} "
        f"tok/s={row['throughput_output_tok_s']:.3f} verify={row['verify_forward_ms_avg']:.3f}ms "
        f"prefetch={row['prefetch_submit_count']}/{row['prefetch_completed_count']}/{row['prefetch_consumed_count']} "
        f"text_ok={row['text_quality_ok']}",
        flush=True,
    )
    return row


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Verify Miss Policy Validation",
        "",
        "| policy | ratio | out | max draft | accept | cache hit | output tok/s | decode tok/s | draft ms | verify ms | graph replays | prefetch submit/done/used | verify-layer submit/done/used | cache-fill promoted/cpu/evicted | text |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|:---|:---|:---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['spec_verify_miss_policy']} | "
            f"{float(row['cache_ratio']):.2f} | "
            f"{int(row['output_len'])} | "
            f"{int(row['max_draft_tokens'])} | "
            f"{float(row['acceptance_rate']):.4f} | "
            f"{float(row['cache_route_hit_rate']):.4f} | "
            f"{float(row['throughput_output_tok_s']):.3f} | "
            f"{float(row['decode_phase_output_tok_s']):.3f} | "
            f"{float(row['draft_forward_ms_avg']):.3f} | "
            f"{float(row['verify_forward_ms_avg']):.3f} | "
            f"{int(row['draft_graph_replay_count'])}/{int(row['total_graph_replay_count'])} | "
            f"{int(row['prefetch_submit_count'])}/{int(row['prefetch_completed_count'])}/{int(row['prefetch_consumed_count'])} | "
            f"{int(row['verify_layer_prefetch_submit_count'])}/{int(row['verify_layer_prefetch_done_count'])}/{int(row['verify_layer_prefetch_used_count'])} | "
            f"{int(row['verify_cache_fill_promoted_expert_count'])}/{int(row['verify_cache_fill_cpu_expert_count'])}/{int(row['verify_cache_fill_evicted_expert_count'])} | "
            f"{'ok' if row['text_quality_ok'] else ','.join(row['text_quality_reasons'])} |"
        )

    lines.extend(["", "## Generated Text Samples", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row['name']}",
                "",
                "```text",
                str(row.get("generated_text_sample", "")).strip(),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare verify miss CPU fallback vs cache-fill policy.")
    parser.add_argument("--model-path", default="/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B")
    parser.add_argument(
        "--profile-artifact",
        default="results/reroute_impl_20260531/offline_profile_20260531_203257.safetensors",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--policies", default="cpu,cache_fill")
    parser.add_argument("--output-lens", default="128,512")
    parser.add_argument("--cache-ratios", default="0.25,0.5,0.75")
    parser.add_argument("--acceptance-strategy", default="standard_sampling")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--acceptance-threshold", type=float, default=0.7)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--cpu-expert-workspace-max-routes", type=int, default=16384)
    parser.add_argument("--cpu-expert-num-threads", type=int, default=4)
    parser.add_argument("--cpu-expert-pin-memory", action="store_true")
    parser.add_argument("--prefetch-verify-wait-ms", type=float, default=0.0)
    parser.add_argument("--prefetch-step-budget", type=int, default=4)
    parser.add_argument("--prefetch-max-inflight", type=int, default=8)
    parser.add_argument("--cache-eviction-budget-per-step", type=int, default=2)
    parser.add_argument("--prefetch-global-queue-capacity", type=int, default=4096)
    parser.add_argument("--dist-port-base", type=int, default=29500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--case-timeout-sec", type=int, default=1800)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if not args.output_dir:
        args.output_dir = str(repo_root / "results" / f"verify_miss_policy_{time.strftime('%Y%m%d_%H%M%S')}")
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    profile_path = Path(args.profile_artifact)
    if not profile_path.is_absolute():
        profile_path = repo_root / profile_path
    args.profile_artifact = str(profile_path)
    if args.profile_artifact and not Path(args.profile_artifact).exists():
        raise FileNotFoundError(f"profile artifact not found: {args.profile_artifact}")

    prompt_file = outdir / "prompt.txt"
    prompt_file.write_text(PROMPT_TEXT, encoding="utf-8")

    policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    output_lens = [int(x.strip()) for x in args.output_lens.split(",") if x.strip()]
    ratios = [_ratio_key(float(x.strip())) for x in args.cache_ratios.split(",") if x.strip()]
    for ratio in ratios:
        if ratio not in MAX_DRAFT_BY_RATIO:
            raise ValueError(f"unsupported cache ratio for max_draft_tokens mapping: {ratio}")

    metadata = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "model_path": args.model_path,
        "profile_artifact": args.profile_artifact,
        "prompt_file": str(prompt_file),
        "settings": {
            "draft_reroute_policy": "entropy_cache_bias",
            "draft_top_c": 0,
            "draft_cuda_graph_enabled": True,
            "prefetch_enabled": True,
            "prefetch_runtime_mode": "draft_segment_indexed",
            "cache_strategy": "lru",
            "max_draft_tokens_by_ratio": MAX_DRAFT_BY_RATIO,
            "acceptance_strategy": args.acceptance_strategy,
            "temperature": args.temperature,
        },
        "argv": sys.argv,
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rows: list[dict[str, Any]] = []
    case_index = 0
    for policy in policies:
        if policy not in {"cpu", "cache_fill"}:
            raise ValueError(f"unsupported policy: {policy}")
        for out_len in output_lens:
            for ratio in ratios:
                row = run_case(
                    args=args,
                    repo_root=repo_root,
                    prompt_file=prompt_file,
                    policy=policy,
                    out_len=out_len,
                    ratio=ratio,
                    case_index=case_index,
                )
                rows.append(row)
                case_index += 1
                (outdir / "summary_incremental.json").write_text(
                    json.dumps({"metadata": metadata, "results": rows}, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                write_markdown(rows, outdir / "summary_incremental.md")

    result = {"metadata": metadata, "results": rows}
    (outdir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(rows, outdir / "summary.md")
    print(f"summary_json={outdir / 'summary.json'}")
    print(f"summary_md={outdir / 'summary.md'}")


if __name__ == "__main__":
    main()
