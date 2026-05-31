#!/usr/bin/env python3
"""Full reroute validation: all policies, deterministic + sampling acceptance,
3 cache ratios, 2 output lengths, text output recorded.

Each case runs as a subprocess (each LLM() calls init_process_group exactly once).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

POLICIES = [
    "round_robin",
    "drop_miss",
    "entropy_cache_bias",
    "bounded_cache_bias",
    "similarity_replace",
]

CACHE_RATIOS = [0.25, 0.50, 0.75]
OUTPUT_LENS = [128, 512]
ACCEPTANCE_STRATEGIES = ["greedy", "standard_sampling"]


def _case_name(policy: str, ratio: float, out_len: int, acc: str) -> str:
    ratio_pct = int(round(ratio * 100))
    return f"{policy}_ratio{ratio_pct}_l{out_len}_{acc}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full reroute validation across policies, ratios, output lengths, and acceptance strategies."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--calibration-artifact", default="")
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--max-draft-tokens", type=int, default=8)
    parser.add_argument("--draft-top-c", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--dist-port-base", type=int, default=26500)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--cpu-expert-pin-memory", action="store_true")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--policies", nargs="+", default=None)
    parser.add_argument("--cache-ratios", nargs="+", type=float, default=None)
    parser.add_argument("--output-lens", nargs="+", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--only-policy", default="")
    parser.add_argument("--only-ratio", type=float, default=0.0)
    parser.add_argument("--only-out-len", type=int, default=0)
    parser.add_argument("--only-acc", default="")
    parser.add_argument("--case-timeout-sec", type=int, default=1200)
    args = parser.parse_args()

    policies = args.policies or POLICIES
    cache_ratios = args.cache_ratios or CACHE_RATIOS
    output_lens = args.output_lens or OUTPUT_LENS
    acceptance_strategies = args.only_acc.split(",") if args.only_acc else ACCEPTANCE_STRATEGIES

    if args.only_policy:
        policies = [args.only_policy]
    if args.only_ratio > 0:
        cache_ratios = [args.only_ratio]
    if args.only_out_len > 0:
        output_lens = [args.only_out_len]

    outdir = Path(args.output_dir) if args.output_dir else Path(
        f"results/reroute_full_validation_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    total = len(policies) * len(cache_ratios) * len(output_lens) * len(acceptance_strategies)
    print(f"Running {total} cases:")
    print(f"  Policies: {policies}")
    print(f"  Ratios: {cache_ratios}")
    print(f"  Output lens: {output_lens}")
    print(f"  Acceptance strategies: {acceptance_strategies}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Output dir: {outdir}")

    script_path = Path(__file__).resolve().parent.parent / "benchmarks" / "scripts" / "spec_verify_expert_count_stats.py"
    env = os.environ.copy()
    repo_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

    results: list[dict[str, Any]] = []
    case_idx = 0
    dist_port = args.dist_port_base

    for policy in policies:
        for ratio in cache_ratios:
            for out_len in output_lens:
                for acc_strat in acceptance_strategies:
                    case_idx += 1
                    name = _case_name(policy, ratio, out_len, acc_strat)
                    case_json = outdir / f"{name}.json"
                    case_log = outdir / f"{name}.log"

                    # Determine temperature: 0 for greedy, configurable for sampling
                    temp = 0.0 if acc_strat == "greedy" else args.temperature
                    artifact = args.calibration_artifact

                    cmd = [
                        sys.executable,
                        str(script_path),
                        "--single-case",
                        "--model-path", args.model_path,
                        "--cache-ratio", str(ratio),
                        "--slots-per-layer", "0",
                        "--prefetch-enabled", "false",
                        "--output", str(case_json),
                        "--dist-port", str(dist_port),
                        "--num-seqs", "1",
                        "--input-len", str(args.input_len),
                        "--output-len", str(out_len),
                        "--max-draft-tokens", str(args.max_draft_tokens),
                        "--draft-top-c", str(args.draft_top_c),
                        "--draft-reroute-policy", policy,
                        "--draft-reroute-artifact", artifact,
                        "--temperature", str(temp),
                        "--acceptance-strategy", acc_strat,
                        "--acceptance-threshold", "0.7",
                        "--cpu-expert-backend", "fused",
                        "--cpu-expert-pin-memory", str(args.cpu_expert_pin_memory).lower(),
                        "--cpu-expert-workspace-max-routes", "16384",
                        "--cpu-expert-packed-min-routes", "1",
                        "--cpu-expert-parallel-mode", "serial",
                        "--cpu-expert-num-threads", "4",
                        "--max-num-batched-tokens", "8192",
                        "--max-num-seqs", "1",
                        "--max-model-len", str(args.max_model_len),
                        "--gpu-memory-utilization", "0.85",
                        "--enforce-eager", str(args.enforce_eager).lower(),
                        "--seed", str(args.seed),
                        "--sync-layer-timing", "true",
                    ]

                    print(f"\n[{case_idx}/{total}] [{time.strftime('%H:%M:%S')}] {name}", flush=True)
                    t0 = time.time()

                    try:
                        with case_log.open("w", encoding="utf-8") as log_f:
                            proc = subprocess.run(
                                cmd,
                                cwd=repo_root,
                                env=env,
                                stdout=log_f,
                                stderr=subprocess.STDOUT,
                                text=True,
                                timeout=args.case_timeout_sec,
                            )
                        dt = time.time() - t0

                        if proc.returncode != 0:
                            tail = ""
                            try:
                                log_content = case_log.read_text(encoding="utf-8", errors="replace")
                                tail = log_content[-2000:]
                            except Exception:
                                pass
                            print(f"  [FAIL] exit={proc.returncode} elapsed={dt:.1f}s", flush=True)
                            results.append({
                                "name": name,
                                "case": {"policy": policy, "cache_ratio": ratio,
                                         "output_len": out_len, "acceptance_strategy": acc_strat,
                                         "temperature": temp},
                                "error": f"exit={proc.returncode}",
                                "log_tail": tail,
                            })
                        else:
                            try:
                                raw = json.loads(case_json.read_text(encoding="utf-8"))
                            except Exception:
                                print(f"  [FAIL] JSON parse error elapsed={dt:.1f}s", flush=True)
                                results.append({
                                    "name": name,
                                    "case": {"policy": policy, "cache_ratio": ratio,
                                             "output_len": out_len, "acceptance_strategy": acc_strat},
                                    "error": "JSON parse error",
                                })
                                dist_port += 1
                                continue

                            summary = raw.get("summary", {})
                            acceptance = summary.get("acceptance", {})
                            acc_rate = acceptance.get("acceptance_rate", 0.0)
                            draft_avg = summary.get("draft_forward_ms_avg", 0.0)
                            replays = summary.get("cuda_graph", {}).get("draft_replay_count", 0)
                            token_ids = raw.get("generated_token_ids", [])
                            digest = raw.get("outputs_digest", "")

                            result_entry = {
                                "name": name,
                                "case": raw.get("case", {}),
                                "elapsed_sec": raw.get("elapsed_sec", 0.0),
                                "generated_output_tokens": raw.get("generated_output_tokens", 0),
                                "outputs_digest": digest,
                                "acceptance_rate": acc_rate,
                                "draft_position_acceptance": acceptance.get("draft_position_acceptance", []),
                                "draft_forward_ms_avg": draft_avg,
                                "draft_replay_count": replays,
                                "generated_token_ids_len": len(token_ids) if token_ids else 0,
                                "generated_text_sample": "",
                            }

                            # Try to decode generated text
                            try:
                                from transformers import AutoTokenizer
                                tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
                                if token_ids:
                                    result_entry["generated_text_sample"] = tok.decode(
                                        token_ids[:min(200, len(token_ids))]
                                    )
                            except Exception:
                                pass

                            results.append(result_entry)
                            print(f"  [OK] accept={acc_rate:.4f} draft_avg={draft_avg:.3f}ms "
                                  f"replays={replays} elapsed={dt:.1f}s digest={digest[:12]}", flush=True)

                        # Write incremental results after each case
                        incremental = {
                            "metadata": {
                                "timestamp": time.strftime("%Y%m%d_%H%M%S"),
                                "model_path": args.model_path,
                                "calibration_artifact": args.calibration_artifact,
                                "input_len": args.input_len,
                                "temperature": args.temperature,
                            },
                            "results": results,
                        }
                        (outdir / "results_incremental.json").write_text(
                            json.dumps(incremental, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8",
                        )

                    except subprocess.TimeoutExpired:
                        print(f"  [TIMEOUT] {args.case_timeout_sec}s", flush=True)
                        results.append({
                            "name": name,
                            "case": {"policy": policy, "cache_ratio": ratio,
                                     "output_len": out_len, "acceptance_strategy": acc_strat},
                            "error": "timeout",
                        })

                    dist_port += 1

    # Final summary table
    print("\n\n" + "=" * 95)
    print("SUMMARY TABLE")
    print("=" * 95)
    header = (
        f"{'Policy':<28} {'Ratio':>6} {'Out':>5} {'Accept':<18} "
        f"{'AccRate':>8} {'DraftAvg':>9} {'Replays':>8} {'Elapsed':>8} {'Tokens':>7} {'Digest':<12}"
    )
    print(header)
    print("-" * 95)
    for r in results:
        if "error" in r:
            err = str(r.get("error", "unknown"))[:50]
            c = r.get("case", {})
            print(f"{c.get('policy',''):<28} {c.get('cache_ratio',0):>6.2f} "
                  f"{c.get('output_len',0):>5} {c.get('acceptance_strategy',''):<18} "
                  f"ERROR: {err}")
            continue
        print(
            f"{r['name'].rsplit('_ratio',1)[0]:<28} "
            f"{r['case'].get('cache_ratio', 0):>6.3f} "
            f"{r['case'].get('output_len', 0):>5} "
            f"{r['case'].get('acceptance_strategy', ''):<18} "
            f"{r.get('acceptance_rate', 0):>8.4f} "
            f"{r.get('draft_forward_ms_avg', 0):>9.3f} "
            f"{r.get('draft_replay_count', 0):>8} "
            f"{r.get('elapsed_sec', 0):>8.1f} "
            f"{r.get('generated_output_tokens', 0):>7} "
            f"{r.get('outputs_digest', '')[:12]}"
        )

    # Write final summary
    final_path = outdir / "results.json"
    final_path.write_text(
        json.dumps({
            "metadata": {
                "timestamp": time.strftime("%Y%m%d_%H%M%S"),
                "model_path": args.model_path,
                "calibration_artifact": args.calibration_artifact,
                "input_len": args.input_len,
                "temperature": args.temperature,
                "seed": args.seed,
            },
            "results": results,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Create text summary file from per-case logs
    text_lines = ["# Reroute Full Validation - Generated Texts\n"]
    for r in results:
        if "error" in r:
            continue
        name = r["name"]
        case_log = outdir / f"{name}.log"
        text_lines.append(f"\n## {name}")
        text_lines.append(f"Acceptance rate: {r.get('acceptance_rate', 0):.4f}")
        text_lines.append(f"Draft forward avg: {r.get('draft_forward_ms_avg', 0):.3f} ms")
        text_lines.append(f"Text sample: {r.get('generated_text_sample', 'N/A')[:500]}")
        text_lines.append(f"\nFull log: {case_log}")

    (outdir / "all_generated_texts.md").write_text("\n".join(text_lines), encoding="utf-8")

    print(f"\nAll results: {outdir}")
    print(f"Summary: {final_path}")


if __name__ == "__main__":
    main()
