#!/usr/bin/env python3
"""Reroute validation with meaningful natural-language prompt.
Same 60-case matrix, but using coherent prose instead of synthetic filler.
Each case runs as a subprocess via spec_verify_expert_count_stats.py.
"""

from __future__ import annotations

import argparse, json, os, subprocess, sys, time, hashlib
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

POLICIES = ["round_robin", "drop_miss", "entropy_cache_bias", "bounded_cache_bias", "similarity_replace"]
CACHE_RATIOS = [0.25, 0.50, 0.75]
OUTPUT_LENS = [128, 512]
ACCEPTANCE_STRATEGIES = ["greedy", "standard_sampling"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--calibration-artifact", default="")
    parser.add_argument("--input-len", type=int, default=0,
                        help="If 0, use full prompt text as-is")
    parser.add_argument("--max-draft-tokens", type=int, default=8)
    parser.add_argument("--draft-top-c", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--dist-port-base", type=int, default=26500)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--cpu-expert-pin-memory", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--only-policy", default="")
    parser.add_argument("--only-ratio", type=float, default=0.0)
    parser.add_argument("--only-out-len", type=int, default=0)
    parser.add_argument("--only-acc", default="")
    parser.add_argument("--case-timeout-sec", type=int, default=1800)
    args = parser.parse_args()

    policies = [args.only_policy] if args.only_policy else POLICIES
    cache_ratios = [args.only_ratio] if args.only_ratio > 0 else CACHE_RATIOS
    output_lens = [args.only_out_len] if args.only_out_len > 0 else OUTPUT_LENS
    acc_strats = args.only_acc.split(",") if args.only_acc else ACCEPTANCE_STRATEGIES

    outdir = Path(args.output_dir) if args.output_dir else Path(
        f"results/reroute_meaningful_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    total = len(policies) * len(cache_ratios) * len(output_lens) * len(acc_strats)
    print(f"Running {total} cases with meaningful prompt (~200 tokens)")
    print(f"  Policies: {policies}")
    print(f"  Ratios: {cache_ratios}")
    print(f"  Output lens: {output_lens}")
    print(f"  Acceptance: {acc_strats}")
    print(f"  Output dir: {outdir}")

    # Save prompt to a temp file for passing to subprocess (avoids shell escaping issues)
    prompt_file = outdir / "prompt.txt"
    prompt_file.write_text(PROMPT_TEXT, encoding="utf-8")

    repo_root = str(Path(__file__).resolve().parents[1])
    script_path = Path(repo_root) / "benchmarks" / "scripts" / "spec_verify_expert_count_stats.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

    results = []
    case_idx = 0
    dist_port = args.dist_port_base

    for policy in policies:
        for ratio in cache_ratios:
            for out_len in output_lens:
                for acc_strat in acc_strats:
                    case_idx += 1
                    ratio_pct = int(round(ratio * 100))
                    name = f"{policy}_ratio{ratio_pct}_l{out_len}_{acc_strat}"
                    case_json = outdir / f"{name}.json"
                    case_log = outdir / f"{name}.log"

                    temp = 0.0 if acc_strat == "greedy" else args.temperature
                    artifact = args.calibration_artifact if policy == "similarity_replace" else ""

                    # Determine input_len: use tokenizer to compute actual token count
                    # We'll let the subprocess do that by passing the prompt file

                    cmd = [
                        sys.executable, str(script_path),
                        "--single-case",
                        "--model-path", args.model_path,
                        "--prompt-text-file", str(prompt_file),
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
                                cmd, cwd=repo_root, env=env,
                                stdout=log_f, stderr=subprocess.STDOUT,
                                text=True, timeout=args.case_timeout_sec,
                            )
                        dt = time.time() - t0

                        if proc.returncode != 0:
                            tail = ""
                            try:
                                tail = case_log.read_text(encoding="utf-8", errors="replace")[-2000:]
                            except Exception:
                                pass
                            print(f"  [FAIL] exit={proc.returncode} elapsed={dt:.1f}s", flush=True)
                            results.append({
                                "name": name, "error": f"exit={proc.returncode}",
                                "case": {"policy": policy, "cache_ratio": ratio,
                                         "output_len": out_len, "acceptance_strategy": acc_strat},
                                "log_tail": tail,
                            })
                        else:
                            raw = json.loads(case_json.read_text(encoding="utf-8"))
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
                                "draft_forward_ms_avg": draft_avg,
                                "draft_replay_count": replays,
                                "generated_token_ids_len": len(token_ids) if token_ids else 0,
                            }

                            # Decode generated text
                            try:
                                from transformers import AutoTokenizer
                                tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
                                if token_ids:
                                    result_entry["generated_text"] = tok.decode(token_ids)
                                    result_entry["generated_text_sample"] = tok.decode(
                                        token_ids[:min(300, len(token_ids))]
                                    )
                            except Exception:
                                result_entry["generated_text"] = ""
                                result_entry["generated_text_sample"] = ""

                            results.append(result_entry)
                            print(f"  [OK] accept={acc_rate:.4f} draft_avg={draft_avg:.3f}ms "
                                  f"replays={replays} elapsed={dt:.1f}s digest={digest[:12]}", flush=True)

                        # Incremental save
                        (outdir / "results_incremental.json").write_text(
                            json.dumps({"metadata": {"timestamp": time.strftime("%Y%m%d_%H%M%S"),
                                                      "prompt": PROMPT_TEXT[:200]},
                                        "results": results},
                                       ensure_ascii=False, indent=2) + "\n")

                    except subprocess.TimeoutExpired:
                        print(f"  [TIMEOUT] {args.case_timeout_sec}s", flush=True)
                        results.append({"name": name, "error": "timeout",
                                        "case": {"policy": policy, "cache_ratio": ratio,
                                                 "output_len": out_len, "acceptance_strategy": acc_strat}})

                    dist_port += 1

    # Final summary
    final_path = outdir / "results.json"
    json.dump({"metadata": {"timestamp": time.strftime("%Y%m%d_%H%M%S"),
                             "model_path": args.model_path,
                             "input_text": PROMPT_TEXT},
               "results": results}, final_path, ensure_ascii=False, indent=2)

    print("\n\n" + "=" * 110)
    print("SUMMARY TABLE - Meaningful Prompt")
    print("=" * 110)
    header = (f"{'Policy':<28} {'Ratio':>6} {'Out':>5} {'Accept':<18} "
              f"{'AccRate':>8} {'DraftAvg':>9} {'Replays':>8} {'Elapsed':>8} {'Tokens':>7} {'Digest':<12}")
    print(header)
    print("-" * 110)
    for r in results:
        if "error" in r:
            c = r.get("case", {})
            print(f"{c.get('policy',''):<28} {c.get('cache_ratio',0):>6.2f} "
                  f"{c.get('output_len',0):>5} {c.get('acceptance_strategy',''):<18} "
                  f"ERROR: {r.get('error','')[:40]}")
            continue
        c = r["case"]
        pol = c.get("draft_reroute_policy", c.get("policy", "?"))
        dig = r.get("outputs_digest", "")[:12]
        print(f"{pol:<28} {c['cache_ratio']:>6.3f} {c['output_len']:>5} "
              f"{c['acceptance_strategy']:<18} {r['acceptance_rate']:>8.4f} "
              f"{r['draft_forward_ms_avg']:>9.3f} {r['draft_replay_count']:>8} "
              f"{r['elapsed_sec']:>8.1f} {r['generated_output_tokens']:>7} {dig}")

    print(f"\nAll results: {outdir}")
    print(f"Summary: {final_path}")


if __name__ == "__main__":
    main()
