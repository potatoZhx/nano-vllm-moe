#!/usr/bin/env python3
"""Precision validation: compare spec+greedy output vs standard output.

Since greedy acceptance always defers to the full model's verify argmax,
spec+greedy tokens MUST match standard tokens exactly. Any mismatch
indicates a numerical precision issue in the heterogeneous execution path.
"""

from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from pathlib import Path

import torch


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


def run_case(args_dict: dict, repo_root: str, env: dict, timeout: int) -> dict | None:
    """Run a single case via subprocess and return parsed result."""
    runner = str(Path(repo_root) / "scripts" / "run_case_inline.py")
    cmd = [sys.executable, runner, json.dumps(args_dict)]
    try:
        proc = subprocess.run(
            cmd, cwd=repo_root, env=env,
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}

    if proc.returncode != 0:
        return {
            "error": f"exit={proc.returncode}",
            "stderr_tail": proc.stderr[-2000:] if proc.stderr else "",
        }

    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    return {"error": "no RESULT_JSON", "stdout_tail": proc.stdout[-500:]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--calibration-artifact", default="")
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dist-port-base", type=int, default=26600)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--cpu-expert-pin-memory", action="store_true")
    parser.add_argument("--output-dir", default="results/precision_validation")
    parser.add_argument("--only-policy", default="")
    parser.add_argument("--only-ratio", type=float, default=0.0)
    parser.add_argument("--case-timeout-sec", type=int, default=1800)
    parser.add_argument("--no-cpu-exec", action="store_true",
                        help="Disable CPU expert execution (use GPU fallback for verify precision)")
    args = parser.parse_args()

    policies = [args.only_policy] if args.only_policy else POLICIES
    cache_ratios = [args.only_ratio] if args.only_ratio > 0 else CACHE_RATIOS

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    repo_root = str(Path(__file__).resolve().parents[1])
    env = os.environ.copy()
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

    # --- Step 1: Standard mode reference ---
    ref_path = outdir / "standard_reference.json"
    print("=" * 70)
    print("STEP 1: Standard mode reference")
    print("=" * 70)

    std_args = {
        "repo_root": repo_root, "model_path": args.model_path,
        "dist_port": args.dist_port_base, "enforce_eager": args.enforce_eager,
        "max_model_len": args.max_model_len, "output_len": args.output_len,
        "prompt_text": PROMPT_TEXT,
        "calibration_artifact": args.calibration_artifact,
        "cpu_expert_pin_memory": args.cpu_expert_pin_memory,
        "cpu_expert_execution_enabled": not args.no_cpu_exec,
        "standard_mode": True,
    }
    print("  Running standard mode in subprocess...", flush=True)
    t0 = time.time()
    ref_result = run_case(std_args, repo_root, env, args.case_timeout_sec)
    dt = time.time() - t0

    if ref_result is None or "error" in ref_result:
        print(f"  FAILED: {ref_result}")
        sys.exit(1)
    ref_ids = ref_result["token_ids"]
    ref_text = ref_result["text"]
    ref_path.write_text(json.dumps(ref_result, ensure_ascii=False, indent=2))
    (outdir / "standard_reference.txt").write_text(ref_text, encoding="utf-8")
    print(f"  Standard mode: {len(ref_ids)} tokens in {dt:.1f}s")
    print(f"  Reference text (first 200 chars): {ref_text[:200]}")
    print()

    # --- Step 2: Spec mode for each policy × ratio ---
    total = len(policies) * len(cache_ratios)
    all_results = []
    mismatch_found = False
    dist_port = args.dist_port_base + 20

    for idx, policy in enumerate(policies):
        for ratio in cache_ratios:
            case_num = len(all_results) + 1
            print(f"[{case_num}/{total}] spec: {policy} r={ratio:.2f}", flush=True)

            spec_args = {
                "repo_root": repo_root, "model_path": args.model_path,
                "dist_port": dist_port, "enforce_eager": args.enforce_eager,
                "max_model_len": args.max_model_len, "output_len": args.output_len,
                "prompt_text": PROMPT_TEXT,
                "policy": policy, "cache_ratio": ratio,
                "calibration_artifact": args.calibration_artifact,
                "cpu_expert_pin_memory": args.cpu_expert_pin_memory,
                "cpu_expert_execution_enabled": not args.no_cpu_exec,
                "standard_mode": False,
            }
            t0 = time.time()
            case_result = run_case(spec_args, repo_root, env, args.case_timeout_sec)
            case_dt = time.time() - t0

            if case_result is None or "error" in case_result:
                err = case_result.get("error", "unknown") if case_result else "null result"
                print(f"  [FAIL] {err}")
                all_results.append({
                    "policy": policy, "ratio": ratio,
                    "error": err,
                    "stderr": case_result.get("stderr_tail", "") if case_result else "",
                })
                dist_port += 1
                continue

            test_ids = case_result["token_ids"]
            match = (ref_ids == test_ids)

            if not match:
                mismatch_found = True
                for mpos in range(min(len(ref_ids), len(test_ids))):
                    if ref_ids[mpos] != test_ids[mpos]:
                        break
                else:
                    mpos = min(len(ref_ids), len(test_ids))
                print(f"  [MISMATCH@{mpos}] ref:{ref_ids[max(0,mpos-2):mpos+3]} "
                      f"test:{test_ids[max(0,mpos-2):mpos+3]}", flush=True)
            else:
                acc = case_result.get("acceptance_rate", 0)
                draft = case_result.get("draft_forward_ms_avg", 0)
                print(f"  [MATCH] acc={acc:.4f} draft={draft:.3f}ms "
                      f"elapsed={case_dt:.1f}s", flush=True)

            all_results.append({
                "policy": policy, "cache_ratio": ratio,
                "effective_cache_ratio": case_result.get("effective_cache_ratio", ratio),
                "slots": case_result.get("slots", 0),
                "match": match,
                "first_mismatch_pos": mpos if not match else -1,
                "ref_len": len(ref_ids), "test_len": len(test_ids),
                "acceptance_rate": case_result.get("acceptance_rate", 0),
                "draft_forward_ms_avg": case_result.get("draft_forward_ms_avg", 0),
                "draft_replays": case_result.get("draft_replays", 0),
                "cpu_route_ratio": case_result.get("cpu_route_ratio", 0),
                "elapsed_sec": case_dt,
            })
            dist_port += 1

    # --- Summary ---
    print("\n" + "=" * 75)
    print("PRECISION VALIDATION SUMMARY")
    print("=" * 75)
    header = f"{'Policy':<28} {'Ratio':>6} {'Match':>6} {'AccRate':>8} {'Draft_ms':>9} {'Mism@':>6}"
    print(header)
    print("-" * 70)
    for r in all_results:
        if "error" in r:
            print(f"{r['policy']:<28} {r['ratio']:>6.2f} ERROR: {r['error'][:40]}")
            continue
        m = "YES" if r["match"] else f"NO@{r['first_mismatch_pos']}"
        print(f"{r['policy']:<28} {r['cache_ratio']:>6.2f} {m:>6} "
              f"{r['acceptance_rate']:>8.4f} {r['draft_forward_ms_avg']:>9.3f} "
              f"{r['first_mismatch_pos'] if not r['match'] else '':>6}")

    summary_path = outdir / "precision_summary.json"
    summary_path.write_text(json.dumps({
        "reference": {"token_ids": ref_ids, "text": ref_text, "prompt": PROMPT_TEXT},
        "results": all_results,
        "all_match": not mismatch_found,
    }, ensure_ascii=False, indent=2) + "\n")
    print(f"\nSummary: {summary_path}")
    if mismatch_found:
        print("*** MISMATCHES FOUND - precision debugging needed ***")
    else:
        print("*** ALL MATCH ***")


if __name__ == "__main__":
    main()
