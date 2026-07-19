#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path


def _normalize_tpot_metrics(row: dict) -> dict:
    normalized = dict(row)
    generated = int(normalized.get("generated_output_tokens", 0) or 0)
    decode_sec = float(normalized.get("decode_sec", 0.0) or 0.0)
    intervals = max(generated - 1, 0)
    if decode_sec > 0.0 and intervals > 0:
        normalized["decode_token_intervals"] = intervals
        normalized["decode_tok_s"] = intervals / decode_sec
        normalized["tpot_ms"] = decode_sec * 1000.0 / intervals
        normalized["tpot_metric_source"] = "decode_sec_intervals"
    else:
        normalized["tpot_metric_source"] = "stored_fallback"
    return normalized


def _load_rows(path: str | Path) -> dict[tuple[object, ...], dict]:
    target = Path(path)
    paths = [target] if target.is_file() else sorted(target.rglob("summary.json"))
    if not paths:
        raise ValueError(f"no summary.json found under {target}")
    rows = {}
    for summary_path in paths:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        for row in data.get("rows", []):
            if row.get("status") != "ok":
                continue
            key = (
                str(row.get("dataset", "")),
                str(row.get("sample_id", "")),
                round(float(row.get("cache_ratio", 0.0)), 8),
                int(row.get("repeat", 0)),
                int(row.get("max_output_tokens", 0)),
            )
            if key in rows:
                raise ValueError(f"duplicate policy result key {key} in {summary_path}")
            rows[key] = _normalize_tpot_metrics(row)
    return rows


def _cluster_bootstrap_ci(
    values_by_cluster: dict[tuple[str, str, int], list[float]],
    *,
    seed: int,
    iterations: int,
) -> list[float] | None:
    clusters = sorted(values_by_cluster)
    if len(clusters) < 2:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        sampled_clusters = [rng.choice(clusters) for _ in clusters]
        values = [
            value
            for cluster in sampled_clusters
            for value in values_by_cluster[cluster]
        ]
        means.append(sum(values) / len(values))
    means.sort()
    return [
        means[int(0.025 * (len(means) - 1))],
        means[int(0.975 * (len(means) - 1))],
    ]


def _comparison(
    candidate: dict,
    baseline: dict,
    keys: list[tuple[object, ...]],
    *,
    seed: int,
    iterations: int,
) -> dict[str, object]:
    speedups = []
    tpot_reductions = []
    speedups_by_cluster: dict[tuple[str, str, int], list[float]] = {}
    reductions_by_cluster: dict[tuple[str, str, int], list[float]] = {}
    for key in keys:
        candidate_tps = float(candidate[key]["decode_tok_s"])
        baseline_tps = float(baseline[key]["decode_tok_s"])
        speedup = candidate_tps / baseline_tps - 1.0
        speedups.append(speedup)
        candidate_tpot = float(candidate[key]["tpot_ms"])
        baseline_tpot = float(baseline[key]["tpot_ms"])
        reduction = 1.0 - candidate_tpot / baseline_tpot
        tpot_reductions.append(reduction)
        cluster = (str(key[0]), str(key[1]), int(key[3]))
        speedups_by_cluster.setdefault(cluster, []).append(speedup)
        reductions_by_cluster.setdefault(cluster, []).append(reduction)
    return {
        "pair_count": len(keys),
        "cluster_count": len(speedups_by_cluster),
        "decode_tps_improvement_mean": sum(speedups) / len(speedups),
        "decode_tps_improvement_geomean": math.exp(
            sum(math.log1p(value) for value in speedups) / len(speedups)
        )
        - 1.0,
        "decode_tps_improvement_95ci": _cluster_bootstrap_ci(
            speedups_by_cluster, seed=seed, iterations=iterations
        ),
        "tpot_reduction_mean": sum(tpot_reductions) / len(tpot_reductions),
        "tpot_reduction_95ci": _cluster_bootstrap_ci(
            reductions_by_cluster, seed=seed + 1, iterations=iterations
        ),
    }


def _degeneration_reasons(row: dict) -> list[str]:
    reasons = []
    if int(row.get("generated_output_tokens", 0) or 0) <= 0:
        reasons.append("empty_output")
    if row.get("output_fixed_length_ok") is False:
        reasons.append("fixed_length_mismatch")
    if int(row.get("max_repeated_token_run", 0) or 0) > 32:
        reasons.append("repeated_token_run_gt_32")

    text = row.get("generated_text")
    if isinstance(text, str):
        if "\ufffd" in text:
            reasons.append("replacement_character")
        if any(ord(char) < 32 and char not in "\n\r\t" for char in text):
            reasons.append("illegal_control_character")
        nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(nonempty_lines) >= 2:
            most_common = Counter(nonempty_lines).most_common(1)[0][1]
            if most_common >= 2 and most_common / len(nonempty_lines) >= 0.5:
                reasons.append("repeated_nonempty_lines_ge_50pct")

    tokens = row.get("generated_token_ids")
    if isinstance(tokens, list) and len(tokens) >= 12:
        grams = Counter(
            tuple(int(value) for value in tokens[index:index + 12])
            for index in range(len(tokens) - 11)
        )
        if grams and grams.most_common(1)[0][1] >= 3:
            reasons.append("repeated_12gram_ge_3")
    return reasons


def _policy_quality(rows: dict[tuple[object, ...], dict]) -> dict[str, object]:
    failures = []
    reason_counts: Counter[str] = Counter()
    text_count = token_count = 0
    for key, row in rows.items():
        text_count += int(isinstance(row.get("generated_text"), str))
        token_count += int(isinstance(row.get("generated_token_ids"), list))
        reasons = _degeneration_reasons(row)
        if reasons:
            failures.append({"key": list(key), "reasons": reasons})
            reason_counts.update(reasons)
    return {
        "row_count": len(rows),
        "text_check_coverage": text_count / len(rows) if rows else 0.0,
        "token_check_coverage": token_count / len(rows) if rows else 0.0,
        "degeneration_failure_count": len(failures),
        "reason_counts": dict(sorted(reason_counts.items())),
        "failures": failures,
    }


def _absolute_metrics(rows: dict[tuple[object, ...], dict]) -> dict[str, object]:
    throughputs = [float(row["decode_tok_s"]) for row in rows.values()]
    tpots = sorted(float(row["tpot_ms"]) for row in rows.values())
    return {
        "row_count": len(rows),
        "decode_tps_geomean": math.exp(
            sum(math.log(value) for value in throughputs) / len(throughputs)
        ),
        "decode_tps_mean": sum(throughputs) / len(throughputs),
        "tpot_ms_mean": sum(tpots) / len(tpots),
        "tpot_ms_p90": tpots[min(len(tpots) - 1, int(0.9 * len(tpots)))],
    }


def _digest_mismatches(
    candidate: dict,
    baseline: dict,
    keys: list[tuple[object, ...]],
) -> list[tuple[object, ...]]:
    return [
        key
        for key in keys
        if not candidate[key].get("outputs_digest")
        or candidate[key].get("outputs_digest")
        != baseline[key].get("outputs_digest")
    ]


def _comparison_passed(
    comparison: dict[str, object],
    quality: dict[str, object],
    *,
    minimum_improvement: float,
    minimum_pairs: int,
    minimum_clusters: int,
) -> bool:
    confidence_interval = comparison["decode_tps_improvement_95ci"]
    return bool(
        int(comparison["pair_count"]) >= minimum_pairs
        and int(comparison["cluster_count"]) >= minimum_clusters
        and confidence_interval is not None
        and float(confidence_interval[0]) > minimum_improvement
        and int(quality["degeneration_failure_count"]) == 0
    )


def _generic_run(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.policies_root)
    policy_names = sorted(path.name for path in root.iterdir() if path.is_dir())
    if args.candidate_policies:
        requested = {
            item.strip() for item in args.candidate_policies.split(",") if item.strip()
        }
        policy_names = [name for name in policy_names if name in requested]
    baseline_name = str(args.baseline_policy)
    if baseline_name not in {path.name for path in root.iterdir() if path.is_dir()}:
        raise SystemExit(f"baseline policy not found: {baseline_name}")
    candidates = [name for name in policy_names if name != baseline_name]
    if not candidates:
        raise SystemExit("no candidate policies found")
    rows_by_policy = {
        name: _load_rows(root / name) for name in [baseline_name, *candidates]
    }
    quality = {
        name: _policy_quality(rows) for name, rows in rows_by_policy.items()
    }
    absolute = {
        name: _absolute_metrics(rows) for name, rows in rows_by_policy.items()
    }
    comparisons = {}
    for index, name in enumerate(candidates):
        keys = sorted(set(rows_by_policy[name]) & set(rows_by_policy[baseline_name]))
        if not keys:
            raise SystemExit(f"no paired rows for {name} vs {baseline_name}")
        comparison = _comparison(
            rows_by_policy[name],
            rows_by_policy[baseline_name],
            keys,
            seed=int(args.seed) + index * 2,
            iterations=int(args.bootstrap_iterations),
        )
        mismatches = _digest_mismatches(
            rows_by_policy[name], rows_by_policy[baseline_name], keys
        )
        comparison["output_digest_mismatch_count"] = len(mismatches)
        comparison["output_digest_match_rate"] = 1.0 - len(mismatches) / len(keys)
        comparison["passed"] = _comparison_passed(
            comparison,
            quality[name],
            minimum_improvement=float(args.minimum_improvement),
            minimum_pairs=int(args.minimum_pairs),
            minimum_clusters=int(args.minimum_clusters),
        )
        comparisons[name] = comparison
    best_candidate = max(
        candidates,
        key=lambda name: float(
            comparisons[name]["decode_tps_improvement_geomean"]
        ),
    )
    output = {
        "passed": all(bool(row["passed"]) for row in comparisons.values()),
        "tpot_definition": "decode_sec / (generated_output_tokens - 1)",
        "baseline_policy": baseline_name,
        "candidate_selection_metric": "decode_tps_improvement_geomean",
        "best_candidate_policy": best_candidate,
        "minimum_improvement": float(args.minimum_improvement),
        "minimum_pairs": int(args.minimum_pairs),
        "minimum_clusters": int(args.minimum_clusters),
        "policies": absolute,
        "quality": quality,
        "comparisons": comparisons,
    }
    return output


def _legacy_run(args: argparse.Namespace) -> dict[str, object]:
    active = _load_rows(args.active)
    static = _load_rows(args.static)
    none = _load_rows(args.none)
    keys = sorted(set(active) & set(static) & set(none))
    if not keys:
        raise SystemExit("no request rows pair across active/static/none")
    quality = _policy_quality(active)
    static_comparison = _comparison(
        active,
        static,
        keys,
        seed=int(args.seed),
        iterations=int(args.bootstrap_iterations),
    )
    none_comparison = _comparison(
        active,
        none,
        keys,
        seed=int(args.seed) + 2,
        iterations=int(args.bootstrap_iterations),
    )
    mismatches = [
        key
        for key in keys
        if len(
            {
                active[key].get("outputs_digest"),
                static[key].get("outputs_digest"),
                none[key].get("outputs_digest"),
            }
        )
        != 1
    ]
    passed = all(
        _comparison_passed(
            comparison,
            quality,
            minimum_improvement=float(args.minimum_improvement),
            minimum_pairs=int(args.minimum_pairs),
            minimum_clusters=int(args.minimum_clusters),
        )
        for comparison in (static_comparison, none_comparison)
    )
    return {
        "passed": passed,
        "tpot_definition": "decode_sec / (generated_output_tokens - 1)",
        "minimum_improvement": float(args.minimum_improvement),
        "minimum_pairs": int(args.minimum_pairs),
        "minimum_clusters": int(args.minimum_clusters),
        "paired_request_count": len(keys),
        "output_digest_mismatch_count": len(mismatches),
        "quality": quality,
        "active_vs_static": static_comparison,
        "active_vs_none": none_comparison,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    output = (
        _generic_run(args)
        if str(getattr(args, "policies_root", "") or "")
        else _legacy_run(args)
    )
    if getattr(args, "output", ""):
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    if bool(getattr(args, "require_pass", False)) and not output["passed"]:
        raise SystemExit(1)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active", default="")
    parser.add_argument("--static", default="")
    parser.add_argument("--none", default="")
    parser.add_argument("--policies-root", default="")
    parser.add_argument("--baseline-policy", default="")
    parser.add_argument("--candidate-policies", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--minimum-improvement", type=float, default=0.0)
    parser.add_argument("--minimum-pairs", type=int, default=20)
    parser.add_argument("--minimum-clusters", type=int, default=6)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    if not args.policies_root and not (args.active and args.static and args.none):
        parser.error("provide --policies-root or --active/--static/--none")
    if args.policies_root and not args.baseline_policy:
        parser.error("--policies-root requires --baseline-policy")
    return args


if __name__ == "__main__":
    run(parse_args())
