#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _summary_paths(inputs: list[str | Path]) -> list[Path]:
    paths = []
    for item in inputs:
        path = Path(item)
        if path.is_file():
            paths.append(path)
        else:
            paths.extend(sorted(path.rglob("summary.json")))
    unique = sorted({path.resolve() for path in paths})
    if not unique:
        raise ValueError("no summary.json files found")
    return unique


def _row_key(row: dict) -> tuple[object, ...]:
    return (
        str(row.get("dataset", "")),
        str(row.get("sample_id", "")),
        round(float(row.get("cache_ratio", 0.0)), 8),
        int(row.get("repeat", 0)),
        int(row.get("max_output_tokens", 0)),
    )


def _load_rows(
    inputs: list[str | Path],
) -> tuple[dict[int, dict[tuple[object, ...], dict]], list[dict[str, object]]]:
    rows_by_k: dict[int, dict[tuple[object, ...], dict]] = defaultdict(dict)
    sources = []
    for path in _summary_paths(inputs):
        data = json.loads(path.read_text(encoding="utf-8"))
        metadata = data.get("metadata", {})
        sources.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "sample_offset": metadata.get("sample_offset"),
                "seed": next(
                    (
                        value
                        for value in (
                            row.get("runtime_seed") for row in data.get("rows", [])
                        )
                        if value is not None
                    ),
                    None,
                ),
                "profile_enabled": bool(
                    metadata.get("collect_profile")
                    or metadata.get("engine_profile")
                    or metadata.get("verify_cost_model_profile")
                ),
                "verify_model_mode": metadata.get("draft_tpot_verify_model_mode"),
                "stop_policy": metadata.get("draft_stop_policy"),
            }
        )
        for row in data.get("rows", []):
            if row.get("status") != "ok":
                continue
            row = _normalize_tpot_metrics(row)
            throughput = float(row.get("decode_tok_s", 0.0))
            tpot_ms = float(row.get("tpot_ms", 0.0))
            if not math.isfinite(throughput) or throughput <= 0.0:
                raise ValueError(f"invalid decode_tok_s in {path}")
            if not math.isfinite(tpot_ms) or tpot_ms <= 0.0:
                raise ValueError(f"invalid tpot_ms in {path}")
            draft_k = int(row["max_draft_tokens"])
            key = _row_key(row)
            if key in rows_by_k[draft_k]:
                raise ValueError(f"duplicate K={draft_k} result key {key}")
            rows_by_k[draft_k][key] = row
    if len(rows_by_k) < 2:
        raise ValueError("fixed-K selection requires at least two K values")
    return dict(rows_by_k), sources


def _score(rows: list[dict]) -> dict[str, float | int]:
    throughputs = [float(row["decode_tok_s"]) for row in rows]
    tpots = [float(row["tpot_ms"]) for row in rows]
    mean_log = sum(math.log(value) for value in throughputs) / len(throughputs)
    return {
        "row_count": len(rows),
        "mean_log_decode_tps": mean_log,
        "decode_tps_geomean": math.exp(mean_log),
        "decode_tps_mean": sum(throughputs) / len(throughputs),
        "tpot_ms_mean": sum(tpots) / len(tpots),
    }


def analyze(
    inputs: list[str | Path],
    *,
    tie_fraction: float = 0.005,
    boundary_k: int = 15,
    boundary_fraction: float = 0.01,
    require_complete: bool = True,
) -> dict[str, object]:
    if not 0.0 <= tie_fraction < 1.0:
        raise ValueError("tie_fraction must be in [0, 1)")
    if not 0.0 <= boundary_fraction < 1.0:
        raise ValueError("boundary_fraction must be in [0, 1)")
    rows_by_k, sources = _load_rows(inputs)
    ks = sorted(rows_by_k)
    key_sets = {draft_k: set(rows) for draft_k, rows in rows_by_k.items()}
    shared_keys = set.intersection(*(key_sets[draft_k] for draft_k in ks))
    union_keys = set.union(*(key_sets[draft_k] for draft_k in ks))
    missing = {
        str(draft_k): [list(key) for key in sorted(union_keys - key_sets[draft_k])]
        for draft_k in ks
        if union_keys - key_sets[draft_k]
    }
    if not shared_keys:
        raise ValueError("no request rows are paired across all K values")
    if require_complete and missing:
        raise ValueError(
            "incomplete fixed-K pairing: "
            + ", ".join(f"K={key} missing {len(value)}" for key, value in missing.items())
        )

    scores = {}
    for draft_k in ks:
        scores[draft_k] = _score(
            [rows_by_k[draft_k][key] for key in sorted(shared_keys)]
        )
    best_k = max(ks, key=lambda draft_k: scores[draft_k]["mean_log_decode_tps"])
    best_score = float(scores[best_k]["decode_tps_geomean"])
    eligible = [
        draft_k
        for draft_k in ks
        if float(scores[draft_k]["decode_tps_geomean"])
        >= best_score * (1.0 - tie_fraction)
    ]
    selected_k = min(eligible)
    for draft_k in ks:
        scores[draft_k]["relative_to_best"] = (
            float(scores[draft_k]["decode_tps_geomean"]) / best_score - 1.0
        )

    per_ratio = {}
    ratios = sorted({float(key[2]) for key in shared_keys})
    for ratio in ratios:
        ratio_keys = [key for key in sorted(shared_keys) if float(key[2]) == ratio]
        per_ratio[f"{ratio:.8f}"] = {
            str(draft_k): _score(
                [rows_by_k[draft_k][key] for key in ratio_keys]
            )
            for draft_k in ks
        }

    boundary_condition = False
    boundary_trend = None
    boundary_relative = None
    if boundary_k in scores:
        initial_ks = [draft_k for draft_k in ks if draft_k <= boundary_k]
        initial_best = max(
            float(scores[draft_k]["decode_tps_geomean"])
            for draft_k in initial_ks
        )
        boundary_score = float(scores[boundary_k]["decode_tps_geomean"])
        boundary_relative = boundary_score / initial_best - 1.0
        previous = max((draft_k for draft_k in initial_ks if draft_k < boundary_k), default=None)
        if previous is not None:
            boundary_trend = (
                boundary_score
                / float(scores[previous]["decode_tps_geomean"])
                - 1.0
            )
            boundary_condition = bool(
                boundary_relative >= -boundary_fraction and boundary_trend > 0.0
            )
    already_extended = max(ks) > boundary_k
    extension_required = bool(
        boundary_condition and not already_extended and max(ks) == boundary_k
    )

    return {
        "schema_version": 2,
        "tpot_definition": "decode_sec / (generated_output_tokens - 1)",
        "selection_metric": "mean_log_decode_tps",
        "tie_fraction": tie_fraction,
        "pair_count": len(shared_keys),
        "cluster_count": len(
            {(str(key[0]), str(key[1]), int(key[3])) for key in shared_keys}
        ),
        "tested_k": ks,
        "raw_best_k": best_k,
        "selected_k": selected_k,
        "scores": {str(key): value for key, value in scores.items()},
        "per_cache_ratio": per_ratio,
        "pairing": {
            "complete": not missing,
            "missing": missing,
        },
        "boundary_extension": {
            "boundary_k": boundary_k,
            "within_fraction": boundary_fraction,
            "relative_to_initial_best": boundary_relative,
            "trend_from_previous_k": boundary_trend,
            "condition_met": boundary_condition,
            "already_extended": already_extended,
            "extension_required": extension_required,
        },
        "sources": sources,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    report = analyze(
        list(args.inputs),
        tie_fraction=float(args.tie_fraction),
        boundary_k=int(args.boundary_k),
        boundary_fraction=float(args.boundary_fraction),
        require_complete=bool(args.require_complete),
    )
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", default="")
    parser.add_argument("--tie-fraction", type=float, default=0.005)
    parser.add_argument("--boundary-k", type=int, default=15)
    parser.add_argument("--boundary-fraction", type=float, default=0.01)
    parser.add_argument(
        "--require-complete",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
