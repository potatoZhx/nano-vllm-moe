"""Build random_cache theoretical-acceptance predictor datasets.

Input JSONL format is produced by collect_random_cache_acceptance.py. The output
.pt file stores separate tensors instead of one opaque concatenated feature so
that the predictor can keep lightweight branch encoders:

  route_raw:      [N, 768] for Qwen3-30B-A3B default 48 layers × 8 × 2
  route_summary:  [N, S_r]
  token_features: [N, S_t]
  hidden:         [N, hidden_dim]
  history:        [N, S_h]
  y:              [N, 1] theoretical acceptance alpha

python -u build_random_cache_dataset.py \
  --train /data2/group_谈海生/mumura/dynamick/predictor/random_cache_runs_smoke/wiki_random_cache_lru_ratio0.5_topc0.5/acceptance_summary_20260613_204930.jsonl \
  --test /data2/group_谈海生/mumura/dynamick/predictor/random_cache_runs_smoke/mtbench_random_cache_lru_ratio0.5_topc0.5/acceptance_summary_20260613_221149.jsonl \
  --output /data2/group_谈海生/mumura/dynamick/predictor/random_cache_acceptance_dataset_20260613.pt
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


def find_latest_summary_file(directory: str) -> Optional[str]:
    if not directory or not os.path.exists(directory):
        print(f"Missing directory: {directory}")
        return None
    files = [f for f in os.listdir(directory) if f.endswith(".jsonl") and "summary" in f]
    if not files:
        files = [f for f in os.listdir(directory) if f.endswith(".jsonl")]
    if not files:
        print(f"No JSONL files found in {directory}")
        return None
    files.sort(reverse=True)
    return os.path.join(directory, files[0])


def parse_dir_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def safe_1d(values, length: int, default: float = 0.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size >= length:
        return arr[:length]
    out = np.full((length,), default, dtype=np.float32)
    out[: arr.size] = arr
    return out


def normalize_probs(x: np.ndarray) -> np.ndarray:
    x = np.maximum(x.astype(np.float32), 0.0)
    s = float(x.sum())
    if s <= 1e-8:
        return np.ones_like(x, dtype=np.float32) / max(1, x.size)
    return x / s


def entropy(p: np.ndarray) -> float:
    p = normalize_probs(p)
    return float(-(p * np.log(p + 1e-9)).sum())


def top2_margin(p: np.ndarray) -> float:
    if p.size < 2:
        return 0.0
    vals = np.sort(p)[::-1]
    return float(vals[0] - vals[1])


def extract_route_arrays(step: Dict, num_layers: int, top_k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return orig_ids, mod_ids, orig_w, mod_w, mask with shapes [L,K]."""
    orig_ids = np.zeros((num_layers, top_k), dtype=np.int64)
    mod_ids = np.zeros((num_layers, top_k), dtype=np.int64)
    orig_w = np.zeros((num_layers, top_k), dtype=np.float32)
    mod_w = np.zeros((num_layers, top_k), dtype=np.float32)
    mask = np.zeros((num_layers, top_k), dtype=np.float32)

    router = sorted(step["router"], key=lambda x: x.get("layer_idx", 0))
    for li, layer_meta in enumerate(router[:num_layers]):
        # Metadata is stored as [1, K] because the collector records one decode token.
        oi = layer_meta.get("original_ids", [[]])
        mi = layer_meta.get("modified_ids", [[]])
        ow = layer_meta.get("original_weights", [[]])
        mw = layer_meta.get("modified_weights", [[]])
        rm = layer_meta.get("replacement_mask", [[]])
        oi = oi[0] if oi and isinstance(oi[0], list) else oi
        mi = mi[0] if mi and isinstance(mi[0], list) else mi
        ow = ow[0] if ow and isinstance(ow[0], list) else ow
        mw = mw[0] if mw and isinstance(mw[0], list) else mw
        rm = rm[0] if rm and isinstance(rm[0], list) else rm
        orig_ids[li] = safe_1d(oi, top_k, 0).astype(np.int64)
        mod_ids[li] = safe_1d(mi, top_k, 0).astype(np.int64)
        orig_w[li] = safe_1d(ow, top_k, 0.0)
        mod_w[li] = safe_1d(mw, top_k, 0.0)
        mask[li] = safe_1d(rm, top_k, 0.0)
    return orig_ids, mod_ids, orig_w, mod_w, mask


def route_features(step: Dict, num_layers: int, top_k: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    _, _, orig_w, mod_w, mask = extract_route_arrays(step, num_layers=num_layers, top_k=top_k)
    route_raw = np.concatenate([orig_w.reshape(-1), mod_w.reshape(-1)]).astype(np.float32)

    diff = np.abs(orig_w - mod_w)
    l1 = diff.sum(axis=1)
    l2 = np.sqrt((diff * diff).sum(axis=1))
    rep_frac = mask.mean(axis=1)
    rep_mass = (orig_w * mask).sum(axis=1)
    ent_o = np.array([entropy(orig_w[i]) for i in range(num_layers)], dtype=np.float32)
    ent_s = np.array([entropy(mod_w[i]) for i in range(num_layers)], dtype=np.float32)
    ent_delta = ent_s - ent_o
    top1_o = orig_w.max(axis=1)
    margin_o = np.array([top2_margin(orig_w[i]) for i in range(num_layers)], dtype=np.float32)
    max_drop = diff.max(axis=1)

    layer_matrix = np.stack([l1, l2, rep_frac, rep_mass, ent_o, ent_s, ent_delta, top1_o, margin_o, max_drop], axis=1)

    def stats(x: np.ndarray) -> List[float]:
        return [float(x.mean()), float(x.max()), float(x.std())]

    summary: List[float] = []
    for col in range(layer_matrix.shape[1]):
        summary.extend(stats(layer_matrix[:, col]))

    # Early/mid/late block means for the most important disturbance signals.
    thirds = np.array_split(np.arange(num_layers), 3)
    for idxs in thirds:
        summary.extend([
            float(l1[idxs].mean()),
            float(rep_mass[idxs].mean()),
            float(rep_frac[idxs].mean()),
            float(ent_o[idxs].mean()),
            float(max_drop[idxs].mean()),
        ])

    scalars = {
        "rsd_l1": float(l1.mean()),
        "rsd_l2": float(l2.mean()),
        "rsd_max": float(l1.max()),
        "rep_count": float(rep_frac.mean()),
        "rep_mass": float(rep_mass.mean()),
        "route_entropy": float(ent_o.mean()),
    }
    return route_raw, np.asarray(summary, dtype=np.float32), scalars


def token_features(step: Dict) -> np.ndarray:
    payload = step.get("q_topk_logits", {})
    values = np.asarray(payload.get("values", []), dtype=np.float32)
    if values.size == 0:
        return np.zeros((10,), dtype=np.float32)

    top1 = float(values[0])
    top2 = float(values[1]) if values.size > 1 else top1
    margin12 = top1 - top2
    top5_mean = float(values[: min(5, values.size)].mean())
    top1_gap5 = top1 - top5_mean
    std = float(values.std())

    # Top-k normalized probability features. This is cheap and avoids full-vocab softmax dependency.
    shifted = values - values.max()
    probs = np.exp(shifted)
    probs = probs / max(float(probs.sum()), 1e-8)
    top1_prob = float(probs[0])
    ent = float(-(probs * np.log(probs + 1e-9)).sum())
    eff_vocab = float(np.exp(ent))
    return np.asarray([top1, top2, margin12, top5_mean, top1_gap5, std, top1_prob, ent, eff_vocab, float(values.size)], dtype=np.float32)


def hidden_vector(step: Dict) -> np.ndarray:
    return np.asarray(step.get("final_embedding", []), dtype=np.float32).reshape(-1)


def process_file(path: str, num_layers: int, top_k: int) -> Dict[str, List[np.ndarray]]:
    out = {"route_raw": [], "route_summary": [], "token_features": [], "hidden": [], "history": [], "y": []}

    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Parsing {os.path.basename(path)}"):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                steps = record.get("steps", [])
                cum_rsd = 0.0
                ema_rsd = 0.0
                max_rsd = 0.0
                cum_rep = 0.0
                ema_rep = 0.0
                max_rep = 0.0
                cum_entropy = 0.0
                ema_entropy = 0.0
                max_entropy = 0.0

                for i, step in enumerate(steps):
                    route_raw, route_sum, scalars = route_features(step, num_layers=num_layers, top_k=top_k)
                    tok = token_features(step)
                    hid = hidden_vector(step)
                    if hid.size == 0:
                        continue

                    rsd = scalars["rsd_l1"]
                    rep = scalars["rep_mass"]
                    ent = float(tok[7]) if tok.size > 7 else 0.0
                    t = i + 1
                    cum_rsd += rsd
                    ema_rsd = 0.7 * ema_rsd + 0.3 * rsd if i > 0 else rsd
                    max_rsd = max(max_rsd, rsd)
                    cum_rep += rep
                    ema_rep = 0.7 * ema_rep + 0.3 * rep if i > 0 else rep
                    max_rep = max(max_rep, rep)
                    cum_entropy += ent
                    ema_entropy = 0.7 * ema_entropy + 0.3 * ent if i > 0 else ent
                    max_entropy = max(max_entropy, ent)

                    hist = np.asarray([
                        t / max(1, len(steps)),
                        cum_rsd / t,
                        ema_rsd,
                        max_rsd,
                        cum_rep / t,
                        ema_rep,
                        max_rep,
                        cum_entropy / t,
                        ema_entropy,
                        max_entropy,
                        float(record.get("metadata", {}).get("prefill_len", 0)) / 8096.0,
                    ], dtype=np.float32)

                    alpha = float(step["alpha_theoretical"])
                    out["route_raw"].append(route_raw)
                    out["route_summary"].append(route_sum)
                    out["token_features"].append(tok)
                    out["hidden"].append(hid)
                    out["history"].append(hist)
                    out["y"].append(np.asarray([alpha], dtype=np.float32))
            except Exception:
                continue
    return out


def merge_parts(parts: Iterable[Dict[str, List[np.ndarray]]]) -> Dict[str, torch.Tensor]:
    merged: Dict[str, List[np.ndarray]] = {"route_raw": [], "route_summary": [], "token_features": [], "hidden": [], "history": [], "y": []}
    for part in parts:
        for k in merged:
            merged[k].extend(part[k])
    if not merged["y"]:
        raise RuntimeError("No valid examples parsed.")
    return {k: torch.tensor(np.stack(v), dtype=torch.float32) for k, v in merged.items()}


def build_split(dirs_or_files: List[str], num_layers: int, top_k: int) -> Dict[str, torch.Tensor]:
    paths: List[str] = []
    for item in dirs_or_files:
        if os.path.isfile(item):
            paths.append(item)
        else:
            latest = find_latest_summary_file(item)
            if latest:
                paths.append(latest)
    print("Files:")
    for p in paths:
        print(f"  - {p}")
    return merge_parts(process_file(p, num_layers, top_k) for p in paths)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Comma-separated train JSONL files or directories, e.g. Wiki random_cache runs")
    parser.add_argument("--test", required=True, help="Comma-separated test JSONL files or directories, e.g. MTBench random_cache runs")
    parser.add_argument("--output", default="/data2/group_谈海生/lagin/data/Sd_Data/data/random_cache_acceptance_dataset.pt")
    parser.add_argument("--num-layers", type=int, default=48)
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    train_items = parse_dir_list(args.train)
    test_items = parse_dir_list(args.test)
    print("=== Building train split ===")
    train = build_split(train_items, args.num_layers, args.top_k)
    print("=== Building test split ===")
    test = build_split(test_items, args.num_layers, args.top_k)

    data = {
        "train": train,
        "test": test,
        "meta": {
            "num_layers": args.num_layers,
            "top_k": args.top_k,
            "route_raw_dim": int(train["route_raw"].shape[1]),
            "route_summary_dim": int(train["route_summary"].shape[1]),
            "token_feature_dim": int(train["token_features"].shape[1]),
            "hidden_dim": int(train["hidden"].shape[1]),
            "history_dim": int(train["history"].shape[1]),
            "target": "theoretical_acceptance_alpha=sum(min(p,q))",
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.output)
    print(f"Saved dataset to {args.output}")
    print("Meta:", data["meta"])
    print(f"Train examples: {len(train['y'])}, Test examples: {len(test['y'])}")
    print(f"Train alpha mean/std: {train['y'].mean().item():.4f}/{train['y'].std().item():.4f}")
    print(f"Test  alpha mean/std: {test['y'].mean().item():.4f}/{test['y'].std().item():.4f}")


if __name__ == "__main__":
    main()
