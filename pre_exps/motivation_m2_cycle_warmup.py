"""
motivation_m2_cycle_warmup.py
=============================
M2 Experiment: Cycle-to-Cycle Cache Warmup Effect

Demonstrates that verify-phase expert loading provides positive feedback
for the next draft cycle's cache hit rate. This effect is unique to
self-speculative architectures — quantization-based drafts (MoE-SpeQ)
have fixed error that does not improve with decode progress.

Method:
  1. Single full-model forward → collect target routing for all positions
  2. Simulate draft-verify cycles with dynamic LFU cache:
     a. Draft K steps: measure miss rate against current cache
     b. Verify: update cache with target routing (LFU mark_access + load)
     c. Repeat
  3. Compare "with verify update" vs "frozen cache" (no update after init)

Key outputs:
  - Per-cycle miss rate (with vs without cache update)
  - Per-cycle average hit rate
  - Post-verify first-step hit rate (cache warmth after verify)

Standalone — no dependency on nano-vllm-moe runtime.

Usage:
  python motivation_m2_cycle_warmup.py \
      --model /path/to/Qwen3-30B-A3B-Base \
      --data_file ../wikitext2_test.txt \
      --cache_ratios 0.25 0.50 --draft_len 8 --prompt_len 128 \
      --n_calib 8 --n_eval 16 --outdir ./results_m2
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Model & data utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_path, device, dtype=torch.float16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True, low_cpu_mem_usage=True)
    if device == "cpu":
        model = model.to(device)
    model.eval()
    return model, tok


def detect_moe_config(model) -> dict:
    cfg = model.config
    N = getattr(cfg, "num_experts", getattr(cfg, "n_routed_experts", None))
    k = getattr(cfg, "num_experts_per_tok", getattr(cfg, "top_k", None))
    if N is None or k is None:
        for layer in model.model.layers:
            moe = getattr(layer, "mlp", None)
            if moe and hasattr(moe, "experts"):
                N = N or len(moe.experts)
                k = k or getattr(moe, "top_k", None)
                break
    moe_attr = "mlp"
    for layer in model.model.layers:
        if hasattr(layer, "block_sparse_moe"):
            moe_attr = "block_sparse_moe"; break
    L = sum(1 for layer in model.model.layers
            if hasattr(getattr(layer, moe_attr, None), "experts"))
    return {"num_experts": N, "top_k": k, "num_layers": L, "moe_attr": moe_attr}


def get_moe_layers(model, moe_attr):
    return [(i, getattr(layer, moe_attr))
            for i, layer in enumerate(model.model.layers)
            if hasattr(getattr(layer, moe_attr, None), "experts")]


_SNIPPET = (
    "Wikipedia is a free online encyclopedia. Language models predict the next "
    "token given previous tokens. Mixture-of-Experts activates a subset of params. "
) * 200


def prepare_chunks(tokenizer, n, seq_len, data_file=None):
    text = None
    if data_file:
        try:
            text = open(data_file, encoding="utf-8").read()
            print(f"  Loaded {len(text):,} chars from {data_file}.")
        except Exception as e:
            print(f"  Warning: {e}")
    if text is None:
        try:
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-2-raw-v1",
                              split="test", trust_remote_code=True)
            text = "\n\n".join(ds["text"])
        except Exception:
            text = _SNIPPET
    enc = tokenizer(text, return_tensors="pt")["input_ids"]
    total = enc.size(1)
    chunks = [enc[:, s:s+seq_len] for s in range(0, total - seq_len, seq_len)][:n]
    if not chunks:
        raise RuntimeError(f"Text too short ({total} tokens) for seq_len={seq_len}.")
    if len(chunks) < n:
        chunks = (chunks * (n // len(chunks) + 1))[:n]
    random.shuffle(chunks)
    print(f"  {len(chunks)} chunks x {seq_len} tokens.")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Dynamic LFU Cache
# ─────────────────────────────────────────────────────────────────────────────

class DynamicLFUCache:
    """LFU cache with dynamic updates (evict-on-load)."""

    def __init__(self, L: int, N: int, S: int,
                 act_freq: Optional[np.ndarray] = None):
        self.L = L
        self.N = N
        self.S = S
        self.freq = [np.zeros(N, dtype=np.float64) for _ in range(L)]
        self.cached = [set() for _ in range(L)]

        if act_freq is not None:
            for l in range(L):
                top = np.argsort(act_freq[l])[::-1][:S]
                self.cached[l] = set(top.tolist())
                for e in top:
                    self.freq[l][e] = float(act_freq[l][e])

    def hit_rate(self, layer: int, expert_ids) -> float:
        expert_ids = set(expert_ids) if not isinstance(expert_ids, set) else expert_ids
        if not expert_ids:
            return 1.0
        return len(expert_ids & self.cached[layer]) / len(expert_ids)

    def hit_rate_all_layers(self, routing: Dict[int, Set[int]]) -> float:
        """Average hit rate across all layers for one token's routing."""
        rates = []
        for li, experts in routing.items():
            rates.append(self.hit_rate(li, experts))
        return float(np.mean(rates)) if rates else 1.0

    def miss_rate_all_layers(self, routing: Dict[int, Set[int]]) -> float:
        return 1.0 - self.hit_rate_all_layers(routing)

    def ensure_loaded(self, layer: int, expert_ids):
        """Load missing experts with LFU eviction."""
        for e in expert_ids:
            if e in self.cached[layer]:
                self.freq[layer][e] += 1.0
                continue
            if len(self.cached[layer]) < self.S:
                self.cached[layer].add(e)
                self.freq[layer][e] += 1.0
            else:
                min_freq = float('inf')
                min_expert = -1
                for ce in self.cached[layer]:
                    if self.freq[layer][ce] < min_freq:
                        min_freq = self.freq[layer][ce]
                        min_expert = ce
                if min_expert >= 0:
                    self.cached[layer].discard(min_expert)
                    self.cached[layer].add(e)
                    self.freq[layer][e] = min_freq + 1.0

    def ensure_loaded_all_layers(self, routing: Dict[int, Set[int]]):
        for li, experts in routing.items():
            self.ensure_loaded(li, experts)

    def snapshot(self):
        return ([s.copy() for s in self.cached],
                [f.copy() for f in self.freq])

    def restore(self, snap):
        cached_snap, freq_snap = snap
        for l in range(self.L):
            self.cached[l] = cached_snap[l].copy()
            self.freq[l] = freq_snap[l].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Routing trace collection
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RoutingTrace:
    """Per-token routing: routing[pos] = {layer_idx: set_of_expert_ids}"""
    routing: List[Dict[int, Set[int]]]


@torch.no_grad()
def collect_routing_and_freq(
    model, moe_attr: str, chunks: List[torch.Tensor],
    device: str, N: int, top_k: int
) -> Tuple[List[RoutingTrace], np.ndarray]:
    """Single forward pass per chunk → routing traces + activation frequency."""
    layers = get_moe_layers(model, moe_attr)
    L = len(layers)

    act_freq = np.zeros((L, N), dtype=np.float64)
    total_tokens = 0

    gate_bufs: List[Optional[torch.Tensor]] = [None] * L
    hooks = []
    for li, (idx, moe) in enumerate(layers):
        if hasattr(moe, "gate"):
            def _ghook(li_):
                def h(m, inp, out):
                    r = out[0] if isinstance(out, tuple) else out
                    gate_bufs[li_] = r.detach().float().cpu()
                return h
            hooks.append(moe.gate.register_forward_hook(_ghook(li)))

    traces = []
    for chunk in tqdm(chunks, desc="  Collecting routing"):
        inp = chunk.to(device)
        T = inp.size(1)
        total_tokens += T

        for i in range(L):
            gate_bufs[i] = None
        model(inp)

        per_pos = []
        for pos in range(T):
            step_routing: Dict[int, Set[int]] = {}
            for li in range(L):
                g = gate_bufs[li]
                if g is None:
                    continue
                if g.dim() == 3:
                    g = g[0]
                if pos >= g.size(0):
                    continue
                w = torch.softmax(g[pos], dim=-1)
                _, topk_i = torch.topk(w, top_k)
                experts = set(topk_i.tolist())
                step_routing[li] = experts
                for e in experts:
                    act_freq[li, e] += 1
            per_pos.append(step_routing)
        traces.append(RoutingTrace(routing=per_pos))

    for h in hooks:
        h.remove()
    if total_tokens > 0:
        act_freq /= total_tokens

    return traces, act_freq


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Cycle simulation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CycleRecord:
    cycle: int
    draft_miss_rates: List[float]       # per-step miss rate within draft
    draft_avg_miss: float               # mean miss rate in this cycle
    post_verify_step1_hit: float        # hit rate of next draft's first step


def simulate_cycles(
    trace: RoutingTrace,
    cache: DynamicLFUCache,
    prompt_len: int,
    draft_len: int,
    update_cache: bool = True,
) -> List[CycleRecord]:
    """Simulate draft-verify cycles on one routing trace."""
    routing = trace.routing
    T = len(routing)
    records = []

    pos = prompt_len
    cycle = 0
    while pos + draft_len < T:
        # Draft K steps: measure miss rate
        step_miss = []
        for step in range(draft_len):
            p = pos + step
            if p >= T:
                break
            mr = cache.miss_rate_all_layers(routing[p])
            step_miss.append(mr)

        avg_miss = float(np.mean(step_miss)) if step_miss else 0.0

        # Verify: update cache with target routing
        if update_cache:
            for step in range(draft_len + 1):  # K + 1 (bonus token)
                p = pos + step
                if p >= T:
                    break
                cache.ensure_loaded_all_layers(routing[p])

        # Post-verify: measure next draft step 1 hit rate
        next_pos = pos + draft_len + 1
        if next_pos < T:
            post_hit = cache.hit_rate_all_layers(routing[next_pos])
        else:
            post_hit = float('nan')

        records.append(CycleRecord(
            cycle=cycle,
            draft_miss_rates=step_miss,
            draft_avg_miss=avg_miss,
            post_verify_step1_hit=post_hit,
        ))

        pos = next_pos
        cycle += 1

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    traces: List[RoutingTrace],
    act_freq: np.ndarray,
    L: int, N: int,
    prompt_len: int, draft_len: int,
    cache_ratios: List[float],
) -> dict:
    results = {}

    for ratio in cache_ratios:
        S = max(1, int(N * ratio))
        print(f"\n{'='*60}")
        print(f"  cache_ratio={ratio:.3f} (S={S}/{N} per layer)")
        print(f"{'='*60}")

        all_with = []
        all_without = []

        for ti, trace in enumerate(tqdm(traces, desc="  Simulating")):
            # With verify cache update
            cache_w = DynamicLFUCache(L, N, S, act_freq=act_freq)
            recs_w = simulate_cycles(trace, cache_w, prompt_len, draft_len,
                                     update_cache=True)
            all_with.append(recs_w)

            # Without update (frozen cache after init)
            cache_f = DynamicLFUCache(L, N, S, act_freq=act_freq)
            recs_f = simulate_cycles(trace, cache_f, prompt_len, draft_len,
                                     update_cache=False)
            all_without.append(recs_f)

        # Aggregate by cycle index
        max_cycles = max(len(recs) for recs in all_with) if all_with else 0
        with_miss_by_cycle = defaultdict(list)
        without_miss_by_cycle = defaultdict(list)
        with_post_hit_by_cycle = defaultdict(list)
        without_post_hit_by_cycle = defaultdict(list)

        for recs in all_with:
            for r in recs:
                with_miss_by_cycle[r.cycle].append(r.draft_avg_miss)
                if math.isfinite(r.post_verify_step1_hit):
                    with_post_hit_by_cycle[r.cycle].append(
                        r.post_verify_step1_hit)
        for recs in all_without:
            for r in recs:
                without_miss_by_cycle[r.cycle].append(r.draft_avg_miss)
                if math.isfinite(r.post_verify_step1_hit):
                    without_post_hit_by_cycle[r.cycle].append(
                        r.post_verify_step1_hit)

        cycles_range = sorted(set(with_miss_by_cycle.keys())
                              | set(without_miss_by_cycle.keys()))

        summary = {
            "cycles": cycles_range,
            "with_update_miss": [float(np.mean(with_miss_by_cycle.get(c, [float('nan')])))
                                 for c in cycles_range],
            "frozen_miss":      [float(np.mean(without_miss_by_cycle.get(c, [float('nan')])))
                                 for c in cycles_range],
            "with_update_post_hit": [float(np.mean(with_post_hit_by_cycle.get(c, [float('nan')])))
                                     for c in cycles_range],
            "frozen_post_hit":      [float(np.mean(without_post_hit_by_cycle.get(c, [float('nan')])))
                                     for c in cycles_range],
        }

        w_miss_all = [r.draft_avg_miss for recs in all_with for r in recs]
        f_miss_all = [r.draft_avg_miss for recs in all_without for r in recs]
        print(f"  With update:  mean miss={np.mean(w_miss_all):.4f}")
        print(f"  Frozen cache: mean miss={np.mean(f_miss_all):.4f}")
        delta = np.mean(f_miss_all) - np.mean(w_miss_all)
        print(f"  Improvement:  {delta:+.4f} ({delta/max(np.mean(f_miss_all),1e-9)*100:+.1f}%)")

        results[ratio] = summary

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(results: dict, outdir: str):
    for ratio, summary in results.items():
        cycles = summary["cycles"]
        if not cycles:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Miss rate per cycle
        ax = axes[0]
        ax.plot(cycles, summary["with_update_miss"], "o-",
                color="#1D9E75", linewidth=2, markersize=5,
                label="With verify cache update")
        ax.plot(cycles, summary["frozen_miss"], "s--",
                color="#d4640c", linewidth=2, markersize=5,
                label="Frozen cache (no update)")
        ax.set_xlabel("Draft-Verify Cycle", fontsize=11)
        ax.set_ylabel("Average Miss Rate", fontsize=11)
        ax.set_title(f"Per-Cycle Miss Rate (ratio={ratio:.3f})", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Post-verify first step hit rate
        ax = axes[1]
        ax.plot(cycles, summary["with_update_post_hit"], "o-",
                color="#1D9E75", linewidth=2, markersize=5,
                label="With verify update")
        ax.plot(cycles, summary["frozen_post_hit"], "s--",
                color="#d4640c", linewidth=2, markersize=5,
                label="Frozen cache")
        ax.set_xlabel("Draft-Verify Cycle", fontsize=11)
        ax.set_ylabel("Hit Rate (step 1 of next draft)", fontsize=11)
        ax.set_title(f"Post-Verify Step-1 Hit Rate (ratio={ratio:.3f})", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.suptitle("M2: Cycle-to-Cycle Cache Warmup Effect", fontsize=13)
        plt.tight_layout()
        path = os.path.join(outdir, f"m2_cycle_warmup_r{ratio:.3f}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")


def save_results(results: dict, outdir: str):
    serial = {}
    for ratio, summary in results.items():
        serial[str(ratio)] = {
            k: ([int(c) for c in v] if k == "cycles"
                else [float(x) if math.isfinite(x) else None for x in v])
            for k, v in summary.items()
        }
    path = os.path.join(outdir, "m2_results.json")
    with open(path, "w") as f:
        json.dump(serial, f, indent=2)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="M2: Cycle-to-cycle cache warmup experiment")
    p.add_argument("--model",       required=True)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--dtype",       default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--data_file",   default=None)
    p.add_argument("--cache_ratios", nargs="+", type=float,
                   default=[0.25, 0.50])
    p.add_argument("--prompt_len",  type=int, default=128)
    p.add_argument("--draft_len",   type=int, default=8)
    p.add_argument("--n_calib",     type=int, default=8,
                   help="Chunks for calibration (activation freq)")
    p.add_argument("--n_eval",      type=int, default=16,
                   help="Chunks for cycle simulation")
    p.add_argument("--seq_len",     type=int, default=512,
                   help="Sequence length (longer → more cycles)")
    p.add_argument("--outdir",      default="./results_m2")
    p.add_argument("--seed",        type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}

    # Load model
    print("Loading model ...")
    model, tok = load_model_and_tokenizer(
        args.model, args.device, dtype_map[args.dtype])
    moe_cfg = detect_moe_config(model)
    L, N, top_k = moe_cfg["num_layers"], moe_cfg["num_experts"], moe_cfg["top_k"]
    print(f"MoE config: L={L}, N={N}, top_k={top_k}")

    # Prepare data
    print(f"\nPreparing {args.n_eval} eval chunks (seq_len={args.seq_len}) ...")
    eval_chunks = prepare_chunks(tok, args.n_eval, args.seq_len, args.data_file)

    # Collect routing traces + activation frequency
    print("\nCollecting routing traces ...")
    traces, act_freq = collect_routing_and_freq(
        model, moe_cfg["moe_attr"], eval_chunks,
        args.device, N, top_k)
    print(f"  Collected {len(traces)} traces")

    # Run cycle simulation
    print("\nRunning cycle simulation ...")
    results = run_experiment(
        traces, act_freq, L, N,
        args.prompt_len, args.draft_len,
        args.cache_ratios)

    # Report
    print(f"\n{'='*60}")
    print("  Saving results ...")
    print(f"{'='*60}")
    plot_results(results, args.outdir)
    save_results(results, args.outdir)
    print(f"\nDone. Results in {args.outdir}/")


if __name__ == "__main__":
    main()
