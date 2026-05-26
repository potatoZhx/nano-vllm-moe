"""
routing_entropy_analysis.py
============================
Experiment: Does MoE routing entropy decrease and expert consistency increase
as context grows?

Hypothesis (from speculative-decoding analysis):
  "With richer context, the router produces more peaked distributions,
   causing repeatedly activated experts to converge."

We decompose this into four measurable sub-claims:

  H1. Per-token routing entropy H(g_l(h_t)) decreases as position t increases.
  H2. Consecutive-token expert overlap Jaccard(S_t, S_{t+1}) increases with t.
  H3. Top-1 expert routing weight max_i w_i(h_t) increases with t.
  H4. Cross-sequence expert agreement increases with t:
      at late positions, different sequences from the same domain activate
      more similar expert sets than at early positions.

Methodology:
  - Prefill the model on full sequences (causal, single forward pass).
  - At each MoE layer, record routing logits and top-k sets for every position.
  - Aggregate statistics across layers and sequences.
  - Plot H1-H4 vs. sequence position.
  - Fit a linear trend to quantify the slope.
  - Report p-value from Spearman rank correlation.

Usage:
  CUDA_VISIBLE_DEVICES=2 python routing_entropy_analysis.py \
      --model /zx_data1/models/Qwen--Qwen3-30B-A3B-Base/ \
      --data_file /zx_data1/models/datasets/wikitext2_test.txt \
      --n_seqs 64 --seq_len 256 --outdir ./routing_analysis
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (same helpers as main eval script)
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_path: str, device: str,
                              dtype=torch.float16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True, low_cpu_mem_usage=True)
    if device == "cpu":
        model = model.to(device)
    model.eval()
    return model, tokenizer


def get_moe_layer_indices(model, moe_attr="mlp") -> List[int]:
    return [i for i, layer in enumerate(model.model.layers)
            if hasattr(getattr(layer, moe_attr, None), "experts")]


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def load_sequences(tokenizer, n_seqs: int, seq_len: int,
                   data_file: Optional[str] = None) -> List[torch.Tensor]:
    full_text = None
    if data_file:
        try:
            with open(data_file, encoding="utf-8") as f:
                full_text = f.read()
            print(f"  Loaded {len(full_text):,} chars from {data_file}")
        except Exception as e:
            print(f"  ⚠ {e}")
    if full_text is None:
        try:
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-2-raw-v1",
                              split="test", trust_remote_code=True)
            full_text = "\n\n".join(ds["text"])
        except Exception as e:
            raise RuntimeError(
                f"No data available ({e}). Provide --data_file.") from e

    enc   = tokenizer(full_text, return_tensors="pt")["input_ids"]
    total = enc.size(1)
    seqs  = []
    for s in range(0, total - seq_len, seq_len):
        seqs.append(enc[:, s:s + seq_len])
        if len(seqs) >= n_seqs:
            break
    if not seqs:
        raise RuntimeError(
            f"Text too short: {total} tokens, need {seq_len}.")
    random.shuffle(seqs)
    print(f"  Prepared {len(seqs)} sequences × {seq_len} tokens")
    return seqs[:n_seqs]


# ─────────────────────────────────────────────────────────────────────────────
# Core measurement
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def measure_routing_statistics(
    model,
    sequences: List[torch.Tensor],
    device: str,
    moe_attr: str = "mlp",
    top_k: int = 8,
) -> Dict:
    """
    Single forward pass per sequence; hooks collect routing logits at every
    MoE layer for every token position.

    Returns a dict with arrays indexed by position (0..seq_len-1):
      entropy_by_pos      [T]   mean routing entropy across layers & sequences
      top1_weight_by_pos  [T]   mean top-1 routing weight
      jaccard_by_pos      [T-1] mean consecutive-token expert-set Jaccard similarity
      cross_seq_jaccard   [T]   mean pairwise Jaccard across sequences
      pertoken_ppl        [T]   per-position mean cross-entropy (token-level PPL proxy)
    """
    moe_indices = get_moe_layer_indices(model, moe_attr)
    L_moe       = len(moe_indices)
    T           = sequences[0].size(1)
    N_exp       = model.config.num_experts \
                  if hasattr(model.config, "num_experts") else \
                  getattr(model.config, "n_routed_experts", 128)

    # Accumulators: shape [T, n_seqs, L_moe]
    # We collect per-sequence per-layer statistics then aggregate.
    all_entropy    = []   # list of [T, L_moe] arrays, one per sequence
    all_top1_w     = []
    all_topk_sets  = []   # list of [T, L_moe, top_k] int arrays
    all_token_nlls = []   # list of [T-1] (position 1..T-1 has a prediction)

    # ── Hook infrastructure ─────────────────────────────────────────────────
    # We buffer gate outputs during a forward pass.
    gate_buf: Dict[int, torch.Tensor] = {}   # layer_pos_in_moe_list → [T, N]

    def make_gate_hook(moe_pos: int):
        def hook(module, inp, out):
            # out: [T, N_exp] logit tensor (before softmax)
            gate_buf[moe_pos] = out.detach().float().cpu()
        return hook

    handles = []
    for moe_pos, global_idx in enumerate(moe_indices):
        moe_mod = getattr(model.model.layers[global_idx], moe_attr)
        h = moe_mod.gate.register_forward_hook(make_gate_hook(moe_pos))
        handles.append(h)

    # ── Forward pass loop ──────────────────────────────────────────────────
    for seq in tqdm(sequences, desc="  Measuring routing"):
        seq = seq.to(device)      # [1, T]
        gate_buf.clear()

        # Single causal forward pass
        out = model(seq, use_cache=False)
        # Per-token NLL: prediction at position t for token t+1
        logits = out.logits[0, :-1, :].float()          # [T-1, vocab]
        labels = seq[0, 1:].cpu()                        # [T-1]
        log_probs = F.log_softmax(logits, dim=-1).cpu()
        token_nlls = -log_probs[
            torch.arange(len(labels)), labels].numpy()   # [T-1]
        all_token_nlls.append(token_nlls)
        del out, logits, log_probs

        # Collect routing statistics: gate_buf[moe_pos] is [T, N]
        if len(gate_buf) != L_moe:
            continue  # incomplete capture, skip

        seq_entropy   = np.zeros((T, L_moe))    # [T, L]
        seq_top1_w    = np.zeros((T, L_moe))    # [T, L]
        seq_topk_sets = np.zeros((T, L_moe, top_k), dtype=np.int32)

        for moe_pos in range(L_moe):
            logits_layer = gate_buf[moe_pos]     # [T, N]
            probs = F.softmax(
                torch.from_numpy(logits_layer.numpy()), dim=-1).numpy()  # [T, N]

            # H1: routing entropy at each position
            ent = -(probs * np.log(probs + 1e-12)).sum(axis=-1)   # [T]
            seq_entropy[:, moe_pos] = ent

            # H3: top-1 routing weight
            seq_top1_w[:, moe_pos] = probs.max(axis=-1)

            # H2/H4: top-k expert sets
            topk_idx = np.argsort(probs, axis=-1)[:, -top_k:]  # [T, k]
            seq_topk_sets[:, moe_pos, :] = topk_idx

        all_entropy.append(seq_entropy)
        all_top1_w.append(seq_top1_w)
        all_topk_sets.append(seq_topk_sets)

    for h in handles:
        h.remove()

    n_valid = len(all_entropy)
    if n_valid == 0:
        raise RuntimeError("No valid sequences collected.")

    print(f"  Collected statistics from {n_valid} sequences.")

    # ── H1: entropy per position (mean over seqs × layers) ─────────────────
    # all_entropy: list of [T, L], stack to [n_valid, T, L]
    ent_arr   = np.stack(all_entropy,   axis=0)   # [S, T, L]
    top1_arr  = np.stack(all_top1_w,   axis=0)   # [S, T, L]
    topk_arr  = np.stack(all_topk_sets, axis=0)   # [S, T, L, k]

    entropy_by_pos = ent_arr.mean(axis=(0, 2))    # [T]
    top1_by_pos    = top1_arr.mean(axis=(0, 2))   # [T]

    # ── H2: consecutive Jaccard over layers & sequences ────────────────────
    def jaccard_sets(a: np.ndarray, b: np.ndarray) -> float:
        """a, b: [k] int arrays (expert indices)."""
        sa, sb = set(a.tolist()), set(b.tolist())
        if not sa and not sb:
            return 1.0
        return len(sa & sb) / len(sa | sb)

    jaccard_by_pos = np.zeros(T - 1)
    n_jac = np.zeros(T - 1)
    for s in range(n_valid):
        for l in range(L_moe):
            for t in range(T - 1):
                j = jaccard_sets(topk_arr[s, t, l, :],
                                 topk_arr[s, t+1, l, :])
                jaccard_by_pos[t] += j
                n_jac[t] += 1
    jaccard_by_pos /= np.maximum(n_jac, 1)

    # ── H4: cross-sequence Jaccard (expert agreement between sequences) ─────
    # For each position t and layer l, compute mean pairwise Jaccard across seqs.
    # Full pairwise is O(S^2 * T * L * k); sample n_pairs pairs instead.
    n_pairs = min(200, n_valid * (n_valid - 1) // 2)
    seq_pairs = []
    rng = np.random.default_rng(42)
    idx_pool = [(i, j) for i in range(n_valid) for j in range(i+1, n_valid)]
    chosen   = rng.choice(len(idx_pool), size=n_pairs, replace=False)
    seq_pairs = [idx_pool[c] for c in chosen]

    cross_jac_by_pos = np.zeros(T)
    n_cj = np.zeros(T)
    for (sa, sb) in seq_pairs:
        for l in range(L_moe):
            for t in range(T):
                j = jaccard_sets(topk_arr[sa, t, l, :],
                                 topk_arr[sb, t, l, :])
                cross_jac_by_pos[t] += j
                n_cj[t] += 1
    cross_jac_by_pos /= np.maximum(n_cj, 1)

    # ── Per-token PPL proxy ─────────────────────────────────────────────────
    # token_nlls[i] corresponds to predicting token at position i+1
    # Pad with nan for position 0 (no prediction), align to positions 1..T-1
    nll_arr = np.stack(all_token_nlls, axis=0)   # [S, T-1]
    ppl_proxy = np.full(T, np.nan)
    ppl_proxy[1:] = nll_arr.mean(axis=0)          # position t → NLL for t

    return {
        "entropy_by_pos":      entropy_by_pos,
        "top1_weight_by_pos":  top1_by_pos,
        "jaccard_by_pos":      jaccard_by_pos,
        "cross_jac_by_pos":    cross_jac_by_pos,
        "ppl_proxy":           ppl_proxy,
        "n_seqs":              n_valid,
        "T":                   T,
        "L_moe":               L_moe,
        # Per-layer breakdown (averaged over sequences only)
        "entropy_per_layer":   ent_arr.mean(axis=0),   # [T, L]
        "top1_per_layer":      top1_arr.mean(axis=0),  # [T, L]
    }


# ─────────────────────────────────────────────────────────────────────────────
# Statistical testing
# ─────────────────────────────────────────────────────────────────────────────

def trend_test(y: np.ndarray, name: str) -> dict:
    """
    Spearman rank correlation between position index and y.
    Returns rho, p-value, and linear slope from least-squares.
    """
    finite = np.isfinite(y)
    x = np.arange(len(y))[finite]
    yf = y[finite]
    if len(x) < 3:
        return {"rho": np.nan, "p": np.nan, "slope": np.nan,
                "confirmed": False, "name": name}
    rho, p = stats.spearmanr(x, yf)
    slope, intercept, *_ = stats.linregress(x, yf)
    confirmed = bool(p < 0.05)
    direction = "↓" if slope < 0 else "↑"
    print(f"  {name:<30}  rho={rho:+.3f}  p={p:.3e}  "
          f"slope={slope:+.5f}/pos  {direction}  "
          f"{'✓ significant' if confirmed else '✗ not significant'}")
    return {"rho": float(rho), "p": float(p), "slope": float(slope),
            "confirmed": confirmed, "name": name}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(stats_dict: dict, outdir: str, smooth: int = 8):
    """
    Four sub-plots testing H1–H4, plus a per-layer entropy heatmap.
    smooth: window size for rolling average.
    """
    T = stats_dict["T"]
    pos = np.arange(T)

    def roll(x):
        """Rolling mean with window `smooth`."""
        return np.convolve(x, np.ones(smooth) / smooth, mode="valid")

    pos_r = pos[smooth // 2: T - smooth // 2 + 1][:len(roll(pos))]
    # For jaccard, length is T-1
    pos_j = pos[:-1]
    pos_jr = pos_j[smooth // 2: (T-1) - smooth // 2 + 1][:len(roll(pos_j))]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "MoE Routing Concentration vs. Sequence Position\n"
        "(each point = mean over all MoE layers and sequences)",
        fontsize=12, y=1.01)

    colors = {"raw": "#B4B2A9", "smooth": "#185FA5",
              "jac": "#1D9E75", "cross": "#D85A30", "ppl": "#9467bd"}

    # ── H1: Routing entropy ──────────────────────────────────────────────────
    ax = axes[0, 0]
    ent = stats_dict["entropy_by_pos"]
    ax.plot(pos, ent, color=colors["raw"], lw=0.8, alpha=0.5)
    ax.plot(pos_r, roll(ent), color=colors["smooth"], lw=2.0,
            label=f"rolling mean (w={smooth})")
    # Linear trend overlay
    slope, intercept, *_ = stats.linregress(pos[np.isfinite(ent)],
                                             ent[np.isfinite(ent)])
    ax.plot(pos, intercept + slope * pos, "--", color=colors["smooth"],
            alpha=0.7, lw=1.2, label=f"trend: {slope:+.5f}/step")
    ax.set_xlabel("Token position in sequence")
    ax.set_ylabel("Routing entropy H(g(h))")
    ax.set_title("H1: Does routing entropy decrease with context?")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)

    # ── H3: Top-1 routing weight ─────────────────────────────────────────────
    ax = axes[0, 1]
    top1 = stats_dict["top1_weight_by_pos"]
    ax.plot(pos, top1, color=colors["raw"], lw=0.8, alpha=0.5)
    ax.plot(pos_r, roll(top1), color=colors["smooth"], lw=2.0,
            label=f"rolling mean (w={smooth})")
    slope3, intercept3, *_ = stats.linregress(pos, top1)
    ax.plot(pos, intercept3 + slope3 * pos, "--", color=colors["smooth"],
            alpha=0.7, lw=1.2, label=f"trend: {slope3:+.5f}/step")
    ax.set_xlabel("Token position in sequence")
    ax.set_ylabel("Mean top-1 routing weight max_i w_i(h)")
    ax.set_title("H3: Does top-1 expert weight grow with context?")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)

    # ── H2: Consecutive Jaccard + PPL proxy ──────────────────────────────────
    ax = axes[1, 0]
    jac = stats_dict["jaccard_by_pos"]
    ax.plot(pos_j, jac, color=colors["raw"], lw=0.8, alpha=0.5)
    ax.plot(pos_jr, roll(jac), color=colors["jac"], lw=2.0,
            label=f"Jaccard(S_t, S_{{t+1}}) (w={smooth})")
    slope2, intercept2, *_ = stats.linregress(pos_j[np.isfinite(jac)],
                                               jac[np.isfinite(jac)])
    ax.plot(pos_j, intercept2 + slope2 * pos_j, "--", color=colors["jac"],
            alpha=0.7, lw=1.2, label=f"trend: {slope2:+.5f}/step")
    ax.set_xlabel("Token position t")
    ax.set_ylabel("Jaccard similarity of top-k expert sets")
    ax.set_title("H2: Do consecutive tokens activate more similar experts?")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)

    # Overlay per-token PPL on secondary axis
    ax2 = ax.twinx()
    ppl = stats_dict["ppl_proxy"]
    ax2.plot(pos[1:], ppl[1:], color=colors["ppl"], lw=1.2, alpha=0.6,
             label="per-token NLL (proxy for PPL)")
    ax2.set_ylabel("Per-token NLL", color=colors["ppl"])
    ax2.tick_params(axis="y", labelcolor=colors["ppl"])
    ax2.legend(loc="upper right", fontsize=8)

    # ── H4: Cross-sequence Jaccard ───────────────────────────────────────────
    ax = axes[1, 1]
    cjac = stats_dict["cross_jac_by_pos"]
    ax.plot(pos, cjac, color=colors["raw"], lw=0.8, alpha=0.5)
    ax.plot(pos_r, roll(cjac), color=colors["cross"], lw=2.0,
            label=f"cross-seq Jaccard (w={smooth})")
    slope4, intercept4, *_ = stats.linregress(pos[np.isfinite(cjac)],
                                               cjac[np.isfinite(cjac)])
    ax.plot(pos, intercept4 + slope4 * pos, "--", color=colors["cross"],
            alpha=0.7, lw=1.2, label=f"trend: {slope4:+.5f}/step")
    ax.set_xlabel("Token position")
    ax.set_ylabel("Mean pairwise Jaccard across sequences")
    ax.set_title(
        "H4: Do different sequences converge on the same experts later?")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)

    plt.tight_layout()
    p = os.path.join(outdir, "routing_concentration.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")

    # ── Per-layer entropy heatmap ────────────────────────────────────────────
    ent_pl = stats_dict["entropy_per_layer"]   # [T, L_moe]
    L_moe  = ent_pl.shape[1]

    # Sample positions evenly for readability
    n_pos_ticks = min(T, 32)
    pos_sample  = np.linspace(0, T - 1, n_pos_ticks, dtype=int)
    ent_sampled = ent_pl[pos_sample, :]   # [n_pos_ticks, L_moe]

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(ent_sampled.T, aspect="auto", cmap="RdYlGn_r",
                   origin="lower")
    ax.set_xlabel("Token position (sampled)")
    ax.set_ylabel("MoE layer index")
    ax.set_title(
        "Per-layer routing entropy H(g_l(h_t))  "
        "(green=low/peaked, red=high/diffuse)")
    ax.set_xticks(range(n_pos_ticks))
    ax.set_xticklabels(pos_sample, rotation=45, fontsize=7)
    plt.colorbar(im, ax=ax, label="entropy")
    plt.tight_layout()
    p2 = os.path.join(outdir, "entropy_heatmap.png")
    plt.savefig(p2, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p2}")

    # ── Scatter: routing entropy vs. per-token NLL ───────────────────────────
    # (tests whether routing concentration correlates with prediction confidence)
    ppl_v = ppl[1:]      # positions 1..T-1
    ent_v = ent[1:]
    finite = np.isfinite(ppl_v) & np.isfinite(ent_v)
    if finite.sum() > 10:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(ppl_v[finite], ent_v[finite],
                   alpha=0.15, s=6, color=colors["smooth"])
        rho_pe, p_pe = stats.spearmanr(ppl_v[finite], ent_v[finite])
        ax.set_xlabel("Per-token NLL (proxy for prediction uncertainty)")
        ax.set_ylabel("Mean routing entropy H(g(h))")
        ax.set_title(
            f"Routing entropy vs. prediction uncertainty\n"
            f"Spearman ρ={rho_pe:.3f}, p={p_pe:.2e}")
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        p3 = os.path.join(outdir, "entropy_vs_ppl.png")
        plt.savefig(p3, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Saved: {p3}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Measure MoE routing entropy vs. sequence position.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--data_file", default=None)
    parser.add_argument("--n_seqs", type=int, default=64,
                        help="Number of sequences to analyze.")
    parser.add_argument("--seq_len", type=int, default=256,
                        help="Tokens per sequence.")
    parser.add_argument("--smooth", type=int, default=12,
                        help="Rolling window size for plots.")
    parser.add_argument("--outdir", default="./routing_analysis")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    dtype_map = {"float16": torch.float16,
                 "bfloat16": torch.bfloat16, "float32": torch.float32}

    print("Loading model ...")
    model, tokenizer = load_model_and_tokenizer(
        args.model, args.device, dtype_map[args.dtype])
    top_k = getattr(model.config, "num_experts_per_tok",
                    getattr(model.config, "top_k", 8))
    print(f"Model: top_k={top_k}")

    print("\nLoading sequences ...")
    seqs = load_sequences(tokenizer, args.n_seqs, args.seq_len, args.data_file)

    print("\n── Measuring routing statistics ─────────────────────────────────")
    results = measure_routing_statistics(
        model, seqs, args.device, top_k=top_k)

    print("\n── Statistical tests ────────────────────────────────────────────")
    print("  Null hypothesis: no monotone trend across sequence positions.")
    test_results = {}
    test_results["H1_entropy"] = trend_test(
        results["entropy_by_pos"],
        "H1 routing entropy")
    test_results["H3_top1_weight"] = trend_test(
        results["top1_weight_by_pos"],
        "H3 top-1 routing weight")
    test_results["H2_consec_jaccard"] = trend_test(
        results["jaccard_by_pos"],
        "H2 consecutive Jaccard")
    test_results["H4_cross_jaccard"] = trend_test(
        results["cross_jac_by_pos"],
        "H4 cross-seq Jaccard")
    test_results["H_ppl"] = trend_test(
        results["ppl_proxy"],
        "per-token NLL")

    print("\n── Summary ──────────────────────────────────────────────────────")
    h1 = test_results["H1_entropy"]
    h3 = test_results["H3_top1_weight"]
    h2 = test_results["H2_consec_jaccard"]
    h4 = test_results["H4_cross_jaccard"]

    if h1["confirmed"] and h1["slope"] < 0:
        print("  ✓ H1 confirmed: routing entropy DECREASES with context.")
        print(f"    Average reduction: {h1['slope']*args.seq_len:.3f} nats "
              f"over {args.seq_len} positions.")
    else:
        print("  ✗ H1 not confirmed (or trend reversed).")

    if h3["confirmed"] and h3["slope"] > 0:
        print("  ✓ H3 confirmed: top-1 routing weight INCREASES with context.")
    else:
        print("  ✗ H3 not confirmed.")

    if h2["confirmed"] and h2["slope"] > 0:
        print("  ✓ H2 confirmed: consecutive expert overlap INCREASES with "
              "context.")
    else:
        print("  ✗ H2 not confirmed.")

    if h4["confirmed"] and h4["slope"] > 0:
        print("  ✓ H4 confirmed: cross-sequence expert agreement INCREASES "
              "with context.")
    else:
        print("  ✗ H4 not confirmed.")

    ppl_h = test_results["H_ppl"]
    if ppl_h["confirmed"] and ppl_h["slope"] < 0:
        print("  ✓ Control: per-token NLL decreases (model more confident "
              "later) — expected.")
    else:
        print("  ⚠ Control: per-token NLL does not decrease as expected.")

    # Correlation between routing entropy and per-token NLL
    ppl_v = results["ppl_proxy"][1:]
    ent_v = results["entropy_by_pos"][1:]
    finite = np.isfinite(ppl_v) & np.isfinite(ent_v)
    if finite.sum() > 10:
        rho_pe, p_pe = stats.spearmanr(ppl_v[finite], ent_v[finite])
        print(f"\n  Routing entropy ↔ NLL correlation: "
              f"ρ={rho_pe:.3f} (p={p_pe:.2e})")
        if p_pe < 0.05 and rho_pe > 0:
            print("  ✓ Routing entropy correlates positively with prediction "
                  "uncertainty — supports a unified mechanism.")

    print("\n── Generating plots ─────────────────────────────────────────────")
    plot_results(results, args.outdir, smooth=args.smooth)

    # Save raw arrays and test results
    out = {
        "test_results": test_results,
        "entropy_by_pos":     results["entropy_by_pos"].tolist(),
        "top1_weight_by_pos": results["top1_weight_by_pos"].tolist(),
        "jaccard_by_pos":     results["jaccard_by_pos"].tolist(),
        "cross_jac_by_pos":   results["cross_jac_by_pos"].tolist(),
        "ppl_proxy":          [float(x) for x in results["ppl_proxy"]],
        "metadata": {
            "n_seqs": results["n_seqs"],
            "seq_len": args.seq_len,
            "n_moe_layers": results["L_moe"],
            "model": args.model,
        }
    }
    json_path = os.path.join(args.outdir, "routing_stats.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {json_path}")
    print(f"\n✓ Done. Results in {args.outdir}/")


if __name__ == "__main__":
    main()
