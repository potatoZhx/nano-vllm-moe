"""
expert_rerouting_eval.py
========================
Standalone evaluation of the five draft-phase expert rerouting algorithms
defined in the DPERP formal framework.  No dependency on nano-vllm-moe.

Supports any Qwen-style MoE model (Qwen1.5-MoE-A2.7B recommended for fast runs).

Usage
-----
  # Full evaluation with default settings
  python expert_rerouting_eval.py --model /path/to/Qwen1.5-MoE-A2.7B

  # Low-memory CPU run for quick sanity check
  python expert_rerouting_eval.py --model /path/to/model \\
      --device cpu --n_calib 32 --n_eval 64 --seq_len 128

  # Sweep cache ratios and save results
  python expert_rerouting_eval.py --model /path/to/model \\
      --cache_ratios 0.125 0.25 0.5 0.75 --outdir ./results

Experiment Structure
--------------------
Phase 1 – Offline Calibration
  • Build pairwise output-similarity matrix S[L, N, N]
  • Build expected substitution-error matrix D[L, N, N]
  • Compute per-layer sensitivity weights ω[L]

Phase 2 – Algorithm Evaluation (per cache_ratio)
  • Simulate GPU cache: highest-frequency S experts per layer are "cached"
  • Apply rerouting wrapper to model; compare draft logits to baseline
  • Metrics: theoretical α (TV distance), PPL, layer output cosine similarity
  • Baselines: exact (upper bound), skip-all, round-robin

Phase 3 – Reporting
  • CSV tables, acceptance-rate vs cache-ratio curves, layer-sensitivity heatmaps

The five algorithms evaluated
-------------------------------
Alg1  Skip + Sim LUT         post-routing, skip/substitute, offline similarity, renormalize
Alg2  Entropy-Biased Routing  pre-routing logit bias scaled by token routing entropy
Alg3  Router-Score Merge      post-routing, joint online+offline score, sim-weighted merge
Alg4  Error-Budget Draft      post-routing, offline error budget, adaptive termination
Alg5  Online Bandit           post-routing, EMA acceptance rate, UCB, similarity prior
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Section 1 – Model utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_path: str, device: str, dtype=torch.float16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading tokenizer from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model ({dtype}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()
    return model, tokenizer


def detect_moe_config(model) -> dict:
    """Return {num_experts, top_k, num_layers, moe_attr} for a Qwen-style MoE."""
    cfg = model.config
    num_experts = getattr(cfg, "num_experts", getattr(cfg, "n_routed_experts", None))
    top_k       = getattr(cfg, "num_experts_per_tok", getattr(cfg, "top_k", None))

    # Fallback: introspect first MoE layer
    if num_experts is None or top_k is None:
        for layer in model.model.layers:
            moe = getattr(layer, "mlp", None)
            if moe is not None and hasattr(moe, "experts"):
                num_experts = num_experts or getattr(moe, "num_experts", len(moe.experts))
                top_k       = top_k or getattr(moe, "top_k", None)
                break

    assert num_experts and top_k, "Cannot detect MoE config; add explicit num_experts/top_k args."

    # Determine attribute name for the MoE sub-module
    moe_attr = "mlp"
    for layer in model.model.layers:
        if hasattr(layer, "block_sparse_moe"):
            moe_attr = "block_sparse_moe"
            break

    num_layers = sum(
        1 for layer in model.model.layers
        if hasattr(getattr(layer, moe_attr, None), "experts")
    )

    return {"num_experts": num_experts, "top_k": top_k,
            "num_layers": num_layers, "moe_attr": moe_attr}


def get_moe_layers(model, moe_attr: str):
    """Return list of (layer_idx, moe_module) for each MoE layer."""
    result = []
    for i, layer in enumerate(model.model.layers):
        moe = getattr(layer, moe_attr, None)
        if moe is not None and hasattr(moe, "experts"):
            result.append((i, moe))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 – Dataset utilities
# ─────────────────────────────────────────────────────────────────────────────

WIKITEXT_SNIPPET = """
Wikipedia is a free online encyclopedia that anyone can edit.
The project was launched in January 2001 by Jimmy Wales and Larry Sanger.
The name Wikipedia is a portmanteau of wiki and encyclopedia.
Mixture-of-Experts models activate only a subset of parameters per token.
Speculative decoding generates multiple draft tokens and verifies them in parallel.
Expert caching stores frequently used experts in GPU memory to reduce PCIe transfers.
The transformer architecture uses self-attention and feedforward layers.
Language models predict the next token given a sequence of previous tokens.
Reinforcement learning trains agents to maximize cumulative reward signals.
Deep learning extracts hierarchical representations from raw data.
"""


def prepare_text_chunks(tokenizer, n_chunks: int, seq_len: int,
                        dataset_name: str = "wikitext",
                        data_file: Optional[str] = None) -> List[torch.Tensor]:
    """Return a list of [1, seq_len] token-ID tensors.

    Priority:
      1. data_file (local plain-text file, one document per line or free-form)
      2. HuggingFace datasets (if `datasets` module is available)
      3. Built-in snippet (last resort, prints a prominent warning)
    """
    full_text = None

    # Priority 1: local file
    if data_file is not None:
        try:
            with open(data_file, "r", encoding="utf-8") as f:
                full_text = f.read()
            print(f"  Loaded text from {data_file} ({len(full_text):,} chars).")
        except Exception as e:
            print(f"  ⚠ Could not read {data_file}: {e}")

    # Priority 2: HuggingFace datasets
    if full_text is None:
        try:
            from datasets import load_dataset
            if dataset_name == "wikitext":
                ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test",
                                  trust_remote_code=True)
                full_text = "\n\n".join(ds["text"])
            else:
                ds = load_dataset(dataset_name, split="train[:5%]",
                                  trust_remote_code=True)
                full_text = "\n\n".join(ds["text"][:500])
        except Exception as e:
            print(f"  ⚠ Could not load dataset ({e}); will use built-in snippet.")

    # Priority 3: built-in snippet — warn loudly
    if full_text is None:
        print(
            "\n  ⚠⚠⚠  WARNING: using built-in snippet as calibration/eval data.\n"
            "  This produces DEGENERATE results because all chunks are near-identical.\n"
            "  Provide a local text file with --data_file /path/to/text.txt\n"
            "  (e.g. a Wikipedia dump, ShareGPT export, or any large plain-text file)\n"
        )
        # Expand snippet enough to avoid all-identical chunks
        snippet_repeat = max(1, (n_chunks * seq_len) // max(len(WIKITEXT_SNIPPET), 1) + 10)
        full_text = WIKITEXT_SNIPPET * snippet_repeat

    enc = tokenizer(full_text, return_tensors="pt")["input_ids"]
    total = enc.size(1)

    if total < seq_len + 1:
        print(f"  ⚠ Text too short ({total} tokens) for seq_len={seq_len}; "
              f"reducing seq_len to {total // 2}.")
        seq_len = max(total // 2, 1)

    chunks = []
    stride = seq_len
    for start in range(0, total - seq_len, stride):
        chunks.append(enc[:, start : start + seq_len])
        if len(chunks) >= n_chunks:
            break

    if len(chunks) < n_chunks:
        if len(chunks) == 0:
            raise RuntimeError(f"No chunks extracted from text (total={total}, seq_len={seq_len}).")
        n_repeat = n_chunks // len(chunks) + 1
        chunks = (chunks * n_repeat)[:n_chunks]
        print(f"  ⚠ Text too short; repeating chunks to reach {n_chunks}.")

    random.shuffle(chunks)
    print(f"  Prepared {len(chunks)} chunks of length {seq_len} "
          f"(from {total} total tokens).")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 – Simulated GPU cache
# ─────────────────────────────────────────────────────────────────────────────

class SimulatedCache:
    """
    Tracks which experts are 'GPU-resident' per layer.

    Policy: warm-start by caching the most frequently activated experts on a
    calibration set (LFU), matching what a real LRU/LFU cache would settle to.
    """

    def __init__(self, num_layers: int, num_experts: int, cache_ratio: float,
                 activation_freq: Optional[np.ndarray] = None):
        self.num_layers  = num_layers
        self.num_experts = num_experts
        self.cache_size  = max(1, int(num_experts * cache_ratio))
        # cached_experts[l] : set of expert indices in GPU cache at layer l
        self.cached_experts: List[set] = []

        if activation_freq is not None:
            # Sort by frequency; top S are cached
            for l in range(num_layers):
                ranked = np.argsort(activation_freq[l])[::-1]
                self.cached_experts.append(set(ranked[:self.cache_size].tolist()))
        else:
            # Random initialization
            for l in range(num_layers):
                cached = random.sample(range(num_experts), self.cache_size)
                self.cached_experts.append(set(cached))

    def cached_mask(self, layer_idx: int, device) -> torch.Tensor:
        """Boolean tensor [N] — True if expert is GPU-resident."""
        mask = torch.zeros(self.num_experts, dtype=torch.bool, device=device)
        mask[list(self.cached_experts[layer_idx])] = True
        return mask

    def cached_list(self, layer_idx: int) -> List[int]:
        return sorted(self.cached_experts[layer_idx])

    @property
    def mean_miss_rate(self) -> float:
        hits = sum(len(c) for c in self.cached_experts)
        total = self.num_layers * self.num_experts
        return 1.0 - hits / total


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 – Offline calibration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CalibrationData:
    """Everything computed during the calibration phase."""
    # [num_moe_layers, num_experts, num_experts] float16  — cosine sim of mean outputs
    similarity_table: Optional[torch.Tensor] = None
    # [num_moe_layers, num_experts, num_experts] float32 — normalised conditional replacement error
    error_table: Optional[torch.Tensor] = None
    # [num_moe_layers] float32 — layer sensitivity (0=insensitive,1=critical)
    sensitivity: Optional[torch.Tensor] = None
    # [num_moe_layers, num_experts] float32 — activation frequency
    activation_freq: Optional[np.ndarray] = None

    # ── Additional tables for similarity-method ablation ──────────────────
    # [num_moe_layers, num_experts, num_experts] float32
    #   co-activation frequency: P(j∈S | e∈S) on calibration set (BuddyMoE-style)
    coact_table: Optional[torch.Tensor] = None
    # [num_moe_layers, num_experts, num_experts] float32
    #   conditional replacement error (normalised by ‖E_e‖): 1 − D_cond(e,j)
    #   = 0 means identical output; > 0 means substitution is better than skip
    cond_error_sim_table: Optional[torch.Tensor] = None
    # [num_moe_layers, num_experts, num_experts] float32
    #   router logit Pearson correlation across calibration set
    router_corr_table: Optional[torch.Tensor] = None
    # [num_moe_layers, num_experts] float32 — mean output ‖E_e(h)‖₂ (when activated)
    expert_output_norm: Optional[torch.Tensor] = None

    # ── Fields for Static-EBB / Online-AB / Hybrid-CP ────────────────────
    # [num_moe_layers, num_experts] float32
    #   D(i,∅) = E[‖E_i(h)‖ / ‖h‖] — expected normalised skip error
    skip_error_table: Optional[torch.Tensor] = None
    # [num_moe_layers, num_experts] float32
    #   mean routing weight when expert i is activated
    routing_mass: Optional[torch.Tensor] = None
    # [num_moe_layers, num_experts] float32
    #   composite critical-expert score in [0,1]
    critical_score: Optional[torch.Tensor] = None
    # [num_moe_layers, num_experts, num_experts] float32
    #   λ_out·S_out + λ_coact·S_coact + λ_route·S_route (combined similarity)
    combined_sim_table: Optional[torch.Tensor] = None

    num_moe_layers: int = 0
    num_experts: int = 0


def build_calibration_data(
    model, tokenizer, moe_attr: str,
    calib_chunks: List[torch.Tensor],
    device: str,
    num_experts: int,
) -> CalibrationData:
    """
    Calibration that captures:
      1. Mean expert output activations  → cosine similarity table (Alg1_CosineOut)
      2. Co-activation frequency matrix  → co-activation table   (Alg1_CoAct)
      3. Conditional replacement error   → cond-error sim table  (Alg1_CondError)
      4. Router logit Pearson correlation→ router-corr table     (Alg1_RouterCorr)
      5. Mean expert output norm         → per-layer per-expert  (contribution skip)
      6. Expected substitution error D   → error table           (Alg4)
      7. Per-layer sensitivity ω         → sensitivity vector
      8. Activation frequency            → cache warm-start
    """
    moe_layers = get_moe_layers(model, moe_attr)
    L = len(moe_layers)
    D_hidden = model.config.hidden_size

    print(f"\n[Calibration] {L} MoE layers, {num_experts} experts, "
          f"{len(calib_chunks)} chunks")

    # ── Accumulators ──────────────────────────────────────────────────────
    # Mean outputs (for cosine sim)
    sum_out   = [torch.zeros(num_experts, D_hidden, dtype=torch.float32)
                 for _ in range(L)]
    count_out = [torch.zeros(num_experts, dtype=torch.float32) for _ in range(L)]

    # Co-activation: coact[l][e,j] = # times e and j both in top-k
    coact_count = [np.zeros((num_experts, num_experts), dtype=np.float64)
                   for _ in range(L)]
    act_count   = np.zeros((L, num_experts), dtype=np.float64)  # # times e activated

    # Router logit stats for Pearson correlation: E[g_e], E[g_e^2], E[g_e*g_j]
    # Store as running sums to avoid O(N²) per token
    gate_sum    = [np.zeros(num_experts, dtype=np.float64) for _ in range(L)]
    gate_sum2   = [np.zeros(num_experts, dtype=np.float64) for _ in range(L)]
    gate_cross  = [np.zeros((num_experts, num_experts), dtype=np.float64)
                   for _ in range(L)]
    gate_n      = [0] * L  # number of tokens seen at this layer

    # Conditional replacement error: sum of ‖E_e(h)-E_j(h)‖ / ‖E_e(h)‖ for each (e,j)
    # Only accumulated when e is in top-k (conditional on e being routed)
    cond_err_sum   = [np.zeros((num_experts, num_experts), dtype=np.float64)
                      for _ in range(L)]
    cond_err_count = [np.zeros(num_experts, dtype=np.float64) for _ in range(L)]

    # Mean output norm per expert
    norm_sum = [np.zeros(num_experts, dtype=np.float64) for _ in range(L)]

    # ── NEW: accumulators for skip error and routing mass ─────────────────
    # skip error D(i,∅) = E[‖E_i(h)‖ / ‖h‖] — need hidden state norm at layer l
    hidden_norm_sum   = np.zeros(L, dtype=np.float64)  # Σ ‖h‖ across all tokens
    hidden_norm_count = np.zeros(L, dtype=np.int64)
    # routing_mass[l, e] = mean routing weight when expert e is activated
    routing_weight_sum = np.zeros((L, num_experts), dtype=np.float64)

    # Activation freq and weight variance
    freq       = np.zeros((L, num_experts), dtype=np.float32)
    weight_var = np.zeros(L, dtype=np.float32)

    # ── Hooks ─────────────────────────────────────────────────────────────
    hook_handles = []
    layer_outputs: List[Dict[int, torch.Tensor]] = [{} for _ in range(L)]

    def make_expert_hook(l_idx, e_idx):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                out = out[0]
            layer_outputs[l_idx][e_idx] = out.detach().float().cpu()
        return hook

    def make_gate_hook(l_idx, top_k_cfg):
        def hook(module, inp, out):
            with torch.no_grad():
                logits = out.float().cpu().numpy()          # [T, N]
                T_local = logits.shape[0]

                # Router corr accumulators
                gate_sum[l_idx]   += logits.sum(0)
                gate_sum2[l_idx]  += (logits ** 2).sum(0)
                gate_cross[l_idx] += logits.T @ logits      # [N, N]
                gate_n[l_idx]     += T_local

                # Top-k routing
                w = np.exp(logits - logits.max(-1, keepdims=True))
                w /= w.sum(-1, keepdims=True)
                topk_idx = np.argsort(w, axis=-1)[:, -top_k_cfg:]  # [T, k]
                topk_w   = np.take_along_axis(w, topk_idx, axis=-1)

                for t in range(T_local):
                    active = topk_idx[t].tolist()
                    active_w = topk_w[t].tolist()
                    for e, we in zip(active, active_w):
                        freq[l_idx, e]              += 1.0
                        act_count[l_idx, e]         += 1.0
                        routing_weight_sum[l_idx, e] += we  # accumulate routing weight
                        for j in active:
                            coact_count[l_idx][e, j] += 1.0
                weight_var[l_idx] += float(topk_w.var())
        return hook

    def make_moe_input_hook(l_idx):
        """Hook on the MoE block input to capture hidden state norms."""
        def hook(module, inp, out):
            h = inp[0].detach().float()                    # [B, T, D]
            norms = h.reshape(-1, h.size(-1)).norm(dim=-1) # [B*T]
            hidden_norm_sum[l_idx]   += float(norms.sum().item())
            hidden_norm_count[l_idx] += int(norms.numel())
        return hook

    top_k_cfg = (model.config.num_experts_per_tok
                 if hasattr(model.config, "num_experts_per_tok") else 4)

    for l_idx, (_, moe_mod) in enumerate(moe_layers):
        h = moe_mod.gate.register_forward_hook(make_gate_hook(l_idx, top_k_cfg))
        hook_handles.append(h)
        h = moe_mod.register_forward_hook(make_moe_input_hook(l_idx))
        hook_handles.append(h)
        for e_idx, expert in enumerate(moe_mod.experts):
            h = expert.register_forward_hook(make_expert_hook(l_idx, e_idx))
            hook_handles.append(h)

    # ── Forward passes ────────────────────────────────────────────────────
    with torch.no_grad():
        for chunk in tqdm(calib_chunks, desc="  Calibration forward"):
            chunk = chunk.to(device)
            for ld in layer_outputs:
                ld.clear()
            try:
                model(chunk)
            except Exception as ex:
                print(f"    ⚠ Forward error (skipping chunk): {ex}")
                continue

            for l_idx in range(L):
                outs = layer_outputs[l_idx]  # {e_idx: Tensor[m, D]}
                for e_idx, out_e in outs.items():
                    if out_e.dim() != 2 or out_e.size(0) == 0:
                        continue
                    m = out_e.size(0)

                    # Mean output accumulation (cosine sim)
                    sum_out[l_idx][e_idx]   += out_e.sum(0)
                    count_out[l_idx][e_idx] += m

                    # Mean output norm
                    norms_e = out_e.norm(dim=-1).numpy()  # [m]
                    norm_sum[l_idx][e_idx] += float(norms_e.sum())
                    cond_err_count[l_idx][e_idx] += m

                    # Conditional replacement error vs every other activated expert
                    # Only meaningful when both e and j were activated for the same token.
                    # We approximate by computing pairwise error across activated experts
                    # in the same chunk (note: these are different tokens; a stricter
                    # version would match token-by-token, but is much more expensive).
                    # For the strict per-token version, see the note below.
                    for j_idx, out_j in outs.items():
                        if j_idx == e_idx or out_j.size(0) == 0:
                            continue
                        min_m = min(out_e.size(0), out_j.size(0))
                        diff  = (out_e[:min_m] - out_j[:min_m]).norm(dim=-1).numpy()
                        norm  = out_e[:min_m].norm(dim=-1).numpy().clip(min=1e-8)
                        cond_err_sum[l_idx][e_idx, j_idx] += float(
                            (diff / norm).sum())

    for h in hook_handles:
        h.remove()

    # ── Build tables ──────────────────────────────────────────────────────
    print("  Building similarity and auxiliary tables ...")

    # 1. Cosine similarity of mean outputs (original)
    sim_table = torch.zeros(L, num_experts, num_experts, dtype=torch.float16)
    # 2. Error table D (for Alg4)
    err_table = torch.zeros(L, num_experts, num_experts, dtype=torch.float32)
    # 3. Co-activation table (BuddyMoE-style)
    coact_table = torch.zeros(L, num_experts, num_experts, dtype=torch.float32)
    # 4. Conditional error similarity: 1 - D_cond(e,j), higher = better substitute
    cond_err_sim = torch.zeros(L, num_experts, num_experts, dtype=torch.float32)
    # 5. Router logit correlation
    router_corr = torch.zeros(L, num_experts, num_experts, dtype=torch.float32)
    # 6. Mean output norm per expert
    expert_norm = torch.zeros(L, num_experts, dtype=torch.float32)

    for l_idx in range(L):
        # ─ Cosine similarity of mean outputs ─
        mean_out = sum_out[l_idx].clone()
        for e in range(num_experts):
            if count_out[l_idx][e] > 0:
                mean_out[e] /= count_out[l_idx][e]
        norms = mean_out.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        unit_out = mean_out / norms
        S_cos = (unit_out @ unit_out.T).float().clamp(-1.0, 1.0)
        sim_table[l_idx] = S_cos.half()

        # ─ Error table D (mean output L2 diff, unnormalised) ─
        diff = unit_out.unsqueeze(0) - unit_out.unsqueeze(1)
        err_table[l_idx] = diff.norm(dim=-1).float()

        # ─ Expert output norm ─
        for e in range(num_experts):
            n = cond_err_count[l_idx][e]
            if n > 0:
                expert_norm[l_idx, e] = float(norm_sum[l_idx][e] / n)

        # ─ Co-activation frequency P(j∈S | e∈S) ─
        act_e = act_count[l_idx]  # [N]
        for e in range(num_experts):
            if act_e[e] > 0:
                coact_table[l_idx, e, :] = torch.tensor(
                    coact_count[l_idx][e] / act_e[e], dtype=torch.float32)
        coact_table[l_idx].fill_diagonal_(1.0)

        # ─ Conditional replacement error similarity ─
        # Never-observed pairs (cond_err_sum=0) would give exp(-0)=1.0, which is
        # wrong (perfect substitute) and causes ppl explosion when the pair is bad.
        # Fix: fill unobserved pairs with the cosine similarity as a conservative prior.
        cos_row = sim_table[l_idx].float()  # [N, N] — float16 → float32 for math
        for e in range(num_experts):
            n = cond_err_count[l_idx][e]
            if n > 0:
                row = cond_err_sum[l_idx][e] / n   # mean relative error per pair
                # exp(-error): 0 error → 1.0, large error → 0.0
                sim_vals = np.exp(-row)
                # For pairs with zero observations (never co-activated in calib set),
                # fall back to cosine sim (scaled to [0,1]) instead of 1.0.
                unobserved = (cond_err_sum[l_idx][e] == 0.0) & (np.arange(num_experts) != e)
                cos_prior = ((cos_row[e].numpy() + 1.0) / 2.0).clip(0.0, 1.0)
                sim_vals[unobserved] = cos_prior[unobserved]
                cond_err_sim[l_idx, e, :] = torch.tensor(sim_vals, dtype=torch.float32)
            else:
                # Expert never activated on calibration set → use cosine prior
                cos_prior = ((cos_row[e].numpy() + 1.0) / 2.0).clip(0.0, 1.0)
                cond_err_sim[l_idx, e, :] = torch.tensor(cos_prior, dtype=torch.float32)
        cond_err_sim[l_idx].fill_diagonal_(1.0)

        # ─ Router logit Pearson correlation ─
        n = gate_n[l_idx]
        if n > 1:
            mu  = gate_sum[l_idx]  / n                     # [N]
            mu2 = gate_sum2[l_idx] / n                     # [N]
            std = np.sqrt((mu2 - mu**2).clip(min=1e-12))   # [N]
            # Pearson: (E[g_e g_j] - mu_e * mu_j) / (std_e * std_j)
            cross = gate_cross[l_idx] / n                  # [N, N]
            cov   = cross - np.outer(mu, mu)
            corr  = cov / np.outer(std, std).clip(min=1e-12)
            corr  = np.clip(corr, -1.0, 1.0)
            router_corr[l_idx] = torch.tensor(corr, dtype=torch.float32)
        else:
            router_corr[l_idx] = torch.eye(num_experts)

    # ── Layer sensitivity ─────────────────────────────────────────────────
    total_calls = max(len(calib_chunks), 1)
    weight_var /= total_calls
    wv_min, wv_max = weight_var.min(), weight_var.max()
    if wv_max > wv_min:
        sens_np = (weight_var - wv_min) / (wv_max - wv_min)
    else:
        sens_np = np.ones(L, dtype=np.float32) * 0.5
    sens_np = 1.0 - sens_np
    sensitivity = torch.tensor(sens_np, dtype=torch.float32)

    # ── Activation frequency ──────────────────────────────────────────────
    total_act = freq.sum(axis=1, keepdims=True).clip(min=1)
    freq_norm = freq / total_act

    # ── Skip error D(i,∅) = expert_output_norm / mean_hidden_norm ─────────
    skip_error_table = torch.zeros(L, num_experts, dtype=torch.float32)
    routing_mass_t   = torch.zeros(L, num_experts, dtype=torch.float32)
    for l_idx in range(L):
        mean_h_norm = float(hidden_norm_sum[l_idx] / max(hidden_norm_count[l_idx], 1))
        for e in range(num_experts):
            n = cond_err_count[l_idx][e]
            if n > 0:
                mean_e_norm = float(norm_sum[l_idx][e] / n)
                skip_error_table[l_idx, e] = mean_e_norm / max(mean_h_norm, 1e-8)
            n_act = act_count[l_idx, e]
            if n_act > 0:
                routing_mass_t[l_idx, e] = float(
                    routing_weight_sum[l_idx, e] / n_act)

    # ── Combined similarity S = λ_out·cos + λ_coact·coact + λ_route·corr ─
    lam_out, lam_coact, lam_route = 0.5, 0.3, 0.2
    combined_sim = (
        lam_out   * sim_table.float() +
        lam_coact * coact_table +
        lam_route * router_corr
    ).clamp(-1.0, 1.0)

    # ── Critical expert score ─────────────────────────────────────────────
    # crit_l(i) = a·freq + b·mass + c·(1 - max_j≠i S(i,j)) + d·ω_l
    # Normalize each component to [0,1] per layer before combining.
    a_w, b_w, c_w, d_w = 0.25, 0.25, 0.25, 0.25
    critical_score_t = torch.zeros(L, num_experts, dtype=torch.float32)
    freq_t = torch.tensor(freq, dtype=torch.float32)  # [L, N]
    for l_idx in range(L):
        freq_row  = freq_t[l_idx]                              # [N]
        mass_row  = routing_mass_t[l_idx]                      # [N]
        S_row     = combined_sim[l_idx].clone()                # [N, N]
        S_row.fill_diagonal_(-1.0)
        non_sub   = 1.0 - S_row.max(dim=-1).values.clamp(0, 1) # [N]
        omega     = float(sensitivity[l_idx].item())

        def _norm01(t: torch.Tensor) -> torch.Tensor:
            lo, hi = t.min(), t.max()
            return (t - lo) / (hi - lo + 1e-8)

        crit = (a_w * _norm01(freq_row)
              + b_w * _norm01(mass_row)
              + c_w * _norm01(non_sub)
              + d_w * omega)
        critical_score_t[l_idx] = crit.clamp(0.0, 1.0)

    print(f"  Calibration complete.  "
          f"CosineSim off-diag mean: {sim_table.float().mean():.3f}  "
          f"CoAct off-diag mean: {coact_table.mean():.3f}  "
          f"CritScore mean: {critical_score_t.mean():.3f}")

    return CalibrationData(
        similarity_table     = sim_table,
        error_table          = err_table,
        sensitivity          = sensitivity,
        activation_freq      = freq_norm,
        coact_table          = coact_table,
        cond_error_sim_table = cond_err_sim,
        router_corr_table    = router_corr,
        expert_output_norm   = expert_norm,
        skip_error_table     = skip_error_table,
        routing_mass         = routing_mass_t,
        critical_score       = critical_score_t,
        combined_sim_table   = combined_sim,
        num_moe_layers       = L,
        num_experts          = num_experts,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 – Rerouting wrappers
# ─────────────────────────────────────────────────────────────────────────────

class ReroutingStats:
    """Accumulated statistics from one forward pass."""
    def __init__(self):
        self.hit_counts: List[int] = []       # per layer, per forward
        self.miss_counts: List[int] = []
        self.skip_counts: List[int] = []
        self.sub_sim_scores: List[float] = [] # similarity of chosen substitute
        self.cum_risk: List[float] = []       # Algorithm 4 budget
        self.early_stop: bool = False         # Algorithm 4 termination flag
        self.early_stop_layer: int = -1


class BaseReroutingWrapper(nn.Module):
    """
    Wraps a single Qwen-style MoE block and applies a rerouting strategy.

    Subclasses override _reroute(logits_raw, router_weights, router_indices,
                                 cached_mask, layer_idx) and return
    (final_weights [T,k], final_indices [T,k]).
    """

    def __init__(self, original_block, cache: SimulatedCache,
                 layer_idx: int, calib: CalibrationData):
        super().__init__()
        self.orig     = original_block
        self.cache    = cache
        self.l        = layer_idx
        self.calib    = calib
        self.top_k    = getattr(original_block, "top_k", None)
        self.num_exp  = getattr(original_block, "num_experts", None)
        self.norm_topk = getattr(original_block, "norm_topk_prob", False)
        self.stats    = ReroutingStats()

    # ── Forward (Qwen-style) ─────────────────────────────────────────────
    def forward(self, hidden_states: torch.Tensor):
        B, T, D = hidden_states.shape
        h_flat = hidden_states.view(-1, D)
        device  = h_flat.device

        # 1. Router
        logits_raw = self.orig.gate(h_flat)              # [T, N]
        probs      = F.softmax(logits_raw, dim=-1)
        router_weights, router_indices = probs.topk(self.top_k, dim=-1)

        # 2. Rerouting strategy
        cached_mask = self.cache.cached_mask(self.l, device)
        final_weights, final_indices = self._reroute(
            logits_raw, router_weights, router_indices, cached_mask)

        # 2b. Zero-weight fallback: if every slot for a token is zero (all experts
        #     skipped and no substitute found), the MoE output would be 0 for that
        #     token.  Over 48 layers this accumulates into hidden-state divergence and
        #     eventual float16 overflow → NaN.  Assign the fallback to the first
        #     cached expert with weight=1.0 so the forward stays numerically stable.
        w_total = final_weights.sum(dim=-1)          # [T]
        zero_rows = (w_total == 0)
        if zero_rows.any():
            cached_indices = cached_mask.nonzero(as_tuple=True)[0]
            if cached_indices.numel() > 0:
                fallback = int(cached_indices[0].item())
                final_indices[zero_rows, 0] = fallback
                final_weights[zero_rows, 0] = 1.0

        # 3. Renormalize if model requires it
        w_sum = final_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        final_weights = final_weights / w_sum
        final_weights = final_weights.to(hidden_states.dtype)

        # 4. Expert computation (standard scatter-gather)
        out = torch.zeros(B * T, D, dtype=hidden_states.dtype, device=device)
        if final_indices.numel() > 0:
            current_k = final_indices.shape[1]
            for slot in range(current_k):
                slot_exp = final_indices[:, slot]      # [T]
                slot_w   = final_weights[:, slot]      # [T]
                for e_idx in slot_exp.unique():
                    e_idx = int(e_idx.item())
                    if e_idx < 0 or e_idx >= self.num_exp:
                        continue
                    mask = (slot_exp == e_idx)
                    if not mask.any():
                        continue
                    h_in  = h_flat[mask]               # [m, D]
                    w_in  = slot_w[mask]               # [m]
                    e_out = self.orig.experts[e_idx](h_in)
                    if isinstance(e_out, tuple):
                        e_out = e_out[0]
                    out[mask] += (e_out * w_in.unsqueeze(-1)).to(hidden_states.dtype)

        # 5. Shared expert (Qwen-style)
        if hasattr(self.orig, "shared_expert") and self.orig.shared_expert is not None:
            shared = self.orig.shared_expert(h_flat)
            if hasattr(self.orig, "shared_expert_gate"):
                gate = torch.sigmoid(self.orig.shared_expert_gate(h_flat))
                shared = gate * shared
            out = out + shared.to(out.dtype)

        return out.view(B, T, D), logits_raw

    def _reroute(self, logits_raw, router_weights, router_indices, cached_mask
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


# ── Algorithm 1 family ───────────────────────────────────────────────────────
#
# All Alg1 variants share the same structure:
#   1. Build a similarity LUT from one of four tables.
#   2. For each miss expert, look up the best cached substitute.
#   3. Skip decision uses contribution-based criterion (literature-aligned).
#   4. Weight handling is "substitute keeps original weight + renormalize".
#
# Skip design rationale (from Deja Vu / structured-pruning literature):
#   The contribution of expert e to the MoE output is proportional to
#   w_e * ‖E_e(h)‖₂.  We proxy ‖E_e(h)‖₂ with the calibrated mean norm
#   ‖Ē_e‖.  We skip expert e when:
#       w_e * ‖Ē_e‖ < θ_contrib * (1/k * Σ_{i∈S} w_i * ‖Ē_i‖)
#   i.e., when the expected contribution is < θ_contrib × the per-expert
#   average contribution.  This is input-adaptive and norm-aware.
#
# For misses that exceed the contribution threshold, we ALWAYS substitute
# (never hard-skip based on substitute quality), because:
#   D_cond(e,j) < 1  →  substitution error < skip error (mathematically)
# for any substitute with cosine similarity > 0.5 with the target.
# Rather than a binary threshold, Alg3's similarity-weighted merge handles
# the "quality" dimension continuously.


class Alg1_Base(BaseReroutingWrapper):
    """
    Common base for all Alg1 similarity-table variants.

    Subclasses set self.sim_key to pick which CalibrationData table to use.
    Skip uses contribution-based criterion; substitute is always preferred
    over skip when contribution exceeds threshold.
    """

    # Subclasses set this to one of:
    #   "cosine"   → calib.similarity_table
    #   "coact"    → calib.coact_table
    #   "cond_err" → calib.cond_error_sim_table
    #   "router"   → calib.router_corr_table
    sim_key: str = "cosine"

    def __init__(self, orig, cache, layer_idx, calib,
                 theta_contrib: float = 0.10):
        """
        theta_contrib: skip an expert whose contribution is < theta_contrib ×
                       mean per-expert contribution in the current token.
                       0.10 means "skip if less than 10% of the average expert".
        """
        super().__init__(orig, cache, layer_idx, calib)
        self.theta_contrib = theta_contrib

    def _get_sim_table(self, device) -> torch.Tensor:
        """Return the [N, N] similarity matrix for this layer."""
        l = self.l
        if self.sim_key == "cosine" and self.calib.similarity_table is not None:
            return self.calib.similarity_table[l].float().to(device)
        if self.sim_key == "coact" and self.calib.coact_table is not None:
            return self.calib.coact_table[l].float().to(device)
        if self.sim_key == "cond_err" and self.calib.cond_error_sim_table is not None:
            return self.calib.cond_error_sim_table[l].float().to(device)
        if self.sim_key == "router" and self.calib.router_corr_table is not None:
            return self.calib.router_corr_table[l].float().to(device)
        # Fallback: identity (self-substitution)
        return torch.eye(self.num_exp, device=device)

    def _contribution_skip_mask(self, router_weights: torch.Tensor,
                                 cached_mask: torch.Tensor) -> torch.Tensor:
        """
        Return [T, k] bool mask: True where the expert should be skipped.

        Criterion: w_e * ‖Ē_e‖ < theta_contrib * mean_contribution_this_token
        where mean_contribution = (1/k) * Σ_i w_i * ‖Ē_i‖.

        If expert_output_norm is unavailable, falls back to w_e < theta/k.
        """
        T, k = router_weights.shape
        device = router_weights.device

        if self.calib.expert_output_norm is not None:
            norm_vec = self.calib.expert_output_norm[self.l].to(device)  # [N]
        else:
            # Fallback: treat all norms as 1 → criterion reduces to w_e < θ/k
            norm_vec = torch.ones(self.num_exp, device=device)

        # Expected contribution per slot: w_e * ‖Ē_e‖
        exp_contrib = router_weights * norm_vec[self.orig_indices_buffer]  # [T, k]
        mean_contrib = exp_contrib.mean(dim=-1, keepdim=True)              # [T, 1]
        skip_mask = exp_contrib < self.theta_contrib * mean_contrib        # [T, k]
        return skip_mask

    # We need the router indices before calling _contribution_skip_mask,
    # so we store them temporarily between _reroute being called.
    orig_indices_buffer: torch.Tensor = None

    def _reroute(self, logits_raw, router_weights, router_indices, cached_mask):
        T, k   = router_weights.shape
        N      = self.num_exp
        device = logits_raw.device

        self.orig_indices_buffer = router_indices  # for contribution skip

        # ── Similarity LUT: best cached substitute per expert ──────────────
        S = self._get_sim_table(device)                         # [N, N]
        S_masked = S.masked_fill(~cached_mask.unsqueeze(0), -1e9)
        S_masked.fill_diagonal_(-1e9)
        best_sim, best_sub = S_masked.max(dim=-1)               # [N], [N]
        # Cached experts: self-mapping
        cached_indices = torch.where(cached_mask)[0]
        best_sub[cached_mask] = torch.arange(N, device=device)[cached_mask]
        best_sim[cached_mask] = 1.0

        # ── Contribution-based skip ────────────────────────────────────────
        skip_mask = self._contribution_skip_mask(router_weights, cached_mask)
        hit_mask  = cached_mask[router_indices]                 # [T, k]

        final_indices = router_indices.clone()
        final_weights = router_weights.clone()

        # Substitute all misses (don't skip based on sim quality)
        sub_ok = (~hit_mask) & (~skip_mask)
        final_indices[sub_ok] = best_sub[router_indices[sub_ok]]

        # Zero weight for contribution-negligible experts (skip)
        skip_slot = (~hit_mask) & skip_mask
        final_weights[skip_slot] = 0.0

        # Stats
        self.stats.hit_counts.append(int(hit_mask.sum()))
        self.stats.miss_counts.append(int((~hit_mask).sum()))
        self.stats.skip_counts.append(int(skip_slot.sum()))
        self.stats.sub_sim_scores.append(
            float(best_sim[router_indices[sub_ok]].mean()) if sub_ok.any() else 1.0)

        return final_weights, final_indices


class Alg1_CosineOut(Alg1_Base):
    """Similarity table: cosine similarity of mean expert outputs."""
    sim_key = "cosine"


class Alg1_CoAct(Alg1_Base):
    """Similarity table: routing co-activation frequency P(j∈S | e∈S)."""
    sim_key = "coact"


class Alg1_CondError(Alg1_Base):
    """Similarity table: conditional replacement error similarity (1 - D_cond)."""
    sim_key = "cond_err"


class Alg1_RouterCorr(Alg1_Base):
    """Similarity table: router logit Pearson correlation."""
    sim_key = "router"


# ── Weight-correction ablation ────────────────────────────────────────────────
#
# Uses Alg1_CondError (best expected similarity table) with fixed substitute
# selection, and varies ONLY the weight treatment for the substitute slot.
# Ablation variants:
#   W0_NoCorr     : substitute inherits miss weight unchanged  (BuddyMoE style)
#   W1_SimScale   : substitute weight scaled by sim(e,j)       (Alg3 style)
#   W2_HitRenorm  : miss weights discarded; hits renormalized   (SkipAll style)
#   W3_UniformRedist: miss weight distributed uniformly over all hits
#   W4_TopHitRedist: miss weight assigned entirely to top-1 hit

class WeightCorrectionAblation(Alg1_Base):
    """
    Fixed CondError similarity table + contribution-based skip.
    Weight handling is the only variable.

    mode: "W0_NoCorr" | "W1_SimScale" | "W2_HitRenorm" |
          "W3_UniformRedist" | "W4_TopHitRedist"
    """
    sim_key = "cond_err"

    def __init__(self, orig, cache, layer_idx, calib,
                 mode: str = "W0_NoCorr", theta_contrib: float = 0.10):
        super().__init__(orig, cache, layer_idx, calib, theta_contrib)
        assert mode in ("W0_NoCorr", "W1_SimScale", "W2_HitRenorm",
                        "W3_UniformRedist", "W4_TopHitRedist")
        self.mode = mode

    def _reroute(self, logits_raw, router_weights, router_indices, cached_mask):
        T, k   = router_weights.shape
        N      = self.num_exp
        device = logits_raw.device
        self.orig_indices_buffer = router_indices

        # ── Similarity and LUT ─────────────────────────────────────────────
        S = self._get_sim_table(device)                     # [N, N]
        S_masked = S.masked_fill(~cached_mask.unsqueeze(0), -1e9)
        S_masked.fill_diagonal_(-1e9)
        best_sim, best_sub = S_masked.max(dim=-1)           # [N]
        best_sub[cached_mask] = torch.arange(N, device=device)[cached_mask]
        best_sim[cached_mask] = 1.0

        # ── Skip and hit masks ─────────────────────────────────────────────
        skip_mask = self._contribution_skip_mask(router_weights, cached_mask)
        hit_mask  = cached_mask[router_indices]

        # Determine substitute index for each slot
        sub_indices = best_sub[router_indices]              # [T, k]
        sub_sims    = best_sim[router_indices]              # [T, k]

        miss_sub_mask = (~hit_mask) & (~skip_mask)          # slots that substitute
        skip_slot     = (~hit_mask) & skip_mask             # slots that are skipped

        # ── Weight computation by mode ────────────────────────────────────
        final_indices = router_indices.clone()
        final_weights = router_weights.clone()
        w_dtype = final_weights.dtype  # float16 on GPU; keep all RHS in this dtype

        # sub_sims comes from the float32 similarity table; cast once here.
        sub_sims = sub_sims.to(w_dtype)

        # Apply substitution indices
        final_indices[miss_sub_mask] = sub_indices[miss_sub_mask]

        if self.mode == "W0_NoCorr":
            final_weights[skip_slot] = 0.0

        elif self.mode == "W1_SimScale":
            final_weights[miss_sub_mask] = (
                router_weights[miss_sub_mask] * sub_sims[miss_sub_mask].clamp(0, 1))
            final_weights[skip_slot] = 0.0

        elif self.mode == "W2_HitRenorm":
            final_weights[~hit_mask] = 0.0

        elif self.mode == "W3_UniformRedist":
            miss_w_total = router_weights.clone()
            miss_w_total[hit_mask]  = 0.0
            miss_w_total[skip_slot] = 0.0
            miss_w_sum = miss_w_total.sum(-1, keepdim=True)                    # [T, 1], w_dtype
            n_hits = hit_mask.float().to(w_dtype).sum(-1, keepdim=True).clamp(min=1)
            bonus  = miss_w_sum / n_hits                                        # w_dtype
            final_weights[~hit_mask] = 0.0
            final_weights[hit_mask]  = (router_weights + bonus.expand_as(router_weights))[hit_mask]

        elif self.mode == "W4_TopHitRedist":
            miss_w_total = router_weights.clone()
            miss_w_total[hit_mask]  = 0.0
            miss_w_total[skip_slot] = 0.0
            miss_w_sum = miss_w_total.sum(-1)                                   # [T], w_dtype
            hit_weights_for_max = router_weights.masked_fill(~hit_mask, -1e4)
            top1_hit = hit_weights_for_max.argmax(dim=-1)                       # [T]
            final_weights[~hit_mask] = 0.0
            for t in range(T):
                final_weights[t, top1_hit[t]] = (
                    router_weights[t, top1_hit[t]] + miss_w_sum[t])

        self.stats.hit_counts.append(int(hit_mask.sum()))
        self.stats.miss_counts.append(int((~hit_mask).sum()))
        self.stats.skip_counts.append(int(skip_slot.sum()))
        return final_weights, final_indices


# ── Algorithm 2 ─────────────────────────────────────────────────────────────

class Alg2_EntropyBias(BaseReroutingWrapper):
    """
    Entropy-Conditioned Pre-Routing Logit Bias.

    Adds gamma(tau) to logits of cached experts before top-k selection.
    Gamma is interpolated based on token routing entropy: high entropy → strong bias.
    Top-1 protection: if original top-1 is uncached, reduce gamma until it is included.
    Weights computed from ORIGINAL logits (not biased), so selected experts carry
    the model's true preference over the selected set.
    """

    def __init__(self, orig, cache, layer_idx, calib,
                 gamma_high: float = 4.0,
                 gamma_low:  float = 0.5,
                 tau_low_pct:  float = 0.25,   # bottom entropy percentile
                 tau_high_pct: float = 0.75):  # top entropy percentile
        super().__init__(orig, cache, layer_idx, calib)
        self.gamma_high   = gamma_high
        self.gamma_low    = gamma_low
        self.tau_low_pct  = tau_low_pct
        self.tau_high_pct = tau_high_pct
        # Calibration: these will be set from calibration pass
        self._tau_low  = None
        self._tau_high = None

    def set_entropy_thresholds(self, tau_low: float, tau_high: float):
        self._tau_low  = tau_low
        self._tau_high = tau_high

    def _compute_gamma(self, entropy: torch.Tensor) -> torch.Tensor:
        """[T] float — adaptive gamma per token."""
        tau_low  = self._tau_low  or math.log(self.num_exp) * self.tau_low_pct
        tau_high = self._tau_high or math.log(self.num_exp) * self.tau_high_pct
        t = (entropy - tau_low) / max(tau_high - tau_low, 1e-6)
        t = t.clamp(0.0, 1.0)
        return self.gamma_low + t * (self.gamma_high - self.gamma_low)  # [T]

    def _reroute(self, logits_raw, router_weights, router_indices, cached_mask):
        T, k = router_weights.shape
        device = logits_raw.device

        # Token routing entropy
        probs   = F.softmax(logits_raw.float(), dim=-1)   # [T, N]
        entropy = -(probs * (probs + 1e-9).log()).sum(-1)  # [T]

        gamma = self._compute_gamma(entropy)               # [T]

        # Build biased logits: add gamma[t] to each cached expert
        bias = (cached_mask.float() * gamma.unsqueeze(-1))  # [T, N]
        biased_logits = logits_raw.float() + bias

        # Top-k on biased logits
        _, new_indices = biased_logits.topk(k, dim=-1)     # [T, k]

        # Top-1 protection: if original top-1 falls out, shrink gamma
        orig_top1 = router_indices[:, 0]                   # [T]
        for t in range(T):
            if not cached_mask[orig_top1[t]] and orig_top1[t] not in new_indices[t]:
                # Force include original top-1 by replacing lowest-priority slot
                new_indices[t, -1] = orig_top1[t]

        # Weights from ORIGINAL logits over selected set (not biased)
        orig_selected_logits = logits_raw.gather(-1, new_indices)  # [T, k]
        new_weights = F.softmax(orig_selected_logits.float(), dim=-1)  # [T, k]

        # Stats
        hit_mask = cached_mask[new_indices]
        self.stats.hit_counts.append(int(hit_mask.sum()))
        self.stats.miss_counts.append(int((~hit_mask).sum()))

        return new_weights.to(router_weights.dtype), new_indices


# ── Algorithm 3 ─────────────────────────────────────────────────────────────

class Alg3_RouterScoreMerge(BaseReroutingWrapper):
    """
    Router-Score Guided Substitution with Similarity-Weighted Weight Merge.

    For each miss, selects the substitute by maximizing a joint score:
        score(j) = lambda * z_norm[j] + (1-lambda) * S[e, j]
    where z_norm is the z-scored router logit among cached experts (online,
    input-conditioned) and S is the offline output similarity.

    Weights: similarity-scaled merge → w_j += w_e * S(e, j).
    This bounds the error amplification when substitutes are poor.
    """

    def __init__(self, orig, cache, layer_idx, calib, lam: float = 0.5):
        super().__init__(orig, cache, layer_idx, calib)
        self.lam = lam

    def _reroute(self, logits_raw, router_weights, router_indices, cached_mask):
        T, k   = router_weights.shape
        N      = self.num_exp
        device = logits_raw.device
        lam    = self.lam

        # ── Online signal: z-scored cached logits ─────────────────────────
        z = logits_raw.float()                           # [T, N]
        n_cached = int(cached_mask.sum().item())
        if n_cached == 0:
            # Degenerate: no cached experts at all — fall back to skip-all behaviour.
            slot_weights = torch.zeros(T, N, device=device, dtype=torch.float32)
            final_weights, final_indices = slot_weights.topk(k, dim=-1)
            self.stats.hit_counts.append(0)
            self.stats.miss_counts.append(T * k)
            self.stats.sub_sim_scores.append(0.0)
            return final_weights.to(router_weights.dtype), final_indices

        z_cached = z.masked_fill(~cached_mask.unsqueeze(0), float("nan"))

        # nanmean: safe when n_cached >= 1; nan_to_num guards the degenerate edge.
        z_mean = torch.nan_to_num(
            torch.nanmean(z_cached, dim=-1, keepdim=True),
            nan=0.0,
        )                                                # [T, 1]

        finite = ~torch.isnan(z_cached)                  # [T, N]
        cnt    = finite.float().sum(-1, keepdim=True).clamp(min=1)
        diff2  = ((z_cached - z_mean) ** 2).masked_fill(~finite, 0.0)
        z_std  = torch.nan_to_num(
            (diff2.sum(-1, keepdim=True) / cnt).sqrt(),
            nan=0.0,
        ).clamp(min=1e-6)                                # [T, 1]

        z_norm = ((z - z_mean) / z_std).masked_fill(~cached_mask.unsqueeze(0), -1e9)
        # [T, N] — high values = cached experts this token prefers

        # ── Offline signal: similarity table ──────────────────────────────
        if self.calib.similarity_table is not None:
            S = self.calib.similarity_table[self.l].float().to(device)  # [N, N]
        else:
            S = torch.eye(N, device=device)
        S_masked = S.masked_fill(~cached_mask.unsqueeze(0), -1e9)  # [N, N]

        # ── Joint score for all (miss, cached) pairs ───────────────────────
        # S_lookup[t, r, j] = S[router_indices[t,r], j]
        S_lookup = S_masked[router_indices]              # [T, k, N]

        # z_norm_exp[t, r, j] = z_norm[t, j]
        z_norm_exp = z_norm.unsqueeze(1).expand(-1, k, -1)  # [T, k, N]

        joint = lam * z_norm_exp + (1 - lam) * S_lookup   # [T, k, N]
        joint = joint.masked_fill(~cached_mask.unsqueeze(0).unsqueeze(0), -1e9)

        best_sub = joint.argmax(dim=-1)                   # [T, k]
        best_sim = S[router_indices, best_sub]             # [T, k] — offline sim

        hit_mask = cached_mask[router_indices]             # [T, k]

        # ── Similarity-weighted weight accumulation ────────────────────────
        # For hits: full weight stays on hit expert
        # For misses: weight * sim(e, sub) goes to sub; remainder dropped
        slot_weights = torch.zeros(T, N, device=device, dtype=torch.float32)

        # Hit contributions
        hit_w = router_weights.float() * hit_mask.float()
        slot_weights.scatter_add_(1, router_indices, hit_w)

        # Miss contributions (similarity-scaled)
        miss_mask = ~hit_mask
        miss_w    = router_weights.float() * miss_mask.float()
        scaled_w  = miss_w * best_sim.clamp(0.0, 1.0)          # [T, k]
        slot_weights.scatter_add_(1, best_sub, scaled_w)

        # Keep only cached slots
        slot_weights = slot_weights * cached_mask.float().unsqueeze(0)

        # Convert back to top-k format: select top-k from slot_weights
        final_weights, final_indices = slot_weights.topk(k, dim=-1)

        self.stats.hit_counts.append(int(hit_mask.sum()))
        self.stats.miss_counts.append(int(miss_mask.sum()))
        self.stats.sub_sim_scores.append(
            float(best_sim[miss_mask].mean()) if miss_mask.any() else 1.0)

        return final_weights.to(router_weights.dtype), final_indices


# ── Algorithm 4 ─────────────────────────────────────────────────────────────

class Alg4_ErrorBudget(BaseReroutingWrapper):
    """
    Error-Budget-Controlled Adaptive Draft with Early Termination.

    Maintains a cumulative risk R = Σ_{l} D(e→j) * ω_l accumulated across all
    layers of ONE draft forward step.  Substitute selection is argmin_{j∈C} D(e,j)*ω_l.

    Design rule (matching nano-vllm-moe integration contract):
      • _reroute() ONLY accumulates risk and selects substitutes.
        It NEVER sets early_stop or terminates mid-forward.
        Every layer executes to completion regardless of budget.
      • After the full model forward returns, call post_forward_check().
        This compares cumulative_risk to budget and sets should_terminate.
      • SpeculativeEngine (or the eval loop) checks should_terminate AFTER
        run_draft() returns, then decides whether to launch another draft step.
    """

    def __init__(self, orig, cache, layer_idx, calib,
                 budget: float = 0.5,
                 theta_skip: float = 0.03):
        super().__init__(orig, cache, layer_idx, calib)
        self.budget     = budget
        self.theta_skip = theta_skip
        self.cumulative_risk: float = 0.0
        # Set only by post_forward_check(), never during _reroute().
        self._terminate_after_step: bool = False

    def reset_risk(self):
        """Reset before each draft step (i.e. before each model forward)."""
        self.cumulative_risk = 0.0
        self._terminate_after_step = False
        self.stats.early_stop = False
        self.stats.early_stop_layer = -1

    @property
    def should_terminate(self) -> bool:
        """True iff post_forward_check() determined budget was exceeded.
        Only valid AFTER post_forward_check() has been called."""
        return self._terminate_after_step

    def post_forward_check(self):
        """Call ONCE after the full model forward completes.
        Checks accumulated risk and signals whether the next draft step
        should be launched.  Never called from inside _reroute.
        """
        if self.cumulative_risk > self.budget and not self._terminate_after_step:
            self._terminate_after_step = True
            self.stats.early_stop = True
            self.stats.early_stop_layer = self.l

    def _reroute(self, logits_raw, router_weights, router_indices, cached_mask):
        T, k   = router_weights.shape
        N      = self.num_exp
        device = logits_raw.device

        # Sensitivity for this layer
        omega = float(self.calib.sensitivity[self.l].item()) \
            if self.calib.sensitivity is not None else 0.5

        # Error table for this layer [N, N]
        if self.calib.error_table is not None:
            D = self.calib.error_table[self.l].to(device)  # [N, N]
        else:
            D = torch.ones(N, N, device=device)

        hit_mask  = cached_mask[router_indices]       # [T, k]
        skip_mask = router_weights < self.theta_skip  # [T, k]

        final_indices = router_indices.clone()
        final_weights = router_weights.clone()

        # For each miss, find lowest-error cached substitute
        D_cached = D.masked_fill(~cached_mask.unsqueeze(0), 1e9)  # [N, N]
        best_sub_err, best_sub_idx = D_cached.min(dim=-1)          # [N]

        step_risk = 0.0
        for t in range(T):
            for r in range(k):
                e = int(router_indices[t, r].item())
                w = float(router_weights[t, r].item())
                if hit_mask[t, r]:
                    continue
                if skip_mask[t, r]:
                    final_weights[t, r] = 0.0
                    continue
                j_star = int(best_sub_idx[e].item())
                risk   = float(best_sub_err[e].item()) * omega * w
                step_risk += risk
                final_indices[t, r] = j_star

        # Accumulate risk across layers.
        # NOTE: Do NOT check budget or set early_stop here.
        # The forward must complete all layers; termination is a between-step decision.
        self.cumulative_risk += step_risk
        self.stats.cum_risk.append(self.cumulative_risk)

        self.stats.hit_counts.append(int(hit_mask.sum()))
        self.stats.miss_counts.append(int((~hit_mask).sum()))

        return final_weights, final_indices


# ── Algorithm 5 ─────────────────────────────────────────────────────────────

class BanditState:
    """
    Shared online bandit state across layers.
    alpha_hat[L, N, N, B] — EMA acceptance rate indexed by (layer, miss, sub, entropy_bin).
    n_obs  [L, N, N, B]   — observation counts.
    """

    def __init__(self, num_layers: int, num_experts: int, num_bins: int = 3,
                 eta: float = 0.05, c: float = 0.3,
                 sim_prior: Optional[torch.Tensor] = None):
        self.eta = eta
        self.c   = c
        self.num_bins = num_bins
        # Initialize from similarity prior
        if sim_prior is not None:
            # [L, N, N] → map to [0.5, 1.0] for alpha prior
            alpha_init = (1.0 + sim_prior.float()) / 2.0  # [L, N, N]
            self.alpha = alpha_init.unsqueeze(-1).expand(
                -1, -1, -1, num_bins).clone()             # [L, N, N, B]
        else:
            self.alpha = torch.full((num_layers, num_experts, num_experts, num_bins),
                                    0.7)
        self.n_obs  = torch.ones_like(self.alpha)

    def select(self, l: int, e_miss: int, cached_set: List[int],
               entropy_bin: int) -> int:
        """UCB selection: argmax_{j in cached} alpha + c * sqrt(log(N)/n)."""
        b = entropy_bin
        scores = self.alpha[l, e_miss, :, b].clone()            # [N]
        n_total = self.n_obs[l, e_miss, :, b].sum().log()
        ucb     = (n_total / self.n_obs[l, e_miss, :, b].clamp(min=1)).sqrt()
        scores  = scores + self.c * ucb

        # Restrict to cached set
        mask = torch.full((scores.size(0),), False)
        for j in cached_set:
            mask[j] = True
        scores[~mask] = -1e9

        return int(scores.argmax().item())

    def update(self, l: int, e_miss: int, e_sub: int, entropy_bin: int,
               accepted: bool):
        b  = entropy_bin
        r  = 1.0 if accepted else 0.0
        self.alpha[l, e_miss, e_sub, b] = (
            (1 - self.eta) * self.alpha[l, e_miss, e_sub, b] + self.eta * r)
        self.n_obs[l, e_miss, e_sub, b] += 1


class Alg5_OnlineBandit(BaseReroutingWrapper):
    """
    Contextual Online Bandit with Similarity Prior and Entropy Binning.

    Maintains per-layer EMA acceptance rates α̂[e_miss, e_sub, entropy_bin].
    Initialized from offline similarity prior (cold start ≈ Algorithm 2).
    UCB exploration over cached expert set.
    Updated via accept/reject signal computed from logit TV distance.
    """

    def __init__(self, orig, cache, layer_idx, calib,
                 bandit_state: BanditState,
                 tau_low: float = 0.5, tau_high: float = 1.5,
                 theta_skip: float = 0.03):
        super().__init__(orig, cache, layer_idx, calib)
        self.bandit    = bandit_state
        self.tau_low   = tau_low
        self.tau_high  = tau_high
        self.theta_skip = theta_skip
        # Buffer for deferred updates
        self._pending: List[Tuple[int, int, int, int]] = []  # (l, e_miss, e_sub, bin)

    def _entropy_bin(self, entropy: torch.Tensor) -> torch.Tensor:
        """Map scalar entropy to bin index {0, 1, 2}."""
        b = torch.zeros_like(entropy, dtype=torch.long)
        b[entropy > self.tau_high] = 2
        b[(entropy >= self.tau_low) & (entropy <= self.tau_high)] = 1
        return b

    def _reroute(self, logits_raw, router_weights, router_indices, cached_mask):
        T, k   = router_weights.shape
        device = logits_raw.device

        probs   = F.softmax(logits_raw.float(), dim=-1)
        entropy = -(probs * (probs + 1e-9).log()).sum(-1)     # [T]
        e_bins  = self._entropy_bin(entropy)                  # [T]

        hit_mask  = cached_mask[router_indices]               # [T, k]
        skip_mask = router_weights < self.theta_skip

        cached_list = self.cache.cached_list(self.l)
        final_indices = router_indices.clone()
        final_weights = router_weights.clone()
        self._pending.clear()

        for t in range(T):
            ebin = int(e_bins[t].item())
            for r in range(k):
                e = int(router_indices[t, r].item())
                w = float(router_weights[t, r].item())
                if hit_mask[t, r]:
                    continue
                if skip_mask[t, r]:
                    final_weights[t, r] = 0.0
                    continue
                j_star = self.bandit.select(self.l, e, cached_list, ebin)
                final_indices[t, r] = j_star
                self._pending.append((self.l, e, j_star, ebin))

        self.stats.hit_counts.append(int(hit_mask.sum()))
        self.stats.miss_counts.append(int((~hit_mask).sum()))
        return final_weights, final_indices

    def bandit_update(self, accepted: bool):
        """Call after verification to update bandit state."""
        for (l, e_miss, e_sub, ebin) in self._pending:
            self.bandit.update(l, e_miss, e_sub, ebin, accepted)


# ── Baselines ────────────────────────────────────────────────────────────────

class Baseline_SkipAll(BaseReroutingWrapper):
    """Zero weight for all missed experts; renormalize over hits."""

    def _reroute(self, logits_raw, router_weights, router_indices, cached_mask):
        hit_mask = cached_mask[router_indices]
        final_weights = router_weights * hit_mask.float()
        return final_weights, router_indices


class Baseline_RoundRobin(BaseReroutingWrapper):
    """Substitute missed experts with cached experts in round-robin order."""

    def _reroute(self, logits_raw, router_weights, router_indices, cached_mask):
        cached_list = self.cache.cached_list(self.l)
        if not cached_list:
            return router_weights * 0.0, router_indices
        T, k = router_weights.shape
        hit_mask      = cached_mask[router_indices]
        final_indices = router_indices.clone()
        rr_idx        = 0
        for t in range(T):
            for r in range(k):
                if not hit_mask[t, r]:
                    final_indices[t, r] = cached_list[rr_idx % len(cached_list)]
                    rr_idx += 1
        return router_weights, final_indices


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 – Model patching / restoring
# ─────────────────────────────────────────────────────────────────────────────

def patch_model(model, moe_attr: str, wrappers: List[BaseReroutingWrapper],
                moe_layer_indices: List[int]):
    """Replace each MoE block with its wrapper."""
    for i, (global_idx, wrapper) in enumerate(zip(moe_layer_indices, wrappers)):
        setattr(model.model.layers[global_idx], moe_attr, wrapper)


def restore_model(model, moe_attr: str, originals: List[nn.Module],
                  moe_layer_indices: List[int]):
    """Restore original MoE blocks."""
    for global_idx, orig in zip(moe_layer_indices, originals):
        setattr(model.model.layers[global_idx], moe_attr, orig)


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 – Evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_acceptance_rate_and_ppl(
    model, chunks: List[torch.Tensor], device: str, max_chunks: int = None,
) -> Tuple[float, float]:
    """
    Run model on chunks; return (mean_alpha, ppl).

    mean_alpha = E_t [ 1 - TV(p_draft, p_target) ] is computed by comparing
    draft logits (current model state) to baseline logits (must be pre-computed
    and stored in the chunk dict, or this function just returns PPL only when
    called on the *draft* model without baseline logits).

    For differential evaluation, call this with a model that has rerouting
    wrappers installed, and separately with the original model.
    """
    total_nll = 0.0
    total_tokens = 0
    model.eval()
    for i, chunk in enumerate(chunks):
        if max_chunks and i >= max_chunks:
            break
        chunk = chunk.to(device)
        out   = model(chunk, labels=chunk)
        loss  = out.loss
        if torch.isfinite(loss):
            total_nll    += float(loss.item()) * (chunk.size(1) - 1)
            total_tokens += (chunk.size(1) - 1)

    ppl = math.exp(total_nll / max(total_tokens, 1))
    return ppl


@torch.no_grad()
def compute_alpha_from_logits(
    baseline_logits: torch.Tensor,
    draft_logits: torch.Tensor,
) -> Optional[float]:
    """
    Theoretical acceptance rate: alpha = 1 - TV(p_draft, p_target).
    = sum_v min(p_d(v), p_t(v))

    Both tensors: [T, vocab_size].
    Returns None if either tensor contains NaN/Inf — callers must skip NaN chunks.
    """
    if draft_logits.numel() == 0 or baseline_logits.numel() == 0:
        return None
    # Guard: any NaN/Inf in logits → the chunk produced a degenerate forward.
    # Do NOT average these in; return None so the caller can log and skip.
    if not torch.isfinite(draft_logits).all():
        return None
    if not torch.isfinite(baseline_logits).all():
        return None

    p_base  = F.softmax(baseline_logits.float(), dim=-1)
    p_draft = F.softmax(draft_logits.float(), dim=-1)
    alpha_per_token = torch.minimum(p_base, p_draft).sum(dim=-1)  # [T]
    result = float(alpha_per_token.mean().item())
    return result if math.isfinite(result) else None


@torch.no_grad()
def run_logit_comparison(
    model, orig_moe_blocks, wrappers, moe_layer_indices: List[int],
    moe_attr: str, chunks: List[torch.Tensor], device: str,
    max_chunks: int = None,
) -> Dict:
    """
    For each chunk: run baseline (exact), then draft (rerouted).
    Return aggregated metrics.

    Returns dict with keys:
      mean_alpha, std_alpha, ppl_baseline, ppl_draft,
      layer_cos_sim  [num_moe_layers]
    """
    L = len(moe_layer_indices)
    alphas: List[float] = []
    baseline_nlls: List[float] = []
    draft_nlls: List[float]    = []
    total_tokens_b: int = 0
    total_tokens_d: int = 0
    nan_chunk_count: int = 0  # chunks where draft produced NaN logits

    # For layer output similarity, hook both runs
    baseline_moe_outs: List[List[torch.Tensor]] = [[] for _ in range(L)]
    draft_moe_outs:    List[List[torch.Tensor]] = [[] for _ in range(L)]
    hook_handles = []

    def make_out_hook(storage: List):
        def hook(module, inp, out):
            if isinstance(out, (tuple, list)):
                storage.append(out[0].detach().float().cpu())
            else:
                storage.append(out.detach().float().cpu())
        return hook

    max_c = max_chunks or len(chunks)

    for ci, chunk in enumerate(tqdm(chunks[:max_c], desc="    Comparing", leave=False)):
        chunk = chunk.to(device)

        # ── Baseline run ──────────────────────────────────────────────────
        restore_model(model, moe_attr, orig_moe_blocks, moe_layer_indices)
        for l_i, global_idx in enumerate(moe_layer_indices):
            moe = getattr(model.model.layers[global_idx], moe_attr)
            h = moe.register_forward_hook(make_out_hook(baseline_moe_outs[l_i]))
            hook_handles.append(h)

        out_b = model(chunk, labels=chunk, output_hidden_states=False)
        logits_b = out_b.logits[:, :-1, :].reshape(-1, out_b.logits.size(-1))
        if torch.isfinite(out_b.loss):
            n_tok = chunk.size(1) - 1
            baseline_nlls.append(float(out_b.loss.item()) * n_tok)
            total_tokens_b += n_tok

        for h in hook_handles:
            h.remove()
        hook_handles.clear()

        # ── Draft run ─────────────────────────────────────────────────────
        patch_model(model, moe_attr, wrappers, moe_layer_indices)
        # Reset per-forward risk accumulators (Alg4, StaticEBB, OnlineAB, HybridCP)
        for w in wrappers:
            if isinstance(w, Alg4_ErrorBudget):
                w.reset_risk()
            if isinstance(w, StaticEBB):
                w.shared.reset()
            if isinstance(w, OnlineAB):
                w.shared.reset()
                w.ab_state._pending.clear()
            if isinstance(w, HybridCP):
                w.shared.reset()

        for l_i, global_idx in enumerate(moe_layer_indices):
            moe = getattr(model.model.layers[global_idx], moe_attr)
            h = moe.register_forward_hook(make_out_hook(draft_moe_outs[l_i]))
            hook_handles.append(h)

        out_d = model(chunk, labels=chunk, output_hidden_states=False)
        logits_d = out_d.logits[:, :-1, :].reshape(-1, out_d.logits.size(-1))

        # Post-forward termination check (Alg4): runs AFTER the full forward,
        # never mid-layer.  Sets should_terminate for the caller to query.
        for w in wrappers:
            if isinstance(w, Alg4_ErrorBudget):
                w.post_forward_check()

        # NaN logits mean the draft forward diverged for this chunk.
        # Record separately; do NOT include in alpha/PPL averages.
        draft_nan = not torch.isfinite(logits_d).all()
        if draft_nan:
            nan_chunk_count += 1
        elif torch.isfinite(out_d.loss):
            n_tok = chunk.size(1) - 1
            draft_nlls.append(float(out_d.loss.item()) * n_tok)
            total_tokens_d += n_tok

        for h in hook_handles:
            h.remove()
        hook_handles.clear()

        # Alpha: skip NaN chunks entirely (don't bias the mean downward).
        if not draft_nan and logits_b.size(0) > 0 and logits_d.size(0) > 0:
            alpha_chunk = compute_alpha_from_logits(
                logits_b[:64],
                logits_d[:64],
            )
            if alpha_chunk is not None:
                alphas.append(alpha_chunk)
                accepted = alpha_chunk > 0.5
                for w in wrappers:
                    if isinstance(w, Alg5_OnlineBandit):
                        w.bandit_update(accepted)
                    if isinstance(w, OnlineAB):
                        w.bandit_update(accepted)
            else:
                nan_chunk_count += 1

    # ── Layer cosine similarity ───────────────────────────────────────────
    layer_cos = []
    for l_i in range(L):
        sims = []
        n_b = len(baseline_moe_outs[l_i])
        n_d = len(draft_moe_outs[l_i])
        for t in range(min(n_b, n_d)):
            b_t = baseline_moe_outs[l_i][t].flatten(0, -2)   # [T*B, D]
            d_t = draft_moe_outs[l_i][t].flatten(0, -2)
            if b_t.shape == d_t.shape and b_t.numel() > 0:
                cos = F.cosine_similarity(b_t, d_t, dim=-1).mean().item()
                sims.append(cos)
        layer_cos.append(float(np.mean(sims)) if sims else 0.0)

    ppl_b = math.exp(sum(baseline_nlls) / max(total_tokens_b, 1)) \
        if baseline_nlls else float("nan")
    # When ALL draft chunks had NaN logits, ppl_d is genuinely unmeasurable.
    # Return nan rather than the previous spurious ≈1.0 fallback.
    ppl_d = math.exp(sum(draft_nlls) / max(total_tokens_d, 1)) \
        if draft_nlls else float("nan")

    if nan_chunk_count > 0:
        total_c = max_chunks or len(chunks)
        print(f"      ⚠ {nan_chunk_count}/{total_c} chunks had NaN draft logits "
              f"(excluded from α and PPL)")

    return {
        "mean_alpha":     float(np.mean(alphas)) if alphas else float("nan"),
        "std_alpha":      float(np.std(alphas))  if alphas else float("nan"),
        "ppl_baseline":   ppl_b,
        "ppl_draft":      ppl_d,
        "layer_cos_sim":  layer_cos,
        "nan_chunk_count": nan_chunk_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 7b – Document algorithms: Static-EBB, Online-AB, Hybrid-CP
# ─────────────────────────────────────────────────────────────────────────────

# ── Shared risk state for budget-controlled algorithms ───────────────────────

class EBBSharedState:
    """
    Cross-layer risk accumulator for one draft forward pass.
    All layer wrappers of a Static-EBB or Hybrid-CP run share ONE instance.
    Risk is accumulated during _reroute(); termination decision is made
    post-forward by post_forward_check() — never mid-layer.
    """
    def __init__(self, budget: float):
        self.budget           = budget
        self.cumulative_risk  = 0.0
        self._terminate       = False

    def reset(self):
        self.cumulative_risk = 0.0
        self._terminate      = False

    def add_risk(self, delta: float):
        self.cumulative_risk += delta

    def post_forward_check(self):
        if self.cumulative_risk > self.budget:
            self._terminate = True

    @property
    def should_terminate(self) -> bool:
        return self._terminate


# ── Algorithm: Static-EBB ────────────────────────────────────────────────────

class StaticEBB(BaseReroutingWrapper):
    """
    Layer-Adaptive Error-Budgeted Buddy Substitution (Static-EBB).

    For each miss expert i, in order:
      1. If w_i < θ_skip(l) AND crit_l(i) < CRIT_LOW → skip, accumulate skip risk.
      2. Else find j* = argmin_{j∈C, S(i,j)≥θ_sim(l)} R(i→j).
         If R(i→j*) fits in budget → substitute; weight merge: w'_j += w_i·S(i,j*).
      3. Else try skip fallback if risk fits and weight is small.
      4. Else flag stop_draft (resolved post-forward by EBBSharedState).

    Risk formula: R(i→j) = w_i · D(i,j) · ω_l · (1 + crit_l(i))
    Skip risk:    R(i→∅) = w_i · D_skip(i) · ω_l

    STOP behavior: the current forward always completes all layers.
    EBBSharedState.post_forward_check() sets should_terminate after the full
    forward, which the eval loop reads between draft steps.
    """

    CRIT_LOW = 0.30       # experts with crit < this can be skipped if weight is small

    def __init__(self, orig, cache, layer_idx, calib,
                 shared_state: EBBSharedState,
                 theta_min: float = 0.02, theta_max: float = 0.08,
                 sim_min:   float = 0.50, sim_max:   float = 0.85):
        super().__init__(orig, cache, layer_idx, calib)
        self.shared    = shared_state
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.sim_min   = sim_min
        self.sim_max   = sim_max

    def _thresholds(self):
        omega = float(self.calib.sensitivity[self.l].item()) \
            if self.calib.sensitivity is not None else 0.5
        theta_skip = self.theta_min + (self.theta_max - self.theta_min) * (1.0 - omega)
        theta_sim  = self.sim_min   + (self.sim_max   - self.sim_min)   * omega
        return theta_skip, theta_sim, omega

    def _get_tables(self, device):
        l = self.l
        # Combined similarity (primary signal)
        if self.calib.combined_sim_table is not None:
            S = self.calib.combined_sim_table[l].float().to(device)
        elif self.calib.cond_error_sim_table is not None:
            S = self.calib.cond_error_sim_table[l].float().to(device)
        else:
            S = torch.eye(self.num_exp, device=device)
        # Error table D(i,j) — normalised replacement error
        if self.calib.error_table is not None:
            D = self.calib.error_table[l].to(device)
        else:
            D = torch.ones(self.num_exp, self.num_exp, device=device)
        # Skip error D(i,∅)
        if self.calib.skip_error_table is not None:
            D_skip = self.calib.skip_error_table[l].to(device)
        else:
            D_skip = torch.ones(self.num_exp, device=device) * 0.5
        # Critical score
        if self.calib.critical_score is not None:
            crit = self.calib.critical_score[l].to(device)
        else:
            crit = torch.ones(self.num_exp, device=device) * 0.5
        return S, D, D_skip, crit

    def _reroute(self, logits_raw, router_weights, router_indices, cached_mask):
        T, k   = router_weights.shape
        device = logits_raw.device
        theta_skip, theta_sim, omega = self._thresholds()
        S, D, D_skip, crit = self._get_tables(device)

        # Output accumulator: slot_weights[e] = total weight assigned to expert e
        N = self.num_exp
        slot_weights = torch.zeros(T, N, device=device, dtype=torch.float32)

        hit_mask = cached_mask[router_indices]  # [T, k]
        # Hits: direct pass-through
        hit_w = router_weights.float() * hit_mask.float()
        slot_weights.scatter_add_(1, router_indices, hit_w)

        # Candidate sims for each miss: masked to cached-only
        S_cached = S.masked_fill(~cached_mask.unsqueeze(0), -1e9)  # [N, N]
        S_cached.fill_diagonal_(-1e9)

        stop_flagged = False
        for t in range(T):
            for r in range(k):
                e  = int(router_indices[t, r].item())
                w  = float(router_weights[t, r].item())
                if hit_mask[t, r]:
                    continue

                cr       = float(crit[e].item())
                skip_rsk = w * float(D_skip[e].item()) * omega
                sub_rsk_multiplier = omega * (1.0 + cr)

                # ── Decision 1: low-weight + low-criticality → skip ────────
                if w < theta_skip and cr < self.CRIT_LOW:
                    self.shared.add_risk(skip_rsk)
                    self.stats.skip_counts.append(1)
                    continue

                # ── Decision 2: find best substitute ──────────────────────
                # Filter by similarity threshold
                sim_row = S_cached[e]                           # [N]
                eligible = (sim_row >= theta_sim) & cached_mask
                best_j, best_risk = -1, float("inf")
                if eligible.any():
                    D_row  = D[e]                               # [N]
                    risks  = w * D_row * sub_rsk_multiplier     # [N]
                    risks  = risks.masked_fill(~eligible, 1e9)
                    best_risk_t, best_j_t = risks.min(dim=0)
                    best_j    = int(best_j_t.item())
                    best_risk = float(best_risk_t.item())

                if best_j >= 0 and (
                    self.shared.cumulative_risk + best_risk <= self.shared.budget):
                    sim_val = float(S[e, best_j].item())
                    slot_weights[t, best_j] += w * max(sim_val, 0.0)
                    self.shared.add_risk(best_risk)
                    self.stats.sub_sim_scores.append(sim_val)
                    continue

                # ── Decision 3: skip fallback ──────────────────────────────
                if (w < theta_skip * 2 and
                        self.shared.cumulative_risk + skip_rsk <= self.shared.budget):
                    self.shared.add_risk(skip_rsk)
                    self.stats.skip_counts.append(1)
                    continue

                # ── Decision 4: flag stop (post-forward) ───────────────────
                stop_flagged = True
                # Still need valid output: assign to best cached by sim (no budget check)
                best_unconstrained = int(S_cached[e].argmax().item())
                if cached_mask[best_unconstrained]:
                    sv = max(float(S[e, best_unconstrained].item()), 0.0)
                    slot_weights[t, best_unconstrained] += w * sv

        if stop_flagged:
            self.shared._terminate = True   # mark directly; post_forward_check() re-confirms

        self.stats.hit_counts.append(int(hit_mask.sum()))
        self.stats.miss_counts.append(int((~hit_mask).sum()))

        # Convert slot_weights → top-k format (keep non-zero cached slots)
        slot_weights = slot_weights * cached_mask.float().unsqueeze(0)
        final_weights, final_indices = slot_weights.topk(k, dim=-1)
        return final_weights.to(router_weights.dtype), final_indices

    def post_forward_check(self):
        self.shared.post_forward_check()

    @property
    def should_terminate(self) -> bool:
        return self.shared.should_terminate


# ── Algorithm: Online-AB ─────────────────────────────────────────────────────

class OnlineABState:
    """
    Bandit state for Online-AB with 3-dimensional context:
      (entropy bin b_H, expert rank bin b_R, draft position bin b_P)
      each ∈ {0,1,2} → 27 contexts total.

    Q[l, i, a, c] ∈ ℝ     — estimated acceptance contribution
    N[l, i, a, c] ∈ ℕ     — observation count
    a ∈ {0..N-1, N}       — expert substitute (0..N-1) or SKIP (index N)
    """

    SKIP_IDX: int = -1   # sentinel; mapped to a real index during storage

    def __init__(self, num_layers: int, num_experts: int,
                 calib: CalibrationData,
                 eta: float = 0.05, beta: float = 0.3,
                 lambda_risk: float = 0.5):
        self.L   = num_layers
        self.N   = num_experts
        self.eta = eta
        self.beta = beta
        self.lambda_risk = lambda_risk
        A = num_experts + 1    # +1 for skip action
        C = 27                 # 3 × 3 × 3 context bins

        # Init Q from similarity prior: (1+S(i,j))/2; skip init from skip error
        if calib.combined_sim_table is not None:
            S = calib.combined_sim_table.float()          # [L, N, N]
        elif calib.similarity_table is not None:
            S = calib.similarity_table.float()
        else:
            S = torch.zeros(num_layers, num_experts, num_experts)

        q_init = torch.zeros(num_layers, num_experts, A, C)
        # Substitute actions: Q(i,j,c) = (1+S(i,j))/2
        q_init[:, :, :num_experts, :] = ((1.0 + S) / 2.0).unsqueeze(-1).expand(
            -1, -1, -1, C)
        # Skip action: Q(i,skip,c) initialised from skip error → lower is worse
        if calib.skip_error_table is not None:
            # skip_error ≈ 0 means the expert contributes nothing → skip is safe
            skip_q = 1.0 - calib.skip_error_table.float().clamp(0, 1)  # [L, N]
            q_init[:, :, num_experts, :] = skip_q.unsqueeze(-1).expand(-1, -1, C)
        else:
            q_init[:, :, num_experts, :] = 0.5

        self.Q = q_init
        self.N_obs = torch.ones(num_layers, num_experts, A, C, dtype=torch.int32)

        # Pending: list of (l, i_miss, a_idx, c_idx) accumulated per forward
        self._pending: List[Tuple[int, int, int, int, float]] = []
        # a_idx == num_experts means skip

    @staticmethod
    def context_idx(entropy_norm: float, rank: int, draft_pos: int) -> int:
        """Map (entropy, rank, draft_pos) to a single context index 0..26."""
        b_H = 0 if entropy_norm < 0.33 else (1 if entropy_norm < 0.67 else 2)
        b_R = 0 if rank == 0 else (1 if rank <= 3 else 2)
        b_P = 0 if draft_pos == 0 else (1 if draft_pos <= 2 else 2)
        return b_H * 9 + b_R * 3 + b_P

    def select(self, l: int, i_miss: int, cached_list: List[int],
               c_idx: int, risk_vec: torch.Tensor) -> Tuple[int, float]:
        """
        UCB + risk-penalty selection.
        Returns (action_expert_idx_or_SKIP_IDX, sim_value).
        """
        A = self.N + 1
        q_row   = self.Q[l, i_miss, :, c_idx]       # [A]
        n_row   = self.N_obs[l, i_miss, :, c_idx]    # [A]
        n_total = float(n_row.sum().item())
        ucb     = self.beta * (math.log(n_total + 1) / (n_row.float() + 1)).sqrt()
        score   = q_row + ucb - self.lambda_risk * risk_vec  # [A]

        # Mask non-cached substitute actions
        mask = torch.zeros(A, dtype=torch.bool)
        for j in cached_list:
            mask[j] = True
        mask[self.N] = True   # skip always available

        score = score.masked_fill(~mask, -1e9)
        best  = int(score.argmax().item())
        return (best if best < self.N else self.SKIP_IDX)

    def record(self, l: int, i_miss: int, action: int, c_idx: int, weight: float):
        """Record an action for deferred update after verify."""
        a_idx = action if action >= 0 else self.N   # SKIP_IDX → column N
        self._pending.append((l, i_miss, a_idx, c_idx, weight))

    def update(self, accepted: bool, verify_miss: bool, latency_norm: float,
               lambda_miss: float = 0.1, lambda_lat: float = 0.1):
        """EMA update with credit assignment by routing weight."""
        if not self._pending:
            return
        r_base = (1.0 if accepted else 0.0) \
            - lambda_miss * float(verify_miss) \
            - lambda_lat  * latency_norm

        # Credit denominator = sum of weights in this step
        denom = sum(w for *_, w in self._pending)
        denom = max(denom, 1e-8)

        for (l, i_miss, a_idx, c_idx, w) in self._pending:
            r_action = (w / denom) * r_base
            old_q = float(self.Q[l, i_miss, a_idx, c_idx].item())
            self.Q[l, i_miss, a_idx, c_idx]     = (1 - self.eta) * old_q + self.eta * r_action
            self.N_obs[l, i_miss, a_idx, c_idx] += 1
        self._pending.clear()


class OnlineAB(BaseReroutingWrapper):
    """
    Acceptance-Guided Contextual Bandit Substitution (Online-AB).

    For each miss expert i, builds a risk vector over all cached substitutes
    and the skip action, then uses UCB + risk-penalty to select the action.
    After verify, updates Q via EMA with credit-weighted reward.
    """

    def __init__(self, orig, cache, layer_idx, calib,
                 ab_state: OnlineABState,
                 shared_state: EBBSharedState,
                 draft_pos: int = 0):
        super().__init__(orig, cache, layer_idx, calib)
        self.ab_state    = ab_state
        self.shared      = shared_state
        self.draft_pos   = draft_pos   # fixed to 0 in eval (no step-level iteration)

    def _get_tables(self, device):
        l = self.l
        S = (self.calib.combined_sim_table[l].float().to(device)
             if self.calib.combined_sim_table is not None
             else torch.eye(self.num_exp, device=device))
        D = (self.calib.error_table[l].to(device)
             if self.calib.error_table is not None
             else torch.ones(self.num_exp, self.num_exp, device=device))
        D_skip = (self.calib.skip_error_table[l].to(device)
                  if self.calib.skip_error_table is not None
                  else torch.ones(self.num_exp, device=device) * 0.5)
        crit = (self.calib.critical_score[l].to(device)
                if self.calib.critical_score is not None
                else torch.ones(self.num_exp, device=device) * 0.5)
        omega = float(self.calib.sensitivity[l].item()) \
            if self.calib.sensitivity is not None else 0.5
        return S, D, D_skip, crit, omega

    def _reroute(self, logits_raw, router_weights, router_indices, cached_mask):
        T, k   = router_weights.shape
        N      = self.num_exp
        device = logits_raw.device
        S, D, D_skip, crit, omega = self._get_tables(device)

        # Routing entropy for context binning (averaged over batch)
        probs    = F.softmax(logits_raw.float(), dim=-1)      # [T, N]
        entropy  = -(probs * (probs + 1e-9).log()).sum(-1)    # [T]
        H_max    = math.log(N)
        H_norm   = float(entropy.mean().item()) / max(H_max, 1e-8)

        cached_list = self.cache.cached_list(self.l)
        slot_weights = torch.zeros(T, N, device=device, dtype=torch.float32)
        hit_mask = cached_mask[router_indices]

        # Hits
        hit_w = router_weights.float() * hit_mask.float()
        slot_weights.scatter_add_(1, router_indices, hit_w)

        # Build a risk tensor [A] for the bandit selection (A = N+1)
        A = N + 1
        for t in range(T):
            for r in range(k):
                e   = int(router_indices[t, r].item())
                w   = float(router_weights[t, r].item())
                rank = r  # rank in top-k (0 = highest weight)
                if hit_mask[t, r]:
                    continue

                cr      = float(crit[e].item())
                c_idx   = OnlineABState.context_idx(H_norm, rank, self.draft_pos)

                # Build risk vector [A]: risk for each possible action
                risk_vec = torch.zeros(A)
                for j in cached_list:
                    risk_vec[j] = w * float(D[e, j].item()) * omega * (1.0 + cr)
                risk_vec[N] = w * float(D_skip[e].item()) * omega  # skip risk

                action = self.ab_state.select(
                    self.l, e, cached_list, c_idx, risk_vec)
                action_risk = float(risk_vec[action if action >= 0 else N].item())

                if self.shared.cumulative_risk + action_risk > self.shared.budget:
                    self.shared._terminate = True
                    # Fallback: assign to highest-sim cached expert without budget check
                    S_row = S[e].masked_fill(~cached_mask, -1e9)
                    S_row[e] = -1e9
                    fallback_j = int(S_row.argmax().item())
                    if cached_mask[fallback_j]:
                        slot_weights[t, fallback_j] += w * max(float(S[e, fallback_j].item()), 0)
                    self.ab_state.record(self.l, e, fallback_j, c_idx, w)
                    continue

                self.shared.add_risk(action_risk)
                self.ab_state.record(self.l, e, action, c_idx, w)

                if action == OnlineABState.SKIP_IDX:
                    self.stats.skip_counts.append(1)
                else:
                    sim_val = max(float(S[e, action].item()), 0.0)
                    slot_weights[t, action] += w * sim_val
                    self.stats.sub_sim_scores.append(sim_val)

        self.stats.hit_counts.append(int(hit_mask.sum()))
        self.stats.miss_counts.append(int((~hit_mask).sum()))

        slot_weights = slot_weights * cached_mask.float().unsqueeze(0)
        final_weights, final_indices = slot_weights.topk(k, dim=-1)
        return final_weights.to(router_weights.dtype), final_indices

    def bandit_update(self, accepted: bool, verify_miss: bool = False,
                      latency_norm: float = 0.0):
        self.ab_state.update(accepted, verify_miss, latency_norm)

    def post_forward_check(self):
        self.shared.post_forward_check()

    @property
    def should_terminate(self) -> bool:
        return self.shared.should_terminate


# ── Algorithm: Hybrid-CP ─────────────────────────────────────────────────────

class HybridCP(nn.Module):
    """
    CachePrior-Guided Hybrid Constrained Rerouting (Hybrid-CP).

    Overrides the full MoE forward to:
      1. Compute original top-k → measure miss ratio ρ_miss.
      2. Compute adaptive bias γ = γ₀·ρ_miss·H_norm·(1-ω_l).
      3. Apply bias only to cached experts within a top-J candidate pool.
      4. Select draft top-k from biased logits.
      5. Compute weights from ORIGINAL logits (not biased).
      6. Deviation guard: if Δ_route = Σ w_i for displaced original experts > τ,
         flag stop (resolved post-forward).
      7. Post-routing fallback: remaining misses handled by EBB-style logic.
    """

    def __init__(self, original_block, cache: SimulatedCache,
                 layer_idx: int, calib: CalibrationData,
                 shared_state: EBBSharedState,
                 gamma0: float = 3.0,
                 J_factor: int = 3,        # candidate pool = J_factor × k
                 tau_min:  float = 0.15,
                 tau_max:  float = 0.40,
                 theta_skip: float = 0.04,
                 sim_threshold: float = 0.55):
        super().__init__()
        self.orig          = original_block
        self.cache         = cache
        self.l             = layer_idx
        self.calib         = calib
        self.shared        = shared_state
        self.gamma0        = gamma0
        self.J_factor      = J_factor
        self.tau_min       = tau_min
        self.tau_max       = tau_max
        self.theta_skip    = theta_skip
        self.sim_threshold = sim_threshold
        self.top_k         = getattr(original_block, "top_k", None)
        self.num_exp       = getattr(original_block, "num_experts", None)
        self.stats         = ReroutingStats()

    def _get_tables(self, device):
        l = self.l
        S = (self.calib.combined_sim_table[l].float().to(device)
             if self.calib.combined_sim_table is not None
             else torch.eye(self.num_exp, device=device))
        D_skip = (self.calib.skip_error_table[l].to(device)
                  if self.calib.skip_error_table is not None
                  else torch.ones(self.num_exp, device=device) * 0.5)
        D = (self.calib.error_table[l].to(device)
             if self.calib.error_table is not None
             else torch.ones(self.num_exp, self.num_exp, device=device))
        crit = (self.calib.critical_score[l].to(device)
                if self.calib.critical_score is not None
                else torch.ones(self.num_exp, device=device) * 0.5)
        omega = float(self.calib.sensitivity[l].item()) \
            if self.calib.sensitivity is not None else 0.5
        return S, D, D_skip, crit, omega

    def forward(self, hidden_states: torch.Tensor):
        B, T, D_dim = hidden_states.shape
        h_flat = hidden_states.view(-1, D_dim)
        device = h_flat.device
        k      = self.top_k
        S, D, D_skip, crit, omega = self._get_tables(device)
        cached_mask = self.cache.cached_mask(self.l, device)

        # ── 1. Original router: top-k ─────────────────────────────────────
        logits_raw = self.orig.gate(h_flat)              # [T, N]
        probs_orig = F.softmax(logits_raw.float(), dim=-1)
        orig_w, orig_idx = probs_orig.topk(k, dim=-1)   # [T,k] each

        # ── 2. Adaptive bias γ ────────────────────────────────────────────
        orig_hit = cached_mask[orig_idx]                 # [T,k] bool
        rho_miss = (~orig_hit).float().mean().item()     # scalar

        entropy  = -(probs_orig * (probs_orig + 1e-9).log()).sum(-1)  # [T]
        H_norm   = float(entropy.mean().item()) / max(math.log(self.num_exp), 1e-8)

        gamma    = self.gamma0 * rho_miss * H_norm * (1.0 - omega)

        # ── 3. Candidate pool: top-J from original logits ─────────────────
        J = min(k * self.J_factor, self.num_exp)
        _, pool_idx = logits_raw.topk(J, dim=-1)         # [T, J]
        pool_mask = torch.zeros(T, self.num_exp, dtype=torch.bool, device=device)
        pool_mask.scatter_(1, pool_idx, True)             # [T, N]

        # ── 4. Apply cache bias only to cached ∩ pool ─────────────────────
        bias_mask   = cached_mask.unsqueeze(0) & pool_mask  # [T, N]
        biased_logits = logits_raw + gamma * bias_mask.float()

        # ── 5. Draft top-k from biased logits ─────────────────────────────
        _, draft_idx = biased_logits.topk(k, dim=-1)     # [T, k]

        # ── 6. Weights from ORIGINAL logits over draft_idx ────────────────
        draft_logits_sel = logits_raw.gather(-1, draft_idx)       # [T,k]
        draft_w = F.softmax(draft_logits_sel.float(), dim=-1)     # [T,k]

        # ── 7. Deviation guard ────────────────────────────────────────────
        tau = self.tau_min + (self.tau_max - self.tau_min) * (1.0 - omega)
        # delta = Σ original weights for experts displaced from top-k
        orig_set   = set()   # placeholder; compute per-token below
        deviation  = 0.0
        for t in range(T):
            orig_set_t  = set(orig_idx[t].tolist())
            draft_set_t = set(draft_idx[t].tolist())
            displaced   = orig_set_t - draft_set_t
            for e in displaced:
                r = (orig_idx[t] == e).nonzero(as_tuple=True)[0]
                if r.numel() > 0:
                    deviation += float(orig_w[t, r[0]].item())
        deviation /= max(T, 1)

        if deviation > tau:
            self.shared._terminate = True

        # ── 8. Post-routing fallback for remaining misses ─────────────────
        draft_hit = cached_mask[draft_idx]               # [T, k] bool
        out = torch.zeros(B * T, D_dim, dtype=hidden_states.dtype, device=device)
        slot_weights = torch.zeros(T, self.num_exp, device=device, dtype=torch.float32)

        hit_w = draft_w * draft_hit.float()
        slot_weights.scatter_add_(1, draft_idx, hit_w)

        S_cached = S.masked_fill(~cached_mask.unsqueeze(0), -1e9)
        S_cached.fill_diagonal_(-1e9)

        for t in range(T):
            for r in range(k):
                e = int(draft_idx[t, r].item())
                w = float(draft_w[t, r].item())
                if draft_hit[t, r]:
                    continue
                cr       = float(crit[e].item())
                skip_rsk = w * float(D_skip[e].item()) * omega

                # Try substitute
                sim_row  = S_cached[e]
                eligible = (sim_row >= self.sim_threshold) & cached_mask
                if eligible.any():
                    D_row  = D[e]
                    risks  = w * D_row * omega * (1.0 + cr)
                    risks  = risks.masked_fill(~eligible, 1e9)
                    best_risk_t, best_j_t = risks.min(0)
                    best_j    = int(best_j_t.item())
                    best_risk = float(best_risk_t.item())
                    if self.shared.cumulative_risk + best_risk <= self.shared.budget:
                        sv = max(float(S[e, best_j].item()), 0.0)
                        slot_weights[t, best_j] += w * sv
                        self.shared.add_risk(best_risk)
                        continue

                # Skip fallback
                if w < self.theta_skip and (
                        self.shared.cumulative_risk + skip_rsk <= self.shared.budget):
                    self.shared.add_risk(skip_rsk)
                    continue

                # Last resort: round-robin over cached (no budget check)
                cached_list = self.cache.cached_list(self.l)
                if cached_list:
                    fallback = cached_list[e % len(cached_list)]
                    slot_weights[t, fallback] += w

        self.shared.post_forward_check()

        # ── 9. Expert computation ─────────────────────────────────────────
        slot_weights = slot_weights * cached_mask.float().unsqueeze(0)
        w_sum = slot_weights.sum(-1, keepdim=True).clamp(min=1e-9)
        slot_weights = slot_weights / w_sum

        final_weights, final_indices = slot_weights.topk(k, dim=-1)
        final_weights = final_weights.to(hidden_states.dtype)

        for slot in range(k):
            slot_exp = final_indices[:, slot]
            slot_w   = final_weights[:, slot]
            for e_idx in slot_exp.unique():
                e_idx = int(e_idx.item())
                if e_idx < 0 or e_idx >= self.num_exp:
                    continue
                mask = (slot_exp == e_idx)
                if not mask.any():
                    continue
                e_out = self.orig.experts[e_idx](h_flat[mask])
                if isinstance(e_out, tuple):
                    e_out = e_out[0]
                out[mask] += (e_out * slot_w[mask].unsqueeze(-1)).to(hidden_states.dtype)

        if hasattr(self.orig, "shared_expert") and self.orig.shared_expert is not None:
            shared = self.orig.shared_expert(h_flat)
            if hasattr(self.orig, "shared_expert_gate"):
                gate = torch.sigmoid(self.orig.shared_expert_gate(h_flat))
                shared = gate * shared
            out = out + shared.to(out.dtype)

        # Stats
        self.stats.hit_counts.append(int(draft_hit.sum()))
        self.stats.miss_counts.append(int((~draft_hit).sum()))
        return out.view(B, T, D_dim), logits_raw

    @property
    def should_terminate(self) -> bool:
        return self.shared.should_terminate


# ─────────────────────────────────────────────────────────────────────────────
# Section 8 – Experiment runner
# ─────────────────────────────────────────────────────────────────────────────

ALGORITHM_NAMES = [
    # Baselines
    "SkipAll", "RoundRobin",
    # Alg1 sim-table ablation
    "Alg1_CosineOut", "Alg1_CoAct", "Alg1_CondError", "Alg1_RouterCorr",
    # Weight-correction ablation
    "W0_NoCorr", "W1_SimScale", "W2_HitRenorm", "W3_UniformRedist", "W4_TopHitRedist",
    # Existing higher algorithms
    "Alg2_EntropyBias", "Alg3_RouterMerge", "Alg4_ErrorBudget", "Alg5_Bandit",
    # Document algorithms
    "StaticEBB", "OnlineAB", "HybridCP",
]

ALG_GROUPS = {
    "baseline":    ["SkipAll", "RoundRobin"],
    "sim_table":   ["Alg1_CosineOut", "Alg1_CoAct", "Alg1_CondError", "Alg1_RouterCorr"],
    "weight_corr": ["W0_NoCorr", "W1_SimScale", "W2_HitRenorm",
                    "W3_UniformRedist", "W4_TopHitRedist"],
    "existing":    ["Alg2_EntropyBias", "Alg3_RouterMerge",
                    "Alg4_ErrorBudget", "Alg5_Bandit"],
    "document":    ["StaticEBB", "OnlineAB", "HybridCP"],
}

COLORS = {
    "SkipAll":           "#d62728",
    "RoundRobin":        "#ff7f0e",
    "Alg1_CosineOut":    "#2ca02c",
    "Alg1_CoAct":        "#98df8a",
    "Alg1_CondError":    "#17becf",
    "Alg1_RouterCorr":   "#aec7e8",
    "W0_NoCorr":         "#9467bd",
    "W1_SimScale":       "#c5b0d5",
    "W2_HitRenorm":      "#8c564b",
    "W3_UniformRedist":  "#c49c94",
    "W4_TopHitRedist":   "#e377c2",
    "Alg2_EntropyBias":  "#1f77b4",
    "Alg3_RouterMerge":  "#0d4f8c",
    "Alg4_ErrorBudget":  "#ff9896",
    "Alg5_Bandit":       "#f7b6d2",
    "StaticEBB":         "#2c7bb6",
    "OnlineAB":          "#d7191c",
    "HybridCP":          "#1a9641",
}
MARKERS = {n: m for n, m in zip(ALGORITHM_NAMES,
           "xD" + "ovs^" + "PDPD>" + "o^sP" + "*HD")}


def build_wrappers_for_algorithm(
    alg_name: str, moe_layers: List[Tuple[int, nn.Module]],
    cache: SimulatedCache, calib: CalibrationData,
    bandit_state: Optional[BanditState] = None,
    ab_state: Optional[OnlineABState] = None,
    ebb_shared: Optional[EBBSharedState] = None,
) -> List:
    wrappers = []
    for l_i, (global_idx, moe_mod) in enumerate(moe_layers):
        if alg_name == "SkipAll":
            w = Baseline_SkipAll(moe_mod, cache, l_i, calib)
        elif alg_name == "RoundRobin":
            w = Baseline_RoundRobin(moe_mod, cache, l_i, calib)
        elif alg_name == "Alg1_CosineOut":
            w = Alg1_CosineOut(moe_mod, cache, l_i, calib)
        elif alg_name == "Alg1_CoAct":
            w = Alg1_CoAct(moe_mod, cache, l_i, calib)
        elif alg_name == "Alg1_CondError":
            w = Alg1_CondError(moe_mod, cache, l_i, calib)
        elif alg_name == "Alg1_RouterCorr":
            w = Alg1_RouterCorr(moe_mod, cache, l_i, calib)
        elif alg_name in ("W0_NoCorr", "W1_SimScale", "W2_HitRenorm",
                          "W3_UniformRedist", "W4_TopHitRedist"):
            w = WeightCorrectionAblation(moe_mod, cache, l_i, calib, mode=alg_name)
        elif alg_name == "Alg2_EntropyBias":
            w = Alg2_EntropyBias(moe_mod, cache, l_i, calib)
        elif alg_name == "Alg3_RouterMerge":
            w = Alg3_RouterScoreMerge(moe_mod, cache, l_i, calib)
        elif alg_name == "Alg4_ErrorBudget":
            w = Alg4_ErrorBudget(moe_mod, cache, l_i, calib)
        elif alg_name == "Alg5_Bandit":
            assert bandit_state is not None
            w = Alg5_OnlineBandit(moe_mod, cache, l_i, calib, bandit_state)
        # ── Document algorithms ───────────────────────────────────────────
        elif alg_name == "StaticEBB":
            assert ebb_shared is not None
            w = StaticEBB(moe_mod, cache, l_i, calib, ebb_shared)
        elif alg_name == "OnlineAB":
            assert ab_state is not None and ebb_shared is not None
            w = OnlineAB(moe_mod, cache, l_i, calib, ab_state, ebb_shared)
        elif alg_name == "HybridCP":
            assert ebb_shared is not None
            w = HybridCP(moe_mod, cache, l_i, calib, ebb_shared)
        else:
            raise ValueError(f"Unknown algorithm: {alg_name}")
        wrappers.append(w)
    return wrappers


def run_all_experiments(
    model, tokenizer, moe_cfg: dict, calib: CalibrationData,
    eval_chunks: List[torch.Tensor],
    cache_ratios: List[float],
    device: str,
    outdir: str,
    n_eval_chunks: int = 100,
    algorithms: Optional[List[str]] = None,   # None → run all ALGORITHM_NAMES
) -> Dict:
    """
    Run selected algorithms × cache_ratios. Returns nested result dict.
    Pass algorithms=["StaticEBB","OnlineAB","HybridCP"] to run a subset.
    """
    run_names = algorithms if algorithms is not None else ALGORITHM_NAMES
    # Validate
    unknown = [a for a in run_names if a not in ALGORITHM_NAMES]
    if unknown:
        raise ValueError(f"Unknown algorithm(s): {unknown}. "
                         f"Valid: {ALGORITHM_NAMES}")
    moe_attr   = moe_cfg["moe_attr"]
    moe_layers = get_moe_layers(model, moe_attr)
    L          = len(moe_layers)
    N          = moe_cfg["num_experts"]
    moe_layer_indices = [gi for gi, _ in moe_layers]

    # Stash original modules (they will be replaced and restored per run)
    originals = [getattr(model.model.layers[gi], moe_attr)
                 for gi in moe_layer_indices]

    results: Dict[str, Dict[float, Dict]] = {a: {} for a in run_names}

    for cache_ratio in cache_ratios:
        print(f"\n{'='*60}")
        print(f"  Cache ratio: {cache_ratio:.3f}  "
              f"({int(N * cache_ratio)}/{N} experts cached per layer)")
        print(f"{'='*60}")

        cache = SimulatedCache(
            L, N, cache_ratio,
            activation_freq=calib.activation_freq,
        )
        miss_rate_expected = 1.0 - cache_ratio
        print(f"  Expected miss rate per slot: {miss_rate_expected:.1%}")

        bandit_state = BanditState(
            num_layers  = L,
            num_experts = N,
            sim_prior   = calib.similarity_table,
        ) if calib.similarity_table is not None else BanditState(L, N)

        ebb_budget  = 0.50
        ebb_shared  = EBBSharedState(budget=ebb_budget)
        ab_state    = OnlineABState(
            num_layers  = L,
            num_experts = N,
            calib       = calib,
        )

        for alg_name in run_names:
            print(f"  ▶ {alg_name:<22}", end="", flush=True)
            t0 = time.time()

            wrappers = build_wrappers_for_algorithm(
                alg_name, moe_layers, cache, calib,
                bandit_state = bandit_state if alg_name == "Alg5_Bandit" else None,
                ab_state     = ab_state     if alg_name == "OnlineAB"    else None,
                ebb_shared   = ebb_shared   if alg_name in ("StaticEBB", "OnlineAB", "HybridCP") else None,
            )

            metrics = run_logit_comparison(
                model, originals, wrappers, moe_layer_indices,
                moe_attr, eval_chunks, device,
                max_chunks=n_eval_chunks,
            )

            # Restore model (run_logit_comparison may leave it patched)
            restore_model(model, moe_attr, originals, moe_layer_indices)

            elapsed = time.time() - t0
            alpha   = metrics["mean_alpha"]
            nan_c   = metrics.get("nan_chunk_count", 0)
            ppl_gap = metrics["ppl_draft"] - metrics["ppl_baseline"] \
                if (math.isfinite(metrics["ppl_draft"]) and
                    math.isfinite(metrics["ppl_baseline"])) else float("nan")
            nan_str = f"  NaN chunks={nan_c}" if nan_c else ""
            print(f"  α={alpha:.4f}  ppl_gap={ppl_gap:+.3f}"
                  f"{nan_str}  ({elapsed:.1f}s)")

            results[alg_name][cache_ratio] = metrics

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Section 9 – Plotting and reporting
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Section 9 – Plotting and reporting
# ─────────────────────────────────────────────────────────────────────────────


def plot_alpha_vs_cache_ratio(results, cache_ratios, outdir):
    """Four panels: sim-table | weight-corr | existing advanced | document algorithms."""
    groups = [
        ("Similarity-table ablation",
         ["SkipAll", "RoundRobin",
          "Alg1_CosineOut", "Alg1_CoAct", "Alg1_CondError", "Alg1_RouterCorr"]),
        ("Weight-correction ablation (CondError table)",
         ["Alg1_CondError",
          "W0_NoCorr", "W1_SimScale", "W2_HitRenorm",
          "W3_UniformRedist", "W4_TopHitRedist"]),
        ("Existing advanced algorithms",
         ["Alg1_CondError", "RoundRobin",
          "Alg2_EntropyBias", "Alg3_RouterMerge",
          "Alg4_ErrorBudget", "Alg5_Bandit"]),
        ("Document algorithms (Static-EBB / Online-AB / Hybrid-CP)",
         ["SkipAll", "RoundRobin", "Alg1_CondError",
          "StaticEBB", "OnlineAB", "HybridCP"]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5), sharey=True)
    for ax, (title, alg_list) in zip(axes, groups):
        for alg in alg_list:
            if alg not in results:
                continue
            ys = [results[alg][r]["mean_alpha"] for r in cache_ratios]
            es = [results[alg][r]["std_alpha"]  for r in cache_ratios]
            ax.errorbar(cache_ratios, ys, yerr=es,
                        label=alg, color=COLORS.get(alg, "gray"),
                        marker=MARKERS.get(alg, "o"),
                        linewidth=2, markersize=7, capsize=4)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Cache Ratio (S/N)", fontsize=9)
        ax.set_ylim(0, 1)
        ax.axhline(0.7, color="gray", linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Theoretical Acceptance Rate α", fontsize=10)
    fig.suptitle("Expert Rerouting: α vs Cache Ratio", fontsize=13)
    plt.tight_layout()
    path = os.path.join(outdir, "alpha_vs_cache_ratio.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved: {path}")


def plot_ppl_gap(results, cache_ratios, outdir):
    fig, ax = plt.subplots(figsize=(9, 5))
    for alg, data in results.items():
        ys = [data[r]["ppl_draft"] - data[r]["ppl_baseline"] for r in cache_ratios]
        ax.plot(cache_ratios, ys, label=alg,
                color=COLORS[alg], marker=MARKERS[alg], linewidth=2, markersize=7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Cache Ratio", fontsize=12)
    ax.set_ylabel("PPL(draft) − PPL(baseline)", fontsize=12)
    ax.set_title("Draft PPL Degradation vs Cache Ratio", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(outdir, "ppl_gap_vs_cache_ratio.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_layer_cos_sim(results, cache_ratios, outdir):
    """One plot per cache ratio: layer-by-layer cosine similarity."""
    for ratio in cache_ratios:
        fig, ax = plt.subplots(figsize=(12, 4))
        for alg, data in results.items():
            cos_sims = data[ratio].get("layer_cos_sim", [])
            if cos_sims:
                ax.plot(cos_sims, label=alg, color=COLORS[alg],
                        linewidth=1.5, alpha=0.85)
        ax.set_xlabel("MoE Layer Index", fontsize=11)
        ax.set_ylabel("Cosine Similarity (draft vs exact output)", fontsize=11)
        ax.set_title(f"Per-Layer Output Similarity  [cache={ratio:.2f}]", fontsize=12)
        ax.legend(fontsize=8, ncol=2)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(outdir, f"layer_cos_sim_cache{ratio:.2f}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")


def plot_similarity_heatmap(calib: CalibrationData, outdir: str,
                            sample_layers: Optional[List[int]] = None):
    if calib.similarity_table is None:
        return
    S = calib.similarity_table.float().numpy()  # [L, N, N]
    L, N, _ = S.shape
    if sample_layers is None:
        sample_layers = [L // 4, L // 2, 3 * L // 4]
    sample_layers = [l for l in sample_layers if 0 <= l < L][:3]

    fig, axes = plt.subplots(1, len(sample_layers), figsize=(5 * len(sample_layers), 4))
    if len(sample_layers) == 1:
        axes = [axes]
    for ax, l in zip(axes, sample_layers):
        im = ax.imshow(S[l], vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        ax.set_title(f"Layer {l}", fontsize=11)
        ax.set_xlabel("Expert j"); ax.set_ylabel("Expert i")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Expert Output Cosine Similarity  S[i,j]", fontsize=12)
    plt.tight_layout()
    path = os.path.join(outdir, "similarity_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_sensitivity(calib: CalibrationData, outdir: str):
    if calib.sensitivity is None:
        return
    sens = calib.sensitivity.numpy()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(range(len(sens)), sens, color=cm.viridis(sens))
    ax.set_xlabel("MoE Layer Index", fontsize=11)
    ax.set_ylabel("Sensitivity ω (0=insensitive, 1=critical)", fontsize=11)
    ax.set_title("Per-Layer Rerouting Sensitivity", fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(outdir, "layer_sensitivity.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def save_csv(results, cache_ratios, outdir):
    path = os.path.join(outdir, "results_summary.csv")
    fieldnames = ["algorithm", "cache_ratio", "mean_alpha", "std_alpha",
                  "ppl_baseline", "ppl_draft", "ppl_gap",
                  "mean_layer_cos_sim", "nan_chunk_count"]
    rows = []
    for alg, data in results.items():
        for ratio in cache_ratios:
            m = data[ratio]
        ppl_b = m['ppl_baseline']
        ppl_d = m['ppl_draft']
        ppl_gap = (ppl_d - ppl_b) if (math.isfinite(ppl_d) and math.isfinite(ppl_b)) \
            else float("nan")
        cos = m.get("layer_cos_sim", [])
        rows.append({
            "algorithm":        alg,
            "cache_ratio":      f"{ratio:.3f}",
            "mean_alpha":       f"{m['mean_alpha']:.4f}",
            "std_alpha":        f"{m['std_alpha']:.4f}",
            "ppl_baseline":     f"{ppl_b:.4f}" if math.isfinite(ppl_b) else "nan",
            "ppl_draft":        f"{ppl_d:.4f}" if math.isfinite(ppl_d) else "nan",
            "ppl_gap":          f"{ppl_gap:+.4f}" if math.isfinite(ppl_gap) else "nan",
            "mean_layer_cos_sim": f"{np.mean(cos):.4f}" if cos else "n/a",
            "nan_chunk_count":  str(m.get("nan_chunk_count", 0)),
        })
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {path}")

    # Also pretty-print the alpha table
    print("\n  ── Acceptance Rate Summary ────────────────────────────────────")
    header = f"{'Algorithm':<24}" + "".join(f" {r:.3f}" for r in cache_ratios)
    print(header)
    print("-" * len(header))
    for alg in ALGORITHM_NAMES:
        row = f"{alg:<24}"
        for ratio in cache_ratios:
            row += f"  {results[alg][ratio]['mean_alpha']:.3f}"
        print(row)


def save_json(results, cache_ratios, outdir):
    """Save full results as JSON (layer_cos_sim serialized as list)."""
    serializable = {}
    for alg, data in results.items():
        serializable[alg] = {}
        for ratio, m in data.items():
            serializable[alg][str(ratio)] = {
                "mean_alpha":     m["mean_alpha"],
                "std_alpha":      m["std_alpha"],
                "ppl_baseline":   m["ppl_baseline"],
                "ppl_draft":      m["ppl_draft"],
                "layer_cos_sim":  m.get("layer_cos_sim", []),
            }
    path = os.path.join(outdir, "results_full.json")
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 10 – Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Standalone evaluation of 5 expert rerouting algorithms.")
    parser.add_argument("--model", required=True,
                        help="Path to a Qwen-style MoE model.")
    parser.add_argument("--device", default="cuda",
                        help="Device: 'cuda', 'cpu', or 'cuda:0' etc.")
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Model dtype.")
    parser.add_argument("--cache_ratios", nargs="+", type=float,
                        default=[0.125, 0.25, 0.375, 0.5, 0.625, 0.75],
                        help="List of cache ratios to sweep.")
    parser.add_argument("--n_calib", type=int, default=128,
                        help="Number of chunks for offline calibration.")
    parser.add_argument("--n_eval", type=int, default=128,
                        help="Number of chunks for algorithm evaluation.")
    parser.add_argument("--seq_len", type=int, default=256,
                        help="Sequence length for each chunk.")
    parser.add_argument("--outdir", default="./results",
                        help="Directory for output plots, CSV, and JSON.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data_file", default=None, metavar="PATH",
        help=(
            "Path to a local plain-text file used for calibration and evaluation. "
            "Recommended when the `datasets` module is not installed. "
            "Any large text file works (Wikipedia dump, ShareGPT export, etc.). "
            "Without this, the script falls back to a tiny built-in snippet which "
            "produces DEGENERATE calibration results."
        ),
    )
    parser.add_argument(
        "--algorithms", nargs="+", default=None,
        metavar="ALG",
        help=(
            "Algorithms to run (space-separated). Default: all. "
            "Groups: 'document' → StaticEBB OnlineAB HybridCP, "
            "'baseline' → SkipAll RoundRobin, "
            "'sim_table' → Alg1_* variants, "
            "'weight_corr' → W0..W4, "
            "'existing' → Alg2..Alg5. "
            f"Individual names: {ALGORITHM_NAMES}"
        ),
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # ── Resolve --algorithms: expand group names to individual algorithm names ──
    if args.algorithms is None:
        selected_algs = None   # run everything
    else:
        selected_algs = []
        for token in args.algorithms:
            if token in ALG_GROUPS:
                selected_algs.extend(ALG_GROUPS[token])
            elif token in ALGORITHM_NAMES:
                selected_algs.append(token)
            else:
                raise SystemExit(
                    f"Unknown algorithm or group: '{token}'\n"
                    f"Valid groups: {list(ALG_GROUPS)}\n"
                    f"Valid algorithms: {ALGORITHM_NAMES}"
                )
        # Deduplicate while preserving order
        seen: set = set()
        selected_algs = [a for a in selected_algs
                         if not (a in seen or seen.add(a))]
        print(f"\nRunning algorithms: {selected_algs}")

    dtype_map = {"float16": torch.float16,
                 "bfloat16": torch.bfloat16,
                 "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # ── Load model ───────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(args.model, args.device, dtype)
    moe_cfg = detect_moe_config(model)
    print(f"\nModel config: {moe_cfg}")

    # ── Prepare data ─────────────────────────────────────────────────────
    print(f"\nPreparing calibration data ({args.n_calib} chunks × {args.seq_len} tokens)...")
    calib_chunks = prepare_text_chunks(
        tokenizer, args.n_calib, args.seq_len, data_file=args.data_file)
    print(f"Preparing evaluation data ({args.n_eval} chunks × {args.seq_len} tokens)...")
    eval_chunks  = prepare_text_chunks(
        tokenizer, args.n_eval,  args.seq_len, data_file=args.data_file)

    # ── Phase 1: Calibration ─────────────────────────────────────────────
    print("\n── Phase 1: Offline Calibration ─────────────────────────────────")
    calib = build_calibration_data(
        model, tokenizer,
        moe_cfg["moe_attr"],
        calib_chunks[:args.n_calib],
        args.device,
        moe_cfg["num_experts"],
    )

    # Set entropy thresholds for Algorithm 2 from calibration data
    # (would be computed by a gate hook; using proxy values here)
    tau_low  = math.log(moe_cfg["num_experts"]) * 0.25
    tau_high = math.log(moe_cfg["num_experts"]) * 0.75

    # ── Plot calibration artefacts ────────────────────────────────────────
    print("\n── Calibration plots ────────────────────────────────────────────")
    plot_similarity_heatmap(calib, args.outdir)
    plot_sensitivity(calib, args.outdir)

    # ── Phase 2: Algorithm evaluation ────────────────────────────────────
    print("\n── Phase 2: Algorithm Evaluation ────────────────────────────────")
    results = run_all_experiments(
        model, tokenizer, moe_cfg, calib,
        eval_chunks, args.cache_ratios,
        args.device, args.outdir,
        n_eval_chunks=args.n_eval,
        algorithms=selected_algs,
    )

    # ── Phase 3: Reporting ────────────────────────────────────────────────
    print("\n── Phase 3: Reporting ───────────────────────────────────────────")
    plot_alpha_vs_cache_ratio(results, args.cache_ratios, args.outdir)
    plot_ppl_gap(results, args.cache_ratios, args.outdir)
    plot_layer_cos_sim(results, args.cache_ratios, args.outdir)
    save_csv(results, args.cache_ratios, args.outdir)
    save_json(results, args.cache_ratios, args.outdir)

    print(f"\n✓ All results saved to {args.outdir}/")


if __name__ == "__main__":
    main()
