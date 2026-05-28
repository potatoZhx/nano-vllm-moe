"""
dynamic_k_analysis.py  (Experiment E3)
======================================
Find the optimal draft length K* for each (algorithm, cache_ratio) pair,
and validate whether Level-1 threshold signals can approximate the stop point.

Core questions:
  Q1. What is K* for each (algorithm, cache_ratio)?
  Q2. How does per-step marginal acceptance probability decay?
  Q3. Can simple threshold signals (CriticalMissRate, per-step ρ_miss) detect K*?
  Q4. What is the theoretical throughput under a simplified T_cycle model?

Method:
  1. For each (algo, ratio), run decode-mode simulation at K=1..K_max
  2. Collect per-step α, miss rate, CriticalMissRate
  3. Compute marginal acceptance probability: α_marginal(k) = Π_{i=1}^{k} α_i
  4. Use simplified T_cycle model to compute throughput for each K
  5. Find K* = argmax throughput
  6. Test Level-1 threshold signals against K*

Standalone — no dependency on nano-vllm-moe runtime.

Usage:
  python dynamic_k_analysis.py \
      --model /path/to/Qwen3-30B-A3B-Base \
      --data_file ../wikitext2_test.txt \
      --cache_ratios 0.75 0.50 0.25 \
      --k_max 12 --prompt_len 128 \
      --n_calib 8 --n_eval 16 --outdir ./results_e3
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import curve_fit
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Model & data utilities (shared)
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
        except Exception:
            pass
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
    if len(chunks) < n:
        chunks = (chunks * (n // len(chunks) + 1))[:n]
    random.shuffle(chunks)
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — SimulatedCache & Calibration
# ─────────────────────────────────────────────────────────────────────────────

class SimulatedCache:
    def __init__(self, L: int, N: int, ratio: float,
                 act_freq: Optional[np.ndarray] = None):
        self.N = N
        self.cache_size = max(1, int(N * ratio))
        self.sets: List[set] = []
        for l in range(L):
            if act_freq is not None:
                top = np.argsort(act_freq[l])[::-1][:self.cache_size]
                self.sets.append(set(top.tolist()))
            else:
                self.sets.append(set(random.sample(range(N), self.cache_size)))

    def mask(self, layer: int, device) -> torch.Tensor:
        m = torch.zeros(self.N, dtype=torch.bool, device=device)
        m[list(self.sets[layer])] = True
        return m

    def is_cached(self, layer: int, expert_id: int) -> bool:
        return expert_id in self.sets[layer]


@torch.no_grad()
def calibrate_freq(model, moe_attr, chunks, device, N, top_k):
    """Lightweight calibration: only activation frequencies."""
    layers = get_moe_layers(model, moe_attr)
    L = len(layers)
    act_freq = np.zeros((L, N), dtype=np.float64)
    total_tokens = 0

    gate_outputs = [None] * L
    hooks = []
    for li, (idx, moe) in enumerate(layers):
        if hasattr(moe, "gate"):
            def _ghook(li_):
                def h(m, inp, out):
                    gate_outputs[li_] = out.detach().float().cpu()
                return h
            hooks.append(moe.gate.register_forward_hook(_ghook(li)))

    for chunk in tqdm(chunks, desc="Calibrating"):
        inp = chunk.to(device)
        gate_outputs = [None] * L
        model(inp)
        T = inp.size(1)
        total_tokens += T
        for li in range(L):
            if gate_outputs[li] is None:
                continue
            g = gate_outputs[li]
            if g.dim() == 3:
                g = g.squeeze(0)
            _, topk_i = torch.topk(g, top_k, dim=-1)
            for t in range(g.size(0)):
                for ei in topk_i[t].tolist():
                    act_freq[li, ei] += 1

    for h in hooks:
        h.remove()
    if total_tokens > 0:
        act_freq /= total_tokens
    return act_freq


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — α computation
# ─────────────────────────────────────────────────────────────────────────────

def alpha_tv(target: torch.Tensor, draft: torch.Tensor) -> Optional[float]:
    if not (torch.isfinite(target).all() and torch.isfinite(draft).all()):
        return None
    pt = F.softmax(target.float(), dim=-1)
    pd = F.softmax(draft.float(), dim=-1)
    r = float(torch.minimum(pt, pd).sum().item())
    return r if math.isfinite(r) else None


def copy_kv(kv):
    """Clone a DynamicCache KV cache."""
    if kv is None:
        return None
    from transformers import DynamicCache
    new_cache = DynamicCache()
    for layer_idx in range(len(kv)):
        key, value = kv[layer_idx]
        new_cache.update(key.clone(), value.clone(), layer_idx)
    return new_cache


def free_kv(kv):
    """No-op: DynamicCache is garbage-collected."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — SkipAll wrapper (baseline rerouting)
# ─────────────────────────────────────────────────────────────────────────────

class SkipAllWrapper(nn.Module):
    def __init__(self, orig_moe, cache, layer_idx, N, top_k):
        super().__init__()
        self.orig = orig_moe
        self.cache = cache
        self.l = layer_idx
        self.N = N
        self.top_k = top_k

    def forward(self, hidden_states):
        B, T, D = hidden_states.shape
        h = hidden_states.view(-1, D)
        dev = h.device
        logits = self.orig.gate(h)
        probs = F.softmax(logits, dim=-1)
        rw, ri = probs.topk(self.top_k, dim=-1)
        cm = self.cache.mask(self.l, dev)
        hit = cm[ri]
        fw = rw * hit.float()
        row_sum = fw.sum(-1)
        if (row_sum == 0).any():
            empty = (row_sum == 0)
            cl = sorted(self.cache.sets[self.l])
            fb = cl[0] if cl else 0
            ri_c = ri.clone()
            ri_c[empty, 0] = fb; fw[empty, 0] = 1.0
            ri = ri_c
        fw = (fw / fw.sum(-1, keepdim=True).clamp(1e-9)).to(h.dtype)
        wb = torch.zeros(B * T, self.N, dtype=h.dtype, device=dev)
        for slot in range(ri.size(1)):
            wb.scatter_add_(1, ri[:, slot:slot + 1], fw[:, slot:slot + 1])
        out = torch.zeros(B * T, D, dtype=h.dtype, device=dev)
        active = wb.any(dim=0).nonzero(as_tuple=False).squeeze(-1)
        for e_t in active:
            ei = int(e_t.item())
            we = wb[:, ei]; mask = we > 0
            if not mask.any():
                continue
            eo = self.orig.experts[ei](h[mask])
            if isinstance(eo, tuple): eo = eo[0]
            out[mask] += (eo * we[mask].unsqueeze(-1)).to(h.dtype)
        if hasattr(self.orig, "shared_expert") and self.orig.shared_expert is not None:
            sh = self.orig.shared_expert(h)
            if hasattr(self.orig, "shared_expert_gate"):
                sh = torch.sigmoid(self.orig.shared_expert_gate(h)) * sh
            out = out + sh.to(out.dtype)
        return out.view(B, T, D), logits


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Alg2_v2 wrapper
# ─────────────────────────────────────────────────────────────────────────────

MISS_GATE = 0.25
GAMMA0 = 4.0

class Alg2V2Wrapper(nn.Module):
    def __init__(self, orig_moe, cache, layer_idx, N, top_k,
                 gamma0=GAMMA0, miss_low=MISS_GATE, miss_high=0.50):
        super().__init__()
        self.orig = orig_moe
        self.cache = cache
        self.l = layer_idx
        self.N = N
        self.top_k = top_k
        self.gamma0 = gamma0
        self.miss_low = miss_low
        self.miss_high = miss_high

    def forward(self, hidden_states):
        B, T, D = hidden_states.shape
        h = hidden_states.view(-1, D)
        dev = h.device
        logits = self.orig.gate(h)
        probs = F.softmax(logits, dim=-1)
        rw, ri = probs.topk(self.top_k, dim=-1)
        cm = self.cache.mask(self.l, dev)
        hit_mask = cm[ri]
        miss_rate_t = (~hit_mask).float().mean(-1)
        lo, hi = self.miss_low, self.miss_high
        t_gate = ((miss_rate_t - lo) / max(hi - lo, 1e-6)).clamp(0.0, 1.0)
        p = F.softmax(logits.float(), dim=-1)
        entropy = -(p * (p + 1e-9).log()).sum(-1)
        H_max = math.log(self.N)
        t_ent = (entropy / H_max).clamp(0, 1)
        gamma_eff = self.gamma0 * t_gate * (0.2 + 0.8 * t_ent)
        bias = gamma_eff.unsqueeze(-1) * cm.float().unsqueeze(0)
        bgt = logits.float() + bias
        _, ni = bgt.topk(self.top_k, dim=-1)
        # Top-1 protection
        orig_top1 = ri[:, 0]
        for t_i in range(B * T):
            ot1 = int(orig_top1[t_i].item())
            if not cm[ot1] and ot1 not in ni[t_i].tolist():
                ni[t_i, -1] = ot1
        nw = F.softmax(logits.gather(-1, ni).float(), dim=-1)
        new_hit = cm[ni]
        nw = nw * new_hit.float()
        row_sum = nw.sum(-1)
        if (row_sum == 0).any():
            empty = (row_sum == 0)
            cl = sorted(self.cache.sets[self.l])
            fb = cl[0] if cl else 0
            ni[empty, 0] = fb; nw[empty, 0] = 1.0
        nw = (nw / nw.sum(-1, keepdim=True).clamp(1e-9)).to(h.dtype)
        wb = torch.zeros(B * T, self.N, dtype=h.dtype, device=dev)
        for slot in range(ni.size(1)):
            wb.scatter_add_(1, ni[:, slot:slot + 1], nw[:, slot:slot + 1])
        out = torch.zeros(B * T, D, dtype=h.dtype, device=dev)
        active = wb.any(dim=0).nonzero(as_tuple=False).squeeze(-1)
        for e_t in active:
            ei = int(e_t.item())
            we = wb[:, ei]; mask = we > 0
            if not mask.any(): continue
            eo = self.orig.experts[ei](h[mask])
            if isinstance(eo, tuple): eo = eo[0]
            out[mask] += (eo * we[mask].unsqueeze(-1)).to(h.dtype)
        if hasattr(self.orig, "shared_expert") and self.orig.shared_expert is not None:
            sh = self.orig.shared_expert(h)
            if hasattr(self.orig, "shared_expert_gate"):
                sh = torch.sigmoid(self.orig.shared_expert_gate(h)) * sh
            out = out + sh.to(out.dtype)
        return out.view(B, T, D), logits


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Decode-mode evaluation with variable K
# ─────────────────────────────────────────────────────────────────────────────

def patch_model(model, moe_attr, wrappers, moe_indices):
    for li, (idx, _) in enumerate(moe_indices):
        setattr(model.model.layers[idx], moe_attr, wrappers[li])


def restore_model(model, moe_attr, originals, moe_indices):
    for li, (idx, _) in enumerate(moe_indices):
        setattr(model.model.layers[idx], moe_attr, originals[li])


@torch.no_grad()
def eval_variable_k(
    model, moe_attr, prompts, device,
    cache, N, top_k, L,
    prompt_len, k_max,
    algorithm="skipall",
) -> dict:
    """Run decode-mode sim and collect per-step α for steps 1..k_max.

    Also collects per-step miss rate and CriticalMissRate for threshold analysis.
    """
    layers = get_moe_layers(model, moe_attr)
    moe_indices = layers
    originals = [moe for _, moe in layers]

    # Build wrappers
    wrappers = []
    for li, (idx, moe) in enumerate(layers):
        if algorithm == "skipall":
            w = SkipAllWrapper(moe, cache, li, N, top_k)
        elif algorithm == "alg2_v2":
            w = Alg2V2Wrapper(moe, cache, li, N, top_k)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        wrappers.append(w)

    total_len = prompt_len + k_max + 1
    step_alphas = [[] for _ in range(k_max)]
    step_miss_rates = [[] for _ in range(k_max)]
    step_crit_miss = [[] for _ in range(k_max)]
    n_seq = 0

    # Gate hooks for miss rate tracking (on original model during target pass)
    gate_bufs = [None] * L

    for prompt in tqdm(prompts, desc=f"  {algorithm} K=1..{k_max}", leave=False):
        prompt = prompt.to(device)
        T = prompt.size(1)
        if T < total_len:
            continue
        n_seq += 1
        inp = prompt[:, :total_len]

        # Target logits (full model)
        restore_model(model, moe_attr, originals, moe_indices)
        tgt_logits = model(inp, use_cache=False).logits[0].float()

        # Prefill
        restore_model(model, moe_attr, originals, moe_indices)
        prefill_out = model(prompt[:, :prompt_len], use_cache=True)
        draft_kv = copy_kv(prefill_out.past_key_values)

        # Install wrappers for draft
        patch_model(model, moe_attr, wrappers, moe_indices)

        # Install gate hooks to track miss rates
        hooks = []
        for li, (idx, moe_orig) in enumerate(layers):
            if hasattr(moe_orig, "gate"):
                def _ghook(li_):
                    def h(m, inp_, out_):
                        gate_bufs[li_] = out_.detach().float().cpu()
                    return h
                hooks.append(moe_orig.gate.register_forward_hook(_ghook(li)))

        for step in range(k_max):
            pos = prompt_len + step
            if pos + 1 >= T:
                break

            for i in range(L):
                gate_bufs[i] = None

            tok = prompt[:, pos:pos + 1]
            out_d = model(tok, past_key_values=draft_kv, use_cache=True)
            dlogit = out_d.logits[0, 0, :].float()
            draft_kv = out_d.past_key_values

            a = alpha_tv(tgt_logits[pos], dlogit)
            if a is not None:
                step_alphas[step].append(a)

            # Compute miss rates from gate outputs
            n_layers_ok = 0
            total_miss = 0
            top1_miss = 0
            top2_miss = 0
            for li in range(L):
                if gate_bufs[li] is None:
                    continue
                g = gate_bufs[li]
                if g.dim() == 3:
                    g = g[0]
                if g.size(0) == 0:
                    continue
                g_tok = g[-1]
                w = torch.softmax(g_tok, dim=-1)
                _, topk_i = torch.topk(w, top_k)
                topk_list = topk_i.tolist()
                n_layers_ok += 1
                n_miss = sum(1 for e in topk_list if not cache.is_cached(li, e))
                total_miss += n_miss
                if not cache.is_cached(li, topk_list[0]):
                    top1_miss += 1
                if top_k >= 2 and not cache.is_cached(li, topk_list[1]):
                    top2_miss += 1

            if n_layers_ok > 0:
                step_miss_rates[step].append(
                    total_miss / (n_layers_ok * top_k))
                step_crit_miss[step].append(
                    (top1_miss + top2_miss) / (2 * n_layers_ok))

        for h in hooks:
            h.remove()
        free_kv(draft_kv)

    restore_model(model, moe_attr, originals, moe_indices)

    # Aggregate
    mean_alphas = [float(np.mean(b)) if b else float("nan") for b in step_alphas]
    mean_miss = [float(np.mean(b)) if b else float("nan") for b in step_miss_rates]
    mean_crit = [float(np.mean(b)) if b else float("nan") for b in step_crit_miss]

    return {
        "alpha_per_step": mean_alphas,
        "miss_rate_per_step": mean_miss,
        "crit_miss_per_step": mean_crit,
        "n_sequences": n_seq,
        "raw_step_alphas": step_alphas,  # for variance analysis
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — T_cycle throughput model
# ─────────────────────────────────────────────────────────────────────────────

def compute_throughput(alpha_per_step: List[float],
                       t_draft_step: float = 2.0,
                       t_verify_layer: float = 0.5,
                       n_moe_layers: int = 48,
                       tau_expert: float = 1.5,
                       k_top: int = 8,
                       miss_rate_per_step: Optional[List[float]] = None,
                       prefetch_rate: int = 2,
                       ) -> dict:
    """Compute theoretical throughput for each K using simplified T_cycle model.

    T_cycle(K) = T_draft(K) + T_verify(K) + T_stall(K)
    T_draft(K) = K * t_draft_step
    T_verify(K) = n_moe_layers * t_verify_layer  (single-pass verify for K+1 tokens)
    T_stall(K) = estimate based on prefetch coverage

    Expected accepted tokens: E[A(K)] = Σ_{k=1}^{K} Π_{i=1}^{k} α_i  + 1 (bonus)

    Throughput = E[A(K)] / T_cycle(K)
    """
    results = {}
    K_max = len(alpha_per_step)

    for K in range(1, K_max + 1):
        # E[A(K)]
        cumulative_alpha = 1.0
        expected_accepted = 0.0
        for k in range(K):
            a = alpha_per_step[k]
            if not math.isfinite(a):
                break
            cumulative_alpha *= a
            expected_accepted += cumulative_alpha
        expected_accepted += 1.0  # bonus token from verify

        # T_draft
        t_draft = K * t_draft_step

        # T_verify (simplified: single forward pass over K+1 tokens)
        t_verify = n_moe_layers * t_verify_layer

        # T_stall: estimated from miss rate and prefetch coverage
        if miss_rate_per_step is not None and K <= len(miss_rate_per_step):
            avg_miss = np.mean([mr for mr in miss_rate_per_step[:K]
                                if math.isfinite(mr)])
            # During K draft steps, prefetcher loads prefetch_rate experts/step
            # Total prefetched: K * prefetch_rate
            # Total needed for verify: n_moe_layers * k_top * avg_miss
            needed = n_moe_layers * k_top * avg_miss
            prefetched = min(K * prefetch_rate, needed)
            remaining = max(0, needed - prefetched)
            t_stall = remaining * tau_expert
        else:
            # Conservative: assume no prefetch benefit
            t_stall = n_moe_layers * k_top * 0.3 * tau_expert  # rough estimate

        t_cycle = t_draft + t_verify + t_stall
        throughput = expected_accepted / t_cycle if t_cycle > 0 else 0

        results[K] = {
            "K": K,
            "expected_accepted": expected_accepted,
            "t_draft_ms": t_draft,
            "t_verify_ms": t_verify,
            "t_stall_ms": t_stall,
            "t_cycle_ms": t_cycle,
            "throughput_tok_per_ms": throughput,
            "cumulative_alpha": cumulative_alpha,
        }

    # Find K*
    best_K = max(results, key=lambda k: results[k]["throughput_tok_per_ms"])
    return results, best_K


# ─────────────────────────────────────────────────────────────────────────────
# Section 8 — Level-1 threshold signal analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_threshold_signals(
    alpha_per_step, miss_per_step, crit_per_step, k_star,
    alpha_thresholds=(0.5, 0.6, 0.7),
    miss_thresholds=(0.3, 0.4, 0.5),
    crit_thresholds=(0.1, 0.2, 0.3),
) -> dict:
    """Test whether threshold signals would trigger stop near K*.

    For each signal type × threshold, find the first step where signal
    crosses threshold. Compare to K*.
    """
    results = {}

    for thr in alpha_thresholds:
        trigger_step = None
        for k, a in enumerate(alpha_per_step):
            if math.isfinite(a) and a < thr:
                trigger_step = k + 1
                break
        results[f"alpha_below_{thr}"] = {
            "threshold": thr,
            "trigger_step": trigger_step,
            "k_star": k_star,
            "delta": (trigger_step - k_star) if trigger_step else None,
        }

    for thr in miss_thresholds:
        trigger_step = None
        for k, m in enumerate(miss_per_step):
            if math.isfinite(m) and m > thr:
                trigger_step = k + 1
                break
        results[f"miss_above_{thr}"] = {
            "threshold": thr,
            "trigger_step": trigger_step,
            "k_star": k_star,
            "delta": (trigger_step - k_star) if trigger_step else None,
        }

    for thr in crit_thresholds:
        trigger_step = None
        for k, c in enumerate(crit_per_step):
            if math.isfinite(c) and c > thr:
                trigger_step = k + 1
                break
        results[f"crit_above_{thr}"] = {
            "threshold": thr,
            "trigger_step": trigger_step,
            "k_star": k_star,
            "delta": (trigger_step - k_star) if trigger_step else None,
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Section 9 — Main experiment
# ─────────────────────────────────────────────────────────────────────────────

ALGORITHMS = ["SkipAll", "Alg2_v2"]

def run_experiment(args):
    os.makedirs(args.outdir, exist_ok=True)

    print("Loading model...")
    dtype = getattr(torch, args.dtype) if hasattr(torch, args.dtype) else torch.float16
    model, tokenizer = load_model_and_tokenizer(args.model, args.device, dtype)
    moe_cfg = detect_moe_config(model)
    N, top_k, L = moe_cfg["num_experts"], moe_cfg["top_k"], moe_cfg["num_layers"]
    moe_attr = moe_cfg["moe_attr"]
    print(f"  MoE config: L={L}, N={N}, top_k={top_k}")

    calib_chunks = prepare_chunks(tokenizer, args.n_calib, args.seq_len, args.data_file)
    eval_chunks = prepare_chunks(tokenizer, args.n_eval,
                                 args.prompt_len + args.k_max + 2, args.data_file)

    # Phase 1: Calibration
    print("\n[Phase 1] Calibration...")
    act_freq = calibrate_freq(model, moe_attr, calib_chunks, args.device, N, top_k)

    all_results = {}

    for ratio in args.cache_ratios:
        print(f"\n{'='*60}")
        print(f"  Cache ratio = {ratio}")
        print(f"{'='*60}")

        cache = SimulatedCache(L, N, ratio, act_freq)
        ratio_results = {}

        for alg_name in ALGORITHMS:
            alg_key = alg_name.lower()
            print(f"\n  Algorithm: {alg_name}")

            # Evaluate at K=1..k_max
            eval_result = eval_variable_k(
                model, moe_attr, eval_chunks, args.device,
                cache, N, top_k, L,
                args.prompt_len, args.k_max,
                algorithm=alg_key,
            )

            alpha_per_step = eval_result["alpha_per_step"]
            miss_per_step = eval_result["miss_rate_per_step"]
            crit_per_step = eval_result["crit_miss_per_step"]

            print(f"    α per step: {[f'{a:.3f}' for a in alpha_per_step]}")
            print(f"    miss per step: {[f'{m:.3f}' for m in miss_per_step]}")

            # Throughput model
            throughput_results, k_star = compute_throughput(
                alpha_per_step,
                t_draft_step=args.t_draft,
                t_verify_layer=args.t_verify_layer,
                n_moe_layers=L,
                tau_expert=args.tau_expert,
                k_top=top_k,
                miss_rate_per_step=miss_per_step,
                prefetch_rate=args.prefetch_rate,
            )
            print(f"    K* = {k_star}  "
                  f"(throughput = {throughput_results[k_star]['throughput_tok_per_ms']:.4f} tok/ms)")

            # Level-1 threshold analysis
            threshold_results = analyze_threshold_signals(
                alpha_per_step, miss_per_step, crit_per_step, k_star)

            # α decay fit: α(k) = α₀ · exp(-λ·k)
            valid_k = [k for k, a in enumerate(alpha_per_step) if math.isfinite(a)]
            valid_a = [alpha_per_step[k] for k in valid_k]
            fit_params = {}
            if len(valid_k) >= 3:
                try:
                    def exp_decay(k, a0, lam):
                        return a0 * np.exp(-lam * k)
                    popt, pcov = curve_fit(exp_decay,
                                          np.array(valid_k, dtype=np.float64),
                                          np.array(valid_a, dtype=np.float64),
                                          p0=[valid_a[0], 0.05], maxfev=5000)
                    fit_params = {"alpha0": float(popt[0]), "lambda": float(popt[1])}
                    fitted = [exp_decay(k, *popt) for k in valid_k]
                    rmse = float(np.sqrt(np.mean((np.array(valid_a) - np.array(fitted)) ** 2)))
                    fit_params["rmse"] = rmse
                    print(f"    Decay fit: α₀={popt[0]:.4f}, λ={popt[1]:.4f}, RMSE={rmse:.4f}")
                except Exception as e:
                    print(f"    Decay fit failed: {e}")
                    fit_params = {"error": str(e)}

            ratio_results[alg_name] = {
                "eval": {
                    "alpha_per_step": alpha_per_step,
                    "miss_rate_per_step": miss_per_step,
                    "crit_miss_per_step": crit_per_step,
                    "n_sequences": eval_result["n_sequences"],
                },
                "throughput": {str(k): v for k, v in throughput_results.items()},
                "k_star": k_star,
                "threshold_signals": threshold_results,
                "decay_fit": fit_params,
            }

        all_results[str(ratio)] = ratio_results

    # ── Save results ───────────────────────────────────────────────────────────
    print("\n[Phase 5] Saving results...")
    with open(os.path.join(args.outdir, "e3_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # ── Plots ──────────────────────────────────────────────────────────────────

    # Plot 1: α decay per (algo, ratio)
    fig, axes = plt.subplots(1, len(args.cache_ratios),
                             figsize=(6 * len(args.cache_ratios), 5))
    if len(args.cache_ratios) == 1:
        axes = [axes]
    colors = {"SkipAll": "#888780", "Alg2_v2": "#1D9E75"}
    for ax, ratio in zip(axes, args.cache_ratios):
        key = str(ratio)
        if key not in all_results:
            continue
        for alg_name in ALGORITHMS:
            if alg_name not in all_results[key]:
                continue
            alphas = all_results[key][alg_name]["eval"]["alpha_per_step"]
            ks = list(range(1, len(alphas) + 1))
            ax.plot(ks, alphas, marker="o", label=alg_name,
                    color=colors.get(alg_name, "gray"))
            k_star = all_results[key][alg_name]["k_star"]
            ax.axvline(k_star, color=colors.get(alg_name, "gray"),
                       linestyle="--", alpha=0.5)
        ax.set_xlabel("Draft step k")
        ax.set_ylabel("α(k)")
        ax.set_title(f"r = {ratio}")
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "alpha_decay.png"), dpi=150)
    plt.close()

    # Plot 2: Throughput vs K
    fig, axes = plt.subplots(1, len(args.cache_ratios),
                             figsize=(6 * len(args.cache_ratios), 5))
    if len(args.cache_ratios) == 1:
        axes = [axes]
    for ax, ratio in zip(axes, args.cache_ratios):
        key = str(ratio)
        if key not in all_results:
            continue
        for alg_name in ALGORITHMS:
            if alg_name not in all_results[key]:
                continue
            tp = all_results[key][alg_name]["throughput"]
            ks = sorted([int(k) for k in tp.keys()])
            vals = [tp[str(k)]["throughput_tok_per_ms"] for k in ks]
            ax.plot(ks, vals, marker="s", label=alg_name,
                    color=colors.get(alg_name, "gray"))
            k_star = all_results[key][alg_name]["k_star"]
            ax.axvline(k_star, color=colors.get(alg_name, "gray"),
                       linestyle="--", alpha=0.5)
        ax.set_xlabel("Draft length K")
        ax.set_ylabel("Throughput (tok/ms)")
        ax.set_title(f"r = {ratio}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "throughput_vs_k.png"), dpi=150)
    plt.close()

    # Plot 3: T_cycle breakdown
    fig, axes = plt.subplots(len(ALGORITHMS), len(args.cache_ratios),
                             figsize=(6 * len(args.cache_ratios),
                                      4 * len(ALGORITHMS)))
    if len(args.cache_ratios) == 1 and len(ALGORITHMS) == 1:
        axes = np.array([[axes]])
    elif len(args.cache_ratios) == 1:
        axes = axes.reshape(-1, 1)
    elif len(ALGORITHMS) == 1:
        axes = axes.reshape(1, -1)

    for ai, alg_name in enumerate(ALGORITHMS):
        for ri, ratio in enumerate(args.cache_ratios):
            ax = axes[ai, ri]
            key = str(ratio)
            if key not in all_results or alg_name not in all_results[key]:
                continue
            tp = all_results[key][alg_name]["throughput"]
            ks = sorted([int(k) for k in tp.keys()])
            t_draft = [tp[str(k)]["t_draft_ms"] for k in ks]
            t_verify = [tp[str(k)]["t_verify_ms"] for k in ks]
            t_stall = [tp[str(k)]["t_stall_ms"] for k in ks]

            ax.bar(ks, t_draft, label="T_draft", color="#4CAF50")
            ax.bar(ks, t_verify, bottom=t_draft, label="T_verify", color="#2196F3")
            ax.bar(ks, t_stall,
                   bottom=[a + b for a, b in zip(t_draft, t_verify)],
                   label="T_stall", color="#FF9800")
            ax.set_xlabel("K")
            ax.set_ylabel("T_cycle (ms)")
            ax.set_title(f"{alg_name}, r={ratio}")
            ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "tcycle_breakdown.png"), dpi=150)
    plt.close()

    # Summary table
    print(f"\n{'─'*70}")
    print(f"{'Summary':^70}")
    print(f"{'─'*70}")
    print(f"{'Algo':<12} {'Ratio':<8} {'K*':<5} {'Throughput':<12} {'α_decay_λ':<10}")
    for ratio in args.cache_ratios:
        key = str(ratio)
        if key not in all_results:
            continue
        for alg_name in ALGORITHMS:
            if alg_name not in all_results[key]:
                continue
            r = all_results[key][alg_name]
            tp = r["throughput"][str(r["k_star"])]["throughput_tok_per_ms"]
            lam = r["decay_fit"].get("lambda", float("nan"))
            print(f"{alg_name:<12} {ratio:<8.2f} {r['k_star']:<5} "
                  f"{tp:<12.4f} {lam:<10.4f}")

    print(f"\nResults saved to {args.outdir}/")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="E3: Dynamic K analysis")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data_file", default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--cache_ratios", nargs="+", type=float, default=[0.75, 0.50, 0.25])
    parser.add_argument("--k_max", type=int, default=12)
    parser.add_argument("--n_calib", type=int, default=8)
    parser.add_argument("--n_eval", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--prompt_len", type=int, default=128)
    # Latency parameters (Qwen3-30B-A3B on PCIe 4.0)
    parser.add_argument("--t_draft", type=float, default=2.0,
                        help="Draft step latency (ms), pure GPU with rerouting")
    parser.add_argument("--t_verify_layer", type=float, default=0.5,
                        help="Verify per-layer latency (ms)")
    parser.add_argument("--tau_expert", type=float, default=1.5,
                        help="Single expert PCIe transfer time (ms)")
    parser.add_argument("--prefetch_rate", type=int, default=2,
                        help="Experts prefetched per draft step")
    parser.add_argument("--outdir", default="./results_e3")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
