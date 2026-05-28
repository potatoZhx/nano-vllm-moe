"""
top12_protection_eval.py  (Experiment E2)
=========================================
Quantify the impact of top-1/2 expert cache misses on acceptance rate,
and verify the benefit of a protection mechanism.

Core questions:
  Q1. How strongly does CriticalMissRate (top-1/2 miss) correlate with per-step α?
  Q2. Does adding top-1/2 protection to Alg2_v2 improve seq_α?
  Q3. What is the optimal protection weight threshold w_protect?

Method:
  1. Decode-mode simulation: prefill → draft K steps with rerouting wrappers
  2. Per step, per layer: record whether top-1 and top-2 are in cache
  3. Compute CriticalMissRate(k) = fraction of layers with top-1 or top-2 miss
  4. Correlate CriticalMissRate with per-step α
  5. Compare Alg2_v2 with and without top-1/2 cache pin protection

Standalone — no dependency on nano-vllm-moe runtime.

Usage:
  python top12_protection_eval.py \
      --model /path/to/Qwen3-30B-A3B-Base \
      --data_file ../wikitext2_test.txt \
      --cache_ratios 0.75 0.50 0.25 \
      --draft_len 8 --prompt_len 128 \
      --n_calib 8 --n_eval 16 --outdir ./results_e2
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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Model & data utilities (shared with draft_decode_eval_v2)
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
# Section 2 — Simulated cache
# ─────────────────────────────────────────────────────────────────────────────

class SimulatedCache:
    """LFU warm-start: cache the S most frequently activated experts per layer."""
    def __init__(self, L: int, N: int, ratio: float,
                 act_freq: Optional[np.ndarray] = None,
                 pin_experts: Optional[Dict[int, set]] = None):
        """
        pin_experts: optional {layer_idx: set_of_expert_ids} to force into cache.
        """
        self.N = N
        self.cache_size = max(1, int(N * ratio))
        self.sets: List[set] = []
        for l in range(L):
            if act_freq is not None:
                top = np.argsort(act_freq[l])[::-1][:self.cache_size]
                base = set(top.tolist())
            else:
                base = set(random.sample(range(N), self.cache_size))
            # Pin top-1/2 experts if provided
            if pin_experts and l in pin_experts:
                for eid in pin_experts[l]:
                    if eid not in base and len(base) < N:
                        # Remove lowest-freq expert and add pinned one
                        if len(base) >= self.cache_size:
                            if act_freq is not None:
                                freqs = [(act_freq[l, e], e) for e in base
                                         if e not in pin_experts.get(l, set())]
                                if freqs:
                                    _, victim = min(freqs)
                                    base.discard(victim)
                        base.add(eid)
            self.sets.append(base)

    def mask(self, layer: int, device) -> torch.Tensor:
        m = torch.zeros(self.N, dtype=torch.bool, device=device)
        m[list(self.sets[layer])] = True
        return m

    def is_cached(self, layer: int, expert_id: int) -> bool:
        return expert_id in self.sets[layer]


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Calibration (lightweight: activation freq + top-1/2 freq)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CalibData:
    act_freq: np.ndarray      # [L, N]
    rank1_freq: np.ndarray    # [L, N]
    rank2_freq: np.ndarray    # [L, N]
    L: int
    N: int
    top_k: int


@torch.no_grad()
def calibrate(model, moe_attr, chunks, device, N, top_k):
    layers = get_moe_layers(model, moe_attr)
    L = len(layers)

    act_freq = np.zeros((L, N), dtype=np.float64)
    rank1_freq = np.zeros((L, N), dtype=np.float64)
    rank2_freq = np.zeros((L, N), dtype=np.float64)
    total_tokens = 0

    gate_outputs = [None] * L
    hooks = []

    def _ghook(li):
        def h(m, inp, out):
            gate_outputs[li] = out.detach().float().cpu()
        return h

    for li, (idx, moe) in enumerate(layers):
        if hasattr(moe, "gate"):
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
            w = torch.softmax(g, dim=-1)
            _, topk_i = torch.topk(w, top_k, dim=-1)
            topk_i_np = topk_i.numpy()
            for t in range(g.size(0)):
                for rank, ei in enumerate(topk_i_np[t]):
                    act_freq[li, ei] += 1
                    if rank == 0:
                        rank1_freq[li, ei] += 1
                    elif rank == 1:
                        rank2_freq[li, ei] += 1

    for h in hooks:
        h.remove()

    if total_tokens > 0:
        act_freq /= total_tokens
        rank1_freq /= total_tokens
        rank2_freq /= total_tokens

    return CalibData(
        act_freq=act_freq, rank1_freq=rank1_freq,
        rank2_freq=rank2_freq, L=L, N=N, top_k=top_k,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Identify top-1/2 frequently-selected experts for pinning
# ─────────────────────────────────────────────────────────────────────────────

def identify_critical_experts(calib: CalibData, w_protect: float = 0.15):
    """Identify experts whose rank-1 or rank-2 frequency exceeds w_protect.

    Returns: {layer_idx: set_of_expert_ids}
    """
    pin = {}
    for l in range(calib.L):
        critical = set()
        for j in range(calib.N):
            if calib.rank1_freq[l, j] >= w_protect:
                critical.add(j)
            if calib.rank2_freq[l, j] >= w_protect:
                critical.add(j)
        if critical:
            pin[l] = critical
    return pin


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Per-step evaluation with CriticalMissRate tracking
# ─────────────────────────────────────────────────────────────────────────────

def alpha_tv(target: torch.Tensor, draft: torch.Tensor) -> Optional[float]:
    """α = Σ_v min(p_t(v), p_d(v)).  Returns None if NaN/Inf."""
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


@torch.no_grad()
def evaluate_with_critical_tracking(
    model, moe_attr, prompts, device,
    cache: SimulatedCache, N: int, top_k: int, L: int,
    prompt_len: int = 128, draft_len: int = 8,
    use_skipall: bool = True,
) -> dict:
    """
    Run decode-mode simulation and track CriticalMissRate per step.

    For simplicity, this uses SkipAll rerouting (zero out miss experts).
    The key output is per-step (alpha, CriticalMissRate, overall miss rate).
    """
    layers = get_moe_layers(model, moe_attr)
    total_len = prompt_len + draft_len + 1

    step_records = [[] for _ in range(draft_len)]
    # Each record: (alpha, critical_miss_rate, overall_miss_rate,
    #               top1_miss_count, top2_miss_count)

    for prompt in tqdm(prompts, desc="Eval critical miss"):
        prompt = prompt.to(device)
        T = prompt.size(1)
        if T < total_len:
            continue

        inp = prompt[:, :total_len]

        # Target logits (full model, no rerouting)
        tgt_logits = model(inp, use_cache=False).logits[0].float()  # [T, V]

        # Prefill with full model (to get KV cache)
        prefill_out = model(prompt[:, :prompt_len], use_cache=True)
        draft_kv = copy_kv(prefill_out.past_key_values)

        # Collect gate outputs per layer for all positions
        gate_bufs = [None] * L
        hooks = []
        for li, (idx, moe) in enumerate(layers):
            if hasattr(moe, "gate"):
                def _ghook(li_):
                    def h(m, inp_, out_):
                        gate_bufs[li_] = out_.detach().float().cpu()
                    return h
                hooks.append(moe.gate.register_forward_hook(_ghook(li)))

        # Draft steps: one token at a time
        for step in range(draft_len):
            pos = prompt_len + step
            if pos + 1 >= T:
                break

            tok = prompt[:, pos:pos + 1]

            # Clear gate buffers
            for i in range(L):
                gate_bufs[i] = None

            out_d = model(tok, past_key_values=draft_kv, use_cache=True)
            dlogit = out_d.logits[0, 0, :].float()
            draft_kv = out_d.past_key_values

            # Compute alpha
            alpha = alpha_tv(tgt_logits[pos], dlogit)

            # Compute CriticalMissRate from gate outputs
            top1_miss_count = 0
            top2_miss_count = 0
            total_miss_count = 0
            n_layers_counted = 0

            for li in range(L):
                if gate_bufs[li] is None:
                    continue
                g = gate_bufs[li]
                if g.dim() == 3:
                    g = g[0]
                if g.size(0) == 0:
                    continue
                # Take last token (only one token in draft step)
                g_tok = g[-1]
                w = torch.softmax(g_tok, dim=-1)
                _, topk_i = torch.topk(w, top_k)
                topk_list = topk_i.tolist()

                n_layers_counted += 1
                n_miss = sum(1 for e in topk_list if not cache.is_cached(li, e))
                total_miss_count += n_miss

                # Top-1 and top-2
                if not cache.is_cached(li, topk_list[0]):
                    top1_miss_count += 1
                if top_k >= 2 and not cache.is_cached(li, topk_list[1]):
                    top2_miss_count += 1

            if n_layers_counted > 0 and alpha is not None:
                critical_miss_rate = (top1_miss_count + top2_miss_count) / (
                    2 * n_layers_counted)
                overall_miss_rate = total_miss_count / (
                    n_layers_counted * top_k)
                step_records[step].append({
                    "alpha": alpha,
                    "critical_miss_rate": critical_miss_rate,
                    "overall_miss_rate": overall_miss_rate,
                    "top1_miss_frac": top1_miss_count / n_layers_counted,
                    "top2_miss_frac": top2_miss_count / n_layers_counted,
                })

        for h in hooks:
            h.remove()
        free_kv(draft_kv)

    return step_records


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Rerouting wrappers (SkipAll + Alg2_v2 with/without protection)
# ─────────────────────────────────────────────────────────────────────────────

MISS_GATE = 0.25
GAMMA0 = 4.0

class SkipAllWrapper(nn.Module):
    """Zero weight for missed experts; renormalize over hits.
    Implemented as a gate-output-modifying wrapper."""

    def __init__(self, orig_moe, cache: SimulatedCache, layer_idx: int,
                 N: int, top_k: int):
        super().__init__()
        self.orig = orig_moe
        self.cache = cache
        self.l = layer_idx
        self.N = N
        self.top_k = top_k

    def forward(self, hidden_states: torch.Tensor):
        B, T, D = hidden_states.shape
        h = hidden_states.view(-1, D)
        dev = h.device

        logits = self.orig.gate(h)
        probs = F.softmax(logits, dim=-1)
        rw, ri = probs.topk(self.top_k, dim=-1)

        cm = self.cache.mask(self.l, dev)
        hit = cm[ri]
        fw = rw * hit.float()

        # Fallback for all-zero rows
        row_sum = fw.sum(-1)
        if (row_sum == 0).any():
            empty = (row_sum == 0)
            cl = sorted(self.cache.sets[self.l])
            fb = cl[0] if cl else 0
            fi_tmp = ri.clone()
            fi_tmp[empty, 0] = fb
            fw[empty, 0] = 1.0
            ri = fi_tmp

        fw = (fw / fw.sum(-1, keepdim=True).clamp(1e-9)).to(h.dtype)

        # scatter-add forward
        N = self.N
        wb = torch.zeros(B * T, N, dtype=h.dtype, device=dev)
        for slot in range(ri.size(1)):
            wb.scatter_add_(1, ri[:, slot:slot + 1], fw[:, slot:slot + 1])

        out = torch.zeros(B * T, D, dtype=h.dtype, device=dev)
        active = wb.any(dim=0).nonzero(as_tuple=False).squeeze(-1)
        for e_t in active:
            ei = int(e_t.item())
            we = wb[:, ei]
            mask = we > 0
            if not mask.any():
                continue
            eo = self.orig.experts[ei](h[mask])
            if isinstance(eo, tuple):
                eo = eo[0]
            out[mask] += (eo * we[mask].unsqueeze(-1)).to(h.dtype)

        if hasattr(self.orig, "shared_expert") and self.orig.shared_expert is not None:
            sh = self.orig.shared_expert(h)
            if hasattr(self.orig, "shared_expert_gate"):
                sh = torch.sigmoid(self.orig.shared_expert_gate(h)) * sh
            out = out + sh.to(out.dtype)

        return out.view(B, T, D), logits


class Alg2V2Wrapper(nn.Module):
    """Alg2_v2: entropy-conditioned pre-routing bias + miss-rate gate.
    With optional top-1/2 protection via cache pinning."""

    def __init__(self, orig_moe, cache: SimulatedCache, layer_idx: int,
                 N: int, top_k: int, gamma0: float = GAMMA0,
                 miss_low: float = MISS_GATE, miss_high: float = 0.50,
                 protect_top12: bool = False):
        super().__init__()
        self.orig = orig_moe
        self.cache = cache
        self.l = layer_idx
        self.N = N
        self.top_k = top_k
        self.gamma0 = gamma0
        self.miss_low = miss_low
        self.miss_high = miss_high
        self.protect_top12 = protect_top12

    def forward(self, hidden_states: torch.Tensor):
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

        # Top-1 protection: restore original top-1 if displaced
        orig_top1 = ri[:, 0]
        for t_i in range(B * T):
            ot1 = int(orig_top1[t_i].item())
            if not cm[ot1] and ot1 not in ni[t_i].tolist():
                ni[t_i, -1] = ot1

        # Top-2 protection (optional)
        if self.protect_top12 and self.top_k >= 2:
            orig_top2 = ri[:, 1]
            for t_i in range(B * T):
                ot2 = int(orig_top2[t_i].item())
                if not cm[ot2] and ot2 not in ni[t_i].tolist():
                    ni[t_i, -2] = ot2

        nw = F.softmax(logits.gather(-1, ni).float(), dim=-1)
        new_hit = cm[ni]
        nw = nw * new_hit.float()

        # Fallback
        row_sum = nw.sum(-1)
        if (row_sum == 0).any():
            empty = (row_sum == 0)
            cl = sorted(self.cache.sets[self.l])
            fb = cl[0] if cl else 0
            ni[empty, 0] = fb
            nw[empty, 0] = 1.0

        nw = (nw / nw.sum(-1, keepdim=True).clamp(1e-9)).to(h.dtype)

        # scatter-add forward
        wb = torch.zeros(B * T, self.N, dtype=h.dtype, device=dev)
        for slot in range(ni.size(1)):
            wb.scatter_add_(1, ni[:, slot:slot + 1], nw[:, slot:slot + 1])

        out = torch.zeros(B * T, D, dtype=h.dtype, device=dev)
        active = wb.any(dim=0).nonzero(as_tuple=False).squeeze(-1)
        for e_t in active:
            ei = int(e_t.item())
            we = wb[:, ei]
            mask = we > 0
            if not mask.any():
                continue
            eo = self.orig.experts[ei](h[mask])
            if isinstance(eo, tuple):
                eo = eo[0]
            out[mask] += (eo * we[mask].unsqueeze(-1)).to(h.dtype)

        if hasattr(self.orig, "shared_expert") and self.orig.shared_expert is not None:
            sh = self.orig.shared_expert(h)
            if hasattr(self.orig, "shared_expert_gate"):
                sh = torch.sigmoid(self.orig.shared_expert_gate(h)) * sh
            out = out + sh.to(out.dtype)

        return out.view(B, T, D), logits


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — Rerouting eval with wrapper patching
# ─────────────────────────────────────────────────────────────────────────────

def patch_model(model, moe_attr, wrappers, moe_indices):
    for li, (idx, _) in enumerate(moe_indices):
        layer = model.model.layers[idx]
        setattr(layer, moe_attr, wrappers[li])


def restore_model(model, moe_attr, originals, moe_indices):
    for li, (idx, _) in enumerate(moe_indices):
        layer = model.model.layers[idx]
        setattr(layer, moe_attr, originals[li])


@torch.no_grad()
def eval_rerouting_alpha(
    model, moe_attr, prompts, device, cache, N, top_k,
    prompt_len, draft_len, algorithm="skipall", protect_top12=False,
) -> dict:
    """Evaluate seq_alpha for a given rerouting algorithm."""
    layers = get_moe_layers(model, moe_attr)
    L = len(layers)
    moe_indices = layers

    # Save originals
    originals = []
    for li, (idx, moe) in enumerate(layers):
        originals.append(moe)

    # Build wrappers
    wrappers = []
    for li, (idx, moe) in enumerate(layers):
        if algorithm == "skipall":
            w = SkipAllWrapper(moe, cache, li, N, top_k)
        elif algorithm == "alg2_v2":
            w = Alg2V2Wrapper(moe, cache, li, N, top_k,
                              protect_top12=protect_top12)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        wrappers.append(w)

    total_len = prompt_len + draft_len + 1
    step_bkt = [[] for _ in range(draft_len)]
    n_seq = 0

    for prompt in tqdm(prompts, desc=f"Eval {algorithm} (protect={protect_top12})",
                       leave=False):
        prompt = prompt.to(device)
        T = prompt.size(1)
        if T < total_len:
            continue
        n_seq += 1

        inp = prompt[:, :total_len]

        # Target logits
        restore_model(model, moe_attr, originals, moe_indices)
        tgt_logits = model(inp, use_cache=False).logits[0].float()

        # Prefill
        restore_model(model, moe_attr, originals, moe_indices)
        prefill_out = model(prompt[:, :prompt_len], use_cache=True)
        draft_kv = copy_kv(prefill_out.past_key_values)

        # Install wrappers
        patch_model(model, moe_attr, wrappers, moe_indices)

        for step in range(draft_len):
            pos = prompt_len + step
            if pos + 1 >= T:
                break
            tok = prompt[:, pos:pos + 1]
            out_d = model(tok, past_key_values=draft_kv, use_cache=True)
            dlogit = out_d.logits[0, 0, :].float()
            draft_kv = out_d.past_key_values

            a = alpha_tv(tgt_logits[pos], dlogit)
            if a is not None:
                step_bkt[step].append(a)

        free_kv(draft_kv)

    restore_model(model, moe_attr, originals, moe_indices)

    means = [float(np.mean(b)) if b else float("nan") for b in step_bkt]
    seq_alpha = float(np.nanmean(means)) if any(math.isfinite(v) for v in means) \
        else float("nan")
    decay = means[-1] / means[0] if (means and math.isfinite(means[-1])
                                       and math.isfinite(means[0])
                                       and means[0] > 0) else float("nan")

    return {
        "seq_alpha": seq_alpha,
        "alpha_per_step": means,
        "alpha_decay": decay,
        "n_sequences": n_seq,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 8 — Main experiment
# ─────────────────────────────────────────────────────────────────────────────

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
                                 args.prompt_len + args.draft_len + 2, args.data_file)

    # Phase 1: Calibration
    print("\n[Phase 1] Calibration...")
    calib = calibrate(model, moe_attr, calib_chunks, args.device, N, top_k)

    all_results = {}

    for ratio in args.cache_ratios:
        print(f"\n{'='*60}")
        print(f"  Cache ratio = {ratio}")
        print(f"{'='*60}")

        ratio_results = {}

        # Build baseline cache (no protection)
        cache_base = SimulatedCache(L, N, ratio, calib.act_freq)

        # Phase 2: CriticalMissRate correlation
        print("\n[Phase 2] CriticalMissRate tracking...")
        step_records = evaluate_with_critical_tracking(
            model, moe_attr, eval_chunks, args.device,
            cache_base, N, top_k, L,
            prompt_len=args.prompt_len, draft_len=args.draft_len,
        )

        # Aggregate correlation data
        all_alphas = []
        all_crit_miss = []
        all_top1_miss = []
        all_top2_miss = []
        all_overall_miss = []

        for step in range(args.draft_len):
            for rec in step_records[step]:
                all_alphas.append(rec["alpha"])
                all_crit_miss.append(rec["critical_miss_rate"])
                all_top1_miss.append(rec["top1_miss_frac"])
                all_top2_miss.append(rec["top2_miss_frac"])
                all_overall_miss.append(rec["overall_miss_rate"])

        if len(all_alphas) > 10:
            rho_crit, p_crit = stats.spearmanr(all_alphas, all_crit_miss)
            rho_top1, p_top1 = stats.spearmanr(all_alphas, all_top1_miss)
            rho_top2, p_top2 = stats.spearmanr(all_alphas, all_top2_miss)
            rho_overall, p_overall = stats.spearmanr(all_alphas, all_overall_miss)
        else:
            rho_crit = rho_top1 = rho_top2 = rho_overall = float("nan")
            p_crit = p_top1 = p_top2 = p_overall = float("nan")

        print(f"  Correlation (α vs CriticalMissRate):  ρ = {rho_crit:.4f}, p = {p_crit:.4e}")
        print(f"  Correlation (α vs Top1MissFrac):      ρ = {rho_top1:.4f}, p = {p_top1:.4e}")
        print(f"  Correlation (α vs Top2MissFrac):      ρ = {rho_top2:.4f}, p = {p_top2:.4e}")
        print(f"  Correlation (α vs OverallMissRate):    ρ = {rho_overall:.4f}, p = {p_overall:.4e}")

        ratio_results["correlations"] = {
            "critical_miss": {"rho": float(rho_crit), "p": float(p_crit)},
            "top1_miss": {"rho": float(rho_top1), "p": float(p_top1)},
            "top2_miss": {"rho": float(rho_top2), "p": float(p_top2)},
            "overall_miss": {"rho": float(rho_overall), "p": float(p_overall)},
        }

        # Phase 3: Protection comparison
        print("\n[Phase 3] Protection mechanism comparison...")
        comparison = {}

        for w_protect in args.w_protects:
            print(f"\n  w_protect = {w_protect}")
            pin_experts = identify_critical_experts(calib, w_protect)
            n_pinned = sum(len(v) for v in pin_experts.values())
            print(f"    Pinned experts: {n_pinned} across {len(pin_experts)} layers")

            # Cache with protection
            cache_protected = SimulatedCache(
                L, N, ratio, calib.act_freq, pin_experts=pin_experts)

            # Eval: SkipAll baseline (no protection)
            res_skipall_base = eval_rerouting_alpha(
                model, moe_attr, eval_chunks, args.device,
                cache_base, N, top_k,
                args.prompt_len, args.draft_len,
                algorithm="skipall", protect_top12=False)

            # Eval: Alg2_v2 without protection
            res_alg2_base = eval_rerouting_alpha(
                model, moe_attr, eval_chunks, args.device,
                cache_base, N, top_k,
                args.prompt_len, args.draft_len,
                algorithm="alg2_v2", protect_top12=False)

            # Eval: Alg2_v2 with cache pin protection
            res_alg2_pin = eval_rerouting_alpha(
                model, moe_attr, eval_chunks, args.device,
                cache_protected, N, top_k,
                args.prompt_len, args.draft_len,
                algorithm="alg2_v2", protect_top12=False)

            # Eval: Alg2_v2 with rerouting-level top-1/2 restore
            res_alg2_reroute = eval_rerouting_alpha(
                model, moe_attr, eval_chunks, args.device,
                cache_base, N, top_k,
                args.prompt_len, args.draft_len,
                algorithm="alg2_v2", protect_top12=True)

            # Eval: Both protections combined
            res_alg2_both = eval_rerouting_alpha(
                model, moe_attr, eval_chunks, args.device,
                cache_protected, N, top_k,
                args.prompt_len, args.draft_len,
                algorithm="alg2_v2", protect_top12=True)

            comp_entry = {
                "w_protect": w_protect,
                "n_pinned": n_pinned,
                "skipall_base": res_skipall_base,
                "alg2_base": res_alg2_base,
                "alg2_cache_pin": res_alg2_pin,
                "alg2_reroute_protect": res_alg2_reroute,
                "alg2_both": res_alg2_both,
            }
            comparison[str(w_protect)] = comp_entry

            delta_pin = res_alg2_pin["seq_alpha"] - res_alg2_base["seq_alpha"]
            delta_reroute = res_alg2_reroute["seq_alpha"] - res_alg2_base["seq_alpha"]
            delta_both = res_alg2_both["seq_alpha"] - res_alg2_base["seq_alpha"]
            print(f"    SkipAll base:          seq_α = {res_skipall_base['seq_alpha']:.4f}")
            print(f"    Alg2_v2 base:          seq_α = {res_alg2_base['seq_alpha']:.4f}")
            print(f"    Alg2_v2 + cache pin:   seq_α = {res_alg2_pin['seq_alpha']:.4f}  (Δ={delta_pin:+.4f})")
            print(f"    Alg2_v2 + reroute:     seq_α = {res_alg2_reroute['seq_alpha']:.4f}  (Δ={delta_reroute:+.4f})")
            print(f"    Alg2_v2 + both:        seq_α = {res_alg2_both['seq_alpha']:.4f}  (Δ={delta_both:+.4f})")

        ratio_results["protection_comparison"] = comparison
        all_results[str(ratio)] = ratio_results

    # ── Save results ───────────────────────────────────────────────────────────
    print("\n[Phase 4] Saving results...")

    with open(os.path.join(args.outdir, "e2_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # ── Plot: correlation scatter ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(args.cache_ratios),
                             figsize=(6 * len(args.cache_ratios), 5))
    if len(args.cache_ratios) == 1:
        axes = [axes]
    for ax, ratio in zip(axes, args.cache_ratios):
        key = str(ratio)
        if key not in all_results:
            continue
        corr = all_results[key].get("correlations", {})
        labels = ["CritMiss", "Top1Miss", "Top2Miss", "Overall"]
        rhos = [corr.get(k, {}).get("rho", 0) for k in
                ["critical_miss", "top1_miss", "top2_miss", "overall_miss"]]
        colors = ["#d32f2f", "#ff9800", "#fbc02d", "#888888"]
        ax.barh(labels, rhos, color=colors)
        ax.set_xlabel("Spearman ρ (with α)")
        ax.set_title(f"r = {ratio}")
        ax.set_xlim(-1, 0.2)
        ax.axvline(0, color="gray", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "correlation_bars.png"), dpi=150)
    plt.close()

    # ── Plot: protection comparison ────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(args.cache_ratios),
                             figsize=(6 * len(args.cache_ratios), 5))
    if len(args.cache_ratios) == 1:
        axes = [axes]
    for ax, ratio in zip(axes, args.cache_ratios):
        key = str(ratio)
        if key not in all_results:
            continue
        comp = all_results[key].get("protection_comparison", {})
        # Use first w_protect value for plotting
        if not comp:
            continue
        first_wp = list(comp.keys())[0]
        c = comp[first_wp]
        names = ["SkipAll", "Alg2 base", "Alg2+pin", "Alg2+reroute", "Alg2+both"]
        vals = [
            c["skipall_base"]["seq_alpha"],
            c["alg2_base"]["seq_alpha"],
            c["alg2_cache_pin"]["seq_alpha"],
            c["alg2_reroute_protect"]["seq_alpha"],
            c["alg2_both"]["seq_alpha"],
        ]
        colors = ["#888888", "#1D9E75", "#3266ad", "#d4640c", "#9c27b0"]
        ax.bar(range(len(names)), vals, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("seq_α")
        ax.set_title(f"r = {ratio} (w_protect = {first_wp})")
        ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "protection_comparison.png"), dpi=150)
    plt.close()

    print(f"\nResults saved to {args.outdir}/")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="E2: Top-1/2 protection eval")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data_file", default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--cache_ratios", nargs="+", type=float, default=[0.75, 0.50, 0.25])
    parser.add_argument("--w_protects", nargs="+", type=float, default=[0.10, 0.15, 0.20])
    parser.add_argument("--n_calib", type=int, default=8)
    parser.add_argument("--n_eval", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--prompt_len", type=int, default=128)
    parser.add_argument("--draft_len", type=int, default=8)
    parser.add_argument("--outdir", default="./results_e2")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
