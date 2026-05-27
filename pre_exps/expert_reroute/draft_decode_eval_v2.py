"""
draft_decode_eval_v2.py
=======================
Optimised draft-decode evaluation. Redesigns the algorithms that were
problematic in v1.

ROOT CAUSES of v1 failures (from three-ratio analysis)
-------------------------------------------------------
1. r=0.75: ALL substitution algorithms underperformed SkipAll.
   Reason: at low miss rate the 2 missing experts are the router's
   LOWEST-ranked selections (weight ~6-8% each).  Any substitute
   produces K/V vectors the router explicitly de-prioritised for this
   token.  These "stranger" vectors corrupt the KV cache over 8 steps
   more than simply omitting the experts (SkipAll).
   Fix → miss-rate gate: disable any substitution when per-token
   miss_rate ≤ MISS_GATE (default 0.25 = ≤2/8 misses).

2. Post-routing algorithms at r=0.25: most substitutes have cosine
   similarity ≈ 0.055 (off-diagonal mean), making them near-random
   replacements that corrupt hidden states.
   Fix → only substitute when the best available sim ≥ SIM_FLOOR.

3. Forward-pass double counting: when substitute j* already appears as
   a hit in the current token's top-k, the old slot-iteration loop
   called expert j* twice.
   Fix → scatter-add all weights into a [T, N] buffer then run each
   expert exactly once.

Algorithms
----------
  SkipAll        baseline: zero miss weights, renorm over hits
  Alg2_v2        entropy-conditioned pre-routing bias + miss-rate gate
  HybridCP_v2    bounded-pool pre-routing bias + miss-rate gate
                 + SkipAll fallback (no post-routing substitution)
  PostSub_v2     post-routing substitution, CondError table,
                 sim floor, miss-rate gate
  Alg2_PostSub   Alg2_v2 pre-routing  +  PostSub_v2 residual fallback

Usage
-----
  CUDA_VISIBLE_DEVICES=1 python draft_decode_eval_v2.py \
      --model /zx_data1/models/Qwen--Qwen3-30B-A3B-Base/ \
      --data_file /zx_data1/models/datasets/wikitext2_test.txt \
      --cache_ratios 0.75 0.5 0.25 \
      --draft_len 8 --prompt_len 128 \
      --n_prompts 32 --outdir ./results_v2

CUDA_VISIBLE_DEVICES=1 python draft_decode_eval_v2.py \
      --model /zx_data1/models/Qwen--Qwen3-30B-A3B-Base/ \
      --data_file /zx_data1/models/datasets/wikitext2_test.txt \
      --cache_ratios 0.75 0.5 0.25 \
      --draft_len 8 --prompt_len 128 \
      --n_calib 2 \
      --n_prompts 2 --outdir ./results_v2
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
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Hyper-parameters (tunable via constructor kwargs)
# ─────────────────────────────────────────────────────────────────────────────

MISS_GATE   = 0.25   # per-token miss_rate threshold; below → use SkipAll
SIM_FLOOR   = 0.40   # minimum cosine sim required for post-routing substitution
GAMMA0      = 4.0    # base entropy-bias strength for Alg2
J_FACTOR    = 3      # candidate pool for HybridCP: top J = J_FACTOR × top_k


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 – Model utilities
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
    N   = getattr(cfg, "num_experts", getattr(cfg, "n_routed_experts", None))
    k   = getattr(cfg, "num_experts_per_tok", getattr(cfg, "top_k", None))
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


def get_moe_layers(model, moe_attr) -> List[Tuple[int, nn.Module]]:
    return [(i, getattr(layer, moe_attr))
            for i, layer in enumerate(model.model.layers)
            if hasattr(getattr(layer, moe_attr, None), "experts")]


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 – Data utilities
# ─────────────────────────────────────────────────────────────────────────────

_SNIPPET = (
    "Wikipedia is a free online encyclopedia. Language models predict the next "
    "token given previous tokens. Mixture-of-Experts activates a subset of params. "
    "Expert caching stores frequently used experts in GPU memory. Speculative "
    "decoding generates draft tokens and verifies them in parallel. "
) * 100


def prepare_chunks(tokenizer, n: int, seq_len: int,
                   data_file: Optional[str] = None) -> List[torch.Tensor]:
    text = None
    if data_file:
        try:
            text = open(data_file, encoding="utf-8").read()
            print(f"  Loaded {len(text):,} chars from {data_file}.")
        except Exception as e:
            print(f"  ⚠ {e}")
    if text is None:
        try:
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-2-raw-v1",
                              split="test", trust_remote_code=True)
            text = "\n\n".join(ds["text"])
        except Exception as e:
            print(f"  ⚠ datasets unavailable ({e}); using built-in snippet.")
            text = _SNIPPET
    enc   = tokenizer(text, return_tensors="pt")["input_ids"]
    total = enc.size(1)
    chunks = [enc[:, s:s+seq_len]
              for s in range(0, total - seq_len, seq_len)][:n]
    if not chunks:
        raise RuntimeError(f"Text too short ({total} tokens) for seq_len={seq_len}.")
    if len(chunks) < n:
        chunks = (chunks * (n // len(chunks) + 1))[:n]
    random.shuffle(chunks)
    print(f"  {len(chunks)} chunks × {seq_len} tokens.")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 – Simulated (static) expert cache
# ─────────────────────────────────────────────────────────────────────────────

class SimulatedCache:
    """LFU warm-start: cache the S most frequently activated experts per layer."""
    def __init__(self, L: int, N: int, ratio: float,
                 act_freq: Optional[np.ndarray] = None):
        self.N          = N
        self.cache_size = max(1, int(N * ratio))
        self.sets: List[set] = []
        for l in range(L):
            if act_freq is not None:
                top = np.argsort(act_freq[l])[::-1][:self.cache_size]
                self.sets.append(set(top.tolist()))
            else:
                self.sets.append(set(
                    random.sample(range(N), self.cache_size)))

    def mask(self, layer: int, device) -> torch.Tensor:
        m = torch.zeros(self.N, dtype=torch.bool, device=device)
        m[list(self.sets[layer])] = True
        return m

    def cached_list(self, layer: int) -> List[int]:
        return sorted(self.sets[layer])


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 – Calibration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Calib:
    """Offline statistics needed by all algorithms."""
    # [L, N, N] float32 — combined similarity (cos 0.5 + coact 0.3 + corr 0.2)
    sim:       Optional[torch.Tensor] = None
    # [L, N, N] float32 — conditional replacement error (similarity form, exp(-D_cond))
    cond_sim:  Optional[torch.Tensor] = None
    # [L, N]   float32 — mean output norm / mean hidden norm (skip error proxy)
    skip_err:  Optional[torch.Tensor] = None
    # [L]      float32 — layer sensitivity [0,1]
    sens:      Optional[torch.Tensor] = None
    # [L, N]   float32 — activation frequency
    act_freq:  Optional[np.ndarray]  = None
    L: int = 0; N: int = 0


@torch.no_grad()
def calibrate(model, moe_attr: str, chunks: List[torch.Tensor],
              device: str, N: int) -> Calib:
    layers = get_moe_layers(model, moe_attr)
    L, D   = len(layers), model.config.hidden_size
    top_k  = getattr(model.config, "num_experts_per_tok",
                     getattr(model.config, "top_k", 4))
    print(f"[Calibration] L={L}, N={N}, chunks={len(chunks)}")

    # ── Accumulators ─────────────────────────────────────────────────────────
    sum_out  = [torch.zeros(N, D)   for _ in range(L)]
    cnt_out  = [torch.zeros(N)      for _ in range(L)]
    coact    = [np.zeros((N, N))    for _ in range(L)]
    act_cnt  = np.zeros((N,), dtype=np.float64)  # flattened; per-layer below
    act_cnt  = [np.zeros(N)         for _ in range(L)]
    g_sum    = [np.zeros(N)         for _ in range(L)]
    g_sum2   = [np.zeros(N)         for _ in range(L)]
    g_cross  = [np.zeros((N, N))    for _ in range(L)]
    g_n      = [0] * L
    ce_sum   = [np.zeros((N, N))    for _ in range(L)]  # cond replacement error
    ce_cnt   = [np.zeros(N)         for _ in range(L)]
    norm_sum = [np.zeros(N)         for _ in range(L)]
    w_sum    = [np.zeros(N)         for _ in range(L)]
    h_norm_s = np.zeros(L); h_norm_n = np.zeros(L, dtype=np.int64)
    weight_v = np.zeros(L)   # routing weight variance proxy for sensitivity
    freq     = np.zeros((L, N), dtype=np.float32)

    # ── Hooks ─────────────────────────────────────────────────────────────────
    bufs: List[Dict] = [{} for _ in range(L)]
    hidden_bufs: List[Optional[torch.Tensor]] = [None for _ in range(L)]
    selected_bufs: List[Optional[torch.Tensor]] = [None for _ in range(L)]
    fused_experts = [
        hasattr(mm.experts, "gate_up_proj") and getattr(mm.experts.gate_up_proj, "ndim", 0) == 3
        for _, mm in layers
    ]
    hs   = []

    def _ehook(li, ei):
        def h(m, inp, out):
            bufs[li][ei] = (out[0] if isinstance(out, tuple) else out
                            ).detach().float().cpu()
        return h

    def _ghook(li):
        def h(m, inp, out):
            router_logits = out[0] if isinstance(out, tuple) else out
            g = router_logits.float().cpu().numpy()
            T = g.shape[0]
            g_sum[li]  += g.sum(0); g_sum2[li] += (g**2).sum(0)
            g_cross[li]+= g.T @ g;  g_n[li]    += T
            w = np.exp(g - g.max(-1, keepdims=True))
            w /= w.sum(-1, keepdims=True)
            ti = np.argsort(w, -1)[:, -top_k:]
            selected_bufs[li] = torch.tensor(ti, dtype=torch.int64)
            tw = np.take_along_axis(w, ti, -1)
            for t in range(T):
                for ei, we in zip(ti[t], tw[t]):
                    freq[li, ei] += 1; act_cnt[li][ei] += 1
                    w_sum[li][ei] += we
                    for ej in ti[t]: coact[li][ei, ej] += 1
            weight_v[li] += float(tw.var())
        return h

    def _mhook(li):
        def h(m, inp, out):
            hid = inp[0].detach().float()
            hidden_bufs[li] = hid.reshape(-1, hid.size(-1))
            ns = hid.reshape(-1, hid.size(-1)).norm(dim=-1)
            h_norm_s[li] += float(ns.sum()); h_norm_n[li] += int(ns.numel())
        return h

    for li, (_, mm) in enumerate(layers):
        hs.append(mm.gate.register_forward_hook(_ghook(li)))
        hs.append(mm.register_forward_hook(_mhook(li)))
        if not fused_experts[li]:
            for ei, exp in enumerate(mm.experts):
                hs.append(exp.register_forward_hook(_ehook(li, ei)))

    for chunk in tqdm(chunks, desc="  Calib forward"):
        for b in bufs: b.clear()
        try:
            model(chunk.to(device))
        except Exception as ex:
            print(f"  ⚠ {ex}"); continue
        for li, (_, mm) in enumerate(layers):
            if not fused_experts[li]:
                continue
            hid = hidden_bufs[li]
            selected = selected_bufs[li]
            if hid is None or selected is None:
                continue
            selected = selected.to(hid.device)
            for expert_id in torch.unique(selected).tolist():
                token_mask = selected.eq(int(expert_id)).any(dim=-1)
                current = hid[token_mask].to(mm.experts.gate_up_proj.dtype)
                gate, up = F.linear(current, mm.experts.gate_up_proj[int(expert_id)]).chunk(2, dim=-1)
                activated = mm.experts.act_fn(gate) * up
                expert_out = F.linear(activated, mm.experts.down_proj[int(expert_id)])
                bufs[li][int(expert_id)] = expert_out.detach().float().cpu()
        for li in range(L):
            for ei, oe in bufs[li].items():
                if oe.dim() != 2 or oe.size(0) == 0: continue
                m = oe.size(0)
                sum_out[li][ei] += oe.sum(0); cnt_out[li][ei] += m
                ns = oe.norm(dim=-1).numpy()
                norm_sum[li][ei] += float(ns.sum()); ce_cnt[li][ei] += m
                for ej, oj in bufs[li].items():
                    if ej == ei or oj.size(0) == 0: continue
                    mm2 = min(m, oj.size(0))
                    d   = (oe[:mm2] - oj[:mm2]).norm(dim=-1).numpy()
                    n   = oe[:mm2].norm(dim=-1).numpy().clip(1e-8)
                    ce_sum[li][ei, ej] += float((d / n).sum())

    for h in hs: h.remove()

    # ── Build tables ──────────────────────────────────────────────────────────
    print("  Building tables ...")
    cos_t = torch.zeros(L, N, N)
    cond  = torch.zeros(L, N, N)
    corr  = torch.zeros(L, N, N)
    coact_t = torch.zeros(L, N, N)
    skip_e  = torch.zeros(L, N)
    sens_arr= np.zeros(L)

    for li in range(L):
        mo = sum_out[li].clone()
        for ei in range(N):
            if cnt_out[li][ei] > 0: mo[ei] /= cnt_out[li][ei]
        unit = mo / mo.norm(dim=-1, keepdim=True).clamp(1e-8)
        cos_t[li] = (unit @ unit.T).clamp(-1, 1)

        ae = act_cnt[li]
        for ei in range(N):
            if ae[ei] > 0:
                coact_t[li, ei, :] = torch.tensor(
                    coact[li][ei] / ae[ei], dtype=torch.float32)
        coact_t[li].fill_diagonal_(1.0)

        n_g = g_n[li]
        if n_g > 1:
            mu = g_sum[li]/n_g; sg = (g_sum2[li]/n_g - mu**2).clip(1e-12)**0.5
            cv = g_cross[li]/n_g - np.outer(mu, mu)
            cr = np.clip(cv / np.outer(sg, sg).clip(1e-12), -1, 1)
            corr[li] = torch.tensor(cr, dtype=torch.float32)
        else:
            corr[li] = torch.eye(N)

        cos_np = cos_t[li].numpy()
        for ei in range(N):
            n = ce_cnt[li][ei]
            cprior = ((cos_np[ei] + 1.0) / 2.0).clip(0, 1)
            if n > 0:
                row = ce_sum[li][ei] / n
                sv  = np.exp(-row)
                unobs = (ce_sum[li][ei] == 0) & (np.arange(N) != ei)
                sv[unobs] = cprior[unobs]
                cond[li, ei, :] = torch.tensor(sv, dtype=torch.float32)
            else:
                cond[li, ei, :] = torch.tensor(cprior, dtype=torch.float32)
        cond[li].fill_diagonal_(1.0)

        mh = float(h_norm_s[li] / max(h_norm_n[li], 1))
        for ei in range(N):
            n = ce_cnt[li][ei]
            if n > 0:
                skip_e[li, ei] = norm_sum[li][ei] / n / max(mh, 1e-8)

        sens_arr[li] = weight_v[li] / max(len(chunks), 1)

    # Combined similarity
    lam_o, lam_c, lam_r = 0.5, 0.3, 0.2
    sim = (lam_o*cos_t + lam_c*coact_t + lam_r*corr).clamp(-1, 1)

    # Normalise sensitivity
    sv_min, sv_max = sens_arr.min(), sens_arr.max()
    sens = torch.tensor(
        1.0 - (sens_arr - sv_min) / (sv_max - sv_min + 1e-8),
        dtype=torch.float32)

    freq_norm = freq / freq.sum(1, keepdims=True).clip(1)

    print(f"  Done. cos mean={cos_t.mean():.3f}  "
          f"coact mean={coact_t.mean():.3f}")
    return Calib(sim=sim, cond_sim=cond, skip_err=skip_e,
                 sens=sens, act_freq=freq_norm, L=L, N=N)


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 – Base wrapper (fixed forward: scatter-add, no double-counting)
# ─────────────────────────────────────────────────────────────────────────────

class BaseWrapper(nn.Module):
    """
    Replaces a single MoE block.
    Key fix vs v1: weights are accumulated into a [T, N] buffer via scatter_add,
    then each expert is called EXACTLY ONCE with its aggregated weight.
    This eliminates double-counting when substitute j* is already in top-k.
    """

    def __init__(self, orig, cache: SimulatedCache,
                 layer_idx: int, calib: Calib):
        super().__init__()
        self.orig    = orig
        self.cache   = cache
        self.l       = layer_idx
        self.calib   = calib
        self.top_k   = getattr(orig, "top_k", 8)
        self.num_exp = getattr(orig, "num_experts", 128)

    def forward(self, hidden_states: torch.Tensor):
        B, T, D = hidden_states.shape
        h     = hidden_states.view(-1, D)
        dev   = h.device

        # Router
        logits = self.orig.gate(h)                          # [T, N]
        probs  = F.softmax(logits, dim=-1)
        rw, ri = probs.topk(self.top_k, dim=-1)            # [T, k]

        # Subclass decision
        cached_mask = self.cache.mask(self.l, dev)
        fw, fi = self._reroute(logits, rw, ri, cached_mask) # [T, k]

        # ── Zero-weight fallback: prevent collapsed hidden states ─────────────
        row_sum = fw.sum(-1)
        if (row_sum == 0).any():
            empty = (row_sum == 0)
            cl = self.cache.cached_list(self.l)
            fb = cl[0] if cl else 0
            fi[empty, 0] = fb; fw[empty, 0] = 1.0

        # Renormalise
        fw = (fw / fw.sum(-1, keepdim=True).clamp(1e-9)).to(h.dtype)

        # ── Fixed forward: accumulate into N-dim buffer, run each expert once ──
        N  = self.num_exp
        wb = torch.zeros(B*T, N, dtype=h.dtype, device=dev)
        for slot in range(fi.size(1)):
            wb.scatter_add_(1, fi[:, slot:slot+1], fw[:, slot:slot+1])

        out = torch.zeros(B*T, D, dtype=h.dtype, device=dev)
        active_experts = wb.any(dim=0).nonzero(as_tuple=False).squeeze(-1)
        for e_t in active_experts:
            ei = int(e_t.item())
            we = wb[:, ei]                              # [BT]
            mask = we > 0
            if not mask.any(): continue
            eo = self.orig.experts[ei](h[mask])
            if isinstance(eo, tuple): eo = eo[0]
            out[mask] += (eo * we[mask].unsqueeze(-1)).to(h.dtype)

        # Shared expert
        if hasattr(self.orig, "shared_expert") and self.orig.shared_expert is not None:
            sh = self.orig.shared_expert(h)
            if hasattr(self.orig, "shared_expert_gate"):
                sh = torch.sigmoid(self.orig.shared_expert_gate(h)) * sh
            out = out + sh.to(out.dtype)

        return out.view(B, T, D), logits

    def _reroute(self, logits, rw, ri, cm) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 – Algorithm implementations
# ─────────────────────────────────────────────────────────────────────────────

# ── Baseline: SkipAll ─────────────────────────────────────────────────────────

class SkipAll(BaseWrapper):
    """Zero weight for missed experts; renormalize over hits."""
    def _reroute(self, logits, rw, ri, cm):
        hit = cm[ri]
        return rw * hit.float(), ri


# ── Alg2_v2: Entropy-Conditioned Pre-Routing Bias with Miss-Rate Gate ─────────

class Alg2_v2(BaseWrapper):
    """
    Improvements over v1:
    ①  Miss-rate gate: gamma_eff = gamma0 * ramp(miss_rate, low, high)
        where ramp = 0 when miss_rate ≤ low, reaches 1 at high.
        Prevents any routing modification when cache is nearly full.
    ②  Residual misses after pre-routing → SkipAll (not substitute).
        Eliminates KV cache contamination from post-routing substitution.
    ③  Weights computed from ORIGINAL unbiased logits (same as v1).
    ④  Top-1 protection (same as v1).
    """
    def __init__(self, orig, cache, layer_idx, calib,
                 gamma0:    float = GAMMA0,
                 miss_low:  float = MISS_GATE,    # gate off below this
                 miss_high: float = 0.50,          # full gamma above this
                 tau_low_q: float = 0.25,           # entropy percentile thresholds
                 tau_high_q:float = 0.75):
        super().__init__(orig, cache, layer_idx, calib)
        self.gamma0     = gamma0
        self.miss_low   = miss_low
        self.miss_high  = miss_high
        self._tau_low   = None
        self._tau_high  = None
        self.tau_low_q  = tau_low_q
        self.tau_high_q = tau_high_q

    def set_entropy_thresholds(self, tau_low: float, tau_high: float):
        self._tau_low  = tau_low
        self._tau_high = tau_high

    def _reroute(self, logits, rw, ri, cm):
        T, k = rw.shape
        N    = self.num_exp
        dev  = logits.device

        # ── Per-token miss rate ───────────────────────────────────────────────
        hit_mask = cm[ri]                           # [T, k] bool
        miss_rate_t = (~hit_mask).float().mean(-1)  # [T] ∈ [0,1]

        # ── Miss-rate gate: ramp gamma between [miss_low, miss_high] ─────────
        lo, hi = self.miss_low, self.miss_high
        t_gate  = ((miss_rate_t - lo) / max(hi - lo, 1e-6)).clamp(0.0, 1.0)  # [T]

        # ── Token routing entropy ─────────────────────────────────────────────
        p       = F.softmax(logits.float(), dim=-1)      # [T, N]
        entropy = -(p * (p + 1e-9).log()).sum(-1)        # [T]
        tau_lo  = self._tau_low  or math.log(N) * self.tau_low_q
        tau_hi  = self._tau_high or math.log(N) * self.tau_high_q
        t_ent   = ((entropy - tau_lo) / max(tau_hi - tau_lo, 1e-6)).clamp(0.0, 1.0)
        gamma_base = self.gamma0 * (0.2 + 0.8 * t_ent)  # [T] entropy-scaled

        # Effective gamma: gated by miss rate
        gamma_eff = gamma_base * t_gate                   # [T] zero when miss_rate ≤ low

        # ── Biased top-k selection ────────────────────────────────────────────
        bias  = gamma_eff.unsqueeze(-1) * cm.float().unsqueeze(0)   # [T, N]
        bgt   = logits.float() + bias                               # [T, N]
        _, ni = bgt.topk(k, dim=-1)                                 # [T, k]

        # Top-1 protection: if original top-1 is NOT cached AND got displaced
        orig_top1 = ri[:, 0]                                        # [T]
        for t_i in range(T):
            ot1 = int(orig_top1[t_i].item())
            if not cm[ot1] and ot1 not in ni[t_i].tolist():
                ni[t_i, -1] = ot1

        # ── Weights from ORIGINAL logits over selected set ────────────────────
        nw = F.softmax(logits.gather(-1, ni).float(), dim=-1)       # [T, k]

        # ── Residual misses → zero weight (SkipAll fallback) ─────────────────
        new_hit = cm[ni]
        nw = nw * new_hit.float()  # zero out residual misses

        return nw.to(rw.dtype), ni


# ── HybridCP_v2: Bounded-Pool Pre-Routing + Miss-Rate Gate + SkipAll Fallback ─

class HybridCP_v2(BaseWrapper):
    """
    Improvements over v1:
    ①  Same miss-rate gate as Alg2_v2: no bias when cache nearly full.
    ②  Candidate pool constraint: bias only applied to cached experts
        within the top-J router candidates (not all cached experts).
        Prevents forcing experts the router strongly disagrees with.
    ③  Deviation guard: if displaced original-top-k weight > tau, skip bias.
    ④  Residual misses after pre-routing → SkipAll (no substitution).
    """
    def __init__(self, orig, cache, layer_idx, calib,
                 gamma0:    float = GAMMA0,
                 j_factor:  int   = J_FACTOR,
                 miss_low:  float = MISS_GATE,
                 miss_high: float = 0.50,
                 tau_dev:   float = 0.20):   # max allowed displacement weight
        super().__init__(orig, cache, layer_idx, calib)
        self.gamma0    = gamma0
        self.j_factor  = j_factor
        self.miss_low  = miss_low
        self.miss_high = miss_high
        self.tau_dev   = tau_dev

    def _reroute(self, logits, rw, ri, cm):
        T, k = rw.shape
        dev  = logits.device

        hit_mask   = cm[ri]
        miss_rate_t = (~hit_mask).float().mean(-1)   # [T]

        lo, hi = self.miss_low, self.miss_high
        t_gate  = ((miss_rate_t - lo) / max(hi - lo, 1e-6)).clamp(0.0, 1.0)  # [T]

        # Per-token entropy scale
        p       = F.softmax(logits.float(), dim=-1)
        entropy = -(p * (p + 1e-9).log()).sum(-1)
        H_max   = math.log(self.num_exp)
        H_norm  = (entropy / H_max).clamp(0, 1)            # [T]
        gamma_eff = self.gamma0 * t_gate * H_norm           # [T]

        # Candidate pool: top J from original router
        J     = min(k * self.j_factor, self.num_exp)
        _, pi = logits.topk(J, dim=-1)                      # [T, J]
        pool  = torch.zeros(T, self.num_exp, dtype=torch.bool, device=dev)
        pool.scatter_(1, pi, True)                           # [T, N] bool

        # Bias = gamma_eff × (cached ∩ pool)
        bias_mask = cm.unsqueeze(0) & pool                  # [T, N] bool
        bgt = logits.float() + gamma_eff.unsqueeze(-1) * bias_mask.float()
        _, ni = bgt.topk(k, dim=-1)                         # [T, k]

        # Deviation guard per token: if too many original experts displaced, use original
        for t_i in range(T):
            if float(gamma_eff[t_i].item()) < 1e-6:
                ni[t_i] = ri[t_i]   # gate was zero anyway
                continue
            orig_set  = set(ri[t_i].tolist())
            draft_set = set(ni[t_i].tolist())
            displaced_w = sum(float(rw[t_i, r].item())
                              for r, e in enumerate(ri[t_i].tolist())
                              if e in orig_set - draft_set)
            if displaced_w > self.tau_dev:
                ni[t_i] = ri[t_i]   # deviation too large → revert to original

        # Weights from original logits
        nw = F.softmax(logits.gather(-1, ni).float(), dim=-1)

        # SkipAll fallback for residual misses
        new_hit = cm[ni]
        nw = nw * new_hit.float()

        return nw.to(rw.dtype), ni


# ── PostSub_v2: Post-Routing Substitution (double-counting aware) ─────────────

class PostSub_v2(BaseWrapper):
    """
    Post-routing substitution using the conditional-error similarity table.

    Improvements over v1:
    ①  Miss-rate gate: skip substitution entirely (SkipAll) when miss_rate ≤ MISS_GATE.
        Prevents KV cache contamination at high cache ratios.
    ②  Sim floor: substitute only when best available sim ≥ SIM_FLOOR.
        Prevents noisy substitution when no good substitute exists (r=0.25).
    ③  Double-counting awareness: when the best substitute j* is already a hit
        in the current token's top-k, do NOT map miss → j* (that would amplify
        j*'s weight).  Instead use SkipAll for that miss slot.
    ④  Contribution threshold: skip miss experts whose expected contribution
        (w_e × output_norm) is below CONTRIB_THRESH × mean_contribution.
    """
    def __init__(self, orig, cache, layer_idx, calib,
                 miss_gate:      float = MISS_GATE,
                 sim_floor:      float = SIM_FLOOR,
                 contrib_thresh: float = 0.10):
        super().__init__(orig, cache, layer_idx, calib)
        self.miss_gate      = miss_gate
        self.sim_floor      = sim_floor
        self.contrib_thresh = contrib_thresh

    def _get_sim(self, dev):
        if self.calib.cond_sim is not None:
            return self.calib.cond_sim[self.l].float().to(dev)
        return torch.eye(self.num_exp, device=dev)

    def _reroute(self, logits, rw, ri, cm):
        T, k = rw.shape
        dev  = logits.device
        N    = self.num_exp
        S    = self._get_sim(dev)                # [N, N]

        hit_mask = cm[ri]                        # [T, k] bool

        # ── Miss-rate gate: below threshold → SkipAll ─────────────────────────
        miss_rate_t = (~hit_mask).float().mean(-1)   # [T]

        # ── Contribution-based skip ───────────────────────────────────────────
        if self.calib.skip_err is not None:
            onorm = self.calib.skip_err[self.l].to(dev)
        else:
            onorm = torch.ones(N, device=dev)
        contrib = rw.float() * onorm[ri]             # [T, k]
        mean_c  = contrib.mean(-1, keepdim=True)
        low_contrib = contrib < self.contrib_thresh * mean_c

        # Build substitution map
        # S_masked[e, j]: sim of e→j, -inf if j not cached or j==e
        S_m = S.masked_fill(~cm.unsqueeze(0), -1e9)
        S_m.fill_diagonal_(-1e9)
        best_sim_all, best_sub_all = S_m.max(-1)    # [N], [N]

        final_ri = ri.clone()
        final_rw = rw.clone()

        for t in range(T):
            mr = float(miss_rate_t[t].item())
            # Build set of experts currently in top-k for this token (for DC check)
            topk_set = set(ri[t].tolist())

            for r in range(k):
                if hit_mask[t, r]:
                    continue    # hit: keep as is

                e    = int(ri[t, r].item())
                w    = float(rw[t, r].item())
                gate = max(0.0, (mr - self.miss_gate) / max(1.0 - self.miss_gate, 1e-6))

                if gate < 1e-6 or low_contrib[t, r]:
                    # Gate closed or negligible contribution → skip
                    final_rw[t, r] = 0.0
                    continue

                j_star  = int(best_sub_all[e].item())
                best_s  = float(best_sim_all[e].item())

                # ③ Double-counting check: if j* already in top-k → skip
                if j_star in topk_set:
                    final_rw[t, r] = 0.0
                    continue

                # Sim floor: if no good substitute → skip
                if best_s < self.sim_floor:
                    final_rw[t, r] = 0.0
                    continue

                # Substitute
                final_ri[t, r] = j_star
                # Weight scaled by similarity × gate ramp
                final_rw[t, r] = rw[t, r] * best_s * gate

        return final_rw, final_ri


# ── Alg2_PostSub: Combined (Alg2_v2 pre-routing  +  PostSub_v2 residuals) ─────

class Alg2_PostSub(nn.Module):
    """
    Two-stage wrapper:
      Stage 1 — Alg2_v2 pre-routing bias (reduces miss count).
      Stage 2 — PostSub_v2 on remaining misses after stage 1.

    Implements a full forward() override because stage 1 changes the routing
    indices that stage 2 receives.
    """
    def __init__(self, orig, cache: SimulatedCache, layer_idx: int, calib: Calib,
                 **kwargs):
        super().__init__()
        self._alg2 = Alg2_v2(orig, cache, layer_idx, calib, **kwargs)
        self._sub  = PostSub_v2(orig, cache, layer_idx, calib)
        self.orig = orig; self.l = layer_idx; self.cache = cache

    def forward(self, hidden_states: torch.Tensor):
        B, T, D = hidden_states.shape
        h   = hidden_states.view(-1, D)
        dev = h.device

        logits = self.orig.gate(h)
        probs  = F.softmax(logits, dim=-1)
        rw, ri = probs.topk(self._alg2.top_k, dim=-1)
        cm     = self.cache.mask(self.l, dev)

        # Stage 1: Alg2_v2 → biased routing, residual misses zeroed
        fw1, fi1 = self._alg2._reroute(logits, rw, ri, cm)

        # Stage 2: PostSub_v2 on stage-1 results
        # Re-compute hit mask from stage-1 indices
        rw2 = fw1; ri2 = fi1
        fw2, fi2 = self._sub._reroute(logits, rw2, ri2, cm)

        # Fallback & renorm via Alg2's base class logic (reuse _alg2.forward flow)
        row_sum = fw2.sum(-1)
        if (row_sum == 0).any():
            empty = (row_sum == 0)
            cl = self.cache.cached_list(self.l)
            fb = cl[0] if cl else 0
            fi2[empty, 0] = fb; fw2[empty, 0] = 1.0
        fw2 = (fw2 / fw2.sum(-1, keepdim=True).clamp(1e-9)).to(h.dtype)

        N  = self._alg2.num_exp
        wb = torch.zeros(B*T, N, dtype=h.dtype, device=dev)
        for slot in range(fi2.size(1)):
            wb.scatter_add_(1, fi2[:, slot:slot+1], fw2[:, slot:slot+1])

        out = torch.zeros(B*T, D, dtype=h.dtype, device=dev)
        for e_t in wb.any(0).nonzero(as_tuple=False).squeeze(-1):
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
# Section 7 – Model patch/restore
# ─────────────────────────────────────────────────────────────────────────────

def patch(model, moe_attr, wrappers, indices):
    for gi, w in zip(indices, wrappers):
        setattr(model.model.layers[gi], moe_attr, w)

def restore(model, moe_attr, originals, indices):
    for gi, o in zip(indices, originals):
        setattr(model.model.layers[gi], moe_attr, o)

def copy_kv(kv):
    if kv is None: return None
    if isinstance(kv, tuple):
        return tuple(tuple(t.clone() for t in lkv) for lkv in kv)
    return copy.deepcopy(kv)

def free_kv(kv):
    if kv is None: return
    del kv


# ─────────────────────────────────────────────────────────────────────────────
# Section 8 – Metrics
# ─────────────────────────────────────────────────────────────────────────────

def alpha_tv(target: torch.Tensor, draft: torch.Tensor) -> Optional[float]:
    """α = Σ_v min(p_t(v), p_d(v)).  Returns None if NaN/Inf."""
    if not (torch.isfinite(target).all() and torch.isfinite(draft).all()):
        return None
    pt = F.softmax(target.float(), dim=-1)
    pd = F.softmax(draft.float(),  dim=-1)
    r  = float(torch.minimum(pt, pd).sum().item())
    return r if math.isfinite(r) else None


# ─────────────────────────────────────────────────────────────────────────────
# Section 9 – Decode-mode evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

ALGORITHM_NAMES = ["SkipAll", "Alg2_v2", "HybridCP_v2", "PostSub_v2", "Alg2_PostSub"]
ALG_COLORS = {
    "SkipAll":       "#888780",
    "Alg2_v2":       "#1D9E75",
    "HybridCP_v2":   "#d4640c",
    "PostSub_v2":    "#3266ad",
    "Alg2_PostSub":  "#9467bd",
}
ALG_MARKERS = {"SkipAll":"x","Alg2_v2":"o","HybridCP_v2":"s",
               "PostSub_v2":"^","Alg2_PostSub":"D"}


def build_wrappers(alg: str, moe_layers, cache, calib,
                   miss_gate: float = MISS_GATE,
                   sim_floor: float = SIM_FLOOR) -> List[nn.Module]:
    ws = []
    for li, (_, moe) in enumerate(moe_layers):
        if   alg == "SkipAll":
            w = SkipAll(moe, cache, li, calib)
        elif alg == "Alg2_v2":
            w = Alg2_v2(moe, cache, li, calib, miss_low=miss_gate)
        elif alg == "HybridCP_v2":
            w = HybridCP_v2(moe, cache, li, calib, miss_low=miss_gate)
        elif alg == "PostSub_v2":
            w = PostSub_v2(moe, cache, li, calib,
                           miss_gate=miss_gate, sim_floor=sim_floor)
        elif alg == "Alg2_PostSub":
            w = Alg2_PostSub(moe, cache, li, calib, miss_low=miss_gate)
            # propagate sim_floor into the inner PostSub
            w._sub.miss_gate = miss_gate
            w._sub.sim_floor = sim_floor
        else:
            raise ValueError(f"Unknown algorithm: {alg}")
        ws.append(w)
    return ws


@torch.no_grad()
def evaluate_one(model, originals, wrappers, moe_idx, moe_attr,
                 prompts, device, prompt_len, draft_len) -> dict:
    restore(model, moe_attr, originals, moe_idx)

    step_bkt = [[] for _ in range(draft_len)]
    ppl_nll  = 0.0; ppl_n = 0; nan_n = 0; n_seq = 0

    for prompt in tqdm(prompts, desc="    eval", leave=False):
        prompt = prompt.to(device)
        T = prompt.size(1)
        if T < prompt_len + draft_len + 1: continue
        n_seq += 1

        # ── target logits (full model, single pass) ───────────────────────────
        restore(model, moe_attr, originals, moe_idx)
        tgt_logits = model(prompt, use_cache=False).logits[0].float()   # [T, V]

        # ── full-model prompt prefill ─────────────────────────────────────────
        prompt_kv = model(prompt[:, :prompt_len], use_cache=True).past_key_values
        draft_kv  = copy_kv(prompt_kv)
        free_kv(prompt_kv); prompt_kv = None

        # ── install rerouting wrappers ────────────────────────────────────────
        patch(model, moe_attr, wrappers, moe_idx)

        for step in range(draft_len):
            pos = prompt_len + step
            if pos + 1 >= T: break

            tok    = prompt[:, pos:pos+1]
            out_d  = model(tok, past_key_values=draft_kv, use_cache=True)
            dlogit = out_d.logits[0, 0, :].float()
            draft_kv = out_d.past_key_values
            del out_d

            a = alpha_tv(tgt_logits[pos], dlogit)
            if a is not None:
                step_bkt[step].append(a)
                ref = int(prompt[0, pos+1].item())
                ppl_nll -= float(F.log_softmax(dlogit, -1)[ref].item())
                ppl_n   += 1
            else:
                nan_n += 1

        free_kv(draft_kv); draft_kv = None
        del tgt_logits

    restore(model, moe_attr, originals, moe_idx)

    means = [float(np.mean(b)) if b else float("nan") for b in step_bkt]
    seq_alpha = float(np.nanmean(means)) if any(math.isfinite(v) for v in means) \
        else float("nan")
    iso_alpha = means[0] if means else float("nan")
    decay     = means[-1] / means[0] if means and math.isfinite(means[-1]) \
        and math.isfinite(means[0]) and means[0] > 0 else float("nan")
    ppl = math.exp(ppl_nll / ppl_n) if ppl_n > 0 else float("nan")

    return {
        "iso_alpha":      iso_alpha,
        "seq_alpha":      seq_alpha,
        "alpha_decay":    decay,
        "ppl_draft":      ppl,
        "alpha_per_step": means,
        "nan_count":      nan_n,
        "n_sequences":    n_seq,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 10 – Main experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def run_experiments(model, moe_cfg, calib, prompts,
                    cache_ratios, device, prompt_len, draft_len,
                    algorithms=None,
                    miss_gate: float = MISS_GATE,
                    sim_floor: float = SIM_FLOOR) -> dict:
    moe_attr  = moe_cfg["moe_attr"]
    moe_layers = get_moe_layers(model, moe_attr)
    moe_idx    = [gi for gi, _ in moe_layers]
    originals  = [getattr(model.model.layers[gi], moe_attr) for gi in moe_idx]
    L, N       = moe_cfg["num_layers"], moe_cfg["num_experts"]
    alg_names  = algorithms or ALGORITHM_NAMES

    results: dict = {a: {} for a in alg_names}

    for ratio in cache_ratios:
        print(f"\n{'='*60}")
        print(f"  cache_ratio={ratio:.3f}  "
              f"({int(N*ratio)}/{N} cached/layer, ~{(1-ratio)*100:.0f}% miss)")
        print(f"{'='*60}")
        cache = SimulatedCache(L, N, ratio, act_freq=calib.act_freq)

        for alg in alg_names:
            print(f"  ▶ {alg:<22}", end="", flush=True)
            t0 = time.time()
            ws  = build_wrappers(alg, moe_layers, cache, calib,
                                 miss_gate=miss_gate, sim_floor=sim_floor)
            m   = evaluate_one(model, originals, ws, moe_idx, moe_attr,
                                prompts, device, prompt_len, draft_len)
            elapsed = time.time() - t0
            a, s, d = m["iso_alpha"], m["seq_alpha"], m["alpha_decay"]
            print(f"  iso_α={a:.4f}  seq_α={s:.4f}  decay={d:.3f}"
                  f"  ({elapsed:.0f}s)")
            results[alg][ratio] = m

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Section 11 – Reporting
# ─────────────────────────────────────────────────────────────────────────────

def plot_alpha_vs_ratio(results, ratios, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, title in zip(
        axes,
        ["iso_alpha", "seq_alpha"],
        ["iso_α (step 1, single-token)",
         "seq_α (8-step average, KV accumulation)"]
    ):
        for alg, data in results.items():
            ys = [data[r][metric] for r in ratios]
            ax.plot(ratios, ys,
                    color=ALG_COLORS.get(alg, "#aaa"),
                    marker=ALG_MARKERS.get(alg, "o"),
                    linewidth=2, markersize=8, label=alg)
        ax.set_xlabel("Cache Ratio", fontsize=11)
        ax.set_ylabel("α", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 1); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
    plt.suptitle("Expert Rerouting v2: Acceptance Rate vs Cache Ratio", fontsize=12)
    plt.tight_layout()
    p = os.path.join(outdir, "alpha_vs_ratio.png")
    plt.savefig(p, dpi=150); plt.close(); print(f"  Saved: {p}")


def plot_step_curves(results, ratios, outdir):
    for ratio in ratios:
        fig, ax = plt.subplots(figsize=(9, 5))
        for alg, data in results.items():
            steps = data[ratio].get("alpha_per_step", [])
            if not steps: continue
            ax.plot(range(1, len(steps)+1), steps,
                    color=ALG_COLORS.get(alg, "#aaa"),
                    marker=ALG_MARKERS.get(alg, "o"),
                    linewidth=2, markersize=7, label=alg)
        ax.set_xlabel("Draft step"); ax.set_ylabel("seq_α")
        ax.set_title(f"Per-step α @ cache_ratio={ratio:.3f}")
        ax.set_ylim(0, 1); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p = os.path.join(outdir, f"alpha_vs_step_cache{ratio:.3f}.png")
        plt.savefig(p, dpi=150); plt.close(); print(f"  Saved: {p}")


def save_csv(results, ratios, outdir):
    path = os.path.join(outdir, "results_v2.csv")
    fields = ["algorithm","cache_ratio","step",
              "iso_alpha","seq_alpha_step","alpha_decay","ppl_draft"]
    rows = []
    for alg, data in results.items():
        for r in ratios:
            m = data[r]
            steps = m.get("alpha_per_step", [])
            for si, sv in enumerate(steps, 1):
                rows.append({
                    "algorithm":      alg,
                    "cache_ratio":    f"{r:.3f}",
                    "step":           str(si),
                    "iso_alpha":      f"{m['iso_alpha']:.4f}",
                    "seq_alpha_step": f"{sv:.4f}",
                    "alpha_decay":    f"{m['alpha_decay']:.3f}",
                    "ppl_draft":      f"{m['ppl_draft']:.3f}" if math.isfinite(m['ppl_draft']) else "nan",
                })
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"  Saved: {path}")

    # Console summary
    print(f"\n  ── Summary (seq_alpha_total) {'─'*30}")
    header = f"  {'Algorithm':<22}" + "".join(f"  r={r:.3f}" for r in ratios)
    print(header); print("  " + "-"*(22 + 9*len(ratios)))
    for alg in results:
        row = f"  {alg:<22}"
        for r in ratios:
            v = results[alg][r]["seq_alpha"]
            row += f"  {v:.4f}" if math.isfinite(v) else "     nan"
        print(row)


def save_json(results, ratios, outdir):
    serial = {}
    for alg, data in results.items():
        serial[alg] = {}
        for r, m in data.items():
            serial[alg][str(r)] = {
                k: (v if not isinstance(v, float) else
                    (v if math.isfinite(v) else None))
                for k, v in m.items()
            }
    with open(os.path.join(outdir, "results_v2.json"), "w") as f:
        json.dump(serial, f, indent=2)
    print(f"  Saved: {os.path.join(outdir, 'results_v2.json')}")


def plot_improvement_bars(results, ratios, outdir):
    """Bar chart: (algo_seq_alpha - SkipAll_seq_alpha) at each ratio."""
    if "SkipAll" not in results: return
    fig, axes = plt.subplots(1, len(ratios), figsize=(5*len(ratios), 5),
                             sharey=False)
    if len(ratios) == 1: axes = [axes]
    for ax, ratio in zip(axes, ratios):
        baseline = results["SkipAll"][ratio]["seq_alpha"]
        algs  = [a for a in results if a != "SkipAll"]
        deltas = [results[a][ratio]["seq_alpha"] - baseline for a in algs]
        colors = [ALG_COLORS.get(a, "#aaa") for a in algs]
        bars = ax.barh(algs, deltas, color=colors, height=0.6)
        ax.axvline(0, color="black", lw=1)
        ax.set_title(f"Δseq_α vs SkipAll  [r={ratio:.3f}]", fontsize=11)
        ax.set_xlabel("Δseq_α"); ax.grid(True, axis="x", alpha=0.3)
        for bar, d in zip(bars, deltas):
            ax.text(d + 0.002, bar.get_y() + bar.get_height()/2,
                    f"{d:+.3f}", va="center", fontsize=9)
    plt.suptitle("Improvement over SkipAll baseline (v2 algorithms)", fontsize=12)
    plt.tight_layout()
    p = os.path.join(outdir, "improvement_bars.png")
    plt.savefig(p, dpi=150); plt.close(); print(f"  Saved: {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 12 – Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Draft-decode expert rerouting v2 (fixed algorithms).")
    p.add_argument("--model",       required=True)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--dtype",       default="float16",
                   choices=["float16","bfloat16","float32"])
    p.add_argument("--data_file",   default=None)
    p.add_argument("--cache_ratios",nargs="+", type=float,
                   default=[0.75, 0.5, 0.25])
    p.add_argument("--prompt_len",  type=int, default=128)
    p.add_argument("--draft_len",   type=int, default=8)
    p.add_argument("--n_prompts",   type=int, default=32)
    p.add_argument("--n_calib",     type=int, default=64)
    p.add_argument("--seq_len",     type=int, default=256)
    p.add_argument("--outdir",      default="./results_v2")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--algorithms",  nargs="+", default=None,
                   help=f"Subset to run. Default: {ALGORITHM_NAMES}")
    p.add_argument("--miss_gate",   type=float, default=MISS_GATE,
                   help="Miss-rate gate threshold (default 0.25).")
    p.add_argument("--sim_floor",   type=float, default=SIM_FLOOR,
                   help="Min substitute similarity floor (default 0.40).")
    p.add_argument("--calibration_artifact", default="",
                   help="Optional .pt path for exporting cond_sim and skip_err for nano-vllm-moe.")
    p.add_argument("--calibration_only", action="store_true",
                   help="Exit after offline calibration and optional artifact export.")
    args = p.parse_args()

    assert args.prompt_len + args.draft_len + 1 <= args.seq_len
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    dtype_map = {"float16": torch.float16,
                 "bfloat16": torch.bfloat16, "float32": torch.float32}
    alg_names  = args.algorithms or ALGORITHM_NAMES
    miss_gate  = args.miss_gate
    sim_floor  = args.sim_floor
    print(f"Algorithms ({len(alg_names)}): {alg_names}")
    print(f"miss_gate={miss_gate}  sim_floor={sim_floor}")

    # ── Load ─────────────────────────────────────────────────────────────────
    model, tok = load_model_and_tokenizer(
        args.model, args.device, dtype_map[args.dtype])
    moe_cfg = detect_moe_config(model)
    print(f"Model config: {moe_cfg}")

    # ── Data ─────────────────────────────────────────────────────────────────
    print(f"\nPreparing {args.n_prompts} prompts (length {args.prompt_len}) ...")
    prompts = prepare_chunks(tok, args.n_prompts, args.prompt_len + args.draft_len + 1,
                             args.data_file)
    print(f"Preparing {args.n_calib} calibration chunks (length {args.seq_len}) ...")
    calib_chunks = prepare_chunks(tok, args.n_calib, args.seq_len,
                                  args.data_file)

    # ── Calibration ───────────────────────────────────────────────────────────
    print(f"\n── Phase 1: Offline Calibration {'─'*30}")
    calib = calibrate(model, moe_cfg["moe_attr"], calib_chunks,
                      args.device, moe_cfg["num_experts"])
    if args.calibration_artifact:
        artifact_path = os.path.abspath(args.calibration_artifact)
        os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
        torch.save(
            {
                "cond_sim": calib.cond_sim.float().cpu().contiguous(),
                "skip_err": calib.skip_err.float().cpu().contiguous(),
            },
            artifact_path,
        )
        print(f"  Saved runtime calibration artifact: {artifact_path}")
    if args.calibration_only:
        print("  Calibration-only run complete.")
        return

    # ── Evaluation ───────────────────────────────────────────────────────────
    print(f"\n── Phase 2: Draft-Decode Simulation {'─'*25}")
    print(f"  prompt_len={args.prompt_len}  draft_len={args.draft_len}"
          f"  n_prompts={args.n_prompts}")
    results = run_experiments(
        model, moe_cfg, calib, prompts,
        args.cache_ratios, args.device,
        args.prompt_len, args.draft_len,
        algorithms=alg_names,
        miss_gate=miss_gate, sim_floor=sim_floor)

    # ── Reports ───────────────────────────────────────────────────────────────
    print(f"\n── Phase 3: Reporting {'─'*38}")
    plot_alpha_vs_ratio(results, args.cache_ratios, args.outdir)
    plot_step_curves(results, args.cache_ratios, args.outdir)
    plot_improvement_bars(results, args.cache_ratios, args.outdir)
    save_csv(results, args.cache_ratios, args.outdir)
    save_json(results, args.cache_ratios, args.outdir)
    print(f"\n✓ Results saved to {args.outdir}/")


if __name__ == "__main__":
    main()
