"""
motivation_m3_post_reject_prefix.py
====================================
M3 Experiment: Post-Reject Prefix Hit Rate Analysis

Demonstrates that after speculative decoding rejection, the next draft
cycle's prefix enjoys high cache hit rate — verify loaded experts for
the current topic, which overlap heavily with subsequent positions.

Uses actual model inference for:
  - Draft: rerouting wrappers (SkipAll / Alg2_v2) with KV cache
  - Target: full model forward (single pass, reused across cycles)
  - Accept/reject: speculative sampling via alpha_tv

Key outputs per cycle:
  - K (draft length, fixed)
  - reject_pos (where rejection occurs, or -1 if all accepted)
  - alpha at each draft step
  - hit_rate at each step of the NEXT draft cycle (per layer & averaged)
  - skip_verify_feasibility: fraction of next-draft steps with hit_rate=1.0

Standalone — no dependency on nano-vllm-moe runtime.

Usage:
  python motivation_m3_post_reject_prefix.py \
      --model /path/to/Qwen3-30B-A3B-Base \
      --data_file ../wikitext2_test.txt \
      --cache_ratio 0.25 --draft_len 8 --prompt_len 128 \
      --n_calib 8 --n_eval 16 --outdir ./results_m3

  # Quick test:
  python motivation_m3_post_reject_prefix.py \
      --model /path/to/Qwen3-30B-A3B-Base \
      --data_file ../wikitext2_test.txt \
      --cache_ratio 0.25 --draft_len 8 --prompt_len 128 \
      --n_calib 2 --n_eval 2 --outdir ./results_m3
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
    """LFU cache with per-layer capacity and evict-on-load policy."""

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

    def mask(self, layer: int, device) -> torch.Tensor:
        m = torch.zeros(self.N, dtype=torch.bool, device=device)
        if self.cached[layer]:
            m[list(self.cached[layer])] = True
        return m

    def cached_list(self, layer: int) -> List[int]:
        return sorted(self.cached[layer])

    def hit_rate(self, layer: int, expert_ids) -> float:
        expert_ids = set(expert_ids) if not isinstance(expert_ids, set) else expert_ids
        if not expert_ids:
            return 1.0
        return len(expert_ids & self.cached[layer]) / len(expert_ids)

    def hit_rate_all_layers(self, routing: Dict[int, Set[int]]) -> float:
        rates = [self.hit_rate(li, exps) for li, exps in routing.items()]
        return float(np.mean(rates)) if rates else 1.0

    def per_layer_hit_rates(self, routing: Dict[int, Set[int]]) -> Dict[int, float]:
        return {li: self.hit_rate(li, exps) for li, exps in routing.items()}

    def all_layers_perfect(self, routing: Dict[int, Set[int]]) -> bool:
        """True if hit_rate=1.0 at every layer (skip-verify candidate)."""
        return all(self.hit_rate(li, exps) == 1.0
                   for li, exps in routing.items())

    def ensure_loaded(self, layer: int, expert_ids):
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
# Section 3 — Calibration (activation frequency)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def calibrate_freq(model, moe_attr, chunks, device, N, top_k):
    layers = get_moe_layers(model, moe_attr)
    L = len(layers)
    act_freq = np.zeros((L, N), dtype=np.float64)
    total_tokens = 0

    gate_bufs = [None] * L
    hooks = []
    for li, (idx, moe) in enumerate(layers):
        if hasattr(moe, "gate"):
            def _ghook(li_):
                def h(m, inp, out):
                    r = out[0] if isinstance(out, tuple) else out
                    gate_bufs[li_] = r.detach().float().cpu()
                return h
            hooks.append(moe.gate.register_forward_hook(_ghook(li)))

    for chunk in tqdm(chunks, desc="  Calibrating"):
        inp = chunk.to(device)
        T = inp.size(1)
        total_tokens += T
        for i in range(L):
            gate_bufs[i] = None
        model(inp)
        for li in range(L):
            g = gate_bufs[li]
            if g is None:
                continue
            if g.dim() == 3:
                g = g[0]
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
# Section 4 — Rerouting wrappers (SkipAll + Alg2_v2, minimal)
# ─────────────────────────────────────────────────────────────────────────────

MISS_GATE = 0.25
GAMMA0 = 4.0


class SkipAllWrapper(nn.Module):
    def __init__(self, orig, cache: DynamicLFUCache, layer_idx: int):
        super().__init__()
        self.orig = orig
        self.cache = cache
        self.l = layer_idx
        self.top_k = getattr(orig, "top_k", 8)
        self.num_exp = getattr(orig, "num_experts", 128)

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

        # Zero-weight fallback
        row_sum = fw.sum(-1)
        if (row_sum == 0).any():
            empty = (row_sum == 0)
            cl = self.cache.cached_list(self.l)
            fb = cl[0] if cl else 0
            ri[empty, 0] = fb
            fw[empty, 0] = 1.0

        fw = (fw / fw.sum(-1, keepdim=True).clamp(1e-9)).to(h.dtype)

        N = self.num_exp
        wb = torch.zeros(B * T, N, dtype=h.dtype, device=dev)
        for slot in range(ri.size(1)):
            wb.scatter_add_(1, ri[:, slot:slot + 1], fw[:, slot:slot + 1])

        out = torch.zeros(B * T, D, dtype=h.dtype, device=dev)
        for e_t in wb.any(0).nonzero(as_tuple=False).squeeze(-1):
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


class Alg2v2Wrapper(nn.Module):
    def __init__(self, orig, cache: DynamicLFUCache, layer_idx: int,
                 gamma0=GAMMA0, miss_low=MISS_GATE, miss_high=0.50):
        super().__init__()
        self.orig = orig
        self.cache = cache
        self.l = layer_idx
        self.top_k = getattr(orig, "top_k", 8)
        self.num_exp = getattr(orig, "num_experts", 128)
        self.gamma0 = gamma0
        self.miss_low = miss_low
        self.miss_high = miss_high

    def forward(self, hidden_states):
        B, T, D = hidden_states.shape
        h = hidden_states.view(-1, D)
        dev = h.device
        N = self.num_exp
        k = self.top_k

        logits = self.orig.gate(h)
        probs = F.softmax(logits, dim=-1)
        rw, ri = probs.topk(k, dim=-1)

        cm = self.cache.mask(self.l, dev)
        hit_mask = cm[ri]
        miss_rate_t = (~hit_mask).float().mean(-1)

        lo, hi = self.miss_low, self.miss_high
        t_gate = ((miss_rate_t - lo) / max(hi - lo, 1e-6)).clamp(0.0, 1.0)

        p = F.softmax(logits.float(), dim=-1)
        entropy = -(p * (p + 1e-9).log()).sum(-1)
        H_max = math.log(N)
        t_ent = (entropy / H_max).clamp(0.0, 1.0)
        gamma_eff = self.gamma0 * t_gate * (0.2 + 0.8 * t_ent)

        bias = gamma_eff.unsqueeze(-1) * cm.float().unsqueeze(0)
        bgt = logits.float() + bias
        _, ni = bgt.topk(k, dim=-1)

        # Top-1 protection
        orig_top1 = ri[:, 0]
        for t_i in range(B * T):
            ot1 = int(orig_top1[t_i].item())
            if not cm[ot1] and ot1 not in ni[t_i].tolist():
                ni[t_i, -1] = ot1

        nw = F.softmax(logits.gather(-1, ni).float(), dim=-1)
        new_hit = cm[ni]
        nw = nw * new_hit.float()

        # Zero-weight fallback
        row_sum = nw.sum(-1)
        if (row_sum == 0).any():
            empty = (row_sum == 0)
            cl = self.cache.cached_list(self.l)
            fb = cl[0] if cl else 0
            ni[empty, 0] = fb
            nw[empty, 0] = 1.0

        nw = (nw / nw.sum(-1, keepdim=True).clamp(1e-9)).to(h.dtype)

        wb = torch.zeros(B * T, N, dtype=h.dtype, device=dev)
        for slot in range(ni.size(1)):
            wb.scatter_add_(1, ni[:, slot:slot + 1], nw[:, slot:slot + 1])

        out = torch.zeros(B * T, D, dtype=h.dtype, device=dev)
        for e_t in wb.any(0).nonzero(as_tuple=False).squeeze(-1):
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


def build_wrappers(alg: str, moe_layers, cache: DynamicLFUCache):
    ws = []
    for li, (_, moe) in enumerate(moe_layers):
        if alg == "SkipAll":
            ws.append(SkipAllWrapper(moe, cache, li))
        elif alg == "Alg2_v2":
            ws.append(Alg2v2Wrapper(moe, cache, li))
        else:
            raise ValueError(f"Unknown algorithm: {alg}")
    return ws


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Model patch/restore + KV utilities
# ─────────────────────────────────────────────────────────────────────────────

def patch(model, moe_attr, wrappers, indices):
    for gi, w in zip(indices, wrappers):
        setattr(model.model.layers[gi], moe_attr, w)


def restore(model, moe_attr, originals, indices):
    for gi, o in zip(indices, originals):
        setattr(model.model.layers[gi], moe_attr, o)


def copy_kv(kv):
    if kv is None:
        return None
    if isinstance(kv, tuple):
        return tuple(tuple(t.clone() for t in lkv) for lkv in kv)
    return copy.deepcopy(kv)


def truncate_kv(kv, end_pos):
    """Truncate KV cache to positions [0, end_pos)."""
    if kv is None:
        return None
    if isinstance(kv, tuple):
        return tuple((k[:, :, :end_pos, :].clone(), v[:, :, :end_pos, :].clone())
                      for k, v in kv)
    # DynamicCache
    from transformers import DynamicCache
    new_cache = DynamicCache()
    for layer_idx in range(len(kv)):
        k, v = kv[layer_idx]
        new_cache.update(k[:, :, :end_pos, :].clone(),
                         v[:, :, :end_pos, :].clone(), layer_idx)
    return new_cache


def alpha_tv(target: torch.Tensor, draft: torch.Tensor) -> Optional[float]:
    if not (torch.isfinite(target).all() and torch.isfinite(draft).all()):
        return None
    pt = F.softmax(target.float(), dim=-1)
    pd = F.softmax(draft.float(), dim=-1)
    r = float(torch.minimum(pt, pd).sum().item())
    return r if math.isfinite(r) else None


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Target data collection (single forward pass)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TargetData:
    """Target model outputs for one sequence."""
    logits: torch.Tensor           # [T, V] on CPU
    routing: List[Dict[int, Set[int]]]  # per-position routing
    kv_cache: object               # full KV cache (on device)


@torch.no_grad()
def collect_target_data(
    model, moe_attr: str, prompt: torch.Tensor,
    device: str, N: int, top_k: int, L: int,
) -> TargetData:
    """Full model forward → logits, routing, KV cache."""
    layers = get_moe_layers(model, moe_attr)

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

    inp = prompt.to(device)
    out = model(inp, use_cache=True)

    # Extract logits
    logits = out.logits[0].float().cpu()  # [T, V]

    # Extract KV cache
    kv = out.past_key_values

    # Extract per-position routing
    T = inp.size(1)
    routing = []
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
            step_routing[li] = set(topk_i.tolist())
        routing.append(step_routing)

    for h in hooks:
        h.remove()

    return TargetData(logits=logits, routing=routing, kv_cache=kv)


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — Draft-Verify cycle simulation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CycleRecord:
    cycle: int
    K: int
    reject_pos: int                      # -1 if all accepted
    accepted: int                        # number of accepted tokens
    alpha_per_step: List[float]          # alpha at each draft step
    next_hit_rates: List[float]          # avg hit rate per step of next draft
    next_hit_rates_per_layer: List[Dict[int, float]]  # per-layer hit rates
    next_perfect_steps: int              # steps with hit_rate=1.0 (all layers)


@torch.no_grad()
def run_draft_verify_cycles(
    model, moe_attr: str, moe_layers, moe_idx, originals,
    prompt: torch.Tensor, target_data: TargetData,
    cache: DynamicLFUCache,
    alg: str, draft_len: int, prompt_len: int,
    device: str, N: int, top_k: int, L: int,
    rng: random.Random,
) -> List[CycleRecord]:
    """Run multiple draft-verify cycles on one sequence."""
    T = prompt.size(1)
    K = draft_len
    records = []

    pos = prompt_len
    cycle_idx = 0

    while pos + K < T:
        # Build rerouting wrappers with current cache state
        wrappers = build_wrappers(alg, moe_layers, cache)

        # Get target KV cache truncated to current position
        draft_kv = truncate_kv(target_data.kv_cache, pos)

        # Draft K steps with rerouting
        patch(model, moe_attr, wrappers, moe_idx)

        draft_logits = []
        for step in range(K):
            p = pos + step
            if p + 1 >= T:
                break
            tok = prompt[:, p:p + 1].to(device)
            out_d = model(tok, past_key_values=draft_kv, use_cache=True)
            dlogit = out_d.logits[0, 0, :].float().cpu()
            draft_logits.append(dlogit)
            draft_kv = out_d.past_key_values

        restore(model, moe_attr, originals, moe_idx)
        del draft_kv

        # Compute alpha at each step
        alphas = []
        for step in range(len(draft_logits)):
            p = pos + step
            a = alpha_tv(target_data.logits[p], draft_logits[step])
            alphas.append(a if a is not None else 0.0)

        # Speculative sampling accept/reject
        reject_pos = -1
        for j in range(len(alphas)):
            if rng.random() > alphas[j]:
                reject_pos = j
                break

        if reject_pos == -1:
            accepted = len(alphas)
        else:
            accepted = reject_pos

        # Verify: target model runs on all K+1 positions.
        # This loads experts for all these positions into cache.
        verify_end = min(pos + K + 1, T)
        for p in range(pos, verify_end):
            if p < len(target_data.routing):
                cache.ensure_loaded_all_layers(target_data.routing[p])

        # Next draft starts after accepted tokens + 1 (correction/bonus)
        if reject_pos == -1:
            next_start = pos + accepted + 1  # bonus token
        else:
            next_start = pos + accepted + 1  # correction token

        # Measure hit rate for next K steps
        next_hit_rates = []
        next_hit_per_layer = []
        next_perfect = 0
        for step in range(K):
            np_ = next_start + step
            if np_ >= len(target_data.routing):
                break
            rt = target_data.routing[np_]
            avg_hr = cache.hit_rate_all_layers(rt)
            per_layer = cache.per_layer_hit_rates(rt)
            perfect = cache.all_layers_perfect(rt)

            next_hit_rates.append(avg_hr)
            next_hit_per_layer.append(per_layer)
            if perfect:
                next_perfect += 1

        records.append(CycleRecord(
            cycle=cycle_idx,
            K=K,
            reject_pos=reject_pos,
            accepted=accepted,
            alpha_per_step=alphas,
            next_hit_rates=next_hit_rates,
            next_hit_rates_per_layer=next_hit_per_layer,
            next_perfect_steps=next_perfect,
        ))

        pos = next_start
        cycle_idx += 1

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Section 8 — Main experiment loop
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    model, moe_cfg: dict, act_freq: np.ndarray,
    prompts: List[torch.Tensor],
    cache_ratios: List[float], algorithms: List[str],
    draft_len: int, prompt_len: int,
    device: str, seed: int,
) -> dict:
    moe_attr = moe_cfg["moe_attr"]
    moe_layers = get_moe_layers(model, moe_attr)
    moe_idx = [gi for gi, _ in moe_layers]
    originals = [getattr(model.model.layers[gi], moe_attr) for gi in moe_idx]
    L, N, top_k = moe_cfg["num_layers"], moe_cfg["num_experts"], moe_cfg["top_k"]

    results = {}

    for ratio in cache_ratios:
        S = max(1, int(N * ratio))
        print(f"\n{'=' * 60}")
        print(f"  cache_ratio={ratio:.3f} (S={S}/{N})")
        print(f"{'=' * 60}")

        for alg in algorithms:
            print(f"\n  Algorithm: {alg}")
            all_records: List[CycleRecord] = []
            rng = random.Random(seed)

            for pi, prompt in enumerate(tqdm(prompts, desc=f"    {alg}")):
                T = prompt.size(1)
                if T < prompt_len + draft_len + 2:
                    continue

                # Collect target data (full model forward)
                restore(model, moe_attr, originals, moe_idx)
                target_data = collect_target_data(
                    model, moe_attr, prompt, device, N, top_k, L)

                # Initialize dynamic cache
                cache = DynamicLFUCache(L, N, S, act_freq=act_freq)

                # Warm up cache with prefill routing
                for p in range(min(prompt_len, len(target_data.routing))):
                    cache.ensure_loaded_all_layers(target_data.routing[p])

                # Run draft-verify cycles
                recs = run_draft_verify_cycles(
                    model, moe_attr, moe_layers, moe_idx, originals,
                    prompt, target_data, cache,
                    alg, draft_len, prompt_len,
                    device, N, top_k, L, rng)
                all_records.extend(recs)

                # Free target KV cache
                del target_data

            restore(model, moe_attr, originals, moe_idx)

            # Aggregate results
            results[(ratio, alg)] = aggregate_records(all_records, draft_len)

    return results


def aggregate_records(records: List[CycleRecord], K: int) -> dict:
    """Aggregate cycle records into summary statistics."""
    if not records:
        return {"n_cycles": 0}

    reject_positions = [r.reject_pos for r in records]
    accepted_counts = [r.accepted for r in records]
    all_alphas = [r.alpha_per_step for r in records]

    # Per-step hit rate for next draft (indexed by step position)
    hit_by_step = defaultdict(list)
    perfect_by_step = defaultdict(int)
    total_by_step = defaultdict(int)

    for r in records:
        for step, hr in enumerate(r.next_hit_rates):
            hit_by_step[step].append(hr)
            total_by_step[step] += 1
        # Track perfect steps
        for step in range(len(r.next_hit_rates)):
            if step < r.next_perfect_steps:
                # Count consecutive perfect from start
                pass
        # More precise: check each step individually
        for step in range(len(r.next_hit_rates)):
            if r.next_hit_rates[step] == 1.0:
                perfect_by_step[step] += 1

    # Reject position distribution
    reject_dist = defaultdict(int)
    for rp in reject_positions:
        reject_dist[rp] += 1

    # Mean alpha per step
    alpha_by_step = defaultdict(list)
    for alphas in all_alphas:
        for step, a in enumerate(alphas):
            alpha_by_step[step].append(a)

    n = len(records)
    summary = {
        "n_cycles": n,
        "mean_accepted": float(np.mean(accepted_counts)),
        "median_accepted": float(np.median(accepted_counts)),
        "reject_distribution": {str(k): v for k, v in sorted(reject_dist.items())},
        "mean_alpha_per_step": {
            str(s): float(np.mean(vs)) for s, vs in sorted(alpha_by_step.items())},
        "mean_next_hit_rate_per_step": {
            str(s): float(np.mean(vs)) for s, vs in sorted(hit_by_step.items())},
        "perfect_fraction_per_step": {
            str(s): perfect_by_step[s] / max(total_by_step[s], 1)
            for s in sorted(total_by_step.keys())},
        "overall_perfect_fraction": (
            sum(perfect_by_step.values()) / max(sum(total_by_step.values()), 1)),
    }

    # Console output
    print(f"\n    Cycles: {n}, Mean accepted: {summary['mean_accepted']:.2f}")
    print(f"    Reject distribution: {dict(sorted(reject_dist.items()))}")
    print(f"    Next-draft hit rate by step:")
    for s in sorted(hit_by_step.keys()):
        hr = np.mean(hit_by_step[s])
        pf = perfect_by_step[s] / max(total_by_step[s], 1)
        print(f"      step {s}: hit_rate={hr:.4f}  "
              f"perfect_frac={pf:.4f} ({perfect_by_step[s]}/{total_by_step[s]})")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Section 9 — Visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(results: dict, outdir: str, draft_len: int):
    # Group by cache ratio
    ratios = sorted(set(r for r, a in results.keys()))
    algs = sorted(set(a for r, a in results.keys()))
    colors = {"SkipAll": "#888780", "Alg2_v2": "#1D9E75"}
    markers = {"SkipAll": "x", "Alg2_v2": "o"}

    for ratio in ratios:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: Next-draft hit rate by step
        ax = axes[0]
        for alg in algs:
            key = (ratio, alg)
            if key not in results or results[key]["n_cycles"] == 0:
                continue
            data = results[key]["mean_next_hit_rate_per_step"]
            steps = sorted(int(s) for s in data.keys())
            vals = [data[str(s)] for s in steps]
            ax.plot(steps, vals,
                    color=colors.get(alg, "#aaa"),
                    marker=markers.get(alg, "o"),
                    linewidth=2, markersize=7, label=alg)
        ax.set_xlabel("Step in next draft", fontsize=11)
        ax.set_ylabel("Average hit rate", fontsize=11)
        ax.set_title(f"Post-reject prefix hit rate (r={ratio:.3f})", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Plot 2: Perfect fraction by step
        ax = axes[1]
        for alg in algs:
            key = (ratio, alg)
            if key not in results or results[key]["n_cycles"] == 0:
                continue
            data = results[key]["perfect_fraction_per_step"]
            steps = sorted(int(s) for s in data.keys())
            vals = [data[str(s)] for s in steps]
            ax.bar([s + 0.2 * (algs.index(alg) - len(algs)/2) for s in steps],
                   vals, width=0.35,
                   color=colors.get(alg, "#aaa"), alpha=0.8, label=alg)
        ax.set_xlabel("Step in next draft", fontsize=11)
        ax.set_ylabel("Fraction with hit_rate=1.0", fontsize=11)
        ax.set_title(f"Skip-verify feasibility (r={ratio:.3f})", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

        # Plot 3: Reject position distribution
        ax = axes[2]
        for alg in algs:
            key = (ratio, alg)
            if key not in results or results[key]["n_cycles"] == 0:
                continue
            data = results[key]["reject_distribution"]
            positions = sorted(int(s) for s in data.keys())
            total = sum(data.values())
            fracs = [data[str(p)] / total for p in positions]
            offset = 0.2 * (algs.index(alg) - len(algs) / 2)
            ax.bar([p + offset for p in positions], fracs, width=0.35,
                   color=colors.get(alg, "#aaa"), alpha=0.8, label=alg)
        ax.set_xlabel("Reject position (-1=all accepted)", fontsize=11)
        ax.set_ylabel("Fraction of cycles", fontsize=11)
        ax.set_title(f"Reject position distribution (r={ratio:.3f})", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

        plt.suptitle("M3: Post-Reject Prefix Hit Rate Analysis", fontsize=14)
        plt.tight_layout()
        path = os.path.join(outdir, f"m3_prefix_hitrate_r{ratio:.3f}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")


def save_results(results: dict, outdir: str):
    serial = {}
    for (ratio, alg), summary in results.items():
        key = f"r{ratio:.3f}_{alg}"
        serial[key] = summary
    path = os.path.join(outdir, "m3_results.json")
    with open(path, "w") as f:
        json.dump(serial, f, indent=2, default=str)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 10 — Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="M3: Post-reject prefix hit rate experiment")
    p.add_argument("--model",        required=True)
    p.add_argument("--device",       default="cuda")
    p.add_argument("--dtype",        default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--data_file",    default=None)
    p.add_argument("--cache_ratios", nargs="+", type=float,
                   default=[0.25])
    p.add_argument("--algorithms",   nargs="+", default=["SkipAll", "Alg2_v2"],
                   choices=["SkipAll", "Alg2_v2"])
    p.add_argument("--prompt_len",   type=int, default=128)
    p.add_argument("--draft_len",    type=int, default=8)
    p.add_argument("--n_calib",      type=int, default=8)
    p.add_argument("--n_eval",       type=int, default=16)
    p.add_argument("--seq_len",      type=int, default=512,
                   help="Sequence length (longer = more cycles)")
    p.add_argument("--outdir",       default="./results_m3")
    p.add_argument("--seed",         type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}

    print("Loading model ...")
    model, tok = load_model_and_tokenizer(
        args.model, args.device, dtype_map[args.dtype])
    moe_cfg = detect_moe_config(model)
    L, N, top_k = moe_cfg["num_layers"], moe_cfg["num_experts"], moe_cfg["top_k"]
    print(f"MoE config: L={L}, N={N}, top_k={top_k}")

    # Prepare data
    print(f"\nPreparing {args.n_calib} calibration chunks ...")
    calib_chunks = prepare_chunks(tok, args.n_calib, args.seq_len, args.data_file)
    print(f"Preparing {args.n_eval} evaluation chunks ...")
    eval_chunks = prepare_chunks(tok, args.n_eval, args.seq_len, args.data_file)

    # Calibrate activation frequency
    print("\nCalibrating activation frequency ...")
    act_freq = calibrate_freq(
        model, moe_cfg["moe_attr"], calib_chunks,
        args.device, N, top_k)

    # Run experiment
    print("\nRunning draft-verify cycles ...")
    results = run_experiment(
        model, moe_cfg, act_freq, eval_chunks,
        args.cache_ratios, args.algorithms,
        args.draft_len, args.prompt_len,
        args.device, args.seed)

    # Report
    print(f"\n{'=' * 60}")
    print("  Saving results ...")
    print(f"{'=' * 60}")
    plot_results(results, args.outdir, args.draft_len)
    save_results(results, args.outdir)
    print(f"\nDone. Results in {args.outdir}/")


if __name__ == "__main__":
    main()
