"""
analyze_m3_step0_breakdown.py
==============================
Break down M3 step-0 perfect fraction by previous cycle's reject position.

Key insight: step 0 of the NEXT draft always follows a verify phase.
But the NUMBER of verified tokens depends on whether the previous draft
was fully accepted (verify loaded K+1 tokens) or rejected early
(verify loaded reject_pos+1 tokens).

We want to see:
- step 0 (next draft, clean KV): perfect fraction by previous reject_pos
- step 1 (next draft, KV polluted by 1 draft step): perfect fraction by previous reject_pos
- ...
- step 7 (next draft, KV polluted by 7 draft steps): perfect fraction by previous reject_pos

Usage (small config for speed):
  CUDA_VISIBLE_DEVICES=2 python analyze_m3_step0_breakdown.py \
      --model /zx_data1/models/Qwen--Qwen3-30B-A3B-Base \
      --data_file ./wikitext2_test.txt \
      --cache_ratio 0.25 --draft_len 8 --prompt_len 128 \
      --n_calib 4 --n_eval 8 --seq_len 384 \
      --outdir ./results_m3_step0_analysis
"""

from __future__ import annotations

import argparse, json, math, os, random, sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
import copy
from copy import deepcopy

# ── Reuse M3 utilities via subprocess-safe copy ────────────────────────────

_SNIPPET = (
    "Wikipedia is a free online encyclopedia. Language models predict the next "
    "token given previous tokens. Mixture-of-Experts activates a subset of params. "
) * 200

MISS_GATE = 0.25
GAMMA0 = 4.0


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


def detect_moe_config(model):
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


class DynamicLFUCache:
    def __init__(self, L, N, S, act_freq=None):
        self.L = L; self.N = N; self.S = S
        self.freq = [np.zeros(N, dtype=np.float64) for _ in range(L)]
        self.cached = [set() for _ in range(L)]
        if act_freq is not None:
            for l in range(L):
                top = np.argsort(act_freq[l])[::-1][:S]
                self.cached[l] = set(top.tolist())
                for e in top:
                    self.freq[l][e] = float(act_freq[l][e])

    def mask(self, layer, device):
        m = torch.zeros(self.N, dtype=torch.bool, device=device)
        if self.cached[layer]:
            m[list(self.cached[layer])] = True
        return m

    def cached_list(self, layer):
        return sorted(self.cached[layer])

    def hit_rate(self, layer, expert_ids):
        expert_ids = set(expert_ids) if not isinstance(expert_ids, set) else expert_ids
        if not expert_ids:
            return 1.0
        return len(expert_ids & self.cached[layer]) / len(expert_ids)

    def hit_rate_all_layers(self, routing):
        rates = [self.hit_rate(li, exps) for li, exps in routing.items()]
        return float(np.mean(rates)) if rates else 1.0

    def all_layers_perfect(self, routing):
        return all(self.hit_rate(li, exps) == 1.0
                   for li, exps in routing.items())

    def ensure_loaded(self, layer, expert_ids):
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

    def ensure_loaded_all_layers(self, routing):
        for li, experts in routing.items():
            self.ensure_loaded(li, experts)


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
            if g is None: continue
            if g.dim() == 3: g = g[0]
            _, topk_i = torch.topk(g, top_k, dim=-1)
            for t in range(g.size(0)):
                for ei in topk_i[t].tolist():
                    act_freq[li, ei] += 1
    for h in hooks: h.remove()
    if total_tokens > 0:
        act_freq /= total_tokens
    return act_freq


class SkipAllWrapper(nn.Module):
    def __init__(self, orig, cache, layer_idx):
        super().__init__()
        self.orig = orig; self.cache = cache; self.l = layer_idx
        self.top_k = getattr(orig, "top_k", 8)
        self.num_exp = getattr(orig, "num_experts", 128)

    def forward(self, hidden_states):
        B, T, D = hidden_states.shape
        h = hidden_states.view(-1, D); dev = h.device
        logits = self.orig.gate(h)
        probs = F.softmax(logits, dim=-1)
        rw, ri = probs.topk(self.top_k, dim=-1)
        cm = self.cache.mask(self.l, dev)
        hit = cm[ri]; fw = rw * hit.float()
        row_sum = fw.sum(-1)
        if (row_sum == 0).any():
            empty = (row_sum == 0)
            cl = self.cache.cached_list(self.l)
            fb = cl[0] if cl else 0
            ri[empty, 0] = fb; fw[empty, 0] = 1.0
        fw = (fw / fw.sum(-1, keepdim=True).clamp(1e-9)).to(h.dtype)
        N = self.num_exp
        wb = torch.zeros(B * T, N, dtype=h.dtype, device=dev)
        for slot in range(ri.size(1)):
            wb.scatter_add_(1, ri[:, slot:slot + 1], fw[:, slot:slot + 1])
        out = torch.zeros(B * T, D, dtype=h.dtype, device=dev)
        for e_t in wb.any(0).nonzero(as_tuple=False).squeeze(-1):
            ei = int(e_t.item()); we = wb[:, ei]; mask = we > 0
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


class Alg2v2Wrapper(nn.Module):
    def __init__(self, orig, cache, layer_idx, gamma0=GAMMA0,
                 miss_low=MISS_GATE, miss_high=0.50):
        super().__init__()
        self.orig = orig; self.cache = cache; self.l = layer_idx
        self.top_k = getattr(orig, "top_k", 8)
        self.num_exp = getattr(orig, "num_experts", 128)
        self.gamma0 = gamma0; self.miss_low = miss_low; self.miss_high = miss_high

    def forward(self, hidden_states):
        B, T, D = hidden_states.shape
        h = hidden_states.view(-1, D); dev = h.device
        N = self.num_exp; k = self.top_k
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
        orig_top1 = ri[:, 0]
        for t_i in range(B * T):
            ot1 = int(orig_top1[t_i].item())
            if not cm[ot1] and ot1 not in ni[t_i].tolist():
                ni[t_i, -1] = ot1
        nw = F.softmax(logits.gather(-1, ni).float(), dim=-1)
        new_hit = cm[ni]; nw = nw * new_hit.float()
        row_sum = nw.sum(-1)
        if (row_sum == 0).any():
            empty = (row_sum == 0)
            cl = self.cache.cached_list(self.l)
            fb = cl[0] if cl else 0
            ni[empty, 0] = fb; nw[empty, 0] = 1.0
        nw = (nw / nw.sum(-1, keepdim=True).clamp(1e-9)).to(h.dtype)
        wb = torch.zeros(B * T, N, dtype=h.dtype, device=dev)
        for slot in range(ni.size(1)):
            wb.scatter_add_(1, ni[:, slot:slot + 1], nw[:, slot:slot + 1])
        out = torch.zeros(B * T, D, dtype=h.dtype, device=dev)
        for e_t in wb.any(0).nonzero(as_tuple=False).squeeze(-1):
            ei = int(e_t.item()); we = wb[:, ei]; mask = we > 0
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


def build_wrappers(alg, moe_layers, cache):
    ws = []
    for li, (_, moe) in enumerate(moe_layers):
        if alg == "SkipAll":
            ws.append(SkipAllWrapper(moe, cache, li))
        elif alg == "Alg2_v2":
            ws.append(Alg2v2Wrapper(moe, cache, li))
        else:
            raise ValueError(f"Unknown algorithm: {alg}")
    return ws


def patch(model, moe_attr, wrappers, indices):
    for gi, w in zip(indices, wrappers):
        setattr(model.model.layers[gi], moe_attr, w)


def restore(model, moe_attr, originals, indices):
    for gi, o in zip(indices, originals):
        setattr(model.model.layers[gi], moe_attr, o)


def truncate_kv(kv, end_pos):
    if kv is None:
        return None
    if isinstance(kv, tuple):
        return tuple((k[:, :, :end_pos, :].clone(), v[:, :, :end_pos, :].clone())
                      for k, v in kv)
    from transformers import DynamicCache
    new_cache = DynamicCache()
    for layer_idx in range(len(kv)):
        k, v = kv[layer_idx]
        new_cache.update(k[:, :, :end_pos, :].clone(),
                         v[:, :, :end_pos, :].clone(), layer_idx)
    return new_cache


def alpha_tv(target, draft):
    if not (torch.isfinite(target).all() and torch.isfinite(draft).all()):
        return None
    pt = F.softmax(target.float(), dim=-1)
    pd = F.softmax(draft.float(), dim=-1)
    r = float(torch.minimum(pt, pd).sum().item())
    return r if math.isfinite(r) else None


@dataclass
class TargetData:
    logits: torch.Tensor
    routing: List[Dict[int, Set[int]]]
    kv_cache: object


@torch.no_grad()
def collect_target_data(model, moe_attr, prompt, device, N, top_k, L):
    layers = get_moe_layers(model, moe_attr)
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
    inp = prompt.to(device)
    out = model(inp, use_cache=True)
    logits = out.logits[0].float().cpu()
    kv = out.past_key_values
    T = inp.size(1)
    routing = []
    for pos in range(T):
        step_routing = {}
        for li in range(L):
            g = gate_bufs[li]
            if g is None: continue
            if g.dim() == 3: g = g[0]
            if pos >= g.size(0): continue
            w = torch.softmax(g[pos], dim=-1)
            _, topk_i = torch.topk(w, top_k)
            step_routing[li] = set(topk_i.tolist())
        routing.append(step_routing)
    for h in hooks: h.remove()
    return TargetData(logits=logits, routing=routing, kv_cache=kv)


@dataclass
class DetailedCycleRecord:
    """Extended record with per-step perfect tracking."""
    cycle: int
    K: int
    reject_pos: int                      # -1 if all accepted
    accepted: int
    alpha_per_step: List[float]
    next_hit_rates: List[float]          # avg hit rate per step of next draft
    next_perfect: List[bool]             # True if hit_rate=1.0 ALL layers


@torch.no_grad()
def run_detailed_cycles(
    model, moe_attr, moe_layers, moe_idx, originals,
    prompt, target_data, cache, alg, draft_len, prompt_len,
    device, N, top_k, L, rng,
) -> List[DetailedCycleRecord]:
    T = prompt.size(1)
    K = draft_len
    records = []
    pos = prompt_len
    cycle_idx = 0

    while pos + K < T:
        wrappers = build_wrappers(alg, moe_layers, cache)
        draft_kv = truncate_kv(target_data.kv_cache, pos)

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

        alphas = []
        for step in range(len(draft_logits)):
            p = pos + step
            a = alpha_tv(target_data.logits[p], draft_logits[step])
            alphas.append(a if a is not None else 0.0)

        reject_pos = -1
        for j in range(len(alphas)):
            if rng.random() > alphas[j]:
                reject_pos = j
                break
        accepted = len(alphas) if reject_pos == -1 else reject_pos

        # Verify: load target routing into cache
        verify_end = min(pos + K + 1, T)
        for p in range(pos, verify_end):
            if p < len(target_data.routing):
                cache.ensure_loaded_all_layers(target_data.routing[p])

        # Next draft start
        next_start = pos + accepted + 1

        # Measure per-step hit rate AND per-step perfection for next K steps
        next_hit_rates = []
        next_perfect = []
        for step in range(K):
            np_ = next_start + step
            if np_ >= len(target_data.routing):
                break
            rt = target_data.routing[np_]
            avg_hr = cache.hit_rate_all_layers(rt)
            all_perfect = cache.all_layers_perfect(rt)
            next_hit_rates.append(avg_hr)
            next_perfect.append(all_perfect)

        records.append(DetailedCycleRecord(
            cycle=cycle_idx, K=K,
            reject_pos=reject_pos, accepted=accepted,
            alpha_per_step=alphas,
            next_hit_rates=next_hit_rates,
            next_perfect=next_perfect,
        ))
        pos = next_start
        cycle_idx += 1

    return records


def breakdown_by_reject(records: List[DetailedCycleRecord], draft_len: int):
    """Group next-draft perfect fraction by previous cycle's reject_pos."""

    # Group records by previous reject_pos
    by_reject: Dict[int, List[DetailedCycleRecord]] = defaultdict(list)
    for r in records:
        by_reject[r.reject_pos].append(r)

    results = {}
    for reject_pos in sorted(by_reject.keys()):
        recs = by_reject[reject_pos]
        n = len(recs)

        # Per-step perfect fraction
        perfect_by_step = defaultdict(int)
        total_by_step = defaultdict(int)
        hit_rates_by_step = defaultdict(list)

        for r in recs:
            for step in range(len(r.next_perfect)):
                if r.next_perfect[step]:
                    perfect_by_step[step] += 1
                total_by_step[step] += 1
                if step < len(r.next_hit_rates):
                    hit_rates_by_step[step].append(r.next_hit_rates[step])

        step_data = {}
        for s in sorted(set(list(perfect_by_step.keys()) + list(hit_rates_by_step.keys()))):
            step_data[s] = {
                "perfect_count": perfect_by_step.get(s, 0),
                "total": total_by_step.get(s, 0),
                "perfect_frac": perfect_by_step.get(s, 0) / max(total_by_step.get(s, 0), 1),
                "mean_hit_rate": float(np.mean(hit_rates_by_step[s])) if s in hit_rates_by_step else None,
            }

        label = "all_accepted" if reject_pos == -1 else f"reject_at_{reject_pos}"
        verified_count = draft_len + 1 if reject_pos == -1 else reject_pos + 1

        results[label] = {
            "reject_pos": reject_pos,
            "verified_tokens": verified_count,
            "n_cycles": n,
            "steps": step_data,
        }

    return results


def print_breakdown(results: dict, draft_len: int):
    print(f"\n{'='*80}")
    print(f"  Step 0 Perfect Fraction Breakdown by Previous Cycle Reject Position")
    print(f"{'='*80}")

    for label, data in sorted(results.items(),
                               key=lambda x: x[1]["reject_pos"]
                               if x[1]["reject_pos"] != -1 else 999):
        rp = data["reject_pos"]
        vt = data["verified_tokens"]
        n = data["n_cycles"]
        print(f"\n--- Previous cycle: {label} "
              f"(verified {vt} tokens, {n} cycles) ---")

        for s in range(draft_len):
            if s not in data["steps"]:
                continue
            sd = data["steps"][s]
            kv_status = "CLEAN (post-verify)" if s == 0 else f"POLLUTED ({s} draft KV steps)"
            print(f"  step {s} [{kv_status}]: "
                  f"perfect={sd['perfect_count']}/{sd['total']} "
                  f"({sd['perfect_frac']:.4f}), "
                  f"hit_rate={sd['mean_hit_rate']:.4f}")

    # Also print combined
    print(f"\n{'='*80}")
    print(f"  Combined (all reject positions) vs Step 0 only")
    print(f"{'='*80}")
    for s in range(draft_len):
        total_perfect = sum(data["steps"][s]["perfect_count"]
                          for data in results.values() if s in data["steps"])
        total_all = sum(data["steps"][s]["total"]
                       for data in results.values() if s in data["steps"])
        all_hit_rates = []
        for data in results.values():
            if s in data["steps"] and data["steps"][s]["mean_hit_rate"] is not None:
                for _ in range(data["steps"][s]["total"]):
                    pass  # can't recover individual values from aggregated
        kv_status = "CLEAN (post-verify)" if s == 0 else f"POLLUTED ({s} draft KV steps)"
        combined_frac = total_perfect / max(total_all, 1)
        print(f"  step {s} [{kv_status}]: "
              f"combined perfect={total_perfect}/{total_all} "
              f"({combined_frac:.4f})")


def plot_breakdown(results: dict, draft_len: int, outdir: str):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Group: full-accept vs any-reject
    full_accept_data = results.get("all_accepted")
    all_reject_data = {}
    for label, data in results.items():
        if data["reject_pos"] != -1:
            for s, sd in data["steps"].items():
                if s not in all_reject_data:
                    all_reject_data[s] = {"perfect": 0, "total": 0, "hits": []}
                all_reject_data[s]["perfect"] += sd["perfect_count"]
                all_reject_data[s]["total"] += sd["total"]

    steps = list(range(draft_len))
    colors = {"CLEAN (post-verify)": "#1D9E75", "POLLUTED": "#d4640c"}

    # Plot 1: Perfect fraction step-by-step, grouped by prev accept/reject
    ax = axes[0]
    for label, color, data_src in [
        ("Prev cycle: full accept", "#1D9E75", full_accept_data),
        ("Prev cycle: rejected", "#d4640c",
         {s: {"perfect_frac": all_reject_data[s]["perfect"] / max(all_reject_data[s]["total"], 1)}
          for s in steps if s in all_reject_data}),
    ]:
        if data_src is None:
            continue
        vals = []
        for s in steps:
            if s in data_src:
                vals.append(data_src[s]["perfect_frac"])
            else:
                vals.append(0)
        ax.plot(steps, vals, "o-", color=color, linewidth=2, markersize=8, label=label)
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.5, label="step 0/1 boundary")
    ax.set_xlabel("Step in next draft", fontsize=11)
    ax.set_ylabel("Perfect fraction (all 48 layers 100% hit)", fontsize=11)
    ax.set_title("M3 Step 0 Perfect Fraction: By Prev Accept/Reject", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)

    # Plot 2: Step 0 only, broken down by verified token count
    ax = axes[1]
    verified_bins = defaultdict(lambda: {"perfect": 0, "total": 0})
    for label, data in results.items():
        vt = data["verified_tokens"]
        if 0 in data["steps"]:
            verified_bins[vt]["perfect"] += data["steps"][0]["perfect_count"]
            verified_bins[vt]["total"] += data["steps"][0]["total"]

    x_labels = []
    x_vals = []
    y_vals = []
    for vt in sorted(verified_bins.keys()):
        d = verified_bins[vt]
        x_labels.append(f"{vt}\n(n={d['total']})")
        x_vals.append(vt)
        y_vals.append(d["perfect"] / max(d["total"], 1))

    bars = ax.bar(range(len(x_vals)), y_vals, color="#1D9E75", alpha=0.8)
    for i, (bar, v) in enumerate(zip(bars, y_vals)):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{v:.3f}', ha='center', fontsize=9)
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Verified tokens in previous cycle", fontsize=11)
    ax.set_ylabel("Step 0 perfect fraction", fontsize=11)
    ax.set_title("Step 0 Perfect Fraction vs Verified Tokens", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(y_vals) * 1.2 if y_vals else 1.0)

    plt.suptitle("M3 Step 0 Perfect Fraction: Detailed Breakdown", fontsize=14)
    plt.tight_layout()
    path = os.path.join(outdir, "m3_step0_breakdown.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def main():
    p = argparse.ArgumentParser(description="M3 Step 0 perfect fraction breakdown")
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--data_file", default=None)
    p.add_argument("--cache_ratio", type=float, default=0.25)
    p.add_argument("--algorithms", nargs="+", default=["SkipAll", "Alg2_v2"])
    p.add_argument("--prompt_len", type=int, default=128)
    p.add_argument("--draft_len", type=int, default=8)
    p.add_argument("--n_calib", type=int, default=4)
    p.add_argument("--n_eval", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=384)
    p.add_argument("--outdir", default="./results_m3_step0_analysis")
    p.add_argument("--seed", type=int, default=42)
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

    moe_attr = moe_cfg["moe_attr"]
    moe_layers = get_moe_layers(model, moe_attr)
    moe_idx = [gi for gi, _ in moe_layers]
    originals = [getattr(model.model.layers[gi], moe_attr) for gi in moe_idx]

    # Prepare data
    print(f"\nPreparing {args.n_calib} calibration chunks ...")
    calib_chunks = prepare_chunks(tok, args.n_calib, args.seq_len, args.data_file)
    print(f"Preparing {args.n_eval} evaluation chunks ...")
    eval_chunks = prepare_chunks(tok, args.n_eval, args.seq_len, args.data_file)

    # Calibrate
    print("\nCalibrating activation frequency ...")
    act_freq = calibrate_freq(model, moe_attr, calib_chunks, args.device, N, top_k)

    S = max(1, int(N * args.cache_ratio))
    print(f"\ncache_ratio={args.cache_ratio:.3f} (S={S}/{N})")

    for alg in args.algorithms:
        print(f"\n{'='*60}")
        print(f"  Algorithm: {alg}")
        print(f"{'='*60}")

        all_records: List[DetailedCycleRecord] = []
        rng = random.Random(args.seed)

        for pi, prompt in enumerate(tqdm(eval_chunks, desc=f"  {alg}")):
            T = prompt.size(1)
            if T < args.prompt_len + args.draft_len + 2:
                continue

            restore(model, moe_attr, originals, moe_idx)
            target_data = collect_target_data(
                model, moe_attr, prompt, args.device, N, top_k, L)

            cache = DynamicLFUCache(L, N, S, act_freq=act_freq)
            for p in range(min(args.prompt_len, len(target_data.routing))):
                cache.ensure_loaded_all_layers(target_data.routing[p])

            recs = run_detailed_cycles(
                model, moe_attr, moe_layers, moe_idx, originals,
                prompt, target_data, cache,
                alg, args.draft_len, args.prompt_len,
                args.device, N, top_k, L, rng)
            all_records.extend(recs)
            del target_data

        restore(model, moe_attr, originals, moe_idx)

        # Breakdown
        print(f"\n  Total cycles: {len(all_records)}")
        results = breakdown_by_reject(all_records, args.draft_len)
        print_breakdown(results, args.draft_len)
        plot_breakdown(results, args.draft_len, args.outdir)

        # Save
        serial = {}
        for label, data in results.items():
            sdata = {
                "reject_pos": data["reject_pos"],
                "verified_tokens": data["verified_tokens"],
                "n_cycles": data["n_cycles"],
                "steps": {str(s): d for s, d in data["steps"].items()},
            }
            serial[label] = sdata
        path = os.path.join(args.outdir, f"m3_step0_breakdown_{alg}_r{args.cache_ratio:.3f}.json")
        with open(path, "w") as f:
            json.dump(serial, f, indent=2, default=str)
        print(f"  Saved: {path}")

    print(f"\nDone. Results in {args.outdir}/")


if __name__ == "__main__":
    main()
