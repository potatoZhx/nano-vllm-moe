"""
motivation_m4_prefetch_signal.py
================================
M4 Experiment: Prefetch Signal Quality — Draft Routing vs INT4 Routing

Compares routing fidelity (vs ground-truth target routing) of:
  1. Our draft routing: FP16 router on approximate hidden states (from rerouting)
  2. INT4 routing: quantized model routing (simulating MoE-SpeQ)

Key insight from Issue 1 analysis: our routing is NOT "precise" — the router
weights are FP16, but the hidden state input is approximate due to upstream
expert substitution. This experiment measures the ACTUAL fidelity of both.

INT4 model loading strategy:
  - Primary: bitsandbytes 4-bit quantization via transformers
  - Fallback: simulated INT4 noise on router logits (calibrated to ~9% mismatch)

Key outputs:
  - Per-layer routing fidelity: |draft_routing ∩ target_routing| / top_k
  - Cumulative fidelity over K draft steps
  - Comparison across cache ratios (our method) vs INT4 (fixed error)

Usage:
  python motivation_m4_prefetch_signal.py \
      --model /path/to/Qwen3-30B-A3B-Base \
      --data_file ../wikitext2_test.txt \
      --cache_ratios 0.25 0.50 0.75 \
      --draft_len 8 --prompt_len 128 \
      --n_calib 8 --n_eval 16 --outdir ./results_m4

  # With actual INT4 model (requires bitsandbytes):
  python motivation_m4_prefetch_signal.py \
      --model /path/to/Qwen3-30B-A3B-Base \
      --int4_mode bnb --outdir ./results_m4

  # Simulated INT4 noise (no extra library needed):
  python motivation_m4_prefetch_signal.py \
      --model /path/to/Qwen3-30B-A3B-Base \
      --int4_mode simulated --outdir ./results_m4
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
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


def load_int4_model(model_path, device):
    """Load INT4 quantized model via bitsandbytes."""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model


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
# Section 2 — Cache & rerouting
# ─────────────────────────────────────────────────────────────────────────────

class SimulatedCache:
    def __init__(self, L, N, ratio, act_freq=None):
        self.N = N
        self.cache_size = max(1, int(N * ratio))
        self.sets: List[set] = []
        for l in range(L):
            if act_freq is not None:
                top = np.argsort(act_freq[l])[::-1][:self.cache_size]
                self.sets.append(set(top.tolist()))
            else:
                self.sets.append(set(random.sample(range(N), self.cache_size)))

    def mask(self, layer, device):
        m = torch.zeros(self.N, dtype=torch.bool, device=device)
        if self.sets[layer]:
            m[list(self.sets[layer])] = True
        return m

    def cached_list(self, layer):
        return sorted(self.sets[layer])


class SkipAllWrapper(nn.Module):
    """SkipAll rerouting: zero missed experts, renormalize hits."""

    def __init__(self, orig, cache, layer_idx):
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


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Calibration
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
# Section 4 — Routing collection: target, draft, INT4
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_routing(
    model, moe_attr: str, prompt: torch.Tensor,
    device: str, N: int, top_k: int, L: int,
    prompt_len: int, draft_len: int,
    use_kv: bool = False,
) -> List[List[Set[int]]]:
    """Collect per-position routing decisions at draft positions.

    Returns: routing[step][layer] = set of expert ids
    """
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
    total_len = prompt_len + draft_len

    if use_kv:
        # Step-by-step decode with KV cache
        routing = []
        # Prefill
        for i in range(L):
            gate_bufs[i] = None
        out = model(inp[:, :prompt_len], use_cache=True)
        kv = out.past_key_values

        for step in range(draft_len):
            pos = prompt_len + step
            if pos >= inp.size(1):
                break
            for i in range(L):
                gate_bufs[i] = None
            tok = inp[:, pos:pos + 1]
            out = model(tok, past_key_values=kv, use_cache=True)
            kv = out.past_key_values

            step_routing = []
            for li in range(L):
                g = gate_bufs[li]
                if g is None:
                    step_routing.append(set())
                    continue
                if g.dim() == 3:
                    g = g[0]
                w = torch.softmax(g[-1], dim=-1)
                _, topk_i = torch.topk(w, top_k)
                step_routing.append(set(topk_i.tolist()))
            routing.append(step_routing)
        del kv
    else:
        # Batch forward (full sequence)
        for i in range(L):
            gate_bufs[i] = None
        model(inp[:, :total_len])

        routing = []
        for step in range(draft_len):
            pos = prompt_len + step
            step_routing = []
            for li in range(L):
                g = gate_bufs[li]
                if g is None:
                    step_routing.append(set())
                    continue
                if g.dim() == 3:
                    g = g[0]
                if pos >= g.size(0):
                    step_routing.append(set())
                    continue
                w = torch.softmax(g[pos], dim=-1)
                _, topk_i = torch.topk(w, top_k)
                step_routing.append(set(topk_i.tolist()))
            routing.append(step_routing)

    for h in hooks:
        h.remove()
    return routing


@torch.no_grad()
def collect_draft_routing(
    model, moe_attr: str, moe_layers, moe_idx, originals,
    prompt: torch.Tensor, cache: SimulatedCache,
    device: str, N: int, top_k: int, L: int,
    prompt_len: int, draft_len: int,
) -> List[List[Set[int]]]:
    """Collect routing from draft model (with rerouting wrappers)."""
    # Build wrappers
    wrappers = []
    for li, (_, moe) in enumerate(moe_layers):
        wrappers.append(SkipAllWrapper(moe, cache, li))

    # Prefill with full model
    restore(model, moe_attr, originals, moe_idx)
    inp = prompt.to(device)
    out = model(inp[:, :prompt_len], use_cache=True)
    kv = out.past_key_values

    # Draft with rerouting
    patch(model, moe_attr, wrappers, moe_idx)

    # Capture gate outputs from wrappers (the wrappers call self.orig.gate)
    gate_bufs: List[Optional[torch.Tensor]] = [None] * L
    hooks = []
    for li, (idx, moe_orig) in enumerate(moe_layers):
        if hasattr(moe_orig, "gate"):
            def _ghook(li_):
                def h(m, inp_h, out_h):
                    r = out_h[0] if isinstance(out_h, tuple) else out_h
                    gate_bufs[li_] = r.detach().float().cpu()
                return h
            hooks.append(moe_orig.gate.register_forward_hook(_ghook(li)))

    routing = []
    for step in range(draft_len):
        pos = prompt_len + step
        if pos >= inp.size(1):
            break
        for i in range(L):
            gate_bufs[i] = None
        tok = inp[:, pos:pos + 1]
        out = model(tok, past_key_values=kv, use_cache=True)
        kv = out.past_key_values

        step_routing = []
        for li in range(L):
            g = gate_bufs[li]
            if g is None:
                step_routing.append(set())
                continue
            if g.dim() == 3:
                g = g[0]
            w = torch.softmax(g[-1], dim=-1)
            _, topk_i = torch.topk(w, top_k)
            step_routing.append(set(topk_i.tolist()))
        routing.append(step_routing)

    for h in hooks:
        h.remove()
    restore(model, moe_attr, originals, moe_idx)
    del kv
    return routing


def simulate_int4_routing(
    target_routing: List[List[Set[int]]],
    N: int, top_k: int, L: int,
    mismatch_rate: float = 0.091,
    rng: random.Random = None,
) -> List[List[Set[int]]]:
    """Simulate INT4 routing noise by randomly flipping experts.

    At each (step, layer), independently replace each expert in the top-k
    with a random different expert, with probability = mismatch_rate.
    This calibrates to MoE-SpeQ's reported ~9.1% expert fidelity loss.
    """
    if rng is None:
        rng = random.Random(42)

    int4_routing = []
    for step_routing in target_routing:
        step_int4 = []
        for layer_experts in step_routing:
            new_experts = set()
            for e in layer_experts:
                if rng.random() < mismatch_rate:
                    # Replace with random expert not in the set
                    replacement = rng.randint(0, N - 1)
                    while replacement in layer_experts:
                        replacement = rng.randint(0, N - 1)
                    new_experts.add(replacement)
                else:
                    new_experts.add(e)
            step_int4.append(new_experts)
        int4_routing.append(step_int4)
    return int4_routing


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Fidelity metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_fidelity(
    routing: List[List[Set[int]]],
    target: List[List[Set[int]]],
    top_k: int,
) -> dict:
    """Compute routing fidelity metrics.

    Returns per-step and per-layer fidelity = |routing ∩ target| / top_k
    """
    K = min(len(routing), len(target))
    L = min(len(routing[0]), len(target[0])) if K > 0 else 0

    step_fidelity = []
    layer_fidelity = defaultdict(list)

    for step in range(K):
        step_scores = []
        for li in range(L):
            r_set = routing[step][li]
            t_set = target[step][li]
            if not t_set:
                continue
            overlap = len(r_set & t_set) / max(len(t_set), 1)
            step_scores.append(overlap)
            layer_fidelity[li].append(overlap)
        step_fidelity.append(float(np.mean(step_scores)) if step_scores else 0.0)

    per_layer_mean = {li: float(np.mean(vs))
                      for li, vs in sorted(layer_fidelity.items())}
    overall = float(np.mean(step_fidelity)) if step_fidelity else 0.0

    return {
        "per_step": step_fidelity,
        "per_layer": per_layer_mean,
        "overall": overall,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    model, moe_cfg: dict, act_freq: np.ndarray,
    prompts: List[torch.Tensor],
    cache_ratios: List[float],
    device: str, prompt_len: int, draft_len: int,
    int4_mode: str, int4_model=None,
    seed: int = 42,
) -> dict:
    moe_attr = moe_cfg["moe_attr"]
    moe_layers = get_moe_layers(model, moe_attr)
    moe_idx = [gi for gi, _ in moe_layers]
    originals = [getattr(model.model.layers[gi], moe_attr) for gi in moe_idx]
    L, N, top_k = moe_cfg["num_layers"], moe_cfg["num_experts"], moe_cfg["top_k"]

    results = {}

    # Collect target routing + INT4 routing (cache-independent)
    print("\n  Collecting target routing ...")
    all_target = []
    all_int4 = []
    rng = random.Random(seed)

    for prompt in tqdm(prompts, desc="    Target"):
        T = prompt.size(1)
        if T < prompt_len + draft_len:
            continue

        # Target: full FP16 model, batch forward
        restore(model, moe_attr, originals, moe_idx)
        target_routing = collect_routing(
            model, moe_attr, prompt, device, N, top_k, L,
            prompt_len, draft_len, use_kv=False)
        all_target.append(target_routing)

        # INT4 routing
        if int4_mode == "bnb" and int4_model is not None:
            int4_routing = collect_routing(
                int4_model, moe_attr, prompt, device, N, top_k, L,
                prompt_len, draft_len, use_kv=False)
        else:
            int4_routing = simulate_int4_routing(
                target_routing, N, top_k, L, mismatch_rate=0.091, rng=rng)
        all_int4.append(int4_routing)

    # INT4 fidelity (fixed, doesn't depend on cache)
    int4_fids = [compute_fidelity(i4, tgt, top_k)
                 for i4, tgt in zip(all_int4, all_target)]
    int4_overall = float(np.mean([f["overall"] for f in int4_fids]))
    int4_per_step = [float(np.mean([f["per_step"][s]
                                     for f in int4_fids if s < len(f["per_step"])]))
                     for s in range(draft_len)]
    print(f"    INT4 fidelity: {int4_overall:.4f} (mode={int4_mode})")

    results["int4"] = {
        "overall": int4_overall,
        "per_step": int4_per_step,
        "mode": int4_mode,
    }

    # Draft routing at each cache ratio
    for ratio in cache_ratios:
        S = max(1, int(N * ratio))
        print(f"\n  cache_ratio={ratio:.3f} (S={S}/{N})")

        cache = SimulatedCache(L, N, ratio, act_freq=act_freq)
        all_draft = []

        for pi, prompt in enumerate(tqdm(prompts, desc=f"    Draft r={ratio:.2f}")):
            T = prompt.size(1)
            if T < prompt_len + draft_len:
                continue

            draft_routing = collect_draft_routing(
                model, moe_attr, moe_layers, moe_idx, originals,
                prompt, cache, device, N, top_k, L,
                prompt_len, draft_len)
            all_draft.append(draft_routing)

        # Compute fidelity
        draft_fids = [compute_fidelity(dr, tgt, top_k)
                      for dr, tgt in zip(all_draft, all_target)]
        draft_overall = float(np.mean([f["overall"] for f in draft_fids]))
        draft_per_step = [float(np.mean([f["per_step"][s]
                                          for f in draft_fids
                                          if s < len(f["per_step"])]))
                          for s in range(draft_len)]
        draft_per_layer = defaultdict(list)
        for f in draft_fids:
            for li, v in f["per_layer"].items():
                draft_per_layer[li].append(v)
        draft_per_layer_mean = {str(li): float(np.mean(vs))
                                for li, vs in sorted(draft_per_layer.items())}

        print(f"    Draft fidelity: {draft_overall:.4f}")
        print(f"    vs INT4:        {int4_overall:.4f}  "
              f"(delta = {draft_overall - int4_overall:+.4f})")

        results[f"draft_r{ratio:.3f}"] = {
            "ratio": ratio,
            "overall": draft_overall,
            "per_step": draft_per_step,
            "per_layer": draft_per_layer_mean,
        }

    restore(model, moe_attr, originals, moe_idx)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — Visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(results: dict, outdir: str, draft_len: int):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Per-step fidelity comparison
    steps = list(range(draft_len))

    # INT4
    int4_data = results.get("int4", {})
    if int4_data:
        ax1.plot(steps, int4_data["per_step"][:draft_len], "s--",
                 color="#d4640c", linewidth=2.5, markersize=7,
                 label=f"INT4 ({int4_data['mode']})")

    # Draft at different ratios
    colors = ["#1D9E75", "#3266ad", "#9467bd", "#8c564b"]
    for ci, (key, data) in enumerate(
        (k, v) for k, v in sorted(results.items()) if k.startswith("draft_")
    ):
        ratio = data["ratio"]
        ax1.plot(steps, data["per_step"][:draft_len], "o-",
                 color=colors[ci % len(colors)], linewidth=2, markersize=6,
                 label=f"Ours (r={ratio:.2f})")

    ax1.set_xlabel("Draft step", fontsize=11)
    ax1.set_ylabel("Routing fidelity", fontsize=11)
    ax1.set_title("Per-step Routing Fidelity vs Target", fontsize=12)
    ax1.set_ylim(0.5, 1.02)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Overall fidelity vs cache ratio
    draft_ratios = []
    draft_fids = []
    for key, data in sorted(results.items()):
        if key.startswith("draft_"):
            draft_ratios.append(data["ratio"])
            draft_fids.append(data["overall"])

    if draft_ratios:
        ax2.plot(draft_ratios, draft_fids, "o-",
                 color="#1D9E75", linewidth=2.5, markersize=8,
                 label="Ours (draft routing)")
    if int4_data:
        ax2.axhline(int4_data["overall"], color="#d4640c", linestyle="--",
                     linewidth=2, label=f"INT4 baseline ({int4_data['overall']:.3f})")
    ax2.set_xlabel("Cache ratio", fontsize=11)
    ax2.set_ylabel("Overall routing fidelity", fontsize=11)
    ax2.set_title("Fidelity vs Cache Ratio", fontsize=12)
    ax2.set_ylim(0.5, 1.02)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("M4: Prefetch Signal Quality (Draft vs INT4 Routing)", fontsize=14)
    plt.tight_layout()
    path = os.path.join(outdir, "m4_prefetch_signal.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def save_results(results: dict, outdir: str):
    path = os.path.join(outdir, "m4_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 8 — Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="M4: Prefetch signal quality experiment")
    p.add_argument("--model",       required=True)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--dtype",       default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--data_file",   default=None)
    p.add_argument("--cache_ratios", nargs="+", type=float,
                   default=[0.25, 0.50, 0.75])
    p.add_argument("--prompt_len",  type=int, default=128)
    p.add_argument("--draft_len",   type=int, default=8)
    p.add_argument("--n_calib",     type=int, default=8)
    p.add_argument("--n_eval",      type=int, default=16)
    p.add_argument("--seq_len",     type=int, default=256)
    p.add_argument("--int4_mode",   default="simulated",
                   choices=["simulated", "bnb"],
                   help="INT4 routing source: 'simulated' (noise model) "
                        "or 'bnb' (bitsandbytes 4-bit)")
    p.add_argument("--mismatch_rate", type=float, default=0.091,
                   help="Expert mismatch rate for simulated INT4 (default: 9.1%%)")
    p.add_argument("--outdir",      default="./results_m4")
    p.add_argument("--seed",        type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}

    # Load FP16 model
    print("Loading FP16 model ...")
    model, tok = load_model_and_tokenizer(
        args.model, args.device, dtype_map[args.dtype])
    moe_cfg = detect_moe_config(model)
    L, N, top_k = moe_cfg["num_layers"], moe_cfg["num_experts"], moe_cfg["top_k"]
    print(f"MoE config: L={L}, N={N}, top_k={top_k}")

    # Load INT4 model (optional)
    int4_model = None
    if args.int4_mode == "bnb":
        print("\nLoading INT4 model (bitsandbytes) ...")
        try:
            int4_model = load_int4_model(args.model, args.device)
            print("  INT4 model loaded.")
        except Exception as e:
            print(f"  Failed to load INT4 model: {e}")
            print("  Falling back to simulated INT4 noise.")
            args.int4_mode = "simulated"

    # Prepare data
    print(f"\nPreparing {args.n_calib} calibration chunks ...")
    calib_chunks = prepare_chunks(tok, args.n_calib, args.seq_len, args.data_file)
    print(f"Preparing {args.n_eval} evaluation chunks ...")
    eval_chunks = prepare_chunks(tok, args.n_eval, args.seq_len, args.data_file)

    # Calibrate
    print("\nCalibrating activation frequency ...")
    act_freq = calibrate_freq(
        model, moe_cfg["moe_attr"], calib_chunks,
        args.device, N, top_k)

    # Run experiment
    print("\nRunning routing fidelity comparison ...")
    results = run_experiment(
        model, moe_cfg, act_freq, eval_chunks,
        args.cache_ratios, args.device,
        args.prompt_len, args.draft_len,
        args.int4_mode, int4_model,
        args.seed)

    # Clean up INT4 model
    if int4_model is not None:
        del int4_model
        torch.cuda.empty_cache()

    # Report
    print(f"\n{'=' * 60}")
    print("  Saving results ...")
    print(f"{'=' * 60}")
    plot_results(results, args.outdir, args.draft_len)
    save_results(results, args.outdir)
    print(f"\nDone. Results in {args.outdir}/")


if __name__ == "__main__":
    main()
