"""
alpha_prediction_eval.py  (Experiment E5)
==========================================
Validate the Level-2 analytical model α̂(k) = α₀·exp(-λ·E(k)) for
predicting per-step acceptance rate, and determine whether an MLP
upgrade is warranted.

Core questions:
  Q1. How well does the analytical model fit actual per-step α?
  Q2. Which additional features (CriticalMissRate, cache ratio,
      prefetch coverage) explain residual variance?
  Q3. Does a simple MLP significantly outperform the analytical model?

Method:
  1. Decode-mode simulation: collect (E(k), actual_α(k)) data pairs
     where E(k) = cumulative rerouting error proxy
  2. Fit α₀, λ to analytical model; compute RMSE
  3. Analyze residual correlation with auxiliary features
  4. Fit a small MLP; compare RMSE and R² improvement
  5. If MLP R² improvement > 0.1, recommend upgrade

Standalone — no dependency on nano-vllm-moe runtime.

Usage:
  python alpha_prediction_eval.py \
      --model /path/to/Qwen3-30B-A3B-Base \
      --data_file ../wikitext2_test.txt \
      --cache_ratios 0.75 0.50 0.25 \
      --k_max 12 --prompt_len 128 \
      --n_calib 8 --n_eval 16 --outdir ./results_e5
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
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
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
# Section 2 — SimulatedCache & calibration
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
        m[list(self.sets[layer])] = True
        return m

    def is_cached(self, layer, expert_id):
        return expert_id in self.sets[layer]


@torch.no_grad()
def calibrate_freq(model, moe_attr, chunks, device, N, top_k):
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
            if g.dim() == 3: g = g.squeeze(0)
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
# Section 3 — α computation & KV utilities
# ─────────────────────────────────────────────────────────────────────────────

def alpha_tv(target, draft):
    if not (torch.isfinite(target).all() and torch.isfinite(draft).all()):
        return None
    pt = F.softmax(target.float(), dim=-1)
    pd = F.softmax(draft.float(), dim=-1)
    r = float(torch.minimum(pt, pd).sum().item())
    return r if math.isfinite(r) else None


def copy_kv(kv):
    if kv is None: return None
    return tuple(tuple(t.clone() for t in layer) for layer in kv)


def free_kv(kv):
    if kv is None: return
    for layer in kv:
        for t in layer:
            del t


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — SkipAll wrapper
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
            ri_c = ri.clone(); ri_c[empty, 0] = fb; fw[empty, 0] = 1.0
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
# Section 5 — Per-step data collection
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StepSample:
    """One data point: (step k, features, actual α)."""
    step: int
    alpha: float
    miss_rate: float          # overall miss rate at this step
    critical_miss_rate: float # top-1/2 miss fraction
    miss_weight_sum: float    # sum of routing weights of missed experts
    cumulative_error: float   # E(k) = Σ_{i=1}^{k} miss_rate_i (cumulative)
    cache_ratio: float        # static: cache_size / N


def patch_model(model, moe_attr, wrappers, moe_indices):
    for li, (idx, _) in enumerate(moe_indices):
        setattr(model.model.layers[idx], moe_attr, wrappers[li])


def restore_model(model, moe_attr, originals, moe_indices):
    for li, (idx, _) in enumerate(moe_indices):
        setattr(model.model.layers[idx], moe_attr, originals[li])


@torch.no_grad()
def collect_step_data(
    model, moe_attr, prompts, device,
    cache, N, top_k, L,
    prompt_len, k_max, cache_ratio,
) -> List[StepSample]:
    """Run decode-mode simulation and collect per-step feature + α pairs."""
    layers = get_moe_layers(model, moe_attr)
    moe_indices = layers
    originals = [moe for _, moe in layers]

    wrappers = [SkipAllWrapper(moe, cache, li, N, top_k)
                for li, (_, moe) in enumerate(layers)]

    total_len = prompt_len + k_max + 1
    samples = []

    gate_bufs = [None] * L

    for prompt in tqdm(prompts, desc="Collecting step data", leave=False):
        prompt = prompt.to(device)
        T = prompt.size(1)
        if T < total_len:
            continue

        inp = prompt[:, :total_len]

        # Target logits
        restore_model(model, moe_attr, originals, moe_indices)
        tgt_logits = model(inp, use_cache=False).logits[0].float()

        # Prefill
        restore_model(model, moe_attr, originals, moe_indices)
        prefill_out = model(prompt[:, :prompt_len], use_cache=True)
        draft_kv = copy_kv(prefill_out.past_key_values)

        # Install wrappers + gate hooks
        patch_model(model, moe_attr, wrappers, moe_indices)
        hooks = []
        for li, (idx, moe_orig) in enumerate(layers):
            if hasattr(moe_orig, "gate"):
                def _ghook(li_):
                    def h(m, inp_, out_):
                        gate_bufs[li_] = out_.detach().float().cpu()
                    return h
                hooks.append(moe_orig.gate.register_forward_hook(_ghook(li)))

        cumulative_error = 0.0
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

            alpha = alpha_tv(tgt_logits[pos], dlogit)
            if alpha is None:
                continue

            # Compute features from gate outputs
            n_miss = 0
            n_total = 0
            top1_miss = 0
            top2_miss = 0
            n_layers_ok = 0
            miss_weight = 0.0

            for li in range(L):
                if gate_bufs[li] is None:
                    continue
                g = gate_bufs[li]
                if g.dim() == 3: g = g[0]
                if g.size(0) == 0: continue
                g_tok = g[-1]
                w = torch.softmax(g_tok, dim=-1)
                topk_w, topk_i = torch.topk(w, top_k)
                topk_list = topk_i.tolist()
                topk_w_list = topk_w.tolist()

                n_layers_ok += 1
                for rank, (eid, ew) in enumerate(zip(topk_list, topk_w_list)):
                    n_total += 1
                    if not cache.is_cached(li, eid):
                        n_miss += 1
                        miss_weight += ew
                        if rank == 0:
                            top1_miss += 1
                        elif rank == 1:
                            top2_miss += 1

            if n_total == 0 or n_layers_ok == 0:
                continue

            miss_rate = n_miss / n_total
            critical_miss_rate = (top1_miss + top2_miss) / (2 * n_layers_ok)
            cumulative_error += miss_rate

            samples.append(StepSample(
                step=step,
                alpha=alpha,
                miss_rate=miss_rate,
                critical_miss_rate=critical_miss_rate,
                miss_weight_sum=miss_weight / n_layers_ok,
                cumulative_error=cumulative_error,
                cache_ratio=cache_ratio,
            ))

        for h in hooks:
            h.remove()
        free_kv(draft_kv)

    restore_model(model, moe_attr, originals, moe_indices)
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Analytical model fitting
# ─────────────────────────────────────────────────────────────────────────────

def fit_analytical_model(samples: List[StepSample]) -> dict:
    """Fit α̂(k) = α₀ · exp(-λ · E(k)) where E(k) is cumulative error.

    Returns fitted parameters and error metrics.
    """
    E = np.array([s.cumulative_error for s in samples])
    alpha = np.array([s.alpha for s in samples])

    # Filter out bad values
    valid = np.isfinite(E) & np.isfinite(alpha) & (alpha > 0)
    E = E[valid]
    alpha = alpha[valid]

    if len(E) < 5:
        return {"error": "insufficient data", "n_samples": len(E)}

    def exp_decay(x, a0, lam):
        return a0 * np.exp(-lam * x)

    try:
        popt, pcov = curve_fit(exp_decay, E, alpha,
                               p0=[alpha.max(), 0.1], maxfev=10000,
                               bounds=([0, 0], [2, 50]))
        alpha_hat = exp_decay(E, *popt)
        residuals = alpha - alpha_hat
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((alpha - np.mean(alpha)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return {
            "alpha0": float(popt[0]),
            "lambda": float(popt[1]),
            "rmse": rmse,
            "r_squared": float(r_squared),
            "n_samples": int(len(E)),
            "residuals": residuals.tolist(),
            "E_values": E.tolist(),
            "alpha_actual": alpha.tolist(),
            "alpha_predicted": alpha_hat.tolist(),
        }
    except Exception as e:
        return {"error": str(e), "n_samples": int(len(E))}


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — Residual analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_residuals(samples: List[StepSample],
                      analytical_result: dict) -> dict:
    """Correlate residuals with auxiliary features."""
    if "residuals" not in analytical_result:
        return {"error": "no residuals available"}

    residuals = np.array(analytical_result["residuals"])
    n = len(residuals)

    # Extract feature arrays (matching the valid samples)
    E_all = np.array([s.cumulative_error for s in samples])
    alpha_all = np.array([s.alpha for s in samples])
    valid = np.isfinite(E_all) & np.isfinite(alpha_all) & (alpha_all > 0)

    valid_samples = [s for s, v in zip(samples, valid) if v]
    if len(valid_samples) != n:
        # Fallback: trim to match
        valid_samples = valid_samples[:n]
        residuals = residuals[:len(valid_samples)]

    features = {
        "miss_rate": np.array([s.miss_rate for s in valid_samples[:len(residuals)]]),
        "critical_miss_rate": np.array([s.critical_miss_rate for s in valid_samples[:len(residuals)]]),
        "miss_weight_sum": np.array([s.miss_weight_sum for s in valid_samples[:len(residuals)]]),
        "step": np.array([s.step for s in valid_samples[:len(residuals)]], dtype=np.float64),
        "cache_ratio": np.array([s.cache_ratio for s in valid_samples[:len(residuals)]]),
    }

    correlations = {}
    for fname, fvals in features.items():
        if len(fvals) != len(residuals):
            continue
        try:
            rho, p_val = stats.spearmanr(residuals, fvals)
            # Linear R² of residual ~ feature
            if np.std(fvals) > 1e-10:
                slope, intercept = np.polyfit(fvals, residuals, 1)
                predicted = slope * fvals + intercept
                ss_res = np.sum((residuals - predicted) ** 2)
                ss_tot = np.sum((residuals - np.mean(residuals)) ** 2)
                partial_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            else:
                partial_r2 = 0
            correlations[fname] = {
                "spearman_rho": float(rho),
                "p_value": float(p_val),
                "partial_r2": float(partial_r2),
            }
        except Exception as e:
            correlations[fname] = {"error": str(e)}

    return correlations


# ─────────────────────────────────────────────────────────────────────────────
# Section 8 — Simple MLP baseline
# ─────────────────────────────────────────────────────────────────────────────

def fit_mlp_model(samples: List[StepSample], n_hidden: int = 16,
                  n_epochs: int = 200, lr: float = 0.01) -> dict:
    """Fit a simple MLP: α̂ = MLP(E(k), miss_rate, crit_miss, miss_weight, step, ratio).

    Uses numpy/scipy for simplicity (no torch training loop needed).
    """
    # Build feature matrix
    E_all = np.array([s.cumulative_error for s in samples])
    alpha_all = np.array([s.alpha for s in samples])
    valid = np.isfinite(E_all) & np.isfinite(alpha_all) & (alpha_all > 0)
    valid_samples = [s for s, v in zip(samples, valid) if v]

    if len(valid_samples) < 20:
        return {"error": "insufficient data", "n_samples": len(valid_samples)}

    X = np.array([
        [s.cumulative_error, s.miss_rate, s.critical_miss_rate,
         s.miss_weight_sum, s.step / 12.0, s.cache_ratio]
        for s in valid_samples
    ])
    y = np.array([s.alpha for s in valid_samples])

    # Normalize features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    # Train/test split (80/20)
    n = len(X_norm)
    indices = np.random.permutation(n)
    n_train = int(0.8 * n)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train, y_train = X_norm[train_idx], y[train_idx]
    X_test, y_test = X_norm[test_idx], y[test_idx]

    # Simple 2-layer MLP using numpy (gradient descent)
    d_in = X.shape[1]
    d_h = n_hidden

    np.random.seed(42)
    W1 = np.random.randn(d_in, d_h) * 0.1
    b1 = np.zeros(d_h)
    W2 = np.random.randn(d_h, 1) * 0.1
    b2 = np.zeros(1)

    def relu(x):
        return np.maximum(0, x)

    def relu_grad(x):
        return (x > 0).astype(np.float64)

    def forward(X):
        h = relu(X @ W1 + b1)
        return (h @ W2 + b2).flatten(), h

    best_test_rmse = float("inf")
    for epoch in range(n_epochs):
        # Forward
        y_pred, h = forward(X_train)

        # Loss
        loss = np.mean((y_pred - y_train) ** 2)

        # Backward
        d_loss = 2 * (y_pred - y_train) / len(y_train)  # [n_train]
        dW2 = h.T @ d_loss.reshape(-1, 1)
        db2 = d_loss.sum()
        dh = d_loss.reshape(-1, 1) @ W2.T  # [n_train, d_h]
        dh *= relu_grad(X_train @ W1 + b1)
        dW1 = X_train.T @ dh
        db1 = dh.sum(axis=0)

        # Update
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

        if epoch % 50 == 0:
            y_test_pred, _ = forward(X_test)
            test_rmse = float(np.sqrt(np.mean((y_test_pred - y_test) ** 2)))
            best_test_rmse = min(best_test_rmse, test_rmse)

    # Final evaluation
    y_test_pred, _ = forward(X_test)
    test_rmse = float(np.sqrt(np.mean((y_test_pred - y_test) ** 2)))
    ss_res = np.sum((y_test_pred - y_test) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    y_all_pred, _ = forward(X_norm)
    all_rmse = float(np.sqrt(np.mean((y_all_pred - y) ** 2)))

    return {
        "test_rmse": test_rmse,
        "test_r_squared": float(r_squared),
        "train_rmse": float(np.sqrt(np.mean((forward(X_train)[0] - y_train) ** 2))),
        "all_rmse": all_rmse,
        "n_train": int(n_train),
        "n_test": int(n - n_train),
        "n_hidden": n_hidden,
        "n_epochs": n_epochs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 9 — Main experiment
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
                                 args.prompt_len + args.k_max + 2, args.data_file)

    # Phase 1: Calibration
    print("\n[Phase 1] Calibration...")
    act_freq = calibrate_freq(model, moe_attr, calib_chunks, args.device, N, top_k)

    all_results = {}
    all_samples_combined = []  # for combined MLP fitting

    for ratio in args.cache_ratios:
        print(f"\n{'='*60}")
        print(f"  Cache ratio = {ratio}")
        print(f"{'='*60}")

        cache = SimulatedCache(L, N, ratio, act_freq)

        # Phase 2: Collect step data
        print("\n[Phase 2] Collecting per-step data...")
        samples = collect_step_data(
            model, moe_attr, eval_chunks, args.device,
            cache, N, top_k, L,
            args.prompt_len, args.k_max, ratio)
        print(f"  Collected {len(samples)} step samples")
        all_samples_combined.extend(samples)

        # Phase 3: Fit analytical model
        print("\n[Phase 3] Fitting analytical model α₀·exp(-λ·E(k))...")
        analytical = fit_analytical_model(samples)
        if "error" not in analytical:
            print(f"  α₀ = {analytical['alpha0']:.4f}, "
                  f"λ = {analytical['lambda']:.4f}")
            print(f"  RMSE = {analytical['rmse']:.4f}, "
                  f"R² = {analytical['r_squared']:.4f}")
        else:
            print(f"  Fit failed: {analytical['error']}")

        # Phase 4: Residual analysis
        print("\n[Phase 4] Residual analysis...")
        residual_corr = analyze_residuals(samples, analytical)
        for fname, corr in residual_corr.items():
            if "error" in corr:
                continue
            print(f"  {fname:>20s}: ρ={corr['spearman_rho']:+.4f}, "
                  f"R²={corr['partial_r2']:.4f}")

        # Phase 5: MLP baseline
        print("\n[Phase 5] Fitting MLP baseline...")
        mlp_result = fit_mlp_model(samples, n_hidden=args.mlp_hidden,
                                   n_epochs=args.mlp_epochs)
        if "error" not in mlp_result:
            print(f"  MLP test RMSE = {mlp_result['test_rmse']:.4f}, "
                  f"R² = {mlp_result['test_r_squared']:.4f}")
            if "rmse" in analytical:
                delta_rmse = analytical["rmse"] - mlp_result["all_rmse"]
                delta_r2 = (mlp_result["test_r_squared"]
                            - analytical.get("r_squared", 0))
                print(f"  ΔRMSE (analytical-MLP) = {delta_rmse:+.4f}")
                print(f"  ΔR² = {delta_r2:+.4f}")
                if delta_r2 > 0.1:
                    print("  → MLP significantly better: recommend upgrade")
                else:
                    print("  → Analytical model sufficient")
        else:
            print(f"  MLP fit failed: {mlp_result['error']}")

        # Save per-ratio results (strip large arrays for JSON)
        ratio_result = {
            "analytical": {k: v for k, v in analytical.items()
                           if k not in ("residuals", "E_values",
                                        "alpha_actual", "alpha_predicted")},
            "residual_correlations": residual_corr,
            "mlp": mlp_result,
            "n_samples": len(samples),
        }
        if "rmse" in analytical and "all_rmse" in mlp_result:
            ratio_result["recommendation"] = (
                "mlp_upgrade" if (mlp_result.get("test_r_squared", 0)
                                  - analytical.get("r_squared", 0)) > 0.1
                else "analytical_sufficient"
            )
        all_results[str(ratio)] = ratio_result

    # ── Combined fit across all ratios ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Combined model (all cache ratios)")
    print(f"{'='*60}")

    analytical_combined = fit_analytical_model(all_samples_combined)
    if "error" not in analytical_combined:
        print(f"  α₀ = {analytical_combined['alpha0']:.4f}, "
              f"λ = {analytical_combined['lambda']:.4f}, "
              f"RMSE = {analytical_combined['rmse']:.4f}, "
              f"R² = {analytical_combined['r_squared']:.4f}")

    mlp_combined = fit_mlp_model(all_samples_combined,
                                  n_hidden=args.mlp_hidden,
                                  n_epochs=args.mlp_epochs)
    if "error" not in mlp_combined:
        print(f"  MLP RMSE = {mlp_combined['all_rmse']:.4f}, "
              f"R² = {mlp_combined['test_r_squared']:.4f}")

    all_results["combined"] = {
        "analytical": {k: v for k, v in analytical_combined.items()
                       if k not in ("residuals", "E_values",
                                    "alpha_actual", "alpha_predicted")},
        "mlp": mlp_combined,
        "n_samples": len(all_samples_combined),
    }

    # ── Save ───────────────────────────────────────────────────────────────────
    print("\n[Phase 6] Saving results...")
    with open(os.path.join(args.outdir, "e5_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # ── Plots ──────────────────────────────────────────────────────────────────

    # Plot 1: E(k) vs α scatter with analytical fit
    fig, axes = plt.subplots(1, len(args.cache_ratios),
                             figsize=(6 * len(args.cache_ratios), 5))
    if len(args.cache_ratios) == 1:
        axes = [axes]
    for ax, ratio in zip(axes, args.cache_ratios):
        ratio_samples = [s for s in all_samples_combined
                         if abs(s.cache_ratio - ratio) < 0.01]
        if not ratio_samples:
            continue
        E = [s.cumulative_error for s in ratio_samples]
        alpha = [s.alpha for s in ratio_samples]
        ax.scatter(E, alpha, alpha=0.3, s=8, color="#2196F3", label="actual")

        # Overlay analytical fit
        key = str(ratio)
        if key in all_results and "alpha0" in all_results[key].get("analytical", {}):
            a0 = all_results[key]["analytical"]["alpha0"]
            lam = all_results[key]["analytical"]["lambda"]
            rmse = all_results[key]["analytical"]["rmse"]
            E_range = np.linspace(0, max(E), 100)
            ax.plot(E_range, a0 * np.exp(-lam * E_range),
                    color="#d32f2f", linewidth=2,
                    label=f"fit: α₀={a0:.2f}, λ={lam:.2f}\nRMSE={rmse:.4f}")

        ax.set_xlabel("Cumulative error E(k)")
        ax.set_ylabel("α")
        ax.set_title(f"r = {ratio}")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "analytical_fit.png"), dpi=150)
    plt.close()

    # Plot 2: Per-step α progression (box plot)
    fig, axes = plt.subplots(1, len(args.cache_ratios),
                             figsize=(6 * len(args.cache_ratios), 5))
    if len(args.cache_ratios) == 1:
        axes = [axes]
    for ax, ratio in zip(axes, args.cache_ratios):
        ratio_samples = [s for s in all_samples_combined
                         if abs(s.cache_ratio - ratio) < 0.01]
        if not ratio_samples:
            continue
        step_data = defaultdict(list)
        for s in ratio_samples:
            step_data[s.step].append(s.alpha)
        steps = sorted(step_data.keys())
        data = [step_data[k] for k in steps]
        bp = ax.boxplot(data, positions=steps, widths=0.6, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#4CAF50")
            patch.set_alpha(0.6)
        ax.set_xlabel("Draft step k")
        ax.set_ylabel("α")
        ax.set_title(f"r = {ratio}")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "alpha_per_step_boxplot.png"), dpi=150)
    plt.close()

    # Plot 3: Residual correlation bar chart
    fig, axes = plt.subplots(1, len(args.cache_ratios),
                             figsize=(6 * len(args.cache_ratios), 5))
    if len(args.cache_ratios) == 1:
        axes = [axes]
    for ax, ratio in zip(axes, args.cache_ratios):
        key = str(ratio)
        if key not in all_results:
            continue
        corr = all_results[key].get("residual_correlations", {})
        names = []
        r2_vals = []
        for fname in ["miss_rate", "critical_miss_rate", "miss_weight_sum",
                       "step", "cache_ratio"]:
            if fname in corr and "partial_r2" in corr[fname]:
                names.append(fname.replace("_", "\n"))
                r2_vals.append(corr[fname]["partial_r2"])
        if names:
            colors = plt.cm.Set2(range(len(names)))
            ax.bar(range(len(names)), r2_vals, color=colors)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, fontsize=7)
            ax.set_ylabel("Partial R² (residual ~ feature)")
            ax.set_title(f"r = {ratio}")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "residual_feature_importance.png"), dpi=150)
    plt.close()

    # Plot 4: Model comparison bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    x_labels = []
    analytical_rmses = []
    mlp_rmses = []
    for ratio in args.cache_ratios:
        key = str(ratio)
        if key not in all_results:
            continue
        x_labels.append(f"r={ratio}")
        a = all_results[key].get("analytical", {})
        m = all_results[key].get("mlp", {})
        analytical_rmses.append(a.get("rmse", float("nan")))
        mlp_rmses.append(m.get("all_rmse", float("nan")))

    x = np.arange(len(x_labels))
    width = 0.35
    ax.bar(x - width / 2, analytical_rmses, width, label="Analytical",
           color="#2196F3")
    ax.bar(x + width / 2, mlp_rmses, width, label="MLP",
           color="#FF9800")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("RMSE")
    ax.set_title("Analytical vs MLP: Prediction Error")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "model_comparison.png"), dpi=150)
    plt.close()

    # ── Final recommendation ───────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  Final Recommendation")
    print(f"{'─'*60}")
    for ratio in args.cache_ratios:
        key = str(ratio)
        if key in all_results and "recommendation" in all_results[key]:
            rec = all_results[key]["recommendation"]
            a_rmse = all_results[key]["analytical"].get("rmse", "N/A")
            m_rmse = all_results[key]["mlp"].get("all_rmse", "N/A")
            print(f"  r={ratio}: {rec}  "
                  f"(analytical RMSE={a_rmse}, MLP RMSE={m_rmse})")

    print(f"\nResults saved to {args.outdir}/")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="E5: α prediction eval")
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
    parser.add_argument("--mlp_hidden", type=int, default=16)
    parser.add_argument("--mlp_epochs", type=int, default=200)
    parser.add_argument("--outdir", default="./results_e5")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
