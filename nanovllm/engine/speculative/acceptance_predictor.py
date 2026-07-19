"""On-GPU theoretical-acceptance predictor for the speculative draft path.

This module mirrors the lightweight branch-encoder MLP trained by
``random_cache_srdp_scripts-1/train_acceptance_predictor.py`` and reproduces the
exact feature math of ``build_random_cache_dataset.py`` with pure PyTorch tensor
ops so the whole thing can run on-GPU inside a CUDA graph during draft decode.

Design (see plan ``random-cache-srdp-scripts-1-nanovllm-sp-transient-tower``):

* ``AcceptancePredictor`` is an architecture-identical copy of the training module
  so nano-vllm has no import dependency on the scripts directory.
* ``DraftAcceptanceFeatureExtractor`` owns persistent GPU buffers and graph-safe
  ops. Per-layer routing (original vs draft-modified top-k experts/weights) is
  written into ``[num_layers, max_bs, top_k]`` buffers from inside the captured
  draft *segment* graphs (mirroring ``ModelRuntimeMetaRecorder.record_layer``).
  ``token_features`` are derived eagerly from the (already eager) draft logits and
  staged into a static buffer. ``route_raw`` / ``route_summary`` / the decode
  ``history`` recurrence and the predictor MLP run inside a captured *tail* graph
  reading only static buffers and emitting ``alpha`` (predicted acceptance) plus
  the updated per-sequence history state.
* The per-sequence history *state* is carried across draft steps on the host
  (keyed by ``seq_id`` so it survives batch re-ordering); it is fed in through a
  static input buffer (tiny H2D) and read back fused with ``alpha`` in a single
  post-sync D2H. This is the lowest-latency option that stays correct under batch
  churn (see plan §4).

Everything here is gated by ``config.acceptance_predictor_enabled`` upstream; when
disabled this module is never constructed and the draft path is untouched.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------- #
#  Predictor module (architecture-identical to the trained checkpoint)
# ---------------------------------------------------------------------------- #
class AcceptancePredictor(nn.Module):
    """Small branch-encoder MLP for alpha = sum(min(p, q)) prediction.

    Architecture copied verbatim from
    ``random_cache_srdp_scripts-1/train_acceptance_predictor.py`` so that
    ``load_state_dict`` matches the saved checkpoint key-for-key.
    """

    def __init__(
        self,
        route_raw_dim: int,
        route_summary_dim: int,
        token_feature_dim: int,
        hidden_dim: int,
        history_dim: int,
        route_raw_embed: int = 32,
        hidden_embed: int = 32,
    ) -> None:
        super().__init__()
        self.route_raw_encoder = nn.Sequential(
            nn.LayerNorm(route_raw_dim),
            nn.Linear(route_raw_dim, route_raw_embed),
            nn.SiLU(),
        )
        self.route_summary_encoder = nn.Sequential(
            nn.LayerNorm(route_summary_dim),
            nn.Linear(route_summary_dim, 32),
            nn.SiLU(),
        )
        self.token_encoder = nn.Sequential(
            nn.LayerNorm(token_feature_dim),
            nn.Linear(token_feature_dim, 16),
            nn.SiLU(),
        )
        self.hidden_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_embed),
            nn.SiLU(),
        )
        self.history_encoder = nn.Sequential(
            nn.LayerNorm(history_dim),
            nn.Linear(history_dim, 16),
            nn.SiLU(),
        )
        fused_dim = route_raw_embed + 32 + 16 + hidden_embed + 16
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, route_raw, route_summary, token_features, hidden, history):
        z = torch.cat(
            [
                self.route_raw_encoder(route_raw),
                self.route_summary_encoder(route_summary),
                self.token_encoder(token_features),
                self.hidden_encoder(hidden),
                self.history_encoder(history),
            ],
            dim=-1,
        )
        return self.head(z)


@dataclass
class PredictorMeta:
    num_layers: int
    top_k: int
    route_raw_dim: int
    route_summary_dim: int
    token_feature_dim: int
    hidden_dim: int
    history_dim: int


def load_acceptance_predictor(
    path: str,
    device: torch.device,
) -> tuple[AcceptancePredictor, PredictorMeta]:
    """Load the trained predictor from ``path`` (dir with config.json + best_model.pth
    or a direct .pth file alongside config.json). Returns (module, meta) in fp32/eval.
    """
    if os.path.isdir(path):
        config_path = os.path.join(path, "config.json")
        weight_path = os.path.join(path, "best_model.pth")
    else:
        weight_path = path
        config_path = os.path.join(os.path.dirname(path), "config.json")

    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"acceptance predictor weights not found: {weight_path}")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"acceptance predictor config.json not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    meta_raw = cfg.get("meta", cfg)
    meta = PredictorMeta(
        num_layers=int(meta_raw["num_layers"]),
        top_k=int(meta_raw["top_k"]),
        route_raw_dim=int(meta_raw["route_raw_dim"]),
        route_summary_dim=int(meta_raw["route_summary_dim"]),
        token_feature_dim=int(meta_raw["token_feature_dim"]),
        hidden_dim=int(meta_raw["hidden_dim"]),
        history_dim=int(meta_raw["history_dim"]),
    )

    model = AcceptancePredictor(
        route_raw_dim=meta.route_raw_dim,
        route_summary_dim=meta.route_summary_dim,
        token_feature_dim=meta.token_feature_dim,
        hidden_dim=meta.hidden_dim,
        history_dim=meta.history_dim,
    )
    # Cast to fp32 BEFORE loading so the fp32 checkpoint is not downcast through a
    # bf16 default dtype (the model runner sets torch's default dtype to bf16).
    model = model.float()
    state = torch.load(weight_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and not any(
        k.startswith(("route_raw_encoder", "head")) for k in state
    ):
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device=device, dtype=torch.float32)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, meta


# ---------------------------------------------------------------------------- #
#  Feature extractor + on-GPU buffers
# ---------------------------------------------------------------------------- #
# History state layout (carried across draft steps on the host, fed in/out via
# static buffers). 11 slots, transformed into the 11-dim `history` feature below.
_STATE_DIM = 11
_S_T = 0          # step counter t
_S_CUM_RSD = 1
_S_EMA_RSD = 2
_S_MAX_RSD = 3
_S_CUM_REP = 4
_S_EMA_REP = 5
_S_MAX_REP = 6
_S_CUM_ENT = 7
_S_EMA_ENT = 8
_S_MAX_ENT = 9
_S_PREFILL = 10

_EMA = 0.3
_PREFILL_NORM = 8096.0


class DraftAcceptanceFeatureExtractor:
    """Owns persistent GPU buffers and graph-safe feature/predictor ops.

    Lifecycle per draft step:
      1. ``write_state_in(seqs)``            (host->device, before the draft forward)
      2. segment graphs replay -> ``record_layer`` writes routing buffers (captured)
      3. ``set_token_features_from_logits``  (eager, after eager LM head)
      4. tail graph replay -> ``run_predictor`` (captured) -> alpha + new state
      5. ``read_outputs(seqs)``              (device->host, after the sampler sync)
    """

    def __init__(
        self,
        predictor: AcceptancePredictor,
        meta: PredictorMeta,
        *,
        max_bs: int,
        hidden_size: int,
        device: torch.device,
        step_horizon: int = 32,
    ) -> None:
        self.predictor = predictor
        self.meta = meta
        self.max_bs = int(max_bs)
        self.device = device
        self.L = int(meta.num_layers)
        self.K = int(meta.top_k)
        self.H = int(meta.hidden_dim)
        self.step_horizon = float(max(1, step_horizon))
        if hidden_size != self.H:
            raise ValueError(
                f"acceptance predictor hidden_dim={self.H} != model hidden_size={hidden_size}"
            )

        f32 = torch.float32
        # Per-layer routing buffers (written from inside segment graphs).
        self.orig_w = torch.zeros(self.L, self.max_bs, self.K, dtype=f32, device=device)
        self.mod_w = torch.zeros(self.L, self.max_bs, self.K, dtype=f32, device=device)
        self.orig_ids = torch.zeros(self.L, self.max_bs, self.K, dtype=torch.int64, device=device)
        self.mod_ids = torch.zeros(self.L, self.max_bs, self.K, dtype=torch.int64, device=device)
        # token_features staged eagerly from draft logits.
        self.token_features_buf = torch.zeros(self.max_bs, meta.token_feature_dim, dtype=f32, device=device)
        # History state in/out (carried on host across steps).
        self.state_in_buf = torch.zeros(self.max_bs, _STATE_DIM, dtype=f32, device=device)
        self.state_out_buf = torch.zeros(self.max_bs, _STATE_DIM, dtype=f32, device=device)
        # Outputs.
        self.alpha_buf = torch.zeros(self.max_bs, dtype=f32, device=device)
        self._output_route_offset = 1 + _STATE_DIM
        self._output_width = self._output_route_offset + self.L * self.K
        self.output_buf = torch.zeros(
            self.max_bs,
            self._output_width,
            dtype=f32,
            device=device,
        )
        # Pinned host staging for the per-step transfers (explicit CPU device: the
        # model runner sets a CUDA default device, and pin_memory requires CPU).
        pin = device.type == "cuda"
        self.state_in_host = torch.zeros(self.max_bs, _STATE_DIM, dtype=f32, device="cpu", pin_memory=pin)
        self.output_host = torch.zeros(
            self.max_bs,
            self._output_width,
            dtype=f32,
            device="cpu",
            pin_memory=pin,
        )
        self._output_host_array = self.output_host.numpy()
        self.original_route_readback_enabled = False

        # np.array_split(arange(L), 3) block boundaries for the early/mid/late aggregates.
        thirds = np.array_split(np.arange(self.L), 3)
        self.block_slices = [(int(t[0]), int(t[-1]) + 1) for t in thirds if len(t) > 0]

        # Host-side per-sequence history state, keyed by seq_id.
        self._host_state: dict[int, np.ndarray] = {}

        self.enabled = True

    # ----------------------------- attach -------------------------------- #
    def attach(self, model) -> None:
        if not hasattr(model, "set_draft_feature_recorder"):
            raise AttributeError("model does not support set_draft_feature_recorder")
        moe_layers = model.set_draft_feature_recorder(self)
        if isinstance(moe_layers, int) and moe_layers != self.L:
            raise ValueError(
                f"acceptance predictor expects num_layers={self.L} MoE blocks, "
                f"but the model has {moe_layers}; checkpoint/model mismatch"
            )

    # ----------------- per-layer routing capture (graph-safe) ------------ #
    def record_layer(
        self,
        layer_idx: int,
        original_experts: torch.Tensor,
        original_weights: torch.Tensor,
        modified_experts: torch.Tensor,
        modified_weights: torch.Tensor,
    ) -> None:
        """Write one layer's routing slice. Pure indexed copies -> capture-safe."""
        if not (0 <= layer_idx < self.L):
            return
        bs = min(int(original_experts.size(0)), self.max_bs)
        if bs <= 0:
            return
        self.orig_ids[layer_idx, :bs].copy_(original_experts[:bs])
        self.mod_ids[layer_idx, :bs].copy_(modified_experts[:bs])
        self.orig_w[layer_idx, :bs].copy_(original_weights[:bs])
        self.mod_w[layer_idx, :bs].copy_(modified_weights[:bs])

    # ----------------- token features (eager, on-GPU) -------------------- #
    def set_token_features_from_logits(self, logits: torch.Tensor) -> None:
        """Reproduce build_random_cache_dataset.token_features() on GPU.

        ``logits`` is [bs, vocab]; stats are over the top-32 logits per row.
        """
        bs = min(int(logits.size(0)), self.max_bs)
        if bs <= 0:
            return
        k = min(32, int(logits.size(-1)))
        # The logits are already quantized to their output dtype. Selecting in
        # that dtype and casting only the top-k values preserves the feature
        # inputs while avoiding a full-vocabulary FP32 temporary every draft step.
        vals = torch.topk(logits[:bs], k=k, dim=-1).values.float()
        top1 = vals[:, 0]
        top2 = vals[:, 1] if k > 1 else top1
        margin12 = top1 - top2
        top5_mean = vals[:, : min(5, k)].mean(dim=1)
        top1_gap5 = top1 - top5_mean
        std = vals.std(dim=1, unbiased=False)
        shifted = vals - vals.max(dim=1, keepdim=True).values
        probs = torch.exp(shifted)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
        top1_prob = probs[:, 0]
        ent = -(probs * torch.log(probs + 1e-9)).sum(dim=1)
        eff_vocab = torch.exp(ent)
        kfull = torch.full_like(top1, float(k))
        feats = torch.stack(
            [top1, top2, margin12, top5_mean, top1_gap5, std, top1_prob, ent, eff_vocab, kfull],
            dim=1,
        )
        self.token_features_buf[:bs].copy_(feats)

    # ----------------- derived feature helpers (graph-safe) -------------- #
    @staticmethod
    def _entropy_over_k(w: torch.Tensor) -> torch.Tensor:
        """entropy() of build_random_cache_dataset over the last (K) dim.

        normalize_probs: clamp>=0, divide by sum (uniform if ~0), then Shannon
        entropy with the +1e-9 stabiliser. ``w`` is [..., K] -> [...].
        """
        p = w.clamp_min(0.0)
        s = p.sum(dim=-1, keepdim=True)
        kdim = w.size(-1)
        uniform = torch.full_like(p, 1.0 / max(1, kdim))
        p = torch.where(s <= 1e-8, uniform, p / s.clamp_min(1e-8))
        return -(p * torch.log(p + 1e-9)).sum(dim=-1)

    def _build_route_features(self, bs: int):
        """Return (route_raw[bs,768], route_summary[bs,45], rsd[bs], rep[bs])."""
        ow = self.orig_w[:, :bs, :]  # [L, bs, K]
        mw = self.mod_w[:, :bs, :]
        oi = self.orig_ids[:, :bs, :]
        mi = self.mod_ids[:, :bs, :]

        # route_raw = concat(orig_w.reshape(-1), mod_w.reshape(-1)) per row.
        # ow is [L, bs, K] -> [bs, L*K] (layer-major, matching np .reshape(-1)).
        route_raw = torch.cat(
            [
                ow.permute(1, 0, 2).reshape(bs, self.L * self.K),
                mw.permute(1, 0, 2).reshape(bs, self.L * self.K),
            ],
            dim=1,
        )

        diff = (ow - mw).abs()                       # [L, bs, K]
        l1 = diff.sum(dim=-1)                         # [L, bs]
        l2 = torch.sqrt((diff * diff).sum(dim=-1))
        mask = (mi != oi).to(ow.dtype)               # replacement_mask
        rep_frac = mask.mean(dim=-1)
        rep_mass = (ow * mask).sum(dim=-1)
        ent_o = self._entropy_over_k(ow)
        ent_s = self._entropy_over_k(mw)
        ent_delta = ent_s - ent_o
        top1_o = ow.max(dim=-1).values
        top2 = torch.topk(ow, k=min(2, self.K), dim=-1).values
        margin_o = (top2[..., 0] - top2[..., 1]) if self.K >= 2 else torch.zeros_like(top1_o)
        max_drop = diff.max(dim=-1).values

        # layer_matrix columns in the exact training order.
        layer_matrix = torch.stack(
            [l1, l2, rep_frac, rep_mass, ent_o, ent_s, ent_delta, top1_o, margin_o, max_drop],
            dim=-1,
        )  # [L, bs, 10]

        mean_all = layer_matrix.mean(dim=0)                       # [bs, 10]
        max_all = layer_matrix.max(dim=0).values                 # [bs, 10]
        std_all = layer_matrix.std(dim=0, unbiased=False)        # [bs, 10] (population std)
        first30 = torch.stack([mean_all, max_all, std_all], dim=2).reshape(bs, 30)

        block_feats = []
        for (lo, hi) in self.block_slices:
            block_feats.append(l1[lo:hi].mean(dim=0))
            block_feats.append(rep_mass[lo:hi].mean(dim=0))
            block_feats.append(rep_frac[lo:hi].mean(dim=0))
            block_feats.append(ent_o[lo:hi].mean(dim=0))
            block_feats.append(max_drop[lo:hi].mean(dim=0))
        block15 = torch.stack(block_feats, dim=1)                # [bs, 15]

        route_summary = torch.cat([first30, block15], dim=1)     # [bs, 45]
        rsd = l1.mean(dim=0)                                      # scalars["rsd_l1"]
        rep = rep_mass.mean(dim=0)                                # scalars["rep_mass"]
        return route_raw, route_summary, rsd, rep

    def _update_history(self, bs: int, rsd: torch.Tensor, rep: torch.Tensor, ent_tok: torch.Tensor):
        """Apply the per-step EMA/cum/max recurrence; return (history[bs,11], new_state[bs,11])."""
        st = self.state_in_buf[:bs]
        t1 = st[:, _S_T] + 1.0
        is_first = t1 <= 1.5
        cum_rsd = st[:, _S_CUM_RSD] + rsd
        ema_rsd = torch.where(is_first, rsd, (1.0 - _EMA) * st[:, _S_EMA_RSD] + _EMA * rsd)
        max_rsd = torch.maximum(st[:, _S_MAX_RSD], rsd)
        cum_rep = st[:, _S_CUM_REP] + rep
        ema_rep = torch.where(is_first, rep, (1.0 - _EMA) * st[:, _S_EMA_REP] + _EMA * rep)
        max_rep = torch.maximum(st[:, _S_MAX_REP], rep)
        cum_ent = st[:, _S_CUM_ENT] + ent_tok
        ema_ent = torch.where(is_first, ent_tok, (1.0 - _EMA) * st[:, _S_EMA_ENT] + _EMA * ent_tok)
        max_ent = torch.maximum(st[:, _S_MAX_ENT], ent_tok)
        prefill = st[:, _S_PREFILL]

        new_state = torch.stack(
            [t1, cum_rsd, ema_rsd, max_rsd, cum_rep, ema_rep, max_rep, cum_ent, ema_ent, max_ent, prefill],
            dim=1,
        )
        history = torch.stack(
            [
                t1 / self.step_horizon,
                cum_rsd / t1,
                ema_rsd,
                max_rsd,
                cum_rep / t1,
                ema_rep,
                max_rep,
                cum_ent / t1,
                ema_ent,
                max_ent,
                prefill / _PREFILL_NORM,
            ],
            dim=1,
        )
        return history, new_state

    # ----------------- the captured tail computation --------------------- #
    def run_predictor(self, bs: int, hidden: torch.Tensor) -> None:
        """Captured tail graph body: static buffers -> alpha_buf + state_out_buf.

        ``hidden`` is the final post-norm hidden state [bs, H] (a static buffer
        slice). All inputs are static buffers so this is fully CUDA-graph safe.
        """
        route_raw, route_summary, rsd, rep = self._build_route_features(bs)
        token_features = self.token_features_buf[:bs]
        ent_tok = token_features[:, 7]
        history, new_state = self._update_history(bs, rsd, rep, ent_tok)
        hidden_f = hidden.to(torch.float32)
        alpha = self.predictor(route_raw, route_summary, token_features, hidden_f, history)
        self.alpha_buf[:bs].copy_(alpha.squeeze(-1))
        self.state_out_buf[:bs].copy_(new_state)
        self._pack_outputs(bs)

    def _pack_outputs(self, bs: int) -> None:
        """Write the fixed-layout D2H buffer from capture-safe GPU buffers."""
        self.output_buf[:bs, 0].copy_(self.alpha_buf[:bs])
        self.output_buf[:bs, 1:self._output_route_offset].copy_(
            self.state_out_buf[:bs]
        )
        if self.original_route_readback_enabled:
            self.output_buf[:bs, self._output_route_offset:].copy_(
                self.orig_ids[:, :bs, :]
                .permute(1, 0, 2)
                .reshape(bs, self.L * self.K)
            )

    # ----------------- host-side state carry ----------------------------- #
    def write_state_in(self, seqs) -> None:
        """Assemble per-seq state for this batch and stage it into state_in_buf."""
        bs = min(len(seqs), self.max_bs)
        host = self.state_in_host[:bs]
        host.zero_()
        for i, seq in enumerate(seqs[:bs]):
            sid = int(seq.seq_id)
            prev = self._host_state.get(sid)
            if prev is None:
                prev = np.zeros(_STATE_DIM, dtype=np.float32)
                prev[_S_PREFILL] = float(getattr(seq, "num_prompt_tokens", 0))
                self._host_state[sid] = prev
            host[i].copy_(torch.from_numpy(prev))
        self.state_in_buf[:bs].copy_(host, non_blocking=(self.device.type == "cuda"))

    def read_outputs(
        self,
        seqs,
        *,
        include_original_routes: bool = False,
    ) -> list[float] | tuple[list[float], np.ndarray]:
        """Read predictor outputs and optionally a transient ``[L, bs, K]`` view.

        ``run_predictor`` packs the output inside the captured tail graph. The
        returned route view aliases the pinned staging buffer and is valid until
        the next call to this method; callers must consume it immediately.
        """
        bs = min(len(seqs), self.max_bs)
        if bs <= 0:
            empty_routes = np.empty((self.L, 0, self.K), dtype=np.float32)
            return ([], empty_routes) if include_original_routes else []
        if include_original_routes and not self.original_route_readback_enabled:
            raise RuntimeError("original route readback was not enabled before graph capture")
        # A blocking copy is deliberate: the caller consumes the numpy view on
        # the CPU immediately. Both source and destination are persistent, so no
        # tensor allocation or Python list materialization occurs on this path.
        width = self._output_width if include_original_routes else self._output_route_offset
        self.output_host[:bs, :width].copy_(
            self.output_buf[:bs, :width], non_blocking=False
        )
        packed_cpu = self._output_host_array[:bs]
        alphas: list[float] = []
        for i, seq in enumerate(seqs[:bs]):
            alphas.append(float(packed_cpu[i, 0]))
            self._host_state[int(seq.seq_id)] = packed_cpu[
                i, 1:self._output_route_offset
            ].copy()
        if not include_original_routes:
            return alphas
        routes = (
            packed_cpu[:, self._output_route_offset:]
            .reshape(bs, self.L, self.K)
            .transpose(1, 0, 2)
        )
        return alphas, routes

    def forget(self, seq_ids) -> None:
        for sid in seq_ids:
            self._host_state.pop(int(sid), None)
