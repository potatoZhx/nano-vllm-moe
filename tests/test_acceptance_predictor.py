"""Unit tests for the on-GPU acceptance predictor (no mocks).

These exercise the *real* ``AcceptancePredictor`` module and
``DraftAcceptanceFeatureExtractor`` feature math against the numpy reference in
``random_cache_srdp_scripts-1/build_random_cache_dataset.py``. They run on CUDA
when available and otherwise on CPU (the feature ops are device-agnostic); no
component is mocked. The full CUDA-graph capture/replay path is covered by the
env-gated real-model integration test in
``tests/test_acceptance_predictor_integration.py``.

Run:
    python -m pytest tests/test_acceptance_predictor.py -v
or:
    python -m unittest tests.test_acceptance_predictor -v
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "random_cache_srdp_scripts-1"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRIPTS))

import build_random_cache_dataset as ref  # noqa: E402
from nanovllm.engine.speculative.acceptance_predictor import (  # noqa: E402
    AcceptancePredictor,
    DraftAcceptanceFeatureExtractor,
    PredictorMeta,
    load_acceptance_predictor,
)

L = 48
K = 8
H = 2048
NUM_EXPERTS = 128
VOCAB = 4096
TOL = 2e-3
CHECKPOINT_DIR = SCRIPTS / "res" / "run_20260614_133025"


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_step(rng: np.random.Generator):
    router = []
    for li in range(L):
        orig_ids = rng.choice(NUM_EXPERTS, size=K, replace=False)
        ow = rng.random(K).astype(np.float32)
        ow = ow / ow.sum()
        mod_ids = orig_ids.copy()
        for slot in rng.choice(K, size=int(rng.integers(0, 4)), replace=False):
            mod_ids[slot] = int(rng.integers(0, NUM_EXPERTS))
        mw = rng.random(K).astype(np.float32)
        mw = mw / mw.sum()
        mask = (mod_ids != orig_ids).astype(np.float32)
        router.append(
            {
                "layer_idx": li,
                "original_ids": [orig_ids.tolist()],
                "original_weights": [ow.tolist()],
                "modified_ids": [mod_ids.tolist()],
                "modified_weights": [mw.tolist()],
                "replacement_mask": [mask.tolist()],
            }
        )
    logits = rng.standard_normal(VOCAB).astype(np.float32)
    topk = np.sort(logits)[::-1][:32]
    step = {
        "router": router,
        "q_topk_logits": {"topk": 32, "indices": [], "values": topk.tolist()},
        "final_embedding": rng.standard_normal(H).astype(np.float32).tolist(),
        "alpha_theoretical": 0.5,
    }
    return step, logits


def _meta() -> PredictorMeta:
    return PredictorMeta(
        num_layers=L, top_k=K, route_raw_dim=2 * L * K,
        route_summary_dim=45, token_feature_dim=10, hidden_dim=H, history_dim=11,
    )


def _build_extractor(device: torch.device, max_bs: int, step_horizon: int = 32):
    meta = _meta()
    predictor = AcceptancePredictor(
        route_raw_dim=meta.route_raw_dim, route_summary_dim=meta.route_summary_dim,
        token_feature_dim=meta.token_feature_dim, hidden_dim=meta.hidden_dim,
        history_dim=meta.history_dim,
    ).to(device=device, dtype=torch.float32).eval()
    ext = DraftAcceptanceFeatureExtractor(
        predictor, meta, max_bs=max_bs, hidden_size=H, device=device, step_horizon=step_horizon,
    )
    return ext


def _fill_row(ext: DraftAcceptanceFeatureExtractor, row: int, step: dict) -> None:
    oi, mi, ow, mw, _ = ref.extract_route_arrays(step, num_layers=L, top_k=K)
    ext.orig_ids[:, row, :].copy_(torch.from_numpy(oi).to(ext.orig_ids.device))
    ext.mod_ids[:, row, :].copy_(torch.from_numpy(mi).to(ext.mod_ids.device))
    ext.orig_w[:, row, :].copy_(torch.from_numpy(ow).to(ext.orig_w.device))
    ext.mod_w[:, row, :].copy_(torch.from_numpy(mw).to(ext.mod_w.device))


class _FakeSeq:
    def __init__(self, seq_id: int, num_prompt_tokens: int):
        self.seq_id = seq_id
        self.num_prompt_tokens = num_prompt_tokens


class TestPredictorModule(unittest.TestCase):
    def test_forward_shape_and_range(self):
        device = _device()
        ext = _build_extractor(device, max_bs=4)
        n = 4
        route_raw = torch.randn(n, 2 * L * K, device=device)
        route_summary = torch.randn(n, 45, device=device)
        token_features = torch.randn(n, 10, device=device)
        hidden = torch.randn(n, H, device=device)
        history = torch.randn(n, 11, device=device)
        out = ext.predictor(route_raw, route_summary, token_features, hidden, history)
        self.assertEqual(tuple(out.shape), (n, 1))
        self.assertTrue(torch.all(out >= 0.0) and torch.all(out <= 1.0))

    @unittest.skipUnless(CHECKPOINT_DIR.is_dir(), f"checkpoint not found: {CHECKPOINT_DIR}")
    def test_load_real_checkpoint(self):
        device = _device()
        model, meta = load_acceptance_predictor(str(CHECKPOINT_DIR), device)
        self.assertEqual(meta.num_layers, L)
        self.assertEqual(meta.top_k, K)
        self.assertEqual(meta.hidden_dim, H)
        # parameters must be fp32 after load
        self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))
        out = model(
            torch.randn(2, meta.route_raw_dim, device=device),
            torch.randn(2, meta.route_summary_dim, device=device),
            torch.randn(2, meta.token_feature_dim, device=device),
            torch.randn(2, meta.hidden_dim, device=device),
            torch.randn(2, meta.history_dim, device=device),
        )
        self.assertEqual(tuple(out.shape), (2, 1))
        self.assertTrue(torch.all((out >= 0.0) & (out <= 1.0)))


class TestFeatureParity(unittest.TestCase):
    def test_token_features_do_not_require_full_vocab_fp32_materialization(self):
        device = _device()
        ext = _build_extractor(device, max_bs=2)
        logits = torch.randn(2, VOCAB, device=device, dtype=torch.bfloat16)
        reference_topk = torch.topk(logits.float(), k=32, dim=-1).values

        ext.set_token_features_from_logits(logits)

        top1 = reference_topk[:, 0]
        top2 = reference_topk[:, 1]
        shifted = reference_topk - reference_topk.max(
            dim=1, keepdim=True
        ).values
        probs = torch.exp(shifted)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=1)
        expected = torch.stack(
            [
                top1,
                top2,
                top1 - top2,
                reference_topk[:, :5].mean(dim=1),
                top1 - reference_topk[:, :5].mean(dim=1),
                reference_topk.std(dim=1, unbiased=False),
                probs[:, 0],
                entropy,
                torch.exp(entropy),
                torch.full_like(top1, 32.0),
            ],
            dim=1,
        )
        self.assertTrue(
            torch.allclose(
                ext.token_features_buf[:2],
                expected,
                atol=1e-6,
                rtol=1e-6,
            )
        )

    def test_read_outputs_can_return_original_routes_in_same_transfer(self):
        device = _device()
        rng = np.random.default_rng(11)
        ext = _build_extractor(device, max_bs=2)
        seqs = [_FakeSeq(seq_id=1, num_prompt_tokens=3), _FakeSeq(seq_id=2, num_prompt_tokens=4)]
        expected = []
        for row in range(2):
            step, _ = _make_step(rng)
            _fill_row(ext, row, step)
            expected.append(
                np.asarray(
                    [layer["original_ids"][0] for layer in step["router"]],
                    dtype=np.int64,
                )
            )
        ext.original_route_readback_enabled = True
        ext.alpha_buf[:2].copy_(torch.tensor([0.25, 0.75], device=device))
        ext._pack_outputs(2)

        alphas, routes = ext.read_outputs(seqs, include_original_routes=True)

        self.assertEqual(alphas, [0.25, 0.75])
        self.assertIsInstance(routes, np.ndarray)
        routes_array = np.asarray(routes, dtype=np.int64).transpose(1, 0, 2)
        self.assertTrue(np.array_equal(routes_array[0], expected[0]))
        self.assertTrue(np.array_equal(routes_array[1], expected[1]))

    def test_route_and_token_features_batched(self):
        device = _device()
        rng = np.random.default_rng(1)
        bs = 3
        ext = _build_extractor(device, max_bs=bs)
        steps = []
        logits_rows = []
        for r in range(bs):
            step, logits = _make_step(rng)
            _fill_row(ext, r, step)
            steps.append(step)
            logits_rows.append(logits)

        route_raw, route_summary, rsd, rep = ext._build_route_features(bs)
        logits_batch = torch.from_numpy(np.stack(logits_rows)).to(device)
        ext.set_token_features_from_logits(logits_batch)

        for r in range(bs):
            ref_raw, ref_sum, ref_scalars = ref.route_features(steps[r], num_layers=L, top_k=K)
            self.assertTrue(
                np.allclose(route_raw[r].cpu().numpy(), ref_raw, atol=TOL),
                f"route_raw row {r}",
            )
            self.assertTrue(
                np.allclose(route_summary[r].cpu().numpy(), ref_sum, atol=TOL),
                f"route_summary row {r} max="
                f"{np.abs(route_summary[r].cpu().numpy() - ref_sum).max()}",
            )
            self.assertAlmostEqual(float(rsd[r]), ref_scalars["rsd_l1"], delta=TOL)
            self.assertAlmostEqual(float(rep[r]), ref_scalars["rep_mass"], delta=TOL)

            ref_tok = ref.token_features(steps[r])
            got_tok = ext.token_features_buf[r].cpu().numpy()
            self.assertTrue(
                np.allclose(got_tok, ref_tok, atol=TOL),
                f"token_features row {r} max={np.abs(got_tok - ref_tok).max()}",
            )

    def test_history_recurrence_and_carry(self):
        device = _device()
        rng = np.random.default_rng(2)
        ext = _build_extractor(device, max_bs=1, step_horizon=32)
        seq = _FakeSeq(seq_id=7, num_prompt_tokens=123)

        cum_rsd = ema_rsd = max_rsd = 0.0
        cum_rep = ema_rep = max_rep = 0.0
        cum_ent = ema_ent = max_ent = 0.0
        for i in range(5):
            step, logits = _make_step(rng)
            _fill_row(ext, 0, step)
            ext.set_token_features_from_logits(torch.from_numpy(logits)[None, :].to(device))
            ext.write_state_in([seq])
            hidden = torch.from_numpy(np.asarray(step["final_embedding"], np.float32))[None, :].to(device)
            ext.run_predictor(1, hidden)
            alpha = ext.read_outputs([seq])[0]
            self.assertGreaterEqual(alpha, 0.0)
            self.assertLessEqual(alpha, 1.0)

            _, _, sc = ref.route_features(step, num_layers=L, top_k=K)
            tok = ref.token_features(step)
            rsd_ref, rep_ref, ent_ref = sc["rsd_l1"], sc["rep_mass"], float(tok[7])
            t = i + 1
            cum_rsd += rsd_ref
            ema_rsd = 0.7 * ema_rsd + 0.3 * rsd_ref if i > 0 else rsd_ref
            max_rsd = max(max_rsd, rsd_ref)
            cum_rep += rep_ref
            ema_rep = 0.7 * ema_rep + 0.3 * rep_ref if i > 0 else rep_ref
            max_rep = max(max_rep, rep_ref)
            cum_ent += ent_ref
            ema_ent = 0.7 * ema_ent + 0.3 * ent_ref if i > 0 else ent_ref
            max_ent = max(max_ent, ent_ref)

            s = ext._host_state[7]
            got_hist = np.asarray([
                s[0] / 32.0, s[1] / s[0], s[2], s[3], s[4] / s[0], s[5], s[6],
                s[7] / s[0], s[8], s[9], s[10] / 8096.0,
            ], dtype=np.float32)
            ref_hist = np.asarray([
                t / 32.0, cum_rsd / t, ema_rsd, max_rsd, cum_rep / t, ema_rep, max_rep,
                cum_ent / t, ema_ent, max_ent, 123.0 / 8096.0,
            ], dtype=np.float32)
            self.assertTrue(
                np.allclose(got_hist, ref_hist, atol=TOL),
                f"history step {i} max={np.abs(got_hist - ref_hist).max()}",
            )

        # forget() drops carried state.
        ext.forget([7])
        self.assertNotIn(7, ext._host_state)

    def test_history_resets_per_new_sequence(self):
        device = _device()
        rng = np.random.default_rng(3)
        ext = _build_extractor(device, max_bs=2)
        seq_a = _FakeSeq(seq_id=1, num_prompt_tokens=10)
        step, logits = _make_step(rng)
        _fill_row(ext, 0, step)
        ext.set_token_features_from_logits(torch.from_numpy(logits)[None, :].to(device))
        ext.write_state_in([seq_a])
        # first step => state_in is zeroed for a fresh seq => t starts at 0
        self.assertAlmostEqual(float(ext.state_in_buf[0, 0].cpu()), 0.0, delta=1e-6)
        self.assertAlmostEqual(float(ext.state_in_buf[0, 10].cpu()), 10.0, delta=1e-6)


class TestRecordLayerCapturesBothRoutings(unittest.TestCase):
    def test_replacement_mask_from_ids(self):
        device = _device()
        ext = _build_extractor(device, max_bs=1)
        # original and modified differ in exactly 2 slots of layer 0.
        orig_ids = torch.arange(K, device=device).view(1, K)
        mod_ids = orig_ids.clone()
        mod_ids[0, 0] = 99
        mod_ids[0, 3] = 88
        ow = torch.full((1, K), 1.0 / K, device=device)
        mw = torch.full((1, K), 1.0 / K, device=device)
        for li in range(L):
            ext.record_layer(li, orig_ids, ow, mod_ids, mw)
        _, route_summary, _, _ = ext._build_route_features(1)
        # rep_frac mean (column index 2 in layer_matrix) -> first30 triplet start at 6.
        rep_frac_mean = float(route_summary[0, 6].cpu())
        self.assertAlmostEqual(rep_frac_mean, 2.0 / K, delta=1e-5)


if __name__ == "__main__":
    unittest.main()
