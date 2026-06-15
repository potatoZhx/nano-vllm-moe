"""Unit tests for the TPOT-based dynamic draft-length stop policy.

These exercise the pure cost model ``expected_tpot_ms`` used by
``SpeculativeEngine.speculative_step`` (cost-aware dynamic draft length) without
needing a GPU or a real model. The engine's reactive-lookback stop rule (stop the
draft loop once ``T(k) > T(k-1)``) is replicated here against the same function the
engine calls, and checked against a brute-force argmin of ``T(k)``.

Run:
    python -m pytest tests/test_draft_tpot_stop.py -v
or:
    python -m unittest tests.test_draft_tpot_stop -v
"""

from __future__ import annotations

import sys
import unittest
from math import prod
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nanovllm.engine.speculative.spec_engine import expected_tpot_ms


def brute_force_tpot(step_alphas, num_seqs, td, tv):
    """Reference T(k): explicit cumulative-product acc_len, no shared code."""
    acc_len_sum = 0.0
    for s in range(num_seqs):
        for i in range(len(step_alphas)):
            acc_len_sum += prod(step_alphas[j][s] for j in range(i + 1))
    return (len(step_alphas) * td + tv) / (acc_len_sum + num_seqs)


def reactive_stop_step(per_step_alphas, num_seqs, td, tv):
    """Mirror the engine's reactive-lookback decision; return drafted-step count."""
    history = []
    prev = expected_tpot_ms([], num_seqs, td, tv)  # T(0) baseline
    for step_alpha in per_step_alphas:
        history.append([float(a) for a in step_alpha])
        now = expected_tpot_ms(history, num_seqs, td, tv)
        if now > prev:
            return len(history)  # drafted this step, then stopped (reactive overshoot)
        prev = now
    return len(per_step_alphas)


class TestExpectedTpot(unittest.TestCase):
    def test_k_zero_is_verify_baseline(self):
        self.assertAlmostEqual(expected_tpot_ms([], 1, 19.0, 80.0), 80.0)
        self.assertAlmostEqual(expected_tpot_ms([], 4, 19.0, 80.0), 20.0)

    def test_single_seq_acc_len_cumprod(self):
        # alphas 0.9, 0.8 -> acc_len = 0.9 + 0.9*0.8 = 1.62
        steps = [[0.9], [0.8]]
        expected = (2 * 19.0 + 80.0) / (1.62 + 1)
        self.assertAlmostEqual(expected_tpot_ms(steps, 1, 19.0, 80.0), expected)

    def test_matches_brute_force_reference(self):
        cases = [
            ([[0.95], [0.9], [0.7], [0.4]], 1),
            ([[0.9, 0.8], [0.7, 0.6], [0.5, 0.5]], 2),
            ([[0.99, 0.98, 0.5], [0.6, 0.6, 0.4]], 3),
        ]
        for steps, n in cases:
            self.assertAlmostEqual(
                expected_tpot_ms(steps, n, 19.0, 80.0),
                brute_force_tpot(steps, n, 19.0, 80.0),
                places=9,
            )

    def test_zero_seqs_is_inf(self):
        self.assertEqual(expected_tpot_ms([], 0, 19.0, 80.0), float("inf"))


class TestReactiveStop(unittest.TestCase):
    def _argmin_k(self, per_step_alphas, num_seqs, td, tv):
        # Brute-force optimal k over all prefixes including k=0.
        best_k, best_t = 0, expected_tpot_ms([], num_seqs, td, tv)
        for k in range(1, len(per_step_alphas) + 1):
            t = expected_tpot_ms(per_step_alphas[:k], num_seqs, td, tv)
            if t < best_t:
                best_k, best_t = k, t
        return best_k

    def test_high_acceptance_drafts_full_budget(self):
        # Constant high alpha: TPOT keeps improving, never stops early.
        steps = [[0.95]] * 6
        self.assertEqual(reactive_stop_step(steps, 1, 19.0, 80.0), 6)

    def test_decaying_acceptance_stops_at_unimodal_min(self):
        # Decaying alpha -> T(k) is unimodal; reactive stops one step past argmin.
        steps = [[0.9], [0.8], [0.6], [0.3], [0.1]]
        argmin = self._argmin_k(steps, 1, 19.0, 80.0)
        stop = reactive_stop_step(steps, 1, 19.0, 80.0)
        self.assertEqual(stop, argmin + 1)

    def test_low_acceptance_stops_immediately(self):
        # Very low alpha makes even the first draft step worse than verify-only.
        steps = [[0.05], [0.05], [0.05]]
        # k=0 baseline 80.0; k=1 -> (19+80)/(0.05+1) = 94.3 > 80 -> stop after 1.
        self.assertEqual(reactive_stop_step(steps, 1, 19.0, 80.0), 1)

    def test_cheap_draft_extends_length(self):
        # Tiny td relative to tv rewards longer drafts even at modest alpha.
        steps = [[0.7]] * 5
        self.assertEqual(reactive_stop_step(steps, 1, 1.0, 80.0), 5)


if __name__ == "__main__":
    unittest.main()
