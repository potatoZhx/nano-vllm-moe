import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nanovllm.config import Config


class TestConfigPredictivePrefetch(unittest.TestCase):
    def _build_config(self, **kwargs):
        fake_hf = SimpleNamespace(max_position_embeddings=4096)
        with patch("nanovllm.config.os.path.isdir", return_value=True), patch(
            "nanovllm.config.AutoConfig.from_pretrained",
            return_value=fake_hf,
        ):
            return Config(model="/fake/model", **kwargs)

    def test_legacy_is_default(self):
        cfg = self._build_config(enable_heterogeneous=True, inference_mode="spec")
        self.assertEqual(cfg.prefetch_runtime_kind, "legacy")
        # legacy must not touch the runtime mode
        self.assertEqual(cfg.prefetch_runtime_mode, "baseline_staging")

    def test_predictive_forces_segment_indexed_mode(self):
        cfg = self._build_config(
            enable_heterogeneous=True,
            inference_mode="spec",
            prefetch_runtime_kind="predictive",
        )
        self.assertEqual(cfg.prefetch_runtime_kind, "predictive")
        # predictive reuses the segment-indexed integration gates.
        self.assertEqual(cfg.prefetch_runtime_mode, "draft_segment_indexed")

    def test_predictive_param_defaults(self):
        cfg = self._build_config(enable_heterogeneous=True, inference_mode="spec")
        self.assertAlmostEqual(cfg.prefetch_verify_attention_ratio, 0.3)
        self.assertEqual(cfg.predictive_phase1_budget, 4)

    def test_invalid_kind_fails(self):
        with self.assertRaises(AssertionError):
            self._build_config(
                enable_heterogeneous=True,
                inference_mode="spec",
                prefetch_runtime_kind="fancy",
            )

    def test_invalid_attention_ratio_fails(self):
        with self.assertRaises(AssertionError):
            self._build_config(
                enable_heterogeneous=True,
                inference_mode="spec",
                prefetch_verify_attention_ratio=1.5,
            )

    def test_zero_attention_ratio_fails(self):
        with self.assertRaises(AssertionError):
            self._build_config(
                enable_heterogeneous=True,
                inference_mode="spec",
                prefetch_verify_attention_ratio=0.0,
            )

    def test_negative_phase1_budget_fails(self):
        with self.assertRaises(AssertionError):
            self._build_config(
                enable_heterogeneous=True,
                inference_mode="spec",
                predictive_phase1_budget=-1,
            )


if __name__ == "__main__":
    unittest.main()
