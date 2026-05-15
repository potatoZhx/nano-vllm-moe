import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nanovllm.config import Config


class TestConfigPrefetch(unittest.TestCase):
    def _build_config(self, **kwargs):
        fake_hf = SimpleNamespace(max_position_embeddings=4096)
        with patch("nanovllm.config.os.path.isdir", return_value=True), patch(
            "nanovllm.config.AutoConfig.from_pretrained",
            return_value=fake_hf,
        ):
            return Config(model="/fake/model", **kwargs)

    def test_prefetch_defaults_are_valid(self):
        cfg = self._build_config(enable_heterogeneous=True, inference_mode="heter")
        self.assertEqual(cfg.cache_strategy, "lru")
        self.assertEqual(cfg.prefetch_strategy, "history_window")
        self.assertEqual(cfg.prefetch_runtime_mode, "baseline_staging")
        self.assertEqual(cfg.draft_cuda_graph_cpu_backend, "none")

    def test_draft_cuda_graph_cpu_backend_accepts_fused(self):
        cfg = self._build_config(
            enable_heterogeneous=True,
            inference_mode="spec",
            draft_cuda_graph_cpu_backend="fused",
            cpu_expert_backend="fused",
        )
        self.assertEqual(cfg.draft_cuda_graph_cpu_backend, "fused")

    def test_draft_cuda_graph_cpu_backend_accepts_fused_sync(self):
        cfg = self._build_config(
            enable_heterogeneous=True,
            inference_mode="spec",
            draft_cuda_graph_cpu_backend="fused_sync",
            cpu_expert_backend="fused",
        )
        self.assertEqual(cfg.draft_cuda_graph_cpu_backend, "fused_sync")

    def test_invalid_draft_cuda_graph_cpu_backend_fails(self):
        with self.assertRaises(AssertionError):
            self._build_config(
                enable_heterogeneous=True,
                inference_mode="spec",
                draft_cuda_graph_cpu_backend="torch",
            )

    def test_invalid_cache_strategy_fails(self):
        with self.assertRaises(AssertionError):
            self._build_config(
                enable_heterogeneous=True,
                inference_mode="heter",
                cache_strategy="bad",
            )

    def test_invalid_prefetch_decay_fails(self):
        with self.assertRaises(AssertionError):
            self._build_config(
                enable_heterogeneous=True,
                inference_mode="heter",
                prefetch_history_decay=1.5,
            )


if __name__ == "__main__":
    unittest.main()
