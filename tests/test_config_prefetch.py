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
        self.assertEqual(cfg.draft_prefetch_frontier_granularity, "segment")
        self.assertEqual(cfg.draft_prefetch_segment_size, 12)
        self.assertEqual(cfg.prefetch_metadata_host_buffer_pool_size, 3)
        self.assertEqual(cfg.draft_prefetch_segment_host_buffer_pool_size, 0)
        self.assertEqual(cfg.draft_prefetch_visible_budget_ms, 3.0)
        self.assertEqual(cfg.draft_prefetch_min_per_boundary, 0)
        self.assertEqual(cfg.draft_prefetch_max_per_boundary, 4)
        self.assertEqual(cfg.draft_cuda_graph_cpu_backend, "none")
        self.assertEqual(cfg.draft_reroute_policy, "entropy_cache_bias")
        self.assertEqual(cfg.draft_reroute_artifact, "")
        self.assertEqual(cfg.spec_verify_miss_policy, "cache_fill")
        self.assertTrue(cfg.prefetch_verify_layer_enabled)
        self.assertGreater(cfg.prefetch_verify_layer_transfer_bandwidth_gbps, 0.0)

    def test_draft_direct_active_runtime_mode_is_opt_in(self):
        cfg = self._build_config(
            enable_heterogeneous=True,
            inference_mode="spec",
            prefetch_runtime_mode="draft_direct_active",
        )
        self.assertEqual(cfg.prefetch_runtime_mode, "draft_direct_active")

    def test_draft_segment_indexed_runtime_mode_is_opt_in(self):
        cfg = self._build_config(
            enable_heterogeneous=True,
            inference_mode="spec",
            prefetch_runtime_mode="draft_segment_indexed",
        )
        self.assertEqual(cfg.prefetch_runtime_mode, "draft_segment_indexed")

    def test_invalid_prefetch_runtime_mode_fails(self):
        with self.assertRaises(AssertionError):
            self._build_config(
                enable_heterogeneous=True,
                inference_mode="spec",
                prefetch_runtime_mode="direct",
            )

    def test_invalid_draft_prefetch_frontier_fails(self):
        with self.assertRaises(AssertionError):
            self._build_config(
                enable_heterogeneous=True,
                inference_mode="spec",
                draft_prefetch_frontier_granularity="token",
            )

    def test_invalid_draft_prefetch_budget_fails(self):
        with self.assertRaises(AssertionError):
            self._build_config(
                enable_heterogeneous=True,
                inference_mode="spec",
                draft_prefetch_min_per_boundary=3,
                draft_prefetch_max_per_boundary=2,
            )

    def test_invalid_prefetch_metadata_host_buffer_pool_fails(self):
        with self.assertRaises(AssertionError):
            self._build_config(
                enable_heterogeneous=True,
                inference_mode="spec",
                prefetch_metadata_host_buffer_pool_size=0,
            )

    def test_invalid_draft_segment_host_buffer_pool_fails(self):
        with self.assertRaises(AssertionError):
            self._build_config(
                enable_heterogeneous=True,
                inference_mode="spec",
                draft_prefetch_segment_host_buffer_pool_size=-1,
            )

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

    def test_topc_zero_accepts_public_draft_reroute_policy(self):
        cfg = self._build_config(
            enable_heterogeneous=True,
            inference_mode="spec",
            draft_top_c=0,
            draft_reroute_policy="entropy_cache_bias",
        )
        self.assertEqual(cfg.draft_reroute_policy, "entropy_cache_bias")

    def test_reroute_policy_requires_topc_zero(self):
        with self.assertRaises(AssertionError):
            self._build_config(
                enable_heterogeneous=True,
                inference_mode="spec",
                draft_top_c=1,
                draft_reroute_policy="drop_miss",
            )

    def test_similarity_replace_requires_offline_artifact(self):
        with self.assertRaises(AssertionError):
            self._build_config(
                enable_heterogeneous=True,
                inference_mode="spec",
                draft_top_c=0,
                draft_reroute_policy="similarity_replace",
            )

    def test_invalid_cache_strategy_fails(self):
        with self.assertRaises(AssertionError):
            self._build_config(
                enable_heterogeneous=True,
                inference_mode="heter",
                cache_strategy="bad",
            )

    def test_verify_cache_fill_policy_can_be_set_explicitly(self):
        cfg = self._build_config(
            enable_heterogeneous=True,
            inference_mode="spec",
            spec_verify_miss_policy="cache_fill",
        )
        self.assertEqual(cfg.spec_verify_miss_policy, "cache_fill")

    def test_verify_cache_fill_no_cpu_policy_can_be_set_explicitly(self):
        cfg = self._build_config(
            enable_heterogeneous=True,
            inference_mode="spec",
            spec_verify_miss_policy="cache_fill_no_cpu",
        )
        self.assertEqual(cfg.spec_verify_miss_policy, "cache_fill_no_cpu")

    def test_invalid_verify_miss_policy_fails(self):
        with self.assertRaises(AssertionError):
            self._build_config(
                enable_heterogeneous=True,
                inference_mode="spec",
                spec_verify_miss_policy="gpu",
            )

    def test_invalid_prefetch_decay_fails(self):
        with self.assertRaises(AssertionError):
            self._build_config(
                enable_heterogeneous=True,
                inference_mode="heter",
                prefetch_history_decay=1.5,
            )

    def test_invalid_verify_layer_prefetch_bandwidth_fails(self):
        with self.assertRaises(AssertionError):
            self._build_config(
                enable_heterogeneous=True,
                inference_mode="heter",
                prefetch_verify_layer_transfer_bandwidth_gbps=0.0,
            )


if __name__ == "__main__":
    unittest.main()
