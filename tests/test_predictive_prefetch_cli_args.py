import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_module(relative_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestPredictivePrefetchCliArgs(unittest.TestCase):
    def test_heterogeneous_case_parser_exposes_predictive_knobs(self):
        mod = _load_module("examples/heterogeneous_benchmark_case.py")
        argv = [
            "heterogeneous_benchmark_case.py",
            "--model-path",
            "/fake/model",
            "--prefetch-runtime-kind",
            "predictive",
            "--prefetch-runtime-mode",
            "draft_segment_indexed",
            "--prefetch-verify-attention-ratio",
            "0.5",
            "--predictive-phase1-budget",
            "3",
            "--cpu-expert-pin-memory",
            "true",
            "--prefetch-transfer-stream-count",
            "4",
            "--spec-verify-miss-policy",
            "cache_fill",
        ]
        with patch.object(sys, "argv", argv):
            args = mod.parse_args()

        self.assertEqual(args.prefetch_runtime_kind, "predictive")
        self.assertEqual(args.prefetch_runtime_mode, "draft_segment_indexed")
        self.assertEqual(args.prefetch_verify_attention_ratio, 0.5)
        self.assertEqual(args.predictive_phase1_budget, 3)
        self.assertTrue(args.cpu_expert_pin_memory)
        self.assertEqual(args.prefetch_transfer_stream_count, 4)
        self.assertEqual(args.spec_verify_miss_policy, "cache_fill")

    def test_draft_forward_bench_parser_accepts_segment_indexed_predictive(self):
        mod = _load_module("examples/benchmarks/draft_standard_decode_forward_bench.py")
        argv = [
            "draft_standard_decode_forward_bench.py",
            "--model-path",
            "/fake/model",
            "--prefetch-runtime-kind",
            "predictive",
            "--prefetch-runtime-mode",
            "draft_segment_indexed",
            "--prefetch-verify-attention-ratio",
            "0.25",
            "--predictive-phase1-budget",
            "2",
            "--cpu-expert-pin-memory",
            "true",
            "--prefetch-transfer-stream-count",
            "4",
        ]
        with patch.object(sys, "argv", argv):
            args = mod.parse_args()

        self.assertEqual(args.prefetch_runtime_kind, "predictive")
        self.assertEqual(args.prefetch_runtime_mode, "draft_segment_indexed")
        self.assertEqual(args.prefetch_verify_attention_ratio, 0.25)
        self.assertEqual(args.predictive_phase1_budget, 2)
        self.assertTrue(args.cpu_expert_pin_memory)
        self.assertEqual(args.prefetch_transfer_stream_count, 4)

    def test_spec_verify_stats_parser_exposes_predictive_knobs(self):
        mod = _load_module("benchmarks/scripts/spec_verify_expert_count_stats.py")
        args = mod.build_parser().parse_args(
            [
                "--single-case",
                "--output",
                "/tmp/out.json",
                "--prefetch-runtime-kind",
                "predictive",
                "--prefetch-runtime-mode",
                "draft_segment_indexed",
                "--prefetch-verify-attention-ratio",
                "0.75",
                "--predictive-phase1-budget",
                "5",
            ]
        )

        self.assertEqual(args.prefetch_runtime_kind, "predictive")
        self.assertEqual(args.prefetch_runtime_mode, "draft_segment_indexed")
        self.assertEqual(args.prefetch_verify_attention_ratio, 0.75)
        self.assertEqual(args.predictive_phase1_budget, 5)


if __name__ == "__main__":
    unittest.main()
