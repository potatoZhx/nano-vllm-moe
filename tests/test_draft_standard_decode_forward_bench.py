import importlib.util
import unittest
from pathlib import Path


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "benchmarks" / "draft_standard_decode_forward_bench.py"
    spec = importlib.util.spec_from_file_location("draft_standard_decode_forward_bench", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestDraftStandardDecodeForwardBench(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_module()

    def test_extract_standard_decode_metrics(self):
        case_result = {
            "num_seqs": 8,
            "engine_profile": {
                "decode_runner_ms": 96.0,
                "decode_step_count": 12,
            },
        }
        metrics = self.mod.extract_standard_decode_metrics(case_result)
        self.assertEqual(metrics["forward_ms"], 8.0)
        self.assertEqual(metrics["tokens_per_forward"], 8.0)
        self.assertAlmostEqual(metrics["forward_tok_s"], 1000.0)
        self.assertIn("profile_breakdown", metrics)

    def test_extract_draft_forward_metrics_with_draft_tokens(self):
        case_result = {
            "num_seqs": 8,
            "engine_profile": {
                "spec_run_draft_infer_ms_total": 120.0,
                "spec_run_draft_calls": 10,
                "spec_draft_tokens_total": 80.0,
                "model_route_ms": 30.0,
            },
        }
        metrics = self.mod.extract_draft_forward_metrics(case_result)
        self.assertEqual(metrics["forward_ms"], 12.0)
        self.assertEqual(metrics["tokens_per_forward"], 8.0)
        self.assertAlmostEqual(metrics["forward_tok_s"], 666.6666666, places=4)
        self.assertAlmostEqual(metrics["profile_breakdown"]["route_ms_per_call"], 3.0)

    def test_extract_draft_forward_metrics_fallback_to_num_seqs(self):
        case_result = {
            "num_seqs": 4,
            "engine_profile": {
                "spec_run_draft_infer_ms_total": 40.0,
                "spec_run_draft_calls": 5,
                "spec_draft_tokens_total": 0.0,
            },
        }
        metrics = self.mod.extract_draft_forward_metrics(case_result)
        self.assertEqual(metrics["tokens_per_forward"], 4.0)
        self.assertEqual(metrics["forward_ms"], 8.0)
        self.assertAlmostEqual(metrics["forward_tok_s"], 500.0)

    def test_extract_json_stdout_accepts_pretty_json(self):
        payload = {"mode": "spec", "value": 3}
        stdout = "{\n  \"mode\": \"spec\",\n  \"value\": 3\n}\n"
        self.assertEqual(self.mod._extract_json_stdout(stdout), payload)

    def test_summarize_repeats(self):
        rows = [
            {
                "standard_decode": {"forward_ms": 10.0, "forward_tok_s": 800.0},
                "draft_forward": {"forward_ms": 12.0, "forward_tok_s": 666.0},
            },
            {
                "standard_decode": {"forward_ms": 9.0, "forward_tok_s": 900.0},
                "draft_forward": {"forward_ms": 10.0, "forward_tok_s": 720.0},
            },
            {
                "standard_decode": {"forward_ms": 11.0, "forward_tok_s": 760.0},
                "draft_forward": {"forward_ms": 11.0, "forward_tok_s": 700.0},
            },
        ]

        summary = self.mod.summarize_repeats(rows)
        self.assertEqual(summary["standard_decode_forward_ms_median"], 10.0)
        self.assertEqual(summary["draft_forward_ms_median"], 11.0)
        self.assertAlmostEqual(summary["draft_over_standard_ms_ratio"], 1.1)
        self.assertEqual(summary["standard_decode_forward_tok_s_median"], 800.0)
        self.assertEqual(summary["draft_forward_tok_s_median"], 700.0)
        self.assertAlmostEqual(summary["draft_over_standard_tok_s_ratio"], 0.875)

    def test_validate_cuda_graph_usage_success(self):
        standard_result = {"engine_profile": {"model_standard_graph_replay_count": 5}}
        spec_result = {"engine_profile": {"model_draft_graph_replay_count": 7}}
        self.mod.validate_cuda_graph_usage(standard_result, spec_result, enforce_eager=False)

    def test_validate_cuda_graph_usage_rejects_eager(self):
        standard_result = {"engine_profile": {"model_standard_graph_replay_count": 5}}
        spec_result = {"engine_profile": {"model_draft_graph_replay_count": 7}}
        with self.assertRaises(RuntimeError):
            self.mod.validate_cuda_graph_usage(standard_result, spec_result, enforce_eager=True)

    def test_validate_cuda_graph_usage_requires_standard_replay(self):
        standard_result = {"engine_profile": {"model_standard_graph_replay_count": 0}}
        spec_result = {"engine_profile": {"model_draft_graph_replay_count": 7}}
        with self.assertRaises(RuntimeError):
            self.mod.validate_cuda_graph_usage(standard_result, spec_result, enforce_eager=False)

    def test_validate_cuda_graph_usage_requires_draft_replay(self):
        standard_result = {"engine_profile": {"model_standard_graph_replay_count": 5}}
        spec_result = {"engine_profile": {"model_draft_graph_replay_count": 0}}
        with self.assertRaises(RuntimeError):
            self.mod.validate_cuda_graph_usage(standard_result, spec_result, enforce_eager=False)

    def test_validate_deterministic_alignment_temperature_zero(self):
        standard = {"outputs_digest": "abc"}
        spec = {"outputs_digest": "abc"}
        self.assertTrue(self.mod.validate_deterministic_alignment(standard, spec, temperature=0.0))

    def test_validate_deterministic_alignment_raises_on_mismatch(self):
        standard = {"outputs_digest": "abc"}
        spec = {"outputs_digest": "def"}
        with self.assertRaises(RuntimeError):
            self.mod.validate_deterministic_alignment(standard, spec, temperature=0.0)

    def test_validate_deterministic_alignment_skips_sampling_mode(self):
        standard = {"outputs_digest": "abc"}
        spec = {"outputs_digest": "def"}
        self.assertTrue(self.mod.validate_deterministic_alignment(standard, spec, temperature=0.7))


if __name__ == "__main__":
    unittest.main()
