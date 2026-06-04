import importlib.util
import unittest
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "benchmarks" / "scripts" / "spec_verify_expert_count_stats.py"
    spec = importlib.util.spec_from_file_location("spec_verify_expert_count_stats", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _Tokenizer:
    def encode(self, prompt: str) -> list[int]:
        return list(range(len(prompt.split())))


class TestSpecVerifyExpertCountStats(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_module()

    def test_prompt_input_length_is_measured_in_tokens(self):
        prompts = ["one two three four", "five six seven eight"]
        tokenized = self.mod._tokenize_prompts_to_length(_Tokenizer(), prompts, 3)
        self.assertEqual(tokenized, [[0, 1, 2], [0, 1, 2]])

    def test_short_generated_prompt_is_rejected(self):
        with self.assertRaises(ValueError):
            self.mod._tokenize_prompts_to_length(_Tokenizer(), ["one two"], 3)

    def test_parser_exposes_cache_and_graph_runtime_knobs(self):
        args = self.mod.build_parser().parse_args(
            [
                "--single-case",
                "--output",
                "/tmp/out.json",
                "--prefetch-runtime-mode",
                "draft_segment_indexed",
                "--draft-cuda-graph-enabled",
                "true",
                "--draft-cuda-graph-cpu-backend",
                "none",
                "--rank-guard-threshold",
                "0.12",
                "--rank-guard-ema-alpha",
                "0.9",
                "--spec-verify-miss-policy",
                "cache_fill",
            ]
        )

        self.assertEqual(args.prefetch_runtime_mode, "draft_segment_indexed")
        self.assertTrue(args.draft_cuda_graph_enabled)
        self.assertEqual(args.draft_cuda_graph_cpu_backend, "none")
        self.assertEqual(args.rank_guard_threshold, 0.12)
        self.assertEqual(args.rank_guard_ema_alpha, 0.9)
        self.assertEqual(args.spec_verify_miss_policy, "cache_fill")

    def test_summary_reports_cache_hit_rate_from_cpu_route_ratio(self):
        summary = self.mod._summarize_case(
            {
                "case": {},
                "generated_output_tokens": 16,
                "throughput_output_tok_s": 2.0,
                "engine_profile": {
                    "model_cpu_route_ratio": 0.25,
                    "spec_step_traces": [],
                },
            },
            [],
        )

        self.assertEqual(summary["cache"]["route_hit_rate"], 0.75)

    def test_summary_reports_cache_fill_no_cpu_counters(self):
        summary = self.mod._summarize_case(
            {
                "case": {"spec_verify_miss_policy": "cache_fill_no_cpu"},
                "generated_output_tokens": 16,
                "throughput_output_tok_s": 2.0,
                "engine_profile": {
                    "model_verify_cache_fill_no_cpu_remaining_miss_count": 3,
                    "model_verify_cache_fill_no_cpu_remaining_miss_expert_count": 2,
                    "model_verify_cache_fill_no_cpu_remaining_miss_route_count": 3,
                    "model_verify_cache_fill_no_cpu_fallback_count": 1,
                    "spec_step_traces": [],
                },
            },
            [],
        )

        self.assertEqual(summary["verify_cache_fill"]["policy"], "cache_fill_no_cpu")
        self.assertEqual(summary["verify_cache_fill"]["no_cpu_remaining_miss_count"], 3)
        self.assertEqual(summary["verify_cache_fill"]["no_cpu_remaining_miss_expert_count"], 2)
        self.assertEqual(summary["verify_cache_fill"]["no_cpu_remaining_miss_route_count"], 3)
        self.assertEqual(summary["verify_cache_fill"]["no_cpu_fallback_count"], 1)

    def test_parser_accepts_cache_fill_no_cpu_policy(self):
        args = self.mod.build_parser().parse_args(
            [
                "--single-case",
                "--output",
                "/tmp/out.json",
                "--spec-verify-miss-policy",
                "cache_fill_no_cpu",
            ]
        )

        self.assertEqual(args.spec_verify_miss_policy, "cache_fill_no_cpu")

    def test_m3_perfect_fraction_groups_draft_layers_and_step0(self):
        events = [
            {"layer_idx": 0, "cpu_expert_count": 0, "cpu_route_ratio": 0.0},
            {"layer_idx": 1, "cpu_expert_count": 0, "cpu_route_ratio": 0.0},
            {"layer_idx": 0, "cpu_expert_count": 1, "cpu_route_ratio": 0.5},
            {"layer_idx": 1, "cpu_expert_count": 0, "cpu_route_ratio": 0.0},
            {"layer_idx": 0, "cpu_expert_count": 0, "cpu_route_ratio": 0.0},
            {"layer_idx": 1, "cpu_expert_count": 0, "cpu_route_ratio": 0.0},
        ]

        metrics = self.mod._m3_perfect_fraction(events, draft_steps_per_step=[2, 1])

        self.assertEqual(metrics["group_count"], 3)
        self.assertEqual(metrics["perfect_count"], 2)
        self.assertAlmostEqual(metrics["perfect_fraction"], 2 / 3)
        self.assertEqual(metrics["step0_group_count"], 2)
        self.assertEqual(metrics["step0_perfect_count"], 2)
        self.assertAlmostEqual(metrics["step0_perfect_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
