import importlib.util
import unittest
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "bench_dual_queue_prefetch.py"
    spec = importlib.util.spec_from_file_location("bench_dual_queue_prefetch", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_single_case_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "benchmarks" / "scripts" / "spec_verify_expert_count_stats.py"
    spec = importlib.util.spec_from_file_location(
        "dual_queue_bench_single_case",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestDualQueueBench(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_module()
        cls.single_case_mod = _load_single_case_module()

    def test_case_matrix_covers_runtime_and_segment_values(self):
        args = self.mod.build_parser().parse_args(
            [
                "--output-dir",
                "/tmp/dual_queue_bench",
                "--runtime-kinds",
                "dual_queue,predictive",
                "--output-lens",
                "16",
                "--cache-ratios",
                "0.25",
                "--max-draft-tokens-values",
                "4",
                "--segment-sizes",
                "6,12",
                "--repeats",
                "2",
            ]
        )

        cases = self.mod.build_cases(args)

        self.assertEqual(len(cases), 8)
        self.assertEqual({case["runtime_kind"] for case in cases}, {
            "dual_queue",
            "predictive",
        })
        self.assertEqual({case["segment_size"] for case in cases}, {6, 12})
        self.assertEqual({case["repeat"] for case in cases}, {0, 1})

    def test_generated_command_is_accepted_by_single_case_parser(self):
        args = self.mod.build_parser().parse_args(
            [
                "--output-dir",
                "/tmp/dual_queue_bench",
                "--runtime-kinds",
                "dual_queue",
                "--output-lens",
                "16",
                "--cache-ratios",
                "0.25",
                "--max-draft-tokens-values",
                "4",
                "--segment-sizes",
                "8",
            ]
        )
        case = self.mod.build_cases(args)[0]
        repo_root = Path(__file__).resolve().parents[1]
        command = self.mod._command(
            args,
            repo_root,
            Path("/tmp/prompt.txt"),
            case,
            Path("/tmp/result.json"),
            0,
        )

        parsed = self.single_case_mod.build_parser().parse_args(command[2:])

        self.assertEqual(parsed.prefetch_runtime_kind, "dual_queue")
        self.assertEqual(parsed.dual_queue_segment_size, 8)
        self.assertEqual(parsed.draft_prefetch_segment_size, 8)
        self.assertEqual(parsed.verify_prefetch_segment_size, 8)
        self.assertEqual(parsed.cpu_expert_backend, "kt_direct")
        self.assertTrue(parsed.verify_cuda_graph)

    def test_row_extracts_dual_queue_sources(self):
        case = {
            "runtime_kind": "dual_queue",
            "output_len": 16,
            "cache_ratio": 0.25,
            "max_draft_tokens": 4,
            "segment_size": 12,
            "repeat": 0,
        }
        raw = {
            "generated_output_tokens": 16,
            "outputs_digest": "abc",
            "summary": {
                "throughput_output_tok_s": 3.0,
                "dual_queue": {
                    "draft_budget": 2,
                    "verify_budget": 3,
                    "submit_count_by_source": {
                        self.mod.DUAL_DRAFT_SOURCE: 4,
                        self.mod.DUAL_GROUND_TRUTH_SOURCE: 6,
                    },
                    "completed_count_by_source": {
                        self.mod.DUAL_DRAFT_SOURCE: 3,
                        self.mod.DUAL_GROUND_TRUTH_SOURCE: 5,
                    },
                    "published_count_by_source": {
                        self.mod.DUAL_DRAFT_SOURCE: 2,
                        self.mod.DUAL_GROUND_TRUTH_SOURCE: 4,
                    },
                    "late_count_by_source": {
                        self.mod.DUAL_DRAFT_SOURCE: 1,
                        self.mod.DUAL_GROUND_TRUTH_SOURCE: 2,
                    },
                    "target_miss_count": 2,
                    "round_end_discard_count": 1,
                },
            },
        }

        row = self.mod._row_from_raw(case, raw, 1.5)

        self.assertEqual(row["draft_budget"], 2)
        self.assertEqual(row["verify_budget"], 3)
        self.assertEqual(row["dual_submit_count"], 10)
        self.assertEqual(row["dual_completed_count"], 8)
        self.assertEqual(row["dual_published_count"], 6)
        self.assertEqual(row["dual_late_count"], 3)
        self.assertEqual(row["dual_publish_ratio"], 0.6)
        self.assertEqual(row["target_miss_count"], 2)
        self.assertEqual(row["round_end_discard_count"], 1)

    def test_comparison_matches_same_case_and_checks_digest(self):
        base = {
            "output_len": 16,
            "cache_ratio": 0.25,
            "max_draft_tokens": 4,
            "segment_size": 12,
            "repeat": 0,
            "draft_forward_ms_avg": 10.0,
            "verify_forward_ms_avg": 20.0,
            "route_hit_rate": 0.8,
            "acceptance_rate": 0.7,
            "avg_miss_routes_per_layer": 1.5,
            "draft_prefetch_per_forward": 8.0,
            "verify_prefetch_per_forward": 12.0,
            "dual_publish_ratio": 0.5,
            "target_miss_count": 2,
            "round_end_discard_count": 1,
        }
        rows = [
            {
                **base,
                "runtime_kind": "dual_queue",
                "throughput_output_tok_s": 4.0,
                "outputs_digest": "same",
            },
            {
                **base,
                "runtime_kind": "predictive",
                "throughput_output_tok_s": 2.0,
                "outputs_digest": "same",
            },
        ]

        comparisons = self.mod._comparison_rows(rows)

        self.assertEqual(len(comparisons), 1)
        self.assertTrue(comparisons[0]["digest_match"])
        self.assertEqual(comparisons[0]["throughput_delta"], 2.0)
        self.assertEqual(comparisons[0]["throughput_ratio"], 2.0)


if __name__ == "__main__":
    unittest.main()
