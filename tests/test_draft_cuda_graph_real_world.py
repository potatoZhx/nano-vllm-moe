import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    value = value.strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def _extract_last_json(stdout: str) -> dict:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("No JSON output found in subprocess stdout")
    return json.loads(lines[-1])


class TestDraftCudaGraphRealWorld(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _as_bool(os.getenv("NANOVLLM_RUN_REAL_GRAPH_TESTS"), default=False):
            raise unittest.SkipTest("Real-world CUDA Graph tests are disabled. Set NANOVLLM_RUN_REAL_GRAPH_TESTS=1 to enable.")

        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.case_script = cls.repo_root / "examples" / "heterogeneous_benchmark_case.py"
        cls.model_path = Path(os.getenv("NANOVLLM_REAL_MODEL_PATH", "/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B"))
        if not cls.model_path.is_dir():
            raise unittest.SkipTest(f"Model path not found: {cls.model_path}")

        cls.num_seqs = int(os.getenv("NANOVLLM_REAL_GRAPH_NUM_SEQS", "1"))
        cls.input_len = int(os.getenv("NANOVLLM_REAL_GRAPH_INPUT_LEN", "12"))
        cls.output_len = int(os.getenv("NANOVLLM_REAL_GRAPH_OUTPUT_LEN", "6"))
        cls.max_num_batched_tokens = int(os.getenv("NANOVLLM_REAL_GRAPH_MAX_NUM_BATCHED_TOKENS", "1024"))
        cls.max_num_seqs = int(os.getenv("NANOVLLM_REAL_GRAPH_MAX_NUM_SEQS", "64"))
        cls.max_model_len = int(os.getenv("NANOVLLM_REAL_GRAPH_MAX_MODEL_LEN", "1024"))
        cls.gpu_memory_utilization = float(os.getenv("NANOVLLM_REAL_GRAPH_GPU_MEMORY_UTIL", "0.85"))
        cls.max_draft_tokens = int(os.getenv("NANOVLLM_REAL_GRAPH_MAX_DRAFT_TOKENS", "4"))
        cls.base_port = int(os.getenv("NANOVLLM_REAL_GRAPH_BASE_PORT", "29920"))
        cls.case_timeout = int(os.getenv("NANOVLLM_REAL_GRAPH_CASE_TIMEOUT_SEC", "2400"))

        cls.std_graph = cls._run_case(mode="standard", enforce_eager=False, dist_port=cls.base_port)
        cls.spec_graph = cls._run_case(mode="spec", enforce_eager=False, dist_port=cls.base_port + 1)
        cls.std_eager = cls._run_case(mode="standard", enforce_eager=True, dist_port=cls.base_port + 2)
        cls.spec_eager = cls._run_case(mode="spec", enforce_eager=True, dist_port=cls.base_port + 3)

    @classmethod
    def _run_case(cls, mode: str, enforce_eager: bool, dist_port: int) -> dict:
        cmd = [
            sys.executable,
            str(cls.case_script),
            "--model-path",
            str(cls.model_path),
            "--mode",
            mode,
            "--slots-per-layer",
            "0",
            "--num-seqs",
            str(cls.num_seqs),
            "--input-len",
            str(cls.input_len),
            "--output-len",
            str(cls.output_len),
            "--max-num-batched-tokens",
            str(cls.max_num_batched_tokens),
            "--max-num-seqs",
            str(cls.max_num_seqs),
            "--max-model-len",
            str(cls.max_model_len),
            "--gpu-memory-utilization",
            str(cls.gpu_memory_utilization),
            "--max-draft-tokens",
            str(cls.max_draft_tokens),
            "--draft-top-c",
            "0",
            "--seed",
            "0",
            "--temperature",
            "0.0",
            "--enforce-eager",
            "true" if enforce_eager else "false",
            "--engine-profile",
            "true",
            "--engine-profile-cuda-sync",
            "true",
            "--return-token-ids",
            "true",
            "--return-text",
            "false",
            "--return-prompts",
            "false",
            "--dist-port",
            str(dist_port),
        ]

        proc = subprocess.run(
            cmd,
            cwd=cls.repo_root,
            text=True,
            capture_output=True,
            check=False,
            timeout=cls.case_timeout,
            env=os.environ.copy(),
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Real-world case failed: mode={mode}, enforce_eager={enforce_eager}, dist_port={dist_port}.\n"
                f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        return _extract_last_json(proc.stdout)

    @staticmethod
    def _standard_forward_metrics(result: dict) -> tuple[float, float]:
        prof = result.get("engine_profile") or {}
        decode_ms = float(prof.get("decode_runner_ms", 0.0))
        decode_steps = int(prof.get("decode_step_count", 0))
        forward_ms = (decode_ms / decode_steps) if decode_steps > 0 else 0.0
        tokens_per_forward = float(result.get("num_seqs", 0))
        tok_s = (tokens_per_forward * 1000.0 / forward_ms) if forward_ms > 0 else 0.0
        return forward_ms, tok_s

    @staticmethod
    def _draft_forward_metrics(result: dict) -> tuple[float, float]:
        prof = result.get("engine_profile") or {}
        draft_ms = float(prof.get("spec_run_draft_infer_ms_total", 0.0))
        draft_calls = int(prof.get("spec_run_draft_calls", 0))
        draft_tokens_total = float(prof.get("spec_draft_tokens_total", 0.0))
        forward_ms = (draft_ms / draft_calls) if draft_calls > 0 else 0.0
        tokens_per_forward = (draft_tokens_total / draft_calls) if draft_calls > 0 and draft_tokens_total > 0 else float(result.get("num_seqs", 0))
        tok_s = (tokens_per_forward * 1000.0 / forward_ms) if forward_ms > 0 else 0.0
        return forward_ms, tok_s

    def test_cuda_graph_replay_enabled_in_real_workload(self):
        std_prof = self.std_graph.get("engine_profile") or {}
        spec_prof = self.spec_graph.get("engine_profile") or {}
        self.assertGreater(int(std_prof.get("model_standard_graph_replay_count", 0)), 0)
        self.assertGreater(int(spec_prof.get("model_draft_graph_replay_count", 0)), 0)

    def test_graph_and_eager_outputs_are_consistent(self):
        self.assertEqual(self.std_graph.get("generated_token_ids"), self.std_eager.get("generated_token_ids"))
        self.assertEqual(self.spec_graph.get("generated_token_ids"), self.spec_eager.get("generated_token_ids"))
        self.assertEqual(self.std_graph.get("generated_token_ids"), self.spec_graph.get("generated_token_ids"))

    def test_draft_forward_speed_is_close_to_standard_decode(self):
        std_forward_ms, std_tok_s = self._standard_forward_metrics(self.std_graph)
        draft_forward_ms, draft_tok_s = self._draft_forward_metrics(self.spec_graph)

        self.assertGreater(std_forward_ms, 0.0)
        self.assertGreater(draft_forward_ms, 0.0)
        self.assertGreater(std_tok_s, 0.0)
        self.assertGreater(draft_tok_s, 0.0)

        ms_upper = float(os.getenv("NANOVLLM_REAL_GRAPH_MS_RATIO_UPPER", "1.35"))
        ms_lower = float(os.getenv("NANOVLLM_REAL_GRAPH_MS_RATIO_LOWER", "0.65"))
        tok_upper = float(os.getenv("NANOVLLM_REAL_GRAPH_TOK_RATIO_UPPER", "1.35"))
        tok_lower = float(os.getenv("NANOVLLM_REAL_GRAPH_TOK_RATIO_LOWER", "0.65"))

        ms_ratio = draft_forward_ms / std_forward_ms
        tok_ratio = draft_tok_s / std_tok_s

        self.assertLessEqual(ms_ratio, ms_upper)
        self.assertGreaterEqual(ms_ratio, ms_lower)
        self.assertLessEqual(tok_ratio, tok_upper)
        self.assertGreaterEqual(tok_ratio, tok_lower)


if __name__ == "__main__":
    unittest.main()
