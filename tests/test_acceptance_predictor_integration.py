"""Real-model integration test for the on-GPU acceptance predictor (no mocks).

Drives the actual benchmark (``scripts/bench_acceptance_predictor.py``), which
runs the real Qwen3-MoE model in spec mode with draft segment graphs, once with
the predictor enabled and once disabled, then asserts:

  * the predictor produced per-step alpha in [0, 1] written to the profile file;
  * enabling the predictor does NOT change the generated tokens (digest match) —
    the predictor only observes, so generation must be bit-identical;
  * the predictor's overhead is reported.

This needs a GPU, the real model, and the trained predictor checkpoint, so it is
gated behind ``NANOVLLM_RUN_ACCEPTANCE_PREDICTOR_TESTS=1``.

Enable with, e.g.:

    NANOVLLM_RUN_ACCEPTANCE_PREDICTOR_TESTS=1 \
    NANOVLLM_REAL_MODEL_PATH=/data1/models/Qwen3-30B-A3B \
    NANOVLLM_ACCEPTANCE_PREDICTOR_PATH=random_cache_srdp_scripts-1/res/run_20260614_133025 \
    python -m pytest tests/test_acceptance_predictor_integration.py -v -s
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


class TestAcceptancePredictorIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _as_bool(os.getenv("NANOVLLM_RUN_ACCEPTANCE_PREDICTOR_TESTS"), default=False):
            raise unittest.SkipTest(
                "Set NANOVLLM_RUN_ACCEPTANCE_PREDICTOR_TESTS=1 to enable (needs GPU + model)."
            )
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.bench = cls.repo_root / "scripts" / "bench_acceptance_predictor.py"
        cls.model_path = Path(
            os.getenv("NANOVLLM_REAL_MODEL_PATH", "/data1/models/Qwen3-30B-A3B")
        )
        cls.predictor_path = Path(
            os.getenv(
                "NANOVLLM_ACCEPTANCE_PREDICTOR_PATH",
                str(cls.repo_root / "random_cache_srdp_scripts-1" / "res" / "run_20260614_133025"),
            )
        )
        if not cls.model_path.is_dir():
            raise unittest.SkipTest(f"Model path not found: {cls.model_path}")
        if not cls.predictor_path.is_dir():
            raise unittest.SkipTest(f"Predictor checkpoint not found: {cls.predictor_path}")

        cls.profile_artifact = os.getenv("NANOVLLM_ACC_PRED_PROFILE_ARTIFACT", "")
        cls.output_len = os.getenv("NANOVLLM_ACC_PRED_OUTPUT_LEN", "16")
        cls.kt_num_threads = os.getenv("NANOVLLM_ACC_PRED_KT_THREADS", "8")
        cls.gpu_mem = os.getenv("NANOVLLM_ACC_PRED_GPU_MEM", "0.9")
        cls.timeout = int(os.getenv("NANOVLLM_ACC_PRED_TIMEOUT_SEC", "3600"))

        cls.tmpdir = tempfile.mkdtemp(prefix="acc_pred_bench_")
        cmd = [
            sys.executable,
            str(cls.bench),
            "--output-dir", cls.tmpdir,
            "--model-path", str(cls.model_path),
            "--acceptance-predictor-path", str(cls.predictor_path),
            "--profile-artifact", cls.profile_artifact,
            "--predictor-modes", "on,off",
            "--cache-ratios", "0.3125",
            "--output-lens", cls.output_len,
            "--max-draft-tokens-values", "4",
            "--segment-sizes", "12",
            "--repeats", "1",
            "--temperature", "0",  # greedy => deterministic => digest must match
            "--gpu-memory-utilization", cls.gpu_mem,
            "--kt-num-threads", cls.kt_num_threads,
            "--max-model-len", "1024",
            "--fail-fast", "true",
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(cls.repo_root) + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            cmd, cwd=cls.repo_root, env=env, capture_output=True, text=True, timeout=cls.timeout
        )
        cls.stdout = proc.stdout
        cls.stderr = proc.stderr
        if proc.returncode != 0:
            raise AssertionError(
                f"bench failed (rc={proc.returncode})\nSTDOUT:\n{proc.stdout[-4000:]}\n"
                f"STDERR:\n{proc.stderr[-4000:]}"
            )
        cls.summary = json.loads((Path(cls.tmpdir) / "summary.json").read_text(encoding="utf-8"))

    def _row(self, predictor_enabled: bool) -> dict:
        for row in self.summary["rows"]:
            if bool(row["predictor_enabled"]) == predictor_enabled:
                return row
        self.fail(f"no row with predictor_enabled={predictor_enabled}")

    def test_predicted_alpha_present_and_in_range(self):
        on = self._row(True)
        self.assertGreater(on["predicted_alpha_count"], 0, "predictor produced no alpha")
        self.assertGreaterEqual(on["predicted_alpha_min"], 0.0)
        self.assertLessEqual(on["predicted_alpha_max"], 1.0)
        self.assertGreaterEqual(on["predicted_alpha_avg"], 0.0)
        self.assertLessEqual(on["predicted_alpha_avg"], 1.0)

    def test_predicted_alpha_in_step_traces(self):
        # Per-sequence alpha must be persisted in the profile file's step traces.
        case_json = next(Path(self.tmpdir).glob("predON_*.json"))
        raw = json.loads(case_json.read_text(encoding="utf-8"))
        traces = raw["engine_profile"]["spec_step_traces"]
        found = any(
            "predicted_alpha" in seq and seq["predicted_alpha"]
            for step in traces
            for seq in step.get("sequences", [])
        )
        self.assertTrue(found, "no predicted_alpha in spec_step_traces")

    def test_predictor_does_not_change_outputs(self):
        on = self._row(True)
        off = self._row(False)
        self.assertEqual(
            on["outputs_digest"], off["outputs_digest"],
            "predictor altered generated tokens; it must only observe",
        )

    def test_overhead_reported(self):
        comps = self.summary["comparisons"]
        self.assertTrue(comps, "no on/off comparison produced")
        c = comps[0]
        # Just sanity: overhead should be a finite number; predictor-on should not be
        # dramatically faster than off (that would indicate a measurement bug).
        self.assertTrue(c["throughput_on"] > 0.0 and c["throughput_off"] > 0.0)
        print(
            f"[overhead] tok/s on={c['throughput_on']:.3f} off={c['throughput_off']:.3f} "
            f"overhead={c['throughput_overhead_pct']:+.2f}% draft_ms_delta={c['draft_ms_delta']:+.3f} "
            f"predict_alpha_avg={c['predicted_alpha_avg']:.4f} measured_accept={c['measured_acceptance_rate']:.4f}"
        )


if __name__ == "__main__":
    unittest.main()
