import importlib.util
import unittest
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "lfu_rankguard_cache_validation.py"
    spec = importlib.util.spec_from_file_location("lfu_rankguard_cache_validation", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestLFURankGuardCacheValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_module()

    def test_text_quality_rejects_replacement_chars(self):
        quality = self.mod.assess_text_quality("normal text \ufffd broken")

        self.assertFalse(quality["ok"])
        self.assertIn("replacement_char", quality["reasons"])

    def test_case_matrix_covers_requested_cache_strategies_lengths_and_ratios(self):
        cases = self.mod.build_cases(
            cache_strategies=["lfu", "lfu_rankguard"],
            output_lens=[128, 512],
            cache_ratios=[0.25, 0.5, 0.75],
        )

        self.assertEqual(len(cases), 12)
        self.assertIn(("lfu_rankguard", 512, 0.75), cases)


if __name__ == "__main__":
    unittest.main()
