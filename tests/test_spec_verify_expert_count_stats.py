import importlib.util
import unittest
from pathlib import Path


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "benchmarks" / "scripts" / "spec_verify_expert_count_stats.py"
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


if __name__ == "__main__":
    unittest.main()
