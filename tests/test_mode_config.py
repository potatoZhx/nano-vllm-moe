import unittest

from nanovllm.config import Config


class TestInferenceModeConfig(unittest.TestCase):
    def test_mode_resolution_from_legacy_flags(self):
        cfg = Config(model="/zx_data1/models/Qwen--Qwen3-30B-A3B-Base", enable_heterogeneous=True, enable_speculative=False)
        self.assertEqual(cfg.inference_mode, "heter")

    def test_spec_requires_heterogeneous(self):
        with self.assertRaises(AssertionError):
            Config(model="/zx_data1/models/Qwen--Qwen3-30B-A3B-Base", inference_mode="spec", enable_heterogeneous=False)


if __name__ == "__main__":
    unittest.main()
