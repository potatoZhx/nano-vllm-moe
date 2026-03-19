import unittest

from nanovllm.config import Config
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class TestSchedulerDraftKVWrappers(unittest.TestCase):
    def test_scheduler_wraps_block_manager_draft_apis(self):
        cfg = Config(
            model="/zx_data1/models/Qwen--Qwen3-30B-A3B-Base",
            max_num_seqs=1,
            max_num_batched_tokens=32,
            max_model_len=16,
            inference_mode="heter",
            enable_heterogeneous=True,
        )
        cfg.num_kvcache_blocks = 64
        sch = Scheduler(cfg)

        seq = Sequence([1, 2, 3, 4], SamplingParams(temperature=1.0, max_tokens=8))
        sch.block_manager.allocate(seq)

        sch.start_draft_kv(seq)
        seq.append_draft_token(5)
        sch.append_draft_kv(seq)
        self.assertEqual(seq.num_tokens, 5)

        sch.rollback_draft_kv(seq)
        seq.rollback_tokens_to_draft_start()
        self.assertEqual(seq.num_tokens, 4)


if __name__ == "__main__":
    unittest.main()
