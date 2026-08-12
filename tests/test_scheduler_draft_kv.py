import unittest
from types import SimpleNamespace

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class TestSchedulerDraftKVWrappers(unittest.TestCase):
    def test_scheduler_wraps_block_manager_draft_apis(self):
        cfg = SimpleNamespace(
            max_num_seqs=1,
            max_num_batched_tokens=32,
            max_model_len=16,
            eos=-1,
            kvcache_block_size=256,
            num_kvcache_blocks=64,
        )
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

    def test_draft_append_reports_exhausted_kv_cache(self):
        cfg = SimpleNamespace(
            max_num_seqs=1,
            max_num_batched_tokens=512,
            max_model_len=512,
            eos=-1,
            kvcache_block_size=256,
            num_kvcache_blocks=1,
        )
        sch = Scheduler(cfg)
        seq = Sequence(
            list(range(257)),
            SamplingParams(temperature=1.0, max_tokens=8),
        )
        # Simulate a sequence whose first block is already resident and hashed.
        seq.num_tokens = 256
        seq.token_ids = seq.token_ids[:256]
        sch.block_manager.allocate(seq)
        seq.append_draft_token(256)

        with self.assertRaisesRegex(RuntimeError, "KV cache exhausted"):
            sch.append_draft_kv(seq)


if __name__ == "__main__":
    unittest.main()
