import unittest

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class TestBlockManagerDraftLifecycle(unittest.TestCase):
    def test_rollback_and_accept_draft(self):
        bm = BlockManager(num_blocks=64, block_size=4)
        seq = Sequence([1, 2, 3, 4], SamplingParams(temperature=1.0, max_tokens=16))
        bm.allocate(seq)

        bm.start_draft(seq)

        seq.append_draft_token(5)
        bm.append_draft_token(seq)
        seq.append_draft_token(6)
        bm.append_draft_token(seq)
        self.assertEqual(seq.num_tokens, 6)

        bm.rollback_draft(seq)
        seq.rollback_tokens_to_draft_start()
        self.assertEqual(seq.num_tokens, 4)
        self.assertEqual(seq.token_ids, [1, 2, 3, 4])

        bm.start_draft(seq)
        seq.append_draft_token(7)
        bm.append_draft_token(seq)
        seq.append_draft_token(8)
        bm.append_draft_token(seq)

        bm.accept_draft(seq, num_accepted=1)
        seq.rollback_tokens_to_draft_start()
        seq.append_token(7)

        self.assertEqual(seq.token_ids, [1, 2, 3, 4, 7])
        self.assertEqual(seq.num_tokens, 5)


if __name__ == "__main__":
    unittest.main()
