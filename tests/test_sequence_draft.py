import unittest

from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class TestSequenceDraft(unittest.TestCase):
    def test_start_append_and_rollback_draft_tokens(self):
        seq = Sequence([10, 11, 12], SamplingParams(temperature=1.0, max_tokens=8))

        seq.start_draft()
        self.assertTrue(seq.is_drafting)
        self.assertEqual(seq._draft_start_num_tokens, 3)
        self.assertEqual(seq.draft_token_ids, [])

        seq.append_draft_token(20)
        seq.append_draft_token(21)
        self.assertEqual(seq.token_ids, [10, 11, 12, 20, 21])
        self.assertEqual(seq.draft_token_ids, [20, 21])
        self.assertEqual(seq.last_token, 21)

        seq.rollback_tokens_to_draft_start()
        self.assertEqual(seq.token_ids, [10, 11, 12])
        self.assertEqual(seq.num_tokens, 3)
        self.assertEqual(seq.last_token, 12)


if __name__ == "__main__":
    unittest.main()
