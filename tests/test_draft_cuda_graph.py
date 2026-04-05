import unittest
from types import SimpleNamespace

from nanovllm.engine.model_runner import ModelRunner


class TestDraftCudaGraphPolicy(unittest.TestCase):
    def test_can_use_draft_cudagraph_requires_top_c_zero(self):
        mr = object.__new__(ModelRunner)
        mr.config = SimpleNamespace(
            draft_cuda_graph_enabled=True,
            draft_top_c=1,
            draft_cuda_graph_max_bs=128,
        )
        mr.enforce_eager = False
        mr.draft_graphs = {1: object()}
        mr.draft_graph_bs = [1, 2, 4]

        self.assertFalse(ModelRunner._can_use_draft_cudagraph(mr, 1))

    def test_can_use_draft_cudagraph_requires_templates(self):
        mr = object.__new__(ModelRunner)
        mr.config = SimpleNamespace(
            draft_cuda_graph_enabled=True,
            draft_top_c=0,
            draft_cuda_graph_max_bs=128,
        )
        mr.enforce_eager = False
        mr.draft_graphs = {}
        mr.draft_graph_bs = []

        self.assertFalse(ModelRunner._can_use_draft_cudagraph(mr, 1))

    def test_can_use_draft_cudagraph_happy_path(self):
        mr = object.__new__(ModelRunner)
        mr.config = SimpleNamespace(
            draft_cuda_graph_enabled=True,
            draft_top_c=0,
            draft_cuda_graph_max_bs=128,
        )
        mr.enforce_eager = False
        mr.draft_graphs = {1: object(), 4: object()}
        mr.draft_graph_bs = [1, 4]

        self.assertTrue(ModelRunner._can_use_draft_cudagraph(mr, 1))
        self.assertTrue(ModelRunner._can_use_draft_cudagraph(mr, 3))
        self.assertFalse(ModelRunner._can_use_draft_cudagraph(mr, 8))


if __name__ == "__main__":
    unittest.main()
