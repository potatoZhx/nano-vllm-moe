from types import SimpleNamespace

from nanovllm.engine.model_runner import ModelRunner


def _capture_warmup_lengths(monkeypatch, *, warmup_tokens, max_batched, max_len, max_seqs):
    runner = object.__new__(ModelRunner)
    runner.config = SimpleNamespace(
        warmup_model_tokens=warmup_tokens,
        max_num_batched_tokens=max_batched,
        max_model_len=max_len,
        max_num_seqs=max_seqs,
    )
    captured = []
    runner.run = lambda seqs, is_prefill: captured.extend(len(seq) for seq in seqs)
    runner._warmup_verify_layer_timings = lambda: None
    monkeypatch.setattr("torch.cuda.empty_cache", lambda: None)
    monkeypatch.setattr("torch.cuda.reset_peak_memory_stats", lambda: None)

    ModelRunner.warmup_model(runner)
    return captured


def test_warmup_model_honors_explicit_total_token_budget(monkeypatch):
    lengths = _capture_warmup_lengths(
        monkeypatch,
        warmup_tokens=1024,
        max_batched=16384,
        max_len=8192,
        max_seqs=1,
    )

    assert lengths == [1024]


def test_warmup_model_zero_preserves_legacy_shape(monkeypatch):
    lengths = _capture_warmup_lengths(
        monkeypatch,
        warmup_tokens=0,
        max_batched=16384,
        max_len=8192,
        max_seqs=2,
    )

    assert lengths == [8192, 8192]
