import importlib.util
import random
import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent


def _load_module(name: str, filename: str):
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


class WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return text.split()


def test_reconstructs_articles_from_spaced_wikitext_headings():
    mod = _load_module("prepare_wiki_articles_test", "prepare_wiki_articles.py")
    rows = [
        "",
        " = First Article = \n",
        "",
        " first paragraph has six content words \n",
        " = = Section Name = = \n",
        " second paragraph has five words \n",
        "",
        " = Second Article = \n",
        " tiny \n",
    ]

    articles = mod.reconstruct_articles(rows)

    assert mod.heading_level(" = First Article = \n") == 1
    assert mod.heading_level(" = = Section Name = = \n") == 2
    assert [item["title"] for item in articles] == [
        "First Article",
        "Second Article",
    ]
    assert "Section Name" in articles[0]["text"]
    assert "Second Article" not in articles[0]["text"]


def test_build_article_records_filters_articles_below_minimum_tokens():
    mod = _load_module("prepare_wiki_articles_records_test", "prepare_wiki_articles.py")
    articles = [
        {"title": "Long", "text": "one two three four five six"},
        {"title": "Short", "text": "one two"},
    ]

    records = mod.build_article_records(
        articles,
        tokenizer=WordTokenizer(),
        tokenizer_path="/model",
        source_parquet="/data/train.parquet",
        min_token_length=5,
    )

    assert records == [
        {
            "article_id": 0,
            "title": "Long",
            "text": "one two three four five six",
            "token_length": 6,
            "tokenizer_path": "/model",
            "source_parquet": "/data/train.parquet",
        }
    ]


def test_allocate_article_quotas_preserves_minimum_and_target_total():
    mod = _load_module(
        "collect_random_cache_acceptance_quota_test",
        "collect_random_cache_acceptance.py",
    )
    articles = [
        {"article_id": 0, "token_length": 100},
        {"article_id": 1, "token_length": 200},
        {"article_id": 2, "token_length": 400},
    ]

    quotas = mod.allocate_article_quotas(
        articles,
        max_samples=10,
        decode_steps=2,
        reserve_tokens=1,
    )

    assert sum(quotas) == 10
    assert min(quotas) >= 1
    assert quotas[2] > quotas[1] > quotas[0]
    assert mod.allocate_article_quotas(
        articles,
        max_samples=2,
        decode_steps=2,
        reserve_tokens=1,
    ) == [1, 1, 1]


def test_sample_article_window_respects_article_specific_limits():
    mod = _load_module(
        "collect_random_cache_acceptance_window_test",
        "collect_random_cache_acceptance.py",
    )
    input_ids = torch.arange(200).unsqueeze(0)

    sample, metadata = mod.sample_article_window(
        input_ids,
        min_prefill_n=8,
        max_prefill_n=100,
        max_seq_len=80,
        decode_steps=10,
        reserve_tokens=6,
        rng=random.Random(7),
    )

    assert metadata["article_max_prefill_n"] == 16
    assert 8 <= metadata["sampled_prefill_n"] <= 16
    assert sample.size(1) == 4 * metadata["sampled_prefill_n"]
    assert 0 <= metadata["window_start"] <= 200 - sample.size(1)
    assert torch.equal(
        sample,
        input_ids[
            :,
            metadata["window_start"] : metadata["window_start"] + sample.size(1),
        ],
    )
