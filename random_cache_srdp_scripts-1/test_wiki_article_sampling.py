import importlib.util
import math
import random
import sys
from pathlib import Path

import pytest
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


def test_normalize_cache_ratio_config_preserves_single_ratio_compatibility():
    mod = _load_module(
        "collect_random_cache_acceptance_single_ratio_test",
        "collect_random_cache_acceptance.py",
    )

    ratios, weights = mod.normalize_cache_ratio_config([0.5], None)

    assert ratios == [0.5]
    assert weights == [1.0]


def test_normalize_cache_ratio_config_defaults_multiple_ratios_to_equal_weights():
    mod = _load_module(
        "collect_random_cache_acceptance_equal_weights_test",
        "collect_random_cache_acceptance.py",
    )

    ratios, weights = mod.normalize_cache_ratio_config([0.25, 0.5, 0.75], None)

    assert ratios == [0.25, 0.5, 0.75]
    assert weights == [1.0, 1.0, 1.0]


def test_choose_cache_ratio_uses_explicit_weights_deterministically():
    mod = _load_module(
        "collect_random_cache_acceptance_weighted_choice_test",
        "collect_random_cache_acceptance.py",
    )
    rng = random.Random(42)

    selected = [
        mod.choose_cache_ratio(
            [0.25, 0.5, 0.75],
            [1.0, 2.0, 1.0],
            rng,
        )
        for _ in range(8)
    ]

    assert selected == [0.5, 0.25, 0.5, 0.25, 0.5, 0.5, 0.75, 0.25]


@pytest.mark.parametrize(
    ("ratios", "weights", "message"),
    [
        ([], None, "at least one"),
        ([0.0], None, "cache ratios"),
        ([1.1], None, "cache ratios"),
        ([math.inf], None, "cache ratios"),
        ([0.25, 0.5], [1.0], "same length"),
        ([0.25], [0.0], "weights"),
        ([0.25], [math.nan], "weights"),
    ],
)
def test_normalize_cache_ratio_config_rejects_invalid_values(
    ratios,
    weights,
    message,
):
    mod = _load_module(
        f"collect_random_cache_acceptance_invalid_{len(ratios)}_{message}",
        "collect_random_cache_acceptance.py",
    )

    with pytest.raises(ValueError, match=message):
        mod.normalize_cache_ratio_config(ratios, weights)


def test_cache_ratio_output_name_keeps_single_ratio_format_and_labels_mixed_runs():
    mod = _load_module(
        "collect_random_cache_acceptance_output_name_test",
        "collect_random_cache_acceptance.py",
    )

    assert mod.cache_ratio_output_name(
        dataset="wiki",
        cache_policy="lfu",
        cache_ratios=[0.5],
        cache_ratio_weights=[1.0],
        cache_topc_ratio=0.5,
    ) == "wiki_random_cache_lfu_ratio0.5_topc0.5"
    assert mod.cache_ratio_output_name(
        dataset="wiki",
        cache_policy="lfu",
        cache_ratios=[0.25, 0.5, 0.75],
        cache_ratio_weights=[1.0, 2.0, 1.0],
        cache_topc_ratio=0.5,
    ) == "wiki_random_cache_lfu_ratios0.25-0.5-0.75_weights1-2-1_topc0.5"
    assert mod.cache_ratio_output_name(
        dataset="wiki",
        cache_policy="lfu",
        cache_ratios=[0.123456789],
        cache_ratio_weights=[1.0],
        cache_topc_ratio=0.5,
    ) == "wiki_random_cache_lfu_ratio0.123456789_topc0.5"
