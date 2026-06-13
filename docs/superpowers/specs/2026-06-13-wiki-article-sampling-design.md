# Wiki Article Sampling Design

## Goal

Convert WikiText raw Parquet rows into article-level JSONL records and make the
random-cache collector sample uniformly distributed prefill windows from those
articles without crossing article boundaries.

## Filtered Dataset

`random_cache_srdp_scripts-1/prepare_wiki_articles.py` reconstructs articles
using WikiText level-one headings such as `= Bob Dylan =`. Each output JSONL
record contains:

- `article_id`
- `title`
- `text`
- `token_length`
- `tokenizer_path`
- `source_parquet`

Articles shorter than
`4 * min_prefill_n + decode_steps + reserve_tokens` are excluded because they
cannot produce a valid collector sample. Token lengths use the same tokenizer
as the model and exclude automatically added special tokens.

## Sample Allocation

For `N` eligible articles, the collector targets:

```text
target_samples = max(max_samples, N)
```

Every article receives one sample. Remaining samples are distributed in
proportion to each article's usable token length with the largest-remainder
method, so the final total is exactly `target_samples`.

## Window Sampling

For an article containing `L` tokens:

```text
article_max_n = min(
    max_prefill_n,
    floor((L - decode_steps - reserve_tokens) / 4),
    floor((max_seq_len - decode_steps - reserve_tokens) / 4),
)
n = UniformInteger(min_prefill_n, article_max_n)
prefill_len = 4 * n
start = UniformInteger(0, L - prefill_len)
```

Each allocated sample is drawn independently. The collector records the article
identity, original token length, quota, window start, sampled `n`, and
article-specific maximum `n` in output metadata.

## Compatibility

MTBench remains supported. Its prefill length is changed to uniform integer
sampling over the configured `n` interval. The obsolete `--length-bias`
argument is removed.

## Verification

Unit tests cover WikiText heading parsing, article reconstruction and filtering,
quota invariants, article-specific maximum lengths, and random-window bounds.
The real training Parquet is then converted and checked for record count,
minimum length, and stored token-length consistency.
