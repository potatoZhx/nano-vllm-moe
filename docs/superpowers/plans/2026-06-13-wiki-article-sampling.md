# Wiki Article Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an article-level WikiText dataset and collect at least the requested number of random-cache samples without crossing article boundaries.

**Architecture:** A standalone preparation script owns WikiText parsing and JSONL creation. The collector owns quota allocation, token-length validation, and random window generation while reusing the existing model execution path.

**Tech Stack:** Python, PyArrow, Hugging Face Transformers, PyTorch, pytest.

---

### Task 1: Article Dataset Preparation

**Files:**
- Create: `random_cache_srdp_scripts-1/prepare_wiki_articles.py`
- Test: `random_cache_srdp_scripts-1/test_wiki_article_sampling.py`

- [ ] Write tests proving spaced WikiText headings are classified correctly.
- [ ] Run the focused test and confirm it fails because the preparation module is absent.
- [ ] Implement heading parsing, article reconstruction, token-length calculation, minimum-length filtering, and JSONL output.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Collector Allocation and Window Sampling

**Files:**
- Modify: `random_cache_srdp_scripts-1/collect_random_cache_acceptance.py`
- Test: `random_cache_srdp_scripts-1/test_wiki_article_sampling.py`

- [ ] Write tests proving every article receives at least one sample and quotas sum to `max(max_samples, article_count)`.
- [ ] Write tests proving sampled lengths are multiples of four and windows remain within article bounds.
- [ ] Run the tests and confirm failures identify the missing allocation and sampling functions.
- [ ] Implement largest-remainder allocation, article-specific `max_n`, uniform `n`, and uniform start selection.
- [ ] Replace `--wiki-parquet` with `--wiki-jsonl`, add `--reserve-tokens`, and remove `--length-bias`.
- [ ] Add article/window fields to output metadata.
- [ ] Run focused tests and syntax checks.

### Task 3: Real Dataset Generation and Validation

**Files:**
- Create: `/data2/group_谈海生/mumura/dynamick/predictor/filtered_wikitext/train_articles_qwen3.jsonl`

- [ ] Run the preparation script with the Qwen3 tokenizer and collector defaults.
- [ ] Validate every record has unique `article_id`, non-empty text, and a sufficient stored token length.
- [ ] Re-tokenize representative records and confirm stored lengths match.
- [ ] Run the collector parser/help path to verify the new CLI.
