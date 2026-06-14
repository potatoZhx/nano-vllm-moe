"""Build an article-level WikiText JSONL dataset with tokenizer lengths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoTokenizer


DEFAULT_PARQUET = (
    "/data2/group_谈海生/lagin/data/wikitext/"
    "wikitext-2-raw-v1/train-00000-of-00001.parquet"
)
DEFAULT_OUTPUT = (
    "/data2/group_谈海生/mumura/dynamick/predictor/"
    "filtered_wikitext/train_articles_qwen3.jsonl"
)


def heading_level(text: str) -> int:
    """Return the WikiText heading level, including its spaced-equals format."""
    parts = text.strip().split()
    if not parts:
        return 0

    leading = 0
    while leading < len(parts) and parts[leading] == "=":
        leading += 1

    trailing = 0
    while trailing < len(parts) and parts[-1 - trailing] == "=":
        trailing += 1

    if leading > 0 and leading == trailing and len(parts) > leading + trailing:
        return leading
    return 0


def heading_title(text: str) -> str:
    level = heading_level(text)
    if level == 0:
        return ""
    parts = text.strip().split()
    return " ".join(parts[level:-level])


def reconstruct_articles(rows: Iterable[str]) -> list[dict]:
    """Group raw WikiText rows by level-one article headings."""
    articles: list[dict] = []
    current_rows: list[str] = []
    current_title = ""

    for text in rows:
        if heading_level(text) == 1:
            if current_rows:
                article_text = "".join(current_rows).strip()
                if article_text:
                    articles.append({"title": current_title, "text": article_text})
            current_rows = [text]
            current_title = heading_title(text)
        elif current_rows:
            current_rows.append(text)

    if current_rows:
        article_text = "".join(current_rows).strip()
        if article_text:
            articles.append({"title": current_title, "text": article_text})

    return articles


def build_article_records(
    articles: Iterable[dict],
    tokenizer,
    tokenizer_path: str,
    source_parquet: str,
    min_token_length: int,
) -> list[dict]:
    records: list[dict] = []
    for article in tqdm(list(articles), desc="Tokenizing articles", unit="article"):
        token_length = len(
            tokenizer.encode(article["text"], add_special_tokens=False)
        )
        if token_length < min_token_length:
            continue
        records.append(
            {
                "article_id": len(records),
                "title": article["title"],
                "text": article["text"],
                "token_length": token_length,
                "tokenizer_path": tokenizer_path,
                "source_parquet": source_parquet,
            }
        )
    return records


def write_jsonl(records: Iterable[dict], output_path: str) -> int:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with tmp_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    tmp_path.replace(path)
    return count


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-parquet", default=DEFAULT_PARQUET)
    parser.add_argument("--output-jsonl", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--model-path",
        default="/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B",
    )
    parser.add_argument("--min-prefill-n", type=int, default=8)
    parser.add_argument("--decode-steps", type=int, default=15)
    parser.add_argument("--reserve-tokens", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.min_prefill_n < 1:
        raise ValueError("--min-prefill-n must be at least 1")
    if args.decode_steps < 0 or args.reserve_tokens < 0:
        raise ValueError("--decode-steps and --reserve-tokens must be non-negative")

    print(f"Loading WikiText rows: {args.input_parquet}")
    rows = (
        pq.read_table(args.input_parquet, columns=["text"])
        .column("text")
        .to_pylist()
    )
    articles = reconstruct_articles(rows)
    min_token_length = (
        4 * args.min_prefill_n + args.decode_steps + args.reserve_tokens
    )

    print(f"Loading tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    records = build_article_records(
        articles,
        tokenizer=tokenizer,
        tokenizer_path=args.model_path,
        source_parquet=args.input_parquet,
        min_token_length=min_token_length,
    )
    count = write_jsonl(records, args.output_jsonl)

    print(
        f"Wrote {count} articles to {args.output_jsonl}. "
        f"Reconstructed={len(articles)}, filtered={len(articles) - count}, "
        f"minimum_tokens={min_token_length}"
    )


if __name__ == "__main__":
    main()
