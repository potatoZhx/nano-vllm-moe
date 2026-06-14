"""Collect random_cache theoretical acceptance data for Qwen/Qwen3 MoE models.

The collector uses the same target model in two modes:
  1. standard: target/baseline MoE routing.
  2. random_cache: draft/intervention MoE routing.

For each sample:
  - Run standard prefill and collect activated experts per layer.
  - Build a per-layer expert cache by LFU or LRU.
  - Decode 20 draft steps with random_cache.
  - In parallel, run standard-mode teacher forcing on the same draft prefix.
  - Store per-step features and the theoretical acceptance label
        alpha = sum_v min(p_target(v), q_draft(v)).

This script stores compact top-k logits, route metadata, final hidden embedding,
and alpha. It does not store full vocabulary logits.

python -u collect_random_cache_acceptance.py \
  --dataset mtbench \
  --model-path /data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B \
  --wiki-jsonl /data2/group_谈海生/mumura/dynamick/predictor/filtered_wikitext/train_articles_qwen3.jsonl \
  --output-dir /data2/group_谈海生/mumura/dynamick/predictor/random_cache_runs \
  --cache-policy lru \
  --cache-ratio 0.1 0.125 0.25 0.31 0.375 0.5 \
  --cache-ratio-weights 1 1 2 3 2 1 \
  --cache-topc-ratio 0.7 \
  --decode-steps 15 \
  --min-prefill-n 3 \
  --max-prefill-n 1024 \
  --max-samples 200

Done. Successful samples: 2000. Output: /data2/group_谈海生/mumura/dynamick/predictor/random_cache_runs/wiki_random_cache_lru_ratios0.1-0.125-0.25-0.31-0.375-0.5_weights1-1-2-3-2-1_topc0.7/acceptance_summary_20260613_225251.jsonl  
Done. Successful samples: 200. Output: /data2/group_谈海生/mumura/dynamick/predictor/random_cache_runs/mtbench_random_cache_lru_ratios0.1-0.125-0.25-0.31-0.375-0.5_weights1-1-2-3-2-1_topc0.7/acceptance_summary_20260614_125411.jsonl

Successful samples: 611. Output:
/data2/group_谈海生/mumura/dynamick/predictor/random_cache_runs_smoke/wiki_random_cache_lru_ratio0.5_topc0.5/acceptance_summary_20260613_204930.jsonl
  
/data2/group_谈海生/mumura/dynamick/predictor/random_cache_runs_smoke/mtbench_random_cache_lru_ratio0.5_topc0.5/acceptance_summary_20260613_221149.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
import traceback
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers.cache_utils import DynamicCache

if not hasattr(DynamicCache, "get_usable_length"):
    def get_usable_length(self, input_seq_len, layer_idx=0):
        return self.get_seq_length(layer_idx)
    DynamicCache.get_usable_length = get_usable_length

from expert_subset_random_cache import (
    apply_random_cache_wrapper_to_qwen,
    build_all_expert_caches,
    collect_moe_metadata,
    enable_prefill_collection,
    reset_prefill_cache_state,
    set_moe_mode,
    summarize_cache,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_request_id() -> int:
    return int(time.time() * 1_000_000) + random.randint(0, 1000)


def normalize_cache_ratio_config(
    cache_ratios: List[float],
    cache_ratio_weights: List[float] | None,
) -> tuple[List[float], List[float]]:
    ratios = [float(value) for value in cache_ratios]
    if not ratios:
        raise ValueError("--cache-ratio requires at least one value")
    if any(not math.isfinite(value) or not (0.0 < value <= 1.0) for value in ratios):
        raise ValueError(
            f"--cache-ratio values must be finite cache ratios in (0, 1], got {ratios}"
        )

    if cache_ratio_weights is None:
        return ratios, [1.0] * len(ratios)

    weights = [float(value) for value in cache_ratio_weights]
    if len(weights) != len(ratios):
        raise ValueError(
            "--cache-ratio-weights must have the same length as --cache-ratio"
        )
    if any(not math.isfinite(value) or value <= 0.0 for value in weights):
        raise ValueError(
            "--cache-ratio-weights values must be finite positive weights, "
            f"got {weights}"
        )
    return ratios, weights


def choose_cache_ratio(
    cache_ratios: List[float],
    cache_ratio_weights: List[float],
    rng: random.Random,
) -> float:
    return float(rng.choices(cache_ratios, weights=cache_ratio_weights, k=1)[0])


def _compact_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return repr(value)


def cache_ratio_output_name(
    dataset: str,
    cache_policy: str,
    cache_ratios: List[float],
    cache_ratio_weights: List[float],
    cache_topc_ratio: float,
) -> str:
    prefix = f"{dataset}_random_cache_{cache_policy}"
    topc = _compact_number(cache_topc_ratio)
    if len(cache_ratios) == 1:
        return f"{prefix}_ratio{_compact_number(cache_ratios[0])}_topc{topc}"

    ratios = "-".join(_compact_number(value) for value in cache_ratios)
    weights = "-".join(_compact_number(value) for value in cache_ratio_weights)
    return f"{prefix}_ratios{ratios}_weights{weights}_topc{topc}"


def sample_prefill_len(
    actual_len: int,
    min_n: int = 1,
    max_n: int = 2024,
    multiple: int = 4,
    rng=random,
) -> int:
    """Uniformly sample prefill length = multiple * n."""
    if min_n < 1:
        raise ValueError(f"min_n must be >= 1, got {min_n}")
    if max_n < min_n:
        raise ValueError(f"max_n must be >= min_n, got min_n={min_n}, max_n={max_n}")

    max_allowed_n = actual_len // multiple
    if max_allowed_n < min_n:
        # Source sequence is shorter than the requested lower bound. Use the
        # longest valid multiple of `multiple`; caller can skip too-short cases.
        return max(multiple, min(actual_len, max_allowed_n * multiple))

    n_lo = min_n
    n_hi = min(max_n, max_allowed_n)
    if n_hi <= n_lo:
        return min(actual_len, multiple * n_lo)

    n = rng.randint(n_lo, n_hi)
    return min(actual_len, multiple * n)


def article_max_prefill_n(
    token_length: int,
    max_prefill_n: int,
    max_seq_len: int,
    decode_steps: int,
    reserve_tokens: int,
) -> int:
    article_limit = (token_length - decode_steps - reserve_tokens) // 4
    context_limit = (max_seq_len - decode_steps - reserve_tokens) // 4
    return min(max_prefill_n, article_limit, context_limit)


def allocate_article_quotas(
    articles: List[dict],
    max_samples: int,
    decode_steps: int,
    reserve_tokens: int,
) -> List[int]:
    """Give each article one sample and allocate extras by usable length."""
    if not articles:
        return []
    if max_samples < 1:
        raise ValueError(f"max_samples must be >= 1, got {max_samples}")

    target = max(max_samples, len(articles))
    extra = target - len(articles)
    quotas = [1] * len(articles)
    if extra == 0:
        return quotas

    weights = [
        max(1, int(article["token_length"]) - decode_steps - reserve_tokens)
        for article in articles
    ]
    total_weight = sum(weights)
    exact_extras = [extra * weight / total_weight for weight in weights]
    floor_extras = [int(value) for value in exact_extras]
    for idx, value in enumerate(floor_extras):
        quotas[idx] += value

    remaining = extra - sum(floor_extras)
    remainder_order = sorted(
        range(len(articles)),
        key=lambda idx: (exact_extras[idx] - floor_extras[idx], weights[idx], -idx),
        reverse=True,
    )
    for idx in remainder_order[:remaining]:
        quotas[idx] += 1
    return quotas


def sample_article_window(
    input_ids: torch.Tensor,
    min_prefill_n: int,
    max_prefill_n: int,
    max_seq_len: int,
    decode_steps: int,
    reserve_tokens: int,
    rng=random,
) -> tuple[torch.Tensor, dict]:
    token_length = int(input_ids.size(1))
    article_max_n = article_max_prefill_n(
        token_length=token_length,
        max_prefill_n=max_prefill_n,
        max_seq_len=max_seq_len,
        decode_steps=decode_steps,
        reserve_tokens=reserve_tokens,
    )
    if article_max_n < min_prefill_n:
        raise ValueError(
            f"Article with {token_length} tokens cannot satisfy "
            f"min_prefill_n={min_prefill_n}; article_max_n={article_max_n}"
        )

    sampled_n = rng.randint(min_prefill_n, article_max_n)
    prefill_len = 4 * sampled_n
    window_start = rng.randint(0, token_length - prefill_len)
    sample = input_ids[:, window_start : window_start + prefill_len]
    return sample, {
        "window_start": window_start,
        "sampled_prefill_n": sampled_n,
        "article_max_prefill_n": article_max_n,
    }


def load_wiki_article_records(wiki_jsonl: str) -> List[dict]:
    records: List[dict] = []
    with open(wiki_jsonl, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            required = {"article_id", "title", "text", "token_length"}
            missing = required.difference(record)
            if missing:
                raise ValueError(
                    f"{wiki_jsonl}:{line_no} missing fields: {sorted(missing)}"
                )
            records.append(record)
    return records


def prepare_wikitext_data(tokenizer, wiki_jsonl: str, args) -> List[dict]:
    print(f"Loading filtered WikiText JSONL: {wiki_jsonl}")
    records = load_wiki_article_records(wiki_jsonl)
    eligible = [
        record
        for record in records
        if article_max_prefill_n(
            token_length=int(record["token_length"]),
            max_prefill_n=args.max_prefill_n,
            max_seq_len=args.max_seq_len,
            decode_steps=args.decode_steps,
            reserve_tokens=args.reserve_tokens,
        )
        >= args.min_prefill_n
    ]
    if not eligible:
        raise RuntimeError("No Wiki articles satisfy the configured prefill limits.")

    quotas = allocate_article_quotas(
        eligible,
        max_samples=args.max_samples,
        decode_steps=args.decode_steps,
        reserve_tokens=args.reserve_tokens,
    )
    prepared: List[dict] = []
    for record, quota in tqdm(
        zip(eligible, quotas),
        total=len(eligible),
        desc="Tokenizing Wiki articles",
        unit="article",
    ):
        input_ids = tokenizer(
            record["text"],
            return_tensors="pt",
            add_special_tokens=False,
            verbose=False,
        ).input_ids
        actual_length = int(input_ids.size(1))
        stored_length = int(record["token_length"])
        if actual_length != stored_length:
            raise ValueError(
                f"Article {record['article_id']} token length mismatch: "
                f"stored={stored_length}, actual={actual_length}. "
                "Rebuild the filtered dataset with the current tokenizer."
            )
        prepared.append(
            {
                **record,
                "input_ids": input_ids,
                "quota": quota,
                "article_max_prefill_n": article_max_prefill_n(
                    token_length=actual_length,
                    max_prefill_n=args.max_prefill_n,
                    max_seq_len=args.max_seq_len,
                    decode_steps=args.decode_steps,
                    reserve_tokens=args.reserve_tokens,
                ),
            }
        )

    total_samples = sum(quotas)
    print(
        f"Prepared {len(prepared)} Wiki articles "
        f"({len(records) - len(prepared)} excluded), "
        f"allocated samples={total_samples}, requested minimum={args.max_samples}."
    )
    return prepared


def prepare_mtbench_data(tokenizer, mtbench_jsonl: str, max_seq_len: int) -> List[torch.Tensor]:
    print(f"Loading MTBench jsonl: {mtbench_jsonl}")
    chunks: List[torch.Tensor] = []
    with open(mtbench_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            history = item.get("history", [])
            text = ""
            for turn in history:
                if "user" in turn:
                    text += f"User: {turn['user']}\n"
                if "bot" in turn:
                    text += f"Bot: {turn['bot']}\n"
            if not text:
                continue
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_len).input_ids
            if ids.size(1) >= 4:
                chunks.append(ids)
    print(f"Prepared {len(chunks)} MTBench samples.")
    return chunks


@torch.no_grad()
def decode_step(model, input_ids, attention_mask, past_key_values):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
        output_hidden_states=True,
    )
    logits = outputs.logits[:, -1, :]
    embedding = outputs.hidden_states[-1][:, -1, :]
    return logits, embedding, outputs.past_key_values


@torch.no_grad()
def theoretical_acceptance(q_logits: torch.Tensor, p_logits: torch.Tensor) -> float:
    # Use float32 for numerical stability. Shape: [1, vocab].
    q = F.softmax(q_logits.float(), dim=-1)
    p = F.softmax(p_logits.float(), dim=-1)
    return float(torch.minimum(p, q).sum(dim=-1).item())


@torch.no_grad()
def topk_logit_payload(logits: torch.Tensor, k: int = 32) -> dict:
    k = min(k, logits.size(-1))
    values, indices = torch.topk(logits[0].float(), k=k, dim=-1)
    return {
        "topk": k,
        "indices": indices.detach().cpu().tolist(),
        "values": values.detach().cpu().tolist(),
    }


def collect_one_sample(
    model,
    sample_input: torch.Tensor,
    args,
    cache_ratio: float,
    sample_metadata: dict | None = None,
) -> dict:
    req_id = generate_request_id()
    device = next(model.parameters()).device
    sample_input = sample_input.to(device)
    attention_mask = torch.ones(sample_input.shape, device=device, dtype=torch.long)

    # A. Standard prefill + activation collection.
    reset_prefill_cache_state(model)
    enable_prefill_collection(model, True)
    set_moe_mode(model, "standard")
    with torch.no_grad():
        outputs = model(input_ids=sample_input, attention_mask=attention_mask, use_cache=True)
        past_key_values = outputs.past_key_values
        prefill_token = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(0)
    enable_prefill_collection(model, False)

    # B. Build cache from prefill activations.
    build_all_expert_caches(
        model,
        cache_ratio=cache_ratio,
        cache_policy=args.cache_policy,
        cache_topc_ratio=args.cache_topc_ratio,
    )

    # C. Decode draft and target in parallel on the same draft prefix.
    q_kv = copy.deepcopy(past_key_values)
    p_kv = copy.deepcopy(past_key_values)
    curr_input = prefill_token
    curr_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device, dtype=torch.long)], dim=1)

    steps = []
    draft_tokens = []

    for step_idx in range(args.decode_steps):
        # Draft/intervention distribution q.
        set_moe_mode(model, "random_cache")
        q_logits, q_embedding, q_new_kv = decode_step(model, curr_input, curr_mask, q_kv)
        route_meta = collect_moe_metadata(model, first_token_only=True)
        draft_token = q_logits.argmax(dim=-1)

        # Target/standard distribution p on the same prefix.
        set_moe_mode(model, "standard")
        p_logits, _, p_new_kv = decode_step(model, curr_input, curr_mask, p_kv)

        alpha = theoretical_acceptance(q_logits, p_logits)
        draft_token_id = int(draft_token.item())
        target_token_id = int(p_logits.argmax(dim=-1).item())

        steps.append(
            {
                "step": step_idx + 1,
                "alpha_theoretical": alpha,
                "draft_token": draft_token_id,
                "target_argmax_token": target_token_id,
                "q_topk_logits": topk_logit_payload(q_logits, k=args.logit_topk),
                "final_embedding": q_embedding[0].float().detach().cpu().tolist(),
                "router": route_meta,
            }
        )
        draft_tokens.append(draft_token_id)

        # Feed the draft token to both branches for next-step teacher forcing.
        curr_input = draft_token.unsqueeze(0)
        q_kv = q_new_kv
        p_kv = p_new_kv
        curr_mask = torch.cat([curr_mask, torch.ones((1, 1), device=device, dtype=torch.long)], dim=1)

    metadata = {
        "req_id": req_id,
        "dataset": args.dataset,
        "prefill_len": int(sample_input.size(1)),
        "model_path": args.model_path,
        "task_mode": "random_cache",
        "cache_ratio": cache_ratio,
        "cache_policy": args.cache_policy,
        "cache_topc_ratio": args.cache_topc_ratio,
        "decode_steps": args.decode_steps,
    }
    if sample_metadata:
        metadata.update(sample_metadata)

    return {
        "metadata": metadata,
        "data": {
            "input": sample_input[0].detach().cpu().tolist(),
            "prefill_token": [int(prefill_token.item())],
            "draft_output": draft_tokens,
        },
        "cache_summary": summarize_cache(model),
        "steps": steps,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B",
    )
    parser.add_argument("--dataset", choices=["wiki", "mtbench"], default="wiki")
    parser.add_argument(
        "--wiki-jsonl",
        default=(
            "/data2/group_谈海生/mumura/dynamick/predictor/"
            "filtered_wikitext/train_articles_qwen3.jsonl"
        ),
    )
    parser.add_argument("--mtbench-jsonl", default="/data2/group_谈海生/lagin/data/mtbench101/mtbench101.jsonl")
    parser.add_argument(
        "--output-dir",
        default="/data2/group_谈海生/mumura/dynamick/predictor/random_cache_runs",
    )
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--max-seq-len", type=int, default=8096)
    parser.add_argument("--min-prefill-n", type=int, default=1, help="Lower bound for n in prefill length = 4*n")
    parser.add_argument("--max-prefill-n", type=int, default=2024, help="Upper bound for n in prefill length = 4*n")
    parser.add_argument(
        "--reserve-tokens",
        type=int,
        default=5,
        help="Context tokens reserved in addition to decode_steps.",
    )
    parser.add_argument("--decode-steps", type=int, default=20)
    parser.add_argument(
        "--cache-ratio",
        type=float,
        nargs="+",
        default=[0.5],
        help="One or more cache ratios in (0, 1].",
    )
    parser.add_argument(
        "--cache-ratio-weights",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional positive sampling weight for each --cache-ratio value. "
            "Defaults to equal weights."
        ),
    )
    parser.add_argument("--cache-policy", choices=["lfu", "lru"], default="lfu")
    parser.add_argument("--cache-topc-ratio", type=float, default=0.5)
    parser.add_argument("--logit-topk", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    cache_ratios, cache_ratio_weights = normalize_cache_ratio_config(
        args.cache_ratio,
        args.cache_ratio_weights,
    )
    if args.min_prefill_n < 1:
        raise ValueError("--min-prefill-n must be at least 1")
    if args.max_prefill_n < args.min_prefill_n:
        raise ValueError("--max-prefill-n must be >= --min-prefill-n")
    if args.max_samples < 1:
        raise ValueError("--max-samples must be at least 1")
    if args.decode_steps < 0 or args.reserve_tokens < 0:
        raise ValueError("--decode-steps and --reserve-tokens must be non-negative")
    minimum_context = (
        4 * args.min_prefill_n + args.decode_steps + args.reserve_tokens
    )
    if args.max_seq_len < minimum_context:
        raise ValueError(
            f"--max-seq-len must be at least {minimum_context} for the "
            "configured prefill/decode limits"
        )
    set_seed(args.seed)
    cache_ratio_rng = random.Random(args.seed)

    run_name = cache_ratio_output_name(
        dataset=args.dataset,
        cache_policy=args.cache_policy,
        cache_ratios=cache_ratios,
        cache_ratio_weights=cache_ratio_weights,
        cache_topc_ratio=args.cache_topc_ratio,
    )
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"acceptance_summary_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    print(f"Writing JSONL to: {out_path}")

    print(f"Loading tokenizer/model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    model = apply_random_cache_wrapper_to_qwen(
        model,
        mode="standard",
        cache_ratio=cache_ratios[0],
        cache_policy=args.cache_policy,
        cache_topc_ratio=args.cache_topc_ratio,
    )

    if args.dataset == "wiki":
        wiki_articles = prepare_wikitext_data(tokenizer, args.wiki_jsonl, args)
        samples = None
    else:
        wiki_articles = None
        samples = prepare_mtbench_data(tokenizer, args.mtbench_jsonl, args.max_seq_len)

    ok = 0
    with open(out_path, "a", encoding="utf-8") as f:
        if args.dataset == "wiki":
            total = sum(article["quota"] for article in wiki_articles)
            with tqdm(total=total, desc="Collecting", unit="sample") as progress:
                for article in wiki_articles:
                    for article_sample_idx in range(article["quota"]):
                        try:
                            sample_input, window_meta = sample_article_window(
                                article["input_ids"],
                                min_prefill_n=args.min_prefill_n,
                                max_prefill_n=args.max_prefill_n,
                                max_seq_len=args.max_seq_len,
                                decode_steps=args.decode_steps,
                                reserve_tokens=args.reserve_tokens,
                            )
                            sample_meta = {
                                "article_id": int(article["article_id"]),
                                "article_title": str(article["title"]),
                                "article_token_length": int(article["token_length"]),
                                "article_quota": int(article["quota"]),
                                "article_sample_index": article_sample_idx,
                                **window_meta,
                            }
                            record = collect_one_sample(
                                model,
                                sample_input,
                                args,
                                cache_ratio=choose_cache_ratio(
                                    cache_ratios,
                                    cache_ratio_weights,
                                    cache_ratio_rng,
                                ),
                                sample_metadata=sample_meta,
                            )
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            ok += 1
                            if ok % 5 == 0:
                                f.flush()
                                torch.cuda.empty_cache()
                        except Exception as e:
                            print(
                                f"Error collecting article "
                                f"{article['article_id']} sample {article_sample_idx}: {e}"
                            )
                            traceback.print_exc()
                        finally:
                            progress.update(1)
        else:
            random.shuffle(samples)
            samples = samples[: args.max_samples]
            if not samples:
                raise RuntimeError("No samples found.")
            for raw in tqdm(samples, desc="Collecting", unit="sample"):
                try:
                    prefill_len = sample_prefill_len(
                        raw.size(1),
                        min_n=args.min_prefill_n,
                        max_n=args.max_prefill_n,
                        multiple=4,
                    )
                    if prefill_len < 4 * args.min_prefill_n:
                        continue
                    sample_input = raw[:, :prefill_len]
                    record = collect_one_sample(
                        model,
                        sample_input,
                        args,
                        cache_ratio=choose_cache_ratio(
                            cache_ratios,
                            cache_ratio_weights,
                            cache_ratio_rng,
                        ),
                    )
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    ok += 1
                    if ok % 5 == 0:
                        f.flush()
                        torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Error: {e}")
                    traceback.print_exc()
                    continue

    print(f"Done. Successful samples: {ok}. Output: {out_path}")


if __name__ == "__main__":
    main()
