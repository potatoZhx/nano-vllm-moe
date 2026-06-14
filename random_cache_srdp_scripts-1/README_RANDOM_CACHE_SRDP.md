# Random-cache theoretical acceptance predictor scripts

This folder contains a fresh implementation for Qwen/Qwen3-30B-A3B random-cache acceptance prediction.

## Files

- `expert_subset_random_cache.py`  
  Qwen/Qwen3 MoE wrapper. Adds `TASK_MODE=random_cache` behavior. During standard prefill it records expert activations, builds a per-layer LFU/LRU expert cache, and during random-cache decode replaces out-of-cache experts with random experts from the top-scored cached candidate pool.

- `prepare_wiki_articles.py`
  Reconstructs WikiText articles from the raw Parquet rows, filters articles
  that are too short for collection, and writes article-level JSONL with Qwen3
  token lengths.

- `collect_random_cache_acceptance.py`  
  Data collector. For WikiText, it gives every filtered article at least one
  sample, distributes additional samples by article length, and draws uniform
  random windows within each article. It runs standard prefill, builds cache,
  decodes under `random_cache`, and teacher-forces the target/standard model on
  the same draft prefix.

- `build_random_cache_dataset.py`  
  Dataset builder. It converts JSONL records into branch tensors: `route_raw`, `route_summary`, `token_features`, `hidden`, `history`, and `y`.

- `train_acceptance_predictor.py`  
  Tiny predictor trainer. It uses branch encoders for route raw signature, route summary, token difficulty, hidden state, and history state. The training target is theoretical acceptance alpha.

- `run_collect_random_cache.sh`  
  Example SLURM launcher.

## Suggested placement

Copy this folder into the repository root as:

```bash
cp -r random_cache_srdp_scripts ~/MOE_SD/random_cache_srdp
```

Then run scripts from the repository root.

## 1. Prepare Wiki training articles

```bash
python -u random_cache_srdp/prepare_wiki_articles.py \
  --input-parquet /data2/group_谈海生/lagin/data/wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet \
  --output-jsonl /data2/group_谈海生/mumura/dynamick/predictor/filtered_wikitext/train_articles_qwen3.jsonl \
  --model-path /data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B \
  --min-prefill-n 8 \
  --decode-steps 15 \
  --reserve-tokens 5
```

## 2. Collect Wiki training data

```bash
python -u random_cache_srdp/collect_random_cache_acceptance.py \
  --dataset wiki \
  --model-path /data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B \
  --wiki-jsonl /data2/group_谈海生/mumura/dynamick/predictor/filtered_wikitext/train_articles_qwen3.jsonl \
  --output-dir /data2/group_谈海生/mumura/dynamick/predictor/random_cache_runs \
  --cache-policy lfu \
  --cache-ratio 0.25 0.5 0.75 \
  --cache-ratio-weights 1 2 1 \
  --cache-topc-ratio 0.5 \
  --decode-steps 20 \
  --min-prefill-n 8 \
  --max-prefill-n 1024 \
  --reserve-tokens 5 \
  --max-samples 300
```

For WikiText, `--max-samples` is a minimum. If the filtered dataset contains
more articles, the collector emits at least one sample per article.

`--cache-ratio` accepts one or more values. For each sample, the collector
selects one value using `--cache-ratio-weights`; weights are relative and do
not need to sum to one. If weights are omitted, all ratios are equally likely.
The selection sequence is reproducible with `--seed`. The original single-value
form, such as `--cache-ratio 0.5`, remains supported.

## 3. Collect MTBench test data

```bash
python -u random_cache_srdp/collect_random_cache_acceptance.py \
  --dataset mtbench \
  --model-path /data2/group_谈海生/lagin/models/Qwen3-30B-A3B-Base \
  --output-dir /data2/group_谈海生/lagin/data/Sd_Data/data/random_cache_runs \
  --cache-policy lfu \
  --cache-ratio 0.25 0.5 0.75 \
  --cache-ratio-weights 1 2 1 \
  --cache-topc-ratio 0.5 \
  --decode-steps 20 \
  --min-prefill-n 8 \
  --max-prefill-n 1024 \
  --max-samples 300
```


## Prefill length sampling

The collector samples prefill length as:

```text
prefill_len = 4 * n
```

`n` is controlled by `--min-prefill-n` and `--max-prefill-n`. For example,
`--min-prefill-n 8 --max-prefill-n 1024` samples prefill lengths from 32 to
4096 tokens, clipped by the article and model-context limits. The collector
samples `n` uniformly and then chooses a uniform random start within the article.

## 4. Build train/test dataset

Use the two output directories produced by the collector. Example:

```bash
python -u random_cache_srdp/build_random_cache_dataset.py \
  --train /data2/group_谈海生/lagin/data/Sd_Data/data/random_cache_runs/wiki_random_cache_lfu_ratios0.25-0.5-0.75_weights1-2-1_topc0.5 \
  --test /data2/group_谈海生/lagin/data/Sd_Data/data/random_cache_runs/mtbench_random_cache_lfu_ratios0.25-0.5-0.75_weights1-2-1_topc0.5 \
  --output /data2/group_谈海生/lagin/data/Sd_Data/data/random_cache_acceptance_dataset.pt
```

## 5. Train predictor

```bash
python -u random_cache_srdp/train_acceptance_predictor.py \
  --data-file /data2/group_谈海生/lagin/data/Sd_Data/data/random_cache_acceptance_dataset.pt \
  --output-dir /data2/group_谈海生/lagin/models/SRDP_Experiments/random_cache_acceptance \
  --epochs 40 \
  --batch-size 512
```

## Notes

- The collector does not save full vocabulary logits. It computes theoretical acceptance online and saves the scalar label.
- The target logits are computed by standard-mode teacher forcing on the same draft prefix, avoiding the prefix-divergence issue in baseline greedy decoding.
- The dataset includes cumulative/EMA/max history features for RSD, REP mass, and draft-logit entropy. It does not use previous ground-truth alpha as a feature, avoiding train/deploy leakage.
- I have not run these scripts on the actual A800/Qwen3 environment here. If your installed Qwen3 implementation exposes different MoE field names, adjust `expert_subset_random_cache.py` around `gate`, `experts`, and shared expert attributes.
