#!/usr/bin/env python3
"""Sweep true synchronous batch TPOT with one loaded nano-vllm-moe engine.

Unlike a sequential collection of batch-one requests, every row here places B
requests in the scheduler together and executes one packed draft/verify model
step.  Request-visible wall TPOT and aggregate ms/output-token are both
reported so the result can be compared with KTransformers' batch report.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from nanovllm.benchmarks.eval_tpot.cases import build_cases
from nanovllm.benchmarks.eval_tpot.config import (
    configure_optimized_env,
    parse_args,
    validate_runtime_config,
)
from nanovllm.benchmarks.eval_tpot.data import (
    load_dataset_samples,
    prepare_prompt_tokens,
)
from nanovllm.benchmarks.eval_tpot.metrics import (
    collect_profile_metrics,
    reset_llm_profile,
    run_prompt_batch_generate,
)
from nanovllm.benchmarks.eval_tpot.runtime import (
    create_llm,
    parse_csv,
    reset_runtime_seed,
    validate_kv_cache_capacity,
    warmup_llm,
)


def _write_markdown(path: Path, metadata: dict, rows: list[dict]) -> None:
    lines = [
        "# nano-vllm-moe true-batch TPOT",
        "",
        f"- model: `{metadata['model_path']}`",
        f"- prompt tokens: `{metadata['prompt_tokens']}` per request",
        f"- output tokens: `{metadata['max_output_tokens']}` per request",
        f"- K: `{metadata['max_draft_tokens']}`",
        f"- temperature: `{metadata['temperature']}`",
        "",
        "| B | full wall TPOT ms | stable replay TPOT ms | stable p50 | stable p95 | stable aggregate tok/s | valid |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['batch_size']} | {row['tpot_ms']:.3f} | "
            f"{row['stable_replay_step_wall_ms_mean']:.3f} | "
            f"{row['stable_replay_step_wall_ms_p50']:.3f} | "
            f"{row['stable_replay_step_wall_ms_p95']:.3f} | "
            f"{row['stable_replay_aggregate_decode_tok_s']:.3f} | "
            f"{'yes' if row['output_fixed_length_ok'] else 'no'} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    batch_sizes = (
        parse_csv(args.batch_sizes, int) if str(args.batch_sizes).strip() else [2, 3, 5]
    )
    if any(batch_size < 1 for batch_size in batch_sizes):
        raise ValueError("--batch-sizes must contain only positive values")
    if len(batch_sizes) != len(set(batch_sizes)):
        raise ValueError("--batch-sizes must not contain duplicates")
    args.batch_size = max(batch_sizes)
    # ModelRunner's generic allocation warmup uses
    # max_num_batched_tokens // max_model_len full-length sequences.  That is
    # useful for a general serving engine, but it allocates a very large GPU
    # MoE fallback tensor for decode-focused CPU-offload experiments (two
    # 8192-token rows with the default 16384-token budget on this model).  A
    # true-batch TPOT run only needs its actual replicated prompts to fit, so
    # profile one max-length sequence and validate the real prefill below.
    requested_max_num_batched_tokens = int(args.max_num_batched_tokens)
    args.max_num_batched_tokens = min(
        requested_max_num_batched_tokens,
        int(args.max_model_len),
    )
    # Re-run the semantic gate after promoting max_num_seqs to the sweep max.
    validate_runtime_config(args)
    args._optimized_env_overrides = configure_optimized_env(args)

    cases = build_cases(args)
    if len(cases) != 1:
        raise ValueError(
            "true-batch sweep requires exactly one workload case; provide one "
            "dataset/cache ratio/output length/K/segment/allocation/repeat"
        )
    case = cases[0]
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    llm = None
    try:
        llm = create_llm(args, case, 0)
        warmup_llm(
            llm,
            temperature=float(args.temperature),
            prompt=str(args.warmup_prompt),
        )
        samples = load_dataset_samples(str(case["dataset"]), args)
        if not samples:
            raise RuntimeError(f"no samples for dataset={case['dataset']}")
        prompt_tokens, prompt_info = prepare_prompt_tokens(
            llm.tokenizer,
            samples[0],
            max_input_tokens=(
                int(args.max_input_tokens)
                if int(args.max_input_tokens) > 0
                else int(args.max_model_len) - 1
            ),
            truncate_prompts=bool(args.truncate_prompts),
        )
        if prompt_tokens is None:
            raise RuntimeError("selected prompt was skipped during tokenization")
        required_prefill_tokens = int(prompt_info["prompt_tokens"]) * max(batch_sizes)
        if required_prefill_tokens > int(args.max_num_batched_tokens):
            raise ValueError(
                "replicated true-batch prompt exceeds the decode benchmark's "
                "single-sequence allocation-warmup budget: "
                f"batch_size={max(batch_sizes)}, "
                f"prompt_tokens={prompt_info['prompt_tokens']}, "
                f"required_prefill_tokens={required_prefill_tokens}, "
                f"budget={args.max_num_batched_tokens}. Reduce the batch or "
                "prompt length (or use a smaller --max-model-len that still "
                "covers one request)."
            )
        requested_max_output = int(case["max_output_tokens"])
        max_tokens = min(
            requested_max_output,
            int(args.max_model_len) - int(prompt_info["prompt_tokens"]),
        )
        if max_tokens < 2:
            raise ValueError("true-batch TPOT requires at least two output tokens")
        validate_kv_cache_capacity(
            llm,
            prompt_tokens=int(prompt_info["prompt_tokens"]),
            max_tokens=max_tokens,
            batch_size=max(batch_sizes),
        )

        rows = []
        for batch_size in batch_sizes:
            runtime_seed = int(args.seed) + int(batch_size)
            # Exclude one-time shape compilation and allocator setup just as
            # the KTransformers batch report excludes graph capture and times
            # stable replay steps.  Warm every requested true-batch shape,
            # rather than relying on the batch-one engine warmup above.
            batch_warmup_output_tokens = min(32, max_tokens)
            reset_runtime_seed(runtime_seed)
            _ = run_prompt_batch_generate(
                llm,
                prompt_tokens,
                batch_size=batch_size,
                temperature=float(args.temperature),
                max_tokens=batch_warmup_output_tokens,
                ignore_eos=True,
                eos_token_id=getattr(llm.config, "eos", None),
                max_model_len=int(args.max_model_len),
            )
            reset_runtime_seed(runtime_seed)
            reset_llm_profile(llm)
            result = run_prompt_batch_generate(
                llm,
                prompt_tokens,
                batch_size=batch_size,
                temperature=float(args.temperature),
                max_tokens=max_tokens,
                ignore_eos=bool(case.get("ignore_eos", False)),
                eos_token_id=getattr(llm.config, "eos", None),
                max_model_len=int(args.max_model_len),
            )
            if bool(args.collect_profile):
                result.update(
                    collect_profile_metrics(llm.get_profile(reset=True), result)
                )
            if bool(args.save_text):
                result["generated_texts"] = [
                    llm.tokenizer.decode(row)
                    for row in result.get("generated_token_ids", [])
                ]
            if not bool(args.save_token_ids):
                result.pop("generated_token_ids", None)
            result.update(
                {
                    "runtime_seed": runtime_seed,
                    "prompt_tokens": int(prompt_info["prompt_tokens"]),
                    "max_output_tokens": max_tokens,
                    "max_draft_tokens": int(case["max_draft_tokens"]),
                    "temperature": float(args.temperature),
                    "batch_warmup_output_tokens": batch_warmup_output_tokens,
                }
            )
            if result.get("output_validation_error") and bool(
                args.fail_on_output_validation_error
            ):
                raise RuntimeError(result["output_validation_error"])
            rows.append(result)
            print(
                "BATCH_RESULT_JSON "
                + json.dumps(result, ensure_ascii=True, sort_keys=True),
                flush=True,
            )

        metadata = {
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "model_path": str(args.model_path),
            "dataset": str(case["dataset"]),
            "prompt_tokens": int(prompt_info["prompt_tokens"]),
            "max_output_tokens": max_tokens,
            "max_draft_tokens": int(case["max_draft_tokens"]),
            "batch_sizes": batch_sizes,
            "temperature": float(args.temperature),
            "batch_warmup_output_tokens": min(32, max_tokens),
            "cache_ratio": float(case["cache_ratio"]),
            "verify_buckets": parse_csv(args.verify_cuda_graph_bucket_steps, int),
            "kt_capture_bs": list(llm.config.kt_capture_bs),
            "requested_max_num_batched_tokens": requested_max_num_batched_tokens,
            "effective_max_num_batched_tokens": int(args.max_num_batched_tokens),
            "optimized_env_overrides": dict(args._optimized_env_overrides),
            "argv": raw_argv,
        }
        summary = {"metadata": metadata, "batch_results": rows}
        (output_dir / "batch_summary.json").write_text(
            json.dumps(summary, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        _write_markdown(output_dir / "batch_summary.md", metadata, rows)
        print("SUMMARY_JSON " + json.dumps(summary, ensure_ascii=True, sort_keys=True))
        return 0
    finally:
        if llm is not None:
            llm.exit()


if __name__ == "__main__":
    raise SystemExit(main())
