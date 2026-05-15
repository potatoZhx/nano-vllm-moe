"""Compare draft vs verify logits with step-index tracking.

Key question: why does output start with draft token 16 when verify predicts 25?
"""
import argparse, json, os, torch, torch.nn.functional as F

_orig_linear = F.linear
_lm_head_info = {"weight_ptr": None, "weight_shape": None}
_captured_lm_head = []

def _patched_linear(input, weight, bias=None):
    out = _orig_linear(input, weight, bias)
    if _lm_head_info["weight_ptr"] is not None and weight.data_ptr() == _lm_head_info["weight_ptr"]:
        _captured_lm_head.append({
            "idx": len(_captured_lm_head),
            "shape": list(out.shape),
            "dim0": out.shape[0],
            "argmax": out.argmax(dim=-1).tolist(),
        })
    return out

# Track what spec_engine does
_acceptance_records = []  # (num_accepted, next_token, keep_after_start)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B")
    p.add_argument("--num-seqs", type=int, default=4)
    p.add_argument("--warmup-tokens", type=int, default=64)
    p.add_argument("--output-len", type=int, default=8)
    p.add_argument("--output", default="/tmp/numdiff.json")
    args = p.parse_args()

    from nanovllm import LLM, SamplingParams
    from nanovllm.layers.sampler import Sampler
    from nanovllm.engine.speculative import spec_engine as se

    # ---- Patch F.linear ----
    F.linear = _patched_linear

    # ---- Patch SpeculativeEngine.speculative_step to trace acceptance ----
    orig_step = se.SpeculativeEngine.speculative_step
    def patched_step(self, seqs):
        """Wrap speculative_step and capture per-seq results."""
        result = orig_step(self, seqs)
        # Capture details from the step_trace
        if hasattr(self, "_step_traces") and self._step_traces:
            trace = self._step_traces[-1]
            for seq_trace in trace["sequences"]:
                _acceptance_records.append({
                    "seq_id": seq_trace.get("seq_id"),
                    "drafted": seq_trace.get("drafted_tokens"),
                    "accepted": seq_trace.get("accepted_draft_tokens"),
                    "next_token": seq_trace.get("next_token"),
                    "reject_position": seq_trace.get("reject_position"),
                    "rejected": seq_trace.get("rejected"),
                })
        return result
    se.SpeculativeEngine.speculative_step = patched_step

    # ---- Build LLM ----
    print("Creating LLM...", flush=True)
    llm = LLM(
        args.model_path, dist_port=5010,
        enforce_eager=False,
        max_num_batched_tokens=16384, max_num_seqs=512, max_model_len=4096,
        gpu_memory_utilization=0.99,
        inference_mode="spec", enable_heterogeneous=True, enable_speculative=True,
        draft_top_c=0, max_draft_tokens=4,
        slots_per_layer=0,
        spec_enable_prefetch=False,
        acceptance_strategy="greedy",
        engine_profile=False,
    )

    runner = llm.model_runner
    _lm_head_info["weight_ptr"] = runner.model.lm_head.weight.data_ptr()
    _lm_head_info["weight_shape"] = list(runner.model.lm_head.weight.shape)
    print(f"lm_head: shape={_lm_head_info['weight_shape']}", flush=True)
    print(f"acceptance_strategy: {llm.spec_engine.acceptance_strategy_name}", flush=True)

    # Warmup
    print("Warmup...", flush=True)
    llm.generate(
        ["warmup prompt "] * args.num_seqs,
        SamplingParams(temperature=0.0, max_tokens=args.warmup_tokens, ignore_eos=True),
        use_tqdm=False,
    )

    _captured_lm_head.clear()
    _acceptance_records.clear()

    # Run benchmark
    print("Benchmark...", flush=True)
    outputs = llm.generate(
        ["test prompt "] * args.num_seqs,
        SamplingParams(temperature=0.0, max_tokens=args.output_len, ignore_eos=True),
        use_tqdm=False,
    )

    print(f"\nAcceptance strategy: {llm.spec_engine.acceptance_strategy_name}", flush=True)

    print(f"\nlm_head calls: {len(_captured_lm_head)}", flush=True)
    for r in _captured_lm_head:
        tag = "DRAFT" if r["dim0"] == args.num_seqs and r["shape"][1] > 1000 else \
               "VERIFY" if r["dim0"] > args.num_seqs else \
               "OTHER"
        # For verify, show per-sequence split
        extra = ""
        if tag == "VERIFY":
            n_seqs = args.num_seqs
            tokens_per_seq = r["dim0"] // n_seqs
            per_seq = []
            for s in range(n_seqs):
                start = s * tokens_per_seq
                end = start + tokens_per_seq
                per_seq.append(r["argmax"][start:end])
            extra = f" per_seq={per_seq}"
        print(f"  lm_head[{r['idx']}]: shape={r['shape']} [{tag}]{extra}", flush=True)
        if tag != "VERIFY":
            print(f"    argmax={r['argmax']}", flush=True)

    print(f"\nAcceptance records:", flush=True)
    for r in _acceptance_records:
        print(f"  seq_id={r['seq_id']}: drafted={r['drafted']} accepted={r['accepted']} next={r['next_token']} rejected={r['rejected']} reject_pos={r['reject_position']}", flush=True)

    print(f"\nOutputs: {len(outputs)} sequences", flush=True)
    for i, o in enumerate(outputs):
        print(f"  seq {i}: tokens={o['token_ids']}", flush=True)

    result = {
        "acceptance_strategy": llm.spec_engine.acceptance_strategy_name,
        "lm_head_calls": _captured_lm_head,
        "acceptance": _acceptance_records,
        "outputs": [o["token_ids"] for o in outputs],
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved to {args.output}", flush=True)

    llm.exit()

if __name__ == "__main__":
    main()
