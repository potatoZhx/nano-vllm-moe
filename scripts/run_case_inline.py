#!/usr/bin/env python3
"""Run a single case for precision validation. Called as subprocess."""
import sys, json, time, os, torch

def main():
    args = json.loads(sys.argv[1])
    repo_root = args["repo_root"]
    sys.path.insert(0, repo_root)
    from nanovllm import LLM, SamplingParams

    if args.get("standard_mode"):
        llm = LLM(
            args["model_path"], dist_port=args["dist_port"],
            enforce_eager=args["enforce_eager"],
            max_num_batched_tokens=min(8192, args["max_model_len"]),
            max_num_seqs=1, max_model_len=args["max_model_len"],
            gpu_memory_utilization=0.85,
            inference_mode="standard", enable_heterogeneous=False,
            enable_speculative=False,
        )
    else:
        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(args["model_path"])
        num_experts = int(getattr(hf_config, "num_experts"))
        slots = max(1, int(round(num_experts * args["cache_ratio"])))
        artifact = args.get("calibration_artifact", "")

        llm = LLM(
            args["model_path"], dist_port=args["dist_port"],
            enforce_eager=args["enforce_eager"],
            max_num_batched_tokens=min(8192, args["max_model_len"]),
            max_num_seqs=1, max_model_len=args["max_model_len"],
            gpu_memory_utilization=0.85,
            inference_mode="spec", enable_heterogeneous=True,
            enable_speculative=True,
            heterogeneous_slots_per_layer=slots,
            max_draft_tokens=8, draft_top_c=0,
            draft_reroute_policy=args["policy"],
            draft_reroute_artifact=artifact,
            acceptance_strategy="greedy",
            cpu_expert_execution_enabled=args.get("cpu_expert_execution_enabled", True),
            cpu_expert_pin_memory=args.get("cpu_expert_pin_memory", False),
            cpu_expert_backend="fused",
            cpu_expert_workspace_max_routes=16384,
            cpu_expert_packed_min_routes=1,
            cpu_expert_parallel_mode="serial", cpu_expert_num_threads=4,
            spec_profile=True, engine_profile=True, engine_profile_cuda_sync=True,
            spec_enable_prefetch=False,
        )

    prompt_text = args["prompt_text"]
    prompt_ids = llm.tokenizer.encode(prompt_text)[:args["max_model_len"] // 2]
    sampling = SamplingParams(temperature=0.0, ignore_eos=True,
                             max_tokens=args["output_len"])

    # Warmup
    llm.generate([prompt_ids],
                 SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=4),
                 use_tqdm=False)

    if not args.get("standard_mode"):
        llm.get_profile(reset=True)

    t0 = time.time()
    outputs = llm.generate([prompt_ids], sampling, use_tqdm=False)
    elapsed = time.time() - t0

    result = {
        "token_ids": outputs[0]["token_ids"],
        "text": outputs[0]["text"],
        "prompt_ids_len": len(prompt_ids),
        "elapsed_sec": elapsed,
    }

    if not args.get("standard_mode"):
        profile = llm.get_profile(reset=True)
        drafted = 0
        accepted = 0
        position_drafted = {}
        position_accepted = {}
        traces = profile.get("spec_step_traces", [])
        for step in traces:
            for seq in step.get("sequences", []):
                d = int(seq.get("drafted_tokens", 0) or 0)
                a = max(0, min(int(seq.get("accepted_draft_tokens", 0) or 0), d))
                drafted += d
                accepted += a
                for position in range(1, d + 1):
                    position_drafted[position] = position_drafted.get(position, 0) + 1
                for position in range(1, a + 1):
                    position_accepted[position] = position_accepted.get(position, 0) + 1
        position_stats = []
        for position in sorted(position_drafted):
            drafted_count = int(position_drafted[position])
            accepted_count = int(position_accepted.get(position, 0))
            position_stats.append(
                {
                    "position": int(position),
                    "drafted_count": drafted_count,
                    "accepted_count": accepted_count,
                    "acceptance_rate": float(accepted_count / drafted_count) if drafted_count else 0.0,
                }
            )
        dc = float(profile.get("run_draft_calls", 0))
        dt = float(profile.get("run_draft_infer_ms_total", 0))
        result["drafted"] = drafted
        result["accepted"] = accepted
        result["acceptance_rate"] = float(accepted / drafted) if drafted > 0 else 0.0
        result["draft_position_acceptance"] = position_stats
        result["draft_forward_ms_avg"] = dt / dc if dc > 0 else 0.0
        result["draft_replays"] = int(profile.get("model_draft_graph_replay_count", 0) or 0)
        result["cpu_route_ratio"] = float(profile.get("cpu_route_ratio_sum", 0) or 0)
        result["effective_cache_ratio"] = float(slots / num_experts)
        result["slots"] = slots

    llm.exit()
    print("RESULT_JSON:", json.dumps(result))

if __name__ == "__main__":
    main()
