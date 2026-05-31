#!/usr/bin/env python3
"""Export offline draft-reroute profile artifacts for nano-vllm-moe."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanovllm.scheduling.draft_reroute_profile import save_draft_reroute_profile
from pre_exps.expert_reroute.draft_decode_eval_v2 import (
    calibrate,
    detect_moe_config,
    load_model_and_tokenizer,
    prepare_chunks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export full offline draft-reroute profile artifact.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-calib", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    model, tokenizer = load_model_and_tokenizer(args.model, args.device, dtype_map[args.dtype])
    moe_cfg = detect_moe_config(model)
    calib_chunks = prepare_chunks(tokenizer, args.n_calib, args.seq_len, args.data_file)
    calib = calibrate(model, moe_cfg["moe_attr"], calib_chunks, args.device, moe_cfg["num_experts"])

    tensors = {
        "cond_sim": calib.cond_sim,
        "skip_err": calib.skip_err,
        "sim": calib.sim,
        "sens": calib.sens,
        "act_freq": torch.tensor(calib.act_freq, dtype=torch.float32),
    }
    metadata = {
        "format_version": "2",
        "num_layers": int(calib.L),
        "num_experts": int(calib.N),
        "top_k": int(moe_cfg["top_k"]),
        "model_type": str(getattr(model.config, "model_type", "")),
        "hidden_size": int(getattr(model.config, "hidden_size", 0)),
        "source_model": os.path.abspath(args.model),
    }
    save_draft_reroute_profile(args.output, tensors=tensors, metadata=metadata)
    print(f"Saved draft reroute offline profile: {os.path.abspath(args.output)}")
    print(
        "Profile tensors: "
        f"cond_sim={tuple(tensors['cond_sim'].shape)}, "
        f"skip_err={tuple(tensors['skip_err'].shape)}, "
        f"sim={tuple(tensors['sim'].shape)}, "
        f"sens={tuple(tensors['sens'].shape)}, "
        f"act_freq={tuple(tensors['act_freq'].shape)}"
    )


if __name__ == "__main__":
    main()
