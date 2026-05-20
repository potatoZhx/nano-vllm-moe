#!/usr/bin/env python3
"""Check model architecture info."""
import json
from transformers import AutoConfig

cfg = AutoConfig.from_pretrained("/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B")
print(f"num_hidden_layers: {cfg.num_hidden_layers}")
print(f"num_experts: {getattr(cfg, 'num_experts', 'N/A')}")
print(f"num_experts_per_tok: {cfg.num_experts_per_tok}")
print(f"intermediate_size: {cfg.intermediate_size}")
print(f"hidden_size: {cfg.hidden_size}")
print(f"moe_intermediate_size: {getattr(cfg, 'moe_intermediate_size', 'N/A')}")
print(f"model_type: {cfg.model_type}")

# Also check which layers are dense vs MoE
if hasattr(cfg, 'decoder_sparse_step'):
    print(f"decoder_sparse_step: {cfg.decoder_sparse_step}")
# Check for dense layers count
for attr in ['num_experts', 'n_routed_experts', 'num_shared_experts']:
    val = getattr(cfg, attr, None)
    if val is not None:
        print(f"{attr}: {val}")
