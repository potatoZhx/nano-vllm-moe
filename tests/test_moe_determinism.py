#!/usr/bin/env python3
"""Directly test heterogeneous MoE determinism: same inputs, different cache states."""
import torch
import sys
import os

# Add to path to import nanovllm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_moe_layer_determinism():
    """Test that heterogeneous_moe_forward produces identical output
    regardless of which experts are cached (GPU) vs uncached (CPU fallback)."""
    from nanovllm.expert.cache import LayerExpertCache
    from nanovllm.layers.fuse_moe.heterogeneous import heterogeneous_moe_forward
    from nanovllm.layers.fuse_moe.heterogeneous import SiluAndMul

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    hidden_dim = 2048
    intermediate_dim = 768  # Qwen3 MoE intermediate
    num_experts = 128
    top_k = 8
    num_tokens = 5  # 1 + max_draft_tokens
    num_slots = 64  # 50% cache ratio

    # Create synthetic expert weights (FP32 on CPU → BF16 on GPU)
    cpu_pool = {}
    for eid in range(num_experts):
        gate_up = torch.randn(intermediate_dim * 2, hidden_dim, dtype=torch.float32)
        down = torch.randn(hidden_dim, intermediate_dim, dtype=torch.float32)
        cpu_pool[eid] = {"gate_up": gate_up, "down": down}

    cache = LayerExpertCache(
        num_experts=num_experts,
        slots_per_layer=num_slots,
        gate_up_shape=(intermediate_dim * 2, hidden_dim),
        down_shape=(hidden_dim, intermediate_dim),
        device=device,
        dtype=dtype,
        cpu_expert_pool=cpu_pool,
    )

    # Populate all 64 slots with experts 0-63
    for eid in range(num_slots):
        cache.put_to_slot(
            eid, eid,
            cpu_pool[eid]["gate_up"],
            cpu_pool[eid]["down"],
        )
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Create test input
    hidden_states = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
    selected_experts = torch.randint(0, num_experts, (num_tokens, top_k), device=device)
    routing_weights = torch.rand(num_tokens, top_k, dtype=torch.float32, device=device)
    routing_weights = routing_weights / routing_weights.sum(dim=1, keepdim=True)
    routing_weights = routing_weights.to(dtype=dtype)  # match model dtype

    act_fn = SiluAndMul()

    # Create GPU fallback workspace (same as model_runner does)
    from nanovllm.layers.fuse_moe.heterogeneous import GpuFallbackWorkspace
    max_fallback_experts = 32
    workspace = GpuFallbackWorkspace(
        max_experts=max_fallback_experts,
        gate_up_shape=(intermediate_dim * 2, hidden_dim),
        down_shape=(hidden_dim, intermediate_dim),
        device=device,
        dtype=dtype,
    )

    # --- Scenario A: Cache as-is (experts 0-63 cached, 64-127 uncached) ---
    out_a = heterogeneous_moe_forward(
        hidden_states=hidden_states,
        selected_experts=selected_experts,
        routing_weights=routing_weights,
        expert_cache=cache,
        cpu_expert_pool=cpu_pool,
        act_fn=act_fn,
        cpu_expert_execution_enabled=False,
        gpu_fallback_workspace=workspace,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()

    # --- Scenario B: Evict expert 0, cache expert 100 (different cache state) ---
    cache.put_to_slot(0, 100, cpu_pool[100]["gate_up"], cpu_pool[100]["down"])
    if device.type == "cuda":
        torch.cuda.synchronize()

    out_b = heterogeneous_moe_forward(
        hidden_states=hidden_states,
        selected_experts=selected_experts,
        routing_weights=routing_weights,
        expert_cache=cache,
        cpu_expert_pool=cpu_pool,
        act_fn=act_fn,
        cpu_expert_execution_enabled=False,
        gpu_fallback_workspace=workspace,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()

    # --- Scenario C: ALL experts uncached (no cache at all) ---
    empty_cache = LayerExpertCache(
        num_experts=num_experts,
        slots_per_layer=num_slots,
        gate_up_shape=(intermediate_dim * 2, hidden_dim),
        down_shape=(hidden_dim, intermediate_dim),
        device=device,
        dtype=dtype,
        cpu_expert_pool=cpu_pool,
    )
    out_c = heterogeneous_moe_forward(
        hidden_states=hidden_states,
        selected_experts=selected_experts,
        routing_weights=routing_weights,
        expert_cache=empty_cache,
        cpu_expert_pool=cpu_pool,
        act_fn=act_fn,
        cpu_expert_execution_enabled=False,
        gpu_fallback_workspace=workspace,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Compare
    diff_ab = (out_a.float() - out_b.float()).abs()
    diff_ac = (out_a.float() - out_c.float()).abs()
    diff_bc = (out_b.float() - out_c.float()).abs()

    print(f"Scenario A (experts 0-63 cached):    {out_a[0, :5].float().tolist()}")
    print(f"Scenario B (expert 100 replaces 0):   {out_b[0, :5].float().tolist()}")
    print(f"Scenario C (all uncached):            {out_c[0, :5].float().tolist()}")
    print(f"\nDiff A vs B: max={diff_ab.max().item():.10f}, mean={diff_ab.mean().item():.10f}")
    print(f"Diff A vs C: max={diff_ac.max().item():.10f}, mean={diff_ac.mean().item():.10f}")
    print(f"Diff B vs C: max={diff_bc.max().item():.10f}, mean={diff_bc.mean().item():.10f}")

    tol = 1e-5
    all_match = (diff_ab.max().item() < tol and diff_ac.max().item() < tol
                 and diff_bc.max().item() < tol)
    print(f"\nAll scenarios match (tol={tol}): {all_match}")

    # Also compare execution plans
    flat_sel = selected_experts.reshape(-1)
    print(f"\nSelected experts: {flat_sel.tolist()[:20]}...")

    # Check which experts are GPU vs CPU in each scenario
    for name, c in [("A (0-63)", cache), ("B (100+1-63)", cache), ("C (empty)", empty_cache)]:
        slot_indices, gpu_mask = c.remap_experts_to_slots(flat_sel)
        n_gpu = gpu_mask.sum().item()
        n_cpu = (~gpu_mask).sum().item()
        print(f"  {name}: {n_gpu} GPU routes, {n_cpu} CPU routes")

    return all_match


if __name__ == "__main__":
    ok = test_moe_layer_determinism()
    sys.exit(0 if ok else 1)
