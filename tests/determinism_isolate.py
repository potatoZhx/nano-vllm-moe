#!/usr/bin/env python3
"""Isolate: test cpu_backend computational equivalence."""
import torch

# Test 1: torch.sigmoid + mul vs F.silu equivalence
torch.manual_seed(42)
x = torch.randn(128, 256)

# Old way
gate = x[:, :128]
up = x[:, 128:]
act_old = torch.sigmoid(gate)
act_old.mul_(gate)
act_old.mul_(up)

# New way
act_new = torch.nn.functional.silu(x[:, :128]) * x[:, 128:]

diff = (act_old - act_new).abs().max().item()
print(f"Test 1 - SiLU equivalence: max_diff={diff:.10f}, match={diff < 1e-6}")

# Test 2: F.linear + silu vs torch.mm + sigmoid + mul
torch.manual_seed(42)
hidden = torch.randn(32, 128)
gate_up_w = torch.randn(256, 128)
down_w = torch.randn(128, 128)

# Old way
gate_up = torch.mm(hidden, gate_up_w.t())
gate_old = gate_up[:, :128]
up_old = gate_up[:, 128:]
act_old2 = torch.sigmoid(gate_old)
act_old2.mul_(gate_old)
act_old2.mul_(up_old)
expert_out_old = torch.mm(act_old2, down_w.t())

# New way
gate_up2 = torch.nn.functional.linear(hidden, gate_up_w)
act_new2 = torch.nn.functional.silu(gate_up2[:, :128]) * gate_up2[:, 128:]
expert_out_new = torch.nn.functional.linear(act_new2, down_w)

diff2 = (expert_out_old - expert_out_new).abs().max().item()
print(f"Test 2 - Full expert compute equivalence: max_diff={diff2:.10f}, match={diff2 < 1e-6}")

# Test 3: Check if numerical differences in intermediate values could propagate
print(f"\nIntermediate analysis:")
print(f"  old gate_up max: {gate_up.abs().max().item():.6f}")
print(f"  new gate_up max: {gate_up2.abs().max().item():.6f}")
print(f"  Gate match: {(gate_up - gate_up2).abs().max().item():.10f}")

# Test 4: With larger tensors simulating real model
torch.manual_seed(42)
hidden2 = torch.randn(512, 2048, dtype=torch.float32)
w1 = torch.randn(11008, 2048, dtype=torch.float32)
w2 = torch.randn(2048, 11008, dtype=torch.float32)

# Small batch test
h = hidden2[:4]
gate_up3 = torch.nn.functional.linear(h, w1)
act_new3 = torch.nn.functional.silu(gate_up3[:, :5504]) * gate_up3[:, 5504:]
out_new = torch.nn.functional.linear(act_new3, w2)

# Old way
gate_up_old = torch.mm(h, w1.t())
gate_old3 = gate_up_old[:, :5504]
up_old3 = gate_up_old[:, 5504:]
act_old3 = torch.sigmoid(gate_old3)
act_old3.mul_(gate_old3)
act_old3.mul_(up_old3)
out_old = torch.mm(act_old3, w2.t())

diff3 = (out_old - out_new).abs().max().item()
print(f"\nTest 3 - Large model equivalence: max_diff={diff3:.10f}, match={diff3 < 1e-10}")

# Test 5: GPU vs CPU computation
if torch.cuda.is_available():
    h_gpu = hidden2[:4].cuda()
    w1_gpu = w1.cuda()
    w2_gpu = w2.cuda()

    # GPU
    gu_gpu = torch.nn.functional.linear(h_gpu, w1_gpu)
    act_gpu = torch.nn.functional.silu(gu_gpu[:, :5504]) * gu_gpu[:, 5504:]
    out_gpu = torch.nn.functional.linear(act_gpu, w2_gpu)

    # CPU
    h_cpu = hidden2[:4].cpu()
    w1_cpu = w1.cpu()
    w2_cpu = w2.cpu()
    gu_cpu = torch.nn.functional.linear(h_cpu, w1_cpu)
    act_cpu = torch.nn.functional.silu(gu_cpu[:, :5504]) * gu_cpu[:, 5504:]
    out_cpu = torch.nn.functional.linear(act_cpu, w2_cpu)

    diff4 = (out_gpu.cpu() - out_cpu).abs().max().item()
    print(f"\nTest 4 - GPU vs CPU equivalence: max_diff={diff4:.10f}")

print("\nAll tests complete.")
