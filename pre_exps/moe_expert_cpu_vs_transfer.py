#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark one MoE expert's single CPU forward time vs single CPU->GPU transfer time.

Designed for Qwen3-30B-A3B local HF checkpoints, including the newer Transformers
implementation where a layer's experts are stored as fused 3D tensors:

    model.layers.N.mlp.experts.gate_up_proj: [num_experts, 2 * intermediate, hidden]
    model.layers.N.mlp.experts.down_proj   : [num_experts, hidden, intermediate]

It also keeps limited compatibility with older ModuleList-style experts named like:

    model.layers.N.mlp.experts.K

The script loads the full model, runs real generate() once or more to observe real
routing, captures a real input batch routed to one expert, then benchmarks:

  1. one CPU forward of that single expert on the captured real routed tokens
  2. one CPU->GPU transfer of that single expert's weights

No HF download is attempted: local_files_only=True is used.
"""

import argparse
import copy
import gc
import json
import os
import re
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ----------------------------- CLI -----------------------------


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--model",
        type=str,
        default="/data1/group_谈海生/mumura/models/Qwen--Qwen3-30B-A3B",
        help="Local Hugging Face model directory.",
    )
    p.add_argument("--device-map", type=str, default="auto", help='"auto" or "none".')
    p.add_argument("--main-device", type=str, default="cuda:0", help='Used when --device-map none.')
    p.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    p.add_argument("--attn-implementation", type=str, default="default")

    p.add_argument("--layer", type=str, default="auto", help='Layer index or "auto".')
    p.add_argument("--expert", type=str, default="auto", help='Expert index or "auto".')
    p.add_argument(
        "--expert-name",
        type=str,
        default=None,
        help=(
            "Exact expert name. For fused Qwen3-MoE use e.g. "
            "model.layers.10.mlp.experts:3 . "
            "For ModuleList style use e.g. model.layers.10.mlp.experts.3"
        ),
    )

    p.add_argument("--prompt", type=str, default="请简要解释一下混合专家模型的推理过程。")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--routing-runs", type=int, default=1, help="How many real generate() runs for auto routing selection.")

    p.add_argument(
        "--capture-policy",
        type=str,
        default="max_tokens",
        choices=["first", "last", "max_tokens"],
        help="Which call of the selected expert to capture during real inference.",
    )
    p.add_argument("--cpu-repeats", type=int, default=20)
    p.add_argument("--transfer-repeats", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)

    p.add_argument("--transfer-device", type=str, default="cuda:0")
    p.add_argument("--cpu-threads", type=int, default=None)
    p.add_argument(
        "--cpu-bench-dtype",
        type=str,
        default="original",
        choices=["original", "float32", "bfloat16", "float16"],
        help="Dtype used for isolated CPU expert benchmark. original keeps checkpoint dtype.",
    )
    p.add_argument(
        "--pin-memory",
        action="store_true",
        help="Use pinned CPU tensors for pure H2D copy benchmark.",
    )
    p.add_argument(
        "--skip-generate",
        action="store_true",
        help="Only inspect/detect experts; useful for debugging load and names.",
    )
    p.add_argument("--output-json", type=str, default=None)

    return p.parse_args()


def resolve_dtype(x: str):
    if x == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[x]


def resolve_cpu_bench_dtype(x: str):
    if x == "original":
        return None
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[x]


# ----------------------------- utilities -----------------------------


def cuda_sync(device: Optional[torch.device] = None):
    if not torch.cuda.is_available():
        return
    if device is None:
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)
    else:
        d = torch.device(device)
        if d.type == "cuda":
            torch.cuda.synchronize(d)


def first_param_device(m: nn.Module) -> torch.device:
    for p in m.parameters(recurse=True):
        return p.device
    for b in m.buffers(recurse=True):
        return b.device
    return torch.device("cpu")


def tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def module_nbytes(m: nn.Module) -> int:
    total = 0
    for p in m.parameters(recurse=True):
        total += tensor_nbytes(p)
    for b in m.buffers(recurse=True):
        total += tensor_nbytes(b)
    return total


def module_params(m: nn.Module) -> int:
    return sum(int(p.numel()) for p in m.parameters(recurse=True))


def parse_layer_from_name(name: str) -> Optional[int]:
    m = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
    return int(m.group(1)) if m else None


def parse_expert_from_name(name: str) -> Optional[int]:
    m = re.search(r"(?:^|\.)experts\.(\d+)(?:\.|$)", name)
    return int(m.group(1)) if m else None


def summarize_ms(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"count": 0}
    ys = sorted(float(x) for x in xs)
    return {
        "count": len(ys),
        "mean_ms": statistics.mean(ys),
        "median_ms": statistics.median(ys),
        "min_ms": min(ys),
        "max_ms": max(ys),
        "p90_ms": ys[int(round((len(ys) - 1) * 0.90))],
    }


def tree_to_cpu_detached(x: Any):
    if torch.is_tensor(x):
        return x.detach().cpu().contiguous()
    if isinstance(x, tuple):
        return tuple(tree_to_cpu_detached(v) for v in x)
    if isinstance(x, list):
        return [tree_to_cpu_detached(v) for v in x]
    if isinstance(x, dict):
        return {k: tree_to_cpu_detached(v) for k, v in x.items()}
    return x


def infer_tokens_from_first_tensor(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> int:
    x = None
    if args and torch.is_tensor(args[0]):
        x = args[0]
    else:
        for v in kwargs.values():
            if torch.is_tensor(v):
                x = v
                break
    if x is None:
        return 0
    if x.ndim == 0:
        return 0
    if x.ndim == 1:
        return int(x.shape[0])
    return int(x.numel() // x.shape[-1])


def maybe_pin(t: torch.Tensor, pin: bool) -> torch.Tensor:
    if not pin:
        return t
    try:
        return t.pin_memory()
    except RuntimeError:
        return t


# ----------------------------- expert abstractions -----------------------------


@dataclass
class ExpertRef:
    kind: str  # "fused" or "module"
    name: str
    module: nn.Module
    layer: Optional[int]
    expert: Optional[int] = None

    @property
    def display_name(self) -> str:
        if self.kind == "fused":
            return f"{self.name}:{self.expert}"
        return self.name


class SingleFusedQwen3Expert(nn.Module):
    """One expert sliced out from Qwen3MoeExperts 3D tensors."""

    def __init__(self, gate_up_proj: torch.Tensor, down_proj: torch.Tensor, act_fn):
        super().__init__()
        self.gate_up_proj = nn.Parameter(gate_up_proj.detach().contiguous())
        self.down_proj = nn.Parameter(down_proj.detach().contiguous())
        self.act_fn = act_fn

    def forward(self, hidden_states: torch.Tensor, top_k_weights: Optional[torch.Tensor] = None):
        gate_up = F.linear(hidden_states, self.gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)
        y = self.act_fn(gate) * up
        y = F.linear(y, self.down_proj)
        if top_k_weights is not None:
            y = y * top_k_weights[:, None]
        return y


@dataclass
class CapturedInput:
    kind: str
    tokens: int
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    hidden_states: Optional[torch.Tensor] = None
    top_k_weights: Optional[torch.Tensor] = None
    calls_seen: int = 0


# ----------------------------- model / expert discovery -----------------------------


def is_fused_qwen3_experts_module(m: nn.Module) -> bool:
    gu = getattr(m, "gate_up_proj", None)
    dp = getattr(m, "down_proj", None)
    if not torch.is_tensor(gu) or not torch.is_tensor(dp):
        return False
    if gu.ndim != 3 or dp.ndim != 3:
        return False
    # Expected: gu [E, 2I, H], dp [E, H, I]
    return gu.shape[0] == dp.shape[0] and gu.shape[2] == dp.shape[1]


def discover_experts(model: nn.Module) -> List[ExpertRef]:
    refs: List[ExpertRef] = []

    # New Qwen3-MoE / Transformers v5 style: one module contains all expert weights as 3D tensors.
    for name, m in model.named_modules():
        if is_fused_qwen3_experts_module(m):
            num_experts = int(m.gate_up_proj.shape[0])
            layer = parse_layer_from_name(name)
            for e in range(num_experts):
                refs.append(ExpertRef(kind="fused", name=name, module=m, layer=layer, expert=e))

    if refs:
        return refs

    # Older Mixtral-like style: each expert is a submodule under experts.N.
    for name, m in model.named_modules():
        parts = name.split(".")
        if len(parts) >= 2 and parts[-2] == "experts" and parts[-1].isdigit():
            refs.append(
                ExpertRef(
                    kind="module",
                    name=name,
                    module=m,
                    layer=parse_layer_from_name(name),
                    expert=int(parts[-1]),
                )
            )

    if refs:
        return refs

    # Fallback: regex for any ...experts.N module.
    for name, m in model.named_modules():
        e = parse_expert_from_name(name)
        if e is not None:
            refs.append(ExpertRef(kind="module", name=name, module=m, layer=parse_layer_from_name(name), expert=e))

    return refs


def print_moe_debug(model: nn.Module, max_lines: int = 80):
    print("\n[Debug] Modules whose name/type/params look MoE-related:")
    shown = 0
    for name, m in model.named_modules():
        cls = m.__class__.__name__
        low = f"{name} {cls}".lower()
        interesting = any(k in low for k in ["moe", "expert", "router", "gate"])
        has_3d = False
        for pn, p in m.named_parameters(recurse=False):
            if p.ndim == 3:
                has_3d = True
                break
        if interesting or has_3d:
            direct_params = []
            for pn, p in m.named_parameters(recurse=False):
                direct_params.append(f"{pn}{tuple(p.shape)}:{p.dtype}:{p.device}")
            print(f"  {name} | {cls} | " + ", ".join(direct_params[:4]))
            shown += 1
            if shown >= max_lines:
                print(f"  ... truncated at {max_lines} lines")
                break


def get_ref_by_name(refs: List[ExpertRef], expert_name: str) -> ExpertRef:
    # For fused: accept "module.name:expert_id".
    if ":" in expert_name:
        base, eid = expert_name.rsplit(":", 1)
        for r in refs:
            if r.kind == "fused" and r.name == base and r.expert == int(eid):
                return r

    for r in refs:
        if r.display_name == expert_name or r.name == expert_name:
            return r

    raise ValueError(
        f"找不到 expert-name={expert_name}. 前 20 个可用名称:\n"
        + "\n".join(r.display_name for r in refs[:20])
    )


def select_ref_by_layer_expert(refs: List[ExpertRef], layer_s: str, expert_s: str) -> ExpertRef:
    layer = int(layer_s)
    expert = int(expert_s)
    for r in refs:
        if r.layer == layer and r.expert == expert:
            return r
    raise ValueError(f"找不到 layer={layer}, expert={expert}")


# ----------------------------- real generation and routing -----------------------------


@torch.inference_mode()
def run_generate(model, tokenizer, prompt: str, max_new_tokens: int):
    input_device = model.get_input_embeddings().weight.device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    cuda_sync()
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    cuda_sync()
    return out


def choose_expert_by_real_routing(model, tokenizer, refs: List[ExpertRef], args) -> ExpertRef:
    stats: Dict[str, Dict[str, int]] = {}
    handles = []

    fused_modules = {}
    module_refs = []
    for r in refs:
        if r.kind == "fused":
            fused_modules[r.name] = r.module
        else:
            module_refs.append(r)

    def fused_pre_hook(name):
        def hook(module, hook_args, hook_kwargs=None):
            # Expected args: hidden_states, top_k_index, top_k_weights
            if len(hook_args) < 2 or not torch.is_tensor(hook_args[1]):
                return
            top_k_index = hook_args[1].detach().reshape(-1).to("cpu")
            if top_k_index.numel() == 0:
                return
            num_experts = int(module.gate_up_proj.shape[0])
            counts = torch.bincount(top_k_index.to(torch.long), minlength=num_experts)
            for eid, c in enumerate(counts.tolist()):
                if c <= 0:
                    continue
                key = f"{name}:{eid}"
                s = stats.setdefault(key, {"tokens": 0, "calls": 0})
                s["tokens"] += int(c)
                s["calls"] += 1
        return hook

    def module_pre_hook(r: ExpertRef):
        def hook(module, hook_args, hook_kwargs=None):
            if hook_kwargs is None:
                hook_kwargs = {}
            n = infer_tokens_from_first_tensor(hook_args, hook_kwargs)
            key = r.display_name
            s = stats.setdefault(key, {"tokens": 0, "calls": 0})
            s["tokens"] += int(n)
            s["calls"] += 1
        return hook

    for name, m in fused_modules.items():
        handles.append(m.register_forward_pre_hook(fused_pre_hook(name), with_kwargs=True))

    for r in module_refs:
        try:
            h = r.module.register_forward_pre_hook(module_pre_hook(r), with_kwargs=True)
        except TypeError:
            h = r.module.register_forward_pre_hook(module_pre_hook(r))
        handles.append(h)

    for i in range(args.routing_runs):
        print(f"  real routing generate {i + 1}/{args.routing_runs}")
        run_generate(model, tokenizer, args.prompt, args.max_new_tokens)

    for h in handles:
        h.remove()

    if not stats:
        raise RuntimeError("真实推理中没有捕获到 MoE expert 路由。")

    ranked = sorted(stats.items(), key=lambda kv: (kv[1]["tokens"], kv[1]["calls"]), reverse=True)

    print("\n[真实路由统计 Top-20]")
    for key, s in ranked[:20]:
        print(f"  {key} | calls={s['calls']}, routed_assignments={s['tokens']}")

    best_name = ranked[0][0]
    return get_ref_by_name(refs, best_name)


def choose_expert(model, tokenizer, refs: List[ExpertRef], args) -> ExpertRef:
    if args.expert_name is not None:
        return get_ref_by_name(refs, args.expert_name)

    if args.layer != "auto" and args.expert != "auto":
        return select_ref_by_layer_expert(refs, args.layer, args.expert)

    return choose_expert_by_real_routing(model, tokenizer, refs, args)


# ----------------------------- capture real input for selected expert -----------------------------


def should_replace_capture(policy: str, old_tokens: int, new_tokens: int, has_old: bool) -> bool:
    if not has_old:
        return True
    if policy == "first":
        return False
    if policy == "last":
        return True
    if policy == "max_tokens":
        return new_tokens > old_tokens
    return False


def capture_fused_input(model, tokenizer, ref: ExpertRef, args) -> CapturedInput:
    cap: Dict[str, Any] = {"hidden_states": None, "top_k_weights": None, "tokens": -1, "calls_seen": 0}
    eid = int(ref.expert)

    def hook(module, hook_args, hook_kwargs=None):
        if len(hook_args) < 3:
            return
        hidden_states, top_k_index, top_k_weights = hook_args[:3]
        if not (torch.is_tensor(hidden_states) and torch.is_tensor(top_k_index) and torch.is_tensor(top_k_weights)):
            return

        mask = top_k_index == eid
        if not bool(mask.any().item()):
            return

        pos = mask.nonzero(as_tuple=False)
        token_idx = pos[:, 0]
        topk_pos = pos[:, 1]

        cur_h = hidden_states[token_idx]
        cur_w = top_k_weights[token_idx, topk_pos]
        n = int(cur_h.shape[0])
        cap["calls_seen"] += 1

        if should_replace_capture(args.capture_policy, cap["tokens"], n, cap["hidden_states"] is not None):
            cap["hidden_states"] = cur_h.detach().cpu().contiguous()
            cap["top_k_weights"] = cur_w.detach().cpu().contiguous()
            cap["tokens"] = n

    h = ref.module.register_forward_pre_hook(hook, with_kwargs=True)
    run_generate(model, tokenizer, args.prompt, args.max_new_tokens)
    h.remove()

    if cap["hidden_states"] is None:
        raise RuntimeError(
            f"所选 expert {ref.display_name} 在 capture generate 中没有被路由到。"
            "可增大 --max-new-tokens，换 prompt，或使用 --layer auto --expert auto。"
        )

    return CapturedInput(
        kind="fused",
        tokens=int(cap["tokens"]),
        args=(),
        kwargs={},
        hidden_states=cap["hidden_states"],
        top_k_weights=cap["top_k_weights"],
        calls_seen=int(cap["calls_seen"]),
    )


def capture_module_input(model, tokenizer, ref: ExpertRef, args) -> CapturedInput:
    cap: Dict[str, Any] = {"args": None, "kwargs": None, "tokens": -1, "calls_seen": 0}

    def hook(module, hook_args, hook_kwargs=None):
        if hook_kwargs is None:
            hook_kwargs = {}
        n = infer_tokens_from_first_tensor(hook_args, hook_kwargs)
        cap["calls_seen"] += 1
        if should_replace_capture(args.capture_policy, cap["tokens"], n, cap["args"] is not None):
            cap["args"] = tree_to_cpu_detached(hook_args)
            cap["kwargs"] = tree_to_cpu_detached(hook_kwargs)
            cap["tokens"] = n

    try:
        h = ref.module.register_forward_pre_hook(hook, with_kwargs=True)
    except TypeError:
        h = ref.module.register_forward_pre_hook(hook)

    run_generate(model, tokenizer, args.prompt, args.max_new_tokens)
    h.remove()

    if cap["args"] is None:
        raise RuntimeError(f"所选 expert {ref.display_name} 在 capture generate 中没有被路由到。")

    return CapturedInput(
        kind="module",
        tokens=int(cap["tokens"]),
        args=cap["args"],
        kwargs=cap["kwargs"],
        calls_seen=int(cap["calls_seen"]),
    )


def capture_real_input(model, tokenizer, ref: ExpertRef, args) -> CapturedInput:
    if ref.kind == "fused":
        return capture_fused_input(model, tokenizer, ref, args)
    return capture_module_input(model, tokenizer, ref, args)


# ----------------------------- build single CPU expert -----------------------------


def build_single_cpu_expert(ref: ExpertRef, cpu_dtype=None) -> nn.Module:
    if ref.kind == "fused":
        eid = int(ref.expert)
        gu = ref.module.gate_up_proj.detach()[eid].contiguous().cpu()
        dp = ref.module.down_proj.detach()[eid].contiguous().cpu()
        if cpu_dtype is not None:
            gu = gu.to(cpu_dtype)
            dp = dp.to(cpu_dtype)
        act_fn = getattr(ref.module, "act_fn", F.silu)
        m = SingleFusedQwen3Expert(gu, dp, act_fn)
        m.eval()
        return m

    m = copy.deepcopy(ref.module).to("cpu")
    if cpu_dtype is not None:
        m = m.to(dtype=cpu_dtype)
    m.eval()
    return m


def convert_capture_dtype(captured: CapturedInput, dtype):
    if dtype is None:
        return captured
    if captured.kind == "fused":
        hs = captured.hidden_states.to(dtype=dtype) if captured.hidden_states is not None else None
        # keep routing weights in same dtype as hidden states for faithful multiply
        tw = captured.top_k_weights.to(dtype=dtype) if captured.top_k_weights is not None else None
        return CapturedInput(
            kind=captured.kind,
            tokens=captured.tokens,
            args=captured.args,
            kwargs=captured.kwargs,
            hidden_states=hs,
            top_k_weights=tw,
            calls_seen=captured.calls_seen,
        )

    def cast_tree(x):
        if torch.is_tensor(x) and x.is_floating_point():
            return x.to(dtype=dtype)
        if isinstance(x, tuple):
            return tuple(cast_tree(v) for v in x)
        if isinstance(x, list):
            return [cast_tree(v) for v in x]
        if isinstance(x, dict):
            return {k: cast_tree(v) for k, v in x.items()}
        return x

    return CapturedInput(
        kind=captured.kind,
        tokens=captured.tokens,
        args=cast_tree(captured.args),
        kwargs=cast_tree(captured.kwargs),
        calls_seen=captured.calls_seen,
    )


# ----------------------------- benchmarks -----------------------------


@torch.inference_mode()
def benchmark_cpu_forward(cpu_expert: nn.Module, captured: CapturedInput, repeats: int, warmup: int) -> List[float]:
    if captured.kind == "fused":
        hs = captured.hidden_states
        w = captured.top_k_weights
        assert hs is not None
        for _ in range(warmup):
            _ = cpu_expert(hs, w)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = cpu_expert(hs, w)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
        return times

    for _ in range(warmup):
        _ = cpu_expert(*captured.args, **captured.kwargs)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = cpu_expert(*captured.args, **captured.kwargs)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return times


def expert_param_tensors(cpu_expert: nn.Module, pin_memory: bool) -> List[torch.Tensor]:
    ts = []
    for p in cpu_expert.parameters(recurse=True):
        ts.append(maybe_pin(p.detach().contiguous().cpu(), pin_memory))
    for b in cpu_expert.buffers(recurse=True):
        ts.append(maybe_pin(b.detach().contiguous().cpu(), pin_memory))
    return ts


def benchmark_h2d_prealloc(cpu_expert: nn.Module, device: torch.device, repeats: int, warmup: int, pin_memory: bool) -> List[float]:
    srcs = expert_param_tensors(cpu_expert, pin_memory)
    dsts = [torch.empty_like(s, device=device) for s in srcs]

    for _ in range(warmup):
        for s, d in zip(srcs, dsts):
            d.copy_(s, non_blocking=pin_memory)
        cuda_sync(device)

    times = []
    for _ in range(repeats):
        cuda_sync(device)
        t0 = time.perf_counter()
        for s, d in zip(srcs, dsts):
            d.copy_(s, non_blocking=pin_memory)
        cuda_sync(device)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    del dsts
    cuda_sync(device)
    torch.cuda.empty_cache()
    return times


def benchmark_h2d_alloc_copy(cpu_expert: nn.Module, device: torch.device, repeats: int, warmup: int, pin_memory: bool) -> List[float]:
    srcs = expert_param_tensors(cpu_expert, pin_memory)
    times = []

    for i in range(warmup + repeats):
        cuda_sync(device)
        t0 = time.perf_counter()
        dsts = [s.to(device, non_blocking=pin_memory) for s in srcs]
        cuda_sync(device)
        t1 = time.perf_counter()
        if i >= warmup:
            times.append((t1 - t0) * 1000.0)
        del dsts
        gc.collect()
        cuda_sync(device)
        torch.cuda.empty_cache()

    return times


def benchmark_module_to_gpu(cpu_expert: nn.Module, device: torch.device, repeats: int, warmup: int) -> List[float]:
    times = []
    for i in range(warmup + repeats):
        tmp = copy.deepcopy(cpu_expert)
        tmp.eval()
        cuda_sync(device)
        t0 = time.perf_counter()
        tmp.to(device)
        cuda_sync(device)
        t1 = time.perf_counter()
        if i >= warmup:
            times.append((t1 - t0) * 1000.0)
        del tmp
        gc.collect()
        cuda_sync(device)
        torch.cuda.empty_cache()
    return times


# ----------------------------- main -----------------------------


def main():
    args = parse_args()

    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)

    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU 才能测试 CPU->GPU expert 传输耗时。")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    print("[1] 加载 tokenizer，本地离线 local_files_only=True")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[2] 加载完整模型，本地离线 local_files_only=True")
    load_kwargs = dict(
        torch_dtype=resolve_dtype(args.torch_dtype),
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    if args.device_map != "none":
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation != "default":
        load_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()
    if args.device_map == "none":
        model.to(args.main_device)

    print("[3] 检测 MoE experts，兼容 fused 3D tensor 和 experts.N 两种结构")
    refs = discover_experts(model)
    if not refs:
        print_moe_debug(model)
        raise RuntimeError(
            "没有检测到 MoE expert。上面已打印 MoE 相关模块，请检查模型结构或 transformers 版本。"
        )

    kind_counts = {}
    for r in refs:
        kind_counts[r.kind] = kind_counts.get(r.kind, 0) + 1
    print(f"    detected logical experts: {len(refs)} | by kind: {kind_counts}")
    print("    examples:")
    for r in refs[:10]:
        print(f"      {r.display_name} | kind={r.kind} | layer={r.layer} | expert={r.expert}")

    if args.skip_generate:
        print("--skip-generate enabled; stop after detection.")
        return

    print("\n[4] 选择 expert")
    ref = choose_expert(model, tokenizer, refs, args)
    print(f"    selected: {ref.display_name} | kind={ref.kind} | layer={ref.layer} | expert={ref.expert}")

    if ref.kind == "fused":
        gu = ref.module.gate_up_proj.detach()[int(ref.expert)]
        dp = ref.module.down_proj.detach()[int(ref.expert)]
        expert_params = int(gu.numel() + dp.numel())
        expert_bytes = int(tensor_nbytes(gu) + tensor_nbytes(dp))
        expert_device = gu.device
        expert_dtype = str(gu.dtype)
    else:
        expert_params = module_params(ref.module)
        expert_bytes = module_nbytes(ref.module)
        expert_device = first_param_device(ref.module)
        first_param = next(ref.module.parameters(recurse=True), None)
        expert_dtype = str(first_param.dtype) if first_param is not None else "unknown"

    print(f"    expert device: {expert_device}")
    print(f"    expert dtype : {expert_dtype}")
    print(f"    expert params: {expert_params:,}")
    print(f"    expert size  : {expert_bytes / 1024 ** 2:.3f} MiB")

    print("\n[5] 再跑一次完整 generate，捕获该 expert 的真实 routed 输入")
    captured = capture_real_input(model, tokenizer, ref, args)
    print(f"    capture_policy       : {args.capture_policy}")
    print(f"    calls_seen           : {captured.calls_seen}")
    print(f"    routed tokens/call   : {captured.tokens}")

    cpu_dtype = resolve_cpu_bench_dtype(args.cpu_bench_dtype)
    if cpu_dtype is not None:
        print(f"\n[6] 构造单个 CPU expert，并转换 CPU benchmark dtype: {cpu_dtype}")
    else:
        print("\n[6] 构造单个 CPU expert，保持 checkpoint 原始 dtype")

    cpu_expert = build_single_cpu_expert(ref, cpu_dtype=cpu_dtype)
    captured_for_cpu = convert_capture_dtype(captured, cpu_dtype)

    print("[7] Benchmark: 单次 CPU expert forward，输入来自真实推理路由")
    try:
        cpu_forward_ms = benchmark_cpu_forward(cpu_expert, captured_for_cpu, args.cpu_repeats, args.warmup)
    except RuntimeError as e:
        raise RuntimeError(
            "CPU forward benchmark 失败。若 checkpoint 是 float16 且 CPU kernel 不支持，"
            "请加 --cpu-bench-dtype bfloat16 或 --cpu-bench-dtype float32 再试。"
        ) from e

    transfer_device = torch.device(args.transfer_device)
    print(f"[8] Benchmark: 单个 expert 权重 CPU -> GPU 传输，target={transfer_device}")
    h2d_prealloc_ms = benchmark_h2d_prealloc(
        cpu_expert, transfer_device, args.transfer_repeats, args.warmup, args.pin_memory
    )
    h2d_alloc_copy_ms = benchmark_h2d_alloc_copy(
        cpu_expert, transfer_device, args.transfer_repeats, args.warmup, args.pin_memory
    )
    module_to_gpu_ms = benchmark_module_to_gpu(cpu_expert, transfer_device, args.transfer_repeats, args.warmup)

    cpu_s = summarize_ms(cpu_forward_ms)
    pre_s = summarize_ms(h2d_prealloc_ms)
    alloc_s = summarize_ms(h2d_alloc_copy_ms)
    mod_s = summarize_ms(module_to_gpu_ms)

    result = {
        "model": args.model,
        "selected_expert": {
            "name": ref.display_name,
            "kind": ref.kind,
            "layer": ref.layer,
            "expert": ref.expert,
            "device_before_cpu_copy": str(expert_device),
            "dtype_before_cpu_copy": expert_dtype,
            "params": expert_params,
            "size_mib": expert_bytes / 1024 ** 2,
        },
        "real_capture": {
            "prompt": args.prompt,
            "max_new_tokens": args.max_new_tokens,
            "capture_policy": args.capture_policy,
            "calls_seen": captured.calls_seen,
            "routed_tokens_for_single_forward": captured.tokens,
        },
        "benchmark_settings": {
            "cpu_threads": torch.get_num_threads(),
            "cpu_bench_dtype": args.cpu_bench_dtype,
            "transfer_device": str(transfer_device),
            "pin_memory": args.pin_memory,
            "warmup": args.warmup,
            "cpu_repeats": args.cpu_repeats,
            "transfer_repeats": args.transfer_repeats,
        },
        "single_cpu_forward_ms": cpu_s,
        "single_expert_transfer_ms": {
            "pure_h2d_preallocated_copy": pre_s,
            "alloc_plus_h2d_copy": alloc_s,
            "deepcopy_module_to_gpu": mod_s,
        },
        "ratios_transfer_div_cpu_forward_median": {
            "pure_h2d_preallocated_copy": pre_s["median_ms"] / cpu_s["median_ms"],
            "alloc_plus_h2d_copy": alloc_s["median_ms"] / cpu_s["median_ms"],
            "deepcopy_module_to_gpu": mod_s["median_ms"] / cpu_s["median_ms"],
        },
    }

    print("\n========== 核心结果 ==========")
    print(f"Expert                         : {ref.display_name}")
    print(f"Expert size                    : {expert_bytes / 1024 ** 2:.3f} MiB")
    print(f"Routed tokens for one forward   : {captured.tokens}")
    print(f"CPU expert forward median       : {cpu_s['median_ms']:.4f} ms")
    print(f"CPU->GPU preallocated copy median: {pre_s['median_ms']:.4f} ms")
    print(f"CPU->GPU alloc+copy median       : {alloc_s['median_ms']:.4f} ms")
    print(f"deepcopy(single expert).to(cuda) : {mod_s['median_ms']:.4f} ms")
    print("\n传输耗时 / CPU计算耗时:")
    for k, v in result["ratios_transfer_div_cpu_forward_median"].items():
        print(f"  {k}: {v:.4f}x")

    print("\n========== 完整 JSON ==========")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n已保存 JSON: {args.output_json}")


if __name__ == "__main__":
    main()
