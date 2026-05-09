from __future__ import annotations

import argparse
import os
import platform
import sys
import time

import torch


def log(message: str) -> None:
    print(message, flush=True)


def cpu_flags_summary() -> str:
    interesting = (
        "avx2",
        "avx512f",
        "avx512bw",
        "avx512_vnni",
        "avx512_vbmi",
        "avx512_bf16",
        "amx_tile",
        "amx_int8",
        "amx_bf16",
    )
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("flags"):
                    flags = set(line.split(":", 1)[1].strip().split())
                    return " ".join(flag for flag in interesting if flag in flags)
    except OSError as exc:
        return f"unavailable: {exc}"
    return "no flags line"


def import_kt_kernel(*, force_avx2_bf16_class: bool):
    import kt_kernel
    import kt_kernel.utils.amx as amx_mod

    log(f"kt_kernel.__file__={getattr(kt_kernel, '__file__', None)}")
    log(f"kt_kernel.__version__={getattr(kt_kernel, '__version__', None)}")
    log(f"kt_kernel.__cpu_variant__={getattr(kt_kernel, '__cpu_variant__', None)}")
    log(f"AMXBF16_MOE={amx_mod.AMXBF16_MOE}")
    log(f"AVX2BF16_MOE={amx_mod.AVX2BF16_MOE}")
    log(f"_HAS_BF16_SUPPORT(before)={amx_mod._HAS_BF16_SUPPORT}")
    log(f"_HAS_AVX2_BF16_SUPPORT={amx_mod._HAS_AVX2_BF16_SUPPORT}")
    if force_avx2_bf16_class:
        amx_mod._HAS_BF16_SUPPORT = False
        log("_HAS_BF16_SUPPORT forced to False")
    return kt_kernel


def make_wrapper(kt_kernel, args: argparse.Namespace, layer_idx: int):
    return kt_kernel.KTMoEWrapper(
        layer_idx=layer_idx,
        num_experts=args.num_experts,
        num_experts_per_tok=args.top_k,
        hidden_size=args.hidden_size,
        moe_intermediate_size=args.moe_intermediate_size,
        gpu_experts_mask=torch.zeros(args.num_experts, dtype=torch.bool, device="cpu"),
        cpuinfer_threads=args.cpuinfer_threads,
        threadpool_count=args.threadpool_count,
        weight_path=args.weight_path,
        chunked_prefill_size=args.chunked_prefill_size,
        cpu_save=False,
        max_deferred_experts_per_token=0,
        method=args.method,
    )


def load_layer(wrapper, args: argparse.Namespace, layer_idx: int) -> None:
    mapping = torch.arange(args.num_experts, dtype=torch.int64, device="cpu")
    log(f"load_weights layer={layer_idx}")
    start = time.perf_counter()
    wrapper.load_weights(mapping)
    log(f"loaded layer={layer_idx} elapsed_ms={(time.perf_counter() - start) * 1000.0:.1f}")


def run_forward(wrapper, args: argparse.Namespace, layer_idx: int) -> None:
    hidden = torch.randn(args.batch_tokens, args.hidden_size, device="cuda", dtype=torch.bfloat16)
    selected = torch.randint(0, args.num_experts, (args.batch_tokens, args.top_k), device="cuda")
    weights = torch.rand(args.batch_tokens, args.top_k, device="cuda", dtype=torch.float32)
    weights = (weights / weights.sum(dim=-1, keepdim=True)).to(torch.bfloat16)
    stream = torch.cuda.current_stream(hidden.device).cuda_stream
    log(f"forward layer={layer_idx} batch_tokens={args.batch_tokens}")
    start = time.perf_counter()
    out = wrapper.forward(hidden, selected, weights, stream)
    torch.cuda.synchronize()
    log(
        f"forward done layer={layer_idx} elapsed_ms={(time.perf_counter() - start) * 1000.0:.1f} "
        f"shape={tuple(out.shape)} dtype={out.dtype} finite={bool(torch.isfinite(out.float()).all().item())}"
    )


def direct_multi_wrapper(args: argparse.Namespace) -> None:
    kt_kernel = import_kt_kernel(force_avx2_bf16_class=args.force_avx2_bf16_class)
    wrappers = []
    for layer_idx in args.layers:
        log(f"create wrapper layer={layer_idx}")
        wrapper = make_wrapper(kt_kernel, args, layer_idx)
        wrappers.append(wrapper)
        log(f"created wrapper layer={layer_idx} type={type(wrapper)}")
        load_layer(wrapper, args, layer_idx)
        if args.forward_after_load:
            run_forward(wrapper, args, layer_idx)
    log(f"success direct_multi_wrapper wrappers={len(wrappers)}")


def single_wrapper_reload(args: argparse.Namespace) -> None:
    kt_kernel = import_kt_kernel(force_avx2_bf16_class=args.force_avx2_bf16_class)
    zombies = []
    first_layer = args.layers[0]
    wrapper = make_wrapper(kt_kernel, args, first_layer)
    log(f"created single wrapper type={type(wrapper)}")
    load_layer(wrapper, args, first_layer)
    for layer_idx in args.layers[1:]:
        if args.zombie_old_moe and getattr(wrapper, "moe", None) is not None:
            zombies.append(wrapper.moe)
            wrapper.moe = None
            log(f"zombied previous moe count={len(zombies)}")
        wrapper.layer_idx = layer_idx
        load_layer(wrapper, args, layer_idx)
        if args.forward_after_load:
            run_forward(wrapper, args, layer_idx)
    log(f"success single_wrapper_reload zombies={len(zombies)}")


def nano_backend_forward(args: argparse.Namespace) -> None:
    from nanovllm.layers.fuse_moe.kt_backend import KtKernelCpuMoeBackend

    backends = []
    for layer_idx in args.layers:
        backend = KtKernelCpuMoeBackend(
            layer_idx=layer_idx,
            cpu_expert_pool=None,
            max_routes=args.batch_tokens * args.top_k,
            moe_intermediate_size=args.moe_intermediate_size,
            hidden_size=args.hidden_size,
            num_experts=args.num_experts,
            num_experts_per_tok=args.top_k,
            gpu_expert_mask=torch.zeros(args.num_experts, dtype=torch.bool, device="cpu"),
            weight_path=args.weight_path,
            kt_method=args.method,
            kt_num_threads=args.cpuinfer_threads,
            kt_threadpool_count=args.threadpool_count,
            kt_chunked_prefill_size=args.chunked_prefill_size,
        )
        backends.append(backend)
        hidden = torch.randn(args.batch_tokens, args.hidden_size, device="cuda", dtype=torch.bfloat16)
        selected = torch.randint(0, args.num_experts, (args.batch_tokens, args.top_k), device="cuda")
        weights = torch.rand(args.batch_tokens, args.top_k, device="cuda", dtype=torch.float32)
        weights = (weights / weights.sum(dim=-1, keepdim=True)).to(torch.bfloat16)
        log(f"nano backend forward layer={layer_idx}")
        result = backend.forward(
            hidden_states=hidden,
            flat_weights=weights.reshape(-1),
            top_k=args.top_k,
            cpu_indices=torch.arange(args.batch_tokens * args.top_k, device="cuda", dtype=torch.int64),
            cpu_task_expert_ids=torch.empty(0, device="cuda", dtype=torch.int64),
            cpu_task_offsets=torch.empty(0, device="cuda", dtype=torch.int64),
            act_fn=lambda x: x,
            selected_experts=selected,
            routing_weights=weights,
        )
        torch.cuda.synchronize()
        log(
            f"nano backend forward done layer={layer_idx} "
            f"shape={tuple(result.outputs_cpu.shape)} dtype={result.outputs_cpu.dtype}"
        )
    log(f"success nano_backend_forward backends={len(backends)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["direct_multi_wrapper", "single_wrapper_reload", "nano_backend_forward"], required=True)
    parser.add_argument("--weight-path", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--moe-intermediate-size", type=int, default=768)
    parser.add_argument("--cpuinfer-threads", type=int, default=16)
    parser.add_argument("--threadpool-count", type=int, default=1)
    parser.add_argument("--chunked-prefill-size", type=int, default=4096)
    parser.add_argument("--method", default="BF16")
    parser.add_argument("--batch-tokens", type=int, default=1)
    parser.add_argument("--force-avx2-bf16-class", action="store_true")
    parser.add_argument("--zombie-old-moe", action="store_true")
    parser.add_argument("--forward-after-load", action="store_true")
    args = parser.parse_args()

    import faulthandler

    faulthandler.enable(all_threads=True)
    log(f"python={sys.executable}")
    log(f"platform={platform.platform()}")
    log(f"hostname={platform.node()}")
    log(f"pid={os.getpid()}")
    log(f"cpu_flags={cpu_flags_summary()}")
    log(f"KT_KERNEL_CPU_VARIANT={os.environ.get('KT_KERNEL_CPU_VARIANT')}")
    log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    log(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"cuda_device={torch.cuda.get_device_name(0)}")

    if args.mode == "direct_multi_wrapper":
        direct_multi_wrapper(args)
    elif args.mode == "single_wrapper_reload":
        single_wrapper_reload(args)
    elif args.mode == "nano_backend_forward":
        nano_backend_forward(args)
    else:
        raise AssertionError(args.mode)


if __name__ == "__main__":
    main()
