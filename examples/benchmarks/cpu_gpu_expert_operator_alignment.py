import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from nanovllm.layers.fuse_moe.heterogeneous import (
    _run_legacy_gpu_fallback,
    _run_real_cpu_expert_execution,
)


def _act_fn(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return F.silu(a) * b


def _run_one_case(
    batch_size: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: torch.dtype,
    weight_scale: float,
    seed: int,
) -> dict[str, float | int | str]:
    torch.manual_seed(seed)
    device = torch.device("cuda")
    expert_id = 3

    hidden_states = (torch.randn(batch_size, hidden_size, device=device, dtype=dtype) * 0.1).contiguous()
    selected_experts = torch.full((batch_size, 1), expert_id, device=device, dtype=torch.int64)
    routing_weights = torch.rand(batch_size, 1, device=device, dtype=dtype)

    flat_selected = selected_experts.reshape(-1)
    flat_weights = routing_weights.reshape(-1)
    route_indices = torch.arange(batch_size, device=device, dtype=torch.int64)
    cpu_task_expert_ids = torch.tensor([expert_id], device=device, dtype=torch.int64)
    cpu_task_offsets = torch.tensor([0, batch_size], device=device, dtype=torch.int64)

    cpu_pool = {
        expert_id: {
            "gate_up": torch.randn(2 * intermediate_size, hidden_size, dtype=dtype) * weight_scale,
            "down": torch.randn(hidden_size, intermediate_size, dtype=dtype) * weight_scale,
        }
    }

    out_cpu = torch.zeros_like(hidden_states)
    out_gpu = torch.zeros_like(hidden_states)

    _run_real_cpu_expert_execution(
        hidden_states=hidden_states,
        output=out_cpu,
        flat_weights=flat_weights,
        top_k=1,
        cpu_indices=route_indices,
        cpu_task_expert_ids=cpu_task_expert_ids,
        cpu_task_offsets=cpu_task_offsets,
        flat_selected_original=flat_selected,
        cpu_expert_pool=cpu_pool,
        act_fn=_act_fn,
    )
    _run_legacy_gpu_fallback(
        hidden_states=hidden_states,
        output=out_gpu,
        flat_weights=flat_weights,
        top_k=1,
        cpu_indices=route_indices,
        flat_selected_original=flat_selected,
        cpu_expert_pool=cpu_pool,
        act_fn=_act_fn,
    )

    diff = (out_cpu - out_gpu).float()
    max_abs = float(diff.abs().max().item())
    mean_abs = float(diff.abs().mean().item())
    ref_norm = float(out_gpu.float().norm().item())
    rel_l2 = float(diff.norm().item() / (ref_norm + 1e-12))

    return {
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "dtype": str(dtype).replace("torch.", ""),
        "weight_scale": weight_scale,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rel_l2": rel_l2,
    }


def _parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU/GPU expert operator precision alignment benchmark")
    parser.add_argument("--batch-sizes", type=str, default="64,128")
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--intermediate-size", type=int, default=1024)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--weight-scale", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rel-l2", type=float, default=5e-4)
    parser.add_argument("--max-mean-abs", type=float, default=1e-5)
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CPU/GPU expert operator alignment benchmark.")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    rows: list[dict[str, float | int | str]] = []
    for i, batch_size in enumerate(_parse_int_list(args.batch_sizes)):
        rows.append(
            _run_one_case(
                batch_size=batch_size,
                hidden_size=args.hidden_size,
                intermediate_size=args.intermediate_size,
                dtype=dtype,
                weight_scale=args.weight_scale,
                seed=args.seed + i,
            )
        )

    worst_rel_l2 = max(float(r["rel_l2"]) for r in rows)
    worst_mean_abs = max(float(r["mean_abs"]) for r in rows)
    passed = worst_rel_l2 <= args.max_rel_l2 and worst_mean_abs <= args.max_mean_abs

    report = {
        "config": {
            "batch_sizes": _parse_int_list(args.batch_sizes),
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "dtype": args.dtype,
            "weight_scale": args.weight_scale,
            "seed": args.seed,
            "max_rel_l2": args.max_rel_l2,
            "max_mean_abs": args.max_mean_abs,
        },
        "results": rows,
        "summary": {
            "worst_rel_l2": worst_rel_l2,
            "worst_mean_abs": worst_mean_abs,
            "passed": passed,
        },
    }

    text = json.dumps(report, ensure_ascii=True, indent=2)
    print(text)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")

    if not passed:
        raise SystemExit(
            f"alignment check failed: worst_rel_l2={worst_rel_l2}, worst_mean_abs={worst_mean_abs}"
        )


if __name__ == "__main__":
    main()
