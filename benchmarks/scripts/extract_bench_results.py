import argparse
import glob
import json
from pathlib import Path
import statistics
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def f3(v: float) -> str:
    return f"{float(v):.3f}"


def f4(v: float) -> str:
    return f"{float(v):.4f}"


def pct(v: float) -> int:
    return int(round(float(v) * 100.0))


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join(lines)


def first_mismatch(tokens_ref: list[list[int]], tokens_cur: list[list[int]]) -> tuple[int, tuple[int, int, int, int] | None]:
    mismatch = 0
    first = None
    for sidx, (a_seq, b_seq) in enumerate(zip(tokens_ref, tokens_cur)):
        for tidx, (a_tok, b_tok) in enumerate(zip(a_seq, b_seq)):
            if a_tok != b_tok:
                mismatch += 1
                if first is None:
                    first = (sidx, tidx, a_tok, b_tok)
    return mismatch, first


def pick(rows: list[dict[str, Any]], **cond: Any) -> dict[str, Any]:
    for r in rows:
        ok = True
        for k, v in cond.items():
            if r.get(k) != v:
                ok = False
                break
        if ok:
            return r
    raise KeyError(f"missing row with condition: {cond}")


def build_alignment_section(result_dir: Path) -> tuple[str, str]:
    f_std = result_dir / "cpu_alignment_standard_phase2_post_rerun_job15779_idlegpu3.json"
    f_ser = result_dir / "cpu_alignment_heter_serial_phase2_post_rerun_job15779_idlegpu3.json"
    f_par = result_dir / "cpu_alignment_heter_parallel_phase2_post_rerun_job15779_idlegpu3.json"
    d_std = load_json(f_std)
    d_ser = load_json(f_ser)
    d_par = load_json(f_par)

    rows = []
    for name, data in [
        ("standard", d_std),
        ("heter_serial", d_ser),
        ("heter_parallel", d_par),
    ]:
        profile = data.get("engine_profile", {})
        wait_ms = float(profile.get("model_cpu_wait_ms", 0.0)) + float(profile.get("model_gpu_wait_ms", 0.0))
        mismatch, _ = first_mismatch(d_std.get("generated_token_ids", []), data.get("generated_token_ids", []))
        rows.append(
            [
                name,
                str(int(data.get("cpu_exec_routes", 0))),
                f4(profile.get("model_cpu_route_ratio", 0.0)),
                f3(wait_ms),
                str(mismatch),
            ]
        )

    mismatch, first = first_mismatch(d_std.get("generated_token_ids", []), d_par.get("generated_token_ids", []))
    note = ""
    if mismatch > 0 and first is not None:
        note = (
            "`heter_parallel` 的首个差异位点："
            f"`seq={first[0]}, token_pos={first[1]}`，`{first[2]} -> {first[3]}`。"
        )

    table = md_table(
        [
            "Case",
            "cpu_exec_routes",
            "model_cpu_route_ratio",
            "wait_ms(model_cpu_wait+model_gpu_wait)",
            "与 standard token mismatch",
        ],
        rows,
    )
    return table, note


def build_single_layer_table(result_path: Path, num_tokens: int) -> str:
    data = load_json(result_path)
    rows = data.get("results", [])
    ratios = sorted({float(r["cpu_ratio"]) for r in rows if int(r.get("num_tokens", -1)) == num_tokens})
    table_rows = []
    for ratio in ratios:
        serial = pick(rows, num_tokens=num_tokens, cpu_ratio=ratio, parallel_enabled=False)
        parallel = pick(rows, num_tokens=num_tokens, cpu_ratio=ratio, parallel_enabled=True)
        serial_ms = float(serial.get("latency_ms_mean", 0.0))
        parallel_ms = float(parallel.get("latency_ms_mean", 0.0))
        speedup = serial_ms / parallel_ms if parallel_ms > 0 else 0.0
        table_rows.append(
            [
                str(pct(ratio)),
                f3(serial_ms),
                f3(parallel_ms),
                f4(speedup),
                f3(parallel.get("latency_breakdown_wait_ms", 0.0)),
                f4(parallel.get("cpu_route_ratio", 0.0)),
            ]
        )
    return md_table(
        [
            "cpu_ratio(%)",
            "serial_ms",
            "parallel_ms",
            "speedup(serial/parallel)",
            "parallel_wait_ms",
            "parallel_cpu_route_ratio",
        ],
        table_rows,
    )


def build_small_tokens_matrix(result_path: Path) -> str:
    data = load_json(result_path)
    rows = data.get("results", [])
    token_sizes = sorted({int(r["num_tokens"]) for r in rows})
    ratios = sorted({float(r["cpu_ratio"]) for r in rows})

    headers = ["token_size"] + [f"cpu{pct(r)}%" for r in ratios]
    matrix_rows: list[list[str]] = []

    for token in token_sizes:
        line = [str(token)]
        for ratio in ratios:
            serial = pick(rows, num_tokens=token, cpu_ratio=ratio, parallel_enabled=False)
            parallel = pick(rows, num_tokens=token, cpu_ratio=ratio, parallel_enabled=True)
            s = float(serial.get("latency_ms_mean", 0.0))
            p = float(parallel.get("latency_ms_mean", 0.0))
            speedup = s / p if p > 0 else 0.0
            line.append(f4(speedup))
        matrix_rows.append(line)
    return md_table(headers, matrix_rows)


def build_small_tokens_detailed_table(result_path: Path) -> str:
    data = load_json(result_path)
    rows = data.get("results", [])
    token_sizes = sorted({int(r["num_tokens"]) for r in rows})
    ratios = sorted({float(r["cpu_ratio"]) for r in rows})

    table_rows: list[list[str]] = []
    for token in token_sizes:
        for ratio in ratios:
            serial = pick(rows, num_tokens=token, cpu_ratio=ratio, parallel_enabled=False)
            parallel = pick(rows, num_tokens=token, cpu_ratio=ratio, parallel_enabled=True)
            s = float(serial.get("latency_ms_mean", 0.0))
            p = float(parallel.get("latency_ms_mean", 0.0))
            speedup = s / p if p > 0 else 0.0
            table_rows.append(
                [
                    str(token),
                    str(pct(ratio)),
                    f3(s),
                    f3(p),
                    f4(speedup),
                    f3(parallel.get("latency_breakdown_wait_ms", 0.0)),
                    f4(serial.get("cpu_route_ratio", 0.0)),
                    f4(parallel.get("cpu_route_ratio", 0.0)),
                ]
            )

    return md_table(
        [
            "token_size",
            "cpu_ratio(%)",
            "serial_ms",
            "parallel_ms",
            "speedup(serial/parallel)",
            "parallel_wait_ms",
            "serial_cpu_route_ratio",
            "parallel_cpu_route_ratio",
        ],
        table_rows,
    )


def build_spec_ratio_table(result_path: Path) -> str:
    data = load_json(result_path)
    rows = data.get("results", [])
    ratios = sorted({float(r["cpu_expert_set_ratio"]) for r in rows})

    table_rows = []
    for ratio in ratios:
        serial = pick(rows, cpu_expert_set_ratio=ratio, parallel_enabled=False)
        parallel = pick(rows, cpu_expert_set_ratio=ratio, parallel_enabled=True)
        s = float(serial.get("latency_ms_mean", 0.0))
        p = float(parallel.get("latency_ms_mean", 0.0))
        table_rows.append(
            [
                str(pct(ratio)),
                f3(s),
                f3(p),
                f4(s / p if p > 0 else 0.0),
                f3(parallel.get("latency_breakdown_wait_ms", 0.0)),
                f4(parallel.get("cpu_route_ratio", 0.0)),
            ]
        )
    return md_table(
        [
            "cpu_ratio(%)",
            "serial_ms",
            "parallel_ms",
            "speedup(serial/parallel)",
            "parallel_wait_ms",
            "parallel_cpu_route_ratio",
        ],
        table_rows,
    )


def build_spec_tail_table(result_dir: Path) -> str:
    files = [
        "spec_verify_cpu_ratio_bench_phase2_post_min_job15779_idlegpu3.json",
        "spec_verify_cpu_ratio_bench_phase2_post_min_rerun_job15779_idlegpu3.json",
        "spec_verify_cpu_ratio_bench_phase2_post_min_threshold0_job15779_idlegpu3.json",
        "spec_verify_cpu_ratio_bench_phase2_post_min_threshold0_rerun_job15779_idlegpu3.json",
    ]
    rows = []
    for fn in files:
        data = load_json(result_dir / fn)
        res = data.get("results", [])
        serial = pick(res, parallel_enabled=False)
        parallel = pick(res, parallel_enabled=True)
        ratio = float(parallel.get("cpu_expert_set_ratio", serial.get("cpu_expert_set_ratio", 0.0)))
        s = float(serial.get("latency_ms_mean", 0.0))
        p = float(parallel.get("latency_ms_mean", 0.0))
        label = fn.replace("spec_verify_cpu_ratio_bench_", "").replace("_job15779_idlegpu3.json", "")
        rows.append([label, str(pct(ratio)), f3(s), f3(p), f4(s / p if p > 0 else 0.0), f3(parallel.get("latency_breakdown_wait_ms", 0.0))])

    return md_table(
        ["file", "cpu_ratio(%)", "serial_ms", "parallel_ms", "speedup(serial/parallel)", "parallel_wait_ms"],
        rows,
    )


def build_real_model_table(path: Path) -> str:
    data = load_json(path)
    rows = data.get("results", [])
    ratios = sorted({float(r["cpu_expert_set_ratio"]) for r in rows})
    out = []
    for ratio in ratios:
        serial = pick(rows, cpu_expert_set_ratio=ratio, parallel_enabled=False)
        parallel = pick(rows, cpu_expert_set_ratio=ratio, parallel_enabled=True)
        s = float(serial.get("latency_ms_mean", 0.0))
        p = float(parallel.get("latency_ms_mean", 0.0))
        out.append(
            [
                str(pct(ratio)),
                f3(s),
                f3(p),
                f4(s / p if p > 0 else 0.0),
                f3(parallel.get("latency_breakdown_wait_ms", 0.0)),
                f4(parallel.get("cpu_route_ratio", 0.0)),
                f3(parallel.get("latency_breakdown_cpu_path_exec_ms", 0.0)),
            ]
        )

    return md_table(
        [
            "cpu_ratio(%)",
            "serial_ms",
            "parallel_ms",
            "speedup(serial/parallel)",
            "parallel_wait_ms",
            "parallel_cpu_route_ratio",
            "parallel_cpu_path_exec_ms",
        ],
        out,
    )


def build_real_model_smallreq_table(result_dir: Path) -> tuple[str, list[str]]:
    files = sorted(glob.glob(str(result_dir / "moe_real_model_cpu_gpu_parallel_bench_phase2_post_smallreq_*_job15932_*.json")))
    rows = []
    used = []
    for fp in files:
        path = Path(fp)
        data = load_json(path)
        cfg = data.get("config", {})
        res = data.get("results", [])
        serial = pick(res, parallel_enabled=False)
        parallel = pick(res, parallel_enabled=True)
        s = float(serial.get("latency_ms_mean", 0.0))
        p = float(parallel.get("latency_ms_mean", 0.0))
        rows.append(
            [
                path.name,
                str(int(cfg.get("num_seqs", 0))),
                str(int(cfg.get("input_len", 0))),
                str(int(cfg.get("output_len", 0))),
                f3(s),
                f3(p),
                f4(s / p if p > 0 else 0.0),
                f3(parallel.get("latency_breakdown_wait_ms", 0.0)),
            ]
        )
        used.append(path.name)

    if not rows:
        return "", []
    table = md_table(
        [
            "file",
            "num_seqs",
            "input_len",
            "output_len",
            "serial_ms",
            "parallel_ms",
            "speedup(serial/parallel)",
            "parallel_wait_ms",
        ],
        rows,
    )
    return table, used


def build_anomaly_stability_table(result_dir: Path) -> str:
    single_files = sorted(
        glob.glob(str(result_dir / "moe_single_layer_cpu_gpu_parallel_bench_phase2_post_rerun_job*_idlegpu*.json"))
    )
    small_files = sorted(
        glob.glob(
            str(
                result_dir
                / "moe_single_layer_cpu_gpu_parallel_bench_phase2_breakdown_small_tokens_1_3_5_10_20_rerun_job*_idlegpu*.json"
            )
        )
    )

    def collect(paths: list[str], num_tokens: int, cpu_ratio: float, parallel_enabled: bool) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        for fp in paths:
            data = load_json(Path(fp))
            rows = data.get("results", [])
            hit = None
            for r in rows:
                if (
                    int(r.get("num_tokens", -1)) == num_tokens
                    and float(r.get("cpu_ratio", -1.0)) == cpu_ratio
                    and bool(r.get("parallel_enabled", False)) == parallel_enabled
                ):
                    hit = r
                    break
            if hit is not None:
                out.append((Path(fp).name, float(hit.get("latency_ms_mean", 0.0))))
        return out

    points = [
        ("token64_cpu25_serial", single_files, 64, 0.25, False),
        ("token64_cpu25_parallel", single_files, 64, 0.25, True),
        ("token64_cpu50_serial", single_files, 64, 0.5, False),
        ("token64_cpu50_parallel", single_files, 64, 0.5, True),
        ("token1_cpu100_serial", small_files, 1, 1.0, False),
        ("token1_cpu100_parallel", small_files, 1, 1.0, True),
    ]

    rows: list[list[str]] = []
    for label, files, token, ratio, parallel in points:
        samples = collect(files, token, ratio, parallel)
        vals = [v for _, v in samples]
        if not vals:
            rows.append([label, "0", "-", "-", "-", "-"])
            continue
        max_idx = max(range(len(vals)), key=lambda i: vals[i])
        rows.append(
            [
                label,
                str(len(vals)),
                f3(min(vals)),
                f3(statistics.median(vals)),
                f3(max(vals)),
                samples[max_idx][0],
            ]
        )

    return md_table(["point", "samples", "min_ms", "median_ms", "max_ms", "max_from_file"], rows)


def build_source_coverage_table(result_dir: Path, extra_files: list[Path] | None = None) -> str:
    files = sorted(result_dir.glob("*.json"))
    rows: list[list[str]] = []
    for fp in files:
        data = load_json(fp)
        results = data.get("results", []) if isinstance(data, dict) else []
        curves = data.get("curves", []) if isinstance(data, dict) else []
        rows.append([fp.name, str(len(results)), str(len(curves))])

    if extra_files:
        for fp in extra_files:
            if not fp.exists():
                continue
            if any(r[0] == fp.name for r in rows):
                continue
            data = load_json(fp)
            results = data.get("results", []) if isinstance(data, dict) else []
            curves = data.get("curves", []) if isinstance(data, dict) else []
            rows.append([fp.name, str(len(results)), str(len(curves))])

    return md_table(["file", "results_rows", "curves_rows"], rows)


def replace_section_by_heading(doc_text: str, new_section_text: str, start_heading: str, next_heading: str) -> str:
    start = doc_text.find(start_heading)
    if start < 0:
        raise ValueError(f"missing heading: {start_heading}")
    end = doc_text.find(next_heading, start)
    if end < 0:
        raise ValueError(f"missing heading: {next_heading}")
    prefix = doc_text[:start].rstrip() + "\n\n"
    suffix = doc_text[end:].lstrip("\n")
    return prefix + new_section_text.rstrip() + "\n\n" + suffix


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebuild phase2_post tables directly from JSON results")
    p.add_argument(
        "--results-dir",
        type=Path,
        default=(Path(__file__).resolve().parents[1] / "results"),
    )
    p.add_argument(
        "--output-md",
        type=Path,
        default=(Path(__file__).resolve().parents[2] / "docs/summary/phase2_post_tables_generated.md"),
    )
    p.add_argument(
        "--main-doc",
        type=Path,
        default=(Path(__file__).resolve().parents[2] / "docs/summary/phase2_post_last.md"),
    )
    p.add_argument(
        "--sync-main-doc",
        action="store_true",
        help="Also replace section 4 in main doc with generated section.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result_dir: Path = args.results_dir

    single_layer_path = result_dir / "moe_single_layer_cpu_gpu_parallel_bench_phase2_post_rerun_job15779_idlegpu3.json"
    small_tokens_path = result_dir / "moe_single_layer_cpu_gpu_parallel_bench_phase2_breakdown_small_tokens_1_3_5_10_20_rerun_job15779_idlegpu3.json"
    spec_ratio_path = result_dir / "spec_verify_cpu_ratio_bench_phase2_post_min_job15779_idlegpu2.json"
    real_model_path = result_dir / "moe_real_model_cpu_gpu_parallel_bench_phase2_post_job15779_idlegpu3.json"

    align_table, align_note = build_alignment_section(result_dir)
    table_42 = build_single_layer_table(single_layer_path, num_tokens=64)
    table_43 = build_single_layer_table(single_layer_path, num_tokens=256)
    table_44 = build_small_tokens_matrix(small_tokens_path)
    table_44_detail = build_small_tokens_detailed_table(small_tokens_path)
    table_45 = build_spec_ratio_table(spec_ratio_path)
    table_46 = build_spec_tail_table(result_dir)
    table_47 = build_real_model_table(real_model_path)
    table_48, used_smallreq_files = build_real_model_smallreq_table(result_dir)
    table_49 = build_anomaly_stability_table(result_dir)
    table_410 = build_source_coverage_table(result_dir)

    parts = [
        "## 4. 全量结果表格（脚本自动生成）",
        "",
        "### 4.1 Alignment（正确性与 profile 观测）",
        "",
        align_table,
        "",
    ]
    if align_note:
        parts.extend([align_note, ""])

    parts.extend(
        [
            "### 4.2 单层 MoE（token=64，按 cpu ratio）",
            "",
            f"文件：`{single_layer_path.name}`",
            "",
            table_42,
            "",
            "### 4.3 单层 MoE（token=256，按 cpu ratio）",
            "",
            f"同文件：`{single_layer_path.name}`",
            "",
            table_43,
            "",
            "### 4.4 Small tokens（1/3/5/10/20）speedup 矩阵",
            "",
            f"文件：`{small_tokens_path.name}`",
            "",
            "speedup 定义为 `serial_latency / parallel_latency`，`>1` 表示并行更快。",
            "",
            table_44,
            "",
            "Small tokens 全量明细（逐 token_size x cpu_ratio）：",
            "",
            table_44_detail,
            "",
            "### 4.5 Spec verify（多 cpu ratio 对比，min 配置）",
            "",
            f"文件：`{spec_ratio_path.name}`",
            "",
            table_45,
            "",
            "### 4.6 Spec verify（idlegpu3 收尾四文件）",
            "",
            table_46,
            "",
            "### 4.7 真实模型 cpugpuparallel（历史基线）",
            "",
            f"文件：`{real_model_path.name}`",
            "",
            table_47,
            "",
        ]
    )

    if table_48:
        parts.extend(
            [
                "### 4.8 真实模型小请求补跑（job15932，带时间戳）",
                "",
                table_48,
                "",
                "涉及文件：",
            ]
        )
        parts.extend([f"- `{name}`" for name in used_smallreq_files])
        parts.append("")

    parts.extend(
        [
            "### 4.9 异常点跨文件稳定性复核（自动统计）",
            "",
            "说明：统计同类 rerun 文件中的 `latency_ms_mean`，用于判断异常点是否稳定复现。",
            "",
            table_49,
            "",
            "### 4.10 JSON 源文件覆盖统计（完整性校验）",
            "",
            "说明：列出 `benchmarks/results` 下参与整理的 JSON 文件及其 `results/curves` 行数。",
            "",
            table_410,
            "",
        ]
    )

    text = "\n".join(parts).rstrip() + "\n"
    args.output_md.write_text(text, encoding="utf-8")

    if args.sync_main_doc:
        main_text = args.main_doc.read_text(encoding="utf-8")
        updated = replace_section_by_heading(main_text, text, "## 4. ", "## 5. ")
        args.main_doc.write_text(updated, encoding="utf-8")

    print(text)


if __name__ == "__main__":
    main()
