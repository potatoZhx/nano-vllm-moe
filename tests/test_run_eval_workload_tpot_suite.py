import csv
import json
from types import SimpleNamespace

from scripts import run_eval_workload_tpot_suite as suite


def option_value(command, option):
    return command[command.index(option) + 1]


def write_success_results(result_dir, dataset, count):
    result_dir.mkdir(parents=True, exist_ok=True)
    with (result_dir / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "sample_count", "ok_count"],
        )
        writer.writeheader()
        writer.writerow(
            {"dataset": dataset, "sample_count": count, "ok_count": count}
        )
    with (result_dir / "rows.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "status",
                "dataset",
                "generated_output_tokens",
                "output_validation_error",
            ],
        )
        writer.writeheader()
        for _ in range(count):
            writer.writerow(
                {
                    "status": "ok",
                    "dataset": dataset,
                    "generated_output_tokens": 512,
                    "output_validation_error": "",
                }
            )


def test_command_enforces_requested_gpu_cpu_and_threads(tmp_path):
    args = suite.build_parser().parse_args([])
    assert args.mmlu_split == "validation"
    command = suite.benchmark_command(
        args,
        "mmlu_pro",
        tmp_path / "mmlu_pro",
        mmlu_pro_path=tmp_path / "mmlu.jsonl",
    )

    assert command[:3] == ["taskset", "--cpu-list", "64-96"]
    assert command[command.index("--kt-num-threads") + 1] == "16"
    assert command[command.index("--mmlu-pro-path") + 1].endswith("mmlu.jsonl")


def test_k6_vpb4_command_contains_fixed_historical_controls(tmp_path):
    args = suite.build_parser().parse_args(
        [
            "--suite-config",
            "k6_vpb4",
            "--cuda-visible-devices",
            "7",
            "--cpu-list",
            "0-3",
            "--num-samples",
            "1",
            "--mmlu-split",
            "test",
        ]
    )
    command = suite.benchmark_command(
        args,
        "mmlu_pro",
        tmp_path / "mmlu_pro",
        mmlu_pro_path=tmp_path / "mmlu_pro_validation.jsonl",
    )

    assert command[:3] == ["taskset", "--cpu-list", "64-96"]
    expected_options = {
        "--optimized-config": "k6_decode",
        "--request-mode": "dataset",
        "--num-samples": "all",
        "--cache-ratios": "0.3125",
        "--output-lens": "512",
        "--max-draft-tokens-values": "6",
        "--segment-sizes": "12",
        "--verify-prefetch-max-per-boundary": "4",
        "--verify-cuda-graph-bucket-steps": "3,5,7,10,13",
        "--draft-stop-policy": "none",
        "--acceptance-predictor-enabled": "false",
        "--gpu-memory-utilization": "0.99",
        "--temperature": "0.8",
        "--acceptance-strategy": "standard_sampling",
        "--kt-num-threads": "16",
        "--collect-profile": "false",
        "--engine-profile": "false",
        "--verify-cost-model-profile": "false",
        "--save-profile-json": "false",
        "--reset-profile-after-warmup": "false",
        "--reset-profile-before-request": "false",
        "--save-token-ids": "true",
        "--save-text": "true",
        "--repeats": "1",
        "--repeat-index-offset": "0",
        "--seed": "20260719",
        "--dist-port-base": "37970",
    }
    for option, value in expected_options.items():
        assert option_value(command, option) == value
    assert option_value(command, "--mmlu-pro-path").endswith(
        "mmlu_pro_validation.jsonl"
    )
    assert "--draft-tpot-stop-rule" not in command
    assert "--draft-tpot-verify-model-mode" not in command
    assert "--draft-tpot-verify-model-path" not in command


def test_k6_vpb4_forces_gpu_2_in_subprocess_environment(tmp_path, monkeypatch):
    args = suite.build_parser().parse_args(
        [
            "--suite-config",
            "k6_vpb4",
            "--datasets",
            "mt_bench",
            "--output-dir",
            str(tmp_path),
            "--cuda-visible-devices",
            "7",
        ]
    )
    environments = []

    def fake_run_command(command, log_path, env):
        environments.append(env.copy())
        write_success_results(
            tmp_path / "datasets" / "mt_bench", "mt_bench", 80
        )
        return 0

    monkeypatch.setattr(suite, "run_command", fake_run_command)

    assert suite.run(args) == 0
    assert environments[0]["CUDA_VISIBLE_DEVICES"] == "2"
    status = json.loads((tmp_path / "suite_status.json").read_text())
    assert status["suite_config"] == "k6_vpb4"
    assert status["cpu_list"] == "64-96"
    assert status["runs"][0]["validation"]["status"] == "passed"


def test_k6_vpb4_defaults_to_70_row_validation_jsonl(tmp_path, monkeypatch):
    source = tmp_path / "validation-00000-of-00001.parquet"
    source.write_bytes(b"placeholder")
    converter = tmp_path / "python"
    converter.write_text("")
    args = suite.build_parser().parse_args(
        [
            "--suite-config",
            "k6_vpb4",
            "--mmlu-split",
            "test",
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    monkeypatch.setitem(
        suite.MMLU_PRO_PARQUET_BY_SPLIT, "validation", source
    )
    monkeypatch.setattr(suite, "converter_candidates", lambda _args: [converter])
    monkeypatch.setattr(suite, "python_has_pyarrow", lambda *_args: True)

    def fake_subprocess_run(command, **kwargs):
        destination = suite.Path(command[-1])
        destination.write_text(
            "".join(json.dumps({"question": str(i)}) + "\n" for i in range(70))
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(suite.subprocess, "run", fake_subprocess_run)

    path, record = suite.ensure_mmlu_jsonl(args, tmp_path / "output")

    assert suite.mmlu_split_for_args(args) == "validation"
    assert path.name == "mmlu_pro_validation.jsonl"
    assert record["split"] == "validation"
    assert record["row_count"] == 70


def test_k6_result_validation_requires_count_and_512_tokens(tmp_path):
    write_success_results(tmp_path, "mmlu_pro", 70)
    passed = suite.validate_k6_vpb4_results(tmp_path, "mmlu_pro")
    assert passed["status"] == "passed"
    assert passed["first_output"]["generated_output_tokens"] == "512"

    rows = list(csv.DictReader((tmp_path / "rows.csv").open(newline="")))
    rows[0]["generated_output_tokens"] = "511"
    rows[1]["output_validation_error"] = "bad output"
    with (tmp_path / "rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    failed = suite.validate_k6_vpb4_results(tmp_path, "mmlu_pro")
    assert failed["status"] == "failed"
    assert failed["all_ok_outputs_are_512_tokens"] is False
    assert failed["output_validation_error_count"] == 1


def test_failed_dataset_does_not_stop_the_next_dataset(tmp_path, monkeypatch):
    args = suite.build_parser().parse_args(
        [
            "--datasets",
            "mt_bench,humaneval",
            "--output-dir",
            str(tmp_path),
        ]
    )
    return_codes = iter([9, 0])
    calls = []

    def fake_run_command(command, log_path, env):
        calls.append(command[command.index("--dataset") + 1])
        return_code = next(return_codes)
        if return_code == 0:
            result_dir = tmp_path / "datasets" / "humaneval"
            result_dir.mkdir(parents=True)
            with (result_dir / "summary.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["dataset", "tpot_ms_mean"],
                )
                writer.writeheader()
                writer.writerow(
                    {"dataset": "humaneval", "tpot_ms_mean": "12.5"}
                )
        return return_code

    monkeypatch.setattr(suite, "run_command", fake_run_command)

    assert suite.run(args) == 1
    assert calls == ["mt_bench", "humaneval"]
    status = json.loads((tmp_path / "suite_status.json").read_text())
    assert [run["status"] for run in status["runs"]] == [
        "failed",
        "completed",
    ]
    with (tmp_path / "tpot_summary.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [row["suite_status"] for row in rows] == ["failed", "completed"]
    assert rows[1]["tpot_ms_mean"] == "12.5"
