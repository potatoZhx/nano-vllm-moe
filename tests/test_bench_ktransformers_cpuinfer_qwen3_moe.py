from __future__ import annotations

import importlib.util
from pathlib import Path


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "bench_ktransformers_cpuinfer_qwen3_moe.py"
)
_SPEC = importlib.util.spec_from_file_location(_MODULE_PATH.stem, _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_make_cpuinfer = _MODULE._make_cpuinfer
_parse_numa_nodes = _MODULE._parse_numa_nodes
_random_expert_ids = _MODULE._random_expert_ids
_split_threads = _MODULE._split_threads


class _FakeWorkerPoolConfig:
    pass


class _FakeCPUInferExt:
    WorkerPoolConfig = _FakeWorkerPoolConfig

    @staticmethod
    def CPUInfer(config):
        return config


def test_split_threads_preserves_total() -> None:
    assert _split_threads(17, 2) == [9, 8]


def test_parse_numa_nodes_defaults_and_validates() -> None:
    assert _parse_numa_nodes("", 2) == [0, 1]
    assert _parse_numa_nodes("2,0", 2) == [2, 0]


def test_make_cpuinfer_builds_multi_numa_worker_config() -> None:
    config = _make_cpuinfer(_FakeCPUInferExt, 16, 2, "0,1")

    assert config.subpool_count == 2
    assert config.subpool_numa_map == [0, 1]
    assert config.subpool_thread_count == [8, 8]


def test_random_expert_ids_applies_exact_route_fraction() -> None:
    expert_ids = _random_expert_ids(
        pool_size=4,
        qlen=3,
        topk=8,
        expert_num=128,
        cpu_route_fraction=0.5,
        seed=7,
    )

    assert expert_ids.shape == (4, 3, 8)
    assert ((expert_ids >= 0).sum(dim=(1, 2)) == 12).all()
