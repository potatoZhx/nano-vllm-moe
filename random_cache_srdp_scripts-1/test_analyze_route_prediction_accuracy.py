"""
 python analyze_route_prediction_accuracy.py \
    --input /data2/group_谈海生/mumura/dynamick/predictor/random_cache_runs/mtbench_random_cache_lru_ratio0.25_topc0.7/acceptance_summary_20260615_190647.jsonl \
    --top-p 4
/data2/group_谈海生/mumura/dynamick/predictor/random_cache_runs/mtbench_random_cache_lru_ratio0.25_topc0.7/acceptance_summary_20260615_190647.jsonl
"""



import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).with_name("analyze_route_prediction_accuracy.py")
SPEC = importlib.util.spec_from_file_location(
    "analyze_route_prediction_accuracy", SCRIPT_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def layer(layer_idx, ids, weights):
    return {
        "layer_idx": layer_idx,
        "original_ids": [ids],
        "original_weights": [weights],
    }


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_full_target_route_statistics(tmp_path):
    input_path = tmp_path / "acceptance_summary.jsonl"
    write_jsonl(
        input_path,
        [
            {
                "steps": [
                    {
                        "router": [
                            layer(1, [1, 2, 3], [0.6, 0.3, 0.1]),
                            layer(0, [4, 5], [0.8, 0.2]),
                        ],
                        "target_router": [
                            layer(0, [4, 7], [0.9, 0.1]),
                            layer(1, [2, 3, 8], [0.5, 0.3, 0.2]),
                        ],
                    },
                    {
                        "router": [
                            layer(0, [8, 9], [0.7, 0.3]),
                            layer(1, [1, 9], [0.6, 0.4]),
                        ],
                        "target_router": [
                            layer(0, [4, 7], [0.9, 0.1]),
                            layer(1, [1, 2], [0.5, 0.5]),
                        ],
                    },
                ]
            }
        ],
    )

    result = MODULE.analyze_file(input_path, top_p=None)

    assert result["record_count"] == 1
    assert result["step_count"] == 2
    assert result["observation_count"] == 4
    assert result["layers"] == [
        {
            "layer_idx": 0,
            "count": 2,
            "mean": 0.25,
            "variance": 0.0625,
            "max": 0.5,
            "min": 0.0,
        },
        {
            "layer_idx": 1,
            "count": 2,
            "mean": pytest.approx(7 / 12),
            "variance": pytest.approx(1 / 144),
            "max": pytest.approx(2 / 3),
            "min": 0.5,
        },
    ]


def test_top_p_uses_target_weights_and_complete_draft_route(tmp_path):
    input_path = tmp_path / "acceptance_summary.jsonl"
    write_jsonl(
        input_path,
        [
            {
                "steps": [
                    {
                        "router": [layer(0, [10, 11, 12], [0.4, 0.3, 0.3])],
                        "target_router": [
                            layer(0, [12, 10, 99], [0.9, 0.8, 0.1])
                        ],
                    }
                ]
            }
        ],
    )

    result = MODULE.analyze_file(input_path, top_p=2)

    assert result["layers"][0]["mean"] == 1.0
    assert result["top_p"] == 2


def test_rejects_layer_mismatch(tmp_path):
    input_path = tmp_path / "acceptance_summary.jsonl"
    write_jsonl(
        input_path,
        [
            {
                "steps": [
                    {
                        "router": [layer(0, [1], [1.0])],
                        "target_router": [layer(1, [1], [1.0])],
                    }
                ]
            }
        ],
    )

    with pytest.raises(ValueError, match="layer mismatch"):
        MODULE.analyze_file(input_path, top_p=None)
