from scripts.analyze_tpot_stop_curves import (
    _boundary_selection,
    _first_increase,
    _lookahead_selection,
    _step_curves,
)


def test_step_curve_aligns_explicit_zero_and_draft_candidates():
    trace = {
        "verify_cost_predictions": [
            {"verify_cost_candidate_len": 0, "verify_cost_prediction_ms": 60.0},
            {"verify_cost_candidate_len": 1, "verify_cost_prediction_ms": 50.0},
            {"verify_cost_candidate_len": 2, "verify_cost_prediction_ms": 45.0},
        ],
        "draft_call_ms": [10.0, 12.0],
        "sequences": [{"predicted_alpha": [0.5, 0.25]}],
    }

    curve = _step_curves(trace)

    assert [row["draft_len"] for row in curve] == [0, 1, 2]
    assert curve[0]["expected_tpot_ms"] == 60.0
    assert curve[1]["expected_tpot_ms"] == 40.0
    assert curve[2]["expected_tpot_ms"] == 67.0 / 1.625
    assert _first_increase(curve, 0.0) == 2


def test_oracle_lookahead_stops_before_reactive_overshoot():
    trace = {
        "verify_cost_predictions": [
            {"verify_cost_candidate_len": 0, "verify_cost_prediction_ms": 80.0},
            {"verify_cost_candidate_len": 1, "verify_cost_prediction_ms": 80.0},
            {"verify_cost_candidate_len": 2, "verify_cost_prediction_ms": 80.0},
        ],
        "draft_call_ms": [10.0, 10.0],
        "sequences": [{"predicted_alpha": [0.9, 0.1]}],
    }
    curve = _step_curves(trace)

    assert _first_increase(curve, 0.0) == 2
    assert _lookahead_selection(curve, 0.0, "oracle_next") == 1
    # Alpha persistence projects another high-acceptance step, so it cannot see
    # the sharp alpha drop even with the future workload prediction.
    assert _lookahead_selection(curve, 0.0, "alpha_oracle_workload") == 2
    assert _lookahead_selection(curve, 0.0, "runtime_proxy") is None


def test_runtime_proxy_lookahead_uses_predraft_verify_prediction():
    trace = {
        "verify_cost_predictions": [
            {"verify_cost_candidate_len": 0, "verify_cost_prediction_ms": 80.0},
            {
                "verify_cost_candidate_len": 1,
                "verify_cost_prediction_ms": 80.0,
                "verify_cost_lookahead_prediction_ms": 200.0,
            },
            {"verify_cost_candidate_len": 2, "verify_cost_prediction_ms": 80.0},
        ],
        "draft_call_ms": [10.0, 10.0],
        "sequences": [{"predicted_alpha": [0.9, 0.9]}],
    }
    curve = _step_curves(trace)

    assert _lookahead_selection(curve, 0.0, "runtime_proxy") == 1


def test_boundary_selection_skips_intermediate_bucket_jump():
    trace = {
        "verify_cost_predictions": [
            {
                "verify_cost_candidate_len": draft_len,
                "verify_cost_prediction_ms": verify_ms,
            }
            for draft_len, verify_ms in [
                (0, 80.0),
                (1, 80.0),
                (2, 80.0),
                (3, 180.0),
                (4, 150.0),
                (5, 130.0),
                (6, 80.0),
                (7, 180.0),
                (8, 130.0),
                (9, 70.0),
                (10, 180.0),
                (11, 130.0),
                (12, 120.0),
            ]
        ],
        "draft_call_ms": [10.0] * 12,
        "sequences": [{"predicted_alpha": [0.95] * 12}],
    }
    curve = _step_curves(trace)

    selected = _boundary_selection(
        curve,
        boundaries=[6, 9, 12],
        min_steps=6,
        margin=0.0,
    )

    # A one-step decision would stop at K6 on the K7 bucket jump. Endpoint
    # comparison sees the amortized K9 point and continues.
    assert selected == 9
