from types import SimpleNamespace

import numpy as np
import pandas as pd

from patbot.path_value import _safe_cost, select_cross_position_challengers


def _sim():
    return SimpleNamespace(
        pos=np.array(["TE", "RB", "WR", "QB", "WR"], dtype=object),
        names=np.array(["Selected TE", "RB One", "WR One", "QB One", "WR Two"], dtype=object),
        vorp=np.array([40.0, 50.0, 45.0, 35.0, 30.0]),
        league_winner_score=np.array([60.0, 55.0, 80.0, 40.0, 50.0]),
        q90_points=np.array([250.0, 240.0, 270.0, 260.0, 220.0]),
        adp=np.array([50.0, 45.0, 52.0, 60.0, 70.0]),
        players=pd.DataFrame(
            {
                "fantasypros_api_ecr": [48.0, 42.0, 51.0, 39.0, np.nan],
            }
        ),
    )


def test_safe_cost_direction_for_high_is_good_metrics():
    assert _safe_cost(50.0, 42.0) == 8.0


def test_safe_cost_direction_for_low_is_good_ecr():
    assert _safe_cost(40.0, 55.0, low_is_good=True) == 15.0


def test_cross_position_challengers_exclude_selected_position_and_use_five_lenses():
    sim = _sim()
    scores = np.array([99.0, 90.0, 88.0, 86.0, 84.0])
    available = np.ones(5, dtype=bool)
    rows = select_cross_position_challengers(sim, scores, available, 0, top_pool=4)
    by_type = {row["Challenger Type"]: row["Player"] for row in rows}
    assert by_type["Score"] == "RB One"
    assert by_type["VORP"] == "RB One"
    assert by_type["ECR"] == "QB One"
    assert by_type["LWS"] == "WR One"
    assert by_type["Q90"] == "WR One"
    assert all(row["Pos"] != "TE" for row in rows)


def test_cross_position_challenger_pool_limits_metric_extremes():
    sim = _sim()
    scores = np.array([99.0, 90.0, 88.0, 86.0, 10.0])
    available = np.ones(5, dtype=bool)
    # WR Two is outside the top-three cross-position score pool and therefore
    # cannot become a metric challenger even if another metric were extreme.
    sim.league_winner_score[4] = 100.0
    rows = select_cross_position_challengers(sim, scores, available, 0, top_pool=3)
    assert "WR Two" not in {row["Player"] for row in rows}
