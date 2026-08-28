from types import SimpleNamespace

import numpy as np
import pandas as pd

from patbot.sequence_audit import (
    _aggregate_sequence_rows,
    _best_legal_excluding_position,
    _best_legal_position,
    select_rbwr_challengers,
)


def _sim():
    positions = np.array(["RB", "WR", "RB", "WR", "QB", "TE"])
    players = pd.DataFrame(
        {
            "fantasypros_api_ecr": [20, 30, 40, 50, 10, 60],
        }
    )
    return SimpleNamespace(
        pos=positions,
        names=np.array(["RB Score", "WR LWS", "RB Q90", "WR Other", "QB", "TE"]),
        vorp=np.array([50, 45, 44, 40, 60, 30], dtype=float),
        league_winner_score=np.array([40, 90, 70, 60, 20, 30], dtype=float),
        q90_points=np.array([250, 240, 310, 230, 400, 200], dtype=float),
        adp=np.array([20, 30, 40, 50, 10, 60], dtype=float),
        players=players,
    )


def test_challengers_use_three_independent_selection_metrics():
    sim = _sim()
    scores = np.array([90, 85, 80, 75, 95, 70], dtype=float)
    available = np.ones(6, dtype=bool)
    rows = select_rbwr_challengers(sim, scores, available, top_pool=4)
    picked = {row["Challenger Type"]: row["Player"] for row in rows}
    assert picked == {
        "Score": "RB Score",
        "LWS": "WR LWS",
        "Q90": "RB Q90",
    }


def test_challenger_pool_prevents_far_down_board_ceiling_pick():
    sim = _sim()
    scores = np.array([90, 85, 80, 10, 95, 70], dtype=float)
    sim.league_winner_score[3] = 100.0
    available = np.ones(6, dtype=bool)
    rows = select_rbwr_challengers(sim, scores, available, top_pool=3)
    lws = next(row for row in rows if row["Challenger Type"] == "LWS")
    assert lws["Player"] == "WR LWS"


def test_best_legal_position_respects_availability_and_score():
    sim = _sim()
    scores = np.array([90, 85, 80, 75, 95, 70], dtype=float)
    available = np.ones(6, dtype=bool)
    assert _best_legal_position(sim, scores, available, "QB") == 4
    available[4] = False
    assert _best_legal_position(sim, scores, available, "QB") is None


def test_best_legal_excluding_position_can_delay_fill():
    sim = _sim()
    scores = np.array([90, 85, 80, 75, 95, 70], dtype=float)
    available = np.ones(6, dtype=bool)
    idx = _best_legal_excluding_position(sim, scores, available, "QB")
    assert idx == 0


def test_aggregate_marks_positive_alt_delta_as_alt_win():
    rows = pd.DataFrame(
        [
            {
                "Pos": "TE",
                "Quality": "solid",
                "Challenger Type": "LWS",
                "Wait Turns": 2,
                "Selected Pos Rank": 4,
                "Challenger Score Gap": 2.0,
                "Challenger LWS": 65.0,
                "Challenger Q90": 260.0,
                "VORP Cost of Waiting": 5.0,
                "Alt Delta vs Fill Now": 10.0,
                "Selected": "TE A",
                "Challenger": "RB A",
                "Wait Fill": "TE B",
                "Selected FP ECR": 50.0,
                "Challenger FP ECR": 40.0,
            },
            {
                "Pos": "TE",
                "Quality": "solid",
                "Challenger Type": "LWS",
                "Wait Turns": 2,
                "Selected Pos Rank": 4,
                "Challenger Score Gap": 3.0,
                "Challenger LWS": 63.0,
                "Challenger Q90": 255.0,
                "VORP Cost of Waiting": 7.0,
                "Alt Delta vs Fill Now": -2.0,
                "Selected": "TE A",
                "Challenger": "RB A",
                "Wait Fill": "TE B",
                "Selected FP ECR": 50.0,
                "Challenger FP ECR": 40.0,
            },
        ]
    )
    summary, common = _aggregate_sequence_rows(rows, runs=2)
    assert float(summary.iloc[0]["Avg Alt Delta"]) == 4.0
    assert float(summary.iloc[0]["Alt Wins %"]) == 50.0
    assert int(common.iloc[0]["Times"]) == 2
