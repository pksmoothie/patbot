from types import SimpleNamespace

import numpy as np
import pandas as pd

from patbot.late_round import (
    DEFAULT_AUDIT_ROUNDS,
    _component_frame,
    _hard_special_teams_constraints,
)


def test_default_audit_rounds_preserve_special_teams_endgame():
    assert DEFAULT_AUDIT_ROUNDS == (8, 10, 12, 13)
    assert max(DEFAULT_AUDIT_ROUNDS) < 14


def test_hard_special_teams_constraints_block_defense_and_kicker_early():
    sim = SimpleNamespace(
        cfg={
            "special_teams_strategy": {
                "draft": {
                    "defense_round": 14,
                    "kicker_round": 15,
                    "rostered_defenses": 1,
                    "rostered_kickers": 1,
                }
            }
        },
        pos=np.array(["WR", "RB", "DEF", "K"]),
        pos_to_code={"WR": 0, "RB": 1, "DEF": 2, "K": 3},
    )
    score = np.array([80.0, 79.0, 99.0, 100.0])
    roster_counts = np.zeros(4, dtype=int)
    constrained = _hard_special_teams_constraints(sim, score, roster_counts, round_no=13)
    assert constrained[0] == 80.0
    assert constrained[1] == 79.0
    assert constrained[2] < -1e8
    assert constrained[3] < -1e8


def test_component_frame_exposes_youth_share_without_making_youth_the_only_signal():
    sim = SimpleNamespace(
        n=2,
        strategy_metrics=pd.DataFrame({
            "q90_points": [260.0, 260.0],
            "market_edge_score": [0.6, 0.0],
            "early_career_score": [1.0, 1.0],
            "league_winner_score": [80.0, 60.0],
            "performance_sigma": [0.25, 0.25],
        }),
        pos=np.array(["WR", "WR"]),
        replacement=np.array([160.0, 160.0]),
        proj=np.array([210.0, 210.0]),
        cfg={
            "championship_strategy": {
                "league_winner_components": {
                    "positional_ceiling": 0.55,
                    "market_edge": 0.25,
                    "early_career": 0.20,
                }
            }
        },
    )
    frame = _component_frame(sim)
    assert frame.loc[0, "market_edge_score"] > frame.loc[1, "market_edge_score"]
    assert frame.loc[0, "youth_component_share"] < frame.loc[1, "youth_component_share"]
    assert frame.loc[0, "youth_component_share"] < 0.45
