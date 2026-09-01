from __future__ import annotations

import numpy as np
import pandas as pd

from patbot.config import load_config
from patbot.decision_strategy import build_final_call_plan


class CaseEngine:
    def __init__(self):
        self.config = load_config("config/league.yaml")
        self.league = {"teams": 12, "draft_slot": 3}
        self.roster_cfg = {
            "QB": 1,
            "RB": 2,
            "WR": 3,
            "TE": 1,
            "FLEX": 1,
            "K": 1,
            "DEF": 1,
            "flex_eligible": ["RB", "WR", "TE"],
        }
        roster = [
            {"player_id": "r1", "name": "RB1", "pos": "RB"},
            {"player_id": "r2", "name": "RB2", "pos": "RB"},
            {"player_id": "r3", "name": "RB3", "pos": "RB"},
            {"player_id": "t1", "name": "TE1", "pos": "TE"},
        ]
        candidates = [
            {"player_id": "burrow", "name": "Burrow", "pos": "QB"},
            {"player_id": "davante", "name": "Davante", "pos": "WR"},
            {"player_id": "rome", "name": "Rome", "pos": "WR"},
            {"player_id": "jamo", "name": "Jamo", "pos": "WR"},
            {"player_id": "bucky", "name": "Bucky", "pos": "RB"},
            {"player_id": "daniels", "name": "Daniels", "pos": "QB"},
        ]
        self.players = pd.DataFrame(roster + candidates)
        self.players["player_id"] = self.players["player_id"].astype(str)
        self.players["team"] = "X"
        self.players["proj_points"] = 100.0
        self.players["adp"] = np.arange(1, len(self.players) + 1, dtype=float)
        self.players["injury_risk"] = 0.0


def test_503_three_empty_wr_slots_beat_small_burrow_quality_edge():
    board = pd.DataFrame(
        [
            {"player_id": "burrow", "name": "Burrow", "pos": "QB", "score": 94.18, "decision_quality_score": 80.0, "scarcity": 16.26, "proj_points": 397.25, "adp": 54.0},
            {"player_id": "davante", "name": "Davante", "pos": "WR", "score": 93.98, "decision_quality_score": 79.8, "scarcity": 14.73, "proj_points": 204.46, "adp": 51.0},
            {"player_id": "bucky", "name": "Bucky", "pos": "RB", "score": 92.74, "decision_quality_score": 78.5, "scarcity": 28.60, "proj_points": 206.35, "adp": 48.0},
            {"player_id": "rome", "name": "Rome", "pos": "WR", "score": 91.77, "decision_quality_score": 79.0, "scarcity": 16.42, "proj_points": 207.13, "adp": 60.0},
            {"player_id": "jamo", "name": "Jamo", "pos": "WR", "score": 89.18, "decision_quality_score": 78.2, "scarcity": 6.69, "proj_points": 211.15, "adp": 58.0},
            {"player_id": "daniels", "name": "Daniels", "pos": "QB", "score": 85.36, "decision_quality_score": 75.0, "scarcity": 8.13, "proj_points": 384.93, "adp": 62.0},
        ]
    )
    engine = CaseEngine()
    plan = build_final_call_plan(
        board,
        engine,
        current_pick=51,
        my_roster_ids=["r1", "r2", "r3", "t1"],
        draft_history=[],
    )

    assert plan["priority_positions"][0] == "WR"
    assert plan["base_row"]["name"] == "Davante"
    assert "Burrow" in set(plan["shortlist"]["name"])
