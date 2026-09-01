from __future__ import annotations

import numpy as np
import pandas as pd

from patbot.config import load_config
from patbot.decision_strategy import (
    adjust_single_qb_board_scores,
    build_final_call_plan,
    decision_strategy_settings,
    expected_position_demand,
)
from patbot.draft import DraftEngine
from patbot.final_call import run_final_call
from patbot.sim import FastDraftSimulator


class PlanEngine:
    def __init__(self, board_players: pd.DataFrame, config: dict | None = None):
        self.config = config or load_config("config/league.yaml")
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
        self.players = board_players.copy()
        self.players["player_id"] = self.players["player_id"].astype(str)


def _roster_and_candidates():
    roster = [
        {"player_id": "r1", "name": "RB One", "pos": "RB"},
        {"player_id": "r2", "name": "RB Two", "pos": "RB"},
        {"player_id": "r3", "name": "RB Three", "pos": "RB"},
        {"player_id": "w1", "name": "WR One", "pos": "WR"},
        {"player_id": "t1", "name": "TE One", "pos": "TE"},
    ]
    candidates = [
        {"player_id": "rh", "name": "Rhamondre", "pos": "RB"},
        {"player_id": "jw", "name": "Warren", "pos": "RB"},
        {"player_id": "ct", "name": "Tate", "pos": "WR"},
        {"player_id": "cs", "name": "Sutton", "pos": "WR"},
        {"player_id": "dk", "name": "Metcalf", "pos": "WR"},
        {"player_id": "bt", "name": "Thomas", "pos": "WR"},
        {"player_id": "cw", "name": "Caleb", "pos": "QB"},
    ]
    players = pd.DataFrame(roster + candidates)
    players["team"] = "X"
    players["proj_points"] = 100.0
    players["adp"] = np.arange(1, len(players) + 1, dtype=float)
    players["injury_risk"] = 0.0
    return players


def _six_ten_board():
    return pd.DataFrame(
        [
            {"player_id": "rh", "name": "Rhamondre", "pos": "RB", "score": 86.55, "decision_quality_score": 72.0, "scarcity": 17.35, "proj_points": 177.75, "adp": 66.0},
            {"player_id": "jw", "name": "Warren", "pos": "RB", "score": 85.63, "decision_quality_score": 71.0, "scarcity": 14.08, "proj_points": 180.77, "adp": 69.0},
            {"player_id": "ct", "name": "Tate", "pos": "WR", "score": 83.01, "decision_quality_score": 73.0, "scarcity": 7.01, "proj_points": 177.78, "adp": 74.0},
            {"player_id": "cs", "name": "Sutton", "pos": "WR", "score": 82.84, "decision_quality_score": 74.0, "scarcity": 9.60, "proj_points": 184.04, "adp": 76.0},
            {"player_id": "dk", "name": "Metcalf", "pos": "WR", "score": 82.29, "decision_quality_score": 76.0, "scarcity": 10.34, "proj_points": 188.12, "adp": 80.0},
            {"player_id": "cw", "name": "Caleb", "pos": "QB", "score": 79.47, "decision_quality_score": 70.0, "scarcity": 12.71, "proj_points": 372.45, "adp": 78.0},
            {"player_id": "bt", "name": "Thomas", "pos": "WR", "score": 78.57, "decision_quality_score": 75.0, "scarcity": 5.95, "proj_points": 189.73, "adp": 82.0},
        ]
    )


def _make_pool():
    rows = []
    pid = 1
    for pos, count, top in [
        ("RB", 30, 300),
        ("WR", 40, 305),
        ("QB", 20, 400),
        ("TE", 20, 240),
        ("K", 12, 150),
        ("DEF", 12, 145),
    ]:
        for i in range(count):
            rows.append(
                {
                    "player_id": str(pid),
                    "name": f"{pos}{i + 1}",
                    "team": "X",
                    "pos": pos,
                    "adp": float(pid),
                    "proj_points": float(top - i),
                    "injury_risk": 0.0,
                    "expert_rank": float(pid),
                    "is_rookie": False,
                }
            )
            pid += 1
    return pd.DataFrame(rows)


def test_single_qb_survival_is_neutralized_when_choosing_qb1():
    cfg = load_config("config/league.yaml")
    board = pd.DataFrame(
        [
            {"player_id": "dart", "name": "Dart", "pos": "QB", "score": 90.0, "survive_next": 0.10, "roster_fit": 1.0, "scarcity_pct": 0.5, "expert_rank": 100.0},
            {"player_id": "nix", "name": "Nix", "pos": "QB", "score": 83.0, "survive_next": 0.80, "roster_fit": 1.0, "scarcity_pct": 0.5, "expert_rank": 100.0},
        ]
    )
    adjusted = adjust_single_qb_board_scores(
        board,
        cfg,
        roster_positions=["RB", "RB", "RB", "WR", "WR", "WR", "TE"],
        round_no=8,
    ).set_index("player_id")

    assert adjusted.loc["nix", "score"] > adjusted.loc["dart", "score"]
    assert adjusted.loc["nix", "decision_effective_urgency"] == adjusted.loc["dart", "decision_effective_urgency"]


def test_qb2_penalty_does_not_disappear_in_round_9():
    cfg = load_config("config/league.yaml")
    engine = DraftEngine(_make_pool(), cfg)
    sim = FastDraftSimulator(engine)
    counts = np.zeros(len(sim.POSITIONS), dtype=np.int16)
    counts[sim.pos_to_code["QB"]] = 1

    r9 = sim._base_roster_need_penalty(counts, 9)
    r12 = sim._base_roster_need_penalty(counts, 12)
    qb = sim.pos == "QB"

    assert np.allclose(r9[qb], 34.0)
    assert np.allclose(r12[qb], 24.0)


def test_qb3_is_even_more_strongly_discouraged():
    cfg = load_config("config/league.yaml")
    engine = DraftEngine(_make_pool(), cfg)
    sim = FastDraftSimulator(engine)
    counts = np.zeros(len(sim.POSITIONS), dtype=np.int16)
    counts[sim.pos_to_code["QB"]] = 2
    penalty = sim._base_roster_need_penalty(counts, 10)
    assert np.allclose(penalty[sim.pos == "QB"], 120.0)


def test_qb_pool_demand_counts_qb1_need_not_repeated_turn_picks():
    engine = PlanEngine(_roster_and_candidates())

    # At 8.10, only slots 1 and 2 pick before PatBot at 9.03. Neither has QB1;
    # their two picks apiece can create roughly two QB1 selections, not four.
    demand_810 = expected_position_demand(
        engine,
        pos="QB",
        current_pick=94,
        draft_history=[],
    )
    assert 2.0 <= demand_810 < 2.2

    # At 9.03, slots 1 and 2 do not pick before 10.10. Give every upcoming
    # manager QB1; only the tiny generic QB2 rate should remain.
    history = [
        {"owner_slot": slot, "pos": "QB", "player_id": f"q{slot}"}
        for slot in range(4, 13)
    ]
    demand_903 = expected_position_demand(
        engine,
        pos="QB",
        current_pick=99,
        draft_history=history,
    )
    assert demand_903 < 0.5


def test_six_ten_shortlist_gives_wr_multiple_seats_and_keeps_raw_value_exception():
    players = _roster_and_candidates()
    engine = PlanEngine(players)
    board = _six_ten_board()
    plan = build_final_call_plan(
        board,
        engine,
        current_pick=70,
        my_roster_ids=["r1", "r2", "r3", "w1", "t1"],
        draft_history=[],
    )

    assert plan["strategy_active"] is True
    assert plan["priority_positions"][0] == "WR"
    names = set(plan["shortlist"]["name"])
    assert {"Metcalf", "Thomas", "Sutton"}.issubset(names)
    assert "Rhamondre" in names
    assert plan["base_row"]["name"] == "Metcalf"


def test_modest_rb4_value_does_not_override_high_wr_pressure():
    players = _roster_and_candidates()
    engine = PlanEngine(players)
    board = _six_ten_board().copy()
    board.loc[board["name"].eq("Rhamondre"), "decision_quality_score"] = 80.0

    plan = build_final_call_plan(
        board,
        engine,
        current_pick=70,
        my_roster_ids=["r1", "r2", "r3", "w1", "t1"],
        draft_history=[],
    )
    assert plan["base_row"]["pos"] == "WR"


def test_gibbs_level_value_can_escape_position_priority():
    players = _roster_and_candidates()
    gibbs = pd.DataFrame(
        [{"player_id": "g", "name": "Gibbs", "pos": "RB", "team": "DET", "proj_points": 340.0, "adp": 1.0, "injury_risk": 0.0}]
    )
    engine = PlanEngine(pd.concat([players, gibbs], ignore_index=True))
    board = pd.concat(
        [
            pd.DataFrame(
                [{"player_id": "g", "name": "Gibbs", "pos": "RB", "score": 99.0, "decision_quality_score": 100.0, "scarcity": 30.0, "proj_points": 340.0, "adp": 1.0}]
            ),
            _six_ten_board(),
        ],
        ignore_index=True,
    )

    plan = build_final_call_plan(
        board,
        engine,
        current_pick=70,
        my_roster_ids=["r1", "r2", "r3", "w1", "t1"],
        draft_history=[],
    )
    assert plan["priority_positions"][0] == "WR"
    assert plan["base_row"]["name"] == "Gibbs"


def test_final_call_uses_strategic_prior_not_raw_board_leader():
    engine = PlanEngine(_roster_and_candidates())
    board = _six_ten_board()
    seen = {}

    def fake_compare(engine, **kwargs):
        seen["candidate_ids"] = list(kwargs["candidate_ids"])
        id_to_name = dict(zip(board["player_id"], board["name"]))
        names = [id_to_name[x] for x in kwargs["candidate_ids"]]
        summary = pd.DataFrame(
            [
                {"Candidate": name, "Avg Lineup Score": 500.0 - i}
                for i, name in enumerate(names)
            ]
        )
        # build_final_call_plan puts its strategic base first, so make the room
        # screen agree with that prior for this deterministic integration test.
        details = [
            {"candidate": name, "candidate_id": pid}
            for name, pid in zip(names, kwargs["candidate_ids"])
        ]
        return summary, details

    result = run_final_call(
        engine,
        current_pick=70,
        drafted_ids=set(),
        my_roster_ids=["r1", "r2", "r3", "w1", "t1"],
        board=board,
        draft_history=[],
        compare_fn=fake_compare,
    )

    assert result["raw_base_winner"] == "Rhamondre"
    assert result["base_winner"] == "Metcalf"
    assert result["recommendation"] == "Metcalf"
    assert result["position_priority"][0] == "WR"
    assert "dk" in seen["candidate_ids"]


def test_league_qb_calibration_records_last_years_13_total_qbs():
    settings = decision_strategy_settings(load_config("config/league.yaml"))
    assert settings["opponent_demand"]["historical_total_qbs_last_draft"] == 13
    assert settings["opponent_demand"]["historical_rounds"] == 15
