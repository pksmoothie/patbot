import pandas as pd

from patbot.draft import DraftEngine, snake_pick


def _engine():
    players = pd.DataFrame([
        {"player_id": "wr", "name": "Upside WR", "pos": "WR", "proj_points": 260.0, "adp": 120.0, "injury_risk": 0.0},
        {"player_id": "rb", "name": "Upside RB", "pos": "RB", "proj_points": 250.0, "adp": 125.0, "injury_risk": 0.0},
        {"player_id": "def", "name": "Streaming DEF", "pos": "DEF", "proj_points": 110.0, "adp": 170.0, "injury_risk": 0.0},
        {"player_id": "k", "name": "Last Round K", "pos": "K", "proj_points": 105.0, "adp": 175.0, "injury_risk": 0.0},
    ])
    cfg = {
        "league": {"teams": 12, "draft_slot": 3},
        "roster": {
            "QB": 1, "RB": 2, "WR": 3, "TE": 1, "K": 1, "DEF": 1,
            "FLEX": 1, "flex_eligible": ["RB", "WR", "TE"],
        },
        "draft_engine": {
            "replacement_rank": {"QB": 12, "RB": 30, "WR": 42, "TE": 14, "K": 12, "DEF": 12},
            "weights": {"vorp": 0.35, "projection": 0.25, "urgency": 0.20, "scarcity": 0.12, "roster_fit": 0.08},
            "injury_risk_penalty": 8.0,
            "min_round_k": 15,
            "min_round_def": 14,
            "bench_position_caps": {"K": 1, "DEF": 1},
        },
        "championship_strategy": {"enabled": False},
        "special_teams_strategy": {
            "draft": {
                "defense_round": 14,
                "kicker_round": 15,
                "rostered_defenses": 1,
                "rostered_kickers": 1,
            }
        },
    }
    return DraftEngine(players, cfg)


def test_patbot_forces_defense_in_round_14_if_missing():
    engine = _engine()
    board = engine.recommend(
        current_pick=snake_pick(14, 12, 3),
        drafted_ids=set(),
        roster_positions=["QB", "RB", "RB", "WR", "WR", "WR", "TE", "RB", "WR", "RB", "WR", "TE", "RB"],
        top_n=4,
    )
    assert board.iloc[0]["pos"] == "DEF"


def test_patbot_forces_kicker_in_round_15_if_missing():
    engine = _engine()
    board = engine.recommend(
        current_pick=snake_pick(15, 12, 3),
        drafted_ids=set(),
        roster_positions=["QB", "RB", "RB", "WR", "WR", "WR", "TE", "RB", "WR", "RB", "WR", "TE", "RB", "DEF"],
        top_n=4,
    )
    assert board.iloc[0]["pos"] == "K"
