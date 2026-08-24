import numpy as np
import pandas as pd

from patbot.config import load_config
from patbot.draft import DraftEngine
from patbot.sim import FastDraftSimulator


def make_pool():
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
            rows.append({
                "player_id": str(pid),
                "name": f"{pos}{i + 1}",
                "team": "X",
                "pos": pos,
                "adp": float(pid),
                "proj_points": float(top - i),
                "injury_risk": 0.0,
                "expert_rank": float(pid),
                "is_rookie": i % 7 == 0,
            })
            pid += 1
    return pd.DataFrame(rows)


def test_real_manager_slots_are_locked():
    cfg = load_config("config/league.yaml")
    engine = DraftEngine(make_pool(), cfg)
    sim = FastDraftSimulator(engine)

    assignments = sim._archetype_assignments(np.random.default_rng(123))

    assert assignments == {
        1: "casual",
        2: "market",
        4: "casual",
        5: "market",
        6: "league_aware",
        7: "sharp",
        8: "casual",
        9: "market",
        10: "league_aware",
        11: "extremely_sharp",
        12: "casual",
    }


def test_manager_specific_overrides_are_applied():
    cfg = load_config("config/league.yaml")
    engine = DraftEngine(make_pool(), cfg)
    sim = FastDraftSimulator(engine)

    faherty = sim._manager_profile(2, "market")
    paul = sim._manager_profile(1, "casual")
    matt = sim._manager_profile(9, "market")
    george = sim._manager_profile(10, "league_aware")

    assert faherty["rookie_rank_bonus"] == 12.0
    assert paul["randomness"] == 1.40
    assert matt["custom_weight"] == 0.32
    assert george["custom_weight"] == 0.57


def test_room_names_match_real_draft_order():
    cfg = load_config("config/league.yaml")
    order = cfg["league"]["draft_order"]

    assert order == {
        1: "Paul",
        2: "Faherty",
        3: "PatBot",
        4: "James",
        5: "Potter",
        6: "Sarlo",
        7: "Dave",
        8: "Frank",
        9: "Matt",
        10: "George",
        11: "Lou",
        12: "Vinnie",
    }
