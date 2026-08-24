import pandas as pd
import numpy as np

from patbot.config import load_config
from patbot.draft import DraftEngine
from patbot.draft_state import (
    drafted_ids_from_history,
    make_pick_record,
    roster_ids_for_slot,
    roster_summary,
    team_slot_for_pick,
)
from patbot.sim import FastDraftSimulator


def make_pool():
    return pd.DataFrame([
        {
            "player_id": "1", "name": "QB One", "team": "AAA", "pos": "QB",
            "adp": 10.0, "proj_points": 350.0, "injury_risk": 0.0,
            "expert_rank": 10.0,
        },
        {
            "player_id": "2", "name": "RB One", "team": "BBB", "pos": "RB",
            "adp": 2.0, "proj_points": 300.0, "injury_risk": 0.0,
            "expert_rank": 2.0,
        },
        {
            "player_id": "3", "name": "WR One", "team": "CCC", "pos": "WR",
            "adp": 3.0, "proj_points": 295.0, "injury_risk": 0.0,
            "expert_rank": 3.0,
        },
        {
            "player_id": "4", "name": "TE One", "team": "DDD", "pos": "TE",
            "adp": 20.0, "proj_points": 240.0, "injury_risk": 0.0,
            "expert_rank": 20.0,
        },
    ])


def test_snake_owner_mapping():
    assert team_slot_for_pick(1, 12) == 1
    assert team_slot_for_pick(3, 12) == 3
    assert team_slot_for_pick(12, 12) == 12
    assert team_slot_for_pick(13, 12) == 12
    assert team_slot_for_pick(14, 12) == 11
    assert team_slot_for_pick(22, 12) == 3
    assert team_slot_for_pick(24, 12) == 1
    assert team_slot_for_pick(25, 12) == 1
    assert team_slot_for_pick(27, 12) == 3


def test_history_builds_specific_team_rosters():
    h = [
        make_pick_record(1, 12, "2", "RB One", "BBB", "RB"),
        make_pick_record(2, 12, "1", "QB One", "AAA", "QB"),
        make_pick_record(3, 12, "3", "WR One", "CCC", "WR"),
        make_pick_record(23, 12, "4", "TE One", "DDD", "TE"),
    ]
    assert drafted_ids_from_history(h) == {"1", "2", "3", "4"}
    assert roster_ids_for_slot(h, 2) == ["1", "4"]
    assert roster_ids_for_slot(h, 3) == ["3"]
    summary = roster_summary(h, 12, {2: "Manager Two", 3: "PatBot"})
    row2 = summary[summary["Slot"] == 2].iloc[0]
    assert row2["Manager"] == "Manager Two"
    assert row2["QB"] == "QB One"
    assert row2["TE"] == "TE One"
    assert row2["Picks"] == 2


def test_simulator_seeds_real_opponent_position_counts():
    cfg = load_config("config/league.yaml")
    df = make_pool()
    engine = DraftEngine(df, cfg)
    sim = FastDraftSimulator(engine)
    h = [
        make_pick_record(1, 12, "2", "RB One", "BBB", "RB"),
        make_pick_record(2, 12, "1", "QB One", "AAA", "QB"),
        make_pick_record(3, 12, "3", "WR One", "CCC", "WR"),
    ]
    counts = sim._seed_opponent_counts(h)
    qb_code = sim.pos_to_code["QB"]
    rb_code = sim.pos_to_code["RB"]
    wr_code = sim.pos_to_code["WR"]
    assert counts[2, qb_code] == 1
    assert counts[1, rb_code] == 1
    assert counts[3, wr_code] == 0
