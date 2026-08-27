from types import SimpleNamespace

import numpy as np

from patbot.timing_audit import assign_roster_slots, starter_quality


def _quality_sim():
    pos = np.array([
        "QB", "QB", "QB", "QB", "QB", "QB", "QB", "QB", "QB",
        "TE", "TE", "TE", "TE", "TE", "TE", "TE", "TE", "TE",
    ])
    vorp = np.array([
        100, 95, 90, 85, 80, 75, 70, 65, 60,
        90, 85, 80, 75, 70, 65, 60, 55, 50,
    ], dtype=float)
    return SimpleNamespace(pos=pos, vorp=vorp)


def test_qb_quality_uses_top3_top8_vorp_buckets():
    sim = _quality_sim()
    assert starter_quality(sim, 0) == "elite"
    assert starter_quality(sim, 4) == "solid"
    assert starter_quality(sim, 8) == "weak"


def test_te_quality_is_model_rank_based():
    sim = _quality_sim()
    assert starter_quality(sim, 9) == "elite"
    assert starter_quality(sim, 13) == "solid"
    assert starter_quality(sim, 17) == "weak"


def _roster_sim():
    pos = np.array([
        "QB",
        "RB", "RB", "RB", "RB",
        "WR", "WR", "WR", "WR", "WR",
        "TE", "TE",
        "DEF", "K",
        "WR",
    ])
    proj = np.array([
        350,
        260, 240, 220, 180,
        280, 260, 240, 230, 170,
        250, 180,
        120, 110,
        210,
    ], dtype=float)
    vorp = proj - 100.0
    lws = np.linspace(30, 70, len(pos))
    engine = SimpleNamespace(
        roster_cfg={
            "QB": 1,
            "RB": 2,
            "WR": 3,
            "TE": 1,
            "FLEX": 1,
            "K": 1,
            "DEF": 1,
            "BN": 5,
            "flex_eligible": ["RB", "WR", "TE"],
        }
    )
    names = np.array([f"P{i}" for i in range(len(pos))])
    return SimpleNamespace(
        POSITIONS=("QB", "RB", "WR", "TE", "K", "DEF"),
        pos=pos,
        proj=proj,
        vorp=vorp,
        league_winner_score=lws,
        names=names,
        engine=engine,
    )


def test_assign_roster_slots_fills_configured_starting_lineup():
    sim = _roster_sim()
    roster = assign_roster_slots(sim, list(range(15)))
    assert set(["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DEF", "K"]).issubset(
        set(roster["Slot"])
    )


def test_assign_roster_slots_uses_best_remaining_flex_projection():
    sim = _roster_sim()
    roster = assign_roster_slots(sim, list(range(15)))
    flex = roster.loc[roster["Slot"].eq("FLEX")].iloc[0]
    # After RB1/RB2, WR1-3 and TE1 are filled, P8 (WR, 230) is the best remaining FLEX.
    assert flex["Player"] == "P8"


def test_assign_roster_slots_leaves_five_bench_spots():
    sim = _roster_sim()
    roster = assign_roster_slots(sim, list(range(15)))
    assert roster["Slot"].astype(str).str.startswith("BN").sum() == 5
