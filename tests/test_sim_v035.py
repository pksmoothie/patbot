import time
import pandas as pd

from patbot.config import load_config
from patbot.draft import DraftEngine
from patbot.sim import FastDraftSimulator, compare_candidates


def make_pool():
    rows = []
    pid = 1
    specs = [
        ("RB", 60, 315, 2.3),
        ("WR", 75, 320, 2.0),
        ("QB", 32, 415, 4.2),
        ("TE", 32, 255, 3.0),
        ("K", 20, 150, 1.0),
        ("DEF", 20, 145, 1.0),
    ]
    for pos, n, top, decline in specs:
        for i in range(n):
            rows.append({
                "player_id": str(pid),
                "name": f"{pos}{i+1}",
                "team": "X",
                "pos": pos,
                "adp": float(pid),
                "proj_points": float(top - i * decline),
                "injury_risk": 0.0,
                "expert_rank": float(pid),
            })
            pid += 1
    return pd.DataFrame(rows)


def test_archetype_mix_sums_to_11():
    cfg = load_config("config/league.yaml")
    counts = cfg["opponent_archetypes"]["counts"]
    assert sum(counts.values()) == 11
    assert counts == {
        "casual": 4,
        "market": 3,
        "league_aware": 2,
        "sharp": 1,
        "extremely_sharp": 1,
    }


def test_archetypes_randomly_fill_opponent_slots():
    cfg = load_config("config/league.yaml")
    engine = DraftEngine(make_pool(), cfg)
    sim = FastDraftSimulator(engine)
    rng = __import__("numpy").random.default_rng(123)
    assignments = sim._archetype_assignments(rng)
    assert len(assignments) == 11
    assert sim.slot not in assignments
    assert sorted(assignments.values()).count("casual") == 4
    assert sorted(assignments.values()).count("extremely_sharp") == 1


def test_lineup_evaluation_penalizes_zero_rb_foundation():
    cfg = load_config("config/league.yaml")
    df = make_pool()
    engine = DraftEngine(df, cfg)
    sim = FastDraftSimulator(engine)
    ids = {row["name"]: i for i, row in df.iterrows()}
    zero_rb = [
        ids["WR1"], ids["WR2"], ids["WR3"],
        ids["QB1"], ids["TE1"], ids["WR4"], ids["WR5"], ids["TE2"]
    ]
    balanced = [
        ids["WR1"], ids["WR2"], ids["WR3"],
        ids["RB1"], ids["RB2"], ids["QB1"], ids["TE1"], ids["WR4"]
    ]
    zero_score = sim.evaluate_roster(zero_rb)["lineup_score"]
    balanced_score = sim.evaluate_roster(balanced)["lineup_score"]
    assert balanced_score > zero_score


def test_six_candidate_sim_is_fast_and_returns_new_metrics():
    cfg = load_config("config/league.yaml")
    df = make_pool()
    engine = DraftEngine(df, cfg)
    start = time.perf_counter()
    summary, details = compare_candidates(
        engine=engine,
        current_pick=3,
        drafted_ids={"1", "2"},
        my_roster_ids=[],
        candidate_ids=["3", "4", "5", "6", "7", "8"],
        runs=30,
        through_round=8,
    )
    elapsed = time.perf_counter() - start
    assert len(summary) == 6
    assert "Avg Lineup Score" in summary.columns
    assert "Avg Starter VORP" in summary.columns
    assert len(details) == 6
    assert elapsed < 20
