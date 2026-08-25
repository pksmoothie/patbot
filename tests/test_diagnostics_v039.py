import json

import pandas as pd

from patbot.config import load_config
from patbot.diagnostics import build_diagnostic_engine, run_model_diagnostics
from patbot.draft import DraftEngine
from patbot.scoring import score_season_projection


def make_engine():
    cfg = load_config("config/league.yaml")
    rows = []
    pid = 1
    specs = [
        ("QB", 18),
        ("RB", 36),
        ("WR", 48),
        ("TE", 24),
        ("K", 14),
        ("DEF", 14),
    ]
    for pos, count in specs:
        for i in range(count):
            if pos == "QB":
                stats = {
                    "gp": 17,
                    "pass_cmp": 420 - i * 4,
                    "pass_yd": 4700 - i * 70,
                    "pass_td": 34 - i * 0.4,
                    "pass_int": 11 + i * 0.2,
                    "rush_yd": 450 - i * 8,
                    "rush_td": 5,
                }
            elif pos == "RB":
                stats = {
                    "gp": 17,
                    "rush_yd": 1350 - i * 18,
                    "rush_td": 10 - i * 0.08,
                    "rec": 55 - i * 0.4,
                    "rec_yd": 430 - i * 5,
                    "rec_td": 3,
                }
            elif pos == "WR":
                stats = {
                    "gp": 17,
                    "rec": 115 - i * 0.9,
                    "rec_yd": 1550 - i * 16,
                    "rec_td": 10 - i * 0.08,
                }
            elif pos == "TE":
                stats = {
                    "gp": 17,
                    "rec": 95 - i * 1.1,
                    "rec_yd": 1100 - i * 16,
                    "rec_td": 8 - i * 0.08,
                }
            else:
                stats = {"pts_ppr": 145 - i}

            scored = score_season_projection(
                stats,
                scoring=cfg["scoring"],
                bonus_model=cfg.get("bonus_model", {}),
                position=pos,
            )
            name = f"{pos}{i + 1}"
            if pos == "QB" and i == 0:
                name = "Josh Allen"
            elif pos == "TE" and i == 0:
                name = "Trey McBride"
            elif pos == "TE" and i == 1:
                name = "Brock Bowers"
            elif pos == "WR" and i == 0:
                name = "Ja'Marr Chase"

            rows.append({
                "player_id": str(pid),
                "name": name,
                "team": "X",
                "pos": pos,
                "adp": float(pid),
                "proj_points": scored["custom_points"],
                "base_custom_points": scored["base_points"],
                "estimated_bonus_points": scored["bonus_points"],
                "injury_risk": 0.0,
                "expert_rank": float(pid),
                "is_rookie": False,
                "raw_stats_json": json.dumps(stats),
            })
            pid += 1

    return DraftEngine(pd.DataFrame(rows), cfg)


def test_no_completion_bonus_rescores_qb_down_by_completion_points():
    engine = make_engine()
    baseline = engine.players.loc[engine.players["name"] == "Josh Allen", "proj_points"].iloc[0]
    variant, rescored = build_diagnostic_engine(engine, "no_completion")
    changed = variant.players.loc[variant.players["name"] == "Josh Allen", "proj_points"].iloc[0]
    assert rescored > 0
    assert round(float(baseline - changed), 2) == 105.0


def test_no_scarcity_variant_zeroes_only_scarcity_weight():
    engine = make_engine()
    variant, rescored = build_diagnostic_engine(engine, "no_scarcity")
    assert rescored == 0
    assert variant.config["draft_engine"]["weights"]["scarcity"] == 0.0
    assert variant.config["draft_engine"]["weights"]["vorp"] == engine.config["draft_engine"]["weights"]["vorp"]


def test_diagnostic_table_measures_elite_te_and_allen_paths():
    engine = make_engine()
    chase_id = engine.players.loc[engine.players["name"] == "Ja'Marr Chase", "player_id"].iloc[0]
    summary, details = run_model_diagnostics(
        engine,
        current_pick=3,
        drafted_ids={"1", "2"},
        my_roster_ids=[],
        candidate_id=str(chase_id),
        runs=2,
        through_round=3,
        draft_history=[],
    )
    assert len(summary) == 6
    assert "2nd Pick McBride/Bowers %" in summary.columns
    assert "3rd Pick Josh Allen %" in summary.columns
    assert "Δ Elite TE pp" in summary.columns
    assert "Δ Allen pp" in summary.columns
    assert len(details) == 6
