import numpy as np
import pandas as pd

import patbot.risk as risk
from patbot.sim import FastDraftSimulator


def test_risk_layer_builds_history_current_and_manual_tails(monkeypatch):
    monkeypatch.setattr(risk.time, "sleep", lambda *_: None)

    def fake_get(path, params):
        if path == "nfl/players":
            return {
                "players": [
                    {"player_id": 1, "player_name": "Alpha Runner", "age": 30},
                    {"player_id": 2, "player_name": "Beta Receiver", "age": 25},
                ]
            }
        if path.endswith("/player-points"):
            year = int(path.split("/")[1])
            games = 4 if year == 2024 else 17
            return {
                "players": [
                    {"player_id": 1, "player_name": "Alpha Runner", "games": games},
                    {"player_id": 2, "player_name": "Beta Receiver", "games": 17},
                ]
            }
        if path == "nfl/injuries":
            return {
                "injuries": [
                    {"player_id": 1, "status": "Questionable", "probability_of_playing": "0.80"}
                ]
            }
        if path == "nfl/news":
            return {
                "items": [
                    {"player_id": 2, "title": "Beta Receiver faces legal investigation", "desc": "", "impact": ""}
                ]
            }
        raise AssertionError(path)

    monkeypatch.setattr(risk, "_fp_get", fake_get)

    players = pd.DataFrame([
        {"name": "Alpha Runner", "pos": "RB", "injury_risk": 0.0},
        {"name": "Beta Receiver", "pos": "WR", "injury_risk": 0.0},
    ])
    config = {
        "league": {"season": 2026},
        "risk_model": {
            "history_seasons": 3,
            "history_weights": [0.5, 0.3, 0.2],
            "fantasypros_request_spacing_seconds": 0,
        },
        "risk_overrides": {
            "Beta Receiver": {
                "off_field_event_probability": 0.08,
                "off_field_max_missed_games": 2,
                "note": "test flag",
            }
        },
    }

    out, status = risk.augment_risk_sources(players, config)
    alpha = out[out["name"] == "Alpha Runner"].iloc[0]
    beta = out[out["name"] == "Beta Receiver"].iloc[0]

    assert alpha["history_weighted_games"] < beta["history_weighted_games"]
    assert alpha["catastrophic_miss_probability"] > beta["catastrophic_miss_probability"]
    assert alpha["current_play_probability"] == 0.8
    assert beta["off_field_miss_probability"] >= 0.08
    assert np.isfinite(alpha["risk_score"])
    assert status["manual_risk_overrides"]["matched"] == 1


class TinyEngine:
    def __init__(self):
        self.players = pd.DataFrame([
            {
                "player_id": "1", "name": "Risky RB", "team": "X", "pos": "RB",
                "adp": 1.0, "proj_points": 340.0, "injury_risk": 0.5,
                "games_projected": 17.0, "catastrophic_miss_probability": 1.0,
                "minor_miss_lambda": 0.0, "off_field_miss_probability": 0.0,
                "off_field_max_missed_games": 0,
            },
            {
                "player_id": "2", "name": "Stable RB", "team": "Y", "pos": "RB",
                "adp": 2.0, "proj_points": 250.0, "injury_risk": 0.0,
                "games_projected": 17.0, "catastrophic_miss_probability": 0.0,
                "minor_miss_lambda": 0.0, "off_field_miss_probability": 0.0,
                "off_field_max_missed_games": 0,
            },
        ])
        self.config = {
            "league": {"teams": 12, "draft_slot": 3},
            "roster": {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "K": 1, "DEF": 1, "FLEX": 1, "flex_eligible": ["RB", "WR", "TE"]},
            "draft_engine": {
                "weights": {"vorp": 0.35, "projection": 0.25, "urgency": 0.20, "scarcity": 0.12, "roster_fit": 0.08},
                "injury_risk_penalty": 8.0,
            },
            "simulation": {},
            "risk_model": {
                "enabled": True,
                "catastrophic_min_missed_games": 4,
                "catastrophic_max_missed_games": 4,
                "replacement_capture_by_position": {"RB": 0.60},
            },
            "opponent_archetypes": {},
            "roster_evaluation": {},
        }
        self.league = self.config["league"]
        self.roster_cfg = self.config["roster"]
        self.engine_cfg = self.config["draft_engine"]

    def replacement_levels(self):
        return {"QB": 0.0, "RB": 200.0, "WR": 0.0, "TE": 0.0, "K": 0.0, "DEF": 0.0}


def test_risk_sampler_applies_tail_and_replacement_value():
    sim = FastDraftSimulator(TinyEngine())
    sampled, meta = sim._sample_run_projection(np.random.default_rng(7))

    assert meta["catastrophic"][0]
    assert meta["games"][0] == 13.0
    assert 340.0 * 13.0 / 17.0 < sampled[0] < 340.0
    assert sampled[1] == 250.0
