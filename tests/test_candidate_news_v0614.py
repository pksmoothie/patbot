import pandas as pd

from patbot import candidate_news
from patbot import candidate_news_final_call


def _player_row(name="Josh Jacobs", player_id="5850", fp_id="18269"):
    return {
        "name": name,
        "player_id": player_id,
        "fp_player_id": fp_id,
        "team": "GB",
        "pos": "RB",
        "injury_status": "",
        "current_injury_status": "",
        "current_status_source": "none",
        "current_play_probability": 1.0,
        "current_status_material": False,
        "current_alert_tier": "none",
        "sleeper_current_injury_risk": 0.0,
        "history_missed_rate": 0.0,
        "history_signal_scale": 1.0,
        "age_tail_bonus": 0.0,
        "off_field_miss_probability": 0.0,
        "off_field_max_missed_games": 0,
        "risk_score": 0.0,
        "injury_risk": 0.0,
        "risk_note": "",
        "fast_news_title": "",
    }


def test_direct_jacobs_followup_remains_red_while_exempt_status_is_unresolved():
    items = [
        {
            "player_id": 18269,
            "created": "2026-09-01 17:04:11",
            "title": "Josh Jacobs expected to play for Packers in 2026 per GM",
            "desc": "Jacobs remains on the commissioner's exempt list and cannot practice or play while listed.",
        },
        {
            "player_id": 18269,
            "created": "2026-08-30 18:29:02",
            "title": "Josh Jacobs placed on commissioner exempt list",
        },
    ]
    signal = candidate_news.newest_direct_signal(
        items,
        now=pd.Timestamp("2026-09-03T23:00:00Z"),
    )
    assert signal["tier"] == "red"
    assert signal["created"] == "2026-09-01 17:04:11"
    assert signal["play_probability_cap"] == 0.20


def test_candidate_direct_lookup_is_cached(monkeypatch):
    calls = []

    def fake_get(path, params):
        calls.append((path, dict(params)))
        return {
            "items": [
                {
                    "player_id": 18269,
                    "created": "2026-09-01 17:04:11",
                    "title": "Josh Jacobs remains on commissioner exempt list",
                }
            ]
        }

    monkeypatch.setattr(candidate_news, "_fp_get", fake_get)
    players = pd.DataFrame([_player_row()])
    cfg = {
        "risk_model": {
            "fantasypros_request_spacing_seconds": 0,
            "candidate_news_cache_ttl_seconds": 600,
        }
    }
    cache = {}
    now = pd.Timestamp("2026-09-03T23:00:00Z")

    first, first_meta = candidate_news.verify_candidate_news(
        players, ["5850"], cfg, cache=cache, now=now
    )
    second, second_meta = candidate_news.verify_candidate_news(
        players, ["5850"], cfg, cache=cache, now=now + pd.Timedelta(minutes=2)
    )

    assert first["5850"]["tier"] == "red"
    assert second["5850"]["tier"] == "red"
    assert len(calls) == 1
    assert first_meta["api_calls"] == 1
    assert second_meta["api_calls"] == 0
    assert second_meta["cache_hits"] == 1


def test_jacobs_manual_backstop_is_red_and_affects_model_risk(monkeypatch):
    monkeypatch.setattr(candidate_news, "_override_is_active", lambda override: True)
    players = pd.DataFrame([_player_row()])
    out = candidate_news.apply_manual_risk_overrides(players, {"risk_model": {}})
    jacobs = out.iloc[0]

    assert jacobs["current_alert_tier"] == "red"
    assert jacobs["current_status_source"] == "manual_red"
    assert float(jacobs["current_play_probability"]) == 0.20
    assert float(jacobs["off_field_miss_probability"]) == 1.0
    assert float(jacobs["injury_risk"]) > 0.0


def test_newer_green_direct_signal_supersedes_manual_jacobs_backstop(monkeypatch):
    monkeypatch.setattr(candidate_news, "_override_is_active", lambda override: True)
    players = pd.DataFrame([_player_row()])
    green = {
        "tier": "green",
        "level": "none",
        "resolved": True,
        "material": False,
        "title": "Josh Jacobs removed from the commissioner's exempt list",
        "created": "2026-09-04T01:00:00Z",
        "_timestamp": "2026-09-04T01:00:00Z",
        "play_probability_cap": 1.0,
        "off_field_event_probability": 0.0,
        "off_field_max_missed_games": 0,
    }
    out = candidate_news.apply_candidate_news_signals(
        players, {"5850": green}, {"risk_model": {}}
    )
    jacobs = out.iloc[0]

    assert jacobs["current_status_material"] is False or bool(jacobs["current_status_material"]) is False
    assert jacobs["current_alert_tier"] == "none"
    assert float(jacobs["current_play_probability"]) == 1.0
    assert float(jacobs["off_field_miss_probability"]) == 0.0


def test_final_call_rebuilds_board_after_direct_candidate_risk(monkeypatch):
    rows = [
        _player_row("Josh Jacobs", "5850", "18269"),
        _player_row("Healthy Player", "9999", "99999"),
    ]
    rows[0]["injury_risk"] = 0.0
    rows[1]["injury_risk"] = 0.1
    players = pd.DataFrame(rows)

    class FakeEngine:
        def __init__(self, frame, config):
            self.players = frame.copy()
            self.config = config

        def recommend(self, **kwargs):
            return self.players.sort_values(
                ["injury_risk", "name"], ascending=[True, True]
            ).reset_index(drop=True)

    engine = FakeEngine(
        players,
        {"risk_model": {"candidate_news_verify_count": 2}},
    )
    board = engine.recommend()

    red = {
        "tier": "red",
        "level": "high",
        "resolved": False,
        "material": True,
        "title": "Josh Jacobs placed on commissioner exempt list",
        "created": "2026-09-01T17:04:11Z",
        "_timestamp": "2026-09-01T17:04:11Z",
        "play_probability_cap": 0.20,
        "off_field_event_probability": 1.0,
        "off_field_max_missed_games": 6,
        "reason": "confirmed unavailability/league action",
    }

    monkeypatch.setattr(
        candidate_news_final_call,
        "verify_candidate_news",
        lambda frame, ids, config: ({"5850": red}, {"ok": True, "checked": 2, "requested": 2, "cache_hits": 0, "api_calls": 2, "failures": 0, "ttl_seconds": 600}),
    )
    monkeypatch.setattr(
        candidate_news_final_call,
        "_ORIGINAL_RUN_FINAL_CALL",
        lambda engine, **kwargs: {
            "recommendation": kwargs["board"].iloc[0]["name"],
            "reason": "base final call",
        },
    )

    result = candidate_news_final_call.run_final_call(
        engine,
        current_pick=50,
        drafted_ids=set(),
        my_roster_ids=[],
        board=board,
        draft_history=[],
    )

    assert result["recommendation"] == "Healthy Player"
    assert result["candidate_news_verification"]["active_signals"]["5850"]["tier"] == "red"
