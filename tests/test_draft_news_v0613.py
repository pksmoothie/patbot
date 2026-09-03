import pandas as pd

from patbot.draft_news import classify_draft_news, select_draft_news_signals
from patbot.fast_risk import (
    _draft_day_status_probability,
    _is_serious_sleeper_status,
    refresh_fast_risk,
)


def test_commissioners_exempt_list_is_red_and_confirmed_availability_risk():
    signal = classify_draft_news(
        "Packers RB Josh Jacobs placed on Commissioner's Exempt List and cannot practice or play"
    )
    assert signal["tier"] == "red"
    assert signal["material"] is True
    assert signal["play_probability_cap"] == 0.20
    assert signal["off_field_event_probability"] == 1.0
    assert signal["off_field_max_missed_games"] == 6


def test_repeated_practice_absence_is_orange_not_red():
    signal = classify_draft_news(
        "TreVeyon Henderson did not practice Thursday and has not practiced in 10 days"
    )
    assert signal["tier"] == "orange"
    assert signal["play_probability_cap"] == 0.65
    assert signal["off_field_event_probability"] == 0.0


def test_legal_news_without_league_action_is_monitor_only_yellow():
    signal = classify_draft_news(
        "Player charged with misdemeanor battery while league investigation continues"
    )
    assert signal["tier"] == "yellow"
    assert signal["play_probability_cap"] == 0.95
    assert signal["off_field_event_probability"] == 0.03


def test_resolution_news_is_green_and_suppresses_stale_negative_story():
    now = pd.Timestamp("2026-09-03T22:00:00Z")
    items = [
        {
            "player_id": "10",
            "title": "Player returned to practice and is expected to play Week 1",
            "created": "2026-09-03T18:00:00Z",
        },
        {
            "player_id": "10",
            "title": "Player did not practice and could miss Week 1",
            "created": "2026-09-02T18:00:00Z",
        },
    ]
    active, meta = select_draft_news_signals(
        items,
        fp_ids={"10"},
        fp_id_to_name={"10": "Example Player"},
        now=now,
    )
    assert "10" not in active
    assert meta["resolved_players"] == 1


def test_missing_player_id_can_match_full_player_name_in_news_text():
    now = pd.Timestamp("2026-09-03T22:00:00Z")
    items = [
        {
            "title": "Josh Jacobs placed on Commissioner's Exempt List",
            "desc": "Green Bay is preparing to play without Josh Jacobs.",
            "created": "2026-09-03T20:00:00Z",
        }
    ]
    active, meta = select_draft_news_signals(
        items,
        fp_ids={"99"},
        fp_id_to_name={"99": "Josh Jacobs"},
        now=now,
    )
    assert active["99"]["tier"] == "red"
    assert active["99"]["matched_by"] == "player_name"
    assert meta["name_fallback_matches"] == 1


def test_sleeper_exempt_or_suspended_status_is_hard_even_without_injury_row():
    assert _is_serious_sleeper_status("Commissioner Exempt")
    assert _is_serious_sleeper_status("Suspended")
    status, probability, source = _draft_day_status_probability(None, "Commissioner Exempt")
    assert status == "Commissioner Exempt"
    assert probability == 0.20
    assert source == "sleeper_hard"


def test_fast_risk_applies_red_news_without_turning_yellow_into_red(monkeypatch):
    players = pd.DataFrame(
        [
            {
                "player_id": "s1",
                "fp_player_id": "99",
                "name": "Josh Jacobs",
                "pos": "RB",
                "team": "GB",
                "injury_risk": 0.0,
                "history_missed_rate": 0.0,
                "history_signal_scale": 1.0,
                "age_tail_bonus": 0.0,
                "history_weighted_games": 17.0,
                "history_seasons_observed": 6,
            },
            {
                "player_id": "s2",
                "fp_player_id": "100",
                "name": "Monitor Player",
                "pos": "WR",
                "team": "X",
                "injury_risk": 0.0,
                "history_missed_rate": 0.0,
                "history_signal_scale": 1.0,
                "age_tail_bonus": 0.0,
                "history_weighted_games": 17.0,
                "history_seasons_observed": 6,
            },
        ]
    )

    monkeypatch.setattr(
        "patbot.fast_risk._fetch_sleeper_current_status",
        lambda _players: ({}, {"ok": True, "matched": 0}),
    )
    monkeypatch.setattr(
        "patbot.fast_risk._fetch_injuries",
        lambda _ids, _season, _config: ({}, {"ok": True, "matched": 0}),
    )
    monkeypatch.setattr(
        "patbot.fast_risk.fetch_draft_news",
        lambda _players, _config: (
            {
                "99": {
                    **classify_draft_news(
                        "Josh Jacobs placed on Commissioner's Exempt List and cannot practice or play"
                    ),
                    "title": "Josh Jacobs placed on Commissioner's Exempt List",
                    "created": "2026-09-03T20:00:00Z",
                },
                "100": {
                    **classify_draft_news(
                        "Monitor Player charged with a misdemeanor; league has not imposed discipline"
                    ),
                    "title": "Monitor Player charged with misdemeanor",
                    "created": "2026-09-03T20:00:00Z",
                },
            },
            {"ok": True, "matched": 2},
        ),
    )

    out, status = refresh_fast_risk(players, {"league": {"season": 2026}, "risk_model": {}})
    jacobs = out[out["name"] == "Josh Jacobs"].iloc[0]
    monitor = out[out["name"] == "Monitor Player"].iloc[0]

    assert jacobs["fast_news_tier"] == "red"
    assert float(jacobs["current_play_probability"]) == 0.20
    assert float(jacobs["off_field_miss_probability"]) == 1.0
    assert str(jacobs["fast_news_title"]).startswith("RED —")
    assert monitor["fast_news_tier"] == "yellow"
    assert float(monitor["current_play_probability"]) == 0.95
    assert float(monitor["off_field_miss_probability"]) == 0.03
    assert status["fast_risk_model"]["alert_tiers"]["red"] == 1
    assert status["fast_risk_model"]["alert_tiers"]["yellow"] == 1
