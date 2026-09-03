import pandas as pd

from patbot.draft_news import classify_draft_news
from patbot.draft_news_hardening import fetch_draft_news


def test_not_placed_on_ir_is_not_red():
    signal = classify_draft_news(
        "Ashton Jeanty received a positive update. He did not practice Tuesday, "
        "but he wasn't placed on IR and could be back as soon as Week 1."
    )
    assert signal["tier"] == "orange"
    assert signal["play_probability_cap"] == 0.65


def test_wide_news_pull_can_rescue_player_outside_priority_slice(monkeypatch):
    players = pd.DataFrame(
        [
            {
                "name": "Josh Jacobs",
                "pos": "RB",
                "fp_player_id": "123",
                "proj_points": 220.0,
            }
        ]
    )

    monkeypatch.setattr(
        "patbot.draft_news_hardening._ORIGINAL_FETCH",
        lambda _players, _config: ({}, {"ok": True, "matched": 0}),
    )
    monkeypatch.setattr(
        "patbot.draft_news_hardening.time.sleep",
        lambda _seconds: None,
    )
    monkeypatch.setattr(
        "patbot.draft_news_hardening._fp_get",
        lambda _path, _params: {
            "items": [
                {
                    "player_id": "123",
                    "title": "Josh Jacobs placed on commissioner exempt list",
                    "desc": "Jacobs is ineligible to practice or play while on the list.",
                    "created": "2026-08-30 18:29:00",
                }
            ]
        },
    )

    news, meta = fetch_draft_news(
        players,
        {
            "risk_model": {
                "fantasypros_request_spacing_seconds": 0,
                "draft_news_wide_limit": 500,
                "draft_news_max_age_days": 14,
            }
        },
    )
    assert news["123"]["tier"] == "red"
    assert news["123"]["play_probability_cap"] == 0.20
    assert meta["wide_news"]["requested_limit"] == 500
    assert meta["wide_news"]["active_overrides"] == 1
