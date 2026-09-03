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


def test_fast_news_stays_broad_without_player_specific_archive_calls(monkeypatch):
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
    calls = []

    def fake_get(_path, params):
        calls.append(dict(params))
        return {"items": []}

    monkeypatch.setattr("patbot.draft_news_hardening._fp_get", fake_get)
    monkeypatch.setattr("patbot.draft_news_hardening.time.sleep", lambda _seconds: None)

    news, meta = fetch_draft_news(
        players,
        {
            "risk_model": {
                "fantasypros_request_spacing_seconds": 0,
                "draft_news_category_limit": 25,
                "draft_news_max_age_days": 14,
            }
        },
    )

    assert news == {}
    assert len(calls) == 5
    assert not any("fpid" in call for call in calls)
    assert meta["successful_calls"] == 5
    assert "Final Call" in meta["note"]
