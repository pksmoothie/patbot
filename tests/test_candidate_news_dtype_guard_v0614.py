import numpy as np
import pandas as pd

from patbot.candidate_news import apply_manual_risk_overrides


def test_manual_backstop_can_write_text_into_all_nan_csv_columns():
    players = pd.DataFrame([
        {
            "player_id": "5850",
            "fp_player_id": 18269.0,
            "name": "Josh Jacobs",
            "team": "GB",
            "pos": "RB",
            "current_status_source": np.nan,
            "current_alert_tier": np.nan,
            "fast_news_tier": np.nan,
            "fast_news_title": np.nan,
            "fast_news_reason": np.nan,
            "fast_news_created": np.nan,
            "off_field_risk_level": np.nan,
            "risk_note": np.nan,
            "candidate_news_verified_at_utc": np.nan,
            "current_play_probability": 1.0,
            "current_status_material": False,
            "history_missed_rate": 0.10,
            "history_signal_scale": 1.0,
            "age_tail_bonus": 0.0,
            "sleeper_current_injury_risk": 0.0,
        }
    ])

    out = apply_manual_risk_overrides(players, {"risk_model": {}})
    row = out.iloc[0]
    assert row["current_status_source"] == "manual_red"
    assert row["current_alert_tier"] == "red"
    assert str(row["fast_news_title"]).startswith("RED —")
    assert float(row["current_play_probability"]) == 0.20
