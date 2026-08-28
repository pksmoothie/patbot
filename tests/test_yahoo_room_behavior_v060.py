import os
import time

import numpy as np
import pandas as pd

from patbot.yahoo_room_behavior import blend_room_market, load_yahoo_room_cache


def test_yahoo_blend_is_supporting_and_capped():
    market = np.array([100.0, 50.0, 80.0])
    yahoo = np.array([80.0, 70.0, 20.0])
    out = blend_room_market(market, yahoo, 0.90)
    assert np.allclose(out, [93.0, 57.0, 59.0])


def test_yahoo_blend_leaves_missing_values_on_existing_market():
    market = np.array([100.0, 50.0, 80.0])
    yahoo = np.array([np.nan, 40.0, np.nan])
    out = blend_room_market(market, yahoo, 0.25)
    assert np.allclose(out, [100.0, 47.5, 80.0], equal_nan=True)


def test_load_yahoo_room_cache_matches_names(tmp_path):
    path = tmp_path / "yahoo.csv"
    pd.DataFrame(
        [
            {"name": "Ja'Marr Chase", "yahoo_adp": 3.4},
            {"name": "Jahmyr Gibbs", "yahoo_adp": 1.4},
            {"name": "Not In PatBot", "yahoo_adp": 20.0},
        ]
    ).to_csv(path, index=False)
    out, status = load_yahoo_room_cache(
        ["Ja'Marr Chase", "Jahmyr Gibbs"],
        path=path,
        max_age_hours=72,
    )
    assert status["ok"] is True
    assert status["matched"] == 2
    assert set(out["name"]) == {"Ja'Marr Chase", "Jahmyr Gibbs"}


def test_stale_yahoo_cache_cleanly_disables(tmp_path):
    path = tmp_path / "yahoo.csv"
    pd.DataFrame([{"name": "Ja'Marr Chase", "yahoo_adp": 3.4}]).to_csv(path, index=False)
    stale = time.time() - 10 * 3600
    os.utime(path, (stale, stale))
    out, status = load_yahoo_room_cache(
        ["Ja'Marr Chase"],
        path=path,
        max_age_hours=2,
    )
    assert out.empty
    assert status["ok"] is False
    assert status["reason"] == "stale"
