import json
from pathlib import Path

import pandas as pd

from patbot.draft_persistence import clear_draft_session, load_draft_session, save_draft_session
from patbot.fast_refresh_pipeline import run_fast_refresh


def test_draft_session_round_trip(tmp_path):
    path = tmp_path / "draft.json"
    history = [{"overall_pick": 1, "player_id": "abc", "player": "Test Player"}]
    save_draft_session(history, path)
    assert load_draft_session(path) == history


def test_draft_session_malformed_fails_closed(tmp_path):
    path = tmp_path / "draft.json"
    path.write_text("not json", encoding="utf-8")
    assert load_draft_session(path) == []


def test_clear_draft_session(tmp_path):
    path = tmp_path / "draft.json"
    save_draft_session([{"overall_pick": 1}], path)
    assert path.exists()
    clear_draft_session(path)
    assert not path.exists()


def test_fast_refresh_pipeline_persists_csv_and_meta(tmp_path, monkeypatch):
    csv_path = tmp_path / "players.csv"
    meta_path = tmp_path / "meta.json"
    before = pd.DataFrame(
        [{
            "name": "Test Player",
            "pos": "WR",
            "team": "TST",
            "risk_score": 0.1,
            "current_injury_status": "",
            "current_play_probability": 1.0,
            "current_status_source": "none",
            "current_status_material": False,
            "off_field_risk_level": "none",
            "fast_news_title": "",
        }]
    )
    before.to_csv(csv_path, index=False)
    meta_path.write_text(json.dumps({"snapshot_at_utc": "2026-08-28T00:00:00Z"}), encoding="utf-8")

    def fake_refresh(frame, cfg):
        out = frame.copy()
        out["risk_score"] = 0.4
        out["current_injury_status"] = "PUP"
        out["current_status_source"] = "fantasypros"
        out["current_status_material"] = True
        return out, {
            "fast_risk_model": {"ok": True, "refreshed_at_utc": "2026-08-28T20:00:00Z"}
        }

    monkeypatch.setattr("patbot.fast_refresh_pipeline.refresh_fast_risk", fake_refresh)
    after, status, alerts, elapsed = run_fast_refresh({}, csv_path, meta_path)
    assert float(after.iloc[0]["risk_score"]) == 0.4
    assert len(alerts) == 1
    assert status["fast_risk_model"]["ok"] is True
    saved = pd.read_csv(csv_path)
    assert float(saved.iloc[0]["risk_score"]) == 0.4
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["fast_risk_refreshed_at_utc"] == "2026-08-28T20:00:00Z"
    assert elapsed >= 0
