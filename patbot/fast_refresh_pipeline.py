from __future__ import annotations

import json
from pathlib import Path
import time

import pandas as pd

from .candidate_news import apply_manual_risk_overrides, clear_candidate_news_cache
from .fast_risk import refresh_fast_risk


DEFAULT_CSV_PATH = Path("data/players_2026_live.csv")
DEFAULT_META_PATH = Path("data/players_2026_live.meta.json")


def run_fast_refresh(
    cfg: dict,
    csv_path: str | Path = DEFAULT_CSV_PATH,
    meta_path: str | Path = DEFAULT_META_PATH,
):
    """Refresh only current injury/news risk fields and persist them in place."""
    csv_path = Path(csv_path)
    meta_path = Path(meta_path)
    if not csv_path.exists():
        raise FileNotFoundError("No live player snapshot found. Run the full refresh first.")

    before = pd.read_csv(csv_path)
    old_risk = pd.to_numeric(before.get("risk_score"), errors="coerce")
    started = time.perf_counter()
    after, status = refresh_fast_risk(before, cfg)

    # Dated draft-night backstops are applied after the feed refresh so a known
    # hard availability state cannot disappear merely because a broad news feed
    # omits the player. Final Call's direct player check can still supersede a
    # backstop when it sees a newer explicit GREEN resolution.
    after = apply_manual_risk_overrides(after, cfg)
    clear_candidate_news_cache()
    after.to_csv(csv_path, index=False)

    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    meta["fast_risk_sources"] = status
    model_status = status.get("fast_risk_model", {})
    if model_status.get("refreshed_at_utc"):
        meta["fast_risk_refreshed_at_utc"] = model_status["refreshed_at_utc"]
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - started
    new_risk = pd.to_numeric(after.get("risk_score"), errors="coerce")
    delta = new_risk - old_risk.reindex(after.index)
    cols = [
        "name", "pos", "team", "current_alert_tier", "current_injury_status",
        "current_play_probability", "current_status_source", "current_status_material",
        "off_field_risk_level", "fast_news_title", "risk_score",
    ]
    cols = [c for c in cols if c in after.columns]
    report = after[cols].copy()
    report["risk_delta"] = delta
    if "current_status_material" in report.columns:
        material_flag = report["current_status_material"].fillna(False).astype(bool)
        alerts = report[material_flag].copy()
    else:
        alerts = report.iloc[0:0].copy()
    if not alerts.empty:
        severity = {"red": 0, "orange": 1, "yellow": 2, "none": 3}
        if "current_alert_tier" in alerts.columns:
            alerts["_tier_order"] = (
                alerts["current_alert_tier"].fillna("none").astype(str).str.lower().map(severity).fillna(3)
            )
        else:
            alerts["_tier_order"] = 3
        alerts = (
            alerts.sort_values(
                ["_tier_order", "risk_score", "risk_delta"],
                ascending=[True, False, False],
            )
            .drop(columns=["_tier_order"])
        )

    return after, status, alerts, elapsed
