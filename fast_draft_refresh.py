from __future__ import annotations

import json
from pathlib import Path
import time

import pandas as pd

from patbot.config import load_config
from patbot.fast_risk import refresh_fast_risk


CSV_PATH = Path("data/players_2026_live.csv")
META_PATH = Path("data/players_2026_live.meta.json")


def main():
    cfg = load_config()
    if not CSV_PATH.exists():
        raise SystemExit("No live player snapshot found. Run UPDATE_AND_RUN.bat first.")

    before = pd.read_csv(CSV_PATH)
    old_risk = pd.to_numeric(before.get("risk_score"), errors="coerce")

    started = time.perf_counter()
    after, status = refresh_fast_risk(before, cfg)
    after.to_csv(CSV_PATH, index=False)

    meta = {}
    if META_PATH.exists():
        try:
            meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    meta["fast_risk_sources"] = status
    model_status = status.get("fast_risk_model", {})
    if model_status.get("refreshed_at_utc"):
        meta["fast_risk_refreshed_at_utc"] = model_status["refreshed_at_utc"]
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - started
    new_risk = pd.to_numeric(after.get("risk_score"), errors="coerce")
    delta = new_risk - old_risk.reindex(after.index)
    report = after[[
        "name", "pos", "team", "current_injury_status", "current_play_probability",
        "off_field_risk_level", "fast_news_title", "risk_score",
    ]].copy()
    report["risk_delta"] = delta

    alerts = report[
        report["current_injury_status"].fillna("").astype(str).str.strip().ne("")
        | report["off_field_risk_level"].fillna("none").astype(str).str.lower().ne("none")
        | report["fast_news_title"].fillna("").astype(str).str.strip().ne("")
        | report["risk_delta"].abs().ge(0.03)
    ].copy()
    alerts = alerts.sort_values(["risk_score", "risk_delta"], ascending=[False, False])

    print("\nPatBot v0.5.2 fast draft-day injury/news refresh")
    print(f"Completed in {elapsed:.1f}s without refetching six-year history or projections.\n")
    for source, item in status.items():
        if item.get("ok"):
            extra = f" | matched {item['matched']}" if item.get("matched") is not None else ""
            if item.get("alerts") is not None:
                extra += f" | alerts {item['alerts']}"
            print(f"OK   {source}{extra}")
        else:
            print(f"WARN {source}: {item.get('error', 'unavailable')}")

    print("\n=== CURRENT DRAFT-DAY ALERTS / MATERIAL RISK CHANGES ===\n")
    if alerts.empty:
        print("No material current-status/news changes detected.")
    else:
        print(alerts.head(40).to_string(index=False))

    print("\nThe live CSV has been updated in place. Streamlit will use these risk fields on its next rerun.\n")


if __name__ == "__main__":
    main()
