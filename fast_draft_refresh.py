from __future__ import annotations

import pandas as pd

from patbot import __version__
from patbot.config import load_config
from patbot.fast_refresh_pipeline import run_fast_refresh


def main():
    cfg = load_config()
    try:
        after, status, alerts, elapsed = run_fast_refresh(cfg)
    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}")
        raise SystemExit(1)

    print(f"\nPatBot v{__version__} fast draft-day injury/news refresh")
    print(f"Completed in {elapsed:.1f}s without refetching six-year history or projections.\n")
    for source, item in status.items():
        if item.get("ok"):
            extra = f" | matched {item['matched']}" if item.get("matched") is not None else ""
            if item.get("alerts") is not None:
                extra += f" | material alerts {item['alerts']}"
            print(f"OK   {source}{extra}")
        else:
            print(f"WARN {source}: {item.get('error', 'unavailable')}")

    print("\n=== MATERIAL DRAFT-DAY ALERTS ===\n")
    if alerts.empty:
        print("No material current-status/news alerts detected.")
    else:
        print(alerts.head(40).to_string(index=False))

    if "current_status_source" in after.columns and "current_injury_status" in after.columns:
        ignored = after[
            after["current_status_source"].fillna("").eq("sleeper_ignored")
            & after["current_injury_status"].fillna("").astype(str).str.strip().ne("")
        ]
        print(f"\nSleeper-only soft status labels ignored for draft risk: {len(ignored)}")
    print("The live CSV has been updated in place. Streamlit will use these risk fields on its next rerun.\n")


if __name__ == "__main__":
    main()
