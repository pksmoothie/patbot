from __future__ import annotations

import json
from pathlib import Path
import py_compile

import pandas as pd

from patbot import __version__
from patbot.config import load_config
from patbot.data import load_players
from patbot.draft import DraftEngine
from patbot.sim import FastDraftSimulator
from patbot.yahoo_room_behavior import load_yahoo_room_cache


LIVE_CSV = Path("data/players_2026_live.csv")
LIVE_META = Path("data/players_2026_live.meta.json")
APP = Path("draft_day_app.py")


def _result(ok: bool, label: str, detail: str = "") -> bool:
    prefix = "OK  " if ok else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"{prefix} {label}{suffix}")
    return ok


def main():
    cfg = load_config()
    checks = []
    print(f"\nPatBot v{__version__} draft-day production preflight")
    print("Local smoke test only: no slow API refresh is performed here.\n")

    try:
        py_compile.compile(str(APP), doraise=True)
        checks.append(_result(True, "Hardened Streamlit app compiles", str(APP)))
    except Exception as exc:
        checks.append(_result(False, "Hardened Streamlit app compiles", f"{type(exc).__name__}: {exc}"))

    checks.append(_result(LIVE_CSV.exists(), "Live player snapshot exists", str(LIVE_CSV)))
    if not LIVE_CSV.exists():
        print("\nPRE-FLIGHT FAILED: run UPDATE_AND_RUN.bat once to create the live snapshot.")
        raise SystemExit(1)

    players = load_players(str(LIVE_CSV))
    players["player_id"] = players["player_id"].astype(str)
    checks.append(_result(len(players) >= 350, "Draftable player coverage", f"{len(players)} rows"))

    offense = players["pos"].astype(str).isin(["QB", "RB", "WR", "TE"])
    blend = pd.to_numeric(
        players.get("projection_blend_source_count", pd.Series([0] * len(players))),
        errors="coerce",
    ).fillna(0)
    blend_pct = 100.0 * float(((blend >= 2) & offense).sum()) / max(int(offense.sum()), 1)
    checks.append(_result(blend_pct >= 90.0, "Production projection blend attached", f"{blend_pct:.1f}% offense coverage"))

    risk_ok = "risk_score" in players.columns and pd.to_numeric(players["risk_score"], errors="coerce").notna().any()
    checks.append(_result(risk_ok, "Risk model fields present"))

    expert_ok = "expert_upside_lws_bonus" in players.columns
    # The expert layer is attached dynamically by DraftEngine if it is not persisted.
    engine = DraftEngine(players, cfg)
    expert_ok = expert_ok or ("expert_upside_lws_bonus" in engine.players.columns)
    checks.append(_result(expert_ok, "Expert late-round upside layer attaches"))

    meta = {}
    if LIVE_META.exists():
        try:
            meta = json.loads(LIVE_META.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    checks.append(_result(bool(meta.get("snapshot_at_utc")), "Slow refresh metadata present", str(meta.get("snapshot_at_utc", "missing"))))
    fast_stamp = meta.get("fast_risk_refreshed_at_utc")
    checks.append(_result(bool(fast_stamp), "Fast-risk refresh metadata present", str(fast_stamp or "run FAST_DRAFT_REFRESH.bat or use the app button before draft")))

    yahoo, yahoo_status = load_yahoo_room_cache(players["name"].astype(str).tolist())
    checks.append(
        _result(
            bool(yahoo_status.get("ok")),
            "Yahoo supporting room cache",
            (
                f"{yahoo_status.get('matched', len(yahoo))} matches, "
                f"{yahoo_status.get('age_hours', '?')}h old"
                if yahoo_status.get("ok")
                else str(yahoo_status.get("reason", "unavailable"))
            ),
        )
    )

    try:
        board = engine.recommend(current_pick=3, drafted_ids=set(), roster_positions=[], top_n=5)
        board_ok = not board.empty and {"name", "score", "proj_points", "vorp"}.issubset(board.columns)
        checks.append(_result(board_ok, "DraftEngine recommendation smoke test", f"top={board.iloc[0]['name'] if not board.empty else 'none'}"))
    except Exception as exc:
        checks.append(_result(False, "DraftEngine recommendation smoke test", f"{type(exc).__name__}: {exc}"))

    try:
        sim = FastDraftSimulator(engine)
        sim_status = getattr(sim, "yahoo_room_status", {}) or {}
        checks.append(_result(bool(sim_status.get("ok")), "Yahoo reaches production opponent simulator", str(sim_status.get("reason", "ok"))))
    except Exception as exc:
        checks.append(_result(False, "Production opponent simulator instantiates", f"{type(exc).__name__}: {exc}"))

    print("\nManual Yahoo draft entry remains the production path until/unless API access is available.")
    print("A browser/app restart will restore recorded picks from the local draft-session backup.\n")

    failed = sum(1 for x in checks if not x)
    if failed:
        print(f"PRE-FLIGHT: {failed} check(s) need attention.")
        raise SystemExit(1)
    print("PRE-FLIGHT: ALL CHECKS PASSED.")


if __name__ == "__main__":
    main()
