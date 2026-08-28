from __future__ import annotations

import numpy as np
import pandas as pd

from patbot.config import load_config
from patbot.data import load_players
from patbot.draft import DraftEngine
from patbot.draft_state import make_pick_record
from patbot.path_value import run_pick_specific_path_audit
from patbot.sim import FastDraftSimulator


def _row_by_name(players: pd.DataFrame, name: str) -> pd.Series:
    match = players[players["name"].eq(name)]
    if match.empty:
        raise RuntimeError(f"Could not find {name!r} in the live player snapshot.")
    return match.iloc[0]


def _structural_rollup(events: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate challenger lenses before looking for broad patterns."""
    if events.empty:
        return pd.DataFrame()

    # The same player can be nominated by Score/VORP/ECR/LWS/Q90 in one room.
    # Count that path once so a multi-lens nomination cannot manufacture support.
    keys = ["Run", "Round", "Overall", "Selected", "Challenger", "Wait Turns"]
    unique = events.sort_values(keys + ["Challenger Type"]).drop_duplicates(keys, keep="first")

    rows = []
    group_keys = ["Round", "Selected Pos", "Wait Turns"]
    for key, group in unique.groupby(group_keys, sort=True):
        delta = pd.to_numeric(group["Alt Delta vs Take Now"], errors="coerce").dropna()
        if delta.empty:
            continue
        alt_win = 100.0 * float((delta > 1e-9).mean())
        mean_delta = float(delta.mean())
        if len(group) >= 8 and alt_win >= 60.0 and mean_delta >= 10.0:
            signal = "WAIT LEAN"
        elif len(group) >= 8 and alt_win <= 40.0 and mean_delta <= -10.0:
            signal = "TAKE-NOW LEAN"
        else:
            signal = "MIXED"

        def avg(column: str) -> float:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            return float(values.mean()) if not values.empty else np.nan

        rows.append(
            {
                "Round": int(key[0]),
                "Selected Pos": str(key[1]),
                "Wait Turns": int(key[2]),
                "Unique Paths": int(len(group)),
                "Selected Survives %": round(100.0 * float(group["Selected Survived"].mean()), 1),
                "Alt Wins %": round(alt_win, 1),
                "Avg Alt Delta": round(mean_delta, 2),
                "P50 Alt Delta": round(float(np.median(delta)), 2),
                "Avg VORP Cost": round(avg("VORP Cost Waiting"), 2),
                "Avg ECR Cost": round(avg("ECR Cost Waiting"), 2),
                "Avg LWS Cost": round(avg("LWS Cost Waiting"), 2),
                "Avg Q90 Cost": round(avg("Q90 Cost Waiting"), 2),
                "Screen Signal": signal,
            }
        )

    return pd.DataFrame(rows).sort_values(["Round", "Selected Pos", "Wait Turns"]).reset_index(drop=True)


def main():
    cfg = load_config()
    players = load_players("data/players_2026_live.csv")
    players["player_id"] = players["player_id"].astype(str)
    engine = DraftEngine(players, cfg)
    teams = int(cfg["league"]["teams"])

    gibbs = _row_by_name(players, "Jahmyr Gibbs")
    bijan = _row_by_name(players, "Bijan Robinson")
    chase = _row_by_name(players, "Ja'Marr Chase")

    opening_history = [
        make_pick_record(1, teams, str(gibbs["player_id"]), gibbs["name"], gibbs["team"], gibbs["pos"]),
        make_pick_record(2, teams, str(bijan["player_id"]), bijan["name"], bijan["team"], bijan["pos"]),
    ]

    # Instantiate once only to show that the production Yahoo cache is actually on.
    probe = FastDraftSimulator(engine)
    yahoo = getattr(probe, "yahoo_room_status", {}) or {}

    print("\nPatBot v0.6.1 Yahoo-informed path-value construction recheck")
    print("DIAGNOSTIC ONLY: no construction/urgency weights are changed by this script.")
    print("Calibration branch: Gibbs 1.01, Bijan 1.02, Ja'Marr Chase locked at 1.03.")
    print("Same 60-room seed family as the v0.5.8 screen, but opponents now use the production Yahoo supporting signal.\n")
    print(
        f"Yahoo room cache: {'ON' if yahoo.get('ok') else 'OFF'} | "
        f"matched {yahoo.get('matched', 0)} | coverage {yahoo.get('coverage_pct', 0)}% | "
        f"age {yahoo.get('age_hours', '—')}h"
    )
    if not yahoo.get("ok"):
        print(f"Yahoo fallback reason: {yahoo.get('reason', 'unknown')} (audit will still run on the base room model).")

    def progress(done: int, total: int):
        print(f"  {done:>3}/{total} rooms complete", flush=True)

    summary, common, events, meta = run_pick_specific_path_audit(
        engine,
        opening_history,
        forced_first_player_id=str(chase["player_id"]),
        runs=60,
        through_round=13,
        audit_start_round=2,
        audit_through_round=7,
        wait_turns=(1, 2),
        challenger_pool=12,
        progress=progress,
    )

    print("\n=== STRUCTURAL ROLLUP (DUPLICATE CHALLENGER LENSES COLLAPSED) ===\n")
    rollup = _structural_rollup(events)
    if rollup.empty:
        print("No structural path-value events were captured.")
    else:
        print(rollup.to_string(index=False))

    print("\n=== ORIGINAL FIVE-LENS SUMMARY ===\n")
    if summary.empty:
        print("No five-lens summary events were captured.")
    else:
        print(summary.to_string(index=False))

    print("\n=== MOST COMMON SPECIFIC PATHS ===\n")
    if common.empty:
        print("No specific paths were captured.")
    else:
        print(common.head(35).to_string(index=False))

    print("\nInterpretation rules:")
    print("- The structural rollup counts the same challenger only once per room/pick/wait horizon even if several lenses nominated him.")
    print("- WAIT LEAN requires >=8 unique paths, >=60% alternative wins and >=+10 average lineup points.")
    print("- TAKE-NOW LEAN requires >=8 unique paths, <=40% alternative wins and <=-10 average lineup points.")
    print("- Everything else is MIXED; these labels are screening diagnostics, not hard draft rules.")
    print("- Positive waiting costs mean the eventual same-position fallback was worse on that metric.")
    print("- Yahoo affects opponent behavior/survival only; PatBot valuation and current-score construction remain unchanged in this recheck.")
    print(f"- Seed: {meta['seed']} | rooms: {meta['runs']} | challenger pool: {meta['challenger_pool']}\n")


if __name__ == "__main__":
    main()
