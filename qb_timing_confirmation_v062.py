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


def _unique_qb_paths(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    qb = events[events["Selected Pos"].eq("QB")].copy()
    if qb.empty:
        return qb
    keys = ["Run", "Round", "Overall", "Selected", "Challenger", "Wait Turns"]
    return qb.sort_values(keys + ["Challenger Type"]).drop_duplicates(keys, keep="first")


def _summarize(group: pd.DataFrame) -> dict:
    delta = pd.to_numeric(group["Alt Delta vs Take Now"], errors="coerce").dropna()
    if delta.empty:
        return {}
    wins = 100.0 * float((delta > 1e-9).mean())
    mean_delta = float(delta.mean())
    if len(group) >= 20 and wins >= 60.0 and mean_delta >= 10.0:
        signal = "WAIT LEAN"
    elif len(group) >= 20 and wins <= 40.0 and mean_delta <= -10.0:
        signal = "TAKE-NOW LEAN"
    else:
        signal = "MIXED"

    def avg(column: str) -> float:
        vals = pd.to_numeric(group[column], errors="coerce").dropna()
        return float(vals.mean()) if not vals.empty else np.nan

    return {
        "Unique Paths": int(len(group)),
        "Selected Survives %": round(100.0 * float(group["Selected Survived"].mean()), 1),
        "Alt Wins %": round(wins, 1),
        "Avg Alt Delta": round(mean_delta, 2),
        "P25 Delta": round(float(np.percentile(delta, 25)), 2),
        "P50 Delta": round(float(np.median(delta)), 2),
        "P75 Delta": round(float(np.percentile(delta, 75)), 2),
        "Avg VORP Cost": round(avg("VORP Cost Waiting"), 2),
        "Avg ECR Cost": round(avg("ECR Cost Waiting"), 2),
        "Avg LWS Cost": round(avg("LWS Cost Waiting"), 2),
        "Avg Q90 Cost": round(avg("Q90 Cost Waiting"), 2),
        "Signal": signal,
    }


def _round_rollup(unique: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (round_no, wait_turns), group in unique.groupby(["Round", "Wait Turns"], sort=True):
        row = _summarize(group)
        if not row:
            continue
        row = {"Round": int(round_no), "Wait Turns": int(wait_turns), **row}
        rows.append(row)
    return pd.DataFrame(rows)


def _player_rollup(unique: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (round_no, selected, wait_turns), group in unique.groupby(
        ["Round", "Selected", "Wait Turns"], sort=True
    ):
        row = _summarize(group)
        if not row:
            continue
        row = {
            "Round": int(round_no),
            "Selected QB": str(selected),
            "Wait Turns": int(wait_turns),
            **row,
        }
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["Round", "Unique Paths", "Avg Alt Delta"], ascending=[True, False, False]
    ).reset_index(drop=True)


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

    probe = FastDraftSimulator(engine)
    yahoo = getattr(probe, "yahoo_room_status", {}) or {}

    print("\nPatBot v0.6.2 QB timing confirmation screen")
    print("DIAGNOSTIC ONLY: no draft weights or lookahead rules change here.")
    print("Why this exists: the 60-room Yahoo-informed construction audit produced one clear WAIT LEAN: Round 7 QB when waiting two PatBot turns.")
    print("This reruns only Rounds 5-7 at 180 paired rooms and reports QB paths after collapsing duplicate challenger lenses.")
    print("Calibration branch remains Gibbs 1.01, Bijan 1.02, Ja'Marr Chase 1.03.\n")
    print(
        f"Yahoo room cache: {'ON' if yahoo.get('ok') else 'OFF'} | matched {yahoo.get('matched', 0)} | "
        f"coverage {yahoo.get('coverage_pct', 0)}% | age {yahoo.get('age_hours', '—')}h"
    )

    def progress(done: int, total: int):
        if done == total or done % 18 == 0:
            print(f"  {done:>3}/{total} rooms complete", flush=True)

    _, _, events, meta = run_pick_specific_path_audit(
        engine,
        opening_history,
        forced_first_player_id=str(chase["player_id"]),
        runs=180,
        through_round=13,
        audit_start_round=5,
        audit_through_round=7,
        wait_turns=(1, 2),
        challenger_pool=12,
        progress=progress,
    )

    unique = _unique_qb_paths(events)
    print("\n=== QB STRUCTURAL ROLLUP ===\n")
    rollup = _round_rollup(unique)
    if rollup.empty:
        print("No QB timing paths were captured.")
    else:
        print(rollup.to_string(index=False))

    print("\n=== BY SELECTED QB ===\n")
    by_qb = _player_rollup(unique)
    if by_qb.empty:
        print("No selected-QB breakdown was captured.")
    else:
        print(by_qb.to_string(index=False))

    print("\nDecision rule for the next step:")
    print("- If Round 7 / wait 2 remains WAIT LEAN with >=20 unique paths AND the effect appears across more than one selected QB, we will promote a generic two-turn sequence-lookahead capability rather than a hard-coded Round-7/QB rule.")
    print("- If the signal collapses or is driven by one selected QB, production construction stays unchanged.")
    print("- Rounds 5-6 are included as controls so we do not accidentally generalize a late-QB effect into an all-QB timing rule.")
    print(f"- Seed: {meta['seed']} | rooms: {meta['runs']} | Yahoo remains opponent-behavior only.\n")


if __name__ == "__main__":
    main()
