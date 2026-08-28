from __future__ import annotations

import itertools

import pandas as pd

from patbot.config import load_config
from patbot.data import load_players
from patbot.draft import DraftEngine
from patbot.draft_state import make_pick_record
from patbot.sim import FastDraftSimulator


CORE_TOP = [
    "Jahmyr Gibbs",
    "Bijan Robinson",
    "Ja'Marr Chase",
    "Puka Nacua",
    "Jaxon Smith-Njigba",
]
CANDIDATE_POOL = CORE_TOP + [
    "Amon-Ra St. Brown",
    "Christian McCaffrey",
    "Jonathan Taylor",
]

# Order matters because pick 1 belongs to Paul and pick 2 belongs to Faherty;
# their roster state and manager tendencies affect later room behavior. Test both
# orders for every pair among the five most plausible names, plus both orders for
# a few representative surprise boards involving Amon-Ra.
SCENARIOS = list(itertools.permutations(CORE_TOP, 2))
for other in ("Jahmyr Gibbs", "Bijan Robinson", "Jaxon Smith-Njigba"):
    SCENARIOS.extend(itertools.permutations((other, "Amon-Ra St. Brown"), 2))


def _row_by_name(players: pd.DataFrame, name: str) -> pd.Series:
    match = players[players["name"].eq(name)]
    if match.empty:
        raise RuntimeError(f"Could not find {name!r} in the live player snapshot.")
    return match.iloc[0]


def _opening_history(players: pd.DataFrame, cfg: dict, gone: tuple[str, str]) -> list[dict]:
    teams = int(cfg["league"]["teams"])
    rows = []
    for overall, name in enumerate(gone, start=1):
        player = _row_by_name(players, name)
        rows.append(
            make_pick_record(
                overall,
                teams,
                str(player["player_id"]),
                player["name"],
                player["team"],
                player["pos"],
            )
        )
    return rows


def run_conditional_103(players: pd.DataFrame, cfg: dict, runs: int = 300) -> pd.DataFrame:
    engine = DraftEngine(players, cfg)
    sim = FastDraftSimulator(engine)
    seed = int(cfg.get("simulation", {}).get("comparison_seed", 20260818)) + 103
    rows = []

    for gone in SCENARIOS:
        history = _opening_history(players, cfg, gone)
        drafted_ids = {str(item["player_id"]) for item in history}
        available_candidates = [name for name in CANDIDATE_POOL if name not in gone]

        scenario_rows = []
        for name in available_candidates:
            player = _row_by_name(players, name)
            result = sim.simulate_candidate(
                current_pick=3,
                drafted_ids=drafted_ids,
                my_roster_ids=[],
                candidate_id=str(player["player_id"]),
                runs=int(runs),
                through_round=8,
                seed=seed,
                draft_history=history,
            )
            scenario_rows.append(
                {
                    "Pick 1": gone[0],
                    "Pick 2": gone[1],
                    "Candidate": name,
                    "Avg Lineup": result["avg_lineup_score"],
                    "P10": result["p10_lineup_score"],
                    "P25": result["p25_lineup_score"],
                    "P75": result["p75_lineup_score"],
                    "P90": result["p90_lineup_score"],
                    "Starter VORP": result["avg_starter_vorp"],
                    "LWS": result["candidate_league_winner_score"],
                    "Bust -25%": result["candidate_bust_pct"],
                    "Cat Tail": result["candidate_catastrophic_pct"],
                }
            )

        frame = pd.DataFrame(scenario_rows).sort_values("Avg Lineup", ascending=False).reset_index(drop=True)
        if not frame.empty:
            leader = float(frame.loc[0, "Avg Lineup"])
            frame["Delta vs Best"] = (frame["Avg Lineup"] - leader).round(2)
            frame["Scenario Rank"] = range(1, len(frame) + 1)
            rows.extend(frame.to_dict("records"))

    return pd.DataFrame(rows)


def _scenario_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pick1, pick2), group in results.groupby(["Pick 1", "Pick 2"], sort=False):
        ordered = group.sort_values("Scenario Rank")
        best = ordered.iloc[0]
        second = ordered.iloc[1] if len(ordered) > 1 else None
        gap = float(best["Avg Lineup"] - second["Avg Lineup"]) if second is not None else 0.0
        rows.append(
            {
                "Pick 1": pick1,
                "Pick 2": pick2,
                "PatBot 1.03": best["Candidate"],
                "Runner-up": second["Candidate"] if second is not None else "—",
                "Mean Edge": round(gap, 2),
                "Best P10": best["P10"],
                "Best P90": best["P90"],
                "Close?": "YES" if gap < 3.0 else "",
            }
        )
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    players = load_players("data/players_2026_live.csv")
    players["player_id"] = players["player_id"].astype(str)

    print("\nPatBot v0.5.0 conditional 1.03 decision-tree audit")
    print("This answers what PatBot should do if Paul/Faherty do NOT take Gibbs/Bijan as modeled.")
    print(f"{len(SCENARIOS)} ordered top-two scenarios; 300 paired runs per candidate per scenario.")
    print("This is a screen, not the final 1,500-run lock for close cases.\n")

    results = run_conditional_103(players, cfg, runs=300)
    summary = _scenario_summary(results)

    print("=== CONDITIONAL 1.03 TREE ===\n")
    print(summary.to_string(index=False))

    print("\n=== FULL CANDIDATE RESULTS BY TOP-TWO SCENARIO ===\n")
    cols = [
        "Pick 1", "Pick 2", "Scenario Rank", "Candidate", "Avg Lineup", "Delta vs Best",
        "P10", "P25", "P75", "P90", "Starter VORP", "LWS", "Bust -25%", "Cat Tail",
    ]
    print(results[cols].to_string(index=False))

    expected = results[
        ((results["Pick 1"] == "Jahmyr Gibbs") & (results["Pick 2"] == "Bijan Robinson"))
        | ((results["Pick 1"] == "Bijan Robinson") & (results["Pick 2"] == "Jahmyr Gibbs"))
    ].sort_values(["Pick 1", "Scenario Rank"])
    if not expected.empty:
        print("\n=== GIBBS + BIJAN ORDER-SENSITIVITY ANCHOR ===\n")
        print(expected[cols].to_string(index=False))

    close = summary[summary["Close?"].eq("YES")]
    print("\n=== SCENARIOS THAT NEED A 1,500-RUN FINAL IF THEY OCCUR ===\n")
    if close.empty:
        print("No scenario had a top-two mean gap under 3.0 points in this screen.")
    else:
        print(close.to_string(index=False))

    print("\nNotes:")
    print("- Chase is not hard-coded here. The expected Gibbs/Bijan branches should reproduce the current Chase preference if the model remains stable.")
    print("- Pick order is explicitly modeled because Paul at 1.01 and Faherty at 1.02 have different tendencies and roster states.")
    print("- The extra Amon-Ra branches provide concrete surprise-top-two cases; no generic 'surprise player' is inserted into the model.")
    print("- Any close or unexpected branch should be rerun at 1,500 paired simulations before we treat it as a draft-day rule.\n")


if __name__ == "__main__":
    main()
