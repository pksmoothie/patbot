import pandas as pd

from patbot.config import load_config
from patbot.construction_audit import run_construction_audit
from patbot.data import load_players
from patbot.draft import DraftEngine
from patbot.draft_state import make_pick_record


def _row_by_name(players: pd.DataFrame, name: str) -> pd.Series:
    match = players[players["name"].eq(name)]
    if match.empty:
        raise RuntimeError(f"Could not find {name!r} in the live player snapshot.")
    return match.iloc[0]


def main():
    cfg = load_config()
    players = load_players("data/players_2026_live.csv")
    players["player_id"] = players["player_id"].astype(str)
    engine = DraftEngine(players, cfg)

    gibbs = _row_by_name(players, "Jahmyr Gibbs")
    bijan = _row_by_name(players, "Bijan Robinson")
    chase = _row_by_name(players, "Ja'Marr Chase")
    teams = int(cfg["league"]["teams"])

    history = [
        make_pick_record(1, teams, str(gibbs["player_id"]), gibbs["name"], gibbs["team"], gibbs["pos"]),
        make_pick_record(2, teams, str(bijan["player_id"]), bijan["name"], bijan["team"], bijan["pos"]),
        make_pick_record(3, teams, str(chase["player_id"]), chase["name"], chase["team"], chase["pos"]),
    ]

    print("\nPatBot v0.4.6 value-aware roster construction audit")
    print("Anchor: Gibbs 1.01, Bijan 1.02, Ja'Marr Chase 1.03")
    print("Auditing Rounds 8, 10, 12 and 13 with the production R2/R3 lookahead path.")
    print("Counterfactual: same simulated room, but PatBot must fill missing offensive starters before bench depth.")
    print("500 paired rooms. Progress will print below.\n")

    def progress(done: int, total: int):
        print(f"  {done:>3}/{total} rooms complete")

    summary, deferrals, comparison, meta = run_construction_audit(
        engine,
        history,
        rounds=(8, 10, 12, 13),
        runs=500,
        progress=progress,
    )

    print("\n=== ROSTER STATE AT PATBOT PICKS ===\n")
    print(summary.to_string(index=False))

    print("\n=== VALUE-AWARE VS RIGID STARTER-FIRST ===\n")
    print(comparison.to_string(index=False))

    print("\n=== BENCH RB/WR PICKS THAT DEFERRED A BASE STARTER ===\n")
    if deferrals.empty:
        print("No starter-deferring RB/WR picks occurred in the 500 rooms.")
    else:
        print(deferrals.head(30).to_string(index=False))

    qb2 = float(pd.to_numeric(summary.get("QB2 Pick %", 0), errors="coerce").fillna(0).max()) if not summary.empty else 0.0
    invalid = float(pd.to_numeric(summary.get("Post-Complete Invalid %", 0), errors="coerce").fillna(0).max()) if not summary.empty else 0.0

    print("\n=== HARD-CONSTRAINT CHECKS ===\n")
    if qb2 == 0.0:
        print("PASS: QB2 was never drafted at any audited checkpoint.")
    else:
        print(f"REVIEW: QB2 appeared in up to {qb2:.1f}% of audited rooms.")

    if invalid == 0.0:
        print("PASS: once the offensive lineup was complete, no QB/TE was drafted; offensive picks stayed RB/WR.")
    else:
        print(f"REVIEW: post-completion non-RB/WR offensive picks reached {invalid:.1f}%.")

    print("\nHow to read the policy comparison:")
    print("- Positive Value-Aware Delta means allowing bench value before starter completion produced the better eventual R13 roster on the same sampled football outcome.")
    print("- 'Immediate Score Edge' compares the chosen bench RB/WR with the best still-missing base starter on that exact board. Negative values can still be rational when R2/R3 lookahead sees a better two-pick path.")
    print("- The starter-first branch is intentionally rigid and exists only as a benchmark, not as a proposed strategy.\n")


if __name__ == "__main__":
    main()
