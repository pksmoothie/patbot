import pandas as pd

from patbot.config import load_config
from patbot.data import load_players
from patbot.draft import DraftEngine
from patbot.draft_state import make_pick_record
from patbot.late_round import run_late_round_audit


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

    print("\nPatBot late-round championship-strategy audit")
    print("Anchor: Gibbs 1.01, Bijan 1.02, Ja'Marr Chase 1.03")
    print("Auditing Rounds 8, 10, 12 and 13. Round 14 is reserved for D/ST; Round 15 for kicker.")
    print("500 simulated rooms. Progress will print below so it does not look frozen.\n")

    def progress(done: int, total: int):
        print(f"  {done:>3}/{total} rooms complete")

    summary, details, meta = run_late_round_audit(
        engine,
        history,
        rounds=(8, 10, 12, 13),
        runs=500,
        progress=progress,
    )

    print("\n=== ROUND-BY-ROUND SUMMARY ===\n")
    print(summary.to_string(index=False))

    print("\n=== PLAYERS MOST AFFECTED BY THE CHAMPIONSHIP STRATEGY ===\n")
    if details.empty:
        print("No player appeared often enough to report.")
    else:
        print(details.to_string(index=False))

    warnings = details[
        details.get("Youth-Heavy Warning", pd.Series(False, index=details.index)).fillna(False)
        & (pd.to_numeric(details.get("PatBot Pick %", 0), errors="coerce").fillna(0) >= 5.0)
    ] if not details.empty else details

    print("\n=== YOUTH-HEAVY CHECK ===\n")
    if warnings.empty:
        print("PASS: no player selected in at least 5% of rooms is being driven primarily by the youth component.")
    else:
        cols = ["Round", "Player", "Pos", "PatBot Pick %", "League Winner Score", "Ceiling Score", "Market Edge", "Early Career", "Youth Component Share %"]
        print("REVIEW: these frequently selected players have unusually youth-heavy upside signals:")
        print(warnings[cols].to_string(index=False))

    print("\nInterpretation: 'Strategy Lift pp' is how much more often the round-aware championship model selects a player than a Foundation-style model on the exact same board and roster.")
    print("A late-round target should ideally combine ceiling and/or market edge with youth optionality, rather than score well only because he is young.\n")


if __name__ == "__main__":
    main()
