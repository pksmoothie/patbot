import pandas as pd

from patbot.config import load_config
from patbot.data import load_players
from patbot.draft import DraftEngine
from patbot.draft_state import make_pick_record
from patbot.timing_audit import run_qb_te_timing_audit, run_representative_draft


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
    teams = int(cfg["league"]["teams"])

    gibbs = _row_by_name(players, "Jahmyr Gibbs")
    bijan = _row_by_name(players, "Bijan Robinson")
    chase = _row_by_name(players, "Ja'Marr Chase")

    # We know the first two picks for the scenario and deliberately lock Chase at
    # 1.03 so every downstream comparison answers the roster-construction question
    # we actually care about rather than reopening the already-audited 1.03 choice.
    opening_history = [
        make_pick_record(1, teams, str(gibbs["player_id"]), gibbs["name"], gibbs["team"], gibbs["pos"]),
        make_pick_record(2, teams, str(bijan["player_id"]), bijan["name"], bijan["team"], bijan["pos"]),
    ]

    print("\nPatBot v0.4.8 draft-path and QB/TE timing audit")
    print("Scenario: Gibbs 1.01, Bijan 1.02, Ja'Marr Chase locked at 1.03")
    print("The representative draft is one fixed-seed room for visibility, not a prediction.")
    print("The QB/TE timing section uses 300 paired rooms and compares fill-now vs RB/WR-and-wait-one-turn.\n")

    draft, early_board, next_two, final_roster, rep_meta = run_representative_draft(
        engine,
        opening_history,
        forced_first_player_id=str(chase["player_id"]),
        through_round=15,
        early_rounds=6,
        board_depth=6,
    )

    print("=== REPRESENTATIVE FULL DRAFT CONSTRUCTION ===\n")
    show_draft = draft[["Round", "Overall", "Player", "Pos", "Role", "PatBot Score", "LWS", "Q90"]]
    print(show_draft.to_string(index=False))

    print("\n=== REPRESENTATIVE FINAL ROSTER BY SLOT ===\n")
    print(final_roster.to_string(index=False))

    print("\n=== ROUNDS 1-6: PATBOT BOARD / WHO IT PASSED ===\n")
    board_view = early_board[[
        "Round", "Overall", "Board Rank", "Selected", "Player", "Pos",
        "Score", "VORP", "LWS", "Q90", "Pos Rank",
    ]].copy()
    board_view["Selected"] = board_view["Selected"].map({True: "YES", False: ""})
    print(board_view.to_string(index=False))

    print("\n=== ROUNDS 1-6: NEXT TWO PICKS AFTER PATBOT ===\n")
    print(next_two.to_string(index=False))
    print(f"\nRepresentative-room seed: {rep_meta['seed']}\n")

    print("=== QB/TE TIMING OPPORTUNITY-COST AUDIT ===\n")

    def progress(done: int, total: int):
        print(f"  {done:>3}/{total} rooms complete")

    quality, decisions, events, timing_meta = run_qb_te_timing_audit(
        engine,
        opening_history,
        forced_first_player_id=str(chase["player_id"]),
        runs=300,
        through_round=13,
        audit_through_round=10,
        progress=progress,
    )

    print("\n=== QB/TE TIMING BY QUALITY ===\n")
    if quality.empty:
        print("No QB1/TE1 timing events were captured.")
    else:
        print(quality.to_string(index=False))

    print("\n=== MOST COMMON QB/TE PICKS AND WHAT THEY PASSED ===\n")
    if decisions.empty:
        print("No QB1/TE1 decisions were captured.")
    else:
        print(decisions.head(40).to_string(index=False))

    print("\nHow to read this audit:")
    print("- 'RB/WR Passed' is the best currently legal RB/WR by PatBot score when QB1/TE1 was selected.")
    print("- 'VORP Cost of Waiting' is selected QB/TE VORP minus the best same-position option available at PatBot's next pick.")
    print("- Positive 'Wait Delta' means taking the RB/WR now, forcing the best QB/TE at the next PatBot turn, then drafting normally produced the better R13 roster on the same sampled football outcome.")
    print("- Elite/solid/weak are diagnostic VORP tiers only; this audit does not change strategy weights by itself.")
    print(f"- Timing seed: {timing_meta['seed']} across {timing_meta['runs']} rooms.\n")


if __name__ == "__main__":
    main()
