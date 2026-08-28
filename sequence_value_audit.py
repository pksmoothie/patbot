import pandas as pd

from patbot.config import load_config
from patbot.data import load_players
from patbot.draft import DraftEngine
from patbot.draft_state import make_pick_record
from patbot.sequence_audit import run_sequence_challenge_audit
from patbot.timing_audit import run_representative_draft


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

    opening_history = [
        make_pick_record(1, teams, str(gibbs["player_id"]), gibbs["name"], gibbs["team"], gibbs["pos"]),
        make_pick_record(2, teams, str(bijan["player_id"]), bijan["name"], bijan["team"], bijan["pos"]),
    ]

    print("\nPatBot v0.4.9 sequence-value audit")
    print("Scenario: Gibbs 1.01, Bijan 1.02, Ja'Marr Chase locked at 1.03")
    print("DIAGNOSTIC ONLY: no production weights or draft rules are changed by this script.")
    print("QB1/TE1 picks are challenged by RB/WR options selected three neutral ways:")
    print("  1) best current PatBot score, 2) highest LWS, 3) highest Q90")
    print("LWS/Q90 challengers come from the top 12 legal RB/WR by current score.")
    print("Each challenger is tested with the QB/TE fill delayed 1, 2 and 3 PatBot turns.\n")

    draft, early_board, next_two, final_roster, rep_meta = run_representative_draft(
        engine,
        opening_history,
        forced_first_player_id=str(chase["player_id"]),
        through_round=15,
        early_rounds=6,
        board_depth=10,
    )

    print("=== REPRESENTATIVE FULL DRAFT CONSTRUCTION ===\n")
    print(draft[["Round", "Overall", "Player", "Pos", "Role", "PatBot Score", "LWS", "Q90"]].to_string(index=False))

    print("\n=== REPRESENTATIVE FINAL ROSTER BY SLOT ===\n")
    print(final_roster.to_string(index=False))

    print("\n=== ROUNDS 1-6: TOP 10 PATBOT BOARD / WHO IT PASSED ===\n")
    board_view = early_board[[
        "Round", "Overall", "Board Rank", "Selected", "Player", "Pos",
        "Score", "VORP", "LWS", "Q90", "Pos Rank",
    ]].copy()
    board_view["Selected"] = board_view["Selected"].map({True: "YES", False: ""})
    print(board_view.to_string(index=False))

    print("\n=== ROUNDS 1-6: NEXT TWO PICKS AFTER PATBOT ===\n")
    print(next_two.to_string(index=False))
    print(f"\nRepresentative-room seed: {rep_meta['seed']}\n")

    print("=== MULTI-HORIZON QB/TE SEQUENCE CHALLENGE ===\n")

    def progress(done: int, total: int):
        print(f"  {done:>3}/{total} rooms complete")

    summary, common, events, meta = run_sequence_challenge_audit(
        engine,
        opening_history,
        forced_first_player_id=str(chase["player_id"]),
        runs=150,
        through_round=13,
        audit_through_round=8,
        wait_turns=(1, 2, 3),
        challenger_pool=12,
        progress=progress,
    )

    print("\n=== SEQUENCE RESULTS BY QB/TE QUALITY / CHALLENGER / WAIT ===\n")
    if summary.empty:
        print("No QB1/TE1 challenge events were captured.")
    else:
        print(summary.to_string(index=False))

    print("\n=== MOST COMMON SPECIFIC PATH CHALLENGES ===\n")
    if common.empty:
        print("No specific path challenges were captured.")
    else:
        print(common.head(60).to_string(index=False))

    if not events.empty:
        display_cols = [
            "Round", "Selected", "Pos", "Quality", "Selected Pos Rank",
            "Selected FP ECR", "Challenger Type", "Challenger", "Challenger Pos",
            "Challenger RB/WR Score Rank", "Challenger FP ECR", "Challenger LWS",
            "Challenger Q90", "Challenger Score Gap", "Wait Turns", "Wait Fill",
            "Wait Fill Pos Rank", "Wait Fill FP ECR", "VORP Cost of Waiting",
            "Alt Delta vs Fill Now",
        ]
        display_cols = [c for c in display_cols if c in events.columns]

        print("\n=== BIGGEST CASES WHERE RB/WR + WAIT BEAT FILL-NOW ===\n")
        print(events.sort_values("Alt Delta vs Fill Now", ascending=False)[display_cols].head(40).to_string(index=False))

        print("\n=== BIGGEST CASES WHERE FILL-NOW BEAT RB/WR + WAIT ===\n")
        print(events.sort_values("Alt Delta vs Fill Now", ascending=True)[display_cols].head(40).to_string(index=False))

        disagreement = events[
            events["Challenger Type"].isin(["LWS", "Q90"])
            & (events["Challenger RB/WR Score Rank"] > 1)
        ].copy()
        if not disagreement.empty:
            disagreement = disagreement.sort_values(
                ["Alt Delta vs Fill Now", "Challenger LWS"],
                ascending=[False, False],
            )
            print("\n=== UPSIDE CHALLENGERS CURRENT SCORE DID NOT RANK #1 ===\n")
            print(disagreement[display_cols].head(50).to_string(index=False))

    print("\nHow to read this audit:")
    print("- Positive Alt Delta means the RB/WR-first path produced the better eventual R13 roster on the same sampled football outcome.")
    print("- Wait Turns = 1/2/3 means the missing QB/TE is deliberately delayed until that future PatBot selection; interim PatBot picks cannot use that position.")
    print("- Challenger Score Gap is fill-now player's current PatBot score minus the challenger score. A positive number means PatBot currently prefers the QB/TE immediately.")
    print("- FP ECR is displayed when the FantasyPros ECR column exists in the local snapshot. It is diagnostic context only, not a new model weight.")
    print("- This is specifically designed to prevent us from hard-coding a Hall/Warren opinion: every QB1/TE1 decision gets the same challenge framework.")
    print(f"- Seed: {meta['seed']} • rooms: {meta['runs']} • challenger pool: top {meta['challenger_pool']} RB/WR by score.\n")


if __name__ == "__main__":
    main()
