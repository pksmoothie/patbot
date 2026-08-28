import pandas as pd

from patbot.config import load_config
from patbot.data import load_players
from patbot.draft import DraftEngine
from patbot.draft_state import make_pick_record
from patbot.path_value import run_pick_specific_path_audit


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

    print("\nPatBot v0.5.8 pick-specific path-value / construction audit")
    print("DIAGNOSTIC ONLY: this does not change production urgency, roster-fit, scarcity, or draft weights.")
    print("Calibration branch: Gibbs 1.01, Bijan 1.02, Ja'Marr Chase locked at 1.03.")
    print("Rounds 2-7 are challenged by cross-position alternatives selected by Score, VORP, FantasyPros ECR, LWS and Q90.")
    print("Each alternative is tested with the production pick's position delayed 1 or 2 PatBot turns, then both paths finish normally through Round 13.")
    print("60 paired rooms are used for this screen so it should be materially lighter than the 26-scenario 1.03 audit.\n")

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

    print("\n=== PATH VALUE BY ROUND / SELECTED POSITION / CHALLENGER / WAIT ===\n")
    if summary.empty:
        print("No path-value events were captured.")
    else:
        print(summary.to_string(index=False))

    print("\n=== MOST COMMON SPECIFIC PATH CHALLENGES ===\n")
    if common.empty:
        print("No specific paths were captured.")
    else:
        print(common.head(60).to_string(index=False))

    if not events.empty:
        cols = [
            "Round", "Selected", "Selected Pos", "Role", "Selected Pos Rank",
            "Selected FP ECR", "Challenger Type", "Challenger", "Challenger Pos",
            "Cross-Pos Score Rank", "Challenger FP ECR", "Current Score Gap",
            "Wait Turns", "Wait Fill", "Wait Fill Pos Rank", "Wait Fill FP ECR",
            "Selected Survived", "VORP Cost Waiting", "ECR Cost Waiting",
            "LWS Cost Waiting", "Q90 Cost Waiting", "Alt Delta vs Take Now",
        ]
        cols = [c for c in cols if c in events.columns]

        print("\n=== BIGGEST CASES WHERE CROSS-POSITION VALUE + WAIT WON ===\n")
        print(events.sort_values("Alt Delta vs Take Now", ascending=False)[cols].head(50).to_string(index=False))

        print("\n=== BIGGEST CASES WHERE TAKE-NOW WON ===\n")
        print(events.sort_values("Alt Delta vs Take Now", ascending=True)[cols].head(50).to_string(index=False))

    print("\nHow to read this audit:")
    print("- Positive Alt Delta means taking the cross-position challenger first and returning to the selected position later produced the better eventual R13 lineup on the same sampled football outcome.")
    print("- Selected Survives % is the empirical room-model chance that the exact production player is still available at the forced future fill turn.")
    print("- VORP/LWS/Q90 Cost Waiting compare the production player with the actual same-position fill that came back. Positive means waiting lost value on that metric.")
    print("- ECR Cost Waiting is positive when the future fill has a worse (higher) FantasyPros ECR than the production player.")
    print("- The five challenger lenses are deliberately separate so current PatBot Score does not get to define every counterfactual.")
    print("- This is still a calibration screen. We will only change production construction logic if the same pattern is broad and repeatable, not because one named player wins a row.")
    print(f"- Seed: {meta['seed']} • rooms: {meta['runs']} • challenger pool: top {meta['challenger_pool']} legal cross-position players by current score.\n")


if __name__ == "__main__":
    main()
