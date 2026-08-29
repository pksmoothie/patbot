from __future__ import annotations

from patbot import __version__
from patbot.config import load_config
from patbot.data import load_players
from patbot.draft import DraftEngine
from patbot.draft_state import make_pick_record
from patbot.final_call import run_final_call


def _row(players, name: str):
    match = players[players["name"].eq(name)]
    if match.empty:
        raise RuntimeError(f"Could not find {name!r} in live snapshot")
    return match.iloc[0]


def main():
    cfg = load_config()
    players = load_players("data/players_2026_live.csv")
    players["player_id"] = players["player_id"].astype(str)
    engine = DraftEngine(players, cfg)
    teams = int(cfg["league"]["teams"])

    gibbs = _row(players, "Jahmyr Gibbs")
    bijan = _row(players, "Bijan Robinson")
    opening = [
        make_pick_record(1, teams, str(gibbs["player_id"]), gibbs["name"], gibbs["team"], gibbs["pos"]),
        make_pick_record(2, teams, str(bijan["player_id"]), bijan["name"], bijan["team"], bijan["pos"]),
    ]
    drafted = {str(gibbs["player_id"]), str(bijan["player_id"])}
    board = engine.recommend(
        current_pick=3,
        drafted_ids=drafted,
        roster_positions=[],
        top_n=18,
    )

    print(f"\nPatBot v{__version__} automatic Final Call live smoke")
    print("Calibration state: Gibbs 1.01, Bijan 1.02, PatBot on the clock at 1.03.\n")
    print("Base score board top 6:")
    print(board[["name", "pos", "score", "proj_points", "vorp", "adp"]].head(6).to_string(index=False))

    result = run_final_call(
        engine,
        current_pick=3,
        drafted_ids=drafted,
        my_roster_ids=[],
        board=board,
        draft_history=opening,
    )

    print("\n=== AUTOMATIC FINAL CALL ===\n")
    print(f"Recommendation: {result.get('recommendation')}")
    print(f"Base leader:    {result.get('base_winner')}")
    print(f"Base agrees:    {result.get('base_agrees')}")
    print(f"Stage:          {result.get('stage')}")
    print(f"Runs:           {result.get('runs')}")
    print(f"Horizon:        Round {result.get('through_round')}")
    print(f"Room-sim edge:  {result.get('edge')} ({result.get('edge_label')})")
    print(f"Elapsed:        {result.get('elapsed_seconds')}s")
    print(f"Reason:         {result.get('reason')}")

    summary = result.get("summary")
    if summary is not None and not summary.empty:
        print("\nFinal comparison table:")
        cols = [
            c for c in [
                "Candidate", "Avg Lineup Score", "10th %ile", "25th %ile",
                "75th %ile", "90th %ile", "League Winner Score", "Runs",
            ] if c in summary.columns
        ]
        print(summary[cols].to_string(index=False))

    if result.get("fallback"):
        raise SystemExit("Final Call fell back to base board; inspect the reason above.")
    print("\nSMOKE: Final Call completed normally.\n")


if __name__ == "__main__":
    main()
