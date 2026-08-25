import pandas as pd

from patbot.config import load_config
from patbot.data import load_players
from patbot.diagnostics import run_off_field_sensitivity
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
    puka = _row_by_name(players, "Puka Nacua")

    history = [
        make_pick_record(1, int(cfg["league"]["teams"]), str(gibbs["player_id"]), gibbs["name"], gibbs["team"], gibbs["pos"]),
        make_pick_record(2, int(cfg["league"]["teams"]), str(bijan["player_id"]), bijan["name"], bijan["team"], bijan["pos"]),
    ]

    table, _ = run_off_field_sensitivity(
        engine,
        current_pick=3,
        drafted_ids={str(gibbs["player_id"]), str(bijan["player_id"])},
        my_roster_ids=[],
        candidate_id=str(puka["player_id"]),
        runs=1000,
        through_round=int(cfg.get("simulation", {}).get("through_round", 8)),
        draft_history=history,
    )

    print("\nPuka Nacua — paired off-field sensitivity (Scenario A: Gibbs 1.01, Bijan 1.02)\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
