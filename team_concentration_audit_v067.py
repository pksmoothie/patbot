from __future__ import annotations

from patbot import __version__
from patbot.config import load_config
from patbot.data import load_players
from patbot.draft import DraftEngine
from patbot.sim import FastDraftSimulator
from patbot.team_concentration import candidate_concentration_penalty, roster_concentration_penalty


def _idx(players, name: str) -> int:
    hit = players.index[players["name"].eq(name)].tolist()
    if not hit:
        raise RuntimeError(f"Could not find {name!r} in live snapshot")
    return int(hit[0])


def main():
    cfg = load_config()
    players = load_players("data/players_2026_live.csv").reset_index(drop=True)
    players["player_id"] = players["player_id"].astype(str)
    engine = DraftEngine(players, cfg)
    sim = FastDraftSimulator(engine)

    chase = _idx(players, "Ja'Marr Chase")
    brown = _idx(players, "Chase Brown")
    higgins = _idx(players, "Tee Higgins")
    burrow = _idx(players, "Joe Burrow")

    print(f"\nPatBot v{__version__} same-team skill concentration audit")
    print("Generic production rule: QB combinations are exempt; non-QB same-team skill pairs receive a soft, round-sensitive penalty.\n")

    print("With Ja'Marr Chase already rostered:")
    print("Candidate        Pos   R2 penalty   R9 penalty   R12 penalty")
    print("----------------------------------------------------------")
    for name, idx in [("Chase Brown", brown), ("Tee Higgins", higgins), ("Joe Burrow", burrow)]:
        vals = []
        for rnd in (2, 9, 12):
            p, _ = candidate_concentration_penalty(
                players,
                candidate_idx=idx,
                roster_indices=[chase],
                round_no=rnd,
                config=cfg,
            )
            vals.append(p)
        print(f"{name:<16} {players.iloc[idx]['pos']:<4} {vals[0]:>10.2f}   {vals[1]:>10.2f}   {vals[2]:>11.2f}")

    print("\nFinal-roster lineup concentration examples:")
    examples = [
        ("Chase + Brown", [chase, brown]),
        ("Chase + Higgins", [chase, higgins]),
        ("Chase + Burrow", [chase, burrow]),
        ("Chase + Brown + Higgins", [chase, brown, higgins]),
    ]
    for label, mine in examples:
        penalty = roster_concentration_penalty(sim, mine)
        print(f"{label:<26} lineup penalty = {penalty:.2f}")

    print("\nInterpretation:")
    print("- QB/pass-catcher stacks remain unpenalized.")
    print("- RB/WR same-team exposure is a modest diversification cost, not a ban.")
    print("- WR/WR overlap is more expensive because the players compete more directly for the same passing volume.")
    print("- A third non-QB skill player from one offense escalates the concentration cost.")
    print("- Draft-score penalties fade in later rounds so exceptional value and bench handcuffs can still win.\n")


if __name__ == "__main__":
    main()
