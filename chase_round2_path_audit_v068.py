from __future__ import annotations

from collections import Counter

import numpy as np

from patbot.config import load_config
from patbot.data import load_players
from patbot.draft import DraftEngine
from patbot.draft_state import make_pick_record
from patbot.sim import FastDraftSimulator
from patbot.team_concentration import candidate_concentration_penalty


RUNS = 1500
SEED = 20260829
TARGET_PICK = 22


def _row(players, name: str):
    match = players[players["name"].eq(name)]
    if match.empty:
        raise RuntimeError(f"Could not find {name!r} in live snapshot")
    return match.iloc[0]


def _pct(x: int, n: int) -> float:
    return 100.0 * float(x) / max(int(n), 1)


def main():
    cfg = load_config()
    players = load_players("data/players_2026_live.csv")
    players["player_id"] = players["player_id"].astype(str)
    engine = DraftEngine(players, cfg)
    sim = FastDraftSimulator(engine)
    teams = int(cfg["league"]["teams"])

    gibbs = _row(players, "Jahmyr Gibbs")
    bijan = _row(players, "Bijan Robinson")
    chase = _row(players, "Ja'Marr Chase")
    brown = _row(players, "Chase Brown")
    higgins = _row(players, "Tee Higgins")

    opening = [
        make_pick_record(1, teams, str(gibbs["player_id"]), gibbs["name"], gibbs["team"], gibbs["pos"]),
        make_pick_record(2, teams, str(bijan["player_id"]), bijan["name"], bijan["team"], bijan["pos"]),
        make_pick_record(3, teams, str(chase["player_id"]), chase["name"], chase["team"], chase["pos"]),
    ]

    gibbs_idx = sim.id_to_idx[str(gibbs["player_id"])]
    bijan_idx = sim.id_to_idx[str(bijan["player_id"])]
    chase_idx = sim.id_to_idx[str(chase["player_id"])]
    brown_idx = sim.id_to_idx[str(brown["player_id"])]
    higgins_idx = sim.id_to_idx[str(higgins["player_id"])]

    base_available = np.ones(sim.n, dtype=bool)
    base_available[[gibbs_idx, bijan_idx, chase_idx]] = False

    my_counts = np.zeros(len(sim.POSITIONS), dtype=np.int16)
    my_counts[sim.pos_to_code["WR"]] = 1

    rng = np.random.default_rng(SEED)
    latent_sd = np.maximum(sim.sd_floor, sim.adp * sim.sd_pct)

    brown_available = 0
    higgins_available = 0
    both_available = 0
    brown_top_adjusted = 0
    brown_top_raw = 0
    higgins_top_adjusted = 0
    best_non_cin_names = Counter()
    first_brown_taken_by = Counter()
    first_higgins_taken_by = Counter()
    brown_adjusted_edges = []
    brown_raw_edges = []
    higgins_adjusted_edges = []

    for _ in range(RUNS):
        available = base_available.copy()
        opp_counts = sim._seed_opponent_counts(opening)
        archetypes = sim._archetype_assignments(rng)
        market_latent = np.maximum(1.0, rng.normal(sim.adp, latent_sd))
        custom_noise_base = rng.normal(0.0, np.maximum(3.0, sim.custom_rank * 0.06))

        brown_taker = None
        higgins_taker = None
        for pick in range(4, TARGET_PICK):
            idx, _ = sim._take_opponent_pick(
                pick,
                available,
                opp_counts,
                archetypes,
                market_latent,
                custom_noise_base,
            )
            if idx == brown_idx and brown_taker is None:
                brown_taker = pick
            if idx == higgins_idx and higgins_taker is None:
                higgins_taker = pick

        if brown_taker is not None:
            first_brown_taken_by[brown_taker] += 1
        if higgins_taker is not None:
            first_higgins_taken_by[higgins_taker] += 1

        brown_is_available = bool(available[brown_idx])
        higgins_is_available = bool(available[higgins_idx])
        brown_available += int(brown_is_available)
        higgins_available += int(higgins_is_available)
        both_available += int(brown_is_available and higgins_is_available)

        scores = sim._patbot_score_vector(available, my_counts, TARGET_PICK)
        legal = available & (scores > -1e8)
        non_cin = legal & (sim.nfl_team != "CIN")
        if not non_cin.any():
            continue
        non_cin_idx = np.where(non_cin)[0]
        best_idx = int(non_cin_idx[np.argmax(scores[non_cin_idx])])
        best_score = float(scores[best_idx])
        best_non_cin_names[str(sim.names[best_idx])] += 1

        if brown_is_available:
            adjusted = float(scores[brown_idx])
            penalty, _ = candidate_concentration_penalty(
                sim.players,
                candidate_idx=brown_idx,
                roster_indices=[chase_idx],
                round_no=2,
                config=sim.cfg,
            )
            raw = adjusted + float(penalty)
            brown_adjusted_edges.append(adjusted - best_score)
            brown_raw_edges.append(raw - best_score)
            brown_top_adjusted += int(adjusted >= best_score)
            brown_top_raw += int(raw >= best_score)

        if higgins_is_available:
            adjusted = float(scores[higgins_idx])
            higgins_adjusted_edges.append(adjusted - best_score)
            higgins_top_adjusted += int(adjusted >= best_score)

    print("\nPatBot v0.6.8 Chase -> pick 22 path audit")
    print("State fixed: Gibbs 1.01, Bijan 1.02, Ja'Marr Chase 1.03; real-room model simulates picks 4-21.")
    print(f"Runs: {RUNS:,} | seed: {SEED} | target: pick {TARGET_PICK} (2.10)\n")

    print("Availability at 2.10:")
    print(f"  Chase Brown: {_pct(brown_available, RUNS):5.1f}%")
    print(f"  Tee Higgins: {_pct(higgins_available, RUNS):5.1f}%")
    print(f"  Both:        {_pct(both_available, RUNS):5.1f}%")

    if brown_available:
        adjusted = np.asarray(brown_adjusted_edges, dtype=float)
        raw = np.asarray(brown_raw_edges, dtype=float)
        print("\nChase Brown when available at 2.10:")
        print(f"  Beats best non-CIN alternative AFTER -1.75 concentration penalty: {_pct(brown_top_adjusted, brown_available):5.1f}%")
        print(f"  Would beat best non-CIN alternative BEFORE concentration penalty: {_pct(brown_top_raw, brown_available):5.1f}%")
        print(f"  Avg adjusted edge vs best non-CIN: {adjusted.mean():+6.2f} PatBot score")
        print(f"  Median adjusted edge:              {np.median(adjusted):+6.2f}")
        print(f"  Avg raw edge before penalty:       {raw.mean():+6.2f}")
        print(f"  P25/P75 adjusted edge:             {np.percentile(adjusted,25):+6.2f} / {np.percentile(adjusted,75):+6.2f}")

    if higgins_available:
        hedges = np.asarray(higgins_adjusted_edges, dtype=float)
        print("\nTee Higgins when available at 2.10:")
        print(f"  Beats best non-CIN alternative AFTER -3.50 concentration penalty: {_pct(higgins_top_adjusted, higgins_available):5.1f}%")
        print(f"  Avg adjusted edge vs best non-CIN: {hedges.mean():+6.2f} PatBot score")

    print("\nMost common best non-CIN alternatives at pick 22:")
    for name, count in best_non_cin_names.most_common(10):
        print(f"  {name:<28} {_pct(count, RUNS):5.1f}% of rooms")

    if first_brown_taken_by:
        print("\nMost common picks where Chase Brown leaves the room before 2.10:")
        for pick, count in first_brown_taken_by.most_common(8):
            print(f"  Pick {pick:>2}: {_pct(count, RUNS):5.1f}% of all rooms")

    print("\nInterpretation guide:")
    print("- The concentration penalty should change close Brown decisions, not erase a true value fall.")
    print("- If Brown still wins frequently after the penalty, PatBot is correctly willing to accept the same-team exposure for value.")
    print("- Higgins should face a materially higher hurdle because WR/WR overlap is more concentrated.")
    print("- This audit is diagnostic only; it does not alter production weights.\n")


if __name__ == "__main__":
    main()
