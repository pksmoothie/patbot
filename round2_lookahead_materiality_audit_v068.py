from __future__ import annotations

from collections import Counter

import numpy as np

from patbot.config import load_config
from patbot.data import load_players
from patbot.draft import DraftEngine
from patbot.draft_state import make_pick_record
from patbot.sim import FastDraftSimulator


RUNS = 1500
SEED = 20260831
PICK_22 = 22
PICK_27 = 27
GATES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]


def _row(players, name: str):
    match = players[players["name"].eq(name)]
    if match.empty:
        raise RuntimeError(f"Could not find {name!r} in live snapshot")
    return match.iloc[0]


def _pct(n: int, d: int) -> float:
    return 100.0 * float(n) / max(int(d), 1)


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
    opening = [
        make_pick_record(1, teams, str(gibbs["player_id"]), gibbs["name"], gibbs["team"], gibbs["pos"]),
        make_pick_record(2, teams, str(bijan["player_id"]), bijan["name"], bijan["team"], bijan["pos"]),
        make_pick_record(3, teams, str(chase["player_id"]), chase["name"], chase["team"], chase["pos"]),
    ]

    gibbs_idx = sim.id_to_idx[str(gibbs["player_id"])]
    bijan_idx = sim.id_to_idx[str(bijan["player_id"])]
    chase_idx = sim.id_to_idx[str(chase["player_id"])]

    base_available = np.ones(sim.n, dtype=bool)
    base_available[[gibbs_idx, bijan_idx, chase_idx]] = False
    base_counts = np.zeros(len(sim.POSITIONS), dtype=np.int16)
    base_counts[sim.pos_to_code["WR"]] = 1

    rng = np.random.default_rng(SEED)
    latent_sd = np.maximum(sim.sd_floor, sim.adp * sim.sd_pct)
    branch_width = int(sim.lookahead_branch_width)
    future_weight = float(sim.lookahead_future_weight)
    vorp_weight = float(sim.lookahead_vorp_weight)

    overturn_edges = []
    current_sacrifices = []
    future_gains = []
    pair_vorp_gains = []
    transitions = Counter()
    player_transitions = Counter()
    gate_counts = Counter()
    greedy_names = Counter()
    lookahead_names = Counter()

    for _ in range(RUNS):
        available = base_available.copy()
        opp_counts = sim._seed_opponent_counts(opening)
        archetypes = sim._archetype_assignments(rng)
        market_latent = np.maximum(1.0, rng.normal(sim.adp, latent_sd))
        custom_noise_base = rng.normal(0.0, np.maximum(3.0, sim.custom_rank * 0.06))

        for pick in range(4, PICK_22):
            sim._take_opponent_pick(
                pick, available, opp_counts, archetypes, market_latent, custom_noise_base
            )

        current_scores = sim._patbot_score_vector(available, base_counts, PICK_22)
        legal = np.where(available & (current_scores > -1e8))[0]
        if legal.size == 0:
            continue
        ordered = legal[np.argsort(current_scores[legal])[::-1]]
        candidates = ordered[:branch_width]
        greedy_idx = int(candidates[0])
        greedy_name = str(sim.names[greedy_idx])
        greedy_names[greedy_name] += 1

        base_patbot = set(getattr(sim, "_patbot_owned_idxs", set())) | {chase_idx}
        base_opponents = set(getattr(sim, "_opponent_owned_idxs", set()))
        results = []

        for candidate in candidates:
            candidate = int(candidate)
            sim._patbot_owned_idxs = set(base_patbot) | {candidate}
            sim._opponent_owned_idxs = set(base_opponents)

            branch_available = available.copy()
            branch_counts = base_counts.copy()
            branch_opp_counts = opp_counts.copy()
            branch_available[candidate] = False
            code = sim.pos_code[candidate]
            if code >= 0:
                branch_counts[code] += 1

            for future_pick in range(PICK_22 + 1, PICK_27):
                sim._take_opponent_pick(
                    future_pick,
                    branch_available,
                    branch_opp_counts,
                    archetypes,
                    market_latent,
                    custom_noise_base,
                )

            future_scores = sim._patbot_score_vector(branch_available, branch_counts, PICK_27)
            future_legal = np.where(branch_available & (future_scores > -1e8))[0]
            if future_legal.size:
                future_idx = int(future_legal[np.argmax(future_scores[future_legal])])
                future_score = float(future_scores[future_idx])
                pair_vorp = max(float(sim.vorp[candidate]), 0.0) + max(float(sim.vorp[future_idx]), 0.0)
            else:
                future_idx = None
                future_score = 0.0
                pair_vorp = max(float(sim.vorp[candidate]), 0.0)

            path = float(current_scores[candidate]) + future_weight * future_score + vorp_weight * pair_vorp
            results.append((path, candidate, future_idx, future_score, pair_vorp))

        results.sort(key=lambda x: x[0], reverse=True)
        winner = results[0]
        greedy = next(x for x in results if int(x[1]) == greedy_idx)
        winner_idx = int(winner[1])
        lookahead_names[str(sim.names[winner_idx])] += 1

        edge = float(winner[0] - greedy[0])
        if winner_idx != greedy_idx:
            overturn_edges.append(edge)
            current_sacrifices.append(float(current_scores[winner_idx] - current_scores[greedy_idx]))
            future_gains.append(float(winner[3] - greedy[3]))
            pair_vorp_gains.append(float(winner[4] - greedy[4]))
            transitions[(str(sim.pos[greedy_idx]), str(sim.pos[winner_idx]))] += 1
            player_transitions[(str(sim.names[greedy_idx]), str(sim.names[winner_idx]))] += 1
            for gate in GATES:
                if edge >= gate:
                    gate_counts[gate] += 1

        sim._patbot_owned_idxs = set(base_patbot)
        sim._opponent_owned_idxs = set(base_opponents)

    n_overturn = len(overturn_edges)
    print("\nPatBot v0.6.8 Round-2 lookahead materiality audit")
    print("State: Gibbs 1.01, Bijan 1.02, Ja'Marr Chase 1.03; room simulated to 2.10.")
    print(f"Runs: {RUNS:,} | seed: {SEED} | production branch width: {branch_width}")
    print("Bye weeks: zero weight. Same-team concentration: production-active.\n")

    print(f"Raw production lookahead overturns: {n_overturn:,}/{RUNS:,} ({_pct(n_overturn, RUNS):.1f}%)")
    if n_overturn:
        e = np.asarray(overturn_edges)
        c = np.asarray(current_sacrifices)
        f = np.asarray(future_gains)
        v = np.asarray(pair_vorp_gains)
        print("\nAmong actual overturns:")
        print(f"  Path edge vs greedy: mean {e.mean():+.2f} | P25 {np.percentile(e,25):+.2f} | median {np.median(e):+.2f} | P75 {np.percentile(e,75):+.2f} | P90 {np.percentile(e,90):+.2f}")
        print(f"  Immediate-score change: mean {c.mean():+.2f} | median {np.median(c):+.2f}")
        print(f"  3.03 return-score change: mean {f.mean():+.2f} | median {np.median(f):+.2f}")
        print(f"  Pair-VORP change: mean {v.mean():+.2f} | median {np.median(v):+.2f}")

    print("\nHypothetical minimum path-edge gate (diagnostic only):")
    for gate in GATES:
        kept = gate_counts[gate]
        print(f"  >= {gate:>3.1f}: keeps {kept:>4}/{n_overturn:<4} overturns ({_pct(kept,n_overturn):5.1f}% of overturns; {_pct(kept,RUNS):5.1f}% of all rooms)")

    print("\nMost common position changes when lookahead overturns greedy:")
    for (a, b), count in transitions.most_common(10):
        print(f"  {a:>2} -> {b:<2}: {count:4} ({_pct(count,n_overturn):5.1f}% of overturns)")

    print("\nMost common player-level overturns:")
    for (a, b), count in player_transitions.most_common(12):
        print(f"  {a:<24} -> {b:<24} {count:4} ({_pct(count,n_overturn):5.1f}% of overturns)")

    print("\nMost common greedy leaders at 2.10:")
    for name, count in greedy_names.most_common(10):
        print(f"  {name:<28} {_pct(count,RUNS):5.1f}%")

    print("\nMost common lookahead selections at 2.10:")
    for name, count in lookahead_names.most_common(10):
        print(f"  {name:<28} {_pct(count,RUNS):5.1f}%")

    print("\nInterpretation guide:")
    print("- Do not promote a gate merely because lookahead changes picks often; inspect the edge distribution and what creates the edge.")
    print("- A useful lookahead may willingly give up a little immediate score for a materially better 3.03 return.")
    print("- If most overturns are sub-1-point path edges, a stability gate is worth testing. If edges are materially larger and driven by return value, leave production alone.")
    print("- This audit changes nothing in production.\n")


if __name__ == "__main__":
    main()
