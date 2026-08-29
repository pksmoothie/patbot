from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from patbot.config import load_config
from patbot.data import load_players
from patbot.draft import DraftEngine
from patbot.draft_state import make_pick_record
from patbot.sim import FastDraftSimulator


RUNS = 1500
SEED = 20260830
PICK_22 = 22
PICK_27 = 27


def _row(players, name: str):
    match = players[players["name"].eq(name)]
    if match.empty:
        raise RuntimeError(f"Could not find {name!r} in live snapshot")
    return match.iloc[0]


def _pct(x: int, n: int) -> float:
    return 100.0 * float(x) / max(int(n), 1)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


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

    base_my_counts = np.zeros(len(sim.POSITIONS), dtype=np.int16)
    base_my_counts[sim.pos_to_code["WR"]] = 1

    rng = np.random.default_rng(SEED)
    latent_sd = np.maximum(sim.sd_floor, sim.adp * sim.sd_pct)
    branch_width = int(sim.lookahead_branch_width)
    future_weight = float(sim.lookahead_future_weight)
    vorp_weight = float(sim.lookahead_vorp_weight)

    greedy_wins = 0
    lookahead_overturns = 0
    winner_first = Counter()
    winner_first_pos = Counter()
    winner_second = Counter()
    winner_pairs = Counter()
    candidate_appearances = Counter()
    candidate_path_wins = Counter()
    candidate_current_scores: dict[str, list[float]] = defaultdict(list)
    candidate_return_scores: dict[str, list[float]] = defaultdict(list)
    candidate_path_values: dict[str, list[float]] = defaultdict(list)
    candidate_returns: dict[str, Counter] = defaultdict(Counter)
    candidate_positions: dict[str, str] = {}
    winner_edge_vs_greedy: list[float] = []

    for _ in range(RUNS):
        available = base_available.copy()
        opp_counts = sim._seed_opponent_counts(opening)
        archetypes = sim._archetype_assignments(rng)
        market_latent = np.maximum(1.0, rng.normal(sim.adp, latent_sd))
        custom_noise_base = rng.normal(0.0, np.maximum(3.0, sim.custom_rank * 0.06))

        # Real room from 1.04 through 2.09.
        for pick in range(4, PICK_22):
            sim._take_opponent_pick(
                pick,
                available,
                opp_counts,
                archetypes,
                market_latent,
                custom_noise_base,
            )

        # Reconcile the production identity-aware layers before scoring PatBot.
        current_scores = sim._patbot_score_vector(available, base_my_counts, PICK_22)
        legal = np.where(available & (current_scores > -1e8))[0]
        if legal.size == 0:
            continue
        ordered = legal[np.argsort(current_scores[legal])[::-1]]
        candidates = ordered[:branch_width]
        greedy_idx = int(candidates[0])
        greedy_name = str(sim.names[greedy_idx])

        base_patbot = set(getattr(sim, "_patbot_owned_idxs", set()))
        base_opponents = set(getattr(sim, "_opponent_owned_idxs", set()))
        if chase_idx not in base_patbot:
            base_patbot.add(chase_idx)

        branch_results = []
        for candidate in candidates:
            candidate = int(candidate)
            name = str(sim.names[candidate])
            pos = str(sim.pos[candidate])
            candidate_positions[name] = pos
            candidate_appearances[name] += 1
            candidate_current_scores[name].append(float(current_scores[candidate]))

            sim._patbot_owned_idxs = set(base_patbot) | {candidate}
            sim._opponent_owned_idxs = set(base_opponents)

            branch_available = available.copy()
            branch_opp_counts = opp_counts.copy()
            branch_my_counts = base_my_counts.copy()
            branch_available[candidate] = False
            code = sim.pos_code[candidate]
            if code >= 0:
                branch_my_counts[code] += 1

            for future_pick in range(PICK_22 + 1, PICK_27):
                sim._take_opponent_pick(
                    future_pick,
                    branch_available,
                    branch_opp_counts,
                    archetypes,
                    market_latent,
                    custom_noise_base,
                )

            future_scores = sim._patbot_score_vector(
                branch_available,
                branch_my_counts,
                PICK_27,
            )
            future_legal = np.where(branch_available & (future_scores > -1e8))[0]
            if future_legal.size:
                future_idx = int(future_legal[np.argmax(future_scores[future_legal])])
                future_name = str(sim.names[future_idx])
                future_score = float(future_scores[future_idx])
                pair_vorp = max(float(sim.vorp[candidate]), 0.0) + max(float(sim.vorp[future_idx]), 0.0)
            else:
                future_idx = None
                future_name = "<none>"
                future_score = 0.0
                pair_vorp = max(float(sim.vorp[candidate]), 0.0)

            path_value = (
                float(current_scores[candidate])
                + future_weight * future_score
                + vorp_weight * pair_vorp
            )
            candidate_return_scores[name].append(future_score)
            candidate_path_values[name].append(path_value)
            candidate_returns[name][future_name] += 1
            branch_results.append((path_value, candidate, future_idx, future_name, future_score))

        branch_results.sort(key=lambda x: x[0], reverse=True)
        best_path_value, best_candidate, _, best_future_name, _ = branch_results[0]
        best_name = str(sim.names[best_candidate])
        best_pos = str(sim.pos[best_candidate])
        candidate_path_wins[best_name] += 1
        winner_first[best_name] += 1
        winner_first_pos[best_pos] += 1
        winner_second[best_future_name] += 1
        winner_pairs[(best_name, best_future_name)] += 1

        greedy_result = next(x for x in branch_results if int(x[1]) == greedy_idx)
        greedy_path_value = float(greedy_result[0])
        if best_candidate == greedy_idx:
            greedy_wins += 1
        else:
            lookahead_overturns += 1
        winner_edge_vs_greedy.append(float(best_path_value - greedy_path_value))

        sim._patbot_owned_idxs = set(base_patbot)
        sim._opponent_owned_idxs = set(base_opponents)

    print("\nPatBot v0.6.8 Ja'Marr 1.03 -> 2.10/3.03 paired path audit")
    print("State fixed: Gibbs 1.01, Bijan 1.02, Ja'Marr Chase 1.03.")
    print("The real-room model simulates picks 4-21, branches across the production top-5 at 2.10,")
    print("then simulates picks 23-26 with common room randomness and scores what comes back at 3.03.")
    print(f"Runs: {RUNS:,} | seed: {SEED} | picks: 22 -> 27")
    print("Bye weeks: informational only; zero draft-value or path-value adjustment.\n")

    print("How often Round-2 lookahead changes the greedy pick:")
    print(f"  Greedy 2.10 leader also wins paired path: {_pct(greedy_wins, RUNS):5.1f}%")
    print(f"  Lookahead prefers a different 2.10 pick:   {_pct(lookahead_overturns, RUNS):5.1f}%")
    if winner_edge_vs_greedy:
        arr = np.asarray(winner_edge_vs_greedy, dtype=float)
        print(f"  Avg path edge over greedy choice:           {arr.mean():+6.2f}")
        print(f"  P25 / median / P75 edge:                    {np.percentile(arr,25):+6.2f} / {np.median(arr):+6.2f} / {np.percentile(arr,75):+6.2f}")

    print("\nMost common paired-path winners at 2.10:")
    for name, count in winner_first.most_common(12):
        print(f"  {name:<28} {candidate_positions.get(name,'?'):>2}  {_pct(count, RUNS):5.1f}%")

    print("\nWinning 2.10 pick by position:")
    for pos, count in winner_first_pos.most_common():
        print(f"  {pos:<3} {_pct(count, RUNS):5.1f}%")

    print("\nMost common 3.03 returns on the winning path:")
    for name, count in winner_second.most_common(12):
        print(f"  {name:<28} {_pct(count, RUNS):5.1f}%")

    print("\nMost common winning 2.10 -> 3.03 pairs:")
    for (first, second), count in winner_pairs.most_common(15):
        print(f"  {first:<24} -> {second:<24} {_pct(count, RUNS):5.1f}%")

    print("\nCandidate-level path summary (when candidate appears in production top-5 at 2.10):")
    ranked = sorted(
        candidate_appearances,
        key=lambda n: (candidate_path_wins[n], candidate_appearances[n]),
        reverse=True,
    )[:15]
    print("  Candidate                    Pos  Appear%  PathWin%  AvgNow  AvgReturn  AvgPath")
    print("  ------------------------------------------------------------------------------")
    for name in ranked:
        apps = candidate_appearances[name]
        wins = candidate_path_wins[name]
        print(
            f"  {name:<28} {candidate_positions.get(name,'?'):>2}  "
            f"{_pct(apps, RUNS):7.1f}  {_pct(wins, apps):8.1f}  "
            f"{_mean(candidate_current_scores[name]):7.2f}  "
            f"{_mean(candidate_return_scores[name]):9.2f}  "
            f"{_mean(candidate_path_values[name]):7.2f}"
        )

    print("\nCommon 3.03 returns by notable 2.10 candidate:")
    for name in ranked[:8]:
        if not candidate_returns[name]:
            continue
        returns = ", ".join(
            f"{ret} {_pct(count, candidate_appearances[name]):.1f}%"
            for ret, count in candidate_returns[name].most_common(5)
        )
        print(f"  {name:<28}: {returns}")

    print("\nInterpretation guide:")
    print("- This uses the same Round-2 lookahead structure PatBot already uses in production: current score + weighted next-turn score + small pair-VORP term.")
    print("- Same-team concentration is already embedded in both the 2.10 and 3.03 production score vectors.")
    print("- The audit is diagnostic only; it does not change weights, lookahead rules, or bye-week treatment.\n")


if __name__ == "__main__":
    main()
