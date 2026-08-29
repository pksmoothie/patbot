from __future__ import annotations

from collections import Counter

import numpy as np

from patbot.config import load_config
from patbot.data import load_players
from patbot.draft import DraftEngine
from patbot.draft_state import make_pick_record
from patbot.sim import FastDraftSimulator


RUNS = 1500
SEED = 20260901
PICK_22 = 22
PICK_27 = 27


def _row(players, name: str):
    match = players[players["name"].eq(name)]
    if match.empty:
        raise RuntimeError(f"Could not find {name!r} in live snapshot")
    return match.iloc[0]


def _pct(x: int, n: int) -> float:
    return 100.0 * float(x) / max(int(n), 1)


def _summ(counter: Counter, total: int, n: int = 10) -> str:
    if not counter:
        return "<none>"
    return ", ".join(
        f"{name} {_pct(count, total):.1f}%"
        for name, count in counter.most_common(n)
    )


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
    bowers = _row(players, "Brock Bowers")
    mcbride = _row(players, "Trey McBride")

    opening = [
        make_pick_record(1, teams, str(gibbs["player_id"]), gibbs["name"], gibbs["team"], gibbs["pos"]),
        make_pick_record(2, teams, str(bijan["player_id"]), bijan["name"], bijan["team"], bijan["pos"]),
        make_pick_record(3, teams, str(chase["player_id"]), chase["name"], chase["team"], chase["pos"]),
    ]

    gibbs_idx = sim.id_to_idx[str(gibbs["player_id"])]
    bijan_idx = sim.id_to_idx[str(bijan["player_id"])]
    chase_idx = sim.id_to_idx[str(chase["player_id"])]
    bowers_idx = sim.id_to_idx[str(bowers["player_id"])]
    mcbride_idx = sim.id_to_idx[str(mcbride["player_id"])]

    base_available = np.ones(sim.n, dtype=bool)
    base_available[[gibbs_idx, bijan_idx, chase_idx]] = False
    my_counts = np.zeros(len(sim.POSITIONS), dtype=np.int16)
    my_counts[sim.pos_to_code["WR"]] = 1

    rng = np.random.default_rng(SEED)
    latent_sd = np.maximum(sim.sd_floor, sim.adp * sim.sd_pct)
    future_weight = float(sim.lookahead_future_weight)
    vorp_weight = float(sim.lookahead_vorp_weight)

    greedy_bowers = 0
    both_top5 = 0
    switch_cases = 0
    switch_edges: list[float] = []
    immediate_changes: list[float] = []
    return_changes: list[float] = []
    pair_vorp_changes: list[float] = []

    mcbride_branch_bowers_taken = 0
    bowers_branch_mcbride_taken = 0
    mcbride_branch_bowers_take_pick = Counter()
    bowers_branch_mcbride_take_pick = Counter()
    mcbride_returns = Counter()
    bowers_returns = Counter()
    branch_return_pairs = Counter()
    first_divergence_pairs = Counter()
    soak_with_better_return = 0
    same_return = 0

    example_rows: list[tuple] = []

    def run_branch(
        candidate_idx: int,
        available: np.ndarray,
        opp_counts: np.ndarray,
        archetypes: dict[int, str],
        market_latent: np.ndarray,
        custom_noise_base: np.ndarray,
        base_patbot: set[int],
        base_opponents: set[int],
        current_scores: np.ndarray,
    ) -> dict:
        sim._patbot_owned_idxs = set(base_patbot) | {int(candidate_idx)}
        sim._opponent_owned_idxs = set(base_opponents)

        branch_available = available.copy()
        branch_counts = opp_counts.copy()
        branch_my_counts = my_counts.copy()
        branch_available[int(candidate_idx)] = False
        code = sim.pos_code[int(candidate_idx)]
        if code >= 0:
            branch_my_counts[code] += 1

        picks = []
        for pick in range(PICK_22 + 1, PICK_27):
            idx, _ = sim._take_opponent_pick(
                pick,
                branch_available,
                branch_counts,
                archetypes,
                market_latent,
                custom_noise_base,
            )
            picks.append((pick, int(idx), str(sim.names[idx])))

        future_scores = sim._patbot_score_vector(
            branch_available,
            branch_my_counts,
            PICK_27,
        )
        legal = np.where(branch_available & (future_scores > -1e8))[0]
        if legal.size:
            future_idx = int(legal[np.argmax(future_scores[legal])])
            future_name = str(sim.names[future_idx])
            future_score = float(future_scores[future_idx])
            pair_vorp = max(float(sim.vorp[candidate_idx]), 0.0) + max(float(sim.vorp[future_idx]), 0.0)
        else:
            future_idx = None
            future_name = "<none>"
            future_score = 0.0
            pair_vorp = max(float(sim.vorp[candidate_idx]), 0.0)

        path_value = (
            float(current_scores[candidate_idx])
            + future_weight * future_score
            + vorp_weight * pair_vorp
        )
        return {
            "candidate": str(sim.names[candidate_idx]),
            "now": float(current_scores[candidate_idx]),
            "picks": picks,
            "return_idx": future_idx,
            "return_name": future_name,
            "return_score": future_score,
            "pair_vorp": pair_vorp,
            "path": path_value,
        }

    for _ in range(RUNS):
        available = base_available.copy()
        opp_counts = sim._seed_opponent_counts(opening)
        archetypes = sim._archetype_assignments(rng)
        market_latent = np.maximum(1.0, rng.normal(sim.adp, latent_sd))
        custom_noise_base = rng.normal(0.0, np.maximum(3.0, sim.custom_rank * 0.06))

        for pick in range(4, PICK_22):
            sim._take_opponent_pick(
                pick,
                available,
                opp_counts,
                archetypes,
                market_latent,
                custom_noise_base,
            )

        current_scores = sim._patbot_score_vector(available, my_counts, PICK_22)
        legal = np.where(available & (current_scores > -1e8))[0]
        if legal.size == 0:
            continue
        ordered = legal[np.argsort(current_scores[legal])[::-1]]
        top5 = ordered[: int(sim.lookahead_branch_width)]
        if int(top5[0]) != bowers_idx:
            continue
        greedy_bowers += 1
        if mcbride_idx not in set(int(x) for x in top5):
            continue
        both_top5 += 1

        base_patbot = set(getattr(sim, "_patbot_owned_idxs", set()))
        base_opponents = set(getattr(sim, "_opponent_owned_idxs", set()))
        base_patbot.add(chase_idx)

        bowers_branch = run_branch(
            bowers_idx,
            available,
            opp_counts,
            archetypes,
            market_latent,
            custom_noise_base,
            base_patbot,
            base_opponents,
            current_scores,
        )
        mcbride_branch = run_branch(
            mcbride_idx,
            available,
            opp_counts,
            archetypes,
            market_latent,
            custom_noise_base,
            base_patbot,
            base_opponents,
            current_scores,
        )

        # We only care about the exact phenomenon under review: Bowers is the
        # greedy leader, yet the same-position McBride branch has higher path value.
        if mcbride_branch["path"] <= bowers_branch["path"]:
            continue

        switch_cases += 1
        edge = float(mcbride_branch["path"] - bowers_branch["path"])
        switch_edges.append(edge)
        immediate_changes.append(float(mcbride_branch["now"] - bowers_branch["now"]))
        return_changes.append(float(mcbride_branch["return_score"] - bowers_branch["return_score"]))
        pair_vorp_changes.append(float(mcbride_branch["pair_vorp"] - bowers_branch["pair_vorp"]))

        mcbride_returns[mcbride_branch["return_name"]] += 1
        bowers_returns[bowers_branch["return_name"]] += 1
        branch_return_pairs[(bowers_branch["return_name"], mcbride_branch["return_name"])] += 1
        if bowers_branch["return_name"] == mcbride_branch["return_name"]:
            same_return += 1

        mc_picks = {pick: (idx, name) for pick, idx, name in mcbride_branch["picks"]}
        bo_picks = {pick: (idx, name) for pick, idx, name in bowers_branch["picks"]}
        for pick in range(PICK_22 + 1, PICK_27):
            bname = bo_picks[pick][1]
            mname = mc_picks[pick][1]
            if bname != mname:
                first_divergence_pairs[(pick, bname, mname)] += 1
                break

        bowers_take = next((pick for pick, idx, _ in mcbride_branch["picks"] if idx == bowers_idx), None)
        if bowers_take is not None:
            mcbride_branch_bowers_taken += 1
            mcbride_branch_bowers_take_pick[int(bowers_take)] += 1
            if mcbride_branch["return_score"] > bowers_branch["return_score"]:
                soak_with_better_return += 1

        mcbride_take = next((pick for pick, idx, _ in bowers_branch["picks"] if idx == mcbride_idx), None)
        if mcbride_take is not None:
            bowers_branch_mcbride_taken += 1
            bowers_branch_mcbride_take_pick[int(mcbride_take)] += 1

        if len(example_rows) < 8:
            example_rows.append((
                edge,
                bowers_branch["now"],
                mcbride_branch["now"],
                bowers_branch["return_name"],
                bowers_branch["return_score"],
                mcbride_branch["return_name"],
                mcbride_branch["return_score"],
                " / ".join(name for _, _, name in bowers_branch["picks"]),
                " / ".join(name for _, _, name in mcbride_branch["picks"]),
            ))

        sim._patbot_owned_idxs = set(base_patbot)
        sim._opponent_owned_idxs = set(base_opponents)

    print("\nPatBot v0.6.8 Bowers -> McBride same-position lookahead audit")
    print("State: Gibbs 1.01, Bijan 1.02, Ja'Marr Chase 1.03; real room simulated to 2.10.")
    print(f"Runs: {RUNS:,} | seed: {SEED} | branch window: picks 23-26 | return: 3.03")
    print("Question: when Bowers is the greedy 2.10 leader, why can lookahead prefer McBride?\n")

    print(f"Rooms with Bowers as greedy 2.10 leader: {greedy_bowers:,} ({_pct(greedy_bowers, RUNS):.1f}% of all rooms)")
    print(f"...and McBride also in top-{sim.lookahead_branch_width}: {both_top5:,}")
    print(f"Exact Bowers -> McBride path switches: {switch_cases:,} ({_pct(switch_cases, RUNS):.1f}% of all rooms; {_pct(switch_cases, greedy_bowers):.1f}% of Bowers-greedy rooms)\n")

    if not switch_cases:
        print("No Bowers -> McBride switches occurred in this seed. Nothing to diagnose.\n")
        return

    edge = np.asarray(switch_edges, dtype=float)
    now = np.asarray(immediate_changes, dtype=float)
    ret = np.asarray(return_changes, dtype=float)
    pv = np.asarray(pair_vorp_changes, dtype=float)
    print("Economics of the switch (McBride branch minus Bowers branch):")
    print(f"  Path edge:            mean {edge.mean():+6.2f} | median {np.median(edge):+6.2f} | P25/P75 {np.percentile(edge,25):+6.2f}/{np.percentile(edge,75):+6.2f}")
    print(f"  Immediate 2.10 score: mean {now.mean():+6.2f} | median {np.median(now):+6.2f}")
    print(f"  3.03 return score:    mean {ret.mean():+6.2f} | median {np.median(ret):+6.2f}")
    print(f"  Pair VORP:            mean {pv.mean():+6.2f} | median {np.median(pv):+6.2f}\n")

    print("Direct soak/block evidence:")
    print(f"  Leave Bowers on board -> Bowers is taken during picks 23-26: {_pct(mcbride_branch_bowers_taken, switch_cases):5.1f}%")
    print(f"  Draft Bowers -> McBride is taken during picks 23-26:          {_pct(bowers_branch_mcbride_taken, switch_cases):5.1f}%")
    print(f"  Leave Bowers + Bowers gets taken + 3.03 return improves:      {_pct(soak_with_better_return, switch_cases):5.1f}%")
    print(f"  Both branches produce the exact same 3.03 return:             {_pct(same_return, switch_cases):5.1f}%")
    if mcbride_branch_bowers_take_pick:
        print("  Bowers take-pick distribution when left available: " + _summ(mcbride_branch_bowers_take_pick, switch_cases, 4))
    if bowers_branch_mcbride_take_pick:
        print("  McBride take-pick distribution when Bowers drafted: " + _summ(bowers_branch_mcbride_take_pick, switch_cases, 4))

    print("\nMost common 3.03 returns if PatBot takes Bowers:")
    for name, count in bowers_returns.most_common(8):
        print(f"  {name:<28} {_pct(count, switch_cases):5.1f}%")
    print("\nMost common 3.03 returns if PatBot takes McBride:")
    for name, count in mcbride_returns.most_common(8):
        print(f"  {name:<28} {_pct(count, switch_cases):5.1f}%")

    print("\nMost common return changes (Bowers branch -> McBride branch):")
    for (before, after), count in branch_return_pairs.most_common(10):
        print(f"  {before:<24} -> {after:<24} {_pct(count, switch_cases):5.1f}%")

    print("\nFirst opponent-pick divergence between the two branches:")
    for (pick, bowers_pick, mcbride_pick), count in first_divergence_pairs.most_common(10):
        print(f"  Pick {pick}: Bowers-branch takes {bowers_pick:<22} | McBride-branch takes {mcbride_pick:<22} {_pct(count, switch_cases):5.1f}%")

    if example_rows:
        print("\nA few switch examples:")
        for edge_v, bnow, mnow, bret, brets, mret, mrets, bpicks, mpicks in example_rows:
            print(
                f"  edge {edge_v:+.2f} | now Bowers {bnow:.2f} vs McBride {mnow:.2f} | "
                f"3.03 Bowers-branch {bret} {brets:.2f} vs McBride-branch {mret} {mrets:.2f}"
            )
            print(f"    picks 23-26 after Bowers : {bpicks}")
            print(f"    picks 23-26 after McBride: {mpicks}")

    print("\nInterpretation guide:")
    print("- If Bowers is frequently consumed by an opponent when left available and that branch returns a better 3.03 player, the switch is a legitimate soak/block effect.")
    print("- If returns are usually identical, or Bowers is rarely taken, then a same-position switch may be simulator noise/artifact and deserves correction.")
    print("- Diagnostic only: no production weights or lookahead rules are changed.\n")


if __name__ == "__main__":
    main()
