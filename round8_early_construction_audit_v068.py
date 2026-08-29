from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from patbot.config import load_config
from patbot.data import load_players
from patbot.draft import DraftEngine
from patbot.draft_state import make_pick_record
from patbot.roster_strategy import offensive_starters_complete
from patbot.sim import FastDraftSimulator


RUNS = 1500
SEED = 20260903
THROUGH_ROUND = 8


def _row(players, name: str):
    match = players[players["name"].eq(name)]
    if match.empty:
        raise RuntimeError(f"Could not find {name!r} in live snapshot")
    return match.iloc[0]


def _pct(x: int, n: int) -> float:
    return 100.0 * float(x) / max(int(n), 1)


def _mean(values) -> float:
    return float(np.mean(values)) if values else float("nan")


def _median(values) -> float:
    return float(np.median(values)) if values else float("nan")


def _build_label(sim: FastDraftSimulator, mine: list[int]) -> str:
    c = Counter(str(sim.pos[i]) for i in mine)
    if c["QB"] >= 1 and c["TE"] >= 1:
        return "WR/QB/TE"
    if c["WR"] >= 2 and c["TE"] >= 1:
        return "WR/WR/TE"
    if c["RB"] >= 1 and c["TE"] >= 1:
        return "WR/RB/TE"
    if c["WR"] >= 2 and c["RB"] >= 1:
        return "WR/WR/RB"
    if c["WR"] >= 2 and c["QB"] >= 1:
        return "WR/WR/QB"
    if c["WR"] >= 3:
        return "WR/WR/WR"
    if c["RB"] >= 2:
        return "WR/RB/RB"
    return "Other"


def _skill_shell_complete(counts: Counter, roster_cfg: dict) -> bool:
    # RB/WR/TE base starters plus FLEX, intentionally ignoring QB.
    for pos in ("RB", "WR", "TE"):
        if counts[pos] < int(roster_cfg.get(pos, 0)):
            return False
    flex_need = int(roster_cfg.get("FLEX", 0))
    eligible = [str(x).upper() for x in roster_cfg.get("flex_eligible", ["RB", "WR", "TE"])]
    excess = sum(max(counts[pos] - int(roster_cfg.get(pos, 0)), 0) for pos in eligible)
    return excess >= flex_need


def main():
    cfg = load_config()
    players = load_players("data/players_2026_live.csv")
    players["player_id"] = players["player_id"].astype(str)
    engine = DraftEngine(players, cfg)
    sim = FastDraftSimulator(engine)
    teams = int(cfg["league"]["teams"])
    slot = int(cfg["league"]["draft_slot"])

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

    my_picks_by_round = {}
    for round_no in range(1, THROUGH_ROUND + 1):
        within = slot if round_no % 2 else teams + 1 - slot
        my_picks_by_round[round_no] = (round_no - 1) * teams + within
    last_pick = my_picks_by_round[THROUGH_ROUND]

    base_available = np.ones(sim.n, dtype=bool)
    base_available[[gibbs_idx, bijan_idx, chase_idx]] = False
    rng = np.random.default_rng(SEED)
    latent_sd = np.maximum(sim.sd_floor, sim.adp * sim.sd_pct)

    # Metrics by R1-R3 construction bucket.
    group_runs = Counter()
    group_full_r7 = Counter()
    group_full_r8 = Counter()
    group_skill_r7 = Counter()
    group_skill_r8 = Counter()
    group_r8_missing = defaultdict(Counter)
    group_r8_lineup = defaultdict(list)
    group_r8_starter_vorp = defaultdict(list)
    group_r8_concentration = defaultdict(list)
    group_r4_8_vorp = defaultdict(list)
    group_r4_8_adp_reach = defaultdict(list)
    group_r4_8_non_rbwr = Counter()
    group_sequences = defaultdict(Counter)
    group_exact_starts = defaultdict(Counter)

    overall_full_r7 = 0
    overall_full_r8 = 0
    overall_skill_r8 = 0

    for _ in range(RUNS):
        available = base_available.copy()
        opp_counts = sim._seed_opponent_counts(opening)
        archetypes = sim._archetype_assignments(rng)
        market_latent = np.maximum(1.0, rng.normal(sim.adp, latent_sd))
        custom_noise_base = rng.normal(0.0, np.maximum(3.0, sim.custom_rank * 0.06))

        mine = [chase_idx]
        selected_by_round = {1: chase_idx}
        my_counts = np.zeros(len(sim.POSITIONS), dtype=np.int16)
        my_counts[sim.pos_to_code["WR"]] = 1
        snapshots = {1: list(mine)}

        sim._patbot_owned_idxs = {chase_idx}
        sim._opponent_owned_idxs = {gibbs_idx, bijan_idx}

        for pick in range(4, last_pick + 1):
            if pick in sim.my_picks:
                idx = int(sim._lookahead_pick(
                    available,
                    my_counts,
                    pick,
                    opp_counts,
                    archetypes,
                    market_latent,
                    custom_noise_base,
                ))
                if not available[idx]:
                    raise RuntimeError(f"PatBot selected unavailable player {sim.names[idx]} at pick {pick}")
                available[idx] = False
                mine.append(idx)
                code = sim.pos_code[idx]
                if code >= 0:
                    my_counts[code] += 1
                round_no = (pick - 1) // teams + 1
                selected_by_round[round_no] = idx
                snapshots[round_no] = list(mine)
            else:
                sim._take_opponent_pick(
                    pick,
                    available,
                    opp_counts,
                    archetypes,
                    market_latent,
                    custom_noise_base,
                )

        r3 = snapshots[3]
        r7 = snapshots[7]
        r8 = snapshots[8]
        label = _build_label(sim, r3)
        group_runs[label] += 1
        group_exact_starts[label][" / ".join(str(sim.names[i]) for i in r3)] += 1

        c7 = Counter(str(sim.pos[i]) for i in r7)
        c8 = Counter(str(sim.pos[i]) for i in r8)
        full7 = offensive_starters_complete(c7, cfg["roster"])
        full8 = offensive_starters_complete(c8, cfg["roster"])
        skill7 = _skill_shell_complete(c7, cfg["roster"])
        skill8 = _skill_shell_complete(c8, cfg["roster"])
        group_full_r7[label] += int(full7)
        group_full_r8[label] += int(full8)
        group_skill_r7[label] += int(skill7)
        group_skill_r8[label] += int(skill8)
        overall_full_r7 += int(full7)
        overall_full_r8 += int(full8)
        overall_skill_r8 += int(skill8)

        eval8 = sim.evaluate_roster(r8)
        group_r8_lineup[label].append(float(eval8["lineup_score"]))
        group_r8_starter_vorp[label].append(float(eval8["starter_vorp"]))
        group_r8_concentration[label].append(float(eval8.get("team_concentration_penalty", 0.0)))
        for pos, count in eval8.get("missing_starters", {}).items():
            group_r8_missing[label][str(pos)] += int(count)

        post3_positions = []
        for round_no in range(4, THROUGH_ROUND + 1):
            idx = int(selected_by_round[round_no])
            post3_positions.append(str(sim.pos[idx]))
            group_r4_8_vorp[label].append(float(sim.vorp[idx]))
            # Positive = player taken this many slots before market ADP.
            group_r4_8_adp_reach[label].append(float(sim.adp[idx]) - float(my_picks_by_round[round_no]))
            if str(sim.pos[idx]) not in {"RB", "WR"}:
                group_r4_8_non_rbwr[label] += 1
        group_sequences[label]["-".join(post3_positions)] += 1

    print("\nPatBot v0.6.8 Round-8 early-construction debt audit")
    print("Expected opening fixed: Gibbs 1.01, Bijan 1.02, Ja'Marr Chase 1.03.")
    print("PatBot drafts normally through Round 8; production owner/Yahoo/lookahead/roster/risk/concentration layers are active.")
    print(f"Runs: {RUNS:,} | seed: {SEED} | picks: " + ", ".join(f"R{r}={p}" for r, p in my_picks_by_round.items()))
    print("Bye weeks: intentionally zero weight. Focus: does an early QB+TE start create harmful RB/WR construction debt?\n")

    print("Structural note:")
    print("  After WR/QB/TE through three rounds, PatBot has exactly five picks in R4-R8 to fill RB1, RB2, WR2, WR3 and FLEX.")
    print("  That path has zero slack if the goal is a complete offensive starting shell by the end of Round 8.\n")

    print("R1-R3 construction buckets and Round-8 outcomes:")
    print("  Build        Rooms   R7 full  R8 full  R8 skill-shell  R8 lineup  StarterVORP  R4-8 VORP  Avg ADP reach")
    print("  ----------------------------------------------------------------------------------------------------")
    order = sorted(group_runs, key=lambda x: group_runs[x], reverse=True)
    for label in order:
        n = group_runs[label]
        print(
            f"  {label:<11} {_pct(n, RUNS):5.1f}%  "
            f"{_pct(group_full_r7[label], n):7.1f}%  "
            f"{_pct(group_full_r8[label], n):7.1f}%  "
            f"{_pct(group_skill_r8[label], n):13.1f}%  "
            f"{_mean(group_r8_lineup[label]):9.2f}  "
            f"{_mean(group_r8_starter_vorp[label]):11.2f}  "
            f"{_mean(group_r4_8_vorp[label]):9.2f}  "
            f"{_mean(group_r4_8_adp_reach[label]):13.2f}"
        )

    print("\nOverall completion:")
    print(f"  Full offense complete after R7: {_pct(overall_full_r7, RUNS):5.1f}%")
    print(f"  Full offense complete after R8: {_pct(overall_full_r8, RUNS):5.1f}%")
    print(f"  RB/WR/TE + FLEX skill shell complete after R8: {_pct(overall_skill_r8, RUNS):5.1f}%")

    print("\nRound-8 missing starter slots by construction bucket (average missing slots per room):")
    for label in order:
        n = group_runs[label]
        pieces = []
        for pos in ("QB", "RB", "WR", "TE", "FLEX"):
            avg = float(group_r8_missing[label][pos]) / max(n, 1)
            if avg > 0.005:
                pieces.append(f"{pos} {avg:.2f}")
        print(f"  {label:<11}: {', '.join(pieces) if pieces else 'none'}")

    print("\nMost common R4-R8 position sequences by construction bucket:")
    for label in order:
        n = group_runs[label]
        vals = ", ".join(
            f"{seq} {_pct(count, n):.1f}%"
            for seq, count in group_sequences[label].most_common(5)
        )
        print(f"  {label:<11}: {vals}")

    print("\nEarly QB+TE focus:")
    qn = group_runs.get("WR/QB/TE", 0)
    if qn:
        print(f"  WR/QB/TE share of rooms: {_pct(qn, RUNS):.1f}%")
        print(f"  Full offense complete by R8: {_pct(group_full_r8['WR/QB/TE'], qn):.1f}%")
        print(f"  Skill shell complete by R8: {_pct(group_skill_r8['WR/QB/TE'], qn):.1f}%")
        print(f"  Avg R8 lineup score: {_mean(group_r8_lineup['WR/QB/TE']):.2f}")
        print(f"  Median R8 lineup score: {_median(group_r8_lineup['WR/QB/TE']):.2f}")
        print(f"  Avg same-team lineup penalty: {_mean(group_r8_concentration['WR/QB/TE']):.2f}")
        non_rbwr_picks = group_r4_8_non_rbwr['WR/QB/TE']
        print(f"  R4-R8 picks spent outside RB/WR: {non_rbwr_picks}/{qn * 5} ({_pct(non_rbwr_picks, qn * 5):.1f}%)")
        print("  Most common exact R1-R3 starts:")
        for start, count in group_exact_starts['WR/QB/TE'].most_common(8):
            print(f"    {start:<70} {_pct(count, qn):5.1f}% of WR/QB/TE rooms")
    else:
        print("  No WR/QB/TE rooms occurred in this sample.")

    print("\nInterpretation guide:")
    print("- WR/QB/TE is not automatically bad; elite one-off positions can justify construction debt.")
    print("- Concern is warranted if WR/QB/TE has materially worse R8 lineup quality, low completion, or obvious reach pressure versus alternative starts.")
    print("- If it repairs cleanly and retains comparable R8 quality, PatBot should remain free to take elite QB + TE when the room makes it worthwhile.")
    print("- Diagnostic only: this changes no production weights, roster rules, lookahead settings, or bye-week treatment.\n")


if __name__ == "__main__":
    main()
