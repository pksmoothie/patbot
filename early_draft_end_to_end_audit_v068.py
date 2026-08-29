from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np

from patbot.config import load_config
from patbot.data import load_players
from patbot.draft import DraftEngine
from patbot.draft_state import make_pick_record
from patbot.sim import FastDraftSimulator


RUNS = 1500
SEED = 20260902
THROUGH_ROUND = 6


def _row(players, name: str):
    match = players[players["name"].eq(name)]
    if match.empty:
        raise RuntimeError(f"Could not find {name!r} in live snapshot")
    return match.iloc[0]


def _pct(x: int, n: int) -> float:
    return 100.0 * float(x) / max(int(n), 1)


def _position_signature(sim: FastDraftSimulator, mine: list[int]) -> str:
    counts = Counter(str(sim.pos[i]) for i in mine)
    ordered = ["QB", "RB", "WR", "TE", "K", "DEF"]
    return " ".join(f"{p}{counts[p]}" for p in ordered if counts[p])


def _sequence_signature(sim: FastDraftSimulator, picks: list[int]) -> str:
    return "-".join(str(sim.pos[i]) for i in picks)


def _skill_concentration(sim: FastDraftSimulator, mine: list[int]):
    skill = {"RB", "WR", "TE"}
    by_team: dict[str, list[int]] = {}
    for idx in mine:
        team = str(sim.nfl_team[idx]).strip().upper()
        pos = str(sim.pos[idx]).upper()
        if pos in skill and team and team not in {"FA", "NONE", "NAN", "UNK", "UNKNOWN"}:
            by_team.setdefault(team, []).append(int(idx))
    pair_count = 0
    triple_plus = 0
    pair_names = []
    for team, idxs in by_team.items():
        if len(idxs) >= 2:
            pair_count += len(list(combinations(idxs, 2)))
            for a, b in combinations(idxs, 2):
                pair_names.append(f"{sim.names[a]} + {sim.names[b]} ({team})")
        if len(idxs) >= 3:
            triple_plus += 1
    return pair_count, triple_plus, pair_names


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

    round_player = {r: Counter() for r in range(1, THROUGH_ROUND + 1)}
    round_position = {r: Counter() for r in range(1, THROUGH_ROUND + 1)}
    positional_sequence = Counter()
    round3_builds = Counter()
    round4_builds = Counter()
    round5_builds = Counter()
    round6_builds = Counter()
    round3_players = Counter()
    round6_players = Counter()
    same_team_pairs = Counter()

    no_rb_r3 = 0
    no_rb_r4 = 0
    no_rb_r5 = 0
    fewer_two_rb_r6 = 0
    fewer_three_wr_r6 = 0
    no_qb_r6 = 0
    no_te_r6 = 0
    qb_te_by_r3 = 0
    qb_te_by_r4 = 0
    two_te_r6 = 0
    four_plus_wr_r6 = 0
    four_plus_rb_r6 = 0
    same_team_skill_any_r6 = 0
    same_team_skill_triple_r6 = 0
    hard_rule_violations = 0

    lineup_scores = []
    concentration_penalties = []
    missing_by_pos = Counter()

    for _ in range(RUNS):
        available = base_available.copy()
        opp_counts = sim._seed_opponent_counts(opening)
        archetypes = sim._archetype_assignments(rng)
        market_latent = np.maximum(1.0, rng.normal(sim.adp, latent_sd))
        custom_noise_base = rng.normal(0.0, np.maximum(3.0, sim.custom_rank * 0.06))

        mine = [chase_idx]
        selected = [chase_idx]
        my_counts = np.zeros(len(sim.POSITIONS), dtype=np.int16)
        my_counts[sim.pos_to_code["WR"]] = 1
        snapshots: dict[int, list[int]] = {1: list(mine)}

        # The identity-aware production patches use these sets for TE2 and
        # same-team concentration logic. _seed_opponent_counts initializes them,
        # but make the expected opening state explicit here too.
        sim._patbot_owned_idxs = {chase_idx}
        sim._opponent_owned_idxs = {gibbs_idx, bijan_idx}

        round_player[1][str(sim.names[chase_idx])] += 1
        round_position[1][str(sim.pos[chase_idx])] += 1

        for pick in range(4, last_pick + 1):
            if pick in sim.my_picks:
                idx = sim._lookahead_pick(
                    available,
                    my_counts,
                    pick,
                    opp_counts,
                    archetypes,
                    market_latent,
                    custom_noise_base,
                )
                idx = int(idx)
                if not available[idx]:
                    raise RuntimeError(f"PatBot selected unavailable player {sim.names[idx]} at pick {pick}")
                available[idx] = False
                mine.append(idx)
                selected.append(idx)
                code = sim.pos_code[idx]
                if code >= 0:
                    my_counts[code] += 1
                round_no = (pick - 1) // teams + 1
                snapshots[round_no] = list(mine)
                round_player[round_no][str(sim.names[idx])] += 1
                round_position[round_no][str(sim.pos[idx])] += 1
            else:
                sim._take_opponent_pick(
                    pick,
                    available,
                    opp_counts,
                    archetypes,
                    market_latent,
                    custom_noise_base,
                )

        r3 = snapshots.get(3, mine)
        r4 = snapshots.get(4, mine)
        r5 = snapshots.get(5, mine)
        r6 = snapshots.get(6, mine)

        round3_builds[_position_signature(sim, r3)] += 1
        round4_builds[_position_signature(sim, r4)] += 1
        round5_builds[_position_signature(sim, r5)] += 1
        round6_builds[_position_signature(sim, r6)] += 1
        round3_players[" / ".join(str(sim.names[i]) for i in r3)] += 1
        round6_players[" / ".join(str(sim.names[i]) for i in r6)] += 1
        positional_sequence[_sequence_signature(sim, selected)] += 1

        c3 = Counter(str(sim.pos[i]) for i in r3)
        c4 = Counter(str(sim.pos[i]) for i in r4)
        c5 = Counter(str(sim.pos[i]) for i in r5)
        c6 = Counter(str(sim.pos[i]) for i in r6)

        no_rb_r3 += int(c3["RB"] == 0)
        no_rb_r4 += int(c4["RB"] == 0)
        no_rb_r5 += int(c5["RB"] == 0)
        fewer_two_rb_r6 += int(c6["RB"] < 2)
        fewer_three_wr_r6 += int(c6["WR"] < 3)
        no_qb_r6 += int(c6["QB"] == 0)
        no_te_r6 += int(c6["TE"] == 0)
        qb_te_by_r3 += int(c3["QB"] >= 1 and c3["TE"] >= 1)
        qb_te_by_r4 += int(c4["QB"] >= 1 and c4["TE"] >= 1)
        two_te_r6 += int(c6["TE"] >= 2)
        four_plus_wr_r6 += int(c6["WR"] >= 4)
        four_plus_rb_r6 += int(c6["RB"] >= 4)

        pair_count, triple_count, pair_names = _skill_concentration(sim, r6)
        same_team_skill_any_r6 += int(pair_count > 0)
        same_team_skill_triple_r6 += int(triple_count > 0)
        for label in pair_names:
            same_team_pairs[label] += 1

        # Hard production construction invariants that should never be violated.
        if c6["QB"] > 1 or c6["TE"] > 2 or c6["K"] > 0 or c6["DEF"] > 0:
            hard_rule_violations += 1

        eval_result = sim.evaluate_roster(r6)
        lineup_scores.append(float(eval_result["lineup_score"]))
        concentration_penalties.append(float(eval_result.get("team_concentration_penalty", 0.0)))
        for pos, count in eval_result.get("missing_starters", {}).items():
            missing_by_pos[str(pos)] += int(count)

    print("\nPatBot v0.6.8 end-to-end early-draft construction audit")
    print("Expected opening branch fixed: Gibbs 1.01, Bijan 1.02, Ja'Marr Chase 1.03.")
    print("From pick 1.04 onward the real-room model runs normally; PatBot makes every selection through Round 6.")
    print(f"Runs: {RUNS:,} | seed: {SEED} | PatBot picks: " + ", ".join(f"R{r}={p}" for r, p in my_picks_by_round.items()))
    print("Production layers active: owner profiles, Yahoo room behavior, R2/R3 lookahead, roster rules, risk/upside, same-team concentration.")
    print("Bye weeks: intentionally zero weight. Final Call is not rerun inside each synthetic room; this audits the production draft/simulator construction engine.\n")

    print("Pick-by-pick position mix:")
    for r in range(1, THROUGH_ROUND + 1):
        mix = ", ".join(f"{pos} {_pct(count, RUNS):.1f}%" for pos, count in round_position[r].most_common())
        print(f"  Round {r}: {mix}")

    print("\nMost common players by PatBot round:")
    for r in range(2, THROUGH_ROUND + 1):
        vals = ", ".join(
            f"{name} {_pct(count, RUNS):.1f}%"
            for name, count in round_player[r].most_common(8)
        )
        print(f"  R{r}: {vals}")

    print("\nMost common positional sequences through Round 6:")
    for sig, count in positional_sequence.most_common(12):
        print(f"  {sig:<24} {_pct(count, RUNS):5.1f}%")

    print("\nMost common roster-position builds:")
    for label, counter in (("After R3", round3_builds), ("After R4", round4_builds), ("After R5", round5_builds), ("After R6", round6_builds)):
        vals = ", ".join(f"{sig} {_pct(count, RUNS):.1f}%" for sig, count in counter.most_common(8))
        print(f"  {label}: {vals}")

    print("\nConstruction stress indicators:")
    print(f"  Zero RB after Round 3:          {_pct(no_rb_r3, RUNS):5.1f}%")
    print(f"  Zero RB after Round 4:          {_pct(no_rb_r4, RUNS):5.1f}%")
    print(f"  Zero RB after Round 5:          {_pct(no_rb_r5, RUNS):5.1f}%")
    print(f"  Fewer than 2 RB after Round 6:  {_pct(fewer_two_rb_r6, RUNS):5.1f}%")
    print(f"  Fewer than 3 WR after Round 6:  {_pct(fewer_three_wr_r6, RUNS):5.1f}%")
    print(f"  No QB after Round 6:            {_pct(no_qb_r6, RUNS):5.1f}%")
    print(f"  No TE after Round 6:            {_pct(no_te_r6, RUNS):5.1f}%")
    print(f"  QB + TE both by Round 3:        {_pct(qb_te_by_r3, RUNS):5.1f}%")
    print(f"  QB + TE both by Round 4:        {_pct(qb_te_by_r4, RUNS):5.1f}%")
    print(f"  Two TEs by Round 6:             {_pct(two_te_r6, RUNS):5.1f}%")
    print(f"  4+ WR by Round 6:               {_pct(four_plus_wr_r6, RUNS):5.1f}%")
    print(f"  4+ RB by Round 6:               {_pct(four_plus_rb_r6, RUNS):5.1f}%")
    print(f"  Same-team skill pair by R6:     {_pct(same_team_skill_any_r6, RUNS):5.1f}%")
    print(f"  Same-team skill triple by R6:   {_pct(same_team_skill_triple_r6, RUNS):5.1f}%")
    print(f"  Hard-rule violation rate:       {_pct(hard_rule_violations, RUNS):5.1f}%")

    if same_team_pairs:
        print("\nMost common same-team non-QB skill pairs through Round 6:")
        for label, count in same_team_pairs.most_common(10):
            print(f"  {label:<52} {_pct(count, RUNS):5.1f}%")

    print("\nMost common exact three-player starts (R1-R3):")
    for sig, count in round3_players.most_common(10):
        print(f"  {sig:<72} {_pct(count, RUNS):5.1f}%")

    print("\nMost common exact six-player starts (R1-R6):")
    for sig, count in round6_players.most_common(10):
        print(f"  {sig:<110} {_pct(count, RUNS):5.1f}%")

    if lineup_scores:
        arr = np.asarray(lineup_scores, dtype=float)
        cpen = np.asarray(concentration_penalties, dtype=float)
        print("\nRound-6 partial-roster evaluation:")
        print(f"  Lineup score mean/median: {arr.mean():.2f} / {np.median(arr):.2f}")
        print(f"  Lineup score P10/P90:     {np.percentile(arr,10):.2f} / {np.percentile(arr,90):.2f}")
        print(f"  Avg same-team lineup penalty: {cpen.mean():.2f}")
        if missing_by_pos:
            print("  Average missing starter slots at R6: " + ", ".join(
                f"{pos} {count / RUNS:.2f}" for pos, count in missing_by_pos.items()
            ))

    print("\nInterpretation guide:")
    print("- We are looking for emergent construction pathologies, not forcing every starter to be filled by Round 6.")
    print("- Some zero-RB or no-QB/no-TE paths can be rational if the model is harvesting clear value; concern rises if one pattern dominates or persists too long.")
    print("- Hard-rule violations should be exactly 0.0%. Same-team triples should be extremely rare after the new concentration layer.")
    print("- This is diagnostic only; no production settings are changed.\n")


if __name__ == "__main__":
    main()
