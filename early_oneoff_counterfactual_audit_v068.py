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
SEED = 20260904
THROUGH_ROUND = 8
NEG_INF = -1_000_000_000.0


def _row(players, name: str):
    match = players[players["name"].eq(name)]
    if match.empty:
        raise RuntimeError(f"Could not find {name!r} in live snapshot")
    return match.iloc[0]


def _pct(x: int, n: int) -> float:
    return 100.0 * float(x) / max(int(n), 1)


def _position_counts(sim: FastDraftSimulator, mine: list[int]) -> Counter:
    return Counter(str(sim.pos[i]) for i in mine)


def _r3_signature(sim: FastDraftSimulator, mine: list[int]) -> str:
    return "/".join(str(sim.pos[i]) for i in mine[:3])


def _make_policy_sim(engine: DraftEngine, policy: str) -> FastDraftSimulator:
    sim = FastDraftSimulator(engine)
    base_score = sim._patbot_score_vector

    def policy_score(available: np.ndarray, roster_counts: np.ndarray, pick: int) -> np.ndarray:
        score = np.asarray(base_score(available, roster_counts, pick), dtype=float).copy()
        round_no = (int(pick) - 1) // sim.teams + 1
        if round_no > 3:
            return score

        qb_code = sim.pos_to_code["QB"]
        te_code = sim.pos_to_code["TE"]
        qbs = int(roster_counts[qb_code])
        tes = int(roster_counts[te_code])

        if policy == "no_double_oneoff":
            if qbs >= 1:
                score[sim.pos == "TE"] = NEG_INF
            if tes >= 1:
                score[sim.pos == "QB"] = NEG_INF
        elif policy == "no_early_qb":
            score[sim.pos == "QB"] = NEG_INF
        return np.where(available, score, NEG_INF)

    if policy != "production":
        sim._patbot_score_vector = policy_score
    return sim


def _simulate_one(
    sim: FastDraftSimulator,
    *,
    opening: list[dict],
    gibbs_idx: int,
    bijan_idx: int,
    chase_idx: int,
    market_latent: np.ndarray,
    custom_noise_base: np.ndarray,
    archetypes: dict[int, str],
    last_pick: int,
) -> dict:
    available = np.ones(sim.n, dtype=bool)
    available[[gibbs_idx, bijan_idx, chase_idx]] = False
    opp_counts = sim._seed_opponent_counts(opening)
    sim._patbot_owned_idxs = {chase_idx}
    sim._opponent_owned_idxs = {gibbs_idx, bijan_idx}

    mine = [chase_idx]
    my_counts = np.zeros(len(sim.POSITIONS), dtype=np.int16)
    my_counts[sim.pos_to_code["WR"]] = 1
    r3 = None

    for pick in range(4, last_pick + 1):
        if pick in sim.my_picks:
            idx = int(
                sim._lookahead_pick(
                    available,
                    my_counts,
                    pick,
                    opp_counts,
                    dict(archetypes),
                    market_latent,
                    custom_noise_base,
                )
            )
            if not available[idx]:
                raise RuntimeError(f"Policy selected unavailable player {sim.names[idx]} at pick {pick}")
            available[idx] = False
            mine.append(idx)
            code = sim.pos_code[idx]
            if code >= 0:
                my_counts[code] += 1
            round_no = (pick - 1) // sim.teams + 1
            if round_no == 3:
                r3 = list(mine)
        else:
            sim._take_opponent_pick(
                pick,
                available,
                opp_counts,
                dict(archetypes),
                market_latent,
                custom_noise_base,
            )

    if r3 is None:
        r3 = list(mine[:3])
    eval_result = sim.evaluate_roster(mine)
    counts = _position_counts(sim, mine)
    complete = offensive_starters_complete(counts, sim.engine.roster_cfg)
    return {
        "mine": mine,
        "r3": r3,
        "r3_sig": _r3_signature(sim, r3),
        "r3_names": " / ".join(str(sim.names[i]) for i in r3),
        "lineup": float(eval_result["lineup_score"]),
        "starter_vorp": float(eval_result["starter_vorp"]),
        "complete": bool(complete),
        "missing": dict(eval_result.get("missing_starters", {})),
    }


def main():
    cfg = load_config()
    players = load_players("data/players_2026_live.csv")
    players["player_id"] = players["player_id"].astype(str)
    engine = DraftEngine(players, cfg)

    sims = {
        "production": _make_policy_sim(engine, "production"),
        "no_double_oneoff": _make_policy_sim(engine, "no_double_oneoff"),
        "no_early_qb": _make_policy_sim(engine, "no_early_qb"),
    }
    ref = sims["production"]
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

    gibbs_idx = ref.id_to_idx[str(gibbs["player_id"])]
    bijan_idx = ref.id_to_idx[str(bijan["player_id"])]
    chase_idx = ref.id_to_idx[str(chase["player_id"])]

    within = slot if THROUGH_ROUND % 2 else teams + 1 - slot
    last_pick = (THROUGH_ROUND - 1) * teams + within

    rng = np.random.default_rng(SEED)
    latent_sd = np.maximum(ref.sd_floor, ref.adp * ref.sd_pct)

    metrics = {
        name: {
            "lineup": [],
            "starter_vorp": [],
            "complete": 0,
            "r3": Counter(),
            "r3_names": Counter(),
            "missing": Counter(),
        }
        for name in sims
    }
    pair_deltas = defaultdict(list)
    changed_r3 = Counter()
    prod_double_runs = 0
    prod_double_deltas = defaultdict(list)
    prod_double_complete = Counter()
    prod_double_alternatives = Counter()

    for _ in range(RUNS):
        archetypes = ref._archetype_assignments(rng)
        market_latent = np.maximum(1.0, rng.normal(ref.adp, latent_sd))
        custom_noise_base = rng.normal(0.0, np.maximum(3.0, ref.custom_rank * 0.06))

        results = {}
        for name, sim in sims.items():
            results[name] = _simulate_one(
                sim,
                opening=opening,
                gibbs_idx=sim.id_to_idx[str(gibbs["player_id"])],
                bijan_idx=sim.id_to_idx[str(bijan["player_id"])],
                chase_idx=sim.id_to_idx[str(chase["player_id"])],
                market_latent=market_latent,
                custom_noise_base=custom_noise_base,
                archetypes=archetypes,
                last_pick=last_pick,
            )
            r = results[name]
            metrics[name]["lineup"].append(r["lineup"])
            metrics[name]["starter_vorp"].append(r["starter_vorp"])
            metrics[name]["complete"] += int(r["complete"])
            metrics[name]["r3"][r["r3_sig"]] += 1
            metrics[name]["r3_names"][r["r3_names"]] += 1
            for pos, count in r["missing"].items():
                metrics[name]["missing"][str(pos)] += int(count)

        prod = results["production"]
        for alt in ("no_double_oneoff", "no_early_qb"):
            pair_deltas[(alt, "lineup")].append(results[alt]["lineup"] - prod["lineup"])
            pair_deltas[(alt, "starter_vorp")].append(results[alt]["starter_vorp"] - prod["starter_vorp"])
            changed_r3[alt] += int(results[alt]["r3_names"] != prod["r3_names"])

        prod_r3_counts = _position_counts(ref, prod["r3"])
        prod_double = prod_r3_counts["QB"] >= 1 and prod_r3_counts["TE"] >= 1
        if prod_double:
            prod_double_runs += 1
            prod_double_complete["production"] += int(prod["complete"])
            for alt in ("no_double_oneoff", "no_early_qb"):
                ar = results[alt]
                prod_double_deltas[(alt, "lineup")].append(ar["lineup"] - prod["lineup"])
                prod_double_deltas[(alt, "starter_vorp")].append(ar["starter_vorp"] - prod["starter_vorp"])
                prod_double_complete[alt] += int(ar["complete"])
                prod_double_alternatives[(alt, ar["r3_names"])] += 1

    labels = {
        "production": "Production",
        "no_double_oneoff": "No QB+TE double by R3",
        "no_early_qb": "No QB through R3",
    }

    print("\nPatBot v0.6.8 early one-off counterfactual audit")
    print("Opening fixed: Gibbs 1.01, Bijan 1.02, Ja'Marr Chase 1.03; paired room randomness through Round 8.")
    print(f"Runs: {RUNS:,} | seed: {SEED} | last PatBot pick: {last_pick}")
    print("Diagnostic policies change only Round-2/3 eligibility; all later production scoring/roster/concentration logic remains active.")
    print("Bye weeks: zero weight. No production setting is changed.\n")

    print("Round-8 outcomes by policy:")
    print("  Policy                         R8 full   Avg lineup   Starter VORP   R3 changed vs prod")
    print("  ---------------------------------------------------------------------------------------")
    for name in ("production", "no_double_oneoff", "no_early_qb"):
        m = metrics[name]
        arr = np.asarray(m["lineup"], dtype=float)
        sv = np.asarray(m["starter_vorp"], dtype=float)
        changed = 0.0 if name == "production" else _pct(changed_r3[name], RUNS)
        print(
            f"  {labels[name]:<30} {_pct(m['complete'], RUNS):7.1f}%"
            f"   {arr.mean():10.2f}   {sv.mean():12.2f}   {changed:8.1f}%"
        )

    print("\nPaired deltas versus production across all rooms:")
    for alt in ("no_double_oneoff", "no_early_qb"):
        d = np.asarray(pair_deltas[(alt, "lineup")], dtype=float)
        v = np.asarray(pair_deltas[(alt, "starter_vorp")], dtype=float)
        print(
            f"  {labels[alt]:<30} lineup mean {d.mean():+6.2f}, median {np.median(d):+6.2f}, "
            f"P25/P75 {np.percentile(d,25):+6.2f}/{np.percentile(d,75):+6.2f}; "
            f"starter VORP {v.mean():+6.2f}"
        )

    print(f"\nProduction WR/QB/TE focus: {prod_double_runs}/{RUNS} rooms ({_pct(prod_double_runs, RUNS):.1f}%)")
    if prod_double_runs:
        print(f"  Production R8 completion: {_pct(prod_double_complete['production'], prod_double_runs):.1f}%")
        for alt in ("no_double_oneoff", "no_early_qb"):
            d = np.asarray(prod_double_deltas[(alt, "lineup")], dtype=float)
            v = np.asarray(prod_double_deltas[(alt, "starter_vorp")], dtype=float)
            print(
                f"  {labels[alt]:<30} completion {_pct(prod_double_complete[alt], prod_double_runs):5.1f}% | "
                f"paired lineup delta {d.mean():+6.2f} mean / {np.median(d):+6.2f} median | "
                f"starter VORP delta {v.mean():+6.2f}"
            )

        print("\n  Most common counterfactual R1-R3 starts in those same production WR/QB/TE rooms:")
        for alt in ("no_double_oneoff", "no_early_qb"):
            vals = [
                (names, count)
                for (policy, names), count in prod_double_alternatives.items()
                if policy == alt
            ]
            vals.sort(key=lambda x: x[1], reverse=True)
            print(f"    {labels[alt]}:")
            for names, count in vals[:8]:
                print(f"      {names:<72} {_pct(count, prod_double_runs):5.1f}%")

    print("\nMost common production R1-R3 starts:")
    for names, count in metrics["production"]["r3_names"].most_common(10):
        print(f"  {names:<72} {_pct(count, RUNS):5.1f}%")

    print("\nInterpretation guide:")
    print("- If only the no-double policy improves WR/QB/TE rooms, the issue is one-off concentration (QB+TE together).")
    print("- If no-early-QB also improves broadly, the stronger diagnosis is early-QB opportunity cost rather than QB+TE specifically.")
    print("- A production change should be a soft/model-derived hurdle, not a hard player-specific ban, and only if paired R8 outcomes are materially better.\n")


if __name__ == "__main__":
    main()
