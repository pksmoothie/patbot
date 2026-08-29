from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from patbot.config import load_config
from patbot.data import load_players
from patbot.draft import DraftEngine
from patbot.draft_state import make_pick_record
from patbot.roster_strategy import offensive_starters_complete
from patbot.sim import FastDraftSimulator


RUNS = 1500
SEED = 20260905
THROUGH_ROUND = 10
PICK_303 = 27
NEG_INF = -1_000_000_000.0

# Pre-declared burden of proof. Passing this audit does NOT itself change
# production; it only earns the right to test a soft early-QB hurdle.
MIN_INTERNAL_MEAN_EDGE = 2.0
MIN_INTERNAL_MEDIAN_EDGE = 1.0
MAX_COMPLETION_LOSS_PP = 3.0
MIN_LATER_QB_RATE = 0.80


def _row(players: pd.DataFrame, name: str) -> pd.Series:
    match = players[players["name"].eq(name)]
    if match.empty:
        raise RuntimeError(f"Could not find {name!r} in live snapshot")
    return match.iloc[0]


def _pct(x: int | float, n: int | float) -> float:
    return 100.0 * float(x) / max(float(n), 1.0)


def _my_pick_for_round(teams: int, slot: int, round_no: int) -> int:
    within = slot if round_no % 2 else teams + 1 - slot
    return (round_no - 1) * teams + within


def _starter_indices(sim: FastDraftSimulator, mine: list[int], proj: np.ndarray) -> tuple[list[int], dict[str, int]]:
    rcfg = sim.engine.roster_cfg
    starters: list[int] = []
    used: set[int] = set()
    missing = Counter()

    for pos in ("QB", "RB", "WR", "TE"):
        idxs = [int(i) for i in mine if str(sim.pos[int(i)]) == pos]
        idxs.sort(key=lambda i: float(proj[i]) if np.isfinite(proj[i]) else -1e18, reverse=True)
        need = int(rcfg.get(pos, 0))
        chosen = idxs[:need]
        starters.extend(chosen)
        used.update(chosen)
        if len(chosen) < need:
            missing[pos] = need - len(chosen)

    flex_need = int(rcfg.get("FLEX", 0))
    eligible = set(str(x).upper() for x in rcfg.get("flex_eligible", ["RB", "WR", "TE"]))
    flex_pool = [
        int(i) for i in mine
        if int(i) not in used and str(sim.pos[int(i)]).upper() in eligible
    ]
    flex_pool.sort(key=lambda i: float(proj[i]) if np.isfinite(proj[i]) else -1e18, reverse=True)
    chosen = flex_pool[:flex_need]
    starters.extend(chosen)
    if len(chosen) < flex_need:
        missing["FLEX"] = flex_need - len(chosen)

    return starters, dict(missing)


def _raw_starter_points(sim: FastDraftSimulator, mine: list[int], proj: np.ndarray) -> float:
    starters, _ = _starter_indices(sim, mine, proj)
    if not starters:
        return float("nan")
    vals = np.asarray([proj[i] for i in starters], dtype=float)
    if not np.isfinite(vals).all():
        return float("nan")
    return float(vals.sum())


def _best_non_qb_lookahead(
    sim: FastDraftSimulator,
    *,
    available: np.ndarray,
    my_counts: np.ndarray,
    pick: int,
    opp_counts: np.ndarray,
    archetypes: dict[int, str],
    market_latent: np.ndarray,
    custom_noise_base: np.ndarray,
    my_owned: set[int],
    opp_owned: set[int],
) -> int:
    """Production Round-3 lookahead, but QB is excluded only as the current candidate.

    QBs remain available to opponents and at the next PatBot pick. This avoids the
    common counterfactual bug where masking QB from the board also changes the room.
    """
    round_no = (int(pick) - 1) // sim.teams + 1
    current_scores = sim._patbot_score_vector(available, my_counts, pick)
    candidates = np.where(available & (sim.pos != "QB"))[0]
    if len(candidates) == 0:
        raise RuntimeError("No non-QB candidate available at 3.03")

    candidates = candidates[np.argsort(current_scores[candidates])[::-1]]
    candidates = candidates[: sim.lookahead_branch_width]

    if not sim.lookahead_enabled or round_no not in sim.lookahead_rounds:
        return int(candidates[0])

    next_pick = sim._next_my_pick(pick)
    gap = next_pick - pick
    if gap <= 0 or gap > sim.lookahead_max_gap:
        return int(candidates[0])

    best_idx = int(candidates[0])
    best_value = -float("inf")

    for candidate in candidates:
        branch_available = available.copy()
        branch_opp_counts = opp_counts.copy()
        branch_my_counts = my_counts.copy()
        branch_available[candidate] = False
        code = sim.pos_code[candidate]
        if code >= 0:
            branch_my_counts[code] += 1

        sim._patbot_owned_idxs = set(my_owned) | {int(candidate)}
        sim._opponent_owned_idxs = set(opp_owned)

        for future_pick in range(pick + 1, next_pick):
            if future_pick in sim.my_picks or not branch_available.any():
                break
            sim._take_opponent_pick(
                future_pick,
                branch_available,
                branch_opp_counts,
                archetypes,
                market_latent,
                custom_noise_base,
            )

        if not branch_available.any():
            future_value = 0.0
            future_idx = None
        else:
            future_scores = sim._patbot_score_vector(branch_available, branch_my_counts, next_pick)
            future_idx = int(np.argmax(future_scores))
            future_value = float(future_scores[future_idx])

        pair_vorp = max(float(sim.vorp[candidate]), 0.0)
        if future_idx is not None:
            pair_vorp += max(float(sim.vorp[future_idx]), 0.0)

        path_value = (
            float(current_scores[candidate])
            + sim.lookahead_future_weight * future_value
            + sim.lookahead_vorp_weight * pair_vorp
        )
        if path_value > best_value:
            best_value = path_value
            best_idx = int(candidate)

    sim._patbot_owned_idxs = set(my_owned)
    sim._opponent_owned_idxs = set(opp_owned)
    return best_idx


def _run_branch(
    sim: FastDraftSimulator,
    *,
    first_idx: int,
    base_available: np.ndarray,
    base_my_counts: np.ndarray,
    base_opp_counts: np.ndarray,
    base_mine: list[int],
    base_my_owned: set[int],
    base_opp_owned: set[int],
    archetypes: dict[int, str],
    market_latent: np.ndarray,
    custom_noise_base: np.ndarray,
    last_pick: int,
    r8_pick: int,
) -> dict:
    available = base_available.copy()
    my_counts = base_my_counts.copy()
    opp_counts = base_opp_counts.copy()
    mine = list(base_mine)
    my_owned = set(base_my_owned)
    opp_owned = set(base_opp_owned)

    if not available[first_idx]:
        raise RuntimeError(f"Forced 3.03 player unavailable: {sim.names[first_idx]}")
    available[first_idx] = False
    mine.append(int(first_idx))
    my_owned.add(int(first_idx))
    code = sim.pos_code[first_idx]
    if code >= 0:
        my_counts[code] += 1

    sim._patbot_owned_idxs = set(my_owned)
    sim._opponent_owned_idxs = set(opp_owned)

    r8_mine: list[int] | None = None
    qb_round = 3 if str(sim.pos[first_idx]) == "QB" else None
    qb_name = str(sim.names[first_idx]) if qb_round is not None else None

    for pick in range(PICK_303 + 1, last_pick + 1):
        if pick in sim.my_picks:
            sim._patbot_owned_idxs = set(my_owned)
            sim._opponent_owned_idxs = set(opp_owned)
            idx = int(sim._lookahead_pick(
                available,
                my_counts,
                pick,
                opp_counts,
                dict(archetypes),
                market_latent,
                custom_noise_base,
            ))
            if not available[idx]:
                raise RuntimeError(f"PatBot selected unavailable player {sim.names[idx]} at pick {pick}")
            available[idx] = False
            mine.append(idx)
            my_owned.add(idx)
            code = sim.pos_code[idx]
            if code >= 0:
                my_counts[code] += 1
            if qb_round is None and str(sim.pos[idx]) == "QB":
                qb_round = (pick - 1) // sim.teams + 1
                qb_name = str(sim.names[idx])
            if pick == r8_pick:
                r8_mine = list(mine)
        else:
            sim._patbot_owned_idxs = set(my_owned)
            sim._opponent_owned_idxs = set(opp_owned)
            idx, _ = sim._take_opponent_pick(
                pick,
                available,
                opp_counts,
                dict(archetypes),
                market_latent,
                custom_noise_base,
            )
            opp_owned.add(int(idx))

    if r8_mine is None:
        r8_mine = list(mine)

    return {
        "r8": r8_mine,
        "r10": list(mine),
        "qb_name": qb_name,
        "qb_round": qb_round,
    }


def _finite_deltas(a: list[float], b: list[float]) -> np.ndarray:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    mask = np.isfinite(aa) & np.isfinite(bb)
    return aa[mask] - bb[mask]


def main() -> None:
    cfg = load_config()
    players = load_players("data/players_2026_live.csv")
    players["player_id"] = players["player_id"].astype(str)
    engine = DraftEngine(players, cfg)
    sim = FastDraftSimulator(engine)

    teams = int(cfg["league"]["teams"])
    slot = int(cfg["league"]["draft_slot"])
    r8_pick = _my_pick_for_round(teams, slot, 8)
    last_pick = _my_pick_for_round(teams, slot, THROUGH_ROUND)

    gibbs = _row(players, "Jahmyr Gibbs")
    bijan = _row(players, "Bijan Robinson")
    chase = _row(players, "Ja'Marr Chase")
    allen = _row(players, "Josh Allen")

    opening = [
        make_pick_record(1, teams, str(gibbs["player_id"]), gibbs["name"], gibbs["team"], gibbs["pos"]),
        make_pick_record(2, teams, str(bijan["player_id"]), bijan["name"], bijan["team"], bijan["pos"]),
        make_pick_record(3, teams, str(chase["player_id"]), chase["name"], chase["team"], chase["pos"]),
    ]

    gibbs_idx = sim.id_to_idx[str(gibbs["player_id"])]
    bijan_idx = sim.id_to_idx[str(bijan["player_id"])]
    chase_idx = sim.id_to_idx[str(chase["player_id"])]
    allen_idx = sim.id_to_idx[str(allen["player_id"])]

    sleeper = pd.to_numeric(
        players.get("sleeper_proj_points", pd.Series([np.nan] * len(players))), errors="coerce"
    ).to_numpy(float)
    fantasypros = pd.to_numeric(
        players.get("fantasypros_proj_points", pd.Series([np.nan] * len(players))), errors="coerce"
    ).to_numpy(float)

    rng = np.random.default_rng(SEED)
    latent_sd = np.maximum(sim.sd_floor, sim.adp * sim.sd_pct)

    metrics = defaultdict(list)
    alt_names = Counter()
    prod_choices = Counter()
    later_qbs = Counter()
    later_qb_rounds = Counter()
    eligible = 0
    prod_allen = 0
    prod_allen_records: list[int] = []

    print("\nPatBot v0.6.8 Josh Allen 3.03 falsification audit")
    print("Opening fixed: Gibbs 1.01, Bijan 1.02, Ja'Marr Chase 1.03; real room model to 3.03.")
    print(f"Runs: {RUNS:,} | seed: {SEED} | paired branches through Round {THROUGH_ROUND} (pick {last_pick})")
    print("Only rooms where Josh Allen is actually available at 3.03 enter the paired test.")
    print("Branch A forces Allen at 3.03. Branch B forces the best NON-QB alternative under the same production Round-3 lookahead.")
    print("After 3.03 both branches revert to unmodified production PatBot. QBs remain available to opponents in the non-QB counterfactual.")
    print("Sleeper-only and FantasyPros-only numbers are EVALUATION ablations; they do not redraft either branch.")
    print("Bye weeks: zero weight. No production setting is changed.\n")

    print("PRE-DECLARED BURDEN OF PROOF before seeing results:")
    print(f"  In rooms where production itself would take Allen, the non-Allen branch must have >= +{MIN_INTERNAL_MEAN_EDGE:.1f} mean")
    print(f"  and >= +{MIN_INTERNAL_MEDIAN_EDGE:.1f} median Round-10 internal lineup edge; positive mean raw starter-point edge")
    print("  under BOTH Sleeper-only and FantasyPros-only projections; no more than 3 percentage points worse Round-10 completion;")
    print(f"  and a later QB on at least {MIN_LATER_QB_RATE:.0%} of non-Allen branches by Round 10.")
    print("  Passing does NOT create an anti-Allen rule; it only justifies testing a soft early-QB hurdle next.\n")

    for run_no in range(RUNS):
        available = np.ones(sim.n, dtype=bool)
        available[[gibbs_idx, bijan_idx, chase_idx]] = False
        opp_counts = sim._seed_opponent_counts(opening)
        my_counts = np.zeros(len(sim.POSITIONS), dtype=np.int16)
        my_counts[sim.pos_to_code["WR"]] = 1
        mine = [chase_idx]
        my_owned = {chase_idx}
        opp_owned = {gibbs_idx, bijan_idx}

        archetypes = sim._archetype_assignments(rng)
        market_latent = np.maximum(1.0, rng.normal(sim.adp, latent_sd))
        custom_noise_base = rng.normal(0.0, np.maximum(3.0, sim.custom_rank * 0.06))

        # Simulate the actual room through 2.10.
        for pick in range(4, 22):
            sim._patbot_owned_idxs = set(my_owned)
            sim._opponent_owned_idxs = set(opp_owned)
            idx, _ = sim._take_opponent_pick(
                pick,
                available,
                opp_counts,
                dict(archetypes),
                market_latent,
                custom_noise_base,
            )
            opp_owned.add(int(idx))

        sim._patbot_owned_idxs = set(my_owned)
        sim._opponent_owned_idxs = set(opp_owned)
        r2_idx = int(sim._lookahead_pick(
            available,
            my_counts,
            22,
            opp_counts,
            dict(archetypes),
            market_latent,
            custom_noise_base,
        ))
        if not available[r2_idx]:
            raise RuntimeError(f"Unavailable R2 selection: {sim.names[r2_idx]}")
        available[r2_idx] = False
        mine.append(r2_idx)
        my_owned.add(r2_idx)
        code = sim.pos_code[r2_idx]
        if code >= 0:
            my_counts[code] += 1

        # Actual opponents 2.11 through 3.02.
        for pick in range(23, PICK_303):
            sim._patbot_owned_idxs = set(my_owned)
            sim._opponent_owned_idxs = set(opp_owned)
            idx, _ = sim._take_opponent_pick(
                pick,
                available,
                opp_counts,
                dict(archetypes),
                market_latent,
                custom_noise_base,
            )
            opp_owned.add(int(idx))

        if not available[allen_idx]:
            continue

        eligible += 1
        base_available = available.copy()
        base_my_counts = my_counts.copy()
        base_opp_counts = opp_counts.copy()
        base_mine = list(mine)
        base_my_owned = set(my_owned)
        base_opp_owned = set(opp_owned)

        # What unmodified production would actually do at 3.03.
        sim._patbot_owned_idxs = set(base_my_owned)
        sim._opponent_owned_idxs = set(base_opp_owned)
        prod_idx = int(sim._lookahead_pick(
            base_available.copy(),
            base_my_counts.copy(),
            PICK_303,
            base_opp_counts.copy(),
            dict(archetypes),
            market_latent,
            custom_noise_base,
        ))
        prod_choices[str(sim.names[prod_idx])] += 1
        production_wants_allen = prod_idx == allen_idx
        if production_wants_allen:
            prod_allen += 1

        # Best non-QB alternative with the board otherwise unchanged.
        sim._patbot_owned_idxs = set(base_my_owned)
        sim._opponent_owned_idxs = set(base_opp_owned)
        alt_idx = _best_non_qb_lookahead(
            sim,
            available=base_available.copy(),
            my_counts=base_my_counts.copy(),
            pick=PICK_303,
            opp_counts=base_opp_counts.copy(),
            archetypes=dict(archetypes),
            market_latent=market_latent,
            custom_noise_base=custom_noise_base,
            my_owned=base_my_owned,
            opp_owned=base_opp_owned,
        )
        alt_names[str(sim.names[alt_idx])] += 1

        allen_branch = _run_branch(
            sim,
            first_idx=allen_idx,
            base_available=base_available,
            base_my_counts=base_my_counts,
            base_opp_counts=base_opp_counts,
            base_mine=base_mine,
            base_my_owned=base_my_owned,
            base_opp_owned=base_opp_owned,
            archetypes=dict(archetypes),
            market_latent=market_latent,
            custom_noise_base=custom_noise_base,
            last_pick=last_pick,
            r8_pick=r8_pick,
        )
        alt_branch = _run_branch(
            sim,
            first_idx=alt_idx,
            base_available=base_available,
            base_my_counts=base_my_counts,
            base_opp_counts=base_opp_counts,
            base_mine=base_mine,
            base_my_owned=base_my_owned,
            base_opp_owned=base_opp_owned,
            archetypes=dict(archetypes),
            market_latent=market_latent,
            custom_noise_base=custom_noise_base,
            last_pick=last_pick,
            r8_pick=r8_pick,
        )

        if alt_branch["qb_name"] is not None:
            later_qbs[str(alt_branch["qb_name"])] += 1
            later_qb_rounds[int(alt_branch["qb_round"])] += 1

        for prefix, branch in (("allen", allen_branch), ("alt", alt_branch)):
            eval8 = sim.evaluate_roster(branch["r8"])
            eval10 = sim.evaluate_roster(branch["r10"])
            counts8 = Counter(str(sim.pos[i]) for i in branch["r8"])
            counts10 = Counter(str(sim.pos[i]) for i in branch["r10"])
            metrics[f"{prefix}_r8_lineup"].append(float(eval8["lineup_score"]))
            metrics[f"{prefix}_r10_lineup"].append(float(eval10["lineup_score"]))
            metrics[f"{prefix}_r10_vorp"].append(float(eval10["starter_vorp"]))
            metrics[f"{prefix}_r8_complete"].append(float(offensive_starters_complete(counts8, engine.roster_cfg)))
            metrics[f"{prefix}_r10_complete"].append(float(offensive_starters_complete(counts10, engine.roster_cfg)))
            metrics[f"{prefix}_blend_raw"].append(_raw_starter_points(sim, branch["r10"], sim.proj))
            metrics[f"{prefix}_sleeper_raw"].append(_raw_starter_points(sim, branch["r10"], sleeper))
            metrics[f"{prefix}_fp_raw"].append(_raw_starter_points(sim, branch["r10"], fantasypros))

        if production_wants_allen:
            prod_allen_records.append(len(metrics["allen_r10_lineup"]) - 1)

    if eligible == 0:
        raise RuntimeError("Josh Allen was never available at 3.03 in this sample")

    def _subset(values: list[float], indices: list[int] | None) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        return arr if indices is None else arr[np.asarray(indices, dtype=int)]

    def _report_subset(label: str, indices: list[int] | None) -> dict:
        n = eligible if indices is None else len(indices)
        print(f"\n{label}: {n:,} paired rooms")
        if n == 0:
            print("  No rooms in subset.")
            return {}

        out = {}
        for stage in ("r8", "r10"):
            a = _subset(metrics[f"allen_{stage}_lineup"], indices)
            b = _subset(metrics[f"alt_{stage}_lineup"], indices)
            d = b - a
            out[f"{stage}_lineup_delta"] = d
            print(
                f"  {stage.upper()} internal lineup: Allen {a.mean():.2f} | non-QB {b.mean():.2f} | "
                f"delta {d.mean():+.2f} mean / {np.median(d):+.2f} median | "
                f"non-QB wins {(d > 0).mean():.1%}"
            )

        a_v = _subset(metrics["allen_r10_vorp"], indices)
        b_v = _subset(metrics["alt_r10_vorp"], indices)
        d_v = b_v - a_v
        print(f"  R10 starter VORP delta (non-QB - Allen): {d_v.mean():+.2f} mean / {np.median(d_v):+.2f} median")

        a_c8 = _subset(metrics["allen_r8_complete"], indices)
        b_c8 = _subset(metrics["alt_r8_complete"], indices)
        a_c10 = _subset(metrics["allen_r10_complete"], indices)
        b_c10 = _subset(metrics["alt_r10_complete"], indices)
        print(
            f"  Completion: R8 Allen {a_c8.mean():.1%} vs non-QB {b_c8.mean():.1%}; "
            f"R10 Allen {a_c10.mean():.1%} vs non-QB {b_c10.mean():.1%}"
        )
        out["r10_completion_delta_pp"] = 100.0 * (b_c10.mean() - a_c10.mean())

        for key, source_label in (
            ("blend_raw", "Production blend raw starter points"),
            ("sleeper_raw", "Sleeper-only raw starter points"),
            ("fp_raw", "FantasyPros-only raw starter points"),
        ):
            a = _subset(metrics[f"allen_{key}"], indices)
            b = _subset(metrics[f"alt_{key}"], indices)
            mask = np.isfinite(a) & np.isfinite(b)
            if not mask.any():
                print(f"  {source_label}: unavailable (no complete paired source coverage)")
                out[f"{key}_delta"] = np.asarray([], dtype=float)
                continue
            d = b[mask] - a[mask]
            out[f"{key}_delta"] = d
            print(
                f"  {source_label}: delta {d.mean():+.2f} mean / {np.median(d):+.2f} median | "
                f"paired source coverage {mask.sum()}/{n} ({mask.mean():.1%})"
            )

        return out

    print("\nROOM FREQUENCY:")
    print(f"  Allen available at 3.03: {eligible}/{RUNS} ({_pct(eligible, RUNS):.1f}%)")
    print(f"  Production would actually take Allen: {prod_allen}/{eligible} Allen-available rooms ({_pct(prod_allen, eligible):.1f}%)")

    print("\nMost common production 3.03 choices when Allen is available:")
    for name, count in prod_choices.most_common(10):
        print(f"  {name:<28} {count:4d} ({_pct(count, eligible):5.1f}%)")

    print("\nBest non-QB 3.03 alternatives in those same rooms:")
    for name, count in alt_names.most_common(10):
        print(f"  {name:<28} {count:4d} ({_pct(count, eligible):5.1f}%)")

    all_results = _report_subset("ALL ALLEN-AVAILABLE ROOMS", None)
    critical_results = _report_subset("CRITICAL SUBSET: PRODUCTION ITSELF WOULD TAKE ALLEN", prod_allen_records)

    print("\nLater QB outcome in the non-Allen branch (all Allen-available rooms):")
    qb_total = sum(later_qbs.values())
    print(f"  QB rostered by Round 10: {qb_total}/{eligible} ({_pct(qb_total, eligible):.1f}%)")
    for name, count in later_qbs.most_common(12):
        print(f"  {name:<28} {count:4d} ({_pct(count, eligible):5.1f}%)")
    if later_qb_rounds:
        rounds = ", ".join(f"R{r} {c} ({_pct(c, eligible):.1f}%)" for r, c in sorted(later_qb_rounds.items()))
        print(f"  Acquisition round: {rounds}")

    print("\nPRE-DECLARED DECISION CHECK (critical production-would-take-Allen subset):")
    if prod_allen == 0:
        print("  INCONCLUSIVE: production never selected Allen in the eligible rooms.")
    else:
        d_internal = critical_results.get("r10_lineup_delta", np.asarray([], dtype=float))
        d_sleeper = critical_results.get("sleeper_raw_delta", np.asarray([], dtype=float))
        d_fp = critical_results.get("fp_raw_delta", np.asarray([], dtype=float))
        completion_delta = float(critical_results.get("r10_completion_delta_pp", -999.0))

        later_qb_critical = 0
        # Reconstruct later-QB rate for the critical subset directly from branch metrics is
        # not stored per-name, so use a conservative all-eligible rate for the formal gate
        # and display that limitation explicitly.
        later_qb_rate = qb_total / eligible

        checks = {
            "internal mean >= +2.0": bool(d_internal.size and d_internal.mean() >= MIN_INTERNAL_MEAN_EDGE),
            "internal median >= +1.0": bool(d_internal.size and np.median(d_internal) >= MIN_INTERNAL_MEDIAN_EDGE),
            "Sleeper-only mean > 0": bool(d_sleeper.size and d_sleeper.mean() > 0.0),
            "FantasyPros-only mean > 0": bool(d_fp.size and d_fp.mean() > 0.0),
            "R10 completion loss <= 3pp": bool(completion_delta >= -MAX_COMPLETION_LOSS_PP),
            "later QB rate >= 80%": bool(later_qb_rate >= MIN_LATER_QB_RATE),
        }
        for label, passed in checks.items():
            print(f"  {'PASS' if passed else 'FAIL':4s}  {label}")
        print("  Note: the later-QB gate uses the all-Allen-available rate; all other gates use the critical subset.")

        if all(checks.values()):
            print("\n  RESULT: FALSIFICATION TEST SUPPORTS postponing Allen often enough to justify a SOFT early-QB hurdle sweep.")
            print("  This is not permission for a hard no-QB rule and does not change production today.")
        else:
            print("\n  RESULT: FALSIFICATION TEST DOES NOT clear the pre-declared bar against Allen at 3.03.")
            print("  Keep Allen fully live in production; do not add an early-QB penalty from the prior construction audit alone.")

    print("\nAnti-circularity note:")
    print("- The two branches share the exact same room randomness and differ only at 3.03.")
    print("- Production PatBot drafts both branches after 3.03, but the audit also grades the resulting starters under two raw projection sources separately.")
    print("- The source ablations are evaluation-only, so they test whether the conclusion survives outside the 60/40 production blend without creating new draft policies.")
    print("- No weights, roster rules, lookahead settings, player-specific rules, or bye-week treatment are changed.\n")


if __name__ == "__main__":
    main()
