from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from patbot.config import load_config
from patbot.data import load_players
from patbot.draft import DraftEngine
from patbot.draft_persistence import load_draft_session
from patbot.draft_state import drafted_ids_from_history, roster_ids_for_slot
from patbot.final_call import run_final_call
from patbot.sim import FastDraftSimulator


LIVE_CSV = Path("data/players_2026_live.csv")
RUNS = 1000
MILESTONES = (100, 500, 1000)
TOP_SURVIVAL_DIFFS = 10
TOP_PICK_OUTCOMES = 8


def _pct(n: int, total: int) -> float:
    return 100.0 * float(n) / max(int(total), 1)


def _player_row(players: pd.DataFrame, player_id: str) -> pd.Series:
    hit = players[players["player_id"].astype(str).eq(str(player_id))]
    if hit.empty:
        raise RuntimeError(f"Player id {player_id!r} not found in live snapshot")
    return hit.iloc[0]


def _candidate_ids(final_call: dict) -> tuple[str, str]:
    base_id = str(final_call.get("base_winner_id", ""))
    sim_id = str(final_call.get("sim_winner_id", ""))
    if not base_id or not sim_id:
        raise RuntimeError("Final Call did not expose both base and simulation candidate ids")
    if base_id == sim_id:
        raise RuntimeError("Current Final Call is not an overturn; base and simulation leader are the same player")
    return base_id, sim_id


def _initial_my_state(sim: FastDraftSimulator, my_roster_ids: list[str]) -> tuple[list[int], np.ndarray]:
    mine = [sim.id_to_idx[str(pid)] for pid in my_roster_ids if str(pid) in sim.id_to_idx]
    counts = np.zeros(len(sim.POSITIONS), dtype=np.int16)
    for idx in mine:
        code = sim.pos_code[idx]
        if code >= 0:
            counts[code] += 1
    return mine, counts


def _simulate_branch(
    sim: FastDraftSimulator,
    *,
    current_pick: int,
    through_round: int,
    drafted_ids: set[str],
    my_roster_ids: list[str],
    candidate_id: str,
    draft_history: list[dict],
    archetypes: dict[int, str],
    market_latent: np.ndarray,
    custom_noise_base: np.ndarray,
    run_projection: np.ndarray,
) -> dict:
    drafted_idx = {
        sim.id_to_idx[str(pid)]
        for pid in drafted_ids
        if str(pid) in sim.id_to_idx
    }
    candidate_idx = sim.id_to_idx[str(candidate_id)]
    available = np.ones(sim.n, dtype=bool)
    if drafted_idx:
        available[list(drafted_idx)] = False

    mine, my_counts = _initial_my_state(sim, my_roster_ids)
    opp_counts = sim._seed_opponent_counts(draft_history)
    last_pick = sim.teams * int(through_round)

    room_picks: dict[int, int] = {}
    my_picks: dict[int, int] = {}
    available_at_my_pick: dict[int, np.ndarray] = {}

    for pick in range(int(current_pick), last_pick + 1):
        if not available.any():
            break

        if pick in sim.my_picks:
            available_at_my_pick[int(pick)] = available.copy()
            if pick == int(current_pick):
                idx = candidate_idx
                if not available[idx]:
                    raise RuntimeError(
                        f"Forced candidate {sim.names[idx]} is unavailable at pick {current_pick}"
                    )
            else:
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
            available[idx] = False
            mine.append(idx)
            my_picks[int(pick)] = idx
            code = sim.pos_code[idx]
            if code >= 0:
                my_counts[code] += 1
        else:
            idx, _ = sim._take_opponent_pick(
                pick,
                available,
                opp_counts,
                dict(archetypes),
                market_latent,
                custom_noise_base,
            )
            room_picks[int(pick)] = int(idx)

    evaluation = sim.evaluate_roster(mine, projection_override=run_projection)
    return {
        "mine": list(mine),
        "my_picks": my_picks,
        "room_picks": room_picks,
        "available_at_my_pick": available_at_my_pick,
        "lineup_score": float(evaluation["lineup_score"]),
        "starter_vorp": float(evaluation.get("starter_vorp", 0.0)),
        "bench_vorp": float(evaluation.get("bench_vorp", 0.0)),
        "team_concentration_penalty": float(evaluation.get("team_concentration_penalty", 0.0)),
        "missing_starters": dict(evaluation.get("missing_starters", {})),
    }


def _swap_candidate(mine: list[int], old_idx: int, new_idx: int) -> list[int] | None:
    if old_idx not in mine:
        return None
    # If the counterfactual candidate was later drafted by PatBot already, a simple
    # one-for-one swap would create a duplicate roster and is not a valid decomposition.
    if new_idx in mine and new_idx != old_idx:
        return None
    out = list(mine)
    out[out.index(old_idx)] = new_idx
    return out


def _first_divergence(a: dict[int, int], b: dict[int, int], start_pick: int, last_pick: int) -> int | None:
    for pick in range(int(start_pick), int(last_pick) + 1):
        if a.get(pick) != b.get(pick):
            return pick
    return None


def _ci95(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 2:
        mean = float(np.mean(values)) if len(values) else float("nan")
        return mean, mean
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1)) / np.sqrt(len(values))
    return mean - 1.96 * se, mean + 1.96 * se


def _top_counter(counter: Counter, total: int, n: int = TOP_PICK_OUTCOMES) -> str:
    if not counter:
        return "<none>"
    return ", ".join(
        f"{name} {_pct(count, total):.1f}%"
        for name, count in counter.most_common(n)
    )


def main() -> None:
    if not LIVE_CSV.exists():
        raise RuntimeError("Missing data/players_2026_live.csv; run the full production refresh first")

    cfg = load_config()
    players = load_players(str(LIVE_CSV))
    players["player_id"] = players["player_id"].astype(str)
    engine = DraftEngine(players, cfg)
    teams = int(cfg["league"]["teams"])
    slot = int(cfg["league"]["draft_slot"])

    history = load_draft_session()
    drafted_ids = drafted_ids_from_history(history)
    my_roster_ids = roster_ids_for_slot(history, slot)
    current_pick = len(history) + 1
    if current_pick not in FastDraftSimulator(engine).my_picks:
        raise RuntimeError(
            f"Current draft state is pick {current_pick}, which is not PatBot's pick. "
            "Record the live room through PatBot's on-clock selection first."
        )

    roster_positions = []
    id_to_pos = dict(zip(players["player_id"].astype(str), players["pos"].astype(str)))
    for pid in my_roster_ids:
        if str(pid) in id_to_pos:
            roster_positions.append(id_to_pos[str(pid)])

    board = engine.recommend(
        current_pick=current_pick,
        drafted_ids=drafted_ids,
        roster_positions=roster_positions,
        top_n=18,
    )
    final = run_final_call(
        engine,
        current_pick=current_pick,
        drafted_ids=drafted_ids,
        my_roster_ids=my_roster_ids,
        board=board,
        draft_history=history,
    )
    base_id, sim_id = _candidate_ids(final)
    base_row = _player_row(players, base_id)
    sim_row = _player_row(players, sim_id)
    base_name = str(base_row["name"])
    sim_name = str(sim_row["name"])
    through_round = int(final.get("through_round", cfg.get("simulation", {}).get("through_round", 8)))

    ref = FastDraftSimulator(engine)
    branches = {
        base_name: FastDraftSimulator(engine),
        sim_name: FastDraftSimulator(engine),
    }
    base_idx = ref.id_to_idx[base_id]
    sim_idx = ref.id_to_idx[sim_id]

    rng = np.random.default_rng(ref.comparison_seed)
    latent_sd = np.maximum(ref.sd_floor, ref.adp * ref.sd_pct)
    last_pick = teams * through_round

    score_a: list[float] = []
    score_b: list[float] = []
    deltas: list[float] = []
    starter_vorp_delta: list[float] = []
    bench_vorp_delta: list[float] = []
    concentration_delta: list[float] = []
    candidate_effects: list[float] = []
    path_effects: list[float] = []
    swap_coverage = 0
    first_divergence = Counter()
    paths_rejoin_after_next = 0
    identical_after_current = 0
    branch_pick_counts: dict[str, dict[int, Counter]] = {
        base_name: defaultdict(Counter),
        sim_name: defaultdict(Counter),
    }
    availability_counts: dict[str, dict[int, np.ndarray]] = {
        base_name: {},
        sim_name: {},
    }
    clone_failures = 0

    for run in range(RUNS):
        archetypes = ref._archetype_assignments(rng)
        market_latent = np.maximum(1.0, rng.normal(ref.adp, latent_sd))
        custom_noise_base = rng.normal(0.0, np.maximum(3.0, ref.custom_rank * 0.06))

        # Production draws performance/availability after the room path. No draft
        # method consumes RNG, so this is the same common-random-number shock for both branches.
        run_projection, _ = ref._sample_run_projection(rng)

        a = _simulate_branch(
            branches[base_name],
            current_pick=current_pick,
            through_round=through_round,
            drafted_ids=drafted_ids,
            my_roster_ids=my_roster_ids,
            candidate_id=base_id,
            draft_history=history,
            archetypes=archetypes,
            market_latent=market_latent,
            custom_noise_base=custom_noise_base,
            run_projection=run_projection,
        )
        b = _simulate_branch(
            branches[sim_name],
            current_pick=current_pick,
            through_round=through_round,
            drafted_ids=drafted_ids,
            my_roster_ids=my_roster_ids,
            candidate_id=sim_id,
            draft_history=history,
            archetypes=archetypes,
            market_latent=market_latent,
            custom_noise_base=custom_noise_base,
            run_projection=run_projection,
        )

        # Clone-control: in the first 20 rooms, rerun the base candidate from a clean
        # simulator with the exact same room inputs. It must be bit-for-bit deterministic.
        if run < 20:
            clone = _simulate_branch(
                FastDraftSimulator(engine),
                current_pick=current_pick,
                through_round=through_round,
                drafted_ids=drafted_ids,
                my_roster_ids=my_roster_ids,
                candidate_id=base_id,
                draft_history=history,
                archetypes=archetypes,
                market_latent=market_latent,
                custom_noise_base=custom_noise_base,
                run_projection=run_projection,
            )
            if (
                clone["my_picks"] != a["my_picks"]
                or clone["room_picks"] != a["room_picks"]
                or abs(clone["lineup_score"] - a["lineup_score"]) > 1e-9
            ):
                clone_failures += 1

        score_a.append(a["lineup_score"])
        score_b.append(b["lineup_score"])
        delta = b["lineup_score"] - a["lineup_score"]
        deltas.append(delta)
        starter_vorp_delta.append(b["starter_vorp"] - a["starter_vorp"])
        bench_vorp_delta.append(b["bench_vorp"] - a["bench_vorp"])
        concentration_delta.append(
            b["team_concentration_penalty"] - a["team_concentration_penalty"]
        )

        # Two-factor Shapley decomposition: candidate identity vs downstream path.
        a_with_b = _swap_candidate(a["mine"], base_idx, sim_idx)
        b_with_a = _swap_candidate(b["mine"], sim_idx, base_idx)
        if a_with_b is not None and b_with_a is not None:
            a_b_score = branches[base_name].evaluate_roster(
                a_with_b, projection_override=run_projection
            )["lineup_score"]
            b_a_score = branches[sim_name].evaluate_roster(
                b_with_a, projection_override=run_projection
            )["lineup_score"]
            candidate_effect = 0.5 * (
                (float(a_b_score) - a["lineup_score"])
                + (b["lineup_score"] - float(b_a_score))
            )
            path_effect = 0.5 * (
                (float(b_a_score) - a["lineup_score"])
                + (b["lineup_score"] - float(a_b_score))
            )
            candidate_effects.append(float(candidate_effect))
            path_effects.append(float(path_effect))
            swap_coverage += 1

        room_a = {**a["room_picks"], **a["my_picks"]}
        room_b = {**b["room_picks"], **b["my_picks"]}
        div = _first_divergence(room_a, room_b, current_pick + 1, last_pick)
        first_divergence[div if div is not None else "never"] += 1
        if div is None:
            identical_after_current += 1
        if room_a.get(current_pick + 1) != room_b.get(current_pick + 1):
            tail_same = all(
                room_a.get(pick) == room_b.get(pick)
                for pick in range(current_pick + 2, last_pick + 1)
            )
            if tail_same:
                paths_rejoin_after_next += 1

        for label, result in ((base_name, a), (sim_name, b)):
            for pick, idx in result["my_picks"].items():
                branch_pick_counts[label][pick][str(ref.names[idx])] += 1
            for pick, avail in result["available_at_my_pick"].items():
                if pick == current_pick:
                    continue
                if pick not in availability_counts[label]:
                    availability_counts[label][pick] = np.zeros(ref.n, dtype=np.int32)
                availability_counts[label][pick] += avail.astype(np.int32)

    arr_a = np.asarray(score_a, dtype=float)
    arr_b = np.asarray(score_b, dtype=float)
    d = np.asarray(deltas, dtype=float)

    print("\nPatBot v0.6.8 Final Call overturn audit")
    print(f"Live draft state: pick {current_pick} | horizon: Round {through_round} | paired runs: {RUNS:,}")
    print(f"Base score leader: {base_name}")
    print(f"Room-sim challenger: {sim_name}")
    print(f"Production Final Call: {final.get('recommendation')} | reported edge: {final.get('edge')} | stage: {final.get('stage')}")
    print(f"Base projections: {base_name} {float(base_row['proj_points']):.2f} | {sim_name} {float(sim_row['proj_points']):.2f} | raw delta {float(sim_row['proj_points']) - float(base_row['proj_points']):+.2f}\n")

    print("1) Clone control")
    if clone_failures == 0:
        print("  PASS: 20/20 identical-candidate reruns produced identical room paths and lineup scores.")
    else:
        print(f"  FAIL: {clone_failures}/20 identical-candidate reruns diverged. Stop trusting Final Call until fixed.")

    print("\n2) Paired score stability")
    for n in MILESTONES:
        if n > len(d):
            continue
        x = d[:n]
        lo, hi = _ci95(x)
        print(
            f"  {n:4d} runs: {sim_name} edge {x.mean():+7.2f} | median {np.median(x):+7.2f} | "
            f"95% paired CI [{lo:+7.2f}, {hi:+7.2f}] | wins {_pct(int(np.sum(x > 0)), n):5.1f}%"
        )
    print(f"  Final branch averages: {base_name} {arr_a.mean():.2f} | {sim_name} {arr_b.mean():.2f} | delta {d.mean():+.2f}")

    print("\n3) Candidate-vs-path decomposition")
    if swap_coverage:
        cand = np.asarray(candidate_effects, dtype=float)
        path = np.asarray(path_effects, dtype=float)
        print(f"  Valid swap decomposition: {swap_coverage}/{RUNS} runs ({_pct(swap_coverage, RUNS):.1f}%)")
        print(f"  Candidate identity effect: {cand.mean():+7.2f} lineup points")
        print(f"  Downstream path effect:    {path.mean():+7.2f} lineup points")
        print(f"  Reconciled total:          {(cand.mean() + path.mean()):+7.2f} lineup points")
    else:
        print("  No valid candidate-swap runs; the alternate candidate was later drafted by PatBot too often for a clean decomposition.")
    print(f"  Starter-VORP branch delta: {np.mean(starter_vorp_delta):+7.2f}")
    print(f"  Bench-VORP branch delta:   {np.mean(bench_vorp_delta):+7.2f}")
    print(
        f"  Concentration-penalty delta ({sim_name} minus {base_name}): "
        f"{np.mean(concentration_delta):+7.2f} (positive means {sim_name} is penalized more)"
    )

    print("\n4) First downstream divergence")
    for key, count in sorted(
        first_divergence.items(),
        key=lambda kv: (10**9 if kv[0] == "never" else int(kv[0])),
    ):
        label = "never" if key == "never" else f"pick {int(key)}"
        print(f"  {label:<10} {_pct(count, RUNS):5.1f}%")
    print(f"  Entire draft identical after the forced candidate: {_pct(identical_after_current, RUNS):.1f}%")
    print(f"  Diverges only at the immediate next pick, then fully rejoins: {_pct(paths_rejoin_after_next, RUNS):.1f}%")

    print("\n5) PatBot follow-up picks by branch")
    followup_picks = sorted(
        set(branch_pick_counts[base_name]).union(branch_pick_counts[sim_name])
    )
    for pick in followup_picks:
        if pick == current_pick:
            continue
        print(f"  Overall pick {pick}:")
        print(f"    after {base_name:<24}: {_top_counter(branch_pick_counts[base_name][pick], RUNS)}")
        print(f"    after {sim_name:<24}: {_top_counter(branch_pick_counts[sim_name][pick], RUNS)}")

    print("\n6) Largest player-survival differences at PatBot turns")
    for pick in sorted(set(availability_counts[base_name]).intersection(availability_counts[sim_name])):
        a_counts = availability_counts[base_name][pick].astype(float)
        b_counts = availability_counts[sim_name][pick].astype(float)
        diff = 100.0 * (b_counts - a_counts) / RUNS
        # Focus the report on plausible draftable players rather than deep fringe noise.
        eligible = (ref.adp <= max(float(pick) + 45.0, 120.0))
        eligible[base_idx] = False
        eligible[sim_idx] = False
        idxs = np.where(eligible)[0]
        if not len(idxs):
            continue
        idxs = idxs[np.argsort(np.abs(diff[idxs]))[::-1]][:TOP_SURVIVAL_DIFFS]
        print(f"  At overall pick {pick} ({base_name} -> {sim_name}; positive = more available after {sim_name}):")
        for idx in idxs:
            if abs(diff[idx]) < 0.1:
                continue
            print(
                f"    {str(ref.names[idx]):<28} ADP {float(ref.adp[idx]):6.1f} | "
                f"{base_name[:10]:<10} {_pct(int(a_counts[idx]), RUNS):5.1f}% | "
                f"{sim_name[:10]:<10} {_pct(int(b_counts[idx]), RUNS):5.1f}% | delta {diff[idx]:+6.1f} pp"
            )

    print("\nAudit interpretation")
    print("- If the candidate identity effect is only a few points but the downstream path effect is large, the overturn is being created by room/roster dynamics rather than Puka-vs-Chase projection value itself.")
    print("- The first-divergence and survival sections should reveal exactly where that path effect enters the tree.")
    print("- If the paired edge collapses from 100 to 500/1000 runs, the 100-run STRONG label was overconfident.")
    print("- If clone control fails or the decomposition cannot reconcile the edge, treat the production Final Call as untrusted and fix the simulator before drafting from it.\n")


if __name__ == "__main__":
    main()
