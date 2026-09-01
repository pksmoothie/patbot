from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .sim import FastDraftSimulator


def _candidate_summary(sim: FastDraftSimulator, candidate_id: str, scores: np.ndarray) -> dict:
    idx = sim.id_to_idx[str(candidate_id)]
    return {
        "Candidate": str(sim.names[idx]),
        "Avg Lineup Score": round(float(np.mean(scores)), 2),
        "10th %ile": round(float(np.percentile(scores, 10)), 2),
        "25th %ile": round(float(np.percentile(scores, 25)), 2),
        "75th %ile": round(float(np.percentile(scores, 75)), 2),
        "90th %ile": round(float(np.percentile(scores, 90)), 2),
        "League Winner Score": round(float(sim.league_winner_score[idx]), 1),
        "Runs": int(len(scores)),
    }


def _initial_my_state(
    sim: FastDraftSimulator,
    my_roster_ids: list[str],
) -> tuple[list[int], np.ndarray]:
    mine = [
        sim.id_to_idx[str(pid)]
        for pid in my_roster_ids
        if str(pid) in sim.id_to_idx
    ]
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
    draft_history: list[dict] | None,
    archetypes: dict[int, str],
    market_latent: np.ndarray,
    custom_noise_base: np.ndarray,
    run_projection: np.ndarray,
) -> float:
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

    for pick in range(int(current_pick), last_pick + 1):
        if not available.any():
            break

        if pick in sim.my_picks:
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
                        archetypes,
                        market_latent,
                        custom_noise_base,
                    )
                )
            available[idx] = False
            mine.append(idx)
            code = sim.pos_code[idx]
            if code >= 0:
                my_counts[code] += 1
        else:
            sim._take_opponent_pick(
                pick,
                available,
                opp_counts,
                archetypes,
                market_latent,
                custom_noise_base,
            )

    result = sim.evaluate_roster(mine, projection_override=run_projection)
    return float(result["lineup_score"])


def paired_stability_check(
    engine,
    *,
    current_pick: int,
    drafted_ids: set[str],
    my_roster_ids: list[str],
    challenger_id: str,
    base_id: str,
    runs: int,
    through_round: int,
    draft_history: list[dict] | None = None,
) -> tuple[pd.DataFrame, list[dict], dict]:
    """Run the large-sample paired gate used before any Final Call overturn.

    Both candidates receive the exact same room randomness and the exact same
    sampled performance/risk shock on paired run N. The branch simulation itself
    consumes no RNG, so the only intentional difference inside each pair is the
    forced current candidate and the downstream room response to that choice.
    """
    n = max(1, int(runs))
    reference = FastDraftSimulator(engine)
    challenger_sim = FastDraftSimulator(engine)
    base_sim = FastDraftSimulator(engine)

    rng = np.random.default_rng(int(reference.comparison_seed))
    latent_sd = np.maximum(reference.sd_floor, reference.adp * reference.sd_pct)
    challenger_scores = np.empty(n, dtype=float)
    base_scores = np.empty(n, dtype=float)

    for i in range(n):
        archetypes = reference._archetype_assignments(rng)
        market_latent = np.maximum(1.0, rng.normal(reference.adp, latent_sd))
        custom_noise_base = rng.normal(
            0.0,
            np.maximum(3.0, reference.custom_rank * 0.06),
        )
        run_projection, _ = reference._sample_run_projection(rng)

        challenger_scores[i] = _simulate_branch(
            challenger_sim,
            current_pick=int(current_pick),
            through_round=int(through_round),
            drafted_ids={str(x) for x in drafted_ids},
            my_roster_ids=[str(x) for x in my_roster_ids],
            candidate_id=str(challenger_id),
            draft_history=draft_history,
            archetypes=dict(archetypes),
            market_latent=market_latent,
            custom_noise_base=custom_noise_base,
            run_projection=run_projection,
        )
        base_scores[i] = _simulate_branch(
            base_sim,
            current_pick=int(current_pick),
            through_round=int(through_round),
            drafted_ids={str(x) for x in drafted_ids},
            my_roster_ids=[str(x) for x in my_roster_ids],
            candidate_id=str(base_id),
            draft_history=draft_history,
            archetypes=dict(archetypes),
            market_latent=market_latent,
            custom_noise_base=custom_noise_base,
            run_projection=run_projection,
        )

    deltas = challenger_scores - base_scores
    mean_delta = float(np.mean(deltas))
    if n >= 2:
        standard_error = float(np.std(deltas, ddof=1)) / math.sqrt(n)
        ci_low = mean_delta - 1.96 * standard_error
        ci_high = mean_delta + 1.96 * standard_error
    else:
        standard_error = 0.0
        ci_low = mean_delta
        ci_high = mean_delta

    challenger_wins = int(np.sum(deltas > 0))
    ties = int(np.sum(deltas == 0))
    paired_win_pct = 100.0 * challenger_wins / n

    rows = [
        _candidate_summary(challenger_sim, str(challenger_id), challenger_scores),
        _candidate_summary(base_sim, str(base_id), base_scores),
    ]
    summary = pd.DataFrame(rows).sort_values(
        "Avg Lineup Score", ascending=False
    ).reset_index(drop=True)

    details = [
        {
            "candidate": str(challenger_sim.names[challenger_sim.id_to_idx[str(challenger_id)]]),
            "candidate_id": str(challenger_id),
            "runs": n,
        },
        {
            "candidate": str(base_sim.names[base_sim.id_to_idx[str(base_id)]]),
            "candidate_id": str(base_id),
            "runs": n,
        },
    ]
    paired = {
        "runs": n,
        "challenger": details[0]["candidate"],
        "base": details[1]["candidate"],
        "mean_delta": round(mean_delta, 4),
        "paired_win_pct": round(paired_win_pct, 2),
        "ties": ties,
        "standard_error": round(standard_error, 4),
        "ci_low": round(float(ci_low), 4),
        "ci_high": round(float(ci_high), 4),
    }
    return summary, details, paired
