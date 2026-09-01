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
    """Run a deterministic paired Final Call stability check.

    Each pair uses the same seed for challenger and base, so both candidates face
    the same room, risk and performance randomness. The seeds vary across pairs
    to produce independent room draws. This is intentionally separate from the
    fast screening simulator because it runs only when a 100-run challenger is
    already strong enough to threaten the base board.
    """
    n = max(1, int(runs))
    challenger_sim = FastDraftSimulator(engine)
    base_sim = FastDraftSimulator(engine)
    start_seed = int(challenger_sim.comparison_seed)

    challenger_scores = np.empty(n, dtype=float)
    base_scores = np.empty(n, dtype=float)

    for i in range(n):
        seed = start_seed + i
        challenger_result = challenger_sim.simulate_candidate(
            current_pick=int(current_pick),
            drafted_ids={str(x) for x in drafted_ids},
            my_roster_ids=[str(x) for x in my_roster_ids],
            candidate_id=str(challenger_id),
            runs=1,
            through_round=int(through_round),
            seed=seed,
            draft_history=draft_history,
        )
        base_result = base_sim.simulate_candidate(
            current_pick=int(current_pick),
            drafted_ids={str(x) for x in drafted_ids},
            my_roster_ids=[str(x) for x in my_roster_ids],
            candidate_id=str(base_id),
            runs=1,
            through_round=int(through_round),
            seed=seed,
            draft_history=draft_history,
        )
        challenger_scores[i] = float(challenger_result["avg_lineup_score"])
        base_scores[i] = float(base_result["avg_lineup_score"])

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
