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


def _paired_stats(challenger_scores: np.ndarray, base_scores: np.ndarray) -> dict:
    deltas = np.asarray(challenger_scores, dtype=float) - np.asarray(base_scores, dtype=float)
    n = int(len(deltas))
    mean_delta = float(np.mean(deltas)) if n else 0.0
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
    paired_win_pct = 100.0 * challenger_wins / max(n, 1)
    return {
        "runs": n,
        "mean_delta": round(mean_delta, 4),
        "paired_win_pct": round(paired_win_pct, 2),
        "ties": ties,
        "standard_error": round(standard_error, 4),
        "ci_low": round(float(ci_low), 4),
        "ci_high": round(float(ci_high), 4),
    }


def _checkpoint_futility_reason(
    paired: dict,
    *,
    required_margin: float,
    min_win_pct: float,
    require_positive_ci: bool,
) -> str | None:
    """Return a conservative early-stop reason, or None if 500 runs are still warranted.

    Early stopping can only preserve the base board; it can never approve an
    overturn. A challenger is futile when its 95% upper bound is already below
    the required mean edge, or when its central mean and win rate both miss the
    final hurdles while the interval is still non-decisive.
    """
    mean_delta = float(paired.get("mean_delta", 0.0))
    win_pct = float(paired.get("paired_win_pct", 0.0))
    ci_low = float(paired.get("ci_low", float("-inf")))
    ci_high = float(paired.get("ci_high", float("inf")))

    if ci_high < float(required_margin):
        return (
            f"95% CI upper bound {ci_high:+.2f} is below the +{float(required_margin):.1f} "
            "mean edge required for an overturn"
        )

    ci_not_decisive = (ci_low <= 0.0) if require_positive_ci else True
    if (
        mean_delta < float(required_margin)
        and win_pct < float(min_win_pct)
        and ci_not_decisive
    ):
        return (
            f"mean edge {mean_delta:+.2f} < +{float(required_margin):.1f}, paired wins "
            f"{win_pct:.1f}% < {float(min_win_pct):.1f}%, and the interval is not decisively positive"
        )
    return None


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


def _branch_template(
    sim: FastDraftSimulator,
    *,
    drafted_ids: set[str],
    my_roster_ids: list[str],
    candidate_id: str,
    draft_history: list[dict] | None,
) -> dict:
    drafted_idx = {
        sim.id_to_idx[str(pid)]
        for pid in drafted_ids
        if str(pid) in sim.id_to_idx
    }
    base_available = np.ones(sim.n, dtype=bool)
    if drafted_idx:
        base_available[list(drafted_idx)] = False

    mine, my_counts = _initial_my_state(sim, my_roster_ids)
    opp_counts = sim._seed_opponent_counts(draft_history)
    return {
        "candidate_idx": sim.id_to_idx[str(candidate_id)],
        "available": base_available,
        "mine": mine,
        "my_counts": my_counts,
        "opp_counts": opp_counts,
        "patbot_owned": set(getattr(sim, "_patbot_owned_idxs", set())),
        "opponent_owned": set(getattr(sim, "_opponent_owned_idxs", set())),
    }


def _simulate_branch(
    sim: FastDraftSimulator,
    template: dict,
    *,
    current_pick: int,
    through_round: int,
    archetypes: dict[int, str],
    market_latent: np.ndarray,
    custom_noise_base: np.ndarray,
    run_projection: np.ndarray,
) -> float:
    available = template["available"].copy()
    mine = list(template["mine"])
    my_counts = template["my_counts"].copy()
    opp_counts = template["opp_counts"].copy()
    candidate_idx = int(template["candidate_idx"])

    # Reset identity-aware production patches without re-parsing draft history
    # on every simulated branch.
    sim._patbot_owned_idxs = set(template["patbot_owned"])
    sim._opponent_owned_idxs = set(template["opponent_owned"])

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
    confirmation_runs: int = 100,
    checkpoint_runs: int = 200,
    required_margin: float = 10.0,
    min_win_pct: float = 55.0,
    require_positive_ci: bool = True,
) -> tuple[pd.DataFrame, list[dict], dict]:
    """Run the staged paired gate used before any Final Call overturn.

    The 100-run confirmation, 200-run futility checkpoint and possible 500-run
    final gate all share one continuous simulation stream. This avoids paying for
    a separate 100-run comparison before starting the larger test. Early stopping
    can only retain the base board; every actual overturn still requires all of
    the requested runs.
    """
    n = max(1, int(runs))
    confirmation_n = min(n, max(1, int(confirmation_runs)))
    checkpoint_n = min(n, max(confirmation_n, int(checkpoint_runs)))

    reference = FastDraftSimulator(engine)
    challenger_sim = FastDraftSimulator(engine)
    base_sim = FastDraftSimulator(engine)
    challenger_template = _branch_template(
        challenger_sim,
        drafted_ids={str(x) for x in drafted_ids},
        my_roster_ids=[str(x) for x in my_roster_ids],
        candidate_id=str(challenger_id),
        draft_history=draft_history,
    )
    base_template = _branch_template(
        base_sim,
        drafted_ids={str(x) for x in drafted_ids},
        my_roster_ids=[str(x) for x in my_roster_ids],
        candidate_id=str(base_id),
        draft_history=draft_history,
    )

    rng = np.random.default_rng(int(reference.comparison_seed))
    latent_sd = np.maximum(reference.sd_floor, reference.adp * reference.sd_pct)
    challenger_scores = np.empty(n, dtype=float)
    base_scores = np.empty(n, dtype=float)
    completed = 0
    stop_stage = "full"
    stop_reason = ""

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
            challenger_template,
            current_pick=int(current_pick),
            through_round=int(through_round),
            archetypes=dict(archetypes),
            market_latent=market_latent,
            custom_noise_base=custom_noise_base,
            run_projection=run_projection,
        )
        base_scores[i] = _simulate_branch(
            base_sim,
            base_template,
            current_pick=int(current_pick),
            through_round=int(through_round),
            archetypes=dict(archetypes),
            market_latent=market_latent,
            custom_noise_base=custom_noise_base,
            run_projection=run_projection,
        )
        completed = i + 1

        if completed == confirmation_n and confirmation_n < n:
            confirmation = _paired_stats(
                challenger_scores[:completed], base_scores[:completed]
            )
            if float(confirmation["mean_delta"]) < float(required_margin):
                stop_stage = "confirmation"
                stop_reason = (
                    f"{completed}-run paired confirmation mean edge "
                    f"{float(confirmation['mean_delta']):+.2f} did not clear the "
                    f"+{float(required_margin):.1f} overturn threshold"
                )
                break

        if completed == checkpoint_n and checkpoint_n < n:
            checkpoint = _paired_stats(
                challenger_scores[:completed], base_scores[:completed]
            )
            futility = _checkpoint_futility_reason(
                checkpoint,
                required_margin=float(required_margin),
                min_win_pct=float(min_win_pct),
                require_positive_ci=bool(require_positive_ci),
            )
            if futility:
                stop_stage = "checkpoint"
                stop_reason = futility
                break

    challenger_scores = challenger_scores[:completed]
    base_scores = base_scores[:completed]
    paired = _paired_stats(challenger_scores, base_scores)

    challenger_name = str(
        challenger_sim.names[challenger_sim.id_to_idx[str(challenger_id)]]
    )
    base_name = str(base_sim.names[base_sim.id_to_idx[str(base_id)]])
    paired.update(
        {
            "challenger": challenger_name,
            "base": base_name,
            "requested_runs": n,
            "confirmation_runs": confirmation_n,
            "checkpoint_runs": checkpoint_n,
            "stop_stage": stop_stage,
            "stopped_early": completed < n,
            "stop_reason": stop_reason,
        }
    )

    rows = [
        _candidate_summary(challenger_sim, str(challenger_id), challenger_scores),
        _candidate_summary(base_sim, str(base_id), base_scores),
    ]
    summary = pd.DataFrame(rows).sort_values(
        "Avg Lineup Score", ascending=False
    ).reset_index(drop=True)

    details = [
        {
            "candidate": challenger_name,
            "candidate_id": str(challenger_id),
            "runs": completed,
        },
        {
            "candidate": base_name,
            "candidate_id": str(base_id),
            "runs": completed,
        },
    ]
    return summary, details, paired
