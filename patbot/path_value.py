from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .draft import snake_pick
from .sequence_audit import (
    _fantasypros_ecr,
    _run_branch,
    _take_patbot,
)
from .sim import FastDraftSimulator
from .timing_audit import (
    NEG_INF,
    _pick_role,
    _position_rank,
    _roster_counts,
    _seed_from_history,
    _set_identity,
)


def select_cross_position_challengers(
    sim: FastDraftSimulator,
    scores: np.ndarray,
    available: np.ndarray,
    selected_idx: int,
    *,
    top_pool: int = 12,
) -> list[dict]:
    """Choose neutral cross-position alternatives by five independent lenses.

    The selected player's position is excluded because this audit asks a sequence
    question: take the current position now, or take value at another position
    and return to the selected position later? VORP/ECR/LWS/Q90 challengers are
    restricted to the top ``top_pool`` legal cross-position options by current
    PatBot score so an extreme metric cannot manufacture an unrealistic option.
    """
    selected_idx = int(selected_idx)
    selected_pos = str(sim.pos[selected_idx]).upper()
    positions = np.asarray(sim.pos).astype(str)
    legal = np.where(
        np.asarray(available, dtype=bool)
        & (positions != selected_pos)
        & np.isin(positions, ["QB", "RB", "WR", "TE"])
        & (np.asarray(scores, dtype=float) > NEG_INF / 2)
    )[0]
    if len(legal) == 0:
        return []

    ordered = legal[np.argsort(np.asarray(scores, dtype=float)[legal])[::-1]]
    pool = ordered[: max(1, int(top_pool))]
    selectors: dict[str, int] = {
        "Score": int(pool[0]),
        "VORP": int(pool[np.argmax(sim.vorp[pool])]),
        "LWS": int(pool[np.argmax(sim.league_winner_score[pool])]),
        "Q90": int(pool[np.argmax(sim.q90_points[pool])]),
    }

    ecr_values = np.array([_fantasypros_ecr(sim, int(i)) for i in pool], dtype=float)
    finite = np.isfinite(ecr_values)
    if finite.any():
        finite_pool = pool[finite]
        finite_ecr = ecr_values[finite]
        selectors["ECR"] = int(finite_pool[np.argmin(finite_ecr)])

    rank_lookup = {int(idx): rank + 1 for rank, idx in enumerate(ordered)}
    rows = []
    for challenger_type in ("Score", "VORP", "ECR", "LWS", "Q90"):
        if challenger_type not in selectors:
            continue
        idx = int(selectors[challenger_type])
        rows.append(
            {
                "Challenger Type": challenger_type,
                "idx": idx,
                "Cross-Pos Score Rank": int(rank_lookup[idx]),
                "Player": str(sim.names[idx]),
                "Pos": str(sim.pos[idx]),
                "Score": float(scores[idx]),
                "VORP": float(sim.vorp[idx]),
                "LWS": float(sim.league_winner_score[idx]),
                "Q90": float(sim.q90_points[idx]),
                "FP ECR": _fantasypros_ecr(sim, idx),
                "ADP": float(sim.adp[idx]),
            }
        )
    return rows


def _safe_cost(now: float, later: float, *, low_is_good: bool = False) -> float:
    if not np.isfinite(now) or not np.isfinite(later):
        return np.nan
    return float(later - now) if low_is_good else float(now - later)


def _aggregate(rows: pd.DataFrame, runs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if rows.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary_rows = []
    keys = ["Round", "Selected Pos", "Challenger Type", "Wait Turns"]
    for key, group in rows.groupby(keys, sort=True):
        deltas = pd.to_numeric(group["Alt Delta vs Take Now"], errors="coerce").dropna()
        summary_rows.append(
            {
                "Round": int(key[0]),
                "Selected Pos": str(key[1]),
                "Challenger Type": str(key[2]),
                "Wait Turns": int(key[3]),
                "Events": int(len(group)),
                "Event % of Runs": round(100.0 * len(group) / max(int(runs), 1), 1),
                "Selected Survives %": round(100.0 * float(group["Selected Survived"].mean()), 1),
                "Avg Current Score Gap": round(float(group["Current Score Gap"].mean()), 2),
                "Avg VORP Cost Waiting": round(float(group["VORP Cost Waiting"].mean()), 2),
                "Avg ECR Cost Waiting": round(float(group["ECR Cost Waiting"].mean()), 2) if group["ECR Cost Waiting"].notna().any() else np.nan,
                "Avg LWS Cost Waiting": round(float(group["LWS Cost Waiting"].mean()), 2),
                "Avg Q90 Cost Waiting": round(float(group["Q90 Cost Waiting"].mean()), 2),
                "Avg Alt Delta": round(float(deltas.mean()), 2) if not deltas.empty else np.nan,
                "Alt Wins %": round(100.0 * float((deltas > 1e-9).mean()), 1) if not deltas.empty else np.nan,
                "P25 Alt Delta": round(float(np.percentile(deltas, 25)), 2) if not deltas.empty else np.nan,
                "P50 Alt Delta": round(float(np.percentile(deltas, 50)), 2) if not deltas.empty else np.nan,
                "P75 Alt Delta": round(float(np.percentile(deltas, 75)), 2) if not deltas.empty else np.nan,
            }
        )

    common_rows = []
    common_keys = [
        "Round",
        "Selected",
        "Selected Pos",
        "Challenger Type",
        "Challenger",
        "Wait Turns",
        "Wait Fill",
    ]
    for key, group in rows.groupby(common_keys, dropna=False, sort=True):
        common_rows.append(
            {
                "Round": int(key[0]),
                "Selected": str(key[1]),
                "Selected Pos": str(key[2]),
                "Challenger Type": str(key[3]),
                "Challenger": str(key[4]),
                "Wait Turns": int(key[5]),
                "Wait Fill": str(key[6]),
                "Times": int(len(group)),
                "Draft %": round(100.0 * len(group) / max(int(runs), 1), 1),
                "Selected Survives %": round(100.0 * float(group["Selected Survived"].mean()), 1),
                "Avg Current Score Gap": round(float(group["Current Score Gap"].mean()), 2),
                "Avg VORP Cost Waiting": round(float(group["VORP Cost Waiting"].mean()), 2),
                "Avg Alt Delta": round(float(group["Alt Delta vs Take Now"].mean()), 2),
                "Alt Wins %": round(100.0 * float((group["Alt Delta vs Take Now"] > 1e-9).mean()), 1),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["Round", "Selected Pos", "Challenger Type", "Wait Turns"]
    ).reset_index(drop=True)
    common = pd.DataFrame(common_rows).sort_values(
        ["Times", "Avg Alt Delta"], ascending=[False, False]
    ).reset_index(drop=True)
    return summary, common


def run_pick_specific_path_audit(
    engine,
    draft_history: list[dict],
    *,
    forced_first_player_id: str | None = None,
    runs: int = 60,
    through_round: int = 13,
    audit_start_round: int = 2,
    audit_through_round: int = 7,
    wait_turns: tuple[int, ...] = (1, 2),
    challenger_pool: int = 12,
    seed: int | None = None,
    progress: Callable[[int, int], None] | None = None,
):
    """Audit pick-specific opportunity cost for every offensive PatBot pick.

    For each PatBot decision in the audit window, the production choice is
    challenged by cross-position alternatives chosen independently by Score,
    VORP, FantasyPros ECR, LWS and Q90. The alternative path takes that challenger
    now and deliberately returns to the production choice's position one or two
    PatBot turns later. Both paths then finish normally to the same horizon and
    are evaluated on the same sampled football outcome.

    This is diagnostic only. It is designed to tell us whether production urgency
    and roster-fit are overvaluing "fill this position now" before we alter any
    live draft weights.
    """
    sim = FastDraftSimulator(engine)
    if seed is None:
        seed = int(sim.comparison_seed) + 5800

    horizons = tuple(sorted({max(1, int(x)) for x in wait_turns}))
    start_pick = max([int(x.get("overall_pick", 0)) for x in draft_history] or [0]) + 1
    first_patbot_pick = min(p for p in sim.my_picks if p >= start_pick)
    final_pick = snake_pick(int(through_round), sim.teams, sim.slot)
    latent_sd = np.maximum(sim.sd_floor, sim.adp * sim.sd_pct)
    rows: list[dict] = []

    for run in range(int(runs)):
        rng = np.random.default_rng(int(seed) + run * 10007)
        drafted_idx, mine = _seed_from_history(sim, draft_history)
        available = np.ones(sim.n, dtype=bool)
        if drafted_idx:
            available[list(drafted_idx)] = False
        opp_counts = sim._seed_opponent_counts(draft_history)
        _set_identity(sim, mine, drafted_idx)
        counts = _roster_counts(sim, mine)

        archetypes = sim._archetype_assignments(rng)
        market_latent = np.maximum(1.0, rng.normal(sim.adp, latent_sd))
        custom_noise_base = rng.normal(0.0, np.maximum(3.0, sim.custom_rank * 0.06))

        perf_rng = np.random.default_rng(int(seed) + 1_000_003 + run * 10009)
        run_proj, _ = sim._sample_run_projection(perf_rng)

        for pick in range(int(start_pick), int(final_pick) + 1):
            if not available.any():
                break
            round_no = (pick - 1) // sim.teams + 1

            if pick in sim.my_picks:
                pre_available = available.copy()
                pre_counts = counts.copy()
                pre_mine = list(mine)
                pre_opp_counts = opp_counts.copy()
                pre_patbot_owned = set(getattr(sim, "_patbot_owned_idxs", set()))
                pre_opponent_owned = set(getattr(sim, "_opponent_owned_idxs", set()))
                scores = sim._patbot_score_vector(available, counts, pick)

                if pick == first_patbot_pick and forced_first_player_id is not None:
                    forced_idx = sim.id_to_idx.get(str(forced_first_player_id))
                    if forced_idx is None or not bool(available[int(forced_idx)]):
                        raise ValueError("Forced first PatBot player is not available in this room.")
                    idx = int(forced_idx)
                else:
                    idx = int(
                        sim._lookahead_pick(
                            available,
                            counts,
                            pick,
                            opp_counts,
                            archetypes,
                            market_latent,
                            custom_noise_base,
                        )
                    )

                selected_pos = str(sim.pos[idx]).upper()
                role = _pick_role(sim, pre_counts, idx)
                actual_patbot_owned = set(getattr(sim, "_patbot_owned_idxs", set())) | {int(idx)}
                actual_opponent_owned = set(getattr(sim, "_opponent_owned_idxs", set()))

                if (
                    int(audit_start_round) <= round_no <= int(audit_through_round)
                    and selected_pos in {"QB", "RB", "WR", "TE"}
                ):
                    baseline_mine, _, _ = _run_branch(
                        sim,
                        current_pick=pick,
                        final_pick=final_pick,
                        available=pre_available,
                        counts=pre_counts,
                        mine=pre_mine,
                        opp_counts=pre_opp_counts,
                        archetypes=archetypes,
                        market_latent=market_latent,
                        custom_noise_base=custom_noise_base,
                        first_idx=idx,
                        patbot_owned=pre_patbot_owned,
                        opponent_owned=pre_opponent_owned,
                    )
                    baseline_eval = sim.evaluate_roster(baseline_mine, projection_override=run_proj)
                    baseline_score = float(baseline_eval["lineup_score"])

                    challengers = select_cross_position_challengers(
                        sim,
                        scores,
                        pre_available,
                        idx,
                        top_pool=int(challenger_pool),
                    )
                    branch_cache: dict[tuple[int, int], tuple[list[int], int | None, int | None, float]] = {}
                    for challenger in challengers:
                        challenger_idx = int(challenger["idx"])
                        for horizon in horizons:
                            cache_key = (challenger_idx, int(horizon))
                            if cache_key not in branch_cache:
                                alt_mine, fill_idx, fill_pick = _run_branch(
                                    sim,
                                    current_pick=pick,
                                    final_pick=final_pick,
                                    available=pre_available,
                                    counts=pre_counts,
                                    mine=pre_mine,
                                    opp_counts=pre_opp_counts,
                                    archetypes=archetypes,
                                    market_latent=market_latent,
                                    custom_noise_base=custom_noise_base,
                                    first_idx=challenger_idx,
                                    patbot_owned=pre_patbot_owned,
                                    opponent_owned=pre_opponent_owned,
                                    fill_pos=selected_pos,
                                    wait_turns=int(horizon),
                                )
                                alt_eval = sim.evaluate_roster(alt_mine, projection_override=run_proj)
                                branch_cache[cache_key] = (
                                    alt_mine,
                                    fill_idx,
                                    fill_pick,
                                    float(alt_eval["lineup_score"]),
                                )
                            _, fill_idx, fill_pick, alt_score = branch_cache[cache_key]

                            fill_vorp = float(sim.vorp[fill_idx]) if fill_idx is not None else np.nan
                            fill_lws = float(sim.league_winner_score[fill_idx]) if fill_idx is not None else np.nan
                            fill_q90 = float(sim.q90_points[fill_idx]) if fill_idx is not None else np.nan
                            fill_ecr = _fantasypros_ecr(sim, fill_idx) if fill_idx is not None else np.nan
                            selected_ecr = _fantasypros_ecr(sim, idx)

                            rows.append(
                                {
                                    "Run": int(run),
                                    "Round": int(round_no),
                                    "Overall": int(pick),
                                    "Role": str(role),
                                    "Selected": str(sim.names[idx]),
                                    "Selected Pos": selected_pos,
                                    "Selected Pos Rank": _position_rank(sim, idx),
                                    "Selected Score": float(scores[idx]),
                                    "Selected VORP": float(sim.vorp[idx]),
                                    "Selected LWS": float(sim.league_winner_score[idx]),
                                    "Selected Q90": float(sim.q90_points[idx]),
                                    "Selected FP ECR": selected_ecr,
                                    "Selected ADP": float(sim.adp[idx]),
                                    "Challenger Type": str(challenger["Challenger Type"]),
                                    "Challenger": str(challenger["Player"]),
                                    "Challenger Pos": str(challenger["Pos"]),
                                    "Cross-Pos Score Rank": int(challenger["Cross-Pos Score Rank"]),
                                    "Challenger Score": float(challenger["Score"]),
                                    "Current Score Gap": float(scores[idx]) - float(challenger["Score"]),
                                    "Challenger VORP": float(challenger["VORP"]),
                                    "Challenger LWS": float(challenger["LWS"]),
                                    "Challenger Q90": float(challenger["Q90"]),
                                    "Challenger FP ECR": float(challenger["FP ECR"]) if np.isfinite(challenger["FP ECR"]) else np.nan,
                                    "Wait Turns": int(horizon),
                                    "Wait Fill": str(sim.names[fill_idx]) if fill_idx is not None else "—",
                                    "Wait Fill Pos Rank": _position_rank(sim, fill_idx) if fill_idx is not None else np.nan,
                                    "Wait Fill FP ECR": fill_ecr,
                                    "Wait Fill Overall": int(fill_pick) if fill_pick is not None else np.nan,
                                    "Selected Survived": bool(fill_idx is not None and int(fill_idx) == int(idx)),
                                    "VORP Cost Waiting": _safe_cost(float(sim.vorp[idx]), fill_vorp),
                                    "ECR Cost Waiting": _safe_cost(selected_ecr, fill_ecr, low_is_good=True),
                                    "LWS Cost Waiting": _safe_cost(float(sim.league_winner_score[idx]), fill_lws),
                                    "Q90 Cost Waiting": _safe_cost(float(sim.q90_points[idx]), fill_q90),
                                    "Take-Now Final Lineup": baseline_score,
                                    "Alt Final Lineup": alt_score,
                                    "Alt Delta vs Take Now": alt_score - baseline_score,
                                }
                            )

                sim._patbot_owned_idxs = actual_patbot_owned
                sim._opponent_owned_idxs = actual_opponent_owned
                _take_patbot(
                    sim,
                    idx=idx,
                    available=available,
                    counts=counts,
                    mine=mine,
                )
            else:
                sim._take_opponent_pick(
                    pick,
                    available,
                    opp_counts,
                    archetypes,
                    market_latent,
                    custom_noise_base,
                )

        if progress is not None and ((run + 1) == int(runs) or (run + 1) % max(1, int(runs) // 10) == 0):
            progress(run + 1, int(runs))

    events = pd.DataFrame(rows)
    summary, common = _aggregate(events, int(runs))
    if not events.empty:
        for column in events.select_dtypes(include=["float"]).columns:
            events[column] = events[column].round(2)
    return summary, common, events, {
        "runs": int(runs),
        "seed": int(seed),
        "through_round": int(through_round),
        "audit_start_round": int(audit_start_round),
        "audit_through_round": int(audit_through_round),
        "wait_turns": list(horizons),
        "challenger_pool": int(challenger_pool),
    }
