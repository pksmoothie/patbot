from __future__ import annotations

from collections import Counter
from typing import Callable

import numpy as np
import pandas as pd

from .draft import snake_pick
from .sim import FastDraftSimulator
from .timing_audit import (
    NEG_INF,
    _pick_role,
    _position_rank,
    _roster_counts,
    _seed_from_history,
    _set_identity,
    starter_quality,
)


def _player_metric(sim: FastDraftSimulator, idx: int, column: str) -> float:
    if column not in sim.players.columns:
        return np.nan
    value = pd.to_numeric(pd.Series([sim.players.iloc[int(idx)].get(column)]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else np.nan


def _fantasypros_ecr(sim: FastDraftSimulator, idx: int) -> float:
    for column in ("fantasypros_api_ecr", "fantasypros_ecr", "fp_ecr"):
        value = _player_metric(sim, idx, column)
        if np.isfinite(value):
            return value
    return np.nan


def select_rbwr_challengers(
    sim: FastDraftSimulator,
    scores: np.ndarray,
    available: np.ndarray,
    *,
    top_pool: int = 12,
) -> list[dict]:
    """Return neutral RB/WR challengers by score, LWS and Q90.

    LWS/Q90 challengers are selected only from the top ``top_pool`` legal RB/WR
    options by current PatBot score. That keeps the audit focused on realistic
    draft alternatives without using the current score to decide which ceiling
    metric wins inside that credible pool.
    """
    positions = np.asarray(sim.pos).astype(str)
    legal = np.where(
        np.asarray(available, dtype=bool)
        & np.isin(positions, ["RB", "WR"])
        & (np.asarray(scores, dtype=float) > NEG_INF / 2)
    )[0]
    if len(legal) == 0:
        return []

    ordered = legal[np.argsort(np.asarray(scores, dtype=float)[legal])[::-1]]
    pool = ordered[: max(1, int(top_pool))]

    selected = {
        "Score": int(pool[0]),
        "LWS": int(pool[np.argmax(sim.league_winner_score[pool])]),
        "Q90": int(pool[np.argmax(sim.q90_points[pool])]),
    }
    rank_lookup = {int(idx): rank + 1 for rank, idx in enumerate(ordered)}

    rows = []
    for challenger_type, idx in selected.items():
        rows.append(
            {
                "Challenger Type": challenger_type,
                "idx": int(idx),
                "RB/WR Score Rank": int(rank_lookup[int(idx)]),
                "Player": str(sim.names[int(idx)]),
                "Pos": str(sim.pos[int(idx)]),
                "Score": float(scores[int(idx)]),
                "VORP": float(sim.vorp[int(idx)]),
                "LWS": float(sim.league_winner_score[int(idx)]),
                "Q90": float(sim.q90_points[int(idx)]),
                "FP ECR": _fantasypros_ecr(sim, int(idx)),
                "ADP": float(sim.adp[int(idx)]),
            }
        )
    return rows


def _best_legal_position(
    sim: FastDraftSimulator,
    scores: np.ndarray,
    available: np.ndarray,
    pos: str,
) -> int | None:
    mask = (
        np.asarray(available, dtype=bool)
        & (np.asarray(sim.pos).astype(str) == str(pos))
        & (np.asarray(scores, dtype=float) > NEG_INF / 2)
    )
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        return None
    return int(idxs[np.argmax(scores[idxs])])


def _best_legal_excluding_position(
    sim: FastDraftSimulator,
    scores: np.ndarray,
    available: np.ndarray,
    blocked_pos: str,
) -> int | None:
    mask = (
        np.asarray(available, dtype=bool)
        & (np.asarray(sim.pos).astype(str) != str(blocked_pos))
        & (np.asarray(scores, dtype=float) > NEG_INF / 2)
    )
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        return None
    return int(idxs[np.argmax(scores[idxs])])


def _take_patbot(
    sim: FastDraftSimulator,
    *,
    idx: int,
    available: np.ndarray,
    counts: np.ndarray,
    mine: list[int],
) -> None:
    idx = int(idx)
    available[idx] = False
    mine.append(idx)
    code = int(sim.pos_code[idx])
    if code >= 0:
        counts[code] += 1
    owned = set(getattr(sim, "_patbot_owned_idxs", set()))
    owned.add(idx)
    sim._patbot_owned_idxs = owned


def _run_branch(
    sim: FastDraftSimulator,
    *,
    current_pick: int,
    final_pick: int,
    available: np.ndarray,
    counts: np.ndarray,
    mine: list[int],
    opp_counts: np.ndarray,
    archetypes: dict[int, str],
    market_latent: np.ndarray,
    custom_noise_base: np.ndarray,
    first_idx: int,
    patbot_owned: set[int],
    opponent_owned: set[int],
    fill_pos: str | None = None,
    wait_turns: int | None = None,
):
    """Run a branch from one PatBot decision through ``final_pick``.

    For delayed-fill counterfactuals, PatBot is deliberately prevented from
    selecting ``fill_pos`` before the requested future PatBot turn. Interim
    selections use the best currently legal non-fill player by PatBot score.
    This is a diagnostic counterfactual, not production draft policy.
    """
    branch_available = np.asarray(available, dtype=bool).copy()
    branch_counts = np.asarray(counts, dtype=np.int16).copy()
    branch_mine = list(mine)
    branch_opp_counts = np.asarray(opp_counts).copy()
    sim._patbot_owned_idxs = set(patbot_owned)
    sim._opponent_owned_idxs = set(opponent_owned)

    _take_patbot(
        sim,
        idx=int(first_idx),
        available=branch_available,
        counts=branch_counts,
        mine=branch_mine,
    )

    fill_idx = None
    fill_pick = None
    patbot_turns_after = 0

    for pick in range(int(current_pick) + 1, int(final_pick) + 1):
        if not branch_available.any():
            break

        if pick in sim.my_picks:
            patbot_turns_after += 1
            scores = sim._patbot_score_vector(branch_available, branch_counts, pick)

            if fill_pos is not None and wait_turns is not None and patbot_turns_after < int(wait_turns):
                idx = _best_legal_excluding_position(
                    sim,
                    scores,
                    branch_available,
                    str(fill_pos),
                )
                if idx is None:
                    idx = int(np.argmax(scores))
            elif fill_pos is not None and wait_turns is not None and patbot_turns_after == int(wait_turns):
                idx = _best_legal_position(sim, scores, branch_available, str(fill_pos))
                if idx is None:
                    idx = int(np.argmax(scores))
                elif str(sim.pos[int(idx)]).upper() == str(fill_pos).upper():
                    fill_idx = int(idx)
                    fill_pick = int(pick)
            else:
                idx = int(
                    sim._lookahead_pick(
                        branch_available,
                        branch_counts,
                        pick,
                        branch_opp_counts,
                        archetypes,
                        market_latent,
                        custom_noise_base,
                    )
                )

            _take_patbot(
                sim,
                idx=int(idx),
                available=branch_available,
                counts=branch_counts,
                mine=branch_mine,
            )
        else:
            sim._take_opponent_pick(
                pick,
                branch_available,
                branch_opp_counts,
                archetypes,
                market_latent,
                custom_noise_base,
            )

    return branch_mine, fill_idx, fill_pick


def _aggregate_sequence_rows(rows: pd.DataFrame, runs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if rows.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary_rows = []
    group_cols = ["Pos", "Quality", "Challenger Type", "Wait Turns"]
    for key, group in rows.groupby(group_cols, sort=True):
        summary_rows.append(
            {
                "Pos": str(key[0]),
                "Quality": str(key[1]),
                "Challenger Type": str(key[2]),
                "Wait Turns": int(key[3]),
                "Events": int(len(group)),
                "Event % of Runs": round(100.0 * len(group) / max(int(runs), 1), 1),
                "Avg Selected Pos Rank": round(float(group["Selected Pos Rank"].mean()), 1),
                "Avg Challenger Score Gap": round(float(group["Challenger Score Gap"].mean()), 2),
                "Avg Challenger LWS": round(float(group["Challenger LWS"].mean()), 1),
                "Avg Challenger Q90": round(float(group["Challenger Q90"].mean()), 1),
                "Avg VORP Cost of Waiting": round(float(group["VORP Cost of Waiting"].mean()), 2),
                "Avg Alt Delta": round(float(group["Alt Delta vs Fill Now"].mean()), 2),
                "Alt Wins %": round(100.0 * float((group["Alt Delta vs Fill Now"] > 1e-9).mean()), 1),
                "P25 Alt Delta": round(float(np.percentile(group["Alt Delta vs Fill Now"], 25)), 2),
                "P50 Alt Delta": round(float(np.percentile(group["Alt Delta vs Fill Now"], 50)), 2),
                "P75 Alt Delta": round(float(np.percentile(group["Alt Delta vs Fill Now"], 75)), 2),
            }
        )

    common_rows = []
    keys = [
        "Pos",
        "Quality",
        "Selected",
        "Challenger Type",
        "Challenger",
        "Wait Turns",
        "Wait Fill",
    ]
    for key, group in rows.groupby(keys, dropna=False, sort=True):
        common_rows.append(
            {
                "Pos": str(key[0]),
                "Quality": str(key[1]),
                "Selected": str(key[2]),
                "Challenger Type": str(key[3]),
                "Challenger": str(key[4]),
                "Wait Turns": int(key[5]),
                "Wait Fill": str(key[6]),
                "Times": int(len(group)),
                "Draft %": round(100.0 * len(group) / max(int(runs), 1), 1),
                "Selected Pos Rank": round(float(group["Selected Pos Rank"].mean()), 1),
                "Selected FP ECR": round(float(group["Selected FP ECR"].mean()), 1) if group["Selected FP ECR"].notna().any() else np.nan,
                "Challenger FP ECR": round(float(group["Challenger FP ECR"].mean()), 1) if group["Challenger FP ECR"].notna().any() else np.nan,
                "Challenger LWS": round(float(group["Challenger LWS"].mean()), 1),
                "Avg Score Gap": round(float(group["Challenger Score Gap"].mean()), 2),
                "Avg VORP Cost Waiting": round(float(group["VORP Cost of Waiting"].mean()), 2),
                "Avg Alt Delta": round(float(group["Alt Delta vs Fill Now"].mean()), 2),
                "Alt Wins %": round(100.0 * float((group["Alt Delta vs Fill Now"] > 1e-9).mean()), 1),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["Pos", "Quality", "Challenger Type", "Wait Turns"]
    ).reset_index(drop=True)
    common = pd.DataFrame(common_rows).sort_values(
        ["Times", "Avg Alt Delta"], ascending=[False, False]
    ).reset_index(drop=True)
    return summary, common


def run_sequence_challenge_audit(
    engine,
    draft_history: list[dict],
    *,
    forced_first_player_id: str | None = None,
    runs: int = 150,
    through_round: int = 13,
    audit_through_round: int = 8,
    wait_turns: tuple[int, ...] = (1, 2, 3),
    challenger_pool: int = 12,
    seed: int | None = None,
    progress: Callable[[int, int], None] | None = None,
):
    """Challenge every QB1/TE1 selection with multiple RB/WR paths.

    This audit does not change PatBot's production weights. It asks whether a
    QB/TE selection still wins when challenged by the best RB/WR by current score,
    league-winner score, or Q90, with the QB/TE fill delayed 1-3 PatBot turns.
    """
    sim = FastDraftSimulator(engine)
    if seed is None:
        seed = int(sim.comparison_seed) + 9490

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

                role = _pick_role(sim, pre_counts, idx)
                selected_pos = str(sim.pos[idx]).upper()
                selected_quality = starter_quality(sim, idx) if role in {"QB1", "TE1"} else "—"
                actual_patbot_owned = set(getattr(sim, "_patbot_owned_idxs", set())) | {int(idx)}
                actual_opponent_owned = set(getattr(sim, "_opponent_owned_idxs", set()))

                if round_no <= int(audit_through_round) and role in {"QB1", "TE1"}:
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

                    challengers = select_rbwr_challengers(
                        sim,
                        scores,
                        pre_available,
                        top_pool=int(challenger_pool),
                    )
                    for challenger in challengers:
                        challenger_idx = int(challenger["idx"])
                        for horizon in horizons:
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
                            alt_score = float(alt_eval["lineup_score"])

                            rows.append(
                                {
                                    "Run": int(run),
                                    "Round": int(round_no),
                                    "Overall": int(pick),
                                    "Pos": selected_pos,
                                    "Quality": str(selected_quality),
                                    "Selected": str(sim.names[idx]),
                                    "Selected Pos Rank": _position_rank(sim, idx),
                                    "Selected Score": float(scores[idx]),
                                    "Selected VORP": float(sim.vorp[idx]),
                                    "Selected LWS": float(sim.league_winner_score[idx]),
                                    "Selected Q90": float(sim.q90_points[idx]),
                                    "Selected FP ECR": _fantasypros_ecr(sim, idx),
                                    "Selected ADP": float(sim.adp[idx]),
                                    "Challenger Type": str(challenger["Challenger Type"]),
                                    "Challenger": str(challenger["Player"]),
                                    "Challenger Pos": str(challenger["Pos"]),
                                    "Challenger RB/WR Score Rank": int(challenger["RB/WR Score Rank"]),
                                    "Challenger Score": float(challenger["Score"]),
                                    "Challenger Score Gap": float(scores[idx]) - float(challenger["Score"]),
                                    "Challenger VORP": float(challenger["VORP"]),
                                    "Challenger LWS": float(challenger["LWS"]),
                                    "Challenger Q90": float(challenger["Q90"]),
                                    "Challenger FP ECR": float(challenger["FP ECR"]) if np.isfinite(challenger["FP ECR"]) else np.nan,
                                    "Challenger ADP": float(challenger["ADP"]),
                                    "Wait Turns": int(horizon),
                                    "Wait Fill": str(sim.names[fill_idx]) if fill_idx is not None else "—",
                                    "Wait Fill Pos Rank": _position_rank(sim, fill_idx) if fill_idx is not None else np.nan,
                                    "Wait Fill FP ECR": _fantasypros_ecr(sim, fill_idx) if fill_idx is not None else np.nan,
                                    "Wait Fill Overall": int(fill_pick) if fill_pick is not None else np.nan,
                                    "VORP Cost of Waiting": float(sim.vorp[idx]) - float(sim.vorp[fill_idx]) if fill_idx is not None else np.nan,
                                    "Fill-Now Final Lineup": baseline_score,
                                    "Alt Final Lineup": alt_score,
                                    "Alt Delta vs Fill Now": alt_score - baseline_score,
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
    summary, common = _aggregate_sequence_rows(events, int(runs))
    if not events.empty:
        for column in events.select_dtypes(include=["float"]).columns:
            events[column] = events[column].round(2)
    return summary, common, events, {
        "runs": int(runs),
        "seed": int(seed),
        "through_round": int(through_round),
        "audit_through_round": int(audit_through_round),
        "wait_turns": list(horizons),
        "challenger_pool": int(challenger_pool),
    }
