from __future__ import annotations

from collections import Counter, defaultdict
from typing import Callable

import numpy as np
import pandas as pd

from .draft import snake_pick
from .sim import FastDraftSimulator
from .strategy import strategy_phase


DEFAULT_AUDIT_ROUNDS = (8, 10, 12, 13)


def _hard_special_teams_constraints(
    sim: FastDraftSimulator,
    score: np.ndarray,
    roster_counts: np.ndarray,
    round_no: int,
) -> np.ndarray:
    """Apply PatBot's cheap-special-teams philosophy as hard constraints.

    The audit intentionally stops at Round 13 because Round 14 is reserved for
    D/ST and Round 15 for kicker. Before those rounds neither position is a
    legitimate upside selection.
    """
    out = score.copy()
    cfg = sim.cfg.get("special_teams_strategy", {}).get("draft", {})
    defense_round = int(cfg.get("defense_round", 14))
    kicker_round = int(cfg.get("kicker_round", 15))
    target_def = int(cfg.get("rostered_defenses", 1))
    target_k = int(cfg.get("rostered_kickers", 1))

    def_code = sim.pos_to_code.get("DEF")
    k_code = sim.pos_to_code.get("K")

    if round_no < defense_round:
        out[sim.pos == "DEF"] = -1_000_000_000.0
    elif def_code is not None and int(roster_counts[def_code]) >= target_def:
        out[sim.pos == "DEF"] = -1_000_000_000.0

    if round_no < kicker_round:
        out[sim.pos == "K"] = -1_000_000_000.0
    elif k_code is not None and int(roster_counts[k_code]) >= target_k:
        out[sim.pos == "K"] = -1_000_000_000.0

    return out


def _score_vector(
    sim: FastDraftSimulator,
    available: np.ndarray,
    roster_counts: np.ndarray,
    pick: int,
    *,
    upside_weight: float,
    risk_penalty_multiplier: float,
) -> np.ndarray:
    """Reproduce PatBot's pick score while making strategy phase explicit."""
    next_pick = sim._next_my_pick(pick)
    urgency = sim._urgency(next_pick)
    roster_fit = sim._roster_fit_vector(roster_counts)
    scarcity_pct = sim._scarcity_pct(available)

    score = 100.0 * (
        sim.vorp_pct * sim.w_vorp
        + sim.proj_pct * sim.w_proj
        + urgency * sim.w_urgency
        + scarcity_pct * sim.w_scarcity
        + roster_fit * sim.w_roster
        + sim.expert_pct * sim.expert_weight
    )
    score += sim.league_winner_score * float(upside_weight)
    score -= sim.injury * sim.injury_penalty * float(risk_penalty_multiplier)

    round_no = (pick - 1) // sim.teams + 1
    score = _hard_special_teams_constraints(sim, score, roster_counts, round_no)
    return np.where(available, score, -1_000_000_000.0)


def _component_frame(sim: FastDraftSimulator) -> pd.DataFrame:
    metrics = sim.strategy_metrics.copy()
    metrics.index = np.arange(sim.n)

    pos = pd.Series(sim.pos, index=metrics.index)
    replacement = pd.Series(sim.replacement, index=metrics.index)
    q90 = pd.to_numeric(metrics["q90_points"], errors="coerce").fillna(
        pd.Series(sim.proj, index=metrics.index)
    )
    ceiling_vorp = q90 - replacement
    ceiling_score = ceiling_vorp.groupby(pos).rank(method="average", pct=True).fillna(0.5)

    scfg = sim.cfg.get("championship_strategy", {})
    weights = scfg.get("league_winner_components", {})
    w_ceiling = float(weights.get("positional_ceiling", 0.55))
    w_edge = float(weights.get("market_edge", 0.25))
    w_youth = float(weights.get("early_career", 0.20))

    edge = pd.to_numeric(metrics["market_edge_score"], errors="coerce").fillna(0.0)
    youth = pd.to_numeric(metrics["early_career_score"], errors="coerce").fillna(0.0)

    ceiling_contrib = ceiling_score * w_ceiling
    edge_contrib = edge * w_edge
    youth_contrib = youth * w_youth
    total = (ceiling_contrib + edge_contrib + youth_contrib).clip(lower=1e-9)

    out = pd.DataFrame(index=metrics.index)
    out["ceiling_score"] = ceiling_score
    out["market_edge_score"] = edge
    out["early_career_score"] = youth
    out["youth_component_share"] = youth_contrib / total
    out["league_winner_score"] = pd.to_numeric(
        metrics["league_winner_score"], errors="coerce"
    ).fillna(0.0)
    out["q90_points"] = q90
    out["performance_sigma"] = pd.to_numeric(
        metrics["performance_sigma"], errors="coerce"
    ).fillna(0.0)
    return out


def _top(counter: Counter, n: int = 5) -> list[tuple[str, int]]:
    return counter.most_common(n)


def run_late_round_audit(
    engine,
    draft_history: list[dict],
    *,
    rounds: tuple[int, ...] = DEFAULT_AUDIT_ROUNDS,
    runs: int = 500,
    seed: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    youth_only_share_threshold: float = 0.45,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Audit whether late PatBot picks become intelligently upside-seeking.

    The simulated room follows the configured real-manager model. At each target
    PatBot pick we compare the actual round-aware strategy with a Foundation
    counterfactual on the exact same board and roster. The Foundation comparison
    removes the league-winner bonus and restores the full injury-risk penalty.

    This is a strategy calibration diagnostic, not a claim that any listed
    player will break out.
    """
    target_rounds = tuple(sorted({int(r) for r in rounds if 1 <= int(r) <= 13}))
    if not target_rounds:
        raise ValueError("Late-round audit needs at least one target round from 1 through 13.")

    sim = FastDraftSimulator(engine)
    if seed is None:
        seed = int(sim.comparison_seed) + 413
    rng = np.random.default_rng(int(seed))

    drafted_idx: set[int] = set()
    my_idx: list[int] = []
    for record in draft_history:
        pid = str(record.get("player_id", ""))
        idx = sim.id_to_idx.get(pid)
        if idx is None:
            continue
        drafted_idx.add(idx)
        if int(record.get("owner_slot", -1)) == sim.slot:
            my_idx.append(idx)

    start_pick = max([int(x.get("overall_pick", 0)) for x in draft_history] or [0]) + 1
    last_pick = snake_pick(max(target_rounds), sim.teams, sim.slot)
    latent_sd = np.maximum(sim.sd_floor, sim.adp * sim.sd_pct)
    components = _component_frame(sim)

    current_counts: dict[int, Counter] = {r: Counter() for r in target_rounds}
    foundation_counts: dict[int, Counter] = {r: Counter() for r in target_rounds}
    changed_counts = Counter()
    position_counts: dict[int, Counter] = {r: Counter() for r in target_rounds}
    youth_only_counts = Counter()
    selected_lw_scores: dict[int, list[float]] = defaultdict(list)
    selected_youth_shares: dict[int, list[float]] = defaultdict(list)
    baseline_ranks: dict[tuple[int, str], list[int]] = defaultdict(list)

    for run in range(int(runs)):
        available = np.ones(sim.n, dtype=bool)
        if drafted_idx:
            available[list(drafted_idx)] = False

        opp_counts = sim._seed_opponent_counts(draft_history)
        archetypes = sim._archetype_assignments(rng)
        my_counts = np.zeros(len(sim.POSITIONS), dtype=np.int16)
        for idx in my_idx:
            code = sim.pos_code[idx]
            if code >= 0:
                my_counts[code] += 1

        market_latent = np.maximum(1.0, rng.normal(sim.adp, latent_sd))
        custom_noise_base = rng.normal(
            0.0,
            np.maximum(3.0, sim.custom_rank * 0.06),
        )

        for pick in range(start_pick, last_pick + 1):
            if not available.any():
                break

            if pick in sim.my_picks:
                round_no = (pick - 1) // sim.teams + 1
                phase = strategy_phase(round_no, sim.cfg)
                current_scores = _score_vector(
                    sim,
                    available,
                    my_counts,
                    pick,
                    upside_weight=float(phase.get("upside_weight", 0.0)),
                    risk_penalty_multiplier=float(phase.get("risk_penalty_multiplier", 1.0)),
                )
                idx = int(np.argmax(current_scores))

                if round_no in current_counts:
                    foundation_scores = _score_vector(
                        sim,
                        available,
                        my_counts,
                        pick,
                        upside_weight=0.0,
                        risk_penalty_multiplier=1.0,
                    )
                    foundation_idx = int(np.argmax(foundation_scores))
                    name = str(sim.names[idx])
                    foundation_name = str(sim.names[foundation_idx])
                    current_counts[round_no][name] += 1
                    foundation_counts[round_no][foundation_name] += 1
                    changed_counts[round_no] += int(idx != foundation_idx)
                    position_counts[round_no][str(sim.pos[idx])] += 1

                    youth_share = float(components.loc[idx, "youth_component_share"])
                    selected_lw_scores[round_no].append(float(components.loc[idx, "league_winner_score"]))
                    selected_youth_shares[round_no].append(youth_share)
                    if youth_share >= float(youth_only_share_threshold):
                        youth_only_counts[round_no] += 1

                    available_idxs = np.where(available)[0]
                    ordered_foundation = available_idxs[
                        np.argsort(foundation_scores[available_idxs])[::-1]
                    ]
                    rank_lookup = {int(player_idx): rank + 1 for rank, player_idx in enumerate(ordered_foundation)}
                    baseline_ranks[(round_no, name)].append(rank_lookup.get(idx, len(ordered_foundation)))

                available[idx] = False
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

        if progress is not None and ((run + 1) == runs or (run + 1) % max(1, runs // 10) == 0):
            progress(run + 1, int(runs))

    summary_rows = []
    detail_rows = []
    name_to_idx = {str(name): i for i, name in enumerate(sim.names)}

    for round_no in target_rounds:
        phase = strategy_phase(round_no, sim.cfg)
        most_current = _top(current_counts[round_no], 1)
        most_foundation = _top(foundation_counts[round_no], 1)
        current_name, current_n = most_current[0] if most_current else ("—", 0)
        foundation_name, foundation_n = most_foundation[0] if most_foundation else ("—", 0)
        rbwr = position_counts[round_no]["RB"] + position_counts[round_no]["WR"]

        summary_rows.append({
            "Round": round_no,
            "Phase": str(phase.get("name", "Baseline")),
            "Most Common PatBot Pick": current_name,
            "PatBot Pick %": round(100.0 * current_n / max(int(runs), 1), 1),
            "Foundation Counterfactual": foundation_name,
            "Foundation Pick %": round(100.0 * foundation_n / max(int(runs), 1), 1),
            "Strategy Changes Pick %": round(100.0 * changed_counts[round_no] / max(int(runs), 1), 1),
            "RB/WR Pick %": round(100.0 * rbwr / max(int(runs), 1), 1),
            "Avg League Winner Score": round(float(np.mean(selected_lw_scores[round_no])) if selected_lw_scores[round_no] else 0.0, 1),
            "Avg Youth Component Share %": round(100.0 * (float(np.mean(selected_youth_shares[round_no])) if selected_youth_shares[round_no] else 0.0), 1),
            "Youth-Heavy Pick %": round(100.0 * youth_only_counts[round_no] / max(int(runs), 1), 1),
            "Runs": int(runs),
        })

        all_names = set(current_counts[round_no]) | set(foundation_counts[round_no])
        for name in all_names:
            idx = name_to_idx.get(name)
            if idx is None:
                continue
            cur_n = current_counts[round_no][name]
            base_n = foundation_counts[round_no][name]
            cur_pct = 100.0 * cur_n / max(int(runs), 1)
            base_pct = 100.0 * base_n / max(int(runs), 1)
            if max(cur_pct, base_pct) < 1.0:
                continue
            risk = float(sim.injury[idx]) if np.isfinite(sim.injury[idx]) else 0.0
            ranks = baseline_ranks.get((round_no, name), [])
            detail_rows.append({
                "Round": round_no,
                "Player": name,
                "Pos": str(sim.pos[idx]),
                "PatBot Pick %": round(cur_pct, 1),
                "Foundation Pick %": round(base_pct, 1),
                "Strategy Lift pp": round(cur_pct - base_pct, 1),
                "ADP": round(float(sim.adp[idx]), 1),
                "Projection": round(float(sim.proj[idx]), 1),
                "Q90 Points": round(float(components.loc[idx, "q90_points"]), 1),
                "League Winner Score": round(float(components.loc[idx, "league_winner_score"]), 1),
                "Ceiling Score": round(100.0 * float(components.loc[idx, "ceiling_score"]), 1),
                "Market Edge": round(100.0 * float(components.loc[idx, "market_edge_score"]), 1),
                "Early Career": round(100.0 * float(components.loc[idx, "early_career_score"]), 1),
                "Youth Component Share %": round(100.0 * float(components.loc[idx, "youth_component_share"]), 1),
                "Risk Score": round(risk, 3),
                "Performance Sigma": round(float(components.loc[idx, "performance_sigma"]), 3),
                "Avg Foundation Rank When Picked": round(float(np.mean(ranks)), 1) if ranks else np.nan,
                "Youth-Heavy Warning": bool(float(components.loc[idx, "youth_component_share"]) >= float(youth_only_share_threshold)),
            })

    summary = pd.DataFrame(summary_rows)
    details = pd.DataFrame(detail_rows)
    if not details.empty:
        details = details.sort_values(
            ["Round", "PatBot Pick %", "Strategy Lift pp"],
            ascending=[True, False, False],
        ).reset_index(drop=True)

    meta = {
        "rounds": list(target_rounds),
        "runs": int(runs),
        "seed": int(seed),
        "start_pick": int(start_pick),
        "anchor_roster": [str(sim.names[i]) for i in my_idx],
        "youth_only_share_threshold": float(youth_only_share_threshold),
    }
    return summary, details, meta
