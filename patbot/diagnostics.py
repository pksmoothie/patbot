from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json

import numpy as np
import pandas as pd

from .draft import DraftEngine
from .scoring import score_season_projection
from .sim import FastDraftSimulator


OFFENSE_POSITIONS = {"QB", "RB", "WR", "TE"}
ELITE_TE_NAMES = {"Trey McBride", "Brock Bowers"}

DIAGNOSTIC_VARIANTS = [
    (
        "baseline",
        "Baseline",
        "Current PatBot scoring, expert blend, scarcity and lookahead.",
    ),
    (
        "no_completion",
        "No +0.25 completion bonus",
        "Re-scores Sleeper projections with completions worth 0 instead of 0.25.",
    ),
    (
        "no_yardage_bonuses",
        "No yardage threshold bonuses",
        "Removes PatBot's expected passing, rushing and receiving yardage bonuses.",
    ),
    (
        "pprish",
        "No completion + no yardage bonuses",
        "A PPR-ish sensitivity case that removes the two most unusual scoring features.",
    ),
    (
        "no_scarcity",
        "No scarcity weight",
        "Sets the four-player positional-drop weight to zero while leaving scoring intact.",
    ),
    (
        "no_expert",
        "No expert-rank signal",
        "Removes the 10% independent/blended expert signal while leaving projections and market intact.",
    ),
]


def _parse_raw_stats(value) -> dict:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    try:
        if pd.isna(value):
            return {}
    except (TypeError, ValueError):
        pass
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _rescore_players(players: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, int]:
    if "raw_stats_json" not in players.columns:
        raise ValueError(
            "This player snapshot predates scoring diagnostics. Refresh live 2026 data "
            "with v0.3.9 so raw Sleeper stat lines are stored locally."
        )

    out = players.copy()
    rescored = 0
    for idx, row in out.iterrows():
        pos = str(row.get("pos", "")).upper()
        if pos not in OFFENSE_POSITIONS:
            continue
        stats = _parse_raw_stats(row.get("raw_stats_json"))
        if not stats:
            continue
        scored = score_season_projection(
            stats,
            scoring=config["scoring"],
            bonus_model=config.get("bonus_model", {}),
            position=pos,
        )
        out.at[idx, "proj_points"] = scored["custom_points"]
        if "base_custom_points" in out.columns:
            out.at[idx, "base_custom_points"] = scored["base_points"]
        if "estimated_bonus_points" in out.columns:
            out.at[idx, "estimated_bonus_points"] = scored["bonus_points"]
        rescored += 1

    if rescored == 0:
        raise ValueError(
            "No offensive players could be re-scored. Refresh live 2026 data and try again."
        )
    return out, rescored


def build_diagnostic_engine(base_engine: DraftEngine, variant: str) -> tuple[DraftEngine, int]:
    if variant == "baseline":
        return base_engine, 0

    cfg = deepcopy(base_engine.config)
    players = base_engine.players.copy()
    rescored = 0

    if variant in {"no_completion", "pprish"}:
        cfg["scoring"]["pass_completion"] = 0.0

    if variant in {"no_yardage_bonuses", "pprish"}:
        cfg["scoring"]["pass_yard_bonuses"] = []
        cfg["scoring"]["rush_yard_bonuses"] = []
        cfg["scoring"]["rec_yard_bonuses"] = []

    if variant in {"no_completion", "no_yardage_bonuses", "pprish"}:
        players, rescored = _rescore_players(players, cfg)

    if variant == "no_scarcity":
        cfg["draft_engine"]["weights"]["scarcity"] = 0.0

    if variant == "no_expert":
        players["expert_rank"] = np.nan

    return DraftEngine(players, cfg), rescored


def _pct(count: int, total: int) -> float:
    return round(100.0 * float(count) / float(total or 1), 1)


def _top(counter: Counter, n: int = 5) -> list[dict]:
    total = sum(counter.values()) or 1
    return [
        {"player": name, "pct": round(100.0 * count / total, 1)}
        for name, count in counter.most_common(n)
    ]


def _simulate_focus(
    sim: FastDraftSimulator,
    current_pick: int,
    drafted_ids: set[str],
    my_roster_ids: list[str],
    candidate_id: str,
    runs: int,
    through_round: int,
    seed: int,
    draft_history: list[dict] | None,
) -> dict:
    drafted_idx = {
        sim.id_to_idx[str(x)]
        for x in drafted_ids
        if str(x) in sim.id_to_idx
    }
    my_idx = [
        sim.id_to_idx[str(x)]
        for x in my_roster_ids
        if str(x) in sim.id_to_idx
    ]
    candidate_idx = sim.id_to_idx[str(candidate_id)]

    rng = np.random.default_rng(seed)
    last_pick = sim.teams * int(through_round)
    lineup_scores = np.empty(runs, dtype=float)
    second_names = Counter()
    third_names = Counter()
    second_positions = Counter()
    third_positions = Counter()

    base_available = np.ones(sim.n, dtype=bool)
    if drafted_idx:
        base_available[list(drafted_idx)] = False
    latent_sd = np.maximum(sim.sd_floor, sim.adp * sim.sd_pct)

    for run in range(runs):
        available = base_available.copy()
        opp_counts = sim._seed_opponent_counts(draft_history)
        archetypes = sim._archetype_assignments(rng)

        my_counts = np.zeros(len(sim.POSITIONS), dtype=np.int16)
        mine = list(my_idx)
        for idx in mine:
            code = sim.pos_code[idx]
            if code >= 0:
                my_counts[code] += 1

        market_latent = np.maximum(1.0, rng.normal(sim.adp, latent_sd))
        custom_noise_base = rng.normal(
            0.0,
            np.maximum(3.0, sim.custom_rank * 0.06),
        )
        followup_pick_no = 0

        for pick in range(int(current_pick), last_pick + 1):
            if not available.any():
                break

            if pick in sim.my_picks:
                if pick == int(current_pick):
                    idx = candidate_idx
                    if not available[idx]:
                        break
                else:
                    idx = sim._lookahead_pick(
                        available,
                        my_counts,
                        pick,
                        opp_counts,
                        archetypes,
                        market_latent,
                        custom_noise_base,
                    )

                available[idx] = False
                mine.append(idx)
                code = sim.pos_code[idx]
                if code >= 0:
                    my_counts[code] += 1

                followup_pick_no += 1
                if followup_pick_no == 2:
                    second_names[sim.names[idx]] += 1
                    second_positions[sim.pos[idx]] += 1
                elif followup_pick_no == 3:
                    third_names[sim.names[idx]] += 1
                    third_positions[sim.pos[idx]] += 1
            else:
                sim._take_opponent_pick(
                    pick,
                    available,
                    opp_counts,
                    archetypes,
                    market_latent,
                    custom_noise_base,
                )

        lineup_scores[run] = sim.evaluate_roster(mine)["lineup_score"]

    second_total = sum(second_names.values())
    third_total = sum(third_names.values())
    elite_te_count = sum(second_names[name] for name in ELITE_TE_NAMES)

    return {
        "candidate": sim.names[candidate_idx],
        "avg_lineup_score": round(float(np.mean(lineup_scores)), 2),
        "second_pick_te_pct": _pct(second_positions["TE"], second_total),
        "second_pick_elite_te_pct": _pct(elite_te_count, second_total),
        "second_pick_mcbride_pct": _pct(second_names["Trey McBride"], second_total),
        "second_pick_bowers_pct": _pct(second_names["Brock Bowers"], second_total),
        "third_pick_qb_pct": _pct(third_positions["QB"], third_total),
        "third_pick_allen_pct": _pct(third_names["Josh Allen"], third_total),
        "most_common_second_pick": _top(second_names),
        "most_common_third_pick": _top(third_names),
    }


def run_model_diagnostics(
    engine: DraftEngine,
    current_pick: int,
    drafted_ids: set[str],
    my_roster_ids: list[str],
    candidate_id: str,
    runs: int = 300,
    through_round: int = 8,
    draft_history: list[dict] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    details = []
    seed = int(engine.config.get("simulation", {}).get("comparison_seed", 20260818))

    for key, label, description in DIAGNOSTIC_VARIANTS:
        variant_engine, rescored = build_diagnostic_engine(engine, key)
        sim = FastDraftSimulator(variant_engine)
        result = _simulate_focus(
            sim,
            current_pick=current_pick,
            drafted_ids=drafted_ids,
            my_roster_ids=my_roster_ids,
            candidate_id=str(candidate_id),
            runs=int(runs),
            through_round=int(through_round),
            seed=seed,
            draft_history=draft_history,
        )
        result.update({
            "scenario_key": key,
            "scenario": label,
            "description": description,
            "rescored_players": rescored,
        })
        details.append(result)
        rows.append({
            "Scenario": label,
            "Avg Lineup Score": result["avg_lineup_score"],
            "2nd Pick TE %": result["second_pick_te_pct"],
            "2nd Pick McBride/Bowers %": result["second_pick_elite_te_pct"],
            "3rd Pick QB %": result["third_pick_qb_pct"],
            "3rd Pick Josh Allen %": result["third_pick_allen_pct"],
            "Runs": int(runs),
        })

    baseline_te = rows[0]["2nd Pick McBride/Bowers %"]
    baseline_allen = rows[0]["3rd Pick Josh Allen %"]
    for row in rows:
        row["Δ Elite TE pp"] = round(row["2nd Pick McBride/Bowers %"] - baseline_te, 1)
        row["Δ Allen pp"] = round(row["3rd Pick Josh Allen %"] - baseline_allen, 1)

    columns = [
        "Scenario",
        "Avg Lineup Score",
        "2nd Pick TE %",
        "2nd Pick McBride/Bowers %",
        "Δ Elite TE pp",
        "3rd Pick QB %",
        "3rd Pick Josh Allen %",
        "Δ Allen pp",
        "Runs",
    ]
    return pd.DataFrame(rows)[columns], details
