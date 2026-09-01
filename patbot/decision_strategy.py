from __future__ import annotations

from collections import Counter
from functools import lru_cache
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


_INSTALLED = False
_NEG_INF = -1_000_000_000.0
_OFFENSE = ("QB", "RB", "WR", "TE")


_DEFAULTS = {
    "enabled": True,
    "strategic_prior_min_round": 4,
    "final_call": {
        "max_candidates": 6,
        "primary_position_candidates": 3,
        "secondary_position_candidates": 1,
        "global_value_candidates": 2,
        "co_priority_pressure_gap": 0.10,
        "value_escape_quality_edge": 6.0,
        "value_escape_extra_per_bench_slot": 4.0,
    },
    "position_pressure": {
        "need_weight": 0.60,
        "wait_cost_weight": 0.30,
        "opponent_demand_weight": 0.10,
        "missing_one_signal": 0.55,
        "additional_missing_signal": 0.15,
        "open_flex_signal": 0.35,
        "rb_wr_bench_signal": 0.10,
        "raw_scarcity_scale": 25.0,
        "quality_wait_cost_scale": 10.0,
        "opponent_demand_scale": 3.0,
        "qb_viable_quality_gap": 8.0,
        "qb_buffer_penalty_per_extra_option": 0.08,
        "qb_buffer_penalty_max": 0.24,
        "strategic_pressure_points": 15.0,
    },
    "single_qb": {
        "neutralize_player_survival_within_position": True,
        "comparison_pool_size": 5,
    },
    "opponent_demand": {
        "historical_total_qbs_last_draft": 13,
        "historical_rounds": 15,
        "generic_qb2_expected_rate_per_team": 0.04,
        "promoted_qb2_owner_expected_rate": 0.28,
        "te2_expected_rate_per_team": 0.10,
        "filled_rb_wr_pick_rate": 0.30,
        "missing_rb_wr_pick_rate": 0.78,
    },
    "opponent_qb2": {
        "penalty_through_round_11": 34.0,
        "penalty_round_12_plus": 24.0,
        "qb3_plus_penalty": 120.0,
    },
}


def _deep_merge(base: dict, override: dict | None) -> dict:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@lru_cache(maxsize=1)
def load_decision_strategy() -> dict:
    path = Path(__file__).resolve().parents[1] / "config" / "decision_strategy.yaml"
    if not path.exists():
        return _deep_merge({}, _DEFAULTS)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        payload = {}
    return _deep_merge(_DEFAULTS, payload)


def decision_strategy_settings(config: dict | None = None) -> dict:
    settings = _deep_merge(_DEFAULTS, load_decision_strategy())
    supplied = (config or {}).get("decision_strategy", {}) or {}
    if isinstance(supplied, dict):
        settings = _deep_merge(settings, supplied)
    return settings


def _score_component_coefficients(config: dict, *, expert_active: bool) -> dict[str, float]:
    engine_cfg = config.get("draft_engine", {}) or {}
    w = engine_cfg.get("weights", {}) or {}
    base_total = sum(float(w.get(k, 0.0)) for k in ("vorp", "projection", "urgency", "scarcity", "roster_fit"))
    if base_total <= 0:
        return {"urgency": 0.0, "scarcity": 0.0, "roster_fit": 0.0}
    expert_weight = 0.10 if expert_active else 0.0
    scale = (1.0 - expert_weight) / base_total
    return {
        "urgency": 100.0 * float(w.get("urgency", 0.0)) * scale,
        "scarcity": 100.0 * float(w.get("scarcity", 0.0)) * scale,
        "roster_fit": 100.0 * float(w.get("roster_fit", 0.0)) * scale,
    }


def _market_urgency(board: pd.DataFrame) -> pd.Series:
    if "survive_next" in board.columns:
        survive = pd.to_numeric(board["survive_next"], errors="coerce").fillna(0.5)
        return (1.0 - survive).clip(0.0, 1.0)
    if "survive_next_pct" in board.columns:
        survive = pd.to_numeric(board["survive_next_pct"], errors="coerce").fillna(50.0) / 100.0
        return (1.0 - survive).clip(0.0, 1.0)
    return pd.Series(0.5, index=board.index, dtype=float)


def _scarcity_percentile(board: pd.DataFrame) -> pd.Series:
    if "scarcity_pct" in board.columns:
        return pd.to_numeric(board["scarcity_pct"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    if "scarcity" not in board.columns or board.empty:
        return pd.Series(0.5, index=board.index, dtype=float)
    vals = pd.to_numeric(board["scarcity"], errors="coerce").fillna(0.0)
    return vals.rank(method="average", pct=True).fillna(0.5)


def adjust_single_qb_board_scores(
    board: pd.DataFrame,
    config: dict,
    *,
    roster_positions: list[str] | tuple[str, ...],
    round_no: int,
) -> pd.DataFrame:
    """Remove player-specific survival as a tiebreaker among QB1 candidates.

    In PatBot's one-QB strategy, once we choose a quarterback we will not draft a
    second one. Individual survival therefore helps decide whether the QB *pool*
    can wait, but it should not make Dart beat Nix merely because Dart is more
    likely to be selected before the next PatBot pick.
    """
    if board is None or board.empty:
        return pd.DataFrame() if board is None else board.copy()

    out = board.copy()
    settings = decision_strategy_settings(config)
    expert_active = "expert_rank" in out.columns and out["expert_rank"].notna().any()
    coeff = _score_component_coefficients(config, expert_active=expert_active)

    score = pd.to_numeric(out["score"], errors="coerce").fillna(_NEG_INF)
    urgency = _market_urgency(out)
    effective_urgency = urgency.copy()
    adjustment = pd.Series(0.0, index=out.index, dtype=float)

    counts = Counter(str(p).upper() for p in roster_positions)
    qb_cfg = settings.get("single_qb", {}) or {}
    active = (
        bool(settings.get("enabled", True))
        and int(round_no) >= int(settings.get("strategic_prior_min_round", 4))
        and bool(qb_cfg.get("neutralize_player_survival_within_position", True))
        and counts["QB"] == 0
    )

    qb_mask = out["pos"].astype(str).str.upper().eq("QB")
    if active and qb_mask.any() and coeff["urgency"] > 0:
        qb_idx = out.index[qb_mask]
        pre_urgency_quality = score.loc[qb_idx] - coeff["urgency"] * urgency.loc[qb_idx]
        pool_n = max(1, int(qb_cfg.get("comparison_pool_size", 5)))
        pool_idx = pre_urgency_quality.sort_values(ascending=False).head(pool_n).index
        common_urgency = float(urgency.loc[pool_idx].median())
        effective_urgency.loc[qb_idx] = common_urgency
        adjustment.loc[qb_idx] = coeff["urgency"] * (common_urgency - urgency.loc[qb_idx])
        score.loc[qb_idx] = score.loc[qb_idx] + adjustment.loc[qb_idx]

    out["score"] = score.round(2)
    out["decision_single_qb_urgency_adjustment"] = adjustment.round(3)
    out["decision_effective_urgency"] = effective_urgency.round(4)
    out["decision_urgency_component"] = (effective_urgency * coeff["urgency"]).round(3)

    roster_fit = pd.to_numeric(
        out.get("roster_fit", pd.Series(0.0, index=out.index)), errors="coerce"
    ).fillna(0.0)
    scarcity_pct = _scarcity_percentile(out)
    out["decision_roster_fit_component"] = (roster_fit * coeff["roster_fit"]).round(3)
    out["decision_scarcity_component"] = (scarcity_pct * coeff["scarcity"]).round(3)

    # Player quality deliberately strips the three timing/construction terms.
    # Projection, VORP, expert signal, upside, risk and concentration remain.
    out["decision_quality_score"] = (
        pd.to_numeric(out["score"], errors="coerce").fillna(_NEG_INF)
        - out["decision_urgency_component"]
        - out["decision_roster_fit_component"]
        - out["decision_scarcity_component"]
    ).round(3)
    return out


def _team_slot_for_pick(pick: int, teams: int) -> int:
    round_no = (int(pick) - 1) // int(teams) + 1
    within = (int(pick) - 1) % int(teams) + 1
    return within if round_no % 2 else int(teams) + 1 - within


def _next_patbot_pick(current_pick: int, teams: int, slot: int) -> int:
    for round_no in range(1, 25):
        within = int(slot) if round_no % 2 else int(teams) + 1 - int(slot)
        overall = (round_no - 1) * int(teams) + within
        if overall > int(current_pick):
            return overall
    return int(current_pick) + int(teams) * 2


def _my_roster_counts(engine, my_roster_ids: list[str] | tuple[str, ...]) -> Counter:
    counts = Counter()
    if not hasattr(engine, "players") or "player_id" not in engine.players.columns:
        return counts
    id_to_pos = dict(
        zip(
            engine.players["player_id"].astype(str),
            engine.players["pos"].astype(str).str.upper(),
        )
    )
    for pid in my_roster_ids:
        pos = id_to_pos.get(str(pid))
        if pos:
            counts[pos] += 1
    return counts


def _history_counts(draft_history: list[dict] | None) -> dict[int, Counter]:
    out: dict[int, Counter] = {}
    for item in draft_history or []:
        try:
            slot = int(item.get("owner_slot"))
        except (TypeError, ValueError):
            continue
        pos = str(item.get("pos", "")).upper()
        if not pos:
            continue
        out.setdefault(slot, Counter())[pos] += 1
    return out


def _promoted_qb2_owner(engine, slot: int) -> bool:
    raw = engine.config.get("opponent_managers", {}).get(slot) or engine.config.get("opponent_managers", {}).get(str(slot)) or {}
    name = str(raw.get("name", "")) if isinstance(raw, dict) else ""
    if not name:
        return False
    try:
        from .opponent_history import load_owner_history

        tendency = load_owner_history().get(name, {}) or {}
        return bool(tendency.get("second_qb_bonus"))
    except Exception:
        return False


def expected_position_demand(
    engine,
    *,
    pos: str,
    current_pick: int,
    draft_history: list[dict] | None,
) -> float:
    """Estimate how many players at one position opponents can consume before us.

    QB is roster-aware at the pool level: each opponent missing QB1 can create at
    most one unit of meaningful demand, while generic QB2 demand stays tiny and
    only promoted historical backup-QB owners receive a larger rate.
    """
    position = str(pos).upper()
    teams = int(engine.league["teams"])
    slot = int(engine.league["draft_slot"])
    next_pick = _next_patbot_pick(int(current_pick), teams, slot)
    upcoming = Counter(
        _team_slot_for_pick(pick, teams)
        for pick in range(int(current_pick) + 1, int(next_pick))
        if _team_slot_for_pick(pick, teams) != slot
    )
    if not upcoming:
        return 0.0

    hist = _history_counts(draft_history)
    starters = int(engine.roster_cfg.get(position, 0))
    settings = decision_strategy_settings(engine.config)
    demand_cfg = settings.get("opponent_demand", {}) or {}
    total = 0.0

    for owner_slot, number_of_picks in upcoming.items():
        count = int(hist.get(int(owner_slot), Counter()).get(position, 0))
        picks = int(number_of_picks)
        if position == "QB":
            if count < max(1, starters):
                total += min(float(picks), float(max(1, starters) - count))
                if picks > 1:
                    rate = (
                        float(demand_cfg.get("promoted_qb2_owner_expected_rate", 0.28))
                        if _promoted_qb2_owner(engine, int(owner_slot))
                        else float(demand_cfg.get("generic_qb2_expected_rate_per_team", 0.04))
                    )
                    total += min(1.0, rate)
            elif count == 1:
                rate = (
                    float(demand_cfg.get("promoted_qb2_owner_expected_rate", 0.28))
                    if _promoted_qb2_owner(engine, int(owner_slot))
                    else float(demand_cfg.get("generic_qb2_expected_rate_per_team", 0.04))
                )
                total += min(1.0, rate)
            continue

        if position == "TE":
            missing = max(starters - count, 0)
            first = min(picks, missing)
            total += 0.85 * float(first)
            if picks > first and count + first >= starters:
                total += float(demand_cfg.get("te2_expected_rate_per_team", 0.10))
            continue

        if position in {"RB", "WR"}:
            missing = max(starters - count, 0)
            starter_picks = min(picks, missing)
            bench_picks = max(picks - starter_picks, 0)
            total += float(demand_cfg.get("missing_rb_wr_pick_rate", 0.78)) * starter_picks
            total += float(demand_cfg.get("filled_rb_wr_pick_rate", 0.30)) * bench_picks
            continue

        if count < starters:
            total += min(picks, starters - count)

    return float(total)


def _expected_quality_after_demand(qualities: list[float], expected_demand: float) -> float:
    if not qualities:
        return 0.0
    ordered = sorted((float(x) for x in qualities), reverse=True)
    if len(ordered) == 1 or expected_demand <= 0:
        return ordered[0]

    rank = min(float(expected_demand), float(len(ordered) - 1))
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    fraction = rank - lo
    return ordered[lo] * (1.0 - fraction) + ordered[hi] * fraction


def position_pressure_table(
    board: pd.DataFrame,
    engine,
    *,
    current_pick: int,
    my_roster_ids: list[str] | tuple[str, ...],
    draft_history: list[dict] | None,
) -> pd.DataFrame:
    """Rank positions by roster need, wait cost and actual room demand."""
    if board is None or board.empty or not hasattr(engine, "roster_cfg"):
        return pd.DataFrame()

    settings = decision_strategy_settings(engine.config)
    pcfg = settings.get("position_pressure", {}) or {}
    counts = _my_roster_counts(engine, my_roster_ids)
    starters = {p: int(engine.roster_cfg.get(p, 0)) for p in _OFFENSE}
    flex_eligible = {str(p).upper() for p in engine.roster_cfg.get("flex_eligible", ["RB", "WR", "TE"])}
    flex_need = int(engine.roster_cfg.get("FLEX", 0))
    current_flex_excess = sum(max(int(counts[p]) - int(starters[p]), 0) for p in flex_eligible)
    flex_open = current_flex_excess < flex_need

    quality_col = "decision_quality_score" if "decision_quality_score" in board.columns else "score"
    work = board.copy()
    work[quality_col] = pd.to_numeric(work[quality_col], errors="coerce").fillna(_NEG_INF)
    work["_pos"] = work["pos"].astype(str).str.upper()

    rows = []
    for pos in _OFFENSE:
        group = work[work["_pos"].eq(pos)].copy()
        if group.empty:
            continue
        group = group.sort_values(quality_col, ascending=False)
        qualities = group[quality_col].astype(float).tolist()
        best_quality = float(qualities[0])
        count = int(counts[pos])
        starter_need = int(starters[pos])
        missing = max(starter_need - count, 0)

        if missing > 0:
            need_signal = float(pcfg.get("missing_one_signal", 0.55))
            need_signal += float(pcfg.get("additional_missing_signal", 0.15)) * max(missing - 1, 0)
            need_signal = min(1.0, need_signal)
        elif pos in flex_eligible and flex_open:
            need_signal = float(pcfg.get("open_flex_signal", 0.35))
        elif pos in {"RB", "WR"}:
            need_signal = float(pcfg.get("rb_wr_bench_signal", 0.10))
        else:
            need_signal = 0.0

        demand = expected_position_demand(
            engine,
            pos=pos,
            current_pick=int(current_pick),
            draft_history=draft_history,
        )
        future_quality = _expected_quality_after_demand(qualities, demand)
        quality_drop = max(0.0, best_quality - future_quality)
        quality_wait = min(
            1.0,
            quality_drop / max(float(pcfg.get("quality_wait_cost_scale", 10.0)), 0.01),
        )

        if pos == "QB":
            wait_signal = quality_wait
        else:
            scarcity = pd.to_numeric(
                group.get("scarcity", pd.Series(0.0, index=group.index)), errors="coerce"
            ).fillna(0.0)
            raw_scarcity = float(scarcity.iloc[0]) if len(scarcity) else 0.0
            scarcity_signal = min(
                1.0,
                max(0.0, raw_scarcity) / max(float(pcfg.get("raw_scarcity_scale", 25.0)), 0.01),
            )
            wait_signal = max(quality_wait, scarcity_signal)

        demand_signal = min(
            1.0,
            max(0.0, demand) / max(float(pcfg.get("opponent_demand_scale", 3.0)), 0.01),
        )
        pressure = (
            float(pcfg.get("need_weight", 0.60)) * need_signal
            + float(pcfg.get("wait_cost_weight", 0.30)) * wait_signal
            + float(pcfg.get("opponent_demand_weight", 0.10)) * demand_signal
        )

        viable_count = None
        supply_buffer = None
        buffer_penalty = 0.0
        if pos == "QB" and count == 0:
            viable_gap = float(pcfg.get("qb_viable_quality_gap", 8.0))
            viable_count = int(sum(q >= best_quality - viable_gap for q in qualities))
            supply_buffer = max(0.0, float(viable_count) - math.ceil(max(demand, 0.0)))
            extra_buffer = max(0.0, supply_buffer - 1.0)
            buffer_penalty = min(
                float(pcfg.get("qb_buffer_penalty_max", 0.24)),
                extra_buffer * float(pcfg.get("qb_buffer_penalty_per_extra_option", 0.08)),
            )
            pressure -= buffer_penalty
        elif pos == "QB" and count >= 1:
            pressure = 0.0

        rows.append(
            {
                "pos": pos,
                "pressure": round(float(np.clip(pressure, 0.0, 1.0)), 4),
                "need_signal": round(float(need_signal), 4),
                "wait_signal": round(float(wait_signal), 4),
                "expected_demand": round(float(demand), 3),
                "demand_signal": round(float(demand_signal), 4),
                "quality_drop_if_wait": round(float(quality_drop), 3),
                "viable_pool": viable_count,
                "supply_buffer": None if supply_buffer is None else round(float(supply_buffer), 3),
                "buffer_penalty": round(float(buffer_penalty), 4),
            }
        )

    return pd.DataFrame(rows).sort_values(["pressure", "pos"], ascending=[False, True]).reset_index(drop=True)


def _prospective_bench_depth(engine, counts: Counter, pos: str) -> int:
    position = str(pos).upper()
    starters = {p: int(engine.roster_cfg.get(p, 0)) for p in _OFFENSE}
    if position not in starters:
        return 0
    count = int(counts[position])
    if count < starters[position]:
        return 0

    flex_eligible = {str(p).upper() for p in engine.roster_cfg.get("flex_eligible", ["RB", "WR", "TE"])}
    if position not in flex_eligible:
        return 1
    flex_need = int(engine.roster_cfg.get("FLEX", 0))
    flex_excess = sum(max(int(counts[p]) - int(starters[p]), 0) for p in flex_eligible)
    if flex_excess < flex_need:
        return 0

    positional_excess = max(count - starters[position], 0)
    return 1 + max(0, positional_excess - flex_need)


def build_final_call_plan(
    board: pd.DataFrame,
    engine,
    *,
    current_pick: int,
    my_roster_ids: list[str] | tuple[str, ...],
    draft_history: list[dict] | None,
) -> dict:
    """Build a position-first Final Call shortlist with explicit value escapes."""
    if board is None or board.empty:
        return {
            "shortlist": pd.DataFrame(),
            "base_row": None,
            "pressure": pd.DataFrame(),
            "priority_positions": [],
            "raw_base_row": None,
            "strategy_active": False,
        }

    settings = decision_strategy_settings(getattr(engine, "config", {}))
    raw = board.copy().reset_index(drop=True)
    raw_base_row = raw.iloc[0]
    teams = int(getattr(engine, "league", {}).get("teams", 12))
    round_no = ((int(current_pick) - 1) // teams) + 1
    active = (
        bool(settings.get("enabled", True))
        and round_no >= int(settings.get("strategic_prior_min_round", 4))
        and hasattr(engine, "roster_cfg")
        and hasattr(engine, "players")
    )
    if not active:
        return {
            "shortlist": raw,
            "base_row": raw_base_row,
            "pressure": pd.DataFrame(),
            "priority_positions": [],
            "raw_base_row": raw_base_row,
            "strategy_active": False,
        }

    roster_positions = []
    if hasattr(engine, "players") and "player_id" in engine.players.columns:
        id_to_pos = dict(
            zip(
                engine.players["player_id"].astype(str),
                engine.players["pos"].astype(str).str.upper(),
            )
        )
        roster_positions = [id_to_pos[str(pid)] for pid in my_roster_ids if str(pid) in id_to_pos]

    if "decision_quality_score" not in raw.columns:
        raw = adjust_single_qb_board_scores(
            raw,
            getattr(engine, "config", {}),
            roster_positions=roster_positions,
            round_no=round_no,
        )

    pressure = position_pressure_table(
        raw,
        engine,
        current_pick=int(current_pick),
        my_roster_ids=my_roster_ids,
        draft_history=draft_history,
    )
    if pressure.empty:
        return {
            "shortlist": raw,
            "base_row": raw.iloc[0],
            "pressure": pressure,
            "priority_positions": [],
            "raw_base_row": raw.iloc[0],
            "strategy_active": False,
        }

    fcfg = settings.get("final_call", {}) or {}
    pcfg = settings.get("position_pressure", {}) or {}
    pressure_lookup = dict(zip(pressure["pos"], pressure["pressure"]))
    max_pressure = float(pressure.iloc[0]["pressure"])
    co_gap = max(0.0, float(fcfg.get("co_priority_pressure_gap", 0.10)))
    priority_positions = [
        str(row["pos"])
        for _, row in pressure.iterrows()
        if max_pressure - float(row["pressure"]) <= co_gap
    ]
    if not priority_positions:
        priority_positions = [str(pressure.iloc[0]["pos"])]

    work = raw.copy()
    work["_pos"] = work["pos"].astype(str).str.upper()
    quality = pd.to_numeric(work["decision_quality_score"], errors="coerce").fillna(_NEG_INF)
    work["decision_position_pressure"] = work["_pos"].map(pressure_lookup).fillna(0.0)
    work["decision_strategic_score"] = (
        quality
        + work["decision_position_pressure"] * float(pcfg.get("strategic_pressure_points", 15.0))
    ).round(3)

    priority_pool = work[work["_pos"].isin(priority_positions)].copy()
    if priority_pool.empty:
        strategic = work.sort_values("decision_strategic_score", ascending=False).iloc[0]
    else:
        strategic = priority_pool.sort_values(
            ["decision_strategic_score", "decision_quality_score"], ascending=[False, False]
        ).iloc[0]

    counts = _my_roster_counts(engine, my_roster_ids)
    global_best = work.sort_values("decision_quality_score", ascending=False).iloc[0]
    if str(global_best["_pos"]) not in priority_positions:
        quality_edge = float(global_best["decision_quality_score"] - strategic["decision_quality_score"])
        bench_depth = _prospective_bench_depth(engine, counts, str(global_best["_pos"]))
        hurdle = (
            float(fcfg.get("value_escape_quality_edge", 6.0))
            + bench_depth * float(fcfg.get("value_escape_extra_per_bench_slot", 4.0))
        )
        if quality_edge >= hurdle:
            strategic = global_best

    primary_pos = str(pressure.iloc[0]["pos"])
    secondary_pos = str(pressure.iloc[1]["pos"]) if len(pressure) > 1 else None
    selected: list[str] = []

    def add_rows(frame: pd.DataFrame, n: int) -> None:
        if n <= 0 or frame.empty:
            return
        ordered = frame.sort_values(
            ["decision_quality_score", "decision_strategic_score"], ascending=[False, False]
        )
        for pid in ordered.head(int(n))["player_id"].astype(str):
            if pid not in selected:
                selected.append(pid)

    strategic_id = str(strategic["player_id"])
    selected.append(strategic_id)
    add_rows(
        work[work["_pos"].eq(primary_pos)],
        int(fcfg.get("primary_position_candidates", 3)),
    )
    if secondary_pos:
        add_rows(
            work[work["_pos"].eq(secondary_pos)],
            int(fcfg.get("secondary_position_candidates", 1)),
        )
    add_rows(work, int(fcfg.get("global_value_candidates", 2)))

    raw_id = str(raw_base_row["player_id"])
    if raw_id not in selected:
        selected.append(raw_id)

    max_candidates = max(2, int(fcfg.get("max_candidates", 6)))
    selected = selected[:max_candidates]
    if raw_id not in selected and raw_id != strategic_id:
        if len(selected) >= max_candidates:
            selected[-1] = raw_id
        else:
            selected.append(raw_id)

    shortlist = work.set_index(work["player_id"].astype(str), drop=False).loc[selected].reset_index(drop=True)
    base_row = shortlist[shortlist["player_id"].astype(str).eq(strategic_id)].iloc[0]
    return {
        "shortlist": shortlist,
        "base_row": base_row,
        "pressure": pressure,
        "priority_positions": priority_positions,
        "raw_base_row": raw_base_row,
        "strategy_active": True,
    }


def _neutralize_sim_qb_urgency(sim, score: np.ndarray, available: np.ndarray, roster_counts: np.ndarray, pick: int) -> np.ndarray:
    settings = decision_strategy_settings(sim.cfg)
    qb_cfg = settings.get("single_qb", {}) or {}
    round_no = (int(pick) - 1) // sim.teams + 1
    qb_code = sim.pos_to_code.get("QB")
    if (
        not bool(settings.get("enabled", True))
        or round_no < int(settings.get("strategic_prior_min_round", 4))
        or not bool(qb_cfg.get("neutralize_player_survival_within_position", True))
        or qb_code is None
        or int(roster_counts[qb_code]) >= 1
    ):
        return score

    out = np.asarray(score, dtype=float).copy()
    mask = np.asarray(available, dtype=bool) & (sim.pos == "QB") & (out > _NEG_INF / 2)
    idx = np.where(mask)[0]
    if len(idx) < 2:
        return out

    urgency = sim._urgency(sim._next_my_pick(int(pick)))
    coeff = 100.0 * float(sim.w_urgency)
    quality = out[idx] - coeff * urgency[idx]
    pool_n = max(1, int(qb_cfg.get("comparison_pool_size", 5)))
    order = idx[np.argsort(quality)[::-1]][:pool_n]
    common = float(np.median(urgency[order]))
    out[idx] += coeff * (common - urgency[idx])
    return out


def install_decision_strategy_patch() -> None:
    """Install v0.6.12 decision logic after all prior production patches."""
    global _INSTALLED
    if _INSTALLED:
        return

    from .draft import DraftEngine
    from .sim import FastDraftSimulator

    original_recommend = DraftEngine.recommend
    original_score_vector = FastDraftSimulator._patbot_score_vector
    original_need_penalty = FastDraftSimulator._base_roster_need_penalty

    def recommend_position_aware(self, current_pick, drafted_ids, roster_positions, top_n=12):
        full = original_recommend(
            self,
            current_pick=current_pick,
            drafted_ids=drafted_ids,
            roster_positions=roster_positions,
            top_n=max(int(top_n), len(self.players)),
        )
        if full.empty:
            return full
        round_no = ((int(current_pick) - 1) // int(self.league["teams"])) + 1
        full = adjust_single_qb_board_scores(
            full,
            self.config,
            roster_positions=list(roster_positions),
            round_no=round_no,
        )
        return full.sort_values(
            ["score", "proj_points", "adp"], ascending=[False, False, True]
        ).head(int(top_n)).reset_index(drop=True)

    def score_vector_position_aware(self, available, roster_counts, pick):
        score = original_score_vector(self, available, roster_counts, pick)
        return _neutralize_sim_qb_urgency(
            self,
            score,
            np.asarray(available, dtype=bool),
            np.asarray(roster_counts),
            int(pick),
        )

    def roster_need_penalty_league_calibrated(self, roster_counts, round_no):
        penalty = np.asarray(original_need_penalty(self, roster_counts, round_no), dtype=float).copy()
        settings = decision_strategy_settings(self.cfg)
        qcfg = settings.get("opponent_qb2", {}) or {}
        qb_code = self.pos_to_code.get("QB")
        if qb_code is None:
            return penalty
        qb_count = int(roster_counts[qb_code])
        qb_mask = self.pos == "QB"
        r = int(round_no)

        if qb_count == 1:
            old = 18.0 if r < 9 else 0.0
            target = (
                float(qcfg.get("penalty_through_round_11", 34.0))
                if r <= 11
                else float(qcfg.get("penalty_round_12_plus", 24.0))
            )
            penalty[qb_mask] += target - old
        elif qb_count >= 2:
            old = 100.0
            target = float(qcfg.get("qb3_plus_penalty", 120.0))
            penalty[qb_mask] += target - old

        return penalty

    DraftEngine.recommend = recommend_position_aware
    FastDraftSimulator._patbot_score_vector = score_vector_position_aware
    FastDraftSimulator._base_roster_need_penalty = roster_need_penalty_league_calibrated
    _INSTALLED = True
