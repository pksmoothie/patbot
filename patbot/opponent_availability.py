from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml


_INSTALLED = False


_DEFAULTS = {
    "enabled": True,
    "start_round": 7,
    "through_round": 11,
    "qb2_with_core_starter_gaps": {
        "per_missing_starter_penalty": 18.0,
        "max_penalty": 36.0,
        "expected_demand_multiplier_per_missing_starter": 0.50,
    },
    "te2_with_core_starter_gaps": {
        "per_missing_starter_penalty": 10.0,
        "max_penalty": 20.0,
    },
    "extreme_one_sided_skill_hoarding": {
        "enabled": True,
        "extra_players_before_guardrail": 2,
        "require_opposite_position_empty": True,
        "base_penalty": 30.0,
        "per_additional_player": 10.0,
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
def opponent_availability_settings() -> dict:
    path = Path(__file__).resolve().parents[1] / "config" / "opponent_availability.yaml"
    if not path.exists():
        return _deep_merge({}, _DEFAULTS)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        payload = {}
    return _deep_merge(_DEFAULTS, payload)


def _core_skill_gaps(
    roster_counts: np.ndarray,
    *,
    pos_to_code: dict[str, int],
    roster_cfg: dict,
) -> int:
    missing = 0
    for pos in ("RB", "WR"):
        code = pos_to_code.get(pos)
        if code is None:
            continue
        need = int(roster_cfg.get(pos, 0))
        missing += max(need - int(roster_counts[code]), 0)
    return int(missing)


def opponent_availability_penalty(sim, roster_counts: np.ndarray, round_no: int) -> np.ndarray:
    """Soft penalties for opponent picks that can distort PatBot availability.

    This is intentionally not a full opponent-roster optimizer. It only acts on
    backup single-position picks made while core RB/WR starters are open, plus
    extreme RB-vs-WR one-sided hoarding. Lower opponent rank-score is better, so
    positive values here make the affected position less likely, never illegal.

    These penalties are applied after ordinary roster_need_strength so a casual
    manager's weak need-awareness does not dilute the anti-distortion layer.
    """
    settings = opponent_availability_settings()
    penalty = np.zeros(sim.n, dtype=float)
    r = int(round_no)
    if (
        not bool(settings.get("enabled", True))
        or r < int(settings.get("start_round", 7))
        or r > int(settings.get("through_round", 11))
    ):
        return penalty

    counts = np.asarray(roster_counts)
    gaps = _core_skill_gaps(
        counts,
        pos_to_code=sim.pos_to_code,
        roster_cfg=sim.engine.roster_cfg,
    )

    if gaps > 0:
        qb_code = sim.pos_to_code.get("QB")
        if qb_code is not None and int(counts[qb_code]) == 1:
            cfg = settings.get("qb2_with_core_starter_gaps", {}) or {}
            amount = min(
                float(cfg.get("max_penalty", 36.0)),
                gaps * float(cfg.get("per_missing_starter_penalty", 18.0)),
            )
            penalty[sim.pos == "QB"] += amount

        te_code = sim.pos_to_code.get("TE")
        if te_code is not None and int(counts[te_code]) == 1:
            cfg = settings.get("te2_with_core_starter_gaps", {}) or {}
            amount = min(
                float(cfg.get("max_penalty", 20.0)),
                gaps * float(cfg.get("per_missing_starter_penalty", 10.0)),
            )
            penalty[sim.pos == "TE"] += amount

    hoard = settings.get("extreme_one_sided_skill_hoarding", {}) or {}
    if bool(hoard.get("enabled", True)):
        extra_before = max(0, int(hoard.get("extra_players_before_guardrail", 2)))
        require_empty = bool(hoard.get("require_opposite_position_empty", True))
        base = float(hoard.get("base_penalty", 30.0))
        per_extra = float(hoard.get("per_additional_player", 10.0))

        for heavy, opposite in (("RB", "WR"), ("WR", "RB")):
            heavy_code = sim.pos_to_code.get(heavy)
            opposite_code = sim.pos_to_code.get(opposite)
            if heavy_code is None or opposite_code is None:
                continue
            heavy_count = int(counts[heavy_code])
            opposite_count = int(counts[opposite_code])
            starter_need = int(sim.engine.roster_cfg.get(heavy, 0))
            trigger = starter_need + extra_before
            if heavy_count < trigger:
                continue
            if require_empty and opposite_count != 0:
                continue

            amount = base + per_extra * max(0, heavy_count - trigger)
            penalty[sim.pos == heavy] += amount

    return penalty


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


def _history_counts(engine, draft_history: list[dict] | None) -> dict[int, Counter]:
    out: dict[int, Counter] = {}
    id_to_pos = {}
    if hasattr(engine, "players") and "player_id" in engine.players.columns and "pos" in engine.players.columns:
        id_to_pos = dict(
            zip(engine.players["player_id"].astype(str), engine.players["pos"].astype(str).str.upper())
        )

    for item in draft_history or []:
        try:
            owner_slot = int(item.get("owner_slot"))
        except (TypeError, ValueError):
            continue
        pos = str(item.get("pos", "")).upper()
        if not pos:
            pos = id_to_pos.get(str(item.get("player_id", "")), "")
        if not pos:
            continue
        out.setdefault(owner_slot, Counter())[pos] += 1
    return out


def _owner_name(engine, owner_slot: int) -> str:
    managers = getattr(engine, "config", {}).get("opponent_managers", {}) or {}
    raw = managers.get(int(owner_slot)) or managers.get(str(owner_slot)) or {}
    return str(raw.get("name", "")) if isinstance(raw, dict) else ""


def _contextual_qb2_correction(
    engine,
    *,
    current_pick: int,
    draft_history: list[dict] | None,
) -> float:
    """Return the amount to subtract from the legacy aggregate QB demand.

    The legacy model gives every rostered-QB owner a small QB2 chance and James a
    larger history-based chance. This correction preserves those tendencies but
    discounts them when the same manager still has mandatory RB/WR starter gaps.
    """
    settings = opponent_availability_settings()
    if not bool(settings.get("enabled", True)):
        return 0.0

    teams = int(engine.league["teams"])
    slot = int(engine.league["draft_slot"])
    next_pick = _next_patbot_pick(int(current_pick), teams, slot)
    upcoming: dict[int, list[int]] = {}
    for pick in range(int(current_pick) + 1, int(next_pick)):
        owner_slot = _team_slot_for_pick(pick, teams)
        if owner_slot == slot:
            continue
        upcoming.setdefault(owner_slot, []).append(pick)
    if not upcoming:
        return 0.0

    round_values = [(pick - 1) // teams + 1 for picks in upcoming.values() for pick in picks]
    if round_values:
        if max(round_values) < int(settings.get("start_round", 7)):
            return 0.0
        if min(round_values) > int(settings.get("through_round", 11)):
            return 0.0

    from .decision_strategy import decision_strategy_settings
    from .opponent_history import load_owner_history

    demand_cfg = decision_strategy_settings(engine.config).get("opponent_demand", {}) or {}
    generic_rate = float(demand_cfg.get("generic_qb2_expected_rate_per_team", 0.04))
    promoted_rate = float(demand_cfg.get("promoted_qb2_owner_expected_rate", 0.28))
    multiplier = float(
        (settings.get("qb2_with_core_starter_gaps", {}) or {}).get(
            "expected_demand_multiplier_per_missing_starter", 0.50
        )
    )
    multiplier = max(0.0, min(1.0, multiplier))

    hist = _history_counts(engine, draft_history)
    correction = 0.0
    for owner_slot, picks in upcoming.items():
        counts = hist.get(int(owner_slot), Counter())
        if int(counts.get("QB", 0)) != 1:
            continue

        owner_name = _owner_name(engine, int(owner_slot))
        tendency = load_owner_history().get(owner_name, {}) or {}
        second_qb = tendency.get("second_qb_bonus", {}) or {}
        originally_promoted = bool(second_qb)
        original_rate = promoted_rate if originally_promoted else generic_rate

        desired_rate = original_rate
        if originally_promoted:
            from_round = int(second_qb.get("from_round", 1))
            upcoming_rounds = [(int(pick) - 1) // teams + 1 for pick in picks]
            if not any(r >= from_round for r in upcoming_rounds):
                desired_rate = generic_rate

        gaps = 0
        for pos in ("RB", "WR"):
            need = int(engine.roster_cfg.get(pos, 0))
            gaps += max(need - int(counts.get(pos, 0)), 0)
        if gaps > 0:
            desired_rate *= multiplier ** gaps

        correction += max(0.0, original_rate - desired_rate)

    return float(correction)


def install_opponent_availability_patch() -> None:
    """Install availability-focused expected-demand calibration after strategy."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import decision_strategy as decision_strategy_module

    original_expected_demand = decision_strategy_module.expected_position_demand

    def expected_position_demand_with_context(
        engine,
        *,
        pos: str,
        current_pick: int,
        draft_history: list[dict] | None,
    ) -> float:
        base = float(
            original_expected_demand(
                engine,
                pos=pos,
                current_pick=int(current_pick),
                draft_history=draft_history,
            )
        )
        if str(pos).upper() != "QB":
            return base
        correction = _contextual_qb2_correction(
            engine,
            current_pick=int(current_pick),
            draft_history=draft_history,
        )
        return max(0.0, base - correction)

    decision_strategy_module.expected_position_demand = expected_position_demand_with_context
    _INSTALLED = True
