from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml


HISTORY_KEY = "_historical_owner_tendency"
_INSTALLED = False


@lru_cache(maxsize=1)
def load_owner_history() -> dict[str, dict]:
    """Load only the explicitly promoted, high-confidence owner tendencies."""
    path = Path(__file__).resolve().parents[1] / "config" / "owner_history.yaml"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    owners = payload.get("owners", {})
    if not isinstance(owners, dict):
        return {}
    return {
        str(name): dict(cfg)
        for name, cfg in owners.items()
        if isinstance(cfg, dict) and str(cfg.get("confidence", "")).lower() == "high"
    }


def apply_history_adjustments(
    score: np.ndarray,
    *,
    positions: np.ndarray,
    pos_to_code: dict[str, int],
    roster_counts: np.ndarray,
    round_no: int,
    tendency: dict | None,
) -> np.ndarray:
    """Apply modest historical rank nudges without turning tendencies into rules.

    Opponent scores are rank-like and lower is better. Historical behavior only
    moves a player a handful of rank points; market/custom signals can still win.
    """
    out = np.asarray(score, dtype=float).copy()
    if not tendency or str(tendency.get("confidence", "")).lower() != "high":
        return out

    r = int(round_no)

    if r == 1:
        for pos, bonus in (tendency.get("round1_position_bonus", {}) or {}).items():
            out[positions == str(pos).upper()] -= float(bonus)

    early = tendency.get("early_position_bonus", {}) or {}
    if early and r <= int(early.get("through_round", 0)):
        bonus = float(early.get("bonus", 0.0))
        for pos in early.get("positions", []) or []:
            out[positions == str(pos).upper()] -= bonus

    qb_code = pos_to_code.get("QB")
    qb_count = int(roster_counts[qb_code]) if qb_code is not None else 0
    qb_mask = positions == "QB"

    first_bonus = tendency.get("first_qb_bonus", {}) or {}
    if first_bonus and qb_count == 0:
        start = int(first_bonus.get("from_round", 1))
        through = int(first_bonus.get("through_round", 99))
        if start <= r <= through:
            out[qb_mask] -= float(first_bonus.get("bonus", 0.0))

    first_penalty = tendency.get("first_qb_penalty_before_round", {}) or {}
    if first_penalty and qb_count == 0:
        target_round = int(first_penalty.get("round", 1))
        if r < target_round:
            out[qb_mask] += float(first_penalty.get("penalty", 0.0))

    second_bonus = tendency.get("second_qb_bonus", {}) or {}
    if second_bonus and qb_count == 1:
        if r >= int(second_bonus.get("from_round", 1)):
            out[qb_mask] -= float(second_bonus.get("bonus", 0.0))

    return out


def install_owner_history_patch() -> None:
    """Extend FastDraftSimulator without rewriting its core simulation file."""
    global _INSTALLED
    if _INSTALLED:
        return

    from .sim import FastDraftSimulator

    original_manager_profile = FastDraftSimulator._manager_profile

    def manager_profile_with_history(self, team_slot: int, archetype: str) -> dict:
        profile = original_manager_profile(self, team_slot, archetype)
        raw = self.manager_cfg.get(team_slot) or self.manager_cfg.get(str(team_slot)) or {}
        owner_name = str(raw.get("name", "")) if isinstance(raw, dict) else ""
        tendency = load_owner_history().get(owner_name)
        if tendency:
            profile[HISTORY_KEY] = tendency
        return profile

    def opponent_pick_with_history(
        self,
        available: np.ndarray,
        market_latent: np.ndarray,
        custom_latent: np.ndarray,
        roster_counts: np.ndarray,
        round_no: int,
        profile: dict,
    ) -> int:
        market_w = float(profile.get("market_weight", 0.8))
        custom_w = float(profile.get("custom_weight", 0.2))
        need_strength = float(profile.get("roster_need_strength", 1.0))

        score = market_w * market_latent + custom_w * custom_latent
        score += self._base_roster_need_penalty(roster_counts, round_no) * need_strength

        rookie_rank_bonus = float(profile.get("rookie_rank_bonus", 0.0))
        if rookie_rank_bonus and self.is_rookie.any():
            score[self.is_rookie] -= rookie_rank_bonus

        score = apply_history_adjustments(
            score,
            positions=self.pos,
            pos_to_code=self.pos_to_code,
            roster_counts=roster_counts,
            round_no=round_no,
            tendency=profile.get(HISTORY_KEY),
        )

        # v0.6.13: availability guardrails protect PatBot's simulated future
        # player pool, so they are intentionally independent of an opponent's
        # ordinary roster_need_strength. A casual manager can still draft oddly,
        # but the anti-distortion layer is not diluted just because that manager
        # is modeled as weakly need-aware.
        from .opponent_availability import opponent_availability_penalty

        score += opponent_availability_penalty(
            self,
            np.asarray(roster_counts),
            int(round_no),
        )

        score = np.where(available, score, 1_000_000_000.0)
        return int(np.argmin(score))

    FastDraftSimulator._manager_profile = manager_profile_with_history
    FastDraftSimulator.opponent_pick = opponent_pick_with_history
    _INSTALLED = True
