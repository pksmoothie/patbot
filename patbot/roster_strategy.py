from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


_INSTALLED = False
_NEG_INF = -1_000_000_000.0


@lru_cache(maxsize=1)
def load_roster_strategy() -> dict:
    path = Path(__file__).resolve().parents[1] / "config" / "roster_strategy.yaml"
    if not path.exists():
        return {
            "value_aware": True,
            "allow_bench_before_starters_complete": True,
            "max_qb_drafted": 1,
            "allow_te2_before_offense_complete": True,
            "post_offense_complete_positions": ["RB", "WR"],
        }
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return dict(payload) if isinstance(payload, dict) else {}


def offensive_starters_complete(
    counts: dict[str, int] | Counter,
    roster_cfg: dict,
) -> bool:
    """Return True once QB/RB/WR/TE plus FLEX are all covered.

    FLEX is covered by any excess RB/WR/TE beyond the base starting minima. K
    and D/ST are intentionally ignored because PatBot drafts those separately.
    """
    c = Counter({str(k).upper(): int(v) for k, v in dict(counts).items()})
    for pos in ("QB", "RB", "WR", "TE"):
        if c[pos] < int(roster_cfg.get(pos, 0)):
            return False

    flex_need = int(roster_cfg.get("FLEX", 0))
    if flex_need <= 0:
        return True

    eligible = [str(p).upper() for p in roster_cfg.get("flex_eligible", ["RB", "WR", "TE"])]
    excess = 0
    for pos in eligible:
        base = int(roster_cfg.get(pos, 0))
        excess += max(c[pos] - base, 0)
    return excess >= flex_need


def _array_counts(
    roster_counts: np.ndarray,
    pos_to_code: dict[str, int],
) -> Counter:
    out = Counter()
    for pos, code in pos_to_code.items():
        if 0 <= int(code) < len(roster_counts):
            out[str(pos).upper()] = int(roster_counts[int(code)])
    return out


def _apply_special_teams_array(
    score: np.ndarray,
    *,
    positions: np.ndarray,
    counts: Counter,
    round_no: int,
    config: dict,
) -> np.ndarray:
    """Mirror the hard Round 14 D/ST and Round 15 kicker draft policy."""
    out = np.asarray(score, dtype=float).copy()
    cfg = config.get("special_teams_strategy", {}).get("draft", {})
    defense_round = int(cfg.get("defense_round", 14))
    kicker_round = int(cfg.get("kicker_round", 15))
    target_def = int(cfg.get("rostered_defenses", 1))
    target_k = int(cfg.get("rostered_kickers", 1))

    def_mask = positions == "DEF"
    k_mask = positions == "K"

    if round_no < defense_round:
        out[def_mask] = _NEG_INF
    elif counts["DEF"] >= target_def:
        out[def_mask] = _NEG_INF
    elif round_no == defense_round and def_mask.any():
        # Reserve Round 14 for D/ST if we still need one.
        out[~def_mask] = _NEG_INF
        return out

    if round_no < kicker_round:
        out[k_mask] = _NEG_INF
    elif counts["K"] >= target_k:
        out[k_mask] = _NEG_INF
    elif round_no >= kicker_round and k_mask.any():
        # Reserve the final round for kicker if we still need one.
        out[~k_mask] = _NEG_INF

    return out


def apply_patbot_array_constraints(
    score: np.ndarray,
    *,
    positions: np.ndarray,
    roster_counts: np.ndarray,
    pos_to_code: dict[str, int],
    roster_cfg: dict,
    round_no: int,
    config: dict,
) -> np.ndarray:
    """Apply PatBot hard roster constraints without forcing starter order."""
    out = np.asarray(score, dtype=float).copy()
    strategy = load_roster_strategy()
    counts = _array_counts(roster_counts, pos_to_code)

    max_qb = int(strategy.get("max_qb_drafted", 1))
    if counts["QB"] >= max_qb:
        out[positions == "QB"] = _NEG_INF

    if offensive_starters_complete(counts, roster_cfg):
        allowed = {
            str(p).upper()
            for p in strategy.get("post_offense_complete_positions", ["RB", "WR"])
        }
        offensive = np.isin(positions, ["QB", "RB", "WR", "TE"])
        keep = np.isin(positions, list(allowed))
        out[offensive & ~keep] = _NEG_INF

    out = _apply_special_teams_array(
        out,
        positions=positions,
        counts=counts,
        round_no=int(round_no),
        config=config,
    )
    return out


def _dataframe_allowed_mask(
    board: pd.DataFrame,
    *,
    roster_positions: list[str],
    roster_cfg: dict,
) -> pd.Series:
    strategy = load_roster_strategy()
    counts = Counter(str(p).upper() for p in roster_positions)
    pos = board["pos"].astype(str).str.upper()
    allowed = pd.Series(True, index=board.index)

    if counts["QB"] >= int(strategy.get("max_qb_drafted", 1)):
        allowed &= ~pos.eq("QB")

    if offensive_starters_complete(counts, roster_cfg):
        post = {
            str(p).upper()
            for p in strategy.get("post_offense_complete_positions", ["RB", "WR"])
        }
        offensive = pos.isin(["QB", "RB", "WR", "TE"])
        allowed &= ~(offensive & ~pos.isin(post))

    return allowed


def install_roster_strategy_patch() -> None:
    """Install value-aware constraints into live recommendations and simulation."""
    global _INSTALLED
    if _INSTALLED:
        return

    from .draft import DraftEngine
    from .sim import FastDraftSimulator

    original_recommend = DraftEngine.recommend
    original_score_vector = FastDraftSimulator._patbot_score_vector

    def recommend_value_aware(
        self,
        current_pick: int,
        drafted_ids,
        roster_positions,
        top_n: int = 12,
    ):
        # Ask the original engine for the whole available board, then remove only
        # the hard-disallowed roster constructions. Missing starters are never
        # force-filled here; base value/urgency/roster-fit logic still decides.
        full_n = max(len(self.players), int(top_n))
        board = original_recommend(
            self,
            current_pick=current_pick,
            drafted_ids=drafted_ids,
            roster_positions=roster_positions,
            top_n=full_n,
        )
        if board.empty:
            return board
        allowed = _dataframe_allowed_mask(
            board,
            roster_positions=list(roster_positions),
            roster_cfg=self.roster_cfg,
        )
        board = board.loc[allowed].copy()
        return board.sort_values(
            ["score", "proj_points", "adp"],
            ascending=[False, False, True],
        ).head(int(top_n)).reset_index(drop=True)

    def score_vector_value_aware(
        self,
        available: np.ndarray,
        roster_counts: np.ndarray,
        pick: int,
    ) -> np.ndarray:
        score = original_score_vector(self, available, roster_counts, pick)
        round_no = (int(pick) - 1) // self.teams + 1
        score = apply_patbot_array_constraints(
            score,
            positions=self.pos,
            roster_counts=roster_counts,
            pos_to_code=self.pos_to_code,
            roster_cfg=self.engine.roster_cfg,
            round_no=round_no,
            config=self.cfg,
        )
        return np.where(available, score, _NEG_INF)

    DraftEngine.recommend = recommend_value_aware
    FastDraftSimulator._patbot_score_vector = score_vector_value_aware

    # The late-round audit reconstructs PatBot's score vector directly so that
    # it can compare championship vs Foundation phases. Wrap that diagnostic too
    # so it evaluates the same hard roster policy as the live simulator.
    try:
        from . import late_round as late_round_module

        original_late_score = late_round_module._score_vector

        def late_score_value_aware(
            sim,
            available: np.ndarray,
            roster_counts: np.ndarray,
            pick: int,
            *,
            upside_weight: float,
            risk_penalty_multiplier: float,
        ) -> np.ndarray:
            score = original_late_score(
                sim,
                available,
                roster_counts,
                pick,
                upside_weight=upside_weight,
                risk_penalty_multiplier=risk_penalty_multiplier,
            )
            round_no = (int(pick) - 1) // sim.teams + 1
            score = apply_patbot_array_constraints(
                score,
                positions=sim.pos,
                roster_counts=roster_counts,
                pos_to_code=sim.pos_to_code,
                roster_cfg=sim.engine.roster_cfg,
                round_no=round_no,
                config=sim.cfg,
            )
            return np.where(available, score, _NEG_INF)

        late_round_module._score_vector = late_score_value_aware
    except Exception:
        # Core draft/sim behavior should not fail merely because an optional
        # diagnostic module is unavailable.
        pass

    _INSTALLED = True
