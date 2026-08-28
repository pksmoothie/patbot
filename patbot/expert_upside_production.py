from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import strategy_phase
from .upside_intel import score_upside_evidence


_INSTALLED = False


def expert_upside_settings(config: dict) -> dict:
    cfg = config.get("championship_strategy", {}).get("expert_upside_intel", {}) or {}
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "production_enabled": bool(cfg.get("production_enabled", True)),
        "minimum_round": max(1, int(cfg.get("minimum_round", 8))),
        "minimum_independent_sources": max(1, int(cfg.get("minimum_independent_sources", 2))),
        "minimum_objective_sources": max(0, int(cfg.get("minimum_objective_sources", 1))),
        "max_lws_bonus": max(0.0, float(cfg.get("max_lws_bonus", 8.0))),
    }


def production_bonus_for_round(
    players: pd.DataFrame,
    config: dict,
    round_no: int,
) -> pd.Series:
    """Return only the incremental expert LWS bonus allowed in this round.

    The signal is intentionally zero before Round 8. It never changes projection,
    VORP, expert rank, market ADP or injury risk; it can only break close late-round
    decisions through the existing phase-weighted league-winner score.
    """
    settings = expert_upside_settings(config)
    zero = pd.Series(0.0, index=players.index, dtype=float)
    if not settings["enabled"] or not settings["production_enabled"]:
        return zero
    if int(round_no) < int(settings["minimum_round"]):
        return zero
    if "expert_upside_lws_bonus" not in players.columns:
        return zero
    bonus = pd.to_numeric(players["expert_upside_lws_bonus"], errors="coerce").fillna(0.0)
    return bonus.clip(lower=0.0, upper=float(settings["max_lws_bonus"]))


def _enrich_engine_players(engine) -> None:
    settings = expert_upside_settings(engine.config)
    if not settings["enabled"] or not settings["production_enabled"]:
        return
    enriched, status = score_upside_evidence(engine.players, engine.config)
    engine.players = enriched
    engine.expert_upside_status = status


def install_expert_upside_production_patch() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from .draft import DraftEngine
    from .sim import FastDraftSimulator

    original_engine_init = DraftEngine.__init__
    original_recommend = DraftEngine.recommend
    original_explain = DraftEngine.explain_row
    original_sim_init = FastDraftSimulator.__init__
    original_score_vector = FastDraftSimulator._patbot_score_vector

    def engine_init(self, players, config):
        original_engine_init(self, players, config)
        _enrich_engine_players(self)

    def recommend(self, current_pick, drafted_ids, roster_positions, top_n=12):
        teams = int(self.league["teams"])
        round_no = ((int(current_pick) - 1) // teams) + 1
        settings = expert_upside_settings(self.config)
        active = (
            settings["enabled"]
            and settings["production_enabled"]
            and round_no >= settings["minimum_round"]
        )

        if not active:
            result = original_recommend(self, current_pick, drafted_ids, roster_positions, top_n=top_n)
            if not result.empty:
                result["expert_upside_score_bonus"] = 0.0
                result["effective_league_winner_score"] = pd.to_numeric(
                    result.get("league_winner_score", 0.0), errors="coerce"
                ).fillna(0.0)
            return result

        # Ask the underlying engine for the whole available board, then apply the
        # tiny late-round expert increment and rerank. This avoids missing a player
        # who was just outside the pre-bonus top_n cutoff.
        full = original_recommend(
            self,
            current_pick,
            drafted_ids,
            roster_positions,
            top_n=max(int(top_n), len(self.players)),
        )
        if full.empty:
            return full

        phase = strategy_phase(round_no, self.config)
        upside_weight = float(phase.get("upside_weight", 0.0))
        bonus_lws = production_bonus_for_round(full, self.config, round_no)
        score_increment = bonus_lws * upside_weight

        full["expert_upside_score_bonus"] = score_increment.round(3)
        full["score"] = pd.to_numeric(full["score"], errors="coerce").fillna(-1e9) + score_increment
        if "upside_bonus" in full.columns:
            full["upside_bonus"] = (
                pd.to_numeric(full["upside_bonus"], errors="coerce").fillna(0.0)
                + score_increment
            )
        base_lws = pd.to_numeric(full.get("league_winner_score", 0.0), errors="coerce").fillna(0.0)
        full["effective_league_winner_score"] = (base_lws + bonus_lws).clip(upper=100.0).round(2)
        full["score"] = full["score"].round(2)
        return full.sort_values(
            ["score", "proj_points", "adp"], ascending=[False, False, True]
        ).head(int(top_n)).reset_index(drop=True)

    def explain_row(row: pd.Series) -> str:
        if pd.notna(row.get("effective_league_winner_score")):
            adjusted = row.copy()
            adjusted["league_winner_score"] = row["effective_league_winner_score"]
            text = original_explain(adjusted)
            expert_increment = float(row.get("expert_upside_score_bonus") or 0.0)
            if expert_increment > 0:
                text += f" Corroborated expert-upside late-round score increment: +{expert_increment:.2f}."
            return text
        return original_explain(row)

    def sim_init(self, engine):
        original_sim_init(self, engine)
        settings = expert_upside_settings(self.cfg)
        self.expert_upside_min_round = int(settings["minimum_round"])
        self.expert_upside_production_enabled = bool(
            settings["enabled"] and settings["production_enabled"]
        )
        self.expert_upside_lws_bonus = pd.to_numeric(
            self.players.get("expert_upside_lws_bonus", pd.Series(0.0, index=self.players.index)),
            errors="coerce",
        ).fillna(0.0).clip(lower=0.0, upper=float(settings["max_lws_bonus"])).to_numpy(float)

    def score_vector(self, available: np.ndarray, roster_counts: np.ndarray, pick: int) -> np.ndarray:
        base = np.asarray(original_score_vector(self, available, roster_counts, pick), dtype=float)
        round_no = (int(pick) - 1) // self.teams + 1
        if not self.expert_upside_production_enabled or round_no < self.expert_upside_min_round:
            return base
        phase = strategy_phase(round_no, self.cfg)
        increment = self.expert_upside_lws_bonus * float(phase.get("upside_weight", 0.0))
        out = base.copy()
        mask = np.asarray(available, dtype=bool)
        out[mask] += increment[mask]
        return out

    DraftEngine.__init__ = engine_init
    DraftEngine.recommend = recommend
    DraftEngine.explain_row = staticmethod(explain_row)
    FastDraftSimulator.__init__ = sim_init
    FastDraftSimulator._patbot_score_vector = score_vector
    _INSTALLED = True
