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
            "max_te_drafted": 2,
            "allow_te2_before_offense_complete": True,
            "post_offense_complete_positions": ["RB", "WR"],
            "te2_quality_strategy": {
                "enabled": True,
                "elite_te1_top_n": 3,
                "solid_te1_top_n": 8,
                "elite_required_score_edge_over_best_rbwr": 7.5,
                "elite_required_projected_points_edge_over_current_flex": 10.0,
                "solid_required_score_edge_over_best_rbwr": 3.0,
                "weak_te1_unrestricted": True,
            },
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
        out[~def_mask] = _NEG_INF
        return out

    if round_no < kicker_round:
        out[k_mask] = _NEG_INF
    elif counts["K"] >= target_k:
        out[k_mask] = _NEG_INF
    elif round_no >= kicker_round and k_mask.any():
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
    """Apply PatBot hard roster-count constraints without forcing starter order."""
    out = np.asarray(score, dtype=float).copy()
    strategy = load_roster_strategy()
    counts = _array_counts(roster_counts, pos_to_code)

    max_qb = int(strategy.get("max_qb_drafted", 1))
    if counts["QB"] >= max_qb:
        out[positions == "QB"] = _NEG_INF

    max_te = int(strategy.get("max_te_drafted", 2))
    if counts["TE"] >= max_te:
        out[positions == "TE"] = _NEG_INF

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


def te1_quality_bucket(
    *,
    positions: np.ndarray,
    vorp: np.ndarray,
    roster_indices: list[int] | set[int] | tuple[int, ...],
    strategy: dict | None = None,
) -> str | None:
    """Classify the best rostered TE from model-derived TE VORP rank.

    This intentionally avoids hard-coding Bowers/McBride or any future player.
    If multiple TEs are already rostered, the best one defines TE1 quality.
    """
    cfg = strategy or load_roster_strategy()
    te_cfg = cfg.get("te2_quality_strategy", {}) or {}
    owned = [
        int(i)
        for i in roster_indices
        if 0 <= int(i) < len(positions) and str(positions[int(i)]).upper() == "TE"
    ]
    if not owned:
        return None

    te_idxs = np.where(np.asarray(positions).astype(str) == "TE")[0]
    if len(te_idxs) == 0:
        return None

    ordered = te_idxs[np.argsort(np.asarray(vorp, dtype=float)[te_idxs])[::-1]]
    rank_lookup = {int(idx): rank + 1 for rank, idx in enumerate(ordered)}
    best_owned = max(owned, key=lambda i: float(vorp[i]))
    rank = rank_lookup.get(int(best_owned), len(te_idxs))

    elite_top_n = max(1, int(te_cfg.get("elite_te1_top_n", 3)))
    solid_top_n = max(elite_top_n, int(te_cfg.get("solid_te1_top_n", 8)))
    if rank <= elite_top_n:
        return "elite"
    if rank <= solid_top_n:
        return "solid"
    return "weak"


def current_flex_projection_benchmark(
    *,
    positions: np.ndarray,
    projections: np.ndarray,
    roster_indices: list[int] | set[int] | tuple[int, ...],
    roster_cfg: dict,
) -> float | None:
    """Best currently rostered FLEX option after reserving base starters."""
    positions_arr = np.asarray(positions).astype(str)
    proj = np.asarray(projections, dtype=float)
    owned = {int(i) for i in roster_indices if 0 <= int(i) < len(positions_arr)}
    eligible = [str(p).upper() for p in roster_cfg.get("flex_eligible", ["RB", "WR", "TE"])]

    excess: list[int] = []
    for pos in eligible:
        idxs = [i for i in owned if positions_arr[i] == pos]
        idxs.sort(key=lambda i: float(proj[i]), reverse=True)
        base_need = int(roster_cfg.get(pos, 0))
        excess.extend(idxs[base_need:])

    if not excess:
        return None
    return max(float(proj[i]) for i in excess)


def apply_te2_quality_gate_array(
    score: np.ndarray,
    *,
    positions: np.ndarray,
    vorp: np.ndarray,
    roster_indices: list[int] | set[int] | tuple[int, ...],
    strategy: dict | None = None,
    projections: np.ndarray | None = None,
    roster_cfg: dict | None = None,
) -> np.ndarray:
    """Gate TE2 based on TE1 quality and FLEX opportunity cost.

    Behind an elite TE1, TE2 must clear two hurdles when applicable: a wide
    PatBot-score edge over every currently available RB/WR and a material raw
    projection edge over the best FLEX option already on our roster.
    """
    out = np.asarray(score, dtype=float).copy()
    cfg = strategy or load_roster_strategy()
    te_cfg = cfg.get("te2_quality_strategy", {}) or {}
    if not bool(te_cfg.get("enabled", True)):
        return out

    owned = {int(i) for i in roster_indices if 0 <= int(i) < len(positions)}
    owned_te = [i for i in owned if str(positions[i]).upper() == "TE"]
    if not owned_te:
        return out
    if len(owned_te) >= int(cfg.get("max_te_drafted", 2)):
        out[np.asarray(positions).astype(str) == "TE"] = _NEG_INF
        return out

    quality = te1_quality_bucket(
        positions=positions,
        vorp=vorp,
        roster_indices=owned,
        strategy=cfg,
    )
    if quality == "weak" and bool(te_cfg.get("weak_te1_unrestricted", True)):
        return out

    positions_arr = np.asarray(positions).astype(str)
    rbwr_mask = np.isin(positions_arr, ["RB", "WR"]) & (out > _NEG_INF / 2)
    te_mask = (positions_arr == "TE") & (out > _NEG_INF / 2)
    if not te_mask.any() or not rbwr_mask.any():
        return out

    best_rbwr = float(np.max(out[rbwr_mask]))
    if quality == "elite":
        required = float(te_cfg.get("elite_required_score_edge_over_best_rbwr", 7.5))
    else:
        required = float(te_cfg.get("solid_required_score_edge_over_best_rbwr", 3.0))
    out[te_mask & (out < best_rbwr + required)] = _NEG_INF

    if quality == "elite" and projections is not None and roster_cfg is not None:
        flex_benchmark = current_flex_projection_benchmark(
            positions=positions_arr,
            projections=np.asarray(projections, dtype=float),
            roster_indices=owned,
            roster_cfg=roster_cfg,
        )
        if flex_benchmark is not None:
            projection_edge = float(
                te_cfg.get("elite_required_projected_points_edge_over_current_flex", 10.0)
            )
            proj = np.asarray(projections, dtype=float)
            out[te_mask & (proj < flex_benchmark + projection_edge)] = _NEG_INF

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

    if counts["TE"] >= int(strategy.get("max_te_drafted", 2)):
        allowed &= ~pos.eq("TE")

    if offensive_starters_complete(counts, roster_cfg):
        post = {
            str(p).upper()
            for p in strategy.get("post_offense_complete_positions", ["RB", "WR"])
        }
        offensive = pos.isin(["QB", "RB", "WR", "TE"])
        allowed &= ~(offensive & ~pos.isin(post))

    return allowed


def _reconcile_sim_ownership(sim, available: np.ndarray) -> None:
    """Keep PatBot/opponent identity sets aligned with the simulated board."""
    unavailable = set(np.where(~np.asarray(available, dtype=bool))[0].tolist())
    patbot = set(getattr(sim, "_patbot_owned_idxs", set())) & unavailable
    opponents = set(getattr(sim, "_opponent_owned_idxs", set())) & unavailable
    opponents -= patbot

    known = patbot | opponents
    # Any newly unavailable player not attributed to an opponent is a PatBot
    # selection (including a forced current candidate in candidate simulations).
    patbot |= unavailable - known
    sim._patbot_owned_idxs = patbot
    sim._opponent_owned_idxs = opponents


def install_roster_strategy_patch() -> None:
    """Install value-aware constraints into live recommendations and simulation."""
    global _INSTALLED
    if _INSTALLED:
        return

    from .draft import DraftEngine
    from .sim import FastDraftSimulator

    original_recommend = DraftEngine.recommend
    original_score_vector = FastDraftSimulator._patbot_score_vector
    original_seed_opponent_counts = FastDraftSimulator._seed_opponent_counts
    original_take_opponent_pick = FastDraftSimulator._take_opponent_pick

    def recommend_value_aware(
        self,
        current_pick: int,
        drafted_ids,
        roster_positions,
        top_n: int = 12,
    ):
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
        if board.empty:
            return board

        roster_ids = [str(x) for x in getattr(self, "_patbot_roster_ids", [])]
        if roster_ids and "player_id" in self.players.columns:
            id_to_idx = {
                str(pid): int(i)
                for i, pid in enumerate(self.players["player_id"].astype(str).tolist())
            }
            roster_indices = [id_to_idx[x] for x in roster_ids if x in id_to_idx]
            positions = self.players["pos"].astype(str).to_numpy()
            levels = self.replacement_levels()
            proj = pd.to_numeric(self.players["proj_points"], errors="coerce").fillna(0.0).to_numpy(float)
            vorp = np.array([proj[i] - float(levels.get(positions[i], 0.0)) for i in range(len(proj))])

            score_map = dict(zip(board["player_id"].astype(str), pd.to_numeric(board["score"], errors="coerce").fillna(_NEG_INF)))
            score_vec = np.full(len(self.players), _NEG_INF, dtype=float)
            for i, pid in enumerate(self.players["player_id"].astype(str)):
                if pid in score_map:
                    score_vec[i] = float(score_map[pid])
            gated = apply_te2_quality_gate_array(
                score_vec,
                positions=positions,
                vorp=vorp,
                roster_indices=roster_indices,
                projections=proj,
                roster_cfg=self.roster_cfg,
            )
            legal_ids = {
                str(self.players.iloc[i]["player_id"])
                for i in np.where(gated > _NEG_INF / 2)[0]
            }
            board = board[board["player_id"].astype(str).isin(legal_ids)].copy()

        return board.sort_values(
            ["score", "proj_points", "adp"],
            ascending=[False, False, True],
        ).head(int(top_n)).reset_index(drop=True)

    def seed_opponent_counts_with_identity(self, draft_history):
        counts = original_seed_opponent_counts(self, draft_history)
        patbot = set()
        opponents = set()
        for pick in draft_history or []:
            idx = self.id_to_idx.get(str(pick.get("player_id", "")))
            if idx is None:
                continue
            try:
                owner_slot = int(pick.get("owner_slot", -1))
            except (TypeError, ValueError):
                continue
            if owner_slot == self.slot:
                patbot.add(int(idx))
            elif 1 <= owner_slot <= self.teams:
                opponents.add(int(idx))
        self._patbot_owned_idxs = patbot
        self._opponent_owned_idxs = opponents
        return counts

    def take_opponent_pick_with_identity(
        self,
        pick: int,
        available: np.ndarray,
        opp_counts: np.ndarray,
        archetypes: dict[int, str],
        market_latent: np.ndarray,
        custom_noise_base: np.ndarray,
    ):
        idx, archetype = original_take_opponent_pick(
            self,
            pick,
            available,
            opp_counts,
            archetypes,
            market_latent,
            custom_noise_base,
        )
        opponents = set(getattr(self, "_opponent_owned_idxs", set()))
        opponents.add(int(idx))
        self._opponent_owned_idxs = opponents
        return idx, archetype

    def score_vector_value_aware(
        self,
        available: np.ndarray,
        roster_counts: np.ndarray,
        pick: int,
    ) -> np.ndarray:
        _reconcile_sim_ownership(self, available)
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
        score = apply_te2_quality_gate_array(
            score,
            positions=self.pos,
            vorp=self.vorp,
            roster_indices=set(getattr(self, "_patbot_owned_idxs", set())),
            projections=self.proj,
            roster_cfg=self.engine.roster_cfg,
        )
        return np.where(available, score, _NEG_INF)

    def lookahead_pick_value_aware(
        self,
        available: np.ndarray,
        my_counts: np.ndarray,
        pick: int,
        opp_counts: np.ndarray,
        archetypes: dict[int, str],
        market_latent: np.ndarray,
        custom_noise_base: np.ndarray,
    ) -> int:
        _reconcile_sim_ownership(self, available)
        base_patbot = set(getattr(self, "_patbot_owned_idxs", set()))
        base_opponents = set(getattr(self, "_opponent_owned_idxs", set()))

        current_scores = self._patbot_score_vector(available, my_counts, pick)
        greedy = int(np.argmax(current_scores))
        round_no = (int(pick) - 1) // self.teams + 1
        if not self.lookahead_enabled or round_no not in self.lookahead_rounds:
            self._patbot_owned_idxs = base_patbot | {greedy}
            self._opponent_owned_idxs = base_opponents
            return greedy

        next_pick = self._next_my_pick(pick)
        gap = next_pick - int(pick)
        if gap <= 0 or gap > self.lookahead_max_gap:
            self._patbot_owned_idxs = base_patbot | {greedy}
            self._opponent_owned_idxs = base_opponents
            return greedy

        candidates = np.where(available & (current_scores > _NEG_INF / 2))[0]
        if len(candidates) <= 1:
            self._patbot_owned_idxs = base_patbot | {greedy}
            self._opponent_owned_idxs = base_opponents
            return greedy
        candidates = candidates[np.argsort(current_scores[candidates])[::-1]][: self.lookahead_branch_width]

        best_idx = greedy
        best_value = -float("inf")
        for candidate in candidates:
            self._patbot_owned_idxs = set(base_patbot) | {int(candidate)}
            self._opponent_owned_idxs = set(base_opponents)

            branch_available = available.copy()
            branch_opp_counts = opp_counts.copy()
            branch_my_counts = my_counts.copy()
            branch_available[int(candidate)] = False
            code = self.pos_code[int(candidate)]
            if code >= 0:
                branch_my_counts[code] += 1

            for future_pick in range(int(pick) + 1, next_pick):
                if future_pick in self.my_picks or not branch_available.any():
                    break
                self._take_opponent_pick(
                    future_pick,
                    branch_available,
                    branch_opp_counts,
                    archetypes,
                    market_latent,
                    custom_noise_base,
                )

            if branch_available.any():
                future_scores = self._patbot_score_vector(
                    branch_available,
                    branch_my_counts,
                    next_pick,
                )
                future_idx = int(np.argmax(future_scores))
                future_value = float(future_scores[future_idx])
            else:
                future_idx = None
                future_value = 0.0

            pair_vorp = max(float(self.vorp[int(candidate)]), 0.0)
            if future_idx is not None:
                pair_vorp += max(float(self.vorp[future_idx]), 0.0)

            path_value = (
                float(current_scores[int(candidate)])
                + self.lookahead_future_weight * future_value
                + self.lookahead_vorp_weight * pair_vorp
            )
            if path_value > best_value:
                best_value = path_value
                best_idx = int(candidate)

        self._patbot_owned_idxs = base_patbot | {best_idx}
        self._opponent_owned_idxs = base_opponents
        return best_idx

    DraftEngine.recommend = recommend_value_aware
    FastDraftSimulator._seed_opponent_counts = seed_opponent_counts_with_identity
    FastDraftSimulator._take_opponent_pick = take_opponent_pick_with_identity
    FastDraftSimulator._patbot_score_vector = score_vector_value_aware
    FastDraftSimulator._lookahead_pick = lookahead_pick_value_aware

    # The late-round audit reconstructs PatBot's score vector directly so that
    # it can compare championship vs Foundation phases. Keep the hard count-based
    # rules synchronized there. TE1-quality gating is exercised by the production
    # simulator and the construction audit, which both use _patbot_score_vector.
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
        pass

    _INSTALLED = True
