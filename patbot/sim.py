from __future__ import annotations

from collections import Counter
import math

import numpy as np
import pandas as pd

from .draft import all_team_picks


def _team_slot_for_pick(pick: int, teams: int) -> int:
    round_no = (pick - 1) // teams + 1
    within = (pick - 1) % teams + 1
    return within if round_no % 2 else teams + 1 - within


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class FastDraftSimulator:
    POSITIONS = ("QB", "RB", "WR", "TE", "K", "DEF")

    def __init__(self, engine):
        self.engine = engine
        self.cfg = engine.config
        self.players = engine.players.reset_index(drop=True).copy()
        self.players["player_id"] = self.players["player_id"].astype(str)

        self.n = len(self.players)
        self.ids = self.players["player_id"].to_numpy()
        self.names = self.players["name"].astype(str).to_numpy()
        self.pos = self.players["pos"].astype(str).to_numpy()
        self.adp = pd.to_numeric(self.players["adp"], errors="coerce").fillna(999.0).to_numpy(float)
        self.proj = pd.to_numeric(self.players["proj_points"], errors="coerce").fillna(0.0).to_numpy(float)
        self.injury = pd.to_numeric(
            self.players.get("injury_risk", pd.Series([0.0] * self.n)),
            errors="coerce",
        ).fillna(0.0).to_numpy(float)

        if "expert_rank" in self.players:
            self.expert_rank = pd.to_numeric(self.players["expert_rank"], errors="coerce").to_numpy(float)
        else:
            self.expert_rank = np.full(self.n, np.nan)

        if "is_rookie" in self.players:
            rookie_series = self.players["is_rookie"].fillna(False)
            self.is_rookie = rookie_series.astype(bool).to_numpy()
        elif "years_exp" in self.players:
            years = pd.to_numeric(self.players["years_exp"], errors="coerce")
            self.is_rookie = years.eq(0).fillna(False).to_numpy()
        else:
            self.is_rookie = np.zeros(self.n, dtype=bool)

        self.id_to_idx = {pid: i for i, pid in enumerate(self.ids)}
        self.pos_to_code = {p: i for i, p in enumerate(self.POSITIONS)}
        self.pos_code = np.array([self.pos_to_code.get(p, -1) for p in self.pos], dtype=int)

        self.teams = int(engine.league["teams"])
        self.slot = int(engine.league["draft_slot"])
        self.my_picks = set(all_team_picks(self.teams, self.slot, rounds=20))

        levels = engine.replacement_levels()
        self.replacement_levels = levels
        self.replacement = np.array([float(levels.get(p, 0.0)) for p in self.pos])
        self.vorp = self.proj - self.replacement
        self.proj_pct = self._percentile(self.proj)
        self.vorp_pct = self._percentile(self.vorp)

        expert_component = np.full(self.n, 0.5)
        if np.isfinite(self.expert_rank).any():
            finite = np.isfinite(self.expert_rank)
            tmp = np.where(finite, self.expert_rank, np.nanmax(self.expert_rank[finite]) + 50)
            expert_rank_pct = self._percentile(tmp)
            expert_component = np.where(
                finite,
                1.0 - expert_rank_pct + 1.0 / max(self.n, 1),
                0.5,
            )

        custom_goodness = 0.58 * self.vorp_pct + 0.27 * self.proj_pct + 0.15 * expert_component
        custom_order = np.argsort(custom_goodness)[::-1]
        self.custom_rank = np.empty(self.n, dtype=float)
        self.custom_rank[custom_order] = np.arange(1, self.n + 1, dtype=float)

        if np.isfinite(self.expert_rank).any():
            finite = np.isfinite(self.expert_rank)
            temp = np.where(finite, self.expert_rank, np.nanmax(self.expert_rank[finite]) + 50)
            rank_pct = self._percentile(temp)
            self.expert_pct = 1.0 - rank_pct + 1.0 / max(self.n, 1)
            self.expert_pct = np.where(finite, self.expert_pct, 0.5)
            self.expert_weight = 0.1
        else:
            self.expert_pct = np.full(self.n, 0.5)
            self.expert_weight = 0.0

        w = engine.engine_cfg["weights"]
        base_total = sum(float(w[k]) for k in ["vorp", "projection", "urgency", "scarcity", "roster_fit"])
        scale = (1.0 - self.expert_weight) / base_total
        self.w_vorp = float(w["vorp"]) * scale
        self.w_proj = float(w["projection"]) * scale
        self.w_urgency = float(w["urgency"]) * scale
        self.w_scarcity = float(w["scarcity"]) * scale
        self.w_roster = float(w["roster_fit"]) * scale
        self.injury_penalty = float(engine.engine_cfg.get("injury_risk_penalty", 0))

        simcfg = self.cfg.get("simulation", {})
        self.sd_floor = float(simcfg.get("opponent_adp_sd_floor", 5.0))
        self.sd_pct = float(simcfg.get("opponent_adp_sd_pct", 0.14))
        self.comparison_seed = int(simcfg.get("comparison_seed", 20260818))

        lookcfg = simcfg.get("patbot_lookahead", {})
        self.lookahead_enabled = bool(lookcfg.get("enabled", True))
        self.lookahead_rounds = {int(x) for x in lookcfg.get("rounds", [2, 3])}
        self.lookahead_branch_width = max(1, int(lookcfg.get("branch_width", 5)))
        self.lookahead_future_weight = float(lookcfg.get("future_pick_weight", 0.90))
        self.lookahead_vorp_weight = float(lookcfg.get("pair_vorp_weight", 0.02))
        self.lookahead_max_gap = max(1, int(lookcfg.get("max_gap_picks", 24)))

        # v0.4 explicit availability tails. Missing risk fields fall back to a
        # deterministic projection, so old/synthetic test snapshots still work.
        riskcfg = self.cfg.get("risk_model", {})
        self.risk_enabled = bool(riskcfg.get("enabled", False)) and (
            "catastrophic_miss_probability" in self.players.columns
        )
        self.games_projected = pd.to_numeric(
            self.players.get("games_projected", pd.Series([17.0] * self.n)),
            errors="coerce",
        ).fillna(17.0).clip(lower=1.0, upper=17.0).to_numpy(float)
        self.catastrophic_prob = pd.to_numeric(
            self.players.get("catastrophic_miss_probability", pd.Series([0.0] * self.n)),
            errors="coerce",
        ).fillna(0.0).clip(lower=0.0, upper=1.0).to_numpy(float)
        self.minor_miss_lambda = pd.to_numeric(
            self.players.get("minor_miss_lambda", pd.Series([0.0] * self.n)),
            errors="coerce",
        ).fillna(0.0).clip(lower=0.0).to_numpy(float)
        self.off_field_prob = pd.to_numeric(
            self.players.get("off_field_miss_probability", pd.Series([0.0] * self.n)),
            errors="coerce",
        ).fillna(0.0).clip(lower=0.0, upper=1.0).to_numpy(float)
        self.off_field_max_games = pd.to_numeric(
            self.players.get("off_field_max_missed_games", pd.Series([0.0] * self.n)),
            errors="coerce",
        ).fillna(0.0).clip(lower=0.0).to_numpy(int)
        self.catastrophic_min_games = max(1, int(riskcfg.get("catastrophic_min_missed_games", 4)))
        self.catastrophic_max_games = max(
            self.catastrophic_min_games,
            int(riskcfg.get("catastrophic_max_missed_games", 9)),
        )
        capture_cfg = riskcfg.get("replacement_capture_by_position", {})
        self.replacement_capture = np.array([
            float(capture_cfg.get(p, 0.60 if p in {"RB", "WR"} else 0.65))
            for p in self.pos
        ])

        self.min_round_k = int(engine.engine_cfg.get("min_round_k", 13))
        self.min_round_def = int(engine.engine_cfg.get("min_round_def", 13))
        self.bench_caps = engine.engine_cfg.get("bench_position_caps", {})
        self.archetype_cfg = self.cfg.get("opponent_archetypes", {})
        self.manager_cfg = self.cfg.get("opponent_managers", {})
        self.roster_eval_cfg = self.cfg.get("roster_evaluation", {})

        counts = self.archetype_cfg.get("counts", {})
        if counts:
            total = sum(int(v) for v in counts.values())
            if total != self.teams - 1:
                raise ValueError(
                    f"Opponent archetype counts sum to {total}; expected {self.teams - 1}."
                )

    @staticmethod
    def _percentile(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=float)
        ranks[order] = np.arange(1, len(values) + 1, dtype=float)
        return ranks / max(len(values), 1)

    def _sample_run_projection(self, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
        if not self.risk_enabled:
            return self.proj.copy(), {
                "games": self.games_projected.copy(),
                "catastrophic": np.zeros(self.n, dtype=bool),
                "off_field": np.zeros(self.n, dtype=bool),
            }

        minor_missed = rng.poisson(self.minor_miss_lambda)
        catastrophic = rng.random(self.n) < self.catastrophic_prob
        catastrophic_extra = rng.integers(
            self.catastrophic_min_games,
            self.catastrophic_max_games + 1,
            size=self.n,
        ) * catastrophic.astype(int)

        off_field = rng.random(self.n) < self.off_field_prob
        off_draw = np.floor(
            rng.random(self.n) * np.maximum(self.off_field_max_games, 1)
        ).astype(int) + 1
        off_extra = off_draw * off_field.astype(int)

        missed = minor_missed + catastrophic_extra + off_extra
        games = np.clip(self.games_projected - missed, 0.0, 17.0)
        active_ppg = self.proj / np.maximum(self.games_projected, 1.0)
        replacement_ppg = self.replacement / 17.0
        additional_missed = np.maximum(0.0, self.games_projected - games)
        sampled = (
            active_ppg * games
            + replacement_ppg * additional_missed * self.replacement_capture
        )
        return sampled, {
            "games": games,
            "catastrophic": catastrophic,
            "off_field": off_field,
        }

    def _archetype_assignments(self, rng: np.random.Generator) -> dict[int, str]:
        opponent_slots = [s for s in range(1, self.teams + 1) if s != self.slot]

        fixed = self.archetype_cfg.get("fixed_by_slot", {})
        if fixed:
            assignments = {
                int(slot): str(archetype)
                for slot, archetype in fixed.items()
                if int(slot) != self.slot
            }
            if set(assignments) != set(opponent_slots):
                missing = sorted(set(opponent_slots) - set(assignments))
                extra = sorted(set(assignments) - set(opponent_slots))
                raise ValueError(
                    "fixed_by_slot must specify every opponent slot exactly once. "
                    f"Missing={missing}; extra={extra}"
                )
            return assignments

        counts = self.archetype_cfg.get("counts", {})
        bag = []
        for name, count in counts.items():
            bag.extend([name] * int(count))
        if not bag:
            bag = ["market"] * (self.teams - 1)
        rng.shuffle(bag)
        return dict(zip(opponent_slots, bag))

    def _manager_profile(self, team_slot: int, archetype: str) -> dict:
        profile = dict(self.archetype_cfg.get(archetype, {}))
        raw = self.manager_cfg.get(team_slot) or self.manager_cfg.get(str(team_slot)) or {}
        if not isinstance(raw, dict):
            return profile

        for key in (
            "market_weight",
            "custom_weight",
            "roster_need_strength",
            "randomness",
            "rookie_rank_bonus",
        ):
            if key in raw:
                profile[key] = raw[key]
        return profile

    def _seed_opponent_counts(self, draft_history: list[dict] | None) -> np.ndarray:
        counts = np.zeros((self.teams + 1, len(self.POSITIONS)), dtype=np.int16)
        for pick in draft_history or []:
            try:
                owner_slot = int(pick["owner_slot"])
                pid = str(pick["player_id"])
            except (KeyError, TypeError, ValueError):
                continue

            if owner_slot == self.slot:
                continue

            idx = self.id_to_idx.get(pid)
            if idx is None:
                continue

            code = self.pos_code[idx]
            if code >= 0 and 1 <= owner_slot <= self.teams:
                counts[owner_slot, code] += 1
        return counts

    def _roster_fit_vector(self, roster_counts: np.ndarray) -> np.ndarray:
        fit = np.full(self.n, 0.45, dtype=float)
        rcfg = self.engine.roster_cfg
        starters = {p: int(rcfg.get(p, 0)) for p in self.POSITIONS}

        for p, code in self.pos_to_code.items():
            mask = self.pos_code == code
            count = int(roster_counts[code])

            if count < starters.get(p, 0):
                fit[mask] = 1.0
                continue

            if p in set(rcfg.get("flex_eligible", [])):
                flex_codes = [
                    self.pos_to_code[x]
                    for x in rcfg.get("flex_eligible", [])
                    if x in self.pos_to_code
                ]
                base_filled = sum(
                    min(int(roster_counts[c]), starters[self.POSITIONS[c]])
                    for c in flex_codes
                )
                flex_excess = sum(int(roster_counts[c]) for c in flex_codes) - base_filled
                if flex_excess < int(rcfg.get("FLEX", 0)):
                    fit[mask] = 0.85

            cap = self.bench_caps.get(p)
            if cap is not None and count >= int(cap):
                fit[mask] = 0.05

        return fit

    def _scarcity_pct(self, available: np.ndarray) -> np.ndarray:
        scarcity = np.zeros(self.n, dtype=float)

        for code in range(len(self.POSITIONS)):
            idx = np.where(available & (self.pos_code == code))[0]
            if len(idx) == 0:
                continue

            ordered = idx[np.argsort(self.proj[idx])[::-1]]
            vals = self.proj[ordered]
            future_index = np.minimum(np.arange(len(vals)) + 4, len(vals) - 1)
            scarcity[ordered] = np.maximum(0.0, vals - vals[future_index])

        if not available.any():
            return scarcity

        vals = scarcity[available]
        order = np.argsort(vals, kind="mergesort")
        ranks = np.empty(len(vals), dtype=float)
        ranks[order] = np.arange(1, len(vals) + 1, dtype=float)
        out = np.zeros(self.n, dtype=float)
        out[np.where(available)[0]] = ranks / len(vals)
        return out

    def _urgency(self, next_pick: int) -> np.ndarray:
        sd = np.maximum(6.0, self.adp * 0.15)
        z = (float(next_pick) - self.adp) / sd
        return np.array([_norm_cdf(x) for x in z], dtype=float)

    def _next_my_pick(self, current_pick: int) -> int:
        for p in sorted(self.my_picks):
            if p > current_pick:
                return p
        return current_pick + self.teams * 2

    def _patbot_score_vector(
        self,
        available: np.ndarray,
        roster_counts: np.ndarray,
        pick: int,
    ) -> np.ndarray:
        next_pick = self._next_my_pick(pick)
        urgency = self._urgency(next_pick)
        roster_fit = self._roster_fit_vector(roster_counts)
        scarcity_pct = self._scarcity_pct(available)

        score = 100.0 * (
            self.vorp_pct * self.w_vorp
            + self.proj_pct * self.w_proj
            + urgency * self.w_urgency
            + scarcity_pct * self.w_scarcity
            + roster_fit * self.w_roster
            + self.expert_pct * self.expert_weight
        )
        score -= self.injury * self.injury_penalty

        round_no = (pick - 1) // self.teams + 1
        if round_no < self.min_round_k:
            score[self.pos == "K"] -= 35.0
        if round_no < self.min_round_def:
            score[self.pos == "DEF"] -= 35.0

        return np.where(available, score, -1_000_000_000.0)

    def patbot_pick(self, available: np.ndarray, roster_counts: np.ndarray, pick: int) -> int:
        return int(np.argmax(self._patbot_score_vector(available, roster_counts, pick)))

    def _base_roster_need_penalty(self, roster_counts: np.ndarray, round_no: int) -> np.ndarray:
        penalty = np.zeros(self.n, dtype=float)

        for p in ("K", "DEF"):
            code = self.pos_to_code[p]
            mask = self.pos_code == code
            if round_no < 12:
                penalty[mask] += 100.0
            if roster_counts[code] >= 1:
                penalty[mask] += 150.0

        code = self.pos_to_code["QB"]
        mask = self.pos_code == code
        if roster_counts[code] >= 2:
            penalty[mask] += 100.0
        elif roster_counts[code] >= 1 and round_no < 9:
            penalty[mask] += 18.0
        elif roster_counts[code] == 0 and round_no >= 8:
            penalty[mask] -= 8.0

        code = self.pos_to_code["TE"]
        mask = self.pos_code == code
        if roster_counts[code] >= 2:
            penalty[mask] += 80.0
        elif roster_counts[code] >= 1 and round_no < 9:
            penalty[mask] += 15.0

        for p, need, cap in (("RB", 2, 6), ("WR", 3, 7)):
            code = self.pos_to_code[p]
            mask = self.pos_code == code
            if roster_counts[code] < need:
                penalty[mask] -= 5.0
            if roster_counts[code] >= cap:
                penalty[mask] += 60.0

        return penalty

    def opponent_pick(
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

        score = np.where(available, score, 1_000_000_000.0)
        return int(np.argmin(score))

    def _take_opponent_pick(
        self,
        pick: int,
        available: np.ndarray,
        opp_counts: np.ndarray,
        archetypes: dict[int, str],
        market_latent: np.ndarray,
        custom_noise_base: np.ndarray,
    ) -> tuple[int, str]:
        team_slot = _team_slot_for_pick(pick, self.teams)
        round_no = (pick - 1) // self.teams + 1
        archetype = archetypes.get(team_slot, "market")
        profile = self._manager_profile(team_slot, archetype)
        randomness = float(profile.get("randomness", 1.0))
        custom_latent = np.maximum(
            1.0,
            self.custom_rank + custom_noise_base * randomness,
        )
        idx = self.opponent_pick(
            available,
            market_latent,
            custom_latent,
            opp_counts[team_slot],
            round_no,
            profile,
        )
        available[idx] = False
        code = self.pos_code[idx]
        if code >= 0:
            opp_counts[team_slot, code] += 1
        return idx, archetype

    def _lookahead_pick(
        self,
        available: np.ndarray,
        my_counts: np.ndarray,
        pick: int,
        opp_counts: np.ndarray,
        archetypes: dict[int, str],
        market_latent: np.ndarray,
        custom_noise_base: np.ndarray,
    ) -> int:
        greedy = self.patbot_pick(available, my_counts, pick)
        round_no = (pick - 1) // self.teams + 1
        if not self.lookahead_enabled or round_no not in self.lookahead_rounds:
            return greedy

        next_pick = self._next_my_pick(pick)
        gap = next_pick - pick
        if gap <= 0 or gap > self.lookahead_max_gap:
            return greedy

        current_scores = self._patbot_score_vector(available, my_counts, pick)
        candidates = np.where(available)[0]
        if len(candidates) <= 1:
            return greedy
        candidates = candidates[np.argsort(current_scores[candidates])[::-1]]
        candidates = candidates[: self.lookahead_branch_width]

        best_idx = int(greedy)
        best_value = -float("inf")

        for candidate in candidates:
            branch_available = available.copy()
            branch_opp_counts = opp_counts.copy()
            branch_my_counts = my_counts.copy()

            branch_available[candidate] = False
            code = self.pos_code[candidate]
            if code >= 0:
                branch_my_counts[code] += 1

            for future_pick in range(pick + 1, next_pick):
                if future_pick in self.my_picks:
                    break
                if not branch_available.any():
                    break
                self._take_opponent_pick(
                    future_pick,
                    branch_available,
                    branch_opp_counts,
                    archetypes,
                    market_latent,
                    custom_noise_base,
                )

            if not branch_available.any():
                future_value = 0.0
                future_idx = None
            else:
                future_scores = self._patbot_score_vector(
                    branch_available,
                    branch_my_counts,
                    next_pick,
                )
                future_idx = int(np.argmax(future_scores))
                future_value = float(future_scores[future_idx])

            pair_vorp = max(float(self.vorp[candidate]), 0.0)
            if future_idx is not None:
                pair_vorp += max(float(self.vorp[future_idx]), 0.0)

            path_value = (
                float(current_scores[candidate])
                + self.lookahead_future_weight * future_value
                + self.lookahead_vorp_weight * pair_vorp
            )
            if path_value > best_value:
                best_value = path_value
                best_idx = int(candidate)

        return best_idx

    def evaluate_roster(self, mine: list[int], projection_override: np.ndarray | None = None) -> dict:
        rcfg = self.engine.roster_cfg
        evalcfg = self.roster_eval_cfg
        proj = self.proj if projection_override is None else projection_override
        vorp = proj - self.replacement

        by_pos = {}
        for p in self.POSITIONS:
            idxs = [i for i in mine if self.pos[i] == p]
            idxs.sort(key=lambda i: proj[i], reverse=True)
            by_pos[p] = idxs

        starters = []
        used = set()
        missing = Counter()

        for p in ("QB", "RB", "WR", "TE"):
            need = int(rcfg.get(p, 0))
            chosen = by_pos[p][:need]
            starters.extend(chosen)
            used.update(chosen)
            if len(chosen) < need:
                missing[p] = need - len(chosen)

        flex_need = int(rcfg.get("FLEX", 0))
        flex_pool = [
            i
            for i in mine
            if i not in used and self.pos[i] in set(rcfg.get("flex_eligible", []))
        ]
        flex_pool.sort(key=lambda i: proj[i], reverse=True)
        flex_chosen = flex_pool[:flex_need]
        starters.extend(flex_chosen)
        used.update(flex_chosen)
        if len(flex_chosen) < flex_need:
            missing["FLEX"] = flex_need - len(flex_chosen)

        starter_vorp = float(np.sum(vorp[starters])) if starters else 0.0
        bench = [i for i in mine if i not in used and self.pos[i] not in {"K", "DEF"}]
        bench_vorp = (
            float(np.sum(np.maximum(vorp[bench], 0.0)))
            if bench
            else 0.0
        )

        bench_discount = float(evalcfg.get("bench_vorp_discount", 0.2))
        score = starter_vorp + bench_discount * bench_vorp

        missing_penalty_cfg = evalcfg.get("missing_starter_penalty", {})
        for p, count in missing.items():
            score -= float(missing_penalty_cfg.get(p, 0.0)) * int(count)

        counts = Counter(self.pos[i] for i in mine)
        empty_cfg = evalcfg.get("empty_group_penalty", {})
        for p, penalty in empty_cfg.items():
            if counts[p] == 0:
                score -= float(penalty)

        bonus_cfg = evalcfg.get("construction_bonus", {})
        if counts["QB"] >= 1:
            score += float(bonus_cfg.get("has_qb", 0))
        if counts["TE"] >= 1:
            score += float(bonus_cfg.get("has_te", 0))
        if counts["RB"] >= 2:
            score += float(bonus_cfg.get("has_two_rb", 0))
        if counts["WR"] >= 3:
            score += float(bonus_cfg.get("has_three_wr", 0))

        flex_eligible_count = counts["RB"] + counts["WR"] + counts["TE"]
        base_flex_need = (
            int(rcfg.get("RB", 0))
            + int(rcfg.get("WR", 0))
            + int(rcfg.get("TE", 0))
            + int(rcfg.get("FLEX", 0))
        )
        if flex_eligible_count >= base_flex_need + 1:
            score += float(bonus_cfg.get("has_flex_depth", 0))

        return {
            "lineup_score": float(score),
            "starter_vorp": starter_vorp,
            "bench_vorp": bench_vorp,
            "missing_starters": dict(missing),
        }

    def simulate_candidate(
        self,
        current_pick: int,
        drafted_ids: set[str],
        my_roster_ids: list[str],
        candidate_id: str,
        runs: int,
        through_round: int,
        seed: int,
        draft_history: list[dict] | None = None,
    ) -> dict:
        drafted_idx = {
            self.id_to_idx[str(x)]
            for x in drafted_ids
            if str(x) in self.id_to_idx
        }
        my_idx = [
            self.id_to_idx[str(x)]
            for x in my_roster_ids
            if str(x) in self.id_to_idx
        ]
        candidate_idx = self.id_to_idx[str(candidate_id)]

        rng = np.random.default_rng(seed)
        last_pick = self.teams * int(through_round)

        lineup_scores = np.empty(runs, dtype=float)
        starter_vorps = np.empty(runs, dtype=float)
        total_projections = np.empty(runs, dtype=float)
        candidate_games = np.empty(runs, dtype=float)
        candidate_catastrophic = np.zeros(runs, dtype=bool)
        candidate_off_field = np.zeros(runs, dtype=bool)
        second_names = Counter()
        third_names = Counter()
        immediate_next_names = Counter()

        base_available = np.ones(self.n, dtype=bool)
        if drafted_idx:
            base_available[list(drafted_idx)] = False

        latent_sd = np.maximum(self.sd_floor, self.adp * self.sd_pct)

        for run in range(runs):
            available = base_available.copy()
            opp_counts = self._seed_opponent_counts(draft_history)
            archetypes = self._archetype_assignments(rng)

            my_counts = np.zeros(len(self.POSITIONS), dtype=np.int16)
            mine = list(my_idx)
            for idx in mine:
                code = self.pos_code[idx]
                if code >= 0:
                    my_counts[code] += 1

            market_latent = np.maximum(1.0, rng.normal(self.adp, latent_sd))
            custom_noise_base = rng.normal(
                0.0,
                np.maximum(3.0, self.custom_rank * 0.06),
            )

            followup_pick_no = 0

            for pick in range(int(current_pick), last_pick + 1):
                if not available.any():
                    break

                if pick in self.my_picks:
                    if pick == int(current_pick):
                        idx = candidate_idx
                        if not available[idx]:
                            break
                    else:
                        idx = self._lookahead_pick(
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

                    code = self.pos_code[idx]
                    if code >= 0:
                        my_counts[code] += 1

                    followup_pick_no += 1
                    if followup_pick_no == 2:
                        second_names[self.names[idx]] += 1
                    elif followup_pick_no == 3:
                        third_names[self.names[idx]] += 1

                else:
                    opp_idx, _ = self._take_opponent_pick(
                        pick,
                        available,
                        opp_counts,
                        archetypes,
                        market_latent,
                        custom_noise_base,
                    )
                    if pick == int(current_pick) + 1:
                        immediate_next_names[self.names[opp_idx]] += 1

            run_proj, risk_meta = self._sample_run_projection(rng)
            eval_result = self.evaluate_roster(mine, projection_override=run_proj)
            lineup_scores[run] = eval_result["lineup_score"]
            starter_vorps[run] = eval_result["starter_vorp"]
            total_projections[run] = float(np.sum(run_proj[mine]))
            candidate_games[run] = float(risk_meta["games"][candidate_idx])
            candidate_catastrophic[run] = bool(risk_meta["catastrophic"][candidate_idx])
            candidate_off_field[run] = bool(risk_meta["off_field"][candidate_idx])

        def top_counter(counter: Counter, n: int = 5):
            total = sum(counter.values()) or 1
            return [
                {
                    "player": name,
                    "pct": round(100.0 * count / total, 1),
                }
                for name, count in counter.most_common(n)
            ]

        return {
            "candidate": self.names[candidate_idx],
            "candidate_id": str(candidate_id),
            "runs": runs,
            "avg_lineup_score": round(float(np.mean(lineup_scores)), 2),
            "p10_lineup_score": round(float(np.percentile(lineup_scores, 10)), 2),
            "p25_lineup_score": round(float(np.percentile(lineup_scores, 25)), 2),
            "p75_lineup_score": round(float(np.percentile(lineup_scores, 75)), 2),
            "avg_starter_vorp": round(float(np.mean(starter_vorps)), 2),
            "avg_roster_projected_points": round(float(np.mean(total_projections)), 2),
            "avg_candidate_games": round(float(np.mean(candidate_games)), 2),
            "candidate_catastrophic_pct": round(100.0 * float(np.mean(candidate_catastrophic)), 1),
            "candidate_off_field_pct": round(100.0 * float(np.mean(candidate_off_field)), 1),
            "most_common_immediate_next_pick": top_counter(immediate_next_names),
            "most_common_second_pick": top_counter(second_names),
            "most_common_third_pick": top_counter(third_names),
            "lookahead_enabled": self.lookahead_enabled,
            "risk_enabled": self.risk_enabled,
        }


def simulate_candidate(
    engine,
    current_pick: int,
    drafted_ids: set[str],
    my_roster_ids: list[str],
    candidate_id: str,
    runs: int = 300,
    through_round: int = 8,
    seed: int = 42,
    draft_history: list[dict] | None = None,
) -> dict:
    sim = FastDraftSimulator(engine)
    return sim.simulate_candidate(
        current_pick=current_pick,
        drafted_ids=drafted_ids,
        my_roster_ids=my_roster_ids,
        candidate_id=candidate_id,
        runs=int(runs),
        through_round=int(through_round),
        seed=int(seed),
        draft_history=draft_history,
    )


def compare_candidates(
    engine,
    current_pick: int,
    drafted_ids: set[str],
    my_roster_ids: list[str],
    candidate_ids: list[str],
    runs: int = 300,
    through_round: int = 8,
    draft_history: list[dict] | None = None,
):
    sim = FastDraftSimulator(engine)
    results = []

    # Common random numbers: every candidate faces the same simulated room on
    # run 1, the same room on run 2, etc. Risk shocks are sampled from the same
    # seeded stream as well, so close comparisons keep paired downside paths.
    for pid in candidate_ids:
        results.append(
            sim.simulate_candidate(
                current_pick=current_pick,
                drafted_ids=drafted_ids,
                my_roster_ids=my_roster_ids,
                candidate_id=str(pid),
                runs=int(runs),
                through_round=int(through_round),
                seed=sim.comparison_seed,
                draft_history=draft_history,
            )
        )

    summary = pd.DataFrame(
        [
            {
                "Candidate": r["candidate"],
                "Avg Lineup Score": r["avg_lineup_score"],
                "10th %ile": r["p10_lineup_score"],
                "25th %ile": r["p25_lineup_score"],
                "75th %ile": r["p75_lineup_score"],
                "Avg Starter VORP": r["avg_starter_vorp"],
                "Avg Drafted Proj": r["avg_roster_projected_points"],
                "Avg Candidate Games": r["avg_candidate_games"],
                "Catastrophic Tail %": r["candidate_catastrophic_pct"],
                "Off-field Event %": r["candidate_off_field_pct"],
                "Runs": r["runs"],
            }
            for r in results
        ]
    ).sort_values("Avg Lineup Score", ascending=False)

    return summary.reset_index(drop=True), results
