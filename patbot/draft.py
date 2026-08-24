from __future__ import annotations

import math
from collections import Counter
import numpy as np
import pandas as pd


def snake_pick(round_number: int, teams: int, slot: int) -> int:
    within = slot if round_number % 2 else teams + 1 - slot
    return (round_number - 1) * teams + within


def all_team_picks(teams: int, slot: int, rounds: int = 20) -> list[int]:
    return [snake_pick(r, teams, slot) for r in range(1, rounds + 1)]


def next_team_pick(current_overall_pick: int, teams: int, slot: int, rounds: int = 20):
    for p in all_team_picks(teams, slot, rounds):
        if p >= current_overall_pick:
            return p
    return None


def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def survive_probability(adp: float, next_pick: int) -> float:
    sd = max(6.0, float(adp) * 0.15)
    selected_by_next = norm_cdf((float(next_pick) - float(adp)) / sd)
    return float(np.clip(1.0 - selected_by_next, 0.0, 1.0))


def percentile(series: pd.Series) -> pd.Series:
    if len(series) <= 1:
        return pd.Series([1.0] * len(series), index=series.index)
    return series.rank(method="average", pct=True)


def low_is_good_percentile(series: pd.Series) -> pd.Series:
    valid = pd.to_numeric(series, errors="coerce")
    return 1.0 - valid.rank(method="average", pct=True) + (1.0 / max(len(valid), 1))


class DraftEngine:
    def __init__(self, players: pd.DataFrame, config: dict):
        self.players = players.copy()
        self.config = config
        self.league = config["league"]
        self.roster_cfg = config["roster"]
        self.engine_cfg = config["draft_engine"]

    def replacement_levels(self) -> dict[str, float]:
        levels = {}
        for pos, rank in self.engine_cfg["replacement_rank"].items():
            group = self.players[self.players["pos"] == pos].sort_values(
                "proj_points", ascending=False
            )
            if group.empty:
                levels[pos] = 0.0
            else:
                idx = min(int(rank) - 1, len(group) - 1)
                levels[pos] = float(group.iloc[idx]["proj_points"])
        return levels

    def roster_fit(self, pos: str, roster_positions: list[str]) -> float:
        counts = Counter(roster_positions)
        starters = {
            p: int(self.roster_cfg.get(p, 0))
            for p in ["QB", "RB", "WR", "TE", "K", "DEF"]
        }

        if counts[pos] < starters.get(pos, 0):
            return 1.0

        flex_eligible = set(self.roster_cfg.get("flex_eligible", []))
        if pos in flex_eligible:
            base_filled = sum(min(counts[p], starters[p]) for p in flex_eligible)
            flex_excess = sum(counts[p] for p in flex_eligible) - base_filled
            if flex_excess < int(self.roster_cfg.get("FLEX", 0)):
                return 0.85

        cap = self.engine_cfg.get("bench_position_caps", {}).get(pos)
        if cap is not None and counts[pos] >= int(cap):
            return 0.05
        return 0.45

    def recommend(self, current_pick: int, drafted_ids, roster_positions, top_n: int = 12):
        drafted_ids = {str(x) for x in drafted_ids}
        available = self.players[~self.players["player_id"].isin(drafted_ids)].copy()
        if available.empty:
            return available

        teams = int(self.league["teams"])
        slot = int(self.league["draft_slot"])
        next_pick = next_team_pick(current_pick + 1, teams, slot)
        if next_pick is None:
            next_pick = current_pick + teams * 2

        levels = self.replacement_levels()
        available["replacement"] = available["pos"].map(levels).fillna(0.0)
        available["vorp"] = available["proj_points"] - available["replacement"]

        scarcity = {}
        for _, group in available.groupby("pos"):
            ordered = group.sort_values("proj_points", ascending=False)
            vals = ordered["proj_points"].to_numpy()
            for j, idx in enumerate(ordered.index):
                future = min(j + 4, len(vals) - 1)
                scarcity[idx] = max(0.0, float(vals[j] - vals[future]))
        available["scarcity"] = pd.Series(scarcity)

        available["survive_next"] = available["adp"].apply(
            lambda x: survive_probability(x, next_pick)
        )
        available["urgency"] = 1 - available["survive_next"]
        available["roster_fit"] = available["pos"].apply(
            lambda p: self.roster_fit(p, roster_positions)
        )

        available["projection_pct"] = percentile(available["proj_points"])
        available["vorp_pct"] = percentile(available["vorp"])
        available["scarcity_pct"] = percentile(available["scarcity"])

        # v0.3 independent expert ranking signal. If unavailable, neutral rather than punitive.
        if "expert_rank" in available.columns and available["expert_rank"].notna().any():
            available["expert_pct"] = low_is_good_percentile(available["expert_rank"])
            available["expert_pct"] = available["expert_pct"].fillna(0.5)
            expert_weight = 0.10
        else:
            available["expert_pct"] = 0.5
            expert_weight = 0.0

        # Keep the v0.2 engine behavior recognizable, but reserve 10% for independent experts.
        w = self.engine_cfg["weights"]
        base_total = sum(float(w[k]) for k in ["vorp","projection","urgency","scarcity","roster_fit"])
        scale = (1.0 - expert_weight) / base_total

        available["score"] = 100 * (
            available["vorp_pct"] * float(w["vorp"]) * scale
            + available["projection_pct"] * float(w["projection"]) * scale
            + available["urgency"] * float(w["urgency"]) * scale
            + available["scarcity_pct"] * float(w["scarcity"]) * scale
            + available["roster_fit"] * float(w["roster_fit"]) * scale
            + available["expert_pct"] * expert_weight
        )

        available["score"] -= available["injury_risk"].clip(0, 1) * float(
            self.engine_cfg.get("injury_risk_penalty", 0)
        )

        round_number = ((current_pick - 1) // teams) + 1
        if round_number < int(self.engine_cfg.get("min_round_k", 13)):
            available.loc[available["pos"] == "K", "score"] -= 35
        if round_number < int(self.engine_cfg.get("min_round_def", 13)):
            available.loc[available["pos"] == "DEF", "score"] -= 35

        available["next_patbot_pick"] = next_pick
        available["survive_next_pct"] = (100 * available["survive_next"]).round(1)
        for c in ["score", "vorp", "scarcity"]:
            available[c] = available[c].round(2)

        return available.sort_values(
            ["score", "proj_points", "adp"], ascending=[False, False, True]
        ).head(top_n).reset_index(drop=True)

    @staticmethod
    def explain_row(row: pd.Series) -> str:
        tier_text = ""
        if pd.notna(row.get("consensus_tier")):
            tier_text = f" Consensus Tier {int(row['consensus_tier'])}."
        expert_text = ""
        if pd.notna(row.get("expert_rank")):
            expert_text = f" Independent expert rank: {row['expert_rank']:.1f}."
        return (
            f"{row['name']} ({row['pos']}) is the current model pick."
            f"{tier_text}{expert_text} "
            f"Custom projection: {row['proj_points']:.1f}; VORP: {row['vorp']:.1f}. "
            f"Estimated chance of surviving to PatBot's next pick "
            f"({int(row['next_patbot_pick'])}): {row['survive_next_pct']:.1f}%. "
            f"Four-player positional drop: {row['scarcity']:.1f} projected points."
        )
