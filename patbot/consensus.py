from __future__ import annotations

import numpy as np
import pandas as pd


def _pct_high_good(series: pd.Series) -> pd.Series:
    valid = pd.to_numeric(series, errors="coerce")
    return valid.rank(pct=True, method="average")


def _pct_low_good(series: pd.Series) -> pd.Series:
    valid = pd.to_numeric(series, errors="coerce")
    return 1.0 - valid.rank(pct=True, method="average") + (1.0 / max(len(valid), 1))


def add_consensus_values(players: pd.DataFrame, config: dict) -> pd.DataFrame:
    out = players.copy()

    levels = {}
    replacement = config["draft_engine"]["replacement_rank"]
    for pos, rank in replacement.items():
        g = out[out["pos"] == pos].sort_values("proj_points", ascending=False)
        if g.empty:
            levels[pos] = 0.0
        else:
            idx = min(int(rank) - 1, len(g) - 1)
            levels[pos] = float(g.iloc[idx]["proj_points"])

    out["static_replacement"] = out["pos"].map(levels).fillna(0.0)
    out["static_vorp"] = out["proj_points"] - out["static_replacement"]
    out["vorp_pct"] = _pct_high_good(out["static_vorp"])

    expert_col = "generic_expert_rank" if "generic_expert_rank" in out else "expert_rank"
    if expert_col in out:
        out["expert_pct"] = _pct_low_good(out[expert_col])
    else:
        out["expert_pct"] = np.nan

    if "athletic_rank" in out:
        out["athletic_pct"] = _pct_low_good(out["athletic_rank"])
    else:
        out["athletic_pct"] = np.nan

    market_col = "market_adp" if "market_adp" in out else "adp"
    out["market_pct"] = _pct_low_good(out[market_col])

    weights = config.get("v03_consensus", {}).get("weights", {})
    component_weights = {
        "vorp_pct": float(weights.get("custom_vorp", 0.45)),
        "athletic_pct": float(weights.get("athletic_custom", 0.20)),
        "expert_pct": float(weights.get("expert_rank", 0.20)),
        "market_pct": float(weights.get("market_rank", 0.15)),
    }

    def blended(row):
        components = []
        for column, weight in component_weights.items():
            value = row.get(column)
            if pd.notna(value) and weight > 0:
                components.append((float(value), weight))
        total_w = sum(w for _, w in components)
        if not total_w:
            return np.nan
        return 100.0 * sum(v * w for v, w in components) / total_w

    out["consensus_value"] = out.apply(blended, axis=1)
    out["consensus_value"] = out["consensus_value"].round(2)
    out = assign_tiers(out, config)
    return out


def assign_tiers(players: pd.DataFrame, config: dict) -> pd.DataFrame:
    out = players.copy()
    ordered = out.sort_values("consensus_value", ascending=False)
    values = ordered["consensus_value"].dropna().to_numpy()

    if len(values) < 2:
        out["consensus_tier"] = 1
        out["tier_cliff"] = False
        return out

    gaps = values[:-1] - values[1:]
    positive = gaps[gaps > 0]
    median_gap = float(np.median(positive)) if len(positive) else 0.5

    tc = config.get("v03_consensus", {}).get("tiering", {})
    threshold = max(
        float(tc.get("absolute_gap_floor", 2.0)),
        median_gap * float(tc.get("gap_multiplier", 2.5)),
    )
    min_size = int(tc.get("minimum_tier_size", 2))
    max_size = int(tc.get("maximum_tier_size", 8))

    tiers = {}
    cliffs = {}
    tier = 1
    current_size = 0
    idxs = list(ordered.index)

    for i, idx in enumerate(idxs):
        tiers[idx] = tier
        cliffs[idx] = False
        current_size += 1

        if i == len(idxs) - 1:
            continue

        gap = float(ordered.iloc[i]["consensus_value"] - ordered.iloc[i + 1]["consensus_value"])
        natural_break = current_size >= min_size and gap >= threshold
        forced_break = current_size >= max_size

        if natural_break or forced_break:
            cliffs[idx] = natural_break
            tier += 1
            current_size = 0

    out["consensus_tier"] = pd.Series(tiers)
    out["tier_cliff"] = pd.Series(cliffs).fillna(False)
    return out
