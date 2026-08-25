from __future__ import annotations

import math

import numpy as np
import pandas as pd


Z90 = 1.2815515655446004


def _numeric_series(frame: pd.DataFrame, name: str, default: float) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").fillna(default).astype(float)


def _percentile(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if len(values) <= 1:
        return pd.Series(1.0, index=series.index, dtype=float)
    return values.rank(method="average", pct=True).fillna(0.5)


def _early_career_score(years_exp: float, age: float) -> float:
    if not np.isnan(years_exp):
        if years_exp <= 0:
            return 1.00
        if years_exp <= 1:
            return 0.82
        if years_exp <= 2:
            return 0.62
        if years_exp <= 3:
            return 0.38
        if years_exp <= 4:
            return 0.22
        return 0.08
    if not np.isnan(age):
        if age <= 23:
            return 0.75
        if age <= 25:
            return 0.48
        if age <= 27:
            return 0.25
    return 0.10


def strategy_phase(round_number: int, config: dict) -> dict:
    scfg = config.get("championship_strategy", {})
    phases = scfg.get("round_phases", [])
    if not phases:
        return {
            "name": "Baseline",
            "through_round": 99,
            "upside_weight": 0.0,
            "risk_penalty_multiplier": 1.0,
        }
    ordered = sorted(phases, key=lambda x: int(x.get("through_round", 99)))
    for phase in ordered:
        if int(round_number) <= int(phase.get("through_round", 99)):
            return phase
    return ordered[-1]


def compute_strategy_metrics(
    players: pd.DataFrame,
    replacement_levels: dict[str, float],
    config: dict,
) -> pd.DataFrame:
    """Estimate performance uncertainty and late-round championship optionality.

    The league-winner signal does not reward injury/legal risk. It rewards a
    plausible high-end football outcome: positional ceiling, positive model-vs-
    market disagreement, and early-career breakout optionality. Round-specific
    strategy decides how much PatBot cares about this signal.
    """
    out = pd.DataFrame(index=players.index)
    scfg = config.get("championship_strategy", {})
    enabled = bool(scfg.get("enabled", False))

    proj = _numeric_series(players, "proj_points", 0.0)
    pos = players.get("pos", pd.Series("", index=players.index)).astype(str).str.upper()
    replacement = pos.map({str(k): float(v) for k, v in replacement_levels.items()}).fillna(0.0)
    vorp = proj - replacement

    out["strategy_model_rank"] = np.arange(1, len(players) + 1, dtype=float)
    out["performance_sigma"] = 0.0
    out["q90_points"] = proj
    out["league_winner_score"] = 0.0
    out["market_edge_score"] = 0.0
    out["early_career_score"] = 0.0

    if not enabled or players.empty:
        return out

    proj_pct = _percentile(proj)
    vorp_pct = _percentile(vorp)

    expert_rank = _numeric_series(players, "expert_rank", np.nan)
    if expert_rank.notna().any():
        expert_pct = 1.0 - expert_rank.rank(method="average", pct=True) + (1.0 / max(len(players), 1))
        expert_pct = expert_pct.fillna(0.5).clip(0.0, 1.0)
    else:
        expert_pct = pd.Series(0.5, index=players.index, dtype=float)

    goodness = 0.58 * vorp_pct + 0.27 * proj_pct + 0.15 * expert_pct
    model_rank = goodness.rank(method="first", ascending=False)
    out["strategy_model_rank"] = model_rank

    adp = _numeric_series(players, "adp", 999.0)
    edge_scale = max(1.0, float(scfg.get("market_edge_scale_picks", 60.0)))
    positive_market_edge = ((adp - model_rank) / edge_scale).clip(lower=0.0, upper=1.0)
    out["market_edge_score"] = positive_market_edge

    years = pd.to_numeric(players.get("years_exp", pd.Series(np.nan, index=players.index)), errors="coerce")
    ages = pd.to_numeric(players.get("fp_age", pd.Series(np.nan, index=players.index)), errors="coerce")
    early = pd.Series(
        [_early_career_score(float(y), float(a)) for y, a in zip(years, ages)],
        index=players.index,
        dtype=float,
    )
    out["early_career_score"] = early

    vcfg = scfg.get("performance_volatility", {})
    base_sigma_cfg = vcfg.get("base_sigma_by_position", {})
    base_sigma = pos.map({str(k): float(v) for k, v in base_sigma_cfg.items()}).fillna(
        float(vcfg.get("default_sigma", 0.12))
    )
    experience_sigma = early * float(vcfg.get("early_career_sigma_max", 0.08))
    market_sigma = positive_market_edge * float(vcfg.get("positive_market_edge_sigma_max", 0.04))
    sigma = base_sigma + experience_sigma + market_sigma
    sigma = sigma.clip(
        lower=0.0,
        upper=float(vcfg.get("max_sigma", 0.30)),
    )
    sigma = sigma.where(~pos.isin(["K", "DEF"]), 0.0)
    out["performance_sigma"] = sigma

    q90_factor = np.exp(-0.5 * np.square(sigma) + Z90 * sigma)
    q90_points = proj * q90_factor
    out["q90_points"] = q90_points
    ceiling_vorp = q90_points - replacement

    within_position_ceiling = ceiling_vorp.groupby(pos).rank(method="average", pct=True).fillna(0.5)
    position_mult_cfg = scfg.get("late_upside_position_multiplier", {})
    position_mult = pos.map({str(k): float(v) for k, v in position_mult_cfg.items()}).fillna(1.0)

    weights = scfg.get("league_winner_components", {})
    w_ceiling = float(weights.get("positional_ceiling", 0.55))
    w_edge = float(weights.get("market_edge", 0.25))
    w_youth = float(weights.get("early_career", 0.20))
    denom = max(w_ceiling + w_edge + w_youth, 1e-9)
    raw = (
        within_position_ceiling * w_ceiling
        + positive_market_edge * w_edge
        + early * w_youth
    ) / denom
    out["league_winner_score"] = (100.0 * raw * position_mult).clip(0.0, 100.0)
    return out


def sample_performance_factors(
    rng: np.random.Generator,
    sigma: np.ndarray,
    config: dict,
) -> np.ndarray:
    scfg = config.get("championship_strategy", {})
    vcfg = scfg.get("performance_volatility", {})
    if not bool(scfg.get("enabled", False)) or not bool(vcfg.get("enabled", True)):
        return np.ones(len(sigma), dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    z = rng.normal(0.0, 1.0, size=len(sigma))
    factor = np.exp(-0.5 * np.square(sigma) + sigma * z)
    return np.clip(
        factor,
        float(vcfg.get("factor_floor", 0.60)),
        float(vcfg.get("factor_cap", 1.60)),
    )
