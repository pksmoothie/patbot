import numpy as np
import pandas as pd

from patbot.strategy import (
    compute_strategy_metrics,
    sample_performance_factors,
    strategy_phase,
)


def _config():
    return {
        "championship_strategy": {
            "enabled": True,
            "market_edge_scale_picks": 60,
            "performance_volatility": {
                "enabled": True,
                "base_sigma_by_position": {"QB": 0.08, "RB": 0.14, "WR": 0.15, "TE": 0.16},
                "default_sigma": 0.12,
                "early_career_sigma_max": 0.08,
                "positive_market_edge_sigma_max": 0.04,
                "max_sigma": 0.30,
                "factor_floor": 0.60,
                "factor_cap": 1.60,
            },
            "league_winner_components": {
                "positional_ceiling": 0.55,
                "market_edge": 0.25,
                "early_career": 0.20,
            },
            "late_upside_position_multiplier": {"QB": 0.45, "RB": 1.0, "WR": 1.0, "TE": 0.75},
            "round_phases": [
                {"name": "Foundation", "through_round": 3, "upside_weight": 0.01, "risk_penalty_multiplier": 1.0},
                {"name": "Build", "through_round": 7, "upside_weight": 0.03, "risk_penalty_multiplier": 0.85},
                {"name": "Upside", "through_round": 11, "upside_weight": 0.08, "risk_penalty_multiplier": 0.55},
                {"name": "Lottery", "through_round": 15, "upside_weight": 0.14, "risk_penalty_multiplier": 0.30},
            ],
        }
    }


def test_strategy_gets_more_upside_seeking_late():
    cfg = _config()
    early = strategy_phase(2, cfg)
    late = strategy_phase(13, cfg)
    assert late["upside_weight"] > early["upside_weight"]
    assert late["risk_penalty_multiplier"] < early["risk_penalty_multiplier"]


def test_rookie_market_discount_creates_more_optionality_than_safe_veteran():
    players = pd.DataFrame([
        {"player_id": "1", "name": "Veteran", "pos": "WR", "proj_points": 210, "adp": 30, "expert_rank": 35, "years_exp": 7, "fp_age": 30},
        {"player_id": "2", "name": "Rookie", "pos": "WR", "proj_points": 210, "adp": 100, "expert_rank": 55, "years_exp": 0, "fp_age": 22},
    ])
    metrics = compute_strategy_metrics(players, {"WR": 160}, _config())
    assert metrics.loc[1, "performance_sigma"] > metrics.loc[0, "performance_sigma"]
    assert metrics.loc[1, "league_winner_score"] > metrics.loc[0, "league_winner_score"]
    assert metrics.loc[1, "q90_points"] > metrics.loc[0, "q90_points"]


def test_zero_sigma_produces_no_performance_shock():
    cfg = _config()
    factors = sample_performance_factors(np.random.default_rng(1), np.zeros(5), cfg)
    assert np.allclose(factors, np.ones(5))


def test_performance_shocks_are_capped_and_include_upside():
    cfg = _config()
    factors = sample_performance_factors(np.random.default_rng(9), np.full(2000, 0.28), cfg)
    assert factors.min() >= 0.60
    assert factors.max() <= 1.60
    assert (factors >= 1.25).any()
