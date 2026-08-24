from __future__ import annotations

import math


def _g(stats: dict, *keys: str) -> float:
    """Return first present numeric stat, defaulting to zero."""
    for key in keys:
        if key in stats and stats[key] is not None:
            try:
                return float(stats[key])
            except (TypeError, ValueError):
                pass
    return 0.0


def _norm_survival(threshold: float, mean: float, sd: float) -> float:
    """P(X >= threshold) for a normal approximation."""
    if sd <= 0:
        return 1.0 if mean >= threshold else 0.0
    z = (threshold - mean) / sd
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return max(0.0, min(1.0, 1.0 - cdf))


def expected_game_threshold_bonuses(
    season_yards: float,
    games: float,
    bonuses: list[dict],
    sd_floor: float,
    sd_pct_of_mean: float,
) -> float:
    """Estimate expected *season* bonus points from per-game yardage thresholds.

    We only have preseason season-total projections. The league bonuses are
    single-game thresholds, so v0.2 models game yardage around projected YPG.
    This is an approximation and is intentionally isolated/tunable.
    """
    if games <= 0 or season_yards <= 0 or not bonuses:
        return 0.0

    mean = season_yards / games
    sd = max(float(sd_floor), mean * float(sd_pct_of_mean))
    expected_per_game = 0.0
    for bonus in bonuses:
        p = _norm_survival(float(bonus["threshold"]), mean, sd)
        expected_per_game += p * float(bonus["points"])
    return games * expected_per_game


def _individual_return_tds(stats: dict) -> float:
    # Prefer Sleeper's combined special-teams TD if present to avoid adding
    # kick-return + punt-return fields on top of an already-combined field.
    if "st_td" in stats and stats["st_td"] is not None:
        return _g(stats, "st_td")
    return _g(stats, "kr_td") + _g(stats, "pr_td")


def score_season_projection(
    stats: dict,
    scoring: dict,
    bonus_model: dict | None = None,
    position: str | None = None,
) -> dict:
    """Score a Sleeper-style season projection under PatBot's custom rules.

    Returns a breakdown so the draft room can explain why players move.
    For K/DEF, v0.2 uses provider projected points until Yahoo settings import.
    """
    position = (position or "").upper()
    if position in {"K", "DEF", "DST"}:
        provider = _g(stats, "pts_ppr", "pts_half_ppr", "pts_std")
        return {
            "base_points": provider,
            "bonus_points": 0.0,
            "custom_points": provider,
            "bonus_method": "provider_until_yahoo_import",
        }

    games = _g(stats, "gp", "games")
    if games <= 0:
        games = 17.0

    points = 0.0

    # Passing
    points += _g(stats, "pass_cmp") * float(scoring.get("pass_completion", 0))
    points += _g(stats, "pass_yd") / float(scoring.get("pass_yards_per_point", 25))
    points += _g(stats, "pass_td") * float(scoring.get("pass_td", 4))
    points += _g(stats, "pass_int", "interceptions") * float(scoring.get("interception", -2))

    # Rushing
    points += _g(stats, "rush_yd") / float(scoring.get("rush_yards_per_point", 10))
    points += _g(stats, "rush_td") * float(scoring.get("rush_td", 6))

    # Receiving
    points += _g(stats, "rec") * float(scoring.get("reception", 1.0))
    points += _g(stats, "rec_yd") / float(scoring.get("rec_yards_per_point", 10))
    points += _g(stats, "rec_td") * float(scoring.get("rec_td", 6))

    # Other offense
    points += _g(stats, "fum_lost") * float(scoring.get("fumble_lost", -2))

    two_pt_total = (
        _g(stats, "pass_2pt") +
        _g(stats, "rush_2pt") +
        _g(stats, "rec_2pt")
    )
    points += two_pt_total * float(scoring.get("two_point_conversion", 2))

    points += _individual_return_tds(stats) * float(scoring.get("return_td", 6))
    points += _g(stats, "fum_rec_td", "off_fum_rec_td") * float(
        scoring.get("offensive_fumble_return_td", 6)
    )

    # Per-game yardage bonus expectation.
    bm = bonus_model or {}
    bonus_points = 0.0

    pass_cfg = bm.get("pass_yards", {})
    bonus_points += expected_game_threshold_bonuses(
        _g(stats, "pass_yd"),
        games,
        scoring.get("pass_yard_bonuses", []),
        pass_cfg.get("sd_floor", 55),
        pass_cfg.get("sd_pct_of_mean", 0.28),
    )

    rush_cfg = bm.get("rush_yards", {})
    bonus_points += expected_game_threshold_bonuses(
        _g(stats, "rush_yd"),
        games,
        scoring.get("rush_yard_bonuses", []),
        rush_cfg.get("sd_floor", 25),
        rush_cfg.get("sd_pct_of_mean", 0.55),
    )

    rec_cfg = bm.get("rec_yards", {})
    bonus_points += expected_game_threshold_bonuses(
        _g(stats, "rec_yd"),
        games,
        scoring.get("rec_yard_bonuses", []),
        rec_cfg.get("sd_floor", 30),
        rec_cfg.get("sd_pct_of_mean", 0.65),
    )

    return {
        "base_points": round(points, 2),
        "bonus_points": round(bonus_points, 2),
        "custom_points": round(points + bonus_points, 2),
        "bonus_method": "normal_yardage_threshold_estimate",
    }
