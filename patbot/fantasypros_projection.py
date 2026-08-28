from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .market import _fp_get, _known_name_match, normalize_name
from .scoring import score_season_projection


OFFENSE_POSITIONS = ("QB", "RB", "WR", "TE")


def _safe_float(value, default=np.nan) -> float:
    try:
        if value is None or pd.isna(value):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _first_numeric(stats: dict, *keys: str, default=0.0) -> float:
    for key in keys:
        if key not in stats:
            continue
        value = _safe_float(stats.get(key))
        if np.isfinite(value):
            return float(value)
    return float(default)


def _stats_dict(player: dict) -> dict:
    value = player.get("stats")
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                return item
    return {}


def fantasypros_stats_to_patbot(stats: dict) -> dict:
    """Normalize FantasyPros projection fields into PatBot's scorer schema.

    FantasyPros' projection payload uses plural field names for many stats while
    Sleeper uses singular/abbreviated names. Keep the mapper permissive so small
    upstream naming changes do not silently zero out an entire category.
    """
    out = {
        "gp": _first_numeric(stats, "games", "game", "g", "gp", default=17.0),
        "pass_cmp": _first_numeric(stats, "pass_cmp", "pass_completions", "cmp"),
        "pass_yd": _first_numeric(stats, "pass_yds", "pass_yd", "pass_yards"),
        "pass_td": _first_numeric(stats, "pass_tds", "pass_td", "passing_tds"),
        "pass_int": _first_numeric(stats, "pass_ints", "pass_int", "interceptions", "ints"),
        "rush_yd": _first_numeric(stats, "rush_yds", "rush_yd", "rushing_yds", "rush_yards"),
        "rush_td": _first_numeric(stats, "rush_tds", "rush_td", "rushing_tds"),
        "rec": _first_numeric(stats, "rec", "receptions", "recs"),
        "rec_yd": _first_numeric(stats, "rec_yds", "rec_yd", "receiving_yds", "rec_yards"),
        "rec_td": _first_numeric(stats, "rec_tds", "rec_td", "receiving_tds"),
        "fum_lost": _first_numeric(stats, "fumbles_lost", "fum_lost", "lost_fumbles"),
        "st_td": _first_numeric(stats, "ret_tds", "return_tds", "st_td"),
        "fum_rec_td": _first_numeric(stats, "fum_rec_tds", "fum_rec_td", "off_fum_rec_td"),
    }

    # FantasyPros commonly exposes a single aggregate 2-point conversion field.
    # PatBot's scorer sums pass/rush/receive 2PTs, so place the aggregate in one
    # bucket only to avoid double counting.
    two_pt = _first_numeric(stats, "2pt_tds", "two_pt_tds", "two_point_conversions")
    out["rush_2pt"] = two_pt
    out["pass_2pt"] = 0.0
    out["rec_2pt"] = 0.0
    return out


def fetch_fantasypros_preseason_projections(
    player_names: list[str],
    config: dict,
) -> tuple[pd.DataFrame, dict]:
    """Fetch FantasyPros preseason projections and score them under league rules."""
    season = int(config.get("league", {}).get("season", 2026))
    pcfg = config.get("projection_sources", {})
    positions = tuple(str(x).upper() for x in pcfg.get("fantasypros_positions", OFFENSE_POSITIONS))
    spacing = float(pcfg.get("fantasypros_request_spacing_seconds", 1.05))
    scoring = config.get("scoring", {})
    bonus_model = config.get("bonus_model", {})
    known = {normalize_name(x): x for x in player_names}

    rows: list[dict] = []
    by_position: dict[str, dict] = {}

    for i, pos in enumerate(positions):
        if i and spacing > 0:
            time.sleep(spacing)
        try:
            data = _fp_get(
                f"nfl/{season}/projections",
                {"position": pos, "week": 0},
            )
            players = data.get("players") or []
            matched = 0
            for player in players:
                raw_name = str(player.get("name") or player.get("player_name") or "").strip()
                name = _known_name_match(raw_name, known)
                if not name:
                    continue
                stats = _stats_dict(player)
                if not stats:
                    continue
                normalized = fantasypros_stats_to_patbot(stats)
                scored = score_season_projection(
                    normalized,
                    scoring=scoring,
                    bonus_model=bonus_model,
                    position=pos,
                )
                provider_ppr = _first_numeric(stats, "points_ppr", "pts_ppr", "ppr_points", default=np.nan)
                rows.append(
                    {
                        "name": name,
                        "fantasypros_proj_points": float(scored["custom_points"]),
                        "fantasypros_proj_base_points": float(scored["base_points"]),
                        "fantasypros_proj_bonus_points": float(scored["bonus_points"]),
                        "fantasypros_proj_provider_ppr": provider_ppr,
                        "fantasypros_proj_games": float(normalized.get("gp", 17.0)),
                    }
                )
                matched += 1
            by_position[pos] = {
                "ok": matched > 0,
                "matched": int(matched),
                "returned": int(len(players)),
                "endpoint": f"nfl/{season}/projections?position={pos}&week=0",
            }
        except Exception as exc:
            by_position[pos] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    if not rows:
        details = "; ".join(
            f"{pos}: {status.get('error', 'no matches')}"
            for pos, status in by_position.items()
        )
        raise RuntimeError(f"FantasyPros preseason projections returned no matched players. {details}")

    frame = (
        pd.DataFrame(rows)
        .sort_values(["name", "fantasypros_proj_points"], ascending=[True, False])
        .drop_duplicates("name", keep="first")
        .reset_index(drop=True)
    )
    status = {
        "fantasypros_preseason_projections": {
            "ok": True,
            "matched": int(frame["name"].nunique()),
            "endpoint": f"nfl/{season}/projections week=0",
            "note": "Raw FantasyPros stat projections rescored under PatBot league rules; diagnostic source only in v0.5.0.",
        }
    }
    for pos, item in by_position.items():
        status[f"fantasypros_projection_{pos.lower()}"] = item
    return frame, status


def augment_fantasypros_projections(players: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    out = players.copy()
    names = out["name"].dropna().astype(str).tolist()
    frame, status = fetch_fantasypros_preseason_projections(names, config)
    out = out.merge(frame, on="name", how="left")
    return out, status
