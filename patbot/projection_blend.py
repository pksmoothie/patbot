from __future__ import annotations

import numpy as np
import pandas as pd


OFFENSE_POSITIONS = {"QB", "RB", "WR", "TE"}


def blend_projection_sources(players: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    """Blend independent full-stat projection sources into production points.

    v0.5.1 uses Sleeper as the continuity anchor and FantasyPros as the new
    independent raw-stat source. Athletic remains a separate custom-ranking/VORP
    input in the consensus layer, so it is intentionally not folded into
    `proj_points` here and therefore is not double-counted.
    """
    out = players.copy()
    pcfg = config.get("projection_sources", {})
    bcfg = pcfg.get("production_blend", {})
    enabled = bool(bcfg.get("enabled", True))

    if "sleeper_proj_points" not in out.columns:
        out["sleeper_proj_points"] = pd.to_numeric(out["proj_points"], errors="coerce")
    else:
        out["sleeper_proj_points"] = pd.to_numeric(out["sleeper_proj_points"], errors="coerce")

    fp = pd.to_numeric(
        out.get("fantasypros_proj_points", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    sleeper = pd.to_numeric(out["sleeper_proj_points"], errors="coerce")

    sleeper_weight = max(0.0, float(bcfg.get("sleeper_weight", 0.60)))
    fp_weight = max(0.0, float(bcfg.get("fantasypros_weight", 0.40)))
    positions = {
        str(x).upper()
        for x in bcfg.get("positions", sorted(OFFENSE_POSITIONS))
    }
    offense_mask = out["pos"].astype(str).str.upper().isin(positions)

    blended = sleeper.copy()
    source_count = pd.Series(1, index=out.index, dtype=int)
    fp_used = offense_mask & fp.notna() & sleeper.notna() & enabled

    total = sleeper_weight + fp_weight
    if total <= 0:
        raise ValueError("projection_sources.production_blend weights must sum to more than zero")

    if fp_used.any():
        sw = sleeper_weight / total
        fw = fp_weight / total
        blended.loc[fp_used] = sleeper.loc[fp_used] * sw + fp.loc[fp_used] * fw
        source_count.loc[fp_used] = 2

    out["projection_blend_points"] = blended.round(2)
    out["projection_blend_source_count"] = source_count
    out["proj_points"] = np.where(offense_mask & enabled, out["projection_blend_points"], sleeper)
    out["proj_points"] = pd.to_numeric(out["proj_points"], errors="coerce").round(2)

    covered_offense = int((offense_mask & fp.notna()).sum())
    total_offense = int(offense_mask.sum())
    status = {
        "production_projection_blend": {
            "ok": True,
            "matched": covered_offense,
            "enabled": enabled,
            "sleeper_weight": round(sleeper_weight / total, 3),
            "fantasypros_weight": round(fp_weight / total, 3),
            "offense_rows": total_offense,
            "coverage_pct": round(100.0 * covered_offense / total_offense, 1) if total_offense else 0.0,
            "note": (
                "Production offense projection is a Sleeper/FantasyPros raw-stat blend. "
                "Athletic remains a separate consensus/VORP input to avoid double counting."
            ),
        }
    }
    return out, status
