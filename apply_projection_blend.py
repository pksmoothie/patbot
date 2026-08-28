from __future__ import annotations

from pathlib import Path

import pandas as pd

from patbot.config import load_config
from patbot.consensus import add_consensus_values
from patbot.projection_blend import blend_projection_sources


SNAPSHOT = Path("data/players_2026_live.csv")


def _replacement_vorp(frame: pd.DataFrame, points_col: str, cfg: dict) -> pd.Series:
    points = pd.to_numeric(frame[points_col], errors="coerce")
    out = pd.Series(float("nan"), index=frame.index, dtype=float)
    for pos, rank in cfg["draft_engine"]["replacement_rank"].items():
        mask = frame["pos"].eq(pos) & points.notna()
        ordered = points[mask].sort_values(ascending=False)
        if ordered.empty:
            continue
        replacement = float(ordered.iloc[min(int(rank) - 1, len(ordered) - 1)])
        out.loc[mask] = points.loc[mask] - replacement
    return out


def main():
    cfg = load_config()
    if not SNAPSHOT.exists():
        raise SystemExit("No local player snapshot found. Run .\\UPDATE_AND_RUN.bat first.")

    frame = pd.read_csv(SNAPSHOT)
    if "fantasypros_proj_points" not in frame or frame["fantasypros_proj_points"].notna().sum() == 0:
        raise SystemExit(
            "The local snapshot does not contain FantasyPros projections. "
            "Run .\\UPDATE_AND_RUN.bat once to refresh them."
        )

    before = pd.to_numeric(frame.get("sleeper_proj_points", frame["proj_points"]), errors="coerce")
    frame["sleeper_proj_points"] = before
    frame["_old_vorp"] = _replacement_vorp(frame, "sleeper_proj_points", cfg)
    frame["_old_rank"] = frame["_old_vorp"].rank(method="min", ascending=False)

    frame, status = blend_projection_sources(frame, cfg)
    frame = add_consensus_values(frame, cfg)
    frame["_new_vorp"] = _replacement_vorp(frame, "proj_points", cfg)
    frame["_new_rank"] = frame["_new_vorp"].rank(method="min", ascending=False)
    frame["_rank_change"] = frame["_old_rank"] - frame["_new_rank"]

    save_cols = [c for c in frame.columns if not c.startswith("_old_") and not c.startswith("_new_") and c != "_rank_change"]
    frame[save_cols].to_csv(SNAPSHOT, index=False)

    blend = status["production_projection_blend"]
    print("\nPatBot v0.5.1 production projection blend applied locally.")
    print(
        f"Sleeper {blend['sleeper_weight']:.0%} / FantasyPros {blend['fantasypros_weight']:.0%}; "
        f"FantasyPros coverage {blend['coverage_pct']}% of offense."
    )
    print("Athletic remains a separate custom VORP/ranking input; it is not double-counted in proj_points.\n")

    relevant = frame[frame["pos"].isin(["QB", "RB", "WR", "TE"])].copy()
    top = relevant.sort_values("_new_vorp", ascending=False).head(30)
    print("=== TOP 30 AFTER PRODUCTION BLEND ===\n")
    print(
        top[["name", "pos", "sleeper_proj_points", "fantasypros_proj_points", "proj_points", "_new_vorp", "_new_rank"]]
        .rename(columns={"_new_vorp": "VORP", "_new_rank": "Rank"})
        .round(2)
        .to_string(index=False)
    )

    movers = relevant[relevant["_old_rank"] <= 150].copy()
    movers["Abs Change"] = movers["_rank_change"].abs()
    movers = movers.sort_values(["Abs Change", "_new_rank"], ascending=[False, True]).head(25)
    print("\n=== BIGGEST TOP-150 RANK MOVERS VS SLEEPER-ONLY BASE ===\n")
    print(
        movers[["name", "pos", "_old_rank", "_new_rank", "_rank_change", "sleeper_proj_points", "fantasypros_proj_points", "proj_points"]]
        .rename(columns={"_old_rank": "Old Rank", "_new_rank": "New Rank", "_rank_change": "Ranks Gained"})
        .round(2)
        .to_string(index=False)
    )
    print("\nPositive 'Ranks Gained' means the player moved up under the blended production projection.\n")


if __name__ == "__main__":
    main()
