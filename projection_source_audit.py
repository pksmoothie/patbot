from __future__ import annotations

import numpy as np
import pandas as pd

from patbot.config import load_config
from patbot.data import load_players
from patbot.draft import DraftEngine
from patbot.draft_state import make_pick_record
from patbot.sim import FastDraftSimulator


OFFENSE = {"QB", "RB", "WR", "TE"}
TOP_CANDIDATES = [
    "Ja'Marr Chase",
    "Puka Nacua",
    "Jaxon Smith-Njigba",
    "Amon-Ra St. Brown",
    "Christian McCaffrey",
    "Jonathan Taylor",
]


def _row_by_name(players: pd.DataFrame, name: str) -> pd.Series:
    match = players[players["name"].eq(name)]
    if match.empty:
        raise RuntimeError(f"Could not find {name!r} in the live player snapshot.")
    return match.iloc[0]


def _source_vorp(players: pd.DataFrame, points_col: str, cfg: dict) -> pd.Series:
    points = pd.to_numeric(players[points_col], errors="coerce")
    out = pd.Series(np.nan, index=players.index, dtype=float)
    for pos, replacement_rank in cfg["draft_engine"]["replacement_rank"].items():
        if pos not in OFFENSE:
            continue
        mask = players["pos"].eq(pos) & points.notna()
        group = points[mask].sort_values(ascending=False)
        if group.empty:
            continue
        idx = min(int(replacement_rank) - 1, len(group) - 1)
        replacement = float(group.iloc[idx])
        out.loc[mask] = points.loc[mask] - replacement
    return out


def _rank_high(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.rank(method="min", ascending=False)


def _build_source_table(players: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = players[["name", "pos"]].copy()
    out["Sleeper Pts"] = pd.to_numeric(players["proj_points"], errors="coerce")
    out["Sleeper VORP"] = _source_vorp(players, "proj_points", cfg)
    out["Sleeper Rank"] = _rank_high(out["Sleeper VORP"])

    if "fantasypros_proj_points" in players:
        out["FP Pts"] = pd.to_numeric(players["fantasypros_proj_points"], errors="coerce")
        tmp = players.copy()
        tmp["_fp_points"] = out["FP Pts"]
        out["FP VORP"] = _source_vorp(tmp, "_fp_points", cfg)
        out["FP Rank"] = _rank_high(out["FP VORP"])
    else:
        out["FP Pts"] = np.nan
        out["FP VORP"] = np.nan
        out["FP Rank"] = np.nan

    out["Athletic Pts"] = pd.to_numeric(
        players.get("athletic_points", pd.Series(np.nan, index=players.index)),
        errors="coerce",
    )
    out["Athletic VORP"] = pd.to_numeric(
        players.get("athletic_vorp", pd.Series(np.nan, index=players.index)),
        errors="coerce",
    )
    if "athletic_rank" in players:
        out["Athletic Rank"] = pd.to_numeric(players["athletic_rank"], errors="coerce")
    else:
        out["Athletic Rank"] = _rank_high(out["Athletic VORP"])

    rank_cols = ["Sleeper Rank", "FP Rank", "Athletic Rank"]
    out["Sources"] = out[rank_cols].notna().sum(axis=1)
    out["Best Rank"] = out[rank_cols].min(axis=1, skipna=True)
    out["Worst Rank"] = out[rank_cols].max(axis=1, skipna=True)
    out["Rank Spread"] = out["Worst Rank"] - out["Best Rank"]
    out["Avg Rank"] = out[rank_cols].mean(axis=1, skipna=True)
    return out


def _variant_players(players: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, float]:
    out = players.copy()
    base = pd.to_numeric(out["proj_points"], errors="coerce")
    coverage = 1.0

    if variant == "Production":
        return out, coverage
    if variant == "Sleeper":
        source = base
    elif variant == "FantasyPros":
        source = pd.to_numeric(out.get("fantasypros_proj_points"), errors="coerce")
    elif variant == "Athletic":
        source = pd.to_numeric(out.get("athletic_points"), errors="coerce")
    elif variant == "Equal Blend":
        source_frame = pd.DataFrame(
            {
                "Sleeper": base,
                "FantasyPros": pd.to_numeric(out.get("fantasypros_proj_points"), errors="coerce"),
                "Athletic": pd.to_numeric(out.get("athletic_points"), errors="coerce"),
            }
        )
        source = source_frame.mean(axis=1, skipna=True)
    else:
        raise ValueError(f"Unknown projection variant: {variant}")

    if variant not in {"Production", "Sleeper", "Equal Blend"}:
        offense = out["pos"].isin(OFFENSE)
        coverage = float(source[offense].notna().mean()) if offense.any() else 0.0
    elif variant == "Equal Blend":
        offense = out["pos"].isin(OFFENSE)
        multi = pd.DataFrame(
            {
                "Sleeper": base,
                "FantasyPros": pd.to_numeric(out.get("fantasypros_proj_points"), errors="coerce"),
                "Athletic": pd.to_numeric(out.get("athletic_points"), errors="coerce"),
            }
        ).notna().sum(axis=1)
        coverage = float((multi[offense] >= 2).mean()) if offense.any() else 0.0

    out["proj_points"] = source.where(source.notna(), base)

    # Source-ablation means the tested projection should not also receive an
    # Athletic-specific expert-rank boost. Keep the independent ECR/FantasyData
    # ranking signal where available, while production retains its current setup.
    if "generic_expert_rank" in out:
        out["expert_rank"] = pd.to_numeric(out["generic_expert_rank"], errors="coerce")
    return out, coverage


def _spearman_without_scipy(left: pd.Series, right: pd.Series) -> float:
    """Spearman rank correlation using pandas' built-in Pearson path only.

    pandas delegates method='spearman' to scipy. PatBot intentionally keeps its
    runtime lightweight, so rank each series first and then calculate ordinary
    Pearson correlation on the ranks. That is exactly Spearman correlation and
    requires no scipy dependency.
    """
    left_rank = pd.to_numeric(left, errors="coerce").rank(method="average")
    right_rank = pd.to_numeric(right, errors="coerce").rank(method="average")
    return float(left_rank.corr(right_rank))


def _projection_correlations(source_table: pd.DataFrame) -> pd.DataFrame:
    rank_cols = ["Sleeper Rank", "FP Rank", "Athletic Rank"]
    offense = source_table[source_table["pos"].isin(OFFENSE)].copy()
    rows = []
    for i, left in enumerate(rank_cols):
        for right in rank_cols[i + 1 :]:
            pair = offense[[left, right]].dropna()
            if len(pair) < 3:
                corr = np.nan
            else:
                corr = _spearman_without_scipy(pair[left], pair[right])
            rows.append(
                {
                    "Source A": left.replace(" Rank", ""),
                    "Source B": right.replace(" Rank", ""),
                    "Matched": int(len(pair)),
                    "Spearman Rank Corr": round(corr, 3) if np.isfinite(corr) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _run_expected_103_ablation(players: pd.DataFrame, cfg: dict, runs: int = 250) -> pd.DataFrame:
    teams = int(cfg["league"]["teams"])
    gibbs = _row_by_name(players, "Jahmyr Gibbs")
    bijan = _row_by_name(players, "Bijan Robinson")
    opening_history = [
        make_pick_record(1, teams, str(gibbs["player_id"]), gibbs["name"], gibbs["team"], gibbs["pos"]),
        make_pick_record(2, teams, str(bijan["player_id"]), bijan["name"], bijan["team"], bijan["pos"]),
    ]
    drafted = {str(gibbs["player_id"]), str(bijan["player_id"])}
    rows = []

    for variant in ("Production", "Sleeper", "FantasyPros", "Athletic", "Equal Blend"):
        variant_players, coverage = _variant_players(players, variant)
        engine = DraftEngine(variant_players, cfg)
        sim = FastDraftSimulator(engine)
        for name in TOP_CANDIDATES:
            candidate = _row_by_name(variant_players, name)
            if str(candidate["player_id"]) in drafted:
                continue
            result = sim.simulate_candidate(
                current_pick=3,
                drafted_ids=drafted,
                my_roster_ids=[],
                candidate_id=str(candidate["player_id"]),
                runs=int(runs),
                through_round=8,
                seed=int(cfg.get("simulation", {}).get("comparison_seed", 20260818)) + 500,
                draft_history=opening_history,
            )
            rows.append(
                {
                    "Variant": variant,
                    "Coverage": round(100.0 * coverage, 1),
                    "Candidate": name,
                    "Avg Lineup": result["avg_lineup_score"],
                    "P10": result["p10_lineup_score"],
                    "P90": result["p90_lineup_score"],
                    "Starter VORP": result["avg_starter_vorp"],
                    "LWS": result["candidate_league_winner_score"],
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["Rank in Variant"] = frame.groupby("Variant")["Avg Lineup"].rank(
            method="min", ascending=False
        ).astype(int)
        frame = frame.sort_values(["Variant", "Rank in Variant", "Candidate"])
    return frame


def main():
    cfg = load_config()
    players = load_players("data/players_2026_live.csv")
    players["player_id"] = players["player_id"].astype(str)

    if "fantasypros_proj_points" not in players.columns or players["fantasypros_proj_points"].notna().sum() == 0:
        raise RuntimeError(
            "FantasyPros projection columns are missing. Run .\\UPDATE_AND_RUN.bat first so v0.5.0 refreshes the local snapshot."
        )

    print("\nPatBot v0.5.0 projection-source diversification audit")
    print("DIAGNOSTIC ONLY: production projection weights are unchanged.")
    print("Sources: Sleeper custom-scored stats, FantasyPros custom-scored stats, Athletic custom projections/VORP.\n")

    source_table = _build_source_table(players, cfg)
    offense = source_table[source_table["pos"].isin(OFFENSE)].copy()

    print("=== SOURCE COVERAGE ===\n")
    coverage = pd.DataFrame(
        [
            {"Source": "Sleeper", "Matched offense": int(offense["Sleeper Pts"].notna().sum())},
            {"Source": "FantasyPros", "Matched offense": int(offense["FP Pts"].notna().sum())},
            {"Source": "Athletic", "Matched offense": int(offense["Athletic Pts"].notna().sum())},
        ]
    )
    print(coverage.to_string(index=False))

    print("\n=== SOURCE RANK CORRELATIONS ===\n")
    print(_projection_correlations(source_table).to_string(index=False))

    print("\n=== TOP 40 BY AVERAGE SOURCE RANK ===\n")
    top = offense[offense["Sources"] >= 2].sort_values(["Avg Rank", "Rank Spread"]).head(40)
    cols = [
        "name", "pos", "Sleeper Pts", "Sleeper VORP", "Sleeper Rank",
        "FP Pts", "FP VORP", "FP Rank", "Athletic Pts", "Athletic VORP",
        "Athletic Rank", "Rank Spread", "Avg Rank",
    ]
    print(top[cols].round(2).to_string(index=False))

    print("\n=== BIGGEST SOURCE DISAGREEMENTS AMONG RELEVANT PLAYERS ===\n")
    disagreements = offense[(offense["Sources"] >= 2) & (offense["Best Rank"] <= 120)].copy()
    disagreements = disagreements.sort_values(["Rank Spread", "Best Rank"], ascending=[False, True]).head(50)
    print(disagreements[cols].round(2).to_string(index=False))

    print("\n=== EXPECTED GIBBS/BIJAN 1.03 SOURCE ABLATION (250 PAIRED RUNS EACH) ===\n")
    ablation = _run_expected_103_ablation(players, cfg, runs=250)
    print(ablation.to_string(index=False))

    print("\nHow to read this:")
    print("- Production is today's PatBot model; the other rows replace only the base projection source for the diagnostic.")
    print("- FantasyPros and Athletic variants fall back to Sleeper for players that source does not cover; Coverage shows the offensive share directly covered.")
    print("- Equal Blend averages every available projection among Sleeper/FantasyPros/Athletic and is diagnostic, not a proposed final weight.")
    print("- Source-specific variants use generic ECR/FantasyData expert rank rather than Athletic's rank boost so Athletic is not counted twice.")
    print("- We are looking for robustness and meaningful disagreement before choosing any new production blend.\n")


if __name__ == "__main__":
    main()
