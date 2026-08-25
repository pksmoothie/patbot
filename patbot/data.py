from pathlib import Path
import pandas as pd

REQUIRED = {"player_id", "name", "team", "pos", "adp", "proj_points"}


def load_players(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["player_id"] = df["player_id"].astype(str)
    df["pos"] = df["pos"].astype(str).str.upper()
    df["adp"] = pd.to_numeric(df["adp"], errors="coerce")
    df["proj_points"] = pd.to_numeric(df["proj_points"], errors="coerce")

    if "injury_risk" not in df:
        df["injury_risk"] = 0.0
    df["injury_risk"] = pd.to_numeric(df["injury_risk"], errors="coerce").fillna(0.0)

    numeric_cols = [
        "tier", "bye", "base_custom_points", "estimated_bonus_points",
        "provider_ppr", "games_projected", "fp_ecr", "fp_adp",
        "fd_rank", "expert_rank", "market_adp", "sleeper_adp",
        "consensus_value", "consensus_tier", "static_vorp",
        "fp_age", "history_seasons_observed", "history_weighted_games",
        "history_missed_rate", "current_play_probability", "age_tail_bonus",
        "catastrophic_miss_probability", "minor_miss_lambda",
        "off_field_miss_probability", "off_field_max_missed_games",
        "risk_score", "sleeper_current_injury_risk",
    ]
    for col in numeric_cols:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "tier_cliff" in df:
        df["tier_cliff"] = df["tier_cliff"].astype(str).str.lower().isin(["true", "1", "yes"])

    return df.dropna(subset=["adp", "proj_points"])
