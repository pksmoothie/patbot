from __future__ import annotations

from io import StringIO
import os
import re
import unicodedata

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0 Safari/537.36 PatBot/0.3.2"
)

FP_API_BASE = "https://api.fantasypros.com/public/v2/json"
FANTASYDATA_PPR = "https://fantasydata.com/nfl/ppr-rankings"

# Legacy scrape URLs retained only as opt-in fallbacks.
FANTASYPROS_RANKINGS = "https://www.fantasypros.com/nfl/rankings/ppr-cheatsheets.php"
FANTASYPROS_ADP = "https://www.fantasypros.com/nfl/adp/ppr-overall.php?year=2026"


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    value = value.lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    tokens = [t for t in value.split() if t not in {"jr", "sr", "ii", "iii", "iv", "v"}]
    return " ".join(tokens)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": UA,
        "Accept": "application/json,text/html,*/*",
    })
    return s


def _numeric(value):
    if pd.isna(value):
        return np.nan
    m = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(m.group()) if m else np.nan


def _known_name_match(cell: str, known_names: dict[str, str]) -> str | None:
    text = normalize_name(cell)
    if not text:
        return None
    if text in known_names:
        return known_names[text]
    matches = [
        (norm, original)
        for norm, original in known_names.items()
        if norm and norm in text
    ]
    if not matches:
        return None
    matches.sort(key=lambda x: len(x[0]), reverse=True)
    return matches[0][1]


def _fp_api_key() -> str | None:
    key = (os.getenv("FANTASYPROS_API_KEY") or "").strip()
    return key or None


def _fp_get(path: str, params: dict) -> dict:
    key = _fp_api_key()
    if not key:
        raise RuntimeError(
            "FANTASYPROS_API_KEY is not set. Put your FantasyPros API key in the local .env file."
        )
    s = _session()
    r = s.get(
        f"{FP_API_BASE}/{path.lstrip('/')}",
        params=params,
        headers={"x-api-key": key},
        timeout=45,
    )
    if r.status_code in {401, 403}:
        raise RuntimeError(
            f"FantasyPros API rejected the key (HTTP {r.status_code}). "
            "Check FANTASYPROS_API_KEY in .env."
        )
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        raise ValueError("FantasyPros API returned an unexpected response.")
    return data


def _api_player_name(player: dict) -> str:
    return str(
        player.get("player_name")
        or player.get("name")
        or player.get("full_name")
        or ""
    ).strip()


def _api_rank(player: dict, candidates: list[str]) -> float:
    for key in candidates:
        if key in player and player[key] is not None:
            value = _numeric(player[key])
            if not np.isnan(value):
                return value
    return np.nan


def fetch_fantasypros_api_ecr(
    player_names: list[str],
    season: int = 2026,
) -> pd.DataFrame:
    data = _fp_get(
        f"nfl/{season}/consensus-rankings",
        {"position": "ALL", "scoring": "PPR", "week": 0},
    )
    players = data.get("players") or []
    known = {normalize_name(x): x for x in player_names}
    rows = []
    for p in players:
        raw_name = _api_player_name(p)
        name = _known_name_match(raw_name, known)
        rank = _api_rank(p, ["rank_ecr", "rank", "rank_ave"])
        if name and not np.isnan(rank) and rank > 0:
            rows.append({"name": name, "fp_ecr": rank})
    if not rows:
        raise ValueError("FantasyPros API ECR returned no matched PatBot players.")
    return pd.DataFrame(rows).drop_duplicates("name", keep="first")


def fetch_fantasypros_api_adp(
    player_names: list[str],
    season: int = 2026,
) -> pd.DataFrame:
    data = _fp_get(
        f"nfl/{season}/consensus-rankings",
        {"position": "ALL", "scoring": "PPR", "week": 0, "type": "ADP"},
    )
    players = data.get("players") or []
    known = {normalize_name(x): x for x in player_names}
    rows = []
    for p in players:
        raw_name = _api_player_name(p)
        name = _known_name_match(raw_name, known)
        rank = _api_rank(p, ["rank_adp_ppr", "rank_adp", "rank_ecr", "rank"])
        if name and not np.isnan(rank) and rank > 0:
            rows.append({"name": name, "fp_adp": rank})
    if not rows:
        raise ValueError("FantasyPros API ADP returned no matched PatBot players.")
    return pd.DataFrame(rows).drop_duplicates("name", keep="first")


def _read_tables(url: str) -> list[pd.DataFrame]:
    r = _session().get(url, timeout=40)
    r.raise_for_status()
    return pd.read_html(StringIO(r.text))


def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            " ".join(str(x) for x in col if str(x) != "nan" and "Unnamed" not in str(x)).strip()
            for col in out.columns
        ]
    else:
        out.columns = [str(c).strip() for c in out.columns]
    return out


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in df.columns:
        lc = str(c).lower()
        if any(candidate.lower() in lc for candidate in candidates):
            return c
    return None


def _best_player_table(tables: list[pd.DataFrame], known: dict[str, str]):
    best = None
    best_col = None
    best_matches = -1
    for raw in tables:
        df = _flatten_cols(raw)
        for col in list(df.columns):
            matches = sum(
                1 for value in df[col].head(300).astype(str)
                if _known_name_match(value, known)
            )
            if matches > best_matches:
                best = df
                best_col = col
                best_matches = matches
    if best is None or best_col is None or best_matches < 3:
        raise ValueError("Could not identify a player rankings table.")
    return best, best_col


def _rank_column(df: pd.DataFrame, player_col: str) -> str | None:
    direct = _find_col(df, ["rank", "rk"])
    if direct is not None and direct != player_col:
        return direct
    for c in df.columns:
        if c == player_col:
            continue
        vals = df[c].head(150).map(_numeric)
        if vals.notna().sum() >= 5:
            return c
    return None


def fetch_fantasydata_rankings(player_names: list[str]) -> pd.DataFrame:
    tables = _read_tables(FANTASYDATA_PPR)
    known = {normalize_name(x): x for x in player_names}
    df, player_col = _best_player_table(tables, known)
    rank_col = _rank_column(df, player_col)
    if rank_col is None:
        raise ValueError("FantasyData ranking column not found.")
    rows = []
    for _, row in df.iterrows():
        name = _known_name_match(row[player_col], known)
        rank = _numeric(row[rank_col])
        if name and not np.isnan(rank) and rank > 0:
            rows.append({"name": name, "fd_rank": rank})
    if not rows:
        raise ValueError("FantasyData rankings parsed but no players matched.")
    return pd.DataFrame(rows).drop_duplicates("name", keep="first")


def fetch_fantasypros_rankings(player_names: list[str]) -> pd.DataFrame:
    tables = _read_tables(FANTASYPROS_RANKINGS)
    known = {normalize_name(x): x for x in player_names}
    df, player_col = _best_player_table(tables, known)
    rank_col = _rank_column(df, player_col)
    rows = []
    for _, row in df.iterrows():
        name = _known_name_match(row[player_col], known)
        rank = _numeric(row[rank_col]) if rank_col else np.nan
        if name and not np.isnan(rank):
            rows.append({"name": name, "fp_ecr": rank})
    if not rows:
        raise ValueError("FantasyPros scrape returned no matched rankings.")
    return pd.DataFrame(rows).drop_duplicates("name", keep="first")


def fetch_fantasypros_adp(player_names: list[str]) -> pd.DataFrame:
    tables = _read_tables(FANTASYPROS_ADP)
    known = {normalize_name(x): x for x in player_names}
    df, player_col = _best_player_table(tables, known)
    avg_col = _find_col(df, ["avg", "average"])
    rank_col = _rank_column(df, player_col)
    rows = []
    for _, row in df.iterrows():
        name = _known_name_match(row[player_col], known)
        value = _numeric(row[avg_col]) if avg_col else np.nan
        if np.isnan(value) and rank_col:
            value = _numeric(row[rank_col])
        if name and not np.isnan(value):
            rows.append({"name": name, "fp_adp": value})
    if not rows:
        raise ValueError("FantasyPros scrape returned no matched ADP.")
    return pd.DataFrame(rows).drop_duplicates("name", keep="first")


def augment_market_sources(players: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    out = players.copy()
    names = out["name"].dropna().astype(str).tolist()
    source_cfg = config.get("v03_consensus", {})
    season = int(config.get("league", {}).get("season", 2026))
    status = {}
    loaders = []

    if source_cfg.get("fantasypros_api", True):
        if _fp_api_key():
            loaders.extend([
                ("fantasypros_api_ecr", lambda n: fetch_fantasypros_api_ecr(n, season)),
                ("fantasypros_api_adp", lambda n: fetch_fantasypros_api_adp(n, season)),
            ])
        else:
            status["fantasypros_api"] = {"ok": False, "error": "No FANTASYPROS_API_KEY in .env (official API preferred)."}

    if source_cfg.get("fantasypros_rankings", False):
        loaders.append(("fantasypros_scrape_ecr", fetch_fantasypros_rankings))
    if source_cfg.get("fantasypros_adp", False):
        loaders.append(("fantasypros_scrape_adp", fetch_fantasypros_adp))
    if source_cfg.get("fantasydata_rankings", True):
        loaders.append(("fantasydata_rank", fetch_fantasydata_rankings))

    for label, loader in loaders:
        try:
            extra = loader(names)
            out = out.merge(extra, on="name", how="left")
            status[label] = {"ok": True, "matched": int(extra["name"].nunique())}
        except Exception as exc:
            status[label] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    expert_cols = [c for c in ["fp_ecr", "fd_rank"] if c in out.columns]
    if expert_cols:
        out["expert_rank"] = out[expert_cols].mean(axis=1, skipna=True)
    else:
        out["expert_rank"] = np.nan

    if "fp_adp" in out.columns:
        out["market_adp"] = out["fp_adp"].fillna(out["adp"])
    else:
        out["market_adp"] = out["adp"]

    out["sleeper_adp"] = out["adp"]
    out["adp"] = out["market_adp"]
    return out, status
