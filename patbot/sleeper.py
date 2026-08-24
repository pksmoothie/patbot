from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import time

import pandas as pd
import requests

from .scoring import score_season_projection


POSITIONS = ["QB", "RB", "WR", "TE", "K", "DEF"]
PROJ_HOST = "https://api.sleeper.com"
APP_HOST = "https://api.sleeper.app"

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0 Safari/537.36 PatBot/0.3.7"
)

FALLBACK_BYES_2026 = {
    "CAR": 5, "KC": 5,
    "CIN": 6, "DET": 6, "MIA": 6, "MIN": 6,
    "BUF": 7, "JAX": 7, "LAC": 7, "WAS": 7,
    "HOU": 8, "NO": 8, "NYG": 8, "SF": 8,
    "PIT": 9, "TEN": 9,
    "CHI": 10, "DEN": 10, "PHI": 10, "TB": 10,
    "ATL": 11, "CLE": 11, "GB": 11, "LAR": 11, "NE": 11, "SEA": 11,
    "BAL": 13, "IND": 13, "LV": 13, "NYJ": 13,
    "ARI": 14, "DAL": 14,
}


class SleeperDataError(RuntimeError):
    pass


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": UA,
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://sleeper.com/",
    })
    return s


def _get_json(session: requests.Session, url: str, params: list[tuple] | dict | None = None):
    r = session.get(url, params=params, timeout=45)
    if r.status_code == 403:
        raise SleeperDataError(
            "Sleeper returned HTTP 403. Try again in a minute; PatBot already sends "
            "a browser-like User-Agent because Sleeper blocks some default clients."
        )
    r.raise_for_status()
    return r.json()


def _player_id(record: dict) -> str:
    player = record.get("player") or {}
    value = (
        record.get("player_id")
        or record.get("id")
        or player.get("player_id")
        or player.get("id")
    )
    return "" if value is None else str(value)


def _player_meta(record: dict) -> dict:
    player = record.get("player") or {}
    first = player.get("first_name") or ""
    last = player.get("last_name") or ""
    full_name = (
        player.get("full_name")
        or player.get("name")
        or record.get("name")
        or f"{first} {last}".strip()
    )
    pos = player.get("position") or record.get("position") or record.get("pos") or ""
    team = player.get("team") or record.get("team") or record.get("nfl_team") or ""

    years_exp = player.get("years_exp")
    if years_exp is None:
        years_exp = record.get("years_exp")

    return {
        "name": str(full_name).strip(),
        "pos": str(pos).upper().strip(),
        "team": str(team).upper().strip(),
        "injury_status": player.get("injury_status") or record.get("injury_status"),
        "active": player.get("active", record.get("active")),
        "years_exp": years_exp,
    }


def _stats(record: dict) -> dict:
    value = record.get("stats")
    return value if isinstance(value, dict) else {}


def _projection_url(season: int, week: int | None = None) -> str:
    suffix = f"/{week}" if week is not None else ""
    return f"{PROJ_HOST}/projections/nfl/{season}{suffix}"


def fetch_position_projections(session, season, position, week=None, order_by="pts_ppr") -> list[dict]:
    params = [
        ("season_type", "regular"),
        ("position[]", position),
        ("order_by", order_by),
    ]
    data = _get_json(session, _projection_url(season, week), params=params)
    if not isinstance(data, list):
        raise SleeperDataError(
            f"Unexpected Sleeper projection response for {position}: {type(data).__name__}"
        )
    return data


def derive_byes(session: requests.Session, season: int) -> dict[str, int]:
    url = f"{APP_HOST}/schedule/nfl/regular/{season}"
    try:
        games = _get_json(session, url)
        if not isinstance(games, list):
            raise ValueError("schedule was not a list")

        all_teams = {
            "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
            "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
            "LV", "LAC", "LAR", "MIA", "MIN", "NE", "NO", "NYG",
            "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
        }

        seen_by_week = {}
        for game in games:
            week = int(game.get("week") or game.get("leg") or 0)
            if not week:
                continue
            seen_by_week.setdefault(week, set())
            home = game.get("home") or game.get("home_team")
            away = game.get("away") or game.get("away_team")
            if home:
                seen_by_week[week].add(str(home).upper())
            if away:
                seen_by_week[week].add(str(away).upper())

        byes = {}
        for week, playing in seen_by_week.items():
            if 1 <= week <= 18:
                for team in all_teams - playing:
                    byes.setdefault(team, week)

        if len(byes) == 32:
            return byes
    except Exception:
        pass

    if season == 2026:
        return FALLBACK_BYES_2026.copy()
    return {}


def _adp_map(adp_records: list[dict]) -> dict[str, float]:
    out = {}
    for record in adp_records:
        pid = _player_id(record)
        if not pid:
            continue

        stats = _stats(record)
        adp = stats.get("adp_dd_ppr")
        try:
            adp = float(adp)
        except (TypeError, ValueError):
            continue

        if 0 < adp < 999:
            out[pid] = adp
    return out


def _injury_risk(meta: dict) -> float:
    status = str(meta.get("injury_status") or "").lower()
    if not status:
        return 0.0
    if "out" in status or "ir" in status or "pup" in status:
        return 0.30
    if "doubt" in status:
        return 0.20
    if "question" in status:
        return 0.10
    return 0.05


def _experience_fields(meta: dict) -> tuple[float | None, bool]:
    raw = meta.get("years_exp")
    try:
        years_exp = float(raw)
    except (TypeError, ValueError):
        return None, False
    return years_exp, years_exp == 0.0


def build_live_dataframe(config: dict) -> tuple[pd.DataFrame, dict]:
    season = int(config["league"]["season"])
    scoring = config["scoring"]
    bonus_model = config.get("bonus_model", {})
    session = _session()
    byes = derive_byes(session, season)

    rows = []
    total_source_records = 0

    for position in POSITIONS:
        season_records = fetch_position_projections(
            session,
            season,
            position,
            week=None,
            order_by="pts_ppr",
        )
        total_source_records += len(season_records)

        adp_records = fetch_position_projections(
            session,
            season,
            position,
            week=1,
            order_by="adp_dd_ppr",
        )
        adp_lookup = _adp_map(adp_records)

        for record in season_records:
            pid = _player_id(record)
            meta = _player_meta(record)
            stats = _stats(record)

            if not pid:
                continue

            pos = meta["pos"] or position
            if pos == "DST":
                pos = "DEF"
            if pos not in POSITIONS:
                pos = position

            name = meta["name"]
            team = meta["team"]
            if pos == "DEF" and not name:
                name = f"{team or pid} Defense"
            if not name:
                continue

            scored = score_season_projection(
                stats,
                scoring=scoring,
                bonus_model=bonus_model,
                position=pos,
            )

            adp = adp_lookup.get(pid)
            if adp is None:
                adp = 220.0 if pos in {"K", "DEF"} else 999.0

            years_exp, is_rookie = _experience_fields(meta)

            rows.append({
                "player_id": pid,
                "name": name,
                "team": team,
                "pos": pos,
                "adp": round(float(adp), 2),
                "proj_points": scored["custom_points"],
                "base_custom_points": scored["base_points"],
                "estimated_bonus_points": scored["bonus_points"],
                "provider_ppr": float(stats.get("pts_ppr") or 0),
                "tier": None,
                "bye": byes.get(team),
                "injury_risk": _injury_risk(meta),
                "injury_status": meta.get("injury_status"),
                "games_projected": float(stats.get("gp") or 17),
                "years_exp": years_exp,
                "is_rookie": bool(is_rookie),
            })

        time.sleep(0.20)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SleeperDataError("Sleeper returned no usable projection rows.")

    df = (
        df.sort_values(["player_id", "proj_points"], ascending=[True, False])
        .drop_duplicates(subset=["player_id"], keep="first")
    )

    pos_limits = {
        "QB": 40,
        "RB": 100,
        "WR": 120,
        "TE": 50,
        "K": 32,
        "DEF": 32,
    }
    kept = []
    for pos, limit in pos_limits.items():
        group = df[df["pos"] == pos].copy()
        group = group.sort_values(
            ["adp", "proj_points"],
            ascending=[True, False],
        ).head(limit)
        kept.append(group)

    df = pd.concat(kept, ignore_index=True)

    tier_values = []
    for pos, group in df.groupby("pos", sort=False):
        ordered = group.sort_values("proj_points", ascending=False)
        tier = 1
        previous = None
        for idx, row in ordered.iterrows():
            if previous is not None:
                gap = previous - float(row["proj_points"])
                threshold = 18 if pos == "QB" else 14 if pos in {"RB", "WR"} else 10
                if gap >= threshold:
                    tier += 1
            tier_values.append((idx, tier))
            previous = float(row["proj_points"])

    tier_map = dict(tier_values)
    df["tier"] = df.index.map(tier_map).fillna(1).astype(int)
    df = df.sort_values(
        ["adp", "proj_points"],
        ascending=[True, False],
    ).reset_index(drop=True)

    snapshot_at = datetime.now(timezone.utc).isoformat()
    meta = {
        "season": season,
        "snapshot_at_utc": snapshot_at,
        "source": "Sleeper projection API (unofficial projection host) + PatBot custom scoring",
        "adp_field": "adp_dd_ppr from Sleeper week-1 projection endpoint",
        "source_records": total_source_records,
        "draftable_rows": len(df),
        "bonus_method": (
            "Expected per-game threshold bonuses estimated from projected YPG "
            "using a normal-distribution approximation."
        ),
        "important_note": "K/DEF use provider projected points until Yahoo league settings import.",
        "rookie_field": "is_rookie derived from Sleeper years_exp when available",
    }
    return df, meta


def refresh_snapshot(
    config: dict,
    csv_path: str | Path = "data/players_2026_live.csv",
    meta_path: str | Path = "data/players_2026_live.meta.json",
) -> tuple[Path, Path, dict]:
    df, meta = build_live_dataframe(config)

    try:
        from .market import augment_market_sources
        from .consensus import add_consensus_values

        df, market_status = augment_market_sources(df, config)
        df = add_consensus_values(df, config)
        meta["market_sources"] = market_status
    except Exception as exc:
        meta["market_sources"] = {
            "pipeline": {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        }

    csv_path = Path(csv_path)
    meta_path = Path(meta_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(csv_path, index=False)
    meta["draftable_rows"] = len(df)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return csv_path, meta_path, meta
