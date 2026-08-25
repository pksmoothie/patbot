from __future__ import annotations

from datetime import date
import math
import time

import numpy as np
import pandas as pd

from .market import _fp_get, _known_name_match, _numeric, normalize_name


OFFENSE_POSITIONS = {"QB", "RB", "WR", "TE"}
HIGH_RISK_NEWS_TERMS = {
    "suspended", "suspension", "commissioner exempt", "arrested", "charged",
    "felony", "criminal investigation",
}
MEDIUM_RISK_NEWS_TERMS = {
    "lawsuit", "legal", "police", "investigation", "misdemeanor", "discipline",
    "court", "diversion", "battery", "assault",
}


def _safe_float(value, default=np.nan) -> float:
    value = _numeric(value)
    return float(default) if np.isnan(value) else float(value)


def _sleep_for_fp_rate_limit(config: dict) -> None:
    seconds = float(config.get("risk_model", {}).get("fantasypros_request_spacing_seconds", 1.05))
    if seconds > 0:
        time.sleep(seconds)


def _fetch_fp_player_meta(player_names: list[str], config: dict) -> tuple[pd.DataFrame, dict]:
    _sleep_for_fp_rate_limit(config)
    data = _fp_get(
        "nfl/players",
        {"ecr": "included", "show": "pos_rank", "limit": 1000},
    )
    players = data.get("players") or []
    known = {normalize_name(x): x for x in player_names}
    rows = []
    for p in players:
        name = _known_name_match(p.get("player_name") or "", known)
        if not name:
            continue
        pid = p.get("player_id")
        if pid is None:
            continue
        rows.append({
            "name": name,
            "fp_player_id": str(pid),
            "fp_age": _safe_float(p.get("age")),
        })
    frame = pd.DataFrame(rows).drop_duplicates("name", keep="first") if rows else pd.DataFrame()
    return frame, {
        "ok": not frame.empty,
        "matched": int(frame["name"].nunique()) if not frame.empty else 0,
    }


def _fetch_history(fp_ids: set[str], season: int, config: dict) -> tuple[dict[str, list[tuple[int, float]]], dict]:
    rcfg = config.get("risk_model", {})
    seasons = int(rcfg.get("history_seasons", 6))
    years = [season - i for i in range(1, seasons + 1)]
    history: dict[str, list[tuple[int, float]]] = {pid: [] for pid in fp_ids}
    successful_years = []

    for year in years:
        _sleep_for_fp_rate_limit(config)
        try:
            data = _fp_get(
                f"nfl/{year}/player-points",
                {"position": "ALL", "scoring": "PPR", "min": "true"},
            )
        except Exception:
            continue
        successful_years.append(year)
        for p in data.get("players") or []:
            pid = p.get("player_id")
            if pid is None:
                continue
            spid = str(pid)
            if spid not in history:
                continue
            games = _safe_float(p.get("games"))
            if np.isnan(games) or games < 0:
                continue
            history[spid].append((year, min(17.0, games)))

    matched = sum(1 for values in history.values() if values)
    return history, {
        "ok": bool(successful_years),
        "matched": int(matched),
        "seasons": successful_years,
    }


def _fetch_injuries(fp_ids: set[str], season: int, config: dict) -> tuple[dict[str, dict], dict]:
    _sleep_for_fp_rate_limit(config)
    try:
        data = _fp_get(
            "nfl/injuries",
            {"year": season, "include_probabilities": "true"},
        )
    except Exception as exc:
        return {}, {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    out = {}
    for item in data.get("injuries") or []:
        pid = item.get("player_id")
        if pid is None or str(pid) not in fp_ids:
            continue
        out[str(pid)] = item
    return out, {"ok": True, "matched": int(len(out))}


def _news_risk_level(text: str) -> str:
    text = str(text or "").lower()
    if any(term in text for term in HIGH_RISK_NEWS_TERMS):
        return "high"
    if any(term in text for term in MEDIUM_RISK_NEWS_TERMS):
        return "medium"
    return "none"


def _fetch_risk_news(fp_ids: set[str], config: dict) -> tuple[dict[str, dict], dict]:
    _sleep_for_fp_rate_limit(config)
    try:
        data = _fp_get(
            "nfl/news",
            {"limit": 100, "order_by": "updated"},
        )
    except Exception as exc:
        return {}, {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    best = {}
    for item in data.get("items") or []:
        pid = item.get("player_id")
        if pid is None or str(pid) not in fp_ids:
            continue
        text = " ".join([
            str(item.get("title") or ""),
            str(item.get("desc") or ""),
            str(item.get("impact") or ""),
        ])
        level = _news_risk_level(text)
        if level == "none":
            continue
        current = best.get(str(pid))
        if current is None or (level == "high" and current.get("level") != "high"):
            best[str(pid)] = {
                "level": level,
                "title": str(item.get("title") or ""),
                "created": str(item.get("created") or ""),
            }
    return best, {"ok": True, "matched": int(len(best)), "items_checked": len(data.get("items") or [])}


def _history_metrics(values: list[tuple[int, float]], season: int, config: dict) -> tuple[int, float, float]:
    if not values:
        return 0, np.nan, 0.0

    rcfg = config.get("risk_model", {})
    raw_weights = rcfg.get("history_weights", [0.30, 0.25, 0.18, 0.12, 0.09, 0.06])
    weight_by_year = {
        season - i: float(raw_weights[i - 1]) if i - 1 < len(raw_weights) else 0.0
        for i in range(1, int(rcfg.get("history_seasons", 6)) + 1)
    }
    weighted = []
    for year, games in values:
        weight = weight_by_year.get(year, 0.0)
        if weight > 0:
            weighted.append((float(games), weight))
    if not weighted:
        return len(values), np.nan, 0.0
    denom = sum(weight for _, weight in weighted)
    games = sum(g * weight for g, weight in weighted) / denom
    missed_rate = max(0.0, min(1.0, (17.0 - games) / 17.0))
    return len(weighted), games, missed_rate


def _age_tail_bonus(pos: str, age: float) -> float:
    if np.isnan(age):
        return 0.0
    pos = str(pos).upper()
    if pos == "RB" and age >= 29:
        return min(0.08, 0.025 * (age - 28.0))
    if pos in {"WR", "TE"} and age >= 31:
        return min(0.05, 0.015 * (age - 30.0))
    if pos == "QB" and age >= 36:
        return min(0.04, 0.010 * (age - 35.0))
    return 0.0


def _status_probability(item: dict | None, fallback_status: str | None) -> tuple[str, float]:
    if item:
        status = str(item.get("status") or fallback_status or "")
        p = _safe_float(item.get("probability_of_playing"))
        if not np.isnan(p):
            return status, max(0.0, min(1.0, p))
    else:
        status = str(fallback_status or "")

    lower = status.lower()
    if not lower:
        return status, 1.0
    if "out" in lower or "ir" in lower or "pup" in lower:
        return status, 0.20
    if "doubt" in lower:
        return status, 0.35
    if "question" in lower:
        return status, 0.75
    if "prob" in lower:
        return status, 0.95
    return status, 0.90


def _override_is_active(override: dict) -> bool:
    expires = str(override.get("expires") or "").strip()
    if not expires:
        return True
    try:
        return date.today() <= date.fromisoformat(expires)
    except ValueError:
        return True


def augment_risk_sources(players: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    """Attach explicit availability risk without replacing source projections.

    Historical games shape tail risk; current injury feeds shape near-term risk;
    recent news/manual overrides shape suspension/legal-event risk. The simulation
    samples missed games and credits partial replacement-player production.
    """
    out = players.copy()
    names = out["name"].dropna().astype(str).tolist()
    season = int(config.get("league", {}).get("season", 2026))
    rcfg = config.get("risk_model", {})
    status = {}

    try:
        meta, meta_status = _fetch_fp_player_meta(names, config)
        status["fantasypros_player_meta"] = meta_status
        if not meta.empty:
            out = out.merge(meta, on="name", how="left")
    except Exception as exc:
        status["fantasypros_player_meta"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        out["fp_player_id"] = np.nan
        out["fp_age"] = np.nan

    if "fp_player_id" not in out:
        out["fp_player_id"] = np.nan
    if "fp_age" not in out:
        out["fp_age"] = np.nan

    fp_ids = set(out["fp_player_id"].dropna().astype(str))
    history, history_status = _fetch_history(fp_ids, season, config) if fp_ids else ({}, {"ok": False, "error": "No FantasyPros player IDs matched."})
    status["fantasypros_availability_history"] = history_status
    injuries, injury_status = _fetch_injuries(fp_ids, season, config) if fp_ids else ({}, {"ok": False, "error": "No FantasyPros player IDs matched."})
    status["fantasypros_injuries"] = injury_status
    news, news_status = _fetch_risk_news(fp_ids, config) if fp_ids else ({}, {"ok": False, "error": "No FantasyPros player IDs matched."})
    status["fantasypros_risk_news"] = news_status

    level_prob = {
        "none": 0.0,
        "medium": float(rcfg.get("news_medium_event_probability", 0.05)),
        "high": float(rcfg.get("news_high_event_probability", 0.18)),
    }
    level_max = {
        "none": 0,
        "medium": int(rcfg.get("news_medium_max_missed_games", 3)),
        "high": int(rcfg.get("news_high_max_missed_games", 6)),
    }
    overrides = config.get("risk_overrides", {}) or {}
    override_hits = 0

    records = []
    for _, row in out.iterrows():
        name = str(row.get("name") or "")
        pos = str(row.get("pos") or "").upper()
        fp_id_value = row.get("fp_player_id")
        fp_id = "" if pd.isna(fp_id_value) else str(fp_id_value)
        age = _safe_float(row.get("fp_age"))

        seasons_obs, hist_games, hist_missed_rate = _history_metrics(history.get(fp_id, []), season, config)
        age_bonus = _age_tail_bonus(pos, age)
        injury_item = injuries.get(fp_id)
        current_status, play_prob = _status_probability(injury_item, row.get("injury_status"))
        current_risk = max(0.0, 1.0 - play_prob)

        news_item = news.get(fp_id, {})
        news_level = news_item.get("level", "none")
        off_prob = level_prob.get(news_level, 0.0)
        off_max = level_max.get(news_level, 0)
        manual_note = ""
        override = overrides.get(name) or {}
        if isinstance(override, dict) and override and _override_is_active(override):
            override_hits += 1
            if "off_field_event_probability" in override:
                off_prob = max(off_prob, float(override["off_field_event_probability"]))
            if "off_field_max_missed_games" in override:
                off_max = max(off_max, int(override["off_field_max_missed_games"]))
            manual_note = str(override.get("note") or "")

        base_cat = float(rcfg.get("base_catastrophic_probability", 0.02))
        cat_prob = base_cat + float(rcfg.get("history_catastrophic_weight", 0.45)) * hist_missed_rate
        cat_prob += age_bonus + float(rcfg.get("current_injury_catastrophic_weight", 0.10)) * current_risk
        cat_prob = max(base_cat, min(float(rcfg.get("max_catastrophic_probability", 0.35)), cat_prob))

        minor_lambda = float(rcfg.get("base_minor_miss_lambda", 0.15))
        minor_lambda += float(rcfg.get("history_minor_weight", 1.0)) * hist_missed_rate
        minor_lambda += float(rcfg.get("current_injury_minor_weight", 0.40)) * current_risk
        minor_lambda = max(0.0, min(float(rcfg.get("max_minor_miss_lambda", 1.50)), minor_lambda))

        history_component = min(1.0, hist_missed_rate / 0.30) if hist_missed_rate > 0 else 0.0
        age_component = min(1.0, age_bonus / 0.08) if age_bonus > 0 else 0.0
        off_component = min(1.0, off_prob / 0.20) if off_prob > 0 else 0.0
        risk_score = (
            0.45 * history_component
            + 0.25 * current_risk
            + 0.15 * age_component
            + 0.15 * off_component
        )
        risk_score = max(float(row.get("injury_risk") or 0.0), min(1.0, risk_score))

        notes = []
        if seasons_obs:
            notes.append(f"{seasons_obs}y weighted games {hist_games:.1f}")
        if current_status:
            notes.append(f"{current_status} ({play_prob:.0%} play probability)")
        if news_item.get("title"):
            notes.append(f"news: {news_item['title']}")
        if manual_note:
            notes.append(manual_note)

        records.append({
            "name": name,
            "history_seasons_observed": seasons_obs,
            "history_weighted_games": round(hist_games, 2) if not np.isnan(hist_games) else np.nan,
            "history_missed_rate": round(hist_missed_rate, 4),
            "current_injury_status": current_status,
            "current_play_probability": round(play_prob, 4),
            "age_tail_bonus": round(age_bonus, 4),
            "catastrophic_miss_probability": round(cat_prob, 4),
            "minor_miss_lambda": round(minor_lambda, 4),
            "off_field_risk_level": news_level,
            "off_field_miss_probability": round(off_prob, 4),
            "off_field_max_missed_games": int(off_max),
            "risk_score": round(risk_score, 4),
            "risk_note": " • ".join(notes),
        })

    risk_frame = pd.DataFrame(records)
    out = out.drop(columns=[c for c in risk_frame.columns if c != "name" and c in out.columns], errors="ignore")
    out = out.merge(risk_frame, on="name", how="left")
    out["sleeper_current_injury_risk"] = pd.to_numeric(out.get("injury_risk"), errors="coerce").fillna(0.0)
    out["injury_risk"] = pd.to_numeric(out["risk_score"], errors="coerce").fillna(out["sleeper_current_injury_risk"])

    status["manual_risk_overrides"] = {"ok": True, "matched": int(override_hits)}
    status["model"] = {
        "ok": True,
        "matched": int(out["risk_score"].notna().sum()),
        "note": "History shapes availability tails; missed games receive partial replacement value in simulation.",
    }
    return out, status
