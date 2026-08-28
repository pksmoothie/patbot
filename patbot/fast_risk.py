from __future__ import annotations

from datetime import datetime, timezone
import time

import numpy as np
import pandas as pd

from .risk import (
    _fetch_injuries,
    _fetch_risk_news,
    _override_is_active,
    _safe_float,
    _status_probability,
)
from .sleeper import APP_HOST, _get_json, _injury_risk, _session


def _norm_id(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.endswith(".0"):
        head = text[:-2]
        if head.isdigit():
            return head
    return text


def _clean_status(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _fetch_sleeper_current_status(players: pd.DataFrame) -> tuple[dict[str, dict], dict]:
    """Fetch the current Sleeper player metadata in one lightweight request."""
    try:
        data = _get_json(_session(), f"{APP_HOST}/v1/players/nfl")
    except Exception as exc:
        return {}, {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    if not isinstance(data, dict):
        return {}, {"ok": False, "error": f"Unexpected Sleeper players payload: {type(data).__name__}"}

    wanted = {_norm_id(x) for x in players.get("player_id", pd.Series(dtype=object))}
    wanted.discard("")
    out: dict[str, dict] = {}
    for pid, item in data.items():
        spid = _norm_id(pid)
        if spid not in wanted or not isinstance(item, dict):
            continue
        out[spid] = {
            "injury_status": item.get("injury_status"),
            "active": item.get("active"),
            "team": item.get("team"),
            "status": item.get("status"),
        }
    return out, {"ok": True, "matched": int(len(out))}


def _history_components(row: pd.Series) -> tuple[float, float, float]:
    missed = _safe_float(row.get("history_missed_rate"), 0.0)
    scale = _safe_float(row.get("history_signal_scale"), 1.0)
    age_bonus = _safe_float(row.get("age_tail_bonus"), 0.0)
    return max(0.0, missed), max(0.0, scale), max(0.0, age_bonus)


def _is_serious_sleeper_status(status: str | None) -> bool:
    text = _clean_status(status).lower()
    return any(term in text for term in ("out", "ir", "pup", "nfi", "doubt"))


def _draft_day_status_probability(
    fantasypros_item: dict | None,
    sleeper_status: str | None,
) -> tuple[str, float, str]:
    """Resolve current play probability for draft-day use.

    FantasyPros is the corroborating injury source. If it has a current row, use
    its explicit probability/status logic. Sleeper alone is still authoritative
    for hard statuses such as PUP/IR/Out/Doubtful, but uncorroborated soft labels
    such as Questionable/Probable are informational only and add zero draft-day
    risk penalty.
    """
    cleaned_sleeper = _clean_status(sleeper_status)
    if fantasypros_item:
        status, probability = _status_probability(fantasypros_item, cleaned_sleeper)
        return _clean_status(status), probability, "fantasypros"

    status = cleaned_sleeper
    lower = status.lower()
    if not lower:
        return "", 1.0, "none"
    if "out" in lower or "ir" in lower or "pup" in lower or "nfi" in lower:
        return status, 0.20, "sleeper_hard"
    if "doubt" in lower:
        return status, 0.50, "sleeper_hard"
    return status, 1.0, "sleeper_ignored"


def refresh_fast_risk(players: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    """Refresh only current injury/news inputs while preserving slow history work.

    This is the draft-day layer: one Sleeper player-status request plus the
    FantasyPros current injuries and recent-news requests. Six-year availability
    history, projections, ECR/ADP and Athletic data are intentionally untouched.
    """
    started = time.perf_counter()
    out = players.copy()
    season = int(config.get("league", {}).get("season", 2026))
    rcfg = config.get("risk_model", {})

    sleeper_status, sleeper_meta = _fetch_sleeper_current_status(out)

    fp_ids = {_norm_id(x) for x in out.get("fp_player_id", pd.Series(dtype=object))}
    fp_ids.discard("")
    if fp_ids:
        injuries, injury_meta = _fetch_injuries(fp_ids, season, config)
        news, news_meta = _fetch_risk_news(fp_ids, config)
    else:
        injuries, injury_meta = {}, {"ok": False, "error": "No cached FantasyPros player IDs in snapshot."}
        news, news_meta = {}, {"ok": False, "error": "No cached FantasyPros player IDs in snapshot."}

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
    refreshed_at = datetime.now(timezone.utc).isoformat()

    rows = []
    alerts = 0
    for _, row in out.iterrows():
        name = str(row.get("name") or "")
        sleeper_id = _norm_id(row.get("player_id"))
        fp_id = _norm_id(row.get("fp_player_id"))
        smeta = sleeper_status.get(sleeper_id, {})
        sleeper_injury_status = _clean_status(smeta.get("injury_status"))
        if not sleeper_injury_status:
            sleeper_injury_status = _clean_status(row.get("injury_status"))

        injury_item = injuries.get(fp_id)
        current_status, play_prob, status_source = _draft_day_status_probability(
            injury_item,
            sleeper_injury_status,
        )
        current_risk = max(0.0, 1.0 - play_prob)

        news_item = news.get(fp_id, {})
        news_level = str(news_item.get("level", "none") or "none")
        off_prob = level_prob.get(news_level, 0.0)
        off_max = level_max.get(news_level, 0)

        manual_note = ""
        override = overrides.get(name) or {}
        if isinstance(override, dict) and override and _override_is_active(override):
            if "off_field_event_probability" in override:
                off_prob = max(off_prob, float(override["off_field_event_probability"]))
            if "off_field_max_missed_games" in override:
                off_max = max(off_max, int(override["off_field_max_missed_games"]))
            manual_note = str(override.get("note") or "")

        hist_missed, history_scale, age_bonus = _history_components(row)
        effective_hist = hist_missed * history_scale

        base_cat = float(rcfg.get("base_catastrophic_probability", 0.02))
        cat_prob = base_cat + float(rcfg.get("history_catastrophic_weight", 0.45)) * effective_hist
        cat_prob += age_bonus + float(rcfg.get("current_injury_catastrophic_weight", 0.10)) * current_risk
        cat_prob = max(base_cat, min(float(rcfg.get("max_catastrophic_probability", 0.35)), cat_prob))

        minor_lambda = float(rcfg.get("base_minor_miss_lambda", 0.15))
        minor_lambda += float(rcfg.get("history_minor_weight", 1.0)) * effective_hist
        minor_lambda += float(rcfg.get("current_injury_minor_weight", 0.40)) * current_risk
        minor_lambda = max(0.0, min(float(rcfg.get("max_minor_miss_lambda", 1.50)), minor_lambda))

        history_component = min(1.0, effective_hist / 0.30) if effective_hist > 0 else 0.0
        age_component = min(1.0, age_bonus / 0.08) if age_bonus > 0 else 0.0
        off_component = min(1.0, off_prob / 0.20) if off_prob > 0 else 0.0
        risk_score = (
            0.45 * history_component
            + 0.25 * current_risk
            + 0.15 * age_component
            + 0.15 * off_component
        )
        sleeper_risk = _injury_risk({"injury_status": sleeper_injury_status})
        if status_source in {"none", "sleeper_ignored"}:
            sleeper_risk = 0.0
        risk_score = max(sleeper_risk, min(1.0, risk_score))

        notes = []
        hist_games = _safe_float(row.get("history_weighted_games"))
        seasons_obs = _safe_float(row.get("history_seasons_observed"), 0.0)
        if seasons_obs > 0 and np.isfinite(hist_games):
            notes.append(f"{int(seasons_obs)}y weighted availability {hist_games:.1f}/17")
            if history_scale < 1.0:
                notes.append(f"young-player history weight {history_scale:.0%}")
        if current_status:
            if status_source == "sleeper_ignored":
                notes.append(f"{current_status} (Sleeper-only soft label ignored for draft risk)")
            else:
                notes.append(f"{current_status} ({play_prob:.0%} play probability; {status_source})")
        if news_item.get("title"):
            notes.append(f"news: {news_item['title']}")
        if manual_note:
            notes.append(manual_note)

        material_current = (
            status_source == "fantasypros"
            or _is_serious_sleeper_status(current_status)
            or news_level != "none"
            or off_prob > 0
        )
        if material_current:
            alerts += 1

        rows.append({
            "name": name,
            "injury_status": sleeper_injury_status,
            "sleeper_current_injury_risk": round(float(sleeper_risk), 4),
            "current_injury_status": current_status,
            "current_play_probability": round(float(play_prob), 4),
            "current_status_source": status_source,
            "current_status_material": bool(material_current),
            "catastrophic_miss_probability": round(float(cat_prob), 4),
            "minor_miss_lambda": round(float(minor_lambda), 4),
            "off_field_risk_level": news_level,
            "off_field_miss_probability": round(float(off_prob), 4),
            "off_field_max_missed_games": int(off_max),
            "fast_news_title": str(news_item.get("title") or ""),
            "risk_score": round(float(risk_score), 4),
            "injury_risk": round(float(risk_score), 4),
            "risk_note": " • ".join(notes),
            "fast_risk_refreshed_at_utc": refreshed_at,
        })

    update = pd.DataFrame(rows)
    replace_cols = [c for c in update.columns if c != "name"]
    out = out.drop(columns=[c for c in replace_cols if c in out.columns], errors="ignore")
    out = out.merge(update, on="name", how="left")

    elapsed = time.perf_counter() - started
    status = {
        "sleeper_current_status": sleeper_meta,
        "fantasypros_current_injuries": injury_meta,
        "fantasypros_recent_risk_news": news_meta,
        "fast_risk_model": {
            "ok": True,
            "matched": int(len(out)),
            "alerts": int(alerts),
            "elapsed_seconds": round(float(elapsed), 2),
            "refreshed_at_utc": refreshed_at,
            "note": (
                "Fast refresh reused cached history/projections and updated only current Sleeper status plus FantasyPros injuries/news. "
                "Uncorroborated Sleeper soft labels are informational only and add zero draft-day risk penalty."
            ),
        },
    }
    return out, status
