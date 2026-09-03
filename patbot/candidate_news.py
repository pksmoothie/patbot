from __future__ import annotations

from datetime import datetime, timezone
import time

import pandas as pd

from . import draft_news as _draft_news
from .market import _fp_get
from .risk import _override_is_active, _safe_float


OFFENSE_POSITIONS = {"QB", "RB", "WR", "TE"}
_CANDIDATE_NEWS_CACHE: dict[str, dict] = {}
_TIER_RANK = {"none": 0, "yellow": 1, "orange": 2, "red": 3}

# Draft-night backstop for a status independently verified through the
# FantasyPros player-specific feed. It expires automatically after the draft
# window and is superseded by a newer explicit GREEN resolution in the direct feed.
_DRAFT_NIGHT_BACKSTOPS = {
    "Josh Jacobs": {
        "current_alert_tier": "red",
        "play_probability_cap": 0.20,
        "off_field_event_probability": 1.0,
        "off_field_max_missed_games": 6,
        "as_of_utc": "2026-09-03T22:00:00Z",
        "expires": "2026-09-04",
        "note": (
            "Draft-night backstop: confirmed on the Commissioner's Exempt List "
            "as of 2026-09-03; he cannot practice or play while listed."
        ),
    }
}


def clear_candidate_news_cache() -> None:
    _CANDIDATE_NEWS_CACHE.clear()


def _clean_id(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def _utc_timestamp(value) -> pd.Timestamp:
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")
    except Exception:
        return pd.Timestamp("1970-01-01", tz="UTC")


def _signal_text(item: dict) -> str:
    return " ".join(
        [
            str(item.get("title") or ""),
            str(item.get("desc") or ""),
            str(item.get("impact") or ""),
        ]
    ).strip()


def newest_direct_signal(
    items: list[dict],
    *,
    max_age_days: int = 14,
    now: pd.Timestamp | None = None,
) -> dict:
    """Return the newest relevant direct-player signal, including GREEN resolution."""
    current = _utc_timestamp(now if now is not None else pd.Timestamp.now(tz="UTC"))
    recent: list[tuple[pd.Timestamp, dict]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        raw = item.get("updated") or item.get("created") or item.get("published")
        ts = _utc_timestamp(raw)
        if raw:
            age_days = (current - ts).total_seconds() / 86400.0
            if age_days > max(1, int(max_age_days)):
                continue
        recent.append((ts, item))
    recent.sort(key=lambda pair: pair[0], reverse=True)

    for ts, item in recent:
        signal = _draft_news.classify_draft_news(_signal_text(item))
        if str(signal.get("tier") or "none").lower() == "none":
            continue
        return {
            **signal,
            "title": str(item.get("title") or ""),
            "created": str(item.get("created") or item.get("updated") or ""),
            "_timestamp": ts.isoformat(),
            "source": "fantasypros_direct_fpid",
        }
    return {}


def _cache_fresh(entry: dict, ttl_seconds: int, now: pd.Timestamp) -> bool:
    checked = entry.get("checked_at_utc")
    if not checked:
        return False
    age = (now - _utc_timestamp(checked)).total_seconds()
    return 0 <= age <= max(0, int(ttl_seconds))


def verify_candidate_news(
    players: pd.DataFrame,
    candidate_ids: list[str],
    config: dict,
    *,
    cache: dict[str, dict] | None = None,
    now: pd.Timestamp | None = None,
) -> tuple[dict[str, dict], dict]:
    """Directly verify FantasyPros news for the live offensive decision set only."""
    rcfg = config.get("risk_model", {}) or {}
    ttl_seconds = max(0, int(rcfg.get("candidate_news_cache_ttl_seconds", 600)))
    direct_limit = max(1, int(rcfg.get("candidate_news_direct_limit", 10)))
    max_age_days = max(1, int(rcfg.get("candidate_news_max_age_days", 14)))
    spacing = max(0.0, float(rcfg.get("fantasypros_request_spacing_seconds", 1.05)))

    current = _utc_timestamp(now if now is not None else pd.Timestamp.now(tz="UTC"))
    store = _CANDIDATE_NEWS_CACHE if cache is None else cache

    frame = players.copy()
    frame["player_id"] = frame["player_id"].astype(str)
    wanted = [str(x) for x in candidate_ids]
    subset = frame[
        frame["player_id"].isin(wanted)
        & frame["pos"].astype(str).str.upper().isin(OFFENSE_POSITIONS)
    ].copy()

    signals: dict[str, dict] = {}
    checked = 0
    cache_hits = 0
    api_calls = 0
    failures = 0
    last_call_at = 0.0

    for _, row in subset.iterrows():
        player_id = str(row.get("player_id") or "")
        fp_id = _clean_id(row.get("fp_player_id"))
        if not fp_id:
            failures += 1
            continue

        entry = store.get(fp_id) or {}
        if _cache_fresh(entry, ttl_seconds, current):
            cache_hits += 1
            checked += 1
            signal = entry.get("signal") or {}
        else:
            if api_calls > 0 and spacing > 0:
                elapsed = time.perf_counter() - last_call_at
                if elapsed < spacing:
                    time.sleep(spacing - elapsed)
            try:
                data = _fp_get("nfl/news", {"fpid": fp_id, "limit": direct_limit})
                api_calls += 1
                last_call_at = time.perf_counter()
                signal = newest_direct_signal(
                    data.get("items") or [],
                    max_age_days=max_age_days,
                    now=current,
                )
                store[fp_id] = {
                    "checked_at_utc": current.isoformat(),
                    "player_id": player_id,
                    "name": str(row.get("name") or ""),
                    "signal": signal,
                }
                checked += 1
            except Exception as exc:
                api_calls += 1
                last_call_at = time.perf_counter()
                failures += 1
                store[fp_id] = {
                    "checked_at_utc": current.isoformat(),
                    "player_id": player_id,
                    "name": str(row.get("name") or ""),
                    "signal": {},
                    "error": f"{type(exc).__name__}: {exc}",
                }
                signal = {}

        if signal:
            signals[player_id] = dict(signal)

    fingerprint = tuple(
        sorted(
            (
                pid,
                str(sig.get("tier") or "none"),
                str(sig.get("created") or ""),
            )
            for pid, sig in signals.items()
        )
    )
    return signals, {
        "ok": failures == 0,
        "checked": int(checked),
        "requested": int(len(subset)),
        "cache_hits": int(cache_hits),
        "api_calls": int(api_calls),
        "failures": int(failures),
        "ttl_seconds": int(ttl_seconds),
        "fingerprint": fingerprint,
    }


def _tier_max(*tiers: str) -> str:
    cleaned = [str(t or "none").lower() for t in tiers]
    return max(cleaned, key=lambda t: _TIER_RANK.get(t, 0), default="none")


def _structured_probability(row: pd.Series) -> tuple[float, str]:
    source = str(row.get("current_status_source") or "none")
    probability = max(
        0.0,
        min(1.0, _safe_float(row.get("current_play_probability"), 1.0)),
    )
    if source in {"fantasypros", "sleeper_hard", "structured_hard"}:
        return probability, source

    status = str(row.get("current_injury_status") or "").lower()
    if any(term in status for term in ("out", "ir", "pup", "nfi", "suspend", "exempt")):
        return min(probability, 0.20), "structured_hard"
    if "doubt" in status:
        return min(probability, 0.50), "structured_hard"
    return 1.0, "none"


def _combined_override(name: str, config: dict) -> dict:
    built_in = _DRAFT_NIGHT_BACKSTOPS.get(name) or {}
    configured = (config.get("risk_overrides", {}) or {}).get(name) or {}
    merged = dict(built_in)
    if isinstance(configured, dict):
        merged.update(configured)
    if not merged or not _override_is_active(merged):
        return {}
    return merged


def _recompute_risk(
    row: pd.Series,
    *,
    play_probability: float,
    off_field_probability: float,
    config: dict,
) -> tuple[float, float, float]:
    rcfg = config.get("risk_model", {}) or {}
    hist_missed = max(0.0, _safe_float(row.get("history_missed_rate"), 0.0))
    history_scale = max(0.0, _safe_float(row.get("history_signal_scale"), 1.0))
    age_bonus = max(0.0, _safe_float(row.get("age_tail_bonus"), 0.0))
    effective_hist = hist_missed * history_scale
    current_risk = max(0.0, 1.0 - play_probability)

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
    off_component = min(1.0, off_field_probability / 0.20) if off_field_probability > 0 else 0.0
    risk_score = (
        0.45 * history_component
        + 0.25 * current_risk
        + 0.15 * age_component
        + 0.15 * off_component
    )
    sleeper_floor = max(0.0, _safe_float(row.get("sleeper_current_injury_risk"), 0.0))
    risk_score = max(sleeper_floor, min(1.0, risk_score))
    return risk_score, cat_prob, minor_lambda


def _manual_as_of(override: dict) -> pd.Timestamp:
    return _utc_timestamp(
        override.get("as_of_utc")
        or override.get("as_of")
        or "1970-01-01T00:00:00Z"
    )


def _overlay_row(
    row: pd.Series,
    *,
    config: dict,
    direct_signal: dict | None = None,
) -> dict:
    name = str(row.get("name") or "")
    override = _combined_override(name, config)
    signal = dict(direct_signal or {})
    signal_tier = str(signal.get("tier") or "none").lower()
    signal_ts = _utc_timestamp(signal.get("_timestamp") or signal.get("created"))

    # A newer explicit GREEN resolution is authoritative over the temporary backstop.
    if signal_tier == "green" and override and signal_ts >= _manual_as_of(override):
        override = {}

    structured_prob, structured_source = _structured_probability(row)
    structured_tier = "none"
    if structured_prob <= 0.35:
        structured_tier = "red"
    elif structured_prob <= 0.75:
        structured_tier = "orange"
    elif structured_prob < 0.98:
        structured_tier = "yellow"

    active_signal = signal if signal_tier in {"red", "orange", "yellow"} else {}
    direct_cap = max(
        0.0,
        min(1.0, _safe_float(active_signal.get("play_probability_cap"), 1.0)),
    )
    manual_cap = max(
        0.0,
        min(1.0, _safe_float(override.get("play_probability_cap"), 1.0)),
    )
    play_probability = min(structured_prob, direct_cap, manual_cap)

    manual_tier = str(
        override.get("current_alert_tier")
        or override.get("draft_news_tier")
        or "none"
    ).lower()
    off_field_probability = max(
        0.0,
        _safe_float(active_signal.get("off_field_event_probability"), 0.0),
        _safe_float(override.get("off_field_event_probability"), 0.0),
    )
    off_field_max = max(
        0,
        int(_safe_float(active_signal.get("off_field_max_missed_games"), 0.0)),
        int(_safe_float(override.get("off_field_max_missed_games"), 0.0)),
    )

    alert_tier = _tier_max(
        structured_tier,
        signal_tier if active_signal else "none",
        manual_tier,
        "yellow" if off_field_probability > 0 else "none",
    )
    material = alert_tier in {"red", "orange", "yellow"}

    manual_changes_play = (
        "play_probability_cap" in override
        or manual_tier in {"red", "orange"}
    )
    if active_signal and direct_cap <= structured_prob and direct_cap <= manual_cap:
        status_source = f"candidate_news_{signal_tier}"
    elif override and manual_changes_play and manual_cap <= structured_prob and manual_cap <= direct_cap:
        status_source = f"manual_{manual_tier if manual_tier != 'none' else 'risk'}"
    elif signal_tier == "green" and structured_source == "none":
        status_source = "candidate_news_resolved"
    elif structured_source != "none":
        status_source = structured_source
    else:
        existing_source = str(row.get("current_status_source") or "none")
        status_source = existing_source if not existing_source.startswith(("news_", "candidate_news_", "manual_")) else "none"

    risk_score, cat_prob, minor_lambda = _recompute_risk(
        row,
        play_probability=play_probability,
        off_field_probability=off_field_probability,
        config=config,
    )

    direct_title = str(active_signal.get("title") or "").strip()
    manual_note = str(override.get("note") or "").strip()
    if direct_title:
        display_news = f"{alert_tier.upper()} — {direct_title}"
    elif override and material:
        display_news = f"{alert_tier.upper()} — manual risk monitor"
    elif signal_tier == "green":
        display_news = f"GREEN — {str(signal.get('title') or 'resolved')}"
    else:
        display_news = str(row.get("fast_news_title") or "")
        if display_news.lower() == "nan":
            display_news = ""

    note_parts = []
    existing_note = str(row.get("risk_note") or "").strip()
    if existing_note and existing_note.lower() != "nan":
        note_parts.append(existing_note)
    if direct_title:
        note_parts.append(f"candidate direct check: {direct_title}")
    if manual_note and manual_note not in note_parts:
        note_parts.append(manual_note)

    if active_signal:
        off_field_level = str(active_signal.get("level") or "none")
    elif manual_tier == "red":
        off_field_level = "high"
    elif manual_tier in {"orange", "yellow"} or off_field_probability > 0:
        off_field_level = "medium"
    else:
        off_field_level = "none"

    return {
        "current_play_probability": round(float(play_probability), 4),
        "current_status_source": status_source,
        "current_status_material": bool(material),
        "current_alert_tier": alert_tier,
        "fast_news_tier": alert_tier,
        "fast_news_title": display_news,
        "fast_news_reason": str(active_signal.get("reason") or ""),
        "fast_news_created": str(active_signal.get("created") or ""),
        "off_field_risk_level": off_field_level,
        "off_field_miss_probability": round(float(off_field_probability), 4),
        "off_field_max_missed_games": int(off_field_max),
        "catastrophic_miss_probability": round(float(cat_prob), 4),
        "minor_miss_lambda": round(float(minor_lambda), 4),
        "risk_score": round(float(risk_score), 4),
        "injury_risk": round(float(risk_score), 4),
        "risk_note": " • ".join(note_parts),
        "candidate_news_verified_at_utc": (
            datetime.now(timezone.utc).isoformat()
            if direct_signal is not None
            else str(row.get("candidate_news_verified_at_utc") or "")
        ),
    }


def apply_manual_risk_overrides(players: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply dated manual draft-night backstops to the current risk fields."""
    out = players.copy()
    for idx, row in out.iterrows():
        if not _combined_override(str(row.get("name") or ""), config):
            continue
        updates = _overlay_row(row, config=config, direct_signal=None)
        for key, value in updates.items():
            out.at[idx, key] = value
    return out


def apply_candidate_news_signals(
    players: pd.DataFrame,
    signals: dict[str, dict],
    config: dict,
) -> pd.DataFrame:
    """Apply direct candidate news to the same risk fields used by scoring/simulation."""
    out = apply_manual_risk_overrides(players, config)
    if not signals:
        return out

    out["player_id"] = out["player_id"].astype(str)
    index_by_id = {str(pid): idx for idx, pid in out["player_id"].items()}
    for player_id, signal in signals.items():
        idx = index_by_id.get(str(player_id))
        if idx is None:
            continue
        updates = _overlay_row(out.loc[idx], config=config, direct_signal=signal)
        for key, value in updates.items():
            out.at[idx, key] = value
    return out
