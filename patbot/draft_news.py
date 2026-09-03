from __future__ import annotations

import time

import pandas as pd

from .market import _fp_get, _known_name_match, normalize_name


RESOLVED_NEWS_TERMS = (
    "removed from the commissioner's exempt list",
    "removed from commissioner's exempt list",
    "removed from the exempt list",
    "reinstated by the nfl",
    "activated from injured reserve",
    "activated off injured reserve",
    "activated from reserve/pup",
    "activated from pup",
    "activated from reserve/nfi",
    "returned to practice",
    "returns to practice",
    "back at practice",
    "full participant",
    "full practice",
    "cleared to play",
    "cleared for week",
    "removed from the injury report",
    "off the injury report",
    "no injury designation",
    "expected to play",
    "will play",
    "good to go",
)

HARD_AVAILABILITY_TERMS = (
    "commissioner's exempt list",
    "commissioner exempt list",
    "exempt list",
    "not permitted to practice or play",
    "not permitted to practice or attend games",
    "cannot practice or play",
    "not eligible to play",
    "suspended",
    "suspension",
    "reserve/pup",
    "placed on pup",
    "physically unable to perform list",
    "reserve/nfi",
    "placed on nfi",
    "non-football injury list",
    "placed on injured reserve",
    "placed on ir",
    "reserve/injured",
    "ruled out",
    "out for the season",
    "out for season",
    "season-ending",
    "season ending",
    "will miss week",
    "will miss at least",
    "will miss multiple",
)

MEANINGFUL_UNCERTAINTY_TERMS = (
    "did not practice",
    "didn't practice",
    "dnp",
    "missed practice",
    "hasn't practiced",
    "has not practiced",
    "not practicing",
    "not cleared",
    "game-time decision",
    "game time decision",
    "availability uncertain",
    "status uncertain",
    "could miss",
    "may miss",
    "might miss",
    "expected to miss",
    "unlikely to play",
    "in doubt",
    "week 1 status",
    "week 1 availability",
    "holding out",
    "holdout",
    "has not reported",
    "hasn't reported",
)

MONITOR_TERMS = (
    "limited practice",
    "limited participant",
    "questionable",
    "day-to-day",
    "day to day",
    "soreness",
    "precautionary",
    "arrested",
    "charged",
    "misdemeanor",
    "felony",
    "lawsuit",
    "legal",
    "police",
    "investigation",
    "discipline",
    "court",
    "battery",
    "assault",
)

DISCIPLINE_CONFIRMED_TERMS = (
    "commissioner's exempt list",
    "commissioner exempt list",
    "exempt list",
    "suspended",
    "suspension",
)

LEGAL_MONITOR_TERMS = (
    "arrested",
    "charged",
    "misdemeanor",
    "felony",
    "lawsuit",
    "legal",
    "police",
    "investigation",
    "discipline",
    "court",
    "battery",
    "assault",
)


def classify_draft_news(text: str) -> dict:
    """Classify news by draft relevance, not by whether the story sounds negative.

    RED means confirmed current unavailability or league action. ORANGE means a
    credible near-term availability question. YELLOW is awareness-only and is
    intentionally a very small risk nudge. GREEN is a newer resolution signal
    that should supersede stale negative news. NONE is ignored.
    """
    lower = str(text or "").lower()
    if not lower.strip():
        return _signal("none", "none", 1.0, 0.0, 0, "irrelevant")

    # Explicit resolution phrases come first so "activated from PUP" does not
    # get misclassified merely because the sentence also contains "PUP".
    if any(term in lower for term in RESOLVED_NEWS_TERMS):
        return _signal("green", "none", 1.0, 0.0, 0, "resolved")

    if any(term in lower for term in HARD_AVAILABILITY_TERMS):
        confirmed_discipline = any(term in lower for term in DISCIPLINE_CONFIRMED_TERMS)
        return _signal(
            "red",
            "high" if confirmed_discipline else "none",
            0.20,
            1.0 if confirmed_discipline else 0.0,
            6 if confirmed_discipline else 0,
            "confirmed unavailability/league action" if confirmed_discipline else "confirmed hard availability issue",
        )

    if any(term in lower for term in MEANINGFUL_UNCERTAINTY_TERMS):
        return _signal("orange", "none", 0.65, 0.0, 0, "meaningful near-term availability uncertainty")

    if any(term in lower for term in MONITOR_TERMS):
        legal = any(term in lower for term in LEGAL_MONITOR_TERMS)
        return _signal(
            "yellow",
            "medium" if legal else "none",
            0.95,
            0.03 if legal else 0.0,
            2 if legal else 0,
            "monitor-only legal/discipline uncertainty" if legal else "monitor-only availability note",
        )

    return _signal("none", "none", 1.0, 0.0, 0, "irrelevant")


def _signal(
    tier: str,
    legacy_level: str,
    play_probability_cap: float,
    off_field_event_probability: float,
    off_field_max_missed_games: int,
    reason: str,
) -> dict:
    return {
        "tier": str(tier),
        "level": str(legacy_level),
        "play_probability_cap": float(play_probability_cap),
        "off_field_event_probability": float(off_field_event_probability),
        "off_field_max_missed_games": int(off_field_max_missed_games),
        "reason": str(reason),
        "material": str(tier) in {"red", "orange", "yellow"},
        "resolved": str(tier) == "green",
    }


def _item_text(item: dict) -> str:
    return " ".join(
        [
            str(item.get("title") or ""),
            str(item.get("desc") or ""),
            str(item.get("impact") or ""),
        ]
    ).strip()


def _item_timestamp(item: dict) -> pd.Timestamp | None:
    raw = item.get("updated") or item.get("created") or item.get("published")
    if not raw:
        return None
    try:
        ts = pd.Timestamp(raw)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts
    except Exception:
        return None


def select_draft_news_signals(
    items: list[dict],
    *,
    fp_ids: set[str],
    fp_id_to_name: dict[str, str],
    max_age_days: int = 14,
    now: pd.Timestamp | None = None,
) -> tuple[dict[str, dict], dict]:
    """Choose the newest relevant availability signal for each known player.

    Player IDs are preferred. If FantasyPros omitted/mis-tagged player_id, a
    conservative full-name match across title/description/impact is allowed.
    Because items are processed newest-first, a GREEN resolution prevents an
    older RED/ORANGE story from lingering as an active draft penalty.
    """
    known_ids = {str(x) for x in fp_ids if str(x)}
    id_to_name = {str(k): str(v) for k, v in fp_id_to_name.items() if str(k) in known_ids}
    normalized_name_to_id = {
        normalize_name(name): pid
        for pid, name in id_to_name.items()
        if normalize_name(name)
    }

    current = now if now is not None else pd.Timestamp.now(tz="UTC")
    if current.tzinfo is None:
        current = current.tz_localize("UTC")
    else:
        current = current.tz_convert("UTC")

    recent = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        ts = _item_timestamp(item)
        if ts is not None:
            age_days = (current - ts).total_seconds() / 86400.0
            if age_days > max(0, int(max_age_days)):
                continue
        recent.append((ts, item))

    recent.sort(
        key=lambda pair: pair[0] if pair[0] is not None else pd.Timestamp("1970-01-01", tz="UTC"),
        reverse=True,
    )

    chosen: dict[str, dict] = {}
    fallback_matches = 0
    classified = 0
    resolved = 0

    for ts, item in recent:
        text = _item_text(item)
        signal = classify_draft_news(text)
        if signal["tier"] == "none":
            continue

        pid_raw = item.get("player_id")
        pid = str(pid_raw) if pid_raw is not None else ""
        if pid not in known_ids:
            pid = _known_name_match(text, normalized_name_to_id) or ""
            if pid:
                fallback_matches += 1
        if pid not in known_ids or pid in chosen:
            continue

        classified += 1
        if signal["resolved"]:
            resolved += 1

        chosen[pid] = {
            **signal,
            "title": str(item.get("title") or ""),
            "created": str(item.get("created") or item.get("updated") or ""),
            "matched_by": "player_id" if str(pid_raw or "") == pid else "player_name",
        }

    active = {pid: item for pid, item in chosen.items() if not item.get("resolved")}
    return active, {
        "ok": True,
        "matched": int(len(active)),
        "classified_players": int(classified),
        "resolved_players": int(resolved),
        "name_fallback_matches": int(fallback_matches),
        "recent_items_checked": int(len(recent)),
    }


def fetch_draft_news(players: pd.DataFrame, config: dict) -> tuple[dict[str, dict], dict]:
    """Fetch a broad FantasyPros news window, then classify locally for materiality."""
    fp_id_to_name = {}
    for _, row in players.iterrows():
        pid = str(row.get("fp_player_id") or "").strip()
        if pid.endswith(".0") and pid[:-2].isdigit():
            pid = pid[:-2]
        name = str(row.get("name") or "").strip()
        if pid and pid.lower() not in {"nan", "none"} and name:
            fp_id_to_name[pid] = name

    fp_ids = set(fp_id_to_name)
    if not fp_ids:
        return {}, {"ok": False, "error": "No cached FantasyPros player IDs in snapshot."}

    spacing = float(config.get("risk_model", {}).get("fantasypros_request_spacing_seconds", 1.05))
    if spacing > 0:
        time.sleep(spacing)

    limit = max(100, int(config.get("risk_model", {}).get("draft_news_fetch_limit", 500)))
    max_age_days = max(1, int(config.get("risk_model", {}).get("draft_news_max_age_days", 14)))
    try:
        data = _fp_get("nfl/news", {"limit": limit, "order_by": "updated"})
    except Exception as exc:
        return {}, {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    items = data.get("items") or []
    signals, meta = select_draft_news_signals(
        items,
        fp_ids=fp_ids,
        fp_id_to_name=fp_id_to_name,
        max_age_days=max_age_days,
    )
    tier_counts = {tier: 0 for tier in ("red", "orange", "yellow")}
    for item in signals.values():
        tier = str(item.get("tier", ""))
        if tier in tier_counts:
            tier_counts[tier] += 1
    return signals, {
        **meta,
        "items_returned": int(len(items)),
        "requested_limit": int(limit),
        "max_age_days": int(max_age_days),
        "tier_counts": tier_counts,
    }
