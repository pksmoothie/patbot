from __future__ import annotations

import time

import pandas as pd

from .market import _fp_get, _known_name_match, normalize_name


OFFENSE_POSITIONS = {"QB", "RB", "WR", "TE"}

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
    "expected to play week 1",
    "expected to play sunday",
    "expected to play monday",
    "expected to play thursday",
    "will play week 1",
    "will play sunday",
    "will play monday",
    "will play thursday",
    "on track to play week 1",
    "no suspension expected",
    "not expected to be suspended",
    "no discipline expected",
    "will not be suspended",
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
    # get misclassified merely because the sentence also contains "PUP". Keep
    # these phrases specific: "hope he can play sometime this season" is not a
    # resolution of a current exempt-list or injury issue.
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


def _clean_fp_id(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    pid = str(value).strip()
    if pid.endswith(".0") and pid[:-2].isdigit():
        pid = pid[:-2]
    return "" if pid.lower() in {"nan", "none", "null"} else pid


def _priority_fp_ids(players: pd.DataFrame, limit: int) -> list[str]:
    """Return the offensive players most likely to matter to a PatBot decision.

    A player stays priority if *any* major signal still likes him: projection
    rank, ADP, expert rank or market rank. That is deliberate. A player whose
    market rank collapses because of breaking news is exactly the player whose
    stale projection could otherwise create a false PatBot value.
    """
    n = max(0, int(limit))
    if n == 0 or players is None or players.empty:
        return []

    frame = players.copy()
    frame = frame[frame["pos"].astype(str).str.upper().isin(OFFENSE_POSITIONS)].copy()
    if frame.empty:
        return []

    frame["_fp_id"] = frame.get("fp_player_id", pd.Series(index=frame.index, dtype=object)).apply(_clean_fp_id)
    frame = frame[frame["_fp_id"].ne("")].copy()
    if frame.empty:
        return []

    proj = pd.to_numeric(frame.get("proj_points"), errors="coerce")
    proj_rank = proj.rank(method="min", ascending=False)
    signals = [proj_rank]
    for col in ("adp", "expert_rank", "market_rank"):
        if col in frame.columns:
            values = pd.to_numeric(frame[col], errors="coerce")
            values = values.where(values > 0)
            signals.append(values)

    frame["_priority"] = pd.concat(signals, axis=1).min(axis=1, skipna=True).fillna(9999.0)
    frame["_projection"] = proj.fillna(-1.0)
    ordered = frame.sort_values(["_priority", "_projection"], ascending=[True, False])
    return ordered["_fp_id"].drop_duplicates().head(n).astype(str).tolist()


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
    known_ids = {_clean_fp_id(x) for x in fp_ids}
    known_ids.discard("")
    id_to_name = {
        _clean_fp_id(k): str(v)
        for k, v in fp_id_to_name.items()
        if _clean_fp_id(k) in known_ids
    }
    normalized_name_to_id = {
        normalize_name(name): pid
        for pid, name in id_to_name.items()
        if normalize_name(name)
    }

    current = now if now is not None else pd.Timestamp.now(tz="UTC")
    if not isinstance(current, pd.Timestamp):
        current = pd.Timestamp(current)
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

    for _ts, item in recent:
        text = _item_text(item)
        signal = classify_draft_news(text)
        if signal["tier"] == "none":
            continue

        pid_raw = _clean_fp_id(item.get("player_id"))
        pid = pid_raw
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
            "matched_by": "player_id" if pid_raw == pid else "player_name",
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
    """Fetch broad news plus player-specific checks for draft-critical offense.

    Broad category slices provide cheap league-wide coverage. FantasyPros also
    supports an `fpid` filter on /nfl/news, so PatBot follows those broad calls
    with a bounded player-specific pass for the offensive assets most likely to
    matter to a draft decision. K/DST are intentionally excluded from this
    player-news layer; their individual personnel transactions should not become
    fake availability alerts for a team defense.
    """
    offense = players[players["pos"].astype(str).str.upper().isin(OFFENSE_POSITIONS)].copy()
    fp_id_to_name = {}
    for _, row in offense.iterrows():
        pid = _clean_fp_id(row.get("fp_player_id"))
        name = str(row.get("name") or "").strip()
        if pid and name:
            fp_id_to_name[pid] = name

    fp_ids = set(fp_id_to_name)
    if not fp_ids:
        return {}, {"ok": False, "error": "No cached FantasyPros offensive player IDs in snapshot."}

    rcfg = config.get("risk_model", {})
    spacing = float(rcfg.get("fantasypros_request_spacing_seconds", 1.05))
    category_limit = max(25, int(rcfg.get("draft_news_category_limit", 100)))
    player_limit = max(1, int(rcfg.get("draft_news_player_limit", 5)))
    priority_limit = max(0, int(rcfg.get("draft_news_priority_player_limit", 48)))
    max_age_days = max(1, int(rcfg.get("draft_news_max_age_days", 14)))
    categories = [None, "injury", "transaction", "rumor", "breaking"]

    items_by_key: dict[str, dict] = {}
    source_status = {}
    successful_calls = 0

    def store_batch(batch) -> int:
        stored = 0
        for item in batch or []:
            if not isinstance(item, dict):
                continue
            key = str(item.get("id") or "").strip()
            if not key:
                key = "|".join(
                    [
                        str(item.get("created") or ""),
                        str(item.get("title") or ""),
                        str(item.get("player_id") or ""),
                    ]
                )
            if key not in items_by_key:
                stored += 1
            items_by_key[key] = item
        return stored

    for idx, category in enumerate(categories):
        if idx > 0 and spacing > 0:
            time.sleep(spacing)
        params = {"limit": category_limit}
        if category:
            params["category"] = category
        label = category or "all"
        try:
            data = _fp_get("nfl/news", params)
            batch = data.get("items") or []
            successful_calls += 1
            source_status[label] = {"ok": True, "items": int(len(batch))}
            store_batch(batch)
        except Exception as exc:
            source_status[label] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    priority_ids = _priority_fp_ids(offense, priority_limit)
    player_calls_ok = 0
    player_calls_failed = 0
    player_items = 0
    for pid in priority_ids:
        if spacing > 0:
            time.sleep(spacing)
        try:
            data = _fp_get("nfl/news", {"fpid": pid, "limit": player_limit})
            batch = data.get("items") or []
            successful_calls += 1
            player_calls_ok += 1
            player_items += len(batch)
            store_batch(batch)
        except Exception:
            player_calls_failed += 1

    source_status["priority_player_news"] = {
        "ok": player_calls_ok > 0 or not priority_ids,
        "players_requested": int(len(priority_ids)),
        "players_ok": int(player_calls_ok),
        "players_failed": int(player_calls_failed),
        "items": int(player_items),
    }

    if not successful_calls:
        errors = "; ".join(
            f"{name}: {meta.get('error', 'failed')}"
            for name, meta in source_status.items()
            if not meta.get("ok")
        )
        return {}, {"ok": False, "error": errors or "FantasyPros news calls failed."}

    items = list(items_by_key.values())
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
        "source_status": source_status,
        "successful_calls": int(successful_calls),
        "unique_items_returned": int(len(items)),
        "category_limit": int(category_limit),
        "priority_player_limit": int(priority_limit),
        "player_news_limit": int(player_limit),
        "max_age_days": int(max_age_days),
        "tier_counts": tier_counts,
    }
