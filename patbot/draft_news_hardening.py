from __future__ import annotations

import time

import pandas as pd

from . import draft_news as _draft_news
from . import fast_risk as _fast_risk
from .market import _fp_get, _known_name_match, normalize_name


_ORIGINAL_CLASSIFY = _draft_news.classify_draft_news
_ORIGINAL_FETCH = _draft_news.fetch_draft_news

_NEGATED_HARD_PHRASES = (
    "wasn't placed on ir",
    "was not placed on ir",
    "not placed on ir",
    "wasn't placed on injured reserve",
    "was not placed on injured reserve",
    "not placed on injured reserve",
    "avoided being placed on injured reserve",
    "avoided injured reserve",
    "avoided ir",
    "wasn't placed on pup",
    "was not placed on pup",
    "not placed on pup",
    "wasn't placed on nfi",
    "was not placed on nfi",
    "not placed on nfi",
)


def classify_draft_news(text: str) -> dict:
    """Protect hard-status matching from obvious negation phrases.

    Example: "wasn't placed on IR" is evidence against a confirmed IR stint,
    not evidence that the player was placed on IR. The remaining sentence can
    still classify as ORANGE if it contains a current practice/availability
    concern.
    """
    cleaned = str(text or "").lower()
    for phrase in _NEGATED_HARD_PHRASES:
        cleaned = cleaned.replace(phrase, "avoided long-term reserve")
    return _ORIGINAL_CLASSIFY(cleaned)


def _clean_fp_id(value) -> str:
    return _draft_news._clean_fp_id(value)


def _timestamp(value) -> pd.Timestamp:
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts
    except Exception:
        return pd.Timestamp("1970-01-01", tz="UTC")


def _newest_relevant_by_player(
    items: list[dict],
    *,
    fp_ids: set[str],
    fp_id_to_name: dict[str, str],
    max_age_days: int,
) -> dict[str, dict]:
    """Return newest RED/ORANGE/YELLOW/GREEN signal for each known player."""
    known_ids = {_clean_fp_id(x) for x in fp_ids}
    known_ids.discard("")
    normalized_name_to_id = {
        normalize_name(name): pid
        for pid, name in fp_id_to_name.items()
        if pid in known_ids and normalize_name(name)
    }
    now = pd.Timestamp.now(tz="UTC")

    recent: list[tuple[pd.Timestamp, dict]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        raw_ts = item.get("updated") or item.get("created") or item.get("published")
        ts = _timestamp(raw_ts)
        if raw_ts:
            age_days = (now - ts).total_seconds() / 86400.0
            if age_days > max(1, int(max_age_days)):
                continue
        recent.append((ts, item))
    recent.sort(key=lambda pair: pair[0], reverse=True)

    chosen: dict[str, dict] = {}
    for ts, item in recent:
        text = " ".join(
            [
                str(item.get("title") or ""),
                str(item.get("desc") or ""),
                str(item.get("impact") or ""),
            ]
        ).strip()
        signal = classify_draft_news(text)
        if signal.get("tier") == "none":
            continue

        pid = _clean_fp_id(item.get("player_id"))
        matched_by = "player_id"
        if pid not in known_ids:
            pid = _known_name_match(text, normalized_name_to_id) or ""
            matched_by = "player_name"
        if pid not in known_ids or pid in chosen:
            continue

        chosen[pid] = {
            **signal,
            "title": str(item.get("title") or ""),
            "created": str(item.get("created") or item.get("updated") or ""),
            "matched_by": matched_by,
            "_ts": ts,
        }
    return chosen


def fetch_draft_news(players: pd.DataFrame, config: dict) -> tuple[dict[str, dict], dict]:
    """Add one wide league news pull to the existing bounded fast-refresh scan.

    FantasyPros documents `limit` on /nfl/news. A 500-item all-news request is
    cheap in wall-clock terms (one rate-limited request) and protects PatBot from
    missing a draft-relevant player whose market rank already collapsed enough
    to fall outside the player-specific priority set.
    """
    existing, meta = _ORIGINAL_FETCH(players, config)

    offense = players[
        players["pos"].astype(str).str.upper().isin(_draft_news.OFFENSE_POSITIONS)
    ].copy()
    fp_id_to_name: dict[str, str] = {}
    for _, row in offense.iterrows():
        pid = _clean_fp_id(row.get("fp_player_id"))
        name = str(row.get("name") or "").strip()
        if pid and name:
            fp_id_to_name[pid] = name
    fp_ids = set(fp_id_to_name)
    if not fp_ids:
        return existing, meta

    rcfg = config.get("risk_model", {})
    spacing = float(rcfg.get("fantasypros_request_spacing_seconds", 1.05))
    if spacing > 0:
        time.sleep(spacing)
    wide_limit = max(100, int(rcfg.get("draft_news_wide_limit", 500)))
    max_age_days = max(1, int(rcfg.get("draft_news_max_age_days", 14)))

    try:
        data = _fp_get("nfl/news", {"limit": wide_limit})
        items = data.get("items") or []
    except Exception as exc:
        out_meta = dict(meta or {})
        out_meta["wide_news"] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "requested_limit": int(wide_limit),
        }
        return existing, out_meta

    wide = _newest_relevant_by_player(
        items,
        fp_ids=fp_ids,
        fp_id_to_name=fp_id_to_name,
        max_age_days=max_age_days,
    )

    merged = dict(existing or {})
    resolved = 0
    overrides = 0
    for pid, signal in wide.items():
        if signal.get("resolved"):
            prior = merged.get(pid)
            if prior is not None and signal["_ts"] >= _timestamp(prior.get("created")):
                merged.pop(pid, None)
                resolved += 1
            continue

        prior = merged.get(pid)
        if prior is None or signal["_ts"] >= _timestamp(prior.get("created")):
            cleaned = {k: v for k, v in signal.items() if k != "_ts"}
            merged[pid] = cleaned
            overrides += 1

    out_meta = dict(meta or {})
    out_meta["wide_news"] = {
        "ok": True,
        "requested_limit": int(wide_limit),
        "items": int(len(items)),
        "relevant_players": int(len(wide)),
        "active_overrides": int(overrides),
        "resolved_overrides": int(resolved),
    }
    tier_counts = {"red": 0, "orange": 0, "yellow": 0}
    for signal in merged.values():
        tier = str(signal.get("tier") or "")
        if tier in tier_counts:
            tier_counts[tier] += 1
    out_meta["tier_counts"] = tier_counts
    out_meta["matched"] = int(len(merged))
    return merged, out_meta


def install_draft_news_hardening_patch() -> None:
    _draft_news.classify_draft_news = classify_draft_news
    _draft_news.fetch_draft_news = fetch_draft_news
    # fast_risk imports fetch_draft_news directly, so update that module binding
    # as well as the source module.
    _fast_risk.fetch_draft_news = fetch_draft_news
