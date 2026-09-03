from __future__ import annotations

import time

import pandas as pd

from . import draft_news as _draft_news
from . import fast_risk as _fast_risk
from .market import _fp_get


_ORIGINAL_CLASSIFY = _draft_news.classify_draft_news

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


def fetch_draft_news(players: pd.DataFrame, config: dict) -> tuple[dict[str, dict], dict]:
    """Keep Fast Refresh league-wide and cheap; direct checks happen in Final Call.

    FantasyPros' broad news endpoint is useful for fresh league-wide alerts but
    is not a dependable archive. Fast Refresh therefore scans a bounded set of
    current categories only. Player-specific fpid lookups are reserved for the
    live Final Call candidate set, where completeness actually changes the pick.
    """
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
        return {}, {"ok": False, "error": "No cached FantasyPros offensive player IDs in snapshot."}

    rcfg = config.get("risk_model", {}) or {}
    spacing = max(0.0, float(rcfg.get("fantasypros_request_spacing_seconds", 1.05)))
    category_limit = max(25, int(rcfg.get("draft_news_category_limit", 100)))
    max_age_days = max(1, int(rcfg.get("draft_news_max_age_days", 14)))
    categories = [None, "injury", "transaction", "rumor", "breaking"]

    items_by_key: dict[str, dict] = {}
    source_status = {}
    successful_calls = 0

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
            for item in batch:
                if not isinstance(item, dict):
                    continue
                key = str(item.get("id") or "").strip()
                if not key:
                    key = "|".join(
                        [
                            str(item.get("created") or item.get("updated") or ""),
                            str(item.get("title") or ""),
                            str(item.get("player_id") or ""),
                        ]
                    )
                items_by_key[key] = item
        except Exception as exc:
            source_status[label] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    if not successful_calls:
        errors = "; ".join(
            f"{name}: {meta.get('error', 'failed')}"
            for name, meta in source_status.items()
        )
        return {}, {"ok": False, "error": errors or "FantasyPros news calls failed."}

    items = list(items_by_key.values())
    signals, meta = _draft_news.select_draft_news_signals(
        items,
        fp_ids=fp_ids,
        fp_id_to_name=fp_id_to_name,
        max_age_days=max_age_days,
    )
    tier_counts = {"red": 0, "orange": 0, "yellow": 0}
    for signal in signals.values():
        tier = str(signal.get("tier") or "none")
        if tier in tier_counts:
            tier_counts[tier] += 1

    return signals, {
        **meta,
        "source_status": source_status,
        "successful_calls": int(successful_calls),
        "unique_items_returned": int(len(items)),
        "category_limit": int(category_limit),
        "max_age_days": int(max_age_days),
        "tier_counts": tier_counts,
        "note": "Broad current-news scan only; player-specific fpid verification is deferred to live Final Call candidates.",
    }


def install_draft_news_hardening_patch() -> None:
    _draft_news.classify_draft_news = classify_draft_news
    _draft_news.fetch_draft_news = fetch_draft_news
    # fast_risk imports fetch_draft_news directly, so update that module binding
    # as well as the source module.
    _fast_risk.fetch_draft_news = fetch_draft_news
