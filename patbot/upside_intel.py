from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import math

import pandas as pd
import yaml

from .market import normalize_name


def _clamp01(value, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(default)
    return max(0.0, min(1.0, number))


def _parse_date(value) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        try:
            return datetime.fromisoformat(text).date()
        except ValueError:
            return None


def load_upside_evidence(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {"sources": {}, "evidence": []}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sources = raw.get("sources") or {}
    evidence = raw.get("evidence") or []
    if not isinstance(sources, dict):
        sources = {}
    if not isinstance(evidence, list):
        evidence = []
    return {"sources": sources, "evidence": evidence}


def score_upside_evidence(
    players: pd.DataFrame,
    config: dict,
    *,
    evidence_path: str | Path | None = None,
    as_of: date | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Attach a conservative, positive-only expert sleeper/upside signal.

    The signal is deliberately separate from projections, VORP and market rank.
    Multiple articles from the same publisher count as one independent source;
    the strongest item from that source wins. Production eligibility requires
    corroboration across independent publishers. v0.5.6 is diagnostic only: the
    computed LWS bonus is reported but is not yet applied by the draft engine.
    """
    out = players.copy()
    scfg = config.get("championship_strategy", {}).get("expert_upside_intel", {})
    enabled = bool(scfg.get("enabled", True))
    if evidence_path is None:
        evidence_path = scfg.get("evidence_path", "config/upside_evidence.yaml")
    payload = load_upside_evidence(evidence_path)
    sources = payload["sources"]
    evidence = payload["evidence"]

    if as_of is None:
        as_of = date.today()

    half_life = max(1.0, float(scfg.get("recency_half_life_days", 21.0)))
    min_sources = max(1, int(scfg.get("minimum_independent_sources", 2)))
    max_bonus = max(0.0, float(scfg.get("max_lws_bonus", 8.0)))

    known = {normalize_name(str(name)): str(name) for name in out["name"].astype(str)}
    by_player: dict[str, dict[str, dict]] = {}
    unmatched = []
    used_items = 0

    if enabled:
        for item in evidence:
            if not isinstance(item, dict):
                continue
            raw_name = str(item.get("player") or "").strip()
            matched_name = known.get(normalize_name(raw_name))
            if not matched_name:
                if raw_name:
                    unmatched.append(raw_name)
                continue

            source_id = str(item.get("source") or "").strip()
            source = sources.get(source_id) or {}
            if not isinstance(source, dict):
                source = {}
            independence_key = str(
                source.get("independence_key")
                or source.get("publisher")
                or source_id
                or "unknown"
            ).strip().lower()
            if not independence_key:
                independence_key = "unknown"

            quality = _clamp01(source.get("quality"), 0.70)
            strength = _clamp01(item.get("strength"), 0.75)
            specificity = _clamp01(item.get("specificity"), 0.75)
            objective = bool(item.get("objective_support", False))

            published = _parse_date(source.get("published") or item.get("published"))
            age_days = max(0, (as_of - published).days) if published else 30
            recency = math.pow(0.5, age_days / half_life)
            objective_mult = 1.05 if objective else 1.00
            item_score = quality * strength * (0.75 + 0.25 * specificity) * recency * objective_mult
            item_score = _clamp01(item_score)

            record = {
                "source_id": source_id,
                "publisher": str(source.get("publisher") or source_id or "Unknown"),
                "analyst": str(source.get("analyst") or ""),
                "published": published.isoformat() if published else "",
                "item_score": item_score,
                "objective_support": objective,
                "signal": str(item.get("signal") or ""),
                "note": str(item.get("note") or ""),
            }
            bucket = by_player.setdefault(matched_name, {})
            current = bucket.get(independence_key)
            if current is None or item_score > float(current["item_score"]):
                bucket[independence_key] = record
            used_items += 1

    score_map = {}
    sources_map = {}
    objective_map = {}
    eligible_map = {}
    bonus_map = {}
    note_map = {}

    for name, source_bucket in by_player.items():
        records = sorted(source_bucket.values(), key=lambda x: float(x["item_score"]), reverse=True)
        scores = [float(x["item_score"]) for x in records]
        source_count = len(scores)
        mean_score = sum(scores) / source_count if source_count else 0.0
        corroboration = min(1.0, 0.55 + 0.20 * max(0, source_count - 1))
        intel_score = 100.0 * mean_score * corroboration
        eligible = source_count >= min_sources
        bonus = max_bonus * intel_score / 100.0 if eligible else 0.0

        score_map[name] = round(intel_score, 2)
        sources_map[name] = source_count
        objective_map[name] = sum(1 for x in records if x["objective_support"])
        eligible_map[name] = bool(eligible)
        bonus_map[name] = round(bonus, 3)
        labels = []
        for record in records:
            label = record["publisher"]
            if record["analyst"]:
                label += f" ({record['analyst']})"
            labels.append(label)
        note_map[name] = " | ".join(labels)

    out["expert_upside_score"] = out["name"].map(score_map).fillna(0.0).astype(float)
    out["expert_upside_sources"] = out["name"].map(sources_map).fillna(0).astype(int)
    out["expert_upside_objective_sources"] = out["name"].map(objective_map).fillna(0).astype(int)
    out["expert_upside_eligible"] = out["name"].map(eligible_map).fillna(False).astype(bool)
    out["expert_upside_lws_bonus"] = out["name"].map(bonus_map).fillna(0.0).astype(float)
    out["expert_upside_note"] = out["name"].map(note_map).fillna("").astype(str)

    status = {
        "enabled": enabled,
        "evidence_path": str(evidence_path),
        "evidence_items": int(len(evidence)),
        "matched_items": int(used_items),
        "matched_players": int((out["expert_upside_sources"] > 0).sum()),
        "eligible_players": int(out["expert_upside_eligible"].sum()),
        "minimum_independent_sources": min_sources,
        "max_lws_bonus": max_bonus,
        "unmatched_players": sorted(set(unmatched)),
        "note": (
            "Diagnostic only in v0.5.6. Expert upside intel is positive-only, publisher-deduplicated, "
            "recency-weighted and requires independent corroboration before any future production bonus."
        ),
    }
    return out, status
