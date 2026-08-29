from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .sleeper import refresh_snapshot


DEFAULT_YAHOO_CACHE_PATH = Path("data/yahoo_adp_2026.csv")


def _write_meta(meta_path: str | Path, meta: dict) -> None:
    Path(meta_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def attach_projection_sources(csv_path, meta_path, meta: dict, cfg: dict) -> dict:
    """Attach FantasyPros full-stat projections and rebuild the production blend."""
    enabled = bool(cfg.get("projection_sources", {}).get("fantasypros_preseason", True))
    if not enabled:
        meta["projection_sources"] = {
            "fantasypros_preseason_projections": {
                "ok": False,
                "error": "Disabled by projection_sources.fantasypros_preseason",
            }
        }
        _write_meta(meta_path, meta)
        return meta

    try:
        from .consensus import add_consensus_values
        from .fantasypros_projection import augment_fantasypros_projections
        from .projection_blend import blend_projection_sources

        frame = pd.read_csv(csv_path)
        frame, fp_status = augment_fantasypros_projections(frame, cfg)
        frame, blend_status = blend_projection_sources(frame, cfg)
        frame = add_consensus_values(frame, cfg)
        frame.to_csv(csv_path, index=False)
        meta["projection_sources"] = {**fp_status, **blend_status}
    except Exception as exc:
        meta["projection_sources"] = {
            "projection_pipeline": {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        }

    _write_meta(meta_path, meta)
    return meta


def refresh_yahoo_room_signal(
    csv_path,
    meta_path,
    meta: dict,
    cfg: dict,
    *,
    cache_path: str | Path = DEFAULT_YAHOO_CACHE_PATH,
) -> dict:
    """Refresh Yahoo ADP separately for opponent/survival modeling.

    Yahoo never gets merged into PatBot's intrinsic player-valuation fields. If
    the refresh fails, the prior cache is retained so the room model can use it
    only while it remains inside its configured freshness window.
    """
    cache = Path(cache_path)
    enabled = bool(cfg.get("yahoo_room_behavior", {}).get("enabled", True))
    if not enabled:
        meta["yahoo_room_behavior"] = {
            "yahoo_adp": {"ok": False, "reason": "disabled"}
        }
        _write_meta(meta_path, meta)
        return meta

    try:
        from .yahoo_adp import fetch_yahoo_adp

        frame = pd.read_csv(csv_path)
        names = frame["name"].dropna().astype(str).tolist()
        yahoo, status = fetch_yahoo_adp(names)
        cache.parent.mkdir(parents=True, exist_ok=True)
        yahoo.to_csv(cache, index=False)
        status = dict(status)
        status["file"] = str(cache)
        status["coverage_pct"] = round(100.0 * len(yahoo) / max(len(frame), 1), 1)
        meta["yahoo_room_behavior"] = {"yahoo_adp": status}
    except Exception as exc:
        prior = cache.exists()
        meta["yahoo_room_behavior"] = {
            "yahoo_adp": {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "warning": "Prior Yahoo cache retained." if prior else "No prior Yahoo cache available.",
            }
        }

    _write_meta(meta_path, meta)
    return meta


def run_full_refresh(
    cfg: dict,
    csv_path: str | Path = "data/players_2026_live.csv",
    meta_path: str | Path = "data/players_2026_live.meta.json",
    *,
    yahoo_cache_path: str | Path = DEFAULT_YAHOO_CACHE_PATH,
):
    """Run the exact production slow-refresh chain used by CLI and Streamlit."""
    csv_path, meta_path, meta = refresh_snapshot(cfg, csv_path, meta_path)
    meta = attach_projection_sources(csv_path, meta_path, meta, cfg)
    meta = refresh_yahoo_room_signal(
        csv_path,
        meta_path,
        meta,
        cfg,
        cache_path=yahoo_cache_path,
    )
    return Path(csv_path), Path(meta_path), meta
