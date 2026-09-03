from __future__ import annotations

import pandas as pd

from . import candidate_news as _candidate_news
from . import candidate_news_final_call as _candidate_final_call


_ORIGINAL_APPLY_MANUAL = _candidate_news.apply_manual_risk_overrides
_ORIGINAL_APPLY_SIGNALS = _candidate_news.apply_candidate_news_signals

_TEXT_COLUMNS = (
    "current_status_source",
    "current_alert_tier",
    "fast_news_tier",
    "fast_news_title",
    "fast_news_reason",
    "fast_news_created",
    "off_field_risk_level",
    "risk_note",
    "candidate_news_verified_at_utc",
)


def _prepare_overlay_dtypes(players: pd.DataFrame) -> pd.DataFrame:
    """Make text/bool overlay columns safe after CSV round-trips.

    Pandas infers an all-empty CSV column as float64. Draft-night risk overlays
    then need to write strings such as ``manual_red`` or ``RED — ...`` into
    those columns. Explicit object dtypes prevent the strict assignment error
    while preserving the existing numeric risk columns.
    """
    out = players.copy()
    for col in _TEXT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.Series("", index=out.index, dtype="object")
        else:
            out[col] = out[col].astype("object")
            out.loc[pd.isna(out[col]), col] = ""
    if "current_status_material" not in out.columns:
        out["current_status_material"] = False
    else:
        out["current_status_material"] = out["current_status_material"].fillna(False).astype(bool)
    return out


def apply_manual_risk_overrides(players: pd.DataFrame, config: dict) -> pd.DataFrame:
    return _ORIGINAL_APPLY_MANUAL(_prepare_overlay_dtypes(players), config)


def apply_candidate_news_signals(
    players: pd.DataFrame,
    signals: dict[str, dict],
    config: dict,
) -> pd.DataFrame:
    prepared = _prepare_overlay_dtypes(players)
    return _ORIGINAL_APPLY_SIGNALS(prepared, signals, config)


def install_candidate_news_dtype_guard() -> None:
    _candidate_news.apply_manual_risk_overrides = apply_manual_risk_overrides
    _candidate_news.apply_candidate_news_signals = apply_candidate_news_signals
    # candidate_news_final_call imports this function directly, so replace that
    # local binding as well.
    _candidate_final_call.apply_candidate_news_signals = apply_candidate_news_signals
