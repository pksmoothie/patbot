from __future__ import annotations

from . import candidate_news as _candidate_news


_ORIGINAL_OVERLAY_ROW = _candidate_news._overlay_row


def _overlay_row(row, *, config: dict, direct_signal: dict | None = None) -> dict:
    """Do not let generic GREEN news erase an undated manual uncertainty flag.

    A configured manual override such as Puka Nacua's legal/discipline monitor
    represents a different risk channel from a routine injury-resolution story.
    Only dated backstops with an explicit as_of/as_of_utc can be superseded by a
    newer GREEN direct-player signal.
    """
    signal = dict(direct_signal or {})
    if str(signal.get("tier") or "none").lower() == "green":
        name = str(row.get("name") or "")
        configured = (config.get("risk_overrides", {}) or {}).get(name) or {}
        if (
            isinstance(configured, dict)
            and configured
            and not configured.get("as_of_utc")
            and not configured.get("as_of")
        ):
            return _ORIGINAL_OVERLAY_ROW(
                row,
                config=config,
                direct_signal=None,
            )
    return _ORIGINAL_OVERLAY_ROW(
        row,
        config=config,
        direct_signal=direct_signal,
    )


def install_candidate_news_manual_guard() -> None:
    _candidate_news._overlay_row = _overlay_row
