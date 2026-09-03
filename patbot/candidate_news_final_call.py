from __future__ import annotations

import pandas as pd

from . import final_call as _final_call
from .candidate_news import apply_candidate_news_signals, verify_candidate_news


_ORIGINAL_RUN_FINAL_CALL = _final_call.run_final_call


def _roster_positions(engine, my_roster_ids: list[str]) -> list[str]:
    ids = {str(x) for x in my_roster_ids}
    frame = engine.players.copy()
    frame["player_id"] = frame["player_id"].astype(str)
    return frame.loc[frame["player_id"].isin(ids), "pos"].astype(str).tolist()


def _has_live_fp_ids(engine, candidate_ids: list[str]) -> bool:
    frame = engine.players.copy()
    if "fp_player_id" not in frame.columns:
        return False
    frame["player_id"] = frame["player_id"].astype(str)
    subset = frame[frame["player_id"].isin({str(x) for x in candidate_ids})]
    if subset.empty:
        return False
    values = subset["fp_player_id"].astype(str).str.strip().str.lower()
    return bool((~values.isin({"", "nan", "none", "null"})).any())


def _call_original(
    engine,
    *,
    current_pick: int,
    drafted_ids: set[str],
    my_roster_ids: list[str],
    board: pd.DataFrame,
    draft_history: list[dict] | None,
    compare_fn,
    stability_fn,
):
    kwargs = {}
    if compare_fn is not None:
        kwargs["compare_fn"] = compare_fn
    if stability_fn is not None:
        kwargs["stability_fn"] = stability_fn
    return _ORIGINAL_RUN_FINAL_CALL(
        engine,
        current_pick=current_pick,
        drafted_ids=drafted_ids,
        my_roster_ids=my_roster_ids,
        board=board,
        draft_history=draft_history,
        **kwargs,
    )


def _merge_meta(first: dict, second: dict | None = None) -> dict:
    second = second or {}
    return {
        "ok": bool(first.get("ok", True)) and bool(second.get("ok", True)),
        "checked": int(first.get("checked", 0)) + int(second.get("checked", 0)),
        "requested": int(first.get("requested", 0)) + int(second.get("requested", 0)),
        "cache_hits": int(first.get("cache_hits", 0)) + int(second.get("cache_hits", 0)),
        "api_calls": int(first.get("api_calls", 0)) + int(second.get("api_calls", 0)),
        "failures": int(first.get("failures", 0)) + int(second.get("failures", 0)),
        "ttl_seconds": int(first.get("ttl_seconds", second.get("ttl_seconds", 600))),
    }


def run_final_call(
    engine,
    *,
    current_pick: int,
    drafted_ids: set[str],
    my_roster_ids: list[str],
    board: pd.DataFrame,
    draft_history: list[dict] | None = None,
    compare_fn=None,
    stability_fn=None,
) -> dict:
    """Verify the live Final Call candidate set directly before simulating it.

    The first pass verifies the top score-board candidates. PatBot then rebuilds
    the board with those risk signals. If the rebuild promotes a new candidate
    into the verification window, a second bounded pass checks only those newly
    promoted names. The underlying Final Call thresholds and simulation code are
    unchanged. Synthetic/test boards without FantasyPros IDs bypass this layer
    and preserve the original Final Call behavior exactly.
    """
    if board is None or board.empty:
        return _call_original(
            engine,
            current_pick=current_pick,
            drafted_ids=drafted_ids,
            my_roster_ids=my_roster_ids,
            board=board,
            draft_history=draft_history,
            compare_fn=compare_fn,
            stability_fn=stability_fn,
        )

    rcfg = engine.config.get("risk_model", {}) or {}
    verify_count = max(2, int(rcfg.get("candidate_news_verify_count", 8)))
    top_ids = board.head(verify_count)["player_id"].astype(str).tolist()
    if not _has_live_fp_ids(engine, top_ids):
        return _call_original(
            engine,
            current_pick=current_pick,
            drafted_ids=drafted_ids,
            my_roster_ids=my_roster_ids,
            board=board,
            draft_history=draft_history,
            compare_fn=compare_fn,
            stability_fn=stability_fn,
        )

    first_signals, first_meta = verify_candidate_news(
        engine.players,
        top_ids,
        engine.config,
    )
    all_signals = dict(first_signals)

    verified_players = apply_candidate_news_signals(
        engine.players,
        all_signals,
        engine.config,
    )
    verified_engine = engine.__class__(verified_players, engine.config)
    verified_board = verified_engine.recommend(
        current_pick=int(current_pick),
        drafted_ids={str(x) for x in drafted_ids},
        roster_positions=_roster_positions(verified_engine, my_roster_ids),
        top_n=max(18, len(board)),
    )

    second_meta = None
    if not verified_board.empty:
        rebuilt_ids = verified_board.head(verify_count)["player_id"].astype(str).tolist()
        new_ids = [pid for pid in rebuilt_ids if pid not in set(top_ids)]
        if new_ids:
            second_signals, second_meta = verify_candidate_news(
                engine.players,
                new_ids,
                engine.config,
            )
            if second_signals:
                all_signals.update(second_signals)
                verified_players = apply_candidate_news_signals(
                    engine.players,
                    all_signals,
                    engine.config,
                )
                verified_engine = engine.__class__(verified_players, engine.config)
                verified_board = verified_engine.recommend(
                    current_pick=int(current_pick),
                    drafted_ids={str(x) for x in drafted_ids},
                    roster_positions=_roster_positions(verified_engine, my_roster_ids),
                    top_n=max(18, len(board)),
                )

    result = _call_original(
        verified_engine,
        current_pick=current_pick,
        drafted_ids=drafted_ids,
        my_roster_ids=my_roster_ids,
        board=verified_board,
        draft_history=draft_history,
        compare_fn=compare_fn,
        stability_fn=stability_fn,
    )
    if not isinstance(result, dict):
        return result

    meta = _merge_meta(first_meta, second_meta)
    meta["active_signals"] = {
        pid: {
            "tier": str(sig.get("tier") or "none"),
            "title": str(sig.get("title") or ""),
            "created": str(sig.get("created") or ""),
        }
        for pid, sig in all_signals.items()
        if str(sig.get("tier") or "none").lower() in {"red", "orange", "yellow"}
    }
    result["candidate_news_verification"] = meta

    prefix = (
        f"Direct FantasyPros news check: {meta['checked']} candidates "
        f"({meta['cache_hits']} cached, {meta['api_calls']} API calls). "
    )
    if meta["failures"]:
        prefix += f"{meta['failures']} verification failures; manual/fast-risk backstops remain active. "
    result["reason"] = prefix + str(result.get("reason") or "")
    return result


def install_candidate_news_final_call_patch() -> None:
    _final_call.run_final_call = run_final_call
