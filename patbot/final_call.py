from __future__ import annotations

from time import perf_counter
from typing import Callable

import pandas as pd

from .sim import compare_candidates


def final_call_settings(config: dict) -> dict:
    cfg = config.get("final_call", {}) or {}
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "min_candidates": max(2, int(cfg.get("min_candidates", 3))),
        "max_candidates": max(2, int(cfg.get("max_candidates", 6))),
        "score_gap": max(0.0, float(cfg.get("score_gap", 10.0))),
        "initial_runs": max(50, int(cfg.get("initial_runs", 100))),
        "refine_runs": max(50, int(cfg.get("refine_runs", 300))),
        "final_runs": max(50, int(cfg.get("final_runs", 600))),
        "refine_margin": max(0.0, float(cfg.get("refine_margin", 8.0))),
        "final_margin": max(0.0, float(cfg.get("final_margin", 2.5))),
        "future_rounds": max(1, int(cfg.get("future_rounds", 3))),
        "max_sim_round": max(1, int(cfg.get("max_sim_round", 13))),
        "bypass_round": max(1, int(cfg.get("bypass_round", 14))),
    }


def candidate_shortlist(board: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Choose a compact legal candidate set from the production score board.

    We always inspect at least a few alternatives, but do not spend draft-clock
    time simulating clearly inferior score-board options unless they remain within
    the configured PatBot-score neighborhood of the leader.
    """
    if board is None or board.empty:
        return pd.DataFrame()
    settings = final_call_settings(config)
    ordered = board.copy()
    ordered["score"] = pd.to_numeric(ordered["score"], errors="coerce")
    ordered = ordered.sort_values(["score", "proj_points", "adp"], ascending=[False, False, True])
    ordered = ordered.drop_duplicates(subset=["player_id"], keep="first")

    max_n = min(int(settings["max_candidates"]), len(ordered))
    min_n = min(int(settings["min_candidates"]), max_n)
    top_score = float(ordered.iloc[0]["score"])
    within = ordered[(top_score - ordered["score"]) <= float(settings["score_gap"])].head(max_n)
    if len(within) < min_n:
        within = ordered.head(min_n)
    return within.head(max_n).reset_index(drop=True)


def _margin(summary: pd.DataFrame) -> float:
    if summary is None or len(summary) < 2:
        return float("inf")
    vals = pd.to_numeric(summary["Avg Lineup Score"], errors="coerce")
    return float(vals.iloc[0] - vals.iloc[1])


def _ensure_base_candidate(ids: list[str], base_id: str, limit: int) -> list[str]:
    out: list[str] = []
    for pid in ids:
        text = str(pid)
        if text not in out:
            out.append(text)
    if str(base_id) not in out:
        if len(out) >= int(limit):
            out = out[: max(0, int(limit) - 1)]
        out.append(str(base_id))
    return out[: int(limit)]


def _edge_label(edge: float) -> str:
    if edge >= 10.0:
        return "STRONG"
    if edge >= 5.0:
        return "CLEAR"
    if edge >= 2.0:
        return "LEAN"
    return "CLOSE"


def _decision_payload(
    summary: pd.DataFrame,
    details: list[dict],
    *,
    base_name: str,
    base_id: str,
    stage: str,
    runs: int,
    through_round: int,
    elapsed: float,
    shortlist: pd.DataFrame,
) -> dict:
    winner = summary.iloc[0]
    recommendation = str(winner["Candidate"])
    detail_by_name = {str(x.get("candidate")): x for x in details}
    winner_detail = detail_by_name.get(recommendation, {})
    runner_up = str(summary.iloc[1]["Candidate"]) if len(summary) >= 2 else None
    edge = _margin(summary)
    winner_id = str(winner_detail.get("candidate_id", ""))
    base_agrees = recommendation == str(base_name)
    reason = (
        "Base score board and Yahoo-informed room simulation agree."
        if base_agrees
        else f"Room simulation overturns the base score leader ({base_name})."
    )
    return {
        "ok": True,
        "fallback": False,
        "recommendation": recommendation,
        "candidate_id": winner_id,
        "runner_up": runner_up,
        "edge": round(float(edge), 2) if edge != float("inf") else None,
        "edge_label": _edge_label(float(edge)) if edge != float("inf") else "ONLY OPTION",
        "base_winner": str(base_name),
        "base_winner_id": str(base_id),
        "base_agrees": bool(base_agrees),
        "stage": str(stage),
        "runs": int(runs),
        "through_round": int(through_round),
        "elapsed_seconds": round(float(elapsed), 2),
        "reason": reason,
        "summary": summary.reset_index(drop=True),
        "details": details,
        "shortlist": shortlist.reset_index(drop=True),
    }


def run_final_call(
    engine,
    *,
    current_pick: int,
    drafted_ids: set[str],
    my_roster_ids: list[str],
    board: pd.DataFrame,
    draft_history: list[dict] | None = None,
    compare_fn: Callable = compare_candidates,
) -> dict:
    """Return the actual draft recommendation, not merely the base score leader.

    Stage 1 compares a compact legal shortlist with common-random-number room
    simulations. A close result, or any result that overturns the base board,
    is automatically rerun at a larger sample. Extremely close refined calls get
    one final two-player confirmation. Yahoo remains an opponent-behavior input
    through FastDraftSimulator; it never becomes intrinsic player value here.
    """
    settings = final_call_settings(engine.config)
    if board is None or board.empty:
        return {"ok": False, "fallback": True, "reason": "No legal players on the base board."}

    base = board.iloc[0]
    base_name = str(base["name"])
    base_id = str(base["player_id"])
    teams = int(engine.league["teams"])
    round_no = ((int(current_pick) - 1) // teams) + 1

    shortlist = candidate_shortlist(board, engine.config)
    if (
        not settings["enabled"]
        or round_no >= int(settings["bypass_round"])
        or len(shortlist) < 2
    ):
        reason = (
            "Final Call simulation is disabled."
            if not settings["enabled"]
            else "Late-round special-teams policy leaves the base board as the production call."
        )
        return {
            "ok": True,
            "fallback": False,
            "recommendation": base_name,
            "candidate_id": base_id,
            "runner_up": None,
            "edge": None,
            "edge_label": "BASE BOARD",
            "base_winner": base_name,
            "base_winner_id": base_id,
            "base_agrees": True,
            "stage": "base",
            "runs": 0,
            "through_round": round_no,
            "elapsed_seconds": 0.0,
            "reason": reason,
            "summary": pd.DataFrame(),
            "details": [],
            "shortlist": shortlist,
        }

    base_sim_round = int(engine.config.get("simulation", {}).get("through_round", 8))
    through_round = min(
        int(settings["max_sim_round"]),
        max(base_sim_round, round_no + int(settings["future_rounds"])),
    )
    candidate_ids = shortlist["player_id"].astype(str).tolist()
    started = perf_counter()

    try:
        summary, details = compare_fn(
            engine,
            current_pick=int(current_pick),
            drafted_ids={str(x) for x in drafted_ids},
            my_roster_ids=[str(x) for x in my_roster_ids],
            candidate_ids=candidate_ids,
            runs=int(settings["initial_runs"]),
            through_round=int(through_round),
            draft_history=draft_history,
        )
        summary = summary.sort_values("Avg Lineup Score", ascending=False).reset_index(drop=True)
        stage = "initial"
        runs = int(settings["initial_runs"])

        initial_winner = str(summary.iloc[0]["Candidate"])
        need_refine = (
            _margin(summary) < float(settings["refine_margin"])
            or initial_winner != base_name
        )
        if need_refine and int(settings["refine_runs"]) > runs:
            name_to_id = {
                str(row["name"]): str(row["player_id"])
                for _, row in shortlist.iterrows()
            }
            refine_names = summary.head(3)["Candidate"].astype(str).tolist()
            refine_ids = [name_to_id[x] for x in refine_names if x in name_to_id]
            refine_ids = _ensure_base_candidate(refine_ids, base_id, 3)
            summary, details = compare_fn(
                engine,
                current_pick=int(current_pick),
                drafted_ids={str(x) for x in drafted_ids},
                my_roster_ids=[str(x) for x in my_roster_ids],
                candidate_ids=refine_ids,
                runs=int(settings["refine_runs"]),
                through_round=int(through_round),
                draft_history=draft_history,
            )
            summary = summary.sort_values("Avg Lineup Score", ascending=False).reset_index(drop=True)
            stage = "refined"
            runs = int(settings["refine_runs"])

        if (
            len(summary) >= 2
            and _margin(summary) < float(settings["final_margin"])
            and int(settings["final_runs"]) > runs
        ):
            name_to_id = {
                str(row["name"]): str(row["player_id"])
                for _, row in shortlist.iterrows()
            }
            final_names = summary.head(2)["Candidate"].astype(str).tolist()
            final_ids = [name_to_id[x] for x in final_names if x in name_to_id]
            if len(final_ids) >= 2:
                summary, details = compare_fn(
                    engine,
                    current_pick=int(current_pick),
                    drafted_ids={str(x) for x in drafted_ids},
                    my_roster_ids=[str(x) for x in my_roster_ids],
                    candidate_ids=final_ids,
                    runs=int(settings["final_runs"]),
                    through_round=int(through_round),
                    draft_history=draft_history,
                )
                summary = summary.sort_values("Avg Lineup Score", ascending=False).reset_index(drop=True)
                stage = "final"
                runs = int(settings["final_runs"])

        return _decision_payload(
            summary,
            details,
            base_name=base_name,
            base_id=base_id,
            stage=stage,
            runs=runs,
            through_round=through_round,
            elapsed=perf_counter() - started,
            shortlist=shortlist,
        )
    except Exception as exc:
        return {
            "ok": False,
            "fallback": True,
            "recommendation": base_name,
            "candidate_id": base_id,
            "runner_up": None,
            "edge": None,
            "edge_label": "FALLBACK",
            "base_winner": base_name,
            "base_winner_id": base_id,
            "base_agrees": True,
            "stage": "fallback",
            "runs": 0,
            "through_round": through_round,
            "elapsed_seconds": round(perf_counter() - started, 2),
            "reason": f"Final Call simulation failed; using base score leader. {type(exc).__name__}: {exc}",
            "summary": pd.DataFrame(),
            "details": [],
            "shortlist": shortlist,
        }
