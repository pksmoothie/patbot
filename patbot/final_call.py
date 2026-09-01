from __future__ import annotations

from time import perf_counter
from typing import Callable

import pandas as pd

from .final_call_stability import paired_stability_check
from .sim import compare_candidates


def final_call_settings(config: dict) -> dict:
    cfg = config.get("final_call", {}) or {}
    min_win_pct = float(cfg.get("overturn_min_paired_win_pct", 55.0))
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "min_candidates": max(2, int(cfg.get("min_candidates", 3))),
        "max_candidates": max(2, int(cfg.get("max_candidates", 4))),
        "score_gap": max(0.0, float(cfg.get("score_gap", 8.0))),
        "initial_runs": max(20, int(cfg.get("initial_runs", 30))),
        "refine_runs": max(40, int(cfg.get("refine_runs", 100))),
        "stability_runs": max(100, int(cfg.get("stability_runs", 500))),
        "overturn_probe_margin": max(0.0, float(cfg.get("overturn_probe_margin", 2.5))),
        "overturn_required_margin": max(0.0, float(cfg.get("overturn_required_margin", 10.0))),
        "overturn_min_paired_win_pct": min(100.0, max(50.0, min_win_pct)),
        "overturn_require_positive_ci": bool(cfg.get("overturn_require_positive_ci", True)),
        "future_rounds": max(1, int(cfg.get("future_rounds", 3))),
        "max_sim_round": max(1, int(cfg.get("max_sim_round", 13))),
        "bypass_round": max(1, int(cfg.get("bypass_round", 14))),
    }


def candidate_shortlist(board: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Choose a compact legal candidate set from the production score board.

    Final Call is a room-aware correction layer, not a second full ranking model.
    It therefore inspects only a few plausible alternatives around the base score
    leader so the recommendation can return comfortably inside a live draft clock.
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


def _edge_label(edge: float) -> str:
    if edge >= 10.0:
        return "STRONG"
    if edge >= 6.0:
        return "CLEAR"
    if edge >= 2.5:
        return "LEAN"
    return "CLOSE"


def _id_for_name(details: list[dict], name: str, fallback: str = "") -> str:
    for item in details:
        if str(item.get("candidate")) == str(name):
            return str(item.get("candidate_id", fallback))
    return str(fallback)


def _decision_payload(
    summary: pd.DataFrame,
    details: list[dict],
    *,
    recommendation: str,
    base_name: str,
    base_id: str,
    stage: str,
    runs: int,
    through_round: int,
    elapsed: float,
    shortlist: pd.DataFrame,
    reason: str,
    paired_evidence: dict | None = None,
    evidence_pass: bool | None = None,
) -> dict:
    sim_winner = str(summary.iloc[0]["Candidate"])
    runner_up = str(summary.iloc[1]["Candidate"]) if len(summary) >= 2 else None
    edge = _margin(summary)
    recommendation_id = (
        base_id
        if str(recommendation) == str(base_name)
        else _id_for_name(details, recommendation)
    )
    edge_label = _edge_label(float(edge)) if edge != float("inf") else "ONLY OPTION"
    if (
        evidence_pass is False
        and sim_winner != str(base_name)
        and str(recommendation) == str(base_name)
    ):
        edge_label = "UNSTABLE"

    paired_evidence = dict(paired_evidence or {})
    return {
        "ok": True,
        "fallback": False,
        "recommendation": str(recommendation),
        "candidate_id": str(recommendation_id),
        "runner_up": runner_up,
        "edge": round(float(edge), 2) if edge != float("inf") else None,
        "edge_label": edge_label,
        "sim_winner": sim_winner,
        "sim_winner_id": _id_for_name(details, sim_winner),
        "base_winner": str(base_name),
        "base_winner_id": str(base_id),
        "base_agrees": str(recommendation) == str(base_name),
        "sim_agrees": str(recommendation) == sim_winner,
        "stage": str(stage),
        "runs": int(runs),
        "through_round": int(through_round),
        "elapsed_seconds": round(float(elapsed), 2),
        "reason": str(reason),
        "paired_challenger": paired_evidence.get("challenger"),
        "paired_mean_delta": paired_evidence.get("mean_delta"),
        "paired_win_pct": paired_evidence.get("paired_win_pct"),
        "paired_ci_low": paired_evidence.get("ci_low"),
        "paired_ci_high": paired_evidence.get("ci_high"),
        "paired_evidence_pass": evidence_pass,
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
    stability_fn: Callable = paired_stability_check,
) -> dict:
    """Return the actual draft recommendation under a live-clock time budget.

    The base PatBot board is the prior. A small Yahoo-informed room screen can
    confirm it quickly. A challenger first has to survive a 100-run direct check.
    Any challenger still strong enough to overturn the base prior then receives a
    larger paired stability test. An overturn is allowed only when the large-sample
    mean edge, paired win rate and paired confidence interval all support it.
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
            "sim_winner": base_name,
            "sim_winner_id": base_id,
            "base_winner": base_name,
            "base_winner_id": base_id,
            "base_agrees": True,
            "sim_agrees": True,
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
        sim_winner = str(summary.iloc[0]["Candidate"])
        initial_edge = _margin(summary)

        # The score board is the prior. If it also wins the first room screen,
        # there is no reason to burn the clock resolving a small numerical edge.
        if sim_winner == base_name:
            return _decision_payload(
                summary,
                details,
                recommendation=base_name,
                base_name=base_name,
                base_id=base_id,
                stage="initial",
                runs=int(settings["initial_runs"]),
                through_round=through_round,
                elapsed=perf_counter() - started,
                shortlist=shortlist,
                reason="Base score board and the fast Yahoo-informed room screen agree.",
            )

        # A tiny challenger lead is not enough evidence to reopen the decision.
        if initial_edge < float(settings["overturn_probe_margin"]):
            return _decision_payload(
                summary,
                details,
                recommendation=base_name,
                base_name=base_name,
                base_id=base_id,
                stage="initial",
                runs=int(settings["initial_runs"]),
                through_round=through_round,
                elapsed=perf_counter() - started,
                shortlist=shortlist,
                reason=(
                    f"Room simulation slightly prefers {sim_winner}, but its +{initial_edge:.2f} edge "
                    "is too small to justify overturning the base score leader."
                ),
            )

        # Only a plausible overturn gets a second pass, and that pass compares
        # the challenger directly with the base leader instead of three players.
        name_to_id = {
            str(row["name"]): str(row["player_id"])
            for _, row in shortlist.iterrows()
        }
        challenger_id = name_to_id.get(sim_winner)
        if challenger_id is None:
            raise RuntimeError(f"Could not map room-sim challenger {sim_winner!r} to shortlist")

        refine_ids = [str(challenger_id), str(base_id)]
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
        confirmed_winner = str(summary.iloc[0]["Candidate"])
        confirmed_edge = _margin(summary)

        # If the challenger reverses or cannot clear the materiality hurdle at
        # 100 runs, the base prior survives without paying for the larger audit.
        if confirmed_winner == base_name:
            return _decision_payload(
                summary,
                details,
                recommendation=base_name,
                base_name=base_name,
                base_id=base_id,
                stage="refined",
                runs=int(settings["refine_runs"]),
                through_round=through_round,
                elapsed=perf_counter() - started,
                shortlist=shortlist,
                reason=f"The initial room-sim challenge did not survive confirmation; retain {base_name}.",
            )

        if confirmed_edge < float(settings["overturn_required_margin"]):
            return _decision_payload(
                summary,
                details,
                recommendation=base_name,
                base_name=base_name,
                base_id=base_id,
                stage="refined",
                runs=int(settings["refine_runs"]),
                through_round=through_round,
                elapsed=perf_counter() - started,
                shortlist=shortlist,
                reason=(
                    f"Room simulation still leans {confirmed_winner}, but the confirmed +{confirmed_edge:.2f} edge "
                    f"does not clear the +{float(settings['overturn_required_margin']):.1f} threshold required to overturn {base_name}."
                ),
            )

        # v0.6.9: a 100-run mean is only a provisional challenge. The audit that
        # exposed Puka/Chase showed that a +20 edge at 100 runs could reverse by
        # 500. Every would-be overturn therefore gets an independent paired
        # stability check before it is allowed to replace the base score leader.
        stable_summary, stable_details, paired = stability_fn(
            engine,
            current_pick=int(current_pick),
            drafted_ids={str(x) for x in drafted_ids},
            my_roster_ids=[str(x) for x in my_roster_ids],
            challenger_id=str(challenger_id),
            base_id=str(base_id),
            runs=int(settings["stability_runs"]),
            through_round=int(through_round),
            draft_history=draft_history,
        )
        stable_summary = stable_summary.sort_values(
            "Avg Lineup Score", ascending=False
        ).reset_index(drop=True)
        stable_winner = str(stable_summary.iloc[0]["Candidate"])
        stable_edge = _margin(stable_summary)
        paired_mean = float(paired.get("mean_delta", 0.0))
        paired_win_pct = float(paired.get("paired_win_pct", 0.0))
        ci_low = float(paired.get("ci_low", float("-inf")))
        ci_high = float(paired.get("ci_high", float("inf")))
        min_win_pct = float(settings["overturn_min_paired_win_pct"])
        positive_ci_ok = (
            ci_low > 0.0
            if bool(settings["overturn_require_positive_ci"])
            else True
        )
        evidence_pass = (
            stable_winner == confirmed_winner
            and stable_winner != base_name
            and paired_mean >= float(settings["overturn_required_margin"])
            and paired_win_pct >= min_win_pct
            and positive_ci_ok
        )

        if evidence_pass:
            recommendation = stable_winner
            reason = (
                f"Paired stability check confirms an overturn of {base_name}: {stable_winner} "
                f"leads by {paired_mean:.2f} lineup points across {int(settings['stability_runs'])} paired runs, "
                f"wins {paired_win_pct:.1f}% of paired rooms, and has a 95% paired CI of "
                f"[{ci_low:+.2f}, {ci_high:+.2f}]."
            )
        else:
            recommendation = base_name
            if stable_winner == base_name or paired_mean <= 0:
                reason = (
                    f"The 100-run challenge did not survive the {int(settings['stability_runs'])}-run paired stability check; "
                    f"retain {base_name}. {confirmed_winner} paired mean delta: {paired_mean:+.2f}; "
                    f"paired wins: {paired_win_pct:.1f}%; 95% CI [{ci_low:+.2f}, {ci_high:+.2f}]."
                )
            else:
                failures = []
                if paired_mean < float(settings["overturn_required_margin"]):
                    failures.append(
                        f"mean edge {paired_mean:+.2f} < +{float(settings['overturn_required_margin']):.1f}"
                    )
                if paired_win_pct < min_win_pct:
                    failures.append(
                        f"paired wins {paired_win_pct:.1f}% < {min_win_pct:.1f}%"
                    )
                if not positive_ci_ok:
                    failures.append(f"95% CI crosses zero [{ci_low:+.2f}, {ci_high:+.2f}]")
                reason = (
                    f"The {int(settings['stability_runs'])}-run paired check still leans {stable_winner}, "
                    f"but the evidence is not stable enough to overturn {base_name}: "
                    + "; ".join(failures)
                    + "."
                )

        return _decision_payload(
            stable_summary,
            stable_details,
            recommendation=recommendation,
            base_name=base_name,
            base_id=base_id,
            stage="stabilized",
            runs=int(settings["stability_runs"]),
            through_round=through_round,
            elapsed=perf_counter() - started,
            shortlist=shortlist,
            reason=reason,
            paired_evidence=paired,
            evidence_pass=evidence_pass,
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
            "sim_winner": base_name,
            "sim_winner_id": base_id,
            "base_winner": base_name,
            "base_winner_id": base_id,
            "base_agrees": True,
            "sim_agrees": True,
            "stage": "fallback",
            "runs": 0,
            "through_round": through_round,
            "elapsed_seconds": round(perf_counter() - started, 2),
            "reason": f"Final Call simulation failed; using base score leader. {type(exc).__name__}: {exc}",
            "summary": pd.DataFrame(),
            "details": [],
            "shortlist": shortlist,
        }
