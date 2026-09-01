from __future__ import annotations

from time import perf_counter
from typing import Callable

import pandas as pd

from .decision_strategy import build_final_call_plan
from .final_call_stability import paired_stability_check
from .sim import compare_candidates


def final_call_settings(config: dict) -> dict:
    cfg = config.get("final_call", {}) or {}
    min_win_pct = float(cfg.get("overturn_min_paired_win_pct", 55.0))
    stability_runs = max(100, int(cfg.get("stability_runs", 500)))
    refine_runs = max(40, int(cfg.get("refine_runs", 100)))
    checkpoint_runs = min(
        stability_runs,
        max(refine_runs, int(cfg.get("stability_checkpoint_runs", 200))),
    )
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "min_candidates": max(2, int(cfg.get("min_candidates", 3))),
        "max_candidates": max(2, int(cfg.get("max_candidates", 4))),
        "score_gap": max(0.0, float(cfg.get("score_gap", 8.0))),
        "initial_runs": max(20, int(cfg.get("initial_runs", 30))),
        "refine_runs": refine_runs,
        "stability_checkpoint_runs": checkpoint_runs,
        "stability_runs": stability_runs,
        "overturn_probe_margin": max(0.0, float(cfg.get("overturn_probe_margin", 2.5))),
        "overturn_required_margin": max(0.0, float(cfg.get("overturn_required_margin", 10.0))),
        "overturn_min_paired_win_pct": min(100.0, max(50.0, min_win_pct)),
        "overturn_require_positive_ci": bool(cfg.get("overturn_require_positive_ci", True)),
        "future_rounds": max(1, int(cfg.get("future_rounds", 3))),
        "max_sim_round": max(1, int(cfg.get("max_sim_round", 13))),
        "bypass_round": max(1, int(cfg.get("bypass_round", 14))),
    }


def candidate_shortlist(board: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Legacy compact score-neighborhood shortlist used before the strategic prior.

    v0.6.12 keeps this path for Rounds 1-3 and for callers that do not provide
    enough live roster/room context. From Round 4 onward production Final Call
    normally uses decision_strategy.build_final_call_plan instead.
    """
    if board is None or board.empty:
        return pd.DataFrame()
    settings = final_call_settings(config)
    ordered = board.copy()
    ordered["score"] = pd.to_numeric(ordered["score"], errors="coerce")
    ordered = ordered.sort_values(
        ["score", "proj_points", "adp"], ascending=[False, False, True]
    ).drop_duplicates(subset=["player_id"], keep="first")

    max_n = min(int(settings["max_candidates"]), len(ordered))
    min_n = min(int(settings["min_candidates"]), max_n)
    top_score = float(ordered.iloc[0]["score"])
    within = ordered[
        (top_score - ordered["score"]) <= float(settings["score_gap"])
    ].head(max_n)
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


def _strategy_metadata(plan: dict | None) -> dict:
    plan = plan or {}
    pressure = plan.get("pressure")
    if isinstance(pressure, pd.DataFrame) and not pressure.empty:
        pressure_records = pressure.to_dict("records")
    else:
        pressure_records = []
    raw_row = plan.get("raw_base_row")
    raw_name = str(raw_row["name"]) if raw_row is not None else None
    return {
        "strategy_active": bool(plan.get("strategy_active", False)),
        "priority_positions": list(plan.get("priority_positions", []) or []),
        "position_pressure": pressure_records,
        "raw_base_winner": raw_name,
    }


def _strategy_reason_prefix(plan: dict | None, base_name: str) -> str:
    meta = _strategy_metadata(plan)
    if not meta["strategy_active"]:
        return ""
    pressure = meta["position_pressure"]
    if pressure:
        top = pressure[:2]
        pressure_text = " > ".join(
            f"{row['pos']} {float(row['pressure']):.2f}" for row in top
        )
        prefix = f"Position pressure {pressure_text}; strategic prior is {base_name}. "
    else:
        prefix = f"Strategic prior is {base_name}. "
    raw = meta.get("raw_base_winner")
    if raw and raw != base_name:
        prefix += f"Raw score leader was {raw}. "
    return prefix


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
    strategy_plan: dict | None = None,
) -> dict:
    sim_winner = str(summary.iloc[0]["Candidate"])
    runner_up = str(summary.iloc[1]["Candidate"]) if len(summary) >= 2 else None
    edge = _margin(summary)
    recommendation_id = (
        base_id if str(recommendation) == str(base_name)
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
    strategy_meta = _strategy_metadata(strategy_plan)
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
        "raw_base_winner": strategy_meta.get("raw_base_winner") or str(base_name),
        "base_agrees": str(recommendation) == str(base_name),
        "sim_agrees": str(recommendation) == sim_winner,
        "strategy_active": strategy_meta.get("strategy_active", False),
        "position_priority": strategy_meta.get("priority_positions", []),
        "position_pressure": strategy_meta.get("position_pressure", []),
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
        "paired_stop_stage": paired_evidence.get("stop_stage"),
        "paired_stop_reason": paired_evidence.get("stop_reason"),
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
    """Return the live production recommendation under a draft-clock budget.

    v0.6.12 keeps the hardened 30 -> 100 -> 200 -> 500 paired evidence stream,
    but changes what reaches that stream from Round 4 onward. A position-pressure
    prior supplies multiple candidates from the roster's most important position,
    keeps a secondary-position candidate, and always preserves global value
    exceptions. Position is therefore a strong prior rather than a hard lock.
    """
    settings = final_call_settings(engine.config)
    if board is None or board.empty:
        return {"ok": False, "fallback": True, "reason": "No legal players on the base board."}

    teams = int(engine.league["teams"])
    round_no = ((int(current_pick) - 1) // teams) + 1
    raw_base = board.iloc[0]
    raw_base_name = str(raw_base["name"])
    raw_base_id = str(raw_base["player_id"])

    strategy_plan = None
    if round_no < int(settings["bypass_round"]):
        try:
            candidate_plan = build_final_call_plan(
                board,
                engine,
                current_pick=int(current_pick),
                my_roster_ids=[str(x) for x in my_roster_ids],
                draft_history=draft_history,
            )
            if bool(candidate_plan.get("strategy_active", False)):
                strategy_plan = candidate_plan
        except Exception:
            strategy_plan = None

    if strategy_plan is not None:
        shortlist = strategy_plan["shortlist"].copy()
        base = strategy_plan["base_row"]
    else:
        shortlist = candidate_shortlist(board, engine.config)
        base = raw_base

    base_name = str(base["name"])
    base_id = str(base["player_id"])

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
            "recommendation": raw_base_name if round_no >= int(settings["bypass_round"]) else base_name,
            "candidate_id": raw_base_id if round_no >= int(settings["bypass_round"]) else base_id,
            "runner_up": None,
            "edge": None,
            "edge_label": "BASE BOARD",
            "sim_winner": raw_base_name if round_no >= int(settings["bypass_round"]) else base_name,
            "sim_winner_id": raw_base_id if round_no >= int(settings["bypass_round"]) else base_id,
            "base_winner": raw_base_name if round_no >= int(settings["bypass_round"]) else base_name,
            "base_winner_id": raw_base_id if round_no >= int(settings["bypass_round"]) else base_id,
            "raw_base_winner": raw_base_name,
            "base_agrees": True,
            "sim_agrees": True,
            "strategy_active": False if round_no >= int(settings["bypass_round"]) else bool(strategy_plan),
            "position_priority": [] if strategy_plan is None else list(strategy_plan.get("priority_positions", [])),
            "position_pressure": [] if strategy_plan is None else _strategy_metadata(strategy_plan)["position_pressure"],
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
    strategy_prefix = _strategy_reason_prefix(strategy_plan, base_name)

    try:
        # Fast screen: most picks end here. v0.6.12 may screen six candidates,
        # but only one challenger can advance to the expensive paired stream.
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

        if sim_winner == base_name:
            reason = (
                strategy_prefix + "Fast Yahoo-informed room screen agrees."
                if strategy_prefix
                else "Base score board and the fast Yahoo-informed room screen agree."
            )
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
                reason=reason,
                strategy_plan=strategy_plan,
            )

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
                    strategy_prefix
                    + f"Room simulation slightly prefers {sim_winner}, but its +{initial_edge:.2f} edge "
                    "is too small to justify reopening the strategic-prior decision."
                ),
                strategy_plan=strategy_plan,
            )

        name_to_id = {
            str(row["name"]): str(row["player_id"])
            for _, row in shortlist.iterrows()
        }
        challenger_id = name_to_id.get(sim_winner)
        if challenger_id is None:
            raise RuntimeError(f"Could not map room-sim challenger {sim_winner!r} to shortlist")

        # One continuous paired stream supplies the 100-run confirmation,
        # 200-run futility checkpoint and, only when needed, the 500-run final gate.
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
            confirmation_runs=int(settings["refine_runs"]),
            checkpoint_runs=int(settings["stability_checkpoint_runs"]),
            required_margin=float(settings["overturn_required_margin"]),
            min_win_pct=float(settings["overturn_min_paired_win_pct"]),
            require_positive_ci=bool(settings["overturn_require_positive_ci"]),
        )
        stable_summary = stable_summary.sort_values(
            "Avg Lineup Score", ascending=False
        ).reset_index(drop=True)
        stable_winner = str(stable_summary.iloc[0]["Candidate"])
        paired_mean = float(paired.get("mean_delta", 0.0))
        paired_win_pct = float(paired.get("paired_win_pct", 0.0))
        ci_low = float(paired.get("ci_low", float("-inf")))
        ci_high = float(paired.get("ci_high", float("inf")))
        actual_runs = int(paired.get("runs", len(stable_summary)))
        stop_stage = str(paired.get("stop_stage", "full"))
        stop_reason = str(paired.get("stop_reason", ""))
        min_win_pct = float(settings["overturn_min_paired_win_pct"])
        positive_ci_ok = (
            ci_low > 0.0
            if bool(settings["overturn_require_positive_ci"])
            else True
        )

        evidence_pass = (
            stop_stage == "full"
            and actual_runs >= int(settings["stability_runs"])
            and stable_winner == sim_winner
            and stable_winner != base_name
            and paired_mean >= float(settings["overturn_required_margin"])
            and paired_win_pct >= min_win_pct
            and positive_ci_ok
        )

        if evidence_pass:
            recommendation = stable_winner
            reason = (
                strategy_prefix
                + f"Paired stability check confirms an overturn of {base_name}: {stable_winner} "
                f"leads by {paired_mean:.2f} lineup points across {actual_runs} paired runs, "
                f"wins {paired_win_pct:.1f}% of paired rooms, and has a 95% paired CI of "
                f"[{ci_low:+.2f}, {ci_high:+.2f}]."
            )
            stage = "stabilized"
        else:
            recommendation = base_name
            if stop_stage == "confirmation":
                reason = (
                    strategy_prefix
                    + f"The initial room-sim challenge did not survive the {actual_runs}-run paired confirmation; "
                    f"retain {base_name}. {sim_winner} paired mean delta: {paired_mean:+.2f}; "
                    f"paired wins: {paired_win_pct:.1f}%; 95% CI [{ci_low:+.2f}, {ci_high:+.2f}]."
                )
                stage = "refined"
            elif stop_stage == "checkpoint":
                reason = (
                    strategy_prefix
                    + f"The 100-run challenge became futile at the {actual_runs}-run paired stability checkpoint; "
                    f"retain {base_name}. {sim_winner} paired mean delta: {paired_mean:+.2f}; "
                    f"paired wins: {paired_win_pct:.1f}%; 95% CI [{ci_low:+.2f}, {ci_high:+.2f}]. "
                    f"Checkpoint reason: {stop_reason}."
                )
                stage = "stabilized"
            elif stable_winner == base_name or paired_mean <= 0:
                reason = (
                    strategy_prefix
                    + f"The challenge did not survive the {actual_runs}-run paired stability check; "
                    f"retain {base_name}. {sim_winner} paired mean delta: {paired_mean:+.2f}; "
                    f"paired wins: {paired_win_pct:.1f}%; 95% CI [{ci_low:+.2f}, {ci_high:+.2f}]."
                )
                stage = "stabilized"
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
                    failures.append(
                        f"95% CI crosses zero [{ci_low:+.2f}, {ci_high:+.2f}]"
                    )
                reason = (
                    strategy_prefix
                    + f"The {actual_runs}-run paired check still leans {stable_winner}, "
                    f"but the evidence is not stable enough to overturn {base_name}: "
                    + "; ".join(failures)
                    + "."
                )
                stage = "stabilized"

        return _decision_payload(
            stable_summary,
            stable_details,
            recommendation=recommendation,
            base_name=base_name,
            base_id=base_id,
            stage=stage,
            runs=actual_runs,
            through_round=through_round,
            elapsed=perf_counter() - started,
            shortlist=shortlist,
            reason=reason,
            paired_evidence=paired,
            evidence_pass=evidence_pass,
            strategy_plan=strategy_plan,
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
            "raw_base_winner": raw_base_name,
            "base_agrees": True,
            "sim_agrees": True,
            "strategy_active": bool(strategy_plan),
            "position_priority": [] if strategy_plan is None else list(strategy_plan.get("priority_positions", [])),
            "position_pressure": [] if strategy_plan is None else _strategy_metadata(strategy_plan)["position_pressure"],
            "stage": "fallback",
            "runs": 0,
            "through_round": through_round,
            "elapsed_seconds": round(perf_counter() - started, 2),
            "reason": f"Final Call simulation failed; using strategic prior. {type(exc).__name__}: {exc}",
            "summary": pd.DataFrame(),
            "details": [],
            "shortlist": shortlist,
        }
