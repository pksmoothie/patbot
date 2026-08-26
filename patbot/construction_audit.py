from __future__ import annotations

from collections import Counter, defaultdict
from typing import Callable

import numpy as np
import pandas as pd

from .draft import snake_pick
from .roster_strategy import offensive_starters_complete
from .sim import FastDraftSimulator


NEG_INF = -1_000_000_000.0
DEFAULT_AUDIT_ROUNDS = (8, 10, 12, 13)


def _counter_from_array(sim: FastDraftSimulator, roster_counts: np.ndarray) -> Counter:
    return Counter(
        {
            pos: int(roster_counts[code])
            for pos, code in sim.pos_to_code.items()
            if 0 <= int(code) < len(roster_counts)
        }
    )


def roster_state(sim: FastDraftSimulator, roster_counts: np.ndarray) -> dict:
    counts = _counter_from_array(sim, roster_counts)
    rcfg = sim.engine.roster_cfg
    eligible = [str(p).upper() for p in rcfg.get("flex_eligible", ["RB", "WR", "TE"])]

    qb_ok = counts["QB"] >= int(rcfg.get("QB", 0))
    rb_ok = counts["RB"] >= int(rcfg.get("RB", 0))
    wr_ok = counts["WR"] >= int(rcfg.get("WR", 0))
    te_ok = counts["TE"] >= int(rcfg.get("TE", 0))

    flex_need = int(rcfg.get("FLEX", 0))
    flex_excess = sum(
        max(counts[pos] - int(rcfg.get(pos, 0)), 0)
        for pos in eligible
    )
    flex_ok = flex_excess >= flex_need

    return {
        "counts": counts,
        "qb1_filled": qb_ok,
        "rb2_filled": rb_ok,
        "wr3_filled": wr_ok,
        "te1_filled": te_ok,
        "flex_filled": flex_ok,
        "offense_complete": offensive_starters_complete(counts, rcfg),
    }


def missing_base_starters(sim: FastDraftSimulator, roster_counts: np.ndarray) -> list[str]:
    state = roster_state(sim, roster_counts)
    counts = state["counts"]
    rcfg = sim.engine.roster_cfg
    missing = []
    for pos in ("QB", "RB", "WR", "TE"):
        if counts[pos] < int(rcfg.get(pos, 0)):
            missing.append(pos)
    return missing


def starter_first_allowed_positions(
    sim: FastDraftSimulator,
    roster_counts: np.ndarray,
) -> set[str] | None:
    """Positions allowed by the intentionally rigid starter-first counterfactual.

    This is not PatBot's strategy. It exists only as a benchmark. If any base
    offensive starter is missing, the benchmark must fill one of those positions.
    Once the base QB/RB/WR/TE minima are filled but FLEX is not, it must fill FLEX
    from an eligible position. After the offense is complete it stops constraining
    PatBot beyond the normal value-aware rules already in the score vector.
    """
    missing = missing_base_starters(sim, roster_counts)
    if missing:
        return set(missing)

    state = roster_state(sim, roster_counts)
    if not state["flex_filled"]:
        return {
            str(p).upper()
            for p in sim.engine.roster_cfg.get("flex_eligible", ["RB", "WR", "TE"])
        }
    return None


def classify_pick(
    sim: FastDraftSimulator,
    roster_counts: np.ndarray,
    player_idx: int,
) -> str:
    state = roster_state(sim, roster_counts)
    counts = state["counts"]
    pos = str(sim.pos[int(player_idx)]).upper()
    rcfg = sim.engine.roster_cfg

    if pos == "QB":
        return "QB1" if counts["QB"] == 0 else "QB2"
    if pos == "TE":
        return "TE1" if counts["TE"] == 0 else "TE2"
    if pos in {"RB", "WR"}:
        if counts[pos] < int(rcfg.get(pos, 0)):
            return f"{pos} starter"
        if not state["flex_filled"] and not missing_base_starters(sim, roster_counts):
            return "FLEX fill"
        if state["offense_complete"]:
            return "RB/WR bench after complete"
        return "RB/WR bench before complete"
    return pos


def _starter_first_score_vector(
    sim: FastDraftSimulator,
    available: np.ndarray,
    roster_counts: np.ndarray,
    pick: int,
) -> np.ndarray:
    score = sim._patbot_score_vector(available, roster_counts, pick)
    allowed = starter_first_allowed_positions(sim, roster_counts)
    if not allowed:
        return score

    allowed_mask = np.isin(sim.pos, list(allowed))
    legal = available & allowed_mask & (score > NEG_INF / 2)
    if not legal.any():
        return score
    return np.where(legal, score, NEG_INF)


def _starter_first_pick(
    sim: FastDraftSimulator,
    available: np.ndarray,
    my_counts: np.ndarray,
    pick: int,
    opp_counts: np.ndarray,
    archetypes: dict[int, str],
    market_latent: np.ndarray,
    custom_noise_base: np.ndarray,
) -> int:
    """Mirror PatBot's R2/R3 lookahead while enforcing starter-first ordering."""
    current_scores = _starter_first_score_vector(sim, available, my_counts, pick)
    greedy = int(np.argmax(current_scores))
    round_no = (int(pick) - 1) // sim.teams + 1
    if not sim.lookahead_enabled or round_no not in sim.lookahead_rounds:
        return greedy

    next_pick = sim._next_my_pick(pick)
    gap = next_pick - int(pick)
    if gap <= 0 or gap > sim.lookahead_max_gap:
        return greedy

    candidates = np.where(available & (current_scores > NEG_INF / 2))[0]
    if len(candidates) <= 1:
        return greedy
    candidates = candidates[np.argsort(current_scores[candidates])[::-1]][: sim.lookahead_branch_width]

    best_idx = greedy
    best_value = -float("inf")
    for candidate in candidates:
        branch_available = available.copy()
        branch_opp_counts = opp_counts.copy()
        branch_my_counts = my_counts.copy()

        branch_available[candidate] = False
        code = sim.pos_code[candidate]
        if code >= 0:
            branch_my_counts[code] += 1

        for future_pick in range(int(pick) + 1, next_pick):
            if future_pick in sim.my_picks or not branch_available.any():
                break
            sim._take_opponent_pick(
                future_pick,
                branch_available,
                branch_opp_counts,
                archetypes,
                market_latent,
                custom_noise_base,
            )

        if branch_available.any():
            future_scores = _starter_first_score_vector(
                sim,
                branch_available,
                branch_my_counts,
                next_pick,
            )
            future_idx = int(np.argmax(future_scores))
            future_value = float(future_scores[future_idx])
        else:
            future_idx = None
            future_value = 0.0

        pair_vorp = max(float(sim.vorp[candidate]), 0.0)
        if future_idx is not None:
            pair_vorp += max(float(sim.vorp[future_idx]), 0.0)

        path_value = (
            float(current_scores[candidate])
            + sim.lookahead_future_weight * future_value
            + sim.lookahead_vorp_weight * pair_vorp
        )
        if path_value > best_value:
            best_value = path_value
            best_idx = int(candidate)

    return best_idx


def _seed_from_history(sim: FastDraftSimulator, draft_history: list[dict]):
    drafted_idx: set[int] = set()
    mine: list[int] = []
    for record in draft_history:
        idx = sim.id_to_idx.get(str(record.get("player_id", "")))
        if idx is None:
            continue
        drafted_idx.add(idx)
        if int(record.get("owner_slot", -1)) == sim.slot:
            mine.append(idx)
    return drafted_idx, mine


def _run_branch(
    sim: FastDraftSimulator,
    draft_history: list[dict],
    *,
    start_pick: int,
    last_pick: int,
    archetypes: dict[int, str],
    market_latent: np.ndarray,
    custom_noise_base: np.ndarray,
    policy: str,
    target_rounds: set[int],
    capture: bool,
):
    drafted_idx, seeded_mine = _seed_from_history(sim, draft_history)
    available = np.ones(sim.n, dtype=bool)
    if drafted_idx:
        available[list(drafted_idx)] = False

    opp_counts = sim._seed_opponent_counts(draft_history)
    my_counts = np.zeros(len(sim.POSITIONS), dtype=np.int16)
    mine = list(seeded_mine)
    for idx in mine:
        code = sim.pos_code[idx]
        if code >= 0:
            my_counts[code] += 1

    checkpoint_rows: list[dict] = []
    deferral_rows: list[dict] = []

    for pick in range(int(start_pick), int(last_pick) + 1):
        if not available.any():
            break

        if pick in sim.my_picks:
            round_no = (pick - 1) // sim.teams + 1
            pre = roster_state(sim, my_counts)
            immediate_scores = sim._patbot_score_vector(available, my_counts, pick)

            if policy == "starter_first":
                idx = _starter_first_pick(
                    sim,
                    available,
                    my_counts,
                    pick,
                    opp_counts,
                    archetypes,
                    market_latent,
                    custom_noise_base,
                )
            else:
                idx = sim._lookahead_pick(
                    available,
                    my_counts,
                    pick,
                    opp_counts,
                    archetypes,
                    market_latent,
                    custom_noise_base,
                )

            role = classify_pick(sim, my_counts, idx)
            missing = missing_base_starters(sim, my_counts)

            if capture and role == "RB/WR bench before complete" and missing:
                missing_mask = available & np.isin(sim.pos, missing)
                if missing_mask.any():
                    best_missing_score = float(np.max(immediate_scores[missing_mask]))
                    edge = float(immediate_scores[idx]) - best_missing_score
                else:
                    edge = np.nan
                deferral_rows.append(
                    {
                        "Round": round_no,
                        "Player": str(sim.names[idx]),
                        "Pos": str(sim.pos[idx]),
                        "Deferred Needs": "/".join(missing),
                        "Immediate Score Edge": edge,
                        "League Winner Score": float(sim.league_winner_score[idx]),
                    }
                )

            available[idx] = False
            mine.append(idx)
            code = sim.pos_code[idx]
            if code >= 0:
                my_counts[code] += 1
            post = roster_state(sim, my_counts)

            if capture and round_no in target_rounds:
                pos = str(sim.pos[idx]).upper()
                checkpoint_rows.append(
                    {
                        "Round": round_no,
                        "Player": str(sim.names[idx]),
                        "Pos": pos,
                        "Role": role,
                        "QB1 Filled Before": pre["qb1_filled"],
                        "RB2 Filled Before": pre["rb2_filled"],
                        "WR3 Filled Before": pre["wr3_filled"],
                        "TE1 Filled Before": pre["te1_filled"],
                        "FLEX Filled Before": pre["flex_filled"],
                        "Offense Complete Before": pre["offense_complete"],
                        "Offense Complete After": post["offense_complete"],
                        "QB2 Violation": role == "QB2",
                        "Post-Complete Invalid": bool(
                            pre["offense_complete"]
                            and pos in {"QB", "RB", "WR", "TE"}
                            and pos not in {"RB", "WR"}
                        ),
                    }
                )
        else:
            sim._take_opponent_pick(
                pick,
                available,
                opp_counts,
                archetypes,
                market_latent,
                custom_noise_base,
            )

    return mine, my_counts, checkpoint_rows, deferral_rows


def _pct(series: pd.Series) -> float:
    return round(100.0 * float(pd.Series(series).astype(bool).mean()), 1) if len(series) else 0.0


def _summarize_checkpoints(records: pd.DataFrame, runs: int) -> pd.DataFrame:
    rows = []
    if records.empty:
        return pd.DataFrame()

    for round_no, group in records.groupby("Round", sort=True):
        roles = group["Role"].value_counts()
        positions = group["Pos"].value_counts()
        top = group["Player"].value_counts()
        top_name = str(top.index[0]) if len(top) else "—"
        top_pct = round(100.0 * float(top.iloc[0]) / max(int(runs), 1), 1) if len(top) else 0.0
        rows.append(
            {
                "Round": int(round_no),
                "Most Common Pick": top_name,
                "Pick %": top_pct,
                "QB1 Filled Before %": _pct(group["QB1 Filled Before"]),
                "TE1 Filled Before %": _pct(group["TE1 Filled Before"]),
                "RB2 Filled Before %": _pct(group["RB2 Filled Before"]),
                "WR3 Filled Before %": _pct(group["WR3 Filled Before"]),
                "FLEX Filled Before %": _pct(group["FLEX Filled Before"]),
                "Offense Complete Before %": _pct(group["Offense Complete Before"]),
                "Offense Complete After %": _pct(group["Offense Complete After"]),
                "QB1 Pick %": round(100.0 * roles.get("QB1", 0) / max(int(runs), 1), 1),
                "TE1 Pick %": round(100.0 * roles.get("TE1", 0) / max(int(runs), 1), 1),
                "TE2 Pick %": round(100.0 * roles.get("TE2", 0) / max(int(runs), 1), 1),
                "RB/WR Pick %": round(100.0 * (positions.get("RB", 0) + positions.get("WR", 0)) / max(int(runs), 1), 1),
                "RB/WR Bench Before Complete %": round(100.0 * roles.get("RB/WR bench before complete", 0) / max(int(runs), 1), 1),
                "RB/WR Bench After Complete %": round(100.0 * roles.get("RB/WR bench after complete", 0) / max(int(runs), 1), 1),
                "QB2 Pick %": round(100.0 * roles.get("QB2", 0) / max(int(runs), 1), 1),
                "Post-Complete Invalid %": _pct(group["Post-Complete Invalid"]),
                "Runs": int(runs),
            }
        )
    return pd.DataFrame(rows)


def _summarize_deferrals(
    deferrals: pd.DataFrame,
    run_deltas: dict[int, float],
    runs: int,
) -> pd.DataFrame:
    if deferrals.empty:
        return pd.DataFrame()

    rows = []
    keys = ["Round", "Player", "Pos", "Deferred Needs"]
    for key, group in deferrals.groupby(keys, dropna=False, sort=True):
        run_ids = group["Run"].astype(int).unique().tolist()
        deltas = [run_deltas[r] for r in run_ids if r in run_deltas]
        rows.append(
            {
                "Round": int(key[0]),
                "Player": str(key[1]),
                "Pos": str(key[2]),
                "Deferred Needs": str(key[3]),
                "Times": int(len(group)),
                "Draft %": round(100.0 * len(group) / max(int(runs), 1), 1),
                "Avg Immediate Score Edge": round(float(group["Immediate Score Edge"].mean()), 2),
                "Avg League Winner Score": round(float(group["League Winner Score"].mean()), 1),
                "Avg Final Lineup Delta vs Starter-First": round(float(np.mean(deltas)), 2) if deltas else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["Times", "Round", "Avg Immediate Score Edge"],
        ascending=[False, True, False],
    ).reset_index(drop=True)


def run_construction_audit(
    engine,
    draft_history: list[dict],
    *,
    rounds: tuple[int, ...] = DEFAULT_AUDIT_ROUNDS,
    runs: int = 500,
    seed: int | None = None,
    progress: Callable[[int, int], None] | None = None,
):
    """Audit PatBot's value-aware roster construction against rigid starter-first.

    The actual branch uses FastDraftSimulator._lookahead_pick, so Rounds 2 and 3
    follow the exact same lookahead path as the production candidate simulator.
    The counterfactual uses the same room inputs but forces missing offensive
    starters (and then FLEX) before allowing bench depth. Both branches are
    evaluated on the same sampled performance/availability outcome in each run.
    """
    target_rounds = tuple(sorted({int(r) for r in rounds if 2 <= int(r) <= 13}))
    if not target_rounds:
        raise ValueError("Construction audit needs at least one target round from 2 through 13.")

    sim = FastDraftSimulator(engine)
    if seed is None:
        seed = int(sim.comparison_seed) + 546

    start_pick = max([int(x.get("overall_pick", 0)) for x in draft_history] or [0]) + 1
    last_pick = snake_pick(max(target_rounds), sim.teams, sim.slot)
    latent_sd = np.maximum(sim.sd_floor, sim.adp * sim.sd_pct)

    all_checkpoints: list[dict] = []
    all_deferrals: list[dict] = []
    actual_scores = np.empty(int(runs), dtype=float)
    forced_scores = np.empty(int(runs), dtype=float)
    actual_vorp = np.empty(int(runs), dtype=float)
    forced_vorp = np.empty(int(runs), dtype=float)
    actual_complete = np.zeros(int(runs), dtype=bool)
    forced_complete = np.zeros(int(runs), dtype=bool)
    run_deltas: dict[int, float] = {}

    for run in range(int(runs)):
        room_rng = np.random.default_rng(int(seed) + run * 10007)
        archetypes = sim._archetype_assignments(room_rng)
        market_latent = np.maximum(1.0, room_rng.normal(sim.adp, latent_sd))
        custom_noise_base = room_rng.normal(
            0.0,
            np.maximum(3.0, sim.custom_rank * 0.06),
        )

        actual_mine, actual_counts, checkpoints, deferrals = _run_branch(
            sim,
            draft_history,
            start_pick=start_pick,
            last_pick=last_pick,
            archetypes=archetypes,
            market_latent=market_latent,
            custom_noise_base=custom_noise_base,
            policy="actual",
            target_rounds=set(target_rounds),
            capture=True,
        )
        forced_mine, forced_counts, _, _ = _run_branch(
            sim,
            draft_history,
            start_pick=start_pick,
            last_pick=last_pick,
            archetypes=archetypes,
            market_latent=market_latent,
            custom_noise_base=custom_noise_base,
            policy="starter_first",
            target_rounds=set(),
            capture=False,
        )

        perf_rng = np.random.default_rng(int(seed) + 1_000_003 + run * 10009)
        run_proj, _ = sim._sample_run_projection(perf_rng)
        actual_eval = sim.evaluate_roster(actual_mine, projection_override=run_proj)
        forced_eval = sim.evaluate_roster(forced_mine, projection_override=run_proj)
        actual_scores[run] = float(actual_eval["lineup_score"])
        forced_scores[run] = float(forced_eval["lineup_score"])
        actual_vorp[run] = float(actual_eval["starter_vorp"])
        forced_vorp[run] = float(forced_eval["starter_vorp"])
        actual_complete[run] = roster_state(sim, actual_counts)["offense_complete"]
        forced_complete[run] = roster_state(sim, forced_counts)["offense_complete"]
        run_deltas[run] = actual_scores[run] - forced_scores[run]

        for row in checkpoints:
            row["Run"] = run
            all_checkpoints.append(row)
        for row in deferrals:
            row["Run"] = run
            all_deferrals.append(row)

        if progress is not None and ((run + 1) == int(runs) or (run + 1) % max(1, int(runs) // 10) == 0):
            progress(run + 1, int(runs))

    checkpoint_df = pd.DataFrame(all_checkpoints)
    deferral_df = pd.DataFrame(all_deferrals)
    summary = _summarize_checkpoints(checkpoint_df, int(runs))
    deferrals = _summarize_deferrals(deferral_df, run_deltas, int(runs))

    diff = actual_scores - forced_scores
    comparison = pd.DataFrame(
        [
            {
                "Actual Value-Aware Avg Lineup": round(float(np.mean(actual_scores)), 2),
                "Starter-First Avg Lineup": round(float(np.mean(forced_scores)), 2),
                "Value-Aware Delta": round(float(np.mean(diff)), 2),
                "Value-Aware Wins %": round(100.0 * float(np.mean(diff > 1e-9)), 1),
                "Ties %": round(100.0 * float(np.mean(np.isclose(diff, 0.0, atol=1e-9))), 1),
                "Value-Aware P25 Delta": round(float(np.percentile(diff, 25)), 2),
                "Value-Aware P50 Delta": round(float(np.percentile(diff, 50)), 2),
                "Value-Aware P75 Delta": round(float(np.percentile(diff, 75)), 2),
                "Actual Avg Starter VORP": round(float(np.mean(actual_vorp)), 2),
                "Starter-First Avg Starter VORP": round(float(np.mean(forced_vorp)), 2),
                "Actual Offense Complete by R13 %": round(100.0 * float(np.mean(actual_complete)), 1),
                "Starter-First Offense Complete by R13 %": round(100.0 * float(np.mean(forced_complete)), 1),
                "Avg Bench-Before-Complete Deferrals": round(len(deferral_df) / max(int(runs), 1), 2),
                "Runs": int(runs),
            }
        ]
    )

    meta = {
        "rounds": list(target_rounds),
        "runs": int(runs),
        "seed": int(seed),
        "start_pick": int(start_pick),
        "last_pick": int(last_pick),
        "lookahead_enabled": bool(sim.lookahead_enabled),
        "lookahead_rounds": sorted(int(x) for x in sim.lookahead_rounds),
    }
    return summary, deferrals, comparison, meta
