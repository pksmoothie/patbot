from __future__ import annotations

from collections import Counter
from typing import Callable

import numpy as np
import pandas as pd

from .draft import snake_pick
from .roster_strategy import offensive_starters_complete, te1_quality_bucket
from .sim import FastDraftSimulator


NEG_INF = -1_000_000_000.0


def _seed_from_history(sim: FastDraftSimulator, draft_history: list[dict]):
    drafted_idx: set[int] = set()
    mine: list[int] = []
    for record in draft_history:
        idx = sim.id_to_idx.get(str(record.get("player_id", "")))
        if idx is None:
            continue
        drafted_idx.add(int(idx))
        if int(record.get("owner_slot", -1)) == sim.slot:
            mine.append(int(idx))
    return drafted_idx, mine


def _set_identity(sim: FastDraftSimulator, mine: list[int], drafted_idx: set[int]) -> None:
    mine_set = {int(i) for i in mine}
    sim._patbot_owned_idxs = set(mine_set)
    sim._opponent_owned_idxs = {int(i) for i in drafted_idx if int(i) not in mine_set}


def _roster_counts(sim: FastDraftSimulator, mine: list[int]) -> np.ndarray:
    counts = np.zeros(len(sim.POSITIONS), dtype=np.int16)
    for idx in mine:
        code = int(sim.pos_code[int(idx)])
        if code >= 0:
            counts[code] += 1
    return counts


def _counts_counter(sim: FastDraftSimulator, counts: np.ndarray) -> Counter:
    return Counter(
        {
            pos: int(counts[code])
            for pos, code in sim.pos_to_code.items()
            if 0 <= int(code) < len(counts)
        }
    )


def _pick_role(sim: FastDraftSimulator, counts: np.ndarray, idx: int) -> str:
    pos = str(sim.pos[int(idx)]).upper()
    c = _counts_counter(sim, counts)
    rcfg = sim.engine.roster_cfg
    if pos == "QB":
        return "QB1" if c["QB"] == 0 else "QB2"
    if pos == "TE":
        return "TE1" if c["TE"] == 0 else "TE2"
    if pos in {"RB", "WR"}:
        if c[pos] < int(rcfg.get(pos, 0)):
            return f"{pos} starter"
        if offensive_starters_complete(c, rcfg):
            return "RB/WR bench after complete"
        return "RB/WR/FLEX value"
    return pos


def _position_rank(sim: FastDraftSimulator, idx: int) -> int:
    pos = str(sim.pos[int(idx)]).upper()
    peers = np.where(np.asarray(sim.pos).astype(str) == pos)[0]
    if len(peers) == 0:
        return 999
    ordered = peers[np.argsort(sim.vorp[peers])[::-1]]
    lookup = {int(player_idx): rank + 1 for rank, player_idx in enumerate(ordered)}
    return int(lookup.get(int(idx), 999))


def starter_quality(sim: FastDraftSimulator, idx: int) -> str:
    pos = str(sim.pos[int(idx)]).upper()
    if pos == "TE":
        return str(
            te1_quality_bucket(
                positions=sim.pos,
                vorp=sim.vorp,
                roster_indices=[int(idx)],
            )
            or "weak"
        )
    rank = _position_rank(sim, idx)
    if rank <= 3:
        return "elite"
    if rank <= 8:
        return "solid"
    return "weak"


def _legal_order(scores: np.ndarray, available: np.ndarray) -> np.ndarray:
    legal = np.where(np.asarray(available, dtype=bool) & (scores > NEG_INF / 2))[0]
    if len(legal) == 0:
        return legal
    return legal[np.argsort(scores[legal])[::-1]]


def _record_pick(
    sim: FastDraftSimulator,
    *,
    pick: int,
    idx: int,
    owner_slot: int,
    score: float | None = None,
) -> dict:
    return {
        "Overall": int(pick),
        "Round": (int(pick) - 1) // sim.teams + 1,
        "Owner Slot": int(owner_slot),
        "Player": str(sim.names[int(idx)]),
        "Pos": str(sim.pos[int(idx)]),
        "PatBot Score": round(float(score), 2) if score is not None and np.isfinite(score) else np.nan,
        "LWS": round(float(sim.league_winner_score[int(idx)]), 1),
        "Q90": round(float(sim.q90_points[int(idx)]), 1),
    }


def _manager_name(sim: FastDraftSimulator, slot: int) -> str:
    order = sim.cfg.get("league", {}).get("draft_order", {})
    return str(order.get(int(slot), order.get(str(slot), f"Slot {slot}")))


def assign_roster_slots(sim: FastDraftSimulator, mine: list[int]) -> pd.DataFrame:
    """Assign a completed 15-player draft to the configured starting slots/bench."""
    rcfg = sim.engine.roster_cfg
    by_pos: dict[str, list[int]] = {}
    for pos in sim.POSITIONS:
        idxs = [int(i) for i in mine if str(sim.pos[int(i)]).upper() == pos]
        idxs.sort(key=lambda i: float(sim.proj[i]), reverse=True)
        by_pos[pos] = idxs

    rows: list[dict] = []
    used: set[int] = set()

    def add_slot(slot_name: str, idx: int):
        used.add(int(idx))
        rows.append(
            {
                "Slot": slot_name,
                "Player": str(sim.names[int(idx)]),
                "Pos": str(sim.pos[int(idx)]),
                "Projection": round(float(sim.proj[int(idx)]), 1),
                "VORP": round(float(sim.vorp[int(idx)]), 1),
                "LWS": round(float(sim.league_winner_score[int(idx)]), 1),
            }
        )

    for pos in ("QB", "RB", "WR", "TE"):
        need = int(rcfg.get(pos, 0))
        chosen = by_pos.get(pos, [])[:need]
        for j, idx in enumerate(chosen, start=1):
            label = pos if need == 1 else f"{pos}{j}"
            add_slot(label, idx)

    flex_pool = [
        int(i)
        for i in mine
        if int(i) not in used and str(sim.pos[int(i)]).upper() in set(rcfg.get("flex_eligible", []))
    ]
    flex_pool.sort(key=lambda i: float(sim.proj[i]), reverse=True)
    for j, idx in enumerate(flex_pool[: int(rcfg.get("FLEX", 0))], start=1):
        add_slot("FLEX" if int(rcfg.get("FLEX", 0)) == 1 else f"FLEX{j}", idx)

    for pos, slot_name in (("DEF", "DEF"), ("K", "K")):
        for idx in by_pos.get(pos, [])[: int(rcfg.get(pos, 0))]:
            add_slot(slot_name, idx)

    bench = [int(i) for i in mine if int(i) not in used]
    draft_order = {int(idx): j for j, idx in enumerate(mine)}
    bench.sort(key=lambda i: draft_order.get(int(i), 999))
    for j, idx in enumerate(bench, start=1):
        add_slot(f"BN{j}", idx)

    return pd.DataFrame(rows)


def run_representative_draft(
    engine,
    draft_history: list[dict],
    *,
    forced_first_player_id: str | None = None,
    through_round: int = 15,
    seed: int | None = None,
    early_rounds: int = 6,
    board_depth: int = 6,
):
    """Run one reproducible room and expose PatBot's board at R1-R6.

    This is intentionally a concrete example, not a forecast. It is useful for
    seeing exactly which players PatBot passes on and the two picks immediately
    after each PatBot selection.
    """
    sim = FastDraftSimulator(engine)
    if seed is None:
        seed = int(sim.comparison_seed) + 848
    rng = np.random.default_rng(int(seed))

    drafted_idx, mine = _seed_from_history(sim, draft_history)
    available = np.ones(sim.n, dtype=bool)
    if drafted_idx:
        available[list(drafted_idx)] = False
    opp_counts = sim._seed_opponent_counts(draft_history)
    _set_identity(sim, mine, drafted_idx)
    my_counts = _roster_counts(sim, mine)

    latent_sd = np.maximum(sim.sd_floor, sim.adp * sim.sd_pct)
    archetypes = sim._archetype_assignments(rng)
    market_latent = np.maximum(1.0, rng.normal(sim.adp, latent_sd))
    custom_noise_base = rng.normal(0.0, np.maximum(3.0, sim.custom_rank * 0.06))

    start_pick = max([int(x.get("overall_pick", 0)) for x in draft_history] or [0]) + 1
    last_pick = snake_pick(int(through_round), sim.teams, sim.slot)
    first_patbot_pick = min(p for p in sim.my_picks if p >= start_pick)
    pick_records: dict[int, dict] = {}
    patbot_rows: list[dict] = []
    board_rows: list[dict] = []

    for pick in range(int(start_pick), int(last_pick) + 1):
        if not available.any():
            break
        round_no = (pick - 1) // sim.teams + 1

        if pick in sim.my_picks:
            scores = sim._patbot_score_vector(available, my_counts, pick)
            ordered = _legal_order(scores, available)

            if pick == first_patbot_pick and forced_first_player_id is not None:
                forced_idx = sim.id_to_idx.get(str(forced_first_player_id))
                if forced_idx is None or not bool(available[int(forced_idx)]):
                    raise ValueError("Forced first PatBot player is not available in this room.")
                idx = int(forced_idx)
            else:
                idx = int(
                    sim._lookahead_pick(
                        available,
                        my_counts,
                        pick,
                        opp_counts,
                        archetypes,
                        market_latent,
                        custom_noise_base,
                    )
                )

            if round_no <= int(early_rounds):
                top = ordered[: max(int(board_depth), 1)]
                if idx not in set(int(x) for x in top):
                    top = np.concatenate([top, np.array([idx], dtype=int)])
                for rank, candidate in enumerate(top, start=1):
                    board_rows.append(
                        {
                            "Round": int(round_no),
                            "Overall": int(pick),
                            "Board Rank": int(rank),
                            "Selected": int(candidate) == int(idx),
                            "Player": str(sim.names[int(candidate)]),
                            "Pos": str(sim.pos[int(candidate)]),
                            "Score": round(float(scores[int(candidate)]), 2),
                            "VORP": round(float(sim.vorp[int(candidate)]), 1),
                            "LWS": round(float(sim.league_winner_score[int(candidate)]), 1),
                            "Q90": round(float(sim.q90_points[int(candidate)]), 1),
                            "Pos Rank": _position_rank(sim, int(candidate)),
                        }
                    )

            role = _pick_role(sim, my_counts, idx)
            row = _record_pick(
                sim,
                pick=pick,
                idx=idx,
                owner_slot=sim.slot,
                score=float(scores[idx]),
            )
            row["Role"] = role
            patbot_rows.append(row)
            pick_records[int(pick)] = {**row, "Manager": _manager_name(sim, sim.slot)}

            available[idx] = False
            mine.append(int(idx))
            code = int(sim.pos_code[idx])
            if code >= 0:
                my_counts[code] += 1
            patbot_owned = set(getattr(sim, "_patbot_owned_idxs", set()))
            patbot_owned.add(int(idx))
            sim._patbot_owned_idxs = patbot_owned
        else:
            idx, _ = sim._take_opponent_pick(
                pick,
                available,
                opp_counts,
                archetypes,
                market_latent,
                custom_noise_base,
            )
            owner_slot = ((pick - 1) % sim.teams + 1) if round_no % 2 else sim.teams + 1 - ((pick - 1) % sim.teams + 1)
            row = _record_pick(sim, pick=pick, idx=idx, owner_slot=owner_slot)
            row["Manager"] = _manager_name(sim, owner_slot)
            pick_records[int(pick)] = row

    next_rows: list[dict] = []
    for row in patbot_rows:
        round_no = int(row["Round"])
        if round_no > int(early_rounds):
            continue
        p = int(row["Overall"])
        one = pick_records.get(p + 1, {})
        two = pick_records.get(p + 2, {})
        next_rows.append(
            {
                "Round": round_no,
                "PatBot Overall": p,
                "PatBot Pick": row["Player"],
                "Pos": row["Pos"],
                "Next Pick": one.get("Player", "—"),
                "Next Pick Manager": one.get("Manager", "—"),
                "Second Next Pick": two.get("Player", "—"),
                "Second Next Manager": two.get("Manager", "—"),
            }
        )

    return (
        pd.DataFrame(patbot_rows),
        pd.DataFrame(board_rows),
        pd.DataFrame(next_rows),
        assign_roster_slots(sim, mine),
        {"seed": int(seed), "through_round": int(through_round), "mine": list(mine)},
    )


def _simulate_wait_branch(
    sim: FastDraftSimulator,
    *,
    current_pick: int,
    final_pick: int,
    available: np.ndarray,
    my_counts: np.ndarray,
    mine: list[int],
    opp_counts: np.ndarray,
    archetypes: dict[int, str],
    market_latent: np.ndarray,
    custom_noise_base: np.ndarray,
    rbwr_idx: int,
    fill_pos: str,
    patbot_owned: set[int],
    opponent_owned: set[int],
):
    branch_available = available.copy()
    branch_counts = my_counts.copy()
    branch_mine = list(mine)
    branch_opp_counts = opp_counts.copy()
    sim._patbot_owned_idxs = set(patbot_owned)
    sim._opponent_owned_idxs = set(opponent_owned)

    rbwr_idx = int(rbwr_idx)
    branch_available[rbwr_idx] = False
    branch_mine.append(rbwr_idx)
    code = int(sim.pos_code[rbwr_idx])
    if code >= 0:
        branch_counts[code] += 1
    sim._patbot_owned_idxs = set(getattr(sim, "_patbot_owned_idxs", set())) | {rbwr_idx}

    next_pick = sim._next_my_pick(int(current_pick))
    for p in range(int(current_pick) + 1, min(int(next_pick), int(final_pick) + 1)):
        if p in sim.my_picks or not branch_available.any():
            break
        sim._take_opponent_pick(
            p,
            branch_available,
            branch_opp_counts,
            archetypes,
            market_latent,
            custom_noise_base,
        )

    fill_idx = None
    fill_score = np.nan
    if int(next_pick) <= int(final_pick) and branch_available.any():
        scores = sim._patbot_score_vector(branch_available, branch_counts, next_pick)
        mask = branch_available & (np.asarray(sim.pos).astype(str) == str(fill_pos)) & (scores > NEG_INF / 2)
        candidates = np.where(mask)[0]
        if len(candidates):
            fill_idx = int(candidates[np.argmax(scores[candidates])])
            fill_score = float(scores[fill_idx])
            branch_available[fill_idx] = False
            branch_mine.append(fill_idx)
            code = int(sim.pos_code[fill_idx])
            if code >= 0:
                branch_counts[code] += 1
            sim._patbot_owned_idxs = set(getattr(sim, "_patbot_owned_idxs", set())) | {fill_idx}
        else:
            fill_idx = int(
                sim._lookahead_pick(
                    branch_available,
                    branch_counts,
                    next_pick,
                    branch_opp_counts,
                    archetypes,
                    market_latent,
                    custom_noise_base,
                )
            )
            branch_available[fill_idx] = False
            branch_mine.append(fill_idx)
            code = int(sim.pos_code[fill_idx])
            if code >= 0:
                branch_counts[code] += 1

    for p in range(int(next_pick) + 1, int(final_pick) + 1):
        if not branch_available.any():
            break
        if p in sim.my_picks:
            idx = int(
                sim._lookahead_pick(
                    branch_available,
                    branch_counts,
                    p,
                    branch_opp_counts,
                    archetypes,
                    market_latent,
                    custom_noise_base,
                )
            )
            branch_available[idx] = False
            branch_mine.append(idx)
            code = int(sim.pos_code[idx])
            if code >= 0:
                branch_counts[code] += 1
            sim._patbot_owned_idxs = set(getattr(sim, "_patbot_owned_idxs", set())) | {idx}
        else:
            sim._take_opponent_pick(
                p,
                branch_available,
                branch_opp_counts,
                archetypes,
                market_latent,
                custom_noise_base,
            )

    return branch_mine, fill_idx, fill_score, int(next_pick)


def run_qb_te_timing_audit(
    engine,
    draft_history: list[dict],
    *,
    forced_first_player_id: str | None = None,
    runs: int = 300,
    through_round: int = 13,
    audit_through_round: int = 10,
    seed: int | None = None,
    progress: Callable[[int, int], None] | None = None,
):
    """Compare every QB1/TE1 pick with taking the best RB/WR and waiting one turn."""
    sim = FastDraftSimulator(engine)
    if seed is None:
        seed = int(sim.comparison_seed) + 8480
    start_pick = max([int(x.get("overall_pick", 0)) for x in draft_history] or [0]) + 1
    first_patbot_pick = min(p for p in sim.my_picks if p >= start_pick)
    final_pick = snake_pick(int(through_round), sim.teams, sim.slot)
    latent_sd = np.maximum(sim.sd_floor, sim.adp * sim.sd_pct)
    rows: list[dict] = []

    for run in range(int(runs)):
        rng = np.random.default_rng(int(seed) + run * 10007)
        drafted_idx, mine = _seed_from_history(sim, draft_history)
        available = np.ones(sim.n, dtype=bool)
        if drafted_idx:
            available[list(drafted_idx)] = False
        opp_counts = sim._seed_opponent_counts(draft_history)
        _set_identity(sim, mine, drafted_idx)
        my_counts = _roster_counts(sim, mine)

        archetypes = sim._archetype_assignments(rng)
        market_latent = np.maximum(1.0, rng.normal(sim.adp, latent_sd))
        custom_noise_base = rng.normal(0.0, np.maximum(3.0, sim.custom_rank * 0.06))
        pending: list[dict] = []

        for pick in range(int(start_pick), int(final_pick) + 1):
            if not available.any():
                break
            round_no = (pick - 1) // sim.teams + 1

            if pick in sim.my_picks:
                pre_available = available.copy()
                pre_counts = my_counts.copy()
                pre_mine = list(mine)
                pre_opp_counts = opp_counts.copy()
                pre_patbot_owned = set(getattr(sim, "_patbot_owned_idxs", set()))
                pre_opponent_owned = set(getattr(sim, "_opponent_owned_idxs", set()))
                scores = sim._patbot_score_vector(available, my_counts, pick)

                if pick == first_patbot_pick and forced_first_player_id is not None:
                    forced_idx = sim.id_to_idx.get(str(forced_first_player_id))
                    if forced_idx is None or not bool(available[int(forced_idx)]):
                        raise ValueError("Forced first PatBot player is not available in this room.")
                    idx = int(forced_idx)
                else:
                    idx = int(
                        sim._lookahead_pick(
                            available,
                            my_counts,
                            pick,
                            opp_counts,
                            archetypes,
                            market_latent,
                            custom_noise_base,
                        )
                    )

                chosen_pos = str(sim.pos[idx]).upper()
                role = _pick_role(sim, pre_counts, idx)
                actual_patbot_owned = set(getattr(sim, "_patbot_owned_idxs", set())) | {idx}
                actual_opponent_owned = set(getattr(sim, "_opponent_owned_idxs", set()))

                if (
                    round_no <= int(audit_through_round)
                    and role in {"QB1", "TE1"}
                ):
                    rbwr_mask = pre_available & np.isin(sim.pos, ["RB", "WR"]) & (scores > NEG_INF / 2)
                    rbwr_candidates = np.where(rbwr_mask)[0]
                    if len(rbwr_candidates):
                        rbwr_idx = int(rbwr_candidates[np.argmax(scores[rbwr_candidates])])
                        branch_mine, fill_idx, fill_score, next_pick = _simulate_wait_branch(
                            sim,
                            current_pick=pick,
                            final_pick=final_pick,
                            available=pre_available,
                            my_counts=pre_counts,
                            mine=pre_mine,
                            opp_counts=pre_opp_counts,
                            archetypes=archetypes,
                            market_latent=market_latent,
                            custom_noise_base=custom_noise_base,
                            rbwr_idx=rbwr_idx,
                            fill_pos=chosen_pos,
                            patbot_owned=pre_patbot_owned,
                            opponent_owned=pre_opponent_owned,
                        )
                        pending.append(
                            {
                                "Run": run,
                                "Round": round_no,
                                "Selected": str(sim.names[idx]),
                                "Pos": chosen_pos,
                                "Quality": starter_quality(sim, idx),
                                "Pos Rank": _position_rank(sim, idx),
                                "Selected Score": float(scores[idx]),
                                "Selected VORP": float(sim.vorp[idx]),
                                "RB/WR Passed": str(sim.names[rbwr_idx]),
                                "RB/WR Pos": str(sim.pos[rbwr_idx]),
                                "RB/WR Score": float(scores[rbwr_idx]),
                                "RB/WR LWS": float(sim.league_winner_score[rbwr_idx]),
                                "RB/WR Q90": float(sim.q90_points[rbwr_idx]),
                                "Score Edge Selected vs RB/WR": float(scores[idx]) - float(scores[rbwr_idx]),
                                "Wait Fill": str(sim.names[fill_idx]) if fill_idx is not None else "—",
                                "Wait Fill Pos Rank": _position_rank(sim, fill_idx) if fill_idx is not None else np.nan,
                                "Wait Fill VORP": float(sim.vorp[fill_idx]) if fill_idx is not None else np.nan,
                                "Wait Fill Score": float(fill_score) if np.isfinite(fill_score) else np.nan,
                                "Wait Until Overall": int(next_pick),
                                "VORP Cost of Waiting": float(sim.vorp[idx]) - float(sim.vorp[fill_idx]) if fill_idx is not None else np.nan,
                                "Branch Mine": branch_mine,
                            }
                        )

                sim._patbot_owned_idxs = actual_patbot_owned
                sim._opponent_owned_idxs = actual_opponent_owned
                available[idx] = False
                mine.append(idx)
                code = int(sim.pos_code[idx])
                if code >= 0:
                    my_counts[code] += 1
            else:
                sim._take_opponent_pick(
                    pick,
                    available,
                    opp_counts,
                    archetypes,
                    market_latent,
                    custom_noise_base,
                )

        perf_rng = np.random.default_rng(int(seed) + 1_000_003 + run * 10009)
        run_proj, _ = sim._sample_run_projection(perf_rng)
        actual_eval = sim.evaluate_roster(mine, projection_override=run_proj)
        actual_score = float(actual_eval["lineup_score"])
        for event in pending:
            branch_eval = sim.evaluate_roster(event.pop("Branch Mine"), projection_override=run_proj)
            event["Actual Final Lineup"] = actual_score
            event["Wait Final Lineup"] = float(branch_eval["lineup_score"])
            event["Wait Delta vs Fill Now"] = float(branch_eval["lineup_score"]) - actual_score
            rows.append(event)

        if progress is not None and ((run + 1) == int(runs) or (run + 1) % max(1, int(runs) // 10) == 0):
            progress(run + 1, int(runs))

    events = pd.DataFrame(rows)
    if events.empty:
        return pd.DataFrame(), pd.DataFrame(), events, {"runs": int(runs), "seed": int(seed)}

    quality_rows: list[dict] = []
    for (pos, quality), group in events.groupby(["Pos", "Quality"], sort=True):
        quality_rows.append(
            {
                "Pos": str(pos),
                "Quality": str(quality),
                "Events": int(len(group)),
                "Event % of Runs": round(100.0 * len(group) / max(int(runs), 1), 1),
                "Avg Round": round(float(group["Round"].mean()), 2),
                "Avg Selected Pos Rank": round(float(group["Pos Rank"].mean()), 1),
                "Avg RB/WR LWS Passed": round(float(group["RB/WR LWS"].mean()), 1),
                "Avg Score Edge Selected vs RB/WR": round(float(group["Score Edge Selected vs RB/WR"].mean()), 2),
                "Avg VORP Cost of Waiting": round(float(group["VORP Cost of Waiting"].mean()), 2),
                "Avg Wait Delta vs Fill Now": round(float(group["Wait Delta vs Fill Now"].mean()), 2),
                "Wait Wins %": round(100.0 * float((group["Wait Delta vs Fill Now"] > 1e-9).mean()), 1),
                "Wait P25 Delta": round(float(np.percentile(group["Wait Delta vs Fill Now"], 25)), 2),
                "Wait P50 Delta": round(float(np.percentile(group["Wait Delta vs Fill Now"], 50)), 2),
                "Wait P75 Delta": round(float(np.percentile(group["Wait Delta vs Fill Now"], 75)), 2),
            }
        )

    decision_rows: list[dict] = []
    keys = ["Pos", "Quality", "Selected", "RB/WR Passed", "Wait Fill"]
    for key, group in events.groupby(keys, dropna=False, sort=True):
        decision_rows.append(
            {
                "Pos": str(key[0]),
                "Quality": str(key[1]),
                "Selected": str(key[2]),
                "RB/WR Passed": str(key[3]),
                "Wait Fill": str(key[4]),
                "Times": int(len(group)),
                "Draft %": round(100.0 * len(group) / max(int(runs), 1), 1),
                "Avg Round": round(float(group["Round"].mean()), 2),
                "Selected Pos Rank": round(float(group["Pos Rank"].mean()), 1),
                "RB/WR LWS": round(float(group["RB/WR LWS"].mean()), 1),
                "Avg Immediate Score Edge": round(float(group["Score Edge Selected vs RB/WR"].mean()), 2),
                "Avg VORP Cost of Waiting": round(float(group["VORP Cost of Waiting"].mean()), 2),
                "Avg Wait Delta": round(float(group["Wait Delta vs Fill Now"].mean()), 2),
                "Wait Wins %": round(100.0 * float((group["Wait Delta vs Fill Now"] > 1e-9).mean()), 1),
            }
        )

    quality_summary = pd.DataFrame(quality_rows).sort_values(["Pos", "Quality"]).reset_index(drop=True)
    decisions = pd.DataFrame(decision_rows).sort_values(
        ["Times", "Avg Wait Delta"], ascending=[False, False]
    ).reset_index(drop=True)
    clean_events = events.drop(columns=[c for c in ["Run"] if c in events.columns]).copy()
    for c in clean_events.select_dtypes(include=["float"]).columns:
        clean_events[c] = clean_events[c].round(2)
    return quality_summary, decisions, clean_events, {
        "runs": int(runs),
        "seed": int(seed),
        "through_round": int(through_round),
        "audit_through_round": int(audit_through_round),
    }
