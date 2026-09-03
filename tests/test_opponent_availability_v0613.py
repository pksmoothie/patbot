from __future__ import annotations

import numpy as np
import pandas as pd

import patbot
from patbot.config import load_config
from patbot.decision_strategy import expected_position_demand
from patbot.draft import DraftEngine
from patbot.opponent_availability import opponent_availability_penalty
from patbot.sim import FastDraftSimulator


def _make_pool():
    rows = []
    pid = 1
    for pos, count, top in [
        ("RB", 30, 300),
        ("WR", 40, 305),
        ("QB", 20, 400),
        ("TE", 20, 240),
        ("K", 12, 150),
        ("DEF", 12, 145),
    ]:
        for i in range(count):
            rows.append(
                {
                    "player_id": str(pid),
                    "name": f"{pos}{i + 1}",
                    "team": "X",
                    "pos": pos,
                    "adp": float(pid),
                    "proj_points": float(top - i),
                    "injury_risk": 0.0,
                    "expert_rank": float(pid),
                    "is_rookie": False,
                }
            )
            pid += 1
    return pd.DataFrame(rows)


def _sim():
    cfg = load_config("config/league.yaml")
    return FastDraftSimulator(DraftEngine(_make_pool(), cfg))


def _counts(sim, **positions):
    counts = np.zeros(len(sim.POSITIONS), dtype=np.int16)
    for pos, value in positions.items():
        counts[sim.pos_to_code[pos]] = int(value)
    return counts


def test_v0614_is_installed():
    assert patbot.__version__ == "0.6.14"


def test_qb2_availability_guardrail_is_separate_from_ordinary_need_penalty():
    sim = _sim()
    with_gaps = _counts(sim, QB=1, RB=2, WR=1, TE=1)
    filled = _counts(sim, QB=1, RB=2, WR=3, TE=1)

    base_gap = sim._base_roster_need_penalty(with_gaps, 8)
    base_filled = sim._base_roster_need_penalty(filled, 8)
    availability_gap = opponent_availability_penalty(sim, with_gaps, 8)
    availability_filled = opponent_availability_penalty(sim, filled, 8)
    qb = sim.pos == "QB"

    # The league-calibrated QB2 baseline remains 34 either way. The separate
    # anti-distortion layer contributes the full +36 when two WR starters are
    # open, so James's 0.40 roster_need_strength cannot dilute it.
    assert np.allclose(base_gap[qb], 34.0)
    assert np.allclose(base_filled[qb], 34.0)
    assert np.allclose(availability_gap[qb], 36.0)
    assert np.allclose(availability_filled[qb], 0.0)


def test_james_qb2_tendency_yields_to_open_wr_starters_but_is_not_deleted():
    sim = _sim()
    available = np.zeros(sim.n, dtype=bool)
    qb_idx = int(np.where(sim.pos == "QB")[0][0])
    wr_idx = int(np.where(sim.pos == "WR")[0][0])
    available[[qb_idx, wr_idx]] = True

    market = np.full(sim.n, 999.0)
    custom = np.full(sim.n, 999.0)
    market[qb_idx] = custom[qb_idx] = 78.0
    market[wr_idx] = custom[wr_idx] = 82.0
    profile = sim._manager_profile(4, "casual")

    # With two WR starters still open, the independent availability guardrail
    # outweighs James's real history-based QB2 nudge and the room takes the WR.
    with_gaps = _counts(sim, QB=1, RB=3, WR=1, TE=1)
    picked_with_gaps = sim.opponent_pick(
        available.copy(), market, custom, with_gaps, 8, profile
    )
    assert picked_with_gaps == wr_idx

    # Once the core RB/WR starters are filled, the guardrail disappears. The
    # same modest QB fall can still activate James's promoted QB2 tendency.
    filled = _counts(sim, QB=1, RB=3, WR=3, TE=1)
    picked_filled = sim.opponent_pick(
        available.copy(), market, custom, filled, 8, profile
    )
    assert picked_filled == qb_idx


def test_extreme_rb_wr_hoarding_guardrail_only_starts_after_two_extra_players():
    sim = _sim()

    rb3_wr0 = _counts(sim, RB=3, WR=0)
    rb4_wr0 = _counts(sim, RB=4, WR=0)
    wr4_rb0 = _counts(sim, RB=0, WR=4)
    wr5_rb0 = _counts(sim, RB=0, WR=5)

    rb = sim.pos == "RB"
    wr = sim.pos == "WR"

    assert np.allclose(opponent_availability_penalty(sim, rb3_wr0, 7)[rb], 0.0)
    assert np.allclose(opponent_availability_penalty(sim, rb4_wr0, 7)[rb], 30.0)
    assert np.allclose(opponent_availability_penalty(sim, wr4_rb0, 7)[wr], 0.0)
    assert np.allclose(opponent_availability_penalty(sim, wr5_rb0, 7)[wr], 30.0)


def test_extreme_value_can_still_beat_the_soft_hoarding_guardrail():
    sim = _sim()
    counts = _counts(sim, RB=4, WR=0)
    available = np.zeros(sim.n, dtype=bool)
    rb_idx = int(np.where(sim.pos == "RB")[0][0])
    wr_idx = int(np.where(sim.pos == "WR")[0][0])
    available[[rb_idx, wr_idx]] = True

    market = np.full(sim.n, 999.0)
    custom = np.full(sim.n, 999.0)
    market[rb_idx] = 1.0
    market[wr_idx] = 50.0
    custom[rb_idx] = 1.0
    custom[wr_idx] = 50.0

    picked = sim.opponent_pick(
        available,
        market,
        custom,
        counts,
        7,
        {
            "market_weight": 1.0,
            "custom_weight": 0.0,
            "roster_need_strength": 1.0,
            "randomness": 0.0,
        },
    )
    assert picked == rb_idx


def _qb_history(*, james_wr_count: int) -> list[dict]:
    history = []
    for slot in range(4, 13):
        history.append({"owner_slot": slot, "pos": "QB", "player_id": f"q{slot}"})
        for i in range(2):
            history.append({"owner_slot": slot, "pos": "RB", "player_id": f"r{slot}_{i}"})
        wr_count = james_wr_count if slot == 4 else 3
        for i in range(wr_count):
            history.append({"owner_slot": slot, "pos": "WR", "player_id": f"w{slot}_{i}"})
    return history


def test_expected_qb2_demand_respects_actual_core_starter_gaps():
    cfg = load_config("config/league.yaml")
    engine = DraftEngine(_make_pool(), cfg)

    james_missing_two_wr = expected_position_demand(
        engine,
        pos="QB",
        current_pick=99,
        draft_history=_qb_history(james_wr_count=1),
    )
    james_core_filled = expected_position_demand(
        engine,
        pos="QB",
        current_pick=99,
        draft_history=_qb_history(james_wr_count=3),
    )

    # Legacy total is 0.60: James at 0.28 plus eight generic 0.04 backups.
    # With two WR starter gaps, James's contextual expected rate is quartered
    # to 0.07, reducing aggregate expected demand to 0.39.
    assert np.isclose(james_core_filled, 0.60)
    assert np.isclose(james_missing_two_wr, 0.39)
