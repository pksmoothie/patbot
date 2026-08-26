from types import SimpleNamespace

import numpy as np

from patbot.construction_audit import (
    classify_pick,
    roster_state,
    starter_first_allowed_positions,
)


ROSTER_CFG = {
    "QB": 1,
    "RB": 2,
    "WR": 3,
    "TE": 1,
    "FLEX": 1,
    "K": 1,
    "DEF": 1,
    "flex_eligible": ["RB", "WR", "TE"],
}


class DummySim:
    POSITIONS = ("QB", "RB", "WR", "TE", "K", "DEF")

    def __init__(self):
        self.pos_to_code = {p: i for i, p in enumerate(self.POSITIONS)}
        self.engine = SimpleNamespace(roster_cfg=ROSTER_CFG)
        self.pos = np.array(["QB", "RB", "WR", "TE", "K", "DEF"])


def _counts(qb=0, rb=0, wr=0, te=0, k=0, de=0):
    return np.array([qb, rb, wr, te, k, de], dtype=np.int16)


def test_starter_first_counterfactual_forces_only_missing_base_positions():
    sim = DummySim()
    allowed = starter_first_allowed_positions(sim, _counts(qb=0, rb=3, wr=4, te=0))
    assert allowed == {"QB", "TE"}


def test_starter_first_counterfactual_fills_flex_after_base_starters():
    sim = DummySim()
    allowed = starter_first_allowed_positions(sim, _counts(qb=1, rb=2, wr=3, te=1))
    assert allowed == {"RB", "WR", "TE"}


def test_starter_first_counterfactual_releases_once_offense_complete():
    sim = DummySim()
    allowed = starter_first_allowed_positions(sim, _counts(qb=1, rb=3, wr=3, te=1))
    assert allowed is None


def test_roster_state_distinguishes_base_starters_from_full_offense():
    sim = DummySim()
    state = roster_state(sim, _counts(qb=1, rb=2, wr=3, te=1))
    assert state["qb1_filled"]
    assert state["te1_filled"]
    assert not state["flex_filled"]
    assert not state["offense_complete"]


def test_classify_pick_calls_depth_before_completion_a_bench_deferral():
    sim = DummySim()
    counts = _counts(qb=0, rb=3, wr=3, te=1)
    assert classify_pick(sim, counts, 1) == "RB/WR bench before complete"


def test_classify_pick_distinguishes_qb1_qb2_flex_and_post_complete_bench():
    sim = DummySim()
    assert classify_pick(sim, _counts(qb=0, rb=2, wr=3, te=1), 0) == "QB1"
    assert classify_pick(sim, _counts(qb=1, rb=2, wr=3, te=1), 0) == "QB2"
    assert classify_pick(sim, _counts(qb=1, rb=2, wr=3, te=1), 1) == "FLEX fill"
    assert classify_pick(sim, _counts(qb=1, rb=3, wr=3, te=1), 2) == "RB/WR bench after complete"
