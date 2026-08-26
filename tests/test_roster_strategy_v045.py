from collections import Counter

import numpy as np

from patbot.roster_strategy import (
    apply_patbot_array_constraints,
    offensive_starters_complete,
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

POS = np.array(["QB", "RB", "WR", "TE", "K", "DEF"])
POS_TO_CODE = {p: i for i, p in enumerate(POS)}
CFG = {
    "special_teams_strategy": {
        "draft": {
            "defense_round": 14,
            "kicker_round": 15,
            "rostered_defenses": 1,
            "rostered_kickers": 1,
        }
    }
}


def _counts(qb=0, rb=0, wr=0, te=0, k=0, de=0):
    return np.array([qb, rb, wr, te, k, de], dtype=np.int16)


def _apply(counts, round_no=8):
    return apply_patbot_array_constraints(
        np.arange(6, dtype=float) + 10.0,
        positions=POS,
        roster_counts=counts,
        pos_to_code=POS_TO_CODE,
        roster_cfg=ROSTER_CFG,
        round_no=round_no,
        config=CFG,
    )


def test_offense_not_complete_when_qb_missing_even_with_bench_depth():
    counts = Counter({"QB": 0, "RB": 3, "WR": 4, "TE": 1})
    assert not offensive_starters_complete(counts, ROSTER_CFG)


def test_offense_complete_requires_flex_excess():
    base = Counter({"QB": 1, "RB": 2, "WR": 3, "TE": 1})
    assert not offensive_starters_complete(base, ROSTER_CFG)
    base["WR"] += 1
    assert offensive_starters_complete(base, ROSTER_CFG)


def test_qb2_is_hard_blocked_once_qb1_exists():
    score = _apply(_counts(qb=1, rb=1, wr=2, te=0), round_no=7)
    assert score[POS_TO_CODE["QB"]] < -1e8


def test_missing_starters_are_not_force_filled_before_offense_complete():
    # We have already built RB/WR depth but still need QB1 and TE1. All four
    # offensive positions remain legal; the value model is free to take another
    # RB/WR instead of mechanically filling QB/TE.
    score = _apply(_counts(qb=0, rb=3, wr=4, te=0), round_no=7)
    for pos in ("QB", "RB", "WR", "TE"):
        assert score[POS_TO_CODE[pos]] > -1e8


def test_after_offense_complete_only_rb_wr_remain_as_offensive_picks():
    score = _apply(_counts(qb=1, rb=3, wr=3, te=1), round_no=10)
    assert score[POS_TO_CODE["QB"]] < -1e8
    assert score[POS_TO_CODE["TE"]] < -1e8
    assert score[POS_TO_CODE["RB"]] > -1e8
    assert score[POS_TO_CODE["WR"]] > -1e8


def test_round_14_forces_defense_when_missing():
    score = _apply(_counts(qb=1, rb=4, wr=5, te=1, k=0, de=0), round_no=14)
    assert score[POS_TO_CODE["DEF"]] > -1e8
    assert all(score[i] < -1e8 for i in range(len(POS)) if i != POS_TO_CODE["DEF"])


def test_round_15_forces_kicker_when_missing():
    score = _apply(_counts(qb=1, rb=4, wr=5, te=1, k=0, de=1), round_no=15)
    assert score[POS_TO_CODE["K"]] > -1e8
    assert all(score[i] < -1e8 for i in range(len(POS)) if i != POS_TO_CODE["K"])
