import numpy as np

from patbot.opponent_history import apply_history_adjustments, load_owner_history


POSITIONS = np.array(["QB", "RB", "WR", "TE"], dtype=object)
POS_TO_CODE = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}


def _scores():
    return np.array([50.0, 50.0, 50.0, 50.0])


def _apply(owner: str, round_no: int, qb_count: int = 0):
    counts = np.array([qb_count, 0, 0, 0], dtype=int)
    tendency = load_owner_history()[owner]
    return apply_history_adjustments(
        _scores(),
        positions=POSITIONS,
        pos_to_code=POS_TO_CODE,
        roster_counts=counts,
        round_no=round_no,
        tendency=tendency,
    )


def test_only_six_history_tendencies_are_promoted():
    assert set(load_owner_history()) == {"Vinnie", "Dave", "Paul", "Potter", "Mateo", "James"}


def test_vinnie_gets_rb_round_one_and_early_qb_nudges():
    round_one = _apply("Vinnie", 1, qb_count=0)
    assert round_one[1] == 40.0
    assert round_one[0] == 50.0

    round_three = _apply("Vinnie", 3, qb_count=0)
    assert round_three[0] == 36.0


def test_paul_and_potter_are_penalized_for_very_early_qb():
    paul = _apply("Paul", 4, qb_count=0)
    potter = _apply("Potter", 4, qb_count=0)
    assert paul[0] == 60.0
    assert potter[0] == 64.0

    assert _apply("Paul", 7, qb_count=0)[0] == 50.0
    assert _apply("Potter", 7, qb_count=0)[0] == 50.0


def test_mateo_is_nudged_toward_rb_wr_and_away_from_early_qb():
    score = _apply("Mateo", 4, qb_count=0)
    assert score[0] == 64.0
    assert score[1] == 41.0
    assert score[2] == 41.0
    assert score[3] == 50.0


def test_dave_first_qb_and_james_second_qb_tendencies_are_role_specific():
    dave = _apply("Dave", 3, qb_count=0)
    assert dave[0] == 38.0
    assert _apply("Dave", 3, qb_count=1)[0] == 50.0

    james_no_qb = _apply("James", 10, qb_count=0)
    james_one_qb = _apply("James", 10, qb_count=1)
    james_two_qb = _apply("James", 10, qb_count=2)
    assert james_no_qb[0] == 50.0
    assert james_one_qb[0] == 40.0
    assert james_two_qb[0] == 50.0
