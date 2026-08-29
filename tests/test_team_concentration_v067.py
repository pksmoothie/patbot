from __future__ import annotations

import pandas as pd

from patbot.team_concentration import (
    candidate_concentration_penalty,
    concentration_settings,
    pair_penalty,
)


def _players():
    return pd.DataFrame(
        [
            {"player_id": "wr1", "name": "WR One", "team": "AAA", "pos": "WR"},
            {"player_id": "wr2", "name": "WR Two", "team": "AAA", "pos": "WR"},
            {"player_id": "rb1", "name": "RB One", "team": "AAA", "pos": "RB"},
            {"player_id": "te1", "name": "TE One", "team": "AAA", "pos": "TE"},
            {"player_id": "qb1", "name": "QB One", "team": "AAA", "pos": "QB"},
            {"player_id": "wr3", "name": "WR Three", "team": "BBB", "pos": "WR"},
        ]
    )


def test_qb_stack_is_not_penalized():
    assert pair_penalty("QB", "WR", {}) == 0.0
    assert pair_penalty("QB", "TE", {}) == 0.0
    assert pair_penalty("QB", "RB", {}) == 0.0


def test_same_team_wr_wr_is_stronger_than_rb_wr():
    assert pair_penalty("WR", "WR", {}) > pair_penalty("RB", "WR", {}) > 0


def test_candidate_penalty_escalates_for_third_skill_player():
    frame = _players()
    second, _ = candidate_concentration_penalty(
        frame,
        candidate_idx=1,
        roster_indices=[0],
        round_no=2,
        config={},
    )
    third, note = candidate_concentration_penalty(
        frame,
        candidate_idx=2,
        roster_indices=[0, 1],
        round_no=3,
        config={},
    )
    assert second > 0
    assert third > second
    assert "skill player #3" in note


def test_penalty_fades_late_but_is_not_a_hard_ban():
    frame = _players()
    early, _ = candidate_concentration_penalty(
        frame,
        candidate_idx=2,
        roster_indices=[0],
        round_no=5,
        config={},
    )
    upside, _ = candidate_concentration_penalty(
        frame,
        candidate_idx=2,
        roster_indices=[0],
        round_no=9,
        config={},
    )
    lottery, _ = candidate_concentration_penalty(
        frame,
        candidate_idx=2,
        roster_indices=[0],
        round_no=12,
        config={},
    )
    special, _ = candidate_concentration_penalty(
        frame,
        candidate_idx=2,
        roster_indices=[0],
        round_no=14,
        config={},
    )
    assert early > upside > lottery > special
    assert special == 0.0


def test_unrelated_team_has_zero_penalty():
    frame = _players()
    penalty, note = candidate_concentration_penalty(
        frame,
        candidate_idx=5,
        roster_indices=[0],
        round_no=2,
        config={},
    )
    assert penalty == 0.0
    assert note == ""


def test_default_lineup_penalties_preserve_pair_ordering():
    settings = concentration_settings({})
    assert settings["lineup_pair_penalty"]["WR|WR"] > settings["lineup_pair_penalty"]["RB|WR"]
