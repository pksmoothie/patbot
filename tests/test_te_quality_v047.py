import numpy as np

from patbot.roster_strategy import (
    apply_patbot_array_constraints,
    apply_te2_quality_gate_array,
    te1_quality_bucket,
)


POSITIONS = np.array(["TE", "TE", "TE", "TE", "RB", "WR", "QB", "K", "DEF"])
VORP = np.array([100.0, 90.0, 80.0, 70.0, 60.0, 55.0, 50.0, 0.0, 0.0])

BASE_STRATEGY = {
    "max_qb_drafted": 1,
    "max_te_drafted": 2,
    "post_offense_complete_positions": ["RB", "WR"],
    "te2_quality_strategy": {
        "enabled": True,
        "elite_te1_top_n": 1,
        "solid_te1_top_n": 2,
        "elite_required_score_edge_over_best_rbwr": 7.5,
        "solid_required_score_edge_over_best_rbwr": 3.0,
        "weak_te1_unrestricted": True,
    },
}


def test_te1_quality_bucket_is_model_rank_based():
    assert te1_quality_bucket(
        positions=POSITIONS, vorp=VORP, roster_indices=[0], strategy=BASE_STRATEGY
    ) == "elite"
    assert te1_quality_bucket(
        positions=POSITIONS, vorp=VORP, roster_indices=[1], strategy=BASE_STRATEGY
    ) == "solid"
    assert te1_quality_bucket(
        positions=POSITIONS, vorp=VORP, roster_indices=[2], strategy=BASE_STRATEGY
    ) == "weak"


def test_elite_te1_blocks_normal_te2_value():
    score = np.array([0.0, 80.0, 78.0, 76.0, 75.0, 74.0, 70.0, -1e9, -1e9])
    gated = apply_te2_quality_gate_array(
        score,
        positions=POSITIONS,
        vorp=VORP,
        roster_indices=[0],
        strategy=BASE_STRATEGY,
    )
    assert gated[1] < -1e8
    assert gated[4] == 75.0


def test_elite_te1_allows_only_wide_flex_edge():
    score = np.array([0.0, 84.0, 78.0, 76.0, 75.0, 74.0, 70.0, -1e9, -1e9])
    gated = apply_te2_quality_gate_array(
        score,
        positions=POSITIONS,
        vorp=VORP,
        roster_indices=[0],
        strategy=BASE_STRATEGY,
    )
    assert gated[1] == 84.0


def test_solid_te1_requires_clear_but_smaller_edge():
    score = np.array([0.0, 0.0, 77.0, 76.0, 75.0, 74.0, 70.0, -1e9, -1e9])
    gated = apply_te2_quality_gate_array(
        score,
        positions=POSITIONS,
        vorp=VORP,
        roster_indices=[1],
        strategy=BASE_STRATEGY,
    )
    assert gated[2] < -1e8

    score[2] = 79.0
    gated = apply_te2_quality_gate_array(
        score,
        positions=POSITIONS,
        vorp=VORP,
        roster_indices=[1],
        strategy=BASE_STRATEGY,
    )
    assert gated[2] == 79.0


def test_weak_te1_keeps_te2_normally_available():
    score = np.array([0.0, 0.0, 0.0, 72.0, 75.0, 74.0, 70.0, -1e9, -1e9])
    gated = apply_te2_quality_gate_array(
        score,
        positions=POSITIONS,
        vorp=VORP,
        roster_indices=[2],
        strategy=BASE_STRATEGY,
    )
    assert gated[3] == 72.0


def test_qb_cap_remains_hard_and_te3_is_blocked():
    positions = np.array(["QB", "RB", "WR", "TE", "K", "DEF"])
    pos_to_code = {p: i for i, p in enumerate(positions)}
    roster_counts = np.array([1, 2, 3, 2, 0, 0], dtype=np.int16)
    roster_cfg = {
        "QB": 1,
        "RB": 2,
        "WR": 3,
        "TE": 1,
        "FLEX": 1,
        "flex_eligible": ["RB", "WR", "TE"],
    }
    config = {
        "special_teams_strategy": {
            "draft": {
                "defense_round": 14,
                "kicker_round": 15,
                "rostered_defenses": 1,
                "rostered_kickers": 1,
            }
        }
    }
    score = apply_patbot_array_constraints(
        np.arange(6, dtype=float) + 10.0,
        positions=positions,
        roster_counts=roster_counts,
        pos_to_code=pos_to_code,
        roster_cfg=roster_cfg,
        round_no=10,
        config=config,
    )
    assert score[pos_to_code["QB"]] < -1e8
    assert score[pos_to_code["TE"]] < -1e8
