import numpy as np
import pandas as pd

from conditional_103_audit import CANDIDATE_POOL, CORE_TOP, SCENARIOS
from patbot.fantasypros_projection import _stats_dict, fantasypros_stats_to_patbot
from patbot.scoring import score_season_projection
from projection_source_audit import _source_vorp, _variant_players, _spearman_without_scipy


SCORING = {
    "pass_completion": 0.25,
    "pass_yards_per_point": 25,
    "pass_td": 4,
    "interception": -2,
    "rush_yards_per_point": 10,
    "rush_td": 6,
    "reception": 1.0,
    "rec_yards_per_point": 10,
    "rec_td": 6,
    "two_point_conversion": 2,
    "fumble_lost": -2,
    "return_td": 6,
    "offensive_fumble_return_td": 6,
    "pass_yard_bonuses": [],
    "rush_yard_bonuses": [],
    "rec_yard_bonuses": [],
}


def test_fantasypros_mapper_handles_projection_field_names():
    raw = {
        "games": 17,
        "pass_cmp": 350,
        "pass_yds": 4200,
        "pass_tds": 31,
        "pass_ints": 9,
        "rush_yds": 410,
        "rush_tds": 5,
        "rec_rec": 72,
        "rec_yds": 950,
        "rec_tds": 7,
        "fumbles": 2.5,
    }
    mapped = fantasypros_stats_to_patbot(raw)
    assert mapped["pass_yd"] == 4200
    assert mapped["pass_td"] == 31
    assert mapped["rush_yd"] == 410
    assert mapped["rec"] == 72
    assert mapped["rec_yd"] == 950
    assert mapped["fum_lost"] == 2.5
    assert mapped["gp"] == 17


def test_fantasypros_two_point_aggregate_is_counted_once():
    mapped = fantasypros_stats_to_patbot({"2pt_tds": 3})
    assert mapped["rush_2pt"] == 3
    assert mapped["pass_2pt"] == 0
    assert mapped["rec_2pt"] == 0
    scored = score_season_projection(mapped, SCORING, {}, "RB")
    assert scored["custom_points"] == 6.0


def test_explicit_fumbles_lost_takes_precedence_over_fumbles():
    mapped = fantasypros_stats_to_patbot({"fumbles_lost": 1.5, "fumbles": 4.0})
    assert mapped["fum_lost"] == 1.5


def test_fantasypros_stats_payload_can_be_a_list():
    assert _stats_dict({"stats": [{"rush_yds": 1000}]}) == {"rush_yds": 1000}


def test_source_vorp_uses_source_specific_replacement_level():
    players = pd.DataFrame(
        {
            "pos": ["RB", "RB", "RB", "WR"],
            "pts": [300.0, 250.0, 200.0, 280.0],
        }
    )
    cfg = {"draft_engine": {"replacement_rank": {"RB": 2, "WR": 1}}}
    vorp = _source_vorp(players, "pts", cfg)
    assert vorp.iloc[0] == 50.0
    assert vorp.iloc[1] == 0.0
    assert vorp.iloc[3] == 0.0


def test_equal_blend_averages_available_projection_sources():
    players = pd.DataFrame(
        {
            "name": ["A", "B"],
            "pos": ["RB", "WR"],
            "proj_points": [300.0, 200.0],
            "fantasypros_proj_points": [330.0, np.nan],
            "athletic_points": [270.0, 220.0],
            "generic_expert_rank": [1.0, 2.0],
            "expert_rank": [1.0, 2.0],
        }
    )
    out, coverage = _variant_players(players, "Equal Blend")
    assert out.loc[0, "proj_points"] == 300.0
    assert out.loc[1, "proj_points"] == 210.0
    assert coverage == 1.0


def test_spearman_helper_does_not_require_scipy():
    left = pd.Series([1.0, 2.0, 3.0, 4.0])
    right = pd.Series([40.0, 30.0, 20.0, 10.0])
    assert _spearman_without_scipy(left, right) == -1.0


def test_conditional_103_scenarios_are_generic_and_ordered():
    assert ("Jahmyr Gibbs", "Bijan Robinson") in SCENARIOS
    assert ("Bijan Robinson", "Jahmyr Gibbs") in SCENARIOS
    assert ("Jaxon Smith-Njigba", "Amon-Ra St. Brown") in SCENARIOS
    assert ("Amon-Ra St. Brown", "Jaxon Smith-Njigba") in SCENARIOS
    assert len(SCENARIOS) == len(set(SCENARIOS))
    assert all(a != b for a, b in SCENARIOS)
    assert set(CORE_TOP).issubset(CANDIDATE_POOL)
