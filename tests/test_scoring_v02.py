from patbot.scoring import score_season_projection, expected_game_threshold_bonuses

SCORING = {
    "pass_completion": 0.25,
    "pass_yards_per_point": 25,
    "pass_td": 4,
    "interception": -2,
    "pass_yard_bonuses": [
        {"threshold": 320, "points": 1},
        {"threshold": 350, "points": 3},
        {"threshold": 380, "points": 1},
    ],
    "rush_yards_per_point": 10,
    "rush_td": 6,
    "rush_yard_bonuses": [
        {"threshold": 130, "points": 1},
        {"threshold": 155, "points": 3},
        {"threshold": 180, "points": 1},
    ],
    "reception": 1,
    "rec_yards_per_point": 10,
    "rec_td": 6,
    "rec_yard_bonuses": [
        {"threshold": 140, "points": 1},
        {"threshold": 165, "points": 3},
        {"threshold": 190, "points": 1},
    ],
    "return_td": 6,
    "two_point_conversion": 2,
    "fumble_lost": -2,
    "offensive_fumble_return_td": 6,
}

BONUS_MODEL = {
    "pass_yards": {"sd_floor": 55, "sd_pct_of_mean": 0.28},
    "rush_yards": {"sd_floor": 25, "sd_pct_of_mean": 0.55},
    "rec_yards": {"sd_floor": 30, "sd_pct_of_mean": 0.65},
}

def test_completion_scoring_is_included():
    stats = {"gp": 17, "pass_cmp": 400, "pass_yd": 4000, "pass_td": 30, "pass_int": 10}
    result = score_season_projection(stats, SCORING, BONUS_MODEL, "QB")
    assert result["base_points"] == 360.0
    assert result["custom_points"] >= 360.0

def test_receiving_bonus_is_per_game_estimate_not_one_season_bonus():
    bonuses = expected_game_threshold_bonuses(
        season_yards=1700,
        games=17,
        bonuses=SCORING["rec_yard_bonuses"],
        sd_floor=30,
        sd_pct_of_mean=0.65,
    )
    assert bonuses > 0
    assert bonuses < 85

def test_two_point_and_return_td():
    stats = {"gp": 17, "rec": 1, "rec_yd": 10, "rec_td": 0, "rec_2pt": 1, "st_td": 1}
    result = score_season_projection(stats, SCORING, BONUS_MODEL, "WR")
    assert result["base_points"] == 10.0

def test_kicker_uses_provider_until_yahoo_import():
    result = score_season_projection({"pts_ppr": 130.5}, SCORING, BONUS_MODEL, "K")
    assert result["custom_points"] == 130.5
    assert result["bonus_method"] == "provider_until_yahoo_import"
